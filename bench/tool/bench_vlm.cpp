// bench_vlm.cpp — per-stage timing harness for llama.cpp VLM inference.
//
// Measures, as FOUR separate numbers:
//   encode_ms   : vision encoder forward pass (MTMD batch encode), alone
//   prefill_ms  : image-embedding tokens + text prompt through the LLM decoder
//                 (everything before the first generated token, minus the encoder)
//   decode_ms   : autoregressive generation loop
//   ttft_ms     : wall time from request start to first generated token
//
// Plus: peak RSS (macOS/Linux), model size, token counts, and the generated text.
//
// Built against the same llama.cpp release as the iOS app (b10549), so the same
// code compiles for the iPhone harness (via llama.xcframework + libmtmd).
//
// Usage:
//   bench_vlm -m <model.gguf> -mm <mmproj.gguf> --image <img> -p "<prompt>" \
//             [-n <max_tokens>] [--no-mmproj-offload] [--mmproj-offload] \
//             [--no-image] [--no-chat] [--json] [--seed N]
//
// --no-image  : run the prompt without any image (text-only reference)
// --no-chat   : feed the raw prompt instead of the chat template
// --no-mmproj-offload : keep the vision encoder on CPU (for GPU-vs-CPU compare)

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

#if defined(__APPLE__) || defined(__unix__)
#include <sys/resource.h>
#include <sys/stat.h>
#endif

static int64_t now_us() {
    return ggml_time_us();
}

static std::string vec_to_string(const char * s, int32_t n) {
    return std::string(s ? s : "", n > 0 ? n : 0);
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out;
}

static void print_json(const char * key, const std::string & val, bool last = false) {
    printf("  \"%s\": \"%s\"%s\n", key, json_escape(val).c_str(), last ? "" : ",");
}

static void print_json(const char * key, int64_t val, bool last = false) {
    printf("  \"%s\": %lld%s\n", key, (long long) val, last ? "" : ",");
}

static void print_json(const char * key, double val, bool last = false) {
    printf("  \"%s\": %.4f%s\n", key, val, last ? "" : ",");
}

struct bench_result {
    std::string model;
    std::string mmproj;
    std::string image;
    std::string prompt;
    bool        chat;
    bool        mmproj_offload;

    int64_t  load_ms        = 0;
    int64_t  encode_ms      = 0;
    int64_t  prefill_ms     = 0;
    int64_t  decode_ms      = 0;
    int64_t  ttft_ms        = 0;
    int64_t  n_prefill_toks = 0;
    int64_t  n_decode_toks  = 0;
    int64_t  peak_rss_mb    = 0;
    int64_t  model_size_mb  = 0;
    int64_t  mmproj_size_mb = 0;
    std::string output;
};

static bench_result g_res;

int main(int argc, char ** argv) {
    std::string model_path;
    std::string mmproj_path;
    std::string image_path;
    std::string prompt;
    int n_predict   = 64;
    bool use_image  = true;
    bool use_chat   = true;
    bool mmproj_gpu = true;
    bool json_out   = false;
    uint32_t seed   = 0;
    int n_ctx       = 8192;
    int n_batch     = 2048;
    int n_threads   = 0; // 0 = default

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")          model_path = need("-m");
        else if (a == "-mm")         mmproj_path = need("-mm");
        else if (a == "--image")     image_path = need("--image");
        else if (a == "-p")          prompt = need("-p");
        else if (a == "-n")          n_predict = atoi(need("-n"));
        else if (a == "--no-image")  use_image = false;
        else if (a == "--no-chat")   use_chat = false;
        else if (a == "--no-mmproj-offload") mmproj_gpu = false;
        else if (a == "--mmproj-offload")    mmproj_gpu = true;
        else if (a == "--json")      json_out = true;
        else if (a == "--seed")      seed = (uint32_t) atoi(need("--seed"));
        else if (a == "--n-ctx")     n_ctx = atoi(need("--n-ctx"));
        else if (a == "--n-batch")   n_batch = atoi(need("--n-batch"));
        else if (a == "--threads")   n_threads = atoi(need("--threads"));
        else if (a == "-h" || a == "--help") {
            fprintf(stderr, "usage: bench_vlm -m model.gguf -mm mmproj.gguf [--image img] -p prompt [-n toks]\n");
            return 1;
        }
        else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 1;
        }
    }

    if (model_path.empty() || prompt.empty()) {
        fprintf(stderr, "error: -m and -p are required\n");
        return 1;
    }
    if (use_image && image_path.empty()) {
        fprintf(stderr, "error: --image required unless --no-image\n");
        return 1;
    }

    g_res.model  = model_path;
    g_res.mmproj = mmproj_path;
    g_res.image  = image_path;
    g_res.prompt = prompt;
    g_res.chat   = use_chat;
    g_res.mmproj_offload = mmproj_gpu;

    llama_backend_init();

    // ---- load main model -------------------------------------------------
    {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 99; // offload all to GPU (Metal on Mac/iOS)

        int64_t t0 = now_us();
        llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model) { fprintf(stderr, "error: failed to load model %s\n", model_path.c_str()); return 1; }
        g_res.load_ms = (now_us() - t0) / 1000;
        g_res.model_size_mb = (int64_t) llama_model_size(model) / (1024*1024);
        g_res.mmproj_size_mb = 0;

        // ---- load mmproj (vision encoder) --------------------------------
        mtmd_context * vctx = nullptr;
        if (!mmproj_path.empty()) {
            struct stat st;
            if (stat(mmproj_path.c_str(), &st) == 0) {
                g_res.mmproj_size_mb = st.st_size / (1024*1024);
            }
            mtmd_context_params vp = mtmd_context_params_default();
            vp.use_gpu       = mmproj_gpu;
            vp.print_timings = false;
            vp.warmup        = false;
            t0 = now_us();
            vctx = mtmd_init_from_file(mmproj_path.c_str(), model, vp);
            if (!vctx) { fprintf(stderr, "error: failed to load mmproj %s\n", mmproj_path.c_str()); return 1; }
            g_res.load_ms += (now_us() - t0) / 1000;
        }

        // ---- load image ----------------------------------------------------
        mtmd_helper_bitmap_wrapper bw{};
        if (use_image) {
            bw = mtmd_helper_bitmap_init_from_file(vctx, image_path.c_str(), /*placeholder=*/false);
            if (!bw.bitmap) {
                fprintf(stderr, "error: failed to load image %s\n", image_path.c_str());
                return 1;
            }
        }

        // ---- format prompt ---------------------------------------------------
        std::string final_prompt = prompt;
        if (use_image && use_chat) {
            // most models require the image marker in the user message
            if (final_prompt.find(mtmd_default_marker()) == std::string::npos) {
                final_prompt = std::string(mtmd_default_marker()) + final_prompt;
            }
        }

        std::string formatted;
        if (use_chat) {
            llama_chat_message msgs[1];
            msgs[0].role    = "user";
            msgs[0].content = final_prompt.c_str();
            // size the buffer
            int32_t len = llama_chat_apply_template(nullptr, msgs, 1, true, nullptr, 0);
            if (len < 0) { fprintf(stderr, "error: chat template failed\n"); return 1; }
            std::vector<char> buf(len + 8, 0);
            llama_chat_apply_template(nullptr, msgs, 1, true, buf.data(), (int32_t) buf.size());
            formatted = std::string(buf.data());
        } else {
            formatted = final_prompt;
        }

        // ---- tokenize into mtmd chunks (image path only) --------------------
        mtmd_input_text text;
        text.text          = formatted.data();
        text.text_len      = formatted.size();
        text.add_special   = true;
        text.parse_special = true;

        // ---- LLM context ------------------------------------------------------
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx     = n_ctx;
        cp.n_batch   = n_batch;
        cp.no_perf   = true;
        if (n_threads > 0) cp.n_threads = n_threads;
        llama_context * lctx = llama_init_from_model(model, cp);
        if (!lctx) { fprintf(stderr, "error: failed to init context\n"); return 1; }
        llama_set_n_threads(lctx, n_threads > 0 ? n_threads : -1, n_threads > 0 ? n_threads : -1);

        // ---- sampler (deterministic) -------------------------------------------
        llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        // ---- encode + prefill ---------------------------------------------------
        int64_t t_start = now_us();
        int64_t encode_sum = 0, prefill_sum = 0;
        llama_pos n_past = 0;
        const llama_vocab * vocab = llama_model_get_vocab(model);

        if (!use_image) {
            // plain text-only path: tokenize and decode in one shot
            std::vector<llama_token> toks(1 + formatted.size() + 2);
            int32_t n_toks = llama_tokenize(vocab, formatted.data(), (int32_t) formatted.size(),
                                            toks.data(), (int32_t) toks.size(), true, true);
            if (n_toks < 0) {
                toks.resize(-n_toks);
                n_toks = llama_tokenize(vocab, formatted.data(), (int32_t) formatted.size(),
                                        toks.data(), (int32_t) toks.size(), true, true);
            }
            toks.resize(n_toks);

            int64_t t0c = now_us();
            for (int32_t i = 0; i < n_toks; i += n_batch) {
                int32_t n_cur = std::min(n_batch, n_toks - i);
                llama_batch batch = llama_batch_get_one(toks.data() + i, n_cur);
                int rc = llama_decode(lctx, batch);
                if (rc != 0) { fprintf(stderr, "error: prefill decode failed (%d)\n", rc); return 1; }
                n_past += n_cur;
            }
            prefill_sum += now_us() - t0c;
            g_res.encode_ms   = 0;
            g_res.prefill_ms  = prefill_sum / 1000;
            g_res.n_prefill_toks = n_past;
        } else {
            // image path: tokenize with mtmd (replaces image marker with image tokens)
            mtmd::input_chunks chunks(mtmd_input_chunks_init());
            std::vector<const mtmd_bitmap *> bitmaps;
            bitmaps.push_back(bw.bitmap);

            int32_t res = mtmd_tokenize(vctx, chunks.ptr.get(), &text,
                                        bitmaps.data(), bitmaps.size());
            if (res != 0) { fprintf(stderr, "error: mtmd_tokenize failed (%d)\n", res); return 1; }
            size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());

            mtmd_batch * mbatch = nullptr;
            for (size_t i = 0; i < n_chunks; i++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
            enum mtmd_input_chunk_type ctype = mtmd_input_chunk_get_type(chunk);

            if (ctype == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                int64_t t0c = now_us();
                llama_pos new_n_past = n_past;
                res = mtmd_helper_eval_chunk_single(vctx, lctx, chunk, n_past, 0, n_batch,
                                                    i == n_chunks - 1, &new_n_past);
                if (res != 0) { fprintf(stderr, "error: text chunk eval failed (%d)\n", res); return 1; }
                n_past = new_n_past;
                prefill_sum += now_us() - t0c;
            } else {
                // media chunk: batch-encode as many media chunks as possible
                if (!mbatch) mbatch = mtmd_batch_init(vctx);
                res = mtmd_batch_add_chunk(mbatch, chunk);
                if (res != 0) {
                    // batch full or incompatible -> encode it, then start a new one
                    int64_t t0e = now_us();
                    res = mtmd_batch_encode(mbatch);
                    if (res != 0) { fprintf(stderr, "error: batch encode failed (%d)\n", res); return 1; }
                    encode_sum += now_us() - t0e;

                    float * embd = mtmd_batch_get_output_embd(mbatch, chunk);
                    llama_pos new_n_past = n_past;
                    res = mtmd_helper_decode_image_chunk(vctx, lctx, chunk, embd, n_past, 0, n_batch,
                                                         &new_n_past, nullptr, nullptr);
                    if (res != 0) { fprintf(stderr, "error: image chunk decode failed (%d)\n", res); return 1; }
                    n_past = new_n_past;

                    mtmd_batch_free(mbatch);
                    mbatch = mtmd_batch_init(vctx);
                    res = mtmd_batch_add_chunk(mbatch, chunk);
                    if (res != 0) { fprintf(stderr, "error: can't add chunk to new batch\n"); return 1; }
                } else {
                    // try to add more media chunks
                    size_t j;
                    for (j = i + 1; j < n_chunks; j++) {
                        const mtmd_input_chunk * nxt = mtmd_input_chunks_get(chunks.ptr.get(), j);
                        if (mtmd_input_chunk_get_type(nxt) == MTMD_INPUT_CHUNK_TYPE_TEXT) break;
                        if (mtmd_batch_add_chunk(mbatch, nxt) != 0) break;
                    }
                    int64_t t0e = now_us();
                    res = mtmd_batch_encode(mbatch);
                    if (res != 0) { fprintf(stderr, "error: batch encode failed (%d)\n", res); return 1; }
                    encode_sum += now_us() - t0e;

                    // decode each media chunk in this batch
                    for (size_t k = i; k < j; k++) {
                        const mtmd_input_chunk * ck = mtmd_input_chunks_get(chunks.ptr.get(), k);
                        if (mtmd_input_chunk_get_type(ck) == MTMD_INPUT_CHUNK_TYPE_TEXT) break;
                        float * embd = mtmd_batch_get_output_embd(mbatch, ck);
                        if (!embd) { fprintf(stderr, "error: no embd for chunk %zu\n", k); return 1; }
                        int64_t t0d = now_us();
                        llama_pos new_n_past = n_past;
                        res = mtmd_helper_decode_image_chunk(vctx, lctx, ck, embd, n_past, 0, n_batch,
                                                             &new_n_past, nullptr, nullptr);
                        if (res != 0) { fprintf(stderr, "error: image chunk decode failed (%d)\n", res); return 1; }
                        n_past = new_n_past;
                        prefill_sum += now_us() - t0d;
                    }
                     i = j - 1; // advance past the processed media chunks
                 }
             }
         }
         if (mbatch) mtmd_batch_free(mbatch);

         g_res.encode_ms   = encode_sum / 1000;
         g_res.prefill_ms  = prefill_sum / 1000;
         g_res.n_prefill_toks = n_past;
        } // end else (image path)

        // ---- decode loop --------------------------------------------------------
        int64_t t_dec_start = now_us();
        int64_t n_dec = 0;
        int64_t t_ttft_us = -1;
        std::string output;
        std::vector<char> piece(64);

        // first token: sampled from the prefill logits (marks TTFT)
        llama_token tok = llama_sampler_sample(smpl, lctx, -1);
        t_ttft_us = now_us();
        llama_sampler_accept(smpl, tok);

        while (n_dec < (int64_t) n_predict) {
            if (llama_vocab_is_eog(vocab, tok)) break;

            int32_t n_p = llama_token_to_piece(vocab, tok, piece.data(), (int32_t) piece.size(), 0, false);
            if (n_p < 0) {
                piece.resize(-n_p);
                n_p = llama_token_to_piece(vocab, tok, piece.data(), (int32_t) piece.size(), 0, false);
            }
            output += vec_to_string(piece.data(), n_p);
            n_dec++;

            llama_batch batch = llama_batch_get_one(&tok, 1);
            int res_dec = llama_decode(lctx, batch);
            if (res_dec != 0) { fprintf(stderr, "error: decode failed (%d)\n", res_dec); break; }

            tok = llama_sampler_sample(smpl, lctx, -1);
            llama_sampler_accept(smpl, tok);
        }

        int64_t t_end = now_us();
        g_res.decode_ms  = (t_end - t_dec_start) / 1000;
        g_res.n_decode_toks = n_dec;
        g_res.ttft_ms    = t_ttft_us >= 0 ? (t_ttft_us - t_start) / 1000 : -1;
        g_res.output     = output;

        // ---- peak RSS ---------------------------------------------------------
#if defined(__APPLE__) || defined(__unix__)
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
            // macOS: bytes; Linux: KB
            g_res.peak_rss_mb = (ru.ru_maxrss + 512*1024) / (1024*1024);
        }
#endif

        // ---- output ------------------------------------------------------------
        if (json_out) {
            printf("{\n");
            print_json("model",  g_res.model);
            print_json("mmproj", g_res.mmproj);
            print_json("image",  g_res.image);
            print_json("prompt", g_res.prompt);
            print_json("chat",   g_res.chat ? "1" : "0");
            print_json("mmproj_offload", g_res.mmproj_offload ? "1" : "0");
            print_json("load_ms",       g_res.load_ms);
            print_json("encode_ms",     g_res.encode_ms);
            print_json("prefill_ms",    g_res.prefill_ms);
            print_json("n_prefill_toks",g_res.n_prefill_toks);
            print_json("decode_ms",     g_res.decode_ms);
            print_json("n_decode_toks", g_res.n_decode_toks);
            print_json("ttft_ms",       g_res.ttft_ms);
            print_json("peak_rss_mb",   g_res.peak_rss_mb);
            print_json("model_size_mb", g_res.model_size_mb);
            print_json("mmproj_size_mb",g_res.mmproj_size_mb);
            print_json("output",        g_res.output, true);
            printf("}\n");
        } else {
            double prefill_tok_s = g_res.prefill_ms > 0 ? 1000.0 * g_res.n_prefill_toks / g_res.prefill_ms : 0;
            double decode_tok_s  = g_res.decode_ms  > 0 ? 1000.0 * g_res.n_decode_toks / g_res.decode_ms  : 0;
            printf("== bench_vlm ==\n");
            printf("model       : %s\n", g_res.model.c_str());
            printf("mmproj      : %s\n", g_res.mmproj.c_str());
            printf("image       : %s\n", g_res.image.c_str());
            printf("load        : %lld ms\n", (long long) g_res.load_ms);
            printf("encode      : %lld ms  (vision encoder)\n", (long long) g_res.encode_ms);
            printf("prefill     : %lld ms  (%lld tokens, %.1f tok/s)\n",
                   (long long) g_res.prefill_ms, (long long) g_res.n_prefill_toks, prefill_tok_s);
            printf("decode      : %lld ms  (%lld tokens, %.1f tok/s)\n",
                   (long long) g_res.decode_ms, (long long) g_res.n_decode_toks, decode_tok_s);
            printf("TTFT        : %lld ms\n", (long long) g_res.ttft_ms);
            printf("peak RSS    : %lld MB\n", (long long) g_res.peak_rss_mb);
            printf("output      : %s\n", g_res.output.c_str());
        }

        llama_sampler_free(smpl);
        llama_free(lctx);
        mtmd_free(vctx);
        llama_model_free(model);
    }

    llama_backend_free();
    return 0;
}
