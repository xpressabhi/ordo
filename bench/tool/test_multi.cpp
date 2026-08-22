// test_multi.cpp — reproduce the app's "clear memory between images" behavior.
// Usage: test_multi -m model.gguf -mm mmproj.gguf --img1 a.jpg --img2 b.jpg -p "prompt"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int64_t now_us() { return ggml_time_us(); }

static int run_inference(llama_model * model, mtmd_context * vctx, llama_context * lctx,
                         llama_sampler * smpl, const std::string & image, const std::string & prompt,
                         const llama_vocab * vocab, int n_predict, int n_batch, const char * label) {
    llama_memory_clear(llama_get_memory(lctx), true);

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    std::string final_prompt = std::string(mtmd_default_marker()) + prompt;
    llama_chat_message msgs[1];
    msgs[0].role = "user";
    msgs[0].content = final_prompt.c_str();
    int32_t len = llama_chat_apply_template(nullptr, msgs, 1, true, nullptr, 0);
    std::vector<char> buf(len + 8, 0);
    llama_chat_apply_template(nullptr, msgs, 1, true, buf.data(), (int32_t) buf.size());
    std::string formatted(buf.data());
    fprintf(stderr, "%s: FORMATTED=[%s]\n", label, formatted.c_str());

    auto bw = mtmd_helper_bitmap_init_from_file(vctx, image.c_str(), false);
    if (!bw.bitmap) { fprintf(stderr, "%s: bitmap load failed\n", label); return -1; }
    mtmd_bitmap * bm = bw.bitmap;

    mtmd_input_text text;
    text.text = formatted.data();
    text.text_len = formatted.size();
    text.add_special = true;
    text.parse_special = true;
    std::vector<const mtmd_bitmap *> bitmaps = { bm };
    int32_t res = mtmd_tokenize(vctx, chunks.ptr.get(), &text, bitmaps.data(), 1);
    mtmd_bitmap_free(bm);
    if (res != 0) { fprintf(stderr, "%s: tokenize failed (%d)\n", label, res); return -1; }

    size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    llama_pos n_past = 0;
    mtmd_batch * mbatch = nullptr;
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            llama_pos new_n = n_past;
            res = mtmd_helper_eval_chunk_single(vctx, lctx, chunk, n_past, 0, n_batch, i == n_chunks - 1, &new_n);
            if (res != 0) { fprintf(stderr, "%s: text eval failed (%d)\n", label, res); return -1; }
            n_past = new_n;
        } else {
            res = mtmd_encode_chunk(vctx, chunk);
            if (res != 0) { fprintf(stderr, "%s: encode failed (%d)\n", label, res); return -1; }
            float * embd = mtmd_get_output_embd(vctx);
            {
                double sum = 0; int cnt = 0;
                for (int k = 0; k < 4096 && embd[k] == embd[k]; k++) { sum += fabs(embd[k]); cnt++; }
                fprintf(stderr, "%s: EMBED mean_abs=%.6f first16=[%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f]\n", label, sum/cnt,
                        embd[0],embd[1],embd[2],embd[3],embd[4],embd[5],embd[6],embd[7],
                        embd[8],embd[9],embd[10],embd[11],embd[12],embd[13],embd[14],embd[15]);
            }
            llama_pos new_n = n_past;
            res = mtmd_helper_decode_image_chunk(vctx, lctx, chunk, embd, n_past, 0, n_batch, &new_n, nullptr, nullptr);
            if (res != 0) { fprintf(stderr, "%s: image chunk decode failed (%d)\n", label, res); return -1; }
            n_past = new_n;
        }
    }

    int64_t t0 = now_us();
    llama_token tok = llama_sampler_sample(smpl, lctx, -1);
    llama_sampler_accept(smpl, tok);
    std::string out;
    std::vector<char> piece(64);
    int64_t n = 0;
    while (n < n_predict && !llama_vocab_is_eog(vocab, tok)) {
        int32_t nP = llama_token_to_piece(vocab, tok, piece.data(), (int32_t) piece.size(), 0, false);
        if (nP < 0) { piece.resize(-nP); nP = llama_token_to_piece(vocab, tok, piece.data(), (int32_t) piece.size(), 0, false); }
        if (nP > 0) out += std::string(piece.data(), nP);
        n++;
        llama_batch b = llama_batch_get_one(&tok, 1);
        int rc = llama_decode(lctx, b);
        if (rc != 0) { fprintf(stderr, "%s: decode loop failed (%d)\n", label, rc); return -1; }
        tok = llama_sampler_sample(smpl, lctx, -1);
        llama_sampler_accept(smpl, tok);
    }
    fprintf(stderr, "%s: OK %lld tok in %lld ms n_past=%lld: %.60s\n", label, (long long) n, (long long)((now_us()-t0)/1000), (long long) n_past, out.c_str());
    return 0;
}

int main(int argc, char ** argv) {
    std::string model_path, mmproj_path, img1, img2, prompt;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char * n) { return argv[++i]; };
        if (a == "-m") model_path = need("-m");
        else if (a == "-mm") mmproj_path = need("-mm");
        else if (a == "--img1") img1 = need("--img1");
        else if (a == "--img2") img2 = need("--img2");
        else if (a == "-p") prompt = need("-p");
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return 1;
    mtmd_context_params vp = mtmd_context_params_default();
    vp.use_gpu = true; vp.print_timings = false; vp.warmup = false;
    mtmd_context * vctx = mtmd_init_from_file(mmproj_path.c_str(), model, vp);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 4096; cp.no_perf = true;
    llama_context * lctx = llama_init_from_model(model, cp);
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int r1 = run_inference(model, vctx, lctx, smpl, img1, prompt, vocab, 24, 2048, "run1");
    int r2 = run_inference(model, vctx, lctx, smpl, img2, prompt, vocab, 24, 2048, "run2");
    fprintf(stderr, "run1=%d run2=%d\n", r1, r2);
    llama_sampler_free(smpl);
    llama_free(lctx);
    mtmd_free(vctx);
    llama_model_free(model);
    llama_backend_free();
    return (r1 == 0 && r2 == 0) ? 0 : 1;
}