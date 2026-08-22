import Foundation
import UIKit
import llama

// MtmdContext — vision inference engine wrapping the llama.cpp + MTMD C API.
// Mirrors bench_vlm.cpp so the per-stage timings are directly comparable:
//   encode_ms  : vision encoder forward (mtmd batch encode), alone
//   prefill_ms : image tokens + text through the decoder, before first token
//   decode_ms  : autoregressive generation
//   ttft_ms    : wall time from request start to first generated token
enum MtmdError: Error {
    case modelLoadFailed(String)
    case mmprojLoadFailed(String)
    case imageLoadFailed(String)
    case contextInitFailed
    case tokenizeFailed(Int32)
    case decodeFailed(Int32)
}

struct BenchResult {
    var image: String
    var prompt: String
    var encodeMs: Int64 = 0
    var prefillMs: Int64 = 0
    var nPrefillToks: Int64 = 0
    var decodeMs: Int64 = 0
    var nDecodeToks: Int64 = 0
    var ttftMs: Int64 = 0
    var output: String = ""
    var peakFootprintMB: Int64 = 0
    var thermalState: String = ""
    var batteryLevel: Float = -1

    func toJSON() -> String {
        let esc = output
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
        let e = prompt
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        return """
        {"image":"\(image)","prompt":"\(e)","encode_ms":\(encodeMs),"prefill_ms":\(prefillMs),\
        "n_prefill_toks":\(nPrefillToks),"decode_ms":\(decodeMs),"n_decode_toks":\(nDecodeToks),\
        "ttft_ms":\(ttftMs),"peak_rss_mb":\(peakFootprintMB),"thermal":"\(thermalState)",\
        "battery":\(batteryLevel),"output":"\(esc)"}
        """
    }
}

actor MtmdContext {
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vctx: OpaquePointer?
    private var vocab: OpaquePointer?
    private var sampling: UnsafeMutablePointer<llama_sampler>?
    private var peakFootprint: Int64 = 0

    init() {}

    var isLoaded: Bool { model != nil && vctx != nil }

    deinit {
        if let s = sampling { llama_sampler_free(s) }
        if let v = vctx { mtmd_free(v) }
        if let c = context { llama_free(c) }
        if let m = model { llama_model_free(m) }
        llama_backend_free()
    }

    // ---- load --------------------------------------------------------------
    func load(modelPath: String, mmprojPath: String) throws {
        llama_backend_init()
        var mp = llama_model_default_params()
        mp.n_gpu_layers = 99
        guard let m = llama_model_load_from_file(modelPath, mp) else {
            throw MtmdError.modelLoadFailed(modelPath)
        }
        model = m
        vocab = llama_model_get_vocab(m)

        var vp = mtmd_context_params_default()
        vp.use_gpu = true
        vp.print_timings = false
        vp.warmup = false
        guard let v = mtmd_init_from_file(mmprojPath, m, vp) else {
            throw MtmdError.mmprojLoadFailed(mmprojPath)
        }
        vctx = v

        var cp = llama_context_default_params()
        cp.n_ctx = 8192
        cp.n_batch = 2048
        cp.no_perf = true
        guard let c = llama_init_from_model(m, cp) else {
            throw MtmdError.contextInitFailed
        }
        context = c

        let smpl = llama_sampler_chain_init(llama_sampler_chain_default_params())
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy())
        sampling = smpl
    }

    // ---- footprint ----------------------------------------------------------
    private func sampleFootprint() {
        var info = task_vm_info()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { pp in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), pp, &count)
            }
        }
        if kr == KERN_SUCCESS {
            let mb = Int64(info.phys_footprint) / (1024 * 1024)
            peakFootprint = max(peakFootprint, mb)
        }
    }

    // ---- one inference --------------------------------------------------------
    func runOnce(imagePath: String, prompt: String, nPredict: Int32) throws -> BenchResult {
        guard let m = model, let c = context, let v = vctx, let smpl = sampling else {
            throw MtmdError.modelLoadFailed("not loaded")
        }
        let tStart = DispatchTime.now().uptimeNanoseconds

        // image -> bitmap
        let bw = mtmd_helper_bitmap_init_from_file(v, imagePath, false)
        guard let bitmap = bw.bitmap else {
            throw MtmdError.imageLoadFailed(imagePath)
        }
        defer { mtmd_bitmap_free(bitmap) }

        // chat format
        var finalPrompt = prompt
        if !finalPrompt.contains(String(cString: mtmd_default_marker())) {
            finalPrompt = String(cString: mtmd_default_marker()) + finalPrompt
        }
        var msg = llama_chat_message(role: "user", content: finalPrompt)
        var formatted = ""
        withUnsafeMutablePointer(to: &msg) { mptr in
            let len = llama_chat_apply_template(nil, mptr, 1, true, nil, 0)
            var buf = [CChar](repeating: 0, count: Int(len) + 8)
            llama_chat_apply_template(nil, mptr, 1, true, &buf, Int32(buf.count))
            formatted = String(cString: buf)
        }

        // tokenize into chunks
        let chunks = mtmd_input_chunks_init()
        defer { mtmd_input_chunks_free(chunks) }
        var text = mtmd_input_text()
        try formatted.withCString { fptr in
            text.text = fptr
            text.text_len = formatted.utf8.count
            text.add_special = true
            text.parse_special = true
            var bm: [OpaquePointer?] = [bitmap]
            do {
                let res = try mtmd_tokenize(v, chunks, &text, &bm, 1)
                if res != 0 { throw MtmdError.tokenizeFailed(res) }
            } catch {
                throw MtmdError.tokenizeFailed(-1)
            }
        }

        var encodeSum: Int64 = 0
        var prefillSum: Int64 = 0
        var nPast: llama_pos = 0
        let nChunks = mtmd_input_chunks_size(chunks)

        for i in 0..<nChunks {
            let chunk = mtmd_input_chunks_get(chunks, i)
            let ctype = mtmd_input_chunk_get_type(chunk)
            if ctype == MTMD_INPUT_CHUNK_TYPE_TEXT {
                let t0 = DispatchTime.now().uptimeNanoseconds
                var newN = nPast
                let r = mtmd_helper_eval_chunk_single(v, c, chunk, nPast, 0, 2048, i == nChunks - 1, &newN)
                if r != 0 { throw MtmdError.decodeFailed(r) }
                nPast = newN
                prefillSum += Int64((DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000)
            } else {
                // encode image chunk (vision encoder pass)
                let t0 = DispatchTime.now().uptimeNanoseconds
                let r = mtmd_encode_chunk(v, chunk)
                if r != 0 { throw MtmdError.decodeFailed(r) }
                encodeSum += Int64((DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000)
                let embd = mtmd_get_output_embd(v)
                var newN = nPast
                let r2 = mtmd_helper_decode_image_chunk(v, c, chunk, embd, nPast, 0, 2048, &newN, nil, nil)
                if r2 != 0 { throw MtmdError.decodeFailed(r2) }
                nPast = newN
            }
        }

        // decode loop
        let tDecStart = DispatchTime.now().uptimeNanoseconds
        var nDec: Int32 = 0
        var output = ""
        var tttfNanos: UInt64 = 0

        var tok: llama_token = llama_sampler_sample(smpl, c, -1)
        tttfNanos = DispatchTime.now().uptimeNanoseconds
        llama_sampler_accept(smpl, tok)

        var piece = [CChar](repeating: 0, count: 64)
        while nDec < nPredict {
            if llama_vocab_is_eog(vocab, tok) { break }
            var nP = llama_token_to_piece(vocab, tok, &piece, Int32(piece.count), 0, false)
            if nP < 0 {
                piece = [CChar](repeating: 0, count: Int(-nP))
                nP = llama_token_to_piece(vocab, tok, &piece, Int32(piece.count), 0, false)
            }
            output += String(cString: piece)
            nDec += 1

            var one: llama_token = tok
            var b = llama_batch_get_one(&one, 1)
            let rc = llama_decode(c, b)
            if rc != 0 { throw MtmdError.decodeFailed(rc) }
            tok = llama_sampler_sample(smpl, c, -1)
            llama_sampler_accept(smpl, tok)
            sampleFootprint()
        }
        let tEnd = DispatchTime.now().uptimeNanoseconds

        var res = BenchResult(
            image: imagePath, prompt: prompt,
            encodeMs: encodeSum,
            prefillMs: prefillSum,
            nPrefillToks: Int64(nPast),
            decodeMs: Int64((tEnd - tDecStart) / 1_000_000),
            nDecodeToks: Int64(nDec),
            ttftMs: Int64((tttfNanos - tStart) / 1_000_000),
            output: output,
            peakFootprintMB: peakFootprint,
            thermalState: thermalStateName(),
            batteryLevel: UIDevice.current.batteryLevel
        )
        return res
    }

    private func thermalStateName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}