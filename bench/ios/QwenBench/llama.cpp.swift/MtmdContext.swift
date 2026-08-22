import Foundation
import UIKit
import llama

// capture mtmd/llama log lines so failures can be shown in the UI
private let logLock = NSLock()
private var gMtmdLogs = ""
private let mtmdLogSink: ggml_log_callback = { _, text, _ in
    guard let text else { return }
    let s = String(cString: text)
    logLock.lock()
    gMtmdLogs = String((gMtmdLogs + s).suffix(4000))
    logLock.unlock()
}

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
    var debug: String = ""

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
    private(set) var lastDebug: String = ""
    // pre-computed vision embeddings, keyed by image path
    private var cachedImageEmbds: [String: [[Float]]] = [:]
    private var cachedImageNTokens: [String: [Int]] = [:]
    private var nEmbdInp: Int32 = 2048

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
        nEmbdInp = llama_model_n_embd_inp(m)

        var vp = mtmd_context_params_default()
        vp.use_gpu = true
        vp.print_timings = false
        vp.warmup = false
        mtmd_log_set(mtmdLogSink, nil)
        guard let v = mtmd_init_from_file(mmprojPath, m, vp) else {
            throw MtmdError.mmprojLoadFailed(mmprojPath)
        }
        vctx = v

        var cp = llama_context_default_params()
        cp.n_ctx = 8192
        cp.n_batch = 2048
        // FA back to default (AUTO): the only config that saw the image on-device
        cp.no_perf = true
        guard let c = llama_init_from_model(m, cp) else {
            throw MtmdError.contextInitFailed
        }
        context = c

        let smpl = llama_sampler_chain_init(llama_sampler_chain_default_params())
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy())
        sampling = smpl
    }

    // ---- pre-compute vision for a photo (while the user types) ---------------
    // Encodes all image chunks of a minimal prompt and caches the embeddings so
    // runOnce can skip the vision tower when the same image is asked about.
    func prepareImage(imagePath: String) async throws -> (encodeMs: Int64, nChunks: Int) {
        guard let m = model, let v = vctx else { throw MtmdError.modelLoadFailed("not loaded") }
        let tStart = DispatchTime.now().uptimeNanoseconds
        let marker = String(cString: mtmd_default_marker())
        // minimal prompt: same structure runOnce will produce, empty question
        let finalPrompt = marker
        let contentPtr = strdup(finalPrompt)
        let rolePtr = strdup("user")
        defer { free(contentPtr); free(rolePtr) }
        var msg = llama_chat_message(role: rolePtr, content: contentPtr)
        var formatted = ""
        withUnsafeMutablePointer(to: &msg) { mptr in
            let len = llama_chat_apply_template(nil, mptr, 1, true, nil, 0)
            var buf = [CChar](repeating: 0, count: Int(len) + 8)
            llama_chat_apply_template(nil, mptr, 1, true, &buf, Int32(buf.count))
            formatted = String(cString: buf)
        }
        let bw = mtmd_helper_bitmap_init_from_file(v, imagePath, false)
        guard let bitmap = bw.bitmap else { throw MtmdError.imageLoadFailed(imagePath + " || stb failed") }
        defer { mtmd_bitmap_free(bitmap) }
        let chunks = mtmd_input_chunks_init()
        defer { mtmd_input_chunks_free(chunks) }
        var text = mtmd_input_text()
        try formatted.withCString { fptr in
            text.text = fptr
            text.text_len = formatted.utf8.count
            text.add_special = true
            text.parse_special = true
            var bm: [OpaquePointer?] = [bitmap]
            let res = try mtmd_tokenize(v, chunks, &text, &bm, 1)
            if res != 0 { throw MtmdError.tokenizeFailed(res) }
        }
        var embds: [[Float]] = []
        var ntoks: [Int] = []
        let nChunks = mtmd_input_chunks_size(chunks)
        for i in 0..<nChunks {
            let ck = mtmd_input_chunks_get(chunks, i)
            if mtmd_input_chunk_get_type(ck) != MTMD_INPUT_CHUNK_TYPE_IMAGE { continue }
            let t0 = DispatchTime.now().uptimeNanoseconds
            let r = mtmd_encode_chunk(v, ck)
            if r != 0 { throw MtmdError.decodeFailed(r) }
            let nTok = mtmd_input_chunk_get_n_tokens(ck)
            if let embd = mtmd_get_output_embd(v) {
                let count = Int(nTok) * Int(nEmbdInp)
                if count > 0 && count < 20_000_000 {
                    embds.append(Array(UnsafeBufferPointer(start: embd, count: count)))
                    ntoks.append(nTok)
                }
            }
        }
        guard !embds.isEmpty else { throw MtmdError.decodeFailed(-1) }
        cachedImageEmbds[imagePath] = embds
        cachedImageNTokens[imagePath] = ntoks
        let totalMs = Int64((DispatchTime.now().uptimeNanoseconds - tStart) / 1_000_000)
        lastDebug += " | prepared \(embds.count) image chunks, \(totalMs)ms"
        return (totalMs, embds.count)
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
    private func attachLogs() {
        logLock.lock(); lastDebug = gMtmdLogs; logLock.unlock()
    }

    private func fetchMtmdLogs() -> String {
        logLock.lock(); defer { logLock.unlock() }
        return String(gMtmdLogs.suffix(1500))
    }

    func runOnce(imagePath: String, prompt: String, nPredict: Int32) throws -> BenchResult {
        guard let m = model, let c = context, let v = vctx, let smpl = sampling else {
            throw MtmdError.modelLoadFailed("not loaded")
        }
        llama_memory_clear(llama_get_memory(c), true) // fresh sequence per query
        let tStart = DispatchTime.now().uptimeNanoseconds

        // image -> bitmap via mtmd's stb_image loader (native path; the earlier
        // failure was a broken framework merge, not memory pressure)
        let bw = mtmd_helper_bitmap_init_from_file(v, imagePath, false)
        var bitmap: OpaquePointer?
        if !imagePath.isEmpty {
            let bw = mtmd_helper_bitmap_init_from_file(v, imagePath, false)
            guard let b = bw.bitmap else {
                throw MtmdError.imageLoadFailed(imagePath + " || stb failed")
            }
            bitmap = b
            let nb = mtmd_bitmap_get_n_bytes(b)
            if let data = mtmd_bitmap_get_data(b) {
                var s: UInt64 = 0, s2: UInt64 = 0
                let stride = max(1, nb / 4096)
                var cnt = 0
                var i = 0
                while i < nb { let v = UInt64(data[i]); s += v; s2 += v*v; cnt += 1; i += stride }
                if cnt > 0 {
                    let mean = Double(s) / Double(cnt)
                    let var_ = Double(s2) / Double(cnt) - mean*mean
                    lastDebug += " | px_n=\(nb) mean=\(String(format: "%.1f", mean)) var=\(String(format: "%.0f", var_))"
                }
            }
        }
        defer { if let b = bitmap { mtmd_bitmap_free(b) } }

        // chat format
        var finalPrompt = prompt
        if !imagePath.isEmpty && !finalPrompt.contains(String(cString: mtmd_default_marker())) {
            finalPrompt = String(cString: mtmd_default_marker()) + finalPrompt
        }
        // role AND content must stay alive across llama_chat_apply_template:
        // the implicit String->C pointer conversion dies at the end of the
        // statement and the template reads dangling pointers otherwise.
        let contentPtr = strdup(finalPrompt)
        let rolePtr = strdup("user")
        defer { free(contentPtr); free(rolePtr) }
        var msg = llama_chat_message(role: rolePtr, content: contentPtr)
        var formatted = ""
        withUnsafeMutablePointer(to: &msg) { mptr in
            let len = llama_chat_apply_template(nil, mptr, 1, true, nil, 0)
            var buf = [CChar](repeating: 0, count: Int(len) + 8)
            llama_chat_apply_template(nil, mptr, 1, true, &buf, Int32(buf.count))
            formatted = String(cString: buf)
        }
        lastDebug += " | formatted=" + formatted

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
                var bm: [OpaquePointer?] = []
                if let bitmap { bm = [bitmap] }
                let res = try mtmd_tokenize(v, chunks, &text, &bm, bm.count)
                if res != 0 {
                    attachLogs()
                    throw MtmdError.tokenizeFailed(res)
                }
            } catch {
                attachLogs()
                throw MtmdError.tokenizeFailed(-1)
            }
        }

        var encodeSum: Int64 = 0
        var prefillSum: Int64 = 0
        var nPast: llama_pos = 0
        let nChunks = mtmd_input_chunks_size(chunks)
        var chunkDesc = ""
        for ci in 0..<nChunks {
            let ck = mtmd_input_chunks_get(chunks, ci)
            let ct = mtmd_input_chunk_get_type(ck)
            chunkDesc += "\(ct == MTMD_INPUT_CHUNK_TYPE_TEXT ? "T" : "I")"
        }
        lastDebug += " | chunks=\(chunkDesc)"

        var imgIdx = 0
        for i in 0..<nChunks {
            let chunk = mtmd_input_chunks_get(chunks, i)
            let ctype = mtmd_input_chunk_get_type(chunk)
            if ctype == MTMD_INPUT_CHUNK_TYPE_TEXT {
                let t0 = DispatchTime.now().uptimeNanoseconds
                var newN = nPast
                let r = mtmd_helper_eval_chunk_single(v, c, chunk, nPast, 0, 2048, i == nChunks - 1, &newN)
                if r != 0 { lastDebug += " | text_chunk_eval r=\(r) chunk=\(i)"; throw MtmdError.decodeFailed(r) }
                nPast = newN
                prefillSum += Int64((DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000)
            } else {
                // image chunk: prefer pre-computed embeddings (photo flow)
                var embd: UnsafeMutablePointer<Float>? = nil
                var scratch: [Float] = []
                let nTok = mtmd_input_chunk_get_n_tokens(chunk)
                if let cached = cachedImageEmbds[imagePath],
                   let cachedNT = cachedImageNTokens[imagePath],
                   imgIdx < cached.count,
                   cachedNT[imgIdx] == nTok,
                   cached[imgIdx].count == Int(nTok) * Int(nEmbdInp) {
                    scratch = cached[imgIdx]
                    scratch.withUnsafeMutableBufferPointer { embd = $0.baseAddress }
                    lastDebug += " | CACHE_HIT(\(imgIdx))"
                } else {
                    // encode image chunk (vision encoder pass)
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    let r = mtmd_encode_chunk(v, chunk)
                    if r != 0 { lastDebug += " | encode_chunk r=\(r) chunk=\(i) | mtmd_logs: " + fetchMtmdLogs(); throw MtmdError.decodeFailed(r) }
                    encodeSum += Int64((DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000)
                    embd = mtmd_get_output_embd(v)
                }
                imgIdx += 1
                if embd == nil { lastDebug += " | no_embd chunk=\(i)"; throw MtmdError.decodeFailed(-1) }
                var esum: Double = 0
                var ecnt = 0
                if let embd {
                    for k in 0..<min(4096, Int(nPast) + 1) {
                        let val = embd[k]
                        if !val.isNaN { esum += Double(abs(val)); ecnt += 1 }
                    }
                }
                var e16 = ""
                if let embd {
                    for k in 0..<16 { e16 += String(format: "%.4f ", embd[k]) }
                    var sum2: Double = 0, n2 = 0
                    for k in 0..<1536 { let v = Double(embd[k]); sum2 += v; n2 += 1 }
                    let m2 = sum2 / Double(max(1, n2))
                    var v2: Double = 0
                    for k in 0..<1536 { let v = Double(embd[k]); v2 += (v - m2) * (v - m2) }
                    var spread = ""
                    for k in stride(from: 0, to: 1536, by: 153) { spread += String(format: "%.3f ", embd[k]) }
                    lastDebug += " | embed_mean_abs=\(esum / Double(max(1, ecnt))) dim1536_mean=\(String(format: "%.4f", m2)) std=\(String(format: "%.4f", sqrt(v2 / Double(max(1, n2))))) spread=[\(spread)]"
                }
                var newN = nPast
                let r2 = mtmd_helper_decode_image_chunk(v, c, chunk, embd, nPast, 0, 2048, &newN, nil, nil)
                if r2 != 0 { lastDebug += " | image_chunk_decode r=\(r2) chunk=\(i) n_past=\(nPast)"; throw MtmdError.decodeFailed(r2) }
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
        var logitStats = "no-logits"
        if let lp = llama_get_logits(c) {
            var mx: Float = -1e30
            var argmax: Int32 = -1
            let nv = llama_vocab_n_tokens(vocab)
            for j in 0..<nv {
                if lp[Int(j)] > mx { mx = lp[Int(j)]; argmax = j }
            }
            logitStats = "max=\(mx) argmax=\(argmax) eog_argmax=\(llama_vocab_is_eog(vocab, argmax))"
            var top: [(Float, Int32)] = []
            for j in 0..<nv { top.append((lp[Int(j)], Int32(j))) }
            top.sort { $0.0 > $1.0 }
            var topStr = ""
            for (v, t) in top.prefix(5) {
                var pc = [CChar](repeating: 0, count: 32)
                let np = llama_token_to_piece(vocab, t, &pc, 32, 0, false)
                let s = String(decoding: pc.prefix(Int(max(np, 0))).map { UInt8(bitPattern: $0) }, as: UTF8.self)
                topStr += " [\(v):\(s)]"
            }
            logitStats += " top5:" + topStr
        }
        lastDebug += " | first_tok=\(tok) eog=\(llama_vocab_is_eog(vocab, tok)) n_past=\(nPast) | " + logitStats
        llama_sampler_accept(smpl, tok)

        var piece = [CChar](repeating: 0, count: 64)
        while nDec < nPredict {
            if llama_vocab_is_eog(vocab, tok) { break }
            var nP = llama_token_to_piece(vocab, tok, &piece, Int32(piece.count), 0, false)
            if nP < 0 {
                piece = [CChar](repeating: 0, count: Int(-nP))
                nP = llama_token_to_piece(vocab, tok, &piece, Int32(piece.count), 0, false)
            }
            if nP > 0 {
                output += String(decoding: piece.prefix(Int(nP)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
            }
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
            batteryLevel: UIDevice.current.batteryLevel,
            debug: lastDebug
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

    // Decode + downscale an image to RGB bytes for mtmd. stb_image (used by
    // mtmd_helper_bitmap_init_from_file) mallocs the full decoded buffer
    // (5712x3213 -> ~55MB) which can fail on-device under memory pressure;
    // UIKit handles memory properly and we cap the longest edge at 1568px,
    // which also reduces the number of vision tiles -> faster encode.
    private func makeBitmap(from path: String) -> OpaquePointer? {
        lastDebug = "makeBitmap path=\(path)"
        guard let ui = UIImage(contentsOfFile: path) else { lastDebug += " | UIImage nil"; return nil }
        guard let cg = ui.cgImage else { lastDebug += " | cgImage nil"; return nil }
        lastDebug += " | decoded \(cg.width)x\(cg.height)"
        // Feed the image at native resolution: mtmd/Qwen2-VL tiling handles it
        // (downscaling here loses small text — the Mac reads the full image).
        let cw = max(1, Int(ui.size.width.rounded()))
        let ch = max(1, Int(ui.size.height.rounded()))
        // iOS bitmap contexts require an alpha channel (kCGImageAlphaNone is
        // macOS-only) -> draw RGBA, then compact to RGB for mtmd.
        let bytesPerRow = cw * 4
        var rgba = [UInt8](repeating: 0, count: ch * bytesPerRow)
        guard let ctx = CGContext(data: &rgba, width: cw, height: ch,
                                  bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            lastDebug += " | CGContext nil"
            return nil
        }
        lastDebug += " | scaled \(cw)x\(ch)"
        // NOTE: CGBitmapContext memory is already top-down; flipping corrupts
        // the image (verified: flipped render is unreadable, plain is correct).
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: CGFloat(cw), height: CGFloat(ch)))
        var rgb = [UInt8](repeating: 0, count: cw * ch * 3)
        var src = 0, dst = 0
        while src < rgba.count {
            rgb[dst] = rgba[src]; rgb[dst+1] = rgba[src+1]; rgb[dst+2] = rgba[src+2]
            src += 4; dst += 3
        }
        return mtmd_bitmap_init(UInt32(cw), UInt32(ch), rgb)
    }
}
