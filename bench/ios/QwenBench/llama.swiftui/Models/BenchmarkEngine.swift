import Foundation
import UIKit

// BenchmarkEngine — drives the eval set and stress tests on-device.
// Mirrors bench/mac/run_bench.py + Part 4 of the assignment.
@MainActor
class BenchmarkEngine: ObservableObject {
    @Published var log = ""
    @Published var isRunning = false
    @Published var isLoaded = false
    // photo flow state
    @Published var photoState = ""
    @Published var photoPath: String?
    @Published var photoEncodeMs: Int64 = 0
    @Published var lastAnswer = ""

    private var ctx: MtmdContext?
    private var modelPath: String?
    private var mmprojPath: String?
    private var results: [String] = []

    private let nsPerS = 1_000_000_000.0

    // ---- paths -------------------------------------------------------------
    private var documentsDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
    private var evalJSONURL: URL {
        let args = ProcessInfo.processInfo.arguments
        let name = args.contains("autorun-512") ? "eval_512"
            : args.contains("autorun-1024") ? "eval_1024" : "eval"
        return Bundle.main.url(forResource: name, withExtension: "json", subdirectory: "eval") ??
            documentsDir.appendingPathComponent("eval.json")
    }

    func photoURLFor(_ rel: String) -> URL? { photoURL(rel) }

    private func photoURL(_ rel: String) -> URL? {
        let name = (rel as NSString).lastPathComponent
        let ext = (name as NSString).pathExtension
        let base = (name as NSString).deletingPathExtension
        let dir = (rel as NSString).deletingLastPathComponent  // e.g. "photos_small"
        return Bundle.main.url(forResource: base, withExtension: ext, subdirectory: "eval/\(dir)")
            ?? documentsDir.appendingPathComponent(rel)
    }

    func appendDiag(_ text: String) {
        let url = documentsDir.appendingPathComponent("diag.txt")
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile()
            h.write((text + "\n").data(using: .utf8)!)
            try? h.close()
        } else {
            try? (text + "\n").write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private func appendIncremental(_ line: String) {
        let url = documentsDir.appendingPathComponent("results.jsonl")
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile()
            h.write((line + "\n").data(using: .utf8)!)
            try? h.close()
        } else {
            try? (line + "\n").write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private func logMsg(_ s: String) {
        log += s + "\n"
        print(s)
    }

    func appendLog(_ s: String) { logMsg(s) }

    func evalItemsForUI() -> [[String: Any]] { loadEvalItems() }

    // ---- load ----------------------------------------------------------------
    func findModelFiles(prefer: String? = nil) -> (model: String, mmproj: String)? {
        // 1. bundled resources (installed with the app): pair model + mmproj by family
        if let res = Bundle.main.resourceURL {
            let md = res.appendingPathComponent("model_data")
            let files = (try? FileManager.default.contentsOfDirectory(at: md, includingPropertiesForKeys: nil)) ?? []
            let gguFs = files.filter { $0.pathExtension == "gguf" }
            let models = gguFs.filter { !$0.lastPathComponent.contains("mmproj") }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            let mmprojs = gguFs.filter { $0.lastPathComponent.contains("mmproj") }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }

            func family(of url: URL) -> String {
                var n = url.lastPathComponent
                n = n.replacingOccurrences(of: "mmproj-", with: "")
                if let range = n.range(of: "-Instruct") { n = String(n[..<range.lowerBound]) }
                return n
            }

            func pair(prefer p: String?) -> (URL, URL)? {
                let cand = models.filter { p == nil || $0.lastPathComponent.contains(p!) }
                let cmp = mmprojs.filter { p == nil || $0.lastPathComponent.contains(p!) }
                for m in cand {
                    if let mm = cmp.first(where: { family(of: $0) == family(of: m) }) {
                        return (m, mm)
                    }
                }
                return nil
            }

            if let prefer, let hit = pair(prefer: prefer) { return (hit.0.path, hit.1.path) }
            // no prefer: default to the eval model (256M) if present, else first family
            if let hit = pair(prefer: "256M") ?? pair(prefer: nil) { return (hit.0.path, hit.1.path) }
        }
        // 2. Documents (dropped in via Finder/Files)
        let docs = try? FileManager.default.contentsOfDirectory(at: documentsDir, includingPropertiesForKeys: nil)
        var model: String?
        var mmproj: String?
        for u in docs ?? [] where u.pathExtension == "gguf" {
            if u.lastPathComponent.contains("mmproj") { mmproj = u.path } else { model = u.path }
        }
        guard let model, let mmproj else { return nil }
        return (model, mmproj)
    }

    func loadModel() {
        guard !isRunning else { return }
        guard let (m, mm) = findModelFiles() else {
            logMsg("No model/mmproj GGUF found in Documents. Copy them via Finder (Files app sharing).")
            return
        }
        logMsg("Loading model: \(URL(fileURLWithPath: m).lastPathComponent)")
        logMsg("Loading mmproj: \(URL(fileURLWithPath: mm).lastPathComponent)")
        let c = MtmdContext()
        Task {
            do {
                try await c.load(modelPath: m, mmprojPath: mm)
                ctx = c
                isLoaded = true
                logMsg("Loaded OK")
            } catch {
                logMsg("Load failed: \(error)")
            }
        }
    }

    // ---- autorun (driven from CLI: launch with -autorun) ---------------------
    func autoRunTextOnly() async {
        UIApplication.shared.isIdleTimerDisabled = true
        logMsg("AUTORUN-TEXT start")
        loadModel()
        for _ in 0..<240 {
            if isLoaded { break }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        if !isLoaded { logMsg("load failed"); return }
        if let res = await askOnce(imagePath: nil, prompt: "What is the capital of France?", nPredict: 32) {
            logMsg("TEXT-ONLY: enc=\(res.encodeMs)ms pre=\(res.prefillMs)ms dec=\(res.decodeMs)ms n_dec=\(res.nDecodeToks) ttft=\(res.ttftMs)ms")
            logMsg("  -> \(res.output)")
            logMsg("  dbg: \(res.debug)")
        }
        logMsg("AUTORUN-TEXT DONE")
    }

    func autoRunQwenFast() async {
        UIApplication.shared.isIdleTimerDisabled = true
        logMsg("AUTORUN-QWEN-FAST start")
        guard let (m, mm) = findModelFiles(prefer: "Qwen2-VL-2B-Instruct-Q4_K_M") else {
            logMsg("no Qwen2-VL Q4 bundle"); return
        }
        logMsg("Loading: \(URL(fileURLWithPath: m).lastPathComponent) + \(URL(fileURLWithPath: mm).lastPathComponent)")
        let c = MtmdContext()
        do {
            try await c.load(modelPath: m, mmprojPath: mm)
            ctx = c
            isLoaded = true
        } catch { logMsg("Load failed: \(error)"); return }
        guard !isRunning else { return }
        isRunning = true
        results = []
        try? FileManager.default.removeItem(at: documentsDir.appendingPathComponent("results.jsonl"))
        try? FileManager.default.removeItem(at: documentsDir.appendingPathComponent("diag.txt"))
        Task {
            let t0 = Date()
            logMsg("== Qwen-fast eval run (Q4 mmproj, n_batch 4096, short answers) ==")
            guard let data = try? Data(contentsOf: evalJSONURL),
                  let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let items = manifest["items"] as? [[String: Any]] else {
                logMsg("eval.json not found or invalid"); isRunning = false; return
            }
            for (idx, item) in items.enumerated() {
                guard let q = item["question"] as? String, let rel = item["path"] as? String,
                      let imgURL = photoURL(rel) else { continue }
                let shortQ = "Answer in one short phrase. " + q
                logMsg("[\(idx + 1)/\(items.count)] \(rel)")
                guard let res = await askOnce(imagePath: imgURL.path, prompt: shortQ, nPredict: 24) else { continue }
                results.append(res.toJSON())
                appendIncremental(res.toJSON())
                appendDiag("\(rel) | \(res.debug)")
                logMsg("  enc=\(res.encodeMs)ms pre=\(res.prefillMs)ms dec=\(res.decodeMs)ms ttft=\(res.ttftMs)ms n_dec=\(res.nDecodeToks) | \(res.output.prefix(50))")
            }
            let dt = Date().timeIntervalSince(t0)
            logMsg("Eval done in \(String(format: "%.0f", dt))s")
            isRunning = false
            if let url = exportResults() { logMsg("results -> \(url.path)") }
            logMsg("AUTORUN-QWEN-FAST DONE")
        }
    }

    func autoRunWithModel(_ prefer: String) async {
        UIApplication.shared.isIdleTimerDisabled = true
        logMsg("AUTORUN-\(prefer) start")
        guard let (m, mm) = findModelFiles(prefer: prefer) else { logMsg("no \(prefer) model bundled"); return }
        logMsg("Loading: \(URL(fileURLWithPath: m).lastPathComponent)")
        let c = MtmdContext()
        do {
            try await c.load(modelPath: m, mmprojPath: mm)
            ctx = c
            isLoaded = true
        } catch { logMsg("Load failed: \(error)"); return }
        runEvalSet()
        while isRunning { try? await Task.sleep(nanoseconds: 500_000_000) }
        if let url = exportResults() { logMsg("AUTORUN-\(prefer) results -> \(url.path)") }
        logMsg("AUTORUN-\(prefer) DONE")
    }

    func autoRunMultiturn() async {
        UIApplication.shared.isIdleTimerDisabled = true
        logMsg("AUTORUN-MULTITURN start")
        loadModel()
        for _ in 0..<240 { if isLoaded { break }; try? await Task.sleep(nanoseconds: 500_000_000) }
        if !isLoaded { logMsg("load failed"); return }
        guard let data = try? Data(contentsOf: evalJSONURL),
              let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = manifest["items"] as? [[String: Any]] else {
            logMsg("eval.json not found"); return
        }
        for (idx, item) in items.prefix(6).enumerated() {
            guard let q = item["question"] as? String, let rel = item["path"] as? String,
                  let imgURL = photoURL(rel) else { continue }
            logMsg("[T1 \(idx + 1)/6] \(rel)")
            if let r1 = await askOnce(imagePath: imgURL.path, prompt: q) {
                logMsg("  turn1 enc=\(r1.encodeMs)ms ttft=\(r1.ttftMs)ms | \(r1.output.prefix(50))")
                appendDiag("\(rel) | T1 \(r1.debug)")
            }
            let follow = "What color is the main object in the image?"
            logMsg("[T2 \(idx + 1)/6] \(rel)")
            if let r2 = await askOnce(imagePath: imgURL.path, prompt: follow) {
                logMsg("  turn2 enc=\(r2.encodeMs)ms ttft=\(r2.ttftMs)ms | \(r2.output.prefix(50))")
                appendDiag("\(rel) | T2 \(r2.debug)")
            }
        }
        logMsg("AUTORUN-MULTITURN DONE")
    }

    func autoRun() async {
        UIApplication.shared.isIdleTimerDisabled = true
        logMsg("AUTORUN start")
        loadModel()
        for _ in 0..<240 {
            if isLoaded { break }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        if !isLoaded { logMsg("AUTORUN: model load failed"); return }
        runEvalSet()
        while isRunning { try? await Task.sleep(nanoseconds: 500_000_000) }
        if let url = exportResults() {
            logMsg("AUTORUN results -> \(url.path)")
        }
        logMsg("AUTORUN DONE")
    }

    // ---- photo flow: compress -> pre-compute vision -> answer -----------------
    func compressJPEG(_ image: UIImage, maxDim: CGFloat = 768, maxBytes: Int = 100_000) -> Data? {
        var img = image
        let longest = max(img.size.width, img.size.height)
        if longest > maxDim {
            let r = maxDim / longest
            let newSize = CGSize(width: img.size.width * r, height: img.size.height * r)
            let renderer = UIGraphicsImageRenderer(size: newSize)
            img = renderer.image { _ in img.draw(in: CGRect(origin: .zero, size: newSize)) }
        }
        var q: CGFloat = 0.7
        while q >= 0.2 {
            if let d = img.jpegData(compressionQuality: q), d.count <= maxBytes { return d }
            q -= 0.1
        }
        return img.jpegData(compressionQuality: 0.2)
    }

    func processPhoto(_ image: UIImage) {
        guard ctx != nil else { logMsg("Load model first"); return }
        photoState = "compressing"
        Task {
            guard let data = compressJPEG(image) else {
                photoState = "compress failed"; logMsg("compress failed"); return
            }
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("qbench_shot.jpg")
            do { try data.write(to: url) } catch { photoState = "write failed"; return }
            photoPath = url.path
            logMsg("photo compressed: \(data.count / 1024)KB")
            photoState = "encoding vision…"
            guard let c = ctx else { photoState = "model not loaded"; return }
            do {
                let r = try await c.prepareImage(imagePath: url.path)
                photoEncodeMs = r.encodeMs
                photoState = "ready \(r.nChunks) chunks \(r.encodeMs)ms"
                logMsg("vision pre-computed: \(r.nChunks) chunks in \(r.encodeMs)ms")
            } catch {
                photoState = "encode failed: \(error)"
                logMsg("encode failed: \(error)")
            }
        }
    }

    func askAboutPhoto(_ prompt: String) {
        guard let photoPath, ctx != nil else { logMsg("Take a photo first"); return }
        photoState = "thinking…"
        Task {
            let t0 = Date()
            if let res = await askOnce(imagePath: photoPath, prompt: prompt, nPredict: 96) {
                let dt = Date().timeIntervalSince(t0)
                lastAnswer = res.output
                photoState = "answered in \(String(format: "%.1f", dt))s (enc \(res.encodeMs)ms pre \(res.prefillMs)ms dec \(res.decodeMs)ms ttft \(res.ttftMs)ms)"
                logMsg("ASK: \(res.output)")
            } else {
                photoState = "ask failed"
            }
        }
    }

    // ---- single prompt -------------------------------------------------------
    func askOnce(imagePath: String?, prompt: String, nPredict: Int32 = 96) async -> BenchResult? {
        guard let c = ctx else { logMsg("Not loaded"); return nil }
        do {
            if let imagePath {
                return try await c.runOnce(imagePath: imagePath, prompt: prompt, nPredict: nPredict)
            } else {
                return try await c.runOnce(imagePath: "", prompt: prompt, nPredict: nPredict)
            }
        } catch {
            let dbg = await c.lastDebug
            logMsg("run error: \(error) || \(dbg)")
            return nil
        }
    }

    // ---- eval set -------------------------------------------------------------
    func runEvalSet() {
        guard !isRunning, ctx != nil else { return }
        isRunning = true
        results = []
        try? FileManager.default.removeItem(at: documentsDir.appendingPathComponent("results.jsonl"))
        try? FileManager.default.removeItem(at: documentsDir.appendingPathComponent("diag.txt"))
        Task {
            let t0 = Date()
            logMsg("== Eval set run ==")
            guard let data = try? Data(contentsOf: evalJSONURL),
                  let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let items = manifest["items"] as? [[String: Any]] else {
                logMsg("eval.json not found or invalid")
                isRunning = false
                return
            }
            for (idx, item) in items.enumerated() {
                guard let q = item["question"] as? String,
                      let rel = item["path"] as? String else { continue }
                guard let imgURL = photoURL(rel) else { logMsg("photo missing: \(rel)"); continue }
                logMsg("[\(idx + 1)/\(items.count)] \(rel)")
                guard let res = await askOnce(imagePath: imgURL.path, prompt: q) else { continue }
                results.append(res.toJSON())
                appendIncremental(res.toJSON())
                appendDiag("\(rel) | \(res.debug)")
                logMsg("  enc=\(res.encodeMs)ms pre=\(res.prefillMs)ms dec=\(res.decodeMs)ms ttft=\(res.ttftMs)ms | \(res.output.prefix(60)) | \(res.debug)")
            }
            let dt = Date().timeIntervalSince(t0)
            logMsg("Eval set done in \(String(format: "%.0f", dt))s")
            isRunning = false
        }
    }

    // ---- stress: 20 sequential queries -----------------------------------------
    func runStress(iterations: Int = 20) {
        guard !isRunning, ctx != nil else { return }
        isRunning = true
        Task {
            UIDevice.current.isBatteryMonitoringEnabled = true
            let b0 = UIDevice.current.batteryLevel
            logMsg("== Stress: \(iterations) sequential queries ==")
            logMsg("battery at start: \(b0 >= 0 ? String(format: "%.0f%%", b0 * 100) : "n/a")")
            let t0 = Date()
            // use the first eval item as the fixed query
            let items = loadEvalItems()
            guard let first = items.first else { logMsg("no eval items"); isRunning = false; return }
            let imgURL = photoURL(first["path"] as! String) ?? documentsDir.appendingPathComponent(first["path"] as! String)
            let q = first["question"] as! String
            var times: [Int64] = []
            for i in 1...iterations {
                let tq0 = Date()
                guard let res = await askOnce(imagePath: imgURL.path, prompt: q, nPredict: 48) else { continue }
                times.append(res.ttftMs)
                let dt = Date().timeIntervalSince(tq0)
                let thermal = ProcessInfo.processInfo.thermalState
                let mem = footprintMB()
                let line = "[\(i)/\(iterations)] ttft=\(res.ttftMs)ms wall=\(String(format: "%.1f", dt))s thermal=\(thermalName(thermal)) mem=\(mem)MB dec=\(res.nDecodeToks)tok"
                logMsg(line)
                appendDiag(line)
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            let total = Date().timeIntervalSince(t0)
            let b1 = UIDevice.current.batteryLevel
            let drain = b0 >= 0 && b1 >= 0 ? (b0 - b1) * 100 : -1
            logMsg("20 queries in \(String(format: "%.0f", total))s")
            logMsg("battery at end: \(b1 >= 0 ? String(format: "%.0f%%", b1 * 100) : "n/a") (drain \(drain >= 0 ? String(format: "%.1f%%", drain) : "n/a"))")
            if !times.isEmpty {
                let avg = times.reduce(0, +) / Int64(times.count)
                let first = times.first ?? 0
                let last = times.last ?? 0
                let sumLine = "TTFT: first=\(first)ms last=\(last)ms avg=\(avg)ms (fall-off: \(last - first)ms)"
            logMsg(sumLine)
            appendDiag(sumLine)
            }
            isRunning = false
        }
    }

    // ---- 10-minute thermal soak -------------------------------------------------
    func runSoak(minutes: Int = 10) {
        guard !isRunning, ctx != nil else { return }
        isRunning = true
        Task {
            logMsg("== Thermal soak \(minutes)min ==")
            let items = loadEvalItems()
            guard let first = items.first else { logMsg("no eval items"); isRunning = false; return }
            let imgURL = photoURL(first["path"] as! String) ?? documentsDir.appendingPathComponent(first["path"] as! String)
            let q = first["question"] as! String
            let end = Date().addingTimeInterval(TimeInterval(minutes * 60))
            var sampleCount = 0
            while Date() < end && !Task.isCancelled {
                let tq0 = Date()
                let res = await askOnce(imagePath: imgURL.path, prompt: q, nPredict: 32)
                let dt = Date().timeIntervalSince(tq0)
                let thermal = ProcessInfo.processInfo.thermalState
                let mem = footprintMB()
                let resStr = res.map { "enc=\($0.encodeMs)ms dec=\($0.decodeMs)ms ttft=\($0.ttftMs)ms" } ?? "error"
                sampleCount += 1
                if sampleCount % 3 == 0 || thermal == .serious || thermal == .critical {
                    let line = "soak t=\(sampleCount) | \(resStr) wall=\(String(format: "%.1f", dt))s thermal=\(thermalName(thermal)) mem=\(mem)MB"
                    logMsg(line)
                    appendDiag(line)
                }
                try? await Task.sleep(nanoseconds: 300_000_000)
            }
            logMsg("Soak done. \(sampleCount) inferences.")
            isRunning = false
        }
    }

    // ---- export ---------------------------------------------------------------
    func exportResults() -> URL? {
        guard !results.isEmpty else { return nil }
        let url = documentsDir.appendingPathComponent("results.json")
        let body = "[\n" + results.joined(separator: ",\n") + "\n]\n"
        try? body.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    // ---- helpers ---------------------------------------------------------------
    private func loadEvalItems() -> [[String: Any]] {
        guard let data = try? Data(contentsOf: evalJSONURL),
              let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = manifest["items"] as? [[String: Any]] else { return [] }
        return items
    }

    private func thermalName(_ t: ProcessInfo.ThermalState) -> String {
        switch t {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private func footprintMB() -> Int64 {
        var info = task_vm_info()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { pp in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), pp, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int64(info.phys_footprint) / (1024 * 1024) : -1
    }
}