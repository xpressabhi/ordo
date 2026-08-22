import Foundation
import UIKit

// BenchmarkEngine — drives the eval set and stress tests on-device.
// Mirrors bench/mac/run_bench.py + Part 4 of the assignment.
@MainActor
class BenchmarkEngine: ObservableObject {
    @Published var log = ""
    @Published var isRunning = false
    @Published var isLoaded = false

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
        Bundle.main.url(forResource: "eval", withExtension: "json", subdirectory: "eval") ??
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
    func findModelFiles() -> (model: String, mmproj: String)? {
        // 1. bundled resources (installed with the app)
        if let m = Bundle.main.url(forResource: "Qwen2-VL-2B-Instruct-Q4_K_M", withExtension: "gguf", subdirectory: "model_data") {
            let mm = Bundle.main.url(forResource: "mmproj-Qwen2-VL-2B-Instruct-f16", withExtension: "gguf", subdirectory: "model_data")
                ?? Bundle.main.url(forResource: "mmproj-Qwen2-VL-2B-Instruct-Q4_K_M", withExtension: "gguf", subdirectory: "model_data")
                ?? Bundle.main.url(forResource: "mmproj-Qwen2-VL-2B-Instruct-Q8_0", withExtension: "gguf", subdirectory: "model_data")
            if let mm { return (m.path, mm.path) }
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
                logMsg("[\(i)/\(iterations)] ttft=\(res.ttftMs)ms wall=\(String(format: "%.1f", dt))s thermal=\(thermalName(thermal)) mem=\(mem)MB dec=\(res.nDecodeToks)tok")
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
                logMsg("TTFT: first=\(first)ms last=\(last)ms avg=\(avg)ms (fall-off: \(last - first)ms)")
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
                    logMsg("t=\(Int(Date().timeIntervalSince(Date().addingTimeInterval(-Double(minutes * 60))).magnitude))s? | \(resStr) wall=\(String(format: "%.1f", dt))s thermal=\(thermalName(thermal)) mem=\(mem)MB")
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