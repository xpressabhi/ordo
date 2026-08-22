import SwiftUI

// QwenBench — on-device vision benchmark harness.
// Runs the eval set (eval.json + bundled photos) through Qwen2-VL-2B and
// measures encode/prefill/decode/TTFT per stage, plus thermal/battery/memory.
struct ContentView: View {
    @StateObject var engine = BenchmarkEngine()
    @State private var prompt = "What does this say?"
    @State private var showShare = false
    @State private var shareURL: URL?
    @State private var showHelp = false

    var body: some View {
        NavigationView {
            VStack(alignment: .leading) {
                ScrollView(.vertical, showsIndicators: true) {
                    Text(engine.log)
                        .font(.system(size: 11, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                TextField("Prompt (used for single-shot + eval if no eval.json)", text: $prompt)
                    .textFieldStyle(.roundedBorder)
                    .padding(.horizontal)

                HStack {
                    Button(engine.isLoaded ? "Reload model" : "Load model") { engine.loadModel() }
                    Button("Single shot") { runSingle() }
                    Button("Eval set") { engine.runEvalSet() }
                }
                .buttonStyle(.bordered)
                .disabled(engine.isRunning)

                HStack {
                    Button("Stress 20x") { engine.runStress(iterations: 20) }
                    Button("Soak 10min") { engine.runSoak(minutes: 10) }
                    Button("Export") { export() }
                }
                .buttonStyle(.bordered)
                .disabled(engine.isRunning)

                HStack {
                    Text(engine.isRunning ? "RUNNING..." : "idle")
                        .font(.caption)
                        .foregroundColor(engine.isRunning ? .red : .secondary)
                    Spacer()
                    Text("thermal: \(thermalName(ProcessInfo.processInfo.thermalState))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal)
            }
            .padding(.vertical)
            .onAppear {
                let args = ProcessInfo.processInfo.arguments
                engine.appendDiag("LAUNCH args: \(args)")
                if args.contains("-autorun-text") || args.contains("autorun-text") {
                    Task { await engine.autoRunTextOnly() }
                } else if args.contains("-autorun-stress") || args.contains("autorun-stress") {
                    Task {
                        engine.loadModel()
                        for _ in 0..<240 { if engine.isLoaded { break }; try? await Task.sleep(nanoseconds: 500_000_000) }
                        engine.runStress(iterations: 20)
                    }
                } else if args.contains("-autorun-soak") || args.contains("autorun-soak") {
                    Task {
                        engine.loadModel()
                        for _ in 0..<240 { if engine.isLoaded { break }; try? await Task.sleep(nanoseconds: 500_000_000) }
                        engine.runSoak(minutes: 5)
                    }
                } else if args.contains("-autorun") || args.contains("autorun") {
                    Task { await engine.autoRun() }
                }
            }
            .navigationBarTitle("QwenBench", displayMode: .inline)
            .navigationBarItems(trailing: Button("Help") { showHelp = true })
            .sheet(isPresented: $showHelp) {
                HelpView()
            }
            .sheet(isPresented: $showShare) {
                if let u = shareURL {
                    ShareSheet(items: [u])
                }
            }
        }
    }

    func runSingle() {
        Task {
            // prefer the first eval photo if present
            let items = engine.evalItemsForUI()
            if let first = items.first,
               let rel = first["path"] as? String,
               let imgURL = engine.photoURLFor(rel) {
                let q = (first["question"] as? String) ?? prompt
                if let res = await engine.askOnce(imagePath: imgURL.path, prompt: q, nPredict: 96) {
                    engine.appendLog("SINGLE: enc=\(res.encodeMs)ms pre=\(res.prefillMs)ms dec=\(res.decodeMs)ms ttft=\(res.ttftMs)ms")
                    engine.appendLog("  -> \(res.output)")
                }
            } else if let res = await engine.askOnce(imagePath: nil, prompt: prompt, nPredict: 96) {
                engine.appendLog("SINGLE(text): ttft=\(res.ttftMs)ms dec=\(res.decodeMs)ms")
                engine.appendLog("  -> \(res.output)")
            }
        }
    }

    func export() {
        if let u = engine.exportResults() {
            shareURL = u
            showShare = true
        }
    }

    func thermalName(_ t: ProcessInfo.ThermalState) -> String {
        switch t {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

struct HelpView: View {
    @Environment(\.dismiss) var dismiss
    var body: some View {
        NavigationView {
            VStack(alignment: .leading, spacing: 8) {
                Text("1. Copy the GGUF model + mmproj into the app's Documents (Files app / Finder: enable File Sharing, drag files in).").font(.footnote)
                Text("2. Tap 'Load model'. It scans Documents for *.gguf.").font(.footnote)
                Text("3. 'Eval set' runs eval.json + bundled photos; 'Stress 20x' runs 20 sequential queries; 'Soak 10min' loops for thermals.").font(.footnote)
                Text("4. 'Export' writes results.json to Documents.").font(.footnote)
                Spacer()
            }
            .padding()
            .navigationTitle("Help")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .navigationBarTrailing) { Button("Done") { dismiss() } } }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View { ContentView() }
}