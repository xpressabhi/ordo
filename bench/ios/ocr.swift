// ocr.swift — OCR a screenshot with Apple Vision. Usage: swift ocr.swift <image>
import Foundation
import Vision
import AppKit

let path = CommandLine.arguments[1]
guard let img = NSImage(contentsOfFile: path),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("cannot load image"); exit(1)
}
let request = VNRecognizeTextRequest { req, _ in
    guard let obs = req.results as? [VNRecognizedTextObservation] else { return }
    for o in obs {
        if let t = o.topCandidates(1).first {
            print(t.string)
        }
    }
}
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
let handler = VNImageRequestHandler(cgImage: cg, options: [:])
try? handler.perform([request])