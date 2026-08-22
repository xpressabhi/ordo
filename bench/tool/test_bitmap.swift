// test_bitmap.swift — replicate the app's UIKit bitmap path on macOS and dump a PNG.
// Usage: swift test_bitmap.swift <input.jpg> <output.png>
import Foundation
import AppKit
import CoreGraphics

let input = CommandLine.arguments[1]
let output = CommandLine.arguments[2]
let noFlip = CommandLine.arguments.contains("--noflip")
guard let img = NSImage(contentsOfFile: input), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("cannot load"); exit(1)
}
let cw = cg.width
let ch = cg.height
let bytesPerRow = cw * 4
var rgba = [UInt8](repeating: 0, count: ch * bytesPerRow)
guard let ctx = CGContext(data: &rgba, width: cw, height: ch,
                          bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                          space: CGColorSpaceCreateDeviceRGB(),
                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
    print("CGContext nil"); exit(1)
}
if !noFlip {
    ctx.translateBy(x: 0, y: CGFloat(ch))
    ctx.scaleBy(x: 1, y: -1)
}
ctx.interpolationQuality = .high
ctx.draw(cg, in: CGRect(x: 0, y: 0, width: CGFloat(cw), height: CGFloat(ch)))

// compact to RGB like the app does
var rgb = [UInt8](repeating: 0, count: cw * ch * 3)
var s = 0, d = 0
while s < rgba.count {
    rgb[d] = rgba[s]; rgb[d+1] = rgba[s+1]; rgb[d+2] = rgba[s+2]
    s += 4; d += 3
}
// dump stats
var nonZero = 0
for i in 0..<rgb.count where rgb[i] > 16 { nonZero += 1 }
print("size \(cw)x\(ch), nonzero-bytes \(nonZero)/\(rgb.count) (\(nonZero*100/rgb.count)%)")

// save via CGImage from RGBA (same buffer the app feeds mtmd after compaction)
let dataProvider = CGDataProvider(data: Data(rgba) as CFData)!
let cgImage = CGImage(width: cw, height: ch, bitsPerComponent: 8, bitsPerPixel: 32,
                      bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                      provider: dataProvider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)!
let rep = NSBitmapImageRep(cgImage: cgImage)
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: output))
print("wrote \(output)")