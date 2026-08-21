#!/usr/bin/env bash
# build_bench.sh — compile bench_vlm.cpp against the local llama.cpp build tree.
# Uses the same llama.cpp sources + Metal backend that the iOS app will use.
#
# Usage: ./bench/tool/build_bench.sh [llama.cpp build dir]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA="${1:-$REPO/.deps/llama.cpp}"
BUILD="$LLAMA/build"
OUT="$REPO/bench/mac/bench_vlm"

# Xcode toolchain (system CLT libc++ is broken on this machine; Xcode SDK works)
XC=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk

mkdir -p "$REPO/bench/mac"

"$XC/clang++" -O2 -std=c++17 \
    -isysroot "$SDK" \
    -I"$LLAMA/include" \
    -I"$LLAMA/ggml/include" \
    -I"$LLAMA/tools/mtmd" \
    "$REPO/bench/tool/bench_vlm.cpp" \
    -L"$BUILD/bin" \
    -lmtmd -lllama -lggml -lggml-base -lggml-cpu -lggml-metal \
    -Wl,-rpath,"$BUILD/bin" \
    -framework Metal -framework MetalKit -framework Foundation -framework QuartzCore \
    -o "$OUT"

echo "built: $OUT"
