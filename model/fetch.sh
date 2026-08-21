#!/usr/bin/env bash
# fetch.sh — download Qwen2-VL-2B GGUF weights (text + vision encoder) from
# ggml-org (llama.cpp's official HF org). Not gated: no token needed.
# Usage: ./model/fetch.sh [--all]
#   default: Q4_K_M + mmproj Q8_0 (the minimal working set)
#   --all:   f16, Q8_0, Q4_K_M text + f16, Q8_0 mmproj
set -euo pipefail

BASE="https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main"
OUT="$(cd "$(dirname "$0")/.." && pwd)/model"

TEXT_Q4="Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
TEXT_Q8="Qwen2-VL-2B-Instruct-Q8_0.gguf"
TEXT_F16="Qwen2-VL-2B-Instruct-f16.gguf"
MM_Q8="mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"
MM_F16="mmproj-Qwen2-VL-2B-Instruct-f16.gguf"

FILES=("$TEXT_Q4" "$MM_Q8")
if [[ "${1:-}" == "--all" ]]; then
    FILES=("$TEXT_Q4" "$TEXT_Q8" "$TEXT_F16" "$MM_Q8" "$MM_F16")
fi

mkdir -p "$OUT"
for f in "${FILES[@]}"; do
    if [[ -f "$OUT/$f" && -s "$OUT/$f" ]]; then
        echo "skip  $f (exists)"
        continue
    fi
    echo "fetch $f"
    curl -L --fail --progress-bar -o "$OUT/$f" "$BASE/$f"
done
echo "done -> $OUT"
ls -lh "$OUT"/*.gguf 2>/dev/null || true
