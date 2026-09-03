#!/usr/bin/env bash
# fetch.sh — download GGUF weights (text + vision encoder) from ggml-org
# (llama.cpp's official HF org). Not gated: no token needed.
# Usage: ./model/fetch.sh [--all] [--smol]
#   default: Qwen Q4_K_M + mmproj Q8_0 (the minimal working set)
#   --all:   Qwen f16, Q8_0, Q4_K_M text + f16, Q8_0 mmproj + SmolVLM Q8 pairs
#   --smol:  SmolVLM-256M/500M Q8_0 text + mmproj (on-device eval) only
# Note: Qwen Q2_K, IQ2_XXS and mmproj-Q4_K_M are NOT upstream — they were
# quantized locally with llama-quantize (IQ2_XXS via an imatrix); re-run the
# quantize to reproduce them.
set -euo pipefail

BASE="https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main"
SMOL256="https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main"
SMOL500="https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main"
OUT="$(cd "$(dirname "$0")/.." && pwd)/model"

fetch() { # $1 = base URL, $2 = filename
    if [[ -f "$OUT/$2" && -s "$OUT/$2" ]]; then
        echo "skip  $2 (exists)"
        return
    fi
    echo "fetch $2"
    curl -L --fail --progress-bar -o "$OUT/$2" "$1/$2"
}

TEXT_Q4="Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
TEXT_Q8="Qwen2-VL-2B-Instruct-Q8_0.gguf"
TEXT_F16="Qwen2-VL-2B-Instruct-f16.gguf"
MM_Q8="mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"
MM_F16="mmproj-Qwen2-VL-2B-Instruct-f16.gguf"
SMOL_FILES=(
    "$SMOL256|SmolVLM-256M-Instruct-Q8_0.gguf"
    "$SMOL256|mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
    "$SMOL500|SmolVLM-500M-Instruct-Q8_0.gguf"
    "$SMOL500|mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"
)

fetch_smol() {
    for pair in "${SMOL_FILES[@]}"; do
        fetch "${pair%%|*}" "${pair##*|}"
    done
}

mkdir -p "$OUT"
case "${1:-}" in
    --smol) fetch_smol ;;
    --all)
        for f in "$TEXT_Q4" "$TEXT_Q8" "$TEXT_F16" "$MM_Q8" "$MM_F16"; do
            fetch "$BASE" "$f"
        done
        fetch_smol
        ;;
    *) fetch "$BASE" "$TEXT_Q4"; fetch "$BASE" "$MM_Q8" ;;
esac
echo "done -> $OUT"
ls -lh "$OUT"/*.gguf 2>/dev/null || true
