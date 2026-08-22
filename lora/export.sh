#!/usr/bin/env bash
# export.sh — DRAFT: fuse LoRA adapters and produce a GGUF for the ladder.
#
# The MLX -> HF-safetensors -> GGUF seam is the risky part; finalized
# empirically at P6. Options:
#   A) mlx_vlm.lora --export -> MLX fused dir -> llama.cpp convert_hf_to_gguf.py
#      (tensor names usually survive; verify with llama-gguf-hash + a smoke run)
#   B) fallback: transformers PEFT on the HF model, then convert_hf_to_gguf.py
#      (guaranteed layout; slower on MPS)
#
# Usage: ./lora/export.sh [--quant Q4_K_M]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
MLX_MODEL="$REPO/lora/mlx-model"
ADAPTERS="$REPO/lora/adapters"
MERGED="$REPO/lora/merged"
QUANT="${QUANT:-Q4_K_M}"

echo "== fusing adapters into model =="
"$PY" -m mlx_vlm.lora \
    --model "$MLX_MODEL" \
    --adapter-path "$ADAPTERS" \
    --export \
    --export-path "$MERGED"

echo "== GGUF conversion + quantize (path A) =="
CONVERT="$REPO/.deps/llama.cpp/convert_hf_to_gguf.py"
"$PY" "$CONVERT" "$MERGED" --outfile "$REPO/model/qwen2vl-lora-f16.gguf"
"$REPO/.deps/llama.cpp/build/bin/llama-quantize" \
    "$REPO/model/qwen2vl-lora-f16.gguf" \
    "$REPO/model/qwen2vl-lora-${QUANT}.gguf" "$QUANT"

echo "== done: model/qwen2vl-lora-${QUANT}.gguf (evaluate via run_bench.py --text qwen2vl-lora-${QUANT})"