#!/usr/bin/env bash
# train.sh — LoRA fine-tune Qwen2-VL-2B on the eval-style dataset with MLX.
#
# Needs: HF token (gated Qwen2-VL-2B-Instruct) + license accepted.
#   export HF_TOKEN=hf_...
#
# Steps:
#   1. convert HF safetensors -> MLX format (fp16)
#   2. mlx-vlm LoRA training (--train-vision: train vision + language adapters)
#   3. outputs adapters into lora/adapters/
#
# Usage: HF_TOKEN=... ./lora/train.sh [--iters 300] [--full-finetune]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
HF_MODEL="Qwen/Qwen2-VL-2B-Instruct"
MLX_MODEL="$REPO/lora/mlx-model"
DATA="$REPO/lora/data"
ADAPTERS="$REPO/lora/adapters"

ITERS=300
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iters) shift; ITERS="$1" ;;
        --full-finetune) EXTRA+=(--full-finetune) ;;
        --train-vision) EXTRA+=(--train-vision) ;;
        *) echo "unknown: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN not set. Export it:  export HF_TOKEN=hf_..."
    exit 1
fi

if ! "$PY" -c "import mlx_vlm" 2>/dev/null; then
    "$PY" -m pip install --quiet mlx-vlm
fi

# 1. convert
if [[ ! -d "$MLX_MODEL" ]]; then
    echo "== converting $HF_MODEL -> MLX =="
    HF_TOKEN="$HF_TOKEN" "$PY" -m mlx_vlm.convert --hf-path "$HF_MODEL" -q --mlx-path "$MLX_MODEL" -d "$HF_TOKEN"
fi

# 2. train
echo "== LoRA training (iters=$ITERS) =="
mkdir -p "$ADAPTERS"
"$PY" -m mlx_vlm.lora \
    --model "$MLX_MODEL" \
    --train \
    --dataset "$DATA/train.jsonl" \
    --data-config '{"split":"train"}' \
    --val-dataset "$DATA/val.jsonl" \
    --val-data-config '{"split":"val"}' \
    --iters "$ITERS" \
    --batch-size 2 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --steps-per-save 100 \
    --save-every 100 \
    --adapter-path "$ADAPTERS" \
    "${EXTRA[@]}"

echo "== done. adapters -> $ADAPTERS =="