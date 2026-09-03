#!/usr/bin/env bash
# fetch_adapters.sh — download the trained LoRA adapters from Hugging Face Hub.
#
# lora/adapters/ is git-ignored (see .gitignore) because it is a build product
# of lora/run_lora.py (final: adapters.safetensors, 35MB, rank 8 / alpha 16,
# 300 iters — AR 38% -> 65% on the 26-item eval). Fresh clones fetch it:
#   ./lora/fetch_adapters.sh
#
# Env:
#   HF_REPO  HF repo id (default: akm85/ordo-qwen2vl-lora)
# Usage: ./lora/fetch_adapters.sh [--force]  (--force re-downloads)
set -euo pipefail

REPO_ID="${HF_REPO:-akm85/ordo-qwen2vl-lora}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/lora/adapters"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

FILES=("adapters.safetensors" "adapter_config.json")

mkdir -p "$OUT"
need=()
for f in "${FILES[@]}"; do
    if [[ $FORCE -eq 0 && -f "$OUT/$f" && -s "$OUT/$f" ]]; then
        echo "skip  $f (exists, use --force to re-download)"
    else
        need+=("$f")
    fi
done
if [[ ${#need[@]} -eq 0 ]]; then
    echo "done -> $OUT"
    exit 0
fi

HF_BIN=""
for c in "$REPO/.venv/bin/hf" hf huggingface-cli; do
    if command -v "$c" >/dev/null 2>&1; then HF_BIN="$c"; break; fi
done

if [[ -n "$HF_BIN" ]]; then
    echo "fetch ${need[*]} from hf:$REPO_ID (via $HF_BIN)"
    for f in "${need[@]}"; do
        "$HF_BIN" download "$REPO_ID" "$f" --local-dir "$OUT"
    done
else
    # No HF CLI: plain HTTPS works for public repos (no token needed).
    for f in "${need[@]}"; do
        echo "fetch $f"
        curl -L --fail --progress-bar -o "$OUT/$f" "https://huggingface.co/$REPO_ID/resolve/main/$f"
    done
fi
echo "done -> $OUT"
ls -lh "$OUT"/adapters.safetensors "$OUT"/adapter_config.json
