#!/usr/bin/env bash
# run_bench.sh — run the quantization ladder over the eval set on the Mac.
#
# For each (text quant x mmproj quant) config and each eval image, runs
# bench_vlm --json and appends a row to results/<tag>.csv
#
# Usage:
#   ./bench/mac/run_bench.sh --text "f16 Q8_0 Q4_K_M Q2_K IQ2_XXS" \
#                            --mmproj "f16 Q8_0 Q4_K_M" \
#                            --n 96 [--images img_001 img_002] [--tag results]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="$REPO/bench/mac/bench_vlm"
EVAL="$REPO/eval/eval.json"
MODEL_DIR="$REPO/model"

TEXT_DEFAULT="f16 Q8_0 Q4_K_M Q2_K IQ2_XXS"
MMPROJ_DEFAULT="f16 Q8_0 Q4_K_M"
TEXT=() MMPROJ=() N_PREDICT=96 TAG="results" IMAGES=()

# ---- arg parsing -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --text)   shift; TEXT=($1) ;;
        --mmproj) shift; MMPROJ=($1) ;;
        --n)      shift; N_PREDICT="$1" ;;
        --tag)    shift; TAG="$1" ;;
        --images) shift; IMAGES=($1) ;;
        *) echo "unknown: $1"; exit 1 ;;
    esac
    shift
done
[[ ${#TEXT[@]} -eq 0 ]] && TEXT=($TEXT_DEFAULT)
[[ ${#MMPROJ[@]} -eq 0 ]] && MMPROJ=($MMPROJ_DEFAULT)

OUT="$REPO/results/${TAG}.csv"
mkdir -p "$(dirname "$OUT")"
: > "$OUT"
echo "text_q,mmproj_q,image,question,ground_truth,encode_ms,prefill_ms,n_prefill_toks,decode_ms,n_decode_toks,ttft_ms,peak_rss_mb,model_size_mb,mmproj_size_mb,output" > "$OUT"

# ---- helper: resolve gguf path for a quant name ----------------------------
resolve_text() { echo "$MODEL_DIR/Qwen2-VL-2B-Instruct-$1.gguf"; }
resolve_mm()   { echo "$MODEL_DIR/mmproj-Qwen2-VL-2B-Instruct-$1.gguf"; }

# ---- iterate ----------------------------------------------------------------
for tq in "${TEXT[@]}"; do
    for mq in "${MMPROJ[@]}"; do
        echo "=== text $tq x mmproj $mq ==="
        for img in "${IMAGES[@]}"; do :; done
        # gather image ids from eval.json
        python3 - "$EVAL" "${IMAGES[@]}" << 'EOF'
import json, sys
with open(sys.argv[1]) as f: m = json.load(f)
ids = set(sys.argv[2:]) if len(sys.argv) > 2 else None
for it in m["items"]:
    if ids is None or it["id"] in ids:
        print(f"{it['id']}\t{it['path']}\t{it['question']}\t{it['ground_truth']}")
EOF
    done
done
