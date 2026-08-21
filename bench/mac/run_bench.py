#!/usr/bin/env python3
# run_bench.py — run the quantization ladder over the eval set on the Mac.
#
# For each (text quant x mmproj quant) config and each eval image, runs
# bench_vlm --json and appends a row to results/<tag>.csv.
#
# Usage:
#   .venv/bin/python bench/mac/run_bench.py \
#       --text "f16 Q8_0 Q4_K_M Q2_K IQ2_XXS" \
#       --mmproj "f16 Q8_0 Q4_K_M" \
#       --n 96 --tag results [--images img_001 img_002] [--dev]

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "bench", "mac", "bench_vlm")
EVAL = os.path.join(REPO, "eval", "eval.json")
MODEL_DIR = os.path.join(REPO, "model")

TEXT_DEFAULT = ["f16", "Q8_0", "Q4_K_M", "Q2_K", "IQ2_XXS"]
MMPROJ_DEFAULT = ["f16", "Q8_0", "Q4_K_M"]

CSV_HEADER = ["text_q", "mmproj_q", "image", "question", "ground_truth",
              "encode_ms", "prefill_ms", "n_prefill_toks", "decode_ms",
              "n_decode_toks", "ttft_ms", "peak_rss_mb", "model_size_mb",
              "mmproj_size_mb", "output"]


def text_gguf(q):
    return os.path.join(MODEL_DIR, f"Qwen2-VL-2B-Instruct-{q}.gguf")


def mm_gguf(q):
    return os.path.join(MODEL_DIR, f"mmproj-Qwen2-VL-2B-Instruct-{q}.gguf")


def run_one(text_q, mm_q, item, n_predict):
    cmd = [BENCH, "-m", text_gguf(text_q), "-mm", mm_gguf(mm_q),
           "--image", os.path.join(REPO, "eval", item["path"]),
           "-p", item["question"], "-n", str(n_predict), "--json"]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    dt = time.time() - t0
    if proc.returncode != 0:
        return {"text_q": text_q, "mmproj_q": mm_q, "image": item["id"],
                "question": item["question"], "ground_truth": item["ground_truth"],
                "error": proc.stderr[-300:], "wall_s": round(dt, 1)}
    try:
        j = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"text_q": text_q, "mmproj_q": mm_q, "image": item["id"],
                "question": item["question"], "ground_truth": item["ground_truth"],
                "error": "bad json: " + proc.stdout[-200:], "wall_s": round(dt, 1)}
    j["text_q"] = text_q
    j["mmproj_q"] = mm_q
    j["image"] = item["id"]
    j["question"] = item["question"]
    j["ground_truth"] = item["ground_truth"]
    j["wall_s"] = round(dt, 1)
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=" ".join(TEXT_DEFAULT), help="space-separated text quants")
    ap.add_argument("--mmproj", default=" ".join(MMPROJ_DEFAULT), help="space-separated mmproj quants")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--tag", default=f"results_{datetime.now().strftime('%Y%m%d_%H%M')}")
    ap.add_argument("--images", nargs="*", help="subset of image ids")
    ap.add_argument("--dev", action="store_true", help="run only 2 images (smoke test)")
    args = ap.parse_args()

    with open(EVAL) as f:
        items = json.load(f)["items"]
    if args.images:
        items = [i for i in items if i["id"] in args.images]
    if args.dev:
        items = items[:2]

    text_qs = args.text.split()
    mm_qs = args.mmproj.split()

    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    out_path = os.path.join(REPO, "results", f"{args.tag}.csv")
    rows = []
    total = len(text_qs) * len(mm_qs) * len(items)
    done = 0
    for tq in text_qs:
        for mq in mm_qs:
            for item in items:
                done += 1
                row = run_one(tq, mq, item, args.n)
                rows.append(row)
                if "error" in row:
                    print(f"[{done}/{total}] {tq}x{mq} {item['id']} ERROR: {row['error'][:80]}")
                else:
                    print(f"[{done}/{total}] {tq}x{mq} {item['id']} "
                          f"enc={row['encode_ms']}ms pre={row['prefill_ms']}ms "
                          f"dec={row['decode_ms']}ms ttft={row['ttft_ms']}ms | {row['output'][:40]!r}")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER + ["error", "wall_s"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_HEADER + ["error", "wall_s"]})
    print(f"\nwrote -> {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
