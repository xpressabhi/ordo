#!/usr/bin/env python3
# decompose.py — attribute TTFT between the vision encoder and text prefill.
#
# Answers the assignment's question: "A 500-token prompt and one image. Which
# contributes more to time-to-first-token?"
#
# Runs 4 configs:
#   no-image x short prompt   -> baseline decode/prefill at N tokens
#   no-image x 500-tok prompt -> how much does prompt length alone add?
#   image    x short prompt   -> adds the vision encoder
#   image    x 500-tok prompt -> both
#
# TTFT = encode (if image) + prefill. The delta between image-short and
# no-image-short is the encoder's contribution; the delta between 500-tok and
# short (same image presence) is the prefill's contribution.
#
# Usage: .venv/bin/python bench/mac/decompose.py [--n 500] [--out results/ttft_decompose.csv]

import argparse
import csv
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "bench", "mac", "bench_vlm")
MODEL = os.path.join(REPO, "model", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf")
MMPROJ = os.path.join(REPO, "model", "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf")
IMAGE = os.path.join(REPO, "bench", "smoke", "menu_test.png")

LONG_PROMPT = ("What is the total of an espresso, a cappuccino and two croissants? "
               "Show your working out step by step, and then state the final total. " * 18)


def run(image, prompt, n):
    cmd = [BENCH, "-m", MODEL]
    if image:
        cmd += ["-mm", MMPROJ, "--image", image]
    else:
        cmd += ["--no-image"]
    cmd += ["-p", prompt, "-n", str(n), "--json"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return json.loads(p.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "ttft_decompose.csv"))
    args = ap.parse_args()

    short = "What is the price of a matcha?"
    rows = []
    cases = [
        ("no-image", "short",  None, short),
        ("no-image", "long",   None, LONG_PROMPT),
        ("image",    "short",  IMAGE, short),
        ("image",    "long",   IMAGE, LONG_PROMPT),
    ]
    for img_label, prompt_label, img, prompt in cases:
        r = run(img, prompt, args.n)
        row = {"media": img_label, "prompt": prompt_label, "prompt_len": len(prompt.split()),
               "encode_ms": r["encode_ms"], "prefill_ms": r["prefill_ms"],
               "n_prefill_toks": r["n_prefill_toks"], "decode_ms": r["decode_ms"],
               "n_decode_toks": r["n_decode_toks"], "ttft_ms": r["ttft_ms"]}
        rows.append(row)
        print(f"{img_label:9s} {prompt_label:6s} prompt_tokens={row['n_prefill_toks']:4d} "
              f"encode={row['encode_ms']:5d}ms prefill={row['prefill_ms']:5d}ms ttft={row['ttft_ms']:5d}ms")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote -> {args.out}")

    enc_contrib = rows[2]["ttft_ms"] - rows[0]["ttft_ms"]
    pre_contrib_long = (rows[1]["ttft_ms"] - rows[0]["ttft_ms"]) + (rows[3]["ttft_ms"] - rows[2]["ttft_ms"])
    print(f"\nimage encoder adds ~{enc_contrib}ms to TTFT")
    print(f"long-vs-short prompt adds ~{pre_contrib_long}ms to TTFT (across image and no-image)")


if __name__ == "__main__":
    main()
