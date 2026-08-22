#!/usr/bin/env python3
# draft_manifest.py — model-assisted draft of eval.json.
# Runs each photo through the model asking for ALL visible text + scene
# description, then writes eval/draft_items.json with suggested
# question/ground_truth entries for the human to correct.
#
# The draft is NOT the gold standard — a human must verify each ground truth
# against the photo (eval set integrity).
#
# Usage: .venv/bin/python eval/scripts/draft_manifest.py [--n 64]

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "bench", "mac", "bench_vlm")
PHOTOS = os.path.join(REPO, "eval", "photos")
OUT = os.path.join(REPO, "eval", "draft_items.json")

MODEL = os.path.join(REPO, "model", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf")
MMPROJ = os.path.join(REPO, "model", "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf")

PROMPT = ("Read and transcribe ALL text visible in this image, accurately, "
          "including prices, names, dates and times. Then in one sentence "
          "describe what the object or scene is.")

def main():
    n_tokens = 96
    if len(sys.argv) > 1 and sys.argv[1] == "--n":
        n_tokens = int(sys.argv[2])
    photos = sorted(f for f in os.listdir(PHOTOS) if f.endswith(".jpg"))
    # skip smoke files
    photos = [f for f in photos if not f.startswith("smoke_")]
    print(f"{len(photos)} photos to scan")
    drafts = []
    for i, fn in enumerate(photos, 1):
        cmd = [BENCH, "-m", MODEL, "-mm", MMPROJ, "--image",
               os.path.join(PHOTOS, fn), "-p", PROMPT, "-n", str(n_tokens), "--json"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        out = ""
        if p.returncode == 0:
            try:
                out = json.loads(p.stdout).get("output", "")
            except Exception:
                pass
        drafts.append({"file": fn, "transcript": out})
        print(f"[{i}/{len(photos)}] {fn}: {out[:80]!r}")
    with open(OUT, "w") as f:
        json.dump(drafts, f, indent=2, ensure_ascii=False)
    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()