#!/usr/bin/env python3
# build_dataset.py — build MLX-LoRA train/val JSONL from eval/eval.json.
#
# Each eval item becomes one training example in the model's own style:
#   {"image": "eval/photos/x.jpg", "prompt": "<question>", "answer": "<ground_truth>"}
# Light question paraphrasing doubles the set so the LoRA sees some variety
# instead of memorizing the exact prompt strings.
#
# Usage: .venv/bin/python lora/build_dataset.py [--augment 2] [--val 0.2]

import argparse
import json
import os
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(REPO, "eval", "eval.json")
OUT_DIR = os.path.join(REPO, "lora", "data")

PARAPHRASES = {
    "what": ["What is", "What's", "What does the text say about"],
    "how much": ["How much", "What is the price of", "What does it cost"],
    "when": ["When", "At what time"],
    "where": ["Where"],
    "who": ["Who"],
}

def paraphrase(q):
    ql = q.lower()
    for key, starters in PARAPHRASES.items():
        for s in starters:
            if ql.startswith(s.lower()) and len(starters) > 1:
                alt = random.choice([x for x in starters if x.lower() != ql[:len(s)].lower()])
                # replace the leading phrase
                return alt + q[len(s):]
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", type=int, default=2, help="paraphrase count per item (0 = no augmentation)")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    with open(EVAL) as f:
        items = json.load(f)["items"]

    rows = []
    for it in items:
        img_rel = it["path"]  # "photos/x.jpg"
        img_abs = os.path.join(REPO, "eval", img_rel)
        base = {"image": img_abs, "prompt": it["question"], "answer": it["ground_truth"]}
        rows.append(base)
        for _ in range(args.augment):
            pq = paraphrase(it["question"])
            if pq != it["question"]:
                rows.append({"image": img_abs, "prompt": pq, "answer": it["ground_truth"]})

    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "train.jsonl"), "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT_DIR, "val.jsonl"), "w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")
    print(f"train: {len(train_rows)} rows, val: {len(val_rows)} rows -> {OUT_DIR}")


if __name__ == "__main__":
    main()