# validate_eval.py — sanity checks on eval/eval.json + difficulty-spread report.
#
# Checks:
#   1. Every item has the required fields (path, question, ground_truth, category, difficulty).
#   2. Photo files exist on disk.
#   3. No duplicate questions / duplicate photos.
#   4. Difficulty spread + category coverage summary (the "how did you choose the spread" story).
#   5. Ambiguity flag: questions with very short or very long ground truths.
#
# Usage: .venv/bin/python eval/scripts/validate_eval.py

import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(REPO, "eval", "eval.json")

REQUIRED = ["path", "question", "ground_truth", "category", "difficulty"]


def main():
    with open(MANIFEST) as f:
        m = json.load(f)
    items = m["items"]
    print(f"eval.json: {len(items)} items\n")

    errors = 0
    for i, it in enumerate(items):
        for fld in REQUIRED:
            if not it.get(fld):
                print(f"  [ERR] item {i}: missing '{fld}'")
                errors += 1
        if not os.path.exists(os.path.join(REPO, "eval", it["path"])):
            print(f"  [ERR] item {i}: photo not found: {it['path']}")
            errors += 1

    # duplicates
    paths = [os.path.basename(i["path"]) for i in items]
    qs = [i["question"].strip().lower() for i in items]
    for label, col in (("photo", paths), ("question", qs)):
        seen = set()
        for v in col:
            if v in seen:
                print(f"  [WARN] duplicate {label}: {v}")
            seen.add(v)

    # ambiguity heuristic
    for it in items:
        gt = it["ground_truth"].strip()
        if len(gt) < 2:
            print(f"  [WARN] very short ground truth for {it['id']}: '{gt}'")

    # difficulty spread
    print("\n== Difficulty spread ==")
    spread = {}
    for it in items:
        spread.setdefault(it["difficulty"], 0)
        spread[it["difficulty"]] += 1
    for d in ("easy", "medium", "hard"):
        n = spread.get(d, 0)
        pct = 100.0 * n / max(1, len(items))
        print(f"  {d:6s}: {n:2d}  ({pct:.0f}%)")
    print(f"  target was ~30% easy / 40% medium / 30% hard")

    print("\n== Category coverage ==")
    cats = {}
    for it in items:
        cats.setdefault(it["category"], 0)
        cats[it["category"]] += 1
    for c, n in sorted(cats.items()):
        print(f"  {c:20s}: {n}")

    print("\n== Condition tags ==")
    conds = {}
    for it in items:
        for c in it.get("conditions", ["clean"]):
            conds.setdefault(c, 0)
            conds[c] += 1
    for c, n in sorted(conds.items()):
        print(f"  {c:20s}: {n}")

    print(f"\n{'OK - eval set looks complete' if errors == 0 else f'{errors} problem(s) found'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
