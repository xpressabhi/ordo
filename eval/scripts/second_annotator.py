# second_annotator.py — blind re-answer pass for inter-annotator agreement.
#
# Re-asks every question from eval.json WITHOUT showing the ground truth, so a
# second (independent) person can answer. Reports exact-match and word-F1
# agreement with the ground truth. This is the empirical "is the eval set any
# good?" check: if two humans disagree with the ground truth, the item is
# ambiguous and shouldn't be scored strictly.
#
# Usage: .venv/bin/python eval/scripts/second_annotator.py [--file out.json]
#   Answers can be piped in (one per line) for automation.

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(REPO, "eval", "eval.json")


def normalize(s):
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9 .$-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def em(a, b):
    return normalize(a) == normalize(b)


def f1(a, b):
    wa = set(normalize(a).split())
    wb = set(normalize(b).split())
    if not wa or not wb:
        return 1.0 if wa == wb else 0.0
    inter = len(wa & wb)
    p = inter / len(wb)
    r = inter / len(wa)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    out_file = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("--file=") else None
    if out_file:
        out_file = out_file.split("=", 1)[1]
    with open(MANIFEST) as f:
        m = json.load(f)
    items = m["items"]
    results = []
    print(f"{len(items)} questions, answering blindly (no ground truth shown).")
    for it in items:
        try:
            ans = input(f"[{it['id']}] {it['question']}\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        results.append({"id": it["id"], "annotator_answer": ans})
    n = len(results)
    ems = sum(em(r["annotator_answer"], next(x["ground_truth"] for x in items if x["id"] == r["id"])) for r in results)
    f1s = sum(f1(r["annotator_answer"], next(x["ground_truth"] for x in items if x["id"] == r["id"])) for r in results)
    print(f"\nAgreement over {n} items:")
    print(f"  exact match: {ems} ({100*ems/max(1,n):.0f}%)")
    print(f"  word F1    : {100*f1s/max(1,n):.0f}%")
    if out_file:
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved -> {out_file}")


if __name__ == "__main__":
    main()
