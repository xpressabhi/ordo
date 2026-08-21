#!/usr/bin/env python3
# report.py — turn results/<tag>.csv into accuracy tables + failure analysis.
#
# For each (text_q x mmproj_q) config computes:
#   - exact match (EM) and word-F1 over the eval set
#   - EM/F1 split by difficulty bucket and by condition tag
#   - median encode/prefill/decode/TTFT + peak RAM
# Then prints a failure table: which images drop from a higher quant to a lower
# one (the "where does it break first?" analysis).
#
# Usage: .venv/bin/python bench/mac/report.py results/<tag>.csv

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "eval", "scripts"))
from metrics import exact_match, word_f1, answer_recall

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.path.join(REPO, "eval", "eval.json")


def load_manifest():
    with open(EVAL) as f:
        return json.load(f)["items"]


def main():
    if len(sys.argv) < 2:
        print("usage: report.py <results.csv>")
        return 1
    csv_path = sys.argv[1]
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    items = {i["id"]: i for i in load_manifest()}

    by_cfg = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        by_cfg[(r["text_q"], r["mmproj_q"])].append(r)

    print(f"{'config':28s} {'AR':>6s} {'EM':>5s} {'F1':>6s} {'enc':>6s} {'pre':>6s} {'dec':>6s} {'TTFT':>6s} {'RSS':>6s}")
    print("-" * 88)
    cfgs = sorted(by_cfg.keys())
    for tq, mq in cfgs:
        rs = by_cfg[(tq, mq)]
        n = len(rs)
        ar = sum(answer_recall(r["output"], r["ground_truth"]) for r in rs) / n
        em = sum(exact_match(r["output"], r["ground_truth"]) for r in rs) / n
        f1 = sum(word_f1(r["output"], r["ground_truth"]) for r in rs) / n
        enc = sorted(int(r["encode_ms"]) for r in rs)[n // 2]
        pre = sorted(int(r["prefill_ms"]) for r in rs)[n // 2]
        dec = sorted(int(r["decode_ms"]) for r in rs)[n // 2]
        ttft = sorted(int(r["ttft_ms"]) for r in rs)[n // 2]
        rss = sorted(int(r["peak_rss_mb"]) for r in rs)[n // 2]
        print(f"{(tq+'x'+mq):28s} {ar*100:4.0f}% {em*100:4.0f}% {f1*100:5.0f}% {enc:5d} {pre:5d} {dec:5d} {ttft:5d} {rss:5d}")

    print("\n== Difficulty breakdown (F1 by bucket) ==")
    print(f"{'config':28s} {'easy':>6s} {'med':>6s} {'hard':>6s}")
    for tq, mq in cfgs:
        rs = by_cfg[(tq, mq)]
        by_d = defaultdict(list)
        for r in rs:
            by_d[items[r["image"]]["difficulty"]].append(r)
        parts = []
        for d in ("easy", "medium", "hard"):
            sub = by_d.get(d, [])
            parts.append(f"{sum(answer_recall(r['output'], r['ground_truth']) for r in sub)/max(1,len(sub))*100:5.0f}%")
        print(f"{(tq+'x'+mq):28s} {'  '.join(parts)}")

    print("\n== Failure analysis: items that pass at a higher quant, fail lower ==")
    # group by image; track max-quant F1==1.0 and min-quant F1<1.0
    img_max = defaultdict(dict)
    for tq, mq in cfgs:
        for r in by_cfg[(tq, mq)]:
            img_max[r["image"]][tq] = max(img_max[r["image"]].get(tq, 0.0), answer_recall(r["output"], r["ground_truth"]))
    # order: f16, Q8_0, Q4_K_M, Q2_K, IQ2_XXS
    order = ["f16", "Q8_0", "Q4_K_M", "Q2_K", "IQ2_XXS"]
    for img, quants in sorted(img_max.items()):
        fails = [q for q in order if q in quants and quants[q] < 0.5]
        passes = [q for q in order if q in quants and quants[q] >= 0.5]
        if fails and passes:
            it = items[img]
            print(f"  {img} [{it['difficulty']}/{','.join(it.get('conditions',['clean']))}] "
                  f"passes@{','.join(passes)} fails@{','.join(fails)}  gt={it['ground_truth']!r}")

    print("\n== Condition breakdown (F1 by condition tag) ==")
    conds = set()
    for it in items.values():
        conds.update(it.get("conditions", ["clean"]))
    for tq, mq in cfgs[:1]:
        rs = by_cfg[(tq, mq)]
        for c in sorted(conds):
            sub = [r for r in rs if c in items[r["image"]].get("conditions", ["clean"])]
            if sub:
                ar = sum(answer_recall(r["output"], r["ground_truth"]) for r in sub) / len(sub)
                print(f"  {c:20s} n={len(sub):2d} AR={ar*100:5.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
