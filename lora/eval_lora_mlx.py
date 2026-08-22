#!/usr/bin/env python3
# eval_lora_mlx.py — eval-set AR for base vs LoRA Qwen2-VL-2B (MLX).
import json
import os
import time

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(REPO, "lora", "mlx-model")
ADAPTER = os.path.join(REPO, "lora", "adapters")
EVAL = os.path.join(REPO, "eval", "eval.json")

items = json.load(open(EVAL))["items"]
items = [it for it in items if it.get("ground_truth")]

import sys
sys.path.insert(0, os.path.join(REPO, "eval", "scripts"))
from metrics import answer_recall, word_f1

def ar(pred, truth):
    if not truth:
        return 0.0
    return 1.0 if answer_recall(pred, truth) >= 1.0 else 0.0

def run(tag, adapter_path=None):
    print(f"=== {tag}: loading model (adapter={adapter_path}) ===", flush=True)
    model, processor = load(MODEL, adapter_path=adapter_path)
    pass  # model.eval() breaks mlx-vlm 0.6.15 generation
    rows = []
    for i, it in enumerate(items, 1):
        img = os.path.join(REPO, "eval", it["path"])
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": it['question']},
        ]}]
        prompt = apply_chat_template(processor, model.config, messages, num_images=1)
        t0 = time.time()
        gen = generate(
            model, processor, prompt,
            image=img, max_tokens=96, verbose=False,
        )
        out = gen.text
        dt = time.time() - t0
        score = ar(out, it["ground_truth"])
        rows.append({"id": it["id"], "ar": score, "out": out[:80], "s": round(dt, 1)})
        print(f"[{i}/{len(items)}] {it['id']} AR={score} {dt:.1f}s | {out[:60]!r}", flush=True)
    n = len(rows)
    print(f"=== {tag}: AR {sum(r['ar'] for r in rows)}/{n} = {100*sum(r['ar'] for r in rows)/n:.0f}% ===", flush=True)
    with open(os.path.join(REPO, "results", f"lora_mlx_{tag}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows

if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("base", "both"):
        run("base")
    if which in ("lora", "both"):
        run("lora", ADAPTER)