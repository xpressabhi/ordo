# Ordo — take-home: a vision model on a phone

Qwen2-VL-2B on llama.cpp, measured across a quantization ladder on a Mac
(M2 Pro) and deployed to an iPhone (15 / A16 Bionic), on a private eval set of 18
real-world text photos.

**Full write-up: `docs/report.md`** — this README is the summary.

## The short version

| | |
|---|---|
| Model | Qwen2-VL-2B-Instruct (GGUF, llama.cpp b10549, Metal) |
| Eval | 18 own photos: product labels, book spines, handwriting, appliance displays (6 easy / 6 medium / 6 hard) |
| Best config | **Q4_K_M text + Q8_0 mmproj — 1.6GB, 81% answer-recall, 2.3GB peak RAM** |
| Headline finding | **The vision encoder IS the latency problem**: ~26s of a ~30s TTFT is image encode; a 600-token prompt costs ~0.6s |
| Second finding | The encoder tolerates quantization almost completely (mmproj f16 vs Q4: identical accuracy); the decoder is where quant bites |
| iPhone status | **Both models run the eval on-device**: Qwen2-VL-2B @768px → 46% AR, TTFT 5.8s, RSS 1.1GB; SmolVLM-256M → sub-second TTFT, 370MB. The A16 tower collapse is resolution-dependent (full-res broken, 768px fine — §13) |

## Reproduce

```
./model/fetch.sh --all          # weights (ungated)
./bench/tool/build_bench.sh     # timing harness (needs Xcode toolchain)
.venv/bin/python bench/mac/run_bench.py --tag results   # ladder
.venv/bin/python bench/mac/report.py results/results_<tag>.csv
.venv/bin/python bench/mac/decompose.py                  # TTFT attribution
```

## Layout

```
model/           quantization ladder: f16 Q8 Q4 Q2_K IQ2_XXS + mmproj f16/Q8/Q4
bench/tool/      bench_vlm.cpp (encode/prefill/decode/TTFT + RSS), test_multi, ocr.swift
bench/mac/       run_bench.py, report.py, decompose.py
bench/ios/       QwenBench Xcode app (on-device harness)
eval/            eval.json manifest (DRAFT ground truths) + capture/validate scripts
lora/            MLX LoRA pipeline (dataset builder / trainer / exporter)
results/         ladder_full.csv (270 runs), ttft_decompose.csv
docs/report.md   the full report (findings, failure analysis, recommendation)
```

## Status

- [x] Part 1 — runtime, weights, ladder produced (f16/Q8/Q4/Q2/IQ2 + mmproj f16/Q8/Q4); shipped-app story in report §1
- [x] Part 2 — eval set (18 photos, 6/6/6 spread, DRAFT answers — **verify `eval/eval.json`**)
- [x] Part 3 — full 270-run ladder on Mac + failure analysis (results/ladder_full.csv)
- [x] TTFT decomposition (image vs prompt)
- [x] Part 4 — 20-query stress (fall-off 13ms), 5-min thermal soak (nominal→serious, TTFT +16%), memory flat 354-362MB. Battery not measurable (API -1.0; 0.0% drain over 6 min by manual readout)
- [x] On-device eval with SmolVLM-256M — TTFT 805ms median, AR 39%, 18/18 answered (results/device_eval_smolvlm_18.jsonl)
- [x] LoRA fine-tune — AR 38% -> 65% (+27pts) on the eval task (MLX, 300 iters)
- [x] On-device resolution sweep (512/768/1024) + size ladder (256M/500M/Qwen2-VL-2B)
- [x] Speed levers: Q4 mmproj + n_batch 4096 + short-answer (wall -68%, RSS -28%)
- [ ] Human verification of the 26 ground truths in eval/eval.json (the one remaining task; shifts all AR numbers)

## The on-device vision bug (what failed — and the workaround that works)

On the iPhone (A16), Qwen2-VL-2B's vision tower emits near-constant
embeddings (values ≈ {-1.79, 1.0}); the model generates confidently but
hallucinates ("The Netherlands" for a puzzle-label photo). Survived: 3
framework builds (2 hand-merged, 1 official xcframework), mmproj Q8/Q4/f16,
FA on/off, KV/batch sizes, image sizes, GPU/CPU decode. Same llama.cpp +
weights work on the M2.

**The workaround: SmolVLM-256M runs perfectly on the same phone** — 18-photo
eval in ~110s, sub-second TTFT, real embeddings. The bug is specific to
Qwen2-VL's 2D-mrope vision kernels on A16-class Metal, not llama.cpp or our
code. Full story in `docs/report.md` §8-§9.