# Ordo — take-home: a vision model on a phone

Qwen2-VL-2B on llama.cpp, measured across a quantization ladder on a Mac
(M2 Pro) and deployed to an iPhone (14 / A15), on a private eval set of 18
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
| iPhone status | App runs, model runs, **text-only works**; vision tower silently broken on A15 (near-constant embeddings — documented, not solved) |

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

- [x] Part 1 — runtime, weights, ladder produced (f16/Q8/Q4/Q2/IQ2 + mmproj f16/Q8/Q4)
- [x] Part 2 — eval set (18 photos, DRAFT answers — **verify `eval/eval.json`**)
- [x] Part 3 — full 270-run ladder on Mac + failure analysis (results/ladder_full.csv)
- [x] TTFT decomposition (image vs prompt)
- [ ] Part 4 — stress/thermal/battery on iPhone: **blocked by the on-device vision bug**
- [ ] LoRA fine-tune: pipeline ready, not run (blocked on manifest verification)
- [ ] Human verification of the 18 ground truths

## The on-device vision bug (what failed)

On the iPhone (A15), the vision tower emits near-constant embeddings
(values ≈ {-1.79, 1.0}); the model generates confidently but hallucinates
("The Netherlands" for a puzzle-label photo). Survived: 3 framework builds
(2 hand-merged, 1 official xcframework), mmproj Q8/Q4/f16, FA on/off,
KV/batch sizes, image sizes, GPU/CPU decode. Same llama.cpp + weights work on
the M2. Text-only inference on the phone works. Full repro in
`docs/report.md` §8.