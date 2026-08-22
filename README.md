# Ordo — take-home: a vision model on a phone

Qwen2-VL-2B on llama.cpp: what a 2B vision-language model costs on real
hardware, measured across a quantization ladder on the Mac (M2 Pro) and an
iPhone, on a private eval set of real-world text photos.

**Status: in progress** — see `docs/report.md` for the full write-up (the
README will be the 3-5 page version once measurements land).

## Layout

```
model/fetch.sh        download the GGUF weights (not gated)
model/                quantized ladder: f16 Q8_0 Q4_K_M Q2_K IQ2_XXS + mmproj f16/Q8/Q4
bench/tool/bench_vlm.cpp   C++ harness: encode / prefill / decode / TTFT as 4 numbers
bench/mac/            run_bench.py (ladder driver), report.py (metrics+tables),
                      decompose.py (TTFT attribution)
bench/ios/QwenBench/  Xcode app: on-device vision benchmark + stress harness
eval/                 eval.json manifest + capture/validate/second-annotator scripts
lora/                 MLX LoRA fine-tune pipeline (build_dataset/train/export)
results/              CSVs + tables (generated)
notes/runtimes_tried.md   what worked / broke / had to be patched
```

## Reproduce (Mac part)

```
./model/fetch.sh --all
./bench/tool/build_bench.sh
.venv/bin/python bench/mac/run_bench.py --tag results
.venv/bin/python bench/mac/report.py results/results_<tag>.csv
```

## Eval set

Own photos only (25-40), real-world text: menus, labels, signs, handwriting,
receipts, medicine, book spines, whiteboard, appliance displays. Built via
`eval/scripts/capture_wizard.py`; checked with `validate_eval.py`;
inter-annotator agreement via `second_annotator.py`.

## Metric

Answer Recall (headline — does the prediction contain the ground-truth
answer?), plus exact-match and word-F1 as strict/secondary checks. See
`eval/scripts/metrics.py` for the reasoning.

## Headline findings so far (Mac, M2 Pro)

- The vision encoder dominates TTFT: ~4.5s encode vs ~1s prefill vs ~0.2s
  decode at Q4 on a 1200x900 image. A 600-token prompt adds only ~0.6s.
- Encoding is ~7.6x slower with the mmproj on CPU (34s vs 4.5s) — GPU matters.
- Quantizing the text model speeds decode ~1.8x (f16 43 tok/s -> Q4 79 tok/s);
  encode is barely affected by quantization (bandwidth-bound on the image).

## Phone part (pending)

- Build XCFramework (`build-xcframework.sh ios-sim ios-device`)
- Open `bench/ios/QwenBench`, set your Apple ID team, deploy, copy the two
  GGUFs into the app's Documents via Finder, run eval/stress/soak, export
  results.json.