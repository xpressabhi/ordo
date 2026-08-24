# Ordo — take-home: a vision model on a phone

Qwen2-VL-2B on llama.cpp, measured across a quantization ladder on a Mac
(M2 Pro) and deployed to an iPhone (15 / A16 Bionic), on a private eval set of 18
real-world text photos.

**Live on GitHub Pages: https://xpressabhi.github.io/ordo/** · **Visual guide: [`docs/index.html`](docs/index.html)** · **Full write-up: [`docs/report.md`](docs/report.md)**

This README is the summary; the visual guide explains the architecture, user flows, task flow and every acronym in one page.

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

## Test on your iPhone (step by step)

The iOS app is `bench/ios/QwenBench/` (bundle `com.ordo.qwenbench`). Models
are bundled, so no downloads are needed on the phone.

**0 · One-time phone setup (iOS 16+)**
1. Connect the iPhone to the Mac with a cable.
2. **Enable Developer Mode**: `Settings → Privacy & Security → Developer Mode → On` — the phone reboots. (Required to run apps signed outside the App Store.)
3. **Trust your certificate** (needed after the first install): `Settings → General → VPN & Device Management → <your Apple ID> → Trust`.
4. Get the device UDID: `xcrun devicectl list devices` (or Finder → the phone's info panel).

**1 · Build (free Apple ID signing)**
```bash
cd bench/ios/QwenBench
xcodebuild -project llama.swiftui.xcodeproj -scheme llama.swiftui \
  -destination "id=<UDID>" -configuration Release -allowProvisioningUpdates build
```
Signing team: set `DEVELOPMENT_TEAM` to your Apple ID team in
`project.pbxproj` if it isn't yours. Free accounts: provisioning is
automatic, apps expire after **7 days** (rebuild + reinstall to refresh).

**2 · Install & launch**
```bash
APP=$(find ~/Library/Developer/Xcode/DerivedData -name "llama.swiftui.app" -path "*Release-iphoneos*" | head -1)
xcrun devicectl device install app --device "<UDID>" "$APP"
xcrun devicectl device process launch --device "<UDID>" com.ordo.qwenbench
```
> The phone must be **unlocked** for install/launch. Reinstall before each
> autorun run — launching a running app just foregrounds it (args ignored).

**3 · Automated test runs (headless)**
| Launch arg | What it runs |
|---|---|
| `autorun` | 28-photo eval set (768px) |
| `autorun-512` / `autorun-1024` | resolution sweep variants |
| `autorun-500` | SmolVLM-500M eval |
| `autorun-qwen` | Qwen2-VL-2B eval (Q8 mmproj) |
| `autorun-qwen-fast` | Qwen + Q4 mmproj, batch 4096, short answers |
| `autorun-multiturn` | 6 photos × 2 turns (KV reuse) |
| `autorun-stress` | 20 sequential queries |
| `autorun-soak` | 5-min continuous thermal soak |
| `autorun-text` | text-only smoke test |

**4 · Pull results off the phone**
```bash
xcrun devicectl device copy from --device "<UDID>" \
  --source Documents/results.jsonl --destination results/device_run.jsonl \
  --domain-type appDataContainer --domain-identifier com.ordo.qwenbench
```
`Documents/diag.txt` has the per-item debug (embedding stats, cache hits).

**5 · The photo flow (manual)**
Tap `Load model` → `Camera` → snap a photo (auto-compressed to <100KB,
vision pre-computed while you type) → type a question → `Ask` — follow-ups
on the same photo answer in ~200ms (cached embeddings).

## LoRA fine-tune (reproduce)

Answers the assignment's final question ("does fine-tuning recover the
quantization accuracy cost?") — on this task it does: **AR 38% -> 65%**
(+27 points, same 26 items, base vs adapter).

```
# 1. dataset from eval.json (paraphrased questions x2, 768px photos)
.venv/bin/python lora/build_dataset.py --augment 2 --val 0.15

# 2. train (mlx-vlm 0.6.15: use the Python trainer API — the CLI dropped
#    --export/--val-dataset; language-LoRA only, vision tower OOMs 16GB)
.venv/bin/python lora/run_lora.py      # 300 iters, rank 8, alpha 16, ~10 min

# 3. eval base vs adapter on the same 26 items
.venv/bin/python lora/eval_lora_mlx.py both
```

Gotchas: `model.eval()` breaks mlx-vlm 0.6.15 generation; images are
silently ignored unless the prompt goes through
`apply_chat_template(..., num_images=1)`; training images capped at 672px
to fit Metal memory. The GGUF-adapter seam (for running the LoRA inside
llama.cpp on-device) is the remaining ship path.

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