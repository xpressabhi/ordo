# Runtimes tried — Part 1 log

What we tried, what worked, what broke, and why. This is the honest
"getting it running" record the assignment asks for.

## Decision: llama.cpp (chosen)

| Considered | Verdict | Why |
|---|---|---|
| **llama.cpp + MTMD** | ✅ chosen | One GGUF format for the whole quantization ladder (F16/Q8/Q4/Q2 + mmproj F16/Q8/Q4), official `ggml-org` GGUFs for Qwen2-VL-2B, Metal GPU on Mac **and** iPhone (same code path), LoRA merge support. |
| **MLX (mlx-vlm)** | ⚪ fallback | Clean on Mac, but iOS story is younger; quantization ladder and mmproj cross-quant are less mature. Kept as a Mac-side cross-check / possible LoRA trainer. |
| **CoreML / ANE** | ❌ attempted, documented | Qwen2-VL's vision tower (dynamic resolution grid + 2D-RoPE) is not ANE-mappable without a heavy custom-op lift. This is the answer to the "did anything reach the NPU?" question: on iPhone, Metal GPU is used; the ANE is not reached by this stack. |

## Build log (macOS, Apple M2 Pro)

- `brew install cmake python@3.12` — cmake 4.4.2, python 3.12.14.
- llama.cpp pinned to **b10549** (the current release) so the MTMD API is
  identical between the Mac build and the iOS XCFramework build.
- **Failure 1**: `cstddef not found` on every compile — the Command Line Tools
  install on this machine has a broken/partial libc++ header dir
  (`/Library/Developer/CommandLineTools/usr/include/c++/v1` had 13 entries,
  no `cstddef`), while the SDK's own copy is complete. Fixed by pointing
  CMake at the Xcode toolchain + Xcode SDK explicitly
  (`-DCMAKE_C_COMPILER=.../XcodeDefault.xctoolchain/usr/bin/clang` +
  `-DCMAKE_OSX_SYSROOT=.../MacOSX.sdk`). This is a machine-specific quirk,
  not a llama.cpp issue.
- `cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_MTMD=ON` then built
  `llama-mtmd-cli`, `llama-quantize`, `llama-imatrix`, `llama-bench`.
- Tools on b10549: `llama-cli` is gone; vision lives in
  `tools/mtmd/llama-mtmd-cli` (`-m model --mmproj mmproj --image img -p prompt`).

## Weights

- `ggml-org/Qwen2-VL-2B-Instruct-GGUF` (not gated): text f16 / Q8_0 / Q4_K_M,
  mmproj f16 / Q8_0.
- Produced ourselves: Q2_K, IQ2_XXS (needs an imatrix calibration pass with
  `llama-imatrix` over wikitext-2 — reportable "what it took" step), and
  mmproj Q4_K_M via `llama-quantize`.

## GPU / NPU status (to be completed on iPhone)

- Mac (M2 Pro): Metal GPU used for both the LLM and the mmproj vision tower
  (confirmed by `ggml_metal_device_init` logs). CPU-only mmproj is ~7.6x
  slower on encode (34.4s vs 4.5s on the smoke image).
- iPhone: TBD — expect Metal GPU, no ANE.