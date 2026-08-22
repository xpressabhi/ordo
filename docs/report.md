# Ordo take-home — report

> Full write-up. The README is the 3-5 page version; this file is the detail.

## 1. Getting it running
(see notes/runtimes_tried.md for the raw log)

## 2. The eval set
- N items, X categories, difficulty spread, how it was built, quality checks
  (baseline sanity, inter-annotator agreement via eval/scripts/second_annotator.py)

## 3. Setup — model, runtime, device, reproduction

## 4. Results
| Precision (text x mmproj) | Size | Accuracy (AR/EM/F1) | Peak RAM | Encode ms | Prefill tok/s | Decode tok/s | TTFT |
|---|---|---|---|---|---|---|---|

## 5. Where it breaks
- failure pattern by condition tag / difficulty
- encoder vs decoder degradation (mmproj cross-quant matrix)

## 6. Recommendation

## 7. What surprised me

## 8. What I cut / couldn't get working

## Notes from the assignment ("Few points to notice")
- Quant halves bytes — does it halve latency? Which stage benefits most?
- Why does the vision encoder tolerate quantization differently from the decoder?
- 500-token prompt vs one image: TTFT attribution (see results/ttft_decompose.csv)
- ASR + TTS slice of the 2-second budget — what's left for the model?
