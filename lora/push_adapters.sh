#!/usr/bin/env bash
# push_adapters.sh — publish the trained LoRA adapters to Hugging Face Hub.
#
# Uploads lora/adapters/adapters.safetensors + adapter_config.json plus a
# generated model card (base model, hyperparams, eval numbers) so fresh clones
# can fetch them via ./lora/fetch_adapters.sh instead of retraining.
#
# Needs: a WRITE token (Settings -> Access Tokens on huggingface.co):
#   export HF_TOKEN=hf_...   # must have write permission, not read
#   export HF_REPO=akm85/ordo-qwen2vl-lora   # optional override
#
# Usage: ./lora/push_adapters.sh [--private]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
ADAPTERS="$REPO/lora/adapters"
REPO_ID="${HF_REPO:-akm85/ordo-qwen2vl-lora}"
PRIVATE="false"
[[ "${1:-}" == "--private" ]] && PRIVATE="true"

for f in adapters.safetensors adapter_config.json; do
    if [[ ! -s "$ADAPTERS/$f" ]]; then
        echo "missing: $ADAPTERS/$f (train first: .venv/bin/python lora/run_lora.py)"
        exit 1
    fi
done
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN not set. Export a write token:  export HF_TOKEN=hf_..."
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/README.md" <<'EOF'
---
license: apache-2.0
base_model: Qwen/Qwen2-VL-2B-Instruct
tags: [lora, mlx, vision-language, qwen2-vl]
---

# ordo-qwen2vl-lora

LoRA adapters fine-tuning **Qwen/Qwen2-VL-2B-Instruct** on the ordo eval-style
task (real-world text photos: product labels, book spines, handwriting,
appliance displays). From https://github.com/xpressabhi/ordo.

- Trainer: `mlx-vlm` 0.6.15, Python API (`lora/run_lora.py`)
- Language-LoRA only (vision tower frozen — OOMs 16GB Metal otherwise)
- rank 8, alpha 16, dropout 0.0, lr 2e-5, 300 iters, batch 1, images capped 672px
- Dataset: `lora/build_dataset.py --augment 2` from `eval/eval.json` (paraphrased x2)
- Result: answer-recall **38% -> 65%** (+27pts, same 26 items, base vs adapter)

## Use (MLX)

```python
from mlx_vlm import load
model, processor = load("lora/mlx-model", adapter_path="lora/adapters")
```

Get the base via `lora/train.sh` (needs Qwen license acceptance), then fetch
these adapters without retraining:

```bash
./lora/fetch_adapters.sh   # -> lora/adapters/
```

## Files

- `adapters.safetensors` — final LoRA weights (= iter 300 checkpoint)
- `adapter_config.json` — rank/keys (all 28 language layers, q/k/v/o + mlp)
EOF

echo "== pushing to hf:$REPO_ID =="
HF_REPO_ID="$REPO_ID" HF_PRIVATE="$PRIVATE" ADAPTERS_DIR="$ADAPTERS" CARD="$TMP/README.md" \
"$PY" - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_REPO_ID"]
api.create_repo(repo_id, exist_ok=True,
                private=os.environ["HF_PRIVATE"] == "true")
for local, name in [(os.environ["CARD"], "README.md"),
                    (f'{os.environ["ADAPTERS_DIR"]}/adapters.safetensors', "adapters.safetensors"),
                    (f'{os.environ["ADAPTERS_DIR"]}/adapter_config.json', "adapter_config.json")]:
    api.upload_file(path_or_fileobj=local, path_in_repo=name, repo_id=repo_id)
    print("pushed", name)
print("done -> https://huggingface.co/" + repo_id)
PYEOF
