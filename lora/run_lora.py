#!/usr/bin/env python3
# run_lora.py — CLI-free LoRA training for mlx-vlm 0.6.15 (local jsonl datasets).
import os
import mlx.optimizers as optim
from datasets import load_dataset
from mlx_vlm.lora import (
    load,
    transform_dataset_to_messages,
    setup_model_for_training,
)
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
from mlx_vlm.trainer.utils import print_trainable_parameters

MODEL = "lora/mlx-model"
TRAIN = "lora/data/train.jsonl"

print("Loading model...", flush=True)
model, processor = load(MODEL, processor_config={"trust_remote_code": True})
model_type = getattr(getattr(model, "config", None), "model_type", None)
config = model.config.__dict__

dataset = load_dataset("json", data_files=TRAIN, split="train")
dataset = transform_dataset_to_messages(dataset, model_type, None)
train_dataset = VisionDataset(dataset, config, processor, image_resize_shape=(672, 672))

import types as _t
args_ns = _t.SimpleNamespace(full_finetune=False, lora_rank=8, lora_alpha=16, lora_dropout=0.0, train_vision=False)
model = setup_model_for_training(model, args_ns, None)
print_trainable_parameters(model)
optimizer = optim.Adam(learning_rate=2e-5)

args = TrainingArgs(
    batch_size=1,
    iters=300,
    steps_per_report=10,
    steps_per_eval=200,
    steps_per_save=100,
    val_batches=4,
    max_seq_length=2048,
    adapter_file="lora/adapters/adapters.safetensors",
    grad_checkpoint=True,
    learning_rate=2e-5,
    grad_clip=None,
    gradient_accumulation_steps=1,
    full_finetune=False,
)
os.makedirs("lora/adapters", exist_ok=True)
print("Training...", flush=True)
train(
    model=model,
    optimizer=optimizer,
    train_dataset=train_dataset,
    val_dataset=None,
    args=args,
    train_on_completions=False,
    assistant_id=77091,
)
print("done", flush=True)