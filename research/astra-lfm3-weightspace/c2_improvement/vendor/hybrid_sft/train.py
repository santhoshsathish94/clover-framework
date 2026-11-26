"""HybridSFT trainer: runs one stage, format activation, HybridLink
activation or self-conditioned HybridSFT, with the recipe's hyperparameters as defaults.

Launch on 8 GPUs with DeepSpeed ZeRO-2 (see scripts/run_hybrid_sft.sh):
  accelerate launch --config_file hybrid_sft/configs/accelerate_zero2.yaml -m hybrid_sft.train \
      --stage format_activation --model Qwen/Qwen3-8B-Base --data <jsonl or HF dataset id> \
      --output_dir outputs/hybrid_sft/format_activation
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from hybrid_agent.model import add_anchor_tokens, resolve_model_dir
from hybrid_sft.data import HybridSFTDataset, collate_batch, resolve_data_file
from hybrid_sft.model import FORMAT_ACTIVATION, HYBRIDLINK_ACTIVATION, SELF_CONDITIONED, STAGES, HybridSFTModel

# Epochs per stage. 1/1/1 and 1/1/2 are good practice as well; evaluate the saved HybridSFT
# checkpoints (e.g. on AIME and MATH-500) and take the best one as the initialization for HybridRL.
STAGE_EPOCHS = {FORMAT_ACTIVATION: 1, HYBRIDLINK_ACTIVATION: 2, SELF_CONDITIONED: 1}
STAGE_HYBRIDLINK_LR = {HYBRIDLINK_ACTIVATION: 2e-4, SELF_CONDITIONED: 5e-5}

WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 4
LOG_EVERY = 10
WANDB_PROJECT = "HybridAgent"


def parse_args():
    p = argparse.ArgumentParser(description="HybridSFT: one training stage")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--model", required=True,
                   help="base model for format activation, otherwise the previous stage's output (dir or HF id)")
    p.add_argument("--data", required=True, help="hybrid trajectories: local JSONL path or HF dataset repo id")
    p.add_argument("--data_file", default="train.jsonl", help="file name inside the HF dataset repo")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=None, help="default per stage: 1 / 2 / 1")
    p.add_argument("--lr", type=float, default=7e-6, help="backbone learning rate")
    p.add_argument("--hybridlink_lr", type=float, default=None,
                   help="HybridLink learning rate (default 2e-4 for activation, 5e-5 for self-conditioning)")
    p.add_argument("--batch_size", type=int, default=1, help="per device")
    p.add_argument("--grad_accum", type=int, default=2, help="must match the accelerate/DeepSpeed config")
    p.add_argument("--max_length", type=int, default=16888, help="longer trajectories are dropped")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()
    if args.epochs is None:
        args.epochs = STAGE_EPOCHS[args.stage]
    if args.hybridlink_lr is None:
        args.hybridlink_lr = STAGE_HYBRIDLINK_LR.get(args.stage, args.lr)
    return args


def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    """Linear warmup, then cosine decay clamped at the end of training."""
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be > 0")

    def lr_lambda(current_step: int) -> float:
        if num_warmup_steps > 0 and current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if num_training_steps <= num_warmup_steps:
            return 0.0
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16")
    set_seed(args.seed)
    is_main = accelerator.is_main_process

    model_dir = resolve_model_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_main:
        print(f"Loading model from {model_dir} ...", flush=True)
    model = HybridSFTModel.from_pretrained(
        model_dir, dtype=torch.bfloat16, require_hybridlink=(args.stage != FORMAT_ACTIVATION),
    )
    model.resize_to_tokenizer(tokenizer)
    model.set_trainable(args.stage)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    data_file = resolve_data_file(args.data, args.data_file)
    dataset = HybridSFTDataset(
        tokenizer, data_file, max_length=args.max_length, text_only=(args.stage == FORMAT_ACTIVATION),
    )
    if is_main:
        print(f"Dataset: {len(dataset)} trajectories from {data_file}", flush=True)
    if len(dataset) == 0:
        raise RuntimeError("no usable trajectories in the data file")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda batch: collate_batch(tokenizer, batch),
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    link_params = [p for p in model.hybridlink.parameters() if p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr})
    if link_params:
        param_groups.append({"params": link_params, "lr": args.hybridlink_lr})
    if is_main:
        print(f"Trainable parameters: backbone={sum(p.numel() for p in backbone_params):,} "
              f"hybridlink={sum(p.numel() for p in link_params):,}", flush=True)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if accelerator.gradient_accumulation_steps != args.grad_accum:
        raise RuntimeError(
            f"gradient accumulation mismatch: --grad_accum {args.grad_accum} vs "
            f"{accelerator.gradient_accumulation_steps} in the accelerate/DeepSpeed config"
        )

    steps_per_epoch = math.ceil(len(dataloader) / args.grad_accum)
    max_steps = args.epochs * steps_per_epoch
    if max_steps <= 0:
        raise ValueError("no optimizer steps: reduce grad_accum or batch_size, or use more data")
    warmup_steps = max(1, int(max_steps * WARMUP_RATIO))
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    for group in optimizer.param_groups:  # first update at the step-0 learning rate
        group["lr"] = 0.0

    os.makedirs(args.output_dir, exist_ok=True)
    wandb_run = None
    if is_main:
        with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Training {args.stage}: {max_steps} steps "
              f"(effective batch {args.batch_size * args.grad_accum * accelerator.num_processes})", flush=True)
        if args.wandb:
            import wandb

            wandb_run = wandb.init(
                project=WANDB_PROJECT, name=f"hybrid_sft-{args.stage}-{Path(args.output_dir).name}",
                config=vars(args), dir=args.output_dir,
            )

    def save(ckpt_dir: str) -> None:
        accelerator.wait_for_everyone()
        full_sd = accelerator.get_state_dict(model)  # collective
        if is_main:
            accelerator.unwrap_model(model).save_pretrained(ckpt_dir, tokenizer, full_sd)
        del full_sd
        accelerator.wait_for_everyone()

    model.train()
    global_step = 0
    pbar = tqdm(total=max_steps, desc=args.stage, disable=not is_main)
    accum = {"loss": 0.0, "backbone_loss": 0.0, "hybridlink_loss": 0.0, "n": 0}

    for epoch in range(args.epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                out = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    loss_mask=batch["loss_mask"], latent_mask=batch["latent_mask"], stage=args.stage,
                )
                accelerator.backward(out["loss"])
                for key in ("loss", "backbone_loss", "hybridlink_loss"):
                    accum[key] += out[key].item()
                accum["n"] += 1

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    n = max(accum["n"], 1)
                    stats = {k: accum[k] / n for k in ("loss", "backbone_loss", "hybridlink_loss")}
                    stats["lr"] = scheduler.get_last_lr()[0]
                    accum = {"loss": 0.0, "backbone_loss": 0.0, "hybridlink_loss": 0.0, "n": 0}
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in stats.items() if k != "lr"})
                    pbar.update(1)
                    if wandb_run is not None:
                        wandb_run.log({f"train/{k}": v for k, v in stats.items()} | {"train/step": global_step})
                    if is_main and global_step % LOG_EVERY == 0:
                        print(f"step={global_step} loss={stats['loss']:.4f} lr={stats['lr']:.2e}", flush=True)
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        save(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    if global_step >= max_steps:
                        break
        if global_step >= max_steps:
            break

    save(args.output_dir)
    if is_main:
        if wandb_run is not None:
            wandb_run.finish()
        print(f"Saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
