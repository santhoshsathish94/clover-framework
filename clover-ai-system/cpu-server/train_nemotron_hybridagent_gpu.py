#!/usr/bin/env python3
"""
8-GPU Distributed 3-Stage HybridSFT Trainer for NVIDIA Nemotron 3.5 Super VL
(`clover-ai-system/src/models/nemotron-3.5-super-vl/train_nemotron_hybridagent_gpu.py`).

Trains both:
  1. Canonical `HybridLink(hidden_size=4096)` (`input_norm`, `up_proj 4096->8192`, `GELU`,
     `down_proj 8192->4096`, `output_norm`, `gate`) — 1:1 dimension match with `HybridAgent-8B-Agentic`.
  2. `NemotronNativeHybridLink(d=4096, bottleneck=1024)` (warm-started from Nemotron Layer 1's
     `fc1_latent_proj` 4096->1024 and `fc2_latent_proj` 1024->4096).
  3. Special token embeddings & LM head rows for `<think>` (12), `</think>` (13), `<latent>` (30),
     and `</latent>` (31) + `norm_f` on Nemotron's 131,072 Tekken vocabulary.

3-Stage Curriculum (launched via `torchrun --nproc_per_node=8`):
  - Stage 1 (`format_activation`):     alpha_max=0.0, beta=0.05 (learns `<think>...<latent>...</latent>...</think>` transitions)
  - Stage 2 (`hybridlink_activation`): alpha_max=1.0, beta=0.15, warmup_ratio=0.15 (ramps latent feedback + cosine alignment)
  - Stage 3 (`self_conditioned`):      alpha_max=1.0, beta=0.10, warmup_ratio=0.0  (full self-conditioned latent reasoning)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/dev/shm/clover-train/lfm3-hybridagent/HybridAgent")

from hybrid_agent.hybridlink import HYBRIDLINK_CONFIG, HybridLink
from nemotron_gpu_backbone import NemotronGPUStageModel
from nemotron_hybrid_ncm import NemotronNativeHybridLink

SPECIAL_IDS = [12, 13, 30, 31]  # `<think>`, `</think>`, `<latent>`, `</latent>`


class NemotronHybridSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, max_seq_len: int = 512):
        self.records: list[dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    self.records.append(
                        {
                            "input_ids": row["input_ids"][:max_seq_len],
                            "labels": row["labels"][:max_seq_len],
                            "latent_mask": row["latent_mask"][:max_seq_len],
                        }
                    )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def collate_hybridsft(batch: list[dict[str, Any]], pad_token_id: int = 11) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    # Round up to multiple of 16 for GPU tensor core alignment
    max_len = ((max_len + 15) // 16) * 16
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    latent_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        L = len(item["input_ids"])
        input_ids[i, :L] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(item["labels"], dtype=torch.long)
        latent_mask[i, :L] = torch.tensor(item["latent_mask"], dtype=torch.bool)
        attention_mask[i, :L] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "latent_mask": latent_mask,
        "attention_mask": attention_mask,
    }


class NemotronHybridSFTTrainableModule(nn.Module):
    """
    Wraps the frozen GPU-resident `NemotronGPUStageModel` (`7.63 GiB` BF16) with trainable:
      - `hybrid_link`: Canonical `HybridLink(hidden_size=4096)`
      - `nemo_native_hybrid_link`: `NemotronNativeHybridLink(d=4096, bottleneck=1024)`
      - `special_emb_delta`: Trainable delta `[4, 4096]` for tokens `[12, 13, 30, 31]`
      - `special_lm_delta`:  Trainable delta `[4, 4096]` for LM head rows `[12, 13, 30, 31]`
      - `norm_f_delta`:      Trainable delta `[4096]` for final RMSNorm
    """

    def __init__(self, backbone: NemotronGPUStageModel, init_bridges_pt: str | None = None):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.hybrid_link = HybridLink(4096)
        self.nemo_native_hybrid_link = NemotronNativeHybridLink(d=4096, bottleneck=1024)

        # Warm-start NemotronNativeHybridLink from Layer 1's fc1_w and fc2_w (or existing bridges checkpoint)
        with torch.no_grad():
            self.nemo_native_hybrid_link.proj1.weight.copy_(self.backbone.moe_stage.fc1_w.float())
            self.nemo_native_hybrid_link.proj2.weight.copy_(self.backbone.moe_stage.fc2_w.float())

        if init_bridges_pt and os.path.exists(init_bridges_pt):
            sd = torch.load(init_bridges_pt, map_location="cpu", weights_only=False)
            if "nemo_native_hybrid_link" in sd:
                self.nemo_native_hybrid_link.load_state_dict(sd["nemo_native_hybrid_link"], strict=False)

        self.special_emb_delta = nn.Parameter(torch.zeros(len(SPECIAL_IDS), 4096, dtype=torch.float32))
        self.special_lm_delta = nn.Parameter(torch.zeros(len(SPECIAL_IDS), 4096, dtype=torch.float32))
        self.norm_f_delta = nn.Parameter(torch.zeros(4096, dtype=torch.float32))

    def get_adapted_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_embeds = self.backbone.embed_tokens(input_ids).float()
        for idx, tok_id in enumerate(SPECIAL_IDS):
            mask = input_ids == tok_id
            if mask.any():
                base_embeds = base_embeds + mask.unsqueeze(-1).float() * self.special_emb_delta[idx]
        return base_embeds

    def run_backbone(self, inputs_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs Mamba-2 -> Latent ReLU^2 MoE -> NoPE 16:1 GQA -> adapted norm_f -> adapted LM Head."""
        from nemotron_precomputed_engine import rms_norm_op

        h0 = inputs_embeds.to(torch.bfloat16)
        h1 = self.backbone.mamba_stage(h0, eps=self.backbone.config.rms_norm_eps)
        h2, routed_branch = self.backbone.moe_stage(h1, eps=self.backbone.config.rms_norm_eps)
        self.backbone.last_moe_routed = routed_branch
        self.backbone.last_moe_out = h2
        h3 = self.backbone.attn_stage(h2, eps=self.backbone.config.rms_norm_eps)
        eff_norm_f = (self.backbone.norm_f.float() + self.norm_f_delta).to(torch.bfloat16)
        h_final = rms_norm_op(h3, eff_norm_f, eps=self.backbone.config.rms_norm_eps)

        logits = self.backbone.lm_head(h_final).float()
        # Add trainable delta for special token rows [12, 13, 30, 31]
        special_logits = F.linear(h_final.float(), self.special_lm_delta)  # [B, T, 4]
        for idx, tok_id in enumerate(SPECIAL_IDS):
            logits[..., tok_id] = logits[..., tok_id] + special_logits[..., idx]

        return logits, h_final

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        latent_mask: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 0.10,
    ) -> dict[str, torch.Tensor]:
        token_embeds = self.get_adapted_embeddings(input_ids)  # [B, T, 4096] float32

        # Teacher pass: obtain pre-head hidden states H_teacher
        with torch.no_grad():
            _, h_teacher = self.run_backbone(token_embeds)

        # Shift hidden states by 1 position so h_{t-1} predicts latent state z_t at position t
        h_prev = torch.zeros_like(h_teacher, dtype=torch.float32)
        h_prev[:, 1:, :] = h_teacher[:, :-1, :].float()

        z_std = self.hybrid_link(h_prev)                     # [B, T, 4096]
        z_native = self.nemo_native_hybrid_link(h_prev)      # [B, T, 4096]
        z_combined = 0.5 * z_std + 0.5 * z_native

        # Cosine + MSE alignment loss on `<latent>...</latent>` positions
        valid_latent = latent_mask.clone()
        valid_latent[:, 0] = False
        if valid_latent.any():
            z_std_sel = z_std[valid_latent]
            z_nat_sel = z_native[valid_latent]
            e_sel = token_embeds[valid_latent].detach()

            cos_std = 1.0 - F.cosine_similarity(z_std_sel, e_sel, dim=-1).mean()
            cos_nat = 1.0 - F.cosine_similarity(z_nat_sel, e_sel, dim=-1).mean()
            mse_both = 0.5 * (
                F.mse_loss(F.normalize(z_std_sel, dim=-1), F.normalize(e_sel, dim=-1))
                + F.mse_loss(F.normalize(z_nat_sel, dim=-1), F.normalize(e_sel, dim=-1))
            )
            l_align = 0.5 * (cos_std + cos_nat) + 0.2 * mse_both
        else:
            # Ensure parameters stay in computation graph for DDP even if a batch has 0 latent tokens
            l_align = 0.0 * (z_std.sum() + z_native.sum())
            cos_std = torch.zeros((), device=input_ids.device)
            cos_nat = torch.zeros((), device=input_ids.device)

        # Student pass with blended latent embeddings inside `<latent>...</latent>` when alpha > 0
        if alpha > 0.0:
            mask_f = valid_latent.unsqueeze(-1).float()
            mixed_embeds = (1.0 - alpha * mask_f) * token_embeds + (alpha * mask_f) * z_combined
        else:
            mixed_embeds = token_embeds + 0.0 * z_combined

        logits, _ = self.run_backbone(mixed_embeds)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        l_ntp = F.cross_entropy(
            shift_logits.view(-1, self.backbone.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Compute accuracy on `<latent>` (30) and `</latent>` (31) transitions
        with torch.no_grad():
            preds = shift_logits.argmax(dim=-1)
            lat_enter_mask = shift_labels == 30
            lat_exit_mask = shift_labels == 31
            think_mask = (shift_labels == 12) | (shift_labels == 13)
            special_mask = lat_enter_mask | lat_exit_mask | think_mask
            if special_mask.any():
                special_acc = (preds[special_mask] == shift_labels[special_mask]).float().mean()
            else:
                special_acc = torch.zeros((), device=input_ids.device)

        total_loss = l_ntp + beta * l_align
        return {
            "loss": total_loss,
            "l_ntp": l_ntp.detach(),
            "l_align": l_align.detach(),
            "cos_std": (1.0 - cos_std.detach()),
            "cos_nat": (1.0 - cos_nat.detach()),
            "special_acc": special_acc.detach(),
        }

    def export_checkpoint(self, out_path: Path, extra_meta: dict[str, Any] | None = None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            adapted_special_emb = (
                self.backbone.embed_tokens.weight[SPECIAL_IDS].float().cpu() + self.special_emb_delta.detach().cpu()
            ).to(torch.bfloat16)
            adapted_special_lm = (
                self.backbone.lm_head.weight[SPECIAL_IDS].float().cpu() + self.special_lm_delta.detach().cpu()
            ).to(torch.bfloat16)
            adapted_norm_f = (
                self.backbone.norm_f.float().cpu() + self.norm_f_delta.detach().cpu()
            ).to(torch.bfloat16)

        payload = {
            "model_id": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
            "hidden_size": 4096,
            "vocab_size": 131072,
            "special_token_ids": {
                "<think>": 12,
                "</think>": 13,
                "<latent>": 30,
                "</latent>": 31,
            },
            "hybrid_link": {k: v.cpu() for k, v in self.hybrid_link.state_dict().items()},
            "nemo_native_hybrid_link": {
                k: v.cpu() for k, v in self.nemo_native_hybrid_link.state_dict().items()
            },
            "special_emb_delta": self.special_emb_delta.detach().cpu(),
            "special_lm_delta": self.special_lm_delta.detach().cpu(),
            "norm_f_delta": self.norm_f_delta.detach().cpu(),
            "adapted_special_emb_bf16": adapted_special_emb,
            "adapted_special_lm_bf16": adapted_special_lm,
            "adapted_norm_f_bf16": adapted_norm_f,
            "meta": extra_meta or {},
        }
        torch.save(payload, out_path)


@torch.no_grad()
def evaluate_stage(
    model: NemotronHybridSFTTrainableModule,
    eval_loader: DataLoader,
    device: torch.device,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    model.eval()
    tot_loss = 0.0
    tot_ntp = 0.0
    tot_align = 0.0
    tot_cos_std = 0.0
    tot_cos_nat = 0.0
    tot_spec_acc = 0.0
    n = 0
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        latent_mask = batch["latent_mask"].to(device, non_blocking=True)
        out = model(input_ids=input_ids, labels=labels, latent_mask=latent_mask, alpha=alpha, beta=beta)
        tot_loss += float(out["loss"].item())
        tot_ntp += float(out["l_ntp"].item())
        tot_align += float(out["l_align"].item())
        tot_cos_std += float(out["cos_std"].item())
        tot_cos_nat += float(out["cos_nat"].item())
        tot_spec_acc += float(out["special_acc"].item())
        n += 1
    model.train()
    n = max(1, n)
    return {
        "eval_loss": round(tot_loss / n, 4),
        "eval_ntp": round(tot_ntp / n, 4),
        "eval_ppl": round(math.exp(min(15.0, tot_ntp / n)), 2),
        "eval_align": round(tot_align / n, 4),
        "eval_cos_std": round(tot_cos_std / n, 4),
        "eval_cos_nat": round(tot_cos_nat / n, 4),
        "eval_special_acc": round(tot_spec_acc / n, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-jsonl",
        type=str,
        default="/dev/shm/clover-train/nemotron/data/train-2400-nemotron.jsonl",
    )
    parser.add_argument(
        "--eval-jsonl",
        type=str,
        default="/dev/shm/clover-train/nemotron/data/eval-32-nemotron.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/dev/shm/clover-train/nemotron/checkpoints/hybridagent",
    )
    parser.add_argument(
        "--init-bridges",
        type=str,
        default="/dev/shm/clover-train/nemotron/hybrid-ncm/nemotron_hybrid_ncm_bridges.pt",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--stage1-steps", type=int, default=50)
    parser.add_argument("--stage2-steps", type=int, default=80)
    parser.add_argument("--stage3-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_ddp = world_size > 1
    if is_ddp:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    t0 = time.perf_counter()
    # Stagger model loading slightly across ranks so `/dev/shm` mmap reads don't contend simultaneously
    time.sleep(local_rank * 0.25)
    backbone = NemotronGPUStageModel(experts_on_gpu=True).to(device)
    core_module = NemotronHybridSFTTrainableModule(backbone, init_bridges_pt=args.init_bridges).to(device)

    if is_ddp:
        dist.barrier()
        ddp_model = DDP(core_module, device_ids=[local_rank], find_unused_parameters=False)
    else:
        ddp_model = core_module

    train_ds = NemotronHybridSFTDataset(args.train_jsonl, max_seq_len=args.max_seq_len)
    eval_ds = NemotronHybridSFTDataset(args.eval_jsonl, max_seq_len=args.max_seq_len)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True, seed=42) if is_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_hybridsft,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_hybridsft,
    )

    trainable_params = [p for p in core_module.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    if local_rank == 0:
        vram_gb = torch.cuda.memory_allocated(device) / (1024**3)
        print(
            f"[Nemotron HybridAgent 8-GPU] Initialized on {world_size}x RTX 3090 in {time.perf_counter() - t0:.2f}s | "
            f"VRAM/GPU={vram_gb:.2f} GiB | Trainable params={num_trainable:,} | "
            f"Train trajectories={len(train_ds)} | Eval trajectories={len(eval_ds)}",
            flush=True,
        )

    stages = [
        {
            "name": "stage1_format_activation",
            "steps": args.stage1_steps,
            "alpha_start": 0.0,
            "alpha_end": 0.0,
            "beta": 0.05,
            "lr": args.lr,
        },
        {
            "name": "stage2_hybridlink_activation",
            "steps": args.stage2_steps,
            "alpha_start": 0.1,
            "alpha_end": 1.0,
            "beta": 0.15,
            "lr": args.lr * 0.8,
        },
        {
            "name": "stage3_self_conditioned",
            "steps": args.stage3_steps,
            "alpha_start": 1.0,
            "alpha_end": 1.0,
            "beta": 0.10,
            "lr": args.lr * 0.5,
        },
    ]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    all_stage_summaries: dict[str, Any] = {}

    if local_rank == 0:
        init_eval = evaluate_stage(core_module, eval_loader, device, alpha=0.0, beta=0.05)
        print(f"[Nemotron HybridAgent 8-GPU] Pre-training eval: {json.dumps(init_eval)}", flush=True)
        all_stage_summaries["pre_train_eval"] = init_eval

    epoch_counter = 0
    for st in stages:
        st_name = st["name"]
        st_steps = st["steps"]
        optimizer = torch.optim.AdamW(trainable_params, lr=st["lr"], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=st_steps, eta_min=st["lr"] * 0.1
        )

        st_t0 = time.perf_counter()
        data_iter = iter(train_loader)
        history: list[dict[str, float]] = []

        for step in range(1, st_steps + 1):
            progress = (step - 1) / float(max(1, st_steps - 1))
            alpha = st["alpha_start"] + progress * (st["alpha_end"] - st["alpha_start"])
            beta = st["beta"]

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_ntp = 0.0
            step_align = 0.0
            step_cos_std = 0.0
            step_cos_nat = 0.0
            step_spec_acc = 0.0

            for _ in range(args.grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch_counter += 1
                    if sampler is not None:
                        sampler.set_epoch(epoch_counter)
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                latent_mask = batch["latent_mask"].to(device, non_blocking=True)

                out = ddp_model(
                    input_ids=input_ids,
                    labels=labels,
                    latent_mask=latent_mask,
                    alpha=alpha,
                    beta=beta,
                )
                loss = out["loss"] / float(args.grad_accum)
                loss.backward()

                step_loss += float(out["loss"].item()) / float(args.grad_accum)
                step_ntp += float(out["l_ntp"].item()) / float(args.grad_accum)
                step_align += float(out["l_align"].item()) / float(args.grad_accum)
                step_cos_std += float(out["cos_std"].item()) / float(args.grad_accum)
                step_cos_nat += float(out["cos_nat"].item()) / float(args.grad_accum)
                step_spec_acc += float(out["special_acc"].item()) / float(args.grad_accum)

            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item())
            optimizer.step()
            scheduler.step()

            # Aggregate metrics across all 8 GPUs
            metrics_t = torch.tensor(
                [step_loss, step_ntp, step_align, step_cos_std, step_cos_nat, step_spec_acc],
                device=device,
                dtype=torch.float64,
            )
            if is_ddp:
                dist.all_reduce(metrics_t, op=dist.ReduceOp.AVG)
            m_loss, m_ntp, m_align, m_cos_std, m_cos_nat, m_spec_acc = metrics_t.tolist()

            if local_rank == 0:
                rec = {
                    "step": step,
                    "alpha": round(alpha, 3),
                    "loss": round(m_loss, 4),
                    "l_ntp": round(m_ntp, 4),
                    "l_align": round(m_align, 4),
                    "cos_std": round(m_cos_std, 4),
                    "cos_nat": round(m_cos_nat, 4),
                    "special_acc": round(m_spec_acc, 4),
                    "grad_norm": round(grad_norm, 4),
                    "lr": round(scheduler.get_last_lr()[0], 6),
                }
                history.append(rec)
                if step == 1 or step % 10 == 0 or step == st_steps:
                    elapsed = time.perf_counter() - st_t0
                    print(
                        f"[{st_name}] step {step:3d}/{st_steps} | alpha={alpha:.2f} | "
                        f"loss={m_loss:.4f} (ntp={m_ntp:.4f}, align={m_align:.4f}) | "
                        f"cos_std={m_cos_std:.4f} | cos_nat={m_cos_nat:.4f} | "
                        f"special_acc={m_spec_acc:.4f} | grad={grad_norm:.3f} | {elapsed:.1f}s",
                        flush=True,
                    )

        if is_ddp:
            dist.barrier()

        if local_rank == 0:
            st_eval = evaluate_stage(core_module, eval_loader, device, alpha=st["alpha_end"], beta=st["beta"])
            st_elapsed = time.perf_counter() - st_t0
            st_summary = {
                "stage": st_name,
                "steps": st_steps,
                "world_size_gpus": world_size,
                "effective_batch_size": args.batch_size * args.grad_accum * world_size,
                "elapsed_s": round(st_elapsed, 2),
                "first_step": history[0],
                "last_step": history[-1],
                "eval": st_eval,
                "hybrid_link_proj2_norm": round(float(core_module.hybrid_link.proj2.weight.norm().item()), 4),
                "out_scale_nemo_native": round(float(core_module.nemo_native_hybrid_link.out_scale.item()), 4),
            }
            all_stage_summaries[st_name] = st_summary
            ckpt_path = out_root / st_name / "adapter.pt"
            core_module.export_checkpoint(ckpt_path, extra_meta=st_summary)
            # Also save standalone `adapter_config.json` + `hybridlink_adapter_4096.pt` in HybridAgent format
            torch.save(
                {k: v.cpu() for k, v in core_module.hybrid_link.state_dict().items()},
                out_root / st_name / "hybridlink_adapter_4096.pt",
            )
            (out_root / st_name / "adapter_config.json").write_text(
                json.dumps(HYBRIDLINK_CONFIG, indent=2), encoding="utf-8"
            )
            print(
                f"[{st_name}] COMPLETED in {st_elapsed:.1f}s | Eval: {json.dumps(st_eval)} | Saved -> {ckpt_path}",
                flush=True,
            )

        if is_ddp:
            dist.barrier()

    if local_rank == 0:
        final_ckpt = out_root / "nemotron_hybridagent_final.pt"
        core_module.export_checkpoint(final_ckpt, extra_meta=all_stage_summaries)
        summary_path = out_root / "nemotron_hybridagent_training_summary.json"
        summary_path.write_text(json.dumps(all_stage_summaries, indent=2), encoding="utf-8")
        print(f"\n[Nemotron HybridAgent 8-GPU] ALL 3 STAGES COMPLETE! Final checkpoint: {final_ckpt}", flush=True)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
