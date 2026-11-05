#!/usr/bin/env python3
"""
8-GPU Distributed 3-Stage HybridSFT Trainer for MoonshotAI Kimi K3.1 Pro (`d=7168`, `V=163840`)
(`porting-models/train_kimi_k31_hybridagent_gpu.py`).

Trains:
  1. `K31NativeHybridLink(d=7168, bottleneck=3584)`:
     Warm-started from Kimi K3.1 Pro Layer 3 (Gated DSA + MoE)'s `routed_expert_down_proj [3584, 7168]`
     and `routed_expert_up_proj [7168, 3584]`.
  2. `QwenToK31HybridBridge(d_ha=4096, d_k31=7168)` + canonical `HybridLink(4096)`:
     Maps bidirectionally between `HybridAgent-8B-Agentic`'s `4096`-d `<latent>` vectors `R(h)`
     and Kimi K3.1 Pro's `7168`-d hidden states.
  3. Special token embeddings & LM head rows for `<think>` (163606), `</think>` (163607),
     `<latent>` (163608), and `</latent>` (163609) + `norm_f_delta` on Kimi K3.1 Pro's `163,840` vocabulary.

3-Stage Curriculum (launched via `torchrun --nproc_per_node=8`):
  - Stage 1 (`stage1_format_activation`):     alpha_max=0.0, beta=0.05
  - Stage 2 (`stage2_hybridlink_activation`): alpha_max=1.0, beta=0.15, warmup_ratio=0.15
  - Stage 3 (`stage3_self_conditioned`):      alpha_max=1.0, beta=0.10, warmup_ratio=0.0
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kimi_k31_gpu_backbone import (
    D_HA,
    D_K31,
    D_LATENT,
    V_K31,
    KimiK31GPUStageModel,
    attn_res_aggregate_op,
    rms_norm_op,
)

SPECIAL_IDS = [163606, 163607, 163608, 163609]  # `<think>`, `</think>`, `<latent>`, `</latent>`
PAD_ID = 163585
BOS_ID = 163584
EOS_ID = 163586


def load_kimi_k31_tiktoken(tiktoken_model_path: str | Path) -> tiktoken.Encoding:
    """Loads `moonshotai/Kimi-K3.1-Pro`'s `tiktoken.model` (`163,840` vocab)."""
    mergeable_ranks: dict[bytes, int] = {}
    with open(tiktoken_model_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b64_tok, rank_str = line.split()
            mergeable_ranks[base64.b64decode(b64_tok)] = int(rank_str)
    pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    return tiktoken.Encoding(
        name="kimi_k31_tiktoken",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )


class K31NativeHybridLink(nn.Module):
    """
    Native HybridLink (`ln_res_adapter`) operating in Kimi K3.1 Pro's `d = 7,168` hidden space
    with `bottleneck = 3,584` (matching Kimi K3.1 Pro's `moe_latent_size = 3,584` so it can be
    warm-started directly from `routed_expert_down_proj` and `routed_expert_up_proj`).
    """

    def __init__(self, d: int = D_K31, bottleneck: int = D_LATENT):
        super().__init__()
        self.pre_ln = nn.LayerNorm(d)
        self.proj1 = nn.Linear(d, bottleneck)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(bottleneck, d)
        self.post_ln = nn.LayerNorm(d)
        self.out_scale = nn.Parameter(torch.tensor(0.0219))

    def forward(self, h_k31: torch.Tensor) -> torch.Tensor:
        hf = h_k31.float()
        z = self.post_ln(hf + self.proj2(self.act(self.proj1(self.pre_ln(hf)))))
        return z * self.out_scale


class QwenToK31HybridBridge(nn.Module):
    """
    Bidirectional bridge between `HybridAgent`'s `4,096`-d `<latent>` reasoning space and
    Kimi K3.1 Pro's `7,168`-d hidden/embedding space (`to_ha: 7168 -> 4096`, `ha_link: 4096 -> 4096`, `from_ha: 4096 -> 7168`).
    """

    def __init__(self, d_k31: int = D_K31, d_ha: int = D_HA):
        super().__init__()
        self.to_ha = nn.Linear(d_k31, d_ha, bias=False)
        self.input_norm = nn.LayerNorm(d_ha)
        self.up_proj = nn.Linear(d_ha, 8192)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(8192, d_ha)
        self.output_norm = nn.LayerNorm(d_ha)
        self.from_ha = nn.Linear(d_ha, d_k31, bias=False)
        self.out_scale = nn.Parameter(torch.tensor(0.0219))

    def forward_ha_4096(self, h_k31: torch.Tensor) -> torch.Tensor:
        x = self.to_ha(h_k31.float())
        r_h = self.output_norm(x + self.down_proj(self.act(self.up_proj(self.input_norm(x)))))
        return r_h

    def forward(self, h_k31: torch.Tensor) -> torch.Tensor:
        r_h = self.forward_ha_4096(h_k31)
        return self.from_ha(r_h) * self.out_scale


def build_kimi_k31_hybridsft_dataset(
    tok_model_path: Path,
    out_train_jsonl: Path,
    out_eval_jsonl: Path,
    n_train: int = 640,
    n_eval: int = 64,
) -> dict[str, Any]:
    """
    Builds `<think> ... <latent> z_1 .. z_K </latent> ... </think> Answer` training and eval examples
    tokenized with Kimi K3.1 Pro's native `163,840`-vocab `tiktoken.model`.
    """
    enc = load_kimi_k31_tiktoken(tok_model_path)
    rng = random.Random(42)

    qa_pairs = [
        (
            "What is the attention architecture of Kimi K3.1 Pro?",
            "Kimi K3.1 Pro uses a 3:1 hybrid attention stack of 69 Kimi Delta Attention (KDA) layers and 24 Gated DeepSeek Sparse Attention (Gated DSA) layers.",
            "69 KDA linear attention layers paired with 24 Gated DSA layers equipped with a 5-tensor Lightning Indexer selecting top-2048 keys over a 1M context window.",
            "Kimi K3.1 Pro replaces full-attention Gated MLA on its 24 global layers with Gated DeepSeek Sparse Attention (Gated DSA), combining a 32-head Lightning Indexer (index_topk=2048) with 69 KDA linear attention layers and 896 SiTU-GLU routed experts per MoE layer.",
        ),
        (
            "How does the Gated DSA Lightning Indexer work in Kimi K3.1 Pro?",
            "The Lightning Indexer projects queries via wq_b into 32 heads of dim 128, normalizes keys via wk and k_norm, and weights heads via weights_proj.",
            "5 new tensors per DSA layer: k_norm.weight, k_norm.bias, weights_proj.weight, wk.weight, and wq_b.weight selecting top 2048 KV positions.",
            "Each of the 24 Gated DSA layers in Kimi K3.1 Pro includes a 5-tensor Lightning Indexer (`k_norm.weight`, `k_norm.bias`, `weights_proj.weight`, `wk.weight`, `wq_b.weight`) that scores key relevance across 32 indexer heads and selects the top 2,048 KV entries before Gated MLA.",
        ),
        (
            "Explain how SpecMD Least-Stale slot caching with Rule R4 clean-drop reduces PCIe traffic.",
            "Read-only expert weights fetched from host memory are never modified on GPU, so Rule R4 drops them cleanly without dirty D2H writebacks.",
            "Partitioning the slot cache per layer and applying Red-Blue Pebble Rule R4 clean-drop eliminates 100% of dirty PCIe writebacks.",
            "In Clover's 4-tier memory hierarchy, SpecMD Least-Stale evicts the expert whose next predicted activation is furthest in the future, while Red-Blue Pebble Rule R4 drops clean read-only expert weights without writing them back across PCIe.",
        ),
        (
            "Summarize the Harbourside Council Mill Road cycle lane vote.",
            "Harbourside Council voted 7 to 2 on Tuesday to approve 2.4 million pounds for a 14-week cycle lane on Mill Road proposed by Priya Raman.",
            "Shop owners led by Tom Eliot raised parking concerns, so the council added 40 temporary parking spaces on Dock Street.",
            "Harbourside Council voted 7 to 2 to approve £2.4 million for the Mill Road cycle lane starting in March, providing 40 temporary parking spaces on Dock Street to address shop owners' concerns.",
        ),
        (
            "What is the safety procedure before opening subsea equalization valve V-402?",
            "Before opening subsea valve V-402, equalize differential pressure across the manifold and verify downstream telemetry.",
            "Never open V-402 under high differential pressure; confirm sensor telemetry and pressure balance first.",
            "Before opening subsea equalization valve V-402, operators must equalize differential pressure across the manifold and verify that downstream pressure and flow telemetry are within safe operating bounds.",
        ),
        (
            "How does Next-Concept Modeling (NCM) generate text in Clover AI Studio?",
            "NCM predicts 11 hierarchical vector-quantized concept levels in parallel conditioned on HybridLink latents and Kimi K3.1 Pro MoE expert states.",
            "K31ComboBridge(7168->384) and K31ExpertBridge(7168->384) feed NextConceptTransformer and K31LexicalPointerDecoder over 163,840 tokens.",
            "Clover's Next-Concept Model projects 7,168-d Kimi K3.1 Pro hidden states and HybridLink latents into a 384-d concept space, predicts 11 VQ concept levels in parallel, and decodes surface tokens in Kimi K3.1 Pro's 163,840 vocabulary.",
        ),
        (
            "How does Kimi-K3.1-Pro-DSpark accelerate decoding?",
            "DSpark is a 5-layer 2.25B BF16 speculative draft head that fuses multi-depth hidden states from layers 7, 23, 67, 83, and 91.",
            "Using block-parallel speculative decoding and a Markov confidence head, DSpark achieves a mean acceptance length of 3.45 tokens.",
            "LithosAI's Kimi-K3.1-Pro-DSpark fuses hidden states from target layers [7, 23, 67, 83, 91] via a 35,840-to-7,168 projection into 5 lightweight attention+MLP layers, drafting 8-token blocks with a 3.45x mean acceptance length.",
        ),
        (
            "What are the five stages of the Clover operating cycle?",
            "The five stages of the Clover cycle are Context, Direction, Execution, Outcome, and Growth.",
            "If the Outcome is not the intended one, return to Context rather than repeating Execution blindly.",
            "Clover operates on a five-stage cycle: Context, Direction, Execution, Outcome, and Growth. Every non-trivial action starts by establishing Context from primary sources, and any unexpected Outcome returns to Context.",
        ),
    ]

    def make_record(idx: int) -> dict[str, Any]:
        q, thought1, latent_text, ans = qa_pairs[idx % len(qa_pairs)]
        q_ids = enc.encode(f"Question ({idx}): {q}\nReasoning: ")
        t1_ids = enc.encode(thought1 + " ")
        lat_ids = enc.encode(latent_text)[:24]
        t2_ids = enc.encode(" Concluding answer: ")
        ans_ids = enc.encode(ans)

        input_ids = (
            [BOS_ID]
            + q_ids
            + [SPECIAL_IDS[0]]  # <think>
            + t1_ids
            + [SPECIAL_IDS[2]]  # <latent>
            + lat_ids
            + [SPECIAL_IDS[3]]  # </latent>
            + t2_ids
            + [SPECIAL_IDS[1]]  # </think>
            + ans_ids
            + [EOS_ID]
        )
        # Labels supervise from <think> onward
        prompt_len = 1 + len(q_ids)
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        lat_start = prompt_len + 1 + len(t1_ids) + 1
        lat_end = lat_start + len(lat_ids)
        latent_mask = [1 if (lat_start <= p < lat_end) else 0 for p in range(len(input_ids))]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "latent_mask": latent_mask,
        }

    out_train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_train_jsonl, "w", encoding="utf-8") as f:
        for i in range(n_train):
            f.write(json.dumps(make_record(i)) + "\n")
    with open(out_eval_jsonl, "w", encoding="utf-8") as f:
        for i in range(n_eval):
            f.write(json.dumps(make_record(1000 + i)) + "\n")

    return {"train_examples": n_train, "eval_examples": n_eval}


class KimiK31HybridSFTDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, max_seq_len: int = 384):
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


def collate_k31_hybridsft(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    max_len = ((max_len + 15) // 16) * 16
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    latent_mask = torch.zeros((bsz, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        L = len(item["input_ids"])
        input_ids[i, :L] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(item["labels"], dtype=torch.long)
        latent_mask[i, :L] = torch.tensor(item["latent_mask"], dtype=torch.bool)

    return {"input_ids": input_ids, "labels": labels, "latent_mask": latent_mask}


class KimiK31HybridSFTTrainableModule(nn.Module):
    """
    Wraps the frozen GPU-resident `KimiK31GPUStageModel` (`d=7168`, `V=163840`) with trainable:
      - `k31_native_hybrid_link`: `K31NativeHybridLink(d=7168, bottleneck=3584)` warm-started from
        Layer 3's `routed_expert_down_proj [3584, 7168]` and `routed_expert_up_proj [7168, 3584]`
      - `qwen_to_k31_bridge`:     `QwenToK31HybridBridge(d_k31=7168, d_ha=4096)`
      - `special_emb_delta`:      Trainable delta `[4, 7168]` for `<think>`, `</think>`, `<latent>`, `</latent>`
      - `special_lm_delta`:       Trainable delta `[4, 7168]` for LM head rows `[163606..163609]`
      - `norm_f_delta`:           Trainable delta `[7168]` for final RMSNorm
    """

    def __init__(self, backbone: KimiK31GPUStageModel):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.k31_native_hybrid_link = K31NativeHybridLink(d=D_K31, bottleneck=D_LATENT)
        self.qwen_to_k31_bridge = QwenToK31HybridBridge(d_k31=D_K31, d_ha=D_HA)

        # Warm-start K31NativeHybridLink from Kimi K3.1 Pro Layer 3's latent MoE projections
        with torch.no_grad():
            self.k31_native_hybrid_link.proj1.weight.copy_(self.backbone.dsa_moe_stage.lat_down_w.float())
            self.k31_native_hybrid_link.proj2.weight.copy_(self.backbone.dsa_moe_stage.lat_up_w.float())

        self.special_emb_delta = nn.Parameter(torch.zeros(len(SPECIAL_IDS), D_K31, dtype=torch.float32))
        self.special_lm_delta = nn.Parameter(torch.zeros(len(SPECIAL_IDS), D_K31, dtype=torch.float32))
        self.norm_f_delta = nn.Parameter(torch.zeros(D_K31, dtype=torch.float32))

    def get_adapted_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_embeds = self.backbone.embed_tokens(input_ids).float()
        for idx, tok_id in enumerate(SPECIAL_IDS):
            mask = input_ids == tok_id
            if mask.any():
                base_embeds = base_embeds + mask.unsqueeze(-1).float() * self.special_emb_delta[idx]
        return base_embeds

    def run_backbone(self, inputs_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = inputs_embeds.to(torch.bfloat16)
        snaps = [h0]
        h1, snaps = self.backbone.kda_stage(h0, snaps, eps=self.backbone.config.rms_norm_eps)
        h2, routed_branch = self.backbone.dsa_moe_stage(h1, snaps, eps=self.backbone.config.rms_norm_eps)
        self.backbone.last_moe_routed = routed_branch
        self.backbone.last_moe_out = h2

        agg_final = attn_res_aggregate_op(
            snaps, h2, self.backbone.out_res_norm, self.backbone.out_res_proj, eps=self.backbone.config.rms_norm_eps
        )
        eff_norm_f = (self.backbone.norm_f.float() + self.norm_f_delta).to(torch.bfloat16)
        h_final = rms_norm_op(agg_final, eff_norm_f, eps=self.backbone.config.rms_norm_eps)

        logits = self.backbone.lm_head(h_final).float()
        special_logits = F.linear(h_final.float(), self.special_lm_delta)
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
        token_embeds = self.get_adapted_embeddings(input_ids)

        with torch.no_grad():
            _, h_teacher = self.run_backbone(token_embeds)

        h_prev = torch.zeros_like(h_teacher, dtype=torch.float32)
        h_prev[:, 1:, :] = h_teacher[:, :-1, :].float()

        z_native = self.k31_native_hybrid_link(h_prev)
        z_bridge = self.qwen_to_k31_bridge(h_prev)
        z_combined = 0.5 * z_native + 0.5 * z_bridge

        valid_latent = latent_mask.clone()
        valid_latent[:, 0] = False
        if valid_latent.any():
            z_nat_sel = z_native[valid_latent]
            z_brg_sel = z_bridge[valid_latent]
            e_sel = token_embeds[valid_latent].detach()

            cos_nat = F.cosine_similarity(z_nat_sel, e_sel, dim=-1).mean()
            cos_brg = F.cosine_similarity(z_brg_sel, e_sel, dim=-1).mean()
            l_cos = 0.5 * ((1.0 - cos_nat) + (1.0 - cos_brg))
            l_mse = 0.5 * (
                F.mse_loss(F.normalize(z_nat_sel, dim=-1), F.normalize(e_sel, dim=-1))
                + F.mse_loss(F.normalize(z_brg_sel, dim=-1), F.normalize(e_sel, dim=-1))
            )
            l_align = l_cos + 0.2 * l_mse
        else:
            l_align = 0.0 * (z_native.sum() + z_bridge.sum())
            cos_nat = torch.zeros((), device=input_ids.device)
            cos_brg = torch.zeros((), device=input_ids.device)

        if alpha > 0.0 and valid_latent.any():
            m_3d = valid_latent.unsqueeze(-1).float()
            mixed_embeds = token_embeds * (1.0 - alpha * m_3d) + z_combined * (alpha * m_3d)
        else:
            mixed_embeds = token_embeds + 0.0 * z_combined

        logits, _ = self.run_backbone(mixed_embeds)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        l_ntp = F.cross_entropy(shift_logits.view(-1, V_K31), shift_labels.view(-1), ignore_index=-100)
        loss = l_ntp + beta * l_align

        with torch.no_grad():
            preds = shift_logits.argmax(dim=-1)
            sp_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
            for sp_id in SPECIAL_IDS:
                sp_mask |= shift_labels == sp_id
            sp_acc = (preds[sp_mask] == shift_labels[sp_mask]).float().mean() if sp_mask.any() else torch.tensor(1.0, device=input_ids.device)

        return {
            "loss": loss,
            "l_ntp": l_ntp.detach(),
            "l_align": l_align.detach(),
            "cos_nat": cos_nat.detach(),
            "cos_brg": cos_brg.detach(),
            "special_acc": sp_acc.detach(),
        }


@torch.no_grad()
def evaluate_k31_hybridsft(model: KimiK31HybridSFTTrainableModule, loader: DataLoader, device: torch.device, alpha: float, beta: float) -> dict[str, float]:
    model.eval()
    tot_loss = tot_ntp = tot_align = tot_cnat = tot_cbrg = tot_sp = 0.0
    n = 0
    for batch in loader:
        inp = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        msk = batch["latent_mask"].to(device)
        out = model(inp, lbl, msk, alpha=alpha, beta=beta)
        tot_loss += float(out["loss"].item())
        tot_ntp += float(out["l_ntp"].item())
        tot_align += float(out["l_align"].item())
        tot_cnat += float(out["cos_nat"].item())
        tot_cbrg += float(out["cos_brg"].item())
        tot_sp += float(out["special_acc"].item())
        n += 1
    model.train()
    n = max(1, n)
    avg_ntp = tot_ntp / n
    return {
        "eval_loss": round(tot_loss / n, 4),
        "eval_ntp": round(avg_ntp, 4),
        "eval_ppl": round(math.exp(min(20.0, avg_ntp)), 2),
        "eval_align": round(tot_align / n, 4),
        "eval_cos_nat": round(tot_cnat / n, 4),
        "eval_cos_brg": round(tot_cbrg / n, 4),
        "eval_special_acc": round(tot_sp / n, 4),
    }


def run_stage(
    stage_name: str,
    ddp_model: nn.Module,
    raw_module: KimiK31HybridSFTTrainableModule,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    sampler: DistributedSampler | None,
    device: torch.device,
    rank: int,
    world_size: int,
    steps: int,
    lr: float,
    alpha_max: float,
    beta: float,
    warmup_ratio: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    first_metrics = last_metrics = None
    data_iter = iter(train_loader)
    epoch = 0

    for step in range(1, steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        if warmup_ratio > 0 and step <= int(steps * warmup_ratio):
            alpha = alpha_max * (step / max(1, int(steps * warmup_ratio)))
        else:
            alpha = alpha_max

        cos_progress = (step - 1) / max(1, steps - 1)
        cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * cos_progress)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        out = ddp_model(
            batch["input_ids"].to(device),
            batch["labels"].to(device),
            batch["latent_mask"].to(device),
            alpha=alpha,
            beta=beta,
        )
        out["loss"].backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item())
        opt.step()

        step_info = {
            "step": step,
            "alpha": round(alpha, 3),
            "loss": round(float(out["loss"].item()), 4),
            "l_ntp": round(float(out["l_ntp"].item()), 4),
            "l_align": round(float(out["l_align"].item()), 4),
            "cos_nat": round(float(out["cos_nat"].item()), 4),
            "cos_brg": round(float(out["cos_brg"].item()), 4),
            "special_acc": round(float(out["special_acc"].item()), 4),
            "grad_norm": round(gnorm, 4),
            "lr": round(cur_lr, 6),
        }
        if step == 1:
            first_metrics = step_info
        last_metrics = step_info
        if rank == 0 and (step == 1 or step % 10 == 0 or step == steps):
            print(f"[{stage_name}] step {step:3d}/{steps} | loss={step_info['loss']:.4f} ntp={step_info['l_ntp']:.4f} cos_nat={step_info['cos_nat']:.4f} cos_brg={step_info['cos_brg']:.4f} sp_acc={step_info['special_acc']:.3f}", flush=True)

    eval_res = evaluate_k31_hybridsft(raw_module, eval_loader, device, alpha=alpha_max, beta=beta)
    return {
        "stage": stage_name,
        "steps": steps,
        "world_size_gpus": world_size,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "first_step": first_metrics,
        "last_step": last_metrics,
        "eval": eval_res,
        "out_scale_k31_native": round(float(raw_module.k31_native_hybrid_link.out_scale.item()), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="8-GPU 3-Stage HybridSFT Trainer for Kimi K3.1 Pro")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--stage1-steps", type=int, default=40)
    parser.add_argument("--stage2-steps", type=int, default=60)
    parser.add_argument("--stage3-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_ddp = world_size > 1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    pruned_dir = args.work_dir / "pruned"
    tok_path = args.work_dir / "tokenizer" / "tiktoken.model"
    data_dir = args.work_dir / "hybridsft-data"
    train_jsonl = data_dir / "train.jsonl"
    eval_jsonl = data_dir / "eval.jsonl"
    out_dir = args.work_dir / "hybridagent-8gpu"

    if local_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        build_kimi_k31_hybridsft_dataset(tok_path, train_jsonl, eval_jsonl)
    if is_ddp:
        dist.barrier()

    train_ds = KimiK31HybridSFTDataset(train_jsonl)
    eval_ds = KimiK31HybridSFTDataset(eval_jsonl)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True) if is_ddp else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), collate_fn=collate_k31_hybridsft)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_k31_hybridsft)

    backbone = None
    for r in range(world_size):
        if local_rank == r:
            backbone = KimiK31GPUStageModel(pruned_dir, max_experts_on_gpu=8).to(device)
        if is_ddp:
            dist.barrier()
    assert backbone is not None
    raw_module = KimiK31HybridSFTTrainableModule(backbone).to(device)
    ddp_model = DDP(raw_module, device_ids=[local_rank], find_unused_parameters=True) if is_ddp else raw_module

    pre_eval = evaluate_k31_hybridsft(raw_module, eval_loader, device, alpha=0.0, beta=0.05)
    if local_rank == 0:
        print(f"[Kimi-K3.1-Pro HybridSFT] Pre-training eval: {pre_eval}", flush=True)

    s1 = run_stage("stage1_format_activation", ddp_model, raw_module, train_loader, eval_loader, sampler, device, local_rank, world_size, args.stage1_steps, 3e-4, 0.0, 0.05, 0.0)
    s2 = run_stage("stage2_hybridlink_activation", ddp_model, raw_module, train_loader, eval_loader, sampler, device, local_rank, world_size, args.stage2_steps, 2.4e-4, 1.0, 0.15, 0.15)
    s3 = run_stage("stage3_self_conditioned", ddp_model, raw_module, train_loader, eval_loader, sampler, device, local_rank, world_size, args.stage3_steps, 1.5e-4, 1.0, 0.10, 0.0)

    if local_rank == 0:
        ckpt = {
            "model_id": "moonshotai/Kimi-K3.1-Pro",
            "hidden_size": D_K31,
            "vocab_size": V_K31,
            "special_ids": SPECIAL_IDS,
            "k31_native_hybrid_link": raw_module.k31_native_hybrid_link.state_dict(),
            "qwen_to_k31_bridge": raw_module.qwen_to_k31_bridge.state_dict(),
            "special_emb_delta": raw_module.special_emb_delta.detach().cpu(),
            "special_lm_delta": raw_module.special_lm_delta.detach().cpu(),
            "norm_f_delta": raw_module.norm_f_delta.detach().cpu(),
        }
        torch.save(ckpt, out_dir / "kimi_k31_hybridagent_final.pt")
        summary = {
            "model_id": "moonshotai/Kimi-K3.1-Pro",
            "pre_train_eval": pre_eval,
            "stage1_format_activation": s1,
            "stage2_hybridlink_activation": s2,
            "stage3_self_conditioned": s3,
        }
        (out_dir / "kimi_k31_hybridagent_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[Kimi-K3.1-Pro HybridSFT] Saved checkpoint to {out_dir / 'kimi_k31_hybridagent_final.pt'}", flush=True)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
