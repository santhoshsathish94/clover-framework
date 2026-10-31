#!/usr/bin/env python3
"""
GPU End-to-End Trainer for NVIDIA Nemotron 3.5 Super VL (`d=4096`, `V=131072` Tekken)
HybridAgent + 11-Level Next-Concept Model (`NCM`) + Lexical Pointer Decoder
(`clover-ai-system/src/models/nemotron-3.5-super-vl/train_ncm_nemotron_gpu.py`).

Executes on the Vast.ai 8x RTX 3090 GPU Server:
  - Phase 0: Extracts real `4,096`-d Nemotron hidden states (`h_win [1024, 64, 4096]` and
    `h_plan [1024, 48, 4096]`) on GPU across 1,024 multi-domain 64-token windows (`65,536` tokens:
    XSum, WMT14, TinyStories, technical/math/code/RAG/security ontology).
  - Phase 1: Warm-starts `NemotronNativeHybridLink(d=4096, bottleneck=1024)` and `HybridLink(d=4096)`
    from the 8-GPU 3-stage HybridSFT checkpoint (`nemotron_hybridagent_final.pt`) and refines across
    all 1,024 64-token windows (`1.0 * cos_loss + 0.2 * mse_loss`).
  - Phase 2: Fine-tunes the `131,072`-vocab 11-level `VQVAETokenizer` (`vqvae`) on GPU for 400 steps
    (`batch_size=32`, `2,048` tokens/step = `819,200` token reconstructions).
  - Phase 3: Jointly trains `NemotronComboBridge(4096 -> 384)` + `NemotronExpertBridge(4096 -> 384)` +
    11-level `NextConceptTransformer (net)` + `K3LexicalPointerDecoder (ptr_dec)` on GPU for 500 steps
    with cosine warmup schedule, level-weighted concept code CE, straight-through + 20% code-corruption
    regularization, and 131,072-vocab pointer-generator token CE.
  - Phase 4: Verifies held-out metrics and exports unified `nemotron_hybrid_ncm_bridges.pt` +
    `nemotron_ncm_core_bf16.pt`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/dev/shm/clover-train/ncm/repo")
sys.path.insert(0, "/dev/shm/clover-train/ncm")
sys.path.insert(0, "/dev/shm/clover-train/lfm3-hybridagent/HybridAgent")

from hybrid_agent.hybridlink import HybridLink
from models.next_concept import NextConceptTransformer
from nemotron_gpu_backbone import NemotronGPUStageModel
from nemotron_hybrid_ncm import (
    D_NEMO,
    D_VQ,
    V_NEMO,
    NemotronComboBridge,
    NemotronExpertBridge,
    NemotronNativeHybridLink,
)
from tokenization.vqvae_tokenizer import VQVAETokenizer

P_RH = 48
P_BPE = 48
T_WIN = 64
PAD_ID = 0
EOS_ID = 2
SEMICOLON_ID = 1059


class PointerCrossBlock(nn.Module):
    def __init__(self, d: int = 384, n_head: int = 6):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_head, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ln_mem = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_head, dropout=0.0, batch_first=True)
        self.ln3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, mem: torch.Tensor, mem_key_padding_mask: torch.Tensor) -> torch.Tensor:
        q1 = self.ln1(x)
        x = x + self.self_attn(q1, q1, q1, need_weights=False)[0]
        x = x + self.cross_attn(
            self.ln2(x),
            self.ln_mem(mem),
            self.ln_mem(mem),
            key_padding_mask=mem_key_padding_mask,
            need_weights=False,
        )[0]
        x = x + self.mlp(self.ln3(x))
        return x


class NemotronLexicalPointerDecoder(nn.Module):
    """
    Exact architectural match to `K3LexicalPointerDecoder` in `benchmark_3_models_70_tasks.py`,
    operating over Nemotron's `V_NEMO = 131,072` Tekken vocabulary (`pad_id=0`, `semicolon_id=1059`).
    """

    def __init__(self, d: int = 384, n_layer: int = 3, n_head: int = 6):
        super().__init__()
        self.tgt_pos = nn.Embedding(T_WIN, d)
        self.plan_pos = nn.Embedding(P_BPE, d)
        self.rh_pos = nn.Embedding(P_RH, d)
        self.blocks = nn.ModuleList([PointerCrossBlock(d, n_head) for _ in range(n_layer)])
        self.ln_out = nn.LayerNorm(d)
        self.res_scale = nn.Parameter(torch.tensor(0.25))
        self.copy_q = nn.Linear(d, d, bias=False)
        self.copy_k = nn.Linear(d, d, bias=False)
        self.copy_gate = nn.Sequential(nn.Linear(2 * d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.copy_scale = nn.Parameter(torch.tensor(10.0))

        with torch.no_grad():
            nn.init.eye_(self.copy_q.weight)
            nn.init.eye_(self.copy_k.weight)
            self.copy_gate[-1].bias.fill_(1.5)

    def forward(
        self,
        z_hat: torch.Tensor,
        z_rh: torch.Tensor,
        msk_rh: torch.Tensor,
        z_bpe: torch.Tensor,
        p_pad: torch.Tensor,
        vqvae_decoder: nn.Module,
        pad_id: int = PAD_ID,
        semicolon_id: int = SEMICOLON_ID,
    ) -> torch.Tensor:
        B, T, C = z_hat.shape
        dev = z_hat.device
        msk_bpe = p_pad == pad_id
        t_pos = self.tgt_pos(torch.arange(T, device=dev))[None, :, :]
        p_pos = self.plan_pos(torch.arange(P_BPE, device=dev))[None, :, :]
        r_pos = self.rh_pos(torch.arange(P_RH, device=dev))[None, :, :]

        mem_bpe = z_bpe + p_pos
        mem_rh = z_rh + r_pos
        mem = torch.cat([mem_rh, mem_bpe], dim=1)
        mem_mask = torch.cat([msk_rh, msk_bpe], dim=1)

        x = z_hat + t_pos
        for blk in self.blocks:
            x = blk(x, mem, mem_mask)
        z_refined = z_hat + self.res_scale * self.ln_out(x)
        base_logits = vqvae_decoder(z_refined)  # [B, 64, 131072]

        cq = F.normalize(self.copy_q(x), dim=-1)
        ck = F.normalize(self.copy_k(mem_bpe), dim=-1)
        sim = torch.bmm(cq, ck.transpose(1, 2))

        p_len = (~msk_bpe).sum(dim=1, keepdim=True).clamp(min=1).float()
        has_semi = (p_pad == semicolon_id).any(dim=1, keepdim=True)
        expected_T = torch.where(has_semi, (p_len * 1.25 + 2.0).clamp(min=4.0, max=float(T)), p_len)
        t_pos_idx = torch.arange(T, device=dev, dtype=torch.float32)[None, :, None]
        p_pos_idx = torch.arange(P_BPE, device=dev, dtype=torch.float32)[None, None, :]

        diag_soft = -2.5 * ((t_pos_idx / expected_T[:, :, None]) - (p_pos_idx / p_len[:, :, None])).pow(2)
        diag_clause = -45.0 * (t_pos_idx - p_pos_idx).pow(2)
        diag_bias = torch.where(has_semi[:, :, None], diag_soft, diag_clause)
        sim = (sim * 12.0 + diag_bias).masked_fill(msk_bpe[:, None, :], -1e4)
        attn_probs = F.softmax(sim, dim=-1)
        ctx_vec = torch.bmm(attn_probs, z_bpe)

        g_copy = torch.sigmoid(self.copy_gate(torch.cat([x, ctx_vec], dim=-1)))
        soft_scale = g_copy * self.copy_scale.clamp(min=0.0, max=16.0)
        in_bounds = t_pos_idx < p_len[:, :, None]
        clause_scale = torch.where(in_bounds, soft_scale + 65.0, torch.zeros_like(soft_scale))
        eff_scale = torch.where(has_semi[:, :, None], soft_scale, clause_scale)
        copy_boost = attn_probs * eff_scale

        idx_exp = p_pad[:, None, :].expand(B, T, P_BPE)
        out_logits = base_logits.clone()
        out_logits.scatter_add_(2, idx_exp, copy_boost.to(out_logits.dtype))
        return out_logits


def collect_multidomain_sentences(ncm_data_dir: Path, target_count: int = 1024) -> list[str]:
    """Collects 1,024 diverse multi-domain sentences/windows across XSum, WMT14, TinyStories, and Clover technical/RAG/math/code domains."""
    rng = random.Random(42)
    sentences: list[str] = []

    # 1. Domain-rich technical, scientific, RAG, security ontology, and benchmark-style sentences
    domain_Templates = [
        "Partitioning the expert cache per layer instead of using a single global table raised the cache hit rate from 23.3% to 48.0% inside the same 51.7 GB host RAM budget.",
        "CVE-2024-21762 is an out-of-bounds write vulnerability in Fortinet FortiOS SSL VPN caused by CWE-787 and exploited via CAPEC-100 buffer overflow.",
        "Before opening subsea equalization valve V-402, equalize differential pressure across the manifold and verify downstream telemetry readings.",
        "Harbourside Council voted 7 to 2 on Tuesday to approve 2.4 million pounds for a new segregated cycle lane along Mill Road.",
        "NVIDIA Nemotron 3.5 Super VL combines Mamba-2 selective state spaces, 512-expert Latent ReLU-squared Mixture of Experts with top-22 routing, and NoPE 16:1 GQA attention.",
        "SpecMD Least-Stale slot caching paired with Red-Blue Pebble Rule R4 clean-drop eliminates dirty PCIe writebacks for read-only expert weights.",
        "Next-Concept Modeling predicts 11 hierarchical vector-quantized concept levels in parallel before decoding surface tokens through the lexical pointer network.",
        "Using Python, 17**4 + 29*13 = 83521 + 377 = 83898.",
        "Recent research papers on transformer memory compression include KVQuant (arXiv:2401.18079, 2024) and Gear (arXiv:2403.05527, 2024).",
        "The image shows a 64x64 synthetic diagnostic test pattern available in PNG, JPG, WebP, and BMP formats.",
        "Audio duration is 1.50 seconds at 16,000 Hz mono PCM containing a 440 Hz A4 sine wave tone.",
        "In the Gaia DR3 stellar catalog, approximately 2,800 spectral stars satisfy the radial-velocity signal-to-noise threshold.",
        "To prevent cross-prompt contamination during clubbed batch inference, per-prompt segment start offsets isolate both Mamba-2 recurrent states and causal GQA attention masks.",
        "HybridAgent switches dynamically between discrete token generation and continuous latent reasoning spans bounded by <latent> and </latent> anchor tokens.",
    ]
    for i in range(160):
        sentences.append(domain_Templates[i % len(domain_Templates)])

    # 2. XSum summaries
    xsum_p = ncm_data_dir / "xsum" / "reference_summaries.pickle"
    if xsum_p.exists():
        with open(xsum_p, "rb") as f:
            xsum_obj = pickle.load(f)
        xsum_list = list(xsum_obj.values()) if isinstance(xsum_obj, dict) else list(xsum_obj)
        rng.shuffle(xsum_list)
        for s in xsum_list[:350]:
            s_clean = str(s).strip()
            if 20 <= len(s_clean) <= 320:
                sentences.append(s_clean)

    # 3. WMT14 translations
    wmt_p = ncm_data_dir / "wmt14" / "reference_translations.pickle"
    if wmt_p.exists():
        with open(wmt_p, "rb") as f:
            wmt_obj = pickle.load(f)
        wmt_list = list(wmt_obj.values()) if isinstance(wmt_obj, dict) else list(wmt_obj)
        rng.shuffle(wmt_list)
        for s in wmt_list[:300]:
            s_clean = str(s).strip()
            if 20 <= len(s_clean) <= 320:
                sentences.append(s_clean)

    # 4. TinyStories narrative sentences
    ts_p = ncm_data_dir / "TinyStoriesV2-GPT4-valid.txt"
    if ts_p.exists():
        with open(ts_p, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = [ln.strip() for ln in f.readlines()[:3000] if 25 <= len(ln.strip()) <= 280]
        rng.shuffle(raw_lines)
        sentences.extend(raw_lines[:300])

    rng.shuffle(sentences)
    while len(sentences) < target_count:
        sentences.extend(sentences[: target_count - len(sentences)])
    return sentences[:target_count]


def build_plan_and_target_windows(
    tokenizer: Any,
    sentences: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds aligned `(tgt_ids [N, 64], p_pad [N, 48], p_mask [N, 48])` in Nemotron's `131,072` Tekken vocabulary!
    For 70% of windows, `p_pad` is the exact clause window (`<= 42` tokens) so `ptr_dec` learns 1-to-1 clause
    rendering (`has_semi = False`). For 30% of windows, `p_pad` is a semicolon-separated key-phrase plan
    (`has_semi = True`) so `net` and `ptr_dec` also learn compressed plan expansion!
    """
    N = len(sentences)
    tgt_ids = torch.full((N, T_WIN), PAD_ID, dtype=torch.long)
    p_pad = torch.full((N, P_BPE), PAD_ID, dtype=torch.long)

    stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "by", "with", "is", "was", "were"}

    for i, s in enumerate(sentences):
        ids = tokenizer.encode(s, add_special_tokens=False)[:42]
        if not ids:
            ids = [100]

        # Target 64-token window: ids + EOS_ID (2) + PAD_ID (0)
        t_seq = ids + [EOS_ID]
        t_seq = t_seq[:T_WIN]
        tgt_ids[i, : len(t_seq)] = torch.tensor(t_seq, dtype=torch.long)

        if i % 10 < 7:
            # Clause mode (exact tokens in p_pad)
            p_seq = ids[:P_BPE]
        else:
            # Semicolon key-phrase plan mode
            words = s.split()
            phrases: list[str] = []
            cur: list[str] = []
            for w in words:
                if w.lower().strip(".,;:") in stop_words and cur:
                    phrases.append(" ".join(cur))
                    cur = []
                else:
                    cur.append(w)
            if cur:
                phrases.append(" ".join(cur))
            plan_str = " ; ".join(phrases[:6]) if phrases else s
            p_seq = tokenizer.encode(plan_str, add_special_tokens=False)[:P_BPE]
            if SEMICOLON_ID not in p_seq and len(p_seq) < P_BPE:
                p_seq = p_seq[: P_BPE - 1] + [SEMICOLON_ID]

        p_pad[i, : len(p_seq)] = torch.tensor(p_seq, dtype=torch.long)

    p_mask = p_pad == PAD_ID
    return tgt_ids, p_pad, p_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="/dev/shm/clover-train/nemotron/model_meta",
    )
    parser.add_argument(
        "--pruned-dir",
        type=str,
        default="/dev/shm/clover-train/nemotron/pruned",
    )
    parser.add_argument(
        "--wmt14-dir",
        type=str,
        default="/dev/shm/clover-train/ncm/artifacts/wmt14",
    )
    parser.add_argument(
        "--ncm-data-dir",
        type=str,
        default="/dev/shm/clover-train/ncm/data",
    )
    parser.add_argument(
        "--init-bridges-pt",
        type=str,
        default="/dev/shm/clover-train/nemotron/hybrid-ncm/nemotron_hybrid_ncm_bridges.pt",
    )
    parser.add_argument(
        "--hybridagent-ckpt",
        type=str,
        default="/dev/shm/clover-train/nemotron/checkpoints/hybridagent/nemotron_hybridagent_final.pt",
    )
    parser.add_argument(
        "--ptr-dec-init-pt",
        type=str,
        default="/dev/shm/clover-train/ncm/ptr_dec_init.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/dev/shm/clover-train/nemotron/hybrid-ncm",
    )
    parser.add_argument("--num-windows", type=int, default=1024)
    parser.add_argument("--hybridlink-steps", type=int, default=300)
    parser.add_argument("--vqvae-steps", type=int, default=400)
    parser.add_argument("--ncm-steps", type=int, default=500)
    args = parser.parse_args()

    t_start = time.perf_counter()
    device = torch.device("cuda:0")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Nemotron-NCM-GPU] Phase 0/4: Loading Tekken tokenizer & multi-domain corpus...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    sentences = collect_multidomain_sentences(Path(args.ncm_data_dir), target_count=args.num_windows)
    tgt_ids, p_pad, p_mask = build_plan_and_target_windows(tokenizer, sentences)

    # Split into 960 train windows + 64 held-out eval windows
    n_eval = 64
    n_train = args.num_windows - n_eval

    ha_ckpt = None
    if os.path.exists(args.hybridagent_ckpt):
        ha_ckpt = torch.load(args.hybridagent_ckpt, map_location="cpu", weights_only=False)
        print(f"[Nemotron-NCM-GPU] Loaded 8-GPU 3-stage HybridAgent checkpoint from {args.hybridagent_ckpt}", flush=True)

    cache_pt = out_dir / "ncm_cached_hidden_1024.pt"
    if cache_pt.exists():
        print(f"[Nemotron-NCM-GPU] Phase 0/4: Loading cached 4096-d Nemotron hidden states from {cache_pt}...", flush=True)
        c_obj = torch.load(cache_pt, map_location="cpu", weights_only=False)
        h_win_all = c_obj["h_win_all"]
        h_plan_all = c_obj["h_plan_all"]
        emb_weight_cpu = c_obj["emb_weight_cpu"]
    else:
        print(
            f"[Nemotron-NCM-GPU] Phase 0/4: Extracting 4096-d Nemotron hidden states on RTX 3090 across {args.num_windows} windows...",
            flush=True,
        )
        backbone = NemotronGPUStageModel(pruned_root=Path(args.pruned_dir), experts_on_gpu=True).to(device)
        backbone.eval()

        h_win_all = torch.empty((args.num_windows, T_WIN, D_NEMO), dtype=torch.bfloat16)
        h_plan_all = torch.empty((args.num_windows, P_BPE, D_NEMO), dtype=torch.bfloat16)
        emb_weight_gpu = backbone.embed_tokens.weight.detach()  # [131072, 4096] bf16

        with torch.no_grad():
            bs = 8
            for st in range(0, args.num_windows, bs):
                ed = min(args.num_windows, st + bs)
                b_tgt = tgt_ids[st:ed].to(device)
                b_plan = p_pad[st:ed].to(device)
                out_tgt = backbone(input_ids=b_tgt, output_hidden_states=True)
                out_plan = backbone(input_ids=b_plan, output_hidden_states=True)
                h_win_all[st:ed] = out_tgt.hidden_states[-1].detach().cpu()
                h_plan_all[st:ed] = out_plan.hidden_states[-1].detach().cpu()

        emb_weight_cpu = emb_weight_gpu.cpu().clone()
        del backbone
        torch.cuda.empty_cache()
        torch.save(
            {"h_win_all": h_win_all, "h_plan_all": h_plan_all, "emb_weight_cpu": emb_weight_cpu},
            cache_pt,
        )
    print(
        f"[Nemotron-NCM-GPU] Ready h_win={tuple(h_win_all.shape)} & h_plan={tuple(h_plan_all.shape)} ({time.perf_counter() - t_start:.1f}s)",
        flush=True,
    )

    # =========================================================================
    # Phase 1: Refine NemotronNativeHybridLink(d=4096, bottleneck=1024) + HybridLink(4096) on GPU
    # =========================================================================
    print(
        f"\n[Nemotron-NCM-GPU] Phase 1/4: Training NemotronNativeHybridLink(4096->1024->4096) + HybridLink(4096) for {args.hybridlink_steps} steps...",
        flush=True,
    )
    nemo_native_hl = NemotronNativeHybridLink(d=D_NEMO, bottleneck=1024).to(device)
    std_hl = HybridLink(D_NEMO).to(device)

    init_sd = torch.load(args.init_bridges_pt, map_location="cpu", weights_only=False)
    if ha_ckpt is not None:
        nemo_native_hl.load_state_dict(ha_ckpt["nemo_native_hybrid_link"], strict=False)
        std_hl.load_state_dict(ha_ckpt["hybrid_link"], strict=False)
    elif "nemo_native_hybrid_link" in init_sd:
        nemo_native_hl.load_state_dict(init_sd["nemo_native_hybrid_link"], strict=False)

    emb_w_dev = emb_weight_cpu.to(device)

    @torch.no_grad()
    def eval_hybridlink() -> dict[str, float]:
        nemo_native_hl.eval()
        std_hl.eval()
        h_ev = h_win_all[n_train:].to(device).float()
        t_ev = tgt_ids[n_train:].to(device)
        valid_m = (t_ev[:, 1:] != PAD_ID) & (t_ev[:, 1:] != EOS_ID)
        h_src = h_ev[:, :-1][valid_m]
        e_tgt = emb_w_dev[t_ev[:, 1:][valid_m]].float()
        pred_nat = nemo_native_hl(h_src)
        pred_std = std_hl(h_src)
        cos_nat = float(F.cosine_similarity(pred_nat, e_tgt, dim=-1).mean().item())
        cos_std = float(F.cosine_similarity(pred_std, e_tgt, dim=-1).mean().item())
        nemo_native_hl.train()
        std_hl.train()
        return {"eval_cos_native": round(cos_nat, 4), "eval_cos_std": round(cos_std, 4)}

    hl_before = eval_hybridlink()
    print(f"[Nemotron-NCM-GPU] HybridLink pre-Phase1 eval: {hl_before}", flush=True)

    hl_opt = torch.optim.AdamW(
        list(nemo_native_hl.parameters()) + list(std_hl.parameters()),
        lr=5e-4,
        weight_decay=0.01,
    )
    hl_sched = torch.optim.lr_scheduler.CosineAnnealingLR(hl_opt, T_max=args.hybridlink_steps, eta_min=2e-5)

    for step in range(1, args.hybridlink_steps + 1):
        idx = torch.randint(0, n_train, (64,))
        h_b = h_win_all[idx].to(device).float()
        t_b = tgt_ids[idx].to(device)
        valid_m = t_b[:, 1:] != PAD_ID
        h_src = h_b[:, :-1][valid_m]
        e_tgt = emb_w_dev[t_b[:, 1:][valid_m]].float()

        p_nat = nemo_native_hl(h_src)
        p_std = std_hl(h_src)

        l_cos_nat = 1.0 - F.cosine_similarity(p_nat, e_tgt, dim=-1).mean()
        l_mse_nat = F.mse_loss(p_nat, e_tgt)
        l_cos_std = 1.0 - F.cosine_similarity(p_std, e_tgt, dim=-1).mean()
        l_mse_std = F.mse_loss(F.normalize(p_std, dim=-1), F.normalize(e_tgt, dim=-1))

        loss_hl = (l_cos_nat + 0.2 * l_mse_nat) + (l_cos_std + 0.2 * l_mse_std)
        hl_opt.zero_grad(set_to_none=True)
        loss_hl.backward()
        torch.nn.utils.clip_grad_norm_(list(nemo_native_hl.parameters()) + list(std_hl.parameters()), 1.0)
        hl_opt.step()
        hl_sched.step()

        if step == 1 or step % 50 == 0 or step == args.hybridlink_steps:
            print(
                f"  [Phase 1 HybridLink] step {step:3d}/{args.hybridlink_steps} | "
                f"cos_nat={1.0 - l_cos_nat.item():.4f} | mse_nat={l_mse_nat.item():.5f} | "
                f"cos_std={1.0 - l_cos_std.item():.4f}",
                flush=True,
            )

    hl_after = eval_hybridlink()
    print(f"[Nemotron-NCM-GPU] HybridLink post-Phase1 eval: {hl_after}", flush=True)
    nemo_native_hl.eval()
    std_hl.eval()
    del emb_w_dev
    torch.cuda.empty_cache()

    # =========================================================================
    # Phase 2: Fine-Tune 11-Level VQVAETokenizer (V_NEMO = 131,072) on GPU
    # =========================================================================
    print(
        f"\n[Nemotron-NCM-GPU] Phase 2/4: Training 131,072-vocab 11-Level VQ-VAE on GPU for {args.vqvae_steps} steps...",
        flush=True,
    )
    wmt14_dir = Path(args.wmt14_dir)
    cfg = OmegaConf.load(str(wmt14_dir / "config.yaml"))
    cfg.tokenizer.vqvae.checkpoint_path = str(wmt14_dir / "tokenizer.pt")
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    vqvae = tk.vqvae.cpu()
    vqvae.in_vocab_size = V_NEMO
    vqvae.encoder.wte = nn.Embedding(V_NEMO, D_VQ)
    vqvae.decoder.output_layer = nn.Linear(D_VQ, V_NEMO, bias=False)
    vqvae.load_state_dict(init_sd["vqvae"])
    vqvae = vqvae.to(device)

    @torch.no_grad()
    def eval_vqvae() -> dict[str, float]:
        vqvae.eval()
        t_ev = tgt_ids[n_train:].to(device)
        idx_ev = vqvae.encode(t_ev)
        logits = vqvae.decode(idx_ev)
        preds = logits.argmax(dim=-1)
        non_pad = t_ev != PAD_ID
        acc_nonpad = float((preds[non_pad] == t_ev[non_pad]).float().mean().item())
        acc_all = float((preds == t_ev).float().mean().item())
        ce = float(F.cross_entropy(logits.float().view(-1, V_NEMO), t_ev.view(-1)).item())
        return {
            "vq_eval_ce": round(ce, 4),
            "vq_eval_acc_nonpad": round(acc_nonpad, 4),
            "vq_eval_acc_all": round(acc_all, 4),
        }

    vq_before = eval_vqvae()
    print(f"[Nemotron-NCM-GPU] VQ-VAE pre-Phase2 eval: {vq_before}", flush=True)

    for name, p in vqvae.named_parameters():
        p.requires_grad_("vq." not in name)
    vqvae.train()
    vqvae.vq.eval()
    vq_opt = torch.optim.AdamW([p for p in vqvae.parameters() if p.requires_grad], lr=4e-4, weight_decay=0.01, fused=True)
    vq_sched = torch.optim.lr_scheduler.CosineAnnealingLR(vq_opt, T_max=args.vqvae_steps, eta_min=2e-5)

    for step in range(1, args.vqvae_steps + 1):
        idx = torch.randint(0, n_train, (24,))
        t_b = tgt_ids[idx].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, commit_loss, _, _, _ = vqvae(t_b)
            ce_per_tok = F.cross_entropy(logits.float().view(-1, V_NEMO), t_b.view(-1), reduction="none").view_as(t_b)
            tok_weights = torch.where(t_b != PAD_ID, 4.0, 0.25)
            ce_loss = (ce_per_tok * tok_weights).sum() / tok_weights.sum()
            loss_vq = ce_loss + 0.25 * commit_loss.mean()

        vq_opt.zero_grad(set_to_none=True)
        loss_vq.backward()
        torch.nn.utils.clip_grad_norm_([p for p in vqvae.parameters() if p.requires_grad], 1.0)
        vq_opt.step()
        vq_sched.step()

        if step == 1 or step % 50 == 0 or step == args.vqvae_steps:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                non_pad = t_b != PAD_ID
                acc_np = float((preds[non_pad] == t_b[non_pad]).float().mean().item())
            print(
                f"  [Phase 2 VQ-VAE] step {step:3d}/{args.vqvae_steps} | "
                f"loss={loss_vq.item():.4f} (ce={ce_loss.item():.4f}) | acc_nonpad={acc_np:.4f}",
                flush=True,
            )

    vq_after = eval_vqvae()
    print(f"[Nemotron-NCM-GPU] VQ-VAE post-Phase2 eval: {vq_after}", flush=True)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    # Precompute ground-truth 11-level VQ codes `idx_all_t [1024, 313]` and `z_clean [1024, 64, 384]`
    # and `r_plan_all [1024, 48, 4096]` across all 1,024 windows!
    print("[Nemotron-NCM-GPU] Precomputing 11-level VQ target codes & R_Nemo(h_plan) for Phase 3...", flush=True)
    idx_chunks = []
    z_clean_all = torch.empty((args.num_windows, T_WIN, D_VQ), dtype=torch.float32)
    r_plan_all = torch.empty((args.num_windows, P_BPE, D_NEMO), dtype=torch.float32)

    with torch.no_grad():
        for st in range(0, args.num_windows, 64):
            ed = min(args.num_windows, st + 64)
            b_t = tgt_ids[st:ed].to(device)
            b_idx = vqvae.encode(b_t)
            idx_chunks.append(b_idx.cpu())
            zh_levels, _ = vqvae.vq.idx_to_zhats(b_idx, return_zlevels=True)
            z_clean_all[st:ed] = zh_levels[-1].float().cpu()

            b_hp = h_plan_all[st:ed].to(device).float()
            r_plan_all[st:ed] = nemo_native_hl(b_hp).float().cpu()

    gt_codes_all = torch.cat(idx_chunks, dim=0)  # [1024, 313]

    # =========================================================================
    # Phase 3: Joint GPU Training of NemotronComboBridge + NemotronExpertBridge +
    #          NextConceptTransformer (net) + NemotronLexicalPointerDecoder (ptr_dec)
    # =========================================================================
    print(
        f"\n[Nemotron-NCM-GPU] Phase 3/4: Jointly training NemotronComboBridge + NemotronExpertBridge + "
        f"NextConceptTransformer (net) + LexicalPointerDecoder (ptr_dec) for {args.ncm_steps} steps...",
        flush=True,
    )
    mcfg = cfg.model
    mcfg.prefix_len = P_RH + P_BPE  # 96
    mcfg.predict_level0 = True
    net = NextConceptTransformer(vqvae_local=vqvae, config=mcfg).to(device)

    gen_pt = wmt14_dir / "generator.pt"
    own_net = net.state_dict()
    if gen_pt.exists():
        gen_sd = torch.load(str(gen_pt), map_location="cpu", weights_only=False)
        if "model" in gen_sd:
            keep_sd = {k: v for k, v in gen_sd["model"].items() if k in own_net and own_net[k].shape == v.shape}
            net.load_state_dict(keep_sd, strict=False)

    ptr_dec = NemotronLexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(device)
    if os.path.exists(args.ptr_dec_init_pt):
        sd_init = torch.load(args.ptr_dec_init_pt, map_location="cpu", weights_only=False)
        if "ptr_dec" in sd_init:
            ptr_dec.load_state_dict(sd_init["ptr_dec"], strict=False)
        if "lm_head_l0" in sd_init:
            keep_top = {k: v.float() for k, v in sd_init["lm_head_l0"].items() if k in own_net and own_net[k].shape == v.shape}
            net.load_state_dict(keep_top, strict=False)

    combo_bridge = NemotronComboBridge(D_NEMO, D_VQ).to(device)
    expert_bridge = NemotronExpertBridge(D_NEMO, D_VQ).to(device)
    if "nemo_combo_bridge" in init_sd:
        combo_bridge.load_state_dict(init_sd["nemo_combo_bridge"], strict=False)
    if "nemo_expert_bridge" in init_sd:
        expert_bridge.load_state_dict(init_sd["nemo_expert_bridge"], strict=False)

    for name, p in net.named_parameters():
        p.requires_grad_(True)

    levels = list(vqvae.vq.levels)
    lid = torch.cat([torch.full((l_k,), k, dtype=torch.long) for k, l_k in enumerate(levels)]).to(device)
    ls = torch.tensor(levels, dtype=torch.float32, device=device)
    w_lvl = (ls ** (-0.7))[lid]

    def corrupt_vq_codes(idx_in: torch.Tensor, p_first: float = 0.35, p_last: float = 0.10) -> torch.Tensor:
        out = idx_in.clone()
        off = 0
        n_in = len(levels) - 1
        for k in range(n_in):
            l_k = levels[k]
            p_k = p_first + (p_last - p_first) * (k / max(1, n_in - 1))
            mask = torch.rand((idx_in.shape[0], l_k), device=idx_in.device) < p_k
            rnd = torch.randint(0, vqvae.vq.vqvae_vocab_sizes[k], (idx_in.shape[0], l_k), device=idx_in.device)
            out[:, off : off + l_k] = torch.where(mask, rnd, out[:, off : off + l_k])
            off += l_k
        return out

    @torch.no_grad()
    def eval_ncm_pipeline() -> dict[str, float]:
        combo_bridge.eval()
        expert_bridge.eval()
        net.eval()
        ptr_dec.eval()

        b_r = r_plan_all[n_train:].to(device)
        b_hp = h_plan_all[n_train:].to(device).float()
        b_ppad = p_pad[n_train:].to(device)
        b_pmask = p_mask[n_train:].to(device)
        b_codes = gt_codes_all[n_train:].to(device)
        b_zclean = z_clean_all[n_train:].to(device)
        b_tgt = tgt_ids[n_train:].to(device)

        z_rh = combo_bridge(b_r).float().masked_fill(b_pmask.unsqueeze(-1), 0.0)
        z_exp = expert_bridge(b_hp).float().masked_fill(b_pmask.unsqueeze(-1), 0.0)
        z_bpe = vqvae.encoder.wte(b_ppad) + z_exp
        z_pref = torch.cat([z_rh, z_bpe], dim=1)
        pmask_cat = torch.cat([b_pmask, b_pmask], dim=1)

        ncm_in = vqvae.vq.idx_to_ncm_input(b_codes)
        lg_ncm = net(torch.cat([z_pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=pmask_cat)
        B_cur, T_c, V_c = lg_ncm.shape
        ce_ncm = F.cross_entropy(lg_ncm.float().reshape(-1, V_c), b_codes.reshape(-1), reduction="none").view(B_cur, T_c)
        code_ce = float(((ce_ncm * w_lvl[None]).sum() / (w_lvl.sum() * B_cur)).item())

        l0_len = levels[0]
        V_0 = vqvae.vq.vqvae_vocab_sizes[0]
        l0_acc = float((lg_ncm[:, :l0_len, :V_0].argmax(dim=-1) == b_codes[:, :l0_len]).float().mean().item())
        all_lvl_acc = float((lg_ncm.argmax(dim=-1) == b_codes).float().mean().item())

        tok_logits = ptr_dec(b_zclean, z_rh, b_pmask, z_bpe, b_ppad, vqvae.decoder)
        non_pad = (b_tgt != PAD_ID) & (b_tgt != EOS_ID)
        tok_acc_nonpad = float((tok_logits.argmax(dim=-1)[non_pad] == b_tgt[non_pad]).float().mean().item())
        tok_ce = float(F.cross_entropy(tok_logits.float().view(-1, V_NEMO), b_tgt.view(-1)).item())

        combo_bridge.train()
        expert_bridge.train()
        net.train()
        ptr_dec.train()
        return {
            "ncm_eval_code_ce": round(code_ce, 4),
            "ncm_eval_level0_acc": round(l0_acc, 4),
            "ncm_eval_all_levels_acc": round(all_lvl_acc, 4),
            "ncm_eval_tok_ce": round(tok_ce, 4),
            "ncm_eval_tok_acc_nonpad": round(tok_acc_nonpad, 4),
        }

    ncm_before = eval_ncm_pipeline()
    print(f"[Nemotron-NCM-GPU] NCM pipeline pre-Phase3 eval: {ncm_before}", flush=True)

    trainable_ncm_params = (
        list(combo_bridge.parameters())
        + list(expert_bridge.parameters())
        + [p for p in net.parameters() if p.requires_grad]
        + list(ptr_dec.parameters())
    )
    ncm_opt = torch.optim.AdamW(
        [
            {"params": combo_bridge.parameters(), "lr": 5e-4},
            {"params": expert_bridge.parameters(), "lr": 5e-4},
            {"params": [p for p in net.parameters() if p.requires_grad], "lr": 8e-5},
            {"params": ptr_dec.parameters(), "lr": 4e-4},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )

    for step in range(1, args.ncm_steps + 1):
        mult = 0.5 * (1.0 + math.cos(math.pi * step / float(args.ncm_steps)))
        ncm_opt.param_groups[0]["lr"] = 5e-4 * mult
        ncm_opt.param_groups[1]["lr"] = 5e-4 * mult
        ncm_opt.param_groups[2]["lr"] = 8e-5 * mult
        ncm_opt.param_groups[3]["lr"] = 4e-4 * mult

        idx = torch.randint(0, n_train, (24,))
        b_r = r_plan_all[idx].to(device)
        b_hp = h_plan_all[idx].to(device).float()
        b_ppad = p_pad[idx].to(device)
        b_pmask = p_mask[idx].to(device)
        b_codes = gt_codes_all[idx].to(device)
        b_zclean = z_clean_all[idx].to(device)
        b_tgt = tgt_ids[idx].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z_rh = combo_bridge(b_r).float().masked_fill(b_pmask.unsqueeze(-1), 0.0)
            z_exp = expert_bridge(b_hp).float().masked_fill(b_pmask.unsqueeze(-1), 0.0)
            z_bpe = vqvae.encoder.wte(b_ppad) + z_exp
            z_pref = torch.cat([z_rh, z_bpe], dim=1)
            pmask_cat = torch.cat([b_pmask, b_pmask], dim=1)

            idx_c = corrupt_vq_codes(b_codes, 0.35, 0.10)
            ncm_in = vqvae.vq.idx_to_ncm_input(idx_c)
            lg_ncm = net(torch.cat([z_pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=pmask_cat)
            B_cur, T_c, V_c = lg_ncm.shape
            ce_ncm = F.cross_entropy(lg_ncm.float().reshape(-1, V_c), b_codes.reshape(-1), reduction="none").view(B_cur, T_c)
            l_code = (ce_ncm * w_lvl[None]).sum() / (w_lvl.sum() * B_cur)

            valid_p = (~b_pmask).float().unsqueeze(-1)
            pref_mean = (0.5 * (z_rh + z_bpe) * valid_p).sum(dim=1) / valid_p.sum(dim=1).clamp(min=1.0)
            tgt_mean = b_zclean.mean(dim=1)
            l_pref = 1.0 - F.cosine_similarity(pref_mean, tgt_mean, dim=-1).mean()

            with torch.no_grad():
                pred_codes = []
                off = 0
                for k, l_k in enumerate(levels):
                    V_k = vqvae.vq.vqvae_vocab_sizes[k]
                    pred_codes.append(lg_ncm[:, off : off + l_k, :V_k].argmax(dim=-1))
                    off += l_k
                g_idx = torch.cat(pred_codes, dim=1)
                zh_levels, _ = vqvae.vq.idx_to_zhats(g_idx, return_zlevels=True)
                z_hat_pred = zh_levels[-1].float()
                drop_z = torch.rand(B_cur, T_WIN, 1, device=device) < 0.20
                z_in_dec = torch.where(drop_z, z_hat_pred, b_zclean)

            tok_logits = ptr_dec(z_in_dec, z_rh, b_pmask, z_bpe, b_ppad, vqvae.decoder)
            ce_tok_raw = F.cross_entropy(tok_logits.float().view(-1, V_NEMO), b_tgt.view(-1), reduction="none").view_as(b_tgt)
            tok_w = torch.where(b_tgt != PAD_ID, 4.0, 0.25)
            l_tok = (ce_tok_raw * tok_w).sum() / tok_w.sum()

            loss_total = l_code + 1.5 * l_tok + 0.25 * l_pref

        ncm_opt.zero_grad(set_to_none=True)
        loss_total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_ncm_params, 1.0).item())
        ncm_opt.step()

        if step == 1 or step % 50 == 0 or step == args.ncm_steps:
            with torch.no_grad():
                non_pad = (b_tgt != PAD_ID) & (b_tgt != EOS_ID)
                acc_t = float((tok_logits.argmax(dim=-1)[non_pad] == b_tgt[non_pad]).float().mean().item())
                l0_len = levels[0]
                V_0 = vqvae.vq.vqvae_vocab_sizes[0]
                l0_a = float((lg_ncm[:, :l0_len, :V_0].argmax(dim=-1) == b_codes[:, :l0_len]).float().mean().item())
            print(
                f"  [Phase 3 NCM+Bridges+Pointer] step {step:3d}/{args.ncm_steps} | "
                f"loss={loss_total.item():.4f} (code={l_code.item():.4f}, tok={l_tok.item():.4f}, pref={l_pref.item():.4f}) | "
                f"l0_acc={l0_a:.4f} | tok_acc={acc_t:.4f} | grad={grad_norm:.3f}",
                flush=True,
            )

    ncm_after = eval_ncm_pipeline()
    print(f"[Nemotron-NCM-GPU] NCM pipeline post-Phase3 eval: {ncm_after}", flush=True)

    # =========================================================================
    # Phase 4: Save Unified Checkpoint (`nemotron_hybrid_ncm_bridges.pt` + `nemotron_ncm_core_bf16.pt`)
    # =========================================================================
    total_elapsed = time.perf_counter() - t_start
    metrics_summary = {
        "model_id": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        "hidden_size": D_NEMO,
        "vocab_size": V_NEMO,
        "vq_dim": D_VQ,
        "num_windows_trained": args.num_windows,
        "hybridagent_3stage_summary": ha_ckpt.get("meta", {}) if ha_ckpt else {},
        "phase1_hybridlink": {"before": hl_before, "after": hl_after},
        "phase2_vqvae": {"before": vq_before, "after": vq_after},
        "phase3_ncm_bridges_pointer": {"before": ncm_before, "after": ncm_after},
        "elapsed_seconds": round(total_elapsed, 2),
    }

    full_payload = {
        "vqvae": {k: v.cpu() for k, v in vqvae.state_dict().items()},
        "nemo_native_hybrid_link": {k: v.cpu() for k, v in nemo_native_hl.state_dict().items()},
        "hybrid_link": {k: v.cpu() for k, v in std_hl.state_dict().items()},
        "nemo_combo_bridge": {k: v.cpu() for k, v in combo_bridge.state_dict().items()},
        "nemo_expert_bridge": {k: v.cpu() for k, v in expert_bridge.state_dict().items()},
        "net": {k: v.cpu() for k, v in net.state_dict().items()},
        "ptr_dec": {k: v.cpu() for k, v in ptr_dec.state_dict().items()},
        "metrics": metrics_summary,
    }
    if ha_ckpt is not None:
        for k_extra in (
            "special_token_ids",
            "special_emb_delta",
            "special_lm_delta",
            "norm_f_delta",
            "adapted_special_emb_bf16",
            "adapted_special_lm_bf16",
            "adapted_norm_f_bf16",
        ):
            if k_extra in ha_ckpt:
                full_payload[k_extra] = ha_ckpt[k_extra]

    out_bridges_pt = out_dir / "nemotron_hybrid_ncm_bridges.pt"
    torch.save(full_payload, out_bridges_pt)

    # Also save compact BF16 checkpoint for fast distribution
    bf16_payload = {
        k: (
            {sk: (sv.to(torch.bfloat16) if sv.is_floating_point() else sv) for sk, sv in v.items()}
            if isinstance(v, dict) and k != "metrics" and k != "special_token_ids"
            else v
        )
        for k, v in full_payload.items()
    }
    out_bf16_pt = out_dir / "nemotron_ncm_core_bf16.pt"
    torch.save(bf16_payload, out_bf16_pt)

    summary_json = out_dir / "nemotron_hybrid_ncm_gpu_training_summary.json"
    summary_json.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    print(
        f"\n[Nemotron-NCM-GPU] ALL PHASES COMPLETE in {total_elapsed:.1f}s!\n"
        f"  -> {out_bridges_pt}\n"
        f"  -> {out_bf16_pt}\n"
        f"  -> {summary_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
