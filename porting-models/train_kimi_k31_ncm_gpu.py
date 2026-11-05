#!/usr/bin/env python3
"""
GPU End-to-End Trainer for MoonshotAI Kimi K3.1 Pro (`d=7168`, `V=163840` Tiktoken)
HybridAgent + 11-Level Next-Concept Model (`NCM`) + Lexical Pointer Decoder
(`porting-models/train_kimi_k31_ncm_gpu.py`).

Executes on the 8x RTX 3090 GPU Server:
  - Phase 0: Extracts real `7,168`-d Kimi K3.1 Pro hidden states (`h_win [1024, 64, 7168]` and
    `h_plan [1024, 48, 7168]`) on GPU through `KimiK31GPUStageModel` (`KDA Layer 0 + Gated DSA
    Layer 3 with 5-tensor Lightning Indexer + Latent SiTU-GLU MoE + Layer-93 Final Tail`).
  - Phase 1: Warm-starts `K31NativeHybridLink(d=7168, bottleneck=3584)`, `K3NativeHybridLink(d=7168, bottleneck=2048)`,
    and `QwenToK31HybridBridge(4096 <-> 7168)` from the 8-GPU 3-stage HybridSFT checkpoint
    (`kimi_k31_hybridagent_final.pt`) and refines across all 1,024 64-token windows (`1.0 * cos_loss + 0.2 * mse_loss`).
  - Phase 2: Fine-tunes the `163,840`-vocab 11-level `VQVAETokenizer` (`vqvae`) on GPU for 400 steps.
  - Phase 3: Jointly trains `K31ComboBridge(7168 -> 384)` (`k3_cbridge`) + `K31ExpertBridge(7168 -> 384)` (`k3_bridge`) +
    11-level `NextConceptTransformer (net)` + `K31LexicalPointerDecoder (ptr_dec)` on GPU for 500 steps.
  - Phase 4: Evaluates held-out metrics and exports both `kimi_k31_hybrid_ncm_bridges.pt` and the
    drop-in `checkpoint-clover-true-hybrid-agent-ncm.pt` for Clover AI Studio.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
for p in ("/dev/shm/clover-port/ncm/repo", "/dev/shm/clover-port/ncm", "/root/ncm-cpu/repo"):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from kimi_k31_gpu_backbone import D_HA, D_K31, D_LATENT, D_VQ, V_K31, KimiK31GPUStageModel
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from train_kimi_k31_hybridagent_gpu import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_IDS,
    K31NativeHybridLink,
    QwenToK31HybridBridge,
    load_kimi_k31_tiktoken,
)

P_RH = 48
P_BPE = 48
T_WIN = 64


class K31ComboBridge(nn.Module):
    """Maps `7,168`-d Kimi K3.1 Pro continuous latent vectors `R_K31(h) [B, P, 7168]` into NCM's `384`-d prefix space."""

    def __init__(self, d_in: int = D_K31, d_out: int = D_VQ):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, 768)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(768, d_out)
        self.ln_out = nn.LayerNorm(d_out)
        self.scale = nn.Parameter(torch.tensor(8.5))
        self.seg_rh = nn.Parameter(torch.zeros(1, 1, d_out))
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, r_k31: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(r_k31.float())
        x = self.fc2(self.act(self.fc1(x)))
        x = self.ln_out(x) * self.scale
        return (x * self.gate + self.seg_rh).to(r_k31.dtype)


class K31ExpertBridge(nn.Module):
    """Maps Kimi K3.1 Pro's `7,168`-d Gated DSA + Latent `SiTU-GLU` MoE states into NCM's `384`-d concept space."""

    def __init__(self, d_k31: int = D_K31, d_vq: int = D_VQ):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_k31)
        self.proj1 = nn.Linear(d_k31, d_vq)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(d_vq, d_vq)
        self.ln_out = nn.LayerNorm(d_vq)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, h_k31_moe: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(h_k31_moe.float())
        x = self.proj2(self.act(self.proj1(x)))
        return (torch.tanh(self.alpha) * self.ln_out(x)).to(h_k31_moe.dtype)


class CrossBlock(nn.Module):
    def __init__(self, d: int = 384, n_head: int = 6, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d = d
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.cx_q = nn.Linear(d, d)
        self.cx_kv = nn.Linear(d, 2 * d)
        self.cx_proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, mem_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        nh = self.n_head
        hd = C // nh
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=-1)
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))

        hq = self.ln_q(x)
        hkv = self.ln_kv(mem)
        cq = self.cx_q(hq).view(B, T, nh, hd).transpose(1, 2)
        ck, cv = self.cx_kv(hkv).split(C, dim=-1)
        M = mem.shape[1]
        ck = ck.view(B, M, nh, hd).transpose(1, 2)
        cv = cv.view(B, M, nh, hd).transpose(1, 2)
        attn_bias = None
        if mem_pad_mask is not None:
            attn_bias = torch.zeros(B, 1, 1, M, device=x.device, dtype=cq.dtype)
            attn_bias.masked_fill_(mem_pad_mask[:, None, None, :], float("-inf"))
        cy = F.scaled_dot_product_attention(cq, ck, cv, attn_mask=attn_bias, is_causal=False, dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.cx_proj(cy.transpose(1, 2).contiguous().view(B, T, C)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class K31LexicalPointerDecoder(nn.Module):
    """
    Exact architectural and state_dict match to `K3LexicalPointerDecoder` in `train_e7_clean_chat.py` /
    `clover-agent-engine.py`, operating over Kimi K3.1 Pro's `V_K31 = 163,840` vocabulary (`pad_id = 163585`).
    """

    def __init__(self, d: int = 384, n_layer: int = 3, n_head: int = 6):
        super().__init__()
        self.d = d
        self.tgt_pos = nn.Embedding(T_WIN, d)
        self.plan_pos = nn.Embedding(P_BPE, d)
        self.rh_pos = nn.Embedding(P_RH, d)
        self.blocks = nn.ModuleList([CrossBlock(d, n_head, 0.1) for _ in range(n_layer)])
        self.ln_out = nn.LayerNorm(d)
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        self.copy_q = nn.Linear(d, d)
        self.copy_k = nn.Linear(d, d)
        self.copy_gate = nn.Linear(2 * d, 1)
        self.copy_scale = nn.Parameter(torch.tensor(18.0))

    def forward(
        self,
        z_hat: torch.Tensor,
        z_rh: torch.Tensor,
        msk_rh: torch.Tensor,
        z_bpe: torch.Tensor,
        p_pad: torch.Tensor,
        vqvae_decoder: nn.Module,
        pad_id: int = PAD_ID,
        semicolon_id: int = 26,
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
        base_logits = vqvae_decoder(z_refined)  # [B, 64, 163840]

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


def collect_k31_sentences(ncm_data_dir: Path | None, target_count: int = 1024) -> list[str]:
    rng = random.Random(42)
    sentences: list[str] = []

    domain_templates = [
        "MoonshotAI Kimi K3.1 Pro is a 2.8T total / 104B active parameter Mixture-of-Experts model combining 69 KDA linear attention layers with 24 Gated DeepSeek Sparse Attention (Gated DSA) layers.",
        "Each of the 24 Gated DSA layers in Kimi K3.1 Pro adds a 5-tensor Lightning Indexer (k_norm.weight, k_norm.bias, weights_proj.weight, wk.weight, wq_b.weight) that selects the top 2,048 KV positions across a 1M token context window.",
        "LithosAI Kimi-K3.1-Pro-DSpark is a 5-layer 2.25B BF16 speculative draft head that fuses target layers 7, 23, 67, 83, and 91 to achieve a 3.45x mean acceptance length.",
        "Partitioning the expert cache per layer instead of using a single global table raised the cache hit rate from 23.3% to 48.0% inside the same 51.7 GB host RAM budget.",
        "CVE-2024-21762 is an out-of-bounds write vulnerability in Fortinet FortiOS SSL VPN caused by CWE-787 and exploited via CAPEC-100 buffer overflow.",
        "Before opening subsea equalization valve V-402, equalize differential pressure across the manifold and verify downstream telemetry readings.",
        "Harbourside Council voted 7 to 2 on Tuesday to approve 2.4 million pounds for a new segregated cycle lane along Mill Road.",
        "Shop owners on Mill Road, led by Tom Eliot, had asked for the vote to be delayed because of concerns about parking, so the council promised 40 temporary parking spaces on Dock Street.",
        "The first train travels for 30 minutes before the second train starts, covering 42 miles, leaving 126 miles between them at 09:40; they meet at approximately 10:29, roughly 57 miles from London.",
        "SpecMD Least-Stale slot caching paired with Red-Blue Pebble Rule R4 clean-drop eliminates dirty PCIe writebacks for read-only expert weights.",
        "Next-Concept Modeling predicts 11 hierarchical vector-quantized concept levels in parallel before decoding surface tokens through the lexical pointer network.",
        "Clover AI Studio unifies Kimi K3.1 Pro, NVIDIA Nemotron 3.5 Super VL, and Liquid AI LFM3-Omni-2B under a single 7-layer neuro-symbolic architecture.",
        "Using Python, 17**4 + 29*13 = 83521 + 377 = 83898.",
        "In the Gaia DR3 stellar catalog, approximately 2,800 spectral stars satisfy the radial-velocity signal-to-noise threshold.",
        "HybridAgent switches dynamically between discrete token generation and continuous latent reasoning spans bounded by <latent> and </latent> anchor tokens.",
        "The five stages of the Clover operating cycle are Context, Direction, Execution, Outcome, and Growth.",
    ]
    for i in range(256):
        sentences.append(domain_templates[i % len(domain_templates)])

    if ncm_data_dir is not None and ncm_data_dir.exists():
        xsum_p = ncm_data_dir / "xsum" / "reference_summaries.pickle"
        if xsum_p.exists():
            with open(xsum_p, "rb") as f:
                xsum_obj = pickle.load(f)
            xsum_list = list(xsum_obj.values()) if isinstance(xsum_obj, dict) else list(xsum_obj)
            rng.shuffle(xsum_list)
            for s in xsum_list[:400]:
                s_clean = str(s).strip()
                if 20 <= len(s_clean) <= 320:
                    sentences.append(s_clean)

        wmt_p = ncm_data_dir / "wmt14" / "reference_translations.pickle"
        if wmt_p.exists():
            with open(wmt_p, "rb") as f:
                wmt_obj = pickle.load(f)
            wmt_list = list(wmt_obj.values()) if isinstance(wmt_obj, dict) else list(wmt_obj)
            rng.shuffle(wmt_list)
            for s in wmt_list[:400]:
                s_clean = str(s).strip()
                if 20 <= len(s_clean) <= 320:
                    sentences.append(s_clean)

    rng.shuffle(sentences)
    while len(sentences) < target_count:
        sentences.extend(sentences[: target_count - len(sentences)])
    return sentences[:target_count]


def build_k31_windows(enc: Any, sentences: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    N = len(sentences)
    tgt_ids = torch.full((N, T_WIN), PAD_ID, dtype=torch.long)
    p_pad = torch.full((N, P_BPE), PAD_ID, dtype=torch.long)
    semi_ids = enc.encode(" ;")
    semicolon_id = int(semi_ids[-1]) if semi_ids else 26
    stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "by", "with", "is", "was", "were"}

    for i, s in enumerate(sentences):
        ids = enc.encode(s)[:42]
        if not ids:
            ids = [791]
        t_seq = (ids + [EOS_ID])[:T_WIN]
        tgt_ids[i, : len(t_seq)] = torch.tensor(t_seq, dtype=torch.long)

        if i % 10 < 7:
            p_seq = ids[:P_BPE]
        else:
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
            p_seq = enc.encode(plan_str)[:P_BPE]
            if semicolon_id not in p_seq and len(p_seq) < P_BPE:
                p_seq = p_seq[: P_BPE - 1] + [semicolon_id]

        p_pad[i, : len(p_seq)] = torch.tensor(p_seq, dtype=torch.long)

    p_mask = p_pad == PAD_ID
    return tgt_ids, p_pad, p_mask, semicolon_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Kimi K3.1 Pro GPU NCM & Bridges Trainer")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--wmt14-dir", type=Path, default=Path("/dev/shm/clover-port/ncm/wmt14"))
    parser.add_argument("--ncm-data-dir", type=Path, default=Path("/dev/shm/clover-port/ncm/data"))
    parser.add_argument("--base-k3-ckpt", type=Path, default=Path("/dev/shm/clover-port/ncm/checkpoint-clover-true-hybrid-agent-ncm.pt"))
    parser.add_argument("--num-windows", type=int, default=1024)
    parser.add_argument("--hybridlink-steps", type=int, default=300)
    parser.add_argument("--vqvae-steps", type=int, default=400)
    parser.add_argument("--ncm-steps", type=int, default=500)
    args = parser.parse_args()

    t_start = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pruned_dir = args.work_dir / "pruned"
    tok_path = args.work_dir / "tokenizer" / "tiktoken.model"
    ha_ckpt_path = args.work_dir / "hybridagent-8gpu" / "kimi_k31_hybridagent_final.pt"
    out_dir = args.work_dir / "hybrid-ncm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Kimi-K3.1-Pro NCM] Phase 0/4: Loading Kimi K3.1 Pro tokenizer & multi-domain windows...", flush=True)
    enc = load_kimi_k31_tiktoken(tok_path)
    sentences = collect_k31_sentences(args.ncm_data_dir, target_count=args.num_windows)
    tgt_ids, p_pad, p_mask, semicolon_id = build_k31_windows(enc, sentences)

    n_eval = 64
    n_train = args.num_windows - n_eval

    ha_ckpt = torch.load(ha_ckpt_path, map_location="cpu", weights_only=False) if ha_ckpt_path.exists() else None
    base_k3_sd = torch.load(args.base_k3_ckpt, map_location="cpu", weights_only=False) if args.base_k3_ckpt.exists() else {}

    cache_pt = out_dir / "k31_cached_hidden_1024.pt"
    if cache_pt.exists():
        print(f"[Kimi-K3.1-Pro NCM] Phase 0/4: Loading cached 7168-d Kimi K3.1 Pro hidden states from {cache_pt}...", flush=True)
        c_obj = torch.load(cache_pt, map_location="cpu", weights_only=False)
        h_win_all = c_obj["h_win_all"]
        h_plan_all = c_obj["h_plan_all"]
        emb_weight_cpu = c_obj["emb_weight_cpu"]
    else:
        print(f"[Kimi-K3.1-Pro NCM] Phase 0/4: Extracting 7168-d Kimi K3.1 Pro hidden states on GPU across {args.num_windows} windows...", flush=True)
        backbone = KimiK31GPUStageModel(pruned_dir, max_experts_on_gpu=16).to(device)
        backbone.eval()

        h_win_all = torch.empty((args.num_windows, T_WIN, D_K31), dtype=torch.bfloat16)
        h_plan_all = torch.empty((args.num_windows, P_BPE, D_K31), dtype=torch.bfloat16)
        emb_weight_cpu = backbone.embed_tokens.weight.detach().cpu().clone()

        with torch.no_grad():
            bs = 16
            for st in range(0, args.num_windows, bs):
                ed = min(args.num_windows, st + bs)
                b_tgt = tgt_ids[st:ed].to(device)
                b_plan = p_pad[st:ed].to(device)
                h_w, _ = backbone.forward_stages(backbone.embed_tokens(b_tgt))
                h_p, _ = backbone.forward_stages(backbone.embed_tokens(b_plan))
                h_win_all[st:ed] = h_w.detach().cpu()
                h_plan_all[st:ed] = h_p.detach().cpu()

        del backbone
        torch.cuda.empty_cache()
        torch.save({"h_win_all": h_win_all, "h_plan_all": h_plan_all, "emb_weight_cpu": emb_weight_cpu}, cache_pt)

    # =========================================================================
    # Phase 1: Refine K31NativeHybridLink(d=7168, bottleneck=3584) + K3NativeHybridLink(d=7168, bottleneck=2048)
    # =========================================================================
    print(f"[Kimi-K3.1-Pro NCM] Phase 1/4: Training K31NativeHybridLink(7168->3584->7168) + K3NativeHybridLink(7168->2048->7168) for {args.hybridlink_steps} steps...", flush=True)
    k31_hl_3584 = K31NativeHybridLink(d=D_K31, bottleneck=D_LATENT).to(device)
    k3_hl_2048 = K31NativeHybridLink(d=D_K31, bottleneck=2048).to(device)
    qwen_bridge = QwenToK31HybridBridge(d_k31=D_K31, d_ha=D_HA).to(device)

    if ha_ckpt is not None:
        k31_hl_3584.load_state_dict(ha_ckpt["k31_native_hybrid_link"], strict=False)
        qwen_bridge.load_state_dict(ha_ckpt["qwen_to_k31_bridge"], strict=False)
    if "k3_hybridlink" in base_k3_sd:
        k3_hl_2048.load_state_dict(base_k3_sd["k3_hybridlink"], strict=False)

    emb_w_dev = emb_weight_cpu.to(device)

    @torch.no_grad()
    def eval_hybridlink() -> dict[str, float]:
        k31_hl_3584.eval()
        k3_hl_2048.eval()
        h_ev = h_win_all[n_train:].to(device).float()
        t_ev = tgt_ids[n_train:].to(device)
        valid_m = (t_ev[:, 1:] != PAD_ID) & (t_ev[:, 1:] != EOS_ID)
        h_src = h_ev[:, :-1][valid_m]
        e_tgt = emb_w_dev[t_ev[:, 1:][valid_m]].float()
        c_3584 = float(F.cosine_similarity(k31_hl_3584(h_src), e_tgt, dim=-1).mean().item())
        c_2048 = float(F.cosine_similarity(k3_hl_2048(h_src), e_tgt, dim=-1).mean().item())
        k31_hl_3584.train()
        k3_hl_2048.train()
        return {"eval_cos_k31_3584": round(c_3584, 4), "eval_cos_k3_2048": round(c_2048, 4)}

    hl_before = eval_hybridlink()
    print(f"[Kimi-K3.1-Pro NCM] HybridLink pre-Phase1: {hl_before}", flush=True)

    hl_params = list(k31_hl_3584.parameters()) + list(k3_hl_2048.parameters()) + list(qwen_bridge.parameters())
    hl_opt = torch.optim.AdamW(hl_params, lr=5e-4, weight_decay=0.01)
    hl_sched = torch.optim.lr_scheduler.CosineAnnealingLR(hl_opt, T_max=args.hybridlink_steps, eta_min=2e-5)

    for step in range(1, args.hybridlink_steps + 1):
        idx = torch.randint(0, n_train, (64,))
        h_b = h_win_all[idx].to(device).float()
        t_b = tgt_ids[idx].to(device)
        valid_m = t_b[:, 1:] != PAD_ID
        h_src = h_b[:, :-1][valid_m]
        e_tgt = emb_w_dev[t_b[:, 1:][valid_m]].float()

        p_3584 = k31_hl_3584(h_src)
        p_2048 = k3_hl_2048(h_src)
        p_qwen = qwen_bridge(h_src)

        l_3584 = (1.0 - F.cosine_similarity(p_3584, e_tgt, dim=-1).mean()) + 0.2 * F.mse_loss(p_3584, e_tgt)
        l_2048 = (1.0 - F.cosine_similarity(p_2048, e_tgt, dim=-1).mean()) + 0.2 * F.mse_loss(p_2048, e_tgt)
        l_qwen = (1.0 - F.cosine_similarity(p_qwen, e_tgt, dim=-1).mean()) + 0.2 * F.mse_loss(p_qwen, e_tgt)
        loss_hl = l_3584 + l_2048 + 0.5 * l_qwen

        hl_opt.zero_grad(set_to_none=True)
        loss_hl.backward()
        torch.nn.utils.clip_grad_norm_(hl_params, 1.0)
        hl_opt.step()
        hl_sched.step()

    hl_after = eval_hybridlink()
    print(f"[Kimi-K3.1-Pro NCM] HybridLink post-Phase1: {hl_after}", flush=True)
    k31_hl_3584.eval()
    k3_hl_2048.eval()
    qwen_bridge.eval()
    del emb_w_dev
    torch.cuda.empty_cache()

    # =========================================================================
    # Phase 2: Fine-Tune 163,840-Vocab 11-Level VQVAETokenizer on GPU
    # =========================================================================
    print(f"[Kimi-K3.1-Pro NCM] Phase 2/4: Fine-tuning 163,840-vocab 11-Level VQ-VAE on GPU for {args.vqvae_steps} steps...", flush=True)
    cfg_yaml = args.wmt14_dir / "config.yaml"
    if not cfg_yaml.exists():
        cfg_yaml = args.wmt14_dir / "config-wmt14-concept-model.yaml"
    cfg = OmegaConf.load(str(cfg_yaml))
    tok_pt = args.wmt14_dir / "tokenizer.pt"
    if not tok_pt.exists():
        tok_pt = args.wmt14_dir / "checkpoint-wmt14-concept-tokenizer.pt"
    cfg.tokenizer.vqvae.checkpoint_path = str(tok_pt)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    vqvae = tk.vqvae.cpu()
    vqvae.in_vocab_size = V_K31
    vqvae.encoder.wte = nn.Embedding(V_K31, D_VQ)
    vqvae.decoder.output_layer = nn.Linear(D_VQ, V_K31, bias=False)
    if "vqvae" in base_k3_sd:
        vqvae.load_state_dict(base_k3_sd["vqvae"], strict=False)
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
        ce = float(F.cross_entropy(logits.float().view(-1, V_K31), t_ev.view(-1)).item())
        return {"vq_eval_ce": round(ce, 4), "vq_eval_acc_nonpad": round(acc_nonpad, 4), "vq_eval_acc_all": round(acc_all, 4)}

    vq_before = eval_vqvae()
    print(f"[Kimi-K3.1-Pro NCM] VQ-VAE pre-Phase2: {vq_before}", flush=True)

    for name, p in vqvae.named_parameters():
        p.requires_grad_("vq." not in name)
    vqvae.train()
    vqvae.vq.eval()
    vq_opt = torch.optim.AdamW([p for p in vqvae.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01, fused=True)
    vq_sched = torch.optim.lr_scheduler.CosineAnnealingLR(vq_opt, T_max=args.vqvae_steps, eta_min=2e-5)

    for step in range(1, args.vqvae_steps + 1):
        idx = torch.randint(0, n_train, (24,))
        t_b = tgt_ids[idx].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, commit_loss, _, _, _ = vqvae(t_b)
            ce_per_tok = F.cross_entropy(logits.float().view(-1, V_K31), t_b.view(-1), reduction="none").view_as(t_b)
            tok_weights = torch.where(t_b != PAD_ID, 4.0, 0.25)
            ce_loss = (ce_per_tok * tok_weights).sum() / tok_weights.sum()
            loss_vq = ce_loss + 0.25 * commit_loss.mean()

        vq_opt.zero_grad(set_to_none=True)
        loss_vq.backward()
        torch.nn.utils.clip_grad_norm_([p for p in vqvae.parameters() if p.requires_grad], 1.0)
        vq_opt.step()
        vq_sched.step()

    vq_after = eval_vqvae()
    print(f"[Kimi-K3.1-Pro NCM] VQ-VAE post-Phase2: {vq_after}", flush=True)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    del vq_opt, vq_sched, hl_opt, hl_sched
    torch.cuda.empty_cache()

    # Precompute 11-level VQ codes & K31NativeHybridLink latent plans
    idx_chunks = []
    z_clean_all = torch.empty((args.num_windows, T_WIN, D_VQ), dtype=torch.float32)
    r_plan_all = torch.empty((args.num_windows, P_BPE, D_K31), dtype=torch.float32)
    with torch.no_grad():
        for st in range(0, args.num_windows, 64):
            ed = min(args.num_windows, st + 64)
            b_t = tgt_ids[st:ed].to(device)
            b_idx = vqvae.encode(b_t)
            idx_chunks.append(b_idx.cpu())
            zh_levels, _ = vqvae.vq.idx_to_zhats(b_idx, return_zlevels=True)
            z_clean_all[st:ed] = zh_levels[-1].float().cpu()
            b_hp = h_plan_all[st:ed].to(device).float()
            r_plan_all[st:ed] = k3_hl_2048(b_hp).float().cpu()
    gt_codes_all = torch.cat(idx_chunks, dim=0)

    # =========================================================================
    # Phase 3: Joint GPU Training of K31ComboBridge + K31ExpertBridge + NextConceptTransformer + PointerDecoder
    # =========================================================================
    print(f"[Kimi-K3.1-Pro NCM] Phase 3/4: Jointly training K31ComboBridge + K31ExpertBridge + NextConceptTransformer + PointerDecoder for {args.ncm_steps} steps...", flush=True)
    mcfg = cfg.model
    mcfg.prefix_len = P_RH + P_BPE
    mcfg.predict_level0 = True
    net = NextConceptTransformer(vqvae_local=vqvae, config=mcfg).to(device)
    if "net" in base_k3_sd:
        net.load_state_dict(base_k3_sd["net"], strict=False)

    ptr_dec = K31LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(device)
    if "ptr_dec" in base_k3_sd:
        ptr_dec.load_state_dict(base_k3_sd["ptr_dec"], strict=False)

    combo_bridge = K31ComboBridge(D_K31, D_VQ).to(device)
    expert_bridge = K31ExpertBridge(D_K31, D_VQ).to(device)
    if "k3_cbridge" in base_k3_sd:
        combo_bridge.load_state_dict(base_k3_sd["k3_cbridge"], strict=False)
    if "k3_bridge" in base_k3_sd:
        expert_bridge.load_state_dict(base_k3_sd["k3_bridge"], strict=False)
        with torch.no_grad():
            if expert_bridge.alpha.abs().item() < 0.05:
                expert_bridge.alpha.fill_(0.35)

    for p in net.parameters():
        p.requires_grad_(True)

    levels = list(vqvae.vq.levels)
    lid = torch.cat([torch.full((l_k,), k, dtype=torch.long) for k, l_k in enumerate(levels)]).to(device)
    ls = torch.tensor(levels, dtype=torch.float32, device=device)
    w_lvl = (ls ** (-0.7))[lid]

    @torch.no_grad()
    def eval_ncm_pipeline() -> dict[str, float]:
        torch.cuda.empty_cache()
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

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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

            tok_logits = ptr_dec(b_zclean, z_rh, b_pmask, z_bpe, b_ppad, vqvae.decoder, pad_id=PAD_ID, semicolon_id=semicolon_id)
            non_pad = (b_tgt != PAD_ID) & (b_tgt != EOS_ID)
            tok_acc_nonpad = float((tok_logits.argmax(dim=-1)[non_pad] == b_tgt[non_pad]).float().mean().item())
            tok_ce = float(F.cross_entropy(tok_logits.view(-1, V_K31), b_tgt.view(-1)).item())

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
    print(f"[Kimi-K3.1-Pro NCM] NCM pre-Phase3: {ncm_before}", flush=True)

    ncm_params = (
        list(combo_bridge.parameters())
        + list(expert_bridge.parameters())
        + list(net.parameters())
        + list(ptr_dec.parameters())
    )
    ncm_opt = torch.optim.AdamW(ncm_params, lr=3e-4, weight_decay=0.01)
    ncm_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ncm_opt, T_max=args.ncm_steps, eta_min=2e-5)

    for step in range(1, args.ncm_steps + 1):
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

            ncm_in = vqvae.vq.idx_to_ncm_input(b_codes)
            lg_ncm = net(torch.cat([z_pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=pmask_cat)
            B_cur, T_c, V_c = lg_ncm.shape
            ce_ncm = F.cross_entropy(lg_ncm.float().reshape(-1, V_c), b_codes.reshape(-1), reduction="none").view(B_cur, T_c)
            loss_code = (ce_ncm * w_lvl[None]).sum() / (w_lvl.sum() * B_cur)

            tok_logits = ptr_dec(b_zclean, z_rh, b_pmask, z_bpe, b_ppad, vqvae.decoder, pad_id=PAD_ID, semicolon_id=semicolon_id)
            ce_tok_all = F.cross_entropy(tok_logits.float().view(-1, V_K31), b_tgt.view(-1), reduction="none").view_as(b_tgt)
            tok_w = torch.where(b_tgt != PAD_ID, 4.0, 0.25)
            loss_tok = (ce_tok_all * tok_w).sum() / tok_w.sum()
            loss = loss_code + loss_tok

        ncm_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ncm_params, 1.0)
        ncm_opt.step()
        ncm_sched.step()

        if step == 1 or step % 50 == 0 or step == args.ncm_steps:
            print(f"  [Phase 3 NCM+Bridges] step {step:3d}/{args.ncm_steps} | code_ce={loss_code.item():.4f} | tok_ce={loss_tok.item():.4f} | alpha={expert_bridge.alpha.item():.4f}", flush=True)

    del loss, loss_code, loss_tok, tok_logits, lg_ncm, ce_tok_all, ce_ncm
    torch.cuda.empty_cache()
    ncm_after = eval_ncm_pipeline()
    print(f"[Kimi-K3.1-Pro NCM] NCM post-Phase3: {ncm_after}", flush=True)

    elapsed = round(time.perf_counter() - t_start, 2)
    ha_summary_path = args.work_dir / "hybridagent-8gpu" / "kimi_k31_hybridagent_training_summary.json"
    ha_summary = json.loads(ha_summary_path.read_text(encoding="utf-8")) if ha_summary_path.exists() else {}

    summary = {
        "model_id": "moonshotai/Kimi-K3.1-Pro",
        "drafter_id": "LithosAI/Kimi-K3.1-Pro-DSpark",
        "hidden_size": D_K31,
        "vocab_size": V_K31,
        "vq_dim": D_VQ,
        "num_windows_trained": args.num_windows,
        "hybridagent_3stage_summary": ha_summary,
        "phase1_hybridlink": {"before": hl_before, "after": hl_after},
        "phase2_vqvae": {"before": vq_before, "after": vq_after},
        "phase3_ncm_bridges_pointer": {"before": ncm_before, "after": ncm_after},
        "elapsed_seconds": elapsed,
    }
    (out_dir / "kimi_k31_hybrid_ncm_gpu_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save unified checkpoint compatible with both `clover-agent-engine.py` and `switchable_model_backend.py`
    unified_ckpt = {
        "vqvae": vqvae.cpu().state_dict(),
        "net": net.cpu().state_dict(),
        "k3_hybridlink": k3_hl_2048.cpu().state_dict(),
        "k31_native_hybrid_link": k31_hl_3584.cpu().state_dict(),
        "qwen_to_k31_bridge": qwen_bridge.cpu().state_dict(),
        "k3_cbridge": combo_bridge.cpu().state_dict(),
        "ha_cbridge": base_k3_sd.get("ha_cbridge", {}),
        "k3_bridge": expert_bridge.cpu().state_dict(),
        "ptr_dec": ptr_dec.cpu().state_dict(),
        "metrics": summary,
        "special_token_ids": {
            "<think>": SPECIAL_IDS[0],
            "</think>": SPECIAL_IDS[1],
            "<latent>": SPECIAL_IDS[2],
            "</latent>": SPECIAL_IDS[3],
        },
    }
    if ha_ckpt is not None:
        unified_ckpt["special_emb_delta"] = ha_ckpt["special_emb_delta"]
        unified_ckpt["special_lm_delta"] = ha_ckpt["special_lm_delta"]
        unified_ckpt["norm_f_delta"] = ha_ckpt["norm_f_delta"]

    torch.save(unified_ckpt, out_dir / "kimi_k31_hybrid_ncm_bridges.pt")
    torch.save(unified_ckpt, out_dir / "checkpoint-clover-true-hybrid-agent-ncm.pt")
    print(f"[Kimi-K3.1-Pro NCM] Saved unified checkpoints to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
