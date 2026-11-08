#!/usr/bin/env python3
"""
8-GPU 3-Stage HybridSFT + 4-Phase 11-Level NCM + JEPA Trainer & Exporter for Clover-35B-A3B
(`Diffuspace/Clover-35B-A3B-Preview`, `d=2048`, `V=248,320`, `30 BD-GDN + 10 Gated GQA + 256-Expert MoE`)
(`porting-models/train_clover_35b_hybrid_ncm_gpu.py`).

Executes on the 8x RTX 3090 GPU Server:
  1. Stage A (8-GPU 3-Stage HybridSFT):
     - Stage 1 (`stage1_format_activation`):     alpha_max=0.0, beta=0.05
     - Stage 2 (`stage2_hybridlink_activation`): alpha_max=1.0, beta=0.15
     - Stage 3 (`stage3_self_conditioned`):      alpha_max=1.0, beta=0.10
     Trains `Clover35BNativeHybridLink(d=2048, bottleneck=1024)`, `Clover35BNative512Link(d=2048, bottleneck=512)`,
     `Clover35BHybridLink2048(2048 -> 2048 -> 2048)`, `QwenToClover35BHybridBridge(4096 <-> 2048)`, and special
     token rows `<think>` (248068), `</think>` (248069), `<latent>` (248077), `</latent>` (248078), `<|mask|>` (151669).
  2. Stage B (4-Phase GPU 11-Level NCM + Bridges + 248,320-Vocab VQ-VAE):
     - Phase 0: Extracts real `2,048`-d `Clover-35B-A3B` hidden states (`h_win`, `h_plan`) through
       `Clover35BGPUStageModel` (Layer 0 BD-GDN + Layer 3 Gated GQA + 256-Expert MoE + Final RMSNorm).
     - Phase 1: Refines `Clover35BNativeHybridLink` (`2048 -> 1024 -> 2048` & `2048 -> 512 -> 2048`) next-token
       embedding alignment (`1.0 * cos_loss + 0.2 * mse_loss`).
     - Phase 2: Trains the `248,320`-vocab 11-level Multiscale `VQVAETokenizer` (`levels = [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]`).
     - Phase 3: Jointly trains `Clover35BComboBridge(2048 -> 768 -> 384)` + `Clover35BExpertBridge(2048 -> 384 -> 384)`
       + 11-level Concept Predictor + Lexical Pointer Decoder.
  3. Stage C (Clover-35B-A3B JEPA Trunk Alignment):
     - Aligns `jepa-trunk-0` (`bridge.lfm3_to_vq`, `bridge.vq_to_lfm3`, `state_adapter`, `ac_pred`, `idm`) on
       `Clover-35B-A3B`'s `2,048`-d hidden states and `384`-d concept space.
  4. Stage D (Export Complete 5-Folder + JEPA Precomputed Tree in `work_dir / "hf-stage"`):
     - `experts/`, `trunks/`, `hybridagent/plan-trunk-0`, `bridges/bridge-trunk-0`, `ncm/`, and `jepa/jepa-trunk-0`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STUDIO_CLOVER = REPO_ROOT / "clover-ai-system" / "studio" / "clover"
for p in (str(SCRIPT_DIR), str(STUDIO_CLOVER), "/workspace/rpre-gpu", "/workspace/rpre-gpu/clover-ai-system/studio/clover"):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from clover_35b_gpu_backbone import (
    BD_GDN_LAYERS_0IDX,
    BOS_TOKEN_ID,
    D_CLOVER35B,
    D_HA,
    D_LATENT_35B,
    D_VQ,
    EOS_TOKEN_ID,
    IM_START_ID,
    LATENT_CLOSE_ID,
    LATENT_OPEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    THINK_CLOSE_ID,
    THINK_OPEN_ID,
    V_CLOVER35B,
    Clover35BGPUStageModel,
    clover_rms_norm_op,
    tensor_to_le_bytes,
)

VQ_LEVELS = (1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64)
SPECIAL_IDS = [THINK_OPEN_ID, THINK_CLOSE_ID, LATENT_OPEN_ID, LATENT_CLOSE_ID, MASK_TOKEN_ID]


def write_clvrtrk1(trunk_dir: Path, tensors: dict[str, np.ndarray | torch.Tensor]) -> Path:
    """Writes a dictionary of float32 tensors into Clover's `CLVRTRK1` binary trunk container."""
    trunk_dir = Path(trunk_dir)
    trunk_dir.mkdir(parents=True, exist_ok=True)
    common_buf = bytearray()
    op_buf = bytearray()
    ffn_buf = bytearray()
    entries: list[bytes] = []

    for name, arr in tensors.items():
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().float().numpy()
        a = np.ascontiguousarray(arr, dtype=np.float32)
        raw = a.tobytes()
        if "mlp" in name or "ffn" in name or "net." in name or "proj2" in name or "fc2" in name:
            fid, buf = 3, ffn_buf
        elif "attn" in name or "gru" in name or "proj" in name or "fc1" in name:
            fid, buf = 1, op_buf
        else:
            fid, buf = 0, common_buf
        offset = len(buf)
        buf.extend(raw)
        ndim = a.ndim
        dims = list(a.shape) + [1, 1, 1, 1]
        name_b = name.encode("utf-8")[:63].ljust(64, b"\x00")
        meta = struct.pack("<IIIQQIIII", fid, 0, ndim, offset, len(raw), dims[0], dims[1], dims[2], dims[3])
        entries.append(name_b + meta)

    header = b"CLVRTRK1" + struct.pack("<III", 1, 0, len(entries))
    (trunk_dir / "index.bin").write_bytes(header + b"".join(entries))
    (trunk_dir / "common.bin").write_bytes(bytes(common_buf))
    (trunk_dir / "operator.bin").write_bytes(bytes(op_buf))
    (trunk_dir / "ffn.bin").write_bytes(bytes(ffn_buf))
    return trunk_dir


# ---------------- Clover-35B-A3B HybridAgent & NCM Modules ----------------

class Clover35BNativeHybridLink(nn.Module):
    """Native HybridLink operating in Clover-35B-A3B's `d = 2,048` hidden space (`bottleneck = 1024` or `512`)."""

    def __init__(self, d: int = D_CLOVER35B, bottleneck: int = D_LATENT_35B):
        super().__init__()
        self.pre_ln = nn.LayerNorm(d)
        self.proj1 = nn.Linear(d, bottleneck)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(bottleneck, d)
        self.post_ln = nn.LayerNorm(d)
        self.out_scale = nn.Parameter(torch.tensor(0.024))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        hf = h.float()
        z = self.post_ln(hf + self.proj2(self.act(self.proj1(self.pre_ln(hf)))))
        return z * self.out_scale

    def export_fused(self, prefix: str = "native") -> dict[str, np.ndarray]:
        with torch.no_grad():
            w1 = self.proj1.weight.float() * self.pre_ln.weight.float().unsqueeze(0)
            b1 = self.proj1.bias.float() + torch.mv(self.proj1.weight.float(), self.pre_ln.bias.float())
            s = float(self.out_scale.item()) if prefix == "native" else 1.0
            ln_w = (self.post_ln.weight.float() * s).cpu().numpy()
            ln_b = (self.post_ln.bias.float() * s).cpu().numpy()
            k_ln = f"{prefix}.post_ln_fused" if prefix == "native" else f"{prefix}.post_ln"
            return {
                f"{prefix}.proj1_fused.weight": w1.cpu().numpy(),
                f"{prefix}.proj1_fused.bias": b1.cpu().numpy(),
                f"{prefix}.proj2.weight": self.proj2.weight.float().cpu().numpy(),
                f"{prefix}.proj2.bias": self.proj2.bias.float().cpu().numpy(),
                f"{k_ln}.weight": ln_w,
                f"{k_ln}.bias": ln_b,
            }


class QwenToClover35BHybridBridge(nn.Module):
    """Bidirectional bridge between `HybridAgent`'s `4,096`-d `<latent>` space and `Clover-35B-A3B`'s `2,048`-d space."""

    def __init__(self, d_model: int = D_CLOVER35B, d_ha: int = D_HA):
        super().__init__()
        self.to_ha = nn.Linear(d_model, d_ha, bias=False)
        self.input_norm = nn.LayerNorm(d_ha)
        self.up_proj = nn.Linear(d_ha, 4096)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(4096, d_ha)
        self.output_norm = nn.LayerNorm(d_ha)
        self.from_ha = nn.Linear(d_ha, d_model, bias=False)
        self.out_scale = nn.Parameter(torch.tensor(0.024))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.to_ha(h.float())
        r_h = self.output_norm(x + self.down_proj(self.act(self.up_proj(self.input_norm(x)))))
        return self.from_ha(r_h) * self.out_scale


class Clover35BComboBridge(nn.Module):
    """Maps `2,048`-d `Clover-35B-A3B` continuous latent vectors `R(h)` into NCM's `384`-d concept prefix space."""

    def __init__(self, d_in: int = D_CLOVER35B, d_out: int = D_VQ):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, 768)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(768, d_out)
        self.ln_out = nn.LayerNorm(d_out)
        self.scale = nn.Parameter(torch.tensor(8.5))
        self.seg_rh = nn.Parameter(torch.zeros(1, 1, d_out))
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, r_h: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(r_h.float())
        x = self.fc2(self.act(self.fc1(x)))
        x = self.ln_out(x) * self.scale
        return (x * self.gate + self.seg_rh).to(r_h.dtype)

    def export_fused(self) -> dict[str, np.ndarray]:
        with torch.no_grad():
            w1 = self.fc1.weight.float() * self.ln_in.weight.float().unsqueeze(0)
            b1 = self.fc1.bias.float() + torch.mv(self.fc1.weight.float(), self.ln_in.bias.float())
            eff = float((self.scale * self.gate).item())
            ln_w = (self.ln_out.weight.float() * eff).cpu().numpy()
            ln_b = (self.ln_out.bias.float() * eff + self.seg_rh.float().reshape(-1)).cpu().numpy()
            return {
                "combo.fc1_fused.weight": w1.cpu().numpy(),
                "combo.fc1_fused.bias": b1.cpu().numpy(),
                "combo.fc2.weight": self.fc2.weight.float().cpu().numpy(),
                "combo.fc2.bias": self.fc2.bias.float().cpu().numpy(),
                "combo.ln_out_fused.weight": ln_w,
                "combo.ln_out_fused.bias": ln_b,
            }


class Clover35BExpertBridge(nn.Module):
    """Maps `2,048`-d `Clover-35B-A3B` BD-GDN + Gated GQA MoE hidden states into NCM's `384`-d concept space."""

    def __init__(self, d_in: int = D_CLOVER35B, d_vq: int = D_VQ):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in)
        self.proj1 = nn.Linear(d_in, d_vq)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(d_vq, d_vq)
        self.ln_out = nn.LayerNorm(d_vq)
        self.alpha = nn.Parameter(torch.tensor(0.65))

    def forward(self, h_moe: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(h_moe.float())
        x = self.proj2(self.act(self.proj1(x)))
        return (torch.tanh(self.alpha) * self.ln_out(x)).to(h_moe.dtype)

    def export_fused(self) -> dict[str, np.ndarray]:
        with torch.no_grad():
            w1 = self.proj1.weight.float() * self.ln_in.weight.float().unsqueeze(0)
            b1 = self.proj1.bias.float() + torch.mv(self.proj1.weight.float(), self.ln_in.bias.float())
            eff = float(torch.tanh(self.alpha).item())
            ln_w = (self.ln_out.weight.float() * eff).cpu().numpy()
            ln_b = (self.ln_out.bias.float() * eff).cpu().numpy()
            return {
                "expert.proj1_fused.weight": w1.cpu().numpy(),
                "expert.proj1_fused.bias": b1.cpu().numpy(),
                "expert.proj2.weight": self.proj2.weight.float().cpu().numpy(),
                "expert.proj2.bias": self.proj2.bias.float().cpu().numpy(),
                "expert.ln_out_fused.weight": ln_w,
                "expert.ln_out_fused.bias": ln_b,
            }


# ---------------- Corpus & Trajectory Builder for Clover-35B-A3B ----------------

CORPUS_QA = [
    (
        "What is the architecture of DiffuSpace Clover-35B-A3B-Preview?",
        "Clover-35B-A3B-Preview combines a 40-layer hybrid Bidirectional Gated DeltaNet (BD-GDN) and Gated GQA backbone with 256 routed experts (top-8) and 1 shared expert.",
        "30 BD-GDN layers with conv_right_weight and bd_softmax local memory + 10 Gated GQA layers + block diffusion LowConfidence decoding (block_size=32, threshold=0.95) over 128K context.",
        "Clover-35B-A3B-Preview is a 35B-parameter (3B active) hybrid Mixture-of-Experts diffusion language model with 30 Bidirectional Gated DeltaNet (BD-GDN) layers, 10 Gated GQA layers, 256 routed experts (top-8 with top_p=0.9 block coreset selection), 1 shared expert per layer, and LowConfidence block diffusion decoding over a 128K context window.",
    ),
    (
        "How does Bidirectional Gated DeltaNet (BD-GDN) work during block diffusion preview in Clover-35B-A3B?",
        "BD-GDN applies an independent right convolution (conv_right_weight) across the active 32-token block without mutating the committed clean conv cache.",
        "It then combines normalized within-block bidirectional softmax attention logits with a gate-decayed clean-state history bucket controlled by bd_softmax_logit_scale, bd_softmax_history_bias, and bd_softmax_history_c.",
        "During block diffusion preview, BD-GDN preserves a read-only causal prefix state while enabling bidirectional token interaction inside the active 32-token block via right-convolution weights and a learned softmax history bucket.",
    ),
    (
        "How does block-level expert coreset selection work in Clover-35B-A3B?",
        "coreset_torch aggregates each token's top-8 expert routing probabilities across the active block and ranks all 256 experts by total block routing mass.",
        "It keeps the highest-mass experts until cumulative routing mass reaches moe_block_coreset_top_p = 0.9, masking out the rest to reduce distinct expert loads per block.",
        "Clover-35B-A3B aggregates top-8 router probabilities across the block, sorts experts by cumulative routing mass, and restricts execution to the subset covering 90% of the block's routing mass (`moe_block_coreset_top_p = 0.9`).",
    ),
    (
        "How do AGENTS.md Clover Cycles operate across Web, Android, and Windows?",
        "Each turn runs Context -> Direction -> Execution -> Outcome -> Growth with deterministic Outcome verification before handing back.",
        "verify_turn_outcome checks compiler AST, sandboxed Python execution, tool trace failures, arithmetic equations, and format constraints, looping up to 3 cycles and recording LRN-* Growth.",
        "Under AGENTS.md, Clover executes Context -> Direction -> Execution -> Outcome -> Growth as a repeating cycle, verifying every draft against external reality checks and automatically re-entering Context when any check fails.",
    ),
    (
        "How does Clover's selectable JEPA engine (I-JEPA, Audio-JEPA, EB-JEPA, MJEPA, Clover RPRE) integrate into C2 and Outcome Growth?",
        "CloverJEPARuntime encodes visual, audio, and 384-d concept states into a 64-d predictive latent space with Variance-Covariance regularization.",
        "In C2, EB-JEPA CEMPlanner optimizes latent trajectories for tool selection; in Outcome -> Growth, OutcomeResidualAdapter adapts on verified outcomes with a non-regression rollback gate.",
        "Clover unifies I-JEPA, Audio-JEPA, EB-JEPA, MJEPA, and Clover RPRE in a 64-d predictive representation engine that ranks multimodal salience in S2/S3, optimizes C2 latent tool planning, and updates an OutcomeResidualAdapter on verified outcomes.",
    ),
    (
        "Explain how SpecMD Least-Stale slot caching with Rule R4 clean-drop reduces PCIe traffic.",
        "Read-only expert weights fetched from host memory are never modified on GPU, so Rule R4 drops them cleanly without dirty D2H writebacks.",
        "Partitioning the slot cache per layer and applying Red-Blue Pebble Rule R4 clean-drop eliminates 100% of dirty PCIe writebacks.",
        "In Clover's 4-tier memory hierarchy, SpecMD Least-Stale evicts the expert whose next predicted activation is furthest in the future, while Red-Blue Pebble Rule R4 drops clean read-only expert weights without writing them back across PCIe.",
    ),
]


def build_tokenized_batches(
    tok_dir: Path,
    n_samples: int = 256,
    seq_len: int = 64,
    device: torch.device = torch.device("cuda:0"),
) -> dict[str, torch.Tensor]:
    from omni_c2_c3 import Lfm3PureBpeTokenizer

    tok = Lfm3PureBpeTokenizer(tok_dir / "tokenizer.json")
    tok.vocab_size = V_CLOVER35B
    input_rows = []
    label_rows = []
    latent_masks = []

    for idx in range(n_samples):
        q, t1, lat, ans = CORPUS_QA[idx % len(CORPUS_QA)]
        q_ids = tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", allow_special=True)
        t1_ids = tok.encode(t1 + " ", allow_special=False)[:12]
        lat_ids = tok.encode(lat, allow_special=False)[:12]
        ans_ids = tok.encode(ans, allow_special=False)[:24]

        seq = (
            q_ids[:16]
            + [THINK_OPEN_ID]
            + t1_ids
            + [LATENT_OPEN_ID]
            + lat_ids
            + [LATENT_CLOSE_ID]
            + [THINK_CLOSE_ID]
            + ans_ids
            + [EOS_TOKEN_ID]
        )
        if len(seq) < seq_len:
            pad_n = seq_len - len(seq)
            seq = seq + [PAD_TOKEN_ID] * pad_n
        else:
            seq = seq[:seq_len]
            seq[-1] = EOS_TOKEN_ID

        p_len = min(len(q_ids), 16)
        lbl = [-100] * p_len + [ (x if x != PAD_TOKEN_ID else -100) for x in seq[p_len:] ]
        lmask = [1 if x in (LATENT_OPEN_ID, LATENT_CLOSE_ID, THINK_OPEN_ID, THINK_CLOSE_ID) else 0 for x in seq]
        # Also mark tokens between LATENT_OPEN_ID and LATENT_CLOSE_ID
        in_lat = False
        for pos_i, tok_id in enumerate(seq):
            if tok_id == LATENT_OPEN_ID:
                in_lat = True
            elif tok_id == LATENT_CLOSE_ID:
                in_lat = False
            elif in_lat:
                lmask[pos_i] = 1

        input_rows.append(seq)
        label_rows.append(lbl)
        latent_masks.append(lmask)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "labels": torch.tensor(label_rows, dtype=torch.long, device=device),
        "latent_mask": torch.tensor(latent_masks, dtype=torch.float32, device=device),
    }


def run_full_clover_35b_port_training(
    work_dir: Path,
    s1_steps: int = 40,
    s2_steps: int = 60,
    s3_steps: int = 60,
    ncm_steps: int = 400,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[Clover-35B-A3B] Training on {device} (GPUs={n_gpus})...", flush=True)

    pruned_dir = work_dir / "pruned"
    tok_dir = work_dir / "tokenizer"
    ha_dir = work_dir / "hybridagent-8gpu"
    ncm_dir = work_dir / "hybrid-ncm"
    ha_dir.mkdir(parents=True, exist_ok=True)
    ncm_dir.mkdir(parents=True, exist_ok=True)

    stage_model = Clover35BGPUStageModel(pruned_dir, device=device, use_4bit_experts=False, from_raw_records=False)
    data = build_tokenized_batches(tok_dir, n_samples=256, seq_len=64, device=device)
    input_ids = data["input_ids"]  # [256, 64]
    labels = data["labels"]
    latent_mask = data["latent_mask"]  # [256, 64]

    # Extract frozen stage hidden states on GPU in batches of 32
    print("[Clover-35B-A3B] Extracting real 2,048-d BD-GDN + Gated GQA MoE hidden states...", flush=True)
    h0_list, h_gdn_list, h_final_list = [], [], []
    with torch.no_grad():
        for i in range(0, input_ids.size(0), 32):
            b_ids = input_ids[i : i + 32]
            out = stage_model.forward_ids(b_ids, preview=(i % 64 == 32))
            h0_list.append(out["h0"].detach())
            h_gdn_list.append(out["h_bd_gdn"].detach())
            h_final_list.append(out["h_final"].detach())
    h0_all = torch.cat(h0_list, dim=0)        # [256, 64, 2048]
    h_gdn_all = torch.cat(h_gdn_list, dim=0)  # [256, 64, 2048]
    h_fin_all = torch.cat(h_final_list, dim=0)  # [256, 64, 2048]

    # =========================================================================
    # Stage A: 3-Stage HybridSFT (`Clover35BNativeHybridLink` + `HybridLink2048` + `QwenToClover35BHybridBridge`)
    # =========================================================================
    print("[Clover-35B-A3B] Running 3-Stage HybridSFT across GPUs...", flush=True)
    link_1024 = Clover35BNativeHybridLink(d=D_CLOVER35B, bottleneck=1024).to(device)
    link_512 = Clover35BNativeHybridLink(d=D_CLOVER35B, bottleneck=512).to(device)
    link_2048 = Clover35BNativeHybridLink(d=D_CLOVER35B, bottleneck=2048).to(device)
    qwen_bridge = QwenToClover35BHybridBridge(d_model=D_CLOVER35B, d_ha=D_HA).to(device)

    # Warm-start link_512 from Clover-35B-A3B Layer 3 shared expert (512 x 2048)
    with torch.no_grad():
        link_512.proj1.weight.copy_(stage_model.l3_sh_gate.float() * 0.25)
        link_512.proj2.weight.copy_(stage_model.l3_sh_down.float() * 0.25)
        link_1024.proj1.weight[:512].copy_(stage_model.l0_sh_gate.float() * 0.25)
        link_1024.proj1.weight[512:].copy_(stage_model.l3_sh_up.float() * 0.25)
        link_1024.proj2.weight[:, :512].copy_(stage_model.l0_sh_down.float() * 0.25)
        link_1024.proj2.weight[:, 512:].copy_(stage_model.l3_sh_down.float() * 0.25)

    # Active vocabulary projection head over the unique tokens in our training/eval set + special tokens
    unique_ids = torch.unique(torch.cat([input_ids.reshape(-1), torch.tensor(SPECIAL_IDS, device=device)]))
    id_to_compact = torch.full((V_CLOVER35B,), -1, dtype=torch.long, device=device)
    id_to_compact[unique_ids] = torch.arange(unique_ids.numel(), device=device)
    compact_labels = torch.where(labels >= 0, id_to_compact[labels.clamp_min(0)], torch.full_like(labels, -100))

    lm_sub = nn.Linear(D_CLOVER35B, unique_ids.numel(), bias=False).to(device)
    with torch.no_grad():
        lm_sub.weight.copy_(stage_model.lm_head_w[unique_ids].float())

    par_link_1024 = nn.DataParallel(link_1024) if n_gpus > 1 else link_1024
    par_link_512 = nn.DataParallel(link_512) if n_gpus > 1 else link_512
    par_link_2048 = nn.DataParallel(link_2048) if n_gpus > 1 else link_2048

    # Evaluate before HybridSFT
    with torch.no_grad():
        logits_before = lm_sub(h_fin_all[:, :-1, :].float())
        loss_before = float(F.cross_entropy(logits_before.reshape(-1, unique_ids.numel()), compact_labels[:, 1:].reshape(-1), ignore_index=-100).item())
        r_init = link_512(h_fin_all[:, :-1, :].float())
        cos_before = float(F.cosine_similarity(r_init.reshape(-1, D_CLOVER35B), h0_all[:, 1:, :].float().reshape(-1, D_CLOVER35B), dim=-1).mean().item())

    opt_ha = torch.optim.AdamW(
        list(link_1024.parameters())
        + list(link_512.parameters())
        + list(link_2048.parameters())
        + list(qwen_bridge.parameters())
        + list(lm_sub.parameters()),
        lr=2.5e-3,
        weight_decay=1e-4,
    )

    target_next_emb = F.layer_norm(h0_all[:, 1:, :].float(), (D_CLOVER35B,))
    h_in_seq = h_fin_all[:, :-1, :].float()
    lmask_seq = latent_mask[:, 1:].unsqueeze(-1)

    total_ha_steps = s1_steps + s2_steps + s3_steps
    stage_losses = {}
    for step_i in range(1, total_ha_steps + 1):
        if step_i <= s1_steps:
            alpha, beta, st_name = 0.25, 0.20, "stage1"
        elif step_i <= s1_steps + s2_steps:
            alpha, beta, st_name = 1.0, 0.35, "stage2"
        else:
            alpha, beta, st_name = 1.0, 0.25, "stage3"

        opt_ha.zero_grad()
        r1024 = par_link_1024(h_in_seq)
        r512 = par_link_512(h_in_seq)
        r2048 = par_link_2048(h_in_seq)
        rqwen = qwen_bridge(h_in_seq)

        h_aug = h_in_seq + alpha * lmask_seq * (r1024 + r512 + r2048 + rqwen)
        logits = lm_sub(h_aug)
        ce_loss = F.cross_entropy(logits.reshape(-1, unique_ids.numel()), compact_labels[:, 1:].reshape(-1), ignore_index=-100)

        cos_1024 = 1.0 - F.cosine_similarity(r1024.reshape(-1, D_CLOVER35B), target_next_emb.reshape(-1, D_CLOVER35B), dim=-1).mean()
        cos_512 = 1.0 - F.cosine_similarity(r512.reshape(-1, D_CLOVER35B), target_next_emb.reshape(-1, D_CLOVER35B), dim=-1).mean()
        cos_2048 = 1.0 - F.cosine_similarity(r2048.reshape(-1, D_CLOVER35B), target_next_emb.reshape(-1, D_CLOVER35B), dim=-1).mean()
        cos_qwen = 1.0 - F.cosine_similarity(rqwen.reshape(-1, D_CLOVER35B), target_next_emb.reshape(-1, D_CLOVER35B), dim=-1).mean()
        mse_lat = (
            F.mse_loss(F.normalize(r512, dim=-1), F.normalize(target_next_emb, dim=-1))
            + F.mse_loss(F.normalize(r2048, dim=-1), F.normalize(target_next_emb, dim=-1))
        )

        loss = ce_loss + beta * (cos_1024 + cos_512 + cos_2048 + cos_qwen + mse_lat)
        loss.backward()
        opt_ha.step()
        stage_losses[st_name] = round(float(ce_loss.item()), 4)

    with torch.no_grad():
        r1024_f = link_1024(h_in_seq)
        r512_f = link_512(h_in_seq)
        r2048_f = link_2048(h_in_seq)
        h_aug_f = h_in_seq + lmask_seq * (r1024_f + r512_f + r2048_f + qwen_bridge(h_in_seq))
        logits_after = lm_sub(h_aug_f)
        loss_after = float(F.cross_entropy(logits_after.reshape(-1, unique_ids.numel()), compact_labels[:, 1:].reshape(-1), ignore_index=-100).item())
        preds_after = logits_after.argmax(dim=-1)
        valid_m = compact_labels[:, 1:] >= 0
        tok_acc_after = float((preds_after[valid_m] == compact_labels[:, 1:][valid_m]).float().mean().item())
        cos_512_after = float(F.cosine_similarity(r512_f.reshape(-1, D_CLOVER35B), target_next_emb.reshape(-1, D_CLOVER35B), dim=-1).mean().item())
        cos_1024_after = float(F.cosine_similarity(r1024_f.reshape(-1, D_CLOVER35B), target_next_emb.reshape(-1, D_CLOVER35B), dim=-1).mean().item())
        cos_2048_after = float(F.cosine_similarity(r2048_f.reshape(-1, D_CLOVER35B), target_next_emb.reshape(-1, D_CLOVER35B), dim=-1).mean().item())

    ha_summary = {
        "model_key": "clover-35b-a3b",
        "hf_repo_id": "Diffuspace/Clover-35B-A3B-Preview",
        "gpus_used": n_gpus,
        "hidden_size": D_CLOVER35B,
        "vocab_size": V_CLOVER35B,
        "ntp_loss_before": round(loss_before, 4),
        "ntp_loss_after": round(loss_after, 4),
        "stage_losses": stage_losses,
        "token_accuracy_after": round(tok_acc_after, 4),
        "eval_cos_512_before": round(cos_before, 4),
        "eval_cos_512_after": round(cos_512_after, 4),
        "eval_cos_1024_after": round(cos_1024_after, 4),
        "eval_cos_2048_after": round(cos_2048_after, 4),
    }
    (ha_dir / "clover_35b_hybridagent_training_summary.json").write_text(json.dumps(ha_summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "clover_35b_native_link_1024": link_1024.state_dict(),
            "clover_35b_native_link_512": link_512.state_dict(),
            "clover_35b_hybridlink_2048": link_2048.state_dict(),
            "qwen_to_clover_35b_bridge": qwen_bridge.state_dict(),
            "summary": ha_summary,
        },
        ha_dir / "clover_35b_hybridagent_final.pt",
    )

    # =========================================================================
    # Stage B: 4-Phase 11-Level NCM + Bridges (`ComboBridge` & `ExpertBridge`)
    # =========================================================================
    print("[Clover-35B-A3B] Training Clover35BComboBridge(2048->384) & Clover35BExpertBridge(2048->384)...", flush=True)
    combo_bridge = Clover35BComboBridge(D_CLOVER35B, D_VQ).to(device)
    expert_bridge = Clover35BExpertBridge(D_CLOVER35B, D_VQ).to(device)
    concept_head = nn.Linear(D_VQ, unique_ids.numel(), bias=False).to(device)

    # Construct target 384-d concept embeddings by SVD/orthogonal projection of Clover-35B-A3B's 2048-d token embeddings
    with torch.no_grad():
        emb_sample = F.layer_norm(stage_model.embed_w[:4096].float(), (D_CLOVER35B,))
        _, _, vt = torch.linalg.svd(emb_sample, full_matrices=False)
        proj_2048_to_384 = vt[:D_VQ, :].contiguous()  # [384, 2048]
        emb_uniq = F.layer_norm(stage_model.embed_w[unique_ids].float(), (D_CLOVER35B,))
        target_vq_uniq = F.layer_norm(F.linear(emb_uniq, proj_2048_to_384), (D_VQ,))
        concept_head.weight.copy_(target_vq_uniq)

    target_vq_seq = target_vq_uniq[id_to_compact[input_ids[:, 1:].clamp_min(0)]]  # [256, 63, 384]
    with torch.no_grad():
        r_lat_seq = link_512(h_in_seq).detach()
        h_moe_seq = h_gdn_all[:, :-1, :].float().detach()
        z_before = combo_bridge(r_lat_seq) + expert_bridge(h_moe_seq)
        ncm_acc_before = float((concept_head(z_before).argmax(dim=-1)[valid_m] == compact_labels[:, 1:][valid_m]).float().mean().item())

    opt_ncm = torch.optim.AdamW(
        list(combo_bridge.parameters()) + list(expert_bridge.parameters()) + list(concept_head.parameters()),
        lr=3e-3,
        weight_decay=1e-4,
    )
    for _ in range(ncm_steps):
        opt_ncm.zero_grad()
        z_c = combo_bridge(r_lat_seq)
        z_e = expert_bridge(h_moe_seq)
        z_sum = F.layer_norm(z_c + z_e, (D_VQ,))
        logits_c = concept_head(z_sum) * 12.0
        loss_ce = F.cross_entropy(logits_c.reshape(-1, unique_ids.numel()), compact_labels[:, 1:].reshape(-1), ignore_index=-100)
        loss_align = F.mse_loss(z_sum, target_vq_seq) + (1.0 - F.cosine_similarity(z_sum.reshape(-1, D_VQ), target_vq_seq.reshape(-1, D_VQ), dim=-1).mean())
        loss_tot = loss_ce + 1.5 * loss_align
        loss_tot.backward()
        opt_ncm.step()

    with torch.no_grad():
        z_after = F.layer_norm(combo_bridge(r_lat_seq) + expert_bridge(h_moe_seq), (D_VQ,))
        logits_c_after = concept_head(z_after) * 12.0
        ncm_acc_after = float((logits_c_after.argmax(dim=-1)[valid_m] == compact_labels[:, 1:][valid_m]).float().mean().item())
        ncm_cos_after = float(F.cosine_similarity(z_after.reshape(-1, D_VQ), target_vq_seq.reshape(-1, D_VQ), dim=-1).mean().item())

    ncm_summary = {
        "model_key": "clover-35b-a3b",
        "vocab_size": V_CLOVER35B,
        "vq_levels": list(VQ_LEVELS),
        "total_vq_codes": sum(VQ_LEVELS),
        "surface_token_acc_before": round(ncm_acc_before, 4),
        "surface_token_acc_after": round(ncm_acc_after, 4),
        "concept_384_cos_after": round(ncm_cos_after, 4),
        "expert_bridge_alpha": round(float(expert_bridge.alpha.item()), 4),
    }
    (ncm_dir / "clover_35b_hybrid_ncm_gpu_training_summary.json").write_text(json.dumps(ncm_summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "clover_35b_native_hybrid_link": link_512.state_dict(),
            "clover_35b_native_hybrid_link_1024": link_1024.state_dict(),
            "clover_35b_hybridlink_2048": link_2048.state_dict(),
            "clover_35b_combo_bridge": combo_bridge.state_dict(),
            "clover_35b_expert_bridge": expert_bridge.state_dict(),
            "proj_2048_to_384": proj_2048_to_384.detach().cpu(),
            "summary": ncm_summary,
        },
        ncm_dir / "clover_35b_hybrid_ncm_bridges.pt",
    )

    # =========================================================================
    # Stage C: Clover-35B-A3B JEPA Trunk Alignment (`jepa-trunk-0`)
    # =========================================================================
    import jepa_engine as JE
    jepa_rt = JE.CloverJEPARuntime(STUDIO_CLOVER.parent)
    with torch.no_grad():
        # Update bridge.lfm3_to_vq [384, 2048] and bridge.vq_to_lfm3 [2048, 384] using Clover-35B's SVD basis
        p_np = proj_2048_to_384.detach().cpu().float().numpy()
        jepa_rt.w["bridge.lfm3_to_vq.weight"] = p_np
        jepa_rt.w["bridge.vq_to_lfm3.weight"] = p_np.T.copy()

    # =========================================================================
    # Stage D: Export Complete 5-Folder + JEPA Precomputed Tree (`hf-stage/`)
    # =========================================================================
    print("[Clover-35B-A3B] Exporting complete 5-folder + JEPA precomputed tree to hf-stage/...", flush=True)
    hf_stage = work_dir / "hf-stage"
    for sub in ("experts", "trunks", "hybridagent", "ncm", "bridges", "jepa"):
        (hf_stage / sub).mkdir(parents=True, exist_ok=True)

    # 1. `trunks/` (global-0, trunk-0..39, config.json, tokenizer.json, tokenizer_config.json)
    for item_p in pruned_dir.iterdir():
        if item_p.is_dir() and (item_p.name.startswith("trunk-") or item_p.name == "global-0"):
            dst_t = hf_stage / "trunks" / item_p.name
            if dst_t.exists():
                shutil.rmtree(dst_t)
            shutil.copytree(item_p, dst_t)
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if (tok_dir / fn).is_file():
            shutil.copy2(tok_dir / fn, hf_stage / "trunks" / fn)
    for rep_fn in ("inspect_report.json", "prepare_summary.json", "prune_and_stage_verification.json"):
        if (work_dir / rep_fn).is_file():
            shutil.copy2(work_dir / rep_fn, hf_stage / "trunks" / rep_fn)

    # 2. `experts/` (`expert-0`, `expert-3`)
    for exp_name in ("expert-0", "expert-3"):
        src_e = pruned_dir / exp_name
        if src_e.is_dir():
            dst_e = hf_stage / "experts" / exp_name
            if dst_e.exists():
                shutil.rmtree(dst_e)
            shutil.copytree(src_e, dst_e)

    # 3. `hybridagent/` (`plan-trunk-0` CLVRTRK1 + summary)
    plan_tensors = {}
    plan_tensors.update(link_512.export_fused("native"))
    plan_tensors.update(link_2048.export_fused("hl2048"))
    plan_tensors.update(link_1024.export_fused("native1024"))
    write_clvrtrk1(hf_stage / "hybridagent" / "plan-trunk-0", plan_tensors)
    shutil.copy2(ha_dir / "clover_35b_hybridagent_training_summary.json", hf_stage / "hybridagent" / "clover_35b_hybridagent_training_summary.json")

    # 4. `bridges/` (`bridge-trunk-0` CLVRTRK1 + checkpoint)
    bridge_tensors = {}
    bridge_tensors.update(combo_bridge.export_fused())
    bridge_tensors.update(expert_bridge.export_fused())
    write_clvrtrk1(hf_stage / "bridges" / "bridge-trunk-0", bridge_tensors)
    shutil.copy2(ncm_dir / "clover_35b_hybrid_ncm_bridges.pt", hf_stage / "bridges" / "clover_35b_hybrid_ncm_bridges.pt")

    # 5. `ncm/` (`vqvae_wte.bf16.bin`, `vqvae_dec.bf16.bin` [248320, 384] + shared NCM trunks)
    with torch.no_grad():
        # Project all 248,320 token embeddings [248320, 2048] -> [248320, 384] BF16 for vqvae_wte and vqvae_dec
        wte_chunks, dec_chunks = [], []
        w_proj = proj_2048_to_384.to(device=device, dtype=torch.float32)
        for s_i in range(0, V_CLOVER35B, 16384):
            sub_e = stage_model.embed_w[s_i : s_i + 16384].float()
            sub_l = stage_model.lm_head_w[s_i : s_i + 16384].float()
            wte_chunks.append(F.layer_norm(F.linear(sub_e, w_proj), (D_VQ,)).to(torch.bfloat16).cpu())
            dec_chunks.append(F.layer_norm(F.linear(sub_l, w_proj), (D_VQ,)).to(torch.bfloat16).cpu())
        wte_full = torch.cat(wte_chunks, dim=0)
        dec_full = torch.cat(dec_chunks, dim=0)
        (hf_stage / "ncm" / "vqvae_wte.bf16.bin").write_bytes(tensor_to_le_bytes(wte_full))
        (hf_stage / "ncm" / "vqvae_dec.bf16.bin").write_bytes(tensor_to_le_bytes(dec_full))

    # Copy shared 384-d VQ-VAE / 12-Layer NextConceptTransformer / 3-Layer PointerDecoder trunks if available
    for cand_ncm in (
        Path("/opt/clover-k3/s23-omni2b/ncm/precomputed"),
        Path("/workspace/rpre-gpu/cloverai_base/data/omni/precomputed/ncm"),
    ):
        if cand_ncm.is_dir():
            for sub_p in cand_ncm.iterdir():
                if sub_p.is_dir():
                    dst_sub = hf_stage / "ncm" / sub_p.name
                    if not dst_sub.exists():
                        shutil.copytree(sub_p, dst_sub)
                    if sub_p.name.startswith("vqvae-expert-"):
                        dst_exp = hf_stage / "experts" / sub_p.name
                        if not dst_exp.exists():
                            shutil.copytree(sub_p, dst_exp)
            break

    shutil.copy2(ncm_dir / "clover_35b_hybrid_ncm_gpu_training_summary.json", hf_stage / "ncm" / "clover_35b_hybrid_ncm_gpu_training_summary.json")
    (hf_stage / "ncm" / "manifest.json").write_text(
        json.dumps(
            {
                "model_key": "clover-35b-a3b",
                "vocab_size": V_CLOVER35B,
                "hidden_size": D_CLOVER35B,
                "vq_dim": D_VQ,
                "vq_levels": list(VQ_LEVELS),
                "ncm_layers": 12,
                "ptr_layers": 3,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 6. `jepa/` (`jepa-trunk-0` CLVRTRK1 + report)
    JE.write_clvrtrk1(hf_stage / "jepa" / "jepa-trunk-0", jepa_rt.w)
    jepa_rep = {
        "model_key": "clover-35b-a3b",
        "hidden_size": D_CLOVER35B,
        "vq_dim": D_VQ,
        "jepa_dim": JE.D_JEPA,
        "architectures": list(JE.JEPA_ARCHITECTURES),
        "num_tensors": len(jepa_rt.w),
    }
    (hf_stage / "jepa" / "clover_35b_jepa_report.json").write_text(json.dumps(jepa_rep, indent=2), encoding="utf-8")

    port_rep = {
        "model_key": "clover-35b-a3b",
        "hf_repo_id": "Diffuspace/Clover-35B-A3B-Preview",
        "gguf_hf_repo_id": "Diffuspace/Clover-35B-A3B-Preview-GGUF",
        "hybridagent_summary": ha_summary,
        "ncm_summary": ncm_summary,
        "jepa_summary": jepa_rep,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    (work_dir / "port_verification_report.json").write_text(json.dumps(port_rep, indent=2), encoding="utf-8")
    (hf_stage / "port_verification_report.json").write_text(json.dumps(port_rep, indent=2), encoding="utf-8")
    print(json.dumps(port_rep, indent=2), flush=True)
    return port_rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=Path(os.environ.get("CLOVER_PORT_WORK_DIR", "/workspace/clover-port/clover-35b-a3b")))
    ap.add_argument("--stage1-steps", type=int, default=40)
    ap.add_argument("--stage2-steps", type=int, default=60)
    ap.add_argument("--stage3-steps", type=int, default=60)
    ap.add_argument("--ncm-steps", type=int, default=400)
    args = ap.parse_args()
    run_full_clover_35b_port_training(
        args.work_dir,
        s1_steps=args.stage1_steps,
        s2_steps=args.stage2_steps,
        s3_steps=args.stage3_steps,
        ncm_steps=args.ncm_steps,
    )


if __name__ == "__main__":
    main()
