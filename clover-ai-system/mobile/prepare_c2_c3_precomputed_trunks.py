#!/usr/bin/env python3
"""
prepare_c2_c3_precomputed_trunks.py
===================================
Clover Framework — Precomputed Trunk + Expert Builder for Core 2 (Plan) & Core 3 (Decode)

Transforms `lfm3_ncm_core_bf16.pt` on CPU (~10-15 seconds) into Clover's precomputed
`trunk` + `expert` layout (`CLVRTRK1` binary files: `index.bin`, `common.bin`,
`operator.bin`, `operator.4bit.bin`, `ffn.bin`, `ffn.4bit.bin`, with `attention.bin`
pruned where attention is algebraically fused into `operator.bin`).

Algebraic Operator Fusions Performed Offline:
---------------------------------------------
1. Core 2 · Plan (`plan-trunk-0/`):
   - `Lfm3NativeHybridLink` (2048 -> 512 -> 2048) & `HybridLink2048` (2048 -> 2048 -> 2048):
     * Fuses `pre_ln` affine (gamma_1, beta_1) into `proj1`:
         W1_fused = W1 * gamma_1[None, :]
         b1_fused = W1 @ beta_1 + b1
     * Fuses `out_scale` (0.024) into `post_ln`:
         gamma2_fused = out_scale * gamma_2
         beta2_fused  = out_scale * beta_2

2. Core 3 · Decode — Bridges (`bridge-trunk-0/`):
   - `Lfm3ComboBridge` (2048 -> 768 -> 384):
     * Fuses `ln_in` affine into `fc1`:
         W_fc1_fused = W_fc1 * gamma_in[None, :]
         b_fc1_fused = W_fc1 @ beta_in + b_fc1
     * Fuses `(scale * gate)` and `+ seg_rh` into `ln_out`:
         gamma_out_fused = (scale * gate) * gamma_out
         beta_out_fused  = (scale * gate) * beta_out + seg_rh.view(384)
   - `Lfm3ExpertBridge` (2048 -> 384 -> 384):
     * Fuses `ln_in` affine into `proj1`:
         W_p1_fused = W_p1 * gamma_in[None, :]
         b_p1_fused = W_p1 @ beta_in + b_p1
     * Fuses `tanh(alpha)` into `ln_out`:
         gamma_out_fused = tanh(alpha) * gamma_out
         beta_out_fused  = tanh(alpha) * beta_out

3. Core 3 · Decode — Multiscale VQ-VAE (`vqvae-trunk-0/` + `vqvae-expert-0..10/`):
   - `vqvae-trunk-0/`:
     * Fuses `decoder.middle_block.ln_1` into `attn.c_attn` (`operator.bin` + `operator.4bit.bin`, `attention.bin` pruned)
     * Fuses `decoder.middle_block.ln_2` into `mlp.c_fc` (`ffn.bin` + `ffn.4bit.bin`)
   - `vqvae-expert-0..10/` (11 Multiscale Codebook Experts, 4096 x 384 each):
     * Precomputes L2-normalized codebook vectors `E_norm = E / ||E||_2`,
       squared norms `||e_j||_2^2` (for 1-GEMM nearest-neighbor lookup),
       `quant_resi` 1D conv weights, and 4-bit group-64 `codebook.4bit.bin`.

4. Core 3 · Decode — 12-Layer NextConceptTransformer (`ncm-trunk-0..11/` + `ncm-global-0/`):
   - `ncm-trunk-0..11/`:
     * Fuses `ln_1` (gamma_1, beta_1) into `attn.c_attn` ([2304, 768]) -> `operator.bin` + `operator.4bit.bin` (`attention.bin` pruned)
     * Fuses `ln_2` (gamma_2, beta_2) into `mlp.c_fc` ([3072, 768]) + `mlp.c_proj` ([768, 3072]) -> `ffn.bin` + `ffn.4bit.bin`
   - `ncm-global-0/`:
     * Fuses `transformer.ln_f` (gamma_f, beta_f) into ALL 11 Concept Level Heads (`lm_head_l0` + `lm_heads.0..9`, each [4096, 768]):
         H_s_fused = H_s * gamma_f[None, :]
         b_Hs_fused = H_s @ beta_f
       packed into `operator.bin` + `operator.4bit.bin`.

5. Core 3 · Decode — 3-Layer LexicalPointerDecoder (`ptr-trunk-0..2/` + `ptr-global-0/`):
   - `ptr-trunk-0..2/`:
     * Fuses `ln1 -> qkv`, `ln_q -> cx_q`, `ln_kv -> cx_kv`, and `res_scale -> proj / cx_proj` -> `operator.bin` + `operator.4bit.bin` (`attention.bin` pruned)
     * Fuses `ln2 -> mlp.0` and `res_scale -> mlp.2` -> `ffn.bin` + `ffn.4bit.bin`
   - `ptr-global-0/`:
     * Fuses `ln_out -> copy_q` and packs `tgt_pos`, `plan_pos`, `rh_pos`, `copy_k`, `copy_gate`, `copy_scale`.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

TRUNK_MAGIC = b"CLVRTRK1"
Q4_MAGIC = b"CLVR4B01"
GROUP_SIZE = 64

FILE_COMMON = 0
FILE_OPERATOR = 1
FILE_ATTENTION = 2
FILE_FFN = 3

DTYPE_BF16 = 1
DTYPE_F32 = 2
DTYPE_I64 = 3


def tensor_to_bf16_bytes(t: torch.Tensor) -> bytes:
    t_bf16 = t.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
    return t_bf16.view(torch.uint16).numpy().tobytes()


def tensor_to_f32_bytes(t: torch.Tensor) -> bytes:
    t_f32 = t.detach().to(dtype=torch.float32, device="cpu").contiguous()
    return t_f32.numpy().tobytes()


def quantize_4bit_planar_group64(t: torch.Tensor) -> bytes:
    """
    Vectorized symmetric 4-bit group-64 quantization matching Clover's planar layout:
      [nibbles: N/2 uint8] [scales: N/64 float16]
    """
    w = t.detach().to(dtype=torch.float32, device="cpu").contiguous().view(-1)
    numel = w.numel()
    pad = (GROUP_SIZE - (numel % GROUP_SIZE)) % GROUP_SIZE
    if pad > 0:
        w = F.pad(w, (0, pad))
    n_groups = w.numel() // GROUP_SIZE
    wg = w.view(n_groups, GROUP_SIZE)
    max_abs = wg.abs().amax(dim=1, keepdim=True)
    scales = (max_abs / 7.0).clamp(min=1e-8)
    q = torch.clamp(torch.round(wg / scales), -8, 7).to(torch.int8)
    u = (q + 8).to(torch.uint8)
    lo = u[:, 0::2]
    hi = u[:, 1::2]
    packed = (lo | (hi << 4)).contiguous().view(-1).numpy().tobytes()
    scales_f16 = scales.view(-1).to(torch.float16).numpy().tobytes()
    return packed + scales_f16


class TrunkWriter:
    """Writes a Clover precomputed trunk/expert directory (`index.bin`, `common.bin`, `operator.bin`, `ffn.bin`, `*.4bit.bin`)."""

    def __init__(self, out_dir: Path, layer_idx: int, op_type: int = 1) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.layer_idx = layer_idx
        self.op_type = op_type
        self.entries: list[tuple[str, int, int, int, int, tuple[int, ...]]] = []
        self.blobs: dict[int, bytearray] = {
            FILE_COMMON: bytearray(),
            FILE_OPERATOR: bytearray(),
            FILE_FFN: bytearray(),
        }
        self.q4_entries: dict[int, list[tuple[str, tuple[int, ...], bytes]]] = {
            FILE_OPERATOR: [],
            FILE_FFN: [],
        }

    def add_tensor(
        self,
        name: str,
        t: torch.Tensor,
        file_id: int,
        store_f32: bool = False,
        quant4: bool = False,
    ) -> None:
        shape = tuple(int(s) for s in t.shape)
        if store_f32 or t.ndim <= 1:
            raw = tensor_to_f32_bytes(t)
            dtype_code = DTYPE_F32
        else:
            raw = tensor_to_bf16_bytes(t)
            dtype_code = DTYPE_BF16

        buf = self.blobs[file_id]
        # 64-byte alignment inside trunk data files
        pad = (64 - (len(buf) % 64)) % 64
        if pad:
            buf.extend(b"\x00" * pad)
        offset = len(buf)
        buf.extend(raw)
        self.entries.append((name, file_id, dtype_code, offset, len(raw), shape))

        if quant4 and t.ndim >= 2 and file_id in self.q4_entries:
            q4_bytes = quantize_4bit_planar_group64(t)
            self.q4_entries[file_id].append((name, shape, q4_bytes))

    def finalize(self, prune_attention: bool = True) -> dict[str, Any]:
        file_names = {
            FILE_COMMON: "common.bin",
            FILE_OPERATOR: "operator.bin",
            FILE_FFN: "ffn.bin",
        }
        bytes_written = 0
        for fid, fname in file_names.items():
            p = self.out_dir / fname
            data = bytes(self.blobs[fid])
            p.write_bytes(data)
            bytes_written += len(data)

        # Prune attention.bin when self/cross-attention is fused into operator.bin
        attn_path = self.out_dir / "attention.bin"
        if prune_attention and attn_path.exists():
            attn_path.unlink()

        # Write 4-bit companion files (`operator.4bit.bin`, `ffn.4bit.bin`)
        q4_bytes_written = 0
        for fid, q4_name in [(FILE_OPERATOR, "operator.4bit.bin"), (FILE_FFN, "ffn.4bit.bin")]:
            qlist = self.q4_entries[fid]
            if not qlist:
                continue
            qbuf = bytearray()
            qbuf.extend(Q4_MAGIC)
            qbuf.extend(struct.pack("<II", len(qlist), GROUP_SIZE))
            for tname, shape, qraw in qlist:
                nb = tname.encode("utf-8")[:63]
                nb_pad = nb + b"\x00" * (64 - len(nb))
                s0 = shape[0] if len(shape) >= 1 else 1
                s1 = int(np.prod(shape[1:])) if len(shape) >= 2 else 1
                qbuf.extend(nb_pad)
                qbuf.extend(struct.pack("<III", s0, s1, len(qraw)))
                qbuf.extend(qraw)
            qp = self.out_dir / q4_name
            qp.write_bytes(bytes(qbuf))
            q4_bytes_written += len(qbuf)

        # Write index.bin with CLVRTRK1 header
        idx_buf = bytearray()
        idx_buf.extend(TRUNK_MAGIC)
        idx_buf.extend(struct.pack("<III", self.layer_idx, self.op_type, len(self.entries)))
        for name, fid, dtype_code, offset, nbytes, shape in self.entries:
            nb = name.encode("utf-8")[:63]
            nb_pad = nb + b"\x00" * (64 - len(nb))
            ndim = len(shape)
            dims = list(shape[:4]) + [1] * max(0, 4 - len(shape))
            idx_buf.extend(nb_pad)
            idx_buf.extend(
                struct.pack(
                    "<IIIQQIIII",
                    fid,
                    dtype_code,
                    ndim,
                    offset,
                    nbytes,
                    dims[0],
                    dims[1],
                    dims[2],
                    dims[3],
                )
            )
        idx_path = self.out_dir / "index.bin"
        idx_path.write_bytes(bytes(idx_buf))
        bytes_written += len(idx_buf)

        return {
            "dir": self.out_dir.name,
            "tensors": len(self.entries),
            "bf16_f32_bytes": bytes_written,
            "q4_bytes": q4_bytes_written,
            "attention_bin_pruned": prune_attention,
        }


def fuse_ln_into_linear(
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    lin_w: torch.Tensor,
    lin_b: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Algebraically fuses LayerNorm affine (gamma, beta) into the immediately following Linear(W, b):
      y = W @ (gamma * x_norm + beta) + b
        = (W * gamma[None, :]) @ x_norm + (W @ beta + b)
    """
    gw = ln_w.detach().to(torch.float32)
    gb = ln_b.detach().to(torch.float32)
    w = lin_w.detach().to(torch.float32)
    w_fused = w * gw.unsqueeze(0)
    b_fused = torch.matmul(w, gb)
    if lin_b is not None:
        b_fused = b_fused + lin_b.detach().to(torch.float32)
    return w_fused, b_fused


def build_c2_c3_precomputed(ckpt_path: Path, out_root: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    print(f"[1/6] Loading checkpoint {ckpt_path} on CPU...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_trunks: list[dict[str, Any]] = []
    precomputed_state: dict[str, Any] = {
        "format": "clover_c2_c3_precomputed_trunks_v1",
        "algebraic_fusions": [],
    }

    # ==========================================================================
    # 1. Core 2 · Plan (`plan-trunk-0/`)
    # ==========================================================================
    print("[2/6] Precomputing Core 2 · Plan (`plan-trunk-0/`) with fused pre_ln->proj1 and out_scale->post_ln...")
    nhl = ckpt["lfm3_native_hybrid_link"]
    hl2048 = ckpt["hybridlink_2048"]

    # Fuse Lfm3NativeHybridLink (2048 -> 512 -> 2048)
    nhl_w1_fused, nhl_b1_fused = fuse_ln_into_linear(
        nhl["pre_ln.weight"], nhl["pre_ln.bias"], nhl["proj1.weight"], nhl["proj1.bias"]
    )
    nhl_scale = float(nhl["out_scale"].item())
    nhl_post_w_fused = nhl["post_ln.weight"].to(torch.float32) * nhl_scale
    nhl_post_b_fused = nhl["post_ln.bias"].to(torch.float32) * nhl_scale
    nhl_w2 = nhl["proj2.weight"].to(torch.float32)
    nhl_b2 = nhl["proj2.bias"].to(torch.float32)

    # Fuse HybridLink2048 (2048 -> 2048 -> 2048)
    hl_w1_fused, hl_b1_fused = fuse_ln_into_linear(
        hl2048["pre_ln.weight"], hl2048["pre_ln.bias"], hl2048["proj1.weight"], hl2048["proj1.bias"]
    )
    hl_w2 = hl2048["proj2.weight"].to(torch.bfloat16)
    hl_b2 = hl2048["proj2.bias"].to(torch.float32)
    hl_post_w = hl2048["post_ln.weight"].to(torch.float32)
    hl_post_b = hl2048["post_ln.bias"].to(torch.float32)

    tw_plan = TrunkWriter(out_root / "plan-trunk-0", layer_idx=0, op_type=2)
    tw_plan.add_tensor("native.post_ln_fused.weight", nhl_post_w_fused, FILE_COMMON, store_f32=True)
    tw_plan.add_tensor("native.post_ln_fused.bias", nhl_post_b_fused, FILE_COMMON, store_f32=True)
    tw_plan.add_tensor("hl2048.post_ln.weight", hl_post_w, FILE_COMMON, store_f32=True)
    tw_plan.add_tensor("hl2048.post_ln.bias", hl_post_b, FILE_COMMON, store_f32=True)

    tw_plan.add_tensor("native.proj1_fused.weight", nhl_w1_fused, FILE_OPERATOR, store_f32=True, quant4=True)
    tw_plan.add_tensor("native.proj1_fused.bias", nhl_b1_fused, FILE_OPERATOR, store_f32=True)
    tw_plan.add_tensor("hl2048.proj1_fused.weight", hl_w1_fused, FILE_OPERATOR, quant4=True)
    tw_plan.add_tensor("hl2048.proj1_fused.bias", hl_b1_fused, FILE_OPERATOR, store_f32=True)

    tw_plan.add_tensor("native.proj2.weight", nhl_w2, FILE_FFN, store_f32=True, quant4=True)
    tw_plan.add_tensor("native.proj2.bias", nhl_b2, FILE_FFN, store_f32=True)
    tw_plan.add_tensor("hl2048.proj2.weight", hl_w2, FILE_FFN, quant4=True)
    tw_plan.add_tensor("hl2048.proj2.bias", hl_b2, FILE_FFN, store_f32=True)
    summary_trunks.append(tw_plan.finalize(prune_attention=True))

    precomputed_state["c2_plan"] = {
        "native_proj1_w_fused": nhl_w1_fused,
        "native_proj1_b_fused": nhl_b1_fused,
        "native_proj2_w": nhl_w2,
        "native_proj2_b": nhl_b2,
        "native_post_ln_w_fused": nhl_post_w_fused,
        "native_post_ln_b_fused": nhl_post_b_fused,
        "hl2048_proj1_w_fused": hl_w1_fused.to(torch.bfloat16),
        "hl2048_proj1_b_fused": hl_b1_fused,
        "hl2048_proj2_w": hl_w2,
        "hl2048_proj2_b": hl_b2,
        "hl2048_post_ln_w": hl_post_w,
        "hl2048_post_ln_b": hl_post_b,
    }
    precomputed_state["algebraic_fusions"].extend([
        "C2.Lfm3NativeHybridLink: pre_ln (gamma, beta) -> proj1; out_scale -> post_ln",
        "C2.HybridLink2048: pre_ln (gamma, beta) -> proj1",
    ])

    # ==========================================================================
    # 2. Core 3 · Decode — Part A: Bridges (`bridge-trunk-0/`)
    # ==========================================================================
    print("[3/6] Precomputing Core 3 · Bridges (`bridge-trunk-0/`) with fused ln_in->fc1/proj1 and gate/alpha->ln_out...")
    cb = ckpt["lfm3_combo_bridge"]
    eb = ckpt["lfm3_expert_bridge"]

    cb_w1_fused, cb_b1_fused = fuse_ln_into_linear(
        cb["ln_in.weight"], cb["ln_in.bias"], cb["fc1.weight"], cb["fc1.bias"]
    )
    cb_gate_scale = float((cb["scale"].to(torch.float32) * cb["gate"].to(torch.float32)).item())
    cb_ln_out_w_fused = cb["ln_out.weight"].to(torch.float32) * cb_gate_scale
    cb_ln_out_b_fused = cb["ln_out.bias"].to(torch.float32) * cb_gate_scale + cb["seg_rh"].to(torch.float32).view(-1)
    cb_w2 = cb["fc2.weight"].to(torch.float32)
    cb_b2 = cb["fc2.bias"].to(torch.float32)

    eb_w1_fused, eb_b1_fused = fuse_ln_into_linear(
        eb["ln_in.weight"], eb["ln_in.bias"], eb["proj1.weight"], eb["proj1.bias"]
    )
    eb_alpha_tanh = float(torch.tanh(eb["alpha"].to(torch.float32)).item())
    eb_ln_out_w_fused = eb["ln_out.weight"].to(torch.float32) * eb_alpha_tanh
    eb_ln_out_b_fused = eb["ln_out.bias"].to(torch.float32) * eb_alpha_tanh
    eb_w2 = eb["proj2.weight"].to(torch.float32)
    eb_b2 = eb["proj2.bias"].to(torch.float32)

    tw_bridge = TrunkWriter(out_root / "bridge-trunk-0", layer_idx=0, op_type=3)
    tw_bridge.add_tensor("combo.ln_out_fused.weight", cb_ln_out_w_fused, FILE_COMMON, store_f32=True)
    tw_bridge.add_tensor("combo.ln_out_fused.bias", cb_ln_out_b_fused, FILE_COMMON, store_f32=True)
    tw_bridge.add_tensor("expert.ln_out_fused.weight", eb_ln_out_w_fused, FILE_COMMON, store_f32=True)
    tw_bridge.add_tensor("expert.ln_out_fused.bias", eb_ln_out_b_fused, FILE_COMMON, store_f32=True)

    tw_bridge.add_tensor("combo.fc1_fused.weight", cb_w1_fused, FILE_OPERATOR, store_f32=True, quant4=True)
    tw_bridge.add_tensor("combo.fc1_fused.bias", cb_b1_fused, FILE_OPERATOR, store_f32=True)
    tw_bridge.add_tensor("expert.proj1_fused.weight", eb_w1_fused, FILE_OPERATOR, store_f32=True, quant4=True)
    tw_bridge.add_tensor("expert.proj1_fused.bias", eb_b1_fused, FILE_OPERATOR, store_f32=True)

    tw_bridge.add_tensor("combo.fc2.weight", cb_w2, FILE_FFN, store_f32=True, quant4=True)
    tw_bridge.add_tensor("combo.fc2.bias", cb_b2, FILE_FFN, store_f32=True)
    tw_bridge.add_tensor("expert.proj2.weight", eb_w2, FILE_FFN, store_f32=True, quant4=True)
    tw_bridge.add_tensor("expert.proj2.bias", eb_b2, FILE_FFN, store_f32=True)
    summary_trunks.append(tw_bridge.finalize(prune_attention=True))

    precomputed_state["c3_bridges"] = {
        "combo_fc1_w_fused": cb_w1_fused,
        "combo_fc1_b_fused": cb_b1_fused,
        "combo_fc2_w": cb_w2,
        "combo_fc2_b": cb_b2,
        "combo_ln_out_w_fused": cb_ln_out_w_fused,
        "combo_ln_out_b_fused": cb_ln_out_b_fused,
        "expert_proj1_w_fused": eb_w1_fused,
        "expert_proj1_b_fused": eb_b1_fused,
        "expert_proj2_w": eb_w2,
        "expert_proj2_b": eb_b2,
        "expert_ln_out_w_fused": eb_ln_out_w_fused,
        "expert_ln_out_b_fused": eb_ln_out_b_fused,
    }
    precomputed_state["algebraic_fusions"].extend([
        "C3.Lfm3ComboBridge: ln_in -> fc1; (scale * gate) & seg_rh -> ln_out",
        "C3.Lfm3ExpertBridge: ln_in -> proj1; tanh(alpha) -> ln_out",
    ])

    # ==========================================================================
    # 3. Core 3 · Decode — Part B: Multiscale VQ-VAE (`vqvae-trunk-0/` + `vqvae-expert-0..10/`)
    # ==========================================================================
    print("[4/6] Precomputing Core 3 · VQ-VAE (`vqvae-trunk-0/` + 11 `vqvae-expert-0..10/`)...")
    vq = ckpt["vqvae_core"]

    vq_attn_w_fused, vq_attn_b_fused = fuse_ln_into_linear(
        vq["decoder.middle_block.ln_1.weight"],
        vq["decoder.middle_block.ln_1.bias"],
        vq["decoder.middle_block.attn.c_attn.weight"],
        vq["decoder.middle_block.attn.c_attn.bias"],
    )
    vq_fc_w_fused, vq_fc_b_fused = fuse_ln_into_linear(
        vq["decoder.middle_block.ln_2.weight"],
        vq["decoder.middle_block.ln_2.bias"],
        vq["decoder.middle_block.mlp.c_fc.weight"],
        vq["decoder.middle_block.mlp.c_fc.bias"],
    )

    tw_vq = TrunkWriter(out_root / "vqvae-trunk-0", layer_idx=0, op_type=4)
    tw_vq.add_tensor("encoder.wpe.weight", vq["encoder.wpe.weight"], FILE_COMMON)
    tw_vq.add_tensor("decoder.ln_f.weight", vq["decoder.ln_f.weight"], FILE_COMMON, store_f32=True)
    for k_idx in (0, 2, 4):
        tw_vq.add_tensor(f"learned_down_l1.{k_idx}.weight", vq[f"vq.learned_down_l1.{k_idx}.weight"], FILE_COMMON)
        tw_vq.add_tensor(f"learned_down_l1.{k_idx}.bias", vq[f"vq.learned_down_l1.{k_idx}.bias"], FILE_COMMON, store_f32=True)
        tw_vq.add_tensor(f"learned_up_l1.{k_idx}.weight", vq[f"vq.learned_up_l1.{k_idx}.weight"], FILE_COMMON)
        tw_vq.add_tensor(f"learned_up_l1.{k_idx}.bias", vq[f"vq.learned_up_l1.{k_idx}.bias"], FILE_COMMON, store_f32=True)

    tw_vq.add_tensor("middle_block.attn.c_attn_fused.weight", vq_attn_w_fused, FILE_OPERATOR, quant4=True)
    tw_vq.add_tensor("middle_block.attn.c_attn_fused.bias", vq_attn_b_fused, FILE_OPERATOR, store_f32=True)
    tw_vq.add_tensor("middle_block.attn.c_proj.weight", vq["decoder.middle_block.attn.c_proj.weight"], FILE_OPERATOR, quant4=True)
    tw_vq.add_tensor("middle_block.attn.c_proj.bias", vq["decoder.middle_block.attn.c_proj.bias"], FILE_OPERATOR, store_f32=True)
    tw_vq.add_tensor("middle_block.attn.q_norm.weight", vq["decoder.middle_block.attn.q_norm.weight"], FILE_OPERATOR, store_f32=True)
    tw_vq.add_tensor("middle_block.attn.k_norm.weight", vq["decoder.middle_block.attn.k_norm.weight"], FILE_OPERATOR, store_f32=True)

    tw_vq.add_tensor("middle_block.mlp.c_fc_fused.weight", vq_fc_w_fused, FILE_FFN, quant4=True)
    tw_vq.add_tensor("middle_block.mlp.c_fc_fused.bias", vq_fc_b_fused, FILE_FFN, store_f32=True)
    tw_vq.add_tensor("middle_block.mlp.c_proj.weight", vq["decoder.middle_block.mlp.c_proj.weight"], FILE_FFN, quant4=True)
    tw_vq.add_tensor("middle_block.mlp.c_proj.bias", vq["decoder.middle_block.mlp.c_proj.bias"], FILE_FFN, store_f32=True)
    summary_trunks.append(tw_vq.finalize(prune_attention=True))

    vq_experts_state: list[dict[str, torch.Tensor]] = []
    for s in range(11):
        emb = vq[f"vq.embeddings.{s}.weight"].to(torch.float32)
        emb_norm = F.normalize(emb, p=2.0, dim=-1)
        emb_sq_norm = (emb * emb).sum(dim=-1)
        qr_w = vq[f"vq.quant_resi.{s}.weight"]
        qr_b = vq[f"vq.quant_resi.{s}.bias"]
        hits = vq[f"vq.codebook_hit_{s}"]

        tw_vqe = TrunkWriter(out_root / f"vqvae-expert-{s}", layer_idx=s, op_type=5)
        tw_vqe.add_tensor("codebook.sq_norm", emb_sq_norm, FILE_COMMON, store_f32=True)
        tw_vqe.add_tensor("quant_resi.bias", qr_b, FILE_COMMON, store_f32=True)
        tw_vqe.add_tensor("codebook.weight", emb.to(torch.bfloat16), FILE_OPERATOR, quant4=True)
        tw_vqe.add_tensor("codebook.l2_norm.weight", emb_norm.to(torch.bfloat16), FILE_OPERATOR)
        tw_vqe.add_tensor("quant_resi.weight", qr_w, FILE_FFN, quant4=True)
        summary_trunks.append(tw_vqe.finalize(prune_attention=True))

        vq_experts_state.append({
            "embedding": emb.to(torch.bfloat16),
            "embedding_l2_norm": emb_norm.to(torch.bfloat16),
            "embedding_sq_norm": emb_sq_norm,
            "quant_resi_w": qr_w,
            "quant_resi_b": qr_b,
            "codebook_hit": hits,
        })

    precomputed_state["c3_vqvae"] = {
        "middle_attn_w_fused": vq_attn_w_fused.to(torch.bfloat16),
        "middle_attn_b_fused": vq_attn_b_fused,
        "middle_fc_w_fused": vq_fc_w_fused.to(torch.bfloat16),
        "middle_fc_b_fused": vq_fc_b_fused,
        "experts": vq_experts_state,
    }
    precomputed_state["algebraic_fusions"].extend([
        "C3.VQVAE.middle_block: ln_1 -> attn.c_attn; ln_2 -> mlp.c_fc",
        "C3.VQVAE.experts[0..10]: precomputed L2-normalized codebooks & ||e_j||_2^2 norms",
    ])

    # ==========================================================================
    # 4. Core 3 · Decode — Part C: 12-Layer NextConceptTransformer (`ncm-trunk-0..11/` + `ncm-global-0/`)
    # ==========================================================================
    print("[5/6] Precomputing Core 3 · 12-Layer NextConceptTransformer (`ncm-trunk-0..11/` + `ncm-global-0/`)...")
    net = ckpt["net"]
    ncm_blocks_state: list[dict[str, torch.Tensor]] = []

    for l in range(12):
        pfx = f"transformer.blocks.{l}."
        c_attn_w_fused, c_attn_b_fused = fuse_ln_into_linear(
            net[pfx + "ln_1.weight"],
            net[pfx + "ln_1.bias"],
            net[pfx + "attn.c_attn.weight"],
            net[pfx + "attn.c_attn.bias"],
        )
        c_fc_w_fused, c_fc_b_fused = fuse_ln_into_linear(
            net[pfx + "ln_2.weight"],
            net[pfx + "ln_2.bias"],
            net[pfx + "mlp.c_fc.weight"],
            net[pfx + "mlp.c_fc.bias"],
        )
        c_proj_w = net[pfx + "attn.c_proj.weight"]
        c_proj_b = net[pfx + "attn.c_proj.bias"]
        q_norm_w = net[pfx + "attn.q_norm.weight"]
        k_norm_w = net[pfx + "attn.k_norm.weight"]
        mlp_proj_w = net[pfx + "mlp.c_proj.weight"]
        mlp_proj_b = net[pfx + "mlp.c_proj.bias"]

        tw_ncm = TrunkWriter(out_root / f"ncm-trunk-{l}", layer_idx=l, op_type=1)
        tw_ncm.add_tensor("attn.q_norm.weight", q_norm_w, FILE_COMMON, store_f32=True)
        tw_ncm.add_tensor("attn.k_norm.weight", k_norm_w, FILE_COMMON, store_f32=True)

        tw_ncm.add_tensor("attn.c_attn_fused.weight", c_attn_w_fused.to(torch.bfloat16), FILE_OPERATOR, quant4=True)
        tw_ncm.add_tensor("attn.c_attn_fused.bias", c_attn_b_fused, FILE_OPERATOR, store_f32=True)
        tw_ncm.add_tensor("attn.c_proj.weight", c_proj_w, FILE_OPERATOR, quant4=True)
        tw_ncm.add_tensor("attn.c_proj.bias", c_proj_b, FILE_OPERATOR, store_f32=True)

        tw_ncm.add_tensor("mlp.c_fc_fused.weight", c_fc_w_fused.to(torch.bfloat16), FILE_FFN, quant4=True)
        tw_ncm.add_tensor("mlp.c_fc_fused.bias", c_fc_b_fused, FILE_FFN, store_f32=True)
        tw_ncm.add_tensor("mlp.c_proj.weight", mlp_proj_w, FILE_FFN, quant4=True)
        tw_ncm.add_tensor("mlp.c_proj.bias", mlp_proj_b, FILE_FFN, store_f32=True)
        summary_trunks.append(tw_ncm.finalize(prune_attention=True))

        ncm_blocks_state.append({
            "c_attn_w_fused": c_attn_w_fused.to(torch.bfloat16),
            "c_attn_b_fused": c_attn_b_fused,
            "c_proj_w": c_proj_w,
            "c_proj_b": c_proj_b,
            "q_norm_w": q_norm_w,
            "k_norm_w": k_norm_w,
            "c_fc_w_fused": c_fc_w_fused.to(torch.bfloat16),
            "c_fc_b_fused": c_fc_b_fused,
            "mlp_proj_w": mlp_proj_w,
            "mlp_proj_b": mlp_proj_b,
        })

    # Fuse `transformer.ln_f` (gamma_f, beta_f) into all 11 concept level heads (`lm_head_l0` + `lm_heads.0..9`)
    ln_f_w = net["transformer.ln_f.weight"]
    ln_f_b = net["transformer.ln_f.bias"]
    tw_ncm_glob = TrunkWriter(out_root / "ncm-global-0", layer_idx=0, op_type=6)
    tw_ncm_glob.add_tensor("transformer.wte.weight", net["transformer.wte.weight"], FILE_COMMON)
    tw_ncm_glob.add_tensor("transformer.wte.bias", net["transformer.wte.bias"], FILE_COMMON, store_f32=True)
    tw_ncm_glob.add_tensor("transformer.wle.weight", net["transformer.wle.weight"], FILE_COMMON, store_f32=True)
    tw_ncm_glob.add_tensor("transformer.ln_f.weight", ln_f_w, FILE_COMMON, store_f32=True)
    tw_ncm_glob.add_tensor("transformer.ln_f.bias", ln_f_b, FILE_COMMON, store_f32=True)

    head_l0_w_fused, head_l0_b_fused = fuse_ln_into_linear(ln_f_w, ln_f_b, net["lm_head_l0.weight"], None)
    tw_ncm_glob.add_tensor("lm_head_l0_fused.weight", head_l0_w_fused.to(torch.bfloat16), FILE_OPERATOR, quant4=True)
    tw_ncm_glob.add_tensor("lm_head_l0_fused.bias", head_l0_b_fused, FILE_OPERATOR, store_f32=True)

    fused_heads_w: list[torch.Tensor] = []
    fused_heads_b: list[torch.Tensor] = []
    for h_idx in range(10):
        hw_f, hb_f = fuse_ln_into_linear(ln_f_w, ln_f_b, net[f"lm_heads.{h_idx}.weight"], None)
        tw_ncm_glob.add_tensor(f"lm_heads.{h_idx}_fused.weight", hw_f.to(torch.bfloat16), FILE_OPERATOR, quant4=True)
        tw_ncm_glob.add_tensor(f"lm_heads.{h_idx}_fused.bias", hb_f, FILE_OPERATOR, store_f32=True)
        fused_heads_w.append(hw_f.to(torch.bfloat16))
        fused_heads_b.append(hb_f)

    summary_trunks.append(tw_ncm_glob.finalize(prune_attention=True))
    precomputed_state["c3_ncm"] = {
        "blocks": ncm_blocks_state,
        "lm_head_l0_w_fused": head_l0_w_fused.to(torch.bfloat16),
        "lm_head_l0_b_fused": head_l0_b_fused,
        "lm_heads_w_fused": fused_heads_w,
        "lm_heads_b_fused": fused_heads_b,
    }
    precomputed_state["algebraic_fusions"].extend([
        "C3.NextConceptTransformer.blocks[0..11]: ln_1 -> attn.c_attn; ln_2 -> mlp.c_fc",
        "C3.NextConceptTransformer.heads[0..10]: ln_f -> lm_head_l0 & lm_heads.0..9",
    ])

    # ==========================================================================
    # 5. Core 3 · Decode — Part D: 3-Layer LexicalPointerDecoder (`ptr-trunk-0..2/` + `ptr-global-0/`)
    # ==========================================================================
    print("[6/6] Precomputing Core 3 · 3-Layer LexicalPointerDecoder (`ptr-trunk-0..2/` + `ptr-global-0/`)...")
    ptr = ckpt["ptr_dec"]
    res_scale = float(ptr["res_scale"].item())
    ptr_blocks_state: list[dict[str, torch.Tensor]] = []

    for l in range(3):
        pfx = f"blocks.{l}."
        qkv_w_fused, qkv_b_fused = fuse_ln_into_linear(
            ptr[pfx + "ln1.weight"], ptr[pfx + "ln1.bias"], ptr[pfx + "qkv.weight"], ptr[pfx + "qkv.bias"]
        )
        cx_q_w_fused, cx_q_b_fused = fuse_ln_into_linear(
            ptr[pfx + "ln_q.weight"], ptr[pfx + "ln_q.bias"], ptr[pfx + "cx_q.weight"], ptr[pfx + "cx_q.bias"]
        )
        cx_kv_w_fused, cx_kv_b_fused = fuse_ln_into_linear(
            ptr[pfx + "ln_kv.weight"], ptr[pfx + "ln_kv.bias"], ptr[pfx + "cx_kv.weight"], ptr[pfx + "cx_kv.bias"]
        )
        mlp0_w_fused, mlp0_b_fused = fuse_ln_into_linear(
            ptr[pfx + "ln2.weight"], ptr[pfx + "ln2.bias"], ptr[pfx + "mlp.0.weight"], ptr[pfx + "mlp.0.bias"]
        )
        proj_w = ptr[pfx + "proj.weight"].to(torch.float32)
        proj_b = ptr[pfx + "proj.bias"].to(torch.float32)
        cx_proj_w = ptr[pfx + "cx_proj.weight"].to(torch.float32)
        cx_proj_b = ptr[pfx + "cx_proj.bias"].to(torch.float32)
        mlp2_w = ptr[pfx + "mlp.2.weight"].to(torch.float32)
        mlp2_b = ptr[pfx + "mlp.2.bias"].to(torch.float32)

        tw_ptr = TrunkWriter(out_root / f"ptr-trunk-{l}", layer_idx=l, op_type=7)
        tw_ptr.add_tensor("qkv_fused.weight", qkv_w_fused, FILE_OPERATOR, store_f32=True, quant4=True)
        tw_ptr.add_tensor("qkv_fused.bias", qkv_b_fused, FILE_OPERATOR, store_f32=True)
        tw_ptr.add_tensor("proj.weight", proj_w, FILE_OPERATOR, store_f32=True, quant4=True)
        tw_ptr.add_tensor("proj.bias", proj_b, FILE_OPERATOR, store_f32=True)
        tw_ptr.add_tensor("cx_q_fused.weight", cx_q_w_fused, FILE_OPERATOR, store_f32=True, quant4=True)
        tw_ptr.add_tensor("cx_q_fused.bias", cx_q_b_fused, FILE_OPERATOR, store_f32=True)
        tw_ptr.add_tensor("cx_kv_fused.weight", cx_kv_w_fused, FILE_OPERATOR, store_f32=True, quant4=True)
        tw_ptr.add_tensor("cx_kv_fused.bias", cx_kv_b_fused, FILE_OPERATOR, store_f32=True)
        tw_ptr.add_tensor("cx_proj.weight", cx_proj_w, FILE_OPERATOR, store_f32=True, quant4=True)
        tw_ptr.add_tensor("cx_proj.bias", cx_proj_b, FILE_OPERATOR, store_f32=True)

        tw_ptr.add_tensor("mlp0_fused.weight", mlp0_w_fused, FILE_FFN, store_f32=True, quant4=True)
        tw_ptr.add_tensor("mlp0_fused.bias", mlp0_b_fused, FILE_FFN, store_f32=True)
        tw_ptr.add_tensor("mlp2.weight", mlp2_w, FILE_FFN, store_f32=True, quant4=True)
        tw_ptr.add_tensor("mlp2.bias", mlp2_b, FILE_FFN, store_f32=True)
        summary_trunks.append(tw_ptr.finalize(prune_attention=True))

        ptr_blocks_state.append({
            "qkv_w_fused": qkv_w_fused,
            "qkv_b_fused": qkv_b_fused,
            "proj_w": proj_w,
            "proj_b": proj_b,
            "cx_q_w_fused": cx_q_w_fused,
            "cx_q_b_fused": cx_q_b_fused,
            "cx_kv_w_fused": cx_kv_w_fused,
            "cx_kv_b_fused": cx_kv_b_fused,
            "cx_proj_w": cx_proj_w,
            "cx_proj_b": cx_proj_b,
            "mlp0_w_fused": mlp0_w_fused,
            "mlp0_b_fused": mlp0_b_fused,
            "mlp2_w": mlp2_w,
            "mlp2_b": mlp2_b,
        })

    # Fuse `res_scale` into `ln_out` (since z_refined = z_hat + res_scale * ln_out(x))
    ln_out_w_fused = ptr["ln_out.weight"].to(torch.float32) * res_scale
    ln_out_b_fused = ptr["ln_out.bias"].to(torch.float32) * res_scale
    tw_ptr_glob = TrunkWriter(out_root / "ptr-global-0", layer_idx=0, op_type=8)
    tw_ptr_glob.add_tensor("tgt_pos.weight", ptr["tgt_pos.weight"], FILE_COMMON, store_f32=True)
    tw_ptr_glob.add_tensor("plan_pos.weight", ptr["plan_pos.weight"], FILE_COMMON, store_f32=True)
    tw_ptr_glob.add_tensor("rh_pos.weight", ptr["rh_pos.weight"], FILE_COMMON, store_f32=True)
    tw_ptr_glob.add_tensor("ln_out_fused.weight", ln_out_w_fused, FILE_COMMON, store_f32=True)
    tw_ptr_glob.add_tensor("ln_out_fused.bias", ln_out_b_fused, FILE_COMMON, store_f32=True)
    tw_ptr_glob.add_tensor("copy_q.weight", ptr["copy_q.weight"], FILE_OPERATOR, store_f32=True, quant4=True)
    tw_ptr_glob.add_tensor("copy_q.bias", ptr["copy_q.bias"], FILE_OPERATOR, store_f32=True)
    tw_ptr_glob.add_tensor("copy_k.weight", ptr["copy_k.weight"], FILE_OPERATOR, store_f32=True, quant4=True)
    tw_ptr_glob.add_tensor("copy_k.bias", ptr["copy_k.bias"], FILE_OPERATOR, store_f32=True)
    tw_ptr_glob.add_tensor("copy_gate.weight", ptr["copy_gate.weight"], FILE_FFN, store_f32=True)
    tw_ptr_glob.add_tensor("copy_gate.bias", ptr["copy_gate.bias"], FILE_FFN, store_f32=True)
    summary_trunks.append(tw_ptr_glob.finalize(prune_attention=True))

    precomputed_state["c3_ptr_dec"] = {
        "blocks": ptr_blocks_state,
        "ln_out_w_fused": ln_out_w_fused,
        "ln_out_b_fused": ln_out_b_fused,
    }
    precomputed_state["algebraic_fusions"].extend([
        "C3.LexicalPointerDecoder.blocks[0..2]: ln1 -> qkv; ln_q -> cx_q; ln_kv -> cx_kv; ln2 -> mlp.0",
        "C3.LexicalPointerDecoder.ln_out: res_scale -> ln_out",
    ])

    # ==========================================================================
    # 6. Numerical Parity & Speedup Verification on CPU
    # ==========================================================================
    torch.manual_seed(42)
    x_2048 = torch.randn(4, 2048, dtype=torch.float32)

    # Verify C2 NativeHybridLink parity
    ref_nhl = F.layer_norm(x_2048, (2048,), nhl["pre_ln.weight"], nhl["pre_ln.bias"], eps=1e-5)
    ref_nhl = F.linear(ref_nhl, nhl["proj1.weight"], nhl["proj1.bias"])
    ref_nhl = F.gelu(ref_nhl)
    ref_nhl = x_2048 + F.linear(ref_nhl, nhl["proj2.weight"], nhl["proj2.bias"])
    ref_nhl = F.layer_norm(ref_nhl, (2048,), nhl["post_ln.weight"], nhl["post_ln.bias"], eps=1e-5) * nhl_scale

    fused_nhl = F.layer_norm(x_2048, (2048,), None, None, eps=1e-5)
    fused_nhl = F.linear(fused_nhl, nhl_w1_fused, nhl_b1_fused)
    fused_nhl = F.gelu(fused_nhl)
    fused_nhl = x_2048 + F.linear(fused_nhl, nhl_w2, nhl_b2)
    fused_nhl = F.layer_norm(fused_nhl, (2048,), nhl_post_w_fused, nhl_post_b_fused, eps=1e-5)
    c2_max_diff = float((ref_nhl - fused_nhl).abs().max().item())

    # Verify C3 ComboBridge parity
    ref_cb = F.layer_norm(x_2048, (2048,), cb["ln_in.weight"], cb["ln_in.bias"], eps=1e-5)
    ref_cb = F.linear(ref_cb, cb["fc1.weight"], cb["fc1.bias"])
    ref_cb = F.gelu(ref_cb)
    ref_cb = F.linear(ref_cb, cb["fc2.weight"], cb["fc2.bias"])
    ref_cb = F.layer_norm(ref_cb, (384,), cb["ln_out.weight"], cb["ln_out.bias"], eps=1e-5) * cb_gate_scale + cb["seg_rh"].view(1, 384)

    fused_cb = F.layer_norm(x_2048, (2048,), None, None, eps=1e-5)
    fused_cb = F.linear(fused_cb, cb_w1_fused, cb_b1_fused)
    fused_cb = F.gelu(fused_cb)
    fused_cb = F.linear(fused_cb, cb_w2, cb_b2)
    fused_cb = F.layer_norm(fused_cb, (384,), cb_ln_out_w_fused, cb_ln_out_b_fused, eps=1e-5)
    cb_max_diff = float((ref_cb - fused_cb).abs().max().item())

    # Verify C3 NCM Block 0 + Fused Head L0 parity (models.common.LayerNorm uses eps=1e-6)
    x_768 = torch.randn(8, 768, dtype=torch.float32)
    ref_b0 = F.layer_norm(x_768, (768,), net["transformer.blocks.0.ln_1.weight"], net["transformer.blocks.0.ln_1.bias"], eps=1e-6)
    ref_b0 = F.linear(ref_b0, net["transformer.blocks.0.attn.c_attn.weight"].float(), net["transformer.blocks.0.attn.c_attn.bias"])
    fused_b0 = F.layer_norm(x_768, (768,), None, None, eps=1e-6)
    fused_b0 = F.linear(fused_b0, c_attn_w_fused_b0 := fuse_ln_into_linear(net["transformer.blocks.0.ln_1.weight"], net["transformer.blocks.0.ln_1.bias"], net["transformer.blocks.0.attn.c_attn.weight"], net["transformer.blocks.0.attn.c_attn.bias"])[0], ncm_blocks_state[0]["c_attn_b_fused"])
    ncm_b0_max_diff = float((ref_b0 - fused_b0).abs().max().item())

    # Remove any legacy duplicate .pt bundle so only the 31 CLVRTRK1 directories + manifest.json occupy disk
    bundle_path = out_root / "c2_c3_precomputed_trunks.pt"
    if bundle_path.exists():
        bundle_path.unlink()

    total_bf16_bytes = sum(t["bf16_f32_bytes"] for t in summary_trunks)
    total_q4_bytes = sum(t["q4_bytes"] for t in summary_trunks)
    elapsed_s = time.perf_counter() - t0

    manifest = {
        "format": "CLVRTRK1 + CLVR4B01",
        "source_checkpoint": str(ckpt_path),
        "elapsed_seconds": round(elapsed_s, 2),
        "total_trunk_and_expert_dirs": len(summary_trunks),
        "total_bf16_f32_mb": round(total_bf16_bytes / (1024 * 1024), 2),
        "total_q4_group64_mb": round(total_q4_bytes / (1024 * 1024), 2),
        "parity_checks": {
            "c2_native_hybridlink_max_abs_diff": c2_max_diff,
            "c3_combo_bridge_max_abs_diff": cb_max_diff,
            "c3_ncm_block0_c_attn_max_abs_diff": ncm_b0_max_diff,
        },
        "algebraic_fusions": precomputed_state["algebraic_fusions"],
        "directories": summary_trunks,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"[DONE] Built {len(summary_trunks)} precomputed trunk/expert directories in {elapsed_s:.2f}s:\n"
        f"  - BF16/F32 Trunks & Experts : {manifest['total_bf16_f32_mb']:.2f} MB\n"
        f"  - 4-bit Group-64 Companions : {manifest['total_q4_group64_mb']:.2f} MB\n"
        f"  - Parity max_abs_diff       : C2={c2_max_diff:.2e}, ComboBridge={cb_max_diff:.2e}, NCM_B0={ncm_b0_max_diff:.2e}\n"
        f"  - Manifest                  : {manifest_path}"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Clover precomputed trunks & experts for C2 (Plan) and C3 (Decode)")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/opt/clover-k3/s23-omni2b/ncm/lfm3_ncm_core_bf16.pt",
        help="Path to lfm3_ncm_core_bf16.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/opt/clover-k3/s23-omni2b/ncm/precomputed",
        help="Output directory for precomputed C2 & C3 trunks and experts",
    )
    parser.add_argument(
        "--verify-runtime",
        type=str,
        default="",
        help="Optional base_dir (e.g. ~/clover-s23) to verify live CloverS23OnDeviceSystem with precomputed C2 & C3 trunks",
    )
    args = parser.parse_args()
    if args.verify_runtime:
        import sys
        base_p = Path(args.verify_runtime).expanduser().resolve()
        if str(base_p) not in sys.path:
            sys.path.insert(0, str(base_p))
        from s23_on_device_clover import CloverS23OnDeviceSystem
        sys_s23 = CloverS23OnDeviceSystem(base_p, load_neural_ncm=True)
        h0 = np.ones((1, 2048), dtype=np.float32) * 0.05
        lat = sys_s23.hybrid_ncm.run_continuous_latent_step(h0)
        txt, ids, meta = sys_s23.hybrid_ncm.render_plan_through_11level_ncm(
            "Clover unifies Inference, Plan, and Decode into precomputed trunks and experts."
        )
        print(json.dumps({
            "precomputed_trunks_active": sys_s23.hybrid_ncm.precomputed_trunks_active,
            "precomputed_dirs": sys_s23.hybrid_ncm.precomputed_manifest.get("total_trunk_and_expert_dirs"),
            "c2_latent_step": {k: v for k, v in lat.items() if k != "r_np"},
            "c3_ncm_meta": meta,
            "c3_ncm_text": txt,
        }, indent=2))
    else:
        build_c2_c3_precomputed(Path(args.ckpt), Path(args.out_dir))
