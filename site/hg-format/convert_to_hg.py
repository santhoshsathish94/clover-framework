#!/usr/bin/env python3
"""Universal Clover `HG4-16R-OP` & `K31H` (`HG4` + `HG2`) Model Converter & Verifier.

Converts any Hugging Face `safetensors` checkpoint (local directory or remote Hub repo ID;
dense Transformer, GQA, SSM/Mamba/BD-GDN, or Mixture-of-Experts; `BF16`, `FP16`, `FP32`,
`FP8_E4M3`, or `MXFP4`) into Clover's canonical Hadamard-Grouped format:

  * `HG4-16R` (`4.125 bpw`): Column-balanced `FWHT-128` orthogonal Walsh-Hadamard rotation +
    15-point MSE-optimal Lloyd-Max Gaussian `LUT16` + 5-point guarded `FP16` scale search +
    16-row lane-interleaved SIMD layout (`[R/16, K/128, 16, 16, 4]`, zero horizontal sums).
  * `K31H` MoE Cold Tier (`--moe-cold-bits 2`, `2.125 bpw`): Keeps trunk, embeddings, shared
    experts, and the top `--hot-experts` routed experts per layer in `HG4-16R`, while packing
    long-tail cold routed experts in `HG2-16R` (4-level Gaussian Lloyd-Max + `FWHT-128`).
  * `HG4OP001` Precomputed Operator Sidecar (`precomputed_ops/`): Losslessly rotates MoE router
    gates (`W_gate * D^-1 * H_128`) in `FP32` (`Top-K Jaccard = 1.000000`) and stores 1D norms,
    biases, and SSM parameters in exact `FP32`.
  * Built-in `--verify` & `--self-test`: Verifies `y_orig = X @ W^T` vs Hadamard-domain `HG`
    execution on heavy-tailed outlier activations and compares against naive symmetric `INT4`.

Quickstart:
  # 1. Instant 3-second local verification (zero downloads):
  python3 convert_to_hg.py --self-test

  # 2. Convert any local or Hugging Face model directory / repo ID + verify accuracy:
  python3 convert_to_hg.py --model Qwen/Qwen3-8B --out ./qwen3-8b-hg4 --verify

  # 3. Convert a large MoE model in K31H hybrid mode (top 42 hot experts HG4 + cold tail HG2):
  python3 convert_to_hg.py --model ./my-moe-checkpoint --out ./my-moe-k31h \\
      --moe-cold-bits 2 --hot-experts 42 --ldlq --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

FORMAT_VERSION = "HG4-16R-OP-v1"
MAGIC_HG4R = b"HG4R"
MAGIC_HG2R = b"HG2R"
MAGIC_HG8R = b"HG8R"
MAGIC_HGOP = b"HG4OP001"

# Canonical 16-entry signed int8 Lloyd-Max Gaussian codebook (matches hg4_fast_packer.c & C11 engines)
LUT16_S8 = np.array(
    [-127, -96, -75, -58, -44, -31, -18, -6, 6, 18, 31, 44, 58, 75, 96, 127],
    dtype=np.int8,
)
LUT16_F32 = LUT16_S8.astype(np.float32)
CODE16_MIDS = 0.5 * (LUT16_F32[:-1] + LUT16_F32[1:])

# Canonical 4-entry signed int8 Lloyd-Max Gaussian codebook for 2-bit cold MoE experts (HG2-16R)
LUT4_S8 = np.array([-127, -38, 38, 127], dtype=np.int8)
LUT4_F32 = LUT4_S8.astype(np.float32)
CODE4_MIDS = np.array([-82.5, 0.0, 82.5], dtype=np.float32)

# OCP MXFP4 (E2M1) 16-nibble lookup table (used by Kimi-K3.1-Pro, gpt-oss, etc.)
MXFP4_E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

SCALE_MULTS = (0.86, 0.93, 1.00, 1.07, 1.14)
FP16_MIN_GUARD = 6.11e-5
FP16_MAX_GUARD = 65504.0

_SKIP_QUANT_PATTERNS = re.compile(
    r"(?:^|\.)(norm|layernorm|rmsnorm|ln_|bias|rotary|inv_freq|A_log|dt_bias|"
    r"conv1d|shortconv|router|gate\.weight|e_score_correction_bias)(?:\.|$)",
    re.IGNORECASE,
)
_MOE_EXPERT_INDEX_PATTERN = re.compile(
    r"(?:experts\.(\d+)\.|expert_(\d+)(?:\.|$)|experts\.[^.]+\.(\d+)\.)", re.IGNORECASE
)
_ROUTER_GATE_PATTERN = re.compile(
    r"(?:mlp\.gate\.weight|block_sparse_moe\.gate\.weight|router\.weight)$",
    re.IGNORECASE,
)


def _ceil_mult(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def deterministic_rademacher(name: str, length: int) -> np.ndarray:
    """Deterministic {-1.0, +1.0} Rademacher sign vector seeded from tensor name."""
    seed_bytes = hashlib.sha256(f"clover-hg4-16r:{name}".encode("utf-8")).digest()
    seed = int.from_bytes(seed_bytes[:8], "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    bits = rng.integers(0, 2, size=length, dtype=np.int8)
    return np.where(bits == 0, -1.0, 1.0).astype(np.float32)


def fwht128_np(x: np.ndarray) -> np.ndarray:
    """Exact orthogonal 128-point Fast Walsh-Hadamard Transform along the last axis."""
    orig_shape = x.shape
    assert orig_shape[-1] % 128 == 0, f"Last dim {orig_shape[-1]} must be divisible by 128"
    y = x.astype(np.float32, copy=True).reshape(-1, 128)
    h = 1
    while h < 128:
        y = y.reshape(-1, 128 // (2 * h), 2, h)
        a = y[:, :, 0, :].copy()
        b = y[:, :, 1, :].copy()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        h <<= 1
    y *= (1.0 / math.sqrt(128.0))
    return y.reshape(orig_shape)


def compute_pre_diag(w_f32: np.ndarray, name: str) -> np.ndarray:
    """Column-balanced incoherence diagonal D = sqrt(col_rms) * rademacher_signs."""
    col_rms = np.sqrt(np.mean(np.square(w_f32), axis=0) + 1e-12)
    med = float(np.median(col_rms))
    if med > 0.0:
        col_rms = np.clip(col_rms, med * 1e-3, med * 1e3)
    else:
        col_rms = np.ones_like(col_rms, dtype=np.float32)
    gain = np.sqrt(col_rms / max(float(np.mean(col_rms)), 1e-12)).astype(np.float32)
    signs = deterministic_rademacher(name, w_f32.shape[1])
    return (gain * signs).astype(np.float32)


def _guarded_fp16(scales_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clamp non-zero subnormals before FP16 rounding so AVX2/NEON/AVX-512 stay bit-exact."""
    abs_s = np.abs(scales_f32)
    guarded = np.where((abs_s > 0.0) & (abs_s < FP16_MIN_GUARD), np.sign(scales_f32) * FP16_MIN_GUARD, scales_f32)
    guarded = np.clip(guarded, -FP16_MAX_GUARD, FP16_MAX_GUARD)
    sc_f16 = guarded.astype(np.float16)
    return sc_f16.view(np.uint16), sc_f16.astype(np.float32)


def _ldlq_block_feedback(block_rot: np.ndarray, sc_f32: np.ndarray, mids: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Sequential Cholesky LDLQ error-feedback rounding within each [R, 128] Hadamard block."""
    # Under column-balanced FWHT-128, rotated columns have near-isotropic diagonal + mild local correlation;
    # compute empirical sample covariance across rows of this matrix to propagate rounding error forward.
    n_rows, blk = block_rot.shape
    cov = (block_rot.T @ block_rot) / max(float(n_rows), 1.0)
    diag_mean = max(float(np.trace(cov)) / blk, 1e-12)
    cov_shrunk = 0.85 * cov + 0.15 * diag_mean * np.eye(blk, dtype=np.float32)
    try:
        # Upper Cholesky of inverse Hessian proxy for causal column-by-column error feedback
        h_inv = np.linalg.inv(cov_shrunk)
        u_chol = np.linalg.cholesky(h_inv[::-1, ::-1])[::-1, ::-1].T
        d_diag = np.maximum(np.diag(u_chol), 1e-8)
        feedback = u_chol / d_diag[:, None]
    except np.linalg.LinAlgError:
        feedback = np.eye(blk, dtype=np.float32)

    w_work = block_rot.copy()
    inv_sc = np.where(sc_f32 > 0.0, 1.0 / sc_f32, 0.0).astype(np.float32)
    idx_out = np.empty((n_rows, blk), dtype=np.uint8)
    for col in range(blk):
        q_idx = np.searchsorted(mids, w_work[:, col] * inv_sc).astype(np.uint8)
        idx_out[:, col] = q_idx
        q_val = lut[q_idx] * sc_f32
        err = w_work[:, col] - q_val
        if col + 1 < blk:
            w_work[:, col + 1 :] += err[:, None] * feedback[col, col + 1 :][None, :]
    return idx_out


def _try_quantize_hg_cuda(
    w_pad: np.ndarray, pre_diag: np.ndarray, bits: int, use_ldlq: bool
) -> tuple[np.ndarray, np.ndarray] | None:
    """Optional CUDA fast-path (~30x faster on RTX 3090/4090) producing identical `(best_sc_u16, best_idx)`."""
    if os.environ.get("CLOVER_HG_DISABLE_CUDA_PACK") == "1":
        return None
    try:
        import torch  # type: ignore[import-not-found]
        if not torch.cuda.is_available():
            return None
    except Exception:
        return None

    dev = torch.device("cuda:0")
    r_pad, k_pad = w_pad.shape
    nb = k_pad // 128
    with torch.no_grad():
        wt = torch.from_numpy(w_pad).to(dev, dtype=torch.float32)
        pd = torch.from_numpy(pre_diag).to(dev, dtype=torch.float32)
        y = (wt * (1.0 / pd).unsqueeze(0)).reshape(-1, 128)
        h = 1
        while h < 128:
            y = y.reshape(-1, 128 // (2 * h), 2, h)
            a = y[:, :, 0, :].clone()
            b = y[:, :, 1, :].clone()
            y[:, :, 0, :] = a + b
            y[:, :, 1, :] = a - b
            h <<= 1
        w_rot = (y * (1.0 / math.sqrt(128.0))).reshape(r_pad, nb, 128)

        lut_t = torch.from_numpy(LUT16_F32 if bits == 4 else LUT4_F32).to(dev)
        mids_t = torch.from_numpy(CODE16_MIDS if bits == 4 else CODE4_MIDS).to(dev)
        rms_factor = (2.7326 / 127.0) if bits == 4 else (1.5104 / 127.0)
        base_sc = torch.sqrt(torch.mean(w_rot * w_rot, dim=-1) + 1e-20) * rms_factor

        best_mse = torch.full((r_pad, nb), float("inf"), device=dev, dtype=torch.float32)
        best_sc_fp16 = torch.zeros((r_pad, nb), device=dev, dtype=torch.float16)
        best_idx = torch.zeros((r_pad, nb, 128), device=dev, dtype=torch.uint8)

        for mult in SCALE_MULTS:
            cand_clamped = (base_sc * mult).clamp(FP16_MIN_GUARD, FP16_MAX_GUARD).to(torch.float16)
            cand_f32 = cand_clamped.to(torch.float32)
            inv_sc = torch.where(cand_f32 > 0.0, 1.0 / cand_f32, torch.zeros_like(cand_f32)).unsqueeze(-1)
            q_idx = torch.bucketize(w_rot * inv_sc, mids_t)
            q_deq = lut_t[q_idx] * cand_f32.unsqueeze(-1)
            mse = torch.mean((w_rot - q_deq) ** 2, dim=-1)
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_sc_fp16 = torch.where(better, cand_clamped, best_sc_fp16)
            best_idx = torch.where(better.unsqueeze(-1), q_idx.to(torch.uint8), best_idx)

        if use_ldlq:
            sc_f32 = best_sc_fp16.to(torch.float32)
            inv_sc2 = torch.where(sc_f32 > 0.0, 1.0 / sc_f32, torch.zeros_like(sc_f32))
            # Batched [nb, 128, 128] covariance + shrinkage Cholesky feedback across all blocks on GPU!
            wr_bnr = w_rot.permute(1, 0, 2).contiguous()  # [nb, r_pad, 128]
            cov = torch.bmm(wr_bnr.transpose(1, 2), wr_bnr) / max(float(r_pad), 1.0)
            diag_mean = cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-8)
            eye = torch.eye(128, device=dev, dtype=torch.float32).unsqueeze(0)
            h_reg = 0.85 * cov + 0.15 * diag_mean * eye
            try:
                h_inv = torch.linalg.inv(h_reg)
                u_chol = torch.linalg.cholesky(torch.flip(h_inv, dims=[-2, -1])).flip(dims=[-2, -1]).transpose(-2, -1)
                d_diag = u_chol.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8)
                fb = (u_chol / d_diag.unsqueeze(-1)).contiguous()  # [nb, 128, 128]
            except Exception:
                fb = eye.expand(nb, 128, 128).contiguous()

            work = wr_bnr.clone()
            inv_bn = inv_sc2.transpose(0, 1).contiguous()  # [nb, r_pad]
            sc_bn = sc_f32.transpose(0, 1).contiguous()
            out_bnk = torch.empty((nb, r_pad, 128), device=dev, dtype=torch.uint8)
            for col in range(128):
                wk = work[:, :, col]
                qk_idx = torch.bucketize(wk * inv_bn, mids_t)
                out_bnk[:, :, col] = qk_idx.to(torch.uint8)
                q_val = lut_t[qk_idx] * sc_bn
                if col + 1 < 128:
                    err = wk - q_val
                    work[:, :, col + 1 :] += err.unsqueeze(-1) * fb[:, col, col + 1 :].unsqueeze(1)
            best_idx = out_bnk.permute(1, 0, 2).contiguous()

        sc_u16_np = best_sc_fp16.view(torch.uint16).cpu().numpy()
        idx_np = best_idx.cpu().numpy()
    return sc_u16_np, idx_np


def pack_matrix_hg(
    w_orig: np.ndarray,
    name: str,
    bits: int = 4,
    use_ldlq: bool = False,
    shared_pre_diag: np.ndarray | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Pack a 2D weight matrix [R, K] into canonical Clover HG4-16R (4b) or HG2-16R (2b) binary blob."""
    if w_orig.ndim != 2:
        raise ValueError(f"pack_matrix_hg expects 2D tensor, got {w_orig.shape} for {name}")
    if bits not in (2, 4):
        raise ValueError(f"Unsupported bits={bits}; expected 2 or 4")

    r_orig, k_orig = int(w_orig.shape[0]), int(w_orig.shape[1])
    r_pad = _ceil_mult(r_orig, 16)
    k_pad = _ceil_mult(k_orig, 128)
    nb = k_pad // 128

    w_pad = np.zeros((r_pad, k_pad), dtype=np.float32)
    w_pad[:r_orig, :k_orig] = w_orig.astype(np.float32, copy=False)

    if shared_pre_diag is not None:
        assert shared_pre_diag.shape == (k_pad,)
        pre_diag = shared_pre_diag.astype(np.float32, copy=False)
    else:
        pre_diag = compute_pre_diag(w_pad, name)

    cuda_res = _try_quantize_hg_cuda(w_pad, pre_diag, bits, use_ldlq)
    if cuda_res is not None:
        best_sc_u16, best_idx = cuda_res
    else:
        # Exact orthogonal pre-conditioning: W_rot = FWHT_128(W * D^-1)
        w_scaled = w_pad * (1.0 / pre_diag)[None, :]
        w_rot = fwht128_np(w_scaled).reshape(r_pad, nb, 128)

        lut_f32 = LUT16_F32 if bits == 4 else LUT4_F32
        mids = CODE16_MIDS if bits == 4 else CODE4_MIDS

        # Base Gaussian scale from block RMS (2.7326 / 127 for 4-bit; 1.5104 / 127 for 2-bit)
        rms_factor = (2.7326 / 127.0) if bits == 4 else (1.5104 / 127.0)
        blk_rms = np.sqrt(np.mean(np.square(w_rot), axis=-1) + 1e-20)
        base_sc = blk_rms * rms_factor

        best_mse = np.full((r_pad, nb), np.inf, dtype=np.float32)
        best_sc_u16 = np.zeros((r_pad, nb), dtype=np.uint16)
        best_sc_f32 = np.zeros((r_pad, nb), dtype=np.float32)
        best_idx = np.zeros((r_pad, nb, 128), dtype=np.uint8)

        for mult in SCALE_MULTS:
            cand_u16, cand_f32 = _guarded_fp16(base_sc * mult)
            inv_sc = np.where(cand_f32 > 0.0, 1.0 / cand_f32, 0.0)[:, :, None]
            q_idx = np.searchsorted(mids, w_rot * inv_sc).astype(np.uint8)
            q_deq = lut_f32[q_idx] * cand_f32[:, :, None]
            mse = np.mean(np.square(w_rot - q_deq), axis=-1)
            better = mse < best_mse
            best_mse = np.where(better, mse, best_mse)
            best_sc_u16 = np.where(better, cand_u16, best_sc_u16)
            best_sc_f32 = np.where(better, cand_f32, best_sc_f32)
            best_idx = np.where(better[:, :, None], q_idx, best_idx)

        if use_ldlq:
            for b in range(nb):
                best_idx[:, b, :] = _ldlq_block_feedback(w_rot[:, b, :], best_sc_f32[:, b], mids, lut_f32)

    # Pack scales into [R/16, nb, 16] uint16 (16-row lane-interleaved)
    sc_16r = np.ascontiguousarray(best_sc_u16.reshape(r_pad // 16, 16, nb).transpose(0, 2, 1))

    if bits == 4:
        magic = MAGIC_HG4R
        # Canonical HG4-16R layout: [R/16, nb, 16_chunks, 16_lanes, 4_bytes]
        idx_6d = best_idx.reshape(r_pad // 16, 16, nb, 16, 2, 4)
        packed = np.ascontiguousarray((idx_6d[:, :, :, :, 0, :] | (idx_6d[:, :, :, :, 1, :] << 4)).transpose(0, 2, 3, 1, 4))
    else:
        magic = MAGIC_HG2R
        # Canonical HG2-16R layout: [R/16, nb, 8_chunks, 16_lanes, 4_bytes] (4 weights/byte)
        idx_6d = best_idx.reshape(r_pad // 16, 16, nb, 8, 4, 4)
        b_pack = (
            idx_6d[:, :, :, :, 0, :]
            | (idx_6d[:, :, :, :, 1, :] << 2)
            | (idx_6d[:, :, :, :, 2, :] << 4)
            | (idx_6d[:, :, :, :, 3, :] << 6)
        )
        packed = np.ascontiguousarray(b_pack.transpose(0, 2, 3, 1, 4))

    header = magic + struct.pack("<IIII", r_orig, k_orig, r_pad, k_pad)
    blob = header + pre_diag.astype(np.float32).tobytes() + sc_16r.tobytes() + packed.tobytes()

    bpw_eff = (len(blob) * 8.0) / max(r_orig * k_orig, 1)
    meta = {
        "name": name,
        "magic": magic.decode("ascii"),
        "bits": bits,
        "ldlq": bool(use_ldlq),
        "orig_shape": [r_orig, k_orig],
        "padded_shape": [r_pad, k_pad],
        "bytes": len(blob),
        "effective_bpw": round(bpw_eff, 4),
    }
    return blob, meta


def unpack_matrix_hg_rotated(blob: bytes) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int]:
    """Unpack a canonical HG4R / HG2R blob into rotated weight matrix W_rot_hat [R_pad, K_pad] and pre_diag [K_pad]."""
    magic = blob[:4]
    if magic not in (MAGIC_HG4R, MAGIC_HG2R):
        raise ValueError(f"Unknown HG magic header: {magic!r}")
    bits = 4 if magic == MAGIC_HG4R else 2
    r_orig, k_orig, r_pad, k_pad = struct.unpack_from("<IIII", blob, 4)
    nb = k_pad // 128
    off = 20
    pre_diag = np.frombuffer(blob, dtype=np.float32, count=k_pad, offset=off).copy()
    off += k_pad * 4
    sc_16r = np.frombuffer(blob, dtype=np.uint16, count=r_pad * nb, offset=off).reshape(r_pad // 16, nb, 16)
    sc_f32 = sc_16r.transpose(0, 2, 1).reshape(r_pad, nb).view(np.float16).astype(np.float32)
    off += r_pad * nb * 2

    if bits == 4:
        pk = np.frombuffer(blob, dtype=np.uint8, count=(r_pad * k_pad) // 2, offset=off).reshape(r_pad // 16, nb, 16, 16, 4)
        pk_t = pk.transpose(0, 3, 1, 2, 4)  # [R/16, 16, nb, 16, 4]
        lo = pk_t & 0x0F
        hi = (pk_t >> 4) & 0x0F
        idx = np.stack([lo, hi], axis=4).reshape(r_pad, nb, 128)
        w_rot_hat = LUT16_F32[idx] * sc_f32[:, :, None]
    else:
        pk = np.frombuffer(blob, dtype=np.uint8, count=(r_pad * k_pad) // 4, offset=off).reshape(r_pad // 16, nb, 8, 16, 4)
        pk_t = pk.transpose(0, 3, 1, 2, 4)  # [R/16, 16, nb, 8, 4]
        c0 = pk_t & 0x03
        c1 = (pk_t >> 2) & 0x03
        c2 = (pk_t >> 4) & 0x03
        c3 = (pk_t >> 6) & 0x03
        idx = np.stack([c0, c1, c2, c3], axis=4).reshape(r_pad, nb, 128)
        w_rot_hat = LUT4_F32[idx] * sc_f32[:, :, None]

    return w_rot_hat.reshape(r_pad, k_pad), pre_diag, r_orig, k_orig, r_pad, k_pad, bits


def forward_hg_matmul(x: np.ndarray, blob: bytes) -> np.ndarray:
    """Execute exact Hadamard-domain forward pass y = (FWHT_128(D * x)) @ W_rot_hat^T (matching C11 kernel math)."""
    w_rot_hat, pre_diag, r_orig, k_orig, _, k_pad, _ = unpack_matrix_hg_rotated(blob)
    t_rows = x.shape[0]
    x_pad = np.zeros((t_rows, k_pad), dtype=np.float32)
    x_pad[:, :k_orig] = x.astype(np.float32, copy=False)
    x_rot = fwht128_np(x_pad * pre_diag[None, :])
    y_pad = x_rot @ w_rot_hat.T
    return y_pad[:, :r_orig]


def naive_int4_group128_matmul(x: np.ndarray, w_orig: np.ndarray) -> np.ndarray:
    """Reference naive symmetric Group-128 INT4 matmul (no Hadamard rotation) to demonstrate outlier clipping."""
    r_orig, k_orig = w_orig.shape
    k_pad = _ceil_mult(k_orig, 128)
    w_pad = np.zeros((r_orig, k_pad), dtype=np.float32)
    w_pad[:, :k_orig] = w_orig.astype(np.float32, copy=False)
    g = w_pad.reshape(r_orig, k_pad // 128, 128)
    amax = np.max(np.abs(g), axis=-1, keepdims=True)
    scale = np.where(amax > 0.0, amax / 7.0, 1.0).astype(np.float16).astype(np.float32)
    q = np.clip(np.round(g / scale), -8.0, 7.0)
    w_deq = (q * scale).reshape(r_orig, k_pad)[:, :k_orig]
    return x.astype(np.float32) @ w_deq.T


def verify_matrix_parity(w_orig: np.ndarray, blob: bytes, seed: int = 42) -> dict[str, float]:
    """Compare original FP32/BF16 matmul against Hadamard-domain HG execution and naive INT4 on outlier activations."""
    rng = np.random.Generator(np.random.PCG64(seed))
    k_orig = w_orig.shape[1]
    # Realistic LLM activation distribution: Gaussian bulk + 2% sharp outlier channels (25x amplitude)
    x = rng.standard_normal((32, k_orig), dtype=np.float32)
    n_outliers = max(1, k_orig // 50)
    outlier_cols = rng.choice(k_orig, size=n_outliers, replace=False)
    x[:, outlier_cols] *= 25.0

    y_ref = x @ w_orig.astype(np.float32).T
    y_hg = forward_hg_matmul(x, blob)
    y_int4 = naive_int4_group128_matmul(x, w_orig)

    def _cos_and_sqnr(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
        num = float(np.sum(a * b))
        den = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-20)
        cos = num / den
        err_pow = max(float(np.sum(np.square(a - b))), 1e-20)
        sig_pow = max(float(np.sum(np.square(a))), 1e-20)
        sqnr_db = 10.0 * math.log10(sig_pow / err_pow)
        top1 = float(np.mean(np.argmax(a, axis=-1) == np.argmax(b, axis=-1)) * 100.0)
        return cos, sqnr_db, top1

    cos_hg, sqnr_hg, top1_hg = _cos_and_sqnr(y_ref, y_hg)
    cos_i4, sqnr_i4, top1_i4 = _cos_and_sqnr(y_ref, y_int4)
    return {
        "hg_cosine": round(cos_hg, 6),
        "hg_sqnr_db": round(sqnr_hg, 2),
        "hg_top1_pct": round(top1_hg, 2),
        "naive_int4_cosine": round(cos_i4, 6),
        "naive_int4_sqnr_db": round(sqnr_i4, 2),
        "naive_int4_top1_pct": round(top1_i4, 2),
    }


def _decode_raw_safetensor_bytes(raw: bytes, dtype_str: str, shape: tuple[int, ...]) -> np.ndarray:
    """Decode safetensors raw bytes (F32, F16, BF16, F8_E4M3) into FP32 NumPy array with zero PyTorch dependency."""
    if dtype_str == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    if dtype_str == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype_str == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32).reshape(shape)
    if dtype_str in ("F8_E4M3", "FLOAT8_E4M3FN"):
        u8 = np.frombuffer(raw, dtype=np.uint8)
        sign = np.where((u8 & 0x80) != 0, -1.0, 1.0).astype(np.float32)
        exp = ((u8 >> 3) & 0x0F).astype(np.int32)
        mant = (u8 & 0x07).astype(np.float32)
        val = np.where(
            exp == 0,
            (mant / 8.0) * (2.0 ** -6),
            np.where((exp == 15) & (mant == 7), 0.0, (1.0 + mant / 8.0) * np.exp2((exp - 7).astype(np.float32))),
        )
        return (sign * val).astype(np.float32).reshape(shape)
    if dtype_str in ("I8", "U8"):
        dt = np.int8 if dtype_str == "I8" else np.uint8
        return np.frombuffer(raw, dtype=dt).astype(np.float32).reshape(shape)
    raise ValueError(f"Unsupported safetensors dtype {dtype_str!r}; install torch/safetensors or convert to BF16/FP16.")


def decode_mxfp4_pair(packed_u8: np.ndarray, scale_e8m0_u8: np.ndarray) -> np.ndarray:
    """Decode OCP MXFP4 `weight_packed` [..., K/2] uint8 + `weight_scale` [..., K/32] E8M0 uint8 into FP32 [..., K]."""
    p = packed_u8.astype(np.uint8, copy=False)
    s = scale_e8m0_u8.astype(np.int32, copy=False)
    lo = MXFP4_E2M1_LUT[p & 0x0F]
    hi = MXFP4_E2M1_LUT[(p >> 4) & 0x0F]
    vals = np.stack([lo, hi], axis=-1).reshape(*p.shape[:-1], p.shape[-1] * 2)
    sc = np.where(s == 255, 0.0, np.exp2((s - 127).astype(np.float32)))
    blk = vals.shape[-1] // sc.shape[-1]
    return (vals.reshape(*sc.shape, blk) * sc[..., None]).reshape(vals.shape).astype(np.float32)


def iter_safetensor_file(sf_path: Path) -> Iterator[tuple[str, np.ndarray, str]]:
    """Stream tensors one by one from a single `.safetensors` file (auto-fusing MXFP4 packed+scale pairs)."""
    with sf_path.open("rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(hlen).decode("utf-8"))
        data_base = 8 + hlen
        meta_keys = {k: v for k, v in header.items() if k != "__metadata__" and isinstance(v, dict)}
        consumed_scales: set[str] = set()
        for k in list(meta_keys):
            if k.endswith(".weight_packed"):
                sc_k = k[:-14] + ".weight_scale"
                if sc_k in meta_keys:
                    consumed_scales.add(sc_k)

        items = [(k, v) for k, v in meta_keys.items() if k not in consumed_scales]
        items.sort(key=lambda kv: int(kv[1]["data_offsets"][0]))
        for name, info in items:
            s_off, e_off = int(info["data_offsets"][0]), int(info["data_offsets"][1])
            fh.seek(data_base + s_off)
            raw = fh.read(e_off - s_off)
            if name.endswith(".weight_packed"):
                sc_k = name[:-14] + ".weight_scale"
                if sc_k in meta_keys:
                    sc_info = meta_keys[sc_k]
                    fh.seek(data_base + int(sc_info["data_offsets"][0]))
                    sc_raw = fh.read(int(sc_info["data_offsets"][1]) - int(sc_info["data_offsets"][0]))
                    pk_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(info["shape"]))
                    sc_u8 = np.frombuffer(sc_raw, dtype=np.uint8).reshape(tuple(sc_info["shape"]))
                    decoded = decode_mxfp4_pair(pk_u8, sc_u8)
                    base_name = name[:-14] + ".weight"
                    if decoded.ndim == 3:
                        stem = name[:-14]
                        for e_i in range(decoded.shape[0]):
                            if ".experts." in stem:
                                pfx, sfx = stem.split(".experts.", 1)
                                yielded_name = f"{pfx}.experts.{e_i}.{sfx}.weight"
                            else:
                                yielded_name = f"{stem}.expert_{e_i}.weight"
                            yield yielded_name, decoded[e_i], "MXFP4"
                    else:
                        yield base_name, decoded, "MXFP4"
                    continue
            arr = _decode_raw_safetensor_bytes(raw, info["dtype"], tuple(info["shape"]))
            yield name, arr, info["dtype"]


def write_safetensor_file(sf_path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write a clean FP32 `.safetensors` sidecar for 1D norms, biases, and small non-quantized weights."""
    header: dict[str, Any] = {"__metadata__": {"format": "clover-hg-passthrough"}}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        f32 = np.ascontiguousarray(arr.astype(np.float32, copy=False))
        raw = f32.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(f32.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_bytes += b" " * pad
    with sf_path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for b in blobs:
            fh.write(b)


def resolve_model_dir(model_spec: str, hf_token: str | None = None) -> Path:
    """Resolve either a local directory or a Hugging Face Hub repo ID (`org/model`)."""
    local = Path(model_spec)
    if local.is_dir():
        return local.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"{model_spec!r} is not a local directory and `huggingface_hub` is not installed. "
            "Run `pip install huggingface_hub` or pass a local model directory."
        ) from exc
    print(f"[clover-hg] Downloading / resolving Hugging Face snapshot for {model_spec!r} ...")
    dl_path = snapshot_download(
        repo_id=model_spec,
        token=hf_token or os.environ.get("HF_TOKEN"),
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.tiktoken"],
    )
    return Path(dl_path).resolve()


def _expert_index_from_name(name: str) -> int | None:
    m = _MOE_EXPERT_INDEX_PATTERN.search(name)
    if not m:
        return None
    for g in m.groups():
        if g is not None:
            return int(g)
    return None


def precompute_rotated_router(w_gate: np.ndarray, input_name_seed: str) -> bytes:
    """Losslessly rotate an FP32 MoE router gate W_gate [E, K] into Hadamard space (`HG4OP001`, Jaccard = 1.0)."""
    e_cnt, k_orig = w_gate.shape
    k_pad = _ceil_mult(k_orig, 128)
    w_pad = np.zeros((e_cnt, k_pad), dtype=np.float32)
    w_pad[:, :k_orig] = w_gate.astype(np.float32, copy=False)
    # Use unit-gain Rademacher rotation when input activation pre_diag is shared at runtime
    signs = deterministic_rademacher(input_name_seed, k_pad)
    w_rot = fwht128_np(w_pad * signs[None, :])
    hdr = MAGIC_HGOP + struct.pack("<III", e_cnt, k_orig, k_pad)
    return hdr + signs.tobytes() + np.ascontiguousarray(w_rot).tobytes()


def convert_checkpoint(
    model_spec: str,
    out_dir: Path,
    bits: int = 4,
    moe_cold_bits: int = 4,
    hot_experts: int = 42,
    use_ldlq: bool = False,
    min_numel: int = 4096,
    verify: bool = False,
    max_verify_tensors: int = 12,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Convert all eligible 2D weight tensors in `model_spec` into canonical Clover HG format."""
    t_start = time.perf_counter()
    out_dir = out_dir.resolve()
    weights_dir = out_dir / "weights"
    ops_dir = out_dir / "precomputed_ops"
    weights_dir.mkdir(parents=True, exist_ok=True)
    ops_dir.mkdir(parents=True, exist_ok=True)

    is_local_dir = Path(model_spec).is_dir()
    eff_tok = hf_token or os.environ.get("HF_TOKEN")
    copied_meta: list[str] = []
    temp_shard_dir: Path | None = None

    if is_local_dir:
        src_dir = Path(model_spec).resolve()
        for pattern in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"):
            cand = src_dir / pattern
            if cand.is_file():
                shutil.copy2(cand, out_dir / cand.name)
                copied_meta.append(cand.name)
        index_file = src_dir / "model.safetensors.index.json"
        if index_file.is_file():
            idx_data = json.loads(index_file.read_text(encoding="utf-8"))
            shard_names = sorted(set(idx_data.get("weight_map", {}).values()))
            shard_specs: list[tuple[str, Path | None]] = [(s, src_dir / s) for s in shard_names]
        else:
            shard_specs = [(p.name, p) for p in sorted(src_dir.glob("*.safetensors"))]
    else:
        # Stream remote HF repo 1 shard at a time so 250 GB GPU containers never run out of disk!
        from huggingface_hub import HfApi, hf_hub_download

        temp_shard_dir = out_dir / "_tmp_shard_dl"
        temp_shard_dir.mkdir(parents=True, exist_ok=True)
        api = HfApi()
        repo_files = set(api.list_repo_files(repo_id=model_spec, token=eff_tok))
        for meta_fn in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "model.safetensors.index.json"):
            if meta_fn in repo_files:
                dl_m = Path(hf_hub_download(repo_id=model_spec, filename=meta_fn, local_dir=str(temp_shard_dir), token=eff_tok))
                if meta_fn != "model.safetensors.index.json":
                    shutil.copy2(dl_m, out_dir / meta_fn)
                    copied_meta.append(meta_fn)
        idx_dl = temp_shard_dir / "model.safetensors.index.json"
        if idx_dl.is_file():
            idx_data = json.loads(idx_dl.read_text(encoding="utf-8"))
            shard_names = sorted(set(idx_data.get("weight_map", {}).values()))
        else:
            shard_names = sorted(f for f in repo_files if f.endswith(".safetensors"))
        shard_specs = [(s, None) for s in shard_names]

    if not shard_specs:
        raise FileNotFoundError(f"No `.safetensors` files found in {model_spec}")

    passthrough: dict[str, np.ndarray] = {}
    tensor_records: list[dict[str, Any]] = []
    verify_records: list[dict[str, Any]] = []
    orig_total_bytes = 0
    hg_total_bytes = 0
    dtype_bytes_map = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "FLOAT8_E4M3FN": 1, "I8": 1, "U8": 1}

    for shard_idx, (sf_label, local_sf) in enumerate(shard_specs, start=1):
        if local_sf is not None:
            sf_path = local_sf
        else:
            from huggingface_hub import hf_hub_download

            assert temp_shard_dir is not None
            sf_path = Path(hf_hub_download(repo_id=model_spec, filename=sf_label, local_dir=str(temp_shard_dir), token=eff_tok))
        print(f"[clover-hg] [{shard_idx}/{len(shard_specs)}] Streaming {sf_path.name} ...", flush=True)
        for name, arr, orig_dtype in iter_safetensor_file(sf_path):
            if orig_dtype == "MXFP4":
                orig_b = int(round(arr.size * (0.5 + 1.0 / 32.0)))
            else:
                elem_b = dtype_bytes_map.get(orig_dtype, 2)
                orig_b = int(arr.size) * elem_b
            orig_total_bytes += orig_b

            # Precompute lossless Hadamard-rotated FP32 MoE router sidecars (`HG4OP001`)
            if arr.ndim == 2 and _ROUTER_GATE_PATTERN.search(name):
                op_blob = precompute_rotated_router(arr, input_name_seed=name)
                safe_op = re.sub(r"[^A-Za-z0-9_.-]+", "__", name) + ".hgop"
                (ops_dir / safe_op).write_bytes(op_blob)
                passthrough[name] = arr
                hg_total_bytes += arr.size * 4
                tensor_records.append({
                    "name": name,
                    "kind": "router_precomputed_op",
                    "op_file": f"precomputed_ops/{safe_op}",
                    "shape": list(arr.shape),
                    "bytes": arr.size * 4,
                })
                continue

            should_quant = (
                arr.ndim == 2
                and arr.size >= min_numel
                and min(arr.shape) >= 16
                and not _SKIP_QUANT_PATTERNS.search(name)
            )
            if not should_quant:
                passthrough[name] = arr
                hg_total_bytes += arr.size * 4
                tensor_records.append({
                    "name": name,
                    "kind": "passthrough_f32",
                    "shape": list(arr.shape),
                    "bytes": arr.size * 4,
                })
                continue

            exp_idx = _expert_index_from_name(name)
            target_bits = bits
            tier = "hg4_hot" if bits == 4 else "hg2_cold"
            if exp_idx is not None and moe_cold_bits == 2 and exp_idx >= hot_experts:
                target_bits = 2
                tier = "hg2_cold"

            blob, meta = pack_matrix_hg(arr, name=name, bits=target_bits, use_ldlq=use_ldlq)
            ext = ".hg4" if target_bits == 4 else ".hg2"
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "__", name) + ext
            out_file = weights_dir / safe_name
            out_file.write_bytes(blob)
            hg_total_bytes += len(blob)

            meta["file"] = f"weights/{safe_name}"
            meta["tier"] = tier
            meta["orig_dtype"] = orig_dtype
            meta["sha256_16"] = hashlib.sha256(blob).hexdigest()[:16]

            if verify and len(verify_records) < max_verify_tensors:
                v_metrics = verify_matrix_parity(arr, blob, seed=42 + len(verify_records))
                meta["verification"] = v_metrics
                verify_records.append({"name": name, "bits": target_bits, **v_metrics})

            tensor_records.append(meta)

        if local_sf is None and sf_path.is_file():
            sf_path.unlink(missing_ok=True)

    if temp_shard_dir is not None and temp_shard_dir.is_dir():
        shutil.rmtree(temp_shard_dir, ignore_errors=True)

    if passthrough:
        pt_path = out_dir / "passthrough.safetensors"
        write_safetensor_file(pt_path, passthrough)

    wall_s = time.perf_counter() - t_start
    summary: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "source_model": str(model_spec),
        "copied_metadata": copied_meta,
        "quantized_tensors": sum(1 for r in tensor_records if "magic" in r),
        "passthrough_tensors": len(passthrough),
        "orig_total_bytes": orig_total_bytes,
        "hg_total_bytes": hg_total_bytes,
        "compression_ratio_vs_orig": round(orig_total_bytes / max(hg_total_bytes, 1), 3),
        "wall_seconds": round(wall_s, 3),
        "verification_summary": None,
        "tensors": tensor_records,
    }
    if verify_records:
        mean_hg_cos = float(np.mean([r["hg_cosine"] for r in verify_records]))
        mean_i4_cos = float(np.mean([r["naive_int4_cosine"] for r in verify_records]))
        mean_hg_sqnr = float(np.mean([r["hg_sqnr_db"] for r in verify_records]))
        mean_i4_sqnr = float(np.mean([r["naive_int4_sqnr_db"] for r in verify_records]))
        hg4_recs = [r for r in verify_records if r["bits"] == 4]
        hg2_recs = [r for r in verify_records if r["bits"] == 2]
        summary["verification_summary"] = {
            "checked_tensors": len(verify_records),
            "mean_hg_cosine": round(mean_hg_cos, 6),
            "hg4_mean_cosine": round(float(np.mean([r["hg_cosine"] for r in hg4_recs])), 6) if hg4_recs else None,
            "hg2_mean_cosine": round(float(np.mean([r["hg_cosine"] for r in hg2_recs])), 6) if hg2_recs else None,
            "mean_naive_int4_cosine": round(mean_i4_cos, 6),
            "mean_hg_sqnr_db": round(mean_hg_sqnr, 2),
            "mean_naive_int4_sqnr_db": round(mean_i4_sqnr, 2),
            "samples": verify_records,
        }

    manifest_path = out_dir / "hg_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_self_test() -> int:
    """Zero-download end-to-end test: builds a synthetic BF16 Transformer+MoE with 30x outlier spikes, converts to HG4+K31H, and verifies parity."""
    print("[clover-hg] Running built-in zero-download self-test (BF16 outlier Transformer + MoE -> HG4-16R / K31H) ...")
    rng = np.random.Generator(np.random.PCG64(20261124))
    with tempfile.TemporaryDirectory(prefix="clover_hg_selftest_") as tmp:
        tmp_p = Path(tmp)
        src_dir = tmp_p / "synthetic_bf16_model"
        out_dir = tmp_p / "converted_hg"
        src_dir.mkdir(parents=True)

        # Create realistic weights with sharp outlier column channels (where naive INT4 collapses)
        q_proj = rng.standard_normal((256, 512), dtype=np.float32) * 0.02
        q_proj[:, [7, 131, 389]] *= 30.0  # 30x channel outliers!
        down_proj = rng.standard_normal((512, 256), dtype=np.float32) * 0.02
        down_proj[:, [19, 144]] *= 25.0
        exp0 = rng.standard_normal((128, 512), dtype=np.float32) * 0.02
        exp3 = rng.standard_normal((128, 512), dtype=np.float32) * 0.02
        router = rng.standard_normal((8, 512), dtype=np.float32) * 0.02
        norm = np.ones((512,), dtype=np.float32)

        # Write real BF16 safetensors shard
        tensors_bf16 = {
            "model.layers.0.self_attn.q_proj.weight": q_proj,
            "model.layers.0.mlp.down_proj.weight": down_proj,
            "model.layers.0.mlp.gate.weight": router,
            "model.layers.0.mlp.experts.0.up_proj.weight": exp0,
            "model.layers.0.mlp.experts.3.up_proj.weight": exp3,
            "model.layers.0.input_layernorm.weight": norm,
        }
        hdr: dict[str, Any] = {}
        blobs: list[bytes] = []
        off = 0
        for k, arr in tensors_bf16.items():
            u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
            bf16_u16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
            raw = bf16_u16.tobytes()
            hdr[k] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
            blobs.append(raw)
            off += len(raw)
        hb = json.dumps(hdr).encode("utf-8")
        hb += b" " * ((8 - len(hb) % 8) % 8)
        with (src_dir / "model.safetensors").open("wb") as f:
            f.write(struct.pack("<Q", len(hb)))
            f.write(hb)
            for b in blobs:
                f.write(b)
        (src_dir / "config.json").write_text(json.dumps({"model_type": "synthetic_moe", "hidden_size": 512}), encoding="utf-8")

        rep = convert_checkpoint(
            str(src_dir),
            out_dir,
            bits=4,
            moe_cold_bits=2,
            hot_experts=2,
            use_ldlq=True,
            verify=True,
        )
        vs = rep["verification_summary"]
        assert vs is not None
        print(
            f"[clover-hg] Self-test summary:\n"
            f"  Quantized tensors : {rep['quantized_tensors']} (3x hot HG4-16R + 1x cold HG2-16R + router HG4OP001)\n"
            f"  HG4-16R (4-bit)   : Cosine = {vs['hg4_mean_cosine']:.6f}  (vs Naive INT4 Cosine = {vs['mean_naive_int4_cosine']:.6f})\n"
            f"  HG2-16R (2-bit)   : Cosine = {vs['hg2_mean_cosine']:.6f}  (cold MoE long-tail expert)\n"
            f"  Combined SQNR     : {vs['mean_hg_sqnr_db']:+.2f} dB  (vs Naive INT4 {vs['mean_naive_int4_sqnr_db']:+.2f} dB)\n"
            f"  Compression ratio : {rep['compression_ratio_vs_orig']:.2f}x vs BF16"
        )
        # Verify lossless Hadamard router Top-2 parity (Jaccard = 1.000000)
        op_files = list((out_dir / "precomputed_ops").glob("*.hgop"))
        assert len(op_files) == 1
        op_raw = op_files[0].read_bytes()
        assert op_raw[:8] == MAGIC_HGOP
        e_cnt, k_orig, k_pad = struct.unpack_from("<III", op_raw, 8)
        signs = np.frombuffer(op_raw, dtype=np.float32, count=k_pad, offset=20)
        w_rot = np.frombuffer(op_raw, dtype=np.float32, count=e_cnt * k_pad, offset=20 + k_pad * 4).reshape(e_cnt, k_pad)
        x_test = rng.standard_normal((64, k_orig), dtype=np.float32)
        logits_orig = x_test @ tensors_bf16["model.layers.0.mlp.gate.weight"].T
        logits_rot = fwht128_np(x_test * signs[:k_orig][None, :]) @ w_rot.T
        top2_orig = np.argsort(-logits_orig, axis=-1)[:, :2]
        top2_rot = np.argsort(-logits_rot, axis=-1)[:, :2]
        jaccard = float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(top2_orig, top2_rot)]))
        print(f"  Router Top-2 Jaccard : {jaccard:.6f} (100% lossless Hadamard routing)")
        assert vs["hg4_mean_cosine"] is not None and vs["hg4_mean_cosine"] > 0.995, (
            f"Expected HG4 cosine > 0.995, got {vs['hg4_mean_cosine']}"
        )
        assert vs["hg2_mean_cosine"] is not None and vs["hg2_mean_cosine"] > 0.930, (
            f"Expected HG2 2-bit cosine > 0.930, got {vs['hg2_mean_cosine']}"
        )
        assert vs["mean_hg_sqnr_db"] > vs["mean_naive_int4_sqnr_db"] + 3.0, "Expected HG to beat naive INT4 on outlier channels"
        assert jaccard == 1.0, f"Expected router Jaccard 1.0, got {jaccard}"
        print("[clover-hg] SELF-TEST PASSED.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert any Hugging Face safetensors model to Clover HG4-16R / K31H format.")
    ap.add_argument("--model", help="Local directory path or Hugging Face Hub repo ID (e.g., Qwen/Qwen3-8B)")
    ap.add_argument("--out", help="Output directory for converted HG weights, precomputed_ops/, and hg_manifest.json")
    ap.add_argument("--bits", type=int, choices=(2, 4), default=4, help="Primary weight bits per parameter (default: 4 = HG4-16R)")
    ap.add_argument("--moe-cold-bits", type=int, choices=(2, 4), default=4, help="Bits for cold MoE experts >= --hot-experts (set 2 for K31H hybrid)")
    ap.add_argument("--hot-experts", type=int, default=42, help="Number of routed experts per layer kept in 4-bit HG4 when --moe-cold-bits=2")
    ap.add_argument("--ldlq", action="store_true", help="Enable sequential Cholesky LDLQ Hessian error-feedback rounding per 128-block")
    ap.add_argument("--verify", action="store_true", help="Run Hadamard-domain forward parity check vs original weights and naive INT4")
    ap.add_argument("--max-verify-tensors", type=int, default=12, help="Maximum tensors to sample during --verify (default: 12)")
    ap.add_argument("--hf-token", default=None, help="Optional Hugging Face token for gated/private repositories")
    ap.add_argument("--self-test", action="store_true", help="Run instant zero-download end-to-end conversion & verification test")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()
    if not args.model or not args.out:
        ap.error("--model and --out are required (or pass --self-test)")

    rep = convert_checkpoint(
        model_spec=args.model,
        out_dir=Path(args.out),
        bits=args.bits,
        moe_cold_bits=args.moe_cold_bits,
        hot_experts=args.hot_experts,
        use_ldlq=args.ldlq,
        verify=args.verify,
        max_verify_tensors=args.max_verify_tensors,
        hf_token=args.hf_token,
    )
    print(
        f"\n[clover-hg] Done in {rep['wall_seconds']:.2f}s -> {Path(args.out).resolve()}\n"
        f"  Quantized tensors : {rep['quantized_tensors']} | Passthrough : {rep['passthrough_tensors']}\n"
        f"  Original bytes    : {rep['orig_total_bytes'] / (1024**2):.2f} MiB -> HG : {rep['hg_total_bytes'] / (1024**2):.2f} MiB "
        f"({rep['compression_ratio_vs_orig']:.2f}x smaller)"
    )
    if rep.get("verification_summary"):
        vs = rep["verification_summary"]
        print(
            f"  Verified Parity   : HG Cosine = {vs['mean_hg_cosine']:.6f} ({vs['mean_hg_sqnr_db']:+.2f} dB) "
            f"vs Naive INT4 = {vs['mean_naive_int4_cosine']:.6f} ({vs['mean_naive_int4_sqnr_db']:+.2f} dB)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
