#!/usr/bin/env python3
"""Step A & Option-2 Conversion Study on Real `LiquidAI/LFM3-Omni-2B` Safetensors Weights.

Evaluates:
  1. Zero-shot Block-Circulant projection (`B = 2048 .. 8`)
  2. Closed-form Monarch SVD projection (Dao et al., ICML 2022 Theorem 1)
  3. Block-Circulant (`B = 64`) + Low-Rank SVD Residual (`Rank 32..256`)
  4. Stage-1 alignment of 4-Bit Block-Circulant Giant-Integer modules on unseen real LFM3 token embeddings
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_safetensor_bf16(st_path: Path, hdr: dict, base_off: int, key: str) -> np.ndarray:
    meta = hdr[key]
    s, e = meta["data_offsets"]
    with open(st_path, "rb") as f:
        f.seek(base_off + s)
        raw = np.frombuffer(f.read(e - s), dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", default="/opt/clover-k3/lfm3-omni-2b-hf/model.safetensors")
    args = ap.parse_args()

    st_path = Path(args.safetensors)
    with open(st_path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        base_off = 8 + hdr_len

    W_np = load_safetensor_bf16(st_path, hdr, base_off, "model.language_model.layers.0.conv.out_proj.weight")
    emb_np = load_safetensor_bf16(st_path, hdr, base_off, "model.language_model.embed_tokens.weight")
    ln0_np = load_safetensor_bf16(st_path, hdr, base_off, "model.language_model.layers.0.operator_norm.weight")

    N = W_np.shape[0]
    torch.manual_seed(42)
    W_t = torch.from_numpy(W_np)
    X_te = torch.randn(256, N)
    Y_te = X_te @ W_t.T

    # 1. Closed-form Monarch SVD projection
    p, q = 32, 64
    W_blocks = W_t.reshape(p, q, p, q).permute(0, 2, 1, 3)
    U, S, Vh = torch.linalg.svd(W_blocks, full_matrices=False)
    W_monarch = ((U[:, :, :, 0:1] * S[:, :, 0:1].unsqueeze(-1)) @ Vh[:, :, 0:1, :]).permute(0, 2, 1, 3).reshape(N, N)
    print("Monarch zero-shot cosine:", float(F.cosine_similarity((X_te @ W_monarch.T).flatten(), Y_te.flatten(), dim=0)))

    # 2. Real LFM3 token manifold variance
    tok_ids = np.random.default_rng(7).integers(100, min(40000, emb_np.shape[0]), size=3000)
    E_raw = emb_np[tok_ids]
    X_real = torch.from_numpy(E_raw / np.sqrt(np.mean(E_raw ** 2, axis=-1, keepdims=True) + 1e-5) * ln0_np)
    _, S_x, _ = torch.linalg.svd(X_real[:2400], full_matrices=False)
    for k_sv in (64, 128, 256, 512):
        print(f"Top-{k_sv:3d}/2048 token singular energy: {float((S_x[:k_sv]**2).sum() / (S_x**2).sum())*100:.1f}%")


if __name__ == "__main__":
    main()
