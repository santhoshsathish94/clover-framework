#!/usr/bin/env python3
"""Singular value spectrum of dumped expert matrices (LB_DUMP). usage: expert-svd.py EXPERT [EXPERT...]"""
import sys, numpy as np

for e in sys.argv[1:]:
    for name in ("gate", "up", "down"):
        path = f"/tmp/one-layer/exp-{e}-{name}.f32"
        raw = np.fromfile(path, dtype=np.float32)
        rows, width = (3584, 3072) if name == "down" else (3072, 3584)
        w = raw.reshape(rows, width)
        s = np.linalg.svd(w, compute_uv=False).astype(np.float64)
        energy = np.cumsum(s * s) / np.sum(s * s)
        full = min(rows, width)
        brk = rows * width / (rows + width)
        r90, r95, r99, r999 = (int(np.searchsorted(energy, q) + 1) for q in (0.90, 0.95, 0.99, 0.999))
        print(f"expert {e} {name:4s} {rows}x{width}: rank {full}, singular values >1e-6*max: {int((s > 1e-6 * s[0]).sum())}, "
              f"rank for 90/95/99/99.9% energy = {r90}/{r95}/{r99}/{r999}, break-even rank {brk:.0f}, "
              f"top sv {s[0]:.3f}, median sv {np.median(s):.4f}, distinct weight values {len(np.unique(w))}", flush=True)
