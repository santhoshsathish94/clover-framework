#!/usr/bin/env python3
"""Do experts of one layer share structure? Uses matrices dumped by LB_DUMP (/tmp/one-layer/exp-E-{gate,up,down}.f32).
usage: expert-shared.py MATRIX SIDE EXPERT,EXPERT,...   MATRIX gate|up|down, SIDE input|output|mean
The first half of the experts fits the shared basis, the second half is held out."""
import sys, numpy as np

name, side = sys.argv[1], sys.argv[2]
ids = [int(x) for x in sys.argv[3].split(",")]
half = len(ids) // 2
fracs = (0.25, 0.5, 0.7, 0.8, 0.9)


def load(e):
    rows, width = (3584, 3072) if name == "down" else (3072, 3584)
    return np.fromfile(f"/tmp/one-layer/exp-{e}-{name}.f32", dtype=np.float32).reshape(rows, width)


def capture(w, basis):
    # fraction of ||w||^2 kept by the first k basis vectors, for every k
    z = (w @ basis) if side == "input" else (basis.T @ w)
    e = (z.astype(np.float64) ** 2).sum(axis=0 if side == "input" else 1)
    return np.cumsum(e) / (w.astype(np.float64) ** 2).sum()


mats = [load(e) for e in ids]
train, test = mats[:half], mats[half:]
rows, width = train[0].shape

if side == "mean":
    total = sum(m.astype(np.float64) for m in mats)
    cos = []
    for m in mats:
        m64 = m.astype(np.float64)
        o = (total - m64) / (len(mats) - 1)
        cos.append(float((m64 * o).sum() / np.sqrt((m64 ** 2).sum() * (o ** 2).sum())))
    print(f"{name}: cosine between an expert and the mean of the other {len(mats) - 1}: mean {np.mean(cos):.4f}, "
          f"max |.| {np.max(np.abs(cos)):.4f}; two unrelated matrices of this size would give about {1 / np.sqrt(rows * width):.5f}", flush=True)
    sys.exit(0)

dim = width if side == "input" else rows
g = np.zeros((dim, dim), dtype=np.float64)
for m in train:
    g += (m.T @ m if side == "input" else m @ m.T).astype(np.float64)
vals, vecs = np.linalg.eigh(g)
basis = vecs[:, np.argsort(vals)[::-1]].astype(np.float32)
cap_tr = np.mean([capture(m, basis) for m in train], axis=0)
cap_te = np.mean([capture(m, basis) for m in test], axis=0)
for f in fracs:
    k = int(f * dim)
    print(f"{name} shared {side:6s}-space basis, rank {k:4d} = {f:.2f} of {dim}: energy kept in-sample {cap_tr[k - 1]:.4f}, "
          f"held-out experts {cap_te[k - 1]:.4f}, unrelated-experts baseline {f:.4f}", flush=True)
