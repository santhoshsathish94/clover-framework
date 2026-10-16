#!/usr/bin/env python3
"""Are the big trunk matrices effectively low-rank?

The model already shows a 7168->12288 map can ride on a 128-dim bottleneck:
FA/FB cost 2.54 MB where the direct matrix would be 88 MB, a 35x saving from
factorisation. The question that follows is whether G, O, Q, K and V - five
88 MB matrices a KDA layer, 56% of the whole trunk - carry similarly
concentrated spectra.

A 12288x7168 matrix factored at rank r costs r*(12288+7168) bytes instead of
12288*7168, so anything below r=4527 saves, and r=1000 would be 4.5x smaller.
If instead the spectrum is flat, these are genuinely full-rank maps and
factorisation is closed - which is as useful to know.

Randomised SVD rather than full: a 12288x7168 exact SVD is not needed to see
whether energy concentrates in the leading directions. Reports the rank needed
to retain 90, 95 and 99 percent of the squared spectrum, and what each would
cost against the real 88 MB.

usage: rank-spectrum.py [layer] [k]
"""
import os
import struct
import sys

import numpy as np

SLOTN = ["ARN", "ARP", "MRN", "MRP", "IN_LN", "POST_LN", "G", "O",
         "Q", "K", "V", "B", "FA", "FB", "CQ", "CK", "CV", "ALOG", "DTB", "ONORM",
         "QA", "QAN", "QB", "KA", "KAN", "KB", "MGATE", "MUP", "MDOWN",
         "GATE", "GBIAS", "EDOWN", "EUP", "ENORM", "SH1", "SH3", "SH2"]
ROOT = "/opt/clover-k3/clover-intelegence/dataset"
HDR, REC = 1168, 72


def read_index(layer):
    raw = open(os.path.join(ROOT, "trunk-%d" % layer, "index.bin"), "rb").read()
    count = struct.unpack_from("<I", raw, 16)[0]
    out = []
    for i in range(count):
        o = HDR + i * REC
        rid, group, kind, rows, cols, _ = struct.unpack_from("<6I", raw, o)
        off, length = struct.unpack_from("<2Q", raw, o + 24)
        out.append(dict(id=rid, group=group, kind=kind, rows=rows,
                        columns=cols, offset=off))
    return out


def load(layer, rec):
    names = ["common.bin", "attention.bin" if layer else "kda.bin",
             "routed-shared.bin" if layer else "dense.bin"]
    path = os.path.join(ROOT, "trunk-%d" % layer, names[rec["group"]])
    rows, cols = rec["rows"], rec["columns"]
    with open(path, "rb") as f:
        f.seek(rec["offset"])
        scales = np.frombuffer(f.read(rows * 4), dtype="<f4")
        codes = np.frombuffer(f.read(rows * cols), dtype=np.int8).reshape(rows, cols)
    return codes.astype(np.float32) * scales[:, None]


def rsvd_values(w, k, p=16):
    """leading singular values by randomised projection"""
    rng = np.random.default_rng(0)
    omega = rng.standard_normal((w.shape[1], k + p), dtype=np.float32)
    y = w @ omega
    q, _ = np.linalg.qr(y)
    b = q.T @ w
    return np.linalg.svd(b, compute_uv=False)


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 46
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

    recs = {SLOTN[r["id"]]: r for r in read_index(layer)
            if r["kind"] == 1 and r["id"] < len(SLOTN)}
    print("layer %d, randomised SVD with k=%d" % (layer, k))
    print("%-5s %12s %10s %9s %8s %8s %8s %11s"
          % ("slot", "shape", "actual MB", "top-k E", "r@90%", "r@95%", "r@99%", "MB at r@95"))

    for name in ("G", "O", "Q", "K", "V"):
        if name not in recs:
            continue
        rec = recs[name]
        w = load(layer, rec)
        rows, cols = w.shape
        actual = rows * (cols + 4) / 1e6
        s = rsvd_values(w, min(k, min(rows, cols) - 1))
        # normalise by the TRUE Frobenius energy, not the sampled singular values.
        # Dividing by sum(s**2) makes the fractions sum to 1 over the sample and
        # reports a low rank for any matrix, however flat its real spectrum.
        total_energy = float(np.sum(w.astype(np.float64) ** 2))
        captured = float(np.sum(s.astype(np.float64) ** 2)) / total_energy
        e = np.cumsum(s.astype(np.float64) ** 2) / total_energy
        def rank_for(frac):
            idx = np.searchsorted(e, frac)
            return int(idx + 1) if idx < len(e) else -1
        r90, r95, r99 = rank_for(0.90), rank_for(0.95), rank_for(0.99)
        cost = (r95 * (rows + cols) / 1e6) if r95 > 0 else float("nan")
        print("%-5s %12s %10.2f %8.1f%% %8s %8s %8s %11s"
              % (name, "%dx%d" % (rows, cols), actual, 100.0 * captured,
                 r90 if r90 > 0 else ">%d" % k,
                 r95 if r95 > 0 else ">%d" % k,
                 r99 if r99 > 0 else ">%d" % k,
                 ("%.2f" % cost) if r95 > 0 else "-"))
        del w

    print()
    print("break-even rank for 12288x7168 is 4527; below that factoring saves bytes.")
    print("'>%d' means the sampled %d directions hold less than that share, so the" % (k, k))
    print("spectrum is flat and the matrix is not usefully low-rank.")


if __name__ == "__main__":
    main()
