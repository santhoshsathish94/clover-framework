#!/usr/bin/env python3
"""Does the model survive 4-bit weight error? Test it without building anything.

Every 4-bit packing measured costs ~11% weight error against the 0.91% our int8
carries - 12x worse - and whether that matters depends on whether
quantization-aware training makes the model absorb it. Finding out by building
the real thing means a 106 GB download, a quantiser, a new record kind, a new
kernel and a gate: days of work on an unknown.

It can be answered now, with none of that. Take the int8 trunk we already have,
dequantise it, round every weight onto the MXFP4 grid, and store the result
back in the *same int8 container*. The engine runs completely unchanged - same
format, same kernel, same sizes - but the weights now carry exactly the error
MXFP4 would introduce. If the tokens hold, the project is worth building. If
they break, it is dead and nothing was spent.

Writes a dataset directory of symlinks with only the chosen layers made real,
so the 1.4 TB of experts is not copied and the original is never touched.

usage: fourbit-sim.py OUTDIR [layer ...]
"""
import os
import shutil
import struct
import sys

import numpy as np

SRC = "/opt/clover-k3/clover-intelegence/dataset"
HDR, REC = 1168, 72
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
MID = ((POS[1:] + POS[:-1]) / 2.0).astype(np.float32)


def read_index(layer):
    raw = open(os.path.join(SRC, "trunk-%d" % layer, "index.bin"), "rb").read()
    if raw[:8] != b"K3TRK001":
        raise SystemExit("bad magic layer %d" % layer)
    count = struct.unpack_from("<I", raw, 16)[0]
    recs = []
    for i in range(count):
        o = HDR + i * REC
        rid, group, kind, rows, cols, _ = struct.unpack_from("<6I", raw, o)
        off, length = struct.unpack_from("<2Q", raw, o + 24)
        recs.append(dict(id=rid, group=group, kind=kind, rows=rows,
                         columns=cols, offset=off, length=length))
    return recs


def to_fp4_grid(w, block=32):
    """round onto the MXFP4 grid: power-of-two shared scale per 32, E2M1 codes"""
    rows, cols = w.shape
    pad = (-cols) % block
    if pad:
        w = np.concatenate([w, np.zeros((rows, pad), dtype=np.float32)], axis=1)
    g = w.reshape(rows, -1, block)
    amax = np.abs(g).max(axis=2, keepdims=True)
    exp = np.where(amax > 0, np.ceil(np.log2(np.maximum(amax, 1e-30) / 6.0)), 0.0)
    scale = np.exp2(exp).astype(np.float32)
    n = g / scale
    # nearest positive grid value via midpoints; a 16-way broadcast would
    # allocate sixteen times the matrix and is what made the first attempt
    # unusable on an 88M-weight slot
    mag = np.abs(n)
    idx = np.searchsorted(MID, mag)
    out = POS[idx] * np.sign(n)
    return (out * scale).reshape(rows, -1)[:, :cols]


def main():
    out = sys.argv[1]
    layers = [int(a) for a in sys.argv[2:]] or list(range(0, 93, 6))
    os.makedirs(out, exist_ok=True)

    for name in os.listdir(SRC):
        dst = os.path.join(out, name)
        if os.path.lexists(dst):
            continue
        os.symlink(os.path.join(SRC, name), dst)

    print("modifying %d layers: %s" % (len(layers), layers))
    total_err, nmat = 0.0, 0
    for layer in layers:
        sub = os.path.join(out, "trunk-%d" % layer)
        if os.path.islink(sub):
            os.unlink(sub)
        os.makedirs(sub, exist_ok=True)
        srcdir = os.path.join(SRC, "trunk-%d" % layer)
        names = ["common.bin", "attention.bin" if layer else "kda.bin",
                 "routed-shared.bin" if layer else "dense.bin"]
        for f in os.listdir(srcdir):
            if f not in names:
                d = os.path.join(sub, f)
                if not os.path.lexists(d):
                    os.symlink(os.path.join(srcdir, f), d)

        recs = read_index(layer)
        buffers = {}
        for g, nm in enumerate(names):
            p = os.path.join(srcdir, nm)
            if os.path.exists(p):
                buffers[g] = bytearray(open(p, "rb").read())

        for rec in recs:
            if rec["kind"] != 1 or rec["group"] not in buffers:
                continue
            rows, cols, off = rec["rows"], rec["columns"], rec["offset"]
            buf = buffers[rec["group"]]
            scales = np.frombuffer(bytes(buf[off:off + rows * 4]), dtype="<f4")
            cbase = off + rows * 4
            codes = np.frombuffer(bytes(buf[cbase:cbase + rows * cols]),
                                  dtype=np.int8).reshape(rows, cols)
            w = codes.astype(np.float32) * scales[:, None]

            q = to_fp4_grid(w)
            # re-encode into the same int8 container; its 0.9% is small beside the 11%
            amax = np.abs(q).max(axis=1, keepdims=True)
            ns = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
            nc = np.clip(np.rint(q / ns), -127, 127).astype(np.int8)

            err = np.linalg.norm((w - nc.astype(np.float32) * ns).ravel()) / max(
                np.linalg.norm(w.ravel()), 1e-30)
            total_err += err
            nmat += 1
            buf[off:off + rows * 4] = ns.ravel().astype("<f4").tobytes()
            buf[cbase:cbase + rows * cols] = nc.tobytes()

        for g, nm in enumerate(names):
            if g in buffers:
                with open(os.path.join(sub, nm), "wb") as f:
                    f.write(bytes(buffers[g]))
        print("  layer %-3d done" % layer)

    print("\n%d matrices rewritten, mean relative error %.5f"
          % (nmat, total_err / nmat if nmat else 0))
    print("dataset ready at %s" % out)


if __name__ == "__main__":
    main()
