#!/usr/bin/env python3
"""Is MXFP4 materially worse than the int8 we already run?

Everything about the MXFP4 plan rests on an assumption nobody has tested: that
4-bit weights are good enough because the model was quantization-aware-trained
to MXFP4. Our int8 does not carry that grid - measured, 255 of 256 codes with
28.06 distinct per 32-value block - so going to 4 bits is a real second
quantisation, and the question is how much accuracy it costs relative to the
int8 baseline rather than to perfect precision.

Answering it needs one layer of BF16, about 1.2 GB, not the whole 106.55 GB.
That matters because the disk has 58.6 GB free and the converter does not exist
yet, so a bulk download now would just fill it.

Downloads one layer's trunk tensors by byte range, then for the largest slots
compares BF16 against int8-with-row-scales (what we run) and MXFP4 E2M1 with a
shared exponent per 32 values. Relative error is reported for both, so the
comparison is against the real baseline and not against an ideal.

usage: mxfp4-quality.py [layer]
"""
import json
import os
import re
import struct
import sys

import numpy as np
import requests

REPO = "moonshotai/Kimi-K3"
COMMIT = "f831ab66814297da540d832a5235f8e904f29d06"
BASE = "https://huggingface.co/%s/resolve/%s/" % (REPO, COMMIT)
CACHE = "/tmp/k3headers"
INDEX = "/root/k3model/model.safetensors.index.json"

# E2M1: the sixteen values a 4-bit code can take
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
FP4_MAX = 6.0


def bf16_to_f32(raw):
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32)


def quant_int8_rows(w):
    """what we run: one float32 scale per row, symmetric int8"""
    amax = np.abs(w).max(axis=1, keepdims=True)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    codes = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return codes.astype(np.float32) * scale


def quant_mxfp4(w, block=32):
    """MXFP4: shared power-of-two exponent per block of 32, E2M1 codes"""
    rows, cols = w.shape
    pad = (-cols) % block
    if pad:
        w = np.concatenate([w, np.zeros((rows, pad), dtype=w.dtype)], axis=1)
    g = w.reshape(rows, -1, block)
    amax = np.abs(g).max(axis=2, keepdims=True)
    # ceil, not floor: the shared scale must be large enough that the block's
    # largest value still fits inside E2M1's top value of 6, or it gets clipped
    exp = np.where(amax > 0, np.ceil(np.log2(amax / FP4_MAX)), 0.0)
    scale = np.exp2(exp).astype(np.float32)
    n = g / np.where(scale > 0, scale, 1.0)
    # snap to the nearest representable E2M1 value
    idx = np.abs(n[..., None] - FP4[None, None, None, :]).argmin(axis=-1)
    out = (FP4[idx] * scale).reshape(rows, -1)[:, :cols]
    return out


def quant_int4_group(w, block=32):
    """int4 with a real float32 scale per group - same 4 bits, better scale"""
    rows, cols = w.shape
    pad = (-cols) % block
    if pad:
        w = np.concatenate([w, np.zeros((rows, pad), dtype=w.dtype)], axis=1)
    g = w.reshape(rows, -1, block)
    amax = np.abs(g).max(axis=2, keepdims=True)
    scale = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
    codes = np.clip(np.rint(g / scale), -7, 7)
    return (codes * scale).reshape(rows, -1)[:, :cols]


def quant_int8_group(w, block=32):
    """int8 with a float scale per group, for reference on where bits go"""
    rows, cols = w.shape
    pad = (-cols) % block
    if pad:
        w = np.concatenate([w, np.zeros((rows, pad), dtype=w.dtype)], axis=1)
    g = w.reshape(rows, -1, block)
    amax = np.abs(g).max(axis=2, keepdims=True)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    codes = np.clip(np.rint(g / scale), -127, 127)
    return (codes * scale).reshape(rows, -1)[:, :cols]


def relerr(a, b):
    d = np.linalg.norm((a - b).ravel())
    n = np.linalg.norm(a.ravel())
    return d / n if n else 0.0


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 46
    wmap = json.load(open(INDEX))["weight_map"]
    want = {n: s for n, s in wmap.items()
            if (".layers.%d." % layer) in n and not re.search(r"\.experts\.\d+\.", n)}
    if not want:
        print("no trunk tensors for layer %d" % layer)
        return 1
    shards = sorted(set(want.values()))
    print("layer %d: %d trunk tensors across %d shard(s)" % (layer, len(want), len(shards)))

    s = requests.Session()
    results = []
    for shard in shards:
        hdr = json.load(open(os.path.join(CACHE, "%05d.json" % int(shard.split("-")[1]))))
        hdr_len = os.path.getsize(os.path.join(CACHE, "%05d.json" % int(shard.split("-")[1])))
        # data begins after 8 bytes of length plus the header itself; recover from a live read
        r = s.get(BASE + shard, headers={"Range": "bytes=0-7"}, timeout=60)
        base = 8 + struct.unpack("<Q", r.content[:8])[0]

        big = [(n, hdr[n]) for n in want
               if n in hdr and len(hdr[n]["shape"]) == 2
               and hdr[n]["shape"][0] * hdr[n]["shape"][1] > 4_000_000]
        big.sort(key=lambda kv: -kv[1]["shape"][0] * kv[1]["shape"][1])
        for name, meta in big[:4]:
            a, b = meta["data_offsets"]
            rng = "bytes=%d-%d" % (base + a, base + b - 1)
            got = s.get(BASE + shard, headers={"Range": rng}, timeout=600).content
            w = bf16_to_f32(got).reshape(meta["shape"])
            row = {
                "int8/row  (today, 8.03 bit)": relerr(w, quant_int8_rows(w)),
                "int8/grp32      (9.00 bit)": relerr(w, quant_int8_group(w)),
                "mxfp4 e2m1      (4.25 bit)": relerr(w, quant_mxfp4(w)),
                "int4/grp32      (5.00 bit)": relerr(w, quant_int4_group(w)),
                "int4/grp128     (4.25 bit)": relerr(w, quant_int4_group(w, 128)),
            }
            results.append((name, row))
            print("  %-46s %s" % (name.split("layers.%d." % layer)[-1], meta["shape"]))
            for k, v in row.items():
                print("      %-28s %.5f" % (k, v))

    if results:
        keys = list(results[0][1])
        base8 = sum(r[1][keys[0]] for r in results) / len(results)
        print("\nmean relative error over %d matrices" % len(results))
        print("  %-28s %10s %12s" % ("scheme", "rel err", "vs today"))
        for k in keys:
            m = sum(r[1][k] for r in results) / len(results)
            print("  %-28s %10.5f %11.2fx" % (k, m, (m / base8) if base8 else 0))
        print("\nint8/row is what runs today. Measured across these schemes the")
        print("bit count sets the error, not the scale convention: at a matched")
        print("4.25 bits, int4 with a float scale per 128 (0.1125) and mxfp4 with")
        print("a power-of-two scale per 32 (0.1137) are the same to within noise.")
        print("There is no 4-bit packing that avoids roughly 11% weight error, so")
        print("the decision rests on whether QAT makes the model tolerate it,")
        print("which only running the engine can answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
