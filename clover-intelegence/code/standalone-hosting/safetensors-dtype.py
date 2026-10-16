#!/usr/bin/env python3
"""What dtype does the original trunk checkpoint actually store?

The model card states MXFP4 weights with MXFP8 activations, applied from SFT
onward as quantization-aware training. Our prepared trunk is int8 at 1.0006
bytes per value. Those cannot both be the native format, and which one is the
source decides whether the 52.39 GB we stream per token is the model's real
size or an artefact of our preparation.

A safetensors file answers this directly: 8 bytes of little-endian header
length, then a JSON header mapping each tensor to its dtype and shape. No need
to read a single weight.

usage: safetensors-dtype.py PATH
"""
import collections
import json
import struct
import sys

BITS = {
    "F64": 64, "F32": 32, "F16": 16, "BF16": 16,
    "I64": 64, "I32": 32, "I16": 16, "I8": 8, "U8": 8,
    "F8_E4M3": 8, "F8_E5M2": 8, "BOOL": 8, "U16": 16, "U32": 32, "U64": 64,
    "F4": 4, "FP4": 4,
}
GB = float(1 << 30)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/opt/clover-k3/clover-data/trunk/trunk.safetensors"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))

    by_dtype = collections.defaultdict(lambda: [0, 0, 0])  # count, values, bytes
    examples = {}
    for name, meta in header.items():
        if name == "__metadata__":
            print("__metadata__: %s" % json.dumps(meta)[:400])
            continue
        dt = meta.get("dtype")
        shape = meta.get("shape", [])
        off = meta.get("data_offsets", [0, 0])
        vals = 1
        for s in shape:
            vals *= s
        nb = off[1] - off[0]
        e = by_dtype[dt]
        e[0] += 1
        e[1] += vals
        e[2] += nb
        examples.setdefault(dt, (name, shape, nb, vals))

    total_b = sum(v[2] for v in by_dtype.values())
    print("%d tensors, %.4f GB on disk" % (sum(v[0] for v in by_dtype.values()), total_b / GB))
    print()
    print("  %-10s %7s %16s %12s %12s  %s" % ("dtype", "count", "values", "GB", "bytes/val", "example"))
    for dt, (c, vals, nb) in sorted(by_dtype.items(), key=lambda kv: -kv[1][2]):
        name, shape, enb, evals = examples[dt]
        print("  %-10s %7d %16d %12.4f %12.4f  %s %s"
              % (dt, c, vals, nb / GB, (nb / float(vals)) if vals else 0,
                 name[:44], shape))

    print()
    # the question that matters
    packed = sum(v[2] for d, v in by_dtype.items() if BITS.get(d, 99) <= 8)
    wide = total_b - packed
    print("<=8-bit tensors: %.4f GB   wider tensors: %.4f GB" % (packed / GB, wide / GB))
    print("prepared trunk is 52.3924 GB of int8 at ~1.0006 bytes/value")
    if total_b < 40 * GB:
        print("the source is SMALLER than our prepared trunk: we upconverted")
    elif total_b > 60 * GB:
        print("the source is LARGER than our prepared trunk: we compressed it")
    else:
        print("source and prepared trunk are comparable in size")


if __name__ == "__main__":
    main()
