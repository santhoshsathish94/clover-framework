#!/usr/bin/env python3
"""Find the real layout by looking at the bytes, not by assuming an order.

The previous attempt computed record offsets as a cumulative sum in id order
and "validated" it against the file size. That check cannot fail: sizes sum to
the total regardless of the order records were actually written in. It passed,
and the weights it read were nonsense - a row scale of -1.46e38 and a perfectly
uniform histogram over all 256 int8 values.

So: locate the int8 matrix empirically. For the interleaved layout the engine's
Q uses (4-byte float32 scale, then `in` int8 weights, repeating), a correct base
offset makes every float at stride 7172 a small positive number. That is a
strong, cheap test, and a wrong offset fails it immediately.

usage: find-layout.py [layer] [slot-id]
"""
import json
import os
import struct
import sys

ROOT = "/opt/clover-k3/clover-intelegence/dataset"
GROUPS = ["common.bin", "attention.bin", "routed-shared.bin", "dense.bin", "kda.bin"]


def plausible(path, base, rowb, nrows, probes=48):
    """fraction of sampled row-starts whose float32 looks like a quantisation scale"""
    good = 0
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        for i in range(probes):
            r = int(nrows * i / float(probes))
            at = base + r * rowb
            if at + 4 > size:
                return 0.0
            f.seek(at)
            v = struct.unpack("<f", f.read(4))[0]
            if v == v and 0.0 < v < 1.0:
                good += 1
    return good / float(probes)


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 46
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    d = os.path.join(ROOT, "trunk-%d" % layer)
    bv = json.load(open(os.path.join(d, "build-values.json")))
    rec = next(r for r in bv["records"] if r["id"] == want)
    g = rec["group"]
    name = GROUPS[g] if g < len(GROUPS) else "group%d" % g
    path = os.path.join(d, name)
    rows, cols, nbytes = rec["rows"], rec["columns"], rec["bytes"]
    rowb = 4 + cols

    print("slot id %d, %dx%d, %d bytes, in %s (%d bytes)"
          % (want, rows, cols, nbytes, name, os.path.getsize(path)))
    print("interleaved row stride would be %d; rows*stride = %d" % (rowb, rows * rowb))

    print("\nfirst 48 bytes of the file:")
    with open(path, "rb") as f:
        head = f.read(48)
    print("  " + " ".join("%02x" % b for b in head))
    print("  as float32: %s" % [round(x, 8) for x in struct.unpack("<12f", head)])

    # try every record boundary in the file as a candidate base
    print("\ncandidate bases (cumulative over records, both id order and file order)")
    cands = []
    off = 0
    for r in sorted((r for r in bv["records"] if r["group"] == g), key=lambda r: r["id"]):
        if r["id"] == want:
            cands.append(("id-order", off))
        off += r["bytes"]
    # also: matrices may be written before the small vectors
    off = 0
    for r in sorted((r for r in bv["records"] if r["group"] == g),
                    key=lambda r: (r["kind"] != 1, r["id"])):
        if r["id"] == want:
            cands.append(("matrices-first", off))
        off += r["bytes"]
    cands.append(("zero", 0))

    best = None
    for label, base in cands:
        sc = plausible(path, base, rowb, rows)
        print("  %-16s base %12d  plausible scales %5.1f%%" % (label, base, 100.0 * sc))
        if best is None or sc > best[2]:
            best = (label, base, sc)

    if best and best[2] >= 0.95:
        label, base, sc = best
        print("\nlayout found: %s, base %d" % (label, base))
        with open(path, "rb") as f:
            f.seek(base)
            raw = f.read(rowb)
        scale = struct.unpack("<f", raw[:4])[0]
        w = [v - 256 if v > 127 else v for v in raw[4:4 + 64]]
        print("  row 0 scale %.8g" % scale)
        print("  row 0 first 64 int8: %s" % w)
    else:
        print("\nNo candidate yields plausible scales. The interleaved assumption")
        print("may be wrong for this record, or the record is stored elsewhere.")
        print("Not guessing further - the engine's index.bin is the authority.")


if __name__ == "__main__":
    main()
