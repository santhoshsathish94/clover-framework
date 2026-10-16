#!/usr/bin/env python3
"""Is the 4-bit structure still present in our int8 trunk?

The model card states MXFP4 weights with quantization-aware training from SFT
onward. We no longer hold the BF16 source for layers 1-92, but if those weights
were genuinely trained to an MXFP4 grid, that structure survives in the int8 we
prepared from them - and converting int8 -> MXFP4 would be recovering the native
format rather than quantising a second time.

MXFP4 is E2M1: magnitudes land on {0, .5, 1, 1.5, 2, 3, 4, 6} scaled by a shared
power of two per block of 32. The signature is therefore few distinct magnitudes
per block, not thirty-two arbitrary ones. A block of generic int8 weights would
show close to 32.

Record offsets are a cumulative sum over build-values.json in id order within
each group. That assumption is checked against the real file size before any
weight is read, so a wrong layout fails loudly instead of producing a plausible
histogram of nonsense.

usage: mxfp4-structure.py [layer] [slot-id]
"""
import collections
import json
import os
import sys

SLOTN = [
    "ARN", "ARP", "MRN", "MRP", "IN_LN", "POST_LN", "G", "O",
    "Q", "K", "V", "B", "FA", "FB", "CQ", "CK", "CV", "ALOG", "DTB", "ONORM",
    "QA", "QAN", "QB", "KA", "KAN", "KB",
    "MGATE", "MUP", "MDOWN",
    "GATE", "GBIAS", "EDOWN", "EUP", "ENORM", "SH1", "SH3", "SH2",
]
GROUPS = ["common.bin", "attention.bin", "routed-shared.bin", "dense.bin", "kda.bin"]
ROOT = "/opt/clover-k3/clover-intelegence/dataset"


def layout(layer):
    """offset of every record within its group file, validated against file size"""
    d = os.path.join(ROOT, "trunk-%d" % layer)
    bv = json.load(open(os.path.join(d, "build-values.json")))
    off = collections.defaultdict(int)
    placed = {}
    for r in bv["records"]:
        g = r["group"]
        placed[r["id"]] = (g, off[g], r)
        off[g] += r["bytes"]

    ok = True
    for g, total in sorted(off.items()):
        name = GROUPS[g] if g < len(GROUPS) else "group%d" % g
        p = os.path.join(d, name)
        actual = os.path.getsize(p) if os.path.exists(p) else -1
        flag = "ok" if actual == total else "MISMATCH"
        if actual != total:
            ok = False
        print("  %-18s computed %12d  actual %12d  %s" % (name, total, actual, flag))
    return d, placed, ok


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 46
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 6   # G, the largest slot

    print("layer %d, validating record layout" % layer)
    d, placed, ok = layout(layer)
    if not ok:
        print("\nLayout assumption is wrong - offsets cannot be trusted. Stopping.")
        return 1

    g, off, rec = placed[want]
    name = GROUPS[g] if g < len(GROUPS) else "group%d" % g
    rows, cols = rec["rows"], rec["columns"]
    print("\nslot %s (id %d), %dx%d, kind %d, at %s+%d"
          % (SLOTN[want], want, rows, cols, rec["kind"], name, off))
    if rec["kind"] != 1:
        print("not an int8 matrix; pick a kind 1 record")
        return 1

    rowb = 4 + cols
    import struct
    f = open(os.path.join(d, name), "rb")

    # sample rows spread across the matrix rather than a contiguous clump
    sample_rows = [int(rows * i / 64.0) for i in range(64)]
    per_block_distinct = collections.Counter()
    all_vals = collections.Counter()
    scales = []
    for r in sample_rows:
        f.seek(off + r * rowb)
        raw = f.read(rowb)
        scales.append(struct.unpack("<f", raw[:4])[0])
        w = raw[4:]
        for b in range(0, cols - 31, 32):
            blk = w[b:b + 32]
            mags = set(abs(v - 256 if v > 127 else v) for v in blk)
            per_block_distinct[len(mags)] += 1
        for v in w:
            all_vals[v - 256 if v > 127 else v] += 1
    f.close()

    nblocks = sum(per_block_distinct.values())
    mean_distinct = sum(k * v for k, v in per_block_distinct.items()) / float(nblocks)
    print("\n%d blocks of 32 sampled from %d rows" % (nblocks, len(sample_rows)))
    print("distinct magnitudes per 32-value block: mean %.2f" % mean_distinct)
    print("  %-10s %10s" % ("distinct", "blocks"))
    for k in sorted(per_block_distinct):
        bar = "#" * int(60.0 * per_block_distinct[k] / nblocks)
        print("  %-10d %10d  %s" % (k, per_block_distinct[k], bar))

    distinct_total = len(all_vals)
    print("\ndistinct int8 values across the whole sample: %d of 256" % distinct_total)
    top = all_vals.most_common(16)
    print("most common: %s" % ", ".join("%d(%.1f%%)" % (v, 100.0 * c / sum(all_vals.values()))
                                        for v, c in top))
    print("row scales: min %.6g max %.6g" % (min(scales), max(scales)))

    print()
    if mean_distinct <= 10:
        print("VERDICT: few distinct magnitudes per block - consistent with an")
        print("MXFP4 grid surviving in the int8. int8 -> MXFP4 would be near-lossless.")
    elif mean_distinct >= 20:
        print("VERDICT: magnitudes per block are close to arbitrary. The int8 does")
        print("NOT carry an MXFP4 grid; converting to 4 bits would be a real second")
        print("quantisation and must be gated on output, not assumed safe.")
    else:
        print("VERDICT: intermediate. Not a clean 4-bit grid; needs the error")
        print("measured directly rather than inferred from this histogram.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
