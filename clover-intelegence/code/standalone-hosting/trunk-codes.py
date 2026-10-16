#!/usr/bin/env python3
"""Is the trunk already 4-bit in substance? Read index.bin, not assumptions.

Two earlier attempts guessed the record layout and both were wrong: offsets were
taken in id order, and the weights were read as a 4-byte scale interleaved per
row. The authority is index.bin plus operator_matrix_view, which together give

  PreparedHeader  1168 B: magic[8], version, layer, count, reserved,
                          lengths[3], hashes[3][32], float palette[256]
  PreparedRecord    72 B: id, group, kind, rows, columns, count,
                          offset, length, hash[32]
  kind 1 record   : rows float32 scales, THEN rows*columns int8 weights

The question is whether int8 -> MXFP4 would be recovering a grid the model was
trained on or quantising a second time. Two things decide it and both are now
readable: whether the palette is the identity, and how many of the 256 codes
the weights actually use. If only ~16 codes appear, the data is already 4-bit
and halving the trunk costs nothing in precision.

usage: trunk-codes.py [layer] [slot-id]
"""
import collections
import json
import os
import struct
import sys

SLOTN = [
    "ARN", "ARP", "MRN", "MRP", "IN_LN", "POST_LN", "G", "O",
    "Q", "K", "V", "B", "FA", "FB", "CQ", "CK", "CV", "ALOG", "DTB", "ONORM",
    "QA", "QAN", "QB", "KA", "KAN", "KB",
    "MGATE", "MUP", "MDOWN",
    "GATE", "GBIAS", "EDOWN", "EUP", "ENORM", "SH1", "SH3", "SH2",
]
ROOT = "/opt/clover-k3/clover-intelegence/dataset"
HDR = 1168
REC = 72


def read_index(layer):
    p = os.path.join(ROOT, "trunk-%d" % layer, "index.bin")
    raw = open(p, "rb").read()
    magic = raw[:8]
    version, lay, count, reserved = struct.unpack_from("<4I", raw, 8)
    lengths = struct.unpack_from("<3Q", raw, 24)
    palette = struct.unpack_from("<256f", raw, 48 + 96)
    if magic != b"K3TRK001" or lay != layer:
        raise SystemExit("index identity mismatch: %r layer %d" % (magic, lay))
    if len(raw) != HDR + count * REC:
        raise SystemExit("index size %d != %d" % (len(raw), HDR + count * REC))
    recs = []
    for i in range(count):
        o = HDR + i * REC
        rid, group, kind, rows, cols, cnt = struct.unpack_from("<6I", raw, o)
        off, length = struct.unpack_from("<2Q", raw, o + 24)
        recs.append(dict(id=rid, group=group, kind=kind, rows=rows,
                         columns=cols, count=cnt, offset=off, length=length))
    return dict(version=version, layer=lay, count=count, lengths=lengths,
                palette=palette), recs


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 46
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    hdr, recs = read_index(layer)
    print("layer %d, %d records, group lengths %s" % (layer, hdr["count"], hdr["lengths"]))

    pal = hdr["palette"]
    identity = all(pal[c] == float(c - 256 if c > 127 else c) for c in range(256))
    distinct_pal = len(set(pal))
    print("palette: identity=%s, %d distinct values, first 8 %s"
          % (identity, distinct_pal, [round(v, 6) for v in pal[:8]]))
    if not identity:
        used = sorted(set(pal))
        print("  NON-IDENTITY palette, values: %s" % [round(v, 6) for v in used[:24]])

    rec = next((r for r in recs if r["id"] == want), None)
    if rec is None:
        print("slot id %d not present in layer %d" % (want, layer))
        return 1
    if rec["kind"] != 1:
        print("slot %s is kind %d, not an int8 matrix" % (SLOTN[want], rec["kind"]))
        return 1

    names = ["common.bin", "attention.bin" if layer else "kda.bin",
             "routed-shared.bin" if layer else "dense.bin"]
    path = os.path.join(ROOT, "trunk-%d" % layer, names[rec["group"]])
    rows, cols = rec["rows"], rec["columns"]
    print("\nslot %s id %d, %dx%d, %s + %d, length %d"
          % (SLOTN[want], want, rows, cols, names[rec["group"]], rec["offset"], rec["length"]))

    f = open(path, "rb")
    f.seek(rec["offset"])
    scales = struct.unpack("<%df" % rows, f.read(rows * 4))
    bad = [s for s in scales if not (s == s) or s <= 0.0 or s > 10.0]
    print("scales: %d values, min %.6g max %.6g, implausible %d"
          % (len(scales), min(scales), max(scales), len(bad)))
    if bad:
        print("  layout still wrong - %d scales outside (0,10]. Stopping." % len(bad))
        return 1
    print("  layout CONFIRMED: all %d scales are plausible" % len(scales))

    # sample weights spread across the matrix
    counts = collections.Counter()
    per_block = collections.Counter()
    wbase = rec["offset"] + rows * 4
    sample = [int(rows * i / 128.0) for i in range(128)]
    for r in sample:
        f.seek(wbase + r * cols)
        w = f.read(cols)
        counts.update(w)
        for b in range(0, cols - 31, 32):
            per_block[len(set(w[b:b + 32]))] += 1
    f.close()

    nz = len(counts)
    total = sum(counts.values())
    print("\n%d weights sampled from %d rows" % (total, len(sample)))
    print("distinct codes used: %d of 256" % nz)
    mean_blk = sum(k * v for k, v in per_block.items()) / float(sum(per_block.values()))
    print("distinct codes per 32-value block: mean %.2f" % mean_blk)
    top = counts.most_common(20)
    print("most common codes: %s"
          % ", ".join("%d(%.2f%%)" % (c - 256 if c > 127 else c, 100.0 * n / total) for c, n in top))

    print()
    if nz <= 20:
        print("VERDICT: only %d distinct codes. The trunk is already 4-bit in" % nz)
        print("substance and int8 -> MXFP4 would be a pure repacking, no precision lost.")
    elif mean_blk <= 10:
        print("VERDICT: %d codes overall but only %.1f per block - consistent with a"
              % (nz, mean_blk))
        print("shared-scale 4-bit grid. Repacking to MXFP4 looks near-lossless.")
    else:
        print("VERDICT: %d codes, %.1f per block. The int8 carries real 8-bit"
              % (nz, mean_blk))
        print("information; MXFP4 would be a second quantisation and must be gated")
        print("on output tokens, not assumed safe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
