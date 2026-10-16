#!/usr/bin/env python3
"""Which layers have negligible rows, and does it follow the block boundaries?

Layer 12 holds a tail three orders of magnitude below its median - 38% of G's
rows under 1% of the matrix maximum - while layers 46, 59 and 90 have no tail at
all. 12 is the first block boundary of {12,24,36,48,60,72,84}, the mod-12
structure the creator pointed at, and the same pod that measured as
position-independent earlier in this work.

Screening all 93 layers on row scales alone, which is a 49 KB read per matrix
rather than a full pass, because scale dominates row magnitude. Any layer that
shows a tail here is worth confirming with full row norms; this pass only has
to be cheap enough to cover everything.

usage: tail-scan.py
"""
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
HDR, REC = 1168, 72
BOUNDARIES = {12, 24, 36, 48, 60, 72, 84}
BIG = ("G", "O", "Q", "K", "V", "SH1", "SH2", "SH3")


def read_index(layer):
    p = os.path.join(ROOT, "trunk-%d" % layer, "index.bin")
    raw = open(p, "rb").read()
    count = struct.unpack_from("<I", raw, 16)[0]
    out = []
    for i in range(count):
        o = HDR + i * REC
        rid, group, kind, rows, cols, _ = struct.unpack_from("<6I", raw, o)
        off, length = struct.unpack_from("<2Q", raw, o + 24)
        out.append(dict(id=rid, group=group, kind=kind, rows=rows,
                        columns=cols, offset=off, length=length))
    return out


def main():
    print("%-6s %-5s %8s %8s %10s %10s %8s"
          % ("layer", "slot", "rows", "<1%", "frac", "p1/p50", "bytes GB"))
    total_skippable = 0.0
    per_layer = {}

    for layer in range(93):
        d = os.path.join(ROOT, "trunk-%d" % layer)
        if not os.path.exists(os.path.join(d, "index.bin")):
            continue
        names = ["common.bin", "attention.bin" if layer else "kda.bin",
                 "routed-shared.bin" if layer else "dense.bin"]
        layer_skip = 0.0
        for rec in read_index(layer):
            if rec["kind"] != 1 or rec["id"] >= len(SLOTN):
                continue
            slot = SLOTN[rec["id"]]
            if slot not in BIG:
                continue
            rows, cols = rec["rows"], rec["columns"]
            with open(os.path.join(d, names[rec["group"]]), "rb") as f:
                f.seek(rec["offset"])
                sc = struct.unpack("<%df" % rows, f.read(rows * 4))
            top = max(sc)
            if top <= 0:
                continue
            tiny = sum(1 for s in sc if s < top / 100.0)
            srt = sorted(sc)
            p1 = srt[max(0, len(srt) // 100)]
            p50 = srt[len(srt) // 2]
            if tiny:
                gb = tiny * cols / float(1 << 30)
                layer_skip += gb
                print("%-6d %-5s %8d %8d %9.1f%% %10.4g %8.3f"
                      % (layer, slot, rows, tiny, 100.0 * tiny / rows,
                         (p1 / p50) if p50 else 0.0, gb))
        if layer_skip:
            per_layer[layer] = layer_skip
            total_skippable += layer_skip

    print()
    print("layers with any negligible rows: %s" % sorted(per_layer))
    hit_b = sorted(set(per_layer) & BOUNDARIES)
    print("of which block boundaries {12,24,36,48,60,72,84}: %s" % hit_b)
    print("boundaries with NO tail: %s" % sorted(BOUNDARIES - set(per_layer)))
    print()
    print("total bytes in rows under 1%% of their matrix max: %.3f GB of 52.39 GB (%.2f%%)"
          % (total_skippable, 100.0 * total_skippable / 52.3924))
    if per_layer:
        print("largest contributors: %s"
              % ", ".join("L%d %.2fGB" % kv for kv in
                          sorted(per_layer.items(), key=lambda kv: -kv[1])[:8]))


if __name__ == "__main__":
    main()
