#!/usr/bin/env python3
"""Are any trunk rows negligible enough to skip reading?

This is the last way to move fewer bytes without changing precision: rows are
contiguous in the trunk layout, so skipping row o saves a contiguous `columns`
byte read. Dedup, huge pages, prefetch and a free 4-bit repack are all closed by
measurement, and the engine's own value histogram (`vh`) cannot answer it -
`val_on` is declared and never assigned, so `vstat` returns on its first line.

A row's output magnitude scales with (quantisation scale) x (int8 row norm).
Both are readable statically: scales are a contiguous rows*4 block at the record
offset, weights follow. No engine run, no activations needed.

This screens for a prunable tail. It cannot prove a row is skippable - that
depends on the activations it meets - but if no tail exists, the idea is dead
without needing a run, and if one does exist it says how large.

usage: row-magnitude.py [layer ...]
"""
import math
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


def read_index(layer):
    raw = open(os.path.join(ROOT, "trunk-%d" % layer, "index.bin"), "rb").read()
    if raw[:8] != b"K3TRK001":
        raise SystemExit("bad magic in layer %d" % layer)
    count = struct.unpack_from("<I", raw, 16)[0]
    if len(raw) != HDR + count * REC:
        raise SystemExit("index size mismatch layer %d" % layer)
    out = []
    for i in range(count):
        o = HDR + i * REC
        rid, group, kind, rows, cols, cnt = struct.unpack_from("<6I", raw, o)
        off, length = struct.unpack_from("<2Q", raw, o + 24)
        out.append(dict(id=rid, group=group, kind=kind, rows=rows,
                        columns=cols, offset=off, length=length))
    return out


def pct(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def main():
    layers = [int(a) for a in sys.argv[1:]] or [1, 12, 46, 59, 90]
    print("%-6s %-7s %9s  %-38s  %s"
          % ("layer", "slot", "rows", "row magnitude percentiles", "tail"))

    for layer in layers:
        recs = read_index(layer)
        names = ["common.bin", "attention.bin" if layer else "kda.bin",
                 "routed-shared.bin" if layer else "dense.bin"]
        for rec in recs:
            if rec["kind"] != 1 or rec["id"] >= len(SLOTN):
                continue
            if SLOTN[rec["id"]] not in ("G", "O", "Q", "V"):
                continue
            path = os.path.join(ROOT, "trunk-%d" % layer, names[rec["group"]])
            rows, cols = rec["rows"], rec["columns"]
            f = open(path, "rb")
            f.seek(rec["offset"])
            scales = struct.unpack("<%df" % rows, f.read(rows * 4))

            # int8 norm for a sample of rows; scale dominates but do not assume it
            step = max(1, rows // 256)
            mags = []
            wbase = rec["offset"] + rows * 4
            for r in range(0, rows, step):
                f.seek(wbase + r * cols)
                w = f.read(cols)
                s = 0
                for b in w:
                    v = b - 256 if b > 127 else b
                    s += v * v
                mags.append(scales[r] * math.sqrt(s))
            f.close()

            mags.sort()
            top = mags[-1]
            # how many rows are far enough below the top to be candidates
            tiny = sum(1 for m in mags if m < top / 100.0)
            small = sum(1 for m in mags if m < top / 10.0)
            print("%-6d %-7s %9d  p1 %.3g p50 %.3g p99 %.3g   <1%%:%d <10%%:%d of %d"
                  % (layer, SLOTN[rec["id"]], rows,
                     pct(mags, 0.01), pct(mags, 0.50), pct(mags, 0.99),
                     tiny, small, len(mags)))

    print()
    print("A prunable tail would show p1 orders of magnitude below p50.")
    print("If p1 and p50 are within a small factor, every row carries comparable")
    print("weight and there is nothing to skip - which closes the last lever.")


if __name__ == "__main__":
    main()
