#!/usr/bin/env python3
"""Read eqidx.bin and report how one layer's expert bytes sit on disk.

Nothing is designed from this until it is read. The question the store layout
depends on is whether an expert's six ranges are contiguous, and whether a
layer's experts are contiguous, because that decides whether a row should be
one range, one expert, or something else.

usage: probe_experts.py <index> [layer]
"""
import collections
import os
import struct
import sys


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("K3_INDEX")
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if not path:
        sys.exit("usage: probe_experts.py <index> [layer]")
    b = open(path, "rb").read()
    if b[:4] != b"K3EQ":
        sys.exit("bad index magic")
    c = 4
    nlay, nslot = struct.unpack_from("<ii", b, c); c += 8
    c += nlay * nslot * (4 + 4 + 8 + 8 + 4 + 4)
    nmodel, = struct.unpack_from("<i", b, c); c += 4
    c += nmodel * (4 + 8 + 8 + 4 + 4 + 4)
    nfiles, = struct.unpack_from("<i", b, c); c += 4
    files = []
    for _ in range(nfiles):
        ln, = struct.unpack_from("<i", b, c); c += 4
        files.append(b[c:c + ln].decode()); c += ln
    nerec, = struct.unpack_from("<i", b, c); c += 4
    print("index: %d layers x %d slots, %d model tensors, %d files, %d expert records"
          % (nlay, nslot, nmodel, nfiles, nerec))

    rec = []
    perlayer = collections.defaultdict(int)
    for _ in range(nerec):
        L, e, which, kind, fid = struct.unpack_from("<iiiii", b, c); c += 20
        off, nb = struct.unpack_from("<qq", b, c); c += 16
        d0, d1 = struct.unpack_from("<ii", b, c); c += 8
        perlayer[L] += nb
        if L == want:
            rec.append((e, which, kind, fid, off, nb, d0, d1))

    print("layers with expert records : %d  (min %d, max %d)"
          % (len(perlayer), min(perlayer), max(perlayer)))
    print("bytes per layer            : %.2f GB, identical across layers: %s"
          % (perlayer[want] / 1e9, len(set(perlayer.values())) == 1))
    print()

    rec.sort()
    print("--- layer %d: the six records of expert 0 ---" % want)
    print("  %-6s %-5s %-5s %14s %12s  %s" % ("which", "kind", "file", "off", "nbytes", "shape"))
    for e, which, kind, fid, off, nb, d0, d1 in rec[:6]:
        print("  %-6d %-5d %-5d %14d %12d  %dx%d" % (which, kind, fid, off, nb, d0, d1))

    six = [r for r in rec if r[0] == 0]
    span = sum(r[5] for r in six)
    lo = min(r[4] for r in six)
    hi = max(r[4] + r[5] for r in six)
    print("  sum of the six  : %d B" % span)
    print("  first..last     : %d B" % (hi - lo))
    print("  contiguous      : %s" % (span == hi - lo and len(set(r[3] for r in six)) == 1))
    print()

    # per expert: are its six ranges one run, and do experts follow each other
    byexp = collections.defaultdict(list)
    for r in rec:
        byexp[r[0]].append(r)
    contig_exp = 0
    sizes = set()
    for e, rs in byexp.items():
        s = sum(x[5] for x in rs)
        sizes.add(s)
        if len(set(x[3] for x in rs)) == 1:
            lo = min(x[4] for x in rs); hi = max(x[4] + x[5] for x in rs)
            if hi - lo == s:
                contig_exp += 1
    print("experts in layer %d          : %d" % (want, len(byexp)))
    print("whose six ranges are one run: %d" % contig_exp)
    print("distinct per-expert sizes   : %s" % sorted(sizes))
    print("files this layer touches    : %d" % len(set(r[3] for r in rec)))

    runs = sorted((r[3], r[4], r[5]) for r in rec)
    merged = 0
    prev = None
    for fid, off, nb in runs:
        if prev and prev[0] == fid and prev[1] + prev[2] == off:
            prev = (fid, prev[1], prev[2] + nb)
        else:
            if prev:
                merged += 1
            prev = (fid, off, nb)
    if prev:
        merged += 1
    print("maximal contiguous runs     : %d (of %d records)" % (merged, len(rec)))


main()
