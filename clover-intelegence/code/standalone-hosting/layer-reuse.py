#!/usr/bin/env python3
"""Are the same experts chosen at different layers, and does the block period show up?

THE CLAIM BEING TESTED
That expert selection is reused across layers because of the sawtooth block structure,
and is therefore in some sense constant.

A caveat that has to be stated before the numbers, because it decides what they are
worth: expert e at layer L and expert e at layer L' are DIFFERENT TENSORS. The dataset
is root-1 .. root-92, each a separate 14.6 GB file of 896 experts. So cross-layer
agreement in expert INDEX cannot save a single byte of I/O -- the weights behind the
same index differ per layer. What it would show is structure in the router, not a
reuse opportunity.

Measured at one position, over all layer pairs:
  - overlap between layer L and L+1          (adjacent)
  - overlap between layer L and L+12         (same phase, next block -- the sawtooth)
  - overlap between layer L and L+4          (same phase in the KKKM cycle)
  - overlap between all random layer pairs   (the null: 16 of 896 twice over)

The null matters. Two independent draws of 16 from 896 overlap by 16*16/896 = 0.29 on
average, so anything near 0.29 is chance.

usage: layer-reuse.py ROUTE.out POSITION
"""
import json
import re
import sys
from collections import defaultdict


def main():
    path = sys.argv[1]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 9

    sel = {}
    for line in open(path):
        m = re.match(r"^ROUTE (\{.*\})\s*$", line)
        if m:
            r = json.loads(m.group(1))
            if r["position"] == want:
                sel[r["layer"]] = set(r["experts"])
    layers = sorted(sel)
    if not layers:
        print("no records at position %d" % want)
        return
    k = len(sel[layers[0]])
    print("position %d, %d layers, k=%d" % (want, len(layers), k))
    print("chance overlap for two independent draws: %.3f" % (k * k / 896.0))

    buckets = defaultdict(list)
    for i, a in enumerate(layers):
        for b in layers[i + 1:]:
            buckets[b - a].append(len(sel[a] & sel[b]))

    print()
    print("%8s %6s %9s %9s %s" % ("gap", "pairs", "mean", "max", ""))
    for gap in (1, 2, 3, 4, 8, 12, 24, 48):
        v = buckets.get(gap)
        if not v:
            continue
        note = {1: "adjacent", 4: "KKKM period", 12: "block period"}.get(gap, "")
        print("%8d %6d %9.3f %9d  %s" % (gap, len(v), sum(v) / float(len(v)), max(v), note))

    allv = [x for v in buckets.values() for x in v]
    print("%8s %6d %9.3f %9d  %s" % ("any", len(allv), sum(allv) / float(len(allv)),
                                     max(allv), "all pairs = the measured null"))

    # Does any single pair of layers agree far more than chance?
    best = []
    for i, a in enumerate(layers):
        for b in layers[i + 1:]:
            n = len(sel[a] & sel[b])
            if n >= 4:
                best.append((n, a, b))
    best.sort(reverse=True)
    print()
    if best:
        print("layer pairs sharing 4 or more of %d: %d pairs" % (k, len(best)))
        for n, a, b in best[:10]:
            print("  layers %2d and %2d share %d  (gap %d)" % (a, b, n, b - a))
    else:
        print("no layer pair shares 4 or more of %d experts" % k)


if __name__ == "__main__":
    main()
