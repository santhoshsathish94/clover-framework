#!/usr/bin/env python3
"""Compare a logits dump against the preserved baseline.

Layout, from the writer: E float32 of final norm, then VOCAB float32 of logits.
Mixture mass lost is a proxy; the emitted token and the logit field are the
observables, so report those.
"""
import struct
import sys

E, VOCAB = 7168, 163840


def load(p):
    with open(p, "rb") as f:
        b = f.read()
    if len(b) != (E + VOCAB) * 4:
        raise SystemExit("%s: unexpected size %d" % (p, len(b)))
    a = struct.unpack("<%df" % (E + VOCAB), b)
    return a[E:]


def topn(v, n):
    return sorted(range(len(v)), key=lambda i: -v[i])[:n]


ref = load(sys.argv[1])
got = load(sys.argv[2])

rt, gt = topn(ref, 8), topn(got, 8)
mx = max(abs(a - b) for a, b in zip(ref, got))
num = sum((a - b) ** 2 for a, b in zip(ref, got)) ** 0.5
den = sum(a * a for a in ref) ** 0.5

print("token      : ref %d   got %d   %s" % (rt[0], gt[0], "SAME" if rt[0] == gt[0] else "CHANGED"))
print("top-8      : ref %s" % rt)
print("             got %s" % gt)
print("top-8 kept : %d of 8   in order: %s" % (len(set(rt) & set(gt)), rt == gt))
print("max |diff| : %.6g" % mx)
print("rel L2     : %.4f%%" % (100.0 * num / den))
print("margin ref : %.6g   got %.6g" % (ref[rt[0]] - ref[rt[1]], got[gt[0]] - got[gt[1]]))
