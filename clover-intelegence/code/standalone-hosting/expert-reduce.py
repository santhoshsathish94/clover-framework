#!/usr/bin/env python3
"""Can an expert's transformation be done with less than the whole matrix?

An expert is  h = silu(gate(x)) * up(x)  then  y = down(h).  The down step is a
sum over h's 3072 entries, each selecting one column of down. Any entry of h
that is zero contributes nothing, so its column is never needed. The question is
how much of h actually carries the output, and whether the entries that carry it
are the same ones every time.

  concentrated + stable pattern -> the stored matrix can be pruned once
  concentrated + unstable       -> columns must be fetched per input
  not concentrated              -> nothing to exploit, stop here

Reads the recorded observations only. Changes nothing.
"""
import sys, struct, math, os

I_, LAT = 3072, 3584
REC = 4 + 3 * I_ * 4 + LAT * 4          # identity, gate, up, activation, down
assert REC == 51204

def records(path):
    data = open(path, "rb").read()
    n = len(data) // REC
    for r in range(n):
        b = data[r * REC:(r + 1) * REC]
        ident = struct.unpack_from("<I", b, 0)[0]
        gate = struct.unpack_from("<%df" % I_, b, 4)
        up = struct.unpack_from("<%df" % I_, b, 4 + I_ * 4)
        act = struct.unpack_from("<%df" % I_, b, 4 + 2 * I_ * 4)
        down = struct.unpack_from("<%df" % LAT, b, 4 + 3 * I_ * 4)
        yield ident, gate, up, act, down

def silu(x):
    return x / (1.0 + math.exp(-x)) if x > -60 else 0.0

root = sys.argv[1] if len(sys.argv) > 1 else \
    "/opt/clover-k3/clover-intelegence/dataset"
which = sys.argv[2] if len(sys.argv) > 2 else "france"
layers = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["1", "45", "90"])]

for layer in layers:
    path = "%s/root-%d/observations/%s/results.bin" % (root, layer, which)
    if not os.path.exists(path):
        print("missing %s" % path); continue
    rows = list(records(path))
    print("=== layer %d: %d records, experts %s..." % (
        layer, len(rows), sorted({r[0] for r in rows})[:6]))

    # 1. is the recorded activation really silu(gate)*up?
    ident, gate, up, act, down = rows[0]
    err = max(abs(silu(gate[j]) * up[j] - act[j]) for j in range(0, I_, 7))
    scale = max(abs(a) for a in act) or 1.0
    print("  activation == silu(gate)*up ?  max abs err %.3e  (peak |act| %.3f)" % (err, scale))

    # 2. how concentrated is the activation
    for ident, gate, up, act, down in rows[:3]:
        mag = sorted((abs(a) for a in act), reverse=True)
        tot = sum(m * m for m in mag)
        acc, marks = 0.0, {}
        for i, m in enumerate(mag, 1):
            acc += m * m
            for k in (128, 256, 512, 768, 1024, 1536):
                if i == k:
                    marks[k] = acc / tot
        zeros = sum(1 for a in act if abs(a) < 1e-6)
        print("  expert %-4d zeros %4d/%d   L2 mass in top-k: %s" % (
            ident, zeros, I_,
            "  ".join("%d:%.3f" % (k, marks[k]) for k in sorted(marks))))

    # 3. is the carrying set the same across inputs for one expert?
    byexp = {}
    for ident, gate, up, act, down in rows:
        byexp.setdefault(ident, []).append(act)
    for ident, acts in list(byexp.items())[:3]:
        if len(acts) < 2:
            continue
        K = 512
        tops = [set(sorted(range(I_), key=lambda j: -abs(a[j]))[:K]) for a in acts[:6]]
        pair = [len(tops[i] & tops[i + 1]) / float(K) for i in range(len(tops) - 1)]
        union = set().union(*tops)
        print("  expert %-4d top-%d overlap between inputs %.1f%%   union over %d inputs %d" % (
            ident, K, 100 * sum(pair) / len(pair), len(tops), len(union)))
