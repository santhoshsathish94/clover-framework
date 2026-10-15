#!/usr/bin/env python3
"""How predictable is each transformation's output from its input? Measured, not judged.

The question is not early exit and not the final token. It is whether ANY transformation
in the stack produces an output that could be obtained more cheaply than by running it.
The cheapest possible predictor is the identity -- output equals input -- so the first
thing to measure is simply how far each transformation moves its input, and in what
direction. A stage with cosine near 1 and small relative movement is close to
predictable by doing nothing. A stage with cosine near 0 is not predictable that way and
needs a different predictor, or none exists.

This reports, per pod, for every state-changing stage the engine records:

  cos      cosine between the vector before the stage and after it
  rel      |after - before| / |before|, how far it moved relative to where it was
  ratio    |after| / |before|

Nothing here is scored against the answer token. These are properties of the
transformations themselves.

Stages, read off the record layout rather than assumed:
  0   -> 2     the pod's input becomes the carried residual
  2   -> 6/9   attention, MLA writes 6 and KDA writes 9
  9/6 -> 20    the attention residual folded in
  20  -> 116   residual += routed + shared, all 80 expert stages land here
  116 -> 119   copy out

usage: transform-predictability.py DUMP-PREFIX POSITION
"""
import glob
import math
import struct
import sys

MLA_POD = lambda p: p % 4 == 3 or p == 92


def records(path, position):
    out = {}
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return out
            _, local, pos, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return out
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                return out
            if pos == position and local not in out:
                out[local] = struct.unpack("<%df" % count, body)
    return out


def norm(v):
    return math.sqrt(math.fsum(x * x for x in v))


def compare(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan"), float("nan")
    dot = math.fsum(x * y for x, y in zip(a, b))
    diff = math.sqrt(math.fsum((x - y) * (x - y) for x, y in zip(a, b)))
    return dot / (na * nb), diff / na, nb / na


def main():
    prefix, position = sys.argv[1], int(sys.argv[2])
    files = {int(p.rsplit(".", 2)[-2]): p for p in glob.glob(prefix + ".*.bin")}
    pods = [p for p in sorted(files) if 1 <= p <= 92]

    print("position %d, %d pods" % (position, len(pods)))
    print("%4s %4s | %-22s | %-22s | %-22s" %
          ("pod", "kind", "attention (2->att)", "att fold (att->20)", "MoE add (20->116)"))
    print("%4s %4s | %6s %6s %6s | %6s %6s %6s | %6s %6s %6s" %
          ("", "", "cos", "rel", "ratio", "cos", "rel", "ratio", "cos", "rel", "ratio"))

    agg = {"att": [], "fold": [], "moe": []}
    for pod in pods:
        r = records(files[pod], position)
        kind = "MLA" if MLA_POD(pod) else "KDA"
        att_local = 6 if MLA_POD(pod) else 9
        cells = []
        for name, before, after in (("att", 2, att_local),
                                    ("fold", att_local, 20),
                                    ("moe", 20, 116)):
            if before in r and after in r and len(r[before]) == len(r[after]):
                c, rel, ratio = compare(r[before], r[after])
                agg[name].append((c, rel, ratio))
                cells.append("%6.3f %6.3f %6.3f" % (c, rel, ratio))
            else:
                cells.append("%6s %6s %6s" % ("-", "-", "-"))
        print("%4d %4s | %s | %s | %s" % (pod, kind, cells[0], cells[1], cells[2]))

    print()
    print("medians across pods")
    for name in ("att", "fold", "moe"):
        v = agg[name]
        if not v:
            continue
        med = lambda i: sorted(x[i] for x in v)[len(v) // 2]
        print("  %-5s n=%2d   cos %7.4f   rel %7.4f   ratio %7.4f"
              % (name, len(v), med(0), med(1), med(2)))


if __name__ == "__main__":
    main()
