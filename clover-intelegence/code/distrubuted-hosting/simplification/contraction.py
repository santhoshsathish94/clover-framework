#!/usr/bin/env python3
"""Does a pod amplify or contract the difference between positions?

THE QUESTION
The trunk weights are identical for every token, so why is any stage's output not
predictable? Because output = f(constant weights, input), and the input differs. That
only pushes the question back: a stage fed slightly different inputs could return
almost the same output, or wildly different ones. Which it does is a property of the
stage and is measurable.

For each pod, taking two positions:
  in   cos between the pod's INPUT at the two positions   (local 0)
  out  cos between the pod's OUTPUT at the two positions  (local 119)

If out > in the pod CONTRACTS -- it pulls two different states closer together, and in
the limit its output stops depending on the input at all, which is what "predictable"
means. If out < in it AMPLIFIES, and prediction gets harder with depth.

Pod 12 is the reason this is worth measuring: its output matches across positions at
cos 0.9999 while the stack around it sits near 0.6. Either its input is already
identical, in which case the interesting event happened earlier, or it is a contraction
and the pod itself is doing it.

usage: contraction.py DUMP-PREFIX POS_A POS_B
"""
import glob
import math
import struct
import sys

MLA_POD = lambda p: p % 4 == 3 or p == 92


def grab(path, positions, wanted):
    out = {p: {} for p in positions}
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
            if pos in out and local in wanted and local not in out[pos]:
                out[pos][local] = struct.unpack("<%df" % count, body)
    return out


def cos(a, b):
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(x * x for x in b))
    if na == 0 or nb == 0:
        return float("nan")
    return math.fsum(x * y for x, y in zip(a, b)) / (na * nb)


def main():
    prefix, pa, pb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    files = {int(p.rsplit(".", 2)[-2]): p for p in glob.glob(prefix + ".*.bin")}
    pods = [p for p in sorted(files) if 1 <= p <= 92]

    print("positions %d vs %d" % (pa, pb))
    print("%4s %4s %9s %9s %9s   %s" % ("pod", "kind", "in cos", "out cos", "change", ""))
    rows = []
    for pod in pods:
        r = grab(files[pod], (pa, pb), (0, 119))
        a0, b0 = r[pa].get(0), r[pb].get(0)
        a1, b1 = r[pa].get(119), r[pb].get(119)
        if not (a0 and b0 and a1 and b1):
            continue
        ci, co = cos(a0, b0), cos(a1, b1)
        rows.append((pod, ci, co))

    for pod, ci, co in rows:
        mark = "CONTRACTS" if co > ci + 0.02 else ("amplifies" if co < ci - 0.02 else "")
        print("%4d %4s %9.4f %9.4f %+9.4f   %s"
              % (pod, "MLA" if MLA_POD(pod) else "KDA", ci, co, co - ci, mark))

    if rows:
        contract = [r for r in rows if r[2] > r[1] + 0.02]
        amplify = [r for r in rows if r[2] < r[1] - 0.02]
        print()
        print("pods that contract: %d   amplify: %d   flat: %d"
              % (len(contract), len(amplify), len(rows) - len(contract) - len(amplify)))
        if contract:
            best = sorted(contract, key=lambda r: r[2] - r[1], reverse=True)[:8]
            print("strongest contractions: %s"
                  % ", ".join("pod %d (%+.3f -> %.4f)" % (p, c - i, c) for p, i, c in best))


if __name__ == "__main__":
    main()
