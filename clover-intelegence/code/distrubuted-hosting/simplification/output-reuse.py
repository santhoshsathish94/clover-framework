#!/usr/bin/env python3
"""Is a transformation's OUTPUT predictable from the same transformation one position back?

The previous measurement showed no stage is near the identity, so "predict by doing
nothing" fails. This asks the next cheapest predictor: reuse what the same stage
produced for the previous position.

This is not hypothetical. The engine already carries the machinery -- generation.json
sets expert_result_cache_mib 256, and CACHE_JSON reports result_hits and result_misses.
On the standalone 8-token run those were 0 hits and 17,664 misses, so the mechanism
exists and is currently returning nothing. Whether that is because the outputs genuinely
differ, or because the cache is keyed on something too strict, is exactly what the
vectors can settle.

Measured per pod, between consecutive positions:
  5101  routed path output      the 16 experts' weighted sum
  5102  shared path output      the always-on expert
  116   residual after the MoE add
  9/6   attention output

cos near 1 with small rel means the previous position's result would have served.

usage: output-reuse.py DUMP-PREFIX POS_A POS_B
"""
import glob
import math
import struct
import sys

MLA_POD = lambda p: p % 4 == 3 or p == 92
WANT = (5101, 5102, 116, 9, 6, 20)


def records(path, positions):
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
            if pos in out and local in WANT and local not in out[pos]:
                out[pos][local] = struct.unpack("<%df" % count, body)
    return out


def compare(a, b):
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(x * x for x in b))
    if na == 0 or nb == 0:
        return None
    dot = math.fsum(x * y for x, y in zip(a, b))
    diff = math.sqrt(math.fsum((x - y) * (x - y) for x, y in zip(a, b)))
    return dot / (na * nb), diff / na


def main():
    prefix, pa, pb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    files = {int(p.rsplit(".", 2)[-2]): p for p in glob.glob(prefix + ".*.bin")}
    pods = [p for p in sorted(files) if 1 <= p <= 92]

    labels = {5101: "routed out", 5102: "shared out", 116: "resid after MoE",
              20: "resid after att fold", 9: "KDA attention", 6: "MLA attention"}
    agg = {k: [] for k in labels}
    for pod in pods:
        r = records(files[pod], (pa, pb))
        for local in labels:
            a, b = r[pa].get(local), r[pb].get(local)
            if a and b and len(a) == len(b):
                got = compare(a, b)
                if got:
                    agg[local].append(got)

    print("position %d against %d, %d pods" % (pa, pb, len(pods)))
    print("%-22s %4s %9s %9s %9s %9s" % ("", "n", "cos med", "cos min", "cos max", "rel med"))
    for local, label in sorted(labels.items(), key=lambda kv: -len(agg[kv[0]])):
        v = agg[local]
        if not v:
            continue
        cs = sorted(x[0] for x in v)
        rs = sorted(x[1] for x in v)
        print("%-22s %4d %9.4f %9.4f %9.4f %9.4f"
              % (label, len(v), cs[len(cs) // 2], cs[0], cs[-1], rs[len(rs) // 2]))

    # The medians say "not reusable in general"; the maxima say some pods are nearly
    # exact. Which pods those are, and whether they are the same pods on another
    # prompt, is the part that could matter, so list them rather than summarising.
    if len(sys.argv) > 4:
        want = int(sys.argv[4])
        rows = []
        for pod in pods:
            r = records(files[pod], (pa, pb))
            a, b = r[pa].get(want), r[pb].get(want)
            if a and b and len(a) == len(b):
                got = compare(a, b)
                if got:
                    rows.append((got[0], got[1], pod))
        rows.sort(reverse=True)
        print()
        print("local %d (%s), pods ranked by cosine between the two positions"
              % (want, labels.get(want, "?")))
        for c, rel, pod in rows[:16]:
            print("  pod %2d  %-3s  cos %7.4f  rel %7.4f"
                  % (pod, "MLA" if MLA_POD(pod) else "KDA", c, rel))
        hi = [p for c, _, p in rows if c > 0.95]
        print("  pods above cos 0.95: %s" % (" ".join(map(str, sorted(hi))) if hi else "none"))


if __name__ == "__main__":
    main()
