#!/usr/bin/env python3
"""Observe routing and the intermediate values, per pod, per position.

No comparison to any expected output. What is reported is what the engine recorded:

  5103  the sixteen expert ids this pod chose for this position
  3000  the sixteen router weights on those experts
  5101  the routed path's output vector
  5102  the shared path's output vector
  2003  the two fold scores at stage 3, 2021 the two at stage 21
  4003  the vector stage 3's fold produced, 4021 stage 21's
  1000+ the block snapshots the pod carries

Reported per pod: how concentrated the routing is, how the two MoE paths compare in
size and direction, and what the folds weighted. Reported across pods: how often an
expert is reused, and whether any expert is chosen by every position.

usage: observe-routing.py DUMP-PREFIX [POSITIONS]
"""
import collections
import glob
import math
import struct
import sys


def records(path, wanted):
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return
            owner, local, pos, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return
            if local not in wanted:
                handle.seek(4 * count, 1)
                continue
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                return
            yield local, pos, struct.unpack("<%df" % count, body)


def norm(values):
    return math.sqrt(math.fsum(float(v) * float(v) for v in values))


def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return math.fsum(float(x) * float(y) for x, y in zip(a, b)) / (na * nb)


def effective(weights):
    """Participation ratio: how many of the sixteen the weight actually spreads over.

    1 means one expert holds everything, 16 means perfectly even. It is a description of
    the weight vector, not a judgement about whether the spread is good."""
    total = math.fsum(weights)
    if total <= 0:
        return float("nan")
    share = [w / total for w in weights]
    squares = math.fsum(s * s for s in share)
    return 1.0 / squares if squares > 0 else float("nan")


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 1
    prefix = sys.argv[1]
    positions = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    files = {int(p.rsplit(".", 2)[-2]): p for p in glob.glob(prefix + ".*.bin")}
    wanted = {5103, 3000, 5101, 5102, 2003, 2021, 4003, 4021}

    chosen = {}
    data = {}
    for pod in range(1, 93):
        data[pod] = collections.defaultdict(dict)
        for local, pos, values in records(files[pod], wanted):
            data[pod][pos][local] = values
            if local == 5103:
                chosen[(pod, pos)] = tuple(int(v) for v in values)

    print("== routing, one row per pod, at position 0 and the last position ==")
    print("   %-4s %6s %8s %8s %10s %10s %9s %9s"
          % ("pod", "eff", "top w", "sum w", "routed", "shared", "r/s", "cos(r,s)"))
    for pod in sorted(data):
        for pos in (0, positions - 1):
            frame = data[pod].get(pos, {})
            weights = frame.get(3000)
            routed, shared = frame.get(5101), frame.get(5102)
            if not weights:
                continue
            if pod % 10 and pos:
                continue
            nr = norm(routed) if routed else float("nan")
            ns = norm(shared) if shared else float("nan")
            print("   %-4d %6.2f %8.4f %8.4f %10.4f %10.4f %9.4f %9.5f"
                  % (pod, effective(weights), max(weights), math.fsum(weights),
                     nr, ns, nr / ns if ns else float("nan"),
                     cosine(routed, shared) if routed and shared else float("nan")))

    print("\n== how much the sixteen choices overlap between positions, per pod ==")
    overlaps = []
    for pod in sorted(data):
        sets = [set(chosen[(pod, p)]) for p in range(positions) if (pod, p) in chosen]
        if len(sets) < 2:
            continue
        pairs = [len(a & b) for i, a in enumerate(sets) for b in sets[i + 1:]]
        everywhere = set.intersection(*sets)
        overlaps.append((pod, sum(pairs) / len(pairs), len(everywhere), len(set().union(*sets))))
    if overlaps:
        shared_all = [o[2] for o in overlaps]
        mean_pair = [o[1] for o in overlaps]
        print("   %d pods" % len(overlaps))
        print("   mean pairwise overlap of 16: min %.2f  median %.2f  max %.2f"
              % (min(mean_pair), sorted(mean_pair)[len(mean_pair) // 2], max(mean_pair)))
        print("   experts chosen by EVERY position: min %d  median %d  max %d"
              % (min(shared_all), sorted(shared_all)[len(shared_all) // 2], max(shared_all)))
        print("   distinct experts used across the %d positions: min %d  median %d  max %d"
              % (positions, min(o[3] for o in overlaps),
                 sorted(o[3] for o in overlaps)[len(overlaps) // 2],
                 max(o[3] for o in overlaps)))

    print("\n== the two folds, stage 3 and stage 21, at the last position ==")
    print("   %-4s %18s %10s %18s %10s" % ("pod", "stage 3 weights", "|fold 3|", "stage 21 weights", "|fold 21|"))
    for pod in sorted(data):
        if pod % 10:
            continue
        frame = data[pod].get(positions - 1, {})
        s3, s21 = frame.get(2003), frame.get(2021)
        f3, f21 = frame.get(4003), frame.get(4021)
        if not s3:
            continue

        def soft(scores):
            top = max(scores)
            exps = [math.exp(s - top) for s in scores]
            total = sum(exps)
            return "/".join("%.4f" % (e / total) for e in exps)

        print("   %-4d %18s %10.4f %18s %10.4f"
              % (pod, soft(s3), norm(f3) if f3 else float("nan"),
                 soft(s21) if s21 else "-", norm(f21) if f21 else float("nan")))

    print("\n== the routed and shared paths across all pods, last position ==")
    ratios = []
    coss = []
    for pod in sorted(data):
        frame = data[pod].get(positions - 1, {})
        routed, shared = frame.get(5101), frame.get(5102)
        if routed and shared:
            ns = norm(shared)
            if ns:
                ratios.append((norm(routed) / ns, pod))
                coss.append((cosine(routed, shared), pod))
    if ratios:
        ratios.sort()
        coss.sort()
        print("   routed/shared norm ratio: min %.3f (pod %d)  median %.3f  max %.3f (pod %d)"
              % (ratios[0][0], ratios[0][1], ratios[len(ratios) // 2][0],
                 ratios[-1][0], ratios[-1][1]))
        print("   cos(routed, shared):      min %.4f (pod %d)  median %.4f  max %.4f (pod %d)"
              % (coss[0][0], coss[0][1], coss[len(coss) // 2][0], coss[-1][0], coss[-1][1]))
        print("   pods where the routed path is SMALLER than the shared path: %d of %d"
              % (sum(1 for r, _ in ratios if r < 1.0), len(ratios)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
