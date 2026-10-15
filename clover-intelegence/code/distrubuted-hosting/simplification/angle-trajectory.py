#!/usr/bin/env python3
"""Angle of each pod's output from the initial embedding, against the answer trajectory.

Tests the third observation -- that the live residual's direction stabilises near 90
degrees from where it started -- on the SAME fresh vectors used for the head projection,
so the two can be read against each other without matching up separate runs.

The question is not whether 90 degrees happens. It is whether the angle settling
coincides with the answer settling. If the direction stabilises long before the answer
does, the angle cannot be a signal that the output is decided, and no early-exit rule
can be built on it.

usage: angle-trajectory.py TRAJ.f32 REFERENCE.f32 [FIRST_POD]
  TRAJ.f32       92 pod output residuals, 7168 floats each, pod order
  REFERENCE.f32  one 7168-float vector, the initial embedding
"""
import math
import struct
import sys

WIDTH = 7168


def load(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    n = len(raw) // (4 * WIDTH)
    return [struct.unpack_from("<%df" % WIDTH, raw, i * 4 * WIDTH) for i in range(n)]


def norm(v):
    return math.sqrt(math.fsum(x * x for x in v))


def angle(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    c = math.fsum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    traj = load(sys.argv[1])
    ref = load(sys.argv[2])[0]
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    print("reference norm %.4f, %d pod vectors" % (norm(ref), len(traj)))
    print("%5s %5s %10s %12s %12s" % ("pod", "kind", "norm", "deg from ref", "deg from prev"))
    prev = None
    for i, v in enumerate(traj):
        pod = first + i
        kind = "MLA" if (pod % 4 == 3 or pod == 92) else "KDA"
        step = angle(prev, v) if prev is not None else float("nan")
        print("%5d %5s %10.4f %12.2f %12.2f" % (pod, kind, norm(v), angle(ref, v), step))
        prev = v


if __name__ == "__main__":
    main()
