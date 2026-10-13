#!/usr/bin/env python3
"""Walk the stage machine as one sequence, not as ninety-two layers.

Every analysis so far sampled five stages per layer and grouped the results. The engine
does not have ninety-two steps, it has 11,280: stages 1..120 are the server, 121..11160
are layers 1..92 at 120 stages each, 11161..11280 are the tail. The residual is recorded
after every one of them.

Global stage for pod L, local stage s, is 120 + (L-1)*120 + s.

Emits one line per recorded stage, in order, with the residual norm and how far it moved
from the previous stage. No grouping, no averaging.

usage: stage-walk.py <dump-dir> <prefix> <position> [first-pod] [last-pod]
"""
import math
import struct
import sys

if len(sys.argv) < 4:
    sys.exit("usage: stage-walk.py <dump-dir> <prefix> <position> [first-pod] [last-pod]")

directory, prefix, position = sys.argv[1], sys.argv[2], int(sys.argv[3])
first = int(sys.argv[4]) if len(sys.argv) > 4 else 1
last = int(sys.argv[5]) if len(sys.argv) > 5 else 92


def records(path, want_position):
    """local -> vector, for one position."""
    found = {}
    try:
        handle = open(path, "rb")
    except OSError:
        return found
    with handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                break
            _owner, local, pos, count = struct.unpack("<4i", header)
            if count < 0 or count > 65536:
                break
            payload = handle.read(count * 4)
            if len(payload) < count * 4:
                break
            if pos == want_position and local not in found:
                found[local] = struct.unpack("<%df" % count, payload)
    return found


def norm(vector):
    return math.sqrt(sum(float(v) * v for v in vector))


previous = None
print("%8s %5s %5s %12s %12s %10s" % ("stage", "pod", "local", "norm", "moved", "cos_prev"))
for pod in range(first, last + 1):
    found = records("%s/%s.%d.bin" % (directory, prefix, pod), position)
    if not found:
        continue
    for local in sorted(k for k in found if 0 <= k <= 120):
        vector = found[local]
        if len(vector) != 7168:
            continue
        moved = cosine = float("nan")
        if previous is not None:
            moved = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, previous)))
            na, nb = norm(vector), norm(previous)
            if na > 0 and nb > 0:
                cosine = sum(a * b for a, b in zip(vector, previous)) / (na * nb)
        stage = 120 + (pod - 1) * 120 + local
        print("%8d %5d %5d %12.4f %12.4f %10.6f" % (stage, pod, local, norm(vector), moved, cosine))
        previous = vector
