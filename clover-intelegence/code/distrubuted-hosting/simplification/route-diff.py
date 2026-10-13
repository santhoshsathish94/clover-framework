#!/usr/bin/env python3
"""Compare expert selection between two runs of the same prompt.

Methods B and C substituted a constant for the attention output. One layer was
tolerated, sixty-eight were not. If the routed experts are where per-token information
lives, a tolerated substitution should leave the selection largely alone and an
intolerable one should scramble it. That is now directly observable rather than
arguable.

Reports, per layer, the mean overlap between the two runs' sixteen picks at matching
positions, and the overlap of the top choice alone.

usage: route-diff.py <baseline.bin> <other.bin>
"""
import struct
import sys


def selections(path):
    """position -> ordered list of sixteen expert ids."""
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
            _owner, local, position, count = struct.unpack("<4i", header)
            if count < 0 or count > 65536:
                break
            payload = handle.read(count * 4)
            if len(payload) < count * 4:
                break
            if local == 5103 and count == 16 and position not in found:
                found[position] = [int(v) for v in struct.unpack("<16f", payload)]
    return found


if len(sys.argv) != 3:
    sys.exit("usage: route-diff.py <baseline.bin> <other.bin>")

left, right = selections(sys.argv[1]), selections(sys.argv[2])
shared = sorted(set(left) & set(right))
if not shared:
    print("NA\tno matching positions")
    sys.exit(1)

overlap_total = 0
top_same = 0
identical = 0
for position in shared:
    a, b = left[position], right[position]
    overlap_total += len(set(a) & set(b))
    if a[0] == b[0]:
        top_same += 1
    if a == b:
        identical += 1

print("%d\t%.2f\t%d\t%d" % (
    len(shared),                      # positions compared
    overlap_total / len(shared),      # mean overlap out of 16
    top_same,                         # positions whose first choice is unchanged
    identical,                        # positions whose whole ordered list is unchanged
))
