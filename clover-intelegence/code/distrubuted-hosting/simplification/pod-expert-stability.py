#!/usr/bin/env python3
"""Does a pod choose the same experts for every token?

Pod 12's residual after the attention fold is the same vector for every position
measured so far -- cos 0.9993 to 0.9998 across generated tokens, and 0.9994 even six
tokens apart. If its MoE input is constant then its router should see a constant input
and choose a constant set of experts, and the whole pod could be computed once per
sequence instead of once per token.

That is a prediction with an obvious failure mode, which is why it is worth checking
rather than assuming: the router reads the POST-NORM vector, not the raw residual, and
a near-constant direction can still produce different top-16 choices if the scores are
close together. Expert ids are integers, so this compares sets, not cosines.

Prints the chosen set per position for the named pods, and how many of the 16 are
common to every position.

usage: pod-expert-stability.py DUMP-PREFIX POD [POD ...]
"""
import glob
import struct
import sys

LOCAL_IDS = 5103


def ids_by_position(path):
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
            if local == LOCAL_IDS and pos not in out:
                out[pos] = [int(round(v)) for v in struct.unpack("<%df" % count, body)]
    return out


def main():
    prefix = sys.argv[1]
    pods = [int(a) for a in sys.argv[2:]] or [12]
    files = {int(p.rsplit(".", 2)[-2]): p for p in glob.glob(prefix + ".*.bin")}

    for pod in pods:
        if pod not in files:
            print("pod %d: no dump" % pod)
            continue
        per = ids_by_position(files[pod])
        if not per:
            print("pod %d: no local %d records" % (pod, LOCAL_IDS))
            continue
        positions = sorted(per)
        sets = [set(per[p]) for p in positions]
        common = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        print("pod %d, %d positions, k=%d" % (pod, len(positions), len(per[positions[0]])))
        for p in positions:
            print("  pos %2d  %s" % (p, " ".join("%3d" % e for e in sorted(per[p]))))
        print("  common to EVERY position : %d of %d  %s"
              % (len(common), len(per[positions[0]]), sorted(common)))
        print("  union across positions   : %d" % len(union))
        adj = [len(sets[i] & sets[i + 1]) for i in range(len(sets) - 1)]
        if adj:
            print("  adjacent-position overlap: %s  (mean %.2f)"
                  % (" ".join(map(str, adj)), sum(adj) / float(len(adj))))
        print()


if __name__ == "__main__":
    main()
