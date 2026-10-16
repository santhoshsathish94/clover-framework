#!/usr/bin/env python3
"""Which stages change the carried state, for a generated token?

Every pod writes a record at every stage it executes, so the number of stages that
actually move the residual can be counted rather than inferred. The earlier walk found
exactly four per pod at a PROMPT position -- input copy, one attention stage, the
attention fold, and the MoE add -- with all 80 expert stages leaving the residual
bit-identical because they write into separate buffers. This asks the same of a token
the model produced itself.

A record is "changed" when it differs from the previous record at the same position by
any bit. Locals 0, 1 and 120 are bookkeeping: 0 is written before the stage loop, 1 is
the previous position's leftover, and 120 is labelled one position ahead of its
contents. They are counted separately rather than silently included, which is the error
that produced a wrong "636 of 9,956 stages move the residual" three times.

usage: stage-changes.py DUMP-PREFIX POSITION [POSITION ...]
"""
import glob
import struct
import sys
from collections import Counter

WIDTH = 7168
BOOKKEEPING = {0, 1, 120}


def pod_records(path, position):
    seq = []
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return seq
            _, local, pos, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return seq
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                return seq
            if pos == position and count == WIDTH:
                seq.append((local, body))
    return seq


def main():
    prefix = sys.argv[1]
    files = {int(p.rsplit(".", 2)[-2]): p for p in glob.glob(prefix + ".*.bin")}
    pods = [p for p in sorted(files) if 1 <= p <= 92]

    for position in [int(a) for a in sys.argv[2:]]:
        changed = Counter()
        total_changed = total_same = total_book = 0
        per_pod = []
        for pod in pods:
            seq = pod_records(files[pod], position)
            if not seq:
                continue
            n_ch = 0
            prev = None
            for local, body in seq:
                if local in BOOKKEEPING:
                    total_book += 1
                    prev = body
                    continue
                if prev is not None and body != prev:
                    changed[local] += 1
                    n_ch += 1
                    total_changed += 1
                elif prev is not None:
                    total_same += 1
                prev = body
            per_pod.append((pod, len(seq), n_ch))

        print("position %d" % position)
        print("  records %d   changed %d   bit-identical %d   bookkeeping %d"
              % (total_changed + total_same + total_book, total_changed, total_same, total_book))
        counts = Counter(n for _, _, n in per_pod)
        print("  stages that change state, per pod: %s"
              % ", ".join("%d pods change %d" % (v, k) for k, v in sorted(counts.items())))
        print("  which locals change, and in how many pods:")
        for local, n in sorted(changed.items()):
            print("      local %3d  %3d pods" % (local, n))
        print()


if __name__ == "__main__":
    main()
