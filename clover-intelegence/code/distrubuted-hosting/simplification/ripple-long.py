#!/usr/bin/env python3
"""Follow the vector that left the stream, one pod at a time, and say what is done with it.

At seven pods the carried vector is dropped from the stream and kept somewhere else. This
walks forward from the pod that dropped it and reports, at every pod after that, the size
of the kept vector and the weight the two folds give it.

The snapshot records (locals 1000-1007) are labelled one position ahead, so position p is
read at p+1. That is not assumed here: the tool prints the kept vector's size at every pod
so a frozen value shows as a repeated number, and `--check` compares slot k+1 against the
residual that arrived at pod 12k.

Fold scores (2003 for fold 37, 2021 for fold 38) run slot 0, slot 1, ... then the live
residual last, so the kept vector's weight is `scores[slot]` and is only present when
`slot < len(scores) - 1`. At the pod that drops the vector, fold 37 runs before the push,
so the slot is not yet a source there and the column is blank.

`holds` is weight x size: how much of the fold the kept vector is, in the fold's units.

usage: ripple-long.py DUMP-PREFIX POSITION SLOT
"""
import glob
import os
import re
import struct
import sys

import numpy

WANTED = {0, 2003, 2021, 4003, 4021} | set(range(1000, 1008))


def read(path, position, ahead):
    out = {}
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return out
            _, local, at, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return out
            want = ahead if 1000 <= local <= 1007 else position
            if local not in WANTED or at != want:
                handle.seek(4 * count, 1)
                continue
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                return out
            out[local] = numpy.frombuffer(body, dtype="<f4")


def main():
    prefix, position, slot = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    born = 12 * slot

    print("position %d, slot %d, left the stream at pod %d" % (position, slot, born))
    print()
    print("%-5s %-7s %10s %8s %8s %9s %8s %8s %9s %10s"
          % ("pod", "sources", "its size", "w 37", "holds", "fold 37", "w 38", "holds",
             "fold 38", "residual"))

    for path in sorted(glob.glob(prefix + ".*.bin"),
                       key=lambda p: int(re.search(r"(\d+)\.bin$", p).group(1))):
        pod = int(re.search(r"(\d+)\.bin$", os.path.basename(path)).group(1))
        if not born <= pod <= 92:
            continue
        frame = read(path, position, position + 1)
        kept = frame.get(1000 + slot)
        if kept is None:
            continue
        size = float(numpy.linalg.norm(kept))
        incoming = frame.get(0)
        cells = []
        for scored, produced in ((2003, 4003), (2021, 4021)):
            scores, output = frame.get(scored), frame.get(produced)
            if scores is None or slot >= len(scores) - 1:
                cells += ["%8s" % "-", "%8s" % "-"]
            else:
                cells += ["%8.4f" % scores[slot], "%8.3f" % (scores[slot] * size)]
            cells.append("%9.4f" % float(numpy.linalg.norm(output))
                         if output is not None else "%9s" % "-")
        print("%-5d %-7s %10.4f %s %s %s %s %s %s %10s"
              % (pod,
                 "%d/%d" % (len(frame.get(2003, [])), len(frame.get(2021, []))),
                 size, cells[0], cells[1], cells[2], cells[3], cells[4], cells[5],
                 "%10.4f" % float(numpy.linalg.norm(incoming)) if incoming is not None else "-"))


if __name__ == "__main__":
    main()
