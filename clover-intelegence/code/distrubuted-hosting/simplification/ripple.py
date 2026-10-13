#!/usr/bin/env python3
"""Follow one residual through the whole chain, one step at a time, and say where it is.

Every earlier tool in here sampled the chain and aggregated. This one does not sample
and does not aggregate. It takes one prompt, one position, and reports the carried
vector at every point it changes, in order, so the sequence can be read as a sequence.

The carried buffer passes through five recorded states inside a pod:

  in          local 0,      what arrived
  attention   local 9 (KDA) or 6 (MLA), the attention output, which overwrites the buffer
  fold        local 20,     the input added back, or not
  moe         local 116,    routed and shared added
  out         local 119,    what leaves

At each point three things about the vector itself, not about any other vector:

  size        its length
  turn        the angle to the point before it, in degrees
  from start  the angle to the vector that entered the chain

Nothing here is a comparison between two pods, two kinds or two prompts.

usage: ripple.py DUMP-PREFIX POSITION [FIRST-POD] [LAST-POD]
"""
import glob
import math
import os
import re
import struct
import sys

import numpy

MLA_PODS = set(range(3, 92, 4)) | {92}
WANTED = {0, 6, 9, 20, 116, 119}


def read(path, position):
    out = {}
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return out
            _, local, at, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return out
            if local not in WANTED or at != position:
                handle.seek(4 * count, 1)
                continue
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                return out
            out[local] = numpy.frombuffer(body, dtype="<f4").astype("f8")


def angle(a, b):
    size = float(numpy.linalg.norm(a)) * float(numpy.linalg.norm(b))
    if not size:
        return float("nan")
    cosine = float(a @ b) / size
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def main():
    prefix, position = sys.argv[1], int(sys.argv[2])
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    last = int(sys.argv[4]) if len(sys.argv) > 4 else 92

    steps = []
    for path in sorted(glob.glob(prefix + ".*.bin"),
                       key=lambda p: int(re.search(r"(\d+)\.bin$", p).group(1))):
        pod = int(re.search(r"(\d+)\.bin$", os.path.basename(path)).group(1))
        if not 1 <= pod <= 92:
            continue
        frame = read(path, position)
        for local, label in ((0, "in"),
                             (6 if pod in MLA_PODS else 9, "attention"),
                             (20, "fold"), (116, "moe"), (119, "out")):
            if local in frame:
                steps.append((pod, "M" if pod in MLA_PODS else "K", label, frame[local]))

    start = steps[0][3]
    shown = [s for s in steps if first <= s[0] <= last]
    print("prompt position %d, pods %d to %d, %d recorded states of the carried vector"
          % (position, first, last, len(shown)))
    print()
    print("%-5s %-3s %-10s %12s %8s %11s" % ("pod", "", "", "size", "turn", "from start"))
    previous = None
    for pod, kind, label, vector in steps:
        if first <= pod <= last:
            turned = "" if previous is None else "%7.2f" % angle(previous, vector)
            print("%-5d %-3s %-10s %12.4f %8s %10.2f"
                  % (pod, kind, label, float(numpy.linalg.norm(vector)),
                     turned, angle(start, vector)))
        previous = vector


if __name__ == "__main__":
    main()
