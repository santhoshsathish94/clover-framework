#!/usr/bin/env python3
"""The answer trajectory: at which pod does the final token become readable?

THE QUESTION THIS ANSWERS
Do the KKKM pattern, the ~90-degree direction stabilisation, and the persisting
112.8886 snapshot let us predict the output before the stack finishes? A cost breakdown
cannot answer that and I wrongly offered one. This can.

WHAT IT DOES
Pulls local 119 -- the pod's output residual, confirmed bit-identical to the tail's
input for pod 92 -- from every pod dump at one position, and writes them in pod order as
raw float32 for head-walk to project through the real vocabulary head.

WHAT THE PROJECTION IS AND IS NOT
It is a probe, not the model's own path. The engine folds nine sources and applies an
RMS gain before the head, so projecting a bare mid-stack residual is NOT what the model
computes there. What it measures is whether the answer is LINEARLY READABLE at that
depth. That is the right question for early exit: if the final token is already top-1 at
pod N and stays top-1, pods after N are candidates for removal. If it is not, no amount
of structure in the residual will let us stop early.

Already known from THE-FLOW-OBSERVED.md at position 4, where the answer is 17374: none
of the seven stored snapshots selects it, and snapshot 5 selects 113476 with p=0.635,
more confident than the correct residual at 0.510. So the stored vectors do not carry
the answer. This asks the same of every pod in between.

usage: answer-trajectory.py DUMP-PREFIX POSITION OUT.f32
"""
import glob
import struct
import sys

LOCAL_OUT = 119          # the pod's output residual
WIDTH = 7168


def record(path, position, want_local):
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return None
            _, local, pos, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return None
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                return None
            if pos == position and local == want_local:
                return body


def main():
    if len(sys.argv) != 4:
        print(__doc__.strip())
        return 1
    prefix, position, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    files = {int(p.rsplit(".", 2)[-2]): p for p in glob.glob(prefix + ".*.bin")}
    pods = sorted(files)
    if not pods:
        print("no dumps matching %s.*.bin" % prefix)
        return 1

    written, missing = [], []
    with open(out, "wb") as sink:
        for pod in pods:
            body = record(files[pod], position, LOCAL_OUT)
            if body is None or len(body) != 4 * WIDTH:
                missing.append(pod)
                continue
            sink.write(body)
            written.append(pod)

    print("position %d, local %d (pod output residual)" % (position, LOCAL_OUT))
    print("wrote %d vectors of %d floats to %s" % (len(written), WIDTH, out))
    print("pods in order: %s" % " ".join(str(p) for p in written))
    if missing:
        print("MISSING from %d pods: %s" % (len(missing), " ".join(str(p) for p in missing)))
        print("  a gap here shifts every later row, so the pod list above is the key")
    return 0


if __name__ == "__main__":
    sys.exit(main())
