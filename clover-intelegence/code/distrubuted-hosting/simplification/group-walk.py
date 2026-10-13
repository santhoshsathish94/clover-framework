#!/usr/bin/env python3
"""Lay out one KDA-KDA-KDA-MLA group, every record, in the order the engine wrote them.

No comparison to another group, no spread, no ratio of max to min. Choosing a lens is how
you find what the lens was built to find; this prints the sequence and nothing else.

What a KDA pod records inside itself, read off transformer-50.c:

  5001 Q    after short convolution, SiLU and per-head L2
  5002 K    the same
  5003 V    after convolution and SiLU, no L2
  5004 beta per head, after sigmoid
  5005 decay  exp(-5 * sigmoid(.)) per row
  5009 magnitude  Frobenius norm of the recurrent state, per head
  5006 attention  raw, straight out of the state update
  5007 attention  after per-head normalize
  5008 gate  attention * sigmoid(gate projection)

An MLA pod records none of these. Its stage 6 is one call that jumps to stage 20, so
between the input and the attention residual there is nothing to show. That absence is a
property of the instrument, not of the model, and it is printed as a hole rather than
skipped.

usage: group-walk.py DUMP-PREFIX GROUP POSITION
"""
import glob
import math
import struct
import sys

GROUP = 4
MLA_PODS = set(range(3, 92, 4)) | {92}

LABEL = {
    0: "input, written before the stage loop",
    1: "previous position's leftover",
    2: "input copied into the residual",
    3: "aggregate, fold 37",
    4: "snapshot push",
    5: "normalize",
    7: "KDA decay",
    8: "KDA state update",
    9: "KDA attention output",
    20: "attention residual folded in",
    21: "aggregate, fold 38",
    22: "postnorm",
    24: "route, choose 16 of 896",
    25: "project to latent",
    110: "latent norm",
    111: "project routed",
    112: "shared gate",
    113: "shared up",
    114: "shared activation",
    115: "shared output",
    116: "residual += routed + shared",
    118: "copy out",
    119: "snapshots out",
    120: "positions++",
}

INNER = {5001: "Q  conv+SiLU+L2", 5002: "K  conv+SiLU+L2", 5003: "V  conv+SiLU",
         5004: "beta per head", 5005: "decay per row",
         5009: "recurrent state magnitude per head",
         5006: "attention raw", 5007: "attention per-head normalized",
         5008: "gate = attention * sigmoid(gate)",
         5011: "query latent, proj 20 + normalize", 5012: "query, proj 22",
         5013: "kv latent, proj 23 + normalize", 5014: "expanded, proj 25",
         5015: "attention, softmax-weighted sum",
         5016: "gate = attention * sigmoid(gate)"}

OTHER = {2003: "fold 37 scores", 2021: "fold 38 scores",
         4003: "fold 37 output", 4021: "fold 38 output",
         3000: "router weights on the 16", 5103: "the 16 expert ids",
         5101: "routed path output", 5102: "shared path output"}


def records(path, position):
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return
            _, local, pos, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                return
            if pos == position:
                yield local, struct.unpack("<%df" % count, body)


def norm(values):
    return math.sqrt(math.fsum(float(v) * float(v) for v in values))


def main():
    if len(sys.argv) != 4:
        print(__doc__.strip())
        return 1
    prefix, group, position = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    files = {int(p.rsplit(".", 2)[-2]): p for p in glob.glob(prefix + ".*.bin")}
    pods = [p for p in range(group * GROUP, group * GROUP + GROUP) if p in files]

    print("group %d, pods %s, position %d"
          % (group, ",".join(str(p) for p in pods), position))
    print("every record in the order the engine wrote it\n")

    for pod in pods:
        kind = "MLA" if pod in MLA_PODS else "KDA"
        rows = list(records(files[pod], position))
        print("=" * 78)
        print("pod %d   %s   %d records at this position" % (pod, kind, len(rows)))
        print("-" * 78)
        previous = None
        for local, values in rows:
            size = norm(values)
            if local in INNER:
                print("   %-5d %-40s %8d wide  norm %12.4f"
                      % (local, INNER[local], len(values), size))
                continue
            if local in OTHER:
                if len(values) <= 16:
                    shown = " ".join("%.4f" % v for v in values[:8])
                    print("   %-5d %-40s %8d wide  %s%s"
                          % (local, OTHER[local], len(values), shown,
                             " ..." if len(values) > 8 else ""))
                else:
                    print("   %-5d %-40s %8d wide  norm %12.4f"
                          % (local, OTHER[local], len(values), size))
                continue
            if 1000 <= local <= 1007:
                print("   %-5d %-40s %8d wide  norm %12.4f"
                      % (local, "snapshot slot %d" % (local - 1000), len(values), size))
                continue
            if 30 <= local <= 109:
                continue
            moved = ""
            if previous is not None and len(previous) == len(values):
                distance = math.sqrt(math.fsum((float(a) - float(b)) ** 2
                                               for a, b in zip(previous, values)))
                moved = "moved %10.4f" % distance
            name = LABEL.get(local, "")
            if local == 6:
                name = "MLA, whole attention" if pod in MLA_PODS else "KDA qkv"
            print("   %-5d %-40s %8d wide  norm %12.4f  %s"
                  % (local, name, len(values), size, moved))
            previous = values
        carried = [l for l, _ in rows if 3 <= l <= 119 and l not in INNER and l not in OTHER
                   and not (1000 <= l <= 1007)]
        experts = [l for l, _ in rows if 30 <= l <= 109]
        print("   %d stages between 30 and 109 ran and wrote the residual unchanged;" % len(experts))
        print("   they are the 16 experts, 5 stages each, and they write `routed`,")
        print("   not `residual`, so nothing here shows what they did.")
        if kind == "MLA":
            if any(l in (5011, 5015) for l, _ in rows):
                print("   the MLA path has no per-head normalize: there is no counterpart")
                print("   to local 5007. It goes from the weighted sum straight to the gate.")
            else:
                print("   stages 7, 8, 9 do not exist on this pod and stage 6 records nothing")
                print("   inside itself, so the attention is one opaque step.")
        del carried
    return 0


if __name__ == "__main__":
    sys.exit(main())
