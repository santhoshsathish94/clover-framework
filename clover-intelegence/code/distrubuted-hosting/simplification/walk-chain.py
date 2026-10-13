#!/usr/bin/env python3
"""Walk the whole chain one step at a time and report what each step did to the state.

Direction this was written to: observe the flow without using the expected output token
to judge or interpret anything. There is no comparison to an answer in this file. What is
reported per step is what the step did to the vector it was handed — norm, distance
moved, relative distance, direction agreement. Whether a step matters is not a property
of the step, and this file does not express an opinion about it.

WHAT A RECORD IS, read off transformer_process rather than assumed:

    clover_vectors_write(LAYER, 0, positions, input, WIDTH);   <- before the loop
    while (stage) { executed = stage; step(); write(LAYER, executed, positions, residual); }

so local 0 is the pod's INPUT, written outside the loop and not a stage at all, and
local N>=1 is the residual AFTER stage N ran. Three consequences that look like movement
and are not:

  local 1    stage 1 copies input into `incoming`, never into `residual`, so this record
             is the residual left over from the previous position. It is 0.0000 at
             position 0 because the sequence starts zeroed.
  local 2    stage 2 copies input into `residual`. This is where the position's input
             actually enters the carried state.
  local 120  stage 120 does positions++ before the record is written, so the label is one
             position ahead of the vector. Same value as local 119.

An earlier count of mine said "636 stages change the residual". That counted locals 0, 1
and 120 as work. They are bookkeeping. This walker separates the two.

usage: walk-chain.py DUMP-PREFIX POSITION [--full OUT.tsv]
"""
import glob
import math
import struct
import sys

SERVER_STEPS = [(5201, "token embedding arrives"), (5202, "normalized"),
                (5203, "residual after attention"), (5204, "aggregate, two-source fold"),
                (5205, "dense output"), (5206, "residual out")]

BOOKKEEPING = {0: "the pod's input, written outside the stage loop",
               1: "previous position's leftover; stage 1 writes `incoming`, not `residual`",
               120: "position counter already incremented; same vector as local 119"}

CARRIED = {2: "input copied into the residual",
           6: "MLA attention output",
           9: "KDA attention output",
           20: "attention residual folded in",
           116: "residual += routed + shared"}

NOT_RESIDUAL = {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
                2003, 2021, 3000, 4003, 4021,
                5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009,
                5101, 5102, 5103}


def records(path, position=None):
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return
            owner, local, pos, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return
            if position is not None and pos != position:
                handle.seek(4 * count, 1)
                continue
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                return
            yield owner, local, pos, struct.unpack("<%df" % count, body)


def norm(values):
    return math.sqrt(math.fsum(float(v) * float(v) for v in values))


def compare(before, after):
    moved = math.sqrt(math.fsum((float(a) - float(b)) ** 2 for a, b in zip(before, after)))
    dot = math.fsum(float(a) * float(b) for a, b in zip(before, after))
    na, nb = norm(before), norm(after)
    cos = dot / (na * nb) if na > 0 and nb > 0 else float("nan")
    return moved, cos


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip())
        return 1
    prefix, position = sys.argv[1], int(sys.argv[2])
    full = None
    if "--full" in sys.argv:
        full = open(sys.argv[sys.argv.index("--full") + 1], "w")
        full.write("owner\tlocal\tglobal\tkind\tnorm\tmoved\trel\tcos\n")

    files = {int(p.rsplit(".", 2)[-2]): p for p in glob.glob(prefix + ".*.bin")}

    print("== step 1, token id to vector ==")
    server = {local: values for _, local, _, values in records(files[0], position)}
    print("   embedding lookup, one row of a 163840 x 7168 table, norm %.4f\n"
          % norm(server[5201]))

    print("== step 2, layer 0, owner 0, a flat sequence of six steps ==")
    print("   %-5s %-28s %10s %10s %10s %11s" % ("local", "", "norm", "moved", "moved/prev", "cos"))
    carried = None
    for local, label in SERVER_STEPS:
        values = server.get(local)
        if values is None:
            continue
        if carried is None:
            print("   %-5d %-28s %10.4f %10s %10s %11s" % (local, label, norm(values), "-", "-", "-"))
        else:
            moved, cos = compare(carried, values)
            print("   %-5d %-28s %10.4f %10.4f %10.4f %11.6f"
                  % (local, label, norm(values), moved, moved / max(norm(carried), 1e-30), cos))
        carried = values

    scores = server.get(5207)
    if scores:
        top = max(scores)
        weights = [math.exp(s - top) for s in scores]
        total = sum(weights)
        print("   fold scores %s -> weights %s"
              % ("/".join("%.4f" % s for s in scores),
                 "/".join("%.6f" % (w / total) for w in weights)))

    print("\n== step 3, layers 1 to 92 ==")
    counts = {"carried": 0, "bookkeeping": 0, "identical": 0}
    per_local = {}
    first_pod_detail = []

    for pod in range(1, 93):
        rows = [(local, values) for _, local, _, values in records(files[pod], position)
                if local not in NOT_RESIDUAL]
        rows.sort(key=lambda r: r[0])
        for local, values in rows:
            moved, cos = compare(carried, values)
            kind = ("bookkeeping" if local in BOOKKEEPING
                    else "carried" if moved != 0.0 else "identical")
            counts[kind] += 1
            if kind == "carried":
                per_local.setdefault(local, []).append(
                    (moved, moved / max(norm(carried), 1e-30), cos))
            if pod == 1:
                first_pod_detail.append((local, norm(values), moved, cos, kind))
            if full:
                full.write("%d\t%d\t%d\t%s\t%.6f\t%.6f\t%.6f\t%.9f\n"
                           % (pod, local, 120 + (pod - 1) * 120 + local, kind,
                              norm(values), moved, moved / max(norm(carried), 1e-30), cos))
            carried = values

    print("   every record of pod 1, so the shape of one layer is visible in full:")
    print("   %-6s %10s %10s %11s  %s" % ("local", "norm", "moved", "cos", "kind"))
    for local, size, moved, cos, kind in first_pod_detail:
        note = BOOKKEEPING.get(local) or CARRIED.get(local, "")
        print("   %-6d %10.4f %10.4f %11.6f  %-11s %s" % (local, size, moved, cos, kind, note))

    print("\n   across all 92 pods at this position:")
    print("     %4d records changed the carried residual" % counts["carried"])
    print("     %4d records left it bit-identical" % counts["identical"])
    print("     %4d records are bookkeeping, not stages (locals 0, 1, 120)" % counts["bookkeeping"])
    print("\n   the stages that changed it, by local:")
    print("     %-6s %5s %10s %10s %10s %11s" % ("local", "count", "moved med", "rel med", "rel max", "cos med"))
    for local in sorted(per_local):
        rows = per_local[local]
        mid = len(rows) // 2
        moved_s = sorted(r[0] for r in rows)
        rel_s = sorted(r[1] for r in rows)
        cos_s = sorted(r[2] for r in rows)
        print("     %-6d %5d %10.4f %10.4f %10.4f %11.6f   %s"
              % (local, len(rows), moved_s[mid], rel_s[mid], rel_s[-1], cos_s[mid],
                 CARRIED.get(local, "")))

    print("\n== steps 4 and 5, the tail ==")
    tail = {local: values for _, local, _, values in records(files[93])}
    if tail:
        print("   %-6s %-28s %10s" % ("local", "", "norm"))
        for local, label in ((5310, "snapshot slot 0"), (5311, "slot 1"), (5312, "slot 2"),
                             (5313, "slot 3"), (5314, "slot 4"), (5315, "slot 5"),
                             (5316, "slot 6"), (5317, "slot 7"), (5301, "residual arriving"),
                             (5304, "after the nine-source fold"), (5305, "after the RMS gain")):
            if local in tail:
                print("   %-6d %-28s %10.4f" % (local, label, norm(tail[local])))
        if 5303 in tail:
            print("   fold weights %s" % "/".join("%.4f" % w for w in tail[5303]))

    if full:
        full.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
