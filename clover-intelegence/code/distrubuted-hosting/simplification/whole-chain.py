#!/usr/bin/env python3
"""Walk the whole chain, one step at a time, and say which steps move the state.

The previous walk covered owners 1 to 92 and reported 636 of 9,956 recorded stages
moving the residual. That count was never the whole chain. This one includes the two
ends: the server, which has no stage machine and so has no trace lines at all, and the
tail, which has sixteen.

The trace records a hash of the residual after each stage. A hash cannot say what the
state became, only whether it changed, which is the question here. The server's steps
come from its vector records instead, because that is the only instrument it has.

usage: whole-chain.py TRACE.csv [SERVER-VECTORS.bin] [POSITION]
"""
import collections
import struct
import sys

STAGES_PER_OWNER = 120
SERVER_STEPS = [(5201, "token embedding arrives"), (5202, "normalized"),
                (5203, "residual after attention"), (5204, "aggregate, 2-source fold"),
                (5205, "dense output"), (5206, "residual out")]


def read_trace(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 6:
                continue
            owner, local, position, nanos, digest, detail = parts
            rows.append((int(owner), int(local), int(position), float(nanos), digest, int(detail)))
    return rows


def server_records(path):
    found = {}
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                break
            owner, local, position, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                break
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                break
            found.setdefault(position, {})[local] = struct.unpack("<%df" % count, body)
    return found


def norm(values):
    return sum(float(v) * float(v) for v in values) ** 0.5


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 1
    rows = read_trace(sys.argv[1])
    vectors = server_records(sys.argv[2]) if len(sys.argv) > 2 else {}
    position = int(sys.argv[3]) if len(sys.argv) > 3 else None

    owners = sorted({row[0] for row in rows})
    positions = sorted({row[2] for row in rows if row[0] != 93})
    if position is None:
        position = positions[-1]
    print("trace: %d lines, owners %d..%d, positions %d..%d"
          % (len(rows), owners[0], owners[-1], positions[0], positions[-1]))
    print("the server, owner 0, contributes %d trace lines: it is a flat sequence, not a"
          " stage machine" % sum(1 for row in rows if row[0] == 0))
    print("\nwalking position %d\n" % position)

    print("== step 1, token id to vector ==")
    if position in vectors and 5201 in vectors[position]:
        print("   embedding lookup, one row of a 163840 x 7168 table, norm %.4f"
              % norm(vectors[position][5201]))
    else:
        print("   no server vectors given")

    print("\n== step 2, layer 0 (owner 0, %d steps) ==" % len(SERVER_STEPS))
    previous = None
    moved_server = 0
    for local, label in SERVER_STEPS:
        values = vectors.get(position, {}).get(local)
        if values is None:
            continue
        changed = previous is None or values != previous
        moved_server += 1 if (previous is not None and changed) else 0
        print("   %-5d %-28s norm %9.4f  %s" % (local, label, norm(values),
              "moved" if previous is not None and changed else
              ("start" if previous is None else "same")))
        previous = values

    print("\n== step 3, layers 1 to 92 (owners 1..92) ==")
    walk = [row for row in rows if 1 <= row[0] <= 92 and row[2] == position]
    walk.sort(key=lambda row: STAGES_PER_OWNER + (row[0] - 1) * STAGES_PER_OWNER + row[1])
    moved_pods, previous = 0, None
    by_local = collections.Counter()
    for owner, local, _, _, digest, _ in walk:
        if previous is not None and digest != previous:
            moved_pods += 1
            by_local[local] += 1
        previous = digest
    print("   %d stages recorded, %d move the residual (%.1f%%)"
          % (len(walk), moved_pods, 100.0 * moved_pods / max(len(walk), 1)))
    print("   locals that ever move it, and on how many of the 92 pods:")
    for local, count in sorted(by_local.items()):
        print("     local %-4d %3d pods" % (local, count))

    print("\n== step 4, normalisation (owner 93, stages 1..9) ==")
    tail = sorted([row for row in rows if row[0] == 93], key=lambda row: row[1])
    previous, moved_tail = None, 0
    names = {1: "finite check", 2: "sources 0..7 = snapshots", 3: "source 8 = residual",
             4: "score the nine sources", 5: "softmax the scores", 6: "fold",
             7: "finite check", 8: "apply RMS gain", 9: "finite check",
             110: "project to vocabulary, argmax", 111: "id to word",
             112: "append text", 113: "timing, position++",
             118: "build the body", 119: "deliver", 120: "end"}
    for owner, local, _, nanos, digest, detail in tail:
        if local == 110:
            print("\n== step 5, vector to token ==")
        changed = previous is not None and digest != previous
        moved_tail += 1 if changed else 0
        extra = "  token %d" % detail if detail >= 0 else ""
        print("   %-4d %-32s %9.0f ns  %s%s" % (local, names.get(local, "?"), nanos,
              "moved" if changed else ("start" if previous is None else "same"), extra))
        previous = digest
    print("\n   %d tail stages, %d move the result" % (len(tail), moved_tail))

    total = len(walk) + len(tail) + len(SERVER_STEPS)
    print("\n== the whole chain at position %d ==" % position)
    print("   %d observable steps: %d in the server, %d across the 92 pods, %d in the tail"
          % (total, len(SERVER_STEPS), len(walk), len(tail)))
    print("   %d of them change the carried state (%.1f%%)"
          % (moved_server + moved_pods + moved_tail,
             100.0 * (moved_server + moved_pods + moved_tail) / max(total, 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
