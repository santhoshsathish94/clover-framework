#!/usr/bin/env python3
"""Fold the 92 pods into the unit that actually repeats, and compare the repeats.

The creator's correction: the chain is 96 steps from input to output, not 92, and the
middle is not 92 independent layers — it is a 12-pod block that runs eight times. An
earlier walk of mine printed the 12-period as an event inside a 92-long sequence
("the residual collapses at every twelfth pod") rather than as the boundary of the unit.
This reads the same capture with the block as the unit.

From the model config, not from memory:
    attn_res_block_size   12
    first_k_dense_replace 1          so pod N is model layer N+1
    full_attn_layers      4, 8, ... 88, 92, 93   so MLA pods are 3, 7, ... 91, and 92

which puts MLA at block offsets 3, 7, 11 in every one of the seven full blocks, and at
3, 7, 8 in the short final block of eight where pod 92 is MLA out of turn.

What is printed is one row per block offset, with the eight blocks side by side, so a
quantity that is a property of the POSITION IN THE BLOCK separates from one that is a
property of DEPTH. No comparison to any expected output appears here.

usage: fold-blocks.py DUMP-PREFIX POSITION
"""
import glob
import math
import struct
import sys

BLOCK = 12
MLA_PODS = set(range(3, 92, 4)) | {92}

NOT_RESIDUAL = {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
                2003, 2021, 3000, 4003, 4021,
                5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009,
                5101, 5102, 5103}
CARRYING = (2, 6, 9, 20, 116)


def records(path, position, wanted=None):
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return
            owner, local, pos, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return
            if pos != position or (wanted is not None and local not in wanted):
                handle.seek(4 * count, 1)
                continue
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                return
            yield local, struct.unpack("<%df" % count, body)


def norm(values):
    return math.sqrt(math.fsum(float(v) * float(v) for v in values))


def spread(weights):
    total = math.fsum(weights)
    if total <= 0:
        return float("nan")
    squares = math.fsum((w / total) ** 2 for w in weights)
    return 1.0 / squares if squares > 0 else float("nan")


def block_of(pod):
    return (pod - 1) // BLOCK + 1


def offset_of(pod):
    return (pod - 1) % BLOCK + 1


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip())
        return 1
    prefix, position = sys.argv[1], int(sys.argv[2])
    files = {int(p.rsplit(".", 2)[-2]): p for p in glob.glob(prefix + ".*.bin")}

    out_norm, rel_move, routing, ratio, is_mla = {}, {}, {}, {}, {}
    for pod in range(1, 93):
        frame = dict(records(files[pod], position))
        if 119 in frame:
            out_norm[pod] = norm(frame[119])
        if 0 in frame and 119 in frame:
            before = norm(frame[0])
            rel_move[pod] = {}
            previous = frame[0]
            for local in CARRYING:
                if local in frame:
                    moved = math.sqrt(math.fsum((float(a) - float(b)) ** 2
                                                for a, b in zip(previous, frame[local])))
                    rel_move[pod][local] = moved / max(norm(previous), 1e-30)
                    previous = frame[local]
            del before
        if 3000 in frame:
            routing[pod] = spread(frame[3000])
        if 5101 in frame and 5102 in frame:
            shared = norm(frame[5102])
            ratio[pod] = norm(frame[5101]) / shared if shared else float("nan")
        is_mla[pod] = pod in MLA_PODS

    blocks = sorted({block_of(p) for p in range(1, 93)})

    print("== the eight repeats, side by side ==")
    print("   pod = 12*(block-1) + offset.  M marks an MLA pod, K a KDA pod.\n")

    for title, table, fmt in (("residual norm leaving the pod", out_norm, "%9.2f"),
                              ("router effective spread, of 16", routing, "%9.2f"),
                              ("routed / shared norm ratio", ratio, "%9.3f")):
        print("   %s" % title)
        print("   %-7s %s" % ("offset", "".join("%9s" % ("block %d" % b) for b in blocks)))
        for offset in range(1, BLOCK + 1):
            cells = []
            kinds = set()
            for b in blocks:
                pod = BLOCK * (b - 1) + offset
                if pod > 92 or pod not in table:
                    cells.append("%9s" % "-")
                    continue
                kinds.add("M" if is_mla[pod] else "K")
                cells.append(fmt % table[pod])
            mark = "".join(sorted(kinds))
            print("   %-5d %-2s%s" % (offset, mark, "".join(cells)))
        print()

    print("   relative movement at each carrying stage, median over the blocks")
    print("   %-7s %10s %10s %10s %10s %10s"
          % ("offset", "2 input", "6 MLA", "9 KDA", "20 fold", "116 moe"))
    for offset in range(1, BLOCK + 1):
        row = ["%10s" % "-"] * 5
        for index, local in enumerate(CARRYING):
            values = []
            for b in blocks:
                pod = BLOCK * (b - 1) + offset
                if pod <= 92 and pod in rel_move and local in rel_move[pod]:
                    values.append(rel_move[pod][local])
            if values:
                row[index] = "%10.3f" % sorted(values)[len(values) // 2]
        pods_here = [BLOCK * (b - 1) + offset for b in blocks if BLOCK * (b - 1) + offset <= 92]
        mark = "M" if all(is_mla[p] for p in pods_here) else "K" if not any(is_mla[p] for p in pods_here) else "mixed"
        print("   %-5d %-6s%s" % (offset, mark, "".join(row)))

    print("\n== what each block does to the state end to end ==")
    print("   %-7s %6s %10s %10s %9s" % ("block", "pods", "in", "out", "out/in"))
    for b in blocks:
        lo = BLOCK * (b - 1) + 1
        hi = min(BLOCK * b, 92)
        entering = dict(records(files[lo], position, {0}))
        if 0 not in entering or hi not in out_norm:
            continue
        start = norm(entering[0])
        print("   %-7d %6s %10.2f %10.2f %9.3f"
              % (b, "%d-%d" % (lo, hi), start, out_norm[hi],
                 out_norm[hi] / start if start else float("nan")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
