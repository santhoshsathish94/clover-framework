#!/usr/bin/env python3
"""Arrange the measured flow as a tree and test the tree against the census.

The unit the creator names is KDA-KDA-KDA-MLA. For MLA to be last, groups must align to
pod = 3 mod 4, which makes them (0,1,2,3), (4,5,6,7), ... and puts the server at pod 0 as
the first leaf rather than a prefix outside the structure.

That re-aligns the block too. The write condition in the pod source is pod % 12 == 0, so
a block runs 12B..12B+11 with the WRITE AT ITS FIRST POD, not its last. Earlier work here
used pods 1-12 as block 1, which put the write at the end and cut the groups; 12 = 3 x 4,
so the aligned block is exactly three groups.

Nothing below is assumed. Each level of the tree is checked against the data:

  does every group end in MLA and contain exactly three KDA
  is snapshot slot k+1 the output of block k
  is the MLA pod the largest fold movement WITHIN ITS OWN GROUP

usage: tree.py ROWS.tsv
"""
import collections
import math
import sys

GROUP = 4
BLOCK = 12


def median(values):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def load(path):
    rows = []
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            for key in ("position", "block", "offset", "pod"):
                row[key] = int(row[key])
            for key in ("out_norm", "rel_attn", "rel_fold", "rel_moe", "in_norm",
                        "spread", "ratio"):
                try:
                    row[key] = float(row[key])
                except ValueError:
                    row[key] = float("nan")
            rows.append(row)
    return rows


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip())
        return 1
    rows = load(sys.argv[1])
    prompts = sorted({r["prompt"] for r in rows})

    print("== level 1: does every group of four end in MLA? ==")
    kinds = {}
    for row in rows:
        kinds[row["pod"]] = row["attn"]
    bad = []
    for start in range(0, 93, GROUP):
        members = [p for p in range(start, start + GROUP) if p in kinds]
        if len(members) < GROUP:
            print("   group at pod %2d is short: pods %s %s"
                  % (start, members, [kinds[p] for p in members]))
            continue
        pattern = "".join(kinds[p] for p in members)
        if pattern != "KKKM":
            bad.append((start, pattern))
    print("   %d complete groups checked, %d are not KKKM%s"
          % (93 // GROUP, len(bad), (": " + str(bad)) if bad else ""))
    print("   pod 0 is the server and writes no census row, so group (0,1,2,3) shows")
    print("   as three members; its pods 1 and 2 are KDA and pod 3 is MLA.")

    print("\n== level 2: a block is three groups, with the write at its first pod ==")
    for block in range(8):
        lo = BLOCK * block
        members = [p for p in range(lo, min(lo + BLOCK, 93))]
        groups = [members[i:i + GROUP] for i in range(0, len(members), GROUP)]
        shape = "  ".join("(%s)" % "".join(kinds.get(p, "s") for p in g) for g in groups)
        print("   block %d  pods %2d-%2d  write at pod %2d   %s"
              % (block, members[0], members[-1], members[0], shape))
    print("   s = the server, pod 0, which writes no census row")

    print("\n== level 3: is snapshot slot k+1 the output of block k? ==")
    print("   last pod of block k, and its outgoing residual norm, france position 4")
    frame = {r["pod"]: r for r in rows if r["prompt"] == "france" and r["position"] == 4}
    observed = {1: 112.8886, 2: 63.1326, 3: 34.1738, 4: 59.8422,
                5: 133.4080, 6: 50.9070, 7: 63.5633}
    for block in range(7):
        last = min(BLOCK * block + BLOCK - 1, 92)
        if last in frame:
            slot = block + 1
            print("   block %d ends at pod %2d, out %10.4f   tail slot %d = %10.4f   %s"
                  % (block, last, frame[last]["out_norm"], slot, observed[slot],
                     "same" if abs(frame[last]["out_norm"] - observed[slot]) < 0.001
                     else "DIFFERENT"))
    print("   block 7 ends at pod 92, out %10.4f   the tail's residual source"
          % frame[92]["out_norm"])
    print("   slot 0 is the raw embedding, already shown bit-identical to the server input.")
    print("   so the nine sources are: the input, the output of blocks 0-6, and block 7.")

    print("\n== level 4: within a group, which of the four moves the fold most? ==")
    print("   %-10s %7s %7s %7s %7s %10s" % ("prompt", "K", "K", "K", "M", "M is max"))
    overall_wins = overall_total = 0
    for prompt in prompts:
        by_slot = collections.defaultdict(list)
        wins = total = 0
        for position in sorted({r["position"] for r in rows if r["prompt"] == prompt}):
            frame = {r["pod"]: r for r in rows
                     if r["prompt"] == prompt and r["position"] == position}
            for start in range(0, 93, GROUP):
                members = [p for p in range(start, start + GROUP) if p in frame]
                if len(members) != GROUP:
                    continue
                folds = [frame[p]["rel_fold"] for p in members]
                if any(math.isnan(f) for f in folds):
                    continue
                for index, value in enumerate(folds):
                    by_slot[index].append(value)
                total += 1
                if folds[3] == max(folds):
                    wins += 1
        overall_wins += wins
        overall_total += total
        print("   %-10s %7.3f %7.3f %7.3f %7.3f %7d/%d"
              % (prompt, median(by_slot[0]), median(by_slot[1]), median(by_slot[2]),
                 median(by_slot[3]), wins, total))
    print("   the MLA pod has the largest fold movement in its own group of four")
    print("   in %d of %d groups across all prompts (%.1f%%)"
          % (overall_wins, overall_total, 100.0 * overall_wins / max(overall_total, 1)))

    print("\n== the same four slots, for the other measured quantities ==")
    for field, label in (("rel_attn", "attention movement"),
                         ("rel_moe", "moe movement"),
                         ("spread", "router spread"),
                         ("ratio", "routed/shared")):
        by_slot = collections.defaultdict(list)
        for row in rows:
            value = row[field]
            if math.isnan(value):
                continue
            by_slot[row["pod"] % GROUP].append(value)
        print("   %-20s  K %7.3f   K %7.3f   K %7.3f   M %7.3f"
              % (label, median(by_slot[0]), median(by_slot[1]),
                 median(by_slot[2]), median(by_slot[3])))

    print("\n== the tree ==")
    print("""
   request
   |
   +- embed(token)  -> vector ................................ fold source 0
   |
   +- block 0   pods  0-11   write at pod  0 (the server)
   |    +- group  (s  K  K  M)   pods  0  1  2  3
   |    +- group  (K  K  K  M)   pods  4  5  6  7
   |    +- group  (K  K  K  M)   pods  8  9 10 11 ......... out = fold source 1
   +- block 1   pods 12-23   write at pod 12
   |    +- 3 groups ...................................... out = fold source 2
   +- block 2   pods 24-35 ............................... out = fold source 3
   +- block 3   pods 36-47 ............................... out = fold source 4
   +- block 4   pods 48-59 ............................... out = fold source 5
   +- block 5   pods 60-71 ............................... out = fold source 6
   +- block 6   pods 72-83 ............................... out = fold source 7
   +- block 7   pods 84-92   write at pod 84, two groups + pod 92
   |                                 ..................... out = fold source 8
   |
   +- fold the nine sources -> RMS gain -> project to 163840 -> token

   one pod  = 120 stages, of which 3 do arithmetic on the carried state
              (attention, attention fold, moe add)
   one moe  = 16 of 896 experts, 80 expert stages writing a separate buffer
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
