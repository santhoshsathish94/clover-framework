#!/usr/bin/env python3
"""Do the six probes route to the same experts?

If they do not, that alone accounts for the additivity and homogeneity failures, and
the transformation class question becomes: is the block affine once routing is held
fixed. This script only reports the selections; it does not answer that.
"""
import collections

NAMES = ["zero", "u", "v", "u+v", "2u", "w(held out)"]


def main():
    runs, current, seen_first = [], None, False
    with open("/tmp/probe-trace.csv") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 6:
                continue
            owner, local, position, nanos, digest, detail = parts
            local, detail = int(local), int(detail)
            if local == 1:
                current = []
                runs.append(current)
                seen_first = True
            if not seen_first or current is None:
                continue
            if detail >= 0:
                current.append(detail)

    print(f"probe runs found: {len(runs)}")
    picks = []
    for index, experts in enumerate(runs):
        ordered = list(dict.fromkeys(experts))
        picks.append(ordered)
        name = NAMES[index] if index < len(NAMES) else f"probe{index}"
        print(f"  {name:12s} {len(ordered)} distinct: {ordered}")

    print()
    print("pairwise overlap of selected experts")
    for i in range(len(picks)):
        for j in range(i + 1, len(picks)):
            a, b = set(picks[i]), set(picks[j])
            shared = len(a & b)
            union = len(a | b)
            print(f"  {NAMES[i]:12s} vs {NAMES[j]:12s} shared {shared:2d}/16   jaccard {shared/union:.3f}")

    print()
    counts = collections.Counter()
    for p in picks:
        counts.update(p)
    print(f"experts used by all six probes: {sorted(e for e, c in counts.items() if c == len(picks))}")
    print(f"experts used by exactly one   : {len([e for e, c in counts.items() if c == 1])}")
    print(f"distinct experts over all six : {len(counts)}")


if __name__ == "__main__":
    main()
