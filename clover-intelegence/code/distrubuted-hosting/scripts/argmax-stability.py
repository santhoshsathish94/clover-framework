#!/usr/bin/env python3
"""Which source wins the aggregate, and is the winner the same every time?

If the argmax is prompt-independent, hard-maxing is not a blend at all: it is a fixed
wiring that could be resolved once. If the winner moves with the prompt, the selection
is itself carrying information and cannot be baked in.

Reads the softmax weights already captured from five prompts. No new run.
"""
import collections
import glob
import struct

PROMPTS = ["p1", "p2", "p3", "p4", "p5"]
BOUNDARY = {12, 24, 36, 48, 60, 72, 84}


def read(prefix):
    folds = {}
    for path in sorted(glob.glob(f"{prefix}.*.bin")):
        with open(path, "rb") as handle:
            while True:
                header = handle.read(16)
                if len(header) < 16:
                    break
                owner, local, position, count = struct.unpack("<4i", header)
                if count <= 0 or count > 1 << 20:
                    break
                if local >= 120 and position > 0:
                    position -= 1
                if position != 0 or not (2000 <= local < 3000):
                    handle.seek(4 * count, 1)
                    continue
                payload = handle.read(4 * count)
                if len(payload) < 4 * count:
                    break
                folds.setdefault(owner, {})[local - 2000] = struct.unpack(f"<{count}f", payload)
    return folds


def main():
    data = {}
    for name in PROMPTS:
        f = read(f"/tmp/vp-{name}")
        if f:
            data[name] = f
    print(f"prompts: {list(data)}")

    owners = sorted(set.intersection(*(set(f) for f in data.values())))
    print(f"owners: {len(owners)}")

    agree, disagree = 0, []
    residual_wins = collections.Counter()
    total = collections.Counter()
    for owner in owners:
        for stage in (3, 21):
            picks = {}
            for name, folds in data.items():
                w = folds[owner].get(stage)
                if not w:
                    continue
                best = max(range(len(w)), key=lambda i: w[i])
                picks[name] = (best, len(w))
            if not picks:
                continue
            winners = {p[0] for p in picks.values()}
            count = next(iter(picks.values()))[1]
            total[stage] += 1
            # the residual is always the last source
            if winners == {count - 1}:
                residual_wins[stage] += 1
            if len(winners) == 1:
                agree += 1
            else:
                disagree.append((owner, stage, picks, count))

    print()
    print(f"aggregate calls where all five prompts pick the same source: {agree}")
    print(f"calls where they disagree: {len(disagree)}")
    for stage in (3, 21):
        print(f"  stage {stage:2d}: residual won in all five prompts for "
              f"{residual_wins[stage]} of {total[stage]} owners")

    if disagree:
        print()
        print("the calls where the winner moves with the prompt:")
        for owner, stage, picks, count in disagree[:20]:
            detail = " ".join(f"{n}:{v[0]}" for n, v in picks.items())
            mark = " <-" if owner in BOUNDARY else ""
            print(f"  owner {owner:3d} stage {stage:2d} ({count} sources)  {detail}{mark}")

    print()
    print("=== when the winner is stable, which source is it? ===")
    stable = collections.Counter()
    for owner in owners:
        for stage in (3, 21):
            picks = set()
            count = 0
            for folds in data.values():
                w = folds[owner].get(stage)
                if not w:
                    continue
                count = len(w)
                picks.add(max(range(len(w)), key=lambda i: w[i]))
            if len(picks) == 1 and count:
                best = next(iter(picks))
                stable["residual" if best == count - 1 else f"snapshot {best}"] += 1
    for key, n in stable.most_common():
        print(f"  {key:12s} {n}")


if __name__ == "__main__":
    main()
