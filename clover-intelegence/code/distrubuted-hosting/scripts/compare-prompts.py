#!/usr/bin/env python3
"""Does the across-owner shape recur, or was it one trajectory's accident?

For each prompt, the final residual of every owner. Then: are the norm peaks and
collapses in the same places, are the growth ratios per owner similar, and do the
twelfth-layer boundaries behave the same way.

Reports agreement and disagreement equally. A structure that appears in one prompt
and not the others is a property of that prompt.
"""
import glob
import statistics
import struct

WIDTH = 7168
PROMPTS = ["p1", "p2", "p3", "p4", "p5"]
MLA = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92}
BOUNDARY = {12, 24, 36, 48, 60, 72, 84}


def read(prefix):
    finals, inputs = {}, {}
    for path in sorted(glob.glob(f"{prefix}.*.bin")):
        with open(path, "rb") as handle:
            while True:
                header = handle.read(16)
                if len(header) < 16:
                    break
                owner, local, position, count = struct.unpack("<4i", header)
                if count <= 0 or count > 1 << 20:
                    break
                if local == 120 and position > 0:
                    position -= 1
                if position != 0 or local not in (0, 120):
                    handle.seek(4 * count, 1)
                    continue
                payload = handle.read(4 * count)
                if len(payload) < 4 * count:
                    break
                values = struct.unpack(f"<{count}f", payload)
                if local == 0:
                    inputs[owner] = values
                else:
                    finals[owner] = values
    return inputs, finals


def norm(v):
    return sum(x * x for x in v) ** 0.5


def main():
    data = {}
    for name in PROMPTS:
        inputs, finals = read(f"/tmp/vp-{name}")
        if finals:
            data[name] = (inputs, finals)
    print(f"prompts with data: {list(data)}")
    if len(data) < 2:
        print("need at least two to compare")
        return

    owners = sorted(set.intersection(*(set(f) for _i, f in data.values())))
    print(f"owners common to all: {len(owners)}  {owners[0]}..{owners[-1]}")

    norms = {name: {o: norm(f[o]) for o in owners} for name, (_i, f) in data.items()}

    print()
    print("=== norm at each owner, per prompt ===")
    print("owner kind " + "".join(f"{n:>11s}" for n in data))
    for o in owners:
        kind = "MLA" if o in MLA else "KDA"
        mark = " <-" if o in BOUNDARY else ""
        print(f"{o:5d} {kind:4s} " + "".join(f"{norms[n][o]:11.4g}" for n in data) + mark)

    print()
    print("=== growth ratio per owner, per prompt ===")
    ratios = {}
    for name in data:
        inputs, finals = data[name]
        r = {}
        for o in owners:
            if o in inputs:
                a = norm(inputs[o])
                r[o] = norms[name][o] / a if a else float("inf")
        ratios[name] = r
    common = sorted(set.intersection(*(set(r) for r in ratios.values())))
    print("owner kind " + "".join(f"{n:>11s}" for n in data) + "   spread")
    for o in common:
        vals = [ratios[n][o] for n in data]
        kind = "MLA" if o in MLA else "KDA"
        mark = " <-" if o in BOUNDARY else ""
        spread = max(vals) / min(vals) if min(vals) > 0 else float("inf")
        print(f"{o:5d} {kind:4s} " + "".join(f"{v:11.4g}" for v in vals) + f" {spread:8.2f}x{mark}")

    print()
    print("=== where is the norm smallest, per prompt? ===")
    for name in data:
        order = sorted(owners, key=lambda o: norms[name][o])
        print(f"  {name}: lowest five owners {[(o, round(norms[name][o], 3)) for o in order[:5]]}")

    print()
    print("=== where is the norm largest, per prompt? ===")
    for name in data:
        order = sorted(owners, key=lambda o: -norms[name][o])
        print(f"  {name}: highest five owners {[(o, round(norms[name][o], 1)) for o in order[:5]]}")

    print()
    print("=== do the twelfth-layer boundaries collapse the norm in every prompt? ===")
    print("owner " + "".join(f"{n:>12s}" for n in data))
    for o in sorted(BOUNDARY):
        if o not in owners or (o - 1) not in owners:
            continue
        cells = []
        for name in data:
            before, after = norms[name][o - 1], norms[name][o]
            cells.append(f"{before:5.1f}->{after:<5.2f}")
        print(f"{o:5d} " + "".join(f"{c:>12s}" for c in cells))

    print()
    print("=== correlation of the norm profile between prompts ===")
    names = list(data)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = [norms[names[i]][o] for o in owners]
            b = [norms[names[j]][o] for o in owners]
            try:
                r = statistics.correlation(a, b)
            except Exception:
                r = float("nan")
            la = [__import__("math").log(x) for x in a if x > 0]
            lb = [__import__("math").log(x) for x in b if x > 0]
            try:
                rl = statistics.correlation(la, lb) if len(la) == len(lb) else float("nan")
            except Exception:
                rl = float("nan")
            print(f"  {names[i]} vs {names[j]}: pearson {r:.4f}   on log norms {rl:.4f}")


if __name__ == "__main__":
    main()
