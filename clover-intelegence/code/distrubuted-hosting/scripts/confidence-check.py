#!/usr/bin/env python3
"""The confidences, compared across prompts.

Two kinds are recorded:
  2003 / 2021  the aggregate softmax over snapshots plus residual, at stages 3 and 21
  3000         the router's weight on each of the sixteen chosen experts

The vectors themselves were near-identical across prompts. If the blend weights are
not, that is where the prompt is being carried.
"""
import glob
import struct

PROMPTS = ["p1", "p2", "p3", "p4", "p5"]
BOUNDARY = {12, 24, 36, 48, 60, 72, 84}


def read(prefix):
    folds, router = {}, {}
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
                if position != 0 or local < 2000:
                    handle.seek(4 * count, 1)
                    continue
                payload = handle.read(4 * count)
                if len(payload) < 4 * count:
                    break
                values = struct.unpack(f"<{count}f", payload)
                if local >= 3000:
                    router[owner] = values
                else:
                    folds.setdefault(owner, {})[local - 2000] = values
    return folds, router


def main():
    data = {}
    for name in PROMPTS:
        f, r = read(f"/tmp/vp-{name}")
        if f or r:
            data[name] = (f, r)
    print(f"prompts: {list(data)}")
    if len(data) < 2:
        print("need two")
        return

    a_f, a_r = data["p1"]
    b_f, b_r = data["p2"]
    d_f, d_r = data["p4"]
    owners = sorted(set(a_f) & set(b_f) & set(d_f))
    print(f"owners with aggregate weights: {len(owners)}")
    print(f"owners with router weights   : {len(set(a_r) & set(b_r))}")

    print()
    print("=== aggregate softmax at stage 21, p1 vs p2 vs p4 ===")
    print("owner  n    p1 weights (residual last)                            maxdiff")
    for o in owners:
        if o not in (2, 11, 12, 13, 23, 24, 36, 48, 60, 72, 83, 84, 91, 92):
            continue
        wa, wb, wd = a_f[o].get(21), b_f[o].get(21), d_f[o].get(21)
        if not wa:
            continue
        diff = max(max(abs(x - y) for x, y in zip(wa, wb)),
                   max(abs(x - z) for x, z in zip(wa, wd)))
        shown = " ".join(f"{x:.4f}" for x in wa)
        mark = " <-" if o in BOUNDARY else ""
        print(f"{o:5d} {len(wa):2d}  {shown:52s} {diff:.6f}{mark}")

    print()
    print("=== how far apart are the aggregate weights, over every owner ===")
    worst = []
    for o in owners:
        for fold in (3, 21):
            wa, wb = a_f[o].get(fold), b_f[o].get(fold)
            if not wa or not wb:
                continue
            d = max(abs(x - y) for x, y in zip(wa, wb))
            worst.append((d, o, fold, wa, wb))
    worst.sort(reverse=True)
    print(f"pairs compared: {len(worst)}")
    print(f"max absolute weight difference p1 vs p2: {worst[0][0]:.6f} at owner {worst[0][1]} fold {worst[0][2]}")
    print(f"median: {worst[len(worst)//2][0]:.6f}   min: {worst[-1][0]:.8f}")
    print()
    print("the five largest disagreements:")
    for d, o, fold, wa, wb in worst[:5]:
        print(f"  owner {o:3d} stage {fold:2d} maxdiff {d:.5f}")
        print(f"    p1 {' '.join(f'{x:.4f}' for x in wa)}")
        print(f"    p2 {' '.join(f'{x:.4f}' for x in wb)}")

    print()
    print("=== router weights on the sixteen chosen experts ===")
    rowners = sorted(set(a_r) & set(b_r) & set(d_r))
    print(f"{'owner':>5s} {'p1 sum':>8s} {'p1 max':>8s} {'p1 min':>8s} {'p2 max':>8s} {'maxdiff':>9s}")
    for o in rowners:
        if o not in (2, 12, 24, 36, 46, 48, 60, 72, 84, 92):
            continue
        wa, wb = a_r[o], b_r[o]
        diff = max(abs(x - y) for x, y in zip(wa, wb))
        print(f"{o:5d} {sum(wa):8.4f} {max(wa):8.4f} {min(wa):8.4f} {max(wb):8.4f} {diff:9.5f}")

    diffs = [max(abs(x - y) for x, y in zip(a_r[o], b_r[o])) for o in rowners]
    print()
    print(f"router weight max difference p1 vs p2 over {len(rowners)} owners: "
          f"min {min(diffs):.5f}  median {sorted(diffs)[len(diffs)//2]:.5f}  max {max(diffs):.5f}")


if __name__ == "__main__":
    main()
