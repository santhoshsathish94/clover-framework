#!/usr/bin/env python3
"""How much does hard-maxing change? The winning weight is what hardmax raises to 1.

If the winner already holds 0.99 the change is slight; where it holds 0.4 the layer
is genuinely blending and hardmax discards most of what it was using.
"""
import glob
import struct

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
    folds = read("/tmp/vp-p1")
    rows = []
    for owner in sorted(folds):
        for fold, weights in sorted(folds[owner].items()):
            best = max(weights)
            rows.append((best, owner, fold, len(weights)))

    print(f"aggregate calls observed: {len(rows)}")
    buckets = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 0.99), (0.99, 1.01)]
    print()
    print("winning weight before hardmax")
    for low, high in buckets:
        hits = [r for r in rows if low <= r[0] < high]
        share = len(hits) / len(rows)
        bar = "#" * int(share * 50)
        print(f"  {low:4.2f} to {high:4.2f} : {len(hits):4d}  {share:6.3f} {bar}")

    ordered = sorted(rows)
    print()
    print(f"median winning weight : {ordered[len(ordered)//2][0]:.4f}")
    print(f"mean                  : {sum(r[0] for r in rows)/len(rows):.4f}")
    print(f"weight discarded by hardmax, mean : {1 - sum(r[0] for r in rows)/len(rows):.4f}")

    print()
    print("the ten calls hardmax changes most (lowest winning weight)")
    for best, owner, fold, n in ordered[:10]:
        mark = " <-" if owner in BOUNDARY else ""
        print(f"  owner {owner:3d} stage {fold:2d}  sources {n}  winner holds {best:.4f}, "
              f"discards {1-best:.4f}{mark}")

    print()
    print("the ten it changes least")
    for best, owner, fold, n in ordered[-10:]:
        mark = " <-" if owner in BOUNDARY else ""
        print(f"  owner {owner:3d} stage {fold:2d}  sources {n}  winner holds {best:.4f}{mark}")


if __name__ == "__main__":
    main()
