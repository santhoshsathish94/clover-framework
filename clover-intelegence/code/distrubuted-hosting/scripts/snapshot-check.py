#!/usr/bin/env python3
"""The snapshot channel, compared across prompts.

The residual at the twelfth-layer boundaries was nearly prompt-independent: cosine
0.999+ between unrelated prompts. If the prompt is carried anywhere, the snapshots are
the remaining candidate. This measures them the same way the residual was measured, so
the two are directly comparable.

Snapshot records are written with local = 1000 + slot.
"""
import glob
import struct

PROMPTS = ["p1", "p2", "p3", "p4", "p5"]
BOUNDARY = {12, 24, 36, 48, 60, 72, 84}


def read(prefix):
    """Per owner: the final residual, and every snapshot slot."""
    residual, snaps = {}, {}
    for path in sorted(glob.glob(f"{prefix}.*.bin")):
        with open(path, "rb") as handle:
            while True:
                header = handle.read(16)
                if len(header) < 16:
                    break
                owner, local, position, count = struct.unpack("<4i", header)
                if count <= 0 or count > 1 << 20:
                    break
                # Stage 120 and the snapshot dump both run after the position counter
                # has been incremented, so they report one position later than they belong to.
                snapshot = 1000 <= local < 1008
                if (local == 120 or snapshot) and position > 0:
                    position -= 1
                keep = position == 0 and (local == 120 or snapshot)
                if not keep:
                    handle.seek(4 * count, 1)
                    continue
                payload = handle.read(4 * count)
                if len(payload) < 4 * count:
                    break
                values = struct.unpack(f"<{count}f", payload)
                if local == 120:
                    residual[owner] = values
                else:
                    snaps.setdefault(owner, {})[local - 1000] = values
    return residual, snaps


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return dot(v, v) ** 0.5


def cosine(a, b):
    na, nb = norm(a), norm(b)
    return dot(a, b) / (na * nb) if na and nb else float("nan")


def main():
    data = {}
    for name in PROMPTS:
        r, s = read(f"/tmp/vp-{name}")
        if r:
            data[name] = (r, s)
    print(f"prompts: {list(data)}")
    if len(data) < 2:
        print("need two")
        return

    a_r, a_s = data["p1"]
    b_r, b_s = data["p2"]
    d_r, d_s = data["p4"]

    owners = sorted(set(a_s) & set(b_s) & set(d_s))
    print(f"owners with snapshots: {len(owners)}")
    counts = {o: len(a_s[o]) for o in owners}
    print(f"snapshot slots per owner: min {min(counts.values())} max {max(counts.values())}")

    print()
    print("=== residual vs snapshots: how much does each differ between prompts? ===")
    print("owner  slots   cos_residual   cos_snap_last   cos_snap_mean   ||snap_last||")
    for o in owners:
        if o not in (2, 11, 12, 13, 23, 24, 35, 36, 47, 48, 59, 60, 71, 72, 83, 84, 91, 92):
            continue
        slots = sorted(a_s[o])
        last = slots[-1]
        cr = cosine(a_r[o], b_r[o]) if o in a_r and o in b_r else float("nan")
        cs_last = cosine(a_s[o][last], b_s[o][last])
        cs_all = [cosine(a_s[o][s], b_s[o][s]) for s in slots if s in b_s[o]]
        mark = " <-" if o in BOUNDARY else ""
        print(f"{o:5d} {len(slots):6d} {cr:14.6f} {cs_last:15.6f} "
              f"{sum(cs_all)/len(cs_all):15.6f} {norm(a_s[o][last]):15.4g}{mark}")

    print()
    print("=== the newest snapshot at each boundary, across three prompts ===")
    print("owner  slot   ||s||     cos(p1,p2)   cos(p1,p4)   cos(p2,p4)")
    for o in sorted(BOUNDARY):
        if o not in owners:
            continue
        last = sorted(a_s[o])[-1]
        if last not in b_s[o] or last not in d_s[o]:
            continue
        print(f"{o:5d} {last:5d} {norm(a_s[o][last]):8.4g} "
              f"{cosine(a_s[o][last], b_s[o][last]):12.6f} "
              f"{cosine(a_s[o][last], d_s[o][last]):12.6f} "
              f"{cosine(b_s[o][last], d_s[o][last]):12.6f}")

    print()
    print("=== summary over every owner and slot ===")
    res_cos, snap_cos = [], []
    for o in owners:
        if o in a_r and o in b_r:
            res_cos.append(cosine(a_r[o], b_r[o]))
        for s in sorted(a_s[o]):
            if s in b_s[o]:
                snap_cos.append(cosine(a_s[o][s], b_s[o][s]))
    print(f"residual  cos(p1,p2): n {len(res_cos):4d}  min {min(res_cos):.6f}  mean {sum(res_cos)/len(res_cos):.6f}")
    print(f"snapshots cos(p1,p2): n {len(snap_cos):4d}  min {min(snap_cos):.6f}  mean {sum(snap_cos)/len(snap_cos):.6f}")


if __name__ == "__main__":
    main()
