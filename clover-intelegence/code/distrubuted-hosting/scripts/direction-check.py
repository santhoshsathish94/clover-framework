#!/usr/bin/env python3
"""Magnitude agrees across prompts. Does direction?

If the norm profile is a property of the network, the prompt has to be carried
somewhere else. This measures the angle between prompts at the same owner.
"""
import glob
import struct

MLA = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92}
BOUNDARY = {12, 24, 36, 48, 60, 72, 84}


def read(prefix):
    finals = {}
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
                if position != 0 or local != 120:
                    handle.seek(4 * count, 1)
                    continue
                payload = handle.read(4 * count)
                if len(payload) < 4 * count:
                    break
                finals[owner] = struct.unpack(f"<{count}f", payload)
    return finals


def cosine(a, b):
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    return sum(x * y for x, y in zip(a, b)) / (da * db) if da and db else float("nan")


def main():
    a = read("/tmp/vp-p1")
    b = read("/tmp/vp-p2")
    c = read("/tmp/vp-p4")
    owners = sorted(set(a) & set(b) & set(c))
    print(f"owners compared: {len(owners)}")
    print()
    print("owner kind  cos(p1,p2)  cos(p1,p4)  cos(p2,p4)")
    rows = []
    for o in owners:
        r = (o, "MLA" if o in MLA else "KDA", cosine(a[o], b[o]), cosine(a[o], c[o]), cosine(b[o], c[o]))
        rows.append(r)
        if o <= 13 or o in BOUNDARY or o >= 90:
            mark = " <-" if o in BOUNDARY else ""
            print(f"{o:5d} {r[1]:4s} {r[2]:11.5f} {r[3]:11.5f} {r[4]:11.5f}{mark}")

    print()
    vals = [r[2] for r in rows]
    print(f"cos(p1,p2) over all owners: min {min(vals):.5f}  max {max(vals):.5f}  mean {sum(vals)/len(vals):.5f}")
    early = [r[2] for r in rows if r[0] <= 20]
    late = [r[2] for r in rows if r[0] >= 70]
    print(f"  owners <= 20: mean {sum(early)/len(early):.5f}")
    print(f"  owners >= 70: mean {sum(late)/len(late):.5f}")


if __name__ == "__main__":
    main()
