#!/usr/bin/env python3
# Step 1. Is snapshot[k] the residual, or merely very close to it?
#
# cos = 1.000000 to six places is not identity. If the two are the same bits then the
# snapshot carries no information the residual stream did not already have at layer
# 12k, and the 22 MB of memcpy per token is transport, not content. If they differ
# even slightly then something happens between stage 2 and stage 4 and the whole idea
# of reconstructing them needs that difference explained first.
#
# Compared as raw float32 bit patterns, not as floats, so -0.0 vs 0.0 and any NaN
# payload would show up rather than being smoothed over by a tolerance.
import glob, struct, collections

want_residual = {}      # owner 12k -> residual at stage 3
want_snapshot = collections.defaultdict(dict)   # owner -> slot -> vector

for path in glob.glob("/tmp/fcv.*"):
    raw = open(path, "rb").read()
    at = 0
    while at + 16 <= len(raw):
        owner, local, position, n = struct.unpack_from("<4i", raw, at)
        at += 16
        payload = raw[at:at + 4 * n]
        at += 4 * n
        if local == 3 and owner % 12 == 0:
            want_residual[(owner, position)] = payload
        elif 1000 <= local < 1008:
            p = position - 1 if position > 0 else position
            want_snapshot[(owner, p)][local - 1000] = payload

print("slot k is pushed at layer 12k, from the same `input` buffer stage 2 copied")
print("into the residual. Comparing raw float32 bits at the layer that made it.")
print()
print("slot  layer  positions  identical  first differing index")
for k in range(1, 8):
    layer = 12 * k
    same = total = 0
    first_diff = None
    for (owner, position), slots in want_snapshot.items():
        if owner != layer or k not in slots:
            continue
        residual = want_residual.get((owner, position))
        if residual is None:
            continue
        total += 1
        if slots[k] == residual:
            same += 1
        elif first_diff is None:
            a = struct.unpack("<%df" % (len(slots[k]) // 4), slots[k])
            b = struct.unpack("<%df" % (len(residual) // 4), residual)
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    first_diff = (i, x, y)
                    break
    note = "-" if first_diff is None else "index %d: %.9g vs %.9g" % first_diff
    print("  %2d  %5d  %9d  %9s  %s"
          % (k, layer, total, "%d/%d" % (same, total), note))

print()
print("control: the same comparison one layer later, where it should NOT hold")
for k in (1, 2):
    layer = 12 * k + 1
    checked = same = 0
    for (owner, position), slots in want_snapshot.items():
        if owner != layer or k not in slots:
            continue
        residual = want_residual.get((12 * k, position))
        if residual is None:
            continue
        checked += 1
        if slots[k] == residual:
            same += 1
    print("  slot %d carried into layer %d still identical to layer %d's residual: %d/%d"
          % (k, layer, 12 * k, same, checked))
