#!/usr/bin/env python3
# Does slot 0 depend on layer 12, or on any layer?
#
# The server writes snapshots[0] before any layer runs, and stage 4 appends at index
# snapshot_count, which is never 0. So slot 0 should be created once and carried
# through untouched. That is what the code says. This checks the bytes instead.
#
# If it is untouched, slot 0 captured at owner 2 and at owner 92 must be identical for
# the same position, and in particular identical either side of layer 12.
import glob, struct, math, collections

slot0 = {}
for path in glob.glob("/tmp/fcv.*"):
    raw = open(path, "rb").read()
    at = 0
    while at + 16 <= len(raw):
        owner, local, position, n = struct.unpack_from("<4i", raw, at)
        at += 16
        if local == 1000:
            if position > 0:
                position -= 1
            slot0[(owner, position)] = struct.unpack_from("<%df" % n, raw, at)
        at += 4 * n

positions = sorted({p for _, p in slot0})
print("positions: %s" % positions)
print()

for position in positions:
    owners = sorted(o for o, p in slot0 if p == position)
    if not owners:
        continue
    reference = slot0[(owners[0], position)]
    exact = sum(1 for o in owners if slot0[(o, position)] == reference)
    worst, worst_owner = 0.0, None
    for o in owners:
        v = slot0[(o, position)]
        d = max(abs(a - b) for a, b in zip(v, reference))
        if d > worst:
            worst, worst_owner = d, o
    print("position %d: %d owners (%d..%d), byte-identical to owner %d: %d of %d, "
          "largest elementwise difference %.3e%s"
          % (position, len(owners), owners[0], owners[-1], owners[0], exact, len(owners),
             worst, "" if worst_owner is None else " at owner %d" % worst_owner))

print()
print("either side of layer 12, same position:")
for position in positions:
    before = [o for o in range(2, 12) if (o, position) in slot0]
    after = [o for o in range(13, 93) if (o, position) in slot0]
    if not before or not after:
        continue
    a, b = slot0[(before[0], position)], slot0[(after[0], position)]
    same = a == b
    print("  position %d: owner %d vs owner %d -> %s"
          % (position, before[0], after[0], "identical" if same else "DIFFERENT"))

print()
print("does slot 0 change between positions? (it should, the server remakes it each token)")
owner = 2
for i in range(1, len(positions)):
    a, b = slot0.get((owner, positions[i - 1])), slot0.get((owner, positions[i]))
    if not a or not b:
        continue
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    cos = sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else float("nan")
    print("  owner %d, position %d vs %d: cos %.6f, norms %.4g and %.4g"
          % (owner, positions[i - 1], positions[i], cos, na, nb))
