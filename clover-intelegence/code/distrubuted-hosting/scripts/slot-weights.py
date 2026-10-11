#!/usr/bin/env python3
# Why is the drop-cosine median high for every slot?
#
# The suspicion is that it is a statement about the WEIGHTS, not about the slots: if
# a fold gives a snapshot almost no weight, removing it cannot change much, whatever
# the snapshot contains. beta = alpha/(1-alpha) is the factor the closed form uses,
# so small alpha means small beta means a_drop is close to a by construction.
#
# This prints the weight distribution per slot so the median cosine can be read as a
# consequence of it rather than as a property of the slot.
import glob, struct, collections, statistics

rows = collections.defaultdict(dict)
for path in glob.glob("/tmp/fcv.*"):
    raw = open(path, "rb").read()
    at = 0
    while at + 16 <= len(raw):
        owner, local, position, n = struct.unpack_from("<4i", raw, at)
        at += 16
        if local in (2003, 2021) or 1000 <= local < 1008:
            if 1000 <= local < 1008 and position > 0:
                position -= 1
            rows[(owner, position)][local] = struct.unpack_from("<%df" % n, raw, at)
        at += 4 * n

per_slot = collections.defaultdict(list)
residual_weight = []
for (owner, position), got in rows.items():
    slots = sorted(l for l in got if 1000 <= l < 1008)
    for w_code in (2003, 2021):
        weights = got.get(w_code)
        if not weights or len(weights) != len(slots) + 1:
            continue
        residual_weight.append(weights[-1])
        for index, local in enumerate(slots):
            per_slot[local - 1000].append(weights[index])

print("weight the fold gives each slot, alpha_j")
print("slot     n     median      mean       p90       max   share of folds alpha<0.01")
for k in sorted(per_slot):
    xs = sorted(per_slot[k])
    tiny = sum(1 for x in xs if x < 0.01)
    print("  %2d %5d  %9.5f %9.5f %9.5f %9.5f        %5.1f%%"
          % (k, len(xs), statistics.median(xs), sum(xs) / len(xs),
             xs[int(0.9 * (len(xs) - 1))], xs[-1], 100.0 * tiny / len(xs)))

xs = sorted(residual_weight)
print()
print("weight the fold gives the residual: median %.5f  mean %.5f  min %.5f"
      % (statistics.median(xs), sum(xs) / len(xs), xs[0]))
print()
print("beta = alpha/(1-alpha) at the median alpha, which is the factor in a_drop:")
for k in sorted(per_slot):
    m = statistics.median(per_slot[k])
    print("  slot %d  median alpha %.5f  ->  beta %.5f" % (k, m, m / (1.0 - m)))
