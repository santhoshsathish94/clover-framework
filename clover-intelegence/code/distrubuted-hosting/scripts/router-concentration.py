#!/usr/bin/env python3
# The folds are 184 of 9,880 executing stages. The experts are 7,360 of them, 74.5%,
# and nothing here has ever measured them.
#
# Stage 24 routes: it chooses 16 experts and weights them. Stages 30-109 compute all
# sixteen, five stages each, and sum them weighted. That is the same question already
# asked of the fold, on three quarters of the machine instead of two percent: if one
# expert already holds nearly all the weight, the other fifteen are 75 stages of work
# whose contribution is rounding.
#
# Reads the capture already on disk. No run needed.
import glob, struct, collections, math, statistics

rows = []
for path in glob.glob("/tmp/fcv.*"):
    raw = open(path, "rb").read()
    at = 0
    while at + 16 <= len(raw):
        owner, local, position, n = struct.unpack_from("<4i", raw, at)
        at += 16
        if local == 3000:
            rows.append((owner, position, struct.unpack_from("<%df" % n, raw, at)))
        at += 4 * n

if not rows:
    raise SystemExit("no router records (local 3000) in /tmp/fcv.* ")

print("router records: %d over %d owners" % (len(rows), len({o for o, _, _ in rows})))
print()

per_owner = collections.defaultdict(list)
tops, top4s, entropies = [], [], []
for owner, position, weights in rows:
    ordered = sorted(weights, reverse=True)
    total = sum(weights)
    if total <= 0:
        continue
    top1 = ordered[0] / total
    top4 = sum(ordered[:4]) / total
    per_owner[owner].append((top1, top4))
    tops.append(top1)
    top4s.append(top4)
    entropies.append(-sum((w / total) * math.log(w / total) for w in weights if w > 0))

def describe(name, xs):
    xs = sorted(xs)
    print("%-28s n %4d  min %.4f  median %.4f  mean %.4f  max %.4f"
          % (name, len(xs), xs[0], statistics.median(xs), sum(xs) / len(xs), xs[-1]))

describe("top expert share", tops)
describe("top four share", top4s)
describe("entropy over 16 (nats)", entropies)
print("uniform 16 would be entropy %.4f, top1 %.4f, top4 %.4f"
      % (math.log(16), 1 / 16, 4 / 16))
print()

for threshold in (0.5, 0.7, 0.9, 0.95, 0.99):
    print("  records where the top expert holds > %-4s : %5d of %d  (%.1f%%)"
          % (threshold, sum(1 for t in tops if t > threshold), len(tops),
             100.0 * sum(1 for t in tops if t > threshold) / len(tops)))
print()
for threshold in (0.9, 0.95, 0.99):
    print("  records where the top FOUR hold      > %-4s : %5d of %d  (%.1f%%)"
          % (threshold, sum(1 for t in top4s if t > threshold), len(top4s),
             100.0 * sum(1 for t in top4s if t > threshold) / len(top4s)))
print()

print("owner  n   top1 mean   top4 mean")
concentrated = []
for owner in sorted(per_owner):
    values = per_owner[owner]
    m1 = sum(v[0] for v in values) / len(values)
    m4 = sum(v[1] for v in values) / len(values)
    print("  %3d %3d      %.4f      %.4f" % (owner, len(values), m1, m4))
    if m4 > 0.9:
        concentrated.append(owner)

def ranges(nums):
    out, start, prev = [], None, None
    for n in sorted(nums):
        if start is None:
            start = prev = n
            continue
        if n == prev + 1:
            prev = n
            continue
        out.append((start, prev))
        start = prev = n
    if start is not None:
        out.append((start, prev))
    return ",".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in out)

print()
print("owners where the top four hold > 0.9 of the weight: %d" % len(concentrated))
print("  %s" % ranges(concentrated))
