#!/usr/bin/env python3
"""How concentrated are the sixteen router weights?

The selection is per token and barely overlaps between tokens, so the routed path is
where the per-token work happens. That does not say all sixteen picks matter equally. If
the weight is concentrated in the first few, the rest are being computed for very little,
and that is a simplification worth testing rather than assuming.

Reports, pooled over every position in the dumps given:

  top        mean weight of the highest-ranked expert
  top4       mean share of weight held by the first four
  effective  mean 1/sum(w^2), the number of experts the distribution behaves like
  need90     mean count needed to reach 90 percent of the weight

usage: weight-stats.py <dump.bin> [more dumps...]
"""
import struct
import sys

if len(sys.argv) < 2:
    sys.exit("usage: weight-stats.py <dump.bin> [more...]")

rows = []
for path in sys.argv[1:]:
    try:
        handle = open(path, "rb")
    except OSError:
        continue
    with handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                break
            _owner, local, _position, count = struct.unpack("<4i", header)
            if count < 0 or count > 65536:
                break
            payload = handle.read(count * 4)
            if len(payload) < count * 4:
                break
            if local == 3000 and count == 16:
                rows.append(list(struct.unpack("<16f", payload)))

if not rows:
    print("NA\tno router records")
    sys.exit(1)

top = top4 = effective = need90 = 0.0
for weights in rows:
    total = sum(weights) or 1.0
    ordered = sorted(weights, reverse=True)
    top += ordered[0] / total
    top4 += sum(ordered[:4]) / total
    effective += 1.0 / sum((w / total) ** 2 for w in weights)
    running = 0.0
    for index, w in enumerate(ordered, 1):
        running += w / total
        if running >= 0.9:
            need90 += index
            break
    else:
        need90 += 16

n = len(rows)
print("%d\t%.4f\t%.4f\t%.2f\t%.2f" % (n, top / n, top4 / n, effective / n, need90 / n))
