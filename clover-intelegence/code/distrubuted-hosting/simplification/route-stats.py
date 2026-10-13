#!/usr/bin/env python3
"""How much does expert selection actually vary?

Reads the 5103 records (sixteen chosen expert ids per token) from one or more dumps and
reports, per layer:

  always    experts chosen by every token
  never     of the 896, how many are never chosen at all
  overlap   mean size of the intersection between two tokens' choices, out of 16
  distinct  how many different experts the layer used across all tokens

An expert chosen by every token is doing the same job as the trunk, whatever the routing
machinery says. That is the thing the output-variance measurement could not see.
"""
import collections
import struct
import sys

if len(sys.argv) < 2:
    sys.exit("usage: route-stats.py <dump.bin> [more dumps...]")

picks = []
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
            if local == 5103 and count == 16:
                picks.append(frozenset(int(v) for v in struct.unpack("<16f", payload)))

if len(picks) < 2:
    print("NA\tonly %d samples" % len(picks))
    sys.exit(1)

counts = collections.Counter()
for pick in picks:
    counts.update(pick)

always = sum(1 for e, c in counts.items() if c == len(picks))
distinct = len(counts)

# Pairwise overlap over all pairs; the sample is small enough that this is exact.
total, pairs = 0, 0
for i in range(len(picks)):
    for j in range(i + 1, len(picks)):
        total += len(picks[i] & picks[j])
        pairs += 1

print("%d\t%d\t%d\t%.2f\t%d" % (
    len(picks),          # tokens
    always,              # experts every token chose
    distinct,            # distinct experts used
    total / pairs,       # mean pairwise overlap out of 16
    896 - distinct,      # experts never used
))
