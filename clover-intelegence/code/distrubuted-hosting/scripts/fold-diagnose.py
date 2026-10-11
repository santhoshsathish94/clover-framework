#!/usr/bin/env python3
"""Why does the pooled answer disagree with the per-position answer?

Pooled said 117 calls always pick a snapshot. Position 0 alone said only about 27
calls pick a snapshot at all. Both cannot be true, so this prints the raw per-key
evidence instead of another summary.
"""
import collections
import glob
import os
import struct

PREFIXES = sorted({p.rsplit('.', 2)[0] for p in glob.glob('/tmp/wt-*.bin')})


def records(path):
    with open(path, 'rb') as handle:
        blob = handle.read()
    at = 0
    while at + 16 <= len(blob):
        owner, local, position, count = struct.unpack_from('<4i', blob, at)
        at += 16
        need = count * 4
        if count < 0 or at + need > len(blob):
            break
        yield owner, local, position, struct.unpack_from('<%df' % count, blob, at)
        at += need


seen = collections.defaultdict(list)
for prefix in PREFIXES:
    prompt = os.path.basename(prefix)
    for path in glob.glob(prefix + '.*.bin'):
        for owner, local, position, values in records(path):
            if 2000 <= local < 3000 and len(values) >= 2:
                seen[(owner, local - 2000)].append((prompt, position, values))

widths = {key: {len(v) for _, _, v in rows} for key, rows in seen.items()}
varying = {key: sizes for key, sizes in widths.items() if len(sizes) > 1}
print('aggregate calls              : %d' % len(seen))
print('calls whose width varies     : %d' % len(varying))
for key, sizes in sorted(varying.items())[:10]:
    print('   owner %3d stage %2d widths %s' % (key[0], key[1], sorted(sizes)))
print()

counts = collections.Counter(len(rows) for rows in seen.values())
print('observations per call        : %s' % dict(sorted(counts.items())))
print()


def winner(values):
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


# Pick a few calls that position 0 calls "residual" and show every observation.
print('=== every observation for the first few calls ===')
shown = 0
for key in sorted(seen):
    rows = sorted(seen[key], key=lambda row: (row[1], row[0]))
    at_zero = {winner(v) for p, pos, v in rows if pos == 0}
    if len(at_zero) != 1:
        continue
    only = at_zero.pop()
    if only != len(rows[0][2]) - 1:
        continue                       # not a residual win at position 0
    picks = collections.Counter(winner(v) for _, _, v in rows)
    print('owner %3d stage %2d width %d   winners %s'
          % (key[0], key[1], len(rows[0][2]),
             {('residual' if s == len(rows[0][2]) - 1 else 'snap%d' % s): n
              for s, n in picks.most_common()}))
    shown += 1
    if shown == 8:
        break
