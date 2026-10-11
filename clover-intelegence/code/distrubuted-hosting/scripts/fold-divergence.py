#!/usr/bin/env python3
"""Where do the standalone and the distributed model stop agreeing?

Both now record the fold weights in the same layout. This walks the folds in
order and reports the first one where the two pick a different source, which is
a fact about the two programs rather than another reading of their source.
"""
import collections
import glob
import struct
import sys

STANDALONE = sys.argv[1] if len(sys.argv) > 1 else '/tmp/sa-folds.bin'
DISTRIBUTED = sys.argv[2] if len(sys.argv) > 2 else '/tmp/wt-france'
POSITION = int(sys.argv[3]) if len(sys.argv) > 3 else 0


def records(path):
    with open(path, 'rb') as handle:
        blob = handle.read()
    at = 0
    while at + 16 <= len(blob):
        owner, local, position, count = struct.unpack_from('<4i', blob, at)
        at += 16
        if count <= 0 or at + count * 4 > len(blob):
            break
        yield owner, local, position, struct.unpack_from('<%df' % count, blob, at)
        at += count * 4


def winner(values):
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def collect(paths):
    found = {}
    for path in paths:
        for owner, local, position, values in records(path):
            if 2000 <= local < 3000 and position == POSITION:
                found[(owner, local - 2000)] = values
    return found


left = collect([STANDALONE])
right = collect(sorted(glob.glob(DISTRIBUTED + '.*.bin')))

print('position %d' % POSITION)
print('standalone folds  %d   owners %s'
      % (len(left), '%d-%d' % (min(o for o, _ in left), max(o for o, _ in left)) if left else '-'))
print('distributed folds %d   owners %s'
      % (len(right), '%d-%d' % (min(o for o, _ in right), max(o for o, _ in right)) if right else '-'))
print()

shared = sorted(set(left) & set(right))
print('folds present in both: %d' % len(shared))
only_left = sorted(set(left) - set(right))
only_right = sorted(set(right) - set(left))
if only_left:
    print('  only in standalone : %d, first few %s' % (len(only_left), only_left[:6]))
if only_right:
    print('  only in distributed: %d, first few %s' % (len(only_right), only_right[:6]))
print()

disagreements = []
for key in shared:
    a, b = left[key], right[key]
    if len(a) != len(b):
        disagreements.append((key, 'width %d vs %d' % (len(a), len(b))))
    elif winner(a) != winner(b):
        disagreements.append((key, 'winner %d vs %d' % (winner(a), winner(b))))

print('folds where the two disagree: %d of %d' % (len(disagreements), len(shared)))
for key, why in disagreements[:20]:
    a, b = left[key], right[key]
    print('  owner %3d stage %2d  %s' % (key[0], key[1], why))
    print('       standalone  %s' % ' '.join('%.4f' % v for v in a))
    print('       distributed %s' % ' '.join('%.4f' % v for v in b))

if not disagreements and shared:
    print('  every shared fold picks the same source')

widths = collections.Counter((len(left[k]), len(right[k])) for k in shared)
print()
print('source counts seen (standalone, distributed): %s' % dict(widths))
