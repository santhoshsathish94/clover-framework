#!/usr/bin/env python3
"""Is the next snapshot the previous one plus a single repeated step?

The direction is that once the first snapshot exists the rest can be had by
adding one more of a predefined residual, which would mean the intervening
layers need not run to produce them. That is a claim with a shape: the
differences between consecutive slots would all have to be the same vector.
"""
import math
import struct
import sys

LAYERS = [int(a) for a in sys.argv[1:]] or [84, 92]
WIDTH = 7168


def load(layer):
    rows = {}
    with open('/tmp/hold-donor.%d.bin' % layer, 'rb') as handle:
        blob = handle.read()
    at = 0
    while at + 16 <= len(blob):
        owner, local, position, count = struct.unpack_from('<4i', blob, at)
        at += 16
        if count <= 0 or at + count * 4 > len(blob):
            break
        if 1000 <= local < 1008 and count == WIDTH:
            rows[(position, local - 1000)] = struct.unpack_from('<%df' % WIDTH, blob, at)
        at += count * 4
    return rows


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def cos(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return float('nan')
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


for layer in LAYERS:
    rows = load(layer)
    positions = sorted({p for p, _ in rows})
    slots = sorted({s for _, s in rows})
    print('=== layer %d, slots %s ===' % (layer, slots))
    for position in positions[:2]:
        have = [s for s in slots if (position, s) in rows]
        if len(have) < 3:
            continue
        vectors = [rows[(position, s)] for s in have]
        print(' position %d' % position)
        print('   norms            %s' % ' '.join('%8.2f' % norm(v) for v in vectors))
        print('   cos(n, n+1)      %s'
              % ' '.join('%8.4f' % cos(vectors[i], vectors[i + 1])
                         for i in range(len(vectors) - 1)))
        steps = [sub(vectors[i + 1], vectors[i]) for i in range(len(vectors) - 1)]
        print('   step norms       %s' % ' '.join('%8.2f' % norm(d) for d in steps))
        print('   cos(step0, stepN)%s'
              % ' '.join('%8.4f' % cos(steps[0], d) for d in steps))

        # If one repeated step explained it, slot n would be slot 0 plus n of it.
        whole = sub(vectors[-1], vectors[0])
        count = len(vectors) - 1
        unit = [x / count for x in whole]
        worst = 0.0
        for index in range(1, len(vectors)):
            predicted = [vectors[0][i] + index * unit[i] for i in range(WIDTH)]
            error = norm(sub(vectors[index], predicted)) / max(norm(vectors[index]), 1e-9)
            worst = max(worst, error)
        print('   worst relative error if one repeated step explained it: %.4f' % worst)
    print()
