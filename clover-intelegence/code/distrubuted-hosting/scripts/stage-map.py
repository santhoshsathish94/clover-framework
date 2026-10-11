#!/usr/bin/env python3
"""The run as 11,280 stages, not 93 layers.

Every summary so far collapsed a layer's 108 stages into one row, which hides what
each stage does. The per-stage residual is already in the captures: local 0 is the
layer input and local 1..120 is the residual after that stage ran. This reads them
as stages and asks two questions the layer view cannot answer.

  moved    how far the residual travelled during that stage
  cos      how similar that stage's residual is across different prompts

Global stage numbering follows common/stage.h: owner N occupies 120*N+1 .. 120*N+120.
"""
import glob
import os
import struct
import sys

import numpy as np

PROMPTS = sys.argv[1:] or ['p1', 'p2', 'p3', 'p4', 'p5']
WIDTH = 7168
STRIDE = 120


def read_owner(path):
    """local -> residual, for stage records only, at the first position."""
    with open(path, 'rb') as handle:
        blob = handle.read()
    out, at = {}, 0
    while at + 16 <= len(blob):
        owner, local, position, count = struct.unpack_from('<4i', blob, at)
        at += 16
        need = count * 4
        if count <= 0 or at + need > len(blob):
            break
        # stage 120 runs after the position counter advances, so it reports one late
        reported = position - 1 if (local == 120 and position > 0) else position
        if reported == 0 and 0 <= local <= 120 and count == WIDTH:
            out[local] = np.frombuffer(blob, dtype='<f4', count=WIDTH, offset=at)
        at += need
    return out


def load(prompt):
    owners = {}
    for path in glob.glob('/tmp/vp-%s.*.bin' % prompt):
        owner = int(os.path.basename(path).split('.')[1])
        rows = read_owner(path)
        if rows:
            owners[owner] = rows
    return owners


data = {p: load(p) for p in PROMPTS}
data = {p: v for p, v in data.items() if v}
if len(data) < 2:
    print('need at least two prompt captures, found', list(data))
    raise SystemExit(1)

first = PROMPTS[0]
owners = sorted(set.intersection(*(set(v) for v in data.values())))
print('prompts %s, owners %d' % (list(data), len(owners)))


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float('nan')


rows = []
for owner in owners:
    stages = sorted(set.intersection(*(set(data[p][owner]) for p in data)))
    for index, local in enumerate(stages):
        if local == 0:
            continue
        previous = stages[index - 1]
        before = data[first][owner][previous]
        after = data[first][owner][local]
        moved = float(np.linalg.norm(after - before))
        pair = [cos(data[a][owner][local], data[b][owner][local])
                for i, a in enumerate(PROMPTS) if a in data
                for b in list(data)[i + 1:]]
        rows.append({
            'global': STRIDE * owner + local,
            'owner': owner,
            'stage': local,
            'moved': moved,
            'norm': float(np.linalg.norm(after)),
            'cos': min(pair) if pair else float('nan'),
        })

print('stage records compared: %d' % len(rows))
print('NOTE: this tracks the RESIDUAL only. The folds at stages 3 and 21 write')
print('      sequence->aggregate, which is not captured, so their "moved" is 0 by')
print('      construction and says nothing about whether they matter.')
print()

movers = [r for r in rows if r['moved'] > 1e-6]
print('=== which stages move the residual at all? ===')
print('stages that moved it : %d of %d (%.1f%%)'
      % (len(movers), len(rows), 100.0 * len(movers) / len(rows)))
bystage = {}
for r in movers:
    bystage.setdefault(r['stage'], []).append(r)
print('distinct stage numbers that ever move it: %s' % sorted(bystage))
for stage in sorted(bystage):
    group = bystage[stage]
    print('   stage %3d  moved in %2d owners  median step %12.4g'
          % (stage, len(group), float(np.median([g['moved'] for g in group]))))
print()

good = [r for r in rows if not np.isnan(r['cos'])]
allcos = np.array([r['cos'] for r in good])
print('=== how prompt-dependent is the residual, stage by stage? ===')
print('comparable stages: %d (%d dropped, residual still zero there)'
      % (len(good), len(rows) - len(good)))
print('cos min %.6f  median %.6f  mean %.6f  max %.6f'
      % (allcos.min(), float(np.median(allcos)), allcos.mean(), allcos.max()))
for cut in (0.99, 0.95, 0.90, 0.80, 0.60):
    n = int((allcos < cut).sum())
    print('  stages below %.2f : %5d  (%.1f%%)' % (cut, n, 100.0 * n / len(good)))
print()

print('=== depth profile: mean cos of the residual, by owner ===')
byowner = {}
for r in good:
    byowner.setdefault(r['owner'], []).append(r['cos'])
keys = sorted(byowner)
for start in range(0, len(keys), 13):
    chunk = keys[start:start + 13]
    print('  owner ' + ' '.join('%6d' % o for o in chunk))
    print('  cos   ' + ' '.join('%6.3f' % float(np.mean(byowner[o])) for o in chunk))
print()

print('=== which stage numbers are most prompt-dependent, across all owners ===')
bynum = {}
for r in good:
    bynum.setdefault(r['stage'], []).append(r['cos'])
ranked = sorted(bynum.items(), key=lambda kv: float(np.mean(kv[1])))
print('  stage   owners   mean cos')
for stage, values in ranked[:12]:
    print('  %5d   %6d   %8.6f' % (stage, len(values), float(np.mean(values))))
print('  ...')
for stage, values in ranked[-4:]:
    print('  %5d   %6d   %8.6f' % (stage, len(values), float(np.mean(values))))
print()

print('=== the two fold stages, per owner ===')
print('  owner   stage 3 cos   stage 21 cos   stage 3 moved  stage 21 moved')
folds = {}
for r in rows:
    if r['stage'] in (3, 21):
        folds.setdefault(r['owner'], {})[r['stage']] = r
for owner in sorted(folds)[:12]:
    a, b = folds[owner].get(3), folds[owner].get(21)
    print('  %5d   %11s   %12s   %13s  %14s'
          % (owner,
             '%.6f' % a['cos'] if a else '-', '%.6f' % b['cos'] if b else '-',
             '%.4g' % a['moved'] if a else '-', '%.4g' % b['moved'] if b else '-'))

