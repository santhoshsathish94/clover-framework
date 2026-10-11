#!/usr/bin/env python3
"""Does the winning source stay the same as the position advances?

Reads the weights-only vector dumps for several prompts, each run for several
positions, and asks one question per aggregate call: across every (prompt,
position) pair we have, did the same source always win? A call that always
picks the same source does not need to be scored at run time.
"""
import collections
import glob
import os
import struct
import sys

PREFIXES = sys.argv[1:] or sorted(
    {p.rsplit('.', 2)[0] for p in glob.glob('/tmp/wt-*.bin')})

# POSITIONS=0 or POSITIONS=0,1 narrows the sample, so the earlier position-0 result
# can be reproduced from the same capture instead of trusted from memory.
KEEP = os.environ.get('POSITIONS')
KEEP = {int(part) for part in KEEP.split(',')} if KEEP else None


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
        values = struct.unpack_from('<%df' % count, blob, at)
        at += need
        yield owner, local, position, values


def winner(values):
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


# transformer_aggregate fills sources[] with the snapshots first and puts the
# residual last, so the residual is index count-1, never index 0.
def name_of(source, count):
    return 'residual' if source == count - 1 else 'snapshot %d' % source


# (owner, stage) -> {(prompt, position): winning source}
picks = collections.defaultdict(dict)
widths = {}
positions_seen = collections.defaultdict(set)

for prefix in PREFIXES:
    prompt = os.path.basename(prefix)
    for path in glob.glob(prefix + '.*.bin'):
        for owner, local, position, values in records(path):
            if local < 2000 or local >= 3000 or len(values) < 2:
                continue
            if KEEP is not None and position not in KEEP:
                continue
            stage = local - 2000
            picks[(owner, stage)][(prompt, position)] = winner(values)
            widths[(owner, stage)] = len(values)
            positions_seen[prompt].add(position)

if not picks:
    print('no aggregate weights found for', PREFIXES)
    raise SystemExit(1)

print('prompts and positions captured:')
for prompt in sorted(positions_seen):
    seen = sorted(positions_seen[prompt])
    print('  %-8s %d positions %s' % (prompt, len(seen), seen[:12]))
print()

fixed, moving = {}, {}
for key, observations in sorted(picks.items()):
    chosen = set(observations.values())
    if len(chosen) == 1:
        fixed[key] = chosen.pop()
    else:
        moving[key] = observations

total = len(picks)
print('aggregate calls observed            : %d' % total)
print('calls where one source always won   : %d (%.1f%%)'
      % (len(fixed), 100.0 * len(fixed) / total))
print('calls where the winner moved        : %d' % len(moving))
print()

# Separate the two reasons a winner can move: a different prompt, or a later
# position in the same prompt. Only the second is new information.
by_position_only, by_prompt_only, by_both = [], [], []
for key, observations in moving.items():
    within = collections.defaultdict(set)
    for (prompt, position), source in observations.items():
        within[prompt].add(source)
    moves_in_prompt = any(len(sources) > 1 for sources in within.values())
    first = {prompt: min(
        (pos, src) for (pr, pos), src in observations.items() if pr == prompt)[1]
        for prompt in within}
    moves_across = len(set(first.values())) > 1
    if moves_in_prompt and moves_across:
        by_both.append(key)
    elif moves_in_prompt:
        by_position_only.append(key)
    else:
        by_prompt_only.append(key)

print('  moves with position inside one prompt only : %d  %s'
      % (len(by_position_only), sorted(by_position_only)[:10]))
print('  moves with the prompt only                 : %d  %s'
      % (len(by_prompt_only), sorted(by_prompt_only)[:10]))
print('  moves with both                            : %d  %s'
      % (len(by_both), sorted(by_both)[:10]))
print()

share = collections.Counter(
    name_of(source, widths[key]) for key, source in fixed.items())
print('=== when the winner is fixed, which source is it? ===')
for name, count in share.most_common():
    print('  %-12s %d' % (name, count))
print()

table = os.environ.get('SELECTION_TABLE')
if table:
    with open(table, 'w') as handle:
        handle.write('owner,stage,source\n')
        for (owner, stage), source in sorted(fixed.items()):
            handle.write('%d,%d,%d\n' % (owner, stage, source))
        for (owner, stage) in sorted(moving):
            handle.write('%d,%d,-1\n' % (owner, stage))
    print('selection table written to', table)
