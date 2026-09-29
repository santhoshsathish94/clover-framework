#!/usr/bin/env python3
"""Evaluate exact combinatorial ranks of the verified expert group's masks."""

import argparse
from itertools import combinations
import json
from math import comb
from pathlib import Path


def rank_mask(mask, width):
    assert 0 <= mask < 1 << width
    positions = [position for position in range(width) if mask & (1 << position)]
    rank = sum(comb(position, ordinal) for ordinal, position in enumerate(positions, 1))
    return len(positions), rank


def unrank_mask(count, rank, width):
    assert 0 <= count <= width and 0 <= rank < comb(width, count)
    mask, upper = 0, width - 1
    for ordinal in range(count, 0, -1):
        position = upper
        while comb(position, ordinal) > rank:
            position -= 1
        mask |= 1 << position
        rank -= comb(position, ordinal)
        upper = position - 1
    assert rank == 0
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.group.read_text())
    group = data['group'] if isinstance(data.get('group'), dict) else data
    assert group['gate'] == 'PASS' and len(group['masks']) == 4

    checks = 0
    for mask in range(1 << 12):
        count, rank = rank_mask(mask, 12)
        assert unrank_mask(count, rank, 12) == mask
        checks += 1
    for count in range(3):
        for positions in combinations(range(32), count):
            mask = sum(1 << position for position in positions)
            for value in (mask, mask ^ 0xFFFFFFFF):
                population, rank = rank_mask(value, 32)
                assert unrank_mask(population, rank, 32) == value
                checks += 1
    for count in range(33):
        limit = comb(32, count)
        for rank in sorted({0, limit // 2, limit - 1}):
            mask = unrank_mask(count, rank, 32)
            assert rank_mask(mask, 32) == (count, rank)
            checks += 1

    rows = []
    for plane, mask in enumerate(group['masks']):
        count, rank = rank_mask(mask, 32)
        possibilities = comb(32, count)
        bits = (possibilities - 1).bit_length()
        assert unrank_mask(count, rank, 32) == mask
        rows.append({'plane': plane, 'mask': mask, 'set_bits': count, 'rank': rank,
                     'possible_masks': possibilities, 'rank_bits': bits,
                     'count_bits': 6, 'count_plus_rank_bits': 6 + bits,
                     'set_positions': [position for position in range(32) if mask & (1 << position)]})
    fixed_bits = 8 + sum(row['count_plus_rank_bits'] for row in rows)
    mixed_bits = 8 + sum(1 + min(32, row['count_plus_rank_bits']) for row in rows)
    result = {
        'site': {name: group[name] for name in ('layer', 'expert_id', 'tensor', 'row', 'group')},
        'scale_byte': group['scale_byte'], 'masks': rows,
        'rank_equation': 'For ascending set positions p_1 < ... < p_k: rank = sum(binomial(p_j, j), j=1..k).',
        'known_count_domain': 'Exactly binomial(32,k) masks. Count stored in six bits for k=0..32.',
        'decoder_checks': checks, 'actual_mask_roundtrips': len(rows),
        'representations': {
            'original': {'bits': 136, 'bytes': 17},
            'all_ranked': {'bits': fixed_bits, 'rounded_bytes': (fixed_bits + 7) // 8},
            'per_plane_choice': {'bits': mixed_bits, 'rounded_bytes': (mixed_bits + 7) // 8,
                                 'accounting': 'One raw/ranked tag per plane, six count bits for ranked planes, 8-bit scale.'},
        },
        'scope': 'Exact mask reconstruction on the real recorded group plus finite decoder controls. No new model execution or changed projection. Bit counts are candidate format accounting, not whole-expert compression.',
    }
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()