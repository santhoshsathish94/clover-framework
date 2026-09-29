#!/usr/bin/env python3
"""Independently check complete-group arrangement coding and AX102 model outputs."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from math import factorial, prod
from pathlib import Path
import struct


def ways(counts):
    return factorial(sum(counts)) // prod(factorial(count) for count in counts)


def rank_sequence(codes, counts):
    remaining = list(counts)
    rank = 0
    for selected in codes:
        for alternative in range(selected):
            if remaining[alternative]:
                remaining[alternative] -= 1
                rank += ways(remaining)
                remaining[alternative] += 1
        assert remaining[selected] > 0
        remaining[selected] -= 1
    assert not any(remaining)
    return rank


def unrank_sequence(counts, rank):
    remaining = list(counts)
    assert 0 <= rank < ways(remaining)
    decoded, steps = [], []
    for position in range(sum(remaining)):
        before = rank
        for code in range(16):
            if not remaining[code]:
                continue
            remaining[code] -= 1
            block = ways(remaining)
            if rank < block:
                decoded.append(code)
                steps.append({'position': position, 'code': code,
                              'rank_before': str(before), 'rank_after': str(rank),
                              'selected_prefix_arrangements': str(block)})
                break
            rank -= block
            remaining[code] += 1
        else:
            raise AssertionError('No code selected')
    assert rank == 0
    return decoded, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = args.directory
    spec = importlib.util.spec_from_file_location('group_reference_check', work / 'expert-group-check.py')
    reference_checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_checker)
    order = json.loads((work / 'order.json').read_text())
    group = json.loads((work / 'group.json').read_text())
    previous = json.loads((work / 'reference-group.json').read_text())
    assert (group['layer'], group['expert_id'], group['row'], group['group']) == (1, 498, 0, 0)
    assert order['width'] == 32 and order['alphabet'] == 16
    assert len(order['counts']) == 16 and sum(order['counts']) == 32
    assert len(order['decoded_codes']) == len(previous['codes']) == 32
    counts = Counter(previous['codes'])
    assert order['counts'] == [counts[code] for code in range(16)]
    assert order['decoded_codes'] == previous['codes'] == group['codes']
    total = ways(order['counts'])
    rank = rank_sequence(previous['codes'], order['counts'])
    assert total == int(order['arrangements']) and rank == int(order['rank'])
    decoded, steps = unrank_sequence(order['counts'], rank)
    assert decoded == previous['codes']
    rank_bits = (total - 1).bit_length()
    assert order['rank_bits'] == rank_bits and order['count_bits'] == 96
    assert order['payload_bits'] == 8 + 96 + rank_bits
    assert order['decoder_checks'] == 4373 and order['invalid_inputs_rejected'] == 5
    assert order['codes_checked'] == 32 and order['codes_changed'] == 0

    descriptor = bytes(order['payload_bytes'])
    assert len(descriptor) == (order['payload_bits'] + 7) // 8
    packed = int.from_bytes(descriptor, 'little')
    assert packed >> order['payload_bits'] == 0
    stored_scale = packed & 255
    packed >>= 8
    stored_counts = []
    for code in range(16):
        stored_counts.append(packed & 63)
        packed >>= 6
    assert stored_scale == order['scale_byte'] == previous['scale_byte']
    assert stored_counts == order['counts'] and packed == rank
    assert unrank_sequence(stored_counts, packed)[0] == previous['codes']

    for key in ('tensor', 'input_width', 'rows', 'positions', 'scale_byte', 'packed_bytes', 'masks'):
        assert group[key] == previous[key]
    assert len(group['decoded_weights']) == len(previous['decoded_weights']) == 32
    for actual, expected in zip(group['decoded_weights'], previous['decoded_weights']):
        assert struct.pack('<f', actual) == struct.pack('<f', expected)
    assert group['group_weights_checked'] == 32 and group['group_weights_changed'] == 0
    assert group['packed_roundtrip_exact'] and group['projection_outputs_changed'] == 0
    assert group['decode_pairs_checked'] == 4096 and group['mask_coordinates_checked'] == 512
    assert len(group['projection_outputs']) == len(previous['projection_outputs']) == group['positions'] == 1
    for actual, expected in zip(group['projection_outputs'], previous['projection_outputs']):
        assert actual['position'] == expected['position'] == 0 and actual['rank'] == expected['rank'] == 0
        assert reference_checker.bits(actual['reference']) == reference_checker.bits(actual['candidate'])
        assert reference_checker.bits(actual['reference']) == reference_checker.bits(expected['reference'])
    reference = (work / 'reference.bin').read_bytes()
    assert hashlib.md5(reference).hexdigest() == reference_checker.BASELINE_MD5
    variants = {name: reference_checker.compare(reference, (work / (name + '.bin')).read_bytes())
                for name in ('disabled', 'order-disabled', 'enabled')}
    repeated_matches = (work / 'reference-after.bin').read_bytes() == reference
    passed = order['gate'] == group['gate'] == 'PASS' and repeated_matches
    passed = passed and all(variant['bit_exact'] for variant in variants.values())
    with (work / 'group-order.bin').open('xb') as output:
        output.write(descriptor)
    names = ('baseline.c', 'candidate.c', 'group-base.h', 'group-base-lf.h',
             'expert-group-equation.h', 'expert-group-equation.patch',
             'expert-group-order.h', 'expert-group-order.patch', 'expert-group-order-check.py',
             'expert-group-check.py', 'reference', 'candidate', 'order.json', 'group.json',
             'reference-group.json', 'group-order.bin')
    result = {
        'host': 'AX102, AMD Ryzen 9 7950X3D',
        'prompt_ids': [1008, 10484, 318, 15383, 387],
        'build_flags': '-O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5 -lm',
        'settings': {'OMP_NUM_THREADS': '16', 'OMP_PROC_BIND': 'close', 'OMP_PLACES': 'cores',
                     'K3_PREFETCH': '4', 'K3_TRUNKRAM': '0', 'K3_NREADER': '14',
                     'K3_XDEC': '2', 'K3_PLGRAN': '1', 'K3_HUGE': '1'},
        'order': order, 'group': group, 'decode_steps': steps,
        'variants': variants, 'reference_repeated_matches': repeated_matches,
        'independent_integer_rank_and_serialization_match': True,
        'reference_md5': reference_checker.BASELINE_MD5,
        'gate': 'PASS' if passed else 'FAIL',
        'representation': {'original_bits': 136, 'original_bytes': 17,
                           'scale_bits': 8, 'count_bits': 96, 'rank_bits': rank_bits,
                           'payload_bits': order['payload_bits'], 'serialized_bytes': len(descriptor),
                           'serialization': 'Scale in first byte, then sixteen 6-bit counts (code order 0..15), then lexicographic rank. All fields least-significant bit first; rank width derived from counts. Fixed group size/alphabet and caller-supplied location.'},
        'sha256': {name: hashlib.sha256((work / name).read_bytes()).hexdigest() for name in names},
        'scope': 'One complete 32-code group, layer1 expert498 w1/gate row0 group0, pos0 rank0, one five-token prefill. Constructs descriptor from loaded original weights, serializes/decodes it, and uses regenerated masks/scale in the expert projection. No alternative store or other groups substituted.',
        'performance': 'Deferred. This representation is larger for the recorded group; no checkpoint, RAM or runtime reduction claimed.',
    }
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('decode_steps', 'group')}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()