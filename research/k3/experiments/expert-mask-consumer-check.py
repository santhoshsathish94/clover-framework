#!/usr/bin/env python3
"""Verify the count/rank consumer against prior group evidence and full outputs."""

import argparse
import hashlib
import importlib.util
import json
from math import comb
from pathlib import Path
import struct


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = args.directory
    reference_checker = load_module('group_check', work / 'expert-group-check.py')
    rank_checker = load_module('rank_check', work / 'expert-mask-rank.py')
    mask = json.loads((work / 'mask.json').read_text())
    group = json.loads((work / 'group.json').read_text())
    previous = json.loads((work / 'reference-group.json').read_text())
    assert (group['layer'], group['expert_id'], group['row'], group['group']) == (1, 498, 0, 0)
    for key in ('tensor', 'input_width', 'rows', 'positions', 'scale_byte', 'packed_bytes', 'codes', 'masks'):
        assert group[key] == previous[key]
    assert mask['plane'] == 2
    assert mask['original_mask'] == previous['masks'][2] == mask['regenerated_mask']
    assert rank_checker.rank_mask(mask['original_mask'], 32) == (mask['count'], mask['rank'])
    assert rank_checker.unrank_mask(mask['count'], mask['rank'], 32) == mask['regenerated_mask']
    assert mask['decoder_roundtrip_checks'] == 5249 and mask['invalid_inputs_rejected'] == 34
    assert mask['mask_bits_checked'] == 32 and mask['mask_bits_changed'] == 0
    rank_bits = (comb(32, mask['count']) - 1).bit_length()
    assert mask['payload_bits'] == 6 + rank_bits
    descriptor = bytes(mask['payload_bytes'])
    packed = int.from_bytes(descriptor, 'little')
    assert len(descriptor) == (mask['payload_bits'] + 7) // 8
    assert packed & 63 == mask['count'] and packed >> 6 == mask['rank']
    assert packed >> mask['payload_bits'] == 0

    assert len(group['codes']) == len(group['decoded_weights']) == len(previous['decoded_weights']) == 32
    rebuilt = bytearray(16)
    for coordinate in range(32):
        code = sum(((plane >> coordinate) & 1) << bit for bit, plane in enumerate(group['masks']))
        assert code == group['codes'][coordinate]
        rebuilt[coordinate // 2] |= code << (4 * (coordinate % 2))
        assert struct.pack('<f', group['decoded_weights'][coordinate]) == struct.pack('<f', previous['decoded_weights'][coordinate])
    assert bytes(rebuilt) == bytes(previous['packed_bytes'])
    assert group['decode_pairs_checked'] == 4096 and group['mask_coordinates_checked'] == 512
    assert group['group_weights_checked'] == 32 and group['group_weights_changed'] == 0
    assert group['packed_roundtrip_exact'] and group['projection_outputs_changed'] == 0
    assert len(group['projection_outputs']) == group['positions'] == 1
    assert len(previous['projection_outputs']) == 1
    for actual, recorded in zip(group['projection_outputs'], previous['projection_outputs']):
        assert actual['position'] == recorded['position'] == 0 and actual['rank'] == recorded['rank'] == 0
        assert reference_checker.bits(actual['reference']) == reference_checker.bits(actual['candidate'])
        assert reference_checker.bits(actual['reference']) == reference_checker.bits(recorded['reference'])

    reference = (work / 'reference.bin').read_bytes()
    assert hashlib.md5(reference).hexdigest() == reference_checker.BASELINE_MD5
    variants = {name: reference_checker.compare(reference, (work / (name + '.bin')).read_bytes())
                for name in ('disabled', 'mask-disabled', 'enabled')}
    repeated_matches = (work / 'reference-after.bin').read_bytes() == reference
    passed = mask['gate'] == group['gate'] == 'PASS' and repeated_matches
    passed = passed and all(variant['bit_exact'] for variant in variants.values())
    with (work / 'count-rank.bin').open('xb') as output:
        output.write(descriptor)
    names = ('baseline.c', 'candidate.c', 'group-base.h', 'group-base-lf.h',
             'expert-group-equation.h', 'expert-group-equation.patch',
             'expert-mask-consumer.h', 'expert-mask-consumer.patch',
             'expert-mask-consumer-check.py', 'expert-group-check.py', 'expert-mask-rank.py',
             'reference', 'candidate', 'mask.json', 'group.json', 'reference-group.json', 'count-rank.bin')
    result = {
        'host': 'AX102, AMD Ryzen 9 7950X3D',
        'prompt_ids': [1008, 10484, 318, 15383, 387],
        'build_flags': '-O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5 -lm',
        'settings': {'OMP_NUM_THREADS': '16', 'OMP_PROC_BIND': 'close', 'OMP_PLACES': 'cores',
                     'K3_PREFETCH': '4', 'K3_TRUNKRAM': '0', 'K3_NREADER': '14',
                     'K3_XDEC': '2', 'K3_PLGRAN': '1', 'K3_HUGE': '1'},
        'reference_md5': reference_checker.BASELINE_MD5,
        'mask': mask, 'group': group, 'variants': variants,
        'reference_repeated_matches': repeated_matches,
        'gate': 'PASS' if passed else 'FAIL',
        'sha256': {name: hashlib.sha256((work / name).read_bytes()).hexdigest() for name in names},
        'representation': {
            'original_plane_bits': 32, 'count_bits': 6, 'rank_bits': rank_bits,
            'logical_plane_bits': mask['payload_bits'], 'serialized_plane_bytes': len(descriptor),
            'serialization': 'Count in low six bits, rank above it, little endian; width 32 and plane 2 supplied by this scoped consumer.',
            'whole_group_compression_demonstrated': False,
        },
        'scope': 'One plane of one group: layer 1 expert 498 gate/w1 row 0 group 0, position 0 rank 0. Plane derived from loaded original weights, encoded, serialized and decoded in memory, then used by the replacement projection. No alternative checkpoint store, other prompt, decode or full-expert substitution.',
        'performance': 'Deferred. Preserving model results is mandatory; no speed, RAM or full-group storage saving claimed.',
    }
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()