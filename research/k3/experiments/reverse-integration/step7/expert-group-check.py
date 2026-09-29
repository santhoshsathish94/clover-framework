#!/usr/bin/env python3
"""Verify the selected-expert group representation and full AX102 outputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


WIDTH = 7168
VOCAB = 163840
BASELINE_MD5 = '23d162dcefb18211a7540ef12948f1eb'


def bits(value):
    return struct.pack('<f', value)


def compare(reference, candidate):
    assert len(reference) == len(candidate) == (WIDTH + VOCAB) * 4
    expected = struct.unpack('<%df' % (WIDTH + VOCAB), reference)
    actual = struct.unpack('<%df' % (WIDTH + VOCAB), candidate)
    assert all(math.isfinite(value) for value in expected + actual)
    expected_words = struct.unpack('<%dI' % (WIDTH + VOCAB), reference)
    actual_words = struct.unpack('<%dI' % (WIDTH + VOCAB), candidate)
    result = {'bit_exact': reference == candidate, 'md5': hashlib.md5(candidate).hexdigest()}
    for name, start, stop in (('norm', 0, WIDTH), ('logits', WIDTH, WIDTH + VOCAB)):
        result[name] = {
            'count': stop - start,
            'changed': sum(expected_words[index] != actual_words[index] for index in range(start, stop)),
            'max_abs': max(abs(expected[index] - actual[index]) for index in range(start, stop)),
        }
    result['token'] = max(range(VOCAB), key=lambda index: actual[WIDTH + index])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    group = json.loads((args.directory / 'group.json').read_text())
    assert group['layer'] == 1 and group['row'] == 0 and group['group'] == 0
    assert group['input_width'] == 3584 and group['rows'] == 3072
    assert group['decode_pairs_checked'] == 4096 and group['mask_coordinates_checked'] == 512
    assert group['group_weights_checked'] == 32
    assert len(group['masks']) == 4 and len(group['packed_bytes']) == 16
    assert len(group['codes']) == len(group['decoded_weights']) == 32
    assert len(group['projection_outputs']) == group['positions']
    assert group['projection_outputs'][0]['position'] == group['projection_outputs'][0]['rank'] == 0

    masks = group['masks']
    scale = group['scale_byte']
    original_packed = bytes(group['packed_bytes'])
    magnitude_table = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    for coordinate in range(32):
        code = sum(((mask >> coordinate) & 1) << bit for bit, mask in enumerate(masks))
        original_code = (original_packed[coordinate // 2] >> (4 * (coordinate % 2))) & 15
        assert code == original_code == group['codes'][coordinate]
        magnitude = magnitude_table[original_code & 7]
        reference_weight = math.ldexp(magnitude, scale - 127) if scale != 255 else 0.0
        reference_weight = math.copysign(reference_weight, -1.0 if original_code & 8 else 1.0)
        assert bits(reference_weight) == bits(group['decoded_weights'][coordinate])
    for row in group['projection_outputs']:
        assert bits(row['reference']) == bits(row['candidate'])
    serialized = struct.pack('<4IB', *masks, scale)
    assert len(serialized) == 17 and struct.unpack('<4IB', serialized) == tuple(masks) + (scale,)
    with (args.directory / 'group-mask.bin').open('xb') as output:
        output.write(serialized)

    reference = (args.directory / 'reference.bin').read_bytes()
    assert hashlib.md5(reference).hexdigest() == BASELINE_MD5
    variants = {name: compare(reference, (args.directory / (name + '.bin')).read_bytes())
                for name in ('disabled', 'enabled')}
    repeated_matches = (args.directory / 'reference-after.bin').read_bytes() == reference
    passed = group['gate'] == 'PASS' and group['packed_roundtrip_exact'] and repeated_matches
    passed = passed and group['group_weights_changed'] == 0 and group['projection_outputs_changed'] == 0
    passed = passed and all(variant['bit_exact'] for variant in variants.values())
    names = ('baseline.c', 'candidate.c', 'expert-group-equation.h', 'expert-group-equation.patch',
             'reference', 'candidate', 'group.json', 'group-mask.bin', 'expert-group-check.py')
    result = {
        'host': 'AX102, AMD Ryzen 9 7950X3D',
        'prompt_ids': [1008, 10484, 318, 15383, 387],
        'build_flags': '-O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5 -lm',
        'settings': {'OMP_NUM_THREADS': '16', 'OMP_PROC_BIND': 'close', 'OMP_PLACES': 'cores',
                     'K3_PREFETCH': '4', 'K3_TRUNKRAM': '0', 'K3_NREADER': '14',
                     'K3_XDEC': '2', 'K3_PLGRAN': '1', 'K3_HUGE': '1'},
        'reference_md5': BASELINE_MD5, 'group': group, 'variants': variants,
        'reference_repeated_matches': repeated_matches,
        'representation': {'old_payload_bytes': 17, 'new_payload_bytes': len(serialized),
                           'serialization': 'Four little-endian uint32 masks in bit order 0..3, then one uint8 scale. Coordinate identity supplied by the caller.'},
        'sha256': {name: hashlib.sha256((args.directory / name).read_bytes()).hexdigest() for name in names},
        'gate': 'PASS' if passed else 'FAIL',
        'scope': 'One real selected expert gate-matrix row, first group of 32. Formula decode domain checked exhaustively; no other expert groups or prompts replaced. The candidate recomputes the complete affected row, uses original decoding for its remaining weights, and supplies its result to the original SiTU/downstream model.',
        'performance': 'Deferred. No smaller checkpoint, cache or faster evaluation claim. Original learned codes and scale remain necessary; masks encode the same information.',
    }
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()