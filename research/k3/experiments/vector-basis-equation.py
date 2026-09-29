#!/usr/bin/env python3
"""Validate retained-source vector equations on a captured K3 aggregation."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct


def float32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def dot(left, right):
    return sum((first * second for first, second in zip(left, right)), Fraction(0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = args.capture.read_bytes()
    magic, width, count, layer = struct.unpack_from('<4I', payload)
    assert magic == 0x31534256 and width == 7168 and count == 9 and layer == 84
    assert len(payload) == 16 + 4 * (count + (count + 2) * width)
    coefficients = struct.unpack_from('<%df' % count, payload, 16)
    offset = 16 + 4 * count
    sources = []
    for source in range(count):
        sources.append(struct.unpack_from('<%df' % width, payload, offset))
        offset += 4 * width
    direction = struct.unpack_from('<%df' % width, payload, offset)
    offset += 4 * width
    actual = struct.unpack_from('<%df' % width, payload, offset)

    rounded = []
    for coordinate in range(width):
        total = 0.0
        for coefficient, source in zip(coefficients, sources):
            total = float32(total + float32(coefficient * source[coordinate]))
        rounded.append(total)
    assert struct.pack('<%df' % width, *rounded) == payload[offset:]

    exact_coefficients = [Fraction(value) for value in coefficients]
    exact_sources = [[Fraction(value) for value in source] for source in sources]
    exact_direction = [Fraction(value) for value in direction]
    exact_vector = [sum((coefficient * source[coordinate]
                        for coefficient, source in zip(exact_coefficients, exact_sources)), Fraction(0))
                    for coordinate in range(width)]
    gram = [[Fraction(0) for column in range(count)] for row in range(count)]
    for row in range(count):
        for column in range(row, count):
            gram[row][column] = dot(exact_sources[row], exact_sources[column])
            gram[column][row] = gram[row][column]
    direct_norm_squared = dot(exact_vector, exact_vector)
    basis_norm_squared = sum((exact_coefficients[row] * exact_coefficients[column] * gram[row][column]
                             for row in range(count) for column in range(count)), Fraction(0))
    assert direct_norm_squared == basis_norm_squared

    direct_projection = dot(exact_direction, exact_vector)
    source_projections = [dot(exact_direction, source) for source in exact_sources]
    basis_projection = dot(exact_coefficients, source_projections)
    assert direct_projection == basis_projection
    deviations = [abs(Fraction(value) - exact) for value, exact in zip(actual, exact_vector)]

    epsilon = Fraction(float32(1e-5))
    assert direct_norm_squared / width + epsilon == basis_norm_squared / width + epsilon
    report = {
        'capture_sha256': hashlib.sha256(payload).hexdigest(),
        'layer': layer, 'site': 'First nine-source AR at layer 84: pre-MLP, position 0',
        'width': width, 'source_count': count,
        'coefficients': list(coefficients),
        'coefficients_sum': str(sum(exact_coefficients)),
        'current_float32_aggregation_reconstructed_bit_exact': True,
        'real_arithmetic_gram_identity_exact': True,
        'real_arithmetic_projection_identity_exact': True,
        'real_arithmetic_rms_denominator_squared_identity_exact': True,
        'projection_row': 'Captured learned AR fold; used as an actual linear functional, not a test of Q or router kernels',
        'direct_norm_squared': str(direct_norm_squared),
        'direct_projection': str(direct_projection),
        'exact_sum_vs_current_float32': {
            'different_components': sum(deviation != 0 for deviation in deviations),
            'components': width,
            'max_abs_difference': float(max(deviations)),
            'scope': 'Exact rational mixture versus current staged float32 sum. Not model logit differences.',
        },
        'representation_counts': {
            'explicit_output_floats': width,
            'coefficient_floats': count,
            'source_references': count,
            'retained_source_floats': width * count,
            'gram_unique_entries': count * (count + 1) // 2,
        },
        'scope': 'One real aggregation. Algebraic identities also follow for arbitrary compatible matrices; no end-to-end alternative evaluator or speed comparison.',
    }
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()