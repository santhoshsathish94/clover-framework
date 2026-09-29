#!/usr/bin/env python3
"""Check an exact signed, scaled dictionary equation on AX102 vector captures."""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


WIDTH = 7168
VECTOR_BYTES = WIDTH * 4


def factor(word):
    sign = -1 if word >> 31 else 1
    exponent_field = (word >> 23) & 255
    mantissa = word & 0x7FFFFF
    assert exponent_field != 255
    if exponent_field:
        mantissa |= 1 << 23
        exponent = exponent_field - 150
    else:
        exponent = -149
    if not mantissa:
        return sign, 0, 0
    trailing = (mantissa & -mantissa).bit_length() - 1
    return sign, mantissa >> trailing, exponent + trailing


def scaled_integer(number, exponent):
    if exponent >= 0:
        return Fraction(number << exponent)
    return Fraction(number, 1 << -exponent)


def byte_count(bits):
    return (bits + 7) // 8


def inspect(payload, layer, position):
    words = struct.unpack('<%dI' % WIDTH, payload)
    values = struct.unpack('<%df' % WIDTH, payload)
    parts = [factor(word) for word in words]
    table = sorted({mantissa for sign, mantissa, exponent in parts})
    indices = {mantissa: index for index, mantissa in enumerate(table)}
    codes = [indices[mantissa] for sign, mantissa, exponent in parts]
    restored = [sign * math.ldexp(float(table[code]), exponent)
                for (sign, mantissa, exponent), code in zip(parts, codes)]
    assert struct.pack('<%df' % WIDTH, *restored) == payload

    minimum_power = min(exponent for sign, mantissa, exponent in parts)
    maximum_power = max(exponent for sign, mantissa, exponent in parts)
    power_bits = (maximum_power - minimum_power).bit_length()
    index_bits = (len(table) - 1).bit_length()
    table_bits = max(mantissa.bit_length() for mantissa in table)
    payload_bits = WIDTH * (1 + power_bits + index_bits) + len(table) * table_bits

    implicit_bytes = VECTOR_BYTES
    implicit_code_bits = None
    if all(mantissa for sign, mantissa, exponent in parts):
        implicit_codes = [(mantissa - 1) // 2 for sign, mantissa, exponent in parts]
        generated = [sign * math.ldexp(float(2 * code + 1), exponent)
                     for (sign, mantissa, exponent), code in zip(parts, implicit_codes)]
        assert struct.pack('<%df' % WIDTH, *generated) == payload
        implicit_code_bits = max(code.bit_length() for code in implicit_codes)
        implicit_bytes = 8 + byte_count(WIDTH * (1 + power_bits + implicit_code_bits))

    direct_square_sum = sum((Fraction(value) ** 2 for value in values), Fraction(0))
    exponent_counts = Counter((code, exponent) for code, (sign, mantissa, exponent) in zip(codes, parts))
    grouped_square_sum = sum((count * scaled_integer(table[code] ** 2, 2 * exponent)
                              for (code, exponent), count in exponent_counts.items()), Fraction(0))
    assert direct_square_sum == grouped_square_sum
    row = {
        'layer': layer, 'position': position,
        'distinct_raw_words': len(set(words)),
        'table_entries': len(table),
        'reused_positions': WIDTH - len(table),
        'most_frequent_entry_count': max(Counter(codes).values()),
        'power_minimum': minimum_power, 'power_maximum': maximum_power,
        'power_bits': power_bits, 'index_bits': index_bits, 'table_entry_bits': table_bits,
        'dictionary_format_bytes': 8 + byte_count(payload_bits),
        'implicit_odd_table_code_bits': implicit_code_bits,
        'implicit_odd_table_format_bytes': implicit_bytes,
        'raw_bytes': VECTOR_BYTES,
        'reconstruction_bit_exact': True, 'norm_square_sum_rational_exact': True,
    }
    return row, parts


def summarize(rows):
    return {
        'vectors': len(rows),
        'table_entries_minimum': min(row['table_entries'] for row in rows),
        'table_entries_median': statistics.median(row['table_entries'] for row in rows),
        'table_entries_maximum': max(row['table_entries'] for row in rows),
        'reused_positions_total': sum(row['reused_positions'] for row in rows),
        'components': WIDTH * len(rows),
        'power_bits_median': statistics.median(row['power_bits'] for row in rows),
        'dictionary_bytes_total': sum(row['dictionary_format_bytes'] for row in rows),
        'dictionary_ratio_to_f32': sum(row['dictionary_format_bytes'] for row in rows) / (VECTOR_BYTES * len(rows)),
        'implicit_table_code_bits_median': statistics.median(row['implicit_odd_table_code_bits'] for row in rows if row['implicit_odd_table_code_bits'] is not None),
        'implicit_table_bytes_total': sum(row['implicit_odd_table_format_bytes'] for row in rows),
        'implicit_table_ratio_to_f32': sum(row['implicit_odd_table_format_bytes'] for row in rows) / (VECTOR_BYTES * len(rows)),
        'most_frequent_entry_count_maximum': max(row['most_frequent_entry_count'] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layers', type=Path)
    parser.add_argument('--positions', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.positions >= 2
    data = args.layers.read_bytes()
    assert len(data) == 93 * args.positions * VECTOR_BYTES
    rows, training_table, heldout = [], set(), Counter()
    layer_training, layer_heldout = {}, {}
    for index in range(93 * args.positions):
        layer, position = divmod(index, args.positions)
        row, parts = inspect(data[index * VECTOR_BYTES:(index + 1) * VECTOR_BYTES], layer, position)
        rows.append(row)
        if layer == 0:
            continue
        mantissas = [mantissa for sign, mantissa, exponent in parts]
        if position == 0:
            layer_training[layer] = set(mantissas)
            training_table.update(mantissas)
        else:
            heldout.update(mantissas)
            layer_heldout.setdefault(layer, Counter()).update(mantissas)

    hits = sum(count for mantissa, count in heldout.items() if mantissa in training_table)
    same_layer_hits = sum(count for layer, counts in layer_heldout.items()
                          for mantissa, count in counts.items() if mantissa in layer_training[layer])
    heldout_count = sum(heldout.values())
    same_layer_null_hits = sum(sum(counts.values()) * len(layer_training[layer]) / 2**23
                               for layer, counts in layer_heldout.items())
    report = {
        'equation': 'x_i = sign_i * 2**exponent_i * table[code_i]; nonzero table entries are odd unsigned integers',
        'capture_sha256': hashlib.sha256(data).hexdigest(),
        'positions': args.positions, 'width': WIDTH,
        'dictionary_accounting': 'Bit-packed fields: one sign, per-vector bounded exponent, table index per component; unsigned table entries at maximum entry width; 8-byte header. Analytical format size, not a measured codec speed.',
        'implicit_table_accounting': 'For nonzero vectors, table[c] = 2*c+1; store sign, bounded exponent and universal odd-integer code per component, plus 8-byte header; no table entries. Vectors containing zero use raw fallback. Analytical format size only.',
        'summary': {
            'embedding': summarize([row for row in rows if row['layer'] == 0]),
            'later_aggregates': summarize([row for row in rows if row['layer'] > 0]),
        },
        'shared_table': {
            'training': 'Position 0 from layers 1..92 of the same captured prompt',
            'evaluation': 'Positions 1..4 from layers 1..92; no unseen-prompt claim',
            'entries': len(training_table), 'payload_bytes_at_24_bits': len(training_table) * 3,
            'index_bits': (len(training_table) - 1).bit_length(),
            'heldout_components': heldout_count, 'covered_components': hits,
            'coverage': hits / heldout_count,
            'same_layer_covered_components': same_layer_hits,
            'same_layer_coverage': same_layer_hits / heldout_count,
            'conditional_uniform_odd_integer_null_coverage': len(training_table) / 2**23,
            'same_layer_conditional_uniform_null_coverage': same_layer_null_hits / heldout_count,
            'null_scope': 'Uniform over the 2**23 possible positive odd integers below 2**24; an explicit comparison model, not a proven activation distribution.',
            'uncovered_require_extra_information': heldout_count - hits,
        },
        'all_reconstructions_bit_exact': all(row['reconstruction_bit_exact'] for row in rows),
        'all_norm_square_sums_rational_exact': all(row['norm_square_sum_rational_exact'] for row in rows),
        'vectors': rows,
        'scope': 'Real captured vectors, exact rational square-sum checks. No projection-kernel execution, model replacement, speed claim, or proof against other generating equations.',
    }
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({name: value for name, value in report.items() if name != 'vectors'}, indent=2))


if __name__ == '__main__':
    main()