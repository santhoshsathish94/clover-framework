#!/usr/bin/env python3
"""Evaluate modal-offset exception coding on complete saved AX102 scale rows."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from math import comb
from pathlib import Path


def rank_positions(positions):
    return sum(comb(position, ordinal) for ordinal, position in enumerate(positions, 1))


def unrank_positions(rank, count, length):
    assert 0 <= count <= length and 0 <= rank < comb(length, count)
    positions = [0] * count
    upper = length - 1
    for ordinal in range(count, 0, -1):
        position = upper
        while comb(position, ordinal) > rank:
            position -= 1
        positions[ordinal - 1] = position
        rank -= comb(position, ordinal)
        upper = position - 1
    assert rank == 0
    return positions


def describe(scales):
    assert 0 < len(scales) <= 112
    base, maximum = min(scales), max(scales)
    width = (maximum - base).bit_length()
    offsets = [scale - base for scale in scales]
    frequencies = Counter(offsets)
    usual = min(frequencies, key=lambda value: (-frequencies[value], value))
    positions = [index for index, value in enumerate(offsets) if value != usual]
    count = len(positions)
    position_rank = rank_positions(positions)
    radix = (1 << width) - 1
    value_rank, multiplier = 0, 1
    for position in positions:
        value = offsets[position]
        digit = value if value < usual else value - 1
        assert 0 <= digit < radix
        value_rank += digit * multiplier
        multiplier *= radix
    position_bits = (comb(len(scales), count) - 1).bit_length()
    value_bits = (multiplier - 1).bit_length()
    count_bits = len(scales).bit_length()
    return {
        'base': base, 'width': width, 'usual_offset': usual, 'usual_scale': base + usual,
        'exception_count': count, 'exception_positions': positions,
        'position_rank': position_rank, 'position_rank_bits': position_bits,
        'exception_value_rank': value_rank, 'exception_value_bits': value_bits,
        'exception_count_bits': count_bits,
        'fixed_bits': 12 + len(scales) * width,
        'exception_bits': 12 + width + count_bits + position_bits + value_bits,
    }, offsets


def encode(scales, format_name):
    info, offsets = describe(scales)
    exceptions = format_name == 'exceptions' or (format_name == 'adaptive' and info['exception_bits'] < info['fixed_bits'])
    fields = [(int(exceptions), 1)] if format_name == 'adaptive' else []
    fields.extend(((info['base'], 8), (info['width'], 4)))
    if exceptions:
        fields.extend(((info['usual_offset'], info['width']),
                       (info['exception_count'], info['exception_count_bits']),
                       (info['position_rank'], info['position_rank_bits']),
                       (info['exception_value_rank'], info['exception_value_bits'])))
    else:
        fields.extend((value, info['width']) for value in offsets)
    packed, offset = 0, 0
    for value, bits in fields:
        assert 0 <= value < 1 << bits
        packed |= value << offset
        offset += bits
    return packed.to_bytes((offset + 7) // 8, 'little'), offset, exceptions


def decode(payload, length, format_name):
    assert 0 < length <= 112
    packed, offset = int.from_bytes(payload, 'little'), 0

    def take(bits):
        nonlocal offset
        assert offset + bits <= len(payload) * 8
        result = (packed >> offset) & ((1 << bits) - 1)
        offset += bits
        return result

    exceptions = bool(take(1)) if format_name == 'adaptive' else format_name == 'exceptions'
    base, width = take(8), take(4)
    assert width <= 8
    if not exceptions:
        offsets = [take(width) for index in range(length)]
    else:
        usual, count = take(width), take(length.bit_length())
        assert count <= length
        positions = unrank_positions(take((comb(length, count) - 1).bit_length()), count, length)
        radix = (1 << width) - 1
        assert radix > 0 or count == 0
        possibilities = radix ** count
        value_rank = take((possibilities - 1).bit_length())
        assert value_rank < possibilities
        offsets = [usual] * length
        for position in positions:
            digit = value_rank % radix
            value_rank //= radix
            offsets[position] = digit if digit < usual else digit + 1
        assert value_rank == 0
    assert len(payload) == (offset + 7) // 8 and packed >> offset == 0
    scales = [base + value for value in offsets]
    assert all(value <= 255 for value in scales)
    return bytes(scales)


def controls():
    cases = [bytes(120 + ((mask >> position) & 1) for position in range(8)) for mask in range(256)]
    cases.extend(bytes((0, (1 << width) - 1, 0, (1 << width) - 1)) for width in range(9))
    cases.extend(bytes([value]) * 112 for value in (0, 121, 255))
    for scales in cases:
        for format_name in ('fixed', 'exceptions', 'adaptive'):
            payload, bits, exceptions = encode(scales, format_name)
            assert decode(payload, len(scales), format_name) == scales
    return len(cases) * 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, nargs='+', required=True)
    parser.add_argument('--baseline-codec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checks = controls()
    spec = importlib.util.spec_from_file_location('previous_scale_codec', args.baseline_codec)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    seen, rows, repeated = {}, [], 0
    for path in args.traces:
        with path.open() as trace:
            for line in trace:
                event = json.loads(line)
                if event['type'] != 'scale_row':
                    continue
                key = tuple(event[field] for field in ('layer', 'expert', 'part', 'row'))
                scales = bytes(event['original_scales'])
                original_payload = bytes(event['payload'])
                assert baseline.decode_scales(original_payload, len(scales)) == scales
                assert baseline.encode_scales(scales)[0] == original_payload
                if key in seen:
                    assert seen[key] == (scales, original_payload)
                    repeated += 1
                    continue
                seen[key] = (scales, original_payload)
                info, offsets = describe(scales)
                encoded = {}
                for format_name in ('fixed', 'exceptions', 'adaptive'):
                    payload, bits, uses_exceptions = encode(scales, format_name)
                    assert decode(payload, len(scales), format_name) == scales
                    if format_name == 'fixed':
                        assert payload == original_payload
                    encoded[format_name] = {'bytes': len(payload), 'bits': bits, 'uses_exceptions': uses_exceptions}
                row = dict(zip(('layer', 'expert', 'part', 'row'), key))
                row.update({'length': len(scales), 'base': info['base'], 'width': info['width'],
                            'usual_scale': info['usual_scale'], 'exception_count': info['exception_count'],
                            'position_rank': str(info['position_rank']), 'exception_value_rank': str(info['exception_value_rank']),
                            'formats': encoded, 'scale_bytes_sha256': hashlib.sha256(scales).hexdigest()})
                if key == (1, 498, 0, 0):
                    row['example'] = {**info, 'original_scales': list(scales), 'offsets': offsets}
                    row['example']['position_rank'] = str(info['position_rank'])
                    row['example']['exception_value_rank'] = str(info['exception_value_rank'])
                rows.append(row)
    assert rows
    totals = {name: sum(row['formats'][name]['bytes'] for row in rows) for name in ('fixed', 'exceptions', 'adaptive')}
    classifications = Counter('smaller' if row['formats']['adaptive']['bytes'] < row['formats']['fixed']['bytes'] else
                              'equal' if row['formats']['adaptive']['bytes'] == row['formats']['fixed']['bytes'] else 'larger'
                              for row in rows)
    scales_total = sum(row['length'] for row in rows)
    report = {
        'equation': 'scale_i = base + usual_offset except at a count/rank-generated position set; ranked non-default offset digits restore exception values.',
        'finite_roundtrip_checks': checks, 'unique_rows': len(rows), 'unique_scales': scales_total,
        'duplicate_rows_verified_and_excluded': repeated, 'all_formats_roundtrip_byte_exact': True,
        'summary': {'original_scale_bytes': scales_total, 'format_bytes': totals,
                    'adaptive_vs_fixed_rows': dict(classifications),
                    'adaptive_exception_rows': sum(row['formats']['adaptive']['uses_exceptions'] for row in rows),
                    'unchanged_code_bytes': 16 * scales_total,
                    'fixed_code_plus_scale_bytes': 16 * scales_total + totals['fixed'],
                    'adaptive_code_plus_scale_bytes': 16 * scales_total + totals['adaptive']},
        'example': next(row for row in rows if 'example' in row),
        'rows': rows,
        'trace_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.traces},
        'layout': 'Shared row length (96 or112 here). Adaptive starts with one raw/exception tag; both have base8,width4. Raw stores length*width bits. Exception stores usual_offset in width bits, count in bit_length(length) bits, colex position rank in ceil(log2(C(length,count))) bits, then ordered exception digits in radix(2**width-1), skipping the usual offset. Digit rank uses ceil(log2(radix**count)) bits; count0 has no position/value payload. Each row byte padded; indexing/container overhead excluded.',
        'scope': 'Exact encode/serialize/decode on saved complete AX102 scale rows, not a new model consumer or speed/RAM/checkpoint measurement. Default chosen by greatest frequency with smallest-offset tie break. No earlier checks modified.',
    }
    with args.output.open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))


if __name__ == '__main__':
    main()