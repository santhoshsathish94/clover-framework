#!/usr/bin/env python3
"""Check a histogram-major joint rank without hiding the original code capacity."""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
from itertools import product
import json
from math import comb, factorial, prod
from pathlib import Path


def arrangements(counts):
    return factorial(sum(counts)) // prod(factorial(count) for count in counts)


def joint_rank(counts, arrangement_rank):
    assert len(counts) >= 2 and all(isinstance(count, int) and count >= 0 for count in counts)
    assert 0 <= arrangement_rank < arrangements(counts)
    remaining = sum(counts)
    multiplicity = 1
    offset = 0
    for code, count in enumerate(counts[:-1]):
        free_symbols = len(counts) - code - 1
        for alternative in range(count):
            offset += multiplicity * comb(remaining, alternative) * free_symbols ** (remaining - alternative)
        multiplicity *= comb(remaining, count)
        remaining -= count
    assert remaining == counts[-1] and multiplicity == arrangements(counts)
    return offset + arrangement_rank


def joint_unrank(rank, total, alphabet):
    assert total >= 0 and alphabet >= 2 and 0 <= rank < alphabet ** total
    remaining, multiplicity = total, 1
    counts = []
    for code in range(alphabet - 1):
        free_symbols = alphabet - code - 1
        for count in range(remaining + 1):
            block = multiplicity * comb(remaining, count) * free_symbols ** (remaining - count)
            if rank < block:
                counts.append(count)
                multiplicity *= comb(remaining, count)
                remaining -= count
                break
            rank -= block
        else:
            raise AssertionError('No histogram block selected')
    counts.append(remaining)
    assert 0 <= rank < arrangements(counts)
    return counts, rank


def controls():
    buckets = defaultdict(list)
    for sequence in product(range(3), repeat=4):
        counts = tuple(sequence.count(code) for code in range(3))
        buckets[counts].append(sequence)
    offset = 0
    for counts, sequences in sorted(buckets.items()):
        assert len(sequences) == arrangements(counts)
        for rank, sequence in enumerate(sequences):
            encoded = joint_rank(counts, rank)
            assert encoded == offset + rank
            recovered_counts, recovered_rank = joint_unrank(encoded, 4, 3)
            assert tuple(recovered_counts) == counts and sequences[recovered_rank] == sequence
        offset += len(sequences)
    assert offset == 3**4
    checked = offset
    for code in range(16):
        counts = [0] * 16
        counts[code] = 32
        assert joint_unrank(joint_rank(counts, 0), 32, 16) == (counts, 0)
        checked += 1
    counts = [2] * 16
    for rank in (0, arrangements(counts) // 2, arrangements(counts) - 1):
        assert joint_unrank(joint_rank(counts, rank), 32, 16) == (counts, rank)
        checked += 1
    for rank in (0, 16**32 - 1):
        counts, arrangement = joint_unrank(rank, 32, 16)
        assert joint_rank(counts, arrangement) == rank
        checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', type=Path, required=True)
    parser.add_argument('--order-checker', type=Path, required=True)
    parser.add_argument('--traces', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checks = controls()
    spec = importlib.util.spec_from_file_location('order_equations', args.order_checker)
    order = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(order)
    example = json.loads(args.example.read_text())
    example_counts = example['order']['counts']
    example_arrangement = int(example['order']['rank'])
    example_joint = joint_rank(example_counts, example_arrangement)
    recovered_counts, recovered_arrangement = joint_unrank(example_joint, 32, 16)
    assert recovered_counts == example_counts and recovered_arrangement == example_arrangement
    assert order.unrank_sequence(recovered_counts, recovered_arrangement)[0] == example['group']['codes']
    seen, rows, repeated = {}, [], 0
    for path in args.traces:
        with path.open() as trace:
            for line in trace:
                event = json.loads(line)
                if event['type'] != 'group':
                    continue
                key = tuple(event[name] for name in ('layer', 'expert', 'part', 'row', 'group'))
                fingerprint = (bytes(event['packed']), event['scale'])
                if key in seen:
                    assert seen[key] == fingerprint
                    repeated += 1
                    continue
                seen[key] = fingerprint
                codes = [(fingerprint[0][position // 2] >> (4 * (position % 2))) & 15 for position in range(32)]
                counts = [codes.count(code) for code in range(16)]
                arrangement = order.rank_sequence(codes, counts)
                assert arrangement == int(event['rank'])
                combined = joint_rank(counts, arrangement)
                assert 0 <= combined < 1 << 128
                payload = ((combined << 8) | event['scale']).to_bytes(17, 'little')
                reloaded = int.from_bytes(payload, 'little')
                assert reloaded & 255 == event['scale']
                restored_counts, restored_arrangement = joint_unrank(reloaded >> 8, 32, 16)
                assert restored_counts == counts and restored_arrangement == arrangement
                assert order.unrank_sequence(restored_counts, restored_arrangement)[0] == codes
                rows.append({'layer': key[0], 'expert': key[1], 'part': key[2], 'row': key[3], 'group': key[4],
                             'joint_rank': str(combined),
                             'separate_rank_bytes': (48 + (arrangements(counts) - 1).bit_length() + 7) // 8,
                             'joint_bytes': len(payload)})
    report = {
        'equation': 'J = sum(N(h_prime) for histograms h_prime lexicographically before h) + arrangement_rank',
        'histogram_order': 'Lexicographic order of the count vector h_0..h_15; distinct from the earlier colexicographic stars-and-bars histogram rank.',
        'total_code_sequences': str(16**32), 'fixed_code_bits': 128, 'scale_bits': 8,
        'group_bytes': 17, 'finite_control_checks': checks,
        'example': {'counts': example_counts, 'arrangement_rank': str(example_arrangement),
                    'histogram_block_offset': str(example_joint - example_arrangement),
                    'joint_rank': str(example_joint), 'scale': example['order']['scale_byte'],
                    'old_example_payload_bits': 132, 'joint_fixed_payload_bits': 136},
        'summary': {'unique_groups': len(rows), 'unique_weights': len(rows) * 32,
                    'repeated_records_not_double_counted': repeated,
                    'all_histograms_arrangements_and_codes_exact': True,
                    'separate_rank_bytes': sum(row['separate_rank_bytes'] for row in rows),
                    'joint_rank_bytes': sum(row['joint_bytes'] for row in rows),
                    'original_packed_bytes': len(rows) * 17},
        'trace_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.traces},
        'groups': rows,
        'scope': 'Integer representation checks on saved AX102 group data; no new projection or model run. Joint ranking returns to original fixed-width code capacity, not universal compression below original.',
        'layout': 'One scale byte then a fixed 128-bit joint rank, little endian. Shared decoder/schema, alphabet16 and length32; group location supplied externally. Model weights remain necessary information.',
    }
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({name: value for name, value in report.items() if name != 'groups'}, indent=2))


if __name__ == '__main__':
    main()