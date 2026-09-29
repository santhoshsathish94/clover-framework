#!/usr/bin/env python3
"""Check stars-and-bars histogram ranking on saved expert-group evidence."""

import argparse
from collections import Counter
import hashlib
import importlib.util
from itertools import combinations
import json
from math import comb
from pathlib import Path


def histogram_rank(counts):
    assert len(counts) >= 2 and all(isinstance(count, int) and count >= 0 for count in counts)
    positions, prefix = [], 0
    for ordinal, count in enumerate(counts[:-1], 1):
        prefix += count
        positions.append(prefix + ordinal - 1)
    rank = sum(comb(position, ordinal) for ordinal, position in enumerate(positions, 1))
    return rank, positions


def histogram_unrank(rank, total, alphabet):
    assert total >= 0 and alphabet >= 2
    width, separators = total + alphabet - 1, alphabet - 1
    assert 0 <= rank < comb(width, separators)
    positions = [0] * separators
    upper = width - 1
    for ordinal in range(separators, 0, -1):
        position = upper
        while comb(position, ordinal) > rank:
            position -= 1
        positions[ordinal - 1] = position
        rank -= comb(position, ordinal)
        upper = position - 1
    assert rank == 0
    counts = [positions[0]]
    counts.extend(second - first - 1 for first, second in zip(positions, positions[1:]))
    counts.append(width - 1 - positions[-1])
    assert len(counts) == alphabet and sum(counts) == total
    return counts


def controls():
    checked = 0
    for bars in combinations(range(9), 3):
        counts = [bars[0], bars[1] - bars[0] - 1, bars[2] - bars[1] - 1, 8 - bars[2]]
        rank, generated_bars = histogram_rank(counts)
        assert generated_bars == list(bars) and histogram_unrank(rank, 6, 4) == counts
        checked += 1
    for code in range(16):
        counts = [0] * 16
        counts[code] = 32
        rank, bars = histogram_rank(counts)
        assert histogram_unrank(rank, 32, 16) == counts
        checked += 1
    total = comb(47, 15)
    for rank in sorted({0, 1, total // 2, total - 2, total - 1}):
        counts = histogram_unrank(rank, 32, 16)
        assert histogram_rank(counts)[0] == rank
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
    possibilities = comb(47, 15)
    histogram_bits = (possibilities - 1).bit_length()
    spec = importlib.util.spec_from_file_location('order_equations', args.order_checker)
    order = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(order)
    example = json.loads(args.example.read_text())
    counts = example['order']['counts']
    example_rank, bars = histogram_rank(counts)
    assert histogram_unrank(example_rank, 32, 16) == counts
    assert order.unrank_sequence(counts, int(example['order']['rank']))[0] == example['group']['codes']
    rows, fingerprints = [], {}
    repeated = 0
    for path in args.traces:
        with path.open() as trace:
            for line in trace:
                event = json.loads(line)
                if event['type'] != 'group':
                    continue
                key = tuple(event[name] for name in ('layer', 'expert', 'part', 'row', 'group'))
                fingerprint = (bytes(event['packed']), event['scale'])
                if key in fingerprints:
                    assert fingerprints[key] == fingerprint
                    repeated += 1
                    continue
                fingerprints[key] = fingerprint
                packed = fingerprint[0]
                codes = [(packed[position // 2] >> (4 * (position % 2))) & 15 for position in range(32)]
                counts = [codes.count(code) for code in range(16)]
                encoded_histogram, positions = histogram_rank(counts)
                restored_counts = histogram_unrank(encoded_histogram, 32, 16)
                assert restored_counts == counts
                arrangement = order.rank_sequence(codes, restored_counts)
                assert arrangement == int(event['rank'])
                rank_bits = (order.ways(restored_counts) - 1).bit_length()
                payload_bits = 8 + histogram_bits + rank_bits
                payload = event['scale'] | (encoded_histogram << 8) | (arrangement << (8 + histogram_bits))
                serialized = payload.to_bytes((payload_bits + 7) // 8, 'little')
                reloaded = int.from_bytes(serialized, 'little')
                assert reloaded >> payload_bits == 0 and reloaded & 255 == event['scale']
                restored_counts = histogram_unrank((reloaded >> 8) & ((1 << histogram_bits) - 1), 32, 16)
                restored_codes = order.unrank_sequence(restored_counts, reloaded >> (8 + histogram_bits))[0]
                assert restored_codes == codes
                rows.append({
                    'layer': key[0], 'expert': key[1], 'part': key[2], 'row': key[3], 'group': key[4],
                    'histogram_rank': str(encoded_histogram), 'arrangement_rank': str(arrangement),
                    'arrangement_bits': rank_bits,
                    'old_bytes': len(event['payload']), 'new_bytes': len(serialized),
                    'new_payload_bits': payload_bits,
                })
    size_counts = Counter(row['new_bytes'] for row in rows)
    summary = {
        'unique_groups': len(rows), 'unique_weights': len(rows) * 32,
        'repeated_group_records_not_double_counted': repeated,
        'histogram_and_order_reconstructions_exact': True,
        'previous_descriptor_bytes': sum(row['old_bytes'] for row in rows),
        'candidate_descriptor_bytes': sum(row['new_bytes'] for row in rows),
        'original_packed_group_bytes': len(rows) * 17,
        'candidate_byte_sizes': dict(sorted(size_counts.items())),
        'smaller_than_original_groups': sum(row['new_bytes'] < 17 for row in rows),
        'equal_to_original_groups': sum(row['new_bytes'] == 17 for row in rows),
        'larger_than_original_groups': sum(row['new_bytes'] > 17 for row in rows),
    }
    report = {
        'histogram_total': 32, 'alphabet': 16,
        'possible_histograms': possibilities, 'histogram_rank_bits': histogram_bits,
        'previous_histogram_bits': 96,
        'example': {'counts': example['order']['counts'], 'bar_positions': bars,
                    'histogram_rank': str(example_rank), 'arrangement_rank': example['order']['rank'],
                    'scale_byte': example['order']['scale_byte'],
                    'new_group_payload_bits': 8 + histogram_bits + example['order']['rank_bits'],
                    'new_group_bytes': (8 + histogram_bits + example['order']['rank_bits'] + 7) // 8},
        'finite_decoder_controls': checks, 'summary': summary,
        'trace_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.traces},
        'groups': rows,
        'scope': 'Mathematical identity and exact Python integer reconstruction/serialization on saved AX102 model data. No new model execution, modified projection, alternate checkpoint or throughput measurement.',
        'layout': 'Scale 8 bits, histogram rank fixed-width for C(47,15), then arrangement rank with width derived from decoded counts. No mode tag; fixed schema, length32/alphabet16 and externally supplied group location. Storage indexing/container overhead not included.',
    }
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({name: value for name, value in report.items() if name != 'groups'}, indent=2))


if __name__ == '__main__':
    main()