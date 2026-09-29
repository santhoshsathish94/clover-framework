#!/usr/bin/env python3
"""Validate compact histogram traces and reuse unchanged coverage/output gates."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_trace(work, audit, name, histogram, order):
    expected_bits = 96 if name == 'legacy' else 40
    groups, summary, sizes = {}, None, []
    with (work / (name + '.jsonl')).open() as original, (audit / (name + '.jsonl')).open('x') as expanded:
        for line in original:
            event = json.loads(line)
            assert summary is None
            translated = dict(event)
            if event['type'] == 'summary':
                summary = event
                assert event['histogram_bits'] == expected_bits
                assert event['histogram_decoder_checks'] == (105 if expected_bits == 40 else 0)
                assert event['histogram_invalid_inputs_rejected'] == (4 if expected_bits == 40 else 0)
            elif event['type'] == 'group':
                assert event['histogram_bits'] == expected_bits
                packed = bytes(event['packed'])
                assert len(packed) == 16
                codes = [(packed[position // 2] >> (4 * (position % 2))) & 15 for position in range(32)]
                payload = bytes(event['payload'])
                value = int.from_bytes(payload, 'little')
                assert len(payload) == (event['payload_bits'] + 7) // 8
                assert value >> event['payload_bits'] == 0 and value & 255 == event['scale']
                value >>= 8
                if expected_bits == 40:
                    histogram_rank = value & ((1 << 40) - 1)
                    counts = histogram.histogram_unrank(histogram_rank, 32, 16)
                    assert histogram.histogram_rank(counts)[0] == histogram_rank
                    value >>= 40
                else:
                    counts = []
                    for code in range(16):
                        counts.append(value & 63)
                        value >>= 6
                assert counts == [codes.count(code) for code in range(16)]
                assert value == int(event['rank']) == order.rank_sequence(codes, counts)
                assert order.unrank_sequence(counts, value)[0] == codes
                rank_bits = (order.ways(counts) - 1).bit_length()
                assert event['payload_bits'] == 8 + expected_bits + rank_bits
                key = tuple(event[field] for field in ('layer', 'expert', 'part', 'row', 'group'))
                assert key not in groups
                groups[key] = (packed, event['scale'], payload)
                sizes.append(len(payload))

                expanded_value = event['scale']
                for code, count in enumerate(counts):
                    expanded_value |= count << (8 + 6 * code)
                expanded_value |= value << 104
                expanded_bits = 104 + rank_bits
                translated['payload'] = list(expanded_value.to_bytes((expanded_bits + 7) // 8, 'little'))
                translated['payload_bits'] = expanded_bits
            else:
                assert event['type'] == 'projection'
            expanded.write(json.dumps(translated, separators=(',', ':')) + '\n')
    assert summary and summary['gate'] == 'PASS'
    return groups, sizes, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--runs', nargs='+', choices=('smoke', 'legacy', 'france', 'japan'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = args.directory
    audit = args.output.with_suffix('.audit')
    audit.mkdir()
    coverage = load('unchanged_coverage_check', work / 'expert-group-coverage-check.py')
    order = load('unchanged_order_check', work / 'expert-group-order-check.py')
    output_check = load('unchanged_output_check', work / 'expert-group-check.py')
    histogram = load('python_histogram_check', work / 'expert-histogram-rank.py')
    assert hashlib.md5((work / 'france-reference.bin').read_bytes()).hexdigest() == output_check.BASELINE_MD5
    assert (work / 'france-disabled.bin').read_bytes() == (work / 'france-reference.bin').read_bytes()

    runs, unique_groups, all_pairs = [], {}, set()
    for name in args.runs:
        reference = 'japan' if name == 'japan' else 'france'
        for filename in (reference + '-reference.bin', reference + '-reference.routes', name + '.bin', name + '.routes'):
            if not (audit / filename).exists():
                os.link(work / filename, audit / filename)
        groups, sizes, summary = audit_trace(work, audit, name, histogram, order)
        result, checked_groups, pairs = coverage.check_run(audit, name, reference, 1 if name == 'smoke' else 2, order, output_check)
        assert set(groups) == checked_groups
        result['raw_trace_sha256'] = hashlib.sha256((work / (name + '.jsonl')).read_bytes()).hexdigest()
        result['validated_expanded_audit_trace_sha256'] = result.pop('trace_sha256')
        result['descriptor_bytes'] = {'minimum': min(sizes), 'maximum': max(sizes), 'total': sum(sizes), 'original_total': len(sizes) * 17}
        result['histogram_bits'] = summary['histogram_bits']
        result['histogram_decoder_checks'] = summary['histogram_decoder_checks']
        result['histogram_invalid_inputs_rejected'] = summary['histogram_invalid_inputs_rejected']
        result['compact_payloads_independently_verified'] = name != 'legacy'
        runs.append(result)
        if name != 'legacy':
            all_pairs.update(pairs)
            for key, fingerprint in groups.items():
                if key in unique_groups:
                    assert unique_groups[key] == fingerprint
                else:
                    unique_groups[key] = fingerprint
    repeated = work / 'france-reference-after.bin'
    repeated_matches = repeated.read_bytes() == (work / 'france-reference.bin').read_bytes() if repeated.exists() else None
    assert repeated_matches is not False
    unique_sizes = [len(fingerprint[2]) for fingerprint in unique_groups.values()]
    names = ('baseline.c', 'candidate.c', 'coverage-base.h', 'coverage-base-lf.h',
             'expert-group-coverage.h', 'expert-group-coverage.patch',
             'expert-group-order.h', 'expert-group-equation.h',
             'expert-histogram-codec.h', 'expert-histogram-integrated.patch',
             'expert-histogram-integrated-check.py', 'expert-histogram-rank.py',
             'expert-group-coverage-check.py', 'expert-group-order-check.py', 'expert-group-check.py',
             'reference', 'candidate')
    report = {
        'host': 'AX102, AMD Ryzen 9 7950X3D',
        'build_flags': '-O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5 -lm',
        'settings': {'OMP_NUM_THREADS': '16', 'OMP_PROC_BIND': 'close', 'OMP_PLACES': 'cores',
                     'K3_PREFETCH': '4', 'K3_TRUNKRAM': '0', 'K3_NREADER': '14',
                     'K3_XDEC': '2', 'K3_PLGRAN': '1', 'K3_HUGE': '1'},
        'integrated_fields': ['scale', '40-bit histogram rank', 'arrangement rank'],
        'runs': runs, 'unique_layer_expert_pairs': len(all_pairs),
        'unique_weight_groups': len(unique_groups), 'unique_weights': len(unique_groups) * 32,
        'unique_descriptor_bytes': {'candidate_total': sum(unique_sizes), 'original_total': len(unique_sizes) * 17,
                                    'minimum': min(unique_sizes), 'maximum': max(unique_sizes),
                                    'size_counts': dict(sorted(Counter(unique_sizes).items()))},
        'repeated_france_reference_matches': repeated_matches,
        'sha256': {name: hashlib.sha256((work / name).read_bytes()).hexdigest() for name in names},
        'validation_method': 'Validate actual compact payload bytes using independent Python histogram/arrangement arithmetic, then expand decoded histogram fields in a derived audit trace for the unchanged earlier coverage validator. Projection events, raw codes, expected sites and output files are not changed; audit copies retained separately.',
        'audit_directory': str(audit), 'gate': 'PASS',
        'scope': 'Complete model forwards with histogram+arrangement replacement at sampled expert groups in layers1,48,92. Not replacement of every group/layer or integration of all prior vector/router/endpoint experiments. Only listed five-token prefills tested.',
        'performance': 'Deferred; no checkpoint or memory-layout rewrite and no speed claim. Each descriptor is formed from loaded original weights during the experiment.',
    }
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('runs', 'sha256')}, indent=2))
    for run in runs:
        print(run['name'], json.dumps({'groups': run['summary']['groups'], 'projections': run['summary']['projection_outputs'],
                                      'histogram_bits': run['histogram_bits'], 'output': run['full_output'],
                                      'descriptor_bytes': run['descriptor_bytes']}))


if __name__ == '__main__':
    main()