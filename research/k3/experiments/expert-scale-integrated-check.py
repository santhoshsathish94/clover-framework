#!/usr/bin/env python3
"""Validate complete scale rows and reuse unchanged joint/coverage model gates."""

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


def scale_trace(work, joint_trace, name, codec):
    enabled = name != 'scale-disabled'
    scale_rows, scale_summary, summary = {}, None, None
    sampled_scales = {}
    with (work / (name + '.jsonl')).open() as source, (joint_trace / (name + '.jsonl')).open('x') as filtered:
        for line in source:
            event = json.loads(line)
            assert summary is None
            if event['type'] == 'scale_summary':
                assert enabled and scale_summary is None
                scale_summary = event
                continue
            if event['type'] == 'scale_row':
                assert enabled and scale_summary is None
                key = tuple(event[field] for field in ('layer', 'expert', 'part', 'row'))
                assert key not in scale_rows
                original = bytes(event['original_scales'])
                payload = bytes(event['payload'])
                assert len(original) == event['groups']
                assert event['width'] == len(original) * 32
                assert event['weights_checked'] == event['width']
                assert event['scale_changes'] == event['weight_changes'] == 0
                assert codec.decode_scales(payload, len(original)) == original
                encoded, base, width, bits = codec.encode_scales(original)
                assert encoded == payload and base == event['base']
                assert width == event['offset_width'] and bits == event['payload_bits']
                scale_rows[key] = event
                continue
            assert scale_summary is None or event['type'] == 'summary'
            if event['type'] == 'group':
                key = tuple(event[field] for field in ('layer', 'expert', 'part', 'row'))
                sampled_scales[key + (event['group'],)] = event['scale']
            elif event['type'] == 'summary':
                summary = event
            else:
                assert event['type'] == 'projection'
            filtered.write(line)
    assert summary and summary['gate'] == 'PASS'
    if enabled:
        assert scale_summary and scale_summary['gate'] == 'PASS'
        assert scale_summary['control_checks'] == 12 and scale_summary['invalid_inputs_rejected'] == 6
        assert scale_summary['rows'] == len(scale_rows) == summary['pairs'] * 9
        assert scale_summary['scales'] == sum(row['groups'] for row in scale_rows.values()) == summary['pairs'] * 960
        assert scale_summary['weights'] == scale_summary['scales'] * 32
        assert scale_summary['payload_bytes'] == sum(len(row['payload']) for row in scale_rows.values())
        for key, scale in sampled_scales.items():
            assert scale_rows[key[:4]]['original_scales'][key[4]] == scale
    else:
        assert not scale_rows and scale_summary is None
    return scale_rows, scale_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--runs', nargs='+', choices=('smoke', 'scale-disabled', 'france', 'japan'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = args.directory
    joint_trace = args.output.with_suffix('.joint')
    audit = args.output.with_suffix('.audit')
    joint_trace.mkdir()
    audit.mkdir()
    scale_codec = load('python_scale_codec', work / 'expert-scale-row.py')
    joint_check = load('joint_checker_unchanged', work / 'expert-joint-integrated-check.py')
    coverage = load('coverage_unchanged', work / 'expert-group-coverage-check.py')
    joint = load('joint_python', work / 'expert-joint-rank.py')
    order = load('order_unchanged', work / 'expert-group-order-check.py')
    output_check = load('output_unchanged', work / 'expert-group-check.py')
    saved = json.loads((work / 'saved-joint-results.json').read_text())
    predictions = {tuple(row[field] for field in ('layer', 'expert', 'part', 'row', 'group')): int(row['joint_rank'])
                   for row in saved['groups']}
    initial_scales = json.loads((work / 'saved-scale-rows.json').read_text())
    initial = {(1, 498, row['part'], row['row']): row for row in initial_scales['rows']}
    assert hashlib.md5((work / 'france-reference.bin').read_bytes()).hexdigest() == output_check.BASELINE_MD5
    assert (work / 'france-disabled.bin').read_bytes() == (work / 'france-reference.bin').read_bytes()
    runs, all_rows, all_joint_groups, all_pairs = [], {}, set(), set()
    for name in args.runs:
        reference = 'japan' if name == 'japan' else 'france'
        for filename in (reference + '-reference.bin', reference + '-reference.routes', name + '.bin', name + '.routes'):
            if not (audit / filename).exists():
                os.link(work / filename, audit / filename)
        rows, scale_summary = scale_trace(work, joint_trace, name, scale_codec)
        groups, sizes, summary = joint_check.audit_trace(joint_trace, audit, name, joint, order, predictions)
        result, checked_groups, pairs = coverage.check_run(audit, name, reference, 1 if name == 'smoke' else 2, order, output_check)
        assert set(groups) == checked_groups
        if name != 'scale-disabled':
            expected_rows = {key[:4] for key in checked_groups}
            assert set(rows) == expected_rows
            for key, row in rows.items():
                assert (row['width'], row['rows']) == ((3584, 3072) if key[2] < 2 else (3072, 3584))
                expected_positions = {item['positions'] for item in []}
                routes = coverage.load_routes(work / (reference + '-reference.routes'))
                assert row['positions'] == sum(key[1] in routes[key[0], position][1] for position in range(5))
                if key in initial:
                    assert row['original_scales'] == initial[key]['scales']
                    assert row['payload'] == initial[key]['payload']
                fingerprint = (bytes(row['original_scales']), bytes(row['payload']))
                if key in all_rows:
                    assert all_rows[key] == fingerprint
                else:
                    all_rows[key] = fingerprint
            all_pairs.update(pairs)
            all_joint_groups.update(checked_groups)
        result['raw_scale_trace_sha256'] = hashlib.sha256((work / (name + '.jsonl')).read_bytes()).hexdigest()
        result['validated_audit_trace_sha256'] = result.pop('trace_sha256')
        result['joint_descriptor_bytes'] = {'minimum': min(sizes), 'maximum': max(sizes), 'total': sum(sizes)}
        result.pop('descriptor_bytes')
        result['scale_summary'] = scale_summary
        result['scale_rows_match_reference_expected_sites'] = name != 'scale-disabled'
        result['scale_descriptor_bytes_independently_verified'] = name != 'scale-disabled'
        runs.append(result)
    repeated = work / 'france-reference-after.bin'
    repeated_matches = repeated.read_bytes() == (work / 'france-reference.bin').read_bytes() if repeated.exists() else None
    assert repeated_matches is not False
    scale_bytes = sum(len(value[0]) for value in all_rows.values())
    compact_bytes = sum(len(value[1]) for value in all_rows.values())
    width_counts = Counter(value[1][1] & 15 for value in all_rows.values())
    names = ('baseline.c', 'candidate.c', 'coverage-base.h', 'coverage-base-lf.h', 'expert-group-coverage.h',
             'expert-group-coverage.patch', 'expert-group-order.h', 'expert-group-equation.h', 'expert-joint-codec.h',
             'expert-scale-codec.h', 'expert-scale-integrated.patch', 'expert-scale-integrated-check.py',
             'expert-scale-row.py', 'saved-scale-rows.json', 'expert-joint-integrated-check.py', 'expert-joint-rank.py',
             'saved-joint-results.json', 'expert-group-coverage-check.py', 'expert-group-order-check.py',
             'expert-group-check.py', 'reference', 'candidate')
    report = {
        'host': 'AX102, AMD Ryzen 9 7950X3D',
        'build_flags': '-O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5 -lm',
        'settings': {'OMP_NUM_THREADS': '16', 'OMP_PROC_BIND': 'close', 'OMP_PLACES': 'cores',
                     'K3_PREFETCH': '4', 'K3_TRUNKRAM': '0', 'K3_NREADER': '14',
                     'K3_XDEC': '2', 'K3_PLGRAN': '1', 'K3_HUGE': '1'},
        'runs': runs, 'unique_layer_expert_pairs': len(all_pairs),
        'unique_scale_rows': len(all_rows), 'unique_scale_groups': scale_bytes,
        'unique_weights_checked_with_decoded_scales': scale_bytes * 32,
        'unique_joint_rank_groups': len(all_joint_groups),
        'payload_accounting': {'original_scale_bytes': scale_bytes, 'candidate_scale_bytes': compact_bytes,
                               'unchanged_code_bytes': 16 * scale_bytes,
                               'original_code_plus_scale_bytes': 17 * scale_bytes,
                               'candidate_code_plus_scale_bytes': 16 * scale_bytes + compact_bytes,
                               'saved_scale_payload_bytes': scale_bytes - compact_bytes,
                               'offset_width_counts': dict(sorted(width_counts.items()))},
        'repeated_france_reference_matches': repeated_matches,
        'sha256': {name: hashlib.sha256((work / name).read_bytes()).hexdigest() for name in names},
        'validation_method': 'Decode actual row-scale payloads independently, match row sets to reference-route-derived group sites, and check original nine rows against checkpoint-read evidence. Separately validate joint payloads with the unchanged prior joint adapter, then run unchanged coverage/output gates on a derived audit trace. Raw records and model outputs are not rewritten.',
        'audit_directory': str(audit), 'joint_trace_directory': str(joint_trace), 'gate': 'PASS',
        'scope': 'Complete scales for first/middle/last rows of all three matrices for selected experts at layers1,48,92. Scales affect every weight in each tested row; joint ranks still apply to only three groups per row. Two five-token prefills if listed, not full checkpoint, longer contexts or decode.',
        'limits': 'Descriptors constructed in memory from original loaded weights. Original reference computations remain for comparisons. Code-plus-scale sizes are payload accounting, not measured checkpoint/RAM reduction; descriptors and diagnostic joint records both exist in the harness, and store indexing is not included. Performance deferred.',
    }
    with args.output.open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('runs', 'sha256')}, indent=2))
    for run in runs:
        print(run['name'], json.dumps({'scale_rows': run['scale_summary'], 'projections': run['summary']['projection_outputs'], 'output': run['full_output']}))


if __name__ == '__main__':
    main()