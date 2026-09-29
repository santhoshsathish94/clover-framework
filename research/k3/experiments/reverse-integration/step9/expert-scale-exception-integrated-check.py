#!/usr/bin/env python3
"""Validate adaptive scale bytes, then reuse unchanged scale/joint/output gates."""

import argparse
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


def translate(work, derived, name, adaptive, fixed, predicted):
    enabled = name != 'fixed-control'
    scale_rows, scale_summary, exception_summary, summary = {}, None, None, None
    candidate_bytes, fixed_bytes, chosen_count = 0, 0, 0
    with (work / (name + '.jsonl')).open() as source, (derived / (name + '.jsonl')).open('x') as destination:
        for line in source:
            event = json.loads(line)
            assert summary is None
            translated = dict(event)
            if event['type'] == 'exception_summary':
                assert enabled and exception_summary is None and scale_summary is None
                exception_summary = event
                continue
            if event['type'] == 'scale_row':
                assert scale_summary is None and exception_summary is None
                key = tuple(event[field] for field in ('layer', 'expert', 'part', 'row'))
                assert key not in scale_rows
                original, payload = bytes(event['original_scales']), bytes(event['payload'])
                assert len(original) == event['groups']
                if enabled:
                    assert event['scale_format'] == 'adaptive-exceptions'
                    assert adaptive.decode(payload, len(original), 'adaptive') == original
                    expected, bits, chosen = adaptive.encode(original, 'adaptive')
                    assert expected == payload and bits == event['payload_bits'] and chosen == event['uses_exceptions']
                    assert len(payload) == predicted[key]['formats']['adaptive']['bytes']
                    assert bits == predicted[key]['formats']['adaptive']['bits']
                    assert chosen == predicted[key]['formats']['adaptive']['uses_exceptions']
                    chosen_count += chosen
                else:
                    assert fixed.decode_scales(payload, len(original)) == original
                baseline_payload, base, width, bits = fixed.encode_scales(original)
                translated.update({'payload': list(baseline_payload), 'base': base,
                                   'offset_width': width, 'payload_bits': bits})
                scale_rows[key] = (original, payload, bool(event.get('uses_exceptions', False)))
                candidate_bytes += len(payload)
                fixed_bytes += len(baseline_payload)
            elif event['type'] == 'scale_summary':
                assert scale_summary is None
                scale_summary = event
                assert event['payload_bytes'] == candidate_bytes
                assert event['rows'] == len(scale_rows) and event['gate'] == 'PASS'
                assert event['scales'] == sum(len(value[0]) for value in scale_rows.values())
                assert event['weights'] == event['scales'] * 32
                if enabled:
                    assert exception_summary and exception_summary['gate'] == 'PASS'
                    assert exception_summary['control_checks'] == 807
                    assert exception_summary['invalid_inputs_rejected'] == 5
                    assert exception_summary['exception_rows'] == chosen_count
                    assert exception_summary['fixed_rows'] == len(scale_rows) - chosen_count
                else:
                    assert exception_summary is None
                translated['payload_bytes'] = fixed_bytes
            elif event['type'] == 'summary':
                assert scale_summary
                summary = event
            else:
                assert event['type'] in ('group', 'projection')
            destination.write(json.dumps(translated, separators=(',', ':')) + '\n')
    assert summary and summary['gate'] == 'PASS'
    return scale_rows, scale_summary, exception_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--runs', nargs='+', choices=('smoke', 'fixed-control', 'france', 'japan'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = args.directory
    derived, joint_trace, audit = (args.output.with_suffix(suffix) for suffix in ('.fixed', '.joint', '.audit'))
    for path in (derived, joint_trace, audit):
        path.mkdir()
    adaptive = load('adaptive_python', work / 'expert-scale-exceptions.py')
    fixed = load('fixed_python', work / 'expert-scale-row.py')
    scale_check = load('unchanged_scale_check', work / 'expert-scale-integrated-check.py')
    joint_check = load('unchanged_joint_check', work / 'expert-joint-integrated-check.py')
    coverage = load('unchanged_coverage_check', work / 'expert-group-coverage-check.py')
    joint = load('joint_python', work / 'expert-joint-rank.py')
    order = load('unchanged_order', work / 'expert-group-order-check.py')
    outputs = load('unchanged_outputs', work / 'expert-group-check.py')
    saved = json.loads((work / 'saved-scale-exceptions.json').read_text())
    predicted = {tuple(row[field] for field in ('layer', 'expert', 'part', 'row')): row for row in saved['rows']}
    assert len(predicted) == 1926
    joint_saved = json.loads((work / 'saved-joint-results.json').read_text())
    joint_predictions = {tuple(row[field] for field in ('layer', 'expert', 'part', 'row', 'group')): int(row['joint_rank'])
                         for row in joint_saved['groups']}
    assert len(joint_predictions) == 5778
    assert hashlib.md5((work / 'france-reference.bin').read_bytes()).hexdigest() == outputs.BASELINE_MD5
    assert (work / 'france-disabled.bin').read_bytes() == (work / 'france-reference.bin').read_bytes()
    runs, unique_rows, unique_pairs, unique_joint = [], {}, set(), set()
    for name in args.runs:
        reference = 'japan' if name == 'japan' else 'france'
        for filename in (reference + '-reference.bin', reference + '-reference.routes', name + '.bin', name + '.routes'):
            if not (audit / filename).exists():
                os.link(work / filename, audit / filename)
        actual_rows, actual_summary, exception_summary = translate(work, derived, name, adaptive, fixed, predicted)
        fixed_rows, fixed_summary = scale_check.scale_trace(derived, joint_trace, name, fixed)
        assert set(actual_rows) == set(fixed_rows)
        groups, sizes, summary = joint_check.audit_trace(joint_trace, audit, name, joint, order, joint_predictions)
        result, checked_groups, pairs = coverage.check_run(audit, name, reference, 1 if name == 'smoke' else 2, order, outputs)
        assert set(groups) == checked_groups and set(actual_rows) == {key[:4] for key in checked_groups}
        routes = coverage.load_routes(work / (reference + '-reference.routes'))
        for key, row in fixed_rows.items():
            assert (row['width'], row['rows']) == ((3584, 3072) if key[2] < 2 else (3072, 3584))
            assert row['positions'] == sum(key[1] in routes[key[0], position][1] for position in range(5))
        if name != 'fixed-control':
            unique_pairs.update(pairs)
            unique_joint.update(checked_groups)
            for key, value in actual_rows.items():
                if key in unique_rows:
                    assert unique_rows[key] == value
                else:
                    unique_rows[key] = value
        result.pop('descriptor_bytes')
        result['raw_adaptive_trace_sha256'] = hashlib.sha256((work / (name + '.jsonl')).read_bytes()).hexdigest()
        result['derived_audit_trace_sha256'] = result.pop('trace_sha256')
        result['scale_summary'] = actual_summary
        result['exception_summary'] = exception_summary
        result['independently_verified_scale_rows'] = len(actual_rows)
        result['fixed_offset_payload_bytes'] = fixed_summary['payload_bytes']
        result['adaptive_bytes_match_python_and_saved_predictions'] = name != 'fixed-control'
        runs.append(result)
    repeated = work / 'france-reference-after.bin'
    repeated_matches = repeated.read_bytes() == (work / 'france-reference.bin').read_bytes() if repeated.exists() else None
    assert repeated_matches is not False
    original_bytes = sum(len(value[0]) for value in unique_rows.values())
    fixed_bytes = sum(len(fixed.encode_scales(value[0])[0]) for value in unique_rows.values())
    candidate_bytes = sum(len(value[1]) for value in unique_rows.values())
    names = ('baseline.c', 'candidate.c', 'expert-group-coverage.h', 'expert-group-coverage.patch',
             'expert-group-order.h', 'expert-group-equation.h', 'expert-joint-codec.h',
             'scale-base.h', 'scale-base-lf.h', 'expert-scale-codec.h', 'expert-scale-exception-codec.h',
             'expert-scale-exception-integrated.patch', 'expert-scale-exception-integrated-check.py',
             'expert-scale-exceptions.py', 'saved-scale-exceptions.json', 'expert-scale-row.py',
             'expert-scale-integrated-check.py', 'expert-joint-integrated-check.py', 'expert-joint-rank.py',
             'saved-joint-results.json', 'expert-group-coverage-check.py', 'expert-group-order-check.py',
             'expert-group-check.py', 'reference', 'candidate',
             'deps/gmp/usr/include/x86_64-linux-gnu/gmp.h', 'deps/gmp/usr/lib/x86_64-linux-gnu/libgmp.a')
    packages = list((work / 'deps').glob('libgmp-dev_*.deb'))
    assert len(packages) == 1
    report = {
        'host': 'AX102, AMD Ryzen 9 7950X3D',
        'build_flags': '-O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5; private GMP include/static archive; -lm',
        'settings': {'OMP_NUM_THREADS': '16', 'OMP_PROC_BIND': 'close', 'OMP_PLACES': 'cores',
                     'K3_PREFETCH': '4', 'K3_TRUNKRAM': '0', 'K3_NREADER': '14',
                     'K3_XDEC': '2', 'K3_PLGRAN': '1', 'K3_HUGE': '1'},
        'runs': runs, 'unique_layer_expert_pairs': len(unique_pairs), 'unique_scale_rows': len(unique_rows),
        'unique_scales': original_bytes, 'unique_weights_using_decoded_scales': original_bytes * 32,
        'unique_joint_rank_groups': len(unique_joint),
        'payload_accounting': {'original_scale_bytes': original_bytes, 'fixed_offset_bytes': fixed_bytes,
                               'adaptive_bytes': candidate_bytes,
                               'unchanged_code_bytes': original_bytes * 16,
                               'original_code_plus_scale': original_bytes * 17,
                               'fixed_code_plus_scale': original_bytes * 16 + fixed_bytes,
                               'adaptive_code_plus_scale': original_bytes * 16 + candidate_bytes,
                               'further_saved_payload_bytes': fixed_bytes - candidate_bytes,
                               'exception_rows': sum(value[2] for value in unique_rows.values()),
                               'fixed_rows': sum(not value[2] for value in unique_rows.values())},
        'repeated_france_reference_matches': repeated_matches,
        'sha256': {name: hashlib.sha256((work / name).read_bytes()).hexdigest() for name in names},
        'dependency': {'package': packages[0].name, 'sha256': hashlib.sha256(packages[0].read_bytes()).hexdigest(),
                       'installation': 'Downloaded Ubuntu libgmp-dev package unpacked inside experiment; linked archive statically. No system package install.'},
        'gate': 'PASS',
        'validation': 'Independently decode/re-encode every actual adaptive byte sequence and compare offline predictions. Produce separate fixed-offset audit events with verified equivalent bytes and recomputed byte totals; reuse unchanged scale, joint and coverage/output checks. Raw original values, projections, model outputs and gates unchanged.',
        'scope': 'Complete forward calls but only sampled rows in three layers; adaptive scale coding plus previously validated joint code groups. Listed five-token prefills only, no full checkpoint or generation/decode coverage.',
        'limits': 'Payload accounting includes tags/headers/padding, excludes persistent indexing and container overhead. Harness retains original data and diagnostics; not measured checkpoint/RAM reduction. Performance deferred.',
    }
    with args.output.open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('runs', 'sha256')}, indent=2))
    for run in runs:
        print(run['name'], json.dumps({'scales': run['scale_summary'], 'exceptions': run['exception_summary'],
                                      'projections': run['summary']['projection_outputs'], 'output': run['full_output']}))


if __name__ == '__main__':
    main()