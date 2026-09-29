#!/usr/bin/env python3
"""Verify actual joint-rank model payloads and reuse unchanged coverage gates."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def audit_trace(work, audit, name, joint, order, predictions):
    enabled = name != 'legacy'
    groups, summary, sizes = {}, None, []
    with (work / (name + '.jsonl')).open() as source, (audit / (name + '.jsonl')).open('x') as translated:
        for line in source:
            event = json.loads(line)
            assert summary is None
            checked = dict(event)
            if event['type'] == 'summary':
                summary = event
                assert event['joint_mode'] == int(enabled)
                assert event['joint_decoder_checks'] == (102 if enabled else 0)
                assert event['joint_invalid_inputs_rejected'] == (5 if enabled else 0)
            elif event['type'] == 'group':
                assert event['joint_mode'] == int(enabled)
                packed = bytes(event['packed'])
                assert len(packed) == 16
                original = [(packed[position // 2] >> (4 * (position % 2))) & 15 for position in range(32)]
                payload = bytes(event['payload'])
                value = int.from_bytes(payload, 'little')
                assert len(payload) == (event['payload_bits'] + 7) // 8
                assert value >> event['payload_bits'] == 0 and value & 255 == event['scale']
                value >>= 8
                key = tuple(event[field] for field in ('layer', 'expert', 'part', 'row', 'group'))
                assert key not in groups
                if enabled:
                    assert len(payload) == 17 and event['payload_bits'] == 136
                    counts, arrangement = joint.joint_unrank(value, 32, 16)
                    assert joint.joint_rank(counts, arrangement) == value
                    assert predictions[key] == value
                else:
                    counts = []
                    for code in range(16):
                        counts.append(value & 63)
                        value >>= 6
                    arrangement = value
                assert counts == [original.count(code) for code in range(16)]
                assert arrangement == int(event['rank']) == order.rank_sequence(original, counts)
                assert order.unrank_sequence(counts, arrangement)[0] == original
                rank_bits = (order.ways(counts) - 1).bit_length()
                if not enabled:
                    assert event['payload_bits'] == 104 + rank_bits
                expanded = event['scale']
                for code, count in enumerate(counts):
                    expanded |= count << (8 + 6 * code)
                expanded |= arrangement << 104
                checked['payload_bits'] = 104 + rank_bits
                checked['payload'] = list(expanded.to_bytes((checked['payload_bits'] + 7) // 8, 'little'))
                groups[key] = (packed, event['scale'], payload)
                sizes.append(len(payload))
            else:
                assert event['type'] == 'projection'
            translated.write(json.dumps(checked, separators=(',', ':')) + '\n')
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
    joint = load('joint_python', work / 'expert-joint-rank.py')
    coverage = load('coverage_unchanged', work / 'expert-group-coverage-check.py')
    order = load('order_unchanged', work / 'expert-group-order-check.py')
    output = load('outputs_unchanged', work / 'expert-group-check.py')
    predicted = json.loads((work / 'saved-joint-results.json').read_text())
    predictions = {tuple(row[field] for field in ('layer', 'expert', 'part', 'row', 'group')):
                   int(row['joint_rank']) for row in predicted['groups']}
    assert len(predictions) == predicted['summary']['unique_groups'] == 5778
    assert hashlib.md5((work / 'france-reference.bin').read_bytes()).hexdigest() == output.BASELINE_MD5
    assert (work / 'france-disabled.bin').read_bytes() == (work / 'france-reference.bin').read_bytes()
    runs, all_groups, all_pairs = [], {}, set()
    for name in args.runs:
        reference = 'japan' if name == 'japan' else 'france'
        for filename in (reference + '-reference.bin', reference + '-reference.routes', name + '.bin', name + '.routes'):
            if not (audit / filename).exists():
                os.link(work / filename, audit / filename)
        groups, sizes, summary = audit_trace(work, audit, name, joint, order, predictions)
        result, checked_groups, pairs = coverage.check_run(audit, name, reference, 1 if name == 'smoke' else 2, order, output)
        assert checked_groups == set(groups)
        result['raw_joint_trace_sha256'] = hashlib.sha256((work / (name + '.jsonl')).read_bytes()).hexdigest()
        result['audit_trace_sha256'] = result.pop('trace_sha256')
        result['descriptor_bytes'] = {'minimum': min(sizes), 'maximum': max(sizes),
                                      'total': sum(sizes), 'original_total': 17 * len(sizes)}
        result['joint_mode'] = summary['joint_mode']
        result['joint_decoder_checks'] = summary['joint_decoder_checks']
        result['joint_invalid_inputs_rejected'] = summary['joint_invalid_inputs_rejected']
        result['joint_payloads_match_python_and_saved_ranks'] = name != 'legacy'
        runs.append(result)
        if name != 'legacy':
            all_pairs.update(pairs)
            for key, fingerprint in groups.items():
                if key in all_groups:
                    assert all_groups[key] == fingerprint
                else:
                    all_groups[key] = fingerprint
    repeated = work / 'france-reference-after.bin'
    repeated_matches = repeated.read_bytes() == (work / 'france-reference.bin').read_bytes() if repeated.exists() else None
    assert repeated_matches is not False
    names = ('baseline.c', 'candidate.c', 'coverage-base.h', 'coverage-base-lf.h',
             'expert-group-coverage.h', 'expert-group-coverage.patch', 'expert-group-order.h',
             'expert-group-equation.h', 'expert-joint-codec.h', 'expert-joint-integrated.patch',
             'expert-joint-integrated-check.py', 'expert-joint-rank.py', 'saved-joint-results.json',
             'expert-group-coverage-check.py', 'expert-group-order-check.py', 'expert-group-check.py',
             'reference', 'candidate')
    report = {
        'host': 'AX102, AMD Ryzen 9 7950X3D',
        'build_flags': '-O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5 -lm',
        'settings': {'OMP_NUM_THREADS': '16', 'OMP_PROC_BIND': 'close', 'OMP_PLACES': 'cores',
                     'K3_PREFETCH': '4', 'K3_TRUNKRAM': '0', 'K3_NREADER': '14',
                     'K3_XDEC': '2', 'K3_PLGRAN': '1', 'K3_HUGE': '1'},
        'integrated_fields': ['8-bit scale', '128-bit joint histogram/arrangement rank'],
        'runs': runs, 'unique_layer_expert_pairs': len(all_pairs),
        'unique_weight_groups': len(all_groups), 'unique_weights': len(all_groups) * 32,
        'unique_descriptor_bytes': sum(len(value[2]) for value in all_groups.values()),
        'original_unique_payload_bytes': len(all_groups) * 17,
        'repeated_france_reference_matches': repeated_matches,
        'sha256': {name: hashlib.sha256((work / name).read_bytes()).hexdigest() for name in names},
        'validation_method': 'Actual raw joint payloads decoded and re-encoded with independent Python integers and matched to saved ranks. Only then expanded to a separate audit trace for unchanged prior coverage/output checks. Raw codes, projections and model outputs untouched.',
        'audit_directory': str(audit), 'gate': 'PASS',
        'scope': 'Complete model forwards with joint-rank decoding replacing sampled groups at layers1,48,92 for the listed five-token prefills. No whole-checkpoint substitution, generation/decode, combined vector/endpoint experiments or alternative persistent weight store.',
        'performance': 'Deferred. Fixed payload equals original17bytes/group; no compression or speed claim. Descriptors constructed from loaded original weights.',
    }
    with args.output.open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('runs', 'sha256')}, indent=2))
    for result in runs:
        print(result['name'], json.dumps({'groups': result['summary']['groups'],
            'projections': result['summary']['projection_outputs'], 'output': result['full_output'],
            'joint_mode': result['joint_mode'], 'descriptor_bytes': result['descriptor_bytes']}))


if __name__ == '__main__':
    main()