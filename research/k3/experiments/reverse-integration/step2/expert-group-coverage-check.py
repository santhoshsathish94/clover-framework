#!/usr/bin/env python3
"""Validate sampled expert-group coverage against reference routes and outputs."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def load_routes(path):
    routes = {}
    for line in path.read_text().splitlines():
        fields = [int(value) for value in line.split()]
        assert len(fields) == 19
        layer, position, token = fields[:3]
        selected = fields[3:]
        assert (layer, position) not in routes and len(set(selected)) == 16
        assert all(0 <= expert < 896 for expert in selected)
        routes[layer, position] = (token, selected)
    assert set(routes) == {(layer, position) for layer in range(1, 93) for position in range(5)}
    return routes


def check_run(work, name, reference_name, mode, codecs, output_check):
    routes_path = work / (reference_name + '-reference.routes')
    routes = load_routes(routes_path)
    assert (work / (name + '.routes')).read_bytes() == routes_path.read_bytes()
    pairs = {(1, routes[1, 0][1][0])} if mode == 1 else {
        (layer, expert) for layer in (1, 48, 92)
        for position in range(5) for expert in routes[layer, position][1]}
    expected_groups, expected_projections = set(), set()
    for layer, expert in pairs:
        for part in range(3):
            width, rows = (3584, 3072) if part < 2 else (3072, 3584)
            for row in (0, rows // 2, rows - 1):
                for group in (0, width // 64, width // 32 - 1):
                    expected_groups.add((layer, expert, part, row, group))
                for position in range(5):
                    choices = routes[layer, position][1]
                    if expert in choices:
                        expected_projections.add((layer, expert, part, row, position, choices.index(expert)))
    groups, projections, codes_total = set(), set(), Counter()
    histograms, scales, sizes = set(), set(), []
    summary = None
    with (work / (name + '.jsonl')).open() as trace:
        for line in trace:
            event = json.loads(line)
            assert summary is None
            if event['type'] == 'summary':
                summary = event
                continue
            layer, expert, part, row = (event[key] for key in ('layer', 'expert', 'part', 'row'))
            if event['type'] == 'projection':
                key = (layer, expert, part, row, event['position'], event['rank'])
                assert key in expected_projections and key not in projections
                assert event['reference_bits'] == event['candidate_bits']
                projections.add(key)
                continue
            assert event['type'] == 'group'
            key = (layer, expert, part, row, event['group'])
            assert key in expected_groups and key not in groups
            assert (event['width'], event['rows']) == ((3584, 3072) if part < 2 else (3072, 3584))
            assert event['positions'] == sum(expert in routes[layer, position][1] for position in range(5))
            assert event['code_changes'] == event['weight_changes'] == 0
            packed = bytes(event['packed'])
            assert len(packed) == 16
            original = [(packed[index // 2] >> (4 * (index % 2))) & 15 for index in range(32)]
            payload = bytes(event['payload'])
            value = int.from_bytes(payload, 'little')
            assert len(payload) == (event['payload_bits'] + 7) // 8
            assert value >> event['payload_bits'] == 0
            assert value & 255 == event['scale']
            value >>= 8
            counts = []
            for code in range(16):
                counts.append(value & 63)
                value >>= 6
            assert sum(counts) == 32 and counts == [original.count(code) for code in range(16)]
            assert value == int(event['rank']) == codecs.rank_sequence(original, counts)
            assert codecs.unrank_sequence(counts, value)[0] == original
            assert event['payload_bits'] == 104 + (codecs.ways(counts) - 1).bit_length()
            groups.add(key)
            histograms.add(tuple(counts))
            codes_total.update(original)
            scales.add(event['scale'])
            sizes.append(len(payload))
    assert summary and summary['gate'] == 'PASS' and summary['mode'] == mode
    assert groups == expected_groups and projections == expected_projections
    assert summary['groups'] == len(groups) == 27 * len(pairs)
    assert summary['weights'] == 32 * len(groups) and summary['pairs'] == len(pairs)
    assert summary['projection_outputs'] == len(projections)
    assert summary['decoder_checks'] == 4373 and summary['invalid_inputs_rejected'] == 5
    assert summary['decode_pairs_checked'] == 4096 and summary['mask_coordinates_checked'] == 512
    for layer in (1, 48, 92):
        assert summary['layer_pairs'][str(layer)] == sum(pair[0] == layer for pair in pairs)
    reference = (work / (reference_name + '-reference.bin')).read_bytes()
    output = output_check.compare(reference, (work / (name + '.bin')).read_bytes())
    assert output['bit_exact']
    result = {
        'name': name, 'mode': mode,
        'prompt_ids': [routes[1, position][0] for position in range(5)],
        'reference_md5': hashlib.md5(reference).hexdigest(),
        'summary': summary, 'full_output': output,
        'reference_routes_identical': True,
        'coverage_matches_reference_route_expectations': True,
        'all_histograms_ranks_and_payloads_independently_checked': True,
        'experts_by_layer': {str(layer): sorted(expert for observed_layer, expert in pairs if observed_layer == layer)
                             for layer in (1, 48, 92)},
        'groups_by_part': {str(part): sum(key[2] == part for key in groups) for part in range(3)},
        'scale_bytes_observed': sorted(scales),
        'code_occurrences': {str(code): codes_total[code] for code in range(16)},
        'distinct_histograms': len(histograms),
        'descriptor_bytes': {'minimum': min(sizes), 'maximum': max(sizes), 'total': sum(sizes),
                             'original_total': 17 * len(sizes)},
        'trace_sha256': hashlib.sha256((work / (name + '.jsonl')).read_bytes()).hexdigest(),
        'reference_routes_sha256': hashlib.sha256(routes_path.read_bytes()).hexdigest(),
        'gate': 'PASS',
    }
    return result, groups, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--runs', nargs='+', choices=('smoke', 'france', 'japan'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = args.directory
    codecs = module('order_checker', work / 'expert-group-order-check.py')
    output_check = module('output_checker', work / 'expert-group-check.py')
    assert hashlib.md5((work / 'france-reference.bin').read_bytes()).hexdigest() == output_check.BASELINE_MD5
    assert (work / 'france-disabled.bin').read_bytes() == (work / 'france-reference.bin').read_bytes()
    results, all_groups, all_pairs = [], set(), set()
    for name in args.runs:
        reference_name = 'france' if name == 'smoke' else name
        result, groups, pairs = check_run(work, name, reference_name, 1 if name == 'smoke' else 2, codecs, output_check)
        results.append(result)
        all_groups.update(groups)
        all_pairs.update(pairs)
    repeated = work / 'france-reference-after.bin'
    repeated_matches = repeated.read_bytes() == (work / 'france-reference.bin').read_bytes() if repeated.exists() else None
    assert repeated_matches is not False
    names = ('baseline.c', 'candidate.c', 'expert-group-coverage.h', 'expert-group-coverage.patch',
             'expert-group-order.h', 'expert-group-equation.h', 'expert-group-coverage-check.py',
             'expert-group-order-check.py', 'expert-group-check.py', 'reference', 'candidate')
    report = {
        'host': 'AX102, AMD Ryzen 9 7950X3D',
        'build_flags': '-O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5 -lm',
        'settings': {'OMP_NUM_THREADS': '16', 'OMP_PROC_BIND': 'close', 'OMP_PLACES': 'cores',
                     'K3_PREFETCH': '4', 'K3_TRUNKRAM': '0', 'K3_NREADER': '14',
                     'K3_XDEC': '2', 'K3_PLGRAN': '1', 'K3_HUGE': '1'},
        'runs': results, 'unique_layer_expert_pairs': len(all_pairs),
        'unique_weight_groups': len(all_groups), 'unique_weights': len(all_groups) * 32,
        'repeated_france_reference_matches': repeated_matches,
        'sha256': {name: hashlib.sha256((work / name).read_bytes()).hexdigest() for name in names},
        'gate': 'PASS',
        'scope': 'First/middle/last groups in first/middle/last rows, all three expert matrices. Mode 1 covers first selected expert at layer1; mode 2 all selected experts at layers1,48,92. These are sampled groups, not complete experts/checkpoint. Only the listed five-token prefills are covered; no generation/decode claim.',
        'performance': 'Deferred. No compression/speed claim; descriptor sizes reported with scale and count overhead.',
    }
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('runs', 'sha256')}, indent=2))
    for result in results:
        print(result['name'], json.dumps({'coverage': result['summary'], 'output': result['full_output'],
                                         'descriptor_bytes': result['descriptor_bytes']}))


if __name__ == '__main__':
    main()