#!/usr/bin/env python3
"""Independently verify named equation consumers and preserve tested evidence."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(base):
    previous = Path('/opt/clover-k3/layer1-compact-model-20260929-a')
    old = json.loads((previous / 'results.json').read_text())
    prepared = json.loads((base / 'prepare.json').read_text())
    assert prepared['source_report_sha256'] == digest(previous / 'results.json')
    model = base / 'model'
    for name, expected in prepared['model_sha256'].items():
        assert digest(model / name) == expected, name
    for name, expected in old['implementation_and_evidence_sha256'].items():
        assert digest(previous / name) == expected, name
    changes = {
        'seed-model.h': [('#include "seed-reader.h"', '#include "seed-reader.h"\n#include "seed-function.h"'),
                         ('seed_read_row(&input, (unsigned)ids[position], seed_model_rows[position]);',
                          'seed(&input, (unsigned)ids[position], seed_model_rows[position]);')],
        'fruit-head.h': [('static void fruit_project(', 'void fruit(')],
        'vector-endpoints.h': [('fruit_project(', 'fruit(')],
        'candidate.c': [('fruit_project(', 'fruit('), ('#include "layer1-compact-model.h"', '#include "equation-functions-model.h"'),
                        ('    l1_model_finish();', '    equation_functions_report();'),
                        ('                            I_, LAT, L, e, 2, mt, mj);',
                         '                            I_, LAT, L, e, 2, mt, mj);\n                        root_checked(L, e, m, zin, go, dgo);')]
    }
    for path in previous.iterdir():
        if not path.is_file() or path.suffix not in ('.c', '.h', '.py'):
            continue
        if path.name in changes:
            text = path.read_text()
            for before, after in changes[path.name]:
                assert text.count(before) == 1
                text = text.replace(before, after)
            assert (model / path.name).read_text() == text, path.name
        else:
            assert digest(model / path.name) == digest(path), path.name
    symbols = subprocess.check_output(['nm', '--defined-only', str(model / 'candidate')], text=True)
    for name in ('seed', 'root', 'fruit'):
        assert any(line.split()[-2:] == ['T', name] for line in symbols.splitlines())
    report = json.loads((base / 'enabled/prior-combined.json').read_text())
    assert report['gate'] == 'PASS' and report['items_integrated'] == list(range(13, 0, -1))
    assert (report['unique_rows'], report['unique_pairs'], report['unique_groups']) == (1926, 214, 5778)
    controls = json.loads((base / 'controls.json').read_text())
    assert controls['gate'] == 'PASS' and len(controls['invalid_cases']) == 7
    cases, experts = [], set()
    for name in ('france', 'japan'):
        folder = base / 'enabled'
        paths = {suffix: folder / (name + suffix) for suffix in ('.bin', '.routes', '-seed.json', '-fruit.json', '-seed-rows.bf16', '-compact.jsonl', '-functions.json')}
        launch = json.loads((folder / (name + '-launch.json')).read_text())
        assert launch['binary_sha256'] == digest(model / 'candidate')
        for suffix in ('.bin', '.routes', '-seed-rows.bf16'):
            assert paths[suffix].read_bytes() == (previous / 'enabled' / (name + suffix)).read_bytes(), suffix
        for suffix in ('-seed.json', '-fruit.json'):
            actual = json.loads(paths[suffix].read_text())
            expected = json.loads((previous / 'enabled' / (name + suffix)).read_text())
            assert actual == expected and actual['gate'] == 'PASS' and actual['guard_probe_sigsegv'], suffix
        reference_routes = [list(map(int, line.split())) for line in (previous / (name + '-reference.routes')).read_text().splitlines()]
        uses = Counter(expert for row in reference_routes if row[0] == 1 for expert in row[3:])
        experts.update(uses)
        events = [json.loads(line) for line in paths['-compact.jsonl'].read_text().splitlines()]
        expected_events = [json.loads(line) for line in (previous / 'enabled' / (name + '-compact.jsonl')).read_text().splitlines()]
        assert events == expected_events, 'compact blocks/positions/bytes changed'
        assert {(item['expert'], item['matrix']) for item in events[:-1]} == {(expert, matrix) for expert in uses for matrix in range(3)}
        functions = json.loads(paths['-functions.json'].read_text())
        assert functions['gate'] == 'PASS'
        assert functions['seed_calls'] == 5 and functions['fruit_calls'] == 2
        assert functions['root_calls'] == len(uses) and functions['root_positions'] == sum(uses.values())
        assert functions['root_stage_values'] == [sum(uses.values()) * width for width in (3072, 3072, 3072, 3584)]
        assert functions['root_activation_scratch_bytes'] == 122880 and functions['root_weight_array_bytes'] == 0
        assert functions['root_output_replaces_oracle'] and functions['root_layer'] == 1
        assert paths['.bin'].stat().st_size == 171008 * 4
        cases.append({'prompt': name, 'functions': functions, 'all_final_output_and_route_bits_exact': True,
                      'output_md5': hashlib.md5(paths['.bin'].read_bytes()).hexdigest(),
                      'seed_fruit_reports_unchanged': True, 'compact_consumer': events[-1],
                      'artifacts_sha256': {path.name: digest(path) for path in paths.values()}})
    for filename, expected in old['protected_sha256_unchanged'].items():
        assert digest(Path(filename)) == expected
    maps = Path('/opt/clover-k3/layer1-map-store-20260929-a/experts.bin')
    assert digest(maps) == old['dataset_sha256_unchanged']
    fruit_previous = json.loads(Path('/opt/clover-k3/fruit-integrated-20260929-a/results.json').read_text())
    assert digest(Path('/opt/clover-k3/seed/seed.bin')) == fruit_previous['seed_sha256_unchanged']
    assert digest(Path('/opt/clover-k3/fruit/fruit.bin')) == fruit_previous['fruit_dataset']['fruit_sha256']
    for dependency in fruit_previous['dependencies_unchanged'].values():
        assert digest(Path(dependency['path'])) == dependency['sha256']
    result = {'gate': 'PASS', 'entry_points': ['seed', 'root', 'fruit'], 'layer': 1,
              'runs': cases, 'unique_experts_exercised': len(experts), 'controls': controls,
              'unchanged_prior_13_stage_gates': True,
              'prior_stage_report_sha256': digest(base / 'enabled/prior-combined.json'),
              'prior_model_report_sha256': digest(previous / 'results.json'),
              'model_sha256': prepared['model_sha256'], 'command': prepared['command'],
              'protected_sha256_unchanged': old['protected_sha256_unchanged'],
              'root_dataset_sha256_unchanged': old['dataset_sha256_unchanged'],
              'source_scripts_sha256': {path.name: digest(path) for path in base.iterdir() if path.is_file() and path.suffix in ('.c', '.h', '.py')},
              'apis': {'seed': 'seed(reader, token, bf16_row[7168])',
                       'root': 'root(layer, expert, positions, input_rows, output_rows, optional_observer, context)',
                       'fruit': 'fruit(logits, normalized_input, expression_mode)'},
              'limits': 'Two five-token prefills, 82 selected layer1 experts, no other layers converted. No complete embedding/head/expert weight object reconstructed. Seed/fruit retain bounded16-row BF16 blocks, root compact64-row code/selector block plus two3072-activation vectors per position (capacity5). Oracle original packed I/O and model activations remain; no speed/peakRSS claim. Root observer optional and validation-only. Seed reader lifecycle/identity checks and fruit normalized-expression context remain with existing model adapters.'}
    with (base / 'results.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    with tarfile.open(base / 'verified-functions.tar.gz', 'x:gz') as archive:
        for path in model.iterdir():
            if path.is_file() and path.suffix in ('.c', '.h', '.py', '.json'):
                archive.add(path, arcname='model/' + path.name, recursive=False)
        archive.add(model / 'expert-constant-palette.h', arcname='expert-constant-palette.h', recursive=False)
        for path in base.iterdir():
            if path.is_file() and path.suffix in ('.c', '.h', '.py', '.json'):
                archive.add(path, arcname=path.name, recursive=False)
        for path in (base / 'enabled').iterdir():
            if path.is_file() and path.name.startswith(('france', 'japan', 'prior-')):
                archive.add(path, arcname='enabled/' + path.name, recursive=False)
    print(json.dumps({'gate': 'PASS', 'functions': result['entry_points'], 'unique_experts': len(experts),
                      'results_sha256': digest(base / 'results.json'),
                      'archive_sha256': digest(base / 'verified-functions.tar.gz')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    check(parser.parse_args().directory)