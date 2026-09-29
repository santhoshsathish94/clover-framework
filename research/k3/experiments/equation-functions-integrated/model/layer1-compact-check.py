#!/usr/bin/env python3
"""Verify actual routed compact consumers, preserved gates and rejection paths."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_routes(path):
    routes = {}
    for line in path.read_text().splitlines():
        layer, position, token, *experts = map(int, line.split())
        assert len(experts) == len(set(experts)) == 16
        assert (layer, position) not in routes
        routes[layer, position] = experts
    assert set(routes) == {(layer, position) for layer in range(1, 93) for position in range(5)}
    return routes


def main(work):
    prior = Path('/opt/clover-k3/fruit-integrated-20260929-a')
    previous = json.loads((prior / 'results.json').read_text())
    maps = Path('/opt/clover-k3/layer1-map-store-20260929-a')
    dataset = json.loads((maps / 'results.json').read_text())
    assert dataset['gate'] == 'PASS' and digest(maps / 'experts.bin') == dataset['dataset_sha256']
    with (maps / 'experts.bin').open('rb') as stream:
        header = struct.unpack('<8sIIQQ', stream.read(32))
        index = stream.read(header[3])
    lengths = [struct.unpack_from('<II', index, 5376 + block * 8)[0] for block in range(136192)]
    for name, expected in previous['sha256'].items():
        assert digest(prior / name) == expected, name
        if name == 'candidate.c' or name in ('candidate', 'reference', 'seed-prior-candidate'):
            continue
        if (work / name).exists():
            assert digest(work / name) == expected, name
    for filename, expected in previous['protected_sha256'].items():
        assert digest(filename) == expected, filename
    for dependency in previous['dependencies_unchanged'].values():
        assert digest(dependency['path']) == dependency['sha256']
    modified = (work / 'candidate.c').read_text()
    expected = (prior / 'candidate.c').read_text().replace('int main(int argc, char **argv)',
                '#include "layer1-compact-model.h"\nint main(int argc, char **argv)', 1)
    expected = expected.replace('    fruit_report();', '    fruit_report();\n    l1_model_finish();', 1)
    assert modified == expected, 'unexpected model edits'
    gates = json.loads((work / 'enabled/results.json').read_text())
    assert gates['gate'] == 'PASS' and gates['items_integrated'] == list(range(13, 0, -1))
    assert (gates['unique_rows'], gates['unique_pairs'], gates['unique_groups']) == (1926, 214, 5778)
    assert json.loads((work / 'disabled/results.json').read_text())['gate'] == 'PASS'
    results, all_experts = [], set()
    candidate_hash = digest(work / 'candidate')
    for name in ('france', 'japan'):
        folder = work / 'enabled'
        launch = json.loads((folder / (name + '-launch.json')).read_text())
        assert launch['binary_sha256'] == candidate_hash
        routes = read_routes(work / (name + '-reference.routes'))
        appearances = Counter(expert for position in range(5) for expert in routes[1, position])
        all_experts.update(appearances)
        events = [json.loads(line) for line in (folder / (name + '-compact.jsonl')).read_text().splitlines()]
        summary = events[-1]
        assert summary['type'] == 'summary' and all(event['type'] == 'projection' for event in events[:-1])
        wanted = {(expert, matrix) for expert in appearances for matrix in range(3)}
        observed = set()
        bytes_read = outputs = blocks = weights = 0
        for event in events[:-1]:
            expert, matrix = event['expert'], event['matrix']
            key = (expert, matrix)
            assert key in wanted and key not in observed and event['layer'] == 1
            observed.add(key)
            rows, width = (3584, 3072) if matrix == 2 else (3072, 3584)
            first = expert * 152 + matrix * 48
            expected_bytes = sum(lengths[first:first + rows // 64])
            assert event['positions'] == appearances[expert]
            assert (event['rows'], event['width'], event['blocks']) == (rows, width, rows // 64)
            assert event['compressed_bytes'] == expected_bytes and event['crc_pass']
            bytes_read += expected_bytes
            outputs += rows * appearances[expert]
            weights += rows * width
            blocks += rows // 64
        assert observed == wanted
        assert (summary['blocks'], summary['compressed_bytes'], summary['weights_consumed'], summary['projection_values']) == (blocks, bytes_read, weights, outputs)
        assert summary['packed_buffer_bytes'] == 121856 and summary['compressed_buffer_bytes'] == max(lengths)
        assert summary['index_bytes'] == 1094912 and summary['derived_offset_bytes'] == 1089544
        assert summary['weight_array_bytes'] == 0 and summary['checksum_array_bytes'] == 256
        assert summary['double_accumulator_bytes_per_worker'] == 640 and summary['positions_capacity'] == 5
        checked = json.loads((folder / (name + '-compact-check.json')).read_text())
        assert checked['gate'] == 'PASS' and checked['projection_calls'] == len(wanted)
        assert checked['projection_values_checked'] == outputs and checked['situ_values_checked'] == sum(appearances.values()) * 3072
        assert checked['compact_outputs_feed_model'] and checked['original_packed_data_retained_for_oracle']
        for suffix in ('.bin', '.routes'):
            actual = (folder / (name + suffix)).read_bytes()
            assert actual == (work / (name + '-reference' + suffix)).read_bytes()
        assert (folder / (name + '.bin')).stat().st_size == 171008 * 4
        seed = json.loads((folder / (name + '-seed.json')).read_text())
        fruit = json.loads((folder / (name + '-fruit.json')).read_text())
        assert seed['gate'] == fruit['gate'] == 'PASS'
        assert seed['guard_probe_sigsegv'] and fruit['guard_probe_sigsegv'] and seed['entry_uses_seed_rows']
        assert seed['values_loaded'] == 35840 and seed['original_embedding_protected_bytes'] == 2348806144
        assert fruit['original_head_protected_bytes'] == 2348806144
        prior_run = next(run for run in previous['runs'] if run['prompt'] == name)
        assert fruit['passes'] == prior_run['fruit']['passes']
        results.append({'prompt': name, 'experts': sorted(appearances), 'expert_count': len(appearances),
                        'positions_per_expert': dict(appearances), 'stage_checks': checked, 'consumer': summary,
                        'all_reference_routes_match': True, 'all_norm_and_logits_bits_match': True,
                        'full_output_float32_count': 171008,
                        'full_output_md5': hashlib.md5((folder / (name + '.bin')).read_bytes()).hexdigest(),
                        'seed_and_fruit_gates_pass': True,
                        'trace_sha256': digest(folder / (name + '-compact.jsonl'))})
    for folder in ('disabled', 'prior-control'):
        for suffix in ('.bin', '.routes'):
            assert (work / folder / ('france' + suffix)).read_bytes() == (work / ('france-reference' + suffix)).read_bytes()
        assert not (work / folder / 'france-compact.jsonl').exists()
        assert not (work / folder / 'france-compact-check.json').exists()
    rejections = []
    original_launch = json.loads((work / 'enabled/france-launch.json').read_text())
    for name, path, error in (('missing', work / 'missing-experts.bin', 'cannot open dataset'),
                              ('wrong-format', maps.parent / 'layer1-constant-store-20260929-a/data/experts.bin', 'wrong format or constant identity')):
        folder = work / ('reject-' + name)
        folder.mkdir()
        env = {key: value for key, value in os.environ.items() if not key.startswith('K3_')}
        for key, value in original_launch['environment'].items():
            env[key] = str(folder / Path(value).name) if str(work / 'enabled') in value else value
        env['K3_L1_COMPACT'] = str(path)
        with (folder / 'stdout.log').open('xb') as stdout, (folder / 'stderr.log').open('xb') as stderr:
            completed = subprocess.run([str(work / 'candidate')], cwd=folder, env=env, stdout=stdout, stderr=stderr)
        assert completed.returncode != 0 and error in (folder / 'stderr.log').read_text()
        assert not Path(env['K3_LOGITS']).exists() and not Path(env['K3_L1_CHECK']).exists()
        rejections.append({'case': name, 'exit_code': completed.returncode, 'expected_error': error, 'no_logits_written': True})
    assert digest(maps / 'experts.bin') == dataset['dataset_sha256']
    for filename, expected in previous['protected_sha256'].items():
        assert digest(filename) == expected
    assert digest('/opt/clover-k3/seed/seed.bin') == previous['seed_sha256_unchanged']
    assert digest('/opt/clover-k3/fruit/fruit.bin') == previous['fruit_dataset']['fruit_sha256']
    files = [path for path in work.iterdir() if path.is_file() and (path.suffix in ('.c', '.h', '.py', '.patch', '.json') or path.name == 'candidate')]
    report = {'gate': 'PASS', 'layer': 1, 'runs': results, 'unique_layer1_experts': len(all_experts),
              'unique_layer1_expert_ids': sorted(all_experts), 'all_prior_13_stage_gates_pass': True,
              'prior_gate_sha256': digest(work / 'enabled/results.json'),
              'disabled_control_pass': True, 'preserved_prior_model_control_pass': True, 'rejections': rejections,
              'dataset_sha256_unchanged': dataset['dataset_sha256'], 'prior_model_sha256_unchanged': previous['sha256']['candidate'],
              'protected_sha256_unchanged': previous['protected_sha256'],
              'implementation_and_evidence_sha256': {path.name: digest(path) for path in files},
              'build_flags': json.loads((work / 'compile-command.json').read_text()),
              'scope': 'All three full matrices of every routed expert actually selected in layer1, for two five-token prefills. Direct compact gate/up/down outputs feed model; SiTU and final outputs/routes compared bitwise. Other layers and all prior seed/fruit/representation checks retained.',
              'limits': 'Correctness integration, not optimized deployment or performance benchmark. Original packed source staging and projection oracle remain; old sampled checks rewrite three verified-equal rows per matrix. Existing gate/up activation arrays remain (no tiled SiTU fusion yet). Compact path has no expanded weight array; reported explicit buffers exclude zlib/OpenMP internals, oracle/model activations and process RSS. No other prompts, generation or longer-context guarantee.'}
    with (work / 'results.json').open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({'gate': report['gate'], 'prompts': len(results), 'unique_layer1_experts': len(all_experts),
                      'results_sha256': digest(work / 'results.json'), 'runs': results, 'rejections': rejections}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    main(parser.parse_args().work)