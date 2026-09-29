#!/usr/bin/env python3
"""Verify performance arithmetic and per-trial evidence, without new model runs."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    report = json.loads((root / 'results.json').read_text())
    assert report['gate'] == 'PASS' and len(report['runs']) == 12 and len(report['warmups']) == 4
    assert report['all_full_outputs_and_routes_bit_exact'] and report['all_integrated_representation_gates_passed']
    metadata = json.loads((root / 'metadata.json').read_text())
    assert report['metadata'] == metadata
    source = Path('/opt/clover-k3/integrated-performance-20260929.py')
    assert digest(source) == metadata['script_sha256']
    seed = Path('/opt/clover-k3/reverse-integration-20260929-a/step1')
    assert digest(seed / 'results.json') == metadata['seed_result_sha256']
    assert digest(Path('/opt/clover-k3/config.env')) == metadata['config_sha256']
    for path, expected in metadata['protected_sha256'].items():
        assert digest(Path(path)) == expected
    expected_order = []
    for prompt_index, prompt in enumerate(('france', 'japan')):
        for repetition in (1, 2, 3):
            order = ('reference', 'integrated') if (prompt_index + repetition) % 2 else ('integrated', 'reference')
            expected_order.extend(f'measure-{prompt}-{repetition}-{variant}' for variant in order)
    assert [run['name'] for run in report['runs']] == expected_order
    archive_files = {root / 'metadata.json', root / 'results.json', root / 'warmup-results.json'}
    all_runs = report['warmups'] + report['runs']
    assert len({run['name'] for run in all_runs}) == 16
    for run in all_runs:
        trial = root / run['name']
        assert json.loads((trial / 'run-result.json').read_text()) == run
        measured = json.loads((trial / 'measurement.json').read_text())
        assert measured == run['metrics'] and measured['exit_code'] == 0
        assert measured['elapsed_seconds'] > 0 and measured['peak_rss_kib'] > 0
        assert measured['peak_rss_gib'] == measured['peak_rss_kib'] / (1024 ** 2)
        assert run['gate'] == 'PASS' and run['full_output_bit_exact'] and run['routes_bit_exact']
        binary = seed / ('candidate' if run['variant'] == 'integrated' else 'reference')
        assert digest(binary) == run['binary_sha256']
        prompt = run['prompt']
        output = (trial / (prompt + '.bin')).read_bytes()
        assert output == (seed / (prompt + '-reference.bin')).read_bytes()
        assert (trial / (prompt + '.routes')).read_bytes() == (seed / (prompt + '-reference.routes')).read_bytes()
        assert hashlib.sha256(output).hexdigest() == run['output_sha256']
        assert hashlib.md5(output).hexdigest() == run['output_md5']
        assert digest(trial / (prompt + '.routes')) == run['routes_sha256']
        if run['variant'] == 'integrated':
            assert digest(trial / 'validation.json') == run['validation_sha256']
            validation = json.loads((trial / 'validation.json').read_text())
            assert validation['gate'] == 'PASS' and validation['items_integrated'] == list(range(13, 0, -1))
            assert len(validation['runs']) == 1 and validation['runs'][0]['full_output']['bit_exact']
            assert digest(trial / (prompt + '.jsonl')) == validation['runs'][0]['raw_trace_sha256']
        for filename in ('measurement.json', 'run-result.json', 'model.log', 'model.stderr',
                         'validation.json', 'validation.log', 'validation.stderr', prompt + '.bin',
                         prompt + '.routes', prompt + '.jsonl', prompt + '-basis.bin',
                         prompt + '-vector.bin', prompt + '-endpoints.json'):
            path = trial / filename
            if path.exists():
                archive_files.add(path)
    for prompt, summary in report['summary'].items():
        for variant, metrics in summary['variants'].items():
            runs = [run for run in report['runs'] if run['prompt'] == prompt and run['variant'] == variant]
            assert len(runs) == 3
            for field, aggregate in metrics.items():
                values = [run['metrics'][field] for run in runs]
                assert aggregate == {'median': statistics.median(values), 'minimum': min(values),
                                     'maximum': max(values), 'values': values}
        reference = summary['variants']['reference']
        integrated = summary['variants']['integrated']
        ratio = integrated['elapsed_seconds']['median'] / reference['elapsed_seconds']['median']
        assert ratio == summary['integrated_time_over_reference']
        assert (ratio - 1) * 100 == summary['elapsed_change_percent']
        assert integrated['peak_rss_gib']['median'] - reference['peak_rss_gib']['median'] == summary['peak_rss_delta_gib']
        for repetition, paired_ratio in enumerate(summary['paired_time_ratios'], 1):
            pair = {run['variant']: run['metrics']['elapsed_seconds'] for run in report['runs']
                    if run['prompt'] == prompt and run['repetition'] == repetition}
            assert pair['integrated'] / pair['reference'] == paired_ratio
    verification = {'gate': 'PASS', 'measured_runs': 12, 'warmups': 4, 'integrated_gate_passes': 8,
                    'all_output_routes_and_source_hashes_match': True, 'all_summary_arithmetic_matches': True,
                    'rss_unit': 'KiB per Linux wait4; GiB = KiB / 1048576',
                    'result_sha256': digest(root / 'results.json'),
                    'measurement_script_sha256': digest(source), 'verifier_sha256': digest(Path(__file__))}
    with (root / 'verification.json').open('x') as destination:
        json.dump(verification, destination, indent=2)
        destination.write('\n')
    archive_files.add(root / 'verification.json')
    with (root / 'evidence.tar.gz').open('xb') as destination:
        with tarfile.open(fileobj=destination, mode='w:gz') as bundle:
            for path in sorted(archive_files):
                bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
            bundle.add(source, arcname='measurement-wrapper.py', recursive=False)
            bundle.add(Path(__file__), arcname='measurement-check.py', recursive=False)
    print(json.dumps({'verification': verification, 'archive_bytes': (root / 'evidence.tar.gz').stat().st_size,
                      'archive_sha256': digest(root / 'evidence.tar.gz')}, indent=2))


if __name__ == '__main__':
    main()