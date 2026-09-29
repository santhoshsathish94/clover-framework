#!/usr/bin/env python3
"""Verify and preserve the thirteen separately gated cumulative candidates."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    protected = {
        '/opt/clover-k3/clover-k3.c': '5628f7c3d2b7932b2b8776dcb498d54bbc730b1785ba7819989bdb51b6604e65',
        '/opt/clover-k3/build/clover-k3': '4cebb69e5379c4f0c9c03d4071c49840d3d7596ecfd2c5c0d8b83f6679717d35',
        '/opt/clover-k3/gate.sh': '752291aa5cda8194bbaf1f68d6a843e9ca02d3e221e8e19a5b714c075b01f5dc',
        '/opt/clover-k3/build/eqidx.bin': 'c2b7962c88fe7d984260a9c5d8f1b23ba0866e9cb9a39bf35a689f6a6a906981',
    }
    for name, expected in protected.items():
        assert digest(Path(name)) == expected
    for suffix in ('bin', 'routes'):
        assert (root / ('france-reference-after.' + suffix)).read_bytes() == (root / 'step13' / ('france-reference.' + suffix)).read_bytes()
    steps, artifacts = [], set()
    for stage in range(13, 0, -1):
        work = root / ('step' + str(stage))
        report = json.loads((work / 'results.json').read_text())
        assert report['stage'] == stage and report['gate'] == 'PASS'
        assert report['items_integrated'] == list(range(13, stage - 1, -1))
        assert (report['unique_rows'], report['unique_pairs'], report['unique_groups'], report['unique_scales']) == (1926, 214, 5778, 205440)
        assert report['adaptive_scale_bytes'] == 23830
        for name, expected in report['sha256'].items():
            path = work / name
            assert digest(path) == expected
            artifacts.add(path)
        artifacts.add(work / 'results.json')
        if (work / 'reverse-run.sh').exists():
            artifacts.add(work / 'reverse-run.sh')
        broad = [run for run in report['runs'] if run['name'] != 'smoke']
        assert [run['name'] for run in broad] == ['france', 'japan']
        assert sum(run['summary']['projection_outputs'] for run in broad) == 4320
        for run in report['runs']:
            name = run['name']
            reference = 'japan' if name == 'japan' else 'france'
            assert run['gate'] == 'PASS' and run['full_output']['bit_exact']
            assert run['full_output']['norm']['changed'] == run['full_output']['logits']['changed'] == 0
            assert run['reference_routes_identical'] and run['coverage_matches_reference_route_expectations']
            assert digest(work / (name + '.jsonl')) == run['raw_trace_sha256']
            for suffix in ('bin', 'routes'):
                assert (work / (name + '.' + suffix)).read_bytes() == (work / (reference + '-reference.' + suffix)).read_bytes()
        steps.append({'stage': stage, 'items_integrated': report['items_integrated'], 'gate': 'PASS',
                      'broad_projections': 4320, 'result_sha256': digest(work / 'results.json'),
                      'candidate_sha256': digest(work / 'candidate'),
                      'runner_sha256': digest(work / 'reverse-run.sh') if (work / 'reverse-run.sh').exists() else None,
                      'outputs': {run['name']: run['full_output'] for run in broad}})
        if stage in (13, 1):
            for name in ('france', 'japan'):
                for path in work.glob(name + '*'):
                    if path.is_file() and path.suffix in ('.bin', '.routes', '.json'):
                        artifacts.add(path)
    for suffix in ('bin', 'routes'):
        artifacts.add(root / ('france-reference-after.' + suffix))
    dependency_paths = list((root / 'deps').glob('*.deb')) + [
        root / 'deps/gmp/usr/include/x86_64-linux-gnu/gmp.h',
        root / 'deps/gmp/usr/lib/x86_64-linux-gnu/libgmp.a',
        root / 'deps/zlib/usr/include/zlib.h', root / 'deps/zlib/usr/include/zconf.h',
        root / 'deps/zlib/usr/lib/x86_64-linux-gnu/libz.a',
    ]
    artifacts.update(dependency_paths)
    dependencies = {str(path.relative_to(root)): digest(path) for path in dependency_paths}
    summary = {
        'gate': 'PASS', 'order': list(range(13, 0, -1)), 'steps': steps,
        'broad_model_runs': 26, 'broad_projection_comparisons': 13 * 4320,
        'unique_final_scope': {'rows': 1926, 'layer_expert_pairs': 214, 'groups': 5778,
                               'scales': 205440, 'weights_using_decoded_scales': 6574080,
                               'vector_components_per_prompt': 7168, 'prompts': 2},
        'protected_sha256': protected, 'dependency_sha256': dependencies,
        'reference_repeated_matches': True,
        'build': '-O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5 -DREVERSE_STAGE=N; private GMP; private zlib from step3; -lm',
        'environment': {'OMP_NUM_THREADS': '16', 'OMP_PROC_BIND': 'close', 'OMP_PLACES': 'cores',
                        'K3_PREFETCH': '4', 'K3_TRUNKRAM': '0', 'K3_NREADER': '14', 'K3_XDEC': '2',
                        'K3_PLGRAN': '1', 'K3_HUGE': '1', 'K3_GROUP_COVERAGE': '2'},
        'scope': 'Sampled expert sites at layers1/48/92; router/vector codecs at L84/pos0; five-token entry and final tail; France/Japan prefills.',
        'limits': 'Intermediate representations composed for correctness. Original data, reference computations and diagnostics retained. Not a persistent weight store, checkpoint-wide codec, decode/generation test, or measured RAM/storage/speed improvement.',
        'raw_evidence': str(root),
        'preserved_warning_build': str(root / 'step8-before-bound'),
    }
    output = root / 'campaign-results.json'
    with output.open('x') as destination:
        json.dump(summary, destination, indent=2)
        destination.write('\n')
    artifacts.add(output)
    artifacts.add(root / 'reverse-summary.py')
    archive = root / 'verified-snapshots.tar.gz'
    with archive.open('xb') as destination:
        with tarfile.open(fileobj=destination, mode='w:gz') as bundle:
            for path in sorted(artifacts):
                bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
    print(json.dumps({'gate': 'PASS', 'steps': len(steps), 'broad_runs': 26,
                      'projection_comparisons': 56160, 'archive_bytes': archive.stat().st_size,
                      'archive_sha256': digest(archive), 'report_sha256': digest(output)}, indent=2))


if __name__ == '__main__':
    main()