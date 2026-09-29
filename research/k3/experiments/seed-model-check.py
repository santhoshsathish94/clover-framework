#!/usr/bin/env python3
"""Verify consumed seed rows, preserved gates and model-integration provenance."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tarfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    work = args.directory
    seed_root = Path('/opt/clover-k3/seed')
    previous = Path('/opt/clover-k3/reverse-integration-20260929-a/step1')
    prior = json.loads((work / 'prior-results.json').read_text())
    assert prior['gate'] == 'PASS'
    preserved = {}
    for name, expected in prior['sha256'].items():
        if name in ('candidate.c', 'vector-endpoints.h'):
            assert digest(previous / name) == expected
            continue
        path = work / ('prior-candidate' if name == 'candidate' else name)
        assert digest(path) == expected, name
        preserved[name] = expected
    protected = {
        '/opt/clover-k3/clover-k3.c': '5628f7c3d2b7932b2b8776dcb498d54bbc730b1785ba7819989bdb51b6604e65',
        '/opt/clover-k3/build/clover-k3': '4cebb69e5379c4f0c9c03d4071c49840d3d7596ecfd2c5c0d8b83f6679717d35',
        '/opt/clover-k3/gate.sh': '752291aa5cda8194bbaf1f68d6a843e9ca02d3e221e8e19a5b714c075b01f5dc',
        '/opt/clover-k3/build/eqidx.bin': 'c2b7962c88fe7d984260a9c5d8f1b23ba0866e9cb9a39bf35a689f6a6a906981',
    }
    for path, expected in protected.items():
        assert digest(Path(path)) == expected
    seed_result = json.loads((seed_root / 'results.json').read_text())
    assert digest(seed_root / 'seed.bin') == seed_result['seed_sha256']
    reader_result = json.loads((work / 'reader-results.json').read_text())
    assert reader_result['gate'] == 'PASS' and reader_result['valid_fixture_checks'] == 30
    assert reader_result['invalid_inputs_rejected'] == 10 and reader_result['full_native_decoded_values'] == 1174405120
    for name, expected in reader_result['sha256'].items():
        assert digest(work / name) == expected
    spec = importlib.util.spec_from_file_location('seed_python', seed_root / 'seed.py')
    seed_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed_module)
    integrated = json.loads((work / 'enabled/results.json').read_text())
    disabled = json.loads((work / 'disabled/france-check.json').read_text())
    assert integrated['gate'] == disabled['gate'] == 'PASS'
    assert integrated['items_integrated'] == list(range(13, 0, -1))
    assert (integrated['unique_rows'], integrated['unique_pairs'], integrated['unique_groups']) == (1926, 214, 5778)
    assert sum(run['summary']['projection_outputs'] for run in integrated['runs']) == 4320
    source = Path('/root/k3model/model-00094-of-000096.safetensors')
    source_before = source.stat()
    runs = []
    with source.open('rb') as stream, seed_module.SeedReader(seed_root / 'seed.bin') as seed:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
        record = header['language_model.model.embed_tokens.weight']
        assert record['shape'] == [163840, 7168] and record['dtype'] == 'BF16'
        offset = 8 + size + record['data_offsets'][0]
        for run in integrated['runs']:
            name, ids = run['name'], run['prompt_ids']
            assert run['gate'] == 'PASS' and run['full_output']['bit_exact']
            assert run['reference_routes_identical'] and run['coverage_matches_reference_route_expectations']
            assert run['full_output']['norm']['changed'] == run['full_output']['logits']['changed'] == 0
            assert digest(work / 'enabled' / (name + '.jsonl')) == run['raw_trace_sha256']
            expected_rows = bytearray()
            for token in ids:
                stream.seek(offset + token * 7168 * 2)
                raw = stream.read(7168 * 2)
                assert raw == seed.read_bf16(token)
                expected_rows.extend(raw)
            actual_path = work / 'enabled' / (name + '-seed-rows.bf16')
            assert actual_path.read_bytes() == expected_rows
            seed_report = json.loads((work / 'enabled' / (name + '-seed.json')).read_text())
            assert seed_report['gate'] == 'PASS' and seed_report['source'] == 'seed'
            assert seed_report['rows_loaded'] == 5 and seed_report['values_loaded'] == 35840
            assert seed_report['blocks_decoded'] == 5 and seed_report['retained_row_bytes'] == 71680
            assert seed_report['original_embedding_protected_bytes'] == 2348806144
            assert seed_report['unprotected_boundary_bytes'] == 4096
            assert seed_report['guard_probe_sigsegv'] and seed_report['entry_uses_seed_rows']
            payload_bytes = sum(seed.entries[token // seed.block_rows][1] for token in ids)
            assert seed_report['payload_bytes_read'] == payload_bytes
            for suffix in ('bin', 'routes'):
                assert (work / 'enabled' / (name + '.' + suffix)).read_bytes() == (work / (name + '-reference.' + suffix)).read_bytes()
            runs.append({'prompt': name, 'ids': ids, 'full_output': run['full_output'], 'routes_identical': True,
                         'consumed_bf16_rows_sha256': digest(actual_path), 'source_rows_match': True,
                         'seed_report': seed_report, 'prior_stages_pass': True})
    assert source.stat().st_mtime_ns == source_before.st_mtime_ns
    for suffix in ('bin', 'routes'):
        reference = (work / ('france-reference.' + suffix)).read_bytes()
        for relative in ('prior/france.', 'disabled/france.', 'france-reference-after.'):
            assert (work / (relative + suffix)).read_bytes() == reference
    assert not (work / 'disabled/france-seed.json').exists()
    assert 'seed open failed' in (work / 'missing-seed.stderr').read_text()
    olddeps = Path('/opt/clover-k3/reverse-integration-20260929-a/deps')
    dependencies = {
        'gmp_static': olddeps / 'gmp/usr/lib/x86_64-linux-gnu/libgmp.a',
        'zlib_static': olddeps / 'zlib/usr/lib/x86_64-linux-gnu/libz.a',
        'openssl_static': work / 'deps/openssl/usr/lib/x86_64-linux-gnu/libcrypto.a',
        'openssl_sha_header': work / 'deps/openssl/usr/include/openssl/sha.h',
    }
    packages = list((work / 'deps').glob('libssl-dev_*.deb'))
    assert len(packages) == 1
    dependencies['openssl_package'] = packages[0]
    names = sorted(path.name for path in work.iterdir() if path.is_file() and
                   (path.suffix in ('.h', '.c', '.py', '.patch') or path.name in ('candidate', 'reference', 'prior-candidate')))
    report = {
        'gate': 'PASS', 'host': 'AX102 AMD Ryzen 9 7950X3D', 'seed': seed_result['seed_sha256'],
        'reader_controls': reader_result, 'runs': runs,
        'seed_disabled_control_passed': True, 'prior_integrated_control_passed': True,
        'repeated_reference_matches': True, 'missing_seed_rejected_without_fallback': True,
        'full_prior_stage_gate_result_sha256': digest(work / 'enabled/results.json'),
        'protected_sha256': protected, 'unchanged_prior_files': preserved,
        'sha256': {name: digest(work / name) for name in names},
        'dependencies': {name: {'path': str(path), 'sha256': digest(path)} for name, path in dependencies.items()},
        'build_flags': '-O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5 -DREVERSE_STAGE=1; private GMP/zlib/OpenSSL static archives; -ldl -pthread -lm',
        'settings': {'OMP_NUM_THREADS': '16', 'OMP_PROC_BIND': 'close', 'OMP_PLACES': 'cores',
                     'K3_PREFETCH': '4', 'K3_TRUNKRAM': '0', 'K3_NREADER': '14', 'K3_XDEC': '2',
                     'K3_PLGRAN': '1', 'K3_HUGE': '1', 'K3_GROUP_COVERAGE': '2', 'K3_VECTOR_ENDPOINTS': '3',
                     'K3_BASIS_ROUTER': '2', 'K3_SEED_INPUT': str(seed_root / 'seed.bin'), 'K3_SEED_GUARD_PROBE': '1'},
        'scope': 'Both tested five-token prefills use seed for initial and entry-normalization rows; every prior sampled representation remains active. Native decoder separately verified over the complete input table.',
        'limits': 'No generation, new context lengths, model latency or peak RAM measurement. Other checkpoint tensors and reference computations remain. PROT_NONE guards the mapped embedding interior, not 4096 bytes of shared boundary pages or arbitrary alternative file access. Source/seed data and installed model unchanged; no system package installation.'}
    with (work / 'results.json').open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    with (work / 'verified-artifacts.tar.gz').open('xb') as destination:
        with tarfile.open(fileobj=destination, mode='w:gz') as archive:
            for name in names:
                archive.add(work / name, arcname='model/' + name, recursive=False)
            archive.add(work / 'reader-results.json', arcname='reader-results.json', recursive=False)
            archive.add(work / 'results.json', arcname='results.json', recursive=False)
            for folder in ('enabled', 'disabled', 'prior'):
                for path in sorted((work / folder).iterdir()):
                    if path.is_file() and path.suffix in ('.bin', '.routes', '.bf16', '.json', '.jsonl', '.log', '.stderr'):
                        archive.add(path, arcname=folder + '/' + path.name, recursive=False)
            for path in sorted(work.glob('*reference*')):
                if path.is_file() and path.suffix in ('.bin', '.routes', '.log', '.stderr'):
                    archive.add(path, arcname=path.name, recursive=False)
            archive.add(work / 'missing-seed.stderr', arcname='missing-seed.stderr', recursive=False)
    print(json.dumps({'gate': 'PASS', 'prompts': len(runs), 'seed_input_values_compared': 71680,
                      'original_embedding_guarded_bytes': 2348806144, 'guard_probes_passed': 2,
                      'result_sha256': digest(work / 'results.json'),
                      'archive_sha256': digest(work / 'verified-artifacts.tar.gz'),
                      'archive_bytes': (work / 'verified-artifacts.tar.gz').stat().st_size}, indent=2))


if __name__ == '__main__':
    main()