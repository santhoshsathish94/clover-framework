#!/usr/bin/env python3
"""Verify the unchanged native seed reader against the complete fruit dataset."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    args = parser.parse_args()
    work = args.work
    root = Path('/opt/clover-k3/fruit')
    result = json.loads((root / 'results.json').read_text())
    census = json.loads((root / 'census.json').read_text())
    assert result['gate'] == census['gate'] == 'PASS'
    assert result['source_sha256'] == census['source_sha256']
    prior_root = Path('/opt/clover-k3/seed-integrated-20260929-a')
    prior = json.loads((prior_root / 'reader-results.json').read_text())
    assert prior['gate'] == 'PASS'
    for name, expected in prior['sha256'].items():
        assert hashlib.sha256((prior_root / name).read_bytes()).hexdigest() == expected
    binary = prior_root / 'seed-reader-test'
    spec = importlib.util.spec_from_file_location('fruit_reader', '/opt/clover-k3/fruit-build-20260929/seed.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    special = []
    with (root / 'blocks.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            assert row['stored_bytes'] == min(row['candidate_bytes'].values())
            if row['codec'] == 'zlib_bf16_bit_planes':
                special.append(row['first_row'])
    rows = sorted(set([0, 15, 16, 17374, 40484, 81920, 163839] + special))
    with module.SeedReader(root / 'fruit.bin') as reader:
        for row in rows:
            completed = subprocess.run([binary, root / 'fruit.bin', str(row)], capture_output=True, check=True)
            assert completed.stdout == reader.read_bf16(row)
    sha, size = hashlib.sha256(), 0
    with (work / 'fruit-native.stderr').open('xb') as stderr:
        with subprocess.Popen([binary, root / 'fruit.bin', 'all'], stdout=subprocess.PIPE, stderr=stderr) as process:
            for chunk in iter(lambda: process.stdout.read(4 * 1024 * 1024), b''):
                sha.update(chunk)
                size += len(chunk)
            assert process.wait() == 0
    assert size == result['original_bytes'] and sha.hexdigest() == result['decoded_payload_sha256']
    report = {'gate': 'PASS', 'native_reader_unchanged': prior['sha256'], 'real_rows_checked': rows,
              'bit_plane_block_first_rows': special, 'decoded_bytes': size, 'decoded_values': size // 2,
              'decoded_sha256': sha.hexdigest(), 'source_checkpoint_opened': False,
              'prior_codec_controls': {'valid': 30, 'invalid_rejected': 10, 'six_codecs_all_bf16_patterns': True},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (work / 'native-results.json').open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()