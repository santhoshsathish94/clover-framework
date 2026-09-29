#!/usr/bin/env python3
"""Independent native-reader controls and complete seed reconstruction hash."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import zlib

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    args = parser.parse_args()
    work = args.work
    fixtures = work / 'reader-fixtures'
    fixtures.mkdir()
    executable = work / 'seed-reader-test'
    header_struct = struct.Struct('<8s8I3Q32s32s')
    entry_struct = struct.Struct('<Q4I32s')
    words = np.arange(65536, dtype='<u2')
    raw = words.tobytes()
    dictionary = raw
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 2), axis=1, bitorder='little')
    planes = np.packbits(bits.T.reshape(-1), bitorder='little').tobytes()
    options = (raw, zlib.compress(raw), zlib.compress(planes), zlib.compress(raw),
               zlib.compress(raw[::2] + raw[1::2]), zlib.compress(planes))

    def fixture(codec, payload, path, mutate=None):
        offset = 128 + len(dictionary) + 56
        fields = [offset, len(payload), 256, codec, zlib.crc32(payload), hashlib.sha256(raw).digest()]
        if mutate:
            mutate(fields)
        index = entry_struct.pack(*fields)
        values = (b'K3SEED1\0', 1, 256, 256, 256, 65536, 16, 1, 0,
                  128, 128 + len(dictionary), offset, hashlib.sha256(raw).digest())
        header = header_struct.pack(*values, bytes(32))
        header = header_struct.pack(*values, hashlib.sha256(header[:-32] + dictionary + index).digest())
        path.write_bytes(header + dictionary + index + payload)

    valid_checks = 0
    for codec, payload in enumerate(options):
        path = fixtures / f'codec{codec}.bin'
        fixture(codec, payload, path)
        completed = subprocess.run([executable, path, 'all'], capture_output=True, check=True)
        assert completed.stdout == raw
        for row in (0, 15, 16, 255):
            completed = subprocess.run([executable, path, str(row)], capture_output=True, check=True)
            assert completed.stdout == raw[row * 512:(row + 1) * 512]
        valid_checks += 5
    damaged = []
    base = (fixtures / 'codec4.bin').read_bytes()
    for name, data, message in (
        ('short_header', base[:127], 'file size invalid'),
        ('truncated', base[:-1], 'outside file'),
        ('metadata', base[:128] + bytes([base[128] ^ 1]) + base[129:], 'metadata checksum differs'),
        ('payload', base[:-1] + bytes([base[-1] ^ 1]), 'payload checksum differs'),
        ('trailing', base + b'\x00', 'trailing bytes'),
    ):
        path = fixtures / (name + '.bin')
        path.write_bytes(data)
        damaged.append((path, 'all', message))
    for name, mutation, message in (
        ('codec', lambda fields: fields.__setitem__(3, 6), 'block index invalid'),
        ('offset', lambda fields: fields.__setitem__(0, fields[0] + 1), 'block index invalid'),
        ('decoded_hash', lambda fields: fields.__setitem__(5, bytes(32)), 'reconstructed block checksum differs'),
    ):
        path = fixtures / (name + '.bin')
        fixture(4, options[4], path, mutation)
        damaged.append((path, 'all', message))
    path = fixtures / 'zlib_trailing.bin'
    fixture(4, options[4] + b'\x00', path)
    damaged.append((path, 'all', 'compressed block framing invalid'))
    damaged.append((fixtures / 'codec4.bin', '256', 'row out of range'))
    for path, mode, message in damaged:
        completed = subprocess.run([executable, path, mode], capture_output=True)
        assert completed.returncode == 2 and message in completed.stderr.decode(), (path, completed.stderr)
    spec = importlib.util.spec_from_file_location('verified_seed', '/opt/clover-k3/seed/seed.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seed_path = Path('/opt/clover-k3/seed/seed.bin')
    rows = (0, 15, 16, 1008, 10484, 81920, 163839)
    with module.SeedReader(seed_path) as reader:
        for row in rows:
            completed = subprocess.run([executable, seed_path, str(row)], capture_output=True, check=True)
            assert completed.stdout == reader.read_bf16(row)
    sha, size = hashlib.sha256(), 0
    with (work / 'full-reader.stderr').open('xb') as stderr:
        with subprocess.Popen([executable, seed_path, 'all'], stdout=subprocess.PIPE, stderr=stderr) as process:
            for chunk in iter(lambda: process.stdout.read(4 * 1024 * 1024), b''):
                sha.update(chunk)
                size += len(chunk)
            assert process.wait() == 0
    census = json.loads(Path('/opt/clover-k3/input-table-values-20260929-a/results.json').read_text())
    assert size == census['tensor_bytes'] and sha.hexdigest() == census['input_payload_sha256']
    report = {'gate': 'PASS', 'valid_fixture_checks': valid_checks, 'invalid_inputs_rejected': len(damaged),
              'all_65536_bf16_patterns_per_codec': True, 'real_seed_rows_checked': rows,
              'full_native_decoded_bytes': size, 'full_native_decoded_values': size // 2,
              'decoded_sha256': sha.hexdigest(), 'source_checkpoint_opened': False,
              'sha256': {name: hashlib.sha256((work / name).read_bytes()).hexdigest()
                         for name in ('seed-reader.h', 'seed-reader-test.c', 'seed-reader-test', 'seed-reader-check.py')}}
    with (work / 'reader-results.json').open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()