#!/usr/bin/env python3
"""Independent container controls and checkpoint-free actual row reads."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import zlib

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('root_store', HERE / 'root-table-store.py')
store = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(store)


def controls():
    rejected = []

    def rejects(name, action):
        try:
            action()
        except (ValueError, zlib.error):
            rejected.append(name)
        else:
            raise AssertionError('Invalid input accepted: ' + name)

    with tempfile.TemporaryDirectory(prefix='root-table-controls-') as temporary:
        folder = Path(temporary)
        root = struct.pack('<HHHH', 1, 0, 0x8000, 0x3d80)
        tables = {'root': root, 'root-up': b'\x01\x00', 'root-down': b'\x01\x00'}
        for name, data in tables.items():
            (folder / (name + '.bin')).write_bytes(data)
        metadata = {'layer': 1, 'block_rows': 64, 'codecs': list(store.CODECS),
                    'tables': {name: {'values': (len(data) - 2) // 2,
                                      'sha256': hashlib.sha256(data).hexdigest()}
                               for name, data in tables.items()}, 'experts': []}
        payload_path = folder / 'payload'
        expected_rows = {}
        with payload_path.open('xb') as output:
            for expert in range(896):
                for matrix, (rows, width) in zip(store.MATRICES, store.SHAPES):
                    expected_row = struct.pack('<HHHH', 0, 0x8000, 0x3d80, 0x3c80) * (width // 4)
                    expected_rows[matrix] = expected_row
                    expected = expected_row * 64
                    packed = b'\x10\x32' * (width * 64 // 4)
                    grouped = packed + bytes(width * 64 // 32)
                    compressed = zlib.compress(grouped, 6)
                    node = {'root_id': 1, 'expert_id': expert, 'matrix': matrix,
                            'local_count': 1, 'local_offset': output.tell(),
                            'group_scales': [121], 'blocks': [],
                            'value_sha256': hashlib.sha256(expected_row * rows).hexdigest()}
                    output.write(struct.pack('<H', 0x3c80))
                    node['group_palette_offset'] = output.tell()
                    output.write(bytes(range(4)) + bytes([255]) * 12)
                    for block in range(rows // 64):
                        node['blocks'].append([output.tell(), len(compressed), 1, hashlib.sha256(expected).hexdigest()])
                        output.write(compressed)
                    metadata['experts'].append(node)

        def write_container(description):
            raw = json.dumps(description, separators=(',', ':')).encode()
            compressed = zlib.compress(raw, 9)
            with (folder / 'expert.bin').open('wb') as output, payload_path.open('rb') as payload:
                output.write(struct.pack('<8sIIQQ32s', b'K3ROOT1\0', 1, len(raw), len(compressed),
                                         payload_path.stat().st_size, hashlib.sha256(raw).digest()))
                output.write(compressed)
                shutil.copyfileobj(payload, output)
            return 64 + len(compressed)

        def open_reader():
            with store.RootTableReader(folder):
                pass

        payload_start = write_container(metadata)
        with store.RootTableReader(folder) as reader:
            for expert in (0, 447, 448, 895):
                for matrix, (rows, width) in zip(store.MATRICES, store.SHAPES):
                    for row in (0, 63, 64, rows - 1):
                        assert reader.read_row(expert, matrix, row, False) == expected_rows[matrix]
                        widened = reader.read_row(expert, matrix, row).view('<u4')
                        assert widened.tolist() == [0, 0x80000000, 0x3d800000, 0x3c800000] * (width // 4)
            for expert, matrix, row in ((-1, 'w1', 0), (896, 'w1', 0), (0, 'bad', 0),
                                        (0, 'w1', -1), (0, 'w1', 3072), (0, 'w2', 3584), (0, 'w1', 0.5)):
                rejects('address:' + repr((expert, matrix, row)), lambda: reader.read_row(expert, matrix, row))
            rejects('negative block', lambda: reader.read_block(0, 'w1', -1))
        for name in tables:
            path = folder / (name + '.bin')
            path.write_bytes(b'\x02\x00' + tables[name][2:])
            rejects('wrong table FK:' + name, open_reader)
            path.write_bytes(tables[name])
        path = folder / 'root.bin'
        path.write_bytes(root[:-1] + bytes([root[-1] ^ 1]))
        rejects('shared value checksum', open_reader)
        path.write_bytes(root)
        container = folder / 'expert.bin'
        for name, offset in (('header magic', 0), ('metadata checksum', 32)):
            with container.open('r+b') as output:
                output.seek(offset)
                original = output.read(1)
                output.seek(offset)
                output.write(bytes([original[0] ^ 1]))
            rejects(name, open_reader)
            with container.open('r+b') as output:
                output.seek(offset)
                output.write(original)
        with container.open('r+b') as output:
            offset = payload_start + metadata['experts'][0]['blocks'][0][0]
            output.seek(offset)
            original = output.read(1)
            output.seek(offset)
            output.write(bytes([original[0] ^ 1]))
        with store.RootTableReader(folder) as reader:
            rejects('damaged block', lambda: reader.read_row(0, 'w1', 0))
        with container.open('r+b') as output:
            output.seek(offset)
            output.write(original)
            output.truncate(container.stat().st_size - 1)
        rejects('truncated container', open_reader)
        bad = copy.deepcopy(metadata)
        bad['experts'][0]['root_id'] = 2
        write_container(bad)
        rejects('expert FK with valid metadata checksum', open_reader)
        bad = copy.deepcopy(metadata)
        bad['experts'][0]['blocks'][0][0] += 1
        write_container(bad)
        rejects('block gap with valid metadata checksum', open_reader)
        write_container(metadata)
        open_reader()
    return {'synthetic_row_checks': 48, 'invalid_cases_rejected': rejected}


def standalone(directory):
    expected = json.loads((directory / 'results.json').read_text())['row_read_examples']
    opened = set()
    allowed = {str(directory / (name + '.bin')) for name in ('root', 'root-up', 'root-down', 'expert')}

    def audit(event, arguments):
        if event == 'open':
            path = arguments[0]
            if not isinstance(path, (str, bytes)) or str(path) not in allowed:
                raise AssertionError('Standalone read attempted non-dataset access: ' + str(path))
            opened.add(str(path))

    sys.addaudithook(audit)
    with store.RootTableReader(directory) as reader:
        for example in expected:
            raw = reader.read_row(example['expert'], example['matrix'], example['row'], False)
            assert hashlib.sha256(raw).hexdigest() == example['sha256']
        values = reader.read_row(0, 'w1', 0)
        assert values[0] == np.float32(0.0625) and values.view('<u4')[9] == 0x80000000
    assert opened == allowed
    print(json.dumps({'gate': 'PASS', 'actual_rows': len(expected), 'only_four_dataset_files_opened': True,
                      'opened': sorted(opened), 'signed_zero_preserved': True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('controls', 'standalone'))
    parser.add_argument('directory', type=Path, nargs='?')
    args = parser.parse_args()
    if args.command == 'standalone':
        standalone(args.directory.resolve())
    else:
        print(json.dumps({'gate': 'PASS', **controls(), 'script_sha256': store.digest(Path(__file__))}, indent=2))


if __name__ == '__main__':
    main()