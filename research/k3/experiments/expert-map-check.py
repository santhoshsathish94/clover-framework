#!/usr/bin/env python3
"""Isolated native map-store controls, without modifying measured datasets."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib


HEADER = struct.Struct('<8sIIQQ')
INDEX_BYTES = 2688 * 2 + 136192 * 8


def check(base, output):
    reader = base / 'reader'
    constants = subprocess.check_output([str(reader), 'constants'])
    assert constants == (base / 'constants.bin').read_bytes()
    manifest = json.loads((base / 'maps.json').read_text())
    templates = manifest['matrix_template_ids']
    map_start = 132
    offset_start = map_start + manifest['compiled_map_bytes']
    offsets = struct.unpack_from('<' + str(manifest['unique_templates'] + 1) + 'H', constants, offset_start)
    references = struct.unpack_from('<' + str(manifest['template_refs']) + 'H', constants,
                                    offset_start + manifest['compiled_offset_bytes'])
    selectors = []
    for template in templates:
        members = references[offsets[template]:offsets[template + 1]]
        selectors.append(next(selector for selector, member in enumerate(members)
                              if constants[map_start + member * 16] == 0
                              and constants[map_start + member * 16 + 8] == 33))
    rejected, row_checks = [], 0
    with tempfile.TemporaryDirectory(prefix='map-store-controls-') as temporary:
        path = Path(temporary) / 'experts.bin'
        index = bytearray(struct.pack('<2688H', *templates))
        compressed = {}
        crcs = {}
        for width in (3072, 3584):
            for selector in set(selectors):
                raw = b'\x80' * (width * 64 // 2) + bytes([selector]) * (width * 64 // 32)
                compressed[width, selector] = zlib.compress(raw, 6)
            crcs[width] = zlib.crc32(b'\x00\x00\x00\x80' * (width * 64 // 2))
        with path.open('xb+') as stream:
            stream.seek(32 + INDEX_BYTES)
            for block in range(136192):
                width = 3584 if block % 152 < 96 else 3072
                record = block // 152 * 3 + min(block % 152 // 48, 2)
                payload = compressed[width, selectors[record]]
                index.extend(struct.pack('<II', len(payload), crcs[width]))
                stream.write(payload)
            payload_size = stream.tell() - 32 - INDEX_BYTES
            header = HEADER.pack(b'K3MAPS01', zlib.crc32(constants), zlib.crc32(index), INDEX_BYTES, payload_size)
            stream.seek(0)
            stream.write(header)
            stream.write(index)
        for expert in (0, 447, 895):
            for matrix in ('w1', 'w3', 'w2'):
                width = 3072 if matrix == 'w2' else 3584
                expected = struct.pack('<II', 0, 0x80000000) * (width // 2)
                for row in (0, 64, 3583 if matrix == 'w2' else 3071):
                    actual = subprocess.check_output([str(reader), 'row', str(path), str(expert), matrix, str(row)])
                    assert actual == expected
                    row_checks += 1

        def reject(name, args=('0', 'w1', '0')):
            result = subprocess.run([str(reader), 'row', str(path), *args], capture_output=True)
            assert result.returncode != 0 and not result.stdout, name
            rejected.append({'name': name, 'error': result.stderr.decode().strip()})

        for args in (('-1', 'w1', '0'), ('896', 'w1', '0'), ('0', 'bad', '0'),
                     ('0', 'w1', '-1'), ('0', 'w1', '3072'), ('0', 'w2', '3584')):
            reject('/'.join(args), args)
        for name, offset in (('magic', 0), ('constants identity', 8), ('index checksum', 12),
                             ('compressed frame', 32 + INDEX_BYTES)):
            with path.open('r+b') as stream:
                stream.seek(offset)
                original = stream.read(1)
                stream.seek(offset)
                stream.write(bytes([original[0] ^ 255]))
            reject(name)
            with path.open('r+b') as stream:
                stream.seek(offset)
                stream.write(original)
        for name, offset, replacement in (('template range with valid CRC', 0, struct.pack('<H', 515)),
                                          ('zero block length with valid CRC', 5376, bytes(4)),
                                          ('decoded checksum with valid CRC', 5380, bytes(4))):
            changed = bytearray(index)
            changed[offset:offset + len(replacement)] = replacement
            with path.open('r+b') as stream:
                stream.write(HEADER.pack(b'K3MAPS01', zlib.crc32(constants), zlib.crc32(changed), INDEX_BYTES, payload_size))
                stream.write(changed)
            reject(name)
            with path.open('r+b') as stream:
                stream.write(header)
                stream.write(index)
        with path.open('r+b') as stream:
            stream.truncate(path.stat().st_size - 1)
        reject('truncated file')
    report = {'gate': 'PASS', 'exact_native_row_checks': row_checks, 'invalid_cases': rejected,
              'reader_sha256': hashlib.sha256(reader.read_bytes()).hexdigest(),
              'compiled_constants_sha256': hashlib.sha256(constants).hexdigest(),
              'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    check(args.directory, args.output)