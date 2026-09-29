#!/usr/bin/env python3
"""Build and independently verify the layer-1 constant-palette expert store."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import zlib

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('shared_store', HERE / 'root-table-store.py')
old = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(old)
HEADER = struct.Struct('<8s8I3Q')
RECORD = struct.Struct('<Q4I')
BLOCK = struct.Struct('<Q4I')
RECORDS, BLOCKS = 2688, 136192
INDEX_SIZE = (RECORDS + BLOCKS) * 24


def compiled_palette(reader):
    raw = subprocess.check_output([str(reader), 'palette'])
    old.require(len(raw) == 132, 'compiled palette length differs')
    return raw, np.frombuffer(raw, dtype='<u2')


def header(index, payload_size, palette):
    return HEADER.pack(b'K3CONST1', 1, 1, RECORDS, BLOCKS, 24, 24,
                       zlib.crc32(palette), zlib.crc32(index), len(index), payload_size, 0)


def controls(reader, output):
    palette_raw, palette = compiled_palette(reader)
    ids = [int(np.flatnonzero(palette == bits)[0]) for bits in (0, 0x8000, 0x3d80, 0xbc80)]
    rejected, checked = [], 0
    with tempfile.TemporaryDirectory(prefix='constant-controls-') as temporary:
        folder = Path(temporary)
        payload = folder / 'payload'
        records, blocks = bytearray(), bytearray()
        expected = {}
        with payload.open('xb') as stream:
            for record in range(RECORDS):
                width, count = (3072, 56) if record % 3 == 2 else (3584, 48)
                records.extend(RECORD.pack(stream.tell(), 1, len(blocks) // 24, count, 0))
                stream.write(bytes(ids) + bytes([255]) * 12)
                bf16 = struct.pack('<4H', 0, 0x8000, 0x3d80, 0xbc80) * (64 * width // 4)
                raw = b'\x10\x32' * (64 * width // 4) + bytes(64 * width // 32)
                compressed = zlib.compress(raw, 6)
                for block in range(count):
                    codec = 0 if record == 0 and block == 0 else 1
                    data = raw if codec == 0 else compressed
                    blocks.extend(BLOCK.pack(stream.tell(), len(data), codec, zlib.crc32(bf16), 0))
                    stream.write(data)
                expected[record % 3] = struct.pack('<4I', 0, 0x80000000, 0x3d800000, 0xbc800000) * (width // 4)
        index = records + blocks
        dataset = folder / 'experts.bin'
        with dataset.open('xb') as stream, payload.open('rb') as source:
            stream.write(header(index, payload.stat().st_size, palette_raw))
            stream.write(index)
            shutil.copyfileobj(source, stream)
        for expert in (0, 447, 895):
            for matrix_index, matrix in enumerate(old.MATRICES):
                for row in (0, 64, 3583 if matrix == 'w2' else 3071):
                    actual = subprocess.check_output([str(reader), 'row', str(dataset), str(expert), matrix, str(row)])
                    old.require(actual == expected[matrix_index], 'native control row differs')
                    checked += 1

        def rejects(name, args, path=dataset):
            result = subprocess.run([str(reader), 'row', str(path), *args], capture_output=True)
            old.require(result.returncode != 0 and not result.stdout, 'malformed input accepted: ' + name)
            rejected.append(name)

        for args in (['-1', 'w1', '0'], ['896', 'w1', '0'], ['0', 'bad', '0'],
                     ['0', 'w1', '-1'], ['0', 'w1', '3072'], ['0', 'w2', '3584'], ['0x', 'w1', '0']):
            rejects('address:' + '/'.join(args), args)
        offset = HEADER.size + INDEX_SIZE
        cases = [('magic', 0, 0), ('palette identity', 32, 0), ('index checksum', 36, 0),
                 ('index record', HEADER.size, 1), ('group reference', offset, 254),
                 ('group selector', offset + 16 + 64 * 3584 // 2, 1),
                 ('decoded checksum', offset + 16, 0x11)]
        for name, position, value in cases:
            with dataset.open('r+b') as stream:
                stream.seek(position)
                original = stream.read(1)
                stream.seek(position)
                stream.write(bytes([value if value != original[0] else value ^ 1]))
            rejects(name, ['0', 'w1', '0'])
            with dataset.open('r+b') as stream:
                stream.seek(position)
                stream.write(original)
        with dataset.open('r+b') as stream:
            stream.truncate(dataset.stat().st_size - 1)
        rejects('truncated file', ['0', 'w1', '0'])
    report = {'gate': 'PASS', 'native_row_checks': checked, 'rejected': rejected,
              'compiled_palette_sha256': hashlib.sha256(palette_raw).hexdigest(),
              'reader_sha256': old.digest(reader), 'checker_sha256': old.digest(Path(__file__))}
    old.save_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


def build(source, directory, reader):
    report = json.loads((source / 'results.json').read_text())
    old.require(report['gate'] == 'PASS' and report['values_verified'] == 29595009024,
                'prior full source verification not complete')
    palette_raw, palette = compiled_palette(reader)
    census, domains, counts, entries, locations = old.inputs(source)
    expected = sorted(set().union(*(set(entry['value_bits_bf16']) for entry in entries)))
    old.require(palette.tolist() == expected, 'C palette differs from complete measured domain')
    for name, checksum in report['file_sha256'].items():
        old.require(old.digest(source / name) == checksum, 'prior dataset hash changed')
    directory.mkdir()
    records, blocks, mappings = bytearray(), bytearray(), bytearray()
    payload_hash, whole = hashlib.sha256(), hashlib.sha256()
    lookup = {int(bits): index for index, bits in enumerate(palette)}
    path = directory / 'experts.bin'
    codec_counts = {0: 0, 1: 0}
    copied_bytes, map_bytes = 0, 0
    with old.RootTableReader(source) as original, path.open('xb+') as output:
        output.seek(HEADER.size + INDEX_SIZE)
        for expert in range(896):
            for matrix in old.MATRICES:
                node, local_palette, mapping = original.bindings(expert, matrix)
                refs = np.array([lookup[int(bits)] for bits in local_palette], dtype=np.uint8)
                remapped = np.full(mapping.shape, 255, dtype=np.uint8)
                present = mapping != 255
                remapped[present] = refs[mapping[present]]
                raw_map = remapped.tobytes()
                records.extend(RECORD.pack(output.tell() - HEADER.size - INDEX_SIZE, len(mapping),
                                           len(blocks) // BLOCK.size, len(node['blocks']), 0))
                output.write(raw_map)
                mappings.extend(raw_map)
                map_bytes += len(raw_map)
                for block, (offset, size, codec, checksum) in enumerate(node['blocks']):
                    old.require(codec in (0, 1), 'source uses a direct layout; requires a separate conversion')
                    original.file.seek(original.payload_start + offset)
                    payload = original.file.read(size)
                    width = old.SHAPES[old.MATRICES.index(matrix)][1]
                    decoded = old.decode_payload(payload, codec, width * 64, palette, len(mapping), remapped)
                    old.require(hashlib.sha256(decoded).hexdigest() == checksum, 'constant-reference decode differs')
                    blocks.extend(BLOCK.pack(output.tell() - HEADER.size - INDEX_SIZE, size, codec,
                                             zlib.crc32(decoded), 0))
                    output.write(payload)
                    whole.update(decoded)
                    payload_hash.update(payload)
                    copied_bytes += len(payload)
                    codec_counts[codec] += 1
            if (expert + 1) % 128 == 0:
                print(json.dumps({'experts_converted': expert + 1}), flush=True)
        payload_size = output.tell() - HEADER.size - INDEX_SIZE
        index = records + blocks
        old.require(len(index) == INDEX_SIZE and whole.hexdigest() == report['ordered_bf16_weights_sha256'], 'complete conversion differs')
        output.seek(0)
        output.write(header(index, payload_size, palette_raw))
        output.write(index)
    total = path.stat().st_size
    built = {'state': 'BUILT_NATIVE_VERIFICATION_PENDING', 'experts': 896, 'matrix_records': RECORDS,
             'blocks': BLOCKS, 'header_bytes': HEADER.size, 'index_bytes': len(index),
             'group_mapping_bytes': map_bytes, 'compressed_block_bytes': copied_bytes,
             'scalar_value_bytes_in_dataset': 0, 'compiled_palette_bytes': len(palette_raw),
             'dataset_bytes': total, 'dataset_plus_compiled_palette_bytes': total + len(palette_raw),
             'previous_dataset_bytes': report['total_storage_bytes'],
             'saved_dataset_bytes_vs_previous': report['total_storage_bytes'] - total,
             'saved_bytes_including_palette_vs_previous': report['total_storage_bytes'] - total - len(palette_raw),
             'original_quantized_bytes': report['original_bytes'],
             'saved_bytes_vs_original_including_palette': report['original_bytes'] - total - len(palette_raw),
             'removed_local_scalar_bytes': report['local_value_bytes'],
             'removed_shared_table_bytes': report['shared_tables_bytes'],
             'selected_codecs': codec_counts, 'dataset_sha256': old.digest(path),
             'ordered_bf16_weights_sha256': whole.hexdigest(), 'compressed_blocks_sha256': payload_hash.hexdigest(),
             'remapped_group_bytes_sha256': hashlib.sha256(mappings).hexdigest(),
             'compiled_palette_sha256': hashlib.sha256(palette_raw).hexdigest(),
             'source_report_sha256': old.digest(source / 'results.json'),
             'implementation_sha256': {name: old.digest(HERE / name) for name in ('expert-constant-store.py', 'expert-constant-store.c')},
             'palette_header_sha256': old.digest(HERE.parent / 'expert-constant-palette.h'),
             'reader_sha256': old.digest(reader),
             'limits': 'Layer1 only, no model integration. Palette bytes counted once in software; executable code and audit files excluded. CRC32 detects accidental corruption; SHA256 identity is in external verification reports. Numeric fields and matrix identities are implicit in fixed format.'}
    old.save_json(directory / 'build-results.json', built)
    print(json.dumps(built, indent=2), flush=True)


def verify(source, directory, reader):
    built = json.loads((directory / 'build-results.json').read_text())
    path = directory / 'experts.bin'
    old.require(old.digest(path) == built['dataset_sha256'], 'dataset hash differs')
    whole, count = hashlib.sha256(), 0
    with (directory / 'native-verify.stderr').open('xb') as errors:
        process = subprocess.Popen([str(reader), 'verify', str(path)], stdout=subprocess.PIPE, stderr=errors)
        for chunk in iter(lambda: process.stdout.read(8 * 1024 * 1024), b''):
            whole.update(chunk)
            count += len(chunk)
        process.stdout.close()
        status = process.wait()
    old.require(status == 0 and count == 29595009024 * 2 and whole.hexdigest() == built['ordered_bf16_weights_sha256'],
                'complete native output differs')
    examples = []
    with old.RootTableReader(source) as original:
        for expert in (0, 24, 447, 448, 895):
            for matrix in old.MATRICES:
                rows, width = old.SHAPES[old.MATRICES.index(matrix)]
                for row in (0, 63, 64, rows - 1):
                    raw = subprocess.check_output([str(reader), 'row', str(path), str(expert), matrix, str(row)])
                    expected = original.read_row(expert, matrix, row).astype('<f4', copy=False).tobytes()
                    old.require(raw == expected, 'native float32 row differs')
                    examples.append({'expert': expert, 'matrix': matrix, 'row': row, 'sha256': hashlib.sha256(raw).hexdigest()})
    old.require(old.digest(reader) == built['reader_sha256'], 'native reader changed')
    old.require(old.digest(source / 'results.json') == built['source_report_sha256'], 'source evidence changed')
    report = {'gate': 'PASS', **{key: value for key, value in built.items() if key != 'state'},
              'all_native_bf16_bits_equal_verified_source': True, 'values_verified': count // 2,
              'native_float32_rows_checked': len(examples), 'row_examples': examples,
              'native_runtime_data_files': ['experts.bin'], 'root_or_local_scalar_files_required': False}
    old.save_json(directory / 'results.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'row_examples'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('controls', 'build', 'verify'))
    parser.add_argument('--reader', required=True, type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'controls':
        controls(args.reader.resolve(), args.output)
    else:
        globals()[args.command](args.source, args.output, args.reader.resolve())


if __name__ == '__main__':
    main()