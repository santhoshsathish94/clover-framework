#!/usr/bin/env python3
"""Factor measured map repetitions into C constants; retain exact arrangements."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import zlib


RECORDS, BLOCKS = 2688, 136192
OLD_HEADER = struct.Struct('<8s8I3Q')
OLD_ENTRY = struct.Struct('<Q4I')
HEADER = struct.Struct('<8sIIQQ')
ENTRY = struct.Struct('<II')
INDEX_BYTES = RECORDS * 2 + BLOCKS * ENTRY.size
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def prior(source):
    result = json.loads((source / 'results.json').read_text())
    require(result['gate'] == 'PASS' and result['values_verified'] == 29595009024, 'prior full verification missing')
    path = source / 'experts.bin'
    require(digest(path) == result['dataset_sha256'], 'source dataset differs')
    with path.open('rb') as stream:
        fields = OLD_HEADER.unpack(stream.read(64))
        require(fields[:7] == (b'K3CONST1', 1, 1, RECORDS, BLOCKS, 24, 24), 'source format differs')
        index = stream.read(fields[9])
        require(zlib.crc32(index) == fields[8], 'source index differs')
    return result, index, 64 + len(index)


def prepare(source, target, reader):
    result, index, payload_start = prior(source)
    target.mkdir()
    rows, lists = [], []
    with (source / 'experts.bin').open('rb') as stream:
        for record in range(RECORDS):
            offset, count, first, blocks, reserved = OLD_ENTRY.unpack_from(index, record * 24)
            require(1 <= count <= 14 and reserved == 0, 'matrix record differs')
            stream.seek(payload_start + offset)
            mapping = stream.read(count * 16)
            require(len(mapping) == count * 16 and all(ref < 66 or ref == 255 for ref in mapping), 'invalid maps')
            group_rows = tuple(mapping[start:start + 16] for start in range(0, len(mapping), 16))
            rows.extend(group_rows)
            lists.append(group_rows)
    unique_rows = sorted(set(rows))
    row_ids = {row: index for index, row in enumerate(unique_rows)}
    mapped_lists = [tuple(row_ids[row] for row in group_rows) for group_rows in lists]
    templates = sorted(set(mapped_lists))
    template_ids = {template: index for index, template in enumerate(templates)}
    assignments = [template_ids[template] for template in mapped_lists]
    flat, offsets = [], [0]
    for template in templates:
        flat.extend(template)
        offsets.append(len(flat))
    require(len(unique_rows) == 465 and len(templates) == 515 and len(flat) < 65536, 'measured map domains differ')
    for original, assignment in zip(lists, assignments):
        recovered = tuple(unique_rows[ref] for ref in flat[offsets[assignment]:offsets[assignment + 1]])
        require(recovered == original, 'factored map reconstruction differs')
    palette = subprocess.check_output([str(reader), 'palette'])
    require(hashlib.sha256(palette).hexdigest() == result['compiled_palette_sha256'], 'compiled scalar constants differ')
    mapping_bytes = b''.join(unique_rows)
    offset_bytes = struct.pack('<' + str(len(offsets)) + 'H', *offsets)
    list_bytes = struct.pack('<' + str(len(flat)) + 'H', *flat)
    constants = palette + mapping_bytes + offset_bytes + list_bytes

    def array(name, values):
        lines = [f'static const uint16_t {name}[{len(values)}] = {{']
        lines.extend('    ' + ', '.join(str(value) for value in values[start:start + 16]) + ',' for start in range(0, len(values), 16))
        return '\n'.join(lines + ['};'])

    lines = ['#ifndef K3_EXPERT_MAP_CONSTANTS_H', '#define K3_EXPERT_MAP_CONSTANTS_H',
             '#include <stdint.h>', f'#define K3_MAP_COUNT {len(unique_rows)}',
             f'#define K3_TEMPLATE_COUNT {len(templates)}', f'#define K3_TEMPLATE_REFS {len(flat)}',
             f'#define K3_CONSTANTS_CRC UINT32_C({zlib.crc32(constants)})',
             'static const uint8_t k3_group_maps[K3_MAP_COUNT][16] = {']
    lines.extend('    {' + ', '.join(str(value) for value in row) + '},' for row in unique_rows)
    lines.extend(['};', array('k3_template_offsets', offsets), array('k3_template_maps', flat), '#endif', ''])
    with (target / 'expert-map-constants.h').open('x') as stream:
        stream.write('\n'.join(lines))
    with (target / 'constants.bin').open('xb') as stream:
        stream.write(constants)
    manifest = {'gate': 'PASS', 'unique_maps': len(unique_rows), 'unique_templates': len(templates),
                'template_refs': len(flat), 'matrix_template_ids': assignments,
                'compiled_scalar_bytes': len(palette), 'compiled_map_bytes': len(mapping_bytes),
                'compiled_offset_bytes': len(offset_bytes), 'compiled_template_bytes': len(list_bytes),
                'compiled_constant_bytes': len(constants), 'constants_crc32': zlib.crc32(constants),
                'constants_sha256': hashlib.sha256(constants).hexdigest(),
                'source_dataset_sha256': result['dataset_sha256'],
                'source_report_sha256': digest(source / 'results.json'),
                'generated_header_sha256': digest(target / 'expert-map-constants.h')}
    save(target / 'maps.json', manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != 'matrix_template_ids'}, indent=2))


def build(source, target):
    original, index, payload_start = prior(source)
    maps = json.loads((target / 'maps.json').read_text())
    require(original['dataset_sha256'] == maps['source_dataset_sha256'], 'map source changed')
    assignments = struct.pack('<2688H', *maps['matrix_template_ids'])
    block_index = bytearray()
    payload_hash = hashlib.sha256()
    filename = target / 'experts.bin'
    copied = 0
    with filename.open('xb+') as output, (source / 'experts.bin').open('rb') as stream:
        output.seek(HEADER.size + INDEX_BYTES)
        for block in range(BLOCKS):
            offset, size, codec, crc, reserved = OLD_ENTRY.unpack_from(index, (RECORDS + block) * 24)
            require(codec == 1 and reserved == 0, 'source codec is not uniform zlib')
            block_index.extend(ENTRY.pack(size, crc))
            stream.seek(payload_start + offset)
            payload = stream.read(size)
            require(len(payload) == size, 'source block truncated')
            output.write(payload)
            copied += size
            payload_hash.update(payload)
        require(payload_hash.hexdigest() == original['compressed_blocks_sha256'], 'copied arrangements differ')
        new_index = assignments + block_index
        output.seek(0)
        output.write(HEADER.pack(b'K3MAPS01', maps['constants_crc32'], zlib.crc32(new_index), len(new_index), copied))
        output.write(new_index)
    total = filename.stat().st_size + maps['compiled_constant_bytes']
    result = {'state': 'BUILT_NATIVE_VERIFICATION_PENDING', 'dataset_bytes': filename.stat().st_size,
              'header_bytes': HEADER.size, 'matrix_assignment_bytes': len(assignments),
              'block_size_and_crc_bytes': len(block_index), 'compressed_arrangement_bytes': copied,
              'compiled_constant_bytes': maps['compiled_constant_bytes'],
              'dataset_plus_constants_bytes': total,
              'previous_dataset_plus_constants_bytes': original['dataset_plus_compiled_palette_bytes'],
              'net_saved_bytes_vs_previous': original['dataset_plus_compiled_palette_bytes'] - total,
              'compiled_byte_breakdown': {key: value for key, value in maps.items() if key.startswith('compiled_')},
              'dataset_sha256': digest(filename), 'compressed_blocks_sha256': payload_hash.hexdigest(),
              'ordered_bf16_weights_sha256': original['ordered_bf16_weights_sha256'],
              'maps_manifest_sha256': digest(target / 'maps.json'),
              'source_dataset_sha256': original['dataset_sha256'],
              'source_report_sha256': digest(source / 'results.json'),
              'limits': 'Layer1 routed expert weights only; all distinct arrangements retained. No full expert forward/model inference, speed or peak-RAM claim. CRC32 index/decoded blocks plus external SHA256 identity retained. Program code, external evidence and OS file overhead excluded; compiled array bytes counted.'}
    save(target / 'build-results.json', result)
    print(json.dumps(result, indent=2), flush=True)


def verify(source, target, reader):
    result = json.loads((target / 'build-results.json').read_text())
    require(digest(target / 'experts.bin') == result['dataset_sha256'], 'built file differs')
    require(subprocess.check_output([str(reader), 'constants']) == (target / 'constants.bin').read_bytes(),
            'compiled map/offset/template constants differ')
    whole, count = hashlib.sha256(), 0
    with (target / 'verify.stderr').open('xb') as errors:
        process = subprocess.Popen([str(reader), 'verify', str(target / 'experts.bin')], stdout=subprocess.PIPE, stderr=errors)
        for chunk in iter(lambda: process.stdout.read(8 * 1024 * 1024), b''):
            whole.update(chunk)
            count += len(chunk)
        process.stdout.close()
        status = process.wait()
    require(status == 0 and count == 59190018048 and whole.hexdigest() == result['ordered_bf16_weights_sha256'],
            'full native equation reconstruction differs')
    previous_reader = source.parent / 'reader'
    previous = json.loads((source / 'results.json').read_text())
    require(digest(previous_reader) == previous['reader_sha256'], 'comparison reader differs')
    examples = []
    for expert in (0, 24, 447, 448, 895):
        for matrix in ('w1', 'w3', 'w2'):
            for row in (0, 63, 64, 3583 if matrix == 'w2' else 3071):
                args = [str(expert), matrix, str(row)]
                expected = subprocess.check_output([str(previous_reader), 'row', str(source / 'experts.bin'), *args])
                actual = subprocess.check_output([str(reader), 'row', str(target / 'experts.bin'), *args])
                require(actual == expected, 'native float32 equation row differs')
                examples.append({'expert': expert, 'matrix': matrix, 'row': row, 'sha256': hashlib.sha256(actual).hexdigest()})
    require(digest(source / 'experts.bin') == result['source_dataset_sha256'], 'prior dataset changed')
    checked = {'gate': 'PASS', **{key: value for key, value in result.items() if key != 'state'},
               'values_verified': count // 2, 'native_float32_rows_checked': len(examples),
               'row_examples': examples, 'prior_dataset_unchanged': True,
               'compiled_constant_sha256': hashlib.sha256((target / 'constants.bin').read_bytes()).hexdigest(),
               'reader_sha256': digest(reader),
               'implementation_sha256': {name: digest(HERE / name) for name in ('expert-map-store.py', 'expert-map-store.c')},
               'generated_header_sha256': digest(target / 'expert-map-constants.h')}
    save(target / 'results.json', checked)
    print(json.dumps({key: value for key, value in checked.items() if key != 'row_examples'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'build', 'verify'))
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--reader', type=Path)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.source, args.target)
    else:
        globals()[args.command](args.source, args.target, args.reader)