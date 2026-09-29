#!/usr/bin/env python3
"""Export actual map-equation records and the compiled lookup arrays."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import zlib


def require(condition, message):
    if not condition:
        raise ValueError(message)


def export(base, previous, output):
    result = json.loads((base / 'results.json').read_text())
    manifest = json.loads((base / 'maps.json').read_text())
    require(result['gate'] == 'PASS', 'native full verification missing')
    require(hashlib.sha256((base / 'maps.json').read_bytes()).hexdigest() == result['maps_manifest_sha256'], 'manifest differs')
    constants = subprocess.check_output([str(base / 'reader'), 'constants'])
    require(hashlib.sha256(constants).hexdigest() == result['compiled_constant_sha256'], 'compiled bytes differ')
    with (base / 'experts.bin').open('rb') as stream:
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == result['dataset_sha256'], 'dataset differs')
    data = json.loads(previous.read_text())
    require(data['sampled_bytes_sha256'] == result['source_dataset_sha256'], 'prior display source differs')
    scalar = struct.unpack_from('<66H', constants)
    maps = [list(constants[132 + index * 16:132 + (index + 1) * 16]) for index in range(manifest['unique_maps'])]
    offset_start = 132 + manifest['compiled_map_bytes']
    offsets = struct.unpack_from('<516H', constants, offset_start)
    references = struct.unpack_from('<3638H', constants, offset_start + manifest['compiled_offset_bytes'])
    with (base / 'experts.bin').open('rb') as stream:
        magic, constants_crc, index_crc, index_size, payload_size = struct.unpack('<8sIIQQ', stream.read(32))
        index = stream.read(index_size)
        require(magic == b'K3MAPS01' and constants_crc == zlib.crc32(constants) and index_crc == zlib.crc32(index), 'header identity differs')
        position = 32 + index_size
        for record_index, record in enumerate(data['records']):
            require(record_index == record['expert'] * 3 + ('w1', 'w3', 'w2').index(record['matrix']), 'record order differs')
            template = struct.unpack_from('<H', index, record_index * 2)[0]
            require(template == manifest['matrix_template_ids'][record_index], 'template assignment differs')
            ids = list(references[offsets[template]:offsets[template + 1]])
            record['template_id'] = template
            record['map_ids'] = ids
            record['record_offset'] = 32 + record_index * 2
            record['record_hex'] = index[record_index * 2:(record_index + 1) * 2].hex(' ')
            record['mapping_offset'] = None
            record['mapping_bytes'] = 0
            record['map_count'] = len(ids)
            for block_number, block in enumerate(record['blocks']):
                block_id = record['first_block_index'] + block_number
                size, checksum = struct.unpack_from('<II', index, 5376 + block_id * 8)
                require(size == block['bytes'] and f'{checksum:08x}' == block['crc32'], 'block index differs')
                block['offset'] = position
                if block_number == 0:
                    stream.seek(position)
                    payload = stream.read(size)
                    require(payload[:32].hex(' ') == record['first_payload_hex'], 'compressed sample differs')
                    count = record['shape'][1] * 64
                    decoder = zlib.decompressobj()
                    raw = decoder.decompress(payload, count // 2 + count // 32 + 1)
                    require(len(raw) == count // 2 + count // 32 and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
                            'sample framing differs')
                    group = record['group']
                    selector = raw[count // 2]
                    map_id = ids[selector]
                    require(selector == group['selector'] and maps[map_id] == group['mapping'], 'equation map differs')
                    codes = [(raw[coordinate // 2] >> (coordinate % 2 * 4)) & 15 for coordinate in range(32)]
                    require(codes == group['codes'], 'sample code order differs')
                    for coordinate, code in enumerate(codes):
                        reference = maps[map_id][code]
                        require(f'{scalar[reference]:04x}' == record['first_values'][coordinate]['bits'], 'equation weight differs')
                    group['compiled_map'] = True
                    group['map_id'] = map_id
                    group['mapping_offset'] = map_id * 16
                position += size
        require(position == 32 + index_size + payload_size == (base / 'experts.bin').stat().st_size, 'complete block ranges differ')
    data.update({'format': 'map-equation', 'status': 'Verified map-constant equation', 'source_directory': str(base),
                 'payload_start': 32 + index_size, 'dataset_bytes': result['dataset_bytes'],
                 'sampled_bytes_sha256': result['dataset_sha256'],
                 'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'compiled_bytes': result['compiled_constant_bytes'],
                 'compiled_maps': maps, 'template_offsets': offsets, 'template_map_ids': references,
                 'sample_scope': 'All weights independently verified by native reader. Display: actual 2-byte matrix map-list IDs, derived block offsets, unchanged code/selector samples, compiled map/scalar arrays. Only compressed arrangements, matrix map-list IDs, block lengths/checksums and a header remain in the file. No full model inference performed.'})
    with output.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
    example = data['records'][2]
    print(json.dumps({'gate': 'PASS', 'records': len(data['records']), 'expert0_w2_template': example['template_id'],
                      'expert0_w2_map_ids': example['map_ids'], 'first_group_map': example['group']['map_id'],
                      'snapshot_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('previous_snapshot', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.directory, args.previous_snapshot, args.output)