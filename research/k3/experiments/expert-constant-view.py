#!/usr/bin/env python3
"""Export the verified constant-format container's actual index and samples."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import zlib

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('constant_store', HERE / 'expert-constant-store.py')
store = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(store)


def cell(bits, offset):
    value = struct.unpack('<f', struct.pack('<I', bits << 16))[0]
    return {'bits': f'{bits:04x}', 'value': '-0' if bits == 32768 else str(value),
            'bytes': struct.pack('<H', bits).hex(' '), 'offset': offset}


def export(directory, reader, output):
    result = json.loads((directory / 'results.json').read_text())
    store.old.require(result['gate'] == 'PASS', 'native verification pending')
    palette_raw, palette = store.compiled_palette(reader)
    store.old.require(hashlib.sha256(palette_raw).hexdigest() == result['compiled_palette_sha256'], 'compiled constants differ')
    path = directory / 'experts.bin'
    store.old.require(store.old.digest(path) == result['dataset_sha256'], 'dataset identity differs')
    values = [cell(int(bits), index * 2) for index, bits in enumerate(palette)]
    bindings = [{'id': index, 'table': 'constants', 'column': index + 1, **value} for index, value in enumerate(values)]
    records = []
    with path.open('rb') as stream:
        raw_header = stream.read(64)
        fields = store.HEADER.unpack(raw_header)
        index = stream.read(store.INDEX_SIZE)
        store.old.require(raw_header == store.header(index, path.stat().st_size - 64 - len(index), palette_raw), 'container header differs')
        payload_start = 64 + len(index)
        for record in range(store.RECORDS):
            map_offset, map_count, first_block, block_count, reserved = store.RECORD.unpack_from(index, record * 24)
            matrix = store.old.MATRICES[record % 3]
            rows, width = store.old.SHAPES[record % 3]
            blocks = []
            for block in range(block_count):
                offset, size, codec, crc, unused = store.BLOCK.unpack_from(index, (store.RECORDS + first_block + block) * 24)
                blocks.append({'offset': payload_start + offset, 'bytes': size,
                               'codec': store.old.CODECS[codec], 'crc32': f'{crc:08x}'})
            stream.seek(payload_start + map_offset)
            mapping_raw = stream.read(map_count * 16)
            stream.seek(blocks[0]['offset'])
            payload = stream.read(blocks[0]['bytes'])
            count = width * 64
            raw = payload if blocks[0]['codec'] == 'grouped_raw' else store.old.inflate(payload, count // 2 + count // 32)
            codes_raw = np.frombuffer(raw, dtype=np.uint8, count=count // 2)
            codes = np.empty(count, dtype=np.uint8)
            codes[::2], codes[1::2] = codes_raw & 15, codes_raw >> 4
            selectors = np.frombuffer(raw, dtype=np.uint8, offset=count // 2)
            mapping = np.frombuffer(mapping_raw, dtype=np.uint8).reshape(map_count, 16)
            store.old.require(np.all(selectors < map_count), 'invalid selector')
            refs = mapping[selectors[:, None], codes.reshape(-1, 32)].ravel()
            store.old.require(np.all(refs < 66), 'invalid C palette reference')
            decoded = palette[refs].astype('<u2').tobytes()
            store.old.require(f'{zlib.crc32(decoded):08x}' == blocks[0]['crc32'], 'sample block differs')
            selector = int(selectors[0])
            group_map = mapping[selector].tolist()
            records.append({'expert': record // 3, 'matrix': matrix, 'root_id': 1, 'shape': [rows, width],
                            'record_offset': 64 + record * 24,
                            'record_hex': index[record * 24:(record + 1) * 24].hex(' '),
                            'first_block_index': first_block, 'map_count': map_count,
                            'local_offset': None, 'locals': [], 'group_scales': [],
                            'mapping_bytes': len(mapping_raw), 'mapping_offset': payload_start + map_offset,
                            'blocks': blocks, 'bindings': bindings, 'first_payload_hex': payload[:32].hex(' '),
                            'first_values': [cell(int(bits), None) for bits in palette[refs[:32]]],
                            'group': {'selector': selector, 'scale': None, 'codes_hex': raw[:16].hex(' '),
                                      'mapping_offset': payload_start + map_offset + selector * 16,
                                      'mapping_hex': mapping_raw[selector * 16:(selector + 1) * 16].hex(' '),
                                      'mapping': group_map, 'codes': codes[:32].tolist(), 'refs': refs[:32].tolist()}})
    report = {'layer': 1, 'format': 'constant-palette', 'status': 'Verified C constant dataset',
              'container_final': True, 'full_verification': True, 'source_directory': str(directory),
              'expert_file': 'experts.bin', 'payload_start': payload_start,
              'tables': [{'name': 'constants', 'id': None, 'bytes': 132, 'hex': palette_raw.hex(' '),
                          'sha256': result['compiled_palette_sha256'], 'values': values}],
              'records': records, 'matrix_records': len(records), 'experts_complete': 896,
              'sampled_bytes_sha256': result['dataset_sha256'], 'exporter_sha256': store.old.digest(Path(__file__)),
              'dataset_bytes': result['dataset_bytes'], 'scalar_value_bytes_in_dataset': 0,
              'sample_scope': 'Full native verification: all 29,595,009,024 weights. Display samples: first 32 weights of each matrix; first 64-row block checked on export. Constants are compiled software bytes, not dataset cells. Matrix identity and shape derive from record index. Offsets displayed are absolute file bytes; raw index stores payload-relative offsets.'}
    store.old.save_json(output, report)
    print(json.dumps({'gate': 'PASS', 'records': len(records), 'snapshot_sha256': store.old.digest(output)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('reader', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.directory, args.reader, args.output)