#!/usr/bin/env python3
"""Export actual stored record bytes for the local visual inspector."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('root_store', HERE / 'root-table-store.py')
store = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(store)


def cell(bits, offset):
    value = struct.unpack('<f', struct.pack('<I', bits << 16))[0]
    return {'bits': f'{bits:04x}', 'value': '-0' if bits == 32768 else str(value),
            'bytes': struct.pack('<H', bits).hex(' '), 'offset': offset}


def export(directory, output):
    census, domains, counts, records, locations = store.inputs(directory)
    final = (directory / 'expert.bin').exists()
    verified = (directory / 'results.json').exists()
    tables = []
    for name in ('root', 'root-up', 'root-down'):
        raw = (directory / (name + '.bin')).read_bytes()
        tables.append({'name': name, 'id': struct.unpack_from('<H', raw)[0], 'bytes': len(raw),
                       'hex': raw.hex(' '), 'sha256': hashlib.sha256(raw).hexdigest(),
                       'values': [cell(int(value), 2 + index * 2) for index, value in
                                  enumerate(np.frombuffer(raw, dtype='<u2', offset=2))]})
    nodes = []
    if final:
        with store.RootTableReader(directory) as reader:
            nodes = reader.metadata['experts']
            payload_start = reader.payload_start
        filename = 'expert.bin'
    else:
        filename, payload_start, cursor = 'expert.payload.pending', 0, 0
        with (directory / 'expert-blocks.jsonl').open() as audit:
            for record in records:
                expert, matrix = record['expert'], record['matrix']
                matrix_index = store.MATRICES.index(matrix)
                palette, scales, mapping = store.palette_for(domains, record, counts[expert, matrix_index])
                rows, width = store.SHAPES[matrix_index]
                node = {'root_id': 1, 'expert_id': expert, 'matrix': matrix,
                        'local_count': len(record['local_value_bits_bf16']), 'local_offset': cursor,
                        'group_scales': scales.tolist(), 'blocks': []}
                cursor += node['local_count'] * 2
                node['group_palette_offset'] = cursor
                cursor += len(scales) * 16
                for first_row in range(0, rows, 64):
                    line = audit.readline()
                    if not line.endswith('\n'):
                        break
                    entry = json.loads(line)
                    store.require((entry['expert'], entry['matrix'], entry['first_row']) ==
                                  (expert, matrix, first_row), 'audit order differs')
                    node['blocks'].append([cursor, entry['stored_bytes'], store.CODECS.index(entry['selected'])])
                    cursor += entry['stored_bytes']
                if len(node['blocks']) != rows // 64:
                    break
                nodes.append(node)
    rows_out = []
    prefix_hash = hashlib.sha256()
    with (directory / filename).open('rb') as stream:
        for node in nodes:
            expert, matrix = node['expert_id'], node['matrix']
            matrix_index = store.MATRICES.index(matrix)
            rows, width = store.SHAPES[matrix_index]
            source_record = records[expert * 3 + matrix_index]
            expected_palette, scales, expected_mapping = store.palette_for(domains, source_record, counts[expert, matrix_index])
            stream.seek(payload_start + node['local_offset'])
            local = stream.read(node['local_count'] * 2)
            store.require(local == np.array(source_record['local_value_bits_bf16'], dtype='<u2').tobytes(), 'stored locals differ')
            stream.seek(payload_start + node['group_palette_offset'])
            mapping_raw = stream.read(len(node['group_scales']) * 16)
            store.require(mapping_raw == expected_mapping.tobytes(), 'stored group map differs')
            offset, size, codec = node['blocks'][0][:3]
            stream.seek(payload_start + offset)
            payload = stream.read(size)
            store.require(len(payload) == size, 'incomplete payload')
            restored = store.decode_payload(payload, codec, width * 64, expected_palette, len(scales), expected_mapping)
            if final:
                store.require(hashlib.sha256(restored).hexdigest() == node['blocks'][0][3], 'block checksum differs')
            branch = 'root-down' if matrix == 'w2' else 'root-up'
            bindings = []
            for name, values in (('root', domains['root']), (branch, domains[branch]),
                                 ('local', source_record['local_value_bits_bf16'])):
                for index, bits in enumerate(values):
                    bindings.append({'id': len(bindings), 'table': name, 'column': index + 1, **cell(bits, None)})
            group = None
            if codec in (0, 1):
                raw = payload if codec == 0 else store.inflate(payload, width * 64 // 2 + width * 64 // 32)
                selector = raw[width * 64 // 2]
                codes = [nibble for byte in raw[:16] for nibble in (byte & 15, byte >> 4)]
                refs = expected_mapping[selector].tolist()
                first = np.frombuffer(restored, dtype='<u2', count=32)
                independent = store.census_code.decode_table()[scales[selector], codes].view('<u4')
                store.require(np.array_equal(first.astype('<u4') << 16, independent), 'group independent decode differs')
                group = {'selector': selector, 'scale': int(scales[selector]), 'codes_hex': raw[:16].hex(' '),
                         'mapping_offset': payload_start + node['group_palette_offset'] + selector * 16,
                         'mapping_hex': mapping_raw[selector * 16:(selector + 1) * 16].hex(' '),
                         'mapping': refs, 'codes': codes, 'refs': [refs[code] for code in codes]}
            prefix_hash.update(local + mapping_raw + payload)
            rows_out.append({'expert': expert, 'matrix': matrix, 'root_id': node['root_id'], 'shape': [rows, width],
                             'local_offset': payload_start + node['local_offset'],
                             'locals': [cell(int(bits), payload_start + node['local_offset'] + index * 2) for index, bits in
                                        enumerate(np.frombuffer(local, dtype='<u2'))],
                             'group_scales': node['group_scales'], 'mapping_bytes': len(mapping_raw),
                             'mapping_offset': payload_start + node['group_palette_offset'],
                             'blocks': [{'offset': payload_start + entry[0], 'bytes': entry[1], 'codec': store.CODECS[entry[2]],
                                         'sha256': entry[3] if len(entry) == 4 else None} for entry in node['blocks']],
                             'first_payload_hex': payload[:32].hex(' '), 'bindings': bindings, 'group': group,
                             'first_values': [cell(int(bits), None) for bits in np.frombuffer(restored, dtype='<u2', count=32)]})
    report = {'layer': 1, 'status': 'Verified dataset' if verified else 'Built dataset / verification pending' if final else 'Partial build snapshot',
              'container_final': final, 'full_verification': verified, 'source_directory': str(directory),
              'expert_file': filename, 'payload_start': payload_start, 'tables': tables, 'records': rows_out,
              'matrix_records': len(rows_out), 'experts_complete': sum(all(any(node['expert_id'] == expert and node['matrix'] == matrix for node in nodes)
                                                                       for matrix in store.MATRICES) for expert in range(896)),
              'sampled_bytes_sha256': prefix_hash.hexdigest(), 'exporter_sha256': store.digest(Path(__file__)),
              'sample_scope': 'Stored local values and group maps plus first 64-row block per available matrix; first 32 weights displayed. Partial record boundaries reconstructed from build audit and census, not final metadata.' if not final else
                              'Records from stored container metadata; local values and maps read from payload; first block decoded and checksum checked for each matrix.'}
    store.save_json(output, report)
    print(json.dumps({key: report[key] for key in ('status', 'matrix_records', 'experts_complete', 'sampled_bytes_sha256')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.directory, args.output)