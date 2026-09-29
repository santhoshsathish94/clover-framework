#!/usr/bin/env python3
"""Read only layer1 routed experts 0..895 and map their stored structure."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np


SOURCE = Path('/root/k3model/model-00002-of-000096.safetensors')
PREFIX = 'language_model.model.layers.1.block_sparse_moe.experts.'
MATRICES = {'w1': ('gate', 3072, 3584), 'w3': ('up', 3072, 3584), 'w2': ('down', 3584, 3072)}
KINDS = ('weight_packed', 'weight_scale')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def code_counts(payload):
    packed = np.frombuffer(payload, dtype=np.uint8)
    return np.bincount(packed & 15, minlength=16) + np.bincount(packed >> 4, minlength=16)


def histogram_dict(counts):
    return {str(index): int(count) for index, count in enumerate(counts) if count}


def select_records(header, payload_base, file_bytes):
    records = {}
    for name, record in header.items():
        if not name.startswith(PREFIX):
            continue
        parts = name[len(PREFIX):].split('.')
        require(len(parts) == 3, 'unexpected routed-expert field: ' + name)
        expert_text, matrix, kind = parts
        require(expert_text.isdecimal() and matrix in MATRICES and kind in KINDS, 'invalid routed-expert name')
        expert = int(expert_text)
        require(0 <= expert < 896, 'expert ID outside layer1 pool')
        role, rows, width = MATRICES[matrix]
        shape = [rows, width // (2 if kind == 'weight_packed' else 32)]
        require(record['dtype'] == 'U8' and record['shape'] == shape, 'stored tensor shape/dtype differs')
        start, stop = record['data_offsets']
        require(0 <= start < stop and stop - start == math.prod(shape), 'invalid tensor range')
        require(payload_base + stop <= file_bytes, 'tensor outside source file')
        key = (expert, matrix, kind)
        require(key not in records, 'duplicate expert tensor')
        records[key] = {'name': name, 'absolute_offset': payload_base + start, 'bytes': stop - start,
                        'dtype': record['dtype'], 'shape': shape}
    expected = {(expert, matrix, kind) for expert in range(896) for matrix in MATRICES for kind in KINDS}
    require(set(records) == expected, 'missing or unexpected expert tensor records')
    previous_end = 0
    for record in sorted(records.values(), key=lambda item: item['absolute_offset']):
        require(record['absolute_offset'] >= previous_end, 'overlapping expert tensor ranges')
        previous_end = record['absolute_offset'] + record['bytes']
    return records


def self_check():
    sample = bytes(range(256)) * 3
    expected = Counter(code for byte in sample for code in (byte & 15, byte >> 4))
    counted = code_counts(sample)
    require(histogram_dict(counted) == {str(code): expected[code] for code in range(16)}, 'code count control failed')
    header = {}
    offset = 0
    for expert in range(896):
        for matrix, (role, rows, width) in MATRICES.items():
            for kind in KINDS:
                shape = [rows, width // (2 if kind == 'weight_packed' else 32)]
                size = math.prod(shape)
                header[f'{PREFIX}{expert}.{matrix}.{kind}'] = {'dtype': 'U8', 'shape': shape, 'data_offsets': [offset, offset + size]}
                offset += size
    header['language_model.model.layers.1.self_attn.q_proj.weight'] = {'not_examined': True}
    header['language_model.model.layers.2.block_sparse_moe.experts.0.w1.weight_packed'] = {'not_examined': True}
    require(len(select_records(header, 4096, offset + 4096)) == 5376, 'selection control failed')
    del header[PREFIX + '0.w1.weight_packed']
    rejected = False
    try:
        select_records(header, 4096, offset + 4096)
    except ValueError:
        rejected = True
    require(rejected, 'missing expert tensor accepted')
    print('PASS: independent nibble counts, exact 896x6 selection, excluded trunk/other-layer records ignored, missing record rejected.')


def inspect(directory):
    directory.mkdir()
    before = SOURCE.stat()
    started = time.perf_counter()
    matrices_totals = {matrix: {'code_counts': np.zeros(16, dtype=np.int64),
                               'scale_counts': np.zeros(256, dtype=np.int64)} for matrix in MATRICES}
    duplicate_candidates = defaultdict(list)
    expert_hashes = defaultdict(list)
    data_hash = hashlib.sha256()
    expert_rows, bytes_read = [], 0
    with SOURCE.open('rb') as stream:
        size_bytes = stream.read(8)
        require(len(size_bytes) == 8, 'short header length')
        header_size = struct.unpack('<Q', size_bytes)[0]
        require(0 < header_size < before.st_size - 8, 'header length outside file')
        raw_header = stream.read(header_size)
        require(len(raw_header) == header_size, 'short checkpoint header')
        records = select_records(json.loads(raw_header), 8 + header_size, before.st_size)
        for expert in range(896):
            branch = {'expert': expert, 'matrices': [], 'bytes': 0}
            expert_hash = hashlib.sha256()
            for matrix, (role, rows, width) in MATRICES.items():
                node = {'name': matrix, 'role': role, 'weight_shape': [rows, width],
                        'groups_per_row': width // 32, 'groups': rows * width // 32,
                        'weight_count': rows * width, 'stored_tensors': []}
                for kind in KINDS:
                    record = records[expert, matrix, kind]
                    stream.seek(record['absolute_offset'])
                    payload = stream.read(record['bytes'])
                    require(len(payload) == record['bytes'], 'short expert payload')
                    bytes_read += len(payload)
                    branch['bytes'] += len(payload)
                    expert_hash.update(payload)
                    data_hash.update(payload)
                    fingerprint = hash_bytes(payload)
                    node['stored_tensors'].append({**record, 'sha256': fingerprint})
                    duplicate_candidates[(kind, tuple(record['shape']), fingerprint)].append((expert, matrix))
                    if kind == 'weight_packed':
                        counts = code_counts(payload)
                        require(int(counts.sum()) == rows * width, 'incomplete code counts')
                        node['code_counts'] = histogram_dict(counts)
                        node['code_bytes'] = len(payload)
                        matrices_totals[matrix]['code_counts'] += counts
                    else:
                        counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
                        require(int(counts.sum()) == node['groups'], 'incomplete scale counts')
                        node['scale_counts'] = histogram_dict(counts)
                        node['scale_bytes'] = len(payload)
                        matrices_totals[matrix]['scale_counts'] += counts
                    del payload
                branch['matrices'].append(node)
            require(branch['bytes'] == 17547264, 'expert stored size differs')
            branch['ordered_payload_sha256'] = expert_hash.hexdigest()
            expert_hashes[expert_hash.hexdigest()].append(expert)
            expert_rows.append(branch)
            if (expert + 1) % 128 == 0:
                print(json.dumps({'experts_read': expert + 1, 'expert_payload_bytes_read': bytes_read}), flush=True)

        confirmed_duplicates = []
        duplicate_comparison_bytes = 0
        for (kind, shape, fingerprint), owners in duplicate_candidates.items():
            if len(owners) < 2:
                continue
            first = records[owners[0][0], owners[0][1], kind]
            stream.seek(first['absolute_offset'])
            original = stream.read(first['bytes'])
            require(len(original) == first['bytes'] and hash_bytes(original) == fingerprint, 'duplicate base changed')
            duplicate_comparison_bytes += len(original)
            for expert, matrix in owners[1:]:
                record = records[expert, matrix, kind]
                stream.seek(record['absolute_offset'])
                compared = stream.read(record['bytes'])
                duplicate_comparison_bytes += len(compared)
                require(compared == original, 'same-hash tensors not byte identical')
            confirmed_duplicates.append({'kind': kind, 'shape': shape, 'sha256': fingerprint,
                                         'owners': [{'expert': expert, 'matrix': matrix} for expert, matrix in owners]})
    after = SOURCE.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'source file changed')
    require(bytes_read == 896 * 17547264 == sum(record['bytes'] for record in records.values()), 'incomplete expert payload coverage')
    aggregate = {}
    for matrix, values in matrices_totals.items():
        scale_values = np.flatnonzero(values['scale_counts'])
        aggregate[matrix] = {'code_counts': histogram_dict(values['code_counts']),
                             'scale_counts': histogram_dict(values['scale_counts']),
                             'distinct_code_patterns': int(np.count_nonzero(values['code_counts'])),
                             'distinct_scale_bytes': int(scale_values.size),
                             'scale_byte_min': int(scale_values.min()), 'scale_byte_max': int(scale_values.max()),
                             'experts_with_all16_codes': sum(len(branch['matrices'][list(MATRICES).index(matrix)]['code_counts']) == 16
                                                            for branch in expert_rows)}
    tree = {'root': 'layer1_routed_experts', 'layer': 1, 'scope': 'Routed expert pool only; trunk, shared experts and other layers excluded.',
            'group_width': 32, 'experts': expert_rows}
    with (directory / 'expert-branches.json').open('x') as destination:
        json.dump(tree, destination, indent=2)
        destination.write('\n')
    report = {'gate': 'PASS', 'layer': 1, 'expert_ids': list(range(896)), 'experts': 896,
              'matrices': 896 * 3, 'stored_tensors': len(records), 'matrix_order': list(MATRICES),
              'stored_code_bytes': 896 * 16515072, 'stored_scale_bytes': 896 * 1032192,
              'complete_expert_payload_bytes_read': bytes_read,
              'matrix_row_nodes': 896 * 9728, 'implicit_32weight_groups': 896 * 1032192,
              'logical_weights': 896 * 33030144, 'per_expert_bytes': 17547264,
              'aggregate_by_matrix': aggregate, 'byte_identical_tensor_groups': confirmed_duplicates,
              'identical_full_expert_hash_groups': [owners for owners in expert_hashes.values() if len(owners) > 1],
              'extra_duplicate_comparison_bytes_read': duplicate_comparison_bytes,
              'ordered_expert_payload_sha256': data_hash.hexdigest(),
              'hash_order': 'Expert IDs ascending, matrix w1/w3/w2, kind weight_packed/weight_scale.',
              'source_file': str(SOURCE), 'source_header_bytes': header_size + 8, 'header_sha256': hash_bytes(raw_header),
              'source_unchanged': True, 'payload_scope': 'Only .layers.1.block_sparse_moe.experts.0..895 tensors were read. Shared shard header was parsed to locate them; no trunk/other-layer payload was inspected.',
              'experts_are_distinct_records_not_an_assumption_of_distinct_values': True,
              'branches_sha256': hash_bytes((directory / 'expert-branches.json').read_bytes()),
              'script_sha256': hash_bytes(Path(__file__).read_bytes()), 'seconds': time.perf_counter() - started,
              'limits': 'Read-only storage exploration, not a model execution, tree compression or a full decoded-weight value census. Separate code/scale histograms do not establish which code-scale pairs occur, spatial correlations or cross-expert compressibility.'}
    with (directory / 'results.json').open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('expert_ids', 'aggregate_by_matrix', 'byte_identical_tensor_groups')}, indent=2))
    print('matrix_summary', json.dumps({matrix: {key: value for key, value in result.items() if key not in ('code_counts', 'scale_counts')}
                                      for matrix, result in aggregate.items()}, indent=2))
    print('duplicate_tensor_groups', len(confirmed_duplicates))
    print('report_sha256', hash_bytes((directory / 'results.json').read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('self-check', 'inspect'))
    parser.add_argument('directory', type=Path, nargs='?')
    args = parser.parse_args()
    if args.command == 'self-check':
        self_check()
    else:
        require(args.directory is not None, 'output directory required')
        inspect(args.directory)


if __name__ == '__main__':
    main()