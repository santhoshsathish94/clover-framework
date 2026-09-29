#!/usr/bin/env python3
"""Exact root/root-up/root-down value factoring for layer1 routed experts."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


SOURCE = Path('/root/k3model/model-00002-of-000096.safetensors')
PREFIX = 'language_model.model.layers.1.block_sparse_moe.experts.'
MATRICES = ('w1', 'w3', 'w2')
SHAPES = ((3072, 3584), (3072, 3584), (3584, 3072))
SOURCE_MAP = Path('/opt/clover-k3/layer1-experts-only-20260929-a/data/expert-branches.json')
SOURCE_MAP_SHA = '78adf5b56e17fab672776784d8f10ee1acac53892945e74ba2b9125efb4b2e7b'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save_json(path, value):
    with Path(path).open('x') as destination:
        json.dump(value, destination, indent=2)
        destination.write('\n')


def decode_table():
    codebook = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype='<f4')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        factors = np.exp2(np.arange(256, dtype=np.float64) - 127).astype('<f4')
        factors[255] = 0
        return (factors[:, None] * codebook[None, :]).astype('<f4')


def pair_counts(packed, scales):
    require(len(packed) == len(scales) * 16, 'code/scale group mismatch')
    pairs = np.zeros((256, 16), dtype=np.uint64)
    for start in range(0, len(scales), 16384):
        count = min(16384, len(scales) - start)
        codes = np.frombuffer(packed, dtype=np.uint8, count=count * 16, offset=start * 16).reshape(count, 16)
        scaling = np.frombuffer(scales, dtype=np.uint8, count=count, offset=start)
        keys = codes.astype(np.uint32) + scaling.astype(np.uint32)[:, None] * 256
        histogram = np.bincount(keys.ravel(), minlength=65536).reshape(256, 16, 16).astype(np.uint64)
        pairs += histogram.sum(axis=1) + histogram.sum(axis=2)
    require(int(pairs.sum()) == len(packed) * 2, 'pair frequencies incomplete')
    return pairs


def common_tables(domains):
    root = set.intersection(*(set(values) for values in domains.values()))
    up = set.intersection(*(set(values) for (expert, matrix), values in domains.items() if matrix < 2)) - root
    down = set.intersection(*(set(values) for (expert, matrix), values in domains.items() if matrix == 2)) - root
    local = {key: sorted(set(values) - root - (up if key[1] < 2 else down)) for key, values in domains.items()}
    return sorted(root), sorted(up), sorted(down), local


def self_check():
    packed = bytes(range(256)) * 16
    scales = bytes(range(256))
    measured = pair_counts(packed, scales)
    expected = Counter()
    for index, byte in enumerate(packed):
        expected[scales[index // 16], byte & 15] += 1
        expected[scales[index // 16], byte >> 4] += 1
    require(all(int(measured[scale, code]) == expected[scale, code] for scale in range(256) for code in range(16)), 'joint count mismatch')
    domains = {(expert, matrix): {1, 2, 3 if matrix < 2 else 4, 100 + expert * 3 + matrix}
               for expert in range(2) for matrix in range(3)}
    root, up, down, local = common_tables(domains)
    require((root, up, down) == ([1, 2], [3], [4]), 'shared table split differs')
    for key, values in domains.items():
        require(set(root + (up if key[1] < 2 else down) + local[key]) == values, 'shared/local reconstruction differs')
    table = decode_table().view('<u4')
    require(int(table[121, 6]) == 0x3d800000 and int(table[121, 8]) == 0x80000000, 'known weight bits differ')
    require(int(table[255, 8]) == 0x80000000 and int(table[255, 0]) == 0, 'signed zero lost')
    print('PASS: independent joint code-scale counts, exact shared/local sets, known weight bits and signed zero.')


def census(directory):
    directory.mkdir()
    require(digest(SOURCE_MAP) == SOURCE_MAP_SHA, 'source map identity differs')
    source_map = json.loads(SOURCE_MAP.read_text())
    expected = {record['name']: record for expert in source_map['experts']
                for matrix in expert['matrices'] for record in matrix['stored_tensors']}
    before = SOURCE.stat()
    all_pairs = np.zeros((896, 3, 256, 16), dtype=np.uint64)
    table_bits = decode_table().view('<u4')
    domains, entries = {}, []
    whole = hashlib.sha256()
    total_bytes = 0
    with SOURCE.open('rb') as source:
        header_size = struct.unpack('<Q', source.read(8))[0]
        raw_header = source.read(header_size)
        header = json.loads(raw_header)
        names = {name for name in header if name.startswith(PREFIX)}
        require(names == set(expected) and len(names) == 5376, 'expert-only scope differs')
        for expert in range(896):
            for matrix_index, matrix in enumerate(MATRICES):
                blobs = []
                for kind in ('weight_packed', 'weight_scale'):
                    name = f'{PREFIX}{expert}.{matrix}.{kind}'
                    stored = header[name]
                    start, end = stored['data_offsets']
                    record = expected[name]
                    require(stored['dtype'] == 'U8' and stored['shape'] == record['shape']
                            and 8 + header_size + start == record['absolute_offset']
                            and end - start == record['bytes'], 'source tensor declaration differs')
                    source.seek(record['absolute_offset'])
                    blob = source.read(record['bytes'])
                    require(len(blob) == record['bytes'] and hashlib.sha256(blob).hexdigest() == record['sha256'], 'source tensor bytes differ')
                    whole.update(blob)
                    total_bytes += len(blob)
                    blobs.append(blob)
                pairs = pair_counts(*blobs)
                all_pairs[expert, matrix_index] = pairs
                observed = table_bits[pairs > 0]
                require(np.all((observed & 65535) == 0), 'observed weight not exactly BF16')
                require(np.all(((observed >> 23) & 255) != 255), 'observed nonfinite weight')
                domain = sorted(set(map(int, observed >> 16)))
                domains[expert, matrix_index] = domain
                entries.append({'expert': expert, 'matrix': matrix, 'value_bits_bf16': domain,
                                'distinct_values': len(domain), 'observed_code_scale_pairs': int(np.count_nonzero(pairs))})
            if (expert + 1) % 128 == 0:
                print(json.dumps({'experts_scanned': expert + 1, 'bytes': total_bytes}), flush=True)
    after = SOURCE.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'source changed')
    require(whole.hexdigest() == 'cfeb895c80fa5b34fb219c3e3a364d5ae9cbbeee414cb35e0651d38d9ec42516', 'full expert-only source hash differs')
    root, up, down, local = common_tables(domains)
    for entry in entries:
        key = (entry['expert'], MATRICES.index(entry['matrix']))
        entry['local_value_bits_bf16'] = local[key]
    with (directory / 'code-scale-counts.npz').open('xb') as output:
        np.savez_compressed(output, counts=all_pairs)
    save_json(directory / 'domains.json', {'root': root, 'root-up': up, 'root-down': down, 'experts': entries})
    for name, value_list in (('root', root), ('root-up', up), ('root-down', down)):
        with (directory / (name + '.bin')).open('xb') as output:
            output.write(struct.pack('<H', 1))
            output.write(np.array(value_list, dtype='<u2').tobytes())
    report = {'gate': 'PASS', 'layer': 1, 'experts': 896, 'matrix_records': len(entries),
              'all_actual_weight_pairs_counted': int(all_pairs.sum()), 'source_bytes_read': total_bytes,
              'source_sha256': whole.hexdigest(), 'source_unchanged': True,
              'shared_value_counts': {'root': len(root), 'root-up': len(up), 'root-down': len(down)},
              'distinct_values_across_pool': len(set().union(*map(set, domains.values()))),
              'expert_local_values_total': sum(map(len, local.values())),
              'expert_local_value_count_range': [min(map(len, local.values())), max(map(len, local.values()))],
              'matrix_palette_size_range': [min(map(len, domains.values())), max(map(len, domains.values()))],
              'table_encoding': 'Each .bin contains uint16LE layer/root_id=1 followed only by uint16LE BF16 value cells. Lengths/schema in external metadata; signed zeros remain distinct.',
              'common_definition': 'root: exact values present in every expert matrix; root-up/down: additional intersection for that matrix family; remaining values local per expert/matrix.',
              'artifacts': {name: digest(directory / name) for name in ('domains.json', 'code-scale-counts.npz', 'root.bin', 'root-up.bin', 'root-down.bin')},
              'script_sha256': digest(Path(__file__)),
              'limits': 'Layer1 expert-only full joint census; other layer rows not populated. Values-only tables plus schema are not yet a standalone expert representation.'}
    save_json(directory / 'census.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('self-check', 'census'))
    parser.add_argument('directory', type=Path, nargs='?')
    args = parser.parse_args()
    if args.command == 'self-check':
        self_check()
    else:
        census(args.directory)


if __name__ == '__main__':
    main()