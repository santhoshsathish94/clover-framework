#!/usr/bin/env python3
"""Self-contained shared-value tables plus bounded expert leaf records."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import zlib

import numpy as np


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('root_value_census', HERE / 'root-tables.py')
census_code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census_code)
require = census_code.require
digest = census_code.digest
save_json = census_code.save_json
MATRICES = census_code.MATRICES
SHAPES = census_code.SHAPES
HEADER = struct.Struct('<8sIIQQ32s')
CODECS = ('grouped_raw', 'grouped_zlib', 'direct_packed', 'direct_zlib', 'direct_bitplanes_zlib')
BLOCK_ROWS = 64


def pack(values, bits, planes=False):
    fields = ((values[:, None].astype(np.uint32) >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
    if planes:
        fields = fields.T
    return np.packbits(fields.ravel(), bitorder='little').tobytes()


def unpack(raw, count, bits, planes=False):
    require(len(raw) == (count * bits + 7) // 8, 'packed reference length differs')
    binary = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='little')
    require(not np.any(binary[count * bits:]), 'nonzero reference padding')
    fields = binary[:count * bits].reshape(bits, count).T if planes else binary[:count * bits].reshape(count, bits)
    return np.sum(fields.astype(np.uint32) << np.arange(bits, dtype=np.uint32), axis=1, dtype=np.uint32)


def inflate(raw, expected):
    decoder = zlib.decompressobj()
    result = decoder.decompress(raw, expected + 1)
    require(len(result) == expected and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
            'compressed expert record framing differs')
    return result


def palette_for(domains, record, pair_counts):
    branch = 'root-up' if record['matrix'] != 'w2' else 'root-down'
    palette = np.array(domains['root'] + domains[branch] + record['local_value_bits_bf16'], dtype='<u2')
    require(len(palette) == len(set(map(int, palette))) == record['distinct_values'], 'palette overlap or incomplete')
    value_to_id = np.full(65536, -1, dtype=np.int32)
    value_to_id[palette] = np.arange(len(palette))
    decoded = census_code.decode_table().view('<u4')
    group_scales = np.flatnonzero(pair_counts.sum(axis=1)).astype(np.uint8)
    mapping = np.full((len(group_scales), 16), 255, dtype=np.uint8)
    for slot, scale in enumerate(group_scales):
        present = pair_counts[scale] > 0
        require(np.all((decoded[scale, present] & 65535) == 0), 'lossy shared value')
        refs = value_to_id[decoded[scale, present] >> 16]
        require(np.all(refs >= 0) and np.all(refs < 255), 'unmapped shared value')
        mapping[slot, present] = refs
    require(len(palette) < 255, 'byte palette mapping capacity exceeded')
    return palette, group_scales, mapping


def encode_candidates(packed, scales, palette, group_scales, mapping, wanted=None):
    wanted = list(range(len(CODECS))) if wanted is None else wanted
    packed_words = np.frombuffer(packed, dtype=np.uint8)
    scale_words = np.frombuffer(scales, dtype=np.uint8)
    require(len(packed) == len(scales) * 16, 'invalid input group dimensions')
    select = np.full(256, -1, dtype=np.int32)
    select[group_scales] = np.arange(len(group_scales))
    selectors = select[scale_words]
    require(np.all(selectors >= 0), 'unrecognized group scale')
    grouped = packed + selectors.astype(np.uint8).tobytes()
    result = {}
    if 0 in wanted:
        result[0] = grouped
    if 1 in wanted:
        result[1] = zlib.compress(grouped, 6)
    codes = np.empty(packed_words.size * 2, dtype=np.uint8)
    codes[::2], codes[1::2] = packed_words & 15, packed_words >> 4
    refs = mapping[selectors[:, None], codes.reshape(-1, 32)].ravel()
    require(np.all(refs < len(palette)), 'invalid group palette reference')
    bits = (len(palette) - 1).bit_length()
    if 2 in wanted:
        result[2] = pack(refs, bits)
    if 3 in wanted:
        result[3] = zlib.compress(refs.tobytes(), 6)
    if 4 in wanted:
        result[4] = zlib.compress(pack(refs, bits, planes=True), 6)
    return result, palette[refs].astype('<u2', copy=False).tobytes()


def decode_payload(payload, codec, count, palette, scale_count, mapping):
    require(codec in range(5) and count % 32 == 0, 'invalid expert codec/count')
    bits = (len(palette) - 1).bit_length()
    if codec in (0, 1):
        expected = count // 2 + count // 32
        grouped = payload if codec == 0 else inflate(payload, expected)
        require(len(grouped) == expected, 'invalid grouped payload size')
        packed = np.frombuffer(grouped, dtype=np.uint8, count=count // 2)
        selectors = np.frombuffer(grouped, dtype=np.uint8, offset=count // 2)
        require(np.all(selectors < scale_count), 'group selector outside palette')
        codes = np.empty(count, dtype=np.uint8)
        codes[::2], codes[1::2] = packed & 15, packed >> 4
        refs = mapping[selectors[:, None], codes.reshape(-1, 32)].ravel()
    elif codec == 2:
        refs = unpack(payload, count, bits)
    elif codec == 3:
        refs = np.frombuffer(inflate(payload, count), dtype=np.uint8)
    else:
        refs = unpack(inflate(payload, (count * bits + 7) // 8), count, bits, True)
    require(np.all(refs < len(palette)), 'expert value reference outside palette')
    return palette[refs].astype('<u2', copy=False).tobytes()


class RootTableReader:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.file = (self.directory / 'expert.bin').open('rb')
        try:
            raw = self.file.read(HEADER.size)
            require(len(raw) == HEADER.size, 'short expert header')
            magic, version, size, compressed_size, payload_size, expected = HEADER.unpack(raw)
            require(magic == b'K3ROOT1\0' and version == 1 and 0 < size <= 128 * 1024 * 1024
                    and 0 < compressed_size <= 64 * 1024 * 1024, 'unsupported expert table header')
            require(HEADER.size + compressed_size + payload_size == (self.directory / 'expert.bin').stat().st_size,
                    'expert file size differs')
            metadata = inflate(self.file.read(compressed_size), size)
            require(hashlib.sha256(metadata).digest() == expected, 'expert metadata checksum differs')
            self.metadata = json.loads(metadata)
            require(self.metadata['layer'] == 1 and self.metadata['block_rows'] == BLOCK_ROWS
                    and self.metadata['codecs'] == list(CODECS), 'unsupported expert metadata')
            self.tables = {}
            for name in ('root', 'root-up', 'root-down'):
                data = (self.directory / (name + '.bin')).read_bytes()
                record = self.metadata['tables'][name]
                require(len(data) == 2 + record['values'] * 2 and struct.unpack_from('<H', data)[0] == 1
                        and hashlib.sha256(data).hexdigest() == record['sha256'], 'shared table identity differs')
                self.tables[name] = np.frombuffer(data, dtype='<u2', offset=2)
            self.payload_start = HEADER.size + compressed_size
            self.records = {}
            cursor = 0
            for record in self.metadata['experts']:
                key = (record['expert_id'], record['matrix'])
                require(key not in self.records and 0 <= key[0] < 896 and key[1] in MATRICES
                        and record['root_id'] == 1, 'invalid expert foreign key or identity')
                require(record['local_offset'] == cursor and 0 <= record['local_count'] <= 254, 'invalid local value range')
                cursor += record['local_count'] * 2
                require(record['group_palette_offset'] == cursor and 1 <= len(record['group_scales']) <= 256,
                        'invalid group palette range')
                cursor += len(record['group_scales']) * 16
                rows, width = SHAPES[MATRICES.index(key[1])]
                require(len(record['blocks']) == rows // BLOCK_ROWS, 'missing expert blocks')
                for entry in record['blocks']:
                    offset, length, codec, checksum = entry
                    require(offset == cursor and 0 < length <= BLOCK_ROWS * width and codec in range(5)
                            and len(checksum) == 64, 'invalid expert block')
                    cursor += length
                require(len(record['value_sha256']) == 64, 'missing matrix value hash')
                self.records[key] = record
            require(set(self.records) == {(expert, matrix) for expert in range(896) for matrix in MATRICES}
                    and cursor == payload_size, 'incomplete expert table')
        except Exception:
            self.file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *ignored):
        self.file.close()

    def bindings(self, expert, matrix):
        require((expert, matrix) in self.records, 'expert record not found')
        record = self.records[expert, matrix]
        self.file.seek(self.payload_start + record['local_offset'])
        local = self.file.read(record['local_count'] * 2)
        require(len(local) == record['local_count'] * 2, 'short local values')
        self.file.seek(self.payload_start + record['group_palette_offset'])
        raw = self.file.read(len(record['group_scales']) * 16)
        require(len(raw) == len(record['group_scales']) * 16, 'short group palettes')
        branch = 'root-up' if matrix != 'w2' else 'root-down'
        palette = np.concatenate((self.tables['root'], self.tables[branch], np.frombuffer(local, dtype='<u2')))
        mapping = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 16)
        require(len(palette) < 255 and np.all((mapping < len(palette)) | (mapping == 255)), 'invalid group references')
        return record, palette, mapping

    def read_block(self, expert, matrix, block, bindings=None):
        record, palette, mapping = bindings or self.bindings(expert, matrix)
        require(0 <= block < len(record['blocks']), 'expert block out of range')
        offset, size, codec, checksum = record['blocks'][block]
        self.file.seek(self.payload_start + offset)
        payload = self.file.read(size)
        require(len(payload) == size, 'short expert block')
        width = SHAPES[MATRICES.index(matrix)][1]
        values = decode_payload(payload, codec, BLOCK_ROWS * width, palette, len(record['group_scales']), mapping)
        require(hashlib.sha256(values).hexdigest() == checksum, 'reconstructed values checksum differs')
        return values

    def read_row(self, expert, matrix, row, float32=True):
        require(matrix in MATRICES, 'invalid expert matrix')
        rows, width = SHAPES[MATRICES.index(matrix)]
        require(isinstance(row, int) and 0 <= row < rows, 'expert row out of range')
        block = self.read_block(expert, matrix, row // BLOCK_ROWS)
        start = row % BLOCK_ROWS * width * 2
        raw = block[start:start + width * 2]
        if not float32:
            return raw
        return (np.frombuffer(raw, dtype='<u2').astype('<u4') << 16).view('<f4')


def self_check():
    census_code.self_check()
    palette = np.array([0, 0x8000, 0x3d80, 0xbc80], dtype='<u2')
    scales = np.array([120, 121], dtype=np.uint8)
    mapping = np.tile(np.arange(16, dtype=np.uint8) % 4, (2, 1))
    packed = bytes(range(256)) * 2
    original_scales = bytes([120, 121] * 16)
    encoded, expected = encode_candidates(packed, original_scales, palette, scales, mapping)
    for codec, payload in encoded.items():
        require(decode_payload(payload, codec, len(packed) * 2, palette, 2, mapping) == expected, 'leaf representation mismatch')
    ids = np.array([0, 1, 7, 2, 5], dtype=np.uint8)
    for planes in (False, True):
        raw = pack(ids, 3, planes)
        order = [int(value >> bit) & 1 for bit in range(3) for value in ids] if planes else [int(value >> bit) & 1 for value in ids for bit in range(3)]
        require(raw == sum(bit << index for index, bit in enumerate(order)).to_bytes(2, 'little'), 'independent bitpacking mismatch')
        require(np.array_equal(unpack(raw, 5, 3, planes), ids), 'field decode mismatch')
    for payload, expected_size in ((zlib.compress(b'abc') + b'junk', 3), (zlib.compress(b'abcd'), 3)):
        rejected = False
        try:
            inflate(payload, expected_size)
        except ValueError:
            rejected = True
        require(rejected, 'malformed compressed payload accepted')
    print('PASS: five leaf layouts, group references, signed-zero values, independent packed fields and bad framing rejection.')


def inputs(directory):
    census = json.loads((directory / 'census.json').read_text())
    require(census['gate'] == 'PASS', 'root census incomplete')
    for name, expected in census['artifacts'].items():
        require(digest(directory / name) == expected, 'census artifact changed')
    domains = json.loads((directory / 'domains.json').read_text())
    with np.load(directory / 'code-scale-counts.npz') as data:
        counts = data['counts']
    records = domains['experts']
    require(len(records) == 2688 and counts.shape == (896, 3, 256, 16), 'census coverage differs')
    source_map = json.loads(census_code.SOURCE_MAP.read_text())
    require(digest(census_code.SOURCE_MAP) == census_code.SOURCE_MAP_SHA, 'source map changed')
    locations = {(expert['expert'], matrix['name']): matrix for expert in source_map['experts'] for matrix in expert['matrices']}
    return census, domains, counts, records, locations


def source_matrix(stream, expert, matrix, locations):
    record = locations[expert, matrix]
    values = []
    for kind in ('weight_packed', 'weight_scale'):
        entry = next(row for row in record['stored_tensors'] if row['name'].endswith('.' + kind))
        stream.seek(entry['absolute_offset'])
        blob = stream.read(entry['bytes'])
        require(len(blob) == entry['bytes'] and hashlib.sha256(blob).hexdigest() == entry['sha256'], 'source expert tensor differs')
        values.append(blob)
    return values


def compare(directory):
    census, domains, counts, records, locations = inputs(directory)
    totals, examples = Counter(), []
    experts = (0, 127, 255, 447, 448, 639, 767, 895)
    with census_code.SOURCE.open('rb') as stream:
        for expert in experts:
            for matrix_index, matrix in enumerate(MATRICES):
                record = records[expert * 3 + matrix_index]
                palette, scales, mapping = palette_for(domains, record, counts[expert, matrix_index])
                packed, original_scales = source_matrix(stream, expert, matrix, locations)
                rows, width = SHAPES[matrix_index]
                for first_row in (0, rows // 2, rows - BLOCK_ROWS):
                    raw = packed[first_row * width // 2:(first_row + BLOCK_ROWS) * width // 2]
                    scale_raw = original_scales[first_row * width // 32:(first_row + BLOCK_ROWS) * width // 32]
                    candidates, expected = encode_candidates(raw, scale_raw, palette, scales, mapping)
                    for codec, payload in candidates.items():
                        require(decode_payload(payload, codec, BLOCK_ROWS * width, palette, len(scales), mapping) == expected,
                                'sample layout reconstruction failed')
                        totals[CODECS[codec]] += len(payload)
                    examples.append({'expert': expert, 'matrix': matrix, 'first_row': first_row,
                                     'candidate_bytes': {CODECS[codec]: len(payload) for codec, payload in candidates.items()}})
    best_direct = min((2, 3, 4), key=lambda codec: totals[CODECS[codec]])
    report = {'gate': 'PASS', 'sample_experts': experts, 'sample_blocks': len(examples), 'block_rows': BLOCK_ROWS,
              'candidate_payload_totals': dict(totals), 'full_build_candidates': [0, 1, best_direct],
              'selection_rule': 'For every full-table block compare raw grouped references, zlib grouped references, and the direct-reference layout with lowest total sample bytes.',
              'limits': 'Layout shortlist based on72blocks from8experts; not a proof of globally minimal layout/block size.',
              'blocks': examples, 'script_sha256': digest(Path(__file__))}
    save_json(directory / 'layout-comparison.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'blocks'}, indent=2))


def build(directory):
    census, domains, counts, records, locations = inputs(directory)
    comparison = json.loads((directory / 'layout-comparison.json').read_text())
    require(comparison['gate'] == 'PASS' and comparison['script_sha256'] == digest(Path(__file__)), 'layout comparison stale')
    chosen = comparison['full_build_candidates']
    require(chosen[:2] == [0, 1] and chosen[2] in (2, 3, 4), 'invalid layout shortlist')
    pending = directory / 'expert.payload.pending'
    metadata = {'layer': 1, 'block_rows': BLOCK_ROWS, 'codecs': list(CODECS),
                'tables': {name: {'values': len(domains[name]), 'sha256': digest(directory / (name + '.bin'))}
                           for name in ('root', 'root-up', 'root-down')}, 'experts': []}
    frequencies, totals = Counter(), Counter()
    source_hash, decoded_hash = hashlib.sha256(), hashlib.sha256()
    before = census_code.SOURCE.stat()
    with census_code.SOURCE.open('rb') as stream, pending.open('xb') as payload_stream, (directory / 'expert-blocks.jsonl').open('x') as audit:
        for record in records:
            expert, matrix = record['expert'], record['matrix']
            matrix_index = MATRICES.index(matrix)
            rows, width = SHAPES[matrix_index]
            palette, scales, mapping = palette_for(domains, record, counts[expert, matrix_index])
            raw_codes, raw_scales = source_matrix(stream, expert, matrix, locations)
            source_hash.update(raw_codes)
            source_hash.update(raw_scales)
            node = {'root_id': 1, 'expert_id': expert, 'matrix': matrix,
                    'local_count': len(record['local_value_bits_bf16']), 'local_offset': payload_stream.tell(),
                    'group_scales': scales.tolist(), 'blocks': []}
            payload_stream.write(np.array(record['local_value_bits_bf16'], dtype='<u2').tobytes())
            node['group_palette_offset'] = payload_stream.tell()
            payload_stream.write(mapping.tobytes())
            matrix_hash = hashlib.sha256()
            for first_row in range(0, rows, BLOCK_ROWS):
                packed = raw_codes[first_row * width // 2:(first_row + BLOCK_ROWS) * width // 2]
                scale_raw = raw_scales[first_row * width // 32:(first_row + BLOCK_ROWS) * width // 32]
                options, expected = encode_candidates(packed, scale_raw, palette, scales, mapping, chosen)
                codec = min(chosen, key=lambda choice: (len(options[choice]), choice))
                payload = options[codec]
                require(decode_payload(payload, codec, BLOCK_ROWS * width, palette, len(scales), mapping) == expected,
                        'built leaf block roundtrip differs')
                checksum = hashlib.sha256(expected).hexdigest()
                node['blocks'].append([payload_stream.tell(), len(payload), codec, checksum])
                payload_stream.write(payload)
                matrix_hash.update(expected)
                decoded_hash.update(expected)
                frequencies[CODECS[codec]] += 1
                totals.update({CODECS[choice]: len(options[choice]) for choice in chosen})
                audit.write(json.dumps({'expert': expert, 'matrix': matrix, 'first_row': first_row,
                                        'candidate_bytes': {CODECS[choice]: len(options[choice]) for choice in chosen},
                                        'selected': CODECS[codec], 'stored_bytes': len(payload)}, separators=(',', ':')) + '\n')
            node['value_sha256'] = matrix_hash.hexdigest()
            metadata['experts'].append(node)
            if matrix == 'w2' and (expert + 1) % 128 == 0:
                print(json.dumps({'experts_built': expert + 1, 'payload_bytes': payload_stream.tell()}), flush=True)
    after = census_code.SOURCE.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'source changed')
    require(source_hash.hexdigest() == census['source_sha256'], 'full source payload hash changed')
    raw_metadata = json.dumps(metadata, separators=(',', ':')).encode()
    compressed_metadata = zlib.compress(raw_metadata, 9)
    final_pending = directory / 'expert.pending'
    with final_pending.open('xb') as output, pending.open('rb') as source:
        output.write(HEADER.pack(b'K3ROOT1\0', 1, len(raw_metadata), len(compressed_metadata), pending.stat().st_size,
                                 hashlib.sha256(raw_metadata).digest()))
        output.write(compressed_metadata)
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            output.write(chunk)
    require(not (directory / 'expert.bin').exists(), 'expert table already exists')
    final_pending.rename(directory / 'expert.bin')
    pending.unlink()
    tables_bytes = sum((directory / (name + '.bin')).stat().st_size for name in ('root', 'root-up', 'root-down'))
    total = (directory / 'expert.bin').stat().st_size + tables_bytes
    report = {'state': 'BUILT_REOPEN_VERIFICATION_PENDING', 'layer': 1, 'experts': 896,
              'matrix_records': 2688, 'blocks': sum(frequencies.values()), 'block_rows': BLOCK_ROWS,
              'shared_value_counts': census['shared_value_counts'], 'local_values_total': census['expert_local_values_total'],
              'selected_codecs': dict(frequencies), 'candidate_payload_totals': dict(totals),
              'expert_file_bytes': (directory / 'expert.bin').stat().st_size, 'shared_tables_bytes': tables_bytes,
              'total_storage_bytes': total, 'original_bytes': census['source_bytes_read'],
              'saved_bytes': census['source_bytes_read'] - total, 'header_bytes': HEADER.size,
              'compressed_index_bytes': len(compressed_metadata), 'uncompressed_index_bytes': len(raw_metadata),
              'local_value_bytes': census['expert_local_values_total'] * 2,
              'group_palette_bytes': sum(len(node['group_scales']) * 16 for node in metadata['experts']),
              'source_sha256': source_hash.hexdigest(), 'ordered_bf16_weights_sha256': decoded_hash.hexdigest(),
              'file_sha256': {name: digest(directory / name) for name in ('root.bin', 'root-up.bin', 'root-down.bin', 'expert.bin')},
              'census_sha256': digest(directory / 'census.json'), 'layout_comparison_sha256': digest(directory / 'layout-comparison.json'),
              'block_audit_sha256': digest(directory / 'expert-blocks.jsonl'),
              'implementation_sha256': {name: digest(HERE / name) for name in ('root-table-store.py', 'root-tables.py')},
              'limits': 'Complete stored-value representation for layer1 only; all grouping/index metadata counted, audit files/software excluded. Reopened full comparison still required. Not model-integrated or globally optimal.'}
    save_json(directory / 'build-results.json', report)
    print(json.dumps(report, indent=2), flush=True)


def verify(directory):
    census, domains, counts, records, locations = inputs(directory)
    built = json.loads((directory / 'build-results.json').read_text())
    for name, expected in built['file_sha256'].items():
        require(digest(directory / name) == expected, 'stored table hash differs')
    decoded = census_code.decode_table().view('<u4')
    whole = hashlib.sha256()
    joint_matches = 0
    group_original_bytes_checked = 0
    sample_rows = []
    before = census_code.SOURCE.stat()
    with RootTableReader(directory) as reader, census_code.SOURCE.open('rb') as source:
        for record in records:
            expert, matrix = record['expert'], record['matrix']
            matrix_index = MATRICES.index(matrix)
            rows, width = SHAPES[matrix_index]
            packed, scales = source_matrix(source, expert, matrix, locations)
            bindings = reader.bindings(expert, matrix)
            observed = np.zeros((256, 16), dtype=np.uint64)
            for block, first_row in enumerate(range(0, rows, BLOCK_ROWS)):
                code_raw = packed[first_row * width // 2:(first_row + BLOCK_ROWS) * width // 2]
                scale_raw = scales[first_row * width // 32:(first_row + BLOCK_ROWS) * width // 32]
                codes = np.frombuffer(code_raw, dtype=np.uint8)
                unpacked = np.empty(codes.size * 2, dtype=np.uint8)
                unpacked[::2], unpacked[1::2] = codes & 15, codes >> 4
                scaled = np.repeat(np.frombuffer(scale_raw, dtype=np.uint8), 32)
                expected_bits = decoded[scaled, unpacked]
                restored = reader.read_block(expert, matrix, block, bindings)
                restored_bits = np.frombuffer(restored, dtype='<u2').astype('<u4') << 16
                require(np.array_equal(restored_bits, expected_bits), 'reopened shared-value expert weights differ')
                whole.update(restored)
                observed += census_code.pair_counts(code_raw, scale_raw)
                offset, size, codec, checksum = bindings[0]['blocks'][block]
                if codec <= 1:
                    reader.file.seek(reader.payload_start + offset)
                    raw = reader.file.read(size)
                    grouped = raw if codec == 0 else inflate(raw, len(code_raw) + len(scale_raw))
                    require(grouped[:len(code_raw)] == code_raw, 'original grouped codes differ')
                    selected = np.frombuffer(grouped, dtype=np.uint8, offset=len(code_raw))
                    rebuilt_scales = np.array(bindings[0]['group_scales'], dtype=np.uint8)[selected].tobytes()
                    require(rebuilt_scales == scale_raw, 'original grouped scales differ')
                    group_original_bytes_checked += len(code_raw) + len(scale_raw)
            require(np.array_equal(observed, counts[expert, matrix_index]), 'joint census changed')
            joint_matches += int(observed.sum())
            if expert in (0, 447, 448, 895):
                row = rows - 1 if expert else 0
                raw = reader.read_row(expert, matrix, row, False)
                sample_rows.append({'expert': expert, 'matrix': matrix, 'row': row,
                                    'values': width, 'sha256': hashlib.sha256(raw).hexdigest(),
                                    'first_eight_float32': reader.read_row(expert, matrix, row)[:8].tolist()})
            if matrix == 'w2' and (expert + 1) % 128 == 0:
                print(json.dumps({'experts_verified': expert + 1, 'weights_checked': joint_matches}), flush=True)
        require(joint_matches == 29595009024 and whole.hexdigest() == built['ordered_bf16_weights_sha256'], 'full weights hash differs')
    after = census_code.SOURCE.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'source changed during verification')
    report = {'gate': 'PASS', **{key: value for key, value in built.items() if key not in ('state', 'limits')},
              'all_decoded_float32_bits_equal_source': True, 'values_verified': joint_matches,
              'all_joint_frequencies_match_census': True, 'source_unchanged': True,
              'original_code_scale_bytes_reconstructed': group_original_bytes_checked,
              'row_read_examples': sample_rows,
              'table_semantics': 'root/root-up/root-down contain only uint16LE layer/root FK plus BF16 value cells; schemas and lengths live in authenticated expert metadata. Expert palettes concatenate root then branch then local values; references index this combined list.',
              'limits': 'Verified standalone layer1 expert-weight storage, no trunk/other layers/model execution. Grouped layouts recover original code/scale bytes; direct layouts guarantee identical decoded weights, not original quantized factorization. Compression is smallest among tested shortlist per64rowblock, not a global optimum. No speed/RAM benefit claim.'}
    save_json(directory / 'results.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'row_read_examples'}, indent=2), flush=True)
    print('results_sha256', digest(directory / 'results.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('self-check', 'compare', 'build', 'verify', 'read'))
    parser.add_argument('directory', type=Path, nargs='?')
    parser.add_argument('--expert', type=int, default=0)
    parser.add_argument('--matrix', choices=MATRICES, default='w1')
    parser.add_argument('--row', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'self-check':
        self_check()
    elif args.command == 'read':
        with RootTableReader(args.directory) as reader:
            values = reader.read_row(args.expert, args.matrix, args.row)
            print(json.dumps({'expert': args.expert, 'matrix': args.matrix, 'row': args.row,
                              'first_eight_values': values[:8].tolist(), 'values': int(values.size)}, indent=2))
    else:
        globals()[args.command](args.directory)


if __name__ == '__main__':
    main()