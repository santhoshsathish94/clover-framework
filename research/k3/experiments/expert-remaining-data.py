#!/usr/bin/env python3
"""Inspect the remaining constant-store bytes without changing the dataset."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import zlib


HEADER = struct.Struct('<8s8I3Q')
ENTRY = struct.Struct('<Q4I')
MATRICES = ('w1', 'w3', 'w2')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def float_value(bits):
    return '-0' if bits == 0x8000 else struct.unpack('<f', struct.pack('<I', bits << 16))[0]


def inspect(directory, reader, output):
    evidence = json.loads((directory / 'results.json').read_text())
    require(evidence['gate'] == 'PASS', 'full native verification missing')
    path = directory / 'experts.bin'
    before = path.stat()
    require(digest(path) == evidence['dataset_sha256'], 'dataset identity differs')
    require(digest(reader) == evidence['reader_sha256'], 'native reader identity differs')
    palette_raw = subprocess.check_output([str(reader), 'palette'])
    require(hashlib.sha256(palette_raw).hexdigest() == evidence['compiled_palette_sha256'], 'C palette differs')
    palette = struct.unpack('<66H', palette_raw)
    map_lists, map_rows = Counter(), Counter()
    by_matrix = {name: {'lists': Counter(), 'rows': Counter(), 'map_counts': Counter(),
                        'compressed_bytes': 0} for name in MATRICES}
    codecs, block_sizes, row_counts = Counter(), Counter(), Counter()
    unique_ref_ids = set()
    map_bytes = compressed_bytes = absent = 0
    cursor = next_block = 0
    samples = []
    with path.open('rb') as stream:
        header_raw = stream.read(HEADER.size)
        magic, version, layer, records, blocks, record_stride, block_stride, palette_crc, index_crc, index_size, payload_size, reserved = HEADER.unpack(header_raw)
        require((magic, version, layer, records, blocks, record_stride, block_stride, reserved) ==
                (b'K3CONST1', 1, 1, 2688, 136192, 24, 24, 0), 'unsupported format')
        index = stream.read(index_size)
        require(index_size == (records + blocks) * 24 and zlib.crc32(index) == index_crc, 'index differs')
        require(zlib.crc32(palette_raw) == palette_crc, 'palette header differs')
        payload_start = HEADER.size + index_size
        require(payload_start + payload_size == before.st_size, 'file length differs')
        for record in range(records):
            expert, matrix_index = divmod(record, 3)
            matrix = MATRICES[matrix_index]
            width, matrix_rows = (3072, 3584) if matrix == 'w2' else (3584, 3072)
            map_offset, map_count, first_block, block_count, unused = ENTRY.unpack_from(index, record * 24)
            require(map_offset == cursor and first_block == next_block and block_count == matrix_rows // 64
                    and unused == 0 and 1 <= map_count <= 256, 'matrix addressing differs')
            stream.seek(payload_start + map_offset)
            mapping = stream.read(map_count * 16)
            require(len(mapping) == map_count * 16 and all(ref < 66 or ref == 255 for ref in mapping), 'map references differ')
            map_lists[mapping] += 1
            by_matrix[matrix]['lists'][mapping] += 1
            by_matrix[matrix]['map_counts'][map_count] += 1
            for start in range(0, len(mapping), 16):
                group_map = mapping[start:start + 16]
                map_rows[group_map] += 1
                by_matrix[matrix]['rows'][group_map] += 1
            unique_ref_ids.update(set(mapping) - {255})
            absent += mapping.count(255)
            map_bytes += len(mapping)
            cursor += len(mapping)
            for block in range(block_count):
                entry_offset = (records + first_block + block) * 24
                offset, size, codec, checksum, unused = ENTRY.unpack_from(index, entry_offset)
                require(offset == cursor and 0 < size <= width * 64 and codec in (0, 1) and unused == 0,
                        'block addressing differs')
                cursor += size
                next_block += 1
                codecs[codec] += 1
                block_sizes[size] += 1
                row_counts[block_count] += 1
                compressed_bytes += size
                by_matrix[matrix]['compressed_bytes'] += size
                if expert != 0 or block != 0:
                    continue
                stream.seek(payload_start + offset)
                payload = stream.read(size)
                require(len(payload) == size, 'short sample block')
                count = width * 64
                expected_bytes = count // 2 + count // 32
                if codec == 0:
                    raw = payload
                else:
                    decoder = zlib.decompressobj()
                    raw = decoder.decompress(payload, expected_bytes + 1)
                    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
                            'sample compressed frame differs')
                require(len(raw) == expected_bytes, 'sample length differs')
                packed, selectors = raw[:count // 2], raw[count // 2:]
                restored = bytearray(count * 2)
                code_counts, selector_counts = Counter(), Counter(selectors)
                for coordinate in range(count):
                    code = (packed[coordinate // 2] >> (coordinate % 2 * 4)) & 15
                    selector = selectors[coordinate // 32]
                    require(selector < map_count, 'invalid sample selector')
                    reference = mapping[selector * 16 + code]
                    require(reference < 66, 'absent code-scale pair selected')
                    struct.pack_into('<H', restored, coordinate * 2, palette[reference])
                    code_counts[code] += 1
                require(zlib.crc32(restored) == checksum, 'decoded block differs')
                native = subprocess.check_output([str(reader), 'row', str(path), '0', matrix, '0'])
                row_bits = struct.unpack('<' + str(width) + 'H', restored[:width * 2])
                require(native == struct.pack('<' + str(width) + 'I', *(bits << 16 for bits in row_bits)),
                        'independent decoder differs from native row')
                first_codes = [(packed[coordinate // 2] >> (coordinate % 2 * 4)) & 15 for coordinate in range(32)]
                first_map = list(mapping[selectors[0] * 16:(selectors[0] + 1) * 16])
                references = [first_map[code] for code in first_codes]
                samples.append({'expert': expert, 'matrix': matrix, 'block': 0, 'rows': 64, 'width': width,
                                'matrix_record_file_offset': 64 + record * 24,
                                'matrix_record_hex': index[record * 24:(record + 1) * 24].hex(' '),
                                'matrix_record_fields': [map_offset, map_count, first_block, block_count, 0],
                                'block_record_file_offset': 64 + entry_offset,
                                'block_record_hex': index[entry_offset:entry_offset + 24].hex(' '),
                                'block_record_fields': [offset, size, codec, checksum, 0],
                                'block_file_offset': payload_start + offset, 'stored_bytes': size,
                                'stored_first32_hex': payload[:32].hex(' '), 'uncompressed_bytes': len(raw),
                                'uncompressed_packed_code_bytes': len(packed), 'uncompressed_selector_bytes': len(selectors),
                                'code_frequencies_in_block': dict(sorted(code_counts.items())),
                                'selector_frequencies_in_block': dict(sorted(selector_counts.items())),
                                'first32_group_selectors': list(selectors[:32]),
                                'first_group_selector': selectors[0],
                                'first_group_packed_hex': packed[:16].hex(' '), 'first_group_codes': first_codes,
                                'first_group_map_file_offset': payload_start + map_offset + selectors[0] * 16,
                                'first_group_map_refs': first_map, 'first_group_value_refs': references,
                                'first_group_values': [float_value(palette[reference]) for reference in references],
                                'native_float32_row_equal': True,
                                'decoded_block_sha256': hashlib.sha256(restored).hexdigest()})
    require(cursor == payload_size and next_block == blocks and map_bytes == evidence['group_mapping_bytes']
            and compressed_bytes == evidence['compressed_block_bytes'], 'complete payload accounting differs')
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns),
            'dataset changed during inspection')
    remaining = {'header': 64, 'matrix_index': records * 24, 'block_index': blocks * 24,
                 'group_reference_maps': map_bytes, 'compressed_codes_and_selectors': compressed_bytes}
    require(sum(remaining.values()) == before.st_size, 'remaining bytes do not sum to file size')
    report = {'gate': 'PASS', 'layer': layer, 'dataset_sha256': evidence['dataset_sha256'],
              'dataset_bytes': before.st_size, 'remaining_bytes': remaining,
              'compressed_payload_percent': 100 * compressed_bytes / before.st_size,
              'whole_dataset_sha256_checked': True, 'all_index_and_group_maps_inspected': True,
              'blocks_decompressed_this_inspection': len(samples), 'dataset_unchanged': True,
              'index_observations': {'codec_counts': dict(codecs), 'all_reserved_fields_zero': True,
                                     'matrix_identity_shape_block_count_derived_from_record_index': True,
                                     'all_offsets_contiguous_and_first_block_indices_derived': True,
                                     'stored_block_size_range': [min(block_sizes), max(block_sizes)],
                                     'reserved_field_bytes': (records + blocks) * 4 + 8,
                                     'codec_field_bytes_all_currently_one': blocks * 4},
              'map_observations': {'stored_matrix_map_lists': records, 'distinct_complete_map_lists': len(map_lists),
                                   'stored_16reference_rows': sum(map_rows.values()), 'distinct_16reference_rows': len(map_rows),
                                   'distinct_map_row_bytes_without_index': len(map_rows) * 16,
                                   'absent_pair_sentinel_bytes': absent, 'global_constant_ids_referenced': sorted(unique_ref_ids),
                                   'most_repeated_map_rows': [{'copies': count, 'refs': list(raw)} for raw, count in map_rows.most_common(5)],
                                   'by_matrix': {name: {'distinct_complete_lists': len(item['lists']),
                                                        'distinct_16reference_rows': len(item['rows']),
                                                        'map_count_distribution': dict(sorted(item['map_counts'].items())),
                                                        'compressed_block_bytes': item['compressed_bytes']}
                                                     for name, item in by_matrix.items()}},
              'samples': samples, 'script_sha256': digest(Path(__file__)),
              'limits': 'Read-only inspection of the existing layer1 store, not a new encoding. Full file hashed; all index and group maps inspected; only first64rowblock of each expert0 matrix decompressed here. Prior full native verification remains the whole-weight evidence. Compressed code and selector byte contributions are not separable inside a joint zlib stream. Repeated maps and fixed fields are candidates, not measured net compression gains.'}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('reader', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    inspect(args.directory, args.reader, args.output)