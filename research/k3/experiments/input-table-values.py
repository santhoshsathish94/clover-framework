#!/usr/bin/env python3
"""Census every BF16 input embedding value and verify its exact dictionary."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np


SOURCE = Path('/root/k3model/model-00094-of-000096.safetensors')
TENSOR = 'language_model.model.embed_tokens.weight'
EXPECTED_SHAPE = [163840, 7168]
CHUNK_ROWS = 256


def numeric_value(word):
    return struct.unpack('<f', struct.pack('<I', int(word) << 16))[0]


def category(word):
    exponent = (int(word) >> 7) & 255
    fraction = int(word) & 127
    if exponent == 255:
        return 'nan' if fraction else 'negative_infinity' if int(word) & 32768 else 'positive_infinity'
    if exponent == 0:
        return 'subnormal' if fraction else 'negative_zero' if int(word) & 32768 else 'positive_zero'
    return 'normal'


def histogram(payload):
    assert len(payload) % 2 == 0
    words = np.frombuffer(payload, dtype='<u2')
    return np.bincount(words, minlength=65536).astype(np.uint64)


def dictionary_for(counts):
    words = np.flatnonzero(counts).astype('<u2')
    identifiers = np.full(65536, -1, dtype=np.int32)
    identifiers[words] = np.arange(words.size, dtype=np.int32)
    return words, identifiers


def verify_dictionary(payload, words, identifiers):
    original = np.frombuffer(payload, dtype='<u2')
    encoded = identifiers[original]
    assert np.all(encoded >= 0)
    decoded = words[encoded].astype('<u2', copy=False)
    assert decoded.tobytes() == payload
    return np.bincount(encoded, minlength=words.size).astype(np.uint64)


def self_check():
    all_words = np.arange(65536, dtype='<u2')
    repeated = np.array([0, 0, 0x8000, 0x3f00, 0x3f00, 0xbf00, 1, 0x7f80, 0xff80, 0x7fc1], dtype='<u2')
    payload = all_words.tobytes() + repeated.tobytes()
    expected = Counter(struct.unpack('<%dH' % (len(payload) // 2), payload))
    counted = np.zeros(65536, dtype=np.uint64)
    for start in range(0, len(payload), 1022):
        counted += histogram(payload[start:start + 1022])
    assert all(int(counted[word]) == expected[word] for word in range(65536))
    dictionary, identifiers = dictionary_for(counted)
    assert dictionary.size == 65536 and (dictionary.size - 1).bit_length() == 16
    assert np.array_equal(verify_dictionary(payload, dictionary, identifiers), counted)
    sparse = histogram(repeated.tobytes())
    dictionary, identifiers = dictionary_for(sparse)
    assert np.array_equal(verify_dictionary(repeated.tobytes(), dictionary, identifiers), sparse[dictionary])
    assert category(0) == 'positive_zero' and category(0x8000) == 'negative_zero'
    assert category(1) == 'subnormal' and category(0x7fc1) == 'nan'
    assert category(0x7f80) == 'positive_infinity' and category(0xff80) == 'negative_infinity'
    assert numeric_value(0x3f00) == 0.5 and numeric_value(0xbf00) == -0.5
    rejected = False
    try:
        histogram(b'\x00')
    except AssertionError:
        rejected = True
    assert rejected
    print('PASS: all 65536 BF16 patterns, exact independent counts, chunk boundaries, dictionary roundtrips and special-value classification.')


def load_header(stream):
    stream.seek(0)
    length_bytes = stream.read(8)
    assert len(length_bytes) == 8
    size = struct.unpack('<Q', length_bytes)[0]
    raw = stream.read(size)
    assert len(raw) == size
    record = json.loads(raw)[TENSOR]
    assert record['dtype'] == 'BF16' and record['shape'] == EXPECTED_SHAPE
    start, end = record['data_offsets']
    assert end - start == math.prod(EXPECTED_SHAPE) * 2
    return record, 8 + size + start, hashlib.sha256(length_bytes + raw).hexdigest()


def describe(word, count, identifier):
    value = numeric_value(word)
    if math.isfinite(value):
        numerator, denominator = value.as_integer_ratio()
        numerator, denominator = str(numerator), str(denominator)
    else:
        numerator = denominator = ''
    return {'id': identifier, 'bf16_hex': f'{int(word):04x}', 'float32_hex': f'{int(word) << 16:08x}',
            'n_decimal': repr(value), 'n_hex_float': value.hex(), 'exact_numerator': numerator,
            'exact_denominator': denominator, 'category': category(word), 'count': int(count)}


def scan(output):
    output.mkdir()
    before = SOURCE.stat()
    counts = np.zeros(65536, dtype=np.uint64)
    digest = hashlib.sha256()
    bytes_read = 0
    rows, width = EXPECTED_SHAPE
    chunk_bytes = CHUNK_ROWS * width * 2
    started = time.perf_counter()
    with SOURCE.open('rb') as stream:
        record, start, header_hash = load_header(stream)
        end = start + rows * width * 2
        assert end <= before.st_size
        stream.seek(start)
        with (output / 'scan-chunks.tsv').open('x', newline='') as destination:
            writer = csv.writer(destination, delimiter='\t')
            writer.writerow(['first_token_row', 'rows', 'values', 'bytes', 'chunk_sha256'])
            for first_row in range(0, rows, CHUNK_ROWS):
                row_count = min(CHUNK_ROWS, rows - first_row)
                size = row_count * width * 2
                payload = stream.read(size)
                assert len(payload) == size
                counts += histogram(payload)
                digest.update(payload)
                bytes_read += size
                writer.writerow([first_row, row_count, row_count * width, size, hashlib.sha256(payload).hexdigest()])
                if (first_row + row_count) % 32768 == 0:
                    print(json.dumps({'pass': 'count', 'rows_read': first_row + row_count,
                                      'distinct_patterns_so_far': int(np.count_nonzero(counts))}), flush=True)
        assert stream.tell() == end and bytes_read == rows * width * 2
    assert int(counts.sum()) == rows * width
    dictionary, identifiers = dictionary_for(counts)
    count_seconds = time.perf_counter() - started
    with (output / 'dictionary.bf16').open('xb') as destination:
        destination.write(dictionary.tobytes())
    with (output / 'counts.u64le').open('xb') as destination:
        destination.write(counts.astype('<u8', copy=False).tobytes())
    records = [describe(word, counts[word], identifier) for identifier, word in enumerate(dictionary)]
    with (output / 'all-n.tsv').open('x', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(records)

    second_hash = hashlib.sha256()
    restored_counts = np.zeros(dictionary.size, dtype=np.uint64)
    verified_bytes = 0
    with SOURCE.open('rb') as stream:
        second_record, second_start, second_header_hash = load_header(stream)
        assert (second_record, second_start, second_header_hash) == (record, start, header_hash)
        stream.seek(start)
        for first_row in range(0, rows, CHUNK_ROWS):
            row_count = min(CHUNK_ROWS, rows - first_row)
            size = row_count * width * 2
            payload = stream.read(size)
            assert len(payload) == size
            restored_counts += verify_dictionary(payload, dictionary, identifiers)
            second_hash.update(payload)
            verified_bytes += size
            if (first_row + row_count) % 32768 == 0:
                print(json.dumps({'pass': 'verify_dictionary', 'rows_verified': first_row + row_count}), flush=True)
        assert stream.tell() == end
    assert verified_bytes == bytes_read and second_hash.digest() == digest.digest()
    assert np.array_equal(restored_counts, counts[dictionary])
    after = SOURCE.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    distinct = int(dictionary.size)
    identifier_bits = (distinct - 1).bit_length()
    category_counts = Counter()
    category_distinct = Counter()
    for item in records:
        category_counts[item['category']] += item['count']
        category_distinct[item['category']] += 1
    finite = [item for item in records if math.isfinite(float(item['n_decimal']))]
    frequency_order = sorted(records, key=lambda item: (-item['count'], item['id']))
    dictionary_bytes = distinct * 2
    ideal_index_bytes = (rows * width * identifier_bits + 7) // 8
    artifact_names = ('all-n.tsv', 'dictionary.bf16', 'counts.u64le', 'scan-chunks.tsv')
    report = {
        'gate': 'PASS', 'source': str(SOURCE), 'tensor': TENSOR, 'dtype': 'BF16', 'shape': EXPECTED_SHAPE,
        'meaning_of_n': 'An actual signed scalar stored in the input embedding table, not a fitted coefficient or extracted odd factor.',
        'absolute_offset': start, 'tensor_bytes': bytes_read, 'values_scanned': rows * width,
        'rows_scanned': rows, 'distinct_n_bit_patterns': distinct, 'fixed_width_identifier_bits': identifier_bits,
        'counts_by_category': dict(category_counts), 'distinct_by_category': dict(category_distinct),
        'minimum_finite': min(finite, key=lambda item: float(item['n_decimal'])) if finite else None,
        'maximum_finite': max(finite, key=lambda item: float(item['n_decimal'])) if finite else None,
        'most_frequent_values': frequency_order[:12],
        'input_payload_sha256': digest.hexdigest(), 'second_pass_payload_sha256': second_hash.hexdigest(),
        'header_with_length_sha256': header_hash, 'source_size_mtime_inode_unchanged': True,
        'dictionary_order': 'Ascending unsigned BF16 bit pattern; both signs of zero and distinct NaN payloads remain separate.',
        'verification': {'all_entries_dictionary_roundtrip_bit_exact': True, 'all_counts_match_second_pass': True,
                         'verified_values': rows * width, 'verified_bytes': verified_bytes},
        'fixed_width_storage_accounting': {
            'original_bytes': bytes_read, 'dictionary_bytes': dictionary_bytes,
            'ideal_bitpacked_identifiers_bytes': ideal_index_bytes,
            'dictionary_plus_identifiers_bytes': dictionary_bytes + ideal_index_bytes,
            'difference_from_original_bytes': dictionary_bytes + ideal_index_bytes - bytes_read,
            'limits': 'Identifier width is calculated; no bitpacked full-table file or model consumer was built. Headers/alignment excluded. Counts and scan logs are audit artifacts, not required dictionary payload.'},
        'count_pass_seconds': count_seconds, 'total_scan_verification_seconds': time.perf_counter() - started,
        'numpy_version': np.__version__, 'chunk_rows': CHUNK_ROWS, 'maximum_raw_chunk_bytes': chunk_bytes,
        'artifacts': {name: {'bytes': (output / name).stat().st_size,
                              'sha256': hashlib.sha256((output / name).read_bytes()).hexdigest()} for name in artifact_names},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Complete input embedding table only. Does not scan output head, trunk, experts or runtime activation vectors.',
        'limits': 'Counts actual stored bit patterns. Fixed-width identifier size is not a universal minimum for other encodings and does not account for positional-mask representations. No model or checkpoint mutation.'}
    with (output / 'results.json').open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps({key: report[key] for key in ('gate', 'shape', 'values_scanned', 'tensor_bytes',
                                                  'distinct_n_bit_patterns', 'fixed_width_identifier_bits',
                                                  'counts_by_category', 'minimum_finite', 'maximum_finite',
                                                  'fixed_width_storage_accounting', 'input_payload_sha256',
                                                  'total_scan_verification_seconds')}, indent=2), flush=True)
    print('results_sha256', hashlib.sha256((output / 'results.json').read_bytes()).hexdigest(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.output is None:
        parser.error('--output or --self-check is required')
    scan(args.output)


if __name__ == '__main__':
    main()