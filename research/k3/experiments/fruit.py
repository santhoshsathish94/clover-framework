#!/usr/bin/env python3
"""Census, build and read the exact output-head dataset using the seed container."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import time

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = Path('/root/k3model/model-00094-of-000096.safetensors')
TENSOR = 'language_model.lm_head.weight'
SHAPE = (163840, 7168)
BLOCK_ROWS = 16


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seed = load('fruit_container', HERE / 'seed.py')
values = load('fruit_value_descriptions', HERE / 'input-table-values.py')
FruitReader = seed.SeedReader


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    with path.open('x') as destination:
        json.dump(value, destination, indent=2)
        destination.write('\n')


def source_info(stream):
    stream.seek(0)
    length = stream.read(8)
    assert len(length) == 8
    size = struct.unpack('<Q', length)[0]
    raw = stream.read(size)
    assert len(raw) == size
    entry = json.loads(raw)[TENSOR]
    assert entry['dtype'] == 'BF16' and tuple(entry['shape']) == SHAPE
    assert entry['data_offsets'][1] - entry['data_offsets'][0] == SHAPE[0] * SHAPE[1] * 2
    return entry, 8 + size + entry['data_offsets'][0], hashlib.sha256(raw).hexdigest()


def signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def census(output):
    output.mkdir()
    before = signature(SOURCE)
    counts = np.zeros(65536, dtype=np.uint64)
    source_hash = hashlib.sha256()
    rows, width = SHAPE
    with SOURCE.open('rb') as stream, (output / 'scan-chunks.jsonl').open('x') as audit:
        record, offset, header_hash = source_info(stream)
        stream.seek(offset)
        for start in range(0, rows, 256):
            count = min(256, rows - start)
            raw = stream.read(count * width * 2)
            assert len(raw) == count * width * 2
            counts += values.histogram(raw)
            source_hash.update(raw)
            audit.write(json.dumps({'first_row': start, 'rows': count, 'bytes': len(raw),
                                    'sha256': hashlib.sha256(raw).hexdigest()}) + '\n')
        assert stream.tell() == offset + rows * width * 2
    assert signature(SOURCE) == before and int(counts.sum()) == rows * width
    dictionary, _ = values.dictionary_for(counts)
    descriptions = [values.describe(word, counts[word], identifier) for identifier, word in enumerate(dictionary)]
    with (output / 'all-n.tsv').open('x', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(descriptions[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(descriptions)
    with (output / 'dictionary.bf16').open('xb') as destination:
        destination.write(dictionary.tobytes())
    with (output / 'counts.u64le').open('xb') as destination:
        destination.write(counts.astype('<u8').tobytes())
    categories = {}
    for row in descriptions:
        categories[row['category']] = categories.get(row['category'], 0) + row['count']
    finite = [row for row in descriptions if row['category'] not in ('nan', 'negative_infinity', 'positive_infinity')]
    report = {'gate': 'PASS', 'dataset': 'fruit', 'source': str(SOURCE), 'tensor': TENSOR, 'shape': SHAPE,
              'dtype': 'BF16', 'offset': offset, 'values_scanned': rows * width, 'bytes_scanned': rows * width * 2,
              'distinct_n': int(dictionary.size), 'identifier_bits': (int(dictionary.size) - 1).bit_length(),
              'categories': categories, 'minimum': min(finite, key=lambda row: float(row['n_decimal'])),
              'maximum': max(finite, key=lambda row: float(row['n_decimal'])),
              'source_sha256': source_hash.hexdigest(), 'source_header_sha256': header_hash,
              'source_unchanged': True,
              'artifacts': {name: digest(output / name) for name in ('all-n.tsv', 'dictionary.bf16', 'counts.u64le', 'scan-chunks.jsonl')},
              'implementation_sha256': {name: digest(HERE / name) for name in ('fruit.py', 'seed.py', 'input-table-values.py')}}
    write_json(output / 'census.json', report)
    print(json.dumps(report, indent=2), flush=True)


def build(output):
    prior = json.loads((output / 'census.json').read_text())
    assert prior['gate'] == 'PASS' and prior['tensor'] == TENSOR
    for name, expected in prior['artifacts'].items():
        assert digest(output / name) == expected
    for name, expected in prior['implementation_sha256'].items():
        assert digest(HERE / name) == expected
    dictionary = np.frombuffer((output / 'dictionary.bf16').read_bytes(), dtype='<u2')
    pending = output / 'fruit.pending'
    final = output / 'fruit.bin'
    assert not final.exists() and not pending.exists()
    before = signature(SOURCE)
    rows, width = SHAPE
    started = time.perf_counter()
    with SOURCE.open('rb') as stream, (output / 'blocks.jsonl').open('x') as audit:
        record, offset, header_hash = source_info(stream)
        assert offset == prior['offset'] and header_hash == prior['source_header_sha256']
        stream.seek(offset)
        def blocks():
            for start in range(0, rows, BLOCK_ROWS):
                raw = stream.read(min(BLOCK_ROWS, rows - start) * width * 2)
                yield raw
        details = seed.create_seed(pending, rows, width, BLOCK_ROWS, dictionary, blocks(), audit)
        assert stream.tell() == offset + prior['bytes_scanned']
    assert details['source_sha256'] == prior['source_sha256']
    build_seconds = time.perf_counter() - started
    decoded_hash = hashlib.sha256()
    counts = np.zeros(65536, dtype=np.uint64)
    with FruitReader(pending) as reader, SOURCE.open('rb') as stream:
        assert source_info(stream) == (record, offset, header_hash)
        stream.seek(offset)
        for block in range(len(reader.entries)):
            raw = reader.read_block(block)
            assert stream.read(len(raw)) == raw
            decoded_hash.update(raw)
            counts += values.histogram(raw)
        assert stream.tell() == offset + prior['bytes_scanned']
        for row in (0, 15, 16, 17374, 40484, 81920, rows - 1):
            stream.seek(offset + row * width * 2)
            raw = stream.read(width * 2)
            assert reader.read_bf16(row) == raw
            assert reader.read_float32(row).tobytes() == (np.frombuffer(raw, dtype='<u2').astype('<u4') << 16).tobytes()
    assert decoded_hash.hexdigest() == prior['source_sha256']
    assert np.array_equal(counts, np.frombuffer((output / 'counts.u64le').read_bytes(), dtype='<u8'))
    assert signature(SOURCE) == before
    file_hash = digest(pending)
    pending.rename(final)
    size = final.stat().st_size
    report = {'gate': 'PASS', 'dataset': 'fruit', 'path': str(final), 'format': 'K3SEED1 container, output-head source identity',
              'shape': SHAPE, 'dtype': 'BF16', 'block_rows': BLOCK_ROWS,
              'dictionary_values': int(dictionary.size), 'identifier_bits': (int(dictionary.size) - 1).bit_length(),
              **details, 'fruit_bytes': size, 'original_bytes': prior['bytes_scanned'],
              'saved_bytes': prior['bytes_scanned'] - size, 'fraction_of_original': size / prior['bytes_scanned'],
              'payload_bytes': size - details['header_bytes'] - details['dictionary_bytes'] - details['index_bytes'],
              'fruit_sha256': file_hash, 'decoded_payload_sha256': decoded_hash.hexdigest(),
              'verified_values': rows * width, 'verified_rows': rows, 'all_reopened_bytes_match_source': True,
              'all_frequencies_match_census': True, 'source_unchanged': True,
              'source': str(SOURCE), 'tensor': TENSOR, 'offset': offset, 'source_header_sha256': header_hash,
              'build_seconds': build_seconds, 'build_and_verification_seconds': time.perf_counter() - started,
              'census_sha256': digest(output / 'census.json'), 'blocks_log_sha256': digest(output / 'blocks.jsonl'),
              'implementation_sha256': prior['implementation_sha256'],
              'limits': 'Full output head only. Smallest of six tested lossless layouts per16rowblock, not global optimum. Includes container header/dictionary/index/payload. No model integration or performance claim at this stage.'}
    write_json(output / 'results.json', report)
    print(json.dumps(report, indent=2), flush=True)
    print('report_sha256', digest(output / 'results.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('self-check', 'census', 'build', 'read'))
    parser.add_argument('path', type=Path, nargs='?')
    parser.add_argument('--row', type=int, default=17374)
    args = parser.parse_args()
    if args.command == 'self-check':
        values.self_check()
        seed.self_check()
    elif args.command == 'census':
        census(args.path)
    elif args.command == 'build':
        build(args.path)
    else:
        with FruitReader(args.path) as reader:
            raw = reader.read_bf16(args.row)
            print(json.dumps({'row': args.row, 'bf16_sha256': hashlib.sha256(raw).hexdigest(),
                              'first_eight_values': reader.read_float32(args.row)[:8].tolist()}, indent=2))


if __name__ == '__main__':
    main()