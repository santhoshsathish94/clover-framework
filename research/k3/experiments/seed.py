#!/usr/bin/env python3
"""Build and read a self-contained, lossless input-vector seed dataset."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time
import zlib

import numpy as np


MAGIC = b'K3SEED1\0'
HEADER = struct.Struct('<8s8I3Q32s32s')
ENTRY = struct.Struct('<Q4I32s')
CODECS = ('packed_ids', 'zlib_packed_ids', 'zlib_id_planes', 'zlib_bf16',
          'zlib_bf16_byte_planes', 'zlib_bf16_bit_planes')
SOURCE = Path('/root/k3model/model-00094-of-000096.safetensors')
TENSOR = 'language_model.model.embed_tokens.weight'
CENSUS = Path('/opt/clover-k3/input-table-values-20260929-a')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pack_fields(values, bits, planes=False):
    fields = ((values[:, None].astype(np.uint32) >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
    if planes:
        fields = fields.T
    return np.packbits(fields.reshape(-1), bitorder='little').tobytes()


def unpack_fields(payload, count, bits, planes=False):
    require(len(payload) == (count * bits + 7) // 8, 'packed field length differs')
    unpacked = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder='little')
    require(not np.any(unpacked[count * bits:]), 'nonzero packed padding')
    fields = unpacked[:count * bits]
    fields = fields.reshape(bits, count).T if planes else fields.reshape(count, bits)
    return (fields.astype(np.uint32) << np.arange(bits, dtype=np.uint32)).sum(axis=1, dtype=np.uint32)


def inflate(payload, size):
    decoder = zlib.decompressobj()
    restored = decoder.decompress(payload, size + 1)
    require(len(restored) == size and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
            'compressed block length or framing differs')
    return restored


def candidates(raw, identifiers, bits):
    words = np.frombuffer(raw, dtype='<u2')
    ids = identifiers[words]
    require(np.all(ids >= 0), 'source value absent from dictionary')
    packed = pack_fields(ids, bits)
    planes = pack_fields(ids, bits, planes=True)
    return [packed, zlib.compress(packed, 6), zlib.compress(planes, 6), zlib.compress(raw, 6),
            zlib.compress(raw[0::2] + raw[1::2], 6),
            zlib.compress(pack_fields(words, 16, planes=True), 6)]


def decode_block(payload, codec, count, dictionary, bits):
    require(0 <= codec < len(CODECS), 'unknown seed codec')
    if codec <= 2:
        expected = (count * bits + 7) // 8
        packed = payload if codec == 0 else inflate(payload, expected)
        ids = unpack_fields(packed, count, bits, planes=codec == 2)
        require(np.all(ids < dictionary.size), 'dictionary identifier out of range')
        return dictionary[ids].astype('<u2', copy=False).tobytes()
    if codec == 3:
        return inflate(payload, count * 2)
    if codec == 4:
        planes = inflate(payload, count * 2)
        raw = bytearray(count * 2)
        raw[0::2], raw[1::2] = planes[:count], planes[count:]
        return bytes(raw)
    words = unpack_fields(inflate(payload, count * 2), count, 16, planes=True)
    return words.astype('<u2').tobytes()


class SeedReader:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open('rb')
        try:
            size = os.fstat(self.file.fileno()).st_size
            raw = self.file.read(HEADER.size)
            require(len(raw) == HEADER.size, 'short seed header')
            fields = HEADER.unpack(raw)
            (magic, version, self.rows, self.width, self.block_rows, dictionary_count,
             self.bits, blocks, reserved, dictionary_offset, index_offset, payload_offset,
             self.source_digest, metadata_digest) = fields
            require(magic == MAGIC and version == 1 and reserved == 0, 'unsupported seed header')
            require(self.rows > 0 and self.width > 0 and 1 <= self.block_rows <= 256, 'invalid seed shape')
            require(self.block_rows * self.width * 2 <= 64 * 1024 * 1024, 'seed block allocation limit')
            require(1 <= dictionary_count <= 65536 and self.bits == (dictionary_count - 1).bit_length(),
                    'invalid seed dictionary dimensions')
            require(blocks == (self.rows + self.block_rows - 1) // self.block_rows, 'invalid seed index count')
            require(dictionary_offset == HEADER.size and index_offset == dictionary_offset + dictionary_count * 2,
                    'invalid dictionary/index offsets')
            require(payload_offset == index_offset + blocks * ENTRY.size and payload_offset <= size,
                    'invalid payload offset')
            dictionary_bytes = self.file.read(dictionary_count * 2)
            index_bytes = self.file.read(blocks * ENTRY.size)
            require(hashlib.sha256(raw[:-32] + dictionary_bytes + index_bytes).digest() == metadata_digest,
                    'seed metadata checksum differs')
            self.dictionary = np.frombuffer(dictionary_bytes, dtype='<u2')
            require(self.dictionary.size == dictionary_count and np.unique(self.dictionary).size == dictionary_count,
                    'invalid seed dictionary')
            self.entries = list(ENTRY.iter_unpack(index_bytes))
            cursor = payload_offset
            for index, entry in enumerate(self.entries):
                offset, length, row_count, codec, crc, raw_hash = entry
                expected_rows = min(self.block_rows, self.rows - index * self.block_rows)
                require(offset == cursor and row_count == expected_rows, 'noncontiguous seed index')
                require(codec < len(CODECS) and length <= row_count * self.width * 2,
                        'invalid seed block size/codec')
                cursor += length
                require(cursor <= size, 'seed block outside file')
            require(cursor == size, 'seed trailing or missing bytes')
        except Exception:
            self.file.close()
            raise

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *ignored):
        self.close()

    def read_block(self, index):
        require(0 <= index < len(self.entries), 'seed block out of range')
        offset, length, rows, codec, crc, raw_hash = self.entries[index]
        self.file.seek(offset)
        payload = self.file.read(length)
        require(len(payload) == length and zlib.crc32(payload) == crc, 'seed payload checksum differs')
        raw = decode_block(payload, codec, rows * self.width, self.dictionary, self.bits)
        require(hashlib.sha256(raw).digest() == raw_hash, 'seed reconstructed block checksum differs')
        return raw

    def read_bf16(self, row):
        require(0 <= row < self.rows, 'seed row out of range')
        block = self.read_block(row // self.block_rows)
        start = (row % self.block_rows) * self.width * 2
        return block[start:start + self.width * 2]

    def read_float32(self, row):
        words = np.frombuffer(self.read_bf16(row), dtype='<u2').astype('<u4') << 16
        return words.view('<f4')


def write_header(stream, rows, width, block_rows, dictionary, entries, source_hash):
    dictionary_bytes = dictionary.astype('<u2', copy=False).tobytes()
    index_bytes = b''.join(ENTRY.pack(*entry) for entry in entries)
    index_offset = HEADER.size + len(dictionary_bytes)
    fields = (MAGIC, 1, rows, width, block_rows, dictionary.size, (dictionary.size - 1).bit_length(),
              len(entries), 0, HEADER.size, index_offset, index_offset + len(index_bytes), source_hash)
    header = HEADER.pack(*fields, bytes(32))
    checksum = hashlib.sha256(header[:-32] + dictionary_bytes + index_bytes).digest()
    stream.seek(0)
    stream.write(HEADER.pack(*fields, checksum))
    stream.write(dictionary_bytes)
    stream.write(index_bytes)


def create_seed(path, rows, width, block_rows, dictionary, blocks, audit):
    require(0 < rows and 0 < width and 1 <= block_rows <= 256, 'invalid build dimensions')
    identifiers = np.full(65536, -1, dtype=np.int32)
    identifiers[dictionary] = np.arange(dictionary.size)
    bits = (dictionary.size - 1).bit_length()
    count = (rows + block_rows - 1) // block_rows
    entries = []
    source_hash = hashlib.sha256()
    totals = [0] * len(CODECS)
    selected_counts = Counter()
    with Path(path).open('x+b') as stream:
        stream.write(bytes(HEADER.size + dictionary.size * 2 + count * ENTRY.size))
        for index, raw in enumerate(blocks):
            require(index < count, 'too many source blocks')
            row_count = min(block_rows, rows - index * block_rows)
            require(len(raw) == row_count * width * 2, 'source block length differs')
            options = candidates(raw, identifiers, bits)
            codec = min(range(len(options)), key=lambda option: (len(options[option]), option))
            payload = options[codec]
            require(decode_block(payload, codec, row_count * width, dictionary, bits) == raw,
                    'selected seed block does not roundtrip')
            raw_hash = hashlib.sha256(raw).digest()
            entries.append((stream.tell(), len(payload), row_count, codec, zlib.crc32(payload), raw_hash))
            stream.write(payload)
            source_hash.update(raw)
            selected_counts[CODECS[codec]] += 1
            for option, candidate in enumerate(options):
                totals[option] += len(candidate)
            audit.write(json.dumps({'block': index, 'first_row': index * block_rows, 'rows': row_count,
                                    'raw_bytes': len(raw), 'codec': CODECS[codec],
                                    'stored_bytes': len(payload), 'candidate_bytes': dict(zip(CODECS, map(len, options))),
                                    'raw_sha256': raw_hash.hex()}, separators=(',', ':')) + '\n')
            if (index + 1) % 2048 == 0:
                print(json.dumps({'built_rows': (index + 1) * block_rows,
                                  'seed_bytes_so_far': stream.tell()}), flush=True)
        require(len(entries) == count, 'missing source blocks')
        write_header(stream, rows, width, block_rows, dictionary, entries, source_hash.digest())
    return {'source_sha256': source_hash.hexdigest(), 'blocks': count,
            'selected_codecs': dict(selected_counts), 'fixed_codec_payload_totals': dict(zip(CODECS, totals)),
            'header_bytes': HEADER.size, 'dictionary_bytes': dictionary.size * 2, 'index_bytes': count * ENTRY.size}


def self_check():
    import io
    dictionary = np.arange(8191, dtype='<u2') * 7
    inverse = np.full(65536, -1, dtype=np.int32)
    inverse[dictionary] = np.arange(dictionary.size)
    ids = np.array([0, 1, 8190, 4000, 2, 8190, 0, 100, 3], dtype=np.uint32)
    raw = dictionary[ids].tobytes()
    for planes in (False, True):
        packed = pack_fields(ids, 13, planes)
        flat = ids if not planes else None
        expected_bits = ([int(value >> bit) & 1 for value in ids for bit in range(13)] if flat is not None
                         else [int(value >> bit) & 1 for bit in range(13) for value in ids])
        expected = sum(bit << offset for offset, bit in enumerate(expected_bits)).to_bytes((len(expected_bits) + 7) // 8, 'little')
        require(packed == expected and np.array_equal(unpack_fields(packed, ids.size, 13, planes), ids), 'independent packed control failed')
    for codec, payload in enumerate(candidates(raw, inverse, 13)):
        require(decode_block(payload, codec, ids.size, dictionary, 13) == raw, 'codec control failed')
    all_words = np.arange(65536, dtype='<u2')
    all_ids = np.arange(65536, dtype=np.int32)
    for codec, payload in enumerate(candidates(all_words.tobytes(), all_ids, 16)):
        require(decode_block(payload, codec, 65536, all_words, 16) == all_words.tobytes(), 'BF16 pattern control failed')
    with tempfile.TemporaryDirectory(prefix='seed-control-') as directory:
        path = Path(directory) / 'seed.bin'
        rows, width = 5, 9
        complete = np.tile(dictionary[ids], rows).tobytes()
        chunks = [complete[offset:offset + 2 * width * 2] for offset in range(0, len(complete), 2 * width * 2)]
        create_seed(path, rows, width, 2, dictionary, iter(chunks), io.StringIO())
        with SeedReader(path) as reader:
            for row in range(rows):
                expected = complete[row * width * 2:(row + 1) * width * 2]
                require(reader.read_bf16(row) == expected, 'on-demand row control failed')
                expected_float = np.frombuffer(expected, dtype='<u2').astype('<u4') << 16
                require(reader.read_float32(row).tobytes() == expected_float.tobytes(), 'float32 widening control failed')
            rejected = False
            try:
                reader.read_bf16(rows)
            except ValueError:
                rejected = True
            require(rejected, 'out-of-range row accepted')
        original = path.read_bytes()
        for label, damaged in (('truncated', original[:-1]), ('metadata', original[:HEADER.size] + bytes([original[HEADER.size] ^ 1]) + original[HEADER.size + 1:]),
                               ('payload', original[:-1] + bytes([original[-1] ^ 1]))):
            corrupt = Path(directory) / (label + '.bin')
            corrupt.write_bytes(damaged)
            rejected = False
            try:
                with SeedReader(corrupt) as reader:
                    reader.read_block(len(reader.entries) - 1)
            except ValueError:
                rejected = True
            require(rejected, 'corrupt seed accepted: ' + label)
    print('PASS: six codecs, independent 13-bit packing, all BF16 patterns, random row reads, float32 widening and corrupt-input controls.')


def source_info(stream):
    stream.seek(0)
    size = struct.unpack('<Q', stream.read(8))[0]
    header = stream.read(size)
    record = json.loads(header)[TENSOR]
    require(record['dtype'] == 'BF16' and record['shape'] == [163840, 7168], 'input source shape/dtype changed')
    return record, 8 + size + record['data_offsets'][0], hashlib.sha256(header).hexdigest()


def build(output):
    output.mkdir()
    before = SOURCE.stat()
    census = json.loads((CENSUS / 'results.json').read_text())
    dictionary_bytes = (CENSUS / 'dictionary.bf16').read_bytes()
    require(hashlib.sha256(dictionary_bytes).hexdigest() == census['artifacts']['dictionary.bf16']['sha256'], 'census dictionary hash differs')
    dictionary = np.frombuffer(dictionary_bytes, dtype='<u2')
    require(dictionary.size == 6658 and np.unique(dictionary).size == dictionary.size, 'census dictionary differs')
    pending, final = output / 'seed.pending', output / 'seed.bin'
    started = time.perf_counter()
    with SOURCE.open('rb') as stream, (output / 'blocks.jsonl').open('x') as audit:
        record, start, header_hash = source_info(stream)
        rows, width = record['shape']
        block_rows = 16
        stream.seek(start)
        def blocks():
            for first_row in range(0, rows, block_rows):
                size = min(block_rows, rows - first_row) * width * 2
                raw = stream.read(size)
                require(len(raw) == size, 'short input source read')
                yield raw
        details = create_seed(pending, rows, width, block_rows, dictionary, blocks(), audit)
        require(stream.tell() == start + rows * width * 2, 'input table coverage differs')
    require(details['source_sha256'] == census['input_payload_sha256'], 'full source hash differs from census')
    build_seconds = time.perf_counter() - started
    decoded_hash = hashlib.sha256()
    seed_hash = hashlib.sha256()
    recovered_counts = np.zeros(65536, dtype=np.uint64)
    with SeedReader(pending) as reader, SOURCE.open('rb') as stream:
        repeated, repeated_start, repeated_header_hash = source_info(stream)
        require((repeated, repeated_start, repeated_header_hash) == (record, start, header_hash), 'source header changed')
        stream.seek(start)
        for index in range(len(reader.entries)):
            raw = reader.read_block(index)
            require(stream.read(len(raw)) == raw, 'reopened seed differs from original source bytes')
            decoded_hash.update(raw)
            recovered_counts += np.bincount(np.frombuffer(raw, dtype='<u2'), minlength=65536).astype(np.uint64)
        require(stream.tell() == start + rows * width * 2, 'reopened verification coverage differs')
        for row in (0, 15, 16, 1008, 10484, rows // 2, rows - 1):
            stream.seek(start + row * width * 2)
            raw = stream.read(width * 2)
            require(reader.read_bf16(row) == raw, 'random row reconstruction differs')
            expected = (np.frombuffer(raw, dtype='<u2').astype('<u4') << 16).tobytes()
            require(reader.read_float32(row).tobytes() == expected, 'on-demand float32 vector differs')
    expected_counts = np.frombuffer((CENSUS / 'counts.u64le').read_bytes(), dtype='<u8')
    require(np.array_equal(expected_counts, recovered_counts), 'seed frequency census differs')
    require(decoded_hash.hexdigest() == census['input_payload_sha256'], 'reconstructed full payload hash differs')
    after = SOURCE.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'source file changed')
    with pending.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            seed_hash.update(chunk)
    require(not final.exists(), 'seed already exists')
    pending.rename(final)
    actual_bytes = final.stat().st_size
    report = {
        'dataset': 'seed', 'gate': 'PASS', 'path': str(final), 'format_version': 1,
        'shape': [rows, width], 'source_dtype': 'BF16', 'dictionary_values': dictionary.size,
        'identifier_bits': 13, 'block_rows': block_rows, 'codecs': CODECS, **details,
        'seed_bytes': actual_bytes, 'original_bytes': rows * width * 2,
        'saved_bytes': rows * width * 2 - actual_bytes, 'fraction_of_original': actual_bytes / (rows * width * 2),
        'payload_bytes': actual_bytes - details['header_bytes'] - details['dictionary_bytes'] - details['index_bytes'],
        'seed_sha256': seed_hash.hexdigest(), 'decoded_payload_sha256': decoded_hash.hexdigest(),
        'verified_values': rows * width, 'verified_rows': rows, 'full_reopened_roundtrip_bit_exact': True,
        'random_access_rows_verified': [0, 15, 16, 1008, 10484, rows // 2, rows - 1],
        'all_frequencies_match_census': True, 'source_file_unchanged': True,
        'source_file': str(SOURCE), 'source_tensor': TENSOR, 'source_offset': start,
        'source_header_sha256': header_hash, 'build_seconds': build_seconds,
        'build_and_verification_seconds': time.perf_counter() - started,
        'numpy_version': np.__version__, 'zlib_version': zlib.ZLIB_VERSION,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'blocks_log_sha256': hashlib.sha256((output / 'blocks.jsonl').read_bytes()).hexdigest(),
        'limits': 'Smallest tested payload among six exact layouts per fixed 16-row block, not a proof of global optimality. File size includes header, dictionary, index, checksums and payload. No model integration or runtime/RAM improvement claim; source retained. Reader needs only seed.bin plus codec software, not the checkpoint.'}
    with (output / 'results.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2), flush=True)
    print('report_sha256', hashlib.sha256((output / 'results.json').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('self-check')
    builder = commands.add_parser('build')
    builder.add_argument('directory', type=Path)
    reader = commands.add_parser('read')
    reader.add_argument('dataset', type=Path)
    reader.add_argument('row', type=int)
    args = parser.parse_args()
    if args.command == 'self-check':
        self_check()
    elif args.command == 'build':
        build(args.directory)
    else:
        with SeedReader(args.dataset) as seed:
            raw = seed.read_bf16(args.row)
            vector = seed.read_float32(args.row)
            print(json.dumps({'dataset': str(args.dataset), 'row': args.row, 'width': seed.width,
                              'bf16_sha256': hashlib.sha256(raw).hexdigest(), 'first_eight_values': vector[:8].tolist()}, indent=2))


if __name__ == '__main__':
    main()