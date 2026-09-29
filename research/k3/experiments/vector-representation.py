#!/usr/bin/env python3
"""Measure lossless representations of actual Clover 7168-component vectors."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import zlib


WIDTH = 7168
VECTOR_BYTES = WIDTH * 4


def dyadic(word):
    exponent_bits = (word >> 23) & 255
    significand = word & 0x7FFFFF
    exponent = -149
    if exponent_bits:
        significand |= 1 << 23
        exponent = exponent_bits - 150
    if not significand:
        return 0, 0
    shift = (significand & -significand).bit_length() - 1
    significand >>= shift
    exponent += shift
    if word >> 31:
        significand = -significand
    return significand, exponent


def encode_dyadic(payload, block_size):
    words = struct.unpack('<%dI' % WIDTH, payload)
    encoded = bytearray(struct.pack('<4sHH', b'DYA1', WIDTH, block_size))
    for start in range(0, WIDTH, block_size):
        block = words[start:start + block_size]
        if any(word == 0x80000000 or ((word >> 23) & 255) == 255 for word in block):
            encoded.extend(struct.pack('<hH', 0, 0))
            encoded.extend(struct.pack('<%dI' % len(block), *block))
            continue
        parts = [dyadic(word) for word in block]
        exponent = min((power for number, power in parts if number), default=0)
        integers = [number << (power - exponent) if number else 0 for number, power in parts]
        bits = max((number if number >= 0 else ~number).bit_length() + 1 for number in integers)
        mask = (1 << bits) - 1
        packed = sum((number & mask) << (index * bits) for index, number in enumerate(integers))
        encoded.extend(struct.pack('<hH', exponent, bits))
        encoded.extend(packed.to_bytes((len(block) * bits + 7) // 8, 'little'))
    return bytes(encoded)


def decode_dyadic(encoded):
    magic, width, block_size = struct.unpack_from('<4sHH', encoded)
    assert magic == b'DYA1' and width == WIDTH and block_size > 0
    offset = 8
    decoded = bytearray()
    for start in range(0, width, block_size):
        count = min(block_size, width - start)
        exponent, bits = struct.unpack_from('<hH', encoded, offset)
        offset += 4
        if bits == 0:
            decoded.extend(encoded[offset:offset + count * 4])
            offset += count * 4
            continue
        size = (count * bits + 7) // 8
        packed = int.from_bytes(encoded[offset:offset + size], 'little')
        offset += size
        mask = (1 << bits) - 1
        values = []
        for index in range(count):
            number = (packed >> (index * bits)) & mask
            if number & (1 << (bits - 1)):
                number -= 1 << bits
            values.append(math.ldexp(float(number), exponent))
        decoded.extend(struct.pack('<%df' % count, *values))
    assert offset == len(encoded)
    return bytes(decoded)


def measure(payload, kind, layer, position):
    assert len(payload) == VECTOR_BYTES
    words = struct.unpack('<%dI' % WIDTH, payload)
    values = struct.unpack('<%df' % WIDTH, payload)
    assert all(math.isfinite(value) for value in values)
    shuffled = b''.join(payload[lane::4] for lane in range(4))
    compressed = zlib.compress(payload, level=6)
    compressed_shuffled = zlib.compress(shuffled, level=6)
    assert zlib.decompress(compressed) == payload
    planes = zlib.decompress(compressed_shuffled)
    restored = bytearray(VECTOR_BYTES)
    for lane in range(4):
        restored[lane::4] = planes[lane * WIDTH:(lane + 1) * WIDTH]
    assert restored == payload
    bf16_exact = sum((word & 65535) == 0 for word in words)
    if bf16_exact == WIDTH:
        bf16 = struct.pack('<%dH' % WIDTH, *(word >> 16 for word in words))
        assert struct.pack('<%dI' % WIDTH, *(word << 16 for word in struct.unpack('<%dH' % WIDTH, bf16))) == payload
    byte_counts = {'raw_f32': VECTOR_BYTES, 'zlib': len(compressed) + 4, 'byte_planes_zlib': len(compressed_shuffled) + 4}
    for block_size in (32, 128):
        encoded = encode_dyadic(payload, block_size)
        assert decode_dyadic(encoded) == payload
        byte_counts['dyadic_%d' % block_size] = len(encoded)
    return {
        'kind': kind, 'layer': layer, 'position': position,
        'sha256': hashlib.sha256(payload).hexdigest(),
        'minimum': min(values), 'maximum': max(values),
        'abs_ge_one': sum(abs(value) >= 1 for value in values),
        'zero_components': sum((word & 0x7FFFFFFF) == 0 for word in words),
        'distinct_bit_patterns': len(set(words)),
        'bf16_exact_components': bf16_exact,
        'bf16_whole_vector_exact': bf16_exact == WIDTH,
        'sizes_bytes': byte_counts,
        'roundtrips_exact': True,
    }


def summarize(rows):
    return {
        'vectors': len(rows),
        'components': len(rows) * WIDTH,
        'bf16_exact_components': sum(row['bf16_exact_components'] for row in rows),
        'bf16_whole_vectors_exact': sum(row['bf16_whole_vector_exact'] for row in rows),
        'zero_components': sum(row['zero_components'] for row in rows),
        'abs_ge_one': sum(row['abs_ge_one'] for row in rows),
        'minimum': min(row['minimum'] for row in rows),
        'maximum': max(row['maximum'] for row in rows),
        'distinct_patterns_per_vector_median': statistics.median(row['distinct_bit_patterns'] for row in rows),
        'sizes_bytes': {
            name: {
                'total': sum(row['sizes_bytes'][name] for row in rows),
                'ratio_to_f32': sum(row['sizes_bytes'][name] for row in rows) / (len(rows) * VECTOR_BYTES),
                'minimum_per_vector': min(row['sizes_bytes'][name] for row in rows),
                'maximum_per_vector': max(row['sizes_bytes'][name] for row in rows),
            }
            for name in rows[0]['sizes_bytes']
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layers', type=Path)
    parser.add_argument('residual', type=Path)
    parser.add_argument('--positions', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.positions > 0
    layer_data = args.layers.read_bytes()
    residual_data = args.residual.read_bytes()
    assert len(layer_data) == 93 * args.positions * VECTOR_BYTES
    assert len(residual_data) == args.positions * VECTOR_BYTES
    rows = []
    for index in range(93 * args.positions):
        layer, position = divmod(index, args.positions)
        payload = layer_data[index * VECTOR_BYTES:(index + 1) * VECTOR_BYTES]
        rows.append(measure(payload, 'pre_attention_aggregate', layer, position))
    for position in range(args.positions):
        payload = residual_data[position * VECTOR_BYTES:(position + 1) * VECTOR_BYTES]
        rows.append(measure(payload, 'final_residual', 92, position))
    groups = {
        'embedding': [row for row in rows if row['kind'] == 'pre_attention_aggregate' and row['layer'] == 0],
        'later_aggregates': [row for row in rows if row['kind'] == 'pre_attention_aggregate' and row['layer'] > 0],
        'final_residual': [row for row in rows if row['kind'] == 'final_residual'],
    }
    summary = {name: summarize(group) for name, group in groups.items()}
    report = {
        'width': WIDTH, 'positions': args.positions,
        'zlib_runtime_version': zlib.ZLIB_RUNTIME_VERSION,
        'layers_sha256': hashlib.sha256(layer_data).hexdigest(),
        'residual_sha256': hashlib.sha256(residual_data).hexdigest(),
        'all_roundtrips_exact': all(row['roundtrips_exact'] for row in rows),
        'scope': 'Actual captured vectors only; per-vector encoding; storage sizes include format headers. No model integration or throughput claim.',
        'summary': summary, 'vectors': rows,
    }
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(summary, indent=2))
    print('All %d captured vectors reconstructed byte-for-byte by every lossless codec.' % len(rows))


if __name__ == '__main__':
    main()