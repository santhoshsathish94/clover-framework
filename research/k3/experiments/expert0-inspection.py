#!/usr/bin/env python3
"""Read the complete layer1/expert0 payload and describe its stored contents."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


SOURCE = Path('/root/k3model/model-00002-of-000096.safetensors')
PREFIX = 'language_model.model.layers.1.block_sparse_moe.experts.0.'
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    before = SOURCE.stat()
    records, matrices = [], []
    expected = {PREFIX + matrix + '.' + kind for matrix in ('w1', 'w3', 'w2')
                for kind in ('weight_packed', 'weight_scale')}
    with SOURCE.open('rb') as stream:
        header_length = struct.unpack('<Q', stream.read(8))[0]
        raw_header = stream.read(header_length)
        header = json.loads(raw_header)
        assert {name for name in header if name.startswith(PREFIX)} == expected
        for matrix, role, rows, width in (('w1', 'gate', 3072, 3584), ('w3', 'up', 3072, 3584), ('w2', 'down', 3584, 3072)):
            payloads = {}
            for kind, shape in (('weight_packed', [rows, width // 2]), ('weight_scale', [rows, width // 32])):
                name = PREFIX + matrix + '.' + kind
                entry = header[name]
                start, end = entry['data_offsets']
                offset = 8 + header_length + start
                assert entry['dtype'] == 'U8' and entry['shape'] == shape
                assert end - start == math.prod(shape) and offset + end - start <= before.st_size
                stream.seek(offset)
                raw = stream.read(end - start)
                assert len(raw) == end - start
                records.append({'name': name, 'dtype': 'U8', 'stored_shape': shape, 'bytes': len(raw),
                                'absolute_offset': offset, 'sha256': hashlib.sha256(raw).hexdigest(),
                                'first16_bytes': list(raw[:16])})
                payloads[kind] = np.frombuffer(raw, dtype=np.uint8)
            packed, scales = payloads['weight_packed'], payloads['weight_scale']
            code_counts = np.bincount(packed & 15, minlength=16) + np.bincount(packed >> 4, minlength=16)
            scale_counts = np.bincount(scales, minlength=256)
            assert int(code_counts.sum()) == rows * width == scales.size * 32
            assert int(scale_counts.sum()) == rows * width // 32
            first_bytes = packed[:16].tolist()
            codes = [code for byte in first_bytes for code in (byte & 15, byte >> 4)]
            scale = int(scales[0])
            decoded = [math.ldexp(E2M1[code], scale - 127) if scale != 255 else math.copysign(0.0, E2M1[code])
                       for code in codes]
            decoded_bytes = struct.pack('<32f', *decoded)
            assert struct.unpack('<32f', decoded_bytes) == tuple(decoded)
            assert all((codes[2 * index] | (codes[2 * index + 1] << 4)) == byte for index, byte in enumerate(first_bytes))
            assert Counter(codes) == Counter([(byte >> shift) & 15 for byte in first_bytes for shift in (0, 4)])
            matrices.append({'matrix': matrix, 'role': role, 'decoded_shape': [rows, width],
                             'weight_count': rows * width, 'groups_of32': int(scales.size),
                             'code_bytes': int(packed.size), 'scale_bytes': int(scales.size),
                             'code_counts': {str(code): int(code_counts[code]) for code in range(16)},
                             'scale_counts': {str(scale_byte): int(count) for scale_byte, count in enumerate(scale_counts) if count},
                             'first_group': {'row': 0, 'group': 0, 'packed_bytes': first_bytes,
                                             'codes': codes, 'scale_byte': scale,
                                             'scale_multiplier': math.ldexp(1.0, scale - 127) if scale != 255 else 0.0,
                                             'decoded_weights': decoded,
                                             'float32_bits_hex': [f'{word:08x}' for word in struct.unpack('<32I', decoded_bytes)]}})
    after = SOURCE.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    total_bytes = sum(record['bytes'] for record in records)
    weights = sum(matrix['weight_count'] for matrix in matrices)
    assert total_bytes == 17547264 and weights == 33030144
    report = {'gate': 'PASS', 'source_file': str(SOURCE), 'layer': 1, 'expert': 0,
              'tensor_count': len(records), 'complete_expert_bytes_read': total_bytes,
              'weight_count': weights, 'groups_of32': sum(matrix['groups_of32'] for matrix in matrices),
              'code_bytes': sum(matrix['code_bytes'] for matrix in matrices),
              'scale_bytes': sum(matrix['scale_bytes'] for matrix in matrices),
              'hypothetical_float32_weight_bytes': weights * 4, 'source_unchanged': True,
              'header_sha256': hashlib.sha256(raw_header).hexdigest(),
              'matrices': matrices, 'stored_tensors': records,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Every code and scale byte in this one actual expert read and counted. First32 weights per matrix decoded as examples; not an expert forward or full decoded-value census.'}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('stored_tensors',)}, indent=2))
    print('result_sha256', hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()