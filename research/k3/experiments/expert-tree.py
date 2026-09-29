#!/usr/bin/env python3
"""Hierarchical, lossless storage and on-demand equations for one routed expert."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import zlib

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('tree_scale_codec', HERE / 'expert-scale-exceptions.py')
SCALES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCALES)
HEADER = struct.Struct('<8sIIQQ32s')
MAGIC = b'K3TREE1\0'
CODECS = ('packed', 'zlib_packed', 'zlib_four_planes')
SOURCE = Path('/root/k3model/model-00002-of-000096.safetensors')
PREFIX = 'language_model.model.layers.1.block_sparse_moe.experts.0.'
SHAPES = {'w1': (3072, 3584), 'w3': (3072, 3584), 'w2': (3584, 3072)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inflate(payload, expected):
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(payload, expected + 1)
    require(len(decoded) == expected and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
            'invalid compressed length or framing')
    return decoded


def codes_from_packed(packed):
    words = np.frombuffer(packed, dtype=np.uint8)
    codes = np.empty(words.size * 2, dtype=np.uint8)
    codes[::2], codes[1::2] = words & 15, words >> 4
    return codes


def code_planes(packed):
    codes = codes_from_packed(packed)
    return np.packbits(((codes[None, :] >> np.arange(4, dtype=np.uint8)[:, None]) & 1),
                       axis=1, bitorder='little').tobytes()


def decode_codes(payload, codec, width):
    require(codec in range(3), 'invalid code codec')
    if codec == 0:
        require(len(payload) == width // 2, 'invalid packed length')
        return payload
    decoded = inflate(payload, width // 2)
    if codec == 1:
        return decoded
    planes = np.unpackbits(np.frombuffer(decoded, dtype=np.uint8).reshape(4, width // 8), axis=1, bitorder='little')
    codes = np.sum(planes << np.arange(4, dtype=np.uint8)[:, None], axis=0, dtype=np.uint8)
    return (codes[::2] | (codes[1::2] << 4)).tobytes()


def equation_weights(packed, scales):
    codes = codes_from_packed(packed)
    require(len(scales) * 32 == codes.size, 'scale group shape differs')
    exponent = (codes >> 1) & 3
    mantissa = codes & 1
    magnitude = np.where(exponent == 0, 0.5 * mantissa,
                         (1.0 + 0.5 * mantissa) * np.exp2(exponent.astype(np.int32) - 1)).astype(np.float32)
    signed = np.copysign(magnitude, np.where(codes & 8, -1.0, 1.0).astype(np.float32))
    scale = np.repeat(np.frombuffer(scales, dtype=np.uint8), 32)
    with np.errstate(over='ignore', under='ignore', invalid='raise'):
        weights = np.ldexp(signed, scale.astype(np.int32) - 127)
    weights[scale == 255] = np.copysign(np.float32(0), signed[scale == 255])
    return weights.astype('<f4')


def ordered_projection(packed, scales, vector):
    vector = np.asarray(vector, dtype=np.float32)
    weights = equation_weights(packed, scales)
    require(vector.ndim == 1 and vector.size == weights.size and vector.size % 16 == 0, 'projection input shape differs')
    lanes = np.zeros(16, dtype=np.float64)
    for start in range(0, vector.size, 16):
        product = weights[start:start + 16].astype(np.float64) * vector[start:start + 16].astype(np.float64)
        lanes = lanes + product
    partial = (lanes[0:4] + lanes[8:12]) + (lanes[4:8] + lanes[12:16])
    return np.float32((partial[0] + partial[2]) + (partial[1] + partial[3]))


def encode_row(packed, scales):
    options = (packed, zlib.compress(packed, 6), zlib.compress(code_planes(packed), 6))
    codec = min(range(3), key=lambda item: (len(options[item]), item))
    scale_payload, _, _ = SCALES.encode(scales, 'adaptive')
    require(SCALES.decode(scale_payload, len(scales), 'adaptive') == scales, 'scale encoding mismatch')
    require(decode_codes(options[codec], codec, len(packed) * 2) == packed, 'code encoding mismatch')
    return options[codec] + scale_payload, codec, len(scale_payload), [len(option) for option in options]


def write_tree(path, matrices, audit):
    root = {'version': 1, 'layer': 1, 'expert': 0, 'group_width': 32,
            'row_fields': ['offset', 'bytes', 'code_codec', 'scale_bytes', 'decoded_sha256'],
            'code_codecs': CODECS, 'scale_codec': 'adaptive_usual_offset_exceptions',
            'equation': 'X(w2,SiTU(X(w1,z),X(w3,z)))', 'matrices': []}
    payload = bytearray()
    choices, alternative_totals = Counter(), Counter()
    scale_total = 0
    for name, shape, packed, scales in matrices:
        rows, width = shape
        require(width % 32 == 0 and 0 < width <= 3584 and 0 < rows <= 3584, 'tree matrix shape invalid')
        require(len(packed) == rows * width // 2 and len(scales) == rows * width // 32, 'matrix payload length differs')
        node = {'name': name, 'shape': list(shape), 'groups_per_row': width // 32,
                'packed_sha256': sha(packed), 'scales_sha256': sha(scales), 'rows': []}
        for row in range(rows):
            code_row = packed[row * (width // 2):(row + 1) * (width // 2)]
            scale_row = scales[row * (width // 32):(row + 1) * (width // 32)]
            encoded, codec, scale_size, sizes = encode_row(code_row, scale_row)
            node['rows'].append([len(payload), len(encoded), codec, scale_size, sha(code_row + scale_row)])
            payload.extend(encoded)
            choices[CODECS[codec]] += 1
            alternative_totals.update(dict(zip(CODECS, sizes)))
            scale_total += scale_size
            audit.write(json.dumps({'matrix': name, 'row': row, 'code_codec': CODECS[codec],
                                    'code_candidate_bytes': dict(zip(CODECS, sizes)), 'scale_bytes': scale_size,
                                    'stored_bytes': len(encoded), 'original_bytes': len(code_row) + len(scale_row)}) + '\n')
        root['matrices'].append(node)
    root_bytes = json.dumps(root, separators=(',', ':')).encode('ascii')
    compressed_root = zlib.compress(root_bytes, 9)
    with path.open('xb') as stream:
        stream.write(HEADER.pack(MAGIC, 1, len(root_bytes), len(compressed_root), len(payload), hashlib.sha256(root_bytes).digest()))
        stream.write(compressed_root)
        stream.write(payload)
    return {'header_bytes': HEADER.size, 'compressed_root_bytes': len(compressed_root), 'root_json_bytes': len(root_bytes),
            'payload_bytes': len(payload), 'scale_descriptor_bytes': scale_total,
            'code_payload_bytes': len(payload) - scale_total, 'selected_codecs': dict(choices),
            'alternative_code_payload_totals': dict(alternative_totals)}, root


class ExpertTree:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open('rb')
        try:
            raw = self.file.read(HEADER.size)
            require(len(raw) == HEADER.size, 'short tree header')
            magic, version, root_size, compressed_size, payload_size, expected_hash = HEADER.unpack(raw)
            require(magic == MAGIC and version == 1, 'unsupported tree format')
            require(0 < root_size <= 16 * 1024 * 1024 and 0 < compressed_size <= 16 * 1024 * 1024,
                    'tree root allocation bound')
            require(HEADER.size + compressed_size + payload_size == self.path.stat().st_size, 'tree length differs')
            decoded = inflate(self.file.read(compressed_size), root_size)
            require(hashlib.sha256(decoded).digest() == expected_hash, 'tree root checksum differs')
            self.root = json.loads(decoded)
            require(self.root['version'] == 1 and self.root['group_width'] == 32, 'unsupported tree root')
            require(self.root['row_fields'] == ['offset', 'bytes', 'code_codec', 'scale_bytes', 'decoded_sha256']
                    and tuple(self.root['code_codecs']) == CODECS
                    and self.root['scale_codec'] == 'adaptive_usual_offset_exceptions', 'unsupported tree node schema')
            require([node['name'] for node in self.root['matrices']] == ['w1', 'w3', 'w2'], 'matrix branches differ')
            self.matrices = {node['name']: node for node in self.root['matrices']}
            cursor = 0
            for node in self.matrices.values():
                rows, width = node['shape']
                require(0 < rows <= 3584 and 0 < width <= 3584 and width % 32 == 0 and node['groups_per_row'] == width // 32,
                        'invalid tree matrix dimensions')
                require(len(node['rows']) == rows, 'missing tree row nodes')
                for entry in node['rows']:
                    require(len(entry) == 5, 'invalid row descriptor')
                    offset, size, codec, scale_size, checksum = entry
                    require(offset == cursor and codec in range(3) and 0 < scale_size <= 128
                            and scale_size < size <= width // 2 + 128 and len(checksum) == 64, 'invalid tree row descriptor')
                    cursor += size
            require(cursor == payload_size, 'row nodes do not cover tree payload')
            self.payload_start = HEADER.size + compressed_size
        except Exception:
            self.file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *ignored):
        self.file.close()

    def read_row(self, matrix, row):
        require(matrix in self.matrices, 'unknown tree matrix')
        node = self.matrices[matrix]
        require(0 <= row < node['shape'][0], 'tree row out of range')
        offset, size, codec, scale_size, checksum = node['rows'][row]
        self.file.seek(self.payload_start + offset)
        payload = self.file.read(size)
        require(len(payload) == size, 'short tree row')
        packed = decode_codes(payload[:-scale_size], codec, node['shape'][1])
        try:
            scales = SCALES.decode(payload[-scale_size:], node['groups_per_row'], 'adaptive')
        except (AssertionError, ValueError, IndexError) as error:
            raise ValueError('invalid tree scale descriptor') from error
        require(sha(packed + scales) == checksum, 'tree reconstructed row checksum differs')
        return packed, scales

    def group(self, matrix, row, group):
        require(matrix in self.matrices and 0 <= group < self.matrices[matrix]['groups_per_row'], 'tree group out of range')
        packed, scales = self.read_row(matrix, row)
        code_bytes, scale = packed[group * 16:(group + 1) * 16], scales[group]
        masks = list(struct.unpack('<4I', code_planes(code_bytes)))
        codes = [sum(((mask >> position) & 1) << bit for bit, mask in enumerate(masks)) for position in range(32)]
        require(bytes(codes[2 * index] | codes[2 * index + 1] << 4 for index in range(16)) == code_bytes,
                'tree mask reconstruction differs')
        weights = equation_weights(code_bytes, bytes([scale]))
        return {'matrix': matrix, 'row': row, 'group': group, 'scale': scale, 'masks': masks,
                'codes': codes, 'weights': weights.tolist(),
                'float32_bits_hex': [f'{word:08x}' for word in weights.view('<u4')]}

    def weight(self, matrix, row, coordinate):
        require(matrix in self.matrices and 0 <= coordinate < self.matrices[matrix]['shape'][1], 'tree coordinate out of range')
        group = self.group(matrix, row, coordinate // 32)
        return group['weights'][coordinate % 32]

    def project_row(self, matrix, row, vector):
        packed, scales = self.read_row(matrix, row)
        return ordered_projection(packed, scales, vector)


def self_check():
    import io
    require(SCALES.controls() == 804, 'prior scale controls differ')
    example = bytes(range(16)) * 4
    for count in (16, 64):
        raw = example[:count]
        for codec, payload in enumerate((raw, zlib.compress(raw), zlib.compress(code_planes(raw)))):
            require(decode_codes(payload, codec, len(raw) * 2) == raw, 'code controls failed')
    table = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
    packed = bytes([code | code << 4 for code in range(16)])
    codes = codes_from_packed(packed)
    for scale in range(256):
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            factor = np.float32(0 if scale == 255 else 2.0 ** (scale - 127))
            expected = table[codes] * factor
        require(equation_weights(packed, bytes([scale])).tobytes() == expected.tobytes(), '4096 decode pair control differs')
    matrices = [(name, (32, 32), bytes(range(256)) * 2, bytes([120, 121]) * 16) for name in ('w1', 'w3', 'w2')]
    with tempfile.TemporaryDirectory(prefix='expert-tree-controls-') as temporary:
        path = Path(temporary) / 'expert.tree'
        write_tree(path, matrices, io.StringIO())
        with ExpertTree(path) as tree:
            for name, shape, packed_rows, scales in matrices:
                for row in (0, 16, 31):
                    require(tree.read_row(name, row) == (packed_rows[row * 16:(row + 1) * 16], scales[row:row + 1]), 'tree row differs')
                    group = tree.group(name, row, 0)
                    require(len(group['masks']) == 4 and len(group['weights']) == 32, 'tree group shape differs')
                    vector = np.arange(32, dtype=np.float32) / 64
                    weights = table[codes_from_packed(packed_rows[row * 16:(row + 1) * 16])] * np.float32(2.0 ** (scales[row] - 127))
                    lanes = [0.0] * 16
                    for start in range(0, 32, 16):
                        for lane in range(16):
                            lanes[lane] += float(weights[start + lane]) * float(vector[start + lane])
                    parts = [(lanes[index] + lanes[index + 8]) + (lanes[index + 4] + lanes[index + 12]) for index in range(4)]
                    expected = struct.pack('<f', (parts[0] + parts[2]) + (parts[1] + parts[3]))
                    require(struct.pack('<f', tree.project_row(name, row, vector)) == expected, 'ordered projection control differs')
            for callback in (lambda: tree.group('w1', 0, 1), lambda: tree.read_row('w1', 32), lambda: tree.weight('w1', 0, -1)):
                rejected = False
                try:
                    callback()
                except ValueError:
                    rejected = True
                require(rejected, 'invalid tree location accepted')
        original = path.read_bytes()
        for label, payload in (('truncated', original[:-1]), ('root', original[:HEADER.size] + bytes([original[HEADER.size] ^ 1]) + original[HEADER.size + 1:]),
                               ('payload', original[:-1] + bytes([original[-1] ^ 1]))):
            damaged = Path(temporary) / (label + '.tree')
            damaged.write_bytes(payload)
            rejected = False
            try:
                with ExpertTree(damaged) as tree:
                    tree.read_row('w2', 31)
            except (ValueError, zlib.error):
                rejected = True
            require(rejected, 'corrupt tree accepted')
    print('PASS: 804 scale controls, 4096 weight decode pairs, code layouts, tree descent, ordered projections and six invalid-location/corruption cases.')


def build(directory):
    directory.mkdir()
    before = SOURCE.stat()
    known = json.loads((HERE / 'expert0-inspection-results.json').read_text())
    expected = {row['name']: row for row in known['stored_tensors']}
    matrices = []
    with SOURCE.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header_bytes = stream.read(length)
        require(sha(header_bytes) == known['header_sha256'], 'source header changed since inspection')
        header = json.loads(header_bytes)
        for name, shape in SHAPES.items():
            payloads = []
            for kind in ('weight_packed', 'weight_scale'):
                tensor = PREFIX + name + '.' + kind
                record = header[tensor]
                start, end = record['data_offsets']
                stream.seek(8 + length + start)
                payload = stream.read(end - start)
                require(len(payload) == expected[tensor]['bytes'] and sha(payload) == expected[tensor]['sha256'], 'source tensor changed')
                payloads.append(payload)
            matrices.append((name, shape, *payloads))
    pending = directory / 'expert.pending'
    with (directory / 'rows.jsonl').open('x') as audit:
        accounting, root = write_tree(pending, matrices, audit)
    decoded_hashes = {}
    value_count = 0
    table = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
    with ExpertTree(pending) as tree, SOURCE.open('rb') as stream:
        for name, shape, _, _ in matrices:
            code_hash, scale_hash, weight_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
            rows, width = shape
            for row in range(rows):
                packed, scales = tree.read_row(name, row)
                stream.seek(expected[PREFIX + name + '.weight_packed']['absolute_offset'] + row * width // 2)
                original_codes = stream.read(width // 2)
                stream.seek(expected[PREFIX + name + '.weight_scale']['absolute_offset'] + row * width // 32)
                original_scales = stream.read(width // 32)
                require((packed, scales) == (original_codes, original_scales), 'reopened tree differs from actual source')
                generated = equation_weights(packed, scales)
                codes = codes_from_packed(original_codes)
                powers = np.repeat(np.frombuffer(original_scales, dtype=np.uint8), 32).astype(np.int32) - 127
                reference = table[codes] * np.exp2(powers).astype(np.float32)
                require(generated.tobytes() == reference.astype('<f4').tobytes(), 'decoded expert weights differ')
                code_hash.update(packed)
                scale_hash.update(scales)
                weight_hash.update(generated.tobytes())
                value_count += width
            require(code_hash.hexdigest() == expected[PREFIX + name + '.weight_packed']['sha256'], 'code tensor hash differs')
            require(scale_hash.hexdigest() == expected[PREFIX + name + '.weight_scale']['sha256'], 'scale tensor hash differs')
            decoded_hashes[name] = {'packed_sha256': code_hash.hexdigest(), 'scales_sha256': scale_hash.hexdigest(),
                                    'float32_weights_sha256': weight_hash.hexdigest()}
        examples = [tree.group(name, 0, 0) for name in SHAPES]
    require(value_count == 33030144, 'incomplete expert verification')
    after = SOURCE.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'source file changed')
    final = directory / 'expert.tree'
    require(not final.exists(), 'expert tree target exists')
    pending.rename(final)
    report = {'gate': 'PASS', 'layer': 1, 'expert': 0, 'tree_bytes': final.stat().st_size,
              'original_code_scale_bytes': 17547264, 'saved_bytes': 17547264 - final.stat().st_size,
              'tree_sha256': sha(final.read_bytes()), **accounting,
              'matrices': 3, 'row_nodes': 9728, 'implicit_group_nodes': 1032192, 'weight_leaves': value_count,
              'all_stored_bytes_match_source': True, 'all_decoded_weights_bit_exact': True,
              'decoded_tensor_hashes': decoded_hashes, 'examples': examples,
              'inspection_result_sha256': sha((HERE / 'expert0-inspection-results.json').read_bytes()),
              'implementation_sha256': {name: sha((HERE / name).read_bytes()) for name in ('expert-tree.py', 'expert-scale-exceptions.py')},
              'rows_audit_sha256': sha((directory / 'rows.jsonl').read_bytes()), 'source_unchanged': True,
              'limits': 'One expert only. Tree uses compressed JSON root, per-row code layouts and exact scale descriptors; groups/leaves are generated by address formulas. Smallest tested row-code payload, not global optimum. No full model or nonlinear expert forward tested.'}
    (directory / 'root.json').write_text(json.dumps(root, indent=2) + '\n')
    with (directory / 'results.json').open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'examples'}, indent=2))
    print('report_sha256', sha((directory / 'results.json').read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('self-check', 'build', 'read'))
    parser.add_argument('path', type=Path, nargs='?')
    parser.add_argument('--matrix', choices=tuple(SHAPES), default='w1')
    parser.add_argument('--row', type=int, default=0)
    parser.add_argument('--group', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'self-check':
        self_check()
    elif args.command == 'build':
        build(args.path)
    else:
        with ExpertTree(args.path) as tree:
            print(json.dumps(tree.group(args.matrix, args.row, args.group), indent=2))


if __name__ == '__main__':
    main()