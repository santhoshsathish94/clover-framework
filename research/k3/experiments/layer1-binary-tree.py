#!/usr/bin/env python3
"""Binary, inherited source-backed hierarchy for layer1's routed experts only."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile


MATRICES = ('w1', 'w3', 'w2')
SHAPES = {'w1': (3072, 3584), 'w3': (3072, 3584), 'w2': (3584, 3072)}
CODEBOOK = (0.0, .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)
EXPECTED_BRANCH_SHA = '78adf5b56e17fab672776784d8f10ee1acac53892945e74ba2b9125efb4b2e7b'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mode(counts):
    return min(counts, key=lambda value: (-counts[value], value))


def branch(axis, start, stop, left, right, **identity):
    common = {name: value for name, value in left['shared'].items()
              if name in right['shared'] and right['shared'][name] == value}
    for name in common:
        del left['shared'][name]
        del right['shared'][name]
    return {'kind': 'branch', 'axis': axis, 'range': [start, stop], **identity,
            'shared': common, 'left': left, 'right': right}


def matrix_leaf(expert, matrix, source_file, layer):
    name = matrix['name']
    roles = {'w1': 'gate', 'w3': 'up', 'w2': 'down'}
    require(name in MATRICES and matrix['role'] == roles[name], 'unexpected expert matrix')
    require(tuple(matrix['weight_shape']) == SHAPES[name], 'matrix shape differs')
    rows, width = SHAPES[name]
    require(matrix['groups_per_row'] == width // 32 and matrix['groups'] == rows * width // 32
            and matrix['weight_count'] == rows * width and matrix['code_bytes'] == rows * width // 2
            and matrix['scale_bytes'] == rows * width // 32, 'matrix accounting differs')
    codes = {int(key): int(value) for key, value in matrix['code_counts'].items()}
    scales = {int(key): int(value) for key, value in matrix['scale_counts'].items()}
    require(sum(codes.values()) == matrix['weight_count'] and sum(scales.values()) == matrix['groups'],
            'incomplete expert histogram')
    require(all(0 <= key < 16 and count > 0 for key, count in codes.items())
            and all(0 <= key < 256 and count > 0 for key, count in scales.items()), 'histogram domain invalid')
    tensors = {}
    for record in matrix['stored_tensors']:
        kind = record['name'].rsplit('.', 1)[1]
        require(kind in ('weight_packed', 'weight_scale') and kind not in tensors, 'unexpected tensor leaf')
        prefix = f'language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{name}.'
        require(record['name'] == prefix + kind and record['dtype'] == 'U8', 'tensor outside requested expert')
        expected_shape = [rows, width // (2 if kind == 'weight_packed' else 32)]
        require(record['shape'] == expected_shape and record['bytes'] == math.prod(expected_shape), 'tensor shape differs')
        tensors[kind] = {'offset': record['absolute_offset'], 'sha256': record['sha256']}
    require(set(tensors) == {'weight_packed', 'weight_scale'}, 'missing tensor leaf')
    return {'kind': 'matrix', 'expert': expert, 'matrix': name,
            'shared': {'layer': layer, 'source_file': source_file, 'stored_dtype': 'U8', 'group_width': 32,
                       'code_bits': 4, 'scale_bits': 8, 'weights_per_matrix': rows * width,
                       'code_payload_bytes': matrix['code_bytes'], 'scale_payload_bytes': matrix['scale_bytes'],
                       'code_domain': sorted(codes), 'scale_domain': sorted(scales),
                       'most_frequent_scale': mode(scales), 'most_frequent_code': mode(codes)},
            'tensors': tensors, 'code_counts': matrix['code_counts'], 'scale_counts': matrix['scale_counts']}


def make_tree(experts, source_file, layer=1):
    require(experts and [item['expert'] for item in experts] == list(range(len(experts))), 'expert IDs not consecutive')

    def matrices_for(expert, start, stop):
        if stop - start == 1:
            return matrix_leaf(expert['expert'], expert['matrices'][start], source_file, layer)
        middle = (start + stop + 1) // 2
        return branch('matrix', start, stop, matrices_for(expert, start, middle),
                      matrices_for(expert, middle, stop), expert=expert['expert'])

    def split(start, stop):
        if stop - start == 1:
            require([item['name'] for item in experts[start]['matrices']] == list(MATRICES), 'matrix order differs')
            return matrices_for(experts[start], 0, len(MATRICES))
        middle = (start + stop) // 2
        return branch('expert', start, stop, split(start, middle), split(middle, stop))

    scales, codes = Counter(), Counter()
    for expert in experts:
        for matrix in expert['matrices']:
            scales.update({int(key): value for key, value in matrix['scale_counts'].items()})
            codes.update({int(key): value for key, value in matrix['code_counts'].items()})
    return {'format': 'layer1-binary-source-tree-v1', 'expert_count': len(experts),
            'scope': 'Source-backed routed-expert hierarchy only, excluding trunk/shared experts/other layers.',
            'matrix_order': list(MATRICES),
            'matrix_schemas': {'w1': {'role': 'gate', 'shape_id': 'gate_up'},
                               'w3': {'role': 'up', 'shape_id': 'gate_up'},
                               'w2': {'role': 'down', 'shape_id': 'down'}},
            'shape_templates': {'gate_up': [3072, 3584], 'down': [3584, 3072]},
            'defaults': {'scale_byte': mode(scales), 'scale_occurrences': scales[mode(scales)],
                         'scale_total': sum(scales.values()), 'code': mode(codes),
                         'code_occurrences': codes[mode(codes)], 'code_total': sum(codes.values()),
                         'semantics': 'Frequency summaries, never replacements for differing stored values. Actual scales and codes are read at leaves.'},
            'descendant_rule': ['binary expert ranges', 'binary matrix ranges', 'binary row ranges',
                                'binary 32-weight group ranges', 'binary coordinate ranges', 'weight'],
            'root': split(0, len(experts))}


def reconstruct(tree):
    output = {expert: [] for expert in range(tree['expert_count'])}
    stats = {'branches': 0, 'matrix_nodes': 0, 'shared_fields': 0, 'max_matrix_path_edges': 0}

    def visit(node, inherited, depth):
        require(set(node['shared']).isdisjoint(inherited), 'redundantly stored inherited fact')
        resolved = {**inherited, **node['shared']}
        stats['shared_fields'] += len(node['shared'])
        if node['kind'] == 'branch':
            require('left' in node and 'right' in node, 'non-binary internal node')
            left, right = node['left']['shared'], node['right']['shared']
            require(not any(key in right and right[key] == value for key, value in left.items()),
                    'common child fact not lifted to parent')
            stats['branches'] += 1
            visit(node['left'], resolved, depth + 1)
            visit(node['right'], resolved, depth + 1)
            return
        require(node['kind'] == 'matrix' and 'left' not in node and 'right' not in node, 'invalid matrix node')
        stats['matrix_nodes'] += 1
        stats['max_matrix_path_edges'] = max(stats['max_matrix_path_edges'], depth)
        matrix = node['matrix']
        rows, width = tree['shape_templates'][tree['matrix_schemas'][matrix]['shape_id']]
        require(resolved['layer'] == 1 and resolved['stored_dtype'] == 'U8' and resolved['group_width'] == 32, 'inherited facts differ')
        result = {'name': matrix, 'role': tree['matrix_schemas'][matrix]['role'], 'weight_shape': [rows, width],
                  'groups_per_row': width // 32, 'groups': rows * width // 32,
                  'weight_count': rows * width, 'code_bytes': resolved['code_payload_bytes'],
                  'scale_bytes': resolved['scale_payload_bytes'], 'code_counts': node['code_counts'],
                  'scale_counts': node['scale_counts'], 'stored_tensors': []}
        for kind in ('weight_packed', 'weight_scale'):
            field = node['tensors'][kind]
            shape = [rows, width // (2 if kind == 'weight_packed' else 32)]
            result['stored_tensors'].append({'name': f'language_model.model.layers.1.block_sparse_moe.experts.{node["expert"]}.{matrix}.{kind}',
                                             'absolute_offset': field['offset'], 'bytes': math.prod(shape),
                                             'dtype': resolved['stored_dtype'], 'shape': shape, 'sha256': field['sha256']})
        require(resolved['weights_per_matrix'] == result['weight_count'], 'shared weight count differs')
        require(resolved['code_domain'] == sorted(map(int, node['code_counts']))
                and resolved['scale_domain'] == sorted(map(int, node['scale_counts'])), 'inherited domain differs')
        require(resolved['most_frequent_scale'] == mode({int(key): value for key, value in node['scale_counts'].items()})
                and resolved['most_frequent_code'] == mode({int(key): value for key, value in node['code_counts'].items()}), 'inherited mode differs')
        output[node['expert']].append(result)

    visit(tree['root'], {}, 0)
    require(stats['branches'] == stats['matrix_nodes'] - 1, 'not a full binary stored tree')
    require(all([matrix['name'] for matrix in matrices] == list(MATRICES) for matrices in output.values()), 'tree expert coverage differs')
    return output, stats


class BinaryExpertTree:
    def __init__(self, path):
        self.tree = json.loads(Path(path).read_text())
        require(self.tree['format'] == 'layer1-binary-source-tree-v1', 'unsupported tree')
        reconstruct(self.tree)

    def matrix(self, expert, matrix):
        require(0 <= expert < self.tree['expert_count'] and matrix in MATRICES, 'expert or matrix out of range')
        target = MATRICES.index(matrix)
        node, inherited, path = self.tree['root'], {}, []
        while True:
            inherited.update(node['shared'])
            if node['kind'] == 'matrix':
                require(node['expert'] == expert and node['matrix'] == matrix, 'binary path resolves wrong matrix')
                return node, inherited, path
            start, stop = node['range']
            index = expert if node['axis'] == 'expert' else target
            require(start <= index < stop, 'invalid branch interval')
            middle = (start + stop) // 2 if node['axis'] == 'expert' else (start + stop + 1) // 2
            direction = 'left' if index < middle else 'right'
            path.append({'axis': node['axis'], 'range': [start, stop], 'direction': direction,
                         'fields_stored_here': sorted(node['shared'])})
            node = node[direction]

    def locate(self, expert, matrix, row, coordinate):
        node, inherited, path = self.matrix(expert, matrix)
        rows, width = self.tree['shape_templates'][self.tree['matrix_schemas'][matrix]['shape_id']]
        require(0 <= row < rows and 0 <= coordinate < width, 'row or coordinate out of range')
        for axis, count, target in (('row', rows, row), ('group', width // 32, coordinate // 32),
                                    ('coordinate_in_group', 32, coordinate % 32)):
            start, stop = 0, count
            while stop - start > 1:
                middle = (start + stop) // 2
                direction = 'left' if target < middle else 'right'
                path.append({'axis': axis, 'range': [start, stop], 'direction': direction})
                start, stop = (start, middle) if direction == 'left' else (middle, stop)
        return {'kind': 'weight', 'layer': inherited['layer'], 'expert': expert, 'matrix': matrix,
                'row': row, 'coordinate': coordinate, 'group': coordinate // 32, 'path': path,
                'source_file': inherited['source_file'], 'inherited': inherited,
                'code_byte_offset': node['tensors']['weight_packed']['offset'] + row * width // 2 + coordinate // 2,
                'nibble_shift': (coordinate % 2) * 4,
                'scale_byte_offset': node['tensors']['weight_scale']['offset'] + row * width // 32 + coordinate // 32}

    def read_weight(self, expert, matrix, row, coordinate):
        location = self.locate(expert, matrix, row, coordinate)
        with Path(location['source_file']).open('rb') as stream:
            stream.seek(location['code_byte_offset'])
            packed = stream.read(1)
            stream.seek(location['scale_byte_offset'])
            scale = stream.read(1)
        require(len(packed) == len(scale) == 1, 'short source leaf read')
        code, scale = (packed[0] >> location['nibble_shift']) & 15, scale[0]
        magnitude = CODEBOOK[code]
        value = math.copysign(0.0, magnitude) if scale == 255 else math.ldexp(magnitude, scale - 127)
        try:
            raw = struct.pack('<f', value)
        except OverflowError:
            raw = struct.pack('<f', math.copysign(math.inf, value))
        location.update({'code': code, 'scale': scale, 'weight': struct.unpack('<f', raw)[0],
                         'float32_bits_hex': f'{struct.unpack("<I", raw)[0]:08x}',
                         'scale_matches_root_default': scale == self.tree['defaults']['scale_byte'],
                         'source_bytes_read': 2})
        return location


def self_check():
    shared = {'dtype': 'U8', 'scale': 121}
    left = {'shared': {**shared, 'min': 110}}
    right = {'shared': {**shared, 'min': 111}}
    parent = branch('expert', 0, 2, left, right)
    require(parent['shared'] == shared and left['shared'] == {'min': 110} and right['shared'] == {'min': 111},
            'common fields did not lift correctly')
    require(mode({120: 4, 121: 9, 122: 1}) == 121 and mode({120: 4, 121: 4}) == 120, 'default rule differs')
    for size in (1, 2, 3, 7, 896):
        for target in range(size):
            start, stop = 0, size
            while stop - start > 1:
                middle = (start + stop) // 2
                start, stop = (start, middle) if target < middle else (middle, stop)
            require(start == target and stop == target + 1, 'binary split omitted leaf')
    print('PASS: exact common-field hoisting, differing fields retained, deterministic mode ties and complete non-power-of-two binary splits.')


def build(source_map, directory):
    require(digest(source_map) == EXPECTED_BRANCH_SHA, 'actual expert map identity differs')
    source = json.loads(source_map.read_text())
    require(source['layer'] == 1 and len(source['experts']) == 896, 'only layer1 experts supported')
    directory.mkdir()
    tree = make_tree(source['experts'], '/root/k3model/model-00002-of-000096.safetensors')
    restored, stats = reconstruct(tree)
    for expert in source['experts']:
        require(restored[expert['expert']] == expert['matrices'], 'inherited tree changes original expert records')
    path = directory / 'binary-tree.json'
    with path.open('x') as destination:
        json.dump(tree, destination, indent=2)
        destination.write('\n')
    reader = BinaryExpertTree(path)
    checks = 0
    for expert in source['experts']:
        for matrix in expert['matrices']:
            rows, width = matrix['weight_shape']
            tensor = {record['name'].rsplit('.', 1)[1]: record for record in matrix['stored_tensors']}
            for row, coordinate in ((0, 0), (rows // 2, width // 2), (rows - 1, width - 1)):
                leaf = reader.locate(expert['expert'], matrix['name'], row, coordinate)
                require(leaf['code_byte_offset'] == tensor['weight_packed']['absolute_offset'] + (row * width + coordinate) // 2
                        and leaf['scale_byte_offset'] == tensor['weight_scale']['absolute_offset'] + row * (width // 32) + coordinate // 32,
                        'binary weight path changes original offsets')
                require(all(step['direction'] in ('left', 'right') for step in leaf['path']), 'non-binary path')
                checks += 1
    examples = [reader.read_weight(expert, matrix, row, coordinate) for expert, matrix, row, coordinate in
                ((0, 'w1', 0, 0), (447, 'w3', 0, 0), (448, 'w2', 3583, 3071), (895, 'w1', 3071, 3583))]
    negative_checks = 0
    for request in ((896, 'w1', 0, 0), (-1, 'w1', 0, 0), (0, 'bad', 0, 0), (0, 'w1', 3072, 0), (0, 'w2', 0, 3072)):
        rejected = False
        try:
            reader.locate(*request)
        except ValueError:
            rejected = True
        require(rejected, 'invalid location accepted')
        negative_checks += 1
    with (directory / 'example-paths.json').open('x') as destination:
        json.dump(examples, destination, indent=2)
        destination.write('\n')
    report = {'gate': 'PASS', 'layer': 1, 'experts': 896, 'matrix_leaves_in_stored_index': 2688,
              **stats, 'stored_nodes': stats['branches'] + stats['matrix_nodes'], 'logical_weight_leaves': 29595009024,
              'all_original_expert_records_reconstructed': True, 'binary_weight_paths_checked': checks,
              'actual_weights_read': len(examples), 'actual_payload_bytes_read': len(examples) * 2,
              'invalid_locations_rejected': negative_checks, 'root_shared': tree['root']['shared'],
              'root_defaults': tree['defaults'], 'tree_json_bytes': path.stat().st_size,
              'tree_sha256': digest(path), 'example_paths_sha256': digest(directory / 'example-paths.json'),
              'verified_source_map_sha256': EXPECTED_BRANCH_SHA, 'script_sha256': digest(Path(__file__)),
              'limits': 'Source-backed structural tree, not standalone encoded weights. Index ends at matrix records; binary row/group/coordinate descendants are generated down to individual weight leaves. Common metadata is hoisted exactly; most-frequent values are labeled defaults, not substitutes for actual source exceptions. No trunk/other-layer access or model execution; original expert payloads required for leaf reads.'}
    with (directory / 'results.json').open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps(report, indent=2))
    print('results_sha256', digest(directory / 'results.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('self-check')
    builder = commands.add_parser('build')
    builder.add_argument('source_map', type=Path)
    builder.add_argument('directory', type=Path)
    read = commands.add_parser('read')
    read.add_argument('tree', type=Path)
    read.add_argument('--expert', type=int, required=True)
    read.add_argument('--matrix', choices=MATRICES, default='w1')
    read.add_argument('--row', type=int, default=0)
    read.add_argument('--coordinate', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'self-check':
        self_check()
    elif args.command == 'build':
        build(args.source_map, args.directory)
    else:
        print(json.dumps(BinaryExpertTree(args.tree).read_weight(args.expert, args.matrix, args.row, args.coordinate), indent=2))


if __name__ == '__main__':
    main()