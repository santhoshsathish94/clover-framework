#!/usr/bin/env python3
"""Independent topology, inherited-fact and actual leaf checks for the binary tree."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('source_map', type=Path)
    args = parser.parse_args()
    directory = args.directory
    report = json.loads((directory / 'results.json').read_text())
    tree = json.loads((directory / 'binary-tree.json').read_text())
    source = json.loads(args.source_map.read_text())
    assert digest(args.source_map) == report['verified_source_map_sha256']
    assert digest(directory / 'binary-tree.json') == report['tree_sha256']
    originals = {(expert['expert'], matrix['name']): matrix for expert in source['experts'] for matrix in expert['matrices']}
    levels, leaves, differing_descendant_fields = Counter(), set(), 0

    def walk(node, inherited, depth):
        nonlocal differing_descendant_fields
        assert not (node['shared'].keys() & inherited.keys())
        facts = {**inherited, **node['shared']}
        levels[depth] += 1
        if node['kind'] == 'matrix':
            assert 'left' not in node and 'right' not in node
            key = (node['expert'], node['matrix'])
            assert key not in leaves and key in originals
            leaves.add(key)
            original = originals[key]
            assert node['code_counts'] == original['code_counts'] and node['scale_counts'] == original['scale_counts']
            assert facts['layer'] == 1 and facts['group_width'] == 32 and facts['code_bits'] == 4
            assert facts['code_domain'] == sorted(map(int, original['code_counts']))
            assert facts['scale_domain'] == sorted(map(int, original['scale_counts']))
            assert facts['most_frequent_scale'] == min(map(int, original['scale_counts']), key=lambda key: (-original['scale_counts'][str(key)], key))
            assert facts['most_frequent_code'] == min(map(int, original['code_counts']), key=lambda key: (-original['code_counts'][str(key)], key))
            shape = tree['shape_templates'][tree['matrix_schemas'][node['matrix']]['shape_id']]
            assert shape == original['weight_shape'] and facts['weights_per_matrix'] == shape[0] * shape[1]
            for tensor in original['stored_tensors']:
                kind = tensor['name'].rsplit('.', 1)[1]
                assert node['tensors'][kind] == {'offset': tensor['absolute_offset'], 'sha256': tensor['sha256']}
                assert facts['code_payload_bytes' if kind == 'weight_packed' else 'scale_payload_bytes'] == tensor['bytes']
            return {key}
        assert node['kind'] == 'branch' and 'left' in node and 'right' in node
        common = node['left']['shared'].keys() & node['right']['shared'].keys()
        assert all(node['left']['shared'][key] != node['right']['shared'][key] for key in common)
        differing_descendant_fields += len(common)
        left = walk(node['left'], facts, depth + 1)
        right = walk(node['right'], facts, depth + 1)
        assert left.isdisjoint(right)
        low, high = node['range']
        if node['axis'] == 'expert':
            middle = (low + high) // 2
            assert {expert for expert, matrix in left} == set(range(low, middle))
            assert {expert for expert, matrix in right} == set(range(middle, high))
        else:
            middle = (low + high + 1) // 2
            assert left == {(node['expert'], matrix) for matrix in tree['matrix_order'][low:middle]}
            assert right == {(node['expert'], matrix) for matrix in tree['matrix_order'][middle:high]}
        return left | right

    assert walk(tree['root'], {}, 0) == set(originals)
    assert sum(levels.values()) == report['stored_nodes'] == 5375
    assert len(leaves) == 2688 and differing_descendant_fields > 0
    scales, codes = Counter(), Counter()
    for matrix in originals.values():
        scales.update({int(key): value for key, value in matrix['scale_counts'].items()})
        codes.update({int(key): value for key, value in matrix['code_counts'].items()})
    defaults = tree['defaults']
    assert defaults['scale_byte'] == min(scales, key=lambda key: (-scales[key], key)) == 121
    assert defaults['code'] == min(codes, key=lambda key: (-codes[key], key)) == 1
    assert defaults['scale_occurrences'] == scales[121] < sum(scales.values())
    assert defaults['code_occurrences'] == codes[1] < sum(codes.values())
    script = directory.parent / 'layer1-binary-tree.py'
    assert digest(script) == report['script_sha256']
    spec = importlib.util.spec_from_file_location('binary_reader', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reader = module.BinaryExpertTree(directory / 'binary-tree.json')
    requests = [(0, 'w1', 0, 0), (0, 'w1', 0, 224), (0, 'w1', 0, 9),
                (447, 'w3', 1536, 1792), (448, 'w2', 3583, 3071), (895, 'w1', 3071, 3583)]
    tested = []
    source_file = Path(tree['root']['shared']['source_file'])
    before = source_file.stat()
    codebook = [0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.]
    with source_file.open('rb') as stream:
        for expert, matrix, row, coordinate in requests:
            result = reader.read_weight(expert, matrix, row, coordinate)
            original = originals[expert, matrix]
            width = original['weight_shape'][1]
            tensors = {tensor['name'].rsplit('.', 1)[1]: tensor for tensor in original['stored_tensors']}
            stream.seek(tensors['weight_packed']['absolute_offset'] + (row * width + coordinate) // 2)
            packed = stream.read(1)[0]
            stream.seek(tensors['weight_scale']['absolute_offset'] + row * (width // 32) + coordinate // 32)
            scale = stream.read(1)[0]
            code = (packed >> (4 * (coordinate % 2))) & 15
            expected = struct.pack('<f', math.ldexp(codebook[code], scale - 127))
            assert result['scale'] == scale and result['code'] == code
            assert result['float32_bits_hex'] == f'{struct.unpack("<I", expected)[0]:08x}'
            assert result['scale_matches_root_default'] == (scale == 121)
            tested.append(result)
    after = source_file.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    assert any(not row['scale_matches_root_default'] for row in tested)
    assert any(row['float32_bits_hex'] == '80000000' for row in tested)
    result = {'gate': 'PASS', 'stored_nodes_verified': 5375, 'matrix_records_verified': 2688,
              'tensor_records_verified': 5376, 'all_inherited_histograms_match': True,
              'all_internal_nodes_have_exactly_two_children': True, 'all_branch_partitions_exact': True,
              'no_equal_sibling_fact_left_unhoisted': True, 'differing_sibling_fields_retained': differing_descendant_fields,
              'nodes_per_depth': {str(key): value for key, value in sorted(levels.items())},
              'actual_leaves_checked': tested, 'nondefault_scale_preserved': True, 'negative_zero_preserved': True,
              'source_unchanged': True, 'tree_sha256': report['tree_sha256'], 'script_sha256': digest(Path(__file__)),
              'limits': 'All structural metadata checked against the prior full expert-only census; six actual weight leaves compared independently. Not a fresh full payload scan, standalone weight store, or model execution.'}
    with (directory / 'checks.json').open('x') as destination:
        json.dump(result, destination, indent=2)
        destination.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'actual_leaves_checked'}, indent=2))
    print('checks_sha256', digest(directory / 'checks.json'))


if __name__ == '__main__':
    main()