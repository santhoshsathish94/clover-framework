#!/usr/bin/env python3
"""Check real tree navigation and ordered row evaluation against checkpoint data."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory
    implementation = directory.parent / 'expert-tree.py'
    spec = importlib.util.spec_from_file_location('tree_reader', implementation)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = json.loads((directory / 'results.json').read_text())
    previous = json.loads((directory.parent / 'expert0-inspection-results.json').read_text())
    require = module.require
    require(report['gate'] == 'PASS' and report['weight_leaves'] == 33030144, 'full tree verification missing')
    require(hashlib.sha256((directory / 'expert.tree').read_bytes()).hexdigest() == report['tree_sha256'], 'tree identity differs')
    for name, expected in report['implementation_sha256'].items():
        require(hashlib.sha256((directory.parent / name).read_bytes()).hexdigest() == expected, 'implementation changed')
    locations = {row['name']: row for row in previous['stored_tensors']}
    codebook = [0.0, .5, 1., 1.5, 2., 3., 4., 6., -0.0, -.5, -1., -1.5, -2., -3., -4., -6.]
    navigation, projections = [], []
    prefix = 'language_model.model.layers.1.block_sparse_moe.experts.0.'
    with module.ExpertTree(directory / 'expert.tree') as tree, Path(previous['source_file']).open('rb') as source:
        require(tree.root == json.loads((directory / 'root.json').read_text()), 'human-readable tree differs from file')
        for matrix in ('w1', 'w3', 'w2'):
            rows, width = tree.matrices[matrix]['shape']
            groups = width // 32
            for row in (0, rows // 2, rows - 1):
                source.seek(locations[prefix + matrix + '.weight_packed']['absolute_offset'] + row * width // 2)
                packed = source.read(width // 2)
                source.seek(locations[prefix + matrix + '.weight_scale']['absolute_offset'] + row * groups)
                scales = source.read(groups)
                codes = [code for byte in packed for code in (byte & 15, byte >> 4)]
                weights = [math.ldexp(codebook[code], scales[index // 32] - 127) for index, code in enumerate(codes)]
                for group in (0, groups // 2, groups - 1):
                    leaf = tree.group(matrix, row, group)
                    expected_codes = codes[group * 32:(group + 1) * 32]
                    expected_masks = [sum(((code >> bit) & 1) << position for position, code in enumerate(expected_codes)) for bit in range(4)]
                    require(leaf['codes'] == expected_codes and leaf['masks'] == expected_masks and leaf['scale'] == scales[group], 'tree group differs')
                    expected_weights = weights[group * 32:(group + 1) * 32]
                    expected_bytes = struct.pack('<32f', *expected_weights)
                    actual_bits = struct.pack('<32I', *(int(word, 16) for word in leaf['float32_bits_hex']))
                    require(expected_bytes == actual_bits, 'group signed float32 bits differ')
                    for coordinate in (group * 32, group * 32 + 31):
                        require(struct.pack('<f', tree.weight(matrix, row, coordinate)) == struct.pack('<f', weights[coordinate]), 'leaf read differs')
                    navigation.append({'matrix': matrix, 'row': row, 'group': group, 'scale': leaf['scale'],
                                       'masks': leaf['masks'], 'weights_sha256': hashlib.sha256(expected_bytes).hexdigest()})
                vector = np.array([((coordinate % 67) - 33) / 64 for coordinate in range(width)], dtype=np.float32)
                lanes = [0.0] * 16
                for start in range(0, width, 16):
                    for lane in range(16):
                        product = weights[start + lane] * float(vector[start + lane])
                        lanes[lane] = lanes[lane] + product
                partial = [(lanes[lane] + lanes[lane + 8]) + (lanes[lane + 4] + lanes[lane + 12]) for lane in range(4)]
                reference = struct.pack('<f', (partial[0] + partial[2]) + (partial[1] + partial[3]))
                candidate = struct.pack('<f', tree.project_row(matrix, row, vector))
                require(candidate == reference, 'actual tree row projection differs')
                projections.append({'matrix': matrix, 'row': row, 'input_values': width, 'result': struct.unpack('<f', reference)[0],
                                    'float32_bits_hex': f'{struct.unpack("<I", reference)[0]:08x}', 'bit_exact': True})
        example = tree.group('w1', 0, 0)
    report_check = {'gate': 'PASS', 'tree_sha256': report['tree_sha256'], 'source_checkpoint_read_only': True,
                    'groups_checked': len(navigation), 'weights_in_checked_groups': len(navigation) * 32,
                    'individual_weight_reads_checked': len(navigation) * 2, 'row_projection_checks': len(projections),
                    'projection_input': 'Deterministic exact float32 vector ((coordinate % 67)-33)/64; not a captured inference input.',
                    'navigation': navigation, 'projections': projections, 'example': example,
                    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'limits': 'Navigation and linear row evaluation only. No SiTU or full expert/model forward tested.'}
    with (directory / 'navigation-results.json').open('x') as destination:
        json.dump(report_check, destination, indent=2)
        destination.write('\n')
    print(json.dumps({key: value for key, value in report_check.items() if key not in ('navigation', 'projections', 'example')}, indent=2))
    print('navigation_results_sha256', hashlib.sha256((directory / 'navigation-results.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    main()