#!/usr/bin/env python3
"""Read actual checkpoint tensor headers and bounded payload samples on AX102."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct


LAYER = re.compile(r'^language_model\.model\.layers\.(\d+)\.(.+)$')
EXPERT = re.compile(r'^block_sparse_moe\.experts\.(\d+)\.(w1|w3|w2)\.(weight_packed|weight_scale)$')


def sample(stream, offset, size):
    count = min(16, size)
    stream.seek(offset)
    first = stream.read(count)
    stream.seek(offset + size - count)
    last = stream.read(count)
    assert len(first) == len(last) == count
    return {'bytes_read': count * 2, 'boundary_sample_sha256': hashlib.sha256(first + last).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = Path('/root/k3model')
    packed_root = Path('/root/k3trunk_i8')
    assert not args.output.exists()
    files = sorted(model.glob('*.safetensors'))
    assert files
    groups = defaultdict(lambda: {'tensors': 0, 'bytes': 0, 'dtypes': Counter(), 'examples': []})
    layers = defaultdict(lambda: {'non_routed_tensors': 0, 'non_routed_bytes': 0,
                                  'shared_tensors': 0, 'shared_bytes': 0,
                                  'expert_tensors': 0, 'expert_bytes': 0, 'shards': set(),
                                  'expert_ids': set(), 'tensor_suffixes': set()})
    experts = defaultdict(set)
    expert_formats = Counter()
    shard_rows, globals_found, extras, records, samples = [], [], [], {}, []
    sample_categories = set()
    for path in files:
        file_size = path.stat().st_size
        with path.open('rb') as stream:
            prefix = stream.read(8)
            assert len(prefix) == 8
            header_size = struct.unpack('<Q', prefix)[0]
            assert 0 < header_size <= file_size - 8
            raw_header = stream.read(header_size)
            header = json.loads(raw_header)
            payload_base = 8 + header_size
            ranges, payload_bytes, count = [], 0, 0
            for name, tensor in header.items():
                if name == '__metadata__':
                    continue
                assert name not in records
                start, end = tensor['data_offsets']
                assert 0 <= start <= end <= file_size - payload_base
                size = end - start
                record = {'name': name, 'file': path.name, 'offset': payload_base + start,
                          'nbytes': size, 'shape': tensor['shape'], 'dtype': tensor['dtype']}
                records[name] = record
                ranges.append((start, end))
                payload_bytes += size
                count += 1
                match = LAYER.fullmatch(name)
                expert = None
                if name == 'language_model.model.embed_tokens.weight':
                    category = 'input_embedding_table'
                elif name == 'language_model.lm_head.weight':
                    category = 'output_head_table'
                elif match:
                    layer_id, suffix = int(match[1]), match[2]
                    layer = layers[layer_id]
                    layer['shards'].add(path.name)
                    expert = EXPERT.fullmatch(suffix)
                    if expert:
                        expert_id = int(expert[1])
                        category = 'routed_expert_pool'
                        layer['expert_tensors'] += 1
                        layer['expert_bytes'] += size
                        layer['expert_ids'].add(expert_id)
                        experts[layer_id, expert_id].add((expert[2], expert[3]))
                        expert_formats[(expert[2], expert[3], tensor['dtype'], tuple(tensor['shape']), size)] += 1
                    else:
                        assert '.experts.' not in suffix, name
                        layer['non_routed_tensors'] += 1
                        layer['non_routed_bytes'] += size
                        layer['tensor_suffixes'].add(suffix)
                        category = 'layer_shared_experts' if '.shared_experts.' in '.' + suffix else 'layer_other_trunk'
                        if category == 'layer_shared_experts':
                            layer['shared_tensors'] += 1
                            layer['shared_bytes'] += size
                elif name.startswith('language_model.'):
                    category = 'other_model_level'
                    globals_found.append(record)
                else:
                    category = 'outside_language_model_layers'
                    extras.append(record)
                group = groups[category]
                group['tensors'] += 1
                group['bytes'] += size
                group['dtypes'][tensor['dtype']] += 1
                if len(group['examples']) < 4:
                    group['examples'].append(record)
                sample_key = category if not expert else (category, expert[2], expert[3])
                if sample_key not in sample_categories:
                    samples.append({**record, **sample(stream, record['offset'], size)})
                    sample_categories.add(sample_key)
            cursor, gap_bytes = 0, 0
            for start, end in sorted(ranges):
                assert start >= cursor, path.name
                gap_bytes += start - cursor
                cursor = end
            trailing = file_size - payload_base - cursor
            assert trailing >= 0
            assert payload_base + payload_bytes + gap_bytes + trailing == file_size
            shard_rows.append({'file': path.name, 'file_bytes': file_size, 'header_bytes': header_size,
                               'header_sha256': hashlib.sha256(raw_header).hexdigest(), 'tensors': count,
                               'payload_bytes': payload_bytes, 'gap_bytes': gap_bytes, 'trailing_bytes': trailing})
    raw_trunk_index = (packed_root / 'trunk.json').read_bytes()
    trunk_index = json.loads(raw_trunk_index)
    packed_size = (packed_root / 'trunk.bin').stat().st_size
    trunk_layers, trunk_names, occupied = [], set(), []
    packed_dtypes = Counter()
    with (packed_root / 'trunk.bin').open('rb') as stream:
        for entry in trunk_index['layers']:
            layer_id = entry['layer']
            base, size = entry['file_off'], entry['nbytes']
            assert 0 <= base <= base + size <= packed_size
            occupied.append((base, base + size))
            fields, tensor_ranges = [], []
            for name, tensor in entry['tensors'].items():
                assert name not in trunk_names and name in records
                assert int(LAYER.fullmatch(name)[1]) == layer_id
                assert '.block_sparse_moe.experts.' not in name
                trunk_names.add(name)
                offset, count = tensor['off'], tensor['nbytes']
                assert 0 <= offset <= offset + count <= size
                tensor_ranges.append((offset, offset + count))
                fields.append({'name': name, 'offset': base + offset, **tensor,
                               'checkpoint_dtype': records[name]['dtype'],
                               'checkpoint_bytes': records[name]['nbytes']})
                packed_dtypes[tensor['dtype']] += 1
            cursor = 0
            for start, end in sorted(tensor_ranges):
                assert start >= cursor
                cursor = end
            samples_for_layer = []
            selected = [fields[0], fields[-1]] if len(fields) > 1 else fields
            for field in selected:
                boundary = sample(stream, field['offset'], field['nbytes'])
                if field['dtype'] == field['checkpoint_dtype']:
                    source = records[field['name']]
                    with (model / source['file']).open('rb') as original:
                        original_sample = sample(original, source['offset'], source['nbytes'])
                    assert boundary['boundary_sample_sha256'] == original_sample['boundary_sample_sha256']
                    boundary['unconverted_checkpoint_boundary_matches'] = True
                samples_for_layer.append({**field, **boundary})
            trunk_layers.append({'layer': layer_id, 'file_off': base, 'nbytes': size,
                                 'tensors': len(fields), 'tensor_payload_bytes': sum(field['nbytes'] for field in fields),
                                 'shared_expert_names': [name for name in entry['tensors'] if '.shared_experts.' in name],
                                 'router_names': [name for name in entry['tensors'] if '.block_sparse_moe.gate.' in name],
                                 'samples': samples_for_layer})
    cursor = 0
    for start, end in sorted(occupied):
        assert start >= cursor
        cursor = end
    layer_ids = sorted(layers)
    expected_parts = {(weight, kind) for weight in ('w1', 'w3', 'w2') for kind in ('weight_packed', 'weight_scale')}
    incomplete = [list(key) for key, parts in experts.items() if parts != expected_parts]
    layer_rows = []
    for layer_id in layer_ids:
        row = layers[layer_id]
        layer_rows.append({'layer': layer_id, **{key: value for key, value in row.items()
                                              if key not in ('shards', 'expert_ids', 'tensor_suffixes')},
                           'shards': sorted(row['shards']), 'expert_ids': sorted(row['expert_ids']),
                           'non_routed_tensor_suffixes': sorted(row['tensor_suffixes'])})
    missing_from_trunk = [name for name in records if LAYER.fullmatch(name)
                         and not EXPERT.fullmatch(LAYER.fullmatch(name)[2]) and name not in trunk_names]
    report = {
        'inspection': 'Read-only full tensor-header census plus bounded tensor boundary samples, not a full weight payload scan.',
        'checkpoint_directory': str(model), 'shard_count': len(files),
        'checkpoint_file_bytes': sum(row['file_bytes'] for row in shard_rows),
        'checkpoint_tensor_bytes': sum(row['payload_bytes'] for row in shard_rows),
        'header_bytes_read': sum(row['header_bytes'] + 8 for row in shard_rows),
        'tensor_count': len(records), 'groups': dict(groups), 'other_model_tensors': globals_found,
        'outside_language_model_tensors': extras, 'layers': layer_rows,
        'routed_expert_layer_ids': sorted({key[0] for key in experts}),
        'layer_expert_pairs': len(experts), 'incomplete_experts': incomplete,
        'expert_formats': [{'matrix': key[0], 'kind': key[1], 'dtype': key[2], 'shape': key[3],
                            'bytes_each': key[4], 'count': count} for key, count in sorted(expert_formats.items())],
        'packed_trunk': {'path': str(packed_root / 'trunk.bin'), 'bytes': packed_size,
                         'index_sha256': hashlib.sha256(raw_trunk_index).hexdigest(),
                         'declared_layers': trunk_index['n_layers'], 'layers': trunk_layers,
                         'tensor_count': len(trunk_names), 'dtype_counts': dict(packed_dtypes),
                         'checkpoint_layer_tensors_not_in_trunk': missing_from_trunk},
        'checkpoint_samples': samples, 'shards': shard_rows,
        'checks': {'all_shard_tensor_byte_ranges_nonoverlapping_and_in_bounds': True,
                   'all_shard_file_sizes_accounted': True, 'all_trunk_layer_tensor_ranges_in_bounds': True,
                   'all_trunk_tensor_names_found_in_checkpoint': True,
                   'routed_experts_excluded_from_packed_trunk': True},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limits': 'Headers establish declared tensor names/shapes/ownership and byte coverage. Boundary reads do not hash or verify every byte of the 1.5TB payload; no full checkpoint or int8 conversion correctness audit.',
    }
    with args.output.open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps({key: report[key] for key in ('shard_count', 'checkpoint_file_bytes', 'checkpoint_tensor_bytes',
                                                  'header_bytes_read', 'tensor_count', 'layer_expert_pairs',
                                                  'incomplete_experts', 'expert_formats', 'other_model_tensors')}, indent=2))
    for category, group in sorted(groups.items()):
        print(category, json.dumps({key: value for key, value in group.items() if key != 'examples'}))
    print('layer_ids', layer_ids)
    print('expert_counts_by_layer', [(row['layer'], len(row['expert_ids'])) for row in layer_rows])
    print('trunk', json.dumps({key: value for key, value in report['packed_trunk'].items() if key != 'layers'}))
    print('sampled_trunk_layers', json.dumps([row for row in trunk_layers if row['layer'] in (0, 1, 2, 92)], indent=2))
    print('report_sha256', hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()