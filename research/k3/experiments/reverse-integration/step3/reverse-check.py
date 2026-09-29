#!/usr/bin/env python3
"""Check cumulative consumers without changing prior correctness validators."""

import argparse
import hashlib
import importlib.util
import json
from math import comb, isfinite
import os
from pathlib import Path
import struct
import zlib


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--stage', type=int, choices=(13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3), required=True)
    parser.add_argument('--runs', nargs='+', choices=('smoke', 'france', 'japan'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = args.directory
    fixed_view = args.output.with_suffix('.fixed')
    group_view = args.output.with_suffix('.groups')
    audit = args.output.with_suffix('.audit')
    fixed_view.mkdir()
    group_view.mkdir()
    audit.mkdir()
    adaptive = load('adaptive', work / 'expert-scale-exceptions.py')
    fixed = load('fixed', work / 'expert-scale-row.py')
    adapter = load('adaptive_check', work / 'expert-scale-exception-integrated-check.py')
    scale_check = load('scale_check', work / 'expert-scale-integrated-check.py')
    coverage = load('coverage', work / 'expert-group-coverage-check.py')
    order = load('order', work / 'expert-group-order-check.py')
    outputs = load('outputs', work / 'expert-group-check.py')
    joint = load('joint', work / 'expert-joint-rank.py')
    joint_check = load('joint_check', work / 'expert-joint-integrated-check.py')
    joint_saved = json.loads((work / 'saved-joint-results.json').read_text())
    joint_predictions = {tuple(row[field] for field in ('layer', 'expert', 'part', 'row', 'group')):
                         int(row['joint_rank']) for row in joint_saved['groups']}
    saved = json.loads((work / 'saved-scale-exceptions.json').read_text())
    predicted = {tuple(row[field] for field in ('layer', 'expert', 'part', 'row')): row
                 for row in saved['rows']}
    assert hashlib.md5((work / 'france-reference.bin').read_bytes()).hexdigest() == outputs.BASELINE_MD5
    all_rows, all_groups, all_pairs, runs = {}, set(), set(), []
    for name in args.runs:
        reference = 'japan' if name == 'japan' else 'france'
        for filename in (reference + '-reference.bin', reference + '-reference.routes', name + '.bin', name + '.routes'):
            if not (audit / filename).exists():
                os.link(work / filename, audit / filename)
        actual_rows, scale_summary, exception_summary = adapter.translate(work, fixed_view, name, adaptive, fixed, predicted)
        rows, _ = scale_check.scale_trace(fixed_view, group_view, name, fixed)
        if args.stage <= 11:
            joint_check.audit_trace(group_view, audit, name, joint, order, joint_predictions)
        else:
            os.link(group_view / (name + '.jsonl'), audit / (name + '.jsonl'))
        consumed_groups = set()
        for line in (work / (name + '.jsonl')).read_text().splitlines():
            event = json.loads(line)
            if event['type'] == 'scale_row' and args.stage <= 12:
                original = bytes(event['original_scales'])
                payload = bytes(event['base_offset_payload'])
                assert fixed.decode_scales(payload, len(original)) == original
                assert fixed.encode_scales(original)[0] == payload
            if event['type'] != 'group':
                continue
            key = tuple(event[field] for field in ('layer', 'expert', 'part', 'row', 'group'))
            assert key not in consumed_groups
            if args.stage > 11:
                assert event['joint_mode'] == 0 and event['consumer_format'] == 'packed'
                assert bytes(event['consumer_payload']) == bytes([event['scale']]) + bytes(event['packed'])
            else:
                assert event['joint_mode'] == 1
            if args.stage <= 10:
                packed = bytes(event['packed'])
                codes = [(packed[position // 2] >> (4 * (position % 2))) & 15 for position in range(32)]
                counts = [codes.count(code) for code in range(16)]
                rank = sum(comb(sum(counts[:ordinal]) + ordinal - 1, ordinal) for ordinal in range(1, 16))
                payload = bytes(event['histogram_payload'])
                assert len(payload) == 5 and int.from_bytes(payload, 'little') == rank < comb(47, 15)
                if args.stage <= 9:
                    arrangement = order.rank_sequence(codes, counts)
                    value = event['scale'] + sum(count << (8 + 6 * code) for code, count in enumerate(counts))
                    value |= arrangement << 104
                    bits = 104 + (order.ways(counts) - 1).bit_length()
                    assert event['order_bits'] == bits
                    assert bytes(event['order_payload']) == value.to_bytes((bits + 7) // 8, 'little')
                if args.stage <= 8:
                    positions = [position for position, code in enumerate(codes) if code & 4]
                    rank = sum(comb(position, ordinal) for ordinal, position in enumerate(positions, 1))
                    bits = 6 + (comb(32, len(positions)) - 1).bit_length()
                    assert event['mask_bits'] == bits
                    value = (rank << 6) | len(positions)
                    assert bytes(event['mask_payload']) == value.to_bytes((bits + 7) // 8, 'little')
                if args.stage <= 7:
                    masks = [sum(((code >> plane) & 1) << position for position, code in enumerate(codes))
                             for plane in range(4)]
                    expected = b''.join(mask.to_bytes(4, 'little') for mask in masks) + bytes([event['scale']])
                    assert bytes(event['four_mask_payload']) == expected
            consumed_groups.add(key)
        result, groups, pairs = coverage.check_run(audit, name, reference, 1 if name == 'smoke' else 2, order, outputs)
        assert consumed_groups == groups
        assert set(rows) == set(actual_rows) == {key[:4] for key in groups}
        routes = coverage.load_routes(work / (reference + '-reference.routes'))
        for key, row in rows.items():
            assert row['positions'] == sum(key[1] in routes[key[0], position][1] for position in range(5))
            assert (row['width'], row['rows']) == ((3584, 3072) if key[2] < 2 else (3072, 3584))
            if key in all_rows:
                assert all_rows[key] == actual_rows[key]
            all_rows[key] = actual_rows[key]
        assert result['summary']['joint_mode'] == int(args.stage <= 11)
        assert result['summary']['joint_decoder_checks'] == (102 if args.stage <= 11 else 0)
        result['legacy_descriptor_role'] = 'Audit only; actual consumed packed/joint bytes independently checked.'
        result['raw_trace_sha256'] = hashlib.sha256((work / (name + '.jsonl')).read_bytes()).hexdigest()
        result['scale_summary'] = scale_summary
        result['exception_summary'] = exception_summary
        if args.stage <= 6:
            endpoint_check = load('endpoint_check', work / 'vector-endpoints-check.py')
            endpoint = json.loads((work / (name + '-endpoints.json')).read_text())
            assert endpoint['mode'] == (3 if args.stage <= 5 else 2)
            assert endpoint['positions'] == 5 and endpoint['tail_sources'] == 9
            entry_count = 35840 if args.stage <= 5 else 0
            expected = {'embedding': entry_count, 'entry_norm': entry_count, 'tail_aggregate': 7168,
                        'tail_norm': 7168, 'head_logits': 163840}
            assert endpoint['counts_valid'] and endpoint['gate'] == 'PASS'
            for label, count in expected.items():
                assert endpoint[label] == {'checked': count, 'changed': 0}
            assert endpoint_check.compare_outputs((work / (reference + '-reference.bin')).read_bytes(),
                                                  (work / (name + '.bin')).read_bytes())['bit_exact']
            result['endpoints'] = endpoint
        if args.stage <= 4:
            data = (work / (name + '-basis.bin')).read_bytes()
            assert len(data) == 16 + 896 * 16
            assert struct.unpack_from('<4I', data) == (0x31505242, 896, 84, 0)
            for offset in range(16, len(data), 16):
                words = struct.unpack_from('<4I', data, offset)
                assert words[0] == words[1] and words[2] == words[3]
                assert all(isfinite(value) for value in struct.unpack_from('<4f', data, offset))
            result['router'] = {'layer': 84, 'position': 0, 'projections': 896, 'scores': 896,
                                'changed': 0, 'sha256': hashlib.sha256(data).hexdigest()}
        if args.stage <= 3:
            data = (work / (name + '-vector.bin')).read_bytes()
            assert struct.unpack_from('<3I', data) == (0x31564552, 7168, args.stage)
            original = data[12:12 + 7168 * 4]
            offset = 12 + len(original)
            stages = []
            for expected_kind in (1, 2):
                kind, size = struct.unpack_from('<2I', data, offset)
                offset += 8
                payload = data[offset:offset + size]
                offset += size
                assert kind == expected_kind and len(payload) == size
                decoded = zlib.decompress(payload)
                expected = original if kind == 1 else b''.join(original[lane::4] for lane in range(4))
                assert decoded == expected
                stages.append({'kind': kind, 'bytes': size, 'exact': True})
            assert offset == len(data)
            result['vector'] = {'layer': 84, 'position': 0, 'components': 7168, 'stages': stages,
                                'sha256': hashlib.sha256(data).hexdigest()}
        result.pop('descriptor_bytes')
        runs.append(result)
        all_groups.update(groups)
        all_pairs.update(pairs)
    source_names = sorted(path.name for path in work.iterdir()
                          if path.suffix in ('.c', '.h', '.py', '.patch') or path.name in ('candidate', 'reference'))
    report = {
        'stage': args.stage, 'items_integrated': list(range(13, args.stage - 1, -1)),
        'gate': 'PASS', 'runs': runs, 'unique_rows': len(all_rows), 'unique_pairs': len(all_pairs),
        'unique_groups': len(all_groups), 'unique_scales': sum(len(value[0]) for value in all_rows.values()),
        'adaptive_scale_bytes': sum(len(value[1]) for value in all_rows.values()),
        'sha256': {name: hashlib.sha256((work / name).read_bytes()).hexdigest() for name in source_names},
        'scope': 'Existing sampled rows/groups at layers 1/48/92, listed five-token prefills only.',
        'limits': 'Intermediate encodings and audit data retained; not a deployed store or RAM/speed measurement.',
    }
    with args.output.open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('runs', 'sha256')}, indent=2))


if __name__ == '__main__':
    main()