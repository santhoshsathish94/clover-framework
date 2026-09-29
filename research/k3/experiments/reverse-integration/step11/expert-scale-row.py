#!/usr/bin/env python3
"""Study exact base-plus-offset scale rows from the indexed AX102 checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import struct


def encode_scales(scales):
    assert scales
    base = min(scales)
    width = (max(scales) - base).bit_length()
    packed = base | (width << 8)
    for index, scale in enumerate(scales):
        packed |= (scale - base) << (12 + index * width)
    bits = 12 + len(scales) * width
    return packed.to_bytes((bits + 7) // 8, 'little'), base, width, bits


def decode_scales(payload, count):
    assert count > 0 and len(payload) >= 2
    packed = int.from_bytes(payload, 'little')
    base, width = packed & 255, (packed >> 8) & 15
    assert width <= 8
    bits = 12 + count * width
    assert len(payload) == (bits + 7) // 8 and packed >> bits == 0
    values = [base + ((packed >> (12 + index * width)) & ((1 << width) - 1))
              for index in range(count)]
    assert all(value <= 255 for value in values)
    return bytes(values)


def records(path, layer, expert):
    data = path.read_bytes()
    assert data[:4] == b'K3EQ'
    layers, slots = struct.unpack_from('<ii', data, 4)
    assert layers == 93 and slots == 37
    offset = 12 + layers * slots * struct.calcsize('<iiqqii')
    model_count, = struct.unpack_from('<i', data, offset)
    assert model_count == 5
    offset += 4 + model_count * struct.calcsize('<iqqiii')
    file_count, = struct.unpack_from('<i', data, offset)
    offset += 4
    assert 0 < file_count < 1024
    files = []
    for index in range(file_count):
        size, = struct.unpack_from('<i', data, offset)
        offset += 4
        assert 0 < size < 65536
        files.append(Path(data[offset:offset + size].decode('utf-8')))
        offset += size
    count, = struct.unpack_from('<i', data, offset)
    offset += 4
    assert count == 92 * 896 * 6
    layout = struct.Struct('<iiiiiqqii')
    assert offset + count * layout.size == len(data)
    selected = {}
    for record in struct.iter_unpack(layout.format, data[offset:]):
        actual_layer, actual_expert, part, kind, file_id, start, size, rows, columns = record
        if (actual_layer, actual_expert, kind) != (layer, expert, 1):
            continue
        assert part in (0, 1, 2) and part not in selected and 0 <= file_id < len(files)
        assert size == rows * columns
        selected[part] = {'path': files[file_id], 'offset': start, 'nbytes': size,
                          'rows': rows, 'columns': columns}
    assert set(selected) == {0, 1, 2}
    return selected, hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--expert', type=int, default=498)
    args = parser.parse_args()
    controls = 0
    for width in range(9):
        sample = bytes([0, (1 << width) - 1, 0, (1 << width) - 1])
        payload, base, observed_width, bits = encode_scales(sample)
        assert observed_width == width and decode_scales(payload, len(sample)) == sample
        controls += 1
    for value in (0, 121, 255):
        sample = bytes([value]) * 112
        payload, base, width, bits = encode_scales(sample)
        assert width == 0 and decode_scales(payload, len(sample)) == sample
        controls += 1
    selected, index_hash = records(args.index, args.layer, args.expert)
    trace_groups = {}
    with args.trace.open() as trace:
        for line in trace:
            event = json.loads(line)
            if event['type'] == 'group' and (event['layer'], event['expert']) == (args.layer, args.expert):
                key = (event['part'], event['row'], event['group'])
                assert key not in trace_groups
                trace_groups[key] = event['scale']
    assert len(trace_groups) == 27
    result_rows = []
    for part, record in sorted(selected.items()):
        rows, columns = record['rows'], record['columns']
        assert (rows, columns) == ((3072, 112) if part < 2 else (3584, 96))
        tensor_part = ('w1', 'w3', 'w2')[part]
        tensor_name = f'language_model.model.layers.{args.layer}.block_sparse_moe.experts.{args.expert}.{tensor_part}.weight_scale'
        with record['path'].open('rb') as source:
            header_size, = struct.unpack('<Q', source.read(8))
            assert 0 < header_size <= 64 * 1024 * 1024
            metadata = json.loads(source.read(header_size))[tensor_name]
            assert metadata['shape'] == [rows, columns]
            assert record['offset'] == 8 + header_size + metadata['data_offsets'][0]
            assert record['nbytes'] == metadata['data_offsets'][1] - metadata['data_offsets'][0]
            for row in (0, rows // 2, rows - 1):
                source.seek(record['offset'] + row * columns)
                scales = source.read(columns)
                assert len(scales) == columns
                for group in (0, columns // 2, columns - 1):
                    assert scales[group] == trace_groups[part, row, group]
                payload, base, width, bits = encode_scales(scales)
                assert decode_scales(payload, columns) == scales
                if (part, row) == (0, 0):
                    assert scales[0] == 121
                result_rows.append({
                    'part': part, 'tensor': tensor_part, 'row': row, 'group_count': columns,
                    'checkpoint_path': str(record['path']), 'tensor_dtype': metadata['dtype'],
                    'absolute_offset': record['offset'] + row * columns,
                    'scales': list(scales), 'base': base, 'maximum': max(scales),
                    'offset_width': width, 'payload_bits': bits, 'payload': list(payload),
                    'original_bytes': columns, 'candidate_bytes': len(payload),
                    'roundtrip_byte_exact': True, 'scale_255_count': scales.count(255),
                    'scales_sha256': hashlib.sha256(scales).hexdigest(),
                })
    original = sum(row['original_bytes'] for row in result_rows)
    candidate = sum(row['candidate_bytes'] for row in result_rows)
    code_bytes = original * 16
    report = {
        'layer': args.layer, 'expert': args.expert, 'control_roundtrips': controls,
        'index_sha256': index_hash, 'reference_trace_sha256': hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        'reference_scale_crosschecks': len(trace_groups), 'rows': result_rows,
        'summary': {'complete_scale_rows': len(result_rows), 'scale_bytes_checked': original,
                    'original_scale_bytes': original, 'candidate_scale_bytes': candidate,
                    'saved_scale_bytes': original - candidate,
                    'unchanged_code_payload_bytes': code_bytes,
                    'original_code_plus_scale_payload_bytes': code_bytes + original,
                    'candidate_code_plus_scale_payload_bytes': code_bytes + candidate},
        'equation': 'scale_g = base + offset_g; base=min(row); width=bit_length(max(row)-base)',
        'layout': '8-bit base, 4-bit width, then fixed-width unsigned offsets in original group order, least-significant bit first. Width may be zero. Row length/order supplied by tensor metadata; each row byte padded; no per-row mode tag or store offset table included.',
        'scope': 'Fresh reads of nine complete contiguous scale rows of one real expert. Exact original-byte reconstruction and header/index/previous trace crosschecks, not a model consumer replacement or whole-expert compression measurement.',
    }
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))
    for row in result_rows:
        print(json.dumps({key: row[key] for key in ('tensor', 'row', 'group_count', 'base', 'maximum', 'offset_width', 'original_bytes', 'candidate_bytes')}))


if __name__ == '__main__':
    main()