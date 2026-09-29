#!/usr/bin/env python3
"""Export a concrete embedding/intermediate-vector example from AX102 captures."""

import argparse
import hashlib
import json
from pathlib import Path
import struct


def rounded(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('basis', type=Path)
    parser.add_argument('layers', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = args.basis.read_bytes()
    capture_hash = hashlib.sha256(payload).hexdigest()
    assert capture_hash == '006abc5dfd8891a2cd16961c1577290ef8e98d341293ae4180a43a964a591371'
    magic, width, count, layer = struct.unpack_from('<4I', payload)
    assert (magic, width, count, layer) == (0x31534256, 7168, 9, 84)
    assert len(payload) == 16 + 4 * (count + (count + 2) * width)
    coefficients = struct.unpack_from('<%df' % count, payload, 16)
    offset = 16 + count * 4
    sources = []
    for source in range(count):
        sources.append(struct.unpack_from('<%df' % width, payload, offset))
        offset += width * 4
    offset += width * 4
    intermediate_bytes = payload[offset:]
    intermediate = struct.unpack('<%df' % width, intermediate_bytes)
    with args.layers.open('rb') as layers:
        embedding_bytes = layers.read(width * 4)
    assert embedding_bytes == struct.pack('<%df' % width, *sources[0])
    embedding = struct.unpack('<%df' % width, embedding_bytes)

    regenerated = []
    coordinate_steps = []
    for coordinate in range(width):
        total = 0.0
        for source, coefficient in enumerate(coefficients):
            contribution = rounded(coefficient * sources[source][coordinate])
            total = rounded(total + contribution)
            if coordinate == 0:
                coordinate_steps.append({'source_index': source,
                                         'source_value': sources[source][coordinate],
                                         'coefficient': coefficient,
                                         'rounded_product': contribution,
                                         'rounded_running_sum': total})
        regenerated.append(total)
    assert struct.pack('<%df' % width, *regenerated) == intermediate_bytes
    labels = ['snapshot entering layer %d' % source_layer for source_layer in range(0, 85, 12)]
    labels.append('live residual after layer 84 attention')
    result = {
        'prompt': 'The capital of France is',
        'token_ids': [1008, 10484, 318, 15383, 387],
        'selected_position': 0, 'selected_token_id': 1008, 'selected_token_text': 'The',
        'width': width,
        'embedding': {'site': 'Layer 0 input, BF16 embedding widened to float32',
                      'sha256': hashlib.sha256(embedding_bytes).hexdigest(), 'values': embedding},
        'intermediate': {'site': 'Layer 84 pre-MLP aggregation, before RMSNorm',
                         'sha256': hashlib.sha256(intermediate_bytes).hexdigest(), 'values': intermediate},
        'source_expression': {
            'sources': labels, 'coefficients': coefficients,
            'rounding': 'For each coordinate: total=0; in source order, total=float32(total+float32(coefficient*source_value)).',
            'source_data_still_required': True,
            'source_vectors_stored_separately_in_basis_capture': str(args.basis),
            'first_coordinate_steps': coordinate_steps,
            'regenerated_identical_components': width,
        },
        'capture_sha256': capture_hash,
        'scope': 'One real position at two different stages, not the same vector before/after compression. Existing captured run, no new model run.',
    }
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    saved = json.loads(args.output.read_text())
    assert struct.pack('<%df' % width, *saved['embedding']['values']) == embedding_bytes
    assert struct.pack('<%df' % width, *saved['intermediate']['values']) == intermediate_bytes
    print(json.dumps({'embedding_first_8': embedding[:8],
                      'intermediate_first_8': intermediate[:8],
                      'coefficients': coefficients, 'coordinate_0': coordinate_steps,
                      'identical_reconstruction': width,
                      'export_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()