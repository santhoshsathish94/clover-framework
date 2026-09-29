#!/usr/bin/env python3
"""Verify full fruit contents, model consumption and preserved seed-model gates."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import tarfile
import zlib


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def verify_fruit(root):
    report = json.loads((root / 'results.json').read_text())
    census = json.loads((root / 'census.json').read_text())
    assert report['gate'] == census['gate'] == 'PASS'
    assert digest(root / 'census.json') == report['census_sha256']
    assert digest(root / 'blocks.jsonl') == report['blocks_log_sha256']
    for name, expected in census['artifacts'].items():
        assert digest(root / name) == expected
    header_format, entry_format = struct.Struct('<8s8I3Q32s32s'), struct.Struct('<Q4I32s')
    file_hash, output_hash, block_hashes = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    chosen, totals = Counter(), Counter()
    with (root / 'fruit.bin').open('rb') as source, (root / 'blocks.jsonl').open() as audit:
        raw_header = source.read(header_format.size)
        (magic, version, rows, width, block_rows, count, bits, blocks, reserved,
         dictionary_offset, index_offset, payload_offset, source_hash, metadata_hash) = header_format.unpack(raw_header)
        assert (magic, version, rows, width, block_rows, count, bits, blocks, reserved) == (b'K3SEED1\0', 1, 163840, 7168, 16, 6590, 13, 10240, 0)
        assert dictionary_offset == 128 and index_offset == 128 + count * 2
        assert payload_offset == index_offset + blocks * entry_format.size
        metadata = source.read(payload_offset - 128)
        assert hashlib.sha256(raw_header[:-32] + metadata).digest() == metadata_hash
        assert metadata[:count * 2] == (root / 'dictionary.bf16').read_bytes()
        file_hash.update(raw_header + metadata)
        for number, (offset, size, row_count, codec, crc, expected_hash) in enumerate(entry_format.iter_unpack(metadata[count * 2:])):
            assert source.tell() == offset and row_count == 16 and codec in (4, 5)
            payload = source.read(size)
            assert len(payload) == size and zlib.crc32(payload) == crc
            file_hash.update(payload)
            decoder = zlib.decompressobj()
            expected_size = row_count * width * 2
            unpacked = decoder.decompress(payload, expected_size + 1)
            assert len(unpacked) == expected_size and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
            values = row_count * width
            raw = bytearray(expected_size)
            if codec == 4:
                raw[0::2], raw[1::2] = unpacked[:values], unpacked[values:]
            else:
                for position in range(values):
                    word = 0
                    for bit in range(16):
                        location = bit * values + position
                        word |= ((unpacked[location // 8] >> (location % 8)) & 1) << bit
                    struct.pack_into('<H', raw, position * 2, word)
            assert hashlib.sha256(raw).digest() == expected_hash
            block_hashes.update(expected_hash)
            output_hash.update(raw)
            event = json.loads(next(audit))
            assert event['block'] == number and event['first_row'] == number * 16 and event['rows'] == 16
            assert event['raw_sha256'] == expected_hash.hex()
            assert event['stored_bytes'] == size == min(event['candidate_bytes'].values())
            expected_codec = 'zlib_bf16_byte_planes' if codec == 4 else 'zlib_bf16_bit_planes'
            assert event['codec'] == expected_codec
            chosen[expected_codec] += 1
            totals.update(event['candidate_bytes'])
        assert source.read(1) == b'' and audit.read() == ''
    assert file_hash.hexdigest() == report['fruit_sha256']
    assert output_hash.digest() == source_hash
    assert output_hash.hexdigest() == report['decoded_payload_sha256'] == census['source_sha256']
    assert dict(chosen) == report['selected_codecs'] and dict(totals) == report['fixed_codec_payload_totals']
    assert report['fruit_bytes'] == payload_offset + report['payload_bytes']
    return report, {'gate': 'PASS', 'blocks': blocks, 'values': rows * width,
                    'source_checkpoint_opened': False, 'independent_standard_library_decoder': True,
                    'ordered_block_hashes_sha256': block_hashes.hexdigest(),
                    'fruit_sha256': file_hash.hexdigest(), 'decoded_payload_sha256': output_hash.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    args = parser.parse_args()
    work = args.work
    fruit = Path('/opt/clover-k3/fruit')
    seed = Path('/opt/clover-k3/seed')
    old = Path('/opt/clover-k3/seed-integrated-20260929-a')
    prior = json.loads((work / 'seed-prior-results.json').read_text())
    assert prior['gate'] == 'PASS'
    for name, expected in prior['sha256'].items():
        assert digest(old / name) == expected
        if name in ('candidate.c', 'vector-endpoints.h'):
            continue
        path = work / ('seed-prior-candidate' if name == 'candidate' else name)
        if path.exists():
            assert digest(path) == expected, name
    for name, expected in prior['protected_sha256'].items():
        assert digest(Path(name)) == expected
    for dependency in prior['dependencies'].values():
        assert digest(Path(dependency['path'])) == dependency['sha256']
    seed_result = json.loads((seed / 'results.json').read_text())
    assert digest(seed / 'seed.bin') == seed_result['seed_sha256'] == prior['seed']
    fruit_result, content_check = verify_fruit(fruit)
    native = json.loads((work / 'native-results.json').read_text())
    assert native['gate'] == 'PASS' and native['decoded_values'] == 1174405120
    assert native['decoded_sha256'] == fruit_result['source_sha256']
    integrated = json.loads((work / 'enabled/results.json').read_text())
    disabled = json.loads((work / 'disabled/france-check.json').read_text())
    assert integrated['gate'] == disabled['gate'] == 'PASS'
    assert integrated['items_integrated'] == list(range(13, 0, -1))
    assert (integrated['unique_rows'], integrated['unique_pairs'], integrated['unique_groups']) == (1926, 214, 5778)
    assert sum(row['summary']['projection_outputs'] for row in integrated['runs']) == 4320
    for name, expected in integrated['sha256'].items():
        assert digest(work / name) == expected, name
    source = Path('/root/k3model/model-00094-of-000096.safetensors')
    runs = []
    with source.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        input_entry = header['language_model.model.embed_tokens.weight']
        input_offset = 8 + header_size + input_entry['data_offsets'][0]
        for run in integrated['runs']:
            name = run['name']
            assert run['gate'] == 'PASS' and run['full_output']['bit_exact']
            assert run['full_output']['norm']['changed'] == run['full_output']['logits']['changed'] == 0
            assert run['reference_routes_identical'] and run['coverage_matches_reference_route_expectations']
            assert digest(work / 'enabled' / (name + '.jsonl')) == run['raw_trace_sha256']
            fruit_report = json.loads((work / 'enabled' / (name + '-fruit.json')).read_text())
            seed_report = json.loads((work / 'enabled' / (name + '-seed.json')).read_text())
            for report, field in ((fruit_report, 'original_head_protected_bytes'), (seed_report, 'original_embedding_protected_bytes')):
                assert report['gate'] == 'PASS' and report[field] == 2348806144
                assert report['unprotected_boundary_bytes'] == 4096 and report['guard_probe_sigsegv']
            assert [entry['consumer'] for entry in fruit_report['passes']] == ['head_projection', 'tail_expression']
            for entry in fruit_report['passes']:
                assert entry['rows'] == 163840 and entry['values'] == 1174405120 and entry['blocks'] == 10240
                assert entry['workers'] == 16 and entry['payload_bytes_read'] == fruit_result['payload_bytes']
                assert entry['ordered_block_hashes_sha256'] == content_check['ordered_block_hashes_sha256']
            actual = (work / 'enabled' / (name + '-seed-rows.bf16')).read_bytes()
            expected = bytearray()
            for token in run['prompt_ids']:
                stream.seek(input_offset + token * 7168 * 2)
                expected.extend(stream.read(7168 * 2))
            assert actual == expected and seed_report['values_loaded'] == 35840 and seed_report['entry_uses_seed_rows']
            for suffix in ('bin', 'routes'):
                assert (work / 'enabled' / (name + '.' + suffix)).read_bytes() == (work / (name + '-reference.' + suffix)).read_bytes()
            runs.append({'prompt': name, 'ids': run['prompt_ids'], 'full_output': run['full_output'],
                         'routes_bit_exact': True, 'seed_rows_match_source': True,
                         'fruit': fruit_report, 'seed': seed_report, 'all_prior_stage_gates_pass': True})
    for suffix in ('bin', 'routes'):
        expected = (work / ('france-reference.' + suffix)).read_bytes()
        for relative in ('prior/france.', 'disabled/france.', 'france-reference-after.'):
            assert (work / (relative + suffix)).read_bytes() == expected
    assert not (work / 'disabled/france-fruit.json').exists()
    rejection = json.loads((work / 'rejection-results.json').read_text())
    assert rejection['gate'] == 'PASS' and len(rejection['checks']) == 2
    for check in rejection['checks']:
        assert check['exit_code'] != 0 and check['no_logits_written']
        folder = work / ('reject-' + check['case'])
        assert check['expected_error'] in (folder / 'stderr.log').read_text()
        assert not (folder / 'logits.bin').exists()
    files = sorted(path.name for path in work.iterdir() if path.is_file() and
                   (path.suffix in ('.c', '.h', '.py', '.patch') or path.name in ('candidate', 'reference', 'seed-prior-candidate')))
    report = {'gate': 'PASS', 'host': 'AX102 AMD Ryzen 9 7950X3D', 'fruit_dataset': fruit_result,
              'independent_fruit_contents': content_check, 'native_reader': native, 'runs': runs,
              'prior_seed_control_pass': True, 'fruit_disabled_control_pass': True, 'reference_repeated_matches': True,
              'rejections': rejection, 'seed_sha256_unchanged': prior['seed'], 'protected_sha256': prior['protected_sha256'],
              'dependencies_unchanged': prior['dependencies'],
              'sha256': {name: digest(work / name) for name in files},
              'prior_model_result_sha256': digest(work / 'seed-prior-results.json'),
              'prior_stage_gate_result_sha256': digest(work / 'enabled/results.json'),
              'build_flags': prior['build_flags'], 'settings': {**prior['settings'],
                  'K3_FRUIT_HEAD': '/opt/clover-k3/fruit/fruit.bin', 'K3_FRUIT_GUARD_PROBE': '1'},
              'scope': 'Complete output head streamed through both ordinary and rounded-source consumers for two five-token prefills, with seed input and all13 prior sampled stages enabled.',
              'limits': 'No new throughput/peak-RAM measurement, longer contexts or generation. Reference computations retained. Guards cover page-aligned interiors of the two original table mappings, not4096boundarybytes per table or arbitrary alternate access. No source data/seed/installed model changes.'}
    with (work / 'results.json').open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    with (work / 'verified-artifacts.tar.gz').open('xb') as destination:
        with tarfile.open(fileobj=destination, mode='w:gz') as archive:
            for name in files:
                archive.add(work / name, arcname='model/' + name, recursive=False)
            for name in ('results.json', 'native-results.json', 'rejection-results.json'):
                archive.add(work / name, arcname=name, recursive=False)
            for folder in ('enabled', 'disabled', 'prior'):
                for path in sorted((work / folder).iterdir()):
                    if path.is_file() and path.suffix in ('.bin', '.bf16', '.json', '.jsonl', '.routes', '.log', '.stderr'):
                        archive.add(path, arcname=folder + '/' + path.name, recursive=False)
            for path in sorted(work.glob('*reference*')):
                if path.is_file() and path.suffix in ('.bin', '.routes', '.log', '.stderr'):
                    archive.add(path, arcname=path.name, recursive=False)
            for case in ('missing', 'wrong_source'):
                archive.add(work / ('reject-' + case) / 'stderr.log', arcname='reject-' + case + '/stderr.log', recursive=False)
            for name in ('census.json', 'results.json', 'all-n.tsv', 'dictionary.bf16', 'counts.u64le', 'scan-chunks.jsonl', 'blocks.jsonl'):
                archive.add(fruit / name, arcname='dataset/' + name, recursive=False)
    print(json.dumps({'gate': 'PASS', 'fruit_bytes': fruit_result['fruit_bytes'], 'runs': len(runs),
                      'fruit_consumers_per_run': 2, 'guard_probes_passed': 4,
                      'results_sha256': digest(work / 'results.json'), 'archive_sha256': digest(work / 'verified-artifacts.tar.gz'),
                      'archive_bytes': (work / 'verified-artifacts.tar.gz').stat().st_size}, indent=2))


if __name__ == '__main__':
    main()