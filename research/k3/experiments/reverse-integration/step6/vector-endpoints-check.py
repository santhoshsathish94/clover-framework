#!/usr/bin/env python3
"""Compare endpoint replacement stage reports and full outputs with Clover."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


WIDTH = 7168
VOCAB = 163840
BASELINE_MD5 = '23d162dcefb18211a7540ef12948f1eb'
MODES = {'disabled': 0, 'entry': 1, 'tail': 2, 'combined': 3}


def compare_outputs(reference, candidate):
    assert len(reference) == len(candidate) == (WIDTH + VOCAB) * 4
    reference_values = struct.unpack('<%df' % (WIDTH + VOCAB), reference)
    candidate_values = struct.unpack('<%df' % (WIDTH + VOCAB), candidate)
    reference_words = struct.unpack('<%dI' % (WIDTH + VOCAB), reference)
    candidate_words = struct.unpack('<%dI' % (WIDTH + VOCAB), candidate)
    assert all(math.isfinite(value) for value in reference_values + candidate_values)
    result = {'bit_exact': reference == candidate, 'md5': hashlib.md5(candidate).hexdigest()}
    for label, start, stop in (('norm', 0, WIDTH), ('logits', WIDTH, WIDTH + VOCAB)):
        result[label] = {
            'count': stop - start,
            'changed': sum(reference_words[index] != candidate_words[index] for index in range(start, stop)),
            'max_abs': max(abs(reference_values[index] - candidate_values[index]) for index in range(start, stop)),
        }
    result['token'] = max(range(VOCAB), key=lambda index: candidate_values[WIDTH + index])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--variants', nargs='+', choices=tuple(MODES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = (args.directory / 'reference.bin').read_bytes()
    assert hashlib.md5(reference).hexdigest() == BASELINE_MD5
    variants = []
    for name in args.variants:
        mode = MODES[name]
        stages = json.loads((args.directory / (name + '-stages.json')).read_text())
        assert stages['mode'] == mode and stages['positions'] == 5
        expected = {
            'embedding': 5 * WIDTH if mode & 1 else 0,
            'entry_norm': 5 * WIDTH if mode & 1 else 0,
            'tail_aggregate': WIDTH if mode & 2 else 0,
            'tail_norm': WIDTH if mode & 2 else 0,
            'head_logits': VOCAB if mode & 2 else 0,
        }
        for stage, count in expected.items():
            assert stages[stage]['checked'] == count
        assert stages['tail_sources'] == (9 if mode & 2 else 0)
        values = compare_outputs(reference, (args.directory / (name + '.bin')).read_bytes())
        passed = values['bit_exact'] and stages['counts_valid'] and stages['gate'] == 'PASS'
        passed = passed and all(stages[stage]['changed'] == 0 for stage in expected)
        variants.append({'variant': name, 'mode': mode, 'stages': stages,
                         'full_output': values, 'gate': 'PASS' if passed else 'FAIL'})
    source_names = ('baseline.c', 'candidate.c', 'vector-endpoints.h',
                    'vector-endpoints.patch', 'reference', 'candidate')
    repeated = args.directory / 'reference-after.bin'
    repeated_matches = repeated.read_bytes() == reference if repeated.exists() else None
    passed = all(variant['gate'] == 'PASS' for variant in variants) and repeated_matches is not False
    report = {
        'host': 'AX102, AMD Ryzen 9 7950X3D',
        'prompt_ids': [1008, 10484, 318, 15383, 387],
        'build_flags': '-O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5 -lm',
        'settings': {'OMP_NUM_THREADS': '16', 'OMP_PROC_BIND': 'close', 'OMP_PLACES': 'cores',
                     'K3_PREFETCH': '4', 'K3_TRUNKRAM': '0', 'K3_NREADER': '14',
                     'K3_XDEC': '2', 'K3_PLGRAN': '1', 'K3_HUGE': '1'},
        'reference_md5': BASELINE_MD5,
        'reference_repeated_matches': repeated_matches,
        'variants': variants,
        'gate': 'PASS' if passed else 'FAIL',
        'sha256': {name: hashlib.sha256((args.directory / name).read_bytes()).hexdigest()
                   for name in source_names},
        'performance': 'Deferred; no speed, memory or compression claim.',
        'scope': 'One five-token prompt, fresh prefill. Entry generates BF16 row values into the first RMSNorm. Tail generates rounded normalized source-expression values into every BF16 head row. Reference computations remain for diagnostics and other consumers; no global removal of vector arrays.',
    }
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()