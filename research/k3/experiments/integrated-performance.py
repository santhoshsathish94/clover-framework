#!/usr/bin/env python3
"""Measure the frozen integrated model and reference, retaining correctness gates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time


SEED = Path('/opt/clover-k3/reverse-integration-20260929-a/step1')
PROMPTS = {'france': '1008,10484,318,15383,387', 'japan': '1008,10484,318,10417,387'}
SETTINGS = {
    'OMP_NUM_THREADS': '16', 'OMP_PROC_BIND': 'close', 'OMP_PLACES': 'cores',
    'K3_PREFETCH': '4', 'K3_TRUNKRAM': '0', 'K3_NREADER': '14', 'K3_XDEC': '2',
    'K3_PLGRAN': '1', 'K3_HUGE': '1', 'K3_INDEX': '/opt/clover-k3/build/eqidx.bin',
}
PROTECTED = {
    '/opt/clover-k3/clover-k3.c': '5628f7c3d2b7932b2b8776dcb498d54bbc730b1785ba7819989bdb51b6604e65',
    '/opt/clover-k3/build/clover-k3': '4cebb69e5379c4f0c9c03d4071c49840d3d7596ecfd2c5c0d8b83f6679717d35',
    '/opt/clover-k3/gate.sh': '752291aa5cda8194bbaf1f68d6a843e9ca02d3e221e8e19a5b714c075b01f5dc',
    '/opt/clover-k3/build/eqidx.bin': 'c2b7962c88fe7d984260a9c5d8f1b23ba0866e9cb9a39bf35a689f6a6a906981',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x') as destination:
        json.dump(value, destination, indent=2)
        destination.write('\n')


def system_state():
    fields = {'MemTotal', 'MemAvailable', 'Cached', 'SwapCached', 'SwapTotal', 'SwapFree'}
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in fields:
            memory[key + '_kib'] = int(value.split()[0])
    return {'memory': memory, 'load_average': Path('/proc/loadavg').read_text().split()[:3]}


def measure(command, directory, environment, prefix='model'):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('This measurement uses Linux wait4 RSS units (KiB).')
    with (directory / (prefix + '.log')).open('xb') as stdout, (directory / (prefix + '.stderr')).open('xb') as stderr:
        started = time.perf_counter_ns()
        with subprocess.Popen(command, cwd=directory, env=environment, stdout=stdout, stderr=stderr) as process:
            waited, status, usage = os.wait4(process.pid, 0)
            finished = time.perf_counter_ns()
            assert waited == process.pid
            process.returncode = os.waitstatus_to_exitcode(status)
            exit_code = process.returncode
    return {
        'exit_code': exit_code, 'elapsed_seconds': (finished - started) / 1e9,
        'peak_rss_kib': usage.ru_maxrss, 'peak_rss_gib': usage.ru_maxrss / (1024 ** 2),
        'user_cpu_seconds': usage.ru_utime, 'system_cpu_seconds': usage.ru_stime,
        'major_faults': usage.ru_majflt, 'minor_faults': usage.ru_minflt,
        'filesystem_input_blocks': usage.ru_inblock, 'filesystem_output_blocks': usage.ru_oublock,
        'voluntary_context_switches': usage.ru_nvcsw, 'involuntary_context_switches': usage.ru_nivcsw,
    }


def self_check():
    with tempfile.TemporaryDirectory(prefix='clover-performance-control-') as temporary:
        directory = Path(temporary)
        program = 'data=bytearray(64*1024*1024); data[::4096]=bytes([1])*(len(data)//4096); print(len(data),sum(data))'
        result = measure([sys.executable, '-c', program], directory, os.environ.copy(), 'allocation')
        assert result['exit_code'] == 0 and result['elapsed_seconds'] > 0
        assert result['peak_rss_kib'] >= 64 * 1024
        assert (directory / 'allocation.log').read_text().strip() == '67108864 16384'
        failure = measure([sys.executable, '-c', 'raise SystemExit(7)'], directory, os.environ.copy(), 'failure')
        assert failure['exit_code'] == 7
        print(json.dumps({'gate': 'PASS', 'allocation_control': result, 'nonzero_exit_propagated': 7}, indent=2))


def verify_sources():
    seed = json.loads((SEED / 'results.json').read_text())
    assert seed['gate'] == 'PASS' and seed['items_integrated'] == list(range(13, 0, -1))
    for name, expected in seed['sha256'].items():
        assert digest(SEED / name) == expected, name
    for name, expected in PROTECTED.items():
        assert digest(Path(name)) == expected, name
    return seed


def run_trial(root, phase, prompt, repetition, variant, seed):
    name = f'{phase}-{prompt}-{repetition}-{variant}'
    directory = root / name
    directory.mkdir()
    for filename in seed['sha256']:
        os.link(SEED / filename, directory / filename)
    for filename in ('saved-scale-exceptions.json', 'saved-joint-results.json',
                     'france-reference.bin', 'france-reference.routes', 'japan-reference.bin', 'japan-reference.routes'):
        os.link(SEED / filename, directory / filename)
    environment = os.environ.copy()
    environment.update(SETTINGS)
    enabled = variant == 'integrated'
    environment.update({
        'K3_IDS': PROMPTS[prompt], 'K3_GROUP_COVERAGE': '2' if enabled else '0',
        'K3_JOINT_RANK': '1' if enabled else '0', 'K3_SCALE_ROWS': '1' if enabled else '0',
        'K3_SCALE_EXCEPTIONS': '1' if enabled else '0', 'K3_VECTOR_ENDPOINTS': '3' if enabled else '0',
        'K3_BASIS_ROUTER': '2' if enabled else '0',
        'K3_GROUP_COVERAGE_REPORT': str(directory / (prompt + '.jsonl')),
        'K3_LOGITS': str(directory / (prompt + '.bin')), 'K3_DUMPSEL': str(directory / (prompt + '.routes')),
        'K3_ENDPOINT_REPORT': str(directory / (prompt + '-endpoints.json')),
        'K3_BASIS_REPORT': str(directory / (prompt + '-basis.bin')),
        'K3_REVERSE_VECTOR_REPORT': str(directory / (prompt + '-vector.bin')),
    })
    binary = SEED / ('candidate' if enabled else 'reference')
    before = system_state()
    timing = measure([str(binary)], directory, environment)
    after = system_state()
    write_json(directory / 'measurement.json', timing)
    assert timing['exit_code'] == 0, name
    assert (directory / (prompt + '.bin')).read_bytes() == (SEED / (prompt + '-reference.bin')).read_bytes(), name
    assert (directory / (prompt + '.routes')).read_bytes() == (SEED / (prompt + '-reference.routes')).read_bytes(), name
    validation = None
    if enabled:
        with (directory / 'validation.log').open('xb') as stdout, (directory / 'validation.stderr').open('xb') as stderr:
            subprocess.run([sys.executable, str(SEED / 'reverse-check.py'), str(directory), '--stage', '1',
                            '--runs', prompt, '--output', str(directory / 'validation.json')],
                           stdout=stdout, stderr=stderr, check=True)
        validation = json.loads((directory / 'validation.json').read_text())
        assert validation['gate'] == 'PASS' and validation['items_integrated'] == list(range(13, 0, -1))
    log = (directory / 'model.log').read_text()
    internal = re.search(r'^total wall time\s*:\s*([0-9.]+) s', log, re.MULTILINE)
    read_bytes = re.search(r'^read_bytes\s*:\s*([0-9.]+) GB', log, re.MULTILINE)
    result = {
        'name': name, 'phase': phase, 'prompt': prompt, 'repetition': repetition, 'variant': variant,
        'binary_sha256': digest(binary), 'metrics': timing,
        'model_reported_timed_seconds': float(internal.group(1)) if internal else None,
        'model_reported_read_gb': float(read_bytes.group(1)) if read_bytes else None,
        'output_md5': hashlib.md5((directory / (prompt + '.bin')).read_bytes()).hexdigest(),
        'output_sha256': digest(directory / (prompt + '.bin')),
        'routes_sha256': digest(directory / (prompt + '.routes')),
        'validation_sha256': digest(directory / 'validation.json') if validation else None,
        'before': before, 'after': after, 'gate': 'PASS',
        'full_output_bit_exact': True, 'routes_bit_exact': True,
    }
    write_json(directory / 'run-result.json', result)
    print(json.dumps({'trial': name, 'seconds': timing['elapsed_seconds'], 'peak_rss_gib': timing['peak_rss_gib'],
                      'major_faults': timing['major_faults'], 'gate': 'PASS'}), flush=True)
    return result


def summarize(runs):
    summary = {}
    for prompt in PROMPTS:
        variants = {}
        for variant in ('reference', 'integrated'):
            selected = [run for run in runs if run['prompt'] == prompt and run['variant'] == variant]
            assert len(selected) == 3
            metrics = {}
            for field in ('elapsed_seconds', 'peak_rss_gib', 'user_cpu_seconds', 'system_cpu_seconds', 'major_faults', 'filesystem_input_blocks'):
                values = [run['metrics'][field] for run in selected]
                metrics[field] = {'median': statistics.median(values), 'minimum': min(values), 'maximum': max(values), 'values': values}
            variants[variant] = metrics
        reference, integrated = variants['reference'], variants['integrated']
        ratio = integrated['elapsed_seconds']['median'] / reference['elapsed_seconds']['median']
        paired_ratios = []
        for repetition in (1, 2, 3):
            pair = {run['variant']: run for run in runs if run['prompt'] == prompt and run['repetition'] == repetition}
            paired_ratios.append(pair['integrated']['metrics']['elapsed_seconds'] / pair['reference']['metrics']['elapsed_seconds'])
        summary[prompt] = {'variants': variants, 'integrated_time_over_reference': ratio,
                           'elapsed_change_percent': (ratio - 1) * 100,
                           'paired_time_ratios': paired_ratios,
                           'peak_rss_delta_gib': integrated['peak_rss_gib']['median'] - reference['peak_rss_gib']['median']}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?')
    parser.add_argument('--phase', choices=('warmup', 'measure'))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.directory or not args.phase:
        parser.error('directory and --phase are required')
    seed = verify_sources()
    root = args.directory.resolve()
    if args.phase == 'warmup':
        root.mkdir()
        cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
        metadata = {'host': 'AX102', 'cpu': cpu, 'kernel': platform.release(), 'settings': SETTINGS,
                    'seed_result_sha256': digest(SEED / 'results.json'), 'script_sha256': digest(Path(__file__)),
                    'config_sha256': digest(Path('/opt/clover-k3/config.env')), 'protected_sha256': PROTECTED,
                    'method': 'Linux wait4 per-process peak RSS (KiB); monotonic parent elapsed time around spawn/wait, including startup, model, diagnostics and teardown. Correctness checking subprocess is outside the timed interval.',
                    'cache_policy': 'One warm-up per prompt/binary. No cache flush, no cold-start claim; page-cache behavior observed through faults/I/O and host snapshots.',
                    'comparison': 'Frozen as-is integrated correctness harness versus unchanged reference. All integrated reference computations, codecs, self-checks and trace writes remain enabled.',
                    'scope': 'Two five-token prefills, final integrated sampled sites; not autoregressive decode throughput or a production codec.'}
        write_json(root / 'metadata.json', metadata)
        runs = []
        for prompt_index, prompt in enumerate(PROMPTS):
            order = ('reference', 'integrated') if prompt_index == 0 else ('integrated', 'reference')
            for variant in order:
                runs.append(run_trial(root, 'warmup', prompt, 0, variant, seed))
        write_json(root / 'warmup-results.json', {'gate': 'PASS', 'runs': runs})
        return
    metadata = json.loads((root / 'metadata.json').read_text())
    assert metadata['script_sha256'] == digest(Path(__file__))
    assert metadata['config_sha256'] == digest(Path('/opt/clover-k3/config.env'))
    warmups = json.loads((root / 'warmup-results.json').read_text())
    assert warmups['gate'] == 'PASS' and len(warmups['runs']) == 4
    runs = []
    for prompt_index, prompt in enumerate(PROMPTS):
        for repetition in (1, 2, 3):
            order = ('reference', 'integrated') if (prompt_index + repetition) % 2 else ('integrated', 'reference')
            for variant in order:
                runs.append(run_trial(root, 'measure', prompt, repetition, variant, seed))
    verify_sources()
    report = {'gate': 'PASS', 'metadata': metadata, 'warmups': warmups['runs'], 'runs': runs,
              'summary': summarize(runs), 'all_full_outputs_and_routes_bit_exact': True,
              'all_integrated_representation_gates_passed': True, 'protected_hashes_unchanged': True,
              'limitations': 'Three measured trials per cell, no significance claim. Peak RSS includes resident mappings and is not system RAM or filesystem cache. This benchmark retains extra checks, duplicate reference work and intermediate buffers; it does not measure an optimized deployed representation.'}
    write_json(root / 'results.json', report)
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()