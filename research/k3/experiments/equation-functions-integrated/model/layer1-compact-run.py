#!/usr/bin/env python3
"""Run a fresh layer-one compact integration trial with prior gates enabled."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


PROMPTS = {'france': [1008, 10484, 318, 15383, 387], 'japan': [1008, 10484, 318, 10417, 387]}


def run(work, folder, name, binary, compact, reference):
    prior = Path('/opt/clover-k3/fruit-integrated-20260929-a')
    previous = json.loads((prior / 'results.json').read_text())
    target = work / folder
    target.mkdir(exist_ok=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith('K3_')}
    env.update(previous['settings'])
    env.update(K3_INDEX='/opt/clover-k3/build/eqidx.bin', K3_TRUNKPATH='/root/k3trunk_i8/trunk.bin',
               K3_IDS=','.join(map(str, PROMPTS[name])), K3_JOINT_RANK='1', K3_SCALE_ROWS='1', K3_SCALE_EXCEPTIONS='1')
    fields = {'K3_LOGITS': '.bin', 'K3_DUMPSEL': '.routes', 'K3_GROUP_COVERAGE_REPORT': '.jsonl',
              'K3_ENDPOINT_REPORT': '-endpoints.json', 'K3_BASIS_REPORT': '-basis.bin',
              'K3_REVERSE_VECTOR_REPORT': '-vector.bin', 'K3_SEED_ROWS_REPORT': '-seed-rows.bf16',
              'K3_SEED_REPORT': '-seed.json', 'K3_FRUIT_REPORT': '-fruit.json',
              'K3_L1_TRACE': '-compact.jsonl', 'K3_L1_CHECK': '-compact-check.json'}
    for key, suffix in fields.items():
        path = target / (name + suffix)
        assert not path.exists(), path
        env[key] = str(path)
    if compact:
        env['K3_L1_COMPACT'] = '/opt/clover-k3/layer1-map-store-20260929-a/experts.bin'
    for path in work.iterdir():
        if path.is_file() and (path.suffix in ('.h', '.c', '.py') or path.name.startswith('saved-')):
            destination = target / path.name
            if not destination.exists():
                os.link(path, destination)
    with (target / (name + '-launch.json')).open('x') as output:
        json.dump({'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                   'environment': {key: value for key, value in env.items() if key.startswith(('K3_', 'OMP_'))}}, output, indent=2)
    with (target / (name + '.log')).open('xb') as stdout, (target / (name + '.stderr')).open('xb') as stderr:
        completed = subprocess.run([str(binary)], cwd=target, env=env, stdout=stdout, stderr=stderr)
    if completed.returncode:
        print((target / (name + '.stderr')).read_text()[-4000:], flush=True)
        raise RuntimeError('Model exited ' + str(completed.returncode))
    if reference:
        for suffix in ('.bin', '.routes'):
            expected = reference / (name + '-reference' + suffix)
            assert (target / (name + suffix)).read_bytes() == expected.read_bytes(), suffix
            destination = target / expected.name
            if not destination.exists():
                shutil.copyfile(expected, destination)
    if compact:
        checks = json.loads((target / (name + '-compact-check.json')).read_text())
        assert checks['gate'] == 'PASS' and checks['compact_outputs_feed_model']
        print(json.dumps(checks), flush=True)
        print((target / (name + '-compact.jsonl')).read_text().splitlines()[-1], flush=True)
    for suffix in ('-seed.json', '-fruit.json'):
        if (target / (name + suffix)).exists():
            report = json.loads((target / (name + suffix)).read_text())
            assert report['gate'] == 'PASS' and report['guard_probe_sigsegv'], suffix
    print('PASS: fresh model run ' + folder + '/' + name + '; full reference outputs/routes match when supplied.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('folder')
    parser.add_argument('name', choices=PROMPTS)
    parser.add_argument('binary', type=Path)
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    run(args.work, args.folder, args.name, args.binary, args.compact, args.reference)