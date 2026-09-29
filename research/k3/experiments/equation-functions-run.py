#!/usr/bin/env python3
"""Exercise named equation functions with unchanged cumulative model gates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(base, names):
    previous = Path('/opt/clover-k3/layer1-compact-model-20260929-a')
    model = base / 'model'
    output = base / 'enabled'
    output.mkdir(exist_ok=True)
    for path in model.iterdir():
        if path.is_file() and (path.suffix in ('.c', '.h', '.py') or path.name.startswith('saved-')):
            destination = output / path.name
            if not destination.exists():
                shutil.copyfile(path, destination)
    for name in names:
        original_launch = json.loads((previous / 'enabled' / (name + '-launch.json')).read_text())
        settings = original_launch['environment']
        env = {key: value for key, value in os.environ.items() if not key.startswith('K3_')}
        for key, value in settings.items():
            env[key] = str(output / Path(value).name) if str(previous / 'enabled') in value else value
        env['K3_FUNCTIONS_REPORT'] = str(output / (name + '-functions.json'))
        for key in ('K3_LOGITS', 'K3_L1_TRACE', 'K3_FUNCTIONS_REPORT'):
            assert not Path(env[key]).exists(), key
        launch = {'binary_sha256': hashlib.sha256((model / 'candidate').read_bytes()).hexdigest(),
                  'environment': {key: value for key, value in env.items() if key.startswith(('K3_', 'OMP_'))}}
        with (output / (name + '-launch.json')).open('x') as stream:
            json.dump(launch, stream, indent=2)
        with (output / (name + '.log')).open('xb') as stdout, (output / (name + '.stderr')).open('xb') as stderr:
            completed = subprocess.run([str(model / 'candidate')], cwd=output, env=env, stdout=stdout, stderr=stderr)
        if completed.returncode:
            print((output / (name + '.stderr')).read_text()[-4000:], flush=True)
            raise RuntimeError('Named equation model failed: ' + str(completed.returncode))
        for suffix in ('.bin', '.routes'):
            expected = previous / (name + '-reference' + suffix)
            assert (output / (name + suffix)).read_bytes() == expected.read_bytes(), suffix
            shutil.copyfile(expected, output / expected.name)
        report = json.loads((output / (name + '-functions.json')).read_text())
        assert report['gate'] == 'PASS' and report['seed_calls'] == 5 and report['fruit_calls'] == 2
        assert report['root_calls'] == 74 and report['root_positions'] == 80
        assert report['root_stage_values'] == [245760, 245760, 245760, 286720]
        assert report['root_weight_array_bytes'] == 0
        print(json.dumps({'prompt': name, 'full_output_and_routes_exact': True, **report}), flush=True)
    subprocess.run([sys.executable, str(model / 'reverse-check.py'), str(output), '--stage', '1', '--runs', *names,
                    '--output', str(output / ('prior-' + '-'.join(names) + '.json'))], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('names', choices=('france', 'japan'), nargs='+')
    args = parser.parse_args()
    run(args.base, args.names)