#!/usr/bin/env python3
"""Create an isolated, minimally changed model with seed/root/fruit entry points."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace(path, old, new, count=1):
    text = path.read_text()
    assert text.count(old) == count, (path.name, old, text.count(old))
    path.write_text(text.replace(old, new))


def prepare(source, target):
    prior = json.loads((source / 'results.json').read_text())
    assert prior['gate'] == 'PASS'
    for name, expected in prior['implementation_and_evidence_sha256'].items():
        assert digest(source / name) == expected, name
    model = target / 'model'
    model.mkdir()
    for path in source.iterdir():
        if path.is_file() and (path.suffix in ('.c', '.h', '.py') or path.name.startswith('saved-')):
            shutil.copyfile(path, model / path.name)
    (model / 'include').mkdir()
    for name in ('root-function.h', 'root-function.c', 'seed-function.h', 'equation-functions-model.h'):
        shutil.copyfile(target / name, model / name)
    replace(model / 'seed-model.h', '#include "seed-reader.h"', '#include "seed-reader.h"\n#include "seed-function.h"')
    replace(model / 'seed-model.h', 'seed_read_row(&input, (unsigned)ids[position], seed_model_rows[position]);',
            'seed(&input, (unsigned)ids[position], seed_model_rows[position]);')
    replace(model / 'fruit-head.h', 'static void fruit_project(', 'void fruit(')
    for name in ('candidate.c', 'vector-endpoints.h'):
        replace(model / name, 'fruit_project(', 'fruit(')
    replace(model / 'candidate.c', '#include "layer1-compact-model.h"', '#include "equation-functions-model.h"')
    replace(model / 'candidate.c', '    l1_model_finish();', '    equation_functions_report();')
    replace(model / 'candidate.c', '                            I_, LAT, L, e, 2, mt, mj);',
            '                            I_, LAT, L, e, 2, mt, mj);\n                        root_checked(L, e, m, zin, go, dgo);')
    previous_command = json.loads((source / 'compile-command.json').read_text())
    includes = [flag.replace(str(source), str(model)) for flag in previous_command if flag.startswith('-I')]
    common = ['gcc', '-O3', '-march=native', '-ffp-contract=off', '-fopenmp', '-DNPOS=5']
    for name in ('layer1-compact', 'root-function'):
        subprocess.run([*common, '-std=c11', '-Wall', '-Wextra', '-Werror', *includes,
                        '-c', name + '.c', '-o', name + '.o'], cwd=model, check=True)
    command = [flag.replace(str(source), str(model)) for flag in previous_command]
    command.insert(command.index('layer1-compact.o') + 1, 'root-function.o')
    subprocess.run(command, cwd=model, check=True)
    symbols = subprocess.check_output(['nm', '--defined-only', str(model / 'candidate')], text=True)
    for name in ('seed', 'root', 'fruit'):
        assert any(line.split()[-2:] == ['T', name] for line in symbols.splitlines()), name
    changed = [name for name in ('candidate.c', 'seed-model.h', 'fruit-head.h', 'vector-endpoints.h')]
    report = {'gate': 'COMPILED', 'source_report_sha256': digest(source / 'results.json'),
              'changed_model_files': changed, 'command': command,
              'model_sha256': {path.name: digest(path) for path in model.iterdir() if path.is_file()},
              'entry_points': ['seed', 'root', 'fruit']}
    with (target / 'prepare.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'gate': 'COMPILED', 'entry_points': report['entry_points'], 'changed_model_files': changed}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    prepare(args.source, args.target)