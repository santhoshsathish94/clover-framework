"""Relocate verified resident datasets one at a time, preserving old-path access."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
import time


def identity(path):
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def inventory(path):
    if path.is_symlink():
        raise ValueError(f'source must be a real file or directory: {path}')
    files = {}
    if path.is_file():
        files['.'] = identity(path)
    else:
        for child in sorted(path.rglob('*')):
            if child.is_symlink():
                raise ValueError(f'nested symlink needs separate review: {child}')
            if child.is_file():
                files[child.relative_to(path).as_posix()] = identity(child)
    if not files:
        raise ValueError(f'empty dataset: {path}')
    return files


def global_files(index):
    raw = index.read_bytes()
    if struct.unpack_from('<4sII', raw) != (b'K3EQ', 93, 37):
        raise ValueError('unexpected resident index identity')
    cursor = 12 + 93 * 37 * 32
    count = struct.unpack_from('<I', raw, cursor)[0]
    cursor += 4
    if count != 5:
        raise ValueError('unexpected global tensor count')
    records = []
    for record in range(count):
        records.append(struct.unpack_from('<iqqiii', raw, cursor))
        cursor += 32
    count = struct.unpack_from('<I', raw, cursor)[0]
    cursor += 4
    paths = []
    for file in range(count):
        length = struct.unpack_from('<I', raw, cursor)[0]
        cursor += 4
        paths.append(Path(raw[cursor:cursor + length].decode()))
        cursor += length
    used = set()
    for file, offset, length, rows, columns, dtype in records:
        if not 0 <= file < len(paths) or offset < 0 or length <= 0:
            raise ValueError('invalid global tensor bounds')
        if offset + length > paths[file].stat().st_size:
            raise ValueError('global tensor file is truncated')
        used.add(paths[file])
    return sorted(used)


def readers(sources):
    roots = [str(path) for path in sources]
    active = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            arguments = (process / 'cmdline').read_bytes().split(bytes([0]))
            campaign = any(b'/opt/clover-k3/prepared-trunks-20261001-a/' in argument and
                           argument.endswith(b'.py') for argument in arguments)
            references = set()
            for line in (process / 'maps').read_text(errors='replace').splitlines():
                parts = line.split(maxsplit=5)
                if len(parts) == 6:
                    references.add(parts[5])
            for descriptor in (process / 'fd').iterdir():
                try:
                    references.add(os.readlink(descriptor))
                except FileNotFoundError:
                    continue
            if campaign or any(target == root or target.startswith(root + '/')
                               for target in references for root in roots):
                active.append({'pid': int(process.name), 'executable': os.readlink(process / 'exe'),
                               'campaign': campaign})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return active


def plan(base, destination):
    if destination.exists():
        raise ValueError('destination already exists; inspect or resume the existing plan')
    data = base / 'clover-data'
    work = base / 'clover-k3-resident-20261002-a'
    build = json.loads((work / 'build.json').read_text())
    for filename, key in (('clover-k3.c', 'source_sha256'), ('clover-k3', 'binary_sha256')):
        if hashlib.sha256((work / filename).read_bytes()).hexdigest() != build[key]:
            raise ValueError('resident code differs from inspected build')
    candidates = [('index', base / 'build/eqidx.bin', Path('eqidx.bin'))]
    candidates += [('prepared_trunks', data / f'trunk-{layer}', Path(f'trunk-{layer}')) for layer in range(93)]
    candidates += [('derived_operators', data / 'operators/trunk-0-qkv', Path('operators/trunk-0-qkv'))]
    candidates += [('derived_operators', data / f'operators/qkv-all/layer-{layer}',
                    Path(f'operators/qkv-all/layer-{layer}')) for layer in range(93)]
    candidates += [('expert_observations', data / f'root-{layer}/observations/{name}',
                    Path(f'root-{layer}/observations/{name}'))
                   for layer in range(1, 93) for name in ('france', 'japan')]
    candidates += [('global_tensors', path, Path('model') / path.name)
                   for path in global_files(base / 'build/eqidx.bin')]
    active = readers([source for group, source, target in candidates])
    if active:
        raise RuntimeError(f'dataset users are active; no process changed: {active}')
    units = []
    device = destination.parent.stat().st_dev
    for number, (group, source, target) in enumerate(candidates, 1):
        files = inventory(source)
        if any(info[0] != device for info in files.values()) or source.stat().st_dev != device:
            raise ValueError('cross-filesystem move requires a separate plan')
        units.append({'unit': number, 'group': group, 'source': str(source),
                      'target': target.as_posix(), 'directory': source.is_dir(),
                      'identity': identity(source), 'files': files,
                      'bytes': sum(info[2] for info in files.values())})
    destination.mkdir()
    record = {'created_unix': time.time(), 'base': str(base), 'destination': str(destination),
              'resident_source_sha256': build['source_sha256'], 'resident_binary_sha256': build['binary_sha256'],
              'groups': dict(Counter(unit['group'] for unit in units)), 'units': units,
              'total_bytes': sum(unit['bytes'] for unit in units), 'compatibility_links': True}
    with (destination / 'MOVE-PLAN.json').open('x') as output:
        json.dump(record, output, indent=2)
        output.write('\n')
    print(json.dumps({'gate': 'PLAN_READY', 'units': len(units), 'groups': record['groups'],
                      'bytes': record['total_bytes'], 'active_users': []}, indent=2))


def check_unit(destination, unit):
    source, target = Path(unit['source']), destination / unit['target']
    aliased = any(path.is_symlink() for path in (source, *source.parents))
    if not aliased or source.resolve(strict=True) != target.resolve(strict=True):
        raise ValueError(f'compatibility link missing or incorrect: {source}')
    if identity(target) != unit['identity'] or inventory(target) != unit['files']:
        raise ValueError(f'dataset identity changed: {target}')
    for relative, expected in unit['files'].items():
        path = source if relative == '.' else source / relative
        if identity(path) != expected:
            raise ValueError(f'old path resolves to different file: {path}')


def move(destination, group, limit):
    record = json.loads((destination / 'MOVE-PLAN.json').read_text())
    units = [unit for unit in record['units'] if group is None or unit['group'] == group]
    pending = [unit for unit in units if not Path(unit['source']).is_symlink()]
    if limit is not None:
        pending = pending[:limit]
    active = readers([Path(unit['source']) for unit in record['units']])
    if active:
        raise RuntimeError(f'active dataset users; move stopped: {active}')
    moved = 0
    for unit in pending:
        source, target = Path(unit['source']), destination / unit['target']
        if os.path.lexists(target):
            raise ValueError(f'target already exists: {target}')
        if identity(source) != unit['identity'] or inventory(source) != unit['files']:
            raise ValueError(f'source changed since plan: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.parent.stat().st_dev != source.stat().st_dev:
            raise ValueError('move crossed filesystem boundary')
        os.rename(source, target)
        try:
            os.symlink(str(target), str(source), target_is_directory=unit['directory'])
            check_unit(destination, unit)
        except BaseException:
            if source.is_symlink() and source.resolve() == target.resolve():
                source.unlink()
            if not os.path.lexists(source):
                os.rename(target, source)
            raise
        with (destination / 'MOVE-JOURNAL.jsonl').open('a') as journal:
            journal.write(json.dumps({'unit': unit['unit'], 'group': group, 'target': unit['target'],
                                      'gate': 'MOVED_AND_VERIFIED', 'bytes': unit['bytes'],
                                      'files': len(unit['files']), 'unix': time.time()}) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        moved += 1
        print(f"PASS {unit['unit']}: {unit['target']} ({len(unit['files'])} files)", flush=True)
    for unit in units:
        if Path(unit['source']).is_symlink():
            check_unit(destination, unit)
    print(json.dumps({'gate': 'MOVE_BATCH_VERIFIED', 'group': group, 'moved_this_batch': moved}))


def verify(destination):
    record = json.loads((destination / 'MOVE-PLAN.json').read_text())
    for unit in record['units']:
        check_unit(destination, unit)
    globals = global_files(destination / 'eqidx.bin')
    if not all(path.resolve().is_relative_to(destination) for path in globals):
        raise ValueError('global tensor binding still resolves outside destination')
    report = {'gate': 'ALL_DATASETS_MOVED', 'units': len(record['units']),
              'files': sum(len(unit['files']) for unit in record['units']),
              'bytes': record['total_bytes'], 'groups': record['groups'],
              'payload_identities_preserved': True, 'all_old_paths_resolve': True,
              'global_index_targets_resolve_inside_dataset': True}
    with (destination / 'RELOCATION.json').open('w') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'move', 'verify'))
    parser.add_argument('--base', type=Path, default=Path('/opt/clover-k3'))
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--group', choices=('index', 'prepared_trunks', 'derived_operators', 'expert_observations', 'global_tensors'))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    if args.action == 'plan':
        plan(args.base.resolve(), args.destination.absolute())
    elif args.action == 'move':
        move(args.destination.absolute(), args.group, args.limit)
    else:
        verify(args.destination.absolute())