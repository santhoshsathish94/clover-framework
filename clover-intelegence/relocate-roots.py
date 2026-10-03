"""Move complete roots into the existing dataset, then inputs, outputs and leaves."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import runpy
import time

TOOLS = runpy.run_path(str(Path(__file__).with_name('relocate.py')))
identity = TOOLS['identity']
inventory = TOOLS['inventory']
readers = TOOLS['readers']
PLAN = 'ROOTS-MOVE-PLAN.json'
JOURNAL = 'ROOTS-MOVE-JOURNAL.jsonl'


def journal(destination, record):
    with (destination / JOURNAL).open('a') as output:
        output.write(json.dumps({**record, 'unix': time.time()}) + '\n')
        output.flush()
        os.fsync(output.fileno())


def root_state(source, target):
    if source.is_symlink() or target.is_symlink() or not source.is_dir() or not target.is_dir():
        raise ValueError(f'expected real source and partial destination roots: {source}')
    if {child.name for child in target.iterdir()} != {'observations'}:
        raise ValueError(f'unexpected destination content: {target}')
    observations = target / 'observations'
    if observations.is_symlink() or {child.name for child in observations.iterdir()} != {'france', 'japan'}:
        raise ValueError('partial observation layout differs')
    existing = inventory(target)
    new_files = {}
    links = {}
    for child in sorted(source.rglob('*')):
        relative = child.relative_to(source).as_posix()
        if child.is_symlink():
            if relative not in ('observations/france', 'observations/japan'):
                raise ValueError(f'unreviewed source symlink: {child}')
            if os.readlink(child) != str(target / relative):
                raise ValueError(f'observation alias differs: {child}')
            if child.resolve(strict=True) != (target / relative).resolve(strict=True):
                raise ValueError(f'observation alias does not resolve: {child}')
            links[relative] = os.readlink(child)
        elif child.is_file():
            if relative in existing:
                raise ValueError(f'payload collision: {relative}')
            new_files[relative] = identity(child)
    if set(links) != {'observations/france', 'observations/japan'}:
        raise ValueError('expected two already-relocated observation aliases')
    for filename in ('experts.bin', 'constants.bin'):
        if filename not in new_files:
            raise ValueError(f'missing computed root payload: {source / filename}')
    directories = {f'observations/{name}': identity(observations / name) for name in ('france', 'japan')}
    return new_files, existing, links, directories


def plan(base, destination):
    if not destination.is_dir() or (destination / PLAN).exists():
        raise ValueError('destination missing or roots plan already exists')
    source = base / 'clover-data'
    wanted = [source / f'root-{layer}' for layer in range(1, 93)]
    wanted += [source / name for name in ('inputs', 'outputs', 'leaves.json')]
    active = readers(wanted + [destination / f'root-{layer}' for layer in range(1, 93)])
    if active:
        raise RuntimeError(f'active dataset users, no changes made: {active}')
    device = destination.stat().st_dev
    units = []
    for layer in range(1, 93):
        old = source / f'root-{layer}'
        target = destination / old.name
        files, existing, links, directories = root_state(old, target)
        if old.stat().st_dev != device or any(info[0] != device for info in [*files.values(), *existing.values()]):
            raise ValueError('root move crosses filesystem')
        units.append({'unit': layer, 'group': 'roots', 'source': str(old), 'target': old.name,
                      'source_identity': identity(old), 'files': {**files, **existing},
                      'new_files': files, 'existing_files': existing, 'old_links': links,
                      'existing_directories': directories,
                      'new_bytes': sum(info[2] for info in files.values())})
    for number, name in enumerate(('inputs', 'outputs', 'leaves.json'), 93):
        old = source / name
        if os.path.lexists(destination / name):
            raise ValueError(f'target already exists: {name}')
        files = inventory(old)
        if old.stat().st_dev != device or any(info[0] != device for info in files.values()):
            raise ValueError('tail move crosses filesystem')
        units.append({'unit': number, 'group': name, 'source': str(old), 'target': name,
                      'source_identity': identity(old), 'files': files, 'new_files': files,
                      'existing_files': {}, 'directory': old.is_dir(),
                      'new_bytes': sum(info[2] for info in files.values())})
    summary = {'units': len(units), 'groups': dict(Counter(unit['group'] for unit in units)),
               'additional_files': sum(len(unit['new_files']) for unit in units),
               'additional_bytes': sum(unit['new_bytes'] for unit in units),
               'already_moved_observation_files': sum(len(unit['existing_files']) for unit in units)}
    record = {'created_unix': time.time(), 'base': str(base), 'destination': str(destination),
              'summary': summary, 'units': units,
              'strategy': 'stage partial root, rename full root, transplant observations, alias original path'}
    with (destination / PLAN).open('x') as output:
        json.dump(record, output, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({'gate': 'ROOTS_PLAN_READY', **summary, 'active_users': []}, indent=2))


def check_unit(destination, unit):
    source = Path(unit['source'])
    target = destination / unit['target']
    if not source.is_symlink() or source.resolve(strict=True) != target.resolve(strict=True):
        raise ValueError(f'whole-dataset alias is incorrect: {source}')
    if identity(target) != unit['source_identity']:
        raise ValueError(f'moved root identity differs: {target}')
    if inventory(target) != unit['files']:
        raise ValueError(f'moved payload inventory differs: {target}')
    for relative, expected in unit['files'].items():
        through_old = source if relative == '.' else source / relative
        if identity(through_old) != expected:
            raise ValueError(f'old-path payload differs: {through_old}')
    for relative, expected in unit.get('existing_directories', {}).items():
        if identity(target / relative) != expected:
            raise ValueError(f'previously relocated observations changed: {relative}')


def consolidate(destination, unit):
    source = Path(unit['source'])
    target = destination / unit['target']
    files, existing, links, directories = root_state(source, target)
    if (files, existing, links, directories) != (unit['new_files'], unit['existing_files'],
                                               unit['old_links'], unit['existing_directories']):
        raise ValueError(f'root changed since plan: {source}')
    if identity(source) != unit['source_identity']:
        raise ValueError('root identity changed since plan')
    staging = destination / '.root-move-inflight'
    staging.mkdir(exist_ok=True)
    temporary = staging / unit['target']
    if os.path.lexists(temporary):
        raise ValueError(f'unfinished root transaction needs inspection: {temporary}')
    journal(destination, {'unit': unit['unit'], 'gate': 'STARTED', 'target': unit['target']})
    os.rename(target, temporary)
    source_moved = False
    transplanted = []
    removed_links = []
    try:
        os.rename(source, target)
        source_moved = True
        for relative, old_link in unit['old_links'].items():
            link = target / relative
            if not link.is_symlink() or os.readlink(link) != old_link:
                raise ValueError('existing alias changed during root move')
            link.unlink()
            removed_links.append(relative)
            os.rename(temporary / relative, link)
            transplanted.append(relative)
        os.symlink(str(target), str(source), target_is_directory=True)
        check_unit(destination, unit)
    except BaseException:
        if source_moved:
            if source.is_symlink() and os.readlink(source) == str(target):
                source.unlink()
            for relative in reversed(transplanted):
                os.rename(target / relative, temporary / relative)
            for relative in removed_links:
                os.symlink(unit['old_links'][relative], target / relative, target_is_directory=True)
            if not os.path.lexists(source):
                os.rename(target, source)
            else:
                raise RuntimeError('source occupied during rollback; preserved staged data for inspection')
        if not os.path.lexists(target):
            os.rename(temporary, target)
        journal(destination, {'unit': unit['unit'], 'gate': 'ROLLED_BACK', 'target': unit['target']})
        raise
    (temporary / 'observations').rmdir()
    temporary.rmdir()


def move_simple(destination, unit):
    source = Path(unit['source'])
    target = destination / unit['target']
    if os.path.lexists(target) or identity(source) != unit['source_identity'] or inventory(source) != unit['files']:
        raise ValueError('source changed or destination occupied')
    journal(destination, {'unit': unit['unit'], 'gate': 'STARTED', 'target': unit['target']})
    os.rename(source, target)
    try:
        os.symlink(str(target), source, target_is_directory=unit['directory'])
        check_unit(destination, unit)
    except BaseException:
        if source.is_symlink() and os.readlink(source) == str(target):
            source.unlink()
        if not os.path.lexists(source):
            os.rename(target, source)
        raise


def move(destination, group, limit):
    record = json.loads((destination / PLAN).read_text())
    units = record['units']
    active = readers([Path(unit['source']) for unit in units] + [destination / unit['target'] for unit in units])
    if active:
        raise RuntimeError(f'active dataset users; relocation stopped: {active}')
    completed = set()
    path = destination / JOURNAL
    if path.exists():
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event['gate'] == 'MOVED_AND_VERIFIED':
                completed.add(event['unit'])
    moved = 0
    for unit in units:
        if unit['unit'] in completed:
            check_unit(destination, unit)
            continue
        if unit['group'] != group:
            raise ValueError(f"move {unit['group']} before {group}")
        if Path(unit['source']).is_symlink():
            raise ValueError('unjournaled transaction needs inspection, not automatic overwrite')
        if unit['group'] == 'roots':
            consolidate(destination, unit)
        else:
            move_simple(destination, unit)
        check_unit(destination, unit)
        journal(destination, {'unit': unit['unit'], 'gate': 'MOVED_AND_VERIFIED', 'target': unit['target'],
                              'new_files': len(unit['new_files']), 'preserved_files': len(unit['existing_files']),
                              'additional_bytes': unit['new_bytes']})
        moved += 1
        completed.add(unit['unit'])
        print(f"PASS {unit['unit']}: {unit['target']} ({len(unit['new_files'])} added files, "
              f"{len(unit['existing_files'])} preserved observation files)", flush=True)
        if limit is not None and moved >= limit:
            break
        if all(item['unit'] in completed for item in units if item['group'] == group):
            break
    staging = destination / '.root-move-inflight'
    if staging.exists() and not any(staging.iterdir()):
        staging.rmdir()
    print(json.dumps({'gate': 'GROUP_VERIFIED', 'group': group, 'moved': moved}))


def verify(destination):
    record = json.loads((destination / PLAN).read_text())
    for unit in record['units']:
        check_unit(destination, unit)
    prior = json.loads((destination / 'MOVE-PLAN.json').read_text())
    for unit in prior['units']:
        source, target = Path(unit['source']), destination / unit['target']
        if source.resolve(strict=True) != target.resolve(strict=True):
            raise ValueError('previous relocation compatibility path differs')
        if identity(target) != unit['identity'] or inventory(target) != unit['files']:
            raise ValueError('previously moved payload changed')
    actual = {}
    for unit in prior['units']:
        for relative, info in unit['files'].items():
            path = unit['target'] if relative == '.' else unit['target'] + '/' + relative
            actual[path] = info
    for unit in record['units']:
        for relative, info in unit['files'].items():
            path = unit['target'] if relative == '.' else unit['target'] + '/' + relative
            if path in actual and actual[path] != info:
                raise ValueError('conflicting accumulated inventory')
            actual[path] = info
    summary = {'gate': 'COMPLETE_ROOTS_INPUTS_OUTPUTS_LEAVES_MOVED', **record['summary'],
               'combined_dataset_files': len(actual), 'combined_dataset_bytes': sum(info[2] for info in actual.values()),
               'all_payload_identities_preserved': True, 'old_paths_resolve': True,
               'previous_observation_files_preserved': True, 'verification': 'same-filesystem inode/size/mtime, not a full payload rehash'}
    with (destination / 'ROOTS-RELOCATION.json').open('w') as output:
        json.dump(summary, output, indent=2)
        output.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'move', 'verify'))
    parser.add_argument('--base', type=Path, default=Path('/opt/clover-k3'))
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--group', choices=('roots', 'inputs', 'outputs', 'leaves.json'))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('limit must be positive')
    if args.action == 'plan':
        plan(args.base.resolve(), args.destination.resolve())
    elif args.action == 'move':
        if not args.group:
            parser.error('move requires group')
        move(args.destination.resolve(), args.group, args.limit)
    else:
        verify(args.destination.resolve())