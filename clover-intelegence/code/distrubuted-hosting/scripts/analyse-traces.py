#!/usr/bin/env python3
"""Describe what the traces contain. No recommendations: counts, identity tests and
distributions only, so the data can be looked at before anything is decided."""
import collections
import glob
import os
import sys

NAMES = ["france", "japan", "onthe", "alpha", "beta"]


def load(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 6:
                continue
            owner, local, position, nanos, digest, detail = parts
            owner, local, position = int(owner), int(local), int(position)
            # Stage 120 increments the position counter before the trace is written,
            # so it is reported one position later than the one it actually finished.
            if local == 120 and position > 0:
                position -= 1
            rows.append((owner, local, position, float(nanos), digest, int(detail)))
    return rows


def main():
    traces = {}
    for name in NAMES:
        path = f"/tmp/trace-{name}.csv"
        if os.path.exists(path):
            traces[name] = load(path)

    if not traces:
        print("no traces found")
        return

    print("=== 1. size of each capture ===")
    for name, rows in traces.items():
        owners = sorted({r[0] for r in rows})
        positions = sorted({r[2] for r in rows})
        print(f"{name:8s} rows {len(rows):8d}  owners {min(owners)}..{max(owners)} ({len(owners)})  positions {positions}")

    print()
    print("=== 2. stages executed per owner, per position ===")
    for name, rows in traces.items():
        counts = collections.Counter((r[0], r[2]) for r in rows)
        per_owner = collections.defaultdict(set)
        for (owner, position), total in counts.items():
            per_owner[owner].add(total)
        spread = {owner: sorted(values) for owner, values in per_owner.items()}
        distinct = {tuple(v) for v in spread.values()}
        print(f"{name:8s} distinct per-position stage counts across owners: {sorted(distinct)}")

    print()
    print("=== 3. is the stage path identical across positions and prompts? ===")
    paths = {}
    for name, rows in traces.items():
        by_key = collections.defaultdict(list)
        for owner, local, position, _n, _h, _d in rows:
            by_key[(name, owner, position)].append(local)
        for (pname, owner, position), seq in by_key.items():
            paths.setdefault(owner, {}).setdefault(tuple(seq), []).append((pname, position))
    same_everywhere = 0
    differing = []
    for owner, variants in sorted(paths.items()):
        if len(variants) == 1:
            same_everywhere += 1
        else:
            differing.append((owner, {len(p): len(v) for p, v in variants.items()}))
    print(f"owners whose executed stage path is byte-identical for every position of every prompt: {same_everywhere}/{len(paths)}")
    for owner, sizes in differing[:6]:
        print(f"  owner {owner}: path lengths -> occurrences {sizes}")

    print()
    print("=== 4. the path itself, one owner of each kind ===")
    for owner in sorted(paths):
        if owner in (1, 3, 46, 93):
            for variant in sorted(paths[owner], key=len):
                print(f"owner {owner}: {len(variant)} stages, seen {len(paths[owner][variant])} times")
                print(f"  {list(variant)}")

    print()
    print("=== 5. stages after which the residual hash was unchanged ===")
    for name, rows in traces.items():
        by_key = collections.defaultdict(list)
        for owner, local, position, _n, digest, _d in rows:
            by_key[(owner, position)].append((local, digest))
        unchanged = collections.Counter()
        total = collections.Counter()
        for seq in by_key.values():
            previous = None
            for local, digest in seq:
                total[local] += 1
                if previous is not None and digest == previous:
                    unchanged[local] += 1
                previous = digest
        always = sorted(s for s in total if unchanged.get(s, 0) == total[s])
        never = sorted(s for s in total if unchanged.get(s, 0) == 0)
        print(f"{name:8s} stages that never altered the residual: {len(always)} of {len(total)}")
        print(f"         always unchanged: {always}")
        print(f"         always changed  : {never}")
        break

    print()
    print("=== 6. expert selections ===")
    for name, rows in traces.items():
        per_layer = collections.defaultdict(set)
        per_layer_position = collections.defaultdict(set)
        for owner, local, position, _n, _h, detail in rows:
            if detail >= 0 and 1 <= owner <= 92:
                per_layer[owner].add(detail)
                per_layer_position[(owner, position)].add(detail)
        if not per_layer:
            print(f"{name:8s} no expert details recorded")
            continue
        distinct = [len(v) for v in per_layer.values()]
        per_pos = [len(v) for v in per_layer_position.values()]
        print(f"{name:8s} layers {len(per_layer)}  distinct experts per layer: min {min(distinct)} max {max(distinct)} mean {sum(distinct)/len(distinct):.1f}"
              f"   per position: min {min(per_pos)} max {max(per_pos)} mean {sum(per_pos)/len(per_pos):.1f}")

    print()
    print("=== 7. do different prompts choose the same experts? ===")
    chosen = {}
    for name, rows in traces.items():
        per_layer = collections.defaultdict(set)
        for owner, local, position, _n, _h, detail in rows:
            if detail >= 0 and 1 <= owner <= 92:
                per_layer[owner].add(detail)
        chosen[name] = per_layer
    names = list(chosen)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = chosen[names[i]], chosen[names[j]]
            layers = sorted(set(a) & set(b))
            if not layers:
                continue
            overlaps = [len(a[l] & b[l]) / max(1, len(a[l] | b[l])) for l in layers]
            shared = [len(a[l] & b[l]) for l in layers]
            print(f"{names[i]:7s} vs {names[j]:7s}  mean Jaccard {sum(overlaps)/len(overlaps):.3f}"
                  f"   shared experts per layer: min {min(shared)} max {max(shared)} mean {sum(shared)/len(shared):.1f}")

    print()
    print("=== 8. within one prompt, do later positions reuse earlier experts? ===")
    for name, rows in traces.items():
        per_layer_position = collections.defaultdict(set)
        for owner, local, position, _n, _h, detail in rows:
            if detail >= 0 and 1 <= owner <= 92:
                per_layer_position[(owner, position)].add(detail)
        positions = sorted({p for (_o, p) in per_layer_position})
        if len(positions) < 2:
            continue
        reuse = []
        for owner in sorted({o for (o, _p) in per_layer_position}):
            seen = set()
            for p in positions:
                picks = per_layer_position.get((owner, p), set())
                if seen and picks:
                    reuse.append(len(picks & seen) / len(picks))
                seen |= picks
        if reuse:
            print(f"{name:8s} mean share of a position's experts already used by an earlier position: {sum(reuse)/len(reuse):.3f}")

    print()
    print("=== 9. time by stage number, one prompt ===")
    name = next(iter(traces))
    by_stage = collections.defaultdict(float)
    hits = collections.Counter()
    for owner, local, position, nanos, _h, _d in traces[name]:
        by_stage[local] += nanos
        hits[local] += 1
    total = sum(by_stage.values())
    print(f"{name}: total traced {total/1e9:.3f} s across {sum(hits.values())} stage executions")
    top = sorted(by_stage.items(), key=lambda kv: -kv[1])[:15]
    print("  stage  executions    seconds   share")
    for local, nanos in top:
        print(f"  {local:5d}  {hits[local]:10d}  {nanos/1e9:9.3f}  {100*nanos/total:5.1f}%")


if __name__ == "__main__":
    main()
