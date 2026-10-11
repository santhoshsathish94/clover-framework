#!/usr/bin/env python3
"""A catalogue of every (owner, stage) unit in the fleet.

One row per distinct unit. The point is to see what kinds of unit exist and how many
of each, before anything is concluded from any single one. Counts and measured times
only.
"""
import collections

NAMES = ["france", "japan", "onthe", "alpha", "beta"]
MLA = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92}


def kind(owner):
    if owner == 93:
        return "tail"
    if owner == 1:
        return "KDA-gen"
    return "MLA" if owner in MLA else "KDA"


def load(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 6:
                continue
            owner, local, position, nanos, digest, detail = parts
            owner, local, position = int(owner), int(local), int(position)
            if local == 120 and position > 0:
                position -= 1
            rows.append((owner, local, position, float(nanos), digest, int(detail)))
    return rows


def main():
    everything = {}
    for name in NAMES:
        try:
            everything[name] = load(f"/tmp/trace-{name}.csv")
        except FileNotFoundError:
            pass

    print("=== how many distinct units exist ===")
    units = set()
    per_kind = collections.defaultdict(set)
    for rows in everything.values():
        for owner, local, *_ in rows:
            units.add((owner, local))
            per_kind[kind(owner)].add(local)
    print(f"distinct (owner, stage) pairs observed: {len(units)}")
    for k in sorted(per_kind):
        print(f"  {k:8s}: {len(per_kind[k])} distinct stage numbers")

    print()
    print("=== do all owners of the same kind run the same stage numbers? ===")
    by_kind = collections.defaultdict(dict)
    for rows in everything.values():
        for owner, local, *_ in rows:
            by_kind[kind(owner)].setdefault(owner, set()).add(local)
    for k in sorted(by_kind):
        shapes = {frozenset(v) for v in by_kind[k].values()}
        print(f"  {k:8s}: {len(by_kind[k])} owners, {len(shapes)} distinct stage sets")

    print()
    print("=== which stages write the residual, by owner kind ===")
    rows = everything["france"]
    seq = collections.defaultdict(list)
    for owner, local, position, _n, digest, _d in rows:
        seq[(owner, position)].append((local, digest))
    writers = collections.defaultdict(collections.Counter)
    seen = collections.defaultdict(collections.Counter)
    for (owner, position), items in seq.items():
        previous = None
        for local, digest in items:
            seen[kind(owner)][local] += 1
            if previous is not None and digest != previous:
                writers[kind(owner)][local] += 1
            previous = digest
    for k in sorted(seen):
        always = sorted(s for s in seen[k] if writers[k].get(s, 0) == seen[k][s])
        never = sorted(s for s in seen[k] if writers[k].get(s, 0) == 0)
        sometimes = sorted(s for s in seen[k] if 0 < writers[k].get(s, 0) < seen[k][s])
        print(f"  {k:8s} of {len(seen[k])} stages: {len(never)} never write, "
              f"{len(sometimes)} sometimes, {len(always)} always")
        print(f"           sometimes {sometimes}   always {always}")

    print()
    print("=== time per unit, grouped by stage number and owner kind ===")
    cost = collections.defaultdict(float)
    hits = collections.Counter()
    for owner, local, _p, nanos, _h, _d in rows:
        cost[(kind(owner), local)] += nanos
        hits[(kind(owner), local)] += 1
    total = sum(cost.values())
    print(f"total traced {total/1e9:.3f} s")
    print(f"  {'kind':8s} {'stage':>5s} {'runs':>6s} {'seconds':>9s} {'share':>7s}")
    for key, nanos in sorted(cost.items(), key=lambda kv: -kv[1])[:12]:
        k, local = key
        print(f"  {k:8s} {local:5d} {hits[key]:6d} {nanos/1e9:9.3f} {100*nanos/total:6.2f}%")

    print()
    print("=== aggregate by owner kind ===")
    by = collections.defaultdict(float)
    runs = collections.Counter()
    for owner, local, _p, nanos, _h, _d in rows:
        by[kind(owner)] += nanos
        runs[kind(owner)] += 1
    for k in sorted(by):
        print(f"  {k:8s} {runs[k]:7d} stage runs  {by[k]/1e9:8.3f} s  {100*by[k]/total:6.2f}%")


if __name__ == "__main__":
    main()
