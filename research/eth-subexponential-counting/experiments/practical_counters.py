"""Measured growth of exact model counting at sizes that can actually run, on the paper's own input class.

Three exact counters, cross-checked wherever another one can run (brute force up to n = 22, the search up to n = 40
and on every chain instance):
  brute     all 2^n assignments (numpy), work = 2^n * m
  dpll      branching + unit propagation + connected components + component cache (the idea behind sharpSAT/Cachet),
            work = search nodes
  contract  variable elimination = tensor-network contraction along a greedy min-degree order, counts mod a
            prime p, work = sum of intermediate table sizes; width = largest table (log2)
Instance families (seeded):
  bounded   3-CNF, every variable in <= 3 clauses (the paper's engine input, Sections 2-4)
  random2   random 3-CNF with m = 2n clauses
  chain     clauses on (i, i+1, i+2): treewidth 2
Exponent per variable = least-squares slope of log2(work) against n (brute force = 1 by construction).

    python practical_counters.py [--quick] [--out results/practical_counters.json]
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

P = 2_147_483_629  # prime < 2^31: table products stay inside int64
sys.setrecursionlimit(100_000)


# ------------------------------------------------------------------------------------------------ instances
def bounded_cnf(n, rnd, occ=3):
    cap = [occ] * n
    clauses = []
    for _ in range(n):
        avail = [v for v in range(n) if cap[v] > 0]
        if len(avail) < 3:
            break
        vs = rnd.sample(avail, 3)
        for v in vs:
            cap[v] -= 1
        clauses.append(tuple(sorted((v + 1) * rnd.choice((1, -1)) for v in vs)))
    return clauses


def random_cnf(n, m, rnd):
    return [tuple(sorted((v + 1) * rnd.choice((1, -1)) for v in rnd.sample(range(n), 3))) for _ in range(m)]


def chain_cnf(n, rnd):
    return [tuple(sorted((v + 1) * rnd.choice((1, -1)) for v in (i, i + 1, i + 2))) for i in range(n - 2)]


FAMILIES = {"bounded": lambda n, rnd: bounded_cnf(n, rnd), "random2": lambda n, rnd: random_cnf(n, 2 * n, rnd),
            "chain": lambda n, rnd: chain_cnf(n, rnd)}


# ------------------------------------------------------------------------------------------------ counters
def count_brute(n, clauses):
    total, chunk = 0, 1 << min(n, 20)
    for start in range(0, 1 << n, chunk):
        idx = np.arange(start, start + chunk, dtype=np.int64)
        ok = np.ones(chunk, dtype=bool)
        for cl in clauses:
            sat = np.zeros(chunk, dtype=bool)
            for lit in cl:
                bit = (idx >> (abs(lit) - 1)) & 1
                sat |= bit.astype(bool) if lit > 0 else ~bit.astype(bool)
            ok &= sat
        total += int(ok.sum())
    return total, (1 << n) * max(1, len(clauses))


def count_dpll(n, clauses):
    cache, nodes = {}, [0]

    def assign(cls, lit):
        out = []
        for c in cls:
            if lit in c:
                continue
            if -lit in c:
                c = tuple(x for x in c if x != -lit)
                if not c:
                    return None
            out.append(c)
        return out

    def propagate(cls, lit):
        """Set lit true and unit-propagate: (clauses left, number of variables set) or (None, 0) on a conflict."""
        fixed, todo = set(), [lit]
        while todo:
            u = todo.pop()
            if abs(u) in fixed:
                continue
            fixed.add(abs(u))
            cls = assign(cls, u)
            if cls is None:
                return None, 0
            todo.extend(c[0] for c in cls if len(c) == 1 and abs(c[0]) not in fixed)
        return cls, len(fixed)

    def vars_of(cls):
        return {abs(x) for c in cls for x in c}

    def components(cls):
        parent = {}

        def find(a):
            while parent.setdefault(a, a) != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for c in cls:
            a = find(abs(c[0]))
            for x in c[1:]:
                b = find(abs(x))
                if a != b:
                    parent[b] = a
        groups = {}
        for c in cls:
            groups.setdefault(find(abs(c[0])), []).append(c)
        return list(groups.values())

    def count(cls):  # solutions over vars_of(cls)
        nodes[0] += 1
        if not cls:
            return 1
        key = frozenset(cls)
        if key in cache:
            return cache[key]
        comps = components(cls)
        if len(comps) > 1:
            res = 1
            for comp in comps:
                res *= count(comp)
                if res == 0:
                    break
        else:
            vs = vars_of(cls)
            occ = {}
            for c in cls:
                for x in c:
                    occ[abs(x)] = occ.get(abs(x), 0) + 1
            v = max(occ, key=occ.get)
            res = 0
            for lit in (v, -v):
                sub, fixed = propagate(cls, lit)
                if sub is None:
                    continue
                free = len(vs) - fixed - len(vars_of(sub))
                res += count(sub) << free
        cache[key] = res
        return res

    used = {abs(x) for c in clauses for x in c}
    return count(list(set(clauses))) << (n - len(used)), nodes[0]


def count_contract(n, clauses, max_width=24):
    factors = []
    for cl in set(clauses):
        scope = tuple(sorted({abs(x) for x in cl}))
        t = np.zeros((2,) * len(scope), dtype=np.int64)
        for a in np.ndindex(*t.shape):
            val = dict(zip(scope, a))
            t[a] = 1 if any(val[abs(x)] == (1 if x > 0 else 0) for x in cl) else 0
        factors.append((scope, t))
    used = {v for s, _ in factors for v in s}
    work, width = 0, 0
    live = set(used)
    while live:
        nb = {v: set() for v in live}
        for s, _ in factors:
            for v in s:
                nb[v].update(s)
        v = min(live, key=lambda u: (len(nb[u]), u))
        mine = [f for f in factors if v in f[0]]
        factors = [f for f in factors if v not in f[0]]
        union = tuple(sorted({u for s, _ in mine for u in s}))
        if len(union) > max_width:
            return None, work, len(union)
        width = max(width, len(union))
        prod = np.ones((2,) * len(union), dtype=np.int64)
        for s, t in mine:
            perm = np.transpose(t, [s.index(u) for u in union if u in s])
            shape = [2 if u in s else 1 for u in union]
            prod = (prod * perm.reshape(shape)) % P
        work += prod.size
        out = prod.sum(axis=union.index(v)) % P
        factors.append((tuple(u for u in union if u != v), out))
        live.discard(v)
    res = 1
    for s, t in factors:
        res = (res * int(t)) % P
    return (res * pow(2, n - len(used), P)) % P, work, width


# ------------------------------------------------------------------------------------------------ experiment
def slope(ns, vals):
    if len(ns) < 3:
        return None
    x, y = np.array(ns, float), np.log2(np.array(vals, float))
    return float(np.polyfit(x, y, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=float, default=60.0, help="seconds per (family, method, n) before stopping")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results" / "practical_counters.json"))
    args = ap.parse_args()
    seeds = 2 if args.quick else args.seeds
    plans = {"brute": range(8, 19 if args.quick else 23, 2),
             "dpll": range(10, 61 if args.quick else 301, 10),
             "contract": range(10, 61 if args.quick else 301, 10)}
    results, checks = {}, 0
    for fam, make in FAMILIES.items():
        for meth, ns in plans.items():
            rows = []
            for n in ns:
                works, secs, widths, counts, fails = [], [], [], [], 0
                for s in range(seeds):
                    cls = make(n, random.Random(1000 * n + s))
                    t0 = time.perf_counter()
                    if meth == "brute":
                        c, w = count_brute(n, cls)
                        width = None
                    elif meth == "dpll":
                        c, w = count_dpll(n, cls)
                        width = None
                    else:
                        c, w, width = count_contract(n, cls)
                        if c is None:
                            fails += 1
                            continue
                    secs.append(time.perf_counter() - t0)
                    works.append(w)
                    widths.append(width)
                    counts.append(c)
                    # cross-check against the other exact counters on the same instance
                    if meth == "contract":
                        if n <= 22:
                            assert c == count_brute(n, cls)[0] % P, (fam, n, s, "contract != brute")
                            checks += 1
                        if n <= 40 or fam == "chain":
                            assert c == count_dpll(n, cls)[0] % P, (fam, n, s, "contract != dpll")
                            checks += 1
                    if meth == "dpll" and n <= 22:
                        assert c == count_brute(n, cls)[0], (fam, n, s, "dpll != brute")
                        checks += 1
                    if sum(secs) > args.budget:
                        break
                if not works:
                    rows.append(dict(n=n, skipped=f"width > 24 on {fails}/{seeds} seeds"))
                    break
                med = sorted(works)[len(works) // 2]
                row = dict(n=n, median_work=med, median_seconds=round(sorted(secs)[len(secs) // 2], 4),
                           seeds=len(works))
                if meth == "contract":
                    row["median_width"] = sorted(widths)[len(widths) // 2]
                    if fails:
                        row["too_wide"] = fails
                rows.append(row)
                print(f"{fam:8} {meth:8} n={n:4}  work {med:>14,}  {row['median_seconds']:8.3f} s"
                      + (f"  width {row['median_width']}" if meth == "contract" else ""), flush=True)
                if sum(secs) > args.budget:
                    break
            done = [r for r in rows if "median_work" in r]
            results.setdefault(fam, {})[meth] = dict(rows=rows, exponent_per_variable=slope(
                [r["n"] for r in done], [r["median_work"] for r in done]))
    print("\nexponent per variable (slope of log2 work vs n; brute force = 1):")
    for fam, ms in results.items():
        print(f"  {fam:8} " + "  ".join(f"{m} {v['exponent_per_variable'] if v['exponent_per_variable'] is None else round(v['exponent_per_variable'], 3)}"
                                         for m, v in ms.items()))
    print(f"cross-checks passed: {checks}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(machine="i7-1355U, Python " + sys.version.split()[0], prime=P,
                                              cross_checks=checks, results=results), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
