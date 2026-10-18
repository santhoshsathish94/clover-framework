#!/usr/bin/env python3
"""Every operator invocation in a five-in five-out run, grouped by layer and stage.

Columns written by stg_emit:
  op seq layer label expert T in out n min max mean rms nz
A matmul of T positions, `in` wide, `out` tall costs 2*T*in*out flops.
"""
import collections

rows = []
with open("/tmp/trace55/stage.tsv") as handle:
    for line in handle:
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        rows.append({
            "op": f[0], "L": int(f[2]), "label": f[3], "expert": int(f[4]),
            "T": int(f[5]), "in": int(f[6]), "out": int(f[7]),
        })

print(f"total invocations recorded: {len(rows)}")
print()

by_op = collections.Counter(r["op"] for r in rows)
print("invocations by operator:")
for op, n in by_op.most_common():
    fl = sum(2 * r["T"] * r["in"] * r["out"] for r in rows if r["op"] == op)
    print(f"  {op:4s} {n:6d} calls   {fl/1e9:10.3f} Gflop")
print()

layers = sorted({r["L"] for r in rows})
print(f"layers touched: {len(layers)}  ({layers[0]}..{layers[-1]})")
print()

# One representative layer of each kind, listing every matrix it applies.
def kind(L):
    if L == 0:
        return "layer 0 (dense)"
    if L == 93:
        return "tail"
    return "MLA" if (L == 92 or L % 4 == 3) else "KDA"

for probe in (0, 1, 3, 93):
    sel = [r for r in rows if r["L"] == probe]
    if not sel:
        continue
    print(f"--- layer {probe}: {kind(probe)}, {len(sel)} invocations ---")
    agg = collections.OrderedDict()
    for r in sel:
        key = (r["op"], r["label"], r["in"], r["out"])
        a = agg.setdefault(key, {"calls": 0, "pos": 0, "flop": 0})
        a["calls"] += 1
        a["pos"] += r["T"]
        a["flop"] += 2 * r["T"] * r["in"] * r["out"]
    for (op, label, i, o), a in agg.items():
        print(f"  {op:3s} {label:12s} {i:6d} -> {o:6d}  calls {a['calls']:3d}"
              f"  positions {a['pos']:3d}  {a['flop']/1e9:8.3f} Gflop")
    print()

# Totals per layer kind, to show where the work actually sits.
per_kind = collections.defaultdict(lambda: {"calls": 0, "flop": 0, "layers": set()})
for r in rows:
    k = kind(r["L"])
    per_kind[k]["calls"] += 1
    per_kind[k]["flop"] += 2 * r["T"] * r["in"] * r["out"]
    per_kind[k]["layers"].add(r["L"])
print("by layer kind:")
total = sum(v["flop"] for v in per_kind.values())
for k, v in sorted(per_kind.items(), key=lambda x: -x[1]["flop"]):
    print(f"  {k:16s} {len(v['layers']):3d} layers  {v['calls']:6d} calls"
          f"  {v['flop']/1e9:10.3f} Gflop  {100*v['flop']/total:5.1f}%")
print(f"  {'TOTAL':16s} {len(layers):3d} layers  {len(rows):6d} calls  {total/1e9:10.3f} Gflop")
