"""Cross-run patterns in the laptop's benchmark records (bench_runs.jsonl): tasks that fail in every Omni configuration,
tasks that pass in every one, flips, and what the retry runs did. Usage: python bench_patterns.py"""
import collections
import json

lines = open("bench_runs.jsonl", encoding="utf-8").read().splitlines()
runs = {}
for l in lines[1:]:
    r = json.loads(l)
    runs[r["run"]] = r["rows"]
omni = [k for k in runs if k.startswith("layers-omni\\step")]
print("Omni configuration runs:", len(omni), omni)
vec = collections.defaultdict(dict)
cat = {}
for k in omni:
    for row in runs[k]:
        if row["passed"] is not None:
            vec[row["task_id"]][k] = bool(row["passed"])
            cat[row["task_id"]] = row["category"]
scored = {t: v for t, v in vec.items() if len(v) == len(omni)}
always_fail = sorted(t for t, v in scored.items() if not any(v.values()))
always_pass = sorted(t for t, v in scored.items() if all(v.values()))
flip = sorted(t for t, v in scored.items() if 0 < sum(v.values()) < len(v))
print(f"tasks scored in all {len(omni)} Omni runs: {len(scored)}; fail in all: {len(always_fail)}; pass in all: {len(always_pass)}; mixed: {len(flip)}")
print("fail in all, by category:", dict(collections.Counter(cat[t] for t in always_fail)))
print("mixed, by category:", dict(collections.Counter(cat[t] for t in flip)))
print("mixed tasks, passes out of", len(omni), ":", sorted(collections.Counter(sum(scored[t].values()) for t in flip).items()))
# LFM3-1.7B and raw runs, same tasks
for k in ("bench70\\raw-lfm3-alone.jsonl", "layers\\step0.jsonl", "layers\\step1.jsonl"):
    f = {r["task_id"] for r in runs[k] if r["passed"] is False}
    print(f"{k}: failed {len(f)}; of the Omni always-fail tasks also failed here: {len(f & set(always_fail))}/{len(always_fail)}")
print("== retry runs")
for k in sorted(runs):
    if "retry" not in k:
        continue
    rows = runs[k]
    act = collections.Counter(r.get("action") for r in rows)
    before = sum(bool(r.get("base_passed")) for r in rows)
    after = sum(bool(r.get("passed")) for r in rows if r.get("passed") is not None)
    ov = [r for r in rows if r.get("overturned")]
    fixed = sum(1 for r in rows if r.get("base_passed") is False and r.get("passed") is True)
    broke = sum(1 for r in rows if r.get("base_passed") is True and r.get("passed") is False)
    lessons = sum(1 for r in rows if r.get("lesson"))
    print(f"  {k:45s} n={len(rows)} base_passed={before} passed={after} fixed={fixed} broke={broke} overturned={len(ov)} lessons={lessons} actions={dict(act)}")
fixed_by = collections.defaultdict(set)
for k in runs:
    if "retry" in k:
        for r in runs[k]:
            if r.get("base_passed") is False and r.get("passed") is True:
                fixed_by[r["task_id"]].add(k.split("\\")[-1])
print("tasks fixed by retry, and in how many retry runs:", {t: len(v) for t, v in fixed_by.items()})
print("of those, in the Omni always-fail set:", sorted(set(fixed_by) & set(always_fail)))
