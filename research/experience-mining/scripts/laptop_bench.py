"""Read-only, laptop: pass/fail of each benchmark task across the recorded runs in verify/ (task ids, categories,
booleans and retry actions only; no outputs). Usage: python <this> <CloverAI dir>"""
import collections
import json
import sys
from pathlib import Path

V = Path(sys.argv[1]) / "verify"
runs = {}
for p in sorted(V.rglob("*.jsonl")):
    if "recall-memory" in str(p):
        continue
    rows = []
    for line in open(p, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and "task_id" in r:
            rows.append({k: r.get(k) for k in ("task_id", "category", "passed", "base_passed", "action", "lesson", "calls")}
                        | {"overturned": r.get("overturned") if isinstance(r.get("overturned"), dict) else None,
                           "tools_used": sorted({c.get("name") or c.get("tool") for c in (r.get("tool_calls") or []) if isinstance(c, dict)} - {None}),
                           "rounds": r.get("rounds"), "finish": r.get("finish")})
    if rows:
        runs[str(p.relative_to(V))] = rows
print(json.dumps({"runs": {k: len(v) for k, v in runs.items()}}))
for k, v in runs.items():
    print(json.dumps({"run": k, "rows": v}, default=str))
