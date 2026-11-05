"""Compact view of cross_runs.jsonl: one block per message, one line per run. Usage: python show_cross_runs.py"""
import collections
import json
import sys

rows = [json.loads(l) for l in open("cross_runs.jsonl", encoding="utf-8")]
by = collections.defaultdict(list)
order = []
for r in rows:
    if r["prompt"] not in by:
        order.append(r["prompt"])
    by[r["prompt"]].append(r)
for p in order:
    print("==", p[:100])
    for r in by[p]:
        fb = [o.get("feedback") for o in r["outcome_records"] if o.get("feedback")]
        res = [o.get("result") for o in r["outcome_records"] if o.get("result")]
        tools = [o.get("tools") for o in r["outcome_records"] if o.get("tools")]
        print(f"   {r['run']:20s} ran={str(r['ran']):18s} label={str(r['label'])[:34]:34s} aligned={r['aligned']} "
              f"cycle={r['cycle'] or ''} result={res} feedback={fb} mem={r['memory_used']} recall={r['recall']}")
        print(f"   {'':20s} reply: {r['reply'][:150]!r}")
        if r["tool_output"]:
            print(f"   {'':20s} tool output: {r['tool_output'][:120]!r}")
