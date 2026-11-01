"""Calls and total time per layer function in a trace (turn threads and background threads separately).
Usage: python counts.py <trace.jsonl>"""
import json
import sys
from collections import defaultdict

calls, ms = defaultdict(int), defaultdict(float)
for line in open(sys.argv[1], encoding="utf-8"):
    e = json.loads(line)
    where = "turn" if e["thread"].startswith(("Thread-", "MainThread")) else e["thread"]
    key = (e["layer"], e["fn"].split(".")[-1], where)
    if e["phase"] == "call":
        calls[key] += 1
    elif e["phase"] in ("return", "stream_end", "error") and e.get("dur_ms") is not None:
        ms[key] += e["dur_ms"]
for key in sorted(calls, key=lambda k: (k[0], -ms[k])):
    print("%-14s %-36s %-16s %4d calls %10.1f ms" % (key[0], key[1], key[2], calls[key], ms[key]))
