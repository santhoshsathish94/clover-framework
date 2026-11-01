"""C1 calls per thread in a framework trace, and the note writer's calls relative to turns.
Usage: python c1_threads.py <framework.jsonl>"""
import json
import sys
from collections import Counter

calls, by_thread, writer = Counter(), Counter(), []
for line in open(sys.argv[1], encoding="utf-8"):
    e = json.loads(line)
    if e.get("layer") == "C1" and e.get("phase") == "call":
        by_thread[e.get("thread")] += 1
        if "memory-index" in str(e.get("thread")):
            writer.append((round(e["t_ms"] / 1000, 1), e["fn"].split(".")[-1], e.get("turn")))
    if e.get("layer") == "turn" and e.get("phase") in ("call", "return"):
        calls[(e["fn"].split(".")[-1], e["phase"])] += 1
print("C1 calls by thread:", dict(by_thread))
print("turn calls/returns:", dict(calls))
print("note-writer C1 calls (t s, fn, turn):", writer[:40], "..." if len(writer) > 40 else "")
