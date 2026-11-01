"""Engine calls as time spans per thread, and spans on different threads that overlap.
Usage: python overlap.py <trace.jsonl> <layer-prefix> [function-suffix-to-skip ...]"""
import json
import sys

path, layer, skip = sys.argv[1], sys.argv[2], tuple(sys.argv[3:])
open_calls, spans = {}, []
for line in open(path, encoding="utf-8"):
    e = json.loads(line)
    if not e["layer"].startswith(layer) or (skip and e["fn"].endswith(skip)):
        continue
    key = (e["fn"], e["thread"])
    if e["phase"] == "call":
        open_calls.setdefault(key, []).append(e)
    elif e["phase"] in ("return", "error", "stream_end") and open_calls.get(key):
        c = open_calls[key].pop(0)
        spans.append((c["t_ms"] / 1000, e["t_ms"] / 1000, c["turn"], e["thread"], e["fn"].split(".")[-1]))
spans.sort()
for s in spans:
    print("%8.1f-%8.1f s  turn %-3s %-18s %s" % s)
overlaps = [(a, b) for i, a in enumerate(spans) for b in spans[i + 1:] if b[0] < a[1] and a[3] != b[3]]
print("overlapping spans on different threads:", len(overlaps))
for a, b in overlaps:
    print("  %s [%s] %.1f-%.1f (turn %s)  <->  %s [%s] %.1f-%.1f (turn %s)"
          % (a[4], a[3], a[0], a[1], a[2], b[4], b[3], b[0], b[1], b[2]))
