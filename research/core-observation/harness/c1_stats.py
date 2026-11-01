"""Per-request C1 statistics of the laptop stack (prefill, decode speed, tokens) from its trace.
Usage: python c1_stats.py <laptop.jsonl>"""
import json
import sys

rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    e = json.loads(line)
    if e["fn"].endswith("._chat") and e["phase"] == "return":
        s = (e.get("result") or {}).get("stats") or {}
        rows.append((e["turn"], s.get("prompt_tokens"), s.get("prefill_s"), s.get("new_tokens"), s.get("decode_tok_s"),
                     e.get("dur_ms")))
for r in rows:
    print("turn %-3s prompt %-5s prefill %-5s s  new %-4s  %s tok/s  total %s ms" % r)
pre = [r[2] for r in rows if r[2] is not None]
spd = [r[4] for r in rows if r[4]]
print("prefill s: min %.2f max %.2f | decode tok/s: min %.1f max %.1f | requests %d" % (min(pre), max(pre), min(spd), max(spd), len(rows)))
