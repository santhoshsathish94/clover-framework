"""Framework: per turn, C1's decoded answer text vs C3's aligned text vs the reply sent; and prefill sizes/times.
Usage: python c3_vs_c1.py <framework.jsonl> <framework-turns.jsonl>"""
import json
import sys
from collections import defaultdict

ev = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
turns = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8")]
by = defaultdict(list)
for e in ev:
    if e["thread"].startswith(("Thread-", "MainThread")):
        by[e["turn"]].append(e)


def txt(v):
    if isinstance(v, dict) and "head" in v:
        return v["head"] + "..." + v.get("tail", "")
    return v if isinstance(v, str) else ""


replies = {i + 1: t.get("reply") for i, t in enumerate(turns)}
for t in sorted(k for k in by if k):
    c1 = c3 = None
    pre = []
    for e in by[t]:
        if e["phase"] != "return":
            continue
        fn = e["fn"].split(".")[-1]
        if fn == "decode_from_cache_with_hidden":
            c1 = txt(((e.get("result") or {}).get("ids") or {}).get("text"))
        if fn == "_dec_model_aligned":
            r = e.get("result")
            c3 = txt(r[0]) if isinstance(r, list) and r else None
    for e in by[t]:
        if e["phase"] == "call" and e["fn"].endswith("prefill_cached"):
            pre.append(((e.get("args") or {}).get("prompt_ids") or {}).get("n"))
    for e in by[t]:
        if e["phase"] == "return" and e["fn"].endswith("prefill_cached"):
            pre.append(round(e.get("dur_ms", 0) / 1000, 1))
    reply = replies.get(t) or ""
    c1s = (c1 or "").replace("</answer>", "").replace("<|im_end|>", "").strip()
    print(f"turn {t}: prefill (ids, s) {pre} | C3 == C1: {c3 is not None and c3.strip() == c1s} | "
          f"C3 is a prefix of C1: {bool(c3) and c1s.startswith(c3.strip())} | reply == C1: {reply.strip() == c1s} | "
          f"len C1 {len(c1s)} C3 {len(c3 or '')} reply {len(reply)}")
