"""Turns a stack's trace into per-turn layer sequences: what each layer received, returned and how long it took.
Usage: python summarize.py <traces-dir> <stack> [--full]"""
import json
import sys
from collections import defaultdict
from pathlib import Path

d, stack = Path(sys.argv[1]), sys.argv[2]
full = "--full" in sys.argv
events = [json.loads(line) for line in open(d / f"{stack}.jsonl", encoding="utf-8")]
turns = [json.loads(line) for line in open(d / f"{stack}-turns.jsonl", encoding="utf-8")]
by_turn = defaultdict(list)
for e in events:
    by_turn[e["turn"]].append(e)


def short(x, n=160):
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "..."


def text_of(v):
    if isinstance(v, dict) and "head" in v:
        return v["head"] + " ... " + v.get("tail", "")
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)


n_turn = 0
for t in sorted(k for k in by_turn if k):
    evs = by_turn[t]
    meta = turns[n_turn] if n_turn < len(turns) else {}
    n_turn += 1
    print(f"\n### {stack} turn {t}: {short(meta.get('prompt'), 90)!s}")
    print(f"- reply ({meta.get('seconds')} s): {short(meta.get('reply'), 260)}")
    calls = [e for e in evs if e["phase"] == "call"]
    rets = {}
    for e in evs:
        if e["phase"] in ("return", "error", "stream_end"):
            rets.setdefault((e["fn"], e["thread"]), []).append(e)
    seen = defaultdict(int)
    print("- sequence (layer: function [thread] ms):")
    for c in calls:
        k = (c["fn"], c["thread"])
        i = seen[k]
        seen[k] += 1
        r = (rets.get(k) or [None] * (i + 1))[i] if i < len(rets.get(k) or []) else None
        ms = r.get("dur_ms") if r else None
        thread = "" if c["thread"].startswith(("Thread-", "MainThread")) else f" [{c['thread']}]"
        line = f"  - {c['layer']}: {c['fn'].split('.')[-1]}{thread} {ms if ms is not None else '?'} ms"
        if full or c["layer"] in ("semantics", "C1", "C2", "C3", "S3", "S3-web", "S1", "S2-recall", "S2-ontology",
                                  "S1-graph", "cycle", "S3-compiler", "S2-grounding"):
            a = c.get("args") or {}
            if c["fn"].endswith("prefill_cached") and isinstance(a.get("prompt_ids"), dict):
                line += f" | in: {a['prompt_ids'].get('n')} ids: {short(text_of(a['prompt_ids'].get('text')), 220)}"
            elif c["fn"].endswith("_chat"):
                msgs = a.get("messages") or []
                line += f" | in: {len(msgs)} messages, tools {a.get('tools')}; last: {short(text_of((msgs[-1] if msgs else {}).get('content')), 200)}"
            elif c["fn"].split(".")[-1] in ("interpret", "route", "align", "social"):
                line += f" | in: {short(a, 140)}"
            else:
                line += f" | in: {short(a, 160)}"
            if r is not None:
                if r["phase"] == "error":
                    line += f" | ERROR {short(r.get('error'), 200)}"
                elif r["phase"] == "stream_end":
                    line += f" | streamed {r.get('items')} chunks, first {r.get('first_item_ms')} ms: {short(text_of(r.get('text')), 200)}"
                else:
                    line += f" | out: {short(r.get('result'), 240)}"
        print(line)
