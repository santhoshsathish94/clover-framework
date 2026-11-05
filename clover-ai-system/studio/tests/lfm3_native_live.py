"""Live checks of LFM3 alone with native tools (Layer 1 + tools) on the laptop. Writes one JSON line per check.
Usage: python tests/lfm3_native_live.py <out.jsonl>"""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import lfm3_native as N  # noqa: E402
import permissions as P  # noqa: E402

BASE = Path.home() / "CloverAI"
OUT = sys.argv[1] if len(sys.argv) > 1 else str(BASE / "verify" / "lfm3_native_live.jsonl")


def log(rec):
    line = json.dumps(rec, ensure_ascii=False, default=str)
    print(line[:3000], flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


perms = P.PermissionStore(Path(tempfile.mkdtemp()) / "perms.json")
perms.update({"code": "allow", "files": "allow", "shell": "allow", "internet": "deny", "edit_files": "deny"})
t0 = time.perf_counter()
eng = N.LFM3Native(BASE, perms)
log({"check": "load", "s": round(time.perf_counter() - t0, 1), "describe": eng.describe()})

# 1. chat template round-trip of a tool-use turn (what the next prompt looks like)
msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "sum?"},
        {"role": "assistant", "content": "\n\n", "reasoning_content": "\nI will use Python.\n",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run_python", "arguments": {"code": "print(1+1)"}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "2"}]
txt = eng._render(msgs, N.STUDIO_TOOLS, True)
tail = txt[txt.index("<|im_start|>user"):]
log({"check": "template tool turn", "ok": "<think>\nI will use Python.\n</think>" in tail and "<function=run_python>" in tail
     and "<tool_response>\n2\n</tool_response>" in tail and tail.endswith("<|im_start|>assistant\n<think>"), "tail": tail})

cases = [
    ("no tools, thinking", "What is 17 * 23? Reply with just the number.", {"tools": False}),
    ("python tool", "Use Python to compute the sum of the squares of the integers from 1 to 100.", {}),
    ("time tool", "What time is it on this laptop right now?", {}),
    ("internet denied", "Search the web for the capital of Australia.", {}),
    ("files tool", f"List the files in the folder {BASE / 'clover' / 'web'}.", {}),
]
for name, q, kw in cases:
    events = []
    t1 = time.perf_counter()
    res = eng.run_turn([], q, sampling="greedy", max_new=1024, on_event=lambda e: events.append(e["type"]), **kw)
    log({"check": name, "s": round(time.perf_counter() - t1, 1), "reply": res["reply"][:800],
         "tools": [(t["tool"], t["ran"], t["why"], str(t["result"])[:200]) for t in res["tool_calls"]],
         "generation": res["generation"], "events": {k: events.count(k) for k in set(events)}, "reasoning": res["reasoning"][:400]})
log({"check": "done", "cache": eng.cache.stats})
