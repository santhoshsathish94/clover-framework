import json
from pathlib import Path
O = Path("/opt/clover-k3/core-observe")
turns = {}
for line in open(O / "traces/after-a-final/framework.jsonl", encoding="utf-8", errors="replace"):
    try:
        e = json.loads(line)
    except ValueError:
        continue
    turns.setdefault(e.get("turn"), []).append(e)
first = next(e for evs in turns.values() for e in evs if e.get("layer") == "input")
print("input event keys:", sorted(first))
for t, evs in sorted(turns.items(), key=lambda x: (x[0] is None, x[0] or 0)):
    blob = json.dumps([e for e in evs if e.get("layer") in ("input", "semantics")], default=str)
    if "try again" not in blob and "2**64" not in blob:
        continue
    print(f"-- turn {t}")
    for e in evs:
        fn = str(e.get("fn"))
        if e.get("phase") in ("call", "request", "error"):
            print(f"   {e.get('seq'):6} {str(e.get('layer')):12s} {fn[:62]:62s} {e.get('phase')}")
        if e.get("phase") == "return" and any(k in fn for k in ("final_answer", "retry", "route", "_claims", "solve", "interpret", "python")):
            print(f"   {'':6} {'':12s} {fn[:62]:62s} -> {json.dumps(e.get('result'), ensure_ascii=False, default=str)[:200]}")
