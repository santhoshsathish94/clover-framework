import json
import sys
from pathlib import Path

base = Path.home() / "CloverAI" / "verify"
lfm = {json.loads(l)["task_id"]: json.loads(l) for l in (base / "layers" / "step1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
lfm0 = {json.loads(l)["task_id"]: json.loads(l) for l in (base / "layers" / "step0.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
for l in (base / sys.argv[1]).read_text(encoding="utf-8").splitlines():
    r = json.loads(l)
    t = r["task_id"]
    print("=" * 100)
    print(t, "omni passed:", r["passed"], "| lfm3 step0:", lfm0.get(t, {}).get("passed"), "step1:", lfm.get(t, {}).get("passed"),
          "tools:", [c[0] for c in lfm.get(t, {}).get("tools") or []])
    for c in (r.get("tool_calls") or [])[:6]:
        print("  CALL", c["tool"], json.dumps(c["arguments"])[:120], "ran:", c["ran"], "->", (c.get("result") or "")[:160].replace("\n", " "))
    print("  REPLY:", (r["raw_output"] or "")[:400].replace("\n", " "))
