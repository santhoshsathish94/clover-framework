"""End-to-end check of the standalone engine on four prompt types; reports outputs, timings and tool calls."""
import json
import sys
import time
from pathlib import Path

base = Path(sys.argv[1])
planner = sys.argv[2] if len(sys.argv) > 2 else str(base / "data" / "lfm3" / "hf")
sys.path.insert(0, str(base / "clover"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import lfm3_agent as A  # noqa: E402

t0 = time.perf_counter()
eng = A.CloverLFM3Engine(base, planner, search_cache=base / "verify" / "smoke-search.jsonl")
info = eng.describe()
print("ENGINE", json.dumps(info), flush=True)
prompts = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Question (astronomy): What is the closest star to Earth after the Sun?\n(A) Sirius\n(B) Proxima Centauri\n(C) Betelgeuse\n(D) Vega\nState the correct option letter (A, B, C, or D) and a brief explanation.",
    "What is the capital of France?",
    "Write the complete Python function `add` to solve the following specification. Include the function signature and wrap your code in a ```python ... ``` block:\n\ndef add(a: int, b: int) -> int:\n    \"\"\"Return the sum of a and b.\n    >>> add(2, 3)\n    5\n    \"\"\"\n",
]
rows = []
for p in prompts:
    t1 = time.perf_counter()
    r = eng.agent_reply(p, cands=2, remember=False)
    rows.append({"prompt": p[:70], "s": round(time.perf_counter() - t1, 1), "plan_ms": r.get("plan_lat_ms"),
                 "render_ms": r.get("k3_moe_plus_ncm_ms"), "latent_steps": r.get("latent_steps"), "exit": r.get("exit_reason"),
                 "tools": [t.get("tool") for t in r.get("tool_calls") or []], "plans": r.get("plans"), "reply": r.get("reply")})
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
out = {"engine": info, "total_s": round(time.perf_counter() - t0, 1), "rows": rows}
(base / "verify" / "smoke-agent.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("SMOKE_AGENT_DONE")
