"""Why the cycle step gained or lost each task: its cycle log (findings, second solution, values, chosen) next to the
shipped-defaults answer for the same task."""
import json
import sys
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "verify" / "layers-omni"
a = {json.loads(l)["task_id"]: json.loads(l) for l in (base / "step7-defaults.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
b = {json.loads(l)["task_id"]: json.loads(l) for l in (base / "step8-cycle.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
retried = sum(1 for r in b.values() for x in r.get("layer_log") or [] if isinstance(x, dict) and x.get("layer") == "cycle" and x.get("retries"))
second = sum(1 for r in b.values() for x in r.get("layer_log") or [] if isinstance(x, dict) and x.get("layer") == "cycle" and x.get("second_solution") is not None)
print(f"tasks with a second solution: {second}; tasks retried: {retried}")
for tid in sys.argv[1:]:
    ra, rb = a[tid], b[tid]
    cyc = next((x for x in rb.get("layer_log") or [] if isinstance(x, dict) and x.get("layer") == "cycle"), {})
    print("=" * 100)
    print(f"{tid}: defaults {ra['passed']} ({ra['score_detail']}) -> cycle {rb['passed']} ({rb['score_detail']})")
    print("  cycle:", json.dumps({k: cyc.get(k) for k in ("retries", "first_findings", "second_solution", "values",
                                                           "chosen", "confirmed", "final_findings")})[:900])
    print("  DEFAULTS:", (ra["raw_output"] or "")[-260:].replace("\n", " | "))
    print("  CYCLE   :", (rb["raw_output"] or "")[-260:].replace("\n", " | "))
