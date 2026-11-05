"""How the v2 cycle acted on each task: counts by action, and for the given tasks the evidence behind the change."""
import json
import sys
from collections import Counter
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "verify" / "layers-omni"
a = {json.loads(l)["task_id"]: json.loads(l) for l in (base / "step7-defaults.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
b = {json.loads(l)["task_id"]: json.loads(l) for l in (base / "step9-cycle-v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
acts, same_first = Counter(), 0
for tid, r in b.items():
    cyc = next((x for x in r.get("layer_log") or [] if isinstance(x, dict) and x.get("layer") == "cycle"), {})
    act = "overturned" if cyc.get("overturned") else "retried" if cyc.get("retries") else \
          ("solved again, agreed" if cyc.get("solutions") and cyc.get("confirmed") else
           "solved again, not confirmed" if cyc.get("solutions") else "checks only")
    acts[act] += 1
print("actions:", dict(acts))
for tid in sys.argv[1:]:
    rb, ra = b[tid], a[tid]
    cyc = next((x for x in rb.get("layer_log") or [] if isinstance(x, dict) and x.get("layer") == "cycle"), {})
    print("=" * 90)
    print(f"{tid}: defaults {ra['passed']} ({ra['score_detail']}) -> cycle {rb['passed']} ({rb['score_detail']})")
    print("  cycle:", json.dumps({k: cyc.get(k) for k in ("retries", "first_findings", "solutions", "overturned",
                                                           "corrected", "confirmed", "final_findings")})[:700])
    print("  CYCLE ANSWER END:", (rb["raw_output"] or "")[-200:].replace("\n", " | "))
