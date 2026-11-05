"""Precision audit of clover_cycle's Outcome checks on real replies: every finding on the Omni's step-0 and step-7
replies, with the reply text it read, so false alarms can be seen before the checks are allowed to trigger retries."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clover"))
import clover_cycle as C  # noqa: E402

base = Path(__file__).resolve().parents[1] / "verify" / "layers-omni"
tasks = {t["task_id"]: t for t in json.loads((Path(__file__).resolve().parents[1] / "clover" / "tasks70.json").read_text(encoding="utf-8"))}
GSM = "\n Solve step by step and end your response with #### <number>."
n_rep = n_hit = 0
for name in ("step0.jsonl", "step7-defaults.jsonl"):
    for ln in (base / name).read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        t = tasks[r["task_id"]]
        q = t["prompt"] + (GSM if t["benchmark"] == "GSM8K" else "")
        f = C.outcome_findings(q, r["raw_output"] or "")
        n_rep += 1
        if f:
            n_hit += 1
            print(f"== {name} {r['task_id']} passed={r['passed']}")
            for x in f:
                print("   FINDING:", x)
            print("   REPLY:", (r["raw_output"] or "")[:500].replace("\n", " | "))
print(f"\nreplies {n_rep}, with findings {n_hit}")
