import json
import re
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "verify" / "layers-omni"
for ln in (base / "step0.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(ln)
    if r["benchmark"] in ("GSM8K",) or r["task_id"].startswith(("agent_", "mem_qa_4", "mem_qa_5")):
        out = r["raw_output"] or ""
        eq = [m.group(0) for m in re.finditer(r"[^\n=]{0,40}=[^\n=]{0,25}", out)][:8]
        print(f"== {r['task_id']} passed={r['passed']} detail={r['score_detail']}")
        for e in eq:
            print("   ", e.strip())
