import json
import sys
from pathlib import Path

base = Path.home() / "CloverAI" / "verify" / "layers-omni"
step1 = {json.loads(l)["task_id"]: json.loads(l) for l in (base / "step1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
for l in (base / "step4-compiler.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(l)
    if r["task_id"] not in sys.argv[1:]:
        continue
    print("=" * 90)
    print(r["task_id"], "step1 passed:", step1[r["task_id"]]["passed"], "-> compiler step:", r["passed"], r["score_detail"])
    for x in r.get("layer_log") or []:
        print("  LAYER", json.dumps(x, ensure_ascii=False)[:700])
    print("  STEP1 REPLY:", (step1[r["task_id"]]["raw_output"] or "")[:300].replace("\n", " | "))
    print("  NEW REPLY  :", (r["raw_output"] or "")[:300].replace("\n", " | "))
