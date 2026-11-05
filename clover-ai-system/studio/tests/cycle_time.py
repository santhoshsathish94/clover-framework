import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clover"))
import clover_cycle as C  # noqa: E402

base = Path(__file__).resolve().parents[1] / "verify" / "layers-omni"
tasks = {t["task_id"]: t for t in json.loads((Path(__file__).resolve().parents[1] / "clover" / "tasks70.json").read_text(encoding="utf-8"))}
for name in ("step0.jsonl", "step9-cycle-v2.jsonl"):
    rows = [json.loads(l) for l in (base / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    one = [r["wall_s"] for r in rows if C.single_answer(tasks[r["task_id"]]["prompt"])]
    rest = [r["wall_s"] for r in rows if not C.single_answer(tasks[r["task_id"]]["prompt"])]
    print(f"{name}: all {sum(r['wall_s'] for r in rows) / 60:.1f} min, median {st.median([r['wall_s'] for r in rows]):.1f} s; "
          f"one-value questions ({len(one)}) median {st.median(one):.1f} s; others ({len(rest)}) median {st.median(rest):.1f} s")
