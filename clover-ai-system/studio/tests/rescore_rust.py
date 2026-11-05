"""On k3: score the Rust tasks of a laptop result file with the frozen scorer (rustc is not on the laptop).
Usage: rescore_rust.py <results.jsonl> <clover dir with clover_scorer.py + tasks70.json>; writes <results>.rust.jsonl"""
import json
import sys
from pathlib import Path

res_path, clover = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(clover))
import clover_scorer as S  # noqa: E402

tasks = {t["task_id"]: t for t in json.loads((clover / "tasks70.json").read_text(encoding="utf-8"))}
out = []
for ln in res_path.read_text(encoding="utf-8").splitlines():
    if not ln.strip():
        continue
    r = json.loads(ln)
    if r.get("benchmark") == "Rust_Generative_Compilation" and r.get("passed") is None:
        ok, sc, det = S.score_single_task(tasks[r["task_id"]], r.get("raw_output") or "")
        r.update({"passed": bool(ok), "score": float(sc), "score_detail": str(det)[:200], "rust_scored_on": "k3"})
        print(r["task_id"], ok, str(det)[:100])
    out.append(r)
dst = res_path.with_suffix(".rust.jsonl")
dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
print("RUST_RESCORED", dst)
