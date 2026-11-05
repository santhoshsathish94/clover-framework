"""On k3: build the 70-task matrix with (a) k3's own checkout and (b) the laptop's verbatim extract, check they are
identical, save tasks70.json for the laptop, and re-score the recorded K3 run with the frozen scorer."""
import hashlib
import json
import os
import sys
from pathlib import Path

LAPTOP_CLOVER = Path(sys.argv[1])
CHECKOUT_TOOLS = Path("/opt/clover-k3/clover-framework/clover-ai-system/src/tools/hybrid-ncm-k3")
os.environ.setdefault("CLOVER_NCM_DATA_DIR", "/mnt/nvme1/clover-k3/datasets/hybrid-ncm-k3/data")


def digest(tasks):
    return hashlib.sha256(json.dumps(tasks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


sys.path.insert(0, str(LAPTOP_CLOVER))
import clover_tasks70 as T  # noqa: E402
import clover_scorer as S  # noqa: E402

ours = T.build_shared_70_task_matrix()

sys.path.insert(0, str(CHECKOUT_TOOLS))
import clover_paths  # noqa: E402
from clover_hyphen_runtime import load_hyphenated_module  # noqa: E402
B = load_hyphenated_module(CHECKOUT_TOOLS / "run-standardized-70-task-benchmark.py", module_alias="bench70_k3")
theirs = B.build_shared_70_task_matrix()

report = {"n_ours": len(ours), "n_checkout": len(theirs), "sha_ours": digest(ours), "sha_checkout": digest(theirs),
          "identical": digest(ours) == digest(theirs), "ncm_data_dir_checkout": clover_paths.find_ncm_eval_data_dir()}
(LAPTOP_CLOVER / "tasks70.json").write_text(json.dumps(ours, indent=1, ensure_ascii=False), encoding="utf-8")

recs = [json.loads(ln) for ln in open("/tmp/k3_baseline_70_results.jsonl", encoding="utf-8") if ln.strip()]
by_id = {t["task_id"]: t for t in ours}
mismatch = []
for r in recs:
    ok, sc, det = S.score_single_task(by_id[r["task_id"]], r["raw_output"])
    if bool(ok) != bool(r["passed"]) or abs(float(sc) - float(r["score"])) > 1e-9:
        mismatch.append({"task_id": r["task_id"], "recorded": [r["passed"], r["score"]], "now": [ok, sc, str(det)[:80]]})
report.update({"k3_recorded_passed": sum(r["passed"] for r in recs), "rescored_mismatches": mismatch,
               "ids_in_same_order": [r["task_id"] for r in recs] == [t["task_id"] for t in ours]})
print(json.dumps(report, indent=2))
print("TASKS70_DONE")
