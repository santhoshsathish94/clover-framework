"""Smoke test of the ladder harness's recall path (no model): the index over verify/recall-memory and the blocks it
would give the first tasks. Usage: python tests/recall_bench_smoke.py"""
import json
import sys
import time
from pathlib import Path

base = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base / "clover"))
import bench_layers  # noqa: E402
import lfm3_layers  # noqa: E402

t0 = time.perf_counter()
repo, idx = bench_layers.recall_memory(base, base / "verify" / "recall-memory")
print(f"index built in {time.perf_counter() - t0:.1f} s", flush=True)
kit = lfm3_layers.LayerKit(base)
kit.records, kit.recall = repo, idx
tasks = json.loads((base / "clover" / "tasks70.json").read_text(encoding="utf-8"))
for t in (tasks[0], tasks[10], tasks[30]):
    block, info = kit.recall_context(t["prompt"])
    print(t["task_id"], "refs", len((info or {}).get("refs") or []), "ms", (info or {}).get("ms"),
          "|", (block or "(none)")[:170].replace("\n", " | "), flush=True)
