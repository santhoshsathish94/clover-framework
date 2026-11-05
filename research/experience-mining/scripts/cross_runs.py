"""k3, read-only: the same core-observation messages across five independent framework runs (synthetic drive). For every
run and turn: what semantics understood and routed, what ran, the reply, the label and alignment notes, and the
outcome record Clover itself wrote (result, tools, inferred feedback) in that run's memory repository.
Usage: python3 cross_runs.py > cross_runs.jsonl"""
import json
from pathlib import Path

O = Path("/opt/clover-k3/core-observe")
RUNS = [("1-first-observation", O / "traces/framework-turns.jsonl", O / "archive/before-a/fw-base/clover-memory-repo"),
        ("2-after-a-pass1", O / "traces/after-a/framework-turns.jsonl", O / "archive/after-a-pass1/fw-base/clover-memory-repo"),
        ("3-after-a-final", O / "traces/after-a-final/framework-turns.jsonl", O / "fw-base/clover-memory-repo"),
        ("4-csi-off", O / "traces/csi-off/framework-turns.jsonl", O / "fw-base-csi-off/clover-memory-repo"),
        ("5-csi-on", O / "traces/csi-on/framework-turns.jsonl", O / "fw-base-csi-on/clover-memory-repo")]
for run, turns, repo in RUNS:
    outcomes = {}
    for p in repo.glob("outcomes/*/*.jsonl"):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            outcomes.setdefault((r["session"], r["n"]), []).append(r)
    seen = {}
    for line in open(turns, encoding="utf-8"):
        r = json.loads(line)
        sid = r.get("session")
        seen[sid] = seen.get(sid, 0) + 1
        n = seen[sid]
        ev = r.get("events") or []
        done = next((e for e in ev if e.get("type") == "done"), {})
        lt = done.get("layer_trace") or {}
        sem = lt.get("semantics_frame") or {}
        l2 = lt.get("layer2_hybrid_agent") or {}
        l4 = lt.get("layer4_memory_graph") or {}
        cyc = done.get("clover_cycle_info") or lt.get("clover_retry_cycle") or {}
        out = outcomes.get((sid, n), [])
        print(json.dumps({
            "run": run, "session": sid, "n": n, "prompt": r.get("prompt"), "seconds": r.get("seconds"),
            "error": r.get("driver_error"), "understood": sem.get("understood"), "act": sem.get("act"),
            "route": sem.get("route"), "declined": sem.get("declined"),
            "routed": next((e.get("offered") for e in ev if e.get("type") == "tools"), None),
            "ran": l2.get("tool_executed"), "tool_output": str(l2.get("python_output") or l2.get("observation_preview") or "")[:200],
            "memory_used": [e.get("id") for e in l4.get("entries") or []],
            "recall": (l4.get("s2_turn_recall") or {}).get("refs"),
            "cycle": {k: cyc.get(k) for k in ("action", "final_answer", "solutions", "overturned") if k in cyc},
            "label": sem.get("label"), "aligned": sem.get("aligned"), "reply": (r.get("reply") or "")[:400],
            "outcome_records": [{k: v for k, v in o.items() if k not in ("at", "session", "n")} for o in out]},
            ensure_ascii=False))
