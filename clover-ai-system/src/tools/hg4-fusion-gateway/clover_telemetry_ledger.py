"""Comprehensive Clover Runtime & Benchmark Telemetry Ledger (`clover_telemetry_ledger.py`).

Captures structured metrics across every completion, MCP tool call, and multi-turn `Clover4PartAgent.solve`
run so fine-tuning and architecture decisions are 100% data-driven:
  1. Used vs. Unused Inventory:
       - `models_used` vs `models_unused` (calls, tokens, decode `tok/s`, active vs idle instances)
       - `tools_used` vs `tools_unused` across the entire `mcp.json` catalog (calls, ok/fail counts, latency)
       - `instances_summary`: total replica instances registered, instances hit, instances idle (`0` calls)
  2. Token Decode Speed (`decode_tok_per_s`):
       - Per-call, per-turn, and per-model `mean`, `p50`, `p95`, and `max` decode `tok/s`
  3. Benchmark & Solver Cycle Telemetry:
       - `problems_attempted`, `problems_passed`, `cycles_histogram` (`1` cycle vs `2` vs `3+` cycles),
         `mean_cycles_per_problem`, `repaired_after_cycle1_count`, `unsolved_after_max_cycles_count`
       - Actionable `fine_tune_signals` highlighting exact models/tools/failure modes needing fine-tuning.
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
import threading
import time
from pathlib import Path
from typing import Any


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round((len(sorted_vals) - 1) * q))))
    return sorted_vals[idx]


class CloverMetricsLedger:
    """Thread-safe in-memory + SQLite telemetry ledger for Clover AI System & Benchmarks."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        env_p = os.environ.get("CLOVER_METRICS_DB", "").strip()
        if db_path is not None:
            self.db_path: Path | None = Path(db_path)
        elif env_p:
            self.db_path = Path(env_p)
        else:
            ctx_dir = os.environ.get("CLOVER_CONTEXT_DIR", "").strip()
            self.db_path = (Path(ctx_dir) / "clover_metrics.sqlite3") if ctx_dir else None

        self._lock = threading.Lock()
        self._reset_memory_unlocked()
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _reset_memory_unlocked(self) -> None:
        self.started_at = time.time()
        self.model_stats: dict[str, dict[str, Any]] = {}
        self.tool_stats: dict[str, dict[str, Any]] = {}
        self.instance_hits: dict[str, int] = {}
        self.solve_records: list[dict[str, Any]] = []

    def _connect(self) -> sqlite3.Connection:
        assert self.db_path is not None
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS completion_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    benchmark_tag TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL,
                    instance_id TEXT NOT NULL DEFAULT 'local',
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    wall_ms REAL NOT NULL,
                    decode_tok_per_s REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    tool_name TEXT NOT NULL,
                    ok INTEGER NOT NULL,
                    wall_ms REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS solve_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    benchmark_tag TEXT NOT NULL DEFAULT '',
                    task_id TEXT NOT NULL,
                    requested_model TEXT NOT NULL,
                    models_used_json TEXT NOT NULL,
                    tools_used_json TEXT NOT NULL,
                    instances_available INTEGER NOT NULL,
                    turns_used INTEGER NOT NULL,
                    cycles_tried INTEGER NOT NULL,
                    passed INTEGER,
                    solved_on_cycle INTEGER NOT NULL DEFAULT 0,
                    discernment_state TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    wall_ms REAL NOT NULL,
                    mean_decode_tok_per_s REAL NOT NULL,
                    failure_category TEXT NOT NULL DEFAULT ''
                );
                """
            )

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._reset_memory_unlocked()
            if self.db_path is not None and self.db_path.is_file():
                with self._connect() as conn:
                    conn.execute("DELETE FROM completion_events;")
                    conn.execute("DELETE FROM tool_events;")
                    conn.execute("DELETE FROM solve_events;")
        return {"status": "reset", "ts": time.time()}

    def record_completion(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        wall_ms: float,
        instance_id: str = "local",
        benchmark_tag: str = "",
    ) -> float:
        wall_s = max(wall_ms / 1000.0, 1e-6)
        tok_s = round(float(completion_tokens) / wall_s, 2) if completion_tokens > 0 else 0.0
        now = time.time()
        with self._lock:
            st = self.model_stats.setdefault(
                model,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "wall_ms_total": 0.0,
                    "tok_s_samples": [],
                    "instances_used": set(),
                },
            )
            st["calls"] += 1
            st["prompt_tokens"] += int(prompt_tokens)
            st["completion_tokens"] += int(completion_tokens)
            st["wall_ms_total"] += float(wall_ms)
            if tok_s > 0:
                if len(st["tok_s_samples"]) < 4096:
                    st["tok_s_samples"].append(tok_s)
            st["instances_used"].add(instance_id)
            self.instance_hits[instance_id] = self.instance_hits.get(instance_id, 0) + 1

        if self.db_path is not None:
            try:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT INTO completion_events (ts, benchmark_tag, model, instance_id, prompt_tokens, completion_tokens, wall_ms, decode_tok_per_s) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (now, benchmark_tag, model, instance_id, int(prompt_tokens), int(completion_tokens), float(wall_ms), tok_s),
                    )
            except sqlite3.Error:
                pass
        return tok_s

    def record_tool(self, tool_name: str, ok: bool, wall_ms: float) -> None:
        now = time.time()
        with self._lock:
            ts = self.tool_stats.setdefault(
                tool_name,
                {"calls": 0, "ok_calls": 0, "failed_calls": 0, "wall_ms_total": 0.0},
            )
            ts["calls"] += 1
            if ok:
                ts["ok_calls"] += 1
            else:
                ts["failed_calls"] += 1
            ts["wall_ms_total"] += float(wall_ms)

        if self.db_path is not None:
            try:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT INTO tool_events (ts, tool_name, ok, wall_ms) VALUES (?, ?, ?, ?)",
                        (now, tool_name, 1 if ok else 0, float(wall_ms)),
                    )
            except sqlite3.Error:
                pass

    def record_solve(self, entry: dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            if len(self.solve_records) >= 10000:
                self.solve_records.pop(0)
            self.solve_records.append(dict(entry))

        if self.db_path is not None:
            try:
                passed_v = entry.get("passed")
                passed_i = None if passed_v is None else (1 if passed_v else 0)
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO solve_events (
                            ts, benchmark_tag, task_id, requested_model, models_used_json, tools_used_json,
                            instances_available, turns_used, cycles_tried, passed, solved_on_cycle,
                            discernment_state, prompt_tokens, completion_tokens, wall_ms,
                            mean_decode_tok_per_s, failure_category
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            now,
                            str(entry.get("benchmark_tag") or ""),
                            str(entry.get("task_id") or ""),
                            str(entry.get("requested_model") or ""),
                            json.dumps(entry.get("models_used") or []),
                            json.dumps(entry.get("tools_used") or []),
                            int(entry.get("instances_available") or 1),
                            int(entry.get("turns_used") or 1),
                            int(entry.get("cycles_tried") or 1),
                            passed_i,
                            int(entry.get("solved_on_cycle") or 0),
                            str(entry.get("discernment_state") or "PROCEED"),
                            int(entry.get("prompt_tokens") or 0),
                            int(entry.get("completion_tokens") or 0),
                            float(entry.get("wall_ms") or 0.0),
                            float(entry.get("mean_decode_tok_per_s") or 0.0),
                            str(entry.get("failure_category") or ""),
                        ),
                    )
            except sqlite3.Error:
                pass

    def snapshot(
        self,
        *,
        registered_models: list[str] | None = None,
        registered_tools: list[str] | None = None,
        registered_instances: dict[str, list[str]] | None = None,
        benchmark_tag: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            reg_m = sorted(set(registered_models or []) | set(self.model_stats.keys()))
            reg_t = sorted(set(registered_tools or []) | set(self.tool_stats.keys()))
            reg_inst = dict(registered_instances or {})

            models_used_out: dict[str, Any] = {}
            models_unused_out: list[str] = []
            for m in reg_m:
                st = self.model_stats.get(m)
                avail_urls = reg_inst.get(m, [])
                if not st or st["calls"] == 0:
                    models_unused_out.append(m)
                    continue
                s_vals = sorted(st["tok_s_samples"])
                models_used_out[m] = {
                    "calls": st["calls"],
                    "prompt_tokens": st["prompt_tokens"],
                    "completion_tokens": st["completion_tokens"],
                    "mean_wall_ms": round(st["wall_ms_total"] / max(1, st["calls"]), 2),
                    "decode_tok_per_s": {
                        "mean": round(statistics.fmean(s_vals), 2) if s_vals else 0.0,
                        "p50": round(_pct(s_vals, 0.50), 2),
                        "p95": round(_pct(s_vals, 0.95), 2),
                        "max": round(max(s_vals), 2) if s_vals else 0.0,
                    },
                    "instances_registered": max(1, len(avail_urls)),
                    "instances_hit_count": len(st["instances_used"]),
                    "instances_hit": sorted(st["instances_used"]),
                }

            tools_used_out: dict[str, Any] = {}
            tools_unused_out: list[str] = []
            for t in reg_t:
                ts = self.tool_stats.get(t)
                if not ts or ts["calls"] == 0:
                    tools_unused_out.append(t)
                    continue
                tools_used_out[t] = {
                    "calls": ts["calls"],
                    "ok_calls": ts["ok_calls"],
                    "failed_calls": ts["failed_calls"],
                    "error_rate_pct": round(100.0 * ts["failed_calls"] / max(1, ts["calls"]), 2),
                    "mean_wall_ms": round(ts["wall_ms_total"] / max(1, ts["calls"]), 2),
                }

            all_reg_inst_ids: set[str] = set()
            for m, urls in reg_inst.items():
                for u in urls:
                    all_reg_inst_ids.add(f"{m}@{u}")
            used_inst_ids = set(self.instance_hits.keys())
            unused_inst_ids = sorted(all_reg_inst_ids - used_inst_ids)

            solves = [
                r for r in self.solve_records
                if not benchmark_tag or r.get("benchmark_tag") == benchmark_tag
            ]
            cycles_hist: dict[str, int] = {}
            passed_cnt = 0
            verified_cnt = 0
            repaired_cnt = 0
            unsolved_cnt = 0
            fail_cats: dict[str, int] = {}
            for s in solves:
                c_str = str(s.get("cycles_tried") or 1)
                cycles_hist[c_str] = cycles_hist.get(c_str, 0) + 1
                p = s.get("passed")
                if p is not None:
                    verified_cnt += 1
                    if p:
                        passed_cnt += 1
                        if int(s.get("cycles_tried") or 1) > 1:
                            repaired_cnt += 1
                    else:
                        unsolved_cnt += 1
                        fc = str(s.get("failure_category") or "verification_failed")
                        fail_cats[fc] = fail_cats.get(fc, 0) + 1

            fine_tune_signals: list[str] = []
            if solves:
                multi_cyc = sum(1 for s in solves if int(s.get("cycles_tried") or 1) > 1)
                if multi_cyc > 0:
                    fine_tune_signals.append(
                        f"{multi_cyc}/{len(solves)} problems required >1 cycle ({repaired_cnt} self-repaired, {unsolved_cnt} remained unsolved) — prime multi-turn repair traces for SFT/GRPO."
                    )
            for tname, tstat in tools_used_out.items():
                if tstat["failed_calls"] > 0:
                    fine_tune_signals.append(
                        f"Tool `{tname}` had {tstat['failed_calls']}/{tstat['calls']} failed invocations ({tstat['error_rate_pct']}%) — inspect argument formatting."
                    )
            for fcat, fcnt in sorted(fail_cats.items(), key=lambda kv: -kv[1]):
                fine_tune_signals.append(f"Failure mode `{fcat}` occurred on {fcnt} problem(s).")

            return {
                "uptime_s": round(time.time() - self.started_at, 2),
                "benchmark_filter": benchmark_tag or "all",
                "used_vs_unused": {
                    "models_used": models_used_out,
                    "models_unused": models_unused_out,
                    "tools_used": tools_used_out,
                    "tools_unused": tools_unused_out,
                    "instances_total_registered": max(len(all_reg_inst_ids), len(used_inst_ids)),
                    "instances_used_count": len(used_inst_ids),
                    "instances_unused": unused_inst_ids,
                    "instance_hit_counts": dict(self.instance_hits),
                },
                "solver_cycles": {
                    "problems_run": len(solves),
                    "verified_problems": verified_cnt,
                    "verified_passed": passed_cnt,
                    "pass_rate_pct": round(100.0 * passed_cnt / max(1, verified_cnt), 2) if verified_cnt else None,
                    "solved_after_retry_count": repaired_cnt,
                    "unsolved_count": unsolved_cnt,
                    "mean_cycles_per_problem": round(
                        statistics.fmean([float(s.get("cycles_tried") or 1) for s in solves]), 3
                    ) if solves else 0.0,
                    "mean_turns_per_problem": round(
                        statistics.fmean([float(s.get("turns_used") or 1) for s in solves]), 3
                    ) if solves else 0.0,
                    "cycles_histogram": dict(sorted(cycles_hist.items(), key=lambda kv: int(kv[0]))),
                    "failure_categories": fail_cats,
                },
                "fine_tune_signals": fine_tune_signals,
            }
