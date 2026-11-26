"""Clover AI Studio v6.0 — Simpler 4-Part Steerable Clover System, Server Model Setup & Live Telemetry Ledger.

Shared across Windows Desktop (`CloverAIStudio.exe`), Android Mobile (Chaquopy `clover.zip`),
and Web Studio (`api.cloverframework.com` / `clover-web-studio.service`).

4-Part Architecture:
  1. `AGENTS.md`       — Steerable epistemic instructions (`Context -> Direction -> Execution -> Outcome -> Growth`
                         + `STOP_AND_ASK` human steering + Tier-0 `2B` Tool-Exposure Gate).
  2. `mcp.json`        — Deterministic `0B` Computed Tools (`xcruncher_exact_math`, `forecast_series_0b`,
                         `resolve_defeasible_rules_0b`, `python_ast_verify`) + remote `HG4` specialist mesh.
  3. `<task_id>.md`    — Human-readable task context ledger (`<base>/clover-ai-studio/contexts/<task_id>.md`).
  4. `CloverTodoStore` — SQLite prerequisite-gated todo graph (`todos` + `todo_deps`).

Model & Server Setup Engine:
  - Users can set up 1 model or the complete Clover AI System across their device and/or provided GPU servers.
  - Small models & 0B tools (`0b-computed-pack`, `omni`, `chronos-2`, `vjepa-2.1`) run interactively on any
    phone, laptop, or CPU/GPU machine.
  - Big models (`boltz-2`, `goedel-prover-v2-32b`, `clover-35b-a3b`, `nemotron-3.5-super-vl`, `kimi-k3`) require
    GPUs to run interactively; CPU-only or undersized machines skip them with explicit minimum server specs.
"""
from __future__ import annotations

import ast
import concurrent.futures
import datetime
import hashlib
import ipaddress
import json
import math
import os
import re
import secrets
import shlex
import shutil
import socket
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

if hasattr(sys, "set_int_max_str_digits"):
    try:
        if sys.get_int_max_str_digits() != 0 and sys.get_int_max_str_digits() < 250_000:
            sys.set_int_max_str_digits(250_000)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 0. Canonical Model ID Map (Catalog <-> Gateway <-> Telemetry Ledger)
# ---------------------------------------------------------------------------
# catalog_id -> (gateway_model_id, chat_capable, specialist_mcp_tool)
CANON: dict[str, tuple[str | None, bool, str | None]] = {
    "omni": ("lfm3-omni-2b", True, None),
    "clover-35b-a3b": ("clover-35b-a3b", True, None),
    "nemotron-3.5-super-vl": ("nemotron-3.5-super-vl", True, None),
    "kimi-k3": ("kimi-k3", True, None),
    "goedel-prover-v2-32b": ("goedel-prover-v2-32b", True, "goedel_lean4_prove"),
    "boltz-2": ("boltz-2", False, "boltz2_fold_structure"),
    "chronos-2": ("chronos-2", False, "chronos2_forecast"),
    "vjepa-2.1": ("vjepa-2.1", False, "vjepa2_predict_physics"),
    "0b-computed-pack": (None, False, None),
}

_ALIASES: dict[str, str] = {
    "boltz2": "boltz-2",
    "chronos2": "chronos-2",
    "vjepa2": "vjepa-2.1",
    "lfm3-omni-2b": "omni",
    "lfm3-1.7b": "omni",
    "goedel-prover": "goedel-prover-v2-32b",
    "goedel32b": "goedel-prover-v2-32b",
    "clover35b": "clover-35b-a3b",
    "nemotron": "nemotron-3.5-super-vl",
}


def canon_model_id(mid: str) -> str:
    s = str(mid or "").strip()
    if s.startswith(("v6:", "k3:", "rm:")):
        s = s.split(":", 1)[1]
    s = s.split("@", 1)[0].strip()
    return _ALIASES.get(s, s)


DEFAULT_REGISTERED_MODELS = tuple(k for k in CANON if k != "0b-computed-pack")

DEFAULT_REGISTERED_TOOLS = (
    "calculate",
    "xcruncher_compute",
    "xcruncher_exact_math",
    "forecast_series",
    "forecast_series_0b",
    "resolve_rules",
    "resolve_defeasible_rules_0b",
    "convert_units",
    "search_files",
    "file_info",
    "git_inspect",
    "read_document",
    "analyze_table",
    "chart_table",
    "media_info",
    "extract_frame",
    "python_ast_verify",
)


# ---------------------------------------------------------------------------
# 1. Deterministic 0B Computed Specialists (Offline on Phone, Laptop & Web)
# ---------------------------------------------------------------------------
def forecast_series_0b(values: list[float] | str, horizon: int = 12) -> dict[str, Any]:
    """0B Deterministic Periodogram + Joint Harmonic OLS + Split-Conformal P10/P50/P90 Forecaster.
    Zero neural parameters, zero external dependencies (pure Python stdlib math), <50 ms on CPU."""
    t0 = time.perf_counter()
    if isinstance(values, str):
        raw_vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", values)]
    else:
        raw_vals = [float(x) for x in (values or ())]
    n = len(raw_vals)
    h = max(1, min(int(horizon or 12), 256))
    if n < 4:
        raise ValueError("forecast_series requires at least 4 historical numeric observations")

    t_mean = (n - 1) / 2.0
    y_mean = sum(raw_vals) / n
    s_tt = sum((i - t_mean) ** 2 for i in range(n)) or 1.0
    slope_init = sum((i - t_mean) * (raw_vals[i] - y_mean) for i in range(n)) / s_tt
    intercept_init = y_mean - slope_init * t_mean
    detrended = [raw_vals[i] - (intercept_init + slope_init * i) for i in range(n)]

    max_k = min(n // 2, 64)
    chosen_omegas: list[tuple[float, float]] = []
    if n >= 8 and max_k >= 2:
        scored: list[tuple[float, int]] = []
        for k in range(1, max_k):
            omega = 2.0 * math.pi * k / n
            c_sum = sum(detrended[t] * math.cos(omega * t) for t in range(n)) * (2.0 / n)
            s_sum = sum(detrended[t] * math.sin(omega * t) for t in range(n)) * (2.0 / n)
            scored.append((c_sum * c_sum + s_sum * s_sum, k))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen_k: list[int] = []
        for amp2, k in scored:
            if len(chosen_k) >= 2:
                break
            if any(abs(k - ck) <= 1 for ck in chosen_k):
                continue
            if amp2 > 1e-12:
                chosen_k.append(k)
                chosen_omegas.append((2.0 * math.pi * k / n, round(n / k, 3)))

    def _features(idx: int) -> list[float]:
        row = [1.0, float(idx)]
        for omega, _period in chosen_omegas:
            row.append(math.cos(omega * idx))
            row.append(math.sin(omega * idx))
        return row

    p_dim = 2 + 2 * len(chosen_omegas)
    ata = [[0.0] * (p_dim + 1) for _ in range(p_dim)]
    for t_i in range(n):
        fx = _features(t_i)
        yv = raw_vals[t_i]
        for r_i in range(p_dim):
            fr = fx[r_i]
            for c_i in range(p_dim):
                ata[r_i][c_i] += fr * fx[c_i]
            ata[r_i][p_dim] += fr * yv
    for d_i in range(p_dim):
        ata[d_i][d_i] += 1e-10
    for col in range(p_dim):
        piv = max(range(col, p_dim), key=lambda r_i: abs(ata[r_i][col]))
        ata[col], ata[piv] = ata[piv], ata[col]
        denom = ata[col][col] or 1e-12
        for j in range(col, p_dim + 1):
            ata[col][j] /= denom
        for r_i in range(p_dim):
            if r_i != col and ata[r_i][col] != 0.0:
                factor = ata[r_i][col]
                for j in range(col, p_dim + 1):
                    ata[r_i][j] -= factor * ata[col][j]
    weights = [ata[i][p_dim] for i in range(p_dim)]
    slope = weights[1]

    def _predict_at(idx: int) -> float:
        fx = _features(idx)
        return sum(w * x for w, x in zip(weights, fx))

    cal_start = max(2, int(n * 0.7))
    abs_res = sorted(abs(raw_vals[i] - _predict_at(i)) for i in range(cal_start, n))
    if not abs_res:
        abs_res = sorted(abs(raw_vals[i] - _predict_at(i)) for i in range(n))
    q80_idx = min(len(abs_res) - 1, max(0, math.ceil(0.80 * (len(abs_res) + 1)) - 1))
    q_radius = abs_res[q80_idx] if abs_res else 0.0

    naive_mae = sum(abs(raw_vals[i] - raw_vals[i - 1]) for i in range(1, n)) / max(1, n - 1)
    fit_mae = sum(abs(raw_vals[i] - _predict_at(i)) for i in range(n)) / n
    insample_mase = round(fit_mae / max(naive_mae, 1e-12), 6)

    p50 = [round(_predict_at(n + step), 6) for step in range(h)]
    p10 = [round(v - q_radius, 6) for v in p50]
    p90 = [round(v + q_radius, 6) for v in p50]
    return {
        "engine": "clover-0b-periodogram-conformal",
        "history_len": n,
        "horizon": h,
        "dominant_periods": [p for _w, p in chosen_omegas],
        "trend_slope": round(slope, 6),
        "conformal_q80_radius": round(q_radius, 6),
        "insample_mase": insample_mase,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }


def resolve_defeasible_rules_0b(
    rules: list[dict[str, Any]],
    context_facts: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """0B Deterministic Defeasible Epistemic Rule Resolver."""
    facts = {str(f).strip().lower() for f in (context_facts or ()) if str(f).strip()}
    rejected_ids: list[str] = []
    blocked_unless_ids: list[str] = []
    unmet_condition_ids: list[str] = []
    applicable: list[tuple[tuple[int, float, int], dict[str, Any]]] = []

    for idx, raw in enumerate(rules or ()):
        if not isinstance(raw, dict):
            continue
        rid = str(raw.get("id") or f"rule_{idx + 1}")
        status = str(raw.get("status") or "accepted").strip().lower()
        if status in ("rejected", "revoked", "quarantined", "denied"):
            rejected_ids.append(rid)
            continue

        conds = [str(c).strip().lower() for c in (raw.get("conditions") or raw.get("when") or ()) if str(c).strip()]
        unless = [str(u).strip().lower() for u in (raw.get("unless") or raw.get("exceptions") or ()) if str(u).strip()]

        if any(u in facts for u in unless):
            blocked_unless_ids.append(rid)
            continue
        if any(c not in facts for c in conds):
            unmet_condition_ids.append(rid)
            continue

        specificity = len(conds)
        priority = float(raw.get("priority", 0.0) or 0.0)
        ts_rank = int(raw.get("updated_seq", idx))
        applicable.append(((specificity, priority, ts_rank), dict(raw, id=rid, matched_specificity=specificity)))

    applicable.sort(key=lambda item: item[0], reverse=True)
    winner = applicable[0][1] if applicable else None
    return {
        "engine": "clover-0b-defeasible-resolver",
        "winner_id": winner["id"] if winner else None,
        "winner_action": (winner.get("action") or winner.get("value") or winner.get("conclusion")) if winner else None,
        "winner": winner,
        "applicable_order": [item[1]["id"] for item in applicable],
        "rejected_filtered": rejected_ids,
        "blocked_by_unless": blocked_unless_ids,
        "unmet_conditions": unmet_condition_ids,
    }


def exact_bigint_power(base_str: str, exp_str: str, max_output_digits: int = 200_000) -> dict[str, Any]:
    """Exact binary exponentiation (`A^B` / `A**B`) safe above Python 3.11's 4,300-digit str() limit."""
    t0 = time.perf_counter()
    a = int(str(base_str).replace("_", "").strip())
    b = int(str(exp_str).replace("_", "").strip())
    if b < 0:
        raise ValueError("exact_bigint_power requires a non-negative integer exponent")
    if a != 0 and b > 0:
        est_digits = int(b * math.log10(abs(a))) + 1 if abs(a) > 1 else 1
        if est_digits > max_output_digits:
            raise ValueError(f"result has ~{est_digits} digits (exceeds safe cap {max_output_digits})")
    val = pow(a, b)
    if hasattr(sys, "set_int_max_str_digits"):
        prev = sys.get_int_max_str_digits()
        if prev != 0 and prev < max_output_digits + 64:
            sys.set_int_max_str_digits(max_output_digits + 64)
    s = str(val)
    out: dict[str, Any] = {
        "operation": "pow",
        "base": a,
        "exponent": b,
        "digit_count": len(s),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }
    if len(s) <= 4096:
        out["exact_decimal"] = s
    else:
        out["leading_64"] = s[:64]
        out["trailing_64"] = s[-64:]
    return out


# ---------------------------------------------------------------------------
# 2. Prerequisite-Gated SQLite Todo Store
# ---------------------------------------------------------------------------
class CloverTodoStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS todos (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS todo_deps (
                    todo_id TEXT NOT NULL,
                    depends_on TEXT NOT NULL,
                    PRIMARY KEY (todo_id, depends_on)
                );
                """
            )

    def add_todo(self, task_id: str, todo_id: str, title: str, description: str = "", depends_on: list[str] | None = None) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO todos (id, task_id, title, description, status, updated_at) VALUES (?, ?, ?, ?, 'pending', ?)",
                (todo_id, task_id, title, description, now),
            )
            for dep in (depends_on or ()):
                self._conn.execute("INSERT OR IGNORE INTO todo_deps (todo_id, depends_on) VALUES (?, ?)", (todo_id, dep))

    def set_status(self, todo_id: str, status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE todos SET status = ?, updated_at = ? WHERE id = ?", (status, time.time(), todo_id))

    def ready_todos(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT t.* FROM todos t
                WHERE t.task_id = ? AND t.status = 'pending'
                  AND NOT EXISTS (
                      SELECT 1 FROM todo_deps td
                      JOIN todos dep ON td.depends_on = dep.id
                      WHERE td.todo_id = t.id AND dep.status != 'done'
                  )
                ORDER BY t.rowid ASC
                """,
                (task_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def all_todos(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM todos WHERE task_id = ? ORDER BY rowid ASC", (task_id,)).fetchall()
            return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# 3. Live Telemetry Ledger (`GET /api/clover/metrics`)
# ---------------------------------------------------------------------------
class CloverMetricsLedger:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    task_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL DEFAULT 1,
                    model TEXT NOT NULL,
                    instance TEXT NOT NULL DEFAULT 'local',
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    elapsed_ms REAL NOT NULL DEFAULT 0.0,
                    decode_tok_per_s REAL NOT NULL DEFAULT 0.0,
                    ok INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    task_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL DEFAULT 1,
                    tool_name TEXT NOT NULL,
                    instance TEXT NOT NULL DEFAULT '0b-local',
                    elapsed_ms REAL NOT NULL DEFAULT 0.0,
                    ok INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS problem_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    task_id TEXT NOT NULL,
                    domain TEXT NOT NULL DEFAULT 'chat',
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cycles_used INTEGER NOT NULL DEFAULT 1,
                    tools_used_json TEXT NOT NULL DEFAULT '[]',
                    total_ms REAL NOT NULL DEFAULT 0.0
                );
                """
            )

    def record_model_call(
        self,
        *,
        task_id: str,
        model: str,
        instance: str = "local",
        cycle: int = 1,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        elapsed_ms: float = 0.0,
        decode_tok_per_s: float | None = None,
        ok: bool = True,
    ) -> None:
        tps = (
            float(decode_tok_per_s)
            if decode_tok_per_s is not None
            else (round(completion_tokens / (elapsed_ms / 1000.0), 2) if elapsed_ms > 1.0 and completion_tokens > 0 else 0.0)
        )
        cm = canon_model_id(model)
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO model_calls
                   (ts, task_id, cycle, model, instance, prompt_tokens, completion_tokens, elapsed_ms, decode_tok_per_s, ok)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (time.time(), str(task_id), int(cycle), cm, str(instance), int(prompt_tokens), int(completion_tokens), float(elapsed_ms), tps, 1 if ok else 0),
            )

    def record_tool_call(self, *, task_id: str, tool_name: str, instance: str = "0b-local", cycle: int = 1, elapsed_ms: float = 0.0, ok: bool = True) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO tool_calls (ts, task_id, cycle, tool_name, instance, elapsed_ms, ok)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (time.time(), str(task_id), int(cycle), str(tool_name), str(instance), float(elapsed_ms), 1 if ok else 0),
            )

    def record_problem_run(self, *, task_id: str, model: str, status: str = "SOLVED", domain: str = "chat", cycles_used: int = 1, tools_used: list[str] | None = None, total_ms: float = 0.0) -> None:
        cm = canon_model_id(model)
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO problem_runs (ts, task_id, domain, model, status, cycles_used, tools_used_json, total_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (time.time(), str(task_id), str(domain), cm, str(status), int(cycles_used), json.dumps(list(tools_used or ())), float(total_ms)),
            )

    def reset(self) -> dict[str, Any]:
        with self._lock, self._conn:
            self._conn.executescript("DELETE FROM model_calls; DELETE FROM tool_calls; DELETE FROM problem_runs;")
            self._started_at = time.time()
        return {"ok": True, "reset_at": self._started_at}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            m_rows = self._conn.execute("SELECT * FROM model_calls ORDER BY id DESC LIMIT 2000").fetchall()
            t_rows = self._conn.execute("SELECT * FROM tool_calls ORDER BY id DESC LIMIT 2000").fetchall()
            p_rows = self._conn.execute("SELECT * FROM problem_runs ORDER BY id DESC LIMIT 2000").fetchall()

        per_model: dict[str, list[sqlite3.Row]] = {}
        instances_hit: set[str] = set()
        for r in m_rows:
            mk = canon_model_id(str(r["model"]))
            per_model.setdefault(mk, []).append(r)
            instances_hit.add(str(r["instance"]))

        model_stats: dict[str, Any] = {}
        for mk, rows in per_model.items():
            tps_vals = sorted(float(x["decode_tok_per_s"]) for x in rows if float(x["decode_tok_per_s"]) > 0)
            lat_vals = [float(x["elapsed_ms"]) for x in rows]
            p50 = round(statistics.median(tps_vals), 2) if tps_vals else 0.0
            p95 = round(tps_vals[min(len(tps_vals) - 1, int(0.95 * len(tps_vals)))], 2) if tps_vals else 0.0
            model_stats[mk] = {
                "calls": len(rows),
                "ok_calls": sum(int(x["ok"]) for x in rows),
                "completion_tokens": sum(int(x["completion_tokens"]) for x in rows),
                "mean_elapsed_ms": round(sum(lat_vals) / len(lat_vals), 2) if lat_vals else 0.0,
                "decode_tok_per_s": {
                    "mean": round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else 0.0,
                    "p50": p50,
                    "p95": p95,
                    "max": round(max(tps_vals), 2) if tps_vals else 0.0,
                },
            }

        per_tool: dict[str, list[sqlite3.Row]] = {}
        for r in t_rows:
            tk = str(r["tool_name"])
            per_tool.setdefault(tk, []).append(r)
            instances_hit.add(str(r["instance"]))

        tool_stats = {
            tk: {
                "calls": len(rows),
                "ok_calls": sum(int(x["ok"]) for x in rows),
                "mean_elapsed_ms": round(sum(float(x["elapsed_ms"]) for x in rows) / len(rows), 2),
            }
            for tk, rows in per_tool.items()
        }

        models_used = sorted(per_model.keys())
        tools_used = sorted(per_tool.keys())
        models_unused = sorted(m for m in DEFAULT_REGISTERED_MODELS if m not in per_model)
        tools_unused = sorted(t for t in DEFAULT_REGISTERED_TOOLS if t not in per_tool)

        cycles_hist: dict[str, int] = {}
        solved_retry = 0
        fine_tune_candidates: list[dict[str, Any]] = []
        for r in p_rows:
            c = int(r["cycles_used"])
            cycles_hist[str(c)] = cycles_hist.get(str(c), 0) + 1
            if c >= 2 and str(r["status"]) == "SOLVED":
                solved_retry += 1
                if len(fine_tune_candidates) < 20:
                    fine_tune_candidates.append({
                        "task_id": r["task_id"],
                        "model": canon_model_id(str(r["model"])),
                        "cycles_used": c,
                        "reason": "needed_multi_cycle_repair",
                    })

        return {
            "version": "6.0.0",
            "architecture": "clover-v6-4part",
            "uptime_s": round(time.time() - self._started_at, 1),
            "used_vs_unused": {
                "models_used": models_used,
                "models_unused": models_unused,
                "tools_used": tools_used,
                "tools_unused": tools_unused,
                "instances_hit": sorted(instances_hit),
            },
            "models": model_stats,
            "tools": tool_stats,
            "solver_cycles": {
                "total_problems": len(p_rows),
                "cycles_histogram": cycles_hist,
                "solved_after_retry_count": solved_retry,
                "mean_cycles": round(sum(int(r["cycles_used"]) for r in p_rows) / len(p_rows), 2) if p_rows else 0.0,
            },
            "fine_tune_signals": fine_tune_candidates,
        }


# ---------------------------------------------------------------------------
# 4. Clover v6.0 Model Catalog, Strict Target Parser & Per-GPU Placement
# ---------------------------------------------------------------------------
CLOVER_V6_MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "id": "0b-computed-pack",
        "name": "0B Computed Specialists (x-cruncher + Conformal + Defeasible)",
        "tier": "small",
        "role": "Exact Math, 0B Time-Series & Epistemic Rule Verification",
        "weights_gb": 0.0,
        "resident_gb": 0.0,
        "max_tp": 1,
        "runs_on_any_device": True,
        "needs_gpu_interactive": False,
        "local_runtime": True,
        "min_vram_gb": 0,
        "min_servers": 1,
        "min_server_spec": "Any phone, laptop, or CPU machine (0 GPUs required)",
        "interactive_speed": "< 50 ms on any CPU",
    },
    {
        "id": "omni",
        "name": "LFM3-Omni-2B (HG4-16R)",
        "tier": "small",
        "role": "Everyday Local & Edge Multimodal Assistant",
        "weights_gb": 0.98,
        "resident_gb": 1.6,
        "max_tp": 1,
        "runs_on_any_device": True,
        "needs_gpu_interactive": False,
        "local_runtime": True,
        "min_vram_gb": 0,
        "min_servers": 1,
        "min_server_spec": "Any phone, laptop, or CPU/GPU machine (0 GPUs required)",
        "interactive_speed": "18–65 tok/s CPU/Phone · 163 tok/s GPU",
    },
    {
        "id": "chronos-2",
        "name": "Chronos-Bolt 0.2B (HG4)",
        "tier": "small",
        "role": "Zero-Shot Probabilistic Time-Series Specialist",
        "weights_gb": 0.12,
        "resident_gb": 0.5,
        "max_tp": 1,
        "runs_on_any_device": True,
        "needs_gpu_interactive": False,
        "local_runtime": True,  # seamlessly backed locally by 0B Periodogram+Conformal when no remote Chronos server
        "min_vram_gb": 0,
        "min_servers": 1,
        "min_server_spec": "Any phone, laptop, or CPU/GPU machine (0 GPUs required)",
        "interactive_speed": "< 50 ms 0B Conformal / 0.8 s CPU · 0.09 s GPU",
    },
    {
        "id": "vjepa-2.1",
        "name": "V-JEPA 2.1 (HG4)",
        "tier": "small",
        "role": "Physical Video Dynamics & Epistemic Surprise Specialist",
        "weights_gb": 1.2,
        "resident_gb": 2.0,
        "max_tp": 1,
        "runs_on_any_device": True,
        "needs_gpu_interactive": False,
        "local_runtime": True,  # Studio bundles local JEPA runtime (`jepa_engine.py`)
        "min_vram_gb": 0,
        "min_servers": 1,
        "min_server_spec": "Any phone, laptop, or CPU/GPU machine (0 GPUs required)",
        "interactive_speed": "0.9 s CPU · 0.12 s GPU",
    },
    {
        "id": "boltz-2",
        "name": "Boltz-2 Biomolecular Folding (HG4)",
        "tier": "big_gpu",
        "role": "3D Protein/RNA/Ligand Structure & Affinity Specialist",
        "weights_gb": 3.5,
        "resident_gb": 14.0,  # pairformer 3D diffusion activations
        "max_tp": 1,
        "runs_on_any_device": False,
        "needs_gpu_interactive": True,
        "local_runtime": False,
        "min_vram_gb": 16,
        "min_servers": 1,
        "min_server_spec": "1x GPU server with >= 16 GB VRAM (e.g. RTX 4080 / A10G / T4 / RTX 3090)",
        "interactive_speed": "~12 s on 16–24 GB GPU (>180 s on CPU — not interactive)",
    },
    {
        "id": "goedel-prover-v2-32b",
        "name": "Goedel-Prover-V2-32B (HG4-16R)",
        "tier": "big_gpu",
        "role": "Formal Lean 4 Mathematical Proof Specialist",
        "weights_gb": 16.4,
        "resident_gb": 21.6,  # 16.4 GB weights + 16k KV cache + runtime
        "max_tp": 2,
        "runs_on_any_device": False,
        "needs_gpu_interactive": True,
        "local_runtime": False,
        "min_vram_gb": 24,
        "min_servers": 1,
        "min_server_spec": "1x GPU server with >= 24 GB VRAM (e.g. 1x RTX 3090 / 4090 / A10G / L4)",
        "interactive_speed": "18.4 tok/s on 24 GB GPU (~0.8 tok/s on CPU — not interactive)",
    },
    {
        "id": "clover-35b-a3b",
        "name": "Clover-35B-A3B Coordinator (Qwen3.6-35B-A3B + LoRA, HG4-16R-OP)",
        "tier": "big_gpu",
        "role": "Primary Clover v6.0 Coordinator Brain & Tool Orchestrator",
        "weights_gb": 17.4,
        "resident_gb": 20.2,  # 17.4 GB weights + hybrid GDN/GQA 32k KV + runtime
        "max_tp": 2,
        "runs_on_any_device": False,
        "needs_gpu_interactive": True,
        "local_runtime": False,
        "min_vram_gb": 24,
        "min_servers": 1,
        "min_server_spec": "1x GPU server with >= 24 GB VRAM (e.g. 1x RTX 3090 / 4090 / A10G / L4)",
        "interactive_speed": "35.8 tok/s on 24 GB GPU (needs GPU for full interactive speed)",
    },
    {
        "id": "nemotron-3.5-super-vl",
        "name": "Nemotron 3.5 Super VL (120B-A12B HG4-16R)",
        "tier": "big_gpu",
        "role": "1M Context Mamba-2 + GQA & C-RADIO Vision Escalation",
        "weights_gb": 60.2,
        "resident_gb": 68.0,  # fits 1x 80 GB GPU or 4x 24 GB GPUs (TP=4)
        "max_tp": 4,
        "runs_on_any_device": False,
        "needs_gpu_interactive": True,
        "local_runtime": False,
        "min_vram_gb": 72,
        "min_servers": 1,
        "min_server_spec": "1x 80 GB GPU (A100/H100) OR 1x 4-GPU 24 GB server (>= 72 GB VRAM)",
        "interactive_speed": "22–38 tok/s on >= 72 GB GPU (< 0.8 tok/s on CPU — not interactive)",
    },
    {
        "id": "kimi-k3",
        "name": "Kimi K3.1 Pro (2.8T MoE K31H)",
        "tier": "big_gpu",
        "role": "Frontier 2.8T Deep Derivation Escalation",
        "weights_gb": 1420.0,
        "resident_gb": 1480.0,
        "max_tp": 8,
        "multi_node": True,
        "needs_same_rack": True,
        "runs_on_any_device": False,
        "needs_gpu_interactive": True,
        "local_runtime": False,
        "min_vram_gb": 1400,
        "min_servers": 8,
        "min_server_spec": "Same-rack GPU cluster with >= 1,400 GB VRAM (e.g. 8x 8-GPU 24 GB servers or 3x 8x80 GB H100 nodes)",
        "interactive_speed": "6–18 tok/s in-rack VRAM (< 0.2 tok/s on CPU/PCIe — not interactive)",
    },
]

_MODEL_BY_ID = {m["id"]: m for m in CLOVER_V6_MODEL_CATALOG}

_HOST_RE = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$"
    r"|^\d{1,3}(?:\.\d{1,3}){3}$|^\[[0-9A-Fa-f:]+\]$"
)
_USER_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,31}$")
_EMAIL_DOMAINS = {"gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "yahoo.com", "icloud.com", "proton.me"}


def parse_target(text: str) -> dict[str, Any]:
    """Strictly validates an SSH or HTTP(S) target, preventing SSH option injection (`-oProxyCommand=...`)."""
    t = str(text or "").strip()
    if not t:
        raise ValueError("empty server target")
    if t.lower() == "k3":
        return {"scheme": "https", "host": "api.cloverframework.com", "port": 443, "target": "https://api.cloverframework.com"}

    is_ssh = t.startswith("ssh://") or ("@" in t and not t.startswith(("http://", "https://")))
    if is_ssh:
        raw_u = t if t.startswith("ssh://") else ("ssh://" + t)
        u = urllib.parse.urlparse(raw_u)
        user = u.username or "root"
        host = u.hostname or ""
        port = u.port or 22
        if not u.port and host.lower() in _EMAIL_DOMAINS:
            raise ValueError(f"{t!r} looks like an email address — enter email/GitHub accounts in the Hosted models box above")
        if user.startswith("-") or host.startswith("-") or not _USER_RE.match(user) or not _HOST_RE.match(host) or not (1 <= port <= 65535) or u.path not in ("", "/"):
            raise ValueError(f"invalid SSH server target {t!r}")
        return {"scheme": "ssh", "user": user, "host": host, "port": port, "target": f"ssh://{user}@{host}:{port}"}

    raw_h = t if "://" in t else ("http://" + t)
    u = urllib.parse.urlparse(raw_h)
    if u.scheme not in ("http", "https") or not u.hostname or u.hostname.startswith("-") or not _HOST_RE.match(u.hostname) or u.username:
        raise ValueError(f"invalid HTTP(S) server target {t!r}")
    port = u.port or (443 if u.scheme == "https" else 80)
    if not (1 <= port <= 65535):
        raise ValueError(f"invalid port in {t!r}")
    clean_path = u.path.rstrip("/")
    if clean_path.endswith("/v1"):
        clean_path = clean_path[:-3]
    return {
        "scheme": u.scheme,
        "host": u.hostname,
        "port": port,
        "target": f"{u.scheme}://{u.netloc}{clean_path}",
    }


def _check_ssrf_safe_for_web(host: str) -> None:
    """Blocks loopback, private, link-local, and metadata IPs when probing user URLs from the shared Web server."""
    if host.lower() == "api.cloverframework.com":
        return
    infos = socket.getaddrinfo(host, None)
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise ValueError(f"refusing private/loopback address {host} from shared Web Studio")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _build_bootstrap_cmd(user: str, host: str, port: int, _api_key: str | None = None) -> str:
    remote_sh = (
        "test -d /opt/clover/.git || git clone --depth 1 https://github.com/santhoshsathish94/clover-framework.git /opt/clover; "
        "cd /opt/clover && (pgrep -f 'gateway.py.*--port 8790' >/dev/null || "
        "nohup python3 clover-ai-system/src/tools/hg4-fusion-gateway/gateway.py --mode real --host 0.0.0.0 --port 8790 --api-key \"$CLOVER_GATEWAY_KEY\" >/var/log/clover-gw.log 2>&1 &)"
    )
    return f"ssh -p {int(port)} -l {shlex.quote(user)} -- {shlex.quote(host)} {shlex.quote(remote_sh)}"


def detect_local_machine_spec(hardware_hint: dict[str, Any] | None = None, *, is_android: bool = False, is_web: bool = False) -> dict[str, Any]:
    """Detects this device's RAM and discrete per-GPU VRAM safely across Phone, Laptop, and Server."""
    hw = dict(hardware_hint or {})
    ram_gb = float(hw.get("ram_gb") or 0.0)
    if ram_gb <= 0 and isinstance(hw.get("memory_mb"), (int, float)):
        ram_gb = round(float(hw["memory_mb"]) / 1024.0, 1)
    if ram_gb <= 0:
        try:
            if os.path.isfile("/proc/meminfo"):
                txt = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore")
                m = re.search(r"MemTotal:\s*(\d+)\s*kB", txt)
                if m:
                    ram_gb = round(int(m.group(1)) / (1024 * 1024), 1)
            elif os.name == "nt":
                import ctypes

                class _MEMSTAT(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                st = _MEMSTAT()
                st.dwLength = ctypes.sizeof(_MEMSTAT)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                    ram_gb = round(st.ullTotalPhys / (1024 ** 3), 1)
        except Exception:
            ram_gb = 8.0
    if ram_gb <= 0:
        ram_gb = 8.0

    gpus: list[dict[str, Any]] = []
    # Android Adreno/Mali uses unified phone RAM, not dedicated discrete server VRAM
    if not is_android:
        if "vram_gb" in hw and float(hw.get("vram_gb") or 0.0) > 0:
            cnt = max(1, int(hw.get("gpu_count") or 1))
            per = round(float(hw["vram_gb"]) / cnt, 1)
            gname = str(hw.get("gpu_name") or hw.get("gpu") or "GPU")
            gpus = [{"name": gname, "vram_gb": per} for _ in range(cnt)]
        elif isinstance(hw.get("gpus"), list):
            for g in hw["gpus"]:
                if not isinstance(g, dict) or g.get("software"):
                    continue
                nm = str(g.get("name") or "GPU")
                if any(s in nm.lower() for s in ("adreno", "mali", "apple", "uhd graphics", "iris xe", "radeon graphics")):
                    continue
                mb = float(g.get("memory_mb") or g.get("vram_mb") or 0.0)
                if mb >= 4096:
                    gpus.append({"name": nm, "vram_gb": round(mb / 1024.0, 1)})
        if not gpus and shutil.which("nvidia-smi"):
            try:
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=4,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    for ln in proc.stdout.strip().splitlines():
                        parts = [p.strip() for p in ln.split(",")]
                        if len(parts) >= 2:
                            mib = float(re.sub(r"[^\d.]", "", parts[1]) or 0.0)
                            if mib > 0:
                                gpus.append({"name": parts[0], "vram_gb": round(mib / 1024.0, 1)})
            except Exception:
                pass

    vram_gb = round(sum(g["vram_gb"] for g in gpus), 1)
    dev_title = "k3 Web Host" if is_web else ("This phone" if is_android else "This device")
    label = (
        f"{dev_title} ({len(gpus)}x {gpus[0]['name']}, {vram_gb:.0f} GB VRAM, {ram_gb:.0f} GB RAM)"
        if gpus
        else f"{dev_title} (CPU / NPU, 0 GB discrete VRAM, {ram_gb:.0f} GB RAM)"
    )
    return {
        "id": "local",
        "target": "local",
        "label": label,
        "kind": "gpu_machine" if vram_gb >= 8.0 else "device_or_cpu",
        "reachable": True,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "vram_gb": vram_gb,
        "ram_gb": ram_gb,
        "gpu_name": gpus[0]["name"] if gpus else "CPU-only",
        "served_models": [],
    }


def _extract_trailing_hint(raw_text: str) -> tuple[str, list[dict[str, Any]] | None, bool]:
    """Extracts an explicit trailing `(4x24GB)`, `[80GB]`, `(cpu)`, or `[same-rack]` hint without matching hostnames."""
    text = str(raw_text or "").strip()
    same_rack = False
    gpus_hint: list[dict[str, Any]] | None = None
    while True:
        m = re.search(r"\s*(?:\(([^()]*)\)|\[([^\[\]]*)\])\s*$", text)
        if not m:
            break
        inner = (m.group(1) if m.group(1) is not None else m.group(2)).strip()
        text = text[: m.start()].strip()
        if re.search(r"\bsame[-_\s]?rack\b", inner, flags=re.I):
            same_rack = True
        m_mul = re.search(r"(\d+)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*GB", inner, flags=re.I)
        if m_mul:
            cnt = int(m_mul.group(1))
            per = float(m_mul.group(2))
            gpus_hint = [{"name": "Declared GPU", "vram_gb": per} for _ in range(cnt)]
            continue
        m_one = re.search(r"^(\d+(?:\.\d+)?)\s*GB(?:\s*VRAM)?$", inner, flags=re.I)
        if m_one:
            v = float(m_one.group(1))
            gpus_hint = [{"name": "Declared GPU", "vram_gb": v}] if v > 0 else []
            continue
        if re.search(r"^cpu(?:-only)?$", inner, flags=re.I):
            gpus_hint = []
    return text, gpus_hint, same_rack


def probe_machine_target(
    raw_spec: str | dict[str, Any],
    *,
    api_key: str = "",
    allow_ssh: bool = True,
    web_mode: bool = False,
    verify_live_completion: bool = True,
    only_models: set[str] | None = None,
) -> dict[str, Any]:
    """Validates and probes a remote server (dict, SSH host, or HTTP(S) Clover/OpenAI gateway)."""
    if isinstance(raw_spec, dict):
        raw_tgt = str(raw_spec.get("target") or raw_spec.get("url") or raw_spec.get("host") or "").strip()
        per_key = str(raw_spec.get("api_key") or api_key or "").strip()
        if "vram_gb" in raw_spec or "gpus" in raw_spec:
            parsed = parse_target(raw_tgt) if "://" in raw_tgt or "@" in raw_tgt or "." in raw_tgt else {
                "scheme": "spec", "host": re.sub(r"[^A-Za-z0-9_.-]", "-", raw_tgt or "server"), "port": 0, "target": raw_tgt or "server"
            }
            if isinstance(raw_spec.get("gpus"), list):
                gpus = [{"name": str(g.get("name") or "GPU"), "vram_gb": float(g.get("vram_gb") or 0.0)} for g in raw_spec["gpus"]]
            else:
                tot_v = float(raw_spec.get("vram_gb") or 0.0)
                cnt = max(1, int(raw_spec.get("gpu_count") or 1)) if tot_v > 0 else 0
                gpus = [{"name": str(raw_spec.get("gpu_name") or "GPU"), "vram_gb": round(tot_v / cnt, 1)} for _ in range(cnt)]
            vram = round(sum(g["vram_gb"] for g in gpus), 1)
            ram = float(raw_spec.get("ram_gb") or 64.0)
            custom_lbl = raw_spec.get("label")
            lbl = str(custom_lbl) if custom_lbl else (
                f"{parsed['target']} ({len(gpus)}x {gpus[0]['name']}, {vram:.0f} GB VRAM)" if vram > 0 else f"{parsed['target']} (CPU-only, {ram:.0f} GB RAM)"
            )
            return {
                "id": parsed["target"],
                "target": parsed["target"],
                "label": lbl,
                "kind": "gpu_server" if vram > 0 else "cpu_server",
                "reachable": raw_spec.get("reachable", True),
                "gpu_count": len(gpus),
                "gpus": gpus,
                "vram_gb": vram,
                "ram_gb": ram,
                "gpu_name": gpus[0]["name"] if gpus else "CPU-only",
                "same_rack": bool(raw_spec.get("same_rack", False)),
                "endpoint_url": parsed["target"] if parsed["scheme"] in ("http", "https") else None,
                "served_models": [canon_model_id(m) for m in (raw_spec.get("served_models") or ())],
                "api_key": per_key,
            }
        return probe_machine_target(raw_tgt, api_key=per_key, allow_ssh=allow_ssh, web_mode=web_mode, verify_live_completion=verify_live_completion, only_models=only_models)

    clean_text, gpus_hint, same_rack = _extract_trailing_hint(str(raw_spec or ""))
    parsed = parse_target(clean_text)

    # 1. SSH Server Target
    if parsed["scheme"] == "ssh":
        boot_cmd = _build_bootstrap_cmd(parsed["user"], parsed["host"], parsed["port"], api_key or None)
        if gpus_hint is not None:
            vram = round(sum(g["vram_gb"] for g in gpus_hint), 1)
            return {
                "id": parsed["target"],
                "target": parsed["target"],
                "label": f"{parsed['target']} (declared {len(gpus_hint)}x GPU, {vram:.0f} GB VRAM — preview)" if vram > 0 else f"{parsed['target']} (declared CPU-only, 0 GB VRAM)",
                "kind": "gpu_server" if vram > 0 else "cpu_server",
                "reachable": None,  # declared hint, not yet live-verified
                "gpu_count": len(gpus_hint),
                "gpus": gpus_hint,
                "vram_gb": vram,
                "ram_gb": 64.0,
                "gpu_name": "Declared GPU" if vram > 0 else "CPU-only",
                "same_rack": same_rack,
                "endpoint_url": None,
                "served_models": [],
                "bootstrap_cmd": boot_cmd,
                "api_key": api_key,
            }
        if allow_ssh and not web_mode and shutil.which("ssh"):
            try:
                ssh_probe = (
                    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo NO_GPU; "
                    "grep MemTotal /proc/meminfo 2>/dev/null || true"
                )
                cmd = [
                    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                    "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=5",
                    "-p", str(parsed["port"]), "-l", parsed["user"], "--", parsed["host"], ssh_probe,
                ]
                pr = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
                if pr.returncode == 0:
                    gpus, ram_tot = [], 32.0
                    for ln in pr.stdout.splitlines():
                        ln = ln.strip()
                        if not ln or ln == "NO_GPU":
                            continue
                        if ln.startswith("MemTotal:"):
                            mk = re.search(r"(\d+)", ln)
                            if mk:
                                ram_tot = round(int(mk.group(1)) / (1024 * 1024), 1)
                        elif "," in ln:
                            pts = [p.strip() for p in ln.split(",")]
                            mib = float(re.sub(r"[^\d.]", "", pts[1]) or 0.0) if len(pts) >= 2 else 0.0
                            if mib > 0:
                                gpus.append({"name": pts[0], "vram_gb": round(mib / 1024.0, 1)})
                    vram_tot = round(sum(g["vram_gb"] for g in gpus), 1)
                    return {
                        "id": parsed["target"],
                        "target": parsed["target"],
                        "label": f"{parsed['target']} ({len(gpus)}x {gpus[0]['name']}, {vram_tot:.0f} GB VRAM, {ram_tot:.0f} GB RAM)" if gpus else f"{parsed['target']} (CPU-only server, 0 GB VRAM, {ram_tot:.0f} GB RAM)",
                        "kind": "gpu_server" if vram_tot > 0 else "cpu_server",
                        "reachable": True,
                        "gpu_count": len(gpus),
                        "gpus": gpus,
                        "vram_gb": vram_tot,
                        "ram_gb": ram_tot,
                        "gpu_name": gpus[0]["name"] if gpus else "CPU-only",
                        "same_rack": same_rack,
                        "endpoint_url": None,
                        "served_models": [],
                        "bootstrap_cmd": boot_cmd,
                        "api_key": api_key,
                    }
            except Exception:
                pass
        why_note = "SSH disabled on Web/Mobile — run bootstrap command on your server and paste its HTTP(S) URL" if (web_mode or not shutil.which("ssh")) else "SSH unreachable — check key or append (1x24GB) to preview"
        return {
            "id": parsed["target"],
            "target": parsed["target"],
            "label": f"{parsed['target']} ({why_note})",
            "kind": "cpu_server",
            "reachable": False,
            "gpu_count": 0,
            "gpus": [],
            "vram_gb": 0.0,
            "ram_gb": 16.0,
            "gpu_name": "Unverified",
            "same_rack": same_rack,
            "endpoint_url": None,
            "served_models": [],
            "bootstrap_cmd": boot_cmd,
            "api_key": api_key,
        }

    # 2. HTTP(S) Model Gateway Target (`gateway.py`, Studio, vLLM, llama-server, Ollama)
    url = parsed["target"]
    if web_mode:
        _check_ssrf_safe_for_web(parsed["host"])

    if gpus_hint is not None:
        vram = round(sum(g["vram_gb"] for g in gpus_hint), 1)
        return {
            "id": url,
            "target": url,
            "label": f"{url} (declared {len(gpus_hint)}x GPU, {vram:.0f} GB VRAM — preview)" if vram > 0 else f"{url} (declared CPU-only, 0 GB VRAM)",
            "kind": "gpu_server" if vram > 0 else "cpu_server",
            "reachable": None,
            "gpu_count": len(gpus_hint),
            "gpus": gpus_hint,
            "vram_gb": vram,
            "ram_gb": 64.0,
            "gpu_name": "Declared GPU" if vram > 0 else "CPU-only",
            "same_rack": same_rack,
            "endpoint_url": url,
            "served_models": [],
            "api_key": api_key,
        }

    opener = urllib.request.build_opener(_NoRedirect) if web_mode else urllib.request.build_opener()
    headers = {"User-Agent": "CloverStudio/6.0"}
    if api_key:
        headers["X-Clover-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"

    served_real: list[str] = []
    served_tps: dict[str, float] = {}
    reachable = False
    gpus_detected: list[dict[str, Any]] = []
    ram_detected = 64.0

    # 2a. Probe `/health` (Clover HG4 Fusion Gateway) — ignore mock backends!
    try:
        req = urllib.request.Request(url + "/health", headers=headers)
        with opener.open(req, timeout=5) as resp:
            h_data = json.loads(resp.read().decode("utf-8", "replace"))
        reachable = True
        backends = h_data.get("backends") or {}
        if isinstance(backends, dict):
            for b_id, b_info in backends.items():
                b_kind = str((b_info or {}).get("kind") or "")
                if b_kind and b_kind not in ("mock", "proxy"):
                    served_real.append(canon_model_id(b_id))
    except Exception:
        pass

    # 2b. Probe `/v1/models` (OpenAI / HG4 Gateway / vLLM / llama-server)
    try:
        req = urllib.request.Request(url + "/v1/models", headers=headers)
        with opener.open(req, timeout=5) as resp:
            m_data = json.loads(resp.read().decode("utf-8", "replace"))
        reachable = True
        for item in (m_data.get("data") or ()):
            if isinstance(item, dict) and item.get("id"):
                if item.get("kind") == "mock":
                    continue
                cid = canon_model_id(str(item["id"]))
                if cid not in served_real:
                    served_real.append(cid)
    except Exception:
        pass

    # 2c. Probe `/api/settings` if target is another Clover Studio server
    try:
        req = urllib.request.Request(url + "/api/settings", headers=headers)
        with opener.open(req, timeout=4) as resp:
            s_data = json.loads(resp.read().decode("utf-8", "replace"))
        reachable = True
        hw = s_data.get("hardware") or {}
        for g in (hw.get("gpus") or ()):
            if isinstance(g, dict) and not g.get("software") and float(g.get("memory_mb") or 0.0) >= 4096:
                gpus_detected.append({"name": str(g.get("name") or "GPU"), "vram_gb": round(float(g["memory_mb"]) / 1024.0, 1)})
        for mc in (s_data.get("model_catalog") or ()):
            if isinstance(mc, dict) and mc.get("id") and mc["id"] != "auto":
                cid = canon_model_id(str(mc.get("remote_model") or mc["id"]))
                if cid not in served_real:
                    served_real.append(cid)
    except Exception:
        pass

    # 2d. Verify interactive decode speed on requested served big models (multi-token probe so WAN RTT doesn't skew 1-token TTFT!)
    interactive_served: list[str] = []
    slow_served: dict[str, float] = {}
    if reachable and served_real and verify_live_completion:
        for cid in list(served_real):
            if only_models is not None and cid not in only_models:
                continue
            gw_id, chat_ok, _ = CANON.get(cid, (cid, False, None))
            if not chat_ok or not gw_id:
                interactive_served.append(cid)
                continue
            try:
                body_p = json.dumps({
                    "model": gw_id,
                    "messages": [{"role": "user", "content": "Output the integers 1 through 16 separated by spaces only."}],
                    "max_tokens": 24,
                    "temperature": 0.0,
                    "stream": False,
                }).encode("utf-8")
                req_c = urllib.request.Request(
                    url + "/v1/chat/completions",
                    data=body_p,
                    headers=dict(headers, **{"Content-Type": "application/json"}),
                    method="POST",
                )
                t_p = time.perf_counter()
                with opener.open(req_c, timeout=18) as resp_c:
                    c_data = json.loads(resp_c.read().decode("utf-8", "replace"))
                dt_s = max(0.001, time.perf_counter() - t_p)
                telem = c_data.get("x_clover_telemetry") or {}
                if telem.get("backend_kind") == "mock":
                    continue
                choices_p = c_data.get("choices") or []
                msg_txt = str(((choices_p[0].get("message") or {}).get("content")) if choices_p else "").strip()
                if not msg_txt:
                    continue
                comp_t = int((c_data.get("usage") or {}).get("completion_tokens") or max(4, len(msg_txt.split())))
                # Subtract ~0.25s WAN RTT/prefill allowance when server does not report pure GPU decode_tok_per_s
                net_adj_dt = max(0.05, dt_s - 0.25) if dt_s > 0.35 else dt_s
                tps = float(telem.get("decode_tok_per_s") or round(comp_t / net_adj_dt, 1))
                served_tps[cid] = round(tps, 1)
                spec_c = _MODEL_BY_ID.get(cid)
                # CPU-offloaded 35B/120B crawls at 0.1–1.8 tok/s; interactive GPU decode is >= 5 tok/s
                if spec_c and spec_c["needs_gpu_interactive"] and tps < 2.5:
                    slow_served[cid] = round(tps, 1)
                else:
                    interactive_served.append(cid)
            except Exception:
                pass
    else:
        interactive_served = list(served_real)

    vram_tot = round(sum(g["vram_gb"] for g in gpus_detected), 1)
    if not reachable:
        return {
            "id": url,
            "target": url,
            "label": f"{url} (unreachable or unauthorized)",
            "kind": "cpu_server",
            "reachable": False,
            "gpu_count": 0,
            "gpus": [],
            "vram_gb": 0.0,
            "ram_gb": 16.0,
            "gpu_name": "Unreachable",
            "same_rack": same_rack,
            "endpoint_url": url,
            "served_models": [],
            "served_tps": {},
            "slow_served": {},
            "api_key": api_key,
        }

    served_desc = (
        f"serving {', '.join(f'{m} ({served_tps[m]} tok/s)' if m in served_tps else m for m in interactive_served)}"
        if interactive_served
        else (f"{len(gpus_detected)}x GPU, {vram_tot:.0f} GB VRAM" if vram_tot > 0 else "CPU server, 0 GB GPU VRAM")
    )
    return {
        "id": url,
        "target": url,
        "label": f"{url} ({served_desc})",
        "kind": "gpu_server" if (vram_tot > 0 or any(_MODEL_BY_ID.get(m, {}).get("needs_gpu_interactive") for m in interactive_served)) else "cpu_server",
        "reachable": True,
        "gpu_count": len(gpus_detected),
        "gpus": gpus_detected,
        "vram_gb": vram_tot,
        "ram_gb": ram_detected,
        "gpu_name": gpus_detected[0]["name"] if gpus_detected else ("Remote GPU Gateway" if interactive_served else "CPU-only"),
        "same_rack": same_rack,
        "endpoint_url": url,
        "served_models": interactive_served,
        "served_tps": served_tps,
        "slow_served": slow_served,
        "api_key": api_key,
    }


def setup_chat_entries(saved: dict[str, Any]) -> list[dict[str, Any]]:
    """Converts ready server placements into selectable Studio `remote_models` picker entries."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p in (saved.get("placements") or ()):
        st = str(p.get("status") or "")
        ep = str(p.get("endpoint_url") or "").strip()
        if st not in ("ready_served", "ready_pinned", "ready_swap") or not ep:
            continue
        mid = str(p.get("model_id") or "")
        gw_id, chat_ok, _ = CANON.get(mid, (None, False, None))
        if not chat_ok or not gw_id:
            continue
        host = urllib.parse.urlparse(ep).hostname or "server"
        port = urllib.parse.urlparse(ep).port
        host_tag = f"{host}-{port}" if port and port not in (80, 443) else host
        entry_id = f"v6:{mid}@{re.sub(r'[^a-z0-9-]+', '-', host_tag.lower())[:44]}"
        if entry_id in seen:
            continue
        seen.add(entry_id)
        sp = _MODEL_BY_ID.get(mid, {})
        mode_lbl = "live" if st == "ready_served" else ("pinned" if st == "ready_pinned" else "on-demand")
        out.append({
            "id": entry_id,
            "name": str(sp.get("name") or mid).split(" (")[0][:80],
            "badge": f"{host[:22]} · {mode_lbl}",
            "desc": str(p.get("reason") or sp.get("role") or "")[:240],
            "host": host[:60],
            "url": ep[:240],
            "remote_model": gw_id,
            "kind": "openai",
            "key_id": str(p.get("key_id") or "")[:60],
            "source": "v6_setup",
        })
    return out


def setup_mcp_routes(saved: dict[str, Any]) -> dict[str, dict[str, str]]:
    routes: dict[str, dict[str, str]] = {}
    for p in (saved.get("placements") or ()):
        st = str(p.get("status") or "")
        ep = str(p.get("endpoint_url") or "").strip()
        mid = str(p.get("model_id") or "")
        _, _, tool_name = CANON.get(mid, (None, False, None))
        if tool_name and st in ("ready_served", "ready_pinned", "ready_swap") and ep:
            routes[tool_name] = {
                "url": ep,
                "key_id": str(p.get("key_id") or ""),
                "model": mid,
            }
    return routes


# ---------------------------------------------------------------------------
# 5. Studio-Wide Clover v6.0 4-Part Coordinator (`CloverStudioV6`)
# ---------------------------------------------------------------------------
_AMBIGUOUS_RISK_RE = re.compile(
    r"\b(?:rm\s+-rf|drop\s+table|delete\s+all\s+production|format\s+disk|wipe\s+database)\b"
    r"|\b(?:deploy\s+it\s+to\s+production|migrate\s+our\s+auth\s+database)\b(?!.*(?:postgres|mysql|sqlite|staging|canary|aws|gcp|azure))",
    re.I,
)
_EXACT_MATH_RE = re.compile(
    r"(?:(\d[\d_]*)!|factorial\(\s*(\d[\d_]*)\s*\)|(\d[\d_]*)\s*(?:\^|\*\*)\s*(\d[\d_]*)|pi_hex\(\s*(\d[\d_]*)(?:\s*,\s*(\d+))?\s*\))",
    re.I,
)
_SINGLE_FUNC_CODE_RE = re.compile(r"^\s*(?:write|implement|complete|create)\s+(?:a\s+)?(?:python\s+)?function\b", re.I)


class CloverStudioV6:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        studio_root = self.base_dir / "clover-ai-studio"
        studio_root.mkdir(parents=True, exist_ok=True)
        self.contexts_dir = studio_root / "contexts"
        self.contexts_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = CloverMetricsLedger(studio_root / "clover_v6_metrics.sqlite3")
        self.todos = CloverTodoStore(studio_root / "clover_v6_todos.sqlite3")
        self.agents_md_path = Path(__file__).resolve().parent / "AGENTS.md"
        self.setup_state_path = studio_root / "clover_v6_model_setup.json"
        self._setup_lock = threading.Lock()
        self._plan_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def close(self) -> None:
        self.metrics.close()
        self.todos.close()

    def load_saved_setup(self) -> dict[str, Any]:
        with self._setup_lock:
            if self.setup_state_path.is_file():
                try:
                    return json.loads(self.setup_state_path.read_text(encoding="utf-8"))
                except Exception:
                    return {}
            return {}

    def setup_catalog(self, hardware_hint: dict[str, Any] | None = None, *, is_android: bool = False, is_web: bool = False) -> dict[str, Any]:
        local_spec = detect_local_machine_spec(hardware_hint, is_android=is_android, is_web=is_web)
        return {
            "version": "6.0.0",
            "rule": "Small models & 0B tools run interactively on any phone, laptop, or CPU machine. Big models require GPUs to be interactive and are skipped when CPU-only or insufficient GPU VRAM is provided.",
            "local_machine": local_spec,
            "models": CLOVER_V6_MODEL_CATALOG,
            "active_setup": self.load_saved_setup(),
        }

    def plan_system_setup(
        self,
        selected_models: list[str] | None = None,
        servers: list[str | dict[str, Any]] | None = None,
        *,
        include_local: bool = True,
        api_key: str = "",
        hardware_hint: dict[str, Any] | None = None,
        is_android: bool = False,
        is_web: bool = False,
        verify_live_completion: bool = True,
    ) -> dict[str, Any]:
        raw_req = [str(m).strip() for m in (selected_models or ()) if str(m).strip()]
        if not raw_req or "all" in raw_req or "full" in raw_req:
            req_ids = [m["id"] for m in CLOVER_V6_MODEL_CATALOG]
            unknown_models: list[str] = []
        else:
            req_ids = []
            unknown_models = []
            seen_m: set[str] = set()
            for rm in raw_req:
                cid = canon_model_id(rm)
                if cid in _MODEL_BY_ID:
                    if cid not in seen_m:
                        seen_m.add(cid)
                        req_ids.append(cid)
                else:
                    unknown_models.append(rm)

        machines: list[dict[str, Any]] = []
        probe_errors: list[dict[str, str]] = []
        seen_targets: set[str] = set()

        if include_local:
            loc = detect_local_machine_spec(hardware_hint, is_android=is_android, is_web=is_web)
            seen_targets.add(loc["target"])
            machines.append(loc)

        raw_sv_list = [sv for sv in (servers or ()) if not (isinstance(sv, str) and not sv.strip())]
        req_set = set(req_ids)
        if raw_sv_list:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(raw_sv_list))) as pool:
                futs = [
                    pool.submit(
                        probe_machine_target,
                        sv,
                        api_key=api_key,
                        allow_ssh=(not is_android and not is_web),
                        web_mode=is_web,
                        verify_live_completion=verify_live_completion,
                        only_models=req_set,
                    )
                    for sv in raw_sv_list
                ]
                for sv_in, fut in zip(raw_sv_list, futs):
                    try:
                        m_res = fut.result(timeout=15)
                        if m_res["target"] not in seen_targets:
                            seen_targets.add(m_res["target"])
                            machines.append(m_res)
                    except Exception as exc:
                        probe_errors.append({"target": str(sv_in), "error": str(exc)})

        if not machines:
            machines.append(detect_local_machine_spec(hardware_hint, is_android=is_android, is_web=is_web))

        # Per-GPU VRAM capacity & remaining free VRAM for First-Fit-Decreasing placement
        free_gpus: dict[str, list[float]] = {}
        cap_gpus: dict[str, list[float]] = {}
        for m in machines:
            if m.get("reachable") in (True, None):
                g_list = [float(g["vram_gb"]) for g in (m.get("gpus") or ())]
                free_gpus[m["id"]] = list(g_list)
                cap_gpus[m["id"]] = list(g_list)
        machine_by_id = {m["id"]: m for m in machines}

        # Evaluate big models largest-first so 72 GB / 24 GB allocations pack deterministically
        big_reqs = sorted(
            [_MODEL_BY_ID[mid] for mid in req_ids if _MODEL_BY_ID[mid]["needs_gpu_interactive"]],
            key=lambda s: -float(s["resident_gb"]),
        )
        big_results: dict[str, dict[str, Any]] = {}

        for spec in big_reqs:
            mid = spec["id"]
            need = float(spec["resident_gb"])
            tp_max = int(spec.get("max_tp") or 1)

            # A. Already actively served at interactive speed on a live HTTP gateway?
            served_host = next(
                (m for m in machines if m.get("reachable") is True and mid in (m.get("served_models") or ())),
                None,
            )
            if served_host is not None:
                tps = (served_host.get("served_tps") or {}).get(mid)
                tps_lbl = f"{tps} tok/s measured" if tps else spec["interactive_speed"]
                ep = served_host.get("endpoint_url")
                kid = ("rm_" + hashlib.sha256(str(ep).encode("utf-8")).hexdigest()[:12]) if (ep and served_host.get("api_key")) else ""
                big_results[mid] = {
                    "model_id": mid,
                    "name": spec["name"],
                    "tier": spec["tier"],
                    "status": "ready_served",
                    "interactive": True,
                    "assigned_to": served_host["label"],
                    "assigned_target": served_host["target"],
                    "endpoint_url": ep,
                    "key_id": kid,
                    "min_server_spec": spec["min_server_spec"],
                    "interactive_speed": tps_lbl,
                    "reason": f"Live & interactive on {served_host['target']} ({tps_lbl}).",
                }
                continue

            # B. Multi-node same-rack cluster model (Kimi K3.1 Pro 2.8T)
            if spec.get("multi_node"):
                rack_hosts = [
                    m for m in machines
                    if m.get("reachable") in (True, None)
                    and float(m.get("vram_gb") or 0.0) > 0
                    and (m.get("same_rack") or len(machines) == 1)
                ]
                rem_cluster = sum(sum(free_gpus.get(m["id"], ())) for m in rack_hosts)
                min_sv = int(spec.get("min_servers") or 8)
                if len(rack_hosts) >= min_sv and rem_cluster >= need:
                    is_preview = any(m.get("reachable") is None for m in rack_hosts)
                    for m in rack_hosts:
                        free_gpus[m["id"]] = [0.0 for _ in free_gpus.get(m["id"], ())]
                    big_results[mid] = {
                        "model_id": mid,
                        "name": spec["name"],
                        "tier": spec["tier"],
                        "status": "preview_unverified" if is_preview else "ready_pinned",
                        "interactive": not is_preview,
                        "assigned_to": f"Same-rack GPU cluster ({len(rack_hosts)} servers, {rem_cluster:.0f} GB VRAM)",
                        "assigned_target": ",".join(m["target"] for m in rack_hosts),
                        "endpoint_url": rack_hosts[0].get("endpoint_url"),
                        "key_id": "",
                        "min_server_spec": spec["min_server_spec"],
                        "interactive_speed": spec["interactive_speed"],
                        "reason": f"Fits across {len(rack_hosts)} same-rack GPU servers ({rem_cluster:.0f} GB free VRAM).",
                    }
                else:
                    tot_any_vram = sum(float(m.get("vram_gb") or 0.0) for m in machines if m.get("reachable") in (True, None))
                    why = (
                        f"Skipped: this machine has no GPU (CPU-only) and cannot run {spec['name']} interactively (<0.2 tok/s on CPU). "
                        f"Minimum required: {spec['min_server_spec']}."
                        if tot_any_vram <= 0
                        else f"Skipped: provided machines cannot run {spec['name']} interactively "
                             f"(need >= {min_sv} same-rack GPU servers with >= {spec['min_vram_gb']} GB free VRAM; found {len(rack_hosts)} same-rack host(s), {rem_cluster:.0f} GB free VRAM)."
                    )
                    big_results[mid] = {
                        "model_id": mid,
                        "name": spec["name"],
                        "tier": spec["tier"],
                        "status": "skipped_not_interactive",
                        "interactive": False,
                        "assigned_to": None,
                        "assigned_target": None,
                        "endpoint_url": None,
                        "key_id": "",
                        "min_server_spec": spec["min_server_spec"],
                        "interactive_speed": spec["interactive_speed"],
                        "reason": why,
                    }
                continue

            # C. Single-node / Intra-node Tensor-Parallel GPU placement (`ready_pinned` vs `ready_swap`)
            best_fit: tuple[str, str, list[int], float] | None = None
            for m_id, gfree in free_gpus.items():
                if not gfree:
                    continue
                order = sorted(range(len(gfree)), key=lambda idx: -gfree[idx])
                gcap = cap_gpus[m_id]
                order_cap = sorted(range(len(gcap)), key=lambda idx: -gcap[idx])
                for k in (1, 2, 4, 8):
                    if k > tp_max or k > len(order):
                        break
                    grp = order[:k]
                    per = (need / k) * (1.0 if k == 1 else 1.08)
                    if all(gfree[i] >= per for i in grp):
                        best_fit = ("ready_pinned", m_id, grp, per)
                        break
                    grp_c = order_cap[:k]
                    if best_fit is None and all(gcap[i] >= per for i in grp_c):
                        best_fit = ("ready_swap", m_id, grp_c, per)
                if best_fit and best_fit[0] == "ready_pinned":
                    break

            if best_fit is not None:
                st_mode, m_id, grp, per = best_fit
                m_obj = machine_by_id[m_id]
                if st_mode == "ready_pinned":
                    for i in grp:
                        free_gpus[m_id][i] = round(max(0.0, free_gpus[m_id][i] - per), 2)
                is_preview = m_obj.get("reachable") is None
                ep = m_obj.get("endpoint_url")
                final_status = "preview_unverified" if is_preview else ("needs_gateway" if not ep else st_mode)
                kid = ("rm_" + hashlib.sha256(str(ep).encode("utf-8")).hexdigest()[:12]) if (ep and m_obj.get("api_key")) else ""
                gpu_desc = f"{len(grp)}x GPU" if len(grp) > 1 else f"GPU{grp[0]}"
                why_ok = (
                    f"Preview (unverified hint) on {m_obj['target']} ({gpu_desc})."
                    if is_preview
                    else (
                        f"GPU verified on {m_obj['label']} ({gpu_desc}) — start Clover HG4 Gateway on that host and paste its HTTP URL."
                        if final_status == "needs_gateway"
                        else (
                            f"Pinned in VRAM on {m_obj['label']} ({gpu_desc}, {spec['interactive_speed']})."
                            if st_mode == "ready_pinned"
                            else f"On-demand GPU swap on {m_obj['label']} ({gpu_desc}; shares GPU VRAM with higher-priority model, ~{max(2, round(spec['weights_gb'] / 2.5))}s cold swap)."
                        )
                    )
                )
                big_results[mid] = {
                    "model_id": mid,
                    "name": spec["name"],
                    "tier": spec["tier"],
                    "status": final_status,
                    "interactive": final_status.startswith("ready_"),
                    "assigned_to": m_obj["label"],
                    "assigned_target": m_obj["target"],
                    "endpoint_url": ep,
                    "key_id": kid,
                    "min_server_spec": spec["min_server_spec"],
                    "interactive_speed": spec["interactive_speed"],
                    "reason": why_ok,
                }
            else:
                # Check if target served it slowly on CPU
                slow_host = next((m for m in machines if mid in (m.get("slow_served") or {})), None)
                max_gpu = max((max((g["vram_gb"] for g in (m.get("gpus") or ())), default=0.0) for m in machines), default=0.0)
                tot_vram = max((float(m.get("vram_gb") or 0.0) for m in machines), default=0.0)
                if slow_host is not None:
                    meas = slow_host["slow_served"][mid]
                    why = f"Skipped: {slow_host['target']} measured only {meas} tok/s — cannot run {spec['name']} interactively. Minimum required: {spec['min_server_spec']}."
                elif tot_vram <= 0:
                    why = f"Skipped: this machine has no discrete GPU (CPU-only) and cannot run {spec['name']} interactively. Minimum required: {spec['min_server_spec']}."
                else:
                    why = (
                        f"Skipped: provided GPU machine(s) (largest {tot_vram:.0f} GB total / {max_gpu:.0f} GB per GPU) cannot fit "
                        f"{spec['name']} ({need:.1f} GB resident VRAM, max TP={tp_max}) interactively. Minimum required: {spec['min_server_spec']}."
                    )
                big_results[mid] = {
                    "model_id": mid,
                    "name": spec["name"],
                    "tier": spec["tier"],
                    "status": "skipped_not_interactive",
                    "interactive": False,
                    "assigned_to": None,
                    "assigned_target": None,
                    "endpoint_url": None,
                    "key_id": "",
                    "min_server_spec": spec["min_server_spec"],
                    "interactive_speed": spec["interactive_speed"],
                    "reason": why,
                }

        # Assemble placements in user/catalog order, placing small models on local device first (or first reachable server)
        placements: list[dict[str, Any]] = []
        ready_ids: list[str] = []
        preview_ids: list[str] = []
        skipped_ids: list[str] = []

        local_m = next((m for m in machines if m["target"] == "local"), None)
        reach_m = next((m for m in machines if m.get("reachable") is True), machines[0])

        for mid in req_ids:
            spec = _MODEL_BY_ID[mid]
            if not spec["needs_gpu_interactive"]:
                # Prefer remote server if it explicitly serves this small specialist, else run on local phone/laptop/CPU!
                remote_srv = next((m for m in machines if m["target"] != "local" and m.get("reachable") is True and mid in (m.get("served_models") or ())), None)
                host_m = remote_srv or local_m or reach_m
                if host_m.get("reachable") is False:
                    skipped_ids.append(mid)
                    placements.append({
                        "model_id": mid,
                        "name": spec["name"],
                        "tier": spec["tier"],
                        "status": "skipped_unreachable",
                        "interactive": False,
                        "assigned_to": None,
                        "assigned_target": None,
                        "endpoint_url": None,
                        "key_id": "",
                        "min_server_spec": spec["min_server_spec"],
                        "interactive_speed": spec["interactive_speed"],
                        "reason": f"Skipped: target server {host_m['target']} is unreachable (enable 'Include this device' to run {spec['name']} locally).",
                    })
                    continue
                ep = host_m.get("endpoint_url") if remote_srv else None
                kid = ("rm_" + hashlib.sha256(str(ep).encode("utf-8")).hexdigest()[:12]) if (ep and host_m.get("api_key")) else ""
                note_extra = " (local 0B Periodogram+Conformal engine)" if (mid == "chronos-2" and remote_srv is None) else ""
                st_sm = "preview_unverified" if host_m.get("reachable") is None else ("ready_served" if remote_srv else "ready_local")
                if st_sm == "preview_unverified":
                    preview_ids.append(mid)
                else:
                    ready_ids.append(mid)
                placements.append({
                    "model_id": mid,
                    "name": spec["name"],
                    "tier": spec["tier"],
                    "status": st_sm,
                    "interactive": st_sm != "preview_unverified",
                    "assigned_to": host_m["label"],
                    "assigned_target": host_m["target"],
                    "endpoint_url": ep,
                    "key_id": kid,
                    "min_server_spec": spec["min_server_spec"],
                    "interactive_speed": spec["interactive_speed"],
                    "reason": f"Interactive on {host_m['label']}{note_extra} ({spec['interactive_speed']}).",
                })
            else:
                br = big_results[mid]
                if br["status"].startswith("ready_"):
                    ready_ids.append(mid)
                elif br["status"] in ("preview_unverified", "needs_gateway"):
                    preview_ids.append(mid)
                else:
                    skipped_ids.append(mid)
                placements.append(br)

        bootstrap_cmds = [m["bootstrap_cmd"] for m in machines if m.get("bootstrap_cmd")]
        plan_id = "plan_" + uuid.uuid4().hex[:12]
        plan = {
            "ok": True,
            "plan_id": plan_id,
            "version": "6.0.0",
            "machines": [{k: v for k, v in m.items() if k != "api_key"} for m in machines],
            "_machine_keys": {m["endpoint_url"]: m["api_key"] for m in machines if m.get("endpoint_url") and m.get("api_key")},
            "errors": probe_errors,
            "unknown_models": unknown_models,
            "selected_count": len(req_ids),
            "ready_count": len(ready_ids),
            "preview_count": len(preview_ids),
            "skipped_count": len(skipped_ids),
            "ready_models": ready_ids,
            "preview_models": preview_ids,
            "skipped_models": skipped_ids,
            "placements": placements,
            "ssh_bootstrap_commands": bootstrap_cmds,
            "summary": (
                f"{len(ready_ids)} of {len(req_ids)} selected model(s) ready to run interactively"
                + (f", {len(preview_ids)} unverified preview" if preview_ids else "")
                + (f", {len(skipped_ids)} big GPU model(s) skipped (cannot run interactively on provided hardware)." if skipped_ids else ".")
            ),
        }
        with self._setup_lock:
            now = time.time()
            self._plan_cache = {k: v for k, v in self._plan_cache.items() if now - v[0] < 600}
            self._plan_cache[plan_id] = (now, plan)
        return {k: v for k, v in plan.items() if k != "_machine_keys"}

    def get_cached_plan_with_keys(self, plan_id: str) -> dict[str, Any] | None:
        with self._setup_lock:
            item = self._plan_cache.get(str(plan_id or ""))
            if item and (time.time() - item[0] < 600):
                return dict(item[1])
            return None

    def apply_system_setup(
        self,
        selected_models: list[str] | None = None,
        servers: list[str | dict[str, Any]] | None = None,
        *,
        plan_id: str = "",
        include_local: bool = True,
        api_key: str = "",
        hardware_hint: dict[str, Any] | None = None,
        is_android: bool = False,
        is_web: bool = False,
    ) -> dict[str, Any]:
        cached = self.get_cached_plan_with_keys(plan_id) if plan_id else None
        if cached is None:
            new_plan = self.plan_system_setup(
                selected_models=selected_models,
                servers=servers,
                include_local=include_local,
                api_key=api_key,
                hardware_hint=hardware_hint,
                is_android=is_android,
                is_web=is_web,
            )
            cached = self.get_cached_plan_with_keys(new_plan["plan_id"]) or new_plan

        saved_record = {
            "applied_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ready_models": cached["ready_models"],
            "skipped_models": cached["skipped_models"],
            "placements": cached["placements"],
            "machines": [{k: v for k, v in m.items() if k != "bootstrap_cmd"} for m in cached["machines"]],
            "summary": cached["summary"],
        }
        with self._setup_lock:
            tmp = self.setup_state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(saved_record, indent=2), encoding="utf-8")
            os.replace(tmp, self.setup_state_path)

        pub = {k: v for k, v in cached.items() if k != "_machine_keys"}
        pub["applied"] = True
        pub["saved_to"] = str(self.setup_state_path)
        pub["chat_entries"] = setup_chat_entries(saved_record)
        pub["mcp_routes"] = setup_mcp_routes(saved_record)
        pub["_machine_keys"] = cached.get("_machine_keys") or {}
        return pub

    def mark_placement_removed(self, model_id: str, endpoint_url: str) -> None:
        cid = canon_model_id(model_id)
        ep_norm = str(endpoint_url or "").rstrip("/")
        with self._setup_lock:
            if not self.setup_state_path.is_file():
                return
            try:
                rec = json.loads(self.setup_state_path.read_text(encoding="utf-8"))
            except Exception:
                return
            for p in (rec.get("placements") or ()):
                if p.get("model_id") == cid and str(p.get("endpoint_url") or "").rstrip("/") == ep_norm:
                    p["status"] = "removed_by_user"
                    p["interactive"] = False
            rec["ready_models"] = [
                p["model_id"]
                for p in (rec.get("placements") or ())
                if str(p.get("status") or "").startswith("ready_")
            ]
            tmp = self.setup_state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            os.replace(tmp, self.setup_state_path)

    def mcp_manifest(self) -> dict[str, Any]:
        saved = self.load_saved_setup()
        return {
            "mcp_version": "2.0",
            "system": "clover-v6-4part",
            "parts": ["AGENTS.md", "mcp.json", "<task_id>.md", "todos.sqlite3"],
            "computed_0b_tools": [
                {
                    "name": "xcruncher_exact_math",
                    "params": ["expression"],
                    "kind": "0b_computed",
                    "description": "Exact big-integer multiplication, binary exponentiation (A^B), factorial (N!), and BBP hexadecimal Pi.",
                },
                {
                    "name": "forecast_series_0b",
                    "params": ["values", "horizon"],
                    "kind": "0b_computed",
                    "description": "0B Periodogram + Harmonic OLS + Split-Conformal P10/P50/P90 time-series forecaster (<50 ms CPU).",
                },
                {
                    "name": "resolve_defeasible_rules_0b",
                    "params": ["rules", "context_facts"],
                    "kind": "0b_computed",
                    "description": "0B Defeasible epistemic rule resolver (filters rejected rules, honors unless exceptions & specificity).",
                },
                {
                    "name": "python_ast_verify",
                    "params": ["code"],
                    "kind": "0b_computed",
                    "description": "Deterministic Python AST syntax & structure verifier.",
                },
            ],
            "specialist_routes": setup_mcp_routes(saved),
        }

    def describe(self) -> dict[str, Any]:
        ms = self.metrics.summary()
        saved = self.load_saved_setup()
        return {
            "version": "6.0.0",
            "architecture": "4-part-steerable-clover",
            "parts": {
                "agents_md": self.agents_md_path.is_file(),
                "mcp_tools_count": len(DEFAULT_REGISTERED_TOOLS),
                "context_dir": str(self.contexts_dir),
                "todo_store": "sqlite-prerequisite-gated",
            },
            "active_setup_summary": saved.get("summary"),
            "ready_models": saved.get("ready_models") or ["0b-computed-pack", "omni"],
            "used_vs_unused": ms["used_vs_unused"],
            "solver_cycles": ms["solver_cycles"],
        }

    def write_task_context(self, task_id: str, prompt: str, direction: str, outcome: str, tools_used: list[str]) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_\-]", "_", str(task_id or "task"))[:80]
        p = self.contexts_dir / f"{safe_id}.md"
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        body = (
            f"# Clover v6.0 Task Context — `{safe_id}`\n"
            f"- Updated: `{ts}`\n"
            f"- Direction: `{direction}`\n"
            f"- Tools Used: `{', '.join(tools_used) if tools_used else 'none'}`\n\n"
            f"## Context\n{prompt.strip()}\n\n"
            f"## Verified Outcome\n{outcome.strip()[:4000]}\n"
        )
        p.write_text(body, encoding="utf-8")
        return p

    def call_mcp_tool(self, tool_name: str, args: dict[str, Any] | None = None, *, task_id: str = "mcp", key_lookup=None) -> dict[str, Any]:
        a = args or {}
        t0 = time.perf_counter()
        ok = True
        inst = "0b-local"
        try:
            if tool_name in ("xcruncher_exact_math", "calculate"):
                import local_tool_packs
                res: Any = local_tool_packs.calculate(str(a.get("expression") or a.get("expr") or ""))
            elif tool_name in ("forecast_series_0b", "forecast_series"):
                res = forecast_series_0b(a.get("values") or [], int(a.get("horizon") or 12))
            elif tool_name in ("resolve_defeasible_rules_0b", "resolve_rules"):
                res = resolve_defeasible_rules_0b(a.get("rules") or [], a.get("context_facts") or [])
            elif tool_name == "python_ast_verify":
                code = str(a.get("code") or "")
                tree = ast.parse(code)
                funcs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                res = {"valid_syntax": True, "functions": funcs}
            else:
                routes = setup_mcp_routes(self.load_saved_setup())
                rt = routes.get(tool_name)
                if rt is None and tool_name == "chronos2_forecast":
                    # Seamless local fallback to 0B Periodogram+Conformal forecaster when no remote Chronos server is configured
                    res = dict(forecast_series_0b(a.get("values") or a.get("series") or [], int(a.get("horizon") or 12)), fallback_from="chronos-2")
                else:
                    target_url = (rt["url"] if rt else os.environ.get("CLOVER_HG4_GATEWAY_URL", "http://127.0.0.1:8790")).rstrip("/")
                    inst = urllib.parse.urlparse(target_url).hostname or target_url
                    hdrs = {"Content-Type": "application/json"}
                    if rt and rt.get("key_id") and callable(key_lookup):
                        kv = key_lookup(rt["key_id"])
                        if kv:
                            hdrs["Authorization"] = f"Bearer {kv}"
                            hdrs["X-Clover-Key"] = kv
                    req = urllib.request.Request(
                        f"{target_url}/mcp",
                        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool_name, "arguments": a}}).encode("utf-8"),
                        headers=hdrs,
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
            return {"ok": True, "tool": tool_name, "instance": inst, "result": res}
        except Exception as exc:
            ok = False
            return {"ok": False, "tool": tool_name, "instance": inst, "error": str(exc)}
        finally:
            self.metrics.record_tool_call(
                task_id=task_id,
                tool_name=tool_name,
                instance=inst,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                ok=ok,
            )

    def solve_4part(self, prompt: str, *, model: str = "omni", instance: str = "local", task_id: str | None = None, generate_fn=None, max_cycles: int = 2) -> dict[str, Any]:
        tid = re.sub(r"[^A-Za-z0-9_\-]", "_", str(task_id or f"t_{uuid.uuid4().hex[:10]}"))[:80]
        q = str(prompt or "").strip()
        t_start = time.perf_counter()
        tools_used: list[str] = []

        self.todos.add_todo(tid, f"{tid}_ctx", "Capturing context & epistemic boundaries", q[:200])
        self.todos.add_todo(tid, f"{tid}_exec", "Executing smallest-capable computed tool or model", "Run deterministic 0B tool or model", depends_on=[f"{tid}_ctx"])
        self.todos.set_status(f"{tid}_ctx", "done")

        if _AMBIGUOUS_RISK_RE.search(q):
            self.todos.set_status(f"{tid}_exec", "blocked")
            msg = (
                "STOP_AND_ASK: This action carries irreversible or underspecified infrastructure risk. "
                "Please confirm: (1) target environment/database, (2) backup/rollback plan, and (3) exact scope."
            )
            ctx_path = self.write_task_context(tid, q, "STOP_AND_ASK_BOUNDARY", msg, tools_used)
            self.metrics.record_problem_run(
                task_id=tid,
                model=model,
                status="STOPPED_FOR_HUMAN",
                domain="steering",
                cycles_used=1,
                tools_used=tools_used,
                total_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
            )
            return {
                "task_id": tid,
                "status": "STOPPED_FOR_HUMAN",
                "cycles_used": 1,
                "model": canon_model_id(model),
                "tools_used": tools_used,
                "answer": msg,
                "context_file": str(ctx_path),
                "todos": self.todos.all_todos(tid),
            }

        is_single_func_code = bool(_SINGLE_FUNC_CODE_RE.match(q))
        computed_context = ""
        if not is_single_func_code:
            m_math = _EXACT_MATH_RE.search(q)
            if m_math:
                expr = m_math.group(0)
                tool_out = self.call_mcp_tool("xcruncher_exact_math", {"expression": expr}, task_id=tid)
                if tool_out.get("ok"):
                    tools_used.append("xcruncher_exact_math")
                    computed_context = f"[Verified x-cruncher exact result for `{expr}`]: {tool_out['result']}"

        self.todos.set_status(f"{tid}_exec", "in_progress")
        cycles_used = 1
        answer = computed_context
        if callable(generate_fn):
            aug_prompt = f"{q}\n\n{computed_context}".strip() if computed_context else q
            for cyc in range(1, max(1, int(max_cycles)) + 1):
                cycles_used = cyc
                t_m = time.perf_counter()
                ans_raw = str(generate_fn(aug_prompt, cyc) or "").strip()
                el_ms = round((time.perf_counter() - t_m) * 1000.0, 2)
                tok_est = max(1, len(ans_raw.split()))
                self.metrics.record_model_call(
                    task_id=tid,
                    model=model,
                    instance=instance,
                    cycle=cyc,
                    prompt_tokens=max(1, len(aug_prompt.split())),
                    completion_tokens=tok_est,
                    elapsed_ms=el_ms,
                    ok=bool(ans_raw),
                )
                answer = ans_raw
                if is_single_func_code and "def " in ans_raw:
                    m_code = re.search(r"```(?:python)?\s*(.*?)```", ans_raw, flags=re.S)
                    code_snip = m_code.group(1) if m_code else ans_raw
                    v_out = self.call_mcp_tool("python_ast_verify", {"code": code_snip}, task_id=tid)
                    tools_used.append("python_ast_verify")
                    if not v_out.get("ok") and cyc < max_cycles:
                        aug_prompt = f"{q}\n\nFix Python syntax error from Cycle {cyc}: {v_out.get('error')}"
                        continue
                break

        self.todos.set_status(f"{tid}_exec", "done")
        ctx_path = self.write_task_context(tid, q, f"4PART_{canon_model_id(model).upper()}", answer or "OK", tools_used)
        self.metrics.record_problem_run(
            task_id=tid,
            model=model,
            status="SOLVED",
            domain="code" if is_single_func_code else ("math" if tools_used else "general"),
            cycles_used=cycles_used,
            tools_used=tools_used,
            total_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
        )
        return {
            "task_id": tid,
            "status": "SOLVED",
            "cycles_used": cycles_used,
            "model": canon_model_id(model),
            "tools_used": tools_used,
            "answer": answer,
            "context_file": str(ctx_path),
            "todos": self.todos.all_todos(tid),
        }


_INSTANCES: dict[str, CloverStudioV6] = {}
_INST_LOCK = threading.Lock()


def for_base(base_dir: str | Path) -> CloverStudioV6:
    key = str(Path(base_dir).resolve())
    with _INST_LOCK:
        inst = _INSTANCES.get(key)
        if inst is None:
            inst = CloverStudioV6(key)
            _INSTANCES[key] = inst
        return inst
