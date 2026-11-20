"""Quality Dashboard & Structured Report Renderer for Clover Studio Quality Engine."""
from __future__ import annotations

import math
from typing import Any


def _percentile(sorted_vals: list[float], pct: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(math.ceil((pct / 100.0) * len(sorted_vals))) - 1))
    return round(sorted_vals[idx], 1)


def compute_metrics(checks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "NOT_RUN": 0}
    by_area: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    by_evidence: dict[str, dict[str, int]] = {}
    latencies: list[float] = []
    first_turn_ok = 0
    first_turn_tot = 0
    tool_ok = 0
    tool_tot = 0

    for c in checks:
        st = str(c.get("status") or "SKIP").upper()
        if st not in counts:
            counts[st] = 0
        counts[st] += 1

        area = str(c.get("area") or (c["name"].split(":", 1)[0] if ":" in c.get("name", "") else "general"))
        by_area.setdefault(area, {"PASS": 0, "FAIL": 0, "NOT_RUN": 0, "SKIP": 0})
        by_area[area][st] = by_area[area].get(st, 0) + 1

        mdl = str(c.get("model") or "n/a")
        if mdl != "n/a":
            by_model.setdefault(mdl, {"PASS": 0, "FAIL": 0, "NOT_RUN": 0, "SKIP": 0})
            by_model[mdl][st] = by_model[mdl].get(st, 0) + 1

        ev = str(c.get("evidence_class") or "unspecified")
        by_evidence.setdefault(ev, {"PASS": 0, "FAIL": 0, "NOT_RUN": 0, "SKIP": 0})
        by_evidence[ev][st] = by_evidence[ev].get(st, 0) + 1

        dur = c.get("duration_ms")
        if isinstance(dur, (int, float)) and st in ("PASS", "FAIL"):
            latencies.append(float(dur))

        if ev in ("live_model_e2e", "live_tool_e2e", "live_media_e2e") and st in ("PASS", "FAIL"):
            first_turn_tot += 1
            if st == "PASS" and int(c.get("retries_used") or 0) == 0:
                first_turn_ok += 1
        if ev == "live_tool_e2e" and st in ("PASS", "FAIL"):
            tool_tot += 1
            if st == "PASS":
                tool_ok += 1

    latencies.sort()
    executed = counts["PASS"] + counts["FAIL"]
    return {
        "counts": counts,
        "pass_rate_pct": round(100.0 * counts["PASS"] / executed, 1) if executed else None,
        "first_turn_success_pct": round(100.0 * first_turn_ok / first_turn_tot, 1) if first_turn_tot else None,
        "tool_execution_success_pct": round(100.0 * tool_ok / tool_tot, 1) if tool_tot else None,
        "latency_ms": {
            "count": len(latencies),
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": round(latencies[-1], 1) if latencies else None,
        },
        "by_area": by_area,
        "by_model": by_model,
        "by_evidence_class": by_evidence,
    }


def render_markdown_dashboard(report: dict[str, Any]) -> str:
    m = report.get("dashboard") or compute_metrics(report.get("checks") or [])
    counts = m["counts"]
    lines = [
        "# Clover Studio Quality Dashboard",
        "",
        f"- **Generated (UTC):** `{report.get('generated_at', '')}`",
        f"- **Overall Gate Verdict:** **`{report.get('overall', 'UNKNOWN')}`**",
        f"- **Executed Pass Rate:** `{m['pass_rate_pct']}%` (`{counts['PASS']}` PASS / `{counts['FAIL']}` FAIL / `{counts['NOT_RUN']}` NOT_RUN / `{counts['SKIP']}` SKIP)",
        f"- **First-Turn E2E Success:** `{m['first_turn_success_pct'] if m['first_turn_success_pct'] is not None else 'n/a'}%`",
        f"- **Tool Execution Success:** `{m['tool_execution_success_pct'] if m['tool_execution_success_pct'] is not None else 'n/a'}%`",
        f"- **Latency (p50 / p95 / max):** `{m['latency_ms']['p50']} ms` / `{m['latency_ms']['p95']} ms` / `{m['latency_ms']['max']} ms`",
        "",
        "## 1. Evidence-Class Breakdown",
        "",
        "| Evidence Class | PASS | FAIL | NOT_RUN | SKIP |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for ev, st_map in sorted(m["by_evidence_class"].items()):
        lines.append(f"| `{ev}` | {st_map.get('PASS', 0)} | {st_map.get('FAIL', 0)} | {st_map.get('NOT_RUN', 0)} | {st_map.get('SKIP', 0)} |")

    lines += [
        "",
        "## 2. Results by Feature Area",
        "",
        "| Feature Area | PASS | FAIL | NOT_RUN | SKIP |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for area, st_map in sorted(m["by_area"].items()):
        lines.append(f"| `{area}` | {st_map.get('PASS', 0)} | {st_map.get('FAIL', 0)} | {st_map.get('NOT_RUN', 0)} | {st_map.get('SKIP', 0)} |")

    if m["by_model"]:
        lines += [
            "",
            "## 3. Results by Model / Configuration",
            "",
            "| Model | PASS | FAIL | NOT_RUN | SKIP |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
        for mdl, st_map in sorted(m["by_model"].items()):
            lines.append(f"| `{mdl}` | {st_map.get('PASS', 0)} | {st_map.get('FAIL', 0)} | {st_map.get('NOT_RUN', 0)} | {st_map.get('SKIP', 0)} |")

    failures = [c for c in (report.get("checks") or []) if c.get("status") == "FAIL"]
    lines += ["", "## 4. Automatic Failure Diagnosis"]
    if not failures:
        lines.append("\n_Zero failures recorded in executed checks._")
    else:
        for f in failures:
            diag = f.get("diagnosis") or {}
            lines += [
                "",
                f"### `{f.get('name')}` (`{f.get('evidence_class')}`)",
                f"- **Failing Layer:** `{diag.get('failure_layer', 'unknown')}`",
                f"- **Expected:** {diag.get('expected', 'PASS')}",
                f"- **Observed:** {diag.get('observed', f.get('reason', 'failed'))}",
                f"- **Proposed Repair:** {diag.get('proposed_repair', 'Inspect trace.')}",
            ]

    not_run = [c for c in (report.get("checks") or []) if c.get("status") == "NOT_RUN"]
    if not_run:
        lines += ["", "## 5. Honest `NOT_RUN` Boundaries (Unexercised Targets)"]
        for nr in not_run:
            lines.append(f"- **`{nr.get('name')}`** (`{nr.get('model') or 'environment'}`): {nr.get('reason', 'target unavailable')}")

    return "\n".join(lines) + "\n"
