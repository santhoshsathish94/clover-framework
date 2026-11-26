#!/usr/bin/env python3
"""Live Benchmark Proof for Myth #6: `AI Alone (Unsteered Autopilot)` vs `Human Steering + AI` (`run_myth6_steering_bench.py`).

Tests 8 real software/scientific tasks where reaching the true destination depends on a domain
invariant owned by the human expert (regulatory convention, wire protocol endianness, custom
calendar epoch, domain unit scaling, etc.):
  * Arm 1 (`unsteered_autopilot_ai_alone`): Model is given full retry access (`max_turns=5`) with no
    human steering when verification fails (`env invariant missing from prompt`) -> measures blind
    retry cycles, `STOP_AND_ASK` halts, tokens burned, and verified pass rate.
  * Arm 2 (`clover_human_steered_1_turn`): Clover detects stagnation / ambiguity (`STOP_AND_ASK`),
    receives a single 1-sentence human steering turn (`Direction`), and autonomously completes the
    verified sub-goals -> measures total cycles, decode `tok/s`, tokens, and verified pass rate.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


MYTH6_TASKS: list[dict[str, str]] = [
    {
        "task_id": "m6_wire_header_crc",
        "ambiguous_prompt": "Write `pack_pkt(seq_id: int, payload: bytes) -> bytes` for our internal telemetry bus.",
        "human_steering": "Use big-endian `<magic=0xC107 uint16><seq_id uint32><len uint16>` followed by payload.",
        "test_code": (
            "assert pack_pkt(7, b'AB') == b'\\xc1\\x07\\x00\\x00\\x00\\x07\\x00\\x02AB'\n"
            "assert pack_pkt(0, b'') == b'\\xc1\\x07\\x00\\x00\\x00\\x00\\x00\\x00'"
        ),
    },
    {
        "task_id": "m6_fiscal_quarter",
        "ambiguous_prompt": "Write `fiscal_quarter(month: int) -> int` returning the company's fiscal quarter `1..4` for calendar `month` `1..12`.",
        "human_steering": "Our fiscal year starts in April (April-June = Q1, July-Sept = Q2, Oct-Dec = Q3, Jan-March = Q4).",
        "test_code": (
            "assert [fiscal_quarter(m) for m in (4, 6, 7, 10, 1, 3)] == [1, 1, 2, 3, 4, 4]"
        ),
    },
    {
        "task_id": "m6_invoice_rounding",
        "ambiguous_prompt": "Write `settle_inr_paise(amount_inr: float, tax_pct: float) -> int` returning total integer paise after tax.",
        "human_steering": "Apply tax first in exact integer paise (`round(amount_inr * 100)`) and round half-up (`int(x + 0.5)`) to the nearest 100 paise (whole rupee).",
        "test_code": (
            "assert settle_inr_paise(10.00, 18.0) == 1200\n"
            "assert settle_inr_paise(12.50, 18.0) == 1500"
        ),
    },
    {
        "task_id": "m6_sensor_kelvin_clamp",
        "ambiguous_prompt": "Write `normalize_temp(raw_adc: int) -> float` converting 12-bit cryo sensor `raw_adc` (`0..4095`) to Kelvin.",
        "human_steering": "Sensor maps linearly `0 -> 4.0 K` and `4095 -> 304.0 K`, rounded to 2 decimals, returning `-1.0` if `raw_adc` is outside `0..4095`.",
        "test_code": (
            "assert normalize_temp(0) == 4.0\n"
            "assert normalize_temp(4095) == 304.0\n"
            "assert normalize_temp(-5) == -1.0\n"
            "assert normalize_temp(5000) == -1.0"
        ),
    },
    {
        "task_id": "m6_rate_bucket_tier",
        "ambiguous_prompt": "Write `quota_rpm(plan: str) -> int` returning allowed requests/minute for our gateway tier.",
        "human_steering": "Tiers are `'seed': 30`, `'petal': 300`, `'grove': 3000`, case-insensitive, and any unknown tier returns `0`.",
        "test_code": (
            "assert quota_rpm('seed') == 30\n"
            "assert quota_rpm('PETAL') == 300\n"
            "assert quota_rpm('grove') == 3000\n"
            "assert quota_rpm('other') == 0"
        ),
    },
    {
        "task_id": "m6_ring_hash_slot",
        "ambiguous_prompt": "Write `shard_for_layer(layer_idx: int) -> int` mapping a 93-layer model (`0..92`) onto shard `0..3`.",
        "human_steering": "Shard 0 holds `0..21`, Shard 1 holds `22..45`, Shard 2 holds `46..69`, and Shard 3 holds `70..92`.",
        "test_code": (
            "assert [shard_for_layer(i) for i in (0, 21, 22, 45, 46, 69, 70, 92)] == [0, 0, 1, 1, 2, 2, 3, 3]"
        ),
    },
]


def _http_json(url: str, payload: dict[str, Any], api_key: str) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hub-url", default="http://127.0.0.1:8790")
    ap.add_argument("--api-key", default=os.environ.get("CLOVER_GATEWAY_API_KEY", ""))
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    # Redact concrete expected literals in verifier error feedback for Arm 1 so an unsteered agent cannot "cheat"
    # by reading the private acceptance assertion literals — simulating real production roadblocks!
    unsteered_rows = []
    steered_rows = []

    for t in MYTH6_TASKS:
        opaque_test = (
            f"def _run_hidden_spec():\n"
            f"    ok = True\n"
            + "\n".join(f"    ok = ok and ({ln.replace('assert ', '')})" for ln in t["test_code"].splitlines())
            + "\n    assert ok, 'Domain specification mismatch (missing stakeholder constraint)'\n_run_hidden_spec()"
        )
        # Arm 1: AI Alone (Unsteered Autopilot)
        r1 = _http_json(
            f"{args.hub_url.rstrip('/')}/v1/clover/solve",
            {
                "task_id": f"unsteered_{t['task_id']}",
                "benchmark_tag": "myth6_unsteered_ai_alone",
                "model": "clover-35b-a3b",
                "prompt": t["ambiguous_prompt"] + " Return Python code in a ```python``` block.",
                "test_code": opaque_test,
                "tools": [],
                "max_turns": 5,
                "max_tokens": 320,
                "temperature": 0.0,
            },
            args.api_key,
        )
        p1 = bool((r1.get("verified_outcome") or {}).get("passed"))
        unsteered_rows.append({
            "task_id": t["task_id"],
            "passed": p1,
            "discernment_state": r1.get("discernment_state"),
            "turns_used": r1.get("turns_used"),
            "cycles_tried": r1.get("cycles_tried"),
            "completion_tokens": r1["usage"]["completion_tokens"],
            "decode_tok_per_s": r1.get("mean_decode_tok_per_s"),
        })

        # Arm 2: Human Steering + AI (1 sentence of human Direction at the wheel!)
        r2 = _http_json(
            f"{args.hub_url.rstrip('/')}/v1/clover/solve",
            {
                "task_id": f"steered_{t['task_id']}",
                "benchmark_tag": "myth6_human_steered",
                "model": "clover-35b-a3b",
                "prompt": f"{t['ambiguous_prompt']}\n[Human Direction at Steering Wheel]: {t['human_steering']}\nReturn Python code in a ```python``` block.",
                "test_code": opaque_test,
                "tools": ["clover_todo_upsert"],
                "max_turns": 4,
                "max_tokens": 320,
                "temperature": 0.0,
            },
            args.api_key,
        )
        p2 = bool((r2.get("verified_outcome") or {}).get("passed"))
        steered_rows.append({
            "task_id": t["task_id"],
            "passed": p2,
            "discernment_state": r2.get("discernment_state"),
            "turns_used": r2.get("turns_used"),
            "cycles_tried": r2.get("cycles_tried"),
            "completion_tokens": r2["usage"]["completion_tokens"],
            "decode_tok_per_s": r2.get("mean_decode_tok_per_s"),
        })
        print(f"[Myth6] {t['task_id']}: AI_Alone={p1} (state={r1.get('discernment_state')}, tok={r1['usage']['completion_tokens']}) | Human+AI={p2} (cyc={r2.get('cycles_tried')}, tok={r2['usage']['completion_tokens']})", flush=True)

    u_pass = sum(1 for x in unsteered_rows if x["passed"])
    s_pass = sum(1 for x in steered_rows if x["passed"])
    u_stop = sum(1 for x in unsteered_rows if x["discernment_state"] == "STOP_AND_ASK")
    out = {
        "benchmark": "myth6_human_steering_vs_ai_alone",
        "tasks_total": len(MYTH6_TASKS),
        "unsteered_ai_alone": {
            "passed": u_pass,
            "pass_rate_pct": round(100.0 * u_pass / len(MYTH6_TASKS), 2),
            "clover_stop_and_ask_triggered": u_stop,
            "mean_completion_tokens": round(statistics.fmean(x["completion_tokens"] for x in unsteered_rows), 1),
            "mean_decode_tok_per_s": round(statistics.fmean(x["decode_tok_per_s"] for x in unsteered_rows), 2),
            "per_task": unsteered_rows,
        },
        "human_steered_plus_ai": {
            "passed": s_pass,
            "pass_rate_pct": round(100.0 * s_pass / len(MYTH6_TASKS), 2),
            "mean_cycles_per_task": round(statistics.fmean(x["cycles_tried"] for x in steered_rows), 2),
            "mean_completion_tokens": round(statistics.fmean(x["completion_tokens"] for x in steered_rows), 1),
            "mean_decode_tok_per_s": round(statistics.fmean(x["decode_tok_per_s"] for x in steered_rows), 2),
            "per_task": steered_rows,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Myth6 DONE] AI_Alone={u_pass}/{len(MYTH6_TASKS)} ({out['unsteered_ai_alone']['pass_rate_pct']}%, STOP_AND_ASK={u_stop}) vs Human+AI={s_pass}/{len(MYTH6_TASKS)} ({out['human_steered_plus_ai']['pass_rate_pct']}%)", flush=True)


if __name__ == "__main__":
    main()
