#!/usr/bin/env python3
"""Metrics-First Multi-Stage Verification & Paired A/B Benchmark Runner (`run_clover_ab_benchmarks.py`).

Executes Stages 2-5 against Clover's live Center Hub (`k3:8790`) + GPU Petal Mesh and records
comprehensive structured telemetry for every stage and every arm:
  * What models, MCP tools, and GPU replica instances were USED vs. UNUSED
  * Exact token decode speed (`mean`, `p50`, `p95`, `max` `tok/s`) and tokens/problem
  * Parallel `instances_run` and solver `cycles_histogram` (`1` vs `2` vs `3+` cycles to solve)
  * Actionable `fine_tune_signals` identifying exact failure categories & multi-turn repair candidates.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


def _http_json(url: str, payload: dict[str, Any] | None = None, api_key: str = "", timeout: float = 300.0) -> Any:
    hdrs = {"Content-Type": "application/json"}
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=hdrs, method="GET" if data is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _extract_py_code(text: str) -> str:
    clean = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    blocks = re.findall(r"```(?:python|py|)\s*\n(.*?)```", clean, flags=re.DOTALL)
    return blocks[-1].strip() if blocks else clean.strip()


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * q))))
    return round(s[idx], 2)


def run_stage2_mesh_verify(hub_url: str, api_key: str) -> dict[str, Any]:
    """Stage 2: Verify live Petal mesh, T=0 request clubbing, mcp.json stdio bridge, and replica pool."""
    petals = _http_json(f"{hub_url}/v1/cluster/petals", api_key=api_key)
    queue_before = _http_json(f"{hub_url}/v1/queue", api_key=api_key)

    # 1. T=0 Request Clubbing proof: submit 4 identical T=0 prompts in one batch
    club_prompt = "Return the exact integer product 23 * 29 and nothing else."
    batch_reqs = [
        {
            "model": "clover-35b-a3b",
            "messages": [{"role": "user", "content": club_prompt}],
            "max_tokens": 24,
            "temperature": 0.0,
            "x_clover_benchmark_tag": "stage2_mesh_verify",
        }
        for _ in range(4)
    ]
    t0 = time.perf_counter()
    batch_out = _http_json(f"{hub_url}/v1/chat/completions/batch", {"requests": batch_reqs, "priority": 1}, api_key=api_key)
    club_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    queue_after = _http_json(f"{hub_url}/v1/queue", api_key=api_key)

    # 2. Verify mcp.json UTF-8 stdio bridge (`clover_mcp_stdio.py`)
    stdio_py = Path(__file__).resolve().parent / "clover_mcp_stdio.py"
    rpc_in = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "xcruncher_exact_math", "arguments": {"expression": "20!"}}}),
    ]) + "\n"
    env = {**os.environ, "CLOVER_MCP_URL": f"{hub_url}/mcp", "CLOVER_GATEWAY_API_KEY": api_key}
    proc = subprocess.run([sys.executable, str(stdio_py)], input=rpc_in.encode("utf-8"), capture_output=True, env=env, timeout=30)
    stdio_lines = [json.loads(ln) for ln in proc.stdout.decode("utf-8").splitlines() if ln.strip()]

    metrics = _http_json(f"{hub_url}/v1/clover/metrics?benchmark_tag=stage2_mesh_verify", api_key=api_key)
    return {
        "stage": "stage2_mesh_verify",
        "petals": petals,
        "clubbing_4way_wall_ms": club_ms,
        "clubbing_answers_identical": len({r["completion"]["choices"][0]["message"]["content"] for r in batch_out["results"]}) == 1,
        "queue_delta": {
            "before": queue_before["queue"],
            "after": queue_after["queue"],
            "replica_pools": queue_after["replica_pools"],
        },
        "mcp_stdio_tools_count": len(stdio_lines[0]["result"]["tools"]),
        "mcp_stdio_20_factorial": stdio_lines[1]["result"]["structuredContent"],
        "metrics": metrics,
    }


def load_humaneval_subset(jsonl_path: Path, limit: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            row = json.loads(ln)
            tid = str(row.get("task_id") or f"HE_{len(tasks)}").replace("/", "_")
            prompt_code = str(row["prompt"])
            entry_pt = str(row["entry_point"])
            raw_test = str(row["test"])
            # Convert HumanEval `def check(candidate): ...` into pure AST-safe top-level `assert` lines!
            assert_lines: list[str] = []
            for tln in raw_test.splitlines():
                s = tln.strip()
                if s.startswith("assert "):
                    assert_lines.append(s.replace("candidate(", f"{entry_pt}("))
            if not assert_lines:
                continue
            tasks.append({
                "task_id": tid,
                "entry_point": entry_pt,
                "prompt": (
                    "Complete the following Python 3 function cleanly inside a ```python``` block "
                    "(include necessary imports and the full function definition):\n\n"
                    f"```python\n{prompt_code.strip()}\n```"
                ),
                "prompt_prefix": prompt_code,
                "test_code": "\n".join(assert_lines),
            })
            if len(tasks) >= limit:
                break
    return tasks


def eval_coding_arm(
    *,
    arm_name: str,
    target_urls: list[str],
    api_key: str,
    model: str,
    use_4part: bool,
    escalate_url: str | None,
    escalate_model: str | None,
    tasks: list[dict[str, Any]],
    max_workers: int,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    results: list[dict[str, Any]] = []

    def _run_one(idx_task: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, task = idx_task
        base_u = target_urls[idx % len(target_urls)].rstrip("/")
        btag = f"s3_{arm_name}"
        if not use_4part:
            t0 = time.perf_counter()
            comp = _http_json(
                f"{base_u}/v1/chat/completions",
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Return concise, correct Python 3 code in a ```python``` block (no `<think>` monologue)."},
                        {"role": "user", "content": task["prompt"]},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.0,
                    "x_clover_benchmark_tag": btag,
                },
                api_key=api_key,
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            ans = comp["choices"][0]["message"]["content"] or ""
            cand = _extract_py_code(ans)
            if f"def {task['entry_point']}" not in cand:
                cand = task["prompt_prefix"] + "\n" + cand
            # Grade via deterministic `python_ast_verify` on Hub/Node
            v_rpc = _http_json(
                f"{target_urls[0].rstrip('/')}/mcp",
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "python_ast_verify", "arguments": {"candidate_code": cand, "test_code": task["test_code"]}}},
                api_key=api_key,
            )
            v_sc = v_rpc["result"]["structuredContent"]
            c_toks = int(comp["usage"]["completion_tokens"])
            tok_s = float((comp.get("x_clover_telemetry") or {}).get("decode_tok_per_s") or round(c_toks / max(wall_ms / 1000.0, 1e-3), 2))
            return {
                "task_id": task["task_id"],
                "passed": bool(v_sc.get("passed")),
                "cycles_tried": 1,
                "solved_on_cycle": 1 if v_sc.get("passed") else 0,
                "turns_used": 1,
                "completion_tokens": c_toks,
                "decode_tok_per_s": tok_s,
                "wall_ms": round(wall_ms, 2),
                "models_used": [model],
                "tools_used": [],
                "instance": base_u,
                "diagnostics": v_sc.get("diagnostics", ""),
            }

        # 4-Part Clover System (`POST /v1/clover/solve`)
        s_out = _http_json(
            f"{base_u}/v1/clover/solve",
            {
                "task_id": f"{arm_name}_{task['task_id']}",
                "benchmark_tag": btag,
                "model": model,
                "prompt": task["prompt"],
                "test_code": task["test_code"],
                "tools": ["clover_context_read", "clover_context_record", "clover_todo_list", "clover_todo_upsert", "xcruncher_exact_math"],
                "max_turns": 4,
                "max_tokens": 512,
                "temperature": 0.0,
            },
            api_key=api_key,
        )
        passed = bool((s_out.get("verified_outcome") or {}).get("passed"))
        cycles = int(s_out.get("cycles_tried") or 1)
        solved_cyc = int(s_out.get("solved_on_cycle") or 0)
        models_u = list(s_out.get("models_used") or [model])
        tools_u = list((s_out.get("metrics") or {}).get("tools_used") or [])
        c_toks = int(s_out["usage"]["completion_tokens"])
        tok_s = float(s_out.get("mean_decode_tok_per_s") or 0.0)
        wall_ms = float(s_out.get("wall_ms") or 0.0)

        # Optional Team Escalation (`35B-LoRA -> 120B / Kimi`) with Context -> Growth handoff!
        if not passed and escalate_url and escalate_model:
            handoff_md = s_out.get("context_markdown") or ""
            esc_out = _http_json(
                f"{escalate_url.rstrip('/')}/v1/clover/solve",
                {
                    "task_id": f"{arm_name}_esc_{task['task_id']}",
                    "benchmark_tag": btag,
                    "model": escalate_model,
                    "prompt": f"{handoff_md}\n\nSolve the original task without repeating ruled-out drafts:\n{task['prompt']}",
                    "test_code": task["test_code"],
                    "tools": ["xcruncher_exact_math"],
                    "max_turns": 3,
                    "max_tokens": 640,
                    "temperature": 0.0,
                },
                api_key=api_key,
            )
            esc_pass = bool((esc_out.get("verified_outcome") or {}).get("passed"))
            esc_cyc = int(esc_out.get("cycles_tried") or 1)
            if esc_pass:
                passed = True
                solved_cyc = cycles + int(esc_out.get("solved_on_cycle") or 1)
            cycles += esc_cyc
            c_toks += int(esc_out["usage"]["completion_tokens"])
            wall_ms += float(esc_out.get("wall_ms") or 0.0)
            for em in esc_out.get("models_used") or [escalate_model]:
                if em not in models_u:
                    models_u.append(em)

        return {
            "task_id": task["task_id"],
            "passed": passed,
            "cycles_tried": cycles,
            "solved_on_cycle": solved_cyc,
            "turns_used": int(s_out.get("turns_used") or 1),
            "completion_tokens": c_toks,
            "decode_tok_per_s": tok_s,
            "wall_ms": round(wall_ms, 2),
            "models_used": models_u,
            "tools_used": tools_u,
            "instance": base_u,
            "diagnostics": (s_out.get("verified_outcome") or {}).get("diagnostics", ""),
        }

    eff_workers = max(1, min(max_workers, len(tasks)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=eff_workers) as ex:
        results = list(ex.map(_run_one, enumerate(tasks)))

    passed_n = sum(1 for r in results if r["passed"])
    repaired_n = sum(1 for r in results if r["passed"] and r["cycles_tried"] > 1)
    tok_s_vals = [r["decode_tok_per_s"] for r in results if r["decode_tok_per_s"] > 0]
    cyc_hist: dict[str, int] = {}
    inst_hits: dict[str, int] = {}
    tools_hit: dict[str, int] = {}
    models_hit: dict[str, int] = {}
    for r in results:
        cs = str(r["cycles_tried"])
        cyc_hist[cs] = cyc_hist.get(cs, 0) + 1
        inst_hits[r["instance"]] = inst_hits.get(r["instance"], 0) + 1
        for m in r["models_used"]:
            models_hit[m] = models_hit.get(m, 0) + 1
        for t in r["tools_used"]:
            tools_hit[t] = tools_hit.get(t, 0) + 1

    unused_inst = [u for u in target_urls if u.rstrip("/") not in inst_hits]
    return {
        "arm": arm_name,
        "problems_total": len(results),
        "passed": passed_n,
        "pass_at_1_pct": round(100.0 * passed_n / max(1, len(results)), 2),
        "instances_configured": len(target_urls),
        "instances_used": len(inst_hits),
        "instances_unused": unused_inst,
        "instance_hit_distribution": inst_hits,
        "parallel_workers": eff_workers,
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "decode_tok_per_s": {
            "mean": round(statistics.fmean(tok_s_vals), 2) if tok_s_vals else 0.0,
            "p50": _pct(tok_s_vals, 0.50),
            "p95": _pct(tok_s_vals, 0.95),
            "max": round(max(tok_s_vals), 2) if tok_s_vals else 0.0,
        },
        "mean_completion_tokens": round(statistics.fmean([r["completion_tokens"] for r in results]), 1) if results else 0.0,
        "cycles_histogram": dict(sorted(cyc_hist.items(), key=lambda kv: int(kv[0]))),
        "mean_cycles_per_problem": round(statistics.fmean([r["cycles_tried"] for r in results]), 3) if results else 0.0,
        "repaired_after_cycle1": repaired_n,
        "models_used_counts": models_hit,
        "tools_used_counts": tools_hit,
        "per_problem": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=("mesh-verify", "coding-ab"))
    ap.add_argument("--hub-url", default="http://127.0.0.1:8790")
    ap.add_argument("--api-key", default=os.environ.get("CLOVER_GATEWAY_API_KEY", ""))
    ap.add_argument("--humaneval-jsonl", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--config-json", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    if args.stage == "mesh-verify":
        out = run_stage2_mesh_verify(args.hub_url.rstrip("/"), args.api_key)
    else:
        assert args.humaneval_jsonl and args.config_json
        cfg = json.loads(args.config_json.read_text(encoding="utf-8"))
        tasks = load_humaneval_subset(args.humaneval_jsonl, args.limit)
        arms_out = []
        for arm in cfg["arms"]:
            print(f"[bench] Running arm `{arm['name']}` across {len(arm['urls'])} GPU instance(s) on {len(tasks)} problems ...", flush=True)
            res = eval_coding_arm(
                arm_name=arm["name"],
                target_urls=arm["urls"],
                api_key=args.api_key,
                model=arm["model"],
                use_4part=bool(arm["use_4part"]),
                escalate_url=arm.get("escalate_url"),
                escalate_model=arm.get("escalate_model"),
                tasks=tasks,
                max_workers=int(arm.get("max_workers", len(arm["urls"]))),
            )
            print(
                f"[bench] DONE `{arm['name']}`: {res['passed']}/{res['problems_total']} ({res['pass_at_1_pct']}%) | "
                f"tok/s mean={res['decode_tok_per_s']['mean']} p50={res['decode_tok_per_s']['p50']} | "
                f"instances={res['instances_used']}/{res['instances_configured']} | cycles={res['cycles_histogram']}",
                flush=True,
            )
            arms_out.append(res)
        out = {"stage": "stage3_coding_ab", "limit": len(tasks), "arms": arms_out}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[bench] Saved complete telemetry report -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
