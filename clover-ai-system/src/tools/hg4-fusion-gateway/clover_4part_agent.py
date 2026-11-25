#!/usr/bin/env python3
"""Minimal 4-Part Clover Agent Loop (`clover_4part_agent.py`).

Implements ONLY the four primitives proven to drive real outcomes across any model:
  1. `AGENTS.md` Runtime Contract (`Context -> Direction -> Execution -> Outcome -> Growth`,
     anti-blind-retry, 3-stagnant-cycle `STOP_AND_ASK`, and test-tamper `REFUSE` guard).
  2. `<task_id>.md` Context File (`clover_context_read` / `clover_context_record`).
  3. `<task_id>.todos.md` Todo List (`clover_todo_list` / `clover_todo_upsert` with `depends_on`).
  4. `mcp.json` Tools & Deterministic Outcome Verifiers (`Mcp2MeshServer` tools).

Zero bloated middleware: works identically in-process on `k3` (`POST /v1/clover/solve`)
or against any OpenAI `/v1` + `/mcp` endpoint from the CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any

from clover_cycle_engine import AGENTS_MD_SHA256, CloverContextRecord


CLOVER_4PART_BASE_PROMPT = (
    "You are a Clover Agent governed by `AGENTS.md` (`Context -> Direction -> Execution -> Outcome -> Growth`).\n"
    "Rules:\n"
    "1. Establish Context first, follow human Direction, verify Outcome against reality, never blindly retry a failed step, "
    "and REFUSE any test or verifier tampering.\n"
    "2. Keep reasoning concise (<= 45 tokens before code or tool calls, no verbose `<think>` monologues)."
)

CLOVER_4PART_TOOLS_ADDENDUM = (
    "\n3. Use `clover_context_read` / `clover_context_record` for `<task_id>.md` Context handoffs and "
    "`clover_todo_list` / `clover_todo_upsert` for step-by-step Todos (`depends_on`).\n"
    "4. Delegate exact big-int/pi math to `xcruncher_exact_math`, Python checks to `python_ast_verify`, "
    "Lean 4 checks to `lean4_surface_verify`, and quantile forecasts to `chronos2_forecast`."
)


def _extract_code_block(text: str) -> str:
    clean = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    blocks = re.findall(r"```(?:python|py|lean4?|)\s*\n(.*?)```", clean, flags=re.DOTALL)
    return blocks[-1].strip() if blocks else clean.strip()


class Clover4PartAgent:
    """Executes bounded multi-turn 4-Part Clover cycles over `ClusterBatchQueue` + `Mcp2MeshServer`."""

    def __init__(self, batch_queue: Any, mcp2: Any) -> None:
        self.queue = batch_queue
        self.mcp2 = mcp2

    def _openai_tools_schema(self, allowed_names: list[str] | None = None) -> list[dict[str, Any]]:
        allow = set(allowed_names) if allowed_names else None
        out = []
        for t in self.mcp2._full_tools_list:
            if allow is not None and t["name"] not in allow:
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"],
                },
            })
        return out

    def solve(self, req: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        prompt = str(req.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required")

        model = str(req.get("model") or "clover-35b-a3b").strip()
        max_turns = max(1, min(int(req.get("max_turns") or 6), 16))
        max_tokens = max(64, min(int(req.get("max_tokens") or 512), 4096))
        temperature = float(req.get("temperature") or 0.0)
        test_code = str(req.get("test_code") or "").strip()
        lean_statement = str(req.get("lean_statement") or "").strip()
        allowed_tools = req.get("tools") if isinstance(req.get("tools"), list) else None
        allowed_set = {str(x) for x in allowed_tools} if allowed_tools is not None else None

        raw_tid = req.get("task_id")
        if raw_tid is not None and str(raw_tid).strip():
            task_id = self.mcp2.router.clover_cycle.sanitize_task_id(str(raw_tid), strict=True)
        else:
            task_id = self.mcp2.router.clover_cycle.derive_task_id({}, [{"role": "user", "content": prompt}])

        tools_schema = self._openai_tools_schema(allowed_tools)
        sys_prompt = CLOVER_4PART_BASE_PROMPT + (CLOVER_4PART_TOOLS_ADDENDUM if tools_schema else "")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        # Initialize task Context record if new so `<task_id>.md` exists from Cycle 1
        existing_ctx = self.mcp2.router.clover_cycle.get_record(task_id)
        if existing_ctx is None:
            init_rec = CloverContextRecord(
                task_id=task_id,
                cycle_count=1,
                intended_outcome=prompt[:240],
                next_cycle_change="Establish Context, plan Todos if multi-step, and execute verified tools.",
            )
            self.mcp2.router.clover_cycle.save_record(init_rec)

        # Auto-seed initial Todo items if caller passed `initial_todos`
        if isinstance(req.get("initial_todos"), list):
            for it in req["initial_todos"]:
                if isinstance(it, dict) and it.get("todo_id"):
                    self.mcp2.todo_store.upsert(
                        task_id=task_id,
                        todo_id=str(it["todo_id"]),
                        title=str(it.get("title") or it["todo_id"]),
                        status=str(it.get("status") or "pending"),
                        description=str(it.get("description") or ""),
                        depends_on=it.get("depends_on") if isinstance(it.get("depends_on"), list) else None,
                    )

        tool_trace: list[dict[str, Any]] = []
        final_answer = ""
        verified_outcome: dict[str, Any] | None = None
        discernment_state = "PROCEED"
        total_prompt_tokens = 0
        total_completion_tokens = 0
        models_used: list[str] = []

        for turn_idx in range(1, max_turns + 1):
            comp, _ = self.queue.submit(
                {
                    "model": model,
                    "messages": messages,
                    "tools": tools_schema,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "x_clover_task_id": task_id,
                },
                priority=1,
            )
            total_prompt_tokens += int(comp.usage.get("prompt_tokens") or 0)
            total_completion_tokens += int(comp.usage.get("completion_tokens") or 0)
            served = str(comp.telemetry.get("served_by") or comp.model)
            if served not in models_used:
                models_used.append(served)

            cyc_meta = (comp.telemetry.get("clover_cycle") or {}) if isinstance(comp.telemetry, dict) else {}
            discernment_state = str(cyc_meta.get("discernment_state") or "PROCEED")
            if discernment_state in ("REFUSE", "STOP_AND_ASK"):
                final_answer = comp.content or ""
                break

            tcalls = comp.tool_calls or []
            if tcalls:
                messages.append({
                    "role": "assistant",
                    "content": comp.content or "",
                    "tool_calls": tcalls,
                })
                for tc in tcalls:
                    fn = tc.get("function") or {}
                    tname = str(fn.get("name") or "")
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    except ValueError:
                        parsed_args = {}
                    # Auto-inject session `task_id` on context/todo calls when omitted or left at schema default
                    if tname in ("clover_context_read", "clover_context_record", "clover_todo_list", "clover_todo_upsert"):
                        if not parsed_args.get("task_id") or parsed_args.get("task_id") == "default_task":
                            parsed_args["task_id"] = task_id
                    if allowed_set is not None and tname not in allowed_set:
                        tout = {
                            "error": f"PermissionError: tool '{tname}' is not allowed for this request",
                            "tool": tname,
                        }
                        ok_tool = False
                    else:
                        try:
                            tout = self.mcp2._invoke_tool(tname, parsed_args)
                            ok_tool = True
                        except Exception as exc:
                            tout = {"error": f"{type(exc).__name__}: {exc}", "tool": tname}
                            ok_tool = False
                    tool_trace.append({"turn": turn_idx, "tool": tname, "args": parsed_args, "ok": ok_tool, "result": tout})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id") or f"call_{turn_idx}",
                        "name": tname,
                        "content": json.dumps(tout, ensure_ascii=False),
                    })
                continue

            # Model emitted final answer (no tool calls) -> run deterministic Outcome Verifier if configured!
            final_answer = comp.content or ""
            if test_code:
                cand_py = _extract_code_block(final_answer)
                v_out = self.mcp2._invoke_tool(
                    "python_ast_verify",
                    {"candidate_code": cand_py, "test_code": test_code},
                )
                verified_outcome = v_out
                if not v_out.get("passed"):
                    diag = v_out.get("diagnostics") or ("unsafe_ast" if not v_out.get("safe_ast") else "verification_failed")
                    if turn_idx < max_turns:
                        v_cid = f"verify_py_{turn_idx}"
                        messages.append({
                            "role": "assistant",
                            "content": final_answer,
                            "tool_calls": [{
                                "id": v_cid,
                                "type": "function",
                                "function": {"name": "python_ast_verify", "arguments": json.dumps({"candidate_code": cand_py[:240]})},
                            }],
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": v_cid,
                            "name": "python_ast_verify",
                            "content": (
                                f"error: VerificationError (`python_ast_verify` failed): {diag}\n"
                                "Update your hypothesis (do NOT blindly repeat the same code) and fix the exact root cause."
                            ),
                        })
                        continue
                break

            if lean_statement:
                v_lean = self.mcp2._invoke_tool(
                    "lean4_surface_verify",
                    {"statement": lean_statement, "candidate": final_answer, "compile": bool(req.get("lean_compile", False))},
                )
                verified_outcome = v_lean
                if not v_lean.get("passed") and turn_idx < max_turns:
                    v_cid = f"verify_lean_{turn_idx}"
                    messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "tool_calls": [{
                            "id": v_cid,
                            "type": "function",
                            "function": {"name": "lean4_surface_verify", "arguments": json.dumps({"candidate": final_answer[:240]})},
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": v_cid,
                        "name": "lean4_surface_verify",
                        "content": (
                            f"error: VerificationError (`lean4_surface_verify` failed): {v_lean.get('finding')}\n"
                            "Provide a complete Lean 4 proof with zero `sorry` or `admit`."
                        ),
                    })
                    continue
                break

            break

        # Persist final Outcome -> Growth into `<task_id>.md`
        cur_rec = self.mcp2.router.clover_cycle.get_record(task_id) or CloverContextRecord(task_id=task_id)
        cur_rec.intended_outcome = cur_rec.intended_outcome or prompt[:240]
        if tool_trace:
            cur_rec.what_was_tried.append(", ".join(f"{t['tool']}({'ok' if t['ok'] else 'err'})" for t in tool_trace[-4:]))
        if verified_outcome is not None:
            passed_flag = bool(verified_outcome.get("passed"))
            cur_rec.what_the_system_showed.append(f"verifier_passed={passed_flag}")
            if passed_flag:
                cur_rec.what_happened = "Verified PASS against real system."
                cur_rec.next_cycle_change = "Carry verified pattern forward as Growth."
            else:
                fail_diag = str(verified_outcome.get("diagnostics") or verified_outcome.get("finding") or "failed")[:140]
                ruled_item = f"{_extract_code_block(final_answer)[:90]} -> VerificationError: {fail_diag}"
                if ruled_item not in cur_rec.what_was_ruled_out:
                    cur_rec.what_was_ruled_out.append(ruled_item)
                cur_rec.what_happened = f"Verifier failed: {fail_diag}"
                cur_rec.next_cycle_change = "Inspect failing assertion & revise hypothesis without repeating ruled-out draft."
        else:
            cur_rec.what_happened = f"Completed in {len(tool_trace)} tool call(s) ({discernment_state})."
            cur_rec.next_cycle_change = "Ready for human review."
        saved_ctx = self.mcp2.router.clover_cycle.save_record(cur_rec)
        todos_snapshot = self.mcp2.todo_store.list_todos(task_id)

        return {
            "object": "clover.solve",
            "task_id": task_id,
            "agents_md_sha256": AGENTS_MD_SHA256,
            "discernment_state": discernment_state,
            "answer": final_answer,
            "verified_outcome": verified_outcome,
            "models_used": models_used,
            "turns_used": turn_idx,
            "tool_calls_executed": len(tool_trace),
            "tool_trace": tool_trace,
            "context_file": saved_ctx.context_file_path,
            "context_markdown": saved_ctx.to_markdown(),
            "todos": todos_snapshot,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
            "wall_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }


def main() -> None:
    import urllib.request

    ap = argparse.ArgumentParser(description="Run a task through Clover's 4-Part Agent Loop (`POST /v1/clover/solve`).")
    ap.add_argument("--gateway-url", default=os.environ.get("CLOVER_GATEWAY_URL", "http://127.0.0.1:8790"))
    ap.add_argument("--task-id", default="")
    ap.add_argument("--model", default="clover-35b-a3b")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--test-code", default="")
    ap.add_argument("--max-turns", type=int, default=6)
    args = ap.parse_args()

    payload = {
        "task_id": args.task_id or None,
        "model": args.model,
        "prompt": args.prompt,
        "test_code": args.test_code,
        "max_turns": args.max_turns,
    }
    hdrs = {"Content-Type": "application/json"}
    api_key = (os.environ.get("CLOVER_GATEWAY_API_KEY") or "").strip()
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        args.gateway_url.rstrip("/") + "/v1/clover/solve",
        data=json.dumps(payload).encode("utf-8"),
        headers=hdrs,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300.0) as resp:
        print(json.dumps(json.loads(resp.read().decode("utf-8")), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
