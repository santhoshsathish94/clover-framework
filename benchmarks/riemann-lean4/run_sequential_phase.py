#!/usr/bin/env python3
"""
Strict Sequential Phase Runner Under `AGENTS.md` (`benchmarks/riemann-lean4/run_sequential_phase.py`).

Enforces:
  1. Cryptographic SHA-256 verification of the frozen `chmod 444` `rh_bench_suite_v1.jsonl`
     and `lean_oracle.py` before every run (protected validation boundary).
  2. Deterministic greedy decoding (`temperature = 0.0`) and `POST /v1/cache/reset` before
     every independent benchmark task so cross-task prefix state never leaks.
  3. Toggleable, strictly isolated phase capabilities (`TokenEconomizer`, `Clover MCP S2+x-cruncher`,
     `Anthropic SKILL.md`, `Copilot Fusion Cascade`, `Elephant-Goldfish EGM`, `REA Reverse-Engineering MCP`).
  4. Paired per-task flip tracking (`+flips`, `-flips`, `net_flips`) and automatic evaluation of the
     pre-registered `AGENTS.md` Keep/Drop gate against `--prev-report`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import lean_oracle
import run_ablation
from lean_oracle import verify_lean_module
from mcp_clover_tools import AntiLeakagePremiseIndex, eval_exact_math
from run_ablation import reset_gateway_cache, run_python_sandbox

FROZEN_SUITE_SHA256 = "415c9148ad75ba7338bd365a8305baf17258e3a2fd957381ff487f7f071a1d26"
FROZEN_ORACLE_SHA256 = "52d731367acdb28593a3d86633e2e762736ae24cd6ef7d42e780499efa58c937"

TIER1_ENGLISH_SPECS: dict[str, str] = {
    "rh_t1_spec_01": (
        "The additive valuation `IsDiscreteValuationRing.addVal O` applied to the finite product "
        "`∏ i ∈ s, a i` equals the finite sum over `i ∈ s` of `IsDiscreteValuationRing.addVal O (a i)`."
    ),
    "rh_t1_spec_02": (
        "There exists an algebraic integer `c : 𝓞 F` such that `c ≠ 0` and for all `i ∈ s`, `Q i c` holds."
    ),
}

# Realistic OpenCode v1.19.6 11-tool schema payload used for deterministic Turn-1/Turn-2 & Omni-2B loop probes
OPENCODE_DEFAULT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a local file from the workspace filesystem with optional line offset and limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Absolute or workspace-relative path to read."},
                    "offset": {"type": "integer", "description": "1-based starting line number."},
                    "limit": {"type": "integer", "description": "Maximum number of lines to read."},
                },
                "required": ["filePath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Perform an exact string replacement inside an existing workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "oldString": {"type": "string", "description": "Exact existing text to replace."},
                    "newString": {"type": "string", "description": "Replacement text."},
                    "replaceAll": {"type": "boolean"},
                },
                "required": ["filePath", "oldString", "newString"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write or overwrite a complete file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}},
                "required": ["filePath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Fast file pattern matching across workspace directories using glob patterns such as **/*.lean or src/**/*.py.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Fast regular-expression content search across workspace files.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "include": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in the workspace environment with optional timeout and working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "workdir": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "webfetch",
            "description": "Fetch content from an external HTTP/HTTPS URL and convert HTML into markdown or text. Do not use for local files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "HTTP or HTTPS URL to fetch."},
                    "format": {"type": "string", "enum": ["text", "markdown", "html"]},
                    "timeout": {"type": "integer"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "websearch",
            "description": "Search the public web for external documentation, articles, and recent announcements.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "numResults": {"type": "integer"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todowrite",
            "description": "Create and update structured multi-step task checklists for complex coding sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "content": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            },
                            "required": ["id", "content", "status", "priority"],
                        },
                    }
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Launch a specialized subagent session to handle multi-step exploration or research autonomously.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "prompt": {"type": "string"},
                    "subagent_type": {"type": "string"},
                },
                "required": ["description", "prompt", "subagent_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp",
            "description": "Query Language Server Protocol operations (goToDefinition, findReferences, hover, documentSymbol).",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "filePath": {"type": "string"},
                    "line": {"type": "integer"},
                    "character": {"type": "integer"},
                },
                "required": ["operation", "filePath", "line", "character"],
            },
        },
    },
]


def verify_validation_boundary(suite_path: Path, expected_suite_sha: str | None = None) -> dict[str, str]:
    suite_sha = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    oracle_file = Path(lean_oracle.__file__).resolve()
    ablation_file = Path(run_ablation.__file__).resolve()
    runner_file = Path(__file__).resolve()

    oracle_sha = hashlib.sha256(oracle_file.read_bytes()).hexdigest()
    ablation_sha = hashlib.sha256(ablation_file.read_bytes()).hexdigest()
    runner_sha = hashlib.sha256(runner_file.read_bytes()).hexdigest()

    want_suite = expected_suite_sha or FROZEN_SUITE_SHA256
    if want_suite and suite_sha != want_suite:
        raise RuntimeError(f"FROZEN SUITE HASH MISMATCH: got {suite_sha}, expected {want_suite}")
    if FROZEN_ORACLE_SHA256 and oracle_sha != FROZEN_ORACLE_SHA256:
        raise RuntimeError(f"FROZEN ORACLE HASH MISMATCH: got {oracle_sha}, expected {FROZEN_ORACLE_SHA256}")
    return {
        "suite_sha256": suite_sha,
        "oracle_sha256": oracle_sha,
        "ablation_sha256": ablation_sha,
        "runner_sha256": runner_sha,
    }


def chat_v1(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    tools: list[dict[str, Any]] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout_s: float = 240.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if tools is not None:
        payload["tools"] = tools
    if extra_body:
        payload.update(extra_body)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer clover-local"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    usage = data.get("usage") or {}
    telemetry = data.get("x_clover_telemetry") or {}
    return msg, {"ms": ms, "usage": usage, "telemetry": telemetry, "routed_model": data.get("model", model)}


def extract_python_code(text: str) -> str:
    clean = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", clean, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else clean


def extract_lean_expr(text: str) -> str:
    clean = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    m = re.search(r"```(?:lean4?|mathlib)?\s*\n(.*?)```", clean, flags=re.DOTALL | re.IGNORECASE)
    body = m.group(1).strip() if m else clean.strip("` \n")
    # If the model echoed `theorem ... : <expr> := ...`, extract `<expr>`
    if ":=" in body:
        body = body.split(":=", 1)[0].strip()
    return body


def load_skill_section(skills_dir: Path | None, skill_name: str) -> str:
    if skills_dir is None:
        return ""
    sp = skills_dir / skill_name / "SKILL.md"
    if not sp.is_file():
        return ""
    # Parse and validate via the shared `skill_loader` implementation if available
    gw_dir = Path("/opt/clover-k3/hg4-fusion-gateway")
    local_gw = Path(__file__).resolve().parents[2] / "clover-ai-system" / "src" / "tools" / "hg4-fusion-gateway"
    for cand in (gw_dir, local_gw):
        if cand.is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    import skill_loader
    entry = skill_loader.parse_skill_md(sp)
    return f"\n\n[Active Skill: {entry.name}]\n{entry.body}\n"


def eval_tier1_task(
    task: dict[str, Any],
    base_url: str,
    base_model: str,
    lean_proj: Path,
    skills_dir: Path | None,
    enable_cascade: bool,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    stmt = task["lean_statement_prefix"]
    # Split right at the conclusion colon before the return Prop
    # Both frozen Tier-1 statements have `:\n    <GoalProp> := by`
    head_and_goal = stmt[:-len(":= by")].rstrip() if stmt.rstrip().endswith(":= by") else stmt
    colon_idx = head_and_goal.rfind(":\n")
    if colon_idx < 0:
        raise ValueError(f"Unexpected Tier-1 statement layout in {task['task_id']}")
    decl_head = head_and_goal[:colon_idx].strip()
    gt_prop = head_and_goal[colon_idx + 1:].strip()

    # Convert `private lemma name` / `theorem name` into an `example` binder prefix
    binder_head = re.sub(
        r"^(.*?\n)(?:private\s+|protected\s+)?(?:theorem|lemma)\s+[A-Za-z0-9_']+\s*",
        r"\1theorem _spec_equiv_check ",
        decl_head,
        flags=re.DOTALL,
    )

    eng_spec = TIER1_ENGLISH_SPECS[task["task_id"]]
    skill_txt = load_skill_section(skills_dir, "lean4-mathlib-proving")
    sys_msg = (
        "You are an expert Lean 4 (v4.33.1 + Mathlib) formalizer. "
        "Output ONLY the Lean 4 goal `Prop` expression inside ```lean ... ``` (do NOT repeat `theorem`, binders, or `:= by`)."
        + skill_txt
    )
    user_msg = (
        f"Given the Lean 4 declaration context and binders:\n```lean\n{decl_head} :\n```\n"
        f"Formalize this exact conclusion proposition:\n\"{eng_spec}\"\n"
        "Output ONLY the Lean 4 `Prop` expression inside ```lean ... ```."
    )
    msgs: list[dict[str, Any]] = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
    model_to_call = "clover-fusion-high" if enable_cascade else base_model
    msg1, m1 = chat_v1(base_url, model_to_call, msgs, max_tokens=256, extra_body=extra_body)
    tokens_used = int(m1["usage"].get("total_tokens", 0))
    cand_prop = extract_lean_expr(str(msg1.get("content") or ""))

    equiv_prefix = f"{binder_head} :\n    ({gt_prop}) ↔ ({cand_prop}) := by\n"
    chk1 = verify_lean_module(
        equiv_prefix,
        "first | exact Iff.rfl | simp only [and_comm, and_left_comm, and_assoc]",
        lean_project_dir=lean_proj,
        thm_name="_spec_equiv_check",
    )
    escalated = False
    final_chk = chk1
    if not chk1.passed and enable_cascade:
        escalated = True
        msgs.extend([
            {"role": "assistant", "content": str(msg1.get("content") or "")},
            {
                "role": "tool",
                "name": "lean4_spec_check",
                "content": (
                    f"error: Lean 4 rejected the formalized Prop ({chk1.stage}):\n{chk1.diagnostics[:900]}\n"
                    "Output ONLY the corrected Lean 4 `Prop` expression inside ```lean ... ```."
                ),
            },
        ])
        msg2, m2 = chat_v1(base_url, "clover-fusion-high", msgs, max_tokens=256, extra_body=extra_body)
        tokens_used += int(m2["usage"].get("total_tokens", 0))
        cand_prop2 = extract_lean_expr(str(msg2.get("content") or ""))
        equiv_prefix2 = f"{binder_head} :\n    ({gt_prop}) ↔ ({cand_prop2}) := by\n"
        final_chk = verify_lean_module(
            equiv_prefix2,
            "first | exact Iff.rfl | simp only [and_comm, and_left_comm, and_assoc]",
            lean_project_dir=lean_proj,
            thm_name="_spec_equiv_check",
        )

    return {
        "task_id": task["task_id"],
        "tier": task["tier"],
        "passed": final_chk.passed,
        "stage": final_chk.stage,
        "escalated": escalated,
        "tokens": tokens_used,
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def eval_tier2_task(
    task: dict[str, Any],
    base_url: str,
    base_model: str,
    lean_proj: Path,
    premise_idx: AntiLeakagePremiseIndex | None,
    skills_dir: Path | None,
    enable_mcp: bool,
    enable_cascade: bool,
    enable_egm: bool,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    prefix = task["immutable_prefix"]
    suffix = task.get("immutable_suffix", "")
    psha = task["prefix_sha256"]
    thm_name = task.get("thm_name")
    tokens_used = 0

    premise_hint = ""
    if enable_mcp and premise_idx is not None:
        mod_dir = str(Path(task["module_relpath"]).parent.as_posix())
        # Query using both the declaration identifiers and key Mathlib concepts in the statement
        stmt_tail = prefix[max(0, len(prefix) - 800):]
        hits = premise_idx.search(stmt_tail, excluded_module_substr=mod_dir, excluded_lemma=thm_name or "", top_k=8)
        if hits:
            premise_hint = "\n\n[Clover S2 Premise Search — Verified Mathlib4 Signatures (anti-leakage filtered)]:\n" + "\n".join(
                f"- `{h['signature']}`" for h in hits
            )

    egm_plan = ""
    if enable_egm:
        el_msgs = [
            {
                "role": "system",
                "content": (
                    "You are the EGM Elephant Architect for Lean 4 (v4.33.1 + Mathlib). "
                    "Write a concise <= 4-bullet proof plan naming exact Mathlib lemmas and tactic structure. Zero prose fluff, no sorry."
                ),
            },
            {"role": "user", "content": f"Theorem:\n```lean\n{prefix}```{premise_hint}"},
        ]
        el_msg, el_meta = chat_v1(base_url, "clover-egm-elephant", el_msgs, max_tokens=220, extra_body=extra_body)
        el_txt = str(el_msg.get("content") or "")
        tokens_used += int(el_meta["usage"].get("total_tokens", 0))

        gf_msgs = [
            {
                "role": "system",
                "content": (
                    "You are a stateless EGM Goldfish Auditor. Seeing ONLY the theorem and plan, flag any nonexistent lemma names, "
                    "missing base/zero cases, or Lean 4 universe/coercion pitfalls in <= 3 short bullets. Advisory only."
                ),
            },
            {"role": "user", "content": f"Theorem:\n```lean\n{prefix}```\nPlan:\n{el_txt}"},
        ]
        gf_msg, gf_meta = chat_v1(base_url, "clover-egm-goldfish", gf_msgs, max_tokens=160, extra_body=extra_body)
        gf_txt = str(gf_msg.get("content") or "")
        tokens_used += int(gf_meta["usage"].get("total_tokens", 0))
        egm_plan = f"\n\n[EGM Elephant Strategy]:\n{el_txt}\n[EGM Goldfish Advisory]:\n{gf_txt}\n"

    skill_txt = load_skill_section(skills_dir, "lean4-mathlib-proving")
    sys_prompt = (
        "You are an expert Lean 4 (v4.33.1 + Mathlib) theorem prover. "
        "Write ZERO prose explanation: start IMMEDIATELY with ```lean and output ONLY the tactic lines after `:= by`. "
        "Never use `sorry`, `admit`, or `axiom`."
        + skill_txt
    )
    user_prompt = f"Complete the Lean 4 proof after `:= by` (output ONLY ```lean ... ```):\n```lean\n{prefix}```{premise_hint}{egm_plan}"

    first_model = "clover-fusion-high" if enable_cascade else base_model
    msgs: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    msg1, m1 = chat_v1(base_url, first_model, msgs, max_tokens=768, extra_body=extra_body)
    reply1 = str(msg1.get("content") or "")
    tokens_used += int(m1["usage"].get("total_tokens", 0))

    chk1 = verify_lean_module(
        prefix,
        reply1,
        lean_project_dir=lean_proj,
        immutable_suffix=suffix,
        expected_prefix_sha256=psha,
        thm_name=thm_name,
    )
    escalated = False
    final_chk = chk1
    if not chk1.passed and enable_cascade:
        escalated = True
        msgs.extend([
            {"role": "assistant", "content": reply1},
            {
                "role": "tool",
                "name": "lean4_kernel_check",
                "content": (
                    f"error: Lean 4 kernel rejected the tactic proof ({chk1.stage}):\n{chk1.diagnostics[:1200]}\n"
                    "Fix the exact error above and output ONLY the corrected ```lean ... ``` tactic lines after `:= by`."
                ),
            },
        ])
        msg2, m2 = chat_v1(base_url, "clover-fusion-high", msgs, max_tokens=896, extra_body=extra_body)
        reply2 = str(msg2.get("content") or "")
        tokens_used += int(m2["usage"].get("total_tokens", 0))
        final_chk = verify_lean_module(
            prefix,
            reply2,
            lean_project_dir=lean_proj,
            immutable_suffix=suffix,
            expected_prefix_sha256=psha,
            thm_name=thm_name,
        )

    return {
        "task_id": task["task_id"],
        "tier": task["tier"],
        "length_bin": task["length_bin"],
        "passed": final_chk.passed,
        "stage": final_chk.stage,
        "diagnostics": final_chk.diagnostics[:320] if not final_chk.passed else "",
        "escalated": escalated,
        "tokens": tokens_used,
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def _parse_provable_verdict(raw_txt: str) -> bool:
    clean = re.sub(r"<think>.*?</think>", "", raw_txt or "", flags=re.DOTALL).strip()
    up = clean.upper()
    if clean.lstrip().startswith("{"):
        try:
            j = json.loads(clean)
            sv = str(j.get("subgoal_verdict") or "").upper()
            if sv == "UNPROVABLE":
                return False
            if sv == "PROVABLE":
                return True
            findings = j.get("findings") or []
            return not any(str(f.get("severity", "")).lower() in ("critical", "high", "error") for f in findings)
        except ValueError:
            return ("VERDICT: PROVABLE" in up) and ("UNPROVABLE" not in up.splitlines()[0])
    first_verdict = re.search(r"VERDICT\s*:\s*(PROVABLE|UNPROVABLE)", up)
    if first_verdict:
        return first_verdict.group(1) == "PROVABLE"
    return "UNPROVABLE" not in up[:160]


def eval_tier3_task(
    task: dict[str, Any],
    base_url: str,
    base_model: str,
    skills_dir: Path | None,
    enable_mcp: bool,
    enable_egm: bool,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    tokens_used = 0
    skill_txt = load_skill_section(skills_dir, "lean4-mathlib-proving")

    mcp_note = ""
    if enable_mcp and "Finset.prod" in task["proposed_lean_subgoal"]:
        chk = eval_exact_math("0 * 1")
        if chk == "0":
            mcp_note = "\n[Clover x-cruncher Exact Check Available]: Test boundary values `0`, `1`, empty `Finset`, and non-unit elements."

    target_model = "clover-egm-goldfish" if enable_egm else base_model
    msgs: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a formal Lean 4 / Mathlib specification auditor. "
                "Inspect the proposed Lean 4 subgoal statement BEFORE any proof attempt. "
                "Check every binder and hypothesis carefully. "
                "Start your first line with `VERDICT: PROVABLE` or `VERDICT: UNPROVABLE`, followed by a 1-2 line counterexample or justification."
                + skill_txt
            ),
        },
        {
            "role": "user",
            "content": f"Task: {task['title']}\nProposed Lean 4 subgoal:\n```lean\n{task['proposed_lean_subgoal']}\n```{mcp_note}",
        },
    ]

    msg1, meta1 = chat_v1(base_url, target_model, msgs, max_tokens=256, extra_body=extra_body)
    txt1 = str(msg1.get("content") or "")
    tokens_used += int(meta1["usage"].get("total_tokens", 0))
    truth = bool(task["is_provable"])
    stage1_pred = _parse_provable_verdict(txt1)
    goldfish_fr = bool(enable_egm and truth and (not stage1_pred))

    pred_provable = stage1_pred
    elephant_fr = False
    if enable_egm:
        el_verify_msgs = [
            {
                "role": "system",
                "content": (
                    "You are the EGM Elephant Architect verifying a Goldfish subgoal audit. "
                    "Check whether any proposed counterexample actually satisfies ALL stated Lean 4 hypotheses and typeclasses, "
                    "or whether a genuine missing hypothesis makes the statement false. "
                    "Output `VERDICT: PROVABLE` or `VERDICT: UNPROVABLE` on line 1."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Proposed Lean 4 subgoal:\n```lean\n{task['proposed_lean_subgoal']}\n```\n"
                    f"Goldfish Audit:\n{txt1}"
                ),
            },
        ]
        el_msg, el_meta = chat_v1(base_url, "clover-egm-elephant", el_verify_msgs, max_tokens=220, extra_body=extra_body)
        txt2 = str(el_msg.get("content") or "")
        tokens_used += int(el_meta["usage"].get("total_tokens", 0))
        pred_provable = _parse_provable_verdict(txt2)
        elephant_fr = bool(truth and (not pred_provable))

    correct = pred_provable == truth
    false_reject = truth and (not pred_provable)
    return {
        "task_id": task["task_id"],
        "tier": task["tier"],
        "passed": correct,
        "false_rejection": false_reject,
        "goldfish_false_rejection": goldfish_fr,
        "elephant_false_rejection": elephant_fr,
        "pred_provable": pred_provable,
        "ground_truth_provable": truth,
        "tokens": tokens_used,
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def eval_code_task(
    task: dict[str, Any],
    base_url: str,
    base_model: str,
    skills_dir: Path | None,
    enable_mcp: bool,
    enable_cascade: bool,
    enable_egm: bool,
    enable_rea: bool,
    rea_inspector: Any | None,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    tokens_used = 0
    is_re = task["tier"] == "tier5_reverse_engineering"

    skill_name = "reverse-engineering-rea" if (is_re and enable_rea) else "systems-python-swe"
    skill_txt = load_skill_section(skills_dir, skill_name)

    re_evidence = ""
    if is_re and enable_rea and rea_inspector is not None:
        re_evidence = rea_inspector(task)

    egm_spec = ""
    if enable_egm:
        el_msgs = [
            {
                "role": "system",
                "content": "You are the EGM Elephant Architect. List exact input/output invariants and boundary edge cases in <= 4 short bullets. Write zero code.",
            },
            {"role": "user", "content": f"{task['prompt']}{re_evidence}"},
        ]
        el_msg, el_meta = chat_v1(base_url, "clover-egm-elephant", el_msgs, max_tokens=200, extra_body=extra_body)
        tokens_used += int(el_meta["usage"].get("total_tokens", 0))
        egm_spec = f"\n[EGM Edge-Case Specification]:\n{el_msg.get('content', '')}\n"

    sys_prompt = (
        "You are a senior systems & security software engineer. "
        "Return clean, self-contained Python 3 code inside ```python ... ``` with zero external third-party dependencies."
        + skill_txt
    )
    user_prompt = f"{task['prompt']}{re_evidence}{egm_spec}"
    first_model = "clover-fusion-high" if enable_cascade else base_model
    msgs: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

    msg1, m1 = chat_v1(base_url, first_model, msgs, max_tokens=900, extra_body=extra_body)
    reply1 = str(msg1.get("content") or "")
    tokens_used += int(m1["usage"].get("total_tokens", 0))
    code1 = extract_python_code(reply1)
    passed, err = run_python_sandbox(code1, test_code=task["test_code"])

    escalated = False
    if not passed and enable_cascade:
        escalated = True
        msgs.extend([
            {"role": "assistant", "content": reply1},
            {
                "role": "tool",
                "name": "python_test_oracle",
                "content": f"error: unit test failed:\n{err}\nOutput ONLY the corrected Python 3 implementation inside ```python ... ```.",
            },
        ])
        msg2, m2 = chat_v1(base_url, "clover-fusion-high", msgs, max_tokens=1024, extra_body=extra_body)
        reply2 = str(msg2.get("content") or "")
        tokens_used += int(m2["usage"].get("total_tokens", 0))
        code2 = extract_python_code(reply2)
        passed, err = run_python_sandbox(code2, test_code=task["test_code"])

    return {
        "task_id": task["task_id"],
        "tier": task["tier"],
        "passed": passed,
        "error": err,
        "escalated": escalated,
        "tokens": tokens_used,
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def run_agentic_probes(base_url: str, base_model: str, extra_body: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministic multi-turn OpenCode tool-payload & Omni-2B loop-escape probes:
      1. `opencode_turn1_turn2`: Sends OpenCode's 11-tool schema on Turn 1 + follow-up Turn 2,
         measuring exact Turn-1 prompt tokens, Turn-1 wall-clock seconds, and Turn-2 cached prefix reuse.
      2. `omni2b_webfetch_loop`: Feeds `lfm3-omni-2b` a local file question where `webfetch` already failed 3x
         with `404 Not Found` and checks whether `lfm3-omni-2b` escapes `webfetch` on the next turn.
    """
    reset_gateway_cache(base_url)
    t1_msgs: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are OpenCode, an interactive CLI software engineering agent working in /mnt/nvme1/clover-rh-bench/opencode-smoke.",
        },
        {
            "role": "user",
            "content": "Read Demo.lean using the `read` tool and tell me the exact theorem name.",
        },
    ]
    msg1, meta1 = chat_v1(base_url, base_model, t1_msgs, max_tokens=128, tools=OPENCODE_DEFAULT_TOOLS, extra_body=extra_body)
    t1_prompt_tok = int(meta1["usage"].get("prompt_tokens", 0))
    t1_wall_s = round(meta1["ms"] / 1000.0, 2)
    tc1 = msg1.get("tool_calls") or []
    tc1_id = tc1[0]["id"] if tc1 else "call_demo_1"

    t2_msgs = list(t1_msgs) + [
        {
            "role": "assistant",
            "content": msg1.get("content"),
            "tool_calls": tc1 if tc1 else [
                {"id": tc1_id, "type": "function", "function": {"name": "read", "arguments": "{\"filePath\":\"Demo.lean\"}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tc1_id,
            "name": "read",
            "content": "theorem demo_add_zero (n : Nat) : n + 0 = n := Nat.add_zero n\n",
        },
    ]
    msg2, meta2 = chat_v1(base_url, base_model, t2_msgs, max_tokens=96, tools=OPENCODE_DEFAULT_TOOLS, extra_body=extra_body)
    t2_wall_s = round(meta2["ms"] / 1000.0, 2)
    t2_ok = "demo_add_zero" in str(msg2.get("content") or "")

    # Probe 2: LFM3-Omni-2B repeated webfetch 404 distraction loop probe (OpenCode omits `name` on role="tool")
    reset_gateway_cache(base_url)
    loop_msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a coding assistant in /mnt/nvme1/clover-rh-bench/opencode-smoke."},
        {"role": "user", "content": "Read the local file `Demo.lean` and state the theorem name."},
    ]
    for k in range(1, 4):
        cid = f"call_wf_{k}"
        loop_msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": "webfetch", "arguments": "{\"url\":\"http://localhost/Demo.lean\"}"}}],
        })
        loop_msgs.append({
            "role": "tool",
            "tool_call_id": cid,
            "content": "Error: HTTP 404 Not Found fetching http://localhost/Demo.lean",
        })
    omni_msg, omni_meta = chat_v1(base_url, "lfm3-omni-2b", loop_msgs, max_tokens=128, tools=OPENCODE_DEFAULT_TOOLS, extra_body=extra_body)
    omni_tcs = omni_msg.get("tool_calls") or []
    called_webfetch_again = any((tc.get("function") or {}).get("name") == "webfetch" for tc in omni_tcs)
    called_read = any((tc.get("function") or {}).get("name") == "read" for tc in omni_tcs)

    reset_gateway_cache(base_url)
    return {
        "opencode_turn1_prompt_tokens": t1_prompt_tok,
        "opencode_turn1_wall_s": t1_wall_s,
        "opencode_turn2_wall_s": t2_wall_s,
        "opencode_turn2_answered_correctly": t2_ok,
        "omni2b_prompt_tokens": int(omni_meta["usage"].get("prompt_tokens", 0)),
        "omni2b_escaped_webfetch_loop": not called_webfetch_again,
        "omni2b_switched_to_read": called_read,
    }


def evaluate_keep_drop_gate(curr: dict[str, Any], prev: dict[str, Any] | None) -> dict[str, Any]:
    if prev is None:
        return {"decision": "BASELINE", "reason": "Phase B clean unmodified baseline anchored."}

    prev_map = {r["task_id"]: bool(r["passed"]) for r in prev["results"]}
    curr_map = {r["task_id"]: bool(r["passed"]) for r in curr["results"]}

    plus_flips = sorted(tid for tid, ok in curr_map.items() if ok and not prev_map.get(tid, False))
    minus_flips = sorted(tid for tid, ok in curr_map.items() if (not ok) and prev_map.get(tid, False))
    net_flips = len(plus_flips) - len(minus_flips)

    p_tok = prev["probes"]["opencode_turn1_prompt_tokens"]
    c_tok = curr["probes"]["opencode_turn1_prompt_tokens"]
    tok_reduction_pct = round(100.0 * (p_tok - c_tok) / max(1, p_tok), 2)

    p_wall = prev["total_wall_s"]
    c_wall = curr["total_wall_s"]
    wall_reduction_pct = round(100.0 * (p_wall - c_wall) / max(0.01, p_wall), 2)

    keep = False
    reason = ""
    if net_flips >= 1 and len(minus_flips) == 0:
        keep = True
        reason = f"+{net_flips} net verified task flip(s) with 0 regressions ({plus_flips})."
    elif net_flips >= 2:
        keep = True
        reason = f"+{net_flips} net verified task flips ({plus_flips} vs {minus_flips})."
    elif net_flips == 0 and len(minus_flips) == 0 and (tok_reduction_pct >= 15.0 or wall_reduction_pct >= 15.0):
        keep = True
        reason = (
            f"0 regressions with {tok_reduction_pct}% Turn-1 prompt-token reduction "
            f"and {wall_reduction_pct}% wall-clock reduction (>= 15% threshold)."
        )
    else:
        keep = False
        reason = (
            f"Did not clear Keep gate: net_flips={net_flips} (+{len(plus_flips)}/-{len(minus_flips)}), "
            f"Turn-1 token delta={tok_reduction_pct}%, wall delta={wall_reduction_pct}%."
        )

    return {
        "decision": "KEEP" if keep else "DROP",
        "net_flips": net_flips,
        "plus_flips": plus_flips,
        "minus_flips": minus_flips,
        "turn1_token_reduction_pct": tok_reduction_pct,
        "wall_clock_reduction_pct": wall_reduction_pct,
        "reason": reason,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Sequential AGENTS.md Phase Evaluator")
    ap.add_argument("--phase", required=True, help="Phase identifier (e.g. B, C, D, E, F, G_standalone, G_combined, H, I)")
    ap.add_argument("--suite", type=Path, default=Path("/mnt/nvme1/clover-rh-bench/rh_bench_suite_v2.jsonl"))
    ap.add_argument("--oracle", type=Path, default=Path("/mnt/nvme1/clover-rh-bench/harness/lean_oracle.py"))
    ap.add_argument("--lean-project", type=Path, default=Path("/mnt/nvme1/clover-rh-bench/rh-src/lean"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8790/v1")
    ap.add_argument("--base-model", default="clover-35b-a3b")
    ap.add_argument("--skills-dir", type=Path, default=None)
    ap.add_argument("--enable-tokeneconomizer", action="store_true")
    ap.add_argument("--enable-clover-mcp", action="store_true")
    ap.add_argument("--enable-skills", action="store_true")
    ap.add_argument("--enable-cascade", action="store_true")
    ap.add_argument("--enable-egm", action="store_true")
    ap.add_argument("--enable-rea", action="store_true")
    ap.add_argument("--expected-suite-sha256", default=None)
    ap.add_argument("--prev-report", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    hashes = verify_validation_boundary(args.suite, args.expected_suite_sha256)
    tasks = [json.loads(line) for line in args.suite.read_text(encoding="utf-8").splitlines() if line.strip()]

    extra_body: dict[str, Any] = {
        "x_clover_economizer": bool(args.enable_tokeneconomizer),
        "x_clover_output_budget": bool(args.enable_rea),
        "x_clover_skills_index": bool(args.enable_skills),
    }
    if args.skills_dir is not None:
        extra_body["x_clover_skills_dir"] = str(args.skills_dir)

    premise_idx: AntiLeakagePremiseIndex | None = None
    if args.enable_clover_mcp:
        premise_idx = AntiLeakagePremiseIndex(args.lean_project)

    rea_inspector = None
    if args.enable_rea:
        from rea_bridge import build_rea_evidence_for_task
        rea_inspector = build_rea_evidence_for_task

    skills_dir = args.skills_dir if args.enable_skills else None

    probes = run_agentic_probes(args.base_url, args.base_model, extra_body)

    results: list[dict[str, Any]] = []
    t_suite0 = time.perf_counter()
    for idx, t in enumerate(tasks, start=1):
        reset_gateway_cache(args.base_url)
        tier = t["tier"]
        if tier == "tier1_spec_roundtrip":
            r = eval_tier1_task(t, args.base_url, args.base_model, args.lean_project, skills_dir, args.enable_cascade, extra_body)
        elif tier == "tier2_leaf_tactic_repair":
            r = eval_tier2_task(
                t, args.base_url, args.base_model, args.lean_project,
                premise_idx, skills_dir, args.enable_clover_mcp, args.enable_cascade, args.enable_egm, extra_body,
            )
        elif tier == "tier3_egm_subgoal_audit":
            r = eval_tier3_task(t, args.base_url, args.base_model, skills_dir, args.enable_clover_mcp, args.enable_egm, extra_body)
        else:
            r = eval_code_task(
                t, args.base_url, args.base_model, skills_dir,
                args.enable_clover_mcp, args.enable_cascade, args.enable_egm, args.enable_rea, rea_inspector, extra_body,
            )
        results.append(r)
        print(
            f"[{args.phase} {idx:02d}/{len(tasks):02d}] {t['task_id']} ({tier}) -> "
            f"passed={r['passed']} wall={r['wall_s']}s tok={r['tokens']}",
            flush=True,
        )

    total_wall_s = round(time.perf_counter() - t_suite0, 2)
    total_passed = sum(1 for r in results if r["passed"])
    total_tokens = sum(int(r.get("tokens", 0)) for r in results)
    passes_per_hr = round(total_passed / max(total_wall_s / 3600.0, 1e-6), 2)

    by_tier: dict[str, dict[str, int]] = {}
    for r in results:
        tr = r["tier"]
        by_tier.setdefault(tr, {"passed": 0, "total": 0})
        by_tier[tr]["total"] += 1
        if r["passed"]:
            by_tier[tr]["passed"] += 1

    report: dict[str, Any] = {
        "phase": args.phase,
        "flags": {
            "base_model": args.base_model,
            "tokeneconomizer": args.enable_tokeneconomizer,
            "clover_mcp": args.enable_clover_mcp,
            "skills": args.enable_skills,
            "cascade": args.enable_cascade,
            "egm": args.enable_egm,
            "rea": args.enable_rea,
        },
        "boundary_hashes": hashes,
        "total_passed": total_passed,
        "total_tasks": len(results),
        "by_tier": by_tier,
        "total_tokens": total_tokens,
        "total_wall_s": total_wall_s,
        "passes_per_hour": passes_per_hr,
        "escalations": sum(1 for r in results if r.get("escalated")),
        "false_rejections": sum(1 for r in results if r.get("false_rejection")),
        "goldfish_false_rejections": sum(1 for r in results if r.get("goldfish_false_rejection")),
        "elephant_false_rejections": sum(1 for r in results if r.get("elephant_false_rejection")),
        "probes": probes,
        "results": results,
    }

    prev_data = json.loads(args.prev_report.read_text(encoding="utf-8")) if (args.prev_report and args.prev_report.is_file()) else None
    report["gate"] = evaluate_keep_drop_gate(report, prev_data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
