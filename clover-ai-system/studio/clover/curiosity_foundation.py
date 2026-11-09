"""Clover 5.x Curiosity Foundation (`curiosity_foundation.py`).

Implements the persistent, developmental curiosity and metacognitive feedback substrate for Clover 5.x across all 13
foundational research papers (`01`–`13`), governed by three architectural principles:
  1. **Reuse Clover's existing components wherever possible** (`CloverMemoryGraphStore`, `CloverOntologyGraph`,
     `CloverJEPARuntime`, `LayerKit`, `clover_cycle`, `Repo`, `MemoryIndex`, `IsolatedPythonSandbox`, `semantics.py`).
  2. **Build the missing underlying motivation and feedback mechanisms rather than adding another question-generating agent**:
     - `CuriosityRegionTree` (`R-IAC` Criterion C2 output-variance splitting + parent-to-child state inheritance, Paper 01)
     - Non-saturating Plasticity-Weighted Progress + Habituation Cooldown Trace (Paper 02)
     - Endogenous Interest Emergence inside Clover's `semantics.py` Discernment Hierarchy (Paper 03)
     - Coupled Knowledge Progress (`LPM`, Eq. 11) + Competence Progress (`CPM`, Eq. 24) + Predictive Surprise Ratio
       (`SM`, Eq. 15) + Controllability Gate defeating the Passive Spectator Trap (Paper 04)
     - Dual-Channel Phasic Surprise vs. Tonic Regional Progress in Category-Level Microcolumns (`MR + metaMR`, Paper 05)
     - POMDP Observing Action Value of Information (`VoI`) with Indifference Threshold `p* = 1 + r_o` + Local `LP`
       on Branch-Expanding Frontiers (Paper 06)
     - Endogenous Belief-Based Utility Loop + Anti-Confirmation Active Sampler separating Neutral Diagnosticity from
       Confirmation Counts (Paper 07)
     - Closed `LP <-> State Curiosity <-> Memory Consolidation` Feedback Loop (Fig. 1B) + `ZPDES` Dynamic Zone of
       Proximal Development with Lateral Prerequisite/Modality Switching (Fig. 5, Paper 08)
     - Deterministic Structural-First Feedback (`_eval`, Compiler, Horn Inference, Sandbox) Driving Emergent
       Workspace-Specific Developmental Attractors (Paper 09)
     - Confidence/Reliability Module `C(R_n)` predicting Expected Error `y_C`, Signed Reliability Improvement
       `dot_y_C(t) = y_C_old - y_C_new`, and Smoothed Signed Shift `|E[Delta z]|` (Jensen's Inequality Guard, Paper 10)
     - `SGIM-ACTS` Epistemic Strategy Bandit (`(region, strategy, actor)`) with Automatic Compensation Against
       Static/Biased Sources & 7-View Coverage Tracking (Paper 11)
     - `SGIM-D` Social Goal Emulation + Policy Imitation with Self-Execution Calibration, `|LP|` Competence-Drop
       Vigilance (Eq. 3), and Reliability-Weighted Retrieval `dist + alpha * var_h` (Algorithm 4.6, Paper 12)
     - `IMOL` Action-Controllable Subspace Filter (`R^2(Action -> Outcome)`) + Dual Episodic-Semantic Memory
       Consolidation with Non-Regression Replay (Paper 13)
  3. **Keep curiosity persistent across interactions** (`<repo_dir>/growth/curiosity/curiosity_state.json`), with its
     own evolving interests, unresolved questions, exploration history, and learning progress — and **zero hardcoded
     lists of topics, books, or questions Clover should find interesting**.
"""
from __future__ import annotations

import collections
import copy
import datetime
import hashlib
import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

SCHEMA_VERSION = "5.1.0"
D_LATENT = 64

# Clover's 5 existing epistemic strategy primitives (Paper 11 SGIM-ACTS)
EPISTEMIC_STRATEGIES: tuple[str, ...] = (
    "memory_graph_recall",
    "workspace_code_inspect",
    "sandbox_counterexample_probe",
    "web_primary_research",
    "human_clarification_ask",
)

# Observation cost r_o per epistemic strategy (Paper 06 Fig. 1A POMDP VoI)
STRATEGY_OBSERVATION_COST: dict[str, float] = {
    "memory_graph_recall": 0.04,
    "workspace_code_inspect": 0.07,
    "sandbox_counterexample_probe": 0.10,
    "web_primary_research": 0.14,
    "human_clarification_ask": 0.18,
}

# Strategy-to-action_prior_8 dimension mapping (bridges into jepa_engine.semantic_action_prior_8)
STRATEGY_TO_ACTION_DIM: dict[str, int] = {
    "web_primary_research": 0,         # Dim 0: web_factual
    "sandbox_counterexample_probe": 2, # Dim 2: code_python (also boosts Dim 1 math_calc)
    "memory_graph_recall": 3,          # Dim 3: memory_knowledge
    "workspace_code_inspect": 4,       # Dim 4: files_workspace
    "human_clarification_ask": 7,      # Dim 7: general_dialogue
}

# Required permission key in permissions.py for each epistemic strategy
STRATEGY_REQUIRED_PERMISSION: dict[str, str | None] = {
    "memory_graph_recall": "memory",
    "workspace_code_inspect": "files",
    "sandbox_counterexample_probe": "code",
    "web_primary_research": "internet",
    "human_clarification_ask": None,
}

# Concrete Studio tool names corresponding to each epistemic strategy
STRATEGY_TO_STUDIO_TOOLS: dict[str, tuple[str, ...]] = {
    "memory_graph_recall": (),
    "workspace_code_inspect": ("read_file", "search_files", "list_directory"),
    "sandbox_counterexample_probe": ("run_python", "calculate"),
    "web_primary_research": ("web_search", "fetch_url"),
    "human_clarification_ask": (),
}


@dataclass
class IdleResourceBudget:
    """Explicit resource and timing budget for one bounded idle curiosity step in `studio_idle()`."""
    max_steps_per_idle_pass: int = 1
    max_wall_ms: float = 1500.0
    max_sandbox_timeout_s: float = 2.0
    max_files_read: int = 2
    max_bytes_read: int = 32768
    max_single_file_bytes: int = 16384
    min_idle_interval_s: float = 0.0


# Content-level sensitive patterns excluded during workspace inspection (Fix 2: Exploration Safety)
_SENSITIVE_CONTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{40,255})\b")),
    ("api_key", re.compile(r"\b(?:sk-(?:proj-)?[A-Za-z0-9_\-]{20,}|tvly-[A-Za-z0-9_\-]{16,}|AIza[0-9A-Za-z_\-]{35})\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("jwt_token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("credential_uri", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^/\s:@\"']+:[^/\s@\"']+@[^\s\"']+")),
    (
        "secret_assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api_key|apikey|access_token|auth_token|private_key|"
            r"client_secret|db_pass|connection_string|ssn|bank_routing|patient_record)\s*[:=]\s*['\"][^'\"]{4,}['\"]"
        ),
    ),
    ("ssn_pattern", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
)


def contains_sensitive_content(text: str) -> tuple[bool, list[str]]:
    """Inspect file content for secrets, credentials, private keys, or PII before any AST parsing or memory storage."""
    if not text:
        return False, []
    matched_kinds: list[str] = []
    for kind, rx in _SENSITIVE_CONTENT_PATTERNS:
        if rx.search(text):
            matched_kinds.append(kind)
    return bool(matched_kinds), matched_kinds


def snapshot_workspace_manifest(workspace_root: str | Path | None) -> dict[str, tuple[int, int]]:
    """Return a fast `{rel_posix_path: (size_bytes, mtime_ns)}` manifest of `workspace_root` to verify that
    idle exploration never creates, modifies, or deletes any workspace file."""
    if workspace_root is None:
        return {}
    root = Path(workspace_root)
    if not root.is_dir():
        return {}
    snap: dict[str, tuple[int, int]] = {}
    for fp in sorted(root.rglob("*")):
        if any(p in ("node_modules", "__pycache__", ".git", "venv") for p in fp.parts):
            continue
        try:
            if fp.is_file() and not fp.is_symlink():
                st = fp.stat()
                snap[fp.relative_to(root).as_posix()] = (int(st.st_size), int(st.st_mtime_ns))
        except OSError:
            continue
    return snap


def validate_behavioral_verification_probe(code: str) -> dict[str, Any]:
    """Fix 1 (Verification Integrity): Statically audit a candidate Python verification script `code` so that
    a hypothesis can NEVER become `externally_verified` merely because its code executes without error, assigns
    a constant `RESULT = True` / `{'passed': True}`, or checks `callable(fn)` / `hasattr(...)`.

    Requires:
      1. Valid Python syntax.
      2. If `code` defines one or more functions (`def fn(...)`), the probe MUST invoke at least one defined
         target function on at least 2 distinct argument tuples inside `assert` statements or inside the
         boolean expression computing `RESULT` / `RESULT['passed']`.
      3. If `code` does not define a function, it must evaluate at least 2 non-constant, non-existence comparisons
         (`assert` or boolean conjunctions in `RESULT`) over computed variables (not literal-to-literal identities
         like `2 + 2 == 4` or `1 == 1`).
      4. Existence-only checks (`callable(...)`, `hasattr(...)`, `isinstance(...)`) are flagged and rejected for
         `externally_verified` promotion.
    """
    import ast as _ast

    if not isinstance(code, str) or not code.strip():
        return {"ok": False, "reason": "empty_code", "behavioral_checks": 0, "distinct_call_args": 0}
    try:
        tree = _ast.parse(code)
    except SyntaxError as exc:
        return {"ok": False, "reason": f"syntax_error:{exc.msg}", "behavioral_checks": 0, "distinct_call_args": 0}

    defined_fns: set[str] = {
        node.name for node in tree.body if isinstance(node, _ast.FunctionDef) and not node.name.startswith("_")
    }
    if not defined_fns:
        defined_fns = {node.name for node in tree.body if isinstance(node, _ast.FunctionDef)}

    existence_calls = {"callable", "hasattr", "isinstance", "issubclass", "type"}

    def _is_pure_literal_or_const(n: _ast.AST) -> bool:
        if isinstance(n, _ast.Constant):
            return True
        if isinstance(n, _ast.UnaryOp):
            return _is_pure_literal_or_const(n.operand)
        if isinstance(n, _ast.BinOp):
            return _is_pure_literal_or_const(n.left) and _is_pure_literal_or_const(n.right)
        if isinstance(n, (_ast.Tuple, _ast.List, _ast.Set)):
            return all(_is_pure_literal_or_const(elt) for elt in n.elts)
        return False

    def _is_existence_only_expr(n: _ast.AST) -> bool:
        for sub in _ast.walk(n):
            if isinstance(sub, _ast.Call) and isinstance(sub.func, _ast.Name) and sub.func.id in existence_calls:
                return True
        return False

    # Collect top-level / test-level calls to defined functions and non-tautological comparisons
    fn_call_arg_sigs: set[tuple[str, str]] = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
            fn_name = node.func.id
            if not defined_fns or fn_name in defined_fns:
                if fn_name not in existence_calls:
                    arg_dump = ",".join(_ast.dump(a) for a in node.args) + ";" + ",".join(
                        f"{k.arg}={_ast.dump(k.value)}" for k in node.keywords
                    )
                    fn_call_arg_sigs.add((fn_name, arg_dump))

    behavioral_checks = 0
    used_existence_only = False

    def _count_behavioral_expr(expr: _ast.AST) -> int:
        nonlocal used_existence_only
        if _is_existence_only_expr(expr):
            used_existence_only = True
            return 0
        if _is_pure_literal_or_const(expr):
            return 0
        if isinstance(expr, _ast.BoolOp) and isinstance(expr.op, _ast.And):
            return sum(_count_behavioral_expr(v) for v in expr.values)
        if isinstance(expr, _ast.Compare):
            if _is_pure_literal_or_const(expr.left) and all(_is_pure_literal_or_const(c) for c in expr.comparators):
                return 0
            return 1
        if isinstance(expr, _ast.Call):
            return 1
        if isinstance(expr, _ast.Name):
            return 1
        return 0

    result_assigned = False
    result_is_constant_true = False

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assert):
            behavioral_checks += _count_behavioral_expr(node.test)
        elif isinstance(node, _ast.Assign):
            for targ in node.targets:
                if isinstance(targ, _ast.Name) and targ.id in ("RESULT", "result"):
                    result_assigned = True
                    val = node.value
                    if isinstance(val, _ast.Constant) and val.value is True:
                        result_is_constant_true = True
                    elif isinstance(val, _ast.Dict):
                        for k_node, v_node in zip(val.keys, val.values):
                            if isinstance(k_node, _ast.Constant) and k_node.value == "passed":
                                if isinstance(v_node, _ast.Constant) and v_node.value is True:
                                    result_is_constant_true = True
                                else:
                                    behavioral_checks += _count_behavioral_expr(v_node)
                    else:
                        behavioral_checks += _count_behavioral_expr(val)

    if used_existence_only and behavioral_checks == 0:
        return {
            "ok": False,
            "reason": "existence_only_check_rejected",
            "behavioral_checks": 0,
            "distinct_call_args": len(fn_call_arg_sigs),
        }
    if defined_fns and len(fn_call_arg_sigs) < 2:
        return {
            "ok": False,
            "reason": f"insufficient_function_calls:need_at_least_2_distinct_inputs_got_{len(fn_call_arg_sigs)}",
            "behavioral_checks": behavioral_checks,
            "distinct_call_args": len(fn_call_arg_sigs),
        }
    if behavioral_checks < 2:
        reason = "tautological_constant_result" if result_is_constant_true or not result_assigned else f"insufficient_behavioral_checks:{behavioral_checks}"
        return {
            "ok": False,
            "reason": reason,
            "behavioral_checks": behavioral_checks,
            "distinct_call_args": len(fn_call_arg_sigs),
        }
    return {
        "ok": True,
        "reason": "verified_behavioral_probe",
        "behavioral_checks": behavioral_checks,
        "distinct_call_args": len(fn_call_arg_sigs),
    }


def extract_behavioral_probe_from_python_source(txt: str, fn_names: Sequence[str] = ()) -> str | None:
    """Extract a genuine behavioral verification probe from a Python source file `txt` ONLY if the source file
    contains real behavioral assertions (`assert ...`), doctests (`>>> fn(...)`), or a `test_*`/`_selftest` function.
    Never fabricates `callable(fn)` existence checks!
    """
    import ast as _ast
    import doctest as _doctest

    if not txt or len(txt) > 12000:
        return None
    try:
        tree = _ast.parse(txt)
    except SyntaxError:
        return None

    # 1. Check for doctest examples (`>>> ...`) in module or function docstrings
    try:
        dt_examples = _doctest.DocTestParser().get_examples(txt)
    except Exception:
        dt_examples = []
    valid_dt = [ex for ex in dt_examples if ex.source.strip() and ex.want.strip()]
    if len(valid_dt) >= 2:
        assert_lines = [txt.strip(), ""]
        for ex in valid_dt[:6]:
            src_expr = ex.source.strip()
            want_expr = ex.want.strip()
            if "\n" not in src_expr and "\n" not in want_expr:
                assert_lines.append(f"assert ({src_expr}) == ({want_expr})")
        assert_lines.append(f"RESULT = {{'passed': True, 'doctest_assertions': {len(valid_dt[:6])}}}")
        cand = "\n".join(assert_lines)
        if validate_behavioral_verification_probe(cand)["ok"]:
            return cand

    # 2. Check if the file already contains top-level or `__main__` `assert` statements or `test_*` functions
    has_asserts = any(isinstance(n, _ast.Assert) for n in _ast.walk(tree))
    test_fns = [
        n.name
        for n in tree.body
        if isinstance(n, _ast.FunctionDef) and (n.name.startswith("test_") or n.name.startswith("verify_") or n.name == "_selftest")
    ]
    if has_asserts or test_fns:
        lines = [txt.strip(), ""]
        for tfn in test_fns:
            lines.append(f"{tfn}()")
        lines.append("RESULT = {'passed': True}")
        cand = "\n".join(lines)
        if validate_behavioral_verification_probe(cand)["ok"]:
            return cand

    # 3. Synthesize a multi-input differential / boundary behavioral probe ONLY when a function has pure numeric/collection/string
    # parameters and deterministic return expressions that can be tested on >=2 distinct inputs and checked for non-constancy.
    for node in tree.body:
        if not isinstance(node, _ast.FunctionDef) or node.name.startswith("_"):
            continue
        if fn_names and node.name not in fn_names:
            continue
        arg_names = [a.arg for a in node.args.args]
        # Look for explicit inline comment assertions or docstring equations (`# test: fn(...) == ...`)
        m_tests = re.findall(rf"\b({re.escape(node.name)}\([^)\n]+\)\s*==\s*[^\n#]+)", txt)
        if len(m_tests) >= 2:
            lines = [txt.strip(), ""]
            for mt in m_tests[:6]:
                lines.append(f"assert {mt.strip()}")
            lines.append(f"RESULT = {{'passed': True, 'function': {node.name!r}}}")
            cand = "\n".join(lines)
            if validate_behavioral_verification_probe(cand)["ok"]:
                return cand
        _ = arg_names

    return build_unannotated_function_candidate_spec(txt, fn_names)


def extract_ast_rule_summary(txt: str, fn_names: Sequence[str] = ()) -> str:
    """Extract a concise, human-and-model-readable summary of top-level function signatures, branch conditions,
    and return formulas from Python AST `txt`."""
    import ast as _ast

    try:
        tree = _ast.parse(txt or "")
    except SyntaxError:
        return ""
    summaries: list[str] = []
    for node in tree.body:
        if not isinstance(node, _ast.FunctionDef) or node.name.startswith("_"):
            continue
        if fn_names and node.name not in fn_names:
            continue
        sig_parts = []
        n_args = len(node.args.args)
        n_defs = len(node.args.defaults)
        for idx_a, a in enumerate(node.args.args):
            p_str = a.arg
            if idx_a >= n_args - n_defs:
                try:
                    p_str += "=" + _ast.unparse(node.args.defaults[idx_a - (n_args - n_defs)])
                except Exception:
                    pass
            sig_parts.append(p_str)
        sig = f"{node.name}({', '.join(sig_parts)})"
        clauses: list[str] = []
        for stmt in node.body:
            if isinstance(stmt, _ast.Expr) and isinstance(stmt.value, _ast.Constant) and isinstance(stmt.value.value, str):
                continue
            try:
                unparsed = " ".join(_ast.unparse(stmt).split())
                if len(unparsed) <= 160:
                    clauses.append(unparsed)
            except Exception:
                continue
        summaries.append(f"{sig}: {'; '.join(clauses[:4])}")
    return " | ".join(summaries[:2])[:420]


def build_unannotated_function_candidate_spec(txt: str, fn_names: Sequence[str] = ()) -> str | None:
    """Step 1 (Autonomous Behavioral Probe Synthesis for Unannotated Workspace Modules):
    When a clean workspace `.py` file has zero pre-written `assert` statements or doctests, inspect the AST of its
    pure functions (parameter annotations, defaults, comparison thresholds in `If` branches, and operators) to build
    a candidate multi-regime boundary input specification (`# __CLOVER_UNANNOTATED_PROBE_SPEC__: ...`).
    This spec is NOT verified yet; it is executed and checked for determinism + non-trivial output variation inside
    `IsolatedPythonSandbox` when `sandbox_counterexample_probe` runs.
    """
    import ast as _ast
    import isolated_sandbox as _SBX

    if not txt or len(txt) > 12000:
        return None
    ast_ok, _ = _SBX.validate_sandbox_ast(txt)
    if not ast_ok:
        return None
    try:
        tree = _ast.parse(txt)
    except SyntaxError:
        return None

    for node in tree.body:
        if not isinstance(node, _ast.FunctionDef) or node.name.startswith("_"):
            continue
        if fn_names and node.name not in fn_names:
            continue
        args = node.args.args
        if not args or len(args) > 6:
            continue

        # Collect numeric comparison constants inside the function body to probe around exact branch boundaries
        num_thresholds: list[float | int] = []
        bool_params: set[str] = set()
        str_params: set[str] = set()
        dict_params: set[str] = set()

        for sub in _ast.walk(node):
            if isinstance(sub, _ast.Compare):
                for comp_node in [sub.left] + list(sub.comparators):
                    if isinstance(comp_node, _ast.Constant) and isinstance(comp_node.value, (int, float)) and not isinstance(comp_node.value, bool):
                        num_thresholds.append(comp_node.value)
                    elif isinstance(comp_node, _ast.UnaryOp) and isinstance(comp_node.op, _ast.USub) and isinstance(comp_node.operand, _ast.Constant):
                        if isinstance(comp_node.operand.value, (int, float)):
                            num_thresholds.append(-comp_node.operand.value)
            elif isinstance(sub, _ast.IfExp) and isinstance(sub.test, _ast.Name):
                bool_params.add(sub.test.id)
            elif isinstance(sub, _ast.If) and isinstance(sub.test, _ast.Name):
                bool_params.add(sub.test.id)
            elif isinstance(sub, _ast.Call) and isinstance(sub.func, _ast.Attribute):
                if sub.func.attr in ("startswith", "endswith", "split", "strip", "lower", "upper") and isinstance(sub.func.value, _ast.Name):
                    str_params.add(sub.func.value.id)
                elif sub.func.attr in ("get", "items", "keys", "values") and isinstance(sub.func.value, _ast.Name):
                    dict_params.add(sub.func.value.id)
            elif isinstance(sub, _ast.Subscript) and isinstance(sub.value, _ast.Name):
                dict_params.add(sub.value.id)

        n_args = len(args)
        n_defs = len(node.args.defaults)
        param_kinds: list[str] = []
        for idx_a, a in enumerate(args):
            ann = ""
            if a.annotation is not None:
                try:
                    ann = _ast.unparse(a.annotation).lower()
                except Exception:
                    ann = ""
            def_val = None
            if idx_a >= n_args - n_defs:
                d_node = node.args.defaults[idx_a - (n_args - n_defs)]
                if isinstance(d_node, _ast.Constant):
                    def_val = d_node.value

            pname = a.arg.lower()
            if "bool" in ann or isinstance(def_val, bool) or a.arg in bool_params or pname.startswith(("is_", "has_", "allow_", "partition")):
                param_kinds.append("bool")
            elif "int" in ann and "float" not in ann:
                param_kinds.append("int")
            elif "dict" in ann or "mapping" in ann or (a.arg in dict_params and "list" not in ann and "seq" not in ann) or "route" in pname or "table" in pname:
                param_kinds.append("dict")
            elif "str" in ann or isinstance(def_val, str) or a.arg in str_params or pname in ("ip", "path", "key", "text", "name"):
                param_kinds.append("str")
            elif "float" in ann or isinstance(def_val, float) or any(w in pname for w in ("rate", "vmax", "km", "substrate", "inhibitor", "ratio", "alpha", "temp", "conc")):
                param_kinds.append("float")
            elif "list" in ann or "seq" in ann or "items" in pname or "values" in pname:
                param_kinds.append("list")
            else:
                param_kinds.append("int")

        bound_val = num_thresholds[0] if num_thresholds else 0

        def _make_arg_val(kind: str, idx_p: int, regime: str) -> Any:
            if kind == "bool":
                return False if regime in ("normal_b", "boundary_alt") else True
            if kind == "dict":
                return {} if regime == "boundary_zero" else {"10.": "corp_a", "10.20.": "subnet_b"}
            if kind == "str":
                if regime == "normal_a":
                    return "10.20.1.5"
                if regime == "normal_b":
                    return "192.168.1.1"
                return ""
            if kind == "list":
                if regime == "boundary_zero":
                    return []
                return [2, 4, 6] if regime == "normal_a" else [10, 20, 30]
            if kind == "float":
                if regime == "normal_a":
                    return 10.0 if idx_p == 0 else (100.0 if idx_p == 1 else (2.0 if idx_p == 2 else 1.0))
                if regime == "normal_b":
                    return 20.0 if idx_p == 0 else (100.0 if idx_p == 1 else (2.0 if idx_p == 2 else 0.0))
                if regime == "boundary_zero":
                    return float(bound_val) if idx_p == 0 else (100.0 if idx_p == 1 else (2.0 if idx_p == 2 else 1.0))
                return 5.0 if idx_p == 0 else (80.0 if idx_p == 1 else (4.0 if idx_p == 2 else 3.0))
            # int
            if regime == "normal_a":
                return 512 if idx_p == 0 else (4 if idx_p == 1 else 2)
            if regime == "normal_b":
                return 256 if idx_p == 0 else (8 if idx_p == 1 else 4)
            if regime == "boundary_zero":
                return int(bound_val) if idx_p == 0 else (4 if idx_p == 1 else 1)
            return -5 if idx_p == 0 else (0 if idx_p == 1 else 0)

        candidate_inputs = [
            [_make_arg_val(k, i, reg_name) for i, k in enumerate(param_kinds)]
            for reg_name in ("normal_a", "normal_b", "boundary_zero", "boundary_alt")
        ]
        spec = {
            "fn_name": node.name,
            "param_names": [a.arg for a in args],
            "param_kinds": param_kinds,
            "candidate_inputs": candidate_inputs,
            "ast_rule_summary": extract_ast_rule_summary(txt, [node.name]),
        }
        return txt.strip() + "\n\n# __CLOVER_UNANNOTATED_PROBE_SPEC__: " + json.dumps(spec)

    return None


def synthesize_and_verify_behavioral_probe_in_sandbox(
    probe_code: str,
    sandbox_runner: Callable[[str], tuple[bool, Any, str]],
) -> dict[str, Any]:
    """Execute a `# __CLOVER_UNANNOTATED_PROBE_SPEC__:` candidate in `IsolatedPythonSandbox` to:
      1. Test determinism across repeated calls `fn(*args) == fn(*args)` (rejecting random/noisy distractors),
      2. Test non-trivial behavioral variation across normal vs. boundary inputs (rejecting constant stubs), and
      3. Synthesize an explicit, AST-verified multi-input `assert fn(*args) == expected` script + human/model summary.
    """
    marker = "# __CLOVER_UNANNOTATED_PROBE_SPEC__:"
    if marker not in (probe_code or ""):
        return {"ok": True, "synthesized_code": probe_code, "was_unannotated": False}

    clean_src, _, spec_raw = probe_code.partition(marker)
    clean_src = clean_src.strip()
    try:
        spec = json.loads(spec_raw.strip().splitlines()[0])
    except Exception as exc:
        return {"ok": False, "reason": f"invalid_probe_spec:{exc}", "was_unannotated": True}

    fn_name = str(spec["fn_name"])
    cand_inputs = list(spec.get("candidate_inputs") or [])
    ast_rule = str(spec.get("ast_rule_summary") or "")

    harness = (
        f"{clean_src}\n\n"
        f"_inputs = {cand_inputs!r}\n"
        "_records = []\n"
        "_deterministic = True\n"
        "for _args in _inputs:\n"
        "    try:\n"
        f"        _r1 = {fn_name}(*_args)\n"
        f"        _r2 = {fn_name}(*_args)\n"
        "        if _r1 != _r2:\n"
        "            _deterministic = False\n"
        "        if isinstance(_r1, float):\n"
        "            _r1 = round(_r1, 6)\n"
        "        _records.append({'args': _args, 'ok': True, 'out': _r1})\n"
        "    except Exception as _e:\n"
        "        _records.append({'args': _args, 'ok': False, 'err': str(_e)})\n"
        "RESULT = {'deterministic': _deterministic, 'records': _records}\n"
    )
    ok_h, val_h, err_h = sandbox_runner(harness)
    if not ok_h or not isinstance(val_h, dict):
        return {"ok": False, "reason": f"sandbox_discovery_failed:{err_h[:120]}", "was_unannotated": True}

    if not val_h.get("deterministic", False):
        return {
            "ok": False,
            "is_noise": True,
            "reason": "non_deterministic_noise_function_rejected",
            "was_unannotated": True,
        }

    records = [r for r in (val_h.get("records") or []) if isinstance(r, dict) and r.get("ok")]
    if len(records) < 2:
        return {"ok": False, "reason": "insufficient_valid_executions", "was_unannotated": True}

    unique_outs = {json.dumps(r["out"], sort_keys=True) for r in records}
    if len(unique_outs) < 2:
        return {
            "ok": False,
            "is_trivial": True,
            "reason": "trivial_constant_function_rejected",
            "was_unannotated": True,
        }

    assert_lines = [clean_src, ""]
    case_descriptions: list[str] = []
    for r in records[:4]:
        args_repr = ", ".join(repr(a) for a in r["args"])
        out_val = r["out"]
        if isinstance(out_val, float):
            assert_lines.append(f"assert round({fn_name}({args_repr}), 6) == {out_val!r}")
        else:
            assert_lines.append(f"assert {fn_name}({args_repr}) == {out_val!r}")
        case_descriptions.append(f"{fn_name}({args_repr}) == {out_val!r}")

    assert_lines.append(f"RESULT = {{'passed': True, 'function': {fn_name!r}, 'verified_cases': {len(records[:4])}}}")
    synth_code = "\n".join(assert_lines)
    integrity = validate_behavioral_verification_probe(synth_code)
    if not integrity["ok"]:
        return {"ok": False, "reason": integrity["reason"], "was_unannotated": True}

    verified_summary = (
        f"Verified workspace function {ast_rule} | "
        f"Verified behavioral & boundary cases: {'; '.join(case_descriptions)}"
    )[:480]
    return {
        "ok": True,
        "was_unannotated": True,
        "synthesized_code": synth_code,
        "verified_summary": verified_summary,
        "verified_cases": case_descriptions,
        "probe_integrity": integrity,
    }

STOPWORDS: frozenset[str] = frozenset(
    "a about above after again against all am an and any are as at be because been before being below between both "
    "but by can could did do does doing down during each few for from further had has have having he her here hers "
    "herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on "
    "once only or other our ours ourselves out over own please same she should so some such than that the their "
    "theirs them themselves then there these they this those through to too under until up very was we were what "
    "when where which while who whom why will with would you your yours yourself yourselves clover question answer "
    "use using used verify check explain compute calculate across behave true false".split()
)


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_emergent_terms(text: str, max_terms: int = 12) -> list[str]:
    """Extract content terms from `text` without any hardcoded topic list."""
    words = re.findall(r"[a-z][a-z0-9_\-]{2,}", (text or "").lower())
    filtered = [w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w for w in words if w not in STOPWORDS]
    counts = collections.Counter(filtered)
    return [w for w, _ in counts.most_common(max_terms)]


def encode_text_to_64d(text: str, latent_cue_64: np.ndarray | Sequence[float] | None = None) -> np.ndarray:
    """Project `text` (and optional `latent_cue_64` from `CloverJEPARuntime`) into a unit-normalized 64-d vector."""
    v = np.zeros(D_LATENT, dtype=np.float64)
    terms = extract_emergent_terms(text, max_terms=32)
    if not terms:
        terms = ["empty"]
    for t in terms:
        digest = hashlib.sha256(t.encode("utf-8")).digest()
        for i in range(0, 32, 2):
            dim = digest[i] % D_LATENT
            sign = 1.0 if (digest[i + 1] & 1) == 0 else -1.0
            v[dim] += sign
    norm = float(np.linalg.norm(v))
    if norm > 1e-9:
        v /= norm
    if latent_cue_64 is not None:
        cue = np.asarray(latent_cue_64, dtype=np.float64).reshape(-1)
        if cue.size == D_LATENT and np.all(np.isfinite(cue)):
            c_norm = float(np.linalg.norm(cue))
            if c_norm > 1e-9:
                v = 0.55 * v + 0.45 * (cue / c_norm)
                v /= max(1e-9, float(np.linalg.norm(v)))
    return v


# ---------------------------------------------------------------------------
# 1. Data Structures for Persistent Curiosity State
# ---------------------------------------------------------------------------

@dataclass
class StrategyStats:
    """Paper 11 (`SGIM-ACTS`) & Paper 06 (`VoI`): per-(region, strategy) empirical progress and reliability."""
    strategy: str
    attempts: int = 0
    progress_window: list[float] = field(default_factory=list)
    error_window: list[float] = field(default_factory=list)
    views_discovered: int = 0
    last_used_step: int = 0

    @property
    def empirical_lp(self) -> float:
        if not self.progress_window:
            return 0.08  # initial optimistic exploration prior for untried strategy
        if self.progress_window[-1] <= 1e-4:
            # Most recent attempt of this strategy stalled (yielded 0 new views / 0 error drop)
            return 0.01 * float(np.mean(self.progress_window[-4:]))
        return float(np.mean(self.progress_window[-6:]))

    @property
    def outcome_variance(self) -> float:
        if len(self.error_window) < 2:
            return 0.02
        return float(np.var(self.error_window[-8:]))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyStats":
        return cls(
            strategy=str(d.get("strategy") or "memory_graph_recall"),
            attempts=int(d.get("attempts") or 0),
            progress_window=[float(x) for x in (d.get("progress_window") or [])][-16:],
            error_window=[float(x) for x in (d.get("error_window") or [])][-16:],
            views_discovered=int(d.get("views_discovered") or 0),
            last_used_step=int(d.get("last_used_step") or 0),
        )


@dataclass
class CuriosityRegion:
    """Cortical Microcolumn (`MR + metaMR`, Paper 05) & `R-IAC` Tree Node (Paper 01, 04, 10, 12, 13)."""
    region_id: str
    parent_id: str | None = None
    left_child_id: str | None = None
    right_child_id: str | None = None
    split_dim: int | None = None
    split_threshold: float | None = None
    centroid_64: list[float] = field(default_factory=lambda: [0.0] * D_LATENT)
    emergent_keywords: list[str] = field(default_factory=list)
    exemplar_vectors: list[list[float]] = field(default_factory=list)
    exemplar_outcomes: list[float] = field(default_factory=list)
    pred_error_window: list[float] = field(default_factory=list)
    comp_error_window: list[float] = field(default_factory=list)
    # Paper 10 (Schmidhuber 1991): Confidence module C(R_n) expected error y_C and signed shift vector E[Delta z]
    expected_error_yc: float = 0.35
    prev_expected_error_yc: float = 0.35
    signed_shift_ema_64: list[float] = field(default_factory=lambda: [0.0] * D_LATENT)
    # Paper 04 & 13: Controllability R^2(Action -> Outcome)
    controllability_score: float = 0.50
    controllable_Try_count: int = 0
    uncontrollable_Try_count: int = 0
    # Tonic progress signals
    tonic_lpm: float = 0.05
    tonic_cpm: float = 0.05
    dot_yc_reliability: float = 0.0
    abs_lp_vigilance: float = 0.0
    phasic_surprise_sm: float = 0.0
    habituation_trace: float = 0.0
    state_curiosity: float = 0.05
    visits: int = 0
    last_visited_step: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @property
    def is_leaf(self) -> bool:
        return self.left_child_id is None and self.right_child_id is None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Keep exemplar buffers bounded for compact persistence
        d["exemplar_vectors"] = self.exemplar_vectors[-40:]
        d["exemplar_outcomes"] = self.exemplar_outcomes[-40:]
        d["pred_error_window"] = self.pred_error_window[-32:]
        d["comp_error_window"] = self.comp_error_window[-32:]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CuriosityRegion":
        return cls(
            region_id=str(d["region_id"]),
            parent_id=d.get("parent_id"),
            left_child_id=d.get("left_child_id"),
            right_child_id=d.get("right_child_id"),
            split_dim=int(d["split_dim"]) if d.get("split_dim") is not None else None,
            split_threshold=float(d["split_threshold"]) if d.get("split_threshold") is not None else None,
            centroid_64=[float(x) for x in (d.get("centroid_64") or [0.0] * D_LATENT)],
            emergent_keywords=list(d.get("emergent_keywords") or []),
            exemplar_vectors=[[float(v) for v in row] for row in (d.get("exemplar_vectors") or [])],
            exemplar_outcomes=[float(x) for x in (d.get("exemplar_outcomes") or [])],
            pred_error_window=[float(x) for x in (d.get("pred_error_window") or [])],
            comp_error_window=[float(x) for x in (d.get("comp_error_window") or [])],
            expected_error_yc=float(d.get("expected_error_yc", 0.35)),
            prev_expected_error_yc=float(d.get("prev_expected_error_yc", 0.35)),
            signed_shift_ema_64=[float(x) for x in (d.get("signed_shift_ema_64") or [0.0] * D_LATENT)],
            controllability_score=float(d.get("controllability_score", 0.50)),
            controllable_Try_count=int(d.get("controllable_Try_count", 0)),
            uncontrollable_Try_count=int(d.get("uncontrollable_Try_count", 0)),
            tonic_lpm=float(d.get("tonic_lpm", 0.05)),
            tonic_cpm=float(d.get("tonic_cpm", 0.05)),
            dot_yc_reliability=float(d.get("dot_yc_reliability", 0.0)),
            abs_lp_vigilance=float(d.get("abs_lp_vigilance", 0.0)),
            phasic_surprise_sm=float(d.get("phasic_surprise_sm", 0.0)),
            habituation_trace=float(d.get("habituation_trace", 0.0)),
            state_curiosity=float(d.get("state_curiosity", 0.05)),
            visits=int(d.get("visits", 0)),
            last_visited_step=int(d.get("last_visited_step", 0)),
            created_at=str(d.get("created_at") or utc_now_iso()),
            updated_at=str(d.get("updated_at") or utc_now_iso()),
        )


@dataclass
class UnresolvedQuestionNode:
    """Papers 06, 07, 08, 13: Dynamic `ZPDES` epistemic question/hypothesis node (never pre-scripted)."""
    question_id: str
    region_id: str
    summary: str
    origin_kind: str  # "turn_discrepancy" | "memory_contradiction" | "unverified_hypothesis" | "competence_regression" | "child_frontier"
    memory_entry_id: str | None = None
    parent_question_id: str | None = None
    prerequisite_ids: list[str] = field(default_factory=list)
    lateral_modality_ids: list[str] = field(default_factory=list)
    zpd_status: str = "active_zpd"  # "active_zpd" | "stalled_deactivated" | "prerequisite_pending" | "resolved_verified"
    attempts: int = 0
    error_history: list[float] = field(default_factory=list)
    diagnosticity_score: float = 0.50  # Paper 07 neutral validity / expected uncertainty reduction
    views_covered: list[str] = field(default_factory=list)  # Paper 11 multi-view accumulation (up to 7 dimensions)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @property
    def recent_lp(self) -> float:
        if len(self.error_history) < 2:
            return 0.06
        half = max(1, len(self.error_history) // 2)
        pst = float(np.mean(self.error_history[:half]))
        rec = float(np.mean(self.error_history[half:]))
        return pst - rec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UnresolvedQuestionNode":
        return cls(
            question_id=str(d["question_id"]),
            region_id=str(d.get("region_id") or "R0"),
            summary=str(d.get("summary") or ""),
            origin_kind=str(d.get("origin_kind") or "turn_discrepancy"),
            memory_entry_id=d.get("memory_entry_id"),
            parent_question_id=d.get("parent_question_id"),
            prerequisite_ids=list(d.get("prerequisite_ids") or []),
            lateral_modality_ids=list(d.get("lateral_modality_ids") or []),
            zpd_status=str(d.get("zpd_status") or "active_zpd"),
            attempts=int(d.get("attempts") or 0),
            error_history=[float(x) for x in (d.get("error_history") or [])][-16:],
            diagnosticity_score=float(d.get("diagnosticity_score", 0.50)),
            views_covered=list(d.get("views_covered") or []),
            created_at=str(d.get("created_at") or utc_now_iso()),
            updated_at=str(d.get("updated_at") or utc_now_iso()),
        )


# ---------------------------------------------------------------------------
# 2. Core Mathematical Primitives (Criterion C2, LPM+CPM, VoI, Controllability)
# ---------------------------------------------------------------------------

def find_criterion_c2_split(
    vectors: Sequence[Sequence[float]],
    outcomes: Sequence[float],
    candidate_dims: Sequence[int] | None = None,
) -> tuple[int, float, float] | None:
    """Paper 01 (Oudeyer, Kaplan & Hafner 2007, p. 270 Criterion C2):
    Find split dimension `j*` and threshold `v*` that minimize the sample-weighted sum of child outcome variances:
      Gamma_n(j, v) = (|L| / N) * Var(outcomes_L) + (|R| / N) * Var(outcomes_R).
    Returns `(best_dim, best_threshold, variance_reduction_ratio)` or `None` if no valid split reduces variance.
    """
    n = len(vectors)
    if n < 8 or len(outcomes) != n:
        return None
    X = np.asarray(vectors, dtype=np.float64)
    Y = np.asarray(outcomes, dtype=np.float64)
    parent_var = float(np.var(Y))
    if parent_var < 1e-6:
        return None

    dims = list(candidate_dims) if candidate_dims is not None else list(range(min(D_LATENT, X.shape[1])))
    best_dim = -1
    best_thresh = 0.0
    best_weighted_var = parent_var

    for j in dims:
        col = X[:, j]
        uniq = np.unique(np.round(col, 6))
        if len(uniq) < 2:
            continue
        midpoints = 0.5 * (uniq[:-1] + uniq[1:])
        if len(midpoints) > 7:
            idx = np.linspace(0, len(midpoints) - 1, 7).astype(int)
            midpoints = midpoints[idx]
        for v in midpoints:
            v_flt = float(v)
            mask_l = col <= v_flt
            n_l = int(np.sum(mask_l))
            n_r = n - n_l
            if n_l < 3 or n_r < 3:
                continue
            w_var = (n_l / n) * float(np.var(Y[mask_l])) + (n_r / n) * float(np.var(Y[~mask_l]))
            if w_var < best_weighted_var:
                best_weighted_var = w_var
                best_dim = int(j)
                best_thresh = v_flt

    if best_dim < 0:
        return None
    reduction_ratio = (parent_var - best_weighted_var) / max(1e-9, parent_var)
    if reduction_ratio < 0.05:
        return None
    return best_dim, best_thresh, round(float(reduction_ratio), 4)


def compute_windowed_lp_and_vigilance(
    errors: Sequence[float],
    window_half: int = 6,
    noise_discount: float = 0.85,
) -> tuple[float, float, float]:
    """Compute:
      - `lp_positive`: Variance-discounted positive Learning Progress `<e(t-theta)> - <e(t)> - beta * SE(e_rec)`
        (Papers 01, 04, 05),
      - `abs_lp_vigilance`: Competence-drop vigilance `|<e(t-theta)> - <e(t)>|` when a previously mastered region
        (`<e(t-theta)> <= 0.20`) regresses (`<e(t)> - <e(t-theta)> >= 0.18`, Paper 12 Eq. 3 & Paper 13),
      - `plasticity_factor`: Non-saturating plasticity weight `min_floor + 4*p*(1-p)` (Paper 02 E3A fix: never 0 at extremes).
    """
    if len(errors) < 2:
        return 0.05, 0.0, 0.65
    arr = np.asarray(errors[-2 * window_half :], dtype=np.float64)
    mid = max(1, len(arr) // 2)
    pst = arr[:mid]
    rec = arr[mid:]
    mean_pst = float(np.mean(pst))
    mean_rec = float(np.mean(rec))
    std_rec = float(np.std(rec))

    raw_drop = mean_pst - mean_rec
    # Discount erratic high-variance noise and non-monotonic upward oscillations (Papers 01, 04, 05, 10)
    diffs = np.diff(arr)
    up_jumps = float(np.mean(np.maximum(0.0, diffs))) if len(diffs) > 0 else 0.0
    se_penalty = 0.0
    if mean_rec > 0.25:
        se_penalty += noise_discount * std_rec + 1.5 * up_jumps
    if mean_rec > 0.50 and float(np.min(rec)) > 0.45:
        se_penalty += 0.50 * (mean_rec - 0.45)
    lp_pos = max(0.0, raw_drop - se_penalty)

    # Paper 12 Eq. (3) & Priority 4: |LP| vigilance when competence in a previously mastered region drops
    tail_err = max(mean_rec, float(np.mean(arr[-2:])), float(arr[-1]))
    if (mean_pst <= 0.20 or float(np.min(pst)) <= 0.12) and (tail_err - mean_pst) >= 0.18 and float(arr[-1]) >= 0.25:
        vigilance = max(abs(raw_drop), tail_err - mean_pst)
    else:
        vigilance = 0.0

    # Paper 02 E3A: Plasticity factor bounded away from zero (`>= 0.15`) so severe errors are never ignored
    p_clamped = min(0.98, max(0.02, mean_rec))
    plasticity = max(0.15, 4.0 * p_clamped * (1.0 - p_clamped))
    return round(lp_pos, 5), round(vigilance, 5), round(plasticity, 4)


def derive_real_turn_outcome(
    user_request: str,
    model_prediction: str,
    *,
    layers: Sequence[dict[str, Any]] = (),
    tool_calls: Sequence[dict[str, Any]] = (),
    jepa_plan: dict[str, Any] | None = None,
    user_feedback: str | None = None,
    correction_text: str | None = None,
    entry_tally: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Priority 1: Ground feedback in real outcomes rather than assigning fixed competence values to ordinary turns.
    Extracts:
      - `query_text`: the actual `user_request` (plus `correction_text` when present), never just `reply[:300]`.
      - `strategy_used`: mapped from the actual tools/layers that executed in the turn.
      - `pred_error` (`float | None`): measured from real C2/JEPA latent discrepancy, cycle first-attempt findings,
        or ontology grounding mismatch (`None` if no predictive signal was measured).
      - `pre_repair_comp_error` (`float | None`): when Cycle 1 failed real checks before repair, records the initial
        measured failure rate so within-turn competence progress (`pre -> post`) is grounded in reality.
      - `comp_error` (`float | None`): measured from real verification checks (`compiler`, `cycle`, `grounding`),
        real tool execution outcomes (`ran` & `worked`), or real `user_feedback` (`continued`, `helpful`,
        `said_not_right`, `tried_again`). Returns `None` on ordinary unverified turns so no synthetic competence
        score is ever fabricated.
    """
    req_clean = " ".join(str(user_request or "").split()).strip()
    pred_clean = " ".join(str(model_prediction or "").split()).strip()
    corr_clean = " ".join(str(correction_text or "").split()).strip()
    query_text = req_clean or pred_clean[:240]

    # 1. Determine which epistemic strategy actually ran
    ran_tools = [tc for tc in (tool_calls or ()) if isinstance(tc, dict) and tc.get("ran")]
    ran_names = [str(tc.get("tool") or tc.get("name") or "") for tc in ran_tools]
    layer_names = {str(lay.get("layer") or "") for lay in (layers or ()) if isinstance(lay, dict)}
    sem_lay = next((lay for lay in (layers or ()) if isinstance(lay, dict) and lay.get("layer") == "semantics"), {})

    if any(t in ("run_python", "python", "calculate", "convert_units", "analyze_table", "chart_table") for t in ran_names):
        strategy_used = "sandbox_counterexample_probe"
    elif any(t in ("read_file", "list_directory", "search_files", "file_info", "git_inspect", "read_document") for t in ran_names) or "workspace" in layer_names:
        strategy_used = "workspace_code_inspect"
    elif any(t in ("web_search", "fetch_url") for t in ran_names):
        strategy_used = "web_primary_research"
    elif sem_lay.get("discernment") == "ask":
        strategy_used = "human_clarification_ask"
    else:
        strategy_used = "memory_graph_recall"

    # 2. Measure real prediction discrepancy (`pred_error`)
    pred_signals: list[float] = []
    latent_before_64 = None
    latent_after_64 = None
    if isinstance(jepa_plan, dict) and jepa_plan.get("enabled"):
        e_bef = float(jepa_plan.get("energy_before") or 0.0)
        e_aft = float(jepa_plan.get("energy_after") or e_bef)
        if e_bef > 1e-6:
            pred_signals.append(float(np.clip(e_aft / max(0.25, e_bef), 0.0, 1.0)))
        if jepa_plan.get("z_init_64") is not None:
            latent_before_64 = jepa_plan.get("z_init_64")
        if jepa_plan.get("z_planned_64") is not None:
            latent_after_64 = jepa_plan.get("z_planned_64")

    cyc_lay = next((lay for lay in (layers or ()) if isinstance(lay, dict) and lay.get("layer") == "cycle"), {})
    comp_lay = next((lay for lay in (layers or ()) if isinstance(lay, dict) and lay.get("layer") == "compiler" and lay.get("checked")), None)
    gnd_lay = next((lay for lay in (layers or ()) if isinstance(lay, dict) and lay.get("layer") == "ontology_grounding"), None)
    out_lay = next((lay for lay in (layers or ()) if isinstance(lay, dict) and lay.get("layer") == "outcome"), None)

    first_findings = list(cyc_lay.get("first_findings") or [])
    final_findings = list(cyc_lay.get("final_findings") or [])
    if first_findings:
        pred_signals.append(min(1.0, len(first_findings) / (len(first_findings) + 1.0)))
    elif cyc_lay.get("action") == "redone":
        pred_signals.append(0.65)
    if gnd_lay and gnd_lay.get("changed"):
        pred_signals.append(0.40)

    # 3. Measure real competence/verification error (`comp_error` & `pre_repair_comp_error`)
    comp_signals: list[float] = []
    pre_repair_comp_error: float | None = None
    view_added: str | None = None
    unresolved_summary: str | None = None

    if ran_tools:
        worked_cnt = sum(
            1 for tc in ran_tools
            if (bool(tc.get("worked")) if "worked" in tc else not str(tc.get("result") or "").lstrip().startswith("Error"))
        )
        tool_fail_rate = 1.0 - (worked_cnt / len(ran_tools))
        comp_signals.append(tool_fail_rate)
        if worked_cnt > 0:
            view_added = "supporting_evidence" if strategy_used == "sandbox_counterexample_probe" else "what_it_consists_of"

    if comp_lay is not None:
        diags = list(comp_lay.get("diagnostics") or [])
        if comp_lay.get("ok"):
            comp_signals.append(0.0)
            view_added = view_added or "where_it_applies"
        else:
            c_err = min(1.0, 0.50 + 0.15 * max(1, len(diags)))
            comp_signals.append(c_err)
            unresolved_summary = f"Compiler diagnostic on '{req_clean[:120]}': {diags[0] if diags else 'syntax/type error'}"

    if out_lay is not None and out_lay.get("checks") and out_lay.get("passed"):
        comp_signals.append(0.0)

    if cyc_lay:
        act = cyc_lay.get("action")
        if first_findings and not final_findings and act in ("corrected", "compiler_repaired"):
            # Cycle 1 failed with `first_findings` and Cycle k repaired it to 0 findings!
            pre_repair_comp_error = round(min(1.0, len(first_findings) / (len(first_findings) + 1.0)), 4)
            comp_signals.append(0.0)
            view_added = "where_it_breaks"
        elif final_findings or act == "unconfirmed":
            f_cnt = max(1, len(final_findings) or len(first_findings))
            c_err = min(1.0, f_cnt / (f_cnt + 1.0))
            comp_signals.append(c_err)
            unresolved_summary = f"Unresolved verification finding on '{req_clean[:120]}': {(final_findings or first_findings or ['unconfirmed'])[0]}"
        elif act == "redone":
            # User triggered retry cycle
            pre_repair_comp_error = 0.80
            comp_signals.append(0.15 if cyc_lay.get("lesson") else 0.35)
            unresolved_summary = str(cyc_lay.get("lesson") or f"User requested retry on '{req_clean[:120]}'")

    if user_feedback is not None:
        fb = str(user_feedback).strip().lower()
        if entry_tally is not None:
            w_cnt, d_cnt = int(entry_tally[0]), int(entry_tally[1])
            tot_cnt = max(1, w_cnt + d_cnt)
            emp_fail_rate = d_cnt / tot_cnt
        else:
            emp_fail_rate = 0.0 if fb in ("continued", "helpful", "worked") else 0.85

        if fb in ("said_not_right", "tried_again", "did_not_work", "rejected"):
            fb_err = max(0.75, emp_fail_rate)
            comp_signals.append(fb_err)
            pred_signals.append(max(0.65, emp_fail_rate))
            unresolved_summary = corr_clean or f"User invalidated previous answer on '{req_clean[:140]}'"
        elif fb in ("helpful", "worked"):
            comp_signals.append(min(0.15, emp_fail_rate))
            pred_signals.append(min(0.10, emp_fail_rate))
        elif fb == "continued":
            comp_signals.append(min(0.20, emp_fail_rate))

    comp_error = round(float(np.mean(comp_signals)), 4) if comp_signals else None
    if pred_signals:
        pred_error: float | None = round(float(np.mean(pred_signals)), 4)
    elif comp_error is not None:
        pred_error = comp_error
    else:
        pred_error = None

    if latent_before_64 is None and req_clean:
        latent_before_64 = encode_text_to_64d(req_clean)
    if latent_after_64 is None and pred_clean:
        latent_after_64 = encode_text_to_64d(pred_clean)

    return {
        "query_text": query_text,
        "user_request": req_clean,
        "model_prediction": pred_clean[:300],
        "correction_text": corr_clean,
        "strategy_used": strategy_used,
        "pred_error": pred_error,
        "pre_repair_comp_error": pre_repair_comp_error,
        "comp_error": comp_error,
        "verified_in_turn": comp_error is not None,
        "controllable": True,
        "latent_before_64": latent_before_64,
        "latent_after_64": latent_after_64,
        "unresolved_summary": unresolved_summary,
        "view_dimension_added": view_added,
    }


def compute_pomdp_voi(
    expected_error_yc: float,
    strategy_gain_rate: float,
    observation_cost_ro: float,
) -> dict[str, Any]:
    """Paper 06 (Gottlieb et al. 2013, Fig. 1A):
    - Prior certainty `p = 1.0 - min(0.5, expected_error_yc)` in `[0.50, 1.00]`.
    - Blind action value `V_blind(p) = p`.
    - Observing action value `V_obs(p) = p + (1 - p) * strategy_gain_rate - |r_o|`.
    - `VoI = V_obs(p) - V_blind(p) = (1 - p) * strategy_gain_rate - |r_o|`.
    - Indifference threshold `p* = 1.0 - |r_o| / max(1e-6, strategy_gain_rate)`.
    """
    err = float(np.clip(expected_error_yc, 0.0, 0.50))
    p_certainty = 1.0 - err
    cost = abs(float(observation_cost_ro))
    gain_rate = max(0.05, float(strategy_gain_rate))
    expected_reduction = err * gain_rate
    voi = expected_reduction - cost
    p_star = max(0.50, 1.0 - cost / gain_rate)
    return {
        "prior_certainty_p": round(p_certainty, 4),
        "indifference_threshold_p_star": round(p_star, 4),
        "observation_cost": round(cost, 4),
        "value_of_information_voi": round(voi, 4),
        "should_observe": bool(voi > 1e-6),
    }


def compute_neutral_diagnosticity(
    support_count: int,
    challenge_count: int,
    is_unverified_hypothesis: bool,
) -> float:
    """Paper 07 (Gottlieb & Oudeyer 2018, Fig. 3c vs Fig. 4a):
    Neutral diagnosticity / cue validity score that peaks when a hypothesis or claim has high epistemic uncertainty
    or conflicting evidence (`support` vs `challenge`), rather than inflating already-confirmed `supports` items.
    """
    tot = support_count + challenge_count
    if tot == 0:
        return 0.85 if is_unverified_hypothesis else 0.45
    p_pos = (support_count + 0.5) / (tot + 1.0)
    # Binary Shannon entropy of evidence balance in [0, 1]
    h_ent = -(p_pos * math.log2(p_pos) + (1.0 - p_pos) * math.log2(1.0 - p_pos))
    sample_factor = 1.0 / (1.0 + 0.15 * tot)
    base = 0.65 * h_ent + 0.35 * sample_factor
    if is_unverified_hypothesis:
        base = max(base, 0.60)
    return round(float(np.clip(base, 0.05, 1.0)), 4)


# ---------------------------------------------------------------------------
# 3. Persistent Curiosity Engine (`PersistentCuriosityEngine`)
# ---------------------------------------------------------------------------

class PersistentCuriosityEngine:
    """Persistent developmental curiosity engine integrated with `LayerKit`, `CloverMemoryGraphStore`,
    `CloverOntologyGraph`, `CloverJEPARuntime`, and `Repo` (`growth/curiosity/curiosity_state.json`).
    """

    def __init__(
        self,
        storage_path: str | Path,
        *,
        split_threshold: int = 12,
        habituation_decay: float = 0.72,
        habituation_weight: float = 0.55,
        zpd_stall_attempts: int = 3,
        zpd_stall_eps: float = 0.015,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.split_threshold = int(split_threshold)
        self.habituation_decay = float(habituation_decay)
        self.habituation_weight = float(habituation_weight)
        self.zpd_stall_attempts = int(zpd_stall_attempts)
        self.zpd_stall_eps = float(zpd_stall_eps)
        self.lock = threading.RLock()

        self.step_counter: int = 0
        self.regions: dict[str, CuriosityRegion] = {}
        self.unresolved_questions: dict[str, UnresolvedQuestionNode] = {}
        self.strategy_bandit: dict[str, dict[str, StrategyStats]] = {}
        self.exploration_history: list[dict[str, Any]] = []
        # Paper 13: 64-d action-controllability subspace weights R^2_j in [0, 1]
        self.controllable_subspace_r2: np.ndarray = np.ones(D_LATENT, dtype=np.float64)
        self._action_history_u: list[np.ndarray] = []
        self._state_delta_history: list[np.ndarray] = []
        self.last_idle_step_ts: float = 0.0

        self.load()

    # ---------------- Persistence (`growth/curiosity/curiosity_state.json`) ----------------

    def _ensure_root_region(self) -> None:
        if "R0" not in self.regions:
            self.regions["R0"] = CuriosityRegion(region_id="R0")
        if "R0" not in self.strategy_bandit:
            self.strategy_bandit["R0"] = {s: StrategyStats(strategy=s) for s in EPISTEMIC_STRATEGIES}

    def load(self) -> bool:
        with self.lock:
            if not self.storage_path.is_file():
                self._ensure_root_region()
                return False
            try:
                doc = json.loads(self.storage_path.read_text(encoding="utf-8"))
                self.step_counter = int(doc.get("step_counter") or 0)
                self.regions = {
                    k: CuriosityRegion.from_dict(v) for k, v in (doc.get("regions") or {}).items()
                }
                self.unresolved_questions = {
                    k: UnresolvedQuestionNode.from_dict(v)
                    for k, v in (doc.get("unresolved_questions") or {}).items()
                }
                self.strategy_bandit = {}
                for r_id, s_map in (doc.get("strategy_bandit") or {}).items():
                    self.strategy_bandit[r_id] = {
                        s: StrategyStats.from_dict(s_map[s]) if s in s_map else StrategyStats(strategy=s)
                        for s in EPISTEMIC_STRATEGIES
                    }
                self.exploration_history = list(doc.get("exploration_history") or [])[-120:]
                r2_raw = doc.get("controllable_subspace_r2")
                if isinstance(r2_raw, list) and len(r2_raw) == D_LATENT:
                    self.controllable_subspace_r2 = np.asarray(r2_raw, dtype=np.float64)
                self._ensure_root_region()
                return True
            except Exception:
                self._ensure_root_region()
                return False

    def save(self) -> Path:
        with self.lock:
            self._ensure_root_region()
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            doc = {
                "schema_version": SCHEMA_VERSION,
                "updated_at": utc_now_iso(),
                "step_counter": self.step_counter,
                "controllable_subspace_r2": [round(float(x), 4) for x in self.controllable_subspace_r2],
                "regions": {k: v.to_dict() for k, v in self.regions.items()},
                "unresolved_questions": {k: v.to_dict() for k, v in self.unresolved_questions.items()},
                "strategy_bandit": {
                    r_id: {s: st.to_dict() for s, st in s_map.items()}
                    for r_id, s_map in self.strategy_bandit.items()
                },
                "exploration_history": self.exploration_history[-120:],
            }
            tmp = self.storage_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            tmp.replace(self.storage_path)
            return self.storage_path

    # ---------------- Region Tree Routing & Criterion C2 Splitting (Paper 01) ----------------

    def leaf_regions(self) -> list[CuriosityRegion]:
        with self.lock:
            self._ensure_root_region()
            return [r for r in self.regions.values() if r.is_leaf]

    def locate_leaf_region(self, vec_64: np.ndarray | Sequence[float]) -> CuriosityRegion:
        """Route a 64-d vector `vec_64` down the `CuriosityRegionTree` to its unique leaf region."""
        with self.lock:
            self._ensure_root_region()
            v = np.asarray(vec_64, dtype=np.float64).reshape(-1)
            cur = self.regions["R0"]
            visited_guard: set[str] = set()
            while not cur.is_leaf and cur.region_id not in visited_guard:
                visited_guard.add(cur.region_id)
                j = cur.split_dim if cur.split_dim is not None else 0
                thresh = cur.split_threshold if cur.split_threshold is not None else 0.0
                next_id = cur.left_child_id if float(v[j]) <= thresh else cur.right_child_id
                if not next_id or next_id not in self.regions:
                    break
                cur = self.regions[next_id]
            return cur

    def _recompute_region_metrics(self, reg: CuriosityRegion) -> None:
        """Update `tonic_lpm`, `tonic_cpm`, `abs_lp_vigilance`, `habituation_trace`, and `state_curiosity`."""
        lpm, _, plast_p = compute_windowed_lp_and_vigilance(reg.pred_error_window)
        cpm, vig, plast_c = compute_windowed_lp_and_vigilance(reg.comp_error_window)
        reg.tonic_lpm = lpm
        reg.tonic_cpm = cpm
        reg.abs_lp_vigilance = vig

        # Priority 4: Dishabituate when new evidence invalidates previous competence (`vig > 0`)
        if vig > 0.0:
            reg.habituation_trace = 0.0

        # Paper 10 Section 4: norm of the smoothed signed shift vector ||E[Delta z | R_n]||
        shift_norm = float(np.linalg.norm(np.asarray(reg.signed_shift_ema_64, dtype=np.float64)))
        reliability_gain = max(0.0, reg.dot_yc_reliability)

        # Paper 04 & 13: Controllability gate prevents Passive Spectator Trap (`controllability_score == 0` -> suppressed)
        ctrl_gate = float(np.clip(reg.controllability_score, 0.02, 1.0))
        plasticity = 0.5 * (plast_p + plast_c)

        # Mastery decay: when a region is mastered (low expected error, low recent error, zero vigilance), interest quiets down
        is_mastered = (
            reg.expected_error_yc < 0.08
            and vig == 0.0
            and len(reg.comp_error_window) >= 3
            and float(reg.comp_error_window[-1]) <= 0.08
        )
        mastery_factor = 0.08 if is_mastered else 1.0

        # Combine Knowledge Progress (LPM), Competence Progress (CPM), Reliability Improvement (dot_y_C), and |LP| Vigilance
        raw_interest = (
            0.30 * lpm
            + 0.45 * cpm
            + 0.15 * reliability_gain
            + 0.10 * min(0.25, shift_norm)
            + 0.65 * vig
        ) * (0.45 + 0.55 * plasticity) * ctrl_gate * mastery_factor

        # Paper 02 E3B: Divide by habitation trace `1 + lambda_h * h_R(t)` so repeated visits cool down smoothly
        denom = 1.0 + self.habituation_weight * reg.habituation_trace
        reg.state_curiosity = round(float(raw_interest / denom), 5)

    def _maybe_split_region_c2(self, reg: CuriosityRegion) -> tuple[CuriosityRegion, CuriosityRegion] | None:
        """Paper 01 (Criterion C2): Split leaf region `reg` if it has accumulated `> self.split_threshold` exemplars
        and a split reduces weighted outcome variance by `>= 5%`.
        Crucial boundary fix (from Paper 01 E2): Children inherit the parent's partitioned exemplar histories,
        error windows, reliability state `expected_error_yc`, and strategy bandit stats so splitting NEVER
        manufactures artificial learning progress in an already-mastered region!
        """
        if not reg.is_leaf or len(reg.exemplar_vectors) <= self.split_threshold:
            return None
        # Prefer splitting along controllable dimensions (Paper 13)
        ctrl_dims = [int(i) for i in np.argsort(-self.controllable_subspace_r2)[:24]]
        split_res = find_criterion_c2_split(reg.exemplar_vectors, reg.exemplar_outcomes, candidate_dims=ctrl_dims)
        if split_res is None:
            return None
        j_star, v_star, _ = split_res
        left_id = f"{reg.region_id}.0"
        right_id = f"{reg.region_id}.1"
        if left_id in self.regions or right_id in self.regions:
            return None

        X = np.asarray(reg.exemplar_vectors, dtype=np.float64)
        Y = np.asarray(reg.exemplar_outcomes, dtype=np.float64)
        P_err = np.asarray(reg.pred_error_window[-len(Y) :], dtype=np.float64)
        C_err = np.asarray(reg.comp_error_window[-len(Y) :], dtype=np.float64)
        mask_l = X[:, j_star] <= v_star

        def _make_child(cid: str, mask: np.ndarray) -> CuriosityRegion:
            sub_x = X[mask]
            sub_y = Y[mask]
            sub_p = P_err[mask] if len(P_err) == len(mask) else sub_y
            sub_c = C_err[mask] if len(C_err) == len(mask) else sub_y
            cent = np.mean(sub_x, axis=0) if len(sub_x) > 0 else np.asarray(reg.centroid_64, dtype=np.float64)
            c_norm = float(np.linalg.norm(cent))
            if c_norm > 1e-9:
                cent = cent / c_norm
            # Inherit partitioned outcome errors and parent's reliability baseline (no artificial reset!)
            inherited_p = [float(v) for v in sub_p[-16:]] if len(sub_p) >= 2 else list(reg.pred_error_window[-8:])
            inherited_c = [float(v) for v in sub_c[-16:]] if len(sub_c) >= 2 else list(reg.comp_error_window[-8:])
            child = CuriosityRegion(
                region_id=cid,
                parent_id=reg.region_id,
                centroid_64=[round(float(v), 5) for v in cent],
                emergent_keywords=list(reg.emergent_keywords[:8]),
                exemplar_vectors=[[round(float(v), 5) for v in row] for row in sub_x[-24:]],
                exemplar_outcomes=[round(float(v), 5) for v in sub_y[-24:]],
                pred_error_window=inherited_p,
                comp_error_window=inherited_c,
                expected_error_yc=float(np.mean(inherited_c)) if inherited_c else reg.expected_error_yc,
                prev_expected_error_yc=reg.expected_error_yc,
                signed_shift_ema_64=list(reg.signed_shift_ema_64),
                controllability_score=reg.controllability_score,
                controllable_Try_count=reg.controllable_Try_count,
                uncontrollable_Try_count=reg.uncontrollable_Try_count,
                habituation_trace=reg.habituation_trace,
                visits=int(np.sum(mask)),
                last_visited_step=reg.last_visited_step,
            )
            self._recompute_region_metrics(child)
            return child

        left_child = _make_child(left_id, mask_l)
        right_child = _make_child(right_id, ~mask_l)

        reg.split_dim = j_star
        reg.split_threshold = v_star
        reg.left_child_id = left_id
        reg.right_child_id = right_id

        self.regions[left_id] = left_child
        self.regions[right_id] = right_child
        parent_bandit = self.strategy_bandit.get(reg.region_id) or {
            s: StrategyStats(strategy=s) for s in EPISTEMIC_STRATEGIES
        }
        self.strategy_bandit[left_id] = copy.deepcopy(parent_bandit)
        self.strategy_bandit[right_id] = copy.deepcopy(parent_bandit)

        # Re-home any unresolved questions that belonged to `reg` into the appropriate child leaf
        for q_node in self.unresolved_questions.values():
            if q_node.region_id == reg.region_id:
                q_vec = encode_text_to_64d(q_node.summary)
                q_node.region_id = left_id if float(q_vec[j_star]) <= v_star else right_id

        return left_child, right_child

    # ---------------- Paper 13: Action-Controllable Subspace Estimation ----------------

    def update_controllable_subspace(
        self,
        action_vec_8: np.ndarray | Sequence[float],
        state_delta_64: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Paper 13 (Santucci et al. 2020 IMOL): Estimate per-dimension controllability `R^2_j(Action -> Delta z_j)`
        across the 64-d latent space so goal similarity and splitting prioritize controllable dimensions over
        uncontrollable background distractors.
        """
        with self.lock:
            u = np.asarray(action_vec_8, dtype=np.float64).reshape(-1)
            dz = np.asarray(state_delta_64, dtype=np.float64).reshape(-1)
            if u.size == 0 or dz.size != D_LATENT:
                return self.controllable_subspace_r2
            self._action_history_u.append(u)
            self._state_delta_history.append(dz)
            if len(self._action_history_u) > 64:
                self._action_history_u = self._action_history_u[-64:]
                self._state_delta_history = self._state_delta_history[-64:]

            if len(self._action_history_u) >= 10:
                U = np.asarray(self._action_history_u, dtype=np.float64)
                S = np.asarray(self._state_delta_history, dtype=np.float64)
                n_obs, p_dim = U.shape
                gram = U.T @ U + 0.5 * np.eye(p_dim, dtype=np.float64)
                r2 = np.zeros(D_LATENT, dtype=np.float64)
                for j in range(D_LATENT):
                    var_j = float(np.var(S[:, j]))
                    if var_j < 1e-8:
                        r2[j] = 0.05
                        continue
                    w = np.linalg.solve(gram, U.T @ S[:, j])
                    pred_j = U @ w
                    res_var = float(np.var(S[:, j] - pred_j))
                    raw_r2 = 1.0 - res_var / var_j
                    # Adjusted R^2 penalizing degrees of freedom p_dim so small-sample noise does not overfit
                    dof_denom = max(1, n_obs - p_dim - 1)
                    adj_r2 = 1.0 - (1.0 - raw_r2) * ((n_obs - 1.0) / dof_denom)
                    r2[j] = float(np.clip(adj_r2, 0.05, 1.0))
                self.controllable_subspace_r2 = 0.2 * self.controllable_subspace_r2 + 0.8 * r2
            return self.controllable_subspace_r2

    def controllable_distance(self, vec_a: np.ndarray | Sequence[float], vec_b: np.ndarray | Sequence[float]) -> float:
        """Weighted Euclidean distance in 64-d space weighted by `self.controllable_subspace_r2` (Paper 13)."""
        a = np.asarray(vec_a, dtype=np.float64).reshape(D_LATENT)
        b = np.asarray(vec_b, dtype=np.float64).reshape(D_LATENT)
        w = self.controllable_subspace_r2 / max(1e-9, float(np.mean(self.controllable_subspace_r2)))
        return float(np.sqrt(np.sum(w * ((a - b) ** 2))))

    # ---------------- Hook A: Pre-Turn Meaning Bridge, VoI & Reliability-Weighted Retrieval ----------------

    def recommend_strategy_and_voi(
        self,
        query: str,
        *,
        latent_cue_64: np.ndarray | Sequence[float] | None = None,
        discernment_state: str = "proceed",
        declined_capabilities: Sequence[str] = (),
        permissions: dict[str, str] | None = None,
        interactive: bool = True,
    ) -> dict[str, Any]:
        """Hook A (`before_turn` / `LayerKit.compute_meaning_bridge` / `studio_idle()`):
        1. Respects Paper 03 / `AGENTS.md` §9 Discernment Hierarchy: if `discernment_state` is `"refuse"` or `"ask"`,
           curiosity never overrides safety/authority boundaries.
        2. Locates the active leaf region `R_n` for `query`.
        3. Filters strategies by `permissions` and `declined_capabilities`, then uses the `SGIM-ACTS` bandit (Paper 11)
           + POMDP `VoI` (Paper 06) to select the highest-value permitted epistemic strategy and its concrete Studio tools.
        """
        with self.lock:
            q_vec = encode_text_to_64d(query, latent_cue_64=latent_cue_64)
            reg = self.locate_leaf_region(q_vec)
            if discernment_state in ("refuse", "ask", "do_not_know"):
                return {
                    "region_id": reg.region_id,
                    "preempted_by_discernment": discernment_state,
                    "selected_strategy": "human_clarification_ask" if discernment_state != "refuse" else "none",
                    "recommended_studio_tools": [],
                    "voi": 0.0,
                    "should_observe": False,
                    "belief_action_boost_8": np.zeros(8, dtype=np.float32),
                    "active_zpd_questions": [],
                }

            bandit = self.strategy_bandit.setdefault(
                reg.region_id, {s: StrategyStats(strategy=s) for s in EPISTEMIC_STRATEGIES}
            )
            declined = set(declined_capabilities or ())
            best_strat = "memory_graph_recall"
            best_score = -1e9
            best_voi_info: dict[str, Any] = {}
            permitted_strategies: list[str] = []

            for strat in EPISTEMIC_STRATEGIES:
                if not interactive and strat == "human_clarification_ask":
                    continue
                if strat == "web_primary_research" and ("web.search" in declined or "web.page" in declined):
                    continue
                if strat == "sandbox_counterexample_probe" and ("python" in declined or "calc" in declined):
                    continue
                if strat == "workspace_code_inspect" and ("files.read" in declined or "files.search" in declined):
                    continue
                if permissions is not None:
                    req_perm = STRATEGY_REQUIRED_PERMISSION.get(strat)
                    if req_perm is not None:
                        p_mode = permissions.get(req_perm, "ask")
                        if p_mode == "deny":
                            continue
                        if not interactive and p_mode != "allow":
                            continue
                permitted_strategies.append(strat)
                st = bandit[strat]
                # Paper 11 SGIM-ACTS: empirical LP minus variance penalty
                gain_rate = max(0.25, st.empirical_lp * 3.5)
                eff_uncertainty = max(reg.expected_error_yc, 0.75 * reg.abs_lp_vigilance)
                voi_info = compute_pomdp_voi(
                    eff_uncertainty,
                    strategy_gain_rate=gain_rate,
                    observation_cost_ro=STRATEGY_OBSERVATION_COST[strat],
                )
                # Combine empirical LP with VoI and small exploration bonus for untried strategies
                score = st.empirical_lp + 0.5 * voi_info["value_of_information_voi"] - 0.15 * st.outcome_variance
                if score > best_score:
                    best_score = score
                    best_strat = strat
                    best_voi_info = voi_info

            boost_8 = np.zeros(8, dtype=np.float32)
            if best_voi_info.get("should_observe"):
                dim_idx = STRATEGY_TO_ACTION_DIM.get(best_strat, 3)
                boost_8[dim_idx] = np.float32(min(0.65, 0.25 + 2.0 * reg.state_curiosity))
                if best_strat == "sandbox_counterexample_probe":
                    boost_8[1] = np.float32(0.20)  # math_calc synergy

            open_qs = [
                q.to_dict()
                for q in self.unresolved_questions.values()
                if q.region_id == reg.region_id and q.zpd_status == "active_zpd"
            ]
            open_qs.sort(key=lambda d: -float(d.get("diagnosticity_score", 0.5)))

            studio_tools = list(STRATEGY_TO_STUDIO_TOOLS.get(best_strat, ())) if best_voi_info.get("should_observe") else []

            return {
                "region_id": reg.region_id,
                "emergent_keywords": list(reg.emergent_keywords[:6]),
                "state_curiosity": reg.state_curiosity,
                "expected_error_yc": round(reg.expected_error_yc, 4),
                "tonic_lpm": reg.tonic_lpm,
                "tonic_cpm": reg.tonic_cpm,
                "abs_lp_vigilance": reg.abs_lp_vigilance,
                "permitted_strategies": permitted_strategies,
                "selected_strategy": best_strat,
                "recommended_studio_tools": studio_tools,
                "voi_analysis": best_voi_info,
                "should_observe": bool(best_voi_info.get("should_observe")),
                "belief_action_boost_8": boost_8,
                "active_zpd_questions": open_qs[:3],
            }

    @staticmethod
    def reliability_weighted_rank(
        candidates: Sequence[dict[str, Any]],
        alpha_variance: float = 0.50,
    ) -> list[dict[str, Any]]:
        """Paper 12 (`SGIM-D` Algorithm 4.6, Section 4.4.2):
        Rank retrieved memory/concept exemplars by minimizing `dist(tau_h, tau_g) + alpha * var_h`
        where `var_h` is the outcome variance of exemplar `h` (`worked` vs `did_not_work`).
        """
        scored = []
        for item in candidates:
            dist = float(item.get("distance", 1.0 - float(item.get("score", 0.5))))
            w = int(item.get("worked", 0))
            d = int(item.get("did_not_work", 0))
            tot = w + d
            if tot == 0:
                var_h = float(item.get("variance", 0.04))
            else:
                p_fail = d / tot
                # Bernoulli outcome variance p*(1-p) + failure penalty when d > w
                var_h = p_fail * (1.0 - p_fail) + (0.35 if d > w else 0.0)
            rel_cost = dist + alpha_variance * var_h
            scored.append((rel_cost, dict(item, reliability_cost=round(rel_cost, 4), outcome_variance=round(var_h, 4))))
        scored.sort(key=lambda pair: pair[0])
        return [ pair[1] for pair in scored ]

    # ---------------- Hook B: Post-Turn Feedback (`on_turn_outcome`) ----------------

    def record_interaction_outcome(
        self,
        query: str,
        *,
        pred_error: float | None = None,
        comp_error: float | None = None,
        pre_repair_comp_error: float | None = None,
        strategy_used: str = "memory_graph_recall",
        actor: str = "clover",
        controllable: bool = True,
        latent_before_64: np.ndarray | Sequence[float] | None = None,
        latent_after_64: np.ndarray | Sequence[float] | None = None,
        unresolved_summary: str | None = None,
        memory_entry_id: str | None = None,
        parent_question_id: str | None = None,
        prerequisite_ids: Sequence[str] = (),
        lateral_modality_ids: Sequence[str] = (),
        view_dimension_added: str | None = None,
        user_request: str | None = None,
        model_prediction: str | None = None,
        verification_result: str | None = None,
        correction: str | None = None,
        session_id: str = "",
        turn_n: int = 0,
    ) -> dict[str, Any]:
        """Hook B (`on_turn_outcome`): Update `CuriosityRegionTree`, Confidence Module `C(R_n)`, `SGIM-ACTS` strategy
        bandit, and `ZPDES` unresolved question frontier from a real interaction outcome.
        Priority 1: When `comp_error is None` and `pred_error is None` (an ordinary conversational turn with no
        objective verification check and no user feedback yet), records the visit and updates emergent keywords/prototype
        WITHOUT assigning any synthetic competence score.
        """
        with self.lock:
            self.step_counter += 1
            anchor_text = (user_request or query or "").strip()
            q_vec = encode_text_to_64d(anchor_text, latent_cue_64=latent_before_64)

            # Decay habitation traces across all regions before boosting the visited region (Paper 02 E3B)
            for r in self.regions.values():
                r.habituation_trace = round(r.habituation_trace * self.habituation_decay, 5)
                if r.is_leaf and r.visits > 0:
                    self._recompute_region_metrics(r)

            reg = self.locate_leaf_region(q_vec)
            reg.visits += 1
            reg.last_visited_step = self.step_counter
            reg.habituation_trace = round(reg.habituation_trace + 1.0, 5)
            reg.updated_at = utc_now_iso()

            # Competitive Hebbian prototype adaptation (`MR` microcolumn receptive field, Paper 05)
            cent = np.asarray(reg.centroid_64, dtype=np.float64)
            if reg.visits == 1 or float(np.linalg.norm(cent)) < 1e-9:
                cent = q_vec.copy()
            else:
                cent = 0.88 * cent + 0.12 * q_vec
                cent /= max(1e-9, float(np.linalg.norm(cent)))
            reg.centroid_64 = [round(float(x), 5) for x in cent]

            # Update emergent keywords from observed query terms (zero hardcoded topics)
            new_terms = extract_emergent_terms(anchor_text, max_terms=6)
            merged_kw = list(dict.fromkeys(new_terms + reg.emergent_keywords))[:10]
            reg.emergent_keywords = merged_kw

            prev_curiosity = reg.state_curiosity
            s_key = strategy_used if strategy_used in EPISTEMIC_STRATEGIES else "memory_graph_recall"

            # Priority 1: If neither competence nor prediction error was measured on this turn, do NOT fabricate fixed numbers
            if comp_error is None and pred_error is None and not unresolved_summary:
                self._recompute_region_metrics(reg)
                episode = {
                    "step": self.step_counter,
                    "at": utc_now_iso(),
                    "session": session_id,
                    "turn": turn_n,
                    "region_id": reg.region_id,
                    "strategy": s_key,
                    "actor": actor,
                    "user_request": anchor_text[:200],
                    "model_prediction": (model_prediction or "")[:200],
                    "verified_in_turn": False,
                    "pred_error": None,
                    "comp_error": None,
                    "state_curiosity_before": prev_curiosity,
                    "state_curiosity_after": reg.state_curiosity,
                    "split_into": None,
                    "spawned_question_id": None,
                    "resolved_question_ids": [],
                }
                self.exploration_history.append(episode)
                self.save()
                return episode

            eff_comp = float(comp_error if comp_error is not None else pred_error)  # type: ignore[arg-type]
            eff_pred = float(pred_error if pred_error is not None else eff_comp)

            # If Cycle 1 failed real checks (`pre_repair_comp_error`) before Cycle k repaired it to `eff_comp`,
            # record the pre-repair sample first so within-turn competence progress is grounded in the window!
            if pre_repair_comp_error is not None and float(pre_repair_comp_error) > eff_comp:
                pre_c = float(np.clip(pre_repair_comp_error, 0.0, 1.5))
                reg.exemplar_vectors.append([round(float(x), 5) for x in q_vec])
                reg.exemplar_outcomes.append(round(pre_c, 5))
                reg.pred_error_window.append(round(max(eff_pred, pre_c), 5))
                reg.comp_error_window.append(round(pre_c, 5))

            # Record exemplar for Criterion C2 splitting (Paper 01)
            combined_err = float(np.clip(0.4 * eff_pred + 0.6 * eff_comp, 0.0, 1.5))
            reg.exemplar_vectors.append([round(float(x), 5) for x in q_vec])
            reg.exemplar_outcomes.append(round(combined_err, 5))
            reg.pred_error_window.append(round(eff_pred, 5))
            reg.comp_error_window.append(round(eff_comp, 5))

            # Paper 04 & 13: Update regional controllability score
            if controllable:
                reg.controllable_Try_count += 1
            else:
                reg.uncontrollable_Try_count += 1
            tot_ctrl = reg.controllable_Try_count + reg.uncontrollable_Try_count
            reg.controllability_score = round(reg.controllable_Try_count / max(1, tot_ctrl), 4)

            # Paper 04 Eq. (15): Phasic predictive surprise ratio SM(t) = Er(t) / <Er_R>
            reg.phasic_surprise_sm = round(combined_err / max(0.02, reg.expected_error_yc), 4)

            # Paper 10 (Schmidhuber 1991 Section 2 & 3): Confidence Module C(R_n) & signed reliability derivative dot_y_C
            yc_old = reg.expected_error_yc
            eta_c = 0.22
            yc_new = (1.0 - eta_c) * yc_old + eta_c * combined_err
            reg.prev_expected_error_yc = round(yc_old, 5)
            reg.expected_error_yc = round(yc_new, 5)
            reg.dot_yc_reliability = round(yc_old - yc_new, 5)

            # Paper 10 Section 4: Smoothed signed shift vector E[Delta z | R_n] (Jensen's inequality guard)
            if latent_before_64 is not None and latent_after_64 is not None:
                zb = np.asarray(latent_before_64, dtype=np.float64).reshape(D_LATENT)
                za = np.asarray(latent_after_64, dtype=np.float64).reshape(D_LATENT)
                dz = za - zb
                cur_ema = np.asarray(reg.signed_shift_ema_64, dtype=np.float64)
                new_ema = 0.82 * cur_ema + 0.18 * dz
                reg.signed_shift_ema_64 = [round(float(x), 5) for x in new_ema]

            self._recompute_region_metrics(reg)

            # Update SGIM-ACTS strategy bandit (Paper 11)
            bandit = self.strategy_bandit.setdefault(
                reg.region_id, {s: StrategyStats(strategy=s) for s in EPISTEMIC_STRATEGIES}
            )
            st = bandit[s_key]
            st.attempts += 1
            st.last_used_step = self.step_counter
            prev_s_err = (
                float(pre_repair_comp_error)
                if pre_repair_comp_error is not None
                else (st.error_window[-1] if st.error_window else 0.55)
            )
            st.error_window.append(round(eff_comp, 5))
            if not controllable:
                s_prog = 0.0
            elif eff_comp <= 0.10 and prev_s_err <= 0.12:
                # Successful verification in an already-mastered region maintains competence (not a high-error stall)
                s_prog = max(0.08, prev_s_err - eff_comp)
            else:
                s_prog = max(0.0, prev_s_err - eff_comp)
            st.progress_window.append(round(s_prog, 5))
            if view_dimension_added:
                st.views_discovered += 1

            # Update or spawn ZPDES unresolved question node (Papers 06, 07, 08, 12)
            spawned_qid = None
            resolved_qids: list[str] = []
            q_lookup_text = (unresolved_summary or anchor_text).strip()[:240]
            existing_qid = "Q-" + hashlib.sha1(q_lookup_text.lower().encode("utf-8")).hexdigest()[:10].upper()
            # Also check if there is an active question in this region with overlapping terms
            if existing_qid not in self.unresolved_questions and eff_comp <= 0.10:
                q_terms = set(extract_emergent_terms(anchor_text, max_terms=8))
                for cand_q in self.unresolved_questions.values():
                    if cand_q.region_id == reg.region_id and cand_q.zpd_status != "resolved_verified":
                        if len(q_terms & set(extract_emergent_terms(cand_q.summary, max_terms=8))) >= 2:
                            existing_qid = cand_q.question_id
                            break

            if unresolved_summary or eff_comp > 0.25 or reg.abs_lp_vigilance > 0.0 or existing_qid in self.unresolved_questions:
                qid = existing_qid
                q_node = self.unresolved_questions.get(qid)
                if q_node is None:
                    origin = (
                        "competence_regression"
                        if reg.abs_lp_vigilance > 0.0
                        else ("child_frontier" if parent_question_id else "turn_discrepancy")
                    )
                    q_node = UnresolvedQuestionNode(
                        question_id=qid,
                        region_id=reg.region_id,
                        summary=q_lookup_text,
                        origin_kind=origin,
                        memory_entry_id=memory_entry_id,
                        parent_question_id=parent_question_id,
                        prerequisite_ids=list(prerequisite_ids),
                        lateral_modality_ids=list(lateral_modality_ids),
                        zpd_status="prerequisite_pending" if prerequisite_ids else "active_zpd",
                    )
                    self.unresolved_questions[qid] = q_node
                    spawned_qid = qid
                elif reg.abs_lp_vigilance > 0.0 and q_node.zpd_status == "resolved_verified":
                    # Priority 4: Revive resolved question when new evidence invalidates competence!
                    q_node.zpd_status = "active_zpd"
                    q_node.origin_kind = "competence_regression"

                q_node.attempts += 1
                q_node.error_history.append(round(eff_comp, 4))
                q_node.updated_at = utc_now_iso()
                if view_dimension_added and view_dimension_added not in q_node.views_covered:
                    q_node.views_covered.append(view_dimension_added)

                if eff_comp <= 0.10:
                    q_node.zpd_status = "resolved_verified"
                    resolved_qids.append(qid)
                    self._on_question_progressed_or_resolved(q_node)
                elif (
                    q_node.attempts >= self.zpd_stall_attempts
                    and q_node.recent_lp <= self.zpd_stall_eps
                    and eff_comp > 0.30
                ):
                    # Paper 08 Figure 5 (ZPDES): stall detected -> deactivate stalled node and open lateral/prerequisite nodes
                    q_node.zpd_status = "stalled_deactivated"
                    for lat_id in list(q_node.lateral_modality_ids) + list(q_node.prerequisite_ids):
                        if lat_id in self.unresolved_questions:
                            lat_q = self.unresolved_questions[lat_id]
                            if lat_q.zpd_status in ("prerequisite_pending", "stalled_deactivated"):
                                lat_q.zpd_status = "active_zpd"
                elif q_node.recent_lp > self.zpd_stall_eps:
                    self._on_question_progressed_or_resolved(q_node)

            # Check Criterion C2 region split (Paper 01)
            split_children = self._maybe_split_region_c2(reg)

            episode = {
                "step": self.step_counter,
                "at": utc_now_iso(),
                "session": session_id,
                "turn": turn_n,
                "region_id": reg.region_id,
                "strategy": s_key,
                "actor": actor,
                "user_request": anchor_text[:200],
                "model_prediction": (model_prediction or "")[:200],
                "verification_result": (verification_result or "")[:200],
                "correction": (correction or "")[:200],
                "verified_in_turn": True,
                "pred_error": round(eff_pred, 4),
                "pre_repair_comp_error": round(float(pre_repair_comp_error), 4) if pre_repair_comp_error is not None else None,
                "comp_error": round(eff_comp, 4),
                "phasic_surprise_sm": reg.phasic_surprise_sm,
                "tonic_lpm": reg.tonic_lpm,
                "tonic_cpm": reg.tonic_cpm,
                "dot_yc_reliability": reg.dot_yc_reliability,
                "abs_lp_vigilance": reg.abs_lp_vigilance,
                "state_curiosity_before": prev_curiosity,
                "state_curiosity_after": reg.state_curiosity,
                "split_into": [split_children[0].region_id, split_children[1].region_id] if split_children else None,
                "spawned_question_id": spawned_qid,
                "resolved_question_ids": resolved_qids,
            }
            self.exploration_history.append(episode)
            self.save()
            return episode

    def _on_question_progressed_or_resolved(self, progressed_node: UnresolvedQuestionNode) -> None:
        """Paper 08 Fig. 5 (`ZPDES`): When a prerequisite or lateral modality question progresses or resolves,
        re-activate any stalled or pending questions that depended on it.
        """
        qid = progressed_node.question_id
        for other in self.unresolved_questions.values():
            if other.question_id == qid or other.zpd_status == "resolved_verified":
                continue
            if qid in other.prerequisite_ids or qid in other.lateral_modality_ids or other.question_id in progressed_node.lateral_modality_ids:
                if other.zpd_status in ("stalled_deactivated", "prerequisite_pending"):
                    other.zpd_status = "active_zpd"
                    # Reset recent stall window so it gets a fresh evaluation with the newly acquired skill
                    if len(other.error_history) >= 2:
                        other.error_history = [other.error_history[-1]]

    # ---------------- Candidate Selection Without Hardcoded Lists or `solved_ids` Crutch ----------------

    def rank_candidates_by_curiosity(
        self,
        candidates: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rank candidate problems, reading units, or topics by developmental curiosity (Papers 01, 02, 04, 08, 10, 12)
        WITHOUT using a hardcoded order or a binary `solved_ids` exclusion set:
          - Evaluates each candidate's leaf region `R_n` in `CuriosityRegionTree`
          - Combines `state_curiosity` (`LPM + CPM + dot_y_C + |LP| vigilance`, cooled by `habituation_trace`)
            with `ZPD` readiness (penalizing candidates whose prerequisites are still unmastered or whose region
            has already plateaued at near-zero error).
        """
        with self.lock:
            scored: list[tuple[float, dict[str, Any]]] = []
            for cand in candidates:
                text = str(cand.get("text") or cand.get("question") or cand.get("title") or "")
                cue = cand.get("cue_64")
                vec = encode_text_to_64d(text, latent_cue_64=cue)
                reg = self.locate_leaf_region(vec)

                # Check if candidate has an associated question node in ZPD
                qid = cand.get("question_id")
                zpd_mult = 1.0
                if qid and qid in self.unresolved_questions:
                    q_node = self.unresolved_questions[qid]
                    if q_node.zpd_status == "stalled_deactivated":
                        zpd_mult = 0.15
                    elif q_node.zpd_status == "prerequisite_pending":
                        # Check if all prerequisites are now resolved
                        if all(
                            p in self.unresolved_questions
                            and self.unresolved_questions[p].zpd_status == "resolved_verified"
                            for p in q_node.prerequisite_ids
                        ):
                            q_node.zpd_status = "active_zpd"
                            zpd_mult = 1.35
                        else:
                            zpd_mult = 0.20
                    elif q_node.zpd_status == "resolved_verified":
                        zpd_mult = 0.05 if reg.abs_lp_vigilance == 0.0 else 1.50
                    else:
                        zpd_mult = 1.25 * (0.5 + 0.5 * q_node.diagnosticity_score)

                # If region is already mastered (expected_error_yc < 0.06) and has zero vigilance spike, interest is low
                mastery_gate = 0.08 if (reg.expected_error_yc < 0.06 and reg.abs_lp_vigilance == 0.0 and reg.visits >= 2) else 1.0
                # Within-region exemplar habitation cooldown (Paper 01 & Paper 02 E3B):
                # Cools down specific stimuli recently repeated inside R_n so diverse points in R_n are sampled
                local_hab = 0.0
                if reg.exemplar_vectors:
                    recent_vecs = np.asarray(reg.exemplar_vectors[-12:], dtype=np.float64)
                    sims = np.maximum(0.0, recent_vecs @ vec)
                    weights = np.array([self.habituation_decay ** (len(sims) - 1 - i) for i in range(len(sims))], dtype=np.float64)
                    local_hab = float(np.sum(weights * (sims ** 4)))
                local_cooldown = 1.0 / (1.0 + self.habituation_weight * local_hab)
                score = reg.state_curiosity * zpd_mult * mastery_gate * local_cooldown
                scored.append((
                    score,
                    dict(
                        cand,
                        region_id=reg.region_id,
                        curiosity_score=round(float(score), 5),
                        tonic_lpm=reg.tonic_lpm,
                        tonic_cpm=reg.tonic_cpm,
                        expected_error_yc=reg.expected_error_yc,
                        habituation_trace=reg.habituation_trace,
                        controllability=reg.controllability_score,
                    ),
                ))
            scored.sort(key=lambda pair: -pair[0])
            return [pair[1] for pair in scored]

    # ---------------- Hook C: Idle Background Consolidation & Non-Regression Replay (Papers 08 & 13) ----------------

    def run_idle_consolidation_step(
        self,
        memory_store: Any | None = None,
        ontology_graph: Any | None = None,
        sandbox_runner: Callable[[str], tuple[bool, Any, str]] | None = None,
        regression_tests: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Hook C (`on_idle_step` for `studio_idle()`):
        1. Syncs any `unresolved`, `hypothesis`, or `contested` entries from `CloverMemoryGraphStore` into `self.unresolved_questions`
           with neutral diagnosticity scores (Paper 07).
        2. Selects the highest-`state_curiosity` active ZPD region/question (closing Fig. 1B of Paper 08:
           `LP -> State Curiosity -> Memory Consolidation Priority`).
        3. If a candidate hypothesis has a verification script (`sandbox_counterexample_probe`), executes it in
           `IsolatedPythonSandbox` AND runs **non-regression replay** (`regression_tests`, Paper 13) before promoting
           the hypothesis to an active `externally_verified` concept. If any regression test fails, promotion is
           rolled back and the hypothesis is marked `contested`!
        """
        with self.lock:
            synced_from_memory = 0
            if memory_store is not None and hasattr(memory_store, "entries"):
                for e in list(memory_store.entries.values()):
                    is_open = (
                        getattr(e, "epistemic_status", "") in ("unresolved", "hypothesis")
                        or getattr(e, "status", "") in ("contested", "under_review")
                    )
                    if not is_open:
                        continue
                    qid = f"Q-MEM-{e.id}"
                    if qid not in self.unresolved_questions:
                        vec = encode_text_to_64d(f"{e.title or ''} {e.summary or ''}")
                        reg = self.locate_leaf_region(vec)
                        rels = getattr(e, "relationships", ()) or ()
                        sup_c = sum(1 for r in rels if getattr(r, "type", "") == "supports")
                        chal_c = sum(1 for r in rels if getattr(r, "type", "") in ("challenges", "unresolved_with"))
                        prereqs = [f"Q-MEM-{r.target}" for r in rels if getattr(r, "type", "") == "depends_on"]
                        diag = compute_neutral_diagnosticity(
                            sup_c, chal_c, is_unverified_hypothesis=(getattr(e, "entry_type", "") == "hypothesis")
                        )
                        self.unresolved_questions[qid] = UnresolvedQuestionNode(
                            question_id=qid,
                            region_id=reg.region_id,
                            summary=str(e.summary or e.title or e.id)[:240],
                            origin_kind="unverified_hypothesis" if getattr(e, "entry_type", "") == "hypothesis" else "memory_contradiction",
                            memory_entry_id=e.id,
                            prerequisite_ids=prereqs,
                            diagnosticity_score=diag,
                        )
                        synced_from_memory += 1

            # Rank active ZPD questions by regional state_curiosity * diagnosticity (Fig. 1B of Paper 08)
            active_qs = [
                q for q in self.unresolved_questions.values() if q.zpd_status == "active_zpd"
            ]
            active_qs.sort(
                key=lambda q: -(
                    self.regions.get(q.region_id, self.regions["R0"]).state_curiosity * (0.5 + 0.5 * q.diagnosticity_score)
                )
            )
            top_q = active_qs[0] if active_qs else None

            consolidation_result: dict[str, Any] = {
                "synced_open_memory_entries": synced_from_memory,
                "active_zpd_count": len(active_qs),
                "selected_question_id": top_q.question_id if top_q else None,
                "selected_region_id": top_q.region_id if top_q else "R0",
                "promoted": False,
                "rolled_back_on_regression": False,
            }

            if top_q and sandbox_runner is not None and memory_store is not None and top_q.memory_entry_id:
                e_hyp = memory_store.entries.get(top_q.memory_entry_id)
                v_code = getattr(e_hyp, "verification_code", None) if e_hyp else None
                if not v_code and e_hyp:
                    m_code = re.search(r"```python\s*\n([\s\S]*?)```", str(getattr(e_hyp, "content", "") or ""))
                    if m_code:
                        v_code = m_code.group(1).strip()
                if v_code:
                    integrity = validate_behavioral_verification_probe(v_code)
                    consolidation_result["probe_integrity"] = integrity
                    if not integrity["ok"]:
                        consolidation_result["rejected_tautological_or_existence_probe"] = True
                    else:
                        ok_new, val_new, _ = sandbox_runner(v_code)
                        passed_new = bool(
                            ok_new and (val_new is True or (isinstance(val_new, dict) and val_new.get("passed") is True))
                        )
                        # Paper 13: Run non-regression replay on existing validated concepts before allowing promotion
                        reg_ok = True
                        for rt in regression_tests or ():
                            ok_r, val_r, _ = sandbox_runner(str(rt.get("code") or ""))
                            if not (ok_r and (val_r is True or (isinstance(val_r, dict) and val_r.get("passed") is True))):
                                reg_ok = False
                                break
                        if passed_new and reg_ok and e_hyp is not None:
                            e_hyp.epistemic_status = "externally_verified"
                            e_hyp.status = "active"
                            if hasattr(memory_store, "_save_entry_in_place"):
                                memory_store._save_entry_in_place(e_hyp)
                            top_q.zpd_status = "resolved_verified"
                            self._on_question_progressed_or_resolved(top_q)
                            consolidation_result["promoted"] = True
                        elif passed_new and not reg_ok and e_hyp is not None:
                            e_hyp.status = "contested"
                            if hasattr(memory_store, "_save_entry_in_place"):
                                memory_store._save_entry_in_place(e_hyp)
                            consolidation_result["rolled_back_on_regression"] = True

            self.save()
            return consolidation_result

    # ---------------- Priority 2 & 3: Closed Strategy-to-Action Loop & Bounded Idle Curiosity ----------------

    def execute_permitted_strategy_action(
        self,
        query: str,
        *,
        strategy: str | None = None,
        permissions: dict[str, str] | None = None,
        workspace_root: str | Path | None = None,
        memory_store: Any | None = None,
        ontology_graph: Any | None = None,
        sandbox_runner: Callable[[str], tuple[bool, Any, str]] | None = None,
        regression_tests: Sequence[dict[str, Any]] = (),
        budget: IdleResourceBudget | None = None,
        interactive: bool = False,
        question_node: UnresolvedQuestionNode | None = None,
        session_id: str = "",
        turn_n: int = 0,
    ) -> dict[str, Any]:
        """Priority 3: Close the strategy loop by executing the selected permitted epistemic strategy
        (`workspace_code_inspect`, `sandbox_counterexample_probe`, or `memory_graph_recall`) within `budget`
        and `permissions`, measuring its real outcome, and feeding that outcome back into `record_interaction_outcome`.
        Enforces Fix 1 (Verification Integrity) and Fix 2 (Exploration Safety: pre-read byte/file limits,
        content-level secret/PII exclusion, and read-only workspace snapshot verification).
        """
        t0 = time.perf_counter()
        bud = budget or IdleResourceBudget()
        perms = dict(permissions or {"memory": "allow", "files": "allow", "code": "allow", "internet": "ask"})
        ws_snap_before = snapshot_workspace_manifest(workspace_root) if not interactive else {}

        with self.lock:
            rec = self.recommend_strategy_and_voi(
                query,
                permissions=perms,
                interactive=interactive,
            )
            chosen_strat = strategy or rec.get("selected_strategy") or "memory_graph_recall"
            if (
                strategy is None
                and chosen_strat == "sandbox_counterexample_probe"
                and question_node is not None
                and not question_node.memory_entry_id
                and "what_it_consists_of" not in question_node.views_covered
                and "```python" not in (query or "")
                and workspace_root is not None
                and "workspace_code_inspect" in rec.get("permitted_strategies", [])
            ):
                chosen_strat = "workspace_code_inspect"
            req_perm = STRATEGY_REQUIRED_PERMISSION.get(chosen_strat)
            if req_perm is not None:
                p_mode = perms.get(req_perm, "ask")
                if p_mode == "deny" or (not interactive and p_mode != "allow"):
                    return {
                        "executed": False,
                        "strategy": chosen_strat,
                        "reason": f"permission_{req_perm}_{p_mode}",
                        "region_id": rec.get("region_id", "R0"),
                    }

            reg_id = rec.get("region_id", "R0")
            reg = self.regions.get(reg_id, self.regions["R0"])
            prev_err = float(
                question_node.error_history[-1]
                if (question_node and question_node.error_history)
                else (reg.comp_error_window[-1] if reg.comp_error_window else reg.expected_error_yc)
            )
            q_terms = (
                set(extract_emergent_terms(query + " " + query.replace("_", " "), max_terms=16))
                | set(reg.emergent_keywords[:6])
            )

            measured_comp_err = prev_err
            measured_pred_err = prev_err
            view_added: str | None = None
            verif_summary = ""
            promoted_id: str | None = None
            discovered_probe_code: str | None = None
            files_read = 0
            bytes_read = 0
            files_skipped_size_limit = 0
            files_excluded_sensitive = 0
            probe_integrity_report: dict[str, Any] | None = None

            # 1. Strategy: workspace_code_inspect (requires files == "allow", strictly read-only)
            if chosen_strat == "workspace_code_inspect" and workspace_root is not None:
                ws_path = Path(workspace_root)
                if ws_path.is_dir():
                    import ast as _ast
                    matched_defs: list[str] = []
                    discovered_fn_concepts: list[str] = []
                    discovered_ast_rule = ""
                    raw_candidates = [
                        fp
                        for fp in (sorted(ws_path.rglob("*.py")) + sorted(ws_path.rglob("*.md")))
                        if not any(p in ("node_modules", "__pycache__", ".git", "venv") for p in fp.parts)
                        and not fp.is_symlink()
                        and not re.search(r"(?i)(?:secret|token|passw|credential|\.env|\.pem|\.key)", fp.name)
                    ]
                    # Rank by filename overlap with `q_terms` BEFORE opening any file so pre-read budget is spent wisely
                    raw_candidates.sort(
                        key=lambda fp: (
                            -len(
                                q_terms
                                & set(extract_emergent_terms(fp.stem + " " + fp.stem.replace("_", " "), max_terms=12))
                            ),
                            fp.name,
                        )
                    )
                    for fp in raw_candidates:
                        if (time.perf_counter() - t0) * 1000.0 >= bud.max_wall_ms:
                            break
                        # Fix 2A: Enforce file-count and byte-budget limits BEFORE stat/read
                        if files_read >= bud.max_files_read or bytes_read >= bud.max_bytes_read:
                            break
                        try:
                            st = fp.stat()
                        except OSError:
                            continue
                        f_size = int(st.st_size)
                        if f_size <= 0:
                            continue
                        if f_size > bud.max_single_file_bytes or (bytes_read + f_size) > bud.max_bytes_read:
                            files_skipped_size_limit += 1
                            continue
                        try:
                            with open(fp, "rb") as f_in:
                                raw = f_in.read(min(f_size, bud.max_single_file_bytes, bud.max_bytes_read - bytes_read))
                        except OSError:
                            continue
                        # Count every file and byte read immediately upon reading
                        files_read += 1
                        bytes_read += len(raw)
                        txt = raw.decode("utf-8", errors="replace")

                        # Fix 2B: Content-level sensitive data inspection BEFORE any AST parsing or memory storage
                        has_sens, _ = contains_sensitive_content(txt)
                        if has_sens:
                            files_excluded_sensitive += 1
                            continue

                        f_terms = set(
                            extract_emergent_terms(
                                fp.stem + " " + fp.stem.replace("_", " ") + " " + txt[:1500], max_terms=28
                            )
                        )
                        if not (q_terms & f_terms) and len(raw_candidates) > 2:
                            continue

                        if fp.suffix.lower() == ".py":
                            try:
                                tree = _ast.parse(txt)
                                fn_names_in_file: list[str] = []
                                for node in tree.body:
                                    if isinstance(node, _ast.FunctionDef):
                                        fn_names_in_file.append(node.name)
                                        if discovered_probe_code is None:
                                            matched_defs.append(f"{fp.name}:{node.name}")
                                            for c_tok in (fp.stem.lower(), node.name.lower()):
                                                if c_tok not in discovered_fn_concepts:
                                                    discovered_fn_concepts.append(c_tok)
                                # Fix 1 & Step 1: Extract explicit assertions/doctests or build unannotated candidate spec
                                if discovered_probe_code is None and fn_names_in_file:
                                    discovered_probe_code = extract_behavioral_probe_from_python_source(txt, fn_names_in_file)
                                    discovered_ast_rule = extract_ast_rule_summary(txt, fn_names_in_file)
                            except SyntaxError as syn_e:
                                verif_summary = f"SyntaxError in {fp.name}: {syn_e}"
                                measured_comp_err = min(1.0, prev_err + 0.15)
                        else:
                            heads = [ln.strip() for ln in txt.splitlines() if ln.strip().startswith("#")]
                            if heads:
                                matched_defs.append(f"{fp.name}:{heads[0][:60]}")

                    if matched_defs:
                        verif_summary = f"Inspected {files_read} workspace file(s) ({bytes_read} B): found {', '.join(matched_defs[:4])}"
                        hid = "MEM-HYP-WS-" + hashlib.sha1((query + verif_summary).encode("utf-8")).hexdigest()[:8].upper()
                        already_inspected = bool(
                            (question_node is not None and "what_it_consists_of" in question_node.views_covered)
                            or (memory_store is not None and hasattr(memory_store, "entries") and hid in memory_store.entries)
                        )
                        if not already_inspected:
                            view_added = "what_it_consists_of"
                            measured_comp_err = max(0.15, prev_err * 0.55)
                            measured_pred_err = max(0.12, prev_err * 0.60)
                            if (
                                memory_store is not None
                                and perms.get("memory", "allow") == "allow"
                                and hasattr(memory_store, "create_entry")
                            ):
                                import k3_memory_graph as _M
                                body_text = f"Workspace evidence: {verif_summary}"
                                if discovered_ast_rule:
                                    body_text += f"\nAST Rule: {discovered_ast_rule}"
                                if discovered_probe_code:
                                    body_text += f"\n```python\n{discovered_probe_code.strip()}\n```"
                                merged_concepts = list(dict.fromkeys(discovered_fn_concepts + list(q_terms)))[:6]
                                h_entry = _M.MemoryEntry(
                                    id=hid,
                                    title=f"Workspace Hypothesis: {', '.join(matched_defs[:2])}"[:80],
                                    entry_type="hypothesis",
                                    epistemic_status="hypothesis",
                                    status="under_review",
                                    summary=(
                                        f"Unverified workspace hypothesis ({', '.join(matched_defs[:2])}): "
                                        f"{discovered_ast_rule or verif_summary}"
                                    )[:400],
                                    content=body_text,
                                    source_evidence=verif_summary,
                                    provenance=_M.MemoryProvenance(
                                        source_type="observation",
                                        source_ref=f"workspace:{ws_path.name}",
                                        actor="clover-curiosity",
                                    ),
                                    concepts=merged_concepts,
                                    tags=merged_concepts + ["hypothesis", "workspace_inspect"],
                                )
                                memory_store.create_entry(h_entry)
                                if question_node is not None and not question_node.memory_entry_id:
                                    question_node.memory_entry_id = hid
                        else:
                            # Paper 11 (Ivaldi et al. 2014): Re-observing the same static view adds 0 new views (LP = 0)
                            verif_summary = f"Workspace view already collected ({', '.join(matched_defs[:2])}); static repeat yielded 0 new views"
                    else:
                        # Zero views discovered -> measured_comp_err unchanged so SGIM-ACTS learns this strategy stalled
                        verif_summary = (
                            f"Workspace inspection ({files_read} files read, {files_skipped_size_limit} skipped size, "
                            f"{files_excluded_sensitive} excluded sensitive) found no matching definitions"
                        )

            # 2. Strategy: sandbox_counterexample_probe (requires code == "allow")
            elif chosen_strat == "sandbox_counterexample_probe":
                if sandbox_runner is None:
                    import isolated_sandbox as _SBX
                    sandbox_runner = lambda code_s: _SBX.run_in_isolated_sandbox(code_s, timeout_s=bud.max_sandbox_timeout_s)

                target_hyp = None
                probe_code = None
                if memory_store is not None and hasattr(memory_store, "entries"):
                    if question_node and question_node.memory_entry_id and question_node.memory_entry_id in memory_store.entries:
                        target_hyp = memory_store.entries[question_node.memory_entry_id]
                    else:
                        for e in memory_store.entries.values():
                            if getattr(e, "status", "") in ("under_review", "contested") or getattr(e, "entry_type", "") == "hypothesis":
                                e_terms = set(extract_emergent_terms(f"{e.title or ''} {e.summary or ''}", max_terms=10))
                                if q_terms & e_terms:
                                    target_hyp = e
                                    break
                    if target_hyp is not None:
                        m_c = re.search(r"```python\s*\n([\s\S]*?)```", str(getattr(target_hyp, "content", "") or ""))
                        if m_c:
                            probe_code = m_c.group(1).strip()

                if not probe_code:
                    m_q = re.search(r"```python\s*\n([\s\S]*?)```", query or "")
                    if m_q:
                        probe_code = m_q.group(1).strip()

                if probe_code:
                    # Step 1 & Step 2: If the probe came from an unannotated workspace file (`# __CLOVER_UNANNOTATED_PROBE_SPEC__:`),
                    # synthesize and verify its multi-input behavioral assertions in `IsolatedPythonSandbox` first!
                    synthesized_summary: str | None = None
                    synth_res = synthesize_and_verify_behavioral_probe_in_sandbox(probe_code, sandbox_runner)
                    if synth_res.get("was_unannotated"):
                        if synth_res.get("is_noise"):
                            measured_comp_err = 0.90
                            measured_pred_err = 0.90
                            verif_summary = f"Rejected non-deterministic noise function in sandbox ({synth_res.get('reason')})"
                            probe_integrity_report = {"ok": False, "reason": synth_res.get("reason"), "is_noise": True}
                            if target_hyp is not None and perms.get("memory", "allow") == "allow":
                                target_hyp.status = "contested"
                                if hasattr(memory_store, "_save_entry_in_place"):
                                    memory_store._save_entry_in_place(target_hyp)
                            if question_node is not None:
                                question_node.zpd_status = "stalled_deactivated"
                            probe_code = None
                        elif synth_res.get("is_trivial"):
                            measured_comp_err = 0.02
                            measured_pred_err = 0.02
                            verif_summary = f"Trivial constant stub ({synth_res.get('reason')}); region habituated without promotion"
                            probe_integrity_report = {"ok": False, "reason": synth_res.get("reason"), "is_trivial": True}
                            if question_node is not None:
                                question_node.zpd_status = "resolved_verified"
                            probe_code = None
                        elif synth_res.get("ok"):
                            probe_code = str(synth_res["synthesized_code"])
                            synthesized_summary = str(synth_res.get("verified_summary") or "")
                        else:
                            measured_comp_err = min(1.0, max(prev_err, 0.75))
                            verif_summary = f"Unannotated probe synthesis failed: {synth_res.get('reason')}"
                            probe_integrity_report = {"ok": False, "reason": synth_res.get("reason")}
                            probe_code = None

                if probe_code:
                    # Fix 1 (Verification Integrity): Audit the probe AST before accepting any sandbox pass!
                    probe_integrity_report = validate_behavioral_verification_probe(probe_code)
                    if not probe_integrity_report["ok"]:
                        verif_summary = (
                            f"Verification integrity gate rejected probe ({probe_integrity_report['reason']}): "
                            f"hypothesis remains unverified"
                        )
                    else:
                        ok_run, val_run, err_run = sandbox_runner(probe_code)
                        passed_probe = bool(
                            ok_run and (val_run is True or (isinstance(val_run, dict) and val_run.get("passed") is True))
                        )
                        refuted_probe = bool(
                            (isinstance(val_run, dict) and val_run.get("refuted") is True)
                            or (not ok_run and "AssertionError" in str(err_run or ""))
                        )
                        reg_ok = True
                        for rt in regression_tests or ():
                            ok_r, val_r, _ = sandbox_runner(str(rt.get("code") or ""))
                            if not (ok_r and (val_r is True or (isinstance(val_r, dict) and val_r.get("passed") is True))):
                                reg_ok = False
                                break

                        if passed_probe and reg_ok:
                            measured_comp_err = 0.0
                            measured_pred_err = 0.02
                            view_added = "supporting_evidence"
                            verif_summary = (
                                f"Sandbox behavioral probe verified hypothesis "
                                f"({probe_integrity_report['behavioral_checks']} checks, {val_run})"
                            )
                            if target_hyp is not None and perms.get("memory", "allow") == "allow":
                                if not synthesized_summary:
                                    ast_r = extract_ast_rule_summary(probe_code)
                                    assert_cases = [
                                        ln.strip()[len("assert "):].strip()
                                        for ln in probe_code.splitlines()
                                        if ln.strip().startswith("assert ")
                                    ]
                                    if ast_r or assert_cases:
                                        synthesized_summary = (
                                            f"Verified workspace function {ast_r} | "
                                            f"Verified behavioral & boundary cases: {'; '.join(assert_cases[:4])}"
                                        )[:480]
                                if synthesized_summary:
                                    target_hyp.summary = synthesized_summary
                                target_hyp.content = (
                                    f"Verified in IsolatedPythonSandbox ({verif_summary}).\n"
                                    f"Summary: {target_hyp.summary}\n"
                                    f"```python\n{probe_code.strip()}\n```"
                                )
                                target_hyp.epistemic_status = "externally_verified"
                                target_hyp.status = "active"
                                if hasattr(memory_store, "_save_entry_in_place"):
                                    memory_store._save_entry_in_place(target_hyp)
                                    memory_store.rebuild_index()
                                promoted_id = target_hyp.id
                        elif passed_probe and not reg_ok:
                            measured_comp_err = 0.65
                            measured_pred_err = 0.55
                            verif_summary = "Sandbox probe passed locally but broke non-regression replay; rolled back"
                            if target_hyp is not None and perms.get("memory", "allow") == "allow":
                                target_hyp.status = "contested"
                                if hasattr(memory_store, "_save_entry_in_place"):
                                    memory_store._save_entry_in_place(target_hyp)
                        elif refuted_probe:
                            # Discovering a counterexample that refutes a false hypothesis reduces epistemic uncertainty!
                            measured_comp_err = max(0.20, prev_err * 0.65)
                            measured_pred_err = 0.45
                            view_added = "where_it_breaks"
                            verif_summary = f"Counterexample refuted hypothesis in sandbox: {err_run or val_run}"
                            if target_hyp is not None and perms.get("memory", "allow") == "allow":
                                target_hyp.status = "contested"
                                if hasattr(memory_store, "_save_entry_in_place"):
                                    memory_store._save_entry_in_place(target_hyp)
                        else:
                            measured_comp_err = min(1.0, max(prev_err, 0.75))
                            verif_summary = f"Sandbox probe execution failed: {err_run[:140]}"
                else:
                    verif_summary = "No executable behavioral Python probe available yet for region; strategy yielded 0 progress"

            # 3. Strategy: memory_graph_recall (requires memory == "allow")
            elif chosen_strat == "memory_graph_recall" and memory_store is not None and hasattr(memory_store, "entries"):
                matched_entries = []
                for e in memory_store.entries.values():
                    if getattr(e, "status", "") == "superseded":
                        continue
                    e_terms = set(extract_emergent_terms(f"{e.title or ''} {e.summary or ''}", max_terms=12))
                    if len(q_terms & e_terms) >= 1:
                        matched_entries.append(e)
                if len(matched_entries) >= 2:
                    e1, e2 = matched_entries[0], matched_entries[1]
                    existing_targets = {r.target for r in getattr(e1, "relationships", ()) or ()}
                    if e2.id not in existing_targets and hasattr(memory_store, "add_relationship"):
                        try:
                            memory_store.add_relationship(
                                e1.id,
                                "similar_to",
                                e2.id,
                                provenance="curiosity:memory_graph_recall",
                                confidence=0.82,
                                note=f"Linked shared terms {sorted(q_terms)[:4]} during curiosity recall",
                            )
                            view_added = "relationships"
                            measured_comp_err = max(0.18, prev_err * 0.70)
                            measured_pred_err = max(0.15, prev_err * 0.70)
                            verif_summary = f"Linked related memory entries {e1.id} <-> {e2.id}"
                        except Exception:
                            verif_summary = "Memory recall inspected entries; no new link added"
                    else:
                        verif_summary = "Memory entries already linked; recall plateaued"
                else:
                    verif_summary = "Insufficient memory entries in region; recall yielded 0 new views"

            ws_snap_after = snapshot_workspace_manifest(workspace_root) if not interactive else {}
            workspace_untouched = (ws_snap_before == ws_snap_after) if not interactive else True

            episode = self.record_interaction_outcome(
                query,
                pred_error=measured_pred_err,
                comp_error=measured_comp_err,
                pre_repair_comp_error=prev_err if measured_comp_err < prev_err else None,
                strategy_used=chosen_strat,
                actor="curiosity_strategy_loop" if not interactive else "clover",
                controllable=True,
                unresolved_summary=question_node.summary if (question_node and measured_comp_err > 0.15) else None,
                memory_entry_id=promoted_id or (question_node.memory_entry_id if question_node else None),
                view_dimension_added=view_added,
                user_request=query,
                verification_result=verif_summary,
                session_id=session_id,
                turn_n=turn_n,
            )

            return {
                "executed": True,
                "strategy": chosen_strat,
                "region_id": episode["region_id"],
                "comp_error_before": round(prev_err, 4),
                "comp_error_after": round(measured_comp_err, 4),
                "progress_made": round(max(0.0, prev_err - measured_comp_err), 4),
                "view_dimension_added": view_added,
                "promoted_entry_id": promoted_id,
                "probe_integrity": probe_integrity_report,
                "files_read": files_read,
                "bytes_read": bytes_read,
                "files_skipped_size_limit": files_skipped_size_limit,
                "files_excluded_sensitive": files_excluded_sensitive,
                "workspace_untouched": workspace_untouched,
                "verification_result": verif_summary,
                "wall_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "episode": episode,
            }

    def run_bounded_idle_curiosity_step(
        self,
        *,
        is_idle_fn: Callable[[], bool] | None = None,
        permissions: dict[str, str] | None = None,
        workspace_root: str | Path | None = None,
        memory_store: Any | None = None,
        ontology_graph: Any | None = None,
        sandbox_runner: Callable[[str], tuple[bool, Any, str]] | None = None,
        regression_tests: Sequence[dict[str, Any]] = (),
        budget: IdleResourceBudget | None = None,
    ) -> dict[str, Any]:
        """Priority 2: Invoke strictly ONE bounded curiosity step through `studio_idle()`, respecting explicit
        `IdleResourceBudget` and `permissions`.
        """
        t0 = time.perf_counter()
        bud = budget or IdleResourceBudget()
        if is_idle_fn is not None and not is_idle_fn():
            return {"ran": False, "reason": "not_idle"}

        now_ts = time.time()
        if bud.min_idle_interval_s > 0.0 and (now_ts - self.last_idle_step_ts) < bud.min_idle_interval_s:
            return {"ran": False, "reason": "idle_cooldown_active"}

        perms = dict(permissions or {"memory": "allow", "files": "allow", "code": "allow", "internet": "ask"})

        with self.lock:
            # 1. Sync any open hypothesis / unresolved / contested entries from memory_store into ZPD
            synced_cnt = 0
            if memory_store is not None and hasattr(memory_store, "entries"):
                for e in list(memory_store.entries.values()):
                    is_open = (
                        getattr(e, "epistemic_status", "") in ("unresolved", "hypothesis")
                        or getattr(e, "status", "") in ("contested", "under_review")
                    )
                    if not is_open:
                        continue
                    qid = f"Q-MEM-{e.id}"
                    if qid not in self.unresolved_questions:
                        vec = encode_text_to_64d(f"{e.title or ''} {e.summary or ''}")
                        reg = self.locate_leaf_region(vec)
                        rels = getattr(e, "relationships", ()) or ()
                        sup_c = sum(1 for r in rels if getattr(r, "type", "") == "supports")
                        chal_c = sum(1 for r in rels if getattr(r, "type", "") in ("challenges", "unresolved_with"))
                        prereqs = [f"Q-MEM-{r.target}" for r in rels if getattr(r, "type", "") == "depends_on"]
                        diag = compute_neutral_diagnosticity(
                            sup_c, chal_c, is_unverified_hypothesis=(getattr(e, "entry_type", "") == "hypothesis")
                        )
                        self.unresolved_questions[qid] = UnresolvedQuestionNode(
                            question_id=qid,
                            region_id=reg.region_id,
                            summary=str(e.summary or e.title or e.id)[:240],
                            origin_kind="unverified_hypothesis" if getattr(e, "entry_type", "") == "hypothesis" else "memory_contradiction",
                            memory_entry_id=e.id,
                            prerequisite_ids=prereqs,
                            diagnosticity_score=diag,
                        )
                        synced_cnt += 1

            # 2. Pick the single highest-curiosity active ZPD question or unmastered leaf region
            active_qs = [q for q in self.unresolved_questions.values() if q.zpd_status == "active_zpd"]
            active_qs.sort(
                key=lambda q: -(
                    self.regions.get(q.region_id, self.regions["R0"]).state_curiosity * (0.5 + 0.5 * q.diagnosticity_score)
                )
            )
            top_q = active_qs[0] if active_qs else None
            if top_q is not None:
                target_query = top_q.summary
            else:
                # Check if any leaf region has positive state_curiosity and unmastered error
                leaves = [r for r in self.leaf_regions() if r.state_curiosity > 0.005 and r.expected_error_yc >= 0.08 and r.emergent_keywords]
                leaves.sort(key=lambda r: -r.state_curiosity)
                if not leaves:
                    return {
                        "ran": False,
                        "reason": "no_active_curiosity_target",
                        "synced_open_memory_entries": synced_cnt,
                    }
                target_query = " ".join(leaves[0].emergent_keywords[:6])

            if is_idle_fn is not None and not is_idle_fn():
                return {"ran": False, "reason": "interrupted_before_action"}

            self.last_idle_step_ts = now_ts
            action_res = self.execute_permitted_strategy_action(
                target_query,
                permissions=perms,
                workspace_root=workspace_root,
                memory_store=memory_store,
                ontology_graph=ontology_graph,
                sandbox_runner=sandbox_runner,
                regression_tests=regression_tests,
                budget=bud,
                interactive=False,
                question_node=top_q,
                session_id="studio_idle",
                turn_n=self.step_counter + 1,
            )
            return {
                "ran": bool(action_res.get("executed")),
                "synced_open_memory_entries": synced_cnt,
                "target_question_id": top_q.question_id if top_q else None,
                "target_query": target_query,
                "action_result": action_res,
                "budget_max_wall_ms": bud.max_wall_ms,
                "wall_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

    # ---------------- Human-Readable Summary for `clover_repo.py` (`CONTEXT.md`) ----------------

    def summary_markdown(self) -> tuple[str, list[str]]:
        """Return `(headline, markdown_lines)` for `clover_repo.py` `_growth_summary()`."""
        with self.lock:
            leaves = self.leaf_regions()
            leaves.sort(key=lambda r: -r.state_curiosity)
            open_zpd = [q for q in self.unresolved_questions.values() if q.zpd_status == "active_zpd"]
            stalled_zpd = [q for q in self.unresolved_questions.values() if q.zpd_status == "stalled_deactivated"]
            resolved_zpd = [q for q in self.unresolved_questions.values() if q.zpd_status == "resolved_verified"]

            head = (
                f"{len(leaves)} emergent interest regions, {len(open_zpd)} active ZPD questions "
                f"({len(stalled_zpd)} stalled/rerouted, {len(resolved_zpd)} resolved)"
            )
            lines = [
                "## Curiosity & Developmental Frontier",
                "",
                f"- {head}; {self.step_counter} exploration episodes logged.",
            ]
            for r in leaves[:6]:
                kws = ", ".join(r.emergent_keywords[:5]) or "initial root region"
                lines.append(
                    f"- Region `{r.region_id}` ({kws}): curiosity `{r.state_curiosity:.4f}`, "
                    f"LPM `{r.tonic_lpm:.4f}`, CPM `{r.tonic_cpm:.4f}`, expected error `{r.expected_error_yc:.3f}`, "
                    f"controllability `{r.controllability_score:.2f}`, visits `{r.visits}`"
                )
            if open_zpd:
                lines.append("")
                lines.append("### Active Unresolved Questions (ZPD)")
                for q in open_zpd[:6]:
                    lines.append(
                        f"- `{q.question_id}` (`{q.region_id}`, {q.origin_kind}, diagnosticity `{q.diagnosticity_score:.2f}`): "
                        f"{q.summary[:140]}"
                    )
            lines.append("")
            return head, lines


_ENGINES_BY_PATH: dict[str, PersistentCuriosityEngine] = {}
_ENGINES_LOCK = threading.Lock()


def for_path(storage_path: str | Path) -> PersistentCuriosityEngine:
    p = str(Path(storage_path).resolve())
    with _ENGINES_LOCK:
        if p not in _ENGINES_BY_PATH:
            _ENGINES_BY_PATH[p] = PersistentCuriosityEngine(Path(p))
        return _ENGINES_BY_PATH[p]
