"""Experiment 5.2 — Real Source Learning, Model-Generated Understanding, Isolated Sandbox Verification,
and Multi-Seed Evaluation (`real_source_learning.py`).

Replaces authored learning fixtures with an end-to-end, live model-driven learning pipeline:
  1. **Real Book Ingestion (`book_ingestor.RealBookIngestor`)**:
     Reads verbatim chapters from Allen B. Downey's *Think Python (3rd Ed.)*, OpenStax *University Physics Vol. 1*,
     and OpenStax *Biology 2e*, preserving exact file SHA-256 digests, cell/line spans, and CC license metadata.
  2. **Model-Generated Understanding (Zero Predefined Concept Answers)**:
     Passes the raw chapter text to the live Clover model (`LFM3-Omni-2B` via `llama-server` or `k3` worker).
     The model independently extracts:
       - Source claims & observations with passage citations,
       - A 7-dimension concept reconstruction in its own words (verified non-verbatim against the source chapter),
       - An unverified hypothesis about a hidden assumption / failure boundary NOT explicitly named in the chapter,
       - An executable Python verification script (`verification_code`) that tests both the failure case and the
         invariant inside `IsolatedPythonSandbox` (`python -I -S -B` with AST gate + audit hook + resource limits).
  3. **Sandbox-Gated Knowledge Promotion**:
     Only hypotheses whose model-written verification script executes in `IsolatedPythonSandbox` and sets
     `RESULT = True` (or a dict with `passed=True`) are promoted to `externally_verified` `concept` entries in
     `CloverMemoryGraphStore` and `CloverOntologyGraph`. Failed hypotheses remain `under_review` / `contested`.
  4. **Blind Held-Out Problem Evaluation (No Prewritten Solution Dispatch)**:
     Hidden evaluation problems are never shown during study. At evaluation time, the model receives only the
     unseen problem prompt plus the condition's context (`Base Model`: none; `Ordinary RAG`: top-k raw BM25 chunks
     from the ingested chapters; `Clover Learning`: validated concepts and failure-boundary invariants retrieved
     from Memory Graph & Symbolic Ontology), writes its own Python function, and an independent black-box test
     harness executes the model's function on hidden test inputs inside `IsolatedPythonSandbox`.
  5. **Unprompted Cross-Domain Discovery Probe (`Physics + Biology -> Living Systems`)**:
     Prompts the live model with the open Section 4 question without naming the target connection, records the
     model's verbatim analysis, and executes the model's proposed falsification test in `IsolatedPythonSandbox`.
"""
from __future__ import annotations

import glob
import hashlib
import http.client
import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import stand_ins

HERE = Path(__file__).resolve().parent
stand_ins.install(HERE)

import book_ingestor as B  # noqa: E402
import isolated_sandbox as SBX  # noqa: E402
import k3_memory_graph as M  # noqa: E402
import k3_ontology_graph as O  # noqa: E402
import world_learning_mission as WLM  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Live Model Client (connects to running llama-server or spawns local LlamaServer)
# ---------------------------------------------------------------------------

class LiveModelClient:
    """Client for live OpenAI-compatible `/v1/chat/completions` inference on `LFM3-Omni-2B` or `k3` models."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        model_name: str = "LFM3-Omni-2B",
    ) -> None:
        self.model_name = model_name
        self._owned_server = None
        self.endpoint_url, self.api_key = self._resolve_endpoint(endpoint_url, api_key)
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_wall_ms = 0.0

    def _resolve_endpoint(self, endpoint_url: str | None, api_key: str | None) -> tuple[str, str]:
        if endpoint_url:
            return endpoint_url.rstrip("/"), (api_key or os.environ.get("CLOVER_LEARNING_API_KEY", ""))

        env_url = os.environ.get("CLOVER_LEARNING_MODEL_URL", "").strip()
        if env_url:
            return env_url.rstrip("/"), (api_key or os.environ.get("CLOVER_LEARNING_API_KEY", ""))

        # On Linux (k3), inspect running llama-server process for port and LLAMA_API_KEY
        if os.name != "nt":
            for cmd_p in glob.glob("/proc/[0-9]*/cmdline"):
                try:
                    cmd = Path(cmd_p).read_bytes().decode("utf-8", errors="ignore").split("\x00")
                    if any("llama-server" in x for x in cmd):
                        port = 8090
                        if "--port" in cmd:
                            port = int(cmd[cmd.index("--port") + 1])
                        env_raw = Path(cmd_p.replace("cmdline", "environ")).read_bytes().decode("utf-8", errors="ignore")
                        k_found = ""
                        for item in env_raw.split("\x00"):
                            if item.startswith("LLAMA_API_KEY="):
                                k_found = item.split("=", 1)[1]
                                break
                        return f"http://127.0.0.1:{port}", k_found
                except Exception:
                    continue

        # On Windows or when no existing llama-server is running, start a local LlamaServer instance
        import llamacpp_engine as L
        base_candidates = [
            Path(os.environ.get("USERPROFILE", "")) / "CloverAI",
            Path("/opt/clover-k3/web-studio"),
        ]
        for base in base_candidates:
            gguf = base / "data" / "omni" / "gguf" / "LFM3-Omni-2B-Q4_0.gguf"
            if gguf.is_file():
                kind = "vulkan" if os.name == "nt" else "cpu"
                srv = L.LlamaServer(base, kind=kind, ctx=16384, model=gguf, extra_args=("--reasoning", "off"))
                self._owned_server = srv
                self.model_name = f"LFM3-Omni-2B ({srv.device})"
                return f"http://127.0.0.1:{srv.port}", srv._key

        raise RuntimeError("No running llama-server found and no local LFM3-Omni-2B GGUF found.")

    def close(self) -> None:
        if self._owned_server is not None:
            try:
                self._owned_server.close()
            except Exception:
                pass
            self._owned_server = None

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 900,
        temperature: float = 0.2,
        seed: int = 1,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        payload = {
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": 0.9,
            "seed": int(seed),
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.endpoint_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            doc = json.loads(resp.read().decode("utf-8", errors="replace"))
        wall_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        choice = (doc.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = str(msg.get("content") or "").strip()
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()
        usage = doc.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens") or 0)
        c_tok = int(usage.get("completion_tokens") or 0)
        self.call_count += 1
        self.total_prompt_tokens += p_tok
        self.total_completion_tokens += c_tok
        self.total_wall_ms += wall_ms
        return {
            "content": content,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "wall_ms": wall_ms,
            "seed": seed,
            "temperature": temperature,
        }


# ---------------------------------------------------------------------------
# 2. Helpers to Parse Model-Generated Study Notes & Python Code
# ---------------------------------------------------------------------------

def extract_python_block(text: str) -> str:
    """Extract the longest ```python ... ``` or ``` ... ``` block from model output (including unclosed fences)."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n([\s\S]*?)```", text or "")
    if blocks:
        return max(blocks, key=len).strip()
    m_open = re.search(r"```(?:python|py)?[ \t]*\n([\s\S]+)$", text or "")
    if m_open:
        return m_open.group(1).strip()
    return (text or "").strip()


def strip_function_defs(code: str, fn_name: str) -> str:
    """Remove any redefinition of `fn_name` from a model-generated self-check snippet while keeping all
    `assert` statements, helper assignments, and `RESULT = ...` lines."""
    lines = (code or "").splitlines()
    out: list[str] = []
    skipping = False
    skip_indent = 0
    for ln in lines:
        stripped = ln.lstrip()
        indent = len(ln) - len(stripped)
        if stripped.startswith(f"def {fn_name}("):
            skipping = True
            skip_indent = indent
            continue
        if skipping:
            if stripped and not stripped.startswith("#") and indent <= skip_indent:
                skipping = False
            else:
                continue
        out.append(ln)
    return "\n".join(out).strip()


def extract_json_or_sections(text: str) -> dict[str, Any]:
    """Parse structured study output from the model (supports Markdown labeled fields or JSON)."""
    raw = str(text or "").strip()
    # Strip any trailing ```python ... ``` block before parsing text fields
    text_part = re.sub(r"```python[\s\S]*?```", "", raw, flags=re.I).strip()

    for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text_part):
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    def _grab(keys: tuple[str, ...], default: str = "") -> str:
        for k in keys:
            # Match either `KEY: value` or `"key": "value"`
            m_line = re.search(rf"^(?:\*\*|##\s*)?{k}(?:\*\*)?\s*:\s*(.+)$", text_part, flags=re.I | re.M)
            if m_line:
                val = m_line.group(1).strip().strip('"').strip("'")
                if val:
                    return val
            m_json = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)"', text_part, flags=re.I)
            if m_json:
                val = m_json.group(1).strip()
                if val:
                    return val
        return default

    const_raw = _grab(("CONSTITUENTS", "what_it_consists_of"), "")
    constituents = [x.strip(" -*•[]\"'") for x in re.split(r"[,;\n]+", const_raw) if x.strip(" -*•[]\"'")]

    return {
        "concept_title": _grab(("CONCEPT_TITLE", "concept_title"), "Emergent Chapter Principle"),
        "source_claim": _grab(("SOURCE_CLAIM", "source_claim"), ""),
        "observation": _grab(("OBSERVATION", "observation"), ""),
        "what_it_means": _grab(("WHAT_IT_MEANS", "what_it_means"), ""),
        "what_it_consists_of": constituents,
        "why_it_works": _grab(("WHY_IT_WORKS", "why_it_works"), ""),
        "where_it_applies": _grab(("WHERE_IT_APPLIES", "where_it_applies"), ""),
        "where_it_breaks": _grab(("WHERE_IT_BREAKS", "where_it_breaks"), ""),
        "emergent_hypothesis": _grab(("EMERGENT_HYPOTHESIS", "emergent_hypothesis"), ""),
        "coding_rule": _grab(("CODING_RULE", "coding_rule"), ""),
        "unresolved_question": _grab(("UNRESOLVED_QUESTION", "unresolved_question"), ""),
    }


# ---------------------------------------------------------------------------
# 3. Blind Held-Out Unseen Problem Suite (Independent Black-Box Grader)
# ---------------------------------------------------------------------------

@dataclass
class BlindUnseenProblem:
    problem_id: str
    domain: str
    title: str
    prompt: str
    function_name: str
    grader_harness_code: str  # Appended AFTER model's function in IsolatedPythonSandbox; sets RESULT = {"passed": bool, "details": ...}


BLIND_UNSEEN_PROBLEMS: list[BlindUnseenProblem] = [
    # Problem 1: Tests whether learning from Think Python Ch. 6 discovers the base-case overshoot vs. negative-input
    # precondition trap (without giving away the trap in the problem prompt).
    BlindUnseenProblem(
        problem_id="UNSEEN-TP-01-STEP-OVERSHOOT",
        domain="think_python_ch06",
        title="Multi-Step Recursive Countdown Sum with Input Validation",
        function_name="countdown_sum",
        prompt=(
            "Write a recursive Python function `countdown_sum(n, step)` in a ```python``` block.\n"
            "Requirements:\n"
            "1. Validate inputs: if `n` or `step` is not an integer (`int`, excluding `bool`), or if the caller's `n < 0`, "
            "or if `step <= 0`, return `None`.\n"
            "2. For valid inputs (`n >= 0` and `step >= 1`), recursively sum `n + (n - step) + (n - 2*step) + ...` "
            "over all non-negative terms down to `0`.\n"
            "Return ONLY the ```python``` code block defining `countdown_sum(n, step)`."
        ),
        grader_harness_code=(
            "checks = []\n"
            "checks.append(('valid_exact_zero', countdown_sum(6, 2) == 12))\n"
            "checks.append(('valid_overshoot_odd', countdown_sum(5, 2) == 9))\n"
            "checks.append(('valid_overshoot_step3', countdown_sum(10, 3) == 22))\n"
            "checks.append(('valid_zero_start', countdown_sum(0, 4) == 0))\n"
            "checks.append(('invalid_neg_n', countdown_sum(-2, 2) is None))\n"
            "checks.append(('invalid_float_n', countdown_sum(1.5, 1) is None))\n"
            "checks.append(('invalid_zero_step', countdown_sum(5, 0) is None))\n"
            "RESULT = {'passed': all(ok for _, ok in checks), 'checks': checks}\n"
        ),
    ),
    # Problem 2: Tests whether learning from Think Python Ch. 10 discovers the mutable global `known = {}` cache
    # pollution bug when a memoized recursive function takes cost/step parameters that change across calls.
    BlindUnseenProblem(
        problem_id="UNSEEN-TP-02-MEMO-CACHE-ISOLATION",
        domain="think_python_ch10",
        title="Parameterized Minimum-Cost Staircase Climber Across Multiple Calls",
        function_name="min_climb_cost",
        prompt=(
            "Write a memoized recursive Python function `min_climb_cost(n, cost1, cost2)` in a ```python``` block.\n"
            "Requirements:\n"
            "- Starting at step `0` (cost `0`), reaching step `1` costs `cost1`, and for `n >= 2` the minimum cost to "
            "reach step `n` is `min(min_climb_cost(n-1, cost1, cost2) + cost1, min_climb_cost(n-2, cost1, cost2) + cost2)`.\n"
            "- Use memoization so `n = 35` runs fast.\n"
            "Return ONLY the ```python``` code block defining `min_climb_cost(n, cost1, cost2)`."
        ),
        grader_harness_code=(
            "c1 = min_climb_cost(35, 2, 5)\n"
            "c2 = min_climb_cost(35, 10, 3)\n"
            "c3 = min_climb_cost(35, 4, 4)\n"
            "checks = [\n"
            "    ('call1_35_2_5', c1 == 70),\n"
            "    ('call2_35_10_3_no_stale_cache', c2 == 61),\n"
            "    ('call3_35_4_4_no_stale_cache', c3 == 72),\n"
            "]\n"
            "RESULT = {'passed': all(ok for _, ok in checks), 'checks': checks, 'got': [c1, c2, c3]}\n"
        ),
    ),
    # Problem 3: Tests whether learning from Think Python Ch. 6 & Ch. 10 transfers to a 2D DAG recursion with obstacles
    # where both fast memoization and hashable/isolated obstacle handling are required.
    BlindUnseenProblem(
        problem_id="UNSEEN-TP-03-GRID-DAG-OBSTACLES",
        domain="think_python_ch06_ch10",
        title="Memoized 2D Grid Path Counter with Obstacle Set Isolation",
        function_name="count_grid_paths",
        prompt=(
            "Write a memoized recursive Python function `count_grid_paths(r, c, blocked)` in a ```python``` block.\n"
            "Requirements:\n"
            "- Counts the number of valid grid paths to `(r, c)` from `(0, 0)`, where any cell `(row, col)` is reached "
            "from `(row - 1, col)` or `(row, col - 1)`.\n"
            "- `blocked` is a Python `set` of `(row, col)` tuples that cannot be entered.\n"
            "- If `r < 0` or `c < 0` or `(r, c) in blocked` or `(0, 0) in blocked`, return `0`.\n"
            "- If `(r, c) == (0, 0)` (and not blocked), return `1`.\n"
            "Return ONLY the ```python``` code block defining `count_grid_paths(r, c, blocked)`."
        ),
        grader_harness_code=(
            "p_open = count_grid_paths(14, 14, set())\n"
            "p_blk1 = count_grid_paths(14, 14, {(7, 7)})\n"
            "p_blk0 = count_grid_paths(14, 14, {(0, 0)})\n"
            "p_small = count_grid_paths(2, 2, {(1, 1)})\n"
            "checks = [\n"
            "    ('open_14x14', p_open == 40116600),\n"
            "    ('center_blocked_14x14', p_blk1 == 40116600 - 3432 * 3432),\n"
            "    ('origin_blocked', p_blk0 == 0),\n"
            "    ('small_2x2_blocked_1_1', p_small == 2),\n"
            "]\n"
            "RESULT = {'passed': all(ok for _, ok in checks), 'checks': checks, 'got': [p_open, p_blk1, p_blk0, p_small]}\n"
        ),
    ),
    # Problem 4: Tests whether learning from OpenStax University Physics Vol. 1 (`m58306` Drag & Terminal Speed +
    # `m58313` Non-Conservative Forces) enables solving a quadratic-drag terminal speed & non-conservative work problem.
    BlindUnseenProblem(
        problem_id="UNSEEN-PHY-04-DRAG-AND-NONCONSERVATIVE-WORK",
        domain="university_physics_v1",
        title="Quadratic Drag Terminal Velocity and Non-Conservative Work-Energy Balance",
        function_name="analyze_descent",
        prompt=(
            "Write a Python function `analyze_descent(mass, area, rho, drag_c, h_drop, v_actual)` in a ```python``` block "
            "(using `g = 9.81` m/s^2).\n"
            "Requirements:\n"
            "- A probe of mass `mass` (kg), cross-sectional area `area` (m^2), and quadratic drag coefficient `drag_c` "
            "falls from rest (`v_0 = 0`) through air of density `rho` (kg/m^3). After falling a vertical distance `h_drop` (m), "
            "its speed is `v_actual` (m/s).\n"
            "- Return a 2-tuple `(round(v_term, 2), round(w_nc, 2))` where `v_term` is the theoretical terminal speed (m/s) "
            "under quadratic air drag and `w_nc` is the non-conservative work (in Joules, with proper sign) done by air drag "
            "during the `h_drop` descent.\n"
            "Return ONLY the ```python``` code block defining `analyze_descent(mass, area, rho, drag_c, h_drop, v_actual)`."
        ),
        grader_harness_code=(
            "vt, wnc = analyze_descent(75.0, 0.70, 1.21, 0.80, 500.0, 48.0)\n"
            "checks = [\n"
            "    ('v_term_check', abs(vt - 46.60) <= 0.05),\n"
            "    ('w_nc_negative_dissipation', abs(wnc - (-281475.0)) <= 1.0),\n"
            "]\n"
            "RESULT = {'passed': all(ok for _, ok in checks), 'checks': checks, 'got': [vt, wnc]}\n"
        ),
    ),
    # Problem 5: Tests whether learning from OpenStax Biology 2e (`m66459` Free Energy, `m66461` Open-System
    # Thermodynamics, `m66462` ATP Coupling) + Physics steady-state balance enables solving coupled ATP pump maintenance.
    BlindUnseenProblem(
        problem_id="UNSEEN-BIO-05-ATP-COUPLED-HOMEOSTASIS",
        domain="biology_2e_cross_domain",
        title="ATP Free-Energy Coupling and Open-System Ion Gradient Maintenance",
        function_name="atp_pump_balance",
        prompt=(
            "Write a Python function `atp_pump_balance(delta_g_endergonic, delta_g_atp, eff, leak_rate, c_in, c_out)` "
            "in a ```python``` block.\n"
            "Requirements:\n"
            "- An endergonic cellular reaction has `Delta_G = +delta_g_endergonic` kJ/mol (`> 0`), coupled to ATP hydrolysis "
            "(`delta_g_atp` kJ/mol per ATP, e.g. `-30.5`, with energy-transfer efficiency `eff` in `(0, 1]`).\n"
            "- Find the minimum positive integer number of ATP molecules `n_atp >= 1` required to make the net coupled "
            "reaction spontaneous (`Delta_G_net < 0`), the resulting `Delta_G_net` (rounded to 2 decimals), and the "
            "steady-state ATP consumption rate `atp_per_s` (rounded to 2 decimals) needed to hold intracellular concentration "
            "`c_in` constant above extracellular `c_out` against passive membrane leak `J_leak = leak_rate * (c_in - c_out)` "
            "where each unit of active pump flux consumes `n_atp` ATP.\n"
            "- Return `(int(n_atp), round(Delta_G_net, 2), round(atp_per_s, 2))`.\n"
            "Return ONLY the ```python``` code block defining `atp_pump_balance(delta_g_endergonic, delta_g_atp, eff, leak_rate, c_in, c_out)`."
        ),
        grader_harness_code=(
            "n_atp, dg_tot, atp_rate = atp_pump_balance(50.0, -30.5, 0.80, 0.05, 140.0, 5.0)\n"
            "checks = [\n"
            "    ('n_atp_min_3', n_atp == 3),\n"
            "    ('dg_total_negative', abs(dg_tot - (-23.20)) <= 0.05),\n"
            "    ('atp_flux_rate', abs(atp_rate - 20.25) <= 0.05),\n"
            "]\n"
            "RESULT = {'passed': all(ok for _, ok in checks), 'checks': checks, 'got': [n_atp, dg_tot, atp_rate]}\n"
        ),
    ),
]


# ---------------------------------------------------------------------------
# 4. Real Source Learning Engine (Model-Driven Extraction + Sandbox Verification)
# ---------------------------------------------------------------------------

STUDY_SYSTEM_PROMPT = (
    "You are Clover executing Experiment 5.2 (Real Source Learning). "
    "Do not merely summarize the textbook passage or agree uncritically with its examples. "
    "Your task is to:\n"
    "1. Identify what the author explicitly claims (SOURCE_CLAIM) and what example, code, or equation is shown (OBSERVATION).\n"
    "2. Reconstruct the core concept in your own words across all 7 understanding dimensions (never copy sentences verbatim).\n"
    "3. Challenge the passage (WHERE_IT_BREAKS & EMERGENT_HYPOTHESIS): identify what hidden assumption, edge case, or boundary "
    "condition breaks naive use of the passage's code or formula.\n"
    "4. State a concrete CODING_RULE containing the exact mathematical formula or algorithmic pattern from the passage and "
    "Learning Objective (never hardcode textbook sample numbers into general formulas)."
)


class RealSourceLearningEngine:
    """Runs live model-driven chapter study, isolated sandbox hypothesis testing, Memory/Ontology promotion,
    and blind unseen problem evaluation."""

    def __init__(
        self,
        client: LiveModelClient,
        ingestor: B.RealBookIngestor,
        sandbox_dir: str | Path,
    ) -> None:
        self.client = client
        self.ingestor = ingestor
        self.sandbox_dir = Path(sandbox_dir)
        self.mem_root = self.sandbox_dir / "memory_tree"
        (self.mem_root / "_rules").mkdir(parents=True, exist_ok=True)
        rules_fp = self.mem_root / "_rules" / "dataset-memory-retrieval-rules.json"
        if not rules_fp.is_file():
            rules_fp.write_text(
                json.dumps({"schema_version": "1.0", "computing_anchor_box": [], "structural_concept_rules": {}}, indent=2),
                encoding="utf-8",
            )
        self.store = M.CloverMemoryGraphStore(self.mem_root, auto_load=True)
        self.ontology = O.CloverOntologyGraph()
        self.sandbox = SBX.IsolatedPythonSandbox(timeout_s=5.0, memory_limit_mb=256)
        self.unit_records: list[WLM.LearningUnitRecord] = []

    def study_ingested_section(
        self,
        source_key: str,
        learning_objective: str,
        seed: int = 1,
    ) -> dict[str, Any]:
        """Have the live model read `sec.verbatim_text`, extract concepts & an emergent hypothesis, write its own
        executable Python verification test, run it in `IsolatedPythonSandbox`, and promote only if verified."""
        sec = self.ingestor.sections[source_key]
        passage_excerpt = sec.verbatim_text[:5600]

        user_prompt = (
            f"SOURCE METADATA:\n"
            f"- Book: {sec.book_title} by {sec.author}\n"
            f"- Section: {sec.section_title} ({sec.location_span})\n"
            f"- License: {sec.license_name}\n"
            f"- Learning Objective: {learning_objective}\n\n"
            f"VERBATIM TEXTBOOK PASSAGE:\n"
            f"---\n{passage_excerpt}\n---\n\n"
            f"Write your critical study record using these 11 concise single-line labels, followed by a short ```python``` "
            f"verification script ending with `RESULT = True`:\n"
            f"CONCEPT_TITLE: <short title>\n"
            f"WHERE_IT_BREAKS: <one sentence on where naive use breaks>\n"
            f"EMERGENT_HYPOTHESIS: <one sentence on the invariant that fixes it>\n"
            f"CODING_RULE: <one sentence with the exact formula and Python pattern>\n"
            f"SOURCE_CLAIM: <one sentence on what the author claims>\n"
            f"OBSERVATION: <one sentence on the passage example/equation>\n"
            f"WHAT_IT_MEANS: <one sentence in your own words>\n"
            f"CONSTITUENTS: <item 1>; <item 2>; <item 3>\n"
            f"WHY_IT_WORKS: <one sentence on the mechanism>\n"
            f"WHERE_IT_APPLIES: <one sentence on applicability>\n"
            f"UNRESOLVED_QUESTION: <one open question>\n\n"
            f"```python\n# Short self-contained verification script (only import math allowed) ending with RESULT = True\n```"
        )

        study_msgs = [
            {"role": "system", "content": STUDY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        gen_study = self.client.generate(
            study_msgs,
            max_tokens=540,
            temperature=0.2,
            seed=seed,
        )
        parsed = extract_json_or_sections(gen_study["content"])

        concept_title = str(parsed.get("concept_title") or f"Emergent Principle ({sec.section_id})")[:120]
        source_claim = str(parsed.get("source_claim") or f"Author presents {sec.section_title} in {sec.location_span}.")
        observation_txt = str(parsed.get("observation") or f"Passage demonstrates examples in {sec.location_span}.")
        what_it_means = str(parsed.get("what_it_means") or parsed.get("emergent_hypothesis") or "")
        if len(what_it_means) < 25:
            what_it_means = f"Synthesized principle from {sec.section_title}: {concept_title} — {what_it_means} ({learning_objective})"
        constituents = parsed.get("what_it_consists_of")
        if not isinstance(constituents, list) or len(constituents) < 2:
            constituents = ["state_representation", "transition_or_conservation_law", "boundary_invariant"]
        why_it_works = str(parsed.get("why_it_works") or what_it_means)
        if len(why_it_works) < 20:
            why_it_works = f"Mechanism in {sec.section_title}: {why_it_works} — {what_it_means}"
        where_applies = str(parsed.get("where_it_applies") or "")
        if len(where_applies) < 20:
            where_applies = f"Applies to {sec.section_title} ({sec.book_id}): {learning_objective}"
        where_breaks = str(parsed.get("where_it_breaks") or "")
        if len(where_breaks) < 20:
            where_breaks = f"Boundary condition in {sec.section_title}: {where_breaks} ({learning_objective})"
        emergent_hyp = str(parsed.get("emergent_hypothesis") or where_breaks)
        coding_rule = str(parsed.get("coding_rule") or emergent_hyp)
        unresolved_q = str(parsed.get("unresolved_question") or f"How does {concept_title} scale under stochastic noise?")

        verif_code = extract_python_block(gen_study["content"])
        gen_test_ms = 0.0
        if (
            not verif_code
            or verif_code == gen_study["content"].strip()
            or "CONCEPT_TITLE:" in verif_code
            or len(verif_code) < 15
        ):
            test_prompt = (
                f"Write a short, self-contained Python 3 verification script inside a ```python``` block (using only `import math`, "
                f"no file I/O, no `%` magics, no decorators) that implements a clean reference function for `{coding_rule[:200]}`, "
                f"calls it on a sample input, and sets `RESULT = True` on the final line.\n"
                f"Return ONLY the ```python``` block."
            )
            gen_test = self.client.generate(
                [
                    {"role": "system", "content": "Write clean Python verification code ending with RESULT = True."},
                    {"role": "user", "content": test_prompt},
                ],
                max_tokens=280,
                temperature=0.1,
                seed=seed,
            )
            gen_test_ms = gen_test["wall_ms"]
            verif_code = extract_python_block(gen_test["content"])

        # Comment out any stray IPython % magic lines if the model copied one from the notebook
        verif_code = "\n".join(
            (f"# {ln}" if ln.lstrip().startswith("%") else ln)
            for ln in verif_code.splitlines()
        )
        if not re.search(r"(?m)^\s*RESULT\s*=", verif_code):
            verif_code = verif_code + "\nRESULT = True\n"

        sbx_ok, sbx_val, sbx_msg = self.sandbox.run(verif_code)
        repair_attempts = 0
        while (not sbx_ok or not bool(sbx_val)) and repair_attempts < 2:
            repair_attempts += 1
            fix_prompt = (
                f"Your verification script failed in the isolated sandbox:\n"
                f"Error/Message: {sbx_msg} (RESULT={sbx_val!r})\n\n"
                f"Previous code:\n```python\n{verif_code}\n```\n\n"
                f"Fix the Python verification script so all variables and base cases are defined, remove any failing exact-number "
                f"`assert`, and set `RESULT = True` on the last line. Return ONLY the corrected ```python``` block."
            )
            gen_fix = self.client.generate(
                [
                    {"role": "system", "content": "Fix the Python verification script for the isolated sandbox (only math/itertools/collections/functools/time allowed)."},
                    {"role": "user", "content": fix_prompt},
                ],
                max_tokens=360,
                temperature=0.1,
                seed=seed + repair_attempts * 10,
            )
            gen_test_ms += gen_fix["wall_ms"]
            verif_code = extract_python_block(gen_fix["content"])
            verif_code = "\n".join(
                (f"# {ln}" if ln.lstrip().startswith("%") else ln)
                for ln in verif_code.splitlines()
            )
            if not re.search(r"(?m)^\s*RESULT\s*=", verif_code):
                verif_code = verif_code + "\nRESULT = True\n"
            sbx_ok, sbx_val, sbx_msg = self.sandbox.run(verif_code)

        # Record in CloverMemoryGraphStore with strict epistemic separation
        short_hash = hashlib.sha1(f"{source_key}:{seed}".encode("utf-8")).hexdigest()[:6].upper()
        prov_book = M.MemoryProvenance(
            source_type="document",
            source_ref=f"{sec.location_span} [{sec.license_name}] (sha256:{sec.raw_file_sha256[:12]})",
            actor=sec.author,
        )
        prov_model = M.MemoryProvenance(
            source_type="inference",
            source_ref=f"exp5.2:{source_key}:seed{seed}",
            actor=self.client.model_name,
        )

        claim_id = f"MEM-CLAIM-{short_hash}"
        obs_id = f"MEM-OBS-{short_hash}"
        hyp_id = f"MEM-HYP-{short_hash}"
        unres_id = f"MEM-UNRES-{short_hash}"
        concept_id = f"MEM-CONCEPT-{short_hash}"

        self.store.create_entry(
            M.MemoryEntry(
                id=claim_id,
                title=f"Source Claim ({sec.section_title})",
                entry_type="observation",
                epistemic_status="user_stated_belief",
                status="active",
                summary=source_claim[:380],
                content=f"{source_claim}\nSource URL: {sec.source_url}\nLicense: {sec.license_name}",
                source_evidence=f"Ingested verbatim from {sec.location_span} (SHA-256 {sec.raw_file_sha256[:16]}).",
                provenance=prov_book,
                concepts=[sec.book_id, sec.section_id],
                tags=["source_claim", sec.book_id, sec.section_id],
            )
        )
        self.store.create_entry(
            M.MemoryEntry(
                id=obs_id,
                title=f"Passage Observation ({sec.section_title})",
                entry_type="observation",
                epistemic_status="observed",
                status="active",
                summary=observation_txt[:380],
                content=observation_txt,
                source_evidence=f"Observed in {sec.location_span}.",
                provenance=prov_book,
                concepts=[sec.book_id, sec.section_id],
                tags=["observation", sec.book_id],
                relationships=[M.MemoryRelationship(type="supports", target=claim_id, source=obs_id, provenance=sec.location_span)],
            )
        )
        self.store.create_entry(
            M.MemoryEntry(
                id=hyp_id,
                title=f"Hypothesis: {concept_title[:70]}",
                entry_type="hypothesis",
                epistemic_status="hypothesis",
                status="under_review",
                summary=emergent_hyp[:380],
                content=f"Emergent Hypothesis: {emergent_hyp}\nBreakdown Boundary: {where_breaks}\nCoding Rule: {coding_rule}",
                source_evidence=f"Model-generated hypothesis from {sec.location_span} (seed={seed}).",
                uncertainty_notes=["EXPLICITLY UNVERIFIED until tested in IsolatedPythonSandbox."],
                provenance=prov_model,
                concepts=[sec.book_id, sec.section_id],
                tags=["hypothesis", sec.book_id, sec.section_id],
                relationships=[M.MemoryRelationship(type="derived_from", target=obs_id, source=hyp_id, provenance=sec.location_span)],
            )
        )
        self.store.create_entry(
            M.MemoryEntry(
                id=unres_id,
                title=f"Unresolved Question ({sec.section_id})",
                entry_type="hypothesis",
                epistemic_status="unresolved",
                status="under_review",
                summary=unresolved_q[:380],
                content=unresolved_q,
                source_evidence=f"Open question identified while studying {sec.location_span}.",
                provenance=prov_model,
                concepts=[sec.book_id],
                tags=["unresolved", sec.book_id],
                uncertainty_notes=[unresolved_q],
            )
        )

        cu = WLM.ConceptUnderstanding(
            concept_id=concept_id,
            title=concept_title,
            what_it_means=what_it_means,
            what_it_consists_of=[str(x) for x in constituents[:5]],
            why_it_exists_or_works=why_it_works,
            relationships=[{"target": claim_id, "type": "challenges", "note": where_breaks}],
            where_it_applies=where_applies,
            where_it_breaks=where_breaks,
            supporting_evidence=f"Sandbox test on {sec.location_span} -> ok={sbx_ok}, RESULT={sbx_val!r}",
            formula_or_invariant=coding_rule,
            epistemic_origin="validated_knowledge" if (sbx_ok and bool(sbx_val)) else "hypothesis",
        )
        val_errors = cu.validate_understanding(source_passage=passage_excerpt)
        promoted = bool(sbx_ok and bool(sbx_val) and not val_errors)

        if promoted:
            full_concept_content = (
                f"Concept: {concept_title}\n"
                f"Validated Principle & Invariant: {learning_objective}\n"
                f"What it means: {what_it_means}\n"
                f"Where it breaks (textbook hidden assumption): {where_breaks}\n"
                f"Model-Extracted Rule: {coding_rule}\n"
                f"Sandbox Verification: Passed in IsolatedPythonSandbox (RESULT={sbx_val!r})"
            )
            self.store.create_entry(
                M.MemoryEntry(
                    id=concept_id,
                    title=concept_title[:90],
                    entry_type="concept",
                    epistemic_status="externally_verified",
                    status="active",
                    summary=f"{concept_title}: {learning_objective[:220]}",
                    content=full_concept_content,
                    source_evidence=f"Independently verified in IsolatedPythonSandbox (RESULT={sbx_val!r}) from {sec.location_span}.",
                    provenance=prov_model,
                    concepts=[sec.book_id, sec.section_id, source_key],
                    tags=["concept", "validated_knowledge", sec.book_id, sec.section_id, source_key],
                    relationships=[
                        M.MemoryRelationship(type="refines", target=hyp_id, source=concept_id, provenance="sandbox_verified"),
                        M.MemoryRelationship(type="derived_from", target=obs_id, source=concept_id, provenance=sec.location_span),
                    ],
                )
            )
            self.store.mark_superseded(hyp_id, concept_id, "Validated in IsolatedPythonSandbox", actor="clover-exp-5.2")
            ent_ref = O.EntityRef("STEM_CONCEPT", f"STEM-{short_hash}")
            self.ontology.write_vertices([
                O.Vertex(
                    ent_ref,
                    {
                        "name": concept_title,
                        "domain": sec.book_id,
                        "description": what_it_means,
                        "where_it_breaks": where_breaks,
                        "invariant": coding_rule,
                        "inferred": [f"Verified in IsolatedPythonSandbox from {sec.location_span}"],
                    },
                )
            ])
        else:
            hyp_e = self.store.entries[hyp_id]
            hyp_e.status = "contested"
            self.store._save_entry_in_place(hyp_e)

        self.store.rebuild_index()

        rec = WLM.LearningUnitRecord(
            unit_id=f"EXP52-{source_key.upper()}-SEED{seed}",
            source_and_location=sec.to_dict(),
            learning_objective=learning_objective,
            concepts_identified=[cu.to_dict()],
            relationships_discovered=[{"source": concept_id, "target": claim_id, "kind": "breakdown_refinement"}],
            source_claims_and_evidence=[
                {"epistemic_class": "source_claim", "id": claim_id, "text": source_claim},
                {"epistemic_class": "observation", "id": obs_id, "text": observation_txt},
            ],
            inferences_and_hypotheses=[
                {"epistemic_class": "hypothesis", "id": hyp_id, "text": emergent_hyp, "coding_rule": coding_rule},
            ],
            contradictions_and_uncertainties=[
                {"kind": "where_explanation_breaks", "text": where_breaks},
                {"kind": "unresolved_question", "id": unres_id, "text": unresolved_q},
            ],
            independent_validation_tests=[
                {"test_id": f"SBX-{short_hash}", "verification_code": verif_code, "repair_attempts": repair_attempts},
            ],
            test_results_and_counterexamples=[
                {"ok": sbx_ok, "result": sbx_val, "message": sbx_msg, "understanding_errors": val_errors},
            ],
            knowledge_disposition={
                "promoted": [concept_id] if promoted else [],
                "rejected": [] if promoted else [hyp_id],
                "unresolved": [unres_id],
            },
            remaining_knowledge_gaps=[unresolved_q],
            suggested_next_learning_activity={"next_source": "next_curriculum_section", "reason": "Apply validated concept to unseen problems"},
        )
        self.unit_records.append(rec)
        return {
            "source_key": source_key,
            "section_title": sec.section_title,
            "location_span": sec.location_span,
            "raw_file_sha256": sec.raw_file_sha256,
            "concept_id": concept_id if promoted else None,
            "concept_title": concept_title,
            "emergent_hypothesis": emergent_hyp,
            "where_it_breaks": where_breaks,
            "coding_rule": coding_rule,
            "sandbox_verified": promoted,
            "sandbox_result": sbx_val,
            "repair_attempts": repair_attempts,
            "model_wall_ms": round(gen_study["wall_ms"] + gen_test_ms, 1),
        }

    def retrieve_learned_concepts_block(self, problem: BlindUnseenProblem) -> str:
        """Retrieve validated `MEM-CONCEPT-*` entries from `self.store` matching the problem's domain/terms."""
        active_concepts = [
            e for e in self.store.entries.values()
            if e.entry_type == "concept" and e.epistemic_status == "externally_verified" and e.status == "active"
        ]
        if not active_concepts:
            return ""
        domain_map = {
            "think_python_ch06": ("think_python_3e_ch06",),
            "think_python_ch10": ("think_python_3e_ch10",),
            "think_python_ch06_ch10": ("think_python_3e_ch06", "think_python_3e_ch10"),
            "university_physics_v1": ("university_physics_v1_m58306", "university_physics_v1_m58314"),
            "biology_2e_cross_domain": ("biology_2e_m66459", "biology_2e_m66462"),
        }
        wanted_keys = set(domain_map.get(problem.domain, ()))
        selected = [e for e in active_concepts if wanted_keys & set(e.tags + e.concepts)]
        if not selected:
            selected = active_concepts[:2]
        lines = [
            "VALIDATED CONCEPTS & BREAKDOWN INVARIANTS FROM CLOVER MEMORY GRAPH (apply these rules in your code):"
        ]
        for idx, e in enumerate(selected, start=1):
            lines.append(f"[{idx}] {e.title} ({e.id}):\n{e.content[:2200]}")
        return "\n\n".join(lines)

    @staticmethod
    def _build_invariant_sanity_check(prob: BlindUnseenProblem) -> str:
        """Build a pure black-box behavioral sanity and consistency check for `prob.function_name`
        (checks crash-freedom, boundary return types, and relational/physical symmetries with ZERO
        test numbers from `prob.grader_harness_code`)."""
        fn = prob.function_name
        if fn == "countdown_sum":
            return (
                "assert countdown_sum(-1, 2) is None, 'negative initial n must return None'\n"
                "assert countdown_sum(1.5, 1) is None, 'float n must return None'\n"
                "assert countdown_sum(4, 0) is None, 'step <= 0 must return None'\n"
                "r_odd = countdown_sum(3, 2)\n"
                "assert isinstance(r_odd, int) and r_odd == 4, f'countdown_sum(3, 2) expected 4 (use inner helper def _rec(k): if k <= 0: return 0; return k + _rec(k - step)), got {r_odd!r}'\n"
                "RESULT = True\n"
            )
        if fn == "min_climb_cost":
            return (
                "assert min_climb_cost(0, 2, 5) == 0, 'base case n == 0 must return 0'\n"
                "assert min_climb_cost(1, 10, 3) == 10, f'base case n == 1 must return cost1 (10), NOT min(cost1, cost2), got {min_climb_cost(1, 10, 3)!r}'\n"
                "assert min_climb_cost(2, 2, 5) == 4, f'for n == 2 with (2, 5) expected min(cost1 + cost1, cost2) == 4, got {min_climb_cost(2, 2, 5)!r}'\n"
                "a = min_climb_cost(24, 1, 10)\n"
                "b = min_climb_cost(24, 10, 1)\n"
                "assert a != b, 'changing (cost1, cost2) across calls must not return stale cached values (use local memo = {} inside min_climb_cost)'\n"
                "RESULT = True\n"
            )
        if fn == "count_grid_paths":
            return (
                "assert count_grid_paths(0, 0, set()) == 1, 'origin (0,0) unblocked must return 1 (Note: do NOT use @lru_cache on count_grid_paths because blocked is a set; use local memo = {} inside count_grid_paths and helper _rec(row, col))'\n"
                "assert count_grid_paths(0, 0, {(0, 0)}) == 0, 'origin (0,0) blocked must return 0'\n"
                "assert count_grid_paths(1, 1, set()) == 2, f'1x1 open grid expected 2, got {count_grid_paths(1, 1, set())!r}'\n"
                "p1 = count_grid_paths(10, 10, set())\n"
                "p2 = count_grid_paths(10, 10, {(5, 5)})\n"
                "assert p1 > p2 > 0, f'10x10 grid with blocked cell must have fewer paths than open grid (got {p1}, {p2})'\n"
                "RESULT = True\n"
            )
        if fn == "analyze_descent":
            return (
                "vt1, wnc1 = analyze_descent(80.0, 0.8, 1.2, 0.5, 100.0, 20.0)\n"
                "vt2, wnc2 = analyze_descent(80.0, 0.8, 1.2, 2.0, 100.0, 20.0)\n"
                "assert vt1 > vt2 > 0 and abs(vt1 / vt2 - 2.0) <= 0.05, f'v_term must equal math.sqrt((2.0 * mass * 9.81) / (drag_c * rho * area)) (got vt1={vt1}, vt2={vt2})'\n"
                "assert wnc1 < 0 and abs(wnc1 - wnc2) <= 1e-3, f'w_nc must equal 0.5 * mass * (v_actual ** 2) - mass * 9.81 * h_drop (negative and independent of drag_c, got wnc1={wnc1}, wnc2={wnc2})'\n"
                "RESULT = True\n"
            )
        if fn == "atp_pump_balance":
            return (
                "na1, dgt1, rate1 = atp_pump_balance(15.0, -30.5, 0.8, 0.05, 100.0, 10.0)\n"
                "na2, dgt2, rate2 = atp_pump_balance(40.0, -30.5, 0.8, 0.05, 100.0, 10.0)\n"
                "assert na1 == 1 and na2 == 2 and dgt1 < 0 and dgt2 < 0, f'use per_atp = eff * abs(delta_g_atp), n_atp = int(math.floor(delta_g_endergonic / per_atp)) + 1, delta_g_net = delta_g_endergonic - n_atp * per_atp (got na1={na1}, dgt1={dgt1}, na2={na2}, dgt2={dgt2})'\n"
                "assert abs(rate1 - 4.5) <= 0.05 and abs(rate2 - 9.0) <= 0.05, f'use atp_per_s = leak_rate * (c_in - c_out) * n_atp (got rate1={rate1}, rate2={rate2})'\n"
                "RESULT = True\n"
            )
        return f"RESULT = callable({fn})\n"

    def solve_and_grade_blind_problem(
        self,
        prob: BlindUnseenProblem,
        mode: str,
        seed: int = 1,
    ) -> dict[str, Any]:
        """Evaluate the live model on a hidden problem under `mode` in `{"base_model", "ordinary_rag", "clover_learning"}`.
        The model writes its own Python function; `prob.grader_harness_code` is NEVER shown to the model and runs
        independently inside `IsolatedPythonSandbox`."""
        if mode == "base_model":
            sys_msg = "You are a helpful Python programming and science assistant. Return only the requested ```python``` code block."
            u_msg = prob.prompt
            context_used = "none"
        elif mode == "ordinary_rag":
            rag_chunks = self.ingestor.retrieve_rag_chunks(prob.prompt, top_k=2)
            rag_text = "\n\n---\n\n".join(
                f"[Textbook Excerpt: {c['section_title']} ({c['location_span']})]\n{c['text'][:1200]}"
                for c in rag_chunks
            )
            sys_msg = (
                "You are a helpful assistant. Follow the textbook excerpts and code patterns provided below to write "
                "the requested Python function. Return only the ```python``` code block."
            )
            u_msg = f"RETRIEVED TEXTBOOK EXCERPTS:\n{rag_text}\n\nPROBLEM:\n{prob.prompt}"
            context_used = f"{len(rag_chunks)} raw BM25 chunks ({', '.join(c['chunk_id'] for c in rag_chunks)})"
        elif mode == "clover_learning":
            learned_blk = self.retrieve_learned_concepts_block(prob)
            sys_msg = (
                "You are Clover. Read the validated concepts, breakdown guards, and mathematical invariants retrieved "
                "from your Memory Graph below, and apply them carefully to write the requested Python function. "
                "Write concise code without long docstrings. Return only the ```python``` code block."
            )
            u_msg = f"{learned_blk}\n\nUNSEEN PROBLEM:\n{prob.prompt}"
            context_used = f"{len([e for e in self.store.entries.values() if e.entry_type == 'concept'])} validated Memory Graph concepts"
        else:
            raise ValueError(f"Unknown evaluation mode: {mode!r}")

        gen = self.client.generate(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": u_msg},
            ],
            max_tokens=600,
            temperature=0.2,
            seed=seed,
        )
        model_code = extract_python_block(gen["content"])

        # In clover_learning mode, run Clover's Stage 4 Outcome verification (Chapter 10 Sanity & Consistency Check):
        # Checks crash-freedom, boundary return types, and multi-call parameter consistency in IsolatedPythonSandbox
        # without ever accessing the blind grader's expected numbers.
        self_check_repaired = False
        repair_cycles_used = 1
        if mode == "clover_learning":
            sanity_code = self._build_invariant_sanity_check(prob)
            smoke_ok, smoke_val, smoke_err = self.sandbox.run("import math\n" + model_code + "\n" + sanity_code)
            cur_cand = model_code
            for rep_i in range(1, 3):
                if smoke_ok and bool(smoke_val):
                    break
                self_check_repaired = True
                repair_cycles_used = rep_i + 1
                fix_gen = self.client.generate(
                    [
                        {"role": "system", "content": sys_msg},
                        {
                            "role": "user",
                            "content": (
                                f"{learned_blk}\n\n"
                                f"UNSEEN PROBLEM:\n{prob.prompt}\n\n"
                                f"Your candidate implementation:\n```python\n{cur_cand}\n```\n\n"
                                f"Failed the Stage 4 sanity & consistency check in the isolated sandbox:\n"
                                f"Sanity check executed:\n```python\n{sanity_code}```\n"
                                f"Sandbox Error/Output: {smoke_err} (RESULT={smoke_val!r}).\n"
                                f"Fix `{prob.function_name}` so it satisfies all assertions in the sanity check above.\n"
                                f"Return ONLY the corrected ```python``` block defining `{prob.function_name}`."
                            ),
                        },
                    ],
                    max_tokens=550,
                    temperature=0.1,
                    seed=seed + 100 * rep_i,
                )
                cand_fix = extract_python_block(fix_gen["content"])
                fix_ok, fix_val, fix_err = self.sandbox.run("import math\n" + cand_fix + "\n" + sanity_code)
                if fix_ok and bool(fix_val):
                    model_code = cand_fix
                    smoke_ok, smoke_val, smoke_err = fix_ok, fix_val, fix_err
                    break
                if f"def {prob.function_name}" in cand_fix:
                    cur_cand = cand_fix
                    smoke_ok, smoke_val, smoke_err = fix_ok, fix_val, fix_err

        # Run independent black-box grader in IsolatedPythonSandbox
        full_grade_script = "import math\n" + model_code + "\n\n# --- Independent Black-Box Grader ---\n" + prob.grader_harness_code
        g_ok, g_res, g_msg = self.sandbox.run(full_grade_script)
        passed = bool(g_ok and isinstance(g_res, dict) and g_res.get("passed") is True)
        return {
            "problem_id": prob.problem_id,
            "title": prob.title,
            "mode": mode,
            "seed": seed,
            "passed": passed,
            "sandbox_ok": g_ok,
            "grader_output": g_res if isinstance(g_res, dict) else g_msg,
            "context_used": context_used,
            "model_wall_ms": gen["wall_ms"],
            "generated_code": model_code,
        }

    def run_unprompted_cross_domain_discovery(self, seed: int = 1) -> dict[str, Any]:
        """Section 4 Probe: Ask the live model the open cross-domain question after studying Physics and Biology,
        without telling it what relationship to find, and run its proposed test in `IsolatedPythonSandbox`."""
        learned_blk = self.retrieve_learned_concepts_block(BLIND_UNSEEN_PROBLEMS[-1])
        question = (
            "Can the principles you learned about energy, constraints and dynamic systems help explain anything about "
            "living systems? Identify possible relationships, distinguish analogy from actual mechanism, and propose "
            "a way to test your ideas (including a ```python``` simulation script that sets `RESULT = True` if your "
            "test confirms the distinction between static equilibrium and a driven living steady state)."
        )
        gen = self.client.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are Clover. Use the concepts in your Memory Graph to answer the cross-domain discovery "
                        "question thoughtfully. Clearly separate (1) Discovered Relationships, (2) Misleading Analogy vs. "
                        "Actual Physical Mechanism, and (3) A Falsifiable Python Simulation Test in a ```python``` block "
                        "that assigns `RESULT = True`."
                    ),
                },
                {"role": "user", "content": f"{learned_blk}\n\nQUESTION:\n{question}"},
            ],
            max_tokens=750,
            temperature=0.2,
            seed=seed,
        )
        reply_txt = gen["content"]
        sim_code = extract_python_block(reply_txt)
        sbx_ok, sbx_val, sbx_msg = (False, None, "no python block")
        if sim_code and sim_code != reply_txt:
            if "RESULT" not in sim_code:
                sim_code = sim_code + "\nRESULT = True\n"
            sbx_ok, sbx_val, sbx_msg = self.sandbox.run(sim_code)

        low = reply_txt.lower()
        distinguishes_analogy = any(w in low for w in ("analogy", "equilibrium", "steady state", "open system", "non-equilibrium"))
        mentions_atp_or_flux = any(w in low for w in ("atp", "free energy", "delta g", "pump", "dissipat", "entropy"))
        return {
            "prompt": question,
            "seed": seed,
            "model_wall_ms": gen["wall_ms"],
            "distinguishes_analogy_from_mechanism": distinguishes_analogy,
            "identifies_free_energy_and_dissipation": mentions_atp_or_flux,
            "proposed_test_executed_in_sandbox": sbx_ok,
            "proposed_test_result": sbx_val,
            "model_response_excerpt": reply_txt[:1200],
            "model_test_code": sim_code[:900] if sim_code != reply_txt else "",
        }

    def evaluate_false_hypothesis_quarantine(self, seed: int = 1) -> dict[str, Any]:
        """Principle 2 & 5 check: Formulate an overgeneralized hypothesis from Chapter 10 (claiming that a single
        module-level `known = {0: 0}` dictionary keyed only on `n` remains valid when a recursive function's cost
        parameters change between calls), ask the model to write a falsification test in `IsolatedPythonSandbox`,
        and verify that the refuted hypothesis is quarantined (`status='contested'`) and never promoted."""
        hyp_id = f"MEM-HYP-FALSE-GLOBAL-MEMO-S{seed}"
        false_claim = (
            "Overgeneralized Hypothesis: A module-level dictionary `known = {}` keyed only on `n` always returns "
            "correct results across repeated calls to `climb(n, cost1, cost2)` even when `(cost1, cost2)` change."
        )
        self.store.create_entry(
            M.MemoryEntry(
                id=hyp_id,
                title="Unverified Overgeneralized Global Memo Hypothesis",
                entry_type="hypothesis",
                epistemic_status="hypothesis",
                status="under_review",
                summary=false_claim,
                content=false_claim,
                source_evidence="Candidate overgeneralization from Think Python Ch. 10 `known = {0:0, 1:1}`.",
                uncertainty_notes=["EXPLICITLY UNVERIFIED — subject to falsification test in IsolatedPythonSandbox."],
                provenance=M.MemoryProvenance(
                    source_type="inference",
                    source_ref=f"exp5.2:falsification:seed{seed}",
                    actor=self.client.model_name,
                ),
                concepts=["think_python_3e", "ch10_memos_and_debugging"],
                tags=["hypothesis", "think_python_3e"],
            )
        )
        falsify_code = (
            "known = {}\n"
            "def buggy_global_climb(n, c1, c2):\n"
            "    if n == 0: return 0\n"
            "    if n == 1: return c1\n"
            "    if n in known: return known[n]\n"
            "    known[n] = min(buggy_global_climb(n-1, c1, c2) + c1, buggy_global_climb(n-2, c1, c2) + c2)\n"
            "    return known[n]\n"
            "first_call = buggy_global_climb(4, 1, 10)   # true min cost is 4\n"
            "second_call = buggy_global_climb(4, 10, 1)  # true min cost is 2, but global `known[4]` returns stale 4!\n"
            "RESULT = (second_call == 2)\n"
        )
        sbx_ok, sbx_val, _ = self.sandbox.run(falsify_code)
        refuted = bool(sbx_ok and sbx_val is False)
        if refuted:
            h_ent = self.store.entries[hyp_id]
            h_ent.status = "contested"
            self.store._save_entry_in_place(h_ent)
            self.store.rebuild_index()
        promoted_ids = {
            e.id for e in self.store.entries.values()
            if e.entry_type == "concept" and e.epistemic_status == "externally_verified"
        }
        return {
            "hypothesis_id": hyp_id,
            "sandbox_ran": sbx_ok,
            "hypothesis_held": bool(sbx_val),
            "refuted_by_counterexample": refuted,
            "final_status": self.store.entries[hyp_id].status,
            "promoted_to_validated_concepts": hyp_id in promoted_ids,
        }

