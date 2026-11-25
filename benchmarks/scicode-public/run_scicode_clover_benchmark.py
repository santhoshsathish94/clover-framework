#!/usr/bin/env python3
"""Multi-GPU (`4x RTX PRO 6000 Blackwell` + `1 TiB` Zero-Copy RAM + `k3` Copilot Fusion)
Reproducible Public `SciCode` (`AA-SciCode` 288-subproblem) Benchmark Runner for the Clover System.

Executes the full Clover Stack on `Kimi-K3.1-Pro 2.8T` (`K31H` + `5L-DSpark`):
  1. Core `AGENTS.md` Verification Contract (verify before returning, preserve exact signatures & arity,
     never silently swallow errors, zero access to held-out `test_data.h5` or `test_cases`).
  2. `SKILL.md` Scientific & Numerical Research Computing Skill (`scientific_computing_skill.md`).
  3. Native C11 `x-cruncher` Exact Big-Integer & BBP-Pi Bridge (`studio_xcruncher_verifier.py`).
  4. `AST` Contract & Dependency Sanitizer (strips stray top-level test invocations, prevents shadowing
     verified prior steps, checks exact function name, parameter compatibility, and return arity).
  5. `OpenCode` Isolated `nobody` Sandbox + `Elephant-Goldfish (EGM)` / `Copilot Fusion` Differential
     Smoke-Test Verification + `TokenEconomizer` Bounded Prefix-Cached Repair on `Kimi-K3.1-Pro 2.8T`.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import os
import pwd
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

OFFICIAL_PROBLEMS_TEST_SHA256 = "38797fef78f434720be6d053b4f3a86839d6f8ea5fb9115450677cd3a6edf81d"
OFFICIAL_PROBLEMS_DEV_SHA256 = "193968aff23b7ed931c8f6d196b611e2f6694adb49e6cc085444a60ad2fc4f7b"
STARTER_STEPS = {("13", 6), ("62", 1), ("76", 3)}

# Official Artificial Analysis / Tian et al. `multistep_template.txt` (with_background=True)
AA_SCICODE_BACKGROUND_PROMPT = """PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```."""

CLOVER_SYSTEM_PROMPT = """You are operating under Clover's AGENTS.md engineering specification:
- Implement ONLY the requested next-step Python function(s), adhering strictly to the provided function header, parameter order, default arguments, and exact `return` structure.
- Derive formulas carefully from the provided scientific background (pay close attention to factors of 2*pi, hbar, signs, boundary indices, broadcasting shapes, and 0-based vs 1-based indexing).
- Cast inputs to float/complex NumPy arrays before in-place division or linear algebra so integer inputs never silently floor-divide or truncate complex phases.
- Do NOT redefine functions from previous steps and do NOT include top-level example/test calls. Output a single ```python``` code block."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_zero_leakage_sandbox(sealed_h5: Path) -> None:
    """Prove OS-level isolation: UID `nobody` cannot read `/root/sealed_scicode_oracle/test_data.h5`."""
    nobody_pw = pwd.getpwnam("nobody")
    r = subprocess.run(
        [sys.executable, "-c", f"open({str(sealed_h5)!r}, 'rb')"],
        user=nobody_pw.pw_uid,
        group=nobody_pw.pw_gid,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if r.returncode == 0:
        raise RuntimeError("SECURITY / LEAKAGE VIOLATION: `nobody` was able to read sealed test_data.h5!")


def load_sanitized_problems(jsonl_path: Path, split: str) -> list[dict[str, Any]]:
    """Load SciCode problems and immediately strip `ground_truth_code` and `test_cases` from memory."""
    actual_sha = sha256_file(jsonl_path)
    expected = OFFICIAL_PROBLEMS_TEST_SHA256 if split == "test" else OFFICIAL_PROBLEMS_DEV_SHA256
    if actual_sha != expected:
        raise RuntimeError(f"SHA-256 mismatch on {jsonl_path}: {actual_sha} != {expected}")

    clean: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        steps = []
        for s in raw["sub_steps"]:
            steps.append({
                "step_number": str(s["step_number"]),
                "step_description_prompt": str(s["step_description_prompt"]),
                "step_background": str(s.get("step_background") or ""),
                "function_header": str(s["function_header"]),
                "return_line": str(s["return_line"]),
            })
        clean.append({
            "problem_id": str(raw["problem_id"]),
            "required_dependencies": str(raw["required_dependencies"]),
            "sub_steps": steps,
        })
    return clean


_FENCE_PY_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_TOP_SYMBOL_RE = re.compile(r"^\s*(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\(:]", re.MULTILINE)


def extract_func_name(header: str) -> str:
    m = _TOP_SYMBOL_RE.search(header or "")
    return m.group(1) if m else ""


def expected_return_arity(return_line: str) -> int | None:
    """Infer tuple arity from SciCode's `return a, b` template when unambiguous."""
    s = (return_line or "").strip()
    if not s.startswith("return"):
        return None
    try:
        mod = ast.parse(f"def _f():\n    {s}\n")
        ret = mod.body[0].body[0]
        if isinstance(ret, ast.Return) and ret.value is not None:
            if isinstance(ret.value, ast.Tuple):
                return len(ret.value.elts)
            return 1
    except SyntaxError:
        pass
    return None


def extract_and_sanitize_code(
    raw_response: str,
    target_func: str,
    prev_func_names: set[str],
) -> tuple[str, list[str]]:
    """Extract Python code block and strip stray top-level test/print/main statements & shadowed prior defs."""
    issues: list[str] = []
    matches = _FENCE_PY_RE.findall(raw_response or "")
    if matches:
        # Prefer the fenced block defining `target_func` (`def` or `class`), else last fenced block
        chosen = matches[-1]
        for blk in reversed(matches):
            if target_func and re.search(rf"\b(?:class|def)\s+{re.escape(target_func)}\s*[\(:]", blk):
                chosen = blk
                break
        code = chosen.strip()
    else:
        # Fallback if model omitted closing ``` fence due to length
        s = raw_response or ""
        if "```python" in s:
            code = s.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in s:
            code = s.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            code = s.strip()

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return code, [f"SyntaxError at line {e.lineno}: {e.msg}"]

    kept_nodes: list[ast.stmt] = []
    defined_syms: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = {}
    for node in tree.body:
        # Drop `if __name__ == "__main__":` blocks
        if isinstance(node, ast.If):
            src_seg = ast.get_source_segment(code, node) or ""
            if "__name__" in src_seg:
                continue
        # Drop bare top-level expressions (e.g. accidental example calls or print() at bottom, except module docstrings)
        if isinstance(node, ast.Expr) and not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.Assert):
            continue
        # Drop accidental re-definition of an earlier step's function/class so we never shadow verified earlier steps
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in prev_func_names and node.name != target_func:
                continue
            defined_syms[node.name] = node
        kept_nodes.append(node)

    if target_func and target_func not in defined_syms:
        issues.append(f"Target symbol `{target_func}` was not defined at module top level.")
    elif target_func in defined_syms and isinstance(defined_syms[target_func], (ast.FunctionDef, ast.AsyncFunctionDef)):
        fn_node = defined_syms[target_func]
        has_return = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn_node))
        if not has_return:
            issues.append(f"Function `{target_func}` has no value-returning `return` statement.")

    cleaned_mod = ast.Module(body=kept_nodes, type_ignores=[])
    try:
        cleaned_code = ast.unparse(cleaned_mod)
    except Exception:
        cleaned_code = code
    return cleaned_code, issues


def run_nobody_sandbox_check(
    dependencies: str,
    previous_code: str,
    candidate_code: str,
    target_func: str,
    exp_arity: int | None,
    smoke_snippet: str | None,
    ref_impl_code: str | None,
    timeout_s: int = 12,
) -> tuple[bool, str]:
    """Run zero-leakage sandbox check as unprivileged `nobody` user (cannot read /root/sealed_scicode_oracle)."""
    nobody_pw = pwd.getpwnam("nobody")
    shim = (
        "import warnings\nwarnings.filterwarnings('ignore')\n"
        "import scipy.integrate\n"
        "if not hasattr(scipy.integrate, 'simps'):\n"
        "    scipy.integrate.simps = scipy.integrate.simpson\n"
    )
    base_prog = f"{shim}\n{dependencies}\n\n{previous_code}\n\n{candidate_code}\n"
    base_check = base_prog + f"\nassert callable({target_func}), '{target_func} is not callable'\n"

    with tempfile.TemporaryDirectory(prefix="clover_sbx_") as td:
        os.chmod(td, 0o755)
        p_mod = Path(td) / "check_load.py"
        p_mod.write_text(base_check, encoding="utf-8")
        os.chmod(p_mod, 0o644)
        r = subprocess.run(
            [sys.executable, str(p_mod)],
            user=nobody_pw.pw_uid,
            group=nobody_pw.pw_gid,
            cwd=td,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip()[-700:]
            return False, f"Module import/definition check failed:\n{err}"

        # Optional Copilot Fusion + Elephant blind synthetic smoke test
        if smoke_snippet and smoke_snippet.strip():
            # If we also have k3's independent reference implementation, verify `smoke_snippet` runs cleanly on it
            # first so an invalid synthetic input never causes a false alarm on Kimi's code!
            if ref_impl_code and ref_impl_code.strip():
                ref_prog = (
                    f"{shim}\n{dependencies}\n\n{previous_code}\n\n{ref_impl_code}\n\n"
                    f"_res = (lambda:\n" + "\n".join("    " + ln for ln in smoke_snippet.strip().splitlines()) + "\n)()\n"
                )
                p_ref = Path(td) / "check_ref.py"
                p_ref.write_text(ref_prog, encoding="utf-8")
                os.chmod(p_ref, 0o644)
                try:
                    rr = subprocess.run(
                        [sys.executable, str(p_ref)],
                        user=nobody_pw.pw_uid,
                        group=nobody_pw.pw_gid,
                        cwd=td,
                        capture_output=True,
                        text=True,
                        timeout=timeout_s,
                    )
                    if rr.returncode != 0:
                        # Synthetic smoke input failed on independent reference too -> skip synthetic test!
                        return True, "ok_load_only"
                except subprocess.TimeoutExpired:
                    return True, "ok_load_only"

            arity_chk = ""
            if exp_arity is not None and exp_arity > 1:
                arity_chk = (
                    f"assert isinstance(_out, tuple) and len(_out) == {exp_arity}, "
                    f"f'Expected return tuple of length {exp_arity}, got {{type(_out).__name__}}'\n"
                )
            elif exp_arity == 1:
                arity_chk = "assert _out is not None, 'Function returned None instead of expected value'\n"

            cand_smoke = (
                base_prog
                + "\nimport numpy as _np\n"
                + smoke_snippet.strip()
                + "\n"
                + arity_chk
                + "def _chk_fin(v):\n"
                + "    if isinstance(v, (tuple, list)):\n"
                + "        for x in v: _chk_fin(x)\n"
                + "    elif isinstance(v, (int, float, complex, _np.ndarray, _np.generic)):\n"
                + "        arr = _np.asarray(v)\n"
                + "        if arr.size > 0 and (_np.issubdtype(arr.dtype, _np.floating) or _np.issubdtype(arr.dtype, _np.complexfloating)):\n"
                + "            assert _np.all(_np.isfinite(arr)), f'Output contains NaN or Inf: {arr}'\n"
                + "_chk_fin(_out)\n"
            )
            p_smk = Path(td) / "check_smoke.py"
            p_smk.write_text(cand_smoke, encoding="utf-8")
            os.chmod(p_smk, 0o644)
            try:
                rs = subprocess.run(
                    [sys.executable, str(p_smk)],
                    user=nobody_pw.pw_uid,
                    group=nobody_pw.pw_gid,
                    cwd=td,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                if rs.returncode != 0:
                    err = (rs.stderr or rs.stdout or "").strip()[-700:]
                    return False, f"Synthetic runtime smoke test failed:\n{err}"
            except subprocess.TimeoutExpired:
                return False, f"Synthetic runtime smoke test timed out after {timeout_s}s (infinite loop?)."

    return True, "ok"


def build_scicode_step_prompt(
    prob: dict[str, Any],
    step_num_1based: int,
    prev_codes: list[str],
) -> tuple[str, str]:
    """Exact SciCode `with_background=True` prompt builder matching `ScicodePromptingAssistant`."""
    output_lines: list[str] = []
    prev_code_list: list[str] = []
    for i in range(step_num_1based - 1):
        sub = prob["sub_steps"][i]
        desc = sub["step_description_prompt"] + "\n" + sub["step_background"]
        output_lines.append(desc)
        output_lines.append(prev_codes[i])
        prev_code_list.append(prev_codes[i])
        output_lines.append("------")

    cur_sub = prob["sub_steps"][step_num_1based - 1]
    next_desc = cur_sub["step_description_prompt"] + "\n" + cur_sub["step_background"]
    next_code_hdr = f"{cur_sub['function_header']}\n\n{cur_sub['return_line']}"
    problem_steps_str = "\n\n".join(output_lines[:-1])
    next_step_str = f"{next_desc}\n\n{next_code_hdr}"
    deps = prob["required_dependencies"]
    prompt = AA_SCICODE_BACKGROUND_PROMPT.format(
        problem_steps_str=problem_steps_str,
        next_step_str=next_step_str,
        dependencies=deps,
    )
    previous_code_full = f"{deps}\n" + "\n\n".join(prev_code_list) + "\n"
    return prompt, previous_code_full


@dataclass
class _GenRes:
    text: str
    reasoning: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float
    prefill_ms: float
    decode_ms: float
    gen_ids: list[int] | None = None


class OpenAICompatBackend:
    """OpenAI-compatible HTTP backend (OpenRouter or local k3 `hg4-fusion-gateway` :8790)."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str = "medium",
        timeout_s: int = 240,
    ) -> None:
        self.endpoint = api_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s

    def generate(self, msgs: list[dict[str, Any]], _a: Any, _b: Any, params: Any) -> _GenRes:
        max_tok = getattr(params, "max_tokens", 6144)
        temp = getattr(params, "temperature", 0.0)
        efforts = [self.reasoning_effort, "low"] if self.reasoning_effort != "none" else ["none"]

        last_err = ""
        for attempt in range(15):
            eff = efforts[min(attempt // 5, len(efforts) - 1)]
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": msgs,
                "temperature": temp,
                "max_tokens": max_tok if attempt < 2 else max(max_tok, 8192),
            }
            if eff and eff != "none" and "openrouter.ai" in self.endpoint:
                if max_tok <= 768:
                    payload["reasoning"] = {"effort": "low"}
                elif eff.isdigit():
                    payload["reasoning"] = {"max_tokens": int(eff)}
                else:
                    payload["reasoning"] = {"effort": eff}

            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://cloverframework.com/hg-format/",
                    "X-Title": "Clover-SciCode-Public-Benchmark",
                },
                method="POST",
            )
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                dt_ms = (time.perf_counter() - t0) * 1000.0
                choices = body.get("choices") or []
                if not choices:
                    last_err = f"No choices in response: {str(body)[:300]}"
                    time.sleep(1.5 * (attempt + 1))
                    continue
                msg = choices[0].get("message") or {}
                content = str(msg.get("content") or "")
                reasoning = str(msg.get("reasoning") or "")
                if not content.strip() and attempt < 4:
                    last_err = f"Empty content (finish_reason={choices[0].get('finish_reason')})"
                    time.sleep(1.0 * (attempt + 1))
                    continue
                u = body.get("usage") or {}
                pt_det = u.get("prompt_tokens_details") or {}
                ct_det = u.get("completion_tokens_details") or {}
                return _GenRes(
                    text=content,
                    reasoning=reasoning,
                    prompt_tokens=int(u.get("prompt_tokens") or 0),
                    cached_tokens=int(pt_det.get("cached_tokens") or 0),
                    completion_tokens=int(u.get("completion_tokens") or 0),
                    reasoning_tokens=int(ct_det.get("reasoning_tokens") or 0),
                    cost_usd=float(u.get("cost") or 0.0),
                    prefill_ms=0.0,
                    decode_ms=dt_ms,
                    gen_ids=None,
                )
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                err_txt = str(e)
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        err_txt += " " + e.read().decode("utf-8", errors="ignore")[:250]
                    except Exception:
                        pass
                last_err = err_txt
                if "429" in err_txt:
                    time.sleep(8.0 + 3.0 * (attempt % 5))
                else:
                    time.sleep(2.0 * min(attempt + 1, 5))
        raise RuntimeError(f"API generation failed after retries: {last_err}")

    def remember_assistant(self, *args: Any, **kwargs: Any) -> None:
        pass


class _SimpleGenParams:
    def __init__(self, max_tokens: int = 6144, temperature: float = 0.0) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature


def worker_loop(
    worker_id: int,
    gpu_id: int | str,
    cpu_lo: int,
    cpu_hi: int,
    prob_queue: "mp.Queue[dict[str, Any] | None]",
    scicode_data_dir: str,
    out_root: str,
    fusion_hints_path: str | None,
    think_cap: int,
    kimi_self_smoke: bool = False,
    backend_mode: str = "native",
    api_url: str = "https://openrouter.ai/api/v1",
    api_key_file: str = "/dev/shm/.or_key",
    api_model: str = "moonshotai/kimi-k3.1-pro",
    reasoning_effort: str = "medium",
    clover_stack_dir: str = "/workspace/clover_stack",
) -> None:
    """SciCode Clover System Worker (supports native HG4/K31H engine or OpenAI-compatible API)."""
    if "GOMP_CPU_AFFINITY" not in os.environ:
        try:
            os.sched_setaffinity(0, set(range(cpu_lo, cpu_hi + 1)))
        except Exception:
            pass

    stack_p = Path(clover_stack_dir)
    for cand_gw in [
        stack_p / "hg4-fusion-gateway",
        stack_p / "clover-ai-system" / "src" / "tools" / "hg4-fusion-gateway",
        Path("/workspace/clover-framework/clover-ai-system/src/tools/hg4-fusion-gateway"),
        Path("/opt/clover-k3/hg4-fusion-gateway"),
    ]:
        if cand_gw.is_dir() and str(cand_gw) not in sys.path:
            sys.path.insert(0, str(cand_gw))

    try:
        import studio_xcruncher_verifier as sxc
    except Exception:
        sxc = None  # type: ignore[assignment]

    skill_txt = ""
    for cand_sk in [
        stack_p / "scientific_computing_skill.md",
        Path("/workspace/clover-framework/clover-ai-system/src/tools/hg4-fusion-gateway/scientific_computing_skill.md"),
        Path("/opt/scicode_bench/scientific_computing_skill.md"),
    ]:
        if cand_sk.is_file():
            skill_txt = cand_sk.read_text(encoding="utf-8")
            break
    sys_prompt = CLOVER_SYSTEM_PROMPT + ("\n\n" + skill_txt if skill_txt else "")

    fusion_hints: dict[str, dict[str, str]] = {}
    if fusion_hints_path and Path(fusion_hints_path).is_file():
        try:
            fusion_hints = json.loads(Path(fusion_hints_path).read_text(encoding="utf-8"))
        except Exception:
            fusion_hints = {}

    if backend_mode == "native":
        os.environ["CLOVER_CUDA_GPUS"] = str(gpu_id)
        os.environ["CLOVER_MMAP_COLD"] = "1"
        os.environ["CLOVER_NATIVE_LIB_DIR"] = os.environ.get("CLOVER_NATIVE_LIB_DIR", str(stack_p))
        os.environ["CLOVER_KIMI_HG4_DIR"] = os.environ.get("CLOVER_KIMI_HG4_DIR", "/workspace/hg/kimi-k31")
        os.environ["CLOVER_KIMI_COLD_DIR"] = os.environ.get("CLOVER_KIMI_COLD_DIR", "/workspace/hg/kimi-k31")
        os.environ["CLOVER_KIMI_PREFER_NATIVE"] = "1"
        os.environ["CLOVER_KIMI_THINKING"] = "bounded"
        os.environ["CLOVER_KIMI_THINK_CAP"] = str(think_cap)
        os.environ.setdefault("CLOVER_NEMO_SPECULATIVE", "1")
        os.environ.setdefault("OMP_NUM_THREADS", str(min(64, max(8, cpu_hi - cpu_lo + 1))))
        import backends as bk
        paths = bk.discover_paths()
        be: Any = bk.make_native_backend(api_model, paths[api_model])
        GenParamsCls: Any = bk.GenParams
        default_max_tok = 1024
        print(f"[Worker {worker_id} | GPU {gpu_id}] Initialized native `{api_model}` backend.", flush=True)
    else:
        api_key = Path(api_key_file).read_text(encoding="utf-8").strip() if Path(api_key_file).is_file() else os.environ.get("OPENROUTER_API_KEY", "local")
        be = OpenAICompatBackend(api_url=api_url, api_key=api_key, model=api_model, reasoning_effort=reasoning_effort)
        GenParamsCls = _SimpleGenParams
        default_max_tok = 6144
        print(f"[Worker {worker_id}] Initialized `{backend_mode}` backend (`{api_model}`).", flush=True)

    mslug = re.sub(r"[^a-zA-Z0-9]+", "_", api_model.split("/")[-1]).strip("_").lower()
    out_base = Path(out_root)
    dir_clover = out_base / "submissions" / f"clover_system_{mslug}" / "with_background"
    dir_turn1  = out_base / "submissions" / f"{mslug}_turn1_draft" / "with_background"
    dir_meta   = out_base / "telemetry"
    dir_clover.mkdir(parents=True, exist_ok=True)
    dir_turn1.mkdir(parents=True, exist_ok=True)
    dir_meta.mkdir(parents=True, exist_ok=True)
    data_dir = Path(scicode_data_dir)

    from scicode.parse.parse import get_function_from_code

    while True:
        prob = prob_queue.get()
        if prob is None:
            break
        pid = prob["problem_id"]
        done_marker = dir_meta / f"prob_{pid}.json"
        if done_marker.is_file():
            print(f"[Worker {worker_id}] Problem {pid} already checkpointed, skipping.", flush=True)
            continue

        # Reload k3 Copilot Fusion / Elephant smoke hints dynamically as k3 streams updates
        if fusion_hints_path and Path(fusion_hints_path).is_file():
            try:
                fusion_hints = json.loads(Path(fusion_hints_path).read_text(encoding="utf-8"))
            except Exception:
                pass

        t_prob0 = time.perf_counter()
        tot_steps = len(prob["sub_steps"])
        prev_codes: list[str] = [""] * tot_steps
        prev_func_names: set[str] = set()
        step_logs: list[dict[str, Any]] = []

        for idx in range(tot_steps):
            step_num = idx + 1
            sub = prob["sub_steps"][idx]
            step_id = sub["step_number"]
            target_fn = extract_func_name(sub["function_header"])

            # Official SciCode v1.0.1 starter steps (`13.6` class Maxwell, `62.1` class EnlargedBlock, `76.3` def)
            if (pid, step_num) in STARTER_STEPS:
                starter_txt = (data_dir / f"{pid}.{step_num}.txt").read_text(encoding="utf-8")
                if sub["function_header"].lstrip().startswith("class "):
                    fn_code = starter_txt.strip()
                else:
                    fn_code = get_function_from_code(starter_txt, target_fn) or starter_txt.strip()
                prev_codes[idx] = fn_code
                if target_fn:
                    prev_func_names.add(target_fn)
                continue

            prompt, prev_full = build_scicode_step_prompt(prob, step_num, prev_codes)
            msgs: list[dict[str, Any]] = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
            # Inject exact x-cruncher math values if high-precision integer/factorial/power expressions appear
            if sxc is not None:
                msgs, xc_meta = sxc.apply_pre_xcruncher_context(msgs)
            else:
                xc_meta = {}

            # Turn 1: Primary Generation
            t_s0 = time.perf_counter()
            res1 = be.generate(msgs, None, None, GenParamsCls(max_tokens=default_max_tok, temperature=0.0))
            be.remember_assistant(res1.text, None, getattr(res1, "gen_ids", None), msgs)
            code1, ast_issues1 = extract_and_sanitize_code(res1.text, target_fn, prev_func_names)
            step_cost_usd = float(getattr(res1, "cost_usd", 0.0))

            # Save Turn-1 un-repaired snapshot
            (dir_turn1 / f"{pid}.{step_num}.py").write_text(f"{prev_full}\n{code1}\n", encoding="utf-8")

            exp_arity = expected_return_arity(sub["return_line"])
            hint = fusion_hints.get(step_id) or {}
            smoke_snip = hint.get("smoke_test")
            ref_impl   = hint.get("ref_impl")

            # If no external k3 smoke snippet exists yet and --kimi-self-smoke is enabled, synthesize via LCP
            if not smoke_snip and kimi_self_smoke and not ast_issues1 and target_fn:
                smk_msgs = msgs + [
                    {"role": "assistant", "content": res1.text},
                    {
                        "role": "user",
                        "content": (
                            f"Write ONLY 2 to 5 lines of Python inside ```python``` that construct small valid "
                            f"synthetic arguments matching `{sub['function_header']}`'s docstring and assign "
                            f"`_out = {target_fn}(...)`. Do NOT redefine functions and do NOT write asserts."
                        ),
                    },
                ]
                res_smk = be.generate(smk_msgs, None, None, GenParamsCls(max_tokens=512 if backend_mode != "native" else 192, temperature=0.0))
                step_cost_usd += float(getattr(res_smk, "cost_usd", 0.0))
                smk_m = _FENCE_PY_RE.findall(res_smk.text or "")
                cand_smk = (smk_m[-1] if smk_m else (res_smk.text or "")).strip()
                if f"{target_fn}(" in cand_smk and "_out" in cand_smk:
                    try:
                        ast.parse(cand_smk)
                        smoke_snip = cand_smk
                    except SyntaxError:
                        smoke_snip = None

            final_code = code1
            repaired = False
            repair_reason = ""

            if ast_issues1:
                ok1, sbx_msg1 = False, "; ".join(ast_issues1)
            else:
                ok1, sbx_msg1 = run_nobody_sandbox_check(
                    prob["required_dependencies"],
                    "\n\n".join(c for c in prev_codes[:idx] if c),
                    code1,
                    target_fn,
                    exp_arity,
                    smoke_snip,
                    ref_impl,
                )
                # Guard against false-positive synthetic call typos: if failure occurred strictly at top-level
                # setup before entering `target_fn` (and wasn't an arity/NaN assertion), treat as load-ok.
                if (
                    not ok1
                    and ref_impl is None
                    and "Synthetic runtime smoke test failed:" in sbx_msg1
                    and f"in {target_fn}" not in sbx_msg1
                    and "Expected return tuple" not in sbx_msg1
                    and "returned None" not in sbx_msg1
                    and "NaN or Inf" not in sbx_msg1
                ):
                    ok1, sbx_msg1 = True, "ok_ignored_caller_setup_err"

            # Turn 2 (OpenCode + TokenEconomizer Targeted Prefix-Cached Repair on Kimi-K3.1-Pro if Check 1 failed)
            if not ok1:
                repair_reason = sbx_msg1[:500]
                repair_msgs = msgs + [
                    {"role": "assistant", "content": res1.text},
                    {
                        "role": "user",
                        "content": (
                            f"Executing `{target_fn}` in an isolated Python sandbox failed with:\n"
                            f"{repair_reason}\n\n"
                            f"Fix `{target_fn}` while strictly preserving:\n"
                            f"{sub['function_header']}\n{sub['return_line']}\n"
                            f"Return ONLY the corrected complete Python code for `{target_fn}` inside ```python```."
                        ),
                    },
                ]
                res2 = be.generate(repair_msgs, None, None, GenParamsCls(max_tokens=default_max_tok, temperature=0.0))
                step_cost_usd += float(getattr(res2, "cost_usd", 0.0))
                code2, ast_issues2 = extract_and_sanitize_code(res2.text, target_fn, prev_func_names)
                if not ast_issues2 and code2.strip():
                    ok2, _ = run_nobody_sandbox_check(
                        prob["required_dependencies"],
                        "\n\n".join(c for c in prev_codes[:idx] if c),
                        code2,
                        target_fn,
                        exp_arity,
                        smoke_snip,
                        ref_impl,
                    )
                    if ok2 or ast_issues1:
                        final_code = code2
                        repaired = True

            (dir_clover / f"{pid}.{step_num}.py").write_text(f"{prev_full}\n{final_code}\n", encoding="utf-8")
            prev_codes[idx] = final_code
            if target_fn:
                prev_func_names.add(target_fn)

            dt_s = time.perf_counter() - t_s0
            step_logs.append({
                "step_id": step_id,
                "target_func": target_fn,
                "prompt_tokens": res1.prompt_tokens,
                "cached_tokens": res1.cached_tokens,
                "completion_tokens": res1.completion_tokens,
                "reasoning_tokens": getattr(res1, "reasoning_tokens", 0),
                "cost_usd": round(step_cost_usd, 6),
                "prefill_ms": round(res1.prefill_ms, 1),
                "decode_ms": round(res1.decode_ms, 1),
                "wall_s": round(dt_s, 2),
                "sandbox_ok_turn1": ok1,
                "repaired": repaired,
                "repair_reason": repair_reason if not ok1 else "",
                "xcruncher_used": bool(xc_meta.get("pre_injected")),
            })
            print(
                f"[W{worker_id}] {step_id} `{target_fn}`: "
                f"{res1.completion_tokens}t (${step_cost_usd:.4f}) in {dt_s:.1f}s "
                f"(lcp={res1.cached_tokens}/{res1.prompt_tokens}, ok1={ok1}, repaired={repaired})",
                flush=True,
            )

        done_marker.write_text(
            json.dumps(
                {
                    "problem_id": pid,
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "wall_s": round(time.perf_counter() - t_prob0, 2),
                    "steps": step_logs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--scicode-dir", type=Path, default=Path("/opt/scicode_bench/SciCode"))
    ap.add_argument("--sealed-h5", type=Path, default=Path("/root/sealed_scicode_oracle/test_data.h5"))
    ap.add_argument("--out-root", type=Path, default=Path("/opt/scicode_bench/run_out"))
    ap.add_argument("--fusion-hints", type=Path, default=None)
    ap.add_argument("--backend", choices=["native", "openrouter", "http"], default="native")
    ap.add_argument("--workers", type=int, default=0, help="Override worker count (used by openrouter/http backends)")
    ap.add_argument("--api-url", type=str, default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-file", type=str, default="/dev/shm/.or_key")
    ap.add_argument("--api-model", type=str, default="kimi-k3")
    ap.add_argument("--reasoning-effort", type=str, default="medium")
    ap.add_argument("--clover-stack-dir", type=str, default="/opt/clover-k3")
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--sharded", action="store_true", help="Single unified engine spanning all --gpus")
    ap.add_argument("--think-cap", type=int, default=96)
    ap.add_argument("--kimi-self-smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    verify_zero_leakage_sandbox(args.sealed_h5)
    data_dir = args.scicode_dir / "eval" / "data"
    jsonl = data_dir / ("problems_test.jsonl" if args.split == "test" else "problems_dev.jsonl")
    problems = load_sanitized_problems(jsonl, args.split)
    if args.limit > 0:
        problems = problems[: args.limit]

    if args.backend != "native":
        n_workers = max(1, args.workers or 12)
        gpu_ids: list[int | str] = list(range(n_workers))
    elif args.sharded:
        gpu_ids = [args.gpus.strip()]
        n_workers = 1
    else:
        gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
        n_workers = len(gpu_ids)

    total_phys_cores = min(112, os.cpu_count() or 32)
    cores_per_w = max(1, total_phys_cores // n_workers)

    # Sort longest multi-step problems first when running parallel workers so tail stragglers don't bottleneck!
    problems_sorted = sorted(problems, key=lambda p: len(p["sub_steps"]), reverse=(n_workers > 1))

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    for p in problems_sorted:
        q.put(p)
    for _ in range(n_workers):
        q.put(None)

    t0 = time.perf_counter()
    procs = []
    for w, gid in enumerate(gpu_ids):
        c_lo = (w * cores_per_w) % total_phys_cores
        c_hi = min(total_phys_cores - 1, c_lo + max(1, cores_per_w) - 1)
        pr = ctx.Process(
            target=worker_loop,
            args=(
                w, gid, c_lo, c_hi, q,
                str(data_dir), str(args.out_root),
                str(args.fusion_hints) if args.fusion_hints else None,
                args.think_cap,
                args.kimi_self_smoke,
                args.backend,
                args.api_url,
                args.api_key_file,
                args.api_model,
                args.reasoning_effort,
                args.clover_stack_dir,
            ),
        )
        pr.start()
        procs.append(pr)
        time.sleep(0.15 if args.backend != "native" else 2.0)

    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"Worker exited with code {pr.exitcode}")

    print(f"ALL_PROBLEMS_GENERATED in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
