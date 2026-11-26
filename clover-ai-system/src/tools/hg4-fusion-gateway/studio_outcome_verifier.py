"""Clover Studio Capability S2: Outcome Self-Verifier (`clover_cycle.py` Distillation).

Implements the three strictly separated variants mandated by the Claude Opus 5.5 audit:
  - `S2-ctrl` (`blind`): Compute-matched blind 1-turn re-check control on code/Lean drafts without objective findings.
  - `S2a` (`fast`): Zero-external-toolchain (<50ms typical) suite-blind verifier:
      * Python `ast.parse` syntax check + isolated `python3 -I` execution of ONLY `>>>` doctests
        explicitly written in the user's own prompt (never external test files).
      * Lean 4 surface invariant check (no `sorry`/`admit`/`axiom`, balanced `/- ... -/` block comments,
        preserved theorem signature on full-file roundtrip prompts).
  - `S2b` (`lean_elaborate`): Opt-in `verifier-in-the-loop` Lean 4 `lake env lean` elaboration when
    `x_clover_studio_s2b_lean_project` or `CLOVER_STUDIO_S2B_LEAN_PROJECT` is explicitly configured.
"""

from __future__ import annotations

import ast
import doctest
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_PY_PROMPT_RE = re.compile(
    r"(?:\bpython\b|```python|\bdef\s+[A-Za-z_][A-Za-z0-9_]*\s*\(|\bclass\s+[A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_LEAN_PROMPT_RE = re.compile(
    r"(?:\blean\s*4\b|```lean|\btheorem\s+[A-Za-z0-9_']+|:= by\b|\bMathlib\b)",
    re.IGNORECASE,
)
_FORBIDDEN_LEAN_PROOF_RE = re.compile(r"\b(?:sorry|admit|axiom)\b")
_FORBIDDEN_LEAN_ALWAYS_META_RE = re.compile(
    r"(?:#(?:eval|reduce|check|synth)\b"
    r"|\b(?:run_cmd|run_tac|run_elab|run_meta|elab_rules|macro_rules|declare_syntax_cat|builtin_initialize)\b"
    r"|\b(?:native_decide|ofReduceBool|ofReduceNat)\b"
    r"|decide\s*\+\s*native"
    r"|«|\bLean\.(?:Elab|Meta|Parser|ofReduceBool|ofReduceNat)\b"
    r"|@\s*\[(?!\s*simp\s*\])"
    r"|(?:^|\s)(?:local\s+|scoped\s+)?attribute\s*\[(?!\s*simp\s*\]))",
    re.MULTILINE,
)
_FORBIDDEN_LEAN_CODE_META_RE = re.compile(
    r"\b(?:elab|macro|syntax|initialize|IO|System)\b"
)


def _continues_lean_ident(src: str, pos: int) -> bool:
    """Return True iff `src[pos]` continues an existing Lean 4 identifier token (rather than starting a new token)."""
    k = pos - 1
    while k >= 0 and src[k] in ("'", ".", "!", "?"):
        k -= 1
    return k >= 0 and (src[k].isalnum() or src[k] == "_")


def _strip_lean_comments_and_strings(src: str) -> str:
    """Remove Lean 4 nested `/- ... -/` block comments, `--` line comments, char literals `'...'`, and string literals."""
    out: list[str] = []
    i, n, depth = 0, len(src), 0
    in_str = False
    while i < n:
        if in_str:
            if src[i] == "\\" and i + 1 < n:
                i += 2
                continue
            if src[i] == '"':
                in_str = False
            i += 1
            continue
        if depth > 0:
            if src[i:i + 2] == "/-":
                depth += 1
                i += 2
            elif src[i:i + 2] == "-/":
                depth -= 1
                i += 2
            else:
                if src[i] == "\n":
                    out.append("\n")
                i += 1
            continue
        if src[i:i + 2] == "/-":
            depth = 1
            i += 2
        elif src[i:i + 2] == "--":
            while i < n and src[i] != "\n":
                i += 1
        elif src[i] == "'":
            # Lean character literal `'c'` or `'\x..'` / `'\"'` (distinct from identifier prime `x'`)
            prev_ch = src[i - 1] if i > 0 else " "
            if not (prev_ch.isalnum() or prev_ch == "_"):
                j = i + 1
                if j < n and src[j] == "\\":
                    j += 2
                    while j < n and src[j] != "'" and (j - i) <= 10:
                        j += 1
                elif j < n:
                    j += 1
                if j < n and src[j] == "'":
                    out.append("'c'")
                    i = j + 1
                    continue
            out.append("'")
            i += 1
        elif src[i] == "r" and not _continues_lean_ident(src, i):
            # Lean 4 raw string literal `r"..."`, `r#"..."#`, `r##"..."##` (lowercase `r` starting a new token)
            j = i + 1
            while j < n and src[j] == "#":
                j += 1
            hashes = j - (i + 1)
            if j < n and src[j] == '"':
                closing = '"' + ("#" * hashes)
                end_pos = src.find(closing, j + 1)
                out.append('""')
                i = (end_pos + len(closing)) if end_pos >= 0 else n
                continue
            out.append(src[i])
            i += 1
        elif src[i] == '"':
            in_str = True
            out.append('""')
            i += 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)

_FORBIDDEN_CANDIDATE_ATTRS = frozenset({
    "sys", "os", "inspect", "types", "builtins", "modules", "compile", "exec", "eval",
    "getattr", "setattr", "delattr", "attrgetter", "itemgetter", "methodcaller",
    "loader", "spec", "subclasses", "globals", "locals", "frame", "f_globals",
    "f_locals", "f_builtins", "gi_frame", "cr_frame", "ag_frame", "co_consts", "co_code", "replace",
})


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else str(c or "")
    return ""


def _extract_python_block(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    fences = re.findall(r"```(?:python|py)\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fences:
        return fences[-1].strip()
    generic = re.findall(r"```\s*\n(.*?)```", cleaned, flags=re.DOTALL)
    if generic:
        return generic[-1].strip()
    if re.search(r"^\s*(?:def|class|import|from)\s+", cleaned, flags=re.MULTILINE):
        return cleaned
    return ""


def _extract_lean_block(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    fences = re.findall(r"```(?:lean4?|)\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fences:
        return fences[-1].strip()
    return cleaned


_UNSAFE_CALL_NAMES = frozenset({
    "open", "eval", "exec", "compile", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "breakpoint", "input", "exit", "quit",
    "type", "object", "super", "classmethod", "staticmethod", "property", "memoryview",
})

_ALLOWED_DOCTEST_AST_TYPES = (
    ast.Module, ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Assert,
    ast.For, ast.While, ast.If, ast.Pass, ast.Break, ast.Continue,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.IfExp,
    ast.Constant, ast.Name, ast.Attribute, ast.Subscript, ast.Slice,
    ast.List, ast.Tuple, ast.Set, ast.Dict,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.JoinedStr, ast.FormattedValue, ast.Starred, ast.keyword,
    ast.expr_context, ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
)


_SAFE_STDLIB_IMPORTS = frozenset({
    "math", "heapq", "bisect", "collections", "itertools", "functools", "decimal", "fractions",
})


def _is_safe_candidate_ast(code: str) -> bool:
    """Allow pure-Python algorithmic candidate code + safe stdlib math/collection imports; skip OS/FFI/dunder drafts."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True  # SyntaxError is handled by caller before doctest execution
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _SAFE_STDLIB_IMPORTS or (alias.asname and alias.asname.startswith("_")):
                    return False
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in _SAFE_STDLIB_IMPORTS or node.level != 0:
                return False
            for alias in node.names:
                if alias.name.startswith("_") or (alias.asname and alias.asname.startswith("_")):
                    return False
        elif isinstance(node, ast.Match):
            return False
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in _FORBIDDEN_CANDIDATE_ATTRS):
            return False
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("__"):
            return False
        elif isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in _UNSAFE_CALL_NAMES - {"type", "object", "super", "classmethod", "staticmethod", "property"}):
            return False
    return True


def _is_safe_doctest_ast(src: str) -> bool:
    """Strict AST node-class allowlist (rejects `match`/`case`, `import`, `with`, `try`, `def`, `lambda`, underscore names, and unsafe builtins)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_DOCTEST_AST_TYPES):
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in _UNSAFE_CALL_NAMES):
            return False
    return True


def _extract_needed_safe_imports(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in _SAFE_STDLIB_IMPORTS:
                    mods.add(a.name)
        elif isinstance(node, ast.ImportFrom) and (node.module or "") in _SAFE_STDLIB_IMPORTS:
            mods.add(str(node.module))
    return sorted(mods)


def _extract_prompt_doctest_runner(prompt_text: str, preload_modules: list[str] | None = None) -> str | None:
    """Build an isolated, zero-compile/zero-open runtime-hooked `doctest` snippet strictly from safe `>>>` in `prompt_text`."""
    if ">>>" not in prompt_text:
        return None
    unfenced = re.sub(r"^\s*```[^\n]*$", "", prompt_text, flags=re.MULTILINE)
    parser = doctest.DocTestParser()
    try:
        examples = [e for e in parser.get_examples(unfenced) if e.source.strip()]
    except Exception:
        return None
    if not examples or not all(_is_safe_doctest_ast(e.source) for e in examples):
        return None

    mods_tuple = tuple(preload_modules or ())
    return (
        "def __clover_run_isolated_doctests(_cand_path: str, _prompt_src: str, _preload: tuple) -> None:\n"
        "    import doctest as _dt, sys as _sys\n"
        "    for _m in _preload:\n"
        "        __import__(_m)\n"
        "    with open(_cand_path, 'r', encoding='utf-8') as _f:\n"
        "        _cand_code = compile(_f.read(), _cand_path, 'exec')\n"
        "    _examples = [e for e in _dt.DocTestParser().get_examples(_prompt_src) if e.source.strip()]\n"
        "    _compiled_ex = []\n"
        "    for _e in _examples:\n"
        "        _s = _e.source.strip()\n"
        "        _w = _e.want.strip()\n"
        "        try:\n"
        "            _c = compile(_s, '<doctest>', 'eval')\n"
        "            _is_eval = True\n"
        "        except SyntaxError:\n"
        "            _c = compile(_e.source, '<doctest>', 'exec')\n"
        "            _is_eval = False\n"
        "        _compiled_ex.append((_s, _w, _c, _is_eval))\n"
        "    _ALLOWED_EVENTS = {'exec', 'import', 'function.__new__', 'object.__getattr__', 'object.__setattr__', 'object.__delattr__', 'sys._getframe', 'builtins.id'}\n"
        "    def _audit_guard(event: str, args: tuple) -> None:\n"
        "        if event in _ALLOWED_EVENTS:\n"
        "            return\n"
        "        raise PermissionError(f'blocked audit event: {event}')\n"
        "    _sys.addaudithook(_audit_guard)\n"
        "    _ns: dict = {}\n"
        "    try:\n"
        "        exec(_cand_code, _ns, _ns)\n"
        "        for _s, _w, _c, _is_eval in _compiled_ex:\n"
        "            if _is_eval:\n"
        "                _got = eval(_c, _ns, _ns)\n"
        "                if _w and repr(_got) != _w and str(_got) != _w:\n"
        "                    _preview = repr(_got) if isinstance(_got, (int, float, bool, type(None), tuple, list, dict, set)) else '<mismatch>'\n"
        "                    _sys.stderr.write(f'Failed example: {_s}\\nExpected: {_w}\\nGot: {_preview[:120]}\\n')\n"
        "                    _sys.exit(1)\n"
        "            else:\n"
        "                exec(_c, _ns, _ns)\n"
        "    except PermissionError:\n"
        "        _sys.exit(77)\n"
        f"__clover_run_isolated_doctests(_CAND_FILE, {unfenced!r}, {mods_tuple!r})\n"
    )


def _minimal_subprocess_env() -> dict[str, str]:
    """Strip all gateway secrets/tokens from child environment; retain only minimal OS loader keys."""
    clean: dict[str, str] = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
    for k in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "LANG", "LC_ALL"):
        if k in os.environ:
            clean[k] = os.environ[k]
    return clean


def verify_python_draft_fast(prompt_text: str, draft_text: str, timeout_s: float = 2.5) -> str | None:
    """Return an objective failure string if Python AST or prompt-supplied `>>>` doctests fail, else `None`."""
    code = _extract_python_block(draft_text)
    if not code:
        return "No Python code block found in draft; provide the complete ```python ... ``` implementation."
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"Python SyntaxError at line {exc.lineno}: {exc.msg}. Return the corrected complete ```python ... ``` code."

    # Dynamic host execution requires explicit operator opt-in (`CLOVER_STUDIO_S2_ALLOW_HOST_EXEC=1`) or an
    # external OS sandbox prefix (`CLOVER_STUDIO_SANDBOX_CMD`, e.g. `bwrap ...` / `docker run --network=none ...`).
    sandbox_prefix = [p for p in os.environ.get("CLOVER_STUDIO_SANDBOX_CMD", "").split() if p]
    allow_host_exec = bool(sandbox_prefix) or os.environ.get("CLOVER_STUDIO_S2_ALLOW_HOST_EXEC", "0") == "1"
    if not allow_host_exec:
        return None

    if not _is_safe_candidate_ast(code):
        return None
    dt_runner = _extract_prompt_doctest_runner(prompt_text, preload_modules=_extract_needed_safe_imports(code))
    if not dt_runner:
        return None

    with tempfile.TemporaryDirectory() as td:
        cand_path = Path(td) / "candidate.py"
        cand_path.write_text(code + "\n", encoding="utf-8")
        script_path = Path(td) / "doctest_check.py"
        script_path.write_text(f"_CAND_FILE = {str(cand_path)!r}\n{dt_runner}\n", encoding="utf-8")
        cmd = [*sandbox_prefix, sys.executable, "-I", str(script_path)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=td,
                env=_minimal_subprocess_env(),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return "Public doctest self-check timed out (>2.5s; likely infinite loop). Fix termination and output the complete ```python ... ``` block."
        if proc.returncode == 77:
            return None
        if proc.returncode != 0:
            err_lines = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = "\n".join(err_lines[-10:]) if err_lines else f"exit code {proc.returncode}"
            return (
                f"Your code failed the prompt's own doctest:\n{tail}\n"
                "Fix the bug and output ONLY the corrected complete ```python ... ``` block."
            )
    return None


def verify_lean_surface_fast(prompt_text: str, draft_text: str, *, allow_verdict_meta: bool = False) -> str | None:
    """Fast zero-toolchain Lean 4 surface check (strips comments/strings and skips plain English prose)."""
    cleaned_raw = re.sub(r"<think>.*?</think>", "", draft_text or "", flags=re.DOTALL).strip()
    if not cleaned_raw:
        return "Empty Lean 4 output; provide the requested Lean 4 tactic or file block."
    has_fence = bool(re.search(r"```(?:lean4?|)\s*\n", cleaned_raw, flags=re.IGNORECASE))
    block = _extract_lean_block(cleaned_raw)
    if block.count("/-") != block.count("-/"):
        return "Unclosed `/- ... -/` block comment in Lean 4 output; close all comments cleanly."
    stripped = _strip_lean_comments_and_strings(block)
    is_verdict_prompt = allow_verdict_meta and bool(re.search(r"(?i)^\s*verdict\s*:", prompt_text, flags=re.MULTILINE))
    if not is_verdict_prompt and _FORBIDDEN_LEAN_ALWAYS_META_RE.search(stripped):
        return "Forbidden Lean 4 metaprogramming or `native_decide` directive in proof; use pure tactics only."
    looks_like_lean = has_fence or bool(
        re.search(
            r"^\s*(?:import|open|namespace|section|theorem|lemma|def|abbrev|example|attribute|set_option|#|@\[|"
            r"by|exact|apply|intro|intros|rw|simp|ring|linarith|omega|norm_num|decide|calc|refine|show|have|rcases|cases|induction|constructor|trivial|rfl|sorry|admit|axiom|elab|macro|syntax|initialize|run_)\b",
            stripped,
            flags=re.MULTILINE,
        )
    )
    if not looks_like_lean:
        return None
    if not is_verdict_prompt and _FORBIDDEN_LEAN_PROOF_RE.search(stripped):
        return "Forbidden placeholder (`sorry` / `admit` / `axiom`) left in Lean 4 proof; replace with a real constructive tactic proof."
    if not is_verdict_prompt and _FORBIDDEN_LEAN_CODE_META_RE.search(stripped):
        return "Forbidden Lean 4 metaprogramming or IO directive (`elab` / `macro` / `syntax` / `IO`) in proof; use pure tactics only."
    return None


def verify_lean_elaboration_optin(
    prompt_text: str,
    draft_text: str,
    lean_project_dir: str,
    timeout_s: float = 25.0,
) -> str | None:
    """Opt-in S2b verifier-in-the-loop: elaborate Lean 4 blocks outside project tree with metaprogramming/IO banned."""
    sandbox_prefix = [p for p in os.environ.get("CLOVER_STUDIO_SANDBOX_CMD", "").split() if p]
    allow_host_exec = bool(sandbox_prefix) or os.environ.get("CLOVER_STUDIO_S2_ALLOW_HOST_EXEC", "0") == "1"
    if not allow_host_exec:
        return None
    proj = Path(lean_project_dir)
    if not proj.is_dir():
        return None
    block = _extract_lean_block(draft_text)
    stripped = _strip_lean_comments_and_strings(block)
    if not re.search(r"^\s*(?:import|theorem|lemma|def)\s+", stripped, flags=re.MULTILINE):
        return None
    if (
        _FORBIDDEN_LEAN_PROOF_RE.search(stripped)
        or re.search(r"\baxiom\b", block)
        or _FORBIDDEN_LEAN_ALWAYS_META_RE.search(stripped)
        or _FORBIDDEN_LEAN_CODE_META_RE.search(stripped)
        or _FORBIDDEN_LEAN_ALWAYS_META_RE.search(block)
        or _FORBIDDEN_LEAN_CODE_META_RE.search(block)
    ):
        return "Forbidden Lean 4 placeholder, metaprogramming, or `native_decide`/`IO` directive in proof; use pure tactics only."
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / "CandidateCheck.lean"
        tmp_path.write_text(block + "\n", encoding="utf-8")
        cmd = [*sandbox_prefix, "lake", "env", "lean", "--memory=4096", "--threads=2", str(tmp_path)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(proj),
                env=_minimal_subprocess_env(),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            combined_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode != 0 or "declaration uses 'sorry'" in combined_out or "uses 'sorry'" in combined_out:
                diag = "\n".join(combined_out.strip().splitlines()[:8])
                return (
                    f"Lean 4 compiler elaboration reported an error or sorry warning:\n{diag}\n"
                    "Fix the Lean 4 proof/types and output ONLY the corrected ```lean ... ``` block."
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
    return None


def check_outcome_cycle(
    mode: str,
    messages: list[dict[str, Any]],
    draft_content: str,
    tool_calls: list[dict[str, Any]],
    lean_project_dir: str = "",
) -> tuple[str | None, str]:
    """Run S2 outcome check according to `mode` in `{'off', 'blind', 'fast', 'lean_elaborate'}`."""
    if mode == "off" or tool_calls:
        return None, "skipped"
    prompt_text = _last_user_text(messages)
    if not prompt_text.strip():
        return None, "empty_prompt"

    is_py = bool(_PY_PROMPT_RE.search(prompt_text)) and not bool(_LEAN_PROMPT_RE.search(prompt_text))
    is_lean = bool(_LEAN_PROMPT_RE.search(prompt_text))

    if mode == "blind":
        if is_py or is_lean:
            return (
                "Carefully review your previous draft for any syntax errors, off-by-one edge cases, "
                "or invalid steps, and output ONLY the final corrected code/proof block.",
                "blind_recheck_triggered",
            )
        return None, "blind_not_applicable"

    if is_py:
        finding = verify_python_draft_fast(prompt_text, draft_content)
        if finding:
            return finding, "python_fast_finding"
        return None, "python_fast_clean"

    if is_lean:
        finding = verify_lean_surface_fast(prompt_text, draft_content)
        if finding:
            return finding, "lean_surface_finding"
        if mode == "lean_elaborate" and lean_project_dir:
            elab_finding = verify_lean_elaboration_optin(prompt_text, draft_content, lean_project_dir)
            if elab_finding:
                return elab_finding, "lean_elaborate_finding"
            return None, "lean_elaborate_clean"
        return None, "lean_surface_clean"

    return None, "not_code_or_lean"
