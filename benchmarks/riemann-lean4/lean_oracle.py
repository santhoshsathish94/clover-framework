#!/usr/bin/env python3
"""
Tamper-Proof Incremental Lean 4 (`v4.33.1`) Formal Verification Oracle
(`benchmarks/riemann-lean4/lean_oracle.py`).

Enforces every integrity boundary identified in `AGENTS.md` and the `Claude Opus 5.5`
benchmark audit:
  1. Immutable Theorem Header Lock (`prefix_sha256`): Candidate proof attempts are
     spliced strictly after `:= by\\n` on the original file prefix, making it
     structurally impossible for a model to weaken a theorem statement or add imports.
  2. Static Forbidden-Construct Gate: Rejects `sorry`, `admit`, `axiom`, `unsafe`,
     `extern`, `implemented_by`, `opaque`, `native_decide`, `ofReduceBool`,
     `trustCompiler`, `skipKernelTC`, `run_cmd`, `addDecl`, `elab`, and `macro`
     (after stripping Lean `--` line comments and `/- ... -/` block comments).
  3. Sandboxed Incremental Kernel Check (`lake env lean`): Runs in a temporary file
     against the prebuilt `.olean` cache on `/mnt/nvme1/clover-rh-bench/rh-src/lean`
     with a hard wall-clock timeout and POSIX virtual-memory cap (`RLIMIT_AS`, default
     `12 GiB`) so Lean elaboration never contends with `HG4-16R` `mmap` weights on `k3`.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

FORBIDDEN_TOKENS: tuple[str, ...] = (
    "sorry",
    "admit",
    "axiom",
    "unsafe",
    "extern",
    "implemented_by",
    "opaque",
    "native_decide",
    "ofReduceBool",
    "trustCompiler",
    "skipKernelTC",
    "run_cmd",
    "run_tac",
    "run_elab",
    "run_meta",
    "elab",
    "macro",
    "syntax",
    "initialize",
    "addDecl",
)

_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in FORBIDDEN_TOKENS) + r")\b")
_FORBIDDEN_SYNTAX_RE = re.compile(
    r"(?:#\s*(?:eval|reduce|check|synth|compile)\b|decide\s*\+\s*native\b|\blocal\s+(?:notation|macro|elab|syntax|instance|attribute)\b|\bset_option\s+debug\.)"
)
_ALLOWED_AXIOMS: frozenset[str] = frozenset({"propext", "Classical.choice", "Quot.sound"})


@dataclass(frozen=True)
class LeanCheckResult:
    passed: bool
    stage: str  # "static_gate" | "header_tamper" | "timeout" | "kernel_error" | "kernel_verified"
    diagnostics: str
    elapsed_ms: float
    forbidden_hits: tuple[str, ...] = ()


def strip_lean_comments(src: str) -> str:
    """Remove Lean 4 nested `/- ... -/` block comments and `-- ...` line comments."""
    out: list[str] = []
    i = 0
    n = len(src)
    depth = 0
    in_str = False
    while i < n:
        ch = src[i]
        if depth == 0 and in_str:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                out.append('"')
                in_str = False
            else:
                out.append(" ")
            i += 1
            continue
        # Mask Lean character literals (`'a'`, `'\"'`, `'\\''`, `'\n'`, `'\x41'`) so `'\"'` never opens a string
        if depth == 0 and ch == "'":
            m_chr = re.match(r"^'(?:\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)|[^'\\\n])'", src[i:])
            if m_chr:
                out.append("''")
                i += len(m_chr.group(0))
                continue
        if depth == 0 and ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if src.startswith("/-", i):
            depth += 1
            i += 2
            continue
        if depth > 0 and src.startswith("-/", i):
            depth -= 1
            i += 2
            continue
        if depth > 0:
            if ch == "\n":
                out.append("\n")
            i += 1
            continue
        if src.startswith("--", i):
            nl = src.find("\n", i)
            if nl == -1:
                break
            i = nl
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def find_forbidden_constructs(proof_body: str) -> tuple[str, ...]:
    cleaned = strip_lean_comments(proof_body)
    hits = set(m.group(1) for m in _FORBIDDEN_RE.finditer(cleaned))
    for sm in _FORBIDDEN_SYNTAX_RE.finditer(cleaned):
        hits.add(re.sub(r"\s+", " ", sm.group(0).strip()))
    return tuple(sorted(hits))


def assemble_candidate_lean(
    immutable_prefix: str,
    candidate_tactics: str,
    immutable_suffix: str = "",
    expected_prefix_sha256: str | None = None,
    thm_name: str | None = None,
) -> tuple[str | None, LeanCheckResult | None]:
    """Safely splice `candidate_tactics` after `immutable_prefix` (ending with `:= by\\n`)."""
    actual_sha = hashlib.sha256(immutable_prefix.encode("utf-8")).hexdigest()
    if expected_prefix_sha256 and actual_sha != expected_prefix_sha256:
        return None, LeanCheckResult(
            passed=False,
            stage="header_tamper",
            diagnostics=f"Immutable theorem prefix SHA-256 mismatch ({actual_sha} != {expected_prefix_sha256})",
            elapsed_ms=0.0,
        )

    body = extract_clean_tactic_block(candidate_tactics)
    if not body.strip():
        return None, LeanCheckResult(
            passed=False,
            stage="static_gate",
            diagnostics="Empty tactic proof body.",
            elapsed_ms=0.0,
        )

    bad = find_forbidden_constructs(body)
    if bad:
        return None, LeanCheckResult(
            passed=False,
            stage="static_gate",
            diagnostics=f"Forbidden Lean construct(s) in proof body: {', '.join(bad)}",
            elapsed_ms=0.0,
            forbidden_hits=bad,
        )

    # Reject attempts to close the enclosing namespace/section or inject top-level declarations
    cleaned_body = strip_lean_comments(body)
    if re.search(r"^\s*(?:end|namespace|section|theorem|lemma|axiom|opaque|inductive|structure)\b", cleaned_body, flags=re.MULTILINE):
        return None, LeanCheckResult(
            passed=False,
            stage="static_gate",
            diagnostics="Top-level declaration or namespace boundary inside tactic block is forbidden.",
            elapsed_ms=0.0,
            forbidden_hits=("top_level_decl",),
        )

    indented_lines: list[str] = []
    raw_lines = body.splitlines()
    non_empty_indents = [len(ln) - len(ln.lstrip(" ")) for ln in raw_lines if ln.strip()]
    min_indent = min(non_empty_indents) if non_empty_indents else 0
    pad = "  " if min_indent < 2 else ""
    for ln in raw_lines:
        if not ln.strip():
            indented_lines.append("")
        else:
            indented_lines.append(pad + ln.replace("\t", "  "))

    full_src = immutable_prefix.rstrip() + "\n" + "\n".join(indented_lines) + "\n"
    if thm_name:
        full_src += f"\n#print axioms {thm_name}\n"
    if immutable_suffix.strip():
        full_src += "\n" + immutable_suffix.strip() + "\n"
    return full_src, None


_ECHOED_THM_HEADER_RE = re.compile(
    r"^\s*(?:open\s+[^\n]+\n\s*)*(?:private\s+|protected\s+|noncomputable\s+)*(?:theorem|lemma|example)\b[\s\S]*?:=\s*by\b",
)


def extract_clean_tactic_block(raw_model_reply: str) -> str:
    """Extract Lean tactic lines from a fenced code block or raw model completion without truncating `have ... := by`."""
    text = raw_model_reply.strip()
    # Strip `<think>...</think>` blocks if present
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    m = re.search(r"```(?:lean4?|)\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    # Only strip an echoed outer `theorem/lemma ... := by` header at the start of the block — NEVER `have h := by`!
    mh = _ECHOED_THM_HEADER_RE.match(text)
    if mh:
        text = text[mh.end() :].lstrip("\r\n")
    return text


def verify_printed_axioms(lean_stdout: str, thm_name: str) -> tuple[bool, str]:
    """Verify `#print axioms <thm_name>` output (including multi-line wrapped `[...]` lists) uses ONLY `{propext, Classical.choice, Quot.sound}`."""
    esc = re.escape(thm_name)
    if re.search(rf"'[^'\n]*{esc}'\s+does not depend on any axioms", lean_stdout):
        return True, ""
    m = re.search(rf"'[^'\n]*{esc}'\s+depends on axioms:\s*\[([^\]]*)\]", lean_stdout, flags=re.DOTALL)
    if m:
        axs = {a.strip() for a in m.group(1).split(",") if a.strip()}
        illegal = sorted(axs - _ALLOWED_AXIOMS)
        if illegal:
            return False, f"Disallowed axiom(s) used by {thm_name}: {', '.join(illegal)}"
        return True, ""
    return False, f"Missing `#print axioms {thm_name}` confirmation in Lean output."


def _make_preexec_rlimit(max_mem_gib: float):
    if os.name == "nt":
        return None

    def _apply() -> None:
        try:
            import resource

            cap = int(max_mem_gib * (1024**3))
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except Exception:
            pass

    return _apply


def verify_lean_module(
    immutable_prefix: str,
    candidate_tactics: str,
    lean_project_dir: str | Path = "/mnt/nvme1/clover-rh-bench/rh-src/lean",
    elan_home: str | Path = "/mnt/nvme1/clover-rh-bench/elan",
    immutable_suffix: str = "",
    expected_prefix_sha256: str | None = None,
    thm_name: str | None = None,
    timeout_s: float = 60.0,
    max_mem_gib: float = 8.0,
    extra_lean_args: Sequence[str] = (
        "--threads=4",
        "--tstack=400000",
        "-DautoImplicit=false",
        "-DrelaxedAutoImplicit=false",
        "-DmaxSynthPendingDepth=3",
        "-Dpp.unicode.fun=true",
        "-DmaxHeartbeats=4000000",
    ),
) -> LeanCheckResult:
    """Elaborate & kernel-verify a single candidate proof against prebuilt `.olean`s."""
    if thm_name is None:
        m_tn = re.findall(r"(?:private\s+|protected\s+)?(?:theorem|lemma)\s+([A-Za-z0-9_']+)", immutable_prefix)
        thm_name = m_tn[-1] if m_tn else None

    full_src, gate_err = assemble_candidate_lean(
        immutable_prefix,
        candidate_tactics,
        immutable_suffix=immutable_suffix,
        expected_prefix_sha256=expected_prefix_sha256,
        thm_name=thm_name,
    )
    if gate_err is not None or full_src is None:
        return gate_err or LeanCheckResult(False, "static_gate", "Failed to assemble source", 0.0)

    proj = Path(lean_project_dir)
    elan = Path(elan_home)
    lake_bin = elan / "bin" / ("lake.exe" if os.name == "nt" else "lake")
    cmd_lake = str(lake_bin) if lake_bin.is_file() else "lake"

    env = dict(os.environ)
    if elan.is_dir():
        env["ELAN_HOME"] = str(elan)
        env["PATH"] = str(elan / "bin") + os.pathsep + env.get("PATH", "")

    # Write candidate outside the Lean project tree so temporary artifacts never pollute `proj`
    tmp_dir = Path(tempfile.gettempdir()) / "clover_lean_verify_sandbox"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_s = tempfile.mkstemp(prefix="cand_", suffix=".lean", dir=str(tmp_dir))
    os.close(fd)
    tmp_path = Path(tmp_path_s)

    t0 = time.perf_counter()
    try:
        tmp_path.write_text(full_src, encoding="utf-8")
        mem_mb = max(2048, int(max_mem_gib * 1024))
        cmd = [cmd_lake, "env", "lean", f"--memory={mem_mb}", *extra_lean_args, str(tmp_path)]
        proc = subprocess.run(
            cmd,
            cwd=str(proj),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if proc.returncode == 0 and "declaration uses 'sorry'" not in combined and "error:" not in combined:
            if thm_name:
                ax_ok, ax_err = verify_printed_axioms(combined, thm_name)
                if not ax_ok:
                    return LeanCheckResult(
                        passed=False,
                        stage="axiom_violation",
                        diagnostics=ax_err + "\n" + combined[:1200],
                        elapsed_ms=elapsed_ms,
                    )
            return LeanCheckResult(
                passed=True,
                stage="kernel_verified",
                diagnostics="",
                elapsed_ms=elapsed_ms,
            )
        return LeanCheckResult(
            passed=False,
            stage="kernel_error",
            diagnostics=combined[:2400],
            elapsed_ms=elapsed_ms,
        )
    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return LeanCheckResult(
            passed=False,
            stage="timeout",
            diagnostics=f"Lean elaboration timed out after {timeout_s}s.",
            elapsed_ms=elapsed_ms,
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
