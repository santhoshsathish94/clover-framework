#!/usr/bin/env python3
"""
Toggleable Clover MCP Server for `OpenCode` & `Copilot Fusion` Phase-5 Ablations
(`benchmarks/riemann-lean4/mcp_clover_tools.py`).

Implements a clean JSON-RPC 2.0 `stdio` Model Context Protocol (MCP) server exposing ONLY
those Clover capabilities with genuine mathematical upside over a standalone `C1` LLM:
  1. `s2_lean_premise_search`: Searches Mathlib + RH declaration *type signatures* with
     strict Trap-C anti-leakage exclusion (excludes the current module, sibling files in
     the same theorem directory, and all proof bodies — returning ONLY theorem names and
     type signatures!).
  2. `xcruncher_exact_math`: Evaluates exact arbitrary-precision integer, prime, GCD,
     totient, and modular arithmetic expressions (`pow(a, b, m)`, `math.gcd`, `is_prime`).
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


def _is_prime_exact(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Deterministic bases up to 2^64, high-confidence Miller-Rabin above
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _safe_pow(a: int, b: int, m: int | None = None) -> int:
    if m is not None:
        return pow(a, b, m)
    if b < 0 or b > 1_000_000:
        raise ValueError("2-arg pow exponent must be in [0, 1_000_000]; use 3-arg pow(a, b, mod) for larger exponents")
    if a not in (-1, 0, 1) and b * max(1, abs(a).bit_length()) > 4_000_000:
        raise ValueError("Result exceeds 4,000,000-bit safety limit")
    return pow(a, b)


def _safe_factorial(n: int) -> int:
    if n < 0 or n > 20_000:
        raise ValueError("factorial(n) requires 0 <= n <= 20,000")
    return math.factorial(n)


def _safe_comb(n: int, k: int) -> int:
    if n < 0 or k < 0 or n > 100_000:
        raise ValueError("comb(n, k) requires 0 <= n, k <= 100,000")
    return math.comb(n, k)


_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "pow": _safe_pow,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "isqrt": math.isqrt,
    "factorial": _safe_factorial,
    "comb": _safe_comb,
    "is_prime": _is_prime_exact,
}


def eval_exact_math(expr: str) -> str:
    tree = ast.parse(expr.strip(), mode="eval")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
            return node.value if isinstance(node.value, bool) else int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = int(_eval(node.operand))
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitXor, ast.LShift, ast.RShift)
        ):
            l, r = int(_eval(node.left)), int(_eval(node.right))
            if isinstance(node.op, ast.Pow):
                return _safe_pow(l, r)
            if isinstance(node.op, ast.LShift):
                if r < 0 or r > 1_000_000 or (l != 0 and abs(l).bit_length() + r > 4_000_000):
                    raise ValueError("Left shift exceeds 4,000,000-bit safety limit")
                return l << r
            if isinstance(node.op, ast.RShift):
                if r < 0:
                    raise ValueError("Negative shift count")
                return l >> min(r, abs(l).bit_length() + 1)
            if isinstance(node.op, ast.Mult) and l != 0 and r != 0 and abs(l).bit_length() + abs(r).bit_length() > 4_000_000:
                raise ValueError("Multiplication exceeds 4,000,000-bit safety limit")
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.FloorDiv):
                return l // r
            if isinstance(node.op, ast.Mod):
                return l % r
            if isinstance(node.op, ast.BitXor):
                return l ^ r
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
            args = [_eval(a) for a in node.args]
            return _SAFE_FUNCS[node.func.id](*args)
        raise ValueError(f"Unsupported syntax node: {type(node).__name__}")

    val = _eval(tree)
    if isinstance(val, bool):
        return str(val)
    iv = int(val)
    if abs(iv).bit_length() > 8192:
        # Summarize without hitting Python 3.11+ int_max_str_digits (4300 digits)
        sign = "-" if iv < 0 else ""
        av = abs(iv)
        approx_digits = int(av.bit_length() * 0.3010299956639812) + 1
        tail60 = str(av % (10**60)).zfill(60)
        return f"{sign}<int bit_length={av.bit_length()} ~digits={approx_digits} mod_10^60=...{tail60}>"
    s = str(iv)
    if len(s) > 512:
        return f"digits={len(s)} head={s[:120]}... tail=...{s[-120:]}"
    return s


class AntiLeakagePremiseIndex:
    """Indexes ONLY declaration signatures (`never proof bodies`) and enforces Trap-C exclusions."""

    def __init__(self, lean_project_dir: Path):
        self.proj = lean_project_dir
        self.entries: list[tuple[str, str, set[str], str]] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        decl_head_re = re.compile(r"^(?:private\s+|protected\s+)?(?:theorem|lemma)\s+([A-Za-z0-9_'.]+)", re.MULTILINE)
        stop_line_re = re.compile(
            r"^\s*(?:\||where\b|(?:private\s+|protected\s+|noncomputable\s+)*(?:theorem|lemma|def|abbrev|instance|axiom|opaque|inductive|structure|class|end|namespace|section)\b)"
        )
        # Index Mathlib `.lean` sources if present in `.lake/packages/mathlib/Mathlib`
        roots = [
            self.proj / ".lake" / "packages" / "mathlib" / "Mathlib",
            self.proj / "src" / "Formalism",
        ]
        for root in roots:
            if not root.is_dir():
                continue
            for fp in root.rglob("*.lean"):
                try:
                    txt = fp.read_text(encoding="utf-8")
                except OSError:
                    continue
                rel = fp.relative_to(self.proj).as_posix()
                for m in decl_head_re.finditer(txt):
                    nm = m.group(1)
                    tail = txt[m.end() : m.end() + 1200]
                    sig_parts: list[str] = []
                    for li, ln in enumerate(tail.splitlines()):
                        if li > 0 and stop_line_re.match(ln):
                            break
                        if ":=" in ln:
                            sig_parts.append(ln.split(":=", 1)[0])
                            break
                        if re.search(r"\bwhere\b", ln):
                            sig_parts.append(re.split(r"\bwhere\b", ln, maxsplit=1)[0])
                            break
                        sig_parts.append(ln)
                    sig = " ".join(" ".join(sig_parts).split())[:260]
                    toks = set(re.findall(r"[A-Za-z0-9_']{3,}", (nm + " " + sig).lower()))
                    self.entries.append((rel, nm, toks, f"{nm} {sig}"))

    def search(self, query: str, excluded_module_substr: str = "", excluded_lemma: str = "", top_k: int = 8) -> list[dict[str, str]]:
        self._ensure_loaded()
        q_toks = set(re.findall(r"[A-Za-z0-9_']{2,}", query.lower()))
        if not q_toks:
            return []
        ex_mod = excluded_module_substr.strip().lower()
        ex_lem = excluded_lemma.strip().lower()
        scored: list[tuple[float, str, str]] = []
        for rel, nm, toks, full_sig in self.entries:
            rel_low = rel.lower()
            if ex_mod and ex_mod in rel_low:
                continue
            if ex_lem and nm.lower() == ex_lem:
                continue
            overlap = len(q_toks & toks)
            if overlap > 0:
                score = overlap / math.sqrt(len(toks) + 4.0)
                scored.append((score, rel, full_sig))
        scored.sort(key=lambda x: -x[0])
        return [{"module": r, "signature": s} for _, r, s in scored[: max(1, min(top_k, 15))]]


TOOLS_SPEC = [
    {
        "name": "s2_lean_premise_search",
        "description": "Search Mathlib4 & Formalism theorem signatures (anti-leakage filtered: never returns proof bodies or the target module).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords or Mathlib identifiers, e.g. 'Submodule.span_induction IsLattice'"},
                "excluded_module": {"type": "string", "description": "Current module path/directory to exclude for anti-leakage"},
                "excluded_lemma": {"type": "string", "description": "Current theorem/lemma name to exclude"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "xcruncher_exact_math",
        "description": "Evaluate exact arbitrary-precision integer arithmetic, pow(a,b,m), gcd, lcm, comb, factorial, or is_prime(n).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Exact integer expression, e.g. 'pow(2, 127) - 1' or 'is_prime(7919)'"},
            },
            "required": ["expression"],
        },
    },
]


def serve_mcp_stdio() -> None:
    lean_proj = Path(os.environ.get("CLOVER_LEAN_PROJECT_DIR", "/mnt/nvme1/clover-rh-bench/rh-src/lean"))
    idx = AntiLeakagePremiseIndex(lean_proj)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "clover-lean-xcruncher-mcp", "version": "1.0.0"},
                },
            }
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS_SPEC}}
        elif method == "tools/call":
            tname = params.get("name")
            targs = params.get("arguments") or {}
            try:
                if tname == "xcruncher_exact_math":
                    txt = eval_exact_math(str(targs.get("expression", "0")))
                elif tname == "s2_lean_premise_search":
                    hits = idx.search(
                        str(targs.get("query", "")),
                        excluded_module_substr=str(targs.get("excluded_module", "")),
                        excluded_lemma=str(targs.get("excluded_lemma", "")),
                    )
                    txt = json.dumps(hits, indent=2)
                else:
                    raise ValueError(f"Unknown tool: {tname}")
                resp = {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": txt}]}}
            except Exception as exc:
                resp = {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {"content": [{"type": "text", "text": f"ERROR: {exc}"}], "isError": True},
                }
        elif rid is not None:
            resp = {"jsonrpc": "2.0", "id": rid, "result": {}}
        else:
            continue

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    serve_mcp_stdio()
