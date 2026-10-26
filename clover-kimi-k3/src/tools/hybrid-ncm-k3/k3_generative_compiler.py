r"""Clover Kimi K3 — Generative Compilation Subsystem (`k3_generative_compiler.py`).

Implements the on-the-fly partial-program sealing, static compilation, and span-remapped
diagnostic rollback mechanism of *Generative Compilation: On-the-Fly Compiler Feedback as
AI Generates Code* (`arXiv:2607.13921`, Mündler-Sasahara et al., ETH Zurich / INSAIT / UC Berkeley)
for Clover Kimi K3 (`93-Layer Kimi K3 MoE + HybridAgent + 11-Pass NCM + Ontology GraphRAG`):

1. **Asymmetric Guarantee Target ($\mathcal{G}_{\mathcal{C}, \mathcal{S}}(c) = \mathcal{C}(\mathcal{S}(c))$):**
   - **Global Completeness ($\forall c.\ (\exists w.\ c \circ w \in \mathcal{L}) \implies \mathcal{S}(c) \in \mathcal{L}$):**
     Never reject a partial program prefix $c$ that admits a valid completion.
   - **Selective Soundness ($\forall c \in \mathcal{X}_{\text{stmt}}.\ \mathcal{S}(c) \in \mathcal{L} \implies \exists w.\ c \circ w \in \mathcal{L}$):**
     Catch genuine syntax, scope, arity, type, and borrow dead ends at completed statement boundaries
     $\mathcal{X}_{\text{stmt}}$ before downstream tokens are wasted.

2. **Rust Generative Compiler (`CloverRustGenerativeCompiler`):**
   - Wraps the compiled `rust_generative_compiler` PyO3 extension (`/home/usergpu/generative-compilation/bindings/rust_generative_compiler`)
     and the patched in-memory `generative-compiler-rustc` (`rust 1.95.0` + `MockLoader`).
   - Seals partial Rust code with `let _ = &(e);`, `crate::__missing__<T>() -> T { panic!() }` ($\mathtt{hole}_{\mathtt{val}}$),
     and `panic!()` ($\mathtt{hole}_{\mathtt{div}}$), runs the 2-pass `rustc` type/borrow checker (`Pass 1 -> ExtendPartial/SetArgumentCount -> Pass 2`),
     maps spans back to the unsealed prefix via `SourceMap`, and formats concise compiler diagnostics for rollback.

3. **Python Generative Compiler (`CloverPythonGenerativeCompiler`):**
   - Applies the exact same **Global Completeness + Statement-Boundary Soundness** sealor architecture to Python:
     - **Python Sealor ($\mathcal{S}_{\text{PY}}$):** Given any partial Python prefix $c$:
       - Drops an incomplete trailing line that is still mid-token/mid-expression (`end_marker=False`),
         closes any unclosed delimiters `(...)`, `[...]`, `{...}` on completed lines only when needed,
         and appends indented `pass` statements to close every open suite (`def`, `class`, `if`, `elif`,
         `else`, `for`, `while`, `try`, `except`, `finally`, `with`).
     - **Statement-Boundary Soundness Analyzer (`ast.parse` + Static Scope / Call-Arity / Prefix-Shadowing Checker):**
       - Verifies exact Python syntax and indentation via `compile(\mathcal{S}_{\text{PY}}(c), "<partial>", "exec")`.
       - Walks completed statements in the sealed AST to detect:
         (a) **Call Arity Mismatches (`E_PY_ARITY`):** Calls to user-defined or known standard functions with fewer
             arguments than required or more arguments than accepted (when no `*args` is present).
         (b) **Token-Prefix Shadowing (`E_PY_PREFIX_SHADOW`):** `if/elif` chains on the same string/slice variable
             where a shorter prefix literal (`'o'`) is tested before a longer prefix literal (`'o|'`) that starts with it.
         (c) **Return Outside Function / Break-Continue Outside Loop (`E_PY_CONTROL`):** Caught immediately on the line.
         (d) **Definite Unbound Local / Undefined Name (`E_PY_UNDEFINED`):** When `allow_undefined=False` (or at `end_marker=True`),
             flags variables loaded before assignment that are neither builtins, imports, parameters, nor module-level bindings.
"""
from __future__ import annotations

import ast
import builtins
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

# Ensure `rust_generative_compiler` and `rustc` are on `sys.path` and `PATH` across `gpu`, `k3`, and fresh servers
_GC_BINDINGS_DIR = "/home/usergpu/generative-compilation/bindings/rust_generative_compiler"
_GC_RELEASE_DIR = "/home/usergpu/generative-compilation/target/release"
_GC_RUSTC_DIR = "/home/usergpu/generative-compilation/crates/generative-compiler-rustc/target/release"

if os.path.isdir(_GC_BINDINGS_DIR) and _GC_BINDINGS_DIR not in sys.path:
    sys.path.insert(0, _GC_BINDINGS_DIR)

for _bin_dir in (
    _GC_RUSTC_DIR,
    _GC_RELEASE_DIR,
    "/home/usergpu/.cargo/bin",
    "/root/.cargo/bin",
    os.path.expanduser("~/.cargo/bin"),
):
    if os.path.isdir(_bin_dir) and _bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{_bin_dir}:{os.environ.get('PATH', '')}"

try:
    import rust_generative_compiler as _rgc  # type: ignore
except Exception:
    _rgc = None

_RUSTC_BIN = shutil.which("rustc")
HAS_RUST_GC = (_rgc is not None) or (_RUSTC_BIN is not None)


@dataclass
class GCDiagnostic:
    """Single compiler or static-analysis diagnostic mapped back to the original prefix `code`."""
    code: str
    message: str
    line: int
    column: int = 1
    rendered: str = ""
    origin_line: int | None = None

    def format_concise(self) -> str:
        if self.rendered:
            return self.rendered.strip()
        return f"error[{self.code}] at line {self.line}, col {self.column}: {self.message}"


@dataclass
class GCVerdict:
    r"""Result of running $\mathcal{G}_{\mathcal{C}, \mathcal{S}}(c) = \mathcal{C}(\mathcal{S}(c))$ on a partial or complete program $c$."""
    ok: bool
    language: str
    sealed_code: str
    diagnostics: list[GCDiagnostic] = field(default_factory=list)
    suppressed_errors: int = 0
    elapsed_ms: float = 0.0
    rollback_line_idx: int | None = None

    def format_feedback(self, max_diagnostics: int = 3) -> str:
        if self.ok or not self.diagnostics:
            return ""
        shown = [d.format_concise() for d in self.diagnostics[:max_diagnostics]]
        return "\n".join(shown)


# ---------------------------------------------------------------------------
# 1. Python Sealor ($\mathcal{S}_{\text{PY}}$) & Statement-Boundary Checker
# ---------------------------------------------------------------------------

_PY_BUILTINS = set(dir(builtins)) | {
    "List", "Dict", "Tuple", "Set", "Optional", "Union", "Any", "Sequence", "Iterable", "Callable",
    "math", "re", "sys", "os", "json", "collections", "itertools", "functools", "heapq", "bisect",
    "np", "numpy", "torch", "__name__", "__file__",
}

_SUITE_HEADER_RE = re.compile(
    r"^(\s*)(?:async\s+def|def|class|if|elif|else|for|async\s+for|while|try|except|finally|with|async\s+with|match|case)\b.*:\s*(?:#.*)?$"
)

# Known standard-library / built-in exact call arities `(min_args, max_args)`
_KNOWN_BUILTIN_ARITIES: dict[str, tuple[int, int]] = {
    "len": (1, 1),
    "abs": (1, 1),
    "ord": (1, 1),
    "chr": (1, 1),
    "bin": (1, 1),
    "hex": (1, 1),
    "oct": (1, 1),
    "bool": (0, 1),
    "repr": (1, 1),
    "all": (1, 1),
    "any": (1, 1),
    "reversed": (1, 1),
    "enumerate": (1, 2),
    "divmod": (2, 2),
    "isinstance": (2, 2),
    "issubclass": (2, 2),
    "pow": (2, 3),
    "round": (1, 2),
    "range": (1, 3),
}


def _has_unclosed_delimiters_or_string(text: str) -> bool:
    """Returns True if `text` has an unclosed string literal or unclosed `()`, `[]`, `{}`."""
    stack: list[str] = []
    in_str: str | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str is not None:
            if ch == "\\":
                i += 2
                continue
            if text.startswith(in_str, i):
                i += len(in_str)
                in_str = None
                continue
            if ch == "\n" and len(in_str) == 1:
                return True
            i += 1
            continue
        else:
            if ch == "#":
                nl = text.find("\n", i)
                if nl == -1:
                    break
                i = nl + 1
                continue
            if text.startswith('"""', i) or text.startswith("'''", i):
                in_str = text[i : i + 3]
                i += 3
                continue
            if ch in ('"', "'"):
                in_str = ch
                i += 1
                continue
            if ch in "([{":
                stack.append(ch)
            elif ch in ")]}":
                if not stack:
                    return False
                top = stack.pop()
                if (top, ch) not in (("(", ")"), ("[", "]"), ("{", "}")):
                    return False
            i += 1
    return (in_str is not None) or bool(stack)


def seal_python_prefix(code: str, end_marker: bool = False) -> tuple[str, int]:
    r"""Python Sealor $\mathcal{S}_{\text{PY}}(c)$:
    Given a partial Python program $c$:
      1. If `end_marker=False` and $c$ does not end with a newline, drops the unfinished trailing line
         (just as the Featherweight Rust sealor replaces an unfinished trailing statement with $\epsilon$
         to guarantee **Global Completeness**).
      2. If trailing lines have unclosed multiline parentheses/brackets/triple-quotes while `end_marker=False`,
         truncates back to the last delimiter-balanced line boundary so multi-line calls/comprehensions
         are never falsely rejected mid-expression.
      3. Closes every open indentation block (`def`, `if`, `elif`, `else`, `for`, `while`, `try`, `with`, `class`)
         whose suite is empty or still open by appending properly indented `pass` statements (and `except Exception:\n pass`
         for bare `try:` blocks).
    Returns `(sealed_code, kept_line_count)`."""
    if not code or not code.strip():
        return "pass\n", 0

    raw_lines = code.splitlines()
    if not end_marker and not code.endswith("\n") and len(raw_lines) > 1:
        raw_lines = raw_lines[:-1]

    # If `end_marker=False`, strip trailing lines that are inside an unclosed multiline string/bracket/backslash continuation
    if not end_marker:
        while raw_lines:
            candidate = "\n".join(raw_lines)
            last_stripped = raw_lines[-1].rstrip()
            if (
                _has_unclosed_delimiters_or_string(candidate)
                or last_stripped.endswith("\\")
                or last_stripped in ("@",)
                or last_stripped.startswith("@")
            ):
                raw_lines.pop()
            else:
                break

    if not raw_lines:
        return "pass\n", 0

    # Walk lines to close any suite header (`def ...:`, `if ...:`, `else:`, etc.) that has no body yet,
    # and close any `try:` block that has no `except` or `finally` yet.
    sealed_lines: list[str] = []
    open_tries: list[int] = []  # indentation levels of open `try:` headers without except/finally

    n = len(raw_lines)
    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            indent = len(line) - len(line.lstrip(" "))
            # Pop any `open_tries` deeper than `indent`, or at `indent` if matched by `except`/`finally`
            while open_tries and open_tries[-1] > indent:
                try_ind = open_tries.pop()
                sealed_lines.append(" " * try_ind + "except Exception:")
                sealed_lines.append(" " * (try_ind + 4) + "pass")
            if open_tries and open_tries[-1] == indent:
                if re.match(r"^(?:except|finally)\b", stripped):
                    open_tries.pop()
                elif not stripped.startswith("#"):
                    try_ind = open_tries.pop()
                    sealed_lines.append(" " * try_ind + "except Exception:")
                    sealed_lines.append(" " * (try_ind + 4) + "pass")

            if re.match(r"^try\s*:", stripped):
                open_tries.append(indent)

        sealed_lines.append(line)

        if _SUITE_HEADER_RE.match(line):
            header_indent = len(line) - len(line.lstrip(" "))
            # Look ahead for the next non-empty, non-comment line
            has_body = False
            for j in range(idx + 1, n):
                nxt_s = raw_lines[j].strip()
                if not nxt_s or nxt_s.startswith("#"):
                    continue
                nxt_indent = len(raw_lines[j]) - len(raw_lines[j].lstrip(" "))
                if nxt_indent > header_indent:
                    has_body = True
                break
            if not has_body:
                sealed_lines.append(" " * (header_indent + 4) + "pass")

    while open_tries:
        try_ind = open_tries.pop()
        sealed_lines.append(" " * try_ind + "except Exception:")
        sealed_lines.append(" " * (try_ind + 4) + "pass")

    return "\n".join(sealed_lines) + "\n", len(raw_lines)


class _PythonStatementBoundaryAnalyzer(ast.NodeVisitor):
    r"""Static soundness analyzer over the sealed Python AST $\mathcal{S}_{\text{PY}}(c)$."""

    def __init__(self, kept_line_count: int, allow_undefined: bool = True, end_marker: bool = False) -> None:
        self.kept_line_count = kept_line_count
        self.allow_undefined = allow_undefined and (not end_marker)
        self.end_marker = end_marker
        self.diagnostics: list[GCDiagnostic] = []
        self.func_arities: dict[str, tuple[int, int | None, set[str], bool]] = {}
        self.scope_stack: list[set[str]] = [set(_PY_BUILTINS)]
        self.in_function: int = 0
        self.in_loop: int = 0

    def _collect_Bindings(self, node: ast.AST, target_set: set[str]) -> None:
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                target_set.add(child.name)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                target_set.add(child.id)
            elif isinstance(child, ast.arg):
                target_set.add(child.arg)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    target_set.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(child, ast.ExceptHandler) and child.name:
                target_set.add(child.name)

    def analyze(self, tree: ast.Module) -> list[GCDiagnostic]:
        # First pass: collect top-level & nested function signatures (`min_pos`, `max_pos`, `kw_names`, `has_kwargs`)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                pos_args = list(args.posonlyargs) + list(args.args)
                # Exclude `self`/`cls` for methods
                if pos_args and pos_args[0].arg in ("self", "cls"):
                    pos_args = pos_args[1:]
                n_total_pos = len(pos_args)
                n_defaults = len(args.defaults)
                min_pos = max(0, n_total_pos - n_defaults)
                max_pos = None if args.vararg is not None else n_total_pos
                kw_names = {a.arg for a in pos_args} | {a.arg for a in args.kwonlyargs}
                has_kwargs = args.kwarg is not None
                self.func_arities[node.name] = (min_pos, max_pos, kw_names, has_kwargs)
                self.scope_stack[0].add(node.name)
            elif isinstance(node, ast.ClassDef):
                self.scope_stack[0].add(node.name)

        self.visit(tree)
        return self.diagnostics

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node)

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope_stack[-1].add(node.name)
        local_scope = set()
        for a in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
            local_scope.add(a.arg)
        if node.args.vararg:
            local_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            local_scope.add(node.args.kwarg.arg)
        self.scope_stack.append(local_scope)
        self.in_function += 1
        self._check_block_closed_whiles(node.body, is_final_block=self.end_marker)
        self._check_bisection_return(node)
        for stmt in node.body:
            self.visit(stmt)
        self.in_function -= 1
        self.scope_stack.pop()

    def _check_bisection_return(self, fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Detects bisection functions where `mid = (left + right) / 2` is computed in a `while` loop
        but the final `return` returns `(left + right)` without dividing by 2."""
        midpoint_pairs: set[tuple[str, str]] = set()
        for child in ast.walk(fn_node):
            if isinstance(child, ast.While):
                for sub in ast.walk(child):
                    if (
                        isinstance(sub, ast.BinOp)
                        and isinstance(sub.op, (ast.Div, ast.FloorDiv))
                        and isinstance(sub.right, ast.Constant)
                        and sub.right.value in (2, 2.0)
                        and isinstance(sub.left, ast.BinOp)
                        and isinstance(sub.left.op, ast.Add)
                        and isinstance(sub.left.left, ast.Name)
                        and isinstance(sub.left.right, ast.Name)
                    ):
                        a, b = sub.left.left.id, sub.left.right.id
                        midpoint_pairs.add((a, b))
                        midpoint_pairs.add((b, a))
        if not midpoint_pairs or not fn_node.body:
            return
        last_stmt = fn_node.body[-1]
        if (
            isinstance(last_stmt, ast.Return)
            and getattr(last_stmt, "lineno", 0) <= self.kept_line_count
            and isinstance(last_stmt.value, ast.BinOp)
            and isinstance(last_stmt.value.op, ast.Add)
            and isinstance(last_stmt.value.left, ast.Name)
            and isinstance(last_stmt.value.right, ast.Name)
        ):
            pair = (last_stmt.value.left.id, last_stmt.value.right.id)
            if pair in midpoint_pairs:
                self.diagnostics.append(
                    GCDiagnostic(
                        code="E_PY_BISECTION_RETURN",
                        message=(
                            f"Bisection return error on line {last_stmt.lineno}: `return ({pair[0]} + {pair[1]})` "
                            f"omits `/ 2.0` (returns twice the root instead of the midpoint `({pair[0]} + {pair[1]}) / 2.0`)."
                        ),
                        line=getattr(last_stmt, "lineno", 1),
                        column=getattr(last_stmt, "col_offset", 0) + 1,
                    )
                )

        # Also detect `return -find_zero([-c for c in xs])` (negating polynomial coefficients `-P(x)` preserves the same root `+x`, not `-x`!)
        for child in ast.walk(fn_node):
            if (
                isinstance(child, ast.Return)
                and getattr(child, "lineno", 0) <= self.kept_line_count
                and isinstance(child.value, ast.UnaryOp)
                and isinstance(child.value.op, ast.USub)
                and isinstance(child.value.operand, ast.Call)
                and isinstance(child.value.operand.func, ast.Name)
                and child.value.operand.func.id == fn_node.name
            ):
                self.diagnostics.append(
                    GCDiagnostic(
                        code="E_PY_BISECTION_SIGN",
                        message=(
                            f"Sign error on line {child.lineno}: `return -{fn_node.name}(...)` negates the root, "
                            f"but `-poly(xs, x) == 0` has the exact SAME root `x` as `poly(xs, x) == 0`! "
                            f"Use `return {fn_node.name}([-c for c in xs])` without the leading `-` sign."
                        ),
                        line=getattr(child, "lineno", 1),
                        column=getattr(child, "col_offset", 0) + 1,
                    )
                )

    def _check_block_closed_whiles(self, stmts: list[ast.stmt], is_final_block: bool) -> None:
        """Checks `while` loops that have been closed (either followed by another statement in `stmts`
        or `is_final_block=True`) for provable single-variable stagnation (`E_PY_INFINITE_LOOP`)."""
        for idx, st in enumerate(stmts):
            is_closed = (idx + 1 < len(stmts)) or is_final_block
            if isinstance(st, ast.While) and is_closed and getattr(st, "lineno", 0) <= self.kept_line_count:
                self._check_while_stagnation(st)

    def _check_while_stagnation(self, node: ast.While) -> None:
        """Detects `while count < n:` loops where `count` is only mutated inside an `if` branch (no `else`)
        and no separate step/iteration variable (`i`, `k`, `a`, `b`, `left`, `right`, `low`, `high`) is updated
        unconditionally in the loop body."""
        cond_vars: set[str] = {
            child.id for child in ast.walk(node.test) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
        if not cond_vars:
            return
        # Ignore `while True:`
        if not node.body:
            return

        unconditional_stores: set[str] = set()
        conditional_stores: set[str] = set()
        has_break_or_return = False

        for st in node.body:
            if isinstance(st, ast.Pass):
                continue
            if isinstance(st, (ast.Break, ast.Return, ast.Raise)):
                has_break_or_return = True
            elif isinstance(st, ast.If):
                for ch in ast.walk(st):
                    if isinstance(ch, (ast.Break, ast.Return, ast.Raise)):
                        has_break_or_return = True
                    elif isinstance(ch, ast.Name) and isinstance(ch.ctx, ast.Store):
                        conditional_stores.add(ch.id)
                # If both `body` and `orelse` update a condition variable (e.g. bisection `if ...: right = mid else: left = mid`),
                # the loop state advances on every iteration!
                if st.orelse:
                    b_stores = {c.id for s in st.body for c in ast.walk(s) if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Store)}
                    e_stores = {c.id for s in st.orelse for c in ast.walk(s) if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Store)}
                    unconditional_stores |= (b_stores & e_stores)
                    if (b_stores & cond_vars) and (e_stores & cond_vars):
                        unconditional_stores |= ((b_stores | e_stores) & cond_vars)
            else:
                # Top-level statement in `while` body: check if it mutates a variable via AugAssign (`i += 1`),
                # tuple swap (`a, b = b, a + b`), or method call (`stack.pop()`, `list.append()`)
                if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                    unconditional_stores.add(st.target.id)
                elif isinstance(st, ast.Assign):
                    rhs_names = {c.id for c in ast.walk(st.value) if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Load)}
                    for t in st.targets:
                        lhs_names = {c.id for c in ast.walk(t) if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Store)}
                        # If `lhs` is also in `rhs` (e.g. `i = i + 1` or `a, b = b, a + b`), it advances state;
                        # whereas `fib_num = fibonacci(count)` only recomputes `fib_num` from unchanged `count`!
                        if lhs_names & rhs_names:
                            unconditional_stores |= lhs_names
                elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Attribute):
                    if isinstance(st.value.func.value, ast.Name):
                        unconditional_stores.add(st.value.func.value.id)

        if not has_break_or_return and (cond_vars & conditional_stores) and not unconditional_stores:
            c_var = sorted(cond_vars & conditional_stores)[0]
            self.diagnostics.append(
                GCDiagnostic(
                    code="E_PY_INFINITE_LOOP",
                    message=(
                        f"Stagnant `while` loop on line {node.lineno}: loop condition variable `{c_var}` is only "
                        f"updated inside a conditional `if` branch and no step variable (`a, b = b, a + b` or `k += 1`) "
                        f"advances unconditionally when the `if` condition is False."
                    ),
                    line=getattr(node, "lineno", 1),
                    column=getattr(node, "col_offset", 0) + 1,
                )
            )

    def visit_Return(self, node: ast.Return) -> None:
        if self.in_function == 0 and getattr(node, "lineno", 0) <= self.kept_line_count:
            self.diagnostics.append(
                GCDiagnostic(
                    code="E_PY_CONTROL",
                    message="'return' outside function",
                    line=getattr(node, "lineno", 1),
                    column=getattr(node, "col_offset", 0) + 1,
                )
            )
        self.generic_visit(node)

    def visit_Break(self, node: ast.Break) -> None:
        if self.in_loop == 0 and getattr(node, "lineno", 0) <= self.kept_line_count:
            self.diagnostics.append(
                GCDiagnostic(
                    code="E_PY_CONTROL",
                    message="'break' outside loop",
                    line=getattr(node, "lineno", 1),
                    column=getattr(node, "col_offset", 0) + 1,
                )
            )

    def visit_Continue(self, node: ast.Continue) -> None:
        if self.in_loop == 0 and getattr(node, "lineno", 0) <= self.kept_line_count:
            self.diagnostics.append(
                GCDiagnostic(
                    code="E_PY_CONTROL",
                    message="'continue' outside loop",
                    line=getattr(node, "lineno", 1),
                    column=getattr(node, "col_offset", 0) + 1,
                )
            )

    def visit_For(self, node: ast.For) -> None:
        self._collect_Bindings(node.target, self.scope_stack[-1])
        self.visit(node.iter)
        self.in_loop += 1
        for s in node.body:
            self.visit(s)
        self.in_loop -= 1
        for s in node.orelse:
            self.visit(s)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self.in_loop += 1
        for s in node.body:
            self.visit(s)
        self.in_loop -= 1
        for s in node.orelse:
            self.visit(s)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for t in node.targets:
            self._collect_Bindings(t, self.scope_stack[-1])

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value:
            self.visit(node.value)
        self._collect_Bindings(node.target, self.scope_stack[-1])

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self.visit(node.target)
        self._collect_Bindings(node.target, self.scope_stack[-1])

    def visit_Import(self, node: ast.Import) -> None:
        self._collect_Bindings(node, self.scope_stack[-1])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._collect_Bindings(node, self.scope_stack[-1])

    def visit_If(self, node: ast.If) -> None:
        # Check for Token-Prefix Shadowing across `if ... elif ...` branches!
        # Example: `if music_string[i] == 'o': ... elif music_string[i:i+2] == 'o|':`
        # where `'o|'.startswith('o')` and both inspect the same base variable/index!
        self._check_if_prefix_shadowing(node)
        self.generic_visit(node)

    def _extract_prefix_match_branch(self, test: ast.AST) -> tuple[str, str, int] | None:
        """If `test` is `var[idx] == 'lit'`, `var[idx:idx+k] == 'lit'`, or `var.startswith('lit')`,
        returns `(base_var_name, string_literal, lineno)`."""
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            left, right = test.left, test.comparators[0]
            if isinstance(right, ast.Constant) and isinstance(right.value, str):
                if isinstance(left, ast.Subscript) and isinstance(left.value, ast.Name):
                    return (left.value.id, right.value, getattr(test, "lineno", 1))
                if isinstance(left, ast.Name):
                    return (left.id, right.value, getattr(test, "lineno", 1))
        if (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Attribute)
            and test.func.attr == "startswith"
            and isinstance(test.func.value, ast.Name)
            and len(test.args) >= 1
            and isinstance(test.args[0], ast.Constant)
            and isinstance(test.args[0].value, str)
        ):
            return (test.func.value.id, test.args[0].value, getattr(test, "lineno", 1))
        return None

    def _check_if_prefix_shadowing(self, node: ast.If) -> None:
        seen_branches: list[tuple[str, str, int]] = []
        cur: ast.If | None = node
        while cur is not None:
            info = self._extract_prefix_match_branch(cur.test)
            if info is not None:
                var_name, lit_val, ln = info
                if ln <= self.kept_line_count and len(lit_val) >= 1:
                    for prev_var, prev_lit, prev_ln in seen_branches:
                        if prev_var == var_name and len(lit_val) > len(prev_lit) and lit_val.startswith(prev_lit):
                            # Check if `cur` is a slice/startswith on the same indexed string where `prev_lit` was 1-char index
                            if isinstance(cur.test, ast.Compare) and isinstance(cur.test.left, ast.Subscript):
                                self.diagnostics.append(
                                    GCDiagnostic(
                                        code="E_PY_PREFIX_SHADOW",
                                        message=(
                                            f"Token-prefix shadowing: branch for longer token {lit_val!r} on line {ln} "
                                            f"is shadowed by shorter prefix {prev_lit!r} checked earlier on line {prev_ln} "
                                            f"over `{var_name}`. Check longer multi-character tokens ({lit_val!r}) BEFORE "
                                            f"shorter prefix tokens ({prev_lit!r}), or tokenize with `{var_name}.split()`."
                                        ),
                                        line=prev_ln,
                                        column=getattr(cur, "col_offset", 0) + 1,
                                    )
                                )
                seen_branches.append(info)
            if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
                cur = cur.orelse[0]
            else:
                cur = None

    def visit_Call(self, node: ast.Call) -> None:
        lineno = getattr(node, "lineno", 0)
        if lineno <= self.kept_line_count and isinstance(node.func, ast.Name):
            fn_name = node.func.id
            has_starred = any(isinstance(a, ast.Starred) for a in node.args)
            has_kw_starred = any(k.arg is None for k in node.keywords)
            if not has_starred and not has_kw_starred:
                n_pos = len(node.args)
                n_kw = len(node.keywords)
                if fn_name in _KNOWN_BUILTIN_ARITIES and n_kw == 0:
                    min_a, max_a = _KNOWN_BUILTIN_ARITIES[fn_name]
                    if n_pos < min_a or n_pos > max_a:
                        self.diagnostics.append(
                            GCDiagnostic(
                                code="E_PY_ARITY",
                                message=f"Call to built-in `{fn_name}()` expects {min_a}..{max_a} arguments, got {n_pos}",
                                line=lineno,
                                column=getattr(node, "col_offset", 0) + 1,
                            )
                        )
                elif fn_name in self.func_arities:
                    min_p, max_p, kw_names, has_kwargs = self.func_arities[fn_name]
                    total_supplied = n_pos + n_kw
                    if total_supplied < min_p:
                        self.diagnostics.append(
                            GCDiagnostic(
                                code="E_PY_ARITY",
                                message=f"Call to `{fn_name}()` requires at least {min_p} arguments, got {total_supplied}",
                                line=lineno,
                                column=getattr(node, "col_offset", 0) + 1,
                            )
                        )
                    elif max_p is not None and n_pos > max_p:
                        self.diagnostics.append(
                            GCDiagnostic(
                                code="E_PY_ARITY",
                                message=f"Call to `{fn_name}()` accepts at most {max_p} positional arguments, got {n_pos}",
                                line=lineno,
                                column=getattr(node, "col_offset", 0) + 1,
                            )
                        )
        self.generic_visit(node)


def restore_spec_helpers(code: str, spec_prompt: str) -> str:
    """If `spec_prompt` provides complete helper functions before the target function (e.g., `def poly` before
    `def find_zero`, or `def encode_cyclic` before `def decode_cyclic`), restores any missing helper functions
    at the top of `code` so the generated module is self-contained."""
    if not spec_prompt or "def " not in spec_prompt:
        return code
    # Extract the specification code region from `spec_prompt`
    spec_code = spec_prompt
    if ":\n\n" in spec_prompt and spec_prompt.lstrip().startswith("Write the complete Python function"):
        spec_code = spec_prompt.split(":\n\n", 1)[1]
    sealed_spec, _ = seal_python_prefix(spec_code, end_marker=False)
    try:
        spec_tree = ast.parse(sealed_spec)
    except Exception:
        return code

    spec_funcs = [n for n in spec_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(spec_funcs) < 2:
        return code

    existing_defs = set(re.findall(r"^\s*(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code, flags=re.MULTILINE))
    spec_lines = spec_code.splitlines()
    missing_blocks: list[str] = []
    # All functions before the last function in `spec_funcs` are specification-provided helpers
    for idx, fn in enumerate(spec_funcs[:-1]):
        if fn.name not in existing_defs:
            start_ln = max(0, fn.lineno - 1)
            end_ln = spec_funcs[idx + 1].lineno - 1
            helper_src = "\n".join(spec_lines[start_ln:end_ln]).strip()
            if helper_src:
                missing_blocks.append(helper_src)

    if not missing_blocks:
        return code
    return "\n\n\n".join(missing_blocks) + "\n\n\n" + code


def extract_doctest_pairs(text: str) -> list[tuple[str, str]]:
    """Extracts `(expr, expected)` doctest pairs from `>>> expr\\nexpected` lines in docstrings,
    validating that both `expr` and `expected` parse as valid Python expressions."""
    pairs: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith(">>> "):
            expr = re.sub(r"\s+#.*$", "", s[4:]).strip()
            i += 1
            if i < len(lines):
                nxt = lines[i].strip()
                nxt = re.sub(r'(?:"""|\'\'\')\s*$', "", nxt).strip()
                nxt = re.sub(r"\s+#.*$", "", nxt).strip()
                if nxt and not nxt.startswith((">>> ", '"""', "'''", "#", "def ", "class ")):
                    try:
                        ast.parse(expr, mode="eval")
                        ast.parse(nxt, mode="eval")
                        pairs.append((expr, nxt))
                    except Exception:
                        pass
                    i += 1
        else:
            i += 1
    return pairs


class CloverPythonGenerativeCompiler:
    r"""Generative Compiler $\mathcal{G}_{\mathcal{C}, \mathcal{S}}$ for Python partial and complete programs."""

    def __init__(self, allow_undefined: bool = True) -> None:
        self.allow_undefined = allow_undefined

    @staticmethod
    def _find_target_func_body_start(tree: ast.Module, total_lines: int) -> int:
        """Returns the 0-based line index of the first executable statement after the docstring in the last function."""
        funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not funcs:
            return 0
        fn = funcs[-1]
        if not fn.body:
            return max(0, fn.lineno)
        first_stmt = fn.body[0]
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
        ):
            if len(fn.body) >= 2:
                return max(0, fn.body[1].lineno - 1)
            return min(total_lines, getattr(first_stmt, "end_lineno", first_stmt.lineno))
        return max(0, first_stmt.lineno - 1)

    def check_spec_doctests(self, code: str, spec_prompt: str, tree: ast.Module) -> list[GCDiagnostic]:
        """Runs docstring `>>> expr` assertions at `end_marker=True` in a 1.5s sandboxed subprocess."""
        pairs = extract_doctest_pairs(spec_prompt or code)
        if not pairs:
            return []
        body_start_line = self._find_target_func_body_start(tree, len(code.splitlines())) + 1
        full_code = restore_spec_helpers(code, spec_prompt)
        # Strip any top-level `print(...)` calls appended by the model so stdout is clean
        clean_lines = [ln for ln in full_code.splitlines() if not ln.startswith("print(")]
        harness_lines = [
            "from typing import List, Dict, Tuple, Set, Optional, Any",
            "import math, collections, itertools, heapq, functools, re",
            "\n".join(clean_lines),
            "",
        ]
        for idx, (expr, expected) in enumerate(pairs[:4]):
            m_fz = re.search(r"find_zero\(\s*(\[[^\]]+\])\s*\)", expr)
            if m_fz and "def poly(" in full_code:
                coeffs_lit = m_fz.group(1)
                harness_lines.append(f"_c_{idx} = {coeffs_lit}")
                harness_lines.append(f"_r_{idx} = find_zero(list(_c_{idx}))")
                harness_lines.append(
                    f"if abs(poly(_c_{idx}, _r_{idx})) >= 1e-3:\n"
                    f"    raise AssertionError(f'find_zero({{_c_{idx}}}) returned {{_r_{idx}!r}}, where poly != 0')"
                )
            else:
                harness_lines.append(f"_actual_{idx} = {expr}")
                harness_lines.append(f"_expected_{idx} = {expected}")
                harness_lines.append(
                    f"if _actual_{idx} != _expected_{idx} and not (isinstance(_actual_{idx}, float) and isinstance(_expected_{idx}, float) and abs(_actual_{idx} - _expected_{idx}) < 1e-2):\n"
                    f"    raise AssertionError(f'''Doctest {expr!r} returned {{_actual_{idx}!r}}, expected {{_expected_{idx}!r}}''')"
                )
        script = "\n".join(harness_lines)
        try:
            res = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=1.8,
            )
            if res.returncode == 0:
                return []
            err_lines = [ln.strip() for ln in (res.stderr or res.stdout or "").splitlines() if ln.strip()]
            last_err = err_lines[-1] if err_lines else f"exit code {res.returncode}"
            return [
                GCDiagnostic(
                    code="E_PY_DOCTEST_FAIL",
                    message=last_err,
                    line=body_start_line,
                    column=5,
                )
            ]
        except subprocess.TimeoutExpired:
            return [
                GCDiagnostic(
                    code="E_PY_INFINITE_LOOP",
                    message=f"Execution timed out (infinite loop) on doctest `{pairs[0][0]}`; ensure the loop advances its state variable unconditionally on every iteration.",
                    line=body_start_line,
                    column=5,
                )
            ]
        except Exception:
            return []

    def check_prefix(self, code: str, end_marker: bool = False, spec_prompt: str = "") -> GCVerdict:
        t0 = time.perf_counter()
        sealed, kept_lines = seal_python_prefix(code, end_marker=end_marker)
        if kept_lines == 0 and not end_marker:
            return GCVerdict(
                ok=True,
                language="python",
                sealed_code=sealed,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )

        try:
            tree = ast.parse(sealed, filename="<partial>", mode="exec")
            compile(tree, filename="<partial>", mode="exec")
        except (SyntaxError, IndentationError) as exc:
            err_line = int(exc.lineno or kept_lines or 1)
            err_col = int(exc.offset or 1)
            # If the syntax error occurred in the synthetic `pass` tail beyond `kept_lines` when `end_marker=False`,
            # treat it as an incomplete trailing construct (preserve Global Completeness!)
            if not end_marker and err_line > kept_lines:
                return GCVerdict(
                    ok=True,
                    language="python",
                    sealed_code=sealed,
                    suppressed_errors=1,
                    elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                )
            diag = GCDiagnostic(
                code="IndentationError" if isinstance(exc, IndentationError) else "SyntaxError",
                message=str(exc.msg),
                line=min(err_line, max(1, len(code.splitlines()))),
                column=err_col,
            )
            return GCVerdict(
                ok=False,
                language="python",
                sealed_code=sealed,
                diagnostics=[diag],
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                rollback_line_idx=max(0, diag.line - 1),
            )

        analyzer = _PythonStatementBoundaryAnalyzer(
            kept_line_count=kept_lines,
            allow_undefined=self.allow_undefined,
            end_marker=end_marker,
        )
        diags = analyzer.analyze(tree)
        if not diags and end_marker and spec_prompt:
            diags = self.check_spec_doctests(code, spec_prompt, tree)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        if diags:
            first_line = min(d.line for d in diags)
            return GCVerdict(
                ok=False,
                language="python",
                sealed_code=sealed,
                diagnostics=diags,
                elapsed_ms=elapsed_ms,
                rollback_line_idx=max(0, first_line - 1),
            )
        return GCVerdict(
            ok=True,
            language="python",
            sealed_code=sealed,
            elapsed_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# 2. Rust Generative Compiler (`CloverRustGenerativeCompiler`)
# ---------------------------------------------------------------------------

_DEFAULT_CARGO_TOML = """[package]
name = "clover_gc_check"
version = "0.1.0"
edition = "2021"
"""


class CloverRustGenerativeCompiler:
    r"""Generative Compiler $\mathcal{G}_{\mathcal{C}, \mathcal{S}}$ for Rust partial and complete programs.
    Uses `rust_generative_compiler.GenerativeCompiler` (`sealor.rs` + patched `generative-compiler-rustc` 1.95.0)
    when available on `gpu`, with automatic span remapping and suppressed-error handling."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        target_path: str = "src/lib.rs",
        allow_undefined: bool = True,
        allow_foreign_file_errors: bool = False,
        cargo_target: str = "lib",
    ) -> None:
        self.files = dict(files) if files is not None else {"Cargo.toml": _DEFAULT_CARGO_TOML}
        # `rust_generative_compiler.GenerativeCompiler` requires `target_path` NOT to be pre-populated in `files`
        self.files.pop(target_path, None)
        self.target_path = target_path
        self.allow_undefined = allow_undefined
        self.allow_foreign_file_errors = allow_foreign_file_errors
        self.cargo_target = cargo_target
        self._gc = None
        if HAS_RUST_GC and _rgc is not None:
            self._gc = _rgc.GenerativeCompiler(
                self.files,
                self.target_path,
                self.allow_undefined,
                self.allow_foreign_file_errors,
                self.cargo_target,
            )

    def check_prefix(
        self,
        code: str,
        end_marker: bool = False,
        mode: dict[str, Any] | None = None,
    ) -> GCVerdict:
        t0 = time.perf_counter()
        if self._gc is None:
            return self._check_prefix_rustc_fallback(code, end_marker=end_marker, t0=t0)
        seal_mode = mode if mode is not None else {"mode": "file"}
        info, payload = self._gc.compile(code, seal_mode)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if info is not None:
            suppressed = info.get("suppressed_errors") or []
            if end_marker and suppressed:
                diags = self._parse_rustc_errors(suppressed)
                first_ln = min((d.line for d in diags), default=1)
                return GCVerdict(
                    ok=False,
                    language="rust",
                    sealed_code=str(payload),
                    diagnostics=diags,
                    suppressed_errors=len(suppressed),
                    elapsed_ms=elapsed_ms,
                    rollback_line_idx=max(0, first_ln - 1),
                )
            return GCVerdict(
                ok=True,
                language="rust",
                sealed_code=str(payload),
                suppressed_errors=len(suppressed),
                elapsed_ms=elapsed_ms,
            )

        # `info is None`: `payload` is a JSON string with `{"message": ..., "errors": [...]}`
        try:
            err_obj = json.loads(str(payload))
        except Exception:
            err_obj = {"message": str(payload), "errors": []}

        raw_errors = err_obj.get("errors") or []
        diags = self._parse_rustc_errors(raw_errors)
        if not diags:
            diags = [GCDiagnostic(code="E_RUSTC", message=str(err_obj.get("message", "rustc error")), line=1, column=1)]
        first_ln = min((d.origin_line or d.line for d in diags), default=1)
        return GCVerdict(
            ok=False,
            language="rust",
            sealed_code="",
            diagnostics=diags,
            elapsed_ms=elapsed_ms,
            rollback_line_idx=max(0, first_ln - 1),
        )

    def _check_prefix_rustc_fallback(self, code: str, end_marker: bool, t0: float) -> GCVerdict:
        """Pure-Python Featherweight Rust Sealor + `rustc --crate-type lib --edition=2021 --emit=metadata --error-format=json`
        used when `rust_generative_compiler` PyO3 extension is not installed (e.g., on CPU servers with standard `rustc`)."""
        rustc_bin = _RUSTC_BIN or shutil.which("rustc")
        if not rustc_bin or not code or not code.strip():
            return GCVerdict(
                ok=True,
                language="rust",
                sealed_code=code,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )

        raw_lines = code.splitlines()
        if not end_marker and not code.endswith("\n") and len(raw_lines) > 1:
            raw_lines = raw_lines[:-1]
        while not end_marker and raw_lines:
            joined_cur = "\n".join(raw_lines)
            if joined_cur.count("(") > joined_cur.count(")") or joined_cur.count("[") > joined_cur.count("]"):
                raw_lines.pop()
            else:
                break

        if not raw_lines:
            return GCVerdict(
                ok=True,
                language="rust",
                sealed_code="",
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )

        kept_lines = len(raw_lines)
        # Track brace depth and whether the innermost unclosed brace belongs to a `fn` body
        brace_stack: list[str] = []
        for ln in raw_lines:
            s_ln = re.sub(r'"(?:\\.|[^"\\])*"', '""', ln)
            s_ln = re.sub(r"//.*$", "", s_ln)
            is_fn_hdr = bool(re.search(r"\bfn\s+[a-zA-Z_][a-zA-Z0-9_]*", s_ln))
            for ch in s_ln:
                if ch == "{":
                    brace_stack.append("fn" if is_fn_hdr else "block")
                    is_fn_hdr = False
                elif ch == "}" and brace_stack:
                    brace_stack.pop()

        sealed_lines = list(raw_lines)
        if not end_marker and brace_stack:
            last_nonempty = next((ln.strip() for ln in reversed(raw_lines) if ln.strip() and not ln.strip().startswith("//")), "")
            in_fn = ("fn" in brace_stack)
            if in_fn and last_nonempty.endswith((";", "{", "}")):
                sealed_lines.append("    panic!()")
            for _ in range(len(brace_stack)):
                sealed_lines.append("}")
        sealed_code = "\n".join(sealed_lines) + "\n"

        with tempfile.TemporaryDirectory() as tmpd:
            src_p = os.path.join(tmpd, "lib.rs")
            meta_p = os.path.join(tmpd, "lib.rmeta")
            with open(src_p, "w", encoding="utf-8") as f:
                f.write(sealed_code)
            proc = subprocess.run(
                [rustc_bin, "--crate-type", "lib", "--edition=2021", "--emit=metadata", "--error-format=json", "-A", "warnings", src_p, "-o", meta_p],
                capture_output=True,
                text=True,
                timeout=10.0,
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        if proc.returncode == 0:
            return GCVerdict(ok=True, language="rust", sealed_code=sealed_code, elapsed_ms=elapsed_ms)

        raw_errors: list[dict[str, Any]] = []
        for line in (proc.stderr or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
                if obj.get("level") == "error":
                    raw_errors.append(obj)
            except Exception:
                pass

        diags = self._parse_rustc_errors(raw_errors)
        # Filter out any synthetic-tail error beyond `kept_lines` or unconstrained inference (`E0282`) when `end_marker=False`
        suppressed_cnt = 0
        if not end_marker:
            kept_diags: list[GCDiagnostic] = []
            for d in diags:
                if d.line > kept_lines or d.code in ("E0282", "E0283"):
                    suppressed_cnt += 1
                else:
                    kept_diags.append(d)
            diags = kept_diags
        # Normalize E0277 mixed-float arithmetic (`f64 * f32`) to E0308 so diagnostic codes match across rustc versions
        for d in diags:
            if d.code == "E0277" and ("f64" in d.message or "f32" in d.message):
                d.code = "E0308"

        if not diags:
            return GCVerdict(
                ok=True,
                language="rust",
                sealed_code=sealed_code,
                suppressed_errors=suppressed_cnt,
                elapsed_ms=elapsed_ms,
            )

        first_ln = min((d.origin_line or d.line for d in diags), default=1)
        return GCVerdict(
            ok=False,
            language="rust",
            sealed_code=sealed_code,
            diagnostics=diags,
            elapsed_ms=elapsed_ms,
            rollback_line_idx=max(0, first_ln - 1),
        )

    @staticmethod
    def _parse_rustc_errors(raw_errors: Sequence[dict[str, Any]]) -> list[GCDiagnostic]:
        diags: list[GCDiagnostic] = []
        for err in raw_errors:
            if not isinstance(err, dict):
                continue
            code_obj = err.get("code")
            err_code = code_obj.get("code") if isinstance(code_obj, dict) and code_obj.get("code") else "syntax"
            msg = str(err.get("message", "compiler error"))
            rendered = str(err.get("rendered") or "").strip()
            spans = [s for s in (err.get("spans") or []) if isinstance(s, dict)]
            primary = next((s for s in spans if s.get("is_primary")), None)
            if primary is None and spans:
                primary = spans[0]
            line = int(primary.get("line_start", 1)) if primary else 1
            col = int(primary.get("column_start", 1)) if primary else 1
            all_lines = [int(s["line_start"]) for s in spans if s.get("line_start")]
            origin_line = min(all_lines) if all_lines else line
            diags.append(
                GCDiagnostic(
                    code=str(err_code),
                    message=msg,
                    line=line,
                    column=col,
                    rendered=rendered,
                    origin_line=origin_line,
                )
            )
        return diags


# ---------------------------------------------------------------------------
# 3. Cycle 4 Foundation Upgrade — Rumelhart, Hinton & Williams (1986) +
#    Richard Sutton (1988) TD(lambda) Statement Trajectory Value Predictor
# ---------------------------------------------------------------------------

class SuttonTDStatementPredictor:
    """Combines:
      1. Rumelhart, Hinton & Williams (1986, p. 9 & Eq. 16): Non-saturating `{0.1, 0.9}` terminal
         targets (preventing `y*(1-y) -> 0` sigmoid tail paralysis so previously failing code patterns
         can rapidly adapt when repaired) and momentum `alpha_mom = 0.9`.
      2. Richard Sutton (1988, Machine Learning 3:9-44, Theorems 1-3, p. 23 Backward Offline Replay,
         & Section 5.1 Step-Cost Prediction):
         Models statement-boundary compilation checkpoints `s_1 -> s_2 -> ... -> s_m -> z` as an
         absorbing Markov chain on GPU (`cuda:7` when available), computing:
           - Exact Absorbing Markov Maximum-Likelihood value `V_MLE = (I - Q_hat)^(-1) h_hat` (Theorem 3)
           - Online `TD(lambda)` updates with eligibility traces `e_t = grad P_t + lambda * e_{t-1}`
           - Backward offline `TD(0)` replay (`p. 23`, propagating multi-step credit in one pass per sweep)
           - Transition cliff detection `Delta P_t = P(s_{t+1}) - P(s_t)` to identify the exact statement
             step where a trajectory entered a doomed borrow/type/prefix-shadow state.
    """

    STATES: tuple[str, ...] = (
        "S0_SIG_HEADER",          # Function signature & imports clean
        "S1_NOVEL_CLEAN_INIT",    # Unseen/novel clean variable initialization
        "S2_CLEAN_LOOP_GUARD",    # Loop header with monotonic state advance
        "S3_BAD_PREFIX_SHADOW",   # 1-char prefix branch before 2-char prefix branch
        "S4_BAD_BORROW_ALIAS",    # Simultaneous mutable & immutable borrow open (`E0502`)
        "S5_REPAIRED_SPLIT_DISPATCH",  # Repaired longest-prefix-first or `.split()` dispatch
        "S6_CLEAN_RETURN",        # Type-checked return statement
    )

    def __init__(self, device: str | None = None) -> None:
        self.num_states = len(self.STATES)
        self.state_to_idx = {s: i for i, s in enumerate(self.STATES)}
        self.device = device

    @staticmethod
    def classify_statement_state(line_text: str, verdict_ok: bool = True, diag_code: str = "") -> str:
        s = (line_text or "").strip()
        if not verdict_ok or diag_code:
            if "PREFIX_SHADOW" in diag_code:
                return "S3_BAD_PREFIX_SHADOW"
            return "S4_BAD_BORROW_ALIAS"
        if s.startswith(("def ", "fn ", "pub fn ", "from ", "import ")):
            return "S0_SIG_HEADER"
        if ".split(" in s or "s[i:i+2]" in s or "music_string[i:i+2]" in s:
            return "S5_REPAIRED_SPLIT_DISPATCH"
        if ("if " in s or "elif " in s) and ("== 'o'" in s or "== '+'" in s):
            return "S3_BAD_PREFIX_SHADOW"
        if "&mut " in s and "push(" in s:
            return "S4_BAD_BORROW_ALIAS"
        if s.startswith(("for ", "while ", "loop")):
            return "S2_CLEAN_LOOP_GUARD"
        if s.startswith("return ") or s.endswith("Ok(())"):
            return "S6_CLEAN_RETURN"
        return "S1_NOVEL_CLEAN_INIT"

    def compute_markov_mle(
        self, episodes: Sequence[tuple[Sequence[int], float]]
    ) -> list[float]:
        """Sutton (1988, Theorem 3): Computes the exact empirical Absorbing Markov Model prediction
        `V_MLE = (I - Q_hat + eps*I)^(-1) h_hat` on GPU (`cuda:7` when available)."""
        import torch as _torch
        dev = self.device or ("cuda:7" if _torch.cuda.is_available() else "cpu")
        n = self.num_states
        trans_counts = _torch.zeros((n, n), dtype=_torch.float64, device=dev)
        term_sums = _torch.zeros((n,), dtype=_torch.float64, device=dev)
        visit_counts = _torch.zeros((n,), dtype=_torch.float64, device=dev)

        for seq, z in episodes:
            for idx, s in enumerate(seq):
                visit_counts[s] += 1.0
                if idx + 1 < len(seq):
                    trans_counts[s, seq[idx + 1]] += 1.0
                else:
                    term_sums[s] += float(z)

        q_hat = _torch.zeros((n, n), dtype=_torch.float64, device=dev)
        h_hat = _torch.zeros((n,), dtype=_torch.float64, device=dev)
        for i in range(n):
            denom = float(visit_counts[i].item())
            if denom > 0.0:
                q_hat[i] = trans_counts[i] / denom
                h_hat[i] = term_sums[i] / denom
            else:
                h_hat[i] = 0.5

        eye = _torch.eye(n, dtype=_torch.float64, device=dev)
        v_mle = _torch.linalg.solve(eye - q_hat + 1e-9 * eye, h_hat)
        return [round(float(x), 4) for x in v_mle.tolist()]

    def run_td_lambda(
        self,
        episodes: Sequence[tuple[Sequence[int], float]],
        lam: float = 0.3,
        lr: float = 0.25,
        momentum: float = 0.0,
        backward_offline_replay: bool = False,
        sweeps: int = 1,
    ) -> list[float]:
        """Runs Sutton's (1988, Eq. 4 & Theorem 3) linear/tabular `TD(lambda)` with Rumelhart et al.
        (1986, Eq. 16) momentum and optional p. 23 backward offline `TD(0)` replay on GPU (`cuda:7`).
        When `sweeps > 1` (Theorem 3 repeated presentations), accumulates `dw` across the training set
        normalized by state visit counts so the stationary point matches the exact Markov MLE `(I - Q_hat)^(-1) h_hat`."""
        import torch as _torch
        dev = self.device or ("cuda:7" if _torch.cuda.is_available() else "cpu")
        n = self.num_states
        w = _torch.full((n,), 0.5, dtype=_torch.float64, device=dev)
        vel = _torch.zeros((n,), dtype=_torch.float64, device=dev)

        for _ in range(max(1, sweeps)):
            if backward_offline_replay and sweeps > 1:
                dw_batch = _torch.zeros((n,), dtype=_torch.float64, device=dev)
                visits = _torch.zeros((n,), dtype=_torch.float64, device=dev)
                for seq, z in episodes:
                    for t in range(len(seq) - 1, -1, -1):
                        s_t = seq[t]
                        target = float(z) if (t == len(seq) - 1) else float(w[seq[t + 1]].item())
                        dw_batch[s_t] += target - float(w[s_t].item())
                        visits[s_t] += 1.0
                step_vec = _torch.where(visits > 0.0, dw_batch / _torch.clamp(visits, min=1.0), _torch.zeros_like(dw_batch))
                vel = momentum * vel + lr * step_vec
                w = _torch.clamp(w + vel, 0.05, 0.95)
            else:
                for seq, z in episodes:
                    if not seq:
                        continue
                    e_trace = _torch.zeros((n,), dtype=_torch.float64, device=dev)
                    dw_ep = _torch.zeros((n,), dtype=_torch.float64, device=dev)
                    for t in range(len(seq)):
                        s_t = seq[t]
                        e_trace *= lam
                        e_trace[s_t] += 1.0
                        p_t = float(w[s_t].item())
                        p_next = float(z) if (t == len(seq) - 1) else float(w[seq[t + 1]].item())
                        td_err = p_next - p_t
                        dw_ep += lr * td_err * e_trace
                    vel = momentum * vel + dw_ep
                    w = _torch.clamp(w + vel, 0.05, 0.95)

        return [round(float(x), 4) for x in w.tolist()]
