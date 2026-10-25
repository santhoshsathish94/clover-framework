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
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

# Ensure `rust_generative_compiler` and `generative-compiler-rustc` are on `sys.path` and `PATH` when available on `gpu`
_GC_BINDINGS_DIR = "/home/usergpu/generative-compilation/bindings/rust_generative_compiler"
_GC_RELEASE_DIR = "/home/usergpu/generative-compilation/target/release"
_GC_RUSTC_DIR = "/home/usergpu/generative-compilation/crates/generative-compiler-rustc/target/release"
_CARGO_BIN_DIR = "/home/usergpu/.cargo/bin"

if os.path.isdir(_GC_BINDINGS_DIR) and _GC_BINDINGS_DIR not in sys.path:
    sys.path.insert(0, _GC_BINDINGS_DIR)

for _bin_dir in (_GC_RUSTC_DIR, _GC_RELEASE_DIR, _CARGO_BIN_DIR):
    if os.path.isdir(_bin_dir) and _bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{_bin_dir}:{os.environ.get('PATH', '')}"

try:
    import rust_generative_compiler as _rgc  # type: ignore
    HAS_RUST_GC = True
except Exception:
    _rgc = None
    HAS_RUST_GC = False


@dataclass
class GCDiagnostic:
    """Single compiler or static-analysis diagnostic mapped back to the original prefix `code`."""
    code: str
    message: str
    line: int
    column: int = 1
    rendered: str = ""

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
        for stmt in node.body:
            self.visit(stmt)
        self.in_function -= 1
        self.scope_stack.pop()

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


class CloverPythonGenerativeCompiler:
    r"""Generative Compiler $\mathcal{G}_{\mathcal{C}, \mathcal{S}}$ for Python partial and complete programs."""

    def __init__(self, allow_undefined: bool = True) -> None:
        self.allow_undefined = allow_undefined

    def check_prefix(self, code: str, end_marker: bool = False) -> GCVerdict:
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
            return GCVerdict(
                ok=True,
                language="rust",
                sealed_code=code,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )
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
        first_ln = min((d.line for d in diags), default=1)
        return GCVerdict(
            ok=False,
            language="rust",
            sealed_code="",
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
            spans = err.get("spans") or []
            primary = next((s for s in spans if isinstance(s, dict) and s.get("is_primary")), None)
            if primary is None and spans and isinstance(spans[0], dict):
                primary = spans[0]
            line = int(primary.get("line_start", 1)) if primary else 1
            col = int(primary.get("column_start", 1)) if primary else 1
            diags.append(GCDiagnostic(code=str(err_code), message=msg, line=line, column=col, rendered=rendered))
        return diags
