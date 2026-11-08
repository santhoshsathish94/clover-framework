"""Hardened Isolated Python Execution Sandbox (`isolated_sandbox.py`).

Provides an OS- and runtime-isolated execution boundary for model-generated verification scripts
and candidate problem solutions in Clover's learning and verification pipelines (replacing in-process `exec()`).

Defense-in-depth layers:
  1. **Static AST Allowlist Gate (`validate_sandbox_ast`)**:
     - Rejects any `import` / `from ... import` outside pure mathematical and data-structure modules
       (`math`, `itertools`, `functools`, `collections`, `heapq`, `bisect`, `fractions`, `decimal`,
       `statistics`, `random`, `typing`, `re`, `dataclasses`, `operator`, `copy`, `string`).
     - Rejects dangerous builtins (`open`, `eval`, `exec`, `compile`, `__import__`, `input`, `breakpoint`,
       `globals`, `locals`, `vars`, `getattr`, `setattr`, `delattr`, `dir`, `memoryview`).
     - Rejects any dunder attribute access (`__class__`, `__subclasses__`, `__globals__`, `__builtins__`,
       `__code__`, `__dict__`, `__mro__`, etc.) to block object-graph sandbox escapes.
  2. **Child Process Isolation (`python -I -S -B`)**:
     - Runs in a separate child process with `-I` (isolated mode), `-S` (no `site`), and `-B` (no `.pyc` writes).
     - Executes in an empty ephemeral working directory with stripped environment variables.
  3. **Kernel Resource & Process Limits**:
     - On POSIX (Linux): sets `RLIMIT_AS` (address-space cap, default 256 MiB), `RLIMIT_CPU` (CPU seconds cap),
       `RLIMIT_NPROC = 0` (cannot fork/spawn child processes), and `RLIMIT_FSIZE = 0` (cannot create or write files).
     - On Windows: binds the child process to a Win32 Job Object with `JOB_OBJECT_LIMIT_PROCESS_MEMORY` (256 MiB),
       `JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 1` (blocks spawning child processes at the kernel level), and
       `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.
  4. **CPython Runtime Audit Hook (`sys.addaudithook`)**:
     - Installed inside the child process before user code runs; blocks `open`, `socket.*`, `os.*`, `subprocess.*`,
       `shutil.*`, `ctypes.*`, `urllib.*`, `winreg.*`, and unauthorized module imports at the CPython C-API level.
"""
from __future__ import annotations

import ast
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "math",
    "itertools",
    "functools",
    "collections",
    "heapq",
    "bisect",
    "fractions",
    "decimal",
    "statistics",
    "random",
    "typing",
    "re",
    "dataclasses",
    "operator",
    "copy",
    "string",
    "time",
})

FORBIDDEN_BUILTINS: frozenset[str] = frozenset({
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "dir",
    "help",
    "memoryview",
    "exit",
    "quit",
})


def validate_sandbox_ast(code: str) -> tuple[bool, str]:
    """Validate `code` statically using Python's AST before spawning the sandbox process."""
    if not isinstance(code, str) or not code.strip():
        return False, "Empty verification code"
    if len(code) > 64_000:
        return False, "Code exceeds 64 KB sandbox limit"
    try:
        tree = ast.parse(code, filename="<sandbox>", mode="exec")
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_mod = alias.name.split(".", 1)[0]
                if top_mod not in ALLOWED_IMPORTS:
                    return False, f"SecurityError: import of module {alias.name!r} is not permitted in sandbox"
        elif isinstance(node, ast.ImportFrom):
            top_mod = (node.module or "").split(".", 1)[0]
            if node.level > 0 or top_mod not in ALLOWED_IMPORTS:
                return False, f"SecurityError: from-import of {node.module!r} is not permitted in sandbox"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr.endswith("__"):
                return False, f"SecurityError: dunder attribute access {node.attr!r} is not permitted in sandbox"
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_BUILTINS:
                return False, f"SecurityError: builtin {node.id!r} is not permitted in sandbox"
            if node.id.startswith("__") and node.id != "__name__":
                return False, f"SecurityError: dunder identifier {node.id!r} is not permitted in sandbox"
    return True, "ok"


_CHILD_RUNNER = r"""
import builtins
import json
import math
import sys

mem_limit_bytes = int(sys.argv[1])
cpu_limit_s = max(1, int(float(sys.argv[2])) + 1)
allowed_mods = set(json.loads(sys.argv[3]))
user_code = sys.argv[4]
sys.argv[:] = ["-c"]

# 1. Pre-import allowed standard library modules before locking down the audit hook
for _m in ("math", "itertools", "functools", "collections", "heapq", "bisect",
           "fractions", "decimal", "statistics", "random", "re", "operator", "copy", "string", "time"):
    try:
        __import__(_m)
    except Exception:
        pass

# 2. Apply POSIX kernel resource limits (address space, CPU time, zero file writes, zero forks)
try:
    import resource
    if hasattr(resource, "RLIMIT_AS") and mem_limit_bytes > 0:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
    if hasattr(resource, "RLIMIT_CPU"):
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_s, cpu_limit_s))
    if hasattr(resource, "RLIMIT_FSIZE"):
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
except Exception:
    pass

# 3. Compile user code before installing the import/compile audit lock
_compiled = compile(user_code, "<isolated_sandbox>", "exec")

# 4. Install safe builtins dictionary (removing open, eval, exec, compile, input, getattr, etc.)
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr",
    "complex", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "hash", "hex", "id", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "object", "oct", "ord", "pow", "print", "property", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "staticmethod", "str", "sum",
    "super", "tuple", "type", "zip",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception", "FloatingPointError",
    "IndexError", "KeyError", "LookupError", "MemoryError", "NameError", "NotImplementedError",
    "OverflowError", "RuntimeError", "StopIteration", "SyntaxError", "TypeError",
    "UnboundLocalError", "ValueError", "ZeroDivisionError", "True", "False", "None",
)
_orig_import = builtins.__import__

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = str(name or "").split(".", 1)[0]
    if level != 0 or top not in allowed_mods:
        raise PermissionError(f"Sandbox blocked import of {name!r}")
    return _orig_import(name, globals, locals, fromlist, level)

_safe_builtins = {k: getattr(builtins, k) for k in _SAFE_BUILTIN_NAMES if hasattr(builtins, k)}
_safe_builtins["__import__"] = _safe_import

# 5. Install CPython runtime audit hook blocking filesystem, network, process, and OS events
_BLOCKED_PREFIXES = (
    "open", "os.", "subprocess.", "socket.", "urllib.", "http.", "ftplib.", "smtplib.",
    "ctypes.", "shutil.", "pathlib.", "tempfile.", "glob.", "msvcrt.", "winreg.",
    "fcntl.", "mmap.", "builtins.input", "builtins.breakpoint", "sys._getframe",
)

def _audit_guard(event, args):
    if event.startswith(_BLOCKED_PREFIXES):
        raise PermissionError(f"Sandbox audit hook blocked event: {event}")
    if event == "import":
        mod = str(args[0] if args else "").split(".", 1)[0]
        if mod not in allowed_mods:
            raise PermissionError(f"Sandbox audit hook blocked import: {mod}")

sys.addaudithook(_audit_guard)

def _to_jsonable(v):
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return str(v)
        return v
    if isinstance(v, tuple):
        return {"__tuple__": [_to_jsonable(x) for x in v]}
    if isinstance(v, list):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, set):
        return {"__set__": sorted(_to_jsonable(x) for x in v)}
    return repr(v)

_ns = {"__builtins__": _safe_builtins, "__name__": "__main__"}
try:
    exec(_compiled, _ns, _ns)
    if "RESULT" in _ns and _ns["RESULT"] is not None:
        _res = _ns["RESULT"]
    elif "result" in _ns and _ns["result"] is not None:
        _res = _ns["result"]
    else:
        _res = True
    sys.stdout.write("\n__CLOVER_SANDBOX_JSON__:" + json.dumps({"ok": True, "result": _to_jsonable(_res)}) + "\n")
    sys.stdout.flush()
except BaseException as _exc:
    _tb_curr = _exc.__traceback__
    _lineno = -1
    while _tb_curr is not None:
        _lineno = _tb_curr.tb_lineno
        _tb_curr = _tb_curr.tb_next
    _u_lines = user_code.splitlines()
    _src_ln = _u_lines[_lineno - 1].strip() if 1 <= _lineno <= len(_u_lines) else ""
    sys.stdout.write("\n__CLOVER_SANDBOX_JSON__:" + json.dumps({
        "ok": False,
        "error": f"{type(_exc).__name__} (line {_lineno}: {_src_ln}): {_exc}",
    }) + "\n")
    sys.stdout.flush()
    raise SystemExit(1)
"""


def _from_jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        if "__tuple__" in v and len(v) == 1:
            return tuple(_from_jsonable(x) for x in v["__tuple__"])
        if "__set__" in v and len(v) == 1:
            return set(_from_jsonable(x) for x in v["__set__"])
        return {k: _from_jsonable(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_from_jsonable(x) for x in v]
    return v


# Windows Job Object support with ActiveProcessLimit = 1 + ProcessMemoryLimit + KillOnJobClose
if os.name == "nt":
    from ctypes import wintypes

    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    _JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _CREATE_NO_WINDOW = 0x08000000

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]

    def _create_sandbox_job(mem_limit_bytes: int):
        job = _k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | _JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        )
        info.BasicLimitInformation.ActiveProcessLimit = 1
        info.ProcessMemoryLimit = int(mem_limit_bytes)
        if not _k32.SetInformationJobObject(
            job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        ):
            _k32.CloseHandle(job)
            return None
        return job


class IsolatedPythonSandbox:
    """Executes untrusted/model-generated Python verification scripts inside a locked-down subprocess."""

    def __init__(
        self,
        python_executable: str | None = None,
        timeout_s: float = 5.0,
        memory_limit_mb: int = 256,
        max_output_chars: int = 8000,
    ) -> None:
        self.python_executable = python_executable or sys.executable
        self.timeout_s = float(timeout_s)
        self.memory_limit_bytes = int(memory_limit_mb) * 1024 * 1024
        self.max_output_chars = int(max_output_chars)

    def run(self, code: str) -> tuple[bool, Any, str]:
        """Run `code` in the isolated sandbox and return `(ok, result_value, message_or_stdout)`."""
        ast_ok, ast_msg = validate_sandbox_ast(code)
        if not ast_ok:
            return False, None, ast_msg

        with tempfile.TemporaryDirectory(prefix="clover_sbx_") as empty_cwd:
            try:
                os.chmod(empty_cwd, 0o700)
            except OSError:
                pass
            cmd = [
                self.python_executable,
                "-I",
                "-S",
                "-B",
                "-c",
                _CHILD_RUNNER,
                str(self.memory_limit_bytes),
                str(self.timeout_s),
                json.dumps(sorted(ALLOWED_IMPORTS)),
                code,
            ]
            clean_env = {"PYTHONIOENCODING": "utf-8"}
            if os.name == "nt":
                # Windows side-by-side CRT loader needs SystemRoot
                clean_env["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")

            job = _create_sandbox_job(self.memory_limit_bytes) if os.name == "nt" else None
            creationflags = _CREATE_NO_WINDOW if os.name == "nt" else 0
            proc = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=empty_cwd,
                    env=clean_env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=creationflags,
                )
                if job is not None and os.name == "nt":
                    _k32.AssignProcessToJobObject(job, int(proc._handle))
                try:
                    out, err = proc.communicate(timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    if job is not None and os.name == "nt":
                        _k32.TerminateJobObject(job, 1)
                    else:
                        proc.kill()
                    proc.communicate()
                    return False, None, f"TimeoutError: sandbox execution exceeded {self.timeout_s:.1f}s limit"
            finally:
                if job is not None and os.name == "nt":
                    _k32.CloseHandle(job)

            marker = "__CLOVER_SANDBOX_JSON__:"
            if marker in (out or ""):
                head, payload_str = out.rsplit(marker, 1)
                user_stdout = head.strip()[: self.max_output_chars]
                try:
                    payload = json.loads(payload_str.strip().splitlines()[0])
                except Exception as exc:
                    return False, None, f"SandboxProtocolError: {exc}"
                if payload.get("ok"):
                    val = _from_jsonable(payload.get("result"))
                    msg = user_stdout if user_stdout else "ok"
                    return True, val, msg
                return False, None, str(payload.get("error") or "Execution error")[: self.max_output_chars]

            err_txt = (err or out or "Process terminated abnormally").strip()[: self.max_output_chars]
            return False, None, f"SandboxProcessError (rc={proc.returncode if proc else -1}): {err_txt}"


_DEFAULT_SANDBOX: IsolatedPythonSandbox | None = None


def run_in_isolated_sandbox(code: str, timeout_s: float = 5.0) -> tuple[bool, Any, str]:
    global _DEFAULT_SANDBOX
    if _DEFAULT_SANDBOX is None or _DEFAULT_SANDBOX.timeout_s != timeout_s:
        _DEFAULT_SANDBOX = IsolatedPythonSandbox(timeout_s=timeout_s)
    return _DEFAULT_SANDBOX.run(code)
