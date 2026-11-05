"""Windows runner for Clover's PythonTool (hybrid_agent/tools/python_tool.py).

Same interface, timeout, output caps and diagnostic strings as upstream; the three POSIX-only parts are
replaced: RLIMIT_AS -> job-object per-process memory limit, selectors on pipes -> reader threads,
killpg -> TerminateJobObject (kills the whole process tree, including the venv launcher's child).
"""
import codecs
import ctypes
import os
import subprocess
import threading
import time
from typing import Tuple

from android_python_tool import _upstream

_UpstreamPythonTool = _upstream()
if os.name == "nt":
    from ctypes import wintypes
else:
    wintypes = None

# Same user-code execution as upstream's bootstrap, without the POSIX rlimit (the job object enforces it).
_BOOTSTRAP = r"""
import sys

limit_bytes = int(sys.argv[1])
source = sys.argv[2]
sys.argv[:] = ["-c"]
def _execute_user_code(source_text, memory_limit):
    import sys as diagnostic_sys

    try:
        exec(
            compile(source_text, "<python-tool>", "exec"),
            globals(),
            globals(),
        )
    except MemoryError:
        print(
            f"MemoryLimitExceeded: Python exceeded its {memory_limit} byte "
            "address-space limit",
            file=diagnostic_sys.stderr,
            flush=True,
        )
        raise SystemExit(137)


_execute_user_code(source, limit_bytes)
"""

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_CREATE_NO_WINDOW = 0x08000000

if os.name == "nt":
    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]

    def _new_job(memory_limit_bytes):
        job = _k32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit_bytes is not None:
            info.BasicLimitInformation.LimitFlags |= _JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = memory_limit_bytes
        if not _k32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)):
            err = ctypes.get_last_error()
            _k32.CloseHandle(job)
            raise OSError(err, "SetInformationJobObject failed")
        return job


class PythonTool(_UpstreamPythonTool):
    def _run_code(self, code: str, timeout: float, max_output_chars: int, max_output_bytes: int,
                  memory_limit_gb: float | None) -> Tuple[str, str]:
        if os.name != "nt":
            return super()._run_code(code, timeout, max_output_chars, max_output_bytes, memory_limit_gb)
        code = self._preprocess_code(code)
        limit_bytes = int(memory_limit_gb * 1024**3) if memory_limit_gb is not None else None
        command = [self._python_path, "-c", _BOOTSTRAP, str(limit_bytes or 0), code]
        try:
            job = _new_job(limit_bytes)
        except OSError as exc:
            return "", self._bounded(f"MemoryLimitConfigurationError: could not create job object: {exc}", max_output_chars)
        try:
            # output is decoded as UTF-8 below; without this the child writes the ANSI code page and print('π') fails
            # when the server was not started by the Studio launcher (which sets it), e.g. by restart_studio.ps1
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    bufsize=0, creationflags=_CREATE_NO_WINDOW,
                                    env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        except Exception as exc:
            _k32.CloseHandle(job)
            return "", self._bounded(f"Error: could not start Python process: {exc}", max_output_chars)
        if not _k32.AssignProcessToJobObject(job, int(proc._handle)):
            err = ctypes.get_last_error()
            proc.kill()
            proc.wait()
            _k32.CloseHandle(job)
            return "", self._bounded(f"MemoryLimitConfigurationError: could not assign job object: error {err}", max_output_chars)

        lock = threading.Lock()
        limit_hit = threading.Event()
        state = {"raw": 0, "chars": 0, "reason": None,
                 "stdout": [], "stderr": [],
                 "dec": {n: codecs.getincrementaldecoder("utf-8")(errors="replace") for n in ("stdout", "stderr")}}

        def append_text(name, text):
            if not text:
                return False
            remaining = max_output_chars - state["chars"]
            if remaining > 0:
                state[name].append(text[:remaining])
                state["chars"] += min(len(text), remaining)
            return len(text) > remaining

        def reader(pipe, name):
            fd = pipe.fileno()
            while True:
                try:
                    chunk = os.read(fd, 65_536)
                except OSError:
                    chunk = b""
                with lock:
                    if not chunk:
                        if append_text(name, state["dec"][name].decode(b"", final=True)) and state["reason"] is None:
                            state["reason"] = f"decoded output exceeded {max_output_chars} characters"
                            limit_hit.set()
                        return
                    if state["reason"] is not None:
                        return
                    kept = chunk[: max(0, max_output_bytes - state["raw"])]
                    state["raw"] += len(kept)
                    if kept and append_text(name, state["dec"][name].decode(kept, final=False)):
                        state["reason"] = f"decoded output exceeded {max_output_chars} characters"
                    if len(chunk) > len(kept):
                        state["reason"] = f"raw stdout+stderr exceeded {max_output_bytes} bytes"
                    if state["reason"] is not None:
                        limit_hit.set()
                        return

        threads = [threading.Thread(target=reader, args=(p, n), daemon=True)
                   for p, n in ((proc.stdout, "stdout"), (proc.stderr, "stderr"))]
        for t in threads:
            t.start()

        deadline = time.monotonic() + timeout
        timed_out = False
        try:
            while True:
                if limit_hit.is_set():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    proc.wait(timeout=min(0.05, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            # The leader is done (or must be stopped): kill anything left in the tree, then drain briefly.
            _k32.TerminateJobObject(job, 1)
            for t in threads:
                t.join(timeout=0.25 if not (timed_out or limit_hit.is_set()) else 1.0)
            if proc.poll() is None:
                proc.wait(timeout=1.0)
        except BaseException:
            _k32.TerminateJobObject(job, 1)
            raise
        finally:
            _k32.CloseHandle(job)
            for p in (proc.stdout, proc.stderr):
                try:
                    p.close()
                except Exception:
                    pass

        with lock:
            stdout = "".join(state["stdout"]).strip()
            stderr = "".join(state["stderr"]).strip()
            limit_reason = state["reason"]
        if limit_reason is not None:
            return "", self._output_limit_diagnostic(limit_reason, stdout, stderr, max_output_chars)
        if timed_out:
            return "", self._diagnostic(f"Execution Timeout: exceeded {timeout:g} seconds", stdout, stderr, max_output_chars)
        returncode = proc.returncode
        if returncode == 0:
            return self._bounded(stdout, max_output_chars), "Done"
        if stderr.startswith("MemoryLimitExceeded:"):
            return "", self._diagnostic(stderr, stdout, "", max_output_chars)
        return "", self._diagnostic(f"Python exited with code {returncode}", stdout, stderr, max_output_chars)
