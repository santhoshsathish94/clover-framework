"""Android runner for Clover's PythonTool (hybrid_agent/tools/python_tool.py).

Android cannot start a second Python: the app has no python program, and programs run only from its library folder.
So the code runs in this process, on its own thread, with the interface, timeout, output caps and diagnostic strings
of upstream ("Python exited with code N", "Execution Timeout: ...", "OutputLimitExceeded:"). What differs: no memory
limit; a run that passes its time limit is stopped with an exception at its next Python step (code waiting inside a C
call finishes that call first); each run gets fresh globals and the working folder it sets is put back afterwards.
"""
import builtins
import ctypes
import io
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Tuple


def _upstream():
    try:
        from hybrid_agent.tools.python_tool import PythonTool as Upstream
        return Upstream
    except ImportError:  # no torch on the phone: hybrid_agent/__init__.py loads its torch engine, the tool needs none of it
        import importlib.util
        import types
        tools = Path(__file__).resolve().parent / "hybrid_agent" / "tools"
        pkg = types.ModuleType("_clover_tools")
        pkg.__path__ = [str(tools)]
        sys.modules.setdefault("_clover_tools", pkg)
        spec = importlib.util.spec_from_file_location("_clover_tools.python_tool", tools / "python_tool.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod.PythonTool


_UpstreamPythonTool = _upstream()

_RUNS = {}  # thread id -> _Run: output written on a tool thread goes to that run
_ROUTE_LOCK = threading.Lock()


class _Stop(BaseException):
    """Raised inside a run that passed its time or output limit."""


class _Run:
    def __init__(self, max_chars, max_bytes):
        self.max_chars, self.max_bytes = max_chars, max_bytes
        self.parts = {"stdout": [], "stderr": []}
        self.chars = self.raw = 0
        self.reason = None

    def write(self, name, text):
        text = str(text)
        if self.reason is not None:
            return len(text)
        raw = len(text.encode("utf-8", "replace"))
        if self.raw + raw > self.max_bytes:
            self.reason = f"raw stdout+stderr exceeded {self.max_bytes} bytes"
        room = self.max_chars - self.chars
        if room > 0:
            self.parts[name].append(text[:room])
            self.chars += min(len(text), room)
        if len(text) > room and self.reason is None:
            self.reason = f"decoded output exceeded {self.max_chars} characters"
        self.raw += raw
        return len(text)

    def text(self, name):
        return "".join(self.parts[name]).strip()


class _Router(io.TextIOBase):
    """Installed once as sys.stdout / sys.stderr: a tool thread's writes go to its run, the rest where they went."""

    def __init__(self, orig, name):
        self.orig, self.name = orig, name

    def write(self, s):
        run = _RUNS.get(threading.get_ident())
        return run.write(self.name, s) if run is not None else self.orig.write(s)

    def flush(self):
        if threading.get_ident() not in _RUNS:
            self.orig.flush()

    def writable(self):
        return True

    @property
    def encoding(self):
        return "utf-8"


def _install_router():
    with _ROUTE_LOCK:
        if not isinstance(sys.stdout, _Router):
            sys.stdout = _Router(sys.stdout, "stdout")
        if not isinstance(sys.stderr, _Router):
            sys.stderr = _Router(sys.stderr, "stderr")


def _raise_in(thread_id):
    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), ctypes.py_object(_Stop))


class PythonTool(_UpstreamPythonTool):
    def _run_code(self, code: str, timeout: float, max_output_chars: int, max_output_bytes: int,
                  memory_limit_gb: float | None) -> Tuple[str, str]:
        code = self._preprocess_code(code)
        _install_router()
        run = _Run(max_output_chars, max_output_bytes)
        result = {"code": None}
        cwd = os.getcwd()

        def body():
            _RUNS[threading.get_ident()] = run
            g = {"__name__": "__main__", "__builtins__": builtins}
            try:
                exec(compile(code, "<python-tool>", "exec"), g, g)
                result["code"] = 0
            except SystemExit as exc:
                c = exc.code
                result["code"] = 0 if c is None else (c if isinstance(c, int) else 1)
                if c is not None and not isinstance(c, int):
                    run.write("stderr", str(c) + "\n")
            except _Stop:
                result["code"] = None
            except MemoryError:
                run.write("stderr", "MemoryLimitExceeded: Python ran out of memory\n")
                result["code"] = 137
            except BaseException:  # noqa: BLE001 - what `python -c` prints for an uncaught exception
                run.write("stderr", traceback.format_exc())
                result["code"] = 1
            finally:
                _RUNS.pop(threading.get_ident(), None)
                try:
                    if "matplotlib.pyplot" in sys.modules:  # figures left open would pile up in this process
                        sys.modules["matplotlib.pyplot"].close("all")
                except Exception:  # noqa: BLE001
                    pass

        t = threading.Thread(target=body, daemon=True, name="python-tool")
        t.start()
        timed_out = False
        step = 0.05
        waited = 0.0
        while t.is_alive():
            t.join(step)
            waited += step
            if run.reason is not None:
                _raise_in(t.ident)
                t.join(1.0)
                break
            if waited >= timeout:
                timed_out = True
                _raise_in(t.ident)
                t.join(1.0)
                break
        try:
            os.chdir(cwd)
        except OSError:
            pass
        stdout, stderr = run.text("stdout"), run.text("stderr")
        if run.reason is not None:
            return "", self._output_limit_diagnostic(run.reason, stdout, stderr, max_output_chars)
        if timed_out:
            return "", self._diagnostic(f"Execution Timeout: exceeded {timeout:g} seconds", stdout, stderr, max_output_chars)
        rc = result["code"]
        if rc == 0:
            return self._bounded(stdout, max_output_chars), "Done"
        if stderr.startswith("MemoryLimitExceeded:"):
            return "", self._diagnostic(stderr, stdout, "", max_output_chars)
        return "", self._diagnostic(f"Python exited with code {rc if rc is not None else 1}", stdout, stderr,
                                    max_output_chars)
