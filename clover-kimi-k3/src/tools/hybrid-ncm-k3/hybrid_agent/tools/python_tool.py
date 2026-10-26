"""Python interpreter tool: runs the code of a <python> call in a separate subprocess
with a timeout, an address-space limit and an output cap; the captured output becomes
the <result> observation.
"""
import ast
import codecs
import math
import os
import selectors
import signal
import subprocess
import sys
import time
from typing import Tuple

from .base import BaseTool


DEFAULT_MAX_OUTPUT_CHARS = 2_000
DEFAULT_MEMORY_LIMIT_GB = 16.0


# Apply the address-space limit in the new interpreter, rather than through
# Popen(preexec_fn=...), which is unsafe in a multithreaded caller.
_MEMORY_LIMIT_BOOTSTRAP = r"""
import resource
import sys

limit_bytes = int(sys.argv[1])
source = sys.argv[2]
try:
    inherited_soft, inherited_hard = resource.getrlimit(resource.RLIMIT_AS)
    finite_inherited = [
        value
        for value in (inherited_soft, inherited_hard)
        if value != resource.RLIM_INFINITY
    ]
    effective_limit = min([limit_bytes, *finite_inherited])
    resource.setrlimit(
        resource.RLIMIT_AS, (effective_limit, effective_limit)
    )
except Exception as exc:
    print(
        f"MemoryLimitConfigurationError: could not bound RLIMIT_AS at "
        f"{limit_bytes} bytes: {exc}",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(125)

# Match the useful parts of ``python -c`` for user code.
sys.argv[:] = ["-c"]
def _execute_user_code(source_text, memory_limit):
    # Keep diagnostic references in function locals so generated code cannot
    # accidentally replace them through globals before raising MemoryError.
    import sys as diagnostic_sys

    try:
        # Execute in the real __main__ module, just like ``python -c``.  A
        # detached dict breaks pickle/multiprocessing lookup for user-defined
        # functions and classes whose ``__module__`` is "__main__".
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


_execute_user_code(source, effective_limit)
"""


class PythonTool(BaseTool):
    """Execute Python in a separate subprocess with a timeout, a memory cap and an
    output cap, and return the model-visible output.

    The limits are guardrails, not a security sandbox: generated code runs as the
    current user with filesystem and network access.
    """

    def __init__(
        self,
        python_path: str | None = None,
        timeout: int | float = 120,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        max_output_bytes: int | None = None,
        memory_limit_gb: float | None = DEFAULT_MEMORY_LIMIT_GB,
    ):
        self._python_path = python_path or sys.executable
        self._timeout = self._positive_number("timeout", timeout)
        self._max_output_chars = self._positive_int(
            "max_output_chars", max_output_chars
        )
        self._max_output_bytes = self._optional_positive_int(
            "max_output_bytes", max_output_bytes
        )
        self._memory_limit_gb = self._optional_positive_number(
            "memory_limit_gb", memory_limit_gb
        )

    @property
    def name(self) -> str:
        return "python_interpreter"

    @property
    def trigger_tag(self) -> str:
        return "python"

    def execute(self, code: str, **kwargs) -> str:
        timeout = self._positive_number(
            "timeout", kwargs.get("timeout", self._timeout)
        )
        max_output_chars = self._positive_int(
            "max_output_chars",
            kwargs.get("max_output_chars", self._max_output_chars),
        )
        configured_bytes = kwargs.get("max_output_bytes", self._max_output_bytes)
        max_output_bytes = self._optional_positive_int(
            "max_output_bytes", configured_bytes
        )
        if max_output_bytes is None:
            # Four bytes cover a complete UTF-8 code point.  Keep a small floor
            # so short diagnostic limits do not make ordinary imports brittle.
            max_output_bytes = max(4_096, max_output_chars * 4)
        memory_limit_gb = self._optional_positive_number(
            "memory_limit_gb",
            kwargs.get("memory_limit_gb", self._memory_limit_gb),
        )

        result, report = self._run_code(
            code,
            timeout=timeout,
            max_output_chars=max_output_chars,
            max_output_bytes=max_output_bytes,
            memory_limit_gb=memory_limit_gb,
        )
        return result if report == "Done" else report

    def _run_code(
        self,
        code: str,
        timeout: float,
        max_output_chars: int,
        max_output_bytes: int,
        memory_limit_gb: float | None,
    ) -> Tuple[str, str]:
        code = self._preprocess_code(code)
        command = [self._python_path, "-c", code]
        if memory_limit_gb is not None:
            memory_limit_bytes = int(memory_limit_gb * 1024**3)
            command = [
                self._python_path,
                "-c",
                _MEMORY_LIMIT_BOOTSTRAP,
                str(memory_limit_bytes),
                code,
            ]

        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                bufsize=0,
                start_new_session=(os.name == "posix"),
            )
        except Exception as exc:
            report = f"Error: could not start Python process: {exc}"
            return "", self._bounded(report, max_output_chars)

        streams = {
            "stdout": {
                "pipe": proc.stdout,
                "decoder": codecs.getincrementaldecoder("utf-8")(
                    errors="replace"
                ),
                "parts": [],
            },
            "stderr": {
                "pipe": proc.stderr,
                "decoder": codecs.getincrementaldecoder("utf-8")(
                    errors="replace"
                ),
                "parts": [],
            },
        }
        selector = None
        try:
            selector = selectors.DefaultSelector()
            for name, state in streams.items():
                pipe = state["pipe"]
                if pipe is not None:
                    selector.register(pipe, selectors.EVENT_READ, name)
        except BaseException:
            self._kill_process_group(proc, reap=True)
            if selector is not None:
                selector.close()
            for state in streams.values():
                pipe = state["pipe"]
                if pipe is not None:
                    pipe.close()
            raise

        assert selector is not None

        deadline = time.monotonic() + timeout
        raw_count = 0
        char_count = 0
        limit_reason: str | None = None
        timed_out = False
        leader_exited_at: float | None = None

        def append_text(name: str, text: str) -> bool:
            """Append within the shared char budget; return True on overflow."""
            nonlocal char_count
            if not text:
                return False
            remaining = max_output_chars - char_count
            if remaining > 0:
                streams[name]["parts"].append(text[:remaining])
                char_count += min(len(text), remaining)
            return len(text) > remaining

        try:
            while selector.get_map():
                now = time.monotonic()
                if now >= deadline:
                    timed_out = True
                    break

                returncode = proc.poll()
                if returncode is not None and leader_exited_at is None:
                    # The main interpreter is done.  Kill any background child
                    # still holding a copied pipe, then briefly drain buffers.
                    self._kill_process_group(proc, reap=False)
                    leader_exited_at = now

                select_timeout = min(0.05, deadline - now)
                if leader_exited_at is not None:
                    select_timeout = min(
                        select_timeout,
                        max(0.0, leader_exited_at + 0.25 - now),
                    )
                events = selector.select(select_timeout)

                if not events:
                    if (
                        leader_exited_at is not None
                        and time.monotonic() >= leader_exited_at + 0.25
                    ):
                        break
                    continue

                for key, _ in events:
                    name = key.data
                    state = streams[name]
                    try:
                        chunk = os.read(key.fd, 65_536)
                    except BlockingIOError:
                        continue

                    if not chunk:
                        tail = state["decoder"].decode(b"", final=True)
                        if append_text(name, tail):
                            limit_reason = (
                                f"decoded output exceeded {max_output_chars} "
                                "characters"
                            )
                        try:
                            selector.unregister(key.fileobj)
                        except Exception:
                            pass
                        try:
                            key.fileobj.close()
                        except Exception:
                            pass
                        if limit_reason is not None:
                            break
                        continue

                    remaining_bytes = max_output_bytes - raw_count
                    kept = chunk[: max(0, remaining_bytes)]
                    raw_count += len(kept)
                    if kept:
                        decoded = state["decoder"].decode(kept, final=False)
                        if append_text(name, decoded):
                            limit_reason = (
                                f"decoded output exceeded {max_output_chars} "
                                "characters"
                            )
                    if len(chunk) > len(kept):
                        limit_reason = (
                            f"raw stdout+stderr exceeded {max_output_bytes} bytes"
                        )
                    if limit_reason is not None:
                        break

                if limit_reason is not None:
                    break

            # Pipes can be deliberately closed while generated code keeps
            # running.  The process still shares the original total deadline.
            if limit_reason is None and not timed_out and proc.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                else:
                    try:
                        proc.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        timed_out = True

            if limit_reason is not None or timed_out:
                self._kill_process_group(proc, reap=True)
            else:
                # Reap the leader and clean up any detached-in-place children
                # that remained in its process group with redirected pipes.
                if proc.poll() is None:
                    proc.wait(timeout=max(0.01, deadline - time.monotonic()))
                self._kill_process_group(proc, reap=False)
        except Exception as exc:
            self._kill_process_group(proc, reap=True)
            stdout = "".join(streams["stdout"]["parts"]).strip()
            stderr = "".join(streams["stderr"]["parts"]).strip()
            report = self._diagnostic(
                f"Error while executing Python: {exc}",
                stdout,
                stderr,
                max_output_chars,
            )
            return "", report
        except BaseException:
            # start_new_session isolates generated code from the caller's
            # terminal signals; on interruption, clean the process tree first.
            self._kill_process_group(proc, reap=True)
            raise
        finally:
            try:
                selector.close()
            except Exception:
                pass
            for state in streams.values():
                pipe = state["pipe"]
                if pipe is not None:
                    try:
                        pipe.close()
                    except Exception:
                        pass

        stdout = "".join(streams["stdout"]["parts"]).strip()
        stderr = "".join(streams["stderr"]["parts"]).strip()

        if limit_reason is not None:
            report = self._output_limit_diagnostic(
                limit_reason,
                stdout,
                stderr,
                max_output_chars,
            )
            return "", report
        if timed_out:
            report = self._diagnostic(
                f"Execution Timeout: exceeded {timeout:g} seconds",
                stdout,
                stderr,
                max_output_chars,
            )
            return "", report

        returncode = proc.returncode
        if returncode == 0:
            return self._bounded(stdout, max_output_chars), "Done"

        if returncode is not None and returncode < 0:
            header = f"Process terminated by signal {-returncode}"
            report = self._diagnostic(
                header, stdout, stderr, max_output_chars
            )
            return "", report

        if stderr.startswith("MemoryLimitExceeded:"):
            report = self._diagnostic(
                stderr, stdout, "", max_output_chars
            )
            return "", report

        if stderr.startswith("MemoryLimitConfigurationError:"):
            report = self._diagnostic(
                stderr, stdout, "", max_output_chars
            )
            return "", report

        header = f"Python exited with code {returncode}"
        report = self._diagnostic(
            header, stdout, stderr, max_output_chars
        )
        return "", report

    @staticmethod
    def _output_limit_diagnostic(
        reason: str,
        stdout: str,
        stderr: str,
        max_chars: int,
    ) -> str:
        # Put captured output immediately after the stable error name.  This
        # preserves useful partial output even under a deliberately tiny
        # per-call character limit; the detailed reason follows when it fits.
        sections = ["OutputLimitExceeded:"]
        if stdout:
            sections.append(stdout)
        if stderr:
            sections.append(stderr)
        sections.append(reason)
        return PythonTool._bounded("\n".join(sections), max_chars)

    @staticmethod
    def _kill_process_group(
        proc: subprocess.Popen, *, reap: bool
    ) -> None:
        """Kill the interpreter's process group and optionally reap its leader."""
        killed_group = False
        if os.name == "posix" and hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                killed_group = True
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
        if not killed_group and proc.poll() is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        if reap and proc.poll() is None:
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass

    @staticmethod
    def _diagnostic(
        header: str,
        stdout: str,
        stderr: str,
        max_chars: int,
    ) -> str:
        sections = [header]
        if stdout:
            sections.append(f"Partial stdout:\n{stdout}")
        if stderr:
            sections.append(f"Partial stderr:\n{stderr}")
        return PythonTool._bounded("\n\n".join(sections), max_chars)

    @staticmethod
    def _bounded(text: str, max_chars: int) -> str:
        return text[:max_chars]

    @staticmethod
    def _positive_int(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        return value

    @staticmethod
    def _optional_positive_int(name: str, value: object) -> int | None:
        if value is None:
            return None
        return PythonTool._positive_int(name, value)

    @staticmethod
    def _positive_number(name: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be positive, got {value!r}")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
        return result

    @staticmethod
    def _optional_positive_number(name: str, value: object) -> float | None:
        if value is None:
            return None
        return PythonTool._positive_number(name, value)

    @staticmethod
    def _preprocess_code(code: str) -> str:
        """Wrap the last bare expression in print() so its value is captured."""
        try:
            tree = ast.parse(code)
            if tree.body:
                last = tree.body[-1]
                if isinstance(last, ast.Expr):
                    is_print = (
                        isinstance(last.value, ast.Call)
                        and isinstance(last.value.func, ast.Name)
                        and last.value.func.id == "print"
                    )
                    if not is_print:
                        tree.body[-1] = ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id="print", ctx=ast.Load()),
                                args=[last.value],
                                keywords=[],
                            )
                        )
                        code = ast.unparse(tree)
        except Exception:
            pass
        return code
