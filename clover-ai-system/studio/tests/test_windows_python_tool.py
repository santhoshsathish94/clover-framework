"""Checks windows_python_tool.PythonTool keeps upstream PythonTool's guarantees on Windows."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
from windows_python_tool import PythonTool  # noqa: E402

tool = PythonTool(timeout=4, max_output_chars=2000, memory_limit_gb=0.5)
child_flag = Path(__file__).resolve().parent / "orphan_flag.txt"
child_started = Path(__file__).resolve().parent / "orphan_started.txt"
child_flag.unlink(missing_ok=True)
child_started.unlink(missing_ok=True)
inner = (f"import time; open({str(child_started)!r}, 'w').write('up'); time.sleep(6); "
         f"open({str(child_flag)!r}, 'w').write('alive')")
cases = {
    "print": ("print(6*7)", lambda r: r == "42"),
    "last_expr_wrapped": ("x = 5\nx * 3", lambda r: r == "15"),
    "unicode": ("print('h\u00e9llo \u2022 \u03c0')", lambda r: r == "h\u00e9llo \u2022 \u03c0"),
    "exception": ("raise ValueError('boom')", lambda r: r.startswith("Python exited with code 1") and "ValueError: boom" in r),
    "timeout": ("import time\ntime.sleep(30)", lambda r: r.startswith("Execution Timeout: exceeded 4 seconds")),
    "memory_cap": ("b = bytearray(2 * 1024**3)\nprint(len(b))", lambda r: "MemoryLimitExceeded" in r or "MemoryError" in r),
    "output_cap": ("print('x' * 100000)", lambda r: r.startswith("OutputLimitExceeded:") and len(r) <= 2000),
    "tree_kill_on_timeout": (
        f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {inner!r}])\ntime.sleep(30)",
        lambda r: r.startswith("Execution Timeout"),
    ),
}
rows = {}
for name, (code, ok) in cases.items():
    t0 = time.perf_counter()
    res = tool.execute(code)
    rows[name] = {"ok": bool(ok(res)), "s": round(time.perf_counter() - t0, 2), "result": res[:160]}
time.sleep(7)
rows["tree_kill_on_timeout"]["child_started"] = child_started.exists()
rows["tree_kill_on_timeout"]["orphan_survived"] = child_flag.exists()
rows["tree_kill_on_timeout"]["ok"] = rows["tree_kill_on_timeout"]["ok"] and child_started.exists() and not child_flag.exists()
child_flag.unlink(missing_ok=True)
child_started.unlink(missing_ok=True)
print(json.dumps(rows, indent=2))  # ASCII escapes: the report prints in any console code page
print("ALL_OK" if all(r["ok"] for r in rows.values()) else "SOME_FAILED")
