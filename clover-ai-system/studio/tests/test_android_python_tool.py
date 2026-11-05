"""Checks android_python_tool (the phone app's in-process Python runner) on any OS: output, last expression, errors,
exit codes, time limit, output cap, working folder put back, other threads' output untouched."""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clover"))
from android_python_tool import PythonTool  # noqa: E402

ok = 0
fails = []


def check(name, cond, got=""):
    global ok
    if cond:
        ok += 1
    else:
        fails.append(f"{name}: {got!r}")


t = PythonTool(timeout=3, max_output_chars=2000)
check("print", t.execute("print('hello', 6*7)") == "hello 42", t.execute("print('hello', 6*7)"))
check("last expression", t.execute("x = 5\nx * 3") == "15", t.execute("x = 5\nx * 3"))
r = t.execute("raise ValueError('boom')")
check("exception", r.startswith("Python exited with code 1") and "ValueError: boom" in r, r)
r = t.execute("import sys; print('before'); sys.exit(3)")
check("exit code", r.startswith("Python exited with code 3") and "before" in r, r)
check("exit 0", t.execute("import sys; print('fine'); sys.exit(0)") == "fine")
r = t.execute("print('loop'); \nwhile True: pass")
check("timeout", r.startswith("Execution Timeout: exceeded 3 seconds") and "loop" in r, r)
small = PythonTool(timeout=5, max_output_chars=50)
r = small.execute("print('x' * 500)")
check("output cap", r.startswith("OutputLimitExceeded:"), r)
before = os.getcwd()
d = tempfile.mkdtemp()
t.execute(f"import os; os.chdir(r'{d}'); print(os.getcwd())")
check("cwd restored", os.getcwd() == before, os.getcwd())
check("fresh globals", t.execute("print('x' in globals())") == "False", t.execute("print('x' in globals())"))
# a server thread printing while a run is going on keeps its output out of the run
seen = []


def other():
    for _ in range(20):
        print("SERVER-LINE")
        time.sleep(0.01)


th = threading.Thread(target=other)
th.start()
r = t.execute("import time\nfor i in range(5):\n    print('tool', i)\n    time.sleep(0.03)")
th.join()
check("routing", "SERVER-LINE" not in r and r.count("tool") == 5, r)
r = t.execute("import sys; print('to stderr', file=sys.stderr); print('out')")
check("stderr kept apart", r == "out", r)
print(f"android_python_tool: {ok} passed, {len(fails)} failed")
for f in fails:
    print("FAIL", f)
sys.exit(1 if fails else 0)
