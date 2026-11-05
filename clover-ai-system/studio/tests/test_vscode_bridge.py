"""Offline checks of the Studio <-> VS Code connection (no VS Code, no engine): token, job routing to the window in
front, results, failures, timeouts, and which messages offer Clover the VS Code tools.
Usage: python tests/test_vscode_bridge.py"""
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import lfm3_native as N  # noqa: E402
import vscode_bridge  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


def raises(fn, text):
    try:
        fn()
    except RuntimeError as exc:
        return text in str(exc)
    return False


ok = True
tmp = Path(tempfile.mkdtemp())
b = vscode_bridge.Bridge(tmp / "config" / "vscode-bridge.json")
ok &= check("token is 64 hex", len(b.token) == 64 and all(c in "0123456789abcdef" for c in b.token))
ok &= check("token survives a restart", vscode_bridge.Bridge(tmp / "config" / "vscode-bridge.json").token == b.token)
ok &= check("token check", b.check(b.token) and not b.check("x" * 64) and not b.check(None))
ok &= check("no window: call refuses", raises(lambda: b.call("status"), "not open"))


def window(name, focus, answer=None, polls=1, wait=3.0, got=None):
    """A fake VS Code window: polls, records the jobs it got and answers them."""
    for _ in range(polls):
        job = b.poll(name, focus, "title " + name, "1.138.0", wait)
        if job is not None:
            if got is not None:
                got.append(job)
            if answer is not None:
                b.result(job["id"], *answer(job))


got_a, got_b = [], []
b.poll("A", 100, "", "", 0)
b.poll("B", 200, "", "", 0)
ok &= check("window in front is preferred", b.connected()["id"] == "B")
b.poll("B", 50, "", "", 0)  # a long poll that started before the last focus carries an older time
ok &= check("newest focus time is kept", b.connected()["id"] == "B" and b.windows["B"]["focus"] == 200)
ta = threading.Thread(target=window, args=("A", 100, lambda j: (True, {"from": "A"}), 1, 2.0, got_a))
tb = threading.Thread(target=window, args=("B", 200, lambda j: (True, {"from": "B", "args": j["args"]}), 1, 3.0, got_b))
ta.start(), tb.start()
res = b.call("open", {"target": "C:\\x.md"}, timeout=5)
ta.join(), tb.join()
ok &= check("job ran in the window in front", res == {"from": "B", "args": {"target": "C:\\x.md"}} and not got_a and len(got_b) == 1)

t = threading.Thread(target=window, args=("B", 300, lambda j: (False, None, "Extension 'x.y' not found."), 1, 3.0))
t.start()
ok &= check("a failed job raises its error", raises(lambda: b.call("install", {"extension": "x.y"}, timeout=5), "not found"))
t.join()

b.poll("B", 300, "", "", 0)
t0 = time.time()
ok &= check("no answer: call times out", raises(lambda: b.call("run", {"command": "x"}, timeout=1.5), "did not finish"))
ok &= check("timed-out job leaves the queue", not b.queue and time.time() - t0 < 4)

b.windows = {k: dict(v, seen=v["seen"] - vscode_bridge.LIVE_S - 1) for k, v in b.windows.items()}
ok &= check("closed windows stop counting", b.connected() is None)

vs = {"vscode_open", "vscode_clone", "vscode_install", "vscode_run"}
for msg, want in [("open my project folder C:\\work\\app in VS Code", True),
                  ("install the Rust extension so I get syntax checks", True),
                  ("Can you install the Python extension?", True),
                  ("clone santhoshsathish94/clover-framework", True),
                  ("open the website https://example.com", True),
                  ("run pip install requests in the terminal", True),
                  ("how do I install numpy on Windows?", False),
                  ("what is a browser extension?", False),
                  ("what is 2 + 2?", False),
                  ("open a new chat and summarise this paragraph", False)]:
    ok &= check(f"tools offered={want}: {msg}", vs.issubset(set(N.asked_tools(msg))) == want)
named = N.asked_tools("Open the file C:\\Users\\me\\notes.md in VS Code.")
ok &= check("VS Code named: its tools, not the hidden file tools", vs.issubset(set(named))
            and not {"read_file", "list_directory", "run_powershell"} & set(named))
ok &= check("file request without VS Code keeps read_file", "read_file" in N.asked_tools("read the file C:\\x.txt"))
gh = {"vscode_github_setup", "vscode_github_sync"}
for msg, want in [("Sync my records to GitHub", True), ("set up my private GitHub repository", True),
                  ("Can you push the records to my repo?", True), ("how do I push to github in general?", False),
                  ("summarise this paragraph about backup strategies", False)]:
    ok &= check(f"GitHub tools offered={want}: {msg}", gh.issubset(set(N.asked_tools(msg))) == want)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
