"""Checks of the terminal tool on Windows (real PowerShell, no model): UTF-8 output, exit codes, the working folder, the
time limit stopping the whole process tree, stopping when the user stops the turn, the output cap, a missing folder,
and the tool call through the engine (permission detail shows the folder and the limit).
Usage: python tests/test_local_terminal.py"""
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import local_terminal as T  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    return bool(cond)


ok = True
out = T.powershell("Write-Output 'café Zürich 日本'")
ok &= check("UTF-8 output (é, ü, CJK) and exit code 0", "café Zürich 日本" in out and out.endswith("[exit code 0]"))
out = T.powershell("cmd /c exit 3")
ok &= check("a native command's exit code is reported", out.endswith("[exit code 3]"))
out = T.powershell("Get-Item C:\\does-not-exist-clover")
ok &= check("a failing PowerShell command ends with a non-zero exit code and its error", "[exit code 1]" in out
            and "does-not-exist-clover" in out)
d = Path(tempfile.mkdtemp()).resolve()  # the temp path can be an 8.3 short name; PowerShell prints the long form
(d / "marker.txt").write_text("x")
out = T.powershell("(Get-Location).Path; Get-ChildItem -Name", cwd=str(d))
ok &= check("runs in the folder it was given", str(d) in out and "marker.txt" in out)

t0 = time.time()
out = T.powershell("Start-Process -NoNewWindow -FilePath ping -ArgumentList '-n','30','127.0.0.1'; Start-Sleep 30",
                   timeout_s=2)
took = time.time() - t0
time.sleep(1)
left = subprocess.run(["tasklist", "/FI", "IMAGENAME eq PING.EXE", "/NH"], capture_output=True, text=True).stdout
ok &= check(f"the time limit stops it ({took:.1f} s) and says so", took < 10 and "[stopped after 2 seconds" in out)
ok &= check("the child process (ping) is stopped too", "PING.EXE" not in left.upper())

cancel = threading.Event()
threading.Timer(1.5, cancel.set).start()
t0 = time.time()
out = T.powershell("Start-Sleep 30", timeout_s=60, cancel=cancel)
ok &= check(f"stopping the turn stops the command ({time.time() - t0:.1f} s)", time.time() - t0 < 10
            and "[stopped: the user stopped the turn]" in out)

out = T.powershell("1..400000 | ForEach-Object { 'line ' + $_ }", timeout_s=120)
ok &= check("endless output is capped and the cut is reported", "[output cut:" in out and len(out) < 1_100_000)
ok &= check("time limits are clamped to 1..600 s", T.clamp_timeout(5000) == 600 and T.clamp_timeout(0) == 1
            and T.clamp_timeout("x") == 60)
p, err = T.resolve_cwd("C:\\no\\such\\folder\\clover")
ok &= check("a missing folder is an error, nothing runs", p is None and "does not exist" in err)
p, err = T.resolve_cwd("", workdir=str(d))
ok &= check("default folder: the open workspace", p == d and err is None)
p, err = T.resolve_cwd("", workdir=None)
ok &= check("default folder without a workspace: the home folder", p == Path.home())

import lfm3_native as N  # noqa: E402

eng = object.__new__(N.LFM3Native)
eng.workdir, eng._extra, eng.base_dir, eng.stage, eng.turn_cancel = str(d), {}, d, {}, None
asked = []
eng.perms = type("P", (), {"get": lambda self: {"shell": "ask"}})()
ran, res, why = eng.execute_tool("run_powershell", {"command": "Get-ChildItem -Name", "timeout_s": 120},
                                 ask=lambda perm, detail: (asked.append((perm, detail)) or True))
ok &= check("engine: Ask shows the command, the folder and the limit", asked and asked[0][0] == "shell"
            and "Get-ChildItem" in asked[0][1] and str(d) in asked[0][1] and "up to 120 s" in asked[0][1])
ok &= check("engine: runs in the workspace and returns the exit code", ran and "marker.txt" in res and "[exit code 0]" in res)
ran, res, why = eng.execute_tool("run_powershell", {"command": "whoami"}, ask=lambda perm, detail: False)
ok &= check("engine: refused -> nothing runs", not ran and "not run" in res.lower())

# no editor, pager or password prompt can hold a hidden command: `git rebase -i` opened Vim and hung (2026-11-01)
g = Path(tempfile.mkdtemp())
T.powershell("git init -q; git config user.email t@example.com; git config user.name t; "
             "Set-Content a.txt 1; git add a.txt; git commit -q -m one; Set-Content a.txt 2; git commit -q -am two",
             cwd=str(g), timeout_s=60)
t0 = time.time()
out = T.powershell("git rebase -i HEAD~1", cwd=str(g), timeout_s=30)
ok &= check(f"git rebase -i finishes without an editor ({time.time() - t0:.1f} s)", time.time() - t0 < 15
            and "stopped after" not in out and "\x1b" not in out)
t0 = time.time()
out = T.powershell("git commit --allow-empty", cwd=str(g), timeout_s=30)
ok &= check(f"git commit without a message ends at once ({time.time() - t0:.1f} s)", time.time() - t0 < 15
            and "stopped after" not in out)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
