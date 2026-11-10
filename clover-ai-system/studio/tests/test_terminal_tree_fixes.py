"""The Studio Terminal app prints what the command prints, and an opened folder in the Workspaces / Memory tree shows
its files instead of "Loading…" (found in the v5.0.4 UI walk, 2026-11-10).
Terminal: real PowerShell, no model. Windows PowerShell 5.1 printed nothing for Get-Location, Select-Object or
`cd ..; Get-Location` when the script ended with `exit`; exit codes, the folder and native stderr must stay as before.
Tree: the opened folder's "Loading…" line is replaced by its entries, and a failure is shown inside the folder.
Usage: python tests/test_terminal_tree_fixes.py"""
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clover"))
import server  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    return bool(cond)


ok = True
if os.name != "nt":
    print("SKIP Terminal checks: the PowerShell wrapper runs on Windows only")
else:
    d = Path(tempfile.mkdtemp()).resolve()  # the temp path can be an 8.3 short name; PowerShell prints the long form
    (d / "marker.txt").write_text("x")
    server.S.workspaces, server.S.repo = None, SimpleNamespace(dir=str(d))

    def term(command, **kw):
        return server.run_terminal_tool(dict({"command": command}, **kw))

    r = term("Get-Location")
    ok &= check("Get-Location prints the folder", str(d) in r["output"] and r["exit_code"] == 0)
    r = term("Get-Item marker.txt | Select-Object Name, Length")
    ok &= check("Select-Object prints its table", "marker.txt" in r["output"] and "Length" in r["output"])
    r = term("cd ..; Get-Location")
    ok &= check("`cd ..; Get-Location` prints the parent and the Terminal moves there",
                str(d.parent) in r["output"] and Path(r["cwd"]) == d.parent)
    r = term("Get-ChildItem -Name")
    ok &= check("plain output still prints; the folder marker is not shown", r["output"].strip() == "marker.txt"
                and "__CLOVER_CWD__" not in r["output"] and Path(r["cwd"]) == d)
    r = term("cmd /c exit 3")
    ok &= check("a native command's exit code is kept", r["exit_code"] == 3)
    r = term("Get-Item no-such-file-clover.txt")
    ok &= check("a failing cmdlet: exit code 1 and its error", r["exit_code"] == 1 and "no-such-file-clover.txt" in r["output"])
    r = term("Get-Item no-such-file-clover.txt; Get-Date -Format yyyy")
    ok &= check("the exit code still follows the last statement", r["exit_code"] == 0
                and re.search(r"(?m)^\d{4}$", r["output"]) is not None)
    r = term('cmd /c "echo to-stderr 1>&2"')
    ok &= check("native stderr stays plain text (no PowerShell error record around it)",
                r["output"].strip() == "to-stderr" and r["exit_code"] == 0)
    r = term("git --version && echo chained")
    ok &= check("`a && b` still runs both", "chained" in r["output"])
    r = term("Get-Location", cwd=str(d.parent))
    ok &= check("a chosen folder is used", str(d.parent) in r["output"] and Path(r["cwd"]) == d.parent)

app = (HERE.parent / "clover" / "web" / "app.html").read_text(encoding="utf-8")
for fn, nxt in (("loadWorkspaceTree", "loadMemoryTree"), ("loadMemoryTree", "openWorkspaceFile")):
    m = re.search(r"function " + fn + r"\(.*?(?=\n  function " + nxt + r"\()", app, re.S)
    body = m.group(0) if m else ""
    then = re.search(r"\.then\(function \(rep\) \{(.*?)var entries = rep\.entries", body, re.S)
    ok &= check(f"{fn}: the box is emptied for an opened folder too, not only for the root",
                bool(then) and re.search(r"\}\s*\n\s*treeBox\.innerHTML = '';\s*$", then.group(1)) is not None)
    catch = re.search(r"\.catch\(function \(e\) \{(.*?)\n    \}\);", body, re.S)
    ok &= check(f"{fn}: a failure is shown inside an opened folder too",
                bool(catch) and "if (!containerEl)" not in catch.group(1) and "treeBox.innerHTML" in catch.group(1))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
