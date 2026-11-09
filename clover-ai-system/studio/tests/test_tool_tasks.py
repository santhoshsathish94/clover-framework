"""Review item 4: every tool call is a SuperAgent task. A write counts as done only when the file reads back as written
(the real LFM3Native.execute_tool, a stub engine without a model); a call left unfinished by a stopped process is
reported once on the next start; arguments are recorded with secrets redacted.
Usage: python tests/test_tool_tasks.py"""
import json
import sys
import tempfile
import threading
from pathlib import Path

CLOVER = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(CLOVER))
import stand_ins  # noqa: E402

stand_ins.install(CLOVER)
import lfm3_native  # noqa: E402
import permissions  # noqa: E402
import tool_tasks  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


def engine(base):
    eng = object.__new__(lfm3_native.LFM3Native)  # no model needed for execute_tool
    eng.base_dir, eng.workdir, eng.workname, eng.after_write = base, base / "work", "", None
    eng.perms = permissions.PermissionStore(base / "permissions.json")
    eng.perms.update({t: "allow" for t in permissions.TOOLS})
    eng._extra, eng.turn_paths, eng.stage, eng.lock = {}, [], {"name": "idle", "since": 0}, threading.Lock()
    eng.tool_tasks = tool_tasks.ToolTasks(base, "laptop")
    return eng


ok = True
base = Path(tempfile.mkdtemp(prefix="tool-tasks-"))
eng = engine(base)
ran, out, why = eng.execute_tool("write_file", {"path": "notes.txt", "content": "line one\nline two\n"})
rec = eng.tool_tasks.recent()[-1]
ok &= check("a write that reads back as written completes, checked by reading the file", ran and out.startswith("Wrote")
            and rec["status"] == "completed" and rec["evidence"]["checked"] == "file read back")
summary = lfm3_native.tool_worked("write_file", ran, out, why)[1]
ok &= check("its step in the chat says it was checked on disk",
            lfm3_native.checked_summary(eng, ran, summary).endswith("checked on disk"))


def wrong_write(name, args):  # liveness: the tool says it wrote, the file holds something else
    p = eng._path(args["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("something else", encoding="utf-8")
    return f"Wrote {len(args['content'])} characters to {p}."


real_run = eng._run_tool
eng._run_tool = wrong_write
ran, out, why = eng.execute_tool("write_file", {"path": "b.txt", "content": "the real text"})
rec = eng.tool_tasks.recent()[-1]
ok &= check("a write that does not read back fails, and the model is told it was not done",
            out.startswith("Error: Clover read the file back") and rec["status"] == "failed"
            and not lfm3_native.tool_worked("write_file", ran, out, why)[0])
ok &= check("its step does not say checked", not lfm3_native.checked_summary(eng, ran, "x").endswith("checked on disk"))
eng._run_tool = real_run
ran, out, why = eng.execute_tool("write_file", {"path": "..", "content": "x"})
rec = eng.tool_tasks.recent()[-1]
ok &= check("a write the tool itself reports as failed is recorded as failed", rec["status"] == "failed")
ran, out, why = eng.execute_tool("read_file", {"path": str(base / "work" / "notes.txt")})
rec = eng.tool_tasks.recent()[-1]
ok &= check("a read completes on its own result, and the record says so", rec["status"] == "completed"
            and rec["evidence"]["checked"] == "the tool's own result")

secret = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
eng.execute_tool("write_file", {"path": "key.txt", "content": f"api key {secret}"})
log_text = eng.tool_tasks.path.read_text(encoding="utf-8")
ok &= check("arguments are recorded with secrets redacted", secret not in log_text)

# a call that never finished: the process stopped while it ran
tt = tool_tasks.ToolTasks(base, "laptop")
tid = tt.begin("run_powershell", {"command": "Start-Sleep 600"})
tt2 = tool_tasks.ToolTasks(base, "laptop")  # the next start
ok &= check("the next start finds the unfinished call", [i["tool"] for i in tt2.interrupted] == ["run_powershell"])
status = tt2.sa.tasks[tid].status
ok &= check("it is recorded as failed, interrupted, and not run again", status == "failed"
            and tt2.sa.tasks[tid].failure_reason == tool_tasks.INTERRUPTED)
note = tool_tasks.interrupted_note(tt2.take_interrupted())
ok &= check("the next answer says what did not finish, once", "run_powershell" in note and not tt2.take_interrupted())
tt3 = tool_tasks.ToolTasks(base, "laptop")
ok &= check("a start after that finds nothing unfinished", tt3.interrupted == [])
lines = [json.loads(x) for x in tt3.path.read_text(encoding="utf-8").splitlines() if x.strip()]
ok &= check("the log is the SuperAgent's event log (one JSON event per line)", lines and all("seq" in x and "type" in x for x in lines))

# The note is for the person. On the laptop the model repeated it in its next answer, so the model never reads it back.
ok &= check("the note reads as before", note == "(Note: when Clover last stopped, it was in the middle of run_powershell. "
            "That step did not finish, so check its result before relying on it.)")
reply = "2 + 2 = 4\n\n" + note
ok &= check("the model's copy of an answer leaves the note out", tool_tasks.strip_note(reply) == "2 + 2 = 4")
ok &= check("an answer without the note is unchanged", tool_tasks.strip_note("plain (Note: other)") == "plain (Note: other)")
import server as SV  # noqa: E402


class Repo:
    def messages(self, sid):
        return [{"n": 1, "role": "user", "text": "What is 2 + 2?"}, {"n": 1, "role": "clover", "text": reply}]


repo_before = SV.S.repo
try:
    SV.S.repo = None
    hist = SV.lfm3_history("tt-seed", [{"user": "What is 2 + 2?", "reply": reply}])
    ok &= check("history from the browser's copy leaves the note out", hist[-1]["content"] == "2 + 2 = 4")
    SV.S.repo = Repo()
    hist = SV.lfm3_history("tt-repo", None)
    ok &= check("history from the saved conversation leaves the note out", hist[-1]["content"] == "2 + 2 = 4")
finally:
    SV.S.repo = repo_before
    SV.S.sessions.pop("tt-seed", None)
    SV.S.sessions.pop("tt-repo", None)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
