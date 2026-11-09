"""Tool calls as verified tasks (review item 4, approved 2026-11-09).

Every tool call Clover runs becomes a task in the SuperAgent's event-sourced log (superagent_jev_dory.CloverSuperAgent):
running -> verifying -> completed | failed, each change written as one JSON line before the next step starts.

- A file write completes only after Clover reads the file back and finds the text it was asked to write. The tool's own
  "written" never completes it: the SuperAgent's gate does not accept a worker certifying itself. Other tools complete
  on their own result, and the record says which kind of check it was.
- The log outlives the process. A call still unfinished when Studio stopped (Android closed the app, the laptop
  restarted) is marked failed as interrupted on the next start, and the next answer says what did not finish. A tool is
  never run again by itself.

Logs: <base>/logs/tool-tasks/<device>-<yyyy-mm-dd>.jsonl. Arguments are kept as the execution log keeps them (one line,
300 characters, secrets redacted); of a result only its first 200 characters, redacted."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from superagent_jev_dory import CloverSuperAgent

WORKER = "clover"
UNFINISHED = ("created", "routed", "running", "waiting", "verifying")
INTERRUPTED = "interrupted: Clover stopped while this step ran"


def _redact(text):
    try:
        import clover_repo
        return clover_repo.redact(text)[0]
    except Exception:  # the record keeps less rather than failing the tool call
        return "(not recorded)"


def _one_line(text, limit):
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[:limit - 3] + "..."


def _normalized(text):
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def write_check(path, content):
    """The write's own evidence: the file read back holds the text that was written (line endings aside)."""
    want = _normalized(content or "")

    def check():
        p = Path(path)
        if not p.is_file():
            return False, {"checked": "file read back", "passed": False, "reason": "the file is not there after writing"}
        got = _normalized(p.read_text(encoding="utf-8", errors="replace"))
        if got != want:
            return False, {"checked": "file read back", "passed": False,
                           "reason": f"the file holds {len(got)} characters, not the {len(want)} that were written"}
        return True, {"checked": "file read back", "passed": True, "chars": len(got),
                      "sha256": hashlib.sha256(got.encode("utf-8")).hexdigest()[:16]}
    return check


class ToolTasks:
    def __init__(self, base_dir, device):
        self.dir = Path(base_dir) / "logs" / "tool-tasks"
        self.device = "".join(c if c.isalnum() or c in "-_" else "-" for c in str(device or "device"))
        self.lock = threading.Lock()
        self.interrupted, self.recover_error = [], None
        self._recover()
        self.path = self.dir / f"{self.device}-{time.strftime('%Y-%m-%d')}.jsonl"
        self.sa = CloverSuperAgent(storage_path=self.path)
        self.sa.register_worker(WORKER)

    def _recover(self):
        files = sorted(self.dir.glob(f"{self.device}-*.jsonl"))
        if not files:
            return
        try:
            prev = CloverSuperAgent(storage_path=files[-1])
            for t in list(prev.tasks.values()):
                if t.status in UNFINISHED:
                    prev.transition_task(t.id, t.revision, "failed", failure_reason=INTERRUPTED)
                    self.interrupted.append({"task": t.id, "tool": t.subject, "what": t.description,
                                             "log": files[-1].name, "updated_ms": t.updated_at_ms})
        except Exception as exc:  # an unreadable log is kept as it is; new calls go on in today's file
            self.recover_error = f"{type(exc).__name__}: {exc}"

    def begin(self, tool, args, write_scope=None):
        desc = _one_line(_redact(json.dumps(args, ensure_ascii=False, default=str)), 300)
        with self.lock:
            t = self.sa.create_task(subject=str(tool), description=desc, owner_id=WORKER,
                                    write_scopes=[str(write_scope)] if write_scope else [])
            return self.sa.transition_task(t.id, t.revision, "running").id

    def finish(self, task_id, worked, result, check=None):
        """worked: the tool's own result read as success; check: () -> (ok, evidence) that looks at the change itself.
        -> {"task", "status", "evidence"}"""
        with self.lock:
            cur = self.sa.tasks[task_id]
            out = {"worked": bool(worked), "result": _one_line(_redact(result), 200)}
            v = self.sa.transition_task(task_id, cur.revision, "verifying", candidate_output=out)

            def verifier(_out):
                if check is not None and worked:
                    try:
                        return check()
                    except Exception as exc:
                        return False, {"checked": "file read back", "passed": False, "reason": f"could not read it back: {exc}"}
                ok = bool(worked)
                return ok, dict({"checked": "the tool's own result", "passed": ok},
                                **({} if ok else {"reason": "the tool reported that it did not work"}))
            done = self.sa.verify_and_complete_task(task_id, v.revision, verifier_fn=verifier)
            return {"task": task_id, "status": done.status, "evidence": done.verification_evidence}

    def take_interrupted(self):
        """What a previous process left unfinished, once: the next answer reports it."""
        with self.lock:
            out, self.interrupted = self.interrupted, []
            return out

    def recent(self, n=50):
        with self.lock:
            views = [self.sa.get_task_view(tid) for tid in list(self.sa.tasks)[-n:]]
        return [{"task": v.id, "tool": v.subject, "what": v.description, "status": v.status,
                 "evidence": v.verification_evidence} for v in views]


NOTE_HEAD = "(Note: when Clover last stopped, it was in the middle of "
NOTE_TAIL = "before relying on it.)"


def interrupted_note(items):
    if not items:
        return ""
    names = ", ".join(dict.fromkeys(i["tool"] for i in items))
    return f"{NOTE_HEAD}{names}. That step did not finish, so check its result {NOTE_TAIL}"


def strip_note(text):
    """The note is for the person. Left in the history the model reads, the model repeated it in its next answer
    (laptop, 2026-11-09), so the history copy leaves it out."""
    text = str(text or "")
    i = text.find(NOTE_HEAD)
    while i >= 0:
        j = text.find(NOTE_TAIL, i)
        if j < 0:
            break
        text = (text[:i].rstrip() + "\n\n" + text[j + len(NOTE_TAIL):].lstrip()).strip()
        i = text.find(NOTE_HEAD)
    return text
