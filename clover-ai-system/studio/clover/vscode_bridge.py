"""The Studio <-> VS Code connection. Clover asks VS Code to open files and pages, clone repositories, install
extensions, run commands in a visible terminal and sign in to GitHub; the Clover extension in each VS Code window polls
for those jobs and VS Code does the work with its own UI, sign-in and confirmations.

Only a client holding the token in config/vscode-bridge.json (a file on this laptop that the extension reads) can poll
or report results, so a web page cannot take or answer jobs. Jobs go to the VS Code window focused most recently."""
import json
import secrets
import subprocess
import threading
import time
import uuid
from pathlib import Path

LIVE_S = 45  # a window that has not polled for this long is gone


class Bridge:
    def __init__(self, token_path):
        self.path = Path(token_path)
        self.cv = threading.Condition()
        self.queue, self.results, self.windows, self.waiting = [], {}, {}, {}
        self.started_at = 0.0  # when ensure() last started VS Code itself
        self.token = self._token()

    def _token(self):
        try:
            t = json.loads(self.path.read_text(encoding="utf-8")).get("token")
            if isinstance(t, str) and len(t) >= 32:
                return t
        except (OSError, ValueError):
            pass
        t = secrets.token_hex(32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"token": t}), encoding="utf-8")
        tmp.replace(self.path)
        return t

    def check(self, header):
        return isinstance(header, str) and secrets.compare_digest(header, self.token)

    def _preferred(self, now):
        live = {k: v for k, v in self.windows.items() if now - v["seen"] < LIVE_S}
        return max(live, key=lambda k: (live[k]["focus"], live[k]["seen"])) if live else None

    def poll(self, window_id, focus=0.0, title="", version="", wait=25.0):
        """Long poll from one VS Code window: its next job, or None after `wait` seconds."""
        deadline = time.time() + min(max(float(wait), 0.0), 30.0)
        with self.cv:
            # a long poll carries the focus time from when it started; keep the newest one this window reported
            focus = max(float(focus or 0), self.windows.get(window_id, {}).get("focus", 0.0))
            while True:
                now = time.time()
                self.windows[window_id] = {"seen": now, "focus": focus, "title": str(title)[:200],
                                           "version": str(version)[:20]}
                if self.queue and self._preferred(now) == window_id:
                    return self.queue.pop(0)
                if now >= deadline:
                    return None
                self.cv.wait(min(deadline - now, 1.0))  # wakes every second so the window stays "seen"

    def result(self, job_id, ok, result=None, error=None):
        with self.cv:
            self.results[str(job_id)] = {"ok": bool(ok), "result": result, "error": error, "at": time.time()}
            for k in [k for k, v in self.results.items() if time.time() - v["at"] > 900]:
                del self.results[k]
            self.cv.notify_all()

    def progress(self, job_id, note):
        """VS Code is showing the user a confirmation for this job ("Open Browser Page?") and waits for the answer."""
        with self.cv:
            self.waiting[str(job_id)] = str(note or "VS Code is waiting for your confirmation")
            self.cv.notify_all()

    def connected(self):
        with self.cv:
            p = self._preferred(time.time())
            return dict(self.windows[p], id=p) if p else None

    def call(self, action, args=None, timeout=120.0, approval_timeout=None, on_wait=None):
        """Runs `action` in VS Code and returns its result; RuntimeError when no window is connected or it failed.
        A job VS Code reports as waiting for the user's confirmation gets `approval_timeout` seconds from then, and
        on_wait(note) is called once (outside the lock) so the user hears about it."""
        with self.cv:
            if self._preferred(time.time()) is None:
                raise RuntimeError("VS Code is not open on this laptop (the Clover extension connects when it starts)")
            job = {"id": uuid.uuid4().hex, "action": action, "args": args or {}}
            self.queue.append(job)
            self.cv.notify_all()
        deadline, told = time.time() + timeout, None
        try:
            while True:
                note = None
                with self.cv:
                    if job["id"] in self.results:
                        res = self.results.pop(job["id"])
                        break
                    if told is None and job["id"] in self.waiting:
                        told = note = self.waiting[job["id"]]
                        if approval_timeout:
                            deadline = max(deadline, time.time() + approval_timeout)
                    left = deadline - time.time()
                    if left <= 0:
                        self.queue = [j for j in self.queue if j["id"] != job["id"]]
                        if told:  # VS Code's dialog stayed unanswered: the user cannot see it in Clover (2026-11-10)
                            raise RuntimeError(f"VS Code is still waiting for your approval ({told}). Approve or cancel "
                                               "it in VS Code, then ask again.")
                        raise RuntimeError(f"VS Code did not finish within {int(timeout)} s")
                    if note is None:
                        self.cv.wait(min(left, 1.0))
                if note is not None and on_wait is not None:
                    on_wait(note)
        finally:
            with self.cv:
                self.waiting.pop(job["id"], None)
        if not res["ok"]:
            raise RuntimeError(res["error"] or "VS Code could not do it")
        return res["result"]

    def ensure(self, code_exe, timeout=30.0):
        """Starts VS Code when no window is connected and waits for the extension to connect."""
        if self.connected():
            return True
        if not code_exe or not Path(code_exe).is_file():
            return False
        self.started_at = time.time()
        subprocess.Popen([str(code_exe), "--new-window"], close_fds=True,
                         creationflags=getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        end = time.time() + timeout
        while time.time() < end:
            if self.connected():
                return True
            time.sleep(0.5)
        return False
