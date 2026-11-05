"""Clover AI Studio's repository: CloverAI\\clover-ai-studio, a local git repository that the human and Clover both read.
Each kind of record is written once, in its own place:

  AGENTS.md          Clover's operating rules. Protected: restored from the copy shipped with the Studio if changed.
  CONTEXT.md         a summary of everything below; every folder has its own CONTEXT.md. Rewritten from the records.
  sessions/          the conversations: each chat's user inputs and Clover's answers, one JSON line each.
  execution/         plain-text logs of how each turn ran: context used, tools, permissions, timings. No attachments.
  outcomes/          what came of each turn, the user's response to it, and whether it worked (JSON lines).
  growth/knowledge/  what Clover learned and the notes the user asked it to keep, with typed relationships between them
                     (Clover's memory graph), updated, corrected and retired as outcomes come in.
  growth/notes/      a short note the model wrote about each exchange (what the user said about themselves, what Clover
                     gave them), written while the Studio is idle; Layer 2 finds earlier messages by them as well.
  config/<device>/   settings and permissions. Secrets are in CloverAI\\vault, attachments in CloverAI\\media.

Nothing here leaves the laptop unless the user syncs it to their private GitHub repository."""
import datetime
import json
import os
import re
import shutil
import stat
import statistics
import subprocess
import threading
import time
from pathlib import Path

import paths

POSITIVE, NEGATIVE = ("continued", "helpful"), ("said_not_right", "tried_again")
FEEDBACK_WORDS = {"continued": "you continued", "helpful": "you marked it helpful",
                  "said_not_right": "you said it was not right", "tried_again": "you asked to try again"}
SECRETS = (
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{40,255})\b")),
    ("API key", re.compile(r"\b(?:sk-(?:proj-)?[A-Za-z0-9_\-]{20,}|tvly-[A-Za-z0-9_\-]{16,}|AIza[0-9A-Za-z_\-]{35})\b")),
    ("AWS key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
)
GITIGNORE = "growth/knowledge/_index/\n*.tmp\n"
PAST = re.compile(r"\b(?:yesterday|last\s+(?:time|week|session|chat|conversation)|earlier\s+today|"
                  r"we\s+(?:talked|discussed|spoke|decided|worked\s+on)|you\s+(?:told\s+me|said|answered|changed)|"
                  r"our\s+(?:last\s+)?(?:conversation|chat|discussion)|where\s+we\s+left\s+off|pick\s+up\s+where)\b", re.I)
LAST_ONE = re.compile(r"\blast\s+(?:time|session|chat|conversation)\b|\bwhere\s+we\s+left\s+off\b", re.I)
ASKS = (
    ("growth", re.compile(r"\bwhat\s+(?:have\s+)?(?:you|we)\s+learn(?:ed|t)\b|\b(?:your|the)\s+(?:lessons|learnings)\b|"
                          r"\bwhat\s+(?:worked|did\s*n[o']?t\s+work|went\s+wrong)\b", re.I)),
    ("outcomes", re.compile(r"\b(?:your\s+)?outcomes\b|\bhow\s+(?:well\s+)?(?:have|did)\s+you\s+(?:do|done)\b|"
                            r"\byour\s+(?:mistakes|errors)\b", re.I)),
    ("execution", re.compile(r"\b(?:your|execution)\s+logs?\b|\bwhat\s+(?:did|have)\s+you\s+(?:do|done|run)\b|"
                             r"\bwhich\s+tools\s+did\b", re.I)),
    ("config", re.compile(r"\b(?:your|my|the)\s+(?:settings|permissions)\b|\bwhat\s+are\s+you\s+allowed\b", re.I)),
)
STOPWORDS = set("the a an and or of to in on for with is are was were be been it this that these those my your our their "
                "i you we they me us them what which who when where why how about from at by as into than then did do "
                "does done have has had can could would should will just yesterday last time earlier today talked "
                "discussed said told chat conversation session again please".split())


def now():
    return datetime.datetime.now().astimezone()


def iso(t=None):
    return (t or now()).isoformat(timespec="seconds")


def redact(text):
    """-> (text, kinds): secret-looking values replaced before anything reaches the repository."""
    kinds = []
    for kind, rx in SECRETS:
        text, n = rx.subn(f"[redacted {kind}]", text)
        if n:
            kinds.append(kind)
    return text, kinds


def one_line(text, limit=300):
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[:limit - 3] + "..."


def _read_jsonl(fp):
    try:
        lines = fp.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def _words(text):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", str(text or "").lower()) if w not in STOPWORDS}


class Repo:
    def __init__(self, base, agents_src, kit=None):
        self.base, self.dir, self.device = Path(base), paths.repo_dir(base), paths.device()
        self.agents_src, self.kit = Path(agents_src), kit  # kit: () -> lfm3_layers.LayerKit over growth/knowledge
        self.lock = threading.RLock()
        self.git = shutil.which("git")
        self.current = None  # (session, turn) being answered
        self._index, self._last, self._cache = {}, {}, {}
        self._dirty, self._messages = threading.Event(), []
        self.init()
        threading.Thread(target=self._worker, daemon=True, name="repo-commit").start()

    # ---------------- the repository ----------------
    def _git(self, *args):
        if not self.git:
            return 1, "", "git is not installed"
        r = subprocess.run([self.git, *args], cwd=self.dir, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode, r.stdout.strip(), r.stderr.strip()

    def init(self):
        for d in ("sessions", f"execution/{self.device}", f"outcomes/{self.device}", "growth/knowledge",
                  f"growth/notes/{self.device}", f"config/{self.device}"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        if not (self.dir / ".gitignore").is_file():
            (self.dir / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
        if self.git and not (self.dir / ".git").is_dir():
            self._git("init", "-q", "-b", "main")
        if self.git:
            for k, v in (("user.name", "Clover AI Studio"), ("user.email", f"clover@{self.device}.local"),
                         ("core.autocrlf", "false")):
                self._git("config", k, v)
        for fp in (self.dir / "sessions").glob("*/*.jsonl"):
            self._index[fp.stem] = fp
        self.protect_agents()
        self.schedule("Records written while the Studio was not running"
                      if self._git("rev-parse", "--verify", "-q", "HEAD")[0] == 0 else "Clover AI Studio records")

    def protect_agents(self):
        """AGENTS.md holds Clover's operating rules: never edited here. A changed or missing copy is restored."""
        dst = self.dir / "AGENTS.md"
        if not self.agents_src.is_file():
            return None
        want = self.agents_src.read_bytes()
        if dst.is_file() and dst.read_bytes() == want:
            if os.stat(dst).st_mode & stat.S_IWRITE:
                os.chmod(dst, stat.S_IREAD)
            return None
        existed = dst.exists()
        if existed:
            os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)
        dst.write_bytes(want)
        os.chmod(dst, stat.S_IREAD)
        if existed:
            self.log(None, None, "AGENTS.md had been changed; restored the protected copy")
        return "restored" if existed else "written"

    def agents_writable(self):
        """For a sync only: git cannot update a read-only file. protect_agents() runs again afterwards."""
        dst = self.dir / "AGENTS.md"
        if dst.exists():
            os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)

    def commit(self, message):
        with self.lock:
            if not self.git:
                return None
            self.protect_agents()
            rc, out, _ = self._git("status", "--porcelain")
            if rc or not out:
                return None
            self._git("add", "-A")
            rc, _, err = self._git("commit", "-q", "-m", message)
            if rc:
                print("repo commit failed:", err, flush=True)
                return None
            return self._git("rev-parse", "--short", "HEAD")[1]

    def schedule(self, message):
        with self.lock:
            self._messages.append(message)
        self._dirty.set()

    def _worker(self):
        while True:
            self._dirty.wait()
            time.sleep(1.5)  # a burst of changes becomes one commit
            self.flush()

    def flush(self):
        with self.lock:
            self._dirty.clear()
            msgs, self._messages = self._messages, []
            try:
                self.refresh_context()
                self.commit((msgs[0] if len(msgs) == 1 else f"{msgs[-1]} (and {len(msgs) - 1} more)") if msgs
                            else "Update records")
            except Exception as exc:  # the next change tries again
                print("repo update failed:", type(exc).__name__, exc, flush=True)

    def _append(self, fp, rec):
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _write(self, fp, text):
        try:
            if fp.read_text(encoding="utf-8") == text:
                return
        except OSError:
            pass
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text, encoding="utf-8", newline="\n")

    # ---------------- execution: plain-text log ----------------
    def log(self, sid, n, text):
        t = now()
        line = t.strftime("%Y-%m-%d %H:%M:%S") + (f" {sid}#{n}" if sid else "") + " " + one_line(text, 600)
        fp = self.dir / "execution" / self.device / f"{t:%Y-%m-%d}.log"
        with self.lock:
            fp.parent.mkdir(parents=True, exist_ok=True)
            with open(fp, "a", encoding="utf-8", newline="\n") as f:
                f.write(redact(line)[0] + "\n")

    def note(self, text):
        self.log(None, None, text)
        self.schedule(one_line(text, 72))

    def event(self, sid, n, ev):
        """Engine events of a turn, as they happen (tokens are not logged)."""
        t = ev.get("type")
        if t == "semantics":
            self.log(sid, n, f"{ev.get('understood')} (route: {ev.get('route')}; discernment: {ev.get('discernment')})")
        elif t == "tools":
            off = ev.get("offered")
            names = " ".join(off) if isinstance(off, list) and off else ("all" if off == "all" else "none")
            self.log(sid, n, f"tools offered: {names} (tools mode {ev.get('mode')})")
        elif t == "tool_call":
            self.log(sid, n, f"call {ev.get('name')} {one_line(json.dumps(ev.get('arguments'), ensure_ascii=False), 300)}")
        elif t == "tool_result":
            self.log(sid, n, f"result {ev.get('name')}: {'ran' if ev.get('ran') else 'not run'} ({ev.get('why')}): "
                             f"{one_line(ev.get('result'), 200)}")
        elif t == "permission":
            self.log(sid, n, f"asked for permission: {ev.get('tool')}")
        elif t == "permission_decided":
            self.log(sid, n, f"permission: {ev.get('decision')}")
        elif t == "cycle":
            self.log(sid, n, f"cycle: {ev.get('stage')}")
        elif t == "layer" and ev.get("feedback"):
            self.log(sid, n, f"{ev.get('layer')} check: {one_line(ev.get('feedback'), 200)}")

    # ---------------- sessions: user inputs and answers only ----------------
    def _session_file(self, sid, create=False):
        fp = self._index.get(sid)
        if fp is None and create:
            fp = self.dir / "sessions" / f"{now():%Y-%m}" / f"{sid}.jsonl"
            self._index[sid] = fp
        return fp

    def messages(self, sid):
        fp = self._session_file(sid)
        return _read_jsonl(fp) if fp else []

    def reindex(self):
        """After a sync brought other devices' records."""
        with self.lock:
            self._index = {fp.stem: fp for fp in (self.dir / "sessions").glob("*/*.jsonl")}
            self._last.clear()

    def sessions(self):
        out = []
        for sid, fp in list(self._index.items()):
            try:
                st = fp.stat()
            except OSError:
                continue
            hit = self._cache.get(fp)
            if not hit or hit[0] != (st.st_mtime_ns, st.st_size):
                msgs = _read_jsonl(fp)
                users = [m for m in msgs if m.get("role") == "user"]
                answers = [m for m in msgs if m.get("role") == "clover"]
                info = {"id": sid, "file": fp.relative_to(self.dir).as_posix(), "turns": len(users),
                        "title": one_line(users[0].get("text"), 80) if users else "",
                        "started": msgs[0].get("at") if msgs else "", "updated": msgs[-1].get("at") if msgs else "",
                        "first": one_line(users[0].get("text"), 160) if users else "",
                        "last_answer": one_line(answers[-1].get("text"), 200) if answers else ""}
                self._cache[fp] = hit = ((st.st_mtime_ns, st.st_size), info)
            if hit[1]["turns"]:
                out.append(hit[1])
        out.sort(key=lambda s: s["updated"], reverse=True)
        return out

    def delete_session(self, sid):
        with self.lock:
            fp = self._index.get(sid)
            if fp is None or not fp.is_file():
                self._index.pop(sid, None)
                return False
            for wait in (0.2, 0.5, 1.0, 2.0, None):  # a scanner can hold a just-written file (WinError 32, 2026-11-01)
                try:
                    fp.unlink()
                    break
                except PermissionError:
                    if wait is None:
                        raise
                    time.sleep(wait)
            self._index.pop(sid, None)
            self._last.pop(sid, None)
            for nf in (self.dir / "growth" / "notes").glob("*/*.jsonl"):  # notes about it go too
                recs = _read_jsonl(nf)
                keep = [r for r in recs if r.get("session") != sid]
                if len(keep) != len(recs):
                    self._write(nf, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
            self.log(sid, None, "conversation deleted by the user (git history still has it)")
            self.schedule(f"Delete conversation {sid}")
            return True

    def session_files(self):
        with self.lock:
            return dict(self._index)

    # ---------------- notes: what each exchange held, written by the model while the Studio is idle ----------------
    def notes(self):
        """-> {(session, n): note} from every device; "" when the model found nothing to note."""
        out = {}
        for fp in sorted((self.dir / "growth" / "notes").glob("*/*.jsonl")):
            for r in _read_jsonl(fp):
                if isinstance(r.get("n"), int) and r.get("session"):
                    out[(r["session"], r["n"])] = str(r.get("note") or "")
        return out

    def add_note(self, sid, n, note):
        """-> False when the chat was deleted meanwhile (a note being written when it was)."""
        with self.lock:
            if sid not in self._index:
                return False
            self._append(self.dir / "growth" / "notes" / self.device / f"{now():%Y-%m}.jsonl",
                         {"at": iso(), "session": sid, "n": n, "note": redact(note)[0], "by": "LFM3-Omni-2B"})
            return True

    def negative_turns(self):
        """-> {(session, n)}: answers the user said were not right or asked to try again (from every device)."""
        bad = set()
        for fp in (self.dir / "outcomes").glob("*/*.jsonl"):
            for r in _read_jsonl(fp):
                if r.get("feedback") in NEGATIVE:
                    bad.add((r.get("session"), r.get("n")))
        return bad

    # ---------------- outcomes and the turn cycle ----------------
    def _outcomes_file(self):
        return self.dir / "outcomes" / self.device / f"{now():%Y-%m}.jsonl"

    def _turn_state(self, sid, n):
        state = {"used": [], "lesson": None, "judged": False}
        for fp in sorted((self.dir / "outcomes" / self.device).glob("*.jsonl"), reverse=True)[:2]:
            for rec in _read_jsonl(fp):
                if rec.get("session") != sid or rec.get("n") != n:
                    continue
                if "result" in rec:
                    state.update(used=rec.get("used") or [], lesson=rec.get("lesson"),
                                 judged=state["judged"] or rec["result"] == "error")
                if "feedback" in rec:
                    state["judged"] = True
        return state

    def _last_turn(self, sid):
        if sid not in self._last:
            n = sum(1 for m in self.messages(sid) if m.get("role") == "user")
            self._last[sid] = (n, self._turn_state(sid, n) if n else None)
        return self._last[sid]

    def turn_start(self, sid, text, retry=None, files=(), media=()):
        """The user's message: their input goes to sessions/, and their response to the previous answer in this chat
        goes to outcomes/ (a new message: continued; "not right" or "try again": did not work), which Growth learns
        from before this answer is written. -> (turn number, growth changes)"""
        with self.lock:
            prev_n, prev = self._last_turn(sid)
            n = prev_n + 1
            self.current = (sid, n)
            clean, kinds = redact(text)
            rec = {"n": n, "role": "user", "at": iso(), "text": clean}
            if files:
                rec["files"] = [one_line(f, 120) for f in files]
            if media:
                rec["media"] = [{"file": m.get("file"), "kind": m.get("kind"), "name": m.get("name")}
                                for m in media if m.get("file")]
            self._append(self._session_file(sid, True), rec)
            desc = f"message: {len(text)} chars"
            if files:
                desc += ", files: " + ", ".join(rec["files"])
            if media:
                desc += ", media: " + ", ".join(m["file"] for m in rec["media"])
            self.log(sid, n, desc)
            if kinds:
                self.log(sid, n, "redacted before saving: " + ", ".join(kinds))
            changes = []
            if prev_n:
                kind = ("said_not_right" if retry == "said" else "tried_again") if retry else "continued"
                changes = self._judge(sid, prev_n, prev, kind, by=n)
            self._last[sid] = (n, None)
            return n, changes

    def _judge(self, sid, n, state, kind, by):
        state = state if state is not None else self._turn_state(sid, n)
        if state.get("judged"):
            return []
        worked = kind in POSITIVE
        self._append(self._outcomes_file(), {"at": iso(), "session": sid, "n": n, "feedback": kind, "by": by,
                                             "worked": worked})
        self.log(sid, n, f"feedback: {FEEDBACK_WORDS[kind]}" + (f" (turn {by})" if isinstance(by, int) else ""))
        state["judged"] = True
        changes = []
        kit = self.kit() if self.kit else None
        if kit is not None:
            for eid in list(state.get("used") or []) + ([state["lesson"]] if state.get("lesson") else []):
                ch = kit.score_learning(eid, worked, f"{sid} turn {n}: {FEEDBACK_WORDS[kind]}")
                if ch:
                    changes.append(ch)
                    self.log(sid, n, f"growth: {ch}")
        return changes

    def turn_end(self, sid, n, res, seconds):
        """The answer goes to sessions/, what came of the turn to outcomes/; a correction the cycle made becomes a
        learning in growth/. -> growth changes"""
        with self.lock:
            reply, stopped = str(res.get("reply") or ""), bool(res.get("cancelled"))
            if reply and not stopped:
                self._append(self._session_file(sid, True), {"n": n, "role": "clover", "at": iso(),
                                                             "text": redact(reply)[0]})
            used, checks, context = [], [], []
            for lay in res.get("layers") or []:
                name = lay.get("layer")
                if name == "memory":
                    used += lay.get("entries") or []
                    context.append("knowledge " + " ".join(lay.get("entries") or []))
                elif name == "ontology":
                    context.append(f"ontology {lay.get('n_evidence')} records")
                elif name == "workspace":
                    context.append(f"workspace {lay.get('workspace')}")
                elif name == "records":
                    context.append("records " + " ".join(lay.get("parts") or []))
                elif name == "recall" and lay.get("refs"):
                    context.append(f"recall {len(lay['refs'])} earlier messages from {lay.get('chats')} chats: "
                                   + " ".join(lay["refs"][:12]))
                elif name == "compiler" and lay.get("checked"):
                    checks.append({"check": "code", "passed": bool(lay.get("ok"))})
                elif name == "cycle" and lay.get("action"):
                    checks.append({"check": "cycle", "result": lay["action"]})
                elif name == "ontology_grounding" and lay.get("changed"):
                    checks.append({"check": "grounding", "changed": True})
                elif name == "semantics" and lay.get("knowledge"):
                    self.log(sid, n, f"aligned: {lay['knowledge']} ({lay.get('label')})"
                                     + ("; " + ", ".join(lay["aligned"]) if lay.get("aligned") else ""))
            if context:
                self.log(sid, n, "context: " + "; ".join(context))
            tools = [{"tool": c.get("tool"), "ran": bool(c.get("ran")),
                      "worked": bool(c.get("ran")) and not str(c.get("result") or "").lstrip().startswith("Error")}
                     for c in res.get("tool_calls") or []]
            changes, lesson = [], None
            cyc = next((x for x in res.get("layers") or [] if x.get("layer") == "cycle"), {})
            kit = self.kit() if self.kit else None
            if cyc.get("lesson") and kit is not None:
                lesson, ch = kit.learn(cyc["lesson"], sid, n, kind=cyc.get("lesson_kind", "user"))
                changes += ch
            rec = {"at": iso(), "session": sid, "n": n, "result": "stopped" if stopped else "answered"}
            for k, v in (("tools", tools), ("checks", checks), ("used", used), ("lesson", lesson)):
                if v:
                    rec[k] = v
            self._append(self._outcomes_file(), rec)
            rounds = res.get("rounds")
            self.log(sid, n, f"answer: {len(reply)} chars in {seconds:.1f} s, "
                             + ("by semantics, no model" if rounds == 0 else f"{rounds or 1} model rounds")
                             + (", stopped by the user" if stopped else ""))
            for ch in changes:
                self.log(sid, n, f"growth: {ch}")
            self._last[sid] = (n, {"used": used, "lesson": lesson, "judged": False})
            self.current = None
            self.schedule(f"Turn {n} of {sid}: {rec['result']}")
            return changes

    def turn_error(self, sid, n, error):
        with self.lock:
            self._append(self._outcomes_file(), {"at": iso(), "session": sid, "n": n, "result": "error",
                                                 "error": one_line(error, 300)})
            self.log(sid, n, f"error: {one_line(error, 300)}")
            self._last[sid] = (n, {"used": [], "lesson": None, "judged": True})
            self.current = None
            self.schedule(f"Turn {n} of {sid}: error")

    def feedback(self, sid, n, kind):
        """An explicit response to an answer (the Helpful button)."""
        if kind != "helpful":
            raise ValueError("unknown feedback")
        with self.lock:
            last_n, state = self._last_turn(sid)
            if not 1 <= int(n) <= last_n:
                raise ValueError("no such turn in this conversation")
            n = int(n)
            state = state if n == last_n and state is not None else self._turn_state(sid, n)
            changes = self._judge(sid, n, state, kind, by="button")
            if n == last_n:
                self._last[sid] = (n, state)
            self.schedule(f"Turn {n} of {sid}: marked helpful")
            return changes

    # ---------------- retrieval: only what this message needs ----------------
    def context_block(self, q):
        """Earlier conversations when the message refers to them; the growth, outcomes, execution or config summary
        when it asks about them. Nothing otherwise. -> (block, info) or (None, None)"""
        parts, used = [], []
        if PAST.search(q):
            hits = self._past_sessions(q)
            if hits:
                used.append("sessions")
                parts.append("Earlier conversations:\n" + "\n".join(
                    f"- {h['updated'][:16].replace('T', ' ')}, {h['turns']} turns: the user asked \"{h['first']}\"; "
                    f"Clover's last answer: \"{h['last_answer']}\"" for h in hits))
        for name, rx in ASKS:
            if rx.search(q):
                used.append(name)
                body = getattr(self, f"_{name}_summary")()[1]
                parts.append(body.split("\n\n", 2)[-1].strip()[:700])  # the summary without its title and description
        if not parts:
            return None, None
        return ("From Clover's records, already retrieved (answer from them; no tools needed):\n"
                + "\n\n".join(parts))[:1800], {"layer": "records", "parts": used}

    def _past_sessions(self, q):
        cur = self.current[0] if self.current else None
        cands = [s for s in self.sessions() if s["id"] != cur]
        if not cands:
            return []
        if re.search(r"\byesterday\b", q, re.I):
            day = (now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            return [s for s in cands if s["updated"].startswith(day)][:3]
        if LAST_ONE.search(q):
            return cands[:1]
        want = _words(q)
        scored = sorted(((len(want & _words(s["first"] + " " + s["last_answer"])), s) for s in cands),
                        key=lambda x: x[0], reverse=True)
        best = [s for k, s in scored if k][:2]
        return best or cands[:2]

    # ---------------- summaries: a CONTEXT.md in every folder and at the top ----------------
    def refresh_context(self):
        with self.lock:
            parts = {name: getattr(self, f"_{name}_summary")()
                     for name in ("sessions", "execution", "outcomes", "growth", "config")}
            for name, (_, body) in parts.items():
                self._write(self.dir / name / "CONTEXT.md", body)
            self._write(self.dir / "CONTEXT.md", "\n".join([
                "# Clover AI Studio", "",
                "The record that you and Clover both read. Each kind of record is kept once, in its own folder, and "
                "each folder has a CONTEXT.md that summarizes it. Clover rewrites the summaries from the records after "
                "every turn.", "",
                "- AGENTS.md: Clover's operating rules. Protected: Clover restores it if it changes.",
                *[f"- {name}/: {head}" for name, (head, _) in parts.items()], "",
                "Not here: attachments (CloverAI\\media, yours to clear or back up) and secrets (CloverAI\\vault).", ""]))

    def _sessions_summary(self):
        ss = self.sessions()
        turns = sum(s["turns"] for s in ss)
        head = (f"{len(ss)} conversations, {turns} turns" +
                (f"; latest {ss[0]['updated'][:16].replace('T', ' ')}: \"{one_line(ss[0]['title'], 60)}\"" if ss else ""))
        lines = ["# Sessions", "",
                 "Each file is one conversation: your messages and Clover's answers, one JSON line each (\"n\" is the "
                 "turn, \"role\" is user or clover). Nothing else is kept here.", "", f"- {head}.", ""]
        if ss:
            lines += ["## Latest", ""] + [f"- {s['updated'][:16].replace('T', ' ')}, {s['turns']} turns: "
                                         f"\"{s['title']}\" ({s['file'][len('sessions/'):]})" for s in ss[:30]] + [""]
        return head, "\n".join(lines)

    def _execution_summary(self):
        rows, problems, total = [], [], 0
        for fp in sorted((self.dir / "execution" / self.device).glob("*.log"), reverse=True)[:7]:
            answers, calls, notrun, failed, secs = 0, 0, 0, 0, []
            for ln in fp.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.search(r" answer: \d+ chars in ([\d.]+) s", ln)
                if m:
                    answers += 1
                    secs.append(float(m.group(1)))
                elif re.search(r" result \S+: ", ln):
                    calls += 1
                    if ": not run (" in ln:
                        notrun += 1
                    elif re.search(r"\): Error", ln):
                        failed += 1
                        problems.append(ln[:220])
                elif " error: " in ln:
                    problems.append(ln[:220])
            total += answers
            med = f", median answer {statistics.median(secs):.1f} s" if secs else ""
            rows.append(f"- {fp.stem}: {answers} answers, {calls} tool calls ({notrun} not run, {failed} failed){med}")
        head = f"{total} answers logged in the last 7 days on {self.device}"
        lines = ["# Execution", "",
                 "Plain-text logs of how each turn ran: the message size, the context Clover retrieved, the tools it "
                 "offered and called, permission decisions, timings and errors. One file per device and day. "
                 "Attachments and media are never copied here.", "", f"## Last 7 days ({self.device})", ""]
        lines += rows or ["- nothing logged yet"]
        if problems:
            lines += ["", "## Latest problems", ""] + [f"- {p}" for p in problems[-8:]]
        return head, "\n".join(lines) + "\n"

    def _outcome_records(self, days=30):
        cut = (now() - datetime.timedelta(days=days)).isoformat()
        recs = []
        for fp in sorted((self.dir / "outcomes" / self.device).glob("*.jsonl"))[-2:]:
            recs += [r for r in _read_jsonl(fp) if str(r.get("at") or "") >= cut]
        return recs

    def _outcomes_summary(self):
        recs = self._outcome_records()
        results = {(r["session"], r["n"]): r for r in recs if "result" in r}
        verdict = {}
        for r in recs:
            if "feedback" in r:
                verdict.setdefault((r["session"], r["n"]), r)
        answered = [k for k, r in results.items() if r["result"] == "answered"]
        good = [k for k in answered if k in verdict and verdict[k]["worked"]]
        bad = [k for k in answered if k in verdict and not verdict[k]["worked"]]
        count = lambda kind: sum(1 for k in answered if k in verdict and verdict[k]["feedback"] == kind)  # noqa: E731
        stopped = sum(1 for r in results.values() if r["result"] == "stopped")
        errors = sum(1 for r in results.values() if r["result"] == "error")
        tools = {}
        for r in results.values():
            for t in r.get("tools") or []:
                ok, n = tools.get(t["tool"], (0, 0))
                tools[t["tool"]] = (ok + bool(t.get("worked")), n + 1)
        head = (f"{len(answered)} answers in 30 days: {len(good)} worked, {len(bad)} did not, "
                f"{len(answered) - len(good) - len(bad)} not judged yet")
        lines = ["# Outcomes", "",
                 "What came of each turn (answered, stopped or failed, the tools that worked, Clover's checks) and how "
                 "you responded: you continued or marked it helpful (worked), or said it was not right or asked to try "
                 "again (did not work). One JSON line per event; \"session\" and \"n\" point to the turn in sessions/.",
                 "", "## Last 30 days", "",
                 f"- {len(answered)} answers: {len(good)} worked ({count('helpful')} marked helpful, "
                 f"{count('continued')} continued), {len(bad)} did not work ({count('said_not_right')} not right, "
                 f"{count('tried_again')} try again), {len(answered) - len(good) - len(bad)} not judged yet; "
                 f"{stopped} stopped, {errors} errors."]
        if tools:
            lines.append("- Tools: " + ", ".join(f"{k} {ok} of {n} worked" for k, (ok, n) in sorted(tools.items())) + ".")
        if bad:
            lines += ["", "## Did not work (latest)", ""]
            for k in sorted(bad, key=lambda k: verdict[k]["at"], reverse=True)[:8]:
                v = verdict[k]
                said = next((m.get("text") for m in self.messages(k[0])
                             if m.get("role") == "user" and m.get("n") == v.get("by")), "")
                lines.append(f"- {v['at'][:16].replace('T', ' ')}, {k[0]} turn {k[1]}: {FEEDBACK_WORDS[v['feedback']]}"
                             + (f": \"{one_line(said, 100)}\"" if said else ""))
        return head, "\n".join(lines) + "\n"

    def _growth_summary(self):
        kit = self.kit() if self.kit else None
        entries = list(kit.store().entries.values()) if kit is not None else []
        learn = [e for e in entries if "learning" in e.tags]
        notes = [e for e in entries if "learning" not in e.tags]
        rels = [(e.id, r.type, r.target) for e in entries for r in e.relationships]
        state = lambda e: {"active": "holding", "contested": "contested"}.get(e.status, e.status)  # noqa: E731
        by = {s: sum(1 for e in learn if state(e) == s) for s in ("holding", "contested", "superseded")}
        head = (f"{len(learn)} learnings ({by['holding']} holding, {by['contested']} contested, "
                f"{by['superseded']} superseded), {len(notes)} notes, {len(rels)} relationships")
        ex = self.notes()
        lines = ["# Growth", "",
                 "What Clover learned from outcomes (what worked and what did not) and the notes you asked it to keep, "
                 "with typed relationships between them. Clover updates, corrects and retires these as outcomes come "
                 "in; only the ones relevant to a request are given to the model. One Markdown file per entry in "
                 "knowledge/. notes/ holds a short note the model wrote about each exchange, used to find earlier "
                 "messages.", "", f"- {head}.",
                 f"- Exchange notes: {len(ex)} ({sum(1 for v in ex.values() if v)} with something to note).", ""]
        if learn:
            lines += ["## Learnings", ""]
            for e in sorted(learn, key=lambda e: e.updated_at, reverse=True)[:30]:
                w, d = kit.tally(e)
                lines.append(f"- {e.id} ({state(e)}; worked {w}, did not work {d}): {one_line(e.summary, 160)}")
            lines.append("")
        if notes:
            lines += ["## Notes", ""] + [f"- {e.id}: {one_line(e.summary, 160)}"
                                         for e in sorted(notes, key=lambda e: e.created_at, reverse=True)[:30]] + [""]
        if rels:
            lines += ["## Relationships", ""] + [f"- {a} {t.replace('_', ' ')} {b}" for a, t, b in rels[:40]] + [""]
        return head, "\n".join(lines)

    def _config_summary(self):
        cfg = paths.config_dir(self.base)

        def load(name):
            try:
                return json.loads((cfg / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
        s, p, w, g = load("settings.json"), load("permissions.json"), load("workspaces.json"), load("github-repo.json")
        search = load("search.json").get("provider") or "wikipedia"
        layers = ", ".join(f"{k} {'on' if v else 'off'}" for k, v in (s.get("layers") or {}).items())
        perms = ", ".join(f"{k} {v}" for k, v in sorted(p.items()) if isinstance(v, str))
        head = (f"{s.get('model', 'omni')} on {s.get('engine', 'gpu')}, tools {s.get('tools', 'ask')}"
                + (f", GitHub {g['repo']}" if g.get("repo") else ""))
        lines = [f"# Config ({self.device})", "",
                 "The boundaries Clover works within on this device. Secrets are not here: they are in CloverAI\\vault.",
                 "", f"- Model: {s.get('model', 'omni')} on the {s.get('engine', 'gpu')} engine; tools: "
                 f"{'only when you ask' if s.get('tools', 'ask') == 'ask' else 'always offered'}.",
                 f"- Layers: {layers or 'defaults'}.", f"- Permissions: {perms or 'defaults (ask)'}.",
                 f"- Web search: {search}.",
                 f"- Workspaces: {len(w.get('workspaces') or [])}" + (f", in use: {w['active']}" if w.get("active") else ""),
                 f"- Private GitHub repository: {g.get('repo') or 'not set up'}"
                 + (f", last sync {g['last_sync']}" if g.get("last_sync") else ""), ""]
        return head, "\n".join(lines)


def migrate(base):
    """Moves what earlier versions kept elsewhere into the repository, the vault and the media folder (once).
    -> list of what moved"""
    import vault
    base, moved = Path(base), []
    old, cfg, vdir, media = base / "config", paths.config_dir(base), paths.vault_dir(base), paths.media_dir(base)
    for d in (cfg, vdir, media):
        d.mkdir(parents=True, exist_ok=True)
    for src_name, dst in (("settings.json", cfg / "settings.json"), ("permissions.json", cfg / "permissions.json"),
                          ("workspaces.json", cfg / "workspaces.json"), ("clover-repo.json", cfg / "github-repo.json"),
                          ("github.json", vdir / "github.json"), ("vscode-bridge.json", vdir / "vscode-bridge.json")):
        if (old / src_name).is_file() and not dst.exists():
            shutil.move(str(old / src_name), str(dst))
            moved.append(f"{src_name} -> {dst.parent.name}/{dst.name}")
    if (old / "search.json").is_file() and not (cfg / "search.json").exists():  # the keys go to the vault, never here
        try:
            saved = json.loads((old / "search.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        keys = {k: v for k, v in (saved.get("keys") or {}).items() if isinstance(v, str)}
        if keys and not (vdir / "search-keys.json").exists():
            (vdir / "search-keys.json").write_text(json.dumps({"keys_dpapi": vault.protect(json.dumps(keys))}),
                                                    encoding="utf-8")
        (cfg / "search.json").write_text(json.dumps({"provider": saved.get("provider") or "wikipedia"}, indent=2),
                                         encoding="utf-8")
        (old / "search.json").unlink()
        moved.append("search.json -> config (provider), vault (keys)")
    old_mem, know = base / "memory", paths.knowledge_dir(base)
    if old_mem.is_dir():
        for fp in old_mem.rglob("*.md"):
            rel = fp.relative_to(old_mem)
            if rel.parts[0].startswith("_") or (know / rel).exists():
                continue
            (know / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(fp), str(know / rel))
            moved.append(f"memory/{rel.as_posix()} -> growth/knowledge")
        if not any(old_mem.rglob("*.md")):
            shutil.rmtree(old_mem, ignore_errors=True)
    old_media = base / "cache" / "media"
    if old_media.is_dir():
        for fp in old_media.iterdir():
            if fp.is_file() and not (media / fp.name).exists():
                shutil.move(str(fp), str(media / fp.name))
        n = len(list(media.iterdir()))
        try:
            old_media.rmdir()
            moved.append(f"cache/media -> media ({n} files)")
        except OSError:
            pass
    if old.is_dir() and not any(old.iterdir()):
        old.rmdir()
    return moved
