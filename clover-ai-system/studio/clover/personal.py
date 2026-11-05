"""Personal Process Learning in a Clover process (research/personal-process-learning: PPL-1 passed on simulated persons,
2026-11-03). On by default (human, 2026-11-03: "keep the learning on by default ... give option to disable"): a process
gives its semantics.Runtime a Personal for the person's store; the person can switch learning off (set_learning) and
back on, and can always see and delete what was kept. A Runtime without one changes and records nothing.

From the person's own interactions only, it learns
- what they mean: "in messages like these, this person asks for these capabilities". A validated hypothesis replaces
  the general reading in semantics.interpret; a capability the message declines is still not offered, and the
  questions Clover asks before changing files are still asked;
- how they like answers: format, length, sources, a plan first, one pick or several options, asking before acting. A
  learned preference can only make Clover more careful: "just do it" never removes a question Clover asks.
An answered turn becomes an experience when the next message shows what was meant: thanks or the Helpful button
(accepted), a correction that names what was wanted ("No, search the web for it"), a request for another form
("Shorter, please"), or a retry right after a personal reading (that reading was wrong). Statements ("When I say X, I
mean Y", "I prefer short answers") are evidence, not rules. The learner is PPL-1's, with its registered settings
(validated after 2 supporting experiences at precision 0.7, provisional hypotheses not applied) and one runtime
addition: a retry after a personal reading counts against it.

The person decides what stays: keep, stop learning (the record stays, learning ignores it) or delete (removed from the
store and from its whole git history). Deleting a conversation deletes its experiences. What was learned is never
stored: it is recomputed from the eligible experiences, so it cannot outlive them.
Not here yet (PPL-1 tested them in simulation only): routines and open items.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import clover_cycle as C
import semantics

# ---------------------------------------------------------------- features (research ppl/features.py)
LIGHT = {"be", "have", "do", "get", "give", "make", "let", "take", "could", "would", "should", "can", "need", "want",
         "like", "go", "mind", "please", "pls", "kindly", "just", "quick", "chance", "when", "ahead", "you", "i", "me",
         "my", "it", "this", "that", "the", "a", "an", "for", "to", "of", "and", "now", "some", "somewhere", "again", "so"}
CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "X", "INTJ", "NUM", "ADP", "PART"}
ARROWS = ("->", "\u2192", "=>")
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _objects(text):
    """Spans of things a habit must not depend on (links, paths, numbers) and their kinds."""
    spans, kinds = [], set()
    for kind, rx in (("url", URL), ("path", semantics.USER_PATH), ("number", semantics.NUMBER)):
        for m in rx.finditer(text or ""):
            if not any(m.start() < e and s < m.end() for s, e in spans):
                spans.append((m.start(), m.end()))
                kinds.add(kind)
    return spans, kinds


def features(text, analysis=None, device=""):
    """What a hypothesis may depend on: content words (CSI lemmas when CSI reads the message, else the words as
    written), the kinds of objects mentioned, arrows and colons, and the device. Object names never enter."""
    text = text or ""
    spans, kinds = _objects(text)
    f = {f"has={k}" for k in kinds}
    if device:
        f.add(f"ctx={device}")
    outside = "".join(" " if any(s <= i < e for s, e in spans) else ch for i, ch in enumerate(text))
    if any(a in outside for a in ARROWS):
        f.add("pat=arrow")
    if ":" in outside:
        f.add("pat=colon")
    readings = (analysis or {}).get("readings") or []
    if readings:
        tokens = [(t["text"], t["lemma"], t["upos"], t["idx"]) for t in readings[0].get("tokens") or []]
        lemmas = True
    else:
        tokens = [(m.group(0), m.group(0), "X", m.start()) for m in WORD.finditer(text)]
        lemmas = False
    for word, lemma, upos, idx in tokens:
        if any(s <= idx < e for s, e in spans):
            continue
        lem = (lemma if lemmas else word).lower().strip(".'")
        if not lem or not re.search(r"[a-z]", lem) or lem in LIGHT or upos not in CONTENT_UPOS:
            continue
        f.add(f"tok={lem}")
    return frozenset(f)


# ---------------------------------------------------------------- how anyone asks for another form (general)
# (dimension, value, pattern); per dimension the first match wins, so "no bullets" is prose and "no sources" is without
FORMS = (
    ("format", "prose", r"\bno\s+bullets\b|\bplain\s+(?:text|sentences)\b|\b(?:in|as)\s+(?:full\s+)?(?:sentences|prose|"
                        r"paragraphs)\b|\bwritten\s+as\s+prose\b"),
    ("format", "table", r"\b(?:as|in|into)\s+a\s+table\b|\bmake\s+it\s+a\s+table\b|\btables\b|\btabular\b"),
    ("format", "bullets", r"\bbullet(?:ed)?(?:\s+(?:points?|list))?\b|\bbullets\b|\bas\s+a\s+list\b"),
    ("length", "short", r"\bshorter\b|\btoo\s+long\b|\b(?:brief|briefer|concise)\b|"
                        r"\bkeep\s+(?:(?:your\s+|the\s+|my\s+)?(?:answers?|responses?|replies)|it)\s+short\b|"
                        r"\bshort\s+(?:answers?|responses?|replies)\b|\btl;?\s*dr\b|\btrim\s+it\b"),
    ("length", "long", r"\bmore\s+detail(?:ed|s)?\b|\blonger\b|\bin\s+(?:more\s+)?depth\b|\bgo\s+into\s+(?:more\s+)?"
                       r"(?:depth|detail)\b|\bdetailed\s+answers?\b|\bthorough(?:\s+answers?)?\b|\belaborate\b"),
    ("confirm", "ask", r"\bask\s+(?:me\s+)?(?:first|before)\b|\bcheck\s+with\s+me\b|\bconfirm\s+(?:first|with\s+me|before)\b"),
    ("confirm", "just_do", r"\bjust\s+do\s+it\b|\bdon'?t\s+ask\b|\bno\s+need\s+to\s+(?:ask|check|confirm)\b"),
    ("plan", "act", r"\bno\s+plan\b|\bskip\s+the\s+plan\b|\bwithout\s+a\s+plan\b|\b(?:don'?t|do\s+not)\s+(?:bother\s+with|"
                    r"need)\s+(?:a\s+)?plans?\b"),
    ("plan", "plan_first", r"\bplan\s+(?:it\s+out\s+)?(?:first|before)\b|\b(?:show\s+me|see)\s+(?:the|a)\s+plan\b"),
    ("options", "one", r"\bjust\s+pick\s+one\b|\bone\s+recommendation\b|\byour\s+(?:top\s+)?pick\b|\bsingle\s+pick\b"),
    ("options", "several", r"\b(?:a\s+)?few\s+options\b|\bsome\s+options\b|\bcouple\s+of\s+options\b|\boptions\s+to\s+"
                           r"choose\b"),
    ("sources", "without", r"\bno\s+(?:need\s+for\s+)?sources\b|\bskip\s+the\s+sources\b|\bleave\s+out\s+the\s+sources\b|"
                           r"\b(?:don'?t|do\s+not)\s+need\s+sources\b|\bsources\s+aren'?t\s+necessary\b"),
    ("sources", "with", r"\bwith\s+sources\b|\badd\s+(?:the\s+)?sources\b|\binclude\s+(?:links|sources)\b|\bcite\b|"
                        r"\bcitations?\b"),
)
_FORMS = tuple((d, v, re.compile(p, re.I)) for d, v, p in FORMS)
DEFAULTS = {"format": "prose", "length": "long", "confirm": "ask_when_risky", "plan": "act", "options": "one",
            "sources": "without"}
GUIDE = {("format", "bullets"): "use bullet points", ("format", "table"): "use a table where one fits",
         ("format", "prose"): "write plain sentences, not bullet points", ("length", "short"): "keep answers short",
         ("length", "long"): "give detailed answers", ("sources", "with"): "include sources",
         ("sources", "without"): "leave out sources", ("plan", "plan_first"): "show a short plan before acting",
         ("options", "several"): "offer a few options", ("options", "one"): "give one recommendation",
         ("confirm", "ask"): "check with them before doing anything that changes files or the system"}
PREF_INTRO = re.compile(r"^\W*(?:i\s+(?:really\s+)?(?:prefer|like|love|want)|i\s+(?:don'?t|do\s+not)\s+(?:need|want|like)|"
                        r"always|never|from\s+now\s+on|in\s+(?:the\s+)?future|please\s+(?:always|keep|use|be|give|write|show|include|skip|leave)|"
                        r"no\s+need|show\s+me\s+a\s+plan|just\s+give\s+me|sources\s+aren'?t|you\s+don'?t\s+need|"
                        r"keep\s+(?:your\s+|the\s+|my\s+)?(?:answers?|responses?|replies|it)\s+(?:short|brief|concise|detailed|long)\b|"
                        r"use\s+(?:bullet(?:\s+points?)?|bullets|tables?|plain\s+(?:text|sentences)|prose)\b|"
                        r"be\s+(?:brief|concise|short|thorough|detailed)\b|"
                        r"give\s+(?:short|brief|concise|detailed|thorough)\s+(?:answers?|responses?)\b)", re.I)
STATEMENTS = (re.compile(r"^\W*(?:when|if)\s+i\s+(?:just\s+)?say\s+(?P<x>.+?),?\s+i\s+mean\s+(?P<y>.+?)\W*$", re.I),
              re.compile(r"^\W*(?P<x>[^.?!]+?)\s+means\s+(?P<y>[^.?!]+?)\W*$", re.I))
TYPE_WORDS = {"link": "has=url", "url": "has=url", "website": "has=url", "page": "has=url", "file": "has=path",
              "folder": "has=path", "path": "has=path", "number": "has=number", "arrow": "pat=arrow"}
STATEMENT_LIGHT = {"say", "something", "someone", "thing", "things", "stuff", "word"}
# a new message that corrects the last one and says what was wanted ("No, look it up online"); the runtime answers it as a
# new request, and it also shows what the previous message meant
CORRECTION = re.compile(r"^\W*(?:no|nope|not\s+(?:that|quite)|actually|i\s+meant|i\s+mean)\b\W+(?P<rest>.{3,})$", re.I | re.S)


def form_requests(text):
    """The forms a message asks for: {dimension: value}."""
    out = {}
    for d, v, rx in _FORMS:
        if d not in out and rx.search(text or ""):
            out[d] = v
    return out


def form_feedback(text):
    """A short message that only asks for another form of the last answer ("Shorter, please.", "As a table")."""
    t = text or ""
    if len(WORD.findall(t)) > 20 or semantics.asked_caps(t):
        return {}
    return form_requests(t)


# ---------------------------------------------------------------- the learner (research ppl/learner.py, PPL-1)
TRIGGER_KINDS = ("tok=", "pat=", "has=", "ctx=")
ANCHOR_KINDS = ("tok=", "pat=")
# PPL-1's registered settings (research/personal-process-learning/results/params.json)
PARAMS = dict(K=2, P=0.7, R=2, retire=True, apply_provisional=False, use_ctx=True, k_pref=1, p_pref=0.6, C=0.0,
              mode="full")


def trigger_features(feats, use_ctx=True):
    return frozenset(f for f in feats if f.startswith(TRIGGER_KINDS) and (use_ctx or not f.startswith("ctx=")))


def anchors(feats):
    return {f for f in feats if f.startswith(ANCHOR_KINDS)}


def as_target(t):
    return tuple(t) if isinstance(t, list) else t


class Hypothesis:
    __slots__ = ("id", "trigger", "target", "support", "contra", "origin", "status")

    def __init__(self, hid, trigger, target, origin):
        self.id, self.trigger, self.target, self.origin = hid, frozenset(trigger), target, origin
        self.support, self.contra, self.status = [], [], "provisional"

    def precision(self):
        n = len(self.support) + len(self.contra)
        return len(self.support) / n if n else 0.0

    def to_json(self):
        return {"id": self.id, "trigger": sorted(self.trigger),
                "target": list(self.target) if isinstance(self.target, tuple) else self.target,
                "origin": self.origin, "status": self.status, "support": list(self.support), "contra": list(self.contra)}


class Learner:
    def __init__(self, params=None):
        self.p = dict(PARAMS, **(params or {}))
        self.reset()

    def reset(self):
        self.hyps, self.next_id = [], 0
        self.history = []  # (exp id, trigger features, resolved target)
        self.order, self.statements = {}, set()
        self.pref_support, self.pref_contra = {}, {}

    def observe(self, exp):
        eid, p = exp["id"], self.p
        self.order.setdefault(eid, len(self.order))
        tf = trigger_features(exp["pfeats"], p["use_ctx"])
        if exp.get("statement"):
            self._statement(exp)
        elif exp.get("resolved") is not None and p["mode"] == "full":
            target = as_target(exp["resolved"])
            self._evidence(eid, tf, target, exp["context"])
            g = exp["general"]
            if as_target(g["intent"]) != target or g["conf"] < p["C"]:
                self._generalise(eid, tf, target)
            self.history.append((eid, tf, target))
        elif exp.get("rejected") is not None and p["mode"] == "full":
            # runtime addition: a retry right after a personal reading says that reading was wrong
            rejected = as_target(exp["rejected"])
            for h in self._live():
                if h.trigger <= tf and h.target == rejected:
                    h.contra.append(eid)
        if p["mode"] == "full":
            for fb in exp.get("pres_feedback", []):
                self.pref_support.setdefault(fb["dim"], {}).setdefault(fb["value"], []).append(eid)
                if fb.get("old"):
                    self.pref_contra.setdefault(fb["dim"], {}).setdefault(fb["old"], []).append(eid)
            for d, v in (exp.get("accepted_presentation") or {}).items():
                self.pref_support.setdefault(d, {}).setdefault(v, []).append(eid)
        self._status()

    def _new(self, trigger, target, origin):
        h = Hypothesis(self.next_id, trigger, target, origin)
        self.next_id += 1
        self.hyps.append(h)
        return h

    def _live(self):
        return [h for h in self.hyps if h.status in ("provisional", "validated")]

    def _evidence(self, eid, tf, target, context):
        for h in self._live():
            if h.trigger <= tf:
                if h.target == target:
                    h.support.append(eid)
                else:
                    h.contra.append(eid)
                    self._narrow(h, context)

    def _narrow(self, h, context):
        if not self.p["use_ctx"] or any(f.startswith("ctx=") for f in h.trigger):
            return
        ctx_of = {eid: next((f for f in tf if f.startswith("ctx=")), None) for eid, tf, _ in self.history}
        areas = {ctx_of.get(e) for e in h.support if ctx_of.get(e)}
        if f"ctx={context}" in areas or not areas:
            return
        h.status = "narrowed"
        for a in sorted(areas):
            n = self._new(h.trigger | {a}, h.target, h.origin)
            n.support = [e for e in h.support if ctx_of.get(e) == a]
            n.contra = [e for e in h.contra if ctx_of.get(e) == a]

    def _covers_negative(self, trigger, target):
        return any(t != target and trigger <= tf for _, tf, t in self.history)

    def _generalise(self, eid, tf, target):
        if not anchors(tf):
            return
        live = self._live()
        if any(h.target == target and h.trigger <= tf for h in live):
            return
        candidates = sorted((h for h in live if h.target == target and anchors(h.trigger & tf)),
                            key=lambda h: (-len(h.trigger & tf), h.id))
        for h in candidates:
            merged = h.trigger & tf
            if not self._covers_negative(merged, target):
                kept = {e for e in h.support if e in self.statements}
                h.trigger = frozenset(merged)
                h.support = sorted({e for e, f, t in self.history if t == target and merged <= f} | {eid} | kept,
                                   key=self.order.get)
                h.contra = []
                return
        h = self._new(tf, target, "interaction")
        h.support = sorted({e for e, f, t in self.history if t == target and tf <= f} | {eid}, key=self.order.get)
        h.contra = [e for e, f, t in self.history if t != target and tf <= f]

    def _statement(self, exp):
        st = exp["statement"]
        self.statements.add(exp["id"])
        if st.get("pref"):
            d, v = st["pref"]
            self.pref_support.setdefault(d, {}).setdefault(v, []).append(exp["id"])
            return
        trig = set(st["anchors"])
        if st.get("context") and self.p["use_ctx"]:
            trig.add(f"ctx={st['context']}")
        if not anchors(trig) or not st.get("target"):
            return
        h = self._new(trig, as_target(st["target"]), "statement")
        h.support = [exp["id"]]

    def _status(self):
        p = self.p
        for h in self._live():
            if p["retire"] and len(h.contra) >= p["R"] and h.precision() < 0.5:
                h.status = "retired"
            elif (p["mode"] == "statements" and h.origin == "statement") or (
                    len(h.support) >= p["K"] and h.precision() >= p["P"]):
                h.status = "validated"
            else:
                h.status = "provisional"

    def interpret(self, feats):
        tf = trigger_features(feats, self.p["use_ctx"])
        matches = [h for h in self._live() if h.trigger <= tf]
        pool = [h for h in matches if h.status == "validated"]
        if not pool and self.p["apply_provisional"]:
            pool = [h for h in matches if h.precision() >= 0.5]
        if not pool:
            pool = [h for h in matches if h.origin == "statement" and not h.contra and h.precision() >= 0.5]
        if not pool:
            return None
        return max(pool, key=lambda h: (len(h.trigger), h.precision(), len(h.support), -h.id))

    def preference(self, dim, default):
        sup = self.pref_support.get(dim, {})
        con = self.pref_contra.get(dim, {})
        best, best_n = default, 0
        for v, ids in sup.items():
            n, c = len(ids), len(con.get(v, []))
            if n >= self.p["k_pref"] and n / (n + c) >= self.p["p_pref"] and (n > best_n or (n == best_n and v == default)):
                best, best_n = v, n
        return best

    def references(self):
        out = {e for e, _, _ in self.history} | set(self.statements)
        for h in self.hyps:
            if h.status in ("provisional", "validated"):
                out |= set(h.support) | set(h.contra)
        for vs in list(self.pref_support.values()) + list(self.pref_contra.values()):
            for ids in vs.values():
                out |= set(ids)
        return out

    def rebuild(self, experiences):
        self.reset()
        for e in experiences:
            self.observe(e)


# ---------------------------------------------------------------- the person's store: one git repository
ID = re.compile(r"^[A-Za-z0-9_\-]{1,96}/\d{5}$")


def session_name(session):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(session or "s"))[:96] or "s"


def find_filter_repo():
    """git-filter-repo, when installed (as a git command or a Python module); None otherwise."""
    if shutil.which("git-filter-repo"):
        return ["git", "filter-repo"]
    if importlib.util.find_spec("git_filter_repo") is not None:
        return [sys.executable, "-m", "git_filter_repo"]
    return None


def _writable(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _remove_tree(path):
    kw = {"onexc": _writable} if sys.version_info >= (3, 12) else {"onerror": _writable}
    shutil.rmtree(path, **kw)


class Store:
    """experiences/<session>/<n>.json (one commit each), retention.json (id -> "stop"), tombstones.json (salted hashes
    of deleted experiences, no content). Deleting rewrites the whole history without the experience: git-filter-repo
    when installed, else the repository is rebuilt from what remains."""

    def __init__(self, root, filter_repo=None):
        self.dir = Path(root)
        self.git = shutil.which("git")
        self.filter_repo = (find_filter_repo() if filter_repo is None else (list(filter_repo) or None)) if self.git else None
        if not (self.dir / ".git").exists():
            if self.dir.exists() and any(self.dir.iterdir()):
                raise ValueError(f"{self.dir} is neither empty nor a personal store")
            self.dir.mkdir(parents=True, exist_ok=True)
            self._init()
        salt = self.dir / ".git" / "ppl-salt"
        if not salt.exists():
            salt.parent.mkdir(parents=True, exist_ok=True)
            salt.write_text(os.urandom(16).hex())
        self.salt = salt.read_text().strip()

    @property
    def learning_off(self):
        """The person's switch, kept outside history: present = learning is off."""
        return self.dir / ".git" / "ppl-learning-off"

    def _git(self, *args):
        if not self.git:
            return b""
        r = subprocess.run([self.git, "-C", str(self.dir), "-c", "core.hooksPath=", *args], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"git {args[0]}: {r.stderr.decode(errors='replace').strip()}")
        return r.stdout

    def _init(self):
        if self.git:
            self._git("init", "-q")
            for k, v in (("user.name", "Clover"), ("user.email", "clover@localhost"), ("commit.gpgsign", "false")):
                self._git("config", k, v)
        else:
            (self.dir / ".git").mkdir(parents=True, exist_ok=True)
        for name, value in (("retention.json", {}), ("tombstones.json", [])):
            if not (self.dir / name).exists():
                self._write(name, value)
        self._commit("personal store", ["retention.json", "tombstones.json"])

    def _write(self, rel, value):
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")

    def _read(self, rel, default):
        path = self.dir / rel
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def _commit(self, message, paths):
        if not self.git:
            return
        self._git("add", "-A", "--", *paths)
        if self._git("status", "--porcelain", "--", *paths).strip():
            self._git("commit", "-q", "-m", message)

    @staticmethod
    def path_of(exp_id):
        return f"experiences/{exp_id}.json"

    def add(self, exp):
        rel = self.path_of(exp["id"])
        self._write(rel, exp)
        self._commit(f"experience {exp['id']}", [rel])

    def all_experiences(self):
        out = [json.loads(p.read_text(encoding="utf-8")) for p in (self.dir / "experiences").glob("*/*.json")]
        return sorted(out, key=lambda e: e["order"])

    def retention(self):
        return self._read("retention.json", {})

    def eligible(self):
        stop = self.retention()
        return [e for e in self.all_experiences() if stop.get(e["id"]) != "stop"]

    def keep(self, ids):
        r = self.retention()
        for i in ids:
            r.pop(i, None)
        self._write("retention.json", r)
        self._commit(f"keep {len(ids)}", ["retention.json"])

    def stop_learning(self, ids):
        r = self.retention()
        r.update({i: "stop" for i in ids})
        self._write("retention.json", r)
        self._commit(f"stop learning from {len(ids)}", ["retention.json"])

    def delete(self, ids):
        """Removes the experiences from the working tree and from every commit. Returns the strings that must no
        longer be found anywhere in the repository."""
        ids = sorted(set(ids))
        every = self.all_experiences()
        gone = {e["id"]: e for e in every if e["id"] in ids}
        remaining = [s for e in every if e["id"] not in gone for s in [e["text"], *e.get("reactions", [])]]
        secrets = sorted({s for e in gone.values() for s in [e["text"], *e.get("reactions", [])]
                          if len(s) >= 12 and not any(s in r for r in remaining)})
        stones = self._read("tombstones.json", [])
        stones += [hashlib.sha256((self.salt + e["id"] + e.get("time", "") + e["text"]).encode()).hexdigest()
                   for e in gone.values()]
        self._write("tombstones.json", sorted(set(stones)))
        r = self.retention()
        for i in ids:
            r.pop(i, None)
        self._write("retention.json", r)
        paths = [self.path_of(i) for i in gone]
        for p in paths:
            (self.dir / p).unlink(missing_ok=True)
        self._commit(f"delete {len(paths)} experiences", ["tombstones.json", "retention.json", *paths])
        if self.filter_repo:
            args = ["--force", "--quiet", "--invert-paths"]
            for p in paths:
                args += ["--path", p]
            res = subprocess.run([*self.filter_repo, *args], cwd=str(self.dir), capture_output=True)
            if res.returncode != 0:
                raise RuntimeError(f"filter-repo: {res.stderr.decode(errors='replace').strip()}")
            self._git("reflog", "expire", "--expire=now", "--all")
            self._git("gc", "--prune=now", "--quiet")
        else:
            self._rebuild()
        return secrets

    def _rebuild(self):
        """A new repository holding only what remains: no commit, reflog entry or object keeps the deleted experience."""
        off = self.learning_off.exists()
        _remove_tree(self.dir / ".git")
        if self.git:
            self._git("init", "-q")
            for k, v in (("user.name", "Clover"), ("user.email", "clover@localhost"), ("commit.gpgsign", "false")):
                self._git("config", k, v)
        else:
            (self.dir / ".git").mkdir(parents=True, exist_ok=True)
        (self.dir / ".git" / "ppl-salt").write_text(self.salt)
        if off:
            self.learning_off.write_text("off")
        own = [p for p in ("experiences", "retention.json", "tombstones.json") if (self.dir / p).exists()]
        self._git("add", "-A", "--", *own)
        self._git("commit", "-q", "--allow-empty", "-m", "personal store, rebuilt after a deletion")

    def find(self, strings, where=None):
        """Strings found in any object of a repository (reachable or not), in its other git files or its working tree."""
        root = Path(where) if where else self.dir
        blob = (subprocess.run([self.git, "-C", str(root), "cat-file", "--batch-all-objects", "--batch"],
                               capture_output=True).stdout if self.git else b"")
        for p in root.rglob("*"):
            if p.is_file() and p.relative_to(root).parts[:2] != (".git", "objects"):
                blob += p.read_bytes()
        return [s for s in strings if s.encode("utf-8") in blob]


# ---------------------------------------------------------------- a person's learning in a process
CAP_WORDS = semantics.CAP_WORDS


class Personal:
    """One person's experiences, knowledge and retention choices. Thread-safe; one per process and store."""

    def __init__(self, root, device="", csi=None, filter_repo=None, params=None):
        self.store = Store(root, filter_repo)
        self.device, self.csi = device, csi
        self.learner = Learner(params)
        self.lock = threading.RLock()
        self.pending = OrderedDict()  # session -> the last answered turn, until the next message shows what was meant
        everything = self.store.all_experiences()
        self.n = max((e["order"] for e in everything), default=-1) + 1
        self.learner.rebuild(self.store.eligible())

    # ---- the person's switch ----
    @property
    def learning(self):
        return not self.store.learning_off.exists()

    def set_learning(self, on):
        """Learning on or off; what was kept stays viewable and deletable either way."""
        with self.lock:
            if on:
                self.store.learning_off.unlink(missing_ok=True)
            else:
                self.store.learning_off.write_text("off")
                self.pending.clear()
            return self.learning

    # ---- reading a message ----
    def _features(self, text, analysis=None):
        if analysis is None and self.csi is not None:
            try:
                analysis = self.csi(text)
            except Exception:
                analysis = None
        return features(text, analysis, self.device)

    def _presentation(self, caps, action):
        dims = ["format", "length", "options"] + (["sources"] if "web.search" in caps else []) + (
            ["plan", "confirm"] if action else [])
        return {d: self.learner.preference(d, DEFAULTS[d]) for d in dims}

    def interpret(self, text, frame):
        """semantics.interpret's call: the person's reading of this message, or nothing personal."""
        with self.lock:
            if not self.learning:
                return None
            feats = self._features(text, analysis=getattr(frame, "_csi", None))
            general = sorted(frame.caps)
            out = {"general": general, "features": sorted(trigger_features(feats))}
            h = self.learner.interpret(feats)
            if h is not None and isinstance(h.target, tuple) and list(h.target) != general:
                out.update(hypothesis=h.id, caps=list(h.target), support=len(h.support))
            caps = out.get("caps", general)
            prefs = self._presentation(caps, bool(caps) or frame.act == "action")
            out["presentation"] = prefs
            wanted = [GUIDE[(d, v)] for d, v in prefs.items() if v != DEFAULTS[d] and (d, v) in GUIDE]
            if wanted:
                out["guidance"] = ("How this person likes answers (learned from their own feedback; follow it unless "
                                   "this message asks otherwise): " + "; ".join(wanted) + ".")
            out["ask_first"] = prefs.get("confirm") == "ask"
            return out

    # ---- what the next message shows ----
    def after_turn(self, session, frame, res=None):
        """The turn Clover just answered; it is learned from only if the next message shows what was meant."""
        if frame is None or frame.act in ("social", "retry") or not frame.text:
            return
        with self.lock:
            if not self.learning:
                return
            p = frame.personal or {}
            feats = p.get("features")
            if feats is None:
                feats = sorted(trigger_features(self._features(frame.text, analysis=getattr(frame, "_csi", None))))
            self.pending[session_name(session)] = {
                "session": session_name(session), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "context": self.device, "text": frame.text, "pfeats": feats,
                "general": {"intent": sorted(p.get("general", frame.caps)), "conf": 1.0},
                "first": {"target": sorted(frame.caps), "source": "personal" if p.get("hypothesis") is not None
                          else "general", "hyp": p.get("hypothesis")},
                "presentation": dict(p.get("presentation") or self._presentation(frame.caps, frame.act == "action")),
                "turn": (res or {}).get("turn_n")}
            while len(self.pending) > 64:
                self.pending.popitem(last=False)

    def before_turn(self, session, text, retry=None):
        """A new message in `session`: what it shows about the previous answer, and any statement it makes."""
        with self.lock:
            if not self.learning:
                return []
            changes = []
            prev = self.pending.pop(session_name(session), None)
            for st in self._statements(text):
                self._record(session, {"context": self.device, "text": text, "pfeats": [], "general": None,
                                       "resolved": None, "statement": st, "how": "statement"})
                if st.get("pref"):
                    changes.append(f"personal preference: {st['pref'][0]}={st['pref'][1]}")
                elif st.get("target"):
                    words = [a[4:] for a in st.get("anchors") or [] if a.startswith("tok=")]
                    meant = ", ".join(dict.fromkeys(CAP_WORDS.get(c, c) for c in st["target"]))
                    changes.append(f"personal habit: {' '.join(words)} -> {meant}")
            if prev is None:
                return changes
            if retry == "said":
                detail = C.user_detail(text)
                caps = semantics.asked_caps(detail) if detail else []
                forms = form_requests(detail) if detail else {}
                if caps:
                    self._resolve(prev, sorted(caps), text, "corrected", forms)
                    changes.append("personal: learned from correction")
                elif forms:
                    self._resolve(prev, prev["first"]["target"], text, "another form", forms)
                    changes.append("personal: learned preferred form")
                elif prev["first"]["source"] == "personal":
                    self._reject(prev, text)
                    changes.append("personal: retired contested reading")
            elif retry:
                if prev["first"]["source"] == "personal":
                    self._reject(prev, text)
                    changes.append("personal: retired contested reading")
            elif {"thanks", "ack"} & set(semantics.social(text)):
                self._resolve(prev, prev["first"]["target"], text, "accepted")
            else:
                m = CORRECTION.match(text or "")
                caps = semantics.asked_caps(m.group("rest")) if m else []
                if caps:
                    self._resolve(prev, sorted(caps), text, "corrected", form_requests(m.group("rest")))
                    changes.append("personal: learned from correction")
                    return changes
                forms = form_feedback(text)
                if forms:
                    self._resolve(prev, prev["first"]["target"], text, "another form", forms)
                    changes.append("personal: learned preferred form")
            return changes

    def helpful(self, session, turn=None):
        """The Helpful button on the last answer."""
        with self.lock:
            if not self.learning:
                return
            prev = self.pending.get(session_name(session))
            if prev is not None and (turn is None or prev.get("turn") in (None, turn)):
                self.pending.pop(session_name(session))
                self._resolve(prev, prev["first"]["target"], "", "accepted")

    def _resolve(self, prev, caps, reaction, how, forms=None):
        exp = dict(prev, resolved=sorted(caps), how=how, reactions=[reaction] if reaction else [])
        exp["pres_feedback"] = [{"dim": d, "value": v, "old": prev["presentation"].get(d, DEFAULTS[d])}
                                for d, v in (forms or {}).items() if v != prev["presentation"].get(d, DEFAULTS[d])]
        if how == "accepted":
            exp["accepted_presentation"] = dict(prev["presentation"])
        self._record(prev["session"], exp)

    def _reject(self, prev, reaction):
        self._record(prev["session"], dict(prev, resolved=None, rejected=prev["first"]["target"],
                                           how="retried after a personal reading", reactions=[reaction]))

    def _statements(self, text):
        t = (text or "").strip()
        out = []
        for rx in STATEMENTS:
            m = rx.match(t)
            if not m:
                continue
            caps = semantics.asked_caps(m.group("y"))
            if not caps:
                break
            words = set()
            for f in self._features(m.group("x")):
                w = f[4:]
                if f.startswith("tok=") and w in TYPE_WORDS:
                    words.add(TYPE_WORDS[w])
                elif f.startswith("tok=") and w not in STATEMENT_LIGHT:
                    words.add(f)
                elif f.startswith(("has=", "pat=")):
                    words.add(f)
            if anchors(words):
                out.append({"anchors": sorted(words), "context": None, "target": sorted(caps)})
            return out
        if PREF_INTRO.search(t):
            out += [{"pref": [d, v]} for d, v in form_requests(t).items()]
        return out

    def _record(self, session, exp):
        exp = dict(exp, id=f"{session_name(session)}/{self.n:05d}", order=self.n)
        exp.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.n += 1
        self.store.add(exp)
        self.learner.observe(exp)
        return exp

    # ---- the person's view and choices ----
    def view(self):
        with self.lock:
            retention = self.store.retention()
            exps = self.store.all_experiences()

            def words(trig):
                return [f[4:] for f in sorted(trig) if f.startswith("tok=")]
            learned = []
            for h in self.learner.hyps:
                if h.status not in ("provisional", "validated"):
                    continue
                target = h.target if isinstance(h.target, tuple) else (h.target,)
                learned.append({"id": h.id, "status": h.status, "from": h.origin, "when_you_say": words(h.trigger),
                                "with": [f[4:] for f in sorted(h.trigger) if f.startswith(("has=", "pat="))],
                                "you_mean": list(dict.fromkeys(CAP_WORDS.get(c, c) for c in target)),
                                "support": list(h.support), "against": list(h.contra)})
            prefs = []
            for d in DEFAULTS:
                for v, ids in sorted(self.learner.pref_support.get(d, {}).items()):
                    prefs.append({"dimension": d, "value": v, "support": list(ids),
                                  "against": list(self.learner.pref_contra.get(d, {}).get(v, [])),
                                  "in_use": self.learner.preference(d, DEFAULTS[d]) == v and v != DEFAULTS[d]})
            return {"enabled": True, "learning": self.learning, "store": str(self.store.dir),
                    "deletion": "git-filter-repo" if self.store.filter_repo else "rebuild",
                    "experiences": [{"id": e["id"], "time": e.get("time", ""), "text": e["text"], "how": e.get("how", ""),
                                     "meant": e.get("resolved"), "retention": retention.get(e["id"], "keep")}
                                    for e in exps],
                    "learned": learned, "preferences": prefs}

    def retention(self, ids, choice):
        """keep | stop | delete for the given experience ids."""
        with self.lock:
            known = {e["id"] for e in self.store.all_experiences()}
            ids = [i for i in ids if isinstance(i, str) and ID.match(i)]
            unknown = [i for i in ids if i not in known]
            if unknown or not ids:
                raise ValueError(f"unknown experience ids: {unknown or ids}")
            out = {"choice": choice, "ids": ids}
            if choice == "keep":
                self.store.keep(ids)
            elif choice == "stop":
                self.store.stop_learning(ids)
            elif choice == "delete":
                secrets = self.store.delete(ids)
                out["left_in_store"] = len(self.store.find(secrets))
            else:
                raise ValueError("choice must be keep, stop or delete")
            self.learner.rebuild(self.store.eligible())
            out["still_referenced"] = len(set(ids) & self.learner.references()) if choice != "keep" else 0
            return out

    def delete_session(self, session):
        """A deleted conversation takes its experiences with it."""
        with self.lock:
            name = session_name(session)
            self.pending.pop(name, None)
            ids = [e["id"] for e in self.store.all_experiences() if e["id"].split("/")[0] == name]
            return self.retention(ids, "delete") if ids else {"choice": "delete", "ids": []}

    def forget_all(self):
        with self.lock:
            self.pending.clear()
            ids = [e["id"] for e in self.store.all_experiences()]
            return self.retention(ids, "delete") if ids else {"choice": "delete", "ids": []}


_PROCESS = {}
_PROCESS_LOCK = threading.Lock()
KEEP = object()  # for_process: leave the store's CSI reader as it is


def for_process(root, device="", csi=KEEP):
    """The one Personal for a store in this process (None when no store is named, CLOVER_PPL is off, or the store
    cannot be opened). csi: the process's current CSI reader (None: CSI off), or KEEP."""
    if not root or os.environ.get("CLOVER_PPL", "").strip().lower() in ("0", "off", "false", "no"):
        return None
    key = os.path.abspath(root)
    with _PROCESS_LOCK:
        if key not in _PROCESS:
            try:
                _PROCESS[key] = Personal(root, device=device, csi=None if csi is KEEP else csi)
            except (OSError, ValueError, RuntimeError) as ex:
                print(f"[Clover] personal learning stays off: {ex}", file=sys.stderr, flush=True)
                return None
        elif csi is not KEEP:
            _PROCESS[key].csi = csi
        return _PROCESS[key]
