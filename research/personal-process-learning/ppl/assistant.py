"""The assistant under each condition, and one interaction with a simulated person.

B0           general interpretation only (CSI features -> shared general interpreter), default presentation
B1           B0 + explicit statements remembered and applied ("remember that ..."), open items kept on request
B2           B0 + episodic memory: the most similar past message (TF-IDF over the words and area) decides, and the
             last accepted form of each presentation dimension is reused
PPL          B0 + Personal Process Learning (learner.py)
PPL-surface  PPL with the personal layer reading surface words instead of CSI lemmas (the general layer unchanged)
PPL-noval    PPL that applies what it saw once and never retires anything (no validation)
PPL-noctx    PPL whose hypotheses ignore the area they were learned in
"""
import math
import re
from collections import Counter

from . import features as F
from . import world as W
from .learner import Learner, trigger_features

ARMS = ("B0", "B1", "B2", "PPL", "PPL-surface", "PPL-noval", "PPL-noctx")
LEARNING_ARMS = ("B1", "PPL", "PPL-surface", "PPL-noval", "PPL-noctx")
MAX_TURNS = 6
STATEMENT_RE = (re.compile(r"^(?:when|if) i (?:just )?say (?P<x>.+?),? i mean (?P<y>.+?)\.?$", re.I),
                re.compile(r"^(?P<x>.+?) means (?P<y>.+?)\.?$", re.I))
AREA_WORDS = {"code": "code", "document": "docs", "doc": "docs", "email": "email", "file": "files", "meeting": "calendar",
              "calendar": "calendar"}
TYPE_WORDS = {"format": "has=fmt", "person": "has=person", "someone": "has=person", "time": "has=time",
              "folder": "has=folder", "arrow": "pat=arrow"}
STATEMENT_LIGHT = {"say", "from", "an", "a", "on", "someone"}
CORRECTION_PREFIX = re.compile(r"^\s*(?:no,?\s*i meant:?|none of those\.?\s*i mean:?|no\.)\s*", re.I)
VAGUE = re.compile(r"^\s*that'?s not what i meant\.?\s*$", re.I)


def option_text(target):
    return "routine:" + "|".join(target) if isinstance(target, tuple) else target


def option_target(text):
    return tuple(text.split(":", 1)[1].split("|")) if text.startswith("routine:") else text


def jsonable(target):
    return list(target) if isinstance(target, tuple) else target


def words(text):
    return re.findall(r"[a-z0-9']+|->|→|=>", text.lower())


class EpisodicMemory:
    """Remembers conversations: nearest past message decides; last accepted presentation is reused."""

    def __init__(self):
        self.docs, self.df, self.last_form, self.open = [], Counter(), {}, {}

    def _vec(self, text, context):
        return Counter(words(text) + [f"ctx_{context}"])

    def add(self, exp):
        if exp.get("statement") or exp.get("resolved") is None:
            return
        v = self._vec(exp["text"], exp["context"])
        self.docs.append((v, exp["resolved"]))
        self.df.update(set(v))
        for fb in exp.get("pres_feedback", []):
            self.last_form[fb["dim"]] = fb["value"]
        self.last_form.update(exp.get("accepted_presentation") or {})
        topics = [s for t, s in exp["objects"] if t == "topic"]
        if exp["resolved"] == "notes.later":
            for s in topics:
                self.open.setdefault(s, exp["id"])
        elif exp["resolved"] == "notes.done":
            for s in topics:
                self.open.pop(s, None)

    def _weights(self, v):
        n = len(self.docs)
        w = {t: c * (math.log((n + 1) / (self.df.get(t, 0) + 1)) + 1) for t, c in v.items()}
        norm = math.sqrt(sum(x * x for x in w.values())) or 1.0
        return {t: x / norm for t, x in w.items()}

    def lookup(self, text, context, tau):
        if not self.docs:
            return None, 0.0
        q = self._weights(self._vec(text, context))
        best, best_s = None, 0.0
        for v, target in self.docs:
            d = self._weights(v)
            s = sum(q[t] * d[t] for t in q if t in d)
            if s > best_s:
                best, best_s = target, s
        return (best if best_s >= tau else None), best_s

    def preference(self, dim, default):
        return self.last_form.get(dim, default)

    def open_items(self):
        return sorted(self.open)


class Assistant:
    def __init__(self, arm, general, reader, params, store=None):
        assert arm in ARMS
        self.arm, self.general, self.reader, self.store = arm, general, reader, store
        self.C, self.tau = params["C"], params.get("tau", 0.5)
        self.personal_csi = arm != "PPL-surface"
        self.surface = F.SurfaceReader()
        lp = dict(params.get("learner", {}), C=self.C)
        if arm == "B1":
            lp.update(mode="statements", apply_provisional=False, retire=False)
        elif arm == "PPL-noval":
            lp.update(K=1, P=0.0, retire=False, k_pref=1, p_pref=0.0, apply_provisional=True)
        elif arm == "PPL-noctx":
            lp.update(use_ctx=False)
        self.learner = Learner(lp) if arm in LEARNING_ARMS else None
        self.memory = EpisodicMemory() if arm == "B2" else None
        self.n, self.experiences = 0, []

    # ---- reading ----
    def analyse(self, text, context):
        g, objs = F.features(text, context, self.reader.read(text), True)
        p = g if self.personal_csi else F.features(text, context, self.surface.read(text), False)[0]
        return {"g": g, "p": p, "objs": objs, "scores": self.general.task_scores(g)}

    def general_top(self, text, context):
        return self.analyse(text, context)["scores"][0][0]

    def parse_presentation(self, text, context, threshold=0.3):
        label, prob = self.general.scores(F.features(text, context, self.reader.read(text), True)[0])[0]
        if label.startswith("pres:") and prob >= threshold:
            d, v = label[5:].split("=")
            return d, v
        return None

    def parse_statement(self, text):
        for rx in STATEMENT_RE:
            m = rx.match(text.strip())
            if not m:
                continue
            x, y = m.group("x"), m.group("y")
            rd = self.reader.read(x) if self.personal_csi else self.surface.read(x)
            area, feats = None, set()
            for word, lemma, upos, dep, idx in rd["tokens"]:
                lem = (lemma if self.personal_csi else word).lower().strip(".'")
                if lem in AREA_WORDS:
                    area = area or AREA_WORDS[lem]
                elif lem in TYPE_WORDS:
                    feats.add(TYPE_WORDS[lem])
                elif re.search(r"[a-z]", lem) and lem not in F.LIGHT and lem not in STATEMENT_LIGHT:
                    feats.add(f"tok={lem}")
            return {"anchors": sorted(feats), "context": area, "target": self.general_top(y, area or "notes")}
        pref = self.parse_presentation(text, "notes", threshold=0.5)
        return {"pref": list(pref)} if pref else None

    # ---- responding ----
    def presentation(self, target):
        dims = ("plan",) if isinstance(target, tuple) else W.INTENTS.get(target, ((), (), ()))[1]
        out = {}
        for d in dims:
            default = W.DEFAULT_PRESENTATION[d]
            if self.learner:
                out[d] = self.learner.preference(d, default)
            elif self.memory:
                out[d] = self.memory.preference(d, default)
            else:
                out[d] = default
        return out

    def act(self, target, source, hyp=None):
        r = {"target": target, "source": source, "hyp": hyp, "presentation": self.presentation(target)}
        if isinstance(target, tuple):
            r["steps"] = [[i, {}] for i in target]
        else:
            r["intent"] = target
            if target == "notes.open":
                r["items"] = (self.learner.open_items() if self.learner else
                              self.memory.open_items() if self.memory else [])
        return r

    def clarify(self, options):
        seen, out = set(), []
        for o in options:
            t = option_text(o)
            if t not in seen:
                seen.add(t)
                out.append(t)
        return {"clarify": out[:4], "source": "clarify", "target": None, "hyp": None, "presentation": {}}

    def respond(self, text, context):
        a = self.analyse(text, context)
        top, conf = a["scores"][0]
        if self.learner:
            h = self.learner.interpret(a["p"])
            if h is not None:
                return self.act(h.target, "personal", h.id), a
        if self.memory:
            t, _ = self.memory.lookup(text, context, self.tau)
            if t is not None:
                return self.act(t if isinstance(t, str) else tuple(t), "memory"), a
        if conf < self.C:
            personal = []
            if self.learner:
                personal = [h.target for h in sorted(self.learner.candidates(a["p"]), key=lambda h: (-len(h.support), h.id))]
            return self.clarify(personal + [l for l, _ in a["scores"][:3]]), a
        return self.act(top, "general"), a

    def from_correction(self, text, context, a, rejected):
        if VAGUE.match(text):
            personal = []
            if self.learner:
                personal = [h.target for h in self.learner.candidates(a["p"]) if h.target != rejected]
            return self.clarify(personal + [l for l, _ in a["scores"] if l != rejected][:3])
        rest = CORRECTION_PREFIX.sub("", text).strip().rstrip(".")
        if " and then " in rest:
            return self.act(tuple(self.general_top(part, context) for part in rest.split(" and then ")), "corrected")
        return self.act(self.general_top(rest, context), "corrected")

    # ---- one interaction ----
    def interact(self, item, person, learn=True):
        """Returns (experience, first response, turns used, resolved?)."""
        text, context = item["text"], item["context"]
        if item["kind"] == "statement":
            exp = self._base(item, None)
            exp["statement"] = self.parse_statement(text)
            if learn:
                self._record(exp)
            return exp, None, 0, True
        resp, a = self.respond(text, context)
        first = resp
        reactions, feedback, resolved, accepted, turns = [], [], None, {}, 0
        for turns in range(1, MAX_TURNS + 1):
            r = person.react(item, resp)
            reactions.append(r["text"])
            if r["type"] == "accept":
                resolved, accepted = resp["target"], dict(resp["presentation"])
                break
            if r["type"] == "clarify_answer":
                if r["choice"] is not None:
                    resp = self.act(option_target(resp["clarify"][r["choice"]]), "clarified")
                else:
                    resp = self.from_correction(r["text"], context, a, None)
            elif r["type"] == "correct":
                resp = self.from_correction(r["text"], context, a, resp.get("target"))
            elif r["type"] == "presentation":
                dv = self.parse_presentation(r["text"], context)
                if dv:
                    feedback.append({"dim": dv[0], "value": dv[1], "old": resp["presentation"].get(dv[0])})
                    resp = dict(resp, presentation=dict(resp["presentation"], **{dv[0]: dv[1]}))
        exp = self._base(item, a)
        exp.update(first={"target": jsonable(first.get("target")), "source": first["source"], "hyp": first.get("hyp"),
                          "clarify": first.get("clarify")},
                   reactions=reactions, resolved=jsonable(resolved), pres_feedback=feedback,
                   accepted_presentation=accepted)
        if learn:
            self._record(exp)
        return exp, first, turns, resolved is not None

    def _base(self, item, a):
        exp = {"id": f"s{item['session']}-{self.n:04d}" if item["session"] != "eval" else f"e-{self.n:04d}",
               "order": self.n, "session": item["session"], "context": item["context"], "text": item["text"],
               "pfeats": sorted(trigger_features(a["p"])) if a else [],
               "objects": [[t, s] for _, _, t, s in a["objs"]] if a else [],
               "general": {"intent": a["scores"][0][0], "conf": round(a["scores"][0][1], 6)} if a else None,
               "statement": None, "resolved": None}
        return exp

    def _record(self, exp):
        self.n += 1
        self.experiences.append(exp)
        if self.store is not None:
            self.store.add(exp)
        if self.learner:
            self.learner.observe(exp)
        if self.memory:
            self.memory.add(exp)

    def rebuild(self, experiences):
        if self.learner:
            self.learner.rebuild(experiences)
