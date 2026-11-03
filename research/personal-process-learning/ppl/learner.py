"""Personal Process Learning: interactions -> observations -> hypotheses -> validation -> adaptation.

An observation is a resolved experience: the features of what the person said (from CSI), what the shared general
interpreter understood, and what the person turned out to mean (through acceptance, correction or a clarifying answer).
Where the person's meaning departs from the general one, a hypothesis is formed: in contexts matching a trigger (feature
set: words by lemma, patterns, object types, area), this person means a target (an intent, or a routine of steps).
Hypotheses generalise by keeping what their observations share, never so far that they cover a known counter-example;
they narrow to their supporting areas when contradicted elsewhere. Every later resolved experience that a hypothesis
covers tests it, whether or not it was applied. Only hypotheses with enough support and precision are validated.
Preferences, routines and open items are learned from the same records. Everything is derived from the experience
store, so it can be recomputed exactly without any experience the person deletes or stops learning from."""

TRIGGER_KINDS = ("tok=", "pat=", "has=", "ctx=")
ANCHOR_KINDS = ("tok=", "pat=")
DEFAULT_PARAMS = dict(K=2, P=0.8, R=2, retire=True, apply_provisional=True, use_ctx=True, k_pref=1, p_pref=0.6,
                      C=0.5, mode="full")


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
        self.p = dict(DEFAULT_PARAMS, **(params or {}))
        self.reset()

    def reset(self):
        self.hyps, self.next_id = [], 0
        self.history = []  # (exp id, trigger features, resolved target)
        self.order, self.statements = {}, set()  # exp id -> observation order; ids of statement experiences
        self.pref_support, self.pref_contra = {}, {}  # dim -> value -> [exp ids]
        self.open = {}  # topic -> exp id that opened it

    # ---- observation ----
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
            if g["intent"] != target or g["conf"] < p["C"]:
                self._generalise(eid, tf, target)
            self.history.append((eid, tf, target))
        if exp.get("resolved") is not None:
            target = as_target(exp["resolved"])
            topics = [s for t, s in exp["objects"] if t == "topic"]
            if target == "notes.later":
                for s in topics:
                    self.open.setdefault(s, eid)
            elif target == "notes.done":
                for s in topics:
                    self.open.pop(s, None)
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
        """A hypothesis without an area, contradicted in an area where it has no support, keeps only its supporting
        areas."""
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
            return  # already covered (and credited in _evidence)
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

    # ---- adaptation ----
    def interpret(self, feats):
        tf = trigger_features(feats, self.p["use_ctx"])
        matches = [h for h in self._live() if h.trigger <= tf]
        pool = [h for h in matches if h.status == "validated"]
        if not pool and self.p["apply_provisional"]:
            pool = [h for h in matches if h.precision() >= 0.5]
        if not pool:
            return None
        return max(pool, key=lambda h: (len(h.trigger), h.precision(), len(h.support), -h.id))

    def candidates(self, feats):
        tf = trigger_features(feats, self.p["use_ctx"])
        return [h for h in self._live() if h.trigger <= tf]

    def preference(self, dim, default):
        sup = self.pref_support.get(dim, {})
        con = self.pref_contra.get(dim, {})
        best, best_n = default, 0
        for v, ids in sup.items():
            n, c = len(ids), len(con.get(v, []))
            if n >= self.p["k_pref"] and n / (n + c) >= self.p["p_pref"] and (n > best_n or (n == best_n and v == default)):
                best, best_n = v, n
        return best

    def open_items(self):
        return sorted(self.open)

    # ---- provenance and recomputation ----
    def knowledge(self):
        return {"hypotheses": [h.to_json() for h in self.hyps],
                "preferences": {d: {v: list(ids) for v, ids in vs.items()} for d, vs in self.pref_support.items()},
                "preference_contra": {d: {v: list(ids) for v, ids in vs.items()} for d, vs in self.pref_contra.items()},
                "open": dict(self.open)}

    def references(self):
        """Every experience id that current knowledge depends on."""
        out = set(self.open.values())
        for h in self.hyps:
            if h.status in ("provisional", "validated"):
                out |= set(h.support) | set(h.contra)
        for vs in list(self.pref_support.values()) + list(self.pref_contra.values()):
            for ids in vs.values():
                out |= set(ids)
        out |= {e for e, _, _ in self.history}
        return out

    def rebuild(self, experiences):
        self.reset()
        for e in experiences:
            self.observe(e)
