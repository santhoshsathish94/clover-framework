"""Checks for Personal Process Learning 0.1. Runs without CSI or sklearn (a surface reader and a small naive-Bayes
stand-in for the general interpreter); the store checks run when PPL_FILTER_REPO names a git-filter-repo command.

    python tests/test_ppl.py
"""
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppl import evaluate as E  # noqa: E402
from ppl import features as F  # noqa: E402
from ppl import general as G  # noqa: E402
from ppl import personas as P  # noqa: E402
from ppl import scenarios as SC  # noqa: E402
from ppl import world as W  # noqa: E402
from ppl.assistant import Assistant  # noqa: E402
from ppl.learner import Learner  # noqa: E402

N = {"ok": 0, "fail": 0}


def check(cond, name):
    N["ok" if cond else "fail"] += 1
    if not cond:
        print("FAIL", name)


class StubGeneral:
    """Naive Bayes over surface features from the same population data (stand-in for the trained interpreter)."""

    def __init__(self, rows, reader):
        self.count, self.label_n, self.vocab = defaultdict(Counter), Counter(), set()
        for t, c, lab in rows:
            fs = F.features(t, c, reader.read(t), False)[0]
            self.count[lab].update(fs)
            self.label_n[lab] += 1
            self.vocab |= fs
        self.labels = sorted(self.label_n)

    def scores(self, feats):
        feats = [f for f in feats if f in self.vocab]
        z = {}
        total = sum(self.label_n.values())
        for lab in self.labels:
            n = self.label_n[lab]
            z[lab] = math.log(n / total) + sum(math.log((self.count[lab][f] + 0.1) / (n + 0.2)) for f in feats)
        m = max(z.values())
        e = {k: math.exp(v - m) for k, v in z.items()}
        s = sum(e.values())
        return sorted(((k, v / s) for k, v in e.items()), key=lambda x: -x[1])

    def task_scores(self, feats):
        s = [(l, p) for l, p in self.scores(feats) if not l.startswith("pres:")]
        t = sum(p for _, p in s) or 1.0
        return [(l, p / t) for l, p in s]

    def digest(self):
        return str(sorted(self.label_n.items()))


def exp(eid, feats, target, general, context="code", order=None):
    return {"id": eid, "order": order or 0, "context": context, "pfeats": sorted(feats), "objects": [],
            "general": {"intent": general, "conf": 0.9}, "resolved": target, "statement": None}


def test_world():
    d = P.sample_person(1001, "dev")
    t = P.sample_person(52001, "test")
    check(all(P.HABIT_BY_ID[h]["pool"] == "dev" for h in d["habits"]), "dev person uses dev habits")
    check(all(P.HABIT_BY_ID[h]["pool"] == "test" for h in t["habits"]), "test person uses test habits")
    check(len(d["habits"]) == 8, "eight habits")
    tr, ev = E.person_items(d)
    train_texts = {it["text"] for it in tr}
    check(not any(it["text"] in train_texts for it in ev if it["kind"] == "habit"), "evaluation habit messages are new")
    train_objs = {v for it in tr for v in it["slots"].values()}
    for it in ev:
        if it["kind"] == "habit":
            for slot, v in it["slots"].items():
                if slot in W.SPLIT_TYPES:
                    check(v not in train_objs or v in W.objects(slot, "eval"), f"eval object fresh {v}")
    kinds = Counter(it["kind"] for it in ev)
    check(kinds["habit"] == 48 and kinds["routine"] == 3 and kinds["open_query"] == 2 and kinds["oneoff_probe"] == 8,
          f"eval kinds {kinds}")
    oneoffs = [it for it in tr if it["kind"] == "oneoff"]
    check(len(oneoffs) == 4 and all(it["intent"] == it["oneoff"]["meant"] != it["oneoff"]["intent"] for it in oneoffs),
          "one-off items mean something else once")
    check(all(it["intent"] == it["oneoff"]["intent"] for it in ev if it["kind"] == "oneoff_probe"),
          "one-off probes mean the ordinary thing")
    check(0.0 <= d["inattention"] <= 0.08, "inattention rate")
    check(all(isinstance(it["session"], int) for it in tr), "training sessions numbered")
    later = [k for k, it in enumerate(tr) if it["kind"] == "open_create"]
    done = [k for k, it in enumerate(tr) if it["kind"] == "open_resolve"]
    check(max(later) < min(done), "open items created before resolved")


def test_features():
    objs = F.find_objects("Move the standup to 3pm and share the Q3 report with Raj.")
    check([o[2] for o in objs] == ["event", "time", "doc", "person"], f"objects {objs}")
    check(F.find_objects("Quantum dots is done.")[0][3] == "quantum dots", "canonical object name")
    fs, _ = F.features("Tidy up utils.py, please.", "code", F.SurfaceReader().read("Tidy up utils.py, please."), False)
    check({"ctx=code", "has=code", "tok=tidy", "tok=up"} <= fs and "tok=please" not in fs, f"features {sorted(fs)}")
    fs, _ = F.features("the proposal -> PDF", "docs", F.SurfaceReader().read("the proposal -> PDF"), False)
    check({"pat=arrow", "has=doc", "has=fmt"} <= fs, f"arrow features {sorted(fs)}")


def test_learner():
    L = Learner({"K": 2, "P": 0.8})
    L.observe(exp("s0-0000", {"ctx=docs", "has=doc", "tok=tidy", "tok=up"}, "docs.proofread", "docs.proofread"))
    L.observe(exp("s0-0001", {"ctx=code", "has=code", "tok=tidy", "tok=up"}, "code.format", "code.refactor"))
    check(len(L.hyps) == 1 and L.hyps[0].status == "provisional", "first mismatch: one provisional hypothesis")
    L.observe(exp("s0-0002", {"ctx=code", "has=code", "tok=tidy"}, "code.format", "code.refactor"))
    h = L.hyps[0]
    check(h.trigger == frozenset({"ctx=code", "has=code", "tok=tidy"}), f"generalised trigger {sorted(h.trigger)}")
    check(h.status == "validated", "validated after two supporting observations")
    check(L.interpret({"ctx=code", "has=code", "tok=tidy", "tok=give"}).target == "code.format", "applies to new form")
    check(L.interpret({"ctx=docs", "has=doc", "tok=tidy"}) is None, "not applied in another area")
    # a merge that would cover a known counter-example is refused
    L.observe(exp("s0-0003", {"ctx=docs", "has=doc", "tok=tidy", "tok=thing"}, "code.format", "docs.proofread", "docs"))
    check(all(not (hh.trigger <= {"ctx=docs", "has=doc", "tok=tidy", "tok=up"}) or hh.target != "code.format"
              for hh in L._live()), "no hypothesis covers the counter-example")
    # area-general habit, then narrowing on contradiction
    M = Learner({"K": 2, "P": 0.8})
    M.observe(exp("a", {"ctx=docs", "has=doc", "tok=park"}, "notes.todo", "docs.share", "docs"))
    M.observe(exp("b", {"ctx=code", "has=code", "tok=park"}, "notes.todo", "code.review", "code"))
    gen = [hh for hh in M._live() if hh.target == "notes.todo"]
    check(len(gen) == 1 and gen[0].trigger == frozenset({"tok=park"}), "generalised across areas")
    check(M.interpret({"ctx=email", "has=msg", "tok=park"}).target == "notes.todo", "applies in a new area")
    M.observe(exp("c", {"ctx=calendar", "has=event", "tok=park"}, "calendar.move", "calendar.move", "calendar"))
    live = [hh for hh in M._live() if hh.target == "notes.todo"]
    check(sorted(sorted(hh.trigger) for hh in live) == [["ctx=code", "tok=park"], ["ctx=docs", "tok=park"]],
          f"narrowed to supporting areas {[sorted(hh.trigger) for hh in live]}")
    # recomputation without an experience removes what depended on it
    exps = [exp("x1", {"ctx=files", "has=file", "tok=zap"}, "files.compress", "files.delete", "files", 1),
            exp("x2", {"ctx=files", "has=file", "tok=zap"}, "files.compress", "files.delete", "files", 2),
            exp("x3", {"ctx=email", "has=msg", "tok=bump"}, "email.flag", "email.archive", "email", 3)]
    R = Learner({"K": 2, "P": 0.8})
    R.rebuild(exps)
    before = R.knowledge()
    R.rebuild(exps)
    check(R.knowledge() == before, "rebuild is deterministic")
    R.rebuild([e for e in exps if e["id"] != "x1"])
    check("x1" not in R.references(), "deleted experience no longer referenced")
    check(R.interpret({"ctx=files", "has=file", "tok=zap"}).status == "provisional", "support fell below K")
    # routines and open items
    Q = Learner({"K": 2})
    for k in range(2):
        Q.observe(exp(f"r{k}", {"ctx=notes", "tok=usual"}, ["docs.export", "docs.share"], "files.open", "notes"))
    check(Q.interpret({"ctx=notes", "tok=usual", "tok=time"}).target == ("docs.export", "docs.share"), "routine")
    o = exp("o1", {"ctx=notes"}, "notes.later", "notes.later", "notes")
    o["objects"] = [["topic", "heat pumps"]]
    Q.observe(o)
    o2 = dict(o, id="o2", resolved="notes.done", general={"intent": "notes.done", "conf": 0.9})
    o3 = dict(o, id="o3", objects=[["topic", "coral reefs"]])
    Q.observe(o3)
    check(Q.open_items() == ["coral reefs", "heat pumps"], "open items")
    Q.observe(o2)
    check(Q.open_items() == ["coral reefs"], "resolved item closed")
    # preferences
    S = Learner({"k_pref": 1})
    S.observe(dict(exp("p1", {"ctx=code"}, "code.review", "code.review"),
                   pres_feedback=[{"dim": "length", "value": "short", "old": "long"}],
                   accepted_presentation={"length": "short", "format": "prose"}))
    check(S.preference("length", "long") == "short" and S.preference("format", "prose") == "prose", "preferences")


def test_end_to_end(stub, reader):
    params = {"C": 0.3, "tau": 0.5, "learner": {"K": 2, "P": 0.8}}
    spec = P.sample_person(1001, "dev")
    res = {}
    for arm in ("B0", "B1", "B2", "PPL", "PPL-noval", "PPL-noctx"):
        asst, log, recs = E.run(spec, arm, stub, reader, params)
        res[arm] = E.summarise(recs)
        check(len(recs) == len(E.person_items(spec)[1]), f"{arm} scored every item")
    print("  end-to-end (stub, one dev person):",
          {a: (round(v["habit"]["acc"], 2), round(v["all"]["acc"], 2), round(v["presentation"]["adherence"] or 0, 2))
           for a, v in res.items()})
    check(res["PPL"]["habit"]["acc"] >= res["B0"]["habit"]["acc"], "PPL not worse than B0 on habits (smoke)")
    a = Assistant("PPL", stub, reader, params)
    st = a.parse_statement("When I say tidy for code, I mean format it.")
    check(st and st["context"] == "code" and "tok=tidy" in st["anchors"], f"statement parse {st}")
    st = a.parse_statement("An arrow from a document to a format means export it.")
    check(st and {"pat=arrow", "has=fmt"} <= set(st["anchors"]) and st["context"] == "docs", f"arrow statement {st}")


def test_store(stub, reader):
    cmd = os.environ.get("PPL_FILTER_REPO")
    if not cmd:
        print("  store checks skipped (PPL_FILTER_REPO not set)")
        return
    from ppl.store import PersonalStore
    params = {"C": 0.3, "tau": 0.5, "learner": {"K": 2, "P": 0.8}}
    spec = P.sample_person(1002, "dev")
    with tempfile.TemporaryDirectory() as tmp:
        store = PersonalStore(tmp, "p", cmd.split())
        tr, ev = E.person_items(spec)
        asst = Assistant("PPL", stub, reader, params, store)
        log = E.train(asst, spec, tr)
        store.save_knowledge(asst.learner.knowledge())
        out = E.retention_test(asst, log, spec, ev, stub, reader, params, store, str(Path(tmp) / "copy"))
        print("  retention:", out)
        check(out["rebuild_identical"], "rebuild from store equals live knowledge")
        check(out["stop"]["records_kept"] == 1.0 and out["stop"]["references_left"] == 0, "stop learning")
        check(out["delete"]["records_removed"] == 1.0 and out["delete"]["references_left"] == 0, "delete")
        check(out["delete"]["found_in_store"] == 0, "deleted text absent from the repository")
        check(out["delete"]["found_in_copy"] > 0, "copy made before deletion still holds it (reported)")


def main():
    reader = F.SurfaceReader()
    stub = StubGeneral(G.population_messages(per_phrase=6), reader)
    test_world()
    test_features()
    test_learner()
    test_end_to_end(stub, reader)
    test_store(stub, reader)
    print(f"{N['ok']} passed, {N['fail']} failed")
    sys.exit(1 if N["fail"] else 0)


if __name__ == "__main__":
    main()
