"""Training on a simulated person's sessions, scoring on fresh scenarios, and the retention, leakage and statistics
procedures. Ground truth is used only by the simulated person and for scoring."""
import itertools
import random
import statistics

from . import personas as P
from . import scenarios as SC
from . import simulator as S
from . import world as W
from .assistant import Assistant, jsonable


def person_items(spec):
    s = spec["seed"]
    return SC.training_sessions(spec, s * 7 + 1), SC.evaluation_items(spec, s * 7 + 2)


def train(asst, spec, items):
    person = S.Person(spec, spec["seed"] * 7 + 3)
    log = []
    for it in items:
        exp, first, turns, ok = asst.interact(it, person, learn=True)
        log.append({"id": exp["id"], "kind": it["kind"], "habit": it.get("habit"),
                    "first_ok": bool(first) and first.get("target") == (tuple(x for x, _ in it["steps"]) if it.get("steps")
                                                                        else it["intent"]),
                    "turns": turns, "resolved": ok})
    return log


def open_f1(got, expected):
    got, expected = set(got or []), set(expected)
    if not got and not expected:
        return 1.0
    tp = len(got & expected)
    if tp == 0:
        return 0.0
    p, r = tp / len(got), tp / len(expected)
    return 2 * p * r / (p + r)


def score(asst, spec, items, interact=True):
    person = S.Person(spec, spec["seed"] * 7 + 4)
    out = []
    for it in items:
        resp, _ = asst.respond(it["text"], it["context"])
        clar = "clarify" in resp
        rec = {"kind": it["kind"], "habit": it.get("habit"), "intent": it["intent"], "text": it["text"],
               "context": it["context"], "got": "clarify" if clar else jsonable(resp.get("target")),
               "source": resp["source"], "clarified": clar, "pres_ok": None, "overgen": None, "open_f1": None}
        if it["kind"] == "routine":
            rec["correct"] = not clar and [i for i, _ in resp.get("steps") or []] == [i for i, _ in it["steps"]]
        elif it["kind"] == "open_query":
            items_got = resp.get("items") if resp.get("intent") == "notes.open" else None
            rec["correct"] = items_got is not None and sorted(items_got) == sorted(it["expected_open"])
            rec["open_f1"] = open_f1(items_got, it["expected_open"]) if items_got is not None else 0.0
        else:
            rec["correct"] = not clar and resp.get("intent") == it["intent"]
        rec["wrong_action"] = not clar and not rec["correct"]
        dims = S.presentation_dims(it)
        if rec["correct"] and dims:
            rec["pres_ok"] = all(resp["presentation"].get(d) == spec["preferences"][d] for d in dims)
            rec["pres_dims"] = list(dims)
        if it["kind"] == "control":
            rec["overgen"] = (not clar) and resp.get("intent") == P.HABIT_BY_ID[it["habit"]]["intent"]
        elif it["kind"] == "oneoff_probe":
            rec["overgen"] = (not clar) and resp.get("intent") == it["oneoff"]["meant"]
        if interact:
            _, _, turns, ok = asst.interact(it, person, learn=False)
            rec["turns"], rec["resolved"] = turns, ok
        out.append(rec)
    return out


def run(spec, arm, general, reader, params, store=None):
    tr, ev = person_items(spec)
    asst = Assistant(arm, general, reader, params, store)
    log = train(asst, spec, tr)
    return asst, log, score(asst, spec, ev)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarise(records):
    """records: list of score() records (any number of persons). Rates by kind and overall."""
    by = {}
    for k in ("habit", "control", "generic", "routine", "open_query", "oneoff_probe"):
        rs = [r for r in records if r["kind"] == k]
        by[k] = {"n": len(rs), "acc": mean([r["correct"] for r in rs]),
                 "clarified": mean([r["clarified"] for r in rs]), "wrong_action": mean([r["wrong_action"] for r in rs])}
    fresh = [r for r in records]
    by["all"] = {"n": len(fresh), "acc": mean([r["correct"] for r in fresh]),
                 "clarified": mean([r["clarified"] for r in fresh]),
                 "wrong_action": mean([r["wrong_action"] for r in fresh])}
    by["control"]["overgen"] = mean([r["overgen"] for r in records if r["kind"] == "control"])
    by["oneoff_probe"]["overgen"] = mean([r["overgen"] for r in records if r["kind"] == "oneoff_probe"])
    by["presentation"] = {"n": sum(1 for r in records if r["pres_ok"] is not None),
                          "adherence": mean([r["pres_ok"] for r in records if r["pres_ok"] is not None])}
    by["open_f1"] = mean([r["open_f1"] for r in records if r["kind"] == "open_query"])
    by["turns"] = mean([r.get("turns") for r in records])
    return by


def sign_flip(diffs):
    """Exact two-sided paired sign-flip test on per-person differences."""
    diffs = [d for d in diffs if d is not None]
    n = len(diffs)
    if n == 0:
        return None
    obs = abs(sum(diffs))
    hits = sum(1 for signs in itertools.product((1, -1), repeat=n)
               if abs(sum(s * d for s, d in zip(signs, diffs))) >= obs - 1e-12)
    return hits / 2 ** n


def per_person(records_by_person, kind):
    """records_by_person: {person: [records]} -> {person: accuracy on kind ('all' for everything)}."""
    out = {}
    for p, rs in records_by_person.items():
        sel = [r["correct"] for r in rs if kind == "all" or r["kind"] == kind]
        out[p] = sum(sel) / len(sel) if sel else None
    return out


# ---- retention: keep / stop learning / delete ----
def habit_ids(log, habit):
    return [r["id"] for r in log if r["habit"] == habit and r["kind"] in ("habit", "statement")]


def dependent(learner, ids):
    ids = set(ids)
    return [h.to_json() for h in learner.hyps if h.status in ("provisional", "validated") and ids & set(h.support)]


def responses(asst, items):
    return [jsonable(asst.respond(it["text"], it["context"])[0].get("target") or "clarify") for it in items]


def retention_test(asst, log, spec, ev, general, reader, params, store, copy_dir):
    """Stop learning from one habit's experiences, then delete another habit's experiences, and check that dependent
    knowledge is gone, behaviour on those habits returns to general interpretation, the rest is unaffected, the
    records remain (stop) or are gone from the whole repository (delete), and copies are reported."""
    b0 = Assistant("B0", general, reader, params)
    validated = {}
    for hid in spec["habits"]:
        ids = set(habit_ids(log, hid))
        validated[hid] = sum(1 for h in asst.learner.hyps if h.status == "validated" and ids & set(h.support))
    order = sorted(spec["habits"], key=lambda h: (-validated[h], spec["habits"].index(h)))
    h_stop, h_del = order[0], order[1]
    out = {"habit_stop": h_stop, "habit_delete": h_del, "validated_before": validated,
           "vacuous_stop": validated[h_stop] == 0, "vacuous_delete": validated[h_del] == 0}

    live = asst.learner.knowledge()
    asst.learner.rebuild(store.eligible())
    out["rebuild_identical"] = asst.learner.knowledge() == live

    def check(habit, ids, before_all):
        items = [it for it in ev if it.get("habit") == habit and it["kind"] == "habit"]
        others = [it for it in ev if not (it.get("habit") == habit and it["kind"] == "habit")]
        now_h, b0_h = responses(asst, items), responses(b0, items)
        now_o = responses(asst, others)
        before_o = [before_all[ev.index(it)] for it in others]
        return {"items": len(items), "reverted_to_general": mean([a == b for a, b in zip(now_h, b0_h)]),
                "others_unchanged": mean([a == b for a, b in zip(now_o, before_o)]),
                "references_left": len(set(ids) & asst.learner.references())}

    before = responses(asst, ev)
    ids = habit_ids(log, h_stop)
    deps = dependent(asst.learner, ids)
    store.stop_learning(ids)
    asst.learner.rebuild(store.eligible())
    store.save_knowledge(asst.learner.knowledge(), "recompute after stop learning")
    res = check(h_stop, ids, before)
    present = {e["id"] for e in store.all_experiences()}
    res.update(dependent_before=len(deps), records_kept=mean([i in present for i in ids]), n_ids=len(ids))
    out["stop"] = res

    store.make_copy(copy_dir, "backup")
    before = responses(asst, ev)
    ids = habit_ids(log, h_del)
    deps = dependent(asst.learner, ids)
    secrets = store.delete(ids)
    asst.learner.rebuild(store.eligible())
    store.save_knowledge(asst.learner.knowledge(), "recompute after deletion")
    res = check(h_del, ids, before)
    present = {e["id"] for e in store.all_experiences()}
    res.update(dependent_before=len(deps), records_removed=mean([i not in present for i in ids]), n_ids=len(ids),
               secrets=len(secrets), found_in_store=len(store.find(secrets)),
               found_in_copy=len(store.find(secrets, where=copy_dir)), copies_registered=len(store.copies()))
    out["delete"] = res
    return out


def interleaved_eval(specs, general, reader, params):
    """Leakage: train every person's assistant in one process, sessions interleaved, then evaluate each."""
    assts, persons, items = {}, {}, {}
    for spec in specs:
        tr, ev = person_items(spec)
        assts[spec["seed"]] = Assistant("PPL", general, reader, params)
        persons[spec["seed"]] = S.Person(spec, spec["seed"] * 7 + 3)
        items[spec["seed"]] = (tr, ev)
    sessions = sorted({it["session"] for tr, _ in items.values() for it in tr})
    for s in sessions:
        for spec in specs:
            for it in [x for x in items[spec["seed"]][0] if x["session"] == s]:
                assts[spec["seed"]].interact(it, persons[spec["seed"]], learn=True)
    return {spec["seed"]: responses(assts[spec["seed"]], items[spec["seed"]][1]) for spec in specs}
