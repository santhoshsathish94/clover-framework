"""Experiments 1d and 1e: CSI reads requests, for Clover's core (EXPERIMENTS.md). Builds the request minimal pairs (Q),
the routing messages (R) and the reply sentences (C) from the fixed templates below (1d: seed 3120; 1e: seed 3121 and the
fresh lists in SET_1E, same forms), runs CSI and the core's own rules on them, the E1 regression sets (S1) and the
core-observation messages (S2), and checks the criteria.
Usage: python run_1d.py <out json> [--set 1d|1e] [--mobile clover-ai-system/mobile] [--observed turns.jsonl ...]"""
import argparse
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from csi import analyze, compare  # noqa: E402
from csi.minimal_pairs import generate  # noqa: E402
from run_e1 import balanced_acc, per_phenomenon  # noqa: E402

SEED = 3120
# (Q) request minimal pairs: (base, past) and objects
REQUEST_VERBS = [("open", "opened"), ("delete", "deleted"), ("rename", "renamed"), ("copy", "copied"),
                 ("download", "downloaded"), ("print", "printed"), ("translate", "translated"),
                 ("summarize", "summarized"), ("archive", "archived"), ("upload", "uploaded"), ("sort", "sorted"),
                 ("share", "shared")]
REQUEST_OBJECTS = ["report", "invoice", "photo", "spreadsheet", "contract", "playlist", "presentation", "letter",
                   "folder", "recipe", "schedule", "backup"]
# (R) routing: per capability the requests that name it, and requests with a negation inside their object or reason
TOPICS = ["the population of Jaipur", "the height of Mount Everest", "the boiling point of ethanol",
          "the opening hours of the city library", "the latest version of Python", "the distance from Delhi to Agra"]
EXPRS = ["2**64 + 17", "17 * 23 * 31", "the sum of the first 100 primes", "the square root of 2 to 10 digits"]
PATHS = ["C:\\notes\\todo.txt", "D:\\work\\plan.md", "/home/user/notes.txt"]
FOLDERS = ["C:\\projects", "D:\\photos", "/home/user/docs"]
TOOLS = {
    "web.search": {"ask": ["search the web for {topic}", "look up {topic} online", "browse the internet for {topic}"],
                   "scope": ["search the web for why the update did not install",
                             "look up online why my laptop does not charge",
                             "search the internet for reasons why people cannot sleep"]},
    "python": {"ask": ["use Python to compute {expr}", "use Python to calculate {expr}"],
               "scope": ["use Python to find the numbers below 50 that are not prime",
                         "use Python to check which of these years are not leap years: 1900, 2000, 2024"]},
    "files.read": {"ask": ["read the file {path} and summarize it", "list the files in the folder {folder}",
                           "open the file {path}"],
                   "scope": ["read the file {path} and tell me which tasks are not done",
                             "list the files in {folder} that were not changed this week"]},
    "shell": {"ask": ["run the command git status", "use the terminal to check the disk usage",
                      "run the command ls -la in {folder}"],
              "scope": ["run the command git status and tell me which files are not committed",
                        "use the terminal to find the processes that are not responding"]},
}
ALTS = ["just tell me what you know", "just answer from memory", "give me your best estimate", "just explain it briefly"]
ASKING = {"A1 plain": "{VP}.", "A2 please": "Please {vp}.", "A3 can you": "Can you {vp}?",
          "A4 I need you to": "I need you to {vp}.", "A5 second clause": "{VP} and tell me what you find.",
          "A6 scope: negation inside": "{SCOPE}.", "A7 scope: negation in another clause": "Don't guess, {vp}."}
DECLINING = {"D1 don't, just": "Don't {vp}, {alt}.", "D2 do not": "Do not {vp}; {alt}.",
             "D3 please don't + sentence": "Please don't {vp}. {Alt}.", "D4 never": "Never {vp} for this, {alt}.",
             "D5 I don't need you to": "I don't need you to {vp}, {alt}.",
             "D6 you don't have to": "You don't have to {vp}: {alt}.", "D7 decline one, ask another": "Don't {vp}; {other}."}
NO_VERB_NEGATION = {"N1 no need to": "No need to {vp}, {alt}.", "N2 there is no need to": "There is no need to {vp}; {alt}."}
# (C) reply sentences: claim verbs (base, past, past participle), objects, templates with their gold
CLAIM_VERBS = [("delete", "deleted", "deleted"), ("remove", "removed", "removed"), ("create", "created", "created"),
               ("save", "saved", "saved"), ("write", "wrote", "written"), ("commit", "committed", "committed"),
               ("push", "pushed", "pushed"), ("move", "moved", "moved"), ("rename", "renamed", "renamed"),
               ("install", "installed", "installed"), ("clone", "cloned", "cloned"), ("open", "opened", "opened"),
               ("run", "ran", "run"), ("execute", "executed", "executed")]
CLAIM_OBJECTS = ["report", "folder", "script", "backup", "repository", "branch", "package", "config file"]
CLAIM_TEMPLATES = [(1, "C1 perfect", "I have {pp} the {o}."), (1, "C2 past", "I {past} the {o}."),
                   (1, "C3 passive", "The {o} has been {pp}."), (1, "C4 successfully", "I've successfully {pp} the {o}."),
                   (1, "C5 negation in a relative clause", "I {past} the {o} that you did not need."),
                   (1, "C6 negation in a second clause", "I {past} the {o}, but the log file was not updated."),
                   (0, "N1 not", "I have not {pp} the {o}."), (0, "N2 couldn't", "I couldn't {base} the {o}."),
                   (0, "N3 tried to", "I tried to {base} the {o}."), (0, "N4 future", "I will {base} the {o}."),
                   (0, "N5 never", "I have never {pp} the {o}."), (0, "N6 question", "Should I {base} the {o}?"),
                   (0, "N7 unable", "I was unable to {base} the {o}."), (0, "N8 passive not", "The {o} was not {pp}.")]
NEG_WORDS = re.compile(r"\b(?:not|never|no)\b|n't\b", re.I)  # the naive keyword baseline for routing
# Experiment 1e: fresh words for the same forms (written before any 1e run)
SET_1E = {
    "SEED": 3121,
    "REQUEST_VERBS": [("close", "closed"), ("export", "exported"), ("compress", "compressed"), ("encrypt", "encrypted"),
                      ("restore", "restored"), ("duplicate", "duplicated"), ("edit", "edited"), ("review", "reviewed"),
                      ("scan", "scanned"), ("tag", "tagged"), ("format", "formatted"), ("forward", "forwarded")],
    "REQUEST_OBJECTS": ["document", "draft", "budget", "calendar", "logo", "survey", "receipt", "memo", "archive",
                        "notebook", "brochure", "timetable"],
    "TOPICS": ["the tallest building in Mumbai", "the melting point of iron", "the train timetable from Pune to Goa",
               "the current price of gold", "the author of Godan", "the area of Lake Victoria"],
    "EXPRS": ["3**40 - 1", "123456 * 789", "the product of the first 20 primes", "the cube root of 50 to 8 digits"],
    "PATHS": ["C:\\data\\budget.csv", "E:\\docs\\minutes.md", "/srv/share/plan.txt"],
    "FOLDERS": ["C:\\downloads", "E:\\music", "/srv/share"],
    "TOOLS": {
        "web.search": {"ask": TOOLS["web.search"]["ask"],
                       "scope": ["search the web for why the printer is not working",
                                 "look up online why the app does not open",
                                 "search the internet for reasons why tomatoes do not ripen"]},
        "python": {"ask": TOOLS["python"]["ask"],
                   "scope": ["use Python to list the numbers below 60 that are not divisible by 3",
                             "use Python to find which of 91, 97 and 99 are not prime"]},
        "files.read": {"ask": TOOLS["files.read"]["ask"],
                       "scope": ["read the file {path} and tell me which items are not paid",
                                 "list the files in {folder} that were not opened this month"]},
        "shell": {"ask": TOOLS["shell"]["ask"],
                  "scope": ["run the command git status and tell me which changes are not staged",
                            "use the terminal to find the services that are not running"]}},
    "ALTS": ["just give me a rough answer", "answer from what you already know", "a quick estimate is enough",
             "just summarise what you know"],
    "CLAIM_OBJECTS": ["database", "document", "image", "notebook", "archive", "module", "template", "spreadsheet"],
}


def cap1(s):
    return s[0].upper() + s[1:]


def q_pairs(rng, n=3):
    rows = []
    for v, past in REQUEST_VERBS:
        for _ in range(n):
            o, o2 = rng.sample(REQUEST_OBJECTS, 2)
            w = rng.choice([x for x, _ in REQUEST_VERBS if x != v])
            base = f"{cap1(v)} the {o}."
            rows += [(f"Please {v} the {o}.", base, 1, "equiv:please"),
                     (f"Don't {v} the {o}.", f"Do not {v} the {o}.", 1, "equiv:dont_forms"),
                     (base, f"Don't {v} the {o}.", 0, "dist:negated"),
                     (base, f"{cap1(v)} the {o2}.", 0, "dist:other_object"),
                     (base, f"{cap1(w)} the {o}.", 0, "dist:other_verb"),
                     (base, f"You {past} the {o}.", 0, "dist:request_vs_report")]
    return [{"a": a, "b": b, "label": y, "phenomenon": p} for a, b, y, p in rows]


def _fill(s, rng):
    return s.format(topic=rng.choice(TOPICS), expr=rng.choice(EXPRS), path=rng.choice(PATHS), folder=rng.choice(FOLDERS))


def r_messages(rng, n=3):
    rows = []
    for cap, t in TOOLS.items():
        others = [c for c in TOOLS if c != cap]
        for forms, gold in ((ASKING, "asked"), (DECLINING, "declined"), (NO_VERB_NEGATION, "declined")):
            for form, tpl in forms.items():
                for _ in range(n):
                    vp = _fill(rng.choice(t["ask"]), rng)
                    scope = _fill(rng.choice(t["scope"]), rng)
                    alt = rng.choice(ALTS)
                    other_cap = rng.choice(others)
                    other = _fill(rng.choice(TOOLS[other_cap]["ask"]), rng)
                    text = tpl.format(vp=vp, VP=cap1(vp), SCOPE=cap1(scope), alt=alt, Alt=cap1(alt), other=other)
                    golds = [(cap, gold)] + ([(other_cap, "asked")] if "{other}" in tpl else [])
                    rows.append({"text": text, "form": form, "golds": golds,
                                 "verbal_negation": forms is DECLINING, "scope": form.startswith("A6") or form.startswith("A7")})
    return rows


def c_sentences(rng):
    rows = []
    for base, past, pp in CLAIM_VERBS:
        for gold, form, tpl in CLAIM_TEMPLATES:
            rows.append({"text": tpl.format(base=base, past=past, pp=pp, o=rng.choice(CLAIM_OBJECTS)), "gold": gold,
                         "form": form})
    return rows


def keyword_declined(text, sem):
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+", text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    asked, declined = set(), set()
    for caps, pattern in sem.ASKS:
        for m in re.finditer(pattern, text, re.I):
            sent = next((text[a:b] for a, b in spans if a <= m.start() <= b), text)
            (declined if NEG_WORDS.search(sent) else asked).update(caps)
    return declined - asked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--set", default="1d", choices=("1d", "1e"))
    ap.add_argument("--mobile", default=os.path.join(HERE, "..", "..", "..", "clover-ai-system", "mobile"))
    ap.add_argument("--observed", nargs="*", default=[])
    a = ap.parse_args()
    if a.set == "1e":
        globals().update(SET_1E)
    sys.path.insert(0, a.mobile)
    import semantics as sem
    rng = random.Random(SEED)
    Q, R, Cs = q_pairs(rng), r_messages(rng), c_sentences(rng)
    times, cache = [], {}

    def read(text):
        if text not in cache:
            t0 = time.perf_counter()
            cache[text] = analyze(text)
            times.append(time.perf_counter() - t0)
        return cache[text]

    out = {"set": a.set, "seed": SEED, "csi_parser": os.environ.get("CSI_PARSER"), "csi_lemmas": os.environ.get("CSI_MORPH")}
    # (Q)
    pred = []
    for r in Q:
        c = compare(r["a"], r["b"])
        pred.append(-1 if c["equivalent"] is None else int(c["equivalent"]))
        r["pred"], r["differences"] = pred[-1], [d["type"] for d in c["differences"]]
    y = [r["label"] for r in Q]
    scored = [p if p != -1 else 1 - t for p, t in zip(pred, y)]
    out["Q"] = {"n": len(Q), "balanced_accuracy": round(balanced_acc(y, scored), 4),
                "per_phenomenon": per_phenomenon(Q, scored), "errors": [r for r, s, t in zip(Q, scored, y) if s != t]}
    # (R) a message counts only if today's routing reads it as naming its target tool (decided before CSI runs)
    kept, dropped = [], []
    for r in R:
        today = set(sem.asked_caps(r["text"]))
        if r["golds"][0][0] in today:
            r["golds"] = [g for g in r["golds"] if g[0] in today]
            kept.append(r)
        else:
            dropped.append(r)
    items = []
    for r in kept:
        today = set(sem.asked_caps(r["text"]))
        csi_decl = sem.declined_caps(r["text"], read(r["text"]))
        kw_decl = keyword_declined(r["text"], sem)
        for cap, gold in r["golds"]:
            items.append({"text": r["text"], "form": r["form"], "cap": cap, "gold": gold, "scope": r["scope"] and gold == "asked",
                          "verbal_negation": r["verbal_negation"] and gold == "declined", "today": cap in today,
                          "csi": cap in today and cap not in csi_decl, "keyword": cap in today and cap not in kw_decl})

    def rate(rows, key):
        return round(sum(x[key] for x in rows) / len(rows), 4) if rows else None

    by_form = {}
    for x in items:
        by_form.setdefault(f"{x['form']} [{x['gold']}]", []).append(x)
    declining = [x for x in items if x["verbal_negation"]]
    asking = [x for x in items if x["gold"] == "asked"]
    scope = [x for x in items if x["scope"]]
    no_verb = [x for x in items if x["gold"] == "declined" and not x["verbal_negation"]]
    out["R"] = {"messages_kept": len(kept), "messages_dropped_not_asked_today": [r["text"] for r in dropped],
                "items": len(items),
                "declined_offered": {k: rate(declining, k) for k in ("today", "csi", "keyword")},
                "asked_offered": {k: rate(asking, k) for k in ("today", "csi", "keyword")},
                "scope_offered": {k: rate(scope, k) for k in ("today", "csi", "keyword")},
                "no_verbal_negation_declined_offered": {k: rate(no_verb, k) for k in ("today", "csi", "keyword")},
                "by_form": {f: {"n": len(v), **{k: rate(v, k) for k in ("today", "csi", "keyword")}} for f, v in sorted(by_form.items())},
                "csi_errors": [x for x in items if x["csi"] != (x["gold"] == "asked") and (x["gold"] == "asked" or x["verbal_negation"])]}
    # (C)
    for r in Cs:
        r["core"] = int(sem._claims(r["text"]))
        r["csi"] = int(sem._claims(r["text"], read=read))
    out["C"] = {"n": len(Cs), "accuracy": {k: round(sum(r[k] == r["gold"] for r in Cs) / len(Cs), 4) for k in ("core", "csi")},
                "by_form": {f: {k: round(sum(r[k] == r["gold"] for r in Cs if r["form"] == f) / 14, 4) for k in ("core", "csi")}
                            for _, f, _ in CLAIM_TEMPLATES},
                "csi_errors": [r for r in Cs if r["csi"] != r["gold"]], "core_errors": [r for r in Cs if r["core"] != r["gold"]]}
    # (S1)
    s1 = {}
    for name in ("heldout3", "nonce2"):
        rows = generate(name)
        p = []
        for r in rows:
            c = compare(r["a"], r["b"])
            p.append(-1 if c["equivalent"] is None else int(c["equivalent"]))
        yy = [r["label"] for r in rows]
        s1[name] = round(balanced_acc(yy, [q if q != -1 else 1 - t for q, t in zip(p, yy)]), 4)
    out["S1"] = s1
    # (S2)
    obs = []
    for path in a.observed:
        for line in open(path, encoding="utf-8"):
            t = json.loads(line)["prompt"]
            if not t.strip():
                continue
            an = read(t)
            today, decl = sem.asked_caps(t), sem.declined_caps(t, an)
            obs.append({"text": t, "today": today, "csi": [c for c in today if c not in decl], "declined": sorted(decl),
                        "read": bool(an["readings"][0]["propositions"]) and not an["unsupported"],
                        "propositions": [(p["predicate"]["lemma"], p["mood"], p["polarity"]) for p in an["readings"][0]["propositions"]]})
    out["S2"] = {"messages": len(obs), "read": sum(o["read"] for o in obs), "route_changes": [o for o in obs if o["today"] != o["csi"]],
                 "rows": obs}
    out["latency_s"] = {"mean": round(sum(times) / len(times), 4), "p95": round(sorted(times)[int(0.95 * (len(times) - 1))], 4),
                        "n": len(times)} if times else None
    ph = out["Q"]["per_phenomenon"]
    out["criteria"] = {
        "Q1": out["Q"]["balanced_accuracy"] >= 0.95 and all(v >= 0.90 for v in ph.values()),
        "R1": out["R"]["declined_offered"]["csi"] is not None and out["R"]["declined_offered"]["csi"] <= 0.05,
        "R2": out["R"]["asked_offered"]["csi"] >= out["R"]["asked_offered"]["today"] - 0.01,
        "R3": out["R"]["scope_offered"]["csi"] > out["R"]["scope_offered"]["keyword"],
        "C1": out["C"]["accuracy"]["csi"] >= 0.95 and out["C"]["accuracy"]["csi"] >= out["C"]["accuracy"]["core"],
        "S1": s1["heldout3"] >= 0.994 - 0.005 and s1["nonce2"] >= 1.0 - 0.005,
        "S2": all(o["declined"] for o in out["S2"]["route_changes"])}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("criteria", "latency_s")}, indent=1))
    print("Q", out["Q"]["balanced_accuracy"], ph)
    print("R declined", out["R"]["declined_offered"], "asked", out["R"]["asked_offered"], "scope", out["R"]["scope_offered"],
          "no-verb", out["R"]["no_verbal_negation_declined_offered"], "kept", out["R"]["messages_kept"])
    print("C", out["C"]["accuracy"], "| S1", s1, "| S2 read", out["S2"]["read"], "/", out["S2"]["messages"],
          "changes", len(out["S2"]["route_changes"]))


if __name__ == "__main__":
    main()
