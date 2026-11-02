"""Experiment 3c lexicon (as amended): pairs whose shared OEWN synset is the first sense of both words. (P) chosen by
hand from that pool for template fit, each checked here (shared first sense; control shares no synset; no word from an
earlier lexicon), failures dropped before anything runs; (A) sampled from the pool by seed 3105, no hand selection.
Usage: python lexicon_3c.py <out json> --pool e3c_pool.json --oewn oewn.xml.gz --ewt-train train.conllu"""
import argparse
import collections
import gzip
import json
import os
import random
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi import minimal_pairs as mp  # noqa: E402

# base-sentence verb a, variant verb b, control c (forms: base, past, past participle, gerund), patient
P_VERBS = [
    (("finish", "finished", "finished", "finishing"), ("complete", "completed", "completed", "completing"),
     ("abandon", "abandoned", "abandoned", "abandoning"), "puzzle"),
    (("get", "got", "gotten", "getting"), ("acquire", "acquired", "acquired", "acquiring"),
     ("damage", "damaged", "damaged", "damaging"), "painting"),
    (("supply", "supplied", "supplied", "supplying"), ("provide", "provided", "provided", "providing"),
     ("refuse", "refused", "refused", "refusing"), "food"),
    (("collect", "collected", "collected", "collecting"), ("accumulate", "accumulated", "accumulated", "accumulating"),
     ("scatter", "scattered", "scattered", "scattering"), "stamps"),
    (("retrieve", "retrieved", "retrieved", "retrieving"), ("recover", "recovered", "recovered", "recovering"),
     ("misplace", "misplaced", "misplaced", "misplacing"), "suitcase"),
    (("build", "built", "built", "building"), ("construct", "constructed", "constructed", "constructing"),
     ("demolish", "demolished", "demolished", "demolishing"), "shed"),
    (("decorate", "decorated", "decorated", "decorating"), ("adorn", "adorned", "adorned", "adorning"),
     ("chop", "chopped", "chopped", "chopping"), "tree"),
    (("publish", "published", "published", "publishing"), ("print", "printed", "printed", "printing"),
     ("censor", "censored", "censored", "censoring"), "book"),
    (("change", "changed", "changed", "changing"), ("alter", "altered", "altered", "altering"),
     ("wear", "wore", "worn", "wearing"), "dress"),
    (("pollute", "polluted", "polluted", "polluting"), ("contaminate", "contaminated", "contaminated", "contaminating"),
     ("purify", "purified", "purified", "purifying"), "river"),
    (("explode", "exploded", "exploded", "exploding"), ("detonate", "detonated", "detonated", "detonating"),
     ("defuse", "defused", "defused", "defusing"), "bomb"),
    (("split", "split", "split", "splitting"), ("divide", "divided", "divided", "dividing"),
     ("bake", "baked", "baked", "baking"), "cake")]
P_ANIMATE_VERBS = [
    (("assist", "assisted", "assisted", "assisting"), ("aid", "aided", "aided", "aiding"),
     ("hinder", "hindered", "hindered", "hindering"), None),
    (("deceive", "deceived", "deceived", "deceiving"), ("delude", "deluded", "deluded", "deluding"),
     ("insult", "insulted", "insulted", "insulting"), None),
    (("supervise", "supervised", "supervised", "supervising"), ("oversee", "oversaw", "overseen", "overseeing"),
     ("replace", "replaced", "replaced", "replacing"), None),
    (("enroll", "enrolled", "enrolled", "enrolling"), ("recruit", "recruited", "recruited", "recruiting"),
     ("dismiss", "dismissed", "dismissed", "dismissing"), None)]
P_AGENTS = [("director", "manager", "pharmacist"), ("dealer", "trader", "dentist"), ("insurgent", "rebel", "librarian"),
            ("provider", "supplier", "astronaut"), ("grandma", "grandmother", "electrician"),
            ("minister", "pastor", "mechanic"), ("author", "writer", "reader"), ("owner", "proprietor", "tenant")]
P_PATIENTS = [("couch", "sofa", "stool"), ("photo", "photograph", "envelope"), ("handbag", "purse", "umbrella"),
              ("airplane", "plane", "helicopter"), ("marionette", "puppet", "drum"), ("cloth", "fabric", "wire"),
              ("cargo", "freight", "crane"), ("detector", "sensor", "battery")]
FILLER_PEOPLE = ["worker", "artist", "neighbor", "visitor", "customer", "engineer", "carpenter", "gardener"]
FILLER_THINGS = ["package", "chest", "tray", "carton"]
NEUTRAL_VERBS = [("move", "moved", "moved", "moving"), ("notice", "noticed", "noticed", "noticing")]


def regular_forms(lemma, irregular):
    """Test-generator forms for sampled verbs: OEWN's listed irregular forms, else regular spelling."""
    if irregular:
        past = next((f for f in irregular if not f.endswith("ing") and not f.endswith("s")), None)
        if past:
            pp = next((f for f in irregular if f != past and not f.endswith("ing") and not f.endswith("s")), past)
            return (lemma, past, pp, lemma + "ing")
    if lemma.endswith("e"):
        return (lemma, lemma + "d", lemma + "d", lemma[:-1] + "ing")
    if lemma.endswith("y") and lemma[-2] not in "aeiou":
        return (lemma, lemma[:-1] + "ied", lemma[:-1] + "ied", lemma + "ing")
    return (lemma, lemma + "ed", lemma + "ed", lemma + "ing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--oewn", required=True)
    ap.add_argument("--ewt-train", required=True)
    a = ap.parse_args()
    senses = collections.defaultdict(list)
    with gzip.open(a.oewn, "rb") as fh:
        entries = []
        for _, el in ET.iterparse(fh):
            if el.tag == "LexicalEntry":
                lem = el.find("Lemma")
                entries.append((el.get("id"), lem.get("writtenForm") or "", lem.get("partOfSpeech"),
                                [s.get("synset") for s in el.findall("Sense")]))
                el.clear()
    for _, lemma, pos, syns in sorted(entries, key=lambda e: (e[2] == "s", e[1] != e[1].lower(), e[0])):
        key = (lemma.lower(), "a" if pos == "s" else pos)
        senses[key] += [s for s in syns if s not in senses[key]]
    used = set()
    for table in (mp.TRANSITIVE, mp.ANIMATE_VERBS, mp.DITRANSITIVE):
        for rows in table.values():
            used |= {r[0] for r in rows}
    for table in (mp.ANIMATE, mp.THINGS):
        for rows in table.values():
            used |= set(rows)
    fillers = set(FILLER_PEOPLE) | set(FILLER_THINGS) | {v[0] for v in NEUTRAL_VERBS}
    dropped, P = [], {"verbs": [], "animate_verbs": [], "agents": [], "patients": []}

    def check(x, y, control, pos):
        sx, sy, sc = senses[(x, pos)], senses[(y, pos)], senses[(control, pos)]
        why = []
        if not (sx and sy and sx[0] == sy[0]):
            why.append("shared synset is not the first sense of both")
        if not sc or set(sc) & set(sx):
            why.append("control shares a synset with the first word or is not in OEWN")
        if {x, y, control} & (used | fillers):
            why.append("word from an earlier lexicon or a filler")
        return why

    for key, rows, pos in (("verbs", P_VERBS, "v"), ("animate_verbs", P_ANIMATE_VERBS, "v")):
        for va, vb, vc, patient in rows:
            why = check(va[0], vb[0], vc[0], pos)
            if why:
                dropped.append({"set": key, "pair": [va[0], vb[0], vc[0]], "why": why})
                continue
            P[key].append({"a": va, "b": vb, "control": vc, "patient": patient, "shared": senses[(va[0], pos)][0]})
    for key, rows in (("agents", P_AGENTS), ("patients", P_PATIENTS)):
        for x, y, control in rows:
            why = check(x, y, control, "n")
            if why:
                dropped.append({"set": key, "pair": [x, y, control], "why": why})
                continue
            P[key].append({"a": x, "b": y, "control": control, "shared": senses[(x, "n")][0]})
    pool = json.load(open(a.pool))["pool"]
    counts = collections.Counter()
    for line in open(a.ewt_train, encoding="utf-8"):
        c = line.split("\t")
        if len(c) > 4 and c[0].isdigit() and c[3] in ("VERB", "NOUN"):
            counts[(c[2].lower(), "v" if c[3] == "VERB" else "n")] += 1
    taken = used | fillers | {w for k in P for it in P[k] for w in
                              ((it["a"][0], it["b"][0], it["control"][0]) if k.endswith("verbs") else (it["a"], it["b"], it["control"]))}
    rng, A = random.Random(3105), {"verbs": [], "agents": [], "patients": []}
    groups = (("verbs", "v", None, 12), ("agents", "n", "noun.person", 8), ("patients", "n", "noun.artifact", 8))
    for key, pos, lexfile, want in groups:
        cands = [p for p in pool if p["pos"] == pos and (lexfile is None or p["lexfile"] == lexfile)
                 and min(p["count_a"], p["count_b"]) >= 20]
        rng.shuffle(cands)
        controls = sorted(w for (w, p), n in counts.items() if p == pos and n >= 20 and senses[(w, pos)])
        for p in cands:
            if len(A[key]) >= want:
                break
            if {p["a"], p["b"]} & taken:
                continue
            options = [c for c in controls if c not in taken | {p["a"], p["b"]}
                       and not set(senses[(c, pos)]) & set(senses[(p["a"], pos)])]
            c = rng.choice(options)
            item = {"a": p["a"], "b": p["b"], "control": c, "shared": p["synset"]}
            if pos == "v":
                irr = {w: (p["irregular_forms"] or {}).get(w, []) for w in (p["a"], p["b"])}
                item.update(a=regular_forms(p["a"], irr[p["a"]]), b=regular_forms(p["b"], irr[p["b"]]),
                            control=regular_forms(c, []), patient=None)
            A[key].append(item)
            taken |= {p["a"], p["b"], c}
    out = {"P": P, "A": A, "dropped_from_P": dropped,
           "fillers": {"people": FILLER_PEOPLE, "things": FILLER_THINGS, "neutral_verbs": NEUTRAL_VERBS}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"P": {k: len(v) for k, v in P.items()}, "A": {k: len(v) for k, v in A.items()}, "dropped": dropped}, indent=1))
    for k, v in A.items():
        print("A", k, [(it["a"] if isinstance(it["a"], str) else it["a"][0], it["b"] if isinstance(it["b"], str) else it["b"][0],
                        it["control"] if isinstance(it["control"], str) else it["control"][0]) for it in v])


if __name__ == "__main__":
    main()
