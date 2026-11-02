"""Experiment 3c2 lexicon (pre-registered in EXPERIMENTS.md): fresh synonym pairs whose first OEWN sense is the same
synset, template fit from OEWN verb frames and noun lexicographer files, and for each pair a related control (first
sense a co-hyponym of the shared meaning) and a random control; split into dev and test by seed. Nothing is chosen by
hand. Usage: python lexicon_3c2.py <out json> --oewn oewn.xml.gz --ewt-dir UD_English-EWT --lex3c e3c_lexicon.json"""
import argparse
import collections
import functools
import glob
import gzip
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi import minimal_pairs as mp  # noqa: E402

SEED = 3110
PATIENT = {"noun.artifact", "noun.food", "noun.object", "noun.plant", "noun.animal", "noun.substance", "noun.body"}
FRAME = {"verb_thing": "vtai", "verb_person": "vtaa"}  # OEWN: "Somebody ----s something" / "Somebody ----s somebody"
FILLERS = {"people": ["child", "teacher", "student", "doctor", "farmer", "woman", "nurse", "pilot"],
           "things": ["box", "bottle", "bag", "chair"],
           "neutral_verbs": [["see", "saw", "seen"], ["find", "found", "found"]]}


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def variant(a, b):
    """Spelling variants and clippings are not synonym pairs here."""
    return edit_distance(a, b) <= 2 or a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)


def rule_past(v):
    if v.endswith("e"):
        return v + "d"
    if v.endswith("y") and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    if re.fullmatch(r"[^aeiou]*[aeiou][^aeiouwxy]", v):
        return v + v[-1] + "ed"
    return v + "ed"


def verb_forms(v, irregular, past, part):
    """[base, past, past participle] and the source of each: EWT-attested (a regular -ed form attested as either one
    serves for both), else OEWN's listed forms (one ending in -n is the participle), else spelling rules."""
    ewt = {k: max(sorted(c), key=c.get) if c else None for k, c in (("past", past), ("part", part))}
    for k, other in (("past", "part"), ("part", "past")):
        if ewt[k] is None and ewt[other] and ewt[other].endswith("ed"):
            ewt[k] = ewt[other]
    irr = [f for f in irregular if not f.endswith("ing") and f != v + "s"]
    oewn_part = next((f for f in irr if f.endswith("n")), irr[-1] if irr else None)
    oewn_past = next((f for f in irr if f != oewn_part), oewn_part)
    forms, source = [v], []
    for k, oewn in (("past", oewn_past), ("part", oewn_part)):
        if ewt[k]:
            forms.append(ewt[k])
            source.append("ewt")
        elif oewn:
            forms.append(oewn)
            source.append("oewn")
        else:
            forms.append(rule_past(v))
            source.append("rule")
    return forms, source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--oewn", required=True)
    ap.add_argument("--ewt-dir", required=True)
    ap.add_argument("--lex3c", required=True)
    a = ap.parse_args()
    entries, syn = [], {}
    with gzip.open(a.oewn, "rb") as fh:
        for _, el in ET.iterparse(fh):
            if el.tag == "LexicalEntry":
                lem = el.find("Lemma")
                entries.append((el.get("id"), lem.get("writtenForm") or "", lem.get("partOfSpeech"),
                                [(s.get("synset"), (s.get("subcat") or "").split()) for s in el.findall("Sense")],
                                [f.get("writtenForm") for f in el.findall("Form")]))
                el.clear()
            elif el.tag == "Synset":
                d = el.find("Definition")
                syn[el.get("id")] = {"lexfile": el.get("lexfile"), "definition": d.text if d is not None else "",
                                     "hypernyms": [r.get("target") for r in el.findall("SynsetRelation")
                                                   if r.get("relType") == "hypernym"]}
                el.clear()
    # meanings in the dictionary's order (csi/senses.py load_inventory); frames and forms from lower-case entries
    senses, frames, irregular, lower = collections.defaultdict(list), {}, collections.defaultdict(list), set()
    for _, lemma, pos, ss, forms in sorted(entries, key=lambda e: (e[2] == "s", e[1] != e[1].lower(), e[0])):
        if pos not in ("n", "v") or not re.fullmatch(r"[A-Za-z]+", lemma):
            continue
        key = (lemma.lower(), pos)
        for s, sc in ss:
            if s not in senses[key]:
                senses[key].append(s)
            if lemma == lemma.lower():
                frames[(lemma, pos, s)] = set(sc)
        if lemma == lemma.lower():
            lower.add(key)
            if pos == "v":
                irregular[lemma] += [f for f in forms if f not in irregular[lemma]]
    counts = collections.Counter()
    past, part = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    ewt_files = sorted(glob.glob(os.path.join(a.ewt_dir, "*.conllu")))
    for path in ewt_files:
        for line in open(path, encoding="utf-8"):
            c = line.rstrip("\n").split("\t")
            if len(c) > 4 and c[0].isdigit() and c[3] in ("VERB", "NOUN"):
                lem, pos = c[2].lower(), "v" if c[3] == "VERB" else "n"
                counts[(lem, pos)] += 1
                if pos == "v" and c[4] in ("VBD", "VBN"):
                    (past if c[4] == "VBD" else part)[lem][c[1].lower()] += 1
    used = set()
    for table in (mp.TRANSITIVE, mp.ANIMATE_VERBS, mp.DITRANSITIVE):
        for rows in table.values():
            used |= {r[0] for r in rows}
    for table in (mp.ANIMATE, mp.THINGS):
        for rows in table.values():
            used |= set(rows)
    lex3c = json.load(open(a.lex3c))
    for split in ("P", "A"):
        for its in lex3c[split].values():
            for it in its:
                used |= {w if isinstance(w, str) else w[0] for w in (it["a"], it["b"], it["control"])}
    for f in (lex3c["fillers"], FILLERS):
        used |= set(f["people"]) | set(f["things"]) | {v[0] for v in f["neutral_verbs"]}
    hypo = collections.defaultdict(set)
    for s, d in syn.items():
        for h in d["hypernyms"]:
            hypo[h].add(s)

    def closure(s, step):
        seen, todo = set(), [s]
        while todo:
            for t in step(todo.pop()):
                if t not in seen:
                    seen.add(t)
                    todo.append(t)
        return seen

    @functools.lru_cache(maxsize=None)
    def near(s):
        """-> (co-hyponyms, every synset a control must avoid as its first sense)."""
        family = closure(s, lambda x: syn[x]["hypernyms"]) | closure(s, lambda x: hypo[x]) | {s}
        sisters = {t for h in syn[s]["hypernyms"] for t in hypo[h]} - family
        return sisters, family | sisters

    def fits(w, pos, k):
        s = senses[(w, pos)][0]
        if pos == "v":
            return FRAME[k] in frames.get((w, "v", s), set())
        lf = syn[s]["lexfile"]
        return lf == "noun.person" if k == "agent" else lf in PATIENT

    fresh = sorted(k for k in lower if counts[k] >= 1 and k[0] not in used and senses[k])
    first = collections.defaultdict(list)
    for w, pos in fresh:
        first[(senses[(w, pos)][0], pos)].append(w)
    pairs = []
    for (s, pos), ws in sorted(first.items()):
        for i, x in enumerate(ws):
            for y in ws[i + 1:]:
                if variant(x, y):
                    continue
                if pos == "n":
                    k = "agent" if fits(x, "n", "agent") else "patient" if fits(x, "n", "patient") else None
                else:
                    k = next((k for k in FRAME if fits(x, "v", k) and fits(y, "v", k)), None)
                if k:
                    pairs.append((x, y, s, pos, k))
    classes = ("verb_thing", "verb_person", "agent", "patient")
    cand = {k: [w for w, pos in fresh if pos == ("v" if k in FRAME else "n") and fits(w, pos, k)] for k in classes}
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    taken, items, skipped = set(), [], collections.Counter()
    for x, y, s, pos, k in pairs:
        if {x, y} & taken:
            continue
        theirs, blocked = set(senses[(x, pos)]) | set(senses[(y, pos)]), taken | {x, y}
        sisters, avoid = near(s)
        free = [w for w in cand[k] if w not in blocked and not set(senses[(w, pos)]) & theirs]
        rel = [w for w in free if senses[(w, pos)][0] in sisters]
        rnd = [w for w in free if senses[(w, pos)][0] not in avoid]
        if not rel or not rnd:
            skipped[k] += 1
            continue
        top = max(counts[(w, pos)] for w in rel)
        r = rng.choice([w for w in rel if counts[(w, pos)] == top])
        q = rng.choice(rnd)
        it = {"class": k, "pos": pos, "shared": s, "shared_definition": syn[s]["definition"], "a": x, "b": y,
              "related": r, "related_synset": senses[(r, pos)][0], "random": q, "random_synset": senses[(q, pos)][0],
              "ewt_counts": {w: counts[(w, pos)] for w in (x, y, r, q)}}
        if pos == "v":
            it["forms"], it["form_source"] = {}, {}
            for w in (x, y, r, q):
                it["forms"][w], it["form_source"][w] = verb_forms(w, irregular[w], past[w], part[w])
        items.append(it)
        taken |= {x, y, r, q}
    by = collections.defaultdict(list)
    for it in items:
        by[it["class"]].append(it)
    for k in classes:
        rng.shuffle(by[k])
        n_dev = round(len(by[k]) / 3)
        for j, it in enumerate(by[k]):
            it["split"], it["id"] = ("dev" if j < n_dev else "test"), f"{k}-{j:02d}"
    items = [it for k in classes for it in by[k]]
    summary = {"pairs_in_pool": dict(collections.Counter(p[4] for p in pairs)), "skipped_without_controls": dict(skipped),
               "items": {k: {s: sum(it["split"] == s for it in by[k]) for s in ("dev", "test")} for k in classes},
               "excluded_words": len(used), "ewt_files": [os.path.basename(p) for p in ewt_files]}
    json.dump({"seed": SEED, "fillers": FILLERS, "summary": summary, "items": items}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for it in items:
        print(it["id"], it["split"], it["a"], it["b"], "| related", it["related"], "| random", it["random"],
              it.get("form_source", ""))


if __name__ == "__main__":
    main()
