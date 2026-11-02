"""Experiment 3c candidate pool: single-word synonym pairs whose shared OEWN synset is the first sense of both words
(verbs from physical/social action lexfiles, nouns from noun.person / noun.artifact), with frequency in UD EWT train,
listed with the shared definition, for choosing (P) and sampling (A).
Usage: python pool_3c.py <out json> --oewn oewn.xml.gz --ewt-train train.conllu"""
import argparse
import collections
import gzip
import json
import re
import xml.etree.ElementTree as ET

LEXFILES = {"v": {"verb.contact", "verb.creation", "verb.change", "verb.possession", "verb.motion", "verb.social"},
            "n": {"noun.person", "noun.artifact"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--oewn", required=True)
    ap.add_argument("--ewt-train", required=True)
    a = ap.parse_args()
    entries, synsets = [], {}
    with gzip.open(a.oewn, "rb") as fh:
        for _, el in ET.iterparse(fh):
            if el.tag == "LexicalEntry":
                lem = el.find("Lemma")
                entries.append((el.get("id"), lem.get("writtenForm") or "", lem.get("partOfSpeech"),
                                [s.get("synset") for s in el.findall("Sense")], [f.get("writtenForm") for f in el.findall("Form")]))
                el.clear()
            elif el.tag == "Synset":
                d = el.find("Definition")
                synsets[el.get("id")] = (el.get("lexfile"), d.text if d is not None else "")
                el.clear()
    senses, forms = collections.defaultdict(list), {}
    for _, lemma, pos, syns, fs in sorted(entries, key=lambda e: (e[2] == "s", e[1] != e[1].lower(), e[0])):
        if pos not in ("n", "v") or not re.fullmatch(r"[a-z]+", lemma):
            continue
        senses[(lemma, pos)] += [s for s in syns if s not in senses[(lemma, pos)]]
        if pos == "v" and fs:
            forms[lemma] = fs
    counts = collections.Counter()
    for line in open(a.ewt_train, encoding="utf-8"):
        c = line.split("\t")
        if len(c) > 4 and c[0].isdigit() and c[3] in ("VERB", "NOUN"):
            counts[(c[2].lower(), "v" if c[3] == "VERB" else "n")] += 1
    first = collections.defaultdict(list)  # first-sense synset -> lemmas whose first sense it is
    for (lemma, pos), syns in senses.items():
        if syns and synsets[syns[0]][0] in LEXFILES[pos]:
            first[(syns[0], pos)].append(lemma)
    pool = []
    for (sid, pos), lemmas in sorted(first.items()):
        lemmas = sorted(lemmas)
        for i, x in enumerate(lemmas):
            for y in lemmas[i + 1:]:
                pool.append({"pos": pos, "a": x, "b": y, "synset": sid, "lexfile": synsets[sid][0],
                             "definition": synsets[sid][1], "count_a": counts[(x, pos)], "count_b": counts[(y, pos)],
                             "irregular_forms": {w: forms.get(w, []) for w in (x, y)} if pos == "v" else None})
    json.dump({"pool": pool, "senses_first3": {f"{l}|{p}": [(s, synsets[s][1][:60]) for s in v[:3]]
                                               for (l, p), v in senses.items() if counts[(l, p)] >= 5}},
              open(a.out, "w"))
    common = [p for p in pool if min(p["count_a"], p["count_b"]) >= 5]
    print("pool", len(pool), "with both words >= 5 in EWT train:", len(common))
    for p in sorted(common, key=lambda p: (p["pos"], p["lexfile"], -min(p["count_a"], p["count_b"]))):
        print(p["pos"], p["lexfile"], f"{p['a']}/{p['b']}", min(p["count_a"], p["count_b"]), "|", p["definition"][:70])


if __name__ == "__main__":
    main()
