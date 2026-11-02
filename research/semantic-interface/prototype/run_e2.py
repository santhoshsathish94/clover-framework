"""Experiment 4 = E2a (morphology): do lemmatisers generalise to verbs they have not seen? UniMorph English verb forms
(CC BY-SA 3.0) placed in fixed sentence frames; lemma accuracy on the verb token for lemmas that occur as verbs in UD
EWT train ("seen") vs lemmas absent from EWT train ("unseen"), per form type. Systems: spaCy rule lemmatisers (sm, trf)
and our EWT-trained edit-tree lemmatiser (3 seeds). Usage:
python run_e2.py <out json> <unimorph eng file> <ewt train conllu> --systems <spacy name or path> ...
"""
import argparse
import collections
import json
import random
import re

FORMS = {  # UniMorph feature set -> (form type, sentence frame)
    frozenset({"V", "PST"}): ("past", "The man {} the box."),
    frozenset({"V", "V.PTCP", "PST"}): ("past_participle", "The box was {} by the man."),
    frozenset({"V", "V.PTCP", "PRS"}): ("gerund", "The man was {} the box."),
    frozenset({"V", "3", "SG", "PRS"}): ("third_singular", "The man {} the box."),
}


def load_unimorph(path):
    verbs = collections.defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) < 3:
            continue
        lemma, form, feats = c[0], c[1], frozenset(c[2].split(";"))
        if feats in FORMS and re.fullmatch(r"[a-z]+", lemma) and re.fullmatch(r"[a-z]+", form):
            verbs[lemma][FORMS[feats][0]] = form
    return {k: v for k, v in verbs.items() if len(v) == len(FORMS)}


def ewt_lemmas(path):
    any_lemma, verb_lemma = set(), set()
    for line in open(path, encoding="utf-8"):
        c = line.split("\t")
        if len(c) == 10 and c[0].isdigit():
            any_lemma.add(c[2].lower())
            any_lemma.add(c[1].lower())
            if c[3] == "VERB":
                verb_lemma.add(c[2].lower())
    return any_lemma, verb_lemma


def main():
    import spacy
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("unimorph")
    ap.add_argument("ewt_train")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--per-split", type=int, default=500)
    a = ap.parse_args()
    verbs = load_unimorph(a.unimorph)
    seen_any, seen_verb = ewt_lemmas(a.ewt_train)
    rng = random.Random(7)
    seen = sorted(l for l in verbs if l in seen_verb)
    unseen = sorted(l for l in verbs if l not in seen_any and all(f not in seen_any for f in verbs[l].values()))
    split = {"seen": rng.sample(seen, min(a.per_split, len(seen))), "unseen": rng.sample(unseen, min(a.per_split, len(unseen)))}
    items = [(sp, l, t, frame.format(verbs[l][t]), verbs[l][t])
             for sp, lemmas in split.items() for l in lemmas for t, frame in FORMS.values()]
    res = {"experiment": "E2a lemmatisation generalisation", "available": {"seen": len(seen), "unseen": len(unseen)},
           "sampled": {k: len(v) for k, v in split.items()}, "systems": {}}
    for name in a.systems:
        nlp = spacy.load(name)
        acc = collections.defaultdict(lambda: [0, 0, 0])
        errors = collections.Counter()
        for (sp, lemma, t, text, form), doc in zip(items, nlp.pipe([x[3] for x in items], batch_size=128)):
            tok = next((x for x in doc if x.text == form), None)
            k = acc[(sp, t)]
            k[0] += 1
            if tok is None:
                continue
            k[1] += tok.lemma_.lower() == lemma
            k[2] += tok.pos_ in ("VERB", "AUX")
            if tok.lemma_.lower() != lemma and sp == "unseen":
                errors[f"{form} -> {tok.lemma_.lower()} (want {lemma})"] += 1
        table = {f"{sp}:{t}": {"n": v[0], "lemma_acc": round(v[1] / v[0], 4), "verb_tag_rate": round(v[2] / v[0], 4)}
                 for (sp, t), v in sorted(acc.items())}
        for sp in split:
            tot = [sum(v[i] for (s, _), v in acc.items() if s == sp) for i in range(3)]
            table[f"{sp}:all"] = {"n": tot[0], "lemma_acc": round(tot[1] / tot[0], 4),
                                  "verb_tag_rate": round(tot[2] / tot[0], 4)}
        res["systems"][name] = {"table": table, "unseen_error_sample": [e for e, _ in errors.most_common(25)]}
        print(name, json.dumps({k: v["lemma_acc"] for k, v in table.items()}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
