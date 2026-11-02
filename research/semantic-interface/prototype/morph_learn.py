"""Experiment 4b (morphology, learned): learn English verb lemmatisation from UniMorph spelling alone and test on the
Experiment 4a lemmas (held out by lemma). The model predicts an edit (strip k final letters, append s) from the
form's final letters (logistic regression over suffix features); no word list, no context.
Usage: python morph_learn.py <out json> <unimorph eng file> <ewt train conllu> [--exclude-words w1,w2,...]"""
import argparse
import collections
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e2 import FORMS, ewt_lemmas, load_unimorph  # noqa: E402


def edit(form, lemma):
    k = 0
    while k < min(len(form), len(lemma)) and form[k] == lemma[k]:
        k += 1
    return (len(form) - k, lemma[k:])


def feats(form):
    f = {f"s{n}={form[-n:]}": 1 for n in range(1, 7) if len(form) >= n}
    f["len"] = min(len(form), 12) / 12.0
    return f


def apply(form, e):
    k, s = e
    return (form[:-k] if k else form) + s


def main():
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("unimorph")
    ap.add_argument("ewt_train")
    ap.add_argument("--per-split", type=int, default=500)
    ap.add_argument("--exclude-words", default="")
    a = ap.parse_args()
    verbs = load_unimorph(a.unimorph)
    seen_any, seen_verb = ewt_lemmas(a.ewt_train)
    rng = random.Random(7)  # identical sampling to run_e2.py -> identical test lemmas
    seen = sorted(l for l in verbs if l in seen_verb)
    unseen = sorted(l for l in verbs if l not in seen_any and all(f not in seen_any for f in verbs[l].values()))
    split = {"seen": rng.sample(seen, min(a.per_split, len(seen))), "unseen": rng.sample(unseen, min(a.per_split, len(unseen)))}
    held = set(split["seen"]) | set(split["unseen"]) | set(w for w in a.exclude_words.split(",") if w)
    train = [(f, l) for l, forms in verbs.items() if l not in held for f in forms.values()]
    edits = [edit(f, l) for f, l in train]
    keep = {e for e, c in collections.Counter(edits).items() if c >= 2}  # one-off irregular edits cannot generalise
    X = [feats(f) for (f, _), e in zip(train, edits) if e in keep]
    y = [f"{e[0]}|{e[1]}" for e in edits if e in keep]
    vec = DictVectorizer()
    clf = LogisticRegression(C=10.0, max_iter=3000)
    clf.fit(vec.fit_transform(X), y)
    res = {"experiment": "E4b learned lemmatiser from UniMorph spelling", "train_pairs": len(y), "edit_classes": len(set(y)),
           "held_out_lemmas": len(held), "table": {}, "unseen_error_sample": []}
    acc = collections.defaultdict(lambda: [0, 0])
    errs = []
    for sp, lemmas in split.items():
        for l in lemmas:
            for t, _ in FORMS.values():
                form = verbs[l][t]
                k, s = clf.predict(vec.transform([feats(form)]))[0].split("|")
                pred = apply(form, (int(k), s))
                acc[(sp, t)][0] += 1
                acc[(sp, t)][1] += pred == l
                if pred != l and sp == "unseen" and len(errs) < 25:
                    errs.append(f"{form} -> {pred} (want {l})")
    for (sp, t), (n, c) in sorted(acc.items()):
        res["table"][f"{sp}:{t}"] = round(c / n, 4)
    for sp in split:
        n = sum(v[0] for (s, _), v in acc.items() if s == sp)
        c = sum(v[1] for (s, _), v in acc.items() if s == sp)
        res["table"][f"{sp}:all"] = round(c / n, 4)
    res["unseen_error_sample"] = errs
    print(json.dumps(res["table"]), "\n", errs[:12])
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
