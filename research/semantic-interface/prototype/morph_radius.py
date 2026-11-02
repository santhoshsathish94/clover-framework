"""Experiment 4c failure analysis (post-hoc, labelled): how far does the learned verb lemmatiser generalise?
Hold out 20% of ending families defined by the last k letters (k = 2..5), train on the rest, test on the held-out
families; also categorise the C2 (k = 3) errors. Usage: python morph_radius.py <out json> <unimorph eng>"""
import collections
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morph_v2  # noqa: E402
from csi.morphology import Morphology  # noqa: E402


def category(form, lemma, pred):
    if pred == lemma:
        return "correct"
    if pred + "e" == lemma or pred == lemma + "e":
        return "silent e (added or missing)"
    if pred[:-1] == lemma and pred[-1] == lemma[-1] or lemma[:-1] == pred and lemma[-1] == pred[-1]:
        return "consonant doubling"
    if lemma.endswith("y") or pred.endswith("y"):
        return "y / ie"
    return "other"


def main():
    out_path, unimorph = sys.argv[1], sys.argv[2]
    morph_v2.setup(types.SimpleNamespace(unimorph=unimorph, ewt_train=None))
    verbs4 = morph_v2.DATA["verbs4"]
    res = {"experiment": "E4c post-hoc generalisation radius", "radius": {}}
    for k in (2, 3, 4, 5):
        held = set(morph_v2.folds(sorted({l[-k:] for l in verbs4}), 5, 13)[0])
        test = [l for l in verbs4 if l[-k:] in held]
        train = [l for l in verbs4 if l[-k:] not in held]
        m = Morphology.train(morph_v2.verb_entries(train), classes=("verb",), generate=False)
        cats = collections.Counter()
        n = ok = 0
        for l, f, _, t in morph_v2.verb_entries(test):
            p = m.analyse(f, "verb", use_exceptions=False)
            n += 1
            ok += p == l
            cats[category(f, l, p)] += 1
        res["radius"][str(k)] = {"held_out_families": len(held), "n_test_lemmas": len(test), "accuracy": round(ok / n, 4),
                                 "error_categories": {c: v for c, v in cats.most_common() if c != "correct"}}
        print(k, json.dumps(res["radius"][str(k)]), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
