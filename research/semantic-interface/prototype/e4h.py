"""Experiment 4h: verb lemmatiser v1.1 (analysis by synthesis). Builds morph-verb-v1.1 from the frozen v1 analyser plus
the 4c verb generators, then H1 (UD EWT train verbs) and H2 (nonce seed 127), and prints the nonce2 lexicon for H3.
Usage: python e4h.py <out json> --frozen <dir with morph-verb-v1> --unimorph eng --oewn oewn.xml.gz
                     --ewt-train train.conllu [--systems en_core_web_trf en_core_web_sm]"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morph_v2 as mv  # noqa: E402
from csi.dictionary import Dictionary, lemmatise  # noqa: E402
from csi.morphology import Morphology, load, save  # noqa: E402
from freeze_morph import sha256  # noqa: E402

TAG_FORM = {"VBD": "past", "VBN": "past_participle", "VBG": "gerund", "VBZ": "third_singular"}


def earlier_nonce(dic):
    used = set()
    for seed, kw in ((19, {}), (107, {}), (113, {"exclude": frozenset(dic.words())})):
        for lem in mv.nonce_verbs(seed=seed, **kw).values():
            for l, fs in lem.items():
                used |= {l, *fs.values()}
    for lem in mv.nonce_verbs(per_class=3, seed=20261104, exclude=frozenset(dic.words()) | frozenset(used)).values():
        for l, fs in lem.items():
            used |= {l, *fs.values()}
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--unimorph", required=True)
    ap.add_argument("--oewn", required=True)
    ap.add_argument("--ewt-train", required=True)
    ap.add_argument("--systems", nargs="*", default=[])
    a = ap.parse_args()
    mv.setup(argparse.Namespace(unimorph=a.unimorph, oewn=None, feature_set="suffix", nonce_seed=19))
    dic = Dictionary.load_oewn(a.oewn)
    v1 = load(os.path.join(a.frozen, "morph-verb-v1"))
    gens = Morphology.train([e for e in mv.DATA["entries"] if e[2] == "verb"], classes=("verb",), generate=True)
    v11 = load(os.path.join(a.frozen, "morph-verb-v1"))
    v11.generators, v11.generate_exceptions = gens.generators, gens.generate_exceptions
    d11 = os.path.join(a.frozen, "morph-verb-v1.1")
    save(v11, d11)
    v11 = load(d11)
    res = {"experiment": "E4h verb lemmatiser v1.1", "artefact": {f: sha256(os.path.join(d11, f)) for f in sorted(os.listdir(d11))}}
    # H1: EWT train verbs
    sents, targets = mv.treebank_targets([a.ewt_train])
    verbs = [(si, i, f, g, TAG_FORM[sents[si][i]["xpos"]]) for si, i, c, f, g in targets if c == "verb"]
    preds = {"v1.1": [lemmatise(f, "verb", v11, dic, form_type=ft)[0] for _, _, f, _, ft in verbs],
             "v1": [lemmatise(f, "verb", v1, dic)[0] for _, _, f, _, _ in verbs],
             "morphology_alone": [v1.analyse(f, "verb", use_exceptions=False) for _, _, f, _, _ in verbs]}
    if a.systems:
        import spacy
        from spacy.tokens import Doc
        for name in a.systems:
            nlp = spacy.load(name)
            docs = list(nlp.pipe([Doc(nlp.vocab, words=[w["form"] for w in s], spaces=[w["space"] for w in s])
                                  for s in sents], batch_size=64))
            preds[name] = [docs[si][i].lemma_.lower() for si, i, *_ in verbs]
    gold = [g for _, _, _, g, _ in verbs]
    res["h1_ewt_train_verbs"] = {"n": len(verbs), "accuracy": {k: round(sum(p == g for p, g in zip(v, gold)) / len(gold), 4)
                                                             for k, v in preds.items()},
                                 "v1_vs_v1.1": collections.Counter(
                                     "v1.1 right, v1 wrong" if p11 == g else "v1 right, v1.1 wrong" if p1 == g else "both wrong"
                                     for p1, p11, g in zip(preds["v1"], preds["v1.1"], gold) if p1 != p11),
                                 "changed_examples": sorted({f"{f}: v1 {p1}, v1.1 {p11} (gold {g})" for (_, _, f, g, _), p1, p11
                                                            in zip(verbs, preds["v1"], preds["v1.1"]) if p1 != p11})[:40]}
    # H2: nonce seed 127
    used = earlier_nonce(dic)
    nonce = mv.nonce_verbs(seed=127, exclude=frozenset(dic.words()) | frozenset(used))
    h2 = {}
    for cls, lem in nonce.items():
        n, ok, errs = 0, collections.Counter(), []
        for l, fs in lem.items():
            for ft, f in fs.items():
                n += 1
                p = {"v1.1": lemmatise(f, "verb", v11, dic, form_type=ft)[0], "v1": lemmatise(f, "verb", v1, dic)[0],
                     "morphology_alone": v1.analyse(f, "verb", use_exceptions=False)}
                ok.update(k for k, v in p.items() if v == l)
                if p["v1.1"] != l and len(errs) < 5:
                    errs.append(f"{f} -> {p['v1.1']} (want {l})")
        if n:
            h2[cls] = {"n_lemmas": len(lem), "n": n, **{k: round(ok[k] / n, 4) for k in ("v1.1", "v1", "morphology_alone")},
                       "errors": errs}
    tot = sum(v["n"] for v in h2.values())
    h2["_all"] = {"n": tot, **{k: round(sum(v[k] * v["n"] for v in h2.values()) / tot, 4) for k in ("v1.1", "v1", "morphology_alone")}}
    res["h2_nonce127"] = h2
    # nonce2 for H3
    for lem in nonce.values():
        for l, fs in lem.items():
            used |= {l, *fs.values()}
    pool = mv.nonce_verbs(per_class=3, seed=20261105, exclude=frozenset(dic.words()) | frozenset(used))
    nonce2 = {"transitive": [], "animate": [], "classes_skipped": []}
    for cls in mv.NONCE:
        items = sorted(pool[cls].items())
        if len(items) < 3:
            nonce2["classes_skipped"].append(f"{cls} ({len(items)} unused)")
            continue
        random.Random(f"20261105:{cls}").shuffle(items)
        for k, (l, fs) in enumerate(items[:3]):
            nonce2["transitive" if k < 2 else "animate"].append((l, fs["past"], fs["past_participle"], fs["gerund"]))
    res["nonce2"] = nonce2
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "h1_ewt_train_verbs"} | {"h1": {k: v for k, v in res["h1_ewt_train_verbs"].items()
                                                                                          if k != "changed_examples"}}, indent=1))


if __name__ == "__main__":
    main()
