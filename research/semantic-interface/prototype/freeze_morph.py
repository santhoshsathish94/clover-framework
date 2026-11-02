"""Freeze the CSI verb lemmatiser (Experiment 4d prerequisite): train the 4c Morphology on all UniMorph verbs, save
the artefact, reload it, and check that the reloaded layered lemmatiser reproduces the 4g numbers (fresh GUM verbs,
nonce seed 113). Also prints the nonce1 lexicon for Experiment 4d (seed 20261104).
Usage: python freeze_morph.py <artefact dir> --unimorph eng --oewn oewn.xml.gz --gum dev.conllu test.conllu"""
import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morph_v2 as mv  # noqa: E402
from csi.dictionary import Dictionary, lemmatise  # noqa: E402
from csi.morphology import Morphology, load, save  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--unimorph", required=True)
    ap.add_argument("--oewn", required=True)
    ap.add_argument("--gum", nargs="+", required=True)
    a = ap.parse_args()
    mv.setup(argparse.Namespace(unimorph=a.unimorph, oewn=None, feature_set="suffix", nonce_seed=19))
    dic = Dictionary.load_oewn(a.oewn)
    m = Morphology.train([e for e in mv.DATA["entries"] if e[2] == "verb"], classes=("verb",), generate=False)
    save(m, a.out)
    frozen = load(a.out)
    _, targets = mv.treebank_targets(a.gum)
    verbs = [t for t in targets if t[2] == "verb"]
    acc = {}
    for name, model in (("trained", m), ("frozen", frozen)):
        acc[name] = round(sum(lemmatise(f, "verb", model, dic)[0] == g for *_, f, g in verbs) / len(verbs), 4)
    differ = sum(lemmatise(f, "verb", m, dic)[0] != lemmatise(f, "verb", frozen, dic)[0] for *_, f, _ in verbs)
    nonce = mv.nonce_verbs(seed=113, exclude=frozenset(dic.words()))
    forms = [(l, f) for lem in nonce.values() for l, fs in lem.items() for f in fs.values()]
    nonce_acc = round(sum(lemmatise(f, "verb", frozen, dic)[0] == l for l, f in forms) / len(forms), 4)
    # nonce1 for 4d: excluded from UniMorph, OEWN and the 4c/4f/4g nonce sets
    used = set()
    for seed, excl in ((19, frozenset()), (107, frozenset()), (113, frozenset(dic.words()))):
        for lem in mv.nonce_verbs(seed=seed, exclude=excl).values():
            for l, fs in lem.items():
                used |= {l, *fs.values()}
    pool = mv.nonce_verbs(per_class=3, seed=20261104, exclude=frozenset(dic.words()) | frozenset(used))
    nonce1 = {"transitive": [], "animate": []}
    for cls in mv.NONCE:
        items = sorted(pool[cls].items())
        random.Random(f"20261104:{cls}").shuffle(items)
        for k, (l, fs) in enumerate(items[:3]):
            row = (l, fs["past"], fs["past_participle"], fs["gerund"])
            nonce1["transitive" if k < 2 else "animate"].append(row)
    files = {f: sha256(os.path.join(a.out, f)) for f in sorted(os.listdir(a.out))}
    card = {"artefact": os.path.basename(a.out.rstrip("/")), "files": files,
            "sizes": {f: os.path.getsize(os.path.join(a.out, f)) for f in files},
            "oewn_sha256": sha256(a.oewn), "unimorph_sha256": sha256(a.unimorph),
            "n_training_pairs": sum(1 for e in mv.DATA["entries"] if e[2] == "verb"),
            "n_edit_classes": len(frozen.analysers["verb"].classes), "n_features": len(frozen.analysers["verb"].vocab),
            "gum_verbs": {"n": len(verbs), "layered_trained": acc["trained"], "layered_frozen": acc["frozen"],
                          "predictions_differing": differ},
            "nonce113_layered_frozen": nonce_acc, "nonce1": nonce1}
    json.dump(card, open(os.path.join(os.path.dirname(a.out.rstrip("/")), "morph-verb-v1.card.json"), "w"), indent=1)
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
