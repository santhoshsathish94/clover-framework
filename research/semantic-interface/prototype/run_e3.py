"""Experiment 2 (E3) evaluation: learned parsers on UD English-EWT with gold word segmentation, measured on what CSI
uses: POS, morphological features, lemmas, verb-headed predicate-argument relations and negator attachment.
ClearNLP-label systems (spaCy en_core_web_*) are mapped to UD for the measured relations only; UAS/LAS are reported for
UD-native systems only. Model choice for Experiment 1b uses dev; test is reported once.
Usage: python run_e3.py <out json> --conllu dev=<path> test=<path> --systems en_core_web_sm <model dir> ... [--limit N]"""
import argparse
import collections
import json
import time

CSI_RELS = ("nsubj", "nsubj:pass", "obj", "iobj", "obl:agent", "aux", "aux:pass", "xcomp", "negator")
FEATS = ("Tense", "VerbForm", "Mood", "Voice", "Polarity", "Number", "Person")
CLEAR_TO_UD = {"nsubj": "nsubj", "nsubjpass": "nsubj:pass", "dobj": "obj", "aux": "aux", "auxpass": "aux:pass",
               "xcomp": "xcomp", "csubj": "csubj", "csubjpass": "csubj:pass", "neg": "advmod", "advmod": "advmod"}
NP_POS = {"NOUN", "PROPN", "PRON", "NUM", "DET", "ADJ"}


def read_conllu(path):
    sents, cur, mwt_end, mwt_space = [], [], -1, True
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if cur:
                sents.append(cur)
            cur, mwt_end = [], -1
            continue
        if line.startswith("#"):
            continue
        c = line.split("\t")
        if "." in c[0]:
            continue  # empty nodes (enhanced graph only)
        if "-" in c[0]:
            mwt_end, mwt_space = int(c[0].split("-")[1]), "SpaceAfter=No" not in c[9]
            continue
        i = int(c[0])
        space = ("SpaceAfter=No" not in c[9]) if i > mwt_end else (mwt_space if i == mwt_end else False)
        feats = dict(kv.split("=", 1) for kv in c[5].split("|")) if c[5] != "_" else {}
        cur.append({"form": c[1], "lemma": c[2], "upos": c[3], "xpos": c[4], "feats": feats,
                    "head": int(c[6]) - 1, "deprel": c[7], "space": space})
    if cur:
        sents.append(cur)
    return sents


def predicted(doc, ud_native):
    """Per token [head (-1 = root), UD label or None] for the measured relations."""
    out = []
    for t in doc:
        head = -1 if t.head.i == t.i else t.head.i
        lab = t.dep_
        if ud_native:
            out.append([head, "root" if lab.lower() == "root" else lab])
        else:
            out.append([head, CLEAR_TO_UD.get(lab)])
    if not ud_native:
        for t in doc:
            if t.dep_ == "dative" and t.pos_ in NP_POS:
                out[t.i][1] = "iobj"
            if t.dep_ == "agent":
                for c in t.children:
                    if c.dep_ == "pobj":
                        out[c.i] = [-1 if t.head.i == t.i else t.head.i, "obl:agent"]
    return out


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "n_gold": n_gold, "n_pred": n_pred}


def score(sents, docs, ud_native):
    n = upos = xpos = lemma = 0
    uas = las = las_u = 0
    fe = {f: collections.Counter() for f in FEATS}
    gold_rel, pred_rel = collections.defaultdict(set), collections.defaultdict(set)
    dobj = collections.Counter()
    for si, (s, doc) in enumerate(zip(sents, docs)):
        assert len(s) == len(doc), (si, [w["form"] for w in s], [t.text for t in doc])
        pr = predicted(doc, ud_native)
        verb = [w["upos"] == "VERB" for w in s]
        for i, (w, t) in enumerate(zip(s, doc)):
            n += 1
            upos += t.pos_ == w["upos"]
            xpos += t.tag_ == w["xpos"]
            lemma += t.lemma_.lower() == w["lemma"].lower()
            for f in FEATS:
                g, p = w["feats"].get(f), ",".join(t.morph.get(f)) or None
                if g is not None:
                    fe[f]["gold"] += 1
                    fe[f]["correct"] += p == g
                else:
                    fe[f]["absent"] += 1
                    fe[f]["false_pos"] += p is not None
            ph, pl = pr[i]
            if ud_native:
                uas += ph == w["head"]
                las += ph == w["head"] and (pl or "").lower() == w["deprel"].lower()
                las_u += ph == w["head"] and (pl or "").split(":")[0].lower() == w["deprel"].split(":")[0].lower()
            neg = w["feats"].get("Polarity") == "Neg" and w["deprel"] == "advmod"
            if w["head"] >= 0 and verb[w["head"]]:
                if w["deprel"] in CSI_RELS:
                    gold_rel[w["deprel"]].add((si, i, w["head"]))
                if neg:
                    gold_rel["negator"].add((si, i, w["head"]))
            if ph >= 0 and verb[ph] and pl:
                if pl in CSI_RELS:
                    pred_rel[pl].add((si, i, ph))
                if neg and pl == "advmod":
                    pred_rel["negator"].add((si, i, ph))
        for i, w in enumerate(s):  # double-object sentences: is the theme found as obj of the same verb?
            if w["deprel"] == "iobj" and w["head"] >= 0 and verb[w["head"]]:
                dobj["iobj_gold"] += 1
                dobj["iobj_found"] += (si, i, w["head"]) in pred_rel["iobj"]
                for j, x in enumerate(s):
                    if x["deprel"] == "obj" and x["head"] == w["head"]:
                        dobj["theme_gold"] += 1
                        dobj["theme_found"] += (si, j, w["head"]) in pred_rel["obj"]
    rels = {r: prf(len(gold_rel[r] & pred_rel[r]), len(pred_rel[r]), len(gold_rel[r])) for r in CSI_RELS}
    res = {"words": n, "upos": round(upos / n, 4), "xpos": round(xpos / n, 4), "lemma_ci": round(lemma / n, 4),
           "features": {f: {"acc": round(c["correct"] / c["gold"], 4) if c["gold"] else None, "n_gold": c["gold"],
                            "false_pos_rate": round(c["false_pos"] / c["absent"], 4) if c["absent"] else None}
                        for f, c in fe.items()},
           "relations": rels, "csi_macro_f1": round(sum(v["f1"] for v in rels.values()) / len(rels), 4),
           "double_object": dict(dobj)}
    if ud_native:
        res.update({"uas": round(uas / n, 4), "las": round(las / n, 4), "las_universal": round(las_u / n, 4)})
    return res


def main():
    import spacy
    from spacy.tokens import Doc
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--conllu", nargs="+", required=True, help="split=path")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    splits = {kv.split("=", 1)[0]: read_conllu(kv.split("=", 1)[1]) for kv in a.conllu}
    if a.limit:
        splits = {k: v[:a.limit] for k, v in splits.items()}
    res = {"experiment": "E3 parsers on UD English-EWT (gold words)", "sizes": {k: len(v) for k, v in splits.items()},
           "systems": {}}
    for name in a.systems:
        nlp = spacy.load(name)
        if "ner" in nlp.pipe_names:
            nlp.remove_pipe("ner")
        labels = set(nlp.get_pipe("parser").labels)
        ud_native = "obj" in labels and "dobj" not in labels
        res["systems"][name] = {"pipeline": nlp.pipe_names, "version": nlp.meta.get("version"),
                                "label_scheme": "UD" if ud_native else "ClearNLP (mapped)"}
        for split, sents in splits.items():
            t0 = time.time()
            docs = list(nlp.pipe([Doc(nlp.vocab, words=[w["form"] for w in s], spaces=[w["space"] for w in s])
                                  for s in sents], batch_size=64))
            r = score(sents, docs, ud_native)
            r["seconds"] = round(time.time() - t0, 1)
            res["systems"][name][split] = r
            print(name, split, json.dumps({k: v for k, v in r.items() if k not in ("features", "relations")}),
                  json.dumps({k: v["f1"] for k, v in r["relations"].items()}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
