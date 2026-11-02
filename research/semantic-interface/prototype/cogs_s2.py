"""Experiment 6b S2 (learned linker): a frozen parser (spaCy en_core_web_trf, not trained on COGS) gives structure; a
multinomial logistic regression learns from COGS train which (verb, noun) pairs are agent / theme / recipient / none.
Regularisation chosen on dev; test and gen scored once with the Experiment 6a metric (main event; CSI's predicate
choice incl. rule 7 embedding) and an index-based variant (predicate by token position, not lemma).
Usage: CSI_PARSER=en_core_web_trf python cogs_s2.py <out json> <COGS data dir> [--limit N]"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi.analyze import SPACY_MODEL  # noqa: E402
from run_cogs import BINARY, ROLE_TO_ARG, UNARY  # noqa: E402

NOMINAL = {"NOUN", "PROPN", "PRON"}
ROLES = ("agent", "theme", "recipient")


def load(d, split, limit=0):
    rows = []
    for line in open(os.path.join(d, f"{split}.tsv"), encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if "LAMBDA" in c[1] or len(c[0].split()) < 3:
            continue  # primitives: no sentence structure to link
        rows.append((c[0], c[1], c[2] if len(c) > 2 else "in_distribution"))
    return rows[:limit] if limit else rows


def gold_events(sent, lf):
    """{verb token index: {"verb": lemma, "roles": {token index: role}, "xcomp": idx}}; main event index."""
    words = sent.split()
    events, links = {}, []
    for c in [c.strip().lstrip("* ").strip() for part in lf.split(" ; ") for c in part.split(" AND ")]:
        if UNARY.match(c):
            continue
        m = BINARY.match(c)
        if not m:
            raise ValueError(c)
        pred, role, e = m.group(1), m.group(2), int(m.group(3))
        if role.startswith("nmod"):
            continue
        ev = events.setdefault(e, {"verb": pred, "roles": {}})
        if role in ("ccomp", "xcomp"):
            ev[role] = int(m.group(4))
            links.append(int(m.group(4)))
        elif role in ROLE_TO_ARG:
            idx = int(m.group(4)) if m.group(4) is not None else words.index(m.group(5))
            ev["roles"][idx] = role
    mains = [e for e in events if e not in links]
    return events, (mains[0] if len(mains) == 1 else None)


def path(doc, n, v):
    """Dependency path n -> v: labels up to the lowest common ancestor, then down; prepositions named."""
    def up(t):
        chain = [t]
        while chain[-1].head.i != chain[-1].i:
            chain.append(chain[-1].head)
        return chain
    un, uv = up(doc[n]), up(doc[v])
    common = next((t for t in un if t in uv), None)
    if common is None:
        return "none"
    a = un[:un.index(common)]
    b = uv[:uv.index(common)]
    if len(a) + len(b) > 4:
        return "far"
    lab = lambda t: t.dep_ + (f"[{t.lemma_.lower()}]" if t.dep_ in ("prep", "agent", "dative") else "")
    return ">".join([lab(t) + "^" for t in a] + ["v" + lab(t) for t in reversed(b)])


def features(doc, v, n, version=1):
    vt = doc[v]
    kids = [c.dep_ for c in vt.children]
    voice = "passive" if ("auxpass" in kids or "nsubjpass" in kids) else "active"
    p = path(doc, n, v)
    f = {"path=" + p: 1, "path_voice=" + p + "|" + voice: 1, "voice=" + voice: 1,
         "before=" + str(n < v): 1, "two_obj=" + str(kids.count("dobj") + kids.count("dative") >= 2): 1,
         "verb=" + vt.lemma_.lower(): 1, "verb_path=" + vt.lemma_.lower() + "|" + p.split(">")[0]: 1}
    if version >= 2:  # post-hoc (Experiment 6b-S2b): passive voice x whether an object remains on the verb
        objs = str(min(kids.count("dobj") + kids.count("dative"), 2))
        f["voice_obj=" + voice + "|" + objs] = 1
        f["path_voice_obj=" + p + "|" + voice + "|" + objs] = 1
    return f


def parse_all(nlp, rows):
    from spacy.tokens import Doc
    return list(nlp.pipe((Doc(nlp.vocab, words=s.split()) for s, _, _ in rows), batch_size=64))


def pairs(doc, events, version=1):
    X, y = [], []
    nouns = [t.i for t in doc if t.pos_ in NOMINAL]
    for e, ev in events.items():
        if e >= len(doc):
            continue
        cands = set(nouns) | set(ev["roles"])
        for n in sorted(cands):
            if n == e:
                continue
            X.append(features(doc, e, n, version))
            y.append(ev["roles"].get(n, "none"))
    return X, y


def main_predicate(doc):
    """CSI v0.1's choice: verbal root, or its verbal xcomp (rule 7)."""
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"):
        return None, None
    x = [c for c in root.children if c.dep_ == "xcomp" and c.pos_ == "VERB"]
    return (x[0], root) if x else (root, None)


def predict_roles(clf, vec, doc, v, version=1):
    nouns = [t.i for t in doc if t.pos_ in NOMINAL and t.i != v]
    if not nouns:
        return {}
    P = clf.predict_proba(vec.transform([features(doc, v, n, version) for n in nouns]))
    cls = list(clf.classes_)
    scored = sorted(((P[k][cls.index(r)], n, r) for k, n in enumerate(nouns) for r in ROLES if r in cls), reverse=True)
    out, used = {}, set()
    for p, n, r in scored:
        k = nouns.index(n)
        if r in out or n in used or P[k].argmax() != cls.index(r):
            continue
        out[r] = n
        used.add(n)
    return out


def evaluate(clf, vec, rows, docs, version=1):
    by = collections.defaultdict(lambda: [0, 0, 0])
    for (s, lf, case), doc in zip(rows, docs):
        b = by[case]
        b[0] += 1
        try:
            events, main = gold_events(s, lf)
        except (ValueError, IndexError):
            continue
        pred, emb = main_predicate(doc)
        if pred is None or main is None:
            continue
        gold_e = main
        if emb is not None and events[main]["verb"] == emb.lemma_.lower() and "xcomp" in events[main]:
            gold_e = events[main]["xcomp"]
        gold = events.get(gold_e)
        if gold is None:
            continue
        roles = predict_roles(clf, vec, doc, pred.i, version)
        gold_heads = {ROLE_TO_ARG[r]: s.split()[i].lower() for i, r in gold["roles"].items()}
        pred_heads = {ROLE_TO_ARG[r]: doc[i].lemma_.lower() if doc[i].pos_ != "PROPN" else doc[i].text.lower()
                      for r, i in roles.items()}
        b[1] += pred.lemma_.lower() == gold["verb"] and pred_heads == gold_heads
        b[2] += pred.i == gold_e and {ROLE_TO_ARG[r]: i for r, i in roles.items()} == \
            {ROLE_TO_ARG[r]: i for i, r in gold["roles"].items()}
    tot = [sum(v[i] for v in by.values()) for i in range(3)]
    return {"n": tot[0], "role_exact_6a": round(tot[1] / tot[0], 4), "role_exact_index": round(tot[2] / tot[0], 4),
            "by_case": {k: {"n": v[0], "role_exact_6a": round(v[1] / v[0], 4), "role_exact_index": round(v[2] / v[0], 4)}
                        for k, v in sorted(by.items())}}


def main():
    import spacy
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--features", type=int, default=1, help="1 = pre-registered set; 2 = post-hoc S2b")
    a = ap.parse_args()
    t0 = time.time()
    nlp = spacy.load(SPACY_MODEL)
    splits = {sp: load(a.data, sp, a.limit) for sp in ("train", "dev", "test", "gen")}
    docs = {sp: parse_all(nlp, rows) for sp, rows in splits.items()}
    print("parsed", {k: len(v) for k, v in docs.items()}, round(time.time() - t0), "s", flush=True)
    X, y, skipped = [], [], 0
    for (s, lf, _), doc in zip(splits["train"], docs["train"]):
        try:
            events, _ = gold_events(s, lf)
        except (ValueError, IndexError):
            skipped += 1
            continue
        xi, yi = pairs(doc, events, a.features)
        X += xi
        y += yi
    vec = DictVectorizer()
    Xv = vec.fit_transform(X)
    res = {"experiment": "E6b S2 learned linker", "parser": SPACY_MODEL, "features": a.features, "train_pairs": len(y),
           "train_label_counts": dict(collections.Counter(y)), "skipped_train": skipped, "dev": {}}
    best = None
    for C in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(Xv, y)
        r = evaluate(clf, vec, splits["dev"], docs["dev"], a.features)
        res["dev"][str(C)] = {k: r[k] for k in ("n", "role_exact_6a", "role_exact_index")}
        print("dev C", C, res["dev"][str(C)], flush=True)
        if best is None or r["role_exact_index"] > best[0]:
            best = (r["role_exact_index"], C, clf)
    res["chosen_C"] = best[1]
    for sp in ("test", "gen"):
        res[sp] = evaluate(best[2], vec, splits[sp], docs[sp], a.features)
        print(sp, {k: v for k, v in res[sp].items() if k != "by_case"}, flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
