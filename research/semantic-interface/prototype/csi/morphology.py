"""CSI morphology component (Experiment 4c): English inflection learned from spelling.
analyse(form, cls) -> lemma and generate(lemma, cls, form_type) -> form. Each direction is a learned edit model
(strip k final letters, append s; logistic regression over the word's last 1-6 letters plus length, as in
Experiment 4b) backed by an exception lexicon: the training words the learned model itself gets wrong (memorised
irregulars). Classes: verb (past, past_participle, gerund, third_singular), noun (plural), adj (comparative,
superlative)."""
import collections
import json
import os

import numpy as np

FORMS = {  # UniMorph feature set -> (class, form type)
    frozenset({"V", "PST"}): ("verb", "past"),
    frozenset({"V", "V.PTCP", "PST"}): ("verb", "past_participle"),
    frozenset({"V", "V.PTCP", "PRS"}): ("verb", "gerund"),
    frozenset({"V", "3", "SG", "PRS"}): ("verb", "third_singular"),
    frozenset({"N", "PL"}): ("noun", "plural"),
    frozenset({"ADJ", "CMPR"}): ("adj", "comparative"),
    frozenset({"ADJ", "SPRL"}): ("adj", "superlative"),
}


def shape(word):
    return "".join("V" if c in "aeiou" else "Y" if c == "y" else "C" for c in word)


def features(word, max_len=6, feature_set="suffix"):
    f = {f"s{n}={word[-n:]}": 1.0 for n in range(1, max_len + 1) if len(word) >= n}
    f["len"] = min(len(word), 12) / 12.0
    if feature_set == "suffix+shape":  # letter classes only: nothing here names a particular ending family
        s = shape(word)
        for n in range(2, 7):
            f[f"sh{n}={s[-n:]}"] = 1.0
        for suf in ("ing", "ied", "ies", "ed", "es", "s", "d"):
            if word.endswith(suf) and len(word) > len(suf) + 1:
                stem = word[:-len(suf)]
                f[f"stem-{suf}:shape={shape(stem)[-3:]}"] = 1.0
                f[f"stem-{suf}:double={stem[-1] == stem[-2]}"] = 1.0
                f[f"stem-{suf}:last={stem[-1]}"] = 1.0
    return f


def edit_of(src, dst):
    k = 0
    while k < min(len(src), len(dst)) and src[k] == dst[k]:
        k += 1
    return len(src) - k, dst[k:]


def apply_edit(word, edit):
    k, s = edit
    return (word[:-k] if k else word) + s


class EditModel:
    """Suffix-feature linear classifier over edit classes; prediction needs only numpy."""

    def __init__(self, vocab, classes, W, b, max_len=6, feature_set="suffix"):
        self.vocab, self.classes, self.W, self.b, self.max_len = vocab, classes, W, b, max_len
        self.feature_set = feature_set

    @classmethod
    def train(cls, pairs, max_len=6, C=10.0, min_count=2, edits=None, feature_set="suffix"):
        """pairs: (source word, target word); edits: optional precomputed edit labels (the shuffled-label control)."""
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        edits = edits if edits is not None else [edit_of(s, d) for s, d in pairs]
        keep = {e for e, c in collections.Counter(edits).items() if c >= min_count}
        rows = [(s, e) for (s, _), e in zip(pairs, edits) if e in keep]
        vec = DictVectorizer()
        X = vec.fit_transform([features(s, max_len, feature_set) for s, _ in rows])
        y = [f"{e[0]}|{e[1]}" for _, e in rows]
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(X, y)
        classes = [(int(c.split("|", 1)[0]), c.split("|", 1)[1]) for c in clf.classes_]
        W = clf.coef_ if len(classes) > 2 else np.vstack([-clf.coef_[0], clf.coef_[0]])
        b = clf.intercept_ if len(classes) > 2 else np.array([-clf.intercept_[0], clf.intercept_[0]])
        return cls(vec.vocabulary_, classes, W.astype(np.float32), b.astype(np.float32), max_len, feature_set)

    def candidates(self, word, n=5):
        """The n best-ranked distinct outputs, best first."""
        f = features(word, self.max_len, self.feature_set)
        scores = self.b.copy()
        for k, v in f.items():
            if k in self.vocab:
                scores += self.W[:, self.vocab[k]] * v
        out = []
        for c in np.argsort(-scores):
            k, s = self.classes[c]
            if k <= len(word):
                w = apply_edit(word, (k, s))
                if w not in out:
                    out.append(w)
                    if len(out) == n:
                        break
        return out

    def predict(self, word):
        c = self.candidates(word, 1)
        return c[0] if c else word


def _exceptions(model, pairs):
    """Words with exactly one answer in training that the learned model gets wrong (ambiguous words stay with it)."""
    answers = collections.defaultdict(set)
    for src, dst in pairs:
        answers[src].add(dst)
    return {src: next(iter(d)) for src, d in sorted(answers.items()) if len(d) == 1 and model.predict(src) not in d}


class Morphology:
    """Exception lexicon first, learned edit model otherwise; one analyser per class, one generator per form type."""

    def __init__(self):
        self.analysers, self.analyse_exceptions = {}, {}
        self.generators, self.generate_exceptions = {}, {}

    @classmethod
    def train(cls, entries, classes=("verb", "noun", "adj"), max_len=6, generate=True, feature_set="suffix"):
        """entries: (lemma, form, cls, form_type) from training lemmas only."""
        m = cls()
        for c in classes:
            pairs = [(f, l) for l, f, k, _ in entries if k == c]  # one pair per form type, as in Experiment 4b
            if not pairs:
                continue
            m.analysers[c] = EditModel.train(pairs, max_len, feature_set=feature_set)
            m.analyse_exceptions[c] = _exceptions(m.analysers[c], pairs)
            if generate:
                for t in sorted({t for _, _, k, t in entries if k == c}):
                    gp = [(l, f) for l, f, k, tt in entries if k == c and tt == t]
                    m.generators[(c, t)] = EditModel.train(gp, max_len, feature_set=feature_set)
                    m.generate_exceptions[(c, t)] = _exceptions(m.generators[(c, t)], gp)
        return m

    def analyse(self, form, cls, use_exceptions=True):
        if use_exceptions and form in self.analyse_exceptions.get(cls, {}):
            return self.analyse_exceptions[cls][form]
        return self.analysers[cls].predict(form) if cls in self.analysers else form

    def candidates(self, form, cls, n=5):
        """Ranked candidate lemmas from the learned model alone (no exception lexicon)."""
        return self.analysers[cls].candidates(form, n) if cls in self.analysers else [form]

    def generate(self, lemma, cls, form_type, use_exceptions=True):
        key = (cls, form_type)
        if use_exceptions and lemma in self.generate_exceptions.get(key, {}):
            return self.generate_exceptions[key][lemma]
        return self.generators[key].predict(lemma)


def load_unimorph(path, alphabetic=True):
    """-> list of (lemma, form, class, form type) for the inflections CSI uses."""
    import re
    out = []
    for line in open(path, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) < 3:
            continue
        key = frozenset(c[2].split(";"))
        if key not in FORMS:
            continue
        if alphabetic and not (re.fullmatch(r"[a-z]+", c[0]) and re.fullmatch(r"[a-z]+", c[1])):
            continue
        out.append((c[0], c[1], *FORMS[key]))
    return sorted(set(out))


def save(m, directory):
    os.makedirs(directory, exist_ok=True)
    meta, arrays = {"analysers": {}, "generators": {}}, {}
    for kind, models in (("analysers", m.analysers), ("generators", {f"{c}|{t}": g for (c, t), g in m.generators.items()})):
        for name, em in models.items():
            key = f"{kind}:{name}"
            meta[kind][name] = {"vocab": em.vocab, "classes": em.classes, "max_len": em.max_len,
                                "feature_set": em.feature_set}
            arrays[key + ":W"], arrays[key + ":b"] = em.W.astype(np.float16), em.b
    meta["analyse_exceptions"] = m.analyse_exceptions
    meta["generate_exceptions"] = {f"{c}|{t}": v for (c, t), v in m.generate_exceptions.items()}
    json.dump(meta, open(os.path.join(directory, "morphology.json"), "w"))
    np.savez_compressed(os.path.join(directory, "morphology.npz"), **arrays)


def load(directory):
    meta = json.load(open(os.path.join(directory, "morphology.json")))
    arrays = np.load(os.path.join(directory, "morphology.npz"))
    m = Morphology()
    for kind, target in (("analysers", m.analysers), ("generators", m.generators)):
        for name, d in meta[kind].items():
            key = f"{kind}:{name}"
            em = EditModel(d["vocab"], [tuple(c) for c in d["classes"]], arrays[key + ":W"].astype(np.float32),
                           arrays[key + ":b"], d["max_len"], d.get("feature_set", "suffix"))
            target[tuple(name.split("|")) if kind == "generators" else name] = em
    m.analyse_exceptions = meta["analyse_exceptions"]
    m.generate_exceptions = {tuple(k.split("|")): v for k, v in meta["generate_exceptions"].items()}
    return m
