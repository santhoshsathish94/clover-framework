"""The general interpreter: intent (or a request for a different form) from CSI features, learned once from population
data - many general users with general phrasings, every frame and object. It is shared by everyone and never updated
by a person's interactions (its parameters are hashed before and after every run)."""
import hashlib
import json
import random

import numpy as np

from . import features as F
from . import world as W


def population_messages(seed=7, per_phrase=25):
    rng = random.Random(seed)
    rows = []
    for intent, (slots, _, phrases) in W.INTENTS.items():
        for p in phrases:
            for _ in range(per_phrase):
                vals = {t: rng.choice(W.objects(t, "all")) for t in slots}
                for ph in [x.split("}")[0] for x in p.split("{")[1:]]:
                    vals.setdefault(ph, rng.choice(W.objects(ph, "all")))
                rows.append((W.frame(rng.choice(W.GENERAL_FRAMES), W.fill(p, vals)), W.CONTEXT_OF[intent], intent))
    for (d, v), phrases in W.PRESENTATION_PHRASES.items():
        for p in phrases:
            for _ in range(per_phrase // 2):
                text = rng.choice((p.capitalize() + ".", p + ".", p.capitalize() + "!", "Ok, " + p + "."))
                rows.append((text, rng.choice(W.CONTEXTS), f"pres:{d}={v}"))
    for (d, v), statements in W.PREFERENCE_STATEMENTS.items():
        for s in statements:
            for _ in range(per_phrase // 5):
                rows.append((rng.choice((s, s.rstrip(".") + ", thanks.", "Just so you know, " + s[0].lower() + s[1:])),
                             rng.choice(W.CONTEXTS), f"pres:{d}={v}"))
    return rows


class GeneralInterpreter:
    def __init__(self, labels, vocab, coef, intercept):
        self.labels, self.vocab = list(labels), {f: k for k, f in enumerate(vocab)}
        self.coef, self.intercept = np.asarray(coef), np.asarray(intercept)

    @classmethod
    def train(cls, rows, reader, use_csi=True):
        from sklearn.linear_model import LogisticRegression
        feats = [F.features(t, c, reader.read(t), use_csi)[0] for t, c, _ in rows]
        vocab = sorted({f for fs in feats for f in fs})
        index = {f: k for k, f in enumerate(vocab)}
        X = np.zeros((len(rows), len(vocab)), dtype=np.float32)
        for r, fs in enumerate(feats):
            for f in fs:
                X[r, index[f]] = 1.0
        y = [lab for _, _, lab in rows]
        m = LogisticRegression(max_iter=2000, C=4.0).fit(X, y)
        return cls(m.classes_, vocab, m.coef_, m.intercept_)

    def scores(self, feats):
        x = np.zeros(len(self.vocab), dtype=np.float32)
        for f in feats:
            k = self.vocab.get(f)
            if k is not None:
                x[k] = 1.0
        z = self.coef @ x + self.intercept
        p = np.exp(z - z.max())
        p /= p.sum()
        order = np.argsort(-p)
        return [(self.labels[k], float(p[k])) for k in order]

    def task_scores(self, feats):
        s = [(l, p) for l, p in self.scores(feats) if not l.startswith("pres:")]
        total = sum(p for _, p in s) or 1.0
        return [(l, p / total) for l, p in s]

    def digest(self):
        h = hashlib.sha256()
        h.update(json.dumps(self.labels).encode())
        h.update(json.dumps(sorted(self.vocab, key=self.vocab.get)).encode())
        h.update(np.ascontiguousarray(self.coef).tobytes())
        h.update(np.ascontiguousarray(self.intercept).tobytes())
        return h.hexdigest()

    def save(self, path):
        np.savez(path, labels=np.array(self.labels), vocab=np.array(sorted(self.vocab, key=self.vocab.get)),
                 coef=self.coef, intercept=self.intercept)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(d["labels"].tolist(), d["vocab"].tolist(), d["coef"], d["intercept"])
