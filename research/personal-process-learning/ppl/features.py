"""Features of a message, from CSI's reading (frozen v0.1r3) and the message itself: what a person's habit, a general
interpretation or a memory may depend on. Object names never become features, so learning is not tied to particular
files, people or meetings. Readings are cached by text."""
import json
import os
import re

from . import world as W

LIGHT = {"be", "have", "do", "get", "give", "make", "let", "take", "could", "would", "should", "can", "need", "want",
         "like", "go", "mind", "please", "pls", "kindly", "just", "quick", "chance", "when", "ahead", "you", "i", "me",
         "my", "it", "this", "that", "the", "a", "an", "for", "to", "of", "and", "now", "some", "somewhere", "again", "so"}
CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "X", "INTJ", "NUM", "ADP", "PART"}
ARROWS = ("->", "→", "=>")
_OBJECTS = sorted({(o.lower(), t, o) for t, items in W.LEXICON.items() for o in items} | {("the team", "person", "the team")},
                  key=lambda x: -len(x[0]))


def find_objects(text):
    """Known objects in the text: [(start, end, type, canonical name)], longest first, no overlaps."""
    low, taken, out = text.lower(), [], []
    for o, t, name in _OBJECTS:
        for m in re.finditer(r"(?<![\w.])" + re.escape(o) + r"(?![\w])", low):
            if not any(m.start() < e and s < m.end() for s, e in taken):
                taken.append((m.start(), m.end()))
                out.append((m.start(), m.end(), t, name))
    return sorted(out)


class CSIReader:
    """CSI's analysis (csi.analyze) reduced to what the features need, with a persistent cache."""

    def __init__(self, cache_path, preload=()):
        from csi import analyze  # the frozen research prototype on k3
        self.analyze, self.path, self.cache = analyze, cache_path, {}
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        for path in [*preload, cache_path]:
            if os.path.exists(path):
                for line in open(path, encoding="utf-8"):
                    r = json.loads(line)
                    self.cache[r["text"]] = r["reading"]
        self.out = open(cache_path, "a", encoding="utf-8")

    def read(self, text):
        if text not in self.cache:
            a = self.analyze(text)
            rd = a["readings"][0]
            self.cache[text] = {
                "props": [{"pred": p["predicate"]["lemma"], "mood": p["mood"], "pol": p["polarity"],
                           "modality": p["modality"], "arg0": (p["roles"].get("ARG0") or {}).get("head_lemma")}
                          for p in rd["propositions"]],
                "tokens": [[t["text"], t["lemma"], t["upos"], t["dep"], t["idx"]] for t in rd["tokens"]],
                "unsupported": bool(a["unsupported"])}
            self.out.write(json.dumps({"text": text, "reading": self.cache[text]}, ensure_ascii=False) + "\n")
            self.out.flush()
        return self.cache[text]


class SurfaceReader:
    """No CSI: tokens split from the text (ablation, and local tests)."""

    def read(self, text):
        toks, pos = [], 0
        for m in re.finditer(r"[\w'.-]+|->|→|=>|[?:,!]", text):
            toks.append([m.group(0), m.group(0).lower(), "X", "", m.start()])
        return {"props": [], "tokens": toks, "unsupported": True}


def features(text, context, reading, use_csi=True):
    objs = find_objects(text)
    spans = [(s, e) for s, e, _, _ in objs]
    f = {f"ctx={context}"} | {f"has={t}" for _, _, t, _ in objs}
    if any(a in text for a in ARROWS):
        f.add("pat=arrow")
    if ":" in text:
        f.add("pat=colon")
    for word, lemma, upos, dep, idx in reading["tokens"]:
        if any(s <= idx < e for s, e in spans):
            continue
        lem = (lemma if use_csi else word).lower().strip(".'")
        if not lem or not re.search(r"[a-z]", lem) or lem in LIGHT or (use_csi and upos not in CONTENT_UPOS):
            continue
        f.add(f"tok={lem}")
    if use_csi:
        props = reading["props"]
        ok = bool(props) and not reading["unsupported"]
        f.add("csi=ok" if ok else "csi=none")
        for p in props:
            if p["pred"] not in LIGHT:
                f.add(f"pred={p['pred']}")
        if any(p["mood"] == "imperative" for p in props) or (
                "?" in text and any(p["arg0"] == "you" for p in props)):
            f.add("mood=request")
        elif props:
            f.add("mood=statement")
        if props and props[0]["pol"] == "negative":
            f.add("pol=neg")
    return frozenset(f), objs
