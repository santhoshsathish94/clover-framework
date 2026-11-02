"""CSI dictionary layer (Experiment 4g): which words exist, per part of speech, and their listed irregular or
spelling-changed forms, from Open English WordNet 2025 (CC BY 4.0). lemmatise() composes the layers: Grammar supplies
the class, Morphology ranks candidate lemmas, the Dictionary says which candidate is a word."""
import collections
import gzip
import xml.etree.ElementTree as ET

POS = {"verb": ("v",), "noun": ("n",), "adj": ("a", "s")}


class Dictionary:
    def __init__(self, lemmas, forms):
        self.lemmas, self.forms = lemmas, forms  # class -> set of lemmas; class -> form -> set of lemmas

    @classmethod
    def load_oewn(cls, path):
        to_cls = {p: c for c, ps in POS.items() for p in ps}
        lemmas = collections.defaultdict(set)
        forms = collections.defaultdict(lambda: collections.defaultdict(set))
        with (gzip.open if path.endswith(".gz") else open)(path, "rb") as fh:
            for _, el in ET.iterparse(fh):
                if el.tag != "LexicalEntry":
                    continue
                lem = el.find("Lemma")
                c, w = to_cls.get(lem.get("partOfSpeech")), (lem.get("writtenForm") or "").lower()
                if c and w and " " not in w:
                    lemmas[c].add(w)
                    for f in el.findall("Form"):
                        forms[c][f.get("writtenForm").lower()].add(w)
                el.clear()
        return cls(dict(lemmas), {c: dict(v) for c, v in forms.items()})

    def knows(self, word, cls):
        return word in self.lemmas.get(cls, ())

    def listed(self, form, cls):
        return self.forms.get(cls, {}).get(form, set())

    def words(self):
        out = set()
        for c in self.lemmas:
            out |= self.lemmas[c] | set(self.forms.get(c, {}))
        return out


def lemmatise(form, cls, morphology, dictionary, n=5, form_type=None):
    """-> (lemma, source); source is "listed", "known" or "unknown" (no candidate is a dictionary word). Given the
    form type and a morphology with generators (v1.1), a known candidate must regenerate the observed form."""
    cands = morphology.candidates(form, cls, n)
    listed = dictionary.listed(form, cls)
    if listed:
        return min(sorted(listed), key=lambda l: cands.index(l) if l in cands else n), "listed"
    check = form_type is not None and (cls, form_type) in morphology.generators
    for c in cands:
        if dictionary.knows(c, cls) and (not check or morphology.generate(c, cls, form_type, use_exceptions=False) == form):
            return c, "known"
    return (cands[0] if cands else form), "unknown"
