"""CSI comparison (ARCHITECTURE.md section 3, rule 8): two texts are equivalent iff their main propositions agree on
predicate, role heads, polarity, tense, aspect, modality and embedding. Voice is a grammatical difference only.
v0.2-lexical (Experiment 3c): two heads with different lemmas still agree if each word's chosen meaning (Dictionary
v1.1) is one of the other lemma's meanings; words the dictionary does not cover agree only by lemma."""
from .analyze import CSI_VERSION, analyze

FIELDS = ("polarity", "tense", "aspect", "modality", "embedded_under")
ROLES = ("ARG0", "ARG1", "ARG2")


def _main(analysis):
    props = analysis["readings"][0]["propositions"]
    return props[0] if props else None


def _same(la, lb, sa, sb):
    """-> (agree, note)."""
    if la == lb:
        return True, None
    if not sa or not sb or sa["synset"] is None or sb["synset"] is None:
        return False, None
    from .analyze import _senses
    s = _senses()
    if sa["synset"] in s.meanings(lb, sb["upos"]) and sb["synset"] in s.meanings(la, sa["upos"]):
        return True, f"{la} ~ {lb}: shared meaning ({sa['synset']}, {sb['synset']})"
    return False, None


def compare(text_a, text_b):
    a, b = analyze(text_a), analyze(text_b)
    pa, pb = _main(a), _main(b)
    if pa is None or pb is None or a["unsupported"] or b["unsupported"]:
        return {"csi_version": CSI_VERSION, "equivalent": None, "confidence": 0.0, "differences": [],
                "notes": [f"unsupported: a={a['unsupported']} b={b['unsupported']}"]}
    diffs, notes = [], []
    agree, note = _same(pa["predicate"]["lemma"], pb["predicate"]["lemma"], pa["predicate"].get("sense"),
                        pb["predicate"].get("sense"))
    if not agree:
        diffs.append({"type": "predicate", "a": pa["predicate"]["lemma"], "b": pb["predicate"]["lemma"]})
    notes += [note] if note else []
    for r in ROLES:
        ra, rb = pa["roles"].get(r) or {}, pb["roles"].get(r) or {}
        agree, note = _same(ra.get("head_lemma"), rb.get("head_lemma"), ra.get("head_sense"), rb.get("head_sense"))
        if not agree:
            diffs.append({"type": f"role:{r}", "a": ra.get("head_lemma"), "b": rb.get("head_lemma")})
        notes += [note] if note else []
    for f in FIELDS:
        if pa[f] != pb[f]:
            diffs.append({"type": f, "a": pa[f], "b": pb[f]})
    if pa["voice"] != pb["voice"]:
        notes.append(f"voice differs ({pa['voice']} vs {pb['voice']}): grammatical, not a meaning difference")
    return {"csi_version": CSI_VERSION, "equivalent": not diffs, "confidence": 1.0, "differences": diffs,
            "notes": notes, "analyses": [a, b]}
