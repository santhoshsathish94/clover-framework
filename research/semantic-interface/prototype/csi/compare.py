"""CSI v0.1 comparison (ARCHITECTURE.md section 3, rule 8): two texts are equivalent iff their main propositions agree
on predicate, role heads, polarity, tense, aspect, modality and embedding. Voice is a grammatical difference only."""
from .analyze import CSI_VERSION, analyze

FIELDS = ("polarity", "tense", "aspect", "modality", "embedded_under")
ROLES = ("ARG0", "ARG1", "ARG2")


def _main(analysis):
    props = analysis["readings"][0]["propositions"]
    return props[0] if props else None


def compare(text_a, text_b):
    a, b = analyze(text_a), analyze(text_b)
    pa, pb = _main(a), _main(b)
    if pa is None or pb is None or a["unsupported"] or b["unsupported"]:
        return {"csi_version": CSI_VERSION, "equivalent": None, "confidence": 0.0, "differences": [],
                "notes": [f"unsupported: a={a['unsupported']} b={b['unsupported']}"]}
    diffs = []
    if pa["predicate"]["lemma"] != pb["predicate"]["lemma"]:
        diffs.append({"type": "predicate", "a": pa["predicate"]["lemma"], "b": pb["predicate"]["lemma"]})
    for r in ROLES:
        ha = (pa["roles"].get(r) or {}).get("head_lemma")
        hb = (pb["roles"].get(r) or {}).get("head_lemma")
        if ha != hb:
            diffs.append({"type": f"role:{r}", "a": ha, "b": hb})
    for f in FIELDS:
        if pa[f] != pb[f]:
            diffs.append({"type": f, "a": pa[f], "b": pb[f]})
    notes = []
    if pa["voice"] != pb["voice"]:
        notes.append(f"voice differs ({pa['voice']} vs {pb['voice']}): grammatical, not a meaning difference")
    return {"csi_version": CSI_VERSION, "equivalent": not diffs, "confidence": 1.0, "differences": diffs,
            "notes": notes, "analyses": [a, b]}
