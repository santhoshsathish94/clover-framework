"""CSI comparison (ARCHITECTURE.md section 3, rule 8): two texts are equivalent iff their main propositions agree on
predicate, role heads, polarity, tense, aspect, modality and embedding. Voice is a grammatical difference only.
v0.2-lexical (Experiment 3c): two heads with different lemmas still agree if each word's chosen meaning (Dictionary
v1.1) is one of the other lemma's meanings; words the dictionary does not cover agree only by lemma.
v0.3-lexical (Experiment 3c2): instead, they agree if the dictionary's vectors of the two chosen meanings are close
(cosine >= CSI_CLOSENESS, a threshold set on development data) and neither word heads another slot of the other
proposition. The dictionary supplies meanings and vectors as evidence; the decision is made here."""
from .analyze import CSI_CLOSENESS, CSI_VERSION, analyze

FIELDS = ("polarity", "tense", "aspect", "modality", "embedded_under")
ROLES = ("ARG0", "ARG1", "ARG2")


def _main(analysis):
    props = analysis["readings"][0]["propositions"]
    return props[0] if props else None


def _heads(prop):
    return {prop["predicate"]["lemma"]} | {r["head_lemma"] for r in prop["roles"].values()}


def _same(la, lb, sa, sb, heads_a, heads_b):
    """-> (agree, note, evidence); evidence describes one word standing where the other stood (None if lemmas match)."""
    if la == lb:
        return True, None, None
    if not la or not lb:
        return False, None, None
    ev = {"a": la, "b": lb, "covered": bool(sa and sb and sa["synset"] and sb["synset"])}
    if not ev["covered"]:
        return False, None, ev
    from .analyze import _senses
    s = _senses()
    ev["shared_meaning"] = sa["synset"] in s.meanings(lb, sb["upos"]) and sb["synset"] in s.meanings(la, sa["upos"])
    if not CSI_CLOSENESS:  # v0.2
        note = f"{la} ~ {lb}: shared meaning ({sa['synset']}, {sb['synset']})" if ev["shared_meaning"] else None
        return ev["shared_meaning"], note, ev
    va, vb = s._gloss_vectors([sa["synset"], sb["synset"]])  # the frozen dictionary's own meaning vectors
    ev["closeness"] = round(float(va @ vb / (va.norm() * vb.norm())), 4)
    ev["aligned"] = la not in heads_b and lb not in heads_a  # a word present in both propositions changed role
    if CSI_CLOSENESS != "record" and ev["aligned"] and ev["closeness"] >= float(CSI_CLOSENESS):
        return True, f"{la} ~ {lb}: close meanings ({sa['synset']}, {sb['synset']}: {ev['closeness']})", ev
    return False, None, ev


def compare(text_a, text_b):
    a, b = analyze(text_a), analyze(text_b)
    pa, pb = _main(a), _main(b)
    if pa is None or pb is None or a["unsupported"] or b["unsupported"]:
        return {"csi_version": CSI_VERSION, "equivalent": None, "confidence": 0.0, "differences": [],
                "notes": [f"unsupported: a={a['unsupported']} b={b['unsupported']}"], "lexical": []}
    diffs, notes, lexical = [], [], []
    slots = [("predicate", pa["predicate"]["lemma"], pb["predicate"]["lemma"], pa["predicate"].get("sense"),
              pb["predicate"].get("sense"))]
    for r in ROLES:
        ra, rb = pa["roles"].get(r) or {}, pb["roles"].get(r) or {}
        slots.append((f"role:{r}", ra.get("head_lemma"), rb.get("head_lemma"), ra.get("head_sense"), rb.get("head_sense")))
    heads_a, heads_b = _heads(pa), _heads(pb)
    for slot, la, lb, sa, sb in slots:
        agree, note, ev = _same(la, lb, sa, sb, heads_a, heads_b)
        if not agree:
            diffs.append({"type": slot, "a": la, "b": lb})
        notes += [note] if note else []
        if ev:
            lexical.append({"slot": slot, **ev, "accepted": agree, "sense_a": sa, "sense_b": sb})
    for f in FIELDS:
        if pa[f] != pb[f]:
            diffs.append({"type": f, "a": pa[f], "b": pb[f]})
    if pa["voice"] != pb["voice"]:
        notes.append(f"voice differs ({pa['voice']} vs {pb['voice']}): grammatical, not a meaning difference")
    return {"csi_version": CSI_VERSION, "equivalent": not diffs, "confidence": 1.0, "differences": diffs,
            "notes": notes, "lexical": lexical, "analyses": [a, b]}
