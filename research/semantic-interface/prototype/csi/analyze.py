"""CSI v0.1 analysis: a learned parser gives tokens, lemmas, tags, morphology and dependencies; the written structural
correspondences of ARCHITECTURE.md section 3 (and only those) turn them into a proposition. The parser is chosen by
CSI_PARSER (default spaCy en_core_web_sm, ClearNLP labels); a UD-label parser (e.g. our EWT-trained pipeline) gets the
same correspondences written for UD labels (ARCHITECTURE.md section 3b). Output follows the JSON contract (section 2)."""
import functools
import os

CSI_VERSION = "0.1"
SPACY_MODEL = os.environ.get("CSI_PARSER", "en_core_web_sm")  # Experiment 2 compares parsers; recorded in provenance
PARSER = f"spacy/{SPACY_MODEL}"
FUTURE = {"will", "shall"}
FINITE_TENSE = {"VBD": "past", "VBZ": "present", "VBP": "present"}
# the same correspondences under the two label schemes (section 3 = ClearNLP, section 3b = UD)
SCHEME = {
    "clearnlp": {"aux": {"aux", "auxpass"}, "passive": {"nsubjpass", "auxpass", "csubjpass"}, "subjpass": {"nsubjpass"},
                 "obj": {"dobj"}},
    "ud": {"aux": {"aux", "aux:pass"}, "passive": {"nsubj:pass", "aux:pass", "csubj:pass"}, "subjpass": {"nsubj:pass"},
           "obj": {"obj"}},
}


@functools.lru_cache(maxsize=1)
def _nlp():
    import spacy
    nlp = spacy.load(SPACY_MODEL)
    labels = set(nlp.get_pipe("parser").labels)
    scheme = "ud" if "obj" in labels and "dobj" not in labels else "clearnlp"
    return nlp, f"{PARSER}@{nlp.meta.get('version', '?')}:{scheme}", scheme


def _kids(tok, deps):
    return [c for c in tok.children if c.dep_ in deps]


def _role(tok):
    span = sorted(tok.subtree, key=lambda t: t.i)
    start, end = span[0].i, span[-1].i
    return {"text": tok.doc[start:end + 1].text, "head_lemma": tok.lemma_.lower(), "tokens": list(range(start, end + 1))}


def _pobj(prep):
    objs = _kids(prep, {"pobj"})
    return objs[0] if objs else None


def _case(tok, lemma):
    """UD: does this nominal carry a case marker with this lemma ("to", "by")?"""
    return any(c.dep_ == "case" and c.lemma_.lower() == lemma for c in tok.children)


def _negated(tok, scheme):
    if scheme == "ud":  # rule 6 for UD: an advmod dependent marked Polarity=Neg (not, n't, never)
        return any(c.dep_ == "advmod" and "Neg" in c.morph.get("Polarity") for c in tok.children)
    return bool(_kids(tok, {"neg"}))


def _proposition(doc, parser_id, scheme="clearnlp"):
    L = SCHEME[scheme]
    root = next((t for t in doc if t.dep_ in ("ROOT", "root")), None)
    if root is None or root.pos_ not in ("VERB", "AUX"):
        return None, "no verbal root"
    pred, embedded = root, None
    xcomp = [c for c in root.children if c.dep_ == "xcomp" and c.pos_ == "VERB"]
    if xcomp:  # rule 7: "tried to open" -> proposition about "open", embedded under "try"
        pred, embedded = xcomp[0], root.lemma_.lower()
    auxes = _kids(root, L["aux"])
    aux_lemmas = [a.lemma_.lower() for a in auxes]
    passive = bool(_kids(pred, L["passive"]))
    roles = {}
    if passive:  # rule 2
        subj = _kids(pred, L["subjpass"]) or _kids(root, L["subjpass"])
        if subj:
            roles["ARG1"] = _role(subj[0])
        if scheme == "ud":
            agent = _kids(pred, {"obl:agent"})
            if agent:
                roles["ARG0"] = _role(agent[0])
        else:
            agent = _kids(pred, {"agent"})
            if agent and _pobj(agent[0]) is not None:
                roles["ARG0"] = _role(_pobj(agent[0]))
    else:  # rule 1
        subj = _kids(root, {"nsubj"})
        if subj:
            roles["ARG0"] = _role(subj[0])
        obj = _kids(pred, L["obj"])
        if obj:
            roles["ARG1"] = _role(obj[0])
    if scheme == "ud":  # rule 1, recipient: iobj; else a "to" oblique when there is an object
        iobj = _kids(pred, {"iobj"})
        if iobj:
            roles["ARG2"] = _role(iobj[0])
        elif "ARG1" in roles:
            to = [c for c in _kids(pred, {"obl"}) if _case(c, "to")]
            if to:
                roles["ARG2"] = _role(to[0])
    else:
        dative = _kids(pred, {"dative"})
        if dative:
            target = _pobj(dative[0]) if dative[0].pos_ == "ADP" else dative[0]
            if target is not None:
                roles["ARG2"] = _role(target)
        elif "ARG1" in roles:
            to = [c for c in _kids(pred, {"prep"}) if c.lemma_.lower() == "to" and _pobj(c) is not None]
            if to:
                roles["ARG2"] = _role(_pobj(to[0]))
    modal = next((a.lemma_.lower() for a in auxes if a.tag_ == "MD" and a.lemma_.lower() not in FUTURE), None)
    if any(m in aux_lemmas for m in FUTURE):  # rule 3
        tense = "future"
    elif modal:
        tense = "unmarked"
    else:  # rule 5
        finite = next((a for a in auxes if a.tag_ in FINITE_TENSE), None)
        if finite is None and root.tag_ in FINITE_TENSE:
            finite = root
        tense = FINITE_TENSE[finite.tag_] if finite is not None else "unmarked"
    perfect = any(a.lemma_.lower() == "have" and a.dep_ == "aux" for a in auxes)  # rule 4
    progressive = any(a.lemma_.lower() == "be" and a.dep_ == "aux" for a in auxes) and root.tag_ == "VBG"
    aspect = ("perfect progressive" if perfect and progressive else "perfect" if perfect
              else "progressive" if progressive else "simple")
    negative = _negated(root, scheme) or _negated(pred, scheme)  # rule 6
    prop = {
        "predicate": {"lemma": pred.lemma_.lower(), "token": pred.i},
        "roles": roles,
        "polarity": "negative" if negative else "positive",
        "tense": tense, "aspect": aspect, "modality": modal,
        "voice": "passive" if passive else "active",
        "embedded_under": embedded,
        "provenance": {"syntax": parser_id, "structure": f"csi.analyze@{CSI_VERSION}"},
    }
    missing = None if "ARG0" in roles or (passive and "ARG1" in roles) else "no subject"
    return prop, missing


def analyze(text):
    nlp, parser_id, scheme = _nlp()
    doc = nlp(text)
    tokens = [{"i": t.i, "text": t.text, "lemma": t.lemma_, "upos": t.pos_, "tag": t.tag_,
               "feats": t.morph.to_dict(), "head": t.head.i, "dep": t.dep_} for t in doc]
    prop, problem = _proposition(doc, parser_id, scheme)
    unsupported = [problem] if problem else []
    reading = {"confidence": 1.0 if not unsupported else 0.5, "tokens": tokens,
               "propositions": [prop] if prop else []}
    return {"csi_version": CSI_VERSION, "text": text, "readings": [reading], "ambiguity": [],
            "unsupported": unsupported, "confidence_note": "v0.1 confidence is uncalibrated (coverage only)"}
