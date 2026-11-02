"""CSI v0.1 analysis: a learned parser gives tokens, lemmas, tags, morphology and dependencies; the written structural
correspondences of ARCHITECTURE.md section 3 (and only those) turn them into a proposition. The parser is chosen by
CSI_PARSER (default spaCy en_core_web_sm, ClearNLP labels); a UD-label parser (e.g. our EWT-trained pipeline) gets the
same correspondences written for UD labels (ARCHITECTURE.md section 3b). Output follows the JSON contract (section 2).
v0.1r (Experiment 1d, section 3c): every sentence and its coordinated or paratactic clauses, imperatives, and the
proposition that governs each token; v0.1r2 (Experiment 1e): rule 9b, main clauses joined in one sentence."""
import functools
import os

CSI_SENSES = os.environ.get("CSI_SENSES")  # frozen Dictionary v1.1 checkpoint (Experiment 3c); unset = lemmas only
CSI_CLOSENESS = os.environ.get("CSI_CLOSENESS")  # Experiment 3c2: closeness threshold from dev data, or "record"
CSI_VERSION = "0.3-lexical" if CSI_SENSES and CSI_CLOSENESS else "0.2-lexical" if CSI_SENSES else "0.1"
SPACY_MODEL = os.environ.get("CSI_PARSER", "en_core_web_sm")  # Experiment 2 compares parsers; recorded in provenance
PARSER = f"spacy/{SPACY_MODEL}"
CSI_MORPH = os.environ.get("CSI_MORPH")  # frozen verb lemmatiser artefact (Experiment 4d); unset = parser lemmas
CSI_DICT = os.environ.get("CSI_DICT")  # OEWN LMF file, the dictionary layer the lemmatiser composes with
TAG_FORM = {"VBD": "past", "VBN": "past_participle", "VBG": "gerund", "VBZ": "third_singular"}  # frozen scope: inflected VERB tokens
FUTURE = {"will", "shall"}
FINITE_TENSE = {"VBD": "past", "VBZ": "present", "VBP": "present"}
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl", "nsubj:pass", "csubj:pass"}
ATTACHED = {"conj", "parataxis"}  # rule 9
JOINS = {",", ";", ":"}  # rule 9b
ADDRESSEE = {"text": "", "head_lemma": "you", "head": None, "tokens": [], "implicit": True}  # rule 10
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


@functools.lru_cache(maxsize=1)
def _lemmatiser():
    from .dictionary import Dictionary
    from .morphology import load
    return load(CSI_MORPH), Dictionary.load_oewn(CSI_DICT), f"csi.morphology:{os.path.basename(CSI_MORPH.rstrip('/'))}+oewn"


@functools.lru_cache(maxsize=1)
def _senses():
    from .senses import Senses
    return Senses(CSI_SENSES, CSI_DICT)


def _sense_of(doc, i):
    t = doc[i]
    r = _senses().choose(doc.text, t.idx, t.idx + len(t.text), t.lemma_.lower(), t.pos_)
    top = r["senses"][0] if r["senses"] else {}
    return {"status": r["status"], "synset": top.get("synset"), "p": top.get("p"), "n_meanings": len(r["senses"]),
            "upos": t.pos_, "usage_trained": r["usage_trained"]}


def _kids(tok, deps):
    return [c for c in tok.children if c.dep_ in deps]


def _role(tok):
    span = sorted(tok.subtree, key=lambda t: t.i)
    start, end = span[0].i, span[-1].i
    return {"text": tok.doc[start:end + 1].text, "head_lemma": tok.lemma_.lower(), "head": tok.i,
            "tokens": list(range(start, end + 1))}


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


def _clause(head, parser_id, scheme="clearnlp", carried=None):
    """One proposition for the clause headed by `head`: a sentence's root (rules 1-8, 10) or a verb attached to a clause
    by conj or parataxis (rule 9). carried: (ARG0, mood) of the clause it is attached to."""
    L = SCHEME[scheme]
    if head.pos_ not in ("VERB", "AUX"):
        return None, "no verbal root"
    pred, embedded = head, None
    xcomp = [c for c in head.children if c.dep_ == "xcomp" and c.pos_ == "VERB"]
    if xcomp:  # rule 7: "tried to open" -> proposition about "open", embedded under "try"
        pred, embedded = xcomp[0], head.lemma_.lower()
    auxes = _kids(head, L["aux"])
    aux_lemmas = [a.lemma_.lower() for a in auxes]
    passive = bool(_kids(pred, L["passive"]))
    roles = {}
    if passive:  # rule 2
        subj = _kids(pred, L["subjpass"]) or _kids(head, L["subjpass"])
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
        subj = _kids(head, {"nsubj"})
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
        if finite is None and head.tag_ in FINITE_TENSE:
            finite = head
        tense = FINITE_TENSE[finite.tag_] if finite is not None else "unmarked"
    perfect = any(a.lemma_.lower() == "have" and a.dep_ == "aux" for a in auxes)  # rule 4
    progressive = any(a.lemma_.lower() == "be" and a.dep_ == "aux" for a in auxes) and head.tag_ == "VBG"
    aspect = ("perfect progressive" if perfect and progressive else "perfect" if perfect
              else "progressive" if progressive else "simple")
    negative = _negated(head, scheme) or _negated(pred, scheme)  # rule 6
    mood = "indicative"
    if "ARG0" not in roles and not (passive and "ARG1" in roles) and not _kids(head, SUBJECTS):
        if carried is not None:  # rule 9: the subject of the clause it is attached to
            roles["ARG0"], mood = dict(carried[0]), carried[1]
        elif head.tag_ == "VB" and all(a == "do" for a in aux_lemmas):  # rule 10: "(Do not) search the web."
            roles["ARG0"], mood = dict(ADDRESSEE), "imperative"
    if mood == "imperative":
        tense = "unmarked"
    prop = {
        "predicate": {"lemma": pred.lemma_.lower(), "token": pred.i},
        "roles": roles,
        "polarity": "negative" if negative else "positive",
        "tense": tense, "aspect": aspect, "modality": modal, "mood": mood,
        "voice": "passive" if passive else "active",
        "embedded_under": embedded,
        "provenance": {"syntax": parser_id, "structure": f"csi.analyze@{CSI_VERSION}"},
    }
    missing = None if "ARG0" in roles or (passive and "ARG1" in roles) else "no subject"
    return prop, missing


def _joined(c, head):
    """Rule 9b: a clause attached to `head` by advcl or ccomp that is a second main clause ("Don't search the web ...,
    just tell me ..."): before the head, a comma, semicolon or colon between them, no subordinating word or "to"."""
    return (c.dep_ in ("advcl", "ccomp") and c.pos_ in ("VERB", "AUX") and c.i < head.i
            and not any(k.dep_ == "mark" or k.tag_ == "TO" for k in c.children)
            and any(t.text in JOINS for t in c.doc[c.right_edge.i + 1:head.i]))


def _sentence(root, parser_id, scheme):
    """Rules 9, 9b: the root's clause first, then every clause attached to it by conj or parataxis (also to those) and
    every joined main clause, in text order. -> [(head token, proposition or None, problem or None)]"""
    first, problem = _clause(root, parser_id, scheme)
    out, todo = [(root, first, problem)], [(root, first)]
    while todo:
        head, prop = todo.pop(0)
        carried = (prop["roles"]["ARG0"], prop["mood"]) if prop and "ARG0" in prop["roles"] else None
        for c in head.children:
            if c.dep_ in ATTACHED and c.pos_ in ("VERB", "AUX"):
                p, why = _clause(c, parser_id, scheme, carried)
            elif _joined(c, head):
                p, why = _clause(c, parser_id, scheme)
            else:
                continue
            out.append((c, p, why))
            todo.append((c, p))
    return out[:1] + sorted(out[1:], key=lambda x: x[0].i)


def analyze(text):
    nlp, parser_id, scheme = _nlp()
    doc = nlp(text)
    lemma_source = parser_id
    if CSI_MORPH:
        from .dictionary import lemmatise
        morph, dic, lemma_source = _lemmatiser()
        for t in doc:
            if t.pos_ == "VERB" and t.tag_ in TAG_FORM:
                t.lemma_ = lemmatise(t.text.lower(), "verb", morph, dic, form_type=TAG_FORM[t.tag_])[0]
    tokens = [{"i": t.i, "idx": t.idx, "text": t.text, "lemma": t.lemma_, "upos": t.pos_, "tag": t.tag_,
               "feats": t.morph.to_dict(), "head": t.head.i, "dep": t.dep_} for t in doc]
    propositions, problems, unsupported, owner = [], [], [], {}
    for s_i, sent in enumerate(doc.sents):
        for head, prop, problem in _sentence(sent.root, parser_id, scheme):
            if problem:
                if s_i == 0 and head is sent.root:  # comparison reads the first sentence's main clause, as in v0.1
                    unsupported = [problem]
                problems.append({"token": head.i, "why": problem})
            if prop is None:
                continue
            owner[head.i] = owner[prop["predicate"]["token"]] = len(propositions)
            prop["provenance"]["lemmas"] = lemma_source
            if CSI_SENSES:
                prop["predicate"]["sense"] = _sense_of(doc, prop["predicate"]["token"])
                for role in prop["roles"].values():
                    if role.get("head") is not None:
                        role["head_sense"] = _sense_of(doc, role["head"])
                prop["provenance"]["senses"] = _senses().provenance["version"]
            propositions.append(prop)
    for tok in tokens:  # rule 11: the proposition of the nearest ancestor that is a proposition's predicate
        t = doc[tok["i"]]
        while t.i not in owner and t.head.i != t.i:
            t = t.head
        tok["prop"] = owner.get(t.i)
    reading = {"confidence": 1.0 if not unsupported else 0.5, "tokens": tokens, "propositions": propositions,
               "problems": problems}
    return {"csi_version": CSI_VERSION, "text": text, "readings": [reading], "ambiguity": [],
            "unsupported": unsupported, "confidence_note": "v0.1 confidence is uncalibrated (coverage only)"}
