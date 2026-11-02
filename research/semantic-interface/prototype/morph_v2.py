"""Experiment 4c (Morphology v2) battery: criteria C1-C7 of EXPERIMENTS.md for the learned morphology component.
All splits are by lemma; every number in the output names its training and test sets.
Usage: python morph_v2.py <out json> --unimorph <eng> --ewt-train <conllu> --ewt-eval <dev> <test>
                          [--systems spaCy names/paths for C4] [--jobs 8]"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi.morphology import EditModel, Morphology, apply_edit, edit_of, load_unimorph  # noqa: E402

VERB_TYPES = ("past", "past_participle", "gerund", "third_singular")
VOWELS = set("aeiou")
DATA = {}


# ---------- data ----------
def setup(a):
    entries = load_unimorph(a.unimorph)
    by_class = collections.defaultdict(lambda: collections.defaultdict(dict))
    for l, f, c, t in entries:
        by_class[c][l][t] = f
    verbs4 = sorted(l for l, fs in by_class["verb"].items() if all(t in fs for t in VERB_TYPES))
    form_lemmas = collections.defaultdict(set)
    for l, f, c, t in entries:
        form_lemmas[(c, f)].add(l)
    edit_count = collections.Counter(edit_of(f, l) for l, f, c, t in entries if c == "verb")
    irregular = {l for l, f, c, t in entries if c == "verb" and edit_count[edit_of(f, l)] < 2}
    words = set()
    for l, f, c, t in entries:
        words.update((l, f))
    DATA.update(entries=entries, by_class=by_class, verbs4=verbs4, form_lemmas=form_lemmas, irregular=irregular,
                all_words=words, args=a)
    if getattr(a, "oewn", None):
        from csi.dictionary import Dictionary
        DATA["dictionary"] = Dictionary.load_oewn(a.oewn)


def verb_entries(lemmas):
    bc = DATA["by_class"]["verb"]
    return [(l, bc[l][t], "verb", t) for l in lemmas for t in VERB_TYPES if t in bc[l]]


def folds(items, k, seed):
    items = sorted(items)
    random.Random(seed).shuffle(items)
    return [items[i::k] for i in range(k)]


# ---------- scoring ----------
def score_analysis(m, test_entries, cls="verb"):
    acc = collections.defaultdict(lambda: [0, 0, 0])  # n, model-only correct, component correct
    errors = []
    for l, f, c, t in test_entries:
        model_pred = m.analyse(f, cls, use_exceptions=False)
        comp_pred = m.analyse(f, cls)
        amb = "ambiguous" if len(DATA["form_lemmas"][(cls, f)]) > 1 else "unambiguous"
        irr = "irregular" if l in DATA["irregular"] else "regular"
        for key in ("all", t, amb, irr):
            a = acc[key]
            a[0] += 1
            a[1] += model_pred == l
            a[2] += comp_pred == l
        if comp_pred != l and len(errors) < 40:
            errors.append(f"{f} -> {comp_pred} (want {l}; {amb}, {irr})")
    return {k: {"n": v[0], "model": round(v[1] / v[0], 4), "component": round(v[2] / v[0], 4)}
            for k, v in sorted(acc.items())}, errors


def rule_lemma(form):
    """Hand-written orthographic rules, no lexicon (the C5 comparison)."""
    if len(form) > 4 and form.endswith(("ies", "ied")):
        return form[:-3] + "y"
    for suf in ("ing", "ed"):
        if form.endswith(suf) and len(form) > len(suf) + 1:
            base = form[:-len(suf)]
            if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "aeiouslzf":
                return base[:-1]
            if (len(base) >= 2 and base[-1] not in VOWELS | set("wxy") and base[-2] in VOWELS
                    and (len(base) < 3 or base[-3] not in VOWELS)):
                return base + "e"
            if base.endswith(("v", "c", "u")) or (len(base) >= 2 and base[-1] in "sz" and base[-2] in VOWELS):
                return base + "e"
            return base
    if form.endswith(("sses", "shes", "ches", "xes", "zes")):
        return form[:-2]
    if form.endswith("s") and not form.endswith("ss"):
        return form[:-1]
    return form


def seeds():
    a = DATA["args"]
    return getattr(a, "fold_seed", 11), getattr(a, "ending_seed", 13), getattr(a, "nonce_seed", 19)


def train_verbs(lemmas, **kw):
    return Morphology.train(verb_entries(lemmas), classes=("verb",),
                            feature_set=getattr(DATA["args"], "feature_set", "suffix"), **kw)


# ---------- tasks (each returns (name, result)) ----------
def task_c1_fold(i):
    fs = folds(DATA["verbs4"], 5, seeds()[0])
    test = set(fs[i])
    train = [l for l in DATA["verbs4"] if l not in test]
    t0 = time.time()
    m = train_verbs(train, generate=False)
    table, errors = score_analysis(m, verb_entries(sorted(test)))
    rule = sum(rule_lemma(f) == l for l, f, _, _ in verb_entries(sorted(test))) / len(verb_entries(sorted(test)))
    return f"c1_fold{i}", {"n_train_lemmas": len(train), "n_test_lemmas": len(test), "table": table,
                           "rule_baseline": round(rule, 4), "errors": errors, "seconds": round(time.time() - t0)}


def task_4b_replication(_):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_e2 import ewt_lemmas
    seen_any, seen_verb = ewt_lemmas(DATA["args"].ewt_train)
    bc = DATA["by_class"]["verb"]
    verbs = {l: bc[l] for l in DATA["verbs4"]}
    rng = random.Random(7)
    seen = sorted(l for l in verbs if l in seen_verb)
    unseen = sorted(l for l in verbs if l not in seen_any and all(f not in seen_any for f in verbs[l].values()))
    split = {"seen": rng.sample(seen, min(500, len(seen))), "unseen": rng.sample(unseen, min(500, len(unseen)))}
    held = set(split["seen"]) | set(split["unseen"])
    from csi.minimal_pairs import ANIMATE_VERBS, DITRANSITIVE, TRANSITIVE
    lexicons_at_4b = ("seen", "unseen", "heldout2")  # heldout3 was added after 4b ran
    held |= {v[0] for d in (TRANSITIVE, ANIMATE_VERBS, DITRANSITIVE) for k in lexicons_at_4b for v in d[k]}
    m = Morphology.train(verb_entries([l for l in DATA["verbs4"] if l not in held]), classes=("verb",), generate=False)
    return "replication_4b", {sp: score_analysis(m, verb_entries(ls))[0]["all"] for sp, ls in split.items()}


def task_c2(k=3):
    k = k or 3
    ends = folds(sorted({l[-k:] for l in DATA["verbs4"]}), 5, seeds()[1])[0]
    held = set(ends)
    test = [l for l in DATA["verbs4"] if l[-k:] in held]
    train = [l for l in DATA["verbs4"] if l[-k:] not in held]
    m = train_verbs(train, generate=False)
    table, errors = score_analysis(m, verb_entries(test))
    return f"c2_novel_endings{'' if k == 3 else f'_k{k}'}", {"ending_letters": k, "held_out_endings": len(held),
                                                            "n_train_lemmas": len(train), "n_test_lemmas": len(test),
                                                            "table": table, "errors": errors[:25]}


NONCE = {  # class -> (lemma builder, form rules)
    "doubling": (lambda o, v, rng: o + v + rng.choice("bdgmnpt"), lambda l: (l + l[-1] + "ed", l + l[-1] + "ing", l + "s")),
    "cluster": (lambda o, v, rng: o + v + rng.choice(["nd", "nt", "st", "mp", "lt", "ft", "sk", "lk", "rn", "rd"]),
                lambda l: (l + "ed", l + "ing", l + "s")),
    "silent_e": (lambda o, v, rng: o + rng.choice("aiou") + rng.choice("bdkmnptvz") + "e",
                 lambda l: (l + "d", l[:-1] + "ing", l + "s")),
    "consonant_y": (lambda o, v, rng: o + v + rng.choice(["bb", "dd", "gg", "mm", "nn", "pp", "tt", "rr"]) + "y",
                    lambda l: (l[:-1] + "ied", l + "ing", l[:-1] + "ies")),
    "sibilant": (lambda o, v, rng: o + v + rng.choice(["sh", "ch", "x", "ss"]), lambda l: (l + "ed", l + "ing", l + "es")),
    "ck": (lambda o, v, rng: o + v + "ck", lambda l: (l + "ed", l + "ing", l + "s")),
    "vowel_wy": (lambda o, v, rng: o + rng.choice("aeo") + rng.choice("wy"), lambda l: (l + "ed", l + "ing", l + "s")),
}
ONSETS = ["b", "bl", "br", "d", "dr", "f", "fl", "fr", "g", "gl", "gr", "k", "cl", "cr", "j", "l", "m", "n", "p", "pl",
          "pr", "sk", "sl", "sm", "sn", "sp", "st", "sw", "t", "tr", "tw", "v", "z", "sh", "ch", "th",
          "h", "r", "w", "scr", "spr", "str", "spl", "shr", "thr", "squ", "dw"]


def nonce_verbs(per_class=100, seed=None, max_tries=200000, exclude=frozenset()):
    """Up to per_class invented verbs per spelling class, none of whose forms occurs in UniMorph (count reported)."""
    rng = random.Random(seeds()[2] if seed is None else seed)
    known = DATA["all_words"] | exclude if exclude else DATA["all_words"]
    out = {}
    for name, (build, rules) in NONCE.items():
        got, tries = {}, 0
        while len(got) < per_class and tries < max_tries:
            tries += 1
            l = build(rng.choice(ONSETS), rng.choice("aeiou"), rng)
            forms = rules(l)
            if l in got or l in known or any(f in known for f in forms):
                continue
            got[l] = {"past": forms[0], "past_participle": forms[0], "gerund": forms[1], "third_singular": forms[2]}
        out[name] = got
    return out


def task_c3_c4_full(_):
    """C3 (nonce analysis + generation) and C4 (EWT real text) with the component trained on ALL UniMorph verbs."""
    entries = [e for e in DATA["entries"] if e[2] == "verb"]
    m = Morphology.train(entries, classes=("verb",), generate=True,
                         feature_set=getattr(DATA["args"], "feature_set", "suffix"))
    nonce = nonce_verbs()
    c3 = {}
    for name, lemmas in nonce.items():
        n = ok_a = ok_g = 0
        errs = []
        for l, forms in lemmas.items():
            for t, f in forms.items():
                n += 1
                pa = m.analyse(f, "verb")
                pg = m.generate(l, "verb", t)
                ok_a += pa == l
                ok_g += pg == f
                if (pa != l or pg != f) and len(errs) < 6:
                    errs.append(f"{l}/{t}: analyse {f}->{pa}; generate ->{pg} (want {f})")
        c3[name] = {"n": n, "analysis": round(ok_a / n, 4), "generation": round(ok_g / n, 4), "errors": errs}
    tot = sum(v["n"] for v in c3.values())
    c3["_all"] = {"analysis": round(sum(v["analysis"] * v["n"] for v in c3.values()) / tot, 4),
                  "generation": round(sum(v["generation"] * v["n"] for v in c3.values()) / tot, 4), "n": tot,
                  "rule_baseline": round(sum(rule_lemma(f) == l for lem in nonce.values() for l, fs in lem.items()
                                             for f in fs.values()) / tot, 4)}
    # C4: EWT verb tokens
    from run_e3 import read_conllu
    sents = [s for p in DATA["args"].ewt_eval for s in read_conllu(p)]
    targets = [(si, i, w["form"].lower(), w["lemma"].lower()) for si, s in enumerate(sents) for i, w in enumerate(s)
               if w["upos"] == "VERB" and w["xpos"] in ("VBD", "VBN", "VBG", "VBZ") and re.fullmatch(r"[A-Za-z]+", w["form"])]
    verb_lemmas = set(DATA["by_class"]["verb"])
    c4 = {"n_tokens": len(targets), "n_in_unimorph": sum(g in verb_lemmas for *_, g in targets), "systems": {}}

    def tally(preds):
        out = {}
        for key, sel in (("all", lambda g: True), ("lemma_in_unimorph", lambda g: g in verb_lemmas),
                         ("lemma_not_in_unimorph", lambda g: g not in verb_lemmas)):
            pairs = [(p, g) for p, (*_, g) in zip(preds, targets) if sel(g)]
            out[key] = {"n": len(pairs), "acc": round(sum(p == g for p, g in pairs) / max(1, len(pairs)), 4)}
        return out
    comp = [m.analyse(f, "verb") for _, _, f, _ in targets]
    c4["systems"]["csi_morphology"] = tally(comp)
    c4["component_errors"] = sorted({f"{f} -> {p} (want {g})" for p, (_, _, f, g) in zip(comp, targets) if p != g})[:40]
    import spacy
    from spacy.tokens import Doc
    for name in DATA["args"].systems:
        nlp = spacy.load(name)
        docs = list(nlp.pipe([Doc(nlp.vocab, words=[w["form"] for w in s], spaces=[w["space"] for w in s]) for s in sents],
                             batch_size=64))
        c4["systems"][name] = tally([docs[si][i].lemma_.lower() for si, i, _, _ in targets])
    return "c3_c4_full", {"c3_nonce": c3, "c4_ewt": c4, "exceptions": len(m.analyse_exceptions["verb"]),
                          "nonce_examples": {k: list(v.items())[:3] for k, v in nonce.items()}}


def task_c5(_):
    fs = folds(DATA["verbs4"], 5, seeds()[0])
    test = verb_entries(sorted(fs[0]))
    train_lemmas = [l for l in DATA["verbs4"] if l not in set(fs[0])]
    out = {"curve": {}, "ablation_max_len": {}}
    rng = random.Random(17)
    shuffled = train_lemmas[:]
    rng.shuffle(shuffled)
    for frac in (0.01, 0.03, 0.1, 0.3, 1.0):
        sub = shuffled[:max(1, int(frac * len(shuffled)))]
        m = train_verbs(sub, generate=False)
        out["curve"][str(frac)] = {"n_train_lemmas": len(sub), "model": score_analysis(m, test)[0]["all"]["model"]}
    for ml in (2, 3, 4):
        m = train_verbs(train_lemmas, max_len=ml, generate=False)
        out["ablation_max_len"][str(ml)] = score_analysis(m, test)[0]["all"]["model"]
    pairs = [(f, l) for l, f, _, _ in verb_entries(train_lemmas)]
    edits = [edit_of(f, l) for f, l in pairs]
    perm = edits[:]
    random.Random(23).shuffle(perm)
    shuffled_model = EditModel.train(pairs, edits=perm, feature_set=getattr(DATA["args"], "feature_set", "suffix"))
    out["shuffled_label_control"] = round(sum(shuffled_model.predict(f) == l for l, f, _, _ in test) / len(test), 4)
    common = collections.Counter(edits).most_common(1)[0][0]
    out["majority_edit_baseline"] = round(sum(apply_edit(f, common) == l for l, f, _, _ in test) / len(test), 4)
    by_type = {}
    for (f, l), e, (_, _, _, t) in zip(pairs, edits, verb_entries(train_lemmas)):
        by_type.setdefault(t, collections.Counter())[e] += 1
    out["majority_edit_per_form_type"] = round(sum(apply_edit(f, by_type[t].most_common(1)[0][0]) == l
                                                   for l, f, _, t in test) / len(test), 4)
    out["rule_baseline"] = round(sum(rule_lemma(f) == l for l, f, _, _ in test) / len(test), 4)
    return "c5_learned_not_trivial", out


def task_c6_fold(i):
    fs = folds(DATA["verbs4"], 5, seeds()[0])
    test = set(fs[i])
    train = [l for l in DATA["verbs4"] if l not in test]
    m = train_verbs(train, generate=True)
    acc = collections.defaultdict(lambda: [0, 0, 0])
    cats = collections.Counter()
    for l, f, _, t in verb_entries(sorted(test)):
        pm, pc = m.generate(l, "verb", t, use_exceptions=False), m.generate(l, "verb", t)
        for key in ("all", t):
            acc[key][0] += 1
            acc[key][1] += pm == f
            acc[key][2] += pc == f
        if pc != f:
            if len(pc) == len(f) + 1 or len(f) == len(pc) + 1:
                cats["doubling or e (one letter)"] += 1
            elif l in DATA["irregular"]:
                cats["irregular"] += 1
            else:
                cats["other"] += 1
    return f"c6_fold{i}", {"table": {k: {"n": v[0], "model": round(v[1] / v[0], 4), "component": round(v[2] / v[0], 4)}
                                     for k, v in acc.items()}, "error_categories": dict(cats)}


def task_c7(spec):
    cls, i = spec
    lemmas = sorted(DATA["by_class"][cls])
    test = set(folds(lemmas, 5, 29)[i])
    ent = [e for e in DATA["entries"] if e[2] == cls]
    m = Morphology.train([e for e in ent if e[0] not in test], classes=(cls,), generate=False,
                         feature_set=getattr(DATA["args"], "feature_set", "suffix"))
    table, errors = score_analysis(m, [e for e in ent if e[0] in test], cls)
    return f"c7_{cls}_fold{i}", {"table": {k: v for k, v in table.items() if k in ("all", "ambiguous", "unambiguous")},
                                 "errors": errors[:15]}


def run(task):
    fn, arg = task
    return globals()[fn](arg)


# ---------- Experiment 4g: layered lemmatiser (Grammar class + Morphology candidates + Dictionary) ----------
TAG_CLASS = {"VBD": "verb", "VBN": "verb", "VBG": "verb", "VBZ": "verb", "NNS": "noun", "JJR": "adj", "JJS": "adj"}
CLASS_UPOS = {"verb": "VERB", "noun": "NOUN", "adj": "ADJ"}


def treebank_targets(paths):
    from run_e3 import read_conllu
    sents = [s for p in paths for s in read_conllu(p)]
    targets = [(si, i, TAG_CLASS[w["xpos"]], w["form"].lower(), w["lemma"].lower())
               for si, s in enumerate(sents) for i, w in enumerate(s)
               if w["xpos"] in TAG_CLASS and w["upos"] == CLASS_UPOS[TAG_CLASS[w["xpos"]]]
               and re.fullmatch(r"[A-Za-z]+", w["form"])]
    return sents, targets


def task_g_layered(_):
    from csi.dictionary import lemmatise
    dic = DATA["dictionary"]
    m = Morphology.train([e for e in DATA["entries"] if e[2] in ("verb", "noun", "adj")], generate=False,
                         feature_set=getattr(DATA["args"], "feature_set", "suffix"))
    import spacy
    from spacy.tokens import Doc
    nlps = {name: spacy.load(name) for name in DATA["args"].systems}
    out = {}
    for spec in DATA["args"].fresh:
        name, paths = spec.split("=", 1)
        sents, targets = treebank_targets(paths.split(","))
        preds = {"layered": [], "morphology_alone": [], "component_4c": []}
        sources = collections.Counter()
        for _, _, c, f, _ in targets:
            lem, src = lemmatise(f, c, m, dic)
            preds["layered"].append(lem)
            sources[src] += 1
            preds["morphology_alone"].append(m.analyse(f, c, use_exceptions=False))
            preds["component_4c"].append(m.analyse(f, c))
        for sysname, nlp in nlps.items():
            docs = list(nlp.pipe([Doc(nlp.vocab, words=[w["form"] for w in s], spaces=[w["space"] for w in s])
                                  for s in sents], batch_size=64))
            preds[sysname] = [docs[si][i].lemma_.lower() for si, i, *_ in targets]
        res = {"n_sentences": len(sents), "layer_source": dict(sources), "by_class": {}}
        for c in ("verb", "noun", "adj"):
            idx = [k for k, t in enumerate(targets) if t[2] == c]
            gold = [targets[k][4] for k in idx]
            res["by_class"][c] = {"n": len(idx), "gold_known_to_dictionary": round(
                sum(dic.knows(g, c) for g in gold) / max(1, len(idx)), 4), "systems": {
                s: round(sum(p[k] == g for k, g in zip(idx, gold)) / max(1, len(idx)), 4) for s, p in preds.items()}}
        cats, errs = collections.Counter(), []
        for k, (_, _, c, f, g) in enumerate(targets):
            if c != "verb" or preds["layered"][k] == g:
                continue
            if not dic.knows(g, c):
                cat = "gold unknown to dictionary"
            elif g not in m.candidates(f, c) and g not in dic.listed(f, c):
                cat = "gold not among morphology candidates"
            else:
                cat = "another known candidate chosen"
            cats[cat] += 1
            if len(errs) < 40:
                errs.append(f"{f} -> {preds['layered'][k]} (want {g}; {cat})")
        res["verb_errors_by_layer"], res["verb_errors"] = dict(cats), errs
        out[name] = res
    return "g_layered", out


def task_g_nonce(_):
    from csi.dictionary import lemmatise
    dic = DATA["dictionary"]
    m = Morphology.train([e for e in DATA["entries"] if e[2] == "verb"], classes=("verb",), generate=False,
                         feature_set=getattr(DATA["args"], "feature_set", "suffix"))
    nonce = nonce_verbs(seed=DATA["args"].g_nonce_seed, exclude=frozenset(dic.words()))
    out = {}
    for name, lemmas in nonce.items():
        n = ok_l = ok_m = 0
        errs = []
        for l, forms in lemmas.items():
            for f in forms.values():
                n += 1
                pl, pm = lemmatise(f, "verb", m, dic)[0], m.analyse(f, "verb", use_exceptions=False)
                ok_l += pl == l
                ok_m += pm == l
                if pl != l and len(errs) < 6:
                    errs.append(f"{f} -> {pl} (want {l}; morphology alone {pm})")
        out[name] = {"n": n, "n_lemmas": len(lemmas), "layered": round(ok_l / n, 4),
                     "morphology_alone": round(ok_m / n, 4), "errors": errs}
    tot = sum(v["n"] for v in out.values())
    out["_all"] = {k: round(sum(v[k] * v["n"] for v in out.values()) / tot, 4) for k in ("layered", "morphology_alone")}
    out["_all"]["n"] = tot
    return "g_nonce", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--unimorph", required=True)
    ap.add_argument("--ewt-train", required=True)
    ap.add_argument("--ewt-eval", nargs="+", required=True)
    ap.add_argument("--systems", nargs="*", default=[])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tasks", nargs="*", default=None, help="run only these task functions; merge into an existing out")
    ap.add_argument("--feature-set", default="suffix", choices=["suffix", "suffix+shape"])
    ap.add_argument("--fold-seed", type=int, default=11)
    ap.add_argument("--ending-seed", type=int, default=13)
    ap.add_argument("--nonce-seed", type=int, default=19)
    ap.add_argument("--oewn", default=None, help="OEWN LMF xml(.gz): the dictionary layer for the 4g tasks")
    ap.add_argument("--fresh", nargs="*", default=[], help="4g real-text sets as name=conllu[,conllu]")
    ap.add_argument("--g-nonce-seed", type=int, default=113)
    a = ap.parse_args()
    setup(a)
    tasks = ([("task_c3_c4_full", None), ("task_c5", None), ("task_c2", 3), ("task_c2", 4), ("task_4b_replication", None)]
             + [("task_c1_fold", i) for i in range(5)] + [("task_c6_fold", i) for i in range(5)]
             + [("task_c7", ("noun", i)) for i in range(5)] + [("task_c7", ("adj", i)) for i in range(5)])
    if a.tasks:
        tasks = [t for t in tasks + [("task_g_layered", None), ("task_g_nonce", None)] if t[0] in a.tasks]
    t0 = time.time()
    res = json.load(open(a.out)) if a.tasks and os.path.exists(a.out) else {}
    g = any(t[0].startswith("task_g") for t in tasks)
    res.update({"experiment": "E4g Morphology v3 (layered)" if g else "E4c Morphology v2", "n_verb_lemmas_4forms": len(DATA["verbs4"]),
                "n_noun_lemmas": len(DATA["by_class"]["noun"]), "n_adj_lemmas": len(DATA["by_class"]["adj"]),
                "n_irregular_verb_lemmas": len(DATA["irregular"]), "feature_set": a.feature_set,
                "seeds": {"folds": a.fold_seed, "endings": a.ending_seed, "nonce": a.nonce_seed}})
    with mp.get_context("fork").Pool(a.jobs) as pool:
        for name, r in pool.imap_unordered(run, tasks):
            res[name] = r
            print(name, json.dumps(r)[:400], flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
