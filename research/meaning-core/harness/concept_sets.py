"""What are the concept sets the paper trains on? For each (word, synonym) in the embedding-method sets, the WordNet
relation between them (see relation()). Also set sizes, coverage and cosine scores.
Usage: python concept_sets.py <synonyms_train.jsonl> <out.json>"""
import json
import sys
from collections import Counter

import nltk

try:
    from nltk.corpus import wordnet as wn
    wn.synsets("dog")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn


def relation(a, b):
    """First matching relation, in this order: same, inflection, synonym (shared synset), antonym (a lemma's WordNet
    antonym), near_synonym (adjective similar_to / also_see, verb_group), hypernym_or_hyponym (within 2 steps),
    co_hyponym (shared parent), derivation (derivationally related form), other; not_in_wordnet if either is absent."""
    a, b = a.lower(), b.lower()
    if a == b:
        return "same"
    la = {wn.morphy(a, p) for p in ("n", "v", "a", "r")} - {None}
    lb = {wn.morphy(b, p) for p in ("n", "v", "a", "r")} - {None}
    if la & lb:
        return "inflection"
    sa, sb = set(wn.synsets(a)), set(wn.synsets(b))
    if not sa or not sb:
        return "not_in_wordnet"
    if sa & sb:
        return "synonym"
    names_b = lb | {b}
    lemmas_a = [lem for s in sa for lem in s.lemmas()]
    if any(ant.name().lower() in names_b or ant.synset() in sb for lem in lemmas_a for ant in lem.antonyms()):
        return "antonym"
    near = {n for s in sa for n in s.similar_tos() + s.also_sees() + s.verb_groups()}
    near |= {n2 for n in list(near) for n2 in n.similar_tos()}
    if near & sb:
        return "near_synonym"
    up_a = {h for s in sa for h in s.hypernyms()} | {h2 for s in sa for h in s.hypernyms() for h2 in h.hypernyms()}
    up_b = {h for s in sb for h in s.hypernyms()} | {h2 for s in sb for h in s.hypernyms() for h2 in h.hypernyms()}
    if sb & up_a or sa & up_b:
        return "hypernym_or_hyponym"
    if {h for s in sa for h in s.hypernyms()} & {h for s in sb for h in s.hypernyms()}:
        return "co_hyponym"
    if any(d.name().lower() in names_b for lem in lemmas_a for d in lem.derivationally_related_forms()):
        return "derivation"
    return "other"


def main(path, out):
    rel, sizes, scores = Counter(), Counter(), []
    targets = with_syn = 0
    samples = {}
    for line in open(path, encoding="utf-8"):
        for t in json.loads(line)["content_word_responses"]:
            targets += 1
            syn = t.get("synonyms") or []
            sizes[len(syn)] += 1
            if syn:
                with_syn += 1
            for s, sc in zip(syn, t.get("scores") or [None] * len(syn)):
                r = relation(t["word"], s)
                rel[r] += 1
                if sc is not None:
                    scores.append(sc)
                samples.setdefault(r, [])
                if len(samples[r]) < 12:
                    samples[r].append(f"{t['word']} -> {s}")
    pairs = sum(rel.values())
    res = {
        "targets": targets, "coverage": round(with_syn / targets, 4),
        "mean_set_size": round(sum(k * v for k, v in sizes.items()) / targets, 2),
        "pairs": pairs, "relation_share": {k: round(v / pairs, 4) for k, v in rel.most_common()},
        "cosine_min_mean": [round(min(scores), 4), round(sum(scores) / len(scores), 4)] if scores else None,
        "samples": samples,
    }
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
