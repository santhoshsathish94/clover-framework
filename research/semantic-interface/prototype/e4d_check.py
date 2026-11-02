"""Experiment 4d/4h check: predicate mismatches on a nonce lexicon (a sentence whose CSI predicate is an inflected form
of one of its invented verbs but whose predicate lemma is not that verb's base form). Uses CSI_PARSER / CSI_MORPH.
Usage: python e4d_check.py <out json> [lexicon, default nonce1]"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi.analyze import CSI_MORPH, PARSER, analyze  # noqa: E402
from csi.minimal_pairs import ANIMATE_VERBS, TRANSITIVE, generate  # noqa: E402


def main():
    lex = sys.argv[2] if len(sys.argv) > 2 else "nonce1"
    base_of = {f: v[0] for v in TRANSITIVE[lex] + ANIMATE_VERBS[lex] for f in v[1:]}
    sentences = sorted({t for r in generate(lex) for t in (r["a"], r["b"])})
    n = mismatches = not_predicate = 0
    errors, misses = [], []
    for s in sentences:
        if not any(w.strip(".").lower() in base_of for w in s.split()):
            continue
        n += 1
        reading = analyze(s)["readings"][0]
        props = reading["propositions"]
        tok = reading["tokens"][props[0]["predicate"]["token"]]["text"].lower() if props else None
        if tok not in base_of:
            not_predicate += 1
            if len(misses) < 15:
                misses.append(f"{s} -> predicate {tok}")
            continue
        if props[0]["predicate"]["lemma"] != base_of[tok]:
            mismatches += 1
            errors.append(f"{s} -> {props[0]['predicate']['lemma']} (want {base_of[tok]})")
    out = {"parser": PARSER, "lemmas": CSI_MORPH or "parser", "lexicon": lex, "sentences_with_inflected_nonce_verb": n,
           "nonce_verb_not_the_predicate": not_predicate, "predicate_mismatches": mismatches,
           "mismatches": errors, "not_predicate_examples": misses}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("mismatches", "not_predicate_examples")}))


if __name__ == "__main__":
    main()
