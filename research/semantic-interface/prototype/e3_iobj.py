"""Experiment 2 diagnostic: what each parser does with gold indirect objects (UD iobj under a verb) on EWT dev+test,
split into double-object clauses (iobj + obj under the same verb, the construction of the Experiment 1 errors) and
other iobj (recipient with a clausal complement or alone). Usage:
python e3_iobj.py <out json> <dev.conllu> <test.conllu> <system> [system ...]"""
import collections
import json
import sys

from run_e3 import read_conllu


def main():
    import spacy
    from spacy.tokens import Doc
    out, dev, test, systems = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    sents = [s for p in (dev, test) for s in read_conllu(p)]
    picked = []
    for s in sents:
        for i, w in enumerate(s):
            if w["deprel"] == "iobj" and w["head"] >= 0 and s[w["head"]]["upos"] == "VERB":
                theme = next((j for j, x in enumerate(s) if x["deprel"] == "obj" and x["head"] == w["head"]), None)
                picked.append((s, i, theme))
    res = {"n_iobj": len(picked), "n_double_object": sum(t is not None for _, _, t in picked), "systems": {}}
    for name in systems:
        nlp = spacy.load(name)
        conf = {"double_object": collections.Counter(), "other": collections.Counter()}
        theme_conf = collections.Counter()
        examples = []
        for s, i, theme in picked:
            doc = nlp(Doc(nlp.vocab, words=[w["form"] for w in s], spaces=[w["space"] for w in s]))
            t = doc[i]
            same_head = t.head.i == s[i]["head"]
            key = f"{t.dep_}{'' if same_head else ' (other head)'}"
            conf["double_object" if theme is not None else "other"][key] += 1
            if theme is not None:
                th = doc[theme]
                theme_conf[f"{th.dep_}{'' if th.head.i == s[theme]['head'] else ' (other head)'}"] += 1
                if len(examples) < 12 and t.dep_ != "dative" and t.dep_ != "iobj":
                    examples.append({"text": doc.text, "recipient": t.text, "got": key, "theme": th.text,
                                     "theme_got": th.dep_})
        res["systems"][name] = {"recipient_in_double_object": dict(conf["double_object"].most_common()),
                                "recipient_other": dict(conf["other"].most_common()),
                                "theme_in_double_object": dict(theme_conf.most_common()), "examples": examples}
        print(name, json.dumps({k: v for k, v in res["systems"][name].items() if k != "examples"}), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
