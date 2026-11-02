"""Experiment 3c2 runner (EXPERIMENTS.md). CSI, whose version comes from the environment (CSI_SENSES; CSI_CLOSENESS = a
threshold, or "record" to collect closeness without accepting it), or --bge (sentence cosines), on the 3c2 dev or
test pairs, the 3c (P) pairs and the E1 regression sets. --sheet writes the label-audit sheet instead (instance 1 of
every row type per item; no system runs).
Usage: python run_3c2.py <lexicon json> <out> --sets dev,test,p3c,heldout3,nonce2 [--audit e3c2_audit.json]
       [--lex3c e3c_lexicon.json] [--bge] | python run_3c2.py <lexicon json> <sheet txt> --sheet"""
import argparse
import functools
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16] if path else None


def pairs(items, fillers, seed=3112, n=3):
    """Sentence pairs per item, n distinct filler choices; each item has its own random stream, so dropping an item
    changes no other sentence."""
    people, things, neutral = fillers["people"], fillers["things"], fillers["neutral_verbs"]
    rows = []
    for it in items:
        rng = random.Random(f"{seed}:{it['id']}")
        if it["pos"] == "v":
            choices = [(agent, None) for agent in rng.sample(people, n)]
        else:
            choices = rng.sample([(v, f) for v in neutral for f in (things if it["class"] == "agent" else people)], n)
        for inst, (x, filler) in enumerate(choices):
            if it["pos"] == "v":  # the object is OEWN's own frame wording, so no filler clashes with the verb's meaning
                f, agent = it["forms"], x
                a, b, r, q = f[it["a"]], f[it["b"]], f[it["related"]], f[it["random"]]
                obj = "somebody" if it["class"] == "verb_person" else "something"
                base = f"The {agent} {a[1]} {obj}."
                out = [(f"The {agent} {b[1]} {obj}.", 1, "equiv:synonym_predicate", b[0]),
                       (f"{obj.capitalize()} was {b[2]} by the {agent}.", 1, "equiv:synonym_predicate_passive", b[0])]
                for kind, c in (("related", r), ("random", q)):
                    out += [(f"The {agent} {c[1]} {obj}.", 0, f"dist:{kind}_predicate", c[0]),
                            (f"{obj.capitalize()} was {c[2]} by the {agent}.", 0, f"dist:{kind}_predicate_passive", c[0])]
            else:
                v = x

                def mk(w):
                    return f"The {w} {v[1]} the {filler}." if it["class"] == "agent" else f"The {filler} {v[1]} the {w}."
                base = mk(it["a"])
                out = [(mk(it["b"]), 1, f"equiv:synonym_{it['class']}", it["b"]),
                       (mk(it["related"]), 0, f"dist:related_{it['class']}", it["related"]),
                       (mk(it["random"]), 0, f"dist:random_{it['class']}", it["random"])]
            rows += [{"a": base, "b": s, "label": y, "phenomenon": ph, "item": it["id"], "class": it["class"],
                      "instance": inst, "words": [it["a"], w]} for s, y, ph, w in out]
    return rows


def write_sheet(lex, path):
    with open(path, "w", encoding="utf-8") as fh:
        for it in lex["items"]:
            fh.write(f"== {it['id']}\n")
            for r in pairs([it], lex["fillers"], n=1):
                fh.write(f"{'SAME' if r['label'] else 'DIFF'}  {r['a']}  ||  {r['b']}\n")


def summarize(rows, preds):
    from run_e1 import balanced_acc, per_phenomenon
    y = [r["label"] for r in rows]
    scored = [p if p != -1 else 1 - t for p, t in zip(preds, y)]
    return {"n": len(rows), "balanced_accuracy": round(balanced_acc(y, scored), 4),
            "coverage": round(sum(p != -1 for p in preds) / len(rows), 4),
            "per_phenomenon_accuracy": per_phenomenon(rows, scored)}


def run_csi(rows):
    import importlib
    cc = importlib.import_module("csi.compare")  # the module: csi/__init__ exports a function of the same name
    cc.analyze = functools.lru_cache(maxsize=None)(cc.analyze)  # sentences repeat across pairs; analysis is deterministic
    out = []
    for r in rows:
        c = cc.compare(r["a"], r["b"])
        out.append({"a": r["a"], "b": r["b"], "label": r["label"], "phenomenon": r["phenomenon"], "item": r.get("item"),
                    "class": r.get("class"), "words": r.get("words"),
                    "pred": -1 if c["equivalent"] is None else int(c["equivalent"]),
                    "differences": [d["type"] for d in c["differences"]], "lexical": c["lexical"]})
    return {**summarize(rows, [o["pred"] for o in out]), "rows": out}


def run_bge(sets):
    from run_3c import BGE_REV, bge_encoder, cosines
    from run_e1 import balanced_acc
    from csi.minimal_pairs import generate
    enc = bge_encoder()
    seen = generate("seen")
    cs, ys = cosines(enc, seen), [r["label"] for r in seen]
    e1_tau = max(sorted(set(cs)), key=lambda t: balanced_acc(ys, [int(c >= t) for c in cs]))
    res = {"system": f"bge-small {BGE_REV[:8]} cosine", "threshold_from_e1_seen": round(e1_tau, 6)}
    for name, rows in sets.items():
        cos = cosines(enc, rows)
        res[name] = {**summarize(rows, [int(c >= e1_tau) for c in cos]),
                     "rows": [{"label": r["label"], "phenomenon": r["phenomenon"], "item": r.get("item"),
                               "class": r.get("class"), "cos": round(c, 6)} for r, c in zip(rows, cos)]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lexicon")
    ap.add_argument("out")
    ap.add_argument("--sets", default="dev")
    ap.add_argument("--audit")
    ap.add_argument("--lex3c")
    ap.add_argument("--bge", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    lex = json.load(open(a.lexicon))
    if a.sheet:
        return write_sheet(lex, a.out)
    flagged = set(json.load(open(a.audit))["flagged"]) if a.audit else set()
    sets = {}
    for name in a.sets.split(","):
        if name in ("dev", "test"):
            sets[name] = pairs([it for it in lex["items"] if it["split"] == name and it["id"] not in flagged],
                               lex["fillers"])
        elif name == "p3c":
            from run_3c import pairs as pairs_3c
            lex3c = json.load(open(a.lex3c))
            sets[name] = pairs_3c(lex3c["P"], lex3c["fillers"])
        else:
            from csi.minimal_pairs import generate
            sets[name] = generate(name)
    prov = {"lexicon_sha256": sha(a.lexicon), "audit_sha256": sha(a.audit), "flagged": sorted(flagged),
            "sizes": {k: len(v) for k, v in sets.items()}}
    if a.bge:
        res = {**run_bge(sets), **prov}
    else:
        from csi import CSI_VERSION
        res = {"system": f"CSI {CSI_VERSION}", **{k: os.environ.get(k) for k in
                                                  ("CSI_PARSER", "CSI_MORPH", "CSI_SENSES", "CSI_CLOSENESS")}, **prov}
        for name, rows in sets.items():
            res[name] = run_csi(rows)
            print(name, {k: v for k, v in res[name].items() if k != "rows"}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
