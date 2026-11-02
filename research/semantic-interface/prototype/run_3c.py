"""Experiment 3c runner: CSI (v0.1 or v0.2-lexical, chosen by the environment: CSI_SENSES) on the 3c pairs (P, A) and
the regression sets (E1 heldout3, nonce2), or with --bge the bge-small cosine baseline (threshold from E1 seen).
Usage: python run_3c.py <lexicon json> <out json> [--bge]"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi import CSI_VERSION, compare  # noqa: E402
from csi.minimal_pairs import generate  # noqa: E402
from run_e1 import balanced_acc, per_phenomenon  # noqa: E402

BGE, BGE_REV = "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # as in E1 (memory_index.EMBED_REV)


def pairs(lex, fillers, seed=3106, n=3):
    rng = random.Random(seed)
    people, things, neutral = fillers["people"], fillers["things"], fillers["neutral_verbs"]
    rows = []
    for key in ("verbs", "animate_verbs"):
        for it in lex.get(key, []):
            a, b, c = it["a"], it["b"], it["control"]
            for _ in range(n):
                agent = rng.choice(people)
                patient = it["patient"] or (rng.choice([p for p in people if p != agent]) if key == "animate_verbs"
                                            else rng.choice(things))
                base = f"The {agent} {a[1]} the {patient}."
                rows += [(base, f"The {agent} {b[1]} the {patient}.", 1, "equiv:synonym_predicate", it["shared"]),
                         (base, f"The {patient} was {b[2]} by the {agent}.", 1, "equiv:synonym_predicate_passive", it["shared"]),
                         (base, f"The {agent} {c[1]} the {patient}.", 0, "dist:other_predicate", it["shared"]),
                         (base, f"The {patient} was {c[2]} by the {agent}.", 0, "dist:other_predicate_passive", it["shared"])]
    for key, slot in (("agents", "agent"), ("patients", "patient")):
        for it in lex.get(key, []):
            for _ in range(n):
                v = rng.choice(neutral)
                if slot == "agent":
                    filler = rng.choice(things)
                    mk = lambda w: f"The {w} {v[1]} the {filler}."  # noqa: E731
                else:
                    filler = rng.choice(people)
                    mk = lambda w: f"The {filler} {v[1]} the {w}."  # noqa: E731
                rows += [(mk(it["a"]), mk(it["b"]), 1, f"equiv:synonym_{slot}", it["shared"]),
                         (mk(it["a"]), mk(it["control"]), 0, f"dist:other_{slot}", it["shared"])]
    return [{"a": a, "b": b, "label": y, "phenomenon": p, "shared": s} for a, b, y, p, s in rows]


def run_csi(rows):
    pred, errors = [], []
    for r in rows:
        c = compare(r["a"], r["b"])
        p = -1 if c["equivalent"] is None else int(c["equivalent"])
        pred.append(p)
        if p != r["label"]:
            senses = []
            for an in c.get("analyses", []):
                prop = an["readings"][0]["propositions"][0] if an["readings"][0]["propositions"] else {}
                senses.append({"predicate": [prop.get("predicate", {}).get("lemma"), prop.get("predicate", {}).get("sense")],
                               "roles": {k: [v.get("head_lemma"), v.get("head_sense")] for k, v in prop.get("roles", {}).items()}})
            errors.append({**{k: r[k] for k in ("a", "b", "label", "phenomenon")}, "pred": p,
                           "differences": c["differences"], "notes": c["notes"], "senses": senses})
    y = [r["label"] for r in rows]
    scored = [p if p != -1 else 1 - t for p, t in zip(pred, y)]
    return {"n": len(rows), "balanced_accuracy": round(balanced_acc(y, scored), 4),
            "coverage": round(sum(p != -1 for p in pred) / len(rows), 4),
            "per_phenomenon_accuracy": per_phenomenon(rows, pred), "correct": [int(s == t) for s, t in zip(scored, y)],
            "n_errors": len(errors), "errors": errors}


def bge_encoder():
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BGE, revision=BGE_REV)
    model = AutoModel.from_pretrained(BGE, revision=BGE_REV).eval()

    def enc(texts):
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 128):
                b = tok(texts[i:i + 128], padding=True, truncation=True, max_length=512, return_tensors="pt")
                out.append(torch.nn.functional.normalize(model(**b).last_hidden_state[:, 0], dim=-1).numpy())
        return np.concatenate(out)
    return enc


def cosines(enc, rows):
    texts = sorted({t for r in rows for t in (r["a"], r["b"])})
    v = enc(texts)
    idx = {t: i for i, t in enumerate(texts)}
    return [float(v[idx[r["a"]]] @ v[idx[r["b"]]]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lexicon")
    ap.add_argument("out")
    ap.add_argument("--bge", action="store_true")
    a = ap.parse_args()
    lex = json.load(open(a.lexicon))
    data = {"P": pairs(lex["P"], lex["fillers"]), "A": pairs(lex["A"], lex["fillers"])}
    if a.bge:
        enc = bge_encoder()
        seen = generate("seen")
        cs = cosines(enc, seen)
        ys = [r["label"] for r in seen]
        best = max(sorted(set(cs)), key=lambda t: balanced_acc(ys, [int(c >= t) for c in cs]))
        res = {"system": f"bge-small {BGE_REV[:8]} cosine", "threshold_from_seen": round(best, 4)}
        for name, rows in data.items():
            cu = cosines(enc, rows)
            pred = [int(c >= best) for c in cu]
            y = [r["label"] for r in rows]
            res[name] = {"n": len(rows), "balanced_accuracy": round(balanced_acc(y, pred), 4),
                         "per_phenomenon_accuracy": per_phenomenon(rows, pred), "correct": [int(p == t) for p, t in zip(pred, y)]}
    else:
        res = {"system": f"CSI {CSI_VERSION}", "parser": os.environ.get("CSI_PARSER"), "lemmas": os.environ.get("CSI_MORPH"),
               "senses": os.environ.get("CSI_SENSES")}
        for name, rows in list(data.items()) + [("heldout3", generate("heldout3")), ("nonce2", generate("nonce2"))]:
            res[name] = run_csi(rows)
            print(name, {k: v for k, v in res[name].items() if k not in ("errors", "correct")}, flush=True)
    res["sizes"] = {k: len(v) for k, v in data.items()}
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
