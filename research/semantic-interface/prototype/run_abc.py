"""Experiment 8 (A/B/C) runner: the same LLM (LFM3-Omni-2B S3, Clover's C1) judges the Experiment 1 minimal pairs
  A  = question only;  B1 = + CSI's proposition analysis of both sentences (JSON, no verdict);
  B2 = B1 + CSI's comparison (equivalent, differences, notes);  C = CSI alone (compare()).
Score = log P(Yes) - log P(No) over the first answer token (single-token case/space variants summed). Decisions: the
model's own choice (score > 0) and the threshold that maximises balanced accuracy on seen, applied to unseen.
Answer position: the S3 chat template ends the generation prompt with "<think>"; the model's own greedy output is
"</think>", "Yes"/"No", "<|im_end|>" (abc_probe.py, 12/12 cases), so "</think>" is appended and the next token scored.
Usage: LFM3_DIR=<model dir> python run_abc.py <out json> [--limit N] [--show K]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi import CSI_VERSION, analyze, compare  # noqa: E402
from csi.minimal_pairs import generate  # noqa: E402
from run_e1 import auc, balanced_acc, per_phenomenon  # noqa: E402

QUESTION = "Do these two sentences mean the same thing? Answer Yes or No."
CONDITIONS = ("A", "B1", "B2")


def prop_json(text):
    an = analyze(text)
    props = an["readings"][0]["propositions"]
    if not props:
        return {"unsupported": an["unsupported"]}
    p = props[0]
    return {"predicate": p["predicate"]["lemma"], "roles": {k: v["text"] for k, v in p["roles"].items()},
            "polarity": p["polarity"], "tense": p["tense"], "aspect": p["aspect"], "modality": p["modality"],
            "voice": p["voice"], "embedded_under": p["embedded_under"]}


def user_message(cond, a, b):
    s = f"Sentence 1: {a}\nSentence 2: {b}\n"
    if cond in ("B1", "B2"):
        s += (f"\nAnalysis of both sentences by the Clover Semantic Interface (CSI v{CSI_VERSION}; ARG0 = doer, "
              "ARG1 = thing acted on, ARG2 = recipient):\n"
              f"Sentence 1: {json.dumps(prop_json(a))}\nSentence 2: {json.dumps(prop_json(b))}\n")
    if cond == "B2":
        c = compare(a, b)
        s += "CSI comparison: " + json.dumps({"equivalent": c["equivalent"], "differences": c["differences"],
                                              "notes": c["notes"]}) + "\n"
    return s + "\n" + QUESTION


def answer_ids(tok, words):
    ids = set()
    for w in words:
        enc = tok.encode(w, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


def evaluate(scores, seen, unseen):
    ys, yu = np.array([r["label"] for r in seen]), np.array([r["label"] for r in unseen])
    ss, su = scores["seen"], scores["unseen"]
    best = max(np.unique(ss), key=lambda t: balanced_acc(ys, (ss >= t).astype(int)))
    out = {"threshold_from_seen": round(float(best), 4)}
    for lex, s, y, rows in (("seen", ss, ys, seen), ("unseen", su, yu, unseen)):
        own = (s > 0).astype(int)
        out[lex] = {"balanced_accuracy_own_choice": round(balanced_acc(y, own), 4),
                    "said_yes_share": round(float(own.mean()), 4),
                    "balanced_accuracy_seen_threshold": round(balanced_acc(y, (s >= best).astype(int)), 4),
                    "auc": round(auc(s, y), 4),
                    "per_phenomenon_accuracy_own_choice": per_phenomenon(rows, own)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mdir = os.environ["LFM3_DIR"]
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mdir, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    yes, no = answer_ids(tok, ["Yes", " Yes", "yes", " yes", "YES"]), answer_ids(tok, ["No", " No", "no", " no", "NO"])
    think_end = tok.convert_tokens_to_ids("</think>")
    assert think_end is not None and think_end != tok.unk_token_id, "no </think> token"
    data = {lex: generate(lex) for lex in ("seen", "unseen")}
    if a.limit:
        data = {k: v[:a.limit] for k, v in data.items()}
    res = {"experiment": "E8 A/B/C on Experiment 1 pairs", "csi_version": CSI_VERSION, "model": mdir,
           "yes_ids": yes, "no_ids": no, "answer_prefix_ids": [think_end], "sizes": {k: len(v) for k, v in data.items()}, "conditions": {},
           "examples": {}}
    shown = 0
    for cond in CONDITIONS:
        t0, scores, mass, top = time.time(), {}, [], {}
        for lex, rows in data.items():
            s = []
            for r in rows:
                text = tok.apply_chat_template([{"role": "user", "content": user_message(cond, r["a"], r["b"])}],
                                               add_generation_prompt=True, tokenize=False)
                ids = tok(text, add_special_tokens=False).input_ids + [think_end]
                ids = torch.tensor([ids], device="cuda")
                with torch.no_grad():
                    logp = torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), -1)
                py, pn = torch.logsumexp(logp[yes], 0), torch.logsumexp(logp[no], 0)
                s.append(float(py - pn))
                mass.append(float(py.exp() + pn.exp()))
                t1 = tok.decode([int(logp.argmax())])
                top[t1] = top.get(t1, 0) + 1
                if shown < a.show:
                    print("=== PROMPT", cond, repr(text), "\n=== top:", repr(t1), "score", round(s[-1], 3), flush=True)
                    res["examples"][f"{cond}:{lex}:{len(s)}"] = text
                    shown += 1
            scores[lex] = np.array(s)
        r = evaluate(scores, data["seen"], data["unseen"])
        r.update({"mean_yes_no_mass": round(float(np.mean(mass)), 4), "seconds": round(time.time() - t0, 1),
                  "top_first_tokens": dict(sorted(top.items(), key=lambda kv: -kv[1])[:8]),
                  "scores": {k: [round(float(x), 4) for x in v] for k, v in scores.items()}})
        res["conditions"][cond] = r
        print(cond, json.dumps({k: v for k, v in r.items() if k != "scores"}), flush=True)
    res["conditions"]["C"] = {}
    for lex, rows in data.items():
        pred = np.array([1 if compare(r["a"], r["b"])["equivalent"] else 0 for r in rows])
        y = np.array([r["label"] for r in rows])
        res["conditions"]["C"][lex] = {"balanced_accuracy": round(balanced_acc(y, pred), 4),
                                       "per_phenomenon_accuracy": per_phenomenon(rows, pred)}
    print("C", json.dumps(res["conditions"]["C"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
