"""Experiment 8b: A/B/C on natural text (PAWS labeled_final test, 8,000 pairs), same LLM and prompts as Experiment 8.
CSI v0.1 alone is near chance here (Experiment 1N), so this measures whether the LLM is helped or misled by CSI's
analysis (B1) and verdict (B2). Threshold-free AUC and the model's own choice; no fitted threshold (no seen split).
Usage: LFM3_DIR=<model dir> python run_abc_paws.py <out json> [--limit N]"""
import argparse
import functools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_abc  # noqa: E402
from csi import CSI_VERSION, compare  # noqa: E402
from run_e1 import auc, balanced_acc  # noqa: E402

run_abc.prop_json = functools.lru_cache(maxsize=None)(run_abc.prop_json)
run_abc.compare = functools.lru_cache(maxsize=None)(compare)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    paws = load_dataset("google-research-datasets/paws", "labeled_final", split="test")
    rows = [{"a": x, "b": z, "label": int(y)} for x, z, y in zip(paws["sentence1"], paws["sentence2"], paws["label"])]
    if a.limit:
        rows = rows[:a.limit]
    mdir = os.environ["LFM3_DIR"]
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mdir, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    yes = run_abc.answer_ids(tok, ["Yes", " Yes", "yes", " yes", "YES"])
    no = run_abc.answer_ids(tok, ["No", " No", "no", " no", "NO"])
    think_end = tok.convert_tokens_to_ids("</think>")
    y = np.array([r["label"] for r in rows])
    csi = [run_abc.compare(r["a"], r["b"])["equivalent"] for r in rows]
    covered = np.array([c is not None for c in csi])
    csi_pred = np.array([int(bool(c)) for c in csi])
    res = {"experiment": "E8b A/B/C on PAWS labeled_final test", "csi_version": CSI_VERSION, "model": mdir, "n": len(rows),
           "C": {"coverage": round(float(covered.mean()), 4),
                 "balanced_accuracy_covered": round(balanced_acc(y[covered], csi_pred[covered]), 4)},
           "conditions": {}}
    for cond in run_abc.CONDITIONS:
        t0, s, mass = time.time(), [], []
        for r in rows:
            text = tok.apply_chat_template([{"role": "user", "content": run_abc.user_message(cond, r["a"], r["b"])}],
                                           add_generation_prompt=True, tokenize=False)
            ids = torch.tensor([tok(text, add_special_tokens=False).input_ids + [think_end]], device="cuda")
            with torch.no_grad():
                logp = torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), -1)
            py, pn = torch.logsumexp(logp[yes], 0), torch.logsumexp(logp[no], 0)
            s.append(float(py - pn))
            mass.append(float(py.exp() + pn.exp()))
        s = np.array(s)
        own = (s > 0).astype(int)
        agree = own[covered] == csi_pred[covered]
        csi_right = csi_pred[covered] == y[covered]
        res["conditions"][cond] = {
            "balanced_accuracy_own_choice": round(balanced_acc(y, own), 4), "auc": round(auc(s, y), 4),
            "said_yes_share": round(float(own.mean()), 4), "mean_yes_no_mass": round(float(np.mean(mass)), 4),
            "agrees_with_csi_when_csi_right": round(float(agree[csi_right].mean()), 4),
            "agrees_with_csi_when_csi_wrong": round(float(agree[~csi_right].mean()), 4),
            "balanced_accuracy_own_choice_covered": round(balanced_acc(y[covered], own[covered]), 4),
            "seconds": round(time.time() - t0, 1), "scores": [round(float(x), 4) for x in s]}
        print(cond, json.dumps({k: v for k, v in res["conditions"][cond].items() if k != "scores"}), flush=True)
    print("C", json.dumps(res["C"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
