"""B-states: export last-layer hidden states (HF hidden_states[-1], what HybridLink reads) at the response positions of Dolly text,
for the frozen Qwen3-8B-Base ('base') or HybridAgent-8B ('ha'), on IDENTICAL token sequences, so states can be paired position by position.

Format (same for both models, no chat template): ids = tok("Question: {q}\\nAnswer:") + tok(" {response}"); position t predicts ids[t+1].
Usage: python b_states.py --model base --split train --n 1500 --out states_base_train.pt
"""
import argparse, glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["base", "ha"], required=True)
ap.add_argument("--split", choices=["train", "held"], required=True)
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)

BASE_SNAP = glob.glob("/root/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/*")[0]
snap = BASE_SNAP if a.model == "base" else HA_SNAP
tok = AutoTokenizer.from_pretrained(snap)
rows = [r for r in dolly_rows(tok=tok) if r["split"] == a.split][: a.n]
log(a.model, a.split, "rows", len(rows))
model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).eval()
H, Y, S, ntok, t0 = [], [], [], 0, time.time()
with torch.no_grad():
    for i, r in enumerate(rows):
        pre = tok(f"Question: {r['instruction']}\nAnswer:", add_special_tokens=False).input_ids
        resp = tok(" " + r["response"], add_special_tokens=False).input_ids
        ids = pre + resp
        out = model(torch.tensor([ids]), output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][0]
        pos = list(range(len(pre) - 1, len(ids) - 1))
        H.append(h[pos].cpu()); Y += [ids[p + 1] for p in pos]; S += [i] * len(pos)
        ntok += len(ids)
        if i % 50 == 0:
            log(f"{i}/{len(rows)} {ntok / (time.time() - t0):.1f} tok/s")
torch.save({"h": torch.cat(H), "y": torch.tensor(Y), "seq": torch.tensor(S), "rows": [(r["instruction"], r["response"]) for r in rows],
            "model": a.model, "snap": snap}, a.out)
log("saved", a.out, len(Y), "positions")
