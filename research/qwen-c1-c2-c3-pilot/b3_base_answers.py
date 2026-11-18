"""B3 - answers of the frozen Qwen3-8B-Base (plain completion, no chat template) for the same held-out Dolly questions as A3,
saved in the A3 answers format so that a3_end_to_end.py renders them with C3 (plan = grounded plan of the base model's answer).
Usage: python b3_base_answers.py --n 40 --out results/b3_answers.json
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2_c3_wmt14 as A2
from common import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40); ap.add_argument("--max-new", type=int, default=60); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--out", default=BASE + "/results/b3_answers.json")
a = ap.parse_args()
torch.set_num_threads(16)
snap = glob.glob("/root/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(snap); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).eval()
_, tk0, _, _, _ = A2.load_models(torch.device("cpu"))
pairs = A2.make_pairs(tk0, "held", a.n)
ans, t0 = [], time.time()
for i in range(0, len(pairs), a.bs):
    chunk = pairs[i:i + a.bs]
    prompts = [f"Question: {p['q']}\nAnswer in at most two short sentences.\nAnswer:" for p in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
    for p, o in zip(chunk, out):
        txt = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True)
        txt = txt.split("\nQuestion:")[0].strip()
        ans.append({"q": p["q"], "human": p["text"], "model_answer": txt})
    log(f"{len(ans)}/{len(pairs)} {time.time() - t0:.0f}s")
json.dump(ans, open(a.out, "w"), indent=1, ensure_ascii=False)
log("wrote", a.out)
