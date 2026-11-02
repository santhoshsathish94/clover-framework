"""Experiment 3c2 independent re-check (GPU server): the closeness of every meaning pair in a CSI v0.3 run, recomputed
from this server's copy of the Dictionary v1.1 checkpoint and the definitions it was trained with (data/e3a2/synsets.json),
one definition at a time, without csi/senses.py. Usage: python check_closeness.py <csi run json> <checkpoint> <synsets json> <out>"""
import json
import sys

import torch
from transformers import AutoModel, AutoTokenizer

run_path, ckpt, syn_path, out = sys.argv[1:5]
run, defs = json.load(open(run_path)), json.load(open(syn_path))
k3 = {}
for name, part in run.items():
    if isinstance(part, dict) and "rows" in part:
        for r in part["rows"]:
            for e in r["lexical"]:
                if e.get("closeness") is not None:
                    k3[(e["sense_a"]["synset"], e["sense_b"]["synset"])] = e["closeness"]
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModel.from_pretrained(f"{ckpt}/gloss", torch_dtype=torch.float32).cuda().eval()
vec = {}
with torch.no_grad():
    for s in sorted({s for pair in k3 for s in pair}):
        enc = tok(defs[s]["definition"], truncation=True, max_length=64, return_tensors="pt").to("cuda")
        vec[s] = model(**enc).last_hidden_state[0, 0].float().cpu()
pairs = [{"a": a, "b": b, "closeness": round(torch.nn.functional.cosine_similarity(vec[a], vec[b], dim=0).item(), 6),
          "k3": c} for (a, b), c in sorted(k3.items())]
json.dump({"pairs": pairs, "device": torch.cuda.get_device_name(0), "torch": torch.__version__}, open(out, "w"), indent=1)
print(len(pairs), "pairs; max |gpu - k3| =", max(abs(p["closeness"] - p["k3"]) for p in pairs))
