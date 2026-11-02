"""Experiment 3c2 independent re-check (GPU server): the closeness of every meaning pair in a CSI v0.3 run, recomputed
from this server's copy of the Dictionary v1.1 checkpoint, one definition at a time, without csi/senses.py.
Definitions: the ones the gloss encoder was trained with (data/e3a2/synsets.json), else read here from OEWN 2025.
Usage: python check_closeness.py <csi run json> <checkpoint> <synsets json> <oewn xml.gz> <out>"""
import gzip
import json
import sys
import xml.etree.ElementTree as ET

import torch
from transformers import AutoModel, AutoTokenizer

run_path, ckpt, syn_path, oewn_path, out = sys.argv[1:6]
run, defs = json.load(open(run_path)), json.load(open(syn_path))
k3 = {}
for name, part in run.items():
    if isinstance(part, dict) and "rows" in part:
        for r in part["rows"]:
            for e in r["lexical"]:
                if e.get("closeness") is not None:
                    k3[(e["sense_a"]["synset"], e["sense_b"]["synset"])] = e["closeness"]
needed = {s for pair in k3 for s in pair}
text, source = {s: defs[s]["definition"] for s in needed if s in defs}, {}
with gzip.open(oewn_path, "rb") as fh:
    for _, el in ET.iterparse(fh):
        if el.tag == "Synset":
            if el.get("id") in needed and el.get("id") not in text:
                d = el.find("Definition")
                text[el.get("id")] = d.text if d is not None else ""
                source[el.get("id")] = "oewn"
            el.clear()
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModel.from_pretrained(f"{ckpt}/gloss", torch_dtype=torch.float32).cuda().eval()
vec = {}
with torch.no_grad():
    for s in sorted(needed):
        enc = tok(text[s], truncation=True, max_length=64, return_tensors="pt").to("cuda")
        vec[s] = model(**enc).last_hidden_state[0, 0].float().cpu()
pairs = [{"a": a, "b": b, "closeness": round(torch.nn.functional.cosine_similarity(vec[a], vec[b], dim=0).item(), 6),
          "k3": c} for (a, b), c in sorted(k3.items())]
json.dump({"pairs": pairs, "definitions_from_training_file": len(needed) - len(source), "definitions_from_oewn": len(source),
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__}, open(out, "w"), indent=1)
print(len(pairs), "pairs;", len(source), "definitions read from OEWN; max |gpu - k3| =",
      max(abs(p["closeness"] - p["k3"]) for p in pairs))
