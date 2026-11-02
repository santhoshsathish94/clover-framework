"""Which LFM3 layer and pooling carry meaning, frozen? One bf16 forward per batch gives all hidden states; each layer
x pooling is scored on development splits only (test splits stay untouched for the final evaluation):
  STS-B validation (1,500 pairs): Spearman of cosine vs human score
  PAWS labeled_final validation (8,000 pairs, same words / different meaning): ROC-AUC of cosine for paraphrase
  SNLI validation: per premise with both, fraction where cos(premise, entailed) > cos(premise, contradiction)
BOS (<|startoftext|>) is prepended (LFM3 needs it on raw text; results/lfm3_probe_*.json).
Usage: python lfm3_layers.py <model dir> <out json>"""
import json
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

path, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
bos = model.config.bos_token_id
pad = tok.pad_token_id if tok.pad_token_id is not None else model.config.pad_token_id
POOLS = ("mean_with_bos", "mean_no_bos", "last")


@torch.no_grad()
def encode(texts, bs=64):
    """{pool: [n_layers+1, N, H] float32 normalized} for the given texts."""
    res = {p: [] for p in POOLS}
    for i in range(0, len(texts), bs):
        ids = [[bos] + tok(t, add_special_tokens=False).input_ids[:254] for t in texts[i:i + bs]]
        L = max(map(len, ids))
        x = torch.full((len(ids), L), pad, dtype=torch.long)
        m = torch.zeros((len(ids), L), dtype=torch.long)
        for j, s in enumerate(ids):
            x[j, :len(s)] = torch.tensor(s)
            m[j, :len(s)] = 1
        x, m = x.cuda(), m.cuda()
        hs = torch.stack(model.model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states).float()
        mf = m.float()[None, :, :, None]
        nb = mf.clone(); nb[:, :, 0] = 0
        lengths = m.sum(1) - 1
        pooled = {"mean_with_bos": (hs * mf).sum(2) / mf.sum(2),
                  "mean_no_bos": (hs * nb).sum(2) / nb.sum(2).clamp(min=1),
                  "last": hs[:, torch.arange(len(ids)), lengths]}
        for p in POOLS:
            res[p].append(torch.nn.functional.normalize(pooled[p], dim=-1).cpu())
    return {p: torch.cat(v, 1) for p, v in res.items()}


def cos_pairs(a, b):
    return (a * b).sum(-1).numpy()  # [layers, N]


t0 = time.time()
sts = load_dataset("sentence-transformers/stsb", split="validation")
paws = load_dataset("google-research-datasets/paws", "labeled_final", split="validation")
snli = load_dataset("stanfordnlp/snli", split="validation")
by = defaultdict(dict)
for r in snli:
    if r["label"] in (0, 2):
        by[r["premise"]][r["label"]] = r["hypothesis"]
trip = [(p, d[0], d[2]) for p, d in by.items() if 0 in d and 2 in d][:3000]

E = {}
E["sts1"], E["sts2"] = encode(list(sts["sentence1"])), encode(list(sts["sentence2"]))
E["paws1"], E["paws2"] = encode(list(paws["sentence1"])), encode(list(paws["sentence2"]))
E["snp"], E["sne"], E["snc"] = (encode([t[k] for t in trip]) for k in range(3))
n_layers = E["sts1"]["last"].shape[0]
report = {"model": path, "n_hidden_states": n_layers, "sizes": {"sts_val": len(sts), "paws_val": len(paws),
          "snli_val_triples": len(trip)}, "layers": {}}
for p in POOLS:
    s = cos_pairs(E["sts1"][p], E["sts2"][p])
    w = cos_pairs(E["paws1"][p], E["paws2"][p])
    ce, cc = cos_pairs(E["snp"][p], E["sne"][p]), cos_pairs(E["snp"][p], E["snc"][p])
    rows = []
    for layer in range(n_layers):
        rows.append({"layer": layer,
                     "sts_spearman": round(float(spearmanr(s[layer], sts["score"]).correlation), 4),
                     "paws_auc": round(float(roc_auc_score(paws["label"], w[layer])), 4),
                     "snli_rank": round(float((ce[layer] > cc[layer]).mean()), 4)})
    report["layers"][p] = rows
report["seconds"] = round(time.time() - t0, 1)
json.dump(report, open(out, "w"), indent=1)
for p in POOLS:
    best = max(report["layers"][p], key=lambda r: r["sts_spearman"])
    print(p, "best STS layer", best, "| last layer", report["layers"][p][-1])
print("seconds", report["seconds"])
