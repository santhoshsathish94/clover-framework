"""Probe LFM3-Omni-2B S3 on the GPU instance: does it load (transformers 5.x, trust_remote_code), what are its
module names (LoRA targets), and a first observation of its next-token distribution on the Learning Concepts
smoke sentences (probability mass on the target vs its concept set).
Usage: python lfm3_probe.py <model dir> <out json> [bos|nobos]"""
import collections
import json
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

path, out = sys.argv[1], sys.argv[2]
use_bos = len(sys.argv) > 3 and sys.argv[3] == "bos"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
report = {"transformers": transformers.__version__, "load_s": round(time.time() - t0, 1),
          "class": type(model).__name__, "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
          "vocab": len(tok), "config": {k: getattr(model.config, k, None) for k in
                                        ("hidden_size", "num_hidden_layers", "layer_types", "tie_word_embeddings")}}

# module names: which Linear layers exist, grouped by suffix (LoRA target candidates)
lin = collections.Counter(n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear))
report["linear_suffixes"] = dict(lin)
report["first_layer_modules"] = [n for n, _ in model.named_modules() if ".layers.0." in n or ".layers.2." in n][:40]

# next-token observations: prefix -> top-10 and mass on a concept set (target + synonyms)
probes = [
    ("this sticker can be placed", " on", [" on", " onto", " upon"]),
    ("The Urban Knife Caddy by Kapoosh holds the same", " amount", [" amount", " number", " quantity"]),
    ("The server stopped because memory ran", " out", [" out"]),
    ("The doctor examined the", " patient", [" patient", " child", " man", " woman"]),
]
obs = []
bos_id = tok.bos_token_id if tok.bos_token_id is not None else model.config.bos_token_id
report["bos"] = {"used": use_bos, "id": bos_id, "token": tok.convert_ids_to_tokens([bos_id])[0]}
with torch.no_grad():
    for prefix, target, concept in probes:
        ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids
        if use_bos:
            ids = torch.cat([torch.tensor([[bos_id]]), ids], dim=1)
        ids = ids.cuda()
        probs = torch.softmax(model(ids).logits[0, -1].float(), -1)
        top = torch.topk(probs, 10)
        cid = [tok(w, add_special_tokens=False).input_ids for w in concept]
        first = sorted({c[0] for c in cid})
        obs.append({"prefix": prefix, "bos_first": tok.convert_ids_to_tokens(ids[0, :2].tolist()),
                    "top10": [(tok.decode([i]), round(p, 4)) for p, i in zip(top.values.tolist(), top.indices.tolist())],
                    "target_p": round(probs[cid[0][0]].item(), 4), "concept_first_tokens": len(first),
                    "concept_mass": round(probs[first].sum().item(), 4)})
report["next_token"] = obs
report["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
json.dump(report, open(out, "w"), indent=1)
print(json.dumps(report, indent=1)[:3500])
