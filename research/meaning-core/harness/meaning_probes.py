"""Meaning probes beyond graded similarity, on public test sets:
  paws_auc   PAWS (labeled_final test): sentence pairs built to share nearly all words; label = paraphrase or not.
             ROC-AUC of cosine (0.5 = cannot tell; word overlap alone sits near 0.5 by construction).
  snli_rank  SNLI test: for a premise with both an entailed and a contradicting hypothesis, how often the entailed
             one is closer (0.5 = chance).
  examples   the human's four server sentences (three causal paraphrases, one contradiction): cosine matrix.
Encoders: clover_hash, clover_indra_hash, bge_small, clover_indra_bge (as in sts_clover.py), and lm:<model>[@adapter]
(mean of final-layer hidden states, L2, as the Learning Concepts evaluation; 4-bit NF4 as there).
Usage: python meaning_probes.py <clover mobile dir> <out.json> <encoder> [encoder ...]"""
import json
import sys
import time
from collections import defaultdict

import numpy as np

mobile, out_path, encoders = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, mobile)
import clover_semantics  # noqa: E402
import memory_index  # noqa: E402
from datasets import load_dataset  # noqa: E402

EXAMPLES = ["The server stopped because memory ran out.",
            "Memory exhaustion caused the server to terminate.",
            "The server ran out of memory and stopped.",
            "The server stopped, but memory was available."]


def auc(scores, labels):
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks for ties
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    pos, neg = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def bge_fn():
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", revision=memory_index.EMBED_REV)
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5", revision=memory_index.EMBED_REV).eval().to("cuda:0")

    def enc(texts):
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 128):
                b = tok(texts[i:i + 128], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda:0")
                out.append(torch.nn.functional.normalize(model(**b).last_hidden_state[:, 0], dim=-1).float().cpu().numpy())
        return np.concatenate(out)
    return enc


def indra_fn(base):
    space = clover_semantics.IndraSemanticSpace(embed_fn=base)

    def enc(texts):
        p = space.yoneda_profile(np.asarray(base(texts), dtype=np.float64), space_name="text")
        p = p - p.mean(axis=1, keepdims=True)
        return p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None)
    return enc


def lm_fn(spec):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    name, _, adapter = spec.partition("@")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map={"": "cuda:0"})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def enc(texts):
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 32):
                b = tok(texts[i:i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda:0")
                h = model(**b, output_hidden_states=True).hidden_states[-1]
                m = b["attention_mask"].to(h.dtype).unsqueeze(-1)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
                out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)
    return enc


def make(name):
    hash_fn = memory_index.SubwordHashEmbedder().encode
    if name == "clover_hash":
        return hash_fn
    if name == "clover_indra_hash":
        return indra_fn(hash_fn)
    if name == "bge_small":
        return bge_fn()
    if name == "clover_indra_bge":
        return indra_fn(bge_fn())
    if name.startswith("lm:"):
        return lm_fn(name[3:])
    raise SystemExit(f"unknown encoder {name}")


def embed_all(fn, texts):
    uniq = sorted(set(texts))
    vecs = fn(uniq)
    return {t: vecs[i] for i, t in enumerate(uniq)}


paws = load_dataset("google-research-datasets/paws", "labeled_final", split="test")
snli = load_dataset("stanfordnlp/snli", split="test")
by_premise = defaultdict(dict)
for r in snli:
    if r["label"] in (0, 2):
        by_premise[r["premise"]].setdefault(r["label"], r["hypothesis"])
triples = [(p, h[0], h[2]) for p, h in by_premise.items() if 0 in h and 2 in h]

results = {}
try:
    results = json.load(open(out_path))
except (OSError, ValueError):
    pass
for name in encoders:
    t0 = time.time()
    fn = make(name)
    texts = list(paws["sentence1"]) + list(paws["sentence2"]) + [x for t in triples for x in t] + EXAMPLES
    v = embed_all(fn, texts)
    paws_cos = [float(v[a] @ v[b]) for a, b in zip(paws["sentence1"], paws["sentence2"])]
    snli_win = [float(v[p] @ v[e]) > float(v[p] @ v[c]) for p, e, c in triples]
    ex = np.stack([v[t] for t in EXAMPLES])
    results[name] = {
        "paws_auc": round(auc(paws_cos, paws["label"]), 4), "paws_pairs": len(paws_cos),
        "snli_rank": round(float(np.mean(snli_win)), 4), "snli_triples": len(triples),
        "examples_cos": np.round(ex @ ex.T, 3).tolist(), "seconds": round(time.time() - t0, 1),
    }
    print(name, {k: results[name][k] for k in ("paws_auc", "snli_rank", "seconds")}, flush=True)
    json.dump(results, open(out_path, "w"), indent=1)
