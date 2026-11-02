"""B4: the meaning core against Clover's current semantics and baselines, on test splits never used in training
or model selection. Same measurements for every encoder:
  MTEB STS, 9 English tasks (Spearman x100; the Learning Concepts preset)
  PAWS labeled_final test ROC-AUC (same words, different meaning)
  SNLI test: entailed closer than contradiction (share)
  CondaQA test: original negated sentence vs its edits; AUC of paraphrase edits above scope/negation edits
  the human's four server sentences (cosine to the first)
Encoders: clover_hash, clover_indra_hash (what the phone runtime builds today), bge_small,
  lfm3_frozen_l4 (frozen C1 hidden state 4, mean pooling), core:<run dir> (trained meaning core).
Usage: python core_eval.py <clover mobile dir> <lfm3 model dir> <out json> <encoder> [encoder ...]"""
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

mobile, model_dir, out_path, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.path.insert(0, mobile)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mteb  # noqa: E402
from datasets import load_dataset  # noqa: E402
from mteb.models.abs_encoder import AbsEncoder  # noqa: E402

STS_PRESET = ["STS12", "STS13", "STS14", "STS15", "STS17", "STS22.v2", "STSBenchmark", "SICK-R", "BIOSSES"]
SERVER = ["The server stopped because memory ran out.", "Memory exhaustion caused the server to terminate.",
          "The server ran out of memory and stopped.", "The server stopped, but memory was available."]


def auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def make(name):
    if name in ("clover_hash", "clover_indra_hash", "bge_small"):
        import meaning_probes_lib as mp
        return mp.make(name)
    import meaning_core as mc
    if name == "lfm3_frozen_l4":
        return mc.baseline_encoder(model_dir, 4)
    if name.startswith("core:"):
        return mc.load_core(model_dir, name[5:])
    raise SystemExit(f"unknown encoder {name}")


class FnEncoder(AbsEncoder):
    mteb_model_meta = None

    def __init__(self, fn):
        self.fn = fn

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs):
        out = []
        for batch in inputs:
            t = batch["text"]
            out.append(np.asarray(self.fn([t] if isinstance(t, str) else list(t)), dtype=np.float32))
        return np.concatenate(out, axis=0)


def cosmap(fn, texts):
    uniq = sorted(set(texts))
    v = np.asarray(fn(uniq), dtype=np.float64)
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    return {t: v[i] for i, t in enumerate(uniq)}


paws = load_dataset("google-research-datasets/paws", "labeled_final", split="test")
snli = load_dataset("stanfordnlp/snli", split="test")
by = defaultdict(dict)
for r in snli:
    if r["label"] in (0, 2):
        by[r["premise"]][r["label"]] = r["hypothesis"]
snli_trip = [(p, d[0], d[2]) for p, d in by.items() if 0 in d and 2 in d]
cq = load_dataset("lasha-nlp/CONDAQA", split="test")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
seen, cq_pairs = set(), []
for r in cq:
    key = (r["PassageID"], int(r["PassageEditID"]))
    if key in seen or int(r["PassageEditID"]) == 0:
        continue
    seen.add(key)
    orig = set(SENT.split(r["original passage"].strip()))
    changed = [s for s in SENT.split(r["sentence1"].strip()) if s not in orig]
    if len(changed) == 1:
        cq_pairs.append((r["original sentence"], changed[0], 1 if int(r["PassageEditID"]) == 1 else 0))
bench = {t.metadata.name: t for t in mteb.get_benchmark("MTEB(eng, v2)").tasks}
tasks = [bench[n] for n in STS_PRESET]
results = json.load(open(out_path)) if os.path.exists(out_path) else {}
for name in names:
    t0 = time.time()
    fn = make(name)
    res = mteb.MTEB(tasks=tasks).run(FnEncoder(fn), output_folder=os.path.join(os.path.dirname(out_path), "mteb",
                                                                               re.sub(r"[^\w.-]", "_", name)),
                                     raise_error=True)
    per = {}
    for tr in res:
        vals = [it["main_score"] for sp in tr.scores.values() for it in (sp if isinstance(sp, list) else [sp])
                if isinstance(it, dict) and "main_score" in it]
        per[tr.task_name] = round(float(np.mean(vals)) * 100, 2)
    E = cosmap(fn, list(paws["sentence1"]) + list(paws["sentence2"]))
    paws_auc = auc([E[a] @ E[b] for a, b in zip(paws["sentence1"], paws["sentence2"])], list(paws["label"]))
    E = cosmap(fn, [t for x in snli_trip for t in x])
    snli_rank = float(np.mean([E[p] @ E[e] > E[p] @ E[c] for p, e, c in snli_trip]))
    E = cosmap(fn, [t for x in cq_pairs for t in x[:2]])
    cq_auc = auc([E[a] @ E[b] for a, b, _ in cq_pairs], [y for *_, y in cq_pairs])
    E = cosmap(fn, SERVER)
    results[name] = {"sts": per, "sts_mean": round(float(np.mean(list(per.values()))), 2),
                     "paws_test_auc": round(paws_auc, 4), "snli_test_rank": round(snli_rank, 4),
                     "condaqa_test_auc": round(cq_auc, 4), "condaqa_test_pairs": len(cq_pairs),
                     "server_cos_to_first": [round(float(E[SERVER[0]] @ E[s]), 3) for s in SERVER],
                     "seconds": round(time.time() - t0, 1)}
    print(name, json.dumps(results[name]), flush=True)
    json.dump(results, open(out_path, "w"), indent=1)
    del fn
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
