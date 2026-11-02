"""How well does each representation agree with human similarity judgments? MTEB STS (the 9 English tasks of
MTEB(eng, v2), the Learning Concepts paper's STS preset), Spearman of cosine vs human scores.

Encoders (each written as the Clover code computes it):
  clover_hash        memory_index.SubwordHashEmbedder: the vectors the phone uses when bge is not on disk
  clover_indra_hash  clover_semantics.IndraSemanticSpace() with no embed_fn (what the runtime constructs): angular
                     profile against CLOVER_SEMANTIC_ANCHORS, compared by relational_similarity (Pearson of the
                     profiles = cosine of the mean-centred profiles)
  bge_small          BAAI/bge-small-en-v1.5, CLS + L2 norm, as memory_index.Embedder (no query prefix: STS is symmetric)
  clover_indra_bge   the same Indra profile built on bge_small vectors
Usage: python sts_clover.py <clover mobile dir> <out dir> [encoder ...]"""
import json
import os
import sys
import time

import numpy as np

mobile, out_root = sys.argv[1], sys.argv[2]
which = sys.argv[3:] or ["clover_hash", "clover_indra_hash", "bge_small", "clover_indra_bge"]
sys.path.insert(0, mobile)
import mteb  # noqa: E402
from mteb.models.abs_encoder import AbsEncoder  # noqa: E402

import clover_semantics  # noqa: E402
import memory_index  # noqa: E402

STS_PRESET = ["STS12", "STS13", "STS14", "STS15", "STS17", "STS22.v2", "STSBenchmark", "SICK-R", "BIOSSES"]


def bge_encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", revision=memory_index.EMBED_REV)
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5", revision=memory_index.EMBED_REV).eval().to("cuda:0")

    def encode(texts):
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 128):
                enc = tok(texts[i:i + 128], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda:0")
                h = model(**enc).last_hidden_state
                out.append(torch.nn.functional.normalize(h[:, 0], dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)
    return encode


def indra_encoder(embed_fn):
    space = clover_semantics.IndraSemanticSpace(embed_fn=embed_fn)

    def encode(texts):
        prof = space.yoneda_profile(np.asarray(embed_fn(texts), dtype=np.float64), space_name="text")
        prof = prof - prof.mean(axis=1, keepdims=True)
        return (prof / np.clip(np.linalg.norm(prof, axis=1, keepdims=True), 1e-12, None)).astype(np.float32)
    return encode, len(space.anchor_ids)


class FnEncoder(AbsEncoder):
    mteb_model_meta = None

    def __init__(self, fn):
        self.fn = fn

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs):
        out = []
        for batch in inputs:
            texts = batch["text"]
            out.append(self.fn([texts] if isinstance(texts, str) else list(texts)))
        return np.concatenate(out, axis=0)


bench = {t.metadata.name: t for t in mteb.get_benchmark("MTEB(eng, v2)").tasks}
tasks = [bench[n] for n in STS_PRESET]
hash_fn = memory_index.SubwordHashEmbedder().encode
summary = {}
for name in which:
    t0 = time.time()
    if name == "clover_hash":
        fn, note = hash_fn, "384-d hashed words + character trigrams"
    elif name == "clover_indra_hash":
        fn, k = indra_encoder(hash_fn)
        note = f"{k} anchors over clover_hash"
    elif name == "bge_small":
        fn, note = bge_encoder(), "bge-small-en-v1.5 CLS"
    elif name == "clover_indra_bge":
        fn, k = indra_encoder(bge_encoder())
        note = f"{k} anchors over bge_small"
    else:
        raise SystemExit(f"unknown encoder {name}")
    res = mteb.MTEB(tasks=tasks).run(FnEncoder(fn), output_folder=os.path.join(out_root, name), raise_error=True)
    per = {}
    for tr in res:
        vals = [item["main_score"] for split in tr.scores.values() for item in (split if isinstance(split, list) else [split])
                if isinstance(item, dict) and "main_score" in item]
        per[tr.task_name] = round(float(np.mean(vals)) * 100, 2)
    summary[name] = {"note": note, "tasks": per, "mean": round(float(np.mean(list(per.values()))), 2),
                     "seconds": round(time.time() - t0, 1)}
    print(name, summary[name], flush=True)
json.dump(summary, open(os.path.join(out_root, "summary.json"), "w"), indent=1)
