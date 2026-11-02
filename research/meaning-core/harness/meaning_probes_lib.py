"""Import-safe copies of the encoders in meaning_probes.py / sts_clover.py (that script runs its evaluation on import):
clover_hash (memory_index.SubwordHashEmbedder, what the phone uses without bge), clover_indra_hash
(clover_semantics.IndraSemanticSpace over it, mean-centred angular profile, as the runtime builds it), bge_small
(BAAI/bge-small-en-v1.5 CLS, pinned revision), clover_indra_bge. Requires clover-ai-system/mobile on sys.path."""
import numpy as np

import clover_semantics
import memory_index


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
    raise SystemExit(f"unknown encoder {name}")
