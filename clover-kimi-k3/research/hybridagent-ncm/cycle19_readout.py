"""Cycle 19: latent readout (paper Fig 2 / Appendix F) read off the existing traces; no new run.
Per latent span: pseudo-text = nearest-token readout of R(h) (recorded as link_nn_id), readability = mean link_nn_cos
(Eq. 19), context specificity (Eq. 20) over our five questions. Noise reference = nearest-token cosine of Gaussian
vectors (2,000, as in the paper's setup) against the same embedding matrix."""
import glob
import json
import os
import re
import statistics as st

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

J = "/root/runs/hybrid/"
MODEL = "togethercomputer/HybridAgent-8B-Agentic"
tok = AutoTokenizer.from_pretrained(MODEL)
Q = {}
for l in open("/root/hybridagent/examples/questions.jsonl"):
    d = json.loads(l)
    Q[d["id"]] = d["question"]
STOP = set("the and for with that this are was were from have has had not but you your all can will would could should what which when where how who whom why than then them they their there here into out about over also any each other some such only own same too very just does did done its our his her him she let get got one two use using used".split())


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP}


def jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


# --- noise reference: nearest-token cosine of Gaussian vectors ---
snap = None
from huggingface_hub import snapshot_download
snap = snapshot_download(MODEL, local_files_only=True)
with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as f:
    key = [k for k in f.keys() if "embed_tokens" in k][0]
    E = f.get_tensor(key).float()
bank = torch.nn.functional.normalize(E, dim=1)
g = torch.Generator().manual_seed(0)
noise = torch.nn.functional.normalize(torch.randn(2000, E.shape[1], generator=g), dim=1)
nn_cos = (noise @ bank.T).max(dim=1).values
tau = float(nn_cos.mean())
print(f"noise reference: mean nearest-token cosine of 2,000 Gaussian vectors = {tau:.4f} (max {float(nn_cos.max()):.4f})")

# runs with latent steps: every latent-allowed trace
FILES = [f for f in sorted(glob.glob(J + "trace-q*.json")) if "nolat" not in f]
spans = []  # (file, qid, readability, ids, cosines)
for f in FILES:
    name = os.path.basename(f)[:-5]
    m = re.search(r"trace-(q\d)", name)
    if not m:
        continue
    qid = m.group(1)
    d = json.load(open(f))
    S, ids = d["steps"], d["ids"]
    cur = None
    for s in S:
        if s["mode_before"] == "DECODE" and s.get("tok") == ids["open"]:
            cur = ([], [])
        elif s["mode_before"] == "LATENT" and cur is not None:
            if s.get("tok") == ids["close"]:
                if cur[0]:
                    spans.append((name, qid, cur[0], cur[1]))
                cur = None
            elif s["kind"] == "latent":
                cur[0].append(s["link_nn_id"])
                cur[1].append(s["link_nn_cos"])
print(f"files {len(FILES)}, spans {len(spans)}, latent steps {sum(len(s[2]) for s in spans)}")

qw = {k: words(v) for k, v in Q.items()}
import collections
cnt = collections.Counter()
read, spec = [], []
per_q = {}
for name, qid, ids_, cos in spans:
    text = tok.decode(ids_)
    r = sum(cos) / len(cos)
    P = words(text)
    own = jac(P, qw[qid])
    others = [jac(P, qw[k]) for k in qw if k != qid]
    den = sum(others) / len(others)
    sp = own / den if den > 0 else None
    cnt[("own>0" if own > 0 else "own=0") + "/" + ("others>0" if den > 0 else "others=0")] += 1
    read.append(r)
    if sp is not None and sp > 0:
        spec.append(sp)
    per_q.setdefault(qid, []).append((r, sp))
print(f"readability per span: mean {st.mean(read):.3f}, median {st.median(read):.3f}, min {min(read):.3f}, max {max(read):.3f}; above noise reference {sum(r > tau for r in read)} of {len(read)}")
print(f"context specificity (positive scores): n {len(spec)} of {len(spans)}, median {st.median(spec):.2f}, mean {st.mean(spec):.2f}; spans with score > 1: {sum(s > 1 for s in spec)}")
allrs = [x for v in per_q.values() for x in v]
print("specificity cases:", dict(cnt), "(den0 = no other question shares a content word with the span, undefined; own0 = span shares none with its own question; pos = defined and positive)")
print("meaningful region (readability > noise and specificity > 1):", sum(1 for r, sp in allrs if sp and sp > 1 and r > tau), "of", len(spans))
for q in sorted(per_q):
    v = per_q[q]
    sps = [x[1] for x in v if x[1]]
    print(f"  {q}: spans {len(v)}, readability mean {st.mean(x[0] for x in v):.3f}, specificity>1 {sum(s > 1 for s in sps)} of {len(sps)}")

# token-level: share of steps whose nearest token is a digit / operator / word piece
allids = [i for s in spans for i in s[2]]
allcos = [c for s in spans for c in s[3]]
print(f"latent steps: nearest-token cosine mean {st.mean(allcos):.3f}, median {st.median(allcos):.3f}, share above 0.9: {sum(c > 0.9 for c in allcos) / len(allcos):.2f}, above 0.5: {sum(c > 0.5 for c in allcos) / len(allcos):.2f}")
print("\nexamples (file, span length, mean cosine, pseudo-text):")
import random
random.seed(1)
for name, qid, ids_, cos in random.sample(spans, 8):
    print(f"- {name} q={qid} len {len(ids_)} cos {sum(cos) / len(cos):.2f}: {tok.decode(ids_[:60])!r}")
