"""Clover meaning core (B3) for LFM3-Omni-2B S3 (Clover's C1). Trains a meaning head over LFM3's hidden states on
the B2 data (data/core, licences in CONTEXT.md), and evaluates on development splits only.

Variants:
  core-f (default): LFM3 frozen (bf16, no grad). Head: softmax mix over all 25 hidden states -> LayerNorm ->
          [mask-mean ; attention pool] -> MLP -> 768-d unit vector. At runtime it can read C1's own forward pass.
  core-l (--lora R): the same head + LoRA (rank R) on every LFM3 linear (attention q/k/v/o, conv in/out,
          feed-forward w1/w2/w3), a separate adapter used only for the semantics pass.
BOS (<|startoftext|>) is always prepended: LFM3 needs it on raw text (results/lfm3_probe_*.json).

Objectives (the human's six), one mixed batch per step:
  triplets  InfoNCE, anchor vs positives (+ in-batch) + hard negatives       equivalence / distinction
  pairs     CoSENT ranking: every label-1 pair above every label-0 pair       equivalence / distinction / composition
  relations classifier on (u, v, |u-v|, u*v) over OEWN + ConceptNet types     relations / grounding
  wic       span vectors, same sense vs different sense of a word             context
Dev metrics: STS-B validation Spearman (never trained on), PAWS dev AUC, NLI dev rank, CondaQA dev AUC, role-swap and
antonym-substitution dev (share of label-0 pairs below the median PAWS-paraphrase cosine), relation dev accuracy,
WiC dev accuracy, and the human's four server sentences.

Usage: python meaning_core.py --model DIR --data DIR --out DIR [--steps N] [--lora R] [--eval-only-baseline LAYER]"""
import argparse
import collections
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

SERVER = ["The server stopped because memory ran out.", "Memory exhaustion caused the server to terminate.",
          "The server ran out of memory and stopped.", "The server stopped, but memory was available."]


def jsonl(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            out.append(json.loads(line))
    return out


class Encoder:
    """LFM3 backbone (frozen or LoRA) -> all hidden states for a list of texts (with BOS), optional span masks."""

    def __init__(self, path, lora_rank=0, max_len=96):
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).cuda()
        self.bos = self.model.config.bos_token_id
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.model.config.pad_token_id
        self.max_len = max_len
        self.lora = lora_rank > 0
        if self.lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05, bias="none",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"])
            self.model = get_peft_model(self.model, cfg)
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
            self.model.train()
        else:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.backbone = self.model.get_base_model().model if self.lora else self.model.model

    def batch(self, texts, spans=None):
        ids, smask = [], []
        for i, t in enumerate(texts):
            enc = self.tok(t, add_special_tokens=False, return_offsets_mapping=spans is not None)
            tid = [self.bos] + enc["input_ids"][: self.max_len - 1]
            ids.append(tid)
            if spans is not None:
                s0, s1 = spans[i]
                offs = enc["offset_mapping"][: self.max_len - 1]
                smask.append([0] + [1 if (a < s1 and b > s0) else 0 for a, b in offs])
        L = max(map(len, ids))
        x = torch.full((len(ids), L), self.pad, dtype=torch.long)
        m = torch.zeros((len(ids), L), dtype=torch.bool)
        sm = torch.zeros((len(ids), L), dtype=torch.bool) if spans is not None else None
        for j, s in enumerate(ids):
            x[j, : len(s)] = torch.tensor(s)
            m[j, : len(s)] = True
            if spans is not None:
                sm[j, : len(smask[j])] = torch.tensor(smask[j], dtype=torch.bool)
                if not sm[j].any():
                    sm[j] = m[j]
        return x.cuda(), m.cuda(), (sm.cuda() if sm is not None else None)

    def states(self, x, m):
        ctx = torch.enable_grad() if self.lora and torch.is_grad_enabled() else torch.no_grad()
        with ctx:
            out = self.backbone(input_ids=x, attention_mask=m.long(), output_hidden_states=True)
        return torch.stack(out.hidden_states)  # [S, B, T, H] bf16


class Head(nn.Module):
    def __init__(self, n_states, hidden, dim=768, n_rel=1):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(n_states))
        self.norm = nn.LayerNorm(hidden)
        self.q = nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.proj = nn.Sequential(nn.Linear(2 * hidden, 1536), nn.GELU(), nn.Linear(1536, dim))
        self.span = nn.Sequential(nn.Linear(hidden, 1536), nn.GELU(), nn.Linear(1536, dim))
        self.rel = nn.Sequential(nn.Linear(4 * dim, 768), nn.GELU(), nn.Linear(768, n_rel))
        self.wic_scale = nn.Parameter(torch.tensor(10.0))
        self.wic_bias = nn.Parameter(torch.tensor(-5.0))

    def mixed(self, hs):
        w = torch.softmax(self.mix, 0).to(hs.dtype)
        return self.norm(torch.einsum("s,sbth->bth", w, hs).float())

    def sentence(self, hs, m):
        h = self.mixed(hs)
        mf = m.float()[..., None]
        mean = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
        sc = (self.k(h) @ self.q) / math.sqrt(h.shape[-1])
        a = torch.softmax(sc.masked_fill(~m, -1e4), 1)[..., None]
        return F.normalize(self.proj(torch.cat([mean, (a * h).sum(1)], -1)), dim=-1)

    def span_vec(self, hs, sm):
        h = self.mixed(hs)
        sf = sm.float()[..., None]
        return F.normalize(self.span((h * sf).sum(1) / sf.sum(1).clamp(min=1)), dim=-1)

    def rel_logits(self, u, v):
        return self.rel(torch.cat([u, v, (u - v).abs(), u * v], -1))


def embed(enc, head, texts, bs=128, grad=False):
    outs = []
    for i in range(0, len(texts), bs):
        x, m, _ = enc.batch(texts[i:i + bs])
        with torch.set_grad_enabled(grad):
            outs.append(head.sentence(enc.states(x, m), m))
    return torch.cat(outs)


def embed_spans(enc, head, texts, spans, bs=128, grad=False):
    outs = []
    for i in range(0, len(texts), bs):
        x, m, sm = enc.batch(texts[i:i + bs], spans[i:i + bs])
        with torch.set_grad_enabled(grad):
            outs.append(head.span_vec(enc.states(x, m), sm))
    return torch.cat(outs)


def embed_bucketed(enc, head, texts, budget=12000, grad=False):
    """Sentence vectors for texts encoded in length-sorted chunks of at most `budget` padded tokens, returned in
    the original order (one forward over everything pads every text to the longest; measured slower)."""
    lens = [min(enc.max_len, 1 + len(enc.tok(t, add_special_tokens=False).input_ids)) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lens[i])
    chunks, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lens[i] > budget:
            chunks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        chunks.append(cur)
    outs = []
    for c in chunks:
        x, m, _ = enc.batch([texts[i] for i in c])
        with torch.set_grad_enabled(grad):
            outs.append(head.sentence(enc.states(x, m), m))
    E = torch.cat(outs)
    pos = torch.empty(len(texts), dtype=torch.long, device=E.device)
    pos[torch.tensor([i for c in chunks for i in c], device=E.device)] = torch.arange(len(texts), device=E.device)
    return E[pos]


class Baseline(nn.Module):
    """Frozen LFM3 mean pooling at one hidden state (the B1 layer probe's best: layer 4, no head)."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def sentence(self, hs, m):
        h = hs[self.layer].float()
        mf = m.float()[..., None]
        return F.normalize((h * mf).sum(1) / mf.sum(1).clamp(min=1), dim=-1)

    def span_vec(self, hs, sm):
        return self.sentence(hs, sm)


@torch.no_grad()
def evaluate(enc, head, dev, rel_names):
    was_training = getattr(head, "training", False)
    head.eval()
    if enc.lora:
        enc.model.eval()
    r = {}
    a, b, s = dev["sts"]
    ea, eb = embed(enc, head, a), embed(enc, head, b)
    r["sts_b_val_spearman"] = round(float(spearmanr((ea * eb).sum(-1).cpu().numpy(), s).correlation), 4)
    pairs = dev["pairs"]
    ea, eb = embed(enc, head, [p["a"] for p in pairs]), embed(enc, head, [p["b"] for p in pairs])
    cos = (ea * eb).sum(-1).cpu().numpy()
    lab = np.array([p["label"] for p in pairs])
    src = np.array([p["source"] for p in pairs])
    obj = np.array([p["objective"] for p in pairs])
    for name in ("paws", "condaqa"):
        sel = src == name
        r[f"{name}_dev_auc"] = round(float(roc_auc_score(lab[sel], cos[sel])), 4)
    med = float(np.median(cos[(src == "paws") & (lab == 1)]))
    for o in ("composition:role_swap", "distinction:antonym_substitution"):
        sel = obj == o
        r[o.split(":")[1] + "_below_paraphrase_median"] = round(float((cos[sel] < med).mean()), 4)
        r[o.split(":")[1] + "_mean_cos"] = round(float(cos[sel].mean()), 4)
    t = dev["triplets"]
    ea = embed(enc, head, [x["anchor"] for x in t])
    ep = embed(enc, head, [x["positive"] for x in t])
    en = embed(enc, head, [x["negative"] for x in t])
    r["nli_dev_rank"] = round(float(((ea * ep).sum(-1) > (ea * en).sum(-1)).float().mean()), 4)
    if hasattr(head, "rel_logits"):
        rel = [x for x in dev["relations"] if x["relation"] in rel_names]
        u, v = embed(enc, head, [x["a"] for x in rel]), embed(enc, head, [x["b"] for x in rel])
        pred = head.rel_logits(u, v).argmax(-1).cpu().numpy()
        gold = np.array([rel_names.index(x["relation"]) for x in rel])
        r["relation_dev_acc"] = round(float((pred == gold).mean()), 4)
        w = dev["wic"]
        s1 = embed_spans(enc, head, [x["s1"] for x in w], [x["span1"] for x in w])
        s2 = embed_spans(enc, head, [x["s2"] for x in w], [x["span2"] for x in w])
        logit = head.wic_scale * (s1 * s2).sum(-1) + head.wic_bias
        r["wic_dev_acc"] = round(float(((logit > 0).long().cpu().numpy() == np.array([x["label"] for x in w])).mean()), 4)
    e = embed(enc, head, SERVER)
    r["server_cos_to_first"] = [round(float(x), 3) for x in (e @ e[0]).cpu()]
    if was_training:
        head.train()
    if enc.lora:
        enc.model.train()
    return r


def load_dev(data, rng):
    from datasets import load_dataset
    sts = load_dataset("sentence-transformers/stsb", split="validation")
    t = [x for x in jsonl(os.path.join(data, "dev", "triplets.jsonl")) if x["negative"]]
    rng.shuffle(t)
    rel = jsonl(os.path.join(data, "dev", "relations.jsonl"))
    rng.shuffle(rel)
    return {"sts": (list(sts["sentence1"]), list(sts["sentence2"]), np.array(sts["score"])),
            "pairs": jsonl(os.path.join(data, "dev", "pairs.jsonl")), "triplets": t[:3000], "relations": rel[:4000],
            "wic": jsonl(os.path.join(data, "dev", "wic.jsonl"))}


def cosent(cos, labels, tau=0.05):
    d = (cos[None, :] - cos[:, None]) / tau  # d[i, j] = (cos_j - cos_i)/tau, used where label_i > label_j
    want = (labels[:, None] > labels[None, :]).float()
    d = d - (1 - want) * 1e4
    return torch.logsumexp(torch.cat([torch.zeros(1, device=cos.device), d.flatten()]), 0)


def load_core(model_dir, run_dir):
    """Load a trained meaning core for inference: returns encode(texts) -> np.ndarray [N, 768] (unit vectors)."""
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    lora = meta["args"]["lora"]
    enc = Encoder(model_dir, 0)
    if lora:
        from peft import PeftModel
        enc.model = PeftModel.from_pretrained(enc.model, os.path.join(run_dir, "lora")).eval()
        enc.backbone = enc.model.get_base_model().model
    head = Head(meta["n_states"], enc.model.config.hidden_size, n_rel=len(meta["rel_names"])).cuda()
    head.load_state_dict(torch.load(os.path.join(run_dir, "head.pt"), map_location="cuda"))
    head.eval()

    @torch.no_grad()
    def encode(texts):
        return embed(enc, head, list(texts)).cpu().numpy()
    return encode


def baseline_encoder(model_dir, layer=4):
    enc = Encoder(model_dir, 0)
    base = Baseline(layer)

    @torch.no_grad()
    def encode(texts):
        return embed(enc, base, list(texts)).cpu().numpy()
    return encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lora", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--n-trip", type=int, default=48)
    ap.add_argument("--n-pair", type=int, default=64)
    ap.add_argument("--n-rel", type=int, default=64)
    ap.add_argument("--n-wic", type=int, default=32)
    ap.add_argument("--w-pair", type=float, default=1.0)
    ap.add_argument("--w-rel", type=float, default=0.3)
    ap.add_argument("--w-wic", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--eval-only-baseline", type=int, default=None)
    ap.add_argument("--token-budget", type=int, default=12000)
    ap.add_argument("--skip-eval0", action="store_true", help="speed tests only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    enc = Encoder(a.model, a.lora)
    dev = load_dev(a.data, rng)
    tr_rel = jsonl(os.path.join(a.data, "train", "relations.jsonl"))
    rel_names = sorted({x["relation"] for x in tr_rel})
    if a.eval_only_baseline is not None:
        r = evaluate(enc, Baseline(a.eval_only_baseline), dev, rel_names)
        r["variant"] = f"frozen LFM3 mean pooling, hidden state {a.eval_only_baseline}, no head"
        json.dump(r, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
        print(json.dumps(r))
        return
    trip = jsonl(os.path.join(a.data, "train", "triplets.jsonl"))
    pairs = jsonl(os.path.join(a.data, "train", "pairs.jsonl"))
    wic = jsonl(os.path.join(a.data, "train", "wic.jsonl"))
    by_rel = collections.defaultdict(list)
    for x in tr_rel:
        by_rel[x["relation"]].append(x)
    pos = collections.defaultdict(list)
    neg = collections.defaultdict(list)
    for p in pairs:
        (pos if p["label"] == 1 else neg)[p["objective"].split(":")[0] + ":" + p["source"] if p["label"] == 1 else p["objective"] + ":" + p["source"]].append(p)
    pos_w = {k: (0.85 if k.startswith("equivalence:paws") else 0.15) for k in pos}
    neg_w = {k: {"distinction:paws": 0.5, "composition:role_swap:propbank_frames": 0.15,
                 "distinction:antonym_substitution:oewn2025": 0.15}.get(k, 0.2 / max(1, sum(1 for q in neg if "condaqa" in q))) for k in neg}
    n_states = enc.model.config.num_hidden_layers + 1
    head = Head(n_states, enc.model.config.hidden_size, n_rel=len(rel_names)).cuda()
    params = [{"params": head.parameters(), "lr": a.lr}]
    if a.lora:
        params.append({"params": [p for p in enc.model.parameters() if p.requires_grad], "lr": a.lora_lr})
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    warm = min(200, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    meta = {"args": vars(a), "rel_names": rel_names, "n_states": n_states, "sizes": {
        "triplets": len(trip), "pairs": len(pairs), "relations": len(tr_rel), "wic": len(wic)},
        "pair_pools": {k: len(v) for k, v in list(pos.items()) + list(neg.items())}}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    hist = [] if a.skip_eval0 else [{"step": 0, **evaluate(enc, head, dev, rel_names)}]
    if hist:
        print(json.dumps(hist[-1]), flush=True)

    def pick(pools, weights, n):
        keys = list(pools)
        w = [weights[k] for k in keys]
        return [rng.choice(pools[k]) for k in rng.choices(keys, w, k=n)]

    t0, run = time.time(), collections.defaultdict(float)
    head.train()
    for step in range(1, a.steps + 1):
        # all sentence texts of the step in one forward (the backbone pass dominates the step), spans in a second
        tb = rng.sample(trip, a.n_trip)
        cands_txt = [x["positive"] for x in tb] + [x["negative"] for x in tb if x["negative"]]
        pb = pick(pos, pos_w, a.n_pair // 2) + pick(neg, neg_w, a.n_pair // 2)
        keys = rng.choices(rel_names, k=a.n_rel)
        rb = [rng.choice(by_rel[k]) for k in keys]
        groups = [[x["anchor"] for x in tb], cands_txt, [p["a"] for p in pb], [p["b"] for p in pb],
                  [x["a"] for x in rb], [x["b"] for x in rb]]
        flat = [t for g in groups for t in g]
        E = embed_bucketed(enc, head, flat, budget=a.token_budget, grad=True)
        parts, i = [], 0
        for g in groups:
            parts.append(E[i:i + len(g)])
            i += len(g)
        anchors, cands, ea, eb, u, v = parts
        l_trip = F.cross_entropy(anchors @ cands.T / 0.05, torch.arange(len(tb), device="cuda"))
        l_pair = cosent((ea * eb).sum(-1), torch.tensor([p["label"] for p in pb], device="cuda"))
        l_rel = F.cross_entropy(head.rel_logits(u, v), torch.tensor([rel_names.index(k) for k in keys], device="cuda"))
        wb = rng.sample(wic, a.n_wic)
        S = embed_spans(enc, head, [x["s1"] for x in wb] + [x["s2"] for x in wb],
                        [x["span1"] for x in wb] + [x["span2"] for x in wb], bs=2 * a.n_wic, grad=True)
        s1, s2 = S[:a.n_wic], S[a.n_wic:]
        l_wic = F.binary_cross_entropy_with_logits(head.wic_scale * (s1 * s2).sum(-1) + head.wic_bias,
                                                   torch.tensor([float(x["label"]) for x in wb], device="cuda"))
        loss = l_trip + a.w_pair * l_pair + a.w_rel * l_rel + a.w_wic * l_wic
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        for k, val in (("trip", l_trip), ("pair", l_pair), ("rel", l_rel), ("wic", l_wic), ("total", loss)):
            run[k] += val.detach().item()
        if step % 50 == 0:
            mix = torch.softmax(head.mix.detach(), 0).cpu().numpy()
            print(json.dumps({"step": step, **{k: round(v / 50, 4) for k, v in run.items()},
                              "top_states": [int(i) for i in np.argsort(mix)[::-1][:4]],
                              "s_per_step": round((time.time() - t0) / step, 3)}), flush=True)
            run.clear()
        if step % a.eval_every == 0 or step == a.steps:
            hist.append({"step": step, **evaluate(enc, head, dev, rel_names)})
            print(json.dumps(hist[-1]), flush=True)
            json.dump(hist, open(os.path.join(a.out, "eval_history.json"), "w"), indent=1)
            torch.save(head.state_dict(), os.path.join(a.out, "head.pt"))
            if a.lora:
                enc.model.save_pretrained(os.path.join(a.out, "lora"))
    mix = torch.softmax(head.mix.detach(), 0).cpu().numpy()
    json.dump({"state_mix": [round(float(x), 4) for x in mix], "final": hist[-1],
               "minutes": round((time.time() - t0) / 60, 1)}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
