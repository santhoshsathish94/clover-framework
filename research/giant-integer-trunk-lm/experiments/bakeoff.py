#!/usr/bin/env python3
"""Independent Block-TrunkLM-24 bake-off: exact replication of Variants A/B/C + Variant D (Claude Opus 5.5).

Budget (identical for all variants): 24 layers, D=512, B=64, 3 x (8x8x64) 4-bit Block-Circulant grids + 512x32 4-bit
causal taps per layer (28,672 u4 weights/layer = 336.0 KB nibble-packed over 24 layers), frozen PCA-tied teacher
embeddings, O(D) vectors only, STEPS=200, BS=16, SEQ=128, evaluated with 100% hard 4-bit trunk weights.

Protocols:
  orig      : vocab + PCA from tokens of ALL 160 chunks (train + val), exactly as distill_omni_to_block_trunklm.py
  trainpca  : same vocab as orig, but PCA basis fit ONLY on tokens seen in train chunks 0..127
  full128k  : full 128,000-token teacher vocab, PCA fit on the full teacher embedding matrix (zero data dependence)
"""
from __future__ import annotations
import argparse, importlib.util, json, math, struct, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("orig_distill", HERE / "distill_omni_to_block_trunklm.py")
orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orig)
quantize_u4 = orig.quantize_u4

D_MODEL, BLOCK, N_LAYERS, K_TAPS = 512, 64, 24, 32


# ----------------------------------------------------------------------------------------------- Variant D model
class ProjD(nn.Module):
    """y = d_out * BlockCirc(d_in * x[perm]); 8x8 grid of B=64 circulants with hard 4-bit STE weights."""

    def __init__(self, zero_init=False, perm=None, d_in=False, d_out=False, d_out_init=1.0):
        super().__init__()
        self.nb = D_MODEL // BLOCK
        scale = 0.002 if zero_init else (0.5 / math.sqrt(D_MODEL))
        self.c = nn.Parameter(torch.randn(self.nb, self.nb, BLOCK) * scale)
        self.register_buffer("perm", perm if perm is not None else torch.empty(0, dtype=torch.long), persistent=True)
        self.d_in = nn.Parameter(torch.ones(D_MODEL)) if d_in else None
        self.d_out = nn.Parameter(torch.full((D_MODEL,), float(d_out_init))) if d_out else None

    def forward(self, x, soft_alpha=1.0):
        if self.perm.numel():
            x = x[..., self.perm]
        if self.d_in is not None:
            x = x * self.d_in
        c_q, _, _ = quantize_u4(self.c, soft_alpha)
        shp = x.shape[:-1]
        xs = x.reshape(-1, self.nb, BLOCK)
        yf = torch.einsum("tib,oib->tob", torch.fft.rfft(xs, n=BLOCK, dim=-1), torch.fft.rfft(c_q, n=BLOCK, dim=-1))
        y = torch.fft.irfft(yf, n=BLOCK, dim=-1).reshape(*shp, D_MODEL)
        if self.d_out is not None:
            y = y * self.d_out
        return y


def butterfly_perm():
    idx = torch.arange(512).reshape(8, 8, 8).transpose(0, 1).reshape(512)
    return idx  # new[b*64+a*8+c] = old[a*64+b*8+c]


class LayerD(nn.Module):
    def __init__(self, li, cfg):
        super().__init__()
        g = torch.Generator().manual_seed(1000 + li)
        def perm_for(kind):
            if kind == "none":
                return None
            if kind == "butterfly":
                return butterfly_perm()
            if kind == "random":
                return torch.randperm(D_MODEL, generator=g)
            if kind == "stride":  # coprime multi-stride shuffle, stride differs per layer
                strides = [5, 7, 11, 13, 17, 19, 23, 29]
                s = strides[li % len(strides)]
                return (torch.arange(D_MODEL) * s) % D_MODEL
            raise ValueError(kind)
        eps = cfg["eps"]
        self.norm1 = nn.RMSNorm(D_MODEL, eps=eps)
        self.time_w = nn.Parameter(torch.randn(D_MODEL, K_TAPS) * (0.3 / math.sqrt(K_TAPS)))
        self.gate_proj = ProjD(perm=perm_for(cfg["perm_gate"]), d_in=cfg["acdc"], d_out=cfg["acdc"])
        self.norm2 = nn.RMSNorm(D_MODEL, eps=eps)
        self.w1 = ProjD(perm=perm_for(cfg["perm_w1"]), d_in=cfg["acdc"], d_out=False)
        self.w2 = ProjD(zero_init=cfg["w2_zero_init"], perm=perm_for(cfg["perm_w2"]), d_in=cfg["acdc"], d_out=cfg["acdc"],
                        d_out_init=cfg["res_init"])
        self.diag_scale = nn.Parameter(torch.ones(D_MODEL) * 0.25)
        self.conv_out = nn.Parameter(torch.full((D_MODEL,), cfg["res_init"])) if cfg["acdc"] else None
        self.k = K_TAPS
        self.drop = cfg["drop"]

    def forward(self, x, soft_alpha=1.0):
        h = self.norm1(x)
        tw_q, _, _ = quantize_u4(self.time_w, soft_alpha)
        gate = torch.sigmoid(self.gate_proj(h, soft_alpha))
        conv = F.conv1d(F.pad(h.transpose(1, 2), (self.k - 1, 0)), tw_q.flip(-1).unsqueeze(1), groups=D_MODEL).transpose(1, 2)
        if self.conv_out is not None:
            conv = conv * self.conv_out
        x = x + F.dropout(gate * conv, self.drop, self.training)
        h2 = self.norm2(x)
        u = F.silu(self.w1(h2, soft_alpha) * self.diag_scale)
        return x + F.dropout(self.w2(u, soft_alpha), self.drop, self.training)


class ModelD(nn.Module):
    def __init__(self, embed_init, cfg):
        super().__init__()
        self.register_buffer("E", embed_init.clone(), persistent=False)  # frozen, tied
        self.layers = nn.ModuleList([LayerD(i, cfg) for i in range(N_LAYERS)])
        self.final_norm = nn.RMSNorm(D_MODEL, eps=cfg["eps"])
        self.logit_scale = nn.Parameter(torch.tensor(3.0))
        # O(D) output prior: logits += E @ h_bias (lets the frozen tied head express a unigram-like prior)
        self.h_bias = nn.Parameter(torch.zeros(D_MODEL)) if cfg["hbias"] else None

    def forward(self, idx, soft_alpha=1.0):
        x = F.embedding(idx, self.E)
        h12 = None
        for i, l in enumerate(self.layers):
            x = l(x, soft_alpha)
            if i == 11:
                h12 = x
        h = self.final_norm(x) * self.logit_scale
        if self.h_bias is not None:
            h = h + self.h_bias
        return h @ self.E.T, h12, h


# ----------------------------------------------------------------------------------------------- data/protocols
def build_data(cache, protocol):
    n_all = cache["input_ids"].size(0)
    if protocol in ("orig", "trainpca") and n_all > 160:  # fresh-test chunks (160..) only exist for full128k
        cache = {k: (v[:160] if k != "embed" else v) for k, v in cache.items()}
    inp, tgt = cache["input_ids"], cache["target_ids"]
    top_ids, top_vals = cache["top_ids"], cache["top_vals"]
    E_full = cache["embed"]
    if protocol in ("orig", "trainpca"):
        uniq = torch.unique(torch.cat([inp.flatten(), tgt.flatten(), top_ids[:, :, :32].flatten().long()]))
        e_t = E_full[uniq].float()
        if protocol == "orig":
            _, _, vh = torch.linalg.svd(e_t, full_matrices=False)
        else:
            uniq_tr = torch.unique(torch.cat([inp[:128].flatten(), tgt[:128].flatten(), top_ids[:128, :, :32].flatten().long()]))
            _, _, vh = torch.linalg.svd(E_full[uniq_tr].float(), full_matrices=False)
        p = vh[:512].T.contiguous()
        old2new = torch.full((128000,), -1, dtype=torch.long)
        old2new[uniq] = torch.arange(uniq.numel())
        emb = (e_t @ p).contiguous()
    elif protocol == "full128k":
        Ef = E_full.float()
        evals, evecs = torch.linalg.eigh(Ef.T @ Ef)  # uncentred PCA == right singular vectors (same as orig recipe)
        p = evecs[:, -512:].flip(-1).contiguous()
        old2new = torch.arange(128000)
        emb = (Ef @ p).contiguous()
        del Ef
    else:
        raise ValueError(protocol)
    d = dict(
        V=emb.size(0), emb=emb, inp=old2new[inp], tgt=old2new[tgt],
        t32=old2new[top_ids[:, :, :32].long()], v32=top_vals[:, :, :32].float(), lse=cache["lse"].float(),
        h12p=cache["h12"].float() @ p, h24p=cache["h24"].float() @ p,
    )
    assert int((d["t32"] < 0).sum()) == 0 and int((d["tgt"] < 0).sum()) == 0
    return d


@torch.no_grad()
def evaluate(model, d, is_orig_cls, lo=128, hi=160):
    model.eval()
    V = d["V"]
    tot = dict(nll=0.0, kl_orig=0.0, kl_manual=0.0, kl_tail=0.0, top1=0.0)
    nb = 0
    for i in range(lo, hi, 8):
        xb, yb, ti, tv, lse = d["inp"][i:i + 8], d["tgt"][i:i + 8], d["t32"][i:i + 8], d["v32"][i:i + 8], d["lse"][i:i + 8]
        logits = model(xb, soft_alpha=1.0)[0]
        nll = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
        p_t = F.softmax(tv, dim=-1)
        lps = F.log_softmax(torch.gather(logits, -1, ti), dim=-1)
        kl_orig = F.kl_div(lps, p_t, reduction="batchmean") / xb.size(1)            # exactly the original metric
        kl_manual = (p_t * (p_t.clamp_min(1e-30).log() - lps)).sum(-1).mean()        # explicit per-token mean
        # Tail-aware KL over {top-32} U {tail} using absolute teacher probs (full 128K LSE) & full student softmax.
        lq = F.log_softmax(logits, dim=-1)
        lq32 = torch.gather(lq, -1, ti)
        lp32 = tv - lse.unsqueeze(-1)
        p32 = lp32.exp()
        p_tail = (1 - p32.sum(-1)).clamp_min(1e-12)
        q_tail = (1 - lq32.exp().sum(-1)).clamp_min(1e-12)
        kl_tail = ((p32 * (lp32 - lq32)).sum(-1) + p_tail * (p_tail.log() - q_tail.log())).mean()
        top1 = (logits.argmax(-1) == ti[:, :, 0]).float().mean()
        for k, v in (("nll", nll), ("kl_orig", kl_orig), ("kl_manual", kl_manual), ("kl_tail", kl_tail), ("top1", top1)):
            tot[k] += float(v)
        nb += 1
    r = {k: v / nb for k, v in tot.items()}
    r["ppl"] = math.exp(r["nll"])
    r["top1"] *= 100.0
    return r


def kd_terms(logits, ti, tv, lse, t_kd, tail):
    if tail:  # T=1 sparse KL with tail bucket (absolute teacher probabilities)
        lq = F.log_softmax(logits, dim=-1)
        lq32 = torch.gather(lq, -1, ti)
        lp32 = tv - lse.unsqueeze(-1)
        p32 = lp32.exp()
        p_tail = (1 - p32.sum(-1)).clamp_min(1e-6)
        q_tail = (1 - lq32.exp().sum(-1)).clamp_min(1e-6)
        return ((p32 * (lp32 - lq32)).sum(-1) + p_tail * (p_tail.log() - q_tail.log())).mean()
    p_t = F.softmax(tv / t_kd, dim=-1)
    s_top = torch.gather(logits, -1, ti) / t_kd
    log_z = torch.logsumexp(logits / t_kd, dim=-1, keepdim=True)
    return -(p_t * (s_top - log_z)).sum(dim=-1).mean() * t_kd


def train(name, d, steps, kind, cfg=None, log_every=50, seed=100):
    V = d["V"]
    torch.manual_seed(seed)
    if kind in ("A", "B", "C"):
        use_kd, use_bp, use_hints = {"A": (False, False, False), "B": (True, False, False), "C": (True, True, True)}[kind]
        m = orig.BlockTrunkLM24(d["emb"], d_model=512, block_b=64, n_layers=24, k_taps=32, butterfly_perm=use_bp)
        trainable = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=3e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-4)
    else:
        use_kd, use_hints = True, cfg["hints"] > 0
        m = ModelD(d["emb"], cfg)
        if cfg["hbias"] == "ls":  # least-squares fit of E @ b to the smoothed TRAIN-ONLY unigram log-prior
            cnt = torch.bincount(d["tgt"][:128].flatten(), minlength=V).double() + 0.1
            lp = (cnt / cnt.sum()).log()
            w = (cnt / cnt.sum()).sqrt().unsqueeze(1)       # frequency-weighted least squares
            E = d["emb"].double()
            sol = torch.linalg.lstsq(E * w, ((lp - lp.mean()) * w.squeeze(1)).unsqueeze(1)).solution.squeeze(1)
            m.h_bias.data.copy_(sol.float())
        vec = [p for n, p in m.named_parameters() if p.ndim <= 1]
        mat = [p for n, p in m.named_parameters() if p.ndim > 1]
        trainable = vec + mat
        opt = torch.optim.AdamW([{"params": mat, "lr": cfg["lr"], "weight_decay": cfg["wd"]},
                                 {"params": vec, "lr": cfg["lr"] * cfg["vec_lr_mult"], "weight_decay": 0.0}],
                                betas=(0.9, cfg["beta2"]))
        warm = cfg["warmup"]
        def lr_lambda(st):
            if st < warm:
                return (st + 1) / warm
            pr = (st - warm) / max(1, steps - warm)
            return cfg["min_lr_frac"] + (1 - cfg["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * pr))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    t0 = time.perf_counter()
    m.train()
    for st in range(1, steps + 1):
        perm = torch.randint(0, 128, (16,))
        xb, yb = d["inp"][perm], d["tgt"][perm]
        if kind == "C":
            alpha = min(1.0, 0.3 + 0.7 * (st / (0.5 * steps)))
        elif kind == "D":
            alpha = min(1.0, cfg["alpha0"] + (1 - cfg["alpha0"]) * (st / (cfg["alpha_frac"] * steps))) if cfg["alpha_frac"] > 0 else 1.0
        else:
            alpha = 1.0
        logits, h12, h24 = m(xb, soft_alpha=alpha)
        ce = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
        if not use_kd:
            loss = ce
        elif kind in ("B", "C"):
            kd = kd_terms(logits, d["t32"][perm], d["v32"][perm], None, 1.5, False)
            loss = 0.7 * kd + 0.3 * ce
            if use_hints:
                loss = loss + 0.35 * ((1 - F.cosine_similarity(h12, d["h12p"][perm], dim=-1).mean())
                                      + (1 - F.cosine_similarity(h24, d["h24p"][perm], dim=-1).mean()))
        else:
            kd = kd_terms(logits, d["t32"][perm], d["v32"][perm], d["lse"][perm], cfg["t_kd"], cfg["tail"])
            if cfg["kd_mix"] > 0:
                kd = kd + cfg["kd_mix"] * kd_terms(logits, d["t32"][perm], d["v32"][perm], None, cfg["t_mix"], False)
            loss = cfg["w_kd"] * kd + cfg["w_ce"] * ce
            if use_hints:
                loss = loss + cfg["hints"] * ((1 - F.cosine_similarity(h12, d["h12p"][perm], dim=-1).mean())
                                              + (1 - F.cosine_similarity(h24, d["h24p"][perm], dim=-1).mean()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        if kind == "D" and cfg["ema"] > 0:
            if st == 1:
                ema = [p.detach().clone() for p in trainable]
            dec = min(cfg["ema"], (1 + st) / (10 + st))
            with torch.no_grad():
                for e, p in zip(ema, trainable):
                    e.mul_(dec).add_(p.detach(), alpha=1 - dec)
        if st % log_every == 0:
            print(f"  [{name}] step {st} loss={loss.item():.4f} ce={ce.item():.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    if kind == "D" and cfg["ema"] > 0:
        with torch.no_grad():
            for e, p in zip(ema, trainable):
                p.copy_(e)
    r = evaluate(m, d, kind != "D")
    if d["inp"].size(0) >= 192:
        r["fresh_test_160_191"] = evaluate(m, d, kind != "D", 160, 192)
    r["train_s"] = time.perf_counter() - t0
    r["name"] = name
    return m, r


DEFAULT_D = dict(eps=1e-5, acdc=True, perm_gate="none", perm_w1="none", perm_w2="butterfly", w2_zero_init=True,
                 res_init=1.0, lr=3e-3, vec_lr_mult=1.0, wd=0.01, beta2=0.999, warmup=0, min_lr_frac=2e-4 / 3e-3,
                 alpha0=0.3, alpha_frac=0.5, t_kd=1.0, tail=True, w_kd=0.9, w_ce=0.1, hints=1.0, drop=0.2,
                 hbias=False, kd_mix=0.0, t_mix=1.5, ema=0.0)  # final Variant D recipe (= ablation run "D14")


# ----------------------------------------------------------------------------------------------- export (Variant D)
def export_d(m, d, out_dir, cfg, n_seq=2, T=64):
    out_dir.mkdir(parents=True, exist_ok=True)
    m.eval()
    flags = 0
    with torch.no_grad():
        xs = d["inp"][128:128 + n_seq, :T].contiguous()
        ref = m(xs, soft_alpha=1.0)[0].float().contiguous()                 # [n_seq, T, V]
        with open(out_dir / "sample_d.bin", "wb") as f:
            f.write(struct.pack("<II", n_seq, T))
            f.write(xs.numpy().astype(np.int32).tobytes())
            f.write(ref.numpy().astype(np.float32).tobytes())
        V = d["V"]
        with open(out_dir / "block_trunklm24_d.bin", "wb") as bf:
            bf.write(b"BTD1")
            bf.write(struct.pack("<IIIIIf", N_LAYERS, D_MODEL, BLOCK, K_TAPS, V, float(cfg["eps"])))
            bf.write(d["emb"].numpy().astype(np.float32).tobytes())
            bf.write((m.final_norm.weight * m.logit_scale).numpy().astype(np.float32).tobytes())
            hb = m.h_bias if m.h_bias is not None else torch.zeros(D_MODEL)
            bf.write(hb.numpy().astype(np.float32).tobytes())
            ones = torch.ones(D_MODEL)
            for l in m.layers:
                bf.write(l.norm1.weight.numpy().astype(np.float32).tobytes())
                bf.write(l.norm2.weight.numpy().astype(np.float32).tobytes())
                bf.write(l.diag_scale.numpy().astype(np.float32).tobytes())
                _, tw_u4, tw_s = quantize_u4(l.time_w, 1.0)
                if l.conv_out is not None:
                    tw_s = tw_s * l.conv_out                       # per-channel conv output gain folds into the tap scale
                bf.write(tw_s.numpy().astype(np.float32).tobytes())
                bf.write(tw_u4.numpy().tobytes())
                for p in (l.gate_proj, l.w1, l.w2):
                    perm = p.perm if p.perm.numel() else torch.arange(D_MODEL)
                    bf.write(perm.numpy().astype(np.uint16).tobytes())
                    bf.write((p.d_in if p.d_in is not None else ones).numpy().astype(np.float32).tobytes())
                    bf.write((p.d_out if p.d_out is not None else ones).numpy().astype(np.float32).tobytes())
                    _, c_u4, c_s = quantize_u4(p.c, 1.0)
                    bf.write(c_s.numpy().astype(np.float32).tobytes())
                    bf.write(c_u4.numpy().tobytes())
    trunk_u4 = sum(l.time_w.numel() + l.gate_proj.c.numel() + l.w1.c.numel() + l.w2.c.numel() for l in m.layers)
    vec = sum(p.numel() for n, p in m.named_parameters() if p.ndim <= 1)
    return dict(trunk_u4_weights=trunk_u4, trunk_kb_nibble=trunk_u4 / 2 / 1024, o_d_vector_params=vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/xc_variant_d/teacher_cache.pt")
    ap.add_argument("--out-dir", default="/tmp/xc_variant_d")
    ap.add_argument("--protocol", default="orig")
    ap.add_argument("--variants", default="A,B,C,D")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dcfg", default="{}")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--export", default="")
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    cache = torch.load(a.cache)
    d = build_data(cache, a.protocol)
    print(f"protocol={a.protocol} V={d['V']}", flush=True)
    cfg = dict(DEFAULT_D); cfg.update(json.loads(a.dcfg))
    results = []
    for v in a.variants.split(","):
        m, r = train(f"{v}", d, a.steps, v, cfg if v == "D" else None, seed=a.seed)
        r["seed"] = a.seed
        r["protocol"] = a.protocol
        if v == "D":
            r["cfg"] = cfg
            if a.export:
                r["budget"] = export_d(m, d, Path(a.export), cfg)
        elif a.export:  # original-format export for the ORIGINAL block_trunklm_c.c verifier (verbatim logic)
            od = Path(a.export); od.mkdir(parents=True, exist_ok=True)
            m.eval()
            with torch.no_grad():
                sx = d["inp"][128:129, :32]
                ref_lg, _, _ = m(sx, soft_alpha=1.0)
                (od / "sample_tokens.bin").write_bytes(sx[0].numpy().astype(np.int32).tobytes())
                (od / "ref_val_logits.bin").write_bytes(ref_lg[0, -1].numpy().astype(np.float32).tobytes())
                with open(od / "block_trunklm24.bin", "wb") as bf:
                    bf.write(struct.pack("<IIIIII", 24, 512, 64, 32, d["V"], 1 if m.layers[0].butterfly_perm else 0))
                    bf.write(m.embed.weight.numpy().astype(np.float32).tobytes())
                    bf.write((m.final_norm.weight * m.logit_scale).numpy().astype(np.float32).tobytes())
                    for lyr in m.layers:
                        for t_ in (lyr.norm1.weight, lyr.norm2.weight, lyr.diag_scale):
                            bf.write(t_.numpy().astype(np.float32).tobytes())
                        _, tw_u4, tw_s = quantize_u4(lyr.time_w, 1.0)
                        bf.write(tw_s.numpy().astype(np.float32).tobytes()); bf.write(tw_u4.numpy().tobytes())
                        for proj in (lyr.gate_proj, lyr.w1, lyr.w2):
                            _, c_u4, c_s = quantize_u4(proj.c, 1.0)
                            bf.write(c_s.numpy().astype(np.float32).tobytes()); bf.write(c_u4.numpy().tobytes())
                # also dump an all-position reference (2 seqs x 64) for a stricter parity check
                xs = d["inp"][128:130, :64].contiguous()
                ref = m(xs, soft_alpha=1.0)[0].float()
                with open(od / "sample_all.bin", "wb") as f:
                    f.write(struct.pack("<II", 2, 64)); f.write(xs.numpy().astype(np.int32).tobytes())
                    f.write(ref.numpy().astype(np.float32).tobytes())
        print(json.dumps(r), flush=True)
        results.append(r)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / f"results_{a.tag}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
