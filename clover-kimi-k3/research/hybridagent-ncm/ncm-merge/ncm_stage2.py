"""Stage-2 (plan-conditioned) training of the released TinyStories NCM generator, tokenizer frozen. Not part of the NCM repo: written for this study.

What the repo ships (read, not changed): VQVAETokenizer, NextConceptTransformer.forward (logits for all levels in one pass), vq.idx_to_ncm_input,
generate(idx=None, prefix) when predict_level0 is on. What the repo does NOT ship and this file adds: the data/pairs pipeline, the loss, the
conditioning corruption, the optimiser loop, the evaluation. The loss, weights and corruption follow the paper's description (Sec. 3.4, App. F);
learning rate, schedule and the exact corruption schedule are MY choices (the paper's values were not available to me) and are flags.

Model change relative to the released generator: prefix_len 1 -> P (plan tokens, right-padded with PAD) and predict_level0 false -> true
(one new head `lm_head_l0` reads the last non-pad plan position). Everything else is loaded from generator.pt; keys whose shape differs
(the attention mask buffer, which depends on P) are rebuilt by the constructor and reported.

Usage:  python ncm_stage2.py train --pairs pairs.jsonl --out run1 --steps 2000 --bs 16 --device cpu
        python ncm_stage2.py eval  --pairs pairs.jsonl --ckpt run1/last.pt --out eval1.json --n 64
pairs.jsonl: one json per line {"ids": [128 BPE ids], "plan": "text"}.
"""
import argparse, json, math, os, random, re, sys, time

ROOT = os.environ.get("NCM_REPO", "/root/ncm-cpu/repo")
ART = os.environ.get("NCM_ART", "/root/ncm-cpu/artifacts/tinystories")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from models.next_concept import NextConceptTransformer, _TiedCodebookHead
from tokenization.vqvae_tokenizer import VQVAETokenizer


def load_models(device, P, ckpt=None, l0_head=None):
    """l0_head: 'tied' (released recipe: logits from distances to the level-0 codebook; its entries have norm ~4e10, so the logits cannot depend on the plan),
    'tiednorm' (same head with the codebook divided by its rms norm) or 'untied' (plain linear). None = the type recorded in the checkpoint, else 'tied'."""
    CFG = OmegaConf.load("config/eval/tinystories.yaml")
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ART, CFG.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ART, CFG.checkpoints[name])
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    for p in tk.vqvae.parameters():
        p.requires_grad_(False)
    mcfg = CFG.model
    mcfg.prefix_len = P
    mcfg.predict_level0 = True
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=mcfg).to(device)
    src = torch.load(ckpt or os.path.join(ART, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
    ck_mode = src.get("l0_head", "tied")
    mode = l0_head or ck_mode
    if mode == "untied":
        net.lm_head_l0 = torch.nn.Linear(mcfg.n_embd, tk.vqvae.vq.vqvae_vocab_sizes[0], bias=False).to(device)
    elif mode == "tiednorm":
        E = tk.vqvae.vq.embeddings[0].weight.detach().double()
        s = float((E ** 2).sum(-1).mean().sqrt())
        net.lm_head_l0 = _TiedCodebookHead(mcfg.n_embd, (E / s).float(), learn_scale=True, trainable=False, use_bias=False).to(device)
    net.l0_head_mode = mode
    own = net.state_dict()
    keep, dropped = {}, []
    for k, v in src["model"].items():
        if mode != ck_mode and k.startswith("lm_head_l0"):
            dropped.append((k, "re-initialised: head type changed from %s to %s" % (ck_mode, mode), None))
            continue
        if k in own and own[k].shape == v.shape:
            keep[k] = v
        else:
            dropped.append((k, tuple(v.shape), tuple(own[k].shape) if k in own else None))
    missing, unexpected = net.load_state_dict(keep, strict=False)
    return CFG, tk, net, src, {"l0_head": mode, "dropped_shape_mismatch": dropped, "missing": list(missing), "unexpected": list(unexpected)}


def corrupt(idx, levels, vocab, p_first, p_last):
    """Replace input codes by uniform random codes of the same level. Level k (k = 0..K-2) feeds level k+1; the last level is never an input.
    Rate falls linearly from p_first (level 0) to p_last (level K-2). Targets are never touched (caller keeps `idx`)."""
    out = idx.clone()
    off, n_in = 0, len(levels) - 1
    for k in range(n_in):
        l = levels[k]
        p = p_first + (p_last - p_first) * (k / max(n_in - 1, 1))
        m = torch.rand(idx.shape[0], l, device=idx.device) < p
        r = torch.randint(0, vocab[k], (idx.shape[0], l), device=idx.device)
        out[:, off:off + l] = torch.where(m, r, idx[:, off:off + l])
        off += l
    return out


def level_ids(levels, device):
    return torch.cat([torch.full((l,), k, dtype=torch.long) for k, l in enumerate(levels)]).to(device)


def step_loss(net, tk, plan_ids, idx, pad_id, corrupt_rates=None, alpha=0.0):
    vq = tk.vqvae.vq
    levels = list(vq.levels)
    with torch.no_grad():
        idx_in = corrupt(idx, levels, vq.vqvae_vocab_sizes, *corrupt_rates) if corrupt_rates else idx
        ncm_in = vq.idx_to_ncm_input(idx_in)
        pref = tk.vqvae.encoder.wte(plan_ids)
    logits = net(torch.cat([pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=(plan_ids == pad_id))
    B, T, V = logits.shape
    assert T == idx.shape[1], (T, idx.shape)
    ce = F.cross_entropy(logits.float().reshape(-1, V), idx.reshape(-1), reduction="none").view(B, T)
    lid = level_ids(levels, idx.device)
    ls = torch.tensor(levels, dtype=torch.float32, device=idx.device)
    w = (ls ** (-alpha))[lid]                       # per-position weight, paper w_k ~ l_k^-alpha (alpha=0: every code equal)
    loss = (ce * w[None]).sum() / (w.sum() * B)
    with torch.no_grad():
        per_level = torch.stack([ce[:, lid == k].mean() for k in range(len(levels))])
    return loss, per_level


def read_pairs(path, n=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows[:n] if n else rows


def encode_plans(tk, plans, P, pad_id):
    out = []
    for t in plans:
        ids = tk.pre_tokenizer.encode(t)[:P]
        out.append(ids + [pad_id] * (P - len(ids)))
    return torch.tensor(out)


@torch.no_grad()
def encode_stories(tk, ids, device, bs=64):
    xs = torch.tensor(ids)
    return torch.cat([tk.vqvae.encode(xs[i:i + bs].to(device)).cpu() for i in range(0, len(xs), bs)]).to(torch.int16)


def cmd_train(a):
    import torch.distributed as dist
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    main = rank == 0
    if ws > 1:
        lr_ = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(lr_)
        dist.init_process_group("nccl")
        dev = torch.device("cuda", lr_)
    else:
        dev = torch.device(a.device)
    CFG, tk, net, src, rep = load_models(dev, a.P, ckpt=(a.init or None), l0_head=(a.l0_head or None))
    pad_id = tk.pre_tokenizer.pad_token_id
    print("load report:", json.dumps(rep), flush=True)
    rows = read_pairs(a.pairs)
    if a.val_pairs:
        vrows = read_pairs(a.val_pairs)[:a.nval]
        rows, nval = vrows + rows, len(vrows)
    else:
        nval = min(a.nval, len(rows) // 10)
    val, tr = rows[:nval], rows[nval:]
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "idx_cache.pt")
    t0 = time.perf_counter()
    idx_all = encode_stories(tk, [r["ids"] for r in rows], dev)
    plans_all = encode_plans(tk, [r["plan"] for r in rows], a.P, pad_id)
    print(f"encoded {len(rows)} stories in {time.perf_counter()-t0:.1f}s; train {len(tr)} val {len(val)}", flush=True)
    idx_val, plan_val = idx_all[:nval].long().to(dev), plans_all[:nval].to(dev)
    idx_tr, plan_tr = idx_all[nval:], plans_all[nval:]
    params = [p for p in net.parameters() if p.requires_grad]
    print("trainable params", sum(p.numel() for p in params) / 1e6, "M", "world", ws, flush=True)
    fwd = net
    if ws > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        fwd = DDP(net, device_ids=[dev.index])
    head_params = [p for n_, p in net.named_parameters() if p.requires_grad and n_.startswith("lm_head_l0")]
    body_params = [p for n_, p in net.named_parameters() if p.requires_grad and not n_.startswith("lm_head_l0")]
    opt = torch.optim.AdamW([{"params": body_params, "base_lr": a.lr, "lr": a.lr},
                             {"params": head_params, "base_lr": a.lr * a.head_lr_mult, "lr": a.lr * a.head_lr_mult}],
                            betas=(0.9, 0.95), weight_decay=0.01)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    use_amp = dev.type == "cuda"
    log = open(os.path.join(a.out, "log.jsonl"), "a") if main else open(os.devnull, "w")
    rates = None if a.corrupt_mean <= 0 else (a.corrupt_first, a.corrupt_last)
    g = torch.Generator().manual_seed(a.seed + rank)
    torch.manual_seed(a.seed + rank)

    def evaluate():
        net.eval()
        res = {}
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            for name, pl in (("matched", plan_val), ("shuffled", torch.roll(plan_val, 1, 0))):
                tot, n = None, 0
                for i in range(0, len(idx_val), a.bs):
                    l, pls = step_loss(net, tk, pl[i:i + a.bs], idx_val[i:i + a.bs], pad_id, None, 0.0)
                    tot = pls * len(pl[i:i + a.bs]) if tot is None else tot + pls * len(pl[i:i + a.bs])
                    n += len(pl[i:i + a.bs])
                res[name] = (tot / n).tolist()
        net.train()
        return res

    if main:
        ev = evaluate()
        print("step 0 val CE matched/shuffled (mean over levels):", sum(ev["matched"]) / 12, sum(ev["shuffled"]) / 12, flush=True)
        log.write(json.dumps({"step": 0, "val": ev}) + "\n"); log.flush()
    if ws > 1:
        dist.barrier()
    net.train()
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = gp["base_lr"] * sched(step - 1)
        sel = torch.randint(0, len(idx_tr), (a.bs,), generator=g)
        idx = idx_tr[sel].long().to(dev)
        plan = plan_tr[sel].to(dev)
        if a.plan_dropout > 0:
            drop = torch.rand(len(plan), generator=g) < a.plan_dropout
            if drop.any():
                null = torch.full((a.P,), pad_id); null[0] = tk.pre_tokenizer.eos_token_id
                plan[drop.to(dev)] = null.to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, pls = step_loss(fwd, tk, plan, idx, pad_id, rates, a.alpha)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if main and (step % a.log_every == 0 or step == 1):
            rec = {"step": step, "loss": round(loss.item(), 4), "gnorm": round(float(gn), 3), "lr": opt.param_groups[0]["lr"], "s_per_step": round((time.perf_counter() - t0) / step, 3)}
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            if main:
                ev = evaluate()
                print(f"step {step} val CE matched/shuffled:", round(sum(ev["matched"]) / 12, 4), round(sum(ev["shuffled"]) / 12, 4), flush=True)
                log.write(json.dumps({"step": step, "val": ev}) + "\n"); log.flush()
                tmp = os.path.join(a.out, "last.pt.tmp")
                torch.save({"model": net.state_dict(), "inference": src.get("inference"), "step": step, "args": vars(a), "l0_head": net.l0_head_mode}, tmp)
                os.replace(tmp, os.path.join(a.out, "last.pt"))
                if a.keep_every and step % a.keep_every == 0:
                    torch.save({"model": net.state_dict(), "inference": src.get("inference"), "step": step, "l0_head": net.l0_head_mode}, os.path.join(a.out, f"step{step}.pt"))
            if ws > 1:
                dist.barrier()
            net.train()
    print("done")


STOP = set("the and was were that with this they them then there their have had has her his she him for not but are you all can from out one very when what said would could about into just like over some day".split())


def plan_words(t):
    return [w for w in dict.fromkeys(re.findall(r"[a-z]+", t.lower())) if len(w) >= 4 and w not in STOP]


def retention(plan, text):
    """Fraction of the plan's content words (len>=4, not in a small stop list) that occur in the text; a word counts if the text has a word
    sharing its first 5 letters (crude stem)."""
    pw = plan_words(plan)
    if not pw:
        return None
    tw = set(w[:5] for w in re.findall(r"[a-z]+", text.lower()))
    return sum(1 for w in pw if w[:5] in tw) / len(pw)


@torch.no_grad()
def cmd_eval(a):
    dev = torch.device(a.device)
    CFG, tk, net, src, rep = load_models(dev, a.P, ckpt=a.ckpt)
    net.eval()
    pad_id, eos = tk.pre_tokenizer.pad_token_id, tk.pre_tokenizer.eos_token_id
    print("load report:", json.dumps(rep), flush=True)
    rows = read_pairs(a.pairs)[: a.n]
    plans = [r["plan"] for r in rows]
    plan_ids = encode_plans(tk, plans, a.P, pad_id).to(dev)

    def dec(r):
        r = list(r)
        if eos in r:
            r = r[: r.index(eos)]
        return tk.pre_tokenizer.decode([t for t in r if t not in (eos, pad_id)], skip_special_tokens=True)

    def gen(pids, seed):
        torch.manual_seed(seed)
        outs = []
        for i in range(0, len(pids), a.bs):
            g = net.generate(None, pids[i:i + a.bs], temperature=a.temp, top_k=a.top_k, top_p=a.top_p, cfg_scale=0.0, cfg_null_eos=eos)
            _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
            outs += [dec(r) for r in tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()]
        return outs

    res = {"n": len(rows), "sampling": {"temp": a.temp, "top_k": a.top_k, "top_p": a.top_p}, "seeds": a.seeds, "load": rep}
    t0 = time.perf_counter()
    mk = lambda texts, pls: [x for x in (retention(p, t) for p, t in zip(pls, texts)) if x is not None]
    mean = lambda v: round(sum(v) / len(v), 4) if v else None
    m_ret, s_ret, matched = [], [], None
    for sd in range(a.seeds):
        m = gen(plan_ids, 42 + sd)
        s = gen(torch.roll(plan_ids, 1, 0), 42 + sd)
        matched = matched or m
        m_ret.append(mean(mk(m, plans))); s_ret.append(mean(mk(s, plans)))
    has_true = all("ids" in r for r in rows)
    true_text = [dec(r["ids"]) for r in rows] if has_true else [""] * len(rows)
    res["retention"] = {
        "matched_plan_generated (mean over seeds)": mean(m_ret), "matched per seed": m_ret,
        "shuffled_plan_generated (control: plan of another story, scored against this plan)": mean(s_ret), "shuffled per seed": s_ret,
        "true_story (ceiling)": mean(mk(true_text, plans)) if has_true else None,
    }
    res["gen_seconds"] = round(time.perf_counter() - t0, 1)
    res["examples"] = [{"plan": p, "generated": m, "true": t} for p, m, t in zip(plans[:8], matched[:8], true_text[:8])]
    res["all_matched_seed42"] = [{"plan": p, "generated": m} for p, m in zip(plans, matched)]
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["retention"], indent=1))
    for e in res["examples"][:4]:
        print("PLAN:", e["plan"]); print("GEN :", e["generated"][:400]); print("TRUE:", e["true"][:400]); print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--pairs", required=True); t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=1000); t.add_argument("--bs", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-4); t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--P", type=int, default=48); t.add_argument("--alpha", type=float, default=0.0)
    t.add_argument("--corrupt-mean", type=float, default=0.5)
    t.add_argument("--corrupt-first", type=float, default=0.9); t.add_argument("--corrupt-last", type=float, default=0.1)
    t.add_argument("--nval", type=int, default=64); t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--log-every", type=int, default=25); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu"); t.add_argument("--keep-every", type=int, default=0)
    t.add_argument("--val-pairs", default=""); t.add_argument("--init", default="")
    t.add_argument("--l0-head", default="", choices=["", "tied", "tiednorm", "untied"]); t.add_argument("--head-lr-mult", type=float, default=1.0)
    t.add_argument("--plan-dropout", type=float, default=0.0)
    e = sub.add_parser("eval")
    e.add_argument("--pairs", required=True); e.add_argument("--ckpt", required=True); e.add_argument("--out", required=True)
    e.add_argument("--n", type=int, default=64); e.add_argument("--bs", type=int, default=16); e.add_argument("--P", type=int, default=48)
    e.add_argument("--temp", type=float, default=1.0); e.add_argument("--top-k", type=int, default=0); e.add_argument("--top-p", type=float, default=1.0)
    e.add_argument("--device", default="cpu"); e.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)
