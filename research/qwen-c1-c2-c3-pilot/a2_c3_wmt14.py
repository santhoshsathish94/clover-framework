"""A2 - C3 gate on the Qwen vocabulary: the released WMT14 NCM (Qwen3-0.6B vocabulary, 64-token windows, 96-token raw prefix),
fine-tuned so that a PLAN in the prefix is rendered as text. Dolly (no-context) responses of 12-63 Qwen tokens are the target windows.

Plan = grounded plan (as Cycle 48 of the research context): the response's own content words and numbers in order of first
appearance (words >= 4 letters, small stop list, at most 30 items). It carries the facts but not the sentences.
Training code is adapted from the study file /root/ncm-cpu/ncm_stage2.py (loss, corruption, loop); the WMT14 generator already has
prefix_len 96, raw_token_prefix and predict_level0, so no shapes change.

Subcommands:  recon  (tokenizer reconstruction on held-out windows, no training)
              eval   (matched / shuffled / blank plan; --ckpt omitted = the released translation generator, zero-shot)
              train  (fine-tune the generator, tokenizer frozen)
"""
import argparse, json, math, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ROOT = "/root/ncm-cpu/repo"
ART = "/root/ncm-cpu/artifacts/wmt14"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

STOP = set("the and was were that with this they them then there their have had has her his she him for not but are you all can from out one very when what said would could about into just like over some also than been will your more most only such these those which while where who whom whose does did doing being both each other another its it's".split())
NUM = re.compile(r"\d[\d,\.]*\d|\d")
L_WIN, P_PREFIX = 64, 96


def load_models(device, ckpt=None):
    CFG = OmegaConf.load(ROOT + "/config/eval/wmt14.yaml")
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ART, CFG.checkpoints.tokenizer)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    for p in tk.vqvae.parameters():
        p.requires_grad_(False)
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(device)
    src = torch.load(ckpt or os.path.join(ART, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
    own = net.state_dict()
    keep, dropped = {}, []
    for k, v in src["model"].items():
        if k in own and own[k].shape == v.shape:
            keep[k] = v
        else:
            dropped.append(k)
    missing, unexpected = net.load_state_dict(keep, strict=False)
    return CFG, tk, net, src, {"dropped": dropped, "missing": list(missing), "unexpected": list(unexpected)}


def plan_of(text):
    seen, items = set(), []
    for m in re.finditer(r"[A-Za-z]{4,}|\d[\d,\.]*\d|\d", text):
        w = m.group(0)
        key = w.lower()
        if key in STOP or key in seen:
            continue
        seen.add(key); items.append(w)
        if len(items) >= 30:
            break
    return " ".join(items)


def make_pairs(tk, which, n=None):
    eos, pad = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
    out = []
    for r in dolly_rows():
        if r["split"] != which:
            continue
        ids = tk.pre_tokenizer.encode(r["response"])
        if not (12 <= len(ids) <= L_WIN - 1):
            continue
        w = ids + [eos] + [pad] * (L_WIN - len(ids) - 1)
        out.append({"q": r["instruction"], "text": r["response"], "ids": w, "plan": plan_of(r["response"]), "n": len(ids)})
        if n and len(out) >= n:
            break
    return out


def plan_ids_of(tk, plans):
    pad = tk.pre_tokenizer.pad_token_id
    out = []
    for t in plans:
        ids = tk.pre_tokenizer.encode(t)[:P_PREFIX]
        out.append(ids + [pad] * (P_PREFIX - len(ids)))
    return torch.tensor(out)


def retention(plan, text):
    items = plan.split()
    if not items:
        return None
    tl = text.lower()
    tw = set(w[:5] for w in re.findall(r"[a-z]+", tl))
    tn = set(NUM.findall(text))
    hit = 0
    for w in items:
        hit += (w in tn) if w[0].isdigit() else (w.lower()[:5] in tw)
    return hit / len(items)


def number_recall(true_text, gen_text):
    a = NUM.findall(true_text)
    if not a:
        return None
    g = list(NUM.findall(gen_text))
    k = 0
    for x in a:
        if x in g:
            g.remove(x); k += 1
    return k / len(a)


def corrupt(idx, levels, vocab, p_first, p_last):
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


def step_loss(net, tk, plan_ids, idx, pad_id, rates=None):
    vq = tk.vqvae.vq
    levels = list(vq.levels)
    with torch.no_grad():
        idx_in = corrupt(idx, levels, vq.vqvae_vocab_sizes, *rates) if rates else idx
        ncm_in = vq.idx_to_ncm_input(idx_in)
        pref = tk.vqvae.encoder.wte(plan_ids)
    logits = net(torch.cat([pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=(plan_ids == pad_id))
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.float().reshape(-1, V), idx.reshape(-1), reduction="none").view(B, T)
    loss = ce.mean()
    lid = level_ids(levels, idx.device)
    with torch.no_grad():
        per_level = torch.stack([ce[:, lid == k].mean() for k in range(len(levels))])
    return loss, per_level


@torch.no_grad()
def encode_windows(tk, ids, device, bs=64):
    xs = torch.tensor(ids)
    return torch.cat([tk.vqvae.encode(xs[i:i + bs].to(device)).cpu() for i in range(0, len(xs), bs)]).to(torch.int16)


def dec_window(tk, r):
    eos, pad = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
    r = list(r)
    if eos in r:
        r = r[: r.index(eos)]
    return tk.pre_tokenizer.decode([t for t in r if t not in (eos, pad)], skip_special_tokens=True)


def cmd_recon(a):
    dev = torch.device("cpu")
    CFG, tk, net, src, rep = load_models(dev)
    pairs = make_pairs(tk, "held", a.n)
    X = torch.tensor([p["ids"] for p in pairs])
    with torch.no_grad():
        idx = torch.cat([tk.vqvae.encode(X[i:i + 32]) for i in range(0, len(X), 32)])
        R = torch.cat([tk.vqvae.decode(idx[i:i + 32]).argmax(-1) for i in range(0, len(idx), 32)])
    hit = tot = exact = 0; nr = []
    for i, p in enumerate(pairs):
        n = p["n"]
        eq = R[i, :n] == X[i, :n]
        hit += int(eq.sum()); tot += n; exact += int(bool(eq.all()))
        x = number_recall(p["text"], dec_window(tk, R[i].tolist()))
        if x is not None:
            nr.append(x)
    res = {"windows": len(pairs), "token_acc": round(hit / tot, 4), "window_exact": round(exact / len(pairs), 4),
           "number_recall": round(sum(nr) / max(len(nr), 1), 4),
           "examples": [{"orig": p["text"], "recon": dec_window(tk, R[i].tolist())} for i, p in enumerate(pairs[:3])]}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}))


@torch.no_grad()
def generate_texts(net, tk, pids, temp, top_k, top_p, bs, seed):
    eos = tk.pre_tokenizer.eos_token_id
    torch.manual_seed(seed)
    outs = []
    for i in range(0, len(pids), bs):
        g = net.generate(None, pids[i:i + bs], temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=0.0, cfg_null_eos=eos)
        _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        outs += [dec_window(tk, r) for r in tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()]
    return outs


def cmd_eval(a):
    dev = torch.device("cpu")
    CFG, tk, net, src, rep = load_models(dev, a.ckpt)
    net.eval()
    pad = tk.pre_tokenizer.pad_token_id
    pairs = make_pairs(tk, "held", a.n)
    plans = [p["plan"] for p in pairs]
    pids = plan_ids_of(tk, plans)
    res = {"ckpt": a.ckpt or "released WMT14 generator (zero-shot)", "n": len(pairs), "load": {k: len(v) for k, v in rep.items()}, "settings": []}
    for temp, top_k, top_p in ((1.0, 0, 1.0), (0.6, 50, 0.9)):
        t0 = time.time()
        matched = generate_texts(net, tk, pids, temp, top_k, top_p, a.bs, 42)
        shuffled = generate_texts(net, tk, torch.roll(pids, 1, 0), temp, top_k, top_p, a.bs, 42)
        blank = generate_texts(net, tk, torch.full_like(pids, pad), temp, top_k, top_p, a.bs, 42)
        true = [p["text"] for p in pairs]
        ret = lambda texts: round(sum(x for x in (retention(p, t) for p, t in zip(plans, texts)) if x is not None) / len(plans), 4)
        nrc = lambda texts: [x for x in (number_recall(t, g) for t, g in zip(true, texts)) if x is not None]
        mean = lambda v: round(sum(v) / len(v), 4) if v else None
        res["settings"].append({"temp": temp, "top_k": top_k, "top_p": top_p, "seconds": round(time.time() - t0, 1),
            "retention_matched": ret(matched), "retention_shuffled_plan(control)": ret(shuffled), "retention_blank_plan(control)": ret(blank),
            "retention_true_text(ceiling)": ret(true), "number_recall_matched": mean(nrc(matched)),
            "number_recall_shuffled(control)": mean(nrc(shuffled)), "number_recall_blank(control)": mean(nrc(blank)),
            "examples": [{"q": p["q"], "plan": p["plan"], "generated": m, "shuffled_plan_generated": s, "blank_plan_generated": b, "true": p["text"]}
                         for p, m, s, b in zip(pairs[:6], matched[:6], shuffled[:6], blank[:6])]})
        print(json.dumps({k: v for k, v in res["settings"][-1].items() if k != "examples"}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


def cmd_train(a):
    dev = torch.device("cpu")
    torch.set_num_threads(a.threads)
    CFG, tk, net, src, rep = load_models(dev)
    pad = tk.pre_tokenizer.pad_token_id
    tr = make_pairs(tk, "train"); va = make_pairs(tk, "held", a.nval)
    log(f"train pairs {len(tr)}, val pairs {len(va)}; load report", {k: len(v) for k, v in rep.items()})
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    idx_tr = encode_windows(tk, [p["ids"] for p in tr], dev); pl_tr = plan_ids_of(tk, [p["plan"] for p in tr])
    idx_va = encode_windows(tk, [p["ids"] for p in va], dev).long(); pl_va = plan_ids_of(tk, [p["plan"] for p in va])
    log(f"encoded in {time.time() - t0:.1f}s")
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    rates = (a.corrupt_first, a.corrupt_last)
    g = torch.Generator().manual_seed(a.seed); torch.manual_seed(a.seed)
    logf = open(os.path.join(a.out, "log.jsonl"), "a")

    def evaluate():
        net.eval(); res = {}
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=a.bf16):
            for name, pl in (("matched", pl_va), ("shuffled", torch.roll(pl_va, 1, 0)), ("blank", torch.full_like(pl_va, pad))):
                tot = 0
                for i in range(0, len(idx_va), a.bs):
                    _, pls = step_loss(net, tk, pl[i:i + a.bs], idx_va[i:i + a.bs], pad, None)
                    tot = tot + pls * len(pl[i:i + a.bs])
                res[name] = (tot / len(idx_va)).tolist()
        net.train(); return res

    ev = evaluate()
    m = lambda e: {k: round(sum(v) / len(v), 4) for k, v in e.items()}
    log("step 0 val CE (mean over levels)", m(ev)); logf.write(json.dumps({"step": 0, "val": ev}) + "\n"); logf.flush()
    net.train(); t0 = time.time()
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * sched(step - 1)
        sel = torch.randint(0, len(idx_tr), (a.bs,), generator=g)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=a.bf16):
            loss, _ = step_loss(net, tk, pl_tr[sel], idx_tr[sel].long(), pad, rates)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % a.log_every == 0 or step == 1:
            rec = {"step": step, "loss": round(loss.item(), 4), "gnorm": round(float(gn), 3), "s_per_step": round((time.time() - t0) / step, 3)}
            log(rec); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(); log(f"step {step} val CE", m(ev)); logf.write(json.dumps({"step": step, "val": ev}) + "\n"); logf.flush()
            tmp = os.path.join(a.out, "last.pt.tmp")
            torch.save({"model": net.state_dict(), "step": step, "args": vars(a)}, tmp); os.replace(tmp, os.path.join(a.out, "last.pt"))
    log("train done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recon"); r.add_argument("--n", type=int, default=200); r.add_argument("--out", required=True)
    e = sub.add_parser("eval"); e.add_argument("--ckpt", default=None); e.add_argument("--n", type=int, default=64)
    e.add_argument("--bs", type=int, default=16); e.add_argument("--out", required=True)
    t = sub.add_parser("train"); t.add_argument("--out", required=True); t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--bs", type=int, default=16); t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--corrupt-first", type=float, default=0.9); t.add_argument("--corrupt-last", type=float, default=0.1)
    t.add_argument("--nval", type=int, default=64); t.add_argument("--eval-every", type=int, default=250); t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--seed", type=int, default=0); t.add_argument("--threads", type=int, default=16)
    t.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    {"recon": cmd_recon, "eval": cmd_eval, "train": cmd_train}[a.cmd](a)
