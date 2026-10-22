"""Quality metrics for plan-conditioned generation (run on the gpu box). Imports the loader/retention from ncm_stage2.
Per config: generate stories from every plan in --pairs (3 sampling seeds, or K candidates + MBR selection when --cands K > 1), then report
  - retention of plan words (as in Cycle 31),
  - Gen. PPL under GPT-2-large (exp of total NLL over total tokens; same reference model as the paper's TinyStories table), for the generated and for the true stories,
  - adjacent repeated-word rate ("started started"), mean per-sample unigram entropy of GPT-2 tokens.
Decoding options: --temp/--top-k/--top-p (level sampling), --adj "a,w,d" adjacency penalty on the decoder logits (as the repo does for XSum/LM1B), --cfg (needs a plan-dropout model).
MBR (--cands K): K sampled candidates per plan, keep the one with the highest mean word-set Jaccard similarity to the other K-1 (never looks at the plan)."""
import argparse, json, math, os, re, sys, time
import ncm_stage2 as S
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=598); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--P", type=int, default=64)
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--cands", type=int, default=1)
ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--top-k", type=int, default=0); ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--adj", default=""); ap.add_argument("--cfg", type=float, default=0.0); ap.add_argument("--name", default="")
ap.add_argument("--ngram", default="")
a = ap.parse_args()
dev = torch.device("cuda:0")
CFG, tk, net, src, rep = S.load_models(dev, a.P, ckpt=a.ckpt)
net.eval()
pad_id, eos = tk.pre_tokenizer.pad_token_id, tk.pre_tokenizer.eos_token_id
rows = S.read_pairs(a.pairs)[: a.n]
plans = [r["plan"] for r in rows]
plan_ids = S.encode_plans(tk, plans, a.P, pad_id).to(dev)
ADJ = [float(x) for x in a.adj.split(",")] if a.adj else None
NG = [int(a.ngram.split(",")[0]), float(a.ngram.split(",")[1])] if a.ngram else None


def dec(r):
    r = list(r)
    if eos in r:
        r = r[: r.index(eos)]
    return tk.pre_tokenizer.decode([t for t in r if t not in (eos, pad_id)], skip_special_tokens=True)


def pick(logits):
    if not ADJ and not NG:
        return logits.argmax(-1)
    al, aw, ad = (ADJ[0], int(ADJ[1]), ADJ[2]) if ADJ else (0.0, 0, 1.0)
    B, T, V = logits.shape
    lg = logits.double()
    out = torch.empty(B, T, dtype=torch.long, device=lg.device)
    hist = []
    toks = [[] for _ in range(B)]
    seen = [dict() for _ in range(B)]
    for t in range(T):
        step = lg[:, t, :].clone()
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64, device=lg.device))
        if NG and t >= NG[0] - 1:
            for b in range(B):
                nxt = seen[b].get(tuple(toks[b][-(NG[0] - 1):]))
                if nxt:
                    step[b, list(nxt)] -= NG[1]
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        if NG:
            for b, v in enumerate(p.tolist()):
                toks[b].append(v)
                if len(toks[b]) >= NG[0]:
                    seen[b].setdefault(tuple(toks[b][-NG[0]:-1]), set()).add(v)
    return out


@torch.no_grad()
def gen(pids, seed):
    torch.manual_seed(seed)
    outs = []
    for i in range(0, len(pids), a.bs):
        g = net.generate(None, pids[i:i + a.bs], temperature=a.temp, top_k=a.top_k, top_p=a.top_p, cfg_scale=a.cfg, cfg_null_eos=eos)
        _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        outs += [dec(r) for r in pick(tk.vqvae.decoder(zl[-1])).tolist()]
    return outs


words = lambda t: re.findall(r"[a-z']+", t.lower())


def jacc(x, y):
    sx, sy = set(words(x)), set(words(y))
    return len(sx & sy) / max(len(sx | sy), 1)


def mbr(cands):
    K = len(cands)
    best = []
    for i in range(len(cands[0])):
        sc = [sum(jacc(cands[k][i], cands[j][i]) for j in range(K) if j != k) for k in range(K)]
        best.append(cands[max(range(K), key=lambda k: sc[k])][i])
    return best


gtok = AutoTokenizer.from_pretrained("gpt2-large")
gm = AutoModelForCausalLM.from_pretrained("gpt2-large", dtype=torch.float16).to(dev).eval()
sys.path.insert(0, S.ROOT)
from evaluation.gen_ppl import generation_perplexity


def gen_ppl(texts):
    r = generation_perplexity(gm, gtok, [gtok.eos_token] * len(texts), texts, device="cuda:0", batch_size=32)
    return math.exp(float(r.nll_sum.sum() / r.token_count.sum()))


def rep_rate(texts):
    n = d = 0
    for t in texts:
        w = words(t)
        n += sum(1 for i in range(1, len(w)) if w[i] == w[i - 1]); d += max(len(w) - 1, 0)
    return n / max(d, 1)


def rep3_rate(texts):
    n = d = 0
    for t in texts:
        w = words(t); seen = set()
        for i in range(len(w) - 2):
            g = tuple(w[i:i + 3])
            n += g in seen; d += 1; seen.add(g)
    return n / max(d, 1)


def entropy(texts):
    hs = []
    for t in texts:
        ids = gtok(t)["input_ids"]
        if len(ids) < 2:
            continue
        c = {}
        for i in ids:
            c[i] = c.get(i, 0) + 1
        hs.append(-sum(v / len(ids) * math.log(v / len(ids)) for v in c.values()))
    return sum(hs) / len(hs)


def metrics(texts):
    r = [S.retention(p, t) for p, t in zip(plans, texts)]
    r = [x for x in r if x is not None]
    return {"retention": sum(r) / len(r), "gen_ppl": gen_ppl(texts), "adj_repeat_rate": rep_rate(texts), "repeat_trigram_rate": rep3_rate(texts), "entropy": entropy(texts)}


t0 = time.perf_counter()
true_text = [dec(r["ids"]) for r in rows] if all("ids" in r for r in rows) else None
res = {"name": a.name, "n": len(rows), "decode": {"temp": a.temp, "top_k": a.top_k, "top_p": a.top_p, "adj": a.adj, "ngram": a.ngram, "cands": a.cands, "cfg": a.cfg}}
if a.cands > 1:
    cands = [gen(plan_ids, 42 + k) for k in range(a.cands)]
    texts_per = [mbr(cands)]
    res["single_candidate_reference"] = metrics(cands[0])
else:
    texts_per = [gen(plan_ids, 42 + s) for s in range(a.seeds)]
ms = [metrics(t) for t in texts_per]
res["generated"] = {k: round(sum(m[k] for m in ms) / len(ms), 4) for k in ms[0]}
res["generated_per_run"] = [{k: round(v, 4) for k, v in m.items()} for m in ms]
if true_text:
    res["true"] = {k: round(v, 4) for k, v in metrics(true_text).items()}
res["examples"] = [{"plan": p, "generated": g} for p, g in zip(plans[:6], texts_per[0][:6])]
res["all_generated_first_run"] = texts_per[0]
res["seconds"] = round(time.perf_counter() - t0, 1)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print(a.name, json.dumps({"generated": res["generated"], "true": res.get("true")}))
