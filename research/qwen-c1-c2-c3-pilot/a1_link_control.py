"""A1 - positive control: the paper-trained HybridLink (adapter.pt of HybridAgent-8B-Agentic) read out on held-out Dolly text.

For each response position t: h = last hidden state (HF hidden_states[-1], what the HybridAgent engine feeds the link),
z = link(h); readout = nearest input-embedding row by cosine (the repo's own readout.py rule); target = response token t+1.
Controls: model's own lm_head (upper reference), raw h without link, zeros, random of matched norm, shuffled z.
The paper's latent steps need not decode to the next TEXT token, so a modest score is not a failure by itself; the
point is whether this instrument separates a paper-trained link from its controls, before it is used on Clover links.
Usage: python a1_link_control.py --n 120 --out results/a1.json
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--threads", type=int, default=24)
ap.add_argument("--ctrl-positions", type=int, default=1500)
ap.add_argument("--out", default=BASE + "/results/a1.json")
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(0); random.seed(0)

tok = AutoTokenizer.from_pretrained(HA_SNAP)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(HA_SNAP, dtype=torch.bfloat16).eval()
log("model loaded", round(time.time() - t0, 1), "s")
link = load_link(4096, HA_SNAP + "/adapter.pt").to(torch.bfloat16).eval()

rows = [r for r in dolly_rows(tok=tok) if r["split"] == "held"][: a.n]
log("held-out prompts:", len(rows))

H, Y, SEQ = [], [], []
ntok, tf = 0, time.time()
with torch.no_grad():
    for i, r in enumerate(rows):
        ids, pl = build_ids(tok, r["instruction"], r["response"])
        x = torch.tensor([ids])
        out = model(x, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][0]                      # [T, D] bf16
        lg = out.logits[0]
        pos = list(range(pl - 1, len(ids) - 1))            # predicts ids[pos+1] (response tokens)
        H.append(h[pos].float()); Y += [ids[p + 1] for p in pos]; SEQ += [i] * len(pos)
        H[-1] = (H[-1], lg[pos].float().argmax(-1), lg[pos].float().topk(5, -1).indices)
        ntok += len(ids)
        if i % 10 == 0:
            log(f"{i}/{len(rows)} fwd {ntok / (time.time() - tf):.1f} tok/s")
lm1 = torch.cat([x[1] for x in H]); lm5 = torch.cat([x[2] for x in H]); Hs = torch.cat([x[0] for x in H])
Y = torch.tensor(Y); N = len(Y)
log("positions:", N, "fwd tok/s", round(ntok / (time.time() - tf), 1))

E = model.get_input_embeddings().weight.detach().float()
bank = F.normalize(E, dim=-1)
with torch.no_grad():
    Z = torch.cat([link(Hs[i:i + 256].to(torch.bfloat16)).float() for i in range(0, N, 256)])


def readout(Zin, idx):
    top1, top5, cosT, rank = [], [], [], []
    for s in range(0, len(idx), 128):
        j = idx[s:s + 128]
        z = F.normalize(Zin[j], dim=-1)
        sims = z @ bank.T
        t = Y[j]
        top = sims.topk(5, -1).indices
        top1.append((top[:, 0] == t).float()); top5.append((top == t[:, None]).any(-1).float())
        tgt = sims.gather(1, t[:, None]).squeeze(1)
        cosT.append(tgt); rank.append((sims > tgt[:, None]).sum(-1).float())
    c = lambda L: torch.cat(L)
    return {"top1": c(top1).mean().item(), "top5": c(top5).mean().item(), "mean_cos_to_next_embedding": c(cosT).mean().item(),
            "median_rank_of_next_token": c(rank).median().item(), "n": len(idx)}


g = torch.Generator().manual_seed(1)
allidx = torch.arange(N)
sub = allidx[torch.randperm(N, generator=g)[: min(a.ctrl_positions, N)]]
res = {"n_prompts": len(rows), "n_positions": N, "model": "HybridAgent-8B-Agentic (adapter.pt = paper link)",
       "fwd_tokens_per_s": round(ntok / (time.time() - tf), 1)}
res["lm_head_own_top1"] = (lm1 == Y).float().mean().item()
res["lm_head_own_top5"] = (lm5 == Y[:, None]).any(-1).float().mean().item()
res["link"] = readout(Z, allidx)
res["link_on_control_subset"] = readout(Z, sub)
res["control_raw_h_no_link"] = readout(Hs, sub)
res["control_zeros"] = readout(torch.zeros_like(Z), sub)
rnd = torch.randn_like(Hs); rnd = rnd / rnd.norm(dim=-1, keepdim=True) * Hs.norm(dim=-1, keepdim=True)
with torch.no_grad():
    Zr = torch.cat([link(rnd[i:i + 256].to(torch.bfloat16)).float() for i in range(0, N, 256)])
res["control_random_state_through_link"] = readout(Zr, sub)
res["control_shuffled_z"] = readout(Z[torch.randperm(N, generator=g)], sub)
res["samples"] = []
for k in range(min(3, len(rows))):
    idx = (torch.tensor(SEQ) == k).nonzero().squeeze(1)[:24]
    z = F.normalize(Z[idx], dim=-1); top = (z @ bank.T).argmax(-1)
    res["samples"].append({"prompt": rows[k]["instruction"], "true_next": [tok.decode([int(t)]) for t in Y[idx]],
                           "link_readout": [tok.decode([int(t)]) for t in top], "lm_head_top1": [tok.decode([int(t)]) for t in lm1[idx]]})
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
log("wrote", a.out)
print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=1))
