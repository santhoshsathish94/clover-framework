"""B-link: what a frozen Qwen3-8B-Base state can do with a HybridLink, and what a bridge to HybridAgent-8B space does.

  b0  paper link (HybridAgent-8B adapter.pt) applied directly to BASE states: held-out next-token readout vs controls
  b1  link trained on BASE states (frozen base), objective cos + 0.2 MSE on unit-normalised vectors to the next token's embedding
      (my choice; the paper's exact normalisation is not in hand), two inits: random and the paper link; held-out readout vs controls
  b2  bridge BASE state -> HA state (ridge, closed form, fitted on train positions); then the paper link on bridged states
      (readout over the HA embedding table) compared with the paper link on the true HA states and on identity (= b0)
Inputs: states_{base,ha}_{train,held}.pt from b_states.py (same sequences, same order).
"""
import argparse, glob, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch
import torch.nn.functional as F
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=BASE + "/states")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--ridge", type=float, default=1e-2)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--out", default=BASE + "/results/b_link.json")
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)

BASE_SNAP = glob.glob("/root/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/*")[0]


def embed_table(snap):
    for f in sorted(glob.glob(snap + "/*.safetensors")):
        with safe_open(f, "pt") as g:
            if "model.embed_tokens.weight" in g.keys():
                return g.get_tensor("model.embed_tokens.weight").float()
    raise RuntimeError("no embed table in " + snap)


E_base, E_ha = embed_table(BASE_SNAP), embed_table(HA_SNAP)
bank_base, bank_ha = F.normalize(E_base, dim=-1), F.normalize(E_ha, dim=-1)
log("embed tables", tuple(E_base.shape), tuple(E_ha.shape))
D = lambda m, s: torch.load(f"{a.dir}/states_{m}_{s}.pt")
btr, bhe = D("base", "train"), D("base", "held")
log("base train/held positions", len(btr["y"]), len(bhe["y"]))
Yh = bhe["y"]


def readout(Z, Y, bank, idx=None):
    idx = torch.arange(len(Y)) if idx is None else idx
    t1 = t5 = c = 0.0; rk = []
    for s in range(0, len(idx), 128):
        j = idx[s:s + 128]
        sims = F.normalize(Z[j].float(), dim=-1) @ bank.T
        y = Y[j]
        top = sims.topk(5, -1).indices
        t1 += (top[:, 0] == y).float().sum().item(); t5 += (top == y[:, None]).any(-1).float().sum().item()
        tg = sims.gather(1, y[:, None]).squeeze(1); c += tg.sum().item(); rk.append((sims > tg[:, None]).sum(-1).float())
    n = len(idx)
    return {"top1": round(t1 / n, 4), "top5": round(t5 / n, 4), "mean_cos_to_next_embedding": round(c / n, 4),
            "median_rank": torch.cat(rk).median().item(), "n": n}


def controls(link, Hh, Y, bank, k=1500):
    g = torch.Generator().manual_seed(1)
    sub = torch.randperm(len(Y), generator=g)[:k]
    with torch.no_grad():
        Z = link(Hh[sub].float())
        rnd = torch.randn_like(Hh[sub].float()); rnd = rnd / rnd.norm(dim=-1, keepdim=True) * Hh[sub].float().norm(dim=-1, keepdim=True)
        return {"link": readout(Z, Y[sub], bank),
                "control_zeros": readout(torch.zeros_like(Z), Y[sub], bank),
                "control_random_state": readout(link(rnd), Y[sub], bank),
                "control_shuffled_z": readout(Z[torch.randperm(len(sub), generator=g)], Y[sub], bank),
                "control_raw_h_no_link": readout(Hh[sub].float(), Y[sub], bank)}


res = {"settings": {"steps": a.steps, "bs": a.bs, "lr": a.lr, "ridge": a.ridge}}
paper = load_link(4096, HA_SNAP + "/adapter.pt").float().eval()

# b0
log("b0 paper link on base states")
res["b0_paper_link_on_base_states__base_embedding_table"] = controls(paper, bhe["h"], Yh, bank_base)
res["b0_paper_link_on_base_states__ha_embedding_table"] = controls(paper, bhe["h"], Yh, bank_ha)
log(json.dumps(res["b0_paper_link_on_base_states__base_embedding_table"]["link"]))

# b1
Htr, Ytr = btr["h"], btr["y"]
Etgt = F.normalize(E_base[Ytr], dim=-1)


def train_link(init_paper, tag):
    m = load_link(4096, HA_SNAP + "/adapter.pt" if init_paper else None).float().train()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    g = torch.Generator().manual_seed(2)
    t0 = time.time(); hist = []
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * min(1.0, step / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        sel = torch.randint(0, len(Ytr), (a.bs,), generator=g)
        z = F.normalize(m(Htr[sel].float()), dim=-1)
        loss = (1 - (z * Etgt[sel]).sum(-1)).mean() + 0.2 * F.mse_loss(z, Etgt[sel])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == 1:
            hist.append((step, round(loss.item(), 4))); log(tag, step, round(loss.item(), 4), f"{(time.time() - t0) / step:.2f}s/step")
    return m.eval(), hist


for init_paper, tag in ((False, "b1_trained_from_random"), (True, "b1_trained_from_paper_link")):
    m, hist = train_link(init_paper, tag)
    r = controls(m, bhe["h"], Yh, bank_base); r["loss_history"] = hist
    res[tag] = r
    log(tag, json.dumps(r["link"]))
    torch.save(m.state_dict(), f"{a.dir}/{tag}.pt")

# b2
hh_tr, hh_he = D("ha", "train"), D("ha", "held")
assert torch.equal(hh_tr["y"], btr["y"]) and torch.equal(hh_he["y"], bhe["y"]), "state files not on identical sequences"
X, Yb = btr["h"].float(), hh_tr["h"].float()
mu_x, mu_y = X.mean(0), Yb.mean(0)
Xc, Yc = X - mu_x, Yb - mu_y
A = Xc.T @ Xc / len(X); A += a.ridge * A.diagonal().mean() * torch.eye(4096)
W = torch.linalg.solve(A, Xc.T @ Yc / len(X))
bridge = lambda h: (h.float() - mu_x) @ W + mu_y
Bh = bridge(bhe["h"]); Th = hh_he["h"].float()
cos = F.cosine_similarity(Bh, Th, dim=-1)
res["b2_bridge"] = {"mean_cos_bridged_vs_true_ha_state": round(cos.mean().item(), 4),
                    "mean_cos_identity_base_vs_ha_state": round(F.cosine_similarity(bhe["h"].float(), Th, dim=-1).mean().item(), 4),
                    "relative_mse_bridged": round(((Bh - Th) ** 2).sum().item() / (Th ** 2).sum().item(), 4)}
res["b2_paper_link_on_true_ha_states__ha_table"] = controls(paper, Th, Yh, bank_ha)
res["b2_paper_link_on_bridged_base_states__ha_table"] = controls(paper, Bh, Yh, bank_ha)
log("b2", json.dumps(res["b2_bridge"]))
json.dump(res, open(a.out, "w"), indent=1)
summ = {k: (v["link"] if isinstance(v, dict) and "link" in v else v) for k, v in res.items() if k != "settings"}
print(json.dumps(summ, indent=1))
