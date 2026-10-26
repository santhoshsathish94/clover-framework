"""Cycle 44: CausalVQDecoder for NCM Stage-2 (applicable to both Track 2B e4_fluency and Track 2A e4_latent).
Why: Diagnosis in Cycle 43 showed that the remaining fluency glitches ('on an big airplane', 'in the. Billy', 'play to played')
come from tk.vqvae.decoder being a 1-layer non-causal parallel readout that selects token[t] independently of token[0..t-1].
CausalVQDecoder (4 causal transformer blocks, d=384, n_head=6, initialized from VQ-VAE wte + output_layer) conditions on:
  - z_hat[:, t, :] (the 128-position concept reconstruction from the 12-pass NCM generator, using coarse+mid+fine levels), AND
  - prev_tokens[:, 0..t-1] (strictly causal self-attention over previously emitted tokens).
Trained with teacher-forced next-token cross-entropy on 78,823 stories where z_hat is corrupted at fine levels (matching generator
inference noise) so the causal decoder learns to realize the concepts in z_hat with clean English grammar.
Then evaluates both Track 2B (e4_fluency + CausalVQDecoder) and Track 2A (e4_latent + CausalVQDecoder) on all 598 validation plans
and 48 e2e topics.
"""
import argparse, json, math, os, re, sys, time
from clover_paths import setup_clover_sys_path

ROOT = setup_clover_sys_path()
ART = os.environ.get("NCM_ART", ROOT)
if os.path.isdir(ROOT):
    os.chdir(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import ncm_stage2 as S
from eval_both_tracks import words, jacc, mbr, generate_from_zpref
from train_latent_bridge import LatentBridge, P_LEN, D_HA


class CausalBlock(nn.Module):
    def __init__(self, d=384, n_head=6, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d = d
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln_cx = nn.LayerNorm(d)
        self.ln_z = nn.LayerNorm(d)
        self.cx_q = nn.Linear(d, d)
        self.cx_kv = nn.Linear(d, 2 * d)
        self.cx_proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, z_mem):
        B, T, C = x.shape
        nh = self.n_head
        hd = C // nh
        # 1. Strictly causal self-attention over [0..t]
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=-1)
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))

        # 2. Cross-attention to full 128-position z_hat concept memory
        hq = self.ln_cx(x)
        hz = self.ln_z(z_mem)
        cq = self.cx_q(hq).view(B, T, nh, hd).transpose(1, 2)
        ck, cv = self.cx_kv(hz).split(C, dim=-1)
        ck = ck.view(B, z_mem.shape[1], nh, hd).transpose(1, 2)
        cv = cv.view(B, z_mem.shape[1], nh, hd).transpose(1, 2)
        cy = F.scaled_dot_product_attention(cq, ck, cv, is_causal=False, dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.cx_proj(cy.transpose(1, 2).contiguous().view(B, T, C)))

        # 3. MLP
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class CausalVQDecoder(nn.Module):
    def __init__(self, vqvae, n_layer=4, n_head=6, dropout=0.1):
        super().__init__()
        d = vqvae.embed_dim
        V = vqvae.in_vocab_size
        L = vqvae.context_length
        self.d = d
        self.V = V
        self.L = L
        self.wte = nn.Embedding(V, d)
        self.wpe = nn.Embedding(L, d)
        self.z_ln = nn.LayerNorm(d)
        self.z_proj = nn.Linear(d, d)
        self.z_pos = nn.Embedding(L, d)
        self.blocks = nn.ModuleList([CausalBlock(d, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, V, bias=False)

        with torch.no_grad():
            self.wte.weight.copy_(vqvae.encoder.wte.weight)
            self.lm_head.weight.copy_(vqvae.decoder.output_layer.weight)
            nn.init.normal_(self.wpe.weight, std=0.02)
            nn.init.normal_(self.z_pos.weight, std=0.02)

    def forward(self, z_hat, prev_ids):
        B, T, _ = prev_ids.shape[0], prev_ids.shape[1], self.d
        pos = torch.arange(T, device=prev_ids.device)
        z_pos = torch.arange(z_hat.shape[1], device=z_hat.device)
        z_mem = self.z_ln(z_hat) + self.z_pos(z_pos)[None, :, :]
        x = self.wte(prev_ids) + self.wpe(pos)[None, :, :] + self.z_proj(self.z_ln(z_hat[:, :T, :]))
        for blk in self.blocks:
            x = blk(x, z_mem)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate_tokens(self, z_hat, eos_id, temp=0.3, adj=(4.0, 6, 0.7), ng=(3, 6.0)):
        B, L, _ = z_hat.shape
        dev = z_hat.device
        prev = torch.full((B, 1), eos_id, dtype=torch.long, device=dev)
        out_ids = []
        hist = []
        toks = [[] for _ in range(B)]
        seen = [dict() for _ in range(B)]
        al, aw, ad = adj if adj else (0.0, 0, 1.0)

        for t in range(L):
            logits = self.forward(z_hat, prev)[:, -1, :].double()
            for j, h in enumerate(reversed(hist[-aw:])):
                logits.scatter_add_(1, h[:, None], torch.full((B, 1), -al * (ad ** j), dtype=torch.float64, device=dev))
            if ng and t >= ng[0] - 1:
                for b in range(B):
                    nxt = seen[b].get(tuple(toks[b][-(ng[0] - 1):]))
                    if nxt:
                        logits[b, list(nxt)] -= ng[1]
            if temp <= 0.0:
                p = logits.argmax(dim=-1)
            else:
                probs = F.softmax((logits / temp).float(), dim=-1)
                # Top-k=8 nucleus to keep high-probability tokens only
                vals, inds = torch.topk(probs, k=8, dim=-1)
                vals = vals / vals.sum(dim=-1, keepdim=True)
                sel = torch.multinomial(vals, 1).squeeze(-1)
                p = inds.gather(1, sel[:, None]).squeeze(-1)
            out_ids.append(p)
            hist.append(p)
            if ng:
                for b, v in enumerate(p.tolist()):
                    toks[b].append(v)
                    if len(toks[b]) >= ng[0]:
                        seen[b].setdefault(tuple(toks[b][-ng[0]:-1]), set()).add(v)
            prev = torch.cat([prev, p[:, None]], dim=1)
        return torch.stack(out_ids, dim=1)


@torch.no_grad()
def build_noisy_zhat(vq, idx, g, p_corrupt_coarse=0.10, p_corrupt_fine=0.45, p_drop_fine=0.35):
    """Simulate inference-time z_hat by corrupting fine VQ codes and randomly dropping levels 8..11."""
    B = idx.shape[0]
    levels = list(vq.levels)
    L_full = levels[-1]
    idx_c = S.corrupt(idx, levels, vq.vqvae_vocab_sizes, p_corrupt_coarse, p_corrupt_fine)
    z_hat = torch.zeros(B, L_full, vq.embed_dim, device=idx.device)
    # Random cutoff level per sample (with probability p_drop_fine, stop between level 6 and 10)
    keep_lvl = torch.full((B,), len(levels), dtype=torch.long, device=idx.device)
    drop_m = torch.rand(B, device=idx.device) < p_drop_fine
    if drop_m.any():
        keep_lvl[drop_m] = torch.randint(6, len(levels), (int(drop_m.sum().item()),), device=idx.device)

    off = 0
    for k, l_k in enumerate(levels):
        emb_k = vq.embeddings[k](idx_c[:, off:off + l_k])
        if k != len(levels) - 1:
            z_q_k = vq._up_to_full(emb_k, k, L_full)
        else:
            z_q_k = emb_k
        z_q_k = vq._apply_quant_resi(z_q_k, k)
        active = (k < keep_lvl).float()[:, None, None]
        z_hat = z_hat + z_q_k * active
        off += l_k
    return z_hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/home/usergpu/plan2text/data/pairs_train.jsonl")
    ap.add_argument("--val-pairs", default="/home/usergpu/plan2text/data/pairs_valid.jsonl")
    ap.add_argument("--e4f-ckpt", default="/home/usergpu/merge_lf/out/e4_fluency/last.pt")
    ap.add_argument("--out", default="/home/usergpu/merge_lf/out/cdec")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    main_proc = rank == 0
    if ws > 1:
        lr_ = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(lr_)
        dist.init_process_group("nccl")
        dev = torch.device("cuda", lr_)
    else:
        dev = torch.device("cuda:0")

    CFG, tk, net, src, _ = S.load_models(dev, 64, ckpt=a.e4f_ckpt)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    eos_id = tk.pre_tokenizer.eos_token_id
    pad_id = tk.pre_tokenizer.pad_token_id

    cdec = CausalVQDecoder(tk.vqvae, n_layer=4, n_head=6, dropout=0.1).to(dev)
    if main_proc:
        print("CausalVQDecoder params (M):", sum(p.numel() for p in cdec.parameters()) / 1e6, flush=True)

    vrows = S.read_pairs(a.val_pairs)[:512]
    trows = S.read_pairs(a.pairs)
    rows = vrows + trows
    nval = len(vrows)

    os.makedirs(a.out, exist_ok=True)
    idx_all = S.encode_stories(tk, [r["ids"] for r in rows], dev)
    tids_all = torch.tensor([r["ids"] for r in rows], dtype=torch.long)

    idx_val = idx_all[:nval].long().to(dev)
    tids_val = tids_all[:nval].to(dev)
    idx_tr = idx_all[nval:]
    tids_tr = tids_all[nval:]

    fwd_cdec = DDP(cdec, device_ids=[dev.index]) if ws > 1 else cdec
    opt = torch.optim.AdamW(cdec.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    g = torch.Generator().manual_seed(a.seed + rank)
    torch.manual_seed(a.seed + rank)

    def make_prev(tids):
        B = tids.shape[0]
        return torch.cat([torch.full((B, 1), eos_id, dtype=torch.long, device=tids.device), tids[:, :-1]], dim=1)

    cdec.train()
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * sched(step - 1)
        sel = torch.randint(0, len(idx_tr), (a.bs,), generator=g)
        b_idx = idx_tr[sel].long().to(dev)
        b_tids = tids_tr[sel].to(dev)
        b_prev = make_prev(b_tids)
        z_hat = build_noisy_zhat(tk.vqvae.vq, b_idx, g)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd_cdec(z_hat, b_prev)
            loss = F.cross_entropy(logits.float().reshape(-1, cdec.V), b_tids.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(cdec.parameters(), 1.0)
        opt.step()

        if main_proc and (step % 100 == 0 or step == 1 or step == a.steps):
            print({"step": step, "tok_ce_loss": round(loss.item(), 4), "gnorm": round(float(gn), 3), "s_per_step": round((time.perf_counter() - t0) / step, 3)}, flush=True)

    if main_proc:
        torch.save({"cdec": cdec.state_dict()}, os.path.join(a.out, "cdec.pt"))
        print("saved cdec.pt", flush=True)
    if ws > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
