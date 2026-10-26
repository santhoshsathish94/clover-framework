"""Cycle 45: CausalConceptDecoder (no position-t residual shortcut; cross-attention ONLY to NCM coarse-to-mid concept levels 0..5 + prefix).
Why: Cycle 44 proved that adding z_hat_11[:, t, :] into the residual stream at position t created a 1-to-1 token identity shortcut
(since level 11 has length 128 and 99.96% token accuracy on clean codes), causing the decoder to copy fine-level VQ collisions
('His name Billy', 'in the. Billy') while ignoring prev_ids.
Fix:
1. Only use NCM coarse-to-mid concept levels 0..5 (lengths 1, 8, 16, 24, 32, 40 -> downsampled to 40 concept slots [B, 40, 384]),
   combined with z_prefix ([B, P, 384], either BPE wte(plan) for Track 2B or LatentBridge(R(h)) for Track 2A) as cross-attention memory.
2. The causal decoder stream receives ONLY wte(prev_ids) + wpe(pos) (zero position-t additive z_hat shortcut!), so it MUST use
   prev_ids (token[0..t-1]) for English grammar while cross-attending to the NCM concept trajectory + prefix for plot and characters.
3. Also reduces NCM generator inference from 12 passes to 6 coarse-to-mid passes (levels 0..5, max length 40 instead of 128).
"""
import argparse, json, math, os, re, sys, time

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
ROOT = os.environ.get("NCM_REPO", "/home/usergpu/ncm/repo")
ART = os.environ.get("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation.gen_ppl import generation_perplexity

import ncm_stage2 as S
from eval_both_tracks import words, jacc, mbr, generate_from_zpref
from train_latent_bridge import LatentBridge, encode_batch_rh, P_LEN, D_HA
from train_causal_decoder import CausalBlock

N_CONCEPTS = 40
MAX_CONCEPT_LVL = 6  # levels 0..5: lengths 1, 8, 16, 24, 32, 40


@torch.no_grad()
def idx_to_concepts(vq, idx, max_lvl=MAX_CONCEPT_LVL, n_slots=N_CONCEPTS):
    """Accumulate NCM levels 0..max_lvl-1 and area-downsample to n_slots=40 concept vectors [B, 40, 384]."""
    levels = list(vq.levels)
    L_full = levels[-1]
    z_hat = torch.zeros(idx.shape[0], L_full, vq.embed_dim, device=idx.device)
    off = 0
    for k in range(min(max_lvl, len(levels))):
        l_k = levels[k]
        emb_k = vq.embeddings[k](idx[:, off:off + l_k])
        z_q_k = vq._up_to_full(emb_k, k, L_full) if k != len(levels) - 1 else emb_k
        z_q_k = vq._apply_quant_resi(z_q_k, k)
        z_hat = z_hat + z_q_k
        off += l_k
    z_c = F.interpolate(z_hat.transpose(-1, -2), size=(n_slots,), mode="area").transpose(-1, -2).contiguous()
    return z_c


class CausalConceptDecoder(nn.Module):
    """6-layer causal transformer decoder (d=384) with NO position-t residual shortcut; conditions strictly via cross-attention
    to [z_prefix, z_concepts_40]."""
    def __init__(self, vqvae, max_mem=104, n_layer=6, n_head=6, dropout=0.1):
        super().__init__()
        d = vqvae.embed_dim
        V = vqvae.in_vocab_size
        L = vqvae.context_length
        self.d = d
        self.V = V
        self.L = L
        self.wte = nn.Embedding(V, d)
        self.wpe = nn.Embedding(L, d)
        self.mem_ln = nn.LayerNorm(d)
        self.mem_pos = nn.Embedding(max_mem, d)
        self.mem_type = nn.Embedding(2, d)  # 0 = prefix slot, 1 = NCM concept slot
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([CausalBlock(d, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, V, bias=False)

        with torch.no_grad():
            self.wte.weight.copy_(vqvae.encoder.wte.weight)
            self.lm_head.weight.copy_(vqvae.decoder.output_layer.weight)
            nn.init.normal_(self.wpe.weight, std=0.02)
            nn.init.normal_(self.mem_pos.weight, std=0.02)
            nn.init.normal_(self.mem_type.weight, std=0.02)

    def build_mem(self, z_pref, z_concepts):
        B, P, _ = z_pref.shape
        K = z_concepts.shape[1]
        mem = torch.cat([z_pref, z_concepts], dim=1)
        mpos = torch.arange(P + K, device=mem.device)
        mtype = torch.cat([torch.zeros(P, dtype=torch.long, device=mem.device), torch.ones(K, dtype=torch.long, device=mem.device)])
        return self.mem_ln(mem) + self.mem_pos(mpos)[None, :, :] + self.mem_type(mtype)[None, :, :]

    def forward(self, z_pref, z_concepts, prev_ids):
        B, T = prev_ids.shape
        pos = torch.arange(T, device=prev_ids.device)
        z_mem = self.build_mem(z_pref, z_concepts)
        x = self.drop(self.wte(prev_ids) + self.wpe(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x, z_mem)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate_tokens(self, z_pref, z_concepts, eos_id, temp=0.5, top_k=10, adj=(3.0, 6, 0.7), ng=(3, 6.0)):
        B = z_pref.shape[0]
        dev = z_pref.device
        z_mem = self.build_mem(z_pref, z_concepts)
        prev = torch.full((B, 1), eos_id, dtype=torch.long, device=dev)
        out_ids = []
        hist = []
        toks = [[] for _ in range(B)]
        seen = [dict() for _ in range(B)]
        al, aw, ad = adj if adj else (0.0, 0, 1.0)

        for t in range(self.L):
            pos = torch.arange(prev.shape[1], device=dev)
            x = self.wte(prev) + self.wpe(pos)[None, :, :]
            for blk in self.blocks:
                x = blk(x, z_mem)
            logits = self.lm_head(self.ln_f(x[:, -1, :])).double()
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
                vals, inds = torch.topk(probs, k=top_k, dim=-1)
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


def cmd_train(a):
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

    CFG, tk, net, _, _ = S.load_models(dev, 64, ckpt=a.e4f_ckpt)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    eos_id = tk.pre_tokenizer.eos_token_id
    pad_id = tk.pre_tokenizer.pad_token_id

    ccdec = CausalConceptDecoder(tk.vqvae, max_mem=64 + N_CONCEPTS, n_layer=6, n_head=6, dropout=0.1).to(dev)
    if main_proc:
        print("CausalConceptDecoder params (M):", sum(p.numel() for p in ccdec.parameters()) / 1e6, flush=True)

    vrows = S.read_pairs(a.val_pairs)[:512]
    trows = S.read_pairs(a.pairs)
    rows = vrows + trows
    nval = len(vrows)

    os.makedirs(a.out, exist_ok=True)
    idx_all = S.encode_stories(tk, [r["ids"] for r in rows], dev)
    plans_all = S.encode_plans(tk, [r["plan"] for r in rows], 64, pad_id)
    tids_all = torch.tensor([r["ids"] for r in rows], dtype=torch.long)

    idx_val = idx_all[:nval].long().to(dev)
    plan_val = plans_all[:nval].to(dev)
    tids_val = tids_all[:nval].to(dev)
    idx_tr = idx_all[nval:]
    plan_tr = plans_all[nval:]
    tids_tr = tids_all[nval:]

    fwd_ccdec = DDP(ccdec, device_ids=[dev.index]) if ws > 1 else ccdec
    opt = torch.optim.AdamW(ccdec.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    g = torch.Generator().manual_seed(a.seed + rank)
    torch.manual_seed(a.seed + rank)

    def make_prev(tids):
        return torch.cat([torch.full((tids.shape[0], 1), eos_id, dtype=torch.long, device=tids.device), tids[:, :-1]], dim=1)

    @torch.no_grad()
    def eval_val():
        ccdec.eval()
        tot, n = 0.0, 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(idx_val), a.bs):
                b_idx = idx_val[i:i + a.bs]
                b_pl = plan_val[i:i + a.bs]
                b_tids = tids_val[i:i + a.bs]
                z_pref = tk.vqvae.encoder.wte(b_pl)
                z_c = idx_to_concepts(tk.vqvae.vq, b_idx)
                lg = ccdec(z_pref, z_c, make_prev(b_tids))
                ce = F.cross_entropy(lg.float().reshape(-1, ccdec.V), b_tids.reshape(-1))
                tot += float(ce.item()) * len(b_idx)
                n += len(b_idx)
        ccdec.train()
        return round(tot / n, 4)

    ccdec.train()
    t0 = time.perf_counter()
    levels = list(tk.vqvae.vq.levels)
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * sched(step - 1)
        sel = torch.randint(0, len(idx_tr), (a.bs,), generator=g)
        b_idx = idx_tr[sel].long().to(dev)
        b_pl = plan_tr[sel].to(dev)
        b_tids = tids_tr[sel].to(dev)
        b_prev = make_prev(b_tids)

        with torch.no_grad():
            # Corrupt coarse levels 0..5 by 15% so decoder generalizes to generator-predicted concept codes
            idx_c = S.corrupt(b_idx, levels, tk.vqvae.vq.vqvae_vocab_sizes, 0.10, 0.25)
            z_c = idx_to_concepts(tk.vqvae.vq, idx_c)
            z_pref = tk.vqvae.encoder.wte(b_pl)
            # Drop prefix 20% of the time so decoder relies strongly on NCM concept slots too
            drop_p = (torch.rand(len(b_pl), 1, 1, device=dev) < 0.20)
            z_pref = torch.where(drop_p, torch.zeros_like(z_pref), z_pref)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd_ccdec(z_pref, z_c, b_prev)
            loss = F.cross_entropy(logits.float().reshape(-1, ccdec.V), b_tids.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(ccdec.parameters(), 1.0)
        opt.step()

        if main_proc and (step % 200 == 0 or step == 1 or step == a.steps):
            v_ce = eval_val()
            print({"step": step, "train_ce": round(loss.item(), 4), "val_ce": v_ce, "gnorm": round(float(gn), 3), "s_per_step": round((time.perf_counter() - t0) / step, 3)}, flush=True)

    if main_proc:
        torch.save({"ccdec": ccdec.state_dict()}, os.path.join(a.out, "ccdec.pt"))
        print("saved ccdec.pt", flush=True)
    if ws > 1:
        dist.barrier()
        dist.destroy_process_group()


def cmd_eval(a):
    dev = torch.device("cuda:0")
    vrows = S.read_pairs(a.val_pairs)[:598]
    val_plans = [r["plan"] for r in vrows]

    from hybrid_agent.model import load_hybrid_agent
    ha_model, ha_tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev, require_hybridlink=True)
    ha_tok.padding_side = "right"
    rh_chunks, msk_chunks = [], []
    for i in range(0, len(val_plans), 128):
        r, m = encode_batch_rh(ha_model, ha_tok, val_plans[i:i + 128], P_LEN, dev)
        rh_chunks.append(r); msk_chunks.append(m)
    rh_val = torch.cat(rh_chunks, dim=0).to(dev)
    msk_val = (torch.cat(msk_chunks, dim=0).to(dev) != 0)
    del ha_model, ha_tok
    torch.cuda.empty_cache()

    gtok = AutoTokenizer.from_pretrained("gpt2-large")
    gm = AutoModelForCausalLM.from_pretrained("gpt2-large", dtype=torch.float16).to(dev).eval()

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
            if len(ids) < 2: continue
            c = {}
            for i in ids: c[i] = c.get(i, 0) + 1
            hs.append(-sum(v / len(ids) * math.log(v / len(ids)) for v in c.values()))
        return sum(hs) / len(hs)

    def score_all(texts, ref_plans):
        r = [S.retention(p, t) for p, t in zip(ref_plans, texts)]
        r = [x for x in r if x is not None]
        return {
            "retention": round(sum(r) / len(r), 4),
            "gen_ppl": round(gen_ppl(texts), 4),
            "adj_repeat_rate": round(rep_rate(texts), 5),
            "repeat_trigram_rate": round(rep3_rate(texts), 5),
            "entropy": round(entropy(texts), 4),
        }

    CFG, tk, net_e4f, _, _ = S.load_models(dev, 64, ckpt=a.e4f_ckpt)
    net_e4f.eval()
    pad_id, eos = tk.pre_tokenizer.pad_token_id, tk.pre_tokenizer.eos_token_id

    ccdec = CausalConceptDecoder(tk.vqvae, max_mem=64 + N_CONCEPTS, n_layer=6, n_head=6, dropout=0.1).to(dev)
    ccdec_sd = torch.load(os.path.join(a.out, "ccdec.pt"), map_location="cpu", weights_only=True)["ccdec"]
    ccdec.load_state_dict(ccdec_sd)
    ccdec.eval()

    def dec(r):
        r = list(r)
        if eos in r:
            r = r[:r.index(eos)]
        return tk.pre_tokenizer.decode([t for t in r if t not in (eos, pad_id)], skip_special_tokens=True)

    sched_10_04 = [1.0 - 0.6 * (k / 11.0) for k in range(12)]
    bs = 64

    @torch.no_grad()
    def gen_ccdec(net_m, z_pref, pmask, temp_dec=0.5, cands=8):
        # Pad z_pref to 64 if P=48 so positional embedding indices match training
        if z_pref.shape[1] < 64:
            z_pref_64 = F.pad(z_pref, (0, 0, 0, 64 - z_pref.shape[1]))
        else:
            z_pref_64 = z_pref
        all_c = []
        for k in range(cands):
            torch.manual_seed(42 + k)
            outs = []
            for i in range(0, len(z_pref), bs):
                g = generate_from_zpref(net_m, tk, z_pref[i:i + bs], pmask[i:i + bs], sched_10_04)
                z_c = idx_to_concepts(tk.vqvae.vq, g)
                t_ids = ccdec.generate_tokens(z_pref_64[i:i + bs], z_c, eos, temp=temp_dec, top_k=10, adj=(3.0, 6, 0.7), ng=(3, 6.0))
                outs += [dec(r) for r in t_ids.tolist()]
            all_c.append(outs)
        return mbr(all_c) if cands > 1 else all_c[0]

    pids_val = S.encode_plans(tk, val_plans, 64, pad_id).to(dev)
    z_pref_bpe = tk.vqvae.encoder.wte(pids_val)
    pmask_bpe = (pids_val == pad_id)

    res = {"val_598": {}, "e2e_48": {}}
    t_e4f_cc_t05 = gen_ccdec(net_e4f, z_pref_bpe, pmask_bpe, temp_dec=0.5, cands=8)
    res["val_598"]["e4_fluency_ccdec_t05"] = score_all(t_e4f_cc_t05, val_plans)
    print("val_598 e4_fluency_ccdec_t05:", res["val_598"]["e4_fluency_ccdec_t05"], flush=True)

    t_e4f_cc_t00 = gen_ccdec(net_e4f, z_pref_bpe, pmask_bpe, temp_dec=0.0, cands=4)
    res["val_598"]["e4_fluency_ccdec_t00"] = score_all(t_e4f_cc_t00, val_plans)
    print("val_598 e4_fluency_ccdec_t00:", res["val_598"]["e4_fluency_ccdec_t00"], flush=True)

    # Track 2A: e4_latent R(h) -> LatentBridge + CausalConceptDecoder
    _, _, net_e4l, src_e4l, _ = S.load_models(dev, P_LEN, ckpt="/home/usergpu/merge_lf/out/e4_latent/last.pt")
    net_e4l.eval()
    bridge = LatentBridge(D_HA, tk.vqvae.embed_dim).to(dev)
    bridge.load_state_dict(src_e4l["latent_bridge"])
    bridge.eval()
    z_pref_lat = bridge(rh_val).float()

    t_e4l_cc_t05 = gen_ccdec(net_e4l, z_pref_lat, msk_val, temp_dec=0.5, cands=8)
    res["val_598"]["e4_latent_ccdec_t05"] = score_all(t_e4l_cc_t05, val_plans)
    print("val_598 e4_latent_ccdec_t05:", res["val_598"]["e4_latent_ccdec_t05"], flush=True)

    # E2E 48 topics
    top_all = torch.load("/home/usergpu/merge_lf/top_shard_0.pt", map_location="cpu", weights_only=True)
    top_all.sort(key=lambda x: x["idx"])
    tp_48 = [x["text_plan"] for x in top_all]
    hp_48 = [x["hybrid_plan"] for x in top_all]
    pids_tp = S.encode_plans(tk, tp_48, 64, pad_id).to(dev)
    pids_hp = S.encode_plans(tk, hp_48, 64, pad_id).to(dev)
    rh_hp = torch.stack([x["rh_hybrid"] for x in top_all]).to(dev)
    msk_hp = (torch.stack([x["msk_hybrid"] for x in top_all]) != 0).to(dev)

    e4f_tp_cc = gen_ccdec(net_e4f, tk.vqvae.encoder.wte(pids_tp), (pids_tp == pad_id), temp_dec=0.5, cands=8)
    e4f_hp_cc = gen_ccdec(net_e4f, tk.vqvae.encoder.wte(pids_hp), (pids_hp == pad_id), temp_dec=0.5, cands=8)
    e4l_hp_cc = gen_ccdec(net_e4l, bridge(rh_hp).float(), msk_hp, temp_dec=0.5, cands=8)

    res["e2e_48"]["e4_fluency_ccdec_textplan"] = score_all(e4f_tp_cc, tp_48)
    res["e2e_48"]["e4_fluency_ccdec_hybridplan"] = score_all(e4f_hp_cc, hp_48)
    res["e2e_48"]["e4_latent_ccdec_hybridplan"] = score_all(e4l_hp_cc, hp_48)
    for k, v in res["e2e_48"].items():
        print("e2e_48", k, v, flush=True)

    sub = json.load(open("/home/usergpu/merge_lf/out/subset20_cdec.json", encoding="utf-8"))
    p_idxs = sub["plan_idxs"]
    sub["systems"]["e4_fluency_ccdec"] = [t_e4f_cc_t05[i] for i in p_idxs]
    sub["systems"]["e4_latent_ccdec"] = [t_e4l_cc_t05[i] for i in p_idxs]
    json.dump(sub, open("/home/usergpu/merge_lf/out/subset20_c45.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(res, open("/home/usergpu/merge_lf/out/eval_c45.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    for idx in range(4):
        pi = p_idxs[idx]
        print(f"=== PLAN {pi}: {val_plans[pi]} ===")
        print(f"  [e4_fluency_ccdec]: {t_e4f_cc_t05[pi][:350]}")
        print(f"  [e4_latent_ccdec ]: {t_e4l_cc_t05[pi][:350]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--pairs", default="/home/usergpu/plan2text/data/pairs_train.jsonl")
    ap.add_argument("--val-pairs", default="/home/usergpu/plan2text/data/pairs_valid.jsonl")
    ap.add_argument("--e4f-ckpt", default="/home/usergpu/merge_lf/out/e4_fluency/last.pt")
    ap.add_argument("--out", default="/home/usergpu/merge_lf/out/ccdec")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)
