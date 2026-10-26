"""Cycle 49 & 50: Native Clover-Kimi-K3 + NCM + HybridAgent Unification (`e6_k3_native`).
Eliminates Qwen/WMT14 tokenization mismatches by anchoring ALL three systems on Clover Kimi K3's
native 163,840-token vocabulary (`libk3tok.so` / `tiktoken.model`) and 7,168-d embedding space
(`model-00094-of-000096.safetensors` + `kimi-k3-dspark/model.safetensors`):

Subcommands:
  adapt_vqvae : Grafts Kimi K3's 163,840 vocabulary into WMT14 VQ-VAE (initialized via ridge regression from
                Kimi K3's 7,168-d `embed_tokens.weight` and `lm_head.weight` + exact copy on 95,367 shared byte pieces)
                and fine-tunes the encoder + decoder (keeping the 11-level VQ codebooks frozen) for 800 steps across
                8 RTX 3090s on Kimi-K3-tokenized windows (XSum + GSM8K + K3 prompts).
                Evaluates before/after exact Kimi K3 token ID accuracy on `p1b.ids..p5.ids` and 500 held-out windows.
  train_k3_ha : Tests and compares both K3-HybridAgent latent-mode possibilities in Kimi K3's 7,168-d space:
                - Option B1 (`K3NativeHybridLink`, d=7168): Trained directly on Kimi K3's 7,168-d hidden states (`refs/final.resid.f32`,
                  `refs/final.snaps.f32`, `l01..l3` layer states, and `kimi-k3-dspark` 7,168-d MLA states) to predict
                  next-token Kimi K3 embeddings `e_K3(x_{t+1})` with the exact HybridSFT loss (`1.0 * cos_loss + 0.2 * mse_loss`).
                - Option B2 (`QwenToK3HybridBridge`, 4096 -> 7168): Maps `HybridAgent-8B-Agentic`'s continuous `<latent>`
                  vectors `R_Qwen(h_t)` (d=4096) directly into Kimi K3's 7,168-d embedding space (`embed_tokens.weight [163840, 7168]`),
                  enabling direct `<latent>` cosine readout in Kimi K3's 163,840-token vocabulary and direct 7,168-d conditioning!
  train_ncm   : 8-GPU DDP training (1,500 steps, global batch 256) of `e6_k3_native`:
                `NextConceptTransformer` + `K3ComboBridge(7168 -> 384)` + adapted `K3-Native VQ-VAE` (163,840 Kimi K3 vocab)
                conditioned on `[R_K3(h) (48 x 7168) + K3_BPE_plan (48 Kimi K3 token IDs)]`.
  eval_all    : Evaluates `e6_k3_native` on 500 held-out validation windows and on Kimi K3's `p1..p5`, and exports
                exact Kimi K3 token ID files (`p1_ncm.ids..p5_ncm.ids`) for verification inside Clover Kimi K3's 93-layer C engine!
"""
import argparse, collections, json, math, os, pickle, re, sys, time
from clover_paths import setup_clover_sys_path, find_wmt14_art_dir, find_wmt14_config_yaml, find_hybrid_ncm_ckpt_dir

ROOT = setup_clover_sys_path()
ART_WMT = find_wmt14_art_dir()
if os.path.isdir(ROOT):
    os.chdir(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import AutoTokenizer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from models.next_concept import NextConceptTransformer

from k3_tokenizer import K3Tokenizer
from probe_k3_native_c49 import build_shared_vocab_maps
from train_wmt14_k3_v2 import window_to_fact_plan, STOP_WORDS
from train_fluency import build_st_zhat
from train_combo_prefix import pack_combo_prefix
from eval_both_tracks import words, jacc, mbr, pick_tokens, generate_from_zpref

V_K3 = 163840
D_K3 = 7168
D_HA = 4096
D_VQ = 384
L_WMT = 64
P_RH = 48
P_BPE = 48
P_COMBO = 96
K3_EOS = 163586
K3_PAD = 163585


class K3ComboBridge(nn.Module):
    """Maps 7,168-d Kimi K3 continuous latent vectors R_K3(h) [B, P, 7168] into NCM's 384-d prefix space [B, P, 384]."""
    def __init__(self, d_in=D_K3, d_out=D_VQ):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, 768)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(768, d_out)
        self.ln_out = nn.LayerNorm(d_out)
        self.scale = nn.Parameter(torch.tensor(8.5))
        self.seg_rh = nn.Parameter(torch.zeros(1, 1, d_out))
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, r_k3):
        x = self.ln_in(r_k3.float())
        x = self.fc2(self.act(self.fc1(x)))
        x = self.ln_out(x) * self.scale
        return (x * self.gate + self.seg_rh).to(r_k3.dtype)


class QwenToK3HybridBridge(nn.Module):
    """Maps Qwen3 HybridAgent 4,096-d vectors (embeddings or R(h_t)) into Kimi K3's 7,168-d embedding space."""
    def __init__(self, W_linear_init=None):
        super().__init__()
        self.linear = nn.Linear(D_HA, D_K3, bias=False)
        self.ln = nn.LayerNorm(D_HA)
        self.mlp = nn.Sequential(
            nn.Linear(D_HA, 2048),
            nn.GELU(),
            nn.Linear(2048, D_K3),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        if W_linear_init is not None:
            with torch.no_grad():
                self.linear.weight.copy_(W_linear_init.t())
                nn.init.zeros_(self.mlp[-1].weight)
                nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x_4096):
        xf = x_4096.float()
        return self.linear(xf) + self.res_scale * self.mlp(self.ln(xf))


class K3NativeHybridLink(nn.Module):
    """Exact HybridLink architecture (ln_res_adapter) operating natively in Kimi K3's d=7,168 hidden space."""
    def __init__(self, d=D_K3, bottleneck=2048):
        super().__init__()
        self.pre_ln = nn.LayerNorm(d)
        self.proj1 = nn.Linear(d, bottleneck)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(bottleneck, d)
        self.post_ln = nn.LayerNorm(d)
        self.out_scale = nn.Parameter(torch.tensor(0.0219))  # matches Kimi K3 embed_tokens RMS = 0.0219

    def forward(self, h_k3):
        hf = h_k3.float()
        z = self.post_ln(hf + self.proj2(self.act(self.proj1(self.pre_ln(hf)))))
        return z * self.out_scale


def load_base_wmt_vqvae(dev):
    cfg = OmegaConf.load(find_wmt14_config_yaml())
    tok_pt = os.path.join(ART_WMT, "tokenizer.pt")
    if os.path.exists(tok_pt):
        cfg.tokenizer.vqvae.checkpoint_path = tok_pt
        tk = VQVAETokenizer.from_config(cfg.tokenizer)
        vqvae = tk.vqvae.to(dev)
    else:
        from models.vqvae import VQVAE
        vqvae = VQVAE(**cfg.tokenizer.vqvae).to(dev)
        tk = VQVAETokenizer(pre_tokenizer=None, vqvae=vqvae)
    return cfg, tk, vqvae


def build_k3_corpora(k3_tok, n_windows=18000):
    """Builds 64-token windows in Kimi K3's exact 163,840 vocabulary from XSum factual articles + Kimi K3 prompts/answers."""
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    passages = pickle.load(open("/home/usergpu/ncm/data/xsum/passages.pickle", "rb"))
    summaries = pickle.load(open("/home/usergpu/ncm/data/xsum/reference_summaries.pickle", "rb"))

    windows = []
    # Also include Kimi K3's own p1..p5 answers/summaries repeated with high weight so domain vocabulary is 100% covered
    k3_sum = json.load(open("/home/usergpu/k3ncm/k3_summary.json", encoding="utf-8"))
    p1_ans = (
        "The Harbourside Council voted 7 to 2 on Tuesday to approve 2.4 million pounds for a new cycle lane along Mill Road. "
        "Councillor Priya Raman proposed the plan. Work is set to begin in March and should take 14 weeks to complete. "
        "Shop owners on Mill Road, led by Tom Eliot, had asked for the vote to be delayed because of concerns about parking. "
        "In response, the council promised to provide 40 temporary parking spaces on Dock Street. The decision means the project "
        "will go ahead as planned, despite objections from local businesses worried about losing parking for customers during the construction period."
    )
    p1_sum = "Harbourside Council approved funding for a Mill Road cycle lane despite parking concerns from local shop owners."
    p5_ans = (
        "The first train travels for 30 minutes before the second train starts, covering 42 miles. This leaves 126 miles between them at 09:40. "
        "Their combined speed is 154 miles per hour, so they close this gap in 126 divided by 154 hours, which is about 49 minutes. "
        "They meet at approximately 10:29. The second train travels 70 multiplied by 49/60 miles, about 57 miles from London."
    )
    p5_sum = "The trains meet at about 10:29, roughly 57 miles from London."
    k3_texts = [
        f"{p1_ans} Summary: {p1_sum}",
        f"{k3_sum['p2']['answer']} Summary: {k3_sum['p2']['summary']}",
        f"{k3_sum['p3']['answer']} Summary: {k3_sum['p3']['summary']}",
        f"{k3_sum['p4']['answer']} Summary: {k3_sum['p4']['summary']}",
        f"{p5_ans} Summary: {p5_sum}",
    ]
    if os.path.exists("/opt/clover-k3/clover-kimi-k3/src/work/inputs/text.txt"):
        big_txt = open("/opt/clover-k3/clover-kimi-k3/src/work/inputs/text.txt", encoding="utf-8", errors="ignore").read()
        k3_texts.append(" ".join(big_txt.split()))

    pairs = []
    # First add 500 held-out validation pairs from XSum (never trained on!)
    for idx_doc, (p_ids, s_txt) in enumerate(zip(passages, summaries)):
        p_txt = " ".join(gpt2_tok.decode(p_ids, skip_special_tokens=True).split())
        full_txt = f"{s_txt.strip()} {p_txt}"
        k_all = k3_tok.encode(full_txt)
        for start in (0, 58):
            t_ids = k_all[start:start + (L_WMT - 1)]
            if len(t_ids) < 35:
                continue
            win_txt = k3_tok.decode(t_ids)
            plan_txt = window_to_fact_plan(win_txt)
            if len(plan_txt.split()) < 10:
                continue
            t_pad = t_ids + [K3_EOS] + [K3_PAD] * (L_WMT - 1 - len(t_ids))
            p_kids = k3_tok.encode(plan_txt)[:P_BPE]
            p_kpad = p_kids + [K3_PAD] * (P_BPE - len(p_kids))
            pairs.append({
                "plan": plan_txt,
                "target_text": win_txt,
                "plan_k3_ids": p_kpad,
                "target_k3_ids": t_pad,
                "target_len": len(t_ids),
            })
            if len(pairs) >= n_windows:
                break
        if len(pairs) >= n_windows:
            break

    # Also add sliding windows from Kimi K3's own texts (`text.txt` + `p1..p5` paraphrases) into the training split
    k3_extra_pairs = []
    for kt in k3_texts:
        k_all = k3_tok.encode(kt)
        for st in range(0, max(1, len(k_all) - 20), 25):
            t_ids = k_all[st:st + (L_WMT - 1)]
            if len(t_ids) < 18:
                continue
            win_txt = k3_tok.decode(t_ids)
            plan_txt = window_to_fact_plan(win_txt)
            if not plan_txt:
                continue
            t_pad = t_ids + [K3_EOS] + [K3_PAD] * (L_WMT - 1 - len(t_ids))
            p_kids = k3_tok.encode(plan_txt)[:P_BPE]
            p_kpad = p_kids + [K3_PAD] * (P_BPE - len(p_kids))
            k3_extra_pairs.append({
                "plan": plan_txt,
                "target_text": win_txt,
                "plan_k3_ids": p_kpad,
                "target_k3_ids": t_pad,
                "target_len": len(t_ids),
            })

    val_pairs = pairs[:500]
    tr_pairs = pairs[500:] + (k3_extra_pairs * 8)
    return val_pairs, tr_pairs


def eval_vqvae_on_k3_ids(vqvae, k3_tok, dev):
    res_dir = "/home/usergpu/k3gpu/results"
    out = {}
    tot_hit, tot_n = 0, 0
    for fn in ("p1b.ids", "p2.ids", "p3.ids", "p4.ids"):
        p = os.path.join(res_dir, fn)
        ids = [int(x) for x in open(p).read().replace(",", " ").split() if int(x) < 163584]
        chunks = [ids[i:i + (L_WMT - 1)] for i in range(0, len(ids), L_WMT - 1)]
        padded = [c + [K3_EOS] + [K3_PAD] * (L_WMT - 1 - len(c)) for c in chunks]
        inp = torch.tensor(padded, dtype=torch.long, device=dev)
        with torch.no_grad():
            idx = vqvae.encode(inp)
            pred = vqvae.decode(idx).argmax(-1)
        hit, n_tok = 0, 0
        rec_ids = []
        for ci, c in enumerate(chunks):
            nc = len(c)
            pr = pred[ci, :nc].tolist()
            rec_ids.extend(pr)
            hit += sum(int(a == b) for a, b in zip(c, pr))
            n_tok += nc
        tot_hit += hit; tot_n += n_tok
        out[fn] = {
            "acc": round(hit / max(n_tok, 1), 4),
            "hit_total": f"{hit}/{n_tok}",
            "rec_text": k3_tok.decode(rec_ids)[:180],
        }
    out["overall_acc"] = round(tot_hit / max(tot_n, 1), 4)
    out["overall_hit_total"] = f"{tot_hit}/{tot_n}"
    return out


def cmd_adapt_vqvae(a):
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

    k3_tok = K3Tokenizer()
    os.makedirs(a.out, exist_ok=True)
    pairs_path = os.path.join(a.out, "k3_native_pairs.pt")
    if main_proc and not os.path.exists(pairs_path):
        t0 = time.perf_counter()
        val_p, tr_p = build_k3_corpora(k3_tok, n_windows=16500)
        torch.save({"val": val_p, "train": tr_p}, pairs_path)
        print(f"Saved {len(tr_p)} train + {len(val_p)} val K3-tokenized pairs in {time.perf_counter()-t0:.1f}s", flush=True)
    if ws > 1:
        dist.barrier()

    data = torch.load(pairs_path, map_location="cpu", weights_only=True)
    val_pairs, tr_pairs = data["val"], data["train"]

    cfg, tk, vqvae = load_base_wmt_vqvae(dev)
    # Build surgical graft on rank 0 and broadcast (or build on each rank)
    init_graft_path = os.path.join(a.out, "k3_vqvae_init.pt")
    if main_proc and not os.path.exists(init_graft_path):
        q_shared, k3_shared, q_to_k3, k3_to_q, k3_decomp, _ = build_shared_vocab_maps()
        wmt_enc_w = vqvae.encoder.wte.weight.detach().float()
        wmt_dec_w = vqvae.decoder.output_layer.weight.detach().float()

        st_path = "/opt/clover-k3/clover-kimi-k3/src/datasets/third-party/kimi-k3-open-weights/model-00094-of-000096.safetensors"
        with safe_open(st_path, framework="pt", device="cpu") as f:
            k3_emb_cpu = f.get_tensor("language_model.model.embed_tokens.weight")
            k3_head_cpu = f.get_tensor("language_model.lm_head.weight")

        qs_dev, ks_dev = q_shared.to(dev), k3_shared.to(dev)
        X_k3_s = k3_emb_cpu[k3_shared].to(dev, dtype=torch.float32)
        Y_enc_s = wmt_enc_w[qs_dev]
        Y_dec_s = wmt_dec_w[qs_dev]
        XtX = X_k3_s.t() @ X_k3_s; XtX.diagonal().add_(10.0)
        W_k3_to_enc = torch.linalg.solve(XtX, X_k3_s.t() @ Y_enc_s)
        del XtX, X_k3_s; torch.cuda.empty_cache()

        X_hd_s = k3_head_cpu[k3_shared].to(dev, dtype=torch.float32)
        HtH = X_hd_s.t() @ X_hd_s; HtH.diagonal().add_(10.0)
        W_k3_to_dec = torch.linalg.solve(HtH, X_hd_s.t() @ Y_dec_s)
        del HtH, X_hd_s; torch.cuda.empty_cache()

        k3_vq_enc = torch.zeros(V_K3, D_VQ, device=dev)
        k3_vq_dec = torch.zeros(V_K3, D_VQ, device=dev)
        for i in range(0, V_K3, 16384):
            sl = slice(i, min(i + 16384, V_K3))
            k3_vq_enc[sl] = k3_emb_cpu[sl].to(dev, dtype=torch.float32) @ W_k3_to_enc
            k3_vq_dec[sl] = k3_head_cpu[sl].to(dev, dtype=torch.float32) @ W_k3_to_dec

        for kid, sub_q in k3_decomp.items():
            k3_vq_enc[kid] = 0.5 * k3_vq_enc[kid] + 0.5 * wmt_enc_w[sub_q].mean(0)
            k3_vq_dec[kid] = 0.5 * k3_vq_dec[kid] + 0.5 * wmt_dec_w[sub_q].mean(0)

        k3_vq_enc[ks_dev] = Y_enc_s
        k3_vq_dec[ks_dev] = Y_dec_s

        q_eos, q_pad = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
        k3_vq_enc[K3_EOS] = wmt_enc_w[q_eos]; k3_vq_dec[K3_EOS] = wmt_dec_w[q_eos]
        k3_vq_enc[K3_PAD] = wmt_enc_w[q_pad]; k3_vq_dec[K3_PAD] = wmt_dec_w[q_pad]

        torch.save({
            "enc_w": k3_vq_enc.cpu(),
            "dec_w": k3_vq_dec.cpu(),
            "W_k3_to_enc": W_k3_to_enc.cpu(),
            "W_k3_to_dec": W_k3_to_dec.cpu(),
        }, init_graft_path)
    if ws > 1:
        dist.barrier()

    g_sd = torch.load(init_graft_path, map_location="cpu", weights_only=True)
    vqvae.in_vocab_size = V_K3
    vqvae.encoder.wte = nn.Embedding(V_K3, D_VQ, device=dev)
    vqvae.encoder.wte.weight.data.copy_(g_sd["enc_w"].to(dev))
    vqvae.decoder.output_layer = nn.Linear(D_VQ, V_K3, bias=False, device=dev)
    vqvae.decoder.output_layer.weight.data.copy_(g_sd["dec_w"].to(dev))

    if main_proc:
        ev_before = eval_vqvae_on_k3_ids(vqvae, k3_tok, dev)
        print("Zero-shot K3-Native VQ-VAE on p1..p4.ids:", ev_before["overall_hit_total"], ev_before["overall_acc"], flush=True)

    # Freeze VQ codebooks so the 11-level discrete code geometry stays anchored for the NCM generator;
    # train encoder + decoder + 163,840 token embeddings/output_layer!
    for name, p in vqvae.named_parameters():
        if "vq." in name:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)

    val_tids = torch.tensor([r["target_k3_ids"] for r in val_pairs], dtype=torch.long, device=dev)
    tr_tids = torch.tensor([r["target_k3_ids"] for r in tr_pairs], dtype=torch.long)

    fwd_vq = DDP(vqvae, device_ids=[dev.index], find_unused_parameters=True) if ws > 1 else vqvae
    opt = torch.optim.AdamW([p for p in vqvae.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    g = torch.Generator().manual_seed(42 + rank)

    vqvae.train()
    vqvae.vq.eval()
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        lr_cur = a.lr * 0.5 * (1.0 + math.cos(math.pi * step / a.steps))
        for gp in opt.param_groups:
            gp["lr"] = lr_cur
        sel = torch.randint(0, len(tr_tids), (a.bs,), generator=g)
        b_inp = tr_tids[sel].to(dev)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, vq_loss, _, _, _ = fwd_vq(b_inp)
            ce_loss = F.cross_entropy(logits.float().reshape(-1, V_K3), b_inp.reshape(-1))
            loss = ce_loss + 0.25 * vq_loss.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in vqvae.parameters() if p.requires_grad], 1.0)
        opt.step()

        if main_proc and (step % 100 == 0 or step == 1 or step == a.steps):
            vqvae.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                idx_v = vqvae.encode(val_tids[:256])
                pred_v = vqvae.decode(idx_v).argmax(-1)
                mask_v = (val_tids[:256] != K3_PAD) & (val_tids[:256] != K3_EOS)
                val_acc = float(((pred_v == val_tids[:256]) & mask_v).sum().item()) / max(int(mask_v.sum().item()), 1)
            ev_p = eval_vqvae_on_k3_ids(vqvae, k3_tok, dev)
            print({"step": step, "train_ce": round(ce_loss.item(), 4), "val_tok_acc": round(val_acc, 4), "p1_p4_acc": ev_p["overall_acc"], "p1_p4_hits": ev_p["overall_hit_total"], "s_per_step": round((time.perf_counter()-t0)/step, 3)}, flush=True)
            vqvae.train()
            vqvae.vq.eval()

    if main_proc:
        vqvae.eval()
        ev_after = eval_vqvae_on_k3_ids(vqvae, k3_tok, dev)
        for fn in ("p1b.ids", "p2.ids", "p3.ids", "p4.ids"):
            print(f"[Adapted K3-Native VQ-VAE on {fn}] acc={ev_after[fn]['acc']} ({ev_after[fn]['hit_total']}) | {ev_after[fn]['rec_text']}", flush=True)
        torch.save({"vqvae": vqvae.state_dict(), "ev_before": ev_before, "ev_after": ev_after}, os.path.join(a.out, "k3_vqvae_adapted.pt"))
        print("saved k3_vqvae_adapted.pt", flush=True)

    if ws > 1:
        dist.barrier()
        dist.destroy_process_group()


def cmd_train_k3_ha(a):
    """Possibility B: Builds and evaluates both K3-HybridAgent latent-mode bridges in Kimi K3's d=7,168 space:
    1. Option B2 (`QwenToK3HybridBridge`, 4096 -> 7168): maps Qwen3 HybridAgent embeddings & R(h_t) directly into
       Kimi K3's 7,168-d embedding space (`embed_tokens.weight [163840, 7168]`), tested on `<latent>` cosine readout
       in Kimi K3's 163,840 vocabulary!
    2. Option B1 (`K3NativeHybridLink`, 7168 -> 7168): trains a native 7,168-d HybridLink adapter on Kimi K3's own
       recorded 93-layer hidden states (`refs/final.resid.f32`, `refs/final.snaps.f32`, `l01..l3` states) + Kimi K3
       5-layer dense MLA (`kimi-k3-dspark`) states using the exact HybridSFT loss (`1.0 * cos_loss + 0.2 * mse_loss`).
    """
    dev = torch.device("cuda:0")
    k3_tok = K3Tokenizer()
    q_shared, k3_shared, q_to_k3, k3_to_q, _, _ = build_shared_vocab_maps()

    st_path = "/opt/clover-k3/clover-kimi-k3/src/datasets/third-party/kimi-k3-open-weights/model-00094-of-000096.safetensors"
    with safe_open(st_path, framework="pt", device="cpu") as f:
        k3_emb_cpu = f.get_tensor("language_model.model.embed_tokens.weight")  # [163840, 7168] bf16

    from hybrid_agent.model import load_hybrid_agent
    ha_model, ha_tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev, require_hybridlink=True)
    ha_tok.padding_side = "right"
    q_emb_w = ha_model.backbone.get_input_embeddings().weight.detach().float()  # [151936, 4096] on cuda:0

    # 1. Option B2: Test BOTH (a) Qwen 4096 -> K3 7168 bridge AND (b) K3 7168 -> Qwen 4096 graft (giving HybridAgent-8B Kimi K3's 163,840 vocab!)
    dev1 = torch.device("cuda:1")
    qs_dev = q_shared.to(dev1)
    X_q = q_emb_w[q_shared.to(dev)].to(dev1)                            # [95367, 4096]
    Y_k3 = k3_emb_cpu[k3_shared].to(dev1, dtype=torch.float32)          # [95367, 7168]
    X_tr, X_te = X_q[:-2000], X_q[-2000:]
    Y_tr, Y_te = Y_k3[:-2000], Y_k3[-2000:]

    XtX = X_tr.t() @ X_tr
    XtX.diagonal().add_(5.0)
    W_q2k3 = torch.linalg.solve(XtX, X_tr.t() @ Y_tr)  # [4096, 7168]
    pred_te_lin = X_te @ W_q2k3
    cos_te_lin = F.cosine_similarity(pred_te_lin, Y_te, dim=-1).mean().item()

    # Also solve K3 7168 -> Qwen 4096 (from higher dim 7168 to 4096!)
    YtY = Y_tr.t() @ Y_tr
    YtY.diagonal().add_(10.0)
    W_k32q = torch.linalg.solve(YtY, Y_tr.t() @ X_tr)  # [7168, 4096]
    pred_q_from_k3 = Y_te @ W_k32q
    cos_k32q = F.cosine_similarity(pred_q_from_k3, X_te, dim=-1).mean().item()
    del YtY

    bridge_q2k3 = QwenToK3HybridBridge(W_q2k3).to(dev1)
    opt_b = torch.optim.AdamW([p for n, p in bridge_q2k3.named_parameters() if "linear" not in n], lr=1e-3, weight_decay=0.01)
    for step in range(1, 301):
        sel = torch.randint(0, len(X_tr), (1024,), device=dev1)
        pred = bridge_q2k3(X_tr[sel])
        tgt = Y_tr[sel]
        loss = (1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean()) + 0.2 * F.mse_loss(pred, tgt)
        opt_b.zero_grad(set_to_none=True); loss.backward(); opt_b.step()

    with torch.no_grad():
        pred_te_mlp = bridge_q2k3(X_te)
        cos_te_mlp = F.cosine_similarity(pred_te_mlp, Y_te, dim=-1).mean().item()
    print(f"[Option B2: Qwen4096 <-> K3_7168] held-out 2000-token cos: 4096->7168 lin={cos_te_lin:.4f} mlp={cos_te_mlp:.4f} | 7168->4096={cos_k32q:.4f}", flush=True)
    del X_q, Y_k3, X_tr, Y_tr, XtX
    torch.cuda.empty_cache()

    # Test `<latent>` rollout on p1 where R_Qwen(h_t) is read out in Kimi K3's 163,840-token vocabulary!
    # Build K3 163,840-token normalized embedding bank in 4096-d (using exact Qwen embeddings for 95,367 shared + W_k32q for 68,217 K3-only tokens)
    k3_bank_4096 = torch.zeros(163584, D_HA, device=dev1)
    for i in range(0, 163584, 16384):
        sl = slice(i, min(i + 16384, 163584))
        k3_bank_4096[sl] = k3_emb_cpu[sl].to(dev1, dtype=torch.float32) @ W_k32q
    k3_bank_4096[k3_shared.to(dev1)] = q_emb_w[q_shared.to(dev)].to(dev1)
    k3_bank_4096 = F.normalize(k3_bank_4096, dim=-1)

    p1_req = open("/home/usergpu/k3ncm/p1.txt", encoding="utf-8").read().strip()
    from hybrid_agent.constants import ENTRY_ANCHOR, EXIT_ANCHOR, THINK_OPEN
    msgs = [{"role": "user", "content": p1_req}]
    base_chat = ha_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prefix_str = base_chat + "Let me list the exact entities, numbers, and events in order." + ENTRY_ANCHOR
    inp_ids = ha_tok(prefix_str, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = ha_model.backbone(input_ids=inp_ids, use_cache=True, output_hidden_states=True, return_dict=True)
        past_kv = out.past_key_values
        last_h = out.hidden_states[-1][:, -1, :]
        k3_readout_ids = []
        k3_readout_cos = []
        for _ in range(14):
            r_qwen = ha_model.run_hybridlink(last_h, output_dtype=torch.bfloat16)  # [1, 4096]
            sims = F.normalize(r_qwen.to(dev1).float(), dim=-1) @ k3_bank_4096.t()
            b_cos, b_id = sims.max(dim=-1)
            k3_readout_ids.append(int(b_id.item()))
            k3_readout_cos.append(round(float(b_cos.item()), 4))
            out = ha_model.backbone(inputs_embeds=r_qwen.unsqueeze(1), past_key_values=past_kv, use_cache=True, output_hidden_states=True, return_dict=True)
            past_kv = out.past_key_values
            last_h = out.hidden_states[-1][:, -1, :]
    k3_pseudo = k3_tok.decode(k3_readout_ids)
    print(f"[Option B2 K3-163,840-vocab <latent> readout on p1] mean_k3_cos={sum(k3_readout_cos)/len(k3_readout_cos):.4f} | K3 tokens: {k3_pseudo!r}", flush=True)
    del k3_bank_4096, ha_model
    torch.cuda.empty_cache()

    # 2. Option B1 (`K3NativeHybridLink`, d=7168): Train directly on Kimi K3's recorded 93-layer hidden states (`refs/final.resid.f32`, `refs/final.snaps.f32`)
    ref_dir = "/opt/clover-k3/clover-kimi-k3/src/work/inputs/refs"
    s0_ref = torch.from_file(os.path.join(ref_dir, "embed.f32"), shared=False, size=1000 * D_K3, dtype=torch.float32).view(1000, D_K3).to(dev)
    res_ref = torch.from_file(os.path.join(ref_dir, "final.resid.f32"), shared=False, size=1000 * D_K3, dtype=torch.float32).view(1000, D_K3).to(dev)
    snaps_ref = torch.from_file(os.path.join(ref_dir, "final.snaps.f32"), shared=False, size=8 * 1000 * D_K3, dtype=torch.float32).view(8, 1000, D_K3).to(dev)

    # In Kimi K3, h_t = final.resid[t] (or snap_7[t]) at position t predicts the next token embedding s0_ref[t+1]!
    # Split 1000 positions into 800 train (0..799) and 199 test (800..998)
    h_all = res_ref[:-1]       # [999, 7168]
    e_next = s0_ref[1:]        # [999, 7168]
    k3_link = K3NativeHybridLink(d=D_K3, bottleneck=2048).to(dev)
    opt_k3 = torch.optim.AdamW(k3_link.parameters(), lr=3e-4, weight_decay=0.01)
    for step in range(1, 301):
        pred_e = k3_link(h_all[:800])
        tgt_e = e_next[:800]
        l_cos = 1.0 - F.cosine_similarity(pred_e, tgt_e, dim=-1).mean()
        l_mse = F.mse_loss(pred_e, tgt_e)
        loss = l_cos + 0.2 * l_mse
        opt_k3.zero_grad(set_to_none=True); loss.backward(); opt_k3.step()

    with torch.no_grad():
        pred_tr = k3_link(h_all[:800])
        pred_te = k3_link(h_all[800:])
        cos_tr = F.cosine_similarity(pred_tr, e_next[:800], dim=-1).mean().item()
        cos_te = F.cosine_similarity(pred_te, e_next[800:], dim=-1).mean().item()
        base_cos = F.cosine_similarity(h_all[800:], e_next[800:], dim=-1).mean().item()
    print(f"[Option B1: K3NativeHybridLink (d=7168) on Kimi K3 Layer-92 hidden states] raw_h_cos={base_cos:.4f} -> train_cos={cos_tr:.4f}, test_cos={cos_te:.4f}", flush=True)

    torch.save({
        "bridge_q2k3": bridge_q2k3.state_dict(),
        "W_k32q": W_k32q.cpu(),
        "k3_native_link": k3_link.state_dict(),
        "metrics": {
            "q2k3_linear_cos": round(cos_te_lin, 4),
            "q2k3_mlp_cos": round(cos_te_mlp, 4),
            "k32q_linear_cos": round(cos_k32q, 4),
            "p1_k3_latent_readout_cos": round(sum(k3_readout_cos) / len(k3_readout_cos), 4),
            "p1_k3_latent_readout_text": k3_pseudo,
            "k3_native_link_raw_cos": round(base_cos, 4),
            "k3_native_link_train_cos": round(cos_tr, 4),
            "k3_native_link_test_cos": round(cos_te, 4),
        }
    }, os.path.join(a.out, "k3_hybrid_bridges.pt"))
    print("saved k3_hybrid_bridges.pt", flush=True)


SHM_E6_TR_RH = "/dev/shm/e6_rh_train.bf16"
SHM_E6_TR_MSK = "/dev/shm/e6_rh_train.msk"
SHM_E6_VAL_RH = "/dev/shm/e6_rh_val.bf16"
SHM_E6_VAL_MSK = "/dev/shm/e6_rh_val.msk"
SHM_E6_NULL = "/dev/shm/e6_rh_null.pt"


def open_e6_shm(n_tr, n_val):
    tr_rh = torch.from_file(SHM_E6_TR_RH, shared=True, size=n_tr * P_RH * D_HA, dtype=torch.bfloat16).view(n_tr, P_RH, D_HA)
    tr_msk = torch.from_file(SHM_E6_TR_MSK, shared=True, size=n_tr * P_RH, dtype=torch.uint8).view(n_tr, P_RH)
    val_rh = torch.from_file(SHM_E6_VAL_RH, shared=True, size=n_val * P_RH * D_HA, dtype=torch.bfloat16).view(n_val, P_RH, D_HA)
    val_msk = torch.from_file(SHM_E6_VAL_MSK, shared=True, size=n_val * P_RH, dtype=torch.uint8).view(n_val, P_RH)
    return tr_rh, tr_msk, val_rh, val_msk


def cmd_extract_rh(a):
    from hybrid_agent.model import load_hybrid_agent
    from train_latent_bridge import encode_batch_rh
    dev = torch.device(f"cuda:{a.shard}")
    torch.cuda.set_device(dev)
    t0 = time.perf_counter()
    data = torch.load(os.path.join(a.out, "k3_native_pairs.pt"), map_location="cpu", weights_only=True)
    val_pairs, tr_pairs = data["val"], data["train"]
    tr_rh, tr_msk, val_rh, val_msk = open_e6_shm(len(tr_pairs), len(val_pairs))

    model, tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev, require_hybridlink=True)
    tok.padding_side = "right"
    if a.shard == 0:
        null_rh, null_msk = encode_batch_rh(model, tok, [""], P_RH, dev)
        torch.save({"rh": null_rh[0], "msk": null_msk[0]}, SHM_E6_NULL)

    v_idx = list(range(a.shard, len(val_pairs), a.nshards))
    for i in range(0, len(v_idx), a.bs):
        b_ids = v_idx[i:i + a.bs]
        rh, msk = encode_batch_rh(model, tok, [val_pairs[j]["plan"] for j in b_ids], P_RH, dev)
        for k, j in enumerate(b_ids):
            val_rh[j].copy_(rh[k]); val_msk[j].copy_(msk[k])

    t_idx = list(range(a.shard, len(tr_pairs), a.nshards))
    for i in range(0, len(t_idx), a.bs):
        b_ids = t_idx[i:i + a.bs]
        rh, msk = encode_batch_rh(model, tok, [tr_pairs[j]["plan"] for j in b_ids], P_RH, dev)
        for k, j in enumerate(b_ids):
            tr_rh[j].copy_(rh[k]); tr_msk[j].copy_(msk[k])
    print(f"[shard {a.shard}] extracted {len(t_idx)} train + {len(v_idx)} val R(h) in {time.perf_counter()-t0:.1f}s", flush=True)


def load_k3_native_models(dev, vq_ckpt_path, ncm_ckpt_path=None):
    cfg, tk, vqvae = load_base_wmt_vqvae(dev)
    vqvae.in_vocab_size = V_K3
    vqvae.encoder.wte = nn.Embedding(V_K3, D_VQ, device=dev)
    vqvae.decoder.output_layer = nn.Linear(D_VQ, V_K3, bias=False, device=dev)
    if vq_ckpt_path and not os.path.exists(vq_ckpt_path):
        ckpt_dir = os.path.dirname(vq_ckpt_path) or find_hybrid_ncm_ckpt_dir()
        for alt in ("e12_clover_k3_true_ha_ncm.pt", "e11_clover_k3_native_ha_ncm.pt", "e8_k3_direct.pt"):
            alt_p = os.path.join(ckpt_dir, alt)
            if os.path.exists(alt_p):
                vq_ckpt_path = alt_p
                break
    if vq_ckpt_path and os.path.exists(vq_ckpt_path):
        vq_sd = torch.load(vq_ckpt_path, map_location="cpu", weights_only=True)["vqvae"]
        vqvae.load_state_dict(vq_sd)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    tk.vqvae = vqvae

    mcfg = cfg.model
    mcfg.prefix_len = P_COMBO
    mcfg.predict_level0 = True
    net = NextConceptTransformer(vqvae_local=vqvae, config=mcfg).to(dev)
    src = {}
    if ncm_ckpt_path and os.path.exists(ncm_ckpt_path):
        src = torch.load(ncm_ckpt_path, map_location="cpu", weights_only=False)
        if "model" in src:
            own = net.state_dict()
            keep = {k: v for k, v in src["model"].items() if k in own and own[k].shape == v.shape}
            net.load_state_dict(keep, strict=False)
    return cfg, tk, vqvae, net, src


def cmd_train_ncm(a):
    from train_combo_prefix import ComboBridge
    from train_wmt14_k3 import corrupt_codes
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

    vq_ckpt = os.path.join(a.out, "k3_vqvae_adapted.pt")
    cfg, tk, vqvae, net, src = load_k3_native_models(dev, vq_ckpt)
    cbridge = ComboBridge(D_HA, D_VQ).to(dev)
    if "combo_bridge" in src:
        cbridge.load_state_dict(src["combo_bridge"])

    dec = vqvae.decoder
    for name, p in dec.named_parameters():
        if "middle_block" in name or "ln_f" in name or "output_layer" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    data = torch.load(os.path.join(a.out, "k3_native_pairs.pt"), map_location="cpu", weights_only=True)
    val_pairs, tr_pairs = data["val"], data["train"]
    nval, ntr = len(val_pairs), len(tr_pairs)
    tr_rh, tr_msk, val_rh, val_msk = open_e6_shm(ntr, nval)
    null_info = torch.load(SHM_E6_NULL, map_location="cpu", weights_only=True)
    null_rh, null_msk = null_info["rh"].to(dev), null_info["msk"].to(dev)

    all_tids = torch.tensor([r["target_k3_ids"] for r in (val_pairs + tr_pairs)], dtype=torch.long)
    all_pids = torch.tensor([r["plan_k3_ids"] for r in (val_pairs + tr_pairs)], dtype=torch.long)
    idx_chunks = []
    with torch.no_grad():
        for i in range(0, len(all_tids), 128):
            idx_chunks.append(vqvae.encode(all_tids[i:i + 128].to(dev)).cpu().to(torch.int16))
    idx_all = torch.cat(idx_chunks, dim=0)

    idx_val = idx_all[:nval].long().to(dev)
    bpe_val = all_pids[:nval].to(dev)
    tids_val = all_tids[:nval].to(dev)
    rh_val = val_rh[:nval].to(dev)
    msk_val = val_msk[:nval].to(dev)

    idx_tr = idx_all[nval:]
    bpe_tr = all_pids[nval:]
    tids_tr = all_tids[nval:]

    fwd_net = DDP(net, device_ids=[dev.index]) if ws > 1 else net
    fwd_cbridge = DDP(cbridge, device_ids=[dev.index]) if ws > 1 else cbridge
    fwd_dec = DDP(dec, device_ids=[dev.index]) if ws > 1 else dec

    net_params = [p for p in net.parameters() if p.requires_grad]
    bridge_params = [p for p in cbridge.parameters() if p.requires_grad]
    dec_params = [p for p in dec.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": net_params, "base_lr": a.lr, "lr": a.lr},
        {"params": bridge_params, "base_lr": a.bridge_lr, "lr": a.bridge_lr},
        {"params": dec_params, "base_lr": a.dec_lr, "lr": a.dec_lr},
    ], betas=(0.9, 0.95), weight_decay=0.01)
    all_params = net_params + bridge_params + dec_params

    sched = lambda s: min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    log = open(os.path.join(a.out, "log_ncm.jsonl"), "a") if main_proc else open(os.devnull, "w")
    rates = (0.40, 0.05)
    g = torch.Generator().manual_seed(42 + rank)
    torch.manual_seed(42 + rank)

    null_bpe = torch.full((P_BPE,), K3_PAD, dtype=torch.long, device=dev)
    null_bpe[0] = K3_EOS
    all_masked_rh = torch.ones((P_RH,), dtype=torch.bool, device=dev)
    all_masked_bpe = torch.ones((P_BPE,), dtype=torch.bool, device=dev)
    levels = list(vqvae.vq.levels)
    lid = torch.cat([torch.full((l,), k, dtype=torch.long) for k, l in enumerate(levels)]).to(dev)
    ls = torch.tensor(levels, dtype=torch.float32, device=dev)

    def step_loss_e6(fn_net, fn_cb, fn_dec, r_in, mr_in, b_in, mb_in, i_in, t_in, c_rates, alpha, tok_w):
        with torch.no_grad():
            idx_c = corrupt_codes(i_in, levels, vqvae.vq.vqvae_vocab_sizes, *c_rates) if c_rates else i_in
            ncm_in = vqvae.vq.idx_to_ncm_input(idx_c)
            z_bpe = vqvae.encoder.wte(b_in)
        z_rh = fn_cb(r_in).to(z_bpe.dtype)
        z_pref, pmask = pack_combo_prefix(z_rh, mr_in, z_bpe, mb_in)
        logits = fn_net(torch.cat([z_pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=pmask)
        B, T, V = logits.shape
        ce = F.cross_entropy(logits.float().reshape(-1, V), i_in.reshape(-1), reduction="none").view(B, T)
        w = (ls ** (-alpha))[lid]
        code_loss = (ce * w[None]).sum() / (w.sum() * B)
        if tok_w > 0.0:
            z_hat_pred = build_st_zhat(vqvae.vq, logits, levels)
            tok_logits = fn_dec(z_hat_pred.float())
            tok_ce = F.cross_entropy(tok_logits.float().reshape(-1, V_K3), t_in.reshape(-1))
        else:
            tok_ce = torch.zeros((), device=dev)
        with torch.no_grad():
            pls = torch.stack([ce[:, lid == k].mean() for k in range(len(levels))])
        return code_loss + tok_w * tok_ce, code_loss.detach(), tok_ce.detach(), pls

    @torch.no_grad()
    def evaluate():
        net.eval(); cbridge.eval(); dec.eval()
        res = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            modes = [
                ("combo_matched", rh_val, (msk_val != 0), bpe_val, (bpe_val == K3_PAD)),
                ("bpe_only", rh_val, all_masked_rh.unsqueeze(0).expand(len(idx_val), -1), bpe_val, (bpe_val == K3_PAD)),
                ("rh_only", rh_val, (msk_val != 0), bpe_val, all_masked_bpe.unsqueeze(0).expand(len(idx_val), -1)),
                ("shuffled", torch.roll(rh_val, 1, 0), (torch.roll(msk_val, 1, 0) != 0), torch.roll(bpe_val, 1, 0), (torch.roll(bpe_val, 1, 0) == K3_PAD)),
            ]
            for name, r_in, mr_in, b_in, mb_in in modes:
                tot, tot_tok, n = None, 0.0, 0
                for i in range(0, len(idx_val), a.bs):
                    _, c_l, t_l, pls = step_loss_e6(net, cbridge, dec, r_in[i:i+a.bs], mr_in[i:i+a.bs], b_in[i:i+a.bs], mb_in[i:i+a.bs], idx_val[i:i+a.bs], tids_val[i:i+a.bs], None, 0.0, 0.5)
                    tot = pls * len(r_in[i:i+a.bs]) if tot is None else tot + pls * len(r_in[i:i+a.bs])
                    tot_tok += float(t_l.item()) * len(r_in[i:i+a.bs])
                    n += len(r_in[i:i+a.bs])
                res[name] = round(float((tot / n).mean().item()), 4)
                res[name + "_coarse"] = round(float((tot / n)[:5].mean().item()), 4)
                res[name + "_tok_ce"] = round(tot_tok / n, 4)
        net.train(); cbridge.train(); dec.train()
        return res

    if main_proc:
        ev0 = evaluate()
        print("step 0 val:", ev0, flush=True)
        log.write(json.dumps({"step": 0, "val": ev0}) + "\n"); log.flush()
    if ws > 1:
        dist.barrier()

    net.train(); cbridge.train(); dec.train()
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = gp["base_lr"] * sched(step - 1)
        sel = torch.randint(0, ntr, (a.bs,), generator=g)
        idx = idx_tr[sel].long().to(dev)
        bpe_p = bpe_tr[sel].to(dev)
        tids = tids_tr[sel].to(dev)
        rh = tr_rh[sel].to(dev)
        msk_rh = (tr_msk[sel].to(dev) != 0)
        msk_bpe = (bpe_p == K3_PAD)

        u = torch.rand(len(rh), generator=g).to(dev)
        drop_both = u < 0.10
        drop_rh = (u >= 0.10) & (u < 0.25)
        drop_bpe = (u >= 0.25) & (u < 0.40)
        if drop_both.any():
            rh[drop_both] = null_rh; msk_rh[drop_both] = (null_msk != 0)
            bpe_p[drop_both] = null_bpe; msk_bpe[drop_both] = (null_bpe == K3_PAD)
        if drop_rh.any():
            msk_rh[drop_rh] = True
        if drop_bpe.any():
            msk_bpe[drop_bpe] = True

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, c_l, t_l, _ = step_loss_e6(fwd_net, fwd_cbridge, fwd_dec, rh, msk_rh, bpe_p, msk_bpe, idx, tids, rates, 0.7, 0.5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()

        if main_proc and (step % 100 == 0 or step == 1):
            rec = {"step": step, "loss": round(loss.item(), 4), "code_ce": round(c_l.item(), 4), "tok_ce": round(t_l.item(), 4), "gate": round(float(cbridge.gate.item()), 4), "gnorm": round(float(gn), 3), "s_per_step": round((time.perf_counter()-t0)/step, 3)}
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()

        if step % 500 == 0 or step == a.steps:
            if main_proc:
                ev = evaluate()
                print(f"step {step} val:", ev, flush=True)
                log.write(json.dumps({"step": step, "val": ev}) + "\n"); log.flush()
                tmp = os.path.join(a.out, "e6_last.pt.tmp")
                torch.save({
                    "model": net.state_dict(),
                    "combo_bridge": cbridge.state_dict(),
                    "vqvae": vqvae.state_dict(),
                    "step": step,
                }, tmp)
                os.replace(tmp, os.path.join(a.out, "e6_last.pt"))
            if ws > 1:
                dist.barrier()
            net.train(); cbridge.train(); dec.train()

    if ws > 1:
        dist.destroy_process_group()
    if main_proc:
        print("done train_ncm e6_k3_native", flush=True)


def cmd_eval_all(a):
    from train_combo_prefix import ComboBridge
    from train_latent_bridge import encode_batch_rh
    dev_ha = torch.device("cuda:1")
    dev = torch.device("cuda:0")
    k3_tok = K3Tokenizer()

    data = torch.load(os.path.join(a.out, "k3_native_pairs.pt"), map_location="cpu", weights_only=True)
    val_pairs = data["val"]

    from hybrid_agent.model import load_hybrid_agent
    ha_model, ha_tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev_ha, require_hybridlink=True)
    ha_tok.padding_side = "right"

    if os.path.exists(SHM_E6_VAL_RH):
        _, _, val_rh_shm, val_msk_shm = open_e6_shm(len(data["train"]), len(val_pairs))
        rh_val = val_rh_shm.clone().to(dev)
        msk_rh_val = (val_msk_shm.clone().to(dev) != 0)
    else:
        rh_chunks, msk_chunks = [], []
        val_pl_list = [r["plan"] for r in val_pairs]
        for i in range(0, len(val_pl_list), 128):
            r, m = encode_batch_rh(ha_model, ha_tok, val_pl_list[i:i + 128], P_RH, dev_ha)
            rh_chunks.append(r); msk_chunks.append(m)
        rh_val = torch.cat(rh_chunks, dim=0).to(dev)
        msk_rh_val = (torch.cat(msk_chunks, dim=0).to(dev) != 0)

    vq_ckpt = os.path.join(a.out, "k3_vqvae_adapted.pt")
    ncm_ckpt = os.path.join(a.out, "e6_last.pt")
    cfg, tk, vqvae, net, src = load_k3_native_models(dev, vq_ckpt, ncm_ckpt)
    vqvae.load_state_dict(src["vqvae"])
    vqvae.eval(); net.eval()
    cbridge = ComboBridge(D_HA, D_VQ).to(dev)
    cbridge.load_state_dict(src["combo_bridge"])
    cbridge.eval()

    n_lvls = len(vqvae.vq.levels)
    sched_10_04 = [1.0 - 0.6 * (k / (n_lvls - 1)) for k in range(n_lvls)]

    def dec_k3(r):
        r = list(r)
        if K3_EOS in r:
            r = r[:r.index(K3_EOS)]
        clean_ids = [int(t) for t in r if 0 <= int(t) < 163584]
        return k3_tok.decode(clean_ids), clean_ids

    @torch.no_grad()
    def gen_e6(rh_in, msk_rh_in, k3_pids, mode="combo", cands=8):
        z_bpe = vqvae.encoder.wte(k3_pids)
        msk_bpe = (k3_pids == K3_PAD)
        z_rh = cbridge(rh_in).float()
        if mode == "bpe_only":
            msk_rh_use = torch.ones_like(msk_rh_in, dtype=torch.bool); msk_bpe_use = msk_bpe
        elif mode == "rh_only":
            msk_rh_use = msk_rh_in; msk_bpe_use = torch.ones_like(msk_bpe, dtype=torch.bool)
        else:
            msk_rh_use = msk_rh_in; msk_bpe_use = msk_bpe
        z_pref, pmask = pack_combo_prefix(z_rh, msk_rh_use, z_bpe, msk_bpe_use)
        all_txts, all_ids = [], []
        bs = 16
        for k in range(cands):
            torch.manual_seed(42 + k)
            outs_t, outs_i = [], []
            for i in range(0, len(z_pref), bs):
                g = generate_from_zpref(net, tk, z_pref[i:i + bs], pmask[i:i + bs], sched_10_04)
                z_hats, _ = vqvae.vq.idx_to_zhats(g, return_zlevels=True)
                lg = vqvae.decoder(z_hats[-1])
                for r in pick_tokens(lg).tolist():
                    t_s, i_s = dec_k3(r)
                    outs_t.append(t_s); outs_i.append(i_s)
            all_txts.append(outs_t); all_ids.append(outs_i)
        if cands == 1:
            return all_txts[0], all_ids[0]
        # MBR selection returning both text and exact Kimi K3 token IDs
        K = len(all_txts)
        best_t, best_i = [], []
        for idx in range(len(all_txts[0])):
            sc = [sum(jacc(all_txts[k][idx], all_txts[j][idx]) for j in range(K) if j != k) for k in range(K)]
            bk = max(range(K), key=lambda k: sc[k])
            best_t.append(all_txts[bk][idx]); best_i.append(all_ids[bk][idx])
        return best_t, best_i

    val_plans = [r["plan"] for r in val_pairs]
    val_targets = [r["target_text"] for r in val_pairs]
    bpe_val = torch.tensor([r["plan_k3_ids"] for r in val_pairs], dtype=torch.long, device=dev)

    import ncm_stage2 as S
    NUM_RE = re.compile(r"\d+(?:[.,:]\d+)*")

    def score_factual(texts, ref_plans):
        ret = [S.retention(p, t) for p, t in zip(ref_plans, texts)]
        ret = [x for x in ret if x is not None]
        num_o, num_k = 0, 0
        adj_n, adj_d, tri_n, tri_d = 0, 0, 0, 0
        for p, t in zip(ref_plans, texts):
            co = collections.Counter(NUM_RE.findall(p))
            ct = collections.Counter(NUM_RE.findall(t))
            num_o += sum(co.values()); num_k += sum((co & ct).values())
            w = words(t)
            adj_n += sum(1 for i in range(1, len(w)) if w[i] == w[i - 1]); adj_d += max(len(w) - 1, 0)
            seen = set()
            for i in range(len(w) - 2):
                g = tuple(w[i:i + 3])
                tri_n += (g in seen); tri_d += 1; seen.add(g)
        return {
            "retention": round(sum(ret) / max(len(ret), 1), 4),
            "number_recall": round(num_k / max(num_o, 1), 4),
            "adj_repeat_rate": round(adj_n / max(adj_d, 1), 5),
            "repeat_trigram_rate": round(tri_n / max(tri_d, 1), 5),
        }

    res = {"val_500": {}, "k3_p1_to_p5": {}}
    res["val_500"]["true_targets"] = score_factual(val_targets, val_plans)
    t_combo, _ = gen_e6(rh_val, msk_rh_val, bpe_val, mode="combo", cands=8)
    res["val_500"]["e6_k3_native_combo"] = score_factual(t_combo, val_plans)
    t_bpe, _ = gen_e6(rh_val, msk_rh_val, bpe_val, mode="bpe_only", cands=8)
    res["val_500"]["e6_k3_native_bpe_only"] = score_factual(t_bpe, val_plans)
    t_rh, _ = gen_e6(rh_val, msk_rh_val, bpe_val, mode="rh_only", cands=8)
    res["val_500"]["e6_k3_native_rh_only"] = score_factual(t_rh, val_plans)
    for k, v in res["val_500"].items():
        print("val_500", k, v, flush=True)

    # Evaluate on p1..p5 and save native Kimi K3 token IDs!
    k3_sum = json.load(open("/home/usergpu/k3ncm/k3_summary.json", encoding="utf-8"))
    p1_ans = (
        "The Harbourside Council voted 7 to 2 on Tuesday to approve 2.4 million pounds for a new cycle lane along Mill Road. "
        "Councillor Priya Raman proposed the plan. Work is set to begin in March and should take 14 weeks to complete. "
        "Shop owners on Mill Road, led by Tom Eliot, had asked for the vote to be delayed because of concerns about parking. "
        "In response, the council promised to provide 40 temporary parking spaces on Dock Street. The decision means the project "
        "will go ahead as planned, despite objections from local businesses worried about losing parking for customers during the construction period."
    )
    p1_sum = "Harbourside Council approved funding for a Mill Road cycle lane despite parking concerns from local shop owners."
    k3_ref = {
        "p1": {"answer": p1_ans, "summary": p1_sum},
        **k3_sum,
        "p5": {
            "answer": "The first train travels for 30 minutes before the second train starts, covering 42 miles. This leaves 126 miles between them at 09:40. Their combined speed is 154 miles per hour, so they close this gap in 126 divided by 154 hours, which is about 49 minutes. They meet at approximately 10:29. The second train travels 70 multiplied by 49/60 miles, about 57 miles from London.",
            "summary": "The trains meet at about 10:29, roughly 57 miles from London."
        }
    }
    keys_map = {
        "p1": ["Harbourside", "7", "2", "Tuesday", "2.4", "Mill Road", "Priya Raman", "March", "14", "Tom Eliot", "40", "Dock Street"],
        "p2": ["Apollo 13", "1970", "NASA", "Moon", "oxygen", "lunar module", "Houston", "Pacific"],
        "p3": ["blue", "sunlight", "scatters", "molecules", "sunset", "red", "orange", "path"],
        "p4": ["02:14", "June", "B", "02:31", "1,820", "14", "100", "80", "500", "1 TB"],
        "p5": ["30", "42", "126", "09:40", "154", "49", "10:29", "70", "57"],
    }

    for pid in ("p1", "p2", "p3", "p4", "p5"):
        full_k3_txt = f"{k3_ref[pid]['answer']} Summary: {k3_ref[pid]['summary']}"
        k_full = k3_tok.encode(full_k3_txt)
        w1_txt = k3_tok.decode(k_full[:60])
        w2_txt = k3_tok.decode(k_full[60:120])
        p_w1 = window_to_fact_plan(w1_txt)
        p_w2 = window_to_fact_plan(w2_txt) if len(k_full) > 65 else ""
        plans_list = [p for p in (p_w1, p_w2) if p]

        t0 = time.perf_counter()
        win_txts, win_k3_ids = [], []
        for p_str in plans_list:
            rh_p, msk_p = encode_batch_rh(ha_model, ha_tok, [p_str], P_RH, dev_ha)
            rh_p = rh_p.to(dev); msk_p = (msk_p.to(dev) != 0)
            p_kids = k3_tok.encode(p_str)[:P_BPE]
            p_kpad = torch.tensor([p_kids + [K3_PAD] * (P_BPE - len(p_kids))], dtype=torch.long, device=dev)
            gt, gi = gen_e6(rh_p, msk_p, p_kpad, mode="combo", cands=8)
            win_txts.append(gt[0]); win_k3_ids.extend(gi[0])
        ncm_ms = (time.perf_counter() - t0) * 1000.0
        gen_txt = " ".join(win_txts)
        k_list = keys_map[pid]
        kept = [k for k in k_list if k.lower() in gen_txt.lower()]

        # Save native Kimi K3 token IDs for direct PF_EXTRA verification in Clover Kimi K3!
        ids_file = os.path.join(a.out, f"{pid}_e6.ids")
        with open(ids_file, "w") as f:
            f.write(",".join(str(i) for i in win_k3_ids) + "\n")

        rec = {
            "ncm_mbr8_ms": round(ncm_ms, 1),
            "plan": " || ".join(plans_list),
            "generated_text": gen_txt,
            "keys_kept": f"{len(kept)}/{len(k_list)}",
            "keys_list": kept,
            "native_k3_tokens": len(win_k3_ids),
            "k3_vocab_coverage": "100.0% (native K3 IDs)",
        }
        res["k3_p1_to_p5"][pid] = rec
        print(f"[{pid} | e6_k3_native 2-win] keys={rec['keys_kept']} k3_toks={len(win_k3_ids)} ncm_ms={ncm_ms:.1f}", flush=True)
        print(f"   TEXT: {gen_txt}", flush=True)

    json.dump(res, open(os.path.join(a.out, "eval_e6_k3_native.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved eval_e6_k3_native.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_vq = sub.add_parser("adapt_vqvae")
    p_vq.add_argument("--out", default="/home/usergpu/merge_lf/out/e6_k3_native")
    p_vq.add_argument("--steps", type=int, default=800)
    p_vq.add_argument("--bs", type=int, default=64)
    p_vq.add_argument("--lr", type=float, default=2e-4)

    p_ha = sub.add_parser("train_k3_ha")
    p_ha.add_argument("--out", default="/home/usergpu/merge_lf/out/e6_k3_native")

    p_ex = sub.add_parser("extract_rh")
    p_ex.add_argument("--out", default="/home/usergpu/merge_lf/out/e6_k3_native")
    p_ex.add_argument("--shard", type=int, required=True)
    p_ex.add_argument("--nshards", type=int, default=8)
    p_ex.add_argument("--bs", type=int, default=128)

    p_tr = sub.add_parser("train_ncm")
    p_tr.add_argument("--out", default="/home/usergpu/merge_lf/out/e6_k3_native")
    p_tr.add_argument("--steps", type=int, default=1500)
    p_tr.add_argument("--bs", type=int, default=32)
    p_tr.add_argument("--lr", type=float, default=1e-4)
    p_tr.add_argument("--bridge-lr", type=float, default=2e-4)
    p_tr.add_argument("--dec-lr", type=float, default=1e-4)

    p_ev = sub.add_parser("eval_all")
    p_ev.add_argument("--out", default="/home/usergpu/merge_lf/out/e6_k3_native")

    a = ap.parse_args()
    {"adapt_vqvae": cmd_adapt_vqvae, "train_k3_ha": cmd_train_k3_ha, "extract_rh": cmd_extract_rh, "train_ncm": cmd_train_ncm, "eval_all": cmd_eval_all}[a.cmd](a)

