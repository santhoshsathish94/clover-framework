"""Cycle 46: Combined [HybridLink R(h) + Discrete BPE Plan] Prefix for NCM Stage-2 (e4_combo).
Tests whether combining continuous HybridAgent R(h) vectors (via LatentBridge) AND discrete BPE plan tokens
in a single packed prefix [valid R(h) tokens, valid BPE tokens, PAD...] captures the best of both tracks:
- Track 2A's continuous semantic conditioning from HybridAgent's 4096-d hidden states R(h)
- Track 2B's exact surface-word/character-name retention from discrete BPE tokens.

Subcommands:
  extract : 8-GPU parallel extraction of R(h) [N, 48, 4096] into /dev/shm (if not already present).
  train   : 8-GPU DDP training (1,500 steps, global batch 256) initialized from e4_fluency/last.pt (net + vq_decoder)
            and e4_latent/last.pt (LatentBridge), with modality dropout (60% both, 15% BPE-only, 15% R(h)-only, 10% null).
  eval    : Evaluates e4_combo on all 598 validation plans and 48 e2e topics (with both the fine-tuned VQ decoder
            and CausalVQDecoder cdec.pt), plus within-checkpoint ablation (both vs BPE-only vs R(h)-only), and exports
            subset20_c46.json for blind 3-model-rater evaluation.
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation.gen_ppl import generation_perplexity

import ncm_stage2 as S
from train_fluency import build_st_zhat
from train_latent_bridge import (
    LatentBridge, open_shm_tensors, encode_batch_rh,
    N_TRAIN, N_VAL, P_LEN, D_HA, SHM_NULL_RH, SHM_TR_RH
)
from eval_both_tracks import words, jacc, mbr, pick_tokens, generate_from_zpref
from train_causal_decoder import CausalVQDecoder
from eval_causal_decoder import idx_to_zhat_upto

P_RH = 48
P_BPE = 64
P_COMBO = P_RH + P_BPE  # 112


class ComboBridge(nn.Module):
    """Wraps LatentBridge(4096 -> 384) with a learnable modality embedding and gate for the R(h) prefix segment."""
    def __init__(self, d_in=4096, d_out=384):
        super().__init__()
        self.bridge = LatentBridge(d_in, d_out)
        self.seg_rh = nn.Parameter(torch.zeros(1, 1, d_out))
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, r_h):
        z = self.bridge(r_h)
        return (z * self.gate + self.seg_rh).to(r_h.dtype)


def pack_combo_prefix(z_rh, msk_rh_bool, z_bpe, msk_bpe_bool):
    """Packs [valid R(h) tokens, valid BPE tokens, PAD...] contiguously so (~pad_mask).sum(1)-1 is always the last valid token."""
    valid_rh = ~msk_rh_bool    # [B, 48] True where valid
    valid_bpe = ~msk_bpe_bool  # [B, 64] True where valid
    z_cat = torch.cat([z_rh, z_bpe], dim=1)            # [B, 112, 384]
    valid_cat = torch.cat([valid_rh, valid_bpe], dim=1)  # [B, 112]
    perm = (~valid_cat).int().argsort(dim=1, stable=True)
    z_combo = z_cat.gather(1, perm.unsqueeze(-1).expand(-1, -1, z_cat.shape[-1]))
    pad_mask_combo = ~valid_cat.gather(1, perm)
    return z_combo, pad_mask_combo


def cmd_extract(a):
    from hybrid_agent.model import load_hybrid_agent
    dev = torch.device(f"cuda:{a.shard}")
    torch.cuda.set_device(dev)
    t0 = time.perf_counter()
    model, tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev, require_hybridlink=True)
    tok.padding_side = "right"
    tr_rh, tr_msk, val_rh, val_msk = open_shm_tensors(create=False)

    if a.shard == 0:
        null_rh, null_msk = encode_batch_rh(model, tok, [""], P_LEN, dev)
        torch.save({"rh": null_rh[0], "msk": null_msk[0]}, SHM_NULL_RH)

    vrows = S.read_pairs(a.val_pairs)[:N_VAL]
    trows = S.read_pairs(a.pairs)[:N_TRAIN]

    v_idx = list(range(a.shard, len(vrows), a.nshards))
    for i in range(0, len(v_idx), a.bs):
        b_ids = v_idx[i:i + a.bs]
        rh, msk = encode_batch_rh(model, tok, [vrows[j]["plan"] for j in b_ids], P_LEN, dev)
        for k, j in enumerate(b_ids):
            val_rh[j].copy_(rh[k]); val_msk[j].copy_(msk[k])

    t_idx = list(range(a.shard, len(trows), a.nshards))
    for i in range(0, len(t_idx), a.bs):
        b_ids = t_idx[i:i + a.bs]
        rh, msk = encode_batch_rh(model, tok, [trows[j]["plan"] for j in b_ids], P_LEN, dev)
        for k, j in enumerate(b_ids):
            tr_rh[j].copy_(rh[k]); tr_msk[j].copy_(msk[k])
    print(f"[shard {a.shard}] extracted {len(t_idx)} train + {len(v_idx)} val R(h) in {time.perf_counter()-t0:.1f}s", flush=True)


def step_loss_combo(fwd_net, fwd_cbridge, tk, fwd_dec, rh, msk_rh_bool, bpe_plan_ids, msk_bpe_bool, idx, true_ids, corrupt_rates, alpha, tok_ce_weight):
    vq = tk.vqvae.vq
    levels = list(vq.levels)
    with torch.no_grad():
        idx_in = S.corrupt(idx, levels, vq.vqvae_vocab_sizes, *corrupt_rates) if corrupt_rates else idx
        ncm_in = vq.idx_to_ncm_input(idx_in)
        z_bpe = tk.vqvae.encoder.wte(bpe_plan_ids)
    z_rh = fwd_cbridge(rh).to(z_bpe.dtype)
    z_pref, pad_mask_combo = pack_combo_prefix(z_rh, msk_rh_bool, z_bpe, msk_bpe_bool)
    logits = fwd_net(torch.cat([z_pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=pad_mask_combo)
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.float().reshape(-1, V), idx.reshape(-1), reduction="none").view(B, T)
    lid = S.level_ids(levels, idx.device)
    ls = torch.tensor(levels, dtype=torch.float32, device=idx.device)
    w = (ls ** (-alpha))[lid]
    code_loss = (ce * w[None]).sum() / (w.sum() * B)

    if tok_ce_weight > 0.0:
        z_hat_pred = build_st_zhat(vq, logits, levels)
        tok_logits = fwd_dec(z_hat_pred.float())
        tok_ce = F.cross_entropy(tok_logits.float().reshape(-1, tok_logits.shape[-1]), true_ids.reshape(-1))
    else:
        tok_ce = torch.zeros((), device=idx.device)

    total_loss = code_loss + tok_ce_weight * tok_ce
    with torch.no_grad():
        per_level = torch.stack([ce[:, lid == k].mean() for k in range(len(levels))])
    return total_loss, code_loss.detach(), tok_ce.detach(), per_level


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

    # Initialize generator and fine-tuned VQ decoder from e4_fluency/last.pt, with prefix_len = P_COMBO = 112
    CFG, tk, net, src_f, rep = S.load_models(dev, P_COMBO, ckpt=a.init_fluency)
    if "vq_decoder" in src_f:
        tk.vqvae.decoder.load_state_dict(src_f["vq_decoder"])
    pad_id = tk.pre_tokenizer.pad_token_id
    eos_id = tk.pre_tokenizer.eos_token_id

    # Initialize ComboBridge from e4_latent/last.pt
    cbridge = ComboBridge(D_HA, tk.vqvae.embed_dim).to(dev)
    src_l = torch.load(a.init_latent, map_location="cpu", weights_only=False)
    cbridge.bridge.load_state_dict(src_l["latent_bridge"])

    dec = tk.vqvae.decoder
    for name, p in dec.named_parameters():
        if "middle_block" in name or "ln_f" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    tr_rh, tr_msk, val_rh, val_msk = open_shm_tensors(create=False)
    null_info = torch.load(SHM_NULL_RH, map_location="cpu", weights_only=True)
    null_rh, null_msk = null_info["rh"].to(dev), null_info["msk"].to(dev)

    vrows = S.read_pairs(a.val_pairs)[:a.nval]
    trows = S.read_pairs(a.pairs)[:N_TRAIN]
    rows = vrows + trows
    nval = len(vrows)

    os.makedirs(a.out, exist_ok=True)
    idx_all = S.encode_stories(tk, [r["ids"] for r in rows], dev)
    plans_all = S.encode_plans(tk, [r["plan"] for r in rows], P_BPE, pad_id)
    true_ids_all = torch.tensor([r["ids"] for r in rows], dtype=torch.long)

    idx_val = idx_all[:nval].long().to(dev)
    bpe_val = plans_all[:nval].to(dev)
    tids_val = true_ids_all[:nval].to(dev)
    rh_val = val_rh[:nval].to(dev)
    msk_val = val_msk[:nval].to(dev)

    idx_tr = idx_all[nval:]
    bpe_tr = plans_all[nval:]
    tids_tr = true_ids_all[nval:]

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

    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    log = open(os.path.join(a.out, "log.jsonl"), "a") if main_proc else open(os.devnull, "w")
    rates = (a.corrupt_first, a.corrupt_last)
    g = torch.Generator().manual_seed(a.seed + rank)
    torch.manual_seed(a.seed + rank)

    null_bpe = torch.full((P_BPE,), pad_id, dtype=torch.long, device=dev)
    null_bpe[0] = eos_id
    all_masked_rh = torch.ones((P_RH,), dtype=torch.bool, device=dev)
    all_masked_bpe = torch.ones((P_BPE,), dtype=torch.bool, device=dev)

    @torch.no_grad()
    def evaluate():
        net.eval(); cbridge.eval(); dec.eval()
        res = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            modes = [
                ("combo_matched", rh_val, (msk_val != 0), bpe_val, (bpe_val == pad_id)),
                ("bpe_only", rh_val, all_masked_rh.unsqueeze(0).expand(len(idx_val), -1), bpe_val, (bpe_val == pad_id)),
                ("rh_only", rh_val, (msk_val != 0), bpe_val, all_masked_bpe.unsqueeze(0).expand(len(idx_val), -1)),
                ("shuffled", torch.roll(rh_val, 1, 0), (torch.roll(msk_val, 1, 0) != 0), torch.roll(bpe_val, 1, 0), (torch.roll(bpe_val, 1, 0) == pad_id)),
            ]
            for name, r_in, mr_in, b_in, mb_in in modes:
                tot, tot_tok, n = None, 0.0, 0
                for i in range(0, len(idx_val), a.bs):
                    _, c_loss, t_loss, pls = step_loss_combo(
                        net, cbridge, tk, dec,
                        r_in[i:i + a.bs], mr_in[i:i + a.bs],
                        b_in[i:i + a.bs], mb_in[i:i + a.bs],
                        idx_val[i:i + a.bs], tids_val[i:i + a.bs],
                        None, 0.0, a.tok_ce_weight
                    )
                    tot = pls * len(r_in[i:i + a.bs]) if tot is None else tot + pls * len(r_in[i:i + a.bs])
                    tot_tok += float(t_loss.item()) * len(r_in[i:i + a.bs])
                    n += len(r_in[i:i + a.bs])
                res[name] = round(float((tot / n).mean().item()), 4)
                res[name + "_coarse"] = round(float((tot / n)[:6].mean().item()), 4)
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
        sel = torch.randint(0, N_TRAIN, (a.bs,), generator=g)
        idx = idx_tr[sel].long().to(dev)
        bpe_p = bpe_tr[sel].to(dev)
        tids = tids_tr[sel].to(dev)
        rh = tr_rh[sel].to(dev)
        msk_rh = (tr_msk[sel].to(dev) != 0)
        msk_bpe = (bpe_p == pad_id)

        # Modality dropout: [0, 0.10) = null both; [0.10, 0.25) = BPE-only (mask R(h)); [0.25, 0.40) = R(h)-only (mask BPE); [0.40, 1.0) = both!
        u = torch.rand(len(rh), generator=g).to(dev)
        drop_both = u < 0.10
        drop_rh = (u >= 0.10) & (u < 0.25)
        drop_bpe = (u >= 0.25) & (u < 0.40)
        if drop_both.any():
            rh[drop_both] = null_rh
            msk_rh[drop_both] = (null_msk != 0)
            bpe_p[drop_both] = null_bpe
            msk_bpe[drop_both] = (null_bpe == pad_id)
        if drop_rh.any():
            msk_rh[drop_rh] = True
        if drop_bpe.any():
            msk_bpe[drop_bpe] = True

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, c_loss, t_loss, _ = step_loss_combo(
                fwd_net, fwd_cbridge, tk, fwd_dec,
                rh, msk_rh, bpe_p, msk_bpe, idx, tids,
                rates, a.alpha, a.tok_ce_weight
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()

        if main_proc and (step % 100 == 0 or step == 1):
            rec = {
                "step": step,
                "loss": round(loss.item(), 4),
                "code_ce": round(c_loss.item(), 4),
                "tok_ce": round(t_loss.item(), 4),
                "gate": round(float(cbridge.gate.item()), 4),
                "gnorm": round(float(gn), 3),
                "s_per_step": round((time.perf_counter() - t0) / step, 3),
            }
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()

        if step % a.eval_every == 0 or step == a.steps:
            if main_proc:
                ev = evaluate()
                print(f"step {step} val:", ev, flush=True)
                log.write(json.dumps({"step": step, "val": ev}) + "\n"); log.flush()
                tmp = os.path.join(a.out, "last.pt.tmp")
                torch.save({
                    "model": net.state_dict(),
                    "combo_bridge": cbridge.state_dict(),
                    "vq_decoder": dec.state_dict(),
                    "inference": src_f.get("inference"),
                    "step": step,
                    "args": vars(a),
                    "l0_head": net.l0_head_mode,
                }, tmp)
                os.replace(tmp, os.path.join(a.out, "last.pt"))
            if ws > 1:
                dist.barrier()
            net.train(); cbridge.train(); dec.train()

    if ws > 1:
        dist.destroy_process_group()
    if main_proc:
        print("done train_combo_prefix", flush=True)


def cmd_eval(a):
    dev = torch.device("cuda:0")
    vrows = S.read_pairs(a.val_pairs)[:598]
    val_plans = [r["plan"] for r in vrows]

    _, _, val_rh_shm, val_msk_shm = open_shm_tensors(create=False)
    rh_val = val_rh_shm[:598].clone().to(dev)
    msk_rh_val = (val_msk_shm[:598].clone().to(dev) != 0)

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

    CFG, tk, net, src, _ = S.load_models(dev, P_COMBO, ckpt=os.path.join(a.out, "last.pt"))
    net.eval()
    tk.vqvae.decoder.load_state_dict(src["vq_decoder"])
    tk.vqvae.decoder.eval()
    pad_id, eos = tk.pre_tokenizer.pad_token_id, tk.pre_tokenizer.eos_token_id

    cbridge = ComboBridge(D_HA, tk.vqvae.embed_dim).to(dev)
    cbridge.load_state_dict(src["combo_bridge"])
    cbridge.eval()

    cdec = CausalVQDecoder(tk.vqvae, n_layer=4, n_head=6, dropout=0.1).to(dev)
    cdec_sd = torch.load("/home/usergpu/merge_lf/out/cdec/cdec.pt", map_location="cpu", weights_only=True)["cdec"]
    cdec.load_state_dict(cdec_sd)
    cdec.eval()

    def dec(r):
        r = list(r)
        if eos in r:
            r = r[:r.index(eos)]
        return tk.pre_tokenizer.decode([t for t in r if t not in (eos, pad_id)], skip_special_tokens=True)

    sched_10_04 = [1.0 - 0.6 * (k / 11.0) for k in range(12)]
    bs = 64

    @torch.no_grad()
    def gen_combo(rh_in, msk_rh_in, plans_list, use_cdec=False, cands=8):
        pids = S.encode_plans(tk, plans_list, P_BPE, pad_id).to(dev)
        z_bpe = tk.vqvae.encoder.wte(pids)
        msk_bpe = (pids == pad_id)
        z_rh = cbridge(rh_in).float()
        z_pref, pmask = pack_combo_prefix(z_rh, msk_rh_in, z_bpe, msk_bpe)
        all_c = []
        for k in range(cands):
            torch.manual_seed(42 + k)
            outs = []
            for i in range(0, len(z_pref), bs):
                g = generate_from_zpref(net, tk, z_pref[i:i + bs], pmask[i:i + bs], sched_10_04)
                if use_cdec:
                    z_hat = idx_to_zhat_upto(tk.vqvae.vq, g, max_lvl=12)
                    t_ids = cdec.generate_tokens(z_hat, eos, temp=0.0, adj=(4.0, 6, 0.7), ng=(3, 6.0))
                else:
                    z_hats, _ = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
                    lg = tk.vqvae.decoder(z_hats[-1])
                    t_ids = pick_tokens(lg)
                outs += [dec(r) for r in t_ids.tolist()]
            all_c.append(outs)
        return mbr(all_c) if cands > 1 else all_c[0]

    res = {"val_598": {}, "e2e_48": {}}
    t_combo_ftdec = gen_combo(rh_val, msk_rh_val, val_plans, use_cdec=False, cands=8)
    res["val_598"]["e4_combo_ftdec"] = score_all(t_combo_ftdec, val_plans)
    print("val_598 e4_combo_ftdec:", res["val_598"]["e4_combo_ftdec"], flush=True)

    t_combo_cdec = gen_combo(rh_val, msk_rh_val, val_plans, use_cdec=True, cands=8)
    res["val_598"]["e4_combo_cdec"] = score_all(t_combo_cdec, val_plans)
    print("val_598 e4_combo_cdec:", res["val_598"]["e4_combo_cdec"], flush=True)

    # E2E 48 topics
    top_all = torch.load("/home/usergpu/merge_lf/top_shard_0.pt", map_location="cpu", weights_only=True)
    top_all.sort(key=lambda x: x["idx"])
    tp_48 = [x["text_plan"] for x in top_all]
    hp_48 = [x["hybrid_plan"] for x in top_all]
    rh_tp = torch.stack([x["rh_text"] for x in top_all]).to(dev)
    msk_tp = (torch.stack([x["msk_text"] for x in top_all]) != 0).to(dev)
    rh_hp = torch.stack([x["rh_hybrid"] for x in top_all]).to(dev)
    msk_hp = (torch.stack([x["msk_hybrid"] for x in top_all]) != 0).to(dev)

    res["e2e_48"]["e4_combo_ftdec_textplan"] = score_all(gen_combo(rh_tp, msk_tp, tp_48, use_cdec=False, cands=8), tp_48)
    res["e2e_48"]["e4_combo_ftdec_hybridplan"] = score_all(gen_combo(rh_hp, msk_hp, hp_48, use_cdec=False, cands=8), hp_48)
    res["e2e_48"]["e4_combo_cdec_textplan"] = score_all(gen_combo(rh_tp, msk_tp, tp_48, use_cdec=True, cands=8), tp_48)
    res["e2e_48"]["e4_combo_cdec_hybridplan"] = score_all(gen_combo(rh_hp, msk_hp, hp_48, use_cdec=True, cands=8), hp_48)
    for k, v in res["e2e_48"].items():
        print("e2e_48", k, v, flush=True)

    sub = json.load(open("/home/usergpu/merge_lf/out/subset20_c45.json", encoding="utf-8"))
    p_idxs = sub["plan_idxs"]
    sub["systems"]["e4_combo_ftdec"] = [t_combo_ftdec[i] for i in p_idxs]
    sub["systems"]["e4_combo_cdec"] = [t_combo_cdec[i] for i in p_idxs]
    json.dump(sub, open("/home/usergpu/merge_lf/out/subset20_c46.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(res, open("/home/usergpu/merge_lf/out/eval_c46.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    for idx in range(4):
        pi = p_idxs[idx]
        print(f"=== PLAN {pi}: {val_plans[pi]} ===")
        print(f"  [e4_combo_ftdec]: {t_combo_ftdec[pi][:350]}")
        print(f"  [e4_combo_cdec ]: {t_combo_cdec[pi][:350]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--shard", type=int, required=True)
    p_ext.add_argument("--nshards", type=int, default=8)
    p_ext.add_argument("--bs", type=int, default=128)
    p_ext.add_argument("--pairs", default="/home/usergpu/plan2text/data/pairs_train.jsonl")
    p_ext.add_argument("--val-pairs", default="/home/usergpu/plan2text/data/pairs_valid.jsonl")

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--pairs", default="/home/usergpu/plan2text/data/pairs_train.jsonl")
    p_tr.add_argument("--val-pairs", default="/home/usergpu/plan2text/data/pairs_valid.jsonl")
    p_tr.add_argument("--init-fluency", default="/home/usergpu/merge_lf/out/e4_fluency/last.pt")
    p_tr.add_argument("--init-latent", default="/home/usergpu/merge_lf/out/e4_latent/last.pt")
    p_tr.add_argument("--out", default="/home/usergpu/merge_lf/out/e4_combo")
    p_tr.add_argument("--steps", type=int, default=1500)
    p_tr.add_argument("--bs", type=int, default=32)
    p_tr.add_argument("--lr", type=float, default=4e-5)
    p_tr.add_argument("--bridge-lr", type=float, default=1e-4)
    p_tr.add_argument("--dec-lr", type=float, default=6e-5)
    p_tr.add_argument("--warmup", type=int, default=100)
    p_tr.add_argument("--alpha", type=float, default=0.7)
    p_tr.add_argument("--corrupt-first", type=float, default=0.4)
    p_tr.add_argument("--corrupt-last", type=float, default=0.05)
    p_tr.add_argument("--tok-ce-weight", type=float, default=0.5)
    p_tr.add_argument("--nval", type=int, default=512)
    p_tr.add_argument("--eval-every", type=int, default=500)
    p_tr.add_argument("--seed", type=int, default=42)

    p_ev = sub.add_parser("eval")
    p_ev.add_argument("--val-pairs", default="/home/usergpu/plan2text/data/pairs_valid.jsonl")
    p_ev.add_argument("--out", default="/home/usergpu/merge_lf/out/e4_combo")

    a = ap.parse_args()
    {"extract": cmd_extract, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)
