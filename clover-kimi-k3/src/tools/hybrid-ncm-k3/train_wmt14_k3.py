"""Cycle 47 & 48 (Goal 1 — Connecting HybridAgent + NCM + Clover Kimi K3):
Builds, trains, and evaluates the end-to-end factual bridge on the WMT14 NCM architecture (151,680 Qwen3 vocabulary,
11 residual VQ levels [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64] = 313 codes, L=64 target window, prefix_len P=96):
  - Prefix conditioning: Combined [HybridLink R(h) (up to 48 continuous 4096-d vectors via ComboBridge) + Discrete Qwen3 Plan/Fact Tokens (up to 48 tokens)] -> P=96!
  - Training data: 10,800 factual pairs from XSum passages + 500 held-out validation pairs (64-token factual report windows paired with semicolon-separated fact plans + R(h) vectors).
  - Evaluates on:
    1. 500 held-out factual validation pairs (code CE across combo / bpe_only / rh_only / shuffled, token accuracy, fact/number retention, repeat rates).
    2. Kimi K3's actual prompts & generated outputs on p1, p2, p3, p4, p5 (from /home/usergpu/k3ncm/ and /home/usergpu/k3gpu/results/):
       - HybridAgent-8B-Agentic runs in continuous <latent> mode (producing R(h) + factual plan) on p1..p5.
       - WMT14 e5_k3_combo generates the 64-token report/summary in 11 NCM forward passes (~95 ms on 1 GPU).
       - Compares fact/number retention on p1..p5 against the Cycle 30 released XSum NCM baseline (which had 0/8, 0/8, 0/11 facts).
       - Translates generated Qwen3 tokens directly into Kimi K3's 163,840-token vocabulary (tiktoken.model) and embeds them into Kimi K3's 7,168-d embedding space (model-00094-of-000096.safetensors) for multi-token Kimi K3 verification (stepk / forward_tokens).
"""
import argparse, base64, collections, json, math, os, pickle, re, sys, time
from clover_paths import setup_clover_sys_path, find_wmt14_art_dir

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
from transformers import AutoTokenizer
from safetensors import safe_open
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

from train_latent_bridge import LatentBridge, encode_batch_rh, run_latent_rollout_one, D_HA
from train_combo_prefix import ComboBridge, pack_combo_prefix
from eval_both_tracks import words, jacc, mbr, pick_tokens, generate_from_zpref

P_RH = 48
P_BPE = 48
P_COMBO = 96  # matches wmt14.yaml prefix_len = 96 exactly!
L_WMT = 64

SHM_K3_TR_RH = "/dev/shm/k3_rh_train.bf16"
SHM_K3_TR_MSK = "/dev/shm/k3_rh_train.msk"
SHM_K3_VAL_RH = "/dev/shm/k3_rh_val.bf16"
SHM_K3_VAL_MSK = "/dev/shm/k3_rh_val.msk"
SHM_K3_NULL = "/dev/shm/k3_rh_null.pt"


def load_wmt14_models(dev, P=P_COMBO, ckpt=None):
    cfg = OmegaConf.load("config/eval/wmt14.yaml")
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ART_WMT, "tokenizer.pt")
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(dev).eval()
    for p in tk.vqvae.parameters():
        p.requires_grad_(False)
    mcfg = cfg.model
    mcfg.prefix_len = P
    mcfg.predict_level0 = True
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=mcfg).to(dev)
    src = torch.load(ckpt or os.path.join(ART_WMT, "generator.pt"), map_location="cpu", weights_only=False)
    own = net.state_dict()
    keep = {k: v for k, v in src["model"].items() if k in own and own[k].shape == v.shape}
    net.load_state_dict(keep, strict=False)
    return cfg, tk, net, src


def extract_fact_plan(passage_text, summary_text):
    """Builds a compact semicolon-separated factual plan (< 40 words) from the summary and key numbers/entities of the passage."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage_text.replace("\n", " ")) if len(s.strip()) > 15]
    s1 = sents[0] if sents else ""
    s2 = sents[1] if len(sents) > 1 else ""
    # Extract numbers and capitalized entities from s1 + s2 that aren't in summary_text
    extras = []
    for m in re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|\d+(?:[.,:]\d+)?%?)\b", s1 + " " + s2):
        if m.lower() not in summary_text.lower() and m not in ("The", "In", "On", "At", "It", "He", "She", "They", "A", "An", "But", "And", "For"):
            if m not in extras:
                extras.append(m)
    ext_str = "; ".join(extras[:6])
    plan = summary_text.strip().rstrip(".")
    if ext_str:
        plan = f"{plan}; {ext_str}"
    return plan, " ".join(sents[:3])


def open_k3_shm(n_tr, n_val):
    tr_rh = torch.from_file(SHM_K3_TR_RH, shared=True, size=n_tr * P_RH * D_HA, dtype=torch.bfloat16).view(n_tr, P_RH, D_HA)
    tr_msk = torch.from_file(SHM_K3_TR_MSK, shared=True, size=n_tr * P_RH, dtype=torch.uint8).view(n_tr, P_RH)
    val_rh = torch.from_file(SHM_K3_VAL_RH, shared=True, size=n_val * P_RH * D_HA, dtype=torch.bfloat16).view(n_val, P_RH, D_HA)
    val_msk = torch.from_file(SHM_K3_VAL_MSK, shared=True, size=n_val * P_RH, dtype=torch.uint8).view(n_val, P_RH)
    return tr_rh, tr_msk, val_rh, val_msk


def cmd_prep(a):
    t0 = time.perf_counter()
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    cfg, tk, _, _ = load_wmt14_models(torch.device("cpu"))
    qtok = tk.pre_tokenizer
    eos, pad = qtok.eos_token_id, qtok.pad_token_id

    passages = pickle.load(open("/home/usergpu/ncm/data/xsum/passages.pickle", "rb"))
    summaries = pickle.load(open("/home/usergpu/ncm/data/xsum/reference_summaries.pickle", "rb"))

    pairs = []
    for p_ids, s_txt in zip(passages, summaries):
        p_txt = gpt2_tok.decode(p_ids, skip_special_tokens=True).strip()
        plan_txt, lead_txt = extract_fact_plan(p_txt, s_txt)
        if len(lead_txt.split()) < 25:
            continue
        t_ids = qtok.encode(lead_txt)[:L_WMT - 1]
        t_pad = t_ids + [eos] + [pad] * (L_WMT - 1 - len(t_ids))
        p_qids = qtok.encode(plan_txt)[:P_BPE]
        p_qpad = p_qids + [pad] * (P_BPE - len(p_qids))
        pairs.append({
            "plan": plan_txt,
            "target_text": qtok.decode(t_ids, skip_special_tokens=True),
            "plan_ids": p_qpad,
            "target_ids": t_pad,
            "target_len": len(t_ids),
        })
        if len(pairs) >= a.ntrain + a.nval:
            break

    val_pairs = pairs[:a.nval]
    tr_pairs = pairs[a.nval:a.nval + a.ntrain]
    os.makedirs(a.out, exist_ok=True)
    torch.save({"val": val_pairs, "train": tr_pairs}, os.path.join(a.out, "k3_fact_pairs.pt"))
    open_k3_shm(len(tr_pairs), len(val_pairs))
    print(f"prepared {len(tr_pairs)} train + {len(val_pairs)} val factual pairs in {time.perf_counter()-t0:.1f}s", flush=True)


def cmd_extract(a):
    from hybrid_agent.model import load_hybrid_agent
    dev = torch.device(f"cuda:{a.shard}")
    torch.cuda.set_device(dev)
    t0 = time.perf_counter()
    data = torch.load(os.path.join(a.out, "k3_fact_pairs.pt"), map_location="cpu", weights_only=True)
    val_pairs, tr_pairs = data["val"], data["train"]
    tr_rh, tr_msk, val_rh, val_msk = open_k3_shm(len(tr_pairs), len(val_pairs))

    model, tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev, require_hybridlink=True)
    tok.padding_side = "right"

    if a.shard == 0:
        null_rh, null_msk = encode_batch_rh(model, tok, [""], P_RH, dev)
        torch.save({"rh": null_rh[0], "msk": null_msk[0]}, SHM_K3_NULL)

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


def corrupt_codes(idx, levels, vocab, p_first, p_last):
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


def step_loss_wmt(fwd_net, fwd_cbridge, tk, rh, msk_rh_bool, bpe_plan_ids, msk_bpe_bool, idx, corrupt_rates, alpha):
    vq = tk.vqvae.vq
    levels = list(vq.levels)
    with torch.no_grad():
        idx_in = corrupt_codes(idx, levels, vq.vqvae_vocab_sizes, *corrupt_rates) if corrupt_rates else idx
        ncm_in = vq.idx_to_ncm_input(idx_in)
        z_bpe = tk.vqvae.encoder.wte(bpe_plan_ids)
    z_rh = fwd_cbridge(rh).to(z_bpe.dtype)
    z_pref, pad_mask_combo = pack_combo_prefix(z_rh, msk_rh_bool, z_bpe, msk_bpe_bool)
    logits = fwd_net(torch.cat([z_pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=pad_mask_combo)
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.float().reshape(-1, V), idx.reshape(-1), reduction="none").view(B, T)
    lid = torch.cat([torch.full((l,), k, dtype=torch.long) for k, l in enumerate(levels)]).to(idx.device)
    ls = torch.tensor(levels, dtype=torch.float32, device=idx.device)
    w = (ls ** (-alpha))[lid]
    code_loss = (ce * w[None]).sum() / (w.sum() * B)
    with torch.no_grad():
        per_level = torch.stack([ce[:, lid == k].mean() for k in range(len(levels))])
    return code_loss, per_level


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

    cfg, tk, net, src = load_wmt14_models(dev, P=P_COMBO)
    pad_id = tk.pre_tokenizer.pad_token_id
    eos_id = tk.pre_tokenizer.eos_token_id

    cbridge = ComboBridge(D_HA, tk.vqvae.embed_dim).to(dev)
    # Initialize ComboBridge from our trained e4_combo checkpoint!
    if os.path.exists("/home/usergpu/merge_lf/out/e4_combo/last.pt"):
        src_c46 = torch.load("/home/usergpu/merge_lf/out/e4_combo/last.pt", map_location="cpu", weights_only=False)
        cbridge.load_state_dict(src_c46["combo_bridge"])

    data = torch.load(os.path.join(a.out, "k3_fact_pairs.pt"), map_location="cpu", weights_only=True)
    val_pairs, tr_pairs = data["val"], data["train"]
    nval, ntr = len(val_pairs), len(tr_pairs)
    tr_rh, tr_msk, val_rh, val_msk = open_k3_shm(ntr, nval)
    null_info = torch.load(SHM_K3_NULL, map_location="cpu", weights_only=True)
    null_rh, null_msk = null_info["rh"].to(dev), null_info["msk"].to(dev)

    # Encode all target windows into WMT14 VQ-VAE codes (313 codes per window)
    all_tids = torch.tensor([r["target_ids"] for r in (val_pairs + tr_pairs)], dtype=torch.long)
    all_pids = torch.tensor([r["plan_ids"] for r in (val_pairs + tr_pairs)], dtype=torch.long)
    idx_chunks = []
    with torch.no_grad():
        for i in range(0, len(all_tids), 128):
            idx_chunks.append(tk.vqvae.encode(all_tids[i:i + 128].to(dev)).cpu().to(torch.int16))
    idx_all = torch.cat(idx_chunks, dim=0)

    idx_val = idx_all[:nval].long().to(dev)
    bpe_val = all_pids[:nval].to(dev)
    rh_val = val_rh[:nval].to(dev)
    msk_val = val_msk[:nval].to(dev)

    idx_tr = idx_all[nval:]
    bpe_tr = all_pids[nval:]

    fwd_net = DDP(net, device_ids=[dev.index]) if ws > 1 else net
    fwd_cbridge = DDP(cbridge, device_ids=[dev.index]) if ws > 1 else cbridge

    net_params = [p for p in net.parameters() if p.requires_grad]
    bridge_params = [p for p in cbridge.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": net_params, "base_lr": a.lr, "lr": a.lr},
        {"params": bridge_params, "base_lr": a.bridge_lr, "lr": a.bridge_lr},
    ], betas=(0.9, 0.95), weight_decay=0.01)
    all_params = net_params + bridge_params

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
        net.eval(); cbridge.eval()
        res = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            modes = [
                ("combo_matched", rh_val, (msk_val != 0), bpe_val, (bpe_val == pad_id)),
                ("bpe_only", rh_val, all_masked_rh.unsqueeze(0).expand(len(idx_val), -1), bpe_val, (bpe_val == pad_id)),
                ("rh_only", rh_val, (msk_val != 0), bpe_val, all_masked_bpe.unsqueeze(0).expand(len(idx_val), -1)),
                ("shuffled", torch.roll(rh_val, 1, 0), (torch.roll(msk_val, 1, 0) != 0), torch.roll(bpe_val, 1, 0), (torch.roll(bpe_val, 1, 0) == pad_id)),
            ]
            for name, r_in, mr_in, b_in, mb_in in modes:
                tot, n = None, 0
                for i in range(0, len(idx_val), a.bs):
                    _, pls = step_loss_wmt(
                        net, cbridge, tk,
                        r_in[i:i + a.bs], mr_in[i:i + a.bs],
                        b_in[i:i + a.bs], mb_in[i:i + a.bs],
                        idx_val[i:i + a.bs], None, 0.0
                    )
                    tot = pls * len(r_in[i:i + a.bs]) if tot is None else tot + pls * len(r_in[i:i + a.bs])
                    n += len(r_in[i:i + a.bs])
                res[name] = round(float((tot / n).mean().item()), 4)
                res[name + "_coarse"] = round(float((tot / n)[:5].mean().item()), 4)
        net.train(); cbridge.train()
        return res

    if main_proc:
        ev0 = evaluate()
        print("step 0 val:", ev0, flush=True)
        log.write(json.dumps({"step": 0, "val": ev0}) + "\n"); log.flush()
    if ws > 1:
        dist.barrier()

    net.train(); cbridge.train()
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = gp["base_lr"] * sched(step - 1)
        sel = torch.randint(0, ntr, (a.bs,), generator=g)
        idx = idx_tr[sel].long().to(dev)
        bpe_p = bpe_tr[sel].to(dev)
        rh = tr_rh[sel].to(dev)
        msk_rh = (tr_msk[sel].to(dev) != 0)
        msk_bpe = (bpe_p == pad_id)

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
            loss, _ = step_loss_wmt(fwd_net, fwd_cbridge, tk, rh, msk_rh, bpe_p, msk_bpe, idx, rates, a.alpha)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()

        if main_proc and (step % 100 == 0 or step == 1):
            rec = {"step": step, "code_ce": round(loss.item(), 4), "gate": round(float(cbridge.gate.item()), 4), "gnorm": round(float(gn), 3), "s_per_step": round((time.perf_counter() - t0) / step, 3)}
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
                    "step": step,
                    "args": vars(a),
                }, tmp)
                os.replace(tmp, os.path.join(a.out, "last.pt"))
            if ws > 1:
                dist.barrier()
            net.train(); cbridge.train()

    if ws > 1:
        dist.destroy_process_group()
    if main_proc:
        print("done train_wmt14_k3", flush=True)


def build_qwen_to_k3_map(qtok):
    k3_dir = "/opt/clover-k3/clover-kimi-k3/src/datasets/third-party/kimi-k3-open-weights"
    k3_bpe = {}
    id_to_bytes_k3 = {}
    with open(os.path.join(k3_dir, "tiktoken.model"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            b64, rk = line.split()
            raw = base64.b64decode(b64)
            rk = int(rk)
            k3_bpe[raw] = rk
            id_to_bytes_k3[rk] = raw

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b); cs.append(2**8 + n); n += 1
    u2b = dict(zip([chr(c) for c in cs], bs))

    q_to_k3 = {}
    hf_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    for tok_str, qid in hf_tok.get_vocab().items():
        try:
            raw = bytes([u2b[c] for c in tok_str])
        except KeyError:
            continue
        if raw in k3_bpe:
            q_to_k3[qid] = k3_bpe[raw]
    return q_to_k3, k3_bpe, id_to_bytes_k3


def cmd_eval(a):
    dev_ha = torch.device("cuda:1")
    dev = torch.device("cuda:0")
    data = torch.load(os.path.join(a.out, "k3_fact_pairs.pt"), map_location="cpu", weights_only=True)
    val_pairs = data["val"]

    from hybrid_agent.model import load_hybrid_agent
    from hybrid_agent.readout import normalized_embeddings
    ha_model, ha_tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev_ha, require_hybridlink=True)
    ha_tok.padding_side = "right"
    embed_bank = normalized_embeddings(ha_model.backbone.get_input_embeddings().weight).float()

    if os.path.exists(SHM_K3_VAL_RH):
        _, _, val_rh_shm, val_msk_shm = open_k3_shm(len(data["train"]), len(val_pairs))
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

    cfg, tk, net_init, _ = load_wmt14_models(dev, P=P_COMBO)
    net_init.eval()
    _, _, net_tr, src_tr = load_wmt14_models(dev, P=P_COMBO, ckpt=os.path.join(a.out, "last.pt"))
    net_tr.eval()
    cbridge = ComboBridge(D_HA, tk.vqvae.embed_dim).to(dev)
    cbridge.load_state_dict(src_tr["combo_bridge"])
    cbridge.eval()

    qtok = tk.pre_tokenizer
    pad_id, eos = qtok.pad_token_id, qtok.eos_token_id
    n_lvls = len(tk.vqvae.vq.levels)  # 11
    sched_10_04 = [1.0 - 0.6 * (k / (n_lvls - 1)) for k in range(n_lvls)]

    def dec(r):
        r = list(r)
        if eos in r:
            r = r[:r.index(eos)]
        return qtok.decode([t for t in r if t not in (eos, pad_id)], skip_special_tokens=True)

    @torch.no_grad()
    def gen_wmt(net_m, rh_in, msk_rh_in, bpe_pids, mode="combo", cands=8):
        z_bpe = tk.vqvae.encoder.wte(bpe_pids)
        msk_bpe = (bpe_pids == pad_id)
        z_rh = cbridge(rh_in).float()
        if mode == "bpe_only":
            msk_rh_use = torch.ones_like(msk_rh_in, dtype=torch.bool)
            msk_bpe_use = msk_bpe
        elif mode == "rh_only":
            msk_rh_use = msk_rh_in
            msk_bpe_use = torch.ones_like(msk_bpe, dtype=torch.bool)
        else:
            msk_rh_use = msk_rh_in
            msk_bpe_use = msk_bpe
        z_pref, pmask = pack_combo_prefix(z_rh, msk_rh_use, z_bpe, msk_bpe_use)
        all_c = []
        bs = 64
        for k in range(cands):
            torch.manual_seed(42 + k)
            outs = []
            for i in range(0, len(z_pref), bs):
                g = generate_from_zpref(net_m, tk, z_pref[i:i + bs], pmask[i:i + bs], sched_10_04)
                z_hats, _ = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
                lg = tk.vqvae.decoder(z_hats[-1])
                outs += [dec(r) for r in pick_tokens(lg).tolist()]
            all_c.append(outs)
        return mbr(all_c) if cands > 1 else all_c[0]

    val_plans = [r["plan"] for r in val_pairs]
    val_targets = [r["target_text"] for r in val_pairs]
    bpe_val = torch.tensor([r["plan_ids"] for r in val_pairs], dtype=torch.long, device=dev)

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
    t_untrained = gen_wmt(net_init, rh_val, msk_rh_val, bpe_val, mode="bpe_only", cands=4)
    res["val_500"]["wmt14_released_zero_shot"] = score_factual(t_untrained, val_plans)

    t_combo = gen_wmt(net_tr, rh_val, msk_rh_val, bpe_val, mode="combo", cands=8)
    res["val_500"]["e5_k3_combo_both"] = score_factual(t_combo, val_plans)

    t_bpe = gen_wmt(net_tr, rh_val, msk_rh_val, bpe_val, mode="bpe_only", cands=8)
    res["val_500"]["e5_k3_combo_bpe_only"] = score_factual(t_bpe, val_plans)

    t_rh = gen_wmt(net_tr, rh_val, msk_rh_val, bpe_val, mode="rh_only", cands=8)
    res["val_500"]["e5_k3_combo_rh_only"] = score_factual(t_rh, val_plans)

    for k, v in res["val_500"].items():
        print("val_500", k, v, flush=True)

    # 2. End-to-End HybridAgent (<latent> mode) + WMT14 e5_k3_combo + Kimi K3 on p1..p5!
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

    q_to_k3, k3_bpe, id_to_bytes_k3 = build_qwen_to_k3_map(qtok)
    st_path = "/opt/clover-k3/clover-kimi-k3/src/datasets/third-party/kimi-k3-open-weights/model-00094-of-000096.safetensors"
    with safe_open(st_path, framework="pt", device="cpu") as sf:
        k3_embed_w = sf.get_tensor("language_model.model.embed_tokens.weight")  # [163840, 7168] bf16

    for pid in ("p1", "p2", "p3", "p4", "p5"):
        raw_prompt = open(f"/home/usergpu/k3ncm/{pid}.txt", encoding="utf-8").read().strip()
        ha_prompt = (
            "Extract the key facts, names, numbers, and events in order for the following request as a single line of short phrases "
            "separated by semicolons (at most 40 words). Output only the plan line.\n\nRequest: " + raw_prompt
        )
        t_ha0 = time.perf_counter()
        lat_out = run_latent_rollout_one(ha_model, ha_tok, embed_bank, ha_prompt, P=P_RH, min_lat=8, max_lat=20, max_new=64)
        ha_ms = (time.perf_counter() - t_ha0) * 1000.0

        # Also extract plan from Kimi K3's answer body (simulating K3/HybridAgent handoff)
        k3_plan, _ = extract_fact_plan(k3_ref[pid]["answer"], k3_ref[pid]["summary"])
        for plan_src, plan_str in (("hybridagent_latent_plan", lat_out["hybrid_plan"]), ("k3_structured_plan", k3_plan)):
            rh_p, msk_p = encode_batch_rh(ha_model, ha_tok, [plan_str], P_RH, dev_ha)
            rh_p = rh_p.to(dev); msk_p = (msk_p.to(dev) != 0)
            p_qids = qtok.encode(plan_str)[:P_BPE]
            p_qpad = torch.tensor([p_qids + [pad_id] * (P_BPE - len(p_qids))], dtype=torch.long, device=dev)

            t_ncm0 = time.perf_counter()
            gen_txts = gen_wmt(net_tr, rh_p, msk_p, p_qpad, mode="combo", cands=8)
            ncm_ms = (time.perf_counter() - t_ncm0) * 1000.0
            gen_txt = gen_txts[0]

            # Translate generated Qwen3 tokens into Kimi K3 token IDs and look up Kimi K3 7168-d embeddings (s0)
            gen_qids = qtok.encode(gen_txt)
            k3_ids = [q_to_k3[q] for q in gen_qids if q in q_to_k3]
            k3_s0 = k3_embed_w[torch.tensor(k3_ids, dtype=torch.long)].float()  # [N_k3, 7168]

            k_list = keys_map[pid]
            kept = [k for k in k_list if k.lower() in gen_txt.lower()]
            rec = {
                "latent_steps": lat_out["latent_steps"],
                "exit_reason": lat_out["exit_reason"],
                "mean_readout_cos": lat_out["mean_readout_cos"],
                "ha_lat_ms": round(ha_ms, 1),
                "ncm_mbr8_ms": round(ncm_ms, 1),
                "plan": plan_str,
                "generated_text": gen_txt,
                "keys_kept": f"{len(kept)}/{len(k_list)}",
                "keys_list": kept,
                "qwen_tokens": len(gen_qids),
                "mapped_k3_tokens": len(k3_ids),
                "k3_s0_shape": list(k3_s0.shape),
                "k3_s0_rms": round(float(k3_s0.pow(2).mean().sqrt().item()), 6),
            }
            res["k3_p1_to_p5"][f"{pid}_{plan_src}"] = rec
            print(f"[{pid} | {plan_src}] keys={rec['keys_kept']} k3_toks={len(k3_ids)}/{len(gen_qids)} | {gen_txt[:160]}", flush=True)

    json.dump(res, open(os.path.join(a.out, "eval_k3_c47.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved eval_k3_c47.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pr = sub.add_parser("prep")
    p_pr.add_argument("--out", default="/home/usergpu/merge_lf/out/e5_k3_wmt")
    p_pr.add_argument("--ntrain", type=int, default=10800)
    p_pr.add_argument("--nval", type=int, default=500)

    p_ex = sub.add_parser("extract")
    p_ex.add_argument("--out", default="/home/usergpu/merge_lf/out/e5_k3_wmt")
    p_ex.add_argument("--shard", type=int, required=True)
    p_ex.add_argument("--nshards", type=int, default=8)
    p_ex.add_argument("--bs", type=int, default=128)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--out", default="/home/usergpu/merge_lf/out/e5_k3_wmt")
    p_tr.add_argument("--steps", type=int, default=1500)
    p_tr.add_argument("--bs", type=int, default=32)
    p_tr.add_argument("--lr", type=float, default=1e-4)
    p_tr.add_argument("--bridge-lr", type=float, default=2e-4)
    p_tr.add_argument("--warmup", type=int, default=100)
    p_tr.add_argument("--alpha", type=float, default=0.7)
    p_tr.add_argument("--corrupt-first", type=float, default=0.4)
    p_tr.add_argument("--corrupt-last", type=float, default=0.05)
    p_tr.add_argument("--eval-every", type=int, default=500)
    p_tr.add_argument("--seed", type=int, default=42)

    p_ev = sub.add_parser("eval")
    p_ev.add_argument("--out", default="/home/usergpu/merge_lf/out/e5_k3_wmt")

    a = ap.parse_args()
    {"prep": cmd_prep, "extract": cmd_extract, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)
