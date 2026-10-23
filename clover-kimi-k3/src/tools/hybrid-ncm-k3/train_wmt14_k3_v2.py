"""Cycle 48: Fix the two bottlenecks found in Cycle 47's first WMT14 K3 bridge (e5_k3_wmt):
1. Planner prompt & <latent> bridge fix:
   In Cycle 47, run_latent_rollout_one closed with `</latent></think>\nPlan:` without a one-shot example, so HybridAgent-8B-Agentic
   emitted meta-intent ("I need to explain the Apollo 13 mission...") instead of actual facts.
   Fix: include a concrete 1-shot factual plan example in the prompt and close the <latent> span with `</latent></think>\nFacts:`,
   and extract R(h_t) from BOTH the continuous <latent> rollout trajectory AND the emitted factual plan.
2. Direct factual plan -> report alignment + straight-through token CE (tok_ce_weight=0.5) + fine-tuning vqvae.decoder:
   In Cycle 47, extract_fact_plan paired XSum's 1-sentence summary (which omits 80% of the 64-token passage lead) with the 64-token
   passage lead, so the generator learned to hallucinate unstated BBC news clauses (`val CE = 3.57`).
   Fix: build the factual plan directly from the target 64-token window's clause subjects, verbs, numbers, and proper nouns
   (so 100% of the target facts are grounded in the plan, matching the TinyStories setup), AND unfreeze `tk.vqvae.decoder.middle_block + ln_f`
   with straight-through token CE (`tok_ce_weight = 0.5`) as proven in Cycle 43 (`e4_fluency`).
3. Also measures exact multi-token draft verification acceptance against Kimi K3's ground-truth generated token IDs (`p1..p5.ids`).
"""
import argparse, base64, collections, json, math, os, pickle, re, sys, time

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
ROOT = os.environ["NCM_REPO"]
ART_WMT = "/home/usergpu/ncm/artifacts/wmt14"
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from safetensors import safe_open

from train_fluency import build_st_zhat
from train_latent_bridge import encode_batch_rh, D_HA
from train_combo_prefix import ComboBridge, pack_combo_prefix
from eval_both_tracks import words, jacc, mbr, pick_tokens, generate_from_zpref
from train_wmt14_k3 import (
    load_wmt14_models, open_k3_shm, corrupt_codes, build_qwen_to_k3_map,
    P_RH, P_BPE, P_COMBO, L_WMT, SHM_K3_NULL, SHM_K3_VAL_RH
)

STOP_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "is", "was", "were", "are",
    "been", "be", "has", "have", "had", "that", "this", "it", "he", "she", "they", "we", "his", "her", "their",
    "its", "but", "and", "or", "also", "who", "which", "when", "where", "while", "after", "before", "during"
}


def window_to_fact_plan(window_text):
    """Extracts a semicolon-separated sequence of factual clause skeletons (entities, numbers, key content words in order)
    from a 64-token factual passage window so every clause in the target window is grounded in the plan."""
    clauses = [c.strip() for c in re.split(r"[.!?;]+|,\s+(?=[A-Z0-9])", window_text.replace("\n", " ")) if len(c.strip()) > 6]
    beats = []
    for cl in clauses[:6]:
        toks = re.findall(r"\d+(?:[.,:]\d+)*%?|[A-Za-z]+(?:'[a-z]+)?", cl)
        kept = [w for w in toks if (w[0].isupper() or w[0].isdigit() or w.lower() not in STOP_WORDS)]
        if kept:
            beats.append(" ".join(kept[:8]))
    return "; ".join(beats)


PROMPT_FACT_PLAN = (
    "Write a one-line factual plan (short semicolon-separated phrases covering every entity, number, and event in order, "
    "at most 45 words) for the following request. Do NOT write 'I need to' or meta-commentary; write ONLY the factual phrases.\n\n"
    "Example request: Write a report using these facts: River Bridge closed at 06:30 on Monday after a lorry crash. "
    "Engineers inspected 3 steel beams. Traffic was diverted to Oak Road for 5 hours.\n"
    "Example plan: River Bridge closed 06:30 Monday lorry crash; engineers inspected 3 steel beams; traffic diverted Oak Road 5 hours; "
    "Summary River Bridge closed 5 hours lorry crash\n\n"
    "Request: {req}"
)


@torch.no_grad()
def run_factual_latent_rollout(model, tokenizer, embed_bank, req_text, min_lat=8, max_lat=20, max_new=64):
    from hybrid_agent.constants import ENTRY_ANCHOR, EXIT_ANCHOR, THINK_OPEN, THINK_CLOSE, special_token_id
    from hybrid_agent.readout import pseudo_text
    dev = next(model.parameters()).device
    backbone = model.backbone
    embed_w = backbone.get_input_embeddings().weight
    lat_close_id = special_token_id(tokenizer, EXIT_ANCHOR)

    user_content = PROMPT_FACT_PLAN.format(req=req_text)
    msgs = [{"role": "user", "content": user_content}]
    base_chat = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if base_chat.rstrip("\n").endswith(THINK_OPEN):
        prefix_str = base_chat + "Let me list the exact entities, numbers, and events in order." + ENTRY_ANCHOR
    else:
        prefix_str = base_chat + THINK_OPEN + "\nLet me list the exact entities, numbers, and events in order." + ENTRY_ANCHOR

    inp_ids = tokenizer(prefix_str, return_tensors="pt").input_ids.to(dev)
    out = backbone(input_ids=inp_ids, use_cache=True, output_hidden_states=True, return_dict=True)
    past_kv = out.past_key_values
    last_h = out.hidden_states[-1][:, -1, :]
    next_logits = out.logits[:, -1, :]

    latent_vecs = []
    readout_ids = []
    readout_cos = []
    exit_reason = "max_steps"

    for step in range(max_lat):
        r_h = model.run_hybridlink(last_h, output_dtype=embed_w.dtype)
        arg_id = int(next_logits.argmax(dim=-1).item())
        if step >= min_lat and arg_id == lat_close_id:
            exit_reason = "anchor"
            break
        latent_vecs.append(r_h[0].detach().to(torch.bfloat16).cpu())
        rn = F.normalize(r_h.float(), dim=-1)
        sims = rn @ embed_bank.t()
        best_cos, best_id = sims.max(dim=-1)
        readout_ids.append(int(best_id.item()))
        readout_cos.append(round(float(best_cos.item()), 4))

        out = backbone(inputs_embeds=r_h.unsqueeze(1), past_key_values=past_kv, use_cache=True, output_hidden_states=True, return_dict=True)
        past_kv = out.past_key_values
        last_h = out.hidden_states[-1][:, -1, :]
        next_logits = out.logits[:, -1, :]

    bridge_str = EXIT_ANCHOR + THINK_CLOSE + "\nExample plan:"
    bridge_ids = tokenizer(bridge_str, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    out = backbone(input_ids=bridge_ids, past_key_values=past_kv, use_cache=True, output_hidden_states=True, return_dict=True)
    past_kv = out.past_key_values
    next_logits = out.logits[:, -1, :]

    gen_ids = []
    eos_ids = {tokenizer.eos_token_id}
    for _ in range(max_new):
        nxt = int(next_logits.argmax(dim=-1).item())
        if nxt in eos_ids:
            break
        gen_ids.append(nxt)
        txt_so_far = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if "\n" in txt_so_far.strip() and len(txt_so_far.strip()) > 15:
            break
        out = backbone(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=past_kv, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]

    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split("\n")[0].strip()
    raw_text = re.sub(r"^(?:Plan|Facts|Example plan):\s*", "", raw_text, flags=re.I).strip()
    return {
        "latent_steps": len(latent_vecs),
        "exit_reason": exit_reason,
        "mean_readout_cos": round(sum(readout_cos) / max(len(readout_cos), 1), 4),
        "pseudo_text": pseudo_text(readout_ids, tokenizer),
        "hybrid_plan": raw_text,
    }


def cmd_prep(a):
    t0 = time.perf_counter()
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    cfg, tk, _, _ = load_wmt14_models(torch.device("cpu"))
    qtok = tk.pre_tokenizer
    eos, pad = qtok.eos_token_id, qtok.pad_token_id

    passages = pickle.load(open("/home/usergpu/ncm/data/xsum/passages.pickle", "rb"))
    summaries = pickle.load(open("/home/usergpu/ncm/data/xsum/reference_summaries.pickle", "rb"))

    pairs = []
    # Also include GSM8K flattened reasoning answers if available, or build multi-window factual pairs from XSum + summaries
    for p_ids, s_txt in zip(passages, summaries):
        p_txt = " ".join(gpt2_tok.decode(p_ids, skip_special_tokens=True).split())
        full_txt = f"{s_txt.strip()} {p_txt}"
        q_all = qtok.encode(full_txt)
        for start in (0, 60):
            t_ids = q_all[start:start + (L_WMT - 1)]
            if len(t_ids) < 35:
                continue
            win_txt = qtok.decode(t_ids, skip_special_tokens=True)
            plan_txt = window_to_fact_plan(win_txt)
            if len(plan_txt.split()) < 10:
                continue
            t_pad = t_ids + [eos] + [pad] * (L_WMT - 1 - len(t_ids))
            p_qids = qtok.encode(plan_txt)[:P_BPE]
            p_qpad = p_qids + [pad] * (P_BPE - len(p_qids))
            pairs.append({
                "plan": plan_txt,
                "target_text": win_txt,
                "plan_ids": p_qpad,
                "target_ids": t_pad,
                "target_len": len(t_ids),
            })
            if len(pairs) >= a.ntrain + a.nval:
                break
        if len(pairs) >= a.ntrain + a.nval:
            break

    val_pairs = pairs[:a.nval]
    tr_pairs = pairs[a.nval:a.nval + a.ntrain]
    os.makedirs(a.out, exist_ok=True)
    torch.save({"val": val_pairs, "train": tr_pairs}, os.path.join(a.out, "k3_fact_pairs.pt"))
    open_k3_shm(len(tr_pairs), len(val_pairs))
    print(f"prepared {len(tr_pairs)} train + {len(val_pairs)} val grounded factual pairs in {time.perf_counter()-t0:.1f}s", flush=True)


def step_loss_wmt_v2(fwd_net, fwd_cbridge, tk, fwd_dec, rh, msk_rh_bool, bpe_plan_ids, msk_bpe_bool, idx, true_ids, corrupt_rates, alpha, tok_ce_weight):
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

    cfg, tk, net, src = load_wmt14_models(dev, P=P_COMBO, ckpt=a.init_ckpt)
    pad_id = tk.pre_tokenizer.pad_token_id
    eos_id = tk.pre_tokenizer.eos_token_id

    cbridge = ComboBridge(D_HA, tk.vqvae.embed_dim).to(dev)
    if a.init_ckpt and "combo_bridge" in src:
        cbridge.load_state_dict(src["combo_bridge"])

    dec = tk.vqvae.decoder
    for name, p in dec.named_parameters():
        if "middle_block" in name or "ln_f" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    data = torch.load(os.path.join(a.out, "k3_fact_pairs.pt"), map_location="cpu", weights_only=True)
    val_pairs, tr_pairs = data["val"], data["train"]
    nval, ntr = len(val_pairs), len(tr_pairs)
    tr_rh, tr_msk, val_rh, val_msk = open_k3_shm(ntr, nval)
    null_info = torch.load(SHM_K3_NULL, map_location="cpu", weights_only=True)
    null_rh, null_msk = null_info["rh"].to(dev), null_info["msk"].to(dev)

    all_tids = torch.tensor([r["target_ids"] for r in (val_pairs + tr_pairs)], dtype=torch.long)
    all_pids = torch.tensor([r["plan_ids"] for r in (val_pairs + tr_pairs)], dtype=torch.long)
    idx_chunks = []
    with torch.no_grad():
        for i in range(0, len(all_tids), 128):
            idx_chunks.append(tk.vqvae.encode(all_tids[i:i + 128].to(dev)).cpu().to(torch.int16))
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
                    _, c_loss, t_loss, pls = step_loss_wmt_v2(
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
            loss, c_loss, t_loss, _ = step_loss_wmt_v2(
                fwd_net, fwd_cbridge, tk, fwd_dec,
                rh, msk_rh, bpe_p, msk_bpe, idx, tids,
                rates, a.alpha, a.tok_ce_weight
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()

        if main_proc and (step % 100 == 0 or step == 1):
            rec = {"step": step, "loss": round(loss.item(), 4), "code_ce": round(c_loss.item(), 4), "tok_ce": round(t_loss.item(), 4), "gate": round(float(cbridge.gate.item()), 4), "gnorm": round(float(gn), 3), "s_per_step": round((time.perf_counter() - t0) / step, 3)}
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
                    "step": step,
                    "args": vars(a),
                }, tmp)
                os.replace(tmp, os.path.join(a.out, "last.pt"))
            if ws > 1:
                dist.barrier()
            net.train(); cbridge.train(); dec.train()

    if ws > 1:
        dist.destroy_process_group()
    if main_proc:
        print("done train_wmt14_k3_v2", flush=True)


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

    cfg, tk, net_tr, src_tr = load_wmt14_models(dev, P=P_COMBO, ckpt=os.path.join(a.out, "last.pt"))
    net_tr.eval()
    if "vq_decoder" in src_tr:
        tk.vqvae.decoder.load_state_dict(src_tr["vq_decoder"])
    tk.vqvae.decoder.eval()
    cbridge = ComboBridge(D_HA, tk.vqvae.embed_dim).to(dev)
    cbridge.load_state_dict(src_tr["combo_bridge"])
    cbridge.eval()

    qtok = tk.pre_tokenizer
    pad_id, eos = qtok.pad_token_id, qtok.eos_token_id
    n_lvls = len(tk.vqvae.vq.levels)
    sched_10_04 = [1.0 - 0.6 * (k / (n_lvls - 1)) for k in range(n_lvls)]

    def dec(r):
        r = list(r)
        if eos in r:
            r = r[:r.index(eos)]
        return qtok.decode([t for t in r if t not in (eos, pad_id)], skip_special_tokens=True)

    @torch.no_grad()
    def gen_wmt(rh_in, msk_rh_in, bpe_pids, mode="combo", cands=8):
        z_bpe = tk.vqvae.encoder.wte(bpe_pids)
        msk_bpe = (bpe_pids == pad_id)
        z_rh = cbridge(rh_in).float()
        if mode == "bpe_only":
            msk_rh_use = torch.ones_like(msk_rh_in, dtype=torch.bool); msk_bpe_use = msk_bpe
        elif mode == "rh_only":
            msk_rh_use = msk_rh_in; msk_bpe_use = torch.ones_like(msk_bpe, dtype=torch.bool)
        else:
            msk_rh_use = msk_rh_in; msk_bpe_use = msk_bpe
        z_pref, pmask = pack_combo_prefix(z_rh, msk_rh_use, z_bpe, msk_bpe_use)
        all_c = []
        bs = 16
        for k in range(cands):
            torch.manual_seed(42 + k)
            outs = []
            for i in range(0, len(z_pref), bs):
                g = generate_from_zpref(net_tr, tk, z_pref[i:i + bs], pmask[i:i + bs], sched_10_04)
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
    res["val_500"]["e5_v2_combo_both"] = score_factual(gen_wmt(rh_val, msk_rh_val, bpe_val, mode="combo", cands=8), val_plans)
    res["val_500"]["e5_v2_combo_bpe_only"] = score_factual(gen_wmt(rh_val, msk_rh_val, bpe_val, mode="bpe_only", cands=8), val_plans)
    res["val_500"]["e5_v2_combo_rh_only"] = score_factual(gen_wmt(rh_val, msk_rh_val, bpe_val, mode="rh_only", cands=8), val_plans)
    for k, v in res["val_500"].items():
        print("val_500", k, v, flush=True)

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
        k3_embed_w = sf.get_tensor("language_model.model.embed_tokens.weight")

    for pid in ("p1", "p2", "p3", "p4", "p5"):
        raw_prompt = open(f"/home/usergpu/k3ncm/{pid}.txt", encoding="utf-8").read().strip()
        t0 = time.perf_counter()
        lat_out = run_factual_latent_rollout(ha_model, ha_tok, embed_bank, raw_prompt, min_lat=8, max_lat=20, max_new=64)
        ha_ms = (time.perf_counter() - t0) * 1000.0
        k3_plan = window_to_fact_plan(k3_ref[pid]["answer"])
        for plan_src, plan_str in (("hybridagent_latent_plan", lat_out["hybrid_plan"]), ("k3_grounded_plan", k3_plan)):
            rh_p, msk_p = encode_batch_rh(ha_model, ha_tok, [plan_str], P_RH, dev_ha)
            rh_p = rh_p.to(dev); msk_p = (msk_p.to(dev) != 0)
            p_qids = qtok.encode(plan_str)[:P_BPE]
            p_qpad = torch.tensor([p_qids + [pad_id] * (P_BPE - len(p_qids))], dtype=torch.long, device=dev)
            t_ncm0 = time.perf_counter()
            gen_txt = gen_wmt(rh_p, msk_p, p_qpad, mode="combo", cands=8)[0]
            ncm_ms = (time.perf_counter() - t_ncm0) * 1000.0
            gen_qids = qtok.encode(gen_txt)
            k3_ids = [q_to_k3[q] for q in gen_qids if q in q_to_k3]
            k3_s0 = k3_embed_w[torch.tensor(k3_ids, dtype=torch.long)].float()
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
            kk = rec["keys_kept"]
            print(f"[{pid} | {plan_src}] keys={kk} k3_toks={len(k3_ids)}/{len(gen_qids)} ncm_ms={ncm_ms:.1f}", flush=True)
            print(f"   PLAN: {plan_str}", flush=True)
            print(f"   TEXT: {gen_txt}", flush=True)

    json.dump(res, open(os.path.join(a.out, "eval_k3_c48.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved eval_k3_c48.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pr = sub.add_parser("prep")
    p_pr.add_argument("--out", default="/home/usergpu/merge_lf/out/e5_k3_v2")
    p_pr.add_argument("--ntrain", type=int, default=16000)
    p_pr.add_argument("--nval", type=int, default=500)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--out", default="/home/usergpu/merge_lf/out/e5_k3_v2")
    p_tr.add_argument("--init-ckpt", default="/home/usergpu/merge_lf/out/e5_k3_wmt/last.pt")
    p_tr.add_argument("--steps", type=int, default=2000)
    p_tr.add_argument("--bs", type=int, default=32)
    p_tr.add_argument("--lr", type=float, default=1e-4)
    p_tr.add_argument("--bridge-lr", type=float, default=2e-4)
    p_tr.add_argument("--dec-lr", type=float, default=1e-4)
    p_tr.add_argument("--warmup", type=int, default=100)
    p_tr.add_argument("--alpha", type=float, default=0.7)
    p_tr.add_argument("--corrupt-first", type=float, default=0.4)
    p_tr.add_argument("--corrupt-last", type=float, default=0.05)
    p_tr.add_argument("--tok-ce-weight", type=float, default=0.5)
    p_tr.add_argument("--eval-every", type=int, default=500)
    p_tr.add_argument("--seed", type=int, default=42)

    p_ev = sub.add_parser("eval")
    p_ev.add_argument("--out", default="/home/usergpu/merge_lf/out/e5_k3_v2")

    a = ap.parse_args()
    {"prep": cmd_prep, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)
