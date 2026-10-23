"""Track 2A (Cycle 43): Continuous HybridLink R(h) Latent-Mode Merge with NCM Stage-2.
Subcommands:
  prep    : downloads adapter.pt if needed and allocates shared /dev/shm buffers.
  extract : runs on shard s of 8 GPUs; extracts R(h) = model.run_hybridlink(last_hidden) [N, P=48, 4096]
            for 32,768 training pairs and 598 validation pairs, AND runs continuous <latent> rollouts
            on validation stories + 48 e2e topics.
  train   : DDP training (on 4 GPUs) of LatentBridge (4096 -> 384) + NextConceptTransformer + VQ-VAE decoder
            conditioned directly on continuous R(h) vectors instead of discrete BPE token IDs.
"""
import argparse, json, math, os, re, sys, time

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
ROOT = os.environ.get("NCM_REPO", "/home/usergpu/ncm")
ART = os.environ.get("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, "/home/usergpu/plan2text")
import ncm_stage2 as S
from train_fluency import build_st_zhat

N_TRAIN = 32768
N_VAL = 598
P_LEN = 48
D_HA = 4096

SHM_TR_RH = "/dev/shm/rh_train.bf16"
SHM_TR_MSK = "/dev/shm/rh_train.msk"
SHM_VAL_RH = "/dev/shm/rh_val.bf16"
SHM_VAL_MSK = "/dev/shm/rh_val.msk"
SHM_NULL_RH = "/dev/shm/rh_null.pt"


class LatentBridge(nn.Module):
    """Maps continuous HybridLink R(h) vectors [B, P, 4096] into NCM's z_prefix space [B, P, 384]."""
    def __init__(self, d_in=4096, d_out=384):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_out * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_out * 2, d_out)
        self.ln_out = nn.LayerNorm(d_out)
        self.scale = nn.Parameter(torch.tensor(8.5))

    def forward(self, r_h):
        x = self.ln_in(r_h.float())
        x = self.fc2(self.act(self.fc1(x)))
        x = self.ln_out(x) * self.scale
        return x.to(r_h.dtype)


def open_shm_tensors(create=False):
    tr_rh = torch.from_file(SHM_TR_RH, shared=True, size=N_TRAIN * P_LEN * D_HA, dtype=torch.bfloat16).view(N_TRAIN, P_LEN, D_HA)
    tr_msk = torch.from_file(SHM_TR_MSK, shared=True, size=N_TRAIN * P_LEN, dtype=torch.uint8).view(N_TRAIN, P_LEN)
    val_rh = torch.from_file(SHM_VAL_RH, shared=True, size=N_VAL * P_LEN * D_HA, dtype=torch.bfloat16).view(N_VAL, P_LEN, D_HA)
    val_msk = torch.from_file(SHM_VAL_MSK, shared=True, size=N_VAL * P_LEN, dtype=torch.uint8).view(N_VAL, P_LEN)
    return tr_rh, tr_msk, val_rh, val_msk


def cmd_prep(a):
    from huggingface_hub import snapshot_download
    p = snapshot_download("togethercomputer/HybridAgent-8B-Agentic")
    print("snapshot ready:", p, flush=True)
    open_shm_tensors(create=True)
    print("allocated /dev/shm buffers", flush=True)


def clean_plan(raw):
    plan = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    plan = re.sub(r"</?think>", "", plan).strip().split("\n")[0].strip()
    plan = re.sub(r"^Plan:\s*", "", plan)
    plan = re.sub(r"(?:This plan|The plan|Note:).*$", "", plan).strip()
    w = plan.split()
    if len(w) > 40:
        plan = " ".join(w[:40])
        if ";" in plan[:-1]:
            plan = plan[:plan.rindex(";")].strip()
    return plan


@torch.no_grad()
def encode_batch_rh(model, tok, texts, P, dev):
    """Right-padded tokenization in Qwen3, backbone forward, then R(h) = model.run_hybridlink(last_hidden)."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids_list = []
    msk_list = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=True)[:P]
        n = len(ids)
        ids_list.append(ids + [pad_id] * (P - n))
        msk_list.append([0] * n + [1] * (P - n))
    inp = torch.tensor(ids_list, dtype=torch.long, device=dev)
    pad_mask = torch.tensor(msk_list, dtype=torch.uint8, device=dev)
    attn_mask = (pad_mask == 0).long()
    out = model.backbone.model(input_ids=inp, attention_mask=attn_mask, use_cache=False, return_dict=True)
    last_h = out.last_hidden_state
    r_h = model.run_hybridlink(last_h, output_dtype=torch.bfloat16)
    return r_h.cpu(), pad_mask.cpu()


PROMPT_TOPIC = (
    "Plan a short children's story about: {topic}. Write the plan in one line: the characters (with names), the setting, "
    "and the events in order, as short phrases separated by semicolons. Use at most 35 words. "
    "Use short common first names (Tim, Lily, Sam, Anna, Ben, Mia, Max) for every person and animal, and plain everyday words: "
    "no abbreviations, digits or unusual names. Output only the plan line.\n\n"
    "Example request: a girl who helps her mom\n"
    "Example plan: thoughtful girl Sue; at home with her mom; Sue loves helping; mom asks her to wipe the table after lunch; Sue is happy to help\n\n"
    "Request: {topic}"
)


@torch.no_grad()
def run_latent_rollout_one(model, tokenizer, embed_bank, user_content, P=48, min_lat=8, max_lat=24, max_new=56):
    from hybrid_agent.constants import ENTRY_ANCHOR, EXIT_ANCHOR, THINK_OPEN, THINK_CLOSE, special_token_id
    from hybrid_agent.readout import pseudo_text
    dev = next(model.parameters()).device
    backbone = model.backbone
    embed_w = backbone.get_input_embeddings().weight
    lat_close_id = special_token_id(tokenizer, EXIT_ANCHOR)

    msgs = [{"role": "user", "content": user_content}]
    base_chat = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if base_chat.rstrip("\n").endswith(THINK_OPEN):
        prefix_str = base_chat + "Let me list the characters, setting, and events in order." + ENTRY_ANCHOR
    else:
        prefix_str = base_chat + THINK_OPEN + "\nLet me list the characters, setting, and events in order." + ENTRY_ANCHOR

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

    bridge_str = EXIT_ANCHOR + THINK_CLOSE + "\nPlan:"
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

    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "latent_steps": len(latent_vecs),
        "exit_reason": exit_reason,
        "mean_readout_cos": round(sum(readout_cos) / max(len(readout_cos), 1), 4),
        "pseudo_text": pseudo_text(readout_ids, tokenizer),
        "hybrid_plan": clean_plan(raw_text),
    }


def cmd_extract(a):
    from hybrid_agent.model import load_hybrid_agent
    from hybrid_agent.readout import normalized_embeddings
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

    # 1. Validation R(h)
    v_idx = list(range(a.shard, len(vrows), a.nshards))
    for i in range(0, len(v_idx), a.bs):
        b_ids = v_idx[i:i + a.bs]
        texts = [vrows[j]["plan"] for j in b_ids]
        rh, msk = encode_batch_rh(model, tok, texts, P_LEN, dev)
        for k, j in enumerate(b_ids):
            val_rh[j].copy_(rh[k])
            val_msk[j].copy_(msk[k])

    # 2. Train R(h)
    t_idx = list(range(a.shard, len(trows), a.nshards))
    for i in range(0, len(t_idx), a.bs):
        b_ids = t_idx[i:i + a.bs]
        texts = [trows[j]["plan"] for j in b_ids]
        rh, msk = encode_batch_rh(model, tok, texts, P_LEN, dev)
        for k, j in enumerate(b_ids):
            tr_rh[j].copy_(rh[k])
            tr_msk[j].copy_(msk[k])
    print(f"[shard {a.shard}] extracted {len(t_idx)} train + {len(v_idx)} val R(h) in {time.perf_counter()-t0:.1f}s", flush=True)

    # 3. Also run true <latent> rollout on the 48 e2e topics (6 topics per GPU shard!)
    top_rows = [json.loads(l) for l in open(a.topics, encoding="utf-8") if l.strip()]
    embed_bank = normalized_embeddings(model.backbone.get_input_embeddings().weight).float()
    my_tops = []
    for j in range(a.shard, len(top_rows), a.nshards):
        r = top_rows[j]
        out = run_latent_rollout_one(model, tok, embed_bank, PROMPT_TOPIC.format(topic=r["topic"]))
        out["idx"] = j
        out["topic"] = r["topic"]
        out["text_plan"] = r["plan"]
        # Also extract R(h) of the resulting hybrid_plan
        rh_hp, msk_hp = encode_batch_rh(model, tok, [out["hybrid_plan"]], P_LEN, dev)
        rh_tp, msk_tp = encode_batch_rh(model, tok, [r["plan"]], P_LEN, dev)
        out["rh_hybrid"] = rh_hp[0]
        out["msk_hybrid"] = msk_hp[0]
        out["rh_text"] = rh_tp[0]
        out["msk_text"] = msk_tp[0]
        my_tops.append(out)
    torch.save(my_tops, f"/home/usergpu/merge_lf/top_shard_{a.shard}.pt")
    print(f"[shard {a.shard}] finished {len(my_tops)} topic latent rollouts in {time.perf_counter()-t0:.1f}s", flush=True)


def step_loss_latent(fwd_net, fwd_bridge, tk, fwd_dec, rh, pad_msk, bpe_plan_ids, idx, true_ids, corrupt_rates, alpha, tok_ce_weight, align_weight):
    vq = tk.vqvae.vq
    levels = list(vq.levels)
    with torch.no_grad():
        idx_in = S.corrupt(idx, levels, vq.vqvae_vocab_sizes, *corrupt_rates) if corrupt_rates else idx
        ncm_in = vq.idx_to_ncm_input(idx_in)
        bpe_pref = tk.vqvae.encoder.wte(bpe_plan_ids)
    z_pref = fwd_bridge(rh)
    logits = fwd_net(torch.cat([z_pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=(pad_msk != 0))
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

    if align_weight > 0.0:
        valid_f = (pad_msk == 0).float().unsqueeze(-1)
        mean_z = (z_pref.float() * valid_f).sum(1) / valid_f.sum(1).clamp(min=1.0)
        mean_bpe = bpe_pref.float().mean(1)
        align_loss = F.mse_loss(mean_z, mean_bpe)
    else:
        align_loss = torch.zeros((), device=idx.device)

    total_loss = code_loss + tok_ce_weight * tok_ce + align_weight * align_loss
    with torch.no_grad():
        per_level = torch.stack([ce[:, lid == k].mean() for k in range(len(levels))])
    return total_loss, code_loss.detach(), tok_ce.detach(), align_loss.detach(), per_level


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

    CFG, tk, net, src, rep = S.load_models(dev, P_LEN, ckpt=a.init)
    pad_id = tk.pre_tokenizer.pad_token_id
    bridge = LatentBridge(D_HA, tk.vqvae.embed_dim).to(dev)

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
    plans_all = S.encode_plans(tk, [r["plan"] for r in rows], P_LEN, pad_id)
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
    fwd_bridge = DDP(bridge, device_ids=[dev.index]) if ws > 1 else bridge
    fwd_dec = DDP(dec, device_ids=[dev.index]) if ws > 1 else dec

    net_params = [p for p in net.parameters() if p.requires_grad]
    bridge_params = [p for p in bridge.parameters() if p.requires_grad]
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

    def evaluate():
        net.eval(); bridge.eval(); dec.eval()
        res = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for name, r_in, m_in in (("matched", rh_val, msk_val), ("shuffled", torch.roll(rh_val, 1, 0), torch.roll(msk_val, 1, 0))):
                tot, tot_tok, n = None, 0.0, 0
                for i in range(0, len(idx_val), a.bs):
                    _, c_loss, t_loss, _, pls = step_loss_latent(
                        net, bridge, tk, dec,
                        r_in[i:i + a.bs], m_in[i:i + a.bs], bpe_val[i:i + a.bs],
                        idx_val[i:i + a.bs], tids_val[i:i + a.bs],
                        None, 0.0, a.tok_ce_weight, 0.0
                    )
                    tot = pls * len(r_in[i:i + a.bs]) if tot is None else tot + pls * len(r_in[i:i + a.bs])
                    tot_tok += float(t_loss.item()) * len(r_in[i:i + a.bs])
                    n += len(r_in[i:i + a.bs])
                res[name] = (tot / n).tolist()
                res[name + "_tok_ce"] = round(tot_tok / n, 4)
        net.train(); bridge.train(); dec.train()
        return res

    if main_proc:
        ev = evaluate()
        print("step 0 val CE matched/shuffled:", round(sum(ev["matched"]) / 12, 4), round(sum(ev["shuffled"]) / 12, 4), "tok_ce:", ev["matched_tok_ce"], flush=True)
        log.write(json.dumps({"step": 0, "val": ev}) + "\n"); log.flush()
    if ws > 1:
        dist.barrier()

    net.train(); bridge.train(); dec.train()
    t_start = time.perf_counter()
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = gp["base_lr"] * sched(step - 1)
        sel = torch.randint(0, N_TRAIN, (a.bs,), generator=g)
        idx = idx_tr[sel].long().to(dev)
        bpe_p = bpe_tr[sel].to(dev)
        tids = tids_tr[sel].to(dev)
        rh = tr_rh[sel].to(dev)
        msk = tr_msk[sel].to(dev)

        if a.plan_dropout > 0:
            drop = torch.rand(len(rh), generator=g) < a.plan_dropout
            if drop.any():
                d_dev = drop.to(dev)
                rh[d_dev] = null_rh
                msk[d_dev] = null_msk

        align_w = max(0.0, 0.2 * (1.0 - step / 500.0))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, c_loss, t_loss, a_loss, _ = step_loss_latent(
                fwd_net, fwd_bridge, tk, fwd_dec,
                rh, msk, bpe_p, idx, tids,
                rates, a.alpha, a.tok_ce_weight, align_w
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()

        if main_proc and (step % a.log_every == 0 or step == 1):
            rec = {
                "step": step,
                "loss": round(loss.item(), 4),
                "code_ce": round(c_loss.item(), 4),
                "tok_ce": round(t_loss.item(), 4),
                "align": round(a_loss.item(), 4),
                "gnorm": round(float(gn), 3),
                "s_per_step": round((time.perf_counter() - t_start) / step, 3),
            }
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()

        if step % a.eval_every == 0 or step == a.steps:
            if main_proc:
                ev = evaluate()
                print(f"step {step} val CE matched/shuffled:", round(sum(ev["matched"]) / 12, 4), round(sum(ev["shuffled"]) / 12, 4),
                      "coarse(0..5):", round(sum(ev["matched"][:6]) / 6, 4), "tok_ce:", ev["matched_tok_ce"], flush=True)
                log.write(json.dumps({"step": step, "val": ev}) + "\n"); log.flush()
                tmp = os.path.join(a.out, "last.pt.tmp")
                torch.save({
                    "model": net.state_dict(),
                    "latent_bridge": bridge.state_dict(),
                    "vq_decoder": dec.state_dict(),
                    "inference": src.get("inference"),
                    "step": step,
                    "args": vars(a),
                    "l0_head": net.l0_head_mode,
                }, tmp)
                os.replace(tmp, os.path.join(a.out, "last.pt"))
            if ws > 1:
                dist.barrier()
            net.train(); bridge.train(); dec.train()

    if ws > 1:
        dist.destroy_process_group()
    if main_proc:
        print("done train_latent_bridge", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    ex = sub.add_parser("extract")
    ex.add_argument("--shard", type=int, required=True)
    ex.add_argument("--nshards", type=int, default=8)
    ex.add_argument("--bs", type=int, default=128)
    ex.add_argument("--pairs", default="/home/usergpu/plan2text/data/pairs_train.jsonl")
    ex.add_argument("--val-pairs", default="/home/usergpu/plan2text/data/pairs_valid.jsonl")
    ex.add_argument("--topics", default="/home/usergpu/plan2text/out/e2e2/plans_v3.jsonl")
    tr = sub.add_parser("train")
    tr.add_argument("--pairs", default="/home/usergpu/plan2text/data/pairs_train.jsonl")
    tr.add_argument("--val-pairs", default="/home/usergpu/plan2text/data/pairs_valid.jsonl")
    tr.add_argument("--init", default="/home/usergpu/plan2text/out/e3_drop/last.pt")
    tr.add_argument("--out", required=True)
    tr.add_argument("--steps", type=int, default=2000)
    tr.add_argument("--bs", type=int, default=32)
    tr.add_argument("--lr", type=float, default=6e-5)
    tr.add_argument("--bridge-lr", type=float, default=5e-4)
    tr.add_argument("--dec-lr", type=float, default=1e-4)
    tr.add_argument("--warmup", type=int, default=150)
    tr.add_argument("--alpha", type=float, default=0.7)
    tr.add_argument("--corrupt-first", type=float, default=0.4)
    tr.add_argument("--corrupt-last", type=float, default=0.05)
    tr.add_argument("--plan-dropout", type=float, default=0.1)
    tr.add_argument("--tok-ce-weight", type=float, default=0.5)
    tr.add_argument("--nval", type=int, default=512)
    tr.add_argument("--eval-every", type=int, default=500)
    tr.add_argument("--log-every", type=int, default=50)
    tr.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    {"prep": cmd_prep, "extract": cmd_extract, "train": cmd_train}[a.cmd](a)
