"""Track 2B (Cycle 42): Fluency improvement for NCM stage-2 plan-to-text generation.
Initializes from e3_drop/last.pt and addresses the two diagnosed causes of the fluency gap:
1. Coarse/mid scaffold under-training: uses alpha=0.7 level weighting (levels 0..5 get 46% of code CE instead of 18.9%)
   and realistic conditioning corruption (0.40 -> 0.05 instead of 0.90 -> 0.10).
2. Decoder mismatch & token-level fluency: builds straight-through predicted z_hat from the generator's 12-level logits,
   passes z_hat_pred through tk.vqvae.decoder, and adds token-level cross-entropy loss (tok_ce_weight=0.5) against the
   ground-truth story tokens while fine-tuning tk.vqvae.decoder.middle_block + ln_f (keeping codebooks and output_layer frozen).
Saves both net.state_dict() and vq_decoder.state_dict() so generator-only and generator+decoder gains can be evaluated separately.
"""
import argparse, json, math, os, sys, time
from clover_paths import setup_clover_sys_path

ROOT = setup_clover_sys_path()
ART = os.environ.get("NCM_ART", ROOT)
if os.path.isdir(ROOT):
    os.chdir(ROOT)

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import ncm_stage2 as S


def build_st_zhat(vq, logits, levels):
    """Build differentiable straight-through z_hat [B, 128, 384] from generator logits [B, 641, 4096]."""
    B = logits.shape[0]
    L_full = levels[-1]
    z_hat = torch.zeros(B, L_full, vq.embed_dim, device=logits.device, dtype=logits.dtype)
    off = 0
    for k, l_k in enumerate(levels):
        V_k = vq.vqvae_vocab_sizes[k]
        lg_k = logits[:, off:off + l_k, :V_k].float()
        probs = F.softmax(lg_k, dim=-1)
        hard = F.one_hot(lg_k.argmax(dim=-1), num_classes=V_k).float()
        st = (hard + probs - probs.detach()).to(vq.embeddings[k].weight.dtype)
        emb_k = st @ vq.embeddings[k].weight  # [B, l_k, 384]
        if k != len(levels) - 1:
            z_q_k = vq._up_to_full(emb_k.float(), k, L_full)
        else:
            z_q_k = emb_k.float()
        z_q_k = vq._apply_quant_resi(z_q_k, k)
        z_hat = z_hat + z_q_k.to(z_hat.dtype)
        off += l_k
    return z_hat


def step_loss_fluency(fwd_net, tk, fwd_dec, plan_ids, idx, true_ids, pad_id, corrupt_rates, alpha, tok_ce_weight):
    vq = tk.vqvae.vq
    levels = list(vq.levels)
    with torch.no_grad():
        idx_in = S.corrupt(idx, levels, vq.vqvae_vocab_sizes, *corrupt_rates) if corrupt_rates else idx
        ncm_in = vq.idx_to_ncm_input(idx_in)
        pref = tk.vqvae.encoder.wte(plan_ids)
    logits = fwd_net(torch.cat([pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=(plan_ids == pad_id))
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
        total_loss = code_loss + tok_ce_weight * tok_ce
    else:
        tok_ce = torch.zeros((), device=idx.device)
        total_loss = code_loss

    with torch.no_grad():
        per_level = torch.stack([ce[:, lid == k].mean() for k in range(len(levels))])
    return total_loss, code_loss.detach(), tok_ce.detach(), per_level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/home/usergpu/plan2text/data/pairs_train.jsonl")
    ap.add_argument("--val-pairs", default="/home/usergpu/plan2text/data/pairs_valid.jsonl")
    ap.add_argument("--init", default="/home/usergpu/plan2text/out/e3_drop/last.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--dec-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--corrupt-first", type=float, default=0.4)
    ap.add_argument("--corrupt-last", type=float, default=0.05)
    ap.add_argument("--plan-dropout", type=float, default=0.1)
    ap.add_argument("--tok-ce-weight", type=float, default=0.5)
    ap.add_argument("--train-decoder", type=int, default=1)
    ap.add_argument("--nval", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
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
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CFG, tk, net, src, rep = S.load_models(dev, a.P, ckpt=a.init)
    pad_id = tk.pre_tokenizer.pad_token_id
    eos_id = tk.pre_tokenizer.eos_token_id

    # Optionally unfreeze VQ-VAE decoder middle_block + ln_f (keep output_layer frozen to preserve vocab embedding geometry)
    dec = tk.vqvae.decoder
    for name, p in dec.named_parameters():
        if a.train_decoder and ("middle_block" in name or "ln_f" in name):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    if main_proc:
        print("load report:", json.dumps(rep), flush=True)
        print("trainable net params (M):", sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6,
              "trainable dec params (M):", sum(p.numel() for p in dec.parameters() if p.requires_grad) / 1e6, flush=True)

    vrows = S.read_pairs(a.val_pairs)[:a.nval]
    trows = S.read_pairs(a.pairs)
    rows = vrows + trows
    nval = len(vrows)

    os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    idx_all = S.encode_stories(tk, [r["ids"] for r in rows], dev)
    plans_all = S.encode_plans(tk, [r["plan"] for r in rows], a.P, pad_id)
    true_ids_all = torch.tensor([r["ids"] for r in rows], dtype=torch.long)
    if main_proc:
        print(f"encoded {len(rows)} stories in {time.perf_counter()-t0:.1f}s; train {len(trows)} val {nval}", flush=True)

    idx_val = idx_all[:nval].long().to(dev)
    plan_val = plans_all[:nval].to(dev)
    tids_val = true_ids_all[:nval].to(dev)

    idx_tr = idx_all[nval:]
    plan_tr = plans_all[nval:]
    tids_tr = true_ids_all[nval:]

    fwd_net = DDP(net, device_ids=[dev.index]) if ws > 1 else net
    fwd_dec = DDP(dec, device_ids=[dev.index]) if (ws > 1 and a.train_decoder) else dec

    net_params = [p for p in net.parameters() if p.requires_grad]
    dec_params = [p for p in dec.parameters() if p.requires_grad]
    param_groups = [{"params": net_params, "base_lr": a.lr, "lr": a.lr}]
    if dec_params:
        param_groups.append({"params": dec_params, "base_lr": a.dec_lr, "lr": a.dec_lr})
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)
    all_params = net_params + dec_params

    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    log = open(os.path.join(a.out, "log.jsonl"), "a") if main_proc else open(os.devnull, "w")
    rates = (a.corrupt_first, a.corrupt_last)
    g = torch.Generator().manual_seed(a.seed + rank)
    torch.manual_seed(a.seed + rank)

    def evaluate():
        net.eval()
        dec.eval()
        res = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for name, pl in (("matched", plan_val), ("shuffled", torch.roll(plan_val, 1, 0))):
                tot, tot_tok, n = None, 0.0, 0
                for i in range(0, len(idx_val), a.bs):
                    b_pl = pl[i:i + a.bs]
                    b_idx = idx_val[i:i + a.bs]
                    b_tids = tids_val[i:i + a.bs]
                    _, c_loss, t_loss, pls = step_loss_fluency(net, tk, dec, b_pl, b_idx, b_tids, pad_id, None, 0.0, a.tok_ce_weight)
                    tot = pls * len(b_pl) if tot is None else tot + pls * len(b_pl)
                    tot_tok += float(t_loss.item()) * len(b_pl)
                    n += len(b_pl)
                res[name] = (tot / n).tolist()
                res[name + "_tok_ce"] = round(tot_tok / n, 4)
        net.train()
        if a.train_decoder:
            dec.train()
        return res

    if main_proc:
        ev = evaluate()
        print("step 0 val CE matched/shuffled (mean over levels):", round(sum(ev["matched"]) / 12, 4), round(sum(ev["shuffled"]) / 12, 4),
              "tok_ce:", ev["matched_tok_ce"], flush=True)
        log.write(json.dumps({"step": 0, "val": ev}) + "\n"); log.flush()
    if ws > 1:
        dist.barrier()

    net.train()
    if a.train_decoder:
        dec.train()
    t_start = time.perf_counter()
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = gp["base_lr"] * sched(step - 1)
        sel = torch.randint(0, len(idx_tr), (a.bs,), generator=g)
        idx = idx_tr[sel].long().to(dev)
        plan = plan_tr[sel].to(dev)
        tids = tids_tr[sel].to(dev)
        if a.plan_dropout > 0:
            drop = torch.rand(len(plan), generator=g) < a.plan_dropout
            if drop.any():
                null = torch.full((a.P,), pad_id)
                null[0] = eos_id
                plan[drop.to(dev)] = null.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, c_loss, t_loss, _ = step_loss_fluency(fwd_net, tk, fwd_dec, plan, idx, tids, pad_id, rates, a.alpha, a.tok_ce_weight)
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
                "gnorm": round(float(gn), 3),
                "lr": round(opt.param_groups[0]["lr"], 7),
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
                    "vq_decoder": dec.state_dict(),
                    "inference": src.get("inference"),
                    "step": step,
                    "args": vars(a),
                    "l0_head": net.l0_head_mode,
                }, tmp)
                os.replace(tmp, os.path.join(a.out, "last.pt"))
            if ws > 1:
                dist.barrier()
            net.train()
            if a.train_decoder:
                dec.train()
    if ws > 1:
        dist.destroy_process_group()
    if main_proc:
        print("done train_fluency", flush=True)


if __name__ == "__main__":
    main()
