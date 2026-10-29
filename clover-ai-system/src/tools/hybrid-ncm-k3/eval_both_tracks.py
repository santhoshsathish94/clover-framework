"""Evaluate Track 2B (e4_fluency) and Track 2A (e4_latent + hybrid_plan) against e3_drop baseline on all 598 validation pairs
and 48 end-to-end requests. Uses the exact metrics of quality_eval.py (retention, GPT-2-large Gen. PPL, adj_repeat_rate,
repeat_trigram_rate, entropy) and exports the 20 blind-rating stories for the 3 model raters.
"""
import argparse, json, math, os, re, sys, time
from clover_paths import setup_clover_sys_path

ROOT = setup_clover_sys_path()
ART = os.environ.get("NCM_ART", ROOT)
if os.path.isdir(ROOT):
    os.chdir(ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation.gen_ppl import generation_perplexity

import ncm_stage2 as S
from train_latent_bridge import LatentBridge, open_shm_tensors, SHM_NULL_RH, P_LEN, D_HA


def words(t):
    return re.findall(r"[a-z']+", t.lower())


def jacc(x, y):
    sx, sy = set(words(x)), set(words(y))
    return len(sx & sy) / max(len(sx | sy), 1)


def mbr(cands):
    K = len(cands)
    best = []
    for i in range(len(cands[0])):
        sc = [sum(jacc(cands[k][i], cands[j][i]) for j in range(K) if j != k) for k in range(K)]
        best.append(cands[max(range(K), key=lambda k: sc[k])][i])
    return best


def pick_tokens(logits, adj=(10.0, 8, 0.8), ng=(3, 8.0)):
    if not adj and not ng:
        return logits.argmax(-1)
    al, aw, ad = (adj[0], int(adj[1]), adj[2]) if adj else (0.0, 0, 1.0)
    B, T, V = logits.shape
    lg = logits.double()
    out = torch.empty(B, T, dtype=torch.long, device=lg.device)
    hist = []
    toks = [[] for _ in range(B)]
    seen = [dict() for _ in range(B)]
    for t in range(T):
        step = lg[:, t, :].clone()
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64, device=lg.device))
        if ng and t >= ng[0] - 1:
            for b in range(B):
                nxt = seen[b].get(tuple(toks[b][-(ng[0] - 1):]))
                if nxt:
                    step[b, list(nxt)] -= ng[1]
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        if ng:
            for b, v in enumerate(p.tolist()):
                toks[b].append(v)
                if len(toks[b]) >= ng[0]:
                    seen[b].setdefault(tuple(toks[b][-ng[0]:-1]), set()).add(v)
    return out


@torch.no_grad()
def generate_from_zpref(net, tk, z_prefix, pad_mask_bool, temps_per_level, top_k=0, top_p=1.0):
    """Generate 12 levels of codes given z_prefix [B, P, 384] and pad_mask_bool [B, P]."""
    dev = z_prefix.device
    B, P, _ = z_prefix.shape
    vq = tk.vqvae.vq
    levels = list(vq.levels)
    z_hat = torch.zeros(B, levels[-1], tk.vqvae.embed_dim, device=dev)

    prefix_emb = net.transformer.wte(z_prefix)
    if not net.use_rope:
        prefix_emb = prefix_emb + net.wpe(net.wp_prefix)
    level_pos_embed = net.transformer.wle(net.lvl_1L) if net.config.use_level_encoding else torch.zeros(1, net.L - levels[0], net.config.n_embd, device=dev)
    if not net.use_rope:
        level_pos_embed = level_pos_embed + net.wpe(net.wp_1L)

    _pbias = torch.zeros(B, 1, 1, P, device=dev)
    _pbias[:, :, :, :P] = torch.where(pad_mask_bool[:, None, None, :], torch.tensor(float("-inf"), device=dev), torch.tensor(0.0, device=dev))

    rope_p = (net.rope_cos_gen_0, net.rope_sin_gen_0) if net.use_rope else None
    h = prefix_emb
    pfx_mask = torch.zeros(1, 1, P, P, device=dev) + _pbias
    for block in net.transformer.blocks:
        h = block(h, attn_mask=pfx_mask, rope_cos_sin=rope_p)
    h = net.transformer.ln_f(h)
    l0_logits = net.lm_head_l0(net._level0_slice(h, pad_mask_bool))
    idx = net._sample_level_raw(l0_logits, temps_per_level[0], top_k, top_p)

    generated_idx = [idx]
    idx_next = idx
    level_acum = 0
    for level_idx in range(len(levels) - 1):
        l_next = levels[level_idx + 1]
        z_q_level = vq.embeddings[level_idx](idx_next)
        z_hat, x = vq.get_next_autoregressive_input(level_idx, z_hat, z_q_level)
        x = net.transformer.wte(x) + level_pos_embed[:, level_acum:level_acum + l_next, :]
        logits = net._decode_level_logits(x, prefix_emb, level_idx, _pbias, hidx=level_idx)
        idx_next = net._sample_level_raw(logits, temps_per_level[level_idx + 1], top_k, top_p)
        generated_idx.append(idx_next)
        level_acum += l_next
    return torch.cat(generated_idx, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-pairs", default="/home/usergpu/plan2text/data/pairs_valid.jsonl")
    ap.add_argument("--e3-ckpt", default="/home/usergpu/plan2text/out/e3_drop/last.pt")
    ap.add_argument("--e4f-ckpt", default="/home/usergpu/merge_lf/out/e4_fluency/last.pt")
    ap.add_argument("--e4l-ckpt", default="/home/usergpu/merge_lf/out/e4_latent/last.pt")
    ap.add_argument("--out", default="/home/usergpu/merge_lf/out/eval_summary.json")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--cands", type=int, default=8)
    a = ap.parse_args()

    dev = torch.device("cuda:0")
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
            if len(ids) < 2:
                continue
            c = {}
            for i in ids:
                c[i] = c.get(i, 0) + 1
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

    vrows = S.read_pairs(a.val_pairs)[:598]
    val_plans = [r["plan"] for r in vrows]

    # Load e3_drop (P=64)
    CFG, tk, net_e3, src_e3, _ = S.load_models(dev, 64, ckpt=a.e3_ckpt)
    net_e3.eval()
    orig_dec_sd = {k: v.clone() for k, v in tk.vqvae.decoder.state_dict().items()}
    pad_id, eos = tk.pre_tokenizer.pad_token_id, tk.pre_tokenizer.eos_token_id

    def dec(r):
        r = list(r)
        if eos in r:
            r = r[:r.index(eos)]
        return tk.pre_tokenizer.decode([t for t in r if t not in (eos, pad_id)], skip_special_tokens=True)

    true_texts = [dec(r["ids"]) for r in vrows]
    summary = {"val_598": {}, "e2e_48": {}, "rating_stories_20": {}}
    summary["val_598"]["true"] = score_all(true_texts, val_plans)
    print("val_598 true:", summary["val_598"]["true"], flush=True)

    @torch.no_grad()
    def run_bpe_model(net_m, plans_list, P_val, temps_sched, cands=8):
        pids = S.encode_plans(tk, plans_list, P_val, pad_id).to(dev)
        z_pref = tk.vqvae.encoder.wte(pids)
        pmask = (pids == pad_id)
        all_c = []
        for k in range(cands):
            torch.manual_seed(42 + k)
            outs = []
            for i in range(0, len(pids), a.bs):
                g = generate_from_zpref(net_m, tk, z_pref[i:i + a.bs], pmask[i:i + a.bs], temps_sched)
                z_hats, _ = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
                lg = tk.vqvae.decoder(z_hats[-1])
                outs += [dec(r) for r in pick_tokens(lg).tolist()]
            all_c.append(outs)
        return mbr(all_c) if cands > 1 else all_c[0]

    sched_const08 = [0.8] * 12
    sched_10_04 = [1.0 - 0.6 * (k / 11.0) for k in range(12)]

    # 1. e3_drop baseline + e3_drop with sched_10_04
    t_e3_bal = run_bpe_model(net_e3, val_plans, 64, sched_const08, a.cands)
    summary["val_598"]["e3_drop_bal"] = score_all(t_e3_bal, val_plans)
    print("val_598 e3_drop_bal:", summary["val_598"]["e3_drop_bal"], flush=True)

    t_e3_sched = run_bpe_model(net_e3, val_plans, 64, sched_10_04, a.cands)
    summary["val_598"]["e3_drop_sched10_04"] = score_all(t_e3_sched, val_plans)
    print("val_598 e3_drop_sched10_04:", summary["val_598"]["e3_drop_sched10_04"], flush=True)

    # 2. e4_fluency (P=64): with orig decoder vs fine-tuned decoder vs sched
    _, _, net_e4f, src_e4f, _ = S.load_models(dev, 64, ckpt=a.e4f_ckpt)
    net_e4f.eval()
    tk.vqvae.decoder.load_state_dict(orig_dec_sd)
    t_e4f_origdec = run_bpe_model(net_e4f, val_plans, 64, sched_const08, a.cands)
    summary["val_598"]["e4_fluency_origdec"] = score_all(t_e4f_origdec, val_plans)
    print("val_598 e4_fluency_origdec:", summary["val_598"]["e4_fluency_origdec"], flush=True)

    if "vq_decoder" in src_e4f:
        tk.vqvae.decoder.load_state_dict(src_e4f["vq_decoder"])
    t_e4f_ftdec = run_bpe_model(net_e4f, val_plans, 64, sched_const08, a.cands)
    summary["val_598"]["e4_fluency_ftdec"] = score_all(t_e4f_ftdec, val_plans)
    print("val_598 e4_fluency_ftdec:", summary["val_598"]["e4_fluency_ftdec"], flush=True)

    t_e4f_sched = run_bpe_model(net_e4f, val_plans, 64, sched_10_04, a.cands)
    summary["val_598"]["e4_fluency_ftdec_sched"] = score_all(t_e4f_sched, val_plans)
    print("val_598 e4_fluency_ftdec_sched:", summary["val_598"]["e4_fluency_ftdec_sched"], flush=True)

    # 3. e4_latent (P=48, continuous R(h) -> LatentBridge)
    _, _, net_e4l, src_e4l, _ = S.load_models(dev, P_LEN, ckpt=a.e4l_ckpt)
    net_e4l.eval()
    bridge = LatentBridge(D_HA, tk.vqvae.embed_dim).to(dev)
    bridge.load_state_dict(src_e4l["latent_bridge"])
    bridge.eval()
    if "vq_decoder" in src_e4l:
        tk.vqvae.decoder.load_state_dict(src_e4l["vq_decoder"])

    _, _, val_rh, val_msk = open_shm_tensors(create=False)
    rh_v = val_rh[:598].to(dev)
    msk_v = (val_msk[:598].to(dev) != 0)

    @torch.no_grad()
    def run_latent_model(rh_in, msk_in, temps_sched, cands=8):
        z_pref = bridge(rh_in).float()
        all_c = []
        for k in range(cands):
            torch.manual_seed(42 + k)
            outs = []
            for i in range(0, len(rh_in), a.bs):
                g = generate_from_zpref(net_e4l, tk, z_pref[i:i + a.bs], msk_in[i:i + a.bs], temps_sched)
                z_hats, _ = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
                lg = tk.vqvae.decoder(z_hats[-1])
                outs += [dec(r) for r in pick_tokens(lg).tolist()]
            all_c.append(outs)
        return mbr(all_c) if cands > 1 else all_c[0]

    t_e4l_bal = run_latent_model(rh_v, msk_v, sched_const08, a.cands)
    summary["val_598"]["e4_latent_rh_bal"] = score_all(t_e4l_bal, val_plans)
    print("val_598 e4_latent_rh_bal:", summary["val_598"]["e4_latent_rh_bal"], flush=True)

    t_e4l_sched = run_latent_model(rh_v, msk_v, sched_10_04, a.cands)
    summary["val_598"]["e4_latent_rh_sched"] = score_all(t_e4l_sched, val_plans)
    print("val_598 e4_latent_rh_sched:", summary["val_598"]["e4_latent_rh_sched"], flush=True)

    # 4. E2E 48 topics (both text_plan and <latent>-rollout hybrid_plan!)
    all_tops = []
    for s in range(8):
        p = f"/home/usergpu/merge_lf/top_shard_{s}.pt"
        if os.path.exists(p):
            all_tops.extend(torch.load(p, map_location="cpu", weights_only=True))
    all_tops.sort(key=lambda x: x["idx"])
    if len(all_tops) == 48:
        text_plans_48 = [x["text_plan"] for x in all_tops]
        hybrid_plans_48 = [x["hybrid_plan"] for x in all_tops]
        rh_text_48 = torch.stack([x["rh_text"] for x in all_tops]).to(dev)
        msk_text_48 = (torch.stack([x["msk_text"] for x in all_tops]).to(dev) != 0)
        rh_hyb_48 = torch.stack([x["rh_hybrid"] for x in all_tops]).to(dev)
        msk_hyb_48 = (torch.stack([x["msk_hybrid"] for x in all_tops]).to(dev) != 0)

        summary["e2e_48"]["latent_rollout_stats"] = {
            "mean_latent_steps": round(sum(x["latent_steps"] for x in all_tops) / 48.0, 2),
            "anchor_exits": sum(1 for x in all_tops if x["exit_reason"] == "anchor"),
            "mean_readout_cos": round(sum(x["mean_readout_cos"] for x in all_tops) / 48.0, 4),
        }

        tk.vqvae.decoder.load_state_dict(orig_dec_sd)
        e3_tp = run_bpe_model(net_e3, text_plans_48, 64, sched_const08, a.cands)
        e3_hp = run_bpe_model(net_e3, hybrid_plans_48, 64, sched_const08, a.cands)
        summary["e2e_48"]["e3_drop_textplan"] = score_all(e3_tp, text_plans_48)
        summary["e2e_48"]["e3_drop_hybridplan"] = score_all(e3_hp, hybrid_plans_48)

        if "vq_decoder" in src_e4f:
            tk.vqvae.decoder.load_state_dict(src_e4f["vq_decoder"])
        e4f_tp = run_bpe_model(net_e4f, text_plans_48, 64, sched_10_04, a.cands)
        e4f_hp = run_bpe_model(net_e4f, hybrid_plans_48, 64, sched_10_04, a.cands)
        summary["e2e_48"]["e4_fluency_textplan"] = score_all(e4f_tp, text_plans_48)
        summary["e2e_48"]["e4_fluency_hybridplan"] = score_all(e4f_hp, hybrid_plans_48)

        if "vq_decoder" in src_e4l:
            tk.vqvae.decoder.load_state_dict(src_e4l["vq_decoder"])
        e4l_tp = run_latent_model(rh_text_48, msk_text_48, sched_10_04, a.cands)
        e4l_hp = run_latent_model(rh_hyb_48, msk_hyb_48, sched_10_04, a.cands)
        summary["e2e_48"]["e4_latent_rh_textplan"] = score_all(e4l_tp, text_plans_48)
        summary["e2e_48"]["e4_latent_rh_hybridplan"] = score_all(e4l_hp, hybrid_plans_48)
        for k, v in summary["e2e_48"].items():
            print("e2e_48", k, v, flush=True)

    # Save stories for all 598 val plans for downstream judge / model-rater evaluation
    summary["stories_val_598"] = {
        "plans": val_plans,
        "true": true_texts,
        "e3_drop_bal": t_e3_bal,
        "e4_fluency_ftdec": t_e4f_ftdec,
        "e4_fluency_ftdec_sched": t_e4f_sched,
        "e4_latent_rh_bal": t_e4l_bal,
        "e4_latent_rh_sched": t_e4l_sched,
    }
    json.dump(summary, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
