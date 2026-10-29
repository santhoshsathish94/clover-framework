"""Evaluate CausalVQDecoder on top of both Track 2B (e4_fluency) and Track 2A (e4_latent) across all 598 validation pairs
and 48 e2e topics, and save subset20_cdec.json for the 20 blind rating plans.
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
from eval_both_tracks import words, jacc, mbr, generate_from_zpref
from train_latent_bridge import LatentBridge, encode_batch_rh, P_LEN, D_HA
from train_causal_decoder import CausalVQDecoder


@torch.no_grad()
def idx_to_zhat_upto(vq, idx, max_lvl=12):
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
    return z_hat


def main():
    dev = torch.device("cuda:0")
    vrows = S.read_pairs("/home/usergpu/plan2text/data/pairs_valid.jsonl")[:598]
    val_plans = [r["plan"] for r in vrows]

    # 1. Extract R(h) for 598 val plans (takes ~1.5s on cuda:0) then free HybridAgentModel
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

    CFG, tk, net_e4f, _, _ = S.load_models(dev, 64, ckpt="/home/usergpu/merge_lf/out/e4_fluency/last.pt")
    net_e4f.eval()
    pad_id, eos = tk.pre_tokenizer.pad_token_id, tk.pre_tokenizer.eos_token_id

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
    def gen_with_cdec(net_m, z_pref, pmask, max_lvl=12, cdec_temp=0.0, cands=8):
        all_c = []
        for k in range(cands):
            torch.manual_seed(42 + k)
            outs = []
            for i in range(0, len(z_pref), bs):
                g = generate_from_zpref(net_m, tk, z_pref[i:i + bs], pmask[i:i + bs], sched_10_04)
                zh = idx_to_zhat_upto(tk.vqvae.vq, g, max_lvl=max_lvl)
                t_ids = cdec.generate_tokens(zh, eos, temp=cdec_temp, adj=(4.0, 6, 0.7), ng=(3, 6.0))
                outs += [dec(r) for r in t_ids.tolist()]
            all_c.append(outs)
        return mbr(all_c) if cands > 1 else all_c[0]

    pids_val = S.encode_plans(tk, val_plans, 64, pad_id).to(dev)
    z_pref_bpe = tk.vqvae.encoder.wte(pids_val)
    pmask_bpe = (pids_val == pad_id)

    res = {"val_598": {}, "e2e_48": {}}

    t_e4f_cdec_t0 = gen_with_cdec(net_e4f, z_pref_bpe, pmask_bpe, max_lvl=12, cdec_temp=0.0, cands=8)
    res["val_598"]["e4_fluency_cdec_t0"] = score_all(t_e4f_cdec_t0, val_plans)
    print("val_598 e4_fluency_cdec_t0:", res["val_598"]["e4_fluency_cdec_t0"], flush=True)

    t_e4f_cdec_l9 = gen_with_cdec(net_e4f, z_pref_bpe, pmask_bpe, max_lvl=9, cdec_temp=0.0, cands=8)
    res["val_598"]["e4_fluency_cdec_lvl9"] = score_all(t_e4f_cdec_l9, val_plans)
    print("val_598 e4_fluency_cdec_lvl9:", res["val_598"]["e4_fluency_cdec_lvl9"], flush=True)

    # Track 2A: e4_latent + CausalVQDecoder
    _, _, net_e4l, src_e4l, _ = S.load_models(dev, P_LEN, ckpt="/home/usergpu/merge_lf/out/e4_latent/last.pt")
    net_e4l.eval()
    bridge = LatentBridge(D_HA, tk.vqvae.embed_dim).to(dev)
    bridge.load_state_dict(src_e4l["latent_bridge"])
    bridge.eval()
    z_pref_lat = bridge(rh_val).float()

    t_e4l_cdec_t0 = gen_with_cdec(net_e4l, z_pref_lat, msk_val, max_lvl=12, cdec_temp=0.0, cands=8)
    res["val_598"]["e4_latent_cdec_t0"] = score_all(t_e4l_cdec_t0, val_plans)
    print("val_598 e4_latent_cdec_t0:", res["val_598"]["e4_latent_cdec_t0"], flush=True)

    # E2E 48 topics
    top_all = torch.load("/home/usergpu/merge_lf/top_shard_0.pt", map_location="cpu", weights_only=True)
    top_all.sort(key=lambda x: x["idx"])
    tp_48 = [x["text_plan"] for x in top_all]
    hp_48 = [x["hybrid_plan"] for x in top_all]
    pids_tp = S.encode_plans(tk, tp_48, 64, pad_id).to(dev)
    pids_hp = S.encode_plans(tk, hp_48, 64, pad_id).to(dev)
    rh_hp = torch.stack([x["rh_hybrid"] for x in top_all]).to(dev)
    msk_hp = (torch.stack([x["msk_hybrid"] for x in top_all]).to(dev) != 0)

    e4f_tp_cdec = gen_with_cdec(net_e4f, tk.vqvae.encoder.wte(pids_tp), (pids_tp == pad_id), max_lvl=12, cdec_temp=0.0, cands=8)
    e4f_hp_cdec = gen_with_cdec(net_e4f, tk.vqvae.encoder.wte(pids_hp), (pids_hp == pad_id), max_lvl=12, cdec_temp=0.0, cands=8)
    e4l_hp_cdec = gen_with_cdec(net_e4l, bridge(rh_hp).float(), msk_hp, max_lvl=12, cdec_temp=0.0, cands=8)

    res["e2e_48"]["e4_fluency_cdec_textplan"] = score_all(e4f_tp_cdec, tp_48)
    res["e2e_48"]["e4_fluency_cdec_hybridplan"] = score_all(e4f_hp_cdec, hp_48)
    res["e2e_48"]["e4_latent_cdec_hybridplan"] = score_all(e4l_hp_cdec, hp_48)
    for k, v in res["e2e_48"].items():
        print("e2e_48", k, v, flush=True)

    # Update subset20
    sub = json.load(open("/home/usergpu/merge_lf/out/subset20.json", encoding="utf-8"))
    p_idxs = sub["plan_idxs"]
    sub["systems"]["e4_fluency_cdec"] = [t_e4f_cdec_t0[i] for i in p_idxs]
    sub["systems"]["e4_latent_cdec"] = [t_e4l_cdec_t0[i] for i in p_idxs]
    json.dump(sub, open("/home/usergpu/merge_lf/out/subset20_cdec.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(res, open("/home/usergpu/merge_lf/out/eval_cdec.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    for idx in range(4):
        pi = p_idxs[idx]
        print(f"=== PLAN {pi}: {val_plans[pi]} ===")
        print(f"  [e4_fluency_cdec]: {t_e4f_cdec_t0[pi][:350]}")
        print(f"  [e4_latent_cdec ]: {t_e4l_cdec_t0[pi][:350]}")


if __name__ == "__main__":
    main()
