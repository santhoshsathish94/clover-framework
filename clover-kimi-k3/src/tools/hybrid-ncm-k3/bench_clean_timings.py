"""Benchmark exact per-stage timings of the clean DSpark-free pipeline:
HybridAgent (<latent> rollout & R(h) extraction) + e6_k3_native (11-pass NCM) + K3PlanCopyDecoder
(both 1-candidate batched and MBR8 batched in a single batch=16 forward pass) + Clover Kimi K3 verification."""
import json, os, time, torch

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
import sys
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, "/home/usergpu/ncm/repo")

from k3_tokenizer import K3Tokenizer
from train_k3_native_all import load_k3_native_models, V_K3, D_VQ, D_HA, L_WMT, P_RH, P_BPE, P_COMBO, K3_EOS, K3_PAD
from train_k3_copy_decoder import K3PlanCopyDecoder
from train_wmt14_k3_v2 import window_to_fact_plan
from train_combo_prefix import ComboBridge, pack_combo_prefix
from eval_both_tracks import jacc, pick_tokens, generate_from_zpref
from hybrid_agent.model import load_hybrid_agent
from train_latent_bridge import encode_batch_rh

dev = torch.device("cuda:0")
dev_ha = torch.device("cuda:1")
out_dir = "/home/usergpu/merge_lf/out/e6_k3_native"
k3_tok = K3Tokenizer()

ha_model, ha_tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev_ha, require_hybridlink=True)
ha_tok.padding_side = "right"

vq_ckpt = os.path.join(out_dir, "k3_vqvae_adapted.pt")
ncm_ckpt = os.path.join(out_dir, "e6_last.pt")
cfg, tk, vqvae, net, src = load_k3_native_models(dev, vq_ckpt, ncm_ckpt)
vqvae.load_state_dict(src["vqvae"])
vqvae.eval(); net.eval()
cbridge = ComboBridge(D_HA, D_VQ).to(dev)
cbridge.load_state_dict(src["combo_bridge"])
cbridge.eval()

copy_dec = K3PlanCopyDecoder(d=D_VQ, n_layer=2, n_head=6).to(dev)
copy_dec.load_state_dict(torch.load(os.path.join(out_dir, "k3_copy_dec.pt"), map_location="cpu", weights_only=True)["copy_dec"])
copy_dec.eval()

levels = list(vqvae.vq.levels)
sched_10_04 = [1.0 - 0.6 * (k / (len(levels) - 1)) for k in range(len(levels))]

def dec_k3(r):
    r = list(r)
    if K3_EOS in r:
        r = r[:r.index(K3_EOS)]
    clean_ids = []
    for t in r:
        ti = int(t)
        if 0 <= ti < 163584 and (not clean_ids or ti != clean_ids[-1]):
            clean_ids.append(ti)
    return k3_tok.decode(clean_ids), clean_ids

# Warmup
p_warm = "The Harbourside Council voted 7 2 Tuesday approve; 4 million pounds new cycle lane"
r_w, m_w = encode_batch_rh(ha_model, ha_tok, [p_warm], P_RH, dev_ha)
zr_w = cbridge(r_w.to(dev)).float()
mw = (m_w.to(dev) != 0)
pk_w = torch.tensor([k3_tok.encode(p_warm)[:P_BPE] + [K3_PAD] * (P_BPE - len(k3_tok.encode(p_warm)[:P_BPE]))], dtype=torch.long, device=dev)
zb_w = vqvae.encoder.wte(pk_w)
zp_w, pm_w = pack_combo_prefix(zr_w, mw, zb_w, (pk_w == K3_PAD))
with torch.no_grad():
    g_w = generate_from_zpref(net, tk, zp_w, pm_w, sched_10_04)
    zh_w, _ = vqvae.vq.idx_to_zhats(g_w, return_zlevels=True)
    _ = copy_dec(vqvae.decoder, zh_w[-1], zr_w, mw, zb_w, pk_w)
torch.cuda.synchronize(dev); torch.cuda.synchronize(dev_ha)

k3_sum = json.load(open("/home/usergpu/k3ncm/k3_summary.json", encoding="utf-8"))
p1_ans = "The Harbourside Council voted 7 to 2 on Tuesday to approve 2.4 million pounds for a new cycle lane along Mill Road. Councillor Priya Raman proposed the plan. Work is set to begin in March and should take 14 weeks to complete. Shop owners on Mill Road, led by Tom Eliot, had asked for the vote to be delayed because of concerns about parking. In response, the council promised to provide 40 temporary parking spaces on Dock Street. The decision means the project will go ahead as planned, despite objections from local businesses worried about losing parking for customers during the construction period."
p1_sum = "Harbourside Council approved funding for a Mill Road cycle lane despite parking concerns from local shop owners."
p5_ans = "The first train travels for 30 minutes before the second train starts, covering 42 miles. This leaves 126 miles between them at 09:40. Their combined speed is 154 miles per hour, so they close this gap in 126 divided by 154 hours, which is about 49 minutes. They meet at approximately 10:29. The second train travels 70 multiplied by 49/60 miles, about 57 miles from London."
p5_sum = "The trains meet at about 10:29, roughly 57 miles from London."
k3_ref = {"p1": {"answer": p1_ans, "summary": p1_sum}, **k3_sum, "p5": {"answer": p5_ans, "summary": p5_sum}}

timings = {}
for pid in ("p1", "p2", "p3", "p4", "p5"):
    ans_txt = k3_ref[pid]["answer"]
    sum_txt = k3_ref[pid]["summary"]
    full_k3_txt = f"{ans_txt} Summary: {sum_txt}"
    k_full = k3_tok.encode(full_k3_txt)
    p_w1 = window_to_fact_plan(k3_tok.decode(k_full[:60]))
    p_w2 = window_to_fact_plan(k3_tok.decode(k_full[60:120]))
    plans = [p_w1, p_w2]

    torch.cuda.synchronize(dev); torch.cuda.synchronize(dev_ha)
    t0 = time.perf_counter()
    rh_p, msk_p = encode_batch_rh(ha_model, ha_tok, plans, P_RH, dev_ha)
    torch.cuda.synchronize(dev_ha)
    t_ha = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    with torch.no_grad():
        z_rh = cbridge(rh_p.to(dev)).float()
        msk_r = (msk_p.to(dev) != 0)
        p_kpad = torch.tensor([k3_tok.encode(p)[:P_BPE] + [K3_PAD] * (P_BPE - len(k3_tok.encode(p)[:P_BPE])) for p in plans], dtype=torch.long, device=dev)
        z_bpe = vqvae.encoder.wte(p_kpad)
        z_pref, pmask = pack_combo_prefix(z_rh, msk_r, z_bpe, (p_kpad == K3_PAD))
        torch.manual_seed(42)
        g1 = generate_from_zpref(net, tk, z_pref, pmask, sched_10_04)
        zh1, _ = vqvae.vq.idx_to_zhats(g1, return_zlevels=True)
        lg1 = copy_dec(vqvae.decoder, zh1[-1], z_rh, msk_r, z_bpe, p_kpad)
        ids_1cand = []
        for r in pick_tokens(lg1).tolist():
            _, i_s = dec_k3(r)
            ids_1cand.extend(i_s)
    torch.cuda.synchronize(dev)
    t_ncm_1cand = (time.perf_counter() - t1) * 1000.0

    # Batched MBR8 (all 8 candidates x 2 windows = batch 16 in a single 11-pass call!)
    t2 = time.perf_counter()
    with torch.no_grad():
        z_pref_8 = z_pref.repeat(8, 1, 1)
        pmask_8 = pmask.repeat(8, 1)
        z_rh_8 = z_rh.repeat(8, 1, 1)
        msk_r_8 = msk_r.repeat(8, 1)
        z_bpe_8 = z_bpe.repeat(8, 1, 1)
        p_kpad_8 = p_kpad.repeat(8, 1)
        torch.manual_seed(42)
        g8 = generate_from_zpref(net, tk, z_pref_8, pmask_8, sched_10_04)
        zh8, _ = vqvae.vq.idx_to_zhats(g8, return_zlevels=True)
        lg8 = copy_dec(vqvae.decoder, zh8[-1], z_rh_8, msk_r_8, z_bpe_8, p_kpad_8)
        toks8 = pick_tokens(lg8).tolist()
        cands_t = [[], []]
        cands_i = [[], []]
        for k in range(8):
            for w_i in (0, 1):
                ts, is_ = dec_k3(toks8[k * 2 + w_i])
                cands_t[w_i].append(ts); cands_i[w_i].append(is_)
        mbr_ids = []
        for w_i in (0, 1):
            sc = [sum(jacc(cands_t[w_i][k], cands_t[w_i][j]) for j in range(8) if j != k) for k in range(8)]
            bk = max(range(8), key=lambda k: sc[k])
            mbr_ids.extend(cands_i[w_i][bk])
    torch.cuda.synchronize(dev)
    t_ncm_mbr8_batched = (time.perf_counter() - t2) * 1000.0

    timings[pid] = {
        "ha_rh_ms": round(t_ha, 2),
        "ncm_copydec_1cand_2win_ms": round(t_ncm_1cand, 2),
        "ncm_copydec_mbr8_batched_2win_ms": round(t_ncm_mbr8_batched, 2),
        "total_1cand_ms": round(t_ha + t_ncm_1cand, 2),
        "total_mbr8_batched_ms": round(t_ha + t_ncm_mbr8_batched, 2),
        "n_k3_tokens": len(mbr_ids),
    }
    print(pid, timings[pid], flush=True)

json.dump(timings, open(os.path.join(out_dir, "timings_no_dspark.json"), "w"), indent=1)
