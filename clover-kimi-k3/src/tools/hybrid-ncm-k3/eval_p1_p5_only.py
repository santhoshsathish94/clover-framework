import json, os, time, torch
os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
import train_wmt14_k3 as T
from safetensors import safe_open
from hybrid_agent.model import load_hybrid_agent
from hybrid_agent.readout import normalized_embeddings

dev_ha = torch.device("cuda:1")
dev = torch.device("cuda:0")
ha_model, ha_tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev_ha, require_hybridlink=True)
ha_tok.padding_side = "right"
embed_bank = normalized_embeddings(ha_model.backbone.get_input_embeddings().weight).float()

cfg, tk, net_tr, src_tr = T.load_wmt14_models(dev, P=T.P_COMBO, ckpt="/home/usergpu/merge_lf/out/e5_k3_wmt/last.pt")
net_tr.eval()
cbridge = T.ComboBridge(T.D_HA, tk.vqvae.embed_dim).to(dev)
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
def gen_wmt(rh_in, msk_rh_in, bpe_pids, cands=8):
    z_bpe = tk.vqvae.encoder.wte(bpe_pids)
    msk_bpe = (bpe_pids == pad_id)
    z_rh = cbridge(rh_in).float()
    z_pref, pmask = T.pack_combo_prefix(z_rh, msk_rh_in, z_bpe, msk_bpe)
    all_c = []
    for k in range(cands):
        torch.manual_seed(42 + k)
        g = T.generate_from_zpref(net_tr, tk, z_pref, pmask, sched_10_04)
        z_hats, _ = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        lg = tk.vqvae.decoder(z_hats[-1])
        all_c.append([dec(r) for r in T.pick_tokens(lg).tolist()])
    return T.mbr(all_c) if cands > 1 else all_c[0]

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

q_to_k3, k3_bpe, id_to_bytes_k3 = T.build_qwen_to_k3_map(qtok)
st_path = "/opt/clover-k3/clover-kimi-k3/src/datasets/third-party/kimi-k3-open-weights/model-00094-of-000096.safetensors"
with safe_open(st_path, framework="pt", device="cpu") as sf:
    k3_embed_w = sf.get_tensor("language_model.model.embed_tokens.weight")

out_p = {}
for pid in ("p1", "p2", "p3", "p4", "p5"):
    raw_prompt = open(f"/home/usergpu/k3ncm/{pid}.txt", encoding="utf-8").read().strip()
    ha_prompt = (
        "Extract the key facts, names, numbers, and events in order for the following request as a single line of short phrases "
        "separated by semicolons (at most 40 words). Output only the plan line.\n\nRequest: " + raw_prompt
    )
    t0 = time.perf_counter()
    lat_out = T.run_latent_rollout_one(ha_model, ha_tok, embed_bank, ha_prompt, P=T.P_RH, min_lat=8, max_lat=20, max_new=64)
    ha_ms = (time.perf_counter() - t0) * 1000.0
    k3_plan, _ = T.extract_fact_plan(k3_ref[pid]["answer"], k3_ref[pid]["summary"])
    for plan_src, plan_str in (("hybridagent_latent_plan", lat_out["hybrid_plan"]), ("k3_structured_plan", k3_plan)):
        rh_p, msk_p = T.encode_batch_rh(ha_model, ha_tok, [plan_str], T.P_RH, dev_ha)
        rh_p = rh_p.to(dev); msk_p = (msk_p.to(dev) != 0)
        p_qids = qtok.encode(plan_str)[:T.P_BPE]
        p_qpad = torch.tensor([p_qids + [pad_id] * (T.P_BPE - len(p_qids))], dtype=torch.long, device=dev)
        t_ncm0 = time.perf_counter()
        gen_txt = gen_wmt(rh_p, msk_p, p_qpad, cands=8)[0]
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
        out_p[f"{pid}_{plan_src}"] = rec
        kk = rec["keys_kept"]
        print(f"[{pid} | {plan_src}] keys={kk} k3_toks={len(k3_ids)}/{len(gen_qids)} ncm_ms={ncm_ms:.1f}", flush=True)
        print(f"   PLAN: {plan_str}", flush=True)
        print(f"   TEXT: {gen_txt}", flush=True)

json.dump(out_p, open("/home/usergpu/merge_lf/out/e5_k3_wmt/eval_p1_p5.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("saved eval_p1_p5.json", flush=True)
