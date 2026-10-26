"""Re-run HybridAgent (<latent> mode) + e5_k3_v2 on p1..p5 with fixed `</latent></think>\nPlan:` bridge
(fixing the `Example plan:` bridge label from run_factual_latent_rollout that caused HybridAgent to echo the 1-shot example),
and also measure exact two-window (126-token) generation so all 100-130 tokens of p1..p5 are covered."""
import json, os, re, time, torch
import torch.nn.functional as F
os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
import train_wmt14_k3 as T
from train_wmt14_k3_v2 import window_to_fact_plan
from safetensors import safe_open
from hybrid_agent.model import load_hybrid_agent
from hybrid_agent.readout import normalized_embeddings, pseudo_text
from hybrid_agent.constants import ENTRY_ANCHOR, EXIT_ANCHOR, THINK_OPEN, THINK_CLOSE, special_token_id

PROMPT_HA = (
    "Extract the exact facts, names, numbers, and events in order for the following request as a single line of short phrases "
    "separated by semicolons (at most 45 words). Write ONLY the semicolon-separated factual phrases.\n\n"
    "Example request: Write a report: River Bridge closed at 06:30 on Monday after a lorry crash. Engineers inspected 3 steel beams. Traffic was diverted to Oak Road for 5 hours.\n"
    "Plan: River Bridge closed 06:30 Monday lorry crash; engineers inspected 3 steel beams; traffic diverted Oak Road 5 hours; Summary River Bridge closed 5 hours lorry crash\n\n"
    "Request: {req}"
)

@torch.no_grad()
def run_ha_latent_clean(model, tokenizer, embed_bank, req_text, min_lat=8, max_lat=20, max_new=72):
    dev = next(model.parameters()).device
    backbone = model.backbone
    embed_w = backbone.get_input_embeddings().weight
    lat_close_id = special_token_id(tokenizer, EXIT_ANCHOR)

    user_content = PROMPT_HA.format(req=req_text)
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

    bridge_str = EXIT_ANCHOR + " Now I will write the semicolon-separated factual plan on one line." + THINK_CLOSE + "\nPlan:"
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
        if "\n" in txt_so_far.strip() and len(txt_so_far.strip()) > 20:
            break
        out = backbone(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=past_kv, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]

    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split("\n")[0].strip()
    raw_text = re.sub(r"^(?:Output|Sample output|Plan|Facts):\s*", "", raw_text, flags=re.I).strip()
    return {
        "latent_steps": len(latent_vecs),
        "exit_reason": exit_reason,
        "mean_readout_cos": round(sum(readout_cos) / max(len(readout_cos), 1), 4),
        "pseudo_text": pseudo_text(readout_ids, tokenizer),
        "hybrid_plan": raw_text,
    }


dev_ha = torch.device("cuda:1")
dev = torch.device("cuda:0")
ha_model, ha_tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev_ha, require_hybridlink=True)
ha_tok.padding_side = "right"
embed_bank = normalized_embeddings(ha_model.backbone.get_input_embeddings().weight).float()

cfg, tk, net_tr, src_tr = T.load_wmt14_models(dev, P=T.P_COMBO, ckpt="/home/usergpu/merge_lf/out/e5_k3_v2/last.pt")
net_tr.eval()
if "vq_decoder" in src_tr:
    tk.vqvae.decoder.load_state_dict(src_tr["vq_decoder"])
tk.vqvae.decoder.eval()
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
    t0 = time.perf_counter()
    lat_out = run_ha_latent_clean(ha_model, ha_tok, embed_bank, raw_prompt, min_lat=8, max_lat=20, max_new=72)
    ha_ms = (time.perf_counter() - t0) * 1000.0

    # Also test 2-window grounded plan (first 60 tokens + second 60 tokens of Kimi K3's answer + summary)
    full_k3_txt = f"{k3_ref[pid]['answer']} Summary: {k3_ref[pid]['summary']}"
    q_full = qtok.encode(full_k3_txt)
    w1_txt = qtok.decode(q_full[:60], skip_special_tokens=True)
    w2_txt = qtok.decode(q_full[60:120], skip_special_tokens=True)
    k3_plan_w1 = window_to_fact_plan(w1_txt)
    k3_plan_w2 = window_to_fact_plan(w2_txt) if len(q_full) > 65 else ""

    for plan_src, plans_list in (("hybridagent_latent_plan", [lat_out["hybrid_plan"]]), ("k3_grounded_1win", [k3_plan_w1]), ("k3_grounded_2win", [p for p in (k3_plan_w1, k3_plan_w2) if p])):
        t_ncm0 = time.perf_counter()
        win_outs = []
        for p_str in plans_list:
            rh_p, msk_p = T.encode_batch_rh(ha_model, ha_tok, [p_str], T.P_RH, dev_ha)
            rh_p = rh_p.to(dev); msk_p = (msk_p.to(dev) != 0)
            p_qids = qtok.encode(p_str)[:T.P_BPE]
            p_qpad = torch.tensor([p_qids + [pad_id] * (T.P_BPE - len(p_qids))], dtype=torch.long, device=dev)
            win_outs.append(gen_wmt(rh_p, msk_p, p_qpad, cands=8)[0])
        ncm_ms = (time.perf_counter() - t_ncm0) * 1000.0
        gen_txt = " ".join(win_outs)
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
            "plan": " || ".join(plans_list),
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
        print(f"   PLAN: {rec['plan']}", flush=True)
        print(f"   TEXT: {gen_txt}", flush=True)

json.dump(out_p, open("/home/usergpu/merge_lf/out/e5_k3_v2/eval_p1_p5_clean.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("saved eval_p1_p5_clean.json", flush=True)
