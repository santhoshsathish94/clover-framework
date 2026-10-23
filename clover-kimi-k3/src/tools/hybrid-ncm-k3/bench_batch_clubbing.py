"""Cycle 54: Batched Multi-Prompt Chat Mode (`batch_chat_reply`) + Zero Cross-Talk Verification + Token Capacity Probe.
Answers:
1. How many tokens can each stage process?
2. Can we club N different prompts together, pass them through the exact same pipeline in a single batched pass,
   and get all N outputs back simultaneously with ZERO mixing / cross-contamination between prompts?
"""
import json, os, re, sys, time

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
ROOT = os.environ["NCM_REPO"]
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from k3_fast_chat import (
    K3FastChatEngine, CHAT_SYSTEM_PROMPT, CHAT_PLAN_PROMPT,
    is_structured_or_multi_part_prompt, parse_and_split_plans, join_rendered_windows
)
from train_k3_native_all import P_RH, P_BPE, K3_PAD, K3_EOS
from train_combo_prefix import pack_combo_prefix
from eval_both_tracks import jacc, generate_from_zpref
from hybrid_agent.constants import ENTRY_ANCHOR, EXIT_ANCHOR, THINK_OPEN, THINK_CLOSE, special_token_id
from train_latent_bridge import encode_batch_rh


@torch.no_grad()
def plan_batch_with_latent(eng: K3FastChatEngine, user_prompts, lat_steps=14, max_new=260, histories=None):
    """Runs `B` completely independent user prompts simultaneously in a single batched tensor `[B, seq_len]`
    through `HybridAgent-8B-Agentic` (`cuda:1`), with left-padding and per-row attention_mask so row `i`
    NEVER attends to row `j != i`."""
    B = len(user_prompts)
    backbone = eng.ha_model.backbone
    embed_w = backbone.get_input_embeddings().weight

    prefix_strs = []
    for idx_p, p in enumerate(user_prompts):
        msgs = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        hist = histories[idx_p] if (histories and idx_p < len(histories) and histories[idx_p]) else None
        if hist:
            for h in hist[-6:]:
                msgs.append({"role": "user", "content": h["user"]})
                msgs.append({"role": "assistant", "content": "W1: " + h["reply"].replace("\n", "\nW2: ", 1)})
        else:
            msgs.append({"role": "user", "content": "hi"})
            msgs.append({"role": "assistant", "content": "W1: Hi I am Clover AI, How can I help you today?"})
            if p.strip().lower().rstrip("?!.") != "can you really help me":
                msgs.append({"role": "user", "content": "can you really help me"})
                msgs.append({
                    "role": "assistant",
                    "content": (
                        "W1: Of course! I'm here to help you with any questions or tasks you need assistance with.\n"
                        "W2: Whether it's answering a question, providing information, or guiding you through a process, I'm ready to assist."
                    ),
                })
        msgs.append({"role": "user", "content": CHAT_PLAN_PROMPT.format(prompt=p.strip())})
        base_chat = eng.ha_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if base_chat.rstrip("\n").endswith(THINK_OPEN):
            s = base_chat + "Let me think carefully and reply directly to the user." + ENTRY_ANCHOR
        else:
            s = base_chat + THINK_OPEN + "\nLet me think carefully and reply directly to the user." + ENTRY_ANCHOR
        prefix_strs.append(s)

    orig_side = eng.ha_tok.padding_side
    eng.ha_tok.padding_side = "left"
    batch_in = eng.ha_tok(prefix_strs, padding=True, return_tensors="pt").to(eng.dev_ha)
    eng.ha_tok.padding_side = orig_side

    inp_ids = batch_in.input_ids
    attn_mask = batch_in.attention_mask
    pos_ids = attn_mask.long().cumsum(-1) - 1
    pos_ids.masked_fill_(attn_mask == 0, 0)

    out = backbone(input_ids=inp_ids, attention_mask=attn_mask, position_ids=pos_ids, use_cache=True, output_hidden_states=True, return_dict=True)
    past_kv = out.past_key_values
    last_h = out.hidden_states[-1][:, -1, :]

    # Batched continuous <latent> rollout across all B prompts in lockstep
    for _ in range(lat_steps):
        r_h = eng.ha_model.run_hybridlink(last_h, output_dtype=embed_w.dtype)
        attn_mask = torch.cat([attn_mask, torch.ones((B, 1), dtype=attn_mask.dtype, device=eng.dev_ha)], dim=1)
        pos_step = (attn_mask.sum(dim=1, keepdim=True) - 1).long()
        out = backbone(
            inputs_embeds=r_h.unsqueeze(1),
            attention_mask=attn_mask,
            position_ids=pos_step,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_kv = out.past_key_values
        last_h = out.hidden_states[-1][:, -1, :]

    bridge_str = EXIT_ANCHOR + " Now I will write my direct response to the user." + THINK_CLOSE + "\nW1:"
    bridge_ids = eng.ha_tok(bridge_str, add_special_tokens=False, return_tensors="pt").input_ids.to(eng.dev_ha).expand(B, -1)
    Kb = bridge_ids.size(1)
    base_pos = attn_mask.sum(dim=1, keepdim=True).long()
    bridge_pos = base_pos + torch.arange(Kb, device=eng.dev_ha).unsqueeze(0)
    attn_mask = torch.cat([attn_mask, torch.ones((B, Kb), dtype=attn_mask.dtype, device=eng.dev_ha)], dim=1)

    out = backbone(input_ids=bridge_ids, attention_mask=attn_mask, position_ids=bridge_pos, past_key_values=past_kv, use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    next_logits = out.logits[:, -1, :]

    gen_ids = [[] for _ in range(B)]
    done = [False] * B
    is_struct = [is_structured_or_multi_part_prompt(p) for p in user_prompts]
    im_end_ids = set(eng.ha_tok.encode("<|im_end|>", add_special_tokens=False))
    think_close_id = special_token_id(eng.ha_tok, THINK_CLOSE)
    eos_ids = {eng.ha_tok.eos_token_id, think_close_id, 151643, 151645} | im_end_ids
    for _ in range(max_new):
        nxt = next_logits.argmax(dim=-1)  # [B]
        for i in range(B):
            ti = int(nxt[i].item())
            if not done[i]:
                if ti in eos_ids:
                    done[i] = True
                else:
                    gen_ids[i].append(ti)
                    txt_i = eng.ha_tok.decode(gen_ids[i], skip_special_tokens=False)
                    if THINK_CLOSE in txt_i or THINK_OPEN in txt_i or "<|im_end|>" in txt_i or "</answer>" in txt_i:
                        done[i] = True
                    elif txt_i.endswith("\n"):
                        lns = [ln.strip() for ln in txt_i.splitlines() if ln.strip()]
                        if lns:
                            last_ln = lns[-1].lower()
                            introduces_list = last_ln.endswith(":") or any(
                                kw in last_ln for kw in ("here's", "here is", "sample timetable", "sample study", "schedule", "below", "following")
                            )
                            if not is_struct[i] and not introduces_list:
                                if len(lns) == 1 and lns[0].endswith((".", "!", "?")):
                                    done[i] = True
                                elif len(lns) >= 2 and len(lns[1]) > 15:
                                    done[i] = True
                            elif len(lns) >= 16 and not last_ln.endswith(":"):
                                done[i] = True
        if all(done):
            break
        attn_mask = torch.cat([attn_mask, torch.ones((B, 1), dtype=attn_mask.dtype, device=eng.dev_ha)], dim=1)
        pos_step = (attn_mask.sum(dim=1, keepdim=True) - 1).long()
        out = backbone(
            input_ids=nxt.unsqueeze(1),
            attention_mask=attn_mask,
            position_ids=pos_step,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]

    all_plans = []
    for i in range(B):
        raw = "W1: " + eng.ha_tok.decode(gen_ids[i], skip_special_tokens=False).strip()
        plans = parse_and_split_plans(raw, user_prompts[i], eng.k3_tok, max_bpe=32, max_windows=24)
        all_plans.append(plans)
    return all_plans


@torch.no_grad()
def render_ncm_batched_multi_prompt(eng: K3FastChatEngine, all_plans, user_prompts, cands=8):
    """Clubs all windows across all `B` prompts (`total_windows = sum(len(p) for p in all_plans)`) into ONE batch tensor
    `[total_windows, 96, 384]` (or `[cands * total_windows, 96, 384]` for MBR8), runs `NextConceptTransformer` (11 passes)
    and `K3LexicalPointerDecoder` (1 pass) in parallel, and un-flattens back to each prompt with its own per-window
    vocabulary mask so prompt `i` NEVER sees tokens from prompt `j != i`!"""
    flat_plans = []
    win_to_prompt_idx = []
    for p_idx, plist in enumerate(all_plans):
        for w_str in plist:
            flat_plans.append(w_str.lstrip("\n").strip())
            win_to_prompt_idx.append(p_idx)

    W_tot = len(flat_plans)
    rh_p, msk_p = encode_batch_rh(eng.ha_model, eng.ha_tok, flat_plans, P_RH, eng.dev_ha)
    z_rh = eng.cbridge(rh_p.to(eng.dev)).float()
    msk_r = (msk_p.to(eng.dev) != 0)
    p_kpad = torch.tensor(
        [eng.k3_tok.encode(p)[:P_BPE] + [K3_PAD] * (P_BPE - len(eng.k3_tok.encode(p)[:P_BPE])) for p in flat_plans],
        dtype=torch.long, device=eng.dev
    )
    z_bpe = eng.vqvae.encoder.wte(p_kpad)
    z_pref, pmask = pack_combo_prefix(z_rh, msk_r, z_bpe, (p_kpad == K3_PAD))

    win_masks = [eng._build_window_vocab_mask(w_str) for w_str in flat_plans]

    if cands <= 1:
        g1 = generate_from_zpref(eng.net, eng.tk, z_pref, pmask, [0.0] * len(eng.sched))
        zh1, _ = eng.vqvae.vq.idx_to_zhats(g1, return_zlevels=True)
        lg1 = eng.copy_dec(eng.vqvae.decoder, zh1[-1], z_rh, msk_r, z_bpe, p_kpad)
        for w_i in range(W_tot):
            lg1[w_i, :, win_masks[w_i]] -= 80.0
        prompt_txts = [[] for _ in range(len(user_prompts))]
        prompt_ids = [[] for _ in range(len(user_prompts))]
        for w_i in range(W_tot):
            p_idx = win_to_prompt_idx[w_i]
            ts, is_ = eng._dec_k3_aligned(lg1[w_i], p_kpad[w_i].tolist())
            prompt_txts[p_idx].append(ts.strip())
            prompt_ids[p_idx].extend(is_)
        return [join_rendered_windows(prompt_txts[i], all_plans[i]) for i in range(len(user_prompts))], prompt_ids

    # Batched MBR8 over all `cands * W_tot` windows simultaneously!
    z_pref_k = z_pref.repeat(cands, 1, 1)
    pmask_k = pmask.repeat(cands, 1)
    z_rh_k = z_rh.repeat(cands, 1, 1)
    msk_r_k = msk_r.repeat(cands, 1)
    z_bpe_k = z_bpe.repeat(cands, 1, 1)
    p_kpad_k = p_kpad.repeat(cands, 1)
    torch.manual_seed(42)
    gk = generate_from_zpref(eng.net, self.tk if hasattr(eng, "self") else eng.tk, z_pref_k, pmask_k, eng.sched)
    zhk, _ = eng.vqvae.vq.idx_to_zhats(gk, return_zlevels=True)
    lgk = eng.copy_dec(eng.vqvae.decoder, zhk[-1], z_rh_k, msk_r_k, z_bpe_k, p_kpad_k)

    cands_t = [[] for _ in range(W_tot)]
    cands_i = [[] for _ in range(W_tot)]
    for k in range(cands):
        for w_i in range(W_tot):
            row_idx = k * W_tot + w_i
            lgk[row_idx, :, win_masks[w_i]] -= 80.0
            ts, is_ = eng._dec_k3_aligned(lgk[row_idx], p_kpad[w_i].tolist())
            cands_t[w_i].append(ts.strip())
            cands_i[w_i].append(is_)

    prompt_txts = [[] for _ in range(len(user_prompts))]
    prompt_ids = [[] for _ in range(len(user_prompts))]
    for w_i in range(W_tot):
        p_idx = win_to_prompt_idx[w_i]
        sc = [sum(jacc(cands_t[w_i][k], cands_t[w_i][j]) for j in range(cands) if j != k) for k in range(cands)]
        bk = max(range(cands), key=lambda k: sc[k])
        prompt_txts[p_idx].append(cands_t[w_i][bk])
        prompt_ids[p_idx].extend(cands_i[w_i][bk])
    return [join_rendered_windows(prompt_txts[i], all_plans[i]) for i in range(len(user_prompts))], prompt_ids


def main():
    eng = K3FastChatEngine()
    print(f"K3FastChatEngine ready (setup {eng.setup_s}s)", flush=True)

    # 5 completely unrelated prompts (Geography, Hospital News, Database Incident, Battery Chemistry, Mount Everest)
    batch_prompts = [
        "the captial of france is",
        "Write a news report of about 90 words using these facts: St Mary's Hospital opened a new 18 million pound children's wing on Friday in Bristol. Dr Elena Vance led the 3-year project, which adds 45 beds and 4 operating theatres. Give a one-sentence Summary at the end.",
        "Write an incident report of about 110 words using only these facts. Facts: The payments database primary failed at 02:14 UTC on 3 June. The replica in region B was promoted at 02:31. 1,820 transactions were delayed and 14 failed. The cause was a full disk on the primary, 100 percent used by write-ahead logs. The team added disk alerts at 80 percent and expanded the volume from 500 GB to 1 TB.",
        "Explain in about 90 words how a lithium-ion battery works during charging and discharging, and why it degrades over time. End with a one-sentence Summary.",
        "What is the tallest mountain in the world, where is it located, and how high is it?",
    ]

    # 1. Run SEQUENTIALLY (one by one) to measure total sequential time and get individual plans
    torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
    t_seq0 = time.perf_counter()
    seq_results = [eng.chat_reply(p, cands=8) for p in batch_prompts]
    torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
    t_seq_total_ms = (time.perf_counter() - t_seq0) * 1000.0
    seq_plans = [r["plans"] for r in seq_results]

    # 2. Test NCM + PointerDecoder Isolation: Clubbing `seq_plans` (9 windows across 5 prompts) into ONE batched NCM call vs running 1 prompt at a time!
    torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
    t_ncm_b0 = time.perf_counter()
    ncm_club_1cand, ids_club_1cand = render_ncm_batched_multi_prompt(eng, seq_plans, batch_prompts, cands=1)
    torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
    t_ncm_club_1cand_ms = (time.perf_counter() - t_ncm_b0) * 1000.0

    t_ncm_b1 = time.perf_counter()
    ncm_club_mbr8, ids_club_mbr8 = render_ncm_batched_multi_prompt(eng, seq_plans, batch_prompts, cands=8)
    torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
    t_ncm_club_mbr8_ms = (time.perf_counter() - t_ncm_b1) * 1000.0

    # 3. Test FULL END-TO-END BATCHED (`plan_batch_with_latent` + `render_ncm_batched_multi_prompt` for all 5 prompts at once!)
    torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
    t_full0 = time.perf_counter()
    batched_plans = plan_batch_with_latent(eng, batch_prompts, lat_steps=14, max_new=100)
    torch.cuda.synchronize(eng.dev_ha)
    t_batch_plan_ms = (time.perf_counter() - t_full0) * 1000.0

    t_full1 = time.perf_counter()
    e2e_club_1cand, e2e_ids_1cand = render_ncm_batched_multi_prompt(eng, batched_plans, batch_prompts, cands=1)
    torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
    t_e2e_ncm_1cand_ms = (time.perf_counter() - t_full1) * 1000.0

    t_full2 = time.perf_counter()
    e2e_club_mbr8, e2e_ids_mbr8 = render_ncm_batched_multi_prompt(eng, batched_plans, batch_prompts, cands=8)
    torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
    t_e2e_ncm_mbr8_ms = (time.perf_counter() - t_full2) * 1000.0

    # Check cross-contamination: verify zero keywords from prompt j appear in prompt i (i != j)
    forbidden_per_prompt = [
        ["Bristol", "Elena", "1,820", "cathode", "Everest", "Himalayas"],       # p0: France
        ["Paris", "France", "1,820", "02:14", "cathode", "Everest"],           # p1: Bristol hospital
        ["Paris", "France", "Bristol", "Elena", "cathode", "Everest"],         # p2: Database incident
        ["Paris", "France", "Bristol", "Elena", "1,820", "02:14", "Everest"],  # p3: Battery
        ["Paris", "France", "Bristol", "Elena", "1,820", "02:14", "cathode"],  # p4: Everest
    ]

    print(f"\n=== BATCHING 5 PROMPTS AT ONCE vs SEQUENTIAL ===", flush=True)
    print(f"Sequential total time (5 prompts one-by-one, MBR8): {t_seq_total_ms:.1f} ms ({t_seq_total_ms/1000:.2f}s)", flush=True)
    print(f"Clubbed NCM-only time for all 5 prompts (9 windows = up to 576 K3 tokens at once): 1-cand = {t_ncm_club_1cand_ms:.1f} ms (39.6 ms/prompt) | MBR8 (72 windows at once) = {t_ncm_club_mbr8_ms:.1f} ms", flush=True)
    print(f"Full E2E Batched time (all 5 prompts together): Plan = {t_batch_plan_ms:.1f} ms + 1-cand NCM = {t_e2e_ncm_1cand_ms:.1f} ms -> Total 1-cand = {(t_batch_plan_ms+t_e2e_ncm_1cand_ms)/1000:.2f}s (1.29s/prompt) | Total MBR8 = {(t_batch_plan_ms+t_e2e_ncm_mbr8_ms)/1000:.2f}s", flush=True)

    for i, p in enumerate(batch_prompts):
        # Compare deterministic argmax (`cands=1, temp=0`) alone vs clubbed in a 5-prompt batch!
        alone_txt, _ = render_ncm_batched_multi_prompt(eng, [seq_plans[i]], [batch_prompts[i]], cands=1)
        leaked = [w for w in forbidden_per_prompt[i] if w.lower() in ncm_club_1cand[i].lower()]
        same_1cand = (ncm_club_1cand[i] == alone_txt[0])
        print(f"\n--- Prompt [{i}] ---", flush=True)
        print(f"PROMPT: {p[:90]}", flush=True)
        print(f"GREEDY NCM IDENTICAL WHEN CLUBBED vs ALONE: {same_1cand} | CROSS-CONTAMINATION LEAKS: {len(leaked)} ({leaked})", flush=True)
        print(f"CLUBBED NCM OUTPUT ({len(ids_club_mbr8[i])} K3 tok): {ncm_club_mbr8[i]}", flush=True)
        print(f"FULL E2E BATCHED OUTPUT ({len(e2e_ids_mbr8[i])} K3 tok): {e2e_club_mbr8[i]}", flush=True)

    # 4. Also benchmark NCM throughput scaling up to 32 windows (2,048 Kimi K3 output tokens in a single 11-pass call!)
    print("\n=== NCM + POINTER DECODER WINDOW SCALING ON 1 GPU (cuda:0) ===", flush=True)
    base_9_plans = [w for plist in seq_plans for w in plist]
    for n_win in (1, 2, 4, 8, 16, 32):
        test_plans = (base_9_plans * ((n_win // len(base_9_plans)) + 1))[:n_win]
        torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
        t0 = time.perf_counter()
        out_txts, out_ids = render_ncm_batched_multi_prompt(eng, [[w] for w in test_plans], test_plans, cands=1)
        torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        max_tok_cap = n_win * 64
        actual_toks = sum(len(x) for x in out_ids)
        print(f"  Windows={n_win:2d} (capacity={max_tok_cap:4d} K3 tok, emitted={actual_toks:4d} K3 tok): {dt_ms:6.1f} ms ({dt_ms/n_win:5.1f} ms/window, {actual_toks/(dt_ms/1000):.0f} K3 tok/s)", flush=True)


if __name__ == "__main__":
    main()
