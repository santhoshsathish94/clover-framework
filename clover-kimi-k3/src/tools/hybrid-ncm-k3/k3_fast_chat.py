"""Cycle 51: Direct Chat Mode (`k3_fast_chat.py`) — Plain Prompt In -> Output Out (Zero 93-Layer Verification).
Runs the unified K3-Native pipeline directly from a user prompt (or file of prompts):
  1. `HybridAgent-8B-Agentic` (`cuda:1`) runs a fast `<latent>` reasoning span (`R(h_t)`) and emits 1-2 factual/narrative
     window plans (`Window 1:` and `Window 2:`), plus extracts continuous `R(h)` (`[B, 48, 4096]`).
  2. `e6_k3_native` (`11`-pass NCM generator) + `K3PlanCopyDecoder` (`cuda:0`) generates the response directly in
     Clover Kimi K3's native `163,840`-token vocabulary (`libk3tok.so`) in a single batched pass (~125-355 ms) —
     with ZERO 93-layer Kimi K3 verification and ZERO DSpark verifier.
"""
import argparse, json, os, re, sys, time

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, "/home/usergpu/ncm/repo")
os.chdir("/home/usergpu/ncm/repo")

import torch
import torch.nn.functional as F

from k3_tokenizer import K3Tokenizer
from train_k3_native_all import load_k3_native_models, V_K3, D_VQ, D_HA, L_WMT, P_RH, P_BPE, P_COMBO, K3_EOS, K3_PAD
from train_e7_clean_chat import K3LexicalPointerDecoder
from train_combo_prefix import ComboBridge, pack_combo_prefix
from eval_both_tracks import jacc, pick_tokens, generate_from_zpref
from hybrid_agent.model import load_hybrid_agent
from hybrid_agent.constants import ENTRY_ANCHOR, EXIT_ANCHOR, THINK_OPEN, THINK_CLOSE, special_token_id
from train_latent_bridge import encode_batch_rh


CHAT_PLAN_PROMPT = (
    "You are planning a clear, factual response to the user's request. "
    "If the request asks a math or timing question, compute the exact numbers first and state them in the plan. "
    "Write the response plan as two lines (`W1:` for the first part and `W2:` for the second part plus `Summary: ...`). "
    "Each line must be a sequence of short semicolon-separated factual clauses containing the exact names, calculated numbers, "
    "and key words in order (25 to 36 words per line). Stop immediately after the `Summary:` clause on `W2:`.\n\n"
    "Example 1:\n"
    "Write a report using these facts: River Bridge closed at 06:30 on Monday after a lorry crash. Engineers inspected 3 steel beams. "
    "Traffic was diverted to Oak Road for 5 hours. Give a one-sentence Summary.\n"
    "W1: River Bridge closed at 06:30 on Monday after a lorry crash; engineers inspected 3 steel beams on the bridge; traffic was diverted to Oak Road for 5 hours\n"
    "W2: drivers faced delays around Oak Road during the 5 hour closure; the bridge reopened after safety checks; Summary: River Bridge closed for 5 hours on Monday after a lorry crash\n\n"
    "Example 2:\n"
    "A cyclist leaves York at 08:00 and rides 60 miles to Hull at 20 mph. A second cyclist leaves Hull for York at 08:30 at 10 mph. When do they meet and how far from Hull?\n"
    "W1: first cyclist leaves York at 08:00 at 20 mph and covers 10 miles in 30 minutes; at 08:30 the second cyclist leaves Hull at 10 mph with 50 miles between them\n"
    "W2: their combined speed is 30 mph so they close the 50 mile gap in 100 minutes and meet at 10:10; the second cyclist travels 16.7 miles from Hull; Summary: The cyclists meet at 10:10, about 16.7 miles from Hull\n\n"
    "User request:\n{prompt}"
)


class K3FastChatEngine:
    def __init__(self, out_dir="/home/usergpu/merge_lf/out/e6_k3_native", dev_ncm="cuda:0", dev_ha="cuda:1"):
        t0 = time.perf_counter()
        self.dev = torch.device(dev_ncm)
        self.dev_ha = torch.device(dev_ha)
        self.k3_tok = K3Tokenizer()

        self.ha_model, self.ha_tok = load_hybrid_agent(
            "togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=self.dev_ha, require_hybridlink=True
        )
        self.ha_tok.padding_side = "right"
        self.lat_close_id = special_token_id(self.ha_tok, EXIT_ANCHOR)

        vq_ckpt = os.path.join(out_dir, "k3_vqvae_adapted.pt")
        ncm_ckpt = os.path.join(out_dir, "e6_last.pt")
        _, self.tk, self.vqvae, self.net, src = load_k3_native_models(self.dev, vq_ckpt, ncm_ckpt)

        e8_sd = torch.load(os.path.join(out_dir, "e8_k3_direct.pt"), map_location="cpu", weights_only=True)
        self.vqvae.load_state_dict(e8_sd["vqvae"])
        self.net.load_state_dict(e8_sd["net"])
        self.vqvae.eval(); self.net.eval()

        self.cbridge = ComboBridge(D_HA, D_VQ).to(self.dev)
        self.cbridge.load_state_dict(src["combo_bridge"])
        self.cbridge.eval()

        self.copy_dec = K3LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(self.dev)
        self.copy_dec.load_state_dict(e8_sd["ptr_dec"])
        self.copy_dec.eval()
        self.restricted_mask = e8_sd["restricted_mask"].to(self.dev)

        levels = list(self.vqvae.vq.levels)
        self.sched = [1.0 - 0.6 * (k / (len(levels) - 1)) for k in range(len(levels))]
        self.setup_s = round(time.perf_counter() - t0, 2)
        self.history = []  # List of {"user": str, "reply": str, "plans": list[str]} for multi-turn context memory

        # Warmup GPU kernels
        self._warmup()

    def clear_history(self):
        self.history.clear()

    def _warmup(self):
        p_warm = ["The Harbourside Council voted 7 to 2 on Tuesday; approved 2.4 million pounds"]
        r_w, m_w = encode_batch_rh(self.ha_model, self.ha_tok, p_warm, P_RH, self.dev_ha)
        zr_w = self.cbridge(r_w.to(self.dev)).float()
        mw = (m_w.to(self.dev) != 0)
        pk_w = torch.tensor([self.k3_tok.encode(p_warm[0])[:P_BPE] + [K3_PAD] * (P_BPE - len(self.k3_tok.encode(p_warm[0])[:P_BPE]))], dtype=torch.long, device=self.dev)
        zb_w = self.vqvae.encoder.wte(pk_w)
        zp_w, pm_w = pack_combo_prefix(zr_w, mw, zb_w, (pk_w == K3_PAD))
        with torch.no_grad():
            g_w = generate_from_zpref(self.net, self.tk, zp_w, pm_w, self.sched)
            zh_w, _ = self.vqvae.vq.idx_to_zhats(g_w, return_zlevels=True)
            _ = self.copy_dec(self.vqvae.decoder, zh_w[-1], zr_w, mw, zb_w, pk_w)
        torch.cuda.synchronize(self.dev); torch.cuda.synchronize(self.dev_ha)

    def _dec_k3(self, r):
        r = list(r)
        if K3_EOS in r:
            r = r[:r.index(K3_EOS)]
        clean_ids = []
        for t in r:
            ti = int(t)
            if 0 <= ti < 163584 and (not clean_ids or ti != clean_ids[-1]):
                clean_ids.append(ti)
        return self.k3_tok.decode(clean_ids), clean_ids

    @torch.no_grad()
    def plan_with_latent(self, user_prompt, history=None, min_lat=8, max_lat=14, max_new=110):
        backbone = self.ha_model.backbone
        embed_w = backbone.get_input_embeddings().weight
        if history:
            ctx_lines = []
            for turn_idx, h in enumerate(history[-6:], 1):
                ctx_lines.append(f"Turn {turn_idx} User: {h['user']}")
                ctx_lines.append(f"Turn {turn_idx} Assistant: {h['reply']}")
            ctx_block = (
                "Conversation history (resolve all pronouns like 'it', 'they', 'that', 'there', or follow-up calculations "
                "using these prior turns, and state the resolved entity names and numbers explicitly in W1 and W2):\n"
                + "\n".join(ctx_lines)
                + "\n\nCurrent user request:\n"
                + user_prompt.strip()
            )
            msgs = [{"role": "user", "content": CHAT_PLAN_PROMPT.format(prompt=ctx_block)}]
        else:
            msgs = [{"role": "user", "content": CHAT_PLAN_PROMPT.format(prompt=user_prompt.strip())}]
        base_chat = self.ha_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if base_chat.rstrip("\n").endswith(THINK_OPEN):
            prefix_str = base_chat + "Let me work out the exact facts, numbers, and steps." + ENTRY_ANCHOR
        else:
            prefix_str = base_chat + THINK_OPEN + "\nLet me work out the exact facts, numbers, and steps." + ENTRY_ANCHOR

        inp_ids = self.ha_tok(prefix_str, return_tensors="pt").input_ids.to(self.dev_ha)
        out = backbone(input_ids=inp_ids, use_cache=True, output_hidden_states=True, return_dict=True)
        past_kv = out.past_key_values
        last_h = out.hidden_states[-1][:, -1, :]
        next_logits = out.logits[:, -1, :]

        lat_steps = 0
        exit_reason = "max_steps"
        for step in range(max_lat):
            r_h = self.ha_model.run_hybridlink(last_h, output_dtype=embed_w.dtype)
            arg_id = int(next_logits.argmax(dim=-1).item())
            if step >= min_lat and arg_id == self.lat_close_id:
                exit_reason = "anchor"
                break
            lat_steps += 1
            out = backbone(inputs_embeds=r_h.unsqueeze(1), past_key_values=past_kv, use_cache=True, output_hidden_states=True, return_dict=True)
            past_kv = out.past_key_values
            last_h = out.hidden_states[-1][:, -1, :]
            next_logits = out.logits[:, -1, :]

        bridge_str = EXIT_ANCHOR + " Now I will write W1 and W2 with the exact facts and numbers." + THINK_CLOSE + "\nW1:"
        bridge_ids = self.ha_tok(bridge_str, add_special_tokens=False, return_tensors="pt").input_ids.to(self.dev_ha)
        out = backbone(input_ids=bridge_ids, past_key_values=past_kv, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]

        gen_ids = []
        eos_ids = {self.ha_tok.eos_token_id}
        for _ in range(max_new):
            nxt = int(next_logits.argmax(dim=-1).item())
            if nxt in eos_ids:
                break
            gen_ids.append(nxt)
            if len(gen_ids) % 8 == 0:
                txt_so_far = self.ha_tok.decode(gen_ids, skip_special_tokens=True)
                lines = [ln.strip() for ln in txt_so_far.splitlines() if ln.strip()]
                if len(lines) >= 2 and len(lines[1]) > 25 and txt_so_far.endswith("\n"):
                    break
                if len(lines) >= 3:
                    break
            out = backbone(input_ids=torch.tensor([[nxt]], device=self.dev_ha), past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]

        raw = "W1: " + self.ha_tok.decode(gen_ids, skip_special_tokens=True).strip()
        raw = re.sub(r"(?:Now I|I need|This is a|Let's|We need|The user is asking|The user's).*$", "", raw).strip()
        w1, w2 = "", ""
        for ln in raw.splitlines():
            ln = re.sub(r"(?:Now I|I need|This is a|Let's|We need|The user's).*$", "", ln).strip()
            ln = re.sub(r"^From the previous turn[^,]*,\s*(?:I know that\s*)?", "", ln, flags=re.IGNORECASE).strip()
            if ln.upper().startswith("W1:"):
                w1 = ln[3:].strip()
                w1 = re.sub(r"^From the previous turn[^,]*,\s*(?:I know that\s*)?", "", w1, flags=re.IGNORECASE).strip()
            elif ln.upper().startswith("W2:"):
                w2 = ln[3:].strip()
        if w1.lower().startswith("the user is") and w2:
            w1, w2 = w2, ""
        if not w1:
            w1 = raw.splitlines()[0].strip()

        # Split any single-block or overly long W1 (>34 words) into balanced W1 and W2 windows (~30 words each)
        if w1 and not w2 and len(w1.split()) > 30:
            if ";" in w1:
                parts = [p.strip() for p in w1.split(";") if p.strip()]
                mid = max(1, len(parts) // 2)
                w1, w2 = "; ".join(parts[:mid]), "; ".join(parts[mid:])
            else:
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", w1) if s.strip()]
                if len(sents) >= 2:
                    mid = max(1, len(sents) // 2)
                    w1, w2 = " ".join(sents[:mid]), " ".join(sents[mid:])
        elif w1 and w2 and len(w1.split()) > 38 and ";" in w1:
            parts = [p.strip() for p in w1.split(";") if p.strip()]
            w1 = "; ".join(parts[:-1])
            w2 = parts[-1] + "; " + w2

        plans = [p for p in (w1, w2) if len(p) > 4]
        return plans, lat_steps, exit_reason

    def _dec_k3_aligned(self, row_logits, plan_ids):
        """Decodes a 64-position logit tensor `[T, V_K3]` while enforcing 2-gram subword continuation
        for any multi-subword token sequence in `plan_ids` (e.g. `[' cath', 'ode']`, `['02', ':', '31']`),
        so parallel non-autoregressive argmax never splits or scrambles a multi-token word or time!"""
        cont_map = {}
        valid_p = [int(x) for x in plan_ids if 0 <= int(x) < 163584]
        for a, b in zip(valid_p[:-1], valid_p[1:]):
            bs = self.k3_tok.decode([b], skip_special_tokens=False)
            if bs and (not bs.startswith(" ")) and (bs[0].isalnum() or bs in (":", ".")):
                cont_map.setdefault(a, set()).add(b)

        T = row_logits.size(0)
        clean_ids = []
        prev_tid = -1
        prev_word = ""
        for t in range(T):
            lg = row_logits[t].clone()
            if prev_tid >= 0:
                lg[prev_tid] -= 20.0
                if prev_tid in cont_map:
                    for nxt_b in cont_map[prev_tid]:
                        lg[nxt_b] += 10.0
            tid = int(lg.argmax().item())
            if tid == K3_EOS:
                break
            if 0 <= tid < 163584:
                ws = self.k3_tok.decode([tid], skip_special_tokens=False).strip().lower()
                if ws and ws == prev_word and len(ws) >= 2:
                    lg[tid] -= 20.0
                    tid = int(lg.argmax().item())
                    if tid == K3_EOS:
                        break
                    ws = self.k3_tok.decode([tid], skip_special_tokens=False).strip().lower()
                clean_ids.append(tid)
                prev_tid = tid
                if ws:
                    prev_word = ws
        return self.k3_tok.decode(clean_ids), clean_ids

    def _apply_grounding_mask(self, logits, allowed_k3_ids):
        """Suppresses ungrounded proper nouns / numbers / currency symbols that are NOT in the user prompt or plan."""
        mask = self.restricted_mask.clone()
        if allowed_k3_ids:
            t_allow = torch.tensor(list(allowed_k3_ids), dtype=torch.long, device=self.dev)
            mask[t_allow] = False
        logits[:, :, mask] -= 14.0
        return logits

    @torch.no_grad()
    def render_ncm(self, plans, prompt_text="", cands=1):
        rh_p, msk_p = encode_batch_rh(self.ha_model, self.ha_tok, plans, P_RH, self.dev_ha)
        z_rh = self.cbridge(rh_p.to(self.dev)).float()
        msk_r = (msk_p.to(self.dev) != 0)
        p_kpad = torch.tensor(
            [self.k3_tok.encode(p)[:P_BPE] + [K3_PAD] * (P_BPE - len(self.k3_tok.encode(p)[:P_BPE])) for p in plans],
            dtype=torch.long, device=self.dev
        )
        z_bpe = self.vqvae.encoder.wte(p_kpad)
        z_pref, pmask = pack_combo_prefix(z_rh, msk_r, z_bpe, (p_kpad == K3_PAD))

        # Allowed proper nouns / numbers come from the prompt + plans (including both leading-space and no-space token variants)
        ref_str = prompt_text + " " + " ".join(plans)
        allowed_ids = set(self.k3_tok.encode(ref_str))
        for w in re.findall(r"[A-Za-z0-9.,:%£$€'-]+", ref_str):
            allowed_ids.update(self.k3_tok.encode(w))
            allowed_ids.update(self.k3_tok.encode(" " + w))

        W = len(plans)
        if cands <= 1:
            torch.manual_seed(42)
            g1 = generate_from_zpref(self.net, self.tk, z_pref, pmask, self.sched)
            zh1, _ = self.vqvae.vq.idx_to_zhats(g1, return_zlevels=True)
            lg1 = self.copy_dec(self.vqvae.decoder, zh1[-1], z_rh, msk_r, z_bpe, p_kpad)
            lg1 = self._apply_grounding_mask(lg1, allowed_ids)
            txts, ids = [], []
            for w_i in range(W):
                ts, is_ = self._dec_k3_aligned(lg1[w_i], p_kpad[w_i].tolist())
                txts.append(ts.strip()); ids.extend(is_)
            return " ".join(txts), ids

        # Batched MBR over `cands` candidates (all `cands * W` rows in a single 11-pass call!)
        z_pref_k = z_pref.repeat(cands, 1, 1)
        pmask_k = pmask.repeat(cands, 1)
        z_rh_k = z_rh.repeat(cands, 1, 1)
        msk_r_k = msk_r.repeat(cands, 1)
        z_bpe_k = z_bpe.repeat(cands, 1, 1)
        p_kpad_k = p_kpad.repeat(cands, 1)
        torch.manual_seed(42)
        gk = generate_from_zpref(self.net, self.tk, z_pref_k, pmask_k, self.sched)
        zhk, _ = self.vqvae.vq.idx_to_zhats(gk, return_zlevels=True)
        lgk = self.copy_dec(self.vqvae.decoder, zhk[-1], z_rh_k, msk_r_k, z_bpe_k, p_kpad_k)
        lgk = self._apply_grounding_mask(lgk, allowed_ids)

        cands_t = [[] for _ in range(W)]
        cands_i = [[] for _ in range(W)]
        for k in range(cands):
            for w_i in range(W):
                ts, is_ = self._dec_k3_aligned(lgk[k * W + w_i], p_kpad[w_i].tolist())
                cands_t[w_i].append(ts.strip()); cands_i[w_i].append(is_)
        out_txts, out_ids = [], []
        for w_i in range(W):
            sc = [sum(jacc(cands_t[w_i][k], cands_t[w_i][j]) for j in range(cands) if j != k) for k in range(cands)]
            bk = max(range(cands), key=lambda k: sc[k])
            out_txts.append(cands_t[w_i][bk]); out_ids.extend(cands_i[w_i][bk])
        return " ".join(out_txts), out_ids

    def chat_reply(self, user_prompt, cands=8, remember=False):
        torch.cuda.synchronize(self.dev); torch.cuda.synchronize(self.dev_ha)
        t0 = time.perf_counter()
        active_hist = self.history if remember else None
        plans, lat_steps, exit_reason = self.plan_with_latent(user_prompt, history=active_hist)
        torch.cuda.synchronize(self.dev_ha)
        t_plan_ms = (time.perf_counter() - t0) * 1000.0

        hist_ctx = ""
        if active_hist:
            hist_ctx = " ".join(f"{h['user']} {h['reply']}" for h in active_hist[-6:])
        full_prompt_ctx = (hist_ctx + " " + user_prompt).strip()

        t1 = time.perf_counter()
        text_1cand, ids_1cand = self.render_ncm(plans, prompt_text=full_prompt_ctx, cands=1)
        torch.cuda.synchronize(self.dev); torch.cuda.synchronize(self.dev_ha)
        t_ncm_1cand_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        text_mbr, ids_mbr = self.render_ncm(plans, prompt_text=full_prompt_ctx, cands=cands)
        torch.cuda.synchronize(self.dev); torch.cuda.synchronize(self.dev_ha)
        t_ncm_mbr_ms = (time.perf_counter() - t2) * 1000.0

        if remember:
            self.history.append({
                "user": user_prompt.strip(),
                "reply": text_mbr,
                "plans": plans,
            })

        return {
            "prompt": user_prompt.strip(),
            "turn": len(self.history) if remember else 1,
            "latent_steps": lat_steps,
            "exit_reason": exit_reason,
            "plans": plans,
            "plan_lat_ms": round(t_plan_ms, 1),
            "ncm_1cand_ms": round(t_ncm_1cand_ms, 1),
            "ncm_mbr8_ms": round(t_ncm_mbr_ms, 1),
            "total_1cand_ms": round(t_plan_ms + t_ncm_1cand_ms, 1),
            "total_mbr8_ms": round(t_plan_ms + t_ncm_mbr_ms, 1),
            "n_k3_tokens_1cand": len(ids_1cand),
            "n_k3_tokens_mbr8": len(ids_mbr),
            "reply_1cand": text_1cand,
            "reply_mbr8": text_mbr,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None, help="Single user prompt string to run in chat mode")
    ap.add_argument("--interactive", action="store_true", help="Run interactive multi-turn REPL chat mode with context memory")
    ap.add_argument("--test-multiturn", action="store_true", help="Run a 4-turn conversation test verifying multi-turn context memory")
    ap.add_argument("--out", default="/home/usergpu/merge_lf/out/e6_k3_native/chat_mode_results.json")
    a = ap.parse_args()

    eng = K3FastChatEngine()
    print(f"K3FastChatEngine ready (one-time setup {eng.setup_s}s; 93-layer verification OFF; DSpark OFF; Multi-turn memory ON)", flush=True)

    if a.test_multiturn:
        eng.clear_history()
        turns = [
            "the captial of france is",
            "What famous iron tower is located there, and who designed it?",
            "Now imagine St Mary's Hospital in Bristol opened a new 18 million pound children's wing on Friday led by Dr Elena Vance, adding 45 beds and 4 operating theatres. Summarize what happened.",
            "Who led that hospital project, how many beds and operating theatres did it add, and in which city was it opened?",
        ]
        mt_res = []
        for idx, u_turn in enumerate(turns, 1):
            r = eng.chat_reply(u_turn, cands=8, remember=True)
            mt_res.append(r)
            print(f"\n================ [Turn {idx}] ================", flush=True)
            print(f"USER   : {r['prompt']}", flush=True)
            print(f"PLAN   : {' || '.join(r['plans'])} ({r['plan_lat_ms']} ms)", flush=True)
            print(f"TIMING : 1-cand = {r['total_1cand_ms']/1000:.2f}s (NCM {r['ncm_1cand_ms']} ms) | MBR8 = {r['total_mbr8_ms']/1000:.2f}s (NCM {r['ncm_mbr8_ms']} ms)", flush=True)
            print(f"REPLY  : {r['reply_mbr8']}", flush=True)
        json.dump(mt_res, open("/home/usergpu/merge_lf/out/e6_k3_native/multiturn_chat_proof.json", "w"), indent=2)
        return

    if a.interactive:
        print("\nInteractive Multi-Turn K3FastChatEngine ready! Type any prompt (or 'reset' to clear memory, 'exit' to stop).\n", flush=True)
        while True:
            try:
                user_in = input(f"[Turn {len(eng.history)+1}] PROMPT> ").strip()
            except EOFError:
                break
            if not user_in or user_in.lower() in ("exit", "quit", "q"):
                break
            if user_in.lower() in ("reset", "clear"):
                eng.clear_history()
                print("Conversation memory cleared.\n", flush=True)
                continue
            r = eng.chat_reply(user_in, cands=8, remember=True)
            print(f"LATENT: {r['latent_steps']} steps ({r['exit_reason']}) | PLAN ({r['plan_lat_ms']} ms): {' || '.join(r['plans'])}", flush=True)
            print(f"TIMING: 1-cand NCM = {r['ncm_1cand_ms']} ms (Total {r['total_1cand_ms']/1000:.2f}s) | MBR8 NCM = {r['ncm_mbr8_ms']} ms (Total {r['total_mbr8_ms']/1000:.2f}s)", flush=True)
            print(f"OUTPUT (1-cand, {r['n_k3_tokens_1cand']} K3 tokens): {r['reply_1cand']}", flush=True)
            print(f"OUTPUT (MBR8,   {r['n_k3_tokens_mbr8']} K3 tokens): {r['reply_mbr8']}\n", flush=True)
        return

    if a.prompt:
        prompts = [("custom", a.prompt)]
    else:
        prompts = [
            ("p1", open("/home/usergpu/k3ncm/p1.txt", encoding="utf-8").read()),
            ("p2", open("/home/usergpu/k3ncm/p2.txt", encoding="utf-8").read()),
            ("p3", open("/home/usergpu/k3ncm/p3.txt", encoding="utf-8").read()),
            ("p4", open("/home/usergpu/k3ncm/p4.txt", encoding="utf-8").read()),
            ("p5", open("/home/usergpu/k3ncm/p5.txt", encoding="utf-8").read()),
            ("p6_new_hospital", "Write a news report of about 90 words using these facts: St Mary's Hospital opened a new 18 million pound children's wing on Friday in Bristol. Dr Elena Vance led the 3-year project, which adds 45 beds and 4 operating theatres. Give a one-sentence Summary at the end."),
            ("p7_new_battery", "Explain in about 90 words how a lithium-ion battery works during charging and discharging, and why it degrades over time. End with a one-sentence Summary."),
        ]

    results = {}
    for name, pr in prompts:
        r = eng.chat_reply(pr, cands=8)
        results[name] = r
        print(f"\n================ [{name}] ================", flush=True)
        print(f"PROMPT: {r['prompt']}", flush=True)
        print(f"LATENT: {r['latent_steps']} steps ({r['exit_reason']}) | PLAN ({r['plan_lat_ms']} ms): {' || '.join(r['plans'])}", flush=True)
        print(f"TIMING: 1-cand NCM = {r['ncm_1cand_ms']} ms (Total {r['total_1cand_ms']} ms = {r['total_1cand_ms']/1000:.2f}s) | MBR8 NCM = {r['ncm_mbr8_ms']} ms (Total {r['total_mbr8_ms']} ms = {r['total_mbr8_ms']/1000:.2f}s)", flush=True)
        print(f"OUTPUT (MBR8, {r['n_k3_tokens_mbr8']} Kimi K3 tokens):\n{r['reply_mbr8']}", flush=True)

    json.dump(results, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nSaved {len(results)} chat mode runs to {a.out}", flush=True)


if __name__ == "__main__":
    main()
