"""Cycle 57: 8-GPU Clubbed Parallel-Prompt Corpus Generation & Multi-GPU NCM + VQ-VAE + Pointer Decoder Training.

Uses all 8 RTX 3090 GPUs (`cuda:0..7`) in two phases:
1. Phase 1 (4 NVLink Worker Pairs `(0,2)`, `(1,3)`, `(4,6)`, `(5,7)` in parallel):
   - Clubs diverse single-turn & multi-turn prompts (timetables, study schedules for parents with infants, competitive exam prep
     like TNPSC/UPSC, coding/engineering, science, medical, math, greetings, clarifications, and follow-ups) into parallel batches
     and runs `HybridAgent-8B-Agentic` (`<latent>` rollout + full multi-window generation) on all 4 GPU pairs simultaneously.
   - Splits all generated responses into `<= 44` Kimi-K3-token windows (both dense semicolon plans and direct sentence windows)
     and extracts `HybridLink` continuous representations `R(h)` (`[N, P_RH, 4096]`) in parallel across all 4 pairs.
2. Phase 2 (All 8 GPUs `cuda:0..7` Multi-GPU Parallel Training via `torch.nn.DataParallel` / multi-GPU shards):
   - Trains `vqvae` (`encoder` + `decoder` in Kimi K3's 163,840 vocabulary) with clubbed multi-GPU batches (`B = 8 * 48 = 384` windows/step)
     across general XSum windows + clubbed HybridAgent multi-turn/timetable/domain windows.
   - Encodes all training windows into VQ codes across all 8 GPUs in parallel.
   - Jointly trains `net` (NCM Transformer) + `ptr_dec` (`K3LexicalPointerDecoder`) across all 8 GPUs with clubbed multi-window batches
     (`B = 8 * 32 = 256` windows/step) on both semicolon clause skeletons and direct natural/structured sentences.
"""
import json, math, os, pickle, re, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers import AutoTokenizer

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
ROOT = os.environ["NCM_REPO"]
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from k3_tokenizer import K3Tokenizer
from train_k3_native_all import (
    load_k3_native_models, eval_vqvae_on_k3_ids, V_K3, D_VQ, D_HA, L_WMT, P_RH, P_BPE, P_COMBO, K3_EOS, K3_PAD
)
from train_wmt14_k3 import corrupt_codes
from train_combo_prefix import ComboBridge, pack_combo_prefix
from train_e8_direct import DOMAIN_PLAN_TARGET_PAIRS, sentence_to_dense_plan


# Curated high-value structured, timetable, multi-turn, and domain windows (guaranteed 100% coverage)
STRUCTURED_CHAT_PAIRS = [
    # TNPSC / competitive exam + 2-hour daily schedule with 2-month-old infant
    (
        "I understand that caring for a 2-month-old baby leaves limited time, so splitting your 2 hours into focused 30 to 45 minute blocks during nap times works best.",
        "I understand that caring for a 2-month-old baby leaves limited time, so splitting your 2 hours into focused 30 to 45 minute blocks during nap times works best."
    ),
    (
        "Block 1 (45 mins, Morning Nap): Study core syllabus topics such as Indian Polity, History, or Economy when your mind is fresh.",
        "Block 1 (45 mins, Morning Nap): Study core syllabus topics such as Indian Polity, History, or Economy when your mind is fresh."
    ),
    (
        "Block 2 (45 mins, Afternoon Nap): Practice Aptitude, Mental Ability, and 25 previous-year TNPSC multiple-choice questions.",
        "Block 2 (45 mins, Afternoon Nap): Practice Aptitude, Mental Ability, and 25 previous-year TNPSC multiple-choice questions."
    ),
    (
        "Block 3 (30 mins, Evening or Night): Review short notes, current affairs, and Tamil/General Studies flashcards before rest.",
        "Block 3 (30 mins, Evening or Night): Review short notes, current affairs, and Tamil/General Studies flashcards before rest."
    ),
    (
        "Tip: Keep audio notes or flashcards on your phone so you can revise lightly while feeding or soothing your baby, and prioritize your sleep.",
        "Tip: Keep audio notes or flashcards on your phone so you can revise lightly while feeding or soothing your baby, and prioritize your sleep."
    ),
    # Weekend & flexible study adjustments
    (
        "On weekends, if family members can watch the baby for an extra hour, use 60 minutes for a timed mock test and 30 minutes to analyze mistakes.",
        "On weekends, if family members can watch the baby for an extra hour, use 60 minutes for a timed mock test and 30 minutes to analyze mistakes."
    ),
    # TNPSC exam overview & books
    (
        "For TNPSC Group 1, Group 2, and Group 4 preparation, start with the Tamil Nadu Samacheer Kalvi school textbooks from Class 6 to Class 12.",
        "For TNPSC Group 1, Group 2, and Group 4 preparation, start with the Tamil Nadu Samacheer Kalvi school textbooks from Class 6 to Class 12."
    ),
    (
        "Focus on General Tamil or General English, Indian Polity (Laxmikanth), Unit 8 and Unit 9 (Tamil Nadu History, Administration, and Development), and daily Aptitude practice.",
        "Focus on General Tamil or General English, Indian Polity (Laxmikanth), Unit 8 and Unit 9 (Tamil Nadu History, Administration, and Development), and daily Aptitude practice."
    ),
    # Conversational greetings, thanks, clarifications
    (
        "You're welcome! If you have any more questions or need adjustments to your study schedule, feel free to ask. Good luck with your preparation!",
        "You're welcome! If you have any more questions or need adjustments to your study schedule, feel free to ask. Good luck with your preparation!"
    ),
    (
        "Hello! How can I assist you today?",
        "Hello! How can I assist you today?"
    ),
    (
        "I'm here to help with any questions or tasks you have. Please let me know what topic, schedule, or problem you'd like to work on.",
        "I'm here to help with any questions or tasks you have. Please let me know what topic, schedule, or problem you'd like to work on."
    ),
    (
        "Paris.",
        "Paris."
    ),
    (
        "The capital of France is Paris.",
        "The capital of France is Paris."
    ),
    (
        "The famous iron tower located in Paris is the Eiffel Tower. It was designed by Gustave Eiffel.",
        "The famous iron tower located in Paris is the Eiffel Tower. It was designed by Gustave Eiffel."
    ),
    # Photosynthesis full windows
    (
        "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose.",
        "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose."
    ),
    (
        "This process occurs in the chloroplasts of plant cells, where chlorophyll captures sunlight to convert carbon dioxide and water into glucose and oxygen.",
        "This process occurs in the chloroplasts of plant cells, where chlorophyll captures sunlight to convert carbon dioxide and water into glucose and oxygen."
    ),
    (
        "The overall equation for photosynthesis is: CO2 + H2O + light energy -> C6H12O6 + O2. It produces oxygen and forms the base of Earth's food chain.",
        "The overall equation for photosynthesis is: CO2 + H2O + light energy -> C6H12O6 + O2. It produces oxygen and forms the base of Earth's food chain."
    ),
]


# Diverse single-turn and multi-turn prompts to run in clubbed parallel batches across all 4 NVLink pairs
CLUBBED_PROMPT_BANK = [
    # Study schedules & exam prep
    [
        {"user": "I am preparing for TNPSC exams. Where should I start?", "reply": "Start with the Tamil Nadu Samacheer Kalvi textbooks (Classes 6 to 12) and previous-year question papers."},
        {"user": "thank you", "reply": "You're welcome! Let me know if you need a study plan or book list."},
        {"user": "can you schedule a time table for my preparation i have a 2months old baby and infant i can prepare only for 2 hrs"},
    ],
    [
        {"user": "Can you make a 2-hour daily study timetable for a working parent preparing for competitive exams?"},
    ],
    [
        {"user": "How should I divide 3 hours a day to study for civil service prelims (Polity, History, Economy, and Aptitude)?"},
    ],
    [
        {"user": "Give me a 4-week revision timetable for math and science exams with 2 hours per day."},
    ],
    [
        {"user": "What are the best strategies to study effectively in short 30-minute nap windows when caring for a newborn baby?"},
    ],
    # Multi-turn follow-ups & clarifications
    [
        {"user": "hi"},
    ],
    [
        {"user": "hi", "reply": "Hello! How can I assist you today?"},
        {"user": "I did not ask that question"},
    ],
    [
        {"user": "thank you so much for your help"},
    ],
    [
        {"user": "the captial of france is", "reply": "Paris."},
        {"user": "What famous iron tower is located there, and who designed it?"},
    ],
    [
        {"user": "What is the largest planet in our solar system?", "reply": "The largest planet in our solar system is Jupiter."},
        {"user": "How many moons does it have and what are its four largest moons called?"},
    ],
    # Science, engineering, computing, medicine, math
    [{"user": "Explain how photosynthesis works in plants and why it matters for life on Earth."}],
    [{"user": "Why does the sky look blue during the day and red at sunset?"}],
    [{"user": "How do lithium-ion batteries store energy and why do they degrade over time?"}],
    [{"user": "What is the tallest mountain in the world, where is it located, and how high is it?"}],
    [{"user": "What happened on the Apollo 13 mission and how did the crew return safely?"}],
    [{"user": "Explain the difference between TCP and UDP in computer networking and when to use each."}],
    [{"user": "How does a binary search algorithm work and what is its time complexity?"}],
    [{"user": "What is the difference between DNA and RNA in human cells?"}],
    [{"user": "Explain how vaccines train the human immune system to fight infections."}],
    [{"user": "Give me a beginner 3-day-per-week home workout schedule that takes 30 minutes per session."}],
    [{"user": "How can I manage lower back posture when working at a desk all day?"}],
    [{"user": "Write a step-by-step morning routine for productivity and energy."}],
    [{"user": "Explain how transformers and self-attention work in modern large language models."}],
    [{"user": "What causes ocean tides on Earth and why are there two high tides each day?"}],
]


SYS_GEN_PROMPT = (
    "You are Clover AI, a helpful, empathetic, and direct AI assistant. "
    "Answer the user's request completely and directly in clear, natural English. "
    "When the user asks for a timetable, schedule, study plan, or step-by-step guide, "
    "write out the actual concrete schedule (for example: Block 1 / Block 2 / Block 3 with durations and topics) "
    "in 3 to 5 concise lines, not just an introductory sentence."
)


def split_text_to_k3_windows(text: str, k3_tok: K3Tokenizer, max_bpe: int = 44):
    """Splits a single-paragraph or multi-line response into windows that each fit within `max_bpe` Kimi K3 tokens."""
    text = re.sub(r"^W[1-9]:\s*", "", text.strip(), flags=re.MULTILINE)
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    windows = []
    for ln in raw_lines:
        ln = re.sub(r"^W[1-9]:\s*", "", ln, flags=re.IGNORECASE).strip()
        ln = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", ln).strip()
        ln = re.sub(r"\*\*", "", ln).strip()
        if not ln or ln.lower().startswith(("okay, so ", "the user ", "hmm", "let me ", "now i ")):
            continue
        if len(k3_tok.encode(ln)) <= max_bpe:
            windows.append(ln)
        else:
            sents = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", ln) if s.strip()]
            cur = ""
            for s in sents:
                cand = (cur + " " + s).strip() if cur else s
                if len(k3_tok.encode(cand)) <= max_bpe:
                    cur = cand
                else:
                    if cur:
                        windows.append(cur)
                    if len(k3_tok.encode(s)) <= max_bpe:
                        cur = s
                    else:
                        # Split long clause by words
                        words = s.split()
                        w_cur = ""
                        for w in words:
                            w_cand = (w_cur + " " + w).strip() if w_cur else w
                            if len(k3_tok.encode(w_cand)) <= max_bpe:
                                w_cur = w_cand
                            else:
                                if w_cur:
                                    windows.append(w_cur)
                                w_cur = w
                        cur = w_cur
            if cur:
                windows.append(cur)
    return [w for w in windows if len(w) >= 2]


def _worker_generate_and_encode(rank: int, gpu_ncm: int, gpu_ha: int, conv_slice: list, extra_plans: list, out_path: str):
    """Runs on one NVLink GPU pair `(cuda:gpu_ncm, cuda:gpu_ha)`:
    1. Clubs prompts into parallel batches on `cuda:gpu_ha` and runs `HybridAgent-8B-Agentic` (`<latent>` rollout + generation).
    2. Splits generated responses into `<= 44`-token windows.
    3. Encodes all assigned plan strings (`extra_plans` + generated windows) into `R(h)` (`[N, P_RH, 4096]`) in parallel batches.
    """
    import torch
    from hybrid_agent.model import load_hybrid_agent
    from hybrid_agent.constants import ENTRY_ANCHOR, EXIT_ANCHOR, THINK_OPEN, THINK_CLOSE
    from train_latent_bridge import encode_batch_rh

    dev_ha = torch.device(f"cuda:{gpu_ha}")
    k3_tok = K3Tokenizer()
    ha_model, ha_tok = load_hybrid_agent(
        "togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev_ha, require_hybridlink=True
    )
    backbone = ha_model.backbone
    embed_w = backbone.get_input_embeddings().weight

    # Run clubbed parallel generation across `conv_slice`
    prefix_strs = []
    for conv in conv_slice:
        msgs = [{"role": "system", "content": SYS_GEN_PROMPT}]
        for turn in conv[:-1]:
            msgs.append({"role": "user", "content": turn["user"]})
            msgs.append({"role": "assistant", "content": turn["reply"]})
        msgs.append({"role": "user", "content": conv[-1]["user"]})
        base_chat = ha_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if base_chat.rstrip("\n").endswith(THINK_OPEN):
            s = base_chat + "Let me think carefully and reply directly to the user." + ENTRY_ANCHOR
        else:
            s = base_chat + THINK_OPEN + "\nLet me think carefully and reply directly to the user." + ENTRY_ANCHOR
        prefix_strs.append(s)

    gen_windows = []
    if prefix_strs:
        ha_tok.padding_side = "left"
        batch_in = ha_tok(prefix_strs, padding=True, return_tensors="pt").to(dev_ha)
        ha_tok.padding_side = "right"
        B = len(prefix_strs)
        inp_ids = batch_in.input_ids
        attn_mask = batch_in.attention_mask
        pos_ids = attn_mask.long().cumsum(-1) - 1
        pos_ids.masked_fill_(attn_mask == 0, 0)

        with torch.no_grad():
            out = backbone(input_ids=inp_ids, attention_mask=attn_mask, position_ids=pos_ids, use_cache=True, output_hidden_states=True, return_dict=True)
            past_kv = out.past_key_values
            last_h = out.hidden_states[-1][:, -1, :]
            for _ in range(12):
                r_h = ha_model.run_hybridlink(last_h, output_dtype=embed_w.dtype)
                attn_mask = torch.cat([attn_mask, torch.ones((B, 1), dtype=attn_mask.dtype, device=dev_ha)], dim=1)
                pos_step = (attn_mask.sum(dim=1, keepdim=True) - 1).long()
                out = backbone(inputs_embeds=r_h.unsqueeze(1), attention_mask=attn_mask, position_ids=pos_step, past_key_values=past_kv, use_cache=True, output_hidden_states=True, return_dict=True)
                past_kv = out.past_key_values
                last_h = out.hidden_states[-1][:, -1, :]

            bridge_str = EXIT_ANCHOR + " Now I will write my complete, direct response to the user." + THINK_CLOSE + "\nW1:"
            bridge_ids = ha_tok(bridge_str, add_special_tokens=False, return_tensors="pt").input_ids.to(dev_ha).expand(B, -1)
            Kb = bridge_ids.size(1)
            base_pos = attn_mask.sum(dim=1, keepdim=True).long()
            bridge_pos = base_pos + torch.arange(Kb, device=dev_ha).unsqueeze(0)
            attn_mask = torch.cat([attn_mask, torch.ones((B, Kb), dtype=attn_mask.dtype, device=dev_ha)], dim=1)
            out = backbone(input_ids=bridge_ids, attention_mask=attn_mask, position_ids=bridge_pos, past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]

            gen_ids = [[] for _ in range(B)]
            done = [False] * B
            eos_id = ha_tok.eos_token_id
            for _ in range(210):
                nxt = next_logits.argmax(dim=-1)
                for i in range(B):
                    ti = int(nxt[i].item())
                    if not done[i]:
                        if ti == eos_id:
                            done[i] = True
                        else:
                            gen_ids[i].append(ti)
                if all(done):
                    break
                attn_mask = torch.cat([attn_mask, torch.ones((B, 1), dtype=attn_mask.dtype, device=dev_ha)], dim=1)
                pos_step = (attn_mask.sum(dim=1, keepdim=True) - 1).long()
                out = backbone(input_ids=nxt.unsqueeze(1), attention_mask=attn_mask, position_ids=pos_step, past_key_values=past_kv, use_cache=True, return_dict=True)
                past_kv = out.past_key_values
                next_logits = out.logits[:, -1, :]

        for i in range(B):
            txt = "W1: " + ha_tok.decode(gen_ids[i], skip_special_tokens=True).strip()
            wins = split_text_to_k3_windows(txt, k3_tok, max_bpe=44)
            for w in wins:
                # Add both direct natural sentence window AND dense semicolon plan -> sentence pair
                gen_windows.append((w, w))
                dense_p = sentence_to_dense_plan(w)
                if dense_p and len(dense_p) >= 6 and dense_p != w:
                    gen_windows.append((dense_p, w))

    # Now encode `extra_plans` + `gen_windows` into `R(h)` on `dev_ha`
    all_pairs = extra_plans + gen_windows
    plans_only = [p for p, _ in all_pairs]
    rh_list, msk_list = [], []
    with torch.no_grad():
        for i in range(0, len(plans_only), 192):
            r, m = encode_batch_rh(ha_model, ha_tok, plans_only[i:i + 192], P_RH, dev_ha)
            rh_list.append(r.cpu().to(torch.bfloat16))
            msk_list.append(m.cpu().to(torch.bool))

    torch.save({
        "pairs": all_pairs,
        "gen_windows": gen_windows,
        "rh": torch.cat(rh_list, dim=0),
        "msk": torch.cat(msk_list, dim=0),
    }, out_path)
    print(f"[Worker {rank} | GPUs ({gpu_ncm},{gpu_ha})] Clubbed {len(conv_slice)} prompts -> {len(gen_windows)} windows; encoded {len(all_pairs)} total windows.", flush=True)


def main():
    t_start = time.perf_counter()
    mp.set_start_method("spawn", force=True)
    out_dir = "/home/usergpu/merge_lf/out/e6_k3_native"
    k3_tok = K3Tokenizer()

    # Build general XSum pairs (both semicolon plans AND direct natural sentences)
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    passages = pickle.load(open("/home/usergpu/ncm/data/xsum/passages.pickle", "rb"))
    summaries = pickle.load(open("/home/usergpu/ncm/data/xsum/reference_summaries.pickle", "rb"))

    general_pairs = []
    for p_ids, s_txt in zip(passages[:4000], summaries[:4000]):
        p_txt = " ".join(gpt2_tok.decode(p_ids, skip_special_tokens=True).split())
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", f"{s_txt.strip()} {p_txt}") if len(s.strip()) > 12]
        for n_s in (1, 2):
            if len(sents) < n_s:
                continue
            chunk_txt = " ".join(sents[:n_s])
            t_ids = k3_tok.encode(chunk_txt)
            if not (10 <= len(t_ids) <= 44):
                continue
            plan_txt = sentence_to_dense_plan(chunk_txt)
            if len(k3_tok.encode(plan_txt)) >= 6:
                general_pairs.append((plan_txt, chunk_txt))
            if len(general_pairs) % 3 == 0:
                # Include direct 1-to-1 natural sentence identity windows too
                general_pairs.append((chunk_txt, chunk_txt))
            if len(general_pairs) >= 5600:
                break
        if len(general_pairs) >= 5600:
            break

    # Combine curated domain + structured chat pairs (balanced 18x weight, NO single-sentence 120x skew!)
    curated_pairs = []
    for pl_txt, tg_txt in (DOMAIN_PLAN_TARGET_PAIRS + STRUCTURED_CHAT_PAIRS):
        if "capital of France is Paris. Summary:" in pl_txt:
            continue
        for w in split_text_to_k3_windows(tg_txt, k3_tok, max_bpe=44):
            curated_pairs.append((w, w))
            dp = sentence_to_dense_plan(w)
            if dp and dp != w:
                curated_pairs.append((dp, w))
        if len(k3_tok.encode(pl_txt)) <= 46 and len(k3_tok.encode(tg_txt)) <= 60:
            curated_pairs.append((pl_txt, tg_txt))

    all_base_pairs = general_pairs + (curated_pairs * 12)
    print(f"[Phase 1] Launching 4 parallel NVLink worker pairs across all 8 GPUs for clubbed prompt generation + R(h) encoding ({len(all_base_pairs)} base pairs + {len(CLUBBED_PROMPT_BANK)} clubbed conversations)...", flush=True)

    nvlink_pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
    shard_paths = [f"/tmp/e8_8gpu_shard_{r_idx}.pt" for r_idx in range(4)]
    if all(os.path.exists(sp) for sp in shard_paths):
        print("[Phase 1] Reusing existing 4 parallel NVLink worker shards from /tmp/e8_8gpu_shard_0..3.pt...", flush=True)
    else:
        procs = []
        for r_idx, (g_ncm, g_ha) in enumerate(nvlink_pairs):
            c_slice = CLUBBED_PROMPT_BANK[r_idx::4]
            p_slice = all_base_pairs[r_idx::4]
            sp = shard_paths[r_idx]
            p = mp.Process(target=_worker_generate_and_encode, args=(r_idx, g_ncm, g_ha, c_slice, p_slice, sp))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker exited with code {p.exitcode}")

    # Load and merge all 4 parallel shards; replicate generated clubbed windows 10x so real multi-turn/timetable windows are strongly represented
    merged_pairs = []
    merged_rh = []
    merged_msk = []
    total_gen_wins = 0
    for sp in shard_paths:
        sd = torch.load(sp, map_location="cpu", weights_only=False)
        pairs_s = sd["pairs"]
        rh_s = sd["rh"]
        msk_s = sd["msk"]
        n_gen = len(sd["gen_windows"])
        total_gen_wins += n_gen
        merged_pairs.extend(pairs_s)
        merged_rh.append(rh_s)
        merged_msk.append(msk_s)
        if n_gen > 0:
            # Repeat the newly generated clubbed windows 10x
            gen_rh = rh_s[-n_gen:].repeat(10, 1, 1)
            gen_msk = msk_s[-n_gen:].repeat(10, 1)
            merged_pairs.extend(sd["gen_windows"] * 10)
            merged_rh.append(gen_rh)
            merged_msk.append(gen_msk)

    rh_all = torch.cat(merged_rh, dim=0)
    msk_all = torch.cat(merged_msk, dim=0)
    print(f"[Phase 1 Complete] {total_gen_wins} clubbed HybridAgent windows generated; total training rows = {len(merged_pairs)} in {time.perf_counter() - t_start:.1f}s", flush=True)

    # Tokenize all `(plan, target)` pairs into Kimi K3 IDs
    pids_list, tids_list, tlen_list = [], [], []
    for pl_txt, tg_txt in merged_pairs:
        t_ids = k3_tok.encode(tg_txt)[:L_WMT - 1]
        p_ids = k3_tok.encode(pl_txt)[:P_BPE]
        if not t_ids:
            t_ids = [K3_EOS]
        if not p_ids:
            p_ids = t_ids[:P_BPE]
        t_pad = t_ids + [K3_EOS] + [K3_PAD] * (L_WMT - 1 - len(t_ids))
        p_pad = p_ids + [K3_PAD] * (P_BPE - len(p_ids))
        pids_list.append(p_pad)
        tids_list.append(t_pad)
        tlen_list.append(len(t_ids))

    all_pids = torch.tensor(pids_list, dtype=torch.long)
    all_tids = torch.tensor(tids_list, dtype=torch.long)

    # --- Phase 2A: Adapt VQ-VAE across all 8 GPUs (`cuda:0..7`) ---
    dev0 = torch.device("cuda:0")
    device_ids = list(range(torch.cuda.device_count()))
    print(f"[Phase 2A] Training VQ-VAE across {len(device_ids)} GPUs ({device_ids}) with clubbed batch size {len(device_ids) * 48}...", flush=True)

    vq_ckpt = os.path.join(out_dir, "k3_vqvae_adapted.pt")
    ncm_ckpt = os.path.join(out_dir, "e6_last.pt")
    cfg, tk, vqvae, net, src = load_k3_native_models(dev0, vq_ckpt, ncm_ckpt)
    e9_path = os.path.join(out_dir, "e9_k3_multimodal.pt")
    e8_prev_path = os.path.join(out_dir, "e8_k3_direct.pt")
    base_sd = torch.load(e9_path if os.path.exists(e9_path) else e8_prev_path, map_location="cpu", weights_only=True)
    vqvae.load_state_dict(base_sd["vqvae"])
    net.load_state_dict(base_sd["net"])

    for name, p in vqvae.named_parameters():
        p.requires_grad_("vq." not in name)
    vqvae.train(); vqvae.vq.eval()

    class VQLossStep(nn.Module):
        def __init__(self, vq_m):
            super().__init__()
            self.vqvae = vq_m

        def forward(self, b_inp):
            logits, vq_loss, _, _, _ = self.vqvae(b_inp)
            ce_loss = F.cross_entropy(logits.float().reshape(-1, V_K3), b_inp.reshape(-1))
            loss = ce_loss + 0.25 * vq_loss.mean()
            return loss.unsqueeze(0), ce_loss.detach().unsqueeze(0)

    dp_vqvae = nn.DataParallel(VQLossStep(vqvae).to(dev0), device_ids=device_ids)
    opt_vq = torch.optim.AdamW([p for p in vqvae.parameters() if p.requires_grad], lr=2.5e-5, betas=(0.9, 0.95), weight_decay=0.01)
    bs_vq = len(device_ids) * 48  # 384 windows per step across 8 GPUs
    for step in range(1, 51):
        ix = torch.randint(0, len(all_tids), (bs_vq,))
        b_inp = all_tids[ix].to(dev0, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_vec, ce_vec = dp_vqvae(b_inp)
            loss = loss_vec.mean()
            ce_loss = ce_vec.mean()
        opt_vq.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in vqvae.parameters() if p.requires_grad], 1.0)
        opt_vq.step()
        if step % 25 == 0 or step == 1:
            print(f"  [VQ-VAE 8-GPU] step {step}/50 | ce={ce_loss.item():.4f}", flush=True)

    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    ev_k3 = eval_vqvae_on_k3_ids(vqvae, k3_tok, dev0)
    print(f"[Phase 2A Complete] VQ-VAE p1..p4 accuracy: {ev_k3['overall_hit_total']} ({ev_k3['overall_acc']}%)", flush=True)

    # Encode all rows with updated VQ-VAE and ComboBridge across GPU 0
    cbridge = ComboBridge(D_HA, D_VQ).to(dev0)
    cbridge.load_state_dict(src["combo_bridge"])
    cbridge.eval()
    for p in cbridge.parameters():
        p.requires_grad_(False)

    idx_chunks, zrh_chunks = [], []
    with torch.no_grad():
        for i in range(0, len(all_tids), 512):
            idx_chunks.append(vqvae.encode(all_tids[i:i + 512].to(dev0)).cpu().to(torch.int16))
            zr = cbridge(rh_all[i:i + 512].to(dev0, dtype=torch.float32)).cpu().to(torch.bfloat16)
            zrh_chunks.append(zr)
    idx_all = torch.cat(idx_chunks, dim=0)
    z_rh_all = torch.cat(zrh_chunks, dim=0)

    # --- Phase 2B: Jointly train NCM (`net`) + `K3LexicalPointerDecoder` across all 8 GPUs (`cuda:0..7`) ---
    from train_e7_clean_chat import K3LexicalPointerDecoder, build_unhallucinated_vocab_mask

    ptr_dec = K3LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(dev0)
    ptr_dec.load_state_dict(base_sd["ptr_dec"])

    for name, p in net.named_parameters():
        if any(k in name for k in ("blocks.8", "blocks.9", "blocks.10", "blocks.11", "ln_f", "lm_heads")):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    levels = list(vqvae.vq.levels)
    lid = torch.cat([torch.full((l,), k, dtype=torch.long) for k, l in enumerate(levels)]).to(dev0)
    ls = torch.tensor(levels, dtype=torch.float32, device=dev0)
    w_lvl = (ls ** (-0.7))[lid]

    class JointNCMAndPointerStep(nn.Module):
        """Wraps NCM forward + VQ-VAE soft decode + K3LexicalPointerDecoder + per-GPU cross-entropy
        so `nn.DataParallel` executes every step in parallel across all 8 RTX 3090 GPUs (`cuda:0..7`)
        without copying 163,840-vocab logit tensors across PCIe."""
        def __init__(self, net_m, vq_m, ptr_m, w_lvl_t):
            super().__init__()
            self.net = net_m
            self.vqvae = vq_m
            self.ptr_dec = ptr_m
            self.levels = list(vq_m.vq.levels)
            self.vsizes = list(vq_m.vq.vqvae_vocab_sizes)
            self.register_buffer("w_lvl_buf", w_lvl_t)

        def forward(self, z_pref, pmask, ncm_in, b_zrh, b_mrh, z_bpe, b_pids, b_idx, b_tids):
            lg_ncm = self.net(torch.cat([z_pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=pmask)
            B_cur, T_c, V_c = lg_ncm.shape
            ce_ncm = F.cross_entropy(lg_ncm.float().reshape(-1, V_c), b_idx.reshape(-1), reduction="none").view(B_cur, T_c)
            loss_ncm = (ce_ncm * self.w_lvl_buf[None]).sum() / (self.w_lvl_buf.sum() * B_cur)
            with torch.no_grad():
                pred_codes = []
                off = 0
                for k, l_k in enumerate(self.levels):
                    V_k = self.vsizes[k]
                    pred_codes.append(lg_ncm[:, off:off + l_k, :V_k].argmax(dim=-1))
                    off += l_k
                pred_idx = torch.cat(pred_codes, dim=1)
                z_hats, _ = self.vqvae.vq.idx_to_zhats(pred_idx, return_zlevels=True)
                z_hat_top = z_hats[-1]
            lg_tok = self.ptr_dec(self.vqvae.decoder, z_hat_top, b_zrh, b_mrh, z_bpe, b_pids)
            loss_tok = F.cross_entropy(lg_tok.float().reshape(-1, V_K3), b_tids.reshape(-1))
            return loss_ncm.unsqueeze(0), loss_tok.unsqueeze(0)

    joint_mod = JointNCMAndPointerStep(net, vqvae, ptr_dec, w_lvl).to(dev0)
    dp_joint = nn.DataParallel(joint_mod, device_ids=device_ids)

    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if p.requires_grad], "lr": 6.0e-5},
        {"params": ptr_dec.parameters(), "lr": 2.0e-4},
    ], betas=(0.9, 0.95), weight_decay=0.01)

    net.train(); ptr_dec.train()
    bs_joint = len(device_ids) * 24  # 192 clubbed windows per step across all 8 GPUs!
    steps = 160
    g = torch.Generator().manual_seed(42)
    t_tr = time.perf_counter()
    print(f"[Phase 2B] Training NCM + K3LexicalPointerDecoder across {len(device_ids)} GPUs ({bs_joint} clubbed windows/step, {steps} steps)...", flush=True)

    for step in range(1, steps + 1):
        mult = 0.5 * (1.0 + math.cos(math.pi * step / float(steps)))
        opt.param_groups[0]["lr"] = 6.0e-5 * mult
        opt.param_groups[1]["lr"] = 2.0e-4 * mult

        sel = torch.randint(0, len(idx_all), (bs_joint,), generator=g)
        b_idx = idx_all[sel].long().to(dev0, non_blocking=True)
        b_pids = all_pids[sel].to(dev0, non_blocking=True)
        b_tids = all_tids[sel].to(dev0, non_blocking=True)
        b_zrh = z_rh_all[sel].to(dev0, dtype=torch.float32, non_blocking=True)
        b_mrh = msk_all[sel].to(dev0, non_blocking=True)

        with torch.no_grad():
            idx_c = corrupt_codes(b_idx, levels, vqvae.vq.vqvae_vocab_sizes, 0.35, 0.10)
            ncm_in = vqvae.vq.idx_to_ncm_input(idx_c)
            z_bpe = vqvae.encoder.wte(b_pids)
            z_pref, pmask = pack_combo_prefix(b_zrh, b_mrh, z_bpe, (b_pids == K3_PAD))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_ncm_v, loss_tok_v = dp_joint(z_pref, pmask, ncm_in, b_zrh, b_mrh, z_bpe, b_pids, b_idx, b_tids)
            loss_ncm = loss_ncm_v.mean()
            loss_tok = loss_tok_v.mean()
            loss = loss_ncm + 1.5 * loss_tok

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ptr_dec.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 40 == 0 or step == 1:
            print(f"  [Joint 8-GPU] step {step}/{steps} | loss_ncm={loss_ncm.item():.4f} | loss_tok={loss_tok.item():.4f} | elapsed={time.perf_counter() - t_tr:.1f}s", flush=True)

    net.eval(); ptr_dec.eval()
    restricted_mask = base_sd["restricted_mask"].cpu() if "restricted_mask" in base_sd else build_unhallucinated_vocab_mask(k3_tok).cpu()
    save_path = os.path.join(out_dir, "e8_k3_direct.pt")
    torch.save({
        "vqvae": vqvae.state_dict(),
        "net": net.state_dict(),
        "ptr_dec": ptr_dec.state_dict(),
        "restricted_mask": restricted_mask,
    }, save_path)
    print(f"[Saved] {save_path} in {time.perf_counter() - t_start:.1f}s total.", flush=True)


if __name__ == "__main__":
    main()
