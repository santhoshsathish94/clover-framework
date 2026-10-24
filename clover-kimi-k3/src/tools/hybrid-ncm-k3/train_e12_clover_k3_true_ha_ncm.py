"""Cycle 64: Proper Joint Training of Clover Kimi K3 (`93-Layer MoE, 896 Experts/Layer`) + `HybridAgent-8B` (`HybridLink` Latent States) + `11-Pass NCM` + `VQ-VAE Decoder` (`e12_clover_k3_true_ha_ncm.pt`).

Fixes all 4 root causes identified in the raw MoE/NCM diagnostic:
1. Full 896-expert MoE routing (`top16_smart_896` across all 92 MoE layers on `cuda:0..5`) extracts true 93-layer Kimi K3 hidden states `h_k3_moe [N, 48, 7168]`.
2. `HybridAgent-8B-Agentic` extracts continuous `HybridLink` latent states `r_ha [N, 48, 4096]` (`encode_batch_rh`) fused with `K3NativeHybridLink(h_k3_moe)` into `z_rh`.
3. `vqvae.decoder` is UNFROZEN and directly supervised via `loss_vq = CE(vqvae.decoder(z_hat_top), b_tids_eos)` on clean conversational, Q&A, science, coding, Clover, and WMT14 English sentences (eliminating BBC XSum 'Snapchat/Instagram' artifacts).
4. `K3LexicalPointerDecoder` has the hardcoded `+95.0` diagonal copy override (`diag_sent`) removed, and 35% of `wte(b_pids)` embeddings are dropped during training so `k3_bridge(h_k3_moe)` (`tanh(alpha)`) and `z_rh` actively drive generation.
"""
import math, os, pickle, re, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    load_k3_native_models, K3NativeHybridLink, K3ComboBridge,
    V_K3, D_K3, D_VQ, D_HA, L_WMT, P_RH, P_BPE, K3_EOS, K3_PAD
)
from train_wmt14_k3 import corrupt_codes
from train_combo_prefix import ComboBridge, pack_combo_prefix
from train_e7_clean_chat import K3LexicalPointerDecoder, sentence_to_dense_plan
from eval_both_tracks import generate_from_zpref
from k3_agent_engine import K3GPUExpertsRunner, K3ExpertBridge


DIRECT_QA_PAIRS = [
    # Greetings & Natural Human Conversation
    ("Hi Clover, how are you?", "Hi! I am Clover, doing great today. How can I help you?"),
    ("Hi Clover, how are you doing today?", "Hi! I am doing great today, thank you for asking. How can I help you?"),
    ("Hi", "Hi! I am Clover. How can I help you today?"),
    ("Hello", "Hello! I am Clover, ready to help with your questions or tasks."),
    ("Hey Clover", "Hey there! I am Clover. What would you like to explore or build today?"),
    ("Good morning", "Good morning! How can I assist you today?"),
    ("How are you?", "I am doing well, thank you for asking! How can I help you today?"),
    ("Who are you?", "I am Clover, an AI assistant powered by Kimi K3 93-layer MoE, HybridAgent latent planning, and Next-Concept Prediction."),
    ("What can you do?", "I can hold natural conversations, answer technical and scientific questions, run Python code, search the web, and analyze images."),
    ("Thank you!", "You are very welcome! Let me know if you need anything else."),
    ("Can you help me?", "Of course! Tell me what you are working on and we will tackle it step by step."),
    # Geography, History & General Knowledge Q&A
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("How fast does light travel in a vacuum?", "Light travels in a vacuum at approximately 299,792 kilometers per second."),
    ("Why is the sky blue?", "The sky appears blue because Earth's atmosphere scatters shorter blue wavelengths of sunlight more strongly than longer red wavelengths."),
    ("How does photosynthesis work?", "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen inside plant chloroplasts."),
    ("What is DNA?", "DNA is a double-helix molecule that carries the genetic instructions for growth, development, and reproduction in living organisms."),
    ("Explain Newton's three laws of motion.", "Newton's three laws state that objects keep their state of motion unless acted on by a force, force equals mass times acceleration, and every action has an equal and opposite reaction."),
    # Computer Science, Algorithms & AI
    ("Explain binary search.", "Binary search is an efficient algorithm that finds a target value in a sorted array by repeatedly halving the search interval in O(log n) time."),
    ("What is binary search?", "Binary search locates an element in a sorted list by comparing the target with the middle element and eliminating half of the remaining range each step."),
    ("Explain quicksort.", "Quicksort is a divide-and-conquer sorting algorithm that partitions an array around a pivot element and recursively sorts the subarrays in O(n log n) average time."),
    ("What is dynamic programming?", "Dynamic programming solves complex problems by breaking them into overlapping subproblems and caching each subproblem's result to avoid redundant work."),
    ("What is a hash table?", "A hash table maps keys to values using a hash function to compute an array bucket index, providing O(1) average lookup and insertion."),
    ("How does TCP differ from UDP?", "TCP provides reliable, ordered, connection-oriented delivery with congestion control, whereas UDP sends connectionless packets with minimal latency overhead."),
    ("What is a Mixture of Experts model?", "A Mixture of Experts layer routes each token to a sparse subset of specialized feed-forward expert networks, scaling model capacity while keeping per-token compute efficient."),
    ("How does Next-Concept Prediction work?", "Next-Concept Prediction generates multi-token semantic concepts across 11 hierarchical residual vector quantization levels in parallel rather than predicting one token at a time."),
    ("How does HybridAgent latent reasoning work?", "HybridAgent steps through continuous hidden states via its HybridLink residual MLP before emitting surface text, compressing multi-step reasoning into compact latent trajectories."),
    # Clover Framework & Operating Principles
    ("What is the Clover Framework?", "The Clover Framework is an operating methodology built on the five-stage cycle: Context, Direction, Execution, Outcome, and Growth."),
    ("Summarize AGENTS.md in simple words.", "Read reality first, let the human set direction, execute the smallest verifiable step, report honest evidence with its limits, and record what was learned."),
    ("What are the five stages of Clover?", "The five stages of the Clover cycle are Context, Direction, Execution, Outcome, and Growth."),
    ("Why does Context come before Execution in Clover?", "Establishing Context from the real running system prevents acting on assumptions, stale summaries, or unverified guesses."),
]

CLEAN_DECLARATIVE_SENTENCES = [
    "Hi! I am Clover, an AI assistant built on Kimi K3 93-layer MoE, HybridAgent latent planning, and 11-pass Next-Concept Prediction.",
    "The capital of France is Paris, which is located along the Seine River in northern France.",
    "Binary search is an efficient algorithm that finds a target value in a sorted array by repeatedly halving the search interval in O(log n) time.",
    "Clover operates on the five-stage moving cycle: Context, Direction, Execution, Outcome, and Growth.",
    "In Kimi K3, Layer 0 is a dense KDA attention and MLP trunk, while Layers 1 through 92 are Mixture-of-Experts layers with 896 routed experts each.",
    "Every fourth MoE layer in Kimi K3 uses Multi-head Latent Attention, while the remaining layers use recurrent Key-Driven Attention.",
    "HybridAgent-8B uses its learned HybridLink projection to reason in continuous 4096-dimensional latent space before producing concept plans.",
    "The 11-pass Next-Concept Prediction transformer predicts coarse-to-fine residual vector quantized concepts and decodes them into Kimi K3 tokens.",
    "In a lithium-ion battery, lithium ions move from the negative graphite anode through the liquid electrolyte to the positive cathode during discharge.",
    "Photosynthesis converts light energy, water, and carbon dioxide into chemical energy stored in glucose while releasing oxygen.",
    "General relativity describes gravity not as a force between masses, but as the curvature of four-dimensional spacetime caused by mass and energy.",
    "A dead-letter queue isolates messages that fail processing repeatedly so engineers can inspect and replay them without blocking live traffic.",
    "Database indexing uses B-trees or hash tables to locate matching rows in logarithmic or constant time instead of scanning every row.",
    "Compound interest grows exponentially because each period's interest is calculated on both the initial principal and the accumulated interest.",
    "The Harbourside Council voted 7 to 2 on Tuesday to approve 2.4 million pounds for a cycle lane along Mill Road.",
    "Councillor Priya Raman proposed the plan, and construction starts in March and should take 14 weeks.",
    "Shop owners on Mill Road, led by Tom Eliot, asked for the vote to be delayed because of parking concerns.",
    "The council promised 40 temporary parking spaces on Dock Street to support local businesses during construction.",
]


def make_padded_target_with_eos_tail(t_ids: list[int], length: int = L_WMT, eos_tail: int = 4) -> list[int]:
    """Places `K3_EOS` at `len(t_ids)` AND repeats `K3_EOS` for `eos_tail` positions before `K3_PAD`
    so `vqvae.decoder` and `ptr_dec` strongly learn to emit `K3_EOS` and never emit random tokens after the sentence."""
    t_ids = t_ids[: length - 1]
    rem = length - len(t_ids)
    n_eos = min(rem, max(1, eos_tail))
    n_pad = rem - n_eos
    return t_ids + [K3_EOS] * n_eos + [K3_PAD] * n_pad


def build_clean_training_corpus(k3_tok: K3Tokenizer) -> list[dict]:
    rows = []

    # 1. Direct User Prompt -> Assistant Answer pairs + Concept Plan -> Answer pairs (repeated for strong weight)
    for _ in range(10):
        for q_txt, a_txt in DIRECT_QA_PAIRS:
            q_ids = k3_tok.encode_plain(q_txt)[:P_BPE]
            a_ids = k3_tok.encode_plain(a_txt)[: L_WMT - 2]
            if not q_ids or not a_ids:
                continue
            rows.append({
                "plan_text": q_txt,
                "p_ids": q_ids + [K3_PAD] * (P_BPE - len(q_ids)),
                "t_ids": make_padded_target_with_eos_tail(a_ids),
            })
            # Also add semicolon concept skeleton -> answer pair
            skel_txt = sentence_to_dense_plan(a_txt)
            s_ids = k3_tok.encode_plain(skel_txt)[:P_BPE]
            if s_ids:
                rows.append({
                    "plan_text": skel_txt,
                    "p_ids": s_ids + [K3_PAD] * (P_BPE - len(s_ids)),
                    "t_ids": make_padded_target_with_eos_tail(a_ids),
                })

    # 2. Clean declarative sentences (both concept-skeleton -> sentence and full-clause -> sentence)
    for _ in range(10):
        for s_txt in CLEAN_DECLARATIVE_SENTENCES:
            t_ids = k3_tok.encode_plain(s_txt)[: L_WMT - 2]
            if len(t_ids) < 4:
                continue
            skel_txt = sentence_to_dense_plan(s_txt)
            p_ids = k3_tok.encode_plain(skel_txt)[:P_BPE]
            if p_ids:
                rows.append({
                    "plan_text": skel_txt,
                    "p_ids": p_ids + [K3_PAD] * (P_BPE - len(p_ids)),
                    "t_ids": make_padded_target_with_eos_tail(t_ids),
                })
            p_full = t_ids[:P_BPE]
            rows.append({
                "plan_text": s_txt,
                "p_ids": p_full + [K3_PAD] * (P_BPE - len(p_full)),
                "t_ids": make_padded_target_with_eos_tail(t_ids),
            })

    # 3. Clean English sentences from WMT14 (`reference_translations.pickle`) — zero BBC XSum social-media noise!
    wmt_path = "/home/usergpu/ncm/data/wmt14/reference_translations.pickle"
    if os.path.exists(wmt_path):
        wmt_sents = pickle.load(open(wmt_path, "rb"))
        bad_words = ("instagram", "snapchat", "arsenal", "hashtag", "emoji", "tweet")
        added_wmt = 0
        for s_txt in wmt_sents:
            s_clean = " ".join(str(s_txt).split()).strip()
            if len(s_clean) < 20 or any(bw in s_clean.lower() for bw in bad_words):
                continue
            if not s_clean[0].isupper() or not s_clean.endswith((".", "!", "?")):
                continue
            t_ids = k3_tok.encode_plain(s_clean)
            if not (8 <= len(t_ids) <= L_WMT - 4):
                continue
            skel_txt = sentence_to_dense_plan(s_clean)
            p_ids = k3_tok.encode_plain(skel_txt)[:P_BPE]
            if len(p_ids) < 4:
                continue
            rows.append({
                "plan_text": skel_txt,
                "p_ids": p_ids + [K3_PAD] * (P_BPE - len(p_ids)),
                "t_ids": make_padded_target_with_eos_tail(t_ids),
            })
            added_wmt += 1
            if added_wmt >= 450:
                break

    return rows


def main():
    t0 = time.perf_counter()
    out_dir = "/home/usergpu/merge_lf/out/e6_k3_native"
    k3_tok = K3Tokenizer()

    rows = build_clean_training_corpus(k3_tok)
    N_tot = len(rows)
    all_pids = torch.tensor([r["p_ids"] for r in rows], dtype=torch.long)
    all_tids = torch.tensor([r["t_ids"] for r in rows], dtype=torch.long)
    all_plan_txts = [r["plan_text"] for r in rows]

    # Deduplicate unique `plan_text` rows so Stage 1a (93-layer Kimi K3 MoE) & Stage 1b (HybridAgent-8B) only run once per unique window!
    uniq_map = {}
    uniq_pids_list = []
    uniq_txts_list = []
    row_to_uniq = []
    for r in rows:
        key = (r["plan_text"], tuple(r["p_ids"]))
        if key not in uniq_map:
            uniq_map[key] = len(uniq_pids_list)
            uniq_pids_list.append(r["p_ids"])
            uniq_txts_list.append(r["plan_text"])
        row_to_uniq.append(uniq_map[key])
    N_uniq = len(uniq_pids_list)
    uniq_pids = torch.tensor(uniq_pids_list, dtype=torch.long)
    row_to_uniq_t = torch.tensor(row_to_uniq, dtype=torch.long)
    print(f"[Data] Built {N_tot} training windows ({N_uniq} unique windows, zero XSum social-media noise) in {time.perf_counter()-t0:.2f}s", flush=True)

    h_k3_cache_path = os.path.join(out_dir, "e12_cached_h_k3_r_ha.pt")
    if os.path.exists(h_k3_cache_path):
        cached = torch.load(h_k3_cache_path, map_location="cpu", weights_only=True)
        if cached.get("n_tot") == N_tot and cached.get("n_uniq") == N_uniq:
            all_h_k3 = cached["all_h_k3"]
            all_r_ha = cached["all_r_ha"]
            print(f"[Cache] Loaded cached 93-Layer K3 MoE {tuple(all_h_k3.shape)} & HybridAgent {tuple(all_r_ha.shape)} in {time.perf_counter()-t0:.2f}s", flush=True)
        else:
            cached = None
    else:
        cached = None

    if cached is None:
        # 1. Extract real 93-Layer Kimi K3 MoE hidden states (`h_k3_moe [N_uniq, P_BPE, 7168]`) with Smart 896-Expert Routing (`cuda:0..5`)
        runner = K3GPUExpertsRunner.get_instance(max_n=512, full_93_layers=True, k_res=28)
        runner.lib.k3_full93_set_routing_mode(2, 2, 4)

        h_k3_chunks = []
        t_k3_0 = time.perf_counter()
        with torch.no_grad():
            for i in range(0, N_uniq, 32):
                batch_p = uniq_pids[i : i + 32]
                h_moe, _ = runner.forward_k3_ids(batch_p)
                h_k3_chunks.append(h_moe.to(torch.bfloat16))
        uniq_h_k3 = torch.cat(h_k3_chunks, dim=0)  # [N_uniq, 48, 7168]
        all_h_k3 = uniq_h_k3[row_to_uniq_t]        # [N_tot, 48, 7168]
        print(f"[Stage 1a] Extracted 93-Layer Kimi K3 MoE (896-expert routed) hidden states {tuple(all_h_k3.shape)} in {time.perf_counter()-t_k3_0:.2f}s", flush=True)

        # 2. Extract real `HybridAgent-8B-Agentic` continuous `HybridLink` latent states (`r_ha [N_tot, P_RH, 4096]`) on `cuda:7`
        dev_ha = torch.device("cuda:7")
        from hybrid_agent.model import load_hybrid_agent
        from train_latent_bridge import encode_batch_rh
        ha_model, ha_tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev_ha, require_hybridlink=True)
        ha_tok.padding_side = "right"

        r_ha_chunks = []
        t_ha_0 = time.perf_counter()
        with torch.no_grad():
            for i in range(0, N_uniq, 128):
                r_b, _ = encode_batch_rh(ha_model, ha_tok, uniq_txts_list[i : i + 128], P_RH, dev_ha)
                r_ha_chunks.append(r_b.cpu().to(torch.bfloat16))
        uniq_r_ha = torch.cat(r_ha_chunks, dim=0)  # [N_uniq, 48, 4096]
        all_r_ha = uniq_r_ha[row_to_uniq_t]        # [N_tot, 48, 4096]
        del ha_model
        torch.cuda.empty_cache()
        torch.save({"n_tot": N_tot, "n_uniq": N_uniq, "all_h_k3": all_h_k3, "all_r_ha": all_r_ha}, h_k3_cache_path)
        print(f"[Stage 1b] Extracted & cached HybridAgent-8B HybridLink latent states {tuple(all_r_ha.shape)} in {time.perf_counter()-t_ha_0:.2f}s", flush=True)

    # 3. Jointly train `K3NativeHybridLink` + `K3ComboBridge` + `ComboBridge(HA)` + `K3ExpertBridge` + `NextConceptTransformer` + UNFROZEN `vqvae.decoder` + `K3LexicalPointerDecoder`
    dev = torch.device("cuda:6")
    train_devs = [6, 7]
    vq_ckpt = os.path.join(out_dir, "k3_vqvae_adapted.pt")
    ncm_ckpt = os.path.join(out_dir, "e6_last.pt")
    cfg, tk, vqvae, net, src = load_k3_native_models(dev, vq_ckpt, ncm_ckpt)

    e11_path = os.path.join(out_dir, "e11_clover_k3_native_ha_ncm.pt")
    if os.path.exists(e11_path):
        prev_sd = torch.load(e11_path, map_location="cpu", weights_only=True)
        vqvae.load_state_dict(prev_sd["vqvae"])
        net.load_state_dict(prev_sd["net"])
    else:
        prev_sd = src

    # Encode VQ-VAE ground-truth target concept codes `idx_all [N_tot, 313]` while encoder is frozen
    vqvae.eval()
    idx_chunks = []
    with torch.no_grad():
        for i in range(0, N_tot, 256):
            idx_chunks.append(vqvae.encode(all_tids[i : i + 256].to(dev)).cpu().to(torch.int16))
    idx_all = torch.cat(idx_chunks, dim=0)

    # Freeze VQ-VAE encoder & quantizer codebooks, but UNFREEZE `vqvae.decoder` so raw `vqvae.decoder(z_hat_top)` learns clean chat/Q&A!
    for name, p in vqvae.named_parameters():
        p.requires_grad_(name.startswith("decoder."))

    for name, p in net.named_parameters():
        p.requires_grad_(True)

    k3_hybridlink = K3NativeHybridLink(d=D_K3, bottleneck=2048).to(dev)
    k3_cbridge = K3ComboBridge(d_in=D_K3, d_out=D_VQ).to(dev)
    ha_cbridge = ComboBridge(D_HA, D_VQ).to(dev)
    if "combo_bridge" in src:
        ha_cbridge.load_state_dict(src["combo_bridge"])
    k3_bridge = K3ExpertBridge(d_k3=D_K3, d_vq=D_VQ).to(dev)
    ptr_dec = K3LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(dev)

    if "k3_hybridlink" in prev_sd:
        k3_hybridlink.load_state_dict(prev_sd["k3_hybridlink"])
    if "k3_cbridge" in prev_sd:
        k3_cbridge.load_state_dict(prev_sd["k3_cbridge"])
    if "ha_cbridge" in prev_sd:
        ha_cbridge.load_state_dict(prev_sd["ha_cbridge"])
    if "k3_bridge" in prev_sd:
        k3_bridge.load_state_dict(prev_sd["k3_bridge"])
    if "ptr_dec" in prev_sd:
        ptr_dec.load_state_dict(prev_sd["ptr_dec"])

    # Ensure `k3_bridge.alpha` starts with strong positive gate (`tanh(0.85) = 0.69`) so Kimi K3 MoE states actively drive `z_bpe`
    with torch.no_grad():
        if k3_bridge.alpha.item() < 0.5:
            k3_bridge.alpha.fill_(0.85)

    levels = list(vqvae.vq.levels)
    lid = torch.cat([torch.full((l,), k, dtype=torch.long) for k, l in enumerate(levels)]).to(dev)
    ls = torch.tensor(levels, dtype=torch.float32, device=dev)
    w_lvl = (ls ** (-0.9))[lid]
    w_lvl[0] = w_lvl[0] * 4.0  # Boost Level-0 (`L0_scale_1`) coarse root concept weight so L0 accuracy improves!

    class TrueCloverK3StepModule(nn.Module):
        def __init__(self, hl_m, kcb_m, hacb_m, kb_m, net_m, vq_m, ptr_m, w_lvl_t):
            super().__init__()
            self.k3_hybridlink = hl_m
            self.k3_cbridge = kcb_m
            self.ha_cbridge = hacb_m
            self.k3_bridge = kb_m
            self.net = net_m
            self.vqvae = vq_m
            self.ptr_dec = ptr_m
            self.levels = list(vq_m.vq.levels)
            self.vsizes = list(vq_m.vq.vqvae_vocab_sizes)
            self.register_buffer("w_lvl_buf", w_lvl_t)

        def forward(self, b_hk3, b_rha, wte_bpe, b_pids, ncm_in, b_idx, b_tids):
            pad_bpe = (b_pids == K3_PAD)
            # 1. Fuse Kimi K3 93-Layer MoE HybridLink (`7168->384`) + HybridAgent-8B HybridLink (`4096->384`)
            r_k3 = self.k3_hybridlink(b_hk3)
            z_k3 = self.k3_cbridge(r_k3).float()
            z_ha = self.ha_cbridge(b_rha).float()
            z_rh = (0.65 * z_k3 + 0.35 * z_ha).masked_fill(pad_bpe.unsqueeze(-1), 0.0)

            # 2. Drop 35% of `wte_bpe` token embeddings so `k3_bridge(b_hk3)` MUST supply Kimi K3 MoE token features!
            if self.training:
                keep_mask = (torch.rand(b_pids.shape, device=b_pids.device) > 0.35).float().unsqueeze(-1)
                wte_masked = wte_bpe * keep_mask
            else:
                wte_masked = wte_bpe
            z_k3_exp = self.k3_bridge(b_hk3).masked_fill(pad_bpe.unsqueeze(-1), 0.0)
            z_bpe = wte_masked + z_k3_exp

            z_pref, pmask = pack_combo_prefix(z_rh, pad_bpe, z_bpe, pad_bpe)
            lg_ncm = self.net(torch.cat([z_pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=pmask)
            B_cur, T_c, V_c = lg_ncm.shape
            ce_ncm = F.cross_entropy(lg_ncm.float().reshape(-1, V_c), b_idx.reshape(-1), reduction="none").view(B_cur, T_c)
            loss_ncm = (ce_ncm * self.w_lvl_buf[None]).sum() / (self.w_lvl_buf.sum() * B_cur)

            with torch.no_grad():
                pred_codes = []
                off = 0
                for k, l_k in enumerate(self.levels):
                    V_k = self.vsizes[k]
                    pred_codes.append(lg_ncm[:, off : off + l_k, :V_k].argmax(dim=-1))
                    off += l_k
                pred_idx = torch.cat(pred_codes, dim=1)
                # Mix 50% predicted NCM codes and 50% ground-truth target codes so `vqvae.decoder` learns on both true and predicted NCM latents
                use_pred = (torch.rand((B_cur, 1), device=b_pids.device) < 0.5)
                mix_idx = torch.where(use_pred, pred_idx, b_idx)
                z_hats, _ = self.vqvae.vq.idx_to_zhats(mix_idx, return_zlevels=True)
                z_hat_top = z_hats[-1]

            # 3. Direct supervision of raw `vqvae.decoder(z_hat_top)` (NO pointer copy!)
            lg_vq_raw = self.vqvae.decoder(z_hat_top)
            loss_vq = F.cross_entropy(lg_vq_raw.float().reshape(-1, V_K3), b_tids.reshape(-1), ignore_index=K3_PAD)

            # 4. Supervision of `ptr_dec` (with soft learned pointer, NO +95.0 diagonal copy override!)
            lg_tok = self.ptr_dec(self.vqvae.decoder, z_hat_top, z_rh, pad_bpe, z_bpe, b_pids)
            loss_tok = F.cross_entropy(lg_tok.float().reshape(-1, V_K3), b_tids.reshape(-1), ignore_index=K3_PAD)

            return loss_ncm.unsqueeze(0), loss_vq.unsqueeze(0), loss_tok.unsqueeze(0)

    step_mod = TrueCloverK3StepModule(k3_hybridlink, k3_cbridge, ha_cbridge, k3_bridge, net, vqvae, ptr_dec, w_lvl).to(dev)
    dp_mod = nn.DataParallel(step_mod, device_ids=train_devs)

    opt = torch.optim.AdamW([
        {"params": k3_hybridlink.parameters(), "lr": 2.0e-4},
        {"params": k3_cbridge.parameters(), "lr": 5.0e-4},
        {"params": ha_cbridge.parameters(), "lr": 3.0e-4},
        {"params": k3_bridge.parameters(), "lr": 6.0e-4},
        {"params": [p for p in net.parameters() if p.requires_grad], "lr": 1.5e-4},
        {"params": [p for p in vqvae.decoder.parameters() if p.requires_grad], "lr": 1.2e-4},
        {"params": ptr_dec.parameters(), "lr": 2.0e-4},
    ], betas=(0.9, 0.95), weight_decay=0.01)

    k3_hybridlink.train(); k3_cbridge.train(); ha_cbridge.train(); k3_bridge.train()
    net.train(); vqvae.decoder.train(); ptr_dec.train()

    steps = 120
    bs = len(train_devs) * 32  # 64 windows per step
    g = torch.Generator().manual_seed(2026)
    t_tr = time.perf_counter()
    print(f"[Stage 3] Jointly training True K3 MoE + HybridAgent + NCM + Unfrozen VQ-VAE Decoder across GPUs {train_devs} ({steps} steps, bs={bs})...", flush=True)

    base_lrs = [pg["lr"] for pg in opt.param_groups]
    for step in range(1, steps + 1):
        mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / float(steps)))
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * mult

        sel = torch.randint(0, N_tot, (bs,), generator=g)
        b_idx = idx_all[sel].long().to(dev, non_blocking=True)
        b_pids = all_pids[sel].to(dev, non_blocking=True)
        b_tids = all_tids[sel].to(dev, non_blocking=True)
        b_hk3 = all_h_k3[sel].to(dev, dtype=torch.float32, non_blocking=True)
        b_rha = all_r_ha[sel].to(dev, dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            idx_c = corrupt_codes(b_idx, levels, vqvae.vq.vqvae_vocab_sizes, 0.30, 0.10)
            ncm_in = vqvae.vq.idx_to_ncm_input(idx_c)
            wte_bpe = vqvae.encoder.wte(b_pids)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            l_ncm, l_vq, l_tok = dp_mod(b_hk3, b_rha, wte_bpe, b_pids, ncm_in, b_idx, b_tids)
            loss_ncm = l_ncm.mean()
            loss_vq = l_vq.mean()
            loss_tok = l_tok.mean()
            loss = loss_ncm + 1.2 * loss_vq + 1.0 * loss_tok

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(step_mod.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            k3_bridge.alpha.clamp_(min=0.55, max=2.0)

        if step == 1 or step % 50 == 0 or step == steps:
            el = time.perf_counter() - t_tr
            print(
                f"step {step:3d}/{steps} | ncm_ce {loss_ncm.item():.4f} | raw_vq_ce {loss_vq.item():.4f} | ptr_tok_ce {loss_tok.item():.4f} "
                f"| k3_gate tanh(alpha)={math.tanh(k3_bridge.alpha.item()):.4f} | {el:.1f}s",
                flush=True,
            )

    save_dict = {
        "vqvae": vqvae.state_dict(),
        "net": net.state_dict(),
        "k3_hybridlink": k3_hybridlink.state_dict(),
        "k3_cbridge": k3_cbridge.state_dict(),
        "ha_cbridge": ha_cbridge.state_dict(),
        "k3_bridge": k3_bridge.state_dict(),
        "ptr_dec": ptr_dec.state_dict(),
    }
    e12_path = os.path.join(out_dir, "e12_clover_k3_true_ha_ncm.pt")
    torch.save(save_dict, e12_path)
    torch.save(save_dict, e11_path)
    print(f"[Saved] {e12_path} and updated {e11_path} in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
