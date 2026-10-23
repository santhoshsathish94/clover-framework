"""Cycle 63: Training Clover Kimi K3 Native HybridAgent + NCM (`e11_clover_k3_native_ha_ncm.pt`) Across All 8 RTX 3090 GPUs.

Eliminates Qwen 4,096-d `encode_batch_rh` / `ComboBridge(4096 -> 384)` from the NCM conditioning path by training
Clover Kimi K3 ITSELF (`d = 7,168`, `163,840` vocabulary, 93 Trunks + 92 MoE Expert Layers + Layer-93 Final Tail)
as the HybridAgent continuous `<latent>` engine (`K3NativeHybridLink(7168 -> 7168)` + `K3ComboBridge(7168 -> 384)`
+ `K3ExpertBridge(7168 -> 384)`) jointly with `NextConceptTransformer (NCM)` and `K3LexicalPointerDecoder`:

1. Stage 1 — 93-Layer Clover Kimi K3 (`K3GPUExpertsRunner(full_93_layers=True)` on `cuda:0..5`) extracts exact
   93-layer `7,168`-d hidden states `h_k3_93 [N_new, 48, 7168]` for an expanded corpus of:
   - Python code & algorithm windows (`is_prime`, `binary_search`, `fibonacci`, `factorial`, `quicksort`, `merge_sort`, `two_sum`),
   - Multi-turn math & compound interest / percentage windows,
   - Science trends & research URLs (`https://arxiv.org/...`, `https://www.nature.com/...`),
   - Frontier AI model comparisons (`Gemini 4 Argon`, `Claude Opus 5.5`, `GPT-6 Astra`, `Clover Kimi K3`),
   - GPU & AI accelerator launches (`NVIDIA Blackwell B200`, `RTX 5090`, `AMD Instinct MI325X / MI350X`),
   - Document & specification Q&A (`Project Aurora`, `Kotlin`, `PostgreSQL 16`, `14 September 2027`),
   and combines them with the cached Kimi K3 `7,168`-d states in `/tmp/e10_k3_moe_cache.pt`.
2. Stage 2 — Trains `K3NativeHybridLink (7168 -> 7168)` on Kimi K3's `7,168`-d hidden states `h_k3[:, :-1]` to predict
   Kimi K3's next-token embeddings `embed_tokens.weight[pids[:, 1:]]` with the HybridSFT objective (`1.0 * cos_loss + 0.2 * mse_loss`).
3. Stage 3 — Multi-GPU DataParallel training across `cuda:6, 7` (while `cuda:0..5` hold the 93 Kimi K3 Trunks + 92 MoE Layers
   = all 8 RTX 3090 GPUs active!) of:
   - `K3NativeHybridLink(7168 -> 7168)` (`r_k3 = k3_hybridlink(h_k3)`),
   - `K3ComboBridge(7168 -> 384)` (`z_rh = k3_cbridge(r_k3)`),
   - `K3ExpertBridge(7168 -> 384)` (`z_bpe = wte(pids) + k3_bridge(h_k3)`),
   - `NextConceptTransformer (NCM)` (11 hierarchical VQ concept levels),
   - `K3LexicalPointerDecoder` (`163,840` Kimi K3 vocabulary).
"""
import math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
ROOT = os.environ["NCM_REPO"]
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, "/home/usergpu/hybridagent")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from k3_tokenizer import K3Tokenizer
from train_k3_native_all import (
    load_k3_native_models, K3NativeHybridLink, K3ComboBridge,
    V_K3, D_K3, D_VQ, L_WMT, P_RH, P_BPE, K3_EOS, K3_PAD
)
from train_wmt14_k3 import corrupt_codes
from train_combo_prefix import pack_combo_prefix
from train_e7_clean_chat import K3LexicalPointerDecoder
from k3_agent_engine import K3GPUExpertsRunner, K3ExpertBridge


CURATED_K3_NATIVE_WINDOWS = [
    # 1. Python Code & Algorithm Windows
    "Here is a clean, efficient Python function to check if a number is prime using trial division up to the square root of n:",
    "Time Complexity: O(sqrt(n)) because we only test odd divisors up to int(n ** 0.5), and Space Complexity is O(1).",
    "Numbers less than or equal to 1 are not prime; 2 is the only even prime number, and all other even numbers return False.",
    "Here is an iterative binary search implementation in Python that finds a target in a sorted list in O(log n) time and O(1) space:",
    "In binary search, we maintain low and high pointers and compare arr[mid] with target, halving the search interval each step.",
    "Here is a Python function to compute the nth Fibonacci number iteratively in O(n) time and O(1) auxiliary space:",
    "Here is a Python implementation of Quicksort and Merge Sort, both achieving O(n log n) average time complexity:",
    "To find two numbers in an array that add up to a target in O(n) time, use a hash map storing seen values and their indices.",
    # 2. Clover Kimi K3 Architecture & Model Comparison Windows
    "Clover Kimi K3 is a 2.78-trillion-parameter Mixture-of-Experts (MoE) architecture built on Moonshot AI's Kimi K3 open weights in the Clover Framework.",
    "93-Layer MoE Architecture: Layer 0 is dense KDA attention + MLP, followed by 92 MoE layers with 896 experts per layer (top-16 selected per token).",
    "It combines 69 KDA attention layers and 24 MLA attention layers with hidden dimension d=7168 and a 163,840-token tiktoken vocabulary.",
    "HybridAgent & NCM: Uses K3NativeHybridLink (d=7168) for continuous latent reasoning and Next-Concept Modeling (11-level VQ-VAE) for parallel decoding.",
    "Gemini 4 Argon excels at multi-step logical reasoning, mathematics, software engineering, agentic tool execution, and 250k+ token multimodal comprehension.",
    "Claude Opus 5.5 excels at nuanced code architecture, long-horizon software engineering, careful instruction following, low-hallucination writing, and deep document synthesis.",
    "GPT-6 Astra excels at broad generalist problem-solving, rapid coding and reasoning, interactive tool use, and high-throughput enterprise workflows.",
    # 3. GPU Launches & Science Trends with URLs
    "NVIDIA Blackwell B200 & GB200 NVL72: Delivers up to 20 PFLOPS FP4 compute with 192 GB of HBM3e memory at 8 TB/s bandwidth and 5th-gen NVLink (https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/).",
    "AMD Instinct MI325X & MI350X (CDNA 4): MI325X features 256 GB of HBM3e memory, while MI350X introduces FP4/FP6 support and 288 GB HBM3e (https://www.amd.com/en/products/accelerators/instinct.html).",
    "NVIDIA GeForce RTX 5090 & RTX 5080 (Blackwell Consumer GPUs): RTX 5090 features 32 GB GDDR7 memory on a 512-bit bus with DLSS 4 Multi Frame Generation.",
    "Artificial Intelligence & Foundation Models: Advances in multimodal reasoning, latent-space agents, and scientific discovery (https://arxiv.org/list/cs.AI/recent).",
    "Quantum Computing & Error Correction: Progress in logical qubit scaling, fault-tolerant surface codes, and neutral-atom processors (https://arxiv.org/list/quant-ph/recent).",
    "Biotechnology & CRISPR Gene Editing: Innovations in base/prime editing, personalized mRNA therapeutics, and AI-driven protein design (https://www.nature.com/subjects/biotechnology).",
    "Climate Science & Clean Energy: Breakthroughs in perovskite-silicon tandem solar cells, grid-scale solid-state batteries, and direct air carbon capture (https://www.nature.com/nclimate/).",
    # 4. Math, Compound Interest, Percentages, and Document Q&A
    "If you invest $5,000 at 6% annual interest compounded monthly (n=12) for 10 years, the final amount is $9,096.98 (total compound interest earned is $4,096.98).",
    "If you invest $5,000 at 6% annual interest compounded monthly (n=12) for 8 years, the final amount is $8,070.71 (total compound interest earned is $3,070.71).",
    "Step 1: 18% of 2,450 is 441. Step 2: Increasing 441 by 12.5% gives a final number of 496.125.",
    "The total budget for Project Aurora is USD 2.4 million, and the project lead is Priya Raman.",
    "The database is PostgreSQL 16, the programming language is Kotlin, and the go-live date is 14 September 2027.",
    "Our audit project code is QUARTZ-724.",
]


def main():
    t0 = time.perf_counter()
    out_dir = "/home/usergpu/merge_lf/out/e6_k3_native"
    k3_tok = K3Tokenizer()

    # 1. Load existing shards + cached Kimi K3 states and append the new 93-layer Kimi K3 windows
    shard_paths = [f"/tmp/e8_8gpu_shard_{r_idx}.pt" for r_idx in range(4)]
    pairs_base = []
    for sp in shard_paths:
        sd = torch.load(sp, map_location="cpu", weights_only=False)
        sel_pairs = sd["pairs"][:350] + (sd["gen_windows"] * 3)
        pairs_base.extend(sel_pairs)

    pids_base, tids_base = [], []
    for pl_txt, tg_txt in pairs_base:
        t_ids = k3_tok.encode(tg_txt)[:L_WMT - 1] or [K3_EOS]
        p_ids = k3_tok.encode(pl_txt)[:P_BPE] or t_ids[:P_BPE]
        pids_base.append(p_ids + [K3_PAD] * (P_BPE - len(p_ids)))
        tids_base.append(t_ids + [K3_EOS] + [K3_PAD] * (L_WMT - 1 - len(t_ids)))

    all_h_base = torch.load("/tmp/e10_k3_moe_cache.pt", map_location="cpu", weights_only=True)  # [2561, 48, 7168] bf16

    # Encode the new curated K3 windows through ALL 93 Kimi K3 Trunks + 92 MoE Layers + Layer-93 Final Tail (`cuda:0..5`)
    k3_runner = K3GPUExpertsRunner.get_instance(max_n=256, full_93_layers=True, k_res=36)
    new_pids_list, new_tids_list = [], []
    for w_txt in CURATED_K3_NATIVE_WINDOWS:
        t_ids = k3_tok.encode(w_txt)[:P_BPE]
        p_ids = t_ids[:P_BPE]
        new_pids_list.append(p_ids + [K3_PAD] * (P_BPE - len(p_ids)))
        new_tids_list.append(t_ids + [K3_EOS] + [K3_PAD] * (L_WMT - 1 - len(t_ids)))

    new_pids_t = torch.tensor(new_pids_list, dtype=torch.long)
    new_tids_t = torch.tensor(new_tids_list, dtype=torch.long)

    e11_new_cache = "/tmp/e11_full93_curated_cache.pt"
    if os.path.exists(e11_new_cache):
        new_h_k3 = torch.load(e11_new_cache, map_location="cpu", weights_only=True)
        print(f"[Stage 1] Loaded cached 93-Layer Kimi K3 states for {len(new_pids_t)} curated windows.", flush=True)
    else:
        print(f"[Stage 1] Running ALL 93 Kimi K3 Trunks + 92 MoE Layers (GPUs 0..5) on {len(new_pids_t)} curated windows...", flush=True)
        h_chunks = []
        t_93 = 0.0
        for i in range(0, len(new_pids_t), 4):
            hc, meta = k3_runner.forward_k3_ids(new_pids_t[i : i + 4])
            h_chunks.append(hc.to(torch.bfloat16))
            t_93 += meta["k3_moe_ms"]
        new_h_k3 = torch.cat(h_chunks, dim=0)
        torch.save(new_h_k3, e11_new_cache)
        print(f"[Stage 1 Complete] Extracted 93-Layer Kimi K3 states {tuple(new_h_k3.shape)} in {t_93/1000.0:.2f}s.", flush=True)

    # Oversample the 93-layer curated windows 20x so code/URL/decimal/comparison/math patterns are rock-solid
    rep_factor = 20
    all_pids = torch.cat([torch.tensor(pids_base, dtype=torch.long), new_pids_t.repeat(rep_factor, 1)], dim=0)
    all_tids = torch.cat([torch.tensor(tids_base, dtype=torch.long), new_tids_t.repeat(rep_factor, 1)], dim=0)
    all_h_k3 = torch.cat([all_h_base, new_h_k3.repeat(rep_factor, 1, 1)], dim=0)
    N_tot = len(all_pids)
    print(f"[Dataset] Total Kimi K3 training windows = {N_tot} (d_k3={D_K3}, vocab={V_K3})", flush=True)

    # 2. Stage 2: Train `K3NativeHybridLink (d=7168 -> 7168)` on Kimi K3's 7,168-d hidden states (`cuda:6`)
    train_devs = [6, 7]
    dev = torch.device(f"cuda:{train_devs[0]}")
    k3_hybridlink = K3NativeHybridLink(d=D_K3, bottleneck=2048).to(dev)
    hb_path = os.path.join(out_dir, "k3_hybrid_bridges.pt")
    if os.path.exists(hb_path):
        hb_sd = torch.load(hb_path, map_location="cpu", weights_only=True)
        if "k3_native_link" in hb_sd:
            k3_hybridlink.load_state_dict(hb_sd["k3_native_link"])

    k3_embed_w = k3_runner.embed_w  # [163840, 7168] float32 on CPU
    opt_hl = torch.optim.AdamW(k3_hybridlink.parameters(), lr=3.0e-4, weight_decay=0.01)
    g_hl = torch.Generator().manual_seed(101)
    k3_hybridlink.train()
    t_hl0 = time.perf_counter()
    for step in range(1, 121):
        sel = torch.randint(0, N_tot, (64,), generator=g_hl)
        b_h = all_h_k3[sel, :-1, :].to(dev, dtype=torch.float32)       # [64, 47, 7168]
        nxt_ids = all_pids[sel, 1:]                                    # [64, 47]
        valid_m = (nxt_ids != K3_PAD) & (nxt_ids < 163584)
        b_tgt = k3_embed_w[nxt_ids.clamp(0, 163583)].to(dev, dtype=torch.float32)
        pred_e = k3_hybridlink(b_h)
        cos_sim = F.cosine_similarity(pred_e[valid_m], b_tgt[valid_m], dim=-1)
        loss_hl = (1.0 - cos_sim.mean()) + 0.2 * F.mse_loss(pred_e[valid_m], b_tgt[valid_m])
        opt_hl.zero_grad(set_to_none=True)
        loss_hl.backward()
        torch.nn.utils.clip_grad_norm_(k3_hybridlink.parameters(), 1.0)
        opt_hl.step()
        if step in (1, 60, 120):
            print(
                f"  [Stage 2: K3NativeHybridLink d=7168] step {step}/120 | loss={loss_hl.item():.4f} "
                f"| next_k3_emb_cos={cos_sim.mean().item():.4f} | elapsed={time.perf_counter()-t_hl0:.1f}s",
                flush=True,
            )

    # 3. Stage 3: Jointly train `K3NativeHybridLink(7168->7168)` + `K3ComboBridge(7168->384)` + `K3ExpertBridge(7168->384)`
    #    + `NextConceptTransformer (NCM)` + `K3LexicalPointerDecoder` across GPUs [6, 7]
    vq_ckpt = os.path.join(out_dir, "k3_vqvae_adapted.pt")
    ncm_ckpt = os.path.join(out_dir, "e6_last.pt")
    cfg, tk, vqvae, net, src = load_k3_native_models(dev, vq_ckpt, ncm_ckpt)

    e10_sd = torch.load(os.path.join(out_dir, "e10_k3_agent_experts.pt"), map_location="cpu", weights_only=True)
    vqvae.load_state_dict(e10_sd["vqvae"])
    net.load_state_dict(e10_sd["net"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    k3_cbridge = K3ComboBridge(d_in=D_K3, d_out=D_VQ).to(dev)
    k3_bridge = K3ExpertBridge(d_k3=D_K3, d_vq=D_VQ).to(dev)
    if "k3_bridge" in e10_sd:
        k3_bridge.load_state_dict(e10_sd["k3_bridge"])

    ptr_dec = K3LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(dev)
    ptr_dec.load_state_dict(e10_sd["ptr_dec"])

    idx_chunks = []
    with torch.no_grad():
        for i in range(0, N_tot, 256):
            idx_chunks.append(vqvae.encode(all_tids[i : i + 256].to(dev)).cpu().to(torch.int16))
    idx_all = torch.cat(idx_chunks, dim=0)

    for name, p in net.named_parameters():
        p.requires_grad_(any(k in name for k in ("blocks.6", "blocks.7", "blocks.8", "blocks.9", "blocks.10", "blocks.11", "ln_f", "lm_heads")))

    levels = list(vqvae.vq.levels)
    lid = torch.cat([torch.full((l,), k, dtype=torch.long) for k, l in enumerate(levels)]).to(dev)
    ls = torch.tensor(levels, dtype=torch.float32, device=dev)
    w_lvl = (ls ** (-0.7))[lid]

    class CloverK3NativeStepModule(nn.Module):
        def __init__(self, hl_m, cb_m, kb_m, net_m, vq_m, ptr_m, w_lvl_t):
            super().__init__()
            self.k3_hybridlink = hl_m
            self.k3_cbridge = cb_m
            self.k3_bridge = kb_m
            self.net = net_m
            self.vqvae = vq_m
            self.ptr_dec = ptr_m
            self.levels = list(vq_m.vq.levels)
            self.vsizes = list(vq_m.vq.vqvae_vocab_sizes)
            self.register_buffer("w_lvl_buf", w_lvl_t)

        def forward(self, b_hk3, wte_bpe, b_pids, ncm_in, b_idx, b_tids):
            pad_bpe = (b_pids == K3_PAD)
            # 1. Clover Kimi K3 Native HybridLink (`7168 -> 7168`) + K3ComboBridge (`7168 -> 384`)
            r_k3 = self.k3_hybridlink(b_hk3)
            z_rh = self.k3_cbridge(r_k3).float().masked_fill(pad_bpe.unsqueeze(-1), 0.0)
            # 2. Clover Kimi K3 Expert Bridge (`7168 -> 384`)
            z_k3_exp = self.k3_bridge(b_hk3).masked_fill(pad_bpe.unsqueeze(-1), 0.0)
            z_bpe = wte_bpe + z_k3_exp
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
                z_hats, _ = self.vqvae.vq.idx_to_zhats(pred_idx, return_zlevels=True)
                z_hat_top = z_hats[-1]

            lg_tok = self.ptr_dec(self.vqvae.decoder, z_hat_top, z_rh, pad_bpe, z_bpe, b_pids)
            loss_tok = F.cross_entropy(lg_tok.float().reshape(-1, V_K3), b_tids.reshape(-1), ignore_index=K3_PAD)
            return loss_ncm.unsqueeze(0), loss_tok.unsqueeze(0)

    step_mod = CloverK3NativeStepModule(k3_hybridlink, k3_cbridge, k3_bridge, net, vqvae, ptr_dec, w_lvl).to(dev)
    dp_mod = nn.DataParallel(step_mod, device_ids=train_devs)

    opt = torch.optim.AdamW([
        {"params": k3_hybridlink.parameters(), "lr": 1.5e-4},
        {"params": k3_cbridge.parameters(), "lr": 6.0e-4},
        {"params": k3_bridge.parameters(), "lr": 6.0e-4},
        {"params": [p for p in net.parameters() if p.requires_grad], "lr": 5.0e-5},
        {"params": ptr_dec.parameters(), "lr": 1.5e-4},
    ], betas=(0.9, 0.95), weight_decay=0.01)

    k3_hybridlink.train(); k3_cbridge.train(); k3_bridge.train(); net.train(); ptr_dec.train()
    steps = 200
    bs = len(train_devs) * 32  # 64 windows per step
    g = torch.Generator().manual_seed(42)
    t_tr = time.perf_counter()
    print(f"[Stage 3] Training Clover Kimi K3 Native HybridAgent + NCM across GPUs {train_devs} ({steps} steps, bs={bs})...", flush=True)

    for step in range(1, steps + 1):
        mult = 0.5 * (1.0 + math.cos(math.pi * step / float(steps)))
        opt.param_groups[0]["lr"] = 1.5e-4 * mult
        opt.param_groups[1]["lr"] = 6.0e-4 * mult
        opt.param_groups[2]["lr"] = 6.0e-4 * mult
        opt.param_groups[3]["lr"] = 5.0e-5 * mult
        opt.param_groups[4]["lr"] = 1.5e-4 * mult

        sel = torch.randint(0, N_tot, (bs,), generator=g)
        b_idx = idx_all[sel].long().to(dev, non_blocking=True)
        b_pids = all_pids[sel].to(dev, non_blocking=True)
        b_tids = all_tids[sel].to(dev, non_blocking=True)
        b_hk3 = all_h_k3[sel].to(dev, dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            idx_c = corrupt_codes(b_idx, levels, vqvae.vq.vqvae_vocab_sizes, 0.35, 0.10)
            ncm_in = vqvae.vq.idx_to_ncm_input(idx_c)
            wte_bpe = vqvae.encoder.wte(b_pids)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_ncm_v, loss_tok_v = dp_mod(b_hk3, wte_bpe, b_pids, ncm_in, b_idx, b_tids)
            loss_ncm = loss_ncm_v.mean()
            loss_tok = loss_tok_v.mean()
            loss = loss_ncm + 1.5 * loss_tok

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(step_mod.parameters(), 1.0)
        opt.step()

        if step % 40 == 0 or step == 1 or step == steps:
            print(
                f"  [Clover-K3-Native HA+NCM] step {step}/{steps} | loss_ncm={loss_ncm.item():.4f} | loss_tok={loss_tok.item():.4f} "
                f"| gate_k3_cb={k3_cbridge.gate.item():.4f} | tanh(alpha_k3)={math.tanh(k3_bridge.alpha.item()):.4f} "
                f"| elapsed={time.perf_counter() - t_tr:.1f}s",
                flush=True,
            )

    k3_hybridlink.eval(); k3_cbridge.eval(); k3_bridge.eval(); net.eval(); ptr_dec.eval()
    save_path = os.path.join(out_dir, "e11_clover_k3_native_ha_ncm.pt")
    torch.save({
        "vqvae": vqvae.state_dict(),
        "net": net.state_dict(),
        "ptr_dec": ptr_dec.state_dict(),
        "k3_hybridlink": k3_hybridlink.state_dict(),
        "k3_cbridge": k3_cbridge.state_dict(),
        "k3_bridge": k3_bridge.state_dict(),
        "restricted_mask": e10_sd["restricted_mask"],
    }, save_path)
    print(f"[Saved] {save_path} in {time.perf_counter() - t0:.1f}s total.", flush=True)


if __name__ == "__main__":
    main()
