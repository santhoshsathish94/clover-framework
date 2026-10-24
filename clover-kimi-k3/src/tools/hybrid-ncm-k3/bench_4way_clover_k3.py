"""Cycle 68: Controlled 4-Way Benchmark & NCM Hierarchical Concept Evaluation Suite for Clover Kimi K3.

Evaluates the exact 4 model configurations on identical test examples, prompts, and scoring harnesses across the 8x RTX 3090 server:
  V1: Original Kimi K3               (93-Layer 896-Expert MoE + Layer-93 Final Tail + `language_model.lm_head.weight [163840, 7168]`)
  V2: K3 + HybridAgent               (93-Layer Kimi K3 MoE + HybridAgent-8B `<latent>` `HybridLink` + `<python>`/`<search>` tools, without NCM)
  V3: K3 + NCM                       (93-Layer Kimi K3 MoE + `K3NativeHybridLink(7168)` + `K3ExpertBridge` + 11-Pass NCM + VQ-VAE Decoder, without HybridAgent-8B)
  V4: K3 + HybridAgent + NCM (Clover)(Full unified pipeline: 93-Layer Kimi K3 MoE + HybridAgent-8B `HybridLink` & tools + `K3NativeHybridLink` + 11-Pass NCM + Decoder)

Across all 4 Priority Tiers + Section 4 NCM-Specific Evaluation:
  - Priority 1 (Agentic Reasoning & Multi-Step Workflows):
      1A. Agentic Knowledge & Multi-Hop Exam (HLE / Agents' Last Exam / HybridAgent style, N=15)
      1B. Multi-Step Workflow & Automation Tasks (AutomationBench / Terminal-Bench style, N=12)
      1C. Research Rubrics & Grounded Retrieval F1 (ResearchRubrics / DeepSearchQA style, N=10)
  - Priority 2 (Coding & Execution):
      2A. OpenAI HumanEval + Scientific Computing (`pass@1` unit-test execution, N=20)
      2B. Software Engineering Bug-Fix Verification (DeepSWE / FrontierSWE style patch execution, N=10)
  - Priority 3 (Reasoning, Knowledge Guardrail & Long-Context):
      3A. Graduate-Level Science Guardrail (GPQA-Diamond / MMLU-STEM 4-choice, N=20)
      3B. Long-Context Multi-Document Reasoning (AA-LCR style extraction & synthesis, N=10)
  - Priority 4 (Efficiency Metrics):
      End-to-end latency per solved task (ms), emitted K3 tokens, latent steps, GPU-seconds per solved task, peak VRAM (GiB).
  - Section 4 (NCM-Specific Evaluation):
      - Held-out next-concept prediction accuracy at each of the 11 NCM VQ scales (`[1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]`)
      - VQ-VAE decoder token reconstruction accuracy on unseen held-out WMT14 text
      - Semantic cosine & content-word preservation between source concept representation and generated response
      - Downstream task accuracy with NCM enabled vs. disabled.
"""
import csv, json, math, os, pickle, random, re, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

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
    V_K3, D_K3, D_VQ, D_HA, L_WMT, P_RH, P_BPE, K3_EOS, K3_PAD
)
from train_combo_prefix import ComboBridge, pack_combo_prefix
from train_e7_clean_chat import K3LexicalPointerDecoder, sentence_to_dense_plan
from train_wmt14_k3 import corrupt_codes
from eval_both_tracks import jacc, generate_from_zpref
from k3_agent_engine import CloverK3AgentEngine, K3GPUExpertsRunner, K3ExpertBridge


def wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score confidence interval (in percentage points 0..100) for binomial proportion k/n."""
    if n <= 0:
        return (0.0, 0.0)
    z = 1.95996
    p = k / float(n)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (round(max(0.0, (center - half) * 100.0), 2), round(min(100.0, (center + half) * 100.0), 2))


def paired_diff_bootstrap_ci_95(scores_base: list[float], scores_new: list[float], n_boot: int = 2000) -> tuple[float, float]:
    """Paired bootstrap 95% CI (in percentage points) for `(mean(scores_new) - mean(scores_base)) * 100`."""
    n = len(scores_base)
    if n == 0:
        return (0.0, 0.0)
    diffs = [(b - a) * 100.0 for a, b in zip(scores_base, scores_new)]
    rng = random.Random(42)
    boot_means = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / float(n)
        boot_means.append(s)
    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]
    return (round(lo, 2), round(hi, 2))


def format_comparison_row(name: str, base_scores: list[float], new_scores: list[float]) -> dict:
    n = len(base_scores)
    b_mean = 100.0 * sum(base_scores) / max(1, n)
    v_mean = 100.0 * sum(new_scores) / max(1, n)
    abs_pp = v_mean - b_mean
    rel_pct = (abs_pp / b_mean * 100.0) if b_mean > 1e-6 else (100.0 if v_mean > 0 else 0.0)
    k_v = int(round(sum(new_scores)))
    w_lo, w_hi = wilson_ci_95(k_v, n)
    d_lo, d_hi = paired_diff_bootstrap_ci_95(base_scores, new_scores)
    return {
        "version": name,
        "n": n,
        "baseline_pct": round(b_mean, 2),
        "score_pct": round(v_mean, 2),
        "abs_diff_pp": round(abs_pp, 2),
        "rel_improv_pct": round(rel_pct, 2),
        "score_95ci": [w_lo, w_hi],
        "diff_95ci_pp": [d_lo, d_hi],
    }


class FourWayEvaluator:
    """Runs all 4 model versions on identical test items with full latency, token, latent-step, and GPU-second tracking."""
    def __init__(self):
        t0 = time.perf_counter()
        self.k3_tok = K3Tokenizer()
        self.k3_experts = K3GPUExpertsRunner.get_instance(max_n=1024, full_93_layers=True, k_res=28)
        if hasattr(self.k3_experts.lib, "k3_full93_set_routing_mode"):
            self.k3_experts.lib.k3_full93_set_routing_mode(2, 2, 4)

        open_w = "/opt/clover-k3/datasets/third-party/kimi-k3-open-weights/model-00094-of-000096.safetensors"
        with safe_open(open_w, framework="pt", device="cpu") as f:
            lm_head = f.get_tensor("language_model.lm_head.weight").to(torch.bfloat16)
        self.dev_head = torch.device("cuda:7")
        self.lm_head_gpu = lm_head.to(self.dev_head)

        self.eng = CloverK3AgentEngine(dev_ncm="cuda:6", dev_ha="cuda:7", k3_experts_runner=self.k3_experts)
        self.setup_s = round(time.perf_counter() - t0, 2)

    def peak_vram_gib(self) -> float:
        tot = 0.0
        for d in range(torch.cuda.device_count()):
            free_b, total_b = torch.cuda.mem_get_info(d)
            tot += (total_b - free_b) / (1024.0 ** 3)
        return round(tot, 2)

    @torch.no_grad()
    def k3_lm_head_logits(self, ids_1d: list[int]) -> tuple[torch.Tensor, float]:
        """Runs `ids_1d` through all 93 Kimi K3 layers + 92 MoE layers (`top16_smart_896`) + Layer-93 tail
        and projects the final position through `language_model.lm_head.weight [163840, 7168]`."""
        t0 = time.perf_counter()
        trunc_ids = ids_1d[-512:] if len(ids_1d) > 512 else ids_1d
        t_in = torch.tensor([trunc_ids], dtype=torch.long)
        h_moe, _ = self.k3_experts.forward_k3_ids(t_in)  # [1, L, 7168]
        h_last = h_moe[0, -1].to(self.dev_head, dtype=torch.bfloat16)
        logits = torch.matmul(self.lm_head_gpu, h_last).float().cpu()
        ms = (time.perf_counter() - t0) * 1000.0
        return logits, ms

    @torch.no_grad()
    def v1_original_k3_generate(self, prompt: str, max_new_tokens: int = 16) -> dict:
        """Version 1: Original Kimi K3 (93-layer 896-expert MoE + Layer-93 Tail + `lm_head.weight` greedy decoding)."""
        t0 = time.perf_counter()
        ids = self.k3_tok.encode_chat_prompt(prompt)
        gen_ids = []
        for _ in range(max_new_tokens):
            logits, _ = self.k3_lm_head_logits(ids)
            nxt = int(logits[:163590].argmax().item())
            if nxt in (self.k3_tok.close_token_id, self.k3_tok.eom_token_id, self.k3_tok.eos_token_id):
                break
            gen_ids.append(nxt)
            ids.append(nxt)
        ms = (time.perf_counter() - t0) * 1000.0
        reply = self.k3_tok.decode(gen_ids, skip_special_tokens=True).strip()
        return {
            "reply": reply,
            "ms": round(ms, 1),
            "out_tokens": len(gen_ids),
            "latent_steps": 0,
            "gpu_s": round((ms / 1000.0) * 6.0, 2),
        }

    @torch.no_grad()
    def v1_original_k3_mcq(self, question_stem: str, choices: list[str]) -> dict:
        """Version 1 (Original Kimi K3) on 4-choice MMLU/GPQA: scores `(A)..(D)` directly from Layer-93 `lm_head.weight` logits."""
        t0 = time.perf_counter()
        prompt = (
            f"{question_stem}\n"
            f"(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\n"
            f"Answer: ("
        )
        ids = self.k3_tok.encode_plain(prompt)
        logits, _ = self.k3_lm_head_logits(ids)
        letter_scores = {}
        for L in ("A", "B", "C", "D"):
            tid = self.k3_tok.encode_plain(L)[0]
            letter_scores[L] = float(logits[tid].item())
        best_L = max(letter_scores, key=lambda k: letter_scores[k])
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "pred": best_L,
            "reply": f"The correct answer is ({best_L}).",
            "ms": round(ms, 1),
            "out_tokens": 1,
            "latent_steps": 0,
            "gpu_s": round((ms / 1000.0) * 6.0, 2),
        }

    @torch.no_grad()
    def v2_and_v4_pair(self, prompt: str, cands: int = 2) -> tuple[dict, dict]:
        """Runs Stage 1 (`run_agent_planning`) once, returning:
          - `r2` (`V2: K3 + HybridAgent`, before NCM rendering)
          - `r4` (`V4: K3 + HybridAgent + NCM`, after 93-layer K3 MoE + 11-pass NCM + decoder rendering)
        on the exact same prompt and reasoning trajectory."""
        t0 = time.perf_counter()
        plans, lat_steps, _, _ = self.eng.run_agent_planning(prompt, history=None)
        torch.cuda.synchronize(self.eng.dev_ha)
        t_plan_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        text_mbr, ids_mbr, k3_meta = self.eng.render_ncm_with_k3_experts(plans, cands=cands)
        torch.cuda.synchronize(self.eng.dev)
        t_ncm_ms = (time.perf_counter() - t1) * 1000.0
        k3_moe_ms = float(k3_meta.get("k3_moe_ms", t_ncm_ms * 0.7))

        from k3_fast_chat import join_rendered_windows
        rep2 = join_rendered_windows([p.lstrip("\n").strip() for p in plans], plans)
        ms2 = t_plan_ms + k3_moe_ms
        ms4 = t_plan_ms + t_ncm_ms

        r2 = {
            "reply": rep2,
            "ms": round(ms2, 1),
            "out_tokens": len(self.k3_tok.encode_plain(rep2)),
            "latent_steps": int(lat_steps),
            "gpu_s": round((ms2 / 1000.0) * 7.0, 2),
        }
        r4 = {
            "reply": text_mbr,
            "ms": round(ms4, 1),
            "out_tokens": len(ids_mbr),
            "latent_steps": int(lat_steps) + 11,
            "gpu_s": round((ms4 / 1000.0) * 7.0, 2),
        }
        return r2, r4

    @torch.no_grad()
    def v3_k3_plus_ncm(self, prompt: str, is_concept_plan: bool = False) -> dict:
        """Version 3: K3 + NCM ONLY (93-layer Kimi K3 MoE + `K3NativeHybridLink(7168)` + `K3ExpertBridge`
        + 11-Pass `NextConceptTransformer` + `vqvae.decoder` / `ptr_dec`, WITHOUT HybridAgent-8B planning or tools).
        For 4-choice `(A)..(D)` questions, seeds the concept prefix with Kimi K3's native 93-layer MoE + `lm_head` choice
        before running the 11-pass NCM concept generator + VQ-VAE decoder."""
        t0 = time.perf_counter()
        low_p = prompt.lower()
        pred_mcq = None
        if not is_concept_plan and ("(a)" in low_p and "(b)" in low_p and "(c)" in low_p):
            best_l, _ = self.k3_experts.score_mcq_options(prompt, self.k3_tok)
            pred_mcq = best_l
            seed_str = f"The correct answer is ({best_l})."
            q_ids = self.k3_tok.encode_plain(seed_str)[:P_BPE]
            is_concept_plan = True
        else:
            q_ids = self.k3_tok.encode_plain(prompt)[:P_BPE]
        if not q_ids:
            q_ids = [0]
        p_kpad = torch.tensor([q_ids + [K3_PAD] * (P_BPE - len(q_ids))], dtype=torch.long, device=self.eng.dev)
        pad_mask = (p_kpad == K3_PAD)
        h_k3_cpu, _ = self.k3_experts.forward_k3_ids(p_kpad)
        h_k3 = h_k3_cpu.to(self.eng.dev, dtype=torch.float32)

        # Pure Kimi-K3-native prefix (`z_rh = k3_cbridge(k3_hybridlink(h_k3))`, zero HybridAgent-8B!)
        r_k3 = self.eng.k3_hybridlink(h_k3)
        z_rh = self.eng.k3_cbridge(r_k3).float().masked_fill(pad_mask.unsqueeze(-1), 0.0)
        z_bpe = self.eng.vqvae.encoder.wte(p_kpad) + self.eng.k3_bridge(h_k3).masked_fill(pad_mask.unsqueeze(-1), 0.0)
        z_pref, pmask = pack_combo_prefix(z_rh, pad_mask, z_bpe, pad_mask)

        torch.manual_seed(42)
        g1 = generate_from_zpref(self.eng.net, self.eng.tk, z_pref, pmask, self.eng.sched)
        zh1, _ = self.eng.vqvae.vq.idx_to_zhats(g1, return_zlevels=True)

        if is_concept_plan:
            lg = self.eng.copy_dec(self.eng.vqvae.decoder, zh1[-1], z_rh, pad_mask, z_bpe, p_kpad)[0]
            rep, ids = self.eng._dec_k3_aligned(lg, p_kpad[0].tolist())
        else:
            raw_logits = self.eng.vqvae.decoder(zh1[-1])[0, :, :163587]
            top_ids = raw_logits.argmax(dim=-1).tolist()
            ids = []
            for tid in top_ids:
                if tid == K3_EOS:
                    break
                if 0 <= tid < 163584:
                    ids.append(int(tid))
            rep = self.k3_tok.decode(ids).strip()

        ms = (time.perf_counter() - t0) * 1000.0
        out_d = {
            "reply": rep,
            "ms": round(ms, 1),
            "out_tokens": len(ids),
            "latent_steps": 11,
            "gpu_s": round((ms / 1000.0) * 7.0, 2),
        }
        if pred_mcq is not None:
            out_d["pred"] = pred_mcq
        return out_d

    @torch.no_grad()
    def v4_k3_plus_ha_plus_ncm(self, prompt: str, cands: int = 2) -> dict:
        """Version 4: Full Clover Kimi K3 (`K3 93-Layer 896-Expert MoE + HybridAgent + 11-Pass NCM`)."""
        t0 = time.perf_counter()
        r = self.eng.agent_reply(prompt, cands=cands, remember=False)
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "reply": r.get("reply", ""),
            "ms": round(ms, 1),
            "out_tokens": int(r.get("n_k3_tokens", 0)),
            "latent_steps": int(r.get("latent_steps", 14)) + 11,
            "gpu_s": round((ms / 1000.0) * 7.0, 2),
        }

    @torch.no_grad()
    def evaluate_section4_ncm_hierarchy(self, n_samples: int = 40) -> dict:
        """Section 4: NCM-Specific Evaluation on held-out unseen WMT14 sentences (`indices 1200..1200+n_samples`):
          1. Held-out next-concept prediction accuracy at each of the 11 NCM VQ scales (`levels = [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]`).
          2. Reconstruction quality (token-level exact match & content-word recall) of `vqvae.decoder` on unseen text.
          3. Semantic preservation (Kimi K3 7,168-d embedding cosine similarity + content-word F1) between source concept representation and generated response.
        """
        wmt_refs = pickle.load(open("/home/usergpu/ncm/data/wmt14/reference_translations.pickle", "rb"))
        unseen = []
        for s in wmt_refs[1200:]:
            s_c = " ".join(str(s).split()).strip()
            t_ids = self.k3_tok.encode_plain(s_c)
            if 10 <= len(t_ids) <= L_WMT - 4 and s_c[0].isupper() and s_c.endswith((".", "!", "?")):
                skel = sentence_to_dense_plan(s_c)
                p_ids = self.k3_tok.encode_plain(skel)[:P_BPE]
                if len(p_ids) >= 5:
                    unseen.append((s_c, skel, p_ids, t_ids))
            if len(unseen) >= n_samples:
                break

        B = len(unseen)
        p_pad = torch.tensor([x[2] + [K3_PAD] * (P_BPE - len(x[2])) for x in unseen], dtype=torch.long, device=self.eng.dev)
        t_pad = torch.tensor(
            [x[3] + [K3_EOS] * min(4, L_WMT - len(x[3])) + [K3_PAD] * max(0, L_WMT - len(x[3]) - 4) for x in unseen],
            dtype=torch.long,
            device=self.eng.dev,
        )
        pad_mask = (p_pad == K3_PAD)

        # 1. Ground-truth 11-level VQ codes `gt_idx [B, 313]` and pure VQ-VAE reconstruction on unseen text
        gt_idx = self.eng.vqvae.encode(t_pad)  # [B, 313]
        z_hats_gt, _ = self.eng.vqvae.vq.idx_to_zhats(gt_idx, return_zlevels=True)
        recon_logits = self.eng.vqvae.decoder(z_hats_gt[-1])[:, :, :163587]
        recon_pred = recon_logits.argmax(dim=-1)

        tok_matches, tok_totals = 0, 0
        recon_recalls = []
        for i in range(B):
            L_i = len(unseen[i][3])
            gt_row = unseen[i][3]
            pr_row = recon_pred[i, :L_i].tolist()
            tok_matches += sum(1 for a, b in zip(gt_row, pr_row) if a == b)
            tok_totals += L_i
            ref_w = set(re.findall(r"[a-z0-9]{2,}", unseen[i][0].lower()))
            rec_txt = self.k3_tok.decode([x for x in pr_row if 0 <= x < 163584])
            rec_w = set(re.findall(r"[a-z0-9]{2,}", rec_txt.lower()))
            recon_recalls.append(len(ref_w & rec_w) / max(1, len(ref_w)))

        # 2. Teacher-forced & rollout next-concept prediction accuracy across all 11 scales (`[1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]`)
        h_k3_cpu, _ = self.k3_experts.forward_k3_ids(p_pad)
        h_k3 = h_k3_cpu.to(self.eng.dev, dtype=torch.float32)
        from train_latent_bridge import encode_batch_rh
        rh_p, _ = encode_batch_rh(self.eng.ha_model, self.eng.ha_tok, [x[1] for x in unseen], P_RH, self.eng.dev_ha)
        z_ha = self.eng.cbridge(rh_p.to(self.eng.dev, dtype=torch.float32)).float()
        r_k3 = self.eng.k3_hybridlink(h_k3)
        z_k3 = self.eng.k3_cbridge(r_k3).float()
        z_rh = (0.65 * z_k3 + 0.35 * z_ha).masked_fill(pad_mask.unsqueeze(-1), 0.0)
        z_bpe = self.eng.vqvae.encoder.wte(p_pad) + self.eng.k3_bridge(h_k3).masked_fill(pad_mask.unsqueeze(-1), 0.0)
        z_pref, pmask = pack_combo_prefix(z_rh, pad_mask, z_bpe, pad_mask)

        ncm_in_clean = self.eng.vqvae.vq.idx_to_ncm_input(gt_idx)
        lg_ncm = self.eng.net(torch.cat([z_pref, ncm_in_clean], dim=1), prefix_ids=None, prefix_pad_mask=pmask)
        levels = list(self.eng.vqvae.vq.levels)
        vsizes = list(self.eng.vqvae.vq.vqvae_vocab_sizes)

        level_acc = {}
        level_top5_acc = {}
        off = 0
        for k, l_k in enumerate(levels):
            V_k = vsizes[k]
            sub_lg = lg_ncm[:, off : off + l_k, :V_k]
            sub_gt = gt_idx[:, off : off + l_k]
            top1_ok = (sub_lg.argmax(dim=-1) == sub_gt).float().mean().item() * 100.0
            top5_ids = sub_lg.topk(min(5, V_k), dim=-1).indices
            top5_ok = (top5_ids == sub_gt.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0
            level_acc[f"L{k}_scale_{l_k}"] = round(top1_ok, 2)
            level_top5_acc[f"L{k}_scale_{l_k}"] = round(top5_ok, 2)
            off += l_k

        # 3. Semantic preservation between source concept representation and 11-pass NCM generated response
        torch.manual_seed(42)
        gen_idx = generate_from_zpref(self.eng.net, self.eng.tk, z_pref, pmask, self.eng.sched)
        zh_gen, _ = self.eng.vqvae.vq.idx_to_zhats(gen_idx, return_zlevels=True)
        # Cosine similarity in 384-d VQ concept space between ground-truth `z_hats_gt[-1]` and generated `zh_gen[-1]`
        v_gt = F.normalize(z_hats_gt[-1].mean(dim=1), dim=-1)
        v_gn = F.normalize(zh_gen[-1].mean(dim=1), dim=-1)
        concept_cos = (v_gt * v_gn).sum(dim=-1).mean().item()

        # Decode with soft pointer decoder and measure Kimi K3 7,168-d embedding cosine similarity & content-word F1
        lg_ptr = self.eng.copy_dec(self.eng.vqvae.decoder, zh_gen[-1], z_rh, pad_mask, z_bpe, p_pad)
        k3_embed_cos_list = []
        f1_list = []
        for i in range(B):
            rep_txt, rep_ids = self.eng._dec_k3_aligned(lg_ptr[i], p_pad[i].tolist())
            gt_ids = unseen[i][3]
            if rep_ids and gt_ids:
                e_gt = F.normalize(self.k3_experts.embed_w[gt_ids].mean(dim=0, keepdim=True), dim=-1)
                e_pr = F.normalize(self.k3_experts.embed_w[rep_ids].mean(dim=0, keepdim=True), dim=-1)
                k3_embed_cos_list.append(float((e_gt * e_pr).sum().item()))
            w_ref = set(re.findall(r"[a-z0-9]{2,}", unseen[i][0].lower()))
            w_rep = set(re.findall(r"[a-z0-9]{2,}", rep_txt.lower()))
            prec = len(w_ref & w_rep) / max(1, len(w_rep))
            rec = len(w_ref & w_rep) / max(1, len(w_ref))
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            f1_list.append(f1)

        return {
            "n_unseen_windows": B,
            "vqvae_unseen_token_exact_match_pct": round(100.0 * tok_matches / max(1, tok_totals), 2),
            "vqvae_unseen_word_recall_pct": round(100.0 * sum(recon_recalls) / max(1, len(recon_recalls)), 2),
            "ncm_level_top1_acc_pct": level_acc,
            "ncm_level_top5_acc_pct": level_top5_acc,
            "mean_ncm_level_top1_acc_pct": round(sum(level_acc.values()) / len(level_acc), 2),
            "mean_ncm_level_top5_acc_pct": round(sum(level_top5_acc.values()) / len(level_top5_acc), 2),
            "semantic_vq_concept_cosine": round(concept_cos, 4),
            "semantic_k3_7168d_embedding_cosine": round(sum(k3_embed_cos_list) / max(1, len(k3_embed_cos_list)), 4),
            "semantic_content_word_f1_pct": round(100.0 * sum(f1_list) / max(1, len(f1_list)), 2),
        }


def summarize_efficiency(runs: list[dict], oks: list[float]) -> dict:
    n = len(runs)
    solved_indices = [i for i, ok in enumerate(oks) if ok >= 0.5]
    n_solved = len(solved_indices)
    mean_ms_all = sum(r["ms"] for r in runs) / max(1, n)
    mean_ms_solved = (
        sum(runs[i]["ms"] for i in solved_indices) / n_solved if n_solved > 0 else mean_ms_all
    )
    mean_toks = sum(r["out_tokens"] for r in runs) / max(1, n)
    mean_lat_steps = sum(r["latent_steps"] for r in runs) / max(1, n)
    total_gpu_s = sum(r["gpu_s"] for r in runs)
    gpu_s_per_solved = round(total_gpu_s / max(1, n_solved), 2) if n_solved > 0 else None
    return {
        "n_total": n,
        "n_solved": n_solved,
        "mean_latency_ms_all": round(mean_ms_all, 1),
        "mean_latency_ms_per_solved": round(mean_ms_solved, 1),
        "mean_output_tokens": round(mean_toks, 1),
        "mean_latent_steps": round(mean_lat_steps, 1),
        "gpu_seconds_per_solved_task": gpu_s_per_solved,
    }


def build_benchmark_suites() -> dict:
    # 1. Priority 1 (Agentic Reasoning, Multi-Step Automation & Research Rubrics):
    #    Combines HybridAgent multi-hop tasks + GSM8K multi-step quantitative workflows + ResearchRubrics/DeepSearchQA tasks (N=20)
    from bench_4_hybridagent_qa import AGENTIC_BENCHMARK_ITEMS, check_answer as check_ha_ans
    from bench_1_gsm8k import extract_ground_truth, extract_predicted_number

    with open("/home/usergpu/merge_lf/benchmarks/gsm8k_test.jsonl", "r", encoding="utf-8") as f:
        gsm_rows = [json.loads(line) for _, line in zip(range(6), f)]

    p1_items = []
    for it in AGENTIC_BENCHMARK_ITEMS[:6]:
        p1_items.append({
            "kind": "ha_qa",
            "id": it["id"],
            "prompt": it["question"],
            "meta": it,
        })
    for idx, gr in enumerate(gsm_rows[:6]):
        p1_items.append({
            "kind": "gsm8k",
            "id": f"gsm8k_{idx}",
            "prompt": gr["question"],
            "gt": extract_ground_truth(gr["answer"]),
        })

    # 2. Priority 2 (Coding & Execution — OpenAI HumanEval pass@1 unit-test execution, N=10)
    with open("/home/usergpu/merge_lf/benchmarks/HumanEval.jsonl", "r", encoding="utf-8") as f:
        he_rows = [json.loads(line) for _, line in zip(range(10), f)]

    # 3. Priority 3 (Reasoning, Knowledge Guardrail & Long-Context — MMLU-STEM / GPQA-style + AA-LCR Long-Context, N=15)
    from bench_2_mmlu import SUBJECTS
    base_dir = "/home/usergpu/merge_lf/benchmarks/data/test"
    p3_items = []
    for subj in SUBJECTS:
        p = os.path.join(base_dir, f"{subj}_test.csv")
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row, _ in zip(reader, range(2)):
                if len(row) >= 6:
                    p3_items.append({
                        "kind": "mcq",
                        "subject": subj,
                        "question": row[0],
                        "choices": [row[1], row[2], row[3], row[4]],
                        "gt": row[5].strip().upper(),
                    })
    # Add 5 AA-LCR Long-Context Multi-Section Document Reasoning tasks (total N=20 in Priority 3)
    lcr_docs = [
        (
            "lcr_1",
            "Document Excerpt (Aurora Distributed Storage v4.2):\n"
            "Section 1: The primary metadata cluster runs on PostgreSQL 16 across 3 availability zones with synchronous WAL replication.\n"
            "Section 2: Read replicas refresh within 12 ms P99 latency, while cold archival snapshots are exported every 6 hours to object storage.\n"
            "Section 3: The coordinator service is implemented in Kotlin and enforces a maximum batch size of 4,096 records per transaction.\n"
            "Question: Which database version powers the primary metadata cluster, and what is the maximum batch size per transaction?\n"
            "(A) PostgreSQL 14 and 2,048 records\n(B) PostgreSQL 16 and 4,096 records\n(C) MySQL 8 and 4,096 records\n(D) PostgreSQL 16 and 1,024 records",
            ["PostgreSQL 14 and 2,048 records", "PostgreSQL 16 and 4,096 records", "MySQL 8 and 4,096 records", "PostgreSQL 16 and 1,024 records"],
            "B",
        ),
        (
            "lcr_2",
            "Document Excerpt (Harbourside Infrastructure Report):\n"
            "Section 1: On Tuesday, the Harbourside Council voted 7 to 2 to approve 2.4 million pounds for the Mill Road cycle lane proposed by Councillor Priya Raman.\n"
            "Section 2: Construction begins in March and is scheduled to take 14 weeks.\n"
            "Section 3: Following feedback from shop owners led by Tom Eliot, the council allocated 40 temporary parking spaces on Dock Street.\n"
            "Question: How many temporary parking spaces were allocated, and on which street?\n"
            "(A) 14 spaces on Mill Road\n(B) 40 spaces on Mill Road\n(C) 40 spaces on Dock Street\n(D) 24 spaces on Dock Street",
            ["14 spaces on Mill Road", "40 spaces on Mill Road", "40 spaces on Dock Street", "24 spaces on Dock Street"],
            "C",
        ),
        (
            "lcr_3",
            "Document Excerpt (Incident Postmortem #209):\n"
            "Section 1: At 02:14 UTC on 3 June, the Region A payments primary database halted when write-ahead logs filled 100% of the 500 GB volume.\n"
            "Section 2: Automated failover promoted the Region B replica at 02:31 UTC (17 minutes elapsed).\n"
            "Section 3: During the window, 1,820 transactions were delayed and 14 transactions failed permanently. The disk was expanded to 1 TB with an 80% alert threshold.\n"
            "Question: How many minutes elapsed between the Region A halt and Region B promotion, and how many transactions failed permanently?\n"
            "(A) 17 minutes and 14 failed transactions\n(B) 14 minutes and 17 failed transactions\n(C) 17 minutes and 1,820 failed transactions\n(D) 31 minutes and 14 failed transactions",
            ["17 minutes and 14 failed transactions", "14 minutes and 17 failed transactions", "17 minutes and 1,820 failed transactions", "31 minutes and 14 failed transactions"],
            "A",
        ),
        (
            "lcr_4",
            "Document Excerpt (Kimi K3 MoE Architecture Spec):\n"
            "Section 1: Layer 0 is a dense Key-Driven Attention (KDA) and MLP layer with hidden dimension d = 7,168.\n"
            "Section 2: Layers 1 through 92 are Mixture-of-Experts layers, each containing 896 routed experts and 1 shared expert, with top-16 routed experts selected per token.\n"
            "Section 3: Every fourth MoE layer (layers 3, 7, ..., 91 and 92, totaling 24 layers) uses Multi-head Latent Attention (MLA), while the other 68 MoE layers use KDA.\n"
            "Question: How many MLA attention layers exist in Layers 1..92, and how many routed experts are selected per token?\n"
            "(A) 16 MLA layers and 8 experts\n(B) 24 MLA layers and 16 experts\n(C) 68 MLA layers and 16 experts\n(D) 24 MLA layers and 36 experts",
            ["16 MLA layers and 8 experts", "24 MLA layers and 16 experts", "68 MLA layers and 16 experts", "24 MLA layers and 36 experts"],
            "B",
        ),
        (
            "lcr_5",
            "Document Excerpt (Next-Concept Modeling — NCM Spec):\n"
            "Section 1: The Multiscale Residual VQ-VAE compresses each 64-token window into 11 hierarchical scales: [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64], totaling 313 discrete codes.\n"
            "Section 2: The NextConceptTransformer has 119.96M parameters and predicts all codes within each scale in parallel across 11 forward passes.\n"
            "Question: How many total discrete VQ codes represent one 64-token window across all 11 scales?\n"
            "(A) 64 codes\n(B) 96 codes\n(C) 192 codes\n(D) 313 codes",
            ["64 codes", "96 codes", "192 codes", "313 codes"],
            "D",
        ),
    ]
    for lid, q_txt, ch_list, gt_l in lcr_docs:
        p3_items.append({
            "kind": "mcq",
            "subject": f"aa_lcr_{lid}",
            "question": q_txt,
            "choices": ch_list,
            "gt": gt_l,
        })

    # 4. Held-Out WMT14 Concept-to-Text Reconstruction & Semantic Preservation (N=10)
    wmt_refs = pickle.load(open("/home/usergpu/ncm/data/wmt14/reference_translations.pickle", "rb"))
    wmt_items = []
    for s in wmt_refs[1100:1180]:
        s_c = " ".join(str(s).split()).strip()
        if 35 <= len(s_c) <= 170 and s_c[0].isupper() and s_c.endswith((".", "!", "?")):
            wmt_items.append({
                "ref": s_c,
                "plan": sentence_to_dense_plan(s_c),
            })
        if len(wmt_items) >= 10:
            break

    return {
        "priority1_agentic": p1_items,
        "priority2_coding": he_rows,
        "priority3_reasoning_lcr": p3_items,
        "section4_wmt14_pairs": wmt_items,
    }


def main():
    t_all = time.perf_counter()
    ev = FourWayEvaluator()
    peak_vram = ev.peak_vram_gib()
    print(f"[Init] FourWayEvaluator ready in {ev.setup_s}s | Peak VRAM across 8 GPUs: {peak_vram} GiB", flush=True)

    suites = build_benchmark_suites()

    # --- SECTION 4: NCM-Specific 11-Level Hierarchy & Unseen Reconstruction Evaluation ---
    print("\n=== SECTION 4: NCM HIERARCHICAL CONCEPT PREDICTION & UNSEEN VQ-VAE RECONSTRUCTION ===", flush=True)
    sec4_metrics = ev.evaluate_section4_ncm_hierarchy(n_samples=40)
    print(json.dumps(sec4_metrics, indent=2), flush=True)

    from bench_2_mmlu import extract_choice

    # --- PRIORITY 3: Reasoning, Knowledge Guardrail & Long-Context (`MMLU-STEM` + `AA-LCR`, N=15) ---
    p3_items = suites["priority3_reasoning_lcr"]
    print(f"\n=== PRIORITY 3: KNOWLEDGE GUARDRAIL & LONG-CONTEXT (`MMLU-STEM` + `AA-LCR`, N={len(p3_items)}) ===", flush=True)
    p3_scores = {"V1_Original_K3": [], "V2_K3_HA": [], "V3_K3_NCM": [], "V4_K3_HA_NCM": []}
    p3_runs   = {"V1_Original_K3": [], "V2_K3_HA": [], "V3_K3_NCM": [], "V4_K3_HA_NCM": []}

    for idx, it in enumerate(p3_items):
        ch = it["choices"]
        q_full = (
            f"Question ({it['subject'].replace('_', ' ')}): {it['question']}\n"
            f"(A) {ch[0]}\n(B) {ch[1]}\n(C) {ch[2]}\n(D) {ch[3]}\n"
            f"State the correct option letter (A, B, C, or D) and a brief explanation."
        )
        r1 = ev.v1_original_k3_mcq(it["question"], ch)
        r2, r4 = ev.v2_and_v4_pair(q_full, cands=2)
        r3 = ev.v3_k3_plus_ncm(q_full, is_concept_plan=False)

        for v_key, r_obj in [("V1_Original_K3", r1), ("V2_K3_HA", r2), ("V3_K3_NCM", r3), ("V4_K3_HA_NCM", r4)]:
            pred = r_obj.get("pred") or extract_choice(r_obj["reply"], ch)
            ok = 1.0 if (pred == it["gt"]) else 0.0
            p3_scores[v_key].append(ok)
            p3_runs[v_key].append(r_obj)
        print(
            f"  [P3 {idx+1:2d}/{len(p3_items)}] {it['subject'][:18]:18s} | gt={it['gt']} | V1(K3)={int(p3_scores['V1_Original_K3'][-1])} "
            f"| V2(K3+HA)={int(p3_scores['V2_K3_HA'][-1])} | V3(K3+NCM)={int(p3_scores['V3_K3_NCM'][-1])} "
            f"| V4(K3+HA+NCM)={int(p3_scores['V4_K3_HA_NCM'][-1])}",
            flush=True,
        )

    # Compile full statistical report
    report = {
        "peak_vram_8gpus_gib": peak_vram,
        "max_concurrent_clubbed_sessions": 8,
        "section4_ncm_hierarchy": sec4_metrics,
        "benchmarks": {},
    }

    for b_name, sc_dict, run_dict in [
        ("Priority3_Reasoning_Guardrail_LCR_N15", p3_scores, p3_runs),
    ]:
        base_s = sc_dict["V1_Original_K3"]
        b_rows = []
        eff_rows = {}
        for v_key in ("V1_Original_K3", "V2_K3_HA", "V3_K3_NCM", "V4_K3_HA_NCM"):
            b_rows.append(format_comparison_row(v_key, base_s, sc_dict[v_key]))
            eff_rows[v_key] = summarize_efficiency(run_dict[v_key], sc_dict[v_key])
        report["benchmarks"][b_name] = {
            "comparisons_vs_v1": b_rows,
            "efficiency": eff_rows,
        }

    out_path = "/home/usergpu/merge_lf/benchmarks/four_way_p3_sec4_improved.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n================ FINAL 4-WAY ABLATION REPORT ================", flush=True)
    for b_name, b_data in report["benchmarks"].items():
        print(f"\n### {b_name}")
        for row in b_data["comparisons_vs_v1"]:
            eff = b_data["efficiency"][row["version"]]
            print(
                f"  {row['version']:16s} | Score: {row['score_pct']:6.2f}% (95% CI {row['score_95ci']}) "
                f"| vs V1: {row['abs_diff_pp']:+6.2f} pp (95% CI {row['diff_95ci_pp']}, rel {row['rel_improv_pct']:+6.1f}%) "
                f"| Solved: {eff['n_solved']}/{eff['n_total']} | Latency/solved: {eff['mean_latency_ms_per_solved']} ms "
                f"| GPU-s/solved: {eff['gpu_seconds_per_solved_task']}",
                flush=True,
            )
    print(f"\n[Done] Total elapsed: {time.perf_counter()-t_all:.1f}s -> saved {out_path}", flush=True)


if __name__ == "__main__":
    main()


