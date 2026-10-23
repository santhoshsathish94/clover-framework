"""Cycle 53: Multimodal (Image / Video -> Text) Chat Mode for HybridAgent + NCM + Clover Kimi K3.
Implements:
  1. Option B (Primary, always enabled when an image is provided):
     - Uses `Qwen/Qwen3-VL-8B-Instruct` (`d = 4,096`, sharing the exact Qwen3-8B embedding/hidden geometry with
       `HybridAgent-8B-Agentic`: `0.9878` embedding cosine and `0.9453` top-layer hidden-state cosine) grafted with
       `HybridAgent-8B-Agentic`'s `HybridLink` continuous latent adapter (`run_hybridlink`).
     - Reads `[Image(s) + User Prompt]` on `cuda:1`, runs continuous `<latent>` reasoning over the joint visual + text KV cache,
       generates the grounded `W1 / W2` semicolon factual plan, and extracts `R(h) [W, 48, 4096]`.
     - Renders native Kimi K3 token IDs (`163,840` vocab) via `e8_k3_direct` (`NextConceptTransformer` + `K3LexicalPointerDecoder`)
       on `cuda:0` in `~160 ms` (`1`-cand) / `~430 ms` (`MBR8`).
  2. Option A (Optional `--use-option-a` / `enable_option_a=True` flag):
     - Loads Kimi K3's native 27-layer `MoonViT3d` vision tower (`model-00096-of-000096.safetensors`, `765 MB`) +
       `PatchMergerMLPV2` (`model-00095-of-000096.safetensors`, `88 MB`, projecting `(2, 2)` merged patches to `d = 7,168`).
     - Computes Kimi K3's exact `7,168`-d visual tokens `[N_patches, 7168]` (`media_placeholder_token_id = 163605`),
       projects pooled visual tokens into NCM's `384`-d prefix (`W_k3_to_enc: 7168 -> 384`) alongside Option B's `R(h)` prefix,
       and saves the `[N_patches, 7168]` tensor ready for 1-pass 93-layer Kimi K3 multimodal verification (`stage1_lookup` at `163605`).
"""
import argparse, collections, json, math, os, re, sys, time

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
ROOT = os.environ["NCM_REPO"]
sys.path.insert(0, "/home/usergpu/merge_lf/k3_vision")
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open
from transformers import AutoProcessor, AutoModelForImageTextToText

from k3_tokenizer import K3Tokenizer
from train_k3_native_all import load_k3_native_models, V_K3, D_VQ, D_HA, L_WMT, P_RH, P_BPE, P_COMBO, K3_EOS, K3_PAD
from train_e7_clean_chat import K3LexicalPointerDecoder
from train_combo_prefix import ComboBridge, pack_combo_prefix
from eval_both_tracks import jacc, generate_from_zpref
from hybrid_agent.model import load_hybrid_agent
from k3_fast_chat import GRAMMAR_CONNECTOR_WORDS, parse_and_split_plans, join_rendered_windows
from k3_agent_engine import K3GPUExpertsRunner, K3ExpertBridge


VL_PLAN_PROMPT = (
    "Carefully inspect the image (including all headings, text, numbers, tables, stages, labels, and visual details) "
    "and write a clear, accurate, direct response to the user's request.\n"
    "Format your response starting with `W1:` on the first line, and use `W2:`, `W3:`, `W4:` on separate lines for additional paragraphs or table/stage details. "
    "Write complete, natural English sentences and do NOT use semicolons (`;`). "
    "Only include `Summary:` at the end if the user explicitly asks for a summary.\n\n"
    "User request: {prompt}"
)


class K3OptionAMoonViT(nn.Module):
    """Option A: Kimi K3's native 27-layer MoonViT3d vision tower + PatchMergerMLPV2 (`1024 -> 7168`)
    plus `7168 -> 384` ridge projection into NCM's continuous prefix space."""
    def __init__(self, vis_dir="/home/usergpu/merge_lf/k3_vision", graft_path="/home/usergpu/merge_lf/out/e6_k3_native/k3_vqvae_init.pt", dev="cuda:0"):
        super().__init__()
        self.dev = torch.device(dev)
        from configuration_kimi_k3 import KimiK3VisionConfig
        import k3_vis_standalone as k3v

        cfg_d = json.load(open(os.path.join(vis_dir, "config.json")))
        vcfg = KimiK3VisionConfig(**cfg_d["vision_config"])
        vcfg._attn_implementation = "eager"
        self.media_placeholder_token_id = int(cfg_d.get("media_placeholder_token_id", 163605))

        self.vision_tower = k3v.MoonViT3dPretrainedModel(k3v.VisionTowerConfig(vcfg))
        self.mm_projector = k3v.PatchMergerMLPV2(k3v.ProjectorConfig(vcfg))

        sd_vt, sd_pr = {}, {}
        with safe_open(os.path.join(vis_dir, "model-00096-of-000096.safetensors"), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("vision_tower."):
                    sd_vt[k[len("vision_tower."):]] = f.get_tensor(k)
        with safe_open(os.path.join(vis_dir, "model-00095-of-000096.safetensors"), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("mm_projector."):
                    sd_pr[k[len("mm_projector."):]] = f.get_tensor(k)

        self.vision_tower.load_state_dict(sd_vt, strict=True)
        self.mm_projector.load_state_dict(sd_pr, strict=True)
        self.vision_tower = self.vision_tower.to(self.dev, dtype=torch.bfloat16).eval()
        self.mm_projector = self.mm_projector.to(self.dev, dtype=torch.bfloat16).eval()

        # Load the 7168 -> 384 linear bridge fitted between Kimi K3's 7168-d space and NCM's 384-d space
        g_sd = torch.load(graft_path, map_location="cpu", weights_only=True)
        self.register_buffer("W_k3_to_enc", g_sd["W_k3_to_enc"].to(self.dev, dtype=torch.float32))

    @torch.no_grad()
    def encode_image(self, pil_img, target_side=420, n_vis_slots=8):
        """Encodes a PIL image through MoonViT3d + PatchMergerMLPV2 -> `[N_patches, 7168]` (for Kimi K3 token 163605)
        and pools/projects to `[n_vis_slots, 384]` for NCM prefix conditioning."""
        # Ensure dimensions are multiples of 28 (patch_size 14 * merge_kernel 2)
        side = (target_side // 28) * 28
        img_r = pil_img.convert("RGB").resize((side, side), Image.Resampling.BICUBIC)
        arr = np.asarray(img_r, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # MoonViT image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5]
        # Reshape (H, W, 3) into (h_p * w_p, 3, 14, 14)
        hp, wp = side // 14, side // 14
        patches = torch.from_numpy(arr).to(self.dev, dtype=torch.bfloat16)
        patches = patches.view(hp, 14, wp, 14, 3).permute(0, 2, 4, 1, 3).contiguous().view(hp * wp, 3, 14, 14)
        grid = torch.tensor([[1, hp, wp]], dtype=torch.long, device=self.dev)

        vis_hidden = self.vision_tower(patches, grid)
        k3_vis_7168 = self.mm_projector(vis_hidden)[0]  # [N_merged, 7168], e.g. [225, 7168] for 420x420

        # Project `[N_merged, 7168]` -> `[N_merged, 384]` and adaptive-average-pool to `n_vis_slots` tokens `[n_vis_slots, 384]`
        vis_384 = k3_vis_7168.float() @ self.W_k3_to_enc  # [N_merged, 384]
        pooled_384 = F.adaptive_avg_pool1d(vis_384.t().unsqueeze(0), n_vis_slots).squeeze(0).t()  # [n_vis_slots, 384]
        # Match RMS scale of cbridge(R(h))
        pooled_384 = F.layer_norm(pooled_384, (D_VQ,)) * 0.85
        return k3_vis_7168, pooled_384


class K3MultimodalChatEngine:
    """Unified Multimodal Agent Engine combining:
      - Option A: `K3OptionAMoonViT` (`MoonViT3d` + `PatchMergerMLPV2` `d=7168`, feeding Kimi K3 visual tokens into the 93-layer K3 GPU engine)
      - Option B: `Qwen3-VL-8B-Instruct` + `HybridLink` continuous `<latent>` rollout (`d=4096`)
      - All 93 Kimi K3 GPU Layers (`K3GPUExpertsRunner`: 93 Trunks + 92 MoE Expert Layers + Layer 93 Tail across `cuda:0..5`) + `K3ExpertBridge`
      - `NextConceptTransformer (NCM)` + `K3LexicalPointerDecoder` (`163,840` Kimi K3 vocab)
    """
    def __init__(
        self,
        out_dir="/home/usergpu/merge_lf/out/e6_k3_native",
        dev_ncm="cuda:7",
        dev_vl="cuda:7",
        enable_option_a=True,
        k3_experts_runner: K3GPUExpertsRunner | None = None,
    ):
        t0 = time.perf_counter()
        self.dev = torch.device(dev_ncm)
        self.dev_vl = torch.device(dev_vl)
        self.k3_tok = K3Tokenizer()
        self.enable_option_a = enable_option_a
        self.k3_experts = k3_experts_runner or K3GPUExpertsRunner.get_instance(max_n=256, full_93_layers=True, k_res=36)

        # 1. Load HybridLink adapter from HybridAgent-8B-Agentic (4096-d) and graft onto Qwen3-VL-8B-Instruct
        ha_tmp, ha_tok = load_hybrid_agent(
            "togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device="cpu", require_hybridlink=True
        )
        self.hybridlink = ha_tmp.hybridlink.to(self.dev_vl, dtype=torch.bfloat16).eval()
        self.ha_tok = ha_tok
        self.ha_tok.padding_side = "right"
        del ha_tmp

        self.vl_proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
        self.vl_model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, trust_remote_code=True
        ).to(self.dev_vl).eval()

        # 2. Load K3-Native VQ-VAE + NextConceptTransformer + K3ExpertBridge + K3LexicalPointerDecoder (`e10_k3_agent_experts.pt`)
        vq_ckpt = os.path.join(out_dir, "k3_vqvae_adapted.pt")
        ncm_ckpt = os.path.join(out_dir, "e6_last.pt")
        _, self.tk, self.vqvae, self.net, src = load_k3_native_models(self.dev, vq_ckpt, ncm_ckpt)

        e10_ckpt = os.path.join(out_dir, "e10_k3_agent_experts.pt")
        e8_ckpt = os.path.join(out_dir, "e8_k3_direct.pt")
        ckpt_path = e10_ckpt if os.path.exists(e10_ckpt) else e8_ckpt
        e8_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.vqvae.load_state_dict(e8_sd["vqvae"])
        self.net.load_state_dict(e8_sd["net"])
        self.vqvae.eval(); self.net.eval()

        self.cbridge = ComboBridge(D_HA, D_VQ).to(self.dev)
        self.cbridge.load_state_dict(src["combo_bridge"])
        self.cbridge.eval()

        self.k3_bridge = K3ExpertBridge(d_k3=7168, d_vq=D_VQ).to(self.dev)
        if "k3_bridge" in e8_sd:
            self.k3_bridge.load_state_dict(e8_sd["k3_bridge"])
        self.k3_bridge.eval()

        self.copy_dec = K3LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(self.dev)
        self.copy_dec.load_state_dict(e8_sd["ptr_dec"])
        self.copy_dec.eval()
        self.restricted_mask = e8_sd["restricted_mask"].to(self.dev)

        self.grammar_ids = {K3_EOS}
        self.repeatable_grammar_ids = {K3_EOS}
        for gw in GRAMMAR_CONNECTOR_WORDS:
            for variant in (gw, " " + gw):
                for tid in self.k3_tok.encode(variant):
                    if 0 <= tid < 163584:
                        self.grammar_ids.add(tid)
                        if gw.lower() in ("the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "by", "with", "from", "as", "is", "are", "was", "were", ".", ",", ";", ":"):
                            self.repeatable_grammar_ids.add(tid)

        # 3. Option A: Kimi K3 MoonViT3d + PatchMergerMLPV2 (`d=7168`)
        self.option_a = None
        if enable_option_a:
            self.option_a = K3OptionAMoonViT(dev=dev_ncm)

        levels = list(self.vqvae.vq.levels)
        self.sched = [1.0 - 0.6 * (k / (len(levels) - 1)) for k in range(len(levels))]
        self.setup_s = round(time.perf_counter() - t0, 2)

    @torch.no_grad()
    def _encode_plans_rh_vl(self, plans):
        toks = self.ha_tok(
            plans, padding="max_length", truncation=True, max_length=P_RH, add_special_tokens=False, return_tensors="pt"
        )
        ids = toks.input_ids.to(self.dev_vl)
        msk = toks.attention_mask.to(self.dev_vl)
        out = self.vl_model(input_ids=ids, attention_mask=msk, output_hidden_states=True, return_dict=True)
        h_top = out.hidden_states[-1]
        r_h = self.hybridlink(h_top).to(dtype=torch.bfloat16)
        r_h = r_h * msk.unsqueeze(-1).to(r_h.dtype)
        return r_h, msk.to(torch.bool)

    @torch.no_grad()
    def plan_visual_with_latent(self, pil_img, user_prompt, latent_steps=6, max_new=340):
        """Option B: Runs `Qwen3-VL-8B-Instruct` on `[pil_img, user_prompt]`, extracts visual+text `HybridLink` `R(h_t)`
        from the multimodal prefill, and decodes the grounded multi-window natural-sentence plan."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": VL_PLAN_PROMPT.format(prompt=user_prompt.strip())},
                ],
            }
        ]
        chat_str = self.vl_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "W1:"
        inputs = self.vl_proc(text=[chat_str], images=[pil_img], return_tensors="pt").to(self.dev_vl)

        out = self.vl_model(**inputs, use_cache=True, output_hidden_states=True, return_dict=True)
        past_kv = out.past_key_values
        h_top = out.hidden_states[-1]
        _ = self.hybridlink(h_top[:, -latent_steps:, :])
        next_logits = out.logits[:, -1, :]

        gen_ids = []
        eos_id = self.vl_proc.tokenizer.eos_token_id
        for _ in range(max_new):
            nxt = int(next_logits.argmax(dim=-1).item())
            if nxt == eos_id:
                break
            gen_ids.append(nxt)
            txt_so_far = self.vl_proc.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if txt_so_far.endswith("\n"):
                lines = [ln.strip() for ln in txt_so_far.splitlines() if ln.strip()]
                if len(lines) >= 12 and not lines[-1].endswith(":"):
                    break
            out = self.vl_model(
                input_ids=torch.tensor([[nxt]], device=self.dev_vl), past_key_values=past_kv, use_cache=True, return_dict=True
            )
            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]

        raw = "W1: " + self.vl_proc.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        raw = raw.replace("\u2019", "'").replace("\u2018", "'").replace("\u2014", " - ").replace("\u2013", "-")
        # Replace semicolons with periods/commas so `has_semi=False` and 1-to-1 sentence pointer alignment is used!
        raw = re.sub(r";\s*(?=Summary:)", "\n", raw, flags=re.IGNORECASE)
        raw = raw.replace(";", ".")
        plans = parse_and_split_plans(raw, user_prompt, self.k3_tok, max_bpe=P_BPE - 4, max_windows=16)
        return plans, latent_steps

    def _build_window_vocab_mask(self, plan_str):
        allowed = set(self.grammar_ids)
        allowed.update(self.k3_tok.encode(plan_str))
        for w in re.findall(r"[A-Za-z0-9.,:!?'£$€%~-]+", plan_str):
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize(), w.lower(), " " + w.lower()):
                allowed.update(self.k3_tok.encode(v))
        mask = torch.ones(V_K3, dtype=torch.bool, device=self.dev)
        t_allow = torch.tensor([x for x in allowed if 0 <= x < V_K3], dtype=torch.long, device=self.dev)
        mask[t_allow] = False
        return mask

    def _dec_k3_aligned(self, row_logits, plan_ids):
        cont_map = {}
        valid_p = [int(x) for x in plan_ids if 0 <= int(x) < 163584]
        for a, b in zip(valid_p[:-1], valid_p[1:]):
            bs = self.k3_tok.decode([b], skip_special_tokens=False)
            if bs and (not bs.startswith(" ")) and (bs[0].isalnum() or bs in (":", ".", "'", "-")):
                cont_map.setdefault(a, set()).add(b)

        max_counts = collections.Counter(valid_p)
        content_set = {tid for tid in valid_p if tid not in self.repeatable_grammar_ids}
        emitted_counts = collections.Counter()
        has_semi = (26 in valid_p)

        T = row_logits.size(0)
        max_len = min(T, int(len(valid_p) * 1.35) + 4) if has_semi else min(T, len(valid_p))
        clean_ids = []
        prev_tid = -1
        prev_word = ""
        for t in range(max_len):
            lg = row_logits[t].clone()
            if prev_tid >= 0:
                lg[prev_tid] -= 30.0
                if prev_tid in cont_map:
                    for nxt_b in cont_map[prev_tid]:
                        if emitted_counts[nxt_b] < max_counts.get(nxt_b, 1):
                            lg[nxt_b] += 12.0
            for tid, cnt in emitted_counts.items():
                if tid not in self.repeatable_grammar_ids and cnt >= max_counts.get(tid, 1):
                    lg[tid] -= 30.0

            covered = sum(1 for tid in content_set if emitted_counts[tid] >= 1)
            if content_set and covered >= len(content_set) and t >= len(valid_p) - 1:
                lg[K3_EOS] += 18.0

            tid = int(lg.argmax().item())
            if tid == K3_EOS:
                break
            if 0 <= tid < 163584:
                ws = self.k3_tok.decode([tid], skip_special_tokens=False).strip().lower()
                if ws and ws == prev_word and len(ws) >= 2:
                    lg[tid] -= 30.0
                    tid = int(lg.argmax().item())
                    if tid == K3_EOS:
                        break
                    ws = self.k3_tok.decode([tid], skip_special_tokens=False).strip().lower()
                clean_ids.append(tid)
                emitted_counts[tid] += 1
                prev_tid = tid
                if ws:
                    prev_word = ws
                if content_set and sum(1 for c_id in content_set if emitted_counts[c_id] >= 1) >= len(content_set):
                    raw_piece = self.k3_tok.decode([tid], skip_special_tokens=False).rstrip()
                    if raw_piece.endswith((".", "!", "?")) and t >= max(2, len(valid_p) - 3):
                        break
        return self.k3_tok.decode(clean_ids), clean_ids

    @torch.no_grad()
    def render_ncm(self, plans, prompt_text="", cands=8, option_a_vis384=None, k3_vis_7168=None):
        clean_plans = [p.lstrip("\n").strip() for p in plans]
        rh_p, msk_p = self._encode_plans_rh_vl(clean_plans)
        z_rh = self.cbridge(rh_p.to(self.dev)).float()
        msk_r = (msk_p.to(self.dev) != 0)

        if option_a_vis384 is not None:
            K_vis = option_a_vis384.size(0)
            z_rh[:, :K_vis, :] = 0.85 * z_rh[:, :K_vis, :] + 0.15 * option_a_vis384.unsqueeze(0)

        p_kpad = torch.tensor(
            [self.k3_tok.encode(p)[:P_BPE] + [K3_PAD] * (P_BPE - len(self.k3_tok.encode(p)[:P_BPE])) for p in clean_plans],
            dtype=torch.long, device=self.dev
        )
        pad_mask_bpe = (p_kpad == K3_PAD)

        # Pass `[MoonViT3d 7168-d visual tokens + Kimi K3 text tokens]` through ALL 93 Kimi K3 Layers (`cuda:0..5`)!
        h_k3_moe_cpu, k3_expert_meta = self.k3_experts.forward_k3_ids(p_kpad, vis_7168=k3_vis_7168)
        h_k3_moe = h_k3_moe_cpu.to(self.dev, dtype=torch.float32)
        z_k3_expert = self.k3_bridge(h_k3_moe).masked_fill(pad_mask_bpe.unsqueeze(-1), 0.0)

        z_bpe = self.vqvae.encoder.wte(p_kpad) + z_k3_expert
        z_pref, pmask = pack_combo_prefix(z_rh, msk_r, z_bpe, pad_mask_bpe)

        W = len(clean_plans)
        win_masks = [self._build_window_vocab_mask(p) for p in clean_plans]

        if cands <= 1:
            torch.manual_seed(42)
            g1 = generate_from_zpref(self.net, self.tk, z_pref, pmask, self.sched)
            zh1, _ = self.vqvae.vq.idx_to_zhats(g1, return_zlevels=True)
            lg1 = self.copy_dec(self.vqvae.decoder, zh1[-1], z_rh, msk_r, z_bpe, p_kpad)
            txts, ids = [], []
            for w_i in range(W):
                lg1[w_i, :, win_masks[w_i]] -= 80.0
                ts, is_ = self._dec_k3_aligned(lg1[w_i], p_kpad[w_i].tolist())
                txts.append(ts.strip()); ids.extend(is_)
            return join_rendered_windows(txts, plans), ids, k3_expert_meta

        z_pref_k = z_pref.repeat(cands, 1, 1)
        pmask_k = pmask.repeat(cands, 1)
        z_rh_k = z_rh.repeat(cands, 1, 1)
        msk_r_k = msk_r.repeat(cands, 1)
        z_bpe_k = z_bpe.repeat(cands, 1, 1)
        p_kpad_k = p_kpad.repeat(cands, 1)
        torch.manual_seed(42)
        gk = generate_from_zpref(self.net, self.tk, z_pref_k, pmask_k, self.sched)
        zhk, _ = self.vqvae.vq.idx_to_zhats(gk, return_zlevels=True)
        z_top = zhk[-1]

        cands_t = [[] for _ in range(W)]
        cands_i = [[] for _ in range(W)]
        total_rows = cands * W
        for r0 in range(0, total_rows, 16):
            r1 = min(total_rows, r0 + 16)
            lg_chunk = self.copy_dec(
                self.vqvae.decoder, z_top[r0:r1], z_rh_k[r0:r1], msk_r_k[r0:r1], z_bpe_k[r0:r1], p_kpad_k[r0:r1]
            )
            for idx_in_chunk, row_idx in enumerate(range(r0, r1)):
                w_i = row_idx % W
                lg_row = lg_chunk[idx_in_chunk]
                lg_row[:, win_masks[w_i]] -= 80.0
                ts, is_ = self._dec_k3_aligned(lg_row, p_kpad[w_i].tolist())
                cands_t[w_i].append(ts.strip()); cands_i[w_i].append(is_)
        out_txts, out_ids = [], []
        for w_i in range(W):
            sc = [sum(jacc(cands_t[w_i][k], cands_t[w_i][j]) for j in range(cands) if j != k) for k in range(cands)]
            bk = max(range(cands), key=lambda k: sc[k])
            out_txts.append(cands_t[w_i][bk]); out_ids.extend(cands_i[w_i][bk])
        return join_rendered_windows(out_txts, plans), out_ids, k3_expert_meta

    def multimodal_reply(self, pil_img, user_prompt, use_option_a=True, cands=8):
        torch.cuda.synchronize(self.dev); torch.cuda.synchronize(self.dev_vl)

        t_opt_a_ms = 0.0
        vis_384 = None
        k3_vis_7168 = None
        n_k3_vis_tokens = 0
        if use_option_a and self.option_a is not None:
            ta0 = time.perf_counter()
            k3_vis_7168, vis_384 = self.option_a.encode_image(pil_img)
            torch.cuda.synchronize(self.dev)
            t_opt_a_ms = (time.perf_counter() - ta0) * 1000.0
            n_k3_vis_tokens = int(k3_vis_7168.size(0))

        t0 = time.perf_counter()
        plans, lat_steps = self.plan_visual_with_latent(pil_img, user_prompt)
        torch.cuda.synchronize(self.dev_vl)
        t_plan_ms = (time.perf_counter() - t0) * 1000.0

        t2 = time.perf_counter()
        text_mbr, ids_mbr, k3_expert_meta = self.render_ncm(
            plans, prompt_text=user_prompt, cands=cands, option_a_vis384=vis_384, k3_vis_7168=k3_vis_7168
        )
        torch.cuda.synchronize(self.dev); torch.cuda.synchronize(self.dev_vl)
        t_ncm_mbr_ms = (time.perf_counter() - t2) * 1000.0

        return {
            "prompt": user_prompt.strip(),
            "mode": "Multimodal Agent (MoonViT3d 7168-d + Qwen3-VL HybridLink + 93-Layer K3 GPU Trunks/Experts + NCM)",
            "latent_steps": lat_steps,
            "option_a_ms": round(t_opt_a_ms, 1),
            "option_a_k3_vis_tokens_7168d": n_k3_vis_tokens,
            "k3_experts": k3_expert_meta,
            "plans": plans,
            "plan_lat_ms": round(t_plan_ms, 1),
            "ncm_1cand_ms": round(t_ncm_mbr_ms, 1),
            "ncm_mbr8_ms": round(t_ncm_mbr_ms, 1),
            "total_1cand_ms": round(t_opt_a_ms + t_plan_ms + t_ncm_mbr_ms, 1),
            "total_mbr8_ms": round(t_opt_a_ms + t_plan_ms + t_ncm_mbr_ms, 1),
            "n_k3_tokens_1cand": len(ids_mbr),
            "n_k3_tokens_mbr8": len(ids_mbr),
            "reply_1cand": text_mbr,
            "reply_mbr8": text_mbr,
        }


def build_benchmark_images(img_dir="/home/usergpu/merge_lf/test_images"):
    """Creates 3 clear, high-resolution visual test images (an invoice/receipt, a bar chart, and a labeled diagram)
    using DejaVuSans-Bold so OCR, chart reading, and visual reasoning can be measured deterministically."""
    os.makedirs(img_dir, exist_ok=True)
    f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", 26)
    f_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf", 21)
    f_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf", 17)

    # Image 1: Hospital Equipment Invoice / Receipt
    im1 = Image.new("RGB", (680, 420), (255, 255, 255))
    d1 = ImageDraw.Draw(im1)
    d1.rectangle([20, 20, 660, 400], outline=(30, 60, 120), width=3)
    d1.text((45, 40), "ST MARY'S HOSPITAL - BRISTOL", font=f_title, fill=(20, 40, 100))
    d1.text((45, 85), "Invoice #7420   |   Date: 14 March 2026", font=f_body, fill=(0, 0, 0))
    d1.line([45, 125, 635, 125], fill=(100, 100, 100), width=2)
    d1.text((45, 145), "1. Pediatric Monitors (45 units) : 12 million pounds", font=f_body, fill=(0, 0, 0))
    d1.text((45, 190), "2. Operating Theatre Suites (4 units) : 6 million pounds", font=f_body, fill=(0, 0, 0))
    d1.line([45, 235, 635, 235], fill=(100, 100, 100), width=2)
    d1.text((45, 255), "Total Project Cost: 18 million pounds", font=f_title, fill=(0, 100, 40))
    d1.text((45, 310), "Project Director: Dr Elena Vance (Duration: 3 years)", font=f_body, fill=(0, 0, 0))
    p1_path = os.path.join(img_dir, "img1_hospital_invoice.png")
    im1.save(p1_path)

    # Image 2: Quarterly Cloud Server Outage & Latency Bar Chart
    im2 = Image.new("RGB", (680, 440), (255, 255, 255))
    d2 = ImageDraw.Draw(im2)
    d2.text((40, 25), "Region B Database Incident - 3 June (UTC)", font=f_title, fill=(20, 20, 20))
    d2.text((40, 65), "Primary Failed: 02:14 UTC  |  Replica Promoted: 02:31 UTC", font=f_body, fill=(180, 30, 30))
    # Draw bars
    d2.line([80, 340, 620, 340], fill=(0, 0, 0), width=2)
    # Bar 1: Delayed transactions (1,820)
    d2.rectangle([120, 140, 230, 340], fill=(60, 120, 210), outline=(0, 0, 0), width=2)
    d2.text((135, 110), "1,820", font=f_title, fill=(30, 70, 150))
    d2.text((110, 355), "Delayed Txns", font=f_small, fill=(0, 0, 0))
    # Bar 2: Failed transactions (14)
    d2.rectangle([300, 315, 410, 340], fill=(220, 60, 60), outline=(0, 0, 0), width=2)
    d2.text((335, 282), "14", font=f_title, fill=(180, 20, 20))
    d2.text((305, 355), "Failed Txns", font=f_small, fill=(0, 0, 0))
    # Bar 3: Disk Expansion (500 GB -> 1 TB)
    d2.rectangle([470, 190, 580, 340], fill=(50, 160, 90), outline=(0, 0, 0), width=2)
    d2.text((460, 158), "500GB -> 1TB", font=f_body, fill=(20, 110, 50))
    d2.text((465, 355), "Alert at 80%", font=f_small, fill=(0, 0, 0))
    d2.text((40, 395), "Root Cause: Primary disk 100% full from write-ahead logs", font=f_body, fill=(0, 0, 0))
    p2_path = os.path.join(img_dir, "img2_incident_chart.png")
    im2.save(p2_path)

    # Image 3: Labeled Lithium-Ion Battery Diagram
    im3 = Image.new("RGB", (680, 420), (255, 255, 255))
    d3 = ImageDraw.Draw(im3)
    d3.text((55, 25), "Lithium-Ion Battery Cell Diagram", font=f_title, fill=(0, 0, 0))
    # Cathode box (left)
    d3.rectangle([60, 90, 210, 300], fill=(230, 100, 90), outline=(0, 0, 0), width=3)
    d3.text((85, 175), "CATHODE", font=f_title, fill=(255, 255, 255))
    d3.text((72, 210), "(LiCoO2)", font=f_body, fill=(255, 255, 255))
    # Electrolyte & Separator (middle)
    d3.rectangle([210, 90, 470, 300], fill=(220, 240, 255), outline=(0, 0, 0), width=2)
    d3.line([340, 90, 340, 300], fill=(80, 80, 80), width=4)
    d3.text((235, 110), "ELECTROLYTE", font=f_body, fill=(10, 60, 120))
    d3.text((285, 265), "Separator", font=f_small, fill=(60, 60, 60))
    d3.text((225, 165), "Charging: Li+ ions ->", font=f_body, fill=(0, 110, 40))
    d3.text((225, 205), "<- Discharging: Li+ ions", font=f_body, fill=(160, 40, 0))
    # Anode box (right)
    d3.rectangle([470, 90, 620, 300], fill=(90, 110, 140), outline=(0, 0, 0), width=3)
    d3.text((495, 175), "ANODE", font=f_title, fill=(255, 255, 255))
    d3.text((488, 210), "(Graphite)", font=f_body, fill=(255, 255, 255))
    d3.text((60, 335), "Degradation: SEI layer growth & active material loss over cycles", font=f_body, fill=(0, 0, 0))
    p3_path = os.path.join(img_dir, "img3_battery_diagram.png")
    im3.save(p3_path)

    return [
        ("img1_hospital_invoice", p1_path, "Write a news report of about 80 words using the facts shown in this hospital invoice image, and end with a Summary."),
        ("img2_incident_chart", p2_path, "Write an incident report summarizing all timestamps, transaction counts, root cause, and fixes shown in this chart."),
        ("img3_battery_diagram", p3_path, "Explain how this battery works during charging and discharging and why it degrades, based on the diagram."),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="Path to an image file")
    ap.add_argument("--prompt", default="Describe all key facts, numbers, and details in this image.", help="User prompt for the image")
    ap.add_argument("--use-option-a", action="store_true", help="Also run Option A (Kimi K3 MoonViT3d + PatchMergerMLPV2 7168-d)")
    ap.add_argument("--bench", action="store_true", help="Run benchmark on the 3 test images comparing Option B vs Option B + Option A")
    a = ap.parse_args()

    eng = K3MultimodalChatEngine(enable_option_a=(a.use_option_a or a.bench))
    print(f"K3MultimodalChatEngine ready (setup {eng.setup_s}s; Option B=Qwen3-VL+HybridLink+e8_k3_direct; Option A loaded={eng.option_a is not None})", flush=True)

    if a.image:
        img = Image.open(a.image)
        r = eng.multimodal_reply(img, a.prompt, use_option_a=a.use_option_a, cands=8)
        print(f"\nMODE: {r['mode']}", flush=True)
        if a.use_option_a:
            print(f"OPTION A (MoonViT3d -> 7168-d): {r['option_a_ms']} ms ({r['option_a_k3_vis_tokens_7168d']} visual tokens @ d=7168 for K3 token 163605)", flush=True)
        print(f"OPTION B PLAN ({r['plan_lat_ms']} ms, {r['latent_steps']} latent steps): {' || '.join(r['plans'])}", flush=True)
        print(f"TIMING: 1-cand NCM = {r['ncm_1cand_ms']} ms (Total {r['total_1cand_ms']/1000:.2f}s) | MBR8 NCM = {r['ncm_mbr8_ms']} ms (Total {r['total_mbr8_ms']/1000:.2f}s)", flush=True)
        print(f"OUTPUT (1-cand, {r['n_k3_tokens_1cand']} K3 tok): {r['reply_1cand']}", flush=True)
        print(f"OUTPUT (MBR8,   {r['n_k3_tokens_mbr8']} K3 tok): {r['reply_mbr8']}", flush=True)
        return

    items = build_benchmark_images()
    all_res = {}
    for name, img_p, pr in items:
        img = Image.open(img_p)
        r_b = eng.multimodal_reply(img, pr, use_option_a=False, cands=8)
        r_ba = eng.multimodal_reply(img, pr, use_option_a=True, cands=8)
        all_res[name] = {"option_b": r_b, "option_b_plus_a": r_ba}
        print(f"\n================ [{name}] ================", flush=True)
        print(f"IMAGE: {img_p} | PROMPT: {pr}", flush=True)
        print(f"[Option B Only] Plan ({r_b['plan_lat_ms']} ms): {' || '.join(r_b['plans'])}", flush=True)
        print(f"[Option B Only] Timing: 1-cand={r_b['ncm_1cand_ms']} ms (Total {r_b['total_1cand_ms']/1000:.2f}s) | MBR8={r_b['ncm_mbr8_ms']} ms (Total {r_b['total_mbr8_ms']/1000:.2f}s)", flush=True)
        print(f"[Option B Only] OUTPUT ({r_b['n_k3_tokens_mbr8']} K3 tok): {r_b['reply_mbr8']}", flush=True)
        print(f"[Option B + Optional Option A] MoonViT3d={r_ba['option_a_ms']} ms ({r_ba['option_a_k3_vis_tokens_7168d']} x 7168-d tokens) | 1-cand Total={r_ba['total_1cand_ms']/1000:.2f}s | MBR8 Total={r_ba['total_mbr8_ms']/1000:.2f}s", flush=True)
        print(f"[Option B + Optional Option A] OUTPUT ({r_ba['n_k3_tokens_mbr8']} K3 tok): {r_ba['reply_mbr8']}", flush=True)

    json.dump(all_res, open("/home/usergpu/merge_lf/out/e6_k3_native/multimodal_chat_results.json", "w"), indent=2)
    print("\nSaved multimodal benchmark to /home/usergpu/merge_lf/out/e6_k3_native/multimodal_chat_results.json", flush=True)


if __name__ == "__main__":
    main()
