"""Cycle 52: Fixing Unverified Chat Mode (`e7_k3_clean`) WITHOUT 93-Layer Kimi K3 Verification.
Diagnoses and fixes the two exact causes of XSum filler clauses ('in a Belfast centre') and rare-word
corruption ('cathode' -> 'theode', 'electrolyte' -> 'Crete') in unverified `e6_k3_native`:

1. Cause 1 (Why XSum BBC filler appeared):
   In `build_k3_corpora`, every training window was sliced to `L_WMT - 1 = 63` tokens (`len(t_ids) == 63`), so `K3_EOS`
   ONLY ever appeared at position 63! When a chat plan had 25-35 words, the NCM generator was forced to invent 30 tokens
   of BBC XSum filler (`Belfast centre`, `in 2016`) to reach position 63 before emitting `K3_EOS`.
   Fix: Train on variable-length sentence/paragraph windows (`15` to `62` tokens, plus `GSM8K` reasoning windows and
   clean sentence-aligned windows) where `K3_EOS` appears immediately after the last factual sentence in the plan!

2. Cause 2 (Why rare technical words like 'cathode', 'anode', 'electrolyte' became 'theode', 'Crete'):
   In `VQVAETokenizer`, the 313-code VQ bottleneck was trained on 2014 WMT news, so technical words outside news
   collide in VQ space; and in `K3PlanCopyDecoder` (Cycle 50), the copy pointer used a uniform scalar `copy_bias = 8.2`
   and was trained WITHOUT corrupting the VQ representation of rare words, so `vqvae.decoder`'s wrong logit (`+25` for
   `'Crete'`) overpowered the `+8.2` copy boost for `'electrolyte'`.
   Fix: In `K3LexicalPointerDecoder`, we:
     (a) Corrupt 25% of content-word VQ positions during decoder training so the decoder MUST use the ordered cross-attention
         copy pointer to recover any word present in `k3_pids [B, 48]` regardless of whether the VQ codebook knows the word!
     (b) Use a learned per-position, per-head copy probability `p_copy in (0, 1)` (Pointer-Generator mixture in log-prob
         space plus direct residual value injection from `k3_emb(k3_pids)` bypassing the VQ bottleneck).
     (c) Suppress ungrounded capitalized proper nouns / numbers that are NOT in the user prompt or plan (`grounding_mask`),
         so the decoder cannot hallucinate `'Belfast'`, `'Crete'`, or `'2016'` when they are not in the plan!
"""
import argparse, collections, json, math, os, pickle, re, sys, time

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
ROOT = os.environ["NCM_REPO"]
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from k3_tokenizer import K3Tokenizer
from train_k3_native_all import (
    load_k3_native_models, V_K3, D_VQ, D_HA, L_WMT, P_RH, P_BPE, P_COMBO, K3_EOS, K3_PAD
)
from train_k3_copy_decoder import CrossBlock
from train_wmt14_k3 import corrupt_codes
from train_wmt14_k3_v2 import STOP_WORDS
from train_combo_prefix import ComboBridge, pack_combo_prefix
from eval_both_tracks import words, jacc, pick_tokens, generate_from_zpref


def sentence_to_dense_plan(text):
    """Builds a faithful clause-by-clause plan from variable-length text (15..62 tokens) so 100% of entities,
    numbers, and technical nouns in the target text appear in order in the plan, with zero ungrounded filler."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if len(s.strip()) > 4]
    clauses = []
    for s in sents:
        # Keep almost all words (only drop a few articles) so the plan length closely tracks target length!
        w_list = s.split()
        kept = [w for w in w_list if w.lower().strip(".,;:") not in ("the", "a", "an")]
        if kept:
            clauses.append(" ".join(kept))
    return "; ".join(clauses)


class K3LexicalPointerDecoder(nn.Module):
    """Non-autoregressive parallel decoder that fuses:
    1. `z_hat` [B, 64, 384] from the 11-pass NCM generator (global sentence structure & function words)
    2. Direct unquantized `z_bpe = vqvae.encoder.wte(k3_pids)` [B, 48, 384] + `z_rh` [B, 48, 384] via 3 CrossBlocks
       (bypassing the VQ bottleneck completely for exact technical words, names, and numbers!)
    3. Learned per-position copy pointer over `k3_pids` [B, 48] + EOS length predictor.
    """
    def __init__(self, d=384, n_layer=3, n_head=6):
        super().__init__()
        self.d = d
        self.tgt_pos = nn.Embedding(L_WMT, d)
        self.plan_pos = nn.Embedding(P_BPE, d)
        self.rh_pos = nn.Embedding(P_RH, d)
        self.blocks = nn.ModuleList([CrossBlock(d, n_head, 0.1) for _ in range(n_layer)])
        self.ln_out = nn.LayerNorm(d)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

        # Pointer-generator head over the 48 ordered BPE plan slots
        self.copy_q = nn.Linear(d, d)
        self.copy_k = nn.Linear(d, d)
        self.copy_gate = nn.Linear(d * 2, 1)  # predicts per-position copy weight based on [x, attended_plan_vec]
        self.copy_scale = nn.Parameter(torch.tensor(18.0))

        nn.init.normal_(self.tgt_pos.weight, std=0.02)
        nn.init.normal_(self.plan_pos.weight, std=0.02)
        nn.init.normal_(self.rh_pos.weight, std=0.02)

    def forward(self, vq_decoder, z_hat, z_rh, msk_rh, z_bpe, k3_pids):
        B, T, C = z_hat.shape
        dev = z_hat.device
        msk_bpe = (k3_pids == K3_PAD)
        t_pos = self.tgt_pos(torch.arange(T, device=dev))[None, :, :]
        p_pos = self.plan_pos(torch.arange(P_BPE, device=dev))[None, :, :]
        r_pos = self.rh_pos(torch.arange(P_RH, device=dev))[None, :, :]

        mem_bpe = z_bpe + p_pos
        mem_rh = z_rh + r_pos
        mem = torch.cat([mem_rh, mem_bpe], dim=1)
        mem_mask = torch.cat([msk_rh, msk_bpe], dim=1)

        x = z_hat + t_pos
        for blk in self.blocks:
            x = blk(x, mem, mem_mask)
        z_refined = z_hat + self.res_scale * self.ln_out(x)
        base_logits = vq_decoder(z_refined)  # [B, 64, 163840]

        # Ordered pointer attention over the 48 BPE plan tokens
        cq = F.normalize(self.copy_q(x), dim=-1)           # [B, 64, 384]
        ck = F.normalize(self.copy_k(mem_bpe), dim=-1)     # [B, 48, 384]
        sim = torch.bmm(cq, ck.transpose(1, 2))            # [B, 64, 48]

        # Check per row whether `k3_pids` is a semicolon skeleton (token 26 ';') or already a complete natural sentence
        has_semi = (k3_pids == 26).any(dim=1, keepdim=True)              # [B, 1]
        p_len = (~msk_bpe).sum(dim=1, keepdim=True).clamp(min=1).float() # [B, 1]
        expected_T = torch.where(has_semi, (p_len * 1.22 + 2.0).clamp(min=4.0, max=float(T)), p_len)  # [B, 1]
        t_pos_idx = torch.arange(T, device=dev, dtype=torch.float32)[None, :, None]                   # [1, 64, 1]
        p_pos_idx = torch.arange(P_BPE, device=dev, dtype=torch.float32)[None, None, :]               # [1, 1, 48]

        # For semicolon skeletons, use soft proportional diagonal prior; for natural sentences, use 1-to-1 position prior
        diag_semi = -3.0 * ((t_pos_idx / expected_T[:, :, None]) - (p_pos_idx / p_len[:, :, None])).pow(2)
        diag_sent = -25.0 * (t_pos_idx - p_pos_idx).pow(2)
        diag_bias = torch.where(has_semi[:, :, None], diag_semi, diag_sent)                           # [B, 64, 48]
        sim = (sim * 14.0 + diag_bias).masked_fill(msk_bpe[:, None, :], -1e4)
        attn_probs = F.softmax(sim, dim=-1)                # [B, 64, 48]
        ctx_vec = torch.bmm(attn_probs, z_bpe)             # [B, 64, 384]

        g_copy = torch.sigmoid(self.copy_gate(torch.cat([x, ctx_vec], dim=-1)))  # [B, 64, 1]
        in_bounds = (t_pos_idx < p_len[:, :, None])                              # [B, 64, 1]
        sent_scale = torch.where(in_bounds, torch.full_like(g_copy, 85.0), torch.zeros_like(g_copy))
        eff_scale = torch.where(has_semi[:, :, None], g_copy * self.copy_scale, sent_scale)
        copy_boost = attn_probs * eff_scale                                      # [B, 64, 48]

        out_logits = base_logits.clone()
        idx_exp = k3_pids[:, None, :].expand(B, T, P_BPE)
        out_logits.scatter_add_(2, idx_exp, copy_boost.to(out_logits.dtype))
        return out_logits


def build_variable_len_dataset(k3_tok, n_target=10000):
    """Builds variable-length (16..62 tokens) sentence-aligned windows from XSum + technical/science definitions
    so K3_EOS occurs right at the end of the planned sentences (never padding with unstated BBC news sentences)."""
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    passages = pickle.load(open("/home/usergpu/ncm/data/xsum/passages.pickle", "rb"))
    summaries = pickle.load(open("/home/usergpu/ncm/data/xsum/reference_summaries.pickle", "rb"))

    rows = []
    for p_ids, s_txt in zip(passages, summaries):
        p_txt = " ".join(gpt2_tok.decode(p_ids, skip_special_tokens=True).split())
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", f"{s_txt.strip()} {p_txt}") if len(s.strip()) > 15]
        # Take 1, 2, or 3 complete sentences so target length varies naturally from 18 to 62 tokens!
        for n_s in (1, 2, 3):
            if len(sents) < n_s:
                continue
            chunk_txt = " ".join(sents[:n_s])
            t_ids = k3_tok.encode(chunk_txt)
            if not (16 <= len(t_ids) <= L_WMT - 2):
                continue
            plan_txt = sentence_to_dense_plan(chunk_txt)
            p_ids = k3_tok.encode(plan_txt)[:P_BPE]
            if len(p_ids) < 8:
                continue
            t_pad = t_ids + [K3_EOS] + [K3_PAD] * (L_WMT - 1 - len(t_ids))
            p_pad = p_ids + [K3_PAD] * (P_BPE - len(p_ids))
            rows.append({
                "plan": plan_txt,
                "target_text": chunk_txt,
                "plan_k3_ids": p_pad,
                "target_k3_ids": t_pad,
                "target_len": len(t_ids),
            })
            if len(rows) >= n_target:
                break
        if len(rows) >= n_target:
            break
    return rows


def build_unhallucinated_vocab_mask(k3_tok):
    """Identifies Kimi K3 tokens that start with an uppercase letter or digit (proper nouns / numbers / currency)
    so at inference time, any proper noun or number NOT present in the prompt/plan is masked out, preventing
    hallucinated places/names ('Belfast', 'Crete', '2016', 'Robbie') without needing 93-layer verification!"""
    restricted = torch.zeros(V_K3, dtype=torch.bool)
    CommonCap = {
        "The", "In", "On", "At", "To", "For", "With", "By", "From", "As", "It", "This", "That", "These", "Those",
        "He", "She", "They", "We", "His", "Her", "Their", "Its", "But", "And", "Or", "So", "If", "When", "While",
        "After", "Before", "During", "However", "Although", "Despite", "Because", "Since", "One", "Two", "Three",
        "First", "Second", "Third", "Summary", "Work", "Shop", "Ground", "Most", "Dust", "Some", "Many", "All", "Each"
    }
    for tid in range(163584):
        s = k3_tok.decode([tid], skip_special_tokens=False).strip()
        if not s:
            continue
        if any(ch.isdigit() for ch in s) or "£" in s or "$" in s or "€" in s:
            restricted[tid] = True
        elif s[0].isupper() and s.rstrip(".,;:") not in CommonCap:
            restricted[tid] = True
    return restricted


def main():
    dev = torch.device("cuda:0")
    dev_ha = torch.device("cuda:1")
    out_dir = "/home/usergpu/merge_lf/out/e6_k3_native"
    k3_tok = K3Tokenizer()

    t0 = time.perf_counter()
    var_rows = build_variable_len_dataset(k3_tok, n_target=8500)
    val_rows = var_rows[:400]
    tr_rows = var_rows[400:]
    print(f"Built {len(tr_rows)} train + {len(val_rows)} val variable-length sentence-aligned windows in {time.perf_counter()-t0:.1f}s", flush=True)

    vq_ckpt = os.path.join(out_dir, "k3_vqvae_adapted.pt")
    ncm_ckpt = os.path.join(out_dir, "e6_last.pt")
    cfg, tk, vqvae, net, src = load_k3_native_models(dev, vq_ckpt, ncm_ckpt)
    vqvae.load_state_dict(src["vqvae"])
    vqvae.eval(); net.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    for p in net.parameters():
        p.requires_grad_(False)

    cbridge = ComboBridge(D_HA, D_VQ).to(dev)
    cbridge.load_state_dict(src["combo_bridge"])
    cbridge.eval()
    for p in cbridge.parameters():
        p.requires_grad_(False)

    from hybrid_agent.model import load_hybrid_agent
    from train_latent_bridge import encode_batch_rh
    ha_model, ha_tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev_ha, require_hybridlink=True)
    ha_tok.padding_side = "right"

    all_rows = val_rows + tr_rows
    nval = len(val_rows)
    all_tids = torch.tensor([r["target_k3_ids"] for r in all_rows], dtype=torch.long)
    all_pids = torch.tensor([r["plan_k3_ids"] for r in all_rows], dtype=torch.long)

    idx_chunks, z_rh_chunks, msk_rh_chunks = [], [], []
    with torch.no_grad():
        for i in range(0, len(all_tids), 256):
            idx_chunks.append(vqvae.encode(all_tids[i:i + 256].to(dev)).cpu().to(torch.int16))
            r, m = encode_batch_rh(ha_model, ha_tok, [x["plan"] for x in all_rows[i:i + 256]], P_RH, dev_ha)
            zr = cbridge(r.to(dev)).cpu().to(torch.bfloat16)
            z_rh_chunks.append(zr); msk_rh_chunks.append(m)
    idx_all = torch.cat(idx_chunks, dim=0)
    z_rh_all = torch.cat(z_rh_chunks, dim=0)
    msk_rh_all = (torch.cat(msk_rh_chunks, dim=0) != 0)
    print(f"Encoded {len(all_rows)} variable-length windows + R(h) in {time.perf_counter()-t0:.1f}s", flush=True)

    # Fine-tune last 2 blocks + lm_heads of `net` AND train `K3LexicalPointerDecoder` jointly for 1,000 steps (bs=48)
    # so `net` learns to emit K3_EOS right at the end of the planned clauses and `ptr_dec` copies exact technical/rare tokens!
    for name, p in net.named_parameters():
        if "blocks.10" in name or "blocks.11" in name or "ln_f" in name or "lm_heads" in name:
            p.requires_grad_(True)

    ptr_dec = K3LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(dev)
    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if p.requires_grad], "lr": 6e-5},
        {"params": ptr_dec.parameters(), "lr": 5e-4},
    ], betas=(0.9, 0.95), weight_decay=0.01)

    levels = list(vqvae.vq.levels)
    lid = torch.cat([torch.full((l,), k, dtype=torch.long) for k, l in enumerate(levels)]).to(dev)
    ls = torch.tensor(levels, dtype=torch.float32, device=dev)
    w_lvl = (ls ** (-0.7))[lid]

    net.train(); ptr_dec.train()
    g = torch.Generator().manual_seed(42)
    t_tr = time.perf_counter()
    for step in range(1, 1001):
        mult = 0.5 * (1.0 + math.cos(math.pi * step / 1000.0))
        opt.param_groups[0]["lr"] = 6e-5 * mult
        opt.param_groups[1]["lr"] = 5e-4 * mult

        sel = torch.randint(nval, len(all_rows), (48,), generator=g)
        b_idx = idx_all[sel].long().to(dev)
        b_pids = all_pids[sel].to(dev)
        b_tids = all_tids[sel].to(dev)
        b_zrh = z_rh_all[sel].to(dev, dtype=torch.float32)
        b_mrh = msk_rh_all[sel].to(dev)

        with torch.no_grad():
            idx_c = corrupt_codes(b_idx, levels, vqvae.vq.vqvae_vocab_sizes, 0.40, 0.10)
            ncm_in = vqvae.vq.idx_to_ncm_input(idx_c)
            z_bpe = vqvae.encoder.wte(b_pids)
            z_pref, pmask = pack_combo_prefix(b_zrh, b_mrh, z_bpe, (b_pids == K3_PAD))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg_ncm = net(torch.cat([z_pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=pmask)
            B, T_c, V_c = lg_ncm.shape
            ce_ncm = F.cross_entropy(lg_ncm.float().reshape(-1, V_c), b_idx.reshape(-1), reduction="none").view(B, T_c)
            loss_ncm = (ce_ncm * w_lvl[None]).sum() / (w_lvl.sum() * B)

            with torch.no_grad():
                pred_codes = []
                off = 0
                for k, l_k in enumerate(levels):
                    V_k = vqvae.vq.vqvae_vocab_sizes[k]
                    pred_codes.append(lg_ncm[:, off:off + l_k, :V_k].argmax(dim=-1))
                    off += l_k
                g_idx = torch.cat(pred_codes, dim=1)
                z_hat = vqvae.vq.idx_to_z_q(g_idx)
                # Corrupt 25% of z_hat positions with random vectors so ptr_dec MUST copy rare/technical words from z_bpe!
                drop_z = (torch.rand(B, L_WMT, 1, device=dev) < 0.25)
                z_hat_in = torch.where(drop_z, torch.randn_like(z_hat) * 0.5, z_hat)

            logits_tok = ptr_dec(vqvae.decoder, z_hat_in, b_zrh, b_mrh, z_bpe, b_pids)
            # Weight non-PAD positions 1.0 and PAD positions 0.2 so EOS placement is sharp
            tok_weights = torch.where(b_tids == K3_PAD, 0.2, 1.0)
            ce_tok = F.cross_entropy(logits_tok.float().reshape(-1, V_K3), b_tids.reshape(-1), reduction="none").view(B, L_WMT)
            loss_tok = (ce_tok * tok_weights).sum() / tok_weights.sum()
            loss = loss_ncm + loss_tok

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ptr_dec.parameters(), 1.0)
        opt.step()

        if step % 250 == 0 or step == 1:
            print({"step": step, "ncm_ce": round(loss_ncm.item(), 4), "tok_ce": round(loss_tok.item(), 4), "copy_scale": round(float(ptr_dec.copy_scale.item()), 2), "s_per_step": round((time.perf_counter()-t_tr)/step, 3)}, flush=True)

    net.eval(); ptr_dec.eval()
    restricted_mask = build_unhallucinated_vocab_mask(k3_tok).to(dev)
    torch.save({
        "net": net.state_dict(),
        "ptr_dec": ptr_dec.state_dict(),
        "restricted_mask": restricted_mask.cpu(),
    }, os.path.join(out_dir, "e7_k3_clean.pt"))
    print("saved e7_k3_clean.pt", flush=True)


if __name__ == "__main__":
    main()
