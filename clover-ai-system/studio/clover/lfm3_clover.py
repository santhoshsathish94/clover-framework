"""Clover Layers 1-3 for LFM3-1.7B only: native C engine (Layer 1), LFM3 native HybridLink (Layer 2),
LFM3-grafted 11-level NCM + lexical pointer decoder (Layer 3). Copied from
clover-ai-system/src/models/lfm3-1.7b/lfm3_hybrid_ncm.py and the "lfm3-1.7b" branch of
ModelSpecificNCMRenderer in clover-ai-system/src/models/benchmark_3_models_70_tasks.py (repo 63c29a8)."""
from __future__ import annotations

import collections
import ctypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "ncm"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models.next_concept import NextConceptTransformer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402
from ncm_runtime import (  # noqa: E402
    GRAMMAR_CONNECTOR_WORDS,
    P_BPE,
    P_RH,
    LexicalPointerDecoder,
    generate_from_zpref,
    join_rendered_windows,
    pack_combo_prefix,
)

V_LFM3 = 128000
D_LFM3 = 2048
D_VQ = 384


class Lfm3NativeCPUEngine:
    """ctypes wrapper around liblfm3_cpu_engine (.dll on Windows, .so on Linux)."""

    def __init__(self, pruned_dataset_dir: str | Path, engine_dir: str | Path):
        engine_dir = Path(engine_dir)
        if os.name == "nt":
            os.add_dll_directory(str(engine_dir))
            self.so_path = engine_dir / "liblfm3_cpu_engine.dll"
        else:
            self.so_path = engine_dir / "liblfm3_cpu_engine.so"
        self.lib = ctypes.CDLL(str(self.so_path))
        self.lib.lfm3_engine_init.argtypes = [ctypes.c_char_p]
        self.lib.lfm3_engine_init.restype = ctypes.c_int
        self.lib.lfm3_forward_24_layers.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self.lib.lfm3_forward_24_layers.restype = ctypes.c_float
        self.lib.lfm3_forward_24_layers_clubbed.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.lfm3_forward_24_layers_clubbed.restype = ctypes.c_float
        rc = self.lib.lfm3_engine_init(str(pruned_dataset_dir).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"lfm3_engine_init({pruned_dataset_dir}) failed with rc={rc}")

    def forward_ids(self, token_ids, seg_starts=None):
        if isinstance(token_ids, torch.Tensor):
            ids_list = [int(x) for x in token_ids.view(-1).tolist()]
        else:
            ids_list = [int(x) for x in token_ids]
        T = len(ids_list)
        c_ids = (ctypes.c_int * T)(*ids_list)
        out_arr = np.zeros((T, D_LFM3), dtype=np.float32)
        out_ptr = out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if seg_starts is not None:
            c_seg = (ctypes.c_int * T)(*[int(x) for x in seg_starts])
            ms = float(self.lib.lfm3_forward_24_layers_clubbed(c_ids, c_seg, T, out_ptr))
        else:
            ms = float(self.lib.lfm3_forward_24_layers(c_ids, T, out_ptr))
        return torch.from_numpy(out_arr), ms


class Lfm3NativeHybridLink(nn.Module):
    def __init__(self, d: int = D_LFM3, bottleneck: int = 512):
        super().__init__()
        self.pre_ln = nn.LayerNorm(d)
        self.proj1 = nn.Linear(d, bottleneck)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(bottleneck, d)
        self.post_ln = nn.LayerNorm(d)
        self.out_scale = nn.Parameter(torch.tensor(0.024))

    def forward(self, h_lfm3: torch.Tensor) -> torch.Tensor:
        hf = h_lfm3.float()
        z = self.post_ln(hf + self.proj2(self.act(self.proj1(self.pre_ln(hf)))))
        return z * self.out_scale


class Lfm3ComboBridge(nn.Module):
    def __init__(self, d_in: int = D_LFM3, d_out: int = D_VQ):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, 768)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(768, d_out)
        self.ln_out = nn.LayerNorm(d_out)
        self.scale = nn.Parameter(torch.tensor(8.5))
        self.seg_rh = nn.Parameter(torch.zeros(1, 1, d_out))
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, r_lfm3: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(r_lfm3.float())
        x = self.fc2(self.act(self.fc1(x)))
        x = self.ln_out(x) * self.scale
        return (x * self.gate + self.seg_rh).to(r_lfm3.dtype)


class Lfm3ExpertBridge(nn.Module):
    def __init__(self, d_lfm3: int = D_LFM3, d_vq: int = D_VQ):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_lfm3)
        self.proj1 = nn.Linear(d_lfm3, d_vq)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(d_vq, d_vq)
        self.ln_out = nn.LayerNorm(d_vq)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, h_lfm3: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(h_lfm3.float())
        x = self.proj2(self.act(self.proj1(x)))
        return (torch.tanh(self.alpha) * self.ln_out(x)).to(h_lfm3.dtype)


class HFModelTokenizerAdapter:
    def __init__(self, model_dir: Path, vocab_size: int, eos_id: int, pad_id: int, semicolon_id: int):
        self.hf_tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.vocab_size = vocab_size
        self.eos_token_id = eos_id
        self.pad_token_id = pad_id
        self.semicolon_id = semicolon_id

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        ids = self.hf_tok.encode(text, add_special_tokens=False)
        return [int(x) for x in ids if 0 <= int(x) < self.vocab_size]

    def encode_plain(self, text: str) -> list[int]:
        return self.encode(text, allow_special=False)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        valid = [int(x) for x in ids if 0 <= int(x) < self.vocab_size and int(x) not in (self.eos_token_id, self.pad_token_id)]
        if not valid:
            return ""
        return self.hf_tok.decode(valid, skip_special_tokens=skip_special_tokens)


class _TokenizerShell:
    """generate_from_zpref only reads tk.vqvae."""

    def __init__(self, vqvae):
        self.vqvae = vqvae


class Lfm3NCMRenderer:
    """The "lfm3-1.7b" branch of ModelSpecificNCMRenderer, with all paths passed in."""

    def __init__(self, data_dir: str | Path, engine_dir: str | Path):
        data_dir = Path(data_dir)
        self.model_key = "lfm3-1.7b"
        self.dev = torch.device("cpu")
        hf_dir = data_dir / "lfm3" / "hf"
        pruned_dir = data_dir / "lfm3" / "pruned"
        ncm_dir = data_dir / "ncm"

        cfg = OmegaConf.load(str(ncm_dir / "config.yaml"))
        vq_cfg = dict(cfg.tokenizer.vqvae)
        vq_cfg.pop("checkpoint_path", None)
        self.vqvae = VQVAE(**vq_cfg).cpu().eval()
        self.tk = _TokenizerShell(self.vqvae)

        mcfg = cfg.model
        mcfg.prefix_len = 96
        mcfg.predict_level0 = True
        self.net = NextConceptTransformer(vqvae_local=self.vqvae, config=mcfg).cpu().eval()

        sd_rt = torch.load(str(ncm_dir / "lfm3_ncm_runtime.pt"), map_location="cpu", weights_only=True)
        self.net.load_state_dict(sd_rt["net"], strict=True)
        self.net.eval()
        self.copy_dec = LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).cpu().eval()
        self.copy_dec.load_state_dict(sd_rt["ptr_dec"])
        self.copy_dec.eval()

        self.vocab_size = V_LFM3
        self.hidden_size = D_LFM3
        self.tok = HFModelTokenizerAdapter(hf_dir, V_LFM3, eos_id=124900, pad_id=124893, semicolon_id=26)
        semi_ids = self.tok.encode(";")
        if semi_ids:
            self.tok.semicolon_id = semi_ids[0]
        self.c_engine = Lfm3NativeCPUEngine(pruned_dir, engine_dir)
        emb_mm = np.memmap(pruned_dir / "global-0" / "embed_tokens.bf16.bin", dtype=np.uint16, mode="c", shape=(V_LFM3, D_LFM3))
        self.lm_head_w = torch.from_numpy(emb_mm).view(torch.bfloat16)

        self.vqvae.in_vocab_size = V_LFM3
        self.vqvae.encoder.wte = nn.Embedding(V_LFM3, D_VQ)
        self.vqvae.decoder.output_layer = nn.Linear(D_VQ, V_LFM3, bias=False)
        sd_m = torch.load(str(ncm_dir / "lfm3_hybrid_ncm_bridges.pt"), map_location="cpu", weights_only=False)
        self.vqvae.load_state_dict(sd_m["vqvae"])
        self.vqvae.eval()

        self.hybrid_link = Lfm3NativeHybridLink(d=D_LFM3, bottleneck=512).eval()
        self.combo_bridge = Lfm3ComboBridge(D_LFM3, D_VQ).eval()
        self.expert_bridge = Lfm3ExpertBridge(D_LFM3, D_VQ).eval()
        self.hybrid_link.load_state_dict(sd_m["lfm3_native_hybrid_link"])
        self.combo_bridge.load_state_dict(sd_m["lfm3_combo_bridge"])
        self.expert_bridge.load_state_dict(sd_m["lfm3_expert_bridge"])
        del sd_m, sd_rt

        self.grammar_ids = {self.tok.eos_token_id}
        self.repeatable_grammar_ids = {self.tok.eos_token_id}
        for gw in GRAMMAR_CONNECTOR_WORDS:
            for variant in (gw, " " + gw):
                for tid in self.tok.encode(variant):
                    if 0 <= tid < self.vocab_size:
                        self.grammar_ids.add(tid)
                        if gw.lower() in (
                            "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "by", "with", "from",
                            "as", "is", "are", "was", "were", ".", ",", ";", ":", "-", "/", "//", "://", "_", "=", "&", "(", ")", "[", "]",
                        ):
                            self.repeatable_grammar_ids.add(tid)

        levels = list(self.vqvae.vq.levels)
        self.sched = [1.0 - 0.6 * (k / (len(levels) - 1)) for k in range(len(levels))]

    def _split_overlong_plans_for_model_tok(self, plans: list[str], max_bpe: int = 38) -> list[str]:
        if not plans:
            return []
        expanded: list[str] = []
        for raw_w in plans:
            if not raw_w or not str(raw_w).strip():
                continue
            has_leading_nl = str(raw_w).startswith("\n")
            cp = str(raw_w).lstrip("\n").strip()
            if cp.startswith("```") or ("####" in cp and cp.startswith("Final Answer:")):
                expanded.append(("\n" + cp) if has_leading_nl else cp)
                continue
            tok_ids = self.tok.encode(cp)
            if len(tok_ids) <= max_bpe:
                expanded.append(("\n" + cp) if has_leading_nl else cp)
                continue
            words = cp.split()
            chunks: list[str] = []
            cur_words: list[str] = []
            for w in words:
                cand = " ".join(cur_words + [w])
                if cur_words and len(self.tok.encode(cand)) > max_bpe:
                    chunks.append(" ".join(cur_words))
                    cur_words = [w]
                else:
                    cur_words.append(w)
            if cur_words:
                chunks.append(" ".join(cur_words))
            for c_idx, ch in enumerate(chunks):
                if c_idx == 0 and has_leading_nl:
                    expanded.append("\n" + ch)
                else:
                    expanded.append(ch)
        return expanded if expanded else plans

    def _build_window_vocab_mask(self, plan_str: str) -> torch.Tensor:
        allowed = set(self.grammar_ids)
        allowed.update(self.tok.encode(plan_str))
        for w in re.findall(r"[A-Za-z0-9.,:!?'£$€%/_\-+=&#()\[\]]+", plan_str):
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize(), w.lower(), " " + w.lower()):
                allowed.update(self.tok.encode(v))
        mask = torch.ones(self.vocab_size, dtype=torch.bool)
        t_allow = torch.tensor([x for x in allowed if 0 <= x < self.vocab_size], dtype=torch.long)
        mask[t_allow] = False
        return mask

    def _forward_pointer_decoder_model_vocab(self, z_hat, z_rh, msk_rh, z_bpe, p_pad):
        dec = self.copy_dec
        B, T, C = z_hat.shape
        msk_bpe = (p_pad == self.tok.pad_token_id)
        t_pos = dec.tgt_pos(torch.arange(T))[None, :, :]
        p_pos = dec.plan_pos(torch.arange(P_BPE))[None, :, :]
        r_pos = dec.rh_pos(torch.arange(P_RH))[None, :, :]

        mem_bpe = z_bpe + p_pos
        mem_rh = z_rh + r_pos
        mem = torch.cat([mem_rh, mem_bpe], dim=1)
        mem_mask = torch.cat([msk_rh, msk_bpe], dim=1)

        x = z_hat + t_pos
        for blk in dec.blocks:
            x = blk(x, mem, mem_mask)
        z_refined = z_hat + dec.res_scale * dec.ln_out(x)
        base_logits = self.vqvae.decoder(z_refined)

        cq = F.normalize(dec.copy_q(x), dim=-1)
        ck = F.normalize(dec.copy_k(mem_bpe), dim=-1)
        sim = torch.bmm(cq, ck.transpose(1, 2))

        p_len = (~msk_bpe).sum(dim=1, keepdim=True).clamp(min=1).float()
        has_semi = (p_pad == self.tok.semicolon_id).any(dim=1, keepdim=True)
        expected_T = torch.where(has_semi, (p_len * 1.25 + 2.0).clamp(min=4.0, max=float(T)), p_len)
        t_pos_idx = torch.arange(T, dtype=torch.float32)[None, :, None]
        p_pos_idx = torch.arange(P_BPE, dtype=torch.float32)[None, None, :]

        diag_soft = -2.5 * ((t_pos_idx / expected_T[:, :, None]) - (p_pos_idx / p_len[:, :, None])).pow(2)
        diag_clause = -45.0 * (t_pos_idx - p_pos_idx).pow(2)
        diag_bias = torch.where(has_semi[:, :, None], diag_soft, diag_clause)
        sim = (sim * 12.0 + diag_bias).masked_fill(msk_bpe[:, None, :], -1e4)
        attn_probs = F.softmax(sim, dim=-1)
        ctx_vec = torch.bmm(attn_probs, z_bpe)

        g_copy = torch.sigmoid(dec.copy_gate(torch.cat([x, ctx_vec], dim=-1)))
        soft_scale = g_copy * dec.copy_scale.clamp(min=0.0, max=16.0)
        in_bounds = (t_pos_idx < p_len[:, :, None])
        clause_scale = torch.where(in_bounds, soft_scale + 65.0, torch.zeros_like(soft_scale))
        eff_scale = torch.where(has_semi[:, :, None], soft_scale, clause_scale)
        copy_boost = attn_probs * eff_scale

        idx_exp = p_pad[:, None, :].expand(B, T, P_BPE)
        base_logits.scatter_add_(2, idx_exp, copy_boost.to(base_logits.dtype))
        return base_logits

    def _dec_model_aligned(self, row_logits, plan_ids):
        pad_id = self.tok.pad_token_id
        eos_id = self.tok.eos_token_id
        valid_p = [int(x) for x in plan_ids if 0 <= int(x) < self.vocab_size and int(x) not in (pad_id, eos_id)]
        if not valid_p:
            return "", []

        cont_map: dict[int, set[int]] = {}
        for a, b in zip(valid_p[:-1], valid_p[1:]):
            bs = self.tok.decode([b], skip_special_tokens=False)
            if bs and (not bs.startswith(" ")) and (bs[0].isalnum() or bs[0] in (":", ".", "'", "/", "-", "_", "=", "?", "&", "#")):
                cont_map.setdefault(a, set()).add(b)

        max_counts = collections.Counter(valid_p)
        content_set = {tid for tid in valid_p if tid not in self.repeatable_grammar_ids}
        emitted_counts: collections.Counter[int] = collections.Counter()

        T = row_logits.size(0)
        max_len = min(T, len(valid_p) + 10)
        clean_ids: list[int] = []
        prev_tid = -1
        prev_word = ""
        for t in range(max_len):
            lg = row_logits[t].clone()
            if prev_tid >= 0:
                if emitted_counts[prev_tid] >= max_counts.get(prev_tid, 1):
                    lg[prev_tid] -= 30.0
                if prev_tid in cont_map:
                    for nxt_b in cont_map[prev_tid]:
                        if emitted_counts[nxt_b] < max_counts.get(nxt_b, 1):
                            lg[nxt_b] += 12.0
            for tid, cnt in emitted_counts.items():
                if tid not in self.repeatable_grammar_ids and cnt >= max_counts.get(tid, 1):
                    lg[tid] -= 30.0

            all_covered = bool(content_set) and all(
                emitted_counts[c_id] >= max_counts.get(c_id, 1) for c_id in content_set
            )
            if all_covered and t >= len(valid_p) - 1:
                lg[eos_id] += 18.0

            tid = int(lg.argmax().item())
            if tid == eos_id:
                break
            if 0 <= tid < self.vocab_size and tid != pad_id:
                ws = self.tok.decode([tid], skip_special_tokens=False).strip().lower()
                if ws and ws == prev_word and len(ws) >= 2 and emitted_counts[tid] >= max_counts.get(tid, 1):
                    lg[tid] -= 30.0
                    tid = int(lg.argmax().item())
                    if tid == eos_id:
                        break
                    ws = self.tok.decode([tid], skip_special_tokens=False).strip().lower()
                clean_ids.append(tid)
                emitted_counts[tid] += 1
                prev_tid = tid
                if ws:
                    prev_word = ws
                if bool(content_set) and all(emitted_counts[c_id] >= max_counts.get(c_id, 1) for c_id in content_set):
                    raw_piece = self.tok.decode([tid], skip_special_tokens=False).rstrip()
                    if raw_piece.endswith((".", "!", "?")) and t >= len(valid_p) - 1:
                        break

        decoded_str = self.tok.decode(clean_ids).strip()
        all_content_emitted = all(
            emitted_counts[c_id] >= max_counts.get(c_id, 1) for c_id in content_set
        )
        has_dangling_tail = bool(
            re.search(
                r"(?:\b(?:and|or|to|for|of|in|with|by|from|the|a|an|as|at|on)\s*$|,\s*$|\d+,\d{1,2}$)",
                decoded_str,
                flags=re.IGNORECASE,
            )
        )
        if not decoded_str or not all_content_emitted or has_dangling_tail:
            return self.tok.decode(valid_p), list(valid_p)
        return decoded_str, clean_ids

    @torch.no_grad()
    def score_mcq_options(self, prompt_text: str):
        base_q = re.sub(r"\n*State the correct option letter.*$", "", prompt_text.strip(), flags=re.IGNORECASE).strip()
        q_stem = f"{base_q}\nAnswer: ("
        ids = self.tok.encode_plain(q_stem)[-256:]
        h_c, c_ms = self.c_engine.forward_ids(ids)
        h_last = h_c[-1].to(torch.bfloat16)
        logits = F.linear(h_last, self.lm_head_w).float()
        l_ids = [self.tok.encode_plain(L)[0] for L in ("A", "B", "C", "D")]
        sub_lg = torch.tensor([float(logits[i].item()) for i in l_ids], dtype=torch.float32)
        probs = F.softmax(sub_lg, dim=-1).tolist()
        prob_map = {L: round(float(p), 4) for L, p in zip(("A", "B", "C", "D"), probs)}
        best_L = max(prob_map, key=lambda k: prob_map[k])
        return best_L, prob_map, c_ms

    @torch.no_grad()
    def render_plans_in_model_vocab(self, plans: list[str], cands: int = 2) -> tuple[str, list[int], dict[str, Any]]:
        t0 = time.perf_counter()
        plans = self._split_overlong_plans_for_model_tok(plans, max_bpe=38)
        clean_plans = [p.lstrip("\n").strip() for p in plans]
        is_code_win = [
            cp.startswith("```") or ("####" in cp and cp.startswith("Final Answer:"))
            for cp in clean_plans
        ]

        pad_id = self.tok.pad_token_id
        encoded_plans = [self.tok.encode(p)[:P_BPE] for p in clean_plans]
        p_pad = torch.tensor(
            [ids + [pad_id] * (P_BPE - len(ids)) for ids in encoded_plans],
            dtype=torch.long,
        )
        pad_mask_bpe = (p_pad == pad_id)

        flat_ids = p_pad.reshape(-1)
        non_pad_idx = torch.nonzero(flat_ids != pad_id, as_tuple=False).view(-1)
        if non_pad_idx.numel() == 0:
            non_pad_idx = torch.tensor([0], dtype=torch.long)
        valid_ids = flat_ids[non_pad_idx]

        chunk_cap = 128
        valid_chunks = []
        c_total_ms = 0.0
        for st in range(0, valid_ids.numel(), chunk_cap):
            ed = min(valid_ids.numel(), st + chunk_cap)
            h_chunk, ms_c = self.c_engine.forward_ids(valid_ids[st:ed])
            valid_chunks.append(h_chunk)
            c_total_ms += ms_c

        valid_h = torch.cat(valid_chunks, dim=0)
        h_full = torch.zeros((flat_ids.numel(), self.hidden_size), dtype=torch.float32)
        h_full[non_pad_idx] = valid_h
        h_backbone = h_full.view(len(clean_plans), P_BPE, self.hidden_size)

        r_latent = self.hybrid_link(h_backbone)
        z_rh = self.combo_bridge(r_latent).float().masked_fill(pad_mask_bpe.unsqueeze(-1), 0.0)
        msk_r = pad_mask_bpe

        z_exp = self.expert_bridge(h_backbone).float().masked_fill(pad_mask_bpe.unsqueeze(-1), 0.0)
        z_bpe = self.vqvae.encoder.wte(p_pad) + z_exp
        z_pref, pmask = pack_combo_prefix(z_rh, msk_r, z_bpe, pad_mask_bpe)

        W = len(clean_plans)
        win_masks = [self._build_window_vocab_mask(p) for p in clean_plans]

        torch.manual_seed(42)
        g1 = generate_from_zpref(self.net, self.tk, z_pref, pmask, self.sched)
        zh1, _ = self.vqvae.vq.idx_to_zhats(g1, return_zlevels=True)
        lg1 = self._forward_pointer_decoder_model_vocab(zh1[-1], z_rh, msk_r, z_bpe, p_pad)

        txts: list[str] = []
        out_ids: list[int] = []
        for w_i in range(W):
            if is_code_win[w_i]:
                c_ids = self.tok.encode(clean_plans[w_i])
                txts.append(clean_plans[w_i])
                out_ids.extend(c_ids)
            else:
                p_list = p_pad[w_i].tolist()
                lg1[w_i, :, win_masks[w_i]] -= 80.0
                ts, is_ = self._dec_model_aligned(lg1[w_i], p_list)
                txts.append(ts.strip())
                out_ids.extend(is_)

        rendered_text = join_rendered_windows(txts, plans)
        total_render_ms = (time.perf_counter() - t0) * 1000.0
        all_ids_in_vocab = all(0 <= int(tid) < self.vocab_size for tid in out_ids)
        meta = {
            "model_key": self.model_key,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "c_engine_ms": round(c_total_ms, 2),
            "ncm_plus_c_ms": round(total_render_ms, 2),
            "emitted_token_count": len(out_ids),
            "all_token_ids_valid_in_model_vocab": all_ids_in_vocab,
        }
        return rendered_text, out_ids, meta
