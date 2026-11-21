#!/usr/bin/env python3
"""Stage-1 Pilot: `LiquidAI/LFM3-Omni-2B` Teacher Logit Distillation -> 24-Layer `Block-TrunkLM-24`.

Extracts true causal SDPA Top-K logits & intermediate hidden states (Layers 12 & 24) from
`LFM3-Omni-2B`, runs a controlled 3-variant 24-layer `4-bit` Giant-Integer `Block-TrunkLM-24`
bake-off (`d_model=512`, `B=64` Block-Circulant, `K=32` causal taps = `336 KB` total 4-bit trunk
weights across all 24 layers), and exports the winning model to `block_trunklm24.bin` for
verification in Pure-C11 `block_trunklm_c.c` (`x-cruncher v1.6.1`).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer


def quantize_u4(w: torch.Tensor, soft_alpha: float = 1.0):
    s = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5) / 3.0
    q_int = torch.clamp(torch.round(w / s), -8, 7)
    q_hard = w + (q_int * s - w).detach()
    u4 = (q_int.detach() + 8).to(torch.uint8)
    if soft_alpha >= 1.0:
        return q_hard, u4, s.squeeze(-1).detach()
    w_clamped = torch.clamp(w / s, -8.0, 7.0) * s
    return soft_alpha * q_hard + (1.0 - soft_alpha) * w_clamped, u4, s.squeeze(-1).detach()


class BlockCirculantProj(nn.Module):
    def __init__(self, d_model: int = 512, block_b: int = 64, zero_init: bool = False):
        super().__init__()
        self.d_model, self.B, self.nb = d_model, block_b, d_model // block_b
        scale = 0.002 if zero_init else (0.5 / math.sqrt(d_model))
        self.c = nn.Parameter(torch.randn(self.nb, self.nb, block_b) * scale)

    def forward(self, x: torch.Tensor, soft_alpha: float = 1.0) -> torch.Tensor:
        c_q, _, _ = quantize_u4(self.c, soft_alpha)
        orig_shape = x.shape[:-1]
        xs = x.reshape(-1, self.nb, self.B)
        yf = torch.einsum("tib,oib->tob", torch.fft.rfft(xs, n=self.B, dim=-1), torch.fft.rfft(c_q, n=self.B, dim=-1))
        return torch.fft.irfft(yf, n=self.B, dim=-1).reshape(*orig_shape, self.d_model)


class BlockTrunkLayer(nn.Module):
    def __init__(self, d_model: int = 512, block_b: int = 64, k_taps: int = 32, butterfly_perm: bool = False):
        super().__init__()
        self.d_model, self.k_taps, self.butterfly_perm = d_model, k_taps, butterfly_perm
        self.norm1 = nn.RMSNorm(d_model)
        self.time_w = nn.Parameter(torch.randn(d_model, k_taps) * (0.3 / math.sqrt(k_taps)))
        self.gate_proj = BlockCirculantProj(d_model, block_b)
        self.norm2 = nn.RMSNorm(d_model)
        self.w1 = BlockCirculantProj(d_model, block_b)
        self.w2 = BlockCirculantProj(d_model, block_b, zero_init=True)
        self.diag_scale = nn.Parameter(torch.ones(d_model) * 0.25)

    def forward(self, x: torch.Tensor, soft_alpha: float = 1.0) -> torch.Tensor:
        h = self.norm1(x)
        tw_q, _, _ = quantize_u4(self.time_w, soft_alpha)
        gate = torch.sigmoid(self.gate_proj(h, soft_alpha))
        conv = F.conv1d(F.pad(h.transpose(1, 2), (self.k_taps - 1, 0)), tw_q.flip(-1).unsqueeze(1), groups=self.d_model).transpose(1, 2)
        x = x + gate * conv

        h2 = self.norm2(x)
        u = F.silu(self.w1(h2, soft_alpha) * self.diag_scale)
        if self.butterfly_perm:
            b_sz, t_sz, _ = u.shape
            u = u.reshape(b_sz, t_sz, 8, 8, 8).transpose(2, 3).reshape(b_sz, t_sz, 512)
        return x + self.w2(u, soft_alpha)


class BlockTrunkLM24(nn.Module):
    def __init__(self, embed_init: torch.Tensor, d_model: int = 512, block_b: int = 64,
                 n_layers: int = 24, k_taps: int = 32, butterfly_perm: bool = False):
        super().__init__()
        vocab_sz = embed_init.size(0)
        self.embed = nn.Embedding(vocab_sz, d_model)
        self.embed.weight.data.copy_(embed_init)
        self.embed.weight.requires_grad_(False)
        self.layers = nn.ModuleList([BlockTrunkLayer(d_model, block_b, k_taps, butterfly_perm) for _ in range(n_layers)])
        self.final_norm = nn.RMSNorm(d_model)
        self.logit_scale = nn.Parameter(torch.tensor(3.0))
        self.lm_head = nn.Linear(d_model, vocab_sz, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor, soft_alpha: float = 1.0):
        x = self.embed(idx)
        h12 = None
        for i, lyr in enumerate(self.layers):
            x = lyr(x, soft_alpha=soft_alpha)
            if i == 11:
                h12 = x
        h_out = self.final_norm(x) * self.logit_scale
        return self.lm_head(h_out), h12, h_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default="/opt/clover-k3/lfm3-omni-2b-hf")
    ap.add_argument("--corpus-dir", default="/opt/clover-k3/clover-framework")
    ap.add_argument("--out-dir", default="/tmp/xc_pilot_omni")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(42)
    hf_dir = Path(args.hf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(hf_dir / "tokenizer.json"))
    tok_cfg = json.loads((hf_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    bos_id = tok.token_to_id(tok_cfg.get("bos_token") or "<|startoftext|>") or 1

    corpus_texts = []
    repo_dir = Path(args.corpus_dir)
    for pat in ("AGENTS.md", "docs/**/*.md", "clover-ai-system/studio/clover/*.py"):
        for p in sorted(repo_dir.glob(pat)):
            if p.is_file():
                try:
                    t = p.read_text(encoding="utf-8", errors="ignore")
                    if len(t) > 400:
                        corpus_texts.append(t[:12000])
                except OSError:
                    pass
            if sum(len(x) for x in corpus_texts) > 350_000:
                break

    all_ids = tok.encode("\n\n".join(corpus_texts)).ids
    seq_len, n_chunks = 128, 160
    raw_chunks = np.array(all_ids[:n_chunks * seq_len], dtype=np.int64).reshape(n_chunks, seq_len)
    full_seq = np.concatenate([np.full((n_chunks, 1), bos_id, dtype=np.int64), raw_chunks], axis=1)
    input_ids = torch.from_numpy(full_seq[:, :-1])
    target_ids = torch.from_numpy(full_seq[:, 1:])

    pkg_name = "lfm3_omni_pkg"
    spec_pkg = importlib.util.spec_from_file_location(pkg_name, hf_dir / "__init__.py", submodule_search_locations=[str(hf_dir)])
    pkg = importlib.util.module_from_spec(spec_pkg)
    sys.modules[pkg_name] = pkg
    for m_name in ("configuration_lfm3", "lfm3_conv", "modeling_lfm3"):
        sp = importlib.util.spec_from_file_location(f"{pkg_name}.{m_name}", hf_dir / f"{m_name}.py")
        mod = importlib.util.module_from_spec(sp)
        sys.modules[f"{pkg_name}.{m_name}"] = mod
        sp.loader.exec_module(mod)

    raw_cfg = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    lm_cfg = sys.modules[f"{pkg_name}.configuration_lfm3"].Lfm3Config(**(raw_cfg.get("language_config") or raw_cfg.get("text_config")))
    teacher = sys.modules[f"{pkg_name}.modeling_lfm3"].Lfm3ForCausalLM(lm_cfg).to(torch.bfloat16)
    sd = {}
    with safe_open(str(hf_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("model.language_model."):
                sd["model." + k[len("model.language_model."):]] = f.get_tensor(k)
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    teacher.load_state_dict(sd, strict=False)
    teacher.eval()

    top_ids_l, top_val_l, h12_l, h24_l = [], [], [], []
    with torch.no_grad():
        for i in range(0, n_chunks, 8):
            out = teacher(input_ids=input_ids[i:i + 8], output_hidden_states=True, use_cache=False)
            lg = out.logits.float()
            vals, idx = torch.topk(lg, k=64, dim=-1)
            top_ids_l.append(idx.to(torch.int32).cpu())
            top_val_l.append(vals.to(torch.float16).cpu())
            h12_l.append(out.hidden_states[12].to(torch.float16).cpu())
            h24_l.append(out.hidden_states[-1].to(torch.float16).cpu())

    top_ids = torch.cat(top_ids_l, dim=0)
    top_vals = torch.cat(top_val_l, dim=0)
    h12_all = torch.cat(h12_l, dim=0)
    h24_all = torch.cat(h24_l, dim=0)

    uniq_toks = torch.unique(torch.cat([input_ids.flatten(), target_ids.flatten(), top_ids[:, :, :32].flatten().long()]))
    v_act = int(uniq_toks.numel())
    old_to_new = torch.full((128000,), -1, dtype=torch.long)
    old_to_new[uniq_toks] = torch.arange(v_act, dtype=torch.long)

    with torch.no_grad():
        e_teacher = sd["model.embed_tokens.weight"][uniq_toks].float()
        _, _, vh_pca = torch.linalg.svd(e_teacher, full_matrices=False)
        p_pca = vh_pca[:512, :].T.contiguous()
        e_init_512 = (e_teacher @ p_pca).contiguous()
        h12_proj = h12_all.float() @ p_pca
        h24_proj = h24_all.float() @ p_pca
    del teacher, sd

    in_local = old_to_new[input_ids]
    tg_local = old_to_new[target_ids]
    top32_local = old_to_new[top_ids[:, :, :32].long()]
    top32_vals = top_vals[:, :, :32].float()

    def eval_hard_4bit(model_eval: BlockTrunkLM24):
        model_eval.eval()
        tot_nll, tot_kl, tot_top1, n_b = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(128, 160, 8):
                xb, yb = in_local[i:i + 8], tg_local[i:i + 8]
                t_idx, t_val = top32_local[i:i + 8], top32_vals[i:i + 8]
                logits, _, _ = model_eval(xb, soft_alpha=1.0)
                nll = F.cross_entropy(logits.reshape(-1, v_act), yb.reshape(-1))
                p_teach = F.softmax(t_val, dim=-1)
                log_p_stud = F.log_softmax(torch.gather(logits, -1, t_idx), dim=-1)
                kl = F.kl_div(log_p_stud, p_teach, reduction="batchmean") / xb.size(1)
                top1_agree = (logits.argmax(dim=-1) == t_idx[:, :, 0]).float().mean()
                tot_nll += float(nll.item())
                tot_kl += float(kl.item())
                tot_top1 += float(top1_agree.item())
                n_b += 1
        return tot_nll / n_b, math.exp(tot_nll / n_b), tot_kl / n_b, (tot_top1 / n_b) * 100.0

    best_model, best_nll = None, 1e9
    for name, use_kd, use_bperm, use_hints in [
        ("Variant A: Plain Block-Circulant (Hard CE only)", False, False, False),
        ("Variant B: Plain Block-Circulant + Omni Top-32 KD", True, False, False),
        ("Variant C: Butterfly Block-Circulant + Soft->Hard STE + Omni KD + Hints", True, True, True),
    ]:
        torch.manual_seed(100)
        m = BlockTrunkLM24(e_init_512, d_model=512, block_b=64, n_layers=24, k_taps=32, butterfly_perm=use_bperm)
        trainable = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=3e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=2e-4)
        t_s = time.perf_counter()
        m.train()
        for st in range(1, args.steps + 1):
            perm = torch.randint(0, 128, (16,))
            xb, yb = in_local[perm], tg_local[perm]
            alpha = min(1.0, 0.3 + 0.7 * (st / (0.5 * args.steps))) if use_hints else 1.0
            logits, h12, h24 = m(xb, soft_alpha=alpha)
            ce_loss = F.cross_entropy(logits.reshape(-1, v_act), yb.reshape(-1))
            if not use_kd:
                loss = ce_loss
            else:
                t_kd = 1.5
                t_idx, t_val = top32_local[perm], top32_vals[perm]
                p_t = F.softmax(t_val / t_kd, dim=-1)
                s_top = torch.gather(logits, -1, t_idx) / t_kd
                log_z = torch.logsumexp(logits / t_kd, dim=-1, keepdim=True)
                kd_loss = -(p_t * (s_top - log_z)).sum(dim=-1).mean() * t_kd
                loss = 0.7 * kd_loss + 0.3 * ce_loss
                if use_hints:
                    cos12 = 1.0 - F.cosine_similarity(h12, h12_proj[perm], dim=-1).mean()
                    cos24 = 1.0 - F.cosine_similarity(h24, h24_proj[perm], dim=-1).mean()
                    loss = loss + 0.35 * (cos12 + cos24)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()

        val_nll, val_ppl, val_kl, val_agree = eval_hard_4bit(m)
        print(f"[{name}] ({time.perf_counter()-t_s:.1f}s) Val PPL={val_ppl:.2f} | KL={val_kl:.4f} | Top1-Agree={val_agree:.2f}%")
        if val_nll < best_nll:
            best_nll = val_nll
            best_model = m

    best_model.eval()
    with torch.no_grad():
        sample_x = in_local[128:129, :32]
        ref_lg, _, _ = best_model(sample_x, soft_alpha=1.0)
        (out_dir / "sample_tokens.bin").write_bytes(sample_x[0].numpy().astype(np.int32).tobytes())
        (out_dir / "ref_val_logits.bin").write_bytes(ref_lg[0, -1].numpy().astype(np.float32).tobytes())
        eff_final_norm = (best_model.final_norm.weight.detach() * best_model.logit_scale.detach()).numpy().astype(np.float32)
        with open(out_dir / "block_trunklm24.bin", "wb") as bf:
            use_bp = 1 if best_model.layers[0].butterfly_perm else 0
            bf.write(struct.pack("<IIIIII", 24, 512, 64, 32, v_act, use_bp))
            bf.write(best_model.embed.weight.detach().numpy().astype(np.float32).tobytes())
            bf.write(eff_final_norm.tobytes())
            for lyr in best_model.layers:
                bf.write(lyr.norm1.weight.detach().numpy().astype(np.float32).tobytes())
                bf.write(lyr.norm2.weight.detach().numpy().astype(np.float32).tobytes())
                bf.write(lyr.diag_scale.detach().numpy().astype(np.float32).tobytes())
                _, tw_u4, tw_s = quantize_u4(lyr.time_w, 1.0)
                bf.write(tw_s.numpy().astype(np.float32).tobytes())
                bf.write(tw_u4.numpy().tobytes())
                for proj in (lyr.gate_proj, lyr.w1, lyr.w2):
                    _, c_u4, c_s = quantize_u4(proj.c, 1.0)
                    bf.write(c_s.numpy().astype(np.float32).tobytes())
                    bf.write(c_u4.numpy().tobytes())


if __name__ == "__main__":
    main()
