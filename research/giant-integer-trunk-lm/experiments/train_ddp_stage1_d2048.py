#!/usr/bin/env python3
"""Full-Scale 4-GPU DDP Stage-1 & Stage-2 Trainer for `Block-TrunkLM-24` (D=2048, B=64, K=64 = 3.75 MiB 4-bit Trunks).

Implements the Claude-Opus-5.5-validated `Variant D` architecture at native LFM3-Omni-2B width (D=2048, V=128000):
  - 32x32 grid of B=64 4-bit Circulant Giant Integers per projection + O(D) in/out channel gain vectors
  - [B, T, 32, 32, 2] Butterfly-Transpose cross-block routing between w1 and w2
  - K=64 causal depthwise Time-Trunk (Salem-Spencer P=4 compatible in Pure-C11 x-cruncher)
  - Tail-Aware Top-64 KL Distillation (T=1.0, modeling p_tail = 1 - sum(p_top64)) + 0.1*CE + Layer 6/12/18/24 Cosine Hints
  - Chunked BF16 128K-vocab LM-head loss for low VRAM & high Tensor-Core throughput
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors import safe_open

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def quantize_u4(w: torch.Tensor, soft_alpha: float = 1.0):
    s = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5) / 3.0
    q_int = torch.clamp(torch.round(w / s), -8, 7)
    q_hard = w + (q_int * s - w).detach()
    if soft_alpha >= 1.0:
        return q_hard, (q_int.detach() + 8).to(torch.uint8), s.squeeze(-1).detach()
    w_clamped = torch.clamp(w / s, -8.0, 7.0) * s
    return soft_alpha * q_hard + (1.0 - soft_alpha) * w_clamped, (q_int.detach() + 8).to(torch.uint8), s.squeeze(-1).detach()


class BlockCirculantProjD(nn.Module):
    """D=2048 -> 2048 via 32x32 grid of B=64 4-bit Circulant Giant Integers (32 KB packed!) + O(D) gains."""
    def __init__(self, d_model: int = 2048, block_b: int = 64, zero_init: bool = False):
        super().__init__()
        self.d_model = d_model
        self.B = block_b
        self.nb = d_model // block_b  # 32
        scale = 0.002 if zero_init else (0.5 / math.sqrt(d_model))
        self.c = nn.Parameter(torch.randn(self.nb, self.nb, block_b) * scale)
        self.in_gain = nn.Parameter(torch.ones(d_model))
        self.out_gain = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor, soft_alpha: float = 1.0) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_f32 = x.float() * self.in_gain
        c_q, _, _ = quantize_u4(self.c, soft_alpha)
        xs = x_f32.reshape(-1, self.nb, self.B)
        xf = torch.fft.rfft(xs, n=self.B, dim=-1)
        cf = torch.fft.rfft(c_q, n=self.B, dim=-1)
        yf = torch.einsum("tib,oib->tob", xf, cf)
        y = torch.fft.irfft(yf, n=self.B, dim=-1).reshape(*orig_shape, self.d_model)
        return (y * self.out_gain).to(x.dtype)


class BlockTrunkLayerD(nn.Module):
    def __init__(self, d_model: int = 2048, block_b: int = 64, k_taps: int = 64, dropout: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.B = block_b
        self.nb = d_model // block_b  # 32
        self.sub = d_model // (self.nb * self.nb)  # 2048 // 1024 = 2
        self.k_taps = k_taps
        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.time_w = nn.Parameter(torch.randn(d_model, k_taps) * (0.3 / math.sqrt(k_taps)))
        self.gate_proj = BlockCirculantProjD(d_model, block_b)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.w1 = BlockCirculantProjD(d_model, block_b)
        self.w2 = BlockCirculantProjD(d_model, block_b, zero_init=True)
        self.diag_scale = nn.Parameter(torch.ones(d_model) * 0.25)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, soft_alpha: float = 1.0) -> torch.Tensor:
        h = self.norm1(x.float()).to(x.dtype)
        tw_q, _, _ = quantize_u4(self.time_w, soft_alpha)
        gate = torch.sigmoid(self.gate_proj(h, soft_alpha))
        h_pad = F.pad(h.float().transpose(1, 2), (self.k_taps - 1, 0))
        conv = F.conv1d(h_pad, tw_q.flip(-1).unsqueeze(1), groups=self.d_model).transpose(1, 2).to(x.dtype)
        x = x + self.drop(gate * conv)

        h2 = self.norm2(x.float()).to(x.dtype)
        u = F.silu(self.w1(h2, soft_alpha).float() * self.diag_scale).to(x.dtype)
        b_sz, t_sz, _ = u.shape
        u = u.reshape(b_sz, t_sz, self.nb, self.nb, self.sub).transpose(2, 3).reshape(b_sz, t_sz, self.d_model)
        return x + self.drop(self.w2(u, soft_alpha))


class BlockTrunkLM24Full(nn.Module):
    def __init__(self, embed_bf16: torch.Tensor, final_norm_bf16: torch.Tensor,
                 d_model: int = 2048, block_b: int = 64, n_layers: int = 24, k_taps: int = 64, dropout: float = 0.05):
        super().__init__()
        vocab_sz = embed_bf16.size(0)
        self.d_model = d_model
        self.B = block_b
        self.k_taps = k_taps
        self.n_layers = n_layers
        self.register_buffer("embed_weight", embed_bf16.to(torch.bfloat16), persistent=False)
        self.layers = nn.ModuleList([BlockTrunkLayerD(d_model, block_b, k_taps, dropout) for _ in range(n_layers)])
        self.final_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.final_norm.weight.data.copy_(final_norm_bf16.float())
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward_hidden(self, idx: torch.Tensor, soft_alpha: float = 1.0):
        x = F.embedding(idx, self.embed_weight)  # [B, T, 2048] in BF16
        hints = {}
        for i, lyr in enumerate(self.layers):
            x = lyr(x, soft_alpha=soft_alpha)
            if (i + 1) in (6, 12, 18):
                hints[i + 1] = x
        h_final = (self.final_norm(x.float()) * self.logit_scale).to(torch.bfloat16)
        hints[24] = h_final
        return h_final, hints


def load_omni_teacher(hf_dir: Path, device: torch.device):
    pkg_name = "lfm3_omni_pkg"
    if pkg_name not in sys.modules:
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
    embed_cpu = sd["model.embed_tokens.weight"].clone()
    fnorm_cpu = sd["model.embedding_norm.weight"].clone()
    teacher.load_state_dict(sd, strict=False)
    del sd
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher.to(device), embed_cpu, fnorm_cpu


def chunked_tail_kd_and_ce(
    h_stud: torch.Tensor,       # [N_tok, 2048] BF16 (requires grad)
    h_teach: torch.Tensor,      # [N_tok, 2048] BF16 (no grad)
    targets: torch.Tensor,      # [N_tok] int64
    embed_w: torch.Tensor,      # [128000, 2048] BF16
    topk: int = 64,
    chunk_sz: int = 512,
    kd_weight: float = 0.9,
):
    """Computes exact Variant-D Tail-Aware Top-K KL + CE in 512-token slices so 128K-vocab logits use <1.2 GB VRAM!"""
    n_tok = h_stud.size(0)
    total_loss = 0.0
    n_chunks = (n_tok + chunk_sz - 1) // chunk_sz
    for s in range(0, n_tok, chunk_sz):
        e = min(s + chunk_sz, n_tok)
        w_frac = (e - s) / float(n_tok)
        hs = h_stud[s:e]
        yb = targets[s:e]

        with torch.no_grad():
            lt = F.linear(h_teach[s:e], embed_w).float()
            t_lse = torch.logsumexp(lt, dim=-1, keepdim=True)
            t_vals, t_idx = torch.topk(lt, k=topk, dim=-1)
            p_top = torch.exp(t_vals - t_lse)                     # [C, topk]
            p_tail = (1.0 - p_top.sum(dim=-1, keepdim=True)).clamp(min=1e-7)

        ls = F.linear(hs, embed_w).float()                        # [C, 128000]
        ce = F.cross_entropy(ls, yb)
        if kd_weight <= 0.0:
            total_loss = total_loss + w_frac * ce
        else:
            s_lse = torch.logsumexp(ls, dim=-1, keepdim=True)
            s_top = torch.gather(ls, -1, t_idx)
            log_q_top = s_top - s_lse                             # [C, topk]
            q_top_sum = torch.exp(log_q_top).sum(dim=-1, keepdim=True)
            log_q_tail = torch.log((1.0 - q_top_sum).clamp(min=1e-7))
            # Cross-entropy H(p_teach, q_stud) over (top-64 + tail bucket) minus H(p_teach)
            nll_kd = -(p_top * log_q_top).sum(dim=-1) - (p_tail * log_q_tail).squeeze(-1)
            ent_t = -(p_top * (t_vals - t_lse)).sum(dim=-1) - (p_tail * torch.log(p_tail)).squeeze(-1)
            kl_tail = (nll_kd - ent_t).mean()
            total_loss = total_loss + w_frac * (kd_weight * kl_tail + (1.0 - kd_weight) * ce)
    return total_loss


@torch.no_grad()
def evaluate_hard_4bit(raw_model: BlockTrunkLM24Full, teacher, data_mm: np.memmap,
                       device: torch.device, seq_len: int = 512, n_batches: int = 16, batch_sz: int = 4):
    raw_model.eval()
    tot_nll, tot_tnll, tot_top1 = 0.0, 0.0, 0.0
    max_start = len(data_mm) - seq_len - 1
    starts = np.linspace(0, max_start, n_batches * batch_sz, dtype=np.int64)
    for b in range(n_batches):
        sb = starts[b * batch_sz:(b + 1) * batch_sz]
        arr = np.stack([np.asarray(data_mm[s:s + seq_len + 1], dtype=np.int64) for s in sb])
        toks = torch.from_numpy(arr).to(device)
        xb, yb = toks[:, :-1], toks[:, 1:]

        t_out = teacher.model(input_ids=xb, use_cache=False)
        ht = t_out.last_hidden_state.reshape(-1, raw_model.d_model)
        hs, _ = raw_model.forward_hidden(xb, soft_alpha=1.0)
        hs = hs.reshape(-1, raw_model.d_model)
        yf = yb.reshape(-1)

        b_nll, b_tnll, b_agree = 0.0, 0.0, 0.0
        n_tok = hs.size(0)
        for s in range(0, n_tok, 512):
            e = min(s + 512, n_tok)
            frac = (e - s) / float(n_tok)
            lt = F.linear(ht[s:e], raw_model.embed_weight).float()
            ls = F.linear(hs[s:e], raw_model.embed_weight).float()
            b_tnll += frac * float(F.cross_entropy(lt, yf[s:e]).item())
            b_nll += frac * float(F.cross_entropy(ls, yf[s:e]).item())
            b_agree += frac * float((ls.argmax(-1) == lt.argmax(-1)).float().mean().item())
        tot_nll += b_nll
        tot_tnll += b_tnll
        tot_top1 += b_agree

    avg_nll = tot_nll / n_batches
    avg_tnll = tot_tnll / n_batches
    return avg_nll, math.exp(min(avg_nll, 20.0)), avg_tnll, math.exp(min(avg_tnll, 20.0)), (tot_top1 / n_batches) * 100.0


def export_c11_d2048(raw_model: BlockTrunkLM24Full, out_path: Path):
    """Export nibble-packed 4-bit Giant-Integer checkpoint (`TRKD` v1 format, 3.75 MiB 4-bit trunks!)."""
    raw_model.eval()
    with torch.no_grad(), open(out_path, "wb") as bf:
        bf.write(b"TRKD")
        bf.write(struct.pack("<IIIIIIf",
                             raw_model.n_layers, raw_model.d_model, raw_model.B,
                             raw_model.k_taps, raw_model.embed_weight.size(0), 1, 1e-5))
        eff_fnorm = (raw_model.final_norm.weight.detach().float() * raw_model.logit_scale.detach().float()).cpu().numpy().astype(np.float32)
        bf.write(eff_fnorm.tobytes())
        for lyr in raw_model.layers:
            bf.write(lyr.norm1.weight.detach().float().cpu().numpy().astype(np.float32).tobytes())
            bf.write(lyr.norm2.weight.detach().float().cpu().numpy().astype(np.float32).tobytes())
            bf.write(lyr.diag_scale.detach().float().cpu().numpy().astype(np.float32).tobytes())
            _, tw_u4, tw_s = quantize_u4(lyr.time_w, 1.0)
            bf.write(tw_s.float().cpu().numpy().astype(np.float32).tobytes())
            tw_np = tw_u4.cpu().numpy().reshape(-1)
            bf.write(((tw_np[0::2] & 0x0F) | ((tw_np[1::2] & 0x0F) << 4)).astype(np.uint8).tobytes())
            for proj in (lyr.gate_proj, lyr.w1, lyr.w2):
                bf.write(proj.in_gain.detach().float().cpu().numpy().astype(np.float32).tobytes())
                bf.write(proj.out_gain.detach().float().cpu().numpy().astype(np.float32).tobytes())
                _, c_u4, c_s = quantize_u4(proj.c, 1.0)
                bf.write(c_s.float().cpu().numpy().astype(np.float32).tobytes())
                c_np = c_u4.cpu().numpy().reshape(-1)
                bf.write(((c_np[0::2] & 0x0F) | ((c_np[1::2] & 0x0F) << 4)).astype(np.uint8).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default="/workspace/trunklm/lfm3-omni-2b-hf")
    ap.add_argument("--data-dir", default="/workspace/trunklm/data")
    ap.add_argument("--train-file", default="train.bin")
    ap.add_argument("--ckpt-dir", default="/workspace/trunklm/checkpoints")
    ap.add_argument("--resume", default="")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--kd-weight", type=float, default=0.9)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--warmup-soft-frac", type=float, default=0.20)
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(1337 + rank)

    hf_dir = Path(args.hf_dir)
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    teacher, embed_cpu, fnorm_cpu = load_omni_teacher(hf_dir, device)
    model = BlockTrunkLM24Full(embed_cpu, fnorm_cpu, d_model=2048, block_b=64, n_layers=24, k_taps=64, dropout=0.05).to(device)
    del embed_cpu, fnorm_cpu

    if args.resume and Path(args.resume).is_file():
        st_dict = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(st_dict["model"], strict=True)
        if rank == 0:
            print(f"Resumed student weights from {args.resume}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trunk_bytes = 24 * (3 * (32 * 32 * 64) + 2048 * 64) // 2
    if rank == 0:
        print(f"=== Block-TrunkLM-24 Full-Scale (D=2048, B=64, K=64) on {world_size}x RTX 3090 ===")
        print(f"  4-Bit Trunk Weights (all 24 layers): {trunk_bytes / (1024*1024):.2f} MiB ({trunk_bytes:,} bytes)")
        print(f"  Total Trainable Parameters (Trunks + O(D) gains/norms): {n_trainable:,}")
        print(f"  Tokens / optimizer step: {world_size * args.micro_batch * args.grad_accum * args.seq_len:,}")

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.08)

    train_mm = np.memmap(data_dir / args.train_file, dtype=np.uint32, mode="r")
    val_mm   = np.memmap(data_dir / "val.bin", dtype=np.uint32, mode="r")
    rng = np.random.default_rng(42 + rank * 997)
    max_idx = len(train_mm) - args.seq_len - 1

    if rank == 0:
        v_nll0, v_ppl0, t_nll0, t_ppl0, ag0 = evaluate_hard_4bit(model, teacher, val_mm, device, seq_len=args.seq_len)
        print(f"  [Step    0] Hard-4Bit Val NLL={v_nll0:.4f} (PPL={v_ppl0:.2f}) | Teacher Val PPL={t_ppl0:.2f} | Top1-Agree={ag0:.2f}%")

    log_path = ckpt_dir / "train_metrics.jsonl"
    t_start = time.time()
    tokens_per_step = world_size * args.micro_batch * args.grad_accum * args.seq_len

    for step in range(1, args.steps + 1):
        ddp_model.train()
        opt.zero_grad(set_to_none=True)
        if args.warmup_soft_frac > 0 and step <= int(args.steps * args.warmup_soft_frac):
            alpha = 0.35 + 0.65 * (step / max(1.0, args.steps * args.warmup_soft_frac))
        else:
            alpha = 1.0

        step_loss = 0.0
        for ga in range(args.grad_accum):
            starts = rng.integers(0, max_idx, size=args.micro_batch)
            arr = np.stack([np.asarray(train_mm[s:s + args.seq_len + 1], dtype=np.int64) for s in starts])
            toks = torch.from_numpy(arr).to(device, non_blocking=True)
            xb, yb = toks[:, :-1], toks[:, 1:]

            with torch.no_grad():
                t_out = teacher.model(input_ids=xb, output_hidden_states=True, use_cache=False)
                ht_final = t_out.last_hidden_state  # [B, T, 2048] BF16
                ht_hints = {6: t_out.hidden_states[6], 12: t_out.hidden_states[12], 18: t_out.hidden_states[18], 24: ht_final}

            # Call forward_hidden via ddp_model.module wrapped through ddp_model forward
            hs_final, hs_hints = ddp_model.module.forward_hidden(xb, soft_alpha=alpha) if world_size == 1 else _ddp_forward(ddp_model, xb, alpha)
            kd_ce = chunked_tail_kd_and_ce(
                hs_final.reshape(-1, 2048),
                ht_final.reshape(-1, 2048),
                yb.reshape(-1),
                model.embed_weight,
                topk=64,
                chunk_sz=512,
                kd_weight=args.kd_weight,
            )
            hint_loss = 0.0
            for lk in (6, 12, 18, 24):
                hint_loss = hint_loss + (1.0 - F.cosine_similarity(hs_hints[lk].float(), ht_hints[lk].float(), dim=-1).mean())
            loss = (kd_ce + 0.25 * hint_loss) / args.grad_accum
            loss.backward()
            step_loss += float(loss.detach().item())

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        sched.step()

        if rank == 0 and (step % 10 == 0 or step == 1):
            el = time.time() - t_start
            tok_s = (step * tokens_per_step) / max(el, 1e-3)
            vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(f"  step {step:4d}/{args.steps} | loss={step_loss:.4f} | alpha={alpha:.2f} | {tok_s:,.0f} tok/s | VRAM={vram_gb:.1f} GB | {el:.1f}s", flush=True)

        if rank == 0 and (step % args.eval_every == 0 or step == args.steps):
            v_nll, v_ppl, _, t_ppl, ag = evaluate_hard_4bit(model, teacher, val_mm, device, seq_len=args.seq_len)
            el = time.time() - t_start
            rec = {"step": step, "tokens_M": round(step * tokens_per_step / 1e6, 2), "val_nll": round(v_nll, 4),
                   "val_ppl": round(v_ppl, 2), "teacher_ppl": round(t_ppl, 2), "top1_agree_pct": round(ag, 2), "elapsed_s": round(el, 1)}
            print(f"  >>> [EVAL Step {step:4d} ({rec['tokens_M']}M tok)] Hard-4Bit Val NLL={v_nll:.4f} | Val PPL={v_ppl:.2f} (Teacher={t_ppl:.2f}) | Top1-Agree={ag:.2f}% <<<", flush=True)
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(rec) + "\n")
            torch.save({"step": step, "model": model.state_dict(), "metrics": rec}, ckpt_dir / "latest.pt")
            export_c11_d2048(model, ckpt_dir / "block_trunklm24_d2048.trkd")

    dist.destroy_process_group()


def _ddp_forward(ddp_mod, xb, alpha):
    return ddp_mod(xb, alpha)


# Bind BlockTrunkLM24Full.forward to forward_hidden so DDP hooks fire on every parameter!
BlockTrunkLM24Full.forward = BlockTrunkLM24Full.forward_hidden


if __name__ == "__main__":
    main()
