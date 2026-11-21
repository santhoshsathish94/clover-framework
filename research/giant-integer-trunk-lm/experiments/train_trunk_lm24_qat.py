#!/usr/bin/env python3
"""Step C (Part 1): Native 24-Layer 4-Bit Giant-Integer Trunk-LM Quantization-Aware Trainer.

Trains a 24-layer neural language model whose channel projections (circulant/Toeplitz) and sequence
mixers (causal depthwise conv) are natively parameterized as 4-bit integers [0..15] (zero-point 8),
exporting a binary checkpoint consumed directly by `trunk_lm24_c.c`.
"""
from __future__ import annotations

import argparse
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ste_uint4(w: torch.Tensor):
    s = w.abs().mean().clamp(min=1e-4) / 3.0
    q_signed = torch.clamp(torch.round(w / s), -8, 7)
    q_ste = w + (q_signed * s - w).detach()
    u4 = (q_signed.detach() + 8).to(torch.uint8)
    return q_ste, u4, float(s.detach())


class GiantIntLayer(nn.Module):
    def __init__(self, d_model: int = 192, k_taps: int = 16):
        super().__init__()
        self.d_model = d_model
        self.k_taps = k_taps
        self.norm1 = nn.RMSNorm(d_model)
        self.time_w = nn.Parameter(torch.randn(d_model, k_taps) * (1.0 / math.sqrt(k_taps)))
        self.time_gate_toeplitz = nn.Parameter(torch.randn(d_model) * (1.0 / math.sqrt(d_model)))
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn_toeplitz_1 = nn.Parameter(torch.randn(d_model) * (1.0 / math.sqrt(d_model)))
        self.ffn_toeplitz_2 = nn.Parameter(torch.randn(d_model) * (1.0 / math.sqrt(d_model)))
        self.ffn_diag = nn.Parameter(torch.ones(d_model))

    @staticmethod
    def circulant_matmul(x: torch.Tensor, c_vec: torch.Tensor) -> torch.Tensor:
        d = c_vec.shape[-1]
        return torch.fft.irfft(torch.fft.rfft(x, n=d, dim=-1) * torch.fft.rfft(c_vec, n=d, dim=-1), n=d, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        tw_q, _, _ = ste_uint4(self.time_w)
        tg_q, _, _ = ste_uint4(self.time_gate_toeplitz)
        gate = torch.sigmoid(self.circulant_matmul(h, tg_q))
        h_pad = F.pad(h.transpose(1, 2), (self.k_taps - 1, 0))
        conv = F.conv1d(h_pad, tw_q.flip(-1).unsqueeze(1), groups=self.d_model).transpose(1, 2)
        x = x + gate * conv

        h2 = self.norm2(x)
        w1_q, _, _ = ste_uint4(self.ffn_toeplitz_1)
        w2_q, _, _ = ste_uint4(self.ffn_toeplitz_2)
        u = F.silu(self.circulant_matmul(h2, w1_q) * self.ffn_diag)
        return x + self.circulant_matmul(u, w2_q)


class TrunkLM24(nn.Module):
    def __init__(self, vocab: int = 128, d_model: int = 192, n_layers: int = 24, k_taps: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([GiantIntLayer(d_model, k_taps) for _ in range(n_layers)])
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for lyr in self.layers:
            x = lyr(x)
        return self.lm_head(self.final_norm(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", default=".", help="Directory of .md files for training corpus")
    ap.add_argument("--out-dir", default="/tmp/xc_step_c", help="Output checkpoint directory")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    torch.manual_seed(42)
    parts = []
    for p in sorted(Path(args.corpus_root).rglob("*.md")):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            if len(t) > 200:
                parts.append(t)
        except OSError:
            pass
    corpus = "\n\n".join(parts)[:80_000]
    clean_bytes = bytes(b if b < 128 else 32 for b in corpus.encode("utf-8"))
    data_t = torch.tensor(list(clean_bytes), dtype=torch.long)

    d_model = 192
    model = TrunkLM24(vocab=128, d_model=d_model, n_layers=24, k_taps=16)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=3e-4)

    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        starts = torch.randint(0, len(data_t) - 129, (32,))
        xb = torch.stack([data_t[s:s + 128] for s in starts])
        yb = torch.stack([data_t[s + 1:s + 129] for s in starts])
        loss = F.cross_entropy(model(xb).reshape(-1, 128), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step in (1, args.steps // 3, 2 * args.steps // 3, args.steps):
            print(f"Step {step:4d}/{args.steps} | CE={loss.item():.4f} ({loss.item()/math.log(2):.3f} bpc) | {time.perf_counter()-t0:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        prompt = b"Clover is a way of working with "
        ref_logits = model(torch.tensor([[b for b in prompt]], dtype=torch.long))[0, -1].numpy().astype(np.float32)
        (out_dir / "ref_logits.bin").write_bytes(ref_logits.tobytes())
        with open(out_dir / "trunk_lm24_weights.bin", "wb") as bf:
            bf.write(struct.pack("<IIII", 24, d_model, 16, 128))
            bf.write(model.embed.weight.detach().numpy().astype(np.float32).tobytes())
            bf.write(model.final_norm.weight.detach().numpy().astype(np.float32).tobytes())
            for lyr in model.layers:
                _, tw_u4, tw_s = ste_uint4(lyr.time_w)
                _, tg_u4, tg_s = ste_uint4(lyr.time_gate_toeplitz)
                _, w1_u4, w1_s = ste_uint4(lyr.ffn_toeplitz_1)
                _, w2_u4, w2_s = ste_uint4(lyr.ffn_toeplitz_2)
                bf.write(lyr.norm1.weight.detach().numpy().astype(np.float32).tobytes())
                bf.write(lyr.norm2.weight.detach().numpy().astype(np.float32).tobytes())
                bf.write(lyr.ffn_diag.detach().numpy().astype(np.float32).tobytes())
                bf.write(struct.pack("<ffff", tw_s, tg_s, w1_s, w2_s))
                bf.write(tw_u4.numpy().tobytes())
                bf.write(tg_u4.numpy().tobytes())
                bf.write(w1_u4.numpy().tobytes())
                bf.write(w2_u4.numpy().tobytes())


if __name__ == "__main__":
    main()
