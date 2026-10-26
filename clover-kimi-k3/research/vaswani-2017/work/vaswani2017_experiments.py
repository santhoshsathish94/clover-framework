#!/usr/bin/env python3
"""
vaswani2017_experiments.py — Verified from-scratch implementation and experimental verification
of Ashish Vaswani et al. (2017), "Attention Is All You Need", NIPS 2017 (arXiv:1706.03762).

Covers:
  - E2: Scaled Dot-Product Attention 1/sqrt(d_k) Variance & Softmax Jacobian Stabilization
        (Section 3.2.1, Eq. 1 & Footnote 4, p. 4) + Exact Parameter Accounting for Transformer
        (base = 65M, big = 213M, Table 3, p. 9).
  - E3: Sinusoidal Positional Encoding (Section 3.5, p. 6): exact 2x2 block-diagonal linear
        relative-offset operator PE_{pos+k} = M_k PE_{pos} and translation-invariant dot product
        PE_{pos}^T PE_{pos+k} = sum_i cos(k * omega_i); plus Multi-Head Attention subspace
        resolution vs Single-Head averaging dilution (Section 2 & 3.2.2, pp. 2, 4-5).
  - E4: Causal Masking prefix invariance & autoregressive leakage test (Section 3.1 & 3.2.3, pp. 3, 5),
        End-to-End Encoder-Decoder sequence transduction training with Eq. 3 warmup schedule &
        Section 5.4 Label Smoothing (eps_ls = 0.1 vs 0.0, pp. 7-8), and Table 1 per-layer
        complexity crossover O(n^2 d) vs O(n d^2) (Section 4, pp. 6-7).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Authentic 2017 Transformer Components (Section 3, Equations 1-3, pp. 3-7)
# ============================================================================

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None, scale: bool = True):
    """
    Equation 1 (p. 4):
      Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    Q: (..., n_q, d_k), K: (..., n_k, d_k), V: (..., n_k, d_v)
    mask: broadcastable boolean/0-1 tensor where True/1 keeps and False/0 masks out (-inf).
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if scale:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn


class MultiHeadAttention2017(nn.Module):
    """Section 3.2.2 (pp. 4-5): MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O."""
    def __init__(self, d_model: int = 512, h: int = 8, d_k: int = None, d_v: int = None, use_bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_k if d_k is not None else (d_model // h)
        self.d_v = d_v if d_v is not None else (d_model // h)
        self.W_Q = nn.Linear(d_model, h * self.d_k, bias=use_bias)
        self.W_K = nn.Linear(d_model, h * self.d_k, bias=use_bias)
        self.W_V = nn.Linear(d_model, h * self.d_v, bias=use_bias)
        self.W_O = nn.Linear(h * self.d_v, d_model, bias=use_bias)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        B = query.size(0)
        Q = self.W_Q(query).view(B, -1, self.h, self.d_k).transpose(1, 2)  # (B, h, n_q, d_k)
        K = self.W_K(key).view(B, -1, self.h, self.d_k).transpose(1, 2)    # (B, h, n_k, d_k)
        V = self.W_V(value).view(B, -1, self.h, self.d_v).transpose(1, 2)  # (B, h, n_k, d_v)
        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask, scale=True)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_v)
        return self.W_O(out), attn


class PositionwiseFFN2017(nn.Module):
    """Section 3.3, Equation 2 (p. 5): FFN(x) = max(0, x W_1 + b_1) W_2 + b_2."""
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    Section 3.5 (p. 6):
      PE_{(pos, 2i)}   = sin(pos / 10000^{2i / d_model})
      PE_{(pos, 2i+1)} = cos(pos / 10000^{2i / d_model})
    """
    pe = np.zeros((max_len, d_model), dtype=np.float64)
    pos = np.arange(max_len, dtype=np.float64)[:, None]
    i = np.arange(0, d_model, 2, dtype=np.float64)[None, :]
    div_term = np.power(10000.0, i / d_model)
    pe[:, 0::2] = np.sin(pos / div_term)
    pe[:, 1::2] = np.cos(pos / div_term)
    return pe


class EncoderLayer2017(nn.Module):
    """Section 3.1 (p. 3): Post-LN LayerNorm(x + Sublayer(x))."""
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention2017(d_model, h)
        self.ffn = PositionwiseFFN2017(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None):
        attn_out, _ = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.drop(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))
        return x


class DecoderLayer2017(nn.Module):
    """Section 3.1 (p. 3): 3 sub-layers with Post-LN LayerNorm(x + Sublayer(x))."""
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention2017(d_model, h)
        self.cross_attn = MultiHeadAttention2017(d_model, h)
        self.ffn = PositionwiseFFN2017(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        sa_out, _ = self.self_attn(y, y, y, mask=tgt_mask)
        y = self.norm1(y + self.drop(sa_out))
        ca_out, _ = self.cross_attn(y, memory, memory, mask=memory_mask)
        y = self.norm2(y + self.drop(ca_out))
        ffn_out = self.ffn(y)
        y = self.norm3(y + self.drop(ffn_out))
        return y


class AuthenticTransformer2017(nn.Module):
    """
    Complete 2017 Encoder-Decoder Transformer with:
      - 3-way shared embedding + pre-softmax weight matrix (Section 3.4, p. 5)
      - sqrt(d_model) embedding scaling (Section 3.4, p. 5)
      - Sinusoidal positional encoding (Section 3.5, p. 6)
      - Post-LN residual sublayers (Section 3.1, p. 3)
    """
    def __init__(self, vocab_size: int = 37000, N: int = 6, d_model: int = 512, d_ff: int = 2048, h: int = 8, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.shared_emb = nn.Embedding(vocab_size, d_model)
        # Section 3.4 (p. 5): shared pre-softmax/embedding matrix has fan-in std = d_model^{-0.5},
        # which is why embedding lookups are multiplied by sqrt(d_model) to match O(1) sinusoidal PE!
        nn.init.normal_(self.shared_emb.weight, mean=0.0, std=d_model ** -0.5)
        pe = torch.tensor(sinusoidal_positional_encoding(max_len, d_model), dtype=torch.float32)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer2017(d_model, h, d_ff, dropout) for _ in range(N)])
        self.decoder_layers = nn.ModuleList([DecoderLayer2017(d_model, h, d_ff, dropout) for _ in range(N)])

    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.shared_emb(src_ids) * math.sqrt(self.d_model)
        x = self.drop(x + self.pe[:, :x.size(1), :])
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask)
        return x

    def decode(self, tgt_ids: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None) -> torch.Tensor:
        y = self.shared_emb(tgt_ids) * math.sqrt(self.d_model)
        y = self.drop(y + self.pe[:, :y.size(1), :])
        for layer in self.decoder_layers:
            y = layer(y, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        # Section 3.4 (p. 5): share weight matrix between embedding layers and pre-softmax linear transformation
        logits = F.linear(y, self.shared_emb.weight)
        return logits


# ============================================================================
# E2: Scaled Dot-Product Attention 1/sqrt(d_k) & Table 3 Parameter Accounting
# ============================================================================

def run_e2_attention_and_params():
    print("=== E2.1: Scaled Dot-Product Attention 1/sqrt(d_k) Variance & Softmax Gradient Stabilization (Eq. 1 & Footnote 4, p. 4) ===")
    rng = np.random.default_rng(2017)
    n_keys = 32
    n_trials = 500
    print("   d_k | Var(q.k) | Var(q.k/sqrt(d_k)) | Unscaled MaxAttn | Scaled MaxAttn | Unscaled ||J_softmax||_F | Scaled ||J_softmax||_F")
    for d_k in [16, 64, 256, 512, 1024]:
        q = rng.normal(0.0, 1.0, size=(n_trials, 1, d_k))
        K = rng.normal(0.0, 1.0, size=(n_trials, n_keys, d_k))
        dots = np.sum(q * K, axis=-1)  # (n_trials, n_keys)
        var_unscaled = float(np.var(dots))
        var_scaled = float(np.var(dots / np.sqrt(d_k)))

        def softmax_and_jacobian_norm(z):
            z_shift = z - np.max(z, axis=-1, keepdims=True)
            p = np.exp(z_shift) / np.sum(np.exp(z_shift), axis=-1, keepdims=True)
            max_p = float(np.mean(np.max(p, axis=-1)))
            # For p = softmax(z), J_ij = p_i (delta_ij - p_j), so ||J||_F^2 = sum_i p_i^2 - 2 sum_i p_i^3 + (sum_i p_i^2)^2
            s2 = np.sum(p ** 2, axis=-1)
            s3 = np.sum(p ** 3, axis=-1)
            jf = np.sqrt(np.maximum(0.0, s2 - 2.0 * s3 + s2 ** 2))
            return max_p, float(np.mean(jf))

        max_u, jf_u = softmax_and_jacobian_norm(dots)
        max_s, jf_s = softmax_and_jacobian_norm(dots / np.sqrt(d_k))
        print(f"  {d_k:4d} | {var_unscaled:8.2f} | {var_scaled:18.4f} | {max_u:16.4f} | {max_s:14.4f} | {jf_u:24.5f} | {jf_s:22.5f}")
        if d_k == 1024:
            assert abs(var_scaled - 1.0) < 0.05 and jf_s > 3.5 * jf_u

    print("\n=== E2.2: Exact Parameter Accounting of Transformer (base = 65M) & (big = 213M) (Table 3, p. 9) ===")
    # Check both 3-way shared embeddings (Section 3.4, V=37,000) and separate pre-softmax/embedding tables:
    base_model = AuthenticTransformer2017(vocab_size=37000, N=6, d_model=512, d_ff=2048, h=8)
    big_model = AuthenticTransformer2017(vocab_size=37000, N=6, d_model=1024, d_ff=4096, h=16)

    p_enc_base = sum(p.numel() for p in base_model.encoder_layers.parameters())
    p_dec_base = sum(p.numel() for p in base_model.decoder_layers.parameters())
    p_emb_base = sum(p.numel() for p in base_model.shared_emb.parameters())
    p_tot_base = sum(p.numel() for p in base_model.parameters())

    p_enc_big = sum(p.numel() for p in big_model.encoder_layers.parameters())
    p_dec_big = sum(p.numel() for p in big_model.decoder_layers.parameters())
    p_emb_big = sum(p.numel() for p in big_model.shared_emb.parameters())
    p_tot_big = sum(p.numel() for p in big_model.parameters())

    print(f"  Transformer (base) [N=6, d_model=512,  d_ff=2048, h=8,  V=37000]:")
    print(f"    6 Encoder layers = {p_enc_base:10,d} | 6 Decoder layers = {p_dec_base:10,d} | Shared Emb = {p_emb_base:10,d} | Total = {p_tot_base:11,d} (~63.1M / Table 3: 65M)")
    print(f"  Transformer (big)  [N=6, d_model=1024, d_ff=4096, h=16, V=37000]:")
    print(f"    6 Encoder layers = {p_enc_big:10,d} | 6 Decoder layers = {p_dec_big:10,d} | Shared Emb = {p_emb_big:10,d} | Total = {p_tot_big:11,d} (~214.2M / Table 3: 213M)")
    assert 60_000_000 < p_tot_base < 66_000_000 and 210_000_000 < p_tot_big < 216_000_000


# ============================================================================
# E3: Sinusoidal Positional Encoding Linear Operator M_k & Multi-Head Subspaces
# ============================================================================

def run_e3_positional_and_multihead():
    print("\n=== E3.1: Sinusoidal Positional Encoding — Exact Linear Relative-Offset Operator M_k (Section 3.5, p. 6) ===")
    d_model = 512
    max_len = 200
    PE = sinusoidal_positional_encoding(max_len, d_model)

    # 1. Verify that for any fixed offset k, PE_{pos+k} = M_k @ PE_{pos} where M_k is block-diagonal with 2x2 rotation blocks:
    #    sin((pos+k) w_i) = cos(k w_i) sin(pos w_i) + sin(k w_i) cos(pos w_i)
    #    cos((pos+k) w_i) = -sin(k w_i) sin(pos w_i) + cos(k w_i) cos(pos w_i)
    max_lin_err = 0.0
    for k_offset in [-13, -1, 0, 1, 7, 29]:
        M_k = np.zeros((d_model, d_model), dtype=np.float64)
        for idx_i in range(d_model // 2):
            omega_i = 1.0 / (10000.0 ** (2.0 * idx_i / d_model))
            c, s = np.cos(k_offset * omega_i), np.sin(k_offset * omega_i)
            M_k[2 * idx_i:2 * idx_i + 2, 2 * idx_i:2 * idx_i + 2] = [[c, s], [-s, c]]
        for pos in range(20, 120):
            pred_pe = M_k @ PE[pos]
            true_pe = PE[pos + k_offset]
            max_lin_err = max(max_lin_err, float(np.max(np.abs(pred_pe - true_pe))))

    # 2. Verify shift invariance of dot product PE_{pos}^T PE_{pos+k} = sum_{i=0}^{d/2-1} cos(k * omega_i):
    max_shift_var = 0.0
    for k_offset in [0, 1, 5, 15, 40]:
        dots_across_pos = [float(PE[pos] @ PE[pos + k_offset]) for pos in range(0, 100)]
        max_shift_var = max(max_shift_var, float(np.std(dots_across_pos)))
    print(f"  Max ||PE_{{pos+k}} - M_k PE_{{pos}}||_inf across offsets k in {{-13..+29}} and pos in {{20..120}}: {max_lin_err:.2e}")
    print(f"  Max std_pos(PE_{{pos}}^T PE_{{pos+k}}) across positions 0..100 (exact shift invariance!): {max_shift_var:.2e}")
    print(f"  Relative dot products PE_{{pos}}^T PE_{{pos+k}}: k=0 -> {PE[0]@PE[0]:.2f}, k=1 -> {PE[0]@PE[1]:.2f}, k=5 -> {PE[0]@PE[5]:.2f}, k=20 -> {PE[0]@PE[20]:.2f}")
    assert max_lin_err < 1e-13 and max_shift_var < 1e-12

    print("\n=== E3.2: Multi-Head Subspace Resolution vs Single-Head Averaging Dilution (Section 2 & 3.2.2, pp. 2, 4-5) ===")
    # Task: A sequence has length n = 6, with d_model = 16:
    #   - Coords 0..3:  key-tag A (1,0,0,0) at pos j_A, key-tag B (0,1,0,0) at pos j_B
    #   - Coords 4..9:  payload subspace A (6 floats)
    #   - Coords 10..15: payload subspace B (6 floats)
    # Query at pos 0 needs to output [payload_A(j_A), payload_B(j_B)] in R^{12}.
    # Compare Single-Head Attention (h=1, d_k=d_v=16) vs 2-Head Attention (h=2, d_k=d_v=8) with identical total parameter count!
    torch.manual_seed(2017)
    N_data = 256
    n_seq = 6
    d_m = 16
    X_seq = torch.zeros(N_data, n_seq, d_m)
    Y_target = torch.zeros(N_data, d_m)
    rng_mh = np.random.default_rng(2017)
    for b in range(N_data):
        j_A, j_B = rng_mh.choice(range(1, n_seq), size=2, replace=False)
        X_seq[b, 0, 0] = 1.0  # Query wants both A and B
        X_seq[b, 0, 1] = 1.0
        X_seq[b, j_A, 2] = 1.0  # Key marker A
        X_seq[b, j_B, 3] = 1.0  # Key marker B
        pay_A = torch.randn(6)
        pay_B = torch.randn(6)
        # Put distractor noise in the other subspace at j_A, j_B and at other positions so averaging corrupts the signal:
        for pos in range(1, n_seq):
            X_seq[b, pos, 4:10] = torch.randn(6)
            X_seq[b, pos, 10:16] = torch.randn(6)
        X_seq[b, j_A, 4:10] = pay_A
        X_seq[b, j_B, 10:16] = pay_B
        Y_target[b, 4:10] = pay_A
        Y_target[b, 10:16] = pay_B

    def train_head_comparison(h_heads: int):
        torch.manual_seed(2017)
        mha = MultiHeadAttention2017(d_model=d_m, h=h_heads, use_bias=False)
        opt = torch.optim.Adam(mha.parameters(), lr=0.02)
        for _ in range(300):
            opt.zero_grad()
            out, attn = mha(X_seq[:, :1, :], X_seq, X_seq)  # query is pos 0: (B, 1, d_m)
            loss = F.mse_loss(out[:, 0, 4:], Y_target[:, 4:])
            loss.backward()
            opt.step()
        with torch.no_grad():
            out, attn = mha(X_seq[:, :1, :], X_seq, X_seq)
            mse = float(F.mse_loss(out[:, 0, 4:], Y_target[:, 4:]).item())
            max_attn = float(attn.max(dim=-1).values.mean().item())
        return mse, max_attn

    mse_h1, peak_h1 = train_head_comparison(h_heads=1)
    mse_h2, peak_h2 = train_head_comparison(h_heads=2)
    print(f"  Single-Head (h=1, d_k=d_v=16): MSE = {mse_h1:.5f}, mean peak attention weight = {peak_h1:.4f} (rank-1 scalar weight cannot unmix j_A & j_B!)")
    print(f"  Multi-Head  (h=2, d_k=d_v= 8): MSE = {mse_h2:.5f}, mean peak attention weight = {peak_h2:.4f} ({mse_h1/max(mse_h2,1e-9):.1f}x lower MSE!)")
    assert mse_h1 > 0.35 and mse_h2 < 0.005


# ============================================================================
# E4: Causal Masking Leakage, Sequence Transduction Training (Eq. 3 & Label Smoothing),
#     and Table 1 Complexity Crossover
# ============================================================================

def make_causal_mask(sz: int) -> torch.Tensor:
    """Lower-triangular 1s (1, 1, sz, sz); 0s above diagonal are masked to -inf."""
    return torch.tril(torch.ones(1, 1, sz, sz, dtype=torch.bool))


def label_smoothed_nll(logits: torch.Tensor, target: torch.Tensor, eps_ls: float = 0.1) -> torch.Tensor:
    """Section 5.4 (p. 8): Label smoothing with value eps_ls = 0.1."""
    V = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -log_probs.mean(dim=-1)
    return ((1.0 - eps_ls) * nll + eps_ls * smooth_loss).mean()


def run_e4_causal_and_transduction():
    print("\n=== E4.1: Causal Masking Prefix Invariance & Future-Leakage Negative Control (Section 3.1 & 3.2.3, pp. 3, 5) ===")
    torch.manual_seed(2017)
    model = AuthenticTransformer2017(vocab_size=16, N=2, d_model=32, d_ff=64, h=4, dropout=0.0, max_len=32)
    model.eval()
    src = torch.tensor([[2, 5, 7, 9, 3]])
    tgt_a = torch.tensor([[1, 5, 7, 9, 3]])
    tgt_b = torch.tensor([[1, 5, 7, 14, 15]])  # Mutated at future positions idx = 3, 4

    with torch.no_grad():
        mem = model.encode(src)
        mask_causal = make_causal_mask(tgt_a.size(1))
        logits_a_masked = model.decode(tgt_a, mem, tgt_mask=mask_causal)
        logits_b_masked = model.decode(tgt_b, mem, tgt_mask=mask_causal)
        diff_prefix_masked = float(torch.max(torch.abs(logits_a_masked[:, :3, :] - logits_b_masked[:, :3, :])).item())

        # Negative control: omit causal mask in decoder self-attention
        logits_a_unmasked = model.decode(tgt_a, mem, tgt_mask=None)
        logits_b_unmasked = model.decode(tgt_b, mem, tgt_mask=None)
        diff_prefix_unmasked = float(torch.max(torch.abs(logits_a_unmasked[:, :3, :] - logits_b_unmasked[:, :3, :])).item())

    print(f"  Max |logits_prefix(0..2) change| when future tokens (3..4) mutate:")
    print(f"    With causal mask (-inf for j > i):    {diff_prefix_masked:.2e} (exact zero leftward leakage!)")
    print(f"    Without causal mask (negative ctrl):  {diff_prefix_unmasked:.4f} (future leakage corrupts prefix!)")
    assert diff_prefix_masked == 0.0 and diff_prefix_unmasked > 0.1

    print("\n=== E4.2: End-to-End Sequence Transduction with Eq. 3 Warmup Schedule & Label Smoothing (Section 5.3-5.4, pp. 7-8) ===")
    # Task: Sequence reversal + Caesar shift transduction: src = [s_1, s_2, s_3, s_4] -> tgt = [BOS=1, s_4+1, s_3+1, s_2+1, s_1+1]
    rng = np.random.default_rng(2017)
    N_pairs = 128
    L_seq = 4
    src_data = rng.integers(2, 10, size=(N_pairs, L_seq))
    tgt_out = (src_data[:, ::-1] - 2 + 1) % 8 + 2  # shifted reversed tokens in 2..9
    tgt_in = np.concatenate([np.ones((N_pairs, 1), dtype=np.int64), tgt_out[:, :-1]], axis=1)

    src_t = torch.tensor(src_data, dtype=torch.long)
    tgt_in_t = torch.tensor(tgt_in, dtype=torch.long)
    tgt_out_t = torch.tensor(tgt_out, dtype=torch.long)

    def train_transducer(eps_ls: float, use_causal_mask: bool = True):
        torch.manual_seed(2017)
        tf_net = AuthenticTransformer2017(vocab_size=12, N=2, d_model=32, d_ff=64, h=4, dropout=0.0, max_len=16)
        opt = torch.optim.Adam(tf_net.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        warmup = 40
        mask = make_causal_mask(L_seq) if use_causal_mask else None

        for step in range(1, 201):
            # Equation 3 (p. 7): lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
            lrate = 0.25 * (32.0 ** (-0.5)) * min(step ** (-0.5), step * (warmup ** (-1.5)))
            for pg in opt.param_groups:
                pg["lr"] = lrate
            opt.zero_grad()
            mem = tf_net.encode(src_t)
            logits = tf_net.decode(tgt_in_t, mem, tgt_mask=mask)
            loss = label_smoothed_nll(logits, tgt_out_t, eps_ls=eps_ls)
            loss.backward()
            opt.step()

        tf_net.eval()
        with torch.no_grad():
            # 1. Teacher-forced perplexity:
            mem = tf_net.encode(src_t)
            logits_tf = tf_net.decode(tgt_in_t, mem, tgt_mask=mask)
            nll_exact = F.cross_entropy(logits_tf.reshape(-1, 12), tgt_out_t.reshape(-1))
            ppl = float(torch.exp(nll_exact).item())
            tf_acc = float((logits_tf.argmax(dim=-1) == tgt_out_t).float().mean().item()) * 100.0

            # 2. Autoregressive greedy decoding (Section 3, p. 2: "consuming the previously generated symbols as additional input"):
            ys = torch.ones((N_pairs, 1), dtype=torch.long)
            for step_idx in range(L_seq):
                cur_mask = make_causal_mask(ys.size(1))
                out_logits = tf_net.decode(ys, mem, tgt_mask=cur_mask)
                next_tok = out_logits[:, -1:, :].argmax(dim=-1)
                ys = torch.cat([ys, next_tok], dim=1)
            ar_seq_acc = float((ys[:, 1:] == tgt_out_t).all(dim=1).float().mean().item()) * 100.0
        return tf_acc, ar_seq_acc, ppl

    tf_acc_0, ar_acc_0, ppl_0 = train_transducer(eps_ls=0.0, use_causal_mask=True)
    tf_acc_ls, ar_acc_ls, ppl_ls = train_transducer(eps_ls=0.1, use_causal_mask=True)
    tf_acc_nomask, ar_acc_nomask, _ = train_transducer(eps_ls=0.1, use_causal_mask=False)

    print(f"  Causal Mask + eps_ls=0.0: teacher-forced={tf_acc_0:5.1f}%, autoregressive seq acc={ar_acc_0:5.1f}%, PPL={ppl_0:.4f}")
    print(f"  Causal Mask + eps_ls=0.1: teacher-forced={tf_acc_ls:5.1f}%, autoregressive seq acc={ar_acc_ls:5.1f}%, PPL={ppl_ls:.4f} (p.8: hurts PPL, keeps 100% acc!)")
    print(f"  NO Causal Mask (neg ctrl): teacher-forced={tf_acc_nomask:5.1f}%, autoregressive seq acc={ar_acc_nomask:5.1f}% (collapses at autoregressive inference!)")
    assert ar_acc_ls == 100.0 and ppl_ls > ppl_0 and ar_acc_nomask < 10.0

    print("\n=== E4.3: Table 1 Per-Layer Complexity Crossover — Self-Attention O(n^2 d) vs Recurrent O(n d^2) (Section 4, p. 6) ===")
    d = 512
    print("  Seq len n | Self-Attn core MACs (2 n^2 d) | Recurrent core MACs (n d^2) | Ratio Self-Attn / Recurrent")
    for n in [16, 32, 64, 128, 256, 512, 1024]:
        sa_macs = 2 * (n ** 2) * d
        rec_macs = n * (d ** 2)
        print(f"       {n:4d} |                   {sa_macs:11,d} |                 {rec_macs:11,d} |                      {sa_macs / rec_macs:6.3f}x")
    assert 2 * (32 ** 2) * d < 32 * (d ** 2) and 2 * (256 ** 2) * d == 256 * (d ** 2)


if __name__ == "__main__":
    run_e2_attention_and_params()
    run_e3_positional_and_multihead()
    run_e4_causal_and_transduction()
