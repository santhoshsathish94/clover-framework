#!/usr/bin/env python3
"""
Precomputed Runtime Stage Engine for Liquid AI LFM3-1.7B
(`clover-ai-system/src/models/lfm3-1.7b/lfm3_precomputed_engine.py`).

Executes all 24 tri-hybrid layers of `LiquidAI/LFM3-1.7B` directly from the PRUNED `LFM3TRK1` trunks
(`trunk-0..23`) and composed `operator.bin` streams (`attention.bin` completely deleted from all 24 trunks!):
  - Stage 1: Tied `global-0/embed_tokens.bf16.bin` (`[128000, 2048]`)
  - Stage 2: Precomputed `global-0/rope_inv_freq.bin` (`full_attention` zero-tail partial RoPE `0.5` + `sliding_attention` full RoPE `1.0`)
  - Stage 3: All 24 Decoder Layers (`trunk-0` .. `trunk-23`):
      * 13 Gated ShortConv layers (`kind = 0`): reads `operator_norm`, `in_proj`, `conv.weight` from `operator.bin`,
        and `post_operator_norm`, `out_proj`, `ffn_norm`, `post_ffn_norm`, `w1`, `w3`, `w2` from pruned `common.bin` + `ffn.bin`.
      * 5 Sliding-Window GQA layers (`kind = 1`, `sliding_window = 256`) & 6 Full GQA layers (`kind = 2`):
        reads `operator_norm`, `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `q_norm`, `k_norm` from `operator.bin`,
        and `post_operator_norm`, `o_proj`, `ffn_norm`, `post_ffn_norm`, `w1`, `w3`, `w2` from pruned `common.bin` + `ffn.bin`.
      * Also supports Clover's 4-bit planar group-64 SwiGLU quantization (`use_4bit_ffn=True`).
  - Stage 4 & 5: `global-0/embedding_norm.bf16.bin` + tied `global-0/embed_tokens.bf16.bin` (`lm_head`).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
NEMO_DIR = SCRIPT_DIR.parent / "nemotron-3.5-super-vl"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(NEMO_DIR) not in sys.path:
    sys.path.insert(0, str(NEMO_DIR))

from prepare_lfm3_datasets import (
    DTYPE_BF16,
    DTYPE_FP32,
    GROUP_NAMES,
    HEADER_FMT,
    HEADER_SIZE,
    OPERATOR_MAGIC,
    RECORD_FMT,
    RECORD_SIZE,
    ROPE_MAGIC,
    TRUNK_MAGIC,
)
from prepare_nemotron_datasets import dequantize_matrix_4bit_group64, quantize_matrix_4bit_group64


def bytes_to_tensor(buf: bytes | memoryview, dtype_code: int, shape: tuple[int, ...]) -> torch.Tensor:
    if dtype_code == DTYPE_BF16:
        arr = np.frombuffer(buf, dtype=np.uint16).copy()
        return torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
    if dtype_code == DTYPE_FP32:
        arr = np.frombuffer(buf, dtype=np.float32).copy()
        return torch.from_numpy(arr).reshape(shape)
    raise ValueError(f"Unsupported dtype_code={dtype_code}")


def rms_norm_op(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    in_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    var = x_fp32.pow(2).mean(-1, keepdim=True)
    return weight * (x_fp32 * torch.rsqrt(var + eps)).to(in_dtype)


def rotate_half_op(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_op(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_u = cos.unsqueeze(1).float()
    sin_u = sin.unsqueeze(1).float()
    q_embed = (q.float() * cos_u) + (rotate_half_op(q.float()) * sin_u)
    k_embed = (k.float() * cos_u) + (rotate_half_op(k.float()) * sin_u)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class Lfm3PrunedTrunkReader:
    """Reads a pruned `LFM3TRK1` trunk folder (`index.bin`, `common.bin`, `ffn.bin`, and `operator.bin`)."""

    def __init__(self, trunk_dir: Path):
        self.trunk_dir = trunk_dir
        idx_bytes = (trunk_dir / "index.bin").read_bytes()
        magic, version, self.layer_id, self.layer_kind, record_count, _, g0_len, _, g1_len, _, g2_len = struct.unpack(
            HEADER_FMT, idx_bytes[:HEADER_SIZE]
        )
        assert magic == TRUNK_MAGIC and version == 1
        self.records: dict[int, tuple[int, int, int, int, int, int]] = {}
        pos = HEADER_SIZE
        for _ in range(record_count):
            rec_id, grp_id, dtype_code, rows, offset, length, cols = struct.unpack(
                RECORD_FMT, idx_bytes[pos : pos + RECORD_SIZE]
            )
            self.records[rec_id] = (grp_id, dtype_code, rows, cols, offset, length)
            pos += RECORD_SIZE

        self.groups: dict[int, bytes] = {}
        for grp_id, grp_name in enumerate(GROUP_NAMES):
            p = trunk_dir / grp_name
            if p.exists():
                self.groups[grp_id] = p.read_bytes()

        op_p = trunk_dir / "operator.bin"
        self.operator_bytes = op_p.read_bytes() if op_p.exists() else None

    def get_record_tensor(self, rec_id: int, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        grp_id, dtype_code, rows, cols, offset, length = self.records[rec_id]
        if length == 0:
            raise RuntimeError(f"Record {rec_id} in layer {self.layer_id} was pruned (length=0); read via operator.bin!")
        raw = memoryview(self.groups[grp_id])[offset : offset + length]
        target_shape = shape if shape is not None else ((rows,) if cols == 1 else (rows, cols))
        return bytes_to_tensor(raw, dtype_code, target_shape)


@torch.no_grad()
def execute_lfm3_layer_from_precomputed(
    trunk: Lfm3PrunedTrunkReader,
    hidden_states: torch.Tensor,
    cos_full: torch.Tensor,
    sin_full: torch.Tensor,
    cos_sliding: torch.Tensor,
    sin_sliding: torch.Tensor,
    sliding_window: int = 256,
    use_4bit_ffn: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Executes a single LFM3-1.7B layer (`kind = 0` ShortConv, `kind = 1` Sliding GQA, or `kind = 2` Full GQA)
    from `operator.bin` + pruned `common.bin` + `ffn.bin`.
    """
    bsz, seq_len, _ = hidden_states.shape
    op_buf = trunk.operator_bytes
    assert op_buf is not None and op_buf[:8] == OPERATOR_MAGIC
    mv = memoryview(op_buf)

    if trunk.layer_kind == 0:
        # Gated ShortConv layer (`kind = 0`)
        _, kind, layer_id, len_op_norm, len_in_proj, len_conv_w = struct.unpack_from("<8sIIIII", op_buf, 0)
        assert kind == 0
        cur = 64
        op_norm_w = bytes_to_tensor(mv[cur : cur + len_op_norm], DTYPE_BF16, (2048,)); cur += len_op_norm
        in_proj_w = bytes_to_tensor(mv[cur : cur + len_in_proj], DTYPE_BF16, (6144, 2048)); cur += len_in_proj
        conv_w = bytes_to_tensor(mv[cur : cur + len_conv_w], DTYPE_BF16, (2048, 1, 7)); cur += len_conv_w

        out_proj_w = trunk.get_record_tensor(2, (2048, 2048))

        op_in = rms_norm_op(hidden_states, op_norm_w, eps=eps)
        bcx = F.linear(op_in, in_proj_w).transpose(-1, -2)
        B_gate, C_gate, x_branch = bcx.chunk(3, dim=-2)
        pre_gated = B_gate * x_branch
        conv_raw = F.conv1d(pre_gated, conv_w, padding=6, groups=2048)[..., :seq_len]
        post_gated = (C_gate * conv_raw).transpose(-1, -2).contiguous()
        op_out_raw = F.linear(post_gated, out_proj_w)
    else:
        # Sliding (`kind = 1`) or Full (`kind = 2`) GQA Attention with per-head Sigmoid Gate
        _, kind, layer_id, len_op_norm, len_q, len_k, len_v, len_gate, len_qn, len_kn = struct.unpack_from(
            "<8sIIIIIIIII", op_buf, 0
        )
        assert kind in (1, 2)
        cur = 64
        op_norm_w = bytes_to_tensor(mv[cur : cur + len_op_norm], DTYPE_BF16, (2048,)); cur += len_op_norm
        q_w = bytes_to_tensor(mv[cur : cur + len_q], DTYPE_BF16, (2048, 2048)); cur += len_q
        k_w = bytes_to_tensor(mv[cur : cur + len_k], DTYPE_BF16, (512, 2048)); cur += len_k
        v_w = bytes_to_tensor(mv[cur : cur + len_v], DTYPE_BF16, (512, 2048)); cur += len_v
        gate_w = bytes_to_tensor(mv[cur : cur + len_gate], DTYPE_BF16, (16, 2048)); cur += len_gate
        qn_w = bytes_to_tensor(mv[cur : cur + len_qn], DTYPE_BF16, (128,)); cur += len_qn
        kn_w = bytes_to_tensor(mv[cur : cur + len_kn], DTYPE_BF16, (128,)); cur += len_kn

        o_proj_w = trunk.get_record_tensor(2, (2048, 2048))

        op_in = rms_norm_op(hidden_states, op_norm_w, eps=eps)
        q = F.linear(op_in, q_w).view(bsz, seq_len, 16, 128)
        k = F.linear(op_in, k_w).view(bsz, seq_len, 4, 128)
        v = F.linear(op_in, v_w).view(bsz, seq_len, 4, 128)

        q = rms_norm_op(q, qn_w, eps=eps).transpose(1, 2)
        k = rms_norm_op(k, kn_w, eps=eps).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = (cos_sliding, sin_sliding) if kind == 1 else (cos_full, sin_full)
        q, k = apply_rope_op(q, k, cos, sin)

        k_exp = k[:, :, None, :, :].expand(bsz, 4, 4, seq_len, 128).reshape(bsz, 16, seq_len, 128)
        v_exp = v[:, :, None, :, :].expand(bsz, 4, 4, seq_len, 128).reshape(bsz, 16, seq_len, 128)

        scores = torch.matmul(q, k_exp.transpose(2, 3)) * (128.0 ** -0.5)
        row_idx = torch.arange(seq_len, device=hidden_states.device)[:, None]
        col_idx = torch.arange(seq_len, device=hidden_states.device)[None, :]
        mask = col_idx <= row_idx
        if kind == 1:
            mask = mask & ((row_idx - col_idx) < sliding_window)
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_val = torch.matmul(probs, v_exp).transpose(1, 2).contiguous()

        head_gate = torch.sigmoid(F.linear(op_in, gate_w)).unsqueeze(-1)
        gated_attn = (attn_val * head_gate).reshape(bsz, seq_len, 2048)
        op_out_raw = F.linear(gated_attn, o_proj_w)

    # Sandwich post_operator_norm + Residual 1
    post_op_norm_w = trunk.get_record_tensor(1, (2048,))
    h_mid = hidden_states + rms_norm_op(op_out_raw, post_op_norm_w, eps=eps)

    # SwiGLU FFN + Sandwich post_ffn_norm + Residual 2
    ffn_norm_w = trunk.get_record_tensor(3, (2048,))
    post_ffn_norm_w = trunk.get_record_tensor(4, (2048,))
    w1_w = trunk.get_record_tensor(10, (7680, 2048))
    w3_w = trunk.get_record_tensor(11, (7680, 2048))
    w2_w = trunk.get_record_tensor(12, (2048, 7680))

    if use_4bit_ffn:
        n1, s1, c1 = quantize_matrix_4bit_group64(w1_w)
        n3, s3, c3 = quantize_matrix_4bit_group64(w3_w)
        n2, s2, c2 = quantize_matrix_4bit_group64(w2_w)
        w1_w = dequantize_matrix_4bit_group64(n1, s1, c1, 7680, 2048).to(torch.bfloat16)
        w3_w = dequantize_matrix_4bit_group64(n3, s3, c3, 7680, 2048).to(torch.bfloat16)
        w2_w = dequantize_matrix_4bit_group64(n2, s2, c2, 2048, 7680).to(torch.bfloat16)

    ffn_in = rms_norm_op(h_mid, ffn_norm_w, eps=eps)
    swiglu = F.silu(F.linear(ffn_in, w1_w)) * F.linear(ffn_in, w3_w)
    ffn_raw = F.linear(swiglu, w2_w)
    h_out = h_mid + rms_norm_op(ffn_raw, post_ffn_norm_w, eps=eps)
    return h_out


@torch.no_grad()
def execute_full_24_layer_lfm3_from_precomputed(
    pruned_computed_dir: Path,
    input_ids: torch.Tensor,
    pad_token_id: int = 124893,
    use_4bit_ffn_on_layer0: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """
    Executes the entire 24-layer LFM3-1.7B model from pruned `trunk-0..23` + `operator.bin` + `global-0/`
    and returns `(logits, per_layer_outputs, final_hidden_states)`.
    """
    bsz, seq_len = input_ids.shape
    global_dir = pruned_computed_dir / "global-0"
    emb_w = bytes_to_tensor((global_dir / "embed_tokens.bf16.bin").read_bytes(), DTYPE_BF16, (128000, 2048))
    emb_norm_w = bytes_to_tensor((global_dir / "embedding_norm.bf16.bin").read_bytes(), DTYPE_BF16, (2048,))

    rope_buf = (global_dir / "rope_inv_freq.bin").read_bytes()
    magic, head_dim, len_full, len_slid = struct.unpack_from("<8sIII", rope_buf, 0)
    assert magic == ROPE_MAGIC
    mv_r = memoryview(rope_buf)
    inv_full = bytes_to_tensor(mv_r[20 : 20 + len_full], DTYPE_FP32, (head_dim // 2,))
    inv_slid = bytes_to_tensor(mv_r[20 + len_full : 20 + len_full + len_slid], DTYPE_FP32, (head_dim // 2,))

    pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).float()
    freqs_full = pos_ids[..., None] * inv_full
    emb_full = torch.cat([freqs_full, freqs_full], dim=-1)
    cos_full, sin_full = emb_full.cos().to(torch.bfloat16), emb_full.sin().to(torch.bfloat16)

    freqs_slid = pos_ids[..., None] * inv_slid
    emb_slid = torch.cat([freqs_slid, freqs_slid], dim=-1)
    cos_sliding, sin_sliding = emb_slid.cos().to(torch.bfloat16), emb_slid.sin().to(torch.bfloat16)

    h = F.embedding(input_ids, emb_w, padding_idx=pad_token_id)
    per_layer_outputs: list[torch.Tensor] = []

    for idx in range(24):
        reader = Lfm3PrunedTrunkReader(pruned_computed_dir / f"trunk-{idx}")
        h = execute_lfm3_layer_from_precomputed(
            reader,
            h,
            cos_full,
            sin_full,
            cos_sliding,
            sin_sliding,
            use_4bit_ffn=(use_4bit_ffn_on_layer0 and idx == 0),
        )
        per_layer_outputs.append(h)

    h_final = rms_norm_op(h, emb_norm_w, eps=1e-5)
    # Tied LM Head (`embed_tokens.bf16.bin`)
    logits = F.linear(h_final, emb_w).float()
    return logits, per_layer_outputs, h_final
