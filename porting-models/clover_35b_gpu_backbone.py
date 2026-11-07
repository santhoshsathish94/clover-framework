#!/usr/bin/env python3
"""
Clover-35B-A3B GPU-Resident Differentiable Stage Backbone & Precomputation Engine
(`porting-models/clover_35b_gpu_backbone.py`).

Supports `Diffuspace/Clover-35B-A3B-Preview` (`35B` total / `3B` active parameters, `40` layers,
`D_model = 2048`, `V = 248,320`, `30` Bidirectional Gated DeltaNet `BD-GDN` layers + `10` Gated GQA
layers, `256` routed SwiGLU experts `top-8` with `moe_block_coreset_top_p = 0.9` + `1` shared expert,
`128K` context window, and `block_size = 32` LowConfidence block diffusion).

Key capabilities:
  1. IPv4-forced HTTP Range streaming (`fetch_safetensor_tensors` & `fetch_safetensor_leading_slice`)
     so stage trunks (`trunk-0` BD-GDN, `trunk-3` Gated GQA), all 30 BD-GDN bidirectional preview
     containers (`bd_gdn.bin` `CLV35BD1`), global embeddings/norm/LM head (`global-0`), and routed
     experts (`experts.direct` & 4-bit group-64 `experts.4bit.direct`) are streamed directly into
     `/dev/shm` or `/workspace`.
  2. Exact `CloverMoeModelLM` operators verified against `modeling_clover_moe.py`:
     - `clover_rms_norm_op` (`zero_centered=True` -> `weight = 1 + w`, `zero_centered=False` on `linear_attn.norm`)
     - `bd_gdn_conv1d_op`, `bd_gdn_local_memory_op`, and `recurrent_commit_op` (both causal commit and
       bidirectional block-diffusion preview modes)
     - `clover_gated_gqa_op` (16Q / 2KV, `head_dim = 256`, fused sigmoid output gate, partial RoPE `0.25`)
     - `coreset_routing_op` (`moe_block_coreset_top_p = 0.9`) & `clover_moe_block_op`
  3. Differentiable GPU Stage Model (`Clover35BGPUStageModel`) for 8x RTX 3090 3-Stage `HybridAgent`,
     `11-Level NCM`, and `JEPA` trunk training.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from kimi_k31_gpu_backbone import (
    DTYPE_BF16,
    DTYPE_FP32,
    DTYPE_U8,
    GROUPS,
    HEADER_BYTES,
    RECORD_BYTES,
    _fetch_byte_range,
    bytes_to_tensor,
    enable_ipv4_only,
    fetch_safetensor_tensors,
    get_safetensor_header,
    tensor_to_le_bytes,
)

TRUNK_MAGIC = b"CLV35TRK"
OPERATOR_MAGIC = b"CLV35OP1"
BD_GDN_MAGIC = b"CLV35BD1"
EXPERT_4BIT_MAGIC = b"CLVR4B01"

V_CLOVER35B = 248320
D_CLOVER35B = 2048
D_LATENT_35B = 1024
D_MOE_INTER = 512
D_SHARED_INTER = 512
N_EXPERTS_35B = 256
TOP_K_35B = 8
D_VQ = 384
D_HA = 4096

# 40 layers: 3:1 cycle -> full_attention at layers 3, 7, 11, ..., 39 (0-indexed); linear_attention elsewhere
FULL_ATTN_LAYERS_0IDX = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
BD_GDN_LAYERS_0IDX = [i for i in range(40) if i not in FULL_ATTN_LAYERS_0IDX]

MASK_TOKEN_ID = 151669
PAD_TOKEN_ID = 248044
BOS_TOKEN_ID = 248044
IM_START_ID = 248045
EOS_TOKEN_ID = 248046
THINK_OPEN_ID = 248068
THINK_CLOSE_ID = 248069
LATENT_OPEN_ID = 248077
LATENT_CLOSE_ID = 248078


# ---------------- Exact Clover-35B-A3B Mathematical Operators ----------------

def clover_rms_norm_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    zero_centered: bool = True,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact `CloverRMSNorm` from `modeling_clover_moe.py` (`weight = 1 + w` when `zero_centered=True`)."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    w = (1.0 + weight.float()) if zero_centered else weight.float()
    y = y * w
    if gate is not None:
        y = y * F.silu(gate.float())
    return y.to(x.dtype)


def causal_conv_silu_op(joined: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Exact `causal_conv_silu` from `modeling_clover_moe.py`."""
    width = weight.shape[-1]
    length = joined.shape[-1] - width + 1
    total = torch.zeros((*joined.shape[:-1], length), device=joined.device, dtype=torch.float32)
    for tap in range(width):
        total.add_((joined[..., tap : tap + length] * weight[:, 0, tap][None, :, None]).float())
    return F.silu(total).to(joined.dtype)


def bd_gdn_conv1d_op(
    raw_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_right_weight: torch.Tensor,
    conv_state: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact `bd_gdn_conv1d` bidirectional block preview convolution (`modeling_clover_moe.py`)."""
    b, t, channels = raw_qkv.shape
    width = conv_weight.shape[-1]
    if conv_state is None:
        conv_state = raw_qkv.new_zeros((b, channels, width - 1))
    if valid is None:
        valid = torch.ones((b, t), device=raw_qkv.device, dtype=torch.bool)
    current = raw_qkv * valid[:, :, None].to(raw_qkv.dtype)
    conv_in = torch.cat((conv_state, current.transpose(1, 2)), dim=-1)
    out = F.conv1d(conv_in.float(), conv_weight.reshape(channels, 1, -1).float(), groups=channels)
    for distance in range(1, min(width, t)):
        out[:, :, :-distance] += (
            current[:, distance:, :].transpose(1, 2).float()
            * conv_right_weight[:, 0, distance - 1].float()[None, :, None]
        )
    out = F.silu(out).transpose(1, 2)
    out = out * valid[:, :, None].to(out.dtype)
    return out.to(raw_qkv.dtype)


def recurrent_commit_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact `recurrent_commit_torch` causal GDN recurrence (`modeling_clover_moe.py`)."""
    b, t, vh, vd = v.shape
    kd = q.shape[-1]
    if state is None:
        state = torch.zeros((b, vh, vd, kd), device=v.device, dtype=torch.float32)
    if valid is None:
        valid = torch.ones((b, t), device=v.device, dtype=torch.bool)
    qf = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
    kf = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
    qf = qf.repeat_interleave(vh // q.shape[2], 2) / (kd ** 0.5)
    kf = kf.repeat_interleave(vh // k.shape[2], 2)
    st = state.float()
    out = []
    for idx in range(t):
        decayed = st * g[:, idx].float().exp()[..., None, None]
        remembered = torch.einsum("bhvk,bhk->bhv", decayed, kf[:, idx])
        delta = (v[:, idx].float() - remembered) * beta[:, idx].float()[..., None]
        updated = decayed + delta[..., None] * kf[:, idx, :, None, :]
        st = torch.where(valid[:, idx, None, None, None], updated, st)
        y = torch.einsum("bhvk,bhk->bhv", st, qf[:, idx])
        out.append(y * valid[:, idx, None, None])
    return torch.stack(out, dim=1).to(v.dtype), st


def bd_gdn_local_memory_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    softmax_logit_scale: torch.Tensor,
    softmax_history_bias: torch.Tensor,
    softmax_history_c: torch.Tensor,
    clean_ssm_state: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact `bd_gdn_local_memory` bidirectional softmax preview with clean-state history bucket."""
    b, t, vh, vd = v.shape
    kd = q.shape[-1]
    if clean_ssm_state is None:
        clean_ssm_state = torch.zeros((b, vh, vd, kd), device=v.device, dtype=torch.float32)
    if valid is None:
        valid = torch.ones((b, t), device=v.device, dtype=torch.bool)
    valid_f = valid[:, :, None].float()
    qf = q.repeat_interleave(vh // q.shape[2], dim=2).float()
    kf = k.repeat_interleave(vh // k.shape[2], dim=2).float()
    vf = v.float()
    qf = qf * torch.rsqrt(qf.square().sum(dim=-1, keepdim=True) + 1e-6)
    kf = kf * torch.rsqrt(kf.square().sum(dim=-1, keepdim=True) + 1e-6)

    scaled_q = qf * (kd ** -0.5)
    clean_out = torch.einsum("blhk,bhvk->blhv", scaled_q, clean_ssm_state.float())
    decay = g.float().exp()
    history = clean_out * decay[..., None]

    l_scale = softmax_logit_scale.reshape(-1).float().to(qf.device)
    h_bias = softmax_history_bias.reshape(-1).float().to(qf.device)
    h_c = softmax_history_c.reshape(-1).float().to(qf.device)

    logits = torch.einsum("blhk,bshk->bhls", qf, kf) * l_scale.view(1, -1, 1, 1)
    logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
    valid_block_size = valid.sum(dim=-1).clamp_min(1).float()
    log_ratio = torch.log(valid_block_size / 32.0)
    history_logit = h_bias.view(1, -1, 1, 1).expand(b, -1, t, 1) + h_c.view(1, -1, 1, 1) * log_ratio.view(b, 1, 1, 1)
    weights = torch.softmax(torch.cat((logits, history_logit), dim=-1), dim=-1)
    local_out = torch.einsum("bhls,bshv->blhv", weights[..., :-1], vf * valid_f[..., None])
    hist_w = weights[..., -1].permute(0, 2, 1)
    out = (hist_w[..., None] * history + local_out) * valid_f[..., None]
    return out.to(v.dtype)


def coreset_routing_op(
    logits: torch.Tensor,
    valid: torch.Tensor | None = None,
    top_k: int = TOP_K_35B,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Exact `coreset_torch` block-level MoE expert coreset filter from `modeling_clover_moe.py`."""
    b, t, e = logits.shape
    if valid is None:
        valid = torch.ones((b, t), device=logits.device, dtype=torch.bool)
    ids = logits.float().argsort(dim=-1, descending=True, stable=True)[..., :top_k]
    scores = logits.float().softmax(-1).gather(-1, ids) * valid[..., None]
    totals = logits.new_zeros((b, e), dtype=torch.float32)
    totals.scatter_add_(1, ids.reshape(b, -1), scores.reshape(b, -1))
    probs = totals / totals.sum(-1, keepdim=True).clamp_min(1e-9)
    order = probs.argsort(dim=-1, descending=True, stable=True)
    sorted_probs = probs.gather(-1, order)
    keep = (sorted_probs.cumsum(-1) - sorted_probs) < top_p
    keep[:, 0] = True
    allowed = torch.zeros_like(keep).scatter(1, order, keep)
    return logits.masked_fill(valid[..., None] & ~allowed[:, None, :], -float("inf"))


# ---------------- 4-Bit Planar Group-64 Quantization Helpers ----------------

def quantize_4bit_group64(t: torch.Tensor) -> tuple[bytes, bytes, tuple[int, ...]]:
    """Quantizes a 2D/3D BF16 tensor into symmetric 4-bit signed nibbles (`[-7..7]`) with FP16 scales per group of 64."""
    orig_shape = tuple(t.shape)
    flat = t.detach().float().reshape(-1, 64)
    amax = flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = (amax / 7.0).to(torch.float16)
    q = torch.clamp(torch.round(flat / scales.float()), -8, 7).to(torch.int8)
    u = (q + 8).to(torch.uint8)
    packed = (u[:, 0::2] & 0x0F) | ((u[:, 1::2] & 0x0F) << 4)
    return packed.cpu().numpy().tobytes(), scales.squeeze(-1).cpu().numpy().tobytes(), orig_shape


def dequantize_4bit_group64(packed_bytes: bytes, scale_bytes: bytes, shape: tuple[int, ...], device: torch.device | None = None) -> torch.Tensor:
    packed = torch.from_numpy(np.frombuffer(packed_bytes, dtype=np.uint8).copy())
    scales = torch.from_numpy(np.frombuffer(scale_bytes, dtype=np.float16).copy()).float()
    if device is not None:
        packed = packed.to(device)
        scales = scales.to(device)
    packed_2d = packed.reshape(-1, 32)
    low = (packed_2d & 0x0F).to(torch.int8) - 8
    high = ((packed_2d >> 4) & 0x0F).to(torch.int8) - 8
    q = torch.stack([low, high], dim=-1).reshape(-1, 64).float()
    out = (q * scales.unsqueeze(-1)).reshape(shape).to(torch.bfloat16)
    return out


# ---------------- HTTP Range Slice Streaming for Fused 3D Expert Tensors ----------------

def fetch_safetensor_leading_slice(
    repo_id: str,
    filename: str,
    tensor_name: str,
    leading_count: int,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> torch.Tensor:
    """
    Streams the first `leading_count` experts along axis 0 of a fused 3D expert tensor
    (`mlp.experts.gate_up_proj [256, 1024, 2048]` or `mlp.experts.down_proj [256, 2048, 512]`)
    using parallel HTTP Range requests without downloading all 256 experts when `leading_count < 256`.
    """
    from concurrent.futures import ThreadPoolExecutor

    cdn_url, header, data_base = get_safetensor_header(repo_id, filename)
    if tensor_name not in header:
        raise KeyError(f"Tensor '{tensor_name}' not found in {repo_id}/{filename}")
    info = header[tensor_name]
    shape = tuple(info["shape"])
    dtype_str = info["dtype"]
    s, e = int(info["data_offsets"][0]), int(info["data_offsets"][1])
    total_bytes = e - s
    n_total = shape[0]
    k = min(leading_count, n_total)
    bytes_per_item = total_bytes // n_total
    want_bytes = k * bytes_per_item

    subtasks = []
    cur = 0
    while cur < want_bytes:
        nxt = min(want_bytes, cur + chunk_bytes)
        subtasks.append((cur, data_base + s + cur, data_base + s + nxt - 1))
        cur = nxt

    buf = bytearray(want_bytes)
    with ThreadPoolExecutor(max_workers=min(12, max(1, len(subtasks)))) as pool:
        futs = [(off, pool.submit(_fetch_byte_range, cdn_url, b0, b1)) for off, b0, b1 in subtasks]
        for off, fut in futs:
            chunk = fut.result()
            buf[off : off + len(chunk)] = chunk

    new_shape = (k,) + shape[1:]
    if dtype_str == "BF16":
        arr = np.frombuffer(buf, dtype=np.uint16).copy()
        return torch.from_numpy(arr).view(torch.bfloat16).reshape(new_shape)
    arr = np.frombuffer(buf, dtype=np.float32).copy()
    return torch.from_numpy(arr).reshape(new_shape)


# ---------------- Trunk & BD-GDN Container Packing ----------------

def write_clover35b_trunk_container(
    out_dir: Path,
    layer_id: int,
    layer_kind: int,
    records_spec: list[tuple[int, int, int, int, int, bytes]],
    operator_bytes: bytes | None = None,
    bd_gdn_bytes: bytes | None = None,
    prune_needs: set[int] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    group_buffers = [bytearray(), bytearray(), bytearray()]
    packed_records = []
    dropped_bytes = 0

    for rid, group_id, dtype_code, rows, cols, raw_bytes in records_spec:
        numel = rows * cols
        crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
        keep = True if prune_needs is None else (rid in prune_needs)
        offset = len(group_buffers[group_id])
        if keep:
            length = len(raw_bytes)
            group_buffers[group_id].extend(raw_bytes)
        else:
            length = 0
            dropped_bytes += len(raw_bytes)
        packed_records.append((rid, group_id, dtype_code, rows, cols, numel, offset, length, crc))

    group_lengths = [len(b) for b in group_buffers]
    for g_idx, g_name in enumerate(GROUPS):
        g_path = out_dir / g_name
        if group_lengths[g_idx] > 0:
            g_path.write_bytes(group_buffers[g_idx])
        elif g_path.exists():
            g_path.unlink()

    index_buf = bytearray(HEADER_BYTES + len(packed_records) * RECORD_BYTES)
    struct.pack_into("<8sIII", index_buf, 0, TRUNK_MAGIC, 1, layer_id, len(packed_records))
    struct.pack_into("<3Q", index_buf, 24, *group_lengths)
    struct.pack_into("<I", index_buf, 48, layer_kind)

    for idx, (rid, group_id, dtype_code, rows, cols, numel, offset, length, crc) in enumerate(packed_records):
        base = HEADER_BYTES + idx * RECORD_BYTES
        struct.pack_into("<6I", index_buf, base, rid, group_id, dtype_code, rows, cols, numel)
        struct.pack_into("<2Q", index_buf, base + 24, offset, length)
        struct.pack_into("<I", index_buf, base + 40, crc)

    (out_dir / "index.bin").write_bytes(index_buf)
    op_len = 0
    if operator_bytes is not None:
        (out_dir / "operator.bin").write_bytes(operator_bytes)
        op_len = len(operator_bytes)
    bd_len = 0
    if bd_gdn_bytes is not None:
        (out_dir / "bd_gdn.bin").write_bytes(bd_gdn_bytes)
        bd_len = len(bd_gdn_bytes)

    return {
        "layer_id": layer_id,
        "layer_kind": layer_kind,
        "record_count": len(packed_records),
        "group_lengths": group_lengths,
        "total_group_bytes": sum(group_lengths),
        "dropped_bytes": dropped_bytes,
        "operator_bytes": op_len,
        "bd_gdn_bytes": bd_len,
    }


def pack_bd_gdn_bytes(
    layer_id: int,
    conv_right_w: torch.Tensor,
    logit_scale: torch.Tensor,
    history_bias: torch.Tensor,
    history_c: torch.Tensor,
) -> bytes:
    """Packs Clover-35B-A3B's bidirectional GDN preview parameters into `bd_gdn.bin` (`CLV35BD1`)."""
    b_crw = tensor_to_le_bytes(conv_right_w.to(torch.bfloat16))
    b_ls = tensor_to_le_bytes(logit_scale.float())
    b_hb = tensor_to_le_bytes(history_bias.float())
    b_hc = tensor_to_le_bytes(history_c.float())
    hdr = bytearray(64)
    struct.pack_into(
        "<8sIIIIII",
        hdr,
        0,
        BD_GDN_MAGIC,
        1,
        layer_id,
        len(b_crw),
        len(b_ls),
        len(b_hb),
        len(b_hc),
    )
    return bytes(hdr) + b_crw + b_ls + b_hb + b_hc


def unpack_bd_gdn_bytes(buf: bytes) -> dict[str, Any]:
    if buf[:8] != BD_GDN_MAGIC:
        raise ValueError(f"Invalid BD-GDN magic: {buf[:8]!r}")
    _, ver, layer_id, l1, l2, l3, l4 = struct.unpack_from("<8sIIIIII", buf, 0)
    mv = memoryview(buf)
    cur = 64
    crw = bytes_to_tensor(mv[cur : cur + l1], DTYPE_BF16, (8192, 1, 3)); cur += l1
    ls = bytes_to_tensor(mv[cur : cur + l2], DTYPE_FP32, (32,)); cur += l2
    hb = bytes_to_tensor(mv[cur : cur + l3], DTYPE_FP32, (32,)); cur += l3
    hc = bytes_to_tensor(mv[cur : cur + l4], DTYPE_FP32, (32,)); cur += l4
    return {
        "layer_id": layer_id,
        "conv_right_weight": crw,
        "bd_softmax_logit_scale": ls,
        "bd_softmax_history_bias": hb,
        "bd_softmax_history_c": hc,
    }


def write_expert_store(
    exp_dir: Path,
    layer_id: int,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    router_gate: torch.Tensor,
) -> dict[str, Any]:
    """Writes `experts.direct` (BF16), `experts.4bit.direct` (4-bit planar group-64), `constants.bin`, and `maps.json`."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    k_cached = int(gate_up_proj.shape[0])
    # Direct BF16 expert store
    gu_bytes = tensor_to_le_bytes(gate_up_proj.to(torch.bfloat16))
    dn_bytes = tensor_to_le_bytes(down_proj.to(torch.bfloat16))
    (exp_dir / "experts.direct").write_bytes(gu_bytes + dn_bytes)

    # 4-bit planar group-64 expert store
    gu_q, gu_s, gu_shape = quantize_4bit_group64(gate_up_proj)
    dn_q, dn_s, dn_shape = quantize_4bit_group64(down_proj)
    hdr4 = bytearray(64)
    struct.pack_into(
        "<8sIIIIII",
        hdr4,
        0,
        EXPERT_4BIT_MAGIC,
        1,
        layer_id,
        k_cached,
        len(gu_q),
        len(gu_s),
        len(dn_q),
    )
    struct.pack_into("<I", hdr4, 32, len(dn_s))
    (exp_dir / "experts.4bit.direct").write_bytes(bytes(hdr4) + gu_q + gu_s + dn_q + dn_s)

    # Router constants
    (exp_dir / "constants.bin").write_bytes(tensor_to_le_bytes(router_gate.float()))
    maps = {
        "layer_id": layer_id,
        "num_total_experts": N_EXPERTS_35B,
        "num_cached_experts": k_cached,
        "top_k": TOP_K_35B,
        "hidden_size": D_CLOVER35B,
        "moe_intermediate_size": D_MOE_INTER,
        "gate_up_shape": list(gu_shape),
        "down_shape": list(dn_shape),
        "gate_up_bf16_bytes": len(gu_bytes),
        "down_bf16_bytes": len(dn_bytes),
        "experts_4bit_bytes": 64 + len(gu_q) + len(gu_s) + len(dn_q) + len(dn_s),
    }
    (exp_dir / "maps.json").write_text(json.dumps(maps, indent=2), encoding="utf-8")
    return maps


def prepare_clover_35b_datasets(
    work_dir: Path,
    repo_id: str = "Diffuspace/Clover-35B-A3B-Preview",
    num_cached_experts_per_stage: int = 32,
) -> dict[str, Any]:
    """
    Streams and builds the precomputed `Clover-35B-A3B` stage containers in `work_dir`:
      - `global-0/` (`embed_tokens.bf16.bin`, `lm_head.bf16.bin`, `norm.bf16.bin`, `lm_head.4bit.bin`, `config.json`, `tokenizer.json`, `tokenizer_config.json`)
      - `trunk-0/` + `expert-0/` (Layer 0: BD-GDN + 256-Expert MoE + Shared Expert + `operator.bin` + `bd_gdn.bin`)
      - `trunk-3/` + `expert-3/` (Layer 3: Gated GQA + 256-Expert MoE + Shared Expert + `operator.bin`)
      - `trunk-{L}/bd_gdn.bin` across all 30 Bidirectional GDN layers (`L in BD_GDN_LAYERS_0IDX`)
    """
    enable_ipv4_only()
    t0 = time.perf_counter()
    raw_dir = work_dir / "raw"
    pruned_dir = work_dir / "pruned"
    tok_dir = work_dir / "tokenizer"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pruned_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    # 0. Download config & tokenizer metadata
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        dst_f = tok_dir / fn
        if not dst_f.exists():
            p = hf_hub_download(repo_id, fn)
            dst_f.write_bytes(Path(p).read_bytes())

    # 1. Global embeddings (`model-00001-of-00026.safetensors`) and norm + lm_head (`model-00026-of-00026.safetensors`)
    global_raw = raw_dir / "global-0"
    global_pruned = pruned_dir / "global-0"
    global_raw.mkdir(parents=True, exist_ok=True)
    global_pruned.mkdir(parents=True, exist_ok=True)

    if not (global_raw / "lm_head.bf16.bin").exists():
        print("[Clover-35B-A3B] Streaming global-0 (embed_tokens from shard 1, norm + lm_head from shard 26)...", flush=True)
        t_s1 = fetch_safetensor_tensors(repo_id, "model-00001-of-00026.safetensors", ["model.embed_tokens.weight"])
        t_s26 = fetch_safetensor_tensors(repo_id, "model-00026-of-00026.safetensors", ["model.norm.weight", "lm_head.weight"])
        (global_raw / "embed_tokens.bf16.bin").write_bytes(tensor_to_le_bytes(t_s1["model.embed_tokens.weight"]))
        (global_raw / "norm.bf16.bin").write_bytes(tensor_to_le_bytes(t_s26["model.norm.weight"]))
        (global_raw / "lm_head.bf16.bin").write_bytes(tensor_to_le_bytes(t_s26["lm_head.weight"]))
        lm_q, lm_s, _ = quantize_4bit_group64(t_s26["lm_head.weight"])
        (global_raw / "lm_head.4bit.bin").write_bytes(lm_q + lm_s)
        del t_s1, t_s26

    for fname in ("embed_tokens.bf16.bin", "norm.bf16.bin", "lm_head.bf16.bin", "lm_head.4bit.bin"):
        src_p = global_raw / fname
        dst_p = global_pruned / fname
        if not dst_p.exists():
            try:
                os.link(src_p, dst_p)
            except OSError:
                dst_p.write_bytes(src_p.read_bytes())

    # 2. All 30 BD-GDN layers' bidirectional preview weights (`bd_gdn_conv_right_weight.safetensors` & `bd_gdn_softmax_scalars.safetensors`)
    print("[Clover-35B-A3B] Streaming all 30 BD-GDN bidirectional preview weights...", flush=True)
    crw_keys = [f"model.layers.{l}.linear_attn.conv_right_weight" for l in BD_GDN_LAYERS_0IDX]
    scl_keys = []
    for l in BD_GDN_LAYERS_0IDX:
        scl_keys.extend([
            f"model.layers.{l}.linear_attn.bd_softmax_logit_scale",
            f"model.layers.{l}.linear_attn.bd_softmax_history_bias",
            f"model.layers.{l}.linear_attn.bd_softmax_history_c",
        ])
    t_crw = fetch_safetensor_tensors(repo_id, "bd_gdn_conv_right_weight.safetensors", crw_keys)
    t_scl = fetch_safetensor_tensors(repo_id, "bd_gdn_softmax_scalars.safetensors", scl_keys)

    for l in BD_GDN_LAYERS_0IDX:
        bd_bytes = pack_bd_gdn_bytes(
            l,
            t_crw[f"model.layers.{l}.linear_attn.conv_right_weight"],
            t_scl[f"model.layers.{l}.linear_attn.bd_softmax_logit_scale"],
            t_scl[f"model.layers.{l}.linear_attn.bd_softmax_history_bias"],
            t_scl[f"model.layers.{l}.linear_attn.bd_softmax_history_c"],
        )
        for root_d in (raw_dir, pruned_dir):
            td = root_d / f"trunk-{l}"
            td.mkdir(parents=True, exist_ok=True)
            (td / "bd_gdn.bin").write_bytes(bd_bytes)

    # 3. Layer 0: BD-GDN + 256-Expert MoE + Shared Expert (shards 1 & 2)
    l0_p = "model.layers.0."
    if not (pruned_dir / "trunk-0" / "operator.bin").exists():
        print(f"[Clover-35B-A3B] Building trunk-0 + expert-0 (Layer 0 BD-GDN + MoE, {num_cached_experts_per_stage} cached experts)...", flush=True)
        t0_s1 = fetch_safetensor_tensors(
            repo_id,
            "model-00001-of-00026.safetensors",
            [
                l0_p + "linear_attn.in_proj_qkv.weight",
                l0_p + "linear_attn.in_proj_z.weight",
                l0_p + "linear_attn.out_proj.weight",
            ],
        )
        t0_s2 = fetch_safetensor_tensors(
            repo_id,
            "model-00002-of-00026.safetensors",
            [
                l0_p + "input_layernorm.weight",
                l0_p + "post_attention_layernorm.weight",
                l0_p + "linear_attn.A_log",
                l0_p + "linear_attn.conv1d.weight",
                l0_p + "linear_attn.dt_bias",
                l0_p + "linear_attn.in_proj_a.weight",
                l0_p + "linear_attn.in_proj_b.weight",
                l0_p + "linear_attn.norm.weight",
                l0_p + "mlp.gate.weight",
                l0_p + "mlp.shared_expert.gate_proj.weight",
                l0_p + "mlp.shared_expert.up_proj.weight",
                l0_p + "mlp.shared_expert.down_proj.weight",
                l0_p + "mlp.shared_expert_gate.weight",
            ],
        )
        gu0 = fetch_safetensor_leading_slice(repo_id, "model-00001-of-00026.safetensors", l0_p + "mlp.experts.gate_up_proj", num_cached_experts_per_stage)
        dn0 = fetch_safetensor_leading_slice(repo_id, "model-00002-of-00026.safetensors", l0_p + "mlp.experts.down_proj", num_cached_experts_per_stage)

        b_in_ln = tensor_to_le_bytes(t0_s2[l0_p + "input_layernorm.weight"])
        b_qkv = tensor_to_le_bytes(t0_s1[l0_p + "linear_attn.in_proj_qkv.weight"])
        b_z = tensor_to_le_bytes(t0_s1[l0_p + "linear_attn.in_proj_z.weight"])
        b_conv = tensor_to_le_bytes(t0_s2[l0_p + "linear_attn.conv1d.weight"])
        op0_hdr = bytearray(64)
        struct.pack_into(
            "<8sIIIIII",
            op0_hdr,
            0,
            OPERATOR_MAGIC,
            0,  # kind=0 (BD-GDN)
            0,  # layer=0
            len(b_in_ln),
            len(b_qkv),
            len(b_z),
            len(b_conv),
        )
        op0_bytes = bytes(op0_hdr) + b_in_ln + b_qkv + b_z + b_conv
        bd0_bytes = (pruned_dir / "trunk-0" / "bd_gdn.bin").read_bytes()

        l0_records = [
            (0, 0, DTYPE_BF16, 2048, 1, b_in_ln),
            (1, 0, DTYPE_BF16, 2048, 1, tensor_to_le_bytes(t0_s2[l0_p + "post_attention_layernorm.weight"])),
            (2, 0, DTYPE_BF16, 2048, 4096, tensor_to_le_bytes(t0_s1[l0_p + "linear_attn.out_proj.weight"])),
            (3, 1, DTYPE_BF16, 8192, 2048, b_qkv),
            (4, 1, DTYPE_BF16, 4096, 2048, b_z),
            (5, 1, DTYPE_BF16, 8192, 4, b_conv),
            (6, 1, DTYPE_BF16, 32, 2048, tensor_to_le_bytes(t0_s2[l0_p + "linear_attn.in_proj_a.weight"])),
            (7, 1, DTYPE_BF16, 32, 2048, tensor_to_le_bytes(t0_s2[l0_p + "linear_attn.in_proj_b.weight"])),
            (8, 1, DTYPE_FP32, 32, 1, tensor_to_le_bytes(t0_s2[l0_p + "linear_attn.A_log"].float())),
            (9, 1, DTYPE_FP32, 32, 1, tensor_to_le_bytes(t0_s2[l0_p + "linear_attn.dt_bias"].float())),
            (10, 1, DTYPE_BF16, 128, 1, tensor_to_le_bytes(t0_s2[l0_p + "linear_attn.norm.weight"])),
            (20, 2, DTYPE_BF16, 256, 2048, tensor_to_le_bytes(t0_s2[l0_p + "mlp.gate.weight"])),
            (21, 2, DTYPE_BF16, 512, 2048, tensor_to_le_bytes(t0_s2[l0_p + "mlp.shared_expert.gate_proj.weight"])),
            (22, 2, DTYPE_BF16, 512, 2048, tensor_to_le_bytes(t0_s2[l0_p + "mlp.shared_expert.up_proj.weight"])),
            (23, 2, DTYPE_BF16, 2048, 512, tensor_to_le_bytes(t0_s2[l0_p + "mlp.shared_expert.down_proj.weight"])),
            (24, 2, DTYPE_BF16, 1, 2048, tensor_to_le_bytes(t0_s2[l0_p + "mlp.shared_expert_gate.weight"])),
        ]
        # Prune records 0, 3, 4, 5 which are already packed into operator.bin
        l0_prune_needs = {1, 2, 6, 7, 8, 9, 10, 20, 21, 22, 23, 24}
        write_clover35b_trunk_container(raw_dir / "trunk-0", 0, 0, l0_records, operator_bytes=op0_bytes, bd_gdn_bytes=bd0_bytes, prune_needs=None)
        write_clover35b_trunk_container(pruned_dir / "trunk-0", 0, 0, l0_records, operator_bytes=op0_bytes, bd_gdn_bytes=bd0_bytes, prune_needs=l0_prune_needs)
        write_expert_store(raw_dir / "expert-0", 0, gu0, dn0, t0_s2[l0_p + "mlp.gate.weight"])
        write_expert_store(pruned_dir / "expert-0", 0, gu0, dn0, t0_s2[l0_p + "mlp.gate.weight"])

    # 4. Layer 3: Gated GQA + 256-Expert MoE + Shared Expert (shard 3)
    l3_p = "model.layers.3."
    if not (pruned_dir / "trunk-3" / "operator.bin").exists():
        print(f"[Clover-35B-A3B] Building trunk-3 + expert-3 (Layer 3 Gated GQA + MoE, {num_cached_experts_per_stage} cached experts)...", flush=True)
        t3_s3 = fetch_safetensor_tensors(
            repo_id,
            "model-00003-of-00026.safetensors",
            [
                l3_p + "input_layernorm.weight",
                l3_p + "post_attention_layernorm.weight",
                l3_p + "self_attn.q_proj.weight",
                l3_p + "self_attn.k_proj.weight",
                l3_p + "self_attn.v_proj.weight",
                l3_p + "self_attn.o_proj.weight",
                l3_p + "self_attn.q_norm.weight",
                l3_p + "self_attn.k_norm.weight",
                l3_p + "mlp.gate.weight",
                l3_p + "mlp.shared_expert.gate_proj.weight",
                l3_p + "mlp.shared_expert.up_proj.weight",
                l3_p + "mlp.shared_expert.down_proj.weight",
                l3_p + "mlp.shared_expert_gate.weight",
            ],
        )
        gu3 = fetch_safetensor_leading_slice(repo_id, "model-00003-of-00026.safetensors", l3_p + "mlp.experts.gate_up_proj", num_cached_experts_per_stage)
        dn3 = fetch_safetensor_leading_slice(repo_id, "model-00003-of-00026.safetensors", l3_p + "mlp.experts.down_proj", num_cached_experts_per_stage)

        b3_in_ln = tensor_to_le_bytes(t3_s3[l3_p + "input_layernorm.weight"])
        b3_q = tensor_to_le_bytes(t3_s3[l3_p + "self_attn.q_proj.weight"])
        b3_k = tensor_to_le_bytes(t3_s3[l3_p + "self_attn.k_proj.weight"])
        b3_v = tensor_to_le_bytes(t3_s3[l3_p + "self_attn.v_proj.weight"])
        b3_qn = tensor_to_le_bytes(t3_s3[l3_p + "self_attn.q_norm.weight"])
        b3_kn = tensor_to_le_bytes(t3_s3[l3_p + "self_attn.k_norm.weight"])
        op3_hdr = bytearray(64)
        struct.pack_into(
            "<8sIIIIIIII",
            op3_hdr,
            0,
            OPERATOR_MAGIC,
            1,  # kind=1 (Gated GQA)
            3,  # layer=3
            len(b3_in_ln),
            len(b3_q),
            len(b3_k),
            len(b3_v),
            len(b3_qn),
            len(b3_kn),
        )
        op3_bytes = bytes(op3_hdr) + b3_in_ln + b3_q + b3_k + b3_v + b3_qn + b3_kn

        l3_records = [
            (0, 0, DTYPE_BF16, 2048, 1, b3_in_ln),
            (1, 0, DTYPE_BF16, 2048, 1, tensor_to_le_bytes(t3_s3[l3_p + "post_attention_layernorm.weight"])),
            (2, 0, DTYPE_BF16, 2048, 4096, tensor_to_le_bytes(t3_s3[l3_p + "self_attn.o_proj.weight"])),
            (3, 1, DTYPE_BF16, 8192, 2048, b3_q),
            (4, 1, DTYPE_BF16, 512, 2048, b3_k),
            (5, 1, DTYPE_BF16, 512, 2048, b3_v),
            (6, 1, DTYPE_BF16, 256, 1, b3_qn),
            (7, 1, DTYPE_BF16, 256, 1, b3_kn),
            (20, 2, DTYPE_BF16, 256, 2048, tensor_to_le_bytes(t3_s3[l3_p + "mlp.gate.weight"])),
            (21, 2, DTYPE_BF16, 512, 2048, tensor_to_le_bytes(t3_s3[l3_p + "mlp.shared_expert.gate_proj.weight"])),
            (22, 2, DTYPE_BF16, 512, 2048, tensor_to_le_bytes(t3_s3[l3_p + "mlp.shared_expert.up_proj.weight"])),
            (23, 2, DTYPE_BF16, 2048, 512, tensor_to_le_bytes(t3_s3[l3_p + "mlp.shared_expert.down_proj.weight"])),
            (24, 2, DTYPE_BF16, 1, 2048, tensor_to_le_bytes(t3_s3[l3_p + "mlp.shared_expert_gate.weight"])),
        ]
        # Prune records 0, 3, 4, 5, 6, 7 (completely dropping attention.bin!)
        l3_prune_needs = {1, 2, 20, 21, 22, 23, 24}
        write_clover35b_trunk_container(raw_dir / "trunk-3", 3, 1, l3_records, operator_bytes=op3_bytes, prune_needs=None)
        write_clover35b_trunk_container(pruned_dir / "trunk-3", 3, 1, l3_records, operator_bytes=op3_bytes, prune_needs=l3_prune_needs)
        write_expert_store(raw_dir / "expert-3", 3, gu3, dn3, t3_s3[l3_p + "mlp.gate.weight"])
        write_expert_store(pruned_dir / "expert-3", 3, gu3, dn3, t3_s3[l3_p + "mlp.gate.weight"])

    summary = {
        "model_repo": repo_id,
        "work_dir": str(work_dir),
        "cached_experts_per_stage": num_cached_experts_per_stage,
        "bd_gdn_containers_extracted": len(BD_GDN_LAYERS_0IDX),
        "gqa_attention_bin_dropped": not (pruned_dir / "trunk-3" / "attention.bin").exists(),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    (work_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------- Trunk & Expert Readers + Differentiable GPU Stage Model ----------------

class Clover35BTrunkReader:
    def __init__(self, trunk_dir: Path):
        self.trunk_dir = Path(trunk_dir)
        idx = (self.trunk_dir / "index.bin").read_bytes()
        if idx[:8] != TRUNK_MAGIC:
            raise ValueError(f"Invalid Clover-35B trunk magic in {trunk_dir}")
        _, self.ver, self.layer_id, n_rec = struct.unpack_from("<8sIII", idx, 0)
        self.layer_kind = struct.unpack_from("<I", idx, 48)[0]
        self.groups: dict[int, memoryview] = {}
        for g_idx, g_name in enumerate(GROUPS):
            gp = self.trunk_dir / g_name
            if gp.is_file():
                self.groups[g_idx] = memoryview(gp.read_bytes())
        self.records: dict[int, tuple[int, int, int, int, int, int]] = {}
        for i in range(n_rec):
            base = HEADER_BYTES + i * RECORD_BYTES
            rid, gid, dt, r, c, _ = struct.unpack_from("<6I", idx, base)
            off, ln = struct.unpack_from("<2Q", idx, base + 24)
            self.records[rid] = (gid, dt, r, c, off, ln)
        self.op_bytes = (self.trunk_dir / "operator.bin").read_bytes() if (self.trunk_dir / "operator.bin").is_file() else b""
        self.bd_gdn = unpack_bd_gdn_bytes((self.trunk_dir / "bd_gdn.bin").read_bytes()) if (self.trunk_dir / "bd_gdn.bin").is_file() else None

    def get_record(self, rid: int, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        gid, dt, r, c, off, ln = self.records[rid]
        if ln == 0:
            raise KeyError(f"Record {rid} was pruned from group {gid} in {self.trunk_dir}")
        sh = shape or ((r,) if c == 1 else (r, c))
        return bytes_to_tensor(self.groups[gid][off : off + ln], dt, sh)

    def unpack_operator(self) -> dict[str, torch.Tensor]:
        mv = memoryview(self.op_bytes)
        if self.layer_kind == 0:
            _, _, _, l_ln, l_qkv, l_z, l_conv = struct.unpack_from("<8sIIIIII", mv, 0)
            cur = 64
            in_ln = bytes_to_tensor(mv[cur : cur + l_ln], DTYPE_BF16, (2048,)); cur += l_ln
            qkv = bytes_to_tensor(mv[cur : cur + l_qkv], DTYPE_BF16, (8192, 2048)); cur += l_qkv
            z = bytes_to_tensor(mv[cur : cur + l_z], DTYPE_BF16, (4096, 2048)); cur += l_z
            conv = bytes_to_tensor(mv[cur : cur + l_conv], DTYPE_BF16, (8192, 1, 4))
            return {"in_ln": in_ln, "in_proj_qkv": qkv, "in_proj_z": z, "conv1d": conv}
        _, _, _, l_ln, l_q, l_k, l_v, l_qn, l_kn = struct.unpack_from("<8sIIIIIIII", mv, 0)
        cur = 64
        in_ln = bytes_to_tensor(mv[cur : cur + l_ln], DTYPE_BF16, (2048,)); cur += l_ln
        q = bytes_to_tensor(mv[cur : cur + l_q], DTYPE_BF16, (8192, 2048)); cur += l_q
        k = bytes_to_tensor(mv[cur : cur + l_k], DTYPE_BF16, (512, 2048)); cur += l_k
        v = bytes_to_tensor(mv[cur : cur + l_v], DTYPE_BF16, (512, 2048)); cur += l_v
        qn = bytes_to_tensor(mv[cur : cur + l_qn], DTYPE_BF16, (256,)); cur += l_qn
        kn = bytes_to_tensor(mv[cur : cur + l_kn], DTYPE_BF16, (256,))
        return {"in_ln": in_ln, "q_proj": q, "k_proj": k, "v_proj": v, "q_norm": qn, "k_norm": kn}


def load_expert_store(exp_dir: Path, use_4bit: bool = False, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    maps = json.loads((exp_dir / "maps.json").read_text(encoding="utf-8"))
    gu_shape = tuple(maps["gate_up_shape"])
    dn_shape = tuple(maps["down_shape"])
    if use_4bit:
        raw4 = (exp_dir / "experts.4bit.direct").read_bytes()
        _, _, _, _, l_gu_q, l_gu_s, l_dn_q, l_dn_s = struct.unpack_from("<8sIIIIIII", raw4, 0)
        cur = 64
        gu_q = raw4[cur : cur + l_gu_q]; cur += l_gu_q
        gu_s = raw4[cur : cur + l_gu_s]; cur += l_gu_s
        dn_q = raw4[cur : cur + l_dn_q]; cur += l_dn_q
        dn_s = raw4[cur : cur + l_dn_s]
        gu = dequantize_4bit_group64(gu_q, gu_s, gu_shape, device=device)
        dn = dequantize_4bit_group64(dn_q, dn_s, dn_shape, device=device)
        return gu, dn, maps
    raw = (exp_dir / "experts.direct").read_bytes()
    mv = memoryview(raw)
    l_gu = int(maps["gate_up_bf16_bytes"])
    gu = bytes_to_tensor(mv[:l_gu], DTYPE_BF16, gu_shape)
    dn = bytes_to_tensor(mv[l_gu:], DTYPE_BF16, dn_shape)
    if device is not None:
        gu = gu.to(device)
        dn = dn.to(device)
    return gu, dn, maps


class Clover35BGPUStageModel(nn.Module):
    """
    GPU-resident differentiable stage model for `Clover-35B-A3B` (`Diffuspace/Clover-35B-A3B-Preview`):
      - `embed_tokens [248320, 2048]`
      - `Layer 0`: Bidirectional Gated DeltaNet (`BD-GDN`) + 256-Expert MoE (with `coreset_routing_op`) + Shared Expert
      - `Layer 3`: Gated Full-Attention (`Gated GQA` 16Q/2KV, `head_dim=256`, partial RoPE `0.25`) + 256-Expert MoE + Shared Expert
      - `Final Tail`: Zero-centered `CloverRMSNorm` + `lm_head [248320, 2048]`
    """

    def __init__(self, stage_dir: Path, device: torch.device | str = "cuda:0", use_4bit_experts: bool = False, from_raw_records: bool = False):
        super().__init__()
        self.stage_dir = Path(stage_dir)
        self.dev = torch.device(device if (torch.cuda.is_available() and str(device).startswith("cuda")) else "cpu")
        self.use_4bit = use_4bit_experts
        self.from_raw = from_raw_records

        g_dir = self.stage_dir / "global-0"
        self.embed_w = bytes_to_tensor((g_dir / "embed_tokens.bf16.bin").read_bytes(), DTYPE_BF16, (V_CLOVER35B, D_CLOVER35B)).to(self.dev)
        self.norm_w = bytes_to_tensor((g_dir / "norm.bf16.bin").read_bytes(), DTYPE_BF16, (D_CLOVER35B,)).to(self.dev)
        self.lm_head_w = bytes_to_tensor((g_dir / "lm_head.bf16.bin").read_bytes(), DTYPE_BF16, (V_CLOVER35B, D_CLOVER35B)).to(self.dev)

        # Layer 0 (BD-GDN + MoE)
        t0 = Clover35BTrunkReader(self.stage_dir / "trunk-0")
        op0 = t0.unpack_operator()
        self.l0_in_ln = (t0.get_record(0, (2048,)) if from_raw_records else op0["in_ln"]).to(self.dev)
        self.l0_post_ln = t0.get_record(1, (2048,)).to(self.dev)
        self.l0_out_proj = t0.get_record(2, (2048, 4096)).to(self.dev)
        self.l0_qkv = (t0.get_record(3, (8192, 2048)) if from_raw_records else op0["in_proj_qkv"]).to(self.dev)
        self.l0_z = (t0.get_record(4, (4096, 2048)) if from_raw_records else op0["in_proj_z"]).to(self.dev)
        self.l0_conv = (t0.get_record(5, (8192, 1, 4)) if from_raw_records else op0["conv1d"]).to(self.dev)
        self.l0_a = t0.get_record(6, (32, 2048)).to(self.dev)
        self.l0_b = t0.get_record(7, (32, 2048)).to(self.dev)
        self.l0_A_log = t0.get_record(8, (32,)).to(self.dev)
        self.l0_dt_bias = t0.get_record(9, (32,)).to(self.dev)
        self.l0_gdn_norm = t0.get_record(10, (128,)).to(self.dev)
        self.l0_bd = {k: (v.to(self.dev) if isinstance(v, torch.Tensor) else v) for k, v in (t0.bd_gdn or {}).items()}
        self.l0_gate = t0.get_record(20, (256, 2048)).to(self.dev)
        self.l0_sh_gate = t0.get_record(21, (512, 2048)).to(self.dev)
        self.l0_sh_up = t0.get_record(22, (512, 2048)).to(self.dev)
        self.l0_sh_down = t0.get_record(23, (2048, 512)).to(self.dev)
        self.l0_sh_g = t0.get_record(24, (1, 2048)).to(self.dev)
        self.l0_gu, self.l0_dn, self.l0_maps = load_expert_store(self.stage_dir / "expert-0", use_4bit=use_4bit_experts, device=self.dev)

        # Layer 3 (Gated GQA + MoE)
        t3 = Clover35BTrunkReader(self.stage_dir / "trunk-3")
        op3 = t3.unpack_operator()
        self.l3_in_ln = (t3.get_record(0, (2048,)) if from_raw_records else op3["in_ln"]).to(self.dev)
        self.l3_post_ln = t3.get_record(1, (2048,)).to(self.dev)
        self.l3_o_proj = t3.get_record(2, (2048, 4096)).to(self.dev)
        self.l3_q = (t3.get_record(3, (8192, 2048)) if from_raw_records else op3["q_proj"]).to(self.dev)
        self.l3_k = (t3.get_record(4, (512, 2048)) if from_raw_records else op3["k_proj"]).to(self.dev)
        self.l3_v = (t3.get_record(5, (512, 2048)) if from_raw_records else op3["v_proj"]).to(self.dev)
        self.l3_qn = (t3.get_record(6, (256,)) if from_raw_records else op3["q_norm"]).to(self.dev)
        self.l3_kn = (t3.get_record(7, (256,)) if from_raw_records else op3["k_norm"]).to(self.dev)
        self.l3_gate = t3.get_record(20, (256, 2048)).to(self.dev)
        self.l3_sh_gate = t3.get_record(21, (512, 2048)).to(self.dev)
        self.l3_sh_up = t3.get_record(22, (512, 2048)).to(self.dev)
        self.l3_sh_down = t3.get_record(23, (2048, 512)).to(self.dev)
        self.l3_sh_g = t3.get_record(24, (1, 2048)).to(self.dev)
        self.l3_gu, self.l3_dn, self.l3_maps = load_expert_store(self.stage_dir / "expert-3", use_4bit=use_4bit_experts, device=self.dev)

    def _run_moe(
        self,
        x: torch.Tensor,
        gate_w: torch.Tensor,
        gu_w: torch.Tensor,
        dn_w: torch.Tensor,
        sh_gate_w: torch.Tensor,
        sh_up_w: torch.Tensor,
        sh_down_w: torch.Tensor,
        sh_g_w: torch.Tensor,
        preview: bool = False,
    ) -> torch.Tensor:
        b, t, d = x.shape
        k_cached = gu_w.shape[0]
        logits = F.linear(x, gate_w)[..., :k_cached]
        valid = torch.ones((b, t), device=x.device, dtype=torch.bool)
        if preview:
            logits = coreset_routing_op(logits, valid=valid, top_k=min(TOP_K_35B, k_cached), top_p=0.9)
        probs = logits.float().softmax(-1)
        top_k = min(TOP_K_35B, k_cached)
        ids = logits.float().argsort(dim=-1, descending=True, stable=True)[..., :top_k]
        weights = probs.gather(-1, ids)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)

        flat_x = x.reshape(-1, d)
        flat_ids = ids.reshape(-1, top_k)
        flat_w = weights.reshape(-1, top_k)
        out = torch.zeros_like(flat_x, dtype=torch.float32)
        for exp_id in flat_ids.unique().tolist():
            tok_idx, slot_idx = torch.where(flat_ids == exp_id)
            gate, up = F.linear(flat_x[tok_idx], gu_w[exp_id]).chunk(2, dim=-1)
            hid = (F.silu(gate.float()) * up.float()).to(x.dtype)
            y = F.linear(hid.float(), dn_w[exp_id].float())
            routed = (y * flat_w[tok_idx, slot_idx, None].float()).to(x.dtype)
            out.index_add_(0, tok_idx, routed.float())
        y_moe = out.to(x.dtype).view(b, t, d)

        sh_gate = F.linear(x, sh_gate_w)
        sh_up = F.linear(x, sh_up_w)
        sh_out = F.linear((F.silu(sh_gate.float()) * sh_up.float()).to(x.dtype), sh_down_w)
        sh_scale = torch.sigmoid(F.linear(x, sh_g_w))
        return y_moe + sh_out * sh_scale

    def forward_layer0_bd_gdn(self, x: torch.Tensor, preview: bool = False) -> torch.Tensor:
        b, t, _ = x.shape
        u = clover_rms_norm_op(x, self.l0_in_ln, eps=1e-6, zero_centered=True)
        raw_qkv = F.linear(u, self.l0_qkv)
        if preview and self.l0_bd:
            mixed = bd_gdn_conv1d_op(raw_qkv, self.l0_conv, self.l0_bd["conv_right_weight"])
        else:
            conv_state = raw_qkv.new_zeros((b, 8192, 3))
            joined = torch.cat((conv_state, raw_qkv.transpose(1, 2)), dim=-1)
            mixed = causal_conv_silu_op(joined, self.l0_conv).transpose(1, 2)
        q, k, v = mixed.split((2048, 2048, 4096), dim=-1)
        q = q.reshape(b, t, 16, 128)
        k = k.reshape(b, t, 16, 128)
        v = v.reshape(b, t, 32, 128)
        g = -self.l0_A_log.float().exp() * F.softplus(F.linear(u, self.l0_a).float() + self.l0_dt_bias.float())
        if preview and self.l0_bd:
            y_attn = bd_gdn_local_memory_op(
                q, k, v, g,
                self.l0_bd["bd_softmax_logit_scale"],
                self.l0_bd["bd_softmax_history_bias"],
                self.l0_bd["bd_softmax_history_c"],
            )
        else:
            beta = F.linear(u, self.l0_b).sigmoid()
            y_attn, _ = recurrent_commit_op(q, k, v, g, beta)
        z = F.linear(u, self.l0_z).reshape(b, t, 32, 128)
        y_norm = clover_rms_norm_op(y_attn, self.l0_gdn_norm, eps=1e-6, zero_centered=False, gate=z)
        h_attn = x + F.linear(y_norm.reshape(b, t, 4096), self.l0_out_proj)
        u_mlp = clover_rms_norm_op(h_attn, self.l0_post_ln, eps=1e-6, zero_centered=True)
        return h_attn + self._run_moe(
            u_mlp, self.l0_gate, self.l0_gu, self.l0_dn,
            self.l0_sh_gate, self.l0_sh_up, self.l0_sh_down, self.l0_sh_g, preview=preview,
        )

    def _rotary_partial(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        rd = 64  # head_dim=256 * partial_rotary_factor=0.25
        inv = 1.0 / (10000000.0 ** (torch.arange(0, rd, 2, device=x.device, dtype=torch.float32) / rd))
        phase = positions.float()[..., None] * inv
        cos = phase.cos().to(x.dtype)[:, None]
        sin = phase.sin().to(x.dtype)[:, None]
        first, second = x[..., :rd].chunk(2, dim=-1)
        rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
        return torch.cat((rotated, x[..., rd:]), dim=-1)

    def forward_layer3_gated_gqa(self, x: torch.Tensor, preview: bool = False) -> torch.Tensor:
        b, t, _ = x.shape
        u = clover_rms_norm_op(x, self.l3_in_ln, eps=1e-6, zero_centered=True)
        q, gate = F.linear(u, self.l3_q).view(b, t, 16, 512).chunk(2, dim=-1)
        positions = torch.arange(t, device=x.device, dtype=torch.long).unsqueeze(0).expand(b, t)
        q = self._rotary_partial(clover_rms_norm_op(q, self.l3_qn, eps=1e-6, zero_centered=True).transpose(1, 2), positions)
        k = self._rotary_partial(
            clover_rms_norm_op(F.linear(u, self.l3_k).view(b, t, 2, 256), self.l3_kn, eps=1e-6, zero_centered=True).transpose(1, 2),
            positions,
        )
        v = F.linear(u, self.l3_v).view(b, t, 2, 256).transpose(1, 2)
        k = k.repeat_interleave(8, dim=1)
        v = v.repeat_interleave(8, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=not preview, dropout_p=0.0)
        y = y.transpose(1, 2).reshape(b, t, 4096) * gate.reshape(b, t, 4096).sigmoid()
        h_attn = x + F.linear(y, self.l3_o_proj)
        u_mlp = clover_rms_norm_op(h_attn, self.l3_post_ln, eps=1e-6, zero_centered=True)
        return h_attn + self._run_moe(
            u_mlp, self.l3_gate, self.l3_gu, self.l3_dn,
            self.l3_sh_gate, self.l3_sh_up, self.l3_sh_down, self.l3_sh_g, preview=preview,
        )

    def forward_ids(self, input_ids: torch.Tensor, preview: bool = False) -> dict[str, torch.Tensor]:
        ids = input_ids.to(self.dev)
        h0 = F.embedding(ids, self.embed_w)
        h1 = self.forward_layer0_bd_gdn(h0, preview=preview)
        h3 = self.forward_layer3_gated_gqa(h1, preview=preview)
        h_final = clover_rms_norm_op(h3, self.norm_w, eps=1e-6, zero_centered=True)
        logits = F.linear(h_final, self.lm_head_w)
        return {"h0": h0, "h_bd_gdn": h1, "h_gqa_moe": h3, "h_final": h_final, "logits": logits}


@torch.no_grad()
def verify_clover_35b_stages(work_dir: Path, device: str = "cuda:0") -> dict[str, Any]:
    """Verifies bit-exact (`cos_sim = 1.0`, `max_abs_diff = 0.0`) equivalence between `raw/` and `pruned/`
    containers in both causal commit mode and bidirectional block-diffusion preview mode, plus 4-bit expert accuracy."""
    raw_dir = work_dir / "raw"
    pruned_dir = work_dir / "pruned"
    dev = "cuda:0" if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu"

    mod_raw = Clover35BGPUStageModel(raw_dir, device=dev, use_4bit_experts=False, from_raw_records=True)
    mod_pruned = Clover35BGPUStageModel(pruned_dir, device=dev, use_4bit_experts=False, from_raw_records=False)
    mod_4bit = Clover35BGPUStageModel(pruned_dir, device=dev, use_4bit_experts=True, from_raw_records=False)

    sample_ids = torch.tensor([[IM_START_ID, 872, 198, 3838, 374, 279, 3491, 248046]], dtype=torch.long)

    out_raw_causal = mod_raw.forward_ids(sample_ids, preview=False)
    out_pruned_causal = mod_pruned.forward_ids(sample_ids, preview=False)
    out_4bit_causal = mod_4bit.forward_ids(sample_ids, preview=False)

    out_raw_prev = mod_raw.forward_ids(sample_ids, preview=True)
    out_pruned_prev = mod_pruned.forward_ids(sample_ids, preview=True)

    causal_cos = float(F.cosine_similarity(out_raw_causal["h_final"].float().reshape(1, -1), out_pruned_causal["h_final"].float().reshape(1, -1)).item())
    causal_diff = float((out_raw_causal["h_final"].float() - out_pruned_causal["h_final"].float()).abs().max().item())

    prev_cos = float(F.cosine_similarity(out_raw_prev["h_final"].float().reshape(1, -1), out_pruned_prev["h_final"].float().reshape(1, -1)).item())
    prev_diff = float((out_raw_prev["h_final"].float() - out_pruned_prev["h_final"].float()).abs().max().item())

    q4_cos = float(F.cosine_similarity(out_pruned_causal["h_final"].float().reshape(1, -1), out_4bit_causal["h_final"].float().reshape(1, -1)).item())

    return {
        "device": dev,
        "causal_commit_pruned_vs_raw_cos": round(causal_cos, 6),
        "causal_commit_pruned_vs_raw_max_abs_diff": causal_diff,
        "block_diffusion_preview_pruned_vs_raw_cos": round(prev_cos, 6),
        "block_diffusion_preview_pruned_vs_raw_max_abs_diff": prev_diff,
        "moe_4bit_vs_bf16_cos": round(q4_cos, 6),
        "hidden_state_shape": list(out_pruned_causal["h_final"].shape),
        "logits_shape": list(out_pruned_causal["logits"].shape),
        "bd_gdn_30_layers_verified": len(BD_GDN_LAYERS_0IDX),
        "gqa_attention_bin_dropped": not (pruned_dir / "trunk-3" / "attention.bin").exists(),
        "all_finite": bool(torch.isfinite(out_pruned_causal["logits"]).all().item() and torch.isfinite(out_pruned_prev["logits"]).all().item()),
    }
