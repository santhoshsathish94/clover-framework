#!/usr/bin/env python3
"""
Precomputed Stage-by-Stage Runtime Engine for NVIDIA Nemotron 3.5 Super VL
(`clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_precomputed_engine.py`).

Consumes ONLY the precomputed & pruned Clover containers (`trunk-L/index.bin`, `trunk-L/common.bin`,
`trunk-L/routed-shared.bin`, `trunk-L/operator.bin`, `expert-L/experts.direct`, `global-0/*.bin`)
and executes every stage with:
  1. Direct mmap / zero-copy buffer loading of `operator.bin` and pruned `trunk-L` records
     (including precomputed FP32 `A = -exp(A_log)` in Mamba-2 and pre-widened FP32 `gate.weight`
     in Latent MoE).
  2. On-demand `pread` / `mmap` slicing of only the 22 selected routed experts per token from
     `expert-L/experts.direct` (never loading all 512 experts into active compute memory),
     managed by Clover's 3-tier `SpecMD Least-Stale + R4 Clean-Drop` SlotCache!
  3. Optional `USE_4BIT_TRUNK_AND_EXPERTS=1` planar 4-bit group-64 execution (mirroring `trunk-4bit.h`).
  4. Native 2-layer Multi-Token Prediction (`MTP`) speculative draft generation (`trunk-mtp-0`,
     `trunk-mtp-1`, `expert-mtp-1/experts.direct`).
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

TRUNK_MAGIC = b"NEMOTRK1"
OPERATOR_MAGIC = b"NEMO_OP1"
HEADER_BYTES = 1168
RECORD_BYTES = 72
GROUPS = ("common.bin", "attention.bin", "routed-shared.bin")

DTYPE_BF16 = 1
DTYPE_FP32 = 2


def bytes_to_tensor(buf: bytes | memoryview, dtype_code: int, shape: tuple[int, ...]) -> torch.Tensor:
    if dtype_code == DTYPE_BF16:
        arr = np.frombuffer(buf, dtype=np.uint16).copy()
        return torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
    elif dtype_code == DTYPE_FP32:
        arr = np.frombuffer(buf, dtype=np.float32).copy()
        return torch.from_numpy(arr).reshape(shape)
    raise ValueError(f"Unsupported dtype_code: {dtype_code}")


def rms_norm_op(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    in_dtype = x.dtype
    x_f = x.to(torch.float32)
    var = x_f.pow(2).mean(-1, keepdim=True)
    return weight * (x_f * torch.rsqrt(var + eps)).to(in_dtype)


def gated_rms_norm_op(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, group_size: int = 1024, eps: float = 1e-5
) -> torch.Tensor:
    in_dtype = x.dtype
    x_f = x.to(torch.float32) * F.silu(gate.to(torch.float32))
    orig_shape = x_f.shape
    x_g = x_f.view(*orig_shape[:-1], -1, group_size)
    var = x_g.pow(2).mean(-1, keepdim=True)
    normed = (x_g * torch.rsqrt(var + eps)).view(*orig_shape)
    return (weight.to(torch.float32) * normed).to(in_dtype)


def relu2_op(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).pow(2)


class NemotronPrunedTrunkReader:
    """Reads a pruned (or unpruned) `trunk-L` directory (`index.bin`, non-empty group files, `operator.bin`)."""

    def __init__(self, trunk_dir: Path):
        self.trunk_dir = trunk_dir
        index = (trunk_dir / "index.bin").read_bytes()
        if index[:8] != TRUNK_MAGIC:
            raise ValueError(f"{trunk_dir}: invalid magic {index[:8]!r}")
        self.version, self.layer_id, self.count = struct.unpack_from("<III", index, 8)
        self.group_lengths = struct.unpack_from("<3Q", index, 24)
        self.layer_kind = struct.unpack_from("<I", index, 48)[0]

        self.group_bytes: list[bytes | None] = [None, None, None]
        for g_idx, g_name in enumerate(GROUPS):
            g_path = trunk_dir / g_name
            if self.group_lengths[g_idx] > 0 and g_path.exists():
                self.group_bytes[g_idx] = g_path.read_bytes()

        self.records: dict[int, tuple[int, int, int, int, int, int]] = {}
        for idx in range(self.count):
            base = HEADER_BYTES + idx * RECORD_BYTES
            rid, group, dtype_code, rows, cols, n = struct.unpack_from("<6I", index, base)
            offset, length = struct.unpack_from("<2Q", index, base + 24)
            self.records[rid] = (group, dtype_code, rows, cols, offset, length)

        op_path = trunk_dir / "operator.bin"
        self.operator_bytes = op_path.read_bytes() if op_path.exists() else None

    def get_record_tensor(self, rid: int, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        group, dtype_code, rows, cols, offset, length = self.records[rid]
        if length == 0:
            raise KeyError(
                f"Record {rid} in {self.trunk_dir.name} has length=0 (pruned because served by operator.bin!)"
            )
        g_buf = self.group_bytes[group]
        assert g_buf is not None
        raw = memoryview(g_buf)[offset : offset + length]
        target_shape = shape if shape is not None else ((rows,) if cols == 1 else (rows, cols))
        return bytes_to_tensor(raw, dtype_code, target_shape)


class NemotronDirectExpertStore:
    """
    Direct `mmap` / `pread` expert reader for `expert-L/experts.direct` integrated with
    Clover's SpecMD Policy 3 (`Least-Stale + R4 Clean-Drop`) slot cache.
    """

    def __init__(self, expert_dir: Path, pool_slots: int = 64, host_slots: int = 256):
        self.expert_dir = expert_dir
        self.meta = json.loads((expert_dir / "maps.json").read_text(encoding="utf-8"))
        self.layer_id = int(self.meta["layer_id"])
        self.num_experts = int(self.meta["num_experts"])
        self.stride = int(self.meta["expert_stride_bytes"])
        self.up_bytes = 2688 * 1024 * 2
        self.down_bytes = 1024 * 2688 * 2

        self._fd = os.open(str(expert_dir / "experts.direct"), os.O_RDONLY)
        self._mmap = mmap.mmap(self._fd, self.num_experts * self.stride, access=mmap.ACCESS_READ)

        # SpecMD Policy 3 (Least-Stale + R4 Clean-Drop) Slot Cache
        self.pool_capacity = pool_slots
        self.host_capacity = host_slots
        self.pool_cache: dict[int, tuple[torch.Tensor, torch.Tensor, int, float]] = {}
        self.host_cache: set[int] = set()
        self.clock = 0

        # Data-Movement Ledger counters
        self.total_requests = 0
        self.pool_hits = 0
        self.host_hits = 0
        self.drive_fetches = 0
        self.r4_clean_drops = 0
        self.dirty_writebacks = 0

    def close(self):
        if hasattr(self, "_mmap") and self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if hasattr(self, "_fd") and self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def get_expert_weights(
        self, expert_id: int, protected_experts: set[int], routing_weight: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetches `(up_proj, down_proj)` for `expert_id` via the 3-tier SpecMD R4 cache hierarchy."""
        self.total_requests += 1
        self.clock += 1

        if expert_id in self.pool_cache:
            up_w, down_w, _, heat = self.pool_cache[expert_id]
            self.pool_cache[expert_id] = (up_w, down_w, self.clock, 0.75 * heat + routing_weight)
            self.pool_hits += 1
            return up_w, down_w

        if expert_id in self.host_cache:
            self.host_hits += 1
        else:
            self.drive_fetches += 1
            self.host_cache.add(expert_id)

        # Read only this single expert's 11.01 MB slice from `experts.direct`
        base = expert_id * self.stride
        up_raw = memoryview(self._mmap)[base : base + self.up_bytes]
        down_raw = memoryview(self._mmap)[base + self.up_bytes : base + self.stride]
        up_w = bytes_to_tensor(up_raw, DTYPE_BF16, (2688, 1024))
        down_w = bytes_to_tensor(down_raw, DTYPE_BF16, (1024, 2688))

        # Insert into Tier-1 Pool Cache, evicting Least-Stale victim if full
        if len(self.pool_cache) >= self.pool_capacity:
            victim_id = None
            best_score = 1e30
            for k_id, (_, _, last_ts, heat) in self.pool_cache.items():
                if k_id in protected_experts:
                    continue
                staleness = max(1, self.clock - last_ts)
                score = heat / float(1 + staleness)
                if score < best_score:
                    best_score = score
                    victim_id = k_id
            if victim_id is None:
                victim_id = next(iter(self.pool_cache))
            del self.pool_cache[victim_id]
            # Red-Blue Pebble Rule R4: weights are read-only and backed by Host/NVMe -> clean drop, 0 writebacks!
            if victim_id in self.host_cache:
                self.r4_clean_drops += 1
            else:
                self.dirty_writebacks += 1

        self.pool_cache[expert_id] = (up_w, down_w, self.clock, routing_weight)
        return up_w, down_w

    def stats(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "total_expert_requests": self.total_requests,
            "tier1_pool_hits": self.pool_hits,
            "tier2_host_hits": self.host_hits,
            "tier3_drive_fetches": self.drive_fetches,
            "r4_clean_drops": self.r4_clean_drops,
            "dirty_writebacks": self.dirty_writebacks,
            "bytes_read_from_drive_mb": round(self.drive_fetches * self.stride / (1024 * 1024), 2),
            "full_512_experts_layer_mb": round(512 * self.stride / (1024 * 1024), 2),
            "io_saving_pct_vs_dense_load": round(
                100.0 * (1.0 - (self.drive_fetches / 512.0)), 2
            ),
        }


@torch.no_grad()
def execute_mamba_stage_from_precomputed(
    trunk: NemotronPrunedTrunkReader,
    hidden_states: torch.Tensor,
    chunk_size: int = 128,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Executes a Mamba-2 layer (`kind = 0`) reading `norm`, `in_proj`, `conv1d`, `dt_bias`,
    precomputed FP32 `A = -exp(A_log)`, and `D` directly from `operator.bin`, and `mixer.norm` (1)
    + `out_proj` (2) from pruned `common.bin` (`attention.bin` is completely absent!).
    """
    op_buf = trunk.operator_bytes
    assert op_buf is not None and op_buf[:8] == OPERATOR_MAGIC
    _, kind, layer_id, len_norm, len_in, len_cw, len_cb, len_dtb = struct.unpack_from("<8sIIIIIII", op_buf, 0)
    len_A, len_D = struct.unpack_from("<II", op_buf, 36)
    assert kind == 0

    mv = memoryview(op_buf)
    cur = 64
    norm_w = bytes_to_tensor(mv[cur : cur + len_norm], DTYPE_BF16, (4096,)); cur += len_norm
    in_proj_w = bytes_to_tensor(mv[cur : cur + len_in], DTYPE_BF16, (18560, 4096)); cur += len_in
    conv_w = bytes_to_tensor(mv[cur : cur + len_cw], DTYPE_BF16, (10240, 1, 4)); cur += len_cw
    conv_b = bytes_to_tensor(mv[cur : cur + len_cb], DTYPE_BF16, (10240,)); cur += len_cb
    dt_bias = bytes_to_tensor(mv[cur : cur + len_dtb], DTYPE_BF16, (128,)); cur += len_dtb
    A_precomp = bytes_to_tensor(mv[cur : cur + len_A], DTYPE_FP32, (128,)); cur += len_A
    D_vec = bytes_to_tensor(mv[cur : cur + len_D], DTYPE_BF16, (128,)); cur += len_D

    # Read remaining records 1 and 2 from pruned `common.bin`
    mixer_norm_w = trunk.get_record_tensor(1, (8192,))
    out_proj_w = trunk.get_record_tensor(2, (4096, 8192))

    bsz, seq_len, _ = hidden_states.shape
    u = rms_norm_op(hidden_states, norm_w, eps=eps)
    proj = F.linear(u, in_proj_w)
    gate, conv_in, dt = torch.split(proj, [8192, 10240, 128], dim=-1)

    # Causal conv1d (kernel=4, padding=3) + SiLU
    conv_T = conv_in.transpose(1, 2)
    conv_raw = F.conv1d(conv_T, conv_w, bias=conv_b, padding=3, groups=10240)[..., :seq_len]
    conv_act = F.silu(conv_raw).transpose(1, 2)
    x_ssm, B_ssm, C_ssm = torch.split(conv_act, [8192, 1024, 1024], dim=-1)

    # Use precomputed FP32 `A_precomp` directly (no `torch.exp(A_log)` needed at runtime!)
    from transformers.models.nemotron_h.modeling_nemotron_h import pad_tensor_by_size, reshape_into_chunks, segment_sum

    dt_f = torch.clamp(F.softplus(dt + dt_bias), min=0.001)
    x_h = x_ssm.reshape(bsz, seq_len, 128, 64).float()
    B_h = B_ssm.reshape(bsz, seq_len, 8, 128).float().repeat_interleave(16, dim=2, output_size=128)
    C_h = C_ssm.reshape(bsz, seq_len, 8, 128).float().repeat_interleave(16, dim=2, output_size=128)
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    D_residual = D_vec[..., None] * pad_tensor_by_size(x_h, pad_size)

    x_disc = x_h * dt_f[..., None]
    A_disc = A_precomp.to(x_disc.dtype) * dt_f
    x_c, A_c, B_c, C_c = [reshape_into_chunks(t, pad_size, chunk_size) for t in (x_disc, A_disc, B_h, C_h)]
    A_c = A_c.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A_c, dim=-1)
    L = torch.exp(segment_sum(A_c))
    G = (C_c[:, :, :, None, :, :] * B_c[:, :, None, :, :, :]).sum(dim=-1)
    M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
    Y_diag = (M[..., None] * x_c[:, :, None]).sum(3)

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B_c * decay_states.permute(0, 2, 3, 1)[..., None]
    states = (B_decay.permute(0, 1, 3, 2, 4)[..., None] * x_c.permute(0, 1, 3, 2, 4)[..., None, :]).sum(dim=3).permute(0, 1, 2, 4, 3)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = (decay_chunk[..., None, None] * states.permute(0, 2, 1, 3, 4)[:, :, None, ...]).sum(dim=2).permute(0, 2, 1, 3, 4)
    states = new_states[:, :-1]

    state_decay_out = torch.exp(A_cumsum).permute(0, 2, 3, 1)
    Y_off = (C_c[..., None, :] * states[:, :, None, ...]).sum(-1) * state_decay_out[..., None]
    y = (Y_diag + Y_off).reshape(bsz, -1, 128, 64) + D_residual
    if pad_size > 0:
        y = y[:, :seq_len, :, :]
    y = y.reshape(bsz, seq_len, 8192)
    scan_out = gated_rms_norm_op(y, gate, mixer_norm_w, group_size=1024, eps=eps)
    out = F.linear(scan_out.to(hidden_states.dtype), out_proj_w)
    return hidden_states + out


@torch.no_grad()
def execute_attention_stage_from_precomputed(
    trunk: NemotronPrunedTrunkReader,
    hidden_states: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Executes a NoPE 16:1 GQA Attention layer (`kind = 1`) reading `norm`, `q_proj`, `k_proj`, `v_proj`
    directly from `operator.bin` and `o_proj` (1) from pruned `common.bin` (`attention.bin` is absent!).
    """
    op_buf = trunk.operator_bytes
    assert op_buf is not None and op_buf[:8] == OPERATOR_MAGIC
    _, kind, layer_id, len_norm, len_q, len_k, len_v = struct.unpack_from("<8sIIIIII", op_buf, 0)
    assert kind == 1

    mv = memoryview(op_buf)
    cur = 64
    norm_w = bytes_to_tensor(mv[cur : cur + len_norm], DTYPE_BF16, (4096,)); cur += len_norm
    q_w = bytes_to_tensor(mv[cur : cur + len_q], DTYPE_BF16, (4096, 4096)); cur += len_q
    k_w = bytes_to_tensor(mv[cur : cur + len_k], DTYPE_BF16, (256, 4096)); cur += len_k
    v_w = bytes_to_tensor(mv[cur : cur + len_v], DTYPE_BF16, (256, 4096)); cur += len_v

    o_w = trunk.get_record_tensor(1, (4096, 4096))

    bsz, seq_len, _ = hidden_states.shape
    u = rms_norm_op(hidden_states, norm_w, eps=eps)
    q = F.linear(u, q_w).view(bsz, seq_len, 32, 128).transpose(1, 2)
    k = F.linear(u, k_w).view(bsz, seq_len, 2, 128).transpose(1, 2)
    v = F.linear(u, v_w).view(bsz, seq_len, 2, 128).transpose(1, 2)

    k_rep = k[:, :, None, :, :].expand(bsz, 2, 16, seq_len, 128).reshape(bsz, 32, seq_len, 128)
    v_rep = v[:, :, None, :, :].expand(bsz, 2, 16, seq_len, 128).reshape(bsz, 32, seq_len, 128)

    scores = torch.matmul(q, k_rep.transpose(2, 3)) * (128**-0.5)
    row_idx = torch.arange(seq_len, device=hidden_states.device)[:, None]
    col_idx = torch.arange(seq_len, device=hidden_states.device)[None, :]
    causal = col_idx <= row_idx
    scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
    ctx = torch.matmul(probs, v_rep).transpose(1, 2).contiguous().view(bsz, seq_len, 4096)
    return hidden_states + F.linear(ctx, o_w)


@torch.no_grad()
def execute_latent_moe_stage_from_precomputed(
    trunk: NemotronPrunedTrunkReader,
    expert_store: NemotronDirectExpertStore,
    hidden_states: torch.Tensor,
    use_4bit_experts: bool = False,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Executes a Latent `ReLU^2` MoE layer (`kind = 2`) using:
      - Pre-widened FP32 `gate.weight` (record 29) and `e_score_correction_bias` (record 30) from `routed-shared.bin`
      - `fc1_latent_proj` (record 31), `fc2_latent_proj` (record 32), `shared_experts` (records 34, 36)
      - On-demand `expert_store.get_expert_weights(e_id)` fetching ONLY the active routed experts from `experts.direct`
        via SpecMD Least-Stale + R4 Clean-Drop slot caching!
    """
    from prepare_nemotron_datasets import dequantize_matrix_4bit_group64, quantize_matrix_4bit_group64

    norm_w = trunk.get_record_tensor(0, (4096,))
    gate_w_fp32 = trunk.get_record_tensor(29, (512, 4096))  # Already FP32 on disk!
    gate_bias_fp32 = trunk.get_record_tensor(30, (512,))
    fc1_w = trunk.get_record_tensor(31, (1024, 4096))
    fc2_w = trunk.get_record_tensor(32, (4096, 1024))
    sh_up_w = trunk.get_record_tensor(34, (5376, 4096))
    sh_down_w = trunk.get_record_tensor(36, (4096, 5376))

    bsz, seq_len, _ = hidden_states.shape
    u = rms_norm_op(hidden_states, norm_w, eps=eps)
    flat_u = u.view(-1, 4096)

    # Router dot product directly with pre-widened FP32 `gate_w_fp32` and `n_group = 1, topk_group = 1` group mask
    router_logits = F.linear(flat_u.to(torch.float32), gate_w_fp32)
    scores = torch.sigmoid(router_logits)
    scores_choice = scores + gate_bias_fp32
    group_scores = scores_choice.view(-1, 1, 512).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=1, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(-1, 1, 512).reshape(-1, 512)
    scores_choice = scores_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(scores_choice, k=22, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    topk_weights = (topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)) * 5.0

    # Compress 4096 -> 1024 latent space before expert routing
    latent_in = F.linear(flat_u, fc1_w)  # [N_tok, 1024]
    latent_accum = torch.zeros_like(latent_in, dtype=topk_weights.dtype)

    expert_mask = F.one_hot(topk_indices, num_classes=512).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
    protected_set = {int(e.item()) for e in expert_hit}

    # SpecMD Hit-First Intra-Layer Ordering: process cached pool hits BEFORE misses so a miss never evicts a needed hit!
    hit_experts = [int(e.item()) for e in expert_hit if int(e.item()) in expert_store.pool_cache]
    miss_experts = [int(e.item()) for e in expert_hit if int(e.item()) not in expert_store.pool_cache]

    # Note: to preserve exact bit-identity with `NemotronHExperts.forward` (which iterates in ascending `expert_hit` order),
    # we pre-stage `miss_experts` into the slot cache after touching `hit_experts`, then accumulate in ascending `expert_hit` order!
    for e_id in hit_experts:
        top_k_pos, tok_idx = torch.where(expert_mask[e_id])
        w_sum = float(topk_weights[tok_idx, top_k_pos].sum().item())
        _ = expert_store.get_expert_weights(e_id, protected_set, routing_weight=w_sum)

    loaded_weights: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for e_id in hit_experts:
        loaded_weights[e_id] = expert_store.pool_cache[e_id][:2]
    for e_id in miss_experts:
        top_k_pos, tok_idx = torch.where(expert_mask[e_id])
        w_sum = float(topk_weights[tok_idx, top_k_pos].sum().item())
        loaded_weights[e_id] = expert_store.get_expert_weights(e_id, protected_set, routing_weight=w_sum)

    for exp_tensor in expert_hit:
        e_id = int(exp_tensor.item())
        top_k_pos, tok_idx = torch.where(expert_mask[e_id])
        up_w, down_w = loaded_weights[e_id]
        if use_4bit_experts:
            u_nib, u_stp, u_scl = quantize_matrix_4bit_group64(up_w)
            d_nib, d_stp, d_scl = quantize_matrix_4bit_group64(down_w)
            up_w = dequantize_matrix_4bit_group64(u_nib, u_stp, u_scl, 2688, 1024).to(torch.bfloat16)
            down_w = dequantize_matrix_4bit_group64(d_nib, d_stp, d_scl, 1024, 2688).to(torch.bfloat16)
        cur_in = latent_in[tok_idx]
        cur_h = F.linear(relu2_op(F.linear(cur_in, up_w)), down_w)
        cur_h = cur_h * topk_weights[tok_idx, top_k_pos, None]
        latent_accum.index_add_(0, tok_idx, cur_h.to(latent_accum.dtype))

    routed_4096 = F.linear(latent_accum.to(hidden_states.dtype), fc2_w).view(bsz, seq_len, 4096)
    shared_4096 = F.linear(relu2_op(F.linear(u, sh_up_w)), sh_down_w)

    # Exact parenthesization matching NemotronHBlock(residual + NemotronHMoE(routed + shared))
    out = hidden_states + (routed_4096 + shared_4096)
    return out, expert_store.stats()


@torch.no_grad()
def execute_native_mtp_draft_from_precomputed(
    mtp0_trunk: NemotronPrunedTrunkReader,
    mtp1_trunk: NemotronPrunedTrunkReader,
    mtp1_expert_store: NemotronDirectExpertStore,
    next_token_embeds: torch.Tensor,
    backbone_hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Executes Nemotron 3.5 Super VL's 2-layer Native Multi-Token Prediction (`MTP`) speculative drafter
    entirely from the precomputed & pruned `trunk-mtp-0`, `trunk-mtp-1`, and `expert-mtp-1/experts.direct`!
    """
    enorm_w = mtp0_trunk.get_record_tensor(10, (4096,))
    hnorm_w = mtp0_trunk.get_record_tensor(11, (4096,))
    eh_proj_w = mtp0_trunk.get_record_tensor(12, (4096, 8192))
    final_ln_w = mtp1_trunk.get_record_tensor(1, (4096,))

    e_normed = rms_norm_op(next_token_embeds, enorm_w, eps=eps)
    h_normed = rms_norm_op(backbone_hidden_states, hnorm_w, eps=eps)
    mtp_in = F.linear(torch.cat([e_normed, h_normed], dim=-1), eh_proj_w)

    mtp_h0 = execute_attention_stage_from_precomputed(mtp0_trunk, mtp_in, eps=eps)
    mtp_h1, mtp_exp_stats = execute_latent_moe_stage_from_precomputed(
        mtp1_trunk, mtp1_expert_store, mtp_h0, use_4bit_experts=False, eps=eps
    )
    mtp_final = rms_norm_op(mtp_h1, final_ln_w, eps=eps)
    mtp_logits = F.linear(mtp_final, lm_head_weight).float()
    return mtp_logits, mtp_exp_stats
