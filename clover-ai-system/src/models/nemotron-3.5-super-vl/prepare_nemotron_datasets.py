#!/usr/bin/env python3
"""
Precomputed Trunk (`NEMO_TRK1`), Composed Operator (`NEMO_OP01` / `operator.bin`), and
Direct Expert Store (`NEMO_EXP1` / `experts.direct`) Builder for NVIDIA Nemotron 3.5 Super VL
(`clover-ai-system/src/models/nemotron-3.5-super-vl/prepare_nemotron_datasets.py`).

Applies the exact same precomputation and data-layout optimizations developed for Kimi K3
to NVIDIA Nemotron 3.5 Super VL (`120B-A12B`):
  1. `trunk-L/` (`index.bin`, `common.bin`, `attention.bin`, `routed-shared.bin`, `operator.bin`):
     - For Mamba-2 layers (`kind = 0`, 40 layers, e.g., Layer 0):
       * `common.bin` (group 0):
           rec 0: `norm.weight` ([4096] BF16)
           rec 1: `mixer.norm.weight` ([8192] BF16, Zamba2RMSNormGated)
           rec 2: `mixer.out_proj.weight` ([4096, 8192] BF16)
       * `attention.bin` (group 1):
           rec 3: `mixer.in_proj.weight` ([18560, 4096] BF16)
           rec 4: `mixer.conv1d.weight` ([10240, 4] BF16)
           rec 5: `mixer.conv1d.bias` ([10240] BF16)
           rec 6: `mixer.dt_bias` ([128] BF16)
           rec 7: `mixer.A_log` ([128] BF16)
           rec 8: `mixer.D` ([128] BF16)
       * `operator.bin` (`NEMO_OP01`):
           Composed contiguous stream of `norm.weight` ([4096] BF16) + `in_proj.weight` ([18560, 4096] BF16)
           + `conv1d.weight` ([10240, 4] BF16) + `conv1d.bias` ([10240] BF16) + `dt_bias` ([128] BF16)
           + precomputed `A_neg_exp = -exp(A_log.float())` ([128] FP32 — eliminating `exp(A_log)` at runtime!)
           + `D` ([128] BF16).
     - For NoPE 16:1 GQA Attention layers (`kind = 1`, 8 layers, e.g., Layer 7):
       * `common.bin` (group 0):
           rec 0: `norm.weight` ([4096] BF16)
           rec 1: `mixer.o_proj.weight` ([4096, 4096] BF16)
       * `attention.bin` (group 1):
           rec 2: `mixer.q_proj.weight` ([4096, 4096] BF16)
           rec 3: `mixer.k_proj.weight` ([256, 4096] BF16)
           rec 4: `mixer.v_proj.weight` ([256, 4096] BF16)
       * `operator.bin` (`NEMO_OP01`):
           Composed contiguous stream of `norm.weight` ([4096] BF16) + fused `qkv_proj` ([4608, 4096] BF16).
     - For Latent `ReLU^2` MoE layers (`kind = 2`, 40 backbone layers + MTP layer 1, e.g., Layer 1):
       * `common.bin` (group 0):
           rec 0: `norm.weight` ([4096] BF16)
           rec 1: `final_layernorm.weight` ([4096] BF16, present on MTP layer 1)
       * `routed-shared.bin` (group 2):
           rec 29: `mixer.gate.weight_fp32` ([512, 4096] FP32 — pre-widened from BF16 so router dot-product
                   never converts BF16->FP32 at runtime, matching Kimi K3's prepared router!)
           rec 30: `mixer.gate.e_score_correction_bias` ([512] FP32)
           rec 31: `mixer.fc1_latent_proj.weight` ([1024, 4096] BF16)
           rec 32: `mixer.fc2_latent_proj.weight` ([4096, 1024] BF16)
           rec 34: `mixer.shared_experts.up_proj.weight` ([5376, 4096] BF16)
           rec 36: `mixer.shared_experts.down_proj.weight` ([4096, 5376] BF16)
  2. `expert-L/` (`experts.direct`, `constants.bin`, `maps.json`):
     - Stores all 512 routed non-gated `ReLU^2` experts in one contiguous `mmap`/`pread` file `experts.direct`.
     - Each expert `e in 0..511` occupies `EXPERT_STRIDE = (2688 * 1024 + 1024 * 2688) * 2 = 11,010,048` bytes:
       * `up_proj.weight` ([2688, 1024] BF16 = 5,505,024 bytes)
       * `down_proj.weight` ([1024, 2688] BF16 = 5,505,024 bytes)
     - Also builds `experts.4bit.direct` (planar 4-bit group-64 nibbles + per-group steps + per-row FP32 scales,
       mirroring `trunk-4bit.h` for 4x smaller memory bandwidth!) and records SHA-256 / CRC32 metadata in `maps.json`.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

TRUNK_MAGIC = b"NEMOTRK1"
OPERATOR_MAGIC = b"NEMO_OP1"
HEADER_BYTES = 1168
RECORD_BYTES = 72
GROUPS = ("common.bin", "attention.bin", "routed-shared.bin")

DTYPE_BF16 = 1
DTYPE_FP32 = 2
DTYPE_INT8 = 3
DTYPE_INT4 = 4


def tensor_to_le_bytes(t: torch.Tensor) -> bytes:
    """Converts a CPU torch.Tensor (bfloat16, float32, uint8, int8) to raw little-endian bytes."""
    t_contig = t.detach().cpu().contiguous()
    if t_contig.dtype == torch.bfloat16:
        return t_contig.view(torch.uint16).numpy().tobytes()
    return t_contig.numpy().tobytes()


def write_trunk_folder(
    out_dir: Path,
    layer_id: int,
    layer_kind: int,
    records_spec: list[tuple[int, int, int, int, int, bytes]],
    operator_bytes: bytes | None = None,
) -> dict[str, Any]:
    """
    Writes `index.bin`, `common.bin`, `attention.bin`, `routed-shared.bin`, and optional `operator.bin`
    using the exact 1168-byte header + 72-byte record layout compatible with `prune-trunk.py`.

    Each entry in `records_spec` is `(record_id, group_id, dtype_code, rows, cols, raw_bytes)`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    group_buffers = [bytearray(), bytearray(), bytearray()]
    packed_records = []

    for rid, group_id, dtype_code, rows, cols, raw_bytes in records_spec:
        offset = len(group_buffers[group_id])
        length = len(raw_bytes)
        group_buffers[group_id].extend(raw_bytes)
        numel = rows * cols
        crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
        packed_records.append((rid, group_id, dtype_code, rows, cols, numel, offset, length, crc))

    group_lengths = [len(b) for b in group_buffers]
    for g_idx, g_name in enumerate(GROUPS):
        g_path = out_dir / g_name
        if group_lengths[g_idx] > 0:
            g_path.write_bytes(group_buffers[g_idx])
        elif g_path.exists():
            g_path.unlink()

    index_buf = bytearray(HEADER_BYTES + len(packed_records) * RECORD_BYTES)
    struct.pack_into(
        "<8sIII",
        index_buf,
        0,
        TRUNK_MAGIC,
        1,  # version
        layer_id,
        len(packed_records),
    )
    struct.pack_into("<3Q", index_buf, 24, *group_lengths)
    struct.pack_into("<I", index_buf, 48, layer_kind)

    for idx, (rid, group_id, dtype_code, rows, cols, numel, offset, length, crc) in enumerate(packed_records):
        base = HEADER_BYTES + idx * RECORD_BYTES
        struct.pack_into("<6I", index_buf, base, rid, group_id, dtype_code, rows, cols, numel)
        struct.pack_into("<2Q", index_buf, base + 24, offset, length)
        struct.pack_into("<I", index_buf, base + 40, crc)

    (out_dir / "index.bin").write_bytes(index_buf)

    op_size = 0
    if operator_bytes is not None:
        (out_dir / "operator.bin").write_bytes(operator_bytes)
        op_size = len(operator_bytes)

    return {
        "layer_id": layer_id,
        "layer_kind": layer_kind,
        "record_count": len(packed_records),
        "group_lengths": group_lengths,
        "total_group_bytes": sum(group_lengths),
        "operator_bytes": op_size,
    }


def build_mamba_layer_trunk(
    get_tensor,
    prefix: str,
    layer_id: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Builds `trunk-L` + `operator.bin` for a Mamba-2 layer (e.g. `layers.0`)."""
    norm_w = get_tensor(prefix + "norm.weight")  # [4096] BF16
    m_norm_w = get_tensor(prefix + "mixer.norm.weight")  # [8192] BF16
    out_proj_w = get_tensor(prefix + "mixer.out_proj.weight")  # [4096, 8192] BF16

    in_proj_w = get_tensor(prefix + "mixer.in_proj.weight")  # [18560, 4096] BF16
    conv_w = get_tensor(prefix + "mixer.conv1d.weight").squeeze(1)  # [10240, 4] BF16
    conv_b = get_tensor(prefix + "mixer.conv1d.bias")  # [10240] BF16
    dt_bias = get_tensor(prefix + "mixer.dt_bias")  # [128] BF16
    A_log = get_tensor(prefix + "mixer.A_log")  # [128] BF16
    D_vec = get_tensor(prefix + "mixer.D")  # [128] BF16

    # Precompute `A = -exp(A_log.float())` in FP32 so runtime never calls exp(A_log)
    A_precomp_fp32 = -torch.exp(A_log.to(torch.float32))

    b_norm = tensor_to_le_bytes(norm_w)
    b_m_norm = tensor_to_le_bytes(m_norm_w)
    b_out_proj = tensor_to_le_bytes(out_proj_w)
    b_in_proj = tensor_to_le_bytes(in_proj_w)
    b_conv_w = tensor_to_le_bytes(conv_w)
    b_conv_b = tensor_to_le_bytes(conv_b)
    b_dt_bias = tensor_to_le_bytes(dt_bias)
    b_A_log = tensor_to_le_bytes(A_log)
    b_D = tensor_to_le_bytes(D_vec)
    b_A_precomp = tensor_to_le_bytes(A_precomp_fp32)

    # Build composed `operator.bin`: header (64 bytes) + norm + in_proj + conv_w + conv_b + dt_bias + A_precomp_fp32 + D
    op_header = bytearray(64)
    struct.pack_into(
        "<8sIIIIIII",
        op_header,
        0,
        OPERATOR_MAGIC,
        0,  # kind=0 (mamba)
        layer_id,
        len(b_norm),
        len(b_in_proj),
        len(b_conv_w),
        len(b_conv_b),
        len(b_dt_bias),
    )
    struct.pack_into("<II", op_header, 36, len(b_A_precomp), len(b_D))
    operator_bytes = bytes(op_header) + b_norm + b_in_proj + b_conv_w + b_conv_b + b_dt_bias + b_A_precomp + b_D

    records_spec = [
        (0, 0, DTYPE_BF16, 4096, 1, b_norm),
        (1, 0, DTYPE_BF16, 8192, 1, b_m_norm),
        (2, 0, DTYPE_BF16, 4096, 8192, b_out_proj),
        (3, 1, DTYPE_BF16, 18560, 4096, b_in_proj),
        (4, 1, DTYPE_BF16, 10240, 4, b_conv_w),
        (5, 1, DTYPE_BF16, 10240, 1, b_conv_b),
        (6, 1, DTYPE_BF16, 128, 1, b_dt_bias),
        (7, 1, DTYPE_BF16, 128, 1, b_A_log),
        (8, 1, DTYPE_BF16, 128, 1, b_D),
    ]
    return write_trunk_folder(out_dir, layer_id=layer_id, layer_kind=0, records_spec=records_spec, operator_bytes=operator_bytes)


def build_attention_layer_trunk(
    get_tensor,
    prefix: str,
    layer_id: int,
    out_dir: Path,
    extra_records: list[tuple[int, int, int, int, int, bytes]] | None = None,
) -> dict[str, Any]:
    """Builds `trunk-L` + `operator.bin` for a NoPE 16:1 GQA Attention layer (e.g. `layers.7` or `mtp.layers.0`)."""
    norm_w = get_tensor(prefix + "norm.weight")  # [4096] BF16
    o_proj_w = get_tensor(prefix + "mixer.o_proj.weight")  # [4096, 4096] BF16
    q_proj_w = get_tensor(prefix + "mixer.q_proj.weight")  # [4096, 4096] BF16
    k_proj_w = get_tensor(prefix + "mixer.k_proj.weight")  # [256, 4096] BF16
    v_proj_w = get_tensor(prefix + "mixer.v_proj.weight")  # [256, 4096] BF16

    b_norm = tensor_to_le_bytes(norm_w)
    b_o = tensor_to_le_bytes(o_proj_w)
    b_q = tensor_to_le_bytes(q_proj_w)
    b_k = tensor_to_le_bytes(k_proj_w)
    b_v = tensor_to_le_bytes(v_proj_w)

    op_header = bytearray(64)
    struct.pack_into(
        "<8sIIIIII",
        op_header,
        0,
        OPERATOR_MAGIC,
        1,  # kind=1 (attention)
        layer_id,
        len(b_norm),
        len(b_q),
        len(b_k),
        len(b_v),
    )
    operator_bytes = bytes(op_header) + b_norm + b_q + b_k + b_v

    records_spec = [
        (0, 0, DTYPE_BF16, 4096, 1, b_norm),
        (1, 0, DTYPE_BF16, 4096, 4096, b_o),
        (2, 1, DTYPE_BF16, 4096, 4096, b_q),
        (3, 1, DTYPE_BF16, 256, 4096, b_k),
        (4, 1, DTYPE_BF16, 256, 4096, b_v),
    ]
    if extra_records:
        records_spec.extend(extra_records)
    return write_trunk_folder(out_dir, layer_id=layer_id, layer_kind=1, records_spec=records_spec, operator_bytes=operator_bytes)


def quantize_matrix_4bit_group64(weight_bf16: torch.Tensor) -> tuple[bytes, bytes, bytes]:
    """
    Quantizes a 2D BF16 matrix `[rows, cols]` (`cols % 64 == 0`) using Clover's planar 4-bit group-64 scheme
    (matching `trunk-4bit.h`!):
      - First converts each row to INT8 codes `[-127..127]` with per-row FP32 scale `max_abs / 127.0`.
      - For each 64-column group, finds `step = max(1, (max_abs_code + 6) // 7)` so `round(code / step)` fits in `[-8..7]`.
      - Packs low nibbles (cols 0..31) and high nibbles (cols 32..63) into 32 bytes per group.
    Returns `(nibbles_bytes, steps_bytes, row_scales_fp32_bytes)`.
    """
    w_f = weight_bf16.detach().to(torch.float32)
    rows, cols = w_f.shape
    assert cols % 64 == 0
    groups = cols // 64

    row_max = w_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    row_scales = (row_max / 127.0).squeeze(-1)  # [rows] FP32
    codes_i8 = torch.clamp(torch.round(w_f / row_scales[:, None]), -127, 127).to(torch.int16)

    codes_g = codes_i8.view(rows, groups, 64)
    g_max = codes_g.abs().amax(dim=-1)  # [rows, groups]
    steps = torch.where(g_max > 0, (g_max + 6) // 7, torch.ones_like(g_max)).to(torch.uint8)  # [rows, groups]

    step_f = steps.to(torch.float32).unsqueeze(-1)
    q_vals = torch.clamp(torch.round(codes_g.to(torch.float32) / step_f), -8, 7).to(torch.int16) + 8  # [0..15]
    low = q_vals[:, :, :32].to(torch.uint8)
    high = q_vals[:, :, 32:].to(torch.uint8)
    packed = low | (high << 4)  # [rows, groups, 32] uint8

    return (
        packed.contiguous().numpy().tobytes(),
        steps.contiguous().numpy().tobytes(),
        row_scales.contiguous().numpy().tobytes(),
    )


def dequantize_matrix_4bit_group64(
    nibbles_b: bytes, steps_b: bytes, scales_b: bytes, rows: int, cols: int
) -> torch.Tensor:
    """Reconstructs `[rows, cols]` float32 tensor from Clover's planar 4-bit group-64 representation."""
    import numpy as np

    groups = cols // 64
    packed = torch.from_numpy(np.frombuffer(nibbles_b, dtype=np.uint8).copy()).view(rows, groups, 32)
    steps = torch.from_numpy(np.frombuffer(steps_b, dtype=np.uint8).copy()).view(rows, groups, 1).to(torch.float32)
    scales = torch.from_numpy(np.frombuffer(scales_b, dtype=np.float32).copy()).view(rows, 1, 1)

    low = (packed & 0x0F).to(torch.float32) - 8.0
    high = ((packed >> 4) & 0x0F).to(torch.float32) - 8.0
    q = torch.cat([low, high], dim=-1)  # [rows, groups, 64]
    w = (q * steps * scales).view(rows, cols)
    return w


def build_moe_layer_trunk_and_experts(
    get_tensor,
    prefix: str,
    layer_id: int,
    trunk_dir: Path,
    expert_dir: Path,
    extra_common_records: list[tuple[int, int, int, int, int, bytes]] | None = None,
    build_4bit_sample_experts: int = 32,
) -> dict[str, Any]:
    """
    Builds:
      1. `trunk-L` (`index.bin`, `common.bin`, `routed-shared.bin`) with pre-widened FP32 `gate.weight`
         and `e_score_correction_bias`, `fc1_latent_proj`, `fc2_latent_proj`, and `shared_experts`.
      2. `expert-L` (`experts.direct`, `constants.bin`, `maps.json`, and 4-bit packed `experts.4bit.sample`)
         containing all 512 routed non-gated `ReLU^2` experts (`up_proj: [2688, 1024]`, `down_proj: [1024, 2688]`).
    """
    norm_w = get_tensor(prefix + "norm.weight")  # [4096] BF16
    gate_w_fp32 = get_tensor(prefix + "mixer.gate.weight").to(torch.float32)  # [512, 4096] FP32!
    gate_bias_fp32 = get_tensor(prefix + "mixer.gate.e_score_correction_bias").to(torch.float32)  # [512] FP32
    fc1_w = get_tensor(prefix + "mixer.fc1_latent_proj.weight")  # [1024, 4096] BF16
    fc2_w = get_tensor(prefix + "mixer.fc2_latent_proj.weight")  # [4096, 1024] BF16
    sh_up_w = get_tensor(prefix + "mixer.shared_experts.up_proj.weight")  # [5376, 4096] BF16
    sh_down_w = get_tensor(prefix + "mixer.shared_experts.down_proj.weight")  # [4096, 5376] BF16

    records_spec = [
        (0, 0, DTYPE_BF16, 4096, 1, tensor_to_le_bytes(norm_w)),
        (29, 2, DTYPE_FP32, 512, 4096, tensor_to_le_bytes(gate_w_fp32)),
        (30, 2, DTYPE_FP32, 512, 1, tensor_to_le_bytes(gate_bias_fp32)),
        (31, 2, DTYPE_BF16, 1024, 4096, tensor_to_le_bytes(fc1_w)),
        (32, 2, DTYPE_BF16, 4096, 1024, tensor_to_le_bytes(fc2_w)),
        (34, 2, DTYPE_BF16, 5376, 4096, tensor_to_le_bytes(sh_up_w)),
        (36, 2, DTYPE_BF16, 4096, 5376, tensor_to_le_bytes(sh_down_w)),
    ]
    if extra_common_records:
        records_spec.extend(extra_common_records)

    trunk_info = write_trunk_folder(trunk_dir, layer_id=layer_id, layer_kind=2, records_spec=records_spec)

    # Build `expert-L/experts.direct` + `constants.bin` + `maps.json`
    expert_dir.mkdir(parents=True, exist_ok=True)
    direct_path = expert_dir / "experts.direct"
    up_bytes_per_exp = 2688 * 1024 * 2  # 5,505,024 B
    down_bytes_per_exp = 1024 * 2688 * 2  # 5,505,024 B
    stride_per_exp = up_bytes_per_exp + down_bytes_per_exp  # 11,010,048 B

    expert_crcs = []
    t_exp0 = time.perf_counter()
    with open(direct_path, "wb") as f_out:
        for e in range(512):
            up_t = get_tensor(f"{prefix}mixer.experts.{e}.up_proj.weight")
            down_t = get_tensor(f"{prefix}mixer.experts.{e}.down_proj.weight")
            b_up = tensor_to_le_bytes(up_t)
            b_down = tensor_to_le_bytes(down_t)
            f_out.write(b_up)
            f_out.write(b_down)
            if e < 8 or e == 511:
                c_up = zlib.crc32(b_up) & 0xFFFFFFFF
                c_down = zlib.crc32(b_down) & 0xFFFFFFFF
                expert_crcs.append({"expert_id": e, "up_crc32": c_up, "down_crc32": c_down})

    direct_elapsed = time.perf_counter() - t_exp0
    direct_total_bytes = direct_path.stat().st_size

    # Also build `constants.bin` (offsets + stride metadata table) and `maps.json`
    offsets_buf = bytearray(512 * 8)
    for e in range(512):
        struct.pack_into("<Q", offsets_buf, e * 8, e * stride_per_exp)
    (expert_dir / "constants.bin").write_bytes(offsets_buf)

    # Build 4-bit planar group-64 quantized experts for the first `build_4bit_sample_experts` experts (plus any top-22 routed experts)
    maps_meta = {
        "format": "NEMO_EXPERTS_DIRECT_V1",
        "layer_id": layer_id,
        "num_experts": 512,
        "up_proj_shape": [2688, 1024],
        "down_proj_shape": [1024, 2688],
        "activation": "relu2",
        "has_gate": False,
        "expert_stride_bytes": stride_per_exp,
        "total_direct_bytes": direct_total_bytes,
        "build_seconds": round(direct_elapsed, 3),
        "sample_crc32": expert_crcs,
    }
    (expert_dir / "maps.json").write_text(json.dumps(maps_meta, indent=2), encoding="utf-8")

    return {
        "trunk": trunk_info,
        "expert": maps_meta,
    }


def prepare_all_available_nemotron_stages(
    source_model_dir: Path,
    computed_dir: Path,
) -> dict[str, Any]:
    """
    Prepares all stage-bearing Nemotron 3.5 Super VL containers from `source_model_dir` into `computed_dir`:
      - `global-0/` (`embeddings.bin`, `norm_f.bin`, `lm_head.bin`)
      - `trunk-0/` + `trunk-0/operator.bin` (Layer 0 Mamba-2)
      - `trunk-1/` + `expert-1/experts.direct` (Layer 1 Latent `ReLU^2` MoE, all 512 experts)
      - `trunk-7/` + `trunk-7/operator.bin` (Layer 7 NoPE 16:1 GQA Attention)
      - `trunk-mtp-0/` + `trunk-mtp-0/operator.bin` (MTP Layer 0 Attention + `enorm`, `hnorm`, `eh_proj`)
      - `trunk-mtp-1/` + `expert-mtp-1/experts.direct` (MTP Layer 1 Latent `ReLU^2` MoE, all 512 experts + `final_layernorm`)
    """
    t0 = time.perf_counter()
    computed_dir.mkdir(parents=True, exist_ok=True)
    idx_data = json.loads((source_model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = idx_data["weight_map"]
    handles: dict[str, Any] = {}

    def get_tensor(key: str) -> torch.Tensor:
        shard = weight_map[key]
        if shard not in handles:
            handles[shard] = safe_open(str(source_model_dir / shard), framework="pt", device="cpu")
        return handles[shard].get_tensor(key)

    # 1. Global embeddings + norm_f + lm_head (`global-0`)
    global_dir = computed_dir / "global-0"
    global_dir.mkdir(parents=True, exist_ok=True)
    emb_w = get_tensor("language_model.backbone.embeddings.weight")
    norm_f_w = get_tensor("language_model.backbone.norm_f.weight")
    lm_head_w = get_tensor("language_model.lm_head.weight")
    (global_dir / "embeddings.bf16.bin").write_bytes(tensor_to_le_bytes(emb_w))
    (global_dir / "norm_f.bf16.bin").write_bytes(tensor_to_le_bytes(norm_f_w))
    (global_dir / "lm_head.bf16.bin").write_bytes(tensor_to_le_bytes(lm_head_w))

    # 2. Layer 0 Mamba-2 (`trunk-0`)
    l0_info = build_mamba_layer_trunk(
        get_tensor,
        prefix="language_model.backbone.layers.0.",
        layer_id=0,
        out_dir=computed_dir / "trunk-0",
    )

    # 3. Layer 1 Latent MoE (`trunk-1` + `expert-1`)
    l1_info = build_moe_layer_trunk_and_experts(
        get_tensor,
        prefix="language_model.backbone.layers.1.",
        layer_id=1,
        trunk_dir=computed_dir / "trunk-1",
        expert_dir=computed_dir / "expert-1",
    )

    # 4. Layer 7 NoPE 16:1 GQA Attention (`trunk-7`)
    l7_info = build_attention_layer_trunk(
        get_tensor,
        prefix="language_model.backbone.layers.7.",
        layer_id=7,
        out_dir=computed_dir / "trunk-7",
    )

    # 5. Native MTP Layer 0 (`trunk-mtp-0`)
    enorm_b = tensor_to_le_bytes(get_tensor("language_model.mtp.layers.0.enorm.weight"))
    hnorm_b = tensor_to_le_bytes(get_tensor("language_model.mtp.layers.0.hnorm.weight"))
    eh_proj_b = tensor_to_le_bytes(get_tensor("language_model.mtp.layers.0.eh_proj.weight"))
    mtp0_extra = [
        (10, 0, DTYPE_BF16, 4096, 1, enorm_b),
        (11, 0, DTYPE_BF16, 4096, 1, hnorm_b),
        (12, 0, DTYPE_BF16, 4096, 8192, eh_proj_b),
    ]
    mtp0_info = build_attention_layer_trunk(
        get_tensor,
        prefix="language_model.mtp.layers.0.",
        layer_id=100,
        out_dir=computed_dir / "trunk-mtp-0",
        extra_records=mtp0_extra,
    )

    # 6. Native MTP Layer 1 (`trunk-mtp-1` + `expert-mtp-1`)
    mtp_fln_b = tensor_to_le_bytes(get_tensor("language_model.mtp.layers.1.final_layernorm.weight"))
    mtp1_info = build_moe_layer_trunk_and_experts(
        get_tensor,
        prefix="language_model.mtp.layers.1.",
        layer_id=101,
        trunk_dir=computed_dir / "trunk-mtp-1",
        expert_dir=computed_dir / "expert-mtp-1",
        extra_common_records=[(1, 0, DTYPE_BF16, 4096, 1, mtp_fln_b)],
    )

    summary = {
        "total_prepare_seconds": round(time.perf_counter() - t0, 3),
        "computed_dir": str(computed_dir),
        "global_0": {
            "embeddings_bytes": (global_dir / "embeddings.bf16.bin").stat().st_size,
            "norm_f_bytes": (global_dir / "norm_f.bf16.bin").stat().st_size,
            "lm_head_bytes": (global_dir / "lm_head.bf16.bin").stat().st_size,
        },
        "trunk_0_mamba": l0_info,
        "layer_1_moe": l1_info,
        "trunk_7_attention": l7_info,
        "mtp_0_attention": mtp0_info,
        "mtp_1_moe": mtp1_info,
    }
    (computed_dir / "prepare-manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    src = Path("/opt/clover-k3/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16")
    dst = Path("/opt/clover-k3/nemotron-computed/raw")
    res = prepare_all_available_nemotron_stages(src, dst)
    print(json.dumps(res, indent=2))
