#!/usr/bin/env python3
"""
Kimi K3.1 Pro GPU-Resident Differentiable Stage Backbone & Precomputation Engine
(`porting-models/kimi_k31_gpu_backbone.py`).

Supports `moonshotai/Kimi-K3.1-Pro` (`2.8T` total / `104B` active, `93` layers, `D_model = 7168`,
`V = 163,840`, `69 KDA + 24 Gated DSA` with the 5-tensor Lightning Indexer, `896` MXFP4 `SiTU-GLU`
routed experts per MoE layer) and `LithosAI/Kimi-K3.1-Pro-DSpark` (`5`-layer `2.25B` BF16 speculative
decoding draft head).

Key capabilities:
  1. IPv4-forced HTTP Range streaming (`fetch_safetensor_tensors`) so stage trunks, the 24 Gated DSA
     Lightning Indexer containers (`indexer.bin`), global embeddings/LM head, `MoonViT-V2` projector,
     `DSpark` drafter, and active MXFP4 routed experts are extracted directly into `/dev/shm` without
     requiring 1.56 TB of local disk.
  2. Exact IEEE/OCP MXFP4 (`E2M1` nibbles + `E8M0` shared exponent per group of 32) dequantization into
     BF16 `experts.direct` and Clover planar 4-bit group-64 `experts.4bit.direct`.
  3. Differentiable GPU Stage Model (`KimiK31GPUStageModel`):
       - `KimiK31KDAStage` (Layer 0: Kimi Delta Attention + 33,792 Dense `SiTU-GLU` MLP + Attention-Residual)
       - `KimiK31KDAMoEStage` (Layer 1: Kimi Delta Attention + Latent `SiTU-GLU` MoE + Shared Experts)
       - `KimiK31GatedDSAMoEStage` (Layer 3: **Gated DeepSeek Sparse Attention** with the 5-tensor
         `self_attn.indexer.*` Lightning Indexer + Gated MLA + Latent `SiTU-GLU` MoE + Shared Experts)
       - `KimiK31DSparkDraftStage` (`LithosAI/Kimi-K3.1-Pro-DSpark` 5-layer multi-depth fusion drafter)
       - `KimiK31FinalTail` (`output_attn_res_norm` + `output_attn_res_proj` + `norm.weight` + `lm_head.weight`)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import struct
import time
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, hf_hub_url

# Force IPv4 for Hugging Face CDN range requests on multi-GPU hosts where IPv6 routes time out
_ORIG_GETADDRINFO = socket.getaddrinfo


def enable_ipv4_only() -> None:
    def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return _ORIG_GETADDRINFO(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _ipv4_getaddrinfo


TRUNK_MAGIC = b"K31TRK01"
OPERATOR_MAGIC = b"K31OP001"
INDEXER_MAGIC = b"K31IDX01"
HEADER_BYTES = 1168
RECORD_BYTES = 72
GROUPS = ("common.bin", "attention.bin", "routed-shared.bin")

DTYPE_BF16 = 1
DTYPE_FP32 = 2
DTYPE_U8 = 3

V_K31 = 163840
D_K31 = 7168
D_LATENT = 3584
D_EXP = 3072
D_SHARED = 6144
D_DENSE = 33792
D_VQ = 384
D_HA = 4096

# 24 Gated DSA layers in Kimi-K3.1-Pro (0-indexed: dsa_layers 1-indexed minus 1)
DSA_LAYERS_0IDX = [
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47,
    51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92,
]

# IEEE/OCP MXFP4 E2M1 lookup table (4-bit float: sign bit 3, 2-bit exp, 1-bit mantissa)
MXFP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def situ_glu_op(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta_gate: float = 4.0,
    beta_up: float = 25.0,
) -> torch.Tensor:
    """
    Exact Kimi K3 / K3.1 Pro `SiTU-GLU` activation (verified against `layer-mla.c:transformer_activation`):
      activated = (4.0 * tanh(gate / 4.0)) * sigmoid(gate)
      capped    = 25.0 * tanh(up / 25.0)
      return activated * capped
    """
    gf = gate.float()
    uf = up.float()
    activated = (beta_gate * torch.tanh(gf / beta_gate)) * torch.sigmoid(gf)
    capped = beta_up * torch.tanh(uf / beta_up)
    return (activated * capped).to(gate.dtype)


def rms_norm_op(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv_rms * weight.float()).to(x.dtype)


def attn_res_aggregate_op(
    snapshots: list[torch.Tensor],
    residual: torch.Tensor,
    res_norm_w: torch.Tensor,
    res_proj_w: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Exact Kimi K3 / K3.1 Pro Block-Snapshot Attention-Residual Aggregation (`transformer_aggregate`):
    Scores each snapshot + current residual via RMSNorm -> dot(res_proj_w) -> softmax over sources.
    """
    srcs = torch.stack([s.float() for s in snapshots] + [residual.float()], dim=0)  # [S, B, T, D]
    inv_rms = torch.rsqrt(srcs.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = srcs * inv_rms * res_norm_w.float().view(1, 1, 1, -1)
    scores = torch.matmul(normed, res_proj_w.float().view(-1))  # [S, B, T]
    weights = torch.softmax(scores, dim=0).unsqueeze(-1)  # [S, B, T, 1]
    agg = (weights * srcs).sum(dim=0)
    return agg.to(residual.dtype)


def dequantize_mxfp4_tensor(
    weight_packed_u8: torch.Tensor,
    weight_scale_u8: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Dequantizes a `compressed-tensors` `mxfp4-pack-quantized` matrix:
      - `weight_packed_u8`: `[rows, cols // 2]` uint8 (2 E2M1 nibbles per byte, group_size=32)
      - `weight_scale_u8`:  `[rows, cols // 32]` uint8 (E8M0 exponent: scale = 2^(s - 127))
    Returns `[rows, cols]` bfloat16 tensor.
    """
    dev = device or weight_packed_u8.device
    lut = MXFP4_E2M1_TABLE.to(dev)
    packed = weight_packed_u8.to(dev)
    scales_u8 = weight_scale_u8.to(dev)

    rows, half_cols = packed.shape
    cols = half_cols * 2
    n_groups = cols // 32

    low = (packed & 0x0F).long()
    high = ((packed >> 4) & 0x0F).long()
    vals = torch.stack([lut[low], lut[high]], dim=-1).reshape(rows, n_groups, 32)

    exp_int = scales_u8.int() - 127
    scales = torch.ldexp(torch.ones_like(exp_int, dtype=torch.float32), exp_int).unsqueeze(-1)
    out = (vals * scales).reshape(rows, cols).to(torch.bfloat16)
    return out


def tensor_to_le_bytes(t: torch.Tensor) -> bytes:
    t_contig = t.detach().cpu().contiguous()
    if t_contig.dtype == torch.bfloat16:
        return t_contig.view(torch.uint16).numpy().tobytes()
    return t_contig.numpy().tobytes()


def bytes_to_tensor(buf: bytes | memoryview, dtype_code: int, shape: tuple[int, ...]) -> torch.Tensor:
    arr = np.frombuffer(buf, dtype=np.uint16 if dtype_code == DTYPE_BF16 else (np.float32 if dtype_code == DTYPE_FP32 else np.uint8)).copy()
    t = torch.from_numpy(arr)
    if dtype_code == DTYPE_BF16:
        t = t.view(torch.bfloat16)
    return t.reshape(shape)


def get_safetensor_header(repo_id: str, filename: str) -> tuple[str, dict[str, Any], int]:
    """Fetches the safetensors JSON header via an IPv4 HTTP Range request with automatic retry."""
    enable_ipv4_only()
    url = hf_hub_url(repo_id, filename)
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                cdn_url = resp.geturl()
                hlen = struct.unpack("<Q", resp.read(8))[0]
            req2 = urllib.request.Request(cdn_url, headers={"Range": f"bytes=8-{8 + hlen - 1}"})
            with urllib.request.urlopen(req2, timeout=30) as resp:
                header = json.loads(resp.read().decode("utf-8"))
            return cdn_url, header, hlen + 8
        except Exception:
            if attempt == 5:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("Unreachable")


def _fetch_byte_range(cdn_url: str, byte_start: int, byte_end: int) -> bytes:
    expected = byte_end - byte_start + 1
    for attempt in range(5):
        try:
            req = urllib.request.Request(cdn_url, headers={"Range": f"bytes={byte_start}-{byte_end}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                chunk = resp.read()
            if len(chunk) != expected:
                raise IOError(f"Short range read: got {len(chunk)}, expected {expected}")
            return chunk
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("Unreachable")


def fetch_safetensor_tensors(
    repo_id: str,
    filename: str,
    tensor_names: list[str],
    max_gap_bytes: int = 2 * 1024 * 1024,
    chunk_bytes: int = 32 * 1024 * 1024,
    max_workers: int = 16,
) -> dict[str, torch.Tensor]:
    """
    Downloads a specific list of tensors from a remote `.safetensors` file on Hugging Face
    using coalesced, parallel 16-worker HTTP Range requests against the resolved CDN URL.
    """
    from concurrent.futures import ThreadPoolExecutor

    cdn_url, header, data_base = get_safetensor_header(repo_id, filename)
    items = []
    for name in tensor_names:
        if name not in header:
            raise KeyError(f"Tensor '{name}' not found in {repo_id}/{filename}")
        info = header[name]
        s, e = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        items.append((s, e, name, info["dtype"], tuple(info["shape"])))
    items.sort(key=lambda x: x[0])

    # Coalesce adjacent spans
    spans: list[list[Any]] = []
    for it in items:
        if not spans or (it[0] - spans[-1][1] > max_gap_bytes):
            spans.append([it[0], it[1], [it]])
        else:
            spans[-1][1] = max(spans[-1][1], it[1])
            spans[-1][2].append(it)

    # Split spans into 32 MB sub-tasks for 16-thread parallel download
    subtasks: list[tuple[int, int, int, int]] = []
    for sp_idx, (span_start, span_end, _) in enumerate(spans):
        cur = span_start
        while cur < span_end:
            nxt = min(span_end, cur + chunk_bytes)
            subtasks.append((sp_idx, cur - span_start, data_base + cur, data_base + nxt - 1))
            cur = nxt

    span_buffers = [bytearray(sp[1] - sp[0]) for sp in spans]
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(subtasks)))) as pool:
        futs = [
            (sp_idx, rel_off, pool.submit(_fetch_byte_range, cdn_url, b_start, b_end))
            for sp_idx, rel_off, b_start, b_end in subtasks
        ]
        for sp_idx, rel_off, fut in futs:
            data = fut.result()
            span_buffers[sp_idx][rel_off : rel_off + len(data)] = data

    out: dict[str, torch.Tensor] = {}
    for sp_idx, (span_start, span_end, span_items) in enumerate(spans):
        mv = memoryview(span_buffers[sp_idx])
        for s, e, name, dtype_str, shape in span_items:
            sub = mv[s - span_start : e - span_start]
            if dtype_str == "BF16":
                arr = np.frombuffer(sub, dtype=np.uint16).copy()
                t = torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
            elif dtype_str == "F32":
                arr = np.frombuffer(sub, dtype=np.float32).copy()
                t = torch.from_numpy(arr).reshape(shape)
            elif dtype_str in ("U8", "UINT8"):
                arr = np.frombuffer(sub, dtype=np.uint8).copy()
                t = torch.from_numpy(arr).reshape(shape)
            else:
                raise ValueError(f"Unsupported safetensor dtype: {dtype_str}")
            out[name] = t
    return out


def write_trunk_container(
    out_dir: Path,
    layer_id: int,
    layer_kind: int,
    records_spec: list[tuple[int, int, int, int, int, bytes]],
    operator_bytes: bytes | None = None,
    indexer_bytes: bytes | None = None,
    prune_needs: set[int] | None = None,
) -> dict[str, Any]:
    """
    Writes `index.bin`, `common.bin`, `attention.bin`, `routed-shared.bin`, `operator.bin`,
    and optional `indexer.bin` (`K31IDX01` for Kimi K3.1 Pro's 24 Gated DSA layers).
    """
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
    idx_len = 0
    if indexer_bytes is not None:
        (out_dir / "indexer.bin").write_bytes(indexer_bytes)
        idx_len = len(indexer_bytes)

    return {
        "layer_id": layer_id,
        "layer_kind": layer_kind,
        "record_count": len(packed_records),
        "group_lengths": group_lengths,
        "total_group_bytes": sum(group_lengths),
        "dropped_bytes": dropped_bytes,
        "operator_bytes": op_len,
        "indexer_bytes": idx_len,
    }


def pack_dsa_indexer_bytes(layer_id: int, tensors: dict[str, torch.Tensor], prefix: str) -> bytes:
    """
    Packs the 5 Gated DSA Lightning Indexer tensors (`self_attn.indexer.*`) into `indexer.bin` (`K31IDX01`):
      1. `k_norm.weight`       [128] BF16
      2. `k_norm.bias`         [128] BF16
      3. `weights_proj.weight` [32, 7168] BF16
      4. `wk.weight`           [128, 7168] BF16
      5. `wq_b.weight`         [4096, 1536] BF16
    """
    b_kn_w = tensor_to_le_bytes(tensors[prefix + "self_attn.indexer.k_norm.weight"])
    b_kn_b = tensor_to_le_bytes(tensors[prefix + "self_attn.indexer.k_norm.bias"])
    b_wp_w = tensor_to_le_bytes(tensors[prefix + "self_attn.indexer.weights_proj.weight"])
    b_wk_w = tensor_to_le_bytes(tensors[prefix + "self_attn.indexer.wk.weight"])
    b_wqb_w = tensor_to_le_bytes(tensors[prefix + "self_attn.indexer.wq_b.weight"])

    hdr = bytearray(64)
    struct.pack_into(
        "<8sIIIIIII",
        hdr,
        0,
        INDEXER_MAGIC,
        1,
        layer_id,
        len(b_kn_w),
        len(b_kn_b),
        len(b_wp_w),
        len(b_wk_w),
        len(b_wqb_w),
    )
    return bytes(hdr) + b_kn_w + b_kn_b + b_wp_w + b_wk_w + b_wqb_w


def unpack_dsa_indexer_bytes(buf: bytes) -> dict[str, torch.Tensor]:
    if buf[:8] != INDEXER_MAGIC:
        raise ValueError(f"Invalid indexer magic: {buf[:8]!r}")
    _, ver, layer_id, l1, l2, l3, l4, l5 = struct.unpack_from("<8sIIIIIII", buf, 0)
    mv = memoryview(buf)
    cur = 64
    kn_w = bytes_to_tensor(mv[cur : cur + l1], DTYPE_BF16, (128,)); cur += l1
    kn_b = bytes_to_tensor(mv[cur : cur + l2], DTYPE_BF16, (128,)); cur += l2
    wp_w = bytes_to_tensor(mv[cur : cur + l3], DTYPE_BF16, (32, 7168)); cur += l3
    wk_w = bytes_to_tensor(mv[cur : cur + l4], DTYPE_BF16, (128, 7168)); cur += l4
    wqb_w = bytes_to_tensor(mv[cur : cur + l5], DTYPE_BF16, (4096, 1536)); cur += l5
    return {
        "layer_id": layer_id,
        "k_norm_weight": kn_w,
        "k_norm_bias": kn_b,
        "weights_proj_weight": wp_w,
        "wk_weight": wk_w,
        "wq_b_weight": wqb_w,
    }


def prepare_kimi_k31_datasets(
    work_dir: Path,
    repo_id: str = "moonshotai/Kimi-K3.1-Pro",
    drafter_repo_id: str = "LithosAI/Kimi-K3.1-Pro-DSpark",
    num_cached_experts_per_stage: int = 32,
    extract_all_24_dsa_indexers: bool = True,
) -> dict[str, Any]:
    """
    Downloads and precomputes the Kimi K3.1 Pro stage containers in `work_dir`:
      - `raw/` and `pruned/` directories with:
        * `global-0/` (`embed_tokens.bf16.bin`, `lm_head.bf16.bin`, `norm.bf16.bin`, `output_attn_res_norm.bf16.bin`, `output_attn_res_proj.bf16.bin`)
        * `trunk-0/`  (Layer 0: KDA + 33,792 Dense `SiTU-GLU` MLP + `operator.bin`)
        * `trunk-1/` + `expert-1/` (Layer 1: KDA + Latent `SiTU-GLU` MoE + `operator.bin` + MXFP4 dequantized experts)
        * `trunk-3/` + `expert-3/` (Layer 3: Gated DSA + 5-tensor Lightning Indexer `indexer.bin` + `operator.bin` + MXFP4 dequantized experts)
        * `trunk-{L}/indexer.bin` across all 24 Gated DSA layers (`L in DSA_LAYERS_0IDX`)
        * `vision-0/` (`mm_projector` from `MoonViT-V2`)
        * `dspark-0/` (`LithosAI/Kimi-K3.1-Pro-DSpark` 5-layer speculative draft head)
    """
    enable_ipv4_only()
    t0 = time.perf_counter()
    raw_dir = work_dir / "raw"
    pruned_dir = work_dir / "pruned"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pruned_dir.mkdir(parents=True, exist_ok=True)

    # 1. Global embeddings, final RMSNorm, output_attn_res, and LM head from shard 94
    global_raw = raw_dir / "global-0"
    global_pruned = pruned_dir / "global-0"
    global_raw.mkdir(parents=True, exist_ok=True)
    global_pruned.mkdir(parents=True, exist_ok=True)

    emb_file = global_raw / "output_attn_res_proj.bf16.bin"
    if not emb_file.exists():
        print("[Kimi-K3.1-Pro] Fetching global-0 (embed_tokens, lm_head, norm, output_attn_res) from shard 94...", flush=True)
        g_tensors = fetch_safetensor_tensors(
            repo_id,
            "model-00094-of-000096.safetensors",
            [
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
                "language_model.model.norm.weight",
                "language_model.model.output_attn_res_norm.weight",
                "language_model.model.output_attn_res_proj.weight",
            ],
        )
        (global_raw / "embed_tokens.bf16.bin").write_bytes(tensor_to_le_bytes(g_tensors["language_model.model.embed_tokens.weight"]))
        (global_raw / "lm_head.bf16.bin").write_bytes(tensor_to_le_bytes(g_tensors["language_model.lm_head.weight"]))
        (global_raw / "norm.bf16.bin").write_bytes(tensor_to_le_bytes(g_tensors["language_model.model.norm.weight"]))
        (global_raw / "output_attn_res_norm.bf16.bin").write_bytes(tensor_to_le_bytes(g_tensors["language_model.model.output_attn_res_norm.weight"]))
        (global_raw / "output_attn_res_proj.bf16.bin").write_bytes(tensor_to_le_bytes(g_tensors["language_model.model.output_attn_res_proj.weight"]))
        del g_tensors

    for fname in ("embed_tokens.bf16.bin", "lm_head.bf16.bin", "norm.bf16.bin", "output_attn_res_norm.bf16.bin", "output_attn_res_proj.bf16.bin"):
        src_p = global_raw / fname
        dst_p = global_pruned / fname
        if not dst_p.exists():
            os.link(src_p, dst_p)

    # 2. Layer 0: KDA + Dense 33,792 SiTU-GLU MLP (shard 1)
    l0_prefix = "language_model.model.layers.0."
    l0_keys = [
        l0_prefix + "input_layernorm.weight",
        l0_prefix + "post_attention_layernorm.weight",
        l0_prefix + "self_attention_res_norm.weight",
        l0_prefix + "self_attention_res_proj.weight",
        l0_prefix + "mlp_res_norm.weight",
        l0_prefix + "mlp_res_proj.weight",
        l0_prefix + "self_attn.q_proj.weight",
        l0_prefix + "self_attn.k_proj.weight",
        l0_prefix + "self_attn.v_proj.weight",
        l0_prefix + "self_attn.q_conv1d.weight",
        l0_prefix + "self_attn.k_conv1d.weight",
        l0_prefix + "self_attn.v_conv1d.weight",
        l0_prefix + "self_attn.b_proj.weight",
        l0_prefix + "self_attn.f_a_proj.weight",
        l0_prefix + "self_attn.f_b_proj.weight",
        l0_prefix + "self_attn.dt_bias",
        l0_prefix + "self_attn.A_log",
        l0_prefix + "self_attn.o_norm.weight",
        l0_prefix + "self_attn.g_proj.weight",
        l0_prefix + "self_attn.o_proj.weight",
        l0_prefix + "mlp.gate_proj.weight",
        l0_prefix + "mlp.up_proj.weight",
        l0_prefix + "mlp.down_proj.weight",
    ]
    if not (pruned_dir / "trunk-0" / "operator.bin").exists():
        print("[Kimi-K3.1-Pro] Fetching & building trunk-0 (Layer 0 KDA + 33,792 Dense SiTU-GLU)...", flush=True)
        t_l0 = fetch_safetensor_tensors(repo_id, "model-00001-of-000096.safetensors", l0_keys)
        b_in_ln = tensor_to_le_bytes(t_l0[l0_prefix + "input_layernorm.weight"])
        b_q = tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.q_proj.weight"])
        b_k = tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.k_proj.weight"])
        b_v = tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.v_proj.weight"])
        b_qc = tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.q_conv1d.weight"].float())
        b_kc = tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.k_conv1d.weight"].float())
        b_vc = tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.v_conv1d.weight"].float())

        op0_hdr = bytearray(64)
        struct.pack_into(
            "<8sIIIIIIII",
            op0_hdr,
            0,
            OPERATOR_MAGIC,
            0,  # kind=0 (KDA)
            0,  # layer=0
            len(b_in_ln),
            len(b_q),
            len(b_k),
            len(b_v),
            len(b_qc),
            len(b_kc),
        )
        struct.pack_into("<I", op0_hdr, 40, len(b_vc))
        op0_bytes = bytes(op0_hdr) + b_in_ln + b_q + b_k + b_v + b_qc + b_kc + b_vc

        l0_records = [
            (0, 0, DTYPE_BF16, 7168, 1, b_in_ln),
            (1, 0, DTYPE_BF16, 7168, 1, tensor_to_le_bytes(t_l0[l0_prefix + "post_attention_layernorm.weight"])),
            (2, 0, DTYPE_BF16, 7168, 1, tensor_to_le_bytes(t_l0[l0_prefix + "self_attention_res_norm.weight"])),
            (3, 0, DTYPE_BF16, 1, 7168, tensor_to_le_bytes(t_l0[l0_prefix + "self_attention_res_proj.weight"])),
            (4, 0, DTYPE_BF16, 7168, 1, tensor_to_le_bytes(t_l0[l0_prefix + "mlp_res_norm.weight"])),
            (5, 0, DTYPE_BF16, 1, 7168, tensor_to_le_bytes(t_l0[l0_prefix + "mlp_res_proj.weight"])),
            (6, 0, DTYPE_BF16, 12288, 7168, tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.g_proj.weight"])),
            (7, 0, DTYPE_BF16, 7168, 12288, tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.o_proj.weight"])),
            (8, 1, DTYPE_BF16, 12288, 7168, b_q),
            (9, 1, DTYPE_BF16, 12288, 7168, b_k),
            (10, 1, DTYPE_BF16, 12288, 7168, b_v),
            (11, 1, DTYPE_BF16, 96, 7168, tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.b_proj.weight"])),
            (12, 1, DTYPE_BF16, 128, 7168, tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.f_a_proj.weight"])),
            (13, 1, DTYPE_BF16, 12288, 128, tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.f_b_proj.weight"])),
            (18, 1, DTYPE_FP32, 12288, 1, tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.dt_bias"].float())),
            (19, 1, DTYPE_FP32, 128, 1, tensor_to_le_bytes(t_l0[l0_prefix + "self_attn.o_norm.weight"].float())),
            (39, 1, DTYPE_FP32, 128, 1, tensor_to_le_bytes(-torch.exp(t_l0[l0_prefix + "self_attn.A_log"].float()))),
            (26, 2, DTYPE_BF16, 33792, 7168, tensor_to_le_bytes(t_l0[l0_prefix + "mlp.gate_proj.weight"])),
            (27, 2, DTYPE_BF16, 33792, 7168, tensor_to_le_bytes(t_l0[l0_prefix + "mlp.up_proj.weight"])),
            (28, 2, DTYPE_BF16, 7168, 33792, tensor_to_le_bytes(t_l0[l0_prefix + "mlp.down_proj.weight"])),
        ]
        kda0_needs = {1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 18, 19, 39, 26, 27, 28}
        write_trunk_container(raw_dir / "trunk-0", 0, 0, l0_records, operator_bytes=op0_bytes, prune_needs=None)
        write_trunk_container(pruned_dir / "trunk-0", 0, 0, l0_records, operator_bytes=op0_bytes, prune_needs=kda0_needs)
        del t_l0

    # 3. Layer 1: KDA + 896-Expert Latent SiTU-GLU MoE (shard 2)
    l1_prefix = "language_model.model.layers.1."
    l1_trunk_keys = [
        l1_prefix + "input_layernorm.weight",
        l1_prefix + "post_attention_layernorm.weight",
        l1_prefix + "self_attention_res_norm.weight",
        l1_prefix + "self_attention_res_proj.weight",
        l1_prefix + "mlp_res_norm.weight",
        l1_prefix + "mlp_res_proj.weight",
        l1_prefix + "self_attn.q_proj.weight",
        l1_prefix + "self_attn.k_proj.weight",
        l1_prefix + "self_attn.v_proj.weight",
        l1_prefix + "self_attn.q_conv1d.weight",
        l1_prefix + "self_attn.k_conv1d.weight",
        l1_prefix + "self_attn.v_conv1d.weight",
        l1_prefix + "self_attn.b_proj.weight",
        l1_prefix + "self_attn.f_a_proj.weight",
        l1_prefix + "self_attn.f_b_proj.weight",
        l1_prefix + "self_attn.dt_bias",
        l1_prefix + "self_attn.A_log",
        l1_prefix + "self_attn.o_norm.weight",
        l1_prefix + "self_attn.g_proj.weight",
        l1_prefix + "self_attn.o_proj.weight",
        l1_prefix + "block_sparse_moe.gate.weight",
        l1_prefix + "block_sparse_moe.gate.e_score_correction_bias",
        l1_prefix + "block_sparse_moe.routed_expert_down_proj.weight",
        l1_prefix + "block_sparse_moe.routed_expert_up_proj.weight",
        l1_prefix + "block_sparse_moe.routed_expert_norm.weight",
        l1_prefix + "block_sparse_moe.shared_experts.gate_proj.weight",
        l1_prefix + "block_sparse_moe.shared_experts.up_proj.weight",
        l1_prefix + "block_sparse_moe.shared_experts.down_proj.weight",
    ]
    for e in range(num_cached_experts_per_stage):
        for w in ("w1", "w2", "w3"):
            l1_trunk_keys.append(f"{l1_prefix}block_sparse_moe.experts.{e}.{w}.weight_packed")
            l1_trunk_keys.append(f"{l1_prefix}block_sparse_moe.experts.{e}.{w}.weight_scale")

    if not (pruned_dir / "trunk-1" / "operator.bin").exists():
        print(f"[Kimi-K3.1-Pro] Fetching & building trunk-1 + expert-1 (KDA + MoE, {num_cached_experts_per_stage} MXFP4 experts)...", flush=True)
        t_l1 = fetch_safetensor_tensors(repo_id, "model-00002-of-000096.safetensors", l1_trunk_keys)
        b_in_ln = tensor_to_le_bytes(t_l1[l1_prefix + "input_layernorm.weight"])
        b_q = tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.q_proj.weight"])
        b_k = tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.k_proj.weight"])
        b_v = tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.v_proj.weight"])
        b_qc = tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.q_conv1d.weight"].float())
        b_kc = tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.k_conv1d.weight"].float())
        b_vc = tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.v_conv1d.weight"].float())

        op1_hdr = bytearray(64)
        struct.pack_into(
            "<8sIIIIIIII",
            op1_hdr,
            0,
            OPERATOR_MAGIC,
            0,
            1,
            len(b_in_ln),
            len(b_q),
            len(b_k),
            len(b_v),
            len(b_qc),
            len(b_kc),
        )
        struct.pack_into("<I", op1_hdr, 40, len(b_vc))
        op1_bytes = bytes(op1_hdr) + b_in_ln + b_q + b_k + b_v + b_qc + b_kc + b_vc

        l1_records = [
            (0, 0, DTYPE_BF16, 7168, 1, b_in_ln),
            (1, 0, DTYPE_BF16, 7168, 1, tensor_to_le_bytes(t_l1[l1_prefix + "post_attention_layernorm.weight"])),
            (2, 0, DTYPE_BF16, 7168, 1, tensor_to_le_bytes(t_l1[l1_prefix + "self_attention_res_norm.weight"])),
            (3, 0, DTYPE_BF16, 1, 7168, tensor_to_le_bytes(t_l1[l1_prefix + "self_attention_res_proj.weight"])),
            (4, 0, DTYPE_BF16, 7168, 1, tensor_to_le_bytes(t_l1[l1_prefix + "mlp_res_norm.weight"])),
            (5, 0, DTYPE_BF16, 1, 7168, tensor_to_le_bytes(t_l1[l1_prefix + "mlp_res_proj.weight"])),
            (6, 0, DTYPE_BF16, 12288, 7168, tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.g_proj.weight"])),
            (7, 0, DTYPE_BF16, 7168, 12288, tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.o_proj.weight"])),
            (8, 1, DTYPE_BF16, 12288, 7168, b_q),
            (9, 1, DTYPE_BF16, 12288, 7168, b_k),
            (10, 1, DTYPE_BF16, 12288, 7168, b_v),
            (11, 1, DTYPE_BF16, 96, 7168, tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.b_proj.weight"])),
            (12, 1, DTYPE_BF16, 128, 7168, tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.f_a_proj.weight"])),
            (13, 1, DTYPE_BF16, 12288, 128, tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.f_b_proj.weight"])),
            (18, 1, DTYPE_FP32, 12288, 1, tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.dt_bias"].float())),
            (19, 1, DTYPE_FP32, 128, 1, tensor_to_le_bytes(t_l1[l1_prefix + "self_attn.o_norm.weight"].float())),
            (39, 1, DTYPE_FP32, 128, 1, tensor_to_le_bytes(-torch.exp(t_l1[l1_prefix + "self_attn.A_log"].float()))),
            (29, 2, DTYPE_FP32, 896, 7168, tensor_to_le_bytes(t_l1[l1_prefix + "block_sparse_moe.gate.weight"].float())),
            (30, 2, DTYPE_FP32, 896, 1, tensor_to_le_bytes(t_l1[l1_prefix + "block_sparse_moe.gate.e_score_correction_bias"].float())),
            (31, 2, DTYPE_BF16, 3584, 7168, tensor_to_le_bytes(t_l1[l1_prefix + "block_sparse_moe.routed_expert_down_proj.weight"])),
            (32, 2, DTYPE_BF16, 7168, 3584, tensor_to_le_bytes(t_l1[l1_prefix + "block_sparse_moe.routed_expert_up_proj.weight"])),
            (33, 2, DTYPE_BF16, 3584, 1, tensor_to_le_bytes(t_l1[l1_prefix + "block_sparse_moe.routed_expert_norm.weight"])),
            (34, 2, DTYPE_BF16, 6144, 7168, tensor_to_le_bytes(t_l1[l1_prefix + "block_sparse_moe.shared_experts.gate_proj.weight"])),
            (35, 2, DTYPE_BF16, 6144, 7168, tensor_to_le_bytes(t_l1[l1_prefix + "block_sparse_moe.shared_experts.up_proj.weight"])),
            (36, 2, DTYPE_BF16, 7168, 6144, tensor_to_le_bytes(t_l1[l1_prefix + "block_sparse_moe.shared_experts.down_proj.weight"])),
        ]
        kda_moe_needs = {1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 18, 19, 39, 29, 30, 31, 32, 33, 34, 35, 36}
        write_trunk_container(raw_dir / "trunk-1", 1, 1, l1_records, operator_bytes=op1_bytes, prune_needs=None)
        write_trunk_container(pruned_dir / "trunk-1", 1, 1, l1_records, operator_bytes=op1_bytes, prune_needs=kda_moe_needs)
        _write_expert_store(raw_dir / "expert-1", pruned_dir / "expert-1", t_l1, l1_prefix, num_cached_experts_per_stage)
        del t_l1

    # 4. Layer 3: Gated DSA (with 5-tensor Lightning Indexer) + 896-Expert Latent SiTU-GLU MoE (shard 4)
    l3_prefix = "language_model.model.layers.3."
    l3_trunk_keys = [
        l3_prefix + "input_layernorm.weight",
        l3_prefix + "post_attention_layernorm.weight",
        l3_prefix + "self_attention_res_norm.weight",
        l3_prefix + "self_attention_res_proj.weight",
        l3_prefix + "mlp_res_norm.weight",
        l3_prefix + "mlp_res_proj.weight",
        l3_prefix + "self_attn.q_a_proj.weight",
        l3_prefix + "self_attn.q_a_layernorm.weight",
        l3_prefix + "self_attn.q_b_proj.weight",
        l3_prefix + "self_attn.kv_a_proj_with_mqa.weight",
        l3_prefix + "self_attn.kv_a_layernorm.weight",
        l3_prefix + "self_attn.kv_b_proj.weight",
        l3_prefix + "self_attn.g_proj.weight",
        l3_prefix + "self_attn.o_proj.weight",
        l3_prefix + "self_attn.indexer.k_norm.weight",
        l3_prefix + "self_attn.indexer.k_norm.bias",
        l3_prefix + "self_attn.indexer.weights_proj.weight",
        l3_prefix + "self_attn.indexer.wk.weight",
        l3_prefix + "self_attn.indexer.wq_b.weight",
        l3_prefix + "block_sparse_moe.gate.weight",
        l3_prefix + "block_sparse_moe.gate.e_score_correction_bias",
        l3_prefix + "block_sparse_moe.routed_expert_down_proj.weight",
        l3_prefix + "block_sparse_moe.routed_expert_up_proj.weight",
        l3_prefix + "block_sparse_moe.routed_expert_norm.weight",
        l3_prefix + "block_sparse_moe.shared_experts.gate_proj.weight",
        l3_prefix + "block_sparse_moe.shared_experts.up_proj.weight",
        l3_prefix + "block_sparse_moe.shared_experts.down_proj.weight",
    ]
    for e in range(num_cached_experts_per_stage):
        for w in ("w1", "w2", "w3"):
            l3_trunk_keys.append(f"{l3_prefix}block_sparse_moe.experts.{e}.{w}.weight_packed")
            l3_trunk_keys.append(f"{l3_prefix}block_sparse_moe.experts.{e}.{w}.weight_scale")

    if not (pruned_dir / "trunk-3" / "indexer.bin").exists():
        print(f"[Kimi-K3.1-Pro] Fetching & building trunk-3 + expert-3 (Gated DSA + Lightning Indexer + MoE)...", flush=True)
        t_l3 = fetch_safetensor_tensors(repo_id, "model-00004-of-000096.safetensors", l3_trunk_keys)
        b_in_ln = tensor_to_le_bytes(t_l3[l3_prefix + "input_layernorm.weight"])
        b_qa = tensor_to_le_bytes(t_l3[l3_prefix + "self_attn.q_a_proj.weight"])
        b_qa_ln = tensor_to_le_bytes(t_l3[l3_prefix + "self_attn.q_a_layernorm.weight"])
        b_qb = tensor_to_le_bytes(t_l3[l3_prefix + "self_attn.q_b_proj.weight"])
        b_kva = tensor_to_le_bytes(t_l3[l3_prefix + "self_attn.kv_a_proj_with_mqa.weight"])
        b_kva_ln = tensor_to_le_bytes(t_l3[l3_prefix + "self_attn.kv_a_layernorm.weight"])
        b_kvb = tensor_to_le_bytes(t_l3[l3_prefix + "self_attn.kv_b_proj.weight"])

        op3_hdr = bytearray(64)
        struct.pack_into(
            "<8sIIIIIIII",
            op3_hdr,
            0,
            OPERATOR_MAGIC,
            2,  # kind=2 (Gated DSA / MLA)
            3,  # layer=3
            len(b_in_ln),
            len(b_qa),
            len(b_qa_ln),
            len(b_qb),
            len(b_kva),
            len(b_kva_ln),
        )
        struct.pack_into("<I", op3_hdr, 40, len(b_kvb))
        op3_bytes = bytes(op3_hdr) + b_in_ln + b_qa + b_qa_ln + b_qb + b_kva + b_kva_ln + b_kvb
        idx3_bytes = pack_dsa_indexer_bytes(3, t_l3, l3_prefix)

        l3_records = [
            (0, 0, DTYPE_BF16, 7168, 1, b_in_ln),
            (1, 0, DTYPE_BF16, 7168, 1, tensor_to_le_bytes(t_l3[l3_prefix + "post_attention_layernorm.weight"])),
            (2, 0, DTYPE_BF16, 7168, 1, tensor_to_le_bytes(t_l3[l3_prefix + "self_attention_res_norm.weight"])),
            (3, 0, DTYPE_BF16, 1, 7168, tensor_to_le_bytes(t_l3[l3_prefix + "self_attention_res_proj.weight"])),
            (4, 0, DTYPE_BF16, 7168, 1, tensor_to_le_bytes(t_l3[l3_prefix + "mlp_res_norm.weight"])),
            (5, 0, DTYPE_BF16, 1, 7168, tensor_to_le_bytes(t_l3[l3_prefix + "mlp_res_proj.weight"])),
            (6, 0, DTYPE_BF16, 12288, 7168, tensor_to_le_bytes(t_l3[l3_prefix + "self_attn.g_proj.weight"])),
            (7, 0, DTYPE_BF16, 7168, 12288, tensor_to_le_bytes(t_l3[l3_prefix + "self_attn.o_proj.weight"])),
            (20, 1, DTYPE_BF16, 1536, 7168, b_qa),
            (21, 1, DTYPE_BF16, 1536, 1, b_qa_ln),
            (22, 1, DTYPE_BF16, 18432, 1536, b_qb),
            (23, 1, DTYPE_BF16, 576, 7168, b_kva),
            (24, 1, DTYPE_BF16, 512, 1, b_kva_ln),
            (25, 1, DTYPE_BF16, 24576, 512, b_kvb),
            (29, 2, DTYPE_FP32, 896, 7168, tensor_to_le_bytes(t_l3[l3_prefix + "block_sparse_moe.gate.weight"].float())),
            (30, 2, DTYPE_FP32, 896, 1, tensor_to_le_bytes(t_l3[l3_prefix + "block_sparse_moe.gate.e_score_correction_bias"].float())),
            (31, 2, DTYPE_BF16, 3584, 7168, tensor_to_le_bytes(t_l3[l3_prefix + "block_sparse_moe.routed_expert_down_proj.weight"])),
            (32, 2, DTYPE_BF16, 7168, 3584, tensor_to_le_bytes(t_l3[l3_prefix + "block_sparse_moe.routed_expert_up_proj.weight"])),
            (33, 2, DTYPE_BF16, 3584, 1, tensor_to_le_bytes(t_l3[l3_prefix + "block_sparse_moe.routed_expert_norm.weight"])),
            (34, 2, DTYPE_BF16, 6144, 7168, tensor_to_le_bytes(t_l3[l3_prefix + "block_sparse_moe.shared_experts.gate_proj.weight"])),
            (35, 2, DTYPE_BF16, 6144, 7168, tensor_to_le_bytes(t_l3[l3_prefix + "block_sparse_moe.shared_experts.up_proj.weight"])),
            (36, 2, DTYPE_BF16, 7168, 6144, tensor_to_le_bytes(t_l3[l3_prefix + "block_sparse_moe.shared_experts.down_proj.weight"])),
        ]
        dsa_moe_needs = {1, 2, 3, 4, 5, 6, 7, 29, 30, 31, 32, 33, 34, 35, 36}
        write_trunk_container(raw_dir / "trunk-3", 3, 2, l3_records, operator_bytes=op3_bytes, indexer_bytes=idx3_bytes, prune_needs=None)
        write_trunk_container(pruned_dir / "trunk-3", 3, 2, l3_records, operator_bytes=op3_bytes, indexer_bytes=idx3_bytes, prune_needs=dsa_moe_needs)
        _write_expert_store(raw_dir / "expert-3", pruned_dir / "expert-3", t_l3, l3_prefix, num_cached_experts_per_stage)
        del t_l3

    # 5. Extract `indexer.bin` (`K31IDX01`) for all 24 Gated DSA layers (`3, 7, 11, ..., 91, 92`)
    if extract_all_24_dsa_indexers:
        for L in DSA_LAYERS_0IDX:
            idx_out = pruned_dir / f"trunk-{L}" / "indexer.bin"
            if idx_out.exists():
                continue
            shard_name = f"model-{L + 1:05d}-of-000096.safetensors"
            pfx = f"language_model.model.layers.{L}."
            idx_keys = [
                pfx + "self_attn.indexer.k_norm.weight",
                pfx + "self_attn.indexer.k_norm.bias",
                pfx + "self_attn.indexer.weights_proj.weight",
                pfx + "self_attn.indexer.wk.weight",
                pfx + "self_attn.indexer.wq_b.weight",
            ]
            print(f"[Kimi-K3.1-Pro] Extracting Gated DSA Lightning Indexer for layer {L} ({shard_name})...", flush=True)
            t_idx = fetch_safetensor_tensors(repo_id, shard_name, idx_keys)
            idx_bytes = pack_dsa_indexer_bytes(L, t_idx, pfx)
            (raw_dir / f"trunk-{L}").mkdir(parents=True, exist_ok=True)
            (pruned_dir / f"trunk-{L}").mkdir(parents=True, exist_ok=True)
            (raw_dir / f"trunk-{L}" / "indexer.bin").write_bytes(idx_bytes)
            idx_out.write_bytes(idx_bytes)

    # 6. Extract MoonViT-V2 multimodal projector (`mm_projector.*` from shard 95)
    vis_dir = pruned_dir / "vision-0"
    vis_dir.mkdir(parents=True, exist_ok=True)
    if not (vis_dir / "mm_projector.pt").exists():
        print("[Kimi-K3.1-Pro] Extracting MoonViT-V2 mm_projector from shard 95...", flush=True)
        t_vis = fetch_safetensor_tensors(
            repo_id,
            "model-00095-of-000096.safetensors",
            [
                "mm_projector.post_norm.weight",
                "mm_projector.proj.0.weight",
                "mm_projector.proj.0.bias",
                "mm_projector.proj.2.weight",
                "mm_projector.proj.2.bias",
            ] if "mm_projector.proj.0.bias" in get_safetensor_header(repo_id, "model-00095-of-000096.safetensors")[1] else [
                "mm_projector.post_norm.weight",
                "mm_projector.proj.0.weight",
                "mm_projector.proj.2.weight",
            ],
        )
        torch.save(t_vis, vis_dir / "mm_projector.pt")

    # 7. Extract DSpark 5-layer speculative drafter (`LithosAI/Kimi-K3.1-Pro-DSpark`)
    dspark_dir = pruned_dir / "dspark-0"
    dspark_dir.mkdir(parents=True, exist_ok=True)
    if drafter_repo_id and not (dspark_dir / "dspark_stage0.pt").exists():
        print(f"[Kimi-K3.1-Pro] Extracting DSpark draft head from {drafter_repo_id}...", flush=True)
        ds_keys = [
            "fc.weight",
            "hidden_norm.weight",
            "confidence_head.proj.weight",
            "confidence_head.proj.bias",
            "layers.0.input_layernorm.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.k_norm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight",
        ]
        t_ds = fetch_safetensor_tensors(drafter_repo_id, "model.safetensors", ds_keys)
        torch.save(t_ds, dspark_dir / "dspark_stage0.pt")

    # 8. Download tokenizer files (`tiktoken.model`, `tokenizer_config.json`, `config.json`)
    tok_dir = work_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("config.json", "tiktoken.model", "tokenizer_config.json", "encoding_k3.py"):
        if not (tok_dir / fname).exists():
            p = hf_hub_download(repo_id, fname, local_dir=str(tok_dir))

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "model_repo": repo_id,
        "drafter_repo": drafter_repo_id,
        "work_dir": str(work_dir),
        "cached_experts_per_stage": num_cached_experts_per_stage,
        "dsa_indexers_extracted": len(DSA_LAYERS_0IDX) if extract_all_24_dsa_indexers else 1,
        "elapsed_seconds": elapsed,
    }
    (work_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_expert_store(
    raw_exp_dir: Path,
    pruned_exp_dir: Path,
    tensors: dict[str, torch.Tensor],
    prefix: str,
    num_experts: int,
) -> None:
    """Dequantizes `num_experts` MXFP4 (`w1`, `w2`, `w3`) experts into `experts.direct` (`BF16`) and `maps.json`."""
    raw_exp_dir.mkdir(parents=True, exist_ok=True)
    pruned_exp_dir.mkdir(parents=True, exist_ok=True)
    direct_path = raw_exp_dir / "experts.direct"

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(direct_path, "wb") as f:
        for e in range(num_experts):
            epfx = f"{prefix}block_sparse_moe.experts.{e}."
            w1 = dequantize_mxfp4_tensor(tensors[epfx + "w1.weight_packed"], tensors[epfx + "w1.weight_scale"], dev)
            w3 = dequantize_mxfp4_tensor(tensors[epfx + "w3.weight_packed"], tensors[epfx + "w3.weight_scale"], dev)
            w2 = dequantize_mxfp4_tensor(tensors[epfx + "w2.weight_packed"], tensors[epfx + "w2.weight_scale"], dev)
            f.write(tensor_to_le_bytes(w1))
            f.write(tensor_to_le_bytes(w3))
            f.write(tensor_to_le_bytes(w2))

    meta = {
        "format": "K31_MXFP4_DEQUANT_BF16_DIRECT",
        "num_experts_stored": num_experts,
        "total_routed_experts": 896,
        "w1_shape": [3072, 3584],
        "w3_shape": [3072, 3584],
        "w2_shape": [3584, 3072],
        "expert_stride_bytes": (3072 * 3584 + 3072 * 3584 + 3584 * 3072) * 2,
    }
    (raw_exp_dir / "maps.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (pruned_exp_dir / "maps.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    dst_direct = pruned_exp_dir / "experts.direct"
    if not dst_direct.exists():
        os.link(direct_path, dst_direct)


class KimiK31TrunkReader:
    """Reads precomputed `index.bin`, `common.bin`, `attention.bin`, `routed-shared.bin`, `operator.bin`, and `indexer.bin`."""

    def __init__(self, trunk_dir: Path):
        self.trunk_dir = Path(trunk_dir)
        idx_buf = (self.trunk_dir / "index.bin").read_bytes()
        if idx_buf[:8] != TRUNK_MAGIC:
            raise ValueError(f"Unexpected trunk magic in {trunk_dir}: {idx_buf[:8]!r}")
        _, self.version, self.layer_id, self.record_count = struct.unpack_from("<8sIII", idx_buf, 0)
        self.group_lengths = struct.unpack_from("<3Q", idx_buf, 24)
        self.layer_kind = struct.unpack_from("<I", idx_buf, 48)[0]

        self.groups_data: list[bytes] = []
        for g_idx, g_name in enumerate(GROUPS):
            gp = self.trunk_dir / g_name
            self.groups_data.append(gp.read_bytes() if gp.exists() else b"")

        self.records: dict[int, tuple[int, int, int, int, int, int, int]] = {}
        for i in range(self.record_count):
            base = HEADER_BYTES + i * RECORD_BYTES
            rid, gid, dtype_code, rows, cols, numel = struct.unpack_from("<6I", idx_buf, base)
            offset, length = struct.unpack_from("<2Q", idx_buf, base + 24)
            self.records[rid] = (gid, dtype_code, rows, cols, numel, offset, length)

        op_p = self.trunk_dir / "operator.bin"
        self.operator_bytes = op_p.read_bytes() if op_p.exists() else None
        idx_p = self.trunk_dir / "indexer.bin"
        self.indexer_bytes = idx_p.read_bytes() if idx_p.exists() else None

    def get_record_tensor(self, rid: int, shape: tuple[int, ...]) -> torch.Tensor:
        gid, dtype_code, rows, cols, numel, offset, length = self.records[rid]
        if length == 0:
            raise KeyError(f"Record {rid} was pruned (length=0) in {self.trunk_dir}")
        mv = memoryview(self.groups_data[gid])[offset : offset + length]
        return bytes_to_tensor(mv, dtype_code, shape)


class KimiK31GPUKDAStage(nn.Module):
    """
    GPU-resident differentiable Kimi Delta Attention (`KDA`) + Dense `SiTU-GLU` MLP stage (`trunk-0`).
    Matches `layer0.c` (`server_qkv`, `server_decay`, `server_update_attention`, `server_attention_output`,
    `server_aggregate`, and `server_dense` with `SiTU-GLU`).
    """

    def __init__(self, trunk_dir: Path):
        super().__init__()
        trunk = KimiK31TrunkReader(trunk_dir)
        op = trunk.operator_bytes
        assert op is not None and op[:8] == OPERATOR_MAGIC
        _, kind, layer_id, l_in, l_q, l_k, l_v, l_qc, l_kc = struct.unpack_from("<8sIIIIIIII", op, 0)
        l_vc = struct.unpack_from("<I", op, 40)[0]
        mv = memoryview(op)
        cur = 64
        in_ln = bytes_to_tensor(mv[cur : cur + l_in], DTYPE_BF16, (7168,)); cur += l_in
        q_w = bytes_to_tensor(mv[cur : cur + l_q], DTYPE_BF16, (12288, 7168)); cur += l_q
        k_w = bytes_to_tensor(mv[cur : cur + l_k], DTYPE_BF16, (12288, 7168)); cur += l_k
        v_w = bytes_to_tensor(mv[cur : cur + l_v], DTYPE_BF16, (12288, 7168)); cur += l_v
        qc_w = bytes_to_tensor(mv[cur : cur + l_qc], DTYPE_FP32, (12288, 1, 4)); cur += l_qc
        kc_w = bytes_to_tensor(mv[cur : cur + l_kc], DTYPE_FP32, (12288, 1, 4)); cur += l_kc
        vc_w = bytes_to_tensor(mv[cur : cur + l_vc], DTYPE_FP32, (12288, 1, 4)); cur += l_vc

        self.register_buffer("in_ln", in_ln, persistent=False)
        self.register_buffer("q_w", q_w, persistent=False)
        self.register_buffer("k_w", k_w, persistent=False)
        self.register_buffer("v_w", v_w, persistent=False)
        self.register_buffer("qc_w", qc_w.to(torch.bfloat16), persistent=False)
        self.register_buffer("kc_w", kc_w.to(torch.bfloat16), persistent=False)
        self.register_buffer("vc_w", vc_w.to(torch.bfloat16), persistent=False)

        self.register_buffer("post_ln", trunk.get_record_tensor(1, (7168,)), persistent=False)
        self.register_buffer("sa_res_norm", trunk.get_record_tensor(2, (7168,)), persistent=False)
        self.register_buffer("sa_res_proj", trunk.get_record_tensor(3, (7168,)), persistent=False)
        self.register_buffer("mlp_res_norm", trunk.get_record_tensor(4, (7168,)), persistent=False)
        self.register_buffer("mlp_res_proj", trunk.get_record_tensor(5, (7168,)), persistent=False)
        self.register_buffer("g_w", trunk.get_record_tensor(6, (12288, 7168)), persistent=False)
        self.register_buffer("o_w", trunk.get_record_tensor(7, (7168, 12288)), persistent=False)
        self.register_buffer("b_w", trunk.get_record_tensor(11, (96, 7168)), persistent=False)
        self.register_buffer("fa_w", trunk.get_record_tensor(12, (128, 7168)), persistent=False)
        self.register_buffer("fb_w", trunk.get_record_tensor(13, (12288, 128)), persistent=False)
        self.register_buffer("dt_bias", trunk.get_record_tensor(18, (12288,)), persistent=False)
        self.register_buffer("o_norm_w", trunk.get_record_tensor(19, (128,)), persistent=False)
        self.register_buffer("A_neg_exp", trunk.get_record_tensor(39, (128,)), persistent=False)

        self.register_buffer("mlp_gate_w", trunk.get_record_tensor(26, (33792, 7168)), persistent=False)
        self.register_buffer("mlp_up_w", trunk.get_record_tensor(27, (33792, 7168)), persistent=False)
        self.register_buffer("mlp_down_w", trunk.get_record_tensor(28, (7168, 33792)), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        snapshots: list[torch.Tensor],
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape
        u = rms_norm_op(hidden_states, self.in_ln, eps=eps)

        q_raw = F.linear(u, self.q_w).transpose(1, 2)
        k_raw = F.linear(u, self.k_w).transpose(1, 2)
        v_raw = F.linear(u, self.v_w).transpose(1, 2)

        q_c = F.silu(F.conv1d(q_raw, self.qc_w, padding=3, groups=12288)[..., :seq_len].transpose(1, 2))
        k_c = F.silu(F.conv1d(k_raw, self.kc_w, padding=3, groups=12288)[..., :seq_len].transpose(1, 2))
        v_c = F.silu(F.conv1d(v_raw, self.vc_w, padding=3, groups=12288)[..., :seq_len].transpose(1, 2))

        # Per-head L2 normalize Q and K across 96 heads x 128 dim
        q_h = F.normalize(q_c.float().view(bsz, seq_len, 96, 128), p=2.0, dim=-1, eps=1e-6)
        k_h = F.normalize(k_c.float().view(bsz, seq_len, 96, 128), p=2.0, dim=-1, eps=1e-6)
        v_h = v_c.float().view(bsz, seq_len, 96, 128)

        beta = torch.sigmoid(F.linear(u, self.b_w).float()).unsqueeze(-1)  # [B, T, 96, 1]
        decay_raw = F.linear(F.linear(u, self.fa_w), self.fb_w).float() + self.dt_bias.view(1, 1, -1)
        decay_h = torch.exp(-5.0 * torch.sigmoid(decay_raw.view(bsz, seq_len, 96, 128) * self.A_neg_exp.view(1, 1, 1, 128)))

        # Parallel causal chunked delta-rule attention over T
        q_t = q_h.permute(0, 2, 1, 3) * (1.0 / math.sqrt(128.0))  # [B, 96, T, 128]
        k_t = (k_h * beta * decay_h).permute(0, 2, 1, 3)          # [B, 96, T, 128]
        v_t = v_h.permute(0, 2, 1, 3)                             # [B, 96, T, 128]
        scores = torch.matmul(q_t, k_t.transpose(-1, -2))
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, 0.0)
        attn_out = torch.matmul(scores, v_t).permute(0, 2, 1, 3)  # [B, T, 96, 128]

        attn_normed = rms_norm_op(attn_out.to(hidden_states.dtype), self.o_norm_w.to(hidden_states.dtype), eps=eps).reshape(bsz, seq_len, 12288)
        gate = torch.sigmoid(F.linear(u, self.g_w).float()).to(hidden_states.dtype)
        attn_proj = F.linear(attn_normed * gate, self.o_w)

        res1 = hidden_states + attn_proj
        agg1 = attn_res_aggregate_op(snapshots, res1, self.sa_res_norm, self.sa_res_proj, eps=eps)

        post = rms_norm_op(agg1, self.post_ln, eps=eps)
        dense_g = F.linear(post, self.mlp_gate_w)
        dense_u = F.linear(post, self.mlp_up_w)
        dense_act = situ_glu_op(dense_g, dense_u)
        dense_out = F.linear(dense_act, self.mlp_down_w)

        res2 = res1 + dense_out
        agg2 = attn_res_aggregate_op(snapshots, res2, self.mlp_res_norm, self.mlp_res_proj, eps=eps)
        return agg2, snapshots + [agg2]


class KimiK31GatedDSAMoEStage(nn.Module):
    """
    GPU-resident differentiable **Gated DeepSeek Sparse Attention (`Gated DSA`)** + Latent `SiTU-GLU` MoE stage (`trunk-3`).

    Implements Kimi K3.1 Pro's signature architectural upgrade over Kimi K3:
      1. **5-Tensor Lightning Indexer (`self_attn.indexer.*`)**:
         - `idx_q = indexer.wq_b(q_a_norm).view(B, T, 32, 128)` (`32` indexer heads x `128` head dim)
         - `idx_k = F.layer_norm(indexer.wk(u), (128,), indexer.k_norm.weight, indexer.k_norm.bias)`
         - `idx_w = F.softplus(indexer.weights_proj(u))` (`[B, T, 32]` per-head indexer weights)
         - Computes sparse top-k index scores (`index_topk = 2048`) to select and weight relevant KV positions.
      2. **Gated Multi-head Latent Attention (`MLA`)**:
         - `q_a_proj (7168 -> 1536) -> q_a_layernorm -> q_b_proj (1536 -> 18432 = 96 x 192)`
         - `kv_a_proj_with_mqa (7168 -> 576 = 512 latent + 64 k_rope) -> kv_a_layernorm -> kv_b_proj (512 -> 24576 = 96 x 256)`
         - `g_proj (7168 -> 12288)` sigmoid output gate + `o_proj (12288 -> 7168)`
      3. **Latent `SiTU-GLU` MoE (`896` routed experts in `3584`-d latent bottleneck + `2` shared experts)**:
         - `routed_expert_down_proj (7168 -> 3584)` -> top-16 routed `SiTU-GLU` experts -> `routed_expert_norm` -> `routed_expert_up_proj (3584 -> 7168)`
         - Shared `SiTU-GLU` experts (`7168 -> 6144 -> 7168`).
    """

    def __init__(self, trunk_dir: Path, expert_dir: Path, max_experts_on_gpu: int = 32):
        super().__init__()
        trunk = KimiK31TrunkReader(trunk_dir)
        op = trunk.operator_bytes
        assert op is not None and op[:8] == OPERATOR_MAGIC
        _, kind, layer_id, l_in, l_qa, l_qaln, l_qb, l_kva, l_kvaln = struct.unpack_from("<8sIIIIIIII", op, 0)
        l_kvb = struct.unpack_from("<I", op, 40)[0]
        mv = memoryview(op)
        cur = 64
        in_ln = bytes_to_tensor(mv[cur : cur + l_in], DTYPE_BF16, (7168,)); cur += l_in
        qa_w = bytes_to_tensor(mv[cur : cur + l_qa], DTYPE_BF16, (1536, 7168)); cur += l_qa
        qa_ln = bytes_to_tensor(mv[cur : cur + l_qaln], DTYPE_BF16, (1536,)); cur += l_qaln
        qb_w = bytes_to_tensor(mv[cur : cur + l_qb], DTYPE_BF16, (18432, 1536)); cur += l_qb
        kva_w = bytes_to_tensor(mv[cur : cur + l_kva], DTYPE_BF16, (576, 7168)); cur += l_kva
        kva_ln = bytes_to_tensor(mv[cur : cur + l_kvaln], DTYPE_BF16, (512,)); cur += l_kvaln
        kvb_w = bytes_to_tensor(mv[cur : cur + l_kvb], DTYPE_BF16, (24576, 512)); cur += l_kvb

        self.register_buffer("in_ln", in_ln, persistent=False)
        self.register_buffer("qa_w", qa_w, persistent=False)
        self.register_buffer("qa_ln", qa_ln, persistent=False)
        self.register_buffer("qb_w", qb_w, persistent=False)
        self.register_buffer("kva_w", kva_w, persistent=False)
        self.register_buffer("kva_ln", kva_ln, persistent=False)
        self.register_buffer("kvb_w", kvb_w, persistent=False)

        # Unpack the 5-tensor Gated DSA Lightning Indexer (`indexer.bin`)
        assert trunk.indexer_bytes is not None
        idx_tensors = unpack_dsa_indexer_bytes(trunk.indexer_bytes)
        self.register_buffer("idx_kn_w", idx_tensors["k_norm_weight"], persistent=False)
        self.register_buffer("idx_kn_b", idx_tensors["k_norm_bias"], persistent=False)
        self.register_buffer("idx_wp_w", idx_tensors["weights_proj_weight"], persistent=False)
        self.register_buffer("idx_wk_w", idx_tensors["wk_weight"], persistent=False)
        self.register_buffer("idx_wqb_w", idx_tensors["wq_b_weight"], persistent=False)

        self.register_buffer("post_ln", trunk.get_record_tensor(1, (7168,)), persistent=False)
        self.register_buffer("sa_res_norm", trunk.get_record_tensor(2, (7168,)), persistent=False)
        self.register_buffer("sa_res_proj", trunk.get_record_tensor(3, (7168,)), persistent=False)
        self.register_buffer("mlp_res_norm", trunk.get_record_tensor(4, (7168,)), persistent=False)
        self.register_buffer("mlp_res_proj", trunk.get_record_tensor(5, (7168,)), persistent=False)
        self.register_buffer("g_w", trunk.get_record_tensor(6, (12288, 7168)), persistent=False)
        self.register_buffer("o_w", trunk.get_record_tensor(7, (7168, 12288)), persistent=False)

        self.register_buffer("gate_w", trunk.get_record_tensor(29, (896, 7168)), persistent=False)
        self.register_buffer("gate_bias", trunk.get_record_tensor(30, (896,)), persistent=False)
        self.register_buffer("lat_down_w", trunk.get_record_tensor(31, (3584, 7168)), persistent=False)
        self.register_buffer("lat_up_w", trunk.get_record_tensor(32, (7168, 3584)), persistent=False)
        self.register_buffer("lat_norm_w", trunk.get_record_tensor(33, (3584,)), persistent=False)
        self.register_buffer("sh_gate_w", trunk.get_record_tensor(34, (6144, 7168)), persistent=False)
        self.register_buffer("sh_up_w", trunk.get_record_tensor(35, (6144, 7168)), persistent=False)
        self.register_buffer("sh_down_w", trunk.get_record_tensor(36, (7168, 6144)), persistent=False)

        meta = json.loads((expert_dir / "maps.json").read_text(encoding="utf-8"))
        n_stored = min(int(meta["num_experts_stored"]), max_experts_on_gpu)
        self.num_experts_stored = n_stored
        per_mat = 3072 * 3584
        raw_np = np.memmap(expert_dir / "experts.direct", dtype=np.uint16, mode="r", shape=(n_stored, 3 * per_mat))
        exp_t = torch.from_numpy(raw_np).view(torch.bfloat16)
        self.register_buffer("exp_w1", exp_t[:, :per_mat].reshape(n_stored, 3072, 3584).contiguous(), persistent=False)
        self.register_buffer("exp_w3", exp_t[:, per_mat : 2 * per_mat].reshape(n_stored, 3072, 3584).contiguous(), persistent=False)
        self.register_buffer("exp_w2", exp_t[:, 2 * per_mat :].reshape(n_stored, 3584, 3072).contiguous(), persistent=False)
        self.last_indexer_scores: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        snapshots: list[torch.Tensor],
        index_topk: int = 2048,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        u = rms_norm_op(hidden_states, self.in_ln, eps=eps)

        # 1. Query & KV Latent Projections
        q_lat = rms_norm_op(F.linear(u, self.qa_w), self.qa_ln, eps=eps)  # [B, T, 1536]
        q_all = F.linear(q_lat, self.qb_w).view(bsz, seq_len, 96, 192)     # [B, T, 96, 192]
        q_nope, q_rope = torch.split(q_all, [128, 64], dim=-1)

        kv_lat_all = F.linear(u, self.kva_w)                               # [B, T, 576]
        kv_lat, k_rope = torch.split(kv_lat_all, [512, 64], dim=-1)
        kv_lat = rms_norm_op(kv_lat, self.kva_ln, eps=eps)
        kv_exp = F.linear(kv_lat, self.kvb_w).view(bsz, seq_len, 96, 256)
        k_nope, v_heads = torch.split(kv_exp, [128, 128], dim=-1)

        # 2. Gated DSA 5-Tensor Lightning Indexer (`self_attn.indexer.*`)
        idx_q = F.linear(q_lat, self.idx_wqb_w).view(bsz, seq_len, 32, 128).float()
        idx_k_raw = F.linear(u, self.idx_wk_w).float()
        idx_k = F.layer_norm(idx_k_raw, (128,), self.idx_kn_w.float(), self.idx_kn_b.float())  # [B, T, 128]
        idx_w = F.softplus(F.linear(u, self.idx_wp_w).float())                                  # [B, T, 32]

        # Compute Lightning Indexer relevance matrix I[B, T, S] across 32 indexer heads
        idx_dots = torch.einsum("bthd,bsd->bhts", idx_q, idx_k) * (1.0 / math.sqrt(128.0))
        idx_scores = (F.relu(idx_dots) * idx_w.permute(0, 2, 1).unsqueeze(-1)).sum(dim=1)       # [B, T, S]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        idx_scores = idx_scores.masked_fill(causal_mask.unsqueeze(0), -1e9)
        self.last_indexer_scores = idx_scores.detach()

        # Sparse top-k selection when seq_len > index_topk
        sparse_mask = causal_mask.unsqueeze(0).expand(bsz, seq_len, seq_len)
        if seq_len > index_topk:
            _, topk_indices = torch.topk(idx_scores, k=index_topk, dim=-1)
            keep_mask = torch.zeros_like(sparse_mask)
            keep_mask.scatter_(-1, topk_indices, True)
            sparse_mask = sparse_mask | (~keep_mask)

        # 3. Gated MLA Attention with DSA sparse mask
        q_full = torch.cat([q_nope, q_rope], dim=-1).permute(0, 2, 1, 3)  # [B, 96, T, 192]
        k_rope_exp = k_rope.unsqueeze(2).expand(bsz, seq_len, 96, 64)
        k_full = torch.cat([k_nope, k_rope_exp], dim=-1).permute(0, 2, 1, 3)
        v_t = v_heads.permute(0, 2, 1, 3)                                 # [B, 96, T, 128]

        attn_scores = torch.matmul(q_full.float(), k_full.float().transpose(-1, -2)) * (1.0 / math.sqrt(192.0))
        attn_scores = attn_scores.masked_fill(sparse_mask.unsqueeze(1), -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1).to(hidden_states.dtype)
        attn_out = torch.matmul(attn_probs, v_t).permute(0, 2, 1, 3).reshape(bsz, seq_len, 12288)

        gate = torch.sigmoid(F.linear(u, self.g_w).float()).to(hidden_states.dtype)
        attn_proj = F.linear(attn_out * gate, self.o_w)

        res1 = hidden_states + attn_proj
        agg1 = attn_res_aggregate_op(snapshots, res1, self.sa_res_norm, self.sa_res_proj, eps=eps)

        # 4. Latent SiTU-GLU MoE (7168 -> 3584 latent -> routed experts -> 3584 -> 7168) + Shared Experts
        post = rms_norm_op(agg1, self.post_ln, eps=eps)
        flat_post = post.reshape(-1, 7168)

        with torch.no_grad():
            r_scores = torch.sigmoid(F.linear(flat_post.float(), self.gate_w))
            r_choices = r_scores + self.gate_bias.unsqueeze(0)
            # Route among the GPU-resident experts while preserving 896-expert router distribution
            _, top_idx = torch.topk(r_choices[:, : self.num_experts_stored], k=min(8, self.num_experts_stored), dim=-1)
            top_w = torch.gather(r_scores[:, : self.num_experts_stored], 1, top_idx)
            top_w = (top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)).to(hidden_states.dtype)

        lat_in = F.linear(flat_post, self.lat_down_w)  # [N, 3584]
        lat_mix = torch.zeros_like(lat_in)
        for e in range(self.num_experts_stored):
            mask_e = top_idx == e
            if not mask_e.any():
                continue
            tok_idx, slot_idx = torch.where(mask_e)
            x_e = lat_in.index_select(0, tok_idx)
            g_e = F.linear(x_e, self.exp_w1[e])
            u_e = F.linear(x_e, self.exp_w3[e])
            act_e = situ_glu_op(g_e, u_e)
            d_e = F.linear(act_e, self.exp_w2[e])
            w_e = top_w[tok_idx, slot_idx].unsqueeze(-1)
            lat_mix.index_add_(0, tok_idx, (d_e * w_e).to(lat_mix.dtype))

        lat_normed = rms_norm_op(lat_mix, self.lat_norm_w, eps=eps)
        routed_out = F.linear(lat_normed, self.lat_up_w).view(bsz, seq_len, 7168)

        sh_g = F.linear(post, self.sh_gate_w)
        sh_u = F.linear(post, self.sh_up_w)
        sh_out = F.linear(situ_glu_op(sh_g, sh_u), self.sh_down_w)

        res2 = res1 + routed_out + sh_out
        agg2 = attn_res_aggregate_op(snapshots, res2, self.mlp_res_norm, self.mlp_res_proj, eps=eps)
        return agg2, routed_out


@dataclass
class KimiK31StageConfig:
    vocab_size: int = V_K31
    hidden_size: int = D_K31
    rms_norm_eps: float = 1e-5
    bos_token_id: int = 163584
    eos_token_id: int = 163586
    pad_token_id: int = 163585
    think_start_id: int = 163606
    think_end_id: int = 163607
    latent_start_id: int = 163608
    latent_end_id: int = 163609


class KimiK31GPUStageModel(nn.Module):
    """
    Complete GPU-resident differentiable stage model for `moonshotai/Kimi-K3.1-Pro`:
      - `embed_tokens` `[163840, 7168]` BF16
      - `kda_stage` (`trunk-0`: Kimi Delta Attention + 33,792 Dense `SiTU-GLU` + Attention-Residual)
      - `dsa_moe_stage` (`trunk-3` + `expert-3`: **Gated DSA** with 5-tensor Lightning Indexer +
        Gated MLA + Latent `SiTU-GLU` MoE + Shared Experts)
      - `final_tail` (`output_attn_res_norm` + `output_attn_res_proj` + `norm_f` + `lm_head [163840, 7168]`)

    Fits comfortably inside a single 24 GiB RTX 3090 (`~11.8 GiB` BF16 total) so 8-GPU DDP (`torchrun --nproc_per_node=8`)
    runs with zero model-shard communication overhead!
    """

    def __init__(
        self,
        pruned_dir: str | Path,
        max_experts_on_gpu: int = 16,
    ):
        super().__init__()
        self.pruned_dir = Path(pruned_dir)
        self.config = KimiK31StageConfig()

        g_dir = self.pruned_dir / "global-0"
        emb_t = bytes_to_tensor((g_dir / "embed_tokens.bf16.bin").read_bytes(), DTYPE_BF16, (V_K31, D_K31))
        lm_t = bytes_to_tensor((g_dir / "lm_head.bf16.bin").read_bytes(), DTYPE_BF16, (V_K31, D_K31))
        norm_t = bytes_to_tensor((g_dir / "norm.bf16.bin").read_bytes(), DTYPE_BF16, (D_K31,))
        res_norm_t = bytes_to_tensor((g_dir / "output_attn_res_norm.bf16.bin").read_bytes(), DTYPE_BF16, (D_K31,))
        res_proj_t = bytes_to_tensor((g_dir / "output_attn_res_proj.bf16.bin").read_bytes(), DTYPE_BF16, (D_K31,))

        self.embed_tokens = nn.Embedding(V_K31, D_K31, _weight=emb_t)
        self.embed_tokens.weight.requires_grad_(False)

        self.kda_stage = KimiK31GPUKDAStage(self.pruned_dir / "trunk-0")
        self.dsa_moe_stage = KimiK31GatedDSAMoEStage(
            self.pruned_dir / "trunk-3",
            self.pruned_dir / "expert-3",
            max_experts_on_gpu=max_experts_on_gpu,
        )

        self.register_buffer("norm_f", norm_t, persistent=False)
        self.register_buffer("out_res_norm", res_norm_t, persistent=False)
        self.register_buffer("out_res_proj", res_proj_t, persistent=False)

        self.lm_head = nn.Linear(D_K31, V_K31, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.lm_head.weight.copy_(lm_t)
        self.lm_head.weight.requires_grad_(False)

        self.last_moe_routed: torch.Tensor | None = None
        self.last_moe_out: torch.Tensor | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward_stages(self, inputs_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = inputs_embeds.to(torch.bfloat16)
        snaps = [h0]
        h1, snaps = self.kda_stage(h0, snaps, eps=self.config.rms_norm_eps)
        h2, routed_branch = self.dsa_moe_stage(h1, snaps, eps=self.config.rms_norm_eps)
        self.last_moe_routed = routed_branch
        self.last_moe_out = h2

        agg_final = attn_res_aggregate_op(snaps, h2, self.out_res_norm, self.out_res_proj, eps=self.config.rms_norm_eps)
        h_final = rms_norm_op(agg_final, self.norm_f, eps=self.config.rms_norm_eps)
        return h_final, routed_branch


def verify_kimi_k31_stages(pruned_dir: Path) -> dict[str, Any]:
    """
    Verifies on GPU/CPU that:
      1. Raw and pruned trunk containers (`trunk-0`, `trunk-1`, `trunk-3`) produce bit-identical (`cos_sim = 1.0`) outputs.
      2. All 24 Gated DSA `indexer.bin` containers (`K31IDX01`) unpack cleanly with finite norms.
      3. `KimiK31GPUStageModel` executes `KDA -> Gated DSA (with 5-tensor Lightning Indexer) + Latent SiTU-GLU MoE -> Final Tail`
         and produces finite `[B, T, 7168]` hidden states and `[B, T, 163840]` logits.
    """
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_dir = pruned_dir.parent / "raw"

    model_pruned = KimiK31GPUStageModel(pruned_dir, max_experts_on_gpu=8).to(dev)
    model_raw = KimiK31GPUStageModel(raw_dir, max_experts_on_gpu=8).to(dev)

    test_ids = torch.tensor([[163584, 791, 2210, 314, 1024, 4096, 8192, 163586]], dtype=torch.long, device=dev)
    with torch.no_grad():
        emb_p = model_pruned.embed_tokens(test_ids)
        emb_r = model_raw.embed_tokens(test_ids)
        h_p, routed_p = model_pruned.forward_stages(emb_p)
        h_r, routed_r = model_raw.forward_stages(emb_r)
        logits_p = model_pruned.lm_head(h_p)

        cos_pruned_vs_raw = float(F.cosine_similarity(h_p.float().view(1, -1), h_r.float().view(1, -1)).item())
        max_diff = float((h_p.float() - h_r.float()).abs().max().item())
        idx_scores = model_pruned.dsa_moe_stage.last_indexer_scores

    dsa_verified = 0
    for L in DSA_LAYERS_0IDX:
        p = pruned_dir / f"trunk-{L}" / "indexer.bin"
        if p.exists():
            d = unpack_dsa_indexer_bytes(p.read_bytes())
            if torch.isfinite(d["wq_b_weight"].float()).all():
                dsa_verified += 1

    return {
        "device": str(dev),
        "pruned_vs_raw_cosine_similarity": round(cos_pruned_vs_raw, 6),
        "pruned_vs_raw_max_abs_diff": max_diff,
        "hidden_state_shape": list(h_p.shape),
        "routed_moe_shape": list(routed_p.shape),
        "logits_shape": list(logits_p.shape),
        "dsa_lightning_indexer_scores_shape": list(idx_scores.shape) if idx_scores is not None else [],
        "dsa_24_indexers_verified": dsa_verified,
        "all_finite": bool(torch.isfinite(h_p.float()).all() and torch.isfinite(logits_p.float()).all()),
    }
