#!/usr/bin/env python3
"""
Precomputed Dataset & Composed Operator Builder for Liquid AI LFM3-1.7B
(`clover-ai-system/src/models/lfm3-1.7b/prepare_lfm3_datasets.py`).

Adapts Clover's `Kimi K3` and `Nemotron 3.5 Super VL` precomputed trunk & operator architecture
to Liquid AI's 24-layer tri-hybrid `LFM3-1.7B` (`LiquidAI/LFM3-1.7B`, `D_model = 2048`, `V = 128000` tied):

1. `global-0/`:
   - `embed_tokens.bf16.bin`: `[128000, 2048]` BF16 (`524,288,000` B — serves BOTH Stage 1 input embedding and Stage 5 tied `lm_head`!)
   - `embedding_norm.bf16.bin`: `[2048]` BF16 (`4,096` B — final RMSNorm before tied `lm_head`)
   - `rope_inv_freq.bin`: Precomputed FP32 inverse frequency tables (`64` floats each = `512` B total) for:
       * `full_attention` (`rope_theta = 1,000,000.0`, `partial_rotary_factor = 0.5`: 32 active frequencies + 32 exact zeros)
       * `sliding_attention` (`rope_theta = 10,000.0`, `partial_rotary_factor = 1.0`: 64 active frequencies)

2. 13 Gated Short Convolution Trunks (`trunk-L` for `L in {0,1,3,5,7,9,11,13,15,17,19,21,23}`, `layer_kind = 0`):
   - `index.bin` (`LFM3TRK1` header + 9 record descriptors):
       * Group 0 (`common.bin`):
           - Record 0: `operator_norm.weight` (`[2048]`, BF16) -> covered by `operator.bin`, dropped by `prune_lfm3_trunk.py`!
           - Record 1: `post_operator_norm.weight` (`[2048]`, BF16)
           - Record 2: `conv.out_proj.weight` (`[2048, 2048]`, BF16)
           - Record 3: `ffn_norm.weight` (`[2048]`, BF16)
           - Record 4: `post_ffn_norm.weight` (`[2048]`, BF16)
       * Group 1 (`attention.bin`):
           - Record 5: `conv.in_proj.weight` (`[6144, 2048]`, BF16) -> covered by `operator.bin`, dropped by pruner!
           - Record 6: `conv.conv.weight` (`[2048, 1, 7]`, BF16) -> covered by `operator.bin`, dropped by pruner!
       * Group 2 (`ffn.bin`):
           - Record 10: `feed_forward.w1.weight` (`[7680, 2048]`, BF16)
           - Record 11: `feed_forward.w3.weight` (`[7680, 2048]`, BF16)
           - Record 12: `feed_forward.w2.weight` (`[2048, 7680]`, BF16)
   - `operator.bin` (`LFM3_OP1`, `kind = 0`):
       * Fuses `operator_norm.weight` (`[2048]`), `conv.in_proj.weight` (`[6144, 2048]`), and `conv.conv.weight` (`[2048, 7]`) into one contiguous mmap-ready binary (`25,198,144` B).

3. 11 GQA Attention Trunks (`trunk-L` for 5 sliding layers `{4,8,12,16,20}` [`layer_kind = 1`] and 6 full layers `{2,6,10,14,18,22}` [`layer_kind = 2`]):
   - `index.bin` (`LFM3TRK1` header + 14 record descriptors):
       * Group 0 (`common.bin`):
           - Record 0: `operator_norm.weight` (`[2048]`, BF16) -> covered by `operator.bin`, dropped by pruner!
           - Record 1: `post_operator_norm.weight` (`[2048]`, BF16)
           - Record 2: `self_attn.o_proj.weight` (`[2048, 2048]`, BF16)
           - Record 3: `ffn_norm.weight` (`[2048]`, BF16)
           - Record 4: `post_ffn_norm.weight` (`[2048]`, BF16)
       * Group 1 (`attention.bin`):
           - Record 5: `self_attn.q_proj.weight` (`[2048, 2048]`, BF16) -> covered by `operator.bin`, dropped by pruner!
           - Record 6: `self_attn.k_proj.weight` (`[512, 2048]`, BF16) -> covered by `operator.bin`, dropped by pruner!
           - Record 7: `self_attn.v_proj.weight` (`[512, 2048]`, BF16) -> covered by `operator.bin`, dropped by pruner!
           - Record 8: `self_attn.gate_proj.weight` (`[16, 2048]`, BF16) -> covered by `operator.bin`, dropped by pruner!
           - Record 9: `self_attn.q_norm.weight` (`[128]`, BF16) -> covered by `operator.bin`, dropped by pruner!
           - Record 13: `self_attn.k_norm.weight` (`[128]`, BF16) -> covered by `operator.bin`, dropped by pruner!
       * Group 2 (`ffn.bin`):
           - Record 10: `feed_forward.w1.weight` (`[7680, 2048]`, BF16)
           - Record 11: `feed_forward.w3.weight` (`[7680, 2048]`, BF16)
           - Record 12: `feed_forward.w2.weight` (`[2048, 7680]`, BF16)
   - `operator.bin` (`LFM3_OP1`, `kind = 1` or `2`):
       * Fuses `operator_norm.weight` (`[2048]`), `q_proj` (`[2048, 2048]`), `k_proj` (`[512, 2048]`), `v_proj` (`[512, 2048]`), `gate_proj` (`[16, 2048]`), `q_norm` (`[128]`), and `k_norm` (`[128]`) into one contiguous binary (`12,653,120` B).
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

TRUNK_MAGIC = b"LFM3TRK1"
OPERATOR_MAGIC = b"LFM3_OP1"
ROPE_MAGIC = b"LFM3ROP1"

DTYPE_BF16 = 1
DTYPE_FP32 = 2

HEADER_FMT = "<8sIIIIQQQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 72 bytes
RECORD_FMT = "<IIIIQQI"
RECORD_SIZE = struct.calcsize(RECORD_FMT)  # 36 bytes

GROUP_NAMES = ("common.bin", "attention.bin", "ffn.bin")


def tensor_to_le_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return bytes(t.view(torch.uint16).numpy().tobytes())
    if t.dtype == torch.float32:
        return bytes(t.numpy().tobytes())
    raise ValueError(f"Unsupported tensor dtype: {t.dtype}")


def write_lfm3_trunk_folder(
    out_dir: Path,
    layer_id: int,
    layer_kind: int,
    records_spec: list[tuple[int, int, int, int, int, bytes]],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    group_buffers = [bytearray(), bytearray(), bytearray()]
    packed_records = []

    for rec_id, grp_id, dtype_code, rows, cols, raw_bytes in records_spec:
        offset = len(group_buffers[grp_id])
        length = len(raw_bytes)
        group_buffers[grp_id].extend(raw_bytes)
        packed_records.append((rec_id, grp_id, dtype_code, rows, cols, offset, length))

    group_lengths = [len(b) for b in group_buffers]
    for grp_id, grp_name in enumerate(GROUP_NAMES):
        grp_path = out_dir / grp_name
        if group_lengths[grp_id] > 0:
            grp_path.write_bytes(group_buffers[grp_id])
        elif grp_path.exists():
            grp_path.unlink()

    header_bytes = struct.pack(
        HEADER_FMT,
        TRUNK_MAGIC,
        1,
        layer_id,
        layer_kind,
        len(packed_records),
        0,
        group_lengths[0],
        0,
        group_lengths[1],
        0,
        group_lengths[2],
    )
    index_buf = bytearray(header_bytes)
    for rec_id, grp_id, dtype_code, rows, cols, offset, length in packed_records:
        index_buf.extend(
            struct.pack(RECORD_FMT, rec_id, grp_id, dtype_code, rows, offset, length, cols)
        )
    (out_dir / "index.bin").write_bytes(index_buf)

    return {
        "layer_id": layer_id,
        "layer_kind": layer_kind,
        "record_count": len(packed_records),
        "group_lengths": group_lengths,
        "total_group_bytes": sum(group_lengths),
    }


def build_conv_layer(
    st_file,
    layer_id: int,
    trunk_dir: Path,
) -> dict[str, Any]:
    p = f"model.layers.{layer_id}."
    op_norm = st_file.get_tensor(p + "operator_norm.weight")
    post_op_norm = st_file.get_tensor(p + "post_operator_norm.weight")
    ffn_norm = st_file.get_tensor(p + "ffn_norm.weight")
    post_ffn_norm = st_file.get_tensor(p + "post_ffn_norm.weight")

    in_proj = st_file.get_tensor(p + "conv.in_proj.weight")  # [6144, 2048]
    conv_w = st_file.get_tensor(p + "conv.conv.weight")  # [2048, 1, 7]
    out_proj = st_file.get_tensor(p + "conv.out_proj.weight")  # [2048, 2048]

    w1 = st_file.get_tensor(p + "feed_forward.w1.weight")  # [7680, 2048]
    w3 = st_file.get_tensor(p + "feed_forward.w3.weight")  # [7680, 2048]
    w2 = st_file.get_tensor(p + "feed_forward.w2.weight")  # [2048, 7680]

    b_op_norm = tensor_to_le_bytes(op_norm)
    b_post_op_norm = tensor_to_le_bytes(post_op_norm)
    b_out_proj = tensor_to_le_bytes(out_proj)
    b_ffn_norm = tensor_to_le_bytes(ffn_norm)
    b_post_ffn_norm = tensor_to_le_bytes(post_ffn_norm)

    b_in_proj = tensor_to_le_bytes(in_proj)
    b_conv_w = tensor_to_le_bytes(conv_w.squeeze(1))

    b_w1 = tensor_to_le_bytes(w1)
    b_w3 = tensor_to_le_bytes(w3)
    b_w2 = tensor_to_le_bytes(w2)

    records_spec = [
        (0, 0, DTYPE_BF16, 2048, 1, b_op_norm),
        (1, 0, DTYPE_BF16, 2048, 1, b_post_op_norm),
        (2, 0, DTYPE_BF16, 2048, 2048, b_out_proj),
        (3, 0, DTYPE_BF16, 2048, 1, b_ffn_norm),
        (4, 0, DTYPE_BF16, 2048, 1, b_post_ffn_norm),
        (5, 1, DTYPE_BF16, 6144, 2048, b_in_proj),
        (6, 1, DTYPE_BF16, 2048, 7, b_conv_w),
        (10, 2, DTYPE_BF16, 7680, 2048, b_w1),
        (11, 2, DTYPE_BF16, 7680, 2048, b_w3),
        (12, 2, DTYPE_BF16, 2048, 7680, b_w2),
    ]
    info = write_lfm3_trunk_folder(trunk_dir, layer_id=layer_id, layer_kind=0, records_spec=records_spec)

    # Build composed `operator.bin` (64-byte header + op_norm + in_proj + conv_w)
    op_header = struct.pack(
        "<8sIIIII36x",
        OPERATOR_MAGIC,
        0,  # kind = 0 (ShortConv)
        layer_id,
        len(b_op_norm),
        len(b_in_proj),
        len(b_conv_w),
    )
    op_payload = b"".join([op_header, b_op_norm, b_in_proj, b_conv_w])
    (trunk_dir / "operator.bin").write_bytes(op_payload)
    info["operator_bytes"] = len(op_payload)
    return info


def build_attention_layer(
    st_file,
    layer_id: int,
    layer_kind: int,
    trunk_dir: Path,
) -> dict[str, Any]:
    p = f"model.layers.{layer_id}."
    op_norm = st_file.get_tensor(p + "operator_norm.weight")
    post_op_norm = st_file.get_tensor(p + "post_operator_norm.weight")
    ffn_norm = st_file.get_tensor(p + "ffn_norm.weight")
    post_ffn_norm = st_file.get_tensor(p + "post_ffn_norm.weight")

    q_proj = st_file.get_tensor(p + "self_attn.q_proj.weight")  # [2048, 2048]
    k_proj = st_file.get_tensor(p + "self_attn.k_proj.weight")  # [512, 2048]
    v_proj = st_file.get_tensor(p + "self_attn.v_proj.weight")  # [512, 2048]
    o_proj = st_file.get_tensor(p + "self_attn.o_proj.weight")  # [2048, 2048]
    gate_proj = st_file.get_tensor(p + "self_attn.gate_proj.weight")  # [16, 2048]
    q_norm = st_file.get_tensor(p + "self_attn.q_norm.weight")  # [128]
    k_norm = st_file.get_tensor(p + "self_attn.k_norm.weight")  # [128]

    w1 = st_file.get_tensor(p + "feed_forward.w1.weight")
    w3 = st_file.get_tensor(p + "feed_forward.w3.weight")
    w2 = st_file.get_tensor(p + "feed_forward.w2.weight")

    b_op_norm = tensor_to_le_bytes(op_norm)
    b_post_op_norm = tensor_to_le_bytes(post_op_norm)
    b_o_proj = tensor_to_le_bytes(o_proj)
    b_ffn_norm = tensor_to_le_bytes(ffn_norm)
    b_post_ffn_norm = tensor_to_le_bytes(post_ffn_norm)

    b_q = tensor_to_le_bytes(q_proj)
    b_k = tensor_to_le_bytes(k_proj)
    b_v = tensor_to_le_bytes(v_proj)
    b_gate = tensor_to_le_bytes(gate_proj)
    b_qn = tensor_to_le_bytes(q_norm)
    b_kn = tensor_to_le_bytes(k_norm)

    b_w1 = tensor_to_le_bytes(w1)
    b_w3 = tensor_to_le_bytes(w3)
    b_w2 = tensor_to_le_bytes(w2)

    records_spec = [
        (0, 0, DTYPE_BF16, 2048, 1, b_op_norm),
        (1, 0, DTYPE_BF16, 2048, 1, b_post_op_norm),
        (2, 0, DTYPE_BF16, 2048, 2048, b_o_proj),
        (3, 0, DTYPE_BF16, 2048, 1, b_ffn_norm),
        (4, 0, DTYPE_BF16, 2048, 1, b_post_ffn_norm),
        (5, 1, DTYPE_BF16, 2048, 2048, b_q),
        (6, 1, DTYPE_BF16, 512, 2048, b_k),
        (7, 1, DTYPE_BF16, 512, 2048, b_v),
        (8, 1, DTYPE_BF16, 16, 2048, b_gate),
        (9, 1, DTYPE_BF16, 128, 1, b_qn),
        (13, 1, DTYPE_BF16, 128, 1, b_kn),
        (10, 2, DTYPE_BF16, 7680, 2048, b_w1),
        (11, 2, DTYPE_BF16, 7680, 2048, b_w3),
        (12, 2, DTYPE_BF16, 2048, 7680, b_w2),
    ]
    info = write_lfm3_trunk_folder(trunk_dir, layer_id=layer_id, layer_kind=layer_kind, records_spec=records_spec)

    op_header = struct.pack(
        "<8sIIIIIIIII20x",
        OPERATOR_MAGIC,
        layer_kind,
        layer_id,
        len(b_op_norm),
        len(b_q),
        len(b_k),
        len(b_v),
        len(b_gate),
        len(b_qn),
        len(b_kn),
    )
    op_payload = b"".join([op_header, b_op_norm, b_q, b_k, b_v, b_gate, b_qn, b_kn])
    (trunk_dir / "operator.bin").write_bytes(op_payload)
    info["operator_bytes"] = len(op_payload)
    return info


def prepare_all_24_lfm3_layers(
    source_model_dir: Path,
    computed_dir: Path,
) -> dict[str, Any]:
    """
    Prepares the entire 24-layer `LiquidAI/LFM3-1.7B` checkpoint into precomputed `LFM3TRK1` trunks
    (`trunk-0..23`), composed `operator.bin` streams, and `global-0/` (tied `embed_tokens`, `embedding_norm`,
    and precomputed zero-tail partial/full RoPE frequency tables).
    """
    t0 = time.perf_counter()
    computed_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((source_model_dir / "config.json").read_text(encoding="utf-8"))
    layer_types = cfg["layer_types"]

    st_path = source_model_dir / "model.safetensors"
    with safe_open(str(st_path), framework="pt", device="cpu") as st_file:
        # 1. `global-0/`
        global_dir = computed_dir / "global-0"
        global_dir.mkdir(parents=True, exist_ok=True)
        emb_w = st_file.get_tensor("model.embed_tokens.weight")  # [128000, 2048]
        emb_norm_w = st_file.get_tensor("model.embedding_norm.weight")  # [2048]
        b_emb = tensor_to_le_bytes(emb_w)
        b_emb_norm = tensor_to_le_bytes(emb_norm_w)
        (global_dir / "embed_tokens.bf16.bin").write_bytes(b_emb)
        (global_dir / "embedding_norm.bf16.bin").write_bytes(b_emb_norm)

        # Precompute `inv_freq` vectors for full_attention (zero-tail partial 0.5) and sliding_attention (1.0)
        head_dim = cfg["head_dim"]  # 128
        full_theta = float(cfg["rope_parameters"]["full_attention"]["rope_theta"])
        full_factor = float(cfg["rope_parameters"]["full_attention"].get("partial_rotary_factor", 0.5))
        full_angles = int(full_factor * head_dim) // 2
        inv_full = 1.0 / (full_theta ** (torch.arange(0, 2 * full_angles, 2, dtype=torch.float32) / head_dim))
        inv_full = torch.cat([inv_full, torch.zeros(head_dim // 2 - full_angles, dtype=torch.float32)])

        slid_theta = float(cfg["rope_parameters"]["sliding_attention"]["rope_theta"])
        slid_factor = float(cfg["rope_parameters"]["sliding_attention"].get("partial_rotary_factor", 1.0))
        slid_angles = int(slid_factor * head_dim) // 2
        inv_slid = 1.0 / (slid_theta ** (torch.arange(0, 2 * slid_angles, 2, dtype=torch.float32) / head_dim))
        inv_slid = torch.cat([inv_slid, torch.zeros(head_dim // 2 - slid_angles, dtype=torch.float32)])

        b_inv_full = tensor_to_le_bytes(inv_full)
        b_inv_slid = tensor_to_le_bytes(inv_slid)
        rope_header = struct.pack("<8sIII", ROPE_MAGIC, head_dim, len(b_inv_full), len(b_inv_slid))
        (global_dir / "rope_inv_freq.bin").write_bytes(rope_header + b_inv_full + b_inv_slid)

        # 2. All 24 layers (`trunk-0` .. `trunk-23`)
        layer_summaries = []
        for idx in range(24):
            ltype = layer_types[idx]
            trunk_dir = computed_dir / f"trunk-{idx}"
            if ltype == "conv":
                l_info = build_conv_layer(st_file, idx, trunk_dir)
            elif ltype == "sliding_attention":
                l_info = build_attention_layer(st_file, idx, 1, trunk_dir)
            else:
                l_info = build_attention_layer(st_file, idx, 2, trunk_dir)
            layer_summaries.append(l_info)

    elapsed = time.perf_counter() - t0
    summary = {
        "total_prepare_seconds": round(elapsed, 3),
        "computed_dir": str(computed_dir),
        "global_0": {
            "embed_tokens_bytes": len(b_emb),
            "embedding_norm_bytes": len(b_emb_norm),
            "rope_inv_freq_bytes": len(rope_header) + len(b_inv_full) + len(b_inv_slid),
        },
        "num_trunks_built": len(layer_summaries),
        "sample_trunks": [layer_summaries[0], layer_summaries[2], layer_summaries[4]],
    }
    (computed_dir / "prepare_lfm3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
