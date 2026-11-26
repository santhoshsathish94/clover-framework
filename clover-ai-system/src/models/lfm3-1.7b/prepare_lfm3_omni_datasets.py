#!/usr/bin/env python3
"""
Precomputed Trunk + 4-Bit Expert Builder for Official `LiquidAI/LFM3-Omni-2B`
(`clover-ai-system/src/models/lfm3-1.7b/prepare_lfm3_omni_datasets.py`).

Precomputes ALL static trunks and dense SwiGLU/MLP experts of `LiquidAI/LFM3-Omni-2B` (`2,166.93M` params)
ONCE so runtime execution requires zero weight repacking, zero duplicate attention buffers, and `O(1)` `mmap`:

1. Language Backbone (`model.language_model`, `24` Tri-Hybrid layers, `1,728.80M` params -> `liblfm3_cpu_engine`):
   - `global-0/`:
       * `embed_tokens.bf16.bin` (`[128000, 2048]`, `524,288,000` B)
       * `embedding_norm.bf16.bin` (`[2048]`, `4,096` B)
       * `rope_inv_freq.bin` (`LFM3ROP1`, `532` B)
       * `lm_head.4bit.bin` (`[128000, 2048]` planar Group-64, `135,680,000` B — Two-Stage 4-bit + Top-64 BF16 exact)
   - `trunk-0..23/`:
       * Pruned `index.bin` + `common.bin` (`attention.bin` 100% pruned across all 24 layers = `445.14 MiB` saved!)
       * Composed `operator.bin` (`LFM3_OP1`) + planar Group-64 `operator.4bit.bin` (`in_proj+out_proj` or `qkv+o_proj`)
       * Planar Group-64 `ffn.4bit.bin` (`24,422,400` B/layer = `3.864x` compression vs `94.37 MB` BF16 `ffn.bin`)
   - `experts/expert-0..23/`:
       * `constants.bin`, `maps.json`, and hardlink/copy of `ffn.4bit.bin`

2. Multimodal Encoders, Projectors & Speech Depthformer (`438.13M` params -> `omni-multimodal/`):
   - `vision-trunk-0..13/` (`14` LFM3 vision layers: composed `operator.bin` + `operator.4bit.bin` + `ffn.4bit.bin`)
   - `vision-global-0/` (`vision_embeddings` + algebraically fused `multi_modal_projector`: `LN` fused into `linear_1` + 4-bit)
   - `audio-trunk-0..16/` (`17` LFM2 causal audio layers: composed `operator.bin` + `operator.4bit.bin` + `ffn.4bit.bin`)
   - `audio-global-0/` (`downsample` + `embedding_norm` + algebraically fused `audio_projector`: `LN` fused into `linear_1` + 4-bit)
   - `depth-global-0/` (`depth_decoder` 3 layers + `codebook_embedding.4bit.bin` `[24204, 2048]`)
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import shutil
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

TRUNK_MAGIC = b"LFM3TRK1"
OPERATOR_MAGIC = b"LFM3_OP1"
ROPE_MAGIC = b"LFM3ROP1"

DTYPE_BF16 = 1
DTYPE_FP32 = 2

HEADER_FMT = "<8sIIIIQQQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
RECORD_FMT = "<IIIIQQI"
RECORD_SIZE = struct.calcsize(RECORD_FMT)

FFN_4BIT_MAT_BYTES = 8140800
OP_CONV_IN_4BIT_BYTES = 6512640
OP_ATTN_QKV_4BIT_BYTES = 3256320
OP_OUT_4BIT_BYTES = 2170880
LM_HEAD_4BIT_BYTES = 135680000


class SafetensorsMmapReader:
    """Zero-dependency safetensors mmap reader returning raw BF16 bytes or FP32 numpy arrays."""

    def __init__(self, path: Path):
        self._fh = open(path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        hdr_len = struct.unpack("<Q", self._mm[:8])[0]
        self.header = json.loads(bytes(self._mm[8 : 8 + hdr_len]).decode("utf-8"))
        self._base = 8 + hdr_len

    def close(self) -> None:
        self._mm.close()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def get_raw_bytes(self, name: str) -> bytes:
        info = self.header[name]
        s, e = info["data_offsets"]
        return bytes(self._mm[self._base + s : self._base + e])

    def get_f32(self, name: str) -> np.ndarray:
        info = self.header[name]
        s, e = info["data_offsets"]
        shape = tuple(info["shape"])
        dt = info["dtype"]
        buf = memoryview(self._mm)[self._base + s : self._base + e]
        if dt == "BF16":
            u16 = np.frombuffer(buf, dtype="<u2").reshape(shape)
            return (u16.astype(np.uint32) << 16).view(np.float32)
        if dt == "F32":
            return np.frombuffer(buf, dtype="<f4").reshape(shape).copy()
        raise ValueError(f"Unsupported safetensors dtype {dt} for {name}")


def f32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    return ((u32 + bias) >> np.uint32(16)).astype("<u2")


def quantize_matrix_4bit_group64(w_f: np.ndarray, bf16_round_first: bool = False) -> bytes:
    """Clover canonical planar 4-bit group-64 packer: [nibbles] + [steps] + [row_scales_fp32]."""
    if bf16_round_first:
        w_f = (f32_to_bf16_u16(w_f).astype(np.uint32) << 16).view(np.float32)
    rows, cols = w_f.shape
    assert cols % 64 == 0, f"cols={cols} must be divisible by 64"
    groups = cols // 64

    chunk_rows = 16384
    nib_parts, stp_parts, scl_parts = [], [], []
    for r0 in range(0, rows, chunk_rows):
        sub = w_f[r0 : min(rows, r0 + chunk_rows)]
        r_sub = sub.shape[0]
        row_max = np.maximum(np.max(np.abs(sub), axis=1, keepdims=True), 1e-12)
        row_scales = (row_max / 127.0).astype(np.float32)
        codes_i8 = np.clip(np.rint(sub / row_scales), -127, 127).astype(np.int16)

        codes_g = codes_i8.reshape(r_sub, groups, 64)
        g_max = np.max(np.abs(codes_g), axis=-1)
        steps = np.where(g_max > 0, (g_max + 6) // 7, 1).astype(np.uint8)

        q_vals = np.clip(np.rint(codes_g.astype(np.float32) / steps[:, :, None].astype(np.float32)), -8, 7).astype(
            np.int16
        ) + 8
        low = q_vals[:, :, :32].astype(np.uint8)
        high = q_vals[:, :, 32:].astype(np.uint8)
        packed = low | (high << 4)
        nib_parts.append(np.ascontiguousarray(packed).tobytes())
        stp_parts.append(np.ascontiguousarray(steps).tobytes())
        scl_parts.append(np.ascontiguousarray(row_scales.reshape(-1)).tobytes())

    return b"".join(nib_parts) + b"".join(stp_parts) + b"".join(scl_parts)


def write_pruned_lfm3_trunk(
    trunk_dir: Path,
    layer_id: int,
    layer_kind: int,
    records_spec: list[tuple[int, int, int, int, int, bytes]],
    keep_ffn_bf16: bool = True,
) -> dict[str, Any]:
    """Writes directly in PRUNED `LFM3TRK1` layout (`attention.bin` omitted, `length=0` for operator-covered records)."""
    trunk_dir.mkdir(parents=True, exist_ok=True)
    keep_ids = {1, 2, 3, 4, 10, 11, 12} if keep_ffn_bf16 else {1, 2, 3, 4}
    group_buffers = [bytearray(), bytearray(), bytearray()]
    packed_records = []
    old_total = 0

    for rec_id, grp_id, dtype_code, rows, cols, raw_bytes in records_spec:
        old_total += len(raw_bytes)
        if rec_id in keep_ids and len(raw_bytes) > 0:
            offset = len(group_buffers[grp_id])
            length = len(raw_bytes)
            group_buffers[grp_id].extend(raw_bytes)
            packed_records.append((rec_id, grp_id, dtype_code, rows, cols, offset, length))
        else:
            packed_records.append((rec_id, grp_id, dtype_code, rows, cols, 0, 0))

    group_names = ("common.bin", "attention.bin", "ffn.bin")
    group_lengths = [len(b) for b in group_buffers]
    for grp_id, grp_name in enumerate(group_names):
        grp_path = trunk_dir / grp_name
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
        index_buf.extend(struct.pack(RECORD_FMT, rec_id, grp_id, dtype_code, rows, offset, length, cols))
    (trunk_dir / "index.bin").write_bytes(index_buf)

    return {
        "layer_id": layer_id,
        "layer_kind": layer_kind,
        "old_bf16_group_bytes": old_total,
        "pruned_group_bytes": sum(group_lengths),
        "dropped_duplicate_bytes": old_total - sum(group_lengths),
    }


def pack_ffn_4bit(w1: np.ndarray, w3: np.ndarray, w2: np.ndarray) -> bytes:
    b1 = quantize_matrix_4bit_group64(w1)
    b3 = quantize_matrix_4bit_group64(w3)
    b2_raw = quantize_matrix_4bit_group64(w2)
    b2 = b2_raw + (b"\x00" * max(0, FFN_4BIT_MAT_BYTES - len(b2_raw)))
    blob = b1 + b3 + b2
    assert len(blob) == 3 * FFN_4BIT_MAT_BYTES
    return blob


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def build_language_backbone_24_layers(
    st_file: SafetensorsMmapReader,
    text_cfg: dict[str, Any],
    pruned_dir: Path | None,
    soc_q4_dir: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    primary_dir = pruned_dir if pruned_dir is not None else soc_q4_dir
    assert primary_dir is not None
    primary_dir.mkdir(parents=True, exist_ok=True)
    global_dir = primary_dir / "global-0"
    global_dir.mkdir(parents=True, exist_ok=True)
    experts_dir = primary_dir / "experts"
    experts_dir.mkdir(parents=True, exist_ok=True)

    b_emb = st_file.get_raw_bytes("model.language_model.embed_tokens.weight")
    b_emb_norm = st_file.get_raw_bytes("model.language_model.embedding_norm.weight")
    (global_dir / "embed_tokens.bf16.bin").write_bytes(b_emb)
    (global_dir / "embedding_norm.bf16.bin").write_bytes(b_emb_norm)

    print("  [Omni-Precompute] Quantizing global-0/lm_head.4bit.bin [128000, 2048]...", flush=True)
    emb_f32 = st_file.get_f32("model.language_model.embed_tokens.weight")
    lm_head_q4 = quantize_matrix_4bit_group64(emb_f32)
    del emb_f32
    assert len(lm_head_q4) == LM_HEAD_4BIT_BYTES
    (global_dir / "lm_head.4bit.bin").write_bytes(lm_head_q4)

    head_dim = int(text_cfg["head_dim"])
    full_theta = float(text_cfg["rope_parameters"]["full_attention"]["rope_theta"])
    full_factor = float(text_cfg["rope_parameters"]["full_attention"].get("partial_rotary_factor", 0.5))
    full_angles = int(full_factor * head_dim) // 2
    inv_full = 1.0 / (full_theta ** (np.arange(0, 2 * full_angles, 2, dtype=np.float32) / head_dim))
    inv_full = np.concatenate([inv_full, np.zeros(head_dim // 2 - full_angles, dtype=np.float32)]).astype("<f4")

    slid_theta = float(text_cfg["rope_parameters"]["sliding_attention"]["rope_theta"])
    slid_factor = float(text_cfg["rope_parameters"]["sliding_attention"].get("partial_rotary_factor", 1.0))
    slid_angles = int(slid_factor * head_dim) // 2
    inv_slid = 1.0 / (slid_theta ** (np.arange(0, 2 * slid_angles, 2, dtype=np.float32) / head_dim))
    inv_slid = np.concatenate([inv_slid, np.zeros(head_dim // 2 - slid_angles, dtype=np.float32)]).astype("<f4")

    b_inv_full = inv_full.tobytes()
    b_inv_slid = inv_slid.tobytes()
    rope_blob = struct.pack("<8sIII", ROPE_MAGIC, head_dim, len(b_inv_full), len(b_inv_slid)) + b_inv_full + b_inv_slid
    (global_dir / "rope_inv_freq.bin").write_bytes(rope_blob)

    if pruned_dir is not None and soc_q4_dir is not None:
        soc_g = soc_q4_dir / "global-0"
        soc_g.mkdir(parents=True, exist_ok=True)
        for fn in ("embed_tokens.bf16.bin", "embedding_norm.bf16.bin", "lm_head.4bit.bin", "rope_inv_freq.bin"):
            _link_or_copy(global_dir / fn, soc_g / fn)

    layer_types = text_cfg["layer_types"]
    layer_stats = []
    total_op_4bit = 0
    total_ffn_4bit = 0

    for idx in range(24):
        ltype = layer_types[idx]
        p = f"model.language_model.layers.{idx}."
        trunk_dir = primary_dir / f"trunk-{idx}"

        b_op_norm = st_file.get_raw_bytes(p + "operator_norm.weight")
        b_post_op_norm = st_file.get_raw_bytes(p + "post_operator_norm.weight")
        b_ffn_norm = st_file.get_raw_bytes(p + "ffn_norm.weight")
        b_post_ffn_norm = st_file.get_raw_bytes(p + "post_ffn_norm.weight")
        b_w1 = st_file.get_raw_bytes(p + "feed_forward.w1.weight") if pruned_dir is not None else b""
        b_w3 = st_file.get_raw_bytes(p + "feed_forward.w3.weight") if pruned_dir is not None else b""
        b_w2 = st_file.get_raw_bytes(p + "feed_forward.w2.weight") if pruned_dir is not None else b""

        w1_f = st_file.get_f32(p + "feed_forward.w1.weight")
        w3_f = st_file.get_f32(p + "feed_forward.w3.weight")
        w2_f = st_file.get_f32(p + "feed_forward.w2.weight")

        if ltype == "conv":
            kind = 0
            b_in_proj = st_file.get_raw_bytes(p + "conv.in_proj.weight")
            b_conv_w = st_file.get_raw_bytes(p + "conv.conv.weight")
            b_out_proj = st_file.get_raw_bytes(p + "conv.out_proj.weight")

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
            op_header = struct.pack(
                "<8sIIIII36x", OPERATOR_MAGIC, 0, idx, len(b_op_norm), len(b_in_proj), len(b_conv_w)
            )
            op_blob = op_header + b_op_norm + b_in_proj + b_conv_w
            op_4bit = quantize_matrix_4bit_group64(st_file.get_f32(p + "conv.in_proj.weight")) + quantize_matrix_4bit_group64(
                st_file.get_f32(p + "conv.out_proj.weight")
            )
            assert len(op_4bit) == OP_CONV_IN_4BIT_BYTES + OP_OUT_4BIT_BYTES
        else:
            kind = 1 if ltype == "sliding_attention" else 2
            b_q = st_file.get_raw_bytes(p + "self_attn.q_proj.weight")
            b_k = st_file.get_raw_bytes(p + "self_attn.k_proj.weight")
            b_v = st_file.get_raw_bytes(p + "self_attn.v_proj.weight")
            b_o_proj = st_file.get_raw_bytes(p + "self_attn.o_proj.weight")
            b_gate = st_file.get_raw_bytes(p + "self_attn.gate_proj.weight")
            b_qn = st_file.get_raw_bytes(p + "self_attn.q_norm.weight")
            b_kn = st_file.get_raw_bytes(p + "self_attn.k_norm.weight")

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
            op_header = struct.pack(
                "<8sIIIIIIIII20x",
                OPERATOR_MAGIC,
                kind,
                idx,
                len(b_op_norm),
                len(b_q),
                len(b_k),
                len(b_v),
                len(b_gate),
                len(b_qn),
                len(b_kn),
            )
            op_blob = op_header + b_op_norm + b_q + b_k + b_v + b_gate + b_qn + b_kn
            qkv_cat = np.concatenate(
                [
                    st_file.get_f32(p + "self_attn.q_proj.weight"),
                    st_file.get_f32(p + "self_attn.k_proj.weight"),
                    st_file.get_f32(p + "self_attn.v_proj.weight"),
                ],
                axis=0,
            )
            op_4bit = quantize_matrix_4bit_group64(qkv_cat) + quantize_matrix_4bit_group64(
                st_file.get_f32(p + "self_attn.o_proj.weight")
            )
            assert len(op_4bit) == OP_ATTN_QKV_4BIT_BYTES + OP_OUT_4BIT_BYTES

        info = write_pruned_lfm3_trunk(trunk_dir, idx, kind, records_spec, keep_ffn_bf16=(pruned_dir is not None))
        (trunk_dir / "operator.bin").write_bytes(op_blob)
        (trunk_dir / "operator.4bit.bin").write_bytes(op_4bit)

        ffn_4bit = pack_ffn_4bit(w1_f, w3_f, w2_f)
        del w1_f, w3_f, w2_f
        (trunk_dir / "ffn.4bit.bin").write_bytes(ffn_4bit)

        exp_l = experts_dir / f"expert-{idx}"
        exp_l.mkdir(parents=True, exist_ok=True)
        (exp_l / "constants.bin").write_bytes(struct.pack("<IIIIII", idx, kind, 2048, 7680, 64, len(ffn_4bit)))
        (exp_l / "maps.json").write_text(
            json.dumps(
                {
                    "model": "LiquidAI/LFM3-Omni-2B",
                    "layer_id": idx,
                    "layer_kind": kind,
                    "expert_type": "dense_swiglu_4bit_group64",
                    "ffn_4bit_bytes": len(ffn_4bit),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _link_or_copy(trunk_dir / "ffn.4bit.bin", exp_l / "ffn.4bit.bin")

        if pruned_dir is not None and soc_q4_dir is not None:
            soc_t = soc_q4_dir / f"trunk-{idx}"
            write_pruned_lfm3_trunk(soc_t, idx, kind, records_spec, keep_ffn_bf16=False)
            for fn in ("operator.bin", "operator.4bit.bin", "ffn.4bit.bin"):
                _link_or_copy(trunk_dir / fn, soc_t / fn)

        total_op_4bit += len(op_4bit)
        total_ffn_4bit += len(ffn_4bit)
        layer_stats.append(info)
        if (idx + 1) % 6 == 0:
            print(f"  [Omni-Precompute] Built language trunk-{idx} & expert-{idx} ({idx + 1}/24)...", flush=True)

    return {
        "language_layers": 24,
        "global_embed_bf16_bytes": len(b_emb),
        "global_lm_head_4bit_bytes": len(lm_head_q4),
        "total_operator_4bit_bytes": total_op_4bit,
        "total_ffn_4bit_bytes": total_ffn_4bit,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }


def build_multimodal_trunks_and_projectors(st_file: SafetensorsMmapReader, cfg: dict[str, Any], mm_dir: Path) -> dict[str, Any]:
    """Precomputes Vision Tower (14 layers), Audio Encoder (17 layers), Algebraically Fused Projectors, and Depthformer."""
    t0 = time.perf_counter()
    mm_dir.mkdir(parents=True, exist_ok=True)

    vis_g = mm_dir / "vision-global-0"
    vis_g.mkdir(parents=True, exist_ok=True)
    ln_w = st_file.get_f32("model.multi_modal_projector.layer_norm.weight")
    ln_b = st_file.get_f32("model.multi_modal_projector.layer_norm.bias")
    w1 = st_file.get_f32("model.multi_modal_projector.linear_1.weight")
    b1 = st_file.get_f32("model.multi_modal_projector.linear_1.bias")
    w2 = st_file.get_f32("model.multi_modal_projector.linear_2.weight")
    b2 = st_file.get_f32("model.multi_modal_projector.linear_2.bias")

    gamma4 = np.tile(ln_w, 4)
    beta4 = np.tile(ln_b, 4)
    w1_fused = w1 * gamma4[None, :]
    b1_fused = (w1 @ beta4 + b1).astype("<f4")
    (vis_g / "projector_w1_fused.4bit.bin").write_bytes(quantize_matrix_4bit_group64(w1_fused, bf16_round_first=True))
    (vis_g / "projector_b1_fused.fp32.bin").write_bytes(b1_fused.tobytes())
    (vis_g / "projector_w2.4bit.bin").write_bytes(quantize_matrix_4bit_group64(w2, bf16_round_first=True))
    (vis_g / "projector_b2.fp32.bin").write_bytes(b2.astype("<f4").tobytes())

    vis_types = cfg["vision_config"]["layer_types"]
    vis_4bit_bytes = 0
    for idx, ltype in enumerate(vis_types):
        vt = mm_dir / f"vision-trunk-{idx}"
        vt.mkdir(parents=True, exist_ok=True)
        p = f"model.vision_tower.layers.{idx}."
        ffn_q4 = (
            quantize_matrix_4bit_group64(st_file.get_f32(p + "feed_forward.w1.weight"))
            + quantize_matrix_4bit_group64(st_file.get_f32(p + "feed_forward.w3.weight"))
            + quantize_matrix_4bit_group64(st_file.get_f32(p + "feed_forward.w2.weight"))
        )
        (vt / "ffn.4bit.bin").write_bytes(ffn_q4)
        if ltype == "conv":
            op_q4 = quantize_matrix_4bit_group64(st_file.get_f32(p + "conv.in_proj.weight")) + quantize_matrix_4bit_group64(
                st_file.get_f32(p + "conv.out_proj.weight")
            )
        else:
            qkv = np.concatenate(
                [
                    st_file.get_f32(p + "self_attn.q_proj.weight"),
                    st_file.get_f32(p + "self_attn.k_proj.weight"),
                    st_file.get_f32(p + "self_attn.v_proj.weight"),
                ],
                axis=0,
            )
            op_q4 = quantize_matrix_4bit_group64(qkv) + quantize_matrix_4bit_group64(
                st_file.get_f32(p + "self_attn.out_proj.weight")
            )
        (vt / "operator.4bit.bin").write_bytes(op_q4)
        vis_4bit_bytes += len(ffn_q4) + len(op_q4)

    aud_g = mm_dir / "audio-global-0"
    aud_g.mkdir(parents=True, exist_ok=True)
    aln_w = st_file.get_f32("model.audio_projector.layer_norm.weight")
    aln_b = st_file.get_f32("model.audio_projector.layer_norm.bias")
    aw1 = st_file.get_f32("model.audio_projector.linear_1.weight")
    ab1 = st_file.get_f32("model.audio_projector.linear_1.bias")
    aw2 = st_file.get_f32("model.audio_projector.linear_2.weight")
    ab2 = st_file.get_f32("model.audio_projector.linear_2.bias")

    aw1_fused = aw1 * aln_w[None, :]
    ab1_fused = (aw1 @ aln_b + ab1).astype("<f4")
    (aud_g / "projector_w1_fused.4bit.bin").write_bytes(quantize_matrix_4bit_group64(aw1_fused, bf16_round_first=True))
    (aud_g / "projector_b1_fused.fp32.bin").write_bytes(ab1_fused.tobytes())
    (aud_g / "projector_w2.4bit.bin").write_bytes(quantize_matrix_4bit_group64(aw2, bf16_round_first=True))
    (aud_g / "projector_b2.fp32.bin").write_bytes(ab2.astype("<f4").tobytes())

    aud_types = cfg["audio_encoder_config"]["text_config"]["layer_types"]
    aud_4bit_bytes = 0
    for idx, ltype in enumerate(aud_types):
        at = mm_dir / f"audio-trunk-{idx}"
        at.mkdir(parents=True, exist_ok=True)
        p = f"model.audio_encoder.lfm.layers.{idx}."
        ffn_q4 = (
            quantize_matrix_4bit_group64(st_file.get_f32(p + "feed_forward.w1.weight"))
            + quantize_matrix_4bit_group64(st_file.get_f32(p + "feed_forward.w3.weight"))
            + quantize_matrix_4bit_group64(st_file.get_f32(p + "feed_forward.w2.weight"))
        )
        (at / "ffn.4bit.bin").write_bytes(ffn_q4)
        if ltype == "conv":
            op_q4 = quantize_matrix_4bit_group64(st_file.get_f32(p + "conv.in_proj.weight")) + quantize_matrix_4bit_group64(
                st_file.get_f32(p + "conv.out_proj.weight")
            )
        else:
            qkv = np.concatenate(
                [
                    st_file.get_f32(p + "self_attn.q_proj.weight"),
                    st_file.get_f32(p + "self_attn.k_proj.weight"),
                    st_file.get_f32(p + "self_attn.v_proj.weight"),
                ],
                axis=0,
            )
            op_q4 = quantize_matrix_4bit_group64(qkv) + quantize_matrix_4bit_group64(
                st_file.get_f32(p + "self_attn.out_proj.weight")
            )
        (at / "operator.4bit.bin").write_bytes(op_q4)
        aud_4bit_bytes += len(ffn_q4) + len(op_q4)

    dep_g = mm_dir / "depth-global-0"
    dep_g.mkdir(parents=True, exist_ok=True)
    cb_q4 = quantize_matrix_4bit_group64(st_file.get_f32("model.codebook_embedding.weight"))
    (dep_g / "codebook_embedding.4bit.bin").write_bytes(cb_q4)

    print("  [Omni-Precompute] Built 14 vision trunks, 17 audio trunks, fused projectors & speech codebook!", flush=True)
    return {
        "vision_trunks": len(vis_types),
        "vision_4bit_bytes": vis_4bit_bytes,
        "audio_trunks": len(aud_types),
        "audio_4bit_bytes": aud_4bit_bytes,
        "codebook_4bit_bytes": len(cb_q4),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Path to LiquidAI/LFM3-Omni-2B directory")
    parser.add_argument("--out", type=Path, required=True, help="Output precomputed root directory")
    parser.add_argument("--soc-q4-only", action="store_true", help="Write compact 4-bit SoC-Q4 layout directly into --out")
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = json.loads((args.source / "config.json").read_text(encoding="utf-8"))
    if args.soc_q4_only:
        pruned_dir = None
        soc_q4_dir = args.out
        soc_q4_dir.mkdir(parents=True, exist_ok=True)
        mm_target = soc_q4_dir / "omni-multimodal"
    else:
        pruned_dir = args.out / "pruned"
        soc_q4_dir = args.out / "soc-q4"
        pruned_dir.mkdir(parents=True, exist_ok=True)
        soc_q4_dir.mkdir(parents=True, exist_ok=True)
        mm_target = pruned_dir / "omni-multimodal"

    tok_src = args.source / "tokenizer.json"
    if tok_src.exists():
        if pruned_dir is not None:
            shutil.copy2(tok_src, pruned_dir / "tokenizer.json")
        if soc_q4_dir is not None:
            shutil.copy2(tok_src, soc_q4_dir / "tokenizer.json")

    with SafetensorsMmapReader(args.source / "model.safetensors") as st_file:
        lang_summary = build_language_backbone_24_layers(st_file, cfg["text_config"], pruned_dir, soc_q4_dir)
        mm_summary = build_multimodal_trunks_and_projectors(st_file, cfg, mm_target)

    if not args.soc_q4_only and soc_q4_dir is not None:
        mm_link = soc_q4_dir / "omni-multimodal"
        if not mm_link.exists():
            try:
                os.symlink(mm_target, mm_link)
            except OSError:
                pass

    summary = {
        "source_model": "LiquidAI/LFM3-Omni-2B",
        "total_build_seconds": round(time.perf_counter() - t0, 2),
        "language_backbone": lang_summary,
        "multimodal_trunks": mm_summary,
    }
    (args.out / "omni_precompute_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
