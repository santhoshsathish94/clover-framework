#!/usr/bin/env python3
"""
stream_precompute_nemotron_all_layers.py
========================================
Stream-downloads NVIDIA Nemotron 3.5 Super VL (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`)
safetensors shards for a specified layer range `[--start-layer, --end-layer]`, builds:
  - `nemotron/trunks/trunk-L/` (`index.bin`, `common.bin`, `operator.bin`, `routed-shared.bin`, with `attention.bin` pruned)
  - `nemotron/experts/expert-L/` (`constants.bin`, `maps.json`, `experts.direct`, and `experts.4bit.direct`)
uploads each batch to `Santhoshsathish94/clover-ai-studio`, and immediately evicts shards no longer
needed by subsequent layers so disk/`/dev/shm` usage stays bounded (< 16 GB at any instant).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from safetensors import safe_open

TRUNK_MAGIC = b"NEMOTRK1"
OPERATOR_MAGIC = b"NEMO_OP1"
HEADER_BYTES = 1168
RECORD_BYTES = 72
GROUPS = ("common.bin", "attention.bin", "routed-shared.bin")

DTYPE_BF16 = 1
DTYPE_FP32 = 2

MAMBA_NEEDS = {1, 2}
ATTENTION_NEEDS = {1, 10, 11, 12}
MOE_NEEDS = {0, 1, 29, 30, 31, 32, 34, 36}
NEEDS_BY_KIND = {0: MAMBA_NEEDS, 1: ATTENTION_NEEDS, 2: MOE_NEEDS}

UP_4BIT_BYTES = (2688 * 16 * 32) + (2688 * 16) + (2688 * 4)
DOWN_4BIT_BYTES = (1024 * 42 * 32) + (1024 * 42) + (1024 * 4)
EXPERT_4BIT_STRIDE = UP_4BIT_BYTES + DOWN_4BIT_BYTES  # 2,853,376 B
EXPERT_BF16_STRIDE = 11010048


def tensor_to_le_bytes(t: torch.Tensor) -> bytes:
    t_contig = t.detach().cpu().contiguous()
    if t_contig.dtype == torch.bfloat16:
        return t_contig.view(torch.uint16).numpy().tobytes()
    return t_contig.numpy().tobytes()


def quantize_batch_4bit_group64(w_bf16: torch.Tensor, device: torch.device) -> tuple[bytes, bytes, bytes]:
    """
    Vectorized batch quantization of `[E, rows, cols]` (`cols % 64 == 0`) into Clover's planar 4-bit group-64 format
    on GPU (`cuda`) or CPU (`cpu`).
    """
    w_f = w_bf16.to(device=device, dtype=torch.float32)
    E, rows, cols = w_f.shape
    groups = cols // 64

    row_max = w_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    row_scales = (row_max / 127.0).squeeze(-1)  # [E, rows] FP32
    codes_i8 = torch.clamp(torch.round(w_f / row_scales.unsqueeze(-1)), -127, 127).to(torch.int16)

    codes_g = codes_i8.view(E, rows, groups, 64)
    g_max = codes_g.abs().amax(dim=-1)  # [E, rows, groups]
    steps = torch.where(g_max > 0, (g_max + 6) // 7, torch.ones_like(g_max)).to(torch.uint8)

    step_f = steps.to(torch.float32).unsqueeze(-1)
    q_vals = torch.clamp(torch.round(codes_g.to(torch.float32) / step_f), -8, 7).to(torch.int16) + 8
    low = q_vals[:, :, :, :32].to(torch.uint8)
    high = q_vals[:, :, :, 32:].to(torch.uint8)
    packed = (low | (high << 4)).contiguous()  # [E, rows, groups, 32]

    return (
        packed.cpu().numpy().tobytes(),
        steps.contiguous().cpu().numpy().tobytes(),
        row_scales.contiguous().cpu().numpy().tobytes(),
    )


def write_pruned_trunk_folder(
    out_dir: Path,
    layer_id: int,
    layer_kind: int,
    records_spec: list[tuple[int, int, int, int, int, bytes]],
    operator_bytes: bytes | None = None,
) -> None:
    """Writes pruned `index.bin`, `common.bin`, `routed-shared.bin`, and `operator.bin` directly."""
    out_dir.mkdir(parents=True, exist_ok=True)
    needs = NEEDS_BY_KIND[layer_kind]
    group_buffers = [bytearray(), bytearray(), bytearray()]
    packed_records = []

    for rid, group_id, dtype_code, rows, cols, raw_bytes in records_spec:
        numel = rows * cols
        crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
        if rid in needs:
            offset = len(group_buffers[group_id])
            length = len(raw_bytes)
            group_buffers[group_id].extend(raw_bytes)
        else:
            offset = len(group_buffers[group_id])
            length = 0
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
    if operator_bytes is not None:
        (out_dir / "operator.bin").write_bytes(operator_bytes)


def build_mamba_trunk(get_tensor, prefix: str, layer_id: int, out_dir: Path) -> None:
    norm_w = get_tensor(prefix + "norm.weight")
    m_norm_w = get_tensor(prefix + "mixer.norm.weight")
    out_proj_w = get_tensor(prefix + "mixer.out_proj.weight")
    in_proj_w = get_tensor(prefix + "mixer.in_proj.weight")
    conv_w = get_tensor(prefix + "mixer.conv1d.weight").squeeze(1)
    conv_b = get_tensor(prefix + "mixer.conv1d.bias")
    dt_bias = get_tensor(prefix + "mixer.dt_bias")
    A_log = get_tensor(prefix + "mixer.A_log")
    D_vec = get_tensor(prefix + "mixer.D")
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

    op_header = bytearray(64)
    struct.pack_into(
        "<8sIIIIIII",
        op_header,
        0,
        OPERATOR_MAGIC,
        0,
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
    write_pruned_trunk_folder(out_dir, layer_id, 0, records_spec, operator_bytes)


def build_attn_trunk(get_tensor, prefix: str, layer_id: int, out_dir: Path) -> None:
    norm_w = get_tensor(prefix + "norm.weight")
    o_proj_w = get_tensor(prefix + "mixer.o_proj.weight")
    q_proj_w = get_tensor(prefix + "mixer.q_proj.weight")
    k_proj_w = get_tensor(prefix + "mixer.k_proj.weight")
    v_proj_w = get_tensor(prefix + "mixer.v_proj.weight")

    b_norm = tensor_to_le_bytes(norm_w)
    b_o = tensor_to_le_bytes(o_proj_w)
    b_q = tensor_to_le_bytes(q_proj_w)
    b_k = tensor_to_le_bytes(k_proj_w)
    b_v = tensor_to_le_bytes(v_proj_w)

    op_header = bytearray(64)
    struct.pack_into("<8sIIIIII", op_header, 0, OPERATOR_MAGIC, 1, layer_id, len(b_norm), len(b_q), len(b_k), len(b_v))
    operator_bytes = bytes(op_header) + b_norm + b_q + b_k + b_v

    records_spec = [
        (0, 0, DTYPE_BF16, 4096, 1, b_norm),
        (1, 0, DTYPE_BF16, 4096, 4096, b_o),
        (2, 1, DTYPE_BF16, 4096, 4096, b_q),
        (3, 1, DTYPE_BF16, 256, 4096, b_k),
        (4, 1, DTYPE_BF16, 256, 4096, b_v),
    ]
    write_pruned_trunk_folder(out_dir, layer_id, 1, records_spec, operator_bytes)


def build_moe_trunk_and_expert(
    get_tensor,
    prefix: str,
    layer_id: int,
    trunk_dir: Path,
    expert_dir: Path,
    device: torch.device,
    include_bf16_direct: bool = False,
) -> None:
    norm_w = get_tensor(prefix + "norm.weight")
    gate_w_fp32 = get_tensor(prefix + "mixer.gate.weight").to(torch.float32)
    gate_bias_fp32 = get_tensor(prefix + "mixer.gate.e_score_correction_bias").to(torch.float32)
    fc1_w = get_tensor(prefix + "mixer.fc1_latent_proj.weight")
    fc2_w = get_tensor(prefix + "mixer.fc2_latent_proj.weight")
    sh_up_w = get_tensor(prefix + "mixer.shared_experts.up_proj.weight")
    sh_down_w = get_tensor(prefix + "mixer.shared_experts.down_proj.weight")

    records_spec = [
        (0, 0, DTYPE_BF16, 4096, 1, tensor_to_le_bytes(norm_w)),
        (29, 2, DTYPE_FP32, 512, 4096, tensor_to_le_bytes(gate_w_fp32)),
        (30, 2, DTYPE_FP32, 512, 1, tensor_to_le_bytes(gate_bias_fp32)),
        (31, 2, DTYPE_BF16, 1024, 4096, tensor_to_le_bytes(fc1_w)),
        (32, 2, DTYPE_BF16, 4096, 1024, tensor_to_le_bytes(fc2_w)),
        (34, 2, DTYPE_BF16, 5376, 4096, tensor_to_le_bytes(sh_up_w)),
        (36, 2, DTYPE_BF16, 4096, 5376, tensor_to_le_bytes(sh_down_w)),
    ]
    write_pruned_trunk_folder(trunk_dir, layer_id, 2, records_spec, None)

    expert_dir.mkdir(parents=True, exist_ok=True)
    q4_path = expert_dir / "experts.4bit.direct"
    direct_path = expert_dir / "experts.direct"
    f_bf16 = open(direct_path, "wb") if include_bf16_direct else None

    expert_crcs = []
    t_e0 = time.perf_counter()
    BATCH_E = 64
    with open(q4_path, "wb") as f_q4:
        for e0 in range(0, 512, BATCH_E):
            ups = []
            downs = []
            for e in range(e0, e0 + BATCH_E):
                ut = get_tensor(f"{prefix}mixer.experts.{e}.up_proj.weight")
                dt = get_tensor(f"{prefix}mixer.experts.{e}.down_proj.weight")
                ups.append(ut)
                downs.append(dt)
                if f_bf16 is not None:
                    bu = tensor_to_le_bytes(ut)
                    bd = tensor_to_le_bytes(dt)
                    f_bf16.write(bu)
                    f_bf16.write(bd)
                if e < 4 or e == 511:
                    bu = tensor_to_le_bytes(ut)
                    bd = tensor_to_le_bytes(dt)
                    expert_crcs.append({
                        "expert_id": e,
                        "up_crc32": zlib.crc32(bu) & 0xFFFFFFFF,
                        "down_crc32": zlib.crc32(bd) & 0xFFFFFFFF,
                    })

            up_batch = torch.stack(ups, dim=0)    # [64, 2688, 1024]
            down_batch = torch.stack(downs, dim=0)  # [64, 1024, 2688]
            u_nib, u_stp, u_scl = quantize_batch_4bit_group64(up_batch, device)
            d_nib, d_stp, d_scl = quantize_batch_4bit_group64(down_batch, device)

            # Write per-expert planar 4-bit record (`EXPERT_4BIT_STRIDE = 2,853,376 B`)
            u_n_sz = 2688 * 16 * 32
            u_st_sz = 2688 * 16
            u_sc_sz = 2688 * 4
            d_n_sz = 1024 * 42 * 32
            d_st_sz = 1024 * 42
            d_sc_sz = 1024 * 4
            for bi in range(BATCH_E):
                f_q4.write(u_nib[bi * u_n_sz : (bi + 1) * u_n_sz])
                f_q4.write(u_stp[bi * u_st_sz : (bi + 1) * u_st_sz])
                f_q4.write(u_scl[bi * u_sc_sz : (bi + 1) * u_sc_sz])
                f_q4.write(d_nib[bi * d_n_sz : (bi + 1) * d_n_sz])
                f_q4.write(d_stp[bi * d_st_sz : (bi + 1) * d_st_sz])
                f_q4.write(d_scl[bi * d_sc_sz : (bi + 1) * d_sc_sz])

    if f_bf16 is not None:
        f_bf16.close()

    offsets_buf = bytearray(512 * 8)
    for e in range(512):
        struct.pack_into("<Q", offsets_buf, e * 8, e * EXPERT_BF16_STRIDE)
    (expert_dir / "constants.bin").write_bytes(offsets_buf)

    maps_meta = {
        "format": "NEMO_EXPERTS_DIRECT_V1",
        "layer_id": layer_id,
        "num_experts": 512,
        "up_proj_shape": [2688, 1024],
        "down_proj_shape": [1024, 2688],
        "activation": "relu2",
        "has_gate": False,
        "expert_stride_bytes": EXPERT_BF16_STRIDE,
        "expert_4bit_stride_bytes": EXPERT_4BIT_STRIDE,
        "total_4bit_bytes": q4_path.stat().st_size,
        "build_seconds": round(time.perf_counter() - t_e0, 3),
        "sample_crc32": expert_crcs,
    }
    (expert_dir / "maps.json").write_text(json.dumps(maps_meta, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--end-layer", type=int, required=True)
    parser.add_argument("--work-dir", type=str, default="/dev/shm/nemo-stream")
    parser.add_argument("--local-shard-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-bf16-direct", action="store_true")
    parser.add_argument("--batch-upload-layers", type=int, default=4, help="Upload every N layers in a single HF commit")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    shard_cache = work_dir / "shards"
    out_stage = work_dir / "out"
    shard_cache.mkdir(parents=True, exist_ok=True)
    out_stage.mkdir(parents=True, exist_ok=True)
    local_shard_dir = Path(args.local_shard_dir) if args.local_shard_dir else None

    src_repo = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
    dst_repo = "Santhoshsathish94/clover-ai-studio"
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    device = torch.device(args.device)

    # Check already uploaded trunks/experts on HF
    existing_files = set(api.list_repo_files(dst_repo, repo_type="dataset"))

    # Download index.json if needed
    if local_shard_dir and (local_shard_dir / "model.safetensors.index.json").exists():
        idx_path = local_shard_dir / "model.safetensors.index.json"
    else:
        idx_path = Path(hf_hub_download(repo_id=src_repo, filename="model.safetensors.index.json", local_dir=str(shard_cache)))
    wm = json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]

    layer_shards: dict[int, set[str]] = {}
    layer_types: dict[int, str] = {}
    for k, s in wm.items():
        if k.startswith("language_model.backbone.layers."):
            l = int(k.split(".")[3])
            layer_shards.setdefault(l, set()).add(s)
            if "experts" in k:
                layer_types[l] = "moe"
            elif "q_proj" in k:
                layer_types[l] = "attn"
            elif "conv1d" in k and l not in layer_types:
                layer_types[l] = "mamba"

    target_layers = [l for l in range(args.start_layer, args.end_layer + 1) if l in layer_shards]
    handles: dict[str, Any] = {}
    downloaded_shards: dict[str, Path] = {}

    def ensure_shard(sh_name: str) -> Path:
        if sh_name in downloaded_shards and downloaded_shards[sh_name].exists():
            return downloaded_shards[sh_name]
        if local_shard_dir and (local_shard_dir / sh_name).exists():
            downloaded_shards[sh_name] = local_shard_dir / sh_name
            return downloaded_shards[sh_name]
        print(f"  [DL] Downloading {sh_name} ...", flush=True)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        p = Path(
            hf_hub_download(
                repo_id=src_repo,
                filename=sh_name,
                local_dir=str(shard_cache),
            )
        )
        downloaded_shards[sh_name] = p
        return p

    def get_tensor(key: str) -> torch.Tensor:
        sh = wm[key]
        if sh not in handles:
            sp = ensure_shard(sh)
            handles[sh] = safe_open(str(sp), framework="pt", device="cpu")
        return handles[sh].get_tensor(key)

    pending_ops: list[CommitOperationAdd] = []
    pending_dirs: list[Path] = []
    pending_lids: list[int] = []

    def flush_pending_uploads():
        nonlocal pending_ops, pending_dirs, pending_lids
        if not pending_ops:
            return
        t_u0 = time.perf_counter()
        l_str = ", ".join(f"{x:02d}" for x in pending_lids)
        for attempt in range(5):
            try:
                info = api.create_commit(
                    repo_id=dst_repo,
                    repo_type="dataset",
                    operations=pending_ops,
                    commit_message=f"Precompute Nemotron layers [{l_str}] into nemotron/trunks and nemotron/experts",
                )
                print(
                    f"[UPLOADED BATCH [{l_str}]] ({len(pending_ops)} files) in {time.perf_counter()-t_u0:.1f}s -> {info.commit_url}",
                    flush=True,
                )
                break
            except Exception as e:
                print(f"  [RETRY {attempt+1}/5] Batch [{l_str}]: {e}", flush=True)
                time.sleep(3 * (attempt + 1))
        for d in pending_dirs:
            shutil.rmtree(d, ignore_errors=True)
        pending_ops = []
        pending_dirs = []
        pending_lids = []

    for idx_pos, lid in enumerate(target_layers):
        ltype = layer_types[lid]
        trunk_done = f"nemotron/trunks/trunk-{lid}/index.bin" in existing_files
        expert_done = (ltype != "moe") or (f"nemotron/experts/expert-{lid}/experts.4bit.direct" in existing_files)
        if trunk_done and expert_done:
            print(f"[SKIP] Layer {lid:02d} ({ltype}) already on HF.", flush=True)
            continue

        t_l0 = time.perf_counter()
        print(f"[BUILD] Layer {lid:02d} ({ltype}) on {device} (shards={sorted(layer_shards[lid])})...", flush=True)
        t_dir = out_stage / "trunks" / f"trunk-{lid}"
        e_dir = out_stage / "experts" / f"expert-{lid}"
        pfx = f"language_model.backbone.layers.{lid}."

        if ltype == "mamba":
            build_mamba_trunk(get_tensor, pfx, lid, t_dir)
        elif ltype == "attn":
            build_attn_trunk(get_tensor, pfx, lid, t_dir)
        elif ltype == "moe":
            build_moe_trunk_and_expert(
                get_tensor, pfx, lid, t_dir, e_dir, device=device, include_bf16_direct=args.include_bf16_direct
            )
        print(f"  [BUILT] Layer {lid:02d} ({ltype}) in {time.perf_counter()-t_l0:.1f}s", flush=True)

        for fp in sorted(t_dir.rglob("*")):
            if fp.is_file():
                pending_ops.append(CommitOperationAdd(path_in_repo=f"nemotron/trunks/trunk-{lid}/{fp.name}", path_or_fileobj=str(fp)))
        pending_dirs.append(t_dir)
        if ltype == "moe" and e_dir.exists():
            for fp in sorted(e_dir.rglob("*")):
                if fp.is_file():
                    pending_ops.append(CommitOperationAdd(path_in_repo=f"nemotron/experts/expert-{lid}/{fp.name}", path_or_fileobj=str(fp)))
            pending_dirs.append(e_dir)
        pending_lids.append(lid)

        # Evict shards not needed by any remaining layer in `target_layers[idx_pos+1:]`
        future_shards = set()
        for fl in target_layers[idx_pos + 1 :]:
            future_shards.update(layer_shards[fl])
        for sh_name in list(handles.keys()):
            if sh_name not in future_shards:
                del handles[sh_name]
                sp = downloaded_shards.pop(sh_name, None)
                if sp is not None and sp.exists() and (local_shard_dir is None or sp.parent != local_shard_dir):
                    sp.unlink(missing_ok=True)
                    print(f"  [EVICT] Deleted temporary shard {sh_name}", flush=True)
        gc.collect()

        if len(pending_lids) >= args.batch_upload_layers:
            flush_pending_uploads()

    flush_pending_uploads()
    print(f"[COMPLETE] Finished layers {args.start_layer}..{args.end_layer}!", flush=True)


if __name__ == "__main__":
    main()
