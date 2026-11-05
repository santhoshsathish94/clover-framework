#!/usr/bin/env python3
"""
Trunk Pruner for Liquid AI LFM3-1.7B (`clover-ai-system/src/models/lfm3-1.7b/prune_lfm3_trunk.py`).

Applies Clover's `prune-trunk.py` (`Kimi K3`) and `prune_nemotron_trunk.py` (`Nemotron 3.5 Super VL`)
discipline to all 24 tri-hybrid layers of `LiquidAI/LFM3-1.7B`:
  - In the **13 Gated ShortConv layers (`layer_kind = 0`)**:
    Records `0` (`operator_norm.weight`), `5` (`conv.in_proj.weight`), and `6` (`conv.conv.weight`)
    are fused into `operator.bin` (`25,198,144` B). `prune_lfm3_trunk.py` drops those 3 records,
    deletes `attention.bin` (`25,194,496` B per layer) completely, and leaves only `common.bin`
    (`post_operator_norm`, `conv.out_proj`, `ffn_norm`, `post_ffn_norm` = `8,400,896` B) and
    `ffn.bin` (`w1`, `w3`, `w2` = `94,371,840` B).
  - In the **11 Sliding/Full GQA Attention layers (`layer_kind = 1` or `2`)**:
    Records `0` (`operator_norm`), `5` (`q_proj`), `6` (`k_proj`), `7` (`v_proj`), `8` (`gate_proj`),
    `9` (`q_norm`), and `13` (`k_norm`) are fused into `operator.bin` (`12,653,120` B).
    `prune_lfm3_trunk.py` drops those 7 records and deletes `attention.bin` (`12,648,960` B per layer)
    completely!
  - Across all 24 layers (`trunk-0..23`), `attention.bin` is eliminated from **24 / 24 trunks**,
    dropping `466,765,056` bytes (`445.14 MiB`) of duplicate trunk storage while keeping
    100% of `index.bin` record metadata intact (`length = 0` for pruned records).
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Any

from prepare_lfm3_datasets import GROUP_NAMES, HEADER_FMT, HEADER_SIZE, RECORD_FMT, RECORD_SIZE, TRUNK_MAGIC


def read_lfm3_trunk_index(index_path: Path):
    data = index_path.read_bytes()
    magic, version, layer_id, layer_kind, record_count, g0_off, g0_len, g1_off, g1_len, g2_off, g2_len = struct.unpack(
        HEADER_FMT, data[:HEADER_SIZE]
    )
    if magic != TRUNK_MAGIC or version != 1:
        raise ValueError(f"Unexpected trunk magic/version in {index_path}: {magic!r} v{version}")
    records = []
    pos = HEADER_SIZE
    for _ in range(record_count):
        rec = list(struct.unpack(RECORD_FMT, data[pos : pos + RECORD_SIZE]))
        records.append(rec)
        pos += RECORD_SIZE
    return layer_id, layer_kind, [g0_len, g1_len, g2_len], records


def prune_single_lfm3_trunk(src_dir: Path, dst_dir: Path) -> dict[str, Any]:
    layer_id, layer_kind, old_group_lens, records = read_lfm3_trunk_index(src_dir / "index.bin")
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy `operator.bin`
    op_src = src_dir / "operator.bin"
    op_bytes = 0
    if op_src.exists():
        shutil.copy2(op_src, dst_dir / "operator.bin")
        op_bytes = op_src.stat().st_size

    # Records kept in trunk after `operator.bin` fusion:
    # Group 0 (`common.bin`): Records 1 (`post_operator_norm`), 2 (`out_proj`/`o_proj`), 3 (`ffn_norm`), 4 (`post_ffn_norm`)
    # Group 2 (`ffn.bin`): Records 10 (`w1`), 11 (`w3`), 12 (`w2`)
    # Group 1 (`attention.bin`) and Record 0 (`operator_norm`) are 100% covered by `operator.bin`!
    keep_ids = {1, 2, 3, 4, 10, 11, 12}

    old_groups: dict[int, bytes] = {}
    for grp_id, grp_name in enumerate(GROUP_NAMES):
        p = src_dir / grp_name
        old_groups[grp_id] = p.read_bytes() if p.exists() else b""

    new_buffers = [bytearray(), bytearray(), bytearray()]
    new_records = []

    for rec_id, grp_id, dtype_code, rows, old_off, old_len, cols in records:
        if rec_id in keep_ids and old_len > 0:
            chunk = old_groups[grp_id][old_off : old_off + old_len]
            new_off = len(new_buffers[grp_id])
            new_buffers[grp_id].extend(chunk)
            new_records.append([rec_id, grp_id, dtype_code, rows, new_off, len(chunk), cols])
        else:
            new_records.append([rec_id, grp_id, dtype_code, rows, 0, 0, cols])

    new_group_lens = [len(b) for b in new_buffers]
    for grp_id, grp_name in enumerate(GROUP_NAMES):
        out_p = dst_dir / grp_name
        if new_group_lens[grp_id] > 0:
            out_p.write_bytes(new_buffers[grp_id])
        elif out_p.exists():
            out_p.unlink()

    header = struct.pack(
        HEADER_FMT,
        TRUNK_MAGIC,
        1,
        layer_id,
        layer_kind,
        len(new_records),
        0,
        new_group_lens[0],
        0,
        new_group_lens[1],
        0,
        new_group_lens[2],
    )
    idx_out = bytearray(header)
    for rec in new_records:
        idx_out.extend(struct.pack(RECORD_FMT, *rec))
    (dst_dir / "index.bin").write_bytes(idx_out)

    old_total = sum(old_group_lens)
    new_total = sum(new_group_lens)
    return {
        "trunk_name": src_dir.name,
        "layer_id": layer_id,
        "layer_kind": layer_kind,
        "old_group_bytes": old_total,
        "new_group_bytes": new_total,
        "dropped_group_bytes": old_total - new_total,
        "reduction_pct": round(100.0 * (old_total - new_total) / max(1, old_total), 2),
        "operator_bytes": op_bytes,
    }


def prune_all_24_lfm3_trunks(raw_computed_dir: Path, pruned_computed_dir: Path) -> dict[str, Any]:
    pruned_computed_dir.mkdir(parents=True, exist_ok=True)
    if (raw_computed_dir / "global-0").exists():
        dst_g = pruned_computed_dir / "global-0"
        if dst_g.exists():
            shutil.rmtree(dst_g)
        shutil.copytree(raw_computed_dir / "global-0", dst_g)

    results = []
    for idx in range(24):
        src_t = raw_computed_dir / f"trunk-{idx}"
        dst_t = pruned_computed_dir / f"trunk-{idx}"
        results.append(prune_single_lfm3_trunk(src_t, dst_t))

    total_old = sum(r["old_group_bytes"] for r in results)
    total_new = sum(r["new_group_bytes"] for r in results)
    total_dropped = total_old - total_new
    conv_old = sum(r["old_group_bytes"] for r in results if r["layer_kind"] == 0)
    conv_new = sum(r["new_group_bytes"] for r in results if r["layer_kind"] == 0)
    attn_old = sum(r["old_group_bytes"] for r in results if r["layer_kind"] in (1, 2))
    attn_new = sum(r["new_group_bytes"] for r in results if r["layer_kind"] in (1, 2))

    summary = {
        "num_trunks_pruned": len(results),
        "all_24_old_trunk_bytes": total_old,
        "all_24_pruned_trunk_bytes": total_new,
        "all_24_dropped_trunk_bytes": total_dropped,
        "all_24_dropped_trunk_mib": round(total_dropped / (1024**2), 2),
        "conv_13_layers_trunk_reduction_pct": round(100.0 * (conv_old - conv_new) / max(1, conv_old), 2),
        "attn_11_layers_trunk_reduction_pct": round(100.0 * (attn_old - attn_new) / max(1, attn_old), 2),
        "overall_24_trunk_reduction_pct": round(100.0 * total_dropped / max(1, total_old), 2),
        "sample_pruned_trunks": [results[0], results[2], results[4]],
    }
    (pruned_computed_dir / "prune_lfm3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
