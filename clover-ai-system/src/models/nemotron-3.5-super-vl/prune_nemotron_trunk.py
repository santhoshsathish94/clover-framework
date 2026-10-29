#!/usr/bin/env python3
"""
Prunes redundant records from NVIDIA Nemotron 3.5 Super VL prepared trunks (`trunk-L`)
that are already served by `operator.bin` (`clover-ai-system/src/models/nemotron-3.5-super-vl/prune_nemotron_trunk.py`).

Directly mirrors `clover-ai-system/src/tools/prune-trunk.py` for Kimi K3:
  - Reads `index.bin` (`NEMOTRK1`, 1168-byte header + 72-byte record entries).
  - Determines needed trunk records by `layer_kind`:
    * `kind == 0` (Mamba-2, e.g., `trunk-0`): `operator.bin` already holds `norm` (0), `in_proj` (3),
      `conv1d.weight` (4), `conv1d.bias` (5), `dt_bias` (6), precomputed `A` (7), and `D` (8).
      Only `mixer.norm` (1) and `mixer.out_proj` (2) are needed from `common.bin` (`attention.bin` is dropped completely!).
    * `kind == 1` (NoPE 16:1 GQA Attention, e.g., `trunk-7`, `trunk-mtp-0`): `operator.bin` already holds
      `norm` (0) and `q_proj` (2), `k_proj` (3), `v_proj` (4).
      Only `o_proj` (1) (plus MTP `enorm` (10), `hnorm` (11), `eh_proj` (12) if present) are kept (`attention.bin` is dropped completely!).
    * `kind == 2` (Latent `ReLU^2` MoE, e.g., `trunk-1`, `trunk-mtp-1`): All trunk records (`0, 1, 29, 30, 31, 32, 34, 36`)
      are needed (routed experts live in `expert-L/experts.direct`).
  - Rewrites `index.bin` with `length = 0` for dropped records, deletes empty group files (`attention.bin`),
    and hard-links `operator.bin` into the pruned output directory.
"""

from __future__ import annotations

import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

TRUNK_MAGIC = b"NEMOTRK1"
GROUPS = ("common.bin", "attention.bin", "routed-shared.bin")
HEADER, RECORD = 1168, 72

MAMBA_NEEDS = {1, 2}
ATTENTION_NEEDS = {1, 10, 11, 12}
MOE_NEEDS = {0, 1, 29, 30, 31, 32, 34, 36}

NEEDS_BY_KIND = {
    0: MAMBA_NEEDS,
    1: ATTENTION_NEEDS,
    2: MOE_NEEDS,
}


def cursor_old(mine: list[list[int]], record: list[int]) -> int:
    pos = 0
    for r in mine:
        if r is record:
            return pos
        pos += r[4]
    return pos


def prune_nemotron_trunk(src_dir: Path, dst_dir: Path) -> dict[str, Any]:
    index_path = src_dir / "index.bin"
    index = bytearray(index_path.read_bytes())
    if index[:8] != TRUNK_MAGIC:
        raise ValueError(f"{src_dir}: not a NEMOTRK1 trunk index")

    version, layer_id, count = struct.unpack_from("<III", index, 8)
    if len(index) != HEADER + count * RECORD:
        raise ValueError(f"{src_dir}: index size {len(index)} != {HEADER + count * RECORD}")

    lengths_old = list(struct.unpack_from("<3Q", index, 24))
    layer_kind = struct.unpack_from("<I", index, 48)[0]
    needs = NEEDS_BY_KIND.get(layer_kind)
    if needs is None:
        raise ValueError(f"{src_dir}: unknown layer_kind {layer_kind}")

    records = []
    for position in range(count):
        base = HEADER + position * RECORD
        rid, group, dtype_code, rows, cols, n = struct.unpack_from("<6I", index, base)
        offset, length = struct.unpack_from("<2Q", index, base + 24)
        records.append([base, rid, group, offset, length])

    dst_dir.mkdir(parents=True, exist_ok=True)
    lengths_new = []
    dropped_bytes = 0

    for group_idx, g_name in enumerate(GROUPS):
        mine = [r for r in records if r[2] == group_idx]
        cursor = 0
        src_group_path = src_dir / g_name
        dst_group_path = dst_dir / g_name

        if not mine or lengths_old[group_idx] == 0:
            if dst_group_path.exists():
                dst_group_path.unlink()
            lengths_new.append(0)
            continue

        if any(r[1] not in needs for r in mine):
            tmp_path = dst_dir / (g_name + ".tmp")
            with open(src_group_path, "rb") as f_in, open(tmp_path, "wb") as f_out:
                if os.fstat(f_in.fileno()).st_size != lengths_old[group_idx]:
                    raise ValueError(f"{src_dir}/{g_name}: size differs from index")
                for r in mine:
                    base, rid, _, offset, length = r
                    if offset != cursor_old(mine, r):
                        raise ValueError(f"{src_dir}/{g_name}: records are not contiguous")
                    keep = rid in needs
                    if keep:
                        f_in.seek(offset)
                        data = f_in.read(length)
                        if len(data) != length:
                            raise ValueError(f"{src_dir}/{g_name}: short read")
                        f_out.write(data)
                    else:
                        dropped_bytes += length
                    new_len = length if keep else 0
                    struct.pack_into("<2Q", index, base + 24, cursor, new_len)
                    cursor += new_len
            os.replace(tmp_path, dst_group_path)
            if cursor == 0:
                dst_group_path.unlink()
        else:
            for r in mine:
                cursor += r[4]
            if dst_group_path.exists():
                dst_group_path.unlink()
            try:
                os.link(src_group_path, dst_group_path)
            except OSError:
                shutil.copy2(src_group_path, dst_group_path)

        lengths_new.append(cursor)

    struct.pack_into("<3Q", index, 24, *lengths_new)
    op_src = src_dir / "operator.bin"
    op_dst = dst_dir / "operator.bin"
    op_bytes = 0
    if op_src.exists():
        if op_dst.exists():
            op_dst.unlink()
        try:
            os.link(op_src, op_dst)
        except OSError:
            shutil.copy2(op_src, op_dst)
        op_bytes = op_dst.stat().st_size

    (dst_dir / "index.bin").write_bytes(index)
    return {
        "trunk_name": src_dir.name,
        "layer_id": layer_id,
        "layer_kind": layer_kind,
        "old_group_bytes": sum(lengths_old),
        "new_group_bytes": sum(lengths_new),
        "dropped_group_bytes": dropped_bytes,
        "reduction_pct": round(100.0 * dropped_bytes / max(1, sum(lengths_old)), 2),
        "operator_bytes": op_bytes,
    }


def prune_all_nemotron_trunks(raw_root: Path, pruned_root: Path) -> dict[str, Any]:
    pruned_root.mkdir(parents=True, exist_ok=True)
    results = []
    total_old = 0
    total_new = 0
    total_dropped = 0

    for item in sorted(raw_root.iterdir()):
        if item.is_dir() and item.name.startswith("trunk-"):
            res = prune_nemotron_trunk(item, pruned_root / item.name)
            results.append(res)
            total_old += res["old_group_bytes"]
            total_new += res["new_group_bytes"]
            total_dropped += res["dropped_group_bytes"]
        elif item.is_dir() and (item.name.startswith("expert-") or item.name.startswith("global-")):
            dst_sub = pruned_root / item.name
            dst_sub.mkdir(parents=True, exist_ok=True)
            for f in item.iterdir():
                target = dst_sub / f.name
                if target.exists():
                    target.unlink()
                try:
                    os.link(f, target)
                except OSError:
                    shutil.copy2(f, target)

    # Projection across all 88 layers of the full 120B model:
    # 40 Mamba-2 trunks + 8 GQA Attention trunks + 40 Latent MoE trunks
    mamba_r = next((r for r in results if r["layer_kind"] == 0), None)
    attn_r = next((r for r in results if r["layer_kind"] == 1 and r["layer_id"] < 100), None)
    moe_r = next((r for r in results if r["layer_kind"] == 2 and r["layer_id"] < 100), None)

    full_88_projection = {}
    if mamba_r and attn_r and moe_r:
        old_88 = 40 * mamba_r["old_group_bytes"] + 8 * attn_r["old_group_bytes"] + 40 * moe_r["old_group_bytes"]
        new_88 = 40 * mamba_r["new_group_bytes"] + 8 * attn_r["new_group_bytes"] + 40 * moe_r["new_group_bytes"]
        dropped_88 = old_88 - new_88
        full_88_projection = {
            "full_88_layers_old_trunk_gib": round(old_88 / (1024**3), 3),
            "full_88_layers_pruned_trunk_gib": round(new_88 / (1024**3), 3),
            "full_88_layers_dropped_gib": round(dropped_88 / (1024**3), 3),
            "mamba_40_layers_reduction_pct": mamba_r["reduction_pct"],
            "attention_8_layers_reduction_pct": attn_r["reduction_pct"],
            "overall_88_trunk_reduction_pct": round(100.0 * dropped_88 / max(1, old_88), 2),
        }

    return {
        "trunks_pruned": results,
        "sampled_total_old_bytes": total_old,
        "sampled_total_new_bytes": total_new,
        "sampled_total_dropped_bytes": total_dropped,
        "full_88_layer_projection": full_88_projection,
    }


if __name__ == "__main__":
    import json

    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/opt/clover-k3/nemotron-computed/raw")
    pruned_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/opt/clover-k3/nemotron-computed/pruned")
    out = prune_all_nemotron_trunks(raw_dir, pruned_dir)
    print(json.dumps(out, indent=2))
