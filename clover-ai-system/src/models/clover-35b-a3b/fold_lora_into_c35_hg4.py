#!/usr/bin/env python3
"""Zero-Overhead LoRA Folder for Clover `HG4-16R-OP` (`fold_lora_into_c35_hg4.py`).

Folds a trained PEFT LoRA adapter (`adapter_model.safetensors`, `adapter_config.json`)
directly into a converted `Qwen3.6-35B-A3B` / `Clover-35B-A3B` `HG4-16R-OP` directory:
  - Hardlinks `meta.bin`, `embed_tokens.hg4`, `lm_head.hg4`, `tokenizer.json`, and all 40
    `experts_XX.hg4` files (`0` extra disk bytes for routed experts).
  - Unpacks every `HG8-16R` & `FP32` trunk / shared-expert weight inside `layer_00..39.hg4`
    (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `q_proj`, `k_proj`,
    `v_proj`, `o_proj`, and `shared_expert.{gate_proj,up_proj,down_proj}`), adds the exact
    `+(lora_alpha / r) * (B @ A)` update, re-packs `layer_00..39.hg4`, and refreshes
    `precompute_hg4_operators.py`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np
from safetensors.torch import load_file
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from convert_clover35b_full_hg4 import (  # noqa: E402
    FULL_ATTN_SET,
    interleave_swiglu_16r,
    pack_hg8_16r_blob,
    swiglu_eq_pre_diag,
)


def fwht128(x: torch.Tensor) -> torch.Tensor:
    orig = x.shape
    v = x.reshape(-1, 128).clone()
    h = 1
    while h < 128:
        v = v.view(-1, 128 // (2 * h), 2, h)
        a = v[:, :, 0, :].clone()
        b = v[:, :, 1, :].clone()
        v[:, :, 0, :] = a + b
        v[:, :, 1, :] = a - b
        h *= 2
    return v.view(orig) * (1.0 / math.sqrt(128.0))


def unpack_hg8_16r_blob(blob: bytes) -> torch.Tensor:
    r_rows, k_tagged = struct.unpack_from("<II", blob, 0)
    assert (k_tagged >> 28) == 8, f"Expected HG8 tag 8, got {k_tagged >> 28}"
    k_cols = k_tagged & 0x0FFFFFFF
    nb = k_cols // 128
    off = 8
    pre_diag = torch.from_numpy(np.frombuffer(blob, dtype=np.float32, count=k_cols, offset=off).copy())
    off += k_cols * 4
    n_sc = (r_rows // 16) * nb * 16
    sc_16r = torch.from_numpy(np.frombuffer(blob, dtype=np.float16, count=n_sc, offset=off).copy().astype(np.float32))
    off += n_sc * 2
    sc_rn = sc_16r.view(r_rows // 16, nb, 16).permute(0, 2, 1).contiguous().view(r_rows, nb, 1)
    n_pk = r_rows * k_cols
    pk_16r = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8, count=n_pk, offset=off).copy())
    q_u8 = pk_16r.view(r_rows // 16, nb, 32, 16, 4).permute(0, 3, 1, 2, 4).contiguous().view(r_rows, nb, 128)
    q_s8 = q_u8.to(torch.float32) - 128.0
    w_f32 = fwht128(q_s8 * sc_rn).view(r_rows, k_cols) * pre_diag.unsqueeze(0)
    return w_f32.contiguous()


def deinterleave_swiglu_16r(w13: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    two_rh, k_cols = w13.shape
    rh = two_rh // 2
    v = w13.view(rh // 8, 2, 8, k_cols)
    return v[:, 0, :, :].reshape(rh, k_cols).contiguous(), v[:, 1, :, :].reshape(rh, k_cols).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-hg4-dir", required=True)
    ap.add_argument("--lora-dir", required=True)
    ap.add_argument("--out-hg4-dir", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="Optional multiplier on (lora_alpha / r)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    base_dir = Path(args.base_hg4_dir)
    lora_dir = Path(args.lora_dir)
    out_dir = Path(args.out_hg4_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acfg = json.loads((lora_dir / "adapter_config.json").read_text(encoding="utf-8"))
    lora_scale = (float(acfg["lora_alpha"]) / float(acfg["r"])) * float(args.scale)
    sd = load_file(str(lora_dir / "adapter_model.safetensors"), device="cpu")
    print(f"[LoRA Fold] Loaded {len(sd)} tensors from {lora_dir} (alpha/r * scale = {lora_scale:.4f})", flush=True)

    all_lora_a_keys = {k for k in sd.keys() if k.endswith(".lora_A.weight")}
    used_lora_a_keys: set[str] = set()

    def get_delta(layer_idx: int, rel_mod: str) -> torch.Tensor | None:
        for pfx in (
            f"base_model.model.model.language_model.layers.{layer_idx}.{rel_mod}",
            f"base_model.model.model.layers.{layer_idx}.{rel_mod}",
        ):
            ka, kb = f"{pfx}.lora_A.weight", f"{pfx}.lora_B.weight"
            if ka in sd and kb in sd:
                used_lora_a_keys.add(ka)
                a = sd[ka].float()
                b = sd[kb].float()
                return (b @ a) * lora_scale
        return None

    # Link or copy all non-layer files (embed_tokens, lm_head, meta, tokenizer, experts_00..39)
    for item in sorted(base_dir.iterdir()):
        if item.name.startswith("layer_") and item.name.endswith(".hg4"):
            continue
        if item.name.endswith(".op.bin") or item.name == "hg4_op_manifest.json":
            continue
        dst = out_dir / item.name
        if dst.exists():
            dst.unlink()
        if item.is_file():
            try:
                os.link(item, dst)
            except OSError:
                shutil.copy2(item, dst)

    folded_count = 0
    for lid in range(40):
        raw = (base_dir / f"layer_{lid:02d}.hg4").read_bytes()
        kind = struct.unpack_from("<I", raw, 0)[0]
        is_gqa = (lid in FULL_ATTN_SET)
        assert kind == (1 if is_gqa else 0)
        off = 4

        def take_bytes(n: int) -> bytes:
            nonlocal off
            b = raw[off : off + n]
            off += n
            return b

        in_ln_b = take_bytes(2048 * 4)
        post_ln_b = take_bytes(2048 * 4)

        if not is_gqa:
            conv_w_b = take_bytes(8192 * 4 * 4)
            conv_rw_b = take_bytes(8192 * 3 * 4)
            a_w = torch.from_numpy(np.frombuffer(take_bytes(32 * 2048 * 4), dtype=np.float32).copy()).view(32, 2048)
            b_w = torch.from_numpy(np.frombuffer(take_bytes(32 * 2048 * 4), dtype=np.float32).copy()).view(32, 2048)
            da = get_delta(lid, "linear_attn.in_proj_a")
            db = get_delta(lid, "linear_attn.in_proj_b")
            if da is not None:
                a_w = (a_w + da).contiguous()
                folded_count += 1
            if db is not None:
                b_w = (b_w + db).contiguous()
                folded_count += 1
            tail_fixed_b = take_bytes((32 + 32 + 128 + 32 + 32 + 32) * 4)
            l_qkv, l_z, l_out, l_moe = struct.unpack_from("<IIII", raw, off)
            off += 16
            b_qkv = take_bytes(l_qkv)
            b_z = take_bytes(l_z)
            b_out = take_bytes(l_out)
            b_moe = take_bytes(l_moe)

            w_qkv = unpack_hg8_16r_blob(b_qkv)
            w_z = unpack_hg8_16r_blob(b_z)
            w_out = unpack_hg8_16r_blob(b_out)
            for name_m, w_ref in [("linear_attn.in_proj_qkv", w_qkv), ("linear_attn.in_proj_z", w_z), ("linear_attn.out_proj", w_out)]:
                d = get_delta(lid, name_m)
                if d is not None:
                    w_ref.add_(d)
                    folded_count += 1
            b_qkv = pack_hg8_16r_blob(w_qkv, seed=300 + lid * 10 + 1)
            b_z = pack_hg8_16r_blob(w_z, seed=300 + lid * 10 + 2)
            b_out = pack_hg8_16r_blob(w_out, seed=300 + lid * 10 + 3)
        else:
            qn_b = take_bytes(256 * 4)
            kn_b = take_bytes(256 * 4)
            l_q, l_k, l_v, l_o, l_moe = struct.unpack_from("<IIIII", raw, off)
            off += 20
            b_q = take_bytes(l_q)
            b_k = take_bytes(l_k)
            b_v = take_bytes(l_v)
            b_o = take_bytes(l_o)
            b_moe = take_bytes(l_moe)

            w_q = unpack_hg8_16r_blob(b_q)
            w_k = unpack_hg8_16r_blob(b_k)
            w_v = unpack_hg8_16r_blob(b_v)
            w_o = unpack_hg8_16r_blob(b_o)
            for name_m, w_ref in [("self_attn.q_proj", w_q), ("self_attn.k_proj", w_k), ("self_attn.v_proj", w_v), ("self_attn.o_proj", w_o)]:
                d = get_delta(lid, name_m)
                if d is not None:
                    w_ref.add_(d)
                    folded_count += 1
            b_q = pack_hg8_16r_blob(w_q, seed=400 + lid * 10 + 1)
            b_k = pack_hg8_16r_blob(w_k, seed=400 + lid * 10 + 2)
            b_v = pack_hg8_16r_blob(w_v, seed=400 + lid * 10 + 3)
            b_o = pack_hg8_16r_blob(w_o, seed=400 + lid * 10 + 4)

        assert off == len(raw), f"Layer {lid} offset mismatch {off} != {len(raw)}"

        # Unpack & fold shared_expert inside b_moe
        n_exp, l_sh13, l_shdn = struct.unpack_from("<III", b_moe, 0)
        moe_hdr_end = 12 + (256 * 2048 * 4) + (2048 * 4)
        moe_fixed = b_moe[12:moe_hdr_end]
        sh13_old = b_moe[moe_hdr_end : moe_hdr_end + l_sh13]
        shdn_old = b_moe[moe_hdr_end + l_sh13 : moe_hdr_end + l_sh13 + l_shdn]

        sh_gate, sh_up = deinterleave_swiglu_16r(unpack_hg8_16r_blob(sh13_old))
        sh_dn = unpack_hg8_16r_blob(shdn_old)
        for name_m, w_ref in [
            ("mlp.shared_expert.gate_proj", sh_gate),
            ("mlp.shared_expert.up_proj", sh_up),
            ("mlp.shared_expert.down_proj", sh_dn),
        ]:
            d = get_delta(lid, name_m)
            if d is not None:
                w_ref.add_(d)
                folded_count += 1

        b_sh13_new = pack_hg8_16r_blob(interleave_swiglu_16r(sh_gate, sh_up), seed=500 + lid * 10 + 1)
        pd_sdn = swiglu_eq_pre_diag(sh_gate, sh_up, sh_dn, seed=500 + lid * 10 + 2)
        b_shdn_new = pack_hg8_16r_blob(sh_dn, fixed_pre_diag=pd_sdn)
        b_moe_new = struct.pack("<III", n_exp, len(b_sh13_new), len(b_shdn_new)) + moe_fixed + b_sh13_new + b_shdn_new

        out_layer = out_dir / f"layer_{lid:02d}.hg4"
        if not is_gqa:
            out_layer.write_bytes(
                struct.pack("<I", 0)
                + in_ln_b + post_ln_b + conv_w_b + conv_rw_b
                + a_w.numpy().tobytes() + b_w.numpy().tobytes() + tail_fixed_b
                + struct.pack("<IIII", len(b_qkv), len(b_z), len(b_out), len(b_moe_new))
                + b_qkv + b_z + b_out + b_moe_new
            )
        else:
            out_layer.write_bytes(
                struct.pack("<I", 1)
                + in_ln_b + post_ln_b + qn_b + kn_b
                + struct.pack("<IIIII", len(b_q), len(b_k), len(b_v), len(b_o), len(b_moe_new))
                + b_q + b_k + b_v + b_o + b_moe_new
            )

    leftover = sorted(all_lora_a_keys - used_lora_a_keys)
    if leftover:
        raise RuntimeError(f"Unhandled LoRA weight keys ({len(leftover)}): {leftover[:8]}")

    op_script = HERE / "precompute_hg4_operators.py"
    subprocess.check_call([
        sys.executable, str(op_script),
        "--clover35b-dir", str(out_dir),
        "--nemotron-dir", "/nonexistent",
        "--kimi-dir", "/nonexistent",
    ])
    print(f"[LoRA Fold Complete] Folded {folded_count} linear projections into {out_dir} in {time.perf_counter()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
