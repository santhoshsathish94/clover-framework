#!/usr/bin/env python3
"""Pack DiffuSpace Clover-35B-A3B-Preview (`BD-GDN + Gated GQA + SwiGLU MoE + 248,320 Vocab`) into `HG4/HG8-128`.

Streams exact unquantized `BF16` expert slices from `Diffuspace/Clover-35B-A3B-Preview` (saving a local `BF16` cache
for exact 3-way `BF16` vs `v5.0.7` vs `HG4-128` benchmarking) and packs `/opt/clover-k3/clover35b-hg4/`:
  - `embed_tokens.hg4` (`[248320, 2048]` in `HG4-128`, `250.1 MiB`, 1-row inverse-`FWHT-128` lookup)
  - `lm_head.hg4` (`[248320, 2048]` in `HG8-128` VNNI, `492.4 MiB`, `< 4.5 ms` 248K readout)
  - `layer0_bdgdn.hg4` (Layer 0 Bidirectional Gated DeltaNet + Shared SwiGLU Expert + `bd_gdn` preview weights)
  - `experts0.hg4` (`16` cached SwiGLU experts with Shared-Input `pre_diag_up`, Register-Fused `[gate, up]` tiles,
    and Closed-Form SwiGLU Equilibrium `pre_diag_down`)
  - `layer3_gqa.hg4` (Layer 3 Gated 16Q/2KV Attention + Partial RoPE 0.25 + Shared SwiGLU Expert)
  - `experts3.hg4` (`16` cached SwiGLU experts in `HG4-128`)
"""
from __future__ import annotations

import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "16")
torch.set_num_threads(16)

for p in (
    Path("/opt/clover-k3/clover-framework/porting-models"),
    Path(__file__).resolve().parents[4] / "porting-models" if len(Path(__file__).resolve().parents) >= 5 else Path("/nonexistent"),
):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import clover_35b_gpu_backbone as cb  # noqa: E402

LUT16_U8 = np.array([1, 32, 53, 70, 84, 97, 110, 122, 134, 146, 159, 172, 186, 203, 224, 255], dtype=np.uint8)
LUT16_S8 = (LUT16_U8.astype(np.int32) - 128).astype(np.float32)
CODE_MIDS = 0.5 * (LUT16_S8[:-1] + LUT16_S8[1:])


def u16_to_f32_torch(arr_u16: np.ndarray) -> torch.Tensor:
    u32 = np.left_shift(np.asarray(arr_u16, dtype=np.uint32), np.uint32(16))
    return torch.from_numpy(u32.view(np.float32))


def fwht128_torch(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    v = x.reshape(-1, 128)
    h = 1
    while h < 128:
        v = v.reshape(-1, 128 // (2 * h), 2, h)
        v = torch.stack((v[:, :, 0, :] + v[:, :, 1, :], v[:, :, 0, :] - v[:, :, 1, :]), dim=2)
        h *= 2
    return v.reshape(orig_shape) * (1.0 / 11.313708498984761)


def make_signs(K: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.where(torch.randint(0, 2, (K,), generator=g) == 0, -1.0, 1.0).to(torch.float32)


def interleave_swiglu_rows(gate_w: torch.Tensor, up_w: torch.Tensor) -> torch.Tensor:
    """Interleave `[R_half, K]` `gate_w` and `up_w` as `[g0, g1, u0, u1]` per 4-row tile (`[2*R_half, K]`)."""
    rh, K = gate_w.shape
    assert rh % 2 == 0
    g2 = gate_w.reshape(rh // 2, 2, K)
    u2 = up_w.reshape(rh // 2, 2, K)
    return torch.cat([g2, u2], dim=1).reshape(2 * rh, K).contiguous()


def swiglu_eq_pre_diag(gate_w: torch.Tensor, up_w: torch.Tensor, dn_w: torch.Tensor, seed: int) -> torch.Tensor:
    """Closed-form SwiGLU Equilibrium `pre_diag`: `sqrt(col_rms(dn) / (row_rms(gate) * row_rms(up))) * signs`."""
    dn_crms = dn_w.pow(2).mean(dim=0).clamp_min(1e-12).sqrt()
    gu_rrms = (gate_w.pow(2).mean(dim=1).clamp_min(1e-12).sqrt()) * (up_w.pow(2).mean(dim=1).clamp_min(1e-12).sqrt())
    return ((dn_crms / gu_rrms.clamp_min(1e-12)).pow(0.5) * make_signs(dn_w.shape[1], seed)).contiguous()


def pack_hg4_mat(W_f32: torch.Tensor, seed: int = 42, fixed_pre_diag: torch.Tensor | None = None) -> bytes:
    R, K = W_f32.shape
    assert R % 4 == 0 and K % 128 == 0
    nb = K // 128
    mids_t = torch.from_numpy(CODE_MIDS)
    lut_t = torch.from_numpy(LUT16_S8)
    if fixed_pre_diag is not None:
        pre_diag = fixed_pre_diag.to(torch.float32).contiguous()
    else:
        col_rms = torch.sqrt(torch.mean(W_f32 * W_f32, dim=0).clamp_min(1e-12)).pow(0.5)
        pre_diag = (col_rms * make_signs(K, seed)).contiguous()

    scales_chunks, tiles_chunks = [], []
    for r0 in range(0, R, 16384):
        r1 = min(R, r0 + 16384)
        wc = W_f32[r0:r1] / pre_diag.unsqueeze(0)
        w_rot = fwht128_torch(wc.reshape(r1 - r0, nb, 128))
        sc = w_rot.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
        for _ in range(2):
            idx = torch.searchsorted(mids_t, w_rot / sc)
            qv = lut_t[idx]
            sc = ((w_rot * qv).sum(dim=-1, keepdim=True) / (qv * qv).sum(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12)
        sc_fp16 = sc.squeeze(-1).to(torch.float16)
        sc_f32 = sc_fp16.to(torch.float32).unsqueeze(-1).clamp_min(1e-12)
        q_idx = torch.searchsorted(mids_t, w_rot / sc_f32).to(torch.uint8)

        rc4 = (r1 - r0) // 4
        sc_tiled = sc_fp16.reshape(rc4, 4, nb).permute(0, 2, 1).contiguous()
        q_tiled = q_idx.reshape(rc4, 4, nb, 2, 64).permute(0, 2, 3, 1, 4).contiguous()
        packed = (q_tiled[..., :32] | (q_tiled[..., 32:] << 4)).contiguous()
        scales_chunks.append(sc_tiled.numpy().view(np.uint16).tobytes())
        tiles_chunks.append(packed.numpy().tobytes())

    return struct.pack("<II", R, K) + pre_diag.numpy().tobytes() + b"".join(scales_chunks) + b"".join(tiles_chunks)


def pack_hg8_mat(W_f32: torch.Tensor, seed: int = 42, fixed_pre_diag: torch.Tensor | None = None) -> bytes:
    R, K = W_f32.shape
    assert R % 4 == 0 and K % 128 == 0
    nb = K // 128
    if fixed_pre_diag is not None:
        pre_diag = fixed_pre_diag.to(torch.float32).contiguous()
    else:
        col_rms = torch.sqrt(torch.mean(W_f32 * W_f32, dim=0).clamp_min(1e-12)).pow(0.5)
        pre_diag = (col_rms * make_signs(K, seed)).contiguous()

    scales_chunks, tiles_chunks = [], []
    for r0 in range(0, R, 16384):
        r1 = min(R, r0 + 16384)
        wc = W_f32[r0:r1] / pre_diag.unsqueeze(0)
        w_rot = fwht128_torch(wc.reshape(r1 - r0, nb, 128))
        sc_fp16 = (w_rot.abs().amax(dim=-1) / 127.0).clamp_min(1e-12).to(torch.float16)
        sc_f32 = sc_fp16.to(torch.float32).unsqueeze(-1).clamp_min(1e-12)
        q_u8 = (torch.round(w_rot / sc_f32).clamp(-127.0, 127.0) + 128.0).to(torch.uint8)

        rc4 = (r1 - r0) // 4
        sc_tiled = sc_fp16.reshape(rc4, 4, nb).permute(0, 2, 1).contiguous()
        q_tiled = q_u8.reshape(rc4, 4, nb, 2, 64).permute(0, 2, 3, 1, 4).contiguous()
        scales_chunks.append(sc_tiled.numpy().view(np.uint16).tobytes())
        tiles_chunks.append(q_tiled.numpy().tobytes())

    return struct.pack("<II", R, K | (8 << 28)) + pre_diag.numpy().tobytes() + b"".join(scales_chunks) + b"".join(tiles_chunks)


def pack_swiglu_experts_hg4(gu_bf16: torch.Tensor, dn_bf16: torch.Tensor, out_path: Path, seed_base: int) -> None:
    """Pack `[K_exp, 1024, 2048]` `gate_up` + `[K_exp, 2048, 512]` `down` into register-fused SwiGLU `HG4-128`."""
    k_exp = gu_bf16.shape[0]
    gu_f = gu_bf16.float()
    dn_f = dn_bf16.float()

    # Shared `pre_diag_up[2048]` across all cached experts so `u_mlp` is rotated & quantized ONCE per token!
    col_rms_up = gu_f.reshape(-1, 2048).pow(2).mean(dim=0).clamp_min(1e-12).sqrt().pow(0.5)
    pre_diag_up = (col_rms_up * make_signs(2048, seed_base)).contiguous()

    # Per-expert:
    #   gu_body (no header/pre_diag): scales_fp16 [256 * 16 * 4 * 2 = 32,768 B] + tiles [256 * 16 * 256 = 1,048,576 B] = 1,081,344 B
    #   dn_blob (with header + pre_diag[512]): 8 + 2048 + (512 * 4 * 4 * 2 = 16,384) + (512 * 4 * 256 = 524,288) = 542,728 B
    #   exp_stride = 1,624,072 B
    exp_stride = 1081344 + 542728
    with open(out_path, "wb") as wf:
        wf.write(struct.pack("<4sIIIIII", b"C35E", k_exp, 1024, 2048, 2048, 512, exp_stride))
        wf.write(pre_diag_up.numpy().tobytes())
        for e in range(k_exp):
            gate_e, up_e = gu_f[e, :512], gu_f[e, 512:]
            w13_inter = interleave_swiglu_rows(gate_e, up_e)  # [1024, 2048]
            gu_blob = pack_hg4_mat(w13_inter, fixed_pre_diag=pre_diag_up)
            gu_body = memoryview(gu_blob)[8 + 2048 * 4:]

            pd_dn = swiglu_eq_pre_diag(gate_e, up_e, dn_f[e], seed=seed_base + 1 + e)
            dn_blob = pack_hg4_mat(dn_f[e], fixed_pre_diag=pd_dn)
            assert len(gu_body) + len(dn_blob) == exp_stride
            wf.write(gu_body)
            wf.write(dn_blob)


def pack_shared_moe_tail(tr: cb.Clover35BTrunkReader, k_cached: int, seed_base: int) -> bytes:
    gate_w = tr.get_record(20, (256, 2048)).float()[:k_cached].contiguous()  # [k_cached, 2048]
    sh_gate = tr.get_record(21, (512, 2048)).float()
    sh_up = tr.get_record(22, (512, 2048)).float()
    sh_dn = tr.get_record(23, (2048, 512)).float()
    sh_g = tr.get_record(24, (2048,)).float()

    sh_w13 = interleave_swiglu_rows(sh_gate, sh_up)  # [1024, 2048]
    b_sh13 = pack_hg8_mat(sh_w13, seed=seed_base + 1)
    pd_sdn = swiglu_eq_pre_diag(sh_gate, sh_up, sh_dn, seed=seed_base + 2)
    b_shdn = pack_hg8_mat(sh_dn, fixed_pre_diag=pd_sdn)
    return (
        struct.pack("<III", k_cached, len(b_sh13), len(b_shdn))
        + gate_w.numpy().tobytes()
        + sh_g.numpy().tobytes()
        + b_sh13 + b_shdn
    )


def main() -> None:
    cb.enable_ipv4_only()
    t0 = time.perf_counter()
    stage = Path("/opt/clover-k3/clover-35b-computed/hf-stage")
    trunks = stage / "trunks"
    dst = Path("/opt/clover-k3/clover35b-hg4")
    dst.mkdir(parents=True, exist_ok=True)
    repo_id = "Diffuspace/Clover-35B-A3B-Preview"

    # Cache exact BF16 expert slices in /opt/clover-k3/clover-35b-computed/pruned_bf16_experts for 3-way reference benchmarking
    bf16_exp_cache = Path("/opt/clover-k3/clover-35b-computed/bf16_experts_cache")
    bf16_exp_cache.mkdir(parents=True, exist_ok=True)
    for lid, s_gu, s_dn in [(0, "00001", "00002"), (3, "00003", "00003")]:
        ed = stage / "experts" / f"expert-{lid}"
        bf_p = ed / "experts.direct"
        if not bf_p.is_file():
            cache_p = bf16_exp_cache / f"expert_{lid}.direct"
            if not cache_p.is_file():
                print(f"[0/5] Streaming exact BF16 expert-{lid} (16 experts) from {repo_id}...", flush=True)
                gu = cb.fetch_safetensor_leading_slice(repo_id, f"model-{s_gu}-of-00026.safetensors", f"model.layers.{lid}.mlp.experts.gate_up_proj", 16)
                dn = cb.fetch_safetensor_leading_slice(repo_id, f"model-{s_dn}-of-00026.safetensors", f"model.layers.{lid}.mlp.experts.down_proj", 16)
                cache_p.write_bytes(cb.tensor_to_le_bytes(gu.to(torch.bfloat16)) + cb.tensor_to_le_bytes(dn.to(torch.bfloat16)))
            os.symlink(cache_p, bf_p)

    print("[1/5] Packing meta.bin + zero-centered norm_w + RoPE inv_freq...", flush=True)
    norm_w = 1.0 + u16_to_f32_torch(np.fromfile(trunks / "global-0" / "norm.bf16.bin", dtype=np.uint16))
    inv_freq = (1.0 / (10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64.0))).contiguous()
    (dst / "meta.bin").write_bytes(
        struct.pack("<4sIII", b"C35M", 1, 2048, 248320)
        + LUT16_U8.tobytes()
        + norm_w.numpy().tobytes()
        + inv_freq.numpy().tobytes()
    )

    print("[2/5] Packing embed_tokens.hg4 ([248320, 2048], HG8-128)...", flush=True)
    emb_mm = np.memmap(trunks / "global-0" / "embed_tokens.bf16.bin", dtype=np.uint16, mode="r").reshape(248320, 2048)
    (dst / "embed_tokens.hg4").write_bytes(pack_hg8_mat(u16_to_f32_torch(emb_mm), seed=101))

    print("[3/5] Packing lm_head.hg4 ([248320, 2048], HG8-128 VNNI)...", flush=True)
    lm_mm = np.memmap(trunks / "global-0" / "lm_head.bf16.bin", dtype=np.uint16, mode="r").reshape(248320, 2048)
    (dst / "lm_head.hg4").write_bytes(pack_hg8_mat(u16_to_f32_torch(lm_mm), seed=202))

    print("[4/5] Packing Layer 0 BD-GDN + 16 SwiGLU Experts (layer0_bdgdn.hg4 + experts0.hg4)...", flush=True)
    tr0 = cb.Clover35BTrunkReader(trunks / "trunk-0")
    op0 = tr0.unpack_operator()
    bd0 = tr0.bd_gdn
    assert bd0 is not None
    in_ln0 = (1.0 + op0["in_ln"].float()).contiguous()
    post_ln0 = (1.0 + tr0.get_record(1, (2048,)).float()).contiguous()
    conv_w0 = op0["conv1d"].float().reshape(8192 * 4).contiguous()
    conv_rw0 = bd0["conv_right_weight"].float().reshape(8192 * 3).contiguous()
    a_w0 = tr0.get_record(6, (32, 2048)).float().contiguous()
    b_w0 = tr0.get_record(7, (32, 2048)).float().contiguous()
    neg_exp_A0 = (-tr0.get_record(8, (32,)).float().exp()).contiguous()
    dt_bias0 = tr0.get_record(9, (32,)).float().contiguous()
    gdn_norm0 = tr0.get_record(10, (128,)).float().contiguous()  # zero_centered=False!
    ls0 = bd0["bd_softmax_logit_scale"].float().reshape(32).contiguous()
    hb0 = bd0["bd_softmax_history_bias"].float().reshape(32).contiguous()
    hc0 = bd0["bd_softmax_history_c"].float().reshape(32).contiguous()

    b_qkv0 = pack_hg8_mat(op0["in_proj_qkv"].float(), seed=301)
    b_z0 = pack_hg8_mat(op0["in_proj_z"].float(), seed=302)
    b_out0 = pack_hg8_mat(tr0.get_record(2, (2048, 4096)).float(), seed=303)
    b_moe0 = pack_shared_moe_tail(tr0, 16, seed_base=310)
    (dst / "layer0_bdgdn.hg4").write_bytes(
        in_ln0.numpy().tobytes()
        + post_ln0.numpy().tobytes()
        + conv_w0.numpy().tobytes()
        + conv_rw0.numpy().tobytes()
        + a_w0.numpy().tobytes()
        + b_w0.numpy().tobytes()
        + neg_exp_A0.numpy().tobytes()
        + dt_bias0.numpy().tobytes()
        + gdn_norm0.numpy().tobytes()
        + ls0.numpy().tobytes()
        + hb0.numpy().tobytes()
        + hc0.numpy().tobytes()
        + struct.pack("<IIII", len(b_qkv0), len(b_z0), len(b_out0), len(b_moe0))
        + b_qkv0 + b_z0 + b_out0 + b_moe0
    )
    gu0, dn0, _ = cb.load_expert_store(stage / "experts" / "expert-0", use_4bit=False)
    pack_swiglu_experts_hg4(gu0, dn0, dst / "experts0.hg4", seed_base=1000)

    print("[5/5] Packing Layer 3 Gated GQA + 16 SwiGLU Experts (layer3_gqa.hg4 + experts3.hg4)...", flush=True)
    tr3 = cb.Clover35BTrunkReader(trunks / "trunk-3")
    op3 = tr3.unpack_operator()
    in_ln3 = (1.0 + op3["in_ln"].float()).contiguous()
    post_ln3 = (1.0 + tr3.get_record(1, (2048,)).float()).contiguous()
    qn3 = (1.0 + op3["q_norm"].float()).contiguous()
    kn3 = (1.0 + op3["k_norm"].float()).contiguous()
    b_q3 = pack_hg8_mat(op3["q_proj"].float(), seed=401)
    b_k3 = pack_hg8_mat(op3["k_proj"].float(), seed=402)
    b_v3 = pack_hg8_mat(op3["v_proj"].float(), seed=403)
    b_o3 = pack_hg8_mat(tr3.get_record(2, (2048, 4096)).float(), seed=404)
    b_moe3 = pack_shared_moe_tail(tr3, 16, seed_base=410)
    (dst / "layer3_gqa.hg4").write_bytes(
        in_ln3.numpy().tobytes()
        + post_ln3.numpy().tobytes()
        + qn3.numpy().tobytes()
        + kn3.numpy().tobytes()
        + struct.pack("<IIIII", len(b_q3), len(b_k3), len(b_v3), len(b_o3), len(b_moe3))
        + b_q3 + b_k3 + b_v3 + b_o3 + b_moe3
    )
    gu3, dn3, _ = cb.load_expert_store(stage / "experts" / "expert-3", use_4bit=False)
    pack_swiglu_experts_hg4(gu3, dn3, dst / "experts3.hg4", seed_base=2000)

    tot_b = sum(f.stat().st_size for f in dst.iterdir() if f.is_file())
    print(
        f"\nDONE in {time.perf_counter() - t0:.1f}s! "
        f"Total Clover-35B-A3B HG4/HG8-128 size = {tot_b:,} bytes ({tot_b / (1024**2):.2f} MiB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
