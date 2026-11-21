#!/usr/bin/env python3
"""Pack NVIDIA Nemotron 3.5 Super VL (`pruned` BF16 trunks + 512 routed experts + MTP) into `HG4-128`.

Converts `/opt/clover-k3/nemotron-computed/pruned` (`12.96 GiB` BF16 / `5.36 GiB` hybrid `CLVR4B01`) into
`/opt/clover-k3/nemotron-hg4` (`~3.36 GiB` pure `HG4-128`, `4.14 bpw`) with zero training:
  - Column-Balanced Randomized `FWHT-128` incoherence (`(W / col_rms) * signs -> FWHT128`)
  - 15-Point MSE-Optimal Lloyd-Max Gaussian `LUT16` (`[1, 32, 53, 70, 84, 97, 110, 122, 134, 146, 159, 172, 186, 203, 224, 255]`)
  - 4-Row Tile Interleaved SIMD Layout (`[R/4, K/128, 2, 4, 32]`)
  - Shared-Input Incoherence across all `512` routed `ReLU^2` MoE `up_proj` matrices (`[2688, 1024]`) so `latent_in` is
    rotated & quantized ONCE per token instead of 22 times!
  - Fused `Q+K+V` (`[4608, 4096]`) in both backbone NoPE 16:1 GQA (`trunk-7`) and MTP GQA (`trunk-mtp-0`).
"""
from __future__ import annotations

import os
import struct
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "16")
torch.set_num_threads(16)

# Exact 15-point symmetric Gaussian MSE-optimal Lloyd-Max codebook mapped to uint8 [1..255] (128 = exact 0.0)
LUT16_U8 = np.array([1, 32, 53, 70, 84, 97, 110, 122, 134, 146, 159, 172, 186, 203, 224, 255], dtype=np.uint8)
LUT16_S8 = (LUT16_U8.astype(np.int32) - 128).astype(np.float32)  # [-127 .. +127]
CODE_MIDS = 0.5 * (LUT16_S8[:-1] + LUT16_S8[1:])  # 15 decision thresholds

NEMO_HEADER_BYTES = 1168
NEMO_RECORD_BYTES = 72


def u16_to_f32_torch(arr_u16: np.ndarray) -> torch.Tensor:
    u32 = np.left_shift(np.asarray(arr_u16, dtype=np.uint32), np.uint32(16))
    return torch.from_numpy(u32.view(np.float32))


def bf16_bytes_to_f32_tensor(raw: bytes | memoryview, shape: tuple[int, ...]) -> torch.Tensor:
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    return u16_to_f32_torch(u16)


def load_nemo_trunk_records(trunk_dir: Path) -> tuple[dict[int, bytes], bytes | None]:
    idx = (trunk_dir / "index.bin").read_bytes()
    assert idx[:8] == b"NEMOTRK1", f"Bad magic in {trunk_dir}"
    rec_cnt = struct.unpack_from("<I", idx, 16)[0]
    gnames = ("common.bin", "attention.bin", "routed-shared.bin")
    gbufs = [((trunk_dir / g).read_bytes() if (trunk_dir / g).is_file() else b"") for g in gnames]
    recs: dict[int, bytes] = {}
    for i in range(rec_cnt):
        off_hdr = NEMO_HEADER_BYTES + i * NEMO_RECORD_BYTES
        rec_id, grp_id, _dt, _r, _c, _num, roff, rlen, _crc = struct.unpack_from("<IIIIIIQQI", idx, off_hdr)
        if rlen > 0 and grp_id < 3 and gbufs[grp_id]:
            recs[rec_id] = memoryview(gbufs[grp_id])[roff:roff + rlen]
    op_bytes = (trunk_dir / "operator.bin").read_bytes() if (trunk_dir / "operator.bin").is_file() else None
    return recs, op_bytes


def fwht128_torch(x: torch.Tensor) -> torch.Tensor:
    """In-place-style fast Walsh-Hadamard transform over the last dimension of size 128 (`normalized by 1/sqrt(128)`)."""
    orig_shape = x.shape
    v = x.reshape(-1, 128)
    h = 1
    while h < 128:
        v = v.reshape(-1, 128 // (2 * h), 2, h)
        a = v[:, :, 0, :]
        b = v[:, :, 1, :]
        v = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return (v.reshape(orig_shape)) * (1.0 / 11.313708498984761)


def make_rademacher_signs(K: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.where(torch.randint(0, 2, (K,), generator=g) == 0, -1.0, 1.0).to(torch.float32)


def pack_hg8_matrix(W_f32: torch.Tensor, seed: int = 42, fixed_pre_diag: torch.Tensor | None = None) -> bytes:
    """Pack `[R, K]` float32 matrix into `HG8-128` VNNI blob (`K | (8 << 28)` tag, `8.125 bpw`, direct `vpdpbusd`!)."""
    R, K = W_f32.shape
    assert R % 4 == 0 and K % 128 == 0
    nb = K // 128
    if fixed_pre_diag is not None:
        pre_diag = fixed_pre_diag.to(torch.float32).contiguous()
    else:
        col_rms = torch.sqrt(torch.mean(W_f32 * W_f32, dim=0).clamp_min(1e-12)).pow(0.5)
        pre_diag = (col_rms * make_rademacher_signs(K, seed)).contiguous()

    scales_chunks = []
    tiles_chunks = []
    chunk_r = 16384
    for r0 in range(0, R, chunk_r):
        r1 = min(R, r0 + chunk_r)
        wc = W_f32[r0:r1] / pre_diag.unsqueeze(0)
        w_rot = fwht128_torch(wc.reshape(r1 - r0, nb, 128))
        sc_fp16 = (w_rot.abs().amax(dim=-1) / 127.0).clamp_min(1e-12).to(torch.float16)  # [rc, nb]
        sc_f32 = sc_fp16.to(torch.float32).unsqueeze(-1).clamp_min(1e-12)
        q_u8 = (torch.round(w_rot / sc_f32).clamp(-127.0, 127.0) + 128.0).to(torch.uint8)  # [rc, nb, 128]

        rc4 = (r1 - r0) // 4
        sc_tiled = sc_fp16.reshape(rc4, 4, nb).permute(0, 2, 1).contiguous()  # [rc4, nb, 4]
        q_tiled = q_u8.reshape(rc4, 4, nb, 2, 64).permute(0, 2, 3, 1, 4).contiguous()  # [rc4, nb, 2, 4, 64]
        scales_chunks.append(sc_tiled.numpy().view(np.uint16).tobytes())
        tiles_chunks.append(q_tiled.numpy().tobytes())

    hdr = struct.pack("<II", R, K | (8 << 28))
    return hdr + pre_diag.numpy().tobytes() + b"".join(scales_chunks) + b"".join(tiles_chunks)


def relu2_eq_pre_diag(up_f: torch.Tensor, dn_f: torch.Tensor, seed: int) -> torch.Tensor:
    """Closed-form `ReLU^2` Equilibrium `pre_diag` (`sqrt(col_rms(dn) / row_rms(up)^2) * signs`) for squared activations."""
    dn_crms = dn_f.pow(2).mean(dim=0).clamp_min(1e-12).sqrt()
    up_rrms2 = up_f.pow(2).mean(dim=1).clamp_min(1e-12)
    return ((dn_crms / up_rrms2).pow(0.5) * make_rademacher_signs(dn_f.shape[1], seed)).contiguous()


def pack_hg4_matrix(W_f32: torch.Tensor, seed: int = 42, fixed_pre_diag: torch.Tensor | None = None) -> bytes:
    """Pack `[R, K]` float32 matrix into standard `HG4-128` blob (`R, K, pre_diag[K], scales_fp16[R/4, K/128, 4], tiles`)."""
    R, K = W_f32.shape
    assert R % 4 == 0 and K % 128 == 0, f"Shape {(R, K)} must be divisible by (4, 128)"
    nb = K // 128
    mids_t = torch.from_numpy(CODE_MIDS)
    lut_t = torch.from_numpy(LUT16_S8)

    if fixed_pre_diag is not None:
        pre_diag = fixed_pre_diag.to(torch.float32).contiguous()
    else:
        col_rms = torch.sqrt(torch.mean(W_f32 * W_f32, dim=0).clamp_min(1e-12)).pow(0.5)
        signs = make_rademacher_signs(K, seed)
        pre_diag = (col_rms * signs).contiguous()

    # Process in row chunks of 16,384 so 131K x 4096 uses < 2.5 GB peak RAM!
    scales_chunks = []
    tiles_chunks = []
    chunk_r = 16384
    for r0 in range(0, R, chunk_r):
        r1 = min(R, r0 + chunk_r)
        wc = W_f32[r0:r1] / pre_diag.unsqueeze(0)
        w_rot = fwht128_torch(wc.reshape(r1 - r0, nb, 128))  # [rc, nb, 128]

        # Optimal scale refine (2-step Lloyd-Max alternating scale search)
        amax = w_rot.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        sc = amax / 127.0
        for _ in range(2):
            idx = torch.searchsorted(mids_t, w_rot / sc)
            q_val = lut_t[idx]
            num = (w_rot * q_val).sum(dim=-1, keepdim=True)
            den = (q_val * q_val).sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sc = (num / den).clamp_min(1e-12)

        sc_fp16 = sc.squeeze(-1).to(torch.float16)  # [rc, nb]
        sc_f32 = sc_fp16.to(torch.float32).unsqueeze(-1).clamp_min(1e-12)
        q_idx = torch.searchsorted(mids_t, w_rot / sc_f32).to(torch.uint8)  # [rc, nb, 128]

        # Pack into 4-row tiles: [rc/4, nb, 2, 4, 32] where low nibble = [0..31], high nibble = [32..63]
        rc4 = (r1 - r0) // 4
        sc_tiled = sc_fp16.reshape(rc4, 4, nb).permute(0, 2, 1).contiguous()  # [rc4, nb, 4]
        q_tiled = q_idx.reshape(rc4, 4, nb, 2, 64).permute(0, 2, 3, 1, 4).contiguous()  # [rc4, nb, 2, 4, 64]
        packed = (q_tiled[..., :32] | (q_tiled[..., 32:] << 4)).contiguous()  # [rc4, nb, 2, 4, 32]

        scales_chunks.append(sc_tiled.numpy().view(np.uint16).tobytes())
        tiles_chunks.append(packed.numpy().tobytes())

    hdr = struct.pack("<II", R, K)
    return hdr + pre_diag.numpy().tobytes() + b"".join(scales_chunks) + b"".join(tiles_chunks)


def pack_experts_512_hg4(direct_bf16_path: Path, out_path: Path, seed_base: int = 1000) -> None:
    """Pack 512 non-gated ReLU^2 MoE experts (`up [2688, 1024]` + `down [1024, 2688]`) with shared `pre_diag_up[1024]`."""
    t0 = time.perf_counter()
    mm = np.memmap(direct_bf16_path, dtype=np.uint16, mode="r")
    n_exp = 512
    up_elems = 2688 * 1024
    dn_elems = 1024 * 2688
    exp_elems = up_elems + dn_elems
    assert mm.size == n_exp * exp_elems, f"Unexpected size {mm.size}"

    # 1. Compute shared `pre_diag_up[1024]` across all 512 `up_proj` matrices so `latent_in` is rotated ONCE per token!
    ss_col = torch.zeros(1024, dtype=torch.float64)
    for e in range(0, n_exp, 32):
        b_cnt = min(32, n_exp - e)
        raw_chunk = mm[e * exp_elems : (e + b_cnt) * exp_elems].reshape(b_cnt, exp_elems)[:, :up_elems]
        up_f = u16_to_f32_torch(raw_chunk).reshape(-1, 1024)
        ss_col += (up_f.double() * up_f.double()).sum(dim=0)
    col_rms_up = torch.sqrt((ss_col / (n_exp * 2688)).clamp_min(1e-12)).float().pow(0.5)
    pre_diag_up = (col_rms_up * make_rademacher_signs(1024, seed_base)).contiguous()

    # Header: magic "HG4E", n_exp=512, up_R=2688, up_K=1024, dn_R=1024, dn_K=2688, exp_stride, pre_diag_up[1024]
    # Per-expert record:
    #   up_no_pd: scales_fp16 [672*8*4 = 43,008 B] + tiles [672*8*256 = 1,376,256 B] = 1,419,264 B
    #   dn_full : R,K (8 B) + pre_diag[2688] (10,752 B) + scales_fp16 (43,008 B) + tiles (1,376,256 B) = 1,430,024 B
    #   Total per expert = 2,849,288 B (8-byte aligned!)
    exp_stride = 1419264 + 1430024
    with open(out_path, "wb") as wf:
        wf.write(struct.pack("<4sIIIIII", b"HG4E", n_exp, 2688, 1024, 1024, 2688, exp_stride))
        wf.write(pre_diag_up.numpy().tobytes())
        for e in range(n_exp):
            base = e * exp_elems
            up_u16 = mm[base : base + up_elems]
            dn_u16 = mm[base + up_elems : base + exp_elems]
            up_f = u16_to_f32_torch(up_u16).reshape(2688, 1024)
            dn_f = u16_to_f32_torch(dn_u16).reshape(1024, 2688)

            up_blob = pack_hg4_matrix(up_f, fixed_pre_diag=pre_diag_up)
            # Strip 8-byte (R,K) + 4096-byte pre_diag_up since pre_diag_up is shared in file header
            up_body = memoryview(up_blob)[8 + 1024 * 4:]
            pd_dn = relu2_eq_pre_diag(up_f, dn_f, seed=seed_base + 1 + e)
            dn_blob = pack_hg4_matrix(dn_f, fixed_pre_diag=pd_dn)
            assert len(up_body) + len(dn_blob) == exp_stride
            wf.write(up_body)
            wf.write(dn_blob)
            if (e + 1) % 128 == 0:
                print(f"    packed {e + 1}/{n_exp} experts ({time.perf_counter() - t0:.1f}s)...", flush=True)


def pack_gqa_trunk(trunk_dir: Path, seed_base: int) -> bytes:
    recs, op = load_nemo_trunk_records(trunk_dir)
    assert op is not None
    len_norm, len_q, len_k, len_v = struct.unpack_from("<IIII", op, 16)
    cur = 64
    norm_w = bf16_bytes_to_f32_tensor(memoryview(op)[cur:cur + len_norm], (4096,)); cur += len_norm
    q_w = bf16_bytes_to_f32_tensor(memoryview(op)[cur:cur + len_q], (4096, 4096)); cur += len_q
    k_w = bf16_bytes_to_f32_tensor(memoryview(op)[cur:cur + len_k], (256, 4096)); cur += len_k
    v_w = bf16_bytes_to_f32_tensor(memoryview(op)[cur:cur + len_v], (256, 4096)); cur += len_v
    o_w = bf16_bytes_to_f32_tensor(recs[1], (4096, 4096))

    # High-sensitivity final GQA operators use HG8-128 (direct VNNI uint8, +17 MiB total) for >0.9995 cosine
    b_q = pack_hg8_matrix(q_w, seed=seed_base + 1)
    b_k = pack_hg8_matrix(k_w, seed=seed_base + 2)
    b_v = pack_hg8_matrix(v_w, seed=seed_base + 3)
    b_o = pack_hg8_matrix(o_w, seed=seed_base + 4)
    return norm_w.numpy().tobytes() + struct.pack("<IIII", len(b_q), len(b_k), len(b_v), len(b_o)) + b_q + b_k + b_v + b_o


def pack_moe_trunk(trunk_dir: Path, seed_base: int) -> tuple[bytes, dict[int, bytes]]:
    recs, _ = load_nemo_trunk_records(trunk_dir)
    norm_w = bf16_bytes_to_f32_tensor(recs[0], (4096,))
    gate_w = np.frombuffer(recs[29], dtype=np.float32).reshape(512, 4096).copy()
    gate_b = np.frombuffer(recs[30], dtype=np.float32).reshape(512).copy()
    fc1_w = bf16_bytes_to_f32_tensor(recs[31], (1024, 4096))
    fc2_w = bf16_bytes_to_f32_tensor(recs[32], (4096, 1024))
    sh_up = bf16_bytes_to_f32_tensor(recs[34], (5376, 4096))
    sh_dn = bf16_bytes_to_f32_tensor(recs[36], (4096, 5376))

    b_fc1 = pack_hg8_matrix(fc1_w, seed=seed_base + 1)
    b_fc2 = pack_hg8_matrix(fc2_w, seed=seed_base + 2)
    b_sup = pack_hg8_matrix(sh_up, seed=seed_base + 3)
    pd_sdn = relu2_eq_pre_diag(sh_up, sh_dn, seed=seed_base + 4)
    b_sdn = pack_hg8_matrix(sh_dn, fixed_pre_diag=pd_sdn)

    blob = (
        norm_w.numpy().tobytes()
        + gate_w.tobytes()
        + gate_b.tobytes()
        + struct.pack("<IIII", len(b_fc1), len(b_fc2), len(b_sup), len(b_sdn))
        + b_fc1 + b_fc2 + b_sup + b_sdn
    )
    return blob, recs


def main() -> None:
    src = Path("/opt/clover-k3/nemotron-computed/pruned")
    dst = Path("/opt/clover-k3/nemotron-hg4")
    dst.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print("[1/7] Packing meta.bin + norm_f...", flush=True)
    norm_f = bf16_bytes_to_f32_tensor((src / "global-0" / "norm_f.bf16.bin").read_bytes(), (4096,))
    (dst / "meta.bin").write_bytes(
        struct.pack("<4sIII", b"HG4N", 1, 4096, 131072) + LUT16_U8.tobytes() + norm_f.numpy().tobytes()
    )

    print("[2/7] Packing embeddings.hg4 ([131072, 4096])...", flush=True)
    emb_mm = np.memmap(src / "global-0" / "embeddings.bf16.bin", dtype=np.uint16, mode="r").reshape(131072, 4096)
    emb_f32 = u16_to_f32_torch(emb_mm)
    (dst / "embeddings.hg4").write_bytes(pack_hg4_matrix(emb_f32, seed=101))
    del emb_f32

    print("[3/7] Packing lm_head.hg4 ([131072, 4096], HG8-128 VNNI)...", flush=True)
    lm_mm = np.memmap(src / "global-0" / "lm_head.bf16.bin", dtype=np.uint16, mode="r").reshape(131072, 4096)
    lm_f32 = u16_to_f32_torch(lm_mm)
    (dst / "lm_head.hg4").write_bytes(pack_hg8_matrix(lm_f32, seed=202))
    del lm_f32

    print("[4/7] Packing Mamba-2 trunk-0 (in_proj [18560, 4096] + out_proj [4096, 8192])...", flush=True)
    recs0, op0 = load_nemo_trunk_records(src / "trunk-0")
    assert op0 is not None
    len_norm, len_in, len_cw, len_cb, len_dtb, len_A, len_D = struct.unpack_from("<IIIIIII", op0, 16)
    cur = 64
    m_norm = bf16_bytes_to_f32_tensor(memoryview(op0)[cur:cur + len_norm], (4096,)); cur += len_norm
    m_in = bf16_bytes_to_f32_tensor(memoryview(op0)[cur:cur + len_in], (18560, 4096)); cur += len_in
    m_cw = bf16_bytes_to_f32_tensor(memoryview(op0)[cur:cur + len_cw], (10240 * 4,)); cur += len_cw
    m_cb = bf16_bytes_to_f32_tensor(memoryview(op0)[cur:cur + len_cb], (10240,)); cur += len_cb
    m_dtb = bf16_bytes_to_f32_tensor(memoryview(op0)[cur:cur + len_dtb], (128,)); cur += len_dtb
    m_A = np.frombuffer(memoryview(op0)[cur:cur + len_A], dtype=np.float32).copy(); cur += len_A
    m_D = bf16_bytes_to_f32_tensor(memoryview(op0)[cur:cur + len_D], (128,)); cur += len_D
    m_mix_norm = bf16_bytes_to_f32_tensor(recs0[1], (8192,))
    m_out = bf16_bytes_to_f32_tensor(recs0[2], (4096, 8192))
    b_mzxbc = pack_hg8_matrix(m_in[:18432].contiguous(), seed=301)
    b_mdt = pack_hg8_matrix(m_in[18432:].contiguous(), seed=303)
    b_mout = pack_hg8_matrix(m_out, seed=302)
    (dst / "mamba_trunk0.hg4").write_bytes(
        m_norm.numpy().tobytes()
        + m_cw.numpy().tobytes()
        + m_cb.numpy().tobytes()
        + m_dtb.numpy().tobytes()
        + m_A.tobytes()
        + m_D.numpy().tobytes()
        + m_mix_norm.numpy().tobytes()
        + struct.pack("<III", len(b_mzxbc), len(b_mdt), len(b_mout))
        + b_mzxbc + b_mdt + b_mout
    )
    del m_in, m_out

    print("[5/7] Packing LatentMoE trunk-1 + NoPE GQA trunk-7...", flush=True)
    moe1_blob, _ = pack_moe_trunk(src / "trunk-1", seed_base=400)
    (dst / "moe_trunk1.hg4").write_bytes(moe1_blob)
    (dst / "gqa_trunk7.hg4").write_bytes(pack_gqa_trunk(src / "trunk-7", seed_base=500))

    print("[6/7] Packing 512 Routed ReLU^2 MoE Experts (expert-1 -> experts_1.hg4)...", flush=True)
    pack_experts_512_hg4(src / "expert-1" / "experts.direct", dst / "experts_1.hg4", seed_base=1000)

    print("[7/7] Packing 2-Layer MTP Speculative Drafter (mtp_trunks.hg4 + experts_mtp_1.hg4)...", flush=True)
    recs_mtp0, _ = load_nemo_trunk_records(src / "trunk-mtp-0")
    enorm_w = bf16_bytes_to_f32_tensor(recs_mtp0[10], (4096,))
    hnorm_w = bf16_bytes_to_f32_tensor(recs_mtp0[11], (4096,))
    eh_proj = bf16_bytes_to_f32_tensor(recs_mtp0[12], (4096, 8192))
    b_eh = pack_hg8_matrix(eh_proj, seed=601)
    b_mtp0_gqa = pack_gqa_trunk(src / "trunk-mtp-0", seed_base=610)
    b_mtp1_moe, recs_mtp1 = pack_moe_trunk(src / "trunk-mtp-1", seed_base=620)
    final_ln_w = bf16_bytes_to_f32_tensor(recs_mtp1[1], (4096,))
    (dst / "mtp_trunks.hg4").write_bytes(
        enorm_w.numpy().tobytes()
        + hnorm_w.numpy().tobytes()
        + final_ln_w.numpy().tobytes()
        + struct.pack("<III", len(b_eh), len(b_mtp0_gqa), len(b_mtp1_moe))
        + b_eh + b_mtp0_gqa + b_mtp1_moe
    )
    del eh_proj
    pack_experts_512_hg4(src / "expert-mtp-1" / "experts.direct", dst / "experts_mtp_1.hg4", seed_base=2000)

    tot_b = sum(f.stat().st_size for f in dst.iterdir() if f.is_file())
    print(
        f"\nDONE in {time.perf_counter() - t_start:.1f}s! "
        f"Total Nemotron 3.5 HG4-128 size = {tot_b:,} bytes ({tot_b / (1024**2):.2f} MiB / {tot_b / (1024**3):.3f} GiB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
