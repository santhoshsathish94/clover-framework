#!/usr/bin/env python3
"""Full 40/40-Layer (`256/256` Experts/Layer) `HG4-16R + Shrinkage-LDLQ` Streaming Converter
for `Diffuspace/Clover-35B-A3B-Preview` -> `/opt/clover-k3/clover35b-full-hg4` (~17.3 GiB).

Streams exact per-layer tensors directly into RAM via 16-worker coalesced HTTP Range requests
(zero temporary shard files on disk), runs sequential 2,048-token multi-domain calibration +
Opus Shrinkage-LDLQ across all 40 layers, and writes canonical 16-row `HG4-16R` / `HG8-16R` files.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import ctypes
import gc
import json
import math
import os
import re
import shutil
import subprocess
import struct
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, hf_hub_url
from tokenizers import Tokenizer

torch.set_num_threads(16)

for p in (
    Path("/opt/clover-k3/clover-framework/porting-models"),
    Path(__file__).resolve().parents[4] / "porting-models" if len(Path(__file__).resolve().parents) >= 5 else Path("/nonexistent"),
):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import clover_35b_gpu_backbone as cb  # noqa: E402

LUT16_U8 = np.array([1, 32, 53, 70, 84, 97, 110, 122, 134, 146, 159, 172, 186, 203, 224, 255], dtype=np.uint8)
LUT16_S8_F = (LUT16_U8.astype(np.int32) - 128).astype(np.float32)
CODE_MIDS = 0.5 * (LUT16_S8_F[:-1] + LUT16_S8_F[1:])
FULL_ATTN_SET = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39}


def auto_load_hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    for cand in (
        Path.home() / ".cache/huggingface/token",
        Path("/opt/clover-k3/upload_to_hf.py"),
        Path("/opt/clover-k3/upload_computed_to_hf.py"),
    ):
        if cand.is_file():
            txt = cand.read_text(errors="ignore")
            m = re.search(r"(hf_[A-Za-z0-9]{16,})", txt)
            if m:
                os.environ["HF_TOKEN"] = m.group(1)
                return m.group(1)
    return None


_FAST_LIB = None


def get_fast_packer_lib():
    global _FAST_LIB
    if _FAST_LIB is not None:
        return _FAST_LIB
    so_path = Path("/tmp/libhg4_fast_packer.so")
    c_cands = [
        Path(__file__).resolve().parent / "hg4_fast_packer.c",
        Path("/tmp/hg4_fast_packer.c"),
    ]
    c_src = next((p for p in c_cands if p.is_file()), None)
    if not so_path.is_file() and c_src is not None:
        subprocess.run(
            ["gcc", "-O3", "-fopenmp", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror", str(c_src), "-o", str(so_path), "-lm"],
            check=True,
        )
    if so_path.is_file():
        lib = ctypes.CDLL(str(so_path))
        fp = ctypes.POINTER(ctypes.c_float)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        lib.hg4_pack_16r_f32.argtypes = [fp, fp, fp, ctypes.c_int, ctypes.c_int, u16p, u8p, fp]
        lib.hg4_pack_16r_f32.restype = None
        lib.hg4_pack_experts_16r_f32.argtypes = [
            fp, fp, ctypes.c_int, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, u16p, u8p, fp
        ]
        lib.hg4_pack_experts_16r_f32.restype = None
        _FAST_LIB = lib
    return _FAST_LIB


_HEADER_CACHE: dict[tuple[str, str], tuple[str, dict[str, Any], int, float]] = {}


def get_safetensor_header_cached(repo_id: str, filename: str) -> tuple[str, dict[str, Any], int]:
    now = time.time()
    key = (repo_id, filename)
    if key in _HEADER_CACHE and (now - _HEADER_CACHE[key][3]) < 600.0:
        c_url, hdr, base, _ = _HEADER_CACHE[key]
        return c_url, hdr, base
    cb.enable_ipv4_only()
    url = hf_hub_url(repo_id, filename)
    tok = os.environ.get("HF_TOKEN")
    auth_hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={**auth_hdr, "Range": "bytes=0-7"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                cdn_url = resp.geturl()
                hlen = struct.unpack("<Q", resp.read(8))[0]
            req2 = urllib.request.Request(cdn_url, headers={"Range": f"bytes=8-{8 + hlen - 1}"})
            with urllib.request.urlopen(req2, timeout=35) as resp:
                header = json.loads(resp.read().decode("utf-8"))
            _HEADER_CACHE[key] = (cdn_url, header, hlen + 8, time.time())
            return cdn_url, header, hlen + 8
        except Exception:
            if attempt == 5:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Unreachable")


def fetch_tensors_ram(
    repo_id: str,
    weight_map: dict[str, str],
    keys: list[str],
    max_workers: int = 16,
) -> dict[str, torch.Tensor]:
    """Stream exact tensors across 1+ remote safetensor shards directly into CPU RAM (0 disk bytes)."""
    by_shard: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by_shard[weight_map[k]].append(k)

    out: dict[str, torch.Tensor] = {}
    for shard, skeys in by_shard.items():
        cdn_url, header, data_base = get_safetensor_header_cached(repo_id, shard)
        items = []
        for name in skeys:
            info = header[name]
            s, e = int(info["data_offsets"][0]), int(info["data_offsets"][1])
            items.append((s, e, name, info["dtype"], tuple(info["shape"])))
        items.sort(key=lambda x: x[0])

        spans: list[list[Any]] = []
        for it in items:
            if not spans or (it[0] - spans[-1][1] > 4 * 1024 * 1024):
                spans.append([it[0], it[1], [it]])
            else:
                spans[-1][1] = max(spans[-1][1], it[1])
                spans[-1][2].append(it)

        chunk_bytes = 32 * 1024 * 1024
        subtasks = []
        for sp_idx, (sp_s, sp_e, _) in enumerate(spans):
            cur = sp_s
            while cur < sp_e:
                nxt = min(sp_e, cur + chunk_bytes)
                subtasks.append((sp_idx, cur - sp_s, data_base + cur, data_base + nxt - 1))
                cur = nxt

        span_bufs = [bytearray(sp[1] - sp[0]) for sp in spans]
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(subtasks)))) as pool:
            futs = [(si, ro, pool.submit(cb._fetch_byte_range, cdn_url, bs, be)) for si, ro, bs, be in subtasks]
            for si, ro, fut in futs:
                data = fut.result()
                span_bufs[si][ro : ro + len(data)] = data

        for sp_idx, (sp_s, _, sp_items) in enumerate(spans):
            mv = memoryview(span_bufs[sp_idx])
            for s, e, name, dtype_s, shape in sp_items:
                sl = mv[s - sp_s : e - sp_s]
                if dtype_s == "BF16":
                    arr = np.frombuffer(sl, dtype=np.uint16).copy()
                    out[name] = torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
                elif dtype_s == "F32":
                    arr = np.frombuffer(sl, dtype=np.float32).copy()
                    out[name] = torch.from_numpy(arr).reshape(shape)
                elif dtype_s == "F16":
                    arr = np.frombuffer(sl, dtype=np.float16).copy()
                    out[name] = torch.from_numpy(arr).reshape(shape)
                else:
                    arr = np.frombuffer(sl, dtype=np.uint8).copy()
                    out[name] = torch.from_numpy(arr).reshape(shape)
    return out


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


def make_signs(K: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.where(torch.randint(0, 2, (K,), generator=g) == 0, -1.0, 1.0).to(torch.float32)


def interleave_swiglu_16r(gate_w: torch.Tensor, up_w: torch.Tensor) -> torch.Tensor:
    rh, K = gate_w.shape
    assert rh % 8 == 0
    return torch.cat([gate_w.reshape(rh // 8, 8, K), up_w.reshape(rh // 8, 8, K)], dim=1).reshape(2 * rh, K).contiguous()


def swiglu_eq_pre_diag(gate_w: torch.Tensor, up_w: torch.Tensor, dn_w: torch.Tensor, seed: int) -> torch.Tensor:
    dn_crms = dn_w.pow(2).mean(dim=0).clamp_min(1e-12).sqrt()
    gu_rrms = gate_w.pow(2).mean(dim=1).clamp_min(1e-12).sqrt() * up_w.pow(2).mean(dim=1).clamp_min(1e-12).sqrt()
    return ((dn_crms / gu_rrms.clamp_min(1e-12)).pow(0.5) * make_signs(dn_w.shape[1], seed)).contiguous()


def guard_fp16_scales(sc_f32: torch.Tensor) -> torch.Tensor:
    nonzero = sc_f32.abs() > 0.0
    sc_safe = torch.where(nonzero & (sc_f32.abs() < 6.11e-5), torch.sign(sc_f32) * 6.11e-5, sc_f32)
    return sc_safe.to(torch.float16)


def ldlq_feedback_factors(H_blocks: torch.Tensor, damp_rel: float = 0.015) -> torch.Tensor:
    B = H_blocks.shape[-1]
    eye = torch.eye(B, dtype=torch.float64, device=H_blocks.device)
    H64 = H_blocks.to(torch.float64)
    diag_mean = H64.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-10).unsqueeze(-1)
    H64 = 0.5 * (H64 + H64.transpose(-1, -2)) + (damp_rel * diag_mean) * eye
    L_flip = torch.linalg.cholesky(torch.flip(H64, dims=[-2, -1]))
    Linv_flip = torch.linalg.solve_triangular(L_flip, eye.expand_as(L_flip), upper=False)
    U_inv = torch.flip(Linv_flip, dims=[-2, -1]).transpose(-1, -2)
    d = U_inv.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).unsqueeze(-1)
    return (-U_inv / d).to(torch.float32).contiguous()


def quantize_rot_blocks_hg4(
    w_rot: torch.Tensor,
    F_ldlq: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mids_t = torch.from_numpy(CODE_MIDS)
    lut_t = torch.from_numpy(LUT16_S8_F)

    std_b = torch.sqrt(w_rot.pow(2).mean(dim=-1, keepdim=True) + 1e-12) / (127.0 / 2.7326)
    best_sc = std_b.clone()
    best_mse = torch.full_like(std_b, 1e30)
    for mult in (0.86, 0.93, 1.00, 1.07, 1.14):
        cand = guard_fp16_scales(std_b * mult).float()
        idx = torch.bucketize(w_rot / cand.clamp_min(1e-12), mids_t)
        qv = lut_t[idx]
        opt = guard_fp16_scales((w_rot * qv).sum(dim=-1, keepdim=True) / (qv * qv).sum(dim=-1, keepdim=True).clamp_min(1e-12)).float()
        mse = (w_rot - qv * opt).pow(2).sum(dim=-1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_sc = torch.where(better, opt, best_sc)

    sc_fp16 = guard_fp16_scales(best_sc.squeeze(-1))
    sc_f32 = sc_fp16.float().unsqueeze(-1).clamp_min(1e-12)

    if F_ldlq is None:
        idx_out = torch.bucketize(w_rot / sc_f32, mids_t).to(torch.uint8)
        return sc_fp16, idx_out, lut_t[idx_out.long()] * sc_f32

    work = w_rot.clone()
    idx_out = torch.empty(w_rot.shape, dtype=torch.uint8)
    inv_sc = 1.0 / sc_f32.squeeze(-1)
    sc_2d = sc_f32.squeeze(-1)
    for k in range(128):
        wk = work[:, :, k]
        ik = torch.bucketize(wk * inv_sc, mids_t)
        idx_out[:, :, k] = ik.to(torch.uint8)
        qk = lut_t[ik] * sc_2d
        if k + 1 < 128:
            err = (wk - qk).clamp(-1.05 * 2.7326 * sc_2d, 1.05 * 2.7326 * sc_2d)
            work[:, :, k + 1:] += err.unsqueeze(-1) * F_ldlq[:, k, k + 1:].unsqueeze(1)
    return sc_fp16, idx_out, lut_t[idx_out.long()] * sc_f32


def pack_hg4_16r_blob(
    W_f32: torch.Tensor,
    seed: int = 42,
    fixed_pre_diag: torch.Tensor | None = None,
    H_blocks: torch.Tensor | None = None,
    return_dequant: bool = False,
) -> bytes | tuple[bytes, torch.Tensor]:
    R, K = W_f32.shape
    assert R % 16 == 0 and K % 128 == 0
    nb = K // 128
    if fixed_pre_diag is not None:
        pre_diag = fixed_pre_diag.to(torch.float32).contiguous()
    else:
        col_rms = torch.sqrt(torch.mean(W_f32 * W_f32, dim=0).clamp_min(1e-12)).pow(0.5)
        pre_diag = (col_rms * make_signs(K, seed)).contiguous()

    F_ldlq = ldlq_feedback_factors(H_blocks).contiguous() if H_blocks is not None else None
    fast_lib = get_fast_packer_lib()
    if fast_lib is not None:
        wc = W_f32.to(torch.float32).contiguous()
        out_sc = np.empty((R // 16, nb, 16), dtype=np.uint16)
        out_pk = np.empty((R * K) // 2, dtype=np.uint8)
        out_dq = torch.empty((R, K), dtype=torch.float32) if return_dequant else None
        fp = ctypes.POINTER(ctypes.c_float)
        fast_lib.hg4_pack_16r_f32(
            wc.data_ptr() and ctypes.cast(wc.data_ptr(), fp),
            ctypes.cast(pre_diag.data_ptr(), fp),
            ctypes.cast(F_ldlq.data_ptr(), fp) if F_ldlq is not None else None,
            int(R), int(K),
            out_sc.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            out_pk.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(out_dq.data_ptr(), fp) if out_dq is not None else None,
        )
        blob = struct.pack("<II", R, K) + pre_diag.numpy().tobytes() + out_sc.tobytes() + out_pk.tobytes()
        return (blob, out_dq) if return_dequant else blob

    sc_chunks, pk_chunks = [], []
    dq_chunks = [] if return_dequant else None

    for r0 in range(0, R, 16384):
        r1 = min(R, r0 + 16384)
        rc = r1 - r0
        wc = (W_f32[r0:r1] / pre_diag.unsqueeze(0)).reshape(rc, nb, 128).permute(1, 0, 2).contiguous()
        w_rot = fwht128(wc)
        sc_b_r, idx_b_r, wq_rot = quantize_rot_blocks_hg4(w_rot, F_ldlq)
        sc_rn = sc_b_r.permute(1, 0).contiguous()
        idx_rnk = idx_b_r.permute(1, 0, 2).contiguous()

        sc_16r = sc_rn.reshape(rc // 16, 16, nb).permute(0, 2, 1).contiguous()
        idx_6d = idx_rnk.reshape(rc // 16, 16, nb, 16, 2, 4)
        pk_16r = (idx_6d[:, :, :, :, 0, :] | (idx_6d[:, :, :, :, 1, :] << 4)).permute(0, 2, 3, 1, 4).contiguous()
        sc_chunks.append(sc_16r.view(torch.uint16).numpy().tobytes())
        pk_chunks.append(pk_16r.numpy().tobytes())
        if return_dequant:
            dq_chunks.append(fwht128(wq_rot.permute(1, 0, 2).contiguous()).reshape(rc, K) * pre_diag.unsqueeze(0))

    blob = struct.pack("<II", R, K) + pre_diag.numpy().tobytes() + b"".join(sc_chunks) + b"".join(pk_chunks)
    if return_dequant:
        return blob, torch.cat(dq_chunks, dim=0)
    return blob


def pack_hg8_16r_blob(
    W_f32: torch.Tensor,
    seed: int = 42,
    fixed_pre_diag: torch.Tensor | None = None,
    return_dequant: bool = False,
) -> bytes | tuple[bytes, torch.Tensor]:
    R, K = W_f32.shape
    assert R % 16 == 0 and K % 128 == 0
    nb = K // 128
    if fixed_pre_diag is not None:
        pre_diag = fixed_pre_diag.to(torch.float32).contiguous()
    else:
        col_rms = torch.sqrt(torch.mean(W_f32 * W_f32, dim=0).clamp_min(1e-12)).pow(0.5)
        pre_diag = (col_rms * make_signs(K, seed)).contiguous()

    sc_chunks, pk_chunks = [], []
    dq_chunks = [] if return_dequant else None
    for r0 in range(0, R, 16384):
        r1 = min(R, r0 + 16384)
        rc = r1 - r0
        wc = W_f32[r0:r1] / pre_diag.unsqueeze(0)
        w_rot = fwht128(wc.reshape(rc, nb, 128))
        sc_fp16 = guard_fp16_scales(w_rot.abs().amax(dim=-1) / 127.0)
        sc_f32 = sc_fp16.float().unsqueeze(-1).clamp_min(1e-12)
        q_s8 = torch.round(w_rot / sc_f32).clamp(-127.0, 127.0)
        q_u8 = (q_s8 + 128.0).to(torch.uint8)

        sc_16r = sc_fp16.reshape(rc // 16, 16, nb).permute(0, 2, 1).contiguous()
        pk_16r = q_u8.reshape(rc // 16, 16, nb, 32, 4).permute(0, 2, 3, 1, 4).contiguous()
        sc_chunks.append(sc_16r.view(torch.uint16).numpy().tobytes())
        pk_chunks.append(pk_16r.numpy().tobytes())
        if return_dequant:
            dq_chunks.append(fwht128(q_s8 * sc_f32).reshape(rc, K) * pre_diag.unsqueeze(0))

    blob = struct.pack("<II", R, K | (8 << 28)) + pre_diag.numpy().tobytes() + b"".join(sc_chunks) + b"".join(pk_chunks)
    if return_dequant:
        return blob, torch.cat(dq_chunks, dim=0)
    return blob


def build_calibration_ids(tok: Tokenizer, n_seqs: int = 16, seq_len: int = 128) -> torch.Tensor:
    base_texts = [
        "Fast Walsh-Hadamard orthogonal rotations disperse activation and weight outliers uniformly across 128-element blocks, enabling 4-bit Gaussian Lloyd-Max codebooks with Cholesky LDLQ error feedback to match 16-bit BF16 perplexity.",
        "def cholesky_ldlq_quantize(W: torch.Tensor, H: torch.Tensor, damp: float = 0.015) -> torch.Tensor:\n    n = H.shape[-1]\n    H_reg = H + damp * torch.diagonal(H).mean() * torch.eye(n)\n    L = torch.linalg.cholesky(torch.flip(H_reg, [-2, -1]))\n    return L\n",
        "Theorem (Cauchy-Schwarz & Parseval Identity): For any orthogonal matrix Q satisfying Q^T Q = I and any vectors w, x in R^d, the inner product <Qw, Qx> equals <w, x> identically, preserving Euclidean norms and second-order Hessian trace.",
        "<|im_start|>user\nCalculate the total energy produced by 240 solar panels rated at 550W operating for 6.5 peak sun hours, accounting for an 8% inverter loss.<|im_end|>\n<|im_start|>assistant\nTotal DC energy = 240 * 0.55 kW * 6.5 h = 858.0 kWh. After 8% loss: 858.0 * 0.92 = 789.36 kWh.<|im_end|>",
        "static inline void avx512_vnni_dpbusd(__m512i *acc, __m512i w_u8, __m512i x_s8) {\n    *acc = _mm512_dpbusd_epi32(*acc, w_u8, x_s8);\n}\nIn modern superscalar CPUs, memory bandwidth across DDR5 channels bounds single-token autoregressive decoding.",
        "Mixture-of-Experts transformers route each token representation to the top-k specialized SwiGLU feed-forward networks while maintaining a gated shared expert for common syntactic and factual transformations across domains.",
        "SELECT d.department_name, COUNT(e.id) AS headcount, AVG(e.salary) AS mean_pay FROM employees e JOIN departments d ON e.dept_id = d.id WHERE e.active = 1 GROUP BY d.department_name ORDER BY mean_pay DESC LIMIT 10;",
        "In numerical linear algebra, shrinkage estimation S_shrunk = (N * S_emp + alpha * S_prior) / (N + alpha) guarantees strict positive-definiteness and minimizes Stein's Frobenius risk when sample size N is comparable to dimension p.",
    ]
    joined = "\n\n".join(base_texts * 12)
    all_ids = tok.encode(joined, add_special_tokens=False).ids
    need = n_seqs * seq_len
    while len(all_ids) < need + 64:
        all_ids = all_ids + all_ids
    stride = max(1, (len(all_ids) - seq_len) // n_seqs)
    seqs = [all_ids[i * stride : i * stride + seq_len] for i in range(n_seqs)]
    return torch.tensor(seqs, dtype=torch.long)


def pack_all_256_experts_ldlq(
    gu_bf16: torch.Tensor,
    dn_bf16: torch.Tensor,
    u_mlp_flat: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    out_path: Path,
    seed_base: int,
) -> torch.Tensor:
    k_exp = gu_bf16.shape[0]
    assert k_exp == 256
    N_cal = u_mlp_flat.shape[0]
    nb_up = 16

    col_rms_up = torch.zeros(2048, dtype=torch.float64)
    for e0 in range(0, 256, 32):
        col_rms_up += gu_bf16[e0:e0 + 32].float().pow(2).sum(dim=(0, 1)).to(torch.float64)
    col_rms_up = (col_rms_up / (256.0 * 1024.0)).clamp_min(1e-12).sqrt().pow(0.5).to(torch.float32)
    pre_diag_up = (col_rms_up * make_signs(2048, seed_base)).contiguous()

    x_rot = fwht128((u_mlp_flat.float() * pre_diag_up.unsqueeze(0)).reshape(N_cal, nb_up, 128))
    H_layer = torch.einsum("nbi,nbj->bij", x_rot, x_rot) / float(N_cal)
    alpha_shrink = 128.0

    moe_out = torch.zeros((N_cal, 2048), dtype=torch.float32)
    exp_stride = 1081344 + 542728

    with open(out_path, "wb") as wf:
        wf.write(struct.pack("<4sIIIIII", b"C35F", k_exp, 1024, 2048, 2048, 512, exp_stride))
        wf.write(pre_diag_up.numpy().tobytes())

        for e0 in range(0, 256, 32):
            e1 = e0 + 32
            B_E = e1 - e0
            H_batch = torch.empty((B_E, nb_up, 128, 128), dtype=torch.float32)
            routed_tokens: list[tuple[torch.Tensor, torch.Tensor]] = []

            for bi, e in enumerate(range(e0, e1)):
                tok_idx, slot_idx = torch.where(topk_ids == e)
                n_e = int(tok_idx.numel())
                routed_tokens.append((tok_idx, topk_w[tok_idx, slot_idx]))
                if n_e >= 8:
                    xe = x_rot[tok_idx]
                    sum_H = torch.einsum("nbi,nbj->bij", xe, xe)
                    H_batch[bi] = (sum_H + alpha_shrink * H_layer) / (float(n_e) + alpha_shrink)
                else:
                    H_batch[bi] = H_layer

            F_up_batch = ldlq_feedback_factors(H_batch.reshape(B_E * nb_up, 128, 128)).reshape(B_E, nb_up, 128, 128).contiguous()

            gu_sub = gu_bf16[e0:e1].float()
            gate_s, up_s = gu_sub[:, :512, :], gu_sub[:, 512:, :]
            w13_s = torch.cat([gate_s.reshape(B_E, 64, 8, 2048), up_s.reshape(B_E, 64, 8, 2048)], dim=2).reshape(B_E, 1024, 2048).contiguous()
            dn_sub = dn_bf16[e0:e1].float().contiguous()
            pd_dn_batch = torch.stack([swiglu_eq_pre_diag(gate_s[bi], up_s[bi], dn_sub[bi], seed=seed_base + 1 + e0 + bi) for bi in range(B_E)], dim=0).contiguous()

            fast_lib = get_fast_packer_lib()
            assert fast_lib is not None, "libhg4_fast_packer.so required"
            fp = ctypes.POINTER(ctypes.c_float)

            out_sc_u = np.empty((B_E, 64 * 16 * 16), dtype=np.uint16)
            out_pk_u = np.empty((B_E, 1024 * 1024), dtype=np.uint8)
            dq_up = torch.empty((B_E, 1024, 2048), dtype=torch.float32)
            fast_lib.hg4_pack_experts_16r_f32(
                ctypes.cast(w13_s.data_ptr(), fp),
                ctypes.cast(pre_diag_up.data_ptr(), fp),
                0,
                ctypes.cast(F_up_batch.data_ptr(), fp),
                B_E, 1024, 2048,
                out_sc_u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out_pk_u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.cast(dq_up.data_ptr(), fp),
            )

            eye4 = torch.eye(128, dtype=torch.float32).unsqueeze(0).expand(4, 128, 128)
            H_dn_batch = torch.empty((B_E, 4, 128, 128), dtype=torch.float32)
            z_acts: list[torch.Tensor | None] = [None] * B_E

            for bi in range(B_E):
                tok_idx, _ = routed_tokens[bi]
                if tok_idx.numel() > 0:
                    gu_out = (u_mlp_flat[tok_idx] @ dq_up[bi].T).reshape(-1, 64, 16)
                    z_act = F.silu(gu_out[:, :, :8].reshape(-1, 512)) * gu_out[:, :, 8:].reshape(-1, 512)
                    z_acts[bi] = z_act
                    if tok_idx.numel() >= 16:
                        z_rot = fwht128((z_act * pd_dn_batch[bi].unsqueeze(0)).reshape(-1, 4, 128))
                        n_e = float(tok_idx.numel())
                        emp_H = torch.einsum("nbi,nbj->bij", z_rot, z_rot)
                        diag_ref = (emp_H.diagonal(dim1=-2, dim2=-1).mean() / n_e).clamp_min(1e-8)
                        H_dn_batch[bi] = (emp_H + 128.0 * diag_ref * eye4) / (n_e + 128.0)
                    else:
                        H_dn_batch[bi] = eye4
                else:
                    H_dn_batch[bi] = eye4

            F_dn_batch = ldlq_feedback_factors(H_dn_batch.reshape(B_E * 4, 128, 128)).reshape(B_E, 4, 128, 128).contiguous()
            out_sc_d = np.empty((B_E, 128 * 4 * 16), dtype=np.uint16)
            out_pk_d = np.empty((B_E, 2048 * 256), dtype=np.uint8)
            dq_dn = torch.empty((B_E, 2048, 512), dtype=torch.float32)
            fast_lib.hg4_pack_experts_16r_f32(
                ctypes.cast(dn_sub.data_ptr(), fp),
                ctypes.cast(pd_dn_batch.data_ptr(), fp),
                1,
                ctypes.cast(F_dn_batch.data_ptr(), fp),
                B_E, 2048, 512,
                out_sc_d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out_pk_d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.cast(dq_dn.data_ptr(), fp),
            )

            for bi in range(B_E):
                gu_body = out_sc_u[bi].tobytes() + out_pk_u[bi].tobytes()
                dn_blob = (
                    struct.pack("<II", 2048, 512)
                    + pd_dn_batch[bi].numpy().tobytes()
                    + out_sc_d[bi].tobytes()
                    + out_pk_d[bi].tobytes()
                )
                assert len(gu_body) + len(dn_blob) == exp_stride
                wf.write(gu_body)
                wf.write(dn_blob)

                tok_idx, rw = routed_tokens[bi]
                z_act = z_acts[bi]
                if z_act is not None and tok_idx.numel() > 0:
                    ey = z_act @ dq_dn[bi].T
                    moe_out.index_add_(0, tok_idx, ey * rw.unsqueeze(-1))

    return moe_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/opt/clover-k3/clover35b-full-hg4")
    ap.add_argument("--max-layers", type=int, default=40)
    ap.add_argument("--calib-seqs", type=int, default=16)
    ap.add_argument("--calib-len", type=int, default=128)
    args = ap.parse_args()

    auto_load_hf_token()
    cb.enable_ipv4_only()
    t0 = time.perf_counter()

    repo_id = "Diffuspace/Clover-35B-A3B-Preview"
    stage_trunks = Path("/opt/clover-k3/clover-35b-computed/hf-stage/trunks")
    pruned_dir = Path("/opt/clover-k3/clover-35b-computed/pruned")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_path = hf_hub_download(repo_id, "model.safetensors.index.json", token=os.environ.get("HF_TOKEN"))
    weight_map: dict[str, str] = json.loads(Path(idx_path).read_text())["weight_map"]
    tok = Tokenizer.from_file(str(stage_trunks / "tokenizer.json"))
    calib_ids = build_calibration_ids(tok, n_seqs=args.calib_seqs, seq_len=args.calib_len)
    B_CAL, T_CAL = calib_ids.shape
    N_CAL = B_CAL * T_CAL
    print(f"Calibration batch: {B_CAL} x {T_CAL} = {N_CAL} tokens", flush=True)

    norm_path = stage_trunks / "global-0" / "norm.bf16.bin"
    if norm_path.is_file():
        norm_raw = cb.bytes_to_tensor(norm_path.read_bytes(), cb.DTYPE_BF16, (2048,)).float()
    else:
        norm_raw = fetch_tensors_ram(repo_id, weight_map, ["model.norm.weight"])["model.norm.weight"].float()
    norm_w = (1.0 + norm_raw).contiguous()
    inv_freq = (1.0 / (10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64.0))).contiguous()
    kinds = [1 if l in FULL_ATTN_SET else 0 for l in range(args.max_layers)]

    (out_dir / "meta.bin").write_bytes(
        struct.pack("<4sIII", b"C35R", 2, args.max_layers, 248320)
        + LUT16_U8.tobytes()
        + norm_w.numpy().tobytes()
        + inv_freq.numpy().tobytes()
        + struct.pack(f"<{args.max_layers}I", *kinds)
    )

    emb_p = out_dir / "embed_tokens.hg4"
    emb_bf16_path = stage_trunks / "global-0" / "embed_tokens.bf16.bin"
    if emb_bf16_path.is_file():
        emb_mm = np.memmap(emb_bf16_path, dtype=np.uint16, mode="r").reshape(248320, 2048)
        emb_f32 = torch.from_numpy(np.left_shift(np.asarray(emb_mm, dtype=np.uint32), np.uint32(16)).view(np.float32))
    else:
        emb_f32 = fetch_tensors_ram(repo_id, weight_map, ["model.embed_tokens.weight"])["model.embed_tokens.weight"].float()

    if not emb_p.is_file():
        t_e = time.perf_counter()
        emb_blob = pack_hg4_16r_blob(emb_f32, seed=101, H_blocks=None)
        emb_p.write_bytes(emb_blob)
        print(f"[Global] embed_tokens.hg4: {len(emb_blob)/1048576:.2f} MiB in {time.perf_counter()-t_e:.1f}s", flush=True)

    h_cal = emb_f32[calib_ids].clone().to(torch.float32)
    del emb_f32
    gc.collect()

    pos_t = torch.arange(T_CAL, dtype=torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    rope_cos, rope_sin = torch.cos(pos_t), torch.sin(pos_t)

    def keys_for_layer(lid: int) -> list[str]:
        lp = f"model.layers.{lid}."
        bd_loc = pruned_dir / f"trunk-{lid}" / "bd_gdn.bin"
        return [
            k for k in weight_map
            if k.startswith(lp) and not (bd_loc.is_file() and k.endswith(("conv_right_weight", "bd_softmax_logit_scale", "bd_softmax_history_bias", "bd_softmax_history_c")))
        ]

    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    next_fut = prefetch_pool.submit(fetch_tensors_ram, repo_id, weight_map, keys_for_layer(0), 16) if args.max_layers > 0 else None

    for l in range(args.max_layers):
        tl0 = time.perf_counter()
        pfx = f"model.layers.{l}."
        is_gqa = l in FULL_ATTN_SET
        bd_local = pruned_dir / f"trunk-{l}" / "bd_gdn.bin"

        assert next_fut is not None
        td = next_fut.result()
        t_dl = time.perf_counter() - tl0
        next_fut = (
            prefetch_pool.submit(fetch_tensors_ram, repo_id, weight_map, keys_for_layer(l + 1), 16)
            if (l + 1 < args.max_layers) else None
        )

        in_ln_raw = td.pop(pfx + "input_layernorm.weight").float()
        post_ln_raw = td.pop(pfx + "post_attention_layernorm.weight").float()
        in_ln = (1.0 + in_ln_raw).contiguous()
        post_ln = (1.0 + post_ln_raw).contiguous()

        u = cb.clover_rms_norm_op(h_cal, in_ln_raw, eps=1e-6, zero_centered=True)
        u_flat = u.reshape(N_CAL, 2048)

        if not is_gqa:
            apfx = pfx + "linear_attn."
            qkv_w = td.pop(apfx + "in_proj_qkv.weight").float()
            z_w = td.pop(apfx + "in_proj_z.weight").float()
            out_w = td.pop(apfx + "out_proj.weight").float()
            conv_w = td.pop(apfx + "conv1d.weight").float().reshape(8192, 1, 4)
            if bd_local.is_file():
                bd_raw = bd_local.read_bytes()
                _, _, _, l_crw, l_ls, l_hb, l_hc = struct.unpack_from("<8sIIIIII", bd_raw, 0)
                off_b = 64
                conv_rw = cb.bytes_to_tensor(bd_raw[off_b:off_b + l_crw], cb.DTYPE_BF16, (8192, 1, 3)).float(); off_b += l_crw
                ls_v = cb.bytes_to_tensor(bd_raw[off_b:off_b + l_ls], cb.DTYPE_FP32, (32,)).float(); off_b += l_ls
                hb_v = cb.bytes_to_tensor(bd_raw[off_b:off_b + l_hb], cb.DTYPE_FP32, (32,)).float(); off_b += l_hb
                hc_v = cb.bytes_to_tensor(bd_raw[off_b:off_b + l_hc], cb.DTYPE_FP32, (32,)).float()
            else:
                conv_rw = td.pop(apfx + "conv_right_weight").float().reshape(8192, 1, 3)
                ls_v = td.pop(apfx + "bd_softmax_logit_scale").float().reshape(32)
                hb_v = td.pop(apfx + "bd_softmax_history_bias").float().reshape(32)
                hc_v = td.pop(apfx + "bd_softmax_history_c").float().reshape(32)

            a_w = td.pop(apfx + "in_proj_a.weight").float().contiguous()
            b_w = td.pop(apfx + "in_proj_b.weight").float().contiguous()
            neg_exp_A = (-td.pop(apfx + "A_log").float().exp()).contiguous()
            dt_bias = td.pop(apfx + "dt_bias").float().contiguous()
            gdn_norm = td.pop(apfx + "norm.weight").float().contiguous()

            b_qkv, qkv_dq = pack_hg8_16r_blob(qkv_w, seed=300 + l * 10 + 1, return_dequant=True)
            b_z, z_dq = pack_hg8_16r_blob(z_w, seed=300 + l * 10 + 2, return_dequant=True)

            raw_qkv = (u_flat @ qkv_dq.T).reshape(B_CAL, T_CAL, 8192)
            z_act = (u_flat @ z_dq.T).reshape(B_CAL, T_CAL, 32, 128)
            joined = F.pad(raw_qkv.transpose(1, 2), (3, 0))
            mixed = cb.causal_conv_silu_op(joined, conv_w).transpose(1, 2)
            q_t = mixed[:, :, :2048].reshape(B_CAL, T_CAL, 16, 128)
            k_t = mixed[:, :, 2048:4096].reshape(B_CAL, T_CAL, 16, 128)
            v_t = mixed[:, :, 4096:].reshape(B_CAL, T_CAL, 32, 128)
            g_val = neg_exp_A.view(1, 1, 32) * F.softplus((u_flat @ a_w.T).reshape(B_CAL, T_CAL, 32) + dt_bias.view(1, 1, 32))
            beta_val = torch.sigmoid((u_flat @ b_w.T).reshape(B_CAL, T_CAL, 32))
            y_rec, _ = cb.recurrent_commit_op(q_t, k_t, v_t, g_val, beta_val)
            y_norm = cb.clover_rms_norm_op(y_rec, gdn_norm, eps=1e-6, zero_centered=False, gate=z_act).reshape(N_CAL, 4096)

            b_out, out_dq = pack_hg8_16r_blob(out_w, seed=300 + l * 10 + 3, return_dequant=True)
            op_out = (y_norm @ out_dq.T).reshape(B_CAL, T_CAL, 2048)
            del qkv_w, z_w, out_w, qkv_dq, z_dq, out_dq
        else:
            apfx = pfx + "self_attn."
            q_w = td.pop(apfx + "q_proj.weight").float()
            k_w = td.pop(apfx + "k_proj.weight").float()
            v_w = td.pop(apfx + "v_proj.weight").float()
            o_w = td.pop(apfx + "o_proj.weight").float()
            qn_raw = td.pop(apfx + "q_norm.weight").float()
            kn_raw = td.pop(apfx + "k_norm.weight").float()
            qn, kn = (1.0 + qn_raw).contiguous(), (1.0 + kn_raw).contiguous()

            b_q, q_dq = pack_hg8_16r_blob(q_w, seed=400 + l * 10 + 1, return_dequant=True)
            b_k, k_dq = pack_hg8_16r_blob(k_w, seed=400 + l * 10 + 2, return_dequant=True)
            b_v, v_dq = pack_hg8_16r_blob(v_w, seed=400 + l * 10 + 3, return_dequant=True)

            qg = (u_flat @ q_dq.T).reshape(B_CAL, T_CAL, 16, 512)
            qh = cb.clover_rms_norm_op(qg[..., :256], qn_raw, eps=1e-6, zero_centered=True)
            gh = qg[..., 256:]
            kh = cb.clover_rms_norm_op((u_flat @ k_dq.T).reshape(B_CAL, T_CAL, 2, 256), kn_raw, eps=1e-6, zero_centered=True)
            vh = (u_flat @ v_dq.T).reshape(B_CAL, T_CAL, 2, 256)

            rc, rs = rope_cos.view(1, T_CAL, 1, 32), rope_sin.view(1, T_CAL, 1, 32)
            q1, q2 = qh[..., :32], qh[..., 32:64]
            qh = torch.cat([q1 * rc - q2 * rs, q2 * rc + q1 * rs, qh[..., 64:]], dim=-1)
            k1, k2 = kh[..., :32], kh[..., 32:64]
            kh = torch.cat([k1 * rc - k2 * rs, k2 * rc + k1 * rs, kh[..., 64:]], dim=-1)

            attn_y = F.scaled_dot_product_attention(
                qh.transpose(1, 2), kh.repeat_interleave(8, dim=2).transpose(1, 2),
                vh.repeat_interleave(8, dim=2).transpose(1, 2), is_causal=True
            ).transpose(1, 2)
            attn_y = (attn_y * torch.sigmoid(gh)).reshape(N_CAL, 4096)

            b_o, o_dq = pack_hg8_16r_blob(o_w, seed=400 + l * 10 + 4, return_dequant=True)
            op_out = (attn_y @ o_dq.T).reshape(B_CAL, T_CAL, 2048)
            del q_w, k_w, v_w, o_w, q_dq, k_dq, v_dq, o_dq

        h_attn = h_cal + op_out
        u_mlp_flat = cb.clover_rms_norm_op(h_attn, post_ln_raw, eps=1e-6, zero_centered=True).reshape(N_CAL, 2048)

        mpfx = pfx + "mlp."
        gate_w = td.pop(mpfx + "gate.weight").float().contiguous()
        sh_gate = td.pop(mpfx + "shared_expert.gate_proj.weight").float()
        sh_up = td.pop(mpfx + "shared_expert.up_proj.weight").float()
        sh_dn = td.pop(mpfx + "shared_expert.down_proj.weight").float()
        sh_g = td.pop(mpfx + "shared_expert_gate.weight").float().reshape(2048).contiguous()

        sh_w13 = interleave_swiglu_16r(sh_gate, sh_up)
        b_sh13, sh13_dq = pack_hg8_16r_blob(sh_w13, seed=500 + l * 10 + 1, return_dequant=True)
        pd_sdn = swiglu_eq_pre_diag(sh_gate, sh_up, sh_dn, seed=500 + l * 10 + 2)
        b_shdn, shdn_dq = pack_hg8_16r_blob(sh_dn, fixed_pre_diag=pd_sdn, return_dequant=True)
        b_moe = struct.pack("<III", 256, len(b_sh13), len(b_shdn)) + gate_w.numpy().tobytes() + sh_g.numpy().tobytes() + b_sh13 + b_shdn

        layer_path = out_dir / f"layer_{l:02d}.hg4"
        if not is_gqa:
            layer_path.write_bytes(
                struct.pack("<I", 0)
                + in_ln.numpy().tobytes() + post_ln.numpy().tobytes()
                + conv_w.reshape(-1).numpy().tobytes() + conv_rw.reshape(-1).numpy().tobytes()
                + a_w.numpy().tobytes() + b_w.numpy().tobytes()
                + neg_exp_A.numpy().tobytes() + dt_bias.numpy().tobytes() + gdn_norm.numpy().tobytes()
                + ls_v.numpy().tobytes() + hb_v.numpy().tobytes() + hc_v.numpy().tobytes()
                + struct.pack("<IIII", len(b_qkv), len(b_z), len(b_out), len(b_moe))
                + b_qkv + b_z + b_out + b_moe
            )
        else:
            layer_path.write_bytes(
                struct.pack("<I", 1)
                + in_ln.numpy().tobytes() + post_ln.numpy().tobytes()
                + qn.numpy().tobytes() + kn.numpy().tobytes()
                + struct.pack("<IIIII", len(b_q), len(b_k), len(b_v), len(b_o), len(b_moe))
                + b_q + b_k + b_v + b_o + b_moe
            )

        r_probs = F.softmax(u_mlp_flat @ gate_w.T, dim=-1)
        topk_w, topk_ids = torch.topk(r_probs, k=8, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        gu_bf16 = td.pop(mpfx + "experts.gate_up_proj")
        dn_bf16 = td.pop(mpfx + "experts.down_proj")
        td.clear()
        exp_path = out_dir / f"experts_{l:02d}.hg4"
        moe_routed = pack_all_256_experts_ldlq(
            gu_bf16, dn_bf16, u_mlp_flat, topk_ids, topk_w, exp_path, seed_base=10000 + l * 300
        )
        del gu_bf16, dn_bf16

        sh_gu = (u_mlp_flat @ sh13_dq.T).reshape(N_CAL, 64, 16)
        sh_hid = (F.silu(sh_gu[:, :, :8]) * sh_gu[:, :, 8:]).reshape(N_CAL, 512)
        sh_y = (sh_hid @ shdn_dq.T) * torch.sigmoid(u_mlp_flat @ sh_g).unsqueeze(-1)
        h_cal = h_attn + (moe_routed + sh_y).reshape(B_CAL, T_CAL, 2048)
        del sh_gate, sh_up, sh_dn, sh13_dq, shdn_dq
        gc.collect()

        dt_l = time.perf_counter() - tl0
        l_mb = (layer_path.stat().st_size + exp_path.stat().st_size) / 1048576.0
        print(
            f"[Layer {l:02d}/{args.max_layers - 1:02d} {'GQA  ' if is_gqa else 'BDGDN'}] "
            f"{l_mb:6.1f} MiB | dl={t_dl:4.1f}s tot={dt_l:5.1f}s (elapsed {(time.perf_counter()-t0)/60:.1f}m)",
            flush=True,
        )

    tl_lm = time.perf_counter()
    h_final = (h_cal.float() * torch.rsqrt(h_cal.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6) * norm_w.view(1, 1, 2048)).reshape(N_CAL, 2048)
    lm_bf16_path = stage_trunks / "global-0" / "lm_head.bf16.bin"
    if lm_bf16_path.is_file():
        lm_mm = np.memmap(lm_bf16_path, dtype=np.uint16, mode="r").reshape(248320, 2048)
        lm_f32 = torch.from_numpy(np.left_shift(np.asarray(lm_mm, dtype=np.uint32), np.uint32(16)).view(np.float32))
    else:
        lm_f32 = fetch_tensors_ram(repo_id, weight_map, ["lm_head.weight"])["lm_head.weight"].float()

    col_rms_lm = torch.sqrt(torch.mean(lm_f32 * lm_f32, dim=0).clamp_min(1e-12)).pow(0.5)
    pd_lm = (col_rms_lm * make_signs(2048, 202)).contiguous()
    x_lm_rot = fwht128((h_final * pd_lm.unsqueeze(0)).reshape(N_CAL, 16, 128))
    H_lm = torch.einsum("nbi,nbj->bij", x_lm_rot, x_lm_rot) / float(N_CAL)
    lm_blob = pack_hg4_16r_blob(lm_f32, fixed_pre_diag=pd_lm, H_blocks=H_lm)
    (out_dir / "lm_head.hg4").write_bytes(lm_blob)
    print(f"[Global] lm_head.hg4 (HG4-16R + LDLQ): {len(lm_blob)/1048576:.2f} MiB in {time.perf_counter()-tl_lm:.1f}s", flush=True)

    if not (out_dir / "tokenizer.json").is_file():
        shutil.copy2(stage_trunks / "tokenizer.json", out_dir / "tokenizer.json")

    tot_b = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file() and f.suffix in (".bin", ".hg4"))
    print(
        f"\n=== COMPLETED Clover-35B-A3B ({args.max_layers} layers) ===\n"
        f"Total HG4-16R weight size: {tot_b / (1024**3):.3f} GiB ({tot_b / 1048576:.1f} MiB) in {(time.perf_counter()-t0)/60:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
