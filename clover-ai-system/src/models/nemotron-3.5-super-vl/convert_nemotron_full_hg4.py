#!/usr/bin/env python3
"""Full 90-Layer (`88` Backbone + `2` MTP, `20,992` Routed `ReLU^2` Experts) `HG4-16R + Shrinkage-LDLQ`
Streaming Converter for `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`
-> `/opt/clover-k3/nemotron-full-hg4` (~59.5 GiB).

Streams exact per-layer tensors into RAM via 16-worker coalesced HTTP Range requests with 1-layer-ahead
background prefetching (0 temporary shard bytes on disk), runs sequential 1,024-token multi-domain
calibration + Opus Shrinkage-LDLQ + Closed-Form `ReLU^2` Equilibrium `pre_diag_down`, and writes
canonical 16-row `HG4-16R` / `HG8-16R` files.
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

import kimi_k31_gpu_backbone as kb  # noqa: E402

LUT16_U8 = np.array([1, 32, 53, 70, 84, 97, 110, 122, 134, 146, 159, 172, 186, 203, 224, 255], dtype=np.uint8)


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
        Path(__file__).resolve().parents[1] / "clover-35b-a3b" / "hg4_fast_packer.c",
        Path("/tmp/hg4_fast_packer.c"),
    ]
    c_src = next((p for p in c_cands if p.is_file()), None)
    if not so_path.is_file() and c_src is not None:
        subprocess.run(
            ["gcc", "-O3", "-fopenmp", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror", str(c_src), "-o", str(so_path), "-lm"],
            check=True,
        )
    assert so_path.is_file(), "Missing /tmp/libhg4_fast_packer.so"
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
    if key in _HEADER_CACHE and (now - _HEADER_CACHE[key][3]) < 540.0:
        c_url, hdr, base, _ = _HEADER_CACHE[key]
        return c_url, hdr, base
    kb.enable_ipv4_only()
    url = hf_hub_url(repo_id, filename)
    tok = os.environ.get("HF_TOKEN")
    auth_hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={**auth_hdr, "Range": "bytes=0-7"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                cdn_url = resp.geturl()
                hlen = struct.unpack("<Q", resp.read(8))[0]
            req2 = urllib.request.Request(cdn_url, headers={"Range": f"bytes=8-{8 + hlen - 1}"})
            with urllib.request.urlopen(req2, timeout=45) as resp:
                header = json.loads(resp.read().decode("utf-8"))
            _HEADER_CACHE[key] = (cdn_url, header, hlen + 8, time.time())
            return cdn_url, header, hlen + 8
        except Exception:
            if attempt == 7:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Unreachable")


def fetch_tensors_ram(
    repo_id: str,
    weight_map: dict[str, str],
    keys: list[str],
    max_workers: int = 16,
) -> dict[str, torch.Tensor]:
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
            futs = [(si, ro, pool.submit(kb._fetch_byte_range, cdn_url, bs, be)) for si, ro, bs, be in subtasks]
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
                else:
                    arr = np.frombuffer(sl, dtype=np.float16).copy()
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


def relu2_eq_pre_diag(up_f: torch.Tensor, dn_f: torch.Tensor, seed: int) -> torch.Tensor:
    dn_crms = dn_f.pow(2).mean(dim=0).clamp_min(1e-12).sqrt()
    up_rrms2 = up_f.pow(2).mean(dim=1).clamp_min(1e-12)
    return ((dn_crms / up_rrms2).pow(0.5) * make_signs(dn_f.shape[1], seed)).contiguous()


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


def compute_input_hessian_blocks(X_flat: torch.Tensor, pre_diag: torch.Tensor) -> torch.Tensor:
    N, K = X_flat.shape
    nb = K // 128
    x_rot = fwht128((X_flat.float() * pre_diag.unsqueeze(0)).reshape(N, nb, 128))
    return torch.einsum("nbi,nbj->bij", x_rot, x_rot) / float(N)


def pack_hg4_16r_blob(
    W_f32: torch.Tensor,
    seed: int = 42,
    fixed_pre_diag: torch.Tensor | None = None,
    X_calib: torch.Tensor | None = None,
    return_dequant: bool = False,
) -> bytes | tuple[bytes, torch.Tensor]:
    R, K = W_f32.shape
    assert R % 16 == 0 and K % 128 == 0, f"Invalid shape {R}x{K}"
    nb = K // 128
    if fixed_pre_diag is not None:
        pre_diag = fixed_pre_diag.to(torch.float32).contiguous()
    else:
        col_rms = torch.sqrt(torch.mean(W_f32 * W_f32, dim=0).clamp_min(1e-12)).pow(0.5)
        pre_diag = (col_rms * make_signs(K, seed)).contiguous()

    F_ldlq = None
    if X_calib is not None:
        H_blk = compute_input_hessian_blocks(X_calib, pre_diag)
        F_ldlq = ldlq_feedback_factors(H_blk).contiguous()

    fast_lib = get_fast_packer_lib()
    wc = W_f32.to(torch.float32).contiguous()
    out_sc = np.empty((R // 16, nb, 16), dtype=np.uint16)
    out_pk = np.empty((R * K) // 2, dtype=np.uint8)
    out_dq = torch.empty((R, K), dtype=torch.float32) if return_dequant else None
    fp = ctypes.POINTER(ctypes.c_float)
    fast_lib.hg4_pack_16r_f32(
        ctypes.cast(wc.data_ptr(), fp),
        ctypes.cast(pre_diag.data_ptr(), fp),
        ctypes.cast(F_ldlq.data_ptr(), fp) if F_ldlq is not None else None,
        int(R), int(K),
        out_sc.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        out_pk.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.cast(out_dq.data_ptr(), fp) if out_dq is not None else None,
    )
    blob = struct.pack("<II", R, K) + pre_diag.numpy().tobytes() + out_sc.tobytes() + out_pk.tobytes()
    return (blob, out_dq) if return_dequant else blob


def pack_hg8_16r_blob(
    W_f32: torch.Tensor,
    seed: int = 42,
    fixed_pre_diag: torch.Tensor | None = None,
    return_dequant: bool = False,
) -> bytes | tuple[bytes, torch.Tensor]:
    R, K = W_f32.shape
    assert R % 16 == 0 and K % 128 == 0, f"Invalid HG8-16R shape {R}x{K}"
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


def rms_norm_bf16round(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    normed = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(torch.bfloat16).float()
    return (normed * w.float()).to(torch.bfloat16).float()


def relu2_bf16round(x: torch.Tensor) -> torch.Tensor:
    xb = x.to(torch.bfloat16).float().clamp_min(0.0)
    return (xb * xb).to(torch.bfloat16).float()


def forward_mamba2_calib(
    u: torch.Tensor,        # [B, T, 4096]
    zxbc_dq: torch.Tensor,  # [18432, 4096]
    dt_dq: torch.Tensor,    # [128, 4096]
    cw: torch.Tensor,       # [10240, 4]
    cb_b: torch.Tensor,     # [10240]
    dtb: torch.Tensor,      # [128]
    A_neg: torch.Tensor,    # [128]
    D_vec: torch.Tensor,    # [128]
    mix_norm: torch.Tensor, # [8192]
) -> torch.Tensor:          # [B*T, 8192]
    B, T, _ = u.shape
    uf = u.reshape(B * T, 4096)
    zxbc = (uf @ zxbc_dq.T).to(torch.bfloat16).float().reshape(B, T, 18432)
    dt_p = (uf @ dt_dq.T).to(torch.bfloat16).float().reshape(B, T, 128)

    gate = zxbc[:, :, :8192]
    xbc = zxbc[:, :, 8192:]  # [B, T, 10240]
    xbc_pad = F.pad(xbc.transpose(1, 2), (3, 0))
    conv_raw = F.conv1d(xbc_pad, cw.view(10240, 1, 4), bias=cb_b, groups=10240).transpose(1, 2).to(torch.bfloat16).float()
    conv_act = F.silu(conv_raw).to(torch.bfloat16).float()

    x_ssm = conv_act[:, :, :8192].reshape(B, T, 128, 64)
    B_ssm = conv_act[:, :, 8192:9216].reshape(B, T, 8, 128).repeat_interleave(16, dim=2)
    C_ssm = conv_act[:, :, 9216:10240].reshape(B, T, 8, 128).repeat_interleave(16, dim=2)

    dt_val = F.softplus(dt_p + dtb.view(1, 1, 128)).clamp_min(0.001).to(torch.bfloat16).float()  # [B, T, 128]
    dA = torch.exp(dt_val * A_neg.view(1, 1, 128))  # [B, T, 128]

    state = torch.zeros((B, 128, 64, 128), dtype=torch.float32)
    ys = []
    D_view = D_vec.view(1, 128, 1)
    for t in range(T):
        xt = x_ssm[:, t]  # [B, 128, 64]
        bt = (dt_val[:, t].unsqueeze(-1) * B_ssm[:, t]).unsqueeze(2)  # [B, 128, 1, 128]
        state = state * dA[:, t, :, None, None] + xt.unsqueeze(-1) * bt
        yt = torch.einsum("bhds,bhs->bhd", state, C_ssm[:, t]) + D_view * xt
        ys.append(yt)
    y_ssm = torch.stack(ys, dim=1).reshape(B, T, 8, 1024)
    gated = y_ssm * F.silu(gate.reshape(B, T, 8, 1024))
    inv = torch.rsqrt(gated.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    scan_out = ((gated * inv) * mix_norm.view(1, 1, 8, 1024)).to(torch.bfloat16).float()
    return scan_out.reshape(B * T, 8192)


def pack_512_experts_ldlq(
    td: dict[str, torch.Tensor],
    epfx: str,
    latent_in: torch.Tensor,  # [N_cal, 1024]
    topk_ids: torch.Tensor,   # [N_cal, 22]
    topk_w: torch.Tensor,     # [N_cal, 22]
    out_path: Path,
    seed_base: int,
) -> torch.Tensor:
    N_cal = latent_in.shape[0]
    nb_up = 8  # 1024 / 128
    nb_dn = 21 # 2688 / 128

    # Stack 512 experts cleanly in chunks of 64 to keep peak RAM < 4 GB
    ss_col = torch.zeros(1024, dtype=torch.float64)
    for e in range(512):
        uf = td[f"{epfx}{e}.up_proj.weight"].float()
        ss_col += uf.pow(2).sum(dim=0).to(torch.float64)
    col_rms_up = torch.sqrt((ss_col / (512.0 * 2688.0)).clamp_min(1e-12)).pow(0.5).to(torch.float32)
    pre_diag_up = (col_rms_up * make_signs(1024, seed_base)).contiguous()

    x_rot = fwht128((latent_in.float() * pre_diag_up.unsqueeze(0)).reshape(N_cal, nb_up, 128))
    H_layer = torch.einsum("nbi,nbj->bij", x_rot, x_rot) / float(N_cal)
    alpha_shrink = 128.0

    latent_acc = torch.zeros((N_cal, 1024), dtype=torch.float32)
    exp_stride = 1419264 + 1430024
    fast_lib = get_fast_packer_lib()
    fp = ctypes.POINTER(ctypes.c_float)
    eye21 = torch.eye(128, dtype=torch.float32).unsqueeze(0).expand(nb_dn, 128, 128)

    with open(out_path, "wb") as wf:
        wf.write(struct.pack("<4sIIIIII", b"HG4F", 512, 2688, 1024, 1024, 2688, exp_stride))
        wf.write(pre_diag_up.numpy().tobytes())

        for e0 in range(0, 512, 64):
            B_E = 64
            up_sub = torch.stack([td.pop(f"{epfx}{e}.up_proj.weight").float() for e in range(e0, e0 + B_E)], dim=0).contiguous()
            dn_sub = torch.stack([td.pop(f"{epfx}{e}.down_proj.weight").float() for e in range(e0, e0 + B_E)], dim=0).contiguous()

            H_up_b = torch.empty((B_E, nb_up, 128, 128), dtype=torch.float32)
            routed_toks: list[tuple[torch.Tensor, torch.Tensor]] = []
            for bi, e in enumerate(range(e0, e0 + B_E)):
                tok_idx, slot_idx = torch.where(topk_ids == e)
                n_e = int(tok_idx.numel())
                routed_toks.append((tok_idx, topk_w[tok_idx, slot_idx]))
                if n_e >= 8:
                    xe = x_rot[tok_idx]
                    H_up_b[bi] = (torch.einsum("nbi,nbj->bij", xe, xe) + alpha_shrink * H_layer) / (float(n_e) + alpha_shrink)
                else:
                    H_up_b[bi] = H_layer

            F_up_b = ldlq_feedback_factors(H_up_b.reshape(B_E * nb_up, 128, 128)).reshape(B_E, nb_up, 128, 128).contiguous()
            out_sc_u = np.empty((B_E, (2688 // 16) * nb_up * 16), dtype=np.uint16)
            out_pk_u = np.empty((B_E, (2688 * 1024) // 2), dtype=np.uint8)
            dq_up = torch.empty((B_E, 2688, 1024), dtype=torch.float32)
            fast_lib.hg4_pack_experts_16r_f32(
                ctypes.cast(up_sub.data_ptr(), fp),
                ctypes.cast(pre_diag_up.data_ptr(), fp),
                0,
                ctypes.cast(F_up_b.data_ptr(), fp),
                B_E, 2688, 1024,
                out_sc_u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out_pk_u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.cast(dq_up.data_ptr(), fp),
            )

            pd_dn_b = torch.stack([relu2_eq_pre_diag(up_sub[bi], dn_sub[bi], seed=seed_base + 1 + e0 + bi) for bi in range(B_E)], dim=0).contiguous()
            H_dn_b = torch.empty((B_E, nb_dn, 128, 128), dtype=torch.float32)
            z_acts: list[torch.Tensor | None] = [None] * B_E
            for bi in range(B_E):
                tok_idx, _ = routed_toks[bi]
                if tok_idx.numel() > 0:
                    z_act = relu2_bf16round(latent_in[tok_idx] @ dq_up[bi].T)  # [n_e, 2688]
                    z_acts[bi] = z_act
                    if tok_idx.numel() >= 16:
                        z_rot = fwht128((z_act * pd_dn_b[bi].unsqueeze(0)).reshape(-1, nb_dn, 128))
                        n_e = float(tok_idx.numel())
                        emp_H = torch.einsum("nbi,nbj->bij", z_rot, z_rot)
                        diag_ref = (emp_H.diagonal(dim1=-2, dim2=-1).mean() / n_e).clamp_min(1e-8)
                        H_dn_b[bi] = (emp_H + 128.0 * diag_ref * eye21) / (n_e + 128.0)
                    else:
                        H_dn_b[bi] = eye21
                else:
                    H_dn_b[bi] = eye21

            F_dn_b = ldlq_feedback_factors(H_dn_b.reshape(B_E * nb_dn, 128, 128)).reshape(B_E, nb_dn, 128, 128).contiguous()
            out_sc_d = np.empty((B_E, (1024 // 16) * nb_dn * 16), dtype=np.uint16)
            out_pk_d = np.empty((B_E, (1024 * 2688) // 2), dtype=np.uint8)
            dq_dn = torch.empty((B_E, 1024, 2688), dtype=torch.float32)
            fast_lib.hg4_pack_experts_16r_f32(
                ctypes.cast(dn_sub.data_ptr(), fp),
                ctypes.cast(pd_dn_b.data_ptr(), fp),
                1,
                ctypes.cast(F_dn_b.data_ptr(), fp),
                B_E, 1024, 2688,
                out_sc_d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out_pk_d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.cast(dq_dn.data_ptr(), fp),
            )

            for bi in range(B_E):
                up_body = out_sc_u[bi].tobytes() + out_pk_u[bi].tobytes()
                dn_blob = (
                    struct.pack("<II", 1024, 2688)
                    + pd_dn_b[bi].numpy().tobytes()
                    + out_sc_d[bi].tobytes()
                    + out_pk_d[bi].tobytes()
                )
                assert len(up_body) + len(dn_blob) == exp_stride
                wf.write(up_body)
                wf.write(dn_blob)

                tok_idx, rw = routed_toks[bi]
                z_act = z_acts[bi]
                if z_act is not None and tok_idx.numel() > 0:
                    ey = (z_act @ dq_dn[bi].T).to(torch.bfloat16).float()
                    latent_acc.index_add_(0, tok_idx, ey * rw.unsqueeze(-1))

            del up_sub, dn_sub, dq_up, dq_dn

    return latent_acc.to(torch.bfloat16).float()


def build_calibration_ids(tok: Tokenizer, n_seqs: int = 8, seq_len: int = 128) -> torch.Tensor:
    base_texts = [
        "Fast Walsh-Hadamard orthogonal rotations disperse activation and weight outliers uniformly across 128-element blocks, enabling 4-bit Gaussian Lloyd-Max codebooks with Cholesky LDLQ error feedback to match 16-bit BF16 perplexity.",
        "def cholesky_ldlq_quantize(W: torch.Tensor, H: torch.Tensor, damp: float = 0.015) -> torch.Tensor:\n    n = H.shape[-1]\n    H_reg = H + damp * torch.diagonal(H).mean() * torch.eye(n)\n    return torch.linalg.cholesky(torch.flip(H_reg, [-2, -1]))\n",
        "Theorem (Parseval Identity): For any orthogonal transform Q satisfying Q^T Q = I, <Qw, Qx> = <w, x> holds identically across Euclidean inner-product spaces.",
        "User: A data center runs 40 racks with 16 servers each at 0.45 kW per server. Find the total power draw.\nAssistant: 40 * 16 = 640 servers; 640 * 0.45 kW = 288.0 kW.",
        "static inline void avx512_vnni_dpbusd(__m512i *acc, __m512i w_u8, __m512i x_s8) { *acc = _mm512_dpbusd_epi32(*acc, w_u8, x_s8); }",
        "Hybrid Mamba-2 State-Space and Latent Mixture-of-Experts models combine linear-time selective recurrence with sparse squared-ReLU expert capacity and periodic grouped-query attention.",
        "SELECT region, SUM(revenue) AS total_rev, COUNT(*) AS orders FROM sales WHERE year = 2026 GROUP BY region ORDER BY total_rev DESC;",
        "Shrinkage covariance estimation stabilizes high-dimensional inverse-Hessian error feedback whenever routed token counts per expert are smaller than layer width.",
    ]
    joined = "\n\n".join(base_texts * 10)
    all_ids = tok.encode(joined, add_special_tokens=False).ids
    need = n_seqs * seq_len
    while len(all_ids) < need + 64:
        all_ids = all_ids + all_ids
    stride = max(1, (len(all_ids) - seq_len) // n_seqs)
    return torch.tensor([all_ids[i * stride : i * stride + seq_len] for i in range(n_seqs)], dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/opt/clover-k3/nemotron-full-hg4")
    ap.add_argument("--max-layers", type=int, default=88)
    ap.add_argument("--include-mtp", type=int, default=1)
    args = ap.parse_args()

    auto_load_hf_token()
    kb.enable_ipv4_only()
    t0 = time.perf_counter()

    repo_id = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
    stage_dir = Path("/opt/clover-k3/nemotron-computed")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((stage_dir / "hf-stage/trunks/config.json").read_text())
    lm_cfg = cfg.get("llm_config") or cfg
    block_types: list[str] = lm_cfg["layers_block_type"][:args.max_layers]
    n_layers = len(block_types)

    idx_path = hf_hub_download(repo_id, "model.safetensors.index.json", token=os.environ.get("HF_TOKEN"))
    weight_map: dict[str, str] = json.loads(Path(idx_path).read_text())["weight_map"]
    tok = Tokenizer.from_file(str(stage_dir / "hf-stage/trunks/tokenizer.json"))
    calib_ids = build_calibration_ids(tok, n_seqs=8, seq_len=128)
    B_CAL, T_CAL = calib_ids.shape
    N_CAL = B_CAL * T_CAL
    print(f"Nemotron 3.5 Full Conversion: {n_layers} backbone layers + {'2 MTP' if args.include_mtp else '0 MTP'} | Calib: {B_CAL}x{T_CAL}={N_CAL} tok", flush=True)

    # Kind codes: 0 = mamba, 1 = moe, 2 = attention
    kinds = [0 if bt == "mamba" else (1 if bt == "moe" else 2) for bt in block_types]

    norm_f_local = stage_dir / "pruned/global-0/norm_f.bf16.bin"
    if norm_f_local.is_file():
        norm_f = kb.bytes_to_tensor(norm_f_local.read_bytes(), kb.DTYPE_BF16, (4096,)).float().contiguous()
    else:
        norm_f = fetch_tensors_ram(repo_id, weight_map, ["language_model.backbone.norm_f.weight"])["language_model.backbone.norm_f.weight"].float().contiguous()

    (out_dir / "meta.bin").write_bytes(
        struct.pack("<4sIIII", b"HG4P", 2, n_layers, 4096, 131072)
        + LUT16_U8.tobytes()
        + norm_f.numpy().tobytes()
        + struct.pack(f"<{n_layers}I", *kinds)
    )

    emb_local = stage_dir / "pruned/global-0/embeddings.bf16.bin"
    if emb_local.is_file():
        emb_mm = np.memmap(emb_local, dtype=np.uint16, mode="r").reshape(131072, 4096)
        emb_f32 = torch.from_numpy(np.left_shift(np.asarray(emb_mm, dtype=np.uint32), np.uint32(16)).view(np.float32))
    else:
        emb_f32 = fetch_tensors_ram(repo_id, weight_map, ["language_model.backbone.embeddings.weight"])["language_model.backbone.embeddings.weight"].float()

    emb_p = out_dir / "embeddings.hg4"
    if not emb_p.is_file():
        te = time.perf_counter()
        emb_blob = pack_hg4_16r_blob(emb_f32, seed=101, X_calib=None)
        emb_p.write_bytes(emb_blob)
        print(f"[Global] embeddings.hg4: {len(emb_blob)/1048576:.2f} MiB in {time.perf_counter()-te:.1f}s", flush=True)

    # Check resume state
    ckpt_p = out_dir / "_calib_resume.pt"
    start_layer = 0
    if ckpt_p.is_file():
        ck = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
        start_layer = int(ck["next_layer"])
        h_cal = ck["h_cal"].float()
        print(f"[Resume] Resuming from Layer {start_layer:02d}/{n_layers - 1:02d}", flush=True)
    else:
        h_cal = emb_f32[calib_ids].to(torch.bfloat16).float().clone()
    e_cal_initial = emb_f32[calib_ids].to(torch.bfloat16).float().clone()
    del emb_f32
    gc.collect()

    def keys_for_backbone_layer(lid: int) -> list[str]:
        lp = f"language_model.backbone.layers.{lid}."
        return [k for k in weight_map if k.startswith(lp)]

    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    next_fut = (
        prefetch_pool.submit(fetch_tensors_ram, repo_id, weight_map, keys_for_backbone_layer(start_layer), 16)
        if start_layer < n_layers else None
    )

    for l in range(start_layer, n_layers):
        tl0 = time.perf_counter()
        bt = block_types[l]
        pfx = f"language_model.backbone.layers.{l}."
        assert next_fut is not None
        td = next_fut.result()
        t_dl = time.perf_counter() - tl0
        next_fut = (
            prefetch_pool.submit(fetch_tensors_ram, repo_id, weight_map, keys_for_backbone_layer(l + 1), 16)
            if (l + 1 < n_layers) else None
        )

        norm_w = td.pop(pfx + "norm.weight").float().contiguous()
        u = rms_norm_bf16round(h_cal, norm_w, eps=1e-5)
        u_flat = u.reshape(N_CAL, 4096)
        layer_path = out_dir / f"layer_{l:02d}.hg4"

        if bt == "mamba":
            mpfx = pfx + "mixer."
            m_in = td.pop(mpfx + "in_proj.weight").float()
            m_cw = td.pop(mpfx + "conv1d.weight").float().reshape(10240 * 4).contiguous()
            m_cb = td.pop(mpfx + "conv1d.bias").float().reshape(10240).contiguous()
            m_dtb = td.pop(mpfx + "dt_bias").float().reshape(128).contiguous()
            m_A = (-td.pop(mpfx + "A_log").float().exp()).reshape(128).contiguous()
            m_D = td.pop(mpfx + "D").float().reshape(128).contiguous()
            m_mix_norm = td.pop(mpfx + "norm.weight").float().reshape(8192).contiguous()
            m_out = td.pop(mpfx + "out_proj.weight").float()
            td.clear()

            b_zxbc, zxbc_dq = pack_hg4_16r_blob(m_in[:18432].contiguous(), seed=300 + l * 10 + 1, X_calib=u_flat, return_dequant=True)
            b_dt, dt_dq = pack_hg8_16r_blob(m_in[18432:].contiguous(), seed=300 + l * 10 + 2, return_dequant=True)
            scan_out = forward_mamba2_calib(u, zxbc_dq, dt_dq, m_cw.view(10240, 4), m_cb, m_dtb, m_A, m_D, m_mix_norm)
            b_out, out_dq = pack_hg4_16r_blob(m_out, seed=300 + l * 10 + 3, X_calib=scan_out, return_dequant=True)

            layer_path.write_bytes(
                struct.pack("<I", 0)
                + norm_w.numpy().tobytes()
                + m_cw.numpy().tobytes() + m_cb.numpy().tobytes()
                + m_dtb.numpy().tobytes() + m_A.numpy().tobytes()
                + m_D.numpy().tobytes() + m_mix_norm.numpy().tobytes()
                + struct.pack("<III", len(b_zxbc), len(b_dt), len(b_out))
                + b_zxbc + b_dt + b_out
            )
            branch = (scan_out @ out_dq.T).to(torch.bfloat16).float().reshape(B_CAL, T_CAL, 4096)
            h_cal = (h_cal + branch).to(torch.bfloat16).float()
            l_mb = layer_path.stat().st_size / 1048576.0
            del m_in, m_out, zxbc_dq, dt_dq, out_dq

        elif bt == "attention":
            apfx = pfx + "mixer."
            q_w = td.pop(apfx + "q_proj.weight").float()
            k_w = td.pop(apfx + "k_proj.weight").float()
            v_w = td.pop(apfx + "v_proj.weight").float()
            o_w = td.pop(apfx + "o_proj.weight").float()
            td.clear()

            b_q, q_dq = pack_hg8_16r_blob(q_w, seed=400 + l * 10 + 1, return_dequant=True)
            b_k, k_dq = pack_hg8_16r_blob(k_w, seed=400 + l * 10 + 2, return_dequant=True)
            b_v, v_dq = pack_hg8_16r_blob(v_w, seed=400 + l * 10 + 3, return_dequant=True)
            b_o, o_dq = pack_hg8_16r_blob(o_w, seed=400 + l * 10 + 4, return_dequant=True)

            layer_path.write_bytes(
                struct.pack("<I", 2)
                + norm_w.numpy().tobytes()
                + struct.pack("<IIII", len(b_q), len(b_k), len(b_v), len(b_o))
                + b_q + b_k + b_v + b_o
            )
            qh = (u_flat @ q_dq.T).to(torch.bfloat16).float().reshape(B_CAL, T_CAL, 32, 128).transpose(1, 2)
            kh = (u_flat @ k_dq.T).to(torch.bfloat16).float().reshape(B_CAL, T_CAL, 2, 128).repeat_interleave(16, dim=2).transpose(1, 2)
            vh = (u_flat @ v_dq.T).to(torch.bfloat16).float().reshape(B_CAL, T_CAL, 2, 128).repeat_interleave(16, dim=2).transpose(1, 2)
            ay = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True).transpose(1, 2).to(torch.bfloat16).float().reshape(N_CAL, 4096)
            branch = (ay @ o_dq.T).to(torch.bfloat16).float().reshape(B_CAL, T_CAL, 4096)
            h_cal = (h_cal + branch).to(torch.bfloat16).float()
            l_mb = layer_path.stat().st_size / 1048576.0
            del q_w, k_w, v_w, o_w, q_dq, k_dq, v_dq, o_dq

        else:  # "moe"
            mpfx = pfx + "mixer."
            gate_w = td.pop(mpfx + "gate.weight").float().contiguous()
            gate_b = td.pop(mpfx + "gate.e_score_correction_bias").float().contiguous()
            fc1_w = td.pop(mpfx + "fc1_latent_proj.weight").float()
            fc2_w = td.pop(mpfx + "fc2_latent_proj.weight").float()
            sh_up = td.pop(mpfx + "shared_experts.up_proj.weight").float()
            sh_dn = td.pop(mpfx + "shared_experts.down_proj.weight").float()

            b_fc1, fc1_dq = pack_hg8_16r_blob(fc1_w, seed=500 + l * 10 + 1, return_dequant=True)
            b_fc2, fc2_dq = pack_hg8_16r_blob(fc2_w, seed=500 + l * 10 + 2, return_dequant=True)
            b_sup, sup_dq = pack_hg4_16r_blob(sh_up, seed=500 + l * 10 + 3, X_calib=u_flat, return_dequant=True)
            sh_inter = relu2_bf16round(u_flat @ sup_dq.T)
            pd_sdn = relu2_eq_pre_diag(sh_up, sh_dn, seed=500 + l * 10 + 4)
            b_sdn, sdn_dq = pack_hg4_16r_blob(sh_dn, fixed_pre_diag=pd_sdn, X_calib=sh_inter, return_dequant=True)

            layer_path.write_bytes(
                struct.pack("<I", 1)
                + norm_w.numpy().tobytes()
                + gate_w.numpy().tobytes()
                + gate_b.numpy().tobytes()
                + struct.pack("<IIII", len(b_fc1), len(b_fc2), len(b_sup), len(b_sdn))
                + b_fc1 + b_fc2 + b_sup + b_sdn
            )

            latent_in = (u_flat @ fc1_dq.T).to(torch.bfloat16).float()
            scores = torch.sigmoid(u_flat @ gate_w.T)
            _, topk_ids = torch.topk(scores + gate_b.unsqueeze(0), k=22, dim=-1)
            sel_sc = torch.gather(scores, 1, topk_ids)
            topk_w = (sel_sc / (sel_sc.sum(dim=-1, keepdim=True) + 1e-20)) * 5.0

            exp_path = out_dir / f"experts_{l:02d}.hg4"
            latent_acc = pack_512_experts_ldlq(
                td, mpfx + "experts.", latent_in, topk_ids, topk_w, exp_path, seed_base=20000 + l * 600
            )
            td.clear()

            routed_4096 = (latent_acc @ fc2_dq.T).to(torch.bfloat16).float()
            shared_4096 = (sh_inter @ sdn_dq.T).to(torch.bfloat16).float()
            comb = (routed_4096 + shared_4096).to(torch.bfloat16).float().reshape(B_CAL, T_CAL, 4096)
            h_cal = (h_cal + comb).to(torch.bfloat16).float()
            l_mb = (layer_path.stat().st_size + exp_path.stat().st_size) / 1048576.0
            del fc1_w, fc2_w, sh_up, sh_dn, fc1_dq, fc2_dq, sup_dq, sdn_dq

        gc.collect()
        if (l + 1) % 8 == 0:
            torch.save({"next_layer": l + 1, "h_cal": h_cal.to(torch.bfloat16)}, str(ckpt_p))
        dt_l = time.perf_counter() - tl0
        print(
            f"[Layer {l:02d}/{n_layers - 1:02d} {bt.upper():9s}] "
            f"{l_mb:7.1f} MiB | dl={t_dl:4.1f}s tot={dt_l:5.1f}s (elapsed {(time.perf_counter()-t0)/60:.1f}m)",
            flush=True,
        )

    # Final Norm + LM-Head (`HG4-16R + Final-Layer LDLQ`)
    tl_lm = time.perf_counter()
    h_final_3d = rms_norm_bf16round(h_cal, norm_f, eps=1e-5)
    h_final = h_final_3d.reshape(N_CAL, 4096)
    lm_local = stage_dir / "pruned/global-0/lm_head.bf16.bin"
    if lm_local.is_file():
        lm_mm = np.memmap(lm_local, dtype=np.uint16, mode="r").reshape(131072, 4096)
        lm_f32 = torch.from_numpy(np.left_shift(np.asarray(lm_mm, dtype=np.uint32), np.uint32(16)).view(np.float32))
    else:
        lm_f32 = fetch_tensors_ram(repo_id, weight_map, ["language_model.lm_head.weight"])["language_model.lm_head.weight"].float()

    lm_blob = pack_hg4_16r_blob(lm_f32, seed=202, X_calib=h_final)
    (out_dir / "lm_head.hg4").write_bytes(lm_blob)
    del lm_f32
    print(f"[Global] lm_head.hg4 (HG4-16R + LDLQ): {len(lm_blob)/1048576:.2f} MiB in {time.perf_counter()-tl_lm:.1f}s", flush=True)

    # 2-Layer MTP Speculative Drafter (`mtp_trunks.hg4` + `experts_mtp_1.hg4` in canonical 16R)
    if args.include_mtp and n_layers == 88:
        t_mtp = time.perf_counter()
        mtp_keys = [k for k in weight_map if k.startswith("language_model.mtp.")]
        td_m = fetch_tensors_ram(repo_id, weight_map, mtp_keys, max_workers=16)

        enorm_w = td_m.pop("language_model.mtp.layers.0.enorm.weight").float().contiguous()
        hnorm_w = td_m.pop("language_model.mtp.layers.0.hnorm.weight").float().contiguous()
        eh_proj = td_m.pop("language_model.mtp.layers.0.eh_proj.weight").float()
        final_ln_w = td_m.pop("language_model.mtp.layers.1.final_layernorm.weight").float().contiguous()

        # MTP input pairs token t's backbone hidden state with token t+1's embedding!
        e_next = rms_norm_bf16round(e_cal_initial[:, 1:, :], enorm_w, eps=1e-5)
        h_prev = rms_norm_bf16round(h_final_3d[:, :-1, :], hnorm_w, eps=1e-5)
        eh_cat = torch.cat([e_next, h_prev], dim=-1).reshape(-1, 8192)
        b_eh, eh_dq = pack_hg8_16r_blob(eh_proj, seed=601, return_dequant=True)
        mtp_in = (eh_cat @ eh_dq.T).to(torch.bfloat16).float().reshape(B_CAL, T_CAL - 1, 4096)

        # MTP Layer 0 GQA
        g0_norm = td_m.pop("language_model.mtp.layers.0.norm.weight").float().contiguous()
        q0 = td_m.pop("language_model.mtp.layers.0.mixer.q_proj.weight").float()
        k0 = td_m.pop("language_model.mtp.layers.0.mixer.k_proj.weight").float()
        v0 = td_m.pop("language_model.mtp.layers.0.mixer.v_proj.weight").float()
        o0 = td_m.pop("language_model.mtp.layers.0.mixer.o_proj.weight").float()
        bq0, q0_dq = pack_hg8_16r_blob(q0, seed=611, return_dequant=True)
        bk0, k0_dq = pack_hg8_16r_blob(k0, seed=612, return_dequant=True)
        bv0, v0_dq = pack_hg8_16r_blob(v0, seed=613, return_dequant=True)
        bo0, o0_dq = pack_hg8_16r_blob(o0, seed=614, return_dequant=True)
        b_mtp0_gqa = g0_norm.numpy().tobytes() + struct.pack("<IIII", len(bq0), len(bk0), len(bv0), len(bo0)) + bq0 + bk0 + bv0 + bo0

        u0_m = rms_norm_bf16round(mtp_in, g0_norm, eps=1e-5).reshape(-1, 4096)
        T_M = T_CAL - 1
        ay0 = F.scaled_dot_product_attention(
            (u0_m @ q0_dq.T).reshape(B_CAL, T_M, 32, 128).transpose(1, 2),
            (u0_m @ k0_dq.T).reshape(B_CAL, T_M, 2, 128).repeat_interleave(16, dim=2).transpose(1, 2),
            (u0_m @ v0_dq.T).reshape(B_CAL, T_M, 2, 128).repeat_interleave(16, dim=2).transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2).reshape(-1, 4096)
        mtp_h0 = (mtp_in + (ay0 @ o0_dq.T).reshape(B_CAL, T_M, 4096)).to(torch.bfloat16).float()

        # MTP Layer 1 MoE (512 routed experts)
        m1_norm = td_m.pop("language_model.mtp.layers.1.norm.weight").float().contiguous()
        m1_gw = td_m.pop("language_model.mtp.layers.1.mixer.gate.weight").float().contiguous()
        m1_gb = td_m.pop("language_model.mtp.layers.1.mixer.gate.e_score_correction_bias").float().contiguous()
        m1_fc1 = td_m.pop("language_model.mtp.layers.1.mixer.fc1_latent_proj.weight").float()
        m1_fc2 = td_m.pop("language_model.mtp.layers.1.mixer.fc2_latent_proj.weight").float()
        m1_sup = td_m.pop("language_model.mtp.layers.1.mixer.shared_experts.up_proj.weight").float()
        m1_sdn = td_m.pop("language_model.mtp.layers.1.mixer.shared_experts.down_proj.weight").float()

        u1_m = rms_norm_bf16round(mtp_h0, m1_norm, eps=1e-5).reshape(-1, 4096)
        bm_fc1, mfc1_dq = pack_hg8_16r_blob(m1_fc1, seed=621, return_dequant=True)
        bm_fc2 = pack_hg8_16r_blob(m1_fc2, seed=622)
        bm_sup, msup_dq = pack_hg4_16r_blob(m1_sup, seed=623, X_calib=u1_m, return_dequant=True)
        m_sh_inter = relu2_bf16round(u1_m @ msup_dq.T)
        bm_sdn = pack_hg4_16r_blob(m1_sdn, fixed_pre_diag=relu2_eq_pre_diag(m1_sup, m1_sdn, seed=624), X_calib=m_sh_inter)

        b_mtp1_moe = (
            m1_norm.numpy().tobytes() + m1_gw.numpy().tobytes() + m1_gb.numpy().tobytes()
            + struct.pack("<IIII", len(bm_fc1), len(bm_fc2), len(bm_sup), len(bm_sdn))
            + bm_fc1 + bm_fc2 + bm_sup + bm_sdn
        )
        (out_dir / "mtp_trunks.hg4").write_bytes(
            enorm_w.numpy().tobytes() + hnorm_w.numpy().tobytes() + final_ln_w.numpy().tobytes()
            + struct.pack("<III", len(b_eh), len(b_mtp0_gqa), len(b_mtp1_moe))
            + b_eh + b_mtp0_gqa + b_mtp1_moe
        )

        m_lat_in = (u1_m @ mfc1_dq.T).to(torch.bfloat16).float()
        m_sc = torch.sigmoid(u1_m @ m1_gw.T)
        _, m_ids = torch.topk(m_sc + m1_gb.unsqueeze(0), k=22, dim=-1)
        m_sel = torch.gather(m_sc, 1, m_ids)
        m_w = (m_sel / (m_sel.sum(dim=-1, keepdim=True) + 1e-20)) * 5.0
        pack_512_experts_ldlq(
            td_m, "language_model.mtp.layers.1.mixer.experts.", m_lat_in, m_ids, m_w,
            out_dir / "experts_mtp_1.hg4", seed_base=90000,
        )
        td_m.clear()
        print(f"[MTP 2L] Packed mtp_trunks.hg4 + experts_mtp_1.hg4 in {time.perf_counter()-t_mtp:.1f}s", flush=True)

    prefetch_pool.shutdown(wait=False)
    ckpt_p.unlink(missing_ok=True)
    if not (out_dir / "tokenizer.json").is_file():
        shutil.copy2(stage_dir / "hf-stage/trunks/tokenizer.json", out_dir / "tokenizer.json")

    tot_b = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file() and f.suffix in (".bin", ".hg4"))
    print(
        f"\n=== COMPLETED Nemotron 3.5 Super VL ({n_layers} backbone + {'2 MTP' if args.include_mtp else '0 MTP'}) ===\n"
        f"Total HG4-16R weight size: {tot_b / (1024**3):.3f} GiB ({tot_b / 1048576:.1f} MiB) in {(time.perf_counter()-t0)/60:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
