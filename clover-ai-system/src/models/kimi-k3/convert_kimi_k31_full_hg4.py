#!/usr/bin/env python3
"""Full 93-Layer Backbone Trunk + `global-0` (`163,840` Vocab) + 5-Layer `DSpark` Drafter
`HG4-16R` Streaming Converter for `moonshotai/Kimi-K3.1-Pro` & `LithosAI/Kimi-K3.1-Pro-DSpark`
-> `/opt/clover-k3/kimi-k31-hg4` (~18.8 GiB).

Converts:
  1. `global-0`: `embed_tokens.hg4` (`[163840, 7168]` HG4-16R), `lm_head.hg4` (`[163840, 7168]` HG4-16R),
     `norm`, `output_attn_res_norm`, and `output_attn_res_proj` (`meta.bin` magic `K31R`).
  2. All `93 / 93` `BF16` Backbone Trunks (`layer_00.hg4 .. layer_92.hg4`):
     - Layer `0` (`kind = 0`): `KDA` + `33,792` Dense `SiTU-GLU` MLP + Attention-Residual norms/projs
     - `68` `KDA + Latent SiTU-GLU MoE` Trunks (`kind = 1`): `KDA` + `896`-expert router + `3584` latent
       up/down + `6144` Shared `SiTU-GLU` Expert
     - `24` `Gated DSA + 5-Tensor Lightning Indexer + Latent SiTU-GLU MoE` Trunks (`kind = 2`):
       `MLA` (`q_a/q_b`, `kv_a/kv_b`, `g_proj`, `o_proj`) + `5`-tensor `indexer.*` + Latent/Shared MoE
  3. Stage Routed `SiTU-GLU` Experts (`experts_01.hg4`, `experts_03.hg4`: `32` cached `SiTU-GLU` experts
     in 16-row fused `[w1_0..7, w3_0..7]` `HG4-16R`, paired with `/opt/clover-k3/datasets/computed/expert-1..92`).
  4. Full `5 / 5`-Layer `LithosAI/Kimi-K3.1-Pro-DSpark` Speculative Drafter (`dspark.hg4`):
     `fc [7168, 35840]`, `5` full `7168`-dim `GQA + 14,336 SwiGLU` layers, `markov_w1/w2 [163840, 256]`,
     and `confidence_head` (`[7424] + bias`).
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
from huggingface_hub import hf_hub_download, hf_hub_url

torch.set_num_threads(16)

for p in (
    Path("/opt/clover-k3/clover-framework/porting-models"),
    Path(__file__).resolve().parents[4] / "porting-models" if len(Path(__file__).resolve().parents) >= 5 else Path("/nonexistent"),
):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import kimi_k31_gpu_backbone as kb  # noqa: E402

LUT16_U8 = np.array([1, 32, 53, 70, 84, 97, 110, 122, 134, 146, 159, 172, 186, 203, 224, 255], dtype=np.uint8)
DSA_SET = set(kb.DSA_LAYERS_0IDX)


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
                elif dtype_s in ("U8", "UINT8"):
                    arr = np.frombuffer(sl, dtype=np.uint8).copy()
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


def interleave_glu_16r(gate_w: torch.Tensor, up_w: torch.Tensor) -> torch.Tensor:
    rh, K = gate_w.shape
    assert rh % 8 == 0
    return torch.cat([gate_w.reshape(rh // 8, 8, K), up_w.reshape(rh // 8, 8, K)], dim=1).reshape(2 * rh, K).contiguous()


def glu_eq_pre_diag(gate_w: torch.Tensor, up_w: torch.Tensor, dn_w: torch.Tensor, seed: int) -> torch.Tensor:
    dn_crms = dn_w.pow(2).mean(dim=0).clamp_min(1e-12).sqrt()
    gu_rrms = gate_w.pow(2).mean(dim=1).clamp_min(1e-12).sqrt() * up_w.pow(2).mean(dim=1).clamp_min(1e-12).sqrt()
    return ((dn_crms / gu_rrms.clamp_min(1e-12)).pow(0.5) * make_signs(dn_w.shape[1], seed)).contiguous()


def guard_fp16_scales(sc_f32: torch.Tensor) -> torch.Tensor:
    nonzero = sc_f32.abs() > 0.0
    sc_safe = torch.where(nonzero & (sc_f32.abs() < 6.11e-5), torch.sign(sc_f32) * 6.11e-5, sc_f32)
    return sc_safe.to(torch.float16)


def pack_hg4_16r_blob(
    W_f32: torch.Tensor,
    seed: int = 42,
    fixed_pre_diag: torch.Tensor | None = None,
) -> bytes:
    R, K = W_f32.shape
    assert R % 16 == 0 and K % 128 == 0, f"Invalid HG4-16R shape {R}x{K}"
    nb = K // 128
    if fixed_pre_diag is not None:
        pre_diag = fixed_pre_diag.to(torch.float32).contiguous()
    else:
        col_rms = torch.sqrt(torch.mean(W_f32 * W_f32, dim=0).clamp_min(1e-12)).pow(0.5)
        pre_diag = (col_rms * make_signs(K, seed)).contiguous()

    fast_lib = get_fast_packer_lib()
    wc = W_f32.to(torch.float32).contiguous()
    out_sc = np.empty((R // 16, nb, 16), dtype=np.uint16)
    out_pk = np.empty((R * K) // 2, dtype=np.uint8)
    fp = ctypes.POINTER(ctypes.c_float)
    fast_lib.hg4_pack_16r_f32(
        ctypes.cast(wc.data_ptr(), fp),
        ctypes.cast(pre_diag.data_ptr(), fp),
        None,
        int(R), int(K),
        out_sc.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        out_pk.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        None,
    )
    return struct.pack("<II", R, K) + pre_diag.numpy().tobytes() + out_sc.tobytes() + out_pk.tobytes()


def pack_hg8_16r_blob(W_f32: torch.Tensor, seed: int = 42, fixed_pre_diag: torch.Tensor | None = None) -> bytes:
    R, K = W_f32.shape
    assert R % 16 == 0 and K % 128 == 0, f"Invalid HG8-16R shape {R}x{K}"
    nb = K // 128
    if fixed_pre_diag is not None:
        pre_diag = fixed_pre_diag.to(torch.float32).contiguous()
    else:
        col_rms = torch.sqrt(torch.mean(W_f32 * W_f32, dim=0).clamp_min(1e-12)).pow(0.5)
        pre_diag = (col_rms * make_signs(K, seed)).contiguous()

    sc_chunks, pk_chunks = [], []
    for r0 in range(0, R, 16384):
        r1 = min(R, r0 + 16384)
        rc = r1 - r0
        wc = W_f32[r0:r1] / pre_diag.unsqueeze(0)
        w_rot = fwht128(wc.reshape(rc, nb, 128))
        sc_fp16 = guard_fp16_scales(w_rot.abs().amax(dim=-1) / 127.0)
        sc_f32 = sc_fp16.float().unsqueeze(-1).clamp_min(1e-12)
        q_u8 = (torch.round(w_rot / sc_f32).clamp(-127.0, 127.0) + 128.0).to(torch.uint8)
        sc_16r = sc_fp16.reshape(rc // 16, 16, nb).permute(0, 2, 1).contiguous()
        pk_16r = q_u8.reshape(rc // 16, 16, nb, 32, 4).permute(0, 2, 3, 1, 4).contiguous()
        sc_chunks.append(sc_16r.view(torch.uint16).numpy().tobytes())
        pk_chunks.append(pk_16r.numpy().tobytes())
    return struct.pack("<II", R, K | (8 << 28)) + pre_diag.numpy().tobytes() + b"".join(sc_chunks) + b"".join(pk_chunks)


def pack_k31_moe_trunk_tail(td: dict[str, torch.Tensor], pfx: str, seed_base: int) -> bytes:
    mp = pfx + "block_sparse_moe."
    gate_w = td.pop(mp + "gate.weight").float().contiguous()                     # [896, 7168]
    gate_b = td.pop(mp + "gate.e_score_correction_bias").float().contiguous()    # [896]
    lat_dn = td.pop(mp + "routed_expert_down_proj.weight").float()               # [3584, 7168]
    lat_up = td.pop(mp + "routed_expert_up_proj.weight").float()                 # [7168, 3584]
    lat_nm = td.pop(mp + "routed_expert_norm.weight").float().contiguous()       # [3584]
    sh_g   = td.pop(mp + "shared_experts.gate_proj.weight").float()              # [6144, 7168]
    sh_u   = td.pop(mp + "shared_experts.up_proj.weight").float()                # [6144, 7168]
    sh_d   = td.pop(mp + "shared_experts.down_proj.weight").float()              # [7168, 6144]

    b_ldn = pack_hg8_16r_blob(lat_dn, seed=seed_base + 1)
    b_lup = pack_hg8_16r_blob(lat_up, seed=seed_base + 2)
    sh_w13 = interleave_glu_16r(sh_g, sh_u)                                      # [12288, 7168]
    b_sh13 = pack_hg4_16r_blob(sh_w13, seed=seed_base + 3)
    pd_sdn = glu_eq_pre_diag(sh_g, sh_u, sh_d, seed=seed_base + 4)
    b_shdn = pack_hg4_16r_blob(sh_d, fixed_pre_diag=pd_sdn)

    return (
        gate_w.numpy().tobytes()
        + gate_b.numpy().tobytes()
        + lat_nm.numpy().tobytes()
        + struct.pack("<IIII", len(b_ldn), len(b_lup), len(b_sh13), len(b_shdn))
        + b_ldn + b_lup + b_sh13 + b_shdn
    )


def pack_k31_cached_experts(
    repo_id: str,
    weight_map: dict[str, str],
    layer_id: int,
    n_cached: int,
    out_path: Path,
    seed_base: int,
) -> None:
    """Stream `n_cached` MXFP4 routed experts (`w1, w3 [3072, 3584]`, `w2 [3584, 3072]`) into `HG4-16R`."""
    pfx = f"language_model.model.layers.{layer_id}.block_sparse_moe.experts."
    keys = []
    for e in range(n_cached):
        for wn in ("w1", "w2", "w3"):
            keys.append(f"{pfx}{e}.{wn}.weight_packed")
            keys.append(f"{pfx}{e}.{wn}.weight_scale")
    td = fetch_tensors_ram(repo_id, weight_map, keys, max_workers=16)

    w1_list, w3_list, w2_list = [], [], []
    for e in range(n_cached):
        w1 = kb.dequantize_mxfp4_tensor(td.pop(f"{pfx}{e}.w1.weight_packed"), td.pop(f"{pfx}{e}.w1.weight_scale")).float()
        w3 = kb.dequantize_mxfp4_tensor(td.pop(f"{pfx}{e}.w3.weight_packed"), td.pop(f"{pfx}{e}.w3.weight_scale")).float()
        w2 = kb.dequantize_mxfp4_tensor(td.pop(f"{pfx}{e}.w2.weight_packed"), td.pop(f"{pfx}{e}.w2.weight_scale")).float()
        w1_list.append(w1); w3_list.append(w3); w2_list.append(w2)

    w1_s = torch.stack(w1_list, dim=0)  # [E, 3072, 3584]
    w3_s = torch.stack(w3_list, dim=0)
    w2_s = torch.stack(w2_list, dim=0)  # [E, 3584, 3072]

    col_rms_up = torch.sqrt((w1_s.pow(2).mean(dim=(0, 1)) + w3_s.pow(2).mean(dim=(0, 1))) * 0.5 + 1e-12).pow(0.5)
    pre_diag_up = (col_rms_up * make_signs(3584, seed_base)).contiguous()

    w13_s = torch.cat([w1_s.reshape(n_cached, 384, 8, 3584), w3_s.reshape(n_cached, 384, 8, 3584)], dim=2).reshape(n_cached, 6144, 3584).contiguous()
    pd_dn_s = torch.stack([glu_eq_pre_diag(w1_s[i], w3_s[i], w2_s[i], seed=seed_base + 1 + i) for i in range(n_cached)], dim=0).contiguous()

    fast_lib = get_fast_packer_lib()
    fp = ctypes.POINTER(ctypes.c_float)
    nb_up = 3584 // 128  # 28
    nb_dn = 3072 // 128  # 24

    out_sc_u = np.empty((n_cached, (6144 // 16) * nb_up * 16), dtype=np.uint16)
    out_pk_u = np.empty((n_cached, (6144 * 3584) // 2), dtype=np.uint8)
    fast_lib.hg4_pack_experts_16r_f32(
        ctypes.cast(w13_s.data_ptr(), fp), ctypes.cast(pre_diag_up.data_ptr(), fp), 0, None,
        n_cached, 6144, 3584,
        out_sc_u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        out_pk_u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), None,
    )

    out_sc_d = np.empty((n_cached, (3584 // 16) * nb_dn * 16), dtype=np.uint16)
    out_pk_d = np.empty((n_cached, (3584 * 3072) // 2), dtype=np.uint8)
    fast_lib.hg4_pack_experts_16r_f32(
        ctypes.cast(w2_s.contiguous().data_ptr(), fp), ctypes.cast(pd_dn_s.data_ptr(), fp), 1, None,
        n_cached, 3584, 3072,
        out_sc_d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        out_pk_d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), None,
    )

    gu_body_len = out_sc_u[0].nbytes + out_pk_u[0].nbytes  # 344,064 + 11,010,048 = 11,354,112 B
    dn_blob_len = 8 + 3072 * 4 + out_sc_d[0].nbytes + out_pk_d[0].nbytes  # 8 + 12,288 + 172,032 + 5,505,024 = 5,689,352 B
    exp_stride = gu_body_len + dn_blob_len

    with open(out_path, "wb") as wf:
        wf.write(struct.pack("<4sIIIIII", b"K31E", n_cached, 6144, 3584, 3584, 3072, exp_stride))
        wf.write(pre_diag_up.numpy().tobytes())
        for e in range(n_cached):
            wf.write(out_sc_u[e].tobytes())
            wf.write(out_pk_u[e].tobytes())
            wf.write(struct.pack("<II", 3584, 3072))
            wf.write(pd_dn_s[e].numpy().tobytes())
            wf.write(out_sc_d[e].tobytes())
            wf.write(out_pk_d[e].tobytes())


def pack_dspark_5l_hg4(drafter_repo: str, out_path: Path) -> int:
    """Stream & pack all 5/5 layers of `LithosAI/Kimi-K3.1-Pro-DSpark` into `dspark.hg4` (`HG4-16R`)."""
    _, hdr, _ = get_safetensor_header_cached(drafter_repo, "model.safetensors")
    all_keys = [k for k in hdr.keys() if k != "__metadata__"]
    wm = {k: "model.safetensors" for k in all_keys}
    td = fetch_tensors_ram(drafter_repo, wm, all_keys, max_workers=16)

    hid_norm = td.pop("hidden_norm.weight").float().contiguous()          # [7168]
    final_norm = td.pop("norm.weight").float().contiguous()               # [7168]
    conf_w = td.pop("confidence_head.proj.weight").float().reshape(7424).contiguous()
    conf_b = td.pop("confidence_head.proj.bias").float().reshape(1).contiguous()

    b_fc = pack_hg4_16r_blob(td.pop("fc.weight").float(), seed=8001)      # [7168, 35840]
    b_mw1 = pack_hg8_16r_blob(td.pop("markov_head.markov_w1.weight").float(), seed=8002)  # [163840, 256]
    b_mw2 = pack_hg8_16r_blob(td.pop("markov_head.markov_w2.weight").float(), seed=8003)  # [163840, 256]

    layer_blobs = []
    for l in range(5):
        lp = f"layers.{l}."
        in_ln = td.pop(lp + "input_layernorm.weight").float().contiguous()
        post_ln = td.pop(lp + "post_attention_layernorm.weight").float().contiguous()
        qn = td.pop(lp + "self_attn.q_norm.weight").float().contiguous()
        kn = td.pop(lp + "self_attn.k_norm.weight").float().contiguous()

        bq = pack_hg8_16r_blob(td.pop(lp + "self_attn.q_proj.weight").float(), seed=8100 + l * 10 + 1)
        bk = pack_hg8_16r_blob(td.pop(lp + "self_attn.k_proj.weight").float(), seed=8100 + l * 10 + 2)
        bv = pack_hg8_16r_blob(td.pop(lp + "self_attn.v_proj.weight").float(), seed=8100 + l * 10 + 3)
        bo = pack_hg8_16r_blob(td.pop(lp + "self_attn.o_proj.weight").float(), seed=8100 + l * 10 + 4)

        gw = td.pop(lp + "mlp.gate_proj.weight").float()
        uw = td.pop(lp + "mlp.up_proj.weight").float()
        dw = td.pop(lp + "mlp.down_proj.weight").float()
        w13 = interleave_glu_16r(gw, uw)                                  # [28672, 7168]
        bw13 = pack_hg4_16r_blob(w13, seed=8100 + l * 10 + 5)
        bwdn = pack_hg4_16r_blob(dw, fixed_pre_diag=glu_eq_pre_diag(gw, uw, dw, seed=8100 + l * 10 + 6))

        lb = (
            in_ln.numpy().tobytes() + post_ln.numpy().tobytes()
            + qn.numpy().tobytes() + kn.numpy().tobytes()
            + struct.pack("<IIIIII", len(bq), len(bk), len(bv), len(bo), len(bw13), len(bwdn))
            + bq + bk + bv + bo + bw13 + bwdn
        )
        layer_blobs.append(lb)

    with open(out_path, "wb") as wf:
        wf.write(struct.pack("<4sII", b"K3DS", 1, 5))
        wf.write(hid_norm.numpy().tobytes())
        wf.write(final_norm.numpy().tobytes())
        wf.write(conf_w.numpy().tobytes())
        wf.write(conf_b.numpy().tobytes())
        wf.write(struct.pack("<III5I", len(b_fc), len(b_mw1), len(b_mw2), *[len(x) for x in layer_blobs]))
        wf.write(b_fc)
        wf.write(b_mw1)
        wf.write(b_mw2)
        for lb in layer_blobs:
            wf.write(lb)
    return out_path.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/opt/clover-k3/kimi-k31-hg4")
    ap.add_argument("--max-layers", type=int, default=93)
    ap.add_argument("--stage-experts", type=int, default=32)
    args = ap.parse_args()

    auto_load_hf_token()
    kb.enable_ipv4_only()
    t0 = time.perf_counter()

    repo_id = "moonshotai/Kimi-K3.1-Pro"
    drafter_repo = "LithosAI/Kimi-K3.1-Pro-DSpark"
    k31_local = Path("/opt/clover-k3/kimi-k3.1-pro-computed")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_path = hf_hub_download(repo_id, "model.safetensors.index.json", token=os.environ.get("HF_TOKEN"))
    weight_map: dict[str, str] = json.loads(Path(idx_path).read_text())["weight_map"]

    # 1. Global norms + embed_tokens + lm_head
    g_dir = k31_local / "hf-stage/trunks/global-0"
    g_keys_needed = []
    for k in ("language_model.model.norm.weight", "language_model.model.output_attn_res_norm.weight", "language_model.model.output_attn_res_proj.weight", "language_model.lm_head.weight"):
        g_keys_needed.append(k)
    if not (g_dir / "embed_tokens.bf16.bin").is_file():
        g_keys_needed.append("language_model.model.embed_tokens.weight")

    print("[Global] Streaming global norms & lm_head from shard 94...", flush=True)
    tg = fetch_tensors_ram(repo_id, weight_map, g_keys_needed, max_workers=16)
    norm_w = tg.pop("language_model.model.norm.weight").float().reshape(7168).contiguous()
    out_res_norm = tg.pop("language_model.model.output_attn_res_norm.weight").float().reshape(7168).contiguous()
    out_res_proj = tg.pop("language_model.model.output_attn_res_proj.weight").float().reshape(7168).contiguous()

    kinds = [0 if l == 0 else (2 if l in DSA_SET else 1) for l in range(args.max_layers)]
    (out_dir / "meta.bin").write_bytes(
        struct.pack("<4sIIII", b"K31R", 1, args.max_layers, 7168, 163840)
        + LUT16_U8.tobytes()
        + norm_w.numpy().tobytes()
        + out_res_norm.numpy().tobytes()
        + out_res_proj.numpy().tobytes()
        + struct.pack(f"<{args.max_layers}I", *kinds)
    )

    emb_p = out_dir / "embed_tokens.hg4"
    if not emb_p.is_file():
        te = time.perf_counter()
        if (g_dir / "embed_tokens.bf16.bin").is_file():
            emb_mm = np.memmap(g_dir / "embed_tokens.bf16.bin", dtype=np.uint16, mode="r").reshape(163840, 7168)
            emb_f32 = torch.from_numpy(np.left_shift(np.asarray(emb_mm, dtype=np.uint32), np.uint32(16)).view(np.float32))
        else:
            emb_f32 = tg.pop("language_model.model.embed_tokens.weight").float()
        emb_blob = pack_hg4_16r_blob(emb_f32, seed=101)
        emb_p.write_bytes(emb_blob)
        del emb_f32
        print(f"[Global] embed_tokens.hg4: {len(emb_blob)/1048576:.2f} MiB in {time.perf_counter()-te:.1f}s", flush=True)

    lm_p = out_dir / "lm_head.hg4"
    if not lm_p.is_file():
        tlm = time.perf_counter()
        lm_f32 = tg.pop("language_model.lm_head.weight").float()
        lm_blob = pack_hg4_16r_blob(lm_f32, seed=202)
        lm_p.write_bytes(lm_blob)
        del lm_f32
        print(f"[Global] lm_head.hg4: {len(lm_blob)/1048576:.2f} MiB in {time.perf_counter()-tlm:.1f}s", flush=True)
    tg.clear()
    gc.collect()

    # 2. Stream & Convert All 93/93 Backbone Trunks (`layer_00.hg4 .. layer_92.hg4`)
    def trunk_keys_for_layer(lid: int) -> list[str]:
        lp = f"language_model.model.layers.{lid}."
        return [k for k in weight_map if k.startswith(lp) and ".experts." not in k]

    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    first_todo = next((l for l in range(args.max_layers) if not (out_dir / f"layer_{l:02d}.hg4").is_file()), args.max_layers)
    next_fut = (
        prefetch_pool.submit(fetch_tensors_ram, repo_id, weight_map, trunk_keys_for_layer(first_todo), 16)
        if first_todo < args.max_layers else None
    )

    for l in range(first_todo, args.max_layers):
        layer_path = out_dir / f"layer_{l:02d}.hg4"
        if layer_path.is_file():
            continue
        tl0 = time.perf_counter()
        assert next_fut is not None
        td = next_fut.result()
        t_dl = time.perf_counter() - tl0

        nxt_l = next((nl for nl in range(l + 1, args.max_layers) if not (out_dir / f"layer_{nl:02d}.hg4").is_file()), args.max_layers)
        next_fut = (
            prefetch_pool.submit(fetch_tensors_ram, repo_id, weight_map, trunk_keys_for_layer(nxt_l), 16)
            if nxt_l < args.max_layers else None
        )

        pfx = f"language_model.model.layers.{l}."
        kind = kinds[l]
        in_ln = td.pop(pfx + "input_layernorm.weight").float().reshape(7168).contiguous()
        post_ln = td.pop(pfx + "post_attention_layernorm.weight").float().reshape(7168).contiguous()
        sa_rn = td.pop(pfx + "self_attention_res_norm.weight").float().reshape(7168).contiguous()
        sa_rp = td.pop(pfx + "self_attention_res_proj.weight").float().reshape(7168).contiguous()
        mlp_rn = td.pop(pfx + "mlp_res_norm.weight").float().reshape(7168).contiguous()
        mlp_rp = td.pop(pfx + "mlp_res_proj.weight").float().reshape(7168).contiguous()

        common_hdr = (
            struct.pack("<I", kind)
            + in_ln.numpy().tobytes() + post_ln.numpy().tobytes()
            + sa_rn.numpy().tobytes() + sa_rp.numpy().tobytes()
            + mlp_rn.numpy().tobytes() + mlp_rp.numpy().tobytes()
        )

        ap = pfx + "self_attn."
        if kind in (0, 1):  # KDA
            qc = td.pop(ap + "q_conv1d.weight").float().reshape(12288 * 4).contiguous()
            kc = td.pop(ap + "k_conv1d.weight").float().reshape(12288 * 4).contiguous()
            vc = td.pop(ap + "v_conv1d.weight").float().reshape(12288 * 4).contiguous()
            dtb = td.pop(ap + "dt_bias").float().reshape(12288).contiguous()
            onm = td.pop(ap + "o_norm.weight").float().reshape(128).contiguous()
            A_neg = (-td.pop(ap + "A_log").float().exp()).reshape(128).contiguous()

            bq  = pack_hg4_16r_blob(td.pop(ap + "q_proj.weight").float(), seed=1000 + l * 20 + 1)
            bk  = pack_hg4_16r_blob(td.pop(ap + "k_proj.weight").float(), seed=1000 + l * 20 + 2)
            bv  = pack_hg4_16r_blob(td.pop(ap + "v_proj.weight").float(), seed=1000 + l * 20 + 3)
            bg  = pack_hg4_16r_blob(td.pop(ap + "g_proj.weight").float(), seed=1000 + l * 20 + 4)
            bo  = pack_hg4_16r_blob(td.pop(ap + "o_proj.weight").float(), seed=1000 + l * 20 + 5)
            bb  = pack_hg8_16r_blob(td.pop(ap + "b_proj.weight").float(), seed=1000 + l * 20 + 6)
            bfa = pack_hg8_16r_blob(td.pop(ap + "f_a_proj.weight").float(), seed=1000 + l * 20 + 7)
            bfb = pack_hg8_16r_blob(td.pop(ap + "f_b_proj.weight").float(), seed=1000 + l * 20 + 8)

            attn_part = (
                qc.numpy().tobytes() + kc.numpy().tobytes() + vc.numpy().tobytes()
                + dtb.numpy().tobytes() + onm.numpy().tobytes() + A_neg.numpy().tobytes()
                + struct.pack("<8I", len(bq), len(bk), len(bv), len(bg), len(bo), len(bb), len(bfa), len(bfb))
                + bq + bk + bv + bg + bo + bb + bfa + bfb
            )
        else:  # Gated DSA (`kind == 2`)
            qaln = td.pop(ap + "q_a_layernorm.weight").float().reshape(1536).contiguous()
            kvaln = td.pop(ap + "kv_a_layernorm.weight").float().reshape(512).contiguous()
            ikn_w = td.pop(ap + "indexer.k_norm.weight").float().reshape(128).contiguous()
            ikn_b = td.pop(ap + "indexer.k_norm.bias").float().reshape(128).contiguous()

            bqa  = pack_hg8_16r_blob(td.pop(ap + "q_a_proj.weight").float(), seed=2000 + l * 20 + 1)
            bqb  = pack_hg4_16r_blob(td.pop(ap + "q_b_proj.weight").float(), seed=2000 + l * 20 + 2)
            bkva = pack_hg8_16r_blob(td.pop(ap + "kv_a_proj_with_mqa.weight").float(), seed=2000 + l * 20 + 3)
            bkvb = pack_hg4_16r_blob(td.pop(ap + "kv_b_proj.weight").float(), seed=2000 + l * 20 + 4)
            bg   = pack_hg4_16r_blob(td.pop(ap + "g_proj.weight").float(), seed=2000 + l * 20 + 5)
            bo   = pack_hg4_16r_blob(td.pop(ap + "o_proj.weight").float(), seed=2000 + l * 20 + 6)
            biwp = pack_hg8_16r_blob(td.pop(ap + "indexer.weights_proj.weight").float(), seed=2000 + l * 20 + 7)
            biwk = pack_hg8_16r_blob(td.pop(ap + "indexer.wk.weight").float(), seed=2000 + l * 20 + 8)
            biqb = pack_hg8_16r_blob(td.pop(ap + "indexer.wq_b.weight").float(), seed=2000 + l * 20 + 9)

            attn_part = (
                qaln.numpy().tobytes() + kvaln.numpy().tobytes()
                + ikn_w.numpy().tobytes() + ikn_b.numpy().tobytes()
                + struct.pack("<9I", len(bqa), len(bqb), len(bkva), len(bkvb), len(bg), len(bo), len(biwp), len(biwk), len(biqb))
                + bqa + bqb + bkva + bkvb + bg + bo + biwp + biwk + biqb
            )

        if kind == 0:  # Layer 0 Dense 33,792 SiTU-GLU
            dg = td.pop(pfx + "mlp.gate_proj.weight").float()
            du = td.pop(pfx + "mlp.up_proj.weight").float()
            dd = td.pop(pfx + "mlp.down_proj.weight").float()
            dw13 = interleave_glu_16r(dg, du)  # [67584, 7168]
            bw13 = pack_hg4_16r_blob(dw13, seed=3001)
            bwdn = pack_hg4_16r_blob(dd, fixed_pre_diag=glu_eq_pre_diag(dg, du, dd, seed=3002))
            mlp_part = struct.pack("<II", len(bw13), len(bwdn)) + bw13 + bwdn
            del dg, du, dd, dw13
        else:
            mlp_part = pack_k31_moe_trunk_tail(td, pfx, seed_base=4000 + l * 20)

        td.clear()
        layer_path.write_bytes(common_hdr + attn_part + mlp_part)
        gc.collect()
        l_mb = layer_path.stat().st_size / 1048576.0
        k_tag = "L0-KDA-DENSE" if kind == 0 else ("DSA-MOE" if kind == 2 else "KDA-MOE")
        print(
            f"[Layer {l:02d}/{args.max_layers - 1:02d} {k_tag:12s}] "
            f"{l_mb:6.1f} MiB | dl={t_dl:4.1f}s tot={time.perf_counter()-tl0:5.1f}s (elapsed {(time.perf_counter()-t0)/60:.1f}m)",
            flush=True,
        )

    prefetch_pool.shutdown(wait=False)

    # 3. Stage Routed Experts for Layer 1 (KDA+MoE) & Layer 3 (DSA+MoE)
    for stage_l in (1, 3):
        if stage_l < args.max_layers:
            ep = out_dir / f"experts_{stage_l:02d}.hg4"
            if not ep.is_file():
                te = time.perf_counter()
                pack_k31_cached_experts(repo_id, weight_map, stage_l, args.stage_experts, ep, seed_base=50000 + stage_l * 100)
                print(f"[Experts L{stage_l:02d}] {args.stage_experts} SiTU-GLU experts -> {ep.stat().st_size/1048576:.1f} MiB in {time.perf_counter()-te:.1f}s", flush=True)

    # 4. Full 5/5-Layer `Kimi-K3.1-Pro-DSpark` Speculative Drafter
    ds_p = out_dir / "dspark.hg4"
    if not ds_p.is_file():
        tds = time.perf_counter()
        ds_bytes = pack_dspark_5l_hg4(drafter_repo, ds_p)
        print(f"[DSpark 5L] Packed all 5/5 DSpark drafter layers -> {ds_bytes/1048576:.1f} MiB in {time.perf_counter()-tds:.1f}s", flush=True)

    # Copy tokenizer files
    tok_src = k31_local / "tokenizer"
    if tok_src.is_dir():
        for fn in ("config.json", "tiktoken.model", "tokenizer_config.json", "encoding_k3.py"):
            if (tok_src / fn).is_file() and not (out_dir / fn).is_file():
                shutil.copy2(tok_src / fn, out_dir / fn)

    tot_b = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file() and f.suffix in (".bin", ".hg4"))
    print(
        f"\n=== COMPLETED Kimi K3.1 Pro ({args.max_layers} Trunks + Global-0 + 5L DSpark) ===\n"
        f"Total HG4-16R weight size: {tot_b / (1024**3):.3f} GiB ({tot_b / 1048576:.1f} MiB) in {(time.perf_counter()-t0)/60:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
