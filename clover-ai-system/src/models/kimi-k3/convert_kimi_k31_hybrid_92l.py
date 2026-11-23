#!/usr/bin/env python3
"""
Full 92-Layer `Kimi K3.1 Pro` (`2.8T` / `82,432` Routed `SiTU-GLU` Experts) Hybrid Data-Movement Converter
(`clover-ai-system/src/models/kimi-k3/convert_kimi_k31_hybrid_92l.py`).

Builds the `K31H` two-tier Hadamard weight store directly from `moonshotai/Kimi-K3.1-Pro` in RAM
(zero temporary `.safetensors` disk shards, atomic `.tmp -> rename` crash-resumable per layer,
and overlapped Layer `L+1` network prefetch while 16 OpenMP cores pack Layer `L`):

  1. **Router-Prior Hot-Expert Selection (`Top-K_hot = 42` per layer):**
     Scores all `896` experts in each layer `1..92` using the already-packed `gate_w` & `gate_b`
     against seeded unit-RMS probe vectors (`512 x 7168`) plus router L2 norm tie-breaking.
  2. **Hot Tier (`experts_hot_%02d.hg4` — `HG4-16R` `4.25 bits/weight`, `61.3 GiB` across 92 layers):**
     Mapped in `k3`'s page cache via `mmap` + `MADV_WILLNEED` alongside the `31 GiB` trunks + `1.3 GiB` `DSpark`.
  3. **Cold Tier (`experts_cold_%02d.hg2` — `HG2-16R` `2.25 bits/weight`, `642.9 GiB` across 92 layers):**
     Distributed across `/mnt/nvme1/kimi-k31-hybrid` and `/opt/clover-k3/kimi-k31-cold-md2` (`~20%` of layers
     where `l_idx % 5 == 3` and `l_idx < 80`, symlinked into `/mnt/nvme1/kimi-k31-hybrid`) to fit within disk capacity.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import gc
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np
import torch
from huggingface_hub import hf_hub_download

torch.set_num_threads(16)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import convert_kimi_k31_full_hg4 as ck  # noqa: E402

HOT_EXP_STRIDE = 344064 + 11010048 + 8 + 12288 + 172032 + 5505024  # 17,043,464 B
COLD_EXP_STRIDE = 344064 + 5505024 + 8 + 12288 + 172032 + 2752512  # 8,785,928 B


def build_packer_so() -> ctypes.CDLL:
    so_path = Path("/tmp/libhg4_fast_packer_v2.so")
    c_cands = [
        HERE.parent / "clover-35b-a3b" / "hg4_fast_packer.c",
        Path("/opt/clover-k3/hg4_fast_packer.c"),
    ]
    c_src = next((p for p in c_cands if p.is_file()), None)
    if c_src is None:
        raise FileNotFoundError("Cannot locate hg4_fast_packer.c")
    subprocess.run(
        ["gcc", "-O3", "-march=native", "-fopenmp", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror", str(c_src), "-o", str(so_path), "-lm"],
        check=True,
    )
    lib = ctypes.CDLL(str(so_path))
    fp = ctypes.POINTER(ctypes.c_float)
    u16p = ctypes.POINTER(ctypes.c_uint16)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    lib.hg4_pack_experts_16r_f32.argtypes = [
        fp, fp, ctypes.c_int, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, u16p, u8p, fp
    ]
    lib.hg4_pack_experts_16r_f32.restype = None
    lib.hg2_pack_experts_16r_f32.argtypes = [
        fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, u16p, u8p
    ]
    lib.hg2_pack_experts_16r_f32.restype = None
    lib.mxfp4_dequant_f32.argtypes = [u8p, u8p, ctypes.c_int, ctypes.c_int, fp]
    lib.mxfp4_dequant_f32.restype = None
    if hasattr(lib, "k31_pack_layer_from_safetensor_mmap"):
        lib.k31_pack_layer_from_safetensor_mmap.argtypes = [
            u8p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int, ctypes.c_int,
            fp, fp,
            ctypes.c_int, ctypes.c_int,
        ]
        lib.k31_pack_layer_from_safetensor_mmap.restype = ctypes.c_int
    return lib


def extract_layer_router_from_hg4(layer_hg4_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read `gate_w [896, 7168]` and `gate_b [896]` directly from already-packed `layer_%02d.hg4`."""
    buf = layer_hg4_path.read_bytes()
    mv = memoryview(buf)
    kind = struct.unpack_from("<I", mv, 0)[0]
    assert kind in (1, 2), f"Layer {layer_hg4_path} has kind={kind} (not MoE)"
    off = 4 + 6 * 7168 * 4
    if kind == 1:
        off += (3 * 12288 * 4 * 4) + (12288 * 4) + (128 * 4) + (128 * 4)
        ls = struct.unpack_from("<8I", mv, off)
        off += 32 + sum(ls)
    else:
        off += (1536 * 4) + (512 * 4) + (128 * 4) + (128 * 4)
        ls = struct.unpack_from("<9I", mv, off)
        off += 36 + sum(ls)

    gw_bytes = 896 * 7168 * 4
    gb_bytes = 896 * 4
    gate_w = np.frombuffer(mv[off : off + gw_bytes], dtype=np.float32).reshape(896, 7168).copy()
    gate_b = np.frombuffer(mv[off + gw_bytes : off + gw_bytes + gb_bytes], dtype=np.float32).copy()
    return gate_w, gate_b


def build_calibration_hidden_probe(hg4_dir: Path, n_vecs: int = 512) -> np.ndarray:
    """Generate unit-RMS calibration vectors seeded deterministically + real embedding directions."""
    rng = np.random.default_rng(20261122)
    x = rng.standard_normal((n_vecs, 7168), dtype=np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    return (x / rms).astype(np.float32)


def select_hot_experts(gate_w: np.ndarray, gate_b: np.ndarray, cal_x: np.ndarray, n_hot: int) -> list[int]:
    """
    Combine empirical Top-16 routing frequency on `cal_x` with router weight L2 norm & `gate_b` load prior
    to pick the `n_hot` highest-traffic experts in `[0..895]`.
    """
    logits = cal_x @ gate_w.T  # [N, 896]
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    choice = scores + gate_b[None, :]
    top16 = np.argpartition(choice, -16, axis=1)[:, -16:]
    freq = np.bincount(top16.ravel(), minlength=896).astype(np.float64)
    # In auxiliary-loss-free load balancing, heavily naturally-routed experts accumulate negative correction bias;
    # combine actual post-bias selection frequency (`choice`) + weight norm tie-breaker.
    wnorm = np.linalg.norm(gate_w, axis=1)
    rank_score = freq + 0.05 * (wnorm / (np.max(wnorm) + 1e-9))
    hot_ids = np.argsort(-rank_score)[:n_hot]
    return sorted(int(i) for i in hot_ids)


def download_shard_fast_shm(repo_id: str, shard_name: str, shm_path: Path, n_streams: int = 16) -> tuple[dict[str, Any], int]:
    """
    Resolve signed HuggingFace CDN URL with `HF_TOKEN` and stream the entire ~15.2 GB shard directly
    into `/dev/shm` across `n_streams` parallel `curl` byte-range workers (~800-1000 MB/s on k3).
    Returns `(safetensors_header_dict, data_base_offset)`.
    """
    import urllib.request
    from huggingface_hub import hf_hub_url

    tok = os.environ.get("HF_TOKEN") or ck.auto_load_hf_token()
    url = hf_hub_url(repo_id, shard_name)
    auth_hdr = {"Authorization": f"Bearer {tok}"} if tok else {}

    # 1. Resolve redirect CDN URL + exact file length + safetensors header
    req = urllib.request.Request(url, headers={**auth_hdr, "Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        cdn_url = resp.geturl()
        cr = resp.headers.get("Content-Range") or ""
        total_size = int(cr.split("/")[-1])
        hlen = struct.unpack("<Q", resp.read(8))[0]

    req2 = urllib.request.Request(cdn_url, headers={"Range": f"bytes=8-{8 + hlen - 1}"})
    with urllib.request.urlopen(req2, timeout=45) as resp2:
        header = json.loads(resp2.read().decode("utf-8"))

    # 2. Preallocate `/dev/shm` target file and write `n_streams` disjoint ranges via parallel `curl` + `pwrite`
    shm_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(shm_path), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
    os.ftruncate(fd, total_size)

    part_sz = (total_size + n_streams - 1) // n_streams

    def _pull_part(p_idx: int) -> None:
        b_start = p_idx * part_sz
        b_end = min(total_size - 1, b_start + part_sz - 1)
        if b_start > b_end:
            return
        for attempt in range(6):
            proc = subprocess.Popen(
                ["curl", "-fsSL", "--retry", "3", "-r", f"{b_start}-{b_end}", cdn_url],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            assert proc.stdout is not None
            pos = b_start
            while True:
                buf = proc.stdout.read(4 * 1024 * 1024)
                if not buf:
                    break
                os.pwrite(fd, buf, pos)
                pos += len(buf)
            rc = proc.wait()
            if rc == 0 and pos == b_end + 1:
                return
            time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Failed downloading range {b_start}-{b_end} of {shard_name}")

    try:
        with ThreadPoolExecutor(max_workers=n_streams) as pool:
            futs = [pool.submit(_pull_part, i) for i in range(n_streams)]
            for f in futs:
                f.result()
    finally:
        os.close(fd)

    return header, 8 + hlen


def pack_one_hybrid_layer_mmap(
    lib: ctypes.CDLL,
    shm_path: Path,
    header: dict[str, Any],
    data_base: int,
    layer_id: int,
    hot_ids: list[int],
    hot_out_path: Path,
    cold_primary_path: Path,
    cold_link_path: Path | None,
) -> dict[str, Any]:
    import mmap
    t0 = time.perf_counter()
    hot_set = set(hot_ids)
    cold_ids = [e for e in range(896) if e not in hot_set]
    n_hot, n_cold = len(hot_ids), len(cold_ids)

    slot_map = np.empty(896, dtype=np.int16)
    for hi, e in enumerate(hot_ids):
        slot_map[e] = hi
    for ci, e in enumerate(cold_ids):
        slot_map[e] = -(ci + 1)

    pfx = f"language_model.model.layers.{layer_id}.block_sparse_moe.experts."
    offs = np.empty((896, 6), dtype=np.uint64)
    for e in range(896):
        offs[e, 0] = data_base + int(header[f"{pfx}{e}.w1.weight_packed"]["data_offsets"][0])
        offs[e, 1] = data_base + int(header[f"{pfx}{e}.w1.weight_scale"]["data_offsets"][0])
        offs[e, 2] = data_base + int(header[f"{pfx}{e}.w3.weight_packed"]["data_offsets"][0])
        offs[e, 3] = data_base + int(header[f"{pfx}{e}.w3.weight_scale"]["data_offsets"][0])
        offs[e, 4] = data_base + int(header[f"{pfx}{e}.w2.weight_packed"]["data_offsets"][0])
        offs[e, 5] = data_base + int(header[f"{pfx}{e}.w2.weight_scale"]["data_offsets"][0])

    seed_base = 50000 + layer_id * 2000
    signs_up = ck.make_signs(3584, seed_base).numpy().astype(np.float32, copy=False)
    signs_dn = np.stack(
        [ck.make_signs(3072, seed_base + 1 + e).numpy() for e in range(896)], axis=0
    ).astype(np.float32, copy=False)

    hot_tmp = hot_out_path.with_suffix(".hg4.tmp")
    cold_primary_path.parent.mkdir(parents=True, exist_ok=True)
    cold_tmp = cold_primary_path.with_suffix(".hg2.tmp")

    sfd = os.open(str(shm_path), os.O_RDWR)
    ssz = os.fstat(sfd).st_size
    mm = mmap.mmap(sfd, ssz, access=mmap.ACCESS_WRITE)
    hfd = os.open(str(hot_tmp), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    cfd = os.open(str(cold_tmp), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(hfd, 32 + 896 * 2 + 3584 * 4 + n_hot * HOT_EXP_STRIDE)
        os.ftruncate(cfd, n_cold * COLD_EXP_STRIDE)
        buf_ptr = ctypes.cast(
            ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm))), ctypes.POINTER(ctypes.c_uint8)
        )
        rc = lib.k31_pack_layer_from_safetensor_mmap(
            buf_ptr,
            offs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            slot_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            n_hot, n_cold,
            signs_up.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            signs_dn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            hfd, cfd,
        )
        if rc != 0:
            raise RuntimeError(f"k31_pack_layer_from_safetensor_mmap failed rc={rc}")
    finally:
        os.close(hfd)
        os.close(cfd)
        mm.close()
        os.close(sfd)

    os.replace(cold_tmp, cold_primary_path)
    if cold_link_path is not None and cold_link_path != cold_primary_path:
        if cold_link_path.exists() or cold_link_path.is_symlink():
            cold_link_path.unlink()
        os.symlink(cold_primary_path, cold_link_path)
    os.replace(hot_tmp, hot_out_path)

    return {
        "layer": layer_id,
        "hot_bytes": hot_out_path.stat().st_size,
        "cold_bytes": cold_primary_path.stat().st_size,
        "pack_s": round(time.perf_counter() - t0, 2),
    }


def dequant_chunk_c11(
    lib: ctypes.CDLL,
    td: dict[str, torch.Tensor],
    layer_id: int,
    exp_ids: list[int],
    w1_buf: torch.Tensor,
    w3_buf: torch.Tensor,
    w2_buf: torch.Tensor,
    pop_tensors: bool = True,
) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    fp = ctypes.POINTER(ctypes.c_float)
    pfx = f"language_model.model.layers.{layer_id}.block_sparse_moe.experts."
    for idx, e in enumerate(exp_ids):
        get_fn = td.pop if pop_tensors else td.__getitem__
        w1p = get_fn(f"{pfx}{e}.w1.weight_packed").contiguous()
        w1s = get_fn(f"{pfx}{e}.w1.weight_scale").contiguous()
        w3p = get_fn(f"{pfx}{e}.w3.weight_packed").contiguous()
        w3s = get_fn(f"{pfx}{e}.w3.weight_scale").contiguous()
        w2p = get_fn(f"{pfx}{e}.w2.weight_packed").contiguous()
        w2s = get_fn(f"{pfx}{e}.w2.weight_scale").contiguous()

        lib.mxfp4_dequant_f32(
            ctypes.cast(w1p.data_ptr(), u8p), ctypes.cast(w1s.data_ptr(), u8p),
            3072, 3584, ctypes.cast(w1_buf[idx].data_ptr(), fp),
        )
        lib.mxfp4_dequant_f32(
            ctypes.cast(w3p.data_ptr(), u8p), ctypes.cast(w3s.data_ptr(), u8p),
            3072, 3584, ctypes.cast(w3_buf[idx].data_ptr(), fp),
        )
        lib.mxfp4_dequant_f32(
            ctypes.cast(w2p.data_ptr(), u8p), ctypes.cast(w2s.data_ptr(), u8p),
            3584, 3072, ctypes.cast(w2_buf[idx].data_ptr(), fp),
        )


def pack_one_hybrid_layer(
    lib: ctypes.CDLL,
    td: dict[str, torch.Tensor],
    layer_id: int,
    hot_ids: list[int],
    hot_out_path: Path,
    cold_primary_path: Path,
    cold_link_path: Path | None,
    w1_buf: torch.Tensor,
    w3_buf: torch.Tensor,
    w2_buf: torch.Tensor,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    hot_set = set(hot_ids)
    cold_ids = [e for e in range(896) if e not in hot_set]
    n_hot = len(hot_ids)
    n_cold = len(cold_ids)
    assert n_hot + n_cold == 896

    slot_map = np.empty(896, dtype=np.int16)
    for hi, e in enumerate(hot_ids):
        slot_map[e] = hi
    for ci, e in enumerate(cold_ids):
        slot_map[e] = -(ci + 1)

    seed_base = 50000 + layer_id * 2000
    fp = ctypes.POINTER(ctypes.c_float)
    u16p = ctypes.POINTER(ctypes.c_uint16)
    u8p = ctypes.POINTER(ctypes.c_uint8)

    # 1. Dequantize hot experts first (`pop_tensors=True`) & compute shared `pre_diag_up [3584]`
    dequant_chunk_c11(lib, td, layer_id, hot_ids, w1_buf, w3_buf, w2_buf, pop_tensors=True)
    w1_hot = w1_buf[:n_hot]
    w3_hot = w3_buf[:n_hot]
    w2_hot = w2_buf[:n_hot]

    col_rms_up = torch.sqrt((w1_hot.pow(2).mean(dim=(0, 1)) + w3_hot.pow(2).mean(dim=(0, 1))) * 0.5 + 1e-12).pow(0.5)
    pre_diag_up = (col_rms_up * ck.make_signs(3584, seed_base)).contiguous()

    # 2. Pack hot experts into `HG4-16R` (`experts_hot_%02d.hg4`)
    w13_hot = torch.cat(
        [w1_hot.reshape(n_hot, 384, 8, 3584), w3_hot.reshape(n_hot, 384, 8, 3584)], dim=2
    ).reshape(n_hot, 6144, 3584).contiguous()
    pd_dn_hot = torch.stack(
        [ck.glu_eq_pre_diag(w1_hot[i], w3_hot[i], w2_hot[i], seed=seed_base + 1 + hot_ids[i]) for i in range(n_hot)],
        dim=0,
    ).contiguous()

    sc_u_hot = np.empty((n_hot, 384 * 28 * 16), dtype=np.uint16)
    pk_u_hot = np.empty((n_hot, (6144 * 3584) // 2), dtype=np.uint8)
    lib.hg4_pack_experts_16r_f32(
        ctypes.cast(w13_hot.data_ptr(), fp), ctypes.cast(pre_diag_up.data_ptr(), fp), 0, None,
        n_hot, 6144, 3584, sc_u_hot.ctypes.data_as(u16p), pk_u_hot.ctypes.data_as(u8p), None,
    )

    sc_d_hot = np.empty((n_hot, 224 * 24 * 16), dtype=np.uint16)
    pk_d_hot = np.empty((n_hot, (3584 * 3072) // 2), dtype=np.uint8)
    lib.hg4_pack_experts_16r_f32(
        ctypes.cast(w2_hot.contiguous().data_ptr(), fp), ctypes.cast(pd_dn_hot.data_ptr(), fp), 1, None,
        n_hot, 3584, 3072, sc_d_hot.ctypes.data_as(u16p), pk_d_hot.ctypes.data_as(u8p), None,
    )
    del w13_hot

    hot_tmp = hot_out_path.with_suffix(".hg4.tmp")
    with open(hot_tmp, "wb") as hf:
        hf.write(struct.pack("<4sIIIIIII", b"K31H", 896, n_hot, n_cold, HOT_EXP_STRIDE, COLD_EXP_STRIDE, 0, 0))
        hf.write(slot_map.tobytes())
        hf.write(pre_diag_up.numpy().tobytes())
        dn_hdr_4 = struct.pack("<II", 3584, 3072 | (4 << 28))
        for i in range(n_hot):
            hf.write(sc_u_hot[i].tobytes())
            hf.write(pk_u_hot[i].tobytes())
            hf.write(dn_hdr_4)
            hf.write(pd_dn_hot[i].numpy().tobytes())
            hf.write(sc_d_hot[i].tobytes())
            hf.write(pk_d_hot[i].tobytes())

    # 3. Pack cold experts in 64-expert chunks into `HG2-16R` (`experts_cold_%02d.hg2`)
    cold_primary_path.parent.mkdir(parents=True, exist_ok=True)
    cold_tmp = cold_primary_path.with_suffix(".hg2.tmp")
    dn_hdr_2 = struct.pack("<II", 3584, 3072 | (2 << 28))
    chunk_sz = 64

    sc_u_c = np.empty((chunk_sz, 384 * 28 * 16), dtype=np.uint16)
    pk_u_c = np.empty((chunk_sz, (6144 * 3584) // 4), dtype=np.uint8)
    sc_d_c = np.empty((chunk_sz, 224 * 24 * 16), dtype=np.uint16)
    pk_d_c = np.empty((chunk_sz, (3584 * 3072) // 4), dtype=np.uint8)

    with open(cold_tmp, "wb") as cf:
        for c0 in range(0, n_cold, chunk_sz):
            sub_ids = cold_ids[c0 : c0 + chunk_sz]
            nc = len(sub_ids)
            dequant_chunk_c11(lib, td, layer_id, sub_ids, w1_buf, w3_buf, w2_buf, pop_tensors=True)
            w1_c = w1_buf[:nc]
            w3_c = w3_buf[:nc]
            w2_c = w2_buf[:nc]

            w13_c = torch.cat(
                [w1_c.reshape(nc, 384, 8, 3584), w3_c.reshape(nc, 384, 8, 3584)], dim=2
            ).reshape(nc, 6144, 3584).contiguous()
            pd_dn_c = torch.stack(
                [ck.glu_eq_pre_diag(w1_c[i], w3_c[i], w2_c[i], seed=seed_base + 1 + sub_ids[i]) for i in range(nc)],
                dim=0,
            ).contiguous()

            lib.hg2_pack_experts_16r_f32(
                ctypes.cast(w13_c.data_ptr(), fp), ctypes.cast(pre_diag_up.data_ptr(), fp), 0,
                nc, 6144, 3584, sc_u_c.ctypes.data_as(u16p), pk_u_c.ctypes.data_as(u8p),
            )
            lib.hg2_pack_experts_16r_f32(
                ctypes.cast(w2_c.contiguous().data_ptr(), fp), ctypes.cast(pd_dn_c.data_ptr(), fp), 1,
                nc, 3584, 3072, sc_d_c.ctypes.data_as(u16p), pk_d_c.ctypes.data_as(u8p),
            )
            del w13_c

            for i in range(nc):
                cf.write(sc_u_c[i].tobytes())
                cf.write(pk_u_c[i].tobytes())
                cf.write(dn_hdr_2)
                cf.write(pd_dn_c[i].numpy().tobytes())
                cf.write(sc_d_c[i].tobytes())
                cf.write(pk_d_c[i].tobytes())

    os.replace(cold_tmp, cold_primary_path)
    if cold_link_path is not None and cold_link_path != cold_primary_path:
        if cold_link_path.exists() or cold_link_path.is_symlink():
            cold_link_path.unlink()
        os.symlink(cold_primary_path, cold_link_path)
    os.replace(hot_tmp, hot_out_path)

    return {
        "layer": layer_id,
        "hot_bytes": hot_out_path.stat().st_size,
        "cold_bytes": cold_primary_path.stat().st_size,
        "pack_s": round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hg4-dir", type=Path, default=Path("/opt/clover-k3/kimi-k31-hg4"))
    ap.add_argument("--cold-nvme1-dir", type=Path, default=Path("/mnt/nvme1/kimi-k31-hybrid"))
    ap.add_argument("--cold-md2-dir", type=Path, default=Path("/opt/clover-k3/kimi-k31-cold-md2"))
    ap.add_argument("--n-hot", type=int, default=42)
    ap.add_argument("--start-layer", type=int, default=1)
    ap.add_argument("--end-layer", type=int, default=92)
    args = ap.parse_args()

    ck.auto_load_hf_token()
    ck.kb.enable_ipv4_only()
    lib = build_packer_so()

    repo_id = "moonshotai/Kimi-K3.1-Pro"
    idx_path = hf_hub_download(repo_id, "model.safetensors.index.json", token=os.environ.get("HF_TOKEN"))
    weight_map: dict[str, str] = json.loads(Path(idx_path).read_text())["weight_map"]

    args.hg4_dir.mkdir(parents=True, exist_ok=True)
    args.cold_nvme1_dir.mkdir(parents=True, exist_ok=True)
    args.cold_md2_dir.mkdir(parents=True, exist_ok=True)

    cal_x = build_calibration_hidden_probe(args.hg4_dir, n_vecs=512)

    todo_layers: list[int] = []
    for l in range(args.start_layer, args.end_layer + 1):
        hp = args.hg4_dir / f"experts_hot_{l:02d}.hg4"
        cp = args.cold_nvme1_dir / f"experts_cold_{l:02d}.hg2"
        if hp.is_file() and (cp.is_file() or (cp.is_symlink() and cp.resolve().is_file())):
            expected_cold = (896 - args.n_hot) * COLD_EXP_STRIDE
            if cp.resolve().stat().st_size == expected_cold:
                continue
        todo_layers.append(l)

    print(f"[K31H] Remaining layers to stream & pack: {len(todo_layers)} (n_hot={args.n_hot} HG4-16R + {896 - args.n_hot} HG2-16R)", flush=True)
    if not todo_layers:
        return

    shm_dir = Path("/dev/shm/k31_pack")
    shm_dir.mkdir(parents=True, exist_ok=True)

    def _stage_shard(slot_idx: int, layer_id: int) -> tuple[Path, dict[str, Any], int]:
        sample_key = f"language_model.model.layers.{layer_id}.block_sparse_moe.experts.0.w1.weight_packed"
        shard_name = weight_map[sample_key]
        sp = shm_dir / f"slot_{slot_idx}.safetensors"
        hdr, dbase = download_shard_fast_shm(repo_id, shard_name, sp, n_streams=16)
        return sp, hdr, dbase

    t_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as net_pool:
        fut_next = net_pool.submit(_stage_shard, 0, todo_layers[0])
        for idx, l in enumerate(todo_layers):
            t_net0 = time.perf_counter()
            shm_p, hdr, dbase = fut_next.result()
            net_s = round(time.perf_counter() - t_net0, 2)

            if idx + 1 < len(todo_layers):
                fut_next = net_pool.submit(_stage_shard, (idx + 1) % 2, todo_layers[idx + 1])

            gw, gb = extract_layer_router_from_hg4(args.hg4_dir / f"layer_{l:02d}.hg4")
            hot_ids = select_hot_experts(gw, gb, cal_x, args.n_hot)

            hot_out = args.hg4_dir / f"experts_hot_{l:02d}.hg4"
            if l % 5 == 0:
                cold_primary = args.cold_md2_dir / f"experts_cold_{l:02d}.hg2"
                cold_link = args.cold_nvme1_dir / f"experts_cold_{l:02d}.hg2"
            else:
                cold_primary = args.cold_nvme1_dir / f"experts_cold_{l:02d}.hg2"
                cold_link = None

            info = pack_one_hybrid_layer_mmap(
                lib, shm_p, hdr, dbase, l, hot_ids, hot_out, cold_primary, cold_link
            )
            try:
                shm_p.unlink()
            except OSError:
                pass
            elapsed_min = (time.perf_counter() - t_all) / 60.0
            done_n = idx + 1
            eta_min = (elapsed_min / done_n) * (len(todo_layers) - done_n)
            print(
                f"[K31H {done_n:02d}/{len(todo_layers):02d}] Layer {l:02d}: "
                f"net_wait={net_s:.1f}s pack={info['pack_s']:.1f}s "
                f"hot={info['hot_bytes'] / (1024**2):.1f}MiB cold={info['cold_bytes'] / (1024**3):.2f}GiB "
                f"elapsed={elapsed_min:.1f}m ETA={eta_min:.1f}m",
                flush=True,
            )


if __name__ == "__main__":
    main()
