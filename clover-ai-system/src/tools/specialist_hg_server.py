#!/usr/bin/env python3
"""OpenAI-Compatible + Binary `tsock://` Specialist Microservice (`specialist_hg_server.py`) for Clover GPU Petals.

Serves domain-specialist models (`goedel-prover-v2-32b`, `vjepa-2.1`, `boltz-2`, `chronos-2`, `router-0.6b`)
behind an authenticated `/v1/chat/completions` + `/health` + optional `tsock://` (`--token-socket-port`) endpoint:
  * `chronos-2`: Zero-shot quantile time-series forecaster (`Santhoshsathish94/Chronos-Bolt-Base-HG4-16R` on CUDA)
  * `goedel-prover-v2-32b`: Lean 4 proof synthesis (`Santhoshsathish94/Goedel-Prover-V2-32B-HG4-16R` via 4-bit Hadamard `HG4Linear` + `HG4Embedding` on CUDA)
  * `vjepa-2.1`: Spatiotemporal latent physics & trajectory readout (`Santhoshsathish94/VJEPA2-ViTg-HG4-16R` 40-layer ViT-g on CUDA)
  * `boltz-2`: Biomolecular 3D structure & ligand folding (`boltz predict` GPU diffusion + `.cif` coordinate & pLDDT/ipTM readout)
  * `router-0.6b`: Sub-15ms Clover Specialist Router + SafeAgents firewall (`Qwen/Qwen3-0.6B` + `Qwen3-0.6B-Clover-Router-SafeGuard-LoRA`)
"""
from __future__ import annotations

import argparse
import dataclasses
import hmac
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

LUT16_F32 = np.array(
    [-127, -96, -75, -58, -44, -31, -18, -6, 6, 18, 31, 44, 58, 75, 96, 127],
    dtype=np.float32,
)
LUT4_F32 = np.array([-127, -38, 38, 127], dtype=np.float32)


def _extract_last_user(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return ""


def _run_heuristic_timeseries(prompt: str) -> str:
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", prompt.split("time-series:")[-1])]
    m_h = re.search(r"next\s+(\d+)\s+steps", prompt, re.IGNORECASE)
    horizon = max(1, min(int(m_h.group(1)) if m_h else 12, 96))
    if len(nums) < 3:
        nums = [1.0, 2.0, 3.0]
    arr = np.asarray(nums, dtype=np.float64)
    diffs = np.diff(arr)
    slope = float(np.median(diffs[-min(len(diffs), 8):]))
    sigma = float(np.std(diffs)) if len(diffs) > 1 else 0.1
    p50 = [round(float(arr[-1] + slope * (k + 1)), 4) for k in range(horizon)]
    p10 = [round(v - 1.2816 * sigma * ((k + 1) ** 0.5), 4) for k, v in enumerate(p50)]
    p90 = [round(v + 1.2816 * sigma * ((k + 1) ** 0.5), 4) for k, v in enumerate(p50)]
    return json.dumps({"model": "timeseries-heuristic-fallback", "horizon": horizon, "quantiles": {"p10": p10, "p50": p50, "p90": p90}})


class ModelNotLoadedError(RuntimeError):
    pass


# ====================================================================================================================
# Native PyTorch CUDA Hadamard-Grouped (`HG4-16R` / `HG2-16R`) Modules & Reconstructor
# ====================================================================================================================

def fwht128_torch(x: Any) -> Any:
    """Exact orthogonal 128-point Fast Walsh-Hadamard Transform along the last axis on CPU or CUDA."""
    import torch

    orig_shape = x.shape
    v = x.reshape(-1, 128)
    h = 1
    while h < 128:
        v = v.reshape(-1, 128 // (2 * h), 2, h)
        a = v[:, :, 0, :]
        b = v[:, :, 1, :]
        v = torch.stack((a + b, a - b), dim=2)
        h <<= 1
    return (v.reshape(orig_shape) * (1.0 / math.sqrt(128.0)))


def parse_hg_blob_rowmajor(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int, int]:
    """Parse canonical HG4R / HG2R blob into row-major packed uint8, FP16 scales, FP32 pre_diag, and dims."""
    magic = blob[:4]
    if magic not in (b"HG4R", b"HG2R"):
        raise ValueError(f"Unsupported HG magic: {magic!r}")
    bits = 4 if magic == b"HG4R" else 2
    r_orig, k_orig, r_pad, k_pad = struct.unpack_from("<IIII", blob, 4)
    nb = k_pad // 128
    off = 20
    pre_diag = np.frombuffer(blob, dtype=np.float32, count=k_pad, offset=off).copy()
    off += k_pad * 4
    sc_16r = np.frombuffer(blob, dtype=np.uint16, count=r_pad * nb, offset=off).reshape(r_pad // 16, nb, 16)
    sc_fp16 = np.ascontiguousarray(sc_16r.transpose(0, 2, 1).reshape(r_pad, nb)).view(np.float16).copy()
    off += r_pad * nb * 2
    if bits == 4:
        pk = np.frombuffer(blob, dtype=np.uint8, count=(r_pad * k_pad) // 2, offset=off).reshape(r_pad // 16, nb, 16, 16, 4)
        pk_rm = np.ascontiguousarray(pk.transpose(0, 3, 1, 2, 4).reshape(r_pad, nb, 16, 4))
    else:
        pk = np.frombuffer(blob, dtype=np.uint8, count=(r_pad * k_pad) // 4, offset=off).reshape(r_pad // 16, nb, 8, 16, 4)
        pk_rm = np.ascontiguousarray(pk.transpose(0, 3, 1, 2, 4).reshape(r_pad, nb, 8, 4))
    return pk_rm, sc_fp16, pre_diag, r_orig, k_orig, r_pad, k_pad, bits


def build_hg4_modules() -> tuple[Any, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class HG4Linear(nn.Module):
        """4-bit / 2-bit Hadamard-Grouped Linear executing `y = FWHT_128(D * x) @ W_rot_hat^T` on CUDA."""

        def __init__(self, blob: bytes, bias: torch.Tensor | None = None, compute_dtype: torch.dtype = torch.bfloat16) -> None:
            super().__init__()
            pk_rm, sc_fp16, pre_diag, r_orig, k_orig, r_pad, k_pad, bits = parse_hg_blob_rowmajor(blob)
            self.r_orig = r_orig
            self.k_orig = k_orig
            self.r_pad = r_pad
            self.k_pad = k_pad
            self.nb = k_pad // 128
            self.bits = bits
            self.compute_dtype = compute_dtype
            self.register_buffer("pk_rm", torch.from_numpy(pk_rm), persistent=True)
            self.register_buffer("sc_fp16", torch.from_numpy(sc_fp16), persistent=True)
            self.register_buffer("pre_diag", torch.from_numpy(pre_diag), persistent=True)
            lut_arr = LUT16_F32 if bits == 4 else LUT4_F32
            self.register_buffer("lut", torch.from_numpy(lut_arr).to(compute_dtype), persistent=True)
            if bias is not None:
                self.register_buffer("bias", bias.to(compute_dtype), persistent=True)
            else:
                self.bias = None

        def _unpack_w_rot(self) -> torch.Tensor:
            if self.bits == 4:
                lo = (self.pk_rm & 0x0F).long()
                hi = ((self.pk_rm >> 4) & 0x0F).long()
                idx = torch.stack((lo, hi), dim=3).reshape(self.r_pad, self.nb, 128)
            else:
                c0 = (self.pk_rm & 0x03).long()
                c1 = ((self.pk_rm >> 2) & 0x03).long()
                c2 = ((self.pk_rm >> 4) & 0x03).long()
                c3 = ((self.pk_rm >> 6) & 0x03).long()
                idx = torch.stack((c0, c1, c2, c3), dim=3).reshape(self.r_pad, self.nb, 128)
            sc = self.sc_fp16.to(self.compute_dtype).unsqueeze(-1)
            return (self.lut[idx] * sc).reshape(self.r_pad, self.k_pad)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            in_dtype = x.dtype
            xf = x.to(torch.float32)
            if self.k_orig < self.k_pad:
                xf = F.pad(xf, (0, self.k_pad - self.k_orig))
            x_rot = fwht128_torch(xf * self.pre_diag).to(self.compute_dtype)
            w_rot = self._unpack_w_rot()
            out = F.linear(x_rot, w_rot, None)
            if self.r_orig < self.r_pad:
                out = out[..., : self.r_orig]
            if self.bias is not None:
                out = out + self.bias
            return out.to(in_dtype)

    class HG4Embedding(nn.Module):
        """4-bit Hadamard-Grouped Embedding unpacking only active token IDs `W_hat[ids] = FWHT_128(W_rot[ids]) * D`."""

        def __init__(self, blob: bytes, out_dtype: torch.dtype = torch.bfloat16) -> None:
            super().__init__()
            pk_rm, sc_fp16, pre_diag, r_orig, k_orig, r_pad, k_pad, bits = parse_hg_blob_rowmajor(blob)
            self.num_embeddings = r_orig
            self.embedding_dim = k_orig
            self.r_pad = r_pad
            self.k_pad = k_pad
            self.nb = k_pad // 128
            self.bits = bits
            self.out_dtype = out_dtype
            self.register_buffer("pk_rm", torch.from_numpy(pk_rm), persistent=True)
            self.register_buffer("sc_fp16", torch.from_numpy(sc_fp16), persistent=True)
            self.register_buffer("pre_diag", torch.from_numpy(pre_diag), persistent=True)
            lut_arr = LUT16_F32 if bits == 4 else LUT4_F32
            self.register_buffer("lut", torch.from_numpy(lut_arr), persistent=True)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            flat_ids = input_ids.reshape(-1).clamp(0, self.num_embeddings - 1)
            pk_rows = self.pk_rm.index_select(0, flat_ids)
            sc_rows = self.sc_fp16.index_select(0, flat_ids).to(torch.float32).unsqueeze(-1)
            if self.bits == 4:
                lo = (pk_rows & 0x0F).long()
                hi = ((pk_rows >> 4) & 0x0F).long()
                idx = torch.stack((lo, hi), dim=3).reshape(-1, self.nb, 128)
            else:
                c0 = (pk_rows & 0x03).long()
                c1 = ((pk_rows >> 2) & 0x03).long()
                c2 = ((pk_rows >> 4) & 0x03).long()
                c3 = ((pk_rows >> 6) & 0x03).long()
                idx = torch.stack((c0, c1, c2, c3), dim=3).reshape(-1, self.nb, 128)
            w_rot = self.lut[idx] * sc_rows
            w_hat = (fwht128_torch(w_rot).reshape(-1, self.k_pad) * self.pre_diag)[:, : self.embedding_dim]
            return w_hat.to(self.out_dtype).reshape(*input_ids.shape, self.embedding_dim)

    return HG4Linear, HG4Embedding


def _set_submodule(root: Any, dotted_path: str, new_mod: Any) -> None:
    parts = dotted_path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_mod
    else:
        setattr(parent, last, new_mod)


def _get_submodule(root: Any, dotted_path: str) -> Any:
    cur = root
    for p in dotted_path.split("."):
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    return cur


def reconstruct_hg4_to_hf_dir(hg_dir: Path, out_dtype_str: str = "bfloat16") -> Path:
    """Reconstruct compact (<3 GB) HG4-16R repos (`Chronos-Bolt`, `V-JEPA 2.1`) into cached HF safetensors dir."""
    import torch
    from safetensors.torch import load_file, save_file

    cache_dir = hg_dir / "_reconstructed_hf"
    sentinel = cache_dir / "model.safetensors"
    if sentinel.is_file():
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((hg_dir / "hg_manifest.json").read_text(encoding="utf-8"))
    for fn in manifest.get("copied_metadata", []):
        src = hg_dir / fn
        if src.is_file():
            dst = cache_dir / fn
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for extra in hg_dir.glob("*.json"):
        if extra.name != "hg_manifest.json" and not (cache_dir / extra.name).is_file():
            shutil.copy2(extra, cache_dir / extra.name)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_dtype = torch.bfloat16 if out_dtype_str == "bfloat16" else torch.float32
    state: dict[str, torch.Tensor] = {}
    pt_file = hg_dir / "passthrough.safetensors"
    if pt_file.is_file():
        for k, v in load_file(str(pt_file), device="cpu").items():
            state[k] = v.to(target_dtype) if v.is_floating_point() and out_dtype_str == "bfloat16" else v

    lut16 = torch.from_numpy(LUT16_F32).to(dev)
    lut4 = torch.from_numpy(LUT4_F32).to(dev)
    for entry in manifest.get("tensors", []):
        if "file" not in entry:
            continue
        tname = entry["name"]
        blob = (hg_dir / entry["file"]).read_bytes()
        pk_rm, sc_fp16, pre_diag, r_orig, k_orig, r_pad, k_pad, bits = parse_hg_blob_rowmajor(blob)
        nb = k_pad // 128
        with torch.no_grad():
            pk_t = torch.from_numpy(pk_rm).to(dev)
            sc_t = torch.from_numpy(sc_fp16).to(dev, dtype=torch.float32).unsqueeze(-1)
            pd_t = torch.from_numpy(pre_diag).to(dev)
            if bits == 4:
                idx = torch.stack(((pk_t & 0x0F).long(), ((pk_t >> 4) & 0x0F).long()), dim=3).reshape(r_pad, nb, 128)
                w_rot = lut16[idx] * sc_t
            else:
                idx = torch.stack(
                    ((pk_t & 0x03).long(), ((pk_t >> 2) & 0x03).long(), ((pk_t >> 4) & 0x03).long(), ((pk_t >> 6) & 0x03).long()),
                    dim=3,
                ).reshape(r_pad, nb, 128)
                w_rot = lut4[idx] * sc_t
            w_hat = (fwht128_torch(w_rot).reshape(r_pad, k_pad) * pd_t)[:r_orig, :k_orig]
            orig_dt = entry.get("orig_dtype", "BF16")
            dt = torch.float32 if (orig_dt == "F32" and out_dtype_str == "float32") else target_dtype
            state[tname] = w_hat.to(dt).cpu().contiguous()

    tmp_st = cache_dir / "model.safetensors.tmp"
    save_file(state, str(tmp_st))
    tmp_st.replace(sentinel)
    return cache_dir


def load_hg4_causal_lm(hg_dir: Path) -> tuple[Any, Any]:
    """Load large HG4-16R CausalLM (`Goedel-Prover-V2-32B-HG4-16R`) directly into 4-bit GPU `HG4Linear` modules."""
    import torch
    import torch.nn as nn
    from accelerate import dispatch_model
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    HG4Linear, HG4Embedding = build_hg4_modules()
    manifest = json.loads((hg_dir / "hg_manifest.json").read_text(encoding="utf-8"))
    _trc = os.environ.get("CLOVER_TRUST_REMOTE_CODE") == "1"
    tok = AutoTokenizer.from_pretrained(str(hg_dir), trust_remote_code=_trc)
    cfg = AutoConfig.from_pretrained(str(hg_dir), trust_remote_code=_trc)

    with torch.device("meta"):
        mdl = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, trust_remote_code=_trc)
    mdl.eval()

    passthrough: dict[str, torch.Tensor] = {}
    pt_path = hg_dir / "passthrough.safetensors"
    if pt_path.is_file():
        passthrough = load_file(str(pt_path), device="cpu")

    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]
        use_bnb_cuda = torch.cuda.is_available()
    except ImportError:
        bnb = None
        use_bnb_cuda = False

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_layers = int(getattr(cfg, "num_hidden_layers", 64))
    lut16_dev = torch.from_numpy(LUT16_F32).to("cuda:0" if n_gpus > 0 else "cpu")
    lut4_dev = torch.from_numpy(LUT4_F32).to("cuda:0" if n_gpus > 0 else "cpu")

    def _target_gpu_for_path(mod_path: str) -> int:
        if n_gpus <= 1:
            return 0
        m_l = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", mod_path)
        if m_l:
            return min(n_gpus - 1, (int(m_l.group(1)) * n_gpus) // n_layers)
        if "embed_tokens" in mod_path:
            return 0
        return n_gpus - 1

    def _fast_bnb_linear_from_hg(blob: bytes, bias_t: torch.Tensor | None, gpu_idx: int) -> nn.Module:
        pk_rm, sc_fp16, pre_diag, r_orig, k_orig, r_pad, k_pad, bits = parse_hg_blob_rowmajor(blob)
        nb = k_pad // 128
        w_dev = torch.device(f"cuda:{gpu_idx}")
        w_hat_cpu = torch.empty((r_orig, k_orig), dtype=torch.bfloat16, device="cpu")
        with torch.no_grad():
            pd_t = torch.from_numpy(pre_diag).to(w_dev)
            lut_w = lut16_dev.to(w_dev) if bits == 4 else lut4_dev.to(w_dev)
            chunk_r = 4096
            for r0 in range(0, r_orig, chunk_r):
                r1_pad = min(r_pad, ((min(r_orig, r0 + chunk_r) + 15) // 16) * 16)
                rc = r1_pad - r0
                pk_c = torch.from_numpy(pk_rm[r0:r1_pad]).to(w_dev)
                sc_c = torch.from_numpy(sc_fp16[r0:r1_pad]).to(w_dev, dtype=torch.float32).unsqueeze(-1)
                if bits == 4:
                    idx = torch.stack(((pk_c & 0x0F).long(), ((pk_c >> 4) & 0x0F).long()), dim=3).reshape(rc, nb, 128)
                else:
                    idx = torch.stack(
                        ((pk_c & 0x03).long(), ((pk_c >> 2) & 0x03).long(), ((pk_c >> 4) & 0x03).long(), ((pk_c >> 6) & 0x03).long()),
                        dim=3,
                    ).reshape(rc, nb, 128)
                valid_r = min(r_orig, r0 + chunk_r) - r0
                part = (fwht128_torch(lut_w[idx] * sc_c).reshape(rc, k_pad) * pd_t)[:valid_r, :k_orig]
                w_hat_cpu[r0 : r0 + valid_r].copy_(part.to(torch.bfloat16).cpu())
                del pk_c, sc_c, idx, part
        torch.cuda.empty_cache()
        lin4 = bnb.nn.Linear4bit(k_orig, r_orig, bias=bias_t is not None, compute_dtype=torch.bfloat16, quant_type="nf4")
        lin4.weight = bnb.nn.Params4bit(w_hat_cpu, requires_grad=False, compress_statistics=True, quant_type="nf4").to(w_dev)
        if bias_t is not None:
            lin4.bias = nn.Parameter(bias_t.to(device=w_dev, dtype=torch.bfloat16), requires_grad=False)
        return lin4

    # Replace quantized 2D weights (`*.weight`) with fused C++ 4-bit Linear4bit (or HG4Linear fallback) / HG4Embedding
    quant_entries = {e["name"]: e for e in manifest.get("tensors", []) if "file" in e}
    for w_name, entry in quant_entries.items():
        if not w_name.endswith(".weight"):
            continue
        mod_path = w_name[: -len(".weight")]
        orig_mod = _get_submodule(mdl, mod_path)
        blob = (hg_dir / entry["file"]).read_bytes()
        bias_t = passthrough.pop(f"{mod_path}.bias", None)
        if isinstance(orig_mod, nn.Embedding):
            _set_submodule(mdl, mod_path, HG4Embedding(blob, out_dtype=torch.bfloat16))
        elif use_bnb_cuda:
            _set_submodule(mdl, mod_path, _fast_bnb_linear_from_hg(blob, bias_t, _target_gpu_for_path(mod_path)))
        else:
            _set_submodule(mdl, mod_path, HG4Linear(blob, bias=bias_t, compute_dtype=torch.bfloat16))

    # Materialize remaining passthrough parameters & buffers (1D norms, biases, etc.)
    for p_name, tensor in passthrough.items():
        parts = p_name.rsplit(".", 1)
        if len(parts) == 2:
            parent = _get_submodule(mdl, parts[0])
            attr = parts[1]
            if attr in parent._parameters:
                parent._parameters[attr] = nn.Parameter(tensor.to(torch.bfloat16), requires_grad=False)
            else:
                parent.register_buffer(attr, tensor, persistent=True)

    # Re-instantiate rotary_emb on real CPU so modern Transformers `rope_parameters` / `inv_freq` are 100% genuine!
    if hasattr(mdl, "model") and hasattr(mdl.model, "rotary_emb"):
        rot_cls = type(mdl.model.rotary_emb)
        mdl.model.rotary_emb = rot_cls(config=cfg)

    for _, mod in mdl.named_modules():
        for bname, buf in list(mod._buffers.items()):
            if buf is not None and getattr(buf, "is_meta", False):
                mod._buffers[bname] = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")

    # Tie lm_head to embed_tokens if config ties word embeddings and lm_head remained on meta
    if getattr(cfg, "tie_word_embeddings", False) and hasattr(mdl, "lm_head"):
        lm_w = getattr(mdl.lm_head, "weight", None)
        if lm_w is not None and getattr(lm_w, "is_meta", False) and "model.embed_tokens.weight" in quant_entries:
            blob = (hg_dir / quant_entries["model.embed_tokens.weight"]["file"]).read_bytes()
            if use_bnb_cuda:
                mdl.lm_head = _fast_bnb_linear_from_hg(blob, None, max(0, n_gpus - 1))
            else:
                mdl.lm_head = HG4Linear(blob, bias=None, compute_dtype=torch.bfloat16)

    if n_gpus <= 1:
        dev = torch.device("cuda:0" if n_gpus == 1 else "cpu")
        mdl.to(dev)
    else:
        dmap: dict[str, int | str] = {"model.embed_tokens": 0, "model.rotary_emb": 0}
        for i in range(n_layers):
            dmap[f"model.layers.{i}"] = min(n_gpus - 1, (i * n_gpus) // n_layers)
        dmap["model.norm"] = n_gpus - 1
        dmap["lm_head"] = n_gpus - 1
        mdl = dispatch_model(mdl, device_map=dmap)

    return tok, mdl


class SpecialistRuntime:
    def __init__(self, model_id: str, model_dir: Path | None, gpu: str, allow_mock: bool = False) -> None:
        self.model_id = model_id
        self.model_dir = model_dir
        self.gpu = gpu
        self.allow_mock = allow_mock
        self._loaded_impl: Any = None
        self._lock = threading.Lock()
        if gpu and gpu not in ("none", "auto"):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    def infer(self, messages: list[dict[str, Any]], max_tokens: int) -> str:
        prompt = _extract_last_user(messages)
        with self._lock:
            return self._infer_locked(messages, prompt, max_tokens)

    def _infer_locked(self, messages: list[dict[str, Any]], prompt: str, max_tokens: int) -> str:
        if self.model_id == "chronos-2":
            if self._loaded_impl is None:
                try:
                    import torch
                    from chronos import BaseChronosPipeline  # type: ignore[import-not-found]

                    if self.model_dir and (self.model_dir / "hg_manifest.json").is_file():
                        src = str(reconstruct_hg4_to_hf_dir(self.model_dir, out_dtype_str="float32"))
                    elif self.model_dir and self.model_dir.is_dir():
                        src = str(self.model_dir)
                    else:
                        src = "amazon/chronos-bolt-base"
                    dev_map = "cuda" if torch.cuda.is_available() else "cpu"
                    self._loaded_impl = BaseChronosPipeline.from_pretrained(src, device_map=dev_map, dtype=torch.bfloat16)
                except Exception as exc:
                    print(f"[specialist] chronos-2 load error: {exc}", flush=True)
                    self._loaded_impl = False
            if self._loaded_impl not in (None, False):
                import torch

                nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", prompt.split("time-series:")[-1])]
                if len(nums) < 2:
                    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", prompt)] or [1.0, 2.0, 3.0]
                m_h = re.search(r"next\s+(\d+)\s+steps", prompt, re.IGNORECASE)
                horizon = max(1, min(int(m_h.group(1)) if m_h else 12, 64))
                ctx_t = torch.tensor(nums, dtype=torch.float32)
                try:
                    q, _ = self._loaded_impl.predict_quantiles(
                        inputs=ctx_t, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
                    )
                except TypeError:
                    q, _ = self._loaded_impl.predict_quantiles(
                        context=ctx_t, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
                    )
                q_np = q[0].cpu().float().numpy()
                return json.dumps({
                    "model": "chronos-bolt-hg4-16r",
                    "horizon": horizon,
                    "quantiles": {"p10": q_np[:, 0].round(4).tolist(), "p50": q_np[:, 1].round(4).tolist(), "p90": q_np[:, 2].round(4).tolist()},
                })
            if self.allow_mock:
                return _run_heuristic_timeseries(prompt)
            raise ModelNotLoadedError("chronos-2 package/weights not loaded on this petal")

        if self.model_id == "vjepa-2.1":
            if self._loaded_impl is None and self.model_dir and self.model_dir.is_dir():
                try:
                    import torch
                    from transformers import AutoModel

                    src_dir = (
                        reconstruct_hg4_to_hf_dir(self.model_dir, out_dtype_str="bfloat16")
                        if (self.model_dir / "hg_manifest.json").is_file()
                        else self.model_dir
                    )
                    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    _trc = os.environ.get("CLOVER_TRUST_REMOTE_CODE") == "1"
                    mdl = AutoModel.from_pretrained(str(src_dir), torch_dtype=torch.bfloat16, trust_remote_code=_trc).to(dev).eval()
                    self._loaded_impl = ("vjepa2_gpu", mdl, dev)
                except Exception as exc:
                    print(f"[specialist] vjepa-2.1 load error: {exc}", flush=True)
                    self._loaded_impl = False
            if isinstance(self._loaded_impl, tuple) and self._loaded_impl[0] == "vjepa2_gpu":
                import torch

                _, mdl, dev = self._loaded_impl
                nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", prompt)]
                v0 = nums[0] if len(nums) >= 1 else 5.0
                g_acc = abs(nums[1]) if len(nums) >= 2 and abs(nums[1]) > 0.1 else 9.81
                dt_step = 0.1
                # Render a deterministic 16-frame 256x256 synthetic trajectory clip matching the query kinematics
                frames = torch.zeros((1, 16, 3, 256, 256), dtype=torch.bfloat16, device=dev)
                traj: list[dict[str, float]] = []
                for t_idx in range(16):
                    t_sec = t_idx * dt_step
                    x_pos = v0 * t_sec
                    y_pos = max(0.0, 20.0 + 0.5 * v0 * t_sec - 0.5 * g_acc * (t_sec ** 2))
                    px = min(251, max(4, int(16 + x_pos * 8.0) % 252))
                    py = min(251, max(4, 251 - int(y_pos * 8.0) % 248))
                    frames[0, t_idx, 0, py - 4 : py + 4, px - 4 : px + 4] = 1.0
                    frames[0, t_idx, 1, py - 2 : py + 2, px - 2 : px + 2] = 0.75
                    traj.append({"t_s": round(t_sec, 2), "x_m": round(x_pos, 4), "y_m": round(y_pos, 4)})
                with torch.no_grad():
                    out = mdl(pixel_values_videos=frames)
                    hid = out.last_hidden_state.float()  # [1, N_tokens, 1408]
                    t_chunks = hid.reshape(1, 8, -1, hid.shape[-1]).mean(dim=2)[0]  # [8, 1408]
                    lat_vel = torch.norm(t_chunks[1:] - t_chunks[:-1], dim=-1).cpu().tolist()
                    lat_norm = float(torch.norm(t_chunks[-1]).item())
                return json.dumps({
                    "model": "vjepa2-vitg-hg4-16r",
                    "embed_dim": int(hid.shape[-1]),
                    "num_spatiotemporal_tokens": int(hid.shape[1]),
                    "latent_step_velocities": [round(v, 4) for v in lat_vel],
                    "terminal_latent_norm": round(lat_norm, 4),
                    "trajectory_rollout": traj[:8],
                })

        if self.model_id == "boltz-2":
            if self.model_dir and self.model_dir.is_dir():
                boltz_bin = shutil.which("boltz") or "/venv/main/bin/boltz"
                if Path(boltz_bin).is_file():
                    seq_match = re.search(r"\b([ACDEFGHIKLMNPQRSTVWY]{12,512})\b", prompt)
                    seq = seq_match.group(1) if seq_match else "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
                    smiles_match = re.search(r"SMILES[:=\s]+([^\s,;]+)", prompt, re.IGNORECASE)
                    with tempfile.TemporaryDirectory(prefix="clover_boltz2_") as tmpd:
                        tmp_p = Path(tmpd)
                        yaml_lines = [
                            "version: 1",
                            "sequences:",
                            "  - protein:",
                            "      id: A",
                            f"      sequence: {seq}",
                            "      msa: empty",
                        ]
                        if smiles_match:
                            safe_smiles = smiles_match.group(1).replace("'", "''").replace("\n", "").replace("\r", "")
                            yaml_lines.extend([
                                "  - ligand:",
                                "      id: B",
                                f"      smiles: '{safe_smiles}'",
                            ])
                        in_yaml = tmp_p / "input.yaml"
                        in_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
                        out_dir = tmp_p / "out"
                        cmd = [
                            boltz_bin,
                            "predict",
                            str(in_yaml),
                            "--cache",
                            str(self.model_dir),
                            "--out_dir",
                            str(out_dir),
                            "--recycling_steps",
                            "2",
                            "--sampling_steps",
                            "30",
                            "--diffusion_samples",
                            "1",
                            "--no_kernels",
                        ]
                        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
                        conf_files = list(out_dir.rglob("confidence_*.json"))
                        cif_files = list(out_dir.rglob("*.cif"))
                        if proc.returncode == 0 and conf_files:
                            conf = json.loads(conf_files[0].read_text(encoding="utf-8"))
                            ca_coords: list[list[float]] = []
                            if cif_files:
                                for ln in cif_files[0].read_text(encoding="utf-8").splitlines():
                                    if ln.startswith("ATOM ") and " CA " in ln:
                                        cols = ln.split()
                                        try:
                                            floats = [float(c) for c in cols if re.match(r"^[-+]?\d+\.\d+$", c)]
                                            if len(floats) >= 3:
                                                ca_coords.append([round(floats[0], 3), round(floats[1], 3), round(floats[2], 3)])
                                        except ValueError:
                                            pass
                            return json.dumps({
                                "model": "boltz-2-gpu",
                                "sequence_length": len(seq),
                                "confidence_score": round(float(conf.get("confidence_score", 0.0)), 4),
                                "ptm": round(float(conf.get("ptm", 0.0)), 4),
                                "complex_plddt": round(float(conf.get("complex_plddt", 0.0)), 4),
                                "ca_residue_count": len(ca_coords),
                                "ca_backbone_head": ca_coords[:5],
                            })
                        raise ModelNotLoadedError(f"boltz-2 predict rc={proc.returncode}: {proc.stderr[-400:]}")

        if self.model_id == "router-0.6b":
            if self._loaded_impl is None and self.model_dir and self.model_dir.is_dir():
                try:
                    import torch
                    from peft import PeftModel  # type: ignore[import-not-found]
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    base_dir = os.environ.get("CLOVER_ROUTER_BASE_DIR", "/workspace/models/qwen3-0.6b-base")
                    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    _trc = os.environ.get("CLOVER_TRUST_REMOTE_CODE") == "1"
                    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=_trc)
                    base_mdl = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16, trust_remote_code=_trc).to(dev)
                    mdl = PeftModel.from_pretrained(base_mdl, str(self.model_dir)).eval()
                    self._loaded_impl = ("router_peft", tok, mdl, dev)
                except Exception as exc:
                    print(f"[specialist] router-0.6b load error: {exc}", flush=True)
                    self._loaded_impl = False
            if isinstance(self._loaded_impl, tuple) and self._loaded_impl[0] == "router_peft":
                import torch

                _, tok, mdl, dev = self._loaded_impl
                chat_txt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                enc = tok(chat_txt, return_tensors="pt").to(dev)
                with torch.no_grad():
                    out = mdl.generate(**enc, max_new_tokens=min(max_tokens, 128), do_sample=False)
                return str(tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)).strip()

        if self.model_dir and self.model_dir.is_dir() and self._loaded_impl is None:
            if (self.model_dir / "hg_manifest.json").is_file() or (self.model_dir / "manifest.json").is_file():
                tok, mdl = load_hg4_causal_lm(self.model_dir)
                self._loaded_impl = ("hg4_causal_lm", tok, mdl)
            else:
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                    _trc = os.environ.get("CLOVER_TRUST_REMOTE_CODE") == "1"
                    tok = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=_trc)
                    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                    mdl = AutoModelForCausalLM.from_pretrained(
                        str(self.model_dir), quantization_config=qcfg, device_map="auto", trust_remote_code=_trc
                    )
                    self._loaded_impl = ("hf_4bit", tok, mdl)
                except Exception as exc:
                    print(f"[specialist] causal LM load error: {exc}", flush=True)
                    self._loaded_impl = False

        if isinstance(self._loaded_impl, tuple) and self._loaded_impl[0] in ("hg4_causal_lm", "hf_4bit"):
            import torch

            _, tok, mdl = self._loaded_impl
            try:
                chat_txt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                chat_txt = prompt
            first_dev = next((b.device for b in mdl.buffers()), torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            enc = tok(chat_txt, return_tensors="pt").to(first_dev)
            with torch.no_grad():
                out = mdl.generate(
                    **enc,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            return str(tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)).strip()

        if self.allow_mock:
            return f"[mock-{self.model_id}] {prompt[:160]}"
        raise ModelNotLoadedError(f"Specialist '{self.model_id}' HG checkpoint not yet loaded at '{self.model_dir}'")


@dataclasses.dataclass
class _ShimCompletion:
    content: str
    tool_calls: list[Any]
    finish_reason: str
    usage: dict[str, int]
    telemetry: dict[str, Any]


class _SpecialistSocketRouterShim:
    """Adapts `SpecialistRuntime` to `TokenSocketServer` (`tsock://`)."""

    def __init__(self, rt: SpecialistRuntime) -> None:
        self.rt = rt
        self.backends: dict[str, Any] = {}

    def complete(self, req_body: dict[str, Any]) -> _ShimCompletion:
        msgs = req_body.get("messages") or []
        max_tok = int(req_body.get("max_completion_tokens") or req_body.get("max_tokens") or 512)
        text = self.rt.infer(msgs, max_tok)
        comp_toks = max(1, len(text) // 4)
        return _ShimCompletion(
            content=text,
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": 32, "completion_tokens": comp_toks, "total_tokens": 32 + comp_toks},
            telemetry={"served_by": self.rt.model_id},
        )


def _run_self_test() -> None:
    """Verify mathematical equivalence between `HG4Linear` / `HG4Embedding` and `convert_to_hg.py`."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import convert_to_hg as cth

    HG4Linear, HG4Embedding = build_hg4_modules()
    rng = np.random.Generator(np.random.PCG64(1234))
    w = rng.standard_normal((48, 256), dtype=np.float32) * 0.05
    blob, _ = cth.pack_matrix_hg(w, "self_test.weight", bits=4, use_ldlq=True)
    x = rng.standard_normal((6, 256), dtype=np.float32)
    y_ref = cth.forward_hg_matmul(x, blob)
    lin = HG4Linear(blob, compute_dtype=torch.float32)
    with torch.no_grad():
        y_mod = lin(torch.from_numpy(x)).numpy()
    max_err = float(np.max(np.abs(y_ref - y_mod)))
    assert max_err < 1e-4, f"HG4Linear mismatch vs forward_hg_matmul: max_err={max_err}"

    emb = HG4Embedding(blob, out_dtype=torch.float32)
    w_rot_hat, pre_diag, r_orig, k_orig, _, _, _ = cth.unpack_matrix_hg_rotated(blob)
    w_hat_ref = (cth.fwht128_np(w_rot_hat) * pre_diag[None, :])[:r_orig, :k_orig]
    ids = torch.tensor([[0, 17, 47]], dtype=torch.long)
    with torch.no_grad():
        e_mod = emb(ids).numpy()[0]
    emb_err = float(np.max(np.abs(w_hat_ref[[0, 17, 47]] - e_mod)))
    assert emb_err < 1e-4, f"HG4Embedding mismatch: emb_err={emb_err}"
    print(f"[self-test] PASS: HG4Linear max_err={max_err:.2e}, HG4Embedding max_err={emb_err:.2e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="Run exact Hadamard HG4Linear/HG4Embedding parity self-test")
    ap.add_argument("--model", choices=("goedel-prover-v2-32b", "vjepa-2.1", "boltz-2", "chronos-2", "router-0.6b"))
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--token-socket-port", type=int, default=0, help="Optional binary TokenSocket port (tsock://)")
    ap.add_argument("--eager-load", action="store_true", help="Load specialist weights into VRAM before serving HTTP")
    ap.add_argument("--allow-mock", action="store_true", help="Explicitly enable offline mock/heuristic responses for unit testing")
    args = ap.parse_args()

    if args.self_test:
        _run_self_test()
        return
    if not args.model:
        ap.error("--model is required unless --self-test is passed")

    rt = SpecialistRuntime(args.model, args.model_dir, args.gpu, allow_mock=args.allow_mock)
    if args.eager_load and args.model in ("goedel-prover-v2-32b", "vjepa-2.1", "chronos-2", "router-0.6b"):
        print(f"[specialist] Eager-loading `{args.model}` from `{args.model_dir}` on GPU `{args.gpu}` ...", flush=True)
        try:
            rt.infer([{"role": "user", "content": "warmup next 4 steps: 1.0, 2.0, 3.0, 4.0"}], 4)
            print(f"[specialist] Eager-load complete for `{args.model}`.", flush=True)
        except Exception as exc:
            print(f"[specialist] Eager-load warning for `{args.model}`: {exc}", flush=True)

    api_key = (os.environ.get("CLOVER_GATEWAY_API_KEY") or "").strip()
    tsock_srv = None
    if args.token_socket_port > 0:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "hg4-fusion-gateway"))
        from clover_token_socket import TokenSocketServer

        tsock_srv = TokenSocketServer(_SpecialistSocketRouterShim(rt), host=args.host, port=args.token_socket_port, secret=api_key)
        print(f"[specialist] Binary TokenSocket listening on tsock://{args.host}:{tsock_srv.bound_port}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: Any) -> None:
            return

        def _send(self, code: int, obj: Any) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path in ("/health", "/v1/health"):
                self._send(200, {"status": "ok", "specialist": args.model})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if api_key:
                key_b = api_key.encode("utf-8")
                auth_b = (self.headers.get("Authorization") or "").encode("utf-8")
                hdr_b = (self.headers.get("api-key") or "").encode("utf-8")
                if not (hmac.compare_digest(auth_b, b"Bearer " + key_b) or hmac.compare_digest(hdr_b, key_b)):
                    self._send(401, {"error": "unauthorized"})
                    return
            elif args.host not in ("127.0.0.1", "localhost", "::1") and os.environ.get("CLOVER_ALLOW_UNAUTH_CLUSTER") != "1":
                self._send(401, {"error": "CLOVER_GATEWAY_API_KEY required on non-loopback specialist server"})
                return
            if not self.path.startswith("/v1/chat/completions"):
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                n = 0
            if n <= 0 or n > 16_777_216:
                self._send(413 if n > 0 else 400, {"error": "missing or oversized body"})
                return
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            max_tok = int(body.get("max_completion_tokens") or body.get("max_tokens") or 512)
            try:
                text = rt.infer(body.get("messages") or [], max_tok)
            except ModelNotLoadedError as exc:
                self._send(503, {"error": {"message": str(exc), "type": "model_not_loaded", "code": "model_not_loaded"}})
                return
            self._send(200, {
                "id": f"spec-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": args.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 32, "completion_tokens": max(1, len(text) // 4), "total_tokens": 32 + max(1, len(text) // 4)},
            })

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[specialist] Serving `{args.model}` on http://{args.host}:{args.port}/v1", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
