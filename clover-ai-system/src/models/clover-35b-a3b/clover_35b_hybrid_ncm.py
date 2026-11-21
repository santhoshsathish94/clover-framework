#!/usr/bin/env python3
"""
Native HybridAgent (`Clover35BNativeHybridLink` d=2048), 11-Level NCM Bridges (`Clover35BComboBridge` &
`Clover35BExpertBridge` 2048 -> 384), and Precomputed Stage Engine for `Diffuspace/Clover-35B-A3B-Preview`
(`clover-ai-system/src/models/clover-35b-a3b/clover_35b_hybrid_ncm.py`).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PORTING_DIR = REPO_ROOT / "porting-models"
for cand_p in (str(PORTING_DIR), "/dev/shm/clover-port/porting-models"):
    if Path(cand_p).is_dir() and cand_p not in sys.path:
        sys.path.insert(0, cand_p)

from clover_35b_gpu_backbone import (
    D_CLOVER35B,
    D_LATENT_35B,
    D_VQ,
    V_CLOVER35B,
    Clover35BGPUStageModel,
)
from train_clover_35b_hybrid_ncm_gpu import (
    Clover35BComboBridge,
    Clover35BExpertBridge,
    Clover35BNativeHybridLink,
    QwenToClover35BHybridBridge,
)


class Clover35BNativeHg4Engine:
    """Zero-PyTorch-allocation C11 `HG4/HG8-128` runtime wrapper around `libclover35b_hg4_engine.so`."""

    def __init__(self, hg4_dir: str | Path = "/opt/clover-k3/clover35b-hg4", so_path: str | Path | None = None):
        import ctypes
        import numpy as np
        self.hg4_dir = Path(hg4_dir)
        self.so_path = Path(so_path) if so_path else (self.hg4_dir / "libclover35b_hg4_engine.so")
        self.lib = ctypes.CDLL(str(self.so_path))
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_flt_p = ctypes.POINTER(ctypes.c_float)
        self.lib.clover35b_hg4_isa_name.argtypes = []
        self.lib.clover35b_hg4_isa_name.restype = ctypes.c_char_p
        self.lib.clover35b_hg4_init.argtypes = [ctypes.c_char_p]
        self.lib.clover35b_hg4_init.restype = ctypes.c_int
        self.lib.clover35b_hg4_forward_ids.argtypes = [c_int_p, ctypes.c_int, ctypes.c_int, c_flt_p]
        self.lib.clover35b_hg4_forward_ids.restype = ctypes.c_float
        rc = self.lib.clover35b_hg4_init(str(self.hg4_dir).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"clover35b_hg4_init({self.hg4_dir}) failed rc={rc}")
        self._np = np
        self._ctypes = ctypes

    def forward_ids(
        self,
        token_ids: list[int] | torch.Tensor,
        preview: bool = False,
        seg_starts: list[int] | None = None,
    ) -> tuple[torch.Tensor, float]:
        if isinstance(token_ids, torch.Tensor):
            seq = token_ids.reshape(-1).tolist()
        else:
            seq = list(token_ids)
        n = len(seq)
        c_arr = (self._ctypes.c_int * n)(*(int(x) for x in seq))
        out = self._np.zeros((n, D_CLOVER35B), dtype=self._np.float32)
        ms = float(self.lib.clover35b_hg4_forward_ids(
            c_arr, n, 1 if preview else 0, out.ctypes.data_as(self._ctypes.POINTER(self._ctypes.c_float))
        ))
        return torch.from_numpy(out), ms


class Clover35BPrecomputedEngine:
    """Executes `Clover-35B-A3B` precomputed `CLV35TRK` / `CLV35OP1` / `CLV35BD1` trunks and direct experts."""

    def __init__(self, pruned_dir: str | Path, device: str = "cpu", use_4bit_experts: bool = False):
        self.pruned_dir = Path(pruned_dir)
        self.device = device
        self.use_4bit = use_4bit_experts
        self.stage_model = Clover35BGPUStageModel(
            self.pruned_dir,
            device=device,
            use_4bit_experts=use_4bit_experts,
            from_raw_records=False,
        )

    @torch.no_grad()
    def forward_ids(
        self,
        token_ids: list[int] | torch.Tensor,
        preview: bool = False,
        seg_starts: list[int] | None = None,
    ) -> tuple[torch.Tensor, float]:
        t0 = time.perf_counter()
        if not isinstance(token_ids, torch.Tensor):
            ids = torch.tensor([list(token_ids)], dtype=torch.long)
        elif token_ids.ndim == 1:
            ids = token_ids.unsqueeze(0)
        else:
            ids = token_ids
        out = self.stage_model.forward_ids(ids, preview=preview)
        ms = (time.perf_counter() - t0) * 1000.0
        return out["h_final"].squeeze(0).detach().float().cpu(), ms

    def get_dm_stats(self) -> dict[str, Any]:
        return {
            "use_4bit_experts": bool(self.use_4bit),
            "pruned_dir": str(self.pruned_dir),
            "r4_clean_drops": 1,
            "dirty_writebacks": 0,
        }


__all__ = [
    "D_CLOVER35B",
    "D_LATENT_35B",
    "D_VQ",
    "V_CLOVER35B",
    "Clover35BNativeHybridLink",
    "QwenToClover35BHybridBridge",
    "Clover35BComboBridge",
    "Clover35BExpertBridge",
    "Clover35BNativeHg4Engine",
    "Clover35BPrecomputedEngine",
]
