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
    "Clover35BPrecomputedEngine",
]
