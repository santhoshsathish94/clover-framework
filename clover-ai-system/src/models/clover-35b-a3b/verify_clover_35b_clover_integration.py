#!/usr/bin/env python3
"""
End-to-End Verification & 7-Layer Clover Switching Suite for DiffuSpace Clover-35B-A3B
(`clover-ai-system/src/models/clover-35b-a3b/verify_clover_35b_clover_integration.py`).

Verifies:
  1. `Clover-35B-A3B` registration in `CLOVER_MODEL_REGISTRY` (`switchable_model_backend.py`) with
     `total_layers = 40` (`30 BD-GDN + 10 Gated GQA`), `hidden_size = 2048`, `vocab_size = 248,320`,
     `routed_experts_per_moe_layer = 256`, `active_experts_per_token = 8`.
  2. Hot-switching `CloverSwitchableModelLayer1` across all four model families:
     `"clover-35b-a3b"`, `"kimi-k3"`, `"nemotron-3.5-super-vl"`, and `"lfm3-1.7b"`.
  3. Projection of `Clover-35B-A3B`'s `2,048`-d hidden states into Layer-2 (`4096`-d `HybridAgent`
     latent space + `2048`-d `Clover35BNativeHybridLink`), Layer-3 (`384`-d `11-Level NCM` via
     `Clover35BComboBridge` & `Clover35BExpertBridge`), and `JEPA` (`64`-d `CloverJEPARuntime`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
REPO_ROOT = MODELS_DIR.parents[2]
STUDIO_CLOVER = REPO_ROOT / "clover-ai-system" / "studio" / "clover"
for p in (str(SCRIPT_DIR), str(MODELS_DIR), str(STUDIO_CLOVER)):
    if p not in sys.path:
        sys.path.insert(0, p)

from clover_35b_hybrid_ncm import (
    D_CLOVER35B,
    D_VQ,
    V_CLOVER35B,
    Clover35BComboBridge,
    Clover35BExpertBridge,
    Clover35BNativeHybridLink,
)
import jepa_engine as JE
from switchable_model_backend import CLOVER_MODEL_REGISTRY, CloverSwitchableModelLayer1


@torch.no_grad()
def main() -> int:
    assert "clover-35b-a3b" in CLOVER_MODEL_REGISTRY, "clover-35b-a3b missing from CLOVER_MODEL_REGISTRY"
    spec = CLOVER_MODEL_REGISTRY["clover-35b-a3b"]
    assert spec.total_layers == 40
    assert spec.hidden_size == D_CLOVER35B == 2048
    assert spec.vocab_size == V_CLOVER35B == 248320
    assert spec.routed_experts_per_moe_layer == 256
    assert spec.active_experts_per_token == 8

    switcher = CloverSwitchableModelLayer1(initial_model_key="clover-35b-a3b")
    switch_results = {}
    for m_key in ("clover-35b-a3b", "kimi-k3", "nemotron-3.5-super-vl", "lfm3-1.7b"):
        sp = switcher.switch_model(m_key)
        h_dummy = torch.randn(1, 8, sp.hidden_size)
        proj = switcher.project_backbone_state_to_clover_layers(h_dummy)
        assert proj["ncm_concept_384_shape"] == [1, 8, 384]
        assert proj["hybrid_agent_4096_shape"] == [1, 8, 4096]
        switch_results[m_key] = proj

    # Verify Clover-35B-A3B Native HybridLink + ComboBridge + ExpertBridge + JEPA
    h_35b = torch.randn(1, 8, D_CLOVER35B)
    link_512 = Clover35BNativeHybridLink(d=D_CLOVER35B, bottleneck=512)
    c_bridge = Clover35BComboBridge(D_CLOVER35B, D_VQ)
    e_bridge = Clover35BExpertBridge(D_CLOVER35B, D_VQ)
    r_35b = link_512(h_35b)
    z_combo = c_bridge(r_35b)
    z_exp = e_bridge(h_35b)
    assert list(r_35b.shape) == [1, 8, 2048]
    assert list(z_combo.shape) == [1, 8, 384]
    assert list(z_exp.shape) == [1, 8, 384]

    jepa_rt = JE.CloverJEPARuntime(STUDIO_CLOVER.parent)
    jepa_plan = jepa_rt.plan_c2_latent(
        h_35b.numpy(),
        r_35b.numpy(),
        offered_tools=["run_python", "web_search", "fetch_url"],
        configured_arch="auto",
    )
    assert jepa_plan["enabled"] is True

    summary = {
        "status": "ALL_CLOVER_35B_INTEGRATION_CHECKS_PASSED",
        "spec": {
            "model_key": spec.model_key,
            "hf_repo_id": spec.hf_repo_id,
            "total_layers": spec.total_layers,
            "hidden_size": spec.hidden_size,
            "vocab_size": spec.vocab_size,
            "routed_experts": spec.routed_experts_per_moe_layer,
            "active_experts": spec.active_experts_per_token,
        },
        "switchable_models_verified": list(switch_results.keys()),
        "jepa_c2_plan": {
            "arch": jepa_plan["arch"],
            "energy_before": jepa_plan["energy_before"],
            "energy_after": jepa_plan["energy_after"],
            "ranked_tools": jepa_plan.get("ranked_tools"),
        },
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
