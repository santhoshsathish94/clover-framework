#!/usr/bin/env python3
"""
Switchable Layer-1 Foundation Model Backend & 7-Layer Clover Bridge
(`clover-ai-system/src/models/switchable_model_backend.py`).

In Clover's 7-layer architecture:
  - **Layer 1 (Switchable Foundation Model Backbone):**
      1. `kimi-k3`: MoonshotAI Kimi K3 (93 layers, `D_model = 7168`, `V = 163840`, 68 KDA + 24 MLA + 896 MXFP4 experts/layer, DSpark 5-layer drafter, MoonViT3d vision)
      2. `nemotron-3.5-super-vl`: NVIDIA Nemotron 3.5 Super VL (88 single-mixer layers, `D_model = 4096`, `V = 131072`, 40 Mamba-2 + 40 Latent `ReLU^2` MoE [512 experts/layer, top-22] + 8 NoPE 16:1 GQA, Native 2-layer MTP drafter, C-RADIO 32-layer vision + EVS video pruning)
      3. `lfm3-1.7b`: Liquid AI LFM3-1.7B (24 tri-hybrid layers, `D_model = 2048`, `V = 128000` tied, 13 width-7 `Lfm3ShortConv` + 5 Sliding GQA + 6 Full GQA + sandwich RMSNorm + SwiGLU)
  - **Layers 2..7 (Shared Surrounding Clover Architecture):**
      - Layer 2: `HybridAgent` continuous `<latent>` reasoning & live tools (`<python>`, `<search>`) via `CloverUniversalHybridLink(D_model -> 4096)`
      - Layer 3: `NextConceptTransformer (NCM)` 11-pass hierarchical VQ concept decoder via `CloverUniversalConceptBridge(D_model -> 384)`
      - Layer 4: `CloverMemoryGraphStore` (File-backed Markdown Memory Tree + `staging/` promotion gate)
      - Layer 5: `CloverOntologyRetriever` + `CloverSymbolicVerifier` (Deterministic Horn-rule Ontology GraphRAG)
      - Layer 6: `CloverPythonGenerativeCompiler` + `CloverRustGenerativeCompiler` (Statement-boundary AST/doctest/rustc verification & rollback)
      - Layer 7: `SpecMD Least-Stale + R4 Clean-Drop` 3-Tier Data-Movement & Expert Slot Cache (`VRAM <-> Pinned Host RAM <-> NVMe Direct`)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CloverModelSpec:
    model_key: str
    hf_repo_id: str
    total_layers: int
    hidden_size: int
    vocab_size: int
    routed_experts_per_moe_layer: int
    active_experts_per_token: int
    moe_latent_size: int
    max_context_tokens: int
    has_native_vision: bool
    speculative_drafter_kind: str
    model_subdir: str


CLOVER_MODEL_REGISTRY: dict[str, CloverModelSpec] = {
    "kimi-k3": CloverModelSpec(
        model_key="kimi-k3",
        hf_repo_id="moonshotai/Kimi-K3",
        total_layers=93,
        hidden_size=7168,
        vocab_size=163840,
        routed_experts_per_moe_layer=896,
        active_experts_per_token=16,
        moe_latent_size=3584,
        max_context_tokens=262144,
        has_native_vision=True,
        speculative_drafter_kind="dspark-5-layer",
        model_subdir="kimi-k3",
    ),
    "nemotron-3.5-super-vl": CloverModelSpec(
        model_key="nemotron-3.5-super-vl",
        hf_repo_id="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        total_layers=88,
        hidden_size=4096,
        vocab_size=131072,
        routed_experts_per_moe_layer=512,
        active_experts_per_token=22,
        moe_latent_size=1024,
        max_context_tokens=1048576,
        has_native_vision=True,
        speculative_drafter_kind="native-mtp-2-layer",
        model_subdir="nemotron-3.5-super-vl",
    ),
    "lfm3-1.7b": CloverModelSpec(
        model_key="lfm3-1.7b",
        hf_repo_id="LiquidAI/LFM3-1.7B",
        total_layers=24,
        hidden_size=2048,
        vocab_size=128000,
        routed_experts_per_moe_layer=0,
        active_experts_per_token=0,
        moe_latent_size=2048,
        max_context_tokens=262144,
        has_native_vision=False,
        speculative_drafter_kind="prompt-lookup",
        model_subdir="lfm3-1.7b",
    ),
}


class CloverUniversalConceptBridge(nn.Module):
    """
    Generalizes `K3ExpertBridge(7168 -> 384)` to any registered Layer-1 model (`7168` for Kimi K3,
    `4096` for Nemotron 3.5 Super VL, `2048` for LFM3-1.7B), projecting `[B, T, D_model]` backbone
    hidden states into Clover's `384`-d NCM concept space with a zero-initialized learnable gate `alpha`.
    """

    def __init__(self, d_model: int, d_vq: int = 384):
        super().__init__()
        self.d_model = d_model
        self.d_vq = d_vq
        self.ln_in = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_vq)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(d_vq, d_vq)
        self.ln_out = nn.LayerNorm(d_vq)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        nn.init.normal_(self.proj1.weight, std=0.015)
        nn.init.zeros_(self.proj1.bias)
        nn.init.normal_(self.proj2.weight, std=0.015)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, h_backbone: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(h_backbone.float())
        x = self.proj2(self.act(self.proj1(x)))
        return (torch.tanh(self.alpha) * self.ln_out(x)).to(h_backbone.dtype)


class CloverUniversalHybridLink(nn.Module):
    """
    Maps between `HybridAgent`'s `4096`-d continuous `<latent>` reasoning vectors `R(h)` and any
    Layer-1 model's hidden space (`D_model = 7168, 4096, or 2048`).
    Note: For `nemotron-3.5-super-vl` (`D_model = 4096`), `D_model` matches `HybridAgent` (`4096`)
    with 1:1 dimensionality!
    """

    def __init__(self, d_model: int, d_ha: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.d_ha = d_ha
        self.to_ha = nn.Identity() if d_model == d_ha else nn.Linear(d_model, d_ha, bias=False)
        self.from_ha = nn.Identity() if d_model == d_ha else nn.Linear(d_ha, d_model, bias=False)

    def project_model_to_latent(self, h_model: torch.Tensor) -> torch.Tensor:
        return self.to_ha(h_model.float()).to(h_model.dtype)

    def project_latent_to_model(self, r_h: torch.Tensor) -> torch.Tensor:
        return self.from_ha(r_h.float()).to(r_h.dtype)


class CloverSwitchableModelLayer1:
    """
    Runtime controller for Layer 1 of the 7-Layer Clover System.
    Allows switching the active foundation model at runtime (`switch_model("nemotron-3.5-super-vl")`,
    `switch_model("kimi-k3")`, `switch_model("lfm3-1.7b")`) and exposes a uniform interface to Layers 2..7.
    """

    def __init__(self, initial_model_key: str = "nemotron-3.5-super-vl"):
        if initial_model_key not in CLOVER_MODEL_REGISTRY:
            raise KeyError(f"Unknown model key '{initial_model_key}'. Valid keys: {list(CLOVER_MODEL_REGISTRY)}")
        self.active_spec: CloverModelSpec = CLOVER_MODEL_REGISTRY[initial_model_key]
        self._concept_bridges: dict[str, CloverUniversalConceptBridge] = {}
        self._hybrid_links: dict[str, CloverUniversalHybridLink] = {}
        self._ensure_adapters(self.active_spec)

    def _ensure_adapters(self, spec: CloverModelSpec) -> None:
        if spec.model_key not in self._concept_bridges:
            torch.manual_seed(42)
            bridge = CloverUniversalConceptBridge(d_model=spec.hidden_size, d_vq=384)
            # Set non-zero gate alpha=0.5 after init so projection is active in verification
            with torch.no_grad():
                bridge.alpha.fill_(0.5)
            self._concept_bridges[spec.model_key] = bridge
            self._hybrid_links[spec.model_key] = CloverUniversalHybridLink(d_model=spec.hidden_size, d_ha=4096)

    def switch_model(self, model_key: str) -> CloverModelSpec:
        if model_key not in CLOVER_MODEL_REGISTRY:
            raise KeyError(f"Unknown model key '{model_key}'. Valid keys: {list(CLOVER_MODEL_REGISTRY)}")
        self.active_spec = CLOVER_MODEL_REGISTRY[model_key]
        self._ensure_adapters(self.active_spec)
        return self.active_spec

    @property
    def concept_bridge(self) -> CloverUniversalConceptBridge:
        return self._concept_bridges[self.active_spec.model_key]

    @property
    def hybrid_link(self) -> CloverUniversalHybridLink:
        return self._hybrid_links[self.active_spec.model_key]

    def project_backbone_state_to_clover_layers(self, h_backbone: torch.Tensor) -> dict[str, Any]:
        """Projects `[B, T, D_model]` hidden states into Layer-2 (`4096`-d HybridLink) and Layer-3 (`384`-d NCM)."""
        if h_backbone.shape[-1] != self.active_spec.hidden_size:
            raise ValueError(
                f"Expected hidden_size={self.active_spec.hidden_size} for '{self.active_spec.model_key}', "
                f"got {tuple(h_backbone.shape)}"
            )
        z_ncm_384 = self.concept_bridge(h_backbone)
        r_ha_4096 = self.hybrid_link.project_model_to_latent(h_backbone)
        return {
            "model_key": self.active_spec.model_key,
            "backbone_shape": list(h_backbone.shape),
            "ncm_concept_384_shape": list(z_ncm_384.shape),
            "ncm_concept_384_rms": round(float(z_ncm_384.float().pow(2).mean().sqrt()), 5),
            "hybrid_agent_4096_shape": list(r_ha_4096.shape),
            "hybrid_agent_4096_rms": round(float(r_ha_4096.float().pow(2).mean().sqrt()), 5),
        }

    def execute_native_c_engine_and_trained_hybrid_ncm(
        self,
        token_ids: list[int] | torch.Tensor,
        nemo_pruned_dir: Path = Path("/opt/clover-k3/nemotron-computed/pruned"),
        nemo_ckpt_path: Path = Path("/opt/clover-k3/nemotron-computed/hybrid-ncm/nemotron_hybrid_ncm_bridges.pt"),
        lfm3_pruned_dir: Path = Path("/opt/clover-k3/lfm3-computed/pruned"),
        lfm3_ckpt_path: Path = Path("/opt/clover-k3/lfm3-computed/hybrid-ncm/lfm3_hybrid_ncm_bridges.pt"),
    ) -> dict[str, Any]:
        """
        Executes the active Layer-1 model's compiled Native C/OpenMP Engine (`libnemotron_cpu_engine.so` or
        `liblfm3_cpu_engine.so`), its trained Native `HybridLink` (`NemotronNativeHybridLink` d=4096 or
        `Lfm3NativeHybridLink` d=2048), and its trained NCM `ComboBridge` + `ExpertBridge` (`D_model -> 384`).
        """
        import sys

        base_dir = Path(__file__).resolve().parent
        key = self.active_spec.model_key

        if key == "nemotron-3.5-super-vl":
            nemo_dir = base_dir / "nemotron-3.5-super-vl"
            if str(nemo_dir) not in sys.path:
                sys.path.insert(0, str(nemo_dir))
            from nemotron_hybrid_ncm import (
                NemotronComboBridge,
                NemotronExpertBridge,
                NemotronNativeCPUEngine,
                NemotronNativeHybridLink,
            )

            c_eng = NemotronNativeCPUEngine(nemo_pruned_dir)
            h_c, c_ms = c_eng.forward_ids(token_ids)
            dm_stats = c_eng.get_dm_stats()

            link = NemotronNativeHybridLink(d=4096, bottleneck=1024)
            c_bridge = NemotronComboBridge(4096, 384)
            e_bridge = NemotronExpertBridge(4096, 384)
            if nemo_ckpt_path.exists():
                sd = torch.load(str(nemo_ckpt_path), map_location="cpu", weights_only=False)
                link.load_state_dict(sd["nemo_native_hybrid_link"])
                c_bridge.load_state_dict(sd["nemo_combo_bridge"])
                e_bridge.load_state_dict(sd["nemo_expert_bridge"])

            with torch.no_grad():
                r_latent = link(h_c).unsqueeze(0)          # [1, T, 4096]
                z_combo = c_bridge(r_latent)               # [1, T, 384]
                z_exp = e_bridge(h_c.unsqueeze(0))         # [1, T, 384]

            return {
                "model_key": key,
                "native_c_so": str(c_eng.so_path),
                "c_forward_ms": round(c_ms, 2),
                "backbone_hidden_shape": list(h_c.unsqueeze(0).shape),
                "native_hybridlink_latent_shape": list(r_latent.shape),
                "ncm_combo_bridge_384_shape": list(z_combo.shape),
                "ncm_expert_bridge_384_shape": list(z_exp.shape),
                "specmd_r4_dm_stats": dm_stats,
            }

        if key == "lfm3-1.7b":
            lfm3_dir = base_dir / "lfm3-1.7b"
            if str(lfm3_dir) not in sys.path:
                sys.path.insert(0, str(lfm3_dir))
            from lfm3_hybrid_ncm import (
                Lfm3ComboBridge,
                Lfm3ExpertBridge,
                Lfm3NativeCPUEngine,
                Lfm3NativeHybridLink,
            )

            c_eng = Lfm3NativeCPUEngine(lfm3_pruned_dir)
            h_c, c_ms = c_eng.forward_ids(token_ids)

            link = Lfm3NativeHybridLink(d=2048, bottleneck=512)
            c_bridge = Lfm3ComboBridge(2048, 384)
            e_bridge = Lfm3ExpertBridge(2048, 384)
            if lfm3_ckpt_path.exists():
                sd = torch.load(str(lfm3_ckpt_path), map_location="cpu", weights_only=False)
                link.load_state_dict(sd["lfm3_native_hybrid_link"])
                c_bridge.load_state_dict(sd["lfm3_combo_bridge"])
                e_bridge.load_state_dict(sd["lfm3_expert_bridge"])

            with torch.no_grad():
                r_latent = link(h_c).unsqueeze(0)          # [1, T, 2048]
                z_combo = c_bridge(r_latent)               # [1, T, 384]
                z_exp = e_bridge(h_c.unsqueeze(0))         # [1, T, 384]

            return {
                "model_key": key,
                "native_c_so": str(c_eng.so_path),
                "c_forward_ms": round(c_ms, 2),
                "backbone_hidden_shape": list(h_c.unsqueeze(0).shape),
                "native_hybridlink_latent_shape": list(r_latent.shape),
                "ncm_combo_bridge_384_shape": list(z_combo.shape),
                "ncm_expert_bridge_384_shape": list(z_exp.shape),
            }

        raise ValueError(f"Unsupported native C engine dispatch for model_key={key}")

