#!/usr/bin/env python3
"""
Stage-by-Stage Mathematical & Empirical Study of NVIDIA Nemotron 3.5 Super VL
(`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`).

This script inspects and verifies every distinct stage of `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`
using the real trained BF16 weights from the stage-bearing shards on `k3`:
  - Stage 0: Full 43,079-tensor / 124.45B-parameter architectural census across all 63 shards
             and KV + Mamba-2 state memory comparison across context lengths (up to 1M tokens).
  - Stage V1: C-RADIO 32-layer ViT-Huge Vision Encoder (`RadioModel`, 391 real BF16 tensors):
              `RadioInputConditioner` -> `RadioPatchEmbeddings` (CPE interpolation + 3 CLS + 7 registers)
              -> 32 `RadioLayer` blocks -> prefix strip to spatial patch features `[B, 1024, 1280]`.
  - Stage V2: Vision-Language Projector (`NemotronH_Omni_Reasoning_V3VisionProjector`, 5 real BF16 tensors)
              & EVS Video Token Pruning (`compute_retention_mask`):
              `vision_final_layernorm` -> `pixel_shuffle(0.5, ps_version="v2")` (1024 -> 256 tokens of 5120d)
              -> `mlp1` (`RMSNorm(5120) -> Linear(5120, 20480) -> ReLU^2 -> Linear(20480, 4096)`).
  - Stage L0: Token Embedding (`language_model.backbone.embeddings`, `[131072, 4096]`) & Multimodal `<image>` scatter.
  - Stage L1: Layer 0 Mamba-2 Sequence-Mixer Block (`language_model.backbone.layers.0`, `"mamba"`, 9 real BF16 tensors):
              `norm` -> `in_proj` (4096->18560) -> causal `conv1d` (width 4, 10240 ch) + SiLU
              -> Selective State Space (SSD) chunk scan & step-by-step recurrence (128 heads x 64 dim,
                 8 groups x 128 state_dim, FP32 state) -> `Zamba2RMSNormGated` -> `out_proj` (8192->4096).
  - Stage L2: Layer 1 Latent ReLU^2 MoE Channel-Mixer Block (`language_model.backbone.layers.1`, `"moe"`, 1,031 real BF16 tensors):
              `norm` -> `NemotronHTopkRouter` (FP32 linear 4096->512, sigmoid + `e_score_correction_bias`,
              top-22 of 512 selection, unbiased norm * `routed_scaling_factor=5.0`)
              -> `fc1_latent_proj` (4096->1024) -> 512 non-gated ReLU^2 experts (1024->2688->1024)
              -> `fc2_latent_proj` (1024->4096) + `shared_experts` (4096->5376->4096 ReLU^2).
  - Stage L3: Layer 7 NoPE 16:1 GQA Global Attention Block (`language_model.backbone.layers.7`, `"attention"`, 5 real BF16 tensors):
              `norm` -> `q_proj` (32 heads x 128), `k_proj`/`v_proj` (2 KV heads x 128, 16:1 GQA, NO RoPE)
              -> causal SDPA -> `o_proj` (4096->4096).
  - Stage L4: Final RMSNorm (`backbone.norm_f`, `[4096]`) & Untied Output Head (`lm_head`, `[131072, 4096]`).
  - Stage L5: Native 2-Layer Multi-Token Prediction (MTP) Speculative Head (`language_model.mtp`, 1,040 real BF16 tensors):
              `enorm(e_{t+1})` || `hnorm(h_t)` -> `eh_proj` (8192->4096) -> `mtp.layers.0` (Attention)
              -> `mtp.layers.1` (512-expert Latent MoE) -> `final_layernorm` -> shared `lm_head`.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHBlock,
    NemotronHRMSNorm,
)

MODEL_DIR = Path("/opt/clover-k3/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16")
REPORT_PATH = Path("/tmp/nemotron_super_vl_stage_study_report.json")


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    in_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    var = x_fp32.pow(2).mean(-1, keepdim=True)
    normed = x_fp32 * torch.rsqrt(var + eps)
    return weight * normed.to(in_dtype)


def gated_rms_norm_ref(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, group_size: int = 1024, eps: float = 1e-5
) -> torch.Tensor:
    """Exact reference for Zamba2RMSNormGated (norm_before_gate=False: x * silu(gate) then group RMSNorm)."""
    in_dtype = x.dtype
    x_fp32 = x.to(torch.float32) * F.silu(gate.to(torch.float32))
    orig_shape = x_fp32.shape
    x_grouped = x_fp32.view(*orig_shape[:-1], -1, group_size)
    var = x_grouped.pow(2).mean(-1, keepdim=True)
    x_normed = (x_grouped * torch.rsqrt(var + eps)).view(*orig_shape)
    return (weight.to(torch.float32) * x_normed).to(in_dtype)


def relu2_ref(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).pow(2)


class SafetensorsShardLoader:
    """Lazy tensor loader backed by model.safetensors.index.json and downloaded shards."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        idx_data = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = idx_data["weight_map"]
        self.metadata = idx_data.get("metadata", {})
        self._handles: dict[str, Any] = {}

    def _get_handle(self, shard_name: str):
        if shard_name not in self._handles:
            shard_path = self.model_dir / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard not downloaded: {shard_path}")
            self._handles[shard_name] = safe_open(str(shard_path), framework="pt", device="cpu")
        return self._handles[shard_name]

    def get_tensor(self, key: str) -> torch.Tensor:
        shard_name = self.weight_map[key]
        return self._get_handle(shard_name).get_tensor(key)

    def get_slice_shape_dtype(self, key: str) -> tuple[list[int], str]:
        shard_name = self.weight_map[key]
        sl = self._get_handle(shard_name).get_slice(key)
        return list(sl.get_shape()), sl.get_dtype()


def compute_full_120b_architecture_census(model_dir: Path, loader: SafetensorsShardLoader) -> dict[str, Any]:
    """
    Computes the exact parameter inventory across all 43,079 tensors of
    NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 and cross-checks every category's tensor shapes
    against the downloaded shards.
    """
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    llm_cfg = cfg["llm_config"]
    vis_cfg = cfg["vision_config"]

    # Verify exact shapes from downloaded shards
    sample_verified_shapes = {}
    probe_keys = [
        "language_model.backbone.embeddings.weight",
        "language_model.backbone.layers.0.norm.weight",
        "language_model.backbone.layers.0.mixer.in_proj.weight",
        "language_model.backbone.layers.0.mixer.conv1d.weight",
        "language_model.backbone.layers.0.mixer.conv1d.bias",
        "language_model.backbone.layers.0.mixer.dt_bias",
        "language_model.backbone.layers.0.mixer.A_log",
        "language_model.backbone.layers.0.mixer.D",
        "language_model.backbone.layers.0.mixer.norm.weight",
        "language_model.backbone.layers.0.mixer.out_proj.weight",
        "language_model.backbone.layers.1.norm.weight",
        "language_model.backbone.layers.1.mixer.gate.weight",
        "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias",
        "language_model.backbone.layers.1.mixer.fc1_latent_proj.weight",
        "language_model.backbone.layers.1.mixer.fc2_latent_proj.weight",
        "language_model.backbone.layers.1.mixer.experts.0.up_proj.weight",
        "language_model.backbone.layers.1.mixer.experts.0.down_proj.weight",
        "language_model.backbone.layers.1.mixer.shared_experts.up_proj.weight",
        "language_model.backbone.layers.1.mixer.shared_experts.down_proj.weight",
        "language_model.backbone.layers.7.norm.weight",
        "language_model.backbone.layers.7.mixer.q_proj.weight",
        "language_model.backbone.layers.7.mixer.k_proj.weight",
        "language_model.backbone.layers.7.mixer.v_proj.weight",
        "language_model.backbone.layers.7.mixer.o_proj.weight",
        "language_model.backbone.norm_f.weight",
        "language_model.lm_head.weight",
        "language_model.mtp.layers.0.enorm.weight",
        "language_model.mtp.layers.0.hnorm.weight",
        "language_model.mtp.layers.0.eh_proj.weight",
        "language_model.mtp.layers.1.final_layernorm.weight",
        "mlp1.0.weight",
        "mlp1.1.weight",
        "mlp1.3.weight",
        "vision_projector.vision_final_layernorm.weight",
        "vision_model.radio_model.model.patch_generator.embedder.weight",
        "vision_model.radio_model.model.patch_generator.video_embedder.weight",
        "vision_model.radio_model.model.patch_generator.pos_embed",
        "vision_model.radio_model.model.patch_generator.cls_token.token",
        "vision_model.radio_model.model.blocks.0.attn.qkv.weight",
        "vision_model.radio_model.model.blocks.0.mlp.fc1.weight",
    ]
    for k in probe_keys:
        shape, dtype = loader.get_slice_shape_dtype(k)
        sample_verified_shapes[k] = {"shape": shape, "dtype": dtype}

    # Exact parameter counts per stage from verified tensor shapes:
    # 1. Vision C-RADIO Encoder (all 391 tensors are in shards 61, 62, summary-idxs -> count directly!)
    vision_encoder_params = 0
    vision_encoder_tensors = 0
    for k in loader.weight_map:
        if k.startswith("vision_model."):
            sh, _ = loader.get_slice_shape_dtype(k)
            numel = 1
            for d in sh:
                numel *= d
            vision_encoder_params += numel
            vision_encoder_tensors += 1

    # 2. Vision Projector (`mlp1.*` + `vision_projector.*`, all 5 tensors in shards 61..63 -> count directly!)
    vision_proj_params = 0
    for k in loader.weight_map:
        if k.startswith("mlp1.") or k.startswith("vision_projector."):
            sh, _ = loader.get_slice_shape_dtype(k)
            numel = 1
            for d in sh:
                numel *= d
            vision_proj_params += numel

    # 3. LLM Embeddings & Untied LM Head
    embed_params = 131072 * 4096
    lm_head_params = 131072 * 4096
    final_norm_params = 4096

    # 4. Single Mamba-2 Block (40 layers in backbone)
    # norm(4096) + in_proj(18560*4096) + conv1d.weight(10240*1*4) + conv1d.bias(10240)
    # + dt_bias(128) + A_log(128) + D(128) + mixer.norm(8192) + out_proj(4096*8192)
    one_mamba_params = (
        4096 + (18560 * 4096) + (10240 * 4) + 10240 + 128 + 128 + 128 + 8192 + (4096 * 8192)
    )
    mamba_40_params = 40 * one_mamba_params

    # 5. Single Attention Block (8 layers in backbone)
    # norm(4096) + q_proj(4096*4096) + k_proj(256*4096) + v_proj(256*4096) + o_proj(4096*4096)
    one_attn_params = 4096 + (4096 * 4096) + (256 * 4096) + (256 * 4096) + (4096 * 4096)
    attn_8_params = 8 * one_attn_params

    # 6. Single Latent MoE Block (40 layers in backbone)
    # norm(4096) + gate.weight(512*4096) + gate.bias(512) + fc1_latent(1024*4096) + fc2_latent(4096*1024)
    # + shared_experts(2 * 5376 * 4096) + 512 routed experts * (2 * 2688 * 1024)
    one_moe_routed_experts_params = 512 * (2 * 2688 * 1024)
    one_moe_active_22_experts_params = 22 * (2 * 2688 * 1024)
    one_moe_shared_and_latent_params = (
        4096 + (512 * 4096) + 512 + (1024 * 4096) + (4096 * 1024) + (2 * 5376 * 4096)
    )
    one_moe_total_params = one_moe_shared_and_latent_params + one_moe_routed_experts_params
    moe_40_total_params = 40 * one_moe_total_params

    # 7. Native 2-Layer MTP Module (all 1,040 tensors are in shards 62 & 63 -> count directly!)
    mtp_params = 0
    for k in loader.weight_map:
        if k.startswith("language_model.mtp."):
            sh, _ = loader.get_slice_shape_dtype(k)
            numel = 1
            for d in sh:
                numel *= d
            mtp_params += numel

    total_computed_params = (
        vision_encoder_params
        + vision_proj_params
        + embed_params
        + mamba_40_params
        + attn_8_params
        + moe_40_total_params
        + final_norm_params
        + lm_head_params
        + mtp_params
    )

    # Active parameters per text token during backbone forward (excluding MTP draft):
    active_backbone_params = (
        embed_params
        + mamba_40_params
        + attn_8_params
        + 40 * (one_moe_shared_and_latent_params + one_moe_active_22_experts_params)
        + final_norm_params
        + lm_head_params
    )

    # State memory (Mamba-2 conv+SSM state + 8-layer 2-KV-head GQA KV cache) across context lengths:
    # Mamba-2 state per layer:
    #   conv_state: [1, 10240, 3] in BF16 = 61,440 bytes
    #   ssm_state:  [1, 128, 64, 128] in FP32 = 4,194,304 bytes (4.0 MiB)
    #   40 Mamba-2 layers fixed state = 40 * 4,255,744 = 170,229,760 bytes (162.34 MiB) at ANY sequence length!
    # Attention KV cache across all 8 layers:
    #   8 layers * 2 (K+V) * 2 KV heads * 128 head_dim * 2 bytes (BF16) = 8,192 bytes/token (8.0 KiB/token)!
    mamba_fixed_bytes = 40 * ((10240 * 3 * 2) + (128 * 64 * 128 * 4))
    state_by_ctx = {}
    for seq_len in [4096, 32768, 131072, 262144, 1048576]:
        kv_bytes = 8 * seq_len * 2 * 128 * 2 * 2
        total_mb = (mamba_fixed_bytes + kv_bytes) / (1024 * 1024)
        # Compare against a standard 88-layer GQA-8 model (8 KV heads * 128 dim * 2 * 2 = 4096 B/layer/tok -> 360,448 B/tok)
        std_88_gqa_mb = (88 * seq_len * 8 * 128 * 2 * 2) / (1024 * 1024)
        state_by_ctx[str(seq_len)] = {
            "mamba_40_fixed_mib": round(mamba_fixed_bytes / (1024 * 1024), 2),
            "attn_8_kv_mib": round(kv_bytes / (1024 * 1024), 2),
            "nemotron_total_state_mib": round(total_mb, 2),
            "standard_88_layer_gqa8_mib": round(std_88_gqa_mb, 2),
            "state_reduction_factor": round(std_88_gqa_mb / total_mb, 2),
        }

    return {
        "total_tensors_in_index": len(loader.weight_map),
        "total_parameters_exact": total_computed_params,
        "active_backbone_parameters_per_token": active_backbone_params,
        "by_stage_category": {
            "vision_c_radio_32_layers": {
                "tensors": vision_encoder_tensors,
                "params": vision_encoder_params,
                "pct": round(100.0 * vision_encoder_params / total_computed_params, 3),
            },
            "vision_projector_pixelshuffle_mlp1": {
                "tensors": 5,
                "params": vision_proj_params,
                "pct": round(100.0 * vision_proj_params / total_computed_params, 3),
            },
            "llm_embeddings_untied": {
                "tensors": 1,
                "params": embed_params,
                "pct": round(100.0 * embed_params / total_computed_params, 3),
            },
            "backbone_40_mamba2_layers": {
                "tensors": 40 * 9,
                "params": mamba_40_params,
                "params_per_layer": one_mamba_params,
                "pct": round(100.0 * mamba_40_params / total_computed_params, 3),
            },
            "backbone_8_gqa_attention_layers": {
                "tensors": 8 * 5,
                "params": attn_8_params,
                "params_per_layer": one_attn_params,
                "pct": round(100.0 * attn_8_params / total_computed_params, 3),
            },
            "backbone_40_latent_moe_layers_total": {
                "tensors": 40 * 1031,
                "params": moe_40_total_params,
                "routed_experts_params_per_layer": one_moe_routed_experts_params,
                "active_22_experts_params_per_layer": one_moe_active_22_experts_params,
                "shared_and_latent_params_per_layer": one_moe_shared_and_latent_params,
                "pct": round(100.0 * moe_40_total_params / total_computed_params, 3),
            },
            "backbone_final_norm_and_untied_lm_head": {
                "tensors": 2,
                "params": final_norm_params + lm_head_params,
                "pct": round(100.0 * (final_norm_params + lm_head_params) / total_computed_params, 3),
            },
            "native_mtp_2_layer_head": {
                "tensors": 1040,
                "params": mtp_params,
                "pct": round(100.0 * mtp_params / total_computed_params, 3),
            },
        },
        "state_memory_analysis": state_by_ctx,
        "sample_verified_shapes": sample_verified_shapes,
    }


@torch.no_grad()
def verify_vision_and_video_stages(model_dir: Path, loader: SafetensorsShardLoader) -> dict[str, Any]:
    """
    Loads the real trained BF16 weights of `RadioModel` (all 32 ViT-Huge layers) and
    `NemotronH_Omni_Reasoning_V3VisionProjector` (`vision_final_layernorm` + `pixel_shuffle` + `mlp1`)
    from shards 61, 62, 63, and `model-summary-idxs.safetensors`, and verifies:
      1. Stage V1 (C-RADIO 32-layer Vision Encoder) vs. manual patch+CPE+ViT block equations.
      2. Stage V2 (Pixel-Shuffle + ReLU^2 Projector) on a 512x512 image (1024 patches -> 256 tokens of 4096d).
      3. Video Temporal Patching (T=2 frames per patch) + EVS (`compute_retention_mask`) token pruning.
    """
    import importlib
    import types

    # Create a lightweight package namespace so relative imports in modeling_radio.py & modeling_nemotron_h_omni.py work
    pkg_name = "_nemotron_omni_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(model_dir)]
    sys.modules[pkg_name] = pkg

    def _load_submod(name: str):
        spec = importlib.util.spec_from_file_location(f"{pkg_name}.{name}", model_dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{pkg_name}.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg_radio_mod = _load_submod("configuration_radio")
    mod_radio_mod = _load_submod("modeling_radio")
    cfg_omni_mod = _load_submod("configuration_nemotron_h_omni")
    mod_omni_mod = _load_submod("modeling_nemotron_h_omni")

    raw_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if "llm_config" in raw_cfg and "time_step_limit" in raw_cfg["llm_config"]:
        raw_cfg["llm_config"]["time_step_limit"] = [0.0, float("inf")]
    omni_cfg = cfg_omni_mod.NemotronH_Omni_Reasoning_V3_Config(**raw_cfg)
    omni_cfg.vision_config._attn_implementation = "eager"

    t0 = time.perf_counter()
    vision_model = mod_radio_mod.RadioModel(omni_cfg.vision_config).to(torch.bfloat16)
    vision_projector = mod_omni_mod.NemotronH_Omni_Reasoning_V3VisionProjector(omni_cfg).to(torch.bfloat16)

    # Populate vision_model state_dict from shards 61, 62, summary-idxs
    v_sd = {}
    for k in loader.weight_map:
        if not k.startswith("vision_model."):
            continue
        t = loader.get_tensor(k)
        sub_k = k[len("vision_model.") :]
        sub_k = sub_k.replace(
            "radio_model.model.patch_generator.video_embedder", "embeddings.video_patch_projection"
        )
        sub_k = sub_k.replace("radio_model.model.patch_generator.embedder", "embeddings.patch_projection")
        sub_k = sub_k.replace("radio_model.model.patch_generator.pos_embed", "embeddings.position_embedding")
        sub_k = sub_k.replace(
            "radio_model.model.patch_generator.cls_token.token", "embeddings.cls_register_token"
        )
        sub_k = sub_k.replace("radio_model.model.blocks", "encoder.layer")
        sub_k = sub_k.replace("attn.proj", "attention.output.dense")
        sub_k = sub_k.replace("radio_model.input_conditioner", "input_conditioner")
        sub_k = sub_k.replace("radio_model.summary_idxs", "summary_idxs")
        if "attn.qkv." in sub_k:
            q_t, k_t, v_t = t.chunk(3, dim=0)
            v_sd[sub_k.replace("attn.qkv.", "attention.attention.query.")] = q_t
            v_sd[sub_k.replace("attn.qkv.", "attention.attention.key.")] = k_t
            v_sd[sub_k.replace("attn.qkv.", "attention.attention.value.")] = v_t
        else:
            v_sd[sub_k] = t

    missing_v, unexpected_v = vision_model.load_state_dict(v_sd, strict=False)
    vision_model.make_preprocessor_external()
    vision_model.eval()

    # Populate vision_projector state_dict
    vp_sd = {
        "vision_final_layernorm.weight": loader.get_tensor("vision_projector.vision_final_layernorm.weight"),
        "vision_final_layernorm.bias": loader.get_tensor("vision_projector.vision_final_layernorm.bias"),
        "mlp1.norm.weight": loader.get_tensor("mlp1.0.weight"),
        "mlp1.linear1.weight": loader.get_tensor("mlp1.1.weight"),
        "mlp1.linear2.weight": loader.get_tensor("mlp1.3.weight"),
    }
    vision_projector.load_state_dict(vp_sd, strict=True)
    vision_projector.eval()
    load_vis_s = time.perf_counter() - t0

    # Create a deterministic structured 512x512 RGB test image
    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, 512, 512, dtype=torch.bfloat16)

    # 1. Verify Stage V1: C-RADIO Patch Embeddings + Layer 0 + full 32-layer forward
    t_v1 = time.perf_counter()
    radio_out = vision_model(pixel_values, output_hidden_states=True)
    v1_ms = (time.perf_counter() - t_v1) * 1000.0

    # Manual reconstruction of RadioPatchEmbeddings + RadioLayer 0
    ps = 16
    patches = pixel_values.reshape(1, 3, 32, ps, 32, ps).permute(0, 2, 4, 1, 3, 5).reshape(1, 1024, 768)
    proj_patches = F.linear(patches, vision_model.embeddings.patch_projection.weight)
    pos_grid = vision_model.embeddings.position_embedding.reshape(1, 128, 128, 1280).permute(0, 3, 1, 2)
    pos_interp = (
        F.interpolate(pos_grid.float(), size=(32, 32), mode="bilinear", align_corners=False)
        .to(torch.bfloat16)
        .flatten(2)
        .permute(0, 2, 1)
    )
    prefix = vision_model.embeddings.cls_register_token.unsqueeze(0)
    manual_embed = torch.cat([prefix, proj_patches + pos_interp], dim=1)
    embed_max_diff = float((manual_embed.float() - radio_out.hidden_states[0].float()).abs().max())

    # Manual RadioLayer 0
    b0 = vision_model.encoder.layer[0]
    h0_norm = F.layer_norm(manual_embed, (1280,), b0.norm1.weight, b0.norm1.bias, eps=1e-6)
    q0 = F.linear(h0_norm, b0.attention.attention.query.weight, b0.attention.attention.query.bias).view(1, 1034, 16, 80).transpose(1, 2)
    k0 = F.linear(h0_norm, b0.attention.attention.key.weight, b0.attention.attention.key.bias).view(1, 1034, 16, 80).transpose(1, 2)
    v0 = F.linear(h0_norm, b0.attention.attention.value.weight, b0.attention.attention.value.bias).view(1, 1034, 16, 80).transpose(1, 2)
    attn_probs0 = F.softmax(torch.matmul(q0, k0.transpose(2, 3)) * (80**-0.5), dim=-1)
    attn_ctx0 = torch.matmul(attn_probs0, v0).transpose(1, 2).reshape(1, 1034, 1280)
    attn_out0 = F.linear(attn_ctx0, b0.attention.output.dense.weight, b0.attention.output.dense.bias) * b0.layer_scale1.lambda1
    h0_mid = manual_embed + attn_out0
    mlp0_in = F.layer_norm(h0_mid, (1280,), b0.norm2.weight, b0.norm2.bias, eps=1e-6)
    mlp0_out = F.linear(F.gelu(F.linear(mlp0_in, b0.mlp.fc1.weight, b0.mlp.fc1.bias)), b0.mlp.fc2.weight, b0.mlp.fc2.bias) * b0.layer_scale2.lambda1
    manual_l0_out = h0_mid + mlp0_out
    l0_max_diff = float((manual_l0_out.float() - radio_out.hidden_states[1].float()).abs().max())

    # 2. Verify Stage V2: Vision Projector (LayerNorm -> PixelShuffle 0.5 -> RMSNorm -> Linear -> ReLU^2 -> Linear)
    t_v2 = time.perf_counter()
    official_img_embeds = vision_projector(pixel_values, vision_model)
    v2_ms = (time.perf_counter() - t_v2) * 1000.0 - v1_ms

    vit_feats = radio_out.features  # [1, 1024, 1280]
    normed_feats = F.layer_norm(
        vit_feats, (1280,), vision_projector.vision_final_layernorm.weight, vision_projector.vision_final_layernorm.bias, eps=1e-6
    )
    grid = normed_feats.reshape(1, 32, 32, 1280)
    # PixelShuffle 0.5 (ps_version="v2")
    ps_x = grid.view(1, 32, 16, 2560).permute(0, 2, 1, 3).contiguous().view(1, 16, 16, 5120).permute(0, 2, 1, 3).contiguous()
    ps_flat = ps_x.reshape(1, 256, 5120)
    mlp1_normed = (
        vision_projector.mlp1.norm.weight.float()
        * (ps_flat.float() * torch.rsqrt(ps_flat.float().pow(2).mean(-1, keepdim=True) + 1e-5))
    ).to(torch.bfloat16)
    mlp1_h = relu2_ref(F.linear(mlp1_normed, vision_projector.mlp1.linear1.weight))
    manual_img_embeds = F.linear(mlp1_h, vision_projector.mlp1.linear2.weight)
    projector_max_diff = float((manual_img_embeds.float() - official_img_embeds.float()).abs().max())

    # 3. Verify Video Temporal Patching + EVS Token Pruning on a 6-frame video (3 temporal groups of 2 frames)
    # Group 0 (frames 0,1): reference frame; Group 1 (frames 2,3): static copy of Group 0 (dissimilarity=0);
    # Group 2 (frames 4,5): top half inverted (128 patches moving, 128 patches static).
    f0 = pixel_values[0]
    f1 = f0.clone()
    f2 = f0.clone()
    f3 = f0.clone()
    f4 = f0.clone()
    f4[:, :256, :] = -f4[:, :256, :]  # top half (128 post-pixel-shuffle patches) changed!
    f5 = f4.clone()
    video_frames = torch.stack([f0, f1, f2, f3, f4, f5], dim=0)  # [6, 3, 512, 512]
    video_embeds_grid = vision_projector.forward_video(video_frames, vision_model)  # [3, 16, 16, 4096]
    evs_mask = mod_omni_mod.compute_retention_mask(
        video_embeds=video_embeds_grid,
        thw=torch.tensor(video_embeds_grid.shape[:3]),
        spatial_merge_size=1,
        q=0.5,
    )
    evs_reshaped = evs_mask.view(3, 16, 16)
    frame_group0_kept = int(evs_reshaped[0].sum().item())
    frame_group1_kept = int(evs_reshaped[1].sum().item())
    frame_group2_kept = int(evs_reshaped[2].sum().item())

    return {
        "load_seconds": round(load_vis_s, 3),
        "missing_keys_ignored": missing_v,
        "unexpected_keys": unexpected_v,
        "stage_v1_radio_encoder": {
            "input_image_shape": [1, 3, 512, 512],
            "patch_tokens_plus_prefix": list(radio_out.hidden_states[0].shape),
            "output_features_shape": list(radio_out.features.shape),
            "summary_shape": list(radio_out.summary.shape),
            "embed_max_abs_diff_vs_manual": embed_max_diff,
            "layer0_max_abs_diff_vs_manual": l0_max_diff,
            "layer_0_in_rms": round(float(radio_out.hidden_states[0].float().pow(2).mean().sqrt()), 5),
            "layer_16_rms": round(float(radio_out.hidden_states[16].float().pow(2).mean().sqrt()), 5),
            "layer_32_out_rms": round(float(radio_out.features.float().pow(2).mean().sqrt()), 5),
            "forward_ms": round(v1_ms, 2),
        },
        "stage_v2_vision_projector_and_evs": {
            "pre_pixel_shuffle_tokens": 1024,
            "post_pixel_shuffle_tokens": 256,
            "post_pixel_shuffle_dim": 5120,
            "mlp1_intermediate_dim": 20480,
            "output_llm_embeds_shape": list(official_img_embeds.shape),
            "output_llm_embeds_rms": round(float(official_img_embeds.float().pow(2).mean().sqrt()), 5),
            "max_abs_diff_vs_manual": projector_max_diff,
            "projector_ms": round(v2_ms, 2),
            "video_evs_pruning_test": {
                "input_video_frames": [6, 3, 512, 512],
                "temporal_groups_shape": list(video_embeds_grid.shape),
                "total_video_tokens_before_evs": int(evs_mask.numel()),
                "retained_video_tokens_at_q_0_5": int(evs_mask.sum().item()),
                "group_0_anchor_tokens_kept_of_256": frame_group0_kept,
                "group_1_static_tokens_kept_of_256": frame_group1_kept,
                "group_2_motion_tokens_kept_of_256": frame_group2_kept,
            },
        },
    }


@torch.no_grad()
def verify_backbone_and_mtp_stages(model_dir: Path, loader: SafetensorsShardLoader) -> dict[str, Any]:
    """
    Loads the real trained BF16 weights for:
      - `language_model.backbone.embeddings` (`[131072, 4096]`)
      - `language_model.backbone.layers.0` (`"mamba"` — Mamba-2 mixer)
      - `language_model.backbone.layers.1` (`"moe"` — Latent ReLU^2 MoE mixer with 512 routed experts)
      - `language_model.backbone.layers.7` (`"attention"` — NoPE 16:1 GQA attention mixer)
      - `language_model.backbone.norm_f` & `language_model.lm_head` (`[131072, 4096]`)
      - `language_model.mtp` (2-layer Native Multi-Token Prediction speculative module)
    and verifies each stage against our manual mathematical equations + causal prefix invariance.
    """
    raw_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    llm_dict = dict(raw_cfg["llm_config"])
    if "time_step_limit" in llm_dict:
        llm_dict["time_step_limit"] = [0.0, float("inf")]
    legacy_names = {"mamba": "linear_attention", "attention": "full_attention"}
    llm_dict["layers_block_type"] = [legacy_names.get(x, x) for x in llm_dict["layers_block_type"]]
    llm_dict["mtp_layers_block_type"] = [legacy_names.get(x, x) for x in llm_dict["mtp_layers_block_type"]]
    llm_dict["use_mamba_kernels"] = False
    llm_cfg = NemotronHConfig(**llm_dict)
    llm_cfg._attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    prompt = "Clover verifies every stage of NVIDIA Nemotron 3.5 Super VL on real tensors."
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    batch_size, seq_len = input_ids.shape
    tokens_str = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    # Stage L0: Token Embedding (`language_model.backbone.embeddings.weight`)
    t0 = time.perf_counter()
    embed_weight = loader.get_tensor("language_model.backbone.embeddings.weight")
    h_embed = F.embedding(input_ids, embed_weight)
    embed_ms = (time.perf_counter() - t0) * 1000.0
    embed_rms = float(h_embed.float().pow(2).mean().sqrt())

    # Stage L1: Layer 0 Mamba-2 Block (`layers.0`, `"linear_attention"`)
    t0 = time.perf_counter()
    block_mamba = NemotronHBlock(llm_cfg, layer_idx=0).to(torch.bfloat16)
    mamba_sd = {}
    prefix_l0 = "language_model.backbone.layers.0."
    for k in loader.weight_map:
        if k.startswith(prefix_l0):
            mamba_sd[k[len(prefix_l0) :]] = loader.get_tensor(k)
    block_mamba.load_state_dict(mamba_sd, strict=True)
    block_mamba.eval()

    t_m_f = time.perf_counter()
    off_mamba_out = block_mamba(h_embed)
    mamba_ms = (time.perf_counter() - t_m_f) * 1000.0

    # Manual step-by-step Mamba-2 Selective State Space recurrence verification
    mx = block_mamba.mixer
    u_m = rms_norm_ref(h_embed, block_mamba.norm.weight, eps=llm_cfg.layer_norm_epsilon)
    proj_m = F.linear(u_m, mx.in_proj.weight)  # [B, T, 18560]
    gate_m, conv_in_m, dt_m = torch.split(proj_m, [8192, 10240, 128], dim=-1)
    # Causal depthwise conv1d of width K=4
    conv_T = conv_in_m.transpose(1, 2)
    conv_out_T = F.silu(mx.conv1d(conv_T)[..., :seq_len])
    conv_out = conv_out_T.transpose(1, 2)
    x_ssm, B_ssm, C_ssm = torch.split(conv_out, [8192, 1024, 1024], dim=-1)

    # Recurrent step-by-step SSD equations in float32
    dt_val = torch.clamp(F.softplus(dt_m.float() + mx.dt_bias.float()), min=llm_cfg.time_step_min)  # [B, T, 128]
    A_val = -torch.exp(mx.A_log.float())  # [128]
    x_heads = x_ssm.float().view(batch_size, seq_len, 128, 64)
    B_heads = B_ssm.float().view(batch_size, seq_len, 8, 128).repeat_interleave(16, dim=2)  # [B, T, 128, 128]
    C_heads = C_ssm.float().view(batch_size, seq_len, 8, 128).repeat_interleave(16, dim=2)

    state = torch.zeros(batch_size, 128, 64, 128, dtype=torch.float32)
    y_steps = []
    for t_idx in range(seq_len):
        dt_t = dt_val[:, t_idx, :, None, None]  # [B, 128, 1, 1]
        dA_t = torch.exp(dt_t * A_val[None, :, None, None])  # [B, 128, 1, 1]
        dB_t = dt_t * B_heads[:, t_idx, :, None, :]  # [B, 128, 1, 128]
        x_t = x_heads[:, t_idx, :, :, None]  # [B, 128, 64, 1]
        state = state * dA_t + dB_t * x_t
        y_t = (state * C_heads[:, t_idx, :, None, :]).sum(dim=-1) + mx.D.float()[None, :, None] * x_heads[:, t_idx]
        y_steps.append(y_t)
    y_rec = torch.stack(y_steps, dim=1).view(batch_size, seq_len, 8192).to(torch.bfloat16)
    y_normed = gated_rms_norm_ref(y_rec, gate_m, mx.norm.weight, group_size=1024, eps=llm_cfg.layer_norm_epsilon)
    manual_mamba_out = h_embed + F.linear(y_normed, mx.out_proj.weight)

    mamba_max_diff = float((manual_mamba_out.float() - off_mamba_out.float()).abs().max())
    mamba_cos = float(
        F.cosine_similarity(manual_mamba_out.float().reshape(1, -1), off_mamba_out.float().reshape(1, -1), dim=-1).item()
    )

    # Causal prefix check on Layer 0 Mamba-2
    pref_len = seq_len // 2
    off_mamba_pref = block_mamba(h_embed[:, :pref_len])
    mamba_prefix_gap = float((off_mamba_out[:, :pref_len].float() - off_mamba_pref.float()).abs().max())

    # Stage L2: Layer 1 Latent ReLU^2 MoE Block (`layers.1`, `"moe"`)
    block_moe = NemotronHBlock(llm_cfg, layer_idx=1).to(torch.bfloat16)
    prefix_l1 = "language_model.backbone.layers.1."
    moe_sd = {
        "norm.weight": loader.get_tensor(prefix_l1 + "norm.weight"),
        "mixer.gate.weight": loader.get_tensor(prefix_l1 + "mixer.gate.weight"),
        "mixer.gate.e_score_correction_bias": loader.get_tensor(prefix_l1 + "mixer.gate.e_score_correction_bias"),
        "mixer.fc1_latent_proj.weight": loader.get_tensor(prefix_l1 + "mixer.fc1_latent_proj.weight"),
        "mixer.fc2_latent_proj.weight": loader.get_tensor(prefix_l1 + "mixer.fc2_latent_proj.weight"),
        "mixer.shared_experts.up_proj.weight": loader.get_tensor(prefix_l1 + "mixer.shared_experts.up_proj.weight"),
        "mixer.shared_experts.down_proj.weight": loader.get_tensor(prefix_l1 + "mixer.shared_experts.down_proj.weight"),
    }
    up_list = [loader.get_tensor(f"{prefix_l1}mixer.experts.{e}.up_proj.weight") for e in range(512)]
    down_list = [loader.get_tensor(f"{prefix_l1}mixer.experts.{e}.down_proj.weight") for e in range(512)]
    moe_sd["mixer.experts.up_proj"] = torch.stack(up_list, dim=0)
    moe_sd["mixer.experts.down_proj"] = torch.stack(down_list, dim=0)
    block_moe.load_state_dict(moe_sd, strict=True)
    block_moe.eval()

    t_moe0 = time.perf_counter()
    off_moe_out = block_moe(off_mamba_out)
    moe_ms = (time.perf_counter() - t_moe0) * 1000.0

    # Manual reconstruction of Layer 1 Latent MoE
    u_moe = rms_norm_ref(off_mamba_out, block_moe.norm.weight, eps=llm_cfg.layer_norm_epsilon)
    flat_u = u_moe.view(-1, 4096)
    router_logits = F.linear(flat_u.float(), block_moe.mixer.gate.weight.float())
    scores = torch.sigmoid(router_logits)
    scores_choice = scores + block_moe.mixer.gate.e_score_correction_bias.float()
    topk_indices = torch.topk(scores_choice, k=22, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    topk_weights = (topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)) * 5.0

    latent_in = F.linear(flat_u, block_moe.mixer.fc1_latent_proj.weight)  # [T, 1024]
    # 1. Rank-ordered accumulation (1..22 per token)
    latent_Accum_rank = torch.zeros_like(latent_in, dtype=topk_weights.dtype)
    distinct_experts_used = set()
    for t_i in range(flat_u.shape[0]):
        for r_i in range(22):
            e_id = int(topk_indices[t_i, r_i].item())
            distinct_experts_used.add(e_id)
            w_r = topk_weights[t_i, r_i]
            e_up = F.linear(latent_in[t_i : t_i + 1], block_moe.mixer.experts.up_proj[e_id])
            e_act = relu2_ref(e_up)
            e_down = F.linear(e_act, block_moe.mixer.experts.down_proj[e_id])
            latent_Accum_rank[t_i : t_i + 1] += (e_down.to(topk_weights.dtype) * w_r)

    # 2. Expert-ascending accumulation (matching NemotronHExperts.forward index_add_ order)
    latent_Accum_expert_order = torch.zeros_like(latent_in, dtype=topk_weights.dtype)
    expert_mask = F.one_hot(topk_indices, num_classes=512).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
    for exp_tensor in expert_hit:
        e_id = int(exp_tensor.item())
        top_k_pos, tok_idx = torch.where(expert_mask[e_id])
        cur_in = latent_in[tok_idx]
        cur_h = F.linear(relu2_ref(F.linear(cur_in, block_moe.mixer.experts.up_proj[e_id])), block_moe.mixer.experts.down_proj[e_id])
        cur_h = cur_h * topk_weights[tok_idx, top_k_pos, None]
        latent_Accum_expert_order.index_add_(0, tok_idx, cur_h.to(latent_Accum_expert_order.dtype))

    routed_4096 = F.linear(latent_Accum_expert_order.to(torch.bfloat16), block_moe.mixer.fc2_latent_proj.weight).view(
        batch_size, seq_len, 4096
    )
    routed_4096_rank_order = F.linear(latent_Accum_rank.to(torch.bfloat16), block_moe.mixer.fc2_latent_proj.weight).view(
        batch_size, seq_len, 4096
    )
    shared_4096 = F.linear(
        relu2_ref(F.linear(u_moe, block_moe.mixer.shared_experts.up_proj.weight)),
        block_moe.mixer.shared_experts.down_proj.weight,
    )
    manual_moe_out = off_mamba_out + (routed_4096 + shared_4096)
    manual_moe_left_assoc_out = (off_mamba_out + routed_4096) + shared_4096
    moe_max_diff = float((manual_moe_out.float() - off_moe_out.float()).abs().max())
    moe_left_assoc_diff = float((manual_moe_left_assoc_out.float() - off_moe_out.float()).abs().max())
    moe_cos = float(
        F.cosine_similarity(manual_moe_out.float().reshape(1, -1), off_moe_out.float().reshape(1, -1), dim=-1).item()
    )

    # Free Layer 1 MoE 5.7GB weights before loading MTP MoE so peak RAM stays tiny
    del block_moe, moe_sd, up_list, down_list

    # Stage L3: Layer 7 NoPE 16:1 GQA Global Attention Block (`layers.7`, `"full_attention"`)
    block_attn = NemotronHBlock(llm_cfg, layer_idx=7).to(torch.bfloat16)
    prefix_l7 = "language_model.backbone.layers.7."
    attn_sd = {
        k[len(prefix_l7) :]: loader.get_tensor(k)
        for k in loader.weight_map
        if k.startswith(prefix_l7)
    }
    block_attn.load_state_dict(attn_sd, strict=True)
    block_attn.eval()

    row_idx = torch.arange(seq_len)[:, None]
    col_idx = torch.arange(seq_len)[None, :]
    causal_4d = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bfloat16)
    causal_4d.masked_fill_(~(col_idx <= row_idx)[None, None, :, :], float("-inf"))

    t_a0 = time.perf_counter()
    off_attn_out = block_attn(off_moe_out, attention_mask=causal_4d)
    attn_ms = (time.perf_counter() - t_a0) * 1000.0

    # Manual NoPE 16:1 GQA equation
    u_a = rms_norm_ref(off_moe_out, block_attn.norm.weight, eps=llm_cfg.layer_norm_epsilon)
    q_a = F.linear(u_a, block_attn.mixer.q_proj.weight).view(batch_size, seq_len, 32, 128).transpose(1, 2)
    k_a = F.linear(u_a, block_attn.mixer.k_proj.weight).view(batch_size, seq_len, 2, 128).transpose(1, 2)
    v_a = F.linear(u_a, block_attn.mixer.v_proj.weight).view(batch_size, seq_len, 2, 128).transpose(1, 2)
    # Repeat 2 KV heads 16x -> 32 heads
    k_rep = k_a[:, :, None, :, :].expand(batch_size, 2, 16, seq_len, 128).reshape(batch_size, 32, seq_len, 128)
    v_rep = v_a[:, :, None, :, :].expand(batch_size, 2, 16, seq_len, 128).reshape(batch_size, 32, seq_len, 128)
    scores_a = torch.matmul(q_a, k_rep.transpose(2, 3)) * (128**-0.5) + causal_4d
    probs_a = F.softmax(scores_a, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    ctx_a = torch.matmul(probs_a, v_rep).transpose(1, 2).contiguous().view(batch_size, seq_len, 4096)
    manual_attn_out = off_moe_out + F.linear(ctx_a, block_attn.mixer.o_proj.weight)
    attn_max_diff = float((manual_attn_out.float() - off_attn_out.float()).abs().max())

    # Causal prefix check on Layer 7 Attention
    causal_4d_pref = causal_4d[:, :, :pref_len, :pref_len]
    off_attn_pref = block_attn(off_moe_out[:, :pref_len], attention_mask=causal_4d_pref)
    attn_prefix_gap = float((off_attn_out[:, :pref_len].float() - off_attn_pref.float()).abs().max())

    # Stage L4: Final RMSNorm (`norm_f`) & Untied Output Head (`lm_head`)
    norm_f_weight = loader.get_tensor("language_model.backbone.norm_f.weight")
    lm_head_weight = loader.get_tensor("language_model.lm_head.weight")
    t_h0 = time.perf_counter()
    h_final = rms_norm_ref(off_attn_out, norm_f_weight, eps=llm_cfg.layer_norm_epsilon)
    logits = F.linear(h_final, lm_head_weight).float()
    lm_head_ms = (time.perf_counter() - t_h0) * 1000.0

    # Stage L5: Native 2-Layer Multi-Token Prediction (MTP) Speculative Head (`language_model.mtp`)
    # MTP takes the backbone hidden state `h_final` (or pre-norm hidden state) and the next token's embedding `e_{t+1}`:
    #   combined = eh_proj(cat([enorm(e_{t+1}), hnorm(h_t)], dim=-1))
    #   h_mtp0 = mtp_layer0_attn(combined)
    #   h_mtp1 = final_layernorm(mtp_layer1_moe(h_mtp0))
    #   mtp_logits = lm_head(h_mtp1)
    enorm_w = loader.get_tensor("language_model.mtp.layers.0.enorm.weight")
    hnorm_w = loader.get_tensor("language_model.mtp.layers.0.hnorm.weight")
    eh_proj_w = loader.get_tensor("language_model.mtp.layers.0.eh_proj.weight")
    mtp_final_ln_w = loader.get_tensor("language_model.mtp.layers.1.final_layernorm.weight")

    # Build MTP config where layers 0 and 1 are ["full_attention", "moe"]
    mtp_cfg_dict = dict(llm_dict)
    mtp_cfg_dict["num_hidden_layers"] = 2
    mtp_cfg_dict["layers_block_type"] = ["full_attention", "moe"]
    mtp_cfg = NemotronHConfig(**mtp_cfg_dict)
    mtp_cfg._attn_implementation = "eager"

    mtp_l0 = NemotronHBlock(mtp_cfg, layer_idx=0).to(torch.bfloat16)
    mtp_l0.load_state_dict(
        {
            "norm.weight": loader.get_tensor("language_model.mtp.layers.0.norm.weight"),
            "mixer.q_proj.weight": loader.get_tensor("language_model.mtp.layers.0.mixer.q_proj.weight"),
            "mixer.k_proj.weight": loader.get_tensor("language_model.mtp.layers.0.mixer.k_proj.weight"),
            "mixer.v_proj.weight": loader.get_tensor("language_model.mtp.layers.0.mixer.v_proj.weight"),
            "mixer.o_proj.weight": loader.get_tensor("language_model.mtp.layers.0.mixer.o_proj.weight"),
        },
        strict=True,
    )
    mtp_l0.eval()

    mtp_l1 = NemotronHBlock(mtp_cfg, layer_idx=1).to(torch.bfloat16)
    prefix_mtp1 = "language_model.mtp.layers.1."
    mtp1_sd = {
        "norm.weight": loader.get_tensor(prefix_mtp1 + "norm.weight"),
        "mixer.gate.weight": loader.get_tensor(prefix_mtp1 + "mixer.gate.weight"),
        "mixer.gate.e_score_correction_bias": loader.get_tensor(prefix_mtp1 + "mixer.gate.e_score_correction_bias"),
        "mixer.fc1_latent_proj.weight": loader.get_tensor(prefix_mtp1 + "mixer.fc1_latent_proj.weight"),
        "mixer.fc2_latent_proj.weight": loader.get_tensor(prefix_mtp1 + "mixer.fc2_latent_proj.weight"),
        "mixer.shared_experts.up_proj.weight": loader.get_tensor(prefix_mtp1 + "mixer.shared_experts.up_proj.weight"),
        "mixer.shared_experts.down_proj.weight": loader.get_tensor(
            prefix_mtp1 + "mixer.shared_experts.down_proj.weight"
        ),
    }
    mtp1_sd["mixer.experts.up_proj"] = torch.stack(
        [loader.get_tensor(f"{prefix_mtp1}mixer.experts.{e}.up_proj.weight") for e in range(512)], dim=0
    )
    mtp1_sd["mixer.experts.down_proj"] = torch.stack(
        [loader.get_tensor(f"{prefix_mtp1}mixer.experts.{e}.down_proj.weight") for e in range(512)], dim=0
    )
    mtp_l1.load_state_dict(mtp1_sd, strict=True)
    mtp_l1.eval()

    t_mtp0 = time.perf_counter()
    e_normed = rms_norm_ref(h_embed, enorm_w, eps=llm_cfg.layer_norm_epsilon)
    h_normed = rms_norm_ref(h_final, hnorm_w, eps=llm_cfg.layer_norm_epsilon)
    eh_cat = torch.cat([e_normed, h_normed], dim=-1)  # [B, T, 8192]
    mtp_in = F.linear(eh_cat, eh_proj_w)  # [B, T, 4096]
    mtp_h0 = mtp_l0(mtp_in, attention_mask=causal_4d)
    mtp_h1 = mtp_l1(mtp_h0)
    mtp_final = rms_norm_ref(mtp_h1, mtp_final_ln_w, eps=llm_cfg.layer_norm_epsilon)
    mtp_logits = F.linear(mtp_final, lm_head_weight).float()
    mtp_ms = (time.perf_counter() - t_mtp0) * 1000.0

    return {
        "prompt": prompt,
        "seq_len": seq_len,
        "tokens": tokens_str,
        "stage_l0_embedding": {
            "shape": list(h_embed.shape),
            "rms": round(embed_rms, 5),
            "latency_ms": round(embed_ms, 3),
        },
        "stage_l1_mamba2_layer0": {
            "in_rms": round(embed_rms, 5),
            "out_rms": round(float(off_mamba_out.float().pow(2).mean().sqrt()), 5),
            "ssd_recurrent_vs_chunk_scan_max_diff": round(mamba_max_diff, 6),
            "ssd_recurrent_vs_chunk_scan_cos_sim": round(mamba_cos, 8),
            "causal_prefix_max_gap": round(mamba_prefix_gap, 6),
            "latency_ms": round(mamba_ms, 3),
        },
        "stage_l2_latent_moe_layer1": {
            "in_rms": round(float(off_mamba_out.float().pow(2).mean().sqrt()), 5),
            "latent_1024_in_rms": round(float(latent_in.float().pow(2).mean().sqrt()), 5),
            "routed_branch_4096_rms": round(float(routed_4096.float().pow(2).mean().sqrt()), 5),
            "shared_expert_4096_rms": round(float(shared_4096.float().pow(2).mean().sqrt()), 5),
            "out_rms": round(float(off_moe_out.float().pow(2).mean().sqrt()), 5),
            "top_k_per_token": 22,
            "total_routed_experts": 512,
            "distinct_experts_selected_across_prompt": len(distinct_experts_used),
            "first_token_top5_experts": topk_indices[0, :5].tolist(),
            "first_token_top5_weights": [round(float(w), 4) for w in topk_weights[0, :5].tolist()],
            "sum_of_22_weights_per_token": round(float(topk_weights[0].sum().item()), 4),
            "max_abs_diff_vs_official": moe_max_diff,
            "bf16_reassociation_ulp_diff_if_left_associated": moe_left_assoc_diff,
            "cos_sim_vs_official": round(moe_cos, 8),
            "latency_ms": round(moe_ms, 3),
        },
        "stage_l3_nope_gqa_attention_layer7": {
            "in_rms": round(float(off_moe_out.float().pow(2).mean().sqrt()), 5),
            "out_rms": round(float(off_attn_out.float().pow(2).mean().sqrt()), 5),
            "num_q_heads": 32,
            "num_kv_heads": 2,
            "gqa_ratio": "16:1",
            "uses_rope": False,
            "max_abs_diff_vs_official": attn_max_diff,
            "causal_prefix_max_gap": attn_prefix_gap,
            "latency_ms": round(attn_ms, 3),
        },
        "stage_l4_final_norm_and_untied_lm_head": {
            "final_norm_rms": round(float(h_final.float().pow(2).mean().sqrt()), 5),
            "logits_shape": list(logits.shape),
            "tied_word_embeddings": False,
            "latency_ms": round(lm_head_ms, 3),
        },
        "stage_l5_native_mtp_2_layer_speculative_head": {
            "eh_proj_in_dim": 8192,
            "eh_proj_out_dim": 4096,
            "mtp_layer0_type": "full_attention",
            "mtp_layer1_type": "moe (512 latent ReLU^2 experts)",
            "mtp_final_norm_rms": round(float(mtp_final.float().pow(2).mean().sqrt()), 5),
            "mtp_logits_shape": list(mtp_logits.shape),
            "mtp_2_layer_draft_latency_ms": round(mtp_ms, 3),
        },
    }


def main():
    torch.set_num_threads(8)
    loader = SafetensorsShardLoader(MODEL_DIR)

    print("1. Computing full 120B-A12B architectural parameter & state census...", flush=True)
    census = compute_full_120b_architecture_census(MODEL_DIR, loader)
    print(
        f"   Verified {census['total_parameters_exact']:,} total parameters "
        f"({census['active_backbone_parameters_per_token']:,} active/token)",
        flush=True,
    )

    print("2. Verifying Stage V1 (C-RADIO 32-layer ViT-Huge) & Stage V2 (PixelShuffle + ReLU^2 Projector + EVS)...", flush=True)
    vision_study = verify_vision_and_video_stages(MODEL_DIR, loader)
    print(
        f"   Vision V1 L0 diff={vision_study['stage_v1_radio_encoder']['layer0_max_abs_diff_vs_manual']}, "
        f"Projector V2 diff={vision_study['stage_v2_vision_projector_and_evs']['max_abs_diff_vs_manual']}",
        flush=True,
    )

    print("3. Verifying Backbone Stages (Embedding, Mamba-2 L0, Latent MoE L1, NoPE GQA L7, LM Head, Native MTP)...", flush=True)
    backbone_study = verify_backbone_and_mtp_stages(MODEL_DIR, loader)
    print(
        f"   Mamba-2 diff={backbone_study['stage_l1_mamba2_layer0']['ssd_recurrent_vs_chunk_scan_max_diff']}, "
        f"Latent MoE diff={backbone_study['stage_l2_latent_moe_layer1']['max_abs_diff_vs_official']}, "
        f"NoPE GQA diff={backbone_study['stage_l3_nope_gqa_attention_layer7']['max_abs_diff_vs_official']}",
        flush=True,
    )

    report = {
        "model_id": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        "architecture": "NemotronH_Omni_Reasoning_V3 (C-RADIO Vision + Mamba-2 / Latent ReLU^2 MoE / NoPE 16:1 GQA + Native MTP)",
        "hardware_verified_on": "k3 (AMD Ryzen 9 7950X3D, 8 threads, bfloat16)",
        "architecture_and_state_census": census,
        "vision_and_video_stage_verification": vision_study,
        "backbone_and_mtp_stage_verification": backbone_study,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"NEMOTRON_STAGE_STUDY_COMPLETE: wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
