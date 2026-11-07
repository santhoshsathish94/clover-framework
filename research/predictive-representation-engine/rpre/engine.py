"""Reusable Predictive Representation Engine (`rpre/engine.py`).

Unifies:
- `VisualPatchAdapter` (I-JEPA / MJEPA 2D image patch tokenizer + 2D sincos pos embed + modality token)
- `AudioSpectrogramAdapter` (Audio-JEPA / MJEPA 2D time-frequency spectrogram tokenizer + 2D sincos pos embed + modality token)
- `StateVectorAdapter` (EB-JEPA continuous state/sensor vector tokenizer)
- `SharedHierarchicalEncoder` (MJEPA single shared Transformer trunk with I-JEPA context-only sparse token dropping
  and 4-level hierarchical feature extraction)
- `LatentTokenPredictor` (I-JEPA / Audio-JEPA / MJEPA narrow bottleneck Transformer predictor with positional mask tokens)
- `CrossModalPredictor` (MJEPA `A2V2_MLP_Predictor` with mean or attentive pooling for cross-modal latent alignment)
- `ActionConditionedPredictor` (EB-JEPA action-conditioned latent world model supporting parallel and autoregressive unrolls)
- `OutcomeResidualAdapter` (Clover Outcome -> Growth closed-loop residual dynamics learner with experience buffer and
  non-regression validation gate)
- `ReusablePredictiveEngine` (The unified engine combining all components)
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import (
    CenteredNormMSELoss,
    InverseDynamicsMLP,
    MJEPA9PathLoss,
    VCLoss,
    VC_IDM_Sim_Regularizer,
)
from .masks import apply_masks, repeat_interleave_batch


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000.0**omega)
    pos = pos.reshape(-1).astype(np.float32)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """2D sine-cosine positional embedding supporting both square (I-JEPA) and rectangular (Audio-JEPA/MJEPA) grids."""
    assert embed_dim % 2 == 0
    gh = np.arange(grid_h, dtype=np.float32)
    gw = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(gw, gh)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_h, grid_w)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return torch.from_numpy(emb).float().unsqueeze(0)  # [1, grid_h * grid_w, embed_dim]


def normalize_and_concat_hierarchical(x: torch.Tensor, base_dim: int, num_levels: int) -> torch.Tensor:
    """MJEPA (`app/mjepa/utils.py:normalize_and_concat`) per-hierarchy-level LayerNorm before L1/SmoothL1 loss."""
    if num_levels <= 1:
        return F.layer_norm(x, (x.size(-1),))
    chunks = []
    for i in range(num_levels):
        sub = x[..., i * base_dim : (i + 1) * base_dim]
        chunks.append(F.layer_norm(sub, (base_dim,)))
    return torch.cat(chunks, dim=-1)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block matching `ijepa/src/models/vision_transformer.py:Block`."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm1(x)
        attn_out, _ = self.attn(xn, xn, xn, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisualPatchAdapter(nn.Module):
    """Visual modality patch tokenizer (`[B, C, H, W] -> [B, N_v, D]`)."""

    def __init__(self, img_size: tuple[int, int] = (32, 32), patch_size: int = 4, in_chans: int = 3, embed_dim: int = 64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size[0] // patch_size
        self.grid_w = img_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer("pos_embed", get_2d_sincos_pos_embed(embed_dim, self.grid_h, self.grid_w))
        self.mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mod_embed, std=1e-4)

    def forward(self, x: torch.Tensor, masks: list[torch.Tensor] | None = None) -> torch.Tensor:
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed + self.mod_embed
        if masks is not None:
            tokens = apply_masks(tokens, masks)
        return tokens


class AudioSpectrogramAdapter(nn.Module):
    """Audio spectrogram modality tokenizer (`[B, 1, T, F] -> [B, N_a, D]`)."""

    def __init__(
        self,
        spec_size: tuple[int, int] = (32, 16),
        patch_size: tuple[int, int] = (4, 4),
        in_chans: int = 1,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.spec_size = spec_size
        self.patch_size = patch_size
        self.grid_t = spec_size[0] // patch_size[0]
        self.grid_f = spec_size[1] // patch_size[1]
        self.num_patches = self.grid_t * self.grid_f
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer("pos_embed", get_2d_sincos_pos_embed(embed_dim, self.grid_t, self.grid_f))
        self.mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mod_embed, std=1e-4)

    def forward(self, x: torch.Tensor, masks: list[torch.Tensor] | None = None) -> torch.Tensor:
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed + self.mod_embed
        if masks is not None:
            tokens = apply_masks(tokens, masks)
        return tokens


class StateVectorAdapter(nn.Module):
    """Continuous state/observation vector tokenizer (`[B, state_dim] -> [B, 1, D]`)."""

    def __init__(self, state_dim: int, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mod_embed, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.proj(x) + self.mod_embed


class SharedHierarchicalEncoder(nn.Module):
    """MJEPA (`src/models/mjepa_encoder.py`) single shared Transformer trunk across all modalities.

    Supports:
    - Context-only sparse token encoding (I-JEPA / Audio-JEPA)
    - Single-modality (`visual` or `audio` or `state`) and joint multimodal (`[audio_tokens, visual_tokens]`) forward passes
    - Multi-level hierarchical output extraction (`hierarchical_layers`, e.g., `[0, 1, 2, 3]` for depth 4)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        hierarchical_levels: int = 4,
        visual_adapter: VisualPatchAdapter | None = None,
        audio_adapter: AudioSpectrogramAdapter | None = None,
        state_adapter: StateVectorAdapter | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.hierarchical_levels = hierarchical_levels
        self.visual_adapter = visual_adapter
        self.audio_adapter = audio_adapter
        self.state_adapter = state_adapter

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        if hierarchical_levels == 4 and depth >= 4:
            step = depth // 4
            self.hier_indices = [step - 1, 2 * step - 1, 3 * step - 1, depth - 1]
        else:
            self.hier_indices = [depth - 1]
        self.norms_block = nn.ModuleList([nn.LayerNorm(embed_dim, eps=1e-6) for _ in self.hier_indices])

    @property
    def out_dim_hierarchical(self) -> int:
        return self.embed_dim * len(self.hier_indices)

    def forward_tokens(self, x: torch.Tensor, return_hierarchical: bool = False) -> torch.Tensor:
        hier_outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.hier_indices:
                idx = self.hier_indices.index(i)
                hier_outs.append(self.norms_block[idx](x))
        if return_hierarchical:
            return torch.cat(hier_outs, dim=-1)
        return hier_outs[-1]

    def forward(
        self,
        visual_x: torch.Tensor | None = None,
        visual_masks: list[torch.Tensor] | None = None,
        audio_x: torch.Tensor | None = None,
        audio_masks: list[torch.Tensor] | None = None,
        state_x: torch.Tensor | None = None,
        return_hierarchical: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        parts = []
        lengths: dict[str, tuple[int, int]] = {}
        cur = 0
        if audio_x is not None and self.audio_adapter is not None:
            at = self.audio_adapter(audio_x, audio_masks)
            parts.append(at)
            lengths["audio"] = (cur, cur + at.size(1))
            cur += at.size(1)
        if visual_x is not None and self.visual_adapter is not None:
            vt = self.visual_adapter(visual_x, visual_masks)
            parts.append(vt)
            lengths["visual"] = (cur, cur + vt.size(1))
            cur += vt.size(1)
        if state_x is not None and self.state_adapter is not None:
            st = self.state_adapter(state_x)
            parts.append(st)
            lengths["state"] = (cur, cur + st.size(1))
            cur += st.size(1)
        if not parts:
            raise ValueError("At least one modality input (visual_x, audio_x, or state_x) is required.")

        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        encoded = self.forward_tokens(x, return_hierarchical=return_hierarchical)

        out: dict[str, torch.Tensor | None] = {"audio": None, "visual": None, "state": None, "joint": encoded}
        for k, (s_i, e_i) in lengths.items():
            out[k] = encoded[:, s_i:e_i, :]
        return out


class LatentTokenPredictor(nn.Module):
    """I-JEPA / Audio-JEPA / MJEPA narrow-bottleneck Transformer Predictor (`src/models/mjepa_predictor.py`).

    Projects context representations (`embed_dim * num_hier_layers`) into a narrower predictor bottleneck
    (`predictor_embed_dim`), appends learnable `[MASK]` tokens at target positions (`masks_y`), runs
    `depth` Transformer blocks, and projects predicted target tokens back to `embed_dim * num_hier_layers`.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        predictor_embed_dim: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        num_hier_layers: int = 1,
        visual_grid: tuple[int, int] = (8, 8),
        audio_grid: tuple[int, int] = (8, 4),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.num_hier_layers = num_hier_layers
        in_out_dim = embed_dim * num_hier_layers

        if num_hier_layers > 1:
            self.predictor_embed = nn.Sequential(
                nn.Linear(in_out_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, predictor_embed_dim),
            )
        else:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.register_buffer(
            "visual_pos_embed",
            get_2d_sincos_pos_embed(predictor_embed_dim, visual_grid[0], visual_grid[1]),
        )
        self.register_buffer(
            "audio_pos_embed",
            get_2d_sincos_pos_embed(predictor_embed_dim, audio_grid[0], audio_grid[1]),
        )
        self.visual_mod_embed = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.audio_mod_embed = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.normal_(self.visual_mod_embed, std=1e-4)
        nn.init.normal_(self.audio_mod_embed, std=1e-4)

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_embed_dim, in_out_dim)

    def _prepare_modality_tokens(
        self,
        x_ctxt: torch.Tensor,
        masks_x: list[torch.Tensor],
        masks_y: list[torch.Tensor],
        pos_embed_buf: torch.Tensor,
        mod_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]
        B = len(x_ctxt) // len(masks_x)
        x = self.predictor_embed(x_ctxt)
        x_pos = pos_embed_buf.repeat(B, 1, 1)
        x = x + apply_masks(x_pos, masks_x)
        _, n_ctxt, _ = x.shape

        pos_y = pos_embed_buf.repeat(B, 1, 1)
        pos_y = apply_masks(pos_y, masks_y)
        pos_y = repeat_interleave_batch(pos_y, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos_y.size(0), pos_y.size(1), 1) + pos_y
        n_pred = pred_tokens.size(1)

        x = x.repeat(len(masks_y), 1, 1)
        combined = torch.cat([x, pred_tokens], dim=1) + mod_embed
        return combined, n_ctxt, n_pred

    def forward(
        self,
        visual_ctxt: torch.Tensor | None = None,
        visual_masks_x: list[torch.Tensor] | None = None,
        visual_masks_y: list[torch.Tensor] | None = None,
        audio_ctxt: torch.Tensor | None = None,
        audio_masks_x: list[torch.Tensor] | None = None,
        audio_masks_y: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        parts = []
        a_info = None
        v_info = None

        if audio_ctxt is not None and audio_masks_x is not None and audio_masks_y is not None:
            a_comb, a_nc, a_np = self._prepare_modality_tokens(
                audio_ctxt, audio_masks_x, audio_masks_y, self.audio_pos_embed, self.audio_mod_embed
            )
            parts.append(a_comb)
            a_info = (a_comb.size(1), a_nc, a_np)

        if visual_ctxt is not None and visual_masks_x is not None and visual_masks_y is not None:
            v_comb, v_nc, v_np = self._prepare_modality_tokens(
                visual_ctxt, visual_masks_x, visual_masks_y, self.visual_pos_embed, self.visual_mod_embed
            )
            parts.append(v_comb)
            v_info = (v_comb.size(1), v_nc, v_np)

        if not parts:
            raise ValueError("Predictor requires at least one modality context and mask pair.")

        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        for blk in self.blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        audio_pred = None
        visual_pred = None
        offset = 0
        if a_info is not None:
            a_tot, a_nc, _ = a_info
            a_slice = x[:, offset : offset + a_tot, :]
            audio_pred = self.predictor_proj(a_slice[:, a_nc:, :])
            offset += a_tot
        if v_info is not None:
            v_tot, v_nc, _ = v_info
            v_slice = x[:, offset : offset + v_tot, :]
            visual_pred = self.predictor_proj(v_slice[:, v_nc:, :])

        return {"audio_pred": audio_pred, "visual_pred": visual_pred}


class CrossModalPredictor(nn.Module):
    """MJEPA (`src/audio_models/a2v_predictors.py:A2V2_MLP_Predictor`) Cross-Modal Pooled Predictor.

    Pools context tokens of source modality (`mean` or `attentive`) and predicts the pooled target
    representation of the destination modality (`a2v`, `v2a`, `av2a`, `av2v`, `a2av`, `v2av`).
    """

    def __init__(self, in_dim: int = 64, out_dim: int = 64, mlp_ratio: int = 2, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        hidden = in_dim * mlp_ratio
        if pooling == "attentive":
            self.query = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)
            self.attn = nn.MultiheadAttention(in_dim, num_heads=max(1, in_dim // 16), batch_first=True)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x
        if self.pooling == "attentive":
            q = self.query.expand(x.size(0), -1, -1)
            out, _ = self.attn(q, x, x, need_weights=False)
            return out.squeeze(1)
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(x)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        return self.norm(self.fc3(h))


class ActionConditionedPredictor(nn.Module):
    """EB-JEPA (`eb_jepa/jepa.py` & `eb_jepa/architectures.py:RNNPredictor`) Action-Conditioned Latent Transition Model.

    Supports both:
    - `unroll_mode="parallel"`: Teacher-forced single/multi-step prediction across a trajectory `[B, D, T]`
    - `unroll_mode="autoregressive"`: Step-by-step open-loop rollout `[B, D, T_ctxt + nsteps]` for MPC planning
    """

    def __init__(self, state_dim: int = 64, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.is_rnn = True
        self.context_length = 1

        self.state_in = nn.Linear(state_dim, hidden_dim)
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim, eps=1e-6),
        )

    def forward(
        self,
        states_bdt: torch.Tensor,
        actions_bat: torch.Tensor | None = None,
        residual_adapter: OutcomeResidualAdapter | None = None,
        context_cue: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict next latent states `[B, D, T]` from `states_bdt` (`[B, D, T]`) and `actions_bat` (`[B, A, T]`)."""
        s_btd = states_bdt.transpose(1, 2)  # [B, T, D]
        h_in = F.gelu(self.state_in(s_btd))
        if actions_bat is not None:
            a_bta = actions_bat.transpose(1, 2)[:, : s_btd.size(1), :]
            h_in = h_in + F.gelu(self.action_in(a_bta))
        h_out, _ = self.gru(h_in)
        pred_btd = self.out_proj(h_out)
        if residual_adapter is not None and actions_bat is not None:
            a_bta = actions_bat.transpose(1, 2)[:, : s_btd.size(1), :]
            delta_btd = residual_adapter(s_btd, a_bta, context_cue=context_cue)
            if float(delta_btd.detach().abs().max().item()) > 1e-7:
                pred_btd = F.layer_norm(pred_btd + delta_btd, (self.state_dim,))
        return pred_btd.transpose(1, 2)  # [B, D, T]


class OutcomeResidualAdapter(nn.Module):
    """Clover Outcome -> Growth Closed-Loop Residual Predictor.

    When the engine acts in reality and observes an outcome $z_{t+1}^{\text{obs}}$ that differs from the
    predicted latent state $\hat{z}_{t+1}$, `OutcomeResidualAdapter` learns the structured residual
    $\Delta z_{t+1} = f_\psi(z_t, a_t, c_{\text{multimodal}})$ subject to two Clover integrity gates:
    1. **Experience Replay + Variance-Covariance Preservation**: Prevents single-sample overfitting or collapse.
    2. **Held-Out Non-Regression Gate (`AGENTS.md` §4 & §16)**: Snapshots weights before each cycle update and
       rolls back if prediction error on previously verified baseline transitions regresses.
    """

    def __init__(self, state_dim: int = 64, action_dim: int = 2, cue_dim: int = 0, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cue_dim = cue_dim
        in_dim = state_dim + action_dim + cue_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Zero-init final layer so the adapter starts as an exact identity (zero residual)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        self.baseline_anchor: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        self.baseline_cue_proto: torch.Tensor | None = None
        self.cycle_history: list[dict[str, Any]] = []

    def forward(
        self,
        state_btd: torch.Tensor,
        action_bta: torch.Tensor,
        context_cue: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_shape = state_btd.shape
        if state_btd.ndim == 3:
            b, t, d = state_btd.shape
            s_flat = state_btd.reshape(b * t, d)
            a_flat = action_bta.reshape(b * t, -1)
        else:
            s_flat = state_btd
            a_flat = action_bta
        parts = [s_flat, a_flat]
        anom_gate = 1.0
        if self.cue_dim > 0:
            if context_cue is None:
                c_flat = torch.zeros((s_flat.size(0), self.cue_dim), device=s_flat.device, dtype=s_flat.dtype)
            else:
                if context_cue.ndim == 2 and state_btd.ndim == 3:
                    c_flat = context_cue.unsqueeze(1).expand(orig_shape[0], orig_shape[1], -1).reshape(s_flat.size(0), -1)
                else:
                    c_flat = context_cue.reshape(s_flat.size(0), -1)
                if self.baseline_cue_proto is not None:
                    proto = self.baseline_cue_proto.to(device=c_flat.device, dtype=c_flat.dtype)
                    dist_sq = ((c_flat - proto) ** 2).mean(dim=-1, keepdim=True)
                    anom_gate = torch.clamp((dist_sq - 0.35) / 0.15, 0.0, 1.0)
            parts.append(c_flat)
        inp = torch.cat(parts, dim=-1)
        out = anom_gate * torch.tanh(self.gate) * self.net(inp)
        return out.reshape(orig_shape)

    def record_outcome(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_tp1_obs: torch.Tensor,
        c_t: torch.Tensor | None = None,
        is_baseline: bool = False,
    ) -> None:
        item = (
            z_t.detach().clone(),
            a_t.detach().clone(),
            z_tp1_obs.detach().clone(),
            c_t.detach().clone() if c_t is not None else None,
        )
        if is_baseline:
            self.baseline_anchor.append(item)
            if len(self.baseline_anchor) > 512:
                self.baseline_anchor.pop(0)
            cues = [x[3] for x in self.baseline_anchor if x[3] is not None and x[3].abs().sum() > 0]
            if cues:
                self.baseline_cue_proto = torch.cat(cues, dim=0).mean(dim=0, keepdim=True)
        else:
            self.buffer.append(item)
            if len(self.buffer) > 1024:
                self.buffer.pop(0)

    def update_from_outcomes(
        self,
        base_predictor: ActionConditionedPredictor,
        steps: int = 40,
        lr: float = 5e-3,
        max_baseline_regression: float = 0.02,
    ) -> dict[str, Any]:
        """Update the residual adapter from observed `(z_t, a_t, z_{t+1}^{obs})` transitions with non-regression gating."""
        if not self.buffer:
            return {"updated": False, "reason": "empty_buffer"}

        snapshot = copy.deepcopy(self.state_dict())
        z_t = torch.cat([x[0] for x in self.buffer], dim=0)
        a_t = torch.cat([x[1] for x in self.buffer], dim=0)
        z_obs = torch.cat([x[2] for x in self.buffer], dim=0)
        c_t = (
            torch.cat([x[3] for x in self.buffer if x[3] is not None], dim=0)
            if (self.cue_dim > 0 and all(x[3] is not None for x in self.buffer))
            else None
        )

        with torch.no_grad():
            base_pred = base_predictor(
                z_t.unsqueeze(-1), a_t.unsqueeze(-1), residual_adapter=None
            ).squeeze(-1)
            before_pred = base_predictor(
                z_t.unsqueeze(-1), a_t.unsqueeze(-1), residual_adapter=self, context_cue=c_t
            ).squeeze(-1)
            mse_before = float(F.mse_loss(before_pred, z_obs).item())

            base_anchor_mse_before = 0.0
            if self.baseline_anchor:
                bz_t = torch.cat([x[0] for x in self.baseline_anchor], dim=0)
                ba_t = torch.cat([x[1] for x in self.baseline_anchor], dim=0)
                bz_obs = torch.cat([x[2] for x in self.baseline_anchor], dim=0)
                bc_t = (
                    torch.cat([x[3] for x in self.baseline_anchor if x[3] is not None], dim=0)
                    if (self.cue_dim > 0 and all(x[3] is not None for x in self.baseline_anchor))
                    else None
                )
                bp_before = base_predictor(
                    bz_t.unsqueeze(-1), ba_t.unsqueeze(-1), residual_adapter=self, context_cue=bc_t
                ).squeeze(-1)
                base_anchor_mse_before = float(F.mse_loss(bp_before, bz_obs).item())

        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            delta = self.forward(z_t, a_t, context_cue=c_t)
            pred = F.layer_norm(base_pred + delta, (self.state_dim,))
            loss = F.mse_loss(pred, z_obs)
            if self.baseline_anchor:
                delta_b = self.forward(bz_t, ba_t, context_cue=bc_t)
                loss = loss + 4.0 * delta_b.pow(2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            after_pred = base_predictor(
                z_t.unsqueeze(-1), a_t.unsqueeze(-1), residual_adapter=self, context_cue=c_t
            ).squeeze(-1)
            mse_after = float(F.mse_loss(after_pred, z_obs).item())
            base_anchor_mse_after = 0.0
            if self.baseline_anchor:
                bp_after = base_predictor(
                    bz_t.unsqueeze(-1), ba_t.unsqueeze(-1), residual_adapter=self, context_cue=bc_t
                ).squeeze(-1)
                base_anchor_mse_after = float(F.mse_loss(bp_after, bz_obs).item())

        regressed = self.baseline_anchor and (base_anchor_mse_after - base_anchor_mse_before > max_baseline_regression)
        if regressed or mse_after > mse_before:
            self.load_state_dict(snapshot)
            rec = {
                "updated": False,
                "rolled_back": True,
                "mse_before": round(mse_before, 6),
                "mse_after": round(mse_after, 6),
                "baseline_mse_before": round(base_anchor_mse_before, 6),
                "baseline_mse_after": round(base_anchor_mse_after, 6),
            }
            self.cycle_history.append(rec)
            return rec

        rec = {
            "updated": True,
            "rolled_back": False,
            "samples": int(z_t.size(0)),
            "mse_before": round(mse_before, 6),
            "mse_after": round(mse_after, 6),
            "baseline_mse_before": round(base_anchor_mse_before, 6),
            "baseline_mse_after": round(base_anchor_mse_after, 6),
        }
        self.cycle_history.append(rec)
        return rec


class ReusablePredictiveEngine(nn.Module):
    """Unified Reusable Predictive Representation Engine (`RPRE`).

    Combines:
    1. Shared multi-level encoder `encoder` + EMA `target_encoder` (`I-JEPA` + `Audio-JEPA` + `MJEPA`)
    2. Intra-modal masked token predictor `token_predictor`
    3. 6 Cross-modal relationship predictors (`a2v`, `v2a`, `av2a`, `av2v`, `a2av`, `v2av`) (`MJEPA`)
    4. Action-conditioned trajectory predictor `transition_predictor` + `VC_IDM_Sim_Regularizer` (`EB-JEPA`)
    5. Closed-loop `outcome_adapter` (`Clover Outcome -> Growth`)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        predictor_embed_dim: int = 32,
        depth: int = 4,
        pred_depth: int = 2,
        num_heads: int = 4,
        hierarchical_levels: int = 4,
        visual_img_size: tuple[int, int] = (32, 32),
        visual_patch_size: int = 4,
        visual_in_chans: int = 3,
        audio_spec_size: tuple[int, int] = (32, 16),
        audio_patch_size: tuple[int, int] = (4, 4),
        state_dim: int = 4,
        action_dim: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hierarchical_levels = hierarchical_levels

        v_adapter = VisualPatchAdapter(visual_img_size, visual_patch_size, visual_in_chans, embed_dim)
        a_adapter = AudioSpectrogramAdapter(audio_spec_size, audio_patch_size, 1, embed_dim)
        s_adapter = StateVectorAdapter(state_dim, embed_dim)

        self.encoder = SharedHierarchicalEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            hierarchical_levels=hierarchical_levels,
            visual_adapter=v_adapter,
            audio_adapter=a_adapter,
            state_adapter=s_adapter,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        num_h = len(self.encoder.hier_indices)
        self.token_predictor = LatentTokenPredictor(
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=pred_depth,
            num_heads=num_heads,
            num_hier_layers=num_h,
            visual_grid=(v_adapter.grid_h, v_adapter.grid_w),
            audio_grid=(a_adapter.grid_t, a_adapter.grid_f),
        )

        # MJEPA 6 Cross-Modal Predictors
        self.a2v_predictor = CrossModalPredictor(embed_dim, embed_dim)
        self.v2a_predictor = CrossModalPredictor(embed_dim, embed_dim)
        self.av2a_predictor = CrossModalPredictor(embed_dim, embed_dim)
        self.av2v_predictor = CrossModalPredictor(embed_dim, embed_dim)
        self.a2av_predictor = CrossModalPredictor(embed_dim, embed_dim)
        self.v2av_predictor = CrossModalPredictor(embed_dim, embed_dim)

        # EB-JEPA Action-Conditioned World Model & Anti-Collapse Regularizer
        self.transition_predictor = ActionConditionedPredictor(
            state_dim=embed_dim, action_dim=action_dim, hidden_dim=embed_dim * 2
        )
        self.idm = InverseDynamicsMLP(state_dim=embed_dim, hidden_dim=embed_dim * 2, action_dim=action_dim)
        self.regularizer = VC_IDM_Sim_Regularizer(
            cov_coeff=8.0, std_coeff=16.0, sim_coeff_t=12.0, idm_coeff=1.0, idm=self.idm
        )
        self.outcome_adapter = OutcomeResidualAdapter(
            state_dim=embed_dim, action_dim=action_dim, cue_dim=embed_dim, hidden_dim=embed_dim
        )
        self.mjepa_loss = MJEPA9PathLoss(loss_exp=1.0)
        self.audio_norm_mse = CenteredNormMSELoss(standardize_target=True)

    @torch.no_grad()
    def update_target_ema(self, momentum: float = 0.99) -> None:
        for pq, pk in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            pk.data.mul_(momentum).add_((1.0 - momentum) * pq.detach().data)

    def multimodal_jepa_step(
        self,
        visual_x: torch.Tensor,
        visual_masks_enc: list[torch.Tensor],
        visual_masks_pred: list[torch.Tensor],
        audio_x: torch.Tensor,
        audio_masks_enc: list[torch.Tensor],
        audio_masks_pred: list[torch.Tensor],
        enable_cross_modal: bool = True,
        enable_av_joint: bool = True,
        use_hierarchical: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute MJEPA's 3 intra-modal (`v2v`, `a2a`, `av2av`) and 6 cross-modal (`a2v`, `v2a`, `av2a`, `av2v`, `a2av`, `v2av`) losses."""
        num_h = len(self.encoder.hier_indices) if use_hierarchical else 1
        d = self.embed_dim

        with torch.no_grad():
            vt_out = self.target_encoder(visual_x=visual_x, return_hierarchical=use_hierarchical)["visual"]
            at_out = self.target_encoder(audio_x=audio_x, return_hierarchical=use_hierarchical)["audio"]
            vh_norm = normalize_and_concat_hierarchical(vt_out, d, num_h)
            ah_norm = normalize_and_concat_hierarchical(at_out, d, num_h)
            vh_last = vh_norm[..., -d:]
            ah_last = ah_norm[..., -d:]
            vh_masked = apply_masks(vh_norm, visual_masks_pred)
            vh_masked = repeat_interleave_batch(vh_masked, visual_x.size(0), repeat=len(visual_masks_enc))
            ah_masked = apply_masks(ah_norm, audio_masks_pred)
            ah_masked = repeat_interleave_batch(ah_masked, audio_x.size(0), repeat=len(audio_masks_enc))

        # 1. Visual context -> v2v + v2a + v2av
        vz_ctxt = self.encoder(visual_x=visual_x, visual_masks=visual_masks_enc, return_hierarchical=use_hierarchical)["visual"]
        v_pred_out = self.token_predictor(
            visual_ctxt=vz_ctxt, visual_masks_x=visual_masks_enc, visual_masks_y=visual_masks_pred
        )["visual_pred"]
        v2v_pred = normalize_and_concat_hierarchical(v_pred_out, d, num_h)
        v2v_loss = self.mjepa_loss.lp_loss(v2v_pred, vh_masked)

        # 2. Audio context -> a2a + a2v + a2av
        az_ctxt = self.encoder(audio_x=audio_x, audio_masks=audio_masks_enc, return_hierarchical=use_hierarchical)["audio"]
        a_pred_out = self.token_predictor(
            audio_ctxt=az_ctxt, audio_masks_x=audio_masks_enc, audio_masks_y=audio_masks_pred
        )["audio_pred"]
        a2a_pred = normalize_and_concat_hierarchical(a_pred_out, d, num_h)
        a2a_loss = self.mjepa_loss.lp_loss(a2a_pred, ah_masked)

        zero = torch.tensor(0.0, device=visual_x.device, dtype=visual_x.dtype)
        a2v_loss = v2a_loss = av2av_loss = av2a_loss = av2v_loss = a2av_loss = v2av_loss = zero

        if enable_cross_modal:
            vz_last = vz_ctxt[..., -d:]
            az_last = az_ctxt[..., -d:]
            v_pool_target = vh_last.mean(dim=1)
            a_pool_target = ah_last.mean(dim=1)
            a2v_loss = self.mjepa_loss.lp_loss(self.a2v_predictor(az_last), v_pool_target)
            v2a_loss = self.mjepa_loss.lp_loss(self.v2a_predictor(vz_last), a_pool_target)

        if enable_av_joint:
            with torch.no_grad():
                av_t = self.target_encoder(audio_x=audio_x, visual_x=visual_x, return_hierarchical=use_hierarchical)
                av_ah = normalize_and_concat_hierarchical(av_t["audio"], d, num_h)
                av_vh = normalize_and_concat_hierarchical(av_t["visual"], d, num_h)
                av_ah_masked = repeat_interleave_batch(
                    apply_masks(av_ah, audio_masks_pred), audio_x.size(0), repeat=len(audio_masks_enc)
                )
                av_vh_masked = repeat_interleave_batch(
                    apply_masks(av_vh, visual_masks_pred), visual_x.size(0), repeat=len(visual_masks_enc)
                )
                av_pool_target = torch.cat([av_ah[..., -d:], av_vh[..., -d:]], dim=1).mean(dim=1)

            av_c = self.encoder(
                audio_x=audio_x,
                audio_masks=audio_masks_enc,
                visual_x=visual_x,
                visual_masks=visual_masks_enc,
                return_hierarchical=use_hierarchical,
            )
            av_preds = self.token_predictor(
                audio_ctxt=av_c["audio"],
                audio_masks_x=audio_masks_enc,
                audio_masks_y=audio_masks_pred,
                visual_ctxt=av_c["visual"],
                visual_masks_x=visual_masks_enc,
                visual_masks_y=visual_masks_pred,
            )
            av_a_pred = normalize_and_concat_hierarchical(av_preds["audio_pred"], d, num_h)
            av_v_pred = normalize_and_concat_hierarchical(av_preds["visual_pred"], d, num_h)
            av2av_loss = self.mjepa_loss.lp_loss(av_a_pred, av_ah_masked) + self.mjepa_loss.lp_loss(
                av_v_pred, av_vh_masked
            )

            if enable_cross_modal:
                av_ctxt_last = torch.cat([av_c["audio"][..., -d:], av_c["visual"][..., -d:]], dim=1)
                av2a_loss = self.mjepa_loss.lp_loss(self.av2a_predictor(av_ctxt_last), ah_last.mean(dim=1))
                av2v_loss = self.mjepa_loss.lp_loss(self.av2v_predictor(av_ctxt_last), vh_last.mean(dim=1))
                a2av_loss = self.mjepa_loss.lp_loss(self.a2av_predictor(az_ctxt[..., -d:]), av_pool_target)
                v2av_loss = self.mjepa_loss.lp_loss(self.v2av_predictor(vz_ctxt[..., -d:]), av_pool_target)

        total_loss = (
            v2v_loss
            + a2a_loss
            + av2av_loss
            + a2v_loss
            + v2a_loss
            + av2a_loss
            + av2v_loss
            + a2av_loss
            + v2av_loss
        )
        return {
            "loss": total_loss,
            "v2v": v2v_loss,
            "a2a": a2a_loss,
            "av2av": av2av_loss,
            "a2v": a2v_loss,
            "v2a": v2a_loss,
            "av2a": av2a_loss,
            "av2v": av2v_loss,
            "a2av": a2av_loss,
            "v2av": v2av_loss,
        }

    def unroll_latent(
        self,
        state_seq_bst: torch.Tensor | None = None,
        init_latent_bd: torch.Tensor | None = None,
        actions_bat: torch.Tensor | None = None,
        nsteps: int = 1,
        unroll_mode: str = "autoregressive",
        compute_loss: bool = False,
        use_outcome_adapter: bool = True,
        context_cue: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple | None]:
        """EB-JEPA (`eb_jepa/jepa.py:unroll`) unified parallel & autoregressive latent unrolling."""
        adapter = self.outcome_adapter if use_outcome_adapter else None

        if state_seq_bst is not None:
            # state_seq_bst: [B, state_dim, T] -> encode each step to [B, D, T]
            b, s_dim, t = state_seq_bst.shape
            s_flat = state_seq_bst.transpose(1, 2).reshape(b * t, s_dim)
            z_flat = self.encoder(state_x=s_flat)["state"].squeeze(1)
            z_bdt = z_flat.reshape(b, t, self.embed_dim).transpose(1, 2).contiguous()
        elif init_latent_bd is not None:
            z_bdt = init_latent_bd.unsqueeze(-1)
        else:
            raise ValueError("Either state_seq_bst or init_latent_bd must be provided.")

        if compute_loss:
            rloss, rloss_unw, rloss_dict = self.regularizer(z_bdt, actions_bat)
            ploss = torch.tensor(0.0, device=z_bdt.device, dtype=z_bdt.dtype)
        else:
            rloss = rloss_unw = rloss_dict = ploss = None

        if unroll_mode == "parallel":
            pred_bdt = z_bdt
            for _ in range(nsteps):
                step_out = self.transition_predictor(
                    pred_bdt, actions_bat, residual_adapter=adapter, context_cue=context_cue
                )[:, :, :-1]
                pred_bdt = torch.cat([z_bdt[:, :, :1], step_out], dim=2)
                if compute_loss:
                    ploss = ploss + F.mse_loss(pred_bdt, z_bdt) / nsteps
        elif unroll_mode == "autoregressive":
            pred_bdt = z_bdt[:, :, :1]
            for i in range(nsteps):
                cur_z = pred_bdt[:, :, -1:]
                cur_a = actions_bat[:, :, i : i + 1] if actions_bat is not None else None
                nxt_z = self.transition_predictor(
                    cur_z, cur_a, residual_adapter=adapter, context_cue=context_cue
                )[:, :, -1:]
                pred_bdt = torch.cat([pred_bdt, nxt_z], dim=2)
                if compute_loss and z_bdt.size(2) > i + 1:
                    ploss = ploss + F.mse_loss(nxt_z, z_bdt[:, :, i + 1 : i + 2]) / nsteps
        else:
            raise ValueError(f"Unknown unroll_mode: {unroll_mode}")

        if compute_loss:
            total = rloss + ploss
            return pred_bdt, (total, rloss, rloss_unw, rloss_dict, ploss)
        return pred_bdt, None
