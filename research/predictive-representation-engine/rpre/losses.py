"""Loss functions and Energy-Based Anti-Collapse Regularizers across I-JEPA, Audio-JEPA, EB-JEPA, and MJEPA (`rpre/losses.py`).

Implements:
1. `CenteredNormMSELoss` (`Audio-JEPA/src/models/components/loss.py`):
   Standardizes target per token (`mean=0, var=1`), L2-normalizes both `pred` and `target`, and computes
   $2 - 2\langle \hat{z}, \hat{h}\rangle$, equivalent to $2(1 - r_{\text{Pearson}})$.
2. `HingeStdLoss`, `CovarianceLoss`, `VCLoss`, `TemporalSimilarityLoss`, `InverseDynamicsLoss`, `VC_IDM_Sim_Regularizer`
   (`eb_jepa/losses.py`).
3. `BCSLoss` (`eb_jepa/losses.py:epps_pulley` & `BCS`): Batched Characteristic Slicing Gaussianity test.
4. `MJEPA9PathLoss` (`MJEPA/app/mjepa/train.py`): Unified $L_p$ (`loss_exp=1.0` L1 or Smooth-L1) loss across the
   3 intra-modal (`v2v`, `a2a`, `av2av`) and 6 cross-modal (`a2v`, `v2a`, `av2a`, `av2v`, `a2av`, `v2av`) paths.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenteredNormMSELoss(nn.Module):
    """Audio-JEPA (`src/models/components/loss.py`) `norm_mse` with feature standardization.

    Note on `Audio-JEPA` (`loss.py:42-46`): the upstream implementation standardizes `target`
    (`(target - mean) / sqrt(var + 1e-6)`) before `F.normalize`, which requires `pred` to also
    be zero-mean (or `center_pred=True`) for `loss(z, z) == 0` on uncentered inputs.
    """

    def __init__(self, standardize_target: bool = True, center_pred: bool = True):
        super().__init__()
        self.standardize_target = standardize_target
        self.center_pred = center_pred

    def forward(self, pred: torch.Tensor, target: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        if self.standardize_target:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        if self.center_pred:
            p_mean = pred.mean(dim=-1, keepdim=True)
            p_var = pred.var(dim=-1, keepdim=True)
            pred = (pred - p_mean) / (p_var + 1e-6).sqrt()
        target_n = F.normalize(target, dim=-1, p=2)
        pred_n = F.normalize(pred, dim=-1, p=2)
        loss = 2.0 - 2.0 * (target_n * pred_n).sum(dim=-1)
        return loss.mean() if reduce else loss


class HingeStdLoss(nn.Module):
    """EB-JEPA (`eb_jepa/losses.py:49`) Hinge Standard Deviation Loss:
    $\frac{1}{D}\sum_{d=1}^D \max(0, \gamma - \sqrt{\text{Var}(x_{\cdot, d}) + 10^{-4}})$."""

    def __init__(self, std_margin: float = 1.0):
        super().__init__()
        self.std_margin = float(std_margin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        x_c = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x_c.var(dim=0) + 1e-4)
        return torch.mean(F.relu(self.std_margin - std))


class CovarianceLoss(nn.Module):
    """EB-JEPA (`eb_jepa/losses.py:81`) Off-Diagonal Covariance Decorrelation Loss."""

    @staticmethod
    def off_diagonal(mat: torch.Tensor) -> torch.Tensor:
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= 1 or x.shape[-1] <= 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        n = x.shape[0]
        x_c = x - x.mean(dim=0, keepdim=True)
        cov = (x_c.T @ x_c) / (n - 1)
        return self.off_diagonal(cov).pow(2).mean()


class VCLoss(nn.Module):
    """EB-JEPA (`eb_jepa/losses.py:29`) Variance-Covariance anti-collapse regularizer."""

    def __init__(self, std_coeff: float = 10.0, cov_coeff: float = 10.0, proj: nn.Module | None = None):
        super().__init__()
        self.std_coeff = float(std_coeff)
        self.cov_coeff = float(cov_coeff)
        self.proj = nn.Identity() if proj is None else proj
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, x: torch.Tensor, actions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if x.ndim == 5:
            # [B, C, T, H, W] -> [B*T*H*W, C]
            flat = x.permute(0, 2, 3, 4, 1).reshape(-1, x.shape[1])
        elif x.ndim == 3:
            # [B, N, D] -> [B*N, D]
            flat = x.reshape(-1, x.shape[-1])
        else:
            flat = x
        fx = self.proj(flat)
        std_loss = self.std_loss_fn(fx)
        cov_loss = self.cov_loss_fn(fx)
        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        unweighted = std_loss + cov_loss
        return loss, unweighted, {"std_loss": float(std_loss.detach()), "cov_loss": float(cov_loss.detach())}


class TemporalSimilarityLoss(nn.Module):
    """EB-JEPA (`eb_jepa/losses.py:109`) Temporal Smoothness Regularizer: $\|z_{t+1} - z_t\|_2^2$."""

    def forward(self, x_tbd: torch.Tensor) -> torch.Tensor:
        if x_tbd.shape[0] <= 1:
            return torch.tensor(0.0, device=x_tbd.device, dtype=x_tbd.dtype)
        return (x_tbd[1:] - x_tbd[:-1]).pow(2).mean()


class InverseDynamicsMLP(nn.Module):
    """EB-JEPA (`eb_jepa/architectures.py:InverseDynamicsModel`) action predictor from $(z_t, z_{t+1})$."""

    def __init__(self, state_dim: int, hidden_dim: int = 128, action_dim: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, s_t: torch.Tensor, s_tp1: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([s_t, s_tp1], dim=-1))


class InverseDynamicsLoss(nn.Module):
    """EB-JEPA (`eb_jepa/losses.py:128`) Inverse Dynamics Loss ensuring latent states retain controllable features."""

    def __init__(self, idm: nn.Module):
        super().__init__()
        self.idm = idm

    def forward(self, x_tbd: torch.Tensor, actions_bat: torch.Tensor | None) -> torch.Tensor:
        if x_tbd.shape[0] <= 1 or actions_bat is None:
            return torch.tensor(0.0, device=x_tbd.device, dtype=x_tbd.dtype)
        t, b, d = x_tbd.shape
        s_t = x_tbd[:-1].transpose(0, 1).reshape(-1, d)
        s_tp1 = x_tbd[1:].transpose(0, 1).reshape(-1, d)
        pred_a = self.idm(s_t, s_tp1)
        target_a = actions_bat.transpose(1, 2)[:, : t - 1].reshape(-1, actions_bat.size(1))
        return F.mse_loss(pred_a, target_a)


class VC_IDM_Sim_Regularizer(nn.Module):
    """EB-JEPA (`eb_jepa/losses.py:159`) Composite Regularizer (`VC + TemporalSimilarity + InverseDynamics`)."""

    def __init__(
        self,
        cov_coeff: float = 8.0,
        std_coeff: float = 16.0,
        sim_coeff_t: float = 12.0,
        idm_coeff: float = 1.0,
        idm: nn.Module | None = None,
        std_margin: float = 1.0,
    ):
        super().__init__()
        self.cov_coeff = float(cov_coeff)
        self.std_coeff = float(std_coeff)
        self.sim_coeff_t = float(sim_coeff_t)
        self.idm_coeff = float(idm_coeff)
        self.std_loss_fn = HingeStdLoss(std_margin=std_margin)
        self.cov_loss_fn = CovarianceLoss()
        self.sim_loss_fn = TemporalSimilarityLoss()
        self.idm_loss_fn = InverseDynamicsLoss(idm) if idm is not None else None

    def forward(self, x: torch.Tensor, actions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Accepts `[B, D, T]` or `[B, C, T, H, W]` state trajectories."""
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            x_tbd = x.permute(2, 0, 1, 3, 4).reshape(t, b, c * h * w)
        elif x.ndim == 3:
            # [B, D, T] -> [T, B, D]
            b, d, t = x.shape
            x_tbd = x.permute(2, 0, 1).contiguous()
        else:
            raise ValueError(f"Expected 3D [B, D, T] or 5D [B, C, T, H, W] tensor, got {x.shape}")

        x_flat = x_tbd.reshape(-1, x_tbd.shape[-1])
        std_loss = self.std_loss_fn(x_flat)
        cov_loss = self.cov_loss_fn(x_flat)
        sim_loss_t = self.sim_loss_fn(x_tbd)
        idm_loss = (
            self.idm_loss_fn(x_tbd, actions)
            if (self.idm_coeff > 0 and self.idm_loss_fn is not None and actions is not None)
            else torch.tensor(0.0, device=x.device, dtype=x.dtype)
        )

        weighted = (
            self.cov_coeff * cov_loss
            + self.std_coeff * std_loss
            + self.sim_coeff_t * sim_loss_t
            + self.idm_coeff * idm_loss
        )
        unweighted = cov_loss + std_loss + sim_loss_t + idm_loss
        info = {
            "cov_loss": float(cov_loss.detach()),
            "std_loss": float(std_loss.detach()),
            "sim_loss_t": float(sim_loss_t.detach()),
            "idm_loss": float(idm_loss.detach()),
        }
        return weighted, unweighted, info


def epps_pulley_statistic(x: torch.Tensor, t_min: float = -3.0, t_max: float = 3.0, n_points: int = 10) -> torch.Tensor:
    """EB-JEPA (`eb_jepa/losses.py:356`) Epps-Pulley characteristic-function test for standard Gaussianity."""
    t = torch.linspace(t_min, t_max, n_points, device=x.device, dtype=x.dtype)
    exp_f = torch.exp(-0.5 * t**2)
    x_t = x.unsqueeze(2) * t  # [N, M, T]
    ecf = (1j * x_t).exp().mean(0)
    err = exp_f * (ecf - exp_f).abs() ** 2
    return torch.trapz(err, t, dim=1)


class BCSLoss(nn.Module):
    """EB-JEPA (`eb_jepa/losses.py:372`) Batched Characteristic Slicing (SIGReg) loss."""

    def __init__(self, num_slices: int = 128, lmbd: float = 10.0):
        super().__init__()
        self.num_slices = int(num_slices)
        self.lmbd = float(lmbd)
        self.step = 0

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            g = torch.Generator(device=z1.device)
            g.manual_seed(self.step)
            a = torch.randn((z1.size(1), self.num_slices), device=z1.device, dtype=z1.dtype, generator=g)
            a = a / (a.norm(p=2, dim=0, keepdim=True) + 1e-8)
        self.step += 1
        v1 = z1 @ a
        v2 = z2 @ a
        bcs = 0.5 * (epps_pulley_statistic(v1).mean() + epps_pulley_statistic(v2).mean())
        inv = F.mse_loss(z1, z2)
        return {"loss": inv + self.lmbd * bcs, "bcs_loss": bcs, "invariance_loss": inv}


class MJEPA9PathLoss(nn.Module):
    """MJEPA (`app/mjepa/train.py:1069-1215`) 3 Intra-Modal + 6 Cross-Modal L1/Lp Loss."""

    def __init__(self, loss_exp: float = 1.0, cross_weight: float = 1.0, av_weight: float = 1.0):
        super().__init__()
        self.loss_exp = float(loss_exp)
        self.cross_weight = float(cross_weight)
        self.av_weight = float(av_weight)

    def lp_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_exp == 1.0:
            return torch.mean(torch.abs(pred - target))
        return torch.mean(torch.abs(pred - target) ** self.loss_exp) / self.loss_exp
