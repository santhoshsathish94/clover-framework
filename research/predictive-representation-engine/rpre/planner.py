"""Latent Model-Predictive Control (`CEMPlanner`, `MPPIPlanner`) & Clover Cycle Agent (`rpre/planner.py`).

Implements:
- `ReprTargetDistObjective` (`eb_jepa/planning.py:ReprTargetDistMPCObjective`): Latent energy distance
  between predicted rollout states `[B, D, T]` and encoded goal state `[1, D, 1]`.
- `CEMPlanner` (`eb_jepa/planning.py:CEMPlanner`): Iterative Cross-Entropy Method optimizer in latent space.
- `MPPIPlanner` (`eb_jepa/planning.py:MPPIPlanner`): Model Predictive Path Integral softmax-weighted trajectory optimizer.
- `CloverCycleAgent`: Executes the 5-stage Clover cycle (`Context -> Direction -> Execution -> Outcome -> Growth`)
  around `ReusablePredictiveEngine`, observing real outcome discrepancies and updating `OutcomeResidualAdapter`
  between episodes.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import torch

from .engine import ReusablePredictiveEngine


class PlanningResult(NamedTuple):
    actions: torch.Tensor
    losses: torch.Tensor
    elite_losses_mean: torch.Tensor
    elite_losses_std: torch.Tensor


class ReprTargetDistObjective:
    """EB-JEPA (`eb_jepa/planning.py:446`) Latent Goal Distance Objective."""

    def __init__(
        self,
        target_enc: torch.Tensor,
        sum_all_diffs: bool = True,
        terminal_weight: float = 2.0,
    ):
        # target_enc: [1, D] or [1, D, 1]
        if target_enc.ndim == 2:
            target_enc = target_enc.unsqueeze(-1)
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs
        self.terminal_weight = float(terminal_weight)

    def __call__(self, predicted_bdt: torch.Tensor) -> torch.Tensor:
        target = self.target_enc.expand(predicted_bdt.size(0), -1, predicted_bdt.size(2))
        diff = ((predicted_bdt - target) ** 2).mean(dim=1)  # [B, T]
        if self.sum_all_diffs:
            return diff[:, 1:].mean(dim=1) + self.terminal_weight * diff[:, -1]
        return diff[:, -1]


class CEMPlanner:
    """EB-JEPA (`eb_jepa/planning.py:514`) Cross-Entropy Method Latent Trajectory Planner."""

    def __init__(
        self,
        unroll_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_iters: int = 8,
        num_samples: int = 128,
        plan_length: int = 8,
        action_dim: int = 2,
        var_scale: float = 1.0,
        num_elites: int = 16,
        max_norm: float | None = 1.0,
        device: str | torch.device = "cpu",
    ):
        self.unroll_fn = unroll_fn
        self.n_iters = int(n_iters)
        self.num_samples = int(num_samples)
        self.plan_length = int(plan_length)
        self.action_dim = int(action_dim)
        self.var_scale = float(var_scale)
        self.num_elites = int(num_elites)
        self.max_norm = max_norm
        self.device = torch.device(device)
        self.objective: Callable[[torch.Tensor], torch.Tensor] | None = None

    def set_objective(self, objective: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.objective = objective

    @torch.no_grad()
    def plan(self, init_latent_bd: torch.Tensor, steps_left: int | None = None, seed: int | None = None) -> PlanningResult:
        assert self.objective is not None, "Planner objective must be set via set_objective()"
        horizon = self.plan_length if steps_left is None else min(self.plan_length, max(1, int(steps_left)))
        dev = init_latent_bd.device
        gen = torch.Generator(device=dev)
        if seed is not None:
            gen.manual_seed(int(seed))

        mean = torch.zeros((horizon, self.action_dim), device=dev, dtype=init_latent_bd.dtype)
        std = self.var_scale * torch.ones((horizon, self.action_dim), device=dev, dtype=init_latent_bd.dtype)

        losses = []
        elite_means = []
        elite_stds = []
        best_action: torch.Tensor | None = None

        for _ in range(self.n_iters):
            noise = torch.randn(
                (horizon, self.num_samples, self.action_dim),
                device=dev,
                dtype=init_latent_bd.dtype,
                generator=gen if seed is not None else None,
            )
            actions_tba = mean.unsqueeze(1) + std.unsqueeze(1) * noise  # [T, B, A]
            if best_action is not None:
                actions_tba[:, 0, :] = best_action
            if self.max_norm is not None:
                norms = actions_tba.norm(dim=-1, keepdim=True)
                scale = torch.clamp(norms, max=self.max_norm) / (norms + 1e-6)
                actions_tba = actions_tba * scale

            actions_bat = actions_tba.permute(1, 2, 0).contiguous()  # [B, A, T]
            init_rep = init_latent_bd.expand(self.num_samples, -1)
            pred_bdt = self.unroll_fn(init_rep, actions_bat)
            cost = self.objective(pred_bdt)  # [B]

            losses.append(float(cost.min().item()))
            elite_idx = torch.topk(-cost, k=min(self.num_elites, self.num_samples), dim=0).indices
            elite_cost = cost[elite_idx]
            elite_acts = actions_tba[:, elite_idx, :]  # [T, K, A]
            best_action = elite_acts[:, 0, :].clone()

            elite_means.append(float(elite_cost.mean().item()))
            elite_stds.append(float(elite_cost.std().item()) if elite_cost.numel() > 1 else 0.0)

            mean = elite_acts.mean(dim=1)
            std = elite_acts.std(dim=1).clamp(min=0.05)

        return PlanningResult(
            actions=mean,
            losses=torch.tensor(losses, device=dev),
            elite_losses_mean=torch.tensor(elite_means, device=dev),
            elite_losses_std=torch.tensor(elite_stds, device=dev),
        )


class MPPIPlanner:
    """EB-JEPA Model Predictive Path Integral (`MPPI`) Latent Trajectory Planner."""

    def __init__(
        self,
        unroll_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_iters: int = 6,
        num_samples: int = 128,
        plan_length: int = 8,
        action_dim: int = 2,
        temperature: float = 0.25,
        var_scale: float = 1.0,
        max_norm: float | None = 1.0,
    ):
        self.unroll_fn = unroll_fn
        self.n_iters = int(n_iters)
        self.num_samples = int(num_samples)
        self.plan_length = int(plan_length)
        self.action_dim = int(action_dim)
        self.temperature = float(temperature)
        self.var_scale = float(var_scale)
        self.max_norm = max_norm
        self.objective: Callable[[torch.Tensor], torch.Tensor] | None = None

    def set_objective(self, objective: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.objective = objective

    @torch.no_grad()
    def plan(self, init_latent_bd: torch.Tensor, steps_left: int | None = None) -> PlanningResult:
        assert self.objective is not None
        horizon = self.plan_length if steps_left is None else min(self.plan_length, max(1, int(steps_left)))
        dev = init_latent_bd.device
        mean = torch.zeros((horizon, self.action_dim), device=dev, dtype=init_latent_bd.dtype)
        losses = []
        for _ in range(self.n_iters):
            noise = self.var_scale * torch.randn((horizon, self.num_samples, self.action_dim), device=dev)
            actions_tba = mean.unsqueeze(1) + noise
            if self.max_norm is not None:
                norms = actions_tba.norm(dim=-1, keepdim=True)
                actions_tba = actions_tba * (torch.clamp(norms, max=self.max_norm) / (norms + 1e-6))
            actions_bat = actions_tba.permute(1, 2, 0).contiguous()
            pred_bdt = self.unroll_fn(init_latent_bd.expand(self.num_samples, -1), actions_bat)
            cost = self.objective(pred_bdt)
            losses.append(float(cost.min().item()))
            weights = torch.softmax(-(cost - cost.min()) / max(1e-4, self.temperature), dim=0)  # [B]
            mean = (actions_tba * weights.view(1, -1, 1)).sum(dim=1)
        return PlanningResult(
            actions=mean,
            losses=torch.tensor(losses, device=dev),
            elite_losses_mean=torch.tensor(losses, device=dev),
            elite_losses_std=torch.zeros(len(losses), device=dev),
        )


class CloverCycleAgent:
    """Runs the 5-stage Clover cycle (`Context -> Direction -> Execution -> Outcome -> Growth`) around `ReusablePredictiveEngine`."""

    def __init__(
        self,
        engine: ReusablePredictiveEngine,
        use_outcome_adapter: bool = True,
        use_cross_modal_cue: bool = True,
        plan_length: int = 8,
        n_iters: int = 8,
        num_samples: int = 128,
        max_norm: float = 0.6,
    ):
        self.engine = engine
        self.use_outcome_adapter = use_outcome_adapter
        self.use_cross_modal_cue = use_cross_modal_cue
        self.current_cue: torch.Tensor | None = None

        def _unroll(init_bd: torch.Tensor, actions_bat: torch.Tensor) -> torch.Tensor:
            cue = (
                self.current_cue.expand(init_bd.size(0), -1)
                if (self.use_cross_modal_cue and self.current_cue is not None)
                else None
            )
            pred_bdt, _ = self.engine.unroll_latent(
                init_latent_bd=init_bd,
                actions_bat=actions_bat,
                nsteps=actions_bat.size(2),
                unroll_mode="autoregressive",
                compute_loss=False,
                use_outcome_adapter=self.use_outcome_adapter,
                context_cue=cue,
            )
            return pred_bdt

        self.planner = CEMPlanner(
            unroll_fn=_unroll,
            n_iters=n_iters,
            num_samples=num_samples,
            plan_length=plan_length,
            action_dim=engine.transition_predictor.action_dim,
            max_norm=max_norm,
        )

    @torch.no_grad()
    def establish_context(
        self,
        state_x: torch.Tensor,
        visual_x: torch.Tensor | None = None,
        audio_x: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Stage 1 (Context): Encode state and multimodal sensory cues into latent representation space."""
        z_state = self.engine.encoder(state_x=state_x)["state"].squeeze(1)
        cue = None
        if visual_x is not None and audio_x is not None:
            vz = self.engine.encoder(visual_x=visual_x, return_hierarchical=False)["visual"]
            az = self.engine.encoder(audio_x=audio_x, return_hierarchical=False)["audio"]
            cue = 0.5 * (az.mean(dim=1) + vz.mean(dim=1))
        elif visual_x is not None:
            vz = self.engine.encoder(visual_x=visual_x, return_hierarchical=False)["visual"]
            # Use cross-modal predictor to infer missing audio representation and fuse
            az_hat = self.engine.v2a_predictor(vz)
            cue = 0.5 * (vz.mean(dim=1) + az_hat)
        elif audio_x is not None:
            az = self.engine.encoder(audio_x=audio_x, return_hierarchical=False)["audio"]
            vz_hat = self.engine.a2v_predictor(az)
            cue = 0.5 * (az.mean(dim=1) + vz_hat)
        self.current_cue = cue
        return z_state, cue

    @torch.no_grad()
    def set_direction(self, goal_state_x: torch.Tensor) -> torch.Tensor:
        """Stage 2 (Direction): Encode human-specified goal state and configure planner objective."""
        z_goal = self.engine.encoder(state_x=goal_state_x)["state"].squeeze(1)
        self.planner.set_objective(ReprTargetDistObjective(z_goal, sum_all_diffs=True))
        return z_goal

    @torch.no_grad()
    def select_action(self, z_state: torch.Tensor, steps_left: int | None = None, seed: int | None = None) -> torch.Tensor:
        """Stage 3 (Execution): Plan action trajectory in latent space and return first action $a_1^*$."""
        res = self.planner.plan(z_state, steps_left=steps_left, seed=seed)
        return res.actions[0:1]  # [1, action_dim]

    @torch.no_grad()
    def observe_outcome(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        next_state_x: torch.Tensor,
        cue: torch.Tensor | None = None,
        is_baseline: bool = False,
    ) -> tuple[torch.Tensor, float]:
        """Stage 4 (Outcome): Encode real next state $z_{t+1}^{\text{obs}}$, measure prediction error, and record for Growth."""
        z_tp1_obs = self.engine.encoder(state_x=next_state_x)["state"].squeeze(1)
        pred_bdt, _ = self.engine.unroll_latent(
            init_latent_bd=z_t,
            actions_bat=a_t.unsqueeze(-1),
            nsteps=1,
            unroll_mode="autoregressive",
            compute_loss=False,
            use_outcome_adapter=self.use_outcome_adapter,
            context_cue=cue if self.use_cross_modal_cue else None,
        )
        z_tp1_pred = pred_bdt[:, :, -1]
        pred_mse = float(((z_tp1_obs - z_tp1_pred) ** 2).mean().item())
        if self.use_outcome_adapter:
            self.engine.outcome_adapter.record_outcome(
                z_t, a_t, z_tp1_obs, c_t=cue if self.use_cross_modal_cue else None, is_baseline=is_baseline
            )
        return z_tp1_obs, pred_mse

    def grow(self, steps: int = 40, lr: float = 5e-3) -> dict:
        """Stage 5 (Growth): Update `OutcomeResidualAdapter` from observed outcome errors with non-regression gating."""
        if not self.use_outcome_adapter:
            return {"updated": False, "reason": "outcome_adapter_disabled"}
        return self.engine.outcome_adapter.update_from_outcomes(
            self.engine.transition_predictor, steps=steps, lr=lr
        )
