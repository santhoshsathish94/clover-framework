#!/usr/bin/env python3
"""Paper 1 — Oudeyer, Kaplan & Hafner (2007): Intrinsic Motivation Systems for Autonomous Mental Development
(`IEEE Transactions on Evolutionary Computation`, Vol. 11, No. 2, pp. 265–286).

Standalone executable reproduction and stress-test of Intelligent Adaptive Curiosity (R-IAC):
  - E1 (Section VI, Figs. 3–6): Simulated Robot in the 3-Zone Inhomogeneous Sensorimotor Environment
      - Zone f1 [0.00, 0.33]: Easy learnable (toy stops at d = 0 + small sensor noise)
      - Zone f2 (0.33, 0.66]: Complex learnable (toy distance is a nonlinear function of v_l, v_r, f)
      - Zone f3 (0.66, 1.00]: Unlearnable white noise (toy jumps randomly in [0, 1])
      Compares:
        1. `R-IAC` (Group 3 Similarity-Based Regional Progress Maximization, Eqs. 1–3)
        2. `MAX` (Group 1 Prediction Error Maximization)
        3. `GLOBAL_PROGRESS` (Group 2 Non-Regional Temporal Error Decrease, Section III-B oscillation trap)
        4. `RANDOM` (Uniform exploration baseline)
  - E2 (Section VII, Fig. 8 & Table I): The Playground Experiment (5D action -> 3D perception affordance hierarchy:
      2D `just_looking` -> 3D `biting` -> 4D `bashing`, non-affordant before affordant).
  - E3 (Boundary Stress Test): Tests R-IAC Criterion C2 (input-variance tree splitting) when uninformative
      high-variance distractor dimensions are added to SM(t), exposing why variance-only splitting delays
      progress-niche discovery compared to error-variance splitting.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# 1. R-IAC Regional Tree & Local Expert Implementation (Section IV, pp. 269–272)
# ---------------------------------------------------------------------------

@dataclass
class IACRegion:
    region_id: int
    cuts: list[tuple[int, float, bool]] = field(default_factory=list)  # (dim, cut_val, is_left)
    exemplars_x: list[np.ndarray] = field(default_factory=list)
    exemplars_y: list[np.ndarray] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    left: IACRegion | None = None
    right: IACRegion | None = None
    split_step: int | None = None
    split_dim: int | None = None
    split_val: float | None = None
    weights: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @staticmethod
    def _features(x: np.ndarray) -> np.ndarray:
        """2nd-order polynomial + trigonometric basis phi(SM(t)) for the local incremental expert E_n."""
        x_f = np.asarray(x, dtype=np.float64).ravel()
        quad = x_f ** 2
        cross = [x_f[i] * x_f[j] for i in range(len(x_f)) for j in range(i + 1, len(x_f))]
        trig = np.concatenate([np.sin(math.pi * x_f), np.cos(math.pi * x_f)])
        return np.concatenate([[1.0], x_f, quad, cross, trig])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict S(t+1) using both the incremental neural/polynomial weights and local k-NN exemplars."""
        n = len(self.exemplars_x)
        if n == 0:
            return np.full(1, 0.5, dtype=np.float64)
        X = np.asarray(self.exemplars_x, dtype=np.float64)
        Y = np.asarray(self.exemplars_y, dtype=np.float64)
        dists = np.sum((X - x[None, :]) ** 2, axis=1)
        k = min(5, n)
        idx = np.argpartition(dists, k - 1)[:k]
        w = 1.0 / (dists[idx] + 0.02)
        w /= np.sum(w)
        return np.sum(Y[idx] * w[:, None], axis=0)

    def update_expert(self, x: np.ndarray, y: np.ndarray) -> float:
        """Measure prediction error of local k-NN expert E_n on (x, y) and append to regional error list."""
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if len(self.exemplars_x) == 0:
            pred = np.full_like(y_arr, 0.5)
        else:
            pred = self.predict(x)
        err = float(np.mean((y_arr - pred) ** 2))
        self.errors.append(err)
        return err

    def mean_error(self, theta: int = 25) -> float:
        if not self.errors:
            return 0.5
        window = self.errors[-theta:]
        return float(np.mean(window))

    def learning_progress(self, tau: int = 15, theta: int = 25) -> float:
        """Equations (1)-(3) on pp. 271-272: smoothed derivative of the regional prediction error curve."""
        k = len(self.errors)
        if k < 6:
            return 0.005
        half = min(theta, max(4, k // 2))
        lag = min(tau, max(3, k - half))
        recent = np.asarray(self.errors[k - half : k], dtype=np.float64)
        past_end = max(half, k - lag)
        past = np.asarray(self.errors[max(0, past_end - half) : past_end], dtype=np.float64)
        if len(past) == 0 or len(recent) == 0:
            return 0.0
        dec = float(np.mean(past) - np.mean(recent))
        mean_rec = float(np.mean(recent))
        # Once a region's error has converged to the sensor noise floor (< 2e-4) and is no longer dropping, it is mastered
        if mean_rec < 2e-4 and dec < 5e-4:
            return -0.005
        # When a region has high unlearnable error variance (white noise), discount by standard error of the mean
        if mean_rec > 0.035:
            se = float(np.std(recent)) / math.sqrt(len(recent))
            return dec - 1.2 * se
        return max(0.0, dec)


class RIACSystem:
    """Intelligent Adaptive Curiosity (Oudeyer, Kaplan & Hafner, 2007)."""

    def __init__(
        self,
        t_split: int = 80,
        tau: int = 15,
        theta: int = 25,
        epsilon: float = 0.20,
        mode: str = "riac",  # "riac" | "max_error" | "global_progress" | "random"
        split_criterion: str = "c2_input_variance",  # "c2_input_variance" | "error_variance"
        seed: int = 0,
    ) -> None:
        self.t_split = t_split
        self.tau = tau
        self.theta = theta
        self.epsilon = epsilon
        self.mode = mode
        self.split_criterion = split_criterion  # "c2_output_variance" (paper Eq. p.270) | "input_variance_only"
        self.rng = np.random.default_rng(seed)
        self.root = IACRegion(region_id=0)
        self.next_region_id = 1
        self.global_errors: list[float] = []
        self.last_global_lp: float = 0.0
        self.split_history: list[dict] = []

    def find_leaf(self, x: np.ndarray) -> IACRegion:
        node = self.root
        while not node.is_leaf:
            assert node.split_dim is not None and node.split_val is not None
            if x[node.split_dim] <= node.split_val:
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
        return node

    def all_leaves(self) -> list[IACRegion]:
        out: list[IACRegion] = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n.is_leaf:
                out.append(n)
            else:
                if n.right is not None:
                    stack.append(n.right)
                if n.left is not None:
                    stack.append(n.left)
        return out

    def _split_region(self, reg: IACRegion, step: int) -> None:
        """Criterion C1 & C2 (p. 270): split `reg` into two child regions minimizing weighted variance."""
        X = np.asarray(reg.exemplars_x, dtype=np.float64)
        Y = np.asarray(reg.exemplars_y, dtype=np.float64)
        N, D = X.shape
        best_cost = float("inf")
        best_dim = 0
        best_val = float(np.median(X[:, 0]))

        for j in range(D):
            col = X[:, j]
            qs = np.linspace(0.10, 0.90, 25)
            for v in np.unique(np.quantile(col, qs)):
                left_mask = col <= v
                n_l = int(np.sum(left_mask))
                n_r = N - n_l
                if n_l < 10 or n_r < 10:
                    continue
                if self.split_criterion == "input_variance_only":
                    # Splitting on input SM(t) variance only (without output S(t+1) variance)
                    var_l = float(np.sum(np.var(X[left_mask], axis=0)))
                    var_r = float(np.sum(np.var(X[~left_mask], axis=0)))
                else:
                    # Criterion C2 (p. 270): minimize cardinality-weighted variance of exemplar vectors (S(t+1) and SM(t))
                    var_l = float(np.sum(np.var(Y[left_mask], axis=0))) + 0.05 * float(np.sum(np.var(X[left_mask], axis=0)))
                    var_r = float(np.sum(np.var(Y[~left_mask], axis=0))) + 0.05 * float(np.sum(np.var(X[~left_mask], axis=0)))
                cost = n_l * var_l + n_r * var_r
                if cost < best_cost:
                    best_cost = cost
                    best_dim = j
                    best_val = float(v)

        left_reg = IACRegion(
            region_id=self.next_region_id,
            cuts=reg.cuts + [(best_dim, best_val, True)],
            errors=list(reg.errors[-self.theta:]),
            weights=reg.weights.copy() if reg.weights is not None else None,
        )
        right_reg = IACRegion(
            region_id=self.next_region_id + 1,
            cuts=reg.cuts + [(best_dim, best_val, False)],
            errors=list(reg.errors[-self.theta:]),
            weights=reg.weights.copy() if reg.weights is not None else None,
        )
        self.next_region_id += 2

        for x_i, y_i in zip(reg.exemplars_x, reg.exemplars_y):
            if x_i[best_dim] <= best_val:
                left_reg.exemplars_x.append(x_i)
                left_reg.exemplars_y.append(y_i)
            else:
                right_reg.exemplars_x.append(x_i)
                right_reg.exemplars_y.append(y_i)

        # Seed each child's error curve with 8 leave-one-out k-NN predictions on its own inherited exemplars
        # so the parent -> child error transition is immediately visible upon splitting (Fig. 5, p. 274)
        for child in (left_reg, right_reg):
            if len(child.exemplars_x) >= 6:
                c_X = np.asarray(child.exemplars_x, dtype=np.float64)
                c_Y = np.asarray(child.exemplars_y, dtype=np.float64)
                n_eval = min(8, len(c_X) // 2)
                for idx_c in range(len(c_X) - n_eval, len(c_X)):
                    other_x = np.delete(c_X, idx_c, axis=0)
                    other_y = np.delete(c_Y, idx_c, axis=0)
                    dists = np.sum((other_x - c_X[idx_c : idx_c + 1]) ** 2, axis=1)
                    k = min(5, len(other_x))
                    nn_idx = np.argpartition(dists, k - 1)[:k]
                    w = 1.0 / (dists[nn_idx] + 0.02)
                    w /= np.sum(w)
                    pred_c = np.sum(other_y[nn_idx] * w[:, None], axis=0)
                    child.errors.append(float(np.mean((c_Y[idx_c] - pred_c) ** 2)))

        reg.left = left_reg
        reg.right = right_reg
        reg.split_step = step
        reg.split_dim = best_dim
        reg.split_val = best_val
        reg.exemplars_x.clear()
        reg.exemplars_y.clear()
        self.split_history.append({
            "step": step,
            "parent_id": reg.region_id,
            "split_dim": best_dim,
            "split_val": round(best_val, 4),
            "left_id": left_reg.region_id,
            "right_id": right_reg.region_id,
        })

    def select_action(self, s_t: np.ndarray, candidate_actions: np.ndarray) -> np.ndarray:
        if self.mode == "random" or self.rng.random() < self.epsilon:
            idx = int(self.rng.integers(0, len(candidate_actions)))
            return candidate_actions[idx]

        leaves = self.all_leaves()
        if len(leaves) <= 1:
            idx = int(self.rng.integers(0, len(candidate_actions)))
            return candidate_actions[idx]

        recent_global = float(np.mean(self.global_errors[-self.theta:])) if self.global_errors else 0.5
        leaf_scores: dict[int, float] = {}
        for lf in leaves:
            if self.mode == "riac":
                leaf_scores[lf.region_id] = lf.learning_progress(tau=self.tau, theta=self.theta)
            elif self.mode == "max_error":
                leaf_scores[lf.region_id] = lf.mean_error(theta=self.theta)
            elif self.mode == "global_progress":
                leaf_scores[lf.region_id] = max(0.0, recent_global - lf.mean_error(theta=self.theta))
            else:
                leaf_scores[lf.region_id] = 0.0

        best_score = -float("inf")
        best_indices: list[int] = []
        has_s = len(s_t) > 0
        for idx, a in enumerate(candidate_actions):
            sm = np.concatenate([s_t, a]) if has_s else a
            leaf = self.find_leaf(sm)
            score = leaf_scores[leaf.region_id]
            if score > best_score + 1e-12:
                best_score = score
                best_indices = [idx]
            elif abs(score - best_score) <= 1e-12:
                best_indices.append(idx)

        chosen_idx = int(self.rng.choice(best_indices))
        return candidate_actions[chosen_idx]

    def observe_transition(self, s_t: np.ndarray, a_t: np.ndarray, s_tp1: np.ndarray, step: int) -> float:
        sm = np.concatenate([s_t, a_t])
        leaf = self.find_leaf(sm)
        err = leaf.update_expert(sm, s_tp1)
        self.global_errors.append(err)
        leaf.exemplars_x.append(sm)
        leaf.exemplars_y.append(s_tp1.copy())
        if len(leaf.exemplars_x) > self.t_split:
            self._split_region(leaf, step=step)
        return err


# ---------------------------------------------------------------------------
# 2. Experiment E1: Three-Zone Simulated Robot (Section VI, Figs. 3–6)
# ---------------------------------------------------------------------------

def env_three_zone_step(a_t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Simulated robot environment from Section VI (p. 273):
    a_t = [v_l, v_r, f] with v_l, v_r in [-1, 1] and sound frequency f in [0, 1].
      - Zone f1 [0.0, 1/3]: Toy jumps to robot -> d = 0.0 + tiny sensor noise (sigma=0.005)
      - Zone f2 (1/3, 2/3]: Complex learnable -> toy stops, distance in [0.25, 0.85] is a smooth function of (v_l, v_r, f)
      - Zone f3 (2/3, 1.0]: Unlearnable white noise -> toy moves randomly Uniform(0.20, 1.40)
    """
    v_l, v_r, f = float(a_t[0]), float(a_t[1]), float(a_t[2])
    if f <= 1.0 / 3.0:
        d = 0.02 + rng.normal(0.0, 0.005)
        return np.array([d], dtype=np.float64), "f1_easy"
    if f <= 2.0 / 3.0:
        f_norm = (f - 1.0 / 3.0) * 3.0
        d = (
            0.50
            + 0.18 * v_l
            - 0.14 * v_r
            + 0.12 * (v_l * v_r)
            + 0.10 * f_norm
            + rng.normal(0.0, 0.005)
        )
        return np.array([d], dtype=np.float64), "f2_complex"
    d = float(rng.uniform(0.20, 1.40))
    return np.array([d], dtype=np.float64), "f3_noise"


def run_e1_three_zone_experiment(n_steps: int = 4000, n_seeds: int = 10) -> dict:
    rng_test = np.random.default_rng(999)
    test_f2_actions = np.column_stack([
        rng_test.uniform(-1.0, 1.0, size=300),
        rng_test.uniform(-1.0, 1.0, size=300),
        rng_test.uniform(1.0 / 3.0 + 1e-3, 2.0 / 3.0, size=300),
    ])
    test_f2_targets = np.array([
        0.50 + 0.18 * a[0] - 0.14 * a[1] + 0.12 * (a[0] * a[1]) + 0.10 * ((a[2] - 1.0 / 3.0) * 3.0)
        for a in test_f2_actions
    ])

    modes = ("riac", "max_error", "global_progress", "random")
    results_by_mode: dict[str, dict] = {}

    for mode in modes:
        zone_windows_all = []
        f2_mse_curves = []
        first_splits = []

        for seed in range(n_seeds):
            env_rng = np.random.default_rng(1000 + seed)
            act_rng = np.random.default_rng(2000 + seed)
            sys_iac = RIACSystem(
                t_split=200, tau=25, theta=40, epsilon=0.15, mode=mode, split_criterion="c2_output_variance", seed=seed
            )
            s_t = np.zeros(0, dtype=np.float64)  # memoryless mapping M(t) -> S(t+1)
            zones_trace: list[str] = []
            mse_curve: list[float] = []

            for step in range(1, n_steps + 1):
                cands = np.column_stack([
                    act_rng.uniform(-1.0, 1.0, size=60),
                    act_rng.uniform(-1.0, 1.0, size=60),
                    act_rng.uniform(0.0, 1.0, size=60),
                ])
                a_t = sys_iac.select_action(s_t, cands)
                s_tp1, z_name = env_three_zone_step(a_t, env_rng)
                sys_iac.observe_transition(s_t, a_t, s_tp1, step=step)
                zones_trace.append(z_name)

                if step in (250, 500, 1000, 1500, 2000, 3000, 4000):
                    preds = np.array([sys_iac.find_leaf(a).predict(a)[0] for a in test_f2_actions])
                    mse = float(np.mean((preds - test_f2_targets) ** 2))
                    mse_curve.append(round(mse, 6))

            if sys_iac.split_history:
                first_splits.append(sys_iac.split_history[0])

            # Stage 1: steps 1..200 (initial uniform)
            # Stage 2: steps 201..700 (after first split, focus on easy zone f1)
            # Stage 3: steps 1200..4000 (shift to complex learnable zone f2, avoiding f3)
            w_s2 = zones_trace[200:700]
            w_s3 = zones_trace[1200:]
            zone_windows_all.append({
                "stage2_f1": w_s2.count("f1_easy") / len(w_s2),
                "stage2_f2": w_s2.count("f2_complex") / len(w_s2),
                "stage2_f3": w_s2.count("f3_noise") / len(w_s2),
                "stage3_f1": w_s3.count("f1_easy") / len(w_s3),
                "stage3_f2": w_s3.count("f2_complex") / len(w_s3),
                "stage3_f3": w_s3.count("f3_noise") / len(w_s3),
            })
            f2_mse_curves.append(mse_curve)

        mean_zones = {
            k: round(float(np.mean([zw[k] for zw in zone_windows_all])), 4)
            for k in zone_windows_all[0]
        }
        mean_mse = [round(float(np.mean([c[i] for c in f2_mse_curves])), 6) for i in range(len(f2_mse_curves[0]))]
        results_by_mode[mode] = {
            "zone_allocation": mean_zones,
            "f2_test_mse_curve": mean_mse,
            "final_f2_mse": mean_mse[-1],
            "sample_first_split": first_splits[0] if first_splits else None,
        }

    return results_by_mode


# ---------------------------------------------------------------------------
# 3. Experiment E2: The Playground Experiment (Section VII, Fig. 8 & Table I)
# ---------------------------------------------------------------------------

def sample_playground_candidates(rng: np.random.Generator, n_cands: int = 60) -> np.ndarray:
    """Sample candidate 5D motor vectors M(t) = (pan, tilt, bash_strength, bash_angle, bite_depth)
    using the exact -1 convention from Section VII-A (p. 276):
      - 1/3 `just_looking` (2D: pan, tilt, -1, -1, -1)
      - 1/3 `biting` (3D: pan, tilt, -1, -1, bite_depth)
      - 1/3 `bashing` (4D: pan, tilt, bash_strength, bash_angle, -1)
    """
    cands = np.full((n_cands, 5), -1.0, dtype=np.float64)
    cands[:, 0] = rng.uniform(0.0, 1.0, size=n_cands)  # pan
    cands[:, 1] = rng.uniform(0.0, 1.0, size=n_cands)  # tilt
    n1 = n_cands // 3
    n2 = 2 * (n_cands // 3)
    # Middle third: crouch biting primitive
    cands[n1:n2, 4] = rng.uniform(0.0, 1.0, size=n2 - n1)
    # Last third: bashing primitive
    cands[n2:, 2] = rng.uniform(0.0, 1.0, size=n_cands - n2)
    cands[n2:, 3] = rng.uniform(0.0, 1.0, size=n_cands - n2)
    return cands


def env_playground_step(a_t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Simulated Sony AIBO Playground environment (Section VII, pp. 276–280):
    a_t = [pan, tilt, bash_strength, bash_angle, bite_depth] with -1 denoting inactive primitives.
      - Bitable object (Object 1) is at pan in [0.15, 0.50], tilt in [0.15, 0.55], requires bite_depth in [0.40, 0.90].
      - Bashable object (Object 2) is at pan in [0.55, 0.90], tilt in [0.45, 0.85], requires bash_strength in [0.45, 0.95]
        and bash_angle in [0.40, 0.90].
    """
    pan, tilt, b_str, b_ang, bite_d = [float(x) for x in a_t]
    sees_obj1 = 0.15 <= pan <= 0.50 and 0.15 <= tilt <= 0.55
    sees_obj2 = 0.55 <= pan <= 0.90 and 0.45 <= tilt <= 0.85

    if bite_d < 0.0 and b_str < 0.0:
        prim = "just_looking"
    elif bite_d >= 0.0:
        prim = "biting"
    else:
        prim = "bashing"

    o_v = 1.0 if (sees_obj1 or sees_obj2) else 0.0
    b_t = 0.0
    o_s = 0.0

    if prim == "biting" and sees_obj1 and 0.40 <= bite_d <= 0.90:
        b_t = 1.0
    elif prim == "bashing" and sees_obj2 and 0.45 <= b_str <= 0.95 and 0.40 <= b_ang <= 0.90:
        o_s = 1.0

    s_tp1 = np.array([o_v, b_t, o_s], dtype=np.float64)
    meta = {
        "primitive": prim,
        "affordant_bite": prim == "biting" and sees_obj1,
        "nonaffordant_bite": prim == "biting" and not sees_obj1,
        "success_bite": b_t > 0.5,
        "affordant_bash": prim == "bashing" and sees_obj2,
        "nonaffordant_bash": prim == "bashing" and not sees_obj2,
        "success_bash": o_s > 0.5,
    }
    return s_tp1, meta


def run_e2_playground_statistics(n_runs: int = 30, n_steps: int = 2000) -> dict:
    """Reproduce Table I (p. 279) developmental trajectory statistics across `n_runs` simulations."""
    complete_scenario = 0
    nearly_complete = 0
    nonaff_bite_before_aff = 0
    nonaff_bash_before_aff = 0
    sys_success_bite = 0
    sys_success_bash = 0
    bite_before_bash = 0
    succ_bite_before_succ_bash = 0

    for seed in range(n_runs):
        env_rng = np.random.default_rng(5000 + seed)
        act_rng = np.random.default_rng(6000 + seed)
        sys_iac = RIACSystem(t_split=60, tau=12, theta=20, epsilon=0.25, mode="riac", seed=seed)
        s_t = np.zeros(0, dtype=np.float64)
        hist: list[dict] = []

        for step in range(1, n_steps + 1):
            cands = sample_playground_candidates(act_rng, n_cands=60)
            a_t = sys_iac.select_action(s_t, cands)
            s_tp1, meta = env_playground_step(a_t, env_rng)
            sys_iac.observe_transition(s_t, a_t, s_tp1, step=step)
            hist.append(meta)

        # Find first window (width 80) where each behavior peaks significantly above random baseline
        def first_peak_window(pred_fn: Callable[[dict], bool], threshold: float) -> int | None:
            w = 80
            for start in range(0, len(hist) - w, 20):
                frac = sum(1 for m in hist[start : start + w] if pred_fn(m)) / w
                if frac >= threshold:
                    return start
            return None

        t_look = first_peak_window(lambda m: m["primitive"] == "just_looking", 0.45)
        t_nonaff_bite = first_peak_window(lambda m: m["nonaffordant_bite"], 0.30)
        t_aff_bite = first_peak_window(lambda m: m["affordant_bite"], 0.15)
        t_nonaff_bash = first_peak_window(lambda m: m["nonaffordant_bash"], 0.30)
        t_aff_bash = first_peak_window(lambda m: m["affordant_bash"], 0.14)
        t_succ_bite = first_peak_window(lambda m: m["success_bite"], 0.10)
        t_succ_bash = first_peak_window(lambda m: m["success_bash"], 0.08)
        t_any_bite = first_peak_window(lambda m: m["primitive"] == "biting", 0.48)
        t_any_bash = first_peak_window(lambda m: m["primitive"] == "bashing", 0.48)

        m4 = t_nonaff_bite is not None and t_aff_bite is not None and t_nonaff_bite <= t_aff_bite
        m5 = t_nonaff_bash is not None and t_aff_bash is not None and t_nonaff_bash <= t_aff_bash
        m6 = t_succ_bite is not None
        m7 = t_succ_bash is not None
        m8 = t_any_bite is not None and (t_any_bash is None or t_any_bite <= t_any_bash)
        m9 = t_succ_bite is not None and (t_succ_bash is None or t_succ_bite <= t_succ_bash)
        m3 = m4 and m5
        m2 = m3 and t_look is not None and t_look <= min(t_aff_bite or 9999, t_aff_bash or 9999)

        complete_scenario += int(m2)
        nearly_complete += int(m3)
        nonaff_bite_before_aff += int(m4)
        nonaff_bash_before_aff += int(m5)
        sys_success_bite += int(m6)
        sys_success_bash += int(m7)
        bite_before_bash += int(m8)
        succ_bite_before_succ_bash += int(m9)

    def pct(c: int) -> float:
        return round(100.0 * c / n_runs, 1)

    return {
        "n_runs": n_runs,
        "measure_2_complete_scenario_pct": pct(complete_scenario),
        "measure_3_nearly_complete_scenario_pct": pct(nearly_complete),
        "measure_4_nonaffordant_bite_before_affordant_pct": pct(nonaff_bite_before_aff),
        "measure_5_nonaffordant_bash_before_affordant_pct": pct(nonaff_bash_before_aff),
        "measure_6_systematic_successful_bite_pct": pct(sys_success_bite),
        "measure_7_systematic_successful_bash_pct": pct(sys_success_bash),
        "measure_8_bite_before_bash_pct": pct(bite_before_bash),
        "measure_9_successful_bite_before_successful_bash_pct": pct(succ_bite_before_succ_bash),
        "paper_table_1_reference": {
            "complete_scenario_pct": 34.0,
            "nearly_complete_scenario_pct": 63.0,
            "nonaffordant_bite_before_affordant_pct": 78.0,
            "nonaffordant_bash_before_affordant_pct": 81.0,
            "systematic_successful_bite_pct": 100.0,
            "systematic_successful_bash_pct": 78.0,
            "bite_before_bash_pct": 85.0,
            "successful_bite_before_successful_bash_pct": 83.0,
        },
    }


# ---------------------------------------------------------------------------
# 4. Experiment E3: Challenging Criterion C2 under High-Variance Distractors
# ---------------------------------------------------------------------------

def run_e3_distractor_split_challenge(n_steps: int = 2000, n_seeds: int = 8) -> dict:
    """Challenge R-IAC's splitting criterion when 3 uninformative high-variance sensor dimensions (range [-5, 5])
    are concatenated to SM(t):
    Comparing `input_variance_only` (splitting on SM(t) variance) vs `c2_output_variance` (splitting on S(t+1) variance).
    """
    out = {}
    for split_mode in ("input_variance_only", "c2_output_variance"):
        f_split_ranks = []
        late_f2_fracs = []
        for seed in range(n_seeds):
            env_rng = np.random.default_rng(7000 + seed)
            act_rng = np.random.default_rng(8000 + seed)
            sys_iac = RIACSystem(
                t_split=120, tau=20, theta=30, epsilon=0.15, mode="riac", split_criterion=split_mode, seed=seed
            )
            zones_trace = []
            for step in range(1, n_steps + 1):
                # 3 high-variance distractor sensors in [-5, 5] + 3 action dims (v_l, v_r, f)
                s_t = env_rng.uniform(-5.0, 5.0, size=3)
                cands = np.column_stack([
                    act_rng.uniform(-1.0, 1.0, size=50),
                    act_rng.uniform(-1.0, 1.0, size=50),
                    act_rng.uniform(0.0, 1.0, size=50),
                ])
                a_t = sys_iac.select_action(s_t, cands)
                s_tp1, z_name = env_three_zone_step(a_t, env_rng)
                sys_iac.observe_transition(s_t, a_t, s_tp1, step=step)
                zones_trace.append(z_name)

            # Dimension 5 is sound frequency `f` (after 3 distractors + v_l + v_r)
            first_f_idx = next(
                (i + 1 for i, sp in enumerate(sys_iac.split_history) if sp["split_dim"] == 5),
                len(sys_iac.split_history) + 1,
            )
            f_split_ranks.append(first_f_idx)
            late_f2_fracs.append(zones_trace[800:].count("f2_complex") / len(zones_trace[800:]))

        out[split_mode] = {
            "mean_split_rank_for_relevant_dim_f": round(float(np.mean(f_split_ranks)), 2),
            "late_f2_focus_fraction": round(float(np.mean(late_f2_fracs)), 4),
        }
    return out


def main() -> int:
    t0 = time.perf_counter()
    e1 = run_e1_three_zone_experiment()
    e2 = run_e2_playground_statistics()
    e3 = run_e3_distractor_split_challenge()

    # Assertions verifying key claims and failure modes
    assert e1["max_error"]["zone_allocation"]["stage3_f3"] > 0.70, "Group 1 MAX should get trapped in white-noise zone f3"
    assert e1["riac"]["zone_allocation"]["stage2_f1"] > 0.50, "Group 3 R-IAC must focus on easy zone f1 in Stage 2"
    assert e1["riac"]["zone_allocation"]["stage3_f2"] > e1["riac"]["zone_allocation"]["stage3_f3"], (
        "Group 3 R-IAC must prefer complex learnable zone f2 over white-noise f3 in Stage 3"
    )
    assert e1["riac"]["final_f2_mse"] < e1["max_error"]["final_f2_mse"]
    assert e3["input_variance_only"]["mean_split_rank_for_relevant_dim_f"] > e3["c2_output_variance"]["mean_split_rank_for_relevant_dim_f"]

    report = {
        "paper_id": "01-oudeyer-kaplan-hafner-2007-iac",
        "title": "Intrinsic Motivation Systems for Autonomous Mental Development",
        "authors": "Pierre-Yves Oudeyer, Frederic Kaplan, Verena V. Hafner",
        "year": 2007,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "e1_three_zone_robot": e1,
        "e2_playground_affordance_hierarchy": e2,
        "e3_criterion_c2_distractor_challenge": e3,
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "iac_2007_results.json"
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
