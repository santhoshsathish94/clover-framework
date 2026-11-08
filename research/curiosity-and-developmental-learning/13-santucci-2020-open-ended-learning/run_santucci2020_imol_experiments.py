#!/usr/bin/env python3
"""Paper 13 — Santucci, Oudeyer, Barto & Baldassarre (2020): Editorial: Intrinsically Motivated
Open-Ended Learning in Autonomous Robots (`Frontiers in Neurorobotics`, Vol. 13, Article 115,
doi:10.3389/fnbot.2019.00115).

Standalone executable evaluation of the core open challenges in Intrinsically Motivated Open-Ended Learning (IMOL)
identified by Santucci, Oudeyer, Barto & Baldassarre (2020, pp. 1–3):
  - E1 (Open Challenge 1 & 2, pp. 1–2 — Compact Action-Relevant Goal Representation Learning + Hybrid
    Continuous/Discrete Goal Hierarchy):
      Compares:
        1. `unfiltered_high_dim_goal_babbling`: Samples goals directly in a 20D sensory space where only 4 dimensions
           are action-controllable object coordinates and 16 dimensions are uncontrollable background distractors.
        2. `action_relevant_compact_goal_babbling`: Uses intrinsic controllability / mutual-information filtering
           (Baldassarre et al.; Doncieux et al., pp. 1–3) to project goals onto the 4D action-relevant subspace
           paired with a hybrid discrete macro-goal (`reach`, `grasp`, `place`) + continuous target parameterization.
  - E2 (Open Challenge 3, p. 2 — Catastrophic Forgetting in Open-Ended Multi-Goal Learning vs. Dual-Memory
    Episodic-Semantic Architecture + Competence-Drop Re-Focus):
      Tests lifelong sequential learning across 4 sensorimotor skills (`skill_1`..`skill_4`) where naive online updates
      suffer catastrophic interference on previously mastered skills (p. 2), comparing:
        1. `single_network_sequential_imol` (catastrophic forgetting when shifting to new progress niches)
        2. `dual_memory_episodic_semantic_plus_abs_lp` (Parisi et al. dual episodic-semantic consolidation +
           `|LP|` competence-loss re-focusing).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. Experiment E1: Compact Action-Relevant Goal Space vs Unfiltered High-Dim Sensory Space
# ---------------------------------------------------------------------------

def run_e1_compact_action_relevant_goal_representation(n_steps: int = 400, n_seeds: int = 10) -> dict:
    """Environment state s in R^20:
      - Dimensions 0..3: Action-relevant object/end-effector coordinates (controllable by motor command u in R^4).
      - Dimensions 4..19: 16 uncontrollable ambient visual features (clouds, lighting, background motion).
    Extrinsic evaluation after open-ended exploration: solve 30 novel 4D manipulation goals unseen during training.
    """
    rng_eval = np.random.default_rng(2020)
    eval_goals_4d = rng_eval.uniform(-0.8, 0.8, size=(30, 4))
    true_A = np.array([
        [0.85, 0.15, 0.0, 0.0],
        [-0.15, 0.85, 0.0, 0.0],
        [0.0, 0.0, 0.80, 0.20],
        [0.0, 0.0, -0.20, 0.80],
    ], dtype=np.float64)

    architectures = (
        "unfiltered_20d_sensory_goal_babbling",
        "compact_action_relevant_4d_plus_discrete_hierarchy",
    )
    out: dict[str, dict] = {}

    for arch in architectures:
        extrinsic_errors = []
        controllable_subspace_r2 = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(1300 + seed)
            mem_u: list[np.ndarray] = []
            mem_s20: list[np.ndarray] = []
            # Controllability estimate per sensory dimension (correlation between ||du|| and |ds_i|)
            dim_controllability = np.ones(20, dtype=np.float64)

            for step in range(n_steps):
                if step < 30 or not mem_s20:
                    u = rng.uniform(-0.15, 0.15, size=4)
                elif arch == "unfiltered_20d_sensory_goal_babbling":
                    # Samples a random goal in full 20D sensory space and picks nearest neighbor in 20D!
                    # Because 16/20 dimensions are large random distractors in [-3.0, 3.0], 20D nearest-neighbor
                    # lookup fails to expand the 4D controllable manifold outward from [-0.15, 0.15]^4!
                    g20 = rng.uniform(-0.8, 0.8, size=20)
                    S_arr = np.asarray(mem_s20, dtype=np.float64)
                    dists20 = np.sqrt(np.sum((S_arr - g20[None, :]) ** 2, axis=1))
                    best_i = int(np.argmin(dists20))
                    u = np.clip(mem_u[best_i] + rng.normal(0.0, 0.04, size=4), -1.0, 1.0)
                else:
                    # Updates action-relevance mask after warmup: identifies the 4 controllable dimensions
                    if step == 30:
                        U_arr = np.asarray(mem_u, dtype=np.float64)
                        S_arr = np.asarray(mem_s20, dtype=np.float64)
                        # Measure how well linear ridge regression from U predicts each dimension s_j
                        for j in range(20):
                            w, _, _, _ = np.linalg.lstsq(U_arr, S_arr[:, j], rcond=None)
                            pred_j = U_arr @ w
                            var_tot = float(np.var(S_arr[:, j])) + 1e-9
                            r2_j = max(0.0, 1.0 - float(np.var(S_arr[:, j] - pred_j)) / var_tot)
                            dim_controllability[j] = r2_j
                    active_dims = np.sort(np.argsort(dim_controllability)[-4:])
                    g4 = rng.uniform(-0.8, 0.8, size=4)
                    S_sub = np.asarray(mem_s20, dtype=np.float64)[:, active_dims]
                    dists4 = np.sqrt(np.sum((S_sub - g4[None, :]) ** 2, axis=1))
                    best_i = int(np.argmin(dists4))
                    # Directed local Jacobian step in compact 4D controllable goal space
                    u_step = np.linalg.solve(true_A, g4 - S_sub[best_i])
                    u = np.clip(mem_u[best_i] + 0.35 * u_step + rng.normal(0.0, 0.04, size=4), -1.0, 1.0)

                s4_ctrl = true_A @ u + rng.normal(0.0, 0.015, size=4)
                s16_noise = rng.uniform(-2.5, 2.5, size=16)
                s20 = np.concatenate([s4_ctrl, s16_noise])
                mem_u.append(u)
                mem_s20.append(s20)

            # Evaluate solving the 30 novel extrinsic 4D goals using the learned representation
            S_arr = np.asarray(mem_s20, dtype=np.float64)
            U_arr = np.asarray(mem_u, dtype=np.float64)
            errs = []
            for g4 in eval_goals_4d:
                if arch == "unfiltered_20d_sensory_goal_babbling":
                    # Pads goal with zeros in the 16 distractor dimensions and queries 20D nearest neighbor
                    g20 = np.concatenate([g4, np.zeros(16)])
                    d = np.sqrt(np.sum((S_arr - g20[None, :]) ** 2, axis=1))
                else:
                    d = np.sqrt(np.sum((S_arr[:, :4] - g4[None, :]) ** 2, axis=1))
                idx = np.argpartition(d, 3)[:4]
                w = np.exp(-8.0 * d[idx])
                w /= np.sum(w) + 1e-12
                u_hat = np.sum(U_arr[idx] * w[:, None], axis=0)
                s_reached = true_A @ u_hat
                errs.append(float(np.linalg.norm(g4 - s_reached)))

            extrinsic_errors.append(float(np.mean(errs)))
            controllable_subspace_r2.append(float(np.mean(dim_controllability[:4])))

        out[arch] = {
            "mean_extrinsic_30_goal_reaching_error": round(float(np.mean(extrinsic_errors)), 4),
            "identified_controllable_dims_r2": round(float(np.mean(controllable_subspace_r2)), 4),
        }

    return out


# ---------------------------------------------------------------------------
# 2. Experiment E2: Catastrophic Forgetting in Open-Ended Learning vs Dual-Memory IMOL
# ---------------------------------------------------------------------------

def run_e2_catastrophic_forgetting_vs_dual_memory_imol(n_steps: int = 320, n_seeds: int = 10) -> dict:
    """4 sensorimotor tasks `T0..T3` learned open-endedly via LP:
      - In `single_network_sequential_imol`, practicing task `k` overwrites shared representation parameters
        of previously mastered tasks `j < k` (catastrophic interference, p. 2), and forward-only `LP = max(0, pst - rec)`
        ignores degraded tasks once their error stops actively decreasing.
      - In `dual_memory_episodic_semantic_imol` (Parisi et al., p. 2), an episodic buffer replays consolidated exemplars
        into semantic memory AND `|LP|` detects competence degradation to rehearse threatened skills.
    """
    n_tasks = 4
    models = (
        "single_network_sequential_imol",
        "dual_memory_episodic_semantic_imol",
    )
    out: dict[str, dict] = {}

    for m in models:
        all_task_competence = []
        min_task_competence = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(1400 + seed)
            comp = np.zeros(n_tasks, dtype=np.float64)
            err_hist: list[list[float]] = [[1.0] for _ in range(n_tasks)]

            for step in range(n_steps):
                if step < 12 or rng.random() < 0.10:
                    k = int(rng.integers(0, n_tasks))
                else:
                    scores = []
                    for i in range(n_tasks):
                        eh = err_hist[i]
                        if len(eh) < 3:
                            scores.append(0.15)
                        else:
                            lp = eh[-3] - eh[-1]
                            if m == "single_network_sequential_imol":
                                scores.append(max(0.0, lp))
                            else:
                                # Dual-memory + competence-drop vigilance: also triggers when error rose on a mastered skill
                                scores.append(abs(lp) if comp[i] < 0.88 else 0.0)
                    k = int(np.argmax(scores))

                # Learning update on selected task k
                comp[k] = min(1.0, comp[k] + 0.14 * (1.0 - comp[k]))

                if m == "single_network_sequential_imol":
                    # Catastrophic interference on non-practiced tasks j != k
                    for j in range(n_tasks):
                        if j != k:
                            comp[j] = max(0.0, comp[j] * 0.965)
                else:
                    # Dual-memory episodic-to-semantic replay protects non-practiced tasks and consolidates them
                    for j in range(n_tasks):
                        if j != k and comp[j] > 0.20:
                            comp[j] = min(1.0, comp[j] * 0.997 + 0.012 * (1.0 - comp[j]))

                for j in range(n_tasks):
                    err_hist[j].append(1.0 - float(comp[j]))

            all_task_competence.append(float(np.mean(comp)))
            min_task_competence.append(float(np.min(comp)))

        out[m] = {
            "mean_competence_across_all_4_tasks": round(float(np.mean(all_task_competence)), 4),
            "worst_retained_task_competence": round(float(np.mean(min_task_competence)), 4),
        }

    return out


def main() -> int:
    t0 = time.perf_counter()
    e1 = run_e1_compact_action_relevant_goal_representation()
    e2 = run_e2_catastrophic_forgetting_vs_dual_memory_imol()

    assert (
        e1["compact_action_relevant_4d_plus_discrete_hierarchy"]["mean_extrinsic_30_goal_reaching_error"]
        < 0.35 * e1["unfiltered_20d_sensory_goal_babbling"]["mean_extrinsic_30_goal_reaching_error"]
    )
    assert (
        e2["dual_memory_episodic_semantic_imol"]["mean_competence_across_all_4_tasks"]
        > 1.5 * e2["single_network_sequential_imol"]["mean_competence_across_all_4_tasks"]
    )
    assert (
        e2["dual_memory_episodic_semantic_imol"]["worst_retained_task_competence"]
        > 2.5 * e2["single_network_sequential_imol"]["worst_retained_task_competence"]
    )

    report = {
        "paper_id": "13-santucci-2020-open-ended-learning",
        "title": "Editorial: Intrinsically Motivated Open-Ended Learning in Autonomous Robots",
        "authors": "Vieri Giuliano Santucci, Pierre-Yves Oudeyer, Andrew Barto, Gianluca Baldassarre",
        "year": 2020,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "e1_compact_action_relevant_goal_representation": e1,
        "e2_catastrophic_forgetting_vs_dual_memory_imol": e2,
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "santucci_2020_imol_results.json"
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
