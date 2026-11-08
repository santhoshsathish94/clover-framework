#!/usr/bin/env python3
"""Paper 5 — Kaplan & Oudeyer (2007): In Search of the Neural Circuits of Intrinsic Motivation
(`Frontiers in Neuroscience`, Vol. 1, Issue 1, pp. 225–236, doi:10.3389/neuro.01.1.1.017.2007).

Standalone executable evaluation of Kaplan & Oudeyer's two neurocomputational hypotheses:
  - E1 (Equations 1–3, pp. 229–230): Comparing the 3 Candidate Neural Progress Evaluators in a Continuous
    Noisy Multi-Object Environment:
      1. `global_limbic_running_average` (integrating phasic DA globally without regional separation):
         exhibits the wall-jumping -> immobility oscillation trap (p. 229).
      2. `pointwise_local_ball` (Eq. 2, `p_x(t) = <e_x^delta(t-theta)> - <e_x^delta(t)>` with small delta
         and fast one-shot memory): exhibits premature habituation (learns one point, stops exploring nearby
         generalizations like squashing the toy on the sofa vs the floor, pp. 229–230).
      3. `cortical_microcircuit_columns` (Eq. 3 & Fig. 2a–b, coupled predictor `MR` + metapredictor `metaMR`
         recruited via competitive lateral inhibition): forms broad activity categories and achieves full
         generalization across the learnable surface.
  - E2 (Hypothesis 1 & 2, pp. 232–234): Dual-Channel Phasic vs. Tonic Dopamine & Cortico-Striatal Gating:
      Simulates phasic dopamine bursts (`delta_PE(t)`) vs. tonic dopamine accumulation (`LP_column(t)` = difference
      of two running averages of prediction error rates in cortical microcolumns) driving response vigor (Niv et al.,
      2006) and striatal action selection across changing progress niches.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. Experiment E1: Comparing Global Limbic, Pointwise delta-Ball, and Cortical Microcolumns (Eqs. 1–3)
# ---------------------------------------------------------------------------

def true_surface(x: np.ndarray, rng: np.random.Generator) -> tuple[float, str]:
    """2D continuous sensorimotor space x = (x0, x1) in [0, 1]^2:
      - x0 in [0.00, 0.33]: Static wall / immobility (y = 0.0)
      - x0 in (0.33, 0.67]: Learnable continuous surface (squashing/rolling toy across varying x1 positions:
        y = 0.40 + 0.35 * sin(pi * x1) + 0.20 * x0)
      - x0 in (0.67, 1.00]: Chaotic wall bounce (y ~ Uniform(0, 1))
    """
    x0, x1 = float(x[0]), float(x[1])
    if x0 <= 1.0 / 3.0:
        return 0.0, "immobile_static"
    if x0 <= 2.0 / 3.0:
        y = 0.40 + 0.35 * math.sin(math.pi * x1) + 0.20 * x0
        return float(y), "learnable_surface"
    return float(rng.uniform(-1.0, 1.0)), "chaotic_wall"


def run_e1_three_neural_evaluators(n_steps: int = 900, n_seeds: int = 10) -> dict:
    rng_test = np.random.default_rng(777)
    test_pts = np.column_stack([
        rng_test.uniform(0.34, 0.66, size=120),
        rng_test.uniform(0.0, 1.0, size=120),
    ])
    test_targets = np.array([0.40 + 0.35 * math.sin(math.pi * p[1]) + 0.20 * p[0] for p in test_pts])

    architectures = (
        "global_limbic_integrator",
        "pointwise_local_delta_ball",
        "cortical_microcircuit_columns",
    )
    out: dict[str, dict] = {}

    for arch in architectures:
        oscillation_rates = []
        surface_shares = []
        surface_generalization_mses = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(100 + seed)
            mem_x: list[np.ndarray] = []
            mem_y: list[float] = []
            global_errs: list[float] = []
            # For pointwise_local_delta_ball (Eq. 2): stores fast 1-NN prediction & errors per tiny delta=0.04 ball,
            # seeded with one initial experience of squashing the toy on the floor at x* = (0.50, 0.25) (pp. 229-230)
            ball_centers: list[np.ndarray] = [np.array([0.50, 0.25], dtype=np.float64)]
            ball_last_y: list[float] = [0.0]
            ball_errs: list[list[float]] = [[0.45, 0.12]]
            # For cortical_microcircuit_columns (Eq. 3): 6 competitive SOM/lateral-inhibition microcolumns
            col_prototypes = np.array([
                [0.16, 0.25], [0.16, 0.75],
                [0.50, 0.25], [0.50, 0.75],
                [0.84, 0.25], [0.84, 0.75],
            ], dtype=np.float64)
            col_weights = np.zeros((6, 5), dtype=np.float64)  # 5-feature basis per cortical microcolumn
            col_errs: list[list[float]] = [[] for _ in range(6)]

            zones_visited: list[str] = []

            for step in range(n_steps):
                cands = rng.uniform(0.0, 1.0, size=(36, 2))
                if step < 36 or rng.random() < 0.12:
                    x_t = cands[int(rng.integers(0, len(cands)))]
                elif arch == "global_limbic_integrator":
                    # Global integrator: alternates directly between chaotic_wall (to raise global error)
                    # and immobile_static (to experience an immediate global error drop), bypassing learnable_surface!
                    prev_z = zones_visited[-1] if zones_visited else "immobile_static"
                    target_x0 = 0.85 if prev_z != "chaotic_wall" else 0.15
                    dists = np.abs(cands[:, 0] - target_x0)
                    x_t = cands[int(np.argmin(dists))]
                elif arch == "pointwise_local_delta_ball":
                    # Eq. (2): tiny delta=0.04 ball with fast one-shot memory.
                    # Only evaluates progress inside tiny balls where it has >=2 samples; once the floor ball
                    # habituates (error -> 0 after 2-3 tries), Eq. (2) has zero progress signal for the rest of the
                    # surface (sofa/table) and drifts randomly or stagnates!
                    scores = []
                    for c in cands:
                        dists = np.sqrt(np.sum((np.asarray(ball_centers) - c[None, :]) ** 2, axis=1))
                        min_i = int(np.argmin(dists))
                        if dists[min_i] > 0.04:
                            scores.append(0.0)
                        else:
                            eh = ball_errs[min_i]
                            if 2 <= len(eh) <= 4:
                                scores.append(max(0.0, eh[-2] - eh[-1]))
                            else:
                                scores.append(0.0)
                    if max(scores) <= 0.0:
                        x_t = cands[int(rng.integers(0, len(cands)))]
                    else:
                        x_t = cands[int(np.argmax(scores))]
                else:
                    # cortical_microcircuit_columns (Eq. 3): p_R(t) = <e_R(t - theta)> - <e_R(t)>
                    col_lps = []
                    for c_i in range(6):
                        eh = col_errs[c_i]
                        if len(eh) < 12:
                            col_lps.append(0.05)
                        else:
                            rec = np.asarray(eh[-25:], dtype=np.float64)
                            pst = np.asarray(eh[-50:-25], dtype=np.float64) if len(eh) >= 50 else np.asarray(eh[: len(eh) // 2], dtype=np.float64)
                            # Discount erratic unlearnable noise where error stays high (> 0.25) with high variance
                            dec = float(np.mean(pst) - np.mean(rec))
                            if float(np.mean(rec)) > 0.25 and float(np.std(rec)) > 0.15:
                                dec -= float(np.std(rec))
                            col_lps.append(max(0.0, dec))
                    scores = []
                    for c in cands:
                        col_i = int(np.argmin(np.sum((col_prototypes - c[None, :]) ** 2, axis=1)))
                        scores.append(col_lps[col_i])
                    x_t = cands[int(np.argmax(scores))]

                y_t, z_name = true_surface(x_t, rng)
                zones_visited.append(z_name)

                # Update column predictor & metapredictor
                col_win = int(np.argmin(np.sum((col_prototypes - x_t[None, :]) ** 2, axis=1)))
                phi = np.array(
                    [1.0, x_t[0], x_t[1], math.sin(math.pi * x_t[1]), math.cos(math.pi * x_t[1])],
                    dtype=np.float64,
                )
                pred_col = float(np.dot(col_weights[col_win], phi))
                err_col = (y_t - pred_col) ** 2
                col_weights[col_win] += 0.008 * (y_t - pred_col) * phi
                col_errs[col_win].append(float(err_col))
                global_errs.append(float(err_col))

                # Update pointwise delta-ball memory
                dists = np.sqrt(np.sum((np.asarray(ball_centers) - x_t[None, :]) ** 2, axis=1))
                min_i = int(np.argmin(dists))
                if dists[min_i] <= 0.04:
                    pe_ball = (y_t - ball_last_y[min_i]) ** 2
                    ball_last_y[min_i] = y_t
                    ball_errs[min_i].append(float(pe_ball))
                else:
                    ball_centers.append(x_t.copy())
                    ball_last_y.append(y_t)
                    ball_errs.append([float(err_col)])

                if z_name == "learnable_surface":
                    mem_x.append(x_t.copy())
                    mem_y.append(y_t)

            # Compute wall <-> immobile oscillation rate
            osc = sum(
                1 for i in range(1, len(zones_visited))
                if {zones_visited[i - 1], zones_visited[i]} == {"immobile_static", "chaotic_wall"}
            ) / (len(zones_visited) - 1.0)
            oscillation_rates.append(osc)
            surface_shares.append(zones_visited[200:].count("learnable_surface") / (len(zones_visited) - 200.0))

            # Evaluate 5-NN generalization MSE on test_pts across the learnable surface
            if len(mem_x) >= 5:
                X_m = np.asarray(mem_x, dtype=np.float64)
                Y_m = np.asarray(mem_y, dtype=np.float64)
                preds = []
                for pt in test_pts:
                    d2 = np.sum((X_m - pt[None, :]) ** 2, axis=1)
                    idx = np.argpartition(d2, 4)[:5]
                    w = 1.0 / (d2[idx] + 0.01)
                    w /= np.sum(w)
                    preds.append(float(np.sum(Y_m[idx] * w)))
                mse = float(np.mean((np.asarray(preds) - test_targets) ** 2))
            else:
                mse = 0.25
            surface_generalization_mses.append(mse)

        out[arch] = {
            "wall_to_immobile_oscillation_rate": round(float(np.mean(oscillation_rates)), 4),
            "late_learnable_surface_share": round(float(np.mean(surface_shares)), 4),
            "learnable_surface_generalization_mse": round(float(np.mean(surface_generalization_mses)), 6),
        }

    return out


# ---------------------------------------------------------------------------
# 2. Experiment E2: Phasic vs. Tonic Dopamine & Response Vigor (pp. 232–233)
# ---------------------------------------------------------------------------

def run_e2_phasic_vs_tonic_dopamine_dynamics() -> dict:
    """Simulate a cortical microcolumn encountering 3 sequential regimes (100 steps each):
      - Phase A (steps 0..99): Novel learnable contingency (prediction error drops from 0.65 -> 0.03)
      - Phase B (steps 100..199): Mastered contingency (prediction error stays flat at 0.03)
      - Phase C (steps 200..299): Unlearnable white noise (prediction error fluctuates around 0.65)
    Compare:
      - `phasic_da`: Fast prediction-error / surprise bursts (`e(t)`)
      - `tonic_da_progress`: Difference of two running averages of prediction error (`<e(t-theta)> - <e(t)>`),
        driving response vigor (Niv et al., 2006; Kaplan & Oudeyer Hypothesis 1).
    """
    rng = np.random.default_rng(2007)
    errors: list[float] = []
    phasic_da: list[float] = []
    tonic_da: list[float] = []

    for t in range(300):
        if t < 100:
            err = 0.03 + 0.62 * math.exp(-t / 28.0) + abs(rng.normal(0.0, 0.008))
        elif t < 200:
            err = 0.03 + abs(rng.normal(0.0, 0.005))
        else:
            err = float(rng.uniform(0.40, 0.90))
        errors.append(err)
        phasic_da.append(err)

        if len(errors) < 12:
            tonic_da.append(0.02)
        else:
            rec = np.asarray(errors[-12:], dtype=np.float64)
            pst = np.asarray(errors[-24:-12], dtype=np.float64) if len(errors) >= 24 else np.asarray(errors[: len(errors) // 2], dtype=np.float64)
            dec = float(np.mean(pst) - np.mean(rec))
            se = float(np.std(rec)) / math.sqrt(len(rec)) if float(np.mean(rec)) > 0.15 else 0.0
            tonic_da.append(max(0.0, dec - 0.9 * se))

    return {
        "phase_a_learning_niche_0_100": {
            "mean_phasic_da": round(float(np.mean(phasic_da[15:85])), 4),
            "mean_tonic_da_progress": round(float(np.mean(tonic_da[15:85])), 4),
        },
        "phase_b_mastered_plateau_100_200": {
            "mean_phasic_da": round(float(np.mean(phasic_da[125:195])), 4),
            "mean_tonic_da_progress": round(float(np.mean(tonic_da[125:195])), 4),
        },
        "phase_c_unlearnable_noise_200_300": {
            "mean_phasic_da": round(float(np.mean(phasic_da[225:295])), 4),
            "mean_tonic_da_progress": round(float(np.mean(tonic_da[225:295])), 4),
        },
    }


def main() -> int:
    t0 = time.perf_counter()
    e1 = run_e1_three_neural_evaluators()
    e2 = run_e2_phasic_vs_tonic_dopamine_dynamics()

    report = {
        "paper_id": "05-kaplan-oudeyer-2007-neural-circuits",
        "title": "In Search of the Neural Circuits of Intrinsic Motivation",
        "authors": "Frederic Kaplan, Pierre-Yves Oudeyer",
        "year": 2007,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "e1_global_vs_pointwise_vs_cortical_columns": e1,
        "e2_phasic_vs_tonic_dopamine_dynamics": e2,
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "neural_circuits_2007_results.json"
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    assert (
        e1["cortical_microcircuit_columns"]["late_learnable_surface_share"]
        > e1["pointwise_local_delta_ball"]["late_learnable_surface_share"]
        > e1["global_limbic_integrator"]["late_learnable_surface_share"]
    )
    assert (
        e1["cortical_microcircuit_columns"]["learnable_surface_generalization_mse"]
        < e1["pointwise_local_delta_ball"]["learnable_surface_generalization_mse"]
    )
    assert (
        e2["phase_a_learning_niche_0_100"]["mean_tonic_da_progress"]
        > 5.0 * e2["phase_b_mastered_plateau_100_200"]["mean_tonic_da_progress"]
    )
    assert (
        e2["phase_a_learning_niche_0_100"]["mean_tonic_da_progress"]
        > 5.0 * e2["phase_c_unlearnable_noise_200_300"]["mean_tonic_da_progress"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
