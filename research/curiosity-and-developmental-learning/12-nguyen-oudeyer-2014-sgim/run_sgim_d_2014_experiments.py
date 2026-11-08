#!/usr/bin/env python3
"""Paper 12 — Nguyen & Oudeyer (2014): Socially Guided Intrinsic Motivation for Robot Learning of Motor Skills
(`Autonomous Robots`, Vol. 36, No. 3, pp. 273–294, doi:10.1007/s10514-013-9339-y).

Standalone executable evaluation of the `SGIM-D` architecture on the 25D-to-2D Fishing Rod Inverse Model task:
  - E1 (Sections 3–6, Figures 1, 7a & 8a — Comparing the 5 Exploration Algorithms on the 25D-to-2D Fishing Task):
      Models a 6-DOF arm parameterized by `theta in [0, 1]^25` (4 Gaussian trajectory parameters per joint + duration
      delta, Eq. 5) casting a flexible fishing line to a 2D water-surface outcome `tau in [-1, 1]^2` with
      action-dependent heteroscedastic noise (Fig. 6) and a hard-to-discover far-cast basin.
      Compares all 5 algorithms from Section 5.1 (Fig. 7a):
        1. `learning_from_observation` (only observes teacher demonstrations, affected by correspondence error Eq. 6)
        2. `random_exploration` (uniform random `theta in [0, 1]^25`)
        3. `sagg_riac` (autonomous competence-progress goal babbling, Algorithm 3.2)
        4. `imitation_learning` (local perturbation `theta_d + theta_rand` around demonstrations)
        5. `sgim_d` (Algorithm 3.1: interleaves short imitation bursts + goal emulation with `SAGG-RIAC` autonomous
           goal babbling and reliability-weighted local inverse optimization `dist(tau_h, tau_g) + alpha * var_h`).
  - E2 (Section 5.3, Figure 7c — Comparing the 3 Demonstrator Types in `SGIM-D`):
      Compares `SGIM-D` taught by:
        - `demonstrator_1_naive_sagg_riac` (randomly chosen exemplars including high-variance unstable whip-casts)
        - `demonstrator_2_minimum_variance_sagg_riac` (reliability-filtered low-variance exemplars, Fig. 6)
        - `demonstrator_3_human_kinesthetic` (smooth human trajectories with small correspondence mapping error Eq. 6).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. 25D Motor Primitive -> 2D Water Surface Fishing Environment (Section 4)
# ---------------------------------------------------------------------------

def fishing_forward_map(theta: np.ndarray, rng: np.random.Generator, noiseless: bool = False) -> tuple[np.ndarray, float]:
    """Map 25D motor parameter `theta in [0, 1]^25` to 2D float landing position `tau in [-1, 1]^2`.
    - Near/mid water zone (`tau_0 in [-0.7, 0.3]`): reachable by broad manifold of parameters.
    - Far-cast outer ring (`tau_0 > 0.35`): requires coordinated whip timing across joints 0..3 (`theta[0:8] > 0.68`),
      which has very low probability under uniform random exploration in `[0, 1]^25`, but is demonstrated by the teacher!
    - Action-dependent noise std (Fig. 6): high jerk (`std(theta[8:24]) > 0.32`) amplifies landing noise by 4x.
    """
    th = np.clip(theta, 0.0, 1.0)
    # Coordinated cast power from first 8 parameters
    whip_coord = float(np.mean(th[0:8]))
    angle_steer = float(np.mean(th[8:16]) - np.mean(th[16:24]))
    duration_delta = float(th[24])

    # Nonlinear reach radius: jumps into the far-cast basin when whip_coord > 0.65 and duration_delta > 0.55
    if whip_coord > 0.65 and duration_delta > 0.55:
        radius = 0.45 + 0.50 * ((whip_coord - 0.65) / 0.35)
    else:
        radius = 0.55 * whip_coord

    phi = math.pi * angle_steer
    tau_x = float(np.clip(-0.20 + radius * math.cos(phi), -0.95, 0.95))
    tau_y = float(np.clip(radius * math.sin(phi), -0.95, 0.95))

    # Action-dependent variance (Section 4.4.2 & Fig. 6)
    jerk = float(np.std(th[0:24]))
    noise_std = 0.012 + (0.085 if jerk > 0.30 else 0.008)
    if not noiseless:
        tau_x = float(np.clip(tau_x + rng.normal(0.0, noise_std), -1.0, 1.0))
        tau_y = float(np.clip(tau_y + rng.normal(0.0, noise_std), -1.0, 1.0))

    return np.array([tau_x, tau_y], dtype=np.float64), noise_std


def make_benchmark_and_teacher_sets() -> tuple[np.ndarray, dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    rng = np.random.default_rng(201412)
    # Generate benchmark grid of 120 reachable target points spanning both near/mid zone and far-cast zone
    bench_pts = []
    for r in np.linspace(0.08, 0.88, 8):
        for ang in np.linspace(-0.85 * math.pi, 0.85 * math.pi, 15):
            tx = float(np.clip(-0.20 + r * math.cos(ang), -0.90, 0.90))
            ty = float(np.clip(r * math.sin(ang), -0.90, 0.90))
            bench_pts.append([tx, ty])
    benchmark = np.asarray(bench_pts, dtype=np.float64)

    # Generate 3 teacher demonstration sets (25 demonstrations each, covering near and far-cast zones)
    demos: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "demonstrator_1_naive_high_variance": [],
        "demonstrator_2_reliable_low_variance": [],
        "demonstrator_3_human_kinesthetic": [],
    }
    for i in range(15):
        # Teacher demonstrations cover a specialized subset of the task space (upper/right sector st in [-0.15, 0.70],
        # Section 1.3 p. 3: "exploration guided by a human teacher tends to be more specialised and resulting in fewer
        # tasks that are learnt faster, combining both can bring out a system that acquires a diversity of skills")
        wc = 0.35 + 0.57 * (i / 14.0)
        st = -0.15 + 0.85 * ((i * 5) % 15) / 14.0

        # Demonstrator 2: smooth low-jerk parameters (low variance)
        th_low_var = np.zeros(25, dtype=np.float64)
        th_low_var[0:8] = wc
        th_low_var[8:16] = np.clip(0.50 + 0.5 * st, 0.0, 1.0)
        th_low_var[16:24] = np.clip(0.50 - 0.5 * st, 0.0, 1.0)
        th_low_var[24] = 0.75 if wc > 0.62 else 0.45
        tau_2, _ = fishing_forward_map(th_low_var, rng, noiseless=True)
        demos["demonstrator_2_reliable_low_variance"].append((th_low_var, tau_2))

        # Demonstrator 1: same mean reach, but alternating high-jerk redundant parameters (high variance!)
        th_high_var = th_low_var.copy()
        th_high_var[0:8:2] = np.clip(wc + 0.35, 0.0, 1.0)
        th_high_var[1:8:2] = np.clip(wc - 0.35, 0.0, 1.0)
        tau_1, _ = fishing_forward_map(th_high_var, rng, noiseless=False)
        demos["demonstrator_1_naive_high_variance"].append((th_high_var, tau_1))

        # Demonstrator 3 (Human): smooth parameters with small correspondence mapping error (Eq. 6, Fig. 5)
        th_hum = np.clip(th_low_var - 0.03, 0.0, 1.0)
        th_hum[8:24] = th_low_var[8:24]
        th_hum[24] = th_low_var[24]
        tau_3, _ = fishing_forward_map(th_low_var, rng, noiseless=True)
        demos["demonstrator_3_human_kinesthetic"].append((th_hum, tau_3))

    return benchmark, demos


def evaluate_inverse_model(
    mem_theta: list[np.ndarray],
    mem_tau: list[np.ndarray],
    mem_var: list[float],
    benchmark: np.ndarray,
    rng: np.random.Generator,
    use_reliability_weighting: bool = True,
) -> float:
    """Evaluate mean Euclidean distance error across all benchmark targets (Eq. 10 & Algorithm 4.6)."""
    if not mem_tau:
        return 1.0
    T_arr = np.asarray(mem_tau, dtype=np.float64)
    Th_arr = np.asarray(mem_theta, dtype=np.float64)
    V_arr = np.asarray(mem_var, dtype=np.float64)

    errs = []
    for goal in benchmark:
        dists = np.sqrt(np.sum((T_arr - goal[None, :]) ** 2, axis=1))
        if use_reliability_weighting:
            # Section 4.4.2: reliability score = dist(tau_h, tau_g) + alpha * var_h (alpha = 0.5)
            scores = dists + 0.5 * V_arr
        else:
            scores = dists
        k = min(4, len(scores))
        idx = np.argpartition(scores, k - 1)[:k]
        w = np.exp(-12.0 * scores[idx])
        w /= np.sum(w) + 1e-12
        theta_g = np.sum(Th_arr[idx] * w[:, None], axis=0)
        tau_reached, _ = fishing_forward_map(theta_g, rng, noiseless=False)
        errs.append(float(np.linalg.norm(goal - tau_reached)))

    return float(np.mean(errs))


def run_e1_five_algorithms_comparison(n_trials: int = 450, n_seeds: int = 8) -> dict:
    benchmark, demos_dict = make_benchmark_and_teacher_sets()
    teacher_demos = demos_dict["demonstrator_3_human_kinesthetic"]

    algos = (
        "learning_from_observation",
        "random_exploration",
        "sagg_riac",
        "imitation_learning",
        "sgim_d",
    )
    out: dict[str, dict] = {}

    for alg in algos:
        final_errors = []
        early_errors_150 = []
        far_cast_shares = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(1200 + seed)
            mem_theta: list[np.ndarray] = []
            mem_tau: list[np.ndarray] = []
            mem_var: list[float] = []

            # 4 task-space regions for SAGG-RIAC / SGIM-D interest tracking:
            # R0: near-left, R1: near-right, R2: mid-water, R3: far-cast outer ring (tau_x > 0.25)
            reg_errs: list[list[float]] = [[] for _ in range(4)]
            err_at_150 = 1.0

            for t in range(n_trials):
                if alg == "learning_from_observation":
                    # Only observes a sparse subset of teacher demonstrations without self-execution;
                    # suffers from both small sample size (15 samples) and uncalibrated correspondence error (Section 6.1)
                    if t % 90 == 0:
                        th_d, tau_d = teacher_demos[(t // 90) % 5]
                        mem_theta.append(np.clip(th_d - 0.25, 0.0, 1.0))
                        mem_tau.append(tau_d.copy())
                        mem_var.append(0.05)
                elif alg == "random_exploration":
                    th = rng.uniform(0.0, 1.0, size=25)
                    tau, n_std = fishing_forward_map(th, rng)
                    mem_theta.append(th)
                    mem_tau.append(tau)
                    mem_var.append(n_std)
                elif alg == "imitation_learning":
                    # Repeats local variations around the latest teacher demonstration (changed every 30 steps)
                    th_d, _ = teacher_demos[(t // 30) % len(teacher_demos)]
                    th = np.clip(th_d + rng.uniform(-0.08, 0.08, size=25), 0.0, 1.0)
                    tau, n_std = fishing_forward_map(th, rng)
                    mem_theta.append(th)
                    mem_tau.append(tau)
                    mem_var.append(n_std)
                elif alg in ("sagg_riac", "sgim_d"):
                    # In SGIM-D, every 30 steps the teacher gives a demonstration -> short imitation burst of 3 steps
                    if alg == "sgim_d" and (t % 30) < 3:
                        th_d, tau_d = teacher_demos[(t // 30) % len(teacher_demos)]
                        # SGIM-D both imitates policy locally AND emulates the demonstrated goal tau_d (Algo 3.1 lines 9-10)
                        th = np.clip(th_d + rng.uniform(-0.04, 0.06, size=25), 0.0, 1.0)
                        tau, n_std = fishing_forward_map(th, rng)
                        mem_theta.append(th)
                        mem_tau.append(tau)
                        mem_var.append(n_std)
                        r_idx = 3 if tau_d[0] > 0.25 else int(rng.integers(0, 3))
                        reg_errs[r_idx].append(float(np.linalg.norm(tau_d - tau)))
                    else:
                        # Autonomous goal babbling (SAGG-RIAC): select region proportional to competence progress (Eq. 3)
                        interests = []
                        for r_i in range(4):
                            eh = reg_errs[r_i]
                            if len(eh) < 4:
                                interests.append(0.10)
                            else:
                                rec = float(np.mean(eh[-4:]))
                                pst = float(np.mean(eh[-8:-4])) if len(eh) >= 8 else float(np.mean(eh[: len(eh) // 2]))
                                interests.append(abs(pst - rec) + 0.01)
                        if rng.random() < 0.20:
                            r_sel = int(rng.integers(0, 4))
                        else:
                            r_sel = int(np.argmax(interests))

                        # Sample goal tau_g inside selected region
                        if r_sel == 0:
                            tau_g = np.array([rng.uniform(-0.6, -0.1), rng.uniform(-0.6, 0.0)], dtype=np.float64)
                        elif r_sel == 1:
                            tau_g = np.array([rng.uniform(-0.6, -0.1), rng.uniform(0.0, 0.6)], dtype=np.float64)
                        elif r_sel == 2:
                            tau_g = np.array([rng.uniform(-0.1, 0.25), rng.uniform(-0.6, 0.6)], dtype=np.float64)
                        else:
                            tau_g = np.array([rng.uniform(0.25, 0.75), rng.uniform(-0.6, 0.6)], dtype=np.float64)

                        # Goal-Directed Policy Optimization (Algorithm 4.5):
                        if not mem_tau:
                            th = rng.uniform(0.0, 1.0, size=25)
                        else:
                            T_arr = np.asarray(mem_tau, dtype=np.float64)
                            dists = np.sqrt(np.sum((T_arr - tau_g[None, :]) ** 2, axis=1))
                            min_i = int(np.argmin(dists))
                            # Global exploration if closest point is far; local optimization if close
                            if dists[min_i] > 0.35 and alg == "sagg_riac":
                                th = rng.uniform(0.0, 1.0, size=25)
                            else:
                                V_arr = np.asarray(mem_var, dtype=np.float64)
                                scores = dists + 0.5 * V_arr
                                best_i = int(np.argmin(scores))
                                th = np.clip(mem_theta[best_i] + rng.normal(0.0, 0.06, size=25), 0.0, 1.0)

                        tau, n_std = fishing_forward_map(th, rng)
                        mem_theta.append(th)
                        mem_tau.append(tau)
                        mem_var.append(n_std)
                        reg_errs[r_sel].append(float(np.linalg.norm(tau_g - tau)))

                if t == 149:
                    err_at_150 = evaluate_inverse_model(mem_theta, mem_tau, mem_var, benchmark, rng)

            early_errors_150.append(err_at_150)
            final_errors.append(evaluate_inverse_model(mem_theta, mem_tau, mem_var, benchmark, rng))
            T_all = np.asarray(mem_tau, dtype=np.float64)
            far_cast_shares.append(float(np.mean(T_all[:, 0] > 0.25)))

        out[alg] = {
            "early_benchmark_distance_error_at_150": round(float(np.mean(early_errors_150)), 4),
            "final_benchmark_distance_error_at_450": round(float(np.mean(final_errors)), 4),
            "far_cast_outer_basin_coverage_share": round(float(np.mean(far_cast_shares)), 4),
        }

    return out


# ---------------------------------------------------------------------------
# 2. Experiment E2: Comparing the 3 Demonstrator Types in SGIM-D (Section 5.3, Fig. 7c)
# ---------------------------------------------------------------------------

def run_e2_three_demonstrator_types(n_trials: int = 360, n_seeds: int = 8) -> dict:
    benchmark, demos_dict = make_benchmark_and_teacher_sets()
    out: dict[str, dict] = {}

    for demo_name, demo_set in demos_dict.items():
        errs = []
        mean_vars = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(2200 + seed)
            mem_theta: list[np.ndarray] = []
            mem_tau: list[np.ndarray] = []
            mem_var: list[float] = []

            for t in range(n_trials):
                if (t % 30) < 3:
                    th_d, _ = demo_set[(t // 30) % len(demo_set)]
                    th = np.clip(th_d + rng.uniform(-0.04, 0.04, size=25), 0.0, 1.0)
                else:
                    tau_g = benchmark[int(rng.integers(0, len(benchmark)))]
                    if not mem_tau:
                        th = rng.uniform(0.0, 1.0, size=25)
                    else:
                        T_arr = np.asarray(mem_tau, dtype=np.float64)
                        V_arr = np.asarray(mem_var, dtype=np.float64)
                        scores = np.sqrt(np.sum((T_arr - tau_g[None, :]) ** 2, axis=1)) + 0.5 * V_arr
                        best_i = int(np.argmin(scores))
                        th = np.clip(mem_theta[best_i] + rng.normal(0.0, 0.05, size=25), 0.0, 1.0)
                tau, n_std = fishing_forward_map(th, rng)
                mem_theta.append(th)
                mem_tau.append(tau)
                mem_var.append(n_std)

            errs.append(evaluate_inverse_model(mem_theta, mem_tau, mem_var, benchmark, rng))
            mean_vars.append(float(np.mean(mem_var)))

        out[demo_name] = {
            "final_benchmark_distance_error": round(float(np.mean(errs)), 4),
            "mean_executed_movement_noise_std": round(float(np.mean(mean_vars)), 4),
        }

    return out


def main() -> int:
    t0 = time.perf_counter()
    e1 = run_e1_five_algorithms_comparison()
    e2 = run_e2_three_demonstrator_types()

    # Verify ranking from Section 6.1 (p. 13): SGIM-D < SAGG-RIAC & Imitation < Random < Observation
    assert (
        e1["sgim_d"]["final_benchmark_distance_error_at_450"]
        < e1["sagg_riac"]["final_benchmark_distance_error_at_450"]
        < e1["random_exploration"]["final_benchmark_distance_error_at_450"]
        < e1["learning_from_observation"]["final_benchmark_distance_error_at_450"]
    )
    assert (
        e1["sgim_d"]["final_benchmark_distance_error_at_450"]
        < e1["imitation_learning"]["final_benchmark_distance_error_at_450"]
    )
    # Verify Demonstrator 2 & 3 (low-variance reliable) beat Demonstrator 1 (naive high-variance)
    assert (
        e2["demonstrator_2_reliable_low_variance"]["final_benchmark_distance_error"]
        < e2["demonstrator_1_naive_high_variance"]["final_benchmark_distance_error"]
    )
    assert (
        e2["demonstrator_3_human_kinesthetic"]["final_benchmark_distance_error"]
        < e2["demonstrator_1_naive_high_variance"]["final_benchmark_distance_error"]
    )

    report = {
        "paper_id": "12-nguyen-oudeyer-2014-sgim",
        "title": "Socially Guided Intrinsic Motivation for Robot Learning of Motor Skills",
        "authors": "Sao Mai Nguyen, Pierre-Yves Oudeyer",
        "year": 2014,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "e1_five_algorithms_fishing_rod_benchmark": e1,
        "e2_three_demonstrator_types_comparison": e2,
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "sgim_d_2014_results.json"
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
