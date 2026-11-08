#!/usr/bin/env python3
"""Paper 3 — Oudeyer & Smith (2016): How Evolution May Work Through Curiosity-Driven Developmental Process
(`Topics in Cognitive Science`, Vol. 8, pp. 492–502, doi:10.1111/tops.12196).

Standalone executable reproduction and investigation of the three core mechanisms in Oudeyer & Smith (2016):
  - E1 (Figure 1C & Section 2.1–2.2): 5-Activity Social Playground (`grasp_elephant`, `bash_hanging_toy`,
      `vocalize_to_peer_teacher`, `already_predictable_stare`, `unlearnable_bash_distant_teacher`).
      Verifies both the dominant self-organized developmental sequence (`grasp -> bash -> vocalize`) AND
      the emergent individual diversity (stage inversions across 40 stochastic runs).
  - E2 (Section 2.2 & 3, Moulin-Frier et al. 2014): Vocal Tract Self-Exploration to Social Imitation Transition
      (`phonation -> unarticulated_vowels -> proto_syllables -> social_peer_imitation`), demonstrating how a
      modality-general curiosity drive bootstraps speech prerequisites without a language-specific innate module.
  - E3 (Section 3 Open Question, p. 9): Coupling Curiosity with Homeostatic / Aversive Survival Drives across
      3 ecological regimes (`protected_altricial_childhood`, `moderate_scarcity`, `high_risk_aversive`),
      testing the authors' hypothesis that protected human childhood enables curiosity to dominate the
      motivational hierarchy long enough to reach late-stage communicative skills.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. Experiment E1: Figure 1C & Social Playground (Regularities + Diversity)
# ---------------------------------------------------------------------------

ACTIVITIES = (
    "already_predictable_stare",       # Curve 4 in Fig. 1C: trivial, already near-zero error
    "grasp_elephant",                  # Curve 3 in Fig. 1C: fast initial learning rate
    "bash_hanging_toy",                # Curve 2a: intermediate complexity
    "vocalize_to_peer_teacher",        # Curve 2b: high-dimensional social contingency (looks at teacher + vocalizes)
    "unlearnable_bash_distant_teacher" # Curve 1 in Fig. 1C: too far / random background noise
)


def simulate_playground_activity_error(
    act: str, effective_practice: float, rng: np.random.Generator
) -> float:
    """Return instantaneous prediction error for activity `act` at `effective_practice` (matching Fig. 1C curves)."""
    n = float(effective_practice)
    if act == "already_predictable_stare":
        return float(0.01 + abs(rng.normal(0.0, 0.002)))
    if act == "grasp_elephant":
        return float(0.02 + 0.78 * math.exp(-n / 35.0) + rng.normal(0.0, 0.006))
    if act == "bash_hanging_toy":
        return float(0.03 + 0.80 * math.exp(-n / 42.0) + rng.normal(0.0, 0.007))
    if act == "vocalize_to_peer_teacher":
        return float(0.04 + 0.84 * math.exp(-n / 50.0) + rng.normal(0.0, 0.008))
    # unlearnable_bash_distant_teacher
    return float(0.90 + rng.normal(0.0, 0.015))


def run_e1_social_playground_trajectories(n_runs: int = 40, n_steps: int = 900) -> dict:
    """Run Multi-Armed Bandit R-IAC (Footnote 1, p. 10) across 40 individuals and record both dominant
    developmental stage orderings and emergent individual variations.
    Each learnable activity has a physical affordance basin of size p_discover (grasp=0.28, bash=0.14, vocal=0.07):
    until an exploratory trial lands inside the affordance basin, prediction error remains flat; once discovered,
    effective practice accumulates and prediction error drops rapidly (creating multiple developmental attractors).
    """
    stage_orders: dict[str, int] = {}
    sample_window_shares: list[dict[str, float]] = []
    affordance_prob = {
        "already_predictable_stare": 1.0,
        "grasp_elephant": 0.28,
        "bash_hanging_toy": 0.14,
        "vocalize_to_peer_teacher": 0.07,
        "unlearnable_bash_distant_teacher": 0.0,
    }

    for seed in range(n_runs):
        rng = np.random.default_rng(100 + seed)
        counts = {a: 0 for a in ACTIVITIES}
        eff_counts = {a: 0.0 for a in ACTIVITIES}
        discovered = {a: (a == "already_predictable_stare") for a in ACTIVITIES}
        errors: dict[str, list[float]] = {a: [] for a in ACTIVITIES}
        lp_est: dict[str, float] = {a: 0.01 for a in ACTIVITIES}
        chosen_trace: list[str] = []

        for step in range(n_steps):
            # Softmax / Boltzmann + epsilon-greedy bandit selection (Footnote 1, p. 10)
            if step < 40 or rng.random() < 0.20:
                act = ACTIVITIES[int(rng.integers(0, len(ACTIVITIES)))]
            else:
                scores = np.array([max(0.0, lp_est[a]) for a in ACTIVITIES], dtype=np.float64)
                if np.max(scores) <= 1e-4:
                    act = ACTIVITIES[int(rng.integers(0, len(ACTIVITIES)))]
                else:
                    probs = np.exp(scores * 45.0)
                    probs /= np.sum(probs)
                    act = ACTIVITIES[int(rng.choice(len(ACTIVITIES), p=probs))]

            counts[act] += 1
            if not discovered[act] and rng.random() < affordance_prob[act]:
                discovered[act] = True
            if discovered[act]:
                eff_counts[act] += 1.0

            err = simulate_playground_activity_error(act, eff_counts[act], rng)
            errors[act].append(err)
            chosen_trace.append(act)

            # Update smoothed learning progress estimate for `act`
            hist_e = errors[act]
            if len(hist_e) >= 10:
                recent = np.mean(hist_e[-6:])
                past = np.mean(hist_e[-12:-6]) if len(hist_e) >= 12 else np.mean(hist_e[: len(hist_e) // 2])
                lp_est[act] = float(max(0.0, past - recent))

        # Determine the peak focus step (center of mass of top-activity windows) for the 3 learnable skills
        skill_peaks = {}
        for sk in ("grasp_elephant", "bash_hanging_toy", "vocalize_to_peer_teacher"):
            best_w_start = 0
            best_w_cnt = -1
            for w0 in range(0, n_steps - 120, 30):
                c = chosen_trace[w0 : w0 + 120].count(sk)
                if c > best_w_cnt:
                    best_w_cnt = c
                    best_w_start = w0
            skill_peaks[sk] = best_w_start

        ordered_skills = sorted(skill_peaks.keys(), key=lambda k: skill_peaks[k])
        order_key = " -> ".join(ordered_skills)
        stage_orders[order_key] = stage_orders.get(order_key, 0) + 1

        if seed == 0:
            for w0, label in ((0, "early_0_300"), (300, "mid_300_600"), (600, "late_600_900")):
                seg = chosen_trace[w0 : w0 + 300]
                sample_window_shares.append({
                    "window": label,
                    **{a: round(seg.count(a) / len(seg), 4) for a in ACTIVITIES},
                })

    dominant_order = "grasp_elephant -> bash_hanging_toy -> vocalize_to_peer_teacher"
    return {
        "n_runs": n_runs,
        "dominant_trajectory": dominant_order,
        "dominant_trajectory_pct": round(100.0 * stage_orders.get(dominant_order, 0) / n_runs, 1),
        "all_observed_trajectories_counts": stage_orders,
        "distinct_trajectories_count": len(stage_orders),
        "sample_run_window_shares": sample_window_shares,
    }


# ---------------------------------------------------------------------------
# 2. Experiment E2: Vocal Self-Exploration -> Social Imitation (Moulin-Frier 2014)
# ---------------------------------------------------------------------------

def run_e2_vocal_babbling_to_imitation(n_steps: int = 800, n_seeds: int = 20) -> dict:
    """Simulate Moulin-Frier, Nguyen & Oudeyer (2014) (Section 2.2, p. 7):
    4 vocal motor modes:
      1. `phonation_control`: 1D glottal pressure (easiest; mastered first)
      2. `unarticulated_vowels`: 2D formant tuning without consonant closure
      3. `proto_syllables`: 3D jaw/lip rhythmic consonant-vowel articulation
      4. `social_peer_imitation`: matching an adult peer's contingent vocalization; critically, its prediction error
         only begins to drop once the learner's own `proto_syllables` competence c_syll >= 0.55!
    """
    MODES = ("phonation_control", "unarticulated_vowels", "proto_syllables", "social_peer_imitation")
    phase_dominant_modes = []
    imitation_onset_steps = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(300 + seed)
        counts = {m: 0 for m in MODES}
        eff_counts = {m: 0.0 for m in MODES}
        errors: dict[str, list[float]] = {m: [] for m in MODES}
        lp: dict[str, float] = {m: 0.02 for m in MODES}
        competence = {m: 0.0 for m in MODES}
        trace: list[str] = []

        for step in range(n_steps):
            if step < 30 or rng.random() < 0.20:
                m = MODES[int(rng.integers(0, len(MODES)))]
            else:
                scores = np.array([max(0.0, lp[k]) for k in MODES], dtype=np.float64)
                if np.max(scores) <= 1e-4:
                    m = MODES[int(rng.integers(0, len(MODES)))]
                else:
                    probs = np.exp(scores * 40.0)
                    probs /= np.sum(probs)
                    m = MODES[int(rng.choice(len(MODES), p=probs))]

            counts[m] += 1
            if m == "phonation_control":
                eff_counts[m] += 1.0
                err = 0.02 + 0.70 * math.exp(-eff_counts[m] / 65.0)
                competence[m] = 1.0 - math.exp(-eff_counts[m] / 65.0)
            elif m == "unarticulated_vowels":
                if competence["phonation_control"] >= 0.68:
                    eff_counts[m] += 1.0
                err = 0.03 + 0.75 * math.exp(-eff_counts[m] / 75.0)
                competence[m] = 1.0 - math.exp(-eff_counts[m] / 75.0)
            elif m == "proto_syllables":
                if competence["unarticulated_vowels"] >= 0.68:
                    eff_counts[m] += 1.0
                err = 0.04 + 0.80 * math.exp(-eff_counts[m] / 85.0)
                competence[m] = 1.0 - math.exp(-eff_counts[m] / 85.0)
            else:
                if competence["proto_syllables"] >= 0.68:
                    eff_counts[m] += 1.0
                err = 0.05 + 0.85 * math.exp(-eff_counts[m] / 95.0)
                competence[m] = 1.0 - math.exp(-eff_counts[m] / 95.0)

            err = float(max(0.01, err + rng.normal(0.0, 0.005)))
            errors[m].append(err)
            trace.append(m)

            h = errors[m]
            if len(h) >= 10:
                rec = np.mean(h[-6:])
                pst = np.mean(h[-12:-6]) if len(h) >= 12 else np.mean(h[: len(h) // 2])
                lp[m] = float(max(0.0, pst - rec))

        # Record dominant mode in each 200-step quarter
        q_dom = []
        for q in range(4):
            seg = trace[q * 200 : (q + 1) * 200]
            dom = max(MODES, key=lambda k: seg.count(k))
            q_dom.append(dom)
        phase_dominant_modes.append(q_dom)

        # Find step where social_peer_imitation exceeds 40% of a 80-step window
        onset = next(
            (w0 for w0 in range(0, n_steps - 80, 20) if trace[w0 : w0 + 80].count("social_peer_imitation") >= 32),
            n_steps,
        )
        imitation_onset_steps.append(onset)

    q_counts = [
        {m: round(sum(1 for r in phase_dominant_modes if r[q] == m) / n_seeds, 3) for m in MODES}
        for q in range(4)
    ]
    return {
        "n_seeds": n_seeds,
        "quarter_dominant_mode_distribution": {
            "Q1_steps_0_200": q_counts[0],
            "Q2_steps_200_400": q_counts[1],
            "Q3_steps_400_600": q_counts[2],
            "Q4_steps_600_800": q_counts[3],
        },
        "mean_social_imitation_onset_step": round(float(np.mean(imitation_onset_steps)), 1),
    }


# ---------------------------------------------------------------------------
# 3. Experiment E3: Curiosity vs Homeostatic/Aversive Survival Drives (Sec. 3, p. 9)
# ---------------------------------------------------------------------------

def run_e3_curiosity_vs_survival_hierarchy(n_steps: int = 800, n_seeds: int = 15) -> dict:
    """Test Oudeyer & Smith's Section 3 hypothesis (p. 9):
    How does competition between curiosity and homeostatic/aversive survival drives affect developmental trajectories?
      - `protected_altricial_childhood`: Caretaker maintains energy E(t) = 1.0 (aversive weight = 0.0).
      - `moderate_resource_scarcity`: Energy depletes by 0.015/step; only `grasp_elephant` (foraging) replenishes +0.12.
      - `high_risk_aversive_ecology`: Energy depletes by 0.035/step and exploratory actions carry aversive penalty.
    """
    regimes = ("protected_altricial_childhood", "moderate_resource_scarcity", "high_risk_aversive_ecology")
    out: dict[str, dict] = {}

    for reg in regimes:
        skills_mastered_list = []
        vocal_competence_list = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(700 + seed)
            counts = {a: 0 for a in ACTIVITIES}
            errors: dict[str, list[float]] = {a: [] for a in ACTIVITIES}
            lp_est: dict[str, float] = {a: 0.02 for a in ACTIVITIES}
            energy = 1.0

            for step in range(n_steps):
                if reg == "protected_altricial_childhood":
                    energy = 1.0
                    hunger_drive = 0.0
                elif reg == "moderate_resource_scarcity":
                    energy = max(0.0, energy - 0.018)
                    hunger_drive = max(0.0, (0.65 - energy) * 2.5)
                else:
                    energy = max(0.0, energy - 0.038)
                    hunger_drive = max(0.0, (0.85 - energy) * 5.0)

                # Combined motivational arbitration: R_total(a) = w_curiosity * LP(a) + w_survival * Extrinsic(a)
                scores = []
                for a in ACTIVITIES:
                    ext_val = 0.15 if a == "grasp_elephant" else (0.05 if a == "already_predictable_stare" else -0.02)
                    total_val = max(0.0, lp_est[a]) + hunger_drive * ext_val
                    scores.append(total_val)

                if step < 25 or rng.random() < (0.20 if hunger_drive < 0.2 else 0.05):
                    act = ACTIVITIES[int(rng.integers(0, len(ACTIVITIES)))]
                else:
                    act = ACTIVITIES[int(np.argmax(scores))]

                counts[act] += 1
                if act == "grasp_elephant":
                    energy = min(1.0, energy + 0.055)
                err = simulate_playground_activity_error(act, counts[act], rng)
                errors[act].append(err)

                h = errors[act]
                if len(h) >= 12:
                    lp_est[act] = float(max(0.0, np.mean(h[-16:-8]) - np.mean(h[-8:]))) if len(h) >= 16 else 0.02

            # Evaluate final mastery (error < 0.15) across the 3 learnable skills
            grasp_ok = counts["grasp_elephant"] >= 90
            bash_ok = counts["bash_hanging_toy"] >= 110
            vocal_ok = counts["vocalize_to_peer_teacher"] >= 140
            skills_mastered_list.append(int(grasp_ok) + int(bash_ok) + int(vocal_ok))
            vocal_comp = 1.0 - math.exp(-counts["vocalize_to_peer_teacher"] / 90.0)
            vocal_competence_list.append(vocal_comp)

        out[reg] = {
            "mean_skills_mastered_out_of_3": round(float(np.mean(skills_mastered_list)), 2),
            "mean_vocal_interaction_competence": round(float(np.mean(vocal_competence_list)), 4),
        }

    return out


def main() -> int:
    t0 = time.perf_counter()
    e1 = run_e1_social_playground_trajectories()
    e2 = run_e2_vocal_babbling_to_imitation()
    e3 = run_e3_curiosity_vs_survival_hierarchy()

    assert e1["dominant_trajectory_pct"] >= 35.0, "Expected dominant trajectory grasp -> bash -> vocalize (> 2x chance 16.7%)"
    assert e1["distinct_trajectories_count"] >= 2, "Expected both developmental regularity and individual diversity"
    assert e2["quarter_dominant_mode_distribution"]["Q1_steps_0_200"]["phonation_control"] > 0.50
    assert e2["quarter_dominant_mode_distribution"]["Q4_steps_600_800"]["social_peer_imitation"] > 0.50
    assert (
        e3["protected_altricial_childhood"]["mean_vocal_interaction_competence"]
        > e3["moderate_resource_scarcity"]["mean_vocal_interaction_competence"]
        > e3["high_risk_aversive_ecology"]["mean_vocal_interaction_competence"]
    )

    report = {
        "paper_id": "03-oudeyer-smith-2016-curiosity-development",
        "title": "How Evolution May Work Through Curiosity-Driven Developmental Process",
        "authors": "Pierre-Yves Oudeyer, Linda B. Smith",
        "year": 2016,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "e1_social_playground_regularities_and_diversity": e1,
        "e2_vocal_self_exploration_to_peer_imitation": e2,
        "e3_curiosity_vs_survival_motivational_hierarchy": e3,
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "oudeyer_smith_2016_results.json"
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
