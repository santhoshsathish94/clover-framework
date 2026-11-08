#!/usr/bin/env python3
"""Paper 8 — Oudeyer, Gottlieb & Lopes (2016): Intrinsic Motivation, Curiosity, and Learning:
Theory and Applications in Educational Technologies (`Progress in Brain Research`, Vol. 229, pp. 257–284).

Standalone executable evaluation of the paper's two main educational & neurocomputational mechanisms:
  - E1 (Figure 1A vs Figure 1B, pp. 258, 266–268):
      Compares a Unidirectional Curiosity->Memory model (Fig. 1A, static novelty allocation) against the
      Closed Positive-Feedback Loop (Fig. 1B: `LP -> State Curiosity -> Dopaminergic-Hippocampal Memory Consolidation
      -> Faster Competence Gain -> New LP Niche`, Kang et al. 2009; Gruber et al. 2014).
  - E2 (Figures 4 & 5, pp. 276–278 — `ZPDES` Intelligent Tutoring System vs. Hand-Crafted Teacher Branching Sequence):
      Implements the `ZPDES` (Zone of Proximal Development and Empirical Success, Clement et al. 2015) multi-armed
      bandit tutoring algorithm across the 4 money-manipulation exercise types (`M`, `R`, `MM`, `RM`) x 3 difficulty
      levels (`L1_integer`, `L2_simple_decimal`, `L3_complex_decimal` = 12 activities) with dynamic ZPD expansion
      and modality switching (`A2` plateau -> open alternative modality `C1/C2`, Fig. 5) on a simulated cohort of
      N=400 heterogeneous schoolchildren.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. Experiment E1: Figure 1A (Unidirectional) vs Figure 1B (Closed LP-Curiosity-Memory Loop)
# ---------------------------------------------------------------------------

def run_e1_closed_curiosity_memory_feedback_loop(n_steps: int = 240, n_seeds: int = 16) -> dict:
    """4 learning topics (`t0_trivial`, `t1_foundational`, `t2_intermediate_bridge`, `t3_advanced_synthesis`)
    where `t2` requires `t1 >= 0.50`, `t3` requires `t2 >= 0.50`, and hippocampal memory retention rate
    `eta_mem = eta_0 * (1 + alpha_da * state_curiosity)` is amplified by state curiosity (Gruber et al., 2014).
    """
    TOPICS = ("t0_trivial", "t1_foundational", "t2_intermediate_bridge", "t3_advanced_synthesis")
    out: dict[str, dict] = {}

    for model in ("fig1a_unidirectional_static_novelty", "fig1b_closed_lp_curiosity_memory_loop"):
        final_masteries = []
        topic_allocations = {t: 0.0 for t in TOPICS}

        for seed in range(n_seeds):
            rng = np.random.default_rng(800 + seed)
            mastery = {"t0_trivial": 0.90, "t1_foundational": 0.05, "t2_intermediate_bridge": 0.0, "t3_advanced_synthesis": 0.0}
            counts = {t: 0 for t in TOPICS}
            err_hist: dict[str, list[float]] = {t: [1.0 - mastery[t]] for t in TOPICS}

            for step in range(n_steps):
                if step < 12 or rng.random() < 0.12:
                    t = TOPICS[int(rng.integers(0, len(TOPICS)))]
                    curiosity_state = 0.25
                elif model == "fig1a_unidirectional_static_novelty":
                    # Selects highest unfamiliarity / prediction error without tracking whether progress is occurring
                    errs = [1.0 - mastery[cand] for cand in TOPICS]
                    t = TOPICS[int(np.argmax(errs))]
                    curiosity_state = 0.30
                else:
                    # Fig. 1B: LP drives state curiosity, which both selects the activity AND boosts hippocampal retention
                    lps = []
                    for cand in TOPICS:
                        eh = err_hist[cand]
                        if len(eh) < 4:
                            lps.append(0.04)
                        else:
                            rec = float(np.mean(eh[-5:]))
                            pst = float(np.mean(eh[-10:-5])) if len(eh) >= 10 else float(np.mean(eh[: len(eh) // 2]))
                            lps.append(max(0.0, pst - rec))
                    best_i = int(np.argmax(lps))
                    t = TOPICS[best_i]
                    curiosity_state = min(1.0, 0.25 + 6.0 * lps[best_i])

                counts[t] += 1
                # Hippocampal memory consolidation multiplier (Kang et al., 2009; Gruber et al., 2014)
                eta = 0.028 * (1.0 + 1.25 * curiosity_state)
                if t == "t0_trivial":
                    mastery[t] = min(1.0, mastery[t] + eta * (1.0 - mastery[t]))
                elif t == "t1_foundational":
                    mastery[t] = min(1.0, mastery[t] + eta * (1.0 - mastery[t]))
                elif t == "t2_intermediate_bridge":
                    if mastery["t1_foundational"] >= 0.50:
                        mastery[t] = min(1.0, mastery[t] + eta * (1.0 - mastery[t]))
                elif t == "t3_advanced_synthesis":
                    if mastery["t2_intermediate_bridge"] >= 0.50:
                        mastery[t] = min(1.0, mastery[t] + eta * (1.0 - mastery[t]))

                err_hist[t].append(1.0 - mastery[t])

            final_masteries.append(float(np.mean(list(mastery.values()))))
            for t in TOPICS:
                topic_allocations[t] += (counts[t] / n_steps) / n_seeds

        out[model] = {
            "topic_allocation": {t: round(topic_allocations[t], 4) for t in TOPICS},
            "mean_final_curriculum_mastery": round(float(np.mean(final_masteries)), 4),
        }

    return out


# ---------------------------------------------------------------------------
# 2. Experiment E2: Figures 4 & 5 — ZPDES Bandit ITS on N=400 Simulated Students
# ---------------------------------------------------------------------------

def run_e2_zpdes_intelligent_tutoring_system(n_students: int = 400, n_trials: int = 72) -> dict:
    """Simulate Clement et al. (2015) / Section 10 (pp. 276–278, Figs. 4–5):
    - 4 Exercise Types (Fig. 4):
        `M`  (Customer / 1 object)
        `R`  (Merchant / 1 object — requires giving change)
        `MM` (Customer / 2 objects — requires addition)
        `RM` (Merchant / 2 objects — requires addition + change, OR alternative modality path when R stalls)
    - 3 Difficulty Levels per type (`1`: integers, `2`: simple decimals, `3`: complex decimals) -> 12 exercises.
    - Each of the N=400 students has heterogeneous learning rates and a randomly drawn "modality bottleneck"
      (e.g., 45% of students stall on `R2` until they build decimal addition fluency via `MM1/MM2`, matching
      Figure 5 where stalling on `A2` temporarily deactivates `A2` and expands the ZPD to modality `C1/C2`).
    """
    EXERCISES = [
        f"{m}{lvl}" for lvl in (1, 2, 3) for m in ("M", "R", "MM", "RM")
    ]
    tutor_modes = ("fixed_linear_sequence", "expert_handcrafted_branching", "zpdes_lp_bandit")
    results: dict[str, dict] = {}

    for mode in tutor_modes:
        posttest_scores = []
        max_level_reached = []
        early_diversity = []

        for s_id in range(n_students):
            rng = np.random.default_rng(4000 + s_id)
            base_rate = float(rng.uniform(0.08, 0.18))
            has_r2_bottleneck = bool(rng.random() < 0.45)

            skills = {ex: 0.10 for ex in EXERCISES}
            skills["M1"] = float(rng.uniform(0.25, 0.45))
            hist: dict[str, list[float]] = {ex: [] for ex in EXERCISES}
            zpd: set[str] = {"M1", "R1"}
            chosen_trace: list[str] = []

            for t in range(n_trials):
                if mode == "fixed_linear_sequence":
                    # Fixed non-adaptive schedule cycling through exercises 1..12 repeatedly:
                    # wastes early trials on Level 2 and Level 3 exercises before Level 1 prerequisites are met
                    ex = EXERCISES[t % 12]
                elif mode == "expert_handcrafted_branching":
                    # Expert teacher branching sequence (Clement et al. 2015): advances when accuracy >= 0.75,
                    # with a static rule fallback after 16 trials if a student stalls on R2
                    order = ["M1", "R1", "M2", "R2", "MM1", "MM2", "RM1", "RM2", "M3", "R3", "MM3", "RM3"]
                    ex = order[-1]
                    for cand in order:
                        if skills[cand] < 0.75 and not (cand == "R2" and len(hist["R2"]) >= 16):
                            ex = cand
                            break
                else:
                    # ZPDES (Fig. 5): maintains active ZPD, selects via empirical LP bandit, and expands ZPD
                    # to alternative modality (MM1/MM2) either on mastery OR when current exercise plateaus!
                    if skills["M1"] >= 0.70:
                        zpd.add("M2")
                        zpd.add("MM1")
                    if skills["R1"] >= 0.70:
                        zpd.add("R2")
                    # Fig. 5 key mechanism: if R2 has >=4 tries with near-zero progress, expand ZPD to MM1/MM2/RM1
                    if len(hist["R2"]) >= 4 and (hist["R2"][-1] - hist["R2"][-4]) < 0.04:
                        zpd.update({"MM1", "MM2", "RM1"})
                    if skills["M2"] >= 0.70:
                        zpd.update({"MM2", "M3"})
                    if skills["MM1"] >= 0.70:
                        zpd.update({"MM2", "RM1"})
                    if skills["RM1"] >= 0.70:
                        zpd.add("RM2")
                    if skills["MM2"] >= 0.70:
                        zpd.update({"RM2", "MM3"})
                        # Once MM2 is mastered, re-activate R2 if it had previously stalled (Fig. 5)
                        if skills["R2"] < 0.20 and len(hist["R2"]) >= 2:
                            hist["R2"] = [skills["R2"]]
                    if skills["R2"] >= 0.70:
                        zpd.update({"RM2", "R3"})
                    if skills["RM2"] >= 0.70:
                        zpd.add("RM3")

                    active_list = sorted(zpd)
                    if rng.random() < 0.10:
                        ex = active_list[int(rng.integers(0, len(active_list)))]
                    else:
                        scores = []
                        for cand in active_list:
                            h = hist[cand]
                            if skills[cand] >= 0.75:
                                scores.append(-0.1)  # deactivate mastered exercise (same 0.75 criterion)
                            elif len(h) < 2:
                                scores.append(0.12)  # initial exploration bonus for newly unlocked ZPD item
                            else:
                                lp = h[-1] - h[max(0, len(h) - 4)]
                                scores.append(lp)
                        ex = active_list[int(np.argmax(scores))]

                chosen_trace.append(ex)
                # Prerequisite check for learning gain
                lvl = int(ex[-1])
                mod = ex[:-1]
                prereq_ok = True
                if lvl == 2 and skills[f"{mod}1"] < 0.55:
                    prereq_ok = False
                if lvl == 3 and skills[f"{mod}2"] < 0.55:
                    prereq_ok = False
                if ex == "R2" and has_r2_bottleneck and skills["MM2"] < 0.55:
                    # Student with R2 bottleneck cannot progress on R2 until MM2 (decimal addition) is learned
                    prereq_ok = False
                if mod in ("MM", "RM") and skills["M1"] < 0.55:
                    prereq_ok = False

                if prereq_ok:
                    skills[ex] = min(1.0, skills[ex] + base_rate * (1.0 - skills[ex]))
                hist[ex].append(skills[ex])

            posttest_scores.append(float(np.mean(list(skills.values()))))
            highest_lvl = max((int(k[-1]) for k, v in skills.items() if v >= 0.70), default=0)
            max_level_reached.append(highest_lvl)
            early_diversity.append(len(set(chosen_trace[:24])))

        results[mode] = {
            "mean_posttest_score_all_12_exercises": round(float(np.mean(posttest_scores)), 4),
            "mean_max_difficulty_level_mastered": round(float(np.mean(max_level_reached)), 3),
            "early_session_exercise_diversity_first_24_trials": round(float(np.mean(early_diversity)), 2),
        }

    return results


def main() -> int:
    t0 = time.perf_counter()
    e1 = run_e1_closed_curiosity_memory_feedback_loop()
    e2 = run_e2_zpdes_intelligent_tutoring_system()

    assert (
        e1["fig1b_closed_lp_curiosity_memory_loop"]["mean_final_curriculum_mastery"]
        > e1["fig1a_unidirectional_static_novelty"]["mean_final_curriculum_mastery"]
    )
    assert (
        e2["zpdes_lp_bandit"]["mean_posttest_score_all_12_exercises"]
        > e2["expert_handcrafted_branching"]["mean_posttest_score_all_12_exercises"]
        > e2["fixed_linear_sequence"]["mean_posttest_score_all_12_exercises"]
    )
    assert (
        e2["zpdes_lp_bandit"]["early_session_exercise_diversity_first_24_trials"]
        > e2["expert_handcrafted_branching"]["early_session_exercise_diversity_first_24_trials"]
    )

    report = {
        "paper_id": "08-oudeyer-gottlieb-lopes-2016-educational",
        "title": "Intrinsic Motivation, Curiosity, and Learning: Theory and Applications in Educational Technologies",
        "authors": "Pierre-Yves Oudeyer, Jacqueline Gottlieb, Manuel Lopes",
        "year": 2016,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "e1_closed_curiosity_memory_feedback_loop": e1,
        "e2_zpdes_intelligent_tutoring_system_n400": e2,
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "educational_2016_results.json"
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
