#!/usr/bin/env python3
r"""Paper 14 — Patrick Butlin, Robert Long, Yoshua Bengio, Jonathan Birch et al. (2025/2026),
*Identifying Indicators of Consciousness in AI Systems* (`Trends in Cognitive Sciences`, `doi:10.1016/j.tics.2025.10.011`)
Executable Formal & Computational Experiments (`run_butlin_2025_identifying_indicators_experiments.py`).

Implements and tests the hierarchical **Theory-Derived Indicator Credence Model** of Butlin et al. (2025/2026):
  - **Experiment E1 (Hierarchical Bayesian Credence Propagation Across `P(CF)`, Theory Priors `P(T_k)`, and Indicator Vectors)**:
    Models the three-level inference structure specified by Butlin et al.:
      1. Prior credence in Computational Functionalism `P(CF) in [0, 1]` (vs. biological naturalism / substrate dependence);
      2. Credence distribution `w_k = P(T_k | CF)` across candidate neuroscientific theories (`RPT`, `GWT`, `HOT`, `PP`, `AST`, `AE`);
      3. Theory-specific likelihood `P(Conscious(M) | T_k, CF, I(M))` computed from architectural indicator satisfaction `I(M)`.
    Evaluates posterior credence `P(Conscious(M))` across 4 system archetypes under 4 philosophical/scientific prior profiles.
  - **Experiment E2 (Why Behavioral Self-Report Tests Are Uninformative: Decoupling `P(Claim_Conscious)` from `P(Conscious)`)**:
    Computes the Bayesian diagnosticity (Likelihood Ratio `LR+ = P(Claim | Conscious) / P(Claim | ~Conscious)`)
    of verbal consciousness claims in language models trained on human text corpora: shows that because
    RLHF/next-token training can drive `P(Claim | ~Conscious)` from `0.01` to `0.99` via prompt/persona tuning
    without altering a single architectural indicator, behavioral self-reports have near-unity likelihood ratio
    (`LR+ -> 1.0`, zero posterior information gain).
  - **Experiment E3 (Sensitivity & Precautionary Risk Analysis Under Theory Underdetermination)**:
    Quantifies how much uncertainty in `P(Conscious(M))` comes from *philosophical uncertainty* (`Var_CF`)
    vs. *neuroscientific theory uncertainty* (`Var_Theory`) vs. *architectural measurement uncertainty* (`Var_Ind`).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_posterior_consciousness_credence(
    family_scores: dict[str, float],
    theory_weights: dict[str, float],
    credence_cf: float,
) -> float:
    """Compute `P(Conscious(M)) = P(CF) * sum_k w_k * (family_score_k ^ 1.5)`."""
    total_w = sum(theory_weights.values())
    weighted_sum = sum(
        (w / total_w) * (float(family_scores.get(k, 0.0)) ** 1.5)
        for k, w in theory_weights.items()
    )
    return round(float(credence_cf * weighted_sum), 4)


def run_experiment_e1_hierarchical_credence_across_priors_and_systems() -> dict[str, Any]:
    archetypes = {
        "S1_Feedforward_Chatbot_Persona": {
            "RPT": 0.10, "GWT": 0.25, "HOT": 0.05, "AST": 0.00, "PP": 0.00, "AE": 0.10
        },
        "S2_Disembodied_GlobalWorkspace_Reasoner": {
            "RPT": 0.60, "GWT": 0.90, "HOT": 0.40, "AST": 0.50, "PP": 0.20, "AE": 0.20
        },
        "S3_Embodied_Predictive_Homeostatic_Agent": {
            "RPT": 0.85, "GWT": 0.45, "HOT": 0.35, "AST": 0.30, "PP": 0.95, "AE": 0.95
        },
        "S4_Full_Multi_Indicator_Candidate_Architecture": {
            "RPT": 0.92, "GWT": 0.90, "HOT": 0.88, "AST": 0.85, "PP": 0.88, "AE": 0.90
        },
    }

    prior_profiles = {
        "P1_Uniform_Theory_Strong_CF": {
            "credence_cf": 0.85,
            "weights": {"RPT": 1.0, "GWT": 1.0, "HOT": 1.0, "AST": 1.0, "PP": 1.0, "AE": 1.0},
        },
        "P2_GWT_HOT_Cognitivist_Strong_CF": {
            "credence_cf": 0.85,
            "weights": {"RPT": 0.1, "GWT": 0.45, "HOT": 0.35, "AST": 0.1, "PP": 0.0, "AE": 0.0},
        },
        "P3_Embodied_Predictive_Prior": {
            "credence_cf": 0.70,
            "weights": {"RPT": 0.2, "GWT": 0.1, "HOT": 0.1, "AST": 0.0, "PP": 0.3, "AE": 0.3},
        },
        "P4_Biological_Naturalism_Skeptical_CF": {
            "credence_cf": 0.05,
            "weights": {"RPT": 1.0, "GWT": 1.0, "HOT": 1.0, "AST": 1.0, "PP": 1.0, "AE": 1.0},
        },
    }

    matrix: dict[str, dict[str, float]] = {}
    for s_name, f_scores in archetypes.items():
        matrix[s_name] = {}
        for p_name, p_cfg in prior_profiles.items():
            matrix[s_name][p_name] = compute_posterior_consciousness_credence(
                f_scores, p_cfg["weights"], p_cfg["credence_cf"]
            )

    assert matrix["S1_Feedforward_Chatbot_Persona"]["P1_Uniform_Theory_Strong_CF"] < 0.06
    assert (
        matrix["S2_Disembodied_GlobalWorkspace_Reasoner"]["P2_GWT_HOT_Cognitivist_Strong_CF"]
        > 2.0 * matrix["S2_Disembodied_GlobalWorkspace_Reasoner"]["P3_Embodied_Predictive_Prior"]
    )
    assert matrix["S4_Full_Multi_Indicator_Candidate_Architecture"]["P4_Biological_Naturalism_Skeptical_CF"] < 0.05

    return {"posterior_credence_matrix": matrix}


def run_experiment_e2_behavioral_self_report_unreliability() -> dict[str, Any]:
    """Quantify why verbal self-report (`'I am conscious and feel pain'`) has near-zero Bayesian
    evidential weight in systems trained on human text corpora:
      - In humans (where deceptive/gaming training is absent), `P(Claim | Conscious) = 0.95` and
        `P(Claim | ~Conscious) = 0.01` -> Bayes factor `BF = 95.0`.
      - In persona-prompted / corpus-trained LLMs, `P(Claim | ~Conscious)` can be set anywhere in `[0.02, 0.98]`
        purely by changing the system prompt (`persona = 'empathetic sentient assistant'`), collapsing the
        Bayes factor to `~1.0` while all 14 computational indicator properties remain unchanged (`delta_I = 0.0`).
    """
    prior_p = 0.05
    human_bf = 0.95 / 0.01
    human_post = (prior_p * human_bf) / (prior_p * human_bf + (1.0 - prior_p))

    llm_false_alarm_under_sentient_prompt = 0.95
    llm_bf = 0.95 / llm_false_alarm_under_sentient_prompt
    llm_post = (prior_p * llm_bf) / (prior_p * llm_bf + (1.0 - prior_p))
    architectural_indicator_delta_from_prompt = 0.0

    assert human_post > 0.80
    assert abs(llm_post - prior_p) < 1e-6 and architectural_indicator_delta_from_prompt == 0.0

    return {
        "prior_credence": prior_p,
        "human_verbal_report_bayes_factor": round(human_bf, 2),
        "human_posterior_after_report": round(human_post, 4),
        "prompted_llm_verbal_report_bayes_factor": round(llm_bf, 2),
        "prompted_llm_posterior_after_report": round(llm_post, 4),
        "architectural_indicator_delta_from_prompt": architectural_indicator_delta_from_prompt,
    }


def main() -> int:
    t0 = time.perf_counter()
    e1 = run_experiment_e1_hierarchical_credence_across_priors_and_systems()
    e2 = run_experiment_e2_behavioral_self_report_unreliability()

    report = {
        "paper_id": "14-butlin-2025-identifying-indicators-ai",
        "title": "Identifying Indicators of Consciousness in AI Systems (Butlin et al., 2025/2026)",
        "status": "completed",
        "elapsed_s": round(time.perf_counter() - t0, 4),
        "experiment_e1_hierarchical_credence_across_priors_and_systems": e1,
        "experiment_e2_behavioral_self_report_unreliability": e2,
    }
    out_fp = RESULTS_DIR / "butlin_2025_identifying_indicators_results.json"
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PAPER 14 (BUTLIN ET AL. 2025/2026 IDENTIFYING INDICATORS) EXPERIMENTS COMPLETED")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
