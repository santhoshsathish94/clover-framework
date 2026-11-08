# Context File — Paper 05: Kaplan & Oudeyer (2007) — *In Search of the Neural Circuits of Intrinsic Motivation*

- **Citation**: Frédéric Kaplan & Pierre-Yves Oudeyer (2007). *In Search of the Neural Circuits of Intrinsic Motivation*. **Frontiers in Neuroscience**, Vol. 1, Issue 1, pp. 225–236 (`doi:10.3389/neuro.01.1.1.017.2007`).
- **Source PDF / Extracted Text**:
  - Primary PDF URL: `https://www.frontiersin.org/articles/10.3389/neuro.01.1.1.017.2007/pdf`
  - Extracted Full Text (`85,157 bytes`): [research/curiosity-and-developmental-learning/papers_text/05-kaplan-oudeyer-2007-neural-circuits.txt](../../../research/curiosity-and-developmental-learning/papers_text/05-kaplan-oudeyer-2007-neural-circuits.txt)
- **Executable Experiment & Saved Results**:
  - Experiment Script: [research/curiosity-and-developmental-learning/05-kaplan-oudeyer-2007-neural-circuits/run_neural_circuits_experiments.py](../../../research/curiosity-and-developmental-learning/05-kaplan-oudeyer-2007-neural-circuits/run_neural_circuits_experiments.py)
  - Saved Results Artifact: [research/curiosity-and-developmental-learning/05-kaplan-oudeyer-2007-neural-circuits/results/neural_circuits_2007_results.json](../../../research/curiosity-and-developmental-learning/05-kaplan-oudeyer-2007-neural-circuits/results/neural_circuits_2007_results.json)

---

## Stage 1 — Context (Primary Source & Neurocomputational Hypotheses)

### 1.1 Reinterpreting Dopamine Beyond Extrinsic Reward Prediction Error
Kaplan & Oudeyer (2007) synthesize neurobiological evidence showing that midbrain dopamine (DA) neurons respond not only to extrinsic reward prediction errors (Schultz et al., 1997), but also to **sensory novelty, salient non-rewarding stimuli, and action-outcome contingencies** (Horvitz, 2000; Kakade & Dayan, 2002; Redgrave & Gurney, 2006; Dayan & Balleine, 2002; Berridge, 2004). They formulate two specific neurocomputational hypotheses:
1. **Hypothesis 1 (Dual-Channel Phasic vs. Tonic Dopamine, pp. 232–233)**:
   - **Phasic DA** signals fast, short-latency prediction errors / surprise bursts ($e(t)$).
   - **Tonic DA** (extrasynaptic, slow-timescale background concentration, Niv et al., 2006; Schultz, 2007) encodes **regional learning progress** — the difference between two running averages of prediction error rates:
     $$p_{\mathcal{R}}(t) = \langle e_{\mathcal{R}}(t - \theta) \rangle - \langle e_{\mathcal{R}}(t) \rangle$$
     driving "wanting" (incentive salience, Berridge 2004) and response vigor.
2. **Hypothesis 2 (Cortical Microcircuits as Coupled Predictor `MR` + Metapredictor `metaMR`, pp. 229–234)**:
   - Why cannot the brain simply integrate phasic DA globally in a single limbic circuit? Because a **global running-average integrator** (Eq. 1) creates pathological oscillations between jumping against a chaotic wall and freezing in front of a static wall (`wall <-> immobile` oscillation, p. 229).
   - Why cannot the brain use a **pointwise $\delta$-ball evaluator** (Eq. 2, $p_x(t) = \langle e_x^\delta(t-\theta)\rangle - \langle e_x^\delta(t)\rangle$)? Because if the local memory is fast and $\delta$ is narrow, squashing a toy on the floor once is immediately predicted with zero error, causing **premature habituation** and preventing the child from generalizing the behavior to squashing the toy on the sofa or table (pp. 229–230).
   - Therefore, Kaplan & Oudeyer propose (Figure 2a–b, pp. 231–233) that **every cortical macrocolumn / microcircuit** implements a coupled predictor (`MR`, predicting afferent streams) and metapredictor (`metaMR`, tracking the column's regional error derivative $p_{\mathcal{R}}(t)$), recruited via competitive lateral inhibition to form broad sensorimotor categories and gating striatal action selection via cortico-striatal loops.

---

## Stage 2 — Direction (Testable Hypotheses)

1. **Experiment E1 (Global Limbic vs. Pointwise $\delta$-Ball vs. Cortical Microcircuit Columns, Eqs. 1–3)**:
   - Compare the three candidate neural evaluators across 10 seeds $\times 900$ steps on a continuous 2D sensorimotor space $x = (x_0, x_1) \in [0, 1]^2$ containing `immobile_static` ($x_0 \le 0.33$), `learnable_surface` ($0.33 < x_0 \le 0.67$, $y = 0.40 + 0.35\sin(\pi x_1) + 0.20 x_0$), and `chaotic_wall` ($x_0 > 0.67$).
   - Verify whether `global_limbic_integrator` oscillates between `chaotic_wall` and `immobile_static`, whether `pointwise_local_delta_ball` prematurely habituates after sampling its initial floor location $(0.50, 0.25)$, and whether `cortical_microcircuit_columns` sustains exploration across the entire `learnable_surface` category and achieves the lowest generalization MSE.
2. **Experiment E2 (Phasic vs. Tonic Dopamine Dissociation Across 3 Regimes)**:
   - Track `phasic_da` vs. `tonic_da_progress` across Phase A (novel learnable contingency), Phase B (mastered plateau), and Phase C (unlearnable white noise) to verify how tonic DA separates true learning progress (Phase A) from high-surprise unlearnable noise (Phase C).

---

## Stage 3 — Execution & Stage 4 — Outcome (Quantitative Results)

Running [run_neural_circuits_experiments.py](../../../research/curiosity-and-developmental-learning/05-kaplan-oudeyer-2007-neural-circuits/run_neural_circuits_experiments.py) (`10` seeds $\times 900$ steps in E1; `300`-step 3-phase trajectory in E2) yields:

### 4.1 Experiment E1: Comparing the 3 Candidate Neural Progress Evaluators (Eqs. 1–3)
| Neural Architecture | `wall <-> immobile` Oscillation Rate | Late `learnable_surface` Share | `learnable_surface` Generalization MSE |
|---|---|---|---|
| **`global_limbic_integrator` (Global Eq. 1)** | **`85.32%`** (`0.8532`) | `4.17%` (`0.0417`) | `0.000652` ($13.3\times$ worse) |
| **`pointwise_local_delta_ball` (Local Eq. 2)** | `18.71%` (`0.1871`) | `38.87%` (`0.3887`) | `0.000082` ($1.67\times$ worse) |
| **`cortical_microcircuit_columns` (Columnar Eq. 3)** | **`11.96%`** (`0.1196`) | **`59.80%`** (`0.5980`) | **`0.000049`** (Best) |

### 4.2 Experiment E2: Phasic vs. Tonic Dopamine Dissociation
| Regime | Mean Phasic DA (Instantaneous Prediction Error) | Mean Tonic DA (Regional Progress Derivative) | Behavioral Consequence |
|---|---|---|---|
| **Phase A: Novel Learnable Contingency (`t=0..100`)** | `0.1711` | **`0.0766`** (Peak Tonic DA) | High response vigor & sustained active practice |
| **Phase B: Mastered Contingency (`t=100..200`)** | `0.0338` | **`0.0007`** ($109\times$ lower) | Habituation / switches away from mastered task |
| **Phase C: Unlearnable White Noise (`t=200..300`)** | **`0.6301`** (Peak Phasic DA!) | **`0.0115`** ($6.66\times$ lower than Phase A) | Phasic surprise is high, but tonic "wanting" stays low |

---

## Stage 5 — Growth (What We Learned & Critical Boundaries)

1. **Why Granularity of the Evaluator ($\mathcal{B}_\delta(x)$ vs. Cortical Column $\mathcal{R}$) Controls Generalization**:
   - If the metapredictor is too global (`global_limbic_integrator`), it suffers **`85.32%` oscillation** between high-error chaos and zero-error stasis.
   - If the metapredictor is too local (`pointwise_local_delta_ball`, Eq. 2), fast memorization inside a tiny ball $\mathcal{B}_\delta(x^*)$ extinguishes curiosity at $x^*$ before that curiosity can spill over to neighboring states $(x_0, x_1')$ in the same functional category ("squashing the toy on the sofa vs. the floor").
   - Only an intermediate **category-level cortical column (`MR` + `metaMR`)** propagates tonic progress across the entire category $\mathcal{R}$, driving `learnable_surface` exploration up to **`59.80%`** and cutting generalization MSE to **`4.9e-5`**.
2. **Phasic Surprise $\neq$ Tonic "Wanting"**:
   - Experiment E2 shows why conflating phasic DA bursts (`0.6301` in Phase C white noise) with intrinsic reward causes novelty-trap addiction, whereas tonic DA (`0.0766` in Phase A vs. `0.0115` in Phase C) cleanly isolates progress niches.
