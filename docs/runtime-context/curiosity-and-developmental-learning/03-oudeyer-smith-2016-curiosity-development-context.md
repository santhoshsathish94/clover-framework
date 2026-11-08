# Pierre-Yves Oudeyer & Linda B. Smith (2016) — *How Evolution May Work Through Curiosity-Driven Developmental Process* — Context

Offered, not imposed. Date: 2026-11-08. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Oudeyer, P.-Y., & Smith, L. B. (2016). *How Evolution May Work Through Curiosity-Driven Developmental Process*. Topics in Cognitive Science, Vol. 8, Issue 2, pp. 492–502. (`doi:10.1111/tops.12196`, `PMID: 26969919`)
- **Primary Source File**: `https://www.pyoudeyer.com/OudeyerSmithTopicsCogSci14.pdf` (`651,763` bytes PDF, 11 pages, extracted text `35,395` bytes in `research/curiosity-and-developmental-learning/papers_text/03-oudeyer-smith-2016-curiosity-development.txt`).
- **Executable Experiments**: [research/curiosity-and-developmental-learning/03-oudeyer-smith-2016-curiosity-development/run_oudeyer_smith_2016_experiments.py](../../../research/curiosity-and-developmental-learning/03-oudeyer-smith-2016-curiosity-development/run_oudeyer_smith_2016_experiments.py)
- **Results Artifact**: [research/curiosity-and-developmental-learning/03-oudeyer-smith-2016-curiosity-development/results/oudeyer_smith_2016_results.json](../../../research/curiosity-and-developmental-learning/03-oudeyer-smith-2016-curiosity-development/results/oudeyer_smith_2016_results.json)

---

## 2. What the Paper Claims & Its Mathematical Mechanism

### 2.1 Self-Organized Epigenesis & Multi-Armed Bandit Progress Niches (Section 1–2, Fig. 1)
Oudeyer and Smith synthesize developmental psychology (Piaget's 4-month-old rattle shaking, Gottlieb's 1991 experiential canalization, Thelen & Smith's dynamic systems theory) with computational developmental robotics (`R-IAC` and Multi-Armed Bandit exploration, Audibert, Munos & Szepesvári, 2009):
- When a learner is confronted with multiple sensorimotor activities $a \in \{1, \dots, K\}$ whose prediction-error curves $e_a(n)$ decrease at different rates depending on morphological and environmental constraints (Figure 1C, p. 4), probabilistic selection of activities proportional to their empirical learning progress:
  $$\text{LP}_a(t) = \max\!\Big(0,\; \langle e_a(t - \tau) \rangle - \langle e_a(t) \rangle\Big), \qquad P(a_t = a) = (1 - \epsilon)\frac{\exp(\beta\,\text{LP}_a(t))}{\sum_{b} \exp(\beta\,\text{LP}_b(t))} + \frac{\epsilon}{K}$$
  automatically avoids both already-predictable activities (Curve 4, $\text{LP} \approx 0$) and unlearnable activities (Curve 1, $\text{LP} \approx 0$), focusing first on the activity with the steepest initial error drop (Curve 3) and shifting to more complex activities (Curve 2) once the simpler ones plateau.

### 2.2 Regularities, Individual Diversity, and Exaptation of Vocal Interaction (Section 2.1–3, pp. 5–9)
1. **Mixture of Universal Stage Regularities and Individual Diversity (Section 2.2, p. 6)**:
   - Because the coupled learner-body-environment system has multiple "progress niche" attractors with different basin sizes, most individuals follow a canonical sequence (`body babbling -> object grasping -> object bashing -> vocalizing to an imitating peer`), while stochasticity in early exploration causes a minority of individuals to invert stages—explaining how universal developmental milestones and individual differences arise from the *exact same* learning mechanism.
2. **Modality-General Curiosity Bootstrapping Speech Prerequisites (Section 2.2 & 3, pp. 7–8; Moulin-Frier et al., 2014)**:
   - A single modality-independent drive to maximize learning progress drives a vocal tract learner through four emergent stages without any innate language-specific module:
     $$\text{Phonation Control} \longrightarrow \text{Unarticulated Vowels} \longrightarrow \text{Proto-Syllables} \longrightarrow \text{Social Vocal Imitation}$$
   - In evolutionary terms, this provides a reservoir of behavioral innovations recruited via **exaptation** (Gould, 1991) for language.
3. **The Motivational Hierarchy Question (Section 3, p. 9)**:
   - Oudeyer and Smith highlight a critical open question: *How does curiosity compete and couple with homeostatic (food) and aversive (bodily safety) drives?* They hypothesize that human altriciality and extended parental protection allow curiosity to dominate the motivational hierarchy long enough for late-stage social/communicative skills to emerge, whereas high survival risk truncates curiosity-driven development.

---

## 3. Executable Experiments Run & Observed Results

We implemented and ran all three mechanisms in [run_oudeyer_smith_2016_experiments.py](../../../research/curiosity-and-developmental-learning/03-oudeyer-smith-2016-curiosity-development/run_oudeyer_smith_2016_experiments.py):

### Experiment E1: 5-Activity Social Playground — Regularities & Individual Diversity (`n = 40` runs)
- **Setup**: 5 activities (`already_predictable_stare`, `grasp_elephant`, `bash_hanging_toy`, `vocalize_to_peer_teacher`, `unlearnable_bash_distant_teacher`) with physical affordance basin probabilities $p_{\text{discover}} \in \{1.0, 0.28, 0.14, 0.07, 0.0\}$.
- **Observed Results**:
  - **Dominant Canonical Trajectory**: `grasp_elephant -> bash_hanging_toy -> vocalize_to_peer_teacher` occurred in **`42.5%` (`17/40`)** of runs ($2.55\times$ the uniform chance rate of $16.7\%$).
  - **Object Manipulation Before Social Vocalization**: Occurred in **`70.0%` (`28/40`)** of runs (`17` `grasp -> bash -> vocal` + `11` `bash -> grasp -> vocal`).
  - **Emergent Individual Diversity**: All `6` possible stage orderings appeared across the 40 runs (`17, 11, 4, 3, 3, 2`), confirming that stage inversions arise naturally when a narrower affordance basin is hit early by chance during random babbling.

### Experiment E2: Vocal Tract Self-Exploration to Social Peer Imitation (`n = 20` seeds, `800` steps)
- **Observed Dominant Mode Across Developmental Quarters**:

| Developmental Quarter | `phonation_control` | `unarticulated_vowels` | `proto_syllables` | `social_peer_imitation` |
| :--- | :---: | :---: | :---: | :---: |
| **`Q1` (Steps `0–200`)** | **`100.0%`** | `0.0%` | `0.0%` | `0.0%` |
| **`Q2` (Steps `200–400`)** | `0.0%` | **`100.0%`** | `0.0%` | `0.0%` |
| **`Q3` (Steps `400–600`)** | `0.0%` | `0.0%` | **`100.0%`** | `0.0%` |
| **`Q4` (Steps `600–800`)** | `0.0%` | `0.0%` | `0.0%` | **`100.0%`** (mean onset step `607.0`) |

### Experiment E3: Coupling Curiosity with Homeostatic / Aversive Survival Drives (Section 3 Open Question, p. 9)
Testing $R_{\text{total}}(a) = \text{LP}(a) + w_{\text{hunger}}(E(t)) \cdot R_{\text{extrinsic}}(a)$ across 3 ecological regimes:

| Ecological Regime | Mean Skills Mastered (out of `3`) | Late-Stage Vocal Interaction Competence | Observed Developmental Effect |
| :--- | :---: | :---: | :--- |
| **`protected_altricial_childhood`** (Caretaker buffers energy) | **`3.00 / 3`** | **`0.9026`** | Full 3-stage progression through object affordances to social vocal interaction. |
| **`moderate_resource_scarcity`** (Periodic foraging required) | **`3.00 / 3`** | `0.8876` | Learner interleaves foraging (`grasp_elephant`) with exploration and still reaches vocal stage. |
| **`high_risk_aversive_ecology`** (Severe depletion + risk penalty) | **`1.00 / 3`** | **`0.6728`** | Aversive survival drive locks learner into exploiting the first foraging habit (`grasp_elephant`), truncating `bash` and `vocalize`. |

---

## 4. Hidden Assumptions & Insights Discovered By Running the Code

1. **Why Smooth Deterministic Curves Alone Produce Zero Stage Diversity**: In our first pass of `E1`, when `grasp`, `bash`, and `vocalize` were modeled as purely deterministic functions of trial count $n$ (as drawn schematics in Figure 1C suggest), `40/40` (`100%`) of runs followed the exact same order (`distinct_trajectories_count = 1`), contradicting Section 2.2's claim of individual stage inversions. Stage inversions only emerge when each activity has a **stochastic affordance-discovery threshold** (a narrow region in continuous parameter space that must be hit before error begins to drop).
2. **Prerequisite Gating on Cumulative Effective Practice (`eff_counts`)**: In `E2`, if random exploration during `Q1–Q3` increments the practice counter $n$ of `social_peer_imitation` *before* the learner has acquired the prerequisite `proto_syllables` competence needed to produce recognizable sounds, then when `proto_syllables` is finally mastered, the error curve of `social_peer_imitation` is evaluated at an artificially high $n$ where its slope is already flat. Tracking **effective post-prerequisite practice** (`eff_counts`) is required so that unlocking a motor prerequisite opens a fresh high-slope progress niche.
