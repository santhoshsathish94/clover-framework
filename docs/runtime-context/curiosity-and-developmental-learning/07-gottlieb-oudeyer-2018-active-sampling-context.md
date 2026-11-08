# Context File — Paper 07: Gottlieb & Oudeyer (2018) — *Towards a Neuroscience of Active Sampling and Curiosity*

- **Citation**: Jacqueline Gottlieb & Pierre-Yves Oudeyer (2018). *Towards a Neuroscience of Active Sampling and Curiosity*. **Nature Reviews Neuroscience**, Vol. 19, pp. 758–770 (`doi:10.1038/s41583-018-0078-0`).
- **Source PDF / Extracted Text**:
  - Primary PDF URL: `https://www.pyoudeyer.com/2018_GottliebOudeyer_ActiveSampling_NatNeuro.pdf`
  - Extracted Full Text (`82,258 bytes`): [research/curiosity-and-developmental-learning/papers_text/07-gottlieb-oudeyer-2018-active-sampling.txt](../../../research/curiosity-and-developmental-learning/papers_text/07-gottlieb-oudeyer-2018-active-sampling.txt)
- **Executable Experiment & Saved Results**:
  - Experiment Script: [research/curiosity-and-developmental-learning/07-gottlieb-oudeyer-2018-active-sampling/run_gottlieb_oudeyer_2018_experiments.py](../../../research/curiosity-and-developmental-learning/07-gottlieb-oudeyer-2018-active-sampling/run_gottlieb_oudeyer_2018_experiments.py)
  - Saved Results Artifact: [research/curiosity-and-developmental-learning/07-gottlieb-oudeyer-2018-active-sampling/results/gottlieb_oudeyer_2018_results.json](../../../research/curiosity-and-developmental-learning/07-gottlieb-oudeyer-2018-active-sampling/results/gottlieb_oudeyer_2018_results.json)

---

## Stage 1 — Context (Primary Source & Architectural Synthesis)

### 1.1 Information Sampling (Familiar Task) vs. Information Search (Open-Ended Discovery)
Gottlieb & Oudeyer (2018) introduce a foundational distinction (pp. 2–3, Fig. 1b):
1. **Information Sampling**: Gathering information within a *known* task structure where the relevant cues and actions are known ex ante (e.g., checking a traffic light; Foley et al., 2017 two-step saccade validity task in Figure 3).
2. **Information Search**: Open-ended investigation under conditions of radical ignorance where the agent does not know ex ante whether a useful regularity exists or what material reward it might yield (e.g., an early human noticing sparks when striking two stones, p. 2; or a humanoid robot discovering tool use in Figure 5a).

### 1.2 Closing the Attention–Decision Loop via Belief-Based Utility (Figure 1b, p. 3)
Traditional attention models start from an unexplained "priority map," while traditional decision models start from a pre-selected sensory input $p(I|\theta)$ (Figure 1a). Gottlieb & Oudeyer unify them by modeling **attention as an endogenous sampling action** that selects $I'$ based on two feedback loops (Figure 1b):
- **Extrinsic Utility**: Expected material payoff of the final action.
- **Belief-Based Utility**: Intrinsic preferences over cognitive states — including **uncertainty reduction** (encoded in `LIP` validity responses independent of reward, Fig. 3c), **anticipatory savouring / reward-based salience** (encoded in `OFC` and midbrain `DA`, driving redundant search for $100\%$-reward cues in Fig. 4a), and **Learning Progress (`LP`)**.

### 1.3 Why Normative Theories Fail at Information Search (p. 8)
Quoting Bossaerts & Murawski (2017), the authors emphasize that in normative theories (such as exact Bayesian active inference / global free-energy minimization), *"a decision problem with two alternatives is not distinguished from one with $2^{100}$ alternatives"* (p. 8). Because biological organisms have bounded computational capacity, they rely on **belief-based heuristics (`LP` goal babbling, Box 1 & Fig. 5)** that create stepping-stone curricula (`hand -> stick -> distant cylindrical object`) without needing to plan over $2^{100}$ futures.

---

## Stage 2 — Direction (Testable Hypotheses)

1. **Experiment E1 (Figure 4a Non-Instrumental Search & The "Double-Edged Sword" in Instrumental Loss Evasion, pp. 5–7)**:
   - Verify that a pure uncertainty-reduction model predicts zero Cue 2 search at both `0%` and `100%` Cue 1 reward probability, whereas a hybrid **Uncertainty + Dopaminergic Savouring** model reproduces the empirical monkey data in Figure 4a (`50% > 100% > 0%`).
   - Test the "double-edged sword" (p. 6, Barbaro et al. 2017, Ref. 92): when an observer must choose between inspecting a Gain Cue (`75%` validity, $+\$1$) vs. a Loss-Evasion Cue (`95%` validity, evading a $-\$2$ hazard), compare an unbiased `LIP` validity sampler against a reward-salience biased sampler.
2. **Experiment E2 (Figure 5a Hierarchical Tool Use & Figure 6 Free-Play Curriculum, pp. 8–10)**:
   - Compare `extrinsic_rl_sparse`, `random_exploration`, and `lp_autonomous_goal_sampling` in the 6-zone Figure 5a environment (`hand_reaching -> stick_grasping -> cylindrical_object_via_stick` + `unmovable_object` + `2 random distractors`).
   - Reproduce Figure 6 (Baranes, Oudeyer & Gottlieb, 2014) across 70 free-play games at 7 dot speeds (`10..70 deg/s`).

---

## Stage 3 — Execution & Stage 4 — Outcome (Quantitative Results)

Running [run_gottlieb_oudeyer_2018_experiments.py](../../../research/curiosity-and-developmental-learning/07-gottlieb-oudeyer-2018-active-sampling/run_gottlieb_oudeyer_2018_experiments.py) produces:

### 4.1 Experiment E1: Non-Instrumental Search (Fig. 4a) & Instrumental Loss-Evasion (Ref. 92)
- **Figure 4a Non-Instrumental Cue 2 Search Probability**:
  - `pure_uncertainty_reduction`: `0% Cue 1 -> 0.0500`, `50% Cue 1 -> 0.8000`, `100% Cue 1 -> 0.0500` (fails to explain redundant good-news search at `100%`).
  - `hybrid_uncertainty_plus_savouring`: `0% Cue 1 -> 0.0500`, **`50% Cue 1 -> 0.8100`**, **`100% Cue 1 -> 0.4700`** (reproduces the exact `50% > 100% > 0%` monkey pattern in Fig. 4a).
- **The Double-Edged Sword in Instrumental Gain vs. Loss-Evasion Sampling**:
  - `lip_unbiased_validity_sampler`: attends to the `95%`-valid Loss-Evasion Cue **`90.83%`** of the time, achieving **`+0.3383`** net payoff.
  - `reward_salience_biased_sampler`: distracted by the Gain Cue, attends to the Loss-Evasion Cue only **`26.83%`** of the time, suffering a net loss of **`-0.0825`**.

### 4.2 Experiment E2: Figure 5a Hierarchical Tool Use & Figure 6 Free-Play Trajectory
| Agent | `hand_reaching` | `stick_grasping` | `cylinder_via_stick` | Distractors (`3` zones) | Final Cylinder Tool Competence |
|---|---|---|---|---|---|
| **`extrinsic_rl_sparse`** | `6.30%` | `6.20%` | `67.85%` (premature!) | `19.65%` | `0.0594` (Fails) |
| **`random_exploration`** | `16.28%` | `16.63%` | `16.39%` | `50.70%` | `0.6758` |
| **`lp_autonomous_goal_sampling`** | **`42.52%`** | **`31.70%`** | **`15.30%`** | **`10.49%`** | **`0.7735`** (Best) |

- **Figure 6 Free-Play 7-Speed Game Trajectory (`70` games)**:
  - Early games (`7..20`) mean dot speed: **`17.69 deg/s`**.
  - Late games (`50..70`) mean dot speed: **`62.00 deg/s`**.
  - Mean fraction correct maintained across session: **`71.53%`** (matching the empirical $70\text{--}80\%$ accuracy zone in Figure 6b!).

---

## Stage 5 — Growth (What We Learned & Critical Boundaries)

1. **Why Belief-Based Heuristics Are a "Double-Edged Sword"**:
   - The very same dopaminergic reward-salience bias that motivates vigorous engagement in non-instrumental tasks (`0.47` search at $100\%$ Cue 1) becomes maladaptive in instrumental environments requiring **loss evasion** (`-0.0825` vs. `+0.3383`), explaining why parietal (`LIP/IPS`) validity representations must remain dissociated from ventral-striatal/OFC reward savouring (Fig. 3c).
2. **Why `extrinsic_rl_sparse` Fails on Hierarchical Tool Use (`0.0594` vs. `0.7735`)**:
   - Optimizing directly for the extrinsic goal (`cylindrical_object_via_stick`, `67.85%` of tries) before acquiring the prerequisite `hand_reaching` and `stick_grasping` skills produces zero reward feedback (`mean_step_cylinder_unlocked = 444.8 / 450`), whereas `LP` goal sampling ignores the extrinsic reward and builds the prerequisite chain (`hand 42.5% -> stick 31.7% -> cylinder 15.3%`).
