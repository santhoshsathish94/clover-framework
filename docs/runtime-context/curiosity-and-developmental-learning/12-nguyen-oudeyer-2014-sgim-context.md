# Context File — Paper 12: Nguyen & Oudeyer (2014) — *Socially Guided Intrinsic Motivation for Robot Learning of Motor Skills*

- **Citation**: Sao Mai Nguyen & Pierre-Yves Oudeyer (2014). *Socially Guided Intrinsic Motivation for Robot Learning of Motor Skills*. **Autonomous Robots**, Vol. 36, No. 3, pp. 273–294 (`doi:10.1007/s10514-013-9339-y`).
- **Source PDF / Extracted Text**:
  - Primary PDF URL: `http://nguyensmai.free.fr/publication/Nguyen_2014Auro.pdf`
  - Extracted Full Text (`102,916 bytes`): [research/curiosity-and-developmental-learning/papers_text/12-nguyen-oudeyer-2014-sgim.txt](../../../research/curiosity-and-developmental-learning/papers_text/12-nguyen-oudeyer-2014-sgim.txt)
- **Executable Experiment & Saved Results**:
  - Experiment Script: [research/curiosity-and-developmental-learning/12-nguyen-oudeyer-2014-sgim/run_sgim_d_2014_experiments.py](../../../research/curiosity-and-developmental-learning/12-nguyen-oudeyer-2014-sgim/run_sgim_d_2014_experiments.py)
  - Saved Results Artifact: [research/curiosity-and-developmental-learning/12-nguyen-oudeyer-2014-sgim/results/sgim_d_2014_results.json](../../../research/curiosity-and-developmental-learning/12-nguyen-oudeyer-2014-sgim/results/sgim_d_2014_results.json)

---

## Stage 1 — Context (Primary Source & Algorithmic Formulation)

### 1.1 Why Pure Goal Babbling (`SAGG-RIAC`) and Pure Imitation Learning Need Each Other (Section 1.2–1.3, pp. 2–4)
Nguyen & Oudeyer (2014) address learning inverse models $L^{-1}: C \times T \to \Pi$ from a low-dimensional task/outcome space $T \subset [-1, 1]^2$ (where a fishing float lands on a 2D water surface) to a high-dimensional redundant motor primitive space $\Pi = [0, 1]^{25}$ (6-DOF arm with 4 Gaussian trajectory waypoints $u_1, u_2, u_3, u_4$ per joint plus duration $\delta$, Eq. 5, manipulating a 30-joint flexible fishing line, Figure 1):
- **Limitation of Pure Autonomous Goal Babbling (`SAGG-RIAC`)**: In high-dimensional continuous spaces ($\Pi = [0, 1]^{25}$), certain reachable regions of $T$ (such as the far-cast basin requiring coordinated whip timing across multiple joints) have tiny measure under random motor initialization, slowing discovery.
- **Limitation of Pure Imitation / Programming by Demonstration (`PbD`)**: Human demonstrations are sparse, cover only a specialized subset of the task space, and suffer from **correspondence mismatch** (Eq. 6 & Fig. 5, where projecting human trajectory $u_{Hd}(t)$ onto the robot's 4-Gaussian basis introduces approximation error).

### 1.2 The `SGIM-D` Architecture (Section 3 & Section 4, Algorithms 3.1–4.6)
1. **Dual-Level Social Interaction (Algorithm 3.1, lines 8–11)**: When a demonstration $(c_d, \zeta_d, \tau_d)$ arrives (every $N=30$ trials), `SGIM-D` operates on *both* levels:
   - **Task Space (Emulation)**: Registers the demonstrated outcome $\tau_d \in T$ as a candidate goal in its interest map.
   - **Policy Space (Imitation)**: Executes a short burst of local policy variations $\theta = \theta_d + \theta_{\text{rand}}$ (`Imitate Policy`, Algorithm 4.3) on its own body and stores the *true* robot outcome $(c, \theta, \tau)$ in `Memo` (overcoming correspondence bias!).
2. **Competence-Progress Task Partitioning (Algorithm 3.2, Eq. 3)**: Computes regional interest as the absolute competence derivative:
   $$\text{interest}_i = \frac{\left|\sum_{j=|R_i|-\zeta}^{|R_i|-\zeta/2} \gamma_j - \sum_{j=|R_i|-\zeta/2}^{|R_i|} \gamma_j\right|}{\zeta}$$
3. **Reliability-Weighted Local Inverse Optimization (Section 4.4.2, Algorithm 4.6 & Figure 6)**:
   Because the flexible 30-segment fishing line has **action-dependent heteroscedastic noise** (Figure 6: smooth casts have $\sigma \approx 0.005\text{--}0.016$, whereas jerky casts have $\sigma \approx 0.072$), `SGIM-D` selects local inverse neighbors by minimizing:
   $$\text{score}(h) = \text{dist}(\tau_h, \tau_g) + \alpha \cdot \text{var}_h \quad (\alpha = 0.5)$$

---

## Stage 2 — Direction (Testable Hypotheses)

1. **Experiment E1 (Section 5.1 & Section 6.1, Figures 7a & 8a — Ranking All 5 Algorithms on the 25D-to-2D Fishing Task)**:
   - Compare `learning_from_observation`, `random_exploration`, `sagg_riac`, `imitation_learning`, and `sgim_d` across 120 benchmark water-surface targets.
   - Verify the exact ranking from Section 6.1 (p. 13): `SGIM-D < Imitation_Learning & SAGG-RIAC < Random_Exploration < Learning_from_Observation`.
2. **Experiment E2 (Section 5.3, Figure 7c — Comparing the 3 Demonstrator Types in `SGIM-D`)**:
   - Compare `demonstrator_1_naive_high_variance` (randomly sampled `SAGG-RIAC` exemplars with high movement variance), `demonstrator_2_reliable_low_variance` (minimum-variance filtered `SAGG-RIAC` exemplars), and `demonstrator_3_human_kinesthetic` (smooth human demonstrations with small correspondence error).

---

## Stage 3 — Execution & Stage 4 — Outcome (Quantitative Results)

Running [run_sgim_d_2014_experiments.py](../../../research/curiosity-and-developmental-learning/12-nguyen-oudeyer-2014-sgim/run_sgim_d_2014_experiments.py) (`8` seeds $\times 450$ policies) produces:

### 4.1 Experiment E1: 5-Algorithm Benchmark on the 25D-to-2D Fishing Task
| Algorithm | Early Error (`t=150`) | Final Benchmark Distance Error (`t=450`) | Far-Cast Outer Basin Share |
|---|---|---|---|
| **`learning_from_observation`** | `0.4675` | `0.4402` (Worst: sparse + correspondence bias) | `0.0000` |
| **`random_exploration`** | `0.3430` | `0.3229` | `0.0300` (`3.0%` by chance) |
| **`sagg_riac` (Autonomous Goal Babbling)** | `0.3428` | `0.3025` | `0.1489` |
| **`imitation_learning` (Local Demo Perturbation)** | `0.3732` | `0.2621` | `0.2242` |
| **`sgim_d` (Socially Guided Intrinsic Motivation)** | `0.3440` | **`0.2085`** (Best — $35.4\%$ lower than Random) | **`0.1314`** (Balanced full-space coverage) |

### 4.2 Experiment E2: Comparing the 3 Demonstrator Types in `SGIM-D`
| Demonstrator Type | Final Benchmark Distance Error | Mean Executed Movement Noise $\sigma$ |
|---|---|---|
| **`demonstrator_1_naive_high_variance`** | `0.2823` | `0.0351` |
| **`demonstrator_2_reliable_low_variance`** | **`0.2525`** | **`0.0219`** |
| **`demonstrator_3_human_kinesthetic`** | **`0.2533`** | **`0.0215`** |

---

## Stage 5 — Growth (What We Learned & Critical Boundaries)

1. **Why Executing Imitated Policies (`SGIM-D`) Solves the Correspondence Problem That Breaks `Learning_from_Observation` (`0.2085` vs. `0.4402`)**:
   - Because the human teacher and the robot have different motor primitives (Eq. 6, Fig. 5), simply storing observed pairs $(\theta_d, \tau_{Hd})$ corrupts the inverse model (`0.4402`).
   - By executing $\theta = \theta_d + \theta_{\text{rand}}$ on the robot's own body (`Algorithm 4.3`) and recording the robot's true outcome $\tau = M(\theta)$, `SGIM-D` turns an imperfect human demonstration into an exact self-calibrated exemplar, then uses `SAGG-RIAC` goal babbling to generalize across undemonstrated regions of the water surface (`0.2085` vs. `0.2621` for pure `Imitation_Learning`).
2. **Why Human Demonstrations Help Even When Imperfect (`Demonstrator 3` vs. `Demonstrator 1`)**:
   - In a redundant 25D motor space with heteroscedastic dynamics, human demonstrators naturally produce smooth, low-jerk trajectories (`mean_noise_std = 0.0215` vs. `0.0351` for naive `SAGG-RIAC` `Demonstrator 1`), seeding the robot's memory in high-reproducibility regions of $\Pi$.
