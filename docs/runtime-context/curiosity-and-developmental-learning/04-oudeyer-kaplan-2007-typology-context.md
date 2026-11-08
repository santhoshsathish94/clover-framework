# Context File — Paper 04: Oudeyer & Kaplan (2007) — *What Is Intrinsic Motivation? A Typology of Computational Approaches*

- **Citation**: Pierre-Yves Oudeyer & Frédéric Kaplan (2007). *What Is Intrinsic Motivation? A Typology of Computational Approaches*. **Frontiers in Neurorobotics**, Vol. 1, Article 6, pp. 1–14 (`doi:10.3389/neuro.12.006.2007`).
- **Source PDF / Extracted Text**:
  - Primary PDF URL: `https://www.pyoudeyer.com/oudeyer-kaplan-neurorobotics.pdf`
  - Extracted Full Text (`77,429 bytes`): [research/curiosity-and-developmental-learning/papers_text/04-oudeyer-kaplan-2007-typology.txt](../../../research/curiosity-and-developmental-learning/papers_text/04-oudeyer-kaplan-2007-typology.txt)
- **Executable Experiment & Saved Results**:
  - Experiment Script: [research/curiosity-and-developmental-learning/04-oudeyer-kaplan-2007-typology/run_typology_experiments.py](../../../research/curiosity-and-developmental-learning/04-oudeyer-kaplan-2007-typology/run_typology_experiments.py)
  - Saved Results Artifact: [research/curiosity-and-developmental-learning/04-oudeyer-kaplan-2007-typology/results/typology_2007_results.json](../../../research/curiosity-and-developmental-learning/04-oudeyer-kaplan-2007-typology/results/typology_2007_results.json)

---

## Stage 1 — Context (Primary Source & Architectural Formalization)

### 1.1 Unifying Computational Framework: `P1` (World + Body) and `P2` (Internal Controller)
Oudeyer & Kaplan (2007) formalize intrinsic motivation within a single reinforcement-learning architecture split into two compartments (Figure 1, p. 4):
1. **`P1` (Physical Body + External Environment)**: Receives motor commands $M(t)$ and transitions from sensorimotor state $S(t)$ to $S(t+1)$.
2. **`P2` (Internal Neural/Algorithmic Controller)**: Observes internal + external state $S_i(t)$, selects actions via an action-selection module driven by total reward $r(t) = r_e(t) + r_i(t)$, and computes **intrinsic reward $r_i(t)$** purely from internal epistemic, competence, or dynamical variables.

Crucially, the authors prove that **"internal reward $\neq$ intrinsic reward"** (pp. 4, 10–11): homeostatic drives like blood-glucose/battery monitoring (`EnerM`, Eq. 37) or hardwired social-approval drives (`SocM`, Eq. 36) are *internal* (`P2`-computed) without being *intrinsic* in the psychological sense of Deci & Ryan (1985) or Berlyne (1960).

### 1.2 The 15-Model Formal Typology (Equations 1–35)
| Family | Sub-Family | Code | Equation | Core Intrinsic Reward $r_i(t)$ |
|---|---|---|---|---|
| **1. Knowledge-Based** | Distributional / Info-Theoretic | `UM` | Eq. (3) | $C \cdot (1 - P(e_k))$ (Uncertainty Motivation — visit rarest states) |
| **1. Knowledge-Based** | Distributional / Info-Theoretic | `IGM` | Eq. (4) | $H(E(t)) - H(E(t+1))$ (Information Gain / Bayesian Surprise, Itti & Baldi 2005; Fedorov 1972) |
| **1. Knowledge-Based** | Distributional / Info-Theoretic | `DSM` | Eq. (5) | $C \cdot \frac{1 - P(e_k)}{P(e_k)}$ (Distributional Surprise) |
| **1. Knowledge-Based** | Distributional / Info-Theoretic | `DFM` | Eq. (6) | $C \cdot P(e_k)$ (Distributional Familiarity) |
| **1. Knowledge-Based** | Predictive | `NM` | Eq. (9) | $C \cdot Er(t) = C \cdot \|\mathbf{SM}_p(t+1) - \mathbf{SM}(t+1)\|$ (Novelty / Max Prediction Error; Thrun 1992) |
| **1. Knowledge-Based** | Predictive | `ILNM` | Eq. (10) | $C \cdot \exp(-\|Er(t) - Er_\sigma\|^2)$ (Intermediate Level of Novelty; Berlyne 1960) |
| **1. Knowledge-Based** | Predictive | `LPM` | Eq. (11) | $\langle Er_{\mathcal{R}_n}(t-\theta)\rangle - \langle Er_{\mathcal{R}_n}(t)\rangle$ (Learning Progress Motivation; Schmidhuber 1991b; Oudeyer et al. 2007) |
| **1. Knowledge-Based** | Predictive | `SM` | Eq. (15) | $C \cdot \frac{Er(t)}{\mathcal{M}\Pi(\mathbf{SM}(t))}$ (Predictive Surprise: instantaneous error divided by expected error; Marsland et al. 2000) |
| **1. Knowledge-Based** | Predictive | `FM` | Eq. (17) | $\frac{C}{\langle Er_{\mathcal{R}_n}(t)\rangle}$ (Predictive Familiarity) |
| **2. Competence-Based** | Self-Determined Goal $g_k$ | `IM` | Eq. (21) | $C \cdot \langle la(g_k, t_g)\rangle$ (Maximizing Incompetence / Goal-Reaching Error) |
| **2. Competence-Based** | Self-Determined Goal $g_k$ | `CPM` | Eq. (24) | $\langle la(g_k, t_g - \theta)\rangle - \langle la(g_k, t_g)\rangle$ (Competence Progress / Csikszentmihalyi's *Flow*, 1991; Barto et al. 2004; Schembri et al. 2007) |
| **2. Competence-Based** | Self-Determined Goal $g_k$ | `CM` | Eq. (26) | $\frac{C}{\langle la(g_k, t_g)\rangle}$ (Familiarity of Competence) |
| **3. Morphological** | Dynamical Coordination | `SyncM` | Eq. (31) | $\text{Sync}(C_1, C_2, t, \theta)$ (Synchronicity Motivation; Barto et al. 2004 *Playroom*) |
| **3. Morphological** | Dynamical Coordination | `StabM` | Eq. (34) | $\frac{C}{\sigma(C_1, t, \theta) + \varepsilon}$ (Stability Motivation; Homeokinesis component, Der 2001) |
| **3. Morphological** | Dynamical Coordination | `VarM` | Eq. (35) | $C \cdot \sigma(C_1, t, \theta)$ (Variance Motivation; Homeokinesis instability component) |

---

## Stage 2 — Direction (Testable Hypotheses)

1. **Hypothesis 1 (Figure 7 Exploration vs. Organization Plane, p. 12)**:
   - Distributional rarity models (`UM`, `DSM`) maximize raw **Exploration Entropy** ($\approx 1.0$) by visiting all zones uniformly, but have mediocre **Organization Potential** ($50\%$ time wasted on trivial stasis and unlearnable noise).
   - First-derivative progress models (`IGM`, `LPM`, `CPM`) achieve the highest combined **Exploration + Developmental Organization** ($> 81\%$ time in learnable zones with structured Stage 1 $\to$ Stage 2 progression).
   - Error/variance maximizers (`NM`, `IM`, `VarM`) collapse into the unlearnable noise trap ($> 88\%$), while familiarity/stability models (`FM`, `CM`, `StabM`) freeze in the static familiar zone ($> 89\%$).
2. **Hypothesis 2 (Where Knowledge-Based `LPM` and Competence-Based `CPM` Diverge, p. 9)**:
   - On p. 9, Oudeyer & Kaplan point out that *"the capacity to predict what happens in a situation is only loosely coupled to the capacity to modify a situation in order to achieve a given self-determined goal."*
   - In an environment containing an unactuated deterministic spectacle (`passive_clockwork` — predictable trajectory, zero motor control authority) alongside a controllable motor task (`controllable_arm_skill`) and an unreachable target (`impossible_out_of_reach_goal`), **`LPM` gets distracted watching and learning to predict the unactuated `passive_clockwork`**, whereas **`CPM` ignores the unactuated spectacle and masters the `controllable_arm_skill`**.

---

## Stage 3 — Execution & Stage 4 — Outcome (Quantitative Results)

Running [run_typology_experiments.py](../../../research/curiosity-and-developmental-learning/04-oudeyer-kaplan-2007-typology/run_typology_experiments.py) (`10` seeds $\times 500$ steps in E1; `12` seeds $\times 450$ steps in E2; runtime `3.29 s`) yields:

### 4.1 Experiment E1: Benchmarking All 15 Typology Models (Figure 7 Quantitative Verification)
| Model | Family | `z0_static` | `z1_easy_learnable` | `z2_complex_learnable` | `z3_noise` | Exploration Entropy | Learnable Niche Share | Stage Progression Score |
|---|---|---|---|---|---|---|---|---|
| `UM` | Distributional | `0.2502` | `0.2500` | `0.2500` | `0.2498` | **`1.0000`** | `0.5000` | `0.2506` |
| `DSM` | Distributional | `0.2502` | `0.2500` | `0.2500` | `0.2498` | **`1.0000`** | `0.5000` | `0.2506` |
| `DFM` | Distributional | `0.3848` | `0.1222` | `0.2034` | `0.2896` | `0.3322` | `0.3256` | `0.1616` |
| `IGM` | Distributional | `0.1244` | `0.2818` | `0.5370` | `0.0568` | **`0.7945`** | **`0.8188`** | **`0.4498`** |
| `LPM` | Predictive | `0.1244` | `0.2818` | `0.5370` | `0.0568` | **`0.7945`** | **`0.8188`** | **`0.4498`** |
| `CPM` | Competence | `0.0574` | `0.3018` | `0.5888` | `0.0520` | **`0.7094`** | **`0.8906`** | **`0.5059`** |
| `SM` | Predictive | `0.0396` | `0.0424` | `0.7912` | `0.1268` | `0.5076` | `0.8336` | `0.4258` |
| `ILNM` | Predictive | `0.5586` | `0.3738` | `0.0332` | `0.0344` | `0.6277` | `0.4070` | `0.0438` |
| `SyncM` | Morphological | `0.0396` | `0.8900` | `0.0352` | `0.0352` | `0.3352` | `0.9252` (fixed $z_1$) | `0.4401` |
| `NM` | Predictive | `0.0422` | `0.0396` | `0.0380` | **`0.8802`** | `0.3578` | `0.0776` | `0.0373` |
| `IM` | Competence | `0.0422` | `0.0396` | `0.0380` | **`0.8802`** | `0.3578` | `0.0776` | `0.0373` |
| `VarM` | Morphological | `0.0422` | `0.0396` | `0.0380` | **`0.8802`** | `0.3578` | `0.0776` | `0.0373` |
| `FM` | Predictive | **`0.8936`** | `0.0362` | `0.0358` | `0.0344` | `0.3266` | `0.0720` | `0.0400` |
| `CM` | Competence | **`0.8936`** | `0.0362` | `0.0358` | `0.0344` | `0.3266` | `0.0720` | `0.0400` |
| `StabM` | Morphological | **`0.8936`** | `0.0362` | `0.0358` | `0.0344` | `0.3266` | `0.0720` | `0.0400` |

### 4.2 Experiment E2: Knowledge-Based `LPM` vs. Competence-Based `CPM` Divergence
| Model | `passive_clockwork` (Predictable, Uncontrollable) | `controllable_arm_skill` (Controllable Goal) | `impossible_out_of_reach_goal` | Final Arm Competence |
|---|---|---|---|---|
| **`LPM` (Prediction Progress)** | **`57.87%`** (spectator trap!) | `35.94%` | `6.19%` | `0.9471` |
| **`CPM` (Competence Progress / Flow)** | `6.20%` | **`87.61%`** | `6.19%` | **`0.9992`** |
| **`IM` (Incompetence Maximization)** | `6.06%` | `6.67%` | **`87.28%`** | `0.4181` |

---

## Stage 5 — Growth (What We Learned & Critical Boundaries)

1. **The "Passive Spectator Trap" of Pure Prediction Progress (`LPM`)**:
   - Experiment E2 demonstrates why a robot or child driven solely by forward-model prediction progress (`LPM`) can spend **`57.87%` of its developmental budget** passively watching an external deterministic process (`passive_clockwork`, such as a ceiling fan, pendulum, or screen animation) whose transitions $S(t) \to S(t+1)$ are learnable even though the agent has **zero motor control authority** over it.
   - By contrast, **Competence Progress (`CPM`, Eq. 24)** measures the derivative of **goal-achievement error** $la(g_k, t_g) = \|g_k - \tilde{g}_k\|$, immediately filtering out unactuated spectacles (`6.20%`) and concentrating **`87.61%`** of practice on controllable motor skills.
2. **Why `ILNM` (Fixed Intermediate Novelty, Eq. 10) Fails Without Adaptive Thresholds**:
   - Berlyne's (1960) fixed-setpoint intermediate novelty (`ILNM`, targeting a constant error $Er_\sigma = 0.22$) gets stuck between `z0_static` (`55.86%`) and `z1_easy_learnable` (`37.38%`), spending only `3.32%` on `z2_complex_learnable` once $z_2$'s initial error exceeds $Er_\sigma$. Progress derivatives (`LPM`, `CPM`, `IGM`) avoid choosing a fixed target error $Er_\sigma$ by rewarding the *negative slope* of error at any baseline scale.
