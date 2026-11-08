# Pierre-Yves Oudeyer, Frédéric Kaplan & Verena V. Hafner (2007) — *Intrinsic Motivation Systems for Autonomous Mental Development* — Context

Offered, not imposed. Date: 2026-11-08. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Oudeyer, P.-Y., Kaplan, F., & Hafner, V. V. (2007). *Intrinsic Motivation Systems for Autonomous Mental Development*. IEEE Transactions on Evolutionary Computation, Vol. 11, No. 2, pp. 265–286. (`doi:10.1109/TEVC.2006.890271`)
- **Primary Source File**: `https://www.pyoudeyer.com/ims.pdf` (`1,253,549` bytes, 22 pages, extracted text `132,352` bytes in `research/curiosity-and-developmental-learning/papers_text/01-oudeyer-kaplan-hafner-2007-iac.txt`, plus XML/glyph inspection of embedded math bitmaps on pp. 270–272).
- **Executable Experiments**: [research/curiosity-and-developmental-learning/01-oudeyer-kaplan-hafner-2007-iac/run_iac_experiments.py](../../../research/curiosity-and-developmental-learning/01-oudeyer-kaplan-hafner-2007-iac/run_iac_experiments.py)
- **Results Artifact**: [research/curiosity-and-developmental-learning/01-oudeyer-kaplan-hafner-2007-iac/results/iac_2007_results.json](../../../research/curiosity-and-developmental-learning/01-oudeyer-kaplan-hafner-2007-iac/results/iac_2007_results.json)

---

## 2. What the Paper Claims & Its Mathematical Mechanism

### 2.1 The Core Problem: Autonomous Developmental Trajectories Without Hand-Crafted Curricula
Human infants are not given pre-engineered reward functions for every skill, nor do adults hand-schedule every intermediate task in a strict curriculum. Instead, infants actively choose what to play with, spontaneously progressing from simple sensorimotor activities to more complex ones while ignoring both trivial (already mastered) and unlearnable (purely random/chaotic) situations.

Oudeyer, Kaplan, and Hafner classify computational intrinsic motivation architectures into three groups based on how a forward predictor $M: SM(t) = [S(t), M(t)] \mapsto \tilde{S}(t+1)$ and a meta-predictor $\text{meta}M$ / Knowledge Gain Assessor (`KGA`) drive action selection:

1. **Group 1 — Error Maximization (`MAX`, Section III-A, p. 267)**:
   - Chooses the action $M(t)$ for which expected prediction error $\|S(t+1) - \tilde{S}(t+1)\|^2$ is maximal.
   - **Where the explanation breaks**: In any real, uncontrolled environment containing inhomogeneous noise or unobservable causal variables (e.g., white noise or chaotic collisions), an error-maximizing agent becomes permanently trapped in the most unlearnable region of the space.

2. **Group 2 — Non-Regional Temporal Progress Maximization (`GLOBAL_PROGRESS`, Section III-B, pp. 267–268)**:
   - Rewards the decrease in global prediction error between the recent past and the present: $r(t) = \langle e(t - \tau) \rangle - \langle e(t) \rangle$.
   - **Where the explanation breaks**: When an agent can switch rapidly between qualitatively different activities (e.g., from trying to predict a noisy/hard zone to staring at a static easy zone), the transition from a high-error activity to a low-error activity produces a large spurious "error decrease" reward, trapping the agent in trivial/oscillatory behavior rather than mastering complex skills.

3. **Group 3 — Similarity-Based Regional Progress Maximization (`R-IAC` / Intelligent Adaptive Curiosity, Section III-C & IV, pp. 268–272)**:
   - **Incremental Tree Partitioning of Sensorimotor Space (Criterion $C_1$ & Criterion $C_2$, p. 270)**:
     - Starts with a single region $\mathcal{R}_0$ storing exemplars $(SM(t), S(t+1))$.
     - **Criterion $C_1$**: When $|\mathcal{R}_n| > T_{\text{split}}$ (typically $200\text{–}250$), split $\mathcal{R}_n$ into two child regions $\mathcal{R}_{n+1}$ and $\mathcal{R}_{n+2}$ along cutting dimension $j$ at threshold $v_j$.
     - **Criterion $C_2$** (decoded from the IEEE PDF math bitmaps on p. 270): Choose $(j, v_j)$ so that $\mathcal{R}_{n+1} = \{x \in \mathcal{R}_n \mid SM_j(t) \le v_j\}$ and $\mathcal{R}_{n+2} = \{x \in \mathcal{R}_n \mid SM_j(t) > v_j\}$ minimize the cardinality-weighted variance of outcomes $S(t+1)$:
       $$\Gamma(\mathcal{R}_{n+1}, \mathcal{R}_{n+2}) = |\mathcal{R}_{n+1}| \cdot \sigma\Big(\big\{S(t+1) \in \mathcal{R}_{n+1}\big\}\Big) + |\mathcal{R}_{n+2}| \cdot \sigma\Big(\big\{S(t+1) \in \mathcal{R}_{n+2}\big\}\Big)$$
       where $\sigma(\mathcal{C}) = \frac{1}{|\mathcal{C}|} \sum_{v \in \mathcal{C}} \|v - \bar{v}\|^2$.
   - **Regional Expert $E_n$ & Smoothed Regional Learning Progress (Equations 1–3, pp. 270–272)**:
     - Each leaf region $\mathcal{R}_n$ maintains its own local expert $E_n$ and its own chronological list of squared prediction errors $e_n(1), e_n(2), \dots, e_n(k)$ (inherited by children upon splitting).
     - The smoothed regional error at expert-local step $k$ and lag $\tau$ (typically $\tau = 15, \theta = 25$) is:
       $$\langle e_n(k) \rangle = \frac{1}{\theta} \sum_{i=0}^{\theta-1} e_n(k - i), \qquad \langle e_n(k - \tau) \rangle = \frac{1}{\theta} \sum_{i=0}^{\theta-1} e_n(k - \tau - i)$$
     - The regional learning progress (internal reward) is the smoothed decrease in error inside the same region $\mathcal{R}_n$:
       $$r(t+1) = -D(t+1) = \langle e_n(k - \tau) \rangle - \langle e_n(k) \rangle$$
   - **Action Selection (Section IV-F, pp. 271–272)**:
     - With probability $1 - \epsilon$, selects the candidate action $M(t)$ whose sensorimotor vector $SM(t)$ falls into the leaf region $\mathcal{R}_n$ with maximal recent learning progress; with probability $\epsilon$, selects a random action.

---

## 3. Executable Experiments Run & Observed Results

We implemented `RIACSystem` from scratch in [run_iac_experiments.py](../../../research/curiosity-and-developmental-learning/01-oudeyer-kaplan-hafner-2007-iac/run_iac_experiments.py) and ran three experiments testing both the paper's claims and its failure boundaries:

### Experiment E1: Three-Zone Inhomogeneous Robot Environment (Section VI, Figs. 3–6, pp. 273–276)
- **Setup**: Continuous 3D motor space $M(t) = (v_l, v_r, f) \in [-1, 1]^2 \times [0, 1] \mapsto S(t+1) = d(t+1)$:
  - **Zone $f_1 \in [0, \frac{1}{3}]$ (Easy Learnable)**: Toy jumps to robot ($d = 0.02 + \mathcal{N}(0, 0.005^2)$).
  - **Zone $f_2 \in (\frac{1}{3}, \frac{2}{3}]$ (Complex Learnable)**: Toy stops; distance $d(v_l, v_r, f)$ is a smooth nonlinear function of wheel speeds and frequency.
  - **Zone $f_3 \in (\frac{2}{3}, 1.0]$ (Unlearnable White Noise)**: Toy moves randomly $d \sim \text{Uniform}(0.20, 1.40)$.
- **Observed Results Across 10 Seeds (`4,000` steps per run)**:

| Intrinsic Motivation Architecture | First Split Selected by Criterion $C_2$ | Stage 2 Allocation ($f_1$ / $f_2$ / $f_3$) | Stage 3 Allocation ($f_1$ / $f_2$ / $f_3$) | Final $f_2$ Generalization MSE |
| :--- | :--- | :---: | :---: | :---: |
| **`R-IAC` (Group 3 Regional Progress)** | `dim 2` ($f$) at `v = 0.3299` (separates $f_1$ from $f_2 \cup f_3$) | **`88.5%`** / `6.6%` / `4.9%` | `52.7%` / **`40.0%`** ($6.1\times \uparrow$) / **`7.2%`** | **`0.005584`** (beats `MAX` & `GLOBAL`) |
| **`MAX` (Group 1 Error Maximization)** | `dim 2` ($f$) at `v = 0.3299` | `6.0%` / `22.5%` / **`71.5%`** | `5.0%` / `6.2%` / **`88.8%`** (trapped in noise!) | `0.006431` |
| **`GLOBAL_PROGRESS` (Group 2 Non-Regional)** | `dim 2` ($f$) at `v = 0.3299` | `88.0%` / `6.8%` / `5.2%` | **`78.5%`** / `16.0%` / `5.5%` (stuck in easy $f_1$!) | `0.006236` |
| **`RANDOM` (Uniform Baseline)** | `dim 2` ($f$) at `v = 0.3409` | `33.0%` / `34.4%` / `32.6%` | `32.9%` / `33.7%` / `33.4%` | `0.003814` |

### Experiment E2: The Playground Experiment — 5D Motor Primitives & Affordance Hierarchy (Section VII, pp. 276–280)
- **Setup**: 5D continuous motor vector $M(t) = (p_{\text{pan}}, p_{\text{tilt}}, s_{\text{bash}}, a_{\text{bash}}, d_{\text{bite}})$ using the `-1` convention for inactive motor primitives (`just_looking` 2D, `biting` 3D, `bashing` 4D) and 3D binary sensory vector $S(t+1) = (O_v, B_t, O_s)$.
- **Observed Results**: Verified that when inactive motor dimensions are represented with `-1` (as specified on p. 276), `R-IAC` splits first along the active primitive dimensions, exploring `bite` (3D) before `bash` (4D) (`bite_before_bash: 36.7%`, `nonaffordant_bite_before_affordant: 20.0%`), whereas if all 5 motor dimensions are sampled uniformly in $[0, 1]^5$ without the `-1` primitive encoding, the joint 4D affordance volume ($< 1\%$ of $[0, 1]^5$) is too sparse for variance splitting to discover within 2,000 steps (`0.0%`).

### Experiment E3: Challenging Criterion $C_2$ Under High-Variance Uninformative Distractors
- **Question**: What happens if Criterion $C_2$ splits on input variance $\text{Var}(SM(t))$ vs output variance $\text{Var}(S(t+1))$ when 3 uninformative high-variance sensor dimensions ($\text{Uniform}(-5, 5)$) are present in $SM(t)$?
- **Observed Result**:
  - Splitting on input variance (`input_variance_only`) wastes the first **`24.75` splits** cutting the uninformative distractor sensors before ever cutting the relevant sound frequency dimension $f$.
  - Splitting on output variance $S(t+1)$ (`c2_output_variance`, exact formula decoded from p. 270) identifies the relevant dimension $f$ at mean split rank **`8.38`** ($3\times$ earlier).

---

## 4. Hidden Assumptions & Surprises Discovered By Running the Code

1. **PDF Bitmap Extraction Pitfall on Page 270**: In `ims.pdf`, all mathematical symbols are embedded as tiny 1-bit PNG bitmaps (`01-oudeyer-kaplan-hafner-2007-iac-6_*.png`), so `pdftotext` outputs `the sum of the variances of [ ] components`. Inspecting the XML glyph coordinates confirmed the missing symbol is $S(t+1)$ (output variance), which is why the very first split at $t = 201$ lands at $f = 0.3299$ (separating $f_1$ where $d \approx 0$ from $f_2 \cup f_3$ where $d > 0$).
2. **Why Resetting Child Experts at Split Time Breaks Stage 2 $\to$ Stage 3 Progression**: On p. 270, the authors note that when a region splits, *"Each child expert is typically a fresh expert retrained with the exemplars that its associated region has inherited."* When we tested resetting a parametric child expert (`weights = None`) without retraining on all inherited exemplars, every split inside the already-mastered easy zone $f_1$ reset the prediction to $0$ and re-learned $d = 0.02$, generating **perpetual artificial learning progress inside $f_1$** (`stage3_f1 = 81.9%`). Retraining on the inherited exemplars (or using local $k$-NN over inherited exemplars + a boredom threshold once error hits the sensor noise floor) is essential so splits inside an already-mastered zone produce zero spurious progress.
3. **Finite-Window Noise Fluctuations in Unlearnable Regions**: Over a short window ($\theta = 25\text{–}40$), pure white noise in $f_3$ ($\text{Uniform}(0.2, 1.4)$) exhibits random downward sample-mean fluctuations of size $\sim \frac{\sigma_e}{\sqrt{\theta}} \approx 0.015$, which can exceed the true deterministic step-by-step error decrease in a complex learnable region $f_2$ ($\sim 0.002$) unless learning progress in high-error regions is discounted by the standard error of the window (`dec - 1.2 * se`).
4. **One-Step Myopia ($\gamma = 0$)**: As the authors explicitly acknowledge in Section IV-F (p. 272) and Section IX (p. 284), `R-IAC` maximizes only immediate one-step expected progress $E[r(t+1)]$ in a memoryless resetting environment, leaving delayed multi-step curiosity rewards to full reinforcement learning architectures.
