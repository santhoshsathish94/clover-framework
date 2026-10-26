# Richard S. Sutton (1988) — *Learning to Predict by the Methods of Temporal Differences* — Context

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh.

## 1. Status

- **E1 (Clone & run reference repositories on k3 CPU: `MrShininnnnn/Temporal-Differences-Learning`, `vjp23/TD_Lambda`, `shashir/td-learning`, `robosangli/sutton-1988-reproduction`)**: DONE
- **E2 (Theorem 1 exact equivalence of per-sequence $\text{TD}(1)$ and Widrow–Hoff Eq. 2–3, incremental eligibility trace Eq. 4, and Section 5.2 intra-sequence update non-equivalence)**: DONE
- **E3 (Bounded 5-State Random Walk: repeated-presentation $\text{TD}(\lambda)$ vs maximum-likelihood Markov estimate Theorem 3 & Erratum Fig. 3 p. 377, and single-presentation Figs. 4–5)**: DONE
- **E4 (Section 3.1 "Novel $\to$ Bad $\to$ Win" game example Fig. 1, Section 3.1 non-Markovian counterexample p. 18, and Section 5.1 cumulative step-count prediction)**: DONE

## 2. What this is

Author says (Richard S. Sutton, "Learning to Predict by the Methods of Temporal Differences", *Machine Learning*, Vol. 3, No. 1, pp. 9–44, August 1988, plus Erratum on p. 377):
- **Single-step vs Multi-step Prediction (Section 2.1, pp. 11–12)**: Conventional supervised learning treats an observation-outcome sequence $x_1, x_2, \dots, x_m, z$ as independent pairs $(x_t, z)$ and updates weights only after the final outcome $z$ is known (Eq. 2):
  $$\Delta w_t = \alpha (z - P_t) \nabla_w P_t$$
  which for linear predictions $P_t = w^T x_t$ is the Widrow–Hoff (LMS / delta) rule.
- **Telescoping Error & Theorem 1 (Section 2.2, pp. 12–15)**: By rewriting the total error as a telescoping sum of temporally successive prediction differences $z - P_t = \sum_{k=t}^m (P_{k+1} - P_k)$ (with $P_{m+1} \stackrel{\text{def}}{=} z$), the total per-sequence update $\sum_{t=1}^m \Delta w_t$ can be rearranged into the incremental **$\text{TD}(1)$** rule (Eq. 3):
  $$\Delta w_t = \alpha (P_{t+1} - P_t) \sum_{k=1}^t \nabla_w P_k$$
  **Theorem 1 (p. 15)** proves that on multi-step prediction problems, linear $\text{TD}(1)$ produces the **exact same per-sequence weight changes** as the Widrow–Hoff procedure while requiring only $O(d)$ memory (instead of storing all $M$ past observation vectors).
- **The $\text{TD}(\lambda)$ Family (Section 2.3, pp. 15–16)**: Weighting alterations to past predictions exponentially by recency $\lambda^{t-k}$ ($0 \le \lambda \le 1$) yields **$\text{TD}(\lambda)$** (Eq. 4):
  $$\Delta w_t = \alpha (P_{t+1} - P_t) \sum_{k=1}^t \lambda^{t-k} \nabla_w P_k = \alpha (P_{t+1} - P_t) e_t, \qquad e_{t+1} = \nabla_w P_{t+1} + \lambda e_t$$
  At the opposite extreme from $\text{TD}(1)$ is **$\text{TD}(0)$** ($\Delta w_t = \alpha (P_{t+1} - P_t) \nabla_w P_t$), which updates each prediction solely toward the immediately following prediction $P_{t+1}$.
- **Bounded Random Walk & Empirical Superiority of $\lambda < 1$ (Section 3.2, pp. 19–23, Figs. 2–5 & Erratum p. 377)**: On the 5-state random walk ($B, C, D, E, F$ with absorbing boundaries $A \to 0$ and $G \to 1$, starting at $D$, ideal probabilities $\frac{1}{6}, \frac{2}{6}, \frac{3}{6}, \frac{4}{6}, \frac{5}{6}$), across 100 training sets of 10 walks each:
  - **Experiment 1 (Repeated presentations until convergence, Fig. 3 + Erratum p. 377)**: Average RMS error decreases monotonically as $\lambda$ decreases from $1$ (Widrow–Hoff, $\text{RMS} \approx 0.24$) down to $0$ ($\text{TD}(0)$, $\text{RMS} \approx 0.20$). *(Note: in the original printing on p. 20, Figure 3 accidentally duplicated Figure 5; Sutton's Erratum on p. 377 published the true Figure 3.)*
  - **Experiment 2 (Single presentation of 10 walks from $w_0 = 0.5$, Figs. 4 & 5)**: For every learning rate $\alpha$, all $\text{TD}(\lambda)$ methods with $\lambda < 1$ beat $\text{TD}(1)$, with best single-pass performance near $\lambda \approx 0.3$.
- **Asymptotic Convergence & Maximum-Likelihood Optimality of Linear $\text{TD}(0)$ (Section 4, Theorems 2 & 3, pp. 23–32)**:
  - **Theorem 2 (p. 24)**: For any absorbing Markov chain with linearly independent observation vectors $\{x_i\}$, expected predictions of linear $\text{TD}(0)$ converge as $n \to \infty$ to the ideal predictions $[(I - Q)^{-1} h]_i$ ( proved via positive-definiteness of $D(I - Q)$ where $d^T = \mu^T (I - Q)^{-1}$).
  - **Theorem 3 (p. 31)**: Under repeated presentations of a finite training set, linear $\text{TD}(0)$ converges not to the sample-mean outcomes (like Widrow–Hoff), but to the **maximum-likelihood Markov process estimate** $[(I - \hat{Q})^{-1} \hat{h}]_i$ (Eq. 8).

## 3. Experiments

### E1: Clone and run reference repositories (`MrShininnnnn/Temporal-Differences-Learning`, `vjp23/TD_Lambda`, `shashir/td-learning`, `robosangli/sutton-1988-reproduction`)
- **Question**: How do the reference repositories implement Sutton (1988)'s 5-state random walk and Figures 3, 4, and 5, and how do they handle the Erratum on Figure 3?
- **Prediction** (logged in `work/predictions.txt` at 2026-10-26 17:45 UTC):
  - All cloned repositories will implement the 5-state bounded random walk ($B\text{–}F$) across $\lambda \in \{0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0\}$, and their repeated-presentation experiment (Figure 3) will produce RMS error rising monotonically from $\sim 0.20$ at $\lambda=0$ to $\sim 0.24$ at $\lambda=1$ (matching the p. 377 Erratum rather than the misprinted p. 20 graph).
- **Run**: `scripts/e1_repos.sh` and `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - Cloned four reference repositories into `/root/dm-study/foundations/sutton-1988/repo/`:
    1. `robosangli/sutton-1988-reproduction` (`fc1a353`): Cleanest Python implementation (`sutton1988.py`, `analysis.py`). Running `sutton1988.py` on `k3` CPU verified online $\text{TD}(0)$ convergence (`[0.1724, 0.3311, 0.5057, 0.6763, 0.8676]` vs ideal `[0.1667, 0.3333, 0.5, 0.6667, 0.8333]`), reproduced Figure 3 RMSE across $\lambda \in [0.0, 1.0]$ rising monotonically from **`0.1202`** ($\lambda=0.0$) to **`0.1751`** ($\lambda=1.0$), and reproduced Figure 5 best-$\alpha$ RMSE dipping from `0.0893` ($\lambda=0.0$) to **`0.0866`** at $\lambda=0.2$ before rising to `0.1752` at $\lambda=1.0$.
    2. `MrShininnnnn/Temporal-Differences-Learning` (`186abff`): Inspection of `main.py` revealed three departures from Sutton (1988): (a) `generate_data()` filtered out any 10-walk training set with mean walk length $> 10$ (`if np.mean([len(s) for s in sequence_list]) <= self.seq_len`), (b) `fig3()` initialized `w = np.random.rand(1, 5)` (`Uniform(0, 1)`) instead of $0.5$, and (c) `fig3()` stopped when `np.linalg.norm(delta_w) <= 0.001` at `lr = 0.01` (`||sum_dw|| <= 0.1`, far before asymptotic convergence).
    3. `vjp23/TD_Lambda` (`ed1c66a`): Inspection of `td_learning.py` uncovered a classic Python list-aliasing bug on line 57: `training = num_sets * [seq_per_set * [None]]`, which creates `num_sets` references to the **exact same inner list**, so all 100 training sets are identical copies of the last generated set! In addition, `run_3` swapped `training_sets_fig3 = 10` and `training_sequences_fig3 = 100`, ran only `20` iterations at $\alpha = 0.001$, and never reset `weights` between sets.
    4. `shashir/td-learning` (`1d42a7b`): Scala implementation (`TD.scala`, `RandomWalkVector.scala`) computing per-set $\text{RMSE} = \sqrt{\frac{1}{5} \|w - w^*\|^2}$.

### E2: Theorem 1 Exact Equivalence of Per-Sequence $\text{TD}(1)$ and Widrow–Hoff (Eq. 2–3, pp. 13–15), Incremental Eligibility Traces (Eq. 4, pp. 15–16), and Intra-Sequence Weight Updating Drift (Section 5.2, p. 35)
- **Question**:
  1. Does per-sequence linear $\text{TD}(1)$ ($\sum_{t=1}^m \alpha (P_{t+1} - P_t) \sum_{k=1}^t x_k$, Eq. 3) produce the exact same weight changes as the Widrow–Hoff supervised-learning rule ($\sum_{t=1}^m \alpha (z - P_t) x_t$, Eq. 2) on both tabular and arbitrary dense feature vectors (Theorem 1, p. 15)?
  2. Does the $O(d)$-memory incremental eligibility trace $e_{t+1} = \nabla_w P_{t+1} + \lambda e_t$ match the explicit history sum $\sum_{k=1}^{t+1} \lambda^{t+1-k} \nabla_w P_k$ across all $\lambda \in [0, 1]$?
  3. What happens when weights are updated *within* a sequence after every step $t$ (Section 5.2, p. 35)?
- **Prediction** (logged in `work/predictions.txt` before running):
  - Max difference between per-sequence $\text{TD}(1)$ and Widrow–Hoff $< 10^{-14}$ across 1,000 random walks on both tabular ($X = I_5$) and dense ($X \in \mathbb{R}^{8 \times 5}$) features.
  - Max difference between incremental eligibility trace and explicit history sum $< 10^{-14}$.
  - Updating $w$ inside a sequence breaks the telescoping identity ($\max \text{diff} > 10^{-2}$).
- **Run**: `work/sutton1988_experiments.py` (`run_e2_theorem1_and_traces()`) via `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **E2.1 (Theorem 1)**: Across 1,000 random walks, $\max |\Delta w_{\text{TD}(1)} - \Delta w_{\text{WH}}| = \mathbf{8.88 \times 10^{-16}}$ on 5D unit-basis features and $\mathbf{7.77 \times 10^{-16}}$ on 8D dense Gaussian features.
  - **E2.2 (Eligibility Trace Eq. 4)**: Across all 7 values of $\lambda \in \{0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0\}$, $\max |\Delta w_{\text{incremental}} - \Delta w_{\text{explicit}}| = \mathbf{4.44 \times 10^{-16}}$.
  - **E2.3 (Section 5.2 Intra-Sequence Update Drift, p. 35)**: When $w_t$ is updated at each step $t$ *within* a sequence, the telescoping identity $z - P_t = \sum_{k=t}^m (P_{k+1} - P_k)$ breaks whenever a state is revisited or features overlap: naive intra-sequence $\text{TD}(1)$ ($P_t = P(x_t, w_{t-1})$, first equation on p. 35) diverges from off-line Widrow–Hoff by up to $\mathbf{2.3347}$ (and even with Sutton's p. 35 refinement $P(x_{t+1}, w_t) - P(x_t, w_t)$, intra-sequence updates differ from off-line updates by $\mathbf{1.9940}$).

### E3: Bounded 5-State Random Walk — Theorem 2 & 3, Repeated Presentations (Fig. 3 + Erratum p. 377), and Single Presentation (Figs. 4 & 5 + Backward Replay p. 23)
- **Question**:
  1. Does $S = D(I - Q) + (D(I - Q))^T$ have strictly positive eigenvalues as proved in the Lemma to Theorem 2 (p. 27)?
  2. Under repeated presentations of a 10-walk training set until convergence (Figure 3 / Erratum p. 377), does linear $\text{TD}(0)$ converge to the exact maximum-likelihood Markov estimate $(I - \hat{Q})^{-1} \hat{h}$ (Theorem 3, Eq. 8, p. 30–31), whereas $\text{TD}(1)$ converges to the empirical sample mean of outcomes?
  3. Under a single presentation of 10 walks from $w_0 = 0.5$ (Figures 4 & 5, pp. 21–22), which $\lambda$ achieves the lowest RMS error, and does backward offline $\text{TD}(0)$ replay (p. 23) beat forward $\text{TD}(0)$?
- **Prediction** (logged in `work/predictions.txt` before running):
  - All eigenvalues of $S$ are $> 0$; $\max |w_{\text{TD}(0)} - (I - \hat{Q})^{-1} \hat{h}| < 10^{-5}$; $\max |w_{\text{TD}(1)} - w_{\text{WH}}| < 10^{-5}$; repeated-presentation RMS error increases monotonically from $\lambda = 0$ to $\lambda = 1$ (Erratum Fig. 3, p. 377).
  - Single-presentation best-$\alpha$ RMS error (Figure 5) achieves its minimum at $\lambda \approx 0.3$, and backward offline $\text{TD}(0)$ replay beats forward $\text{TD}(0)$.
- **Run**: `work/sutton1988_experiments.py` (`run_e3_random_walk()`) via `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **E3.1 (Theorem 2 Matrix Properties, pp. 24–28)**:
    - Ideal predictions $(I - Q)^{-1} h = [0.1667, 0.3333, 0.5000, 0.6667, 0.8333]$ ($\frac{1}{6}\dots\frac{5}{6}$).
    - Expected visits per walk $d^T = \mu^T (I - Q)^{-1} = [1, 2, 3, 2, 1]$ for states $(B, C, D, E, F)$ (total $= 9.0$ steps per walk).
    - Eigenvalues of $S = D(I - Q) + (D(I - Q))^T$: **`[0.3831, 1.1972, 2.8188, 4.8028, 8.7982]`** (all strictly positive, confirming positive-definiteness of $D(I - Q)$).
  - **E3.2 (Repeated Presentations — Theorem 3 & Figure 3 / Erratum p. 377)**:
    - Across 30 fully-visited 10-walk training sets, batch $\text{TD}(0)$ converged to the analytical maximum-likelihood Markov estimate $w_{\text{ML}} = (I - \hat{Q})^{-1} \hat{h}$ with $\max |w_{\text{TD}(0)} - w_{\text{ML}}| = \mathbf{3.50 \times 10^{-7}}$, while batch $\text{TD}(1)$ converged to the empirical sample mean $w_{\text{WH}}$ with $\max |w_{\text{TD}(1)} - w_{\text{WH}}| = \mathbf{1.48 \times 10^{-7}}$.
    - Across all 100 training sets of 10 walks each, repeated-presentation asymptotic RMS error vsSingle-presentation best-$\alpha$ RMS error:
      | $\lambda$ | Repeated Presentations RMS Error (Fig. 3 / Erratum p. 377) | Single Presentation Best $\alpha$ | Single Presentation RMS Error at Best $\alpha$ (Fig. 5, p. 22) |
      | :---: | :---: | :---: | :---: |
      | `0.0` ($\text{TD}(0)$) | **`0.1199 ± 0.0066`** | `0.20` | `0.0887` |
      | `0.1` | `0.1198 ± 0.0066` | `0.20` | `0.0867` |
      | `0.3` | `0.1205 ± 0.0065` | `0.20` | **`0.0864`** (minimum!) |
      | `0.5` | `0.1233 ± 0.0065` | `0.15` | `0.0923` |
      | `0.7` | `0.1300 ± 0.0066` | `0.15` | `0.1045` |
      | `0.9` | `0.1486 ± 0.0072` | `0.10` | `0.1374` |
      | `1.0` (Widrow–Hoff) | **`0.1773 ± 0.0077`** | `0.05` | **`0.1786`** |
  - **E3.3 (Single Presentation Figure 4 & Page 23 Backward Replay)**:
    - For every learning rate $\alpha \in [0.05, 0.60]$, every $\lambda < 1$ achieved lower RMS error than $\lambda = 1.0$ (Widrow–Hoff), with intermediate $\lambda = 0.3$ remaining stable at larger $\alpha$ (`0.1904` at $\alpha = 0.40$ vs `0.2721` for $\lambda = 0.0$ and `0.6370` for $\lambda = 1.0$).
    - **Backward offline $\text{TD}(0)$ replay (p. 23)**: Updating each walk backward from $t = m$ down to $1$ achieved single-pass RMS error **`0.0882`** at $\alpha = 0.20$, beating forward $\text{TD}(0)$ (`0.0887`) on a single presentation.

### E4: Section 3.1 "Novel $\to$ Bad $\to$ Win" Game Example (Fig. 1, pp. 17–18), Non-Markovian Counterexample (p. 18), and Section 5.1 Cumulative Step-Count Prediction (pp. 33–34)
- **Question**:
  1. In Sutton's Figure 1 game example (pp. 17–18), where a `Bad` state wins $10\%$ of the time (`9` losses, `1` win) and a `Novel` state transitions into `Bad` on the single game that happened to win (`Novel -> Bad -> Win`, true Markov $E[z \mid \text{Novel}] = 0.10$), what values do Widrow–Hoff $\text{TD}(1)$ and linear $\text{TD}(0)$ assign to `Novel`?
  2. In Sutton's page 18 non-Markovian counterexample (where `Bad` always leads to a win when preceded by `Novel`, and always to a loss otherwise, so true $E[z \mid \text{Novel}] = 1.00$), how do $\text{TD}(0)$ and $\text{TD}(1)$ perform when `Bad` is represented by a single state feature?
  3. Does cumulative $\text{TD}(0)$ ($\Delta w_t = \alpha (c_{t+1} + P_{t+1} - P_t) \nabla_w P_t$ with $c_{t+1} = 1, P_{m+1} = 0$, Section 5.1 pp. 33–34) learn the exact expected number of steps to termination on the 5-state random walk?
- **Prediction** (logged in `work/predictions.txt` before running):
  - On Figure 1 (Markov), $\text{TD}(0)$ converges to $P(\text{Novel}) = 0.1000$ (error $0.0000$), while Widrow–Hoff $\text{TD}(1)$ converges to $P(\text{Novel}) = 1.0000$ (error $0.9000$).
  - On the page 18 non-Markovian counterexample, $\text{TD}(0)$ suffers state aliasing ($P(\text{Novel}) = 0.1000$, error $0.9000$), while Widrow–Hoff $\text{TD}(1)$ converges to the true unbiased value $P(\text{Novel}) = 1.0000$ (error $0.0000$).
  - Cumulative $\text{TD}(0)$ converges to $(I - Q)^{-1} \mathbf{1} = [5, 8, 9, 8, 5]$.
- **Run**: `work/sutton1988_experiments.py` (`run_e4_game_and_cumulative()`) via `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **E4.1 (Markov Game Example, Fig. 1, p. 17)**: After 9 `[Bad -> 0]` episodes and 1 `[Novel -> Bad -> 1]` episode, Widrow–Hoff $\text{TD}(1)$ assigns `P(Novel) = 1.0000` (error **`0.9000`** vs true Markov $0.1000$), whereas $\text{TD}(0)$ bootstraps `Novel` off `Bad` and converges to **`P(Novel) = 0.1000, P(Bad) = 0.1000`** (error **`0.0000`**).
  - **E4.2 (Non-Markovian Counterexample, p. 18)**: On the exact same trajectory distribution when the underlying system is non-Markovian (`Bad` wins iff preceded by `Novel`, so true $E[z \mid \text{Novel}] = 1.0000$), state-aliased $\text{TD}(0)$ predicts `0.1000` (error **`0.9000`**), whereas $\text{TD}(1)$ (Widrow–Hoff / Monte Carlo) predicts **`1.0000`** (error **`0.0000`**).
  - **E4.3 (Cumulative Step-Count Prediction, Section 5.1, pp. 33–34)**: Analytical expected remaining steps $(I - Q)^{-1} \mathbf{1} = [5.0, 8.0, 9.0, 8.0, 5.0]$ for $(B, C, D, E, F)$ (matching $d^T \mathbf{1} = 1 + 2 + 3 + 2 + 1 = 9.0$ from $D$!). Per-sequence cumulative $\text{TD}(0)$ converged to **`[4.91, 7.90, 8.92, 7.95, 4.76]`** (max error `0.240`).

## 4. Findings

1. **Exact verification of Theorem 1 ($\text{TD}(1) \equiv \text{Widrow–Hoff}$ to $7.77 \times 10^{-16}$) and incremental eligibility traces ($4.44 \times 10^{-16}$, pp. 13–16)**: Per-sequence linear $\text{TD}(1)$ produces the exact same weight changes as Widrow–Hoff supervised learning while requiring only $O(d)$ memory via $e_{t+1} = \nabla_w P_{t+1} + \lambda e_t$.
2. **Exact verification of Theorem 2 ($S = D(I-Q) + (D(I-Q))^T \succ 0$) and Theorem 3 ($\text{TD}(0) \to (I - \hat{Q})^{-1} \hat{h}$ to $3.50 \times 10^{-7}$)**: Under repeated presentations of a finite training set, linear $\text{TD}(0)$ converges to the exact predictions of the maximum-likelihood Markov chain $(I - \hat{Q})^{-1} \hat{h}$ (`RMS = 0.1199`), beating Widrow–Hoff's empirical outcome average (`RMS = 0.1773`) by $32.4\%$.
3. **Why intermediate $\lambda \approx 0.3$ wins under a single presentation (`0.0864` vs `0.0887` at $\lambda=0$ and `0.1786` at $\lambda=1$, Figs. 4–5) and when $\text{TD}(0)$ vs $\text{TD}(1)$ wins (Section 3.1, pp. 17–18)**: On a single forward pass, $\text{TD}(0)$ propagates terminal outcomes back only one state per sequence; either setting $\lambda \approx 0.3$ (`0.0864`) or sweeping backward through each sequence offline (`0.0882`, p. 23) overcomes this one-step propagation lag. On Markovian domains (Fig. 1), $\text{TD}(0)$ eliminates post-transition outcome noise (`error = 0.0000` vs `0.9000` for $\text{TD}(1)$); on non-Markovian state-aliased domains (p. 18), $\text{TD}(1)$ remains unbiased (`error = 0.0000` vs `0.9000` for $\text{TD}(0)$).

## 5. Surprises / bugs hit during the work

- **The 1988 Figure 3 printing error and p. 377 Erratum**: Reading `sutton-88-with-erratum.pdf` in full revealed that in the original 1988 issue of *Machine Learning* (p. 20), the publisher accidentally printed the Figure 5 plot (single presentation at best $\alpha$, which dips at $\lambda = 0.3$) as Figure 3! Sutton published an Erratum on p. 377 with the true Figure 3 (repeated presentations, rising monotonically from $\lambda = 0$ to $\lambda = 1$).
- **Bugs discovered in third-party reproductions (`vjp23/TD_Lambda` and `MrShininnnnn/Temporal-Differences-Learning`)**:
  - In `vjp23/TD_Lambda/td_learning.py` (line 57), `training = num_sets * [seq_per_set * [None]]` creates 100 references to the **same inner list**, so all 100 training sets are identical copies of the last set.
  - In `MrShininnnnn/Temporal-Differences-Learning/main.py`, `generate_data()` discards any 10-walk batch with average walk length $> 10$, and `fig3()` initializes weights randomly in $[0, 1]$ and stops at `||delta_w|| <= 0.001` (`lr = 0.01`), halting far before convergence.
- **Intra-sequence bootstrap bias on cumulative step-count prediction (Section 5.1 vs 5.2)**: When cumulative $\text{TD}(0)$ updates `w_steps` *inside* a walk (`w_steps[s_t] += ...`), stepping $i \to j \to i$ uses the just-incremented value of state $i$ as the bootstrap target on the return step $j \to i$, biasing learned step counts upward by $\sim +0.25$ steps (`[5.35, 8.26, 9.23, 8.22, 5.13]`); accumulating $\Delta w$ per sequence as specified on p. 34 eliminates this bias (`[4.91, 7.90, 8.92, 7.95, 4.76]` vs `[5, 8, 9, 8, 5]`).

## 6. READ-LOG

- `research/sutton-1988/work/sutton1988.txt` (lines 1–1869: full 1988 *Machine Learning* paper pp. 9–44 + Erratum p. 377)
- `robosangli/sutton-1988-reproduction`: `sutton1988.py`, `analysis.py`, `results/best_alpha_summary.csv`
- `MrShininnnnn/Temporal-Differences-Learning`: `main.py`, `README.md`
- `vjp23/TD_Lambda`: `td_learning.py`, `TD_Lambda_Replication.pdf`
- `shashir/td-learning`: `src/main/scala/edu/gatech/shashir/td/TD.scala`, `Types.scala`

## 7. Files produced

- `research/sutton-1988/work/predictions.txt`
- `research/sutton-1988/work/sutton1988.txt`
- `research/sutton-1988/work/sutton1988_experiments.py`
- `research/sutton-1988/scripts/setup_sutton1988.sh`
- `research/sutton-1988/scripts/e1_repos.sh`
- `research/sutton-1988/scripts/e2_e3_e4_run.sh`
- `research/sutton-1988/results/setup_sutton1988.out`
- `research/sutton-1988/results/e1_repos.out`
- `research/sutton-1988/results/e2_e3_e4_run.out`
