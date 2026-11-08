# Context File — Paper 10: Schmidhuber (1991) — *Curious Model-Building Control Systems*

- **Citation**: Jürgen Schmidhuber (1991). *Curious Model-Building Control Systems*. **Proceedings of the IEEE International Joint Conference on Neural Networks (IJCNN)**, Singapore, Vol. 2, pp. 1458–1463 (`doi:10.1109/IJCNN.1991.170605`).
- **Source PDF / Extracted Text**:
  - Primary PDF URL: `https://sferics.idsia.ch/pub/juergen/curioussingapore.pdf`
  - Extracted Full Text (`45,931 bytes`, decoded from `dvips` Type-3 glyph stream): [research/curiosity-and-developmental-learning/papers_text/10-schmidhuber-1991-curious-control.txt](../../../research/curiosity-and-developmental-learning/papers_text/10-schmidhuber-1991-curious-control.txt)
- **Executable Experiment & Saved Results**:
  - Experiment Script: [research/curiosity-and-developmental-learning/10-schmidhuber-1991-curious-control/run_schmidhuber1991_experiments.py](../../../research/curiosity-and-developmental-learning/10-schmidhuber-1991-curious-control/run_schmidhuber1991_experiments.py)
  - Saved Results Artifact: [research/curiosity-and-developmental-learning/10-schmidhuber-1991-curious-control/results/schmidhuber_1991_results.json](../../../research/curiosity-and-developmental-learning/10-schmidhuber-1991-curious-control/results/schmidhuber_1991_results.json)

---

## Stage 1 — Context (Primary Source & Architectural Formalization)

### 1.1 Why Schmidhuber (1991) Replaced His Own 1990 "Error-Predicting" Curiosity (`[8]`, SAB 1990)
In Section 1 & Section 3 (pp. 1458–1460), Schmidhuber diagnoses two fundamental flaws in previous adaptive world-model controllers (Werbos, 1987; Jordan & Rumelhart, 1990; Sutton's Dyna, 1990) and in his own earlier 1990 SAB paper (`[8]`), which rewarded the controller proportional to the world model's **predicted error** $y_C(t)$:
1. **The Non-Deterministic White-Noise Trap (p. 1460)**:
   > *"One problem with the idea above is that in non-deterministic environments the controller will focus on parts of the environmental dynamics which are inherently unpredictable. This is because the adaptive model usually will produce incorrect predictions for the uncertain parts of the environment. Therefore the control system will receive reinforcement although it cannot be expected that the world model will improve."*
2. **Easy-Before-Hard Ordering (p. 1460)**:
   > *"Instead of simply learning to predict errors as the approach described in `[8]`, the new approach learns to predict cumulative error changes."*

### 1.2 The 4 Reliability-Modeling Methods (Section 2, pp. 1458–1459)
Given an adaptive discrete-time predictor $P$ with input $x_P(t) = y_A(t) \circ x(t)$, output $y_P(t)$, and target $d_P(t) = x(t+1)$, Schmidhuber introduces a **Confidence Module $C$** with output $y_C(t)$ estimating the reliability of $P$'s predictions via four methods:
- **Method 1**: Global failure probability $P(y_P(t) \ne d_P(t) \mid x_P(t))$.
- **Method 2**: Local component-wise failure probability $P(y_P^k(t) \ne d_P^k(t) \mid x_P(t))$.
- **Method 3**: Global expected error $E[\frac{1}{2}\|d_P(t) - y_P(t)\|^2 \mid x_P(t)]$, or in Section 5's implementation, $E[|d_P(t) - y_P(t)| \mid x_P(t)]$ trained by gradient descent with a small learning rate $\eta_C$.
- **Method 4**: Local component-wise expected squared error $E[(d_P^k(t) - y_P^k(t))^2 \mid x_P(t)]$.

### 1.3 Section 3.1–3.2: Curiosity as Discounted Cumulative Reliability Change + Watkins' Q-Learning
At each step $t$, after updating $P$ and $C$, the **curiosity reinforcement** is the current change in assumed reliability:
$$\dot{y}_C(t) = y_C^{\text{old}}(t) - y_C^{\text{new}}(t)$$
and the controller's curiosity goal at $t_0$ is to maximize $E\left[\sum_{\tau=0}^\infty \gamma^\tau \dot{y}_C(t_0 + \tau + 1)\right]$ via Watkins' (1989) Q-learning:
$$Q(x(t), a(t)) \leftarrow (1 - \alpha) Q(x(t), a(t)) + \alpha \left(r_{\text{ext}}(t) + \dot{y}_C(t) + \gamma U(x(t+1)) - Q(x(t), a(t))\right)$$

### 1.4 Section 4: The Prediction-Shift Network $\Delta$
Section 4 proposes an alternative where a slow network $\Delta$ learns to approximate the **expected signed shift** $E[\Delta y_P(t) \mid x_P(t)] = E[y_P^{\text{after}}(t) - y_P^{\text{before}}(t) \mid x_P(t)]$, and the absolute value $|\Delta(t)|$ is used as reinforcement.

---

## Stage 2 — Direction (Testable Hypotheses)

1. **Experiment E1 (Section 5 Replication — 100-State `10x10` Reactive 2D World)**:
   - Replicate Section 5 (`100` states: `70` trivial deterministic states $d(s)=0$, `15` complex deterministic learnable states $d(s) \in [0.60, 0.95]$, and `15` inherently unpredictable random states $d(s) \sim \text{Uniform}(-1, 1)$).
   - Compare `random_search`, `schmidhuber_1990_mismatch_error_curiosity` (rewarding $y_C(t)$), `schmidhuber_1991_adaptive_curiosity_dyc` (rewarding $\dot{y}_C(t)$), and `schmidhuber_1991_sec4_smoothed_delta` (rewarding $|E[\Delta y_P \mid s]|$).
   - Verify Section 5's reported speedup ($\ge 4.5\times\text{--}5\times$ faster reduction of deterministic world-model squared error $E(t)$ than random search).
2. **Experiment E2 (Section 4 Boundary — Why Expectation Must Precede the Absolute Value in $|\Delta(t)|$)**:
   - Test what happens if an implementation takes the absolute value *before* the expectation ($|\Delta y_P(t)| = |y_P^{\text{after}}(t) - y_P^{\text{before}}(t)| = \eta_P |d_P(t) - y_P(t)|$) vs. taking the absolute value *of the learned signed expectation* ($|E[\Delta y_P(t) \mid x_P]|$, Section 4) in a zero-mean white-noise state.

---

## Stage 3 — Execution & Stage 4 — Outcome (Quantitative Results)

Running [run_schmidhuber1991_experiments.py](../../../research/curiosity-and-developmental-learning/10-schmidhuber-1991-curious-control/run_schmidhuber1991_experiments.py) (`10` seeds $\times 2,200$ steps) produces:

### 4.1 Experiment E1: 100-State Reactive World Replication (Section 5)
| Controller | Trivial States (`0..69`) | Learnable Deterministic (`70..84`) | Non-Deterministic Noise (`85..99`) | Steps to $E(t) \le 0.25$ | Final Deterministic Error $E$ |
|---|---|---|---|---|---|
| **`random_search`** | `69.89%` | `14.84%` | `15.27%` | `1578.7` | `0.064596` |
| **`schmidhuber_1990_mismatch_error_curiosity`** | `8.40%` | `8.30%` | **`83.30%`** (Noise trap!) | `2200.0` (Never) | `0.851518` |
| **`schmidhuber_1991_adaptive_curiosity_dyc` (Sec. 3)** | `8.38%` | **`89.68%`** | **`1.94%`** | **`349.4`** (**`4.52x` faster**) | **`0.000000`** |
| **`schmidhuber_1991_sec4_smoothed_delta` (Sec. 4)** | `8.53%` | `23.43%` | `68.04%` | `625.8` (`2.52x` faster) | `0.001437` |

### 4.2 Experiment E2: Unsmoothed $|\Delta y_P(t)|$ vs. Smoothed $|E[\Delta y_P(t)]|$ & $\dot{y}_C(t)$
| State Regime | Unsmoothed $E[\|\Delta y_P(t)\|]$ | Section 4 $\|E[\Delta y_P(t)]\|$ | Section 3 $\dot{y}_C(t)$ |
|---|---|---|---|
| **Learnable Deterministic (`d = 0.80`, after convergence)** | `0.00012` | `0.00243` | `0.00104` |
| **Zero-Mean White Noise (`d ~ Uniform(-1, 1)`)** | **`0.06954`** (High false reward!) | **`0.00880`** ($7.9\times$ lower) | `0.01831` |

---

## Stage 5 — Growth (What We Learned & Critical Boundaries)

1. **Jensen's Inequality Explains Why Section 4 Requires $|E[\Delta y_P]|$ Rather Than $E[|\Delta y_P|]$**:
   - Because a single gradient step satisfies $\Delta y_P(t) = y_P^{\text{after}}(t) - y_P^{\text{before}}(t) = \eta_P (d_P(t) - y_P(t)) \|x\|^2$, taking the absolute value *before* averaging gives $|\Delta y_P(t)| \propto |d_P(t) - y_P(t)|$, which is identical to Schmidhuber's 1990 raw prediction-error reward (`0.06954` in white noise)!
   - By training a slow network $\Delta$ with learning rate $\eta_\Delta \ll 1$ to predict the *signed* shift $E[\Delta y_P \mid x_P]$ first, zero-mean noise cancels out in expectation before $|\Delta(t)|$ is taken (`0.00880`), though finite-sample variance in $\Delta(t)$ still leaves Section 4 more susceptible to noise (`68.04%` after deterministic states saturate) than Section 3's signed reliability derivative $\dot{y}_C(t)$ (`1.94%`).
