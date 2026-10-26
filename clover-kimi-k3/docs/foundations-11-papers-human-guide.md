# Eleven Foundational AI Papers (1943–2017) — Plain-Language Guide & Verified Findings

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh.

This document explains in plain language what each of the 11 foundational Artificial Intelligence and Neural Computation papers is about, what question each paper asked, how its mechanism actually works, and what we observed when we read the primary source texts, audited open-source reference repositories, and built and ran our own verified implementations on CPU (`k3`).

---

## Overview: How the 11 Papers Connect Across 74 Years

| # | Year | Authors & Paper | Core Question | The Mechanism Introduced | Detailed Context File |
|---:|:---:|---|---|---|---|
| **1** | 1943 | **McCulloch & Pitts** — *A Logical Calculus of the Ideas Immanent in Nervous Activity* | Can binary all-or-none neurons compute logic and store memory? | Threshold logic gates with excitation, absolute inhibition, and recurrent feedback loops | `docs/mcculloch-pitts-1943-context.md` |
| **2** | 1950 | **Alan M. Turing** — *Computing Machinery and Intelligence* | How do we test machine intelligence, and how can a digital computer learn? | The Imitation Game, stored-program universality, and educating a "Child-Machine" | `docs/turing-1950-context.md` |
| **3** | 1956 | **Newell & Simon** — *The Logic Theory Machine (RAND P-868)* | How can a program prove mathematical theorems without exponential brute force? | List-processing (`IPL-I`), working memories, backward subgoal trees, and heuristic filters (`CSm`) | `docs/newell-simon-1956-context.md` |
| **4** | 1958 | **Frank Rosenblatt** — *The Perceptron: A Probabilistic Model* | How can a randomly wired network of neurons learn to recognize visual shapes from experience? | `S -> A -> R` architecture, random local receptive fields, and zero-sum (`gamma`) weight reinforcement | `docs/rosenblatt-1958-context.md` |
| **5** | 1969 | **Minsky & Papert** — *Perceptrons: An Introduction to Computational Geometry* | What are the exact mathematical limits of single-layer linear threshold classifiers? | Predicate order $k$, Group Invariance Theorem, Parity order $|R|$, and topological connectivity impossibility | `docs/minsky-papert-1969-context.md` |
| **6** | 1982 | **John J. Hopfield** — *Neural networks and physical systems with emergent collective computational abilities* | How can a recurrent network reconstruct a complete memory from a noisy partial cue? | Symmetric synapses ($T_{ij} = T_{ji}$), Lyapunov Energy minimization $E$, and $0.15 N$ attractor basins | `docs/hopfield-1982-context.md` |
| **7** | 1986 | **Rumelhart, Hinton & Williams** — *Learning representations by back-propagating errors* | How do we train hidden units in multi-layer networks so they invent their own internal features? | Smooth differentiable units, the Generalized Delta Rule (Backpropagation), momentum, and spatial weight sharing | `docs/rumelhart-hinton-williams-1986-context.md` |
| **8** | 1988 | **Richard S. Sutton** — *Learning to Predict by the Methods of Temporal Differences* | How can a learner predict future outcomes incrementally without waiting for the final outcome? | Temporal-difference error ($P_{t+1} - P_t$), eligibility traces $\text{TD}(\lambda)$, and maximum-likelihood Markov convergence | `docs/sutton-1988-context.md` |
| **9** | 1998 | **LeCun, Bottou, Bengio & Haffner** — *Gradient-Based Learning Applied to Document Recognition* | How do we recognize 2D characters and read multi-character documents end-to-end without manual segmentation? | `LeNet-5` (`60,000` params), Space-Displacement Networks (`SDNN`), and Graph Transformer Networks (`GTN`) | `docs/lecun-1998-context.md` |
| **10** | 2012 | **Krizhevsky, Sutskever & Hinton** — *ImageNet Classification with Deep Convolutional Neural Networks* | How do we scale deep CNNs to millions of high-resolution natural images without vanishing gradients or overfitting? | Non-saturating `ReLU`, 2-GPU split-stream `AlexNet` (`60.97M` params), `LRN`, overlapping pooling, PCA color aug, and `Dropout` | `docs/krizhevsky-2012-context.md` |
| **11** | 2017 | **Vaswani et al.** — *Attention Is All You Need* | Can we replace recurrence and convolutions entirely with direct all-to-all token attention? | `The Transformer`: Scaled Dot-Product Attention ($1/\sqrt{d_k}$), Multi-Head subspaces, Sinusoidal `PE`, and causal masking | `docs/vaswani-2017-context.md` |

---

## 1. Warren S. McCulloch & Walter Pitts (1943)
**Paper**: *A Logical Calculus of the Ideas Immanent in Nervous Activity* (*Bulletin of Mathematical Biophysics*, 5: 115–133)
- **Detailed context**: `docs/mcculloch-pitts-1943-context.md`
- **Verified code**: `research/mcculloch-pitts-1943/work/mp1943_calculus.py`, `e3_illusions_and_equivalences.py`, `e4_cyclic_nets.py`

### What the paper is about
Before 1943, neurobiology was described in continuous physiological terms, while symbolic logic lived in mathematics. McCulloch and Pitts showed that because a neuron obeys an **all-or-none firing law** (`1` = fires an impulse, `0` = stays silent), any network of neurons is a **discrete-time logic circuit**, and every proposition of temporal logic corresponds to a specific neural wiring diagram.

### What they speak about (and what we verified)
1. **The Fundamental Neuron Rule (Equation 1)**:
   - Each neuron $c_i$ has an integer threshold $\theta_i \ge 1$, excitatory incoming synapses, and absolute inhibitory incoming synapses.
   - Neuron $c_i$ fires at tick $t$ if and only if **zero** inhibitory inputs fired at tick $t-1$ **and** at least $\theta_i$ excitatory inputs fired at tick $t-1$:
     $$N_i(t) \equiv \neg \bigvee_{j \in I_i} N_j(t-1) \;\wedge\; \left(\sum_{k \in E_i} N_k(t-1) \ge \theta_i\right)$$
2. **What Single Neurons vs Multi-Layer Networks Without Loops Can Compute (Section 2, Theorems 1–3)**:
   - There are $2^{2^2} = 16$ possible Boolean functions of two inputs $(x_1, x_2)$. By exhaustive search across all thresholds and synapse combinations, we verified that a **single McCulloch–Pitts neuron can only compute `7/16`** of them (`FALSE`, `AND`, `x1 AND NOT x2`, `x2 AND NOT x1`, `x1`, `x2`, `OR`). It cannot compute `XOR`, `IFF`, `NAND`, `NOR`, or `NOT` because excitation and inhibition cannot be mixed to invert an all-zero input without spontaneous background firing.
   - When you wire neurons into a **multi-layer acyclic network** (Theorem 2), you can build disjunctive normal forms (`DNF`) and compute **`8/16`** 2-input functions (including `XOR`) and **`128/256`** 3-input functions—specifically, **every** Boolean function that satisfies the physical silence constraint $f(0, \dots, 0) = 0$ (verified across `1,000` random acyclic networks: `0/1,000` ever fire when all inputs are zero).
3. **Why Touching Cold Metal Briefly to the Skin Feels Hot (Figure 1e)**:
   - Cold receptors fire continuously while cold is applied, whereas the "cold-stopped" pathway fires a 1-tick burst when cooling stops and excites the heat neuron. We ran Figure 1e's exact 4-neuron circuit: a brief 1-tick cold touch triggers **both** the cold and heat channels (`[1, 1]`, producing the paradoxical heat illusion), whereas sustained cooling triggers **only** the cold channel (`[0, 1]`).
4. **Equivalence of Relative vs Absolute Inhibition & Hebbian Synapses vs Loops (Theorems 4–7)**:
   - Theorems 4–6 prove that biological details like relative inhibition (subtracting threshold weights instead of an absolute veto), post-tetanic extinction, and facilitation do **not** change what a neural network can compute—each can be replaced by an equivalent standard McCulloch–Pitts circuit (verified across all `256/256` 4-step histories).
   - **Theorem 7 (Figure 1i)** proves something even deeper: a network with an **alterable synapse** that permanently closes when two neurons fire together (Hebbian learning) is **100% behaviorally identical (`4,096/4,096` histories)** to a fixed network with no changeable synapses that instead routes the signal into a **1-neuron circular feedback loop** (`order-1 circle`).
5. **Cyclic Networks & Why Brain States Cannot Reconstruct the Past (Sections 3–4, Theorems 8–10)**:
   - Adding circular loops gives the network indefinite memory (`∃ t'<t` existential latches, `∀ t'<t` universal latches, and modulo-$k$ counters), but also introduces **causal irreciprocity**: in a 4-neuron cyclic network, only `8` of the `16` state vectors are reachable, and each reachable state has `4` to `12` different predecessor states that collapse into it. Looking at the present state of a nervous system determines its future, but can never uniquely recover *when* or *how* a circulating memory began.

---

## 2. Alan M. Turing (1950)
**Paper**: *Computing Machinery and Intelligence* (*Mind*, Vol. 59, No. 236, pp. 433–460)
- **Detailed context**: `docs/turing-1950-context.md`
- **Verified code**: `research/turing-1950/work/e2_stored_program_and_universal.py`, `e3_e4_imitation_and_learning.py`

### What the paper is about
Turing opens with *"I propose to consider the question, 'Can machines think?'"* and immediately rejects defining "think" by opinion poll. Instead, he replaces it with an operational behavioral test (**The Imitation Game**), explains the physical and mathematical principles of **stored-program digital computers** and **universality**, refutes nine philosophical and mathematical objections, and closes by proposing how to build a **learning "Child-Machine."**

### What he speaks about (and what we verified)
1. **The Imitation Game & Why an Honest Calculator Loses Immediately (Sections 1–2)**:
   - An interrogator $C$ asks questions over a text channel to two hidden players ($A$ and $B$) and tries to identify which is the machine.
   - In Turing's sample dialogue (p. 434), the interrogator asks: *"Add 34957 to 70764."* The witness pauses for **30 seconds** and answers **`105621`**.
   - When we ran the actual arithmetic in Section 4, `34957 + 70764 = 105721`—the witness's answer `105621` is intentionally wrong by `100` (a missed carry into the hundreds column!). As Turing explains on page 448, if a machine answers `105721` in milliseconds, an optimal Bayesian interrogator unmasks it as the machine **`100.0%` of the time**; when the machine spoofs human reaction latency and occasional carry slips, the interrogator's accuracy drops to **`48.4%`** (chance).
2. **Stored-Program Digital Computers & Self-Modifying Code (Section 4)**:
   - A digital computer has three parts: **Store** (memory), **Executive unit** (arithmetic/logic), and **Control** (instruction table).
   - Instructions are stored in memory as ordinary numbers. Turing gives the concrete Manchester 10-digit instruction `6809430217`, which means *"add the number in address 6809 to the number in address 4302 and put the sum into address 17."* Because instructions are data, the machine can add constants to its own instruction words to step through array addresses inside a loop.
   - Turing also estimates the storage capacity of the Manchester Ferranti Mark I: `10` cathode-ray tube pages $\times 32$ lines $\times 40$ digits $+$ `8` magnetic drums $\times 256$ tracks $\times 80$ digits $= \mathbf{174,380\text{ bits}}$.
3. **Discrete-State Machines & Universality (Section 5)**:
   - Turing illustrates a 3-state wheel machine ($q_1, q_2, q_3$) driven by inputs $i_0, i_1$ and proves that a single **Universal Digital Computer** can be programmed with the transition table of *any* discrete-state machine and mimic its outputs bit-for-bit (`256/256` 8-step input sequences).
4. **Refuting the Mathematical, Lovelace, & Continuous-Nervous-System Objections (Section 6)**:
   - **Mathematical Objection (Gödel / Halting, Section 6.1)**: A machine $M_0$ cannot answer whether *it itself* will answer "Yes" to its own diagonal code (`SELF_APPLIED_PARADOX`), **but** a slightly larger machine $M_1$ can evaluate $M_0$'s code without trouble; Gödel's theorem applies only to self-reference and provides zero proof that humans are free of the same limit.
   - **Continuous Nervous System (Section 6.7)**: A discrete machine with random choices can mimic a continuous analog system (like a Differential Analyser returning a narrow Gaussian distribution `40% -> 3.1416, 26% -> 3.1415` for $\pi$) so closely that an interrogator cannot tell them apart.
5. **The Learning Child-Machine & Symbolic vs Reinforcing Channels (Section 7)**:
   - Rather than hand-coding an adult mind, Turing proposes programming a small **Child-Machine** (heredity) and training it through **reward and punishment** ($R/P$) plus **symbolic teacher commands**.
   - Turing calculates that if a child-machine must guess among $k$ choices across $d$ sequential steps using only blind punishment blows, it takes on average **`22,400` blows**; once an unemotional symbolic channel is established (`202` setup signals to teach a 20-letter alphabet), the teacher can state the rule in **1 symbolic command**.

---

## 3. Allen Newell & Herbert A. Simon (1956)
**Paper**: *The Logic Theory Machine: A Complex Information Processing System* (RAND Report `P-868`, 15 June 1956 / *IRE Transactions on Information Theory*, Vol. 2, No. 3, pp. 61–79)
- **Detailed context**: `docs/newell-simon-1956-context.md`
- **Verified code**: `research/newell-simon-1956/work/lt1956_engine.py`, `lt1956_methods.py`, `run_e2_e3_e4.py`

### What the paper is about
Newell and Simon present the complete specification and source code (in **IPL-I**, Information Processing Language I) of **The Logic Theorist (`LT`)**, a system that discovers formal proofs for theorems in Chapter 2 of Whitehead and Russell's *Principia Mathematica*. It is the foundational paper of **heuristic search** and **symbolic list processing**.

### What they speak about (and what we verified)
1. **Representing Logic as List Trees in Working Memory (Sections I–II)**:
   - Logical expressions such as $*1.2$: $(p \vee p) \supset p$ are stored as binary trees of elements (`A, N, V, L, R`) where each node has a location code, number of negation signs, connective (`OR` or `IMPLIES`), and variable name.
2. **The Four Methods (`MSb`, `MDt`, `MCh`, `MVb`) & The Executive Routine (`Ex`)**:
   - Given a problem expression to prove, the Executive (`Ex`) tries three methods against every axiom and previously proved theorem on the theorem list $T$:
     - **Substitution (`MSb`)**: Tries to match a theorem on $T$ directly to the target problem by substituting variables (`LMc`) and definitions (`LSb`, replacing $A \supset B$ with $\sim A \vee B$).
     - **Detachment (`MDt`)**: Searches $T$ for a theorem of the form $A \supset B$ where the right side $B$ can be unified with our target problem; if found, `A` becomes a new subproblem.
     - **Chaining (`MCh`)**: If our target problem has the form $A \supset C$, searches $T$ for $A \supset B$ (forward chaining, creating subproblem $B \supset C$) or $B \supset C$ (backward chaining, creating subproblem $A \supset B$).
   - If none of the three methods proves the problem in one step, all generated subproblems are appended to the subproblem list $P$, and `Ex(CK)` selects the next subproblem with the lowest complexity level $K$.
3. **Ten Concrete Bugs Found in the Printed 1956 `P-868` IPL-I Code (`E1`)**:
   - By reading all 701 lines of Section III's formal IPL-I program line by line and cross-checking against David Moews's interpreter (`dmoews/logic-theorist`), we cataloged **10 concrete bugs** in the printed 1956 monograph (including calling an undefined routine `R011` inside `CXn`, inverted branch conditions in `CAr` and `CSb`, an uninitialized working memory `2` in `FAl`, and a missing `P 0 A` in `LSb`).
4. **How the `CSm` Similarity Filter Cuts Search by `13.1x`—and Its Blind Spot (`E3`)**:
   - Before attempting full tree unification (`LMc`), `MDt` and `MCh` call `CSm` (Test Similarity), which compares only a 3-number summary $(K_L, K_R, J)$ (left depth, right depth, and count of distinct variables).
   - Across all 67 theorems in Chapter 2, `CSm` rejects **`93.6%`** of non-matching theorem pairs (`25,926 / 27,701`), reducing expensive `LMc` calls by **`13.1x`**.
   - However, because `CSm` requires the number of distinct variables $J$ to be **strictly equal**, it has a blind spot: when proving theorem `*2.07` ($p \supset (p \vee p)$, whose right side has $J_R = 1$ distinct variable) from Axiom `*1.3` ($q \supset (p \vee q)$, whose right side has $J_R = 2$ distinct variables), `CSm` **falsely rejects** the match! Disabling `CSm` proves `*2.07` in 1 step via `MSb(*1.3)`.
5. **Contraction (`VCt`), Theorem-Memory Learning, & `Ex(CK)` Tie-Breaking (`E3–E4`)**:
   - Contraction (`VCt`, pre-condensing $A \vee A \to A$ before `MDt`/`MCh` matching) unlocks **9 additional one-step Detachment proofs**, including theorem `*2.06` (proved in 2 steps via `MDt(*1.5) + MSb(*1.6)` in the 1956 IPL-I code).
   - Giving `LT` access to all prior theorems proves **`46/67`** theorems (`48/67` with multi-level chaining), compared to **`22/67`** when restricted to theorems it proved itself, and **`12/67`** from the 5 axioms alone.
   - Tracing why `dmoews` and our initial `LogicTheorist1956` differed on 5 theorems exposed a single instruction on page 50 of `P-868`: `CKG 2 1 - C` inside `Ex(CK)` tests $K(2) - K(1) \ge 0$, which overwrites the best candidate on **ties** ($K_i \le K_{\text{best}}$), turning equal-level subproblem selection from FIFO into LIFO.

---

## 4. Frank Rosenblatt (1958)
**Paper**: *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain* (*Psychological Review*, Vol. 65, No. 6, pp. 386–408)
- **Detailed context**: `docs/rosenblatt-1958-context.md`
- **Verified code**: `research/rosenblatt-1958/work/rosenblatt1958_models.py`

### What the paper is about
Rosenblatt asks how a biological brain—whose initial sensory-to-association wiring is largely random rather than precision-engineered like a logic circuit—can learn to recognize and generalize visual patterns. He introduces **The Perceptron**, deriving both its **Predominant Phase** (exact combinatorics of random feature detectors) and its **Postdominant Phase** (weight reinforcement rules $\alpha, \beta, \gamma$).

### What he speaks about (and what we verified)
1. **The `S -> A -> R` Architecture & Predominant Phase Combinatorics (Equations 1–3, Figure 4)**:
   - Sensory points ($S$) connect randomly ($x$ excitatory and $y$ inhibitory inputs per cell) to Association units ($A$, threshold $\theta$), which feed modifiable weights $V_i$ into Response units ($R$).
   - We verified Rosenblatt's exact hypergeometric/binomial equations for $P_a$ (the probability an $A$-unit fires on an image covering fraction $R$ of the retina) and $P_c$ (the conditional probability the same $A$-unit fires on a second image sharing overlap $L$) against $N_A = 100,000$ Monte Carlo simulations (`max error = 0.00238`).
   - Adding inhibitory connections ($x=5, y=5$, Figure 4c) flattens the variation of $P_a$ across stimulus sizes $R \in [0.1, 0.7]$ by **`10.3x`**, preventing large bright shapes from blinding the network.
2. **The Three Reinforcement Rules ($\alpha, \beta, \gamma$, Table 1) & The Page 400 Inequality $P_{c11} > P_a > P_{c12}$**:
   - **$\alpha$-system (uncompensated)**: Adds $+1$ to active $A$-units; if Class 1 appears $3\times$ as often as Class 2, Class 1 weights grow $3\times$ larger and drown out the minority class.
   - **$\beta$-system (constant-feed)**: Adds a constant drift to each pool plus activity reinforcement.
   - **$\gamma$-system (parasitic / zero-sum)**: Active $A$-units gain $+1$ while inactive $A$-units in the same pool lose $-\frac{N_a}{N_e - N_a}$ so the sum of all weights in the pool remains strictly $\sum V_i = 0$.
   - On page 400, Rosenblatt states a critical condition for the $\gamma$-system to beat the $\alpha$-system under class frequency imbalance: **within-class overlap $P_{c11}$ must exceed mean activity $P_a$, which must exceed between-class overlap $P_{c12}$** ($P_{c11} > P_a > P_{c12}$). We verified both sides of this inequality experimentally:
     - When horizontal and vertical bars cross in the center (`Regime A`), they share a $3\times 3$ overlap larger than random chance, **violating** $P_a > P_{c12}$ (`0.1700 < 0.1762 < 0.1831`), and the $\gamma$-system drops to `50.0%` under $3:1$ imbalance.
     - When the classes satisfy $P_{c11} > P_a > P_{c12}$ (`0.2370 > 0.1585 > 0.0939`, `Regime B`), the $\gamma$-system achieves **`100.0%` balanced accuracy** even under $3:1$ class frequency imbalance!
3. **Spontaneous Unsupervised Organization (p. 404) & The Relational Judgment Boundary (pp. 406–407)**:
   - Without any teacher labels, a perceptron whose weights decay proportionally ($(1 - \delta) V_i$) spontaneously separates two disjoint stimulus classes into `+1` and `-1` (`100.0%` unsupervised clustering accuracy vs `50.0%` without decay).
   - On Rosenblatt's relational task (*"Is the top square larger than the bottom square?"*), slot-local $A$-units that only inspect one region at a time get `99.8%` on concrete size (`"Is the top square large?"`) but fail at **`42.0%`** on the relative comparison, whereas cross-slot $A$-units reach **`100.0%`**.

---

## 5. Marvin Minsky & Seymour Papert (1969)
**Paper**: *Perceptrons: An Introduction to Computational Geometry* (MIT Press, 1969 / MIT AI Lab Memos `AIM-140` & `AIM-167`)
- **Detailed context**: `docs/minsky-papert-1969-context.md`
- **Verified code**: `research/minsky-papert-1969/work/minsky_papert_1969_experiments.py`

### What the paper is about
Minsky and Papert ask a fundamental mathematical question: **if a single-layer linear threshold classifier is only allowed to use local feature masks $\varphi(X)$ that inspect at most $k$ pixels at a time (an "order-$k$" perceptron), what geometric properties of an image can it compute, and what properties are mathematically impossible for it to compute?**

### What they speak about (and what we verified)
1. **Positive Normal Form, Order of a Predicate, & Exact Counting (Chapters 1–2)**:
   - Any mask $\varphi(X)$ of support $\le k$ can be rewritten in **Möbius Positive Normal Form** as a linear combination of pure conjunction masks $\bigwedge_{x \in S} x$ (`256/256` verified).
   - For $n = 3$ binary variables (`256` total Boolean functions), we verified the exact distribution of minimal perceptron orders: `2` functions of order 0 (`TRUE`, `FALSE`), `102` of order 1 (linearly separable), `150` of order 2, and `2` of order 3 (`3-bit PARITY` and `NOT-PARITY`).
   - Using the Group Invariance Theorem, the predicate *"the image contains exactly $M$ active pixels"* ($\psi_M(X) = [|X| = M]$) has **order 2 for every $M$**, because $-2 \sum_{|S|=2} \varphi_S(X) + (2M - 1) \sum_{|S|=1} \varphi_S(X) = M - (|X| - M)^2$ peaks strictly at $|X| = M$ (`1,792/1,792` verified).
2. **Why Parity Has Order $|R|$ (`100%` of the Retina, Chapter 3)**:
   - Determining whether $|X|$ is odd or even requires inspecting **all $|R|$ pixels simultaneously** ($k = |R|$), and the unique symmetric Positive Normal Form weights $\alpha_{|S|} = (-2)^{|S|}$ explode exponentially (`max |w| = 1,024` by $|R| = 10$).
3. **Why Topological Connectivity Cannot Be Computed by Local Masks—and How We Caught a Dataset Confound (`E1` & `E3`, Chapter 5)**:
   - Minsky & Papert prove (`One-in-a-Box` Theorem) that patterns `(3,3,0)` and `(4,1,1)` across three 4-pixel boxes have the **exact same order-2 mask spectrum `(6, 6, 9)`**, so no order-2 perceptron can tell whether every box is non-empty (`One-in-a-Box` has order 3; at box size $|A_i|=3$, the order-2 polynomial $N^1 + N^{12} - 2 N^{11} > 4$ separates all `512/512` configurations!).
   - Consequently, **topological connectivity** ($\Psi^{\text{CONNECTED}}$) cannot be computed by any diameter-limited perceptron (`0/308` bridge toggle configurations) or any bounded-order perceptron, whereas Minsky & Papert's **2-pointer serial boundary-tracing algorithm** (`SEARCH/SCAN/TRACE`, Chapter 9) decides connectivity in $O(|R|)$ time using only $2 \log_2 |R|$ bits of memory.
   - **Real-world lesson from `TaraAlcira/PerceptronsRevisited` (`E1`)**: When we inspected `TaraAlcira`'s spiral generator (`one_spiral_function.py`), we found it created 1-spirals by erasing the two outer legs of a 2-spiral—giving 1-spirals `652` ink pixels vs `804` ink pixels for 2-spirals! An order-1 perceptron immediately got `100%` accuracy simply by thresholding total pixel area $[|X| \le 728]$—a direct illustration of **Corollary 3 on page 187 of Minsky & Papert (1969)** (*"If $|X|$ is bounded ($\le K$), then $\Psi^{\text{CONNECTED}}(X)$ is of order 1"*). Varying spiral size across `{56, 64, 72, 80}` destroyed the area shortcut (`62.0%` order-1 perceptron vs `100.0%` for the 2-pointer serial tracer).
4. **Chapter 12 (`AIM-167`): Perceptron vs Bayes vs Nearest Neighbor**:
   - On correlated linearly separable data (Figure 12.3a), the Perceptron Learning Rule makes **`0.0%` errors** while Naive Bayes makes **`25.0%` errors**.
   - On independent noisy channels (Figure 12.3b), optimal log-odds Bayes achieves `10.3%` error vs `19.3%` for the Perceptron.
   - On $n=25$-bit parity where only `4/25` bits determine the label (Section 12.4.4), 1-Nearest Neighbor fails on **`99.6%–100.0%`** of unseen queries (matching Minsky & Papert's formula $1 - q^n = 99.62\%$).

---

## 6. John J. Hopfield (1982)
**Paper**: *Neural networks and physical systems with emergent collective computational abilities* (*PNAS*, Vol. 79, No. 8, pp. 2554–2558)
- **Detailed context**: `docs/hopfield-1982-context.md`
- **Verified code**: `research/hopfield-1982/work/hopfield1982_experiments.py`

### What the paper is about
Hopfield shows that a recurrent network of binary neurons with **symmetric connections ($T_{ij} = T_{ji}, T_{ii} = 0$)** has a global **Lyapunov Energy function $E$** that decreases on every asynchronous neuron flip. Each stored pattern becomes a stable valley (attractor) of the energy landscape, allowing the network to reconstruct a complete memory from any partial or corrupted portion of it (**content-addressable memory**).

### What he speaks about (and what we verified)
1. **Asynchronous Energy Monotonicity vs Synchronous Oscillations (Equations 2, 7)**:
   - Under Hopfield's asynchronous update ($V_i \to 1$ if $\sum_j T_{ij} V_j > U_i$, else $0$, one random neuron at a time), the change in energy is $\Delta E = -\Delta V_i \sum_{j \ne i} T_{ij} V_j \le 0$. Across `200` random initial states and `7,162` asynchronous flips, energy increased **`0` times** (`100%` converged to fixed points).
   - Negative control: when all neurons update *synchronously* on the same clock tick, **`197/200` runs (`98.5%`)** increased energy on step 1 and fell into 2-step limit-cycle oscillations!
2. **Why Hopfield's $\{0, 1\}$ State Convention Has Half the Capacity of Bipolar $\{-1, +1\}$ (`E2`)**:
   - Hopfield writes neurons as $V_i \in \{0, 1\}$ while storing synapses as bipolar outer products $T_{ij} = \sum_{s=1}^n (2V^s_i - 1)(2V^s_j - 1)$.
   - Because $V_j \in \{0, 1\}$ zeroes out half the terms in the readout sum $\sum_j T_{ij} V_j$, the signal is $+N/2$ and the cross-talk noise variance is $(n-1)N/2$, giving a signal-to-noise variance ratio of $\frac{N}{2(n-1)}$. If the readout states are also bipolar $S_j = 2V_j - 1 \in \{-1, +1\}$, the signal doubles to $+N$ while noise variance only doubles to $(n-1)N$, **doubling the SNR to $\frac{N}{n-1}$**!
   - Consequently, $\{0, 1\}$ states with $n = 10$ patterns have the **exact same single-step bit error rate** (`0.00977` vs `0.00983`, theoretical `0.00982`) as bipolar $\{-1, +1\}$ states with $n = 2(10) - 1 = 19$ patterns!
3. **Attraction Basins ($N=30, n=5$), Memory Merging ($N=100, n=8$), & Biological Robustness (`E3–E4`)**:
   - Starting from cues corrupted by $d$ bit errors ($N=30, n=5$), exact recovery goes from `90.6%` at $d=1$ to `20.8%` at $d=12$ (matching Hopfield's p. 2556 statement that at $12$ errors the probability of returning to the nearest nominal state is $\approx 0.2$).
   - When two stored memories are separated by only $d=10$ bits out of $N=100$, their energy valleys merge into a fused attractor (`47.5%` fused, `0.0%` exact), whereas at $d=30$ they remain distinct (`78.0%` exact).
   - Clipping all synapses to $\pm 1$ retains `0.7966` of the capacity (matching $\sqrt{2/\pi} = 0.7979$); bounding synapses to $\pm 3$ with forgetting step $\Delta T_{ij} = \pm 1$ creates a palimpsest that recalls recent memories at `99.8%` vs `55.8%` for older ones; and at $25\times$ overload ($n=250$ patterns in $N=100$ neurons), the initial neuron flip rate still distinguishes familiar vs novel patterns with **`92.8%` accuracy (`0.9828` AUC)**.

---

## 7. David E. Rumelhart, Geoffrey E. Hinton & Ronald J. Williams (1986)
**Paper**: *Learning representations by back-propagating errors* (*Nature*, 323: 533–536, 9 Oct 1986; and ICS Report 8506 / *PDP Vol. 1*, Chapter 8)
- **Detailed context**: `docs/rumelhart-hinton-williams-1986-context.md`
- **Verified code**: `research/rumelhart-hinton-williams-1986/work/rhw1986_experiments.py`

### What the paper is about
Single-layer perceptrons can only map similar inputs to similar outputs, failing whenever a task requires constructing a new intermediate representation (like `XOR`, `Parity`, or `Symmetry`). Rumelhart, Hinton, and Williams replace hard threshold steps with smooth differentiable logistic sigmoids $o_j = \frac{1}{1 + e^{-\text{net}_j}}$ and derive the **Generalized Delta Rule (Backpropagation)**, demonstrating across seven classic tasks that gradient descent with momentum automatically discovers compact, interpretable hidden representations.

### What they speak about (and what we verified)
1. **The Generalized Delta Rule, Symmetry Saddle Points, & Momentum Speedup (Equations 9–16, pp. 4–12)**:
   - Backpropagating $\delta_{pj} = o_{pj}(1 - o_{pj}) \sum_k \delta_{pk} w_{kj}$ matches central finite differences to **`1.36e-10`** across a 4-layer network `[2, 3, 2, 1]` (omitting the hidden derivative $o_{pj}(1-o_{pj})$ increases gradient error by $1.4 \times 10^8\times$ to `1.90e-2`).
   - **Symmetry breaking (p. 10)**: If all hidden weights start equal ($w = 0.00$ or $w = 0.25$), all hidden units receive identical gradients forever (`max |w_h0 - w_h1| = 0.00`), locking the network at $E = 0.5000$ (`2/4` on XOR at $w=0.00$) or collapsing it into a 1-hidden-unit threshold at $E = 0.3419$ (`3/4` on XOR at $w=0.25$).
   - **Momentum ($\alpha = 0.9$, Eq. 16) & Chauvin's Hidden-Unit Scaling (p. 12)**: Adding momentum $\alpha = 0.9$ accelerates XOR convergence by **`7.90x`** (`201.2` vs `1,590.1` sweeps), and increasing hidden units $H$ from `2` to `32` monotonically reduces mean sweeps from `593.3` down to `224.7`.
2. **Exact Reproduction of the 1986 Discovered Internal Representations (`E1, E3, E4`)**:
   - **4-Bit Parity (`4 -> 4 -> 1`, Fig. 6 & Table 3, p. 13)**: All 4 hidden units learn equal input weights with staggered biases, turning themselves into a **thermometer counter** of $|x| \in \{0, 1, 2, 3, 4\}$ that matches Table 3 on page 13 bit-for-bit: `|x|=0 -> 1111`, `|x|=1 -> 1011`, `|x|=2 -> 1010`, `|x|=3 -> 0010`, `|x|=4 -> 0000` (`16/16 = 100%`).
   - **$8\text{–}3\text{–}8$ Bottleneck Encoder (Fig. 7 & Table 5, p. 15)**: Compresses 8 one-hot vectors into 3 hidden units using corners plus $\sim 0.5$ graded values (`8/8 = 100%` when using the paper's page 9 `{0.1, 0.9}` targets; raw `{0.0, 1.0}` targets stall at `4/8` due to sigmoid tail saturation on the 7/8 zero targets!).
   - **6-Bit Mirror Symmetry (`6 -> 2 -> 1`, Fig. 9, p. 17 / *Nature* Fig. 2)**: Converges in `1,084` sweeps at $\eta = 0.1$ (vs `1,206` in the paper) to `64/64 = 100%` accuracy by learning equal-and-opposite mirror weights in the exact **$1 : 2 : 4$ geometric ratio** (`[-6.17, +3.10, +12.25, -12.25, -3.10, +6.17]` and `[+6.00, -3.01, -11.91, +11.91, +3.01, -6.00]`).
   - **24-Person / 12-Relation English & Italian Family Trees (*Nature* Figs. 3–5)**: Generalizes to `4/4` (`100%`) held-out relational triples (`Christopher + has-wife -> Penelope`, `Roberto + has-wife -> Maria`, etc.) while dedicating hidden units in the 6-unit person embedding to **Nationality** (`English = 0.94 vs Italian = 0.06`) and **Generation**.
   - **Order-3 $T\text{–}C$ Problem with $3 \times 3$ Spatial Weight Sharing (ICS-8506 Figs. 13–15, pp. 22–25)**: Both the 5-square `T` and 5-square `C` have the exact same pairwise squared-distance spectrum `{1: 4, 2: 2, 4: 2, 5: 2}` across all 4 rotations (`21` vs `20` overlapped $3 \times 3$ receptive fields, p. 24), and a weight-shared $3 \times 3$ receptive field layer learns the 4-fold symmetric **ON-center (`+6.98`) / OFF-surround (`-4.35`) protrusion detector** of Figure 15A (`8/8 = 100%`).

---

## 8. Richard S. Sutton (1988)
**Paper**: *Learning to Predict by the Methods of Temporal Differences* (*Machine Learning*, Vol. 3, No. 1, pp. 9–44, plus Erratum on p. 377)
- **Detailed context**: `docs/sutton-1988-context.md`
- **Verified code**: `research/sutton-1988/work/sutton1988_experiments.py`

### What the paper is about
In multi-step prediction problems (like predicting Monday through Friday whether it will rain on Saturday, or evaluating board positions during a game), conventional supervised learning waits until the final outcome $z$ is known and updates every step $t$ by $(z - P_t)$. Sutton introduces **Temporal-Difference ($\text{TD}(\lambda)$) learning**, which updates each prediction $P_t$ immediately on step $t+1$ using the **change in successive predictions ($P_{t+1} - P_t$)**—and proves that $\text{TD}(0)$ not only uses less memory and compute, but actually **converges to more accurate predictions from finite experience** than supervised learning.

### What he speaks about (and what we verified)
1. **Theorem 1 ($\text{TD}(1) \equiv \text{Widrow–Hoff}$) & Incremental Eligibility Traces (Equations 2–4, pp. 13–16)**:
   - Because total error telescopes as $z - P_t = \sum_{k=t}^m (P_{k+1} - P_k)$, updating weights at each step by $\Delta w_t = \alpha (P_{t+1} - P_t) e_t$ where $e_t = \nabla_w P_t + \lambda e_{t-1}$ with $\lambda = 1$ ($\text{TD}(1)$) produces the **exact same per-sequence weight change** as offline Widrow–Hoff supervised learning (`max diff = 8.88e-16` on tabular features and `7.77e-16` on dense features across 1,000 random walks) using $O(d)$ memory instead of storing the entire trajectory.
2. **Why Bootstrapping Beats Supervised Learning: The "Novel $\to$ Bad $\to$ Win" Game (Figure 1, p. 17) vs Non-Markovian Counterexample (p. 18)**:
   - Suppose a `Bad` state wins only $10\%$ of the time (`9` losses, `1` win), and on the single game where a `Novel` state appears, it transitions into `Bad` and happens to win (`Novel -> Bad -> Win`).
   - Supervised learning ($\text{TD}(1)$) pairs `Novel` directly with the final outcome `Win` ($z=1$), converging to $P(\text{Novel}) = 1.0000$ (**error `0.9000`**!).
   - Linear $\text{TD}(0)$ ($\lambda = 0$) pairs `Novel` with `Bad`'s learned value, eliminating all random noise that occurred *after* `Bad` was entered, and converges to **`P(Novel) = 0.1000, P(Bad) = 0.1000`** (**error `0.0000`**!).
   - Conversely, on Sutton's page 18 non-Markovian counterexample (where `Bad` *always* wins when preceded by `Novel` and *always* loses otherwise, but both paths share a single aliased `Bad` feature), $\text{TD}(0)$ is misled by the Markov assumption (`error = 0.9000`) while Monte Carlo $\text{TD}(1)$ remains unbiased (`error = 0.0000`).
3. **The 5-State Bounded Random Walk (`A..G`), Theorems 2 & 3, and the Page 377 Erratum (`E3`)**:
   - On the 5-state random walk ($B, C, D, E, F$, starting at $D$, terminating at $A \to 0$ or $G \to 1$, true probabilities $\frac{1}{6}, \frac{2}{6}, \frac{3}{6}, \frac{4}{6}, \frac{5}{6}$):
     - **Theorem 2 (p. 24)**: Both $\text{TD}(0)$ and $\text{TD}(1)$ converge in the mean to the ideal predictions $(I - Q)^{-1} h$ as $n \to \infty$ because $S = D(I - Q) + (D(I - Q))^T$ is strictly positive definite (eigenvalues `[0.3831, 1.1972, 2.8188, 4.8028, 8.7982] > 0`).
     - **Theorem 3 & Figure 3 (Erratum p. 377)**: When a finite 10-walk training set is presented repeatedly until convergence, $\text{TD}(0)$ converges to the **exact Maximum-Likelihood Markov model** $(I - \hat{Q})^{-1} \hat{h}$ (`max diff = 3.50e-7`), achieving **`RMS = 0.1199`**, whereas Widrow–Hoff $\text{TD}(1)$ converges to the sample-mean outcomes (`max diff = 1.48e-7`), achieving a $32.4\%$ worse **`RMS = 0.1773`**. *(Note: in the original 1988 printing on p. 20, Figure 3 accidentally printed Figure 5's graph; Sutton's Erratum on p. 377 published the true monotonic Figure 3.)*
     - **Single Presentation (Figures 4 & 5, pp. 21–23)**: When the 10 walks are presented only once, $\lambda = 0.3$ achieves the lowest error (**`RMS = 0.0864`** vs `0.0887` at $\lambda=0.0$ and `0.1786` at $\lambda=1.0$), and sweeping backward through each walk offline (p. 23) improves single-pass $\text{TD}(0)$ to `0.0882`.

---

## 9. Yann LeCun, Léon Bottou, Yoshua Bengio & Patrick Haffner (1998)
**Paper**: *Gradient-Based Learning Applied to Document Recognition* (*Proceedings of the IEEE*, Vol. 86, No. 11, pp. 2278–2324, November 1998)
- **Detailed context**: `docs/lecun-1998-context.md`
- **Verified code**: `research/lecun-1998/work/lecun1998_experiments.py`

### What the paper is about
This 46-page monograph argues that hand-crafted feature extractors and separately tuned document-processing stages should be replaced by **end-to-end gradient-based learning**. Part 1 introduces **LeNet-5** and benchmarks 25 classifiers on MNIST (`60,000` train / `10,000` test); Part 2 introduces **Space-Displacement Neural Networks (`SDNN`)** and **Graph Transformer Networks (`GTNs`)**, which generalize backpropagation from fixed-size vectors to **directed acyclic graphs (DAGs)** representing alternative segmentations, character interpretations, and stochastic grammars.

### What they speak about (and what we verified)
1. **Exact `LeNet-5` Architecture (`60,000` Trainable Parameters, `340,908` Connections, Section II-B, pp. 7–9)**:
   - None of the four GitHub repos we cloned (`pop123-ux`, `jmlee8939`, `VTongTV`, `SunnyHaze`) simultaneously got all three 1998 architectural rules right (`Table I` `C3` connections + trainable `S2`/`S4` subsampling + $7\times 12 = 84$ Euclidean RBF bitmaps).
   - Our `AuthenticLeNet5_1998` reproduces every single number in Section II-B to the exact integer:
     - `C1` ($6 \times 28 \times 28$, $5\times 5$ conv): `156` params, `122,304` connections
     - `S2` ($6 \times 14 \times 14$, $2\times 2$ sum $\times$ trainable weight $+$ bias): `12` params, `5,880` connections
     - `C3` ($16 \times 10 \times 10$, $5\times 5$ conv connected via **Table I**'s 60 kernels): `1,516` params, `151,600` connections (saves `37.3%` vs dense `2,416` params)
     - `S4` ($16 \times 5 \times 5$, $2\times 2$ trainable subsampling): `32` params, `2,000` connections
     - `C5` ($120 \times 1 \times 1$, $5\times 5$ conv): `48,120` params, `48,120` connections
     - `F6` ($84$ units): `10,164` params, `10,164` connections
     - `RBF` ($10$ Euclidean distance units $y_i = \sum_{j=1}^{84} (x_j - w_{ij})^2$ with fixed $\pm 1$ $7\times 12$ stylized digit bitmaps): `0` trainable params, `840` connections
     - **Total**: **`60,000` trainable parameters** and **`340,908` connections**.
2. **Scaled Hyperbolic Tangent, Gauss–Newton Diagonal Hessian, & The RBF Collapse Problem (Appendices A–C & Section II-C)**:
   - Setting $f(a) = 1.7159 \tanh(\frac{2}{3}a)$ makes $f(\pm 1) = \pm 0.999997$ and places the maximum second derivative $|f''(a)|$ at $a = \pm 0.9877 \approx \pm 1$, keeping `F6` unsaturated when matching $\pm 1$ RBF bitmap targets.
   - If RBF centers are allowed to adapt under plain MSE (Eq. 8), all 10 class centers collapse to a single vector (`spread = 0.0017`, `class margin = 0.0000`, `12.5%` accuracy); adding the competitive **MAP log-sum-exp criterion** (Eq. 9) prevents collapse (`spread = 7.4574`, `margin = 26.86`, `100.0%` accuracy).
3. **Space-Displacement Neural Networks (`SDNN`, Section VII, p. 27)**:
   - Evaluating `AuthenticLeNet5_1998` in a single convolutional pass across a wide $32 \times 92$ multi-digit canvas matches cropping 16 separate $32 \times 32$ windows to **`1.53e-5`**, spots `[2, 7, 4]` at their exact positions, and cuts `C1..S4` multiply-accumulates by **`4.43x`** (`1,017,264` vs `4,508,544` MACs).
4. **Graph Transformer Networks (`GTNs`, Sections IV–VIII & X)**:
   - We verified that backpropagating through the **Forward Scorer** ($\text{logadd}$ over parallel segmentation paths, Eqs. 13–16) matches central finite differences to **`8.66e-11`**, aggregates parallel segmentations of the same string (`1.5069 < 1.8000`) where Viterbi picks a wrong single path, and trains an over-segmented multi-character recognizer from **string labels alone** (`90/90 = 100.0%` accuracy, check confidence $\exp(-E_{\text{dforw}}) = \mathbf{1.000}$, whereas Non-Discriminative Viterbi collapses all penalties to `0.00032` with `confidence = 0.044`).

---

## 10. Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton (2012)
**Paper**: *ImageNet Classification with Deep Convolutional Neural Networks* (*Advances in Neural Information Processing Systems 25 (NIPS 2012)*, pp. 1097–1105)
- **Detailed context**: `docs/krizhevsky-2012-context.md`
- **Verified code**: `research/krizhevsky-2012/work/krizhevsky2012_experiments.py`

### What the paper is about
Krizhevsky, Sutskever, and Hinton show that an 8-layer deep Convolutional Neural Network (**AlexNet**, `60.97M` parameters, `659,272` neurons) trained directly on raw RGB pixels across two GPUs can classify 1.2 million high-resolution ImageNet images into 1,000 categories with **`15.3%` top-5 error** (beating the best hand-engineered SIFT + Fisher Vector pipelines at `26.2%` by more than 10 percentage points).

### What they speak about (and what we verified)
1. **Exact 2012 Two-GPU Split-Stream Architecture (`60,965,224` Params, `659,272` Neurons, `724,406,816` MACs, Section 3.5 & Fig. 2)**:
   - Note that `torchvision.models.alexnet` (`61,100,840` params) actually implements Krizhevsky's 2014 single-stream variant (`64->192->384->256->256` with `groups=1` and no LRN), **not** the 2012 NIPS paper!
   - In the authentic 2012 two-GPU architecture (`96 -> 256 -> 384 -> 384 -> 256 -> 4096 -> 4096 -> 1000` with `groups=2` in `conv2, conv4, conv5` and cross-GPU communication only in `conv3` and `fc6..8`):
     - All 5 convolutional layers combined contain only **`3.83%`** of the parameters (`2,334,080 / 60,965,224`), yet perform **`91.91%`** of all multiply-accumulates (`665,784,864 / 724,406,816`).
     - Conversely, the 3 fully-connected layers (`fc6, fc7, fc8`) contain **`96.17%`** of the parameters (`58,631,144`) while performing only **`8.09%`** of the MACs.
     - **Figure 2 Caption Audit**: Why does Figure 2's caption list the first layer's neuron count as `253,440` instead of $55 \times 55 \times 96 = 290,400$? Because $55 \times 48 \times 96 = \mathbf{253,440}$ (multiplying one spatial dimension `55` by the per-GPU filter count `48` instead of `55 * 55 * 96`)!
2. **Non-Saturating `ReLU` (`7.80x` Faster Training) & Section 5 $+1$ Bias Initialization**:
   - Inspecting Krizhevsky's original 2012 `cuda-convnet` configs (`layers-19pct.cfg`, `layers-80sec.cfg`) showed that `pool2` and `pool3` use spatial **average pooling**: odd-symmetric $\tanh(-z) = -\tanh(z)$ cancels positive/negative half-cycles of oriented textures and saturates in its tails. On our 4-layer CNN with independently tuned optimal learning rates, `ReLU` reached $\le 25\%$ training error in **`10` iterations** vs **`78` iterations** for $\tanh$ (**`7.80x` faster**, matching Figure 1's $\sim 6\times$ on CIFAR-10).
   - With Section 5's $\mathcal{N}(0, 0.01^2)$ weight initialization, initializing biases to $0$ attenuates the signal across 5 conv layers to `0.00059`, whereas Krizhevsky's **$+1$ bias initialization** in `conv2, conv4, conv5` keeps **`100.0%`** of `conv5` ReLUs active with mean `1.0314` ($1,748\times$ stronger initial signal).
3. **Local Response Normalization (`LRN`), Overlapping Pooling ($s=2, z=3$), PCA Color Augmentation, & `Dropout` ($p=0.5$)**:
   - **LRN**: Krizhevsky's formula $b^i_{x,y} = a^i_{x,y} / (k + \alpha \sum_{j=i-2}^{i+2} (a^j_{x,y})^2)^\beta$ multiplies the 5-channel sum directly by $\alpha = 10^{-4}$ (whereas PyTorch's `F.local_response_norm` divides $\alpha$ by `size=5`, causing a `31.95` discrepancy unless `alpha=5e-4` is passed), producing **`56.0%` lateral inhibition** when an adjacent filter fires strongly.
   - **Overlapping Pooling ($s=2, z=3$)**: Preserves 1-pixel shift cosine similarity (`0.7071` vs `0.0000` for non-overlapping $s=2, z=2$).
   - **PCA RGB Augmentation**: Extracts the luminance eigenvector $p_1 = [0.5670, 0.5757, 0.5891]^T \approx \frac{1}{\sqrt{3}}[1, 1, 1]^T$ (`90%` of variance) and chromaticity axes of the $3\times 3$ RGB covariance matrix.
   - **Dropout ($p=0.5$ with $\times 0.5$ test scaling)**: Cuts hidden-unit co-adaptation by **`2.52x`** (`0.0058` vs `0.0146`) and boosts held-out test accuracy from `73.3%` to `77.1%`.

---

## 11. Ashish Vaswani et al. (2017)
**Paper**: *Attention Is All You Need* (*Advances in Neural Information Processing Systems 30 (NIPS 2017)* / `arXiv:1706.03762`)
- **Detailed context**: `docs/vaswani-2017-context.md`
- **Verified code**: `research/vaswani-2017/work/vaswani2017_experiments.py`

### What the paper is about
Recurrent sequence models (RNNs/LSTMs) process tokens one by one ($h_t = f(h_{t-1}, x_t)$), which prevents parallelization across positions during training and forces signals between distant words to travel through $O(n)$ sequential steps. Vaswani et al. introduce **The Transformer**, an Encoder-Decoder architecture built **solely out of self-attention and point-wise feed-forward networks**, connecting any two positions in **$O(1)$ path length** and **$O(1)$ sequential steps**.

### What they speak about (and what we verified)
1. **Scaled Dot-Product Attention & Why $1/\sqrt{d_k}$ Prevents Softmax Saturation (Section 3.2.1, Eq. 1 & Footnote 4)**:
   - In $\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$, if $q, k \sim \mathcal{N}(0, I_{d_k})$, the unscaled dot product $q \cdot k$ has variance $d_k$ (`16.55` at $d_k=16$ up to `1023.96` at $d_k=1024$).
   - Without $1/\sqrt{d_k}$ scaling, at $d_k = 1024$ the largest softmax weight saturates to **`0.9574`** and the softmax Jacobian norm $\|J_{\text{softmax}}\|_F$ collapses by **`4.2x`** (`0.05750` vs `0.24115`). Dividing by $\sqrt{d_k}$ locks the logit variance at **`1.0000`**, max attention at **`0.1660`**, and Jacobian norm at **`0.24115`** across all dimensions.
2. **Multi-Head Subspace Resolution vs Single-Head Averaging Dilution (Section 2 & 3.2.2)**:
   - On page 2, the authors note that compressing a sequence via attention comes *"at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention."*
   - When a query token must simultaneously retrieve subspace $A$ from position $j_A$ and orthogonal subspace $B$ from position $j_B$ with the exact same total parameter count, Single-Head Attention ($h=1, d_k=d_v=16$) cannot unmix the two positions (`MSE = 0.37236`), whereas Multi-Head Attention ($h=2, d_k=d_v=8$) dedicates one head to $j_A$ and one head to $j_B$, achieving **`MSE = 0.00048` (`782.1x` lower error)**.
3. **Sinusoidal Positional Encoding: Exact $2\times 2$ Linear Offset Operator $M_k$ & Shift-Invariant Dot Product (Section 3.5, p. 6)**:
   - Because $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$ and $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$, for any fixed relative offset $k$, $PE_{pos+k} = M_k PE_{pos}$ where $M_k$ is a block-diagonal matrix of $2 \times 2$ rotation matrices (`max error = 2.05e-14` across all positions and offsets).
   - Furthermore, by $\sin(\alpha)\sin(\beta) + \cos(\alpha)\cos(\beta) = \cos(\alpha - \beta)$, the unprojected inner product $PE_{pos}^T PE_{pos+k} = \sum_{i=0}^{d_{\text{model}}/2-1} \cos(k \omega_i)$ depends **strictly on the relative distance $k$ alone** (`std across pos 0..100 = 8.78e-14`).
4. **Causal Masking, 3-Way Weight Tying + $\sqrt{d_{\text{model}}}$ Scaling, Label Smoothing, & Table 1 Crossover (`E4`)**:
   - **Causal Masking**: Upper-triangular $-\infty$ masking in Decoder self-attention gives **`0.00e+00`** prefix leakage when future tokens change; omitting the mask during teacher-forced training allows the model to cheat (`100.0%` teacher-forced accuracy) and causes autoregressive greedy inference to collapse to **`8.6%`** (vs **`100.0%`** with the causal mask).
   - **Why $\sqrt{d_{\text{model}}}$ Embedding Scaling Exists (Section 3.4, p. 5)**: Because the shared matrix $W \in \mathbb{R}^{V \times d_{\text{model}}}$ is used as the pre-softmax linear projection, its weights are initialized with fan-in $\text{std} = d_{\text{model}}^{-0.5}$; multiplying embedding lookups by $\sqrt{d_{\text{model}}}$ restores embedding variance from $1/d_{\text{model}}$ back to $1.0$ so it matches the $O(1)$ sinusoidal positional encoding (whereas using PyTorch's default `nn.Embedding` $\text{std}=1.0$ with `* sqrt(d_model)` drowns out positional encodings by $d_{\text{model}}\times$!).
   - **Label Smoothing ($\epsilon_{ls} = 0.1$, Section 5.4, p. 8)**: Increases nominal perplexity from `1.0016` to `1.1014` while preserving `100.0%` autoregressive accuracy, confirming p. 8 (*"This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score"*).
   - **Unpublished "Attention over Parameters (`AOP`)" Found in the arXiv `.tex` Archive (`E1`)**: Inspecting the raw `arXiv:1706.03762` TeX tarball uncovered `parameter_attention.tex` (commented out in `ms.tex`), in which the authors showed that $\text{FFN}(x) = \text{ReLU}(x W_1) W_2$ is itself attention over trainable key-value parameters and reported that replacing all FFNs with 16-head parameter attention (`AOP_2`, `65M` params) achieved **`4.86` PPL and `25.9` BLEU** on WMT14 EN-DE dev (beating the base Transformer's `4.92` PPL and `25.8` BLEU).

---

## 12. How Each of the 11 Foundational Papers Helps Current Clover on the GPU Server (`8 × NVIDIA RTX 3090`)

See [clover-foundations-to-clover-gpu-context.md](clover-foundations-to-clover-gpu-context.md) for the full cycle-by-cycle GPU context, predictions, return-to-context discoveries, and evaluation scripts (`cycle1..cycle6` in `src/tools/hybrid-ncm-k3/`):

1. **Cycle 1 — McCulloch & Pitts (1943) + Minsky & Papert (1969) (`k3_ontology_graph.py`, `k3_memory_graph.py`):**
   - Replaced flat order-1 token/phrase overlap triggers with **McCulloch-Pitts All-or-None Inhibitory Veto** ($\neg \bigvee_{j \in I_i} N_j$) and **Minsky-Papert Order-$k$ "One-in-a-Box" Conjunctive Predicates** ($\bigwedge_{m=1}^k \bigvee_{x \in B_m} x$), cutting One-in-a-Box and out-of-domain stress false positives from **`16 / 16` (`100.0%` FP) $\to$ `0 / 16` (`0.0%` FP)** while preserving **`10/10` Ontology** and **`6/6` Memory** positive recall.
2. **Cycle 2 — Alan Turing (1950) + Newell & Simon (1956) (`k3_ontology_graph.py`):**
   - Added Newell & Simon's **`CSm` 3-number structural signature pre-filter** $(K_L, K_R, J)$ with our verified `*2.07` contraction fix ($J_{\text{rule}} \ge J_{\text{target}}$) and **`VCt` contraction** to `CloverGraphTraversal.paths` and `CloverSymbolicVerifier.match_formula_csm_lmc`, plus Turing Section 6.1 **$M_0 \to M_1$ self-reference stratification** (`check_turing_self_reference_stratification`). Eliminated **`90.44%`** of `LMc` unifications (**`10.46x` fewer**), raised contracted `*2.07` theorem recall from **`2/8` $\to$ `8/8` (`100%`)**, cut 2-hop ontology DFS steps from **`630 -> 28` (`22.5x` fewer)** with `100%` identical causal paths, and stratified **`3/3`** diagonal paradox prompts.
3. **Cycle 3 — Frank Rosenblatt (1958) + John Hopfield (1982) (`k3_memory_graph.py` on `cuda:7`):**
   - Added Rosenblatt's **Parasitic Zero-Sum $\gamma$-System** ($\sum_i \Delta V_i = 0$ across the `5.4 : 1` imbalanced `systems` vs `philosophy` branches) and Hopfield's **Bipolar $\{-1, +1\}$ Associative Constellation Cleanup & Familiarity Gate** on GPU (`cuda:7`). Restored cross-branch `top-3` representation from **`3/4 -> 4/4` (`100%`)**, raised Hopfield constellation cleanup mean Jaccard from **`0.3777` (`4/25` unipolar $\{0,1\}$) $\to$ `0.9334` (`24/25` bipolar $\{-1,+1\}$ on `cuda:7`)**, and achieved **`1.0000` ROC AUC** (`10/10` pass, `0/10` FP) on familiarity detection.
4. **Cycle 4 — Rumelhart, Hinton & Williams (1986) + Richard Sutton (1988) (`k3_generative_compiler.py`, `k3_memory_graph.py` on `cuda:7`):**
   - Added `SuttonTDStatementPredictor` and `compute_td_lambda_path_credit` with **Rumelhart's `{0.1, 0.9}` non-saturating targets** (`90.1x` stronger recovery slope than `{0.001, 0.999}`) and **Sutton's $\text{TD}(\lambda)$ / Theorem 3 Batch Backward `TD(0)` Absorbing Markov Model** $(I - \hat{Q})^{-1} \hat{h}$ on `cuda:7`. Reduced statement-value RMSE against the exact Markov MLE from **`0.1253` (`TD(1)`) $\to$ `0.0860` (`TD(0.3)`) $\to$ `0.0004` (Batch Backward `TD(0)`, `313x` lower error)** and localized the exact offending statement cliff on **`6 / 6` (`100%`)** faulty code trajectories.
5. **Cycle 5 — Yann LeCun et al. (1998) (`k3_ontology_graph.py`, `k3_memory_graph.py`):**
   - Added the **GTN Forward Scorer (`logadd` Eq. 13–16)** over parallel corroborating provenance paths (`score_targets_gtn_forward`, `compute_gtn_forward_confidence`) and exact posterior **Discriminative Forward Confidence** $\exp(-E_{\text{dforw}})$ (Eq. 17, p. 39) with directed epistemic flow from superseded hypotheses into active corrections. Raised parallel-corroboration accuracy from **`0 / 6` (Viterbi `min`) $\to$ `6 / 6` (`100%`, GTN Forward)**, broke the `CVE-2024-6387` Viterbi tie in favor of parent hub `CWE/362`, and ranked the active `MEM-CORR-*` entry `#1` across **`3 / 3`** correction chains.
6. **Cycle 6 — Alex Krizhevsky et al. (2012) + Ashish Vaswani et al. (2017) (`k3_memory_graph.py` on `cuda:7`):**
   - Added `multihead_lrn_subspace_retrieve` on `cuda:7` combining **Vaswani's $h = 3$ Orthogonal Subspace Heads** (*Concept Principle*, *Empirical Measurement*, *Superseding Correction*) and **$1/\sqrt{d_k}$ logit scaling** (`520.64 -> 1.017` variance, `1.82x` larger softmax Jacobian norm at $d_k = 512$) with **Krizhevsky's Cross-Channel LRN** operating strictly inside the monotonic regime $x < \sqrt{k / ((2\beta - 1)\alpha)}$. Lifted 3-facet subspace coverage at `top_k = 3` from **`4 / 6` (`2.67/3` facets, single-head) $\to$ `6 / 6` (`3.00/3` facets, `18/18` exact head winners)**.

