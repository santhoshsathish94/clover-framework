# Frank Rosenblatt (1958) — *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain* — Context

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh.

## 1. Status

- **E1 (Clone & run reference repositories on k3 CPU)**: DONE
- **E2 (Exact 1958 Predominant Phase Equations: $P_a$ Eq. 1, $P_c$ Eq. 2, $P_{c\min}$ Eq. 3, and Fig. 4c balanced-inhibition flattening)**: DONE
- **E3 (Exact 1958 Postdominant Learning Dynamics: Table 1 $\alpha/\beta/\gamma$ systems, $\Sigma$ vs $\mu$ discrimination, and Ideal vs Differentiated environment $P_r / P_g$)**: DONE
- **E4 (Bivalent error-correction, Spontaneous unsupervised organization p. 404, and Relational judgment boundary pp. 404–405)**: DONE

## 2. What this is

Authors say (Frank Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain", *Psychological Review*, Vol. 65, No. 6, pp. 386–408, November 1958):
- Contrast between **coded representational memory** (one-to-one mapping or stored templates requiring systematic comparison) and the **empiricist / connectionist position**: *"the images of stimuli may never really be recorded at all, and ... the central nervous system simply acts as an intricate switching network, where retention takes the form of new connections, or pathways, between centers of activity... Whatever information is retained must somehow be stored as a preference for a particular response; i.e., the information is contained in connections or associations rather than topographic representations."* (pp. 386–387)
- Unlike deterministic symbolic-logic nerve nets (McCulloch & Pitts 1943), which require exact wiring and lack equipotentiality, Rosenblatt formulates the **theory of statistical separability** in terms of probability theory for networks whose connections at birth are largely random (pp. 387–389).
- **Organization of a Photo-Perceptron (Fig. 1 & Fig. 2, pp. 389–391)**:
  1. **Sensory retina ($S$-points)** responding on an all-or-nothing basis.
  2. **Association cells ($A$-units)**: each $A$-unit receives $x$ excitatory and $y$ inhibitory connections from randomly chosen origin points in the retina (or via a focalized contour-sensitive projection area $A_I$) and fires in the **predominant phase** if $e - i \ge \theta$.
  3. **Response cells ($R_1, \dots, R_{N_R}$)**: each $R_r$ receives forward connections from a random **source-set** of $A$-units and sends inhibitory feedback to the complement of its source-set in the **postdominant phase**.
- **Predominant phase probabilities (Equations 1–3, pp. 392–394)**: $P_a$ (Eq. 1) is the expected fraction of $A$-units activated by a stimulus illuminating fraction $R$ of the retina; $P_c$ (Eq. 2, with minimum $P_{c\min} = (1-L)^x(1-G)^y$ in Eq. 3) is the conditional probability that an $A$-unit responding to $S_1$ also responds to $S_2$.
- **Value dynamics ($\alpha, \beta, \gamma$ systems, Table 1, p. 391) and response discrimination ($\Sigma$-system sum vs $\mu$-system mean, pp. 394–399)**: Each $A$-unit carries a modifiable plastic **value $V$**. In the **$\alpha$-system** (uncompensated gain), active $A$-units in the reinforced source-set gain $\Delta V = +1$; in the **$\beta$-system** (constant feed), each source-set gains a fixed total $K$; in the **$\gamma$-system** (parasitic zero-sum gain), active $A$-units gain $+1$ at the expense of inactive $A$-units in the same source-set ($\Delta V = -N_{ar}/(N_{Ar}-N_{ar})$), keeping total source-set value constant ($0$).
- **Ideal vs Differentiated environments (Equations 4–11, pp. 396–402)**: In an **ideal environment** (uncorrelated random stimuli, $c_1 = 0$), recall $P_r$ decays toward $0.5$ as $n_{sr} \to \infty$ and generalization $P_g = 0.5$. In a **differentiated environment** where intra-class overlap exceeds inter-class overlap ($P_{c11} > P_a > P_{c12}$), both recall $P_r$ and unseen generalization $P_g$ converge to the **same** asymptote $P_{r\infty} = P_{g\infty} = [1 - (1-P_a)^{N_e}]\Phi(c_1/\sqrt{c_3})$, which approaches $1$ as the number of association cells $N_e$ increases (Eq. 9, p. 400).
- **Bivalent reinforcement (pp. 402–403), Spontaneous organization (p. 404), and the Relational boundary (pp. 404–405)**: Bivalent systems apply $+\Delta V$ and $-\Delta V$ for reward/punishment (trial-and-error learning); adding proportional value decay allows **spontaneous (unsupervised) concept formation**; and the limit of the perceptron's capability lies in **relative judgment and abstraction of relationships** ("Name the object left of the square", pp. 404–405).

## 3. Experiments

### E1: Clone and run reference repositories (`vukrosic/rosenblatt-from-scratch`, `kaisdukes/rosenblatt-perceptron`, `Honghe/perceptron`)
- **Question**: What do the three reference repositories implement, how do their equations relate to Rosenblatt's 1958 paper, and what exact numbers do they produce when executed on `k3` CPU?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 15:48 UTC):
  1. `vukrosic/rosenblatt-from-scratch` will run all 7 scripts (`step1_neuron.py`..`step7_backprop.py`) and pass `tests.py` (`rc=0`), reproducing 14 mistakes on 10,000 cards in `step2_learn.py`, `976/1000` on the 400-photocell / 512-$A$-unit Mark I simulation in `step3_mark1.py`, `1000/1000` convergence within $(R/\gamma)^2$ in `step4_converge.py`, and XOR failure (`3/4`) in `step5_xor.py`.
  2. `kaisdukes/rosenblatt-perceptron` will pass `perceptron_test.py` (`rc=0`) and separate Iris setosa vs versicolor cleanly.
  3. `Honghe/perceptron` will train its 2D perceptron cleanly (`rc=0`).
- **Run**: `scripts/e1_repos.sh` and `scripts/e2_e3_e4_run.sh`. Outputs: `results/e1_repos.out`, `results/e2_e3_e4_run.out`.
- **Observation**:
  - **Repository 1 (`vukrosic/rosenblatt-from-scratch`)**:
    - Commit `a0167b1eb3527362557105c0f8a933d24da6aecc` (MIT License, 9 Python files totaling 648 lines of code plus `README.md` 37 lines and `demo-output.txt` 147 lines).
    - Ran `bash run_all.sh` (`results/e1_repos.out`, `All checks passed`, `rc=0`):
      - `step1_neuron.py`: hand-wired threshold neuron (`AND`, `OR`, `NOT`, and 400-weight left/right punch-card detector scoring `1,000/1,000`).
      - `step2_learn.py`: 400-weight perceptron starting at zero weights (`w=[0]*400, b=0`) learning left vs right punch cards via mistake-driven updates (`w_i += step, b += step` where `step = +1` if `want==1` else `-1`): test score on 2,000 fresh cards rose from `1,036/2,000` (0 cards) to `1,724/2,000` (20 cards), `1,928/2,000` (50 cards), and `2,000/2,000` (200 cards), making only `14` total mistakes across `10,000` cards (last mistake on card `145`).
      - `step3_mark1.py`: Mark I simulation (`400` photocells $\to$ `512` random association units with $x=10$ excitatory, $y=10$ inhibitory, $\theta=2$ $\to$ `4` response units for `X, E, T, O`, `2,048` learnable weights): on average `94` of `512` $A$-units fired per letter (`18.4%`), and test accuracy on 1,000 unseen shifted/noisy letters rose from `250/1,000` (`25.0%`) before training to `905/1,000` after pass 1 and `976/1,000` (`97.6%`) after pass 10 (`241,659` total weight updates).
      - `step4_converge.py`: tested 1,000 random linearly separable 2D problems across margins `gap` $\in \{20, 10, 5, 2, 1\}$ (200 each): 100% converged (`avg` mistakes `2.9, 5.3, 9.5, 19.3, 24.8` vs average theoretical $(R/\text{gap})^2$ ceilings `17, 62, 212, 986, 2613`; maximum mistake count was `30%` of its ceiling).
      - `step5_xor.py`: single neuron on XOR made `480,332` weight changes over `1,000,000` cards, visiting `53` distinct `(w0, w1, b)` states and never getting more than `3/4` cases right; exhaustive grid over `[-10..10]^3` (`9,261` cells) realized `14/16` 2-input Boolean functions (missing only `XOR` `0110` and `XNOR` `1001`); 9-bit parity on all `512` $3 \times 3$ images failed for every polynomial order $k \in \{1..8\}$ (`326..511/512`) and succeeded (`512/512`) only at order $k = 9$ (requiring a middle unit that sees all 9 squares).
  - **Repository 2 (`kaisdukes/rosenblatt-perceptron`)**:
    - Commit `edd29d9f7773ab3442c0e1d2a37facab851b9323` (MIT License, 6 Python files totaling 120 lines plus `data/iris.csv` 150 lines).
    - Ran 10-fold cross-validation on Fisher's Iris dataset (`Setosa` vs `Versicolor`, `learning_rate=0.01`, `epochs=10`): achieved `100.00%` test accuracy on all `10/10` folds (`rc=0`).
  - **Repository 3 (`Honghe/perceptron`)**:
    - Commit `c4caa5a26f933e6363521cb6c2452d91e217496b` (3 Python files, 101 lines).
    - Failed initially under Python 3 (`SyntaxError` on Python 2 `print 'not hit.'` and `xrange`). Repaired in a work copy (`e2_e3_e4_run.sh`) by replacing `xrange` with `range` and parenthesizing `print(...)`: converged in `1`–`2` epochs with 0 test errors (`rc=0`).
- **What it showed**: All three predictions held; furthermore, inspecting the three repositories side-by-side with Rosenblatt's 1958 *Psychological Review* paper showed that all three repositories implement the **bivalent error-correcting rule** ($w \leftarrow w + (y - \hat{y}) x$, which Rosenblatt introduces on pages 402–403 and in his 1960 Mark I work), whereas pages 389–402 of the 1958 paper develop the probabilistic theory of $P_a$, $P_c$, the $\alpha/\beta/\gamma$ value systems (Table 1), $\Sigma$ vs $\mu$ discrimination, and spontaneous organization (p. 404)—none of which are in the repos. We therefore built and tested all of those in E2–E4.

### E2: Exact 1958 Predominant Phase Equations ($P_a$ Eq. 1, $P_c$ Eq. 2, $P_{c\min}$ Eq. 3, and Fig. 4c flattening, pp. 392–394)
- **Question**: Do Rosenblatt's analytical Equations (1), (2), and (3) for $P_a$ and $P_c$ match Monte Carlo simulation of randomly wired $A$-units on a finite retina, and how much does balancing excitatory and inhibitory connections ($x = y = 5$, Fig. 4c, p. 393) flatten $P_a(R)$ across stimulus sizes $R \in [0.1, 0.5]$ compared with purely excitatory connections ($x = 10, y = 0$, Fig. 4b)?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 15:48 UTC):
  1. Analytical Equations (1) and (2) will match Monte Carlo simulation within $< 0.008$ (`Pa`) and $< 0.015$ (`Pc`), and at $\theta = x, y = 0$, Equation (2) will equal Equation (3) $P_{c\min} = (1-L)^x$ to $< 10^{-12}$, while corrupting the $+g_e$ sign in Eq. (2)'s side condition will fail.
  2. Purely excitatory $P_a(R)$ ($x=10, y=0, \theta=3$) will vary by $> 0.80$ over $R \in [0.1, 0.5]$, whereas balanced $x=5, y=5, \theta=2$ will flatten the variation by $> 7\times$.
- **Run**: `scripts/e2_e3_e4_run.sh` (executing `work/rosenblatt1958_models.py`). Output: `results/e2_e3_e4_run.out`.
- **Observation**:
  - **Analytical vs Monte Carlo ($N_A = 100,000$ $A$-units on $N_S = 2,000$ retina)**:
    - `(R=0.30, L=0.40, G=0.15, x=6, y=2, th=2)`: $P_a$ analytical `0.3978` vs MC `0.3965`; $P_c$ analytical `0.5715` vs MC `0.5693` (corrupted side-condition control: `0.2843`).
    - `(R=0.20, L=0.50, G=0.125, x=8, y=0, th=3)`: $P_a$ analytical `0.2031` vs MC `0.2026`; $P_c$ analytical `0.3987` vs MC `0.3970` (corrupted: `0.1137`).
    - `(R=0.40, L=0.25, G=0.1667, x=5, y=5, th=2)`: $P_a$ analytical `0.1663` vs MC `0.1650`; $P_c$ analytical `0.4752` vs MC `0.4728` (corrupted: `0.3118`).
    - `(R=0.50, L=0.30, G=0.30, x=4, y=0, th=4)`: $P_a$ analytical `0.0625` vs MC `0.0615`; $P_c$ analytical `0.2401` vs MC `0.2382` (matches $P_{c\min} = 0.7^4 = 0.2401$).
    - At $x=10, y=0, \theta=10$ (Fig. 6 bottom curve, p. 394): `pc_eq2 = 0.02824752` matched `pc_min_eq3 = (1 - 0.30)^10 = 0.02824752` with difference `0.00e+00`.
    - *Sample-size note*: On our first run with $N_A = 20,000$ at $x=10, y=0, \theta=10$, $P_a = 0.5^{10} = 0.000977$ activated only $\sim 20$ units on $S_1$, so $P_c = 0.0282$ had expected count $0.6 < 1$ (`0/22 = 0.0000`). Testing Monte Carlo at $x=4, \theta=4$ ($P_a = 0.0625$, $\sim 6,150$ active units at $N_A = 100,000$) reduced maximum error across all four configurations to `0.0024`.
  - **Fig. 4c Flattening of $P_a(R)$ across $R \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$**:
    - Purely excitatory ($x=10, y=0, \theta=3$, Fig. 4b): $P_a = [0.0702, 0.3222, 0.6172, 0.8327, 0.9453]$ — spread `0.8751` over $R \in [0.1, 0.5]$ (`0.6231` over $R \in [0.2, 0.5]$).
    - Balanced excitatory & inhibitory ($x=5, y=5, \theta=2$, Fig. 4c): $P_a = [0.0509, 0.1112, 0.1478, 0.1663, 0.1719]$ — spread `0.1209` over $R \in [0.1, 0.5]$ (`0.0607` over $R \in [0.2, 0.5]$, a **$10.3\times$ reduction in sensitivity to stimulus size**!).
- **What it showed**: All predictions held (`< 0.0024` MC error, `0.00e+00` Eq. 3 identity, and $7.2\times$–$10.3\times$ flattening of $P_a(R)$ when $x = y = 5$).

### E3: Exact 1958 Postdominant Learning Dynamics — Table 1 ($\alpha, \beta, \gamma$), $\Sigma$ vs $\mu$, and the Page 400 Inequality $P_{c11} > P_a > P_{c12}$
- **Question**: How do Rosenblatt's three Table 1 value systems ($\alpha$ uncompensated, $\beta$ constant-feed, $\gamma$ parasitic zero-sum) and two discriminators ($\Sigma$ sum vs $\mu$ mean) behave under 3:1 class-frequency imbalance ($n_{s0} = 150, n_{s1} = 50$), how does the page 400 inequality $P_{c11} > P_a > P_{c12}$ govern generalization, and how do recall $P_r$ and unseen generalization $P_g$ evolve with $n_{sr}$ in an Ideal environment vs a Differentiated environment?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 15:48 UTC):
  1. Per reinforcement, net source-set value changes by $+N_{ar}$ in $\alpha$, $+K$ in $\beta$, and $0.0$ in $\gamma$; under 3:1 class-frequency imbalance, the $\alpha$-system will be strongly biased toward the frequent class $R_0$, whereas the $\gamma$-system will maintain zero net source-set value and achieve $> 90\%$ balanced accuracy when $P_{c11} > P_a > P_{c12}$.
  2. In the Ideal environment (random patterns), recall $P_r$ starts near $100\%$ at $n_{sr}=10$ and decays as $n_{sr}$ increases while unseen $P_g \approx 50\%$; in a Differentiated environment, $P_g$ rises with $n_{sr}$ well above chance.
- **Run**: `scripts/e2_e3_e4_run.sh` (executing `work/rosenblatt1958_models.py`). Output: `results/e2_e3_e4_run.out`.
- **Observation**:
  - **Discovery of a geometric violation of $P_{c11} > P_a > P_{c12}$ on centered Horizontal vs Vertical bars (`Regime A`)**:
    - On our first test (`Regime A`: $3 \times 12$ horizontal bars vs $12 \times 3$ vertical bars shifted on a $16 \times 16$ grid), under 3:1 class imbalance ($n_{s0}=150, n_{s1}=50$), the $\alpha$-system had `net_V = (6843.0, 2112.0)` and predicted Class 0 on 100% of test inputs (`Class0=100%, Class1=0%, Balanced Pg=50.0%`), while the $\gamma$-system had `net_V = (0.0, -0.0)` and improved to `Class0=97%, Class1=44%, Balanced Pg=70.5%`.
    - Why did the $\gamma$-system get only `44%` on the minority class in `Regime A`? Measuring $(P_{c11}, P_a, P_{c12})$ directly revealed why: two randomly shifted horizontal bars on a $16 \times 16$ grid often fall on disjoint rows ($0$ overlap), whereas a $3 \times 12$ horizontal bar and a $12 \times 3$ vertical bar **always intersect in a $3 \times 3 = 9$-pixel square** ($9/36 = 25.0\%$ overlap $> R = 36/256 = 14.1\%$). Consequently, `Regime A` had **$P_{c11} = 0.1700 < P_a = 0.1762 < P_{c12} = 0.1831$**, violating Rosenblatt's fundamental inequality $P_{c11} > P_a > P_{c12}$ (p. 400)! Because $P_{c12} > P_a$, reinforcing Class 0 added *positive* expected value $+n_{s0}(P_{c12} - P_a) > 0$ to $R_0$ on Class 1 vertical bars!
    - When we tested `Regime B` (separated class pools where $P_{c11} = 0.2370 > P_a = 0.1585 > P_{c12} = 0.0939$, satisfying $P_{c11} > P_a > P_{c12}$) under the exact same 3:1 frequency imbalance ($n_{s0}=150, n_{s1}=50$):
      - `alpha`: `Class0 = 100%, Class1 = 53%, Balanced Pg = 76.5%` (still biased by the $3\times$ uncompensated value sum).
      - `gamma`: **`Class0 = 100%, Class1 = 100%, Balanced Pg = 100.0%`**!
      - `bivalent_gamma`: **`Class0 = 100%, Class1 = 100%, Balanced Pg = 100.0%`**!
  - **Ideal Environment vs Differentiated Environment across $n_{sr} \in \{10, 50, 150\}$ (`E3.2`)**:
    - **Ideal environment (uncorrelated random 36-pixel patterns)**: Memorized recall $P_r$ dropped from **`100.0%`** ($n_{sr}=10$) to **`92.0%`** ($n_{sr}=50$) and **`79.3%`** ($n_{sr}=150$) as overlapping random associations interfered (matching Fig. 7–8, pp. 397–398), while unseen generalization $P_g$ stayed at chance (**`54.0%, 49.0%, 51.0%`**).
    - **Differentiated environment**: Unseen generalization $P_g$ rose from **`64.5%`** ($n_{sr}=10$) to **`79.5%`** ($n_{sr}=150$), approaching memorized recall $P_r$ (**`86.7%`** at $n_{sr}=150$, matching Fig. 11, p. 401).
- **What it showed**: Both predictions held, and diagnosing `Regime A` vs `Regime B` provided a direct quantitative demonstration of Rosenblatt's page 400 theorem ($P_{c11} > P_a > P_{c12}$) and why the parasitic zero-sum $\gamma$-system eliminates class-frequency bias when $P_{c11} > P_a > P_{c12}$.

### E4: Spontaneous Unsupervised Organization (p. 404) and the Relational Judgment Boundary (pp. 404–405)
- **Question**: (a) Can a perceptron with **zero teacher labels** spontaneously separate two dissimilar stimulus classes when its $A$-unit values undergo proportional decay ($-\lambda V$) and automatic winner-take-all self-reinforcement, as claimed by Rosenblatt on page 404? (b) On a 2-object scene ("Left object" and "Right object"), how do slot-local $A$-units vs cross-slot $A$-units perform on a concrete conjunction ("Left is Square AND Right is Circle") vs a relational judgment ("Left object has the SAME shape as Right object", pp. 404–405)?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 15:48 UTC):
  1. With proportional value decay ($\lambda = 0.05$) and zero-sum $\gamma$ competition, automatic winner self-reinforcement will spontaneously separate the two classes with $> 95\%$ accuracy across 20 random trials, whereas the uncompensated $\alpha$ control without decay/competition will collapse to $50.0\%$ (one response winning 100% of stimuli).
  2. Slot-local $A$-units will learn the concrete conjunction to $\sim 100\%$ accuracy but fail ($\le 50\%$) on the relational equality judgment, whereas cross-slot $A$-units will learn both to $100\%$.
- **Run**: `scripts/e2_e3_e4_run.sh` (executing `work/rosenblatt1958_models.py`). Output: `results/e2_e3_e4_run.out`.
- **Observation**:
  - **E4.1 — Spontaneous Unsupervised Organization (p. 404)**:
    - Across 20 independent random seeds (300 unlabeled stimuli per trial), proportional decay ($\lambda = 0.05$) + $\gamma$ self-reinforcement achieved **`100.0%` mean unsupervised class separation** (assigning `R=0` to one class and `R=1` to the other without ever being told "right" or "wrong"), whereas the negative control (uncompensated $\alpha$ self-reinforcement without decay/competition) achieved **`50.0%`** (whichever response won step 1 accumulated positive weight and won all 300 steps).
  - **E4.2 — Concrete Conjunction vs Relational Judgment Boundary (pp. 404–405)**:
    - `Slot-local A-units` (each $A$-unit connected to either the Left slot or the Right slot): achieved **`99.8%`** on the concrete conjunction (`Left=Square AND Right=Circle`), but only **`42.0%`** (worse than chance) on the relational judgment (`Left shape == Right shape`), because for any slot-local $A$-unit the marginal distribution of Left shapes and Right shapes is identical between the "Same shape" class (`(Sq,Sq), (Cir,Cir)`) and the "Different shape" class (`(Sq,Cir), (Cir,Sq)`), so $P_{c11} == P_{c12}$!
    - `Cross-slot A-units` (each $A$-unit drawing 2 excitatory connections from Left and 2 from Right with $\theta=4$): achieved **`100.0%`** on the concrete conjunction and **`100.0%`** on the relational judgment.
- **What it showed**: Both predictions held (`100.0%` vs `50.0%` on spontaneous organization; `99.8%` vs `42.0%` on slot-local concrete vs relational tasks, and `100.0%` with cross-slot $A$-units).

## 4. How it works

1. **Fixed random feature projection ($S \to A$) followed by plastic linear readout ($A \to R$) (pp. 389–394)**: The 1958 Photo-Perceptron does not connect sensory inputs directly to output weights; it first projects the binary retina ($S$-points) through a large layer of randomly wired association units ($A$-units, each with $x$ excitatory and $y$ inhibitory origin points and threshold $\theta$) and modifies only the $A$-unit values $V$ ($A \to R$ weights) (E1, E2, E3).
2. **Predominant phase statistics ($P_a$ and $P_c$) and excitatory/inhibitory balance (pp. 392–394)**: The fraction of active $A$-units $P_a$ (Eq. 1) and the conditional co-activation probability $P_c$ between two stimuli (Eq. 2–3) are determined entirely by $(R, L, G, x, y, \theta)$. Setting $x \approx y$ (balanced excitation and inhibition, Fig. 4c) flattens $P_a(R)$ across stimulus sizes $R \in [0.2, 0.5]$ by $10.3\times$ (`0.0607` vs `0.6231`), preventing large stimuli from saturating the association layer (E2).
3. **Why $\gamma$-system zero-sum plasticity and $P_{c11} > P_a > P_{c12}$ are required for statistical separability (pp. 391, 396–402)**:
   - In an uncompensated $\alpha$-system ($\Delta V = +1$ on active units), source-set net value grows linearly with class presentation count $n_{sr}$, creating a severe bias toward frequent classes (`50.0%`–`76.5%` under 3:1 imbalance, E3.1).
   - In the parasitic $\gamma$-system ($\Delta V = +1$ on active units, $\Delta V = -N_{ar}/(N_{Ar}-N_{ar})$ on inactive units), total source-set value remains identically `0.0`, so after $n_{sr}$ stimuli of Class $r$, the expected value on a test stimulus having overlap $P_c$ with Class $r$ is proportional to $n_{sr}(P_c - P_a)/(1 - P_a)$.
   - Consequently, when $P_{c11} > P_a > P_{c12}$, within-class stimuli receive positive drive ($P_{c11} - P_a > 0$) while rival-class stimuli receive *negative* drive ($P_{c12} - P_a < 0$), giving `100.0%` generalization even under 3:1 class imbalance (`Regime B`, E3.1); whereas if two geometric classes happen to overlap more with each other than the retinal average ($P_{c12} > P_a$, as in centered crossing bars `Regime A`), rival reinforcements add positive drive and degrade minority-class recognition (`44%`, E3.1).
4. **Bivalent error correction, spontaneous unsupervised clustering, and the relational limit (pp. 402–405)**:
   - Bivalent reinforcement ($+\Delta V$ on "on" / $-\Delta V$ on "off", or updating only on mistakes) guarantees finite-step convergence on any linearly separable problem within $(R/\gamma)^2$ mistakes (`1,000/1,000` in E1 `step4_converge.py`).
   - Combining proportional weight decay ($-\lambda V$) with zero-sum $\gamma$ competition allows **spontaneous unsupervised clustering** (`100.0%` separation without a teacher, E4.1).
   - Because statistical separability relies on $P_{c11} \ne P_{c12}$ across individual $A$-units, any relational or parity predicate ("Left shape == Right shape", or $k$-bit XOR/parity) where first-order marginal overlaps are identical ($P_{c11} = P_{c12}$) cannot be learned (`42.0%` in E4.2; `0` of `9,261` cells in E1 `step5_xor.py`) unless $A$-units have joint receptive fields spanning the interacting parts (`100.0%` in E4.2; order $k=9$ in E1).

## 5. What surprised me and what I got wrong

1. **The centered horizontal-vs-vertical bar trap (`Regime A` in E3.1)**: When I designed `make_bar_stimulus` (a $3 \times 12$ horizontal bar vs a $12 \times 3$ vertical bar on a $16 \times 16$ grid), I assumed that "horizontal vs vertical bars" obviously satisfied Rosenblatt's differentiated-environment condition $P_{c11} > P_a > P_{c12}$ (p. 400). In fact, because two horizontal bars on different rows have `0` overlap while a $3 \times 12$ horizontal bar and a $12 \times 3$ vertical bar **always** cross in a $3 \times 3 = 9$-pixel square ($9/36 = 25.0\% > R = 36/256 = 14.1\%$), our bars had $P_{c11} = 0.1700 < P_a = 0.1762 < P_{c12} = 0.1831$! That explained why the $\gamma$-system dropped to `44%` on the minority class under 3:1 frequency imbalance, and led directly to testing `Regime B` ($P_{c11} = 0.2370 > P_a = 0.1585 > P_{c12} = 0.0939$), where the $\gamma$-system jumped to `100.0%`.
2. **How much of the 1958 paper is absent from modern "Rosenblatt perceptron" repositories (E1 vs E2–E4)**: All three GitHub repositories (`vukrosic`, `kaisdukes`, `Honghe`) present $\hat{y} = \mathbf{1}[w \cdot x + b \ge 0]$ with mistake update $\Delta w = \eta (y - \hat{y}) x$ as "the 1958 Rosenblatt perceptron." Reading the actual 1958 *Psychological Review* paper showed that the 1958 paper does not even write $\Delta w = \eta (y - \hat{y}) x$: its core mathematical apparatus is the 3-layer $S \to A \to R$ random-projection architecture, binomial/multinomial $P_a$ and $P_c$ (Eq. 1–3), the $\alpha, \beta, \gamma$ metabolic value-competition rules (Table 1), normal-integral generalization $Z(c_1, c_2, c_3, c_4)$ (Eq. 4–11), bivalent reward/punishment (pp. 402–403), and spontaneous unsupervised decay clustering (p. 404).

## 6. Authors say / code does / I observed

- **Authors say (1958 paper, pp. 389–404) vs Reference Repos (`vukrosic`, `kaisdukes`, `Honghe`)**: The 1958 paper models a 3-stage $S \to A \to R$ network with fixed random $S \to A$ threshold units and plastic $A$-unit values $V$ updated by $\alpha, \beta, \gamma$ reinforcement rules; `kaisdukes` and `Honghe` implement a 1-layer linear threshold classifier $w \cdot x + b$ with error-driven updates, while `vukrosic` implements both the 1-layer classifier (`step1`–`step2`, `step4`–`step5`) and a 3-layer `400 -> 512 -> 4` Mark I architecture (`step3_mark1.py`) using the bivalent error-correcting rule.
- **Authors say (p. 398: "$P_r(\Sigma) = P_r(\mu)$" in the $\gamma$-system) vs finite-sample simulation (E3.1)**: In the normal-approximation equations (Eq. 6–8), $\bar{V}_r = 0$ in the $\gamma$-system eliminates the $O(n_{sr}^2)$ sum-variance term $c_3$, making the analytical formulas for $P_r(\Sigma)$ and $P_r(\mu)$ identical; in finite-sample simulation (`E3.1`), $\Sigma$ (`s0 >= s1`) and $\mu$ (`s0/Na0 >= s1/Na1`) agree whenever $s_0$ and $s_1$ have opposite signs and can differ only on borderline stimuli where $s_0$ and $s_1$ have the same sign and $N_{a0} \ne N_{a1}$ (`70.5%` balanced accuracy for both).

## 7. Not run, and why. Unknowns

- **Temporal trace perceptrons ("velocities, sound sequences", p. 403–404)**: Rosenblatt remarks qualitatively on page 403 that temporal patterns can be recognized "provided the stimuli leave some temporarily persistent trace, such as an altered threshold, which causes the activity in the A-system at time $t$ to depend to some degree on the activity at time $t-1$", referring to CAL Report VG-1196-G-1 for details without giving equations in the 1958 paper.

## 8. Sources and Sources read

### Primary paper and repositories
- **Primary paper**: Frank Rosenblatt (1958), "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain", *Psychological Review*, Vol. 65, No. 6, pp. 386–408 (November 1958). Retrieved 2026-10-26 from `https://web.archive.org/web/20220120042254id_/https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf` (1,631,543 bytes, 23 pages; extracted text `work/rosenblatt1958.txt`, 2,006 lines, 62,754 bytes).
- **Reference repo 1**: `https://github.com/vukrosic/rosenblatt-from-scratch.git`, commit `a0167b1eb3527362557105c0f8a933d24da6aecc`, MIT License.
- **Reference repo 2**: `https://github.com/kaisdukes/rosenblatt-perceptron.git`, commit `edd29d9f7773ab3442c0e1d2a37facab851b9323`, MIT License.
- **Reference repo 3**: `https://github.com/Honghe/perceptron.git`, commit `c4caa5a26f933e6363521cb6c2452d91e217496b`.

### Sources read (READ-LOG)
| File | Lines | Status | Why |
| --- | ---: | --- | --- |
| `work/rosenblatt1958.txt` (Rosenblatt 1958, pp. 386–408) | 2006 | `full` | Complete 23-page 1958 *Psychological Review* paper (Equations 1–11, Table 1, Figs. 1–12). |
| `vukrosic-rosenblatt-from-scratch/README.md` | 37 | `full` | Overview of the 7 scripts and Mark I / convergence / XOR experiments. |
| `vukrosic-rosenblatt-from-scratch/perc.py` | 101 | `full` | Core perceptron rule, $20 \times 20$ punch cards, and $12 \times 12$ noisy shifted letters `X, E, T, O`. |
| `vukrosic-rosenblatt-from-scratch/step1_neuron.py`..`step5_xor.py` | 347 | `full` | Hand-set weights, 1958 learning rule, Mark I (`400->512->4`), Novikoff bound, and XOR/parity wall. |
| `vukrosic-rosenblatt-from-scratch/step6_hidden.py` & `step7_backprop.py` | 203 | `full` | Random hidden units vs backpropagation comparison (`demo-output.txt`). |
| `vukrosic-rosenblatt-from-scratch/tests.py` | 67 | `full` | Verification suite across all 7 steps. |
| `kaisdukes-rosenblatt-perceptron/README.md` & `src/*.py`, `tests/*.py` | 307 | `full` | Iris 10-fold cross-validation perceptron implementation and tests. |
| `Honghe-perceptron/perceptron/*.py` | 101 | `full` | 2D linearly separable perceptron implementation and test script. |

## 9. Files

### Local worktree (`C:\personal\oss\clover-framework-foundations\clover-kimi-k3\` on branch `foundations-research`)
- Context file: `docs/rosenblatt-1958-context.md`
- Predictions log: `research/rosenblatt-1958/work/predictions.txt`
- Primary paper text: `research/rosenblatt-1958/work/rosenblatt1958.txt`
- Own implementations:
  - `research/rosenblatt-1958/work/rosenblatt1958_models.py` (Equations 1–3 $P_a, P_c, P_{c\min}$ + Monte Carlo verifier + Fig. 4c flattening; Table 1 $\alpha, \beta, \gamma$ and $\Sigma$ vs $\mu$ learning simulator + $P_{c11} > P_a > P_{c12}$ verification + Ideal vs Differentiated $P_r, P_g$; Spontaneous unsupervised organization with proportional decay + Relational judgment boundary)
- Scripts:
  - `research/rosenblatt-1958/scripts/e0_setup.sh`
  - `research/rosenblatt-1958/scripts/e0b_fetch_rosenblatt1958.sh`
  - `research/rosenblatt-1958/scripts/e0c_find_paper.sh`
  - `research/rosenblatt-1958/scripts/e0d_search_pdf.sh`
  - `research/rosenblatt-1958/scripts/e0e_wayback_pdf.sh`
  - `research/rosenblatt-1958/scripts/e1_repos.sh`
  - `research/rosenblatt-1958/scripts/e2_e3_e4_run.sh`
- Results:
  - `research/rosenblatt-1958/results/e0_setup.out`
  - `research/rosenblatt-1958/results/e0b_fetch_rosenblatt1958.out`
  - `research/rosenblatt-1958/results/e0c_find_paper.out`
  - `research/rosenblatt-1958/results/e0d_search_pdf.out`
  - `research/rosenblatt-1958/results/e0e_wayback_pdf.out`
  - `research/rosenblatt-1958/results/e1_repos.out`
  - `research/rosenblatt-1958/results/e2_e3_e4_run.out`

### Remote (`k3` CPU server)
- `/root/dm-study/foundations/rosenblatt-1958/` (`repo/vukrosic-rosenblatt-from-scratch`, `repo/kaisdukes-rosenblatt-perceptron`, `repo/Honghe-perceptron`, `work/`)

