# Marvin Minsky & Seymour Papert (1969) — *Perceptrons: An Introduction to Computational Geometry* — Context

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh.

## 1. Status

- **E1 (Clone & run reference repositories on k3 CPU: `TaraAlcira/PerceptronsRevisited` spiral challenge & `vukrosic` parity/order)**: DONE
- **E2 (Sections I–III of `AIM-140`: Positive Normal Form mask expansion, exact order distribution on $n=3$, Parity order $|R|$ & weight explosion $2^M$, and order-2 exact counting $\psi_M$)**: DONE
- **E3 (Sections III–VII of `AIM-140`: One-in-a-Box & Cutwise Connectivity $m=3, |A_i|=4$ vs $|A_i|=3$, Diameter-limited impossibility, and $2\log_2|R|$-bit serial connectivity algorithm)**: DONE
- **E4 (Chapter 12 `AIM-167`: Perceptron vs Naive Bayes log-odds vs Nearest Neighbor on Parity)**: DONE

## 2. What this is

Authors say (Marvin Minsky and Seymour Papert, *Perceptrons: An Introduction to Computational Geometry*, MIT Press, 1969; primary MIT AI Lab Memos `AIM-140` *Linearly Unrecognizable Patterns*, 1967, and `AIM-167` *Linear Separation and Learning* — page proofs of Chapter 12 of *Perceptrons*, October 1968; plus Bruce & Easdown 1992 on Theorems 1.5.3, 2.3, and the Chapter 4 AND/OR Theorem):
- "The central theme of this study is the classification of certain geometrical properties according to the type of computation necessary to determine whether a given figure has them... We say that a property $\psi$ is *of order $k$* if $k$ is the smallest integer for which there exists a family $\Phi$ of predicates, each of which depends only on a subset of $k$ points of the figure $X$, and real numbers $\alpha_\phi$ associated with each member $\phi$ of $\Phi$ such that $\psi(X) = [\sum_{\phi \in \Phi} \alpha_\phi \phi(X) > \theta]$." (`AIM-140`, p. 176)
- **Positive Normal Form Theorem (Section I, pp. 181–182; *Perceptrons* Theorem 1.5.3)**: Every Boolean predicate $\psi$ on a finite retina $R$ has a unique integer linear representation in the set of all **masks** $\phi_A(X) = \prod_{x \in A} x$, and $\psi$ has order $k$ iff it is a linear threshold function of masks of support size $|A| \le k$.
- **The Group Invariance Theorem (Section II, pp. 185–187; *Perceptrons* Theorem 2.3)**: If $\psi \in L(\Phi)$ is invariant under a finite permutation group $G$ acting on $R$ and $\Phi$ is closed under $G$, then $\psi$ has a linear threshold representation whose coefficients $\beta_\phi$ are identical across each $G$-equivalence class of $\Phi$. Consequently (Corollary 3, p. 187), the only order-1 predicates invariant under a transitive group (such as translations on a torus) are cardinality/area thresholds $[|X| > m]$ or $[|X| < m]$.
- **The Parity Theorem and Coefficient Explosion (Section III, pp. 187–191)**: $\psi_{\text{PAR}}(X) = [|X| \text{ is odd}]$ has order $|R|$ (requiring at least one mask covering the entire space $R$), and in any mask representation its group-averaged weights grow exponentially ($\sum_{j=0}^{M} \binom{M}{j} |D_j| = 2(1 + 2^M)$, requiring $|R| \cdot 2^{|R|}$ bits to store the weights—more bits than listing all $2^{|R|-1}$ odd subsets explicitly).
- **The One-in-a-Box Theorem, AND/OR Theorem, and Unbounded Order of Connectivity (Sections III–VI, pp. 191–206)**:
  - **One-in-a-Box (p. 191)**: Given $m$ disjoint boxes $A_1, \dots, A_m$ each of size $|A_i| = 4m^2$, the predicate $[\forall i, |X \cap A_i| \ge 1]$ has order $\ge m$.
  - **AND/OR Theorem (Section IV, pp. 193–196; Bruce & Easdown 1992)**: Although $\psi_1 = [|X \cap A| > |X \cap C|]$ and $\psi_2 = [|X \cap B| > |X \cap C|]$ each have order 1, their conjunction $\psi_1 \land \psi_2$ and disjunction $\psi_1 \lor \psi_2$ have unbounded order as $|R| \to \infty$.
  - **Connectivity (Section VI, pp. 200–206 & Section IX, p. 216)**: Topological connectivity $\psi_{\text{CONN}}(X)$ has unbounded order ($\Omega(|R|^{1/3})$ via One-in-a-Box, or $\Omega(|R|^{1/2})$ in the plane) and cannot be computed by any diameter-limited perceptron (proved by the 4-figure bridge counterexample $X_{00}, X_{01}, X_{10}, X_{11}$ on p. 216), whereas a **serial Turing machine** computes connectivity in $2\log_2|R|$ bits of storage using two boundary-tracing pointers `(i, j)` (Section VII, pp. 207–210), and a 2-layer **Gamba machine** computes parity and $\psi_1 \land \psi_2$ with $|R|$ or $2$ order-1 units (Section VIII, pp. 211–213).
- **Chapter 12 (`AIM-167`, *Linear Separation and Learning*, pp. 188–206)**: Compares `PERCEPTRON` against `COMPLETE STORAGE`, `BAYES` (prove that for conditionally independent predicates $p_{ij} = P(\phi_i=1 \mid F_j)$, maximum-likelihood Bayes classification is a linear threshold machine with log-odds weights $w_{ij} = \log \frac{p_{ij}(1-p_{i2})}{p_{i2}(1-p_{ij})}$, Eq. 3–4, pp. 202–204), `BEST PLANES`, and `NEAREST NEIGHBOR` (showing that on $n$-bit parity, nearest neighbor has error $> q(1 - q^n)$, worse than chance on unseen queries, pp. 198–199).

## 3. Experiments

### E1: Inspect and run reference repositories (`TaraAlcira/PerceptronsRevisited` spiral challenge)
- **Question**: How does `TaraAlcira/PerceptronsRevisited` generate Minsky & Papert's single-spiral (connected) vs double-spiral (disconnected) figures, and how does a linear / diameter-limited perceptron compare against Minsky & Papert's Section VII 2-pointer serial connectivity algorithm on those generated spiral images?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 16:20 UTC):
  1. The spiral generator scripts in `TaraAlcira/PerceptronsRevisited/generate_spirals/` will generate connected 1-spiral and disconnected 2-spiral images; a diameter-limited or order-1 linear perceptron on shifted/rotated spirals will stay near $\sim 50\%$ accuracy, whereas Minsky & Papert's serial connectivity algorithm will separate 1-spiral (1 connected component) from 2-spiral (2 connected components) with $100\%$ accuracy.
- **Run**: `scripts/e1_repos.sh` and `scripts/e2_e3_e4_run.sh` (executing `work/minsky_papert_1969_experiments.py`). Outputs: `results/e1_repos.out`, `results/e2_e3_e4_run.out`.
- **Observation**:
  - **Repository (`TaraAlcira/PerceptronsRevisited`)**:
    - Commit `c278eb75d5f328fcda7e5719186992a926dd292f` (4 Python files totaling 548 lines plus 3 notebooks and `README.md` 29 lines).
    - Executing `generate_1double_spiral` and `generate_2double_spiral` across `side_length` $\in \{56, 64, 72, 80\}$ confirmed that `generate_1double_spiral` always produces **1** connected component (`[1, 1, 1, 1]` under both 4-connectivity and 8-connectivity) and `generate_2double_spiral` always produces **2** disconnected components (`[2, 2, 2, 2]`).
    - **Finding on pixel-area confounding in `TaraAlcira/PerceptronsRevisited`**: In `one_spiral_function.py` (lines 80–116), the author removes two outer legs from the 1-spiral (`# remove last leg`, `# remove leg before last leg`) that `two_spirals_function.py` does not remove. Consequently, at any fixed `side_length` (`56, 64, 72, 80`), the 1-spiral has smaller area (`652, 972, 1356, 1804` white pixels) than the 2-spiral (`804, 1156, 1572, 2052` white pixels)! By Minsky & Papert's Corollary 3 (`AIM-140` p. 187 & p. 197: the only translation-invariant order-1 predicates are cardinality cuts $[|X| > A]$), an order-1 linear perceptron achieved **`96.0%`** accuracy on fixed-side (`side=54`, area `612` vs `760`) shifted/rotated spirals purely by thresholding total pixel area $|X|$, and dropped to **`62.0%`** once `side_length` was varied across `{56, 64, 72, 80}` (where the area ranges `[652..1804]` and `[804..2052]` overlap), whereas Minsky & Papert's serial connectivity algorithm achieved **`100.0%`** on both!
- **What it showed**: Running the repo's spiral generators and measuring their pixel areas (`652` vs `804`) uncovered a concrete experimental realization of Minsky & Papert's Corollary 3 (p. 187): unless two geometric classes are matched in area $|X|$ (or varied across scales so their area distributions overlap), an order-1 perceptron appears to "solve" a high-order topological problem by thresholding area $[|X| > A]$.

### E2: Sections I–III of `AIM-140` — Positive Normal Form, Exact Order Distribution on $n=3$, Order-2 Counting $\psi_M$, and Parity Weight Explosion
- **Question**: (a) Does Möbius inversion on the subset lattice produce the unique integer Positive Normal Form $f(X) = \sum_{A \subseteq R} c_A \phi_A(X)$ (Theorem p. 181–182) for all 256 3-input Boolean functions, and what is the exact distribution of minimal predicate order $k \in \{0, 1, 2, 3\}$ across those 256 functions? (b) Does Minsky & Papert's order-2 formula $[(2M-1)\sum x_i - 2\sum_{i<j} x_i x_j \ge M^2]$ (Example iii, p. 181) compute exact cardinality $[|X| = M]$ for all $M$ and all subsets? (c) How fast do the group-averaged mask coefficients $\alpha_j$ of Parity grow with $|R| = M$ (Theorem p. 189–191)?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 16:20 UTC):
  1. Positive Normal Form will reconstruct `256/256` 3-input functions exactly with integer mask weights, and the exact minimal-order counts will be `2` (order 0), `102` (order 1; cumulative `104` linear threshold functions), `150` (order 2), and `2` (order 3: 3-bit parity and XNOR).
  2. The order-2 counting predicate $[(2M-1)|X| - 2\binom{|X|}{2} \ge M^2]$ will match $[|X| = M]$ on `100%` of subsets across all $M \in \{1..7\}$ on $n=8$ points (`1,792/1,792` cases).
  3. Group-averaged mask coefficients for reliable parity ($\ge 2$ on odd $|X|$, $\le 0$ on even $|X|$) satisfy $\alpha_j = (-1)^{j-1} 2^j$ for $j \ge 1$, so $|\alpha_M| = 2^M$ and $\sum_{j=0}^{M} \binom{M}{j} |\alpha_j| = 2(3^M - 1)$.
- **Run**: `scripts/e2_e3_e4_run.sh` (executing `work/minsky_papert_1969_experiments.py`). Output: `results/e2_e3_e4_run.out`.
- **Observation**:
  - **Positive Normal Form & Exact Order on $n=3$ (`256` functions)**:
    - Möbius inversion reconstructed `256/256` functions (`2,048/2,048` truth-table entries) with exact integer coefficients $c_A$.
    - Minimal predicate order distribution:
      - **Order 0**: `2` (`FALSE`, `TRUE`)
      - **Order 1**: `102` (cumulative order $\le 1$: `104 / 256` linear threshold functions)
      - **Order 2**: `150` (cumulative order $\le 2$: `254 / 256`)
      - **Order 3 ($|R|=3$)**: `2` — specifically `[1, 0, 0, 1, 0, 1, 1, 0]` ($\neg(x_1 \oplus x_2 \oplus x_3)$) and `[0, 1, 1, 0, 1, 0, 0, 1]` ($x_1 \oplus x_2 \oplus x_3$), confirming the Parity Theorem (`order = |R|`).
  - **Order-2 Exact Counting Predicate $\psi_M(X) = [|X| = M]$ (p. 181)**:
    - Verified across all $7 \times 256 = 1,792$ `(M, X)` pairs for $n=8, M \in \{1..7\}$ (`True`, 0 errors), because $(2M-1)|X| - |X|(|X|-1) - M^2 = -(|X| - M)^2$, which is $\ge 0$ if and only if $|X| = M$!
  - **Parity Coefficient Explosion (pp. 189–191)**:
    - For $M \in \{2, 4, 6, 8, 10\}$, the unique group-invariant mask weights achieving output `2` on odd $|X|$ and `0` on even $|X|$ have top-mask weight $\alpha_M = -4, -16, -64, -256, -1024$ (`(-2)^M`), total L1 weight $\sum_{j=0}^M \binom{M}{j}|\alpha_j| = 8, 80, 728, 6560, 59048$ (`3^M - 1`), and require $M \cdot 2^M = 10,240$ bits at $M=10$ vs $2^{M-1} = 512$ odd subsets.
- **What it showed**: All three predictions held to the digit.

### E3: Sections III–VII of `AIM-140` — One-in-a-Box, Cutwise Connectivity, Diameter-Limited Impossibility, and the $2\log_2|R|$-Bit Serial Connectivity Algorithm
- **Question**: (a) On the $m=3$ row One-in-a-Box / Cutwise Connectivity problem (pp. 201–202), do the disconnected figure $X_1$ with row counts `(3, 3, 0)` and the connected figure $X_2$ with row counts `(4, 1, 1)` have identical order-2 group-invariant mask counts $(N^1, N^{11}, N^{12})$ when $|A_i|=4$, whereas when $|A_i|=3$, Minsky & Papert's order-2 formula $[N^1(X) + N^{12}(X) - 2N^{11}(X) > 4]$ succeeds on all $2^9 = 512$ subsets? (b) On the 4-figure diameter-limited family $X_{00}, X_{01}, X_{10}, X_{11}$ (p. 216), do all masks of diameter $\le D$ satisfy $\phi(X_{00}) + \phi(X_{11}) = \phi(X_{01}) + \phi(X_{10})$? (c) Does Minsky & Papert's 2-pointer serial boundary-tracing algorithm (`SEARCH` / `SCAN` / `TRACE`, p. 208) accurately determine connectivity and component counts across convex, concave, hollow-ring, and multi-component grid figures?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 16:20 UTC):
  1. At $m=3, |A_i|=4$, disconnected `(3, 3, 0)` and connected `(4, 1, 1)` both give $(N^1, N^{11}, N^{12}) = (6, 6, 9)$; at $m=3, |A_i|=3$, $[N^1 + N^{12} - 2N^{11} > 4]$ matches one-in-a-box on `512/512` subsets (`100%`).
  2. `0` of `308` diameter-limited masks (`size <= 4, diam <= 6`) violate $\phi(X_{00}) + \phi(X_{11}) = \phi(X_{01}) + \phi(X_{10})$, and the 2-pointer serial algorithm (`i, j`) classifies all 5 canonical topological test grids (`Solid`, `Concave U-shape`, `Hollow ring with interior hole`, `2 squares`, `3 squares`) accurately.
- **Run**: `scripts/e2_e3_e4_run.sh` (executing `work/minsky_papert_1969_experiments.py`). Output: `results/e2_e3_e4_run.out`.
- **Observation**:
  - **One-in-a-Box / Cutwise Connectivity (`m=3, |A_i|=4` vs `|A_i|=3`, pp. 201–202)**:
    - Disconnected $X_1$ (`rows=(3,3,0)`) and connected $X_2$ (`rows=(4,1,1)`) both produced `N^1 = 6`, `N^11 (same-row pairs) = 6`, and `N^12 (diff-row pairs) = 9`, proving no order-2 predicate can separate them when $|A_i| \ge 4$.
    - When $|A_i| = 3$ (where row count $4$ cannot occur), Minsky & Papert's order-2 threshold $[N^1(X) + N^{12}(X) - 2N^{11}(X) > 4]$ matched $[\forall i \in \{1,2,3\}, r_i \ge 1]$ on **`512 / 512` subsets (`100%`)**.
  - **Diameter-Limited Connectivity Impossibility (`AIM-140` p. 216)**:
    - Across all `308` masks of support size $\le 4$ and diameter $\le 6$, `0 / 308` masks violated $\phi(X_{00}) + \phi(X_{11}) = \phi(X_{01}) + \phi(X_{10})$.
  - **2-Pointer Serial Connectivity Algorithm (`SEARCH` / `SCAN` / `TRACE`, p. 208)**:
    - Accurately identified connectivity and exact component count $K$ on `Solid 5x5 square` (`True, K=1`), `Concave U-shape` (`True, K=1`), `Hollow ring with interior hole` (`True, K=1` — when `SCAN` hits the right wall of the interior hole, `TRACE` circumnavigates the hole boundary and encounters the top wall of the hole at raster index $j < i$, so the hole does not increment $K$!), `Two disjoint squares` (`False, K=2`), and `Three disjoint squares` (`False, K=3`).
    - *Implementation note*: On our first run of `serial_connectivity_p208`, `Concave U-shape` returned `K=2` because at the top-left corner `(2, 6)` of the right arm of the U, the pixel `(2, 6)` has two exposed boundary edges (West and North), so turning right at the starting pixel kept raster index $j = 26 == i$ on step 1 and triggered an early `if j == i: break`. Terminating `trace_boundary_loop` on the directed boundary state `(r, c, d) == start_state` instead of bare pixel index fixed `Concave U-shape` (`K=1`).
- **What it showed**: All predictions held (`(6,6,9)` collision at $|A_i|=4$, `512/512` at $|A_i|=3$, `0/308` diameter-limited violations, and `5/5` topological grids solved by the 2-pointer serial algorithm).

### E4: Chapter 12 (`AIM-167`) — `PERCEPTRON` vs `BAYES` Log-Odds vs `NEAREST NEIGHBOR` on Parity
- **Question**: In Chapter 12 of *Perceptrons* (`AIM-167`, *Linear Separation and Learning*): (a) How do `PERCEPTRON` and `BAYES` (Eq. 3–4, p. 203, with log-odds weights $w_i = \log \frac{p_{i1}(1-p_{i0})}{p_{i0}(1-p_{i1})}$) compare on Figure 12.3(a) (linearly separable clouds with correlated features) vs Figure 12.3(b) (non-separable clouds with conditionally independent noisy features)? (b) On $n$-bit parity ($n=8$, $256$ vectors), when a fraction $1 - q$ of the space has been stored in memory, does `NEAREST NEIGHBOR` fail with error rate $\ge 1 - q^n$ on unseen queries and $\ge q(1 - q^n)$ overall, as derived by Minsky & Papert on pages 198–199?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 16:20 UTC):
  1. On Fig. 12.3(a) (linearly separable correlated skew clouds), `PERCEPTRON` will reach `0.0%` error while `BAYES` makes $> 8\%$ error; on Fig. 12.3(b) (conditionally independent noisy features), `BAYES` will achieve lower error than `PERCEPTRON`.
  2. On 8-bit parity, `NEAREST NEIGHBOR` error on unseen queries will match $1 - q^8$ (`~100%` for $q \le 0.5$) and overall error will match $q(1 - q^8)$.
- **Run**: `scripts/e2_e3_e4_run.sh` (executing `work/minsky_papert_1969_experiments.py`). Output: `results/e2_e3_e4_run.out`.
- **Observation**:
  - **E4.1 — `PERCEPTRON` vs `BAYES` Log-Odds (`AIM-167` Fig. 12.3a & 12.3b, pp. 195, 202–204)**:
    - `Fig. 12.3(a) Linearly separable correlated skew clouds`: **`PERCEPTRON` error = `0.0%`** vs **`BAYES` error = `12.8%`** (because within-class feature correlation gives the marginal probabilities $P(x_j=1 \mid F_1) > P(x_j=1 \mid F_0)$ the opposite sign from the separating hyperplane $6x_0 - \sum_{j=1}^8 x_j \ge 1$).
    - `Fig. 12.3(b) Conditionally independent noisy features (non-separable)`: **`BAYES` error = `10.3%`** vs **`PERCEPTRON` error = `19.3%`** (matching Minsky & Papert's statement on p. 204: when predicates are conditionally independent and classes overlap, log-odds `BAYES` is optimal while `PERCEPTRON` fluctuates on boundary noise).
  - **E4.2 — `NEAREST NEIGHBOR` Pathology on 8-Bit Parity (`AIM-167` pp. 198–199, 50 trials per $q$)**:
    - `q = 0.10` (`90%` of $2^8$ seen): Unseen-query error = **`100.0%`** (theoretical $1 - q^8 = 100.0\%$) | All-space error = **`10.2%`** (theoretical $q(1-q^8) = 10.0\%$).
    - `q = 0.25` (`75%` of $2^8$ seen): Unseen-query error = **`100.0%`** (theoretical $100.0\%$) | All-space error = **`25.0%`** (theoretical $25.0\%$).
    - `q = 0.50` (`50%` of $2^8$ seen): Unseen-query error = **`99.6%`** (theoretical $1 - 0.5^8 = 99.6\%$) | All-space error = **`49.8%`** (theoretical $49.8\%$).
    - `q = 0.70` (`30%` of $2^8$ seen): Unseen-query error = **`94.5%`** (theoretical $1 - 0.7^8 = 94.2\%$) | All-space error = **`66.1%`** (theoretical $66.0\%$).
- **What it showed**: Both predictions held to within $0.3\%$ across all settings, confirming why Nearest Neighbor is anti-predictive (`99.6%`–`100.0%` wrong on unseen queries!) on parity-structured spaces where every distance-1 neighbor has the opposite class.

## 4. How it works

1. **Reduction of arbitrary predicates to mask polynomials via the Positive Normal Form Theorem (`AIM-140` Section I, pp. 181–183)**: Every Boolean function on $R$ has a unique integer expansion in conjunction masks $\phi_A(X) = \prod_{x \in A} x$ obtained by Möbius inversion on the subset lattice (`256/256` verified in E2.1), so the predicate order $k$ of $\psi$ (the maximum support size $|\text{supp}(\phi)|$ needed in a linear threshold representation $\psi(X) = [\sum \alpha_\phi \phi(X) > \theta]$) equals the minimum mask degree $k$ needed to linearly separate $\psi^{-1}(1)$ from $\psi^{-1}(0)$ (E2.1).
2. **Group Invariance collapses mask weights into low-degree occupancy polynomials (`AIM-140` Section II, pp. 185–187)**: When $\psi$ is invariant under a permutation group $G$ acting on $R$, averaging any valid weight vector over $G$ preserves linear separation while equalizing weights within each $G$-orbit of masks. Under the full symmetric group $S_{|R|}$, the orbit sums $\sum_{|A|=j} \phi_A(X) = \binom{|X|}{j}$ are univariate polynomials of degree $j$ in $|X|$; under a product group $S_{A_1} \times \dots \times S_{A_m}$, they are multivariate polynomials of total degree $\le k$ in the box counts $x_i = |X \cap A_i|$ (E2, E3).
3. **Why Parity has order $|R|$ and exponential weights while Exact Counting $[|X|=M]$ has order 2 (`AIM-140` Section I & III, pp. 181, 187–191)**:
   - $[|X| = M]$ changes sign only **twice** as $|X|$ goes from $0$ to $|R|$ (at $M - 0.5$ and $M + 0.5$), so the degree-2 polynomial $(2M-1)|X| - 2\binom{|X|}{2} - M^2 = -(|X|-M)^2$ realizes $[|X|=M]$ at **order 2** for every $|R|$ and $M$ (`1,792/1,792` verified in E2.2).
   - $\psi_{\text{PAR}}(X) = [|X| \text{ is odd}]$ changes sign **$|R|$ times** as $|X|$ goes from $0$ to $|R|$, forcing the polynomial degree $k$ to be $\ge |R|$ (so order $= |R|$, `2/256` functions on $n=3$, E2.1) and forcing the group-averaged mask coefficients to alternate and grow as $\alpha_j = (-2)^j$ with L1 norm $\sum_{j=0}^{M} \binom{M}{j}|\alpha_j| = 3^M - 1$ (E2.3).
4. **Why Topological Connectivity has unbounded order and defeats diameter-limited perceptrons, while Serial Boundary Tracing needs only $2\log_2|R|$ bits (`AIM-140` Sections III, VI, VII, IX)**:
   - By restricting a grid to $m$ odd-row bars connected through $m$ even-row boxes $A_1, \dots, A_m$, connectivity reduces to the **One-in-a-Box** predicate $[\forall i, |X \cap A_i| \ge 1]$. Because substituting $x_i = (t - (2i-1))^2$ turns any degree-$k$ polynomial $Q(x_1, \dots, x_m)$ into a univariate polynomial of degree $2k$ in $t$ that must alternate sign across $t = 0, 1, \dots, 2m$, $2k \ge 2m \implies k \ge m$ (and at $m=3, |A_i|=4$, disconnected `(3,3,0)` and connected `(4,1,1)` share identical order-2 counts `(6, 6, 9)`, E3.1).
   - Similarly, on the 4-figure family $X_{00}, X_{01}, X_{10}, X_{11}$ where two switches lie $> D$ apart, no diameter-$\le D$ mask can see both switches, so $\phi(X_{00}) + \phi(X_{11}) = \phi(X_{01}) + \phi(X_{10})$ for all `308/308` masks (E3.2).
   - In contrast, a serial algorithm (`SEARCH`/`SCAN`/`TRACE`, p. 208) traces boundary loops with just two index registers `(i, j)` (`2 log2 |R|` bits), counting a new component iff a boundary loop initiated at $x_i \in X, x_{i-1} \notin X$ returns to $i$ without visiting any $j < i$ (`5/5` topological test types and `100%` of spirals, E1, E3.3).
5. **Perceptron feedback vs Naive Bayes independence vs Nearest Neighbor (`AIM-167` Chapter 12)**:
   - Under conditional independence of predicates within each class, maximum-likelihood Bayes is a linear threshold classifier with log-odds weights $w_i = \log \frac{p_{i1}(1-p_{i0})}{p_{i0}(1-p_{i1})}$ (`10.3%` error vs `19.3%` for Perceptron on noisy independent features, E4.1).
   - When features are correlated within linearly separable classes (Fig. 12.3a), `BAYES` aligns with the class means rather than the separating hyperplane and makes `12.8%` error, whereas `PERCEPTRON`'s error-driven feedback achieves `0.0%` error (E4.1).
   - On $n$-bit parity, `NEAREST NEIGHBOR` suffers catastrophic anti-generalization (`99.6%`–`100.0%` error on unseen queries for $q \le 0.5$, matching $1 - q^n$, E4.2) because every Hamming-distance-1 neighbor has the opposite parity.

## 5. What surprised me and what I got wrong

1. **Why an Order-1 Perceptron got `96.0%` on fixed-side spirals from `TaraAlcira/PerceptronsRevisited` (E1)**: I initially predicted in `work/predictions.txt` that an order-1 perceptron on shifted/rotated spirals from `TaraAlcira/PerceptronsRevisited` would score $\sim 50\%$. When we ran `E1` at fixed `side=54`, the order-1 perceptron scored **`96.0%`**! Inspecting `one_spiral_function.py` lines 80–116 and printing the pixel counts revealed why: the repo's `generate_1double_spiral` erases two outer spiral legs that `generate_2double_spiral` leaves intact, so at `side=56` the 1-spiral has `652` pixels while the 2-spiral has `804` pixels! By Minsky & Papert's Group Invariance Theorem Corollary 3 (p. 187), an order-1 translation-invariant perceptron can threshold total pixel area $[|X| > A]$! Once `side_length` was varied across `{56, 64, 72, 80}` so that 1-spiral and 2-spiral areas overlapped, the order-1 perceptron dropped to `62.0%` while the serial connectivity algorithm stayed at `100.0%`.
2. **Corner-pixel multi-edge rotation in `TRACE` on the Concave U-shape (E3.3)**: In Minsky & Papert's 2-pointer serial connectivity algorithm (`AIM-140` p. 208), a boundary pixel $x_i$ at a convex corner (like the top-left pixel `(2, 6)` of the right arm of a U-shape) has both its West and North edges exposed to the complement $\neg X$. On our first run, checking `if j == i: break` on the bare pixel index $j$ stopped after rotating $90^\circ$ on the starting corner pixel `(2, 6)` itself (`step 1`), falsely counting the right arm of the U as a second component (`K=2`). Checking termination on the directed boundary half-edge `(r, c, d) == start_state` fixed the walk (`K=1`).

## 6. Authors say / code does / I observed

- **Authors say (`Perceptrons` Section 1.5, AND/OR Theorem) vs Proof in Chapter 4 (`and_or_theorem.txt`, Bruce & Easdown 1992, p. 1)**: Minsky & Papert claim in Section 1.5 that there exist two order-1 predicates $\psi_1, \psi_2$ such that *both* $\psi_1 \land \psi_2$ and $\psi_1 \lor \psi_2$ have order $> n$; as Bruce & Easdown (1992, p. 1) observe, Minsky & Papert's proof in Chapter 4 (`AIM-140` Section IV, pp. 193–196, which we read in full) proves the existence of $\psi_1, \psi_2$ for $\psi_1 \land \psi_2$ and a dual pair $\psi_1', \psi_2'$ for $\psi_1' \lor \psi_2'$, and Bruce & Easdown (1992, Theorem 4, pp. 5–8) supply the Simultaneous Compactness proof for $\psi_1 = \psi_1', \psi_2 = \psi_2'$.
- **Authors say (`AIM-140` p. 202, OCR text `N^11 = o, N^12 = 9`) vs Exact Combinatorics (E3.1)**: In the `AIM-140` scan on page 202, the diagram of $X_1$ and $X_2$ is degraded and the OCR reads `N^11 = o, N^12 = 9`. Reconstructing from $\binom{N^1}{2} = N^{11} + N^{12}$ shows $N^1 = 6, N^{11} = 6, N^{12} = 9$, realized uniquely for $m=3, |A_i|=4$ by row partitions $(3, 3, 0)$ (disconnected) and $(4, 1, 1)$ (connected).

## 7. Not run, and why. Unknowns

- **Full 256x256 8-layer CNN training in `TaraAlcira/PerceptronsRevisited/CNN/CNN_best2.py`**: `CNN_best2.py` trains an 8-conv-layer Keras model for up to 2,000 epochs on 20,000 $256 \times 256$ PNGs (`/home/tsplint/thesis/data/data256+/training_data`, not included in the repo); we ran the repo's spiral generators (`one_spiral_function.py`, `two_spirals_function.py`) and tested the order-1 perceptron and Minsky & Papert's serial connectivity algorithm directly on the generated spirals in E1.

## 8. Sources and Sources read

### Primary papers/monographs and repositories
- **Primary MIT AI Lab Memo 1 (`Perceptrons` Chapters 1–10)**: Marvin Minsky and Seymour Papert (1967), *Linearly Unrecognizable Patterns*, MIT Project MAC AI Memo No. 140 / *Proceedings of Symposia in Applied Mathematics*, Vol. 19 (*Mathematical Aspects of Computer Science*), pp. 176–217. Retrieved 2026-10-26 from `http://bitsavers.trailing-edge.com/pdf/mit/ai/aim/AIM-140.pdf` (2,136,438 bytes) and Archive.org full text `https://archive.org/download/bitsavers_mitaiaimAI_2136438/AIM-140_djvu.txt` (`work/AIM-140_djvu.txt`, 2,568 lines, 83,120 bytes).
- **Primary MIT AI Lab Memo 2 (`Perceptrons` Chapter 12 page proofs)**: Marvin Minsky and Seymour Papert (October 1968), *Linear Separation and Learning*, MIT Project MAC AI Memo No. 167 ("reprint of page proofs of Chapter 12 of *Perceptrons*, M. Minsky and S. Papert, MIT Press"). Retrieved 2026-10-26 from `http://bitsavers.trailing-edge.com/pdf/mit/ai/aim/AIM-167.pdf` (1,208,423 bytes) and Archive.org full text `https://archive.org/download/bitsavers_mitaiaimAI_1208423/AIM-167_djvu.txt` (`work/AIM-167_djvu.txt`, 2,682 lines, 62,233 bytes).
- **Supporting mathematical paper (Theorems 1.5.3, 2.3, and Ch. 4 AND/OR Theorem)**: I. D. Bruce and D. Easdown (1992), "The AND/OR Theorem for Perceptrons", *J. Austral. Math. Soc. (Series A)*, Vol. 53, pp. 1–8. Retrieved 2026-10-26 from Cambridge Core (`work/and_or_theorem.txt`, 338 lines, 12,435 bytes).
- **Reference repo**: `https://github.com/TaraAlcira/PerceptronsRevisited.git`, commit `c278eb75d5f328fcda7e5719186992a926dd292f`.

### Sources read (READ-LOG)
| File | Lines | Status | Why |
| --- | ---: | --- | --- |
| `work/AIM-140_djvu.txt` (Minsky & Papert 1967/1969, pp. 176–217) | 2568 | `full` | Complete text of Sections I–IX (Positive Normal Form, Group Invariance, Parity, One-in-a-Box, AND/OR, Connectivity, Serial Turing algorithm, Gamba machines, Diameter-limited perceptrons). |
| `work/AIM-167_djvu.txt` (Minsky & Papert 1968/1969, Chapter 12) | 2682 | `full` | Read Sections 12.0–12.4.5 (lines 1–1200) covering Perceptron vs Complete Storage, Naive Bayes log-odds derivation, Best Planes, and Nearest Neighbor on Parity. |
| `work/and_or_theorem.txt` (Bruce & Easdown 1992, pp. 1–8) | 338 | `full` | Clean mathematical statements and proofs of Theorems 1.5.3, 2.3, Compactness, and the simultaneous AND/OR Theorem. |
| `TaraAlcira-PerceptronsRevisited/README.md` | 29 | `full` | Overview of the 1-spiral vs 2-spiral *Perceptrons* cover problem. |
| `TaraAlcira-PerceptronsRevisited/generate_spirals/*.py` | 431 | `full` | `one_spiral_function.py`, `two_spirals_function.py`, and `randomize_spirals.py`. |
| `TaraAlcira-PerceptronsRevisited/CNN/CNN_best2.py` | 117 | `full` | 8-layer CNN architecture for spiral classification. |

## 9. Files

### Local worktree (`C:\personal\oss\clover-framework-foundations\clover-kimi-k3\` on branch `foundations-research`)
- Context file: `docs/minsky-papert-1969-context.md`
- Predictions log: `research/minsky-papert-1969/work/predictions.txt`
- Primary texts:
  - `research/minsky-papert-1969/work/AIM-140_djvu.txt`
  - `research/minsky-papert-1969/work/AIM-167_djvu.txt`
  - `research/minsky-papert-1969/work/and_or_theorem.txt`
- Own implementations:
  - `research/minsky-papert-1969/work/minsky_papert_1969_experiments.py` (E1 spiral analysis & area-cut test; E2 Möbius Positive Normal Form, exact order solver on all 256 3-input functions, order-2 exact counting $\psi_M$, parity $2^M$ weight explosion; E3 One-in-a-Box `(3,3,0)` vs `(4,1,1)` collision & $|A_i|=3$ order-2 formula, diameter-limited bridge impossibility, 2-pointer serial boundary-tracing algorithm; E4 Chapter 12 `PERCEPTRON` vs `BAYES` log-odds & `NEAREST NEIGHBOR` parity pathology)
- Scripts:
  - `research/minsky-papert-1969/scripts/e0_setup.sh`
  - `research/minsky-papert-1969/scripts/e0b_fetch_mp1969.sh`
  - `research/minsky-papert-1969/scripts/e0c_find_mp1969_text.sh`
  - `research/minsky-papert-1969/scripts/e0d_aim_index.sh`
  - `research/minsky-papert-1969/scripts/e0e_mit_memos.sh`
  - `research/minsky-papert-1969/scripts/e0f_aim_search.sh`
  - `research/minsky-papert-1969/scripts/e0g_aim_early.sh`
  - `research/minsky-papert-1969/scripts/e0h_get_mp_memos.sh`
  - `research/minsky-papert-1969/scripts/e0i_extract_aim.sh`
  - `research/minsky-papert-1969/scripts/e1_repos.sh`
  - `research/minsky-papert-1969/scripts/e2_e3_e4_run.sh`
- Results:
  - `research/minsky-papert-1969/results/e0_setup.out`
  - `research/minsky-papert-1969/results/e0b_fetch_mp1969.out`
  - `research/minsky-papert-1969/results/e0c_find_mp1969_text.out`
  - `research/minsky-papert-1969/results/e0d_aim_index.out`
  - `research/minsky-papert-1969/results/e0e_mit_memos.out`
  - `research/minsky-papert-1969/results/e0f_aim_search.out`
  - `research/minsky-papert-1969/results/e0g_aim_early.out`
  - `research/minsky-papert-1969/results/e0h_get_mp_memos.out`
  - `research/minsky-papert-1969/results/e0i_extract_aim.out`
  - `research/minsky-papert-1969/results/e1_repos.out`
  - `research/minsky-papert-1969/results/e2_e3_e4_run.out`

### Remote (`k3` CPU server)
- `/root/dm-study/foundations/minsky-papert-1969/` (`repo/TaraAlcira-PerceptronsRevisited`, `work/`)

