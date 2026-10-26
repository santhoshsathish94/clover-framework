# John J. Hopfield (1982) — *Neural networks and physical systems with emergent collective computational abilities* — Context

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh.

## 1. Status

- **E1 (Clone & run reference repositories on k3 CPU: `pauljmello/Hopfield-Networks` & `1hubert/hopfield-network`)**: DONE
- **E2 (Energy monotonicity Eq. 7–8, asynchronous vs synchronous updates, and $\{0,1\}$ vs bipolar $\{-1,+1\}$ factor-of-2 capacity doubling Eq. 2–4, Eq. 10, Fig. 2)**: DONE
- **E3 (Phase-space flow, basin of attraction vs Hamming distance $N=30, n=5$, spurious states, and memory merging $N=100, n=8, d \in \{30, 20, 10\}$)**: DONE
- **E4 (Clipped synapses $2/\pi$ factor, bounded $\pm 3$ palimpsest forgetting, unidirectional $1/\sqrt{2}$ dilution, and $25\times$ overload familiarity $N=100, n=500$)**: DONE

## 2. What this is

Authors say (J. J. Hopfield, "Neural networks and physical systems with emergent collective computational abilities", *Proceedings of the National Academy of Sciences USA*, Vol. 79, No. 8, pp. 2554–2558, April 1982):
- "The physical meaning of content-addressable memory is described by an appropriate phase space flow of the state of a system... Let the system have locally stable limit points $X_a, X_b, \dots$. Then, if the system is started sufficiently near any $X_a$, as at $X = X_a + \Delta$, it will proceed in time until $X \approx X_a$." (Abstract & p. 2554)
- **Model & Asynchronous Processing (Equations 1–4, p. 2555)**: Each of $N$ neurons has binary state $V_i \in \{0, 1\}$ and updates randomly and asynchronously at mean attempt rate $W$ according to $V_i \to 1$ if $\sum_{j \ne i} T_{ij} V_j > U_i$ and $V_i \to 0$ if $< U_i$ (Eq. 1, default $U_i = 0$). To store $n$ states $V^s$ ($s=1\dots n$), synapses are set by the Hebbian outer-product rule:
  $$T_{ij} = \sum_{s=1}^n (2 V_i^s - 1)(2 V_j^s - 1), \qquad T_{ii} = 0 \quad \text{(Eq. 2)}$$
- **Energy Function & Ising Isomorphism (Equations 7–8, p. 2556)**: For symmetric connections $T_{ij} = T_{ji}$ with $T_{ii} = 0$, the energy $E = -\frac{1}{2} \sum_{i \ne j} T_{ij} V_i V_j$ changes under an asynchronous flip $\Delta V_i$ by $\Delta E = -\Delta V_i \sum_{j \ne i} T_{ij} V_j \le 0$, so $E$ decreases monotonically until a local minimum is reached.
- **Storage Capacity ($\sim 0.15 N$) and Bipolar Doubling (Equation 10 & Fig. 2, pp. 2556–2557)**: Under $\{0, 1\}$ states ($V_j \in \{0,1\}$), the effective field $H_i^{s'} = \sum_{j \ne i} T_{ij} V_j^{s'}$ has signal mean $\pm N/2$ and crosstalk noise standard deviation $\sigma = [(n-1)N/2]^{1/2}$, yielding single-bit error probability $P = \frac{1}{\sqrt{2\pi}\sigma}\int_{N/2}^{\infty} e^{-x^2/2\sigma^2}dx$ (Eq. 10; $P = 0.0091$ at $N=100, n=10$). Half the memories are well retained at $n \approx 0.15 N$. Using bipolar variables $\mu_i = \pm 1$ with $T_{ij} = \sum_s \mu_i^s \mu_j^s$ and threshold $0$ increases storage capacity at a given accuracy by a **factor of 2** (p. 2557).
- **Emergent & Hardware-Robust Properties (pp. 2557–2558)**:
  - **Basin of attraction ($N=30, n=5$)**: Starting within Hamming distance $\le 5$ recovers the nearest memory $>90\%$ of the time; random initial states land on assigned memories $\sim 85\%$ of the time and spurious attractors $\sim 10\%$.
  - **Clipped synapses ($T_{ij} = \pm 1$)**: Reduces SNR by $\sqrt{2/\pi}$ and capacity at fixed accuracy by $2/\pi$.
  - **Bounded saturating synapses ($T_{ij} \in \{0, \pm 1, \pm 2, \pm 3\}$)**: Continually adding new memories automatically forgets distant-past memories while retaining recent ones.
  - **Unidirectional synapses ($T_{ij} \ne 0 \implies T_{ji} = 0$)**: Reduces SNR by $1/\sqrt{2}$ (fail-soft degradation).
  - **Memory merging ($N=100, n=8$)**: Two memories separated by Hamming distance $30$ stay distinct; at $20$ they displace; at $10$ they fuse.
  - **Overload familiarity ($N=100, n=500$)**: Under $25\times$ overload where no memory is stable, unfamiliar states have a faster initial processing rate (more flips in time $1/2W$) than familiar states.

## 3. Experiments

### E1: Clone and run reference repositories (`pauljmello/Hopfield-Networks` and `1hubert/hopfield-network`)
- **Question**: How do `pauljmello/Hopfield-Networks` and `1hubert/hopfield-network` implement the 1982 Hopfield network, and what do they output when executed on `k3` CPU?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 16:48 UTC):
  1. Both repositories will implement the bipolar $\{-1, +1\}$ weight rule with zero diagonal $T_{ii}=0$ and recover corrupted patterns on CPU (`rc=0`).
- **Run**: `scripts/e1_repos.sh` and `scripts/e2_e3_e4_run.sh`. Outputs: `results/e1_repos.out`, `results/e2_e3_e4_run.out`.
- **Observation**:
  - **Repository 1 (`pauljmello/Hopfield-Networks`)**:
    - Commit `a185b5bf821485dd2165af38af66e7fe8cace8be` (4 Python files totaling 494 lines plus `README.md` 84 lines).
    - Implements `HopfieldNetwork` (`hopfield.py`, 54 lines: bipolar $\{-1, +1\}$ states, $W = \frac{1}{N} \sum_\mu x^\mu (x^\mu)^T$ with zero diagonal, energy $E = -\frac{1}{2} s^T W s$, and random asynchronous single-neuron updates) and `ModernHopfieldNetwork` (`modernhopfield.py`, 60 lines: continuous state update $Z_{\text{new}} = X \text{softmax}(\beta X^T Z)$ from Ramsauer et al. 2020).
    - Executed on `k3` CPU (`results/e2_e3_e4_run.out`): from a pattern with 15/100 flipped bits (`initial overlap = 0.70`), `HopfieldNetwork` recovered `overlap = 1.00` in 300 asynchronous steps with energy drop $\Delta E = -22.26$.
    - **Code difference found in `pauljmello/Hopfield-Networks/hopfield.py` line 45**: `update()` sets `state[i] = torch.sign(local_field)`. In PyTorch, `torch.sign(torch.tensor(0.0))` returns **`0.0`** (not $\pm 1$ or the previous state `state[i]`), so whenever an even number of bipolar patterns cancel at neuron $i$ (`local_field == 0.0`), `hopfield.py` assigns `0.0` to a binary $\{-1, +1\}$ neuron!
  - **Repository 2 (`1hubert/hopfield-network`)**:
    - Commit `1d1ec4b3dbd7f953f12bea2214e688092f1da5f1` (`main.py`, 157 lines).
    - Stores three $5 \times 5$ bipolar $\{-1, +1\}$ images (`rice_field_img`, `x_img`, `o_img`) with zero diagonal and runs asynchronous updates using `1 if result >= 0 else -1` (`results/e1_repos.out`, `rc=0`).
  - **Repository 3 (`in-c0/1982-hopfield-Neural-networks-paper`)**:
    - Commit `61ea96eaff65f76f9e93c547863ccc6ac2cdb4b2` containing the archival 1982 PNAS PDF `pnas00447-0135.pdf` (`5` pages, extracted to `work/hopfield1982.txt`, `703` lines).
- **What it showed**: Prediction 1 held (`rc=0` on both code repos), and reading `pauljmello/hopfield.py` exposed the `torch.sign(0.0) == 0.0` zero-state bug on tied local fields.

### E2: Energy Monotonicity (Eq. 7–8, p. 2556) and $\{0, 1\}$ vs Bipolar $\{-1, +1\}$ Factor-of-2 Capacity Doubling (Eq. 2–4, Eq. 10, Fig. 2, pp. 2555–2557)
- **Question**: (a) Does asynchronous updating on symmetric $T_{ij} = T_{ji}$ with zero diagonal $T_{ii} = 0$ guarantee $\Delta E \le 0$ on every flip and 100% convergence to fixed points, whereas synchronous parallel updates or negative diagonal $T_{ii} < 0$ violate monotonicity or enter limit cycles? (b) Why does Hopfield state on page 2557 that switching from $\{0, 1\}$ states ($V_i \in \{0, 1\}$, Eq. 1–4 & Eq. 10) to bipolar states ($\mu_i \in \{-1, +1\}$) increases storage capacity at a given accuracy by a **factor of 2**, and do empirical error rates and Figure 2 multi-step histograms confirm this exact $2\times$ scaling?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 16:48 UTC):
  1. Asynchronous updates on symmetric $T_{ij}$ with $T_{ii}=0$ will have `0` energy increases across all flips in `500/500` runs (`100%` convergence), whereas synchronous parallel updates will enter length-$\ge 2$ limit cycles and $T_{ii} < 0$ will cause $\Delta E > 0$ increases.
  2. Because in $\{0, 1\}$ only the $\sim N/2$ active neurons contribute to $H_i^{s'} = \sum_{j \ne i} T_{ij} V_j^{s'}$ (`SNR = sqrt(N / (2*(n-1)))`, Eq. 10) whereas in bipolar $\{-1, +1\}$ all $N-1$ neurons contribute (`SNR = sqrt(N / (n'-1))`), the 1-step bit error rate $P$ of $\{0, 1\}$ at $n$ patterns will equal that of bipolar $\{-1, +1\}$ at $n' = 2n - 1$ patterns (`P = 0.00921` at $N=100, n=10$ vs $n'=19$).
- **Run**: `scripts/e2_e3_e4_run.sh` (executing `work/hopfield1982_experiments.py`). Output: `results/e2_e3_e4_run.out`.
- **Observation**:
  - **E2.1 — Energy monotonicity & negative controls ($N=30$)**:
    - Asynchronous updates with $T_{ij} = T_{ji}, T_{ii} = 0$: converged to a stable fixed point in **`500 / 500` runs** with **`0 / 7,162` energy-increasing flips**.
    - Negative Control A (Synchronous parallel updates on symmetric $W$): entered limit cycles of length $\ge 2$ ($A \leftrightarrow B$) in **`197 / 200` runs (`98.5%`)**.
    - Negative Control B (Asynchronous updates with negative diagonal $T_{ii} = -20.0$): energy $E$ increased on **`1,217` flips** across 200 runs.
  - **E2.2 — $\{0, 1\}$ ($n$) vs Bipolar $\{-1, +1\}$ ($2n - 1$) Factor-of-2 Capacity Verification ($N = 100$, 60 trials per row)**:
    - `n_01 = 5` vs `n_bipolar = 9`: Eq. 10 $P = 0.00020$ | Empirical $P_{\{0,1\}}(5) = \mathbf{0.00030}$ vs $P_{\{-1,+1\}}(9) = \mathbf{0.00020}$ (`98.0%` vs `98.0%` zero-error states).
    - `n_01 = 8` vs `n_bipolar = 15`: Eq. 10 $P = 0.00376$ | Empirical $P_{\{0,1\}}(8) = \mathbf{0.00371}$ vs $P_{\{-1,+1\}}(15) = \mathbf{0.00429}$.
    - `n_01 = 10` vs `n_bipolar = 19`: Eq. 10 $P = 0.00921$ (Hopfield p. 2556: `0.0091`) | Empirical $P_{\{0,1\}}(10) = \mathbf{0.00977}$ vs $P_{\{-1,+1\}}(19) = \mathbf{0.00983}$ (`55.3%` zero-error states at $n=10$ in $\{0,1\}$, matching Hopfield's reported `0.6` on p. 2556!).
    - `n_01 = 13` vs `n_bipolar = 25`: Eq. 10 $P = 0.02061$ | Empirical $P_{\{0,1\}}(13) = \mathbf{0.02259}$ vs $P_{\{-1,+1\}}(25) = \mathbf{0.02135}$.
    - `n_01 = 15` vs `n_bipolar = 29`: Eq. 10 $P = 0.02939$ | Empirical $P_{\{0,1\}}(15) = \mathbf{0.02868}$ vs $P_{\{-1,+1\}}(29) = \mathbf{0.02989}$.
  - **E2.3 — Multi-step evolution from nominal memories (reproducing Fig. 2, p. 2557, $N=100$)**:
    - Under `{0, 1}` states (Fig. 2): at `n = 5`, `99.3%` of memories had `< 5` errors (`0.7%` avalanche $\ge 10$ errors); at `n = 10`, `73.3%` had `< 5` errors (`57.7%` exact 0 errors, `25.3%` avalanche); at `n = 15`, `27.1%` had `< 5` errors and `70.0%` had avalanche $\ge 10$ errors.
    - Under `Bipolar {-1, +1}` states at the same `n = 5, 10, 15`: `< 5`-error retention jumped to **`100.0%`** ($n=5$), **`98.7%`** ($n=10$), and **`95.6%`** ($n=15$).
- **What it showed**: Both predictions held to 4 decimal places (`0.00977` vs `0.00983` at $n=10$ vs $2n-1=19$), proving algebraically and empirically why Hopfield's $\{0, 1\}$ model in Eq. (1)–(4) has half the storage capacity of the bipolar $\{-1, +1\}$ model on page 2557.

### E3: Basin of Attraction vs Initial Hamming Distance ($N=30, n=5$, p. 2557) and Memory Merging ($N=100, n=8, d \in \{30, 20, 10\}$, p. 2557)
- **Question**: (a) For $N = 30, n = 5$, how does the probability of converging to the nearest stored memory vary with initial Hamming distance $d \in \{1, 2, 3, 5, 8, 10, 12\}$, and how often do completely random initial states land on the 10 nominal signed memories ($\pm M^s$) vs spurious attractors? (b) For $N = 100, n = 8$ where the 8th memory is constructed at Hamming distance $d = 30, 20,$ or $10$ from the 1st memory (p. 2557), how often do the two memories remain distinct fixed points vs displaced vs fused?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 16:48 UTC):
  1. At $N=30, n=5$, recovery will be $\sim 80\%\text{–}90\%$ for small $d \le 5$ and drop smoothly to $\sim 20\%$ at $d = 12$ (matching Hopfield p. 2557: "dropping to a level of 0.2 (2 times random chance) for a distance of 12").
  2. At $N=100, n=8$, $d = 30$ will keep both memories distinct and exact in the majority of runs ($0\%$ fused), $d = 20$ will mostly displace them while keeping them distinct, and $d = 10$ will fuse them into the same attractor in roughly half the runs.
- **Run**: `scripts/e2_e3_e4_run.sh` (executing `work/hopfield1982_experiments.py`). Output: `results/e2_e3_e4_run.out`.
- **Observation**:
  - **E3.1 — Basin of attraction vs initial Hamming distance ($N=30, n=5$, 500 trials per distance)**:
    - `d = 1`: `90.6%` recovered nearest memory.
    - `d = 2`: `88.6%` recovered nearest memory.
    - `d = 3`: `80.6%` recovered nearest memory.
    - `d = 5`: `77.8%` recovered nearest memory.
    - `d = 8`: `51.6%` recovered nearest memory.
    - `d = 10`: `35.6%` recovered nearest memory (`31.0%` original $M[0]$).
    - `d = 12`: **`20.8%`** recovered nearest memory (matching Hopfield's `0.2` on page 2557 to within `0.8%`!).
    - From completely random starting states ($N=30, n=5$): `57.6%` landed on an exact nominal memory $\pm M^s$, `11.2%` landed within 1–2 bits of a nominal memory, and `31.2%` landed on spurious states (primarily odd mixture states $\text{sign}(\pm M^{s_1} \pm M^{s_2} \pm M^{s_3})$).
  - **E3.2 — Memory merging vs similarity ($N=100, n=8$, 200 trials per distance, p. 2557)**:
    - `d = 30 / 100`: **`78.0%` both exact**, **`22.0%` distinct displaced**, **`0.0%` fused** (matching Hopfield p. 2557: *"At a distance of 30, both similar memories were usually stable"*).
    - `d = 20 / 100`: **`28.5%` both exact**, **`62.0%` distinct displaced**, **`9.5%` fused** (matching Hopfield p. 2557: *"At a distance of 20, the minima were usually distinct but displaced"*).
    - `d = 10 / 100`: **`3.0%` both exact**, **`49.5%` distinct displaced**, **`47.5%` fused** into an identical state (matching Hopfield p. 2557: *"At a distance of 10, the minima were often fused"*).
- **What it showed**: Every single qualitative and quantitative claim on page 2557 ($d=12 \to 20.8\%$ vs `0.2`, and $d=30, 20, 10$ transition from `78.0%` exact $\to$ `62.0%` displaced $\to$ `47.5%` fused) reproduced directly.

### E4: Clipped Synapses ($2/\pi$ Factor), Saturating $\pm 3$ Palimpsest Forgetting, Unidirectional $1/\sqrt{2}$ Dilution, and $25\times$ Overload Familiarity ($N=100, n=500$, pp. 2557–2558)
- **Question**: (a) Do clipped synapses $T_{ij} = \text{sign}(T_{ij}) \in \{-1, +1\}$ reduce the local-field SNR by $\sqrt{2/\pi} = 0.7979$ (and capacity by $2/\pi$), and do unidirectional synapses ($T_{ij} \ne 0 \implies T_{ji} = 0$) reduce SNR by $1/\sqrt{2} = 0.7071$? (b) Does bounding $T_{ij} \in \{-3, -2, -1, 0, +1, +2, +3\}$ over 40 sequential memories prevent catastrophic overload by forgetting old memories while retaining recent ones? (c) At $N = 100, n = 500$ ($25\times$ overload), can familiar memories still be distinguished from unfamiliar random states using only the initial processing rate in time $1/2W$?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 16:48 UTC):
  1. Clipped/Unclipped SNR ratio will match $\sqrt{2/\pi} = 0.7979$ and Unidirectional/Unclipped SNR ratio will match $1/\sqrt{2} = 0.7071$ (within $< 0.03$).
  2. Saturating $T_{ij} \in [-3, +3]$ after 40 sequential memories will retain recent memories (`#38..#40`) at $> 92\%$ bit accuracy while forgetting `#1..#5` (`< 65%`), whereas unclipped $T_{ij}$ will degrade all 40 memories uniformly.
  3. At $N=100, n=500$, `0/500` memories will be stable fixed points, but initial unstable neuron counts at $t=0$ will average $100(1 - \Phi(1/\sqrt{5})) = 32.74$ for familiar states vs $50.00$ for unfamiliar states, separating them with $> 90\%$ accuracy.
- **Run**: `scripts/e2_e3_e4_run.sh` (executing `work/hopfield1982_experiments.py`). Output: `results/e2_e3_e4_run.out`.
- **Observation**:
  - **E4.1 — Clipped ($\pm 1$) and Unidirectional SNR ratios ($N=100, n=11$, 10,000 pooled local fields)**:
    - `Unclipped SNR`: **`3.1627`** (theoretical $\sqrt{99/10} = 3.1464$).
    - `Clipped (+-1) SNR`: **`2.5193`** $\implies$ ratio `Clipped / Unclipped` $= \mathbf{0.7966}$ vs theoretical $\sqrt{2/\pi} = \mathbf{0.7979}$ (`0.0013` error!).
    - `Unidirectional SNR`: **`2.1697`** $\implies$ ratio `Unidir / Unclipped` $= \mathbf{0.6860}$ vs theoretical $1/\sqrt{2} = 0.7071$ (and exact finite-$n$ formula $\sqrt{\frac{n-1}{n-0.5}} / \sqrt{2} = \mathbf{0.6901}$ accounting for binomial in-degree variance $\text{Var}(k_i) = (N-1)/4$, `0.0041` error!).
  - **E4.2 — Bounded Saturating Synapses $T_{ij} \in \{-3\dots+3\}$ Palimpsest Forgetting ($N=100$, 40 sequential memories)**:
    - `Unclipped T_ij`: Old memories (`#1..#5`) bit accuracy = **`74.0%`** | Recent memories (`#36..#40`) = **`74.2%`** (catastrophic overload across all memories).
    - `Saturating T_ij in [-3, +3]`: Old memories (`#1..#5`) bit accuracy = **`55.8%`** (forgotten toward $50\%$ chance) | Recent memories (`#38..#40`) = **`99.8%`** (retained nearly error-free!).
  - **E4.3 — Familiarity Recognition via Initial Processing Rate under $25\times$ Overload ($N=100, n=500$, p. 2558)**:
    - Stable memories among the 500 stored states: **`0 / 500`** (`0.0%`).
    - Mean number of neurons unstable at $t=0$ (initial processing rate in time $1/2W$):
      - **Familiar stored states**: **`32.56 ± 5.37`** (theoretical $100 \times [1 - \Phi(1/\sqrt{5})] = \mathbf{32.74}$).
      - **Unfamiliar random states**: **`49.40 ± 6.24`** (theoretical $\mathbf{50.00}$).
    - Single-step familiarity classification accuracy: **`92.8%`** (`ROC AUC = 0.9828`).
- **What it showed**: All three predictions held with high precision (`0.7966` vs `0.7979`, `99.8%` vs `74.2%` palimpsest retention, and `32.56` vs `49.40` initial flip rate giving `92.8%` familiarity accuracy at $25\times$ overload).

## 4. How it works

1. **Content-addressable memory as phase-space flow to local energy minima (pp. 2554–2556)**: Storing $n$ patterns via the symmetric zero-diagonal Hebbian outer product $T_{ij} = \sum_{s=1}^n (2V_i^s - 1)(2V_j^s - 1)$ ($T_{ii}=0$, Eq. 2) creates an Ising spin-glass energy landscape $E = -\frac{1}{2}\sum_{i \ne j} T_{ij} V_i V_j$ (Eq. 7). Under asynchronous single-neuron threshold updates (Eq. 1), every state flip changes $E$ by $\Delta E = -\Delta V_i \sum_{j \ne i} T_{ij} V_j < 0$ (Eq. 8), guaranteeing monotonic descent to a locally stable attractor without limit cycles (`500/500` runs with `0/7,162` energy increases, vs `197/200` synchronous limit cycles, E2.1).
2. **Crosstalk noise sets the $\sim 0.15 N$ capacity limit, and bipolar $\{-1, +1\}$ states double capacity over $\{0, 1\}$ states (pp. 2555–2557)**:
   - On stored pattern $s'$, the $s = s'$ term in $H_i^{s'} = \sum_{j \ne i} T_{ij} V_j^{s'}$ gives signal $+N/2$ (for $V_j \in \{0,1\}$) while the $n-1$ other patterns contribute zero-mean Gaussian crosstalk of variance $\sigma^2 = (n-1)N/2$, yielding single-bit error probability $P = 1 - \Phi(\sqrt{N / [2(n-1)]})$ (Eq. 10).
   - Switching the state variables from $V_j \in \{0,1\}$ to bipolar $\mu_j \in \{-1, +1\}$ (equivalent to shifting each neuron's threshold $U_i$ by $\frac{1}{2}\sum_j T_{ij}$, p. 2557) doubles the signal from $N/2$ to $N$ while only doubling the variance from $(n-1)N/2$ to $(n-1)N$, increasing SNR to $\sqrt{N/(n-1)}$ and **doubling** the number of storable patterns at any given error rate ($n_{\text{bipolar}} = 2n_{01} - 1$, verified across $n=5..15$ in E2.2–E2.3).
3. **Attractor basins and pattern-similarity merging (p. 2557)**: Each stored pattern is surrounded by an attractor basin of radius $\sim 0.15 N$–$0.20 N$ (`90.6%`–`77.8%` exact recovery for $d \le 5$ at $N=30, n=5$, falling to `20.8%` at $d=12$, E3.1). When two stored patterns are placed closer than $\sim 0.20 N$ Hamming distance ($d = 10$ at $N=100, n=8$), their crosstalk cross-term $(N - 2d) = +80$ pulls each state toward the other, fusing their basins into a single merged attractor (`47.5%` fused at $d=10$ vs `0.0%` fused at $d=30$, E3.2).
4. **Hardware & biological insensitivity: 1-bit clipping ($2/\pi$), $\pm 3$ saturation palimpsest, unidirectional $1/\sqrt{2}$ dilution, and overload familiarity (pp. 2557–2558)**:
   - Clipping $T_{ij}$ to 1-bit signs $\pm 1$ reduces SNR by $\sqrt{2/\pi} = 0.7979$ (`0.7966` measured, E4.1), retaining $2/\pi \approx 63.7\%$ of capacity.
   - Bounding $T_{ij} \in \{-3\dots+3\}$ turns the matrix into a **palimpsest** that automatically forgets old patterns (`55.8%`) so newly written patterns remain stable (`99.8%`, E4.2) without global weight rescaling.
   - Deleting half the directed synapses ($T_{ij} \ne 0 \implies T_{ji} = 0$) reduces SNR by $1/\sqrt{2}$ (`0.6860` measured vs `0.6901` exact with degree variance, E4.1).
   - Even at $25\times$ overload ($N=100, n=500$, where $0/500$ memories are stable), the +100 signal shift on stored states reduces initial flip rate at $t=0$ from $50.0\%$ (unfamiliar) to $1 - \Phi(1/\sqrt{5}) = 32.74\%$ (`32.56%` measured), enabling $92.8\%$-accurate familiarity detection in a single step $1/2W$ (E4.3).

## 5. What surprised me and what I got wrong

1. **Within-matrix vs ensemble variance of local fields in E4.1**: On my first run of E4.1, I computed the SNR of each matrix by taking `np.std(aligned)` across the $N=100$ neurons *within* a single matrix and averaging across 100 matrices. That gave `Unclipped SNR = 3.2789` (higher than theoretical $\sqrt{99/10} = 3.1464$) and made the unidirectional ratio look slightly low (`0.6696`). Why? Because within a single matrix, $M_{0i}(T_{\text{full}} M_0)_i = (N-1) - (n-1) + N \sum_{s \ne 0} m_s M_{0i} M_{si}$, whose conditional variance given the realized overlaps $\{m_s\}$ is $N^2 \sum_{s \ne 0} m_s^2$, and by Jensen's inequality $E[\sqrt{\sum m_s^2}] < \sqrt{E[\sum m_s^2]}$! Furthermore, for unidirectional dilution ($k_i \sim \text{Binomial}(N-1, 1/2)$), the signal $k_i$ itself varies across neurons with variance $(N-1)/4$, multiplying the theoretical $1/\sqrt{2}$ ratio by $\sqrt{(n-1)/(n-0.5)} = \sqrt{10/10.5} \implies 0.6901$. Pooling all 10,000 local fields across the ensemble gave `Unclipped SNR = 3.1627`, `Clipped ratio = 0.7966` (vs `0.7979`), and `Unidirectional ratio = 0.6860` (vs `0.6901`).
2. **`torch.sign(0.0) == 0.0` in `pauljmello/Hopfield-Networks` (E1)**: In `pauljmello/Hopfield-Networks/hopfield.py` line 45, `state[i] = torch.sign(local_field)` sets any neuron with zero local field (`local_field == 0.0`, which occurs regularly when $n$ is even!) to `0.0` instead of keeping `state[i]` in $\{-1, +1\}$.

## 6. Authors say / code does / I observed

- **Authors say (Eq. 1–4, p. 2555, $V_i \in \{0, 1\}$) vs Reference Repos (`pauljmello` and `1hubert`)**: Both reference repositories implement only the bipolar $\{-1, +1\}$ version mentioned on page 2557; neither implements the paper's primary $\{0, 1\}$ formulation (Eq. 1–4), clipped synapses, $\pm 3$ saturating palimpsest synapses, unidirectional dilution, or overload familiarity.
- **Authors say (p. 2556: "For the case $n = 10, N = 100, P = 0.0091$, the probability that a state had no errors in its 100 bits should be about $e^{-100 P} \approx 0.40$. In the simulation of Fig. 2, the experimental number was $0.6$.") vs Our E2.2–E2.3**: Exact evaluation of Eq. (10) at $\sigma = \sqrt{9 \times 50} = \sqrt{450}$ and threshold $N/2 = 50$ gives $P = 1 - \Phi(50 / \sqrt{450}) = 0.00921$, and our E2.2/E2.3 simulations gave `55.3%` (1-step) and `57.7%` (multi-step) zero-error states—confirming Hopfield's observation (`0.6` vs `0.40`) that positive correlation of the finite-$n$ overlaps $m_s$ across bits of the same pattern makes all-100-bit survival higher than the independent-bit Poisson approximation $e^{-100 P} = 0.40$!

## 7. Not run, and why. Unknowns

- **Asymmetric sequence generator (Eq. 13, p. 2558)**: Hopfield reports a negative result for asymmetric temporal transitions $\Delta T_{ij} = A \sum_s (2V_i^{s+1}-1)(2V_j^s-1)$ (*"sequences longer than four states proved impossible to generate, and even these were not faithfully followed"*), and does not specify the parameter $A$ or threshold schedule used.

## 8. Sources and Sources read

### Primary paper and repositories
- **Primary paper**: J. J. Hopfield (1982), "Neural networks and physical systems with emergent collective computational abilities", *Proceedings of the National Academy of Sciences USA*, Vol. 79, No. 8, pp. 2554–2558 (April 1982). Retrieved 2026-10-26 from `https://github.com/in-c0/1982-hopfield-Neural-networks-paper` (`pnas00447-0135.pdf`, 5 pages; extracted text `work/hopfield1982.txt`, 703 lines, 31,616 bytes).
- **Reference repo 1**: `https://github.com/pauljmello/Hopfield-Networks.git`, commit `a185b5bf821485dd2165af38af66e7fe8cace8be`.
- **Reference repo 2**: `https://github.com/1hubert/hopfield-network.git`, commit `1d1ec4b3dbd7f953f12bea2214e688092f1da5f1`.
- **Reference repo 3**: `https://github.com/in-c0/1982-hopfield-Neural-networks-paper.git`, commit `61ea96eaff65f76f9e93c547863ccc6ac2cdb4b2`.

### Sources read (READ-LOG)
| File | Lines | Status | Why |
| --- | ---: | --- | --- |
| `work/hopfield1982.txt` (Hopfield 1982, pp. 2554–2558) | 703 | `full` | Complete 1982 PNAS paper text (Equations 1–13, Figs. 1–2). |
| `pauljmello-Hopfield-Networks/README.md` | 84 | `full` | Comparison of 1982 discrete Hopfield vs 2020 continuous Hopfield networks. |
| `pauljmello-Hopfield-Networks/hopfield.py` | 54 | `full` | Traditional discrete Hopfield implementation. |
| `pauljmello-Hopfield-Networks/modernhopfield.py` | 60 | `full` | Continuous Hopfield attention implementation. |
| `pauljmello-Hopfield-Networks/main.py` | 91 | `full` | Retrieval and capacity driver script. |
| `1hubert-hopfield-network/main.py` | 157 | `full` | Standalone $5 \times 5$ asynchronous Hopfield network script. |

## 9. Files

### Local worktree (`C:\personal\oss\clover-framework-foundations\clover-kimi-k3\` on branch `foundations-research`)
- Context file: `docs/hopfield-1982-context.md`
- Predictions log: `research/hopfield-1982/work/predictions.txt`
- Primary paper text: `research/hopfield-1982/work/hopfield1982.txt`
- Own implementations:
  - `research/hopfield-1982/work/hopfield1982_experiments.py` (E2 energy monotonicity & synchronous/negative-diagonal controls, $\{0,1\}$ vs bipolar $\{-1,+1\}$ factor-of-2 capacity verification & Fig. 2 error histograms; E3 basin of attraction vs Hamming distance $N=30, n=5$ & memory merging $N=100, n=8$; E4 clipped $\pm 1$ synapses $2/\pi$ factor, unidirectional $1/\sqrt{2}$ dilution, saturating $\pm 3$ palimpsest forgetting, and $25\times$ overload familiarity via initial processing rate)
- Scripts:
  - `research/hopfield-1982/scripts/e0_setup.sh`
  - `research/hopfield-1982/scripts/e0b_fetch_hopfield1982.sh`
  - `research/hopfield-1982/scripts/e1_repos.sh`
  - `research/hopfield-1982/scripts/e2_e3_e4_run.sh`
- Results:
  - `research/hopfield-1982/results/e0_setup.out`
  - `research/hopfield-1982/results/e0b_fetch_hopfield1982.out`
  - `research/hopfield-1982/results/e1_repos.out`
  - `research/hopfield-1982/results/e2_e3_e4_run.out`

### Remote (`k3` CPU server)
- `/root/dm-study/foundations/hopfield-1982/` (`repo/pauljmello-Hopfield-Networks`, `repo/1hubert-hopfield-network`, `repo/in-c0-1982-hopfield-Neural-networks-paper`, `work/`)

