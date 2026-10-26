# McCulloch & Pitts (1943) — *A Logical Calculus of the Ideas Immanent in Nervous Activity* — Context

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh.

## 1. Status

- **E1 (Clone & run reference repositories on CPU, inspect rules vs 1943 paper)**: DONE
- **E2 (Exact 1943 Equation (1) & Theorem 1–3 TPE compiler/verifier across all 16 2-input & 256 3-input Boolean functions)**: DONE
- **E3 (Cutaneous heat-from-transient-cooling illusion Fig. 1e & Equivalence Theorems 4–6 Fig. 1f–h)**: DONE
- **E4 (Nets with circles: Theorem 7 learning-by-circle Fig. 1i, Theorem 8 state-transition matrix over $2^p$ cyclic states, and Theorem 10 universal/existential/modulo-$n$ cyclic nets)**: DONE

## 2. What this is

Authors say (Warren S. McCulloch and Walter Pitts, "A Logical Calculus of the Ideas Immanent in Nervous Activity", *Bulletin of Mathematical Biophysics*, Vol. 5, pp. 115–133, 1943; reprinted in *Bulletin of Mathematical Biology*, Vol. 52, No. 1/2, pp. 99–115, 1990):
- "Because of the 'all-or-none' character of nervous activity, neural events and the relations among them can be treated by means of propositional logic. It is found that the behavior of every net can be described in these terms, with the addition of more complicated logical means for nets containing circles; and that for any logical expression satisfying certain conditions, one can find a net behaving in the fashion it describes." (Abstract, p. 99)
- Section 2 (p. 101) states five physical assumptions: (1) neuron activity is an "all-or-none" process; (2) a fixed number $\theta_i > 0$ of synapses must be excited within the period of latent addition to excite a neuron; (3) the only significant delay is the unit synaptic delay (represented by the precession functor $S$, where $S(P)(t) \equiv P(t-1)$); (4) the activity of **any** inhibitory synapse absolutely prevents excitation of the neuron at that time; (5) the net structure does not change with time.
- Under Equation (1) (Section 2, p. 103), neuron $c_i$ fires at discrete time $t$ iff at time $t-1$ none of its inhibitory afferents $c_{j_m}$ fired and the active excitatory afferents supply enough synapses to meet its threshold.
- Theorems 1–3 (pp. 103–106) prove that acyclic ("order 0") nets compute and realize exactly the class of Temporal Propositional Expressions (TPEs), which are those Boolean functions of delayed afferents that evaluate to **false when all constituent afferents are false**.
- Theorems 4–7 (pp. 107–108, Fig. 1f–i) prove extended equivalence (up to synaptic delay) between absolute and relative inhibition, extinction and inhibition, temporal and spatial summation, and alterable (learning) synapses and cyclic nets.
- Section 3 (Theorems 8–10, pp. 108–113) solves cyclic nets of order $p$ via transition equations over $2^p$ state conjunctions $X_1, \dots, X_{2^p}$ and characterizes realizable temporal predicates with existential/universal quantifiers over past time and modulo-$n$ periodic circuits.

## 3. Experiments

### E1: Clone and run reference repositories on CPU, inspect rules vs 1943 paper
- **Question**: How do open-source reference repositories (`kaisdukes/mcculloch-pitts-neuron` and `CodeForge-00/mcculloch_pitts_neuron`) define the McCulloch–Pitts neuron, what do they output when run on CPU, and where do their equations match or differ from McCulloch & Pitts's 1943 Equation (1) and Theorem 3?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 14:05 UTC):
  1. Both reference repositories will implement the post-1943 textbook linear-threshold variant (summing signed weights or excitatory minus inhibitory inputs against $\theta$) rather than 1943 Equation (1) (unit synaptic delay $S$, non-negative excitatory synapse counts, and absolute veto inhibition).
  2. `CodeForge-00/mcculloch_pitts_neuron` will implement basic Boolean logic gates, a 2-layer XOR network, and a cyclic memory cell, and will run cleanly on CPU with exit code 0.
  3. Because textbook implementations allow negative weights and $\theta \le 0$, their single neuron can output 1 on all-zero input $(0, \dots, 0)$ (e.g., $\text{NOT}(0)=1$ or $\text{NAND}(0,0)=1$), whereas 1943 Theorem 3 proves no order-0 net under the 1943 axioms can output 1 when all peripheral afferents are 0.
- **Run**: `scripts/e1_repos.sh` and `scripts/e1b_tests.sh`. Outputs: `results/e1_repos.out`, `results/e1b_tests.out`.
- **Observation**:
  - **Repositories cloned & line counts**:
    1. `kaisdukes/mcculloch-pitts-neuron` at commit `7881b596492b366130b379f3372dd3801e9606eb` (2023-06-25, MIT License, 5 Python files totaling 91 lines of code plus 121 lines of `README.md`).
    2. `CodeForge-00/mcculloch_pitts_neuron` at commit `54ae5cfadb6672c21dd7e6b0daaccf6894d8f933` (2026-10-25, 4 Python files totaling 182 lines of code plus 70 lines of `README.md`).
  - **What they actually implement (surprise vs Prediction 1)**:
    - Contrary to Prediction 1 (which expected signed linear-threshold weights), **both** repositories actually implement McCulloch & Pitts's explicit excitatory/inhibitory separation with **absolute veto inhibition**:
      - `kaisdukes/src/mcp_neuron.py` (lines 14–17): `sum_excitatory = sum(input.value for input in self.inputs if input.type == InputType.EXCITATORY)`, `has_inhibitory = any(input.value for input in self.inputs if input.type == InputType.INHIBITORY)`, returning `sum_excitatory >= self.threshold and not has_inhibitory`.
      - `CodeForge-00/neuron.py` (lines 17–20): `if any(inhibitory): return 0; return int(sum(excitatory) >= self.threshold)`.
    - Moreover, `CodeForge-00/network.py` (lines 10–30) explicitly implements the 1943 **unit synaptic delay** (`step()` computes all neurons' `t+1` outputs simultaneously from their `t` inputs, so `xorNetwork()` requires `steps=2` synaptic delays and fails at `steps=1` in `test_xor_needs_two_steps`, and `memoryNetwork()` implements a 1-neuron cyclic latch `mem` with excitatory `['set', 'mem']` and inhibitory `['reset']`).
  - **How each repo circumvents Theorem 3 (`f(0,...,0) = 0`) to build a `NOT` gate**:
    - In the 1943 paper, Theorem 1 (p. 103) explicitly requires threshold $\theta_i > 0$ ("Let $c_i$ be any neuron of $\mathcal{N}$ with a threshold $\theta_i > 0$"), and Theorem 3 (p. 104) proves that a formula is realizable by an order-0 net of peripheral afferents **if and only if** it is false when all constituent afferents are false ("the last line in its truth-table contains an 'F'—or there is no term in its Hilbert disjunctive normal form composed exclusively of negated terms").
    - Repo 1 (`kaisdukes/src/logic_gates.py`, lines 6–11) builds `not_gate(a)` by setting **`threshold=0`** (`McpNeuron(inputs=[Input(type=InputType.INHIBITORY, value=a)], threshold=0)`), violating $\theta_i > 0$ so the neuron fires spontaneously with zero excitatory inputs whenever `a=False`.
    - Repo 2 (`CodeForge-00/neuron.py`, lines 30–32) keeps `threshold=1` (`> 0`), but builds `notGate(a)` by injecting a **hardcoded constant `1` excitatory input** (`McCullochPittsNeuron(threshold=1).fire([1], inhibitory=[a])`), which corresponds in the 1943 paper (p. 108, remark under Theorem 7) to adding an external continuously active afferent or a reverberating circle.
  - **Execution & unit tests**:
    - Running `demo.py` in `CodeForge-00` (`results/e1_repos.out`) printed truth tables for `AND (umbral 2)`, `OR (umbral 1)`, `NOT`, `a Y NO b`, 2-step `XOR` (`00->0, 01->1, 10->1, 11->0`), and a 7-step trace of the cyclic memory cell (`t=0: 0, t=1(set=1): 1, t=2: 1, t=3: 1, t=4(reset=1): 0, t=5: 0, t=6: 0`).
    - Running unit tests directly (`e1_repos.sh`) failed first when invoked as `python ./tests/neuron_test.py` without setting package root (`ModuleNotFoundError: No module named 'src'` / `'neuron'`); running with `PYTHONPATH=. python3 -m unittest` (`e1b_tests.sh`, `results/e1b_tests.out`) passed all `3/3` tests in Repo 1 and `6/6` tests in Repo 2 (`rc=0`).
- **What it showed**: Prediction 1 was **partially wrong** in an informative way: neither repository used signed perceptron weights—both implemented absolute inhibitory veto, and `CodeForge-00` also implemented discrete unit synaptic delay and a cyclic latch. Prediction 2 and Prediction 3 held, and comparing the two repositories' `not_gate` implementations exposed the exact two ways (`threshold=0` vs an always-on auxiliary excitatory line `[1]`) that implementations bypass McCulloch & Pitts's Theorem 3 (`f(0,...,0)=0`).

### E2: Exact 1943 Equation (1) & Theorem 1–3 TPE compiler/verifier across all 16 2-input and 256 3-input Boolean functions
- **Question**: Under McCulloch & Pitts's exact 1943 assumptions ($\theta_i \ge 1$ integer, integer excitatory synapse counts $n_{is} \ge 0$, absolute veto inhibition, unit synaptic delay $S$, and no auxiliary constant-1 input): (a) Which of the 16 2-input Boolean functions can a **single** neuron compute? (b) Does Theorem 2's constructive synthesis (using only the 4 fundamental elements of Fig. 1a–d: precession delay $S$, 2-input disjunction, 2-input conjunction, and conjoined negation $S_1 \cdot \neg S_2$) compile all 8 of 16 2-input and all 128 of 256 3-input Boolean functions with $f(0,\dots,0)=0$ into valid order-0 nets whose Equation (1) simulation matches the truth table on 100% of inputs, while all 128 functions with $f(0,0,0)=1$ are impossible?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 14:15 UTC):
  1. A single 1943 neuron ($\theta \ge 1$) realizes exactly **7 of 16** 2-input functions: `FALSE` (`0000`), `AND` (`0001`), `x1 & ~x2` (`0010`), `x1` (`0011`), `~x1 & x2` (`0100`), `x2` (`0101`), and `OR` (`0111`). `XOR` (`0110`) is the unique 2-input function with $f(0,0)=0$ that requires a multi-layer net (depth 2).
  2. Multi-layer order-0 compiler (Theorems 1–3): All **8/16** 2-input functions with $f(0,0)=0$ and all **128/256** 3-input functions with $f(0,0,0)=0$ will compile via Theorem 2's Hilbert DNF decomposition $(S_1 \cdot \dots \cdot S_m) \cdot \neg (S_{m+1} \lor \dots \lor S_n)$ (page 106, top) into order-0 nets that pass 100% truth-table verification under Equation (1), whereas all **128/256** functions with $f(0,0,0)=1$ fail Theorem 3 and cannot be realized by any order-0 net with $\theta_i \ge 1$.
- **Run**: `scripts/e2_tpe_compiler.sh` (executing `work/mp1943_calculus.py`). Output: `results/e2_tpe_compiler.out`.
- **Observation**:
  - **Single 1943 neuron on 2 inputs ($\theta \in \{1..4\}$, $n_{is} \in \{0..3\}$, inhibitory subsets of $\{x_1, x_2\}$)**:
    - Exhaustive enumeration over all parameter combinations realized **7 of 16** truth tables:
      1. `(0, 0, 0, 0)` `FALSE`: $\theta=1, \text{exc}=\{\}, \text{inh}=[]$
      2. `(0, 0, 0, 1)` `AND` ($x_1 \cdot x_2$, Fig. 1c): $\theta=2, \text{exc}=\{x_1:1, x_2:1\}, \text{inh}=[]$
      3. `(0, 0, 1, 0)` $x_1 \cdot \neg x_2$ (Fig. 1d): $\theta=1, \text{exc}=\{x_1:1\}, \text{inh}=[x_2]$
      4. `(0, 0, 1, 1)` `PROJ` $x_1$ (Fig. 1a): $\theta=1, \text{exc}=\{x_1:1\}, \text{inh}=[]$
      5. `(0, 1, 0, 0)` $\neg x_1 \cdot x_2$: $\theta=1, \text{exc}=\{x_2:1\}, \text{inh}=[x_1]$
      6. `(0, 1, 0, 1)` `PROJ` $x_2$: $\theta=1, \text{exc}=\{x_2:1\}, \text{inh}=[]$
      7. `(0, 1, 1, 1)` `OR` ($x_1 \lor x_2$, Fig. 1b): $\theta=1, \text{exc}=\{x_1:1, x_2:1\}, \text{inh}=[]$
    - `XOR` `(0, 1, 1, 0)` is the unique 2-input function with $f(0,0)=0$ that is impossible for a single 1943 neuron, and all 8 tables with $f(0,0)=1$ (`#8..#15`) are impossible.
  - **Theorem 2 & 3 constructive compiler across all 16 2-input and 256 3-input Boolean functions**:
    - Using only the four fundamental elements of Fig. 1a–d and the Hilbert DNF rewrite on page 106 line 2 ($(S_1 \cdot \dots \cdot S_m) \cdot \neg (S_{m+1} \lor \dots \lor S_n)$), `compile_tpe_theorem2` compiled and verified **8/16** 2-input functions and **128/256** 3-input functions with **100% truth-table agreement** under Equation (1) at uniform synaptic delay $d=3$ (with non-afferent neuron counts ranging from `min=1` for `FALSE` to `median=11`, `max=17` for the 7-minterm function).
    - All **8/16** 2-input and **128/256** 3-input functions with $f(0,\dots,0)=1$ were rejected by the Theorem 3 condition.
  - **Negative controls**:
    - **Negative Control 1**: Simulating `1,000` randomly wired order-0 nets of 6 neurons ($\theta_i \in \{1..4\}$) on all-zero peripheral afferents resulted in `0/1000` firing at any time step $t \in \{0..6\}$, empirically confirming Theorem 3 necessity.
    - **Negative Control 2**: Incrementing `theta` by `+1` on a single minterm neuron (`term_0`) inside the compiled 3-input parity ($x_1 \oplus x_2 \oplus x_3$) network caused `verify_compiled_net` to return `False`.
- **What it showed**: Both predictions held (`7/16` single-neuron tables, `8/16` and `128/256` multi-layer order-0 tables compiled and verified with 0 errors).

### E3: Cutaneous heat-from-transient-cooling illusion (Fig. 1e) & Equivalence Theorems 4–6 (Fig. 1f–h)
- **Question**: When we build and simulate the exact networks from pages 105–108 of McCulloch & Pitts (1943)—specifically (a) the Fig. 1e network for the cutaneous illusion of heat produced by transient cooling, (b) the Theorem 4 (Fig. 1f) replacement of relative inhibition by absolute inhibition, and (c) the Theorem 5 (Fig. 1g) replacement of extinction and Theorem 6 (Fig. 1h) replacement of temporal summation by spatial summation—what exact firing traces do they produce, and does extended equivalence ($S^n$ time shift) hold across all input sequences?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 14:25 UTC):
  1. On Fig. 1e (`N3(t) = N1(t-1) v (N2(t-3) . ~N2(t-2))` and `N4(t) = N2(t-2) . N2(t-1)`): a 1-step cold pulse (`N2=[1,0,0,0,0]`) will fire heat sensation `N3` at `t=3` (`N3(3)=1`) and cold sensation `N4` at **no** time step (`N4=0` for all $t$), whereas a sustained cold stimulus (`N2=[1,1,1,0,0]`) will suppress `N3` (`N3=0` at all $t$, no preliminary warmth) and fire cold sensation `N4` at `t=2, 3`.
  2. On Theorem 4 (Fig. 1f): a single neuron with **relative inhibition** (threshold $\theta=1$, where $c_1$ has 1 relative-inhibitory synapse raising threshold by $+1$, $c_2$ has 1 excitatory synapse, and $c_3$ has 2 excitatory synapses, firing at $t$ iff $\neg N_1(t-1)\cdot(N_2(t-1)\lor N_3(t-1)) \lor (N_1(t-1)\cdot N_3(t-1))$ or the exact Fig. 1f formula `~N1(t-1).(N2(t-1) v N3(t-1)) v (N1(t-1).N2(t-1).N3(t-1))` corresponding to $\theta=1$, $c_1$ raising threshold by $+1$, $c_2$ having 1 excitatory synapse, $c_3$ having 1 or 2 excitatory synapses) will match the 2-layer **absolute-inhibition** net at delay $t+1$ ($S^1$ extended equivalence) on all $2^3=8$ input triples.
  3. On Theorem 5 (Fig. 1g) and Theorem 6 (Fig. 1h): the absolute-inhibition net with a delay chain will match the extinction neuron and temporal-summation neuron across all binary input streams of length 6.
- **Run**: `scripts/e3_illusions_and_equivalences.sh` (executing `work/e3_illusions_and_equivalences.py`). Output: `results/e3_illusions_and_equivalences.out`.
- **Observation**:
  - **Cutaneous illusion (Fig. 1e, pp. 106–107)**:
    - On a 1-step transient cold stimulus (`c2=[1,0,0,0,0]`), delay neuron `ca` fires at `t=1`, relative-complement neuron `cb` (`ca(t-1) & ~c2(t-1)`) fires at `t=2`, and heat sensation `c3` fires at `t=3` (`c3=[0,0,0,1,0]`) while cold sensation `c4` remains `0` at every step (`[0,0,0,0,0]`).
    - On a 3-step sustained cold stimulus (`c2=[1,1,1,0,0,0,0]`), `cb` is inhibited at `t=2,3` by the ongoing `c2=1` input, so cold sensation `c4` fires at `t=2,3` (`c4=[0,0,1,1,0,0,0]`) with **zero preliminary warmth** during `t=0..4`.
    - **Unpredicted edge effect found by extending the trace to `t=6`**: when the sustained cold stimulus is finally removed (`c2(2)=1, c2(3)=0`), the falling edge of the sustained cold stimulus satisfies $N_2(t-3) \cdot \neg N_2(t-2)$ two steps after release (`cb(4)=1` $\implies$ `c3(5)=1`), so McCulloch & Pitts's exact formula $N_3(t) \equiv N_1(t-1) \lor (N_2(t-3) \cdot \neg N_2(t-2))$ is actually a **cold-offset (falling-edge) detector**, not a duration-bounded pulse detector: it prevents preliminary warmth at the onset of sustained cooling (`t=1..4`), but fires a 1-step heat pulse 2 delays after any cold stimulus ends!
  - **Theorem 4 (Fig. 1f, relative vs absolute inhibition)**:
    - Across all $2^3 = 8$ input triples `(c1, c2, c3)`, the single relative-inhibition neuron ($\theta_{\text{eff}} = 1 + c_1$, excitation $c_2 + c_3$) at $t=1$ matched the 4-neuron absolute-inhibition network (`ca, cb, cc -> c4`) at $t=2$ ($S^1$ extended equivalence) and the Principia formula $\neg N_1 \cdot (N_2 \lor N_3) \lor (N_1 \cdot N_2 \cdot N_3)$ on `8/8` rows (`0, 1, 1, 1, 0, 0, 0, 1`).
  - **Theorem 5 (Fig. 1g, extinction) and Theorem 6 (Fig. 1h, temporal summation)**:
    - Across all $2^6 = 64$ binary input streams of length 6, the absolute-inhibition delay circuit of Fig. 1g matched $N_3(t) \equiv N_1(t-2) \cdot \neg N_1(t-3)$ on `64/64` streams, and the spatial-summation delay circuit of Fig. 1h matched $N_3(t) \equiv N_1(t-1) \cdot N_1(t-2)$ on `64/64` streams.
- **What it showed**: All three predictions held, and running the extended trace of Fig. 1e (`t=0..6`) exposed a concrete property of the 1943 equation $N_3(t) \equiv N_1(t-1) \lor (N_2(t-3) \cdot \neg N_2(t-2))$ that reading the prose on page 106 misses: because $N_2(t-3) \cdot \neg N_2(t-2)$ checks only two consecutive steps of $N_2$, it suppresses heat at the *onset* of sustained cold (`t=1..4`), but fires a heat pulse at `t=5` after sustained cold is lifted (`c2(2)=1, c2(3)=0`).

### E4: Nets with circles (Theorem 7 learning-by-circle Fig. 1i, Theorem 8 state-transition solver over $2^p$ cyclic states, and Theorem 10 universal/existential/modulo-$n$ cyclic nets)
- **Question**: How do cyclic nets (order $p \ge 1$) behave under McCulloch & Pitts's Section 3 theory? Specifically: (a) Does replacing an alterable (Hebbian-style) synapse by a reverberating circle (Theorem 7, Fig. 1i, p. 108) produce identical outputs across all $4^5 = 1,024$ 2-input histories of length 5? (b) Does Theorem 8's state-transition equation over the $2^p$ cyclic state conjunctions $X_1, \dots, X_{2^p}$ (Equations 5–9, pp. 109–110) exactly match step-by-step Equation (1) simulation on an order-$p=2$ cyclic net? (c) Do Theorem 10's three canonical cyclic structures (pp. 112–113) realize existential $(\exists z_2 < z_1)$, universal $(\forall z_2 < z_1)$, and modulo-$n$ ($C_{mn}$) predicates, and how many preimage states collapse per step under Section 4's "irreciprocity of causality" (pp. 113–114)?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 14:35 UTC):
  1. Theorem 7 (Fig. 1i): The fixed-weight cyclic network realizing $N_3(t) \equiv N_2(t-1) \lor (N_1(t-1) \cdot (\exists x < t-1)[N_1(x) \cdot N_2(x)])$ will match the alterable-synapse neuron (whose $c_1 \to c_3$ termination permanently converts into an active excitatory synapse once $N_1(x)$ and $N_2(x)$ co-occur) on `1024/1024` input histories.
  2. Theorem 8 (Eq. 5–9): On an order-$p=2$ cyclic net ($2^p = 4$ cyclic state conjunctions $X_1, \dots, X_4$), the analytical transition equation $X_i(t) \equiv \bigvee_{j=1}^{4} \Psi_{ij}(t) \cdot X_j(t-1)$ will match step-by-step Equation (1) simulation on 100% of steps across all $4^5 = 1,024$ input streams, while a negative control swapping two $\Psi_{ij}$ entries will fail.
  3. Theorem 10 & Section 4 irreciprocity: All three Theorem 10 cyclic circuits will match their exact quantified/modular predicates on 100% of streams, and computing the exact state-transition map $(s_t, u_t) \mapsto s_{t+1}$ on Fig. 1e and Fig. 1i will show forward determinism (1 image per $(s_t, u_t)$) vs backward non-invertibility (multiple $(s_t, u_t)$ pairs mapping to the same $s_{t+1}$).
- **Run**: `scripts/e4_cyclic_nets.sh` (executing `work/e4_cyclic_nets.py`). Output: `results/e4_cyclic_nets.out`.
- **Observation**:
  - **Theorem 7 (Fig. 1i, p. 108, replacing alterable synapses by a circle)**:
    - Across all $4^6 = 4,096$ 2-input histories of length 6, the stateful alterable-synapse neuron (whose $c_1 \to c_3$ termination is initially inactive and permanently becomes an active excitatory synapse after any step $x$ where $N_1(x)$ and $N_2(x)$ co-occur) produced outputs identical at every step to the fixed-weight order-1 cyclic net (`cm` with $\theta=2, \text{exc}=\{c_1:1, c_2:1, c_m:2\}$ feeding `c3` with $\theta=2, \text{exc}=\{c_2:2, c_1:1, c_m:1\}$) and to the exact Fig. 1i formula $N_3(t) \equiv N_2(t-1) \lor (N_1(t-1) \cdot (\exists x < t-1)[N_1(x) \cdot N_2(x)])$ (`4096/4096` histories matched).
  - **Theorem 8 (Eq. 5–9, pp. 109–110, solving cyclic nets via $2^p$ state conjunctions)**:
    - On the order-$p=2$ cyclic net (`c1`, `c2` with peripheral afferents `u1`, `u2`), tracking the $2^p = 4$ mutually exclusive state conjunctions $X_0, X_1, X_2, X_3$ via Equation (5) ($X_i(t) \equiv \bigvee_{j=0}^{3} \Psi_{ij}(t) \cdot X_j(t-1)$) matched step-by-step Equation (1) simulation on `1024/1024` 5-step sequences, while the negative control (permuting $\Psi_{i,2}$) dropped agreement to `432/1024`.
  - **Theorem 10 (pp. 112–113, existential, universal, and modulo-$n$ cyclic circuits)**:
    - The existential self-loop $N_i(z_1) \equiv \text{Pr}_1(z_1-1) \lor N_i(z_1-1)$ matched $(\exists z_2 < z_1)\text{Pr}_1(z_2)$ on `64/64` 6-step streams; the universal self-loop $N_i(z_1) \equiv \text{Pr}_1(z_1-1) \cdot N_i(z_1-1)$ matched $(\forall z_2 < z_1)\text{Pr}_1(z_2)$ on `64/64` streams; and the 3-neuron ring $c_1 \to c_2 \to c_3 \to c_1$ fired at $c_1=[1,4,7]$ ($t \equiv 1 \pmod 3$), $c_2=[2,5,8]$ ($t \equiv 2 \pmod 3$), and $c_3=[3,6,9]$ ($t \equiv 0 \pmod 3$).
  - **Section 4 (pp. 113–114, irreciprocity of causality)**:
    - Enumerating all $2^6 = 64$ predecessor configurations $(c_1, c_2, c_a, c_b, c_3, c_4)$ at time $t$ of the Fig. 1e network showed that only `8/16` next internal states $(c_a, c_b, c_3, c_4)$ at $t+1$ are reachable, and every reachable next state has between `4` and `12` distinct predecessor configurations (`[12, 12, 12, 12, 4, 4, 4, 4]`), demonstrating concretely why forward evolution is deterministic while backward inference of past states or afferents is non-unique.
- **What it showed**: All three predictions held across all `4,096` (Theorem 7), `1,024` (Theorem 8), and `64` (Theorem 10 & Section 4) test cases.

## 4. How it works

1. **Discrete-time all-or-none firing with unit synaptic delay and absolute inhibitory veto**: Every non-afferent neuron $c_i$ computes its binary state $N_i(t) \in \{0,1\}$ at discrete time $t$ from the state at $t-1$ using Equation (1): $N_i(t) = 1$ if and only if zero inhibitory afferents $c_{j_m}$ fired at $t-1$ and the weighted sum of active excitatory afferents $\sum_s n_{is} N_{i_s}(t-1)$ is at least the positive integer threshold $\theta_i \ge 1$ (E1, E2).
2. **Single-neuron expressivity vs multi-layer order-0 nets**: A single 1943 neuron ($\theta_i \ge 1$) on 2 inputs can compute only 7 of the 16 Boolean truth tables (`FALSE`, `AND`, `x1 & ~x2`, `x1`, `~x1 & x2`, `x2`, `OR`), missing `XOR` (`0110`) and all 8 functions with $f(0,0)=1$ (E2).
3. **Theorems 1–3 (Order-0 completeness for TPEs and the all-zero boundary)**: Composing the four fundamental elements of Fig. 1a–d (`S` delay, `OR`, `AND`, and relative complement `S1 . ~S2`) via the Hilbert DNF rewrite $(S_1 \cdot \dots \cdot S_m) \cdot \neg(S_{m+1} \lor \dots \lor S_n)$ realizes **every** Boolean function of delayed afferents that satisfies $f(0,\dots,0)=0$ (`8/16` for 2 inputs, `128/256` for 3 inputs) at a uniform synaptic delay, and **none** of the functions with $f(0,\dots,0)=1$ unless an external background afferent or cyclic reverberation is added (E1, E2).
4. **Temporal coding of sensory illusions and physiological equivalences (Theorems 4–6)**: Because every synapse introduces 1 unit of delay $S$, an acyclic net converts temporal patterns into spatial conjunctions (`N4(t) = N2(t-2) . N2(t-1)` in Fig. 1e and Fig. 1h) and relative complements (`N3(t) = N1(t-1) v (N2(t-3) . ~N2(t-2))` in Fig. 1e), making relative inhibition (Fig. 1f), post-firing threshold elevation/extinction (Fig. 1g), and temporal summation (Fig. 1h) formally equivalent to absolute inhibition and spatial summation up to a fixed number of synaptic delays $S^n$ (E3).
5. **Cycles replace state and synaptic plasticity (Theorems 7, 8, 10) and make causality irreciprocal (Section 4)**: Introducing feedback loops of order $p \ge 1$ allows a net with **fixed** thresholds and synapses to store existential quantifiers over arbitrarily remote past coincidences ($\exists x < t-1 [N_1(x) \cdot N_2(x)]$), matching a Hebbian-like alterable synapse bit-for-bit across all input histories (Theorem 7, E4), evolving deterministically over $2^p$ state conjunctions $X_1, \dots, X_{2^p}$ (Theorem 8, E4), and collapsing multiple past configurations into each successor state (`4` to `12` preimages per reachable state in Fig. 1e) so that future state is computable from present state while past state is not (Section 4, E4).

## 5. What surprised me and what I got wrong

1. **Wrong prediction on reference repositories (E1)**: I predicted that both GitHub reference repositories (`kaisdukes/mcculloch-pitts-neuron` and `CodeForge-00/mcculloch_pitts_neuron`) would implement post-1943 signed perceptron weights. In fact, both implemented McCulloch & Pitts's exact excitatory/inhibitory split with **absolute inhibitory veto**, and `CodeForge-00` also implemented the unit synaptic delay and a cyclic memory cell. What differed was how each repo bypassed Theorem 3 ($f(0,\dots,0)=0$) to build a `NOT` gate: `kaisdukes` set `threshold=0` (violating $\theta_i > 0$), while `CodeForge-00` kept `threshold=1` and passed a constant `[1]` excitatory input (matching the paper's remark on page 108).
2. **Trailing-edge heat pulse after sustained cooling in Fig. 1e (E3)**: In the paper's prose (p. 106), McCulloch and Pitts write that if a cold object "is applied for a longer time, the sensation will be only of cold, with no preliminary warmth, however transient." Running the Fig. 1e network for 5 steps (`t=0..4`) confirmed zero preliminary warmth during sustained cooling; however, extending the trace to `t=6` after removing the sustained cold stimulus at `t=3` revealed that `c3` (heat) fires at `t=5`! Because the paper's equation $N_3(t) \equiv N_1(t-1) \lor (N_2(t-3) \cdot \neg N_2(t-2))$ only inspects a 2-step window $(t-3, t-2)$, it is a **cold-offset (falling-edge) detector**: it suppresses heat while cold continues, but triggers a 1-step heat pulse 2 synaptic delays after any cold stimulus is removed.

## 6. Authors say / code does / I observed

- **Authors say (Theorem 1, p. 103) vs reference repos (`not_gate`)**: Authors require $\theta_i > 0$ on every neuron, which entails Theorem 3 ($f(0,\dots,0)=0$ on every order-0 net of peripheral afferents). `kaisdukes/src/logic_gates.py` sets `threshold=0` for `not_gate`, whereas `CodeForge-00/neuron.py` supplies an external constant `[1]` excitatory input with `threshold=1`.
- **Authors say (Section 3, p. 111–113, Theorems 9 & 10)**: Theorem 9's necessary and sufficient condition for "prehensible classes" requires exploring $2^{2^m}$ classes of functions (`B(K)`), which the authors note "we have presented ... very sketchily; we propose to expand it and certain of its implications in a further publication" and therefore provide Theorem 10 as the constructive sufficient condition (`(Ez2 < z1)`, `(z2 < z1)`, and `C_{mn}`). Our E4 experiments tested Theorem 7, Theorem 8, and Theorem 10 directly.

## 7. Not run, and why. Unknowns

- **Turing machine tape/scanner coupling (concluding paragraph of Section 3, p. 113)**: McCulloch and Pitts remark in one paragraph without construction that a net furnished with a tape, scanners connected to afferents, and motor efferents computes the Turing-computable numbers. No specific net diagram or equation is given in the 1943 paper for the tape head, so it was not simulated here (Turing's 1936/1950 machine model is covered in Study 2).
- **Theorem 9 ($2^{2^m}$ Boolean ring $\mathcal{B}_S(K)$ decision procedure, p. 111)**: The paper states Theorem 9's symbolic formula (14) with typographical corruptions in the 1990 typeset reprint (p. 111–112) and defers the full proof to a future publication; we verified the constructive cyclic theorems (Theorem 7, Theorem 8, Theorem 10) instead.

## 8. Sources and Sources read

### Primary paper and repositories
- **Primary paper**: Warren S. McCulloch and Walter Pitts (1943), "A Logical Calculus of the Ideas Immanent in Nervous Activity", *Bulletin of Mathematical Biophysics*, Vol. 5, pp. 115–133 (1943); reprinted in *Bulletin of Mathematical Biology*, Vol. 52, No. 1/2, pp. 99–115 (1990). Retrieved 2026-10-26 from `https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf` (1,272,966 bytes, 17 pages; extracted text `work/mp1943.txt`, 941 lines, 40,720 bytes).
- **Reference repo 1**: `https://github.com/kaisdukes/mcculloch-pitts-neuron.git`, commit `7881b596492b366130b379f3372dd3801e9606eb` (2023-06-25), MIT License.
- **Reference repo 2**: `https://github.com/CodeForge-00/mcculloch_pitts_neuron.git`, commit `54ae5cfadb6672c21dd7e6b0daaccf6894d8f933` (2026-10-25).

### Sources read (READ-LOG)
| File | Lines | Status | Why |
| --- | ---: | --- | --- |
| `work/mp1943.txt` (McCulloch & Pitts 1943, pp. 99–115) | 941 | `full` | Complete 1943 paper text (Sections 1–4, Theorems 1–10, Fig. 1a–i captions). |
| `kaisdukes-mp/README.md` | 121 | `full` | Historical overview and explanation of NOT/AND gates in Repo 1. |
| `kaisdukes-mp/src/mcp_neuron.py` | 17 | `full` | `McpNeuron.fire()` implementation. |
| `kaisdukes-mp/src/logic_gates.py` | 21 | `full` | `not_gate` (`threshold=0`) and `and_gate` (`threshold=2`). |
| `kaisdukes-mp/src/input.py` & `input_type.py` | 15 | `full` | `EXCITATORY` and `INHIBITORY` input data structures. |
| `kaisdukes-mp/tests/neuron_test.py` | 38 | `full` | Unit tests for Repo 1. |
| `codeforge-mp/README.md` | 70 | `full` | Documentation of Repo 2 rules and limitations. |
| `codeforge-mp/neuron.py` | 39 | `full` | `McCullochPittsNeuron.fire()` and `notGate` with constant `[1]` excitation. |
| `codeforge-mp/network.py` | 55 | `full` | Discrete-time `Network.step()`, 2-layer `xorNetwork()`, and cyclic `memoryNetwork()`. |
| `codeforge-mp/demo.py` | 45 | `full` | CLI demonstration of truth tables, 2-step XOR, and cyclic memory trace. |
| `codeforge-mp/tests/test_mcp.py` | 43 | `full` | Unit tests for Repo 2. |

## 9. Files

### Local worktree (`C:\personal\oss\clover-framework-foundations\clover-kimi-k3\` on branch `foundations-research`)
- Context file: `docs/mcculloch-pitts-1943-context.md`
- Predictions log: `research/mcculloch-pitts-1943/work/predictions.txt`
- Extracted paper text: `research/mcculloch-pitts-1943/work/mp1943.txt`
- Own implementations:
  - `research/mcculloch-pitts-1943/work/mp1943_calculus.py` (Equation 1 simulator, single-neuron enumerator, Theorem 2 & 3 compiler/verifier, negative controls)
  - `research/mcculloch-pitts-1943/work/e3_illusions_and_equivalences.py` (Fig. 1e cutaneous heat-from-cold illusion, Theorems 4–6 Fig. 1f–h equivalence verifiers)
  - `research/mcculloch-pitts-1943/work/e4_cyclic_nets.py` (Theorem 7 Fig. 1i alterable synapse vs circle, Theorem 8 $2^p$ state-conjunction solver, Theorem 10 quantified/modulo-$n$ nets, Section 4 causal irreciprocity preimage counter)
- Scripts:
  - `research/mcculloch-pitts-1943/scripts/e0_setup.sh`
  - `research/mcculloch-pitts-1943/scripts/e1_repos.sh`
  - `research/mcculloch-pitts-1943/scripts/e1b_tests.sh`
  - `research/mcculloch-pitts-1943/scripts/e2_tpe_compiler.sh`
  - `research/mcculloch-pitts-1943/scripts/e3_illusions_and_equivalences.sh`
  - `research/mcculloch-pitts-1943/scripts/e4_cyclic_nets.sh`
- Results:
  - `research/mcculloch-pitts-1943/results/e0_setup.out`
  - `research/mcculloch-pitts-1943/results/e1_repos.out`
  - `research/mcculloch-pitts-1943/results/e1b_tests.out`
  - `research/mcculloch-pitts-1943/results/e2_tpe_compiler.out`
  - `research/mcculloch-pitts-1943/results/e3_illusions_and_equivalences.out`
  - `research/mcculloch-pitts-1943/results/e4_cyclic_nets.out`

### Remote (`gpu` server)
- `/home/usergpu/foundations-study/mcculloch-pitts-1943/` (`repo/kaisdukes-mp`, `repo/codeforge-mp`, `work/`)

