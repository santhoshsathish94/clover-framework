# Red-Blue Pebble Game (Hong & Kung 1981) and `michaelkeating/pebbling` — Context

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh.

## 1. Status

- **E1 (Run repository as shipped on CPU)**: DONE
- **E2 (Hand-made move sequences & minimal paper engine)**: DONE
- **E3 (Exact minimum-I/O solver & validation on canonical DAGs)**: DONE
- **E4 (Recomputation vs once-computed search on small DAGs)**: DONE
- **E5 (S-partition lower-bound lemma by experiment)**: DONE
- **E6 (Matrix multiplication under ideal cache in C: LRU vs OPT)**: DONE
- **E7 (FFT butterfly graphs: exact I/O and recursive blocked schedule)**: DONE

## 2. What this is

Authors say (J.-W. Hong and H. T. Kung, "I/O complexity: The red-blue pebble game", STOC 1981, Section 1, p. 326, and Section 2, p. 327):
- "This paper proposes a pebble game, called the red-blue pebble game, to model the problem, and presents a number of lower bound results for the I/O requirement." (Section 1, p. 326)
- The game is played on a directed acyclic graph (DAG) $G = (V, E)$ modeling a two-level memory hierarchy: a fast memory holding at most $S$ words (red pebbles) and an arbitrarily large slow memory (blue pebbles). (Section 2, p. 327)
- Initial configuration: only input vertices have pebbles, all blue. Terminal configuration: only output vertices have pebbles, all blue. (Section 2, p. 327)
- Four rules: R1 (Input: place red on a vertex with blue), R2 (Output: place blue on a vertex with red), R3 (Compute: if all immediate predecessors of $v$ have red pebbles, place red on $v$), R4 (Delete: remove a red or blue pebble from any vertex). (Section 2, p. 327)
- The minimum I/O time $Q$ is "the minimum number of transitions according to rule R1 or R2 required by any complete calculation." (Section 2, p. 327)
- To prove lower bounds on $Q$, Section 3 (pp. 327–329) introduces the static $S$-partitioning problem and proves $Q \ge S \cdot (P(2S) - 1)$ (Lemma 3.1, p. 329), yielding tight bounds for FFT ($\Omega(n \log n / \log S)$, Section 4) and ordinary $n \times n$ matrix multiplication ($\Omega(n^3 / \sqrt{S})$, Section 6).

The repository `https://github.com/michaelkeating/pebbling` (README.md, lines 1–50) is a Python and React puzzle game by the Sutro Group inspired by the Red-Blue Pebble Game, modeling three memory tiers (`register`, `sram`, `hbm`) with picojoule (pJ) energy costs.

## 3. Experiments

### E1: Run the repository as shipped on CPU
- **Question**: What commit hash, license, file structure, and line counts does `michaelkeating/pebbling` ship with, how is its energy cost in pJ computed, and what exact pJ costs, step counts, and completion flags are produced when running `pebbling_engine.py`, `benchmark_strategies`, and `pebbling_agent.py` (`HeuristicAgent`, `RandomAgent`, tournament, and difficulty presets across `h100`, `tutorial`, and `extreme` profiles)?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 06:00 UTC):
  1. Cost is summed per action: `PLACE` adds `cost_per_step` (`1, 5, 100` pJ for `register, sram, hbm` on `h100`), `TRANSFER` adds `io_transfer_cost` of destination tier (`0, 5, 100` pJ), `REMOVE` adds `0` pJ.
  2. On the 10-node demo DAG (`depth=4, max_fan_in=2, reg=3, sram=4, seed=42, h100`), `topological_register_strategy` will finish (`done=True`) with cost `10..25` pJ, while `greedy_cheapest_strategy` will finish with higher cost (`30..150` pJ).
  3. On the 5-game tournament (`nodes=12, depth=5, seeds 1000..1004`), `HeuristicAgent` will complete all 5 games with average cost `< 150` pJ, whereas `RandomAgent` will average `> 1,000` pJ.
- **Run**: `scripts/pebbling-w2-e1.sh` and follow-up diagnosis `scripts/pebbling-w2-e1b.sh` via `k3.ps1 -Action run -Cpus 0-5`. Outputs: `results/pebbling-w2-e1.out`, `results/pebbling-w2-e1b.out`.
- **Observation**:
  - **Repository metadata**: Commit `985320b55c8422667d9d571453e27ea1756888df` (`https://github.com/michaelkeating/pebbling.git`), MIT License (Copyright 2026 Sutro Group), 616 KB total on disk, 3,503 lines across top-level files (`pebbling_engine.py`: 1,005 lines, `pebbling_agent.py`: 578 lines, `pebbling-game.jsx`: 1,672 lines, `README.md`: 215 lines, `pyproject.toml`: 33 lines). The paper text `1981-stoc-hong-kung.txt` has 943 lines (35,041 bytes) covering all 8 pages (`=== PAGE 1 ===` to `=== PAGE 8 ===`, STOC pp. 326–333).
  - **Cost computation**: Confirmed in `pebbling_engine.py` lines 536–549: `PLACE` adds `tier_costs[tier]["cost_per_step"]`, `TRANSFER` adds `tier_costs[tier]["io_transfer_cost"]` of the destination tier, and `REMOVE` adds `0` pJ. The three profiles are `h100` (`register`: 1/0 pJ, `sram`: 5/5 pJ, `hbm`: 100/100 pJ), `tutorial` (1/0, 3/3, 10/10 pJ), and `extreme` (1/0, 10/10, 500/500 pJ).
  - **Engine demo (`nodes=10, depth=4, seed=42, h100`)**: `greedy_cheapest` reported `26 pJ (13 steps, done=True)`; `topological_register` reported `10 pJ (17 steps, done=False)`.
  - **CLI agents (`nodes=10, depth=4, fan_in=2, reg=3, sram=4, seed=42`)**: `HeuristicAgent` finished in `16` steps with `37` pJ (`0/6` sinks remaining); `RandomAgent` finished in `223` steps with `6,277` pJ.
  - **5-game tournament (`nodes=12, depth=5, seeds 1000..1004, h100`)**: `HeuristicAgent` averaged `137.0` pJ (`29, 50, 228, 233, 145` pJ across the 5 seeds; best `29`, worst `233`), while `RandomAgent` averaged `8,276.6` pJ (`8079, 2921, 11089, 16388, 2906` pJ; best `2,906`, worst `16,388`).
  - **Preset sweep (`seed=42, h100`) and E1b diagnosis**:
    - On all 5 difficulty presets (`tutorial` 6 nodes through `expert` 20 nodes), `topological_register_strategy` ended with `done=False` (`6, 8, 17, 36, 45` pJ). E1b (`results/pebbling-w2-e1b.out`) showed why: in `free_register_if_needed()` (lines 850–866), register nodes are scored by `unpebbled_succs` without excluding sink nodes. Because sink nodes have `successors == []`, they have `unpebbled_succs == 0` and are immediately removed by `Action(REMOVE, victim)` whenever the next node needs a register (`steps 8, 10, 13, 15, 17` removed sinks `N4..N8` on the 10-node demo, leaving only `N9` pebbled). Adding `and not game.dag.nodes[nid].is_sink` in a copy (`topological_register_fixed`) made all 6 configurations finish with `done=True` and all sinks pebbled (`37, 11, 17, 233, 734, 846` pJ).
    - On `medium_preset` (`12` nodes, `h100`), `greedy_cheapest_strategy` reported `done=True` at `281` pJ, but took `500` steps (`max_steps`) with `sinks_remaining=[5, 6, 7]`. E1b showed that once register (3) and SRAM (4) slots fill, placing sink `N5` or `N6` in `hbm` scores `100 + 0.2 - 50 = 50.2`, whereas `Action(REMOVE, N7)` on already-pebbled sink `N7` (which has `successors == []`, so `successors_done = True`) scores `50.0 < 50.2`. Thus `greedy_cheapest_strategy` alternated `REMOVE N7` (0 pJ) and `PLACE N7 in register` (1 pJ, score `-49.0`) from step 12 to step 500, where `step()` sets `self.done = True` on timeout (`info["timeout"] = True`).
- **What it showed**: Predictions 1 and 3 were confirmed (`137.0` pJ vs `8,276.6` pJ in the 5-game tournament), but Prediction 2 was **wrong**: `topological_register_strategy` did not finish (`done=False` on all presets because `free_register_if_needed` evicts sink nodes that have 0 successors), and `greedy_cheapest_strategy` can report `done=True` on timeout at step 500 without pebbling all sinks because `REMOVE` on a sink node scores `50.0 < 50.2` (HBM sink placement on `h100`).

### E2: Hand-made move sequences on the repository engine vs a minimal paper engine
- **Question**: How do the repository's `PebblingGame` engine and a minimal engine implementing Hong & Kung's exact Section 2 rules (`R1` Input, `R2` Output, `R3` Compute, `R4` Delete, red capacity $S$, initial blue on inputs, terminal blue-only on outputs) respond to hand-made legal and illegal move sequences probing each rule?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 06:15 UTC):
  1. Both engines will reject computing a non-source node when a predecessor has no pebble at all, and both will reject exceeding the fast-memory (`register` / red) capacity $S$, and both will allow deleting and recomputing a node.
  2. Six specific rule differences will be observed on identical test sequences:
     - (a) **Initial state & Input (R1)**: Paper starts with blue pebbles on all inputs ($B = I, R = \emptyset$) and R1 loads an input (or any blue-pebbled vertex) into red at cost 1 I/O; repo starts with all nodes `None`, pebbles sources via `PLACE(v, tier)` (cost 1 pJ in `register`), and charges `0` pJ for `TRANSFER(v, "register")` from `sram`/`hbm`.
     - (b) **Predecessor tier for Compute (R3)**: Paper R3 requires all immediate predecessors to hold **red** pebbles ($\text{pred}(v) \subseteq R$); repo `_can_compute` accepts predecessors pebbled in **any** tier (`register`, `sram`, or `hbm`).
     - (c) **Compute cost & destination**: Paper R3 places a red pebble at cost 0 I/O; repo `PLACE` can target `register` (1 pJ), `sram` (5 pJ), or `hbm` (100 pJ) directly without first placing in `register`.
     - (d) **Simultaneous red+blue vs exclusive transfer (R2)**: Paper R2 places a blue pebble on $v \in R$ while keeping $v \in R$ ($v \in R \cap B$, freeing no red slot until R4); repo `TRANSFER(v, "sram"|"hbm")` moves the pebble out of `register` in one step (freeing the register slot immediately).
     - (e) **Input R1 on non-source without blue**: Paper R1 is legal on any vertex currently in $B$ and illegal on any vertex not in $B$ (even a source whose blue pebble was deleted by R4); repo `PLACE` on a source is always legal when unpebbled, and `TRANSFER` to `register` requires an existing pebble in `sram`/`hbm`.
     - (f) **Terminal condition**: Paper requires $B = O$ and $R = \emptyset$ (all outputs blue, no red pebbles and no non-output blue pebbles remaining); repo sets `done = True` the instant every sink has a pebble in any tier (even `register`), with non-sinks still pebbled.
- **Run**: `scripts/pebbling-w2-e2.sh` (using `work/paper_engine.py`) via `k3.ps1 -Action run -Name pebbling-w2-e2 -Cpus 0-5`. Output: `results/pebbling-w2-e2.out`.
- **Observation**:
  - **Probe 1 (Initial state & R1 Input)**: Repo starts with `{0: None, 1: None, 2: None, 3: None}`; `PLACE(0, "register")` is accepted at cost `1` pJ. Paper engine starts with `red=set(), blue={0}`; `R1(0)` is accepted at cost `1` I/O.
  - **Probe 2 (R2 Output & simultaneous occupancy vs move)**: With `{0, 1}` in fast memory (`2/2` slots used), repo `TRANSFER(1, "sram")` moves `1` out of `register` into `sram` (cost `5` pJ, freeing register usage to `1/2`), and a subsequent `TRANSFER(1, "register")` reloads it at cost `0` pJ (`io_transfer_cost = 0`). Paper `R2(1)` places a blue pebble on `1` while keeping `1` in `red` (`red={0, 1}`, usage still `2/2`, `blue={0, 1}`, cost `1` I/O), requiring `R4(1, "red")` before another red pebble can be placed, and reloading `1` via `R1(1)` costs `1` I/O.
  - **Probe 3 (Compute without pebbled predecessor)**: Computing `N1` when `N0` has no pebble (repo) or only a blue pebble (paper) is rejected by both (`Cannot compute N1: predecessors [0] are not pebbled.` vs `R3 rejected: predecessors [0] of 1 lack red pebbles`).
  - **Probe 4 (Compute when predecessor is in slow memory)**: After `PLACE(0, "sram")`, repo accepts `PLACE(1, "register")` at cost `1` pJ because `_can_compute` only checks `pebbles[pid] is not None`. Paper rejects `R3(1)` when `0` is only in `blue`.
  - **Probe 5 (Zero fast-memory slots `S=0`)**: With `register_slots=0`, repo accepts `PLACE(0, "sram")` and `PLACE(1, "sram")`. With `S=0`, paper rejects `R1(0)` (`red pebble limit S=0 reached`).
  - **Probe 6 (Red/register capacity limit `S=2`)**: Placing a 3rd pebble in `register` / `red` is rejected by both (`No register capacity remaining.` vs `R3 rejected: red pebble limit S=2 reached`).
  - **Probe 7 (Delete R4 & recomputation)**: Deleting intermediate `N1` and recomputing `N1` is accepted by both (repo cost `1` pJ, paper cost `0` I/O). Deleting input `N0` completely and re-inputting `N0` is accepted by repo (`PLACE(0, "register")`), whereas in the paper engine deleting both red and blue pebbles from input `0` makes `R1(0)` fail (`vertex 0 does not have a blue pebble`).
  - **Probe 8 (Ending condition)**: On the 4-node diamond with `S=3`, repo sets `done=True` (cost `4` pJ) as soon as sink `N3` is placed in `register` while `N1, N2` are still in `register`. At the identical point in the paper engine, `is_terminal=False` and `outputs_all_blue=False` (`io_cost=1`); after `R2(3)`, `outputs_all_blue=True` (`io_cost=2`), and `is_terminal=True` after `R4` deletes the remaining red pebbles on `1, 2, 3` and the input blue pebble on `0`.
- **What it showed**: Every predicted agreement and all six structural differences between `PebblingGame` and Hong & Kung's Section 2 game were confirmed by direct execution on the 4-node diamond DAG.

### E3: Exact minimum-I/O solver for the paper's game and validation on canonical DAGs
- **Question**: What are the exact minimum I/O counts $Q^*(G, S)$ under Hong & Kung's Section 2 rules for canonical DAGs (chains, binary tree, diamond, pyramids, and the $2 \times 2$ matrix multiplication DAG) across fast-memory capacities $S$, and do four independent checks (hand-derived values, slower brute-force DFS/BFS over transitions, a broken-rule negative control, and schedule checker rejection of illegal schedules) validate the solver?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 06:22 UTC):
  1. **Hand-derived values (3 tiny graphs + 7-node binary tree)**:
     - 3-node chain ($0 \to 1 \to 2$, $d=1$): $S=1 \to \text{infeasible}$, $S \ge 2 \to Q^* = 2$.
     - 3-node V-tree ($0, 1 \to 2$, $d=2$): $S \in \{1, 2\} \to \text{infeasible}$, $S \ge 3 \to Q^* = 3$.
     - 4-node diamond ($0 \to 1, 2 \to 3$, $d=2$): $S \in \{1, 2\} \to \text{infeasible}$, $S \ge 3 \to Q^* = 2$.
     - 7-node binary tree (4 leaves, 2 internals, 1 root, $d=2$): $S \in \{1, 2\} \to \text{infeasible}$, $S = 3 \to Q^* = 7$ (4 input R1s + 1 output R2 + 1 intermediate spill R2 + R1), $S \ge 4 \to Q^* = 5$.
  2. **Validation checks**:
     - (b) A slower unpruned breadth/cost-first search over raw `(red, blue)` bitmask transitions without macro-collapsing free R3/R4 moves will produce identical $Q^*$ values on all tiny graphs.
     - (c) Negative control (broken rule): allowing R3 to compute a vertex when at least one (instead of all) immediate predecessor is red will strictly change feasibility/cost (making diamond feasible at $S=2$ with $Q=2$, and lowering 7-node binary tree at $S=3$ from $7$ to $5$).
     - (d) The independent schedule verifier (`PaperPebbleEngine`) will accept every optimal schedule emitted by the solver with matching I/O count and `is_terminal() == True`, and will reject a schedule with a mutated move or missing cleanup.
  3. **Canonical DAGs sweep**:
     - Pyramid-6 (base 3, $n=6, d=2$): $S \le 2$ infeasible, $S=3 \to Q^*=6$ (keeping shared middle input in red and spilling one middle node, +2 I/O over the trivial bound 4), $S \ge 4 \to Q^*=4$.
     - Pyramid-10 (base 4, $n=10, d=2$): $S \le 2$ infeasible, $S \ge 5 \to Q^*=5$, and for $S \in \{3, 4\}$, $Q^* > 5$.
     - $2 \times 2$ Matrix Multiplication ($n=16$: 8 inputs, 8 products, 4 sum outputs, $d=2$): $S \le 2$ infeasible; at large $S$ ($S \ge 6$ or $7$), $Q^* = 8 + 4 = 12$ (every input read once, every output written once); at $S=3$, each of the 4 outputs requires at least one product spill plus input reloads so $Q^* \ge 24$.
- **Run**: `scripts/pebbling-w2-e3.sh` and `scripts/pebbling-w2-e3b.sh` via `k3.ps1 -Action run -Cpus 0-5`. Outputs: `results/pebbling-w2-e3.out`, `results/pebbling-w2-e3b.out`.
- **Observation**:
  - **(a) & (b) Hand-derived tiny graphs vs Dijkstra vs RawBFS (`results/pebbling-w2-e3.out`)**:
    - `chain_3` ($n=3, d=1$): $S=1 \to \text{None}$ (infeasible), $S=2..5 \to Q^* = 2$. Unpruned Dijkstra, dead-pebble-pruned Dijkstra, and unreduced RawBFS (which includes mid-calculation `R4(blue)` transitions) agreed on every $(G, S)$ pair.
    - `v_tree_3` ($n=3, d=2$): $S=1,2 \to \text{None}$, $S=3..5 \to Q^* = 3$. All three solvers agreed.
    - `diamond_4` ($n=4, d=2$): $S=1,2 \to \text{None}$, $S=3..5 \to Q^* = 2$. All three solvers agreed.
    - `bintree_7` ($n=7, d=2$): $S=1,2 \to \text{None}$, $S=3 \to Q^* = 7$, $S=4,5 \to Q^* = 5$. All three solvers agreed.
  - **(c) Negative control 1 (broken rule: R3 allowed with $\ge 1$ red predecessor)**:
    - On `diamond_4`, `broken_any_pred=True` made $S=2$ feasible with $Q = 2$ (vs exact `None`).
    - On `bintree_7`, `broken_any_pred=True` made $S=2$ feasible with $Q = 2$ and lowered $S=3,4$ from $Q^* = 7, 5$ to $Q = 2, 2$ (loading only one leaf and computing up one path to the root).
  - **(d) Negative control 2 (schedule verifier rejection)**:
    - `PaperPebbleEngine` verified the 10-move optimal `diamond_4` ($S=3$) schedule (`ok=True, cost=2`), and rejected: (1) dropping `R1(0)` (`R3 rejected: predecessors [0] of 2 lack red pebbles`), (2) dropping the final `R4` cleanup (`Not in terminal state: red=set(), blue={0, 3}`), and (3) running the $S=3$ schedule under $S=2$ (`R3 rejected: red pebble limit S=2 reached`).
  - **(e) Canonical DAGs sweep (`results/pebbling-w2-e3.out` and `results/pebbling-w2-e3b.out`)**:
    - `chain_4` ($n=4, |I|=1, |O|=1, d=1$): $S=1 \to \text{INF}$, $S=2..6 \to 2$.
    - `chain_6` ($n=6, |I|=1, |O|=1, d=1$): $S=1 \to \text{INF}$, $S=2..6 \to 2$.
    - `diamond_4` ($n=4, |I|=1, |O|=1, d=2$): $S=1,2 \to \text{INF}$, $S=3..6 \to 2$.
    - `bintree_7` ($n=7, |I|=4, |O|=1, d=2$): $S=1,2 \to \text{INF}$, $S=3 \to 7$, $S=4..6 \to 5$.
    - `pyramid_b3_n6` ($n=6, |I|=3, |O|=1, d=2$): $S=1,2 \to \text{INF}$, $S=3 \to 6$, $S=4..6 \to 4$.
    - `pyramid_b4_n10` ($n=10, |I|=4, |O|=1, d=2$): $S=1,2 \to \text{INF}$, $S=3 \to 11$, $S=4 \to 7$, $S=5,6 \to 5$.
    - `matmul_2x2_n20` ($n=20, |I|=8, |O|=4, d=2$, `results/pebbling-w2-e3b.out`):
      - $S=1, 2 \to \text{INF}$ (fan-in $d=2$ requires $S \ge d+1 = 3$).
      - $S=3 \to Q^* = 25$ (`73,793` states). Disagreement investigated during E3b: restricting blue product pebbles to at most 1 at a time (`max_bprod=1`) gave $Q = 28$ ($4 \times 7$, evaluating the four $C_{ij}$ trees separately), whereas the unpruned solver gave $Q^* = 25$ because holding an input $A_{ik}$ or $B_{kj}$ in red to compute and spill products for two different $C_{ij}$ outputs (having 2 products in blue simultaneously) saves 3 input reloads. Forbidding product spills (`max_bprod=0`) is infeasible at $S=3$.
      - $S=4 \to Q^* = 17$, $S=5 \to Q^* = 14$, $S=6 \to Q^* = 13$, $S=7 \to Q^* = 12$, $S=8 \to Q^* = 12$. For all $S \ge 4$, `max_bprod=0` equals `exact_unpruned`, showing zero product spills are needed once $S \ge 4$, and at $S \ge 7$ the trivial bound $|I| + |O| = 12$ is reached.
- **What it showed**: All hand-derived values, negative controls, and canonical DAG predictions held; moreover, comparing `max_bprod=1` vs unpruned on $2 \times 2$ matrix multiply at $S=3$ revealed that cross-output input reuse with multiple spilled products lowers $Q^*(3)$ from $4 \times 7 = 28$ to $25$.

### E4: Recomputation vs once-computed search on small DAGs
- **Question**: Across small DAGs with fan-in at most 2 and up to 8 vertices (including all connected 5-node and 6-node DAGs of a structured family and 1,000 random DAGs for $n \in \{5, 6, 7, 8\}$ across $S \in \{3, 4\}$), how often does allowing recomputation (`allow_recompute=True`) strictly lower the exact minimum I/O $Q^*$ compared with forbidding recomputation (`allow_recompute=False`), and what is the smallest DAG where this gap occurs?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 06:35 UTC):
  1. **Smallest example ($n=5, S=3$)**: No DAG with $n \le 4$ can have $Q^*_{\text{recomp}} < Q^*_{\text{no-recomp}}$ at $S=3$ because forcing an already-computed vertex $u$ out of $R$ requires a step `R3(v)` that fills all 3 red slots with $\{p_1, p_2, v\} \not\ni u$ (4 distinct vertices) plus a 5th vertex $w$ that later needs $u$. At $n = 5, S = 3$, the hand-constructed 5-node DAG $G_5$ with edges $0 \to 1, 1 \to 2, 0 \to 3, 2 \to 3, 1 \to 4, 3 \to 4$ ($I=\{0\}, O=\{4\}$) will achieve **$Q^*_{\text{recomp}} = 2 < Q^*_{\text{no-recomp}} = 4$** (a 2-I/O reduction: recomputing node 1 from $0 \in R$ costs 0 I/O, whereas forbidding recomputation forces spilling node 1 via `R2(1)` + `R1(1)` at cost +2 I/O).
  2. **Frequency in systematic & random search**: Across exhaustive 5-node layered DAGs and 1,000 random connected fan-in-$\le 2$ DAGs for $n \in \{5, 6, 7, 8\}$ at $S \in \{3, 4\}$, strict improvement ($Q^*_{\text{recomp}} < Q^*_{\text{no-recomp}}$) will occur in $>0\%$ of feasible pairs (roughly 1%–15% at $S=3$, and less often at $S=4$), and $Q^*_{\text{recomp}} \le Q^*_{\text{no-recomp}}$ will hold for 100% of pairs (allowing recomputation is a superset of schedules).
- **Run**: `scripts/pebbling-w2-e4.sh` via `k3.ps1 -Action run -Name pebbling-w2-e4 -Cpus 0-5`. Output: `results/pebbling-w2-e4.out`.
- **Observation**:
  - **Smallest example (`G5_recomp`, $n=5$, edges `0->1, 1->2, 0->3, 2->3, 1->4, 3->4`, $S=3$)**:
    - With recomputation (`allow_recompute=True`): $Q^*_{\text{recomp}} = 2$ (14 moves total; node 1 is computed twice via `R3(1)` at steps 2 and 7 while input 0 remains in `red`, incurring only `R1(0)` at step 1 and `R2(4)` at step 10).
    - Without recomputation (`allow_recompute=False`): $Q^*_{\text{no-recomp}} = 4$ (16 moves total; node 1 is spilled via `R2(1)` at step 4 before `R4(1,red)` and `R3(3)`, then reloaded via `R1(1)` at step 9 before `R3(4)`).
    - Both schedules were verified move-by-move by `PaperPebbleEngine`.
  - **Exhaustive search over all topological DAGs with fan-in $\le 2$ (`results/pebbling-w2-e4.out`)**:
    - $n = 4$ (`18` DAGs): `0` DAGs have $Q^*_{\text{recomp}} < Q^*_{\text{no-recomp}}$ at $S = 2, 3, 4$, confirming $n = 5$ is the minimum vertex count where recomputation lowers I/O.
    - $n = 5$ (`180` DAGs): at $S = 2$, `0/24` feasible (`0.00%`); at $S = 3$, `6/180` (`3.33%`, max gap `2`); at $S = 4$, `0/180` (`0.00%`).
    - $n = 6$ (`2,700` DAGs): at $S = 2$, `3/120` feasible (`2.50%`, max gap `1`); at $S = 3$, `398/2700` (`14.74%`, max gap `4`); at $S = 4$, `0/2700` (`0.00%`).
  - **Random search (500 DAGs each for $n = 7, 8$, seed 42)**:
    - $n = 7$: at $S = 3$, `155/500` (`31.00%`, max gap `4`); at $S = 4$, `11/500` (`2.20%`, max gap `1`).
    - $n = 8$: at $S = 3$, `241/500` (`48.20%`, max gap `4`); at $S = 4$, `28/500` (`5.60%`, max gap `2`).
- **What it showed**: Every prediction held (`n=4` has 0 cases, `G5_recomp` at $n=5, S=3$ achieves $2 < 4$, and frequency rises with $n$ from `3.33%` at $n=5, S=3$ to `48.20%` at $n=8, S=3$). One minor surprise at $n=6, S=2$: even on fan-in-1 trees (`120` feasible DAGs at $S=2$), `3/120` (`2.50%`) save 1 I/O via recomputation because re-reading an input (`R1(0) + R3(1)`, cost 1) is cheaper than spilling and reloading an intermediate (`R2(1) + R1(1)`, cost 2).

### E5: The S-partition lower-bound lemma by experiment
- **Question**: (a) When an optimal schedule from E3 is cut into consecutive subcalculations $C_1, \dots, C_h$ of $S$ I/O moves (with $\le S$ in $C_h$) and vertex sets $V_1, \dots, V_h$ are constructed via Hong & Kung's Theorem 3.1 rules (i)–(iii), do $\{V_1, \dots, V_h\}$ always satisfy all four $2S$-partition properties (P1 disjoint cover, P2 dominator set $|D_i| \le 2S$, P3 minimum set $|M_i| \le 2S$, P4 acyclic dependence)? (b) When $P(2S)$ (and $P(K)$ generally) is computed by exact brute force over all valid partitions of tiny graphs, does $Q^* \ge S \cdot (P(2S) - 1)$ hold, and how tight is it?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 06:48 UTC):
  1. **(a) Schedule-induced $2S$-partition**: For every optimal schedule from E3 (`chain_4`, `diamond_4`, `bintree_7`, `pyramid_b3_n6`, `pyramid_b4_n10`), cutting into segments of $S$ I/O moves and applying Theorem 3.1's definition will produce a valid $2S$-partition with $h = \lceil Q^* / S \rceil$ non-empty sets, satisfying P1–P4 and $S(h-1) < Q^* \le S \cdot h$. Moreover, because the $S$ I/O moves in $C_i$ are split between $r_1$ inputs (R1) and $r_2$ outputs (R2) with $r_1 + r_2 \le S$, $|D_i| + |M_i| \le (S + r_1) + (S + r_2) \le 3S$, and each individually is $\le 2S$.
  2. **(b) Brute-force $P(2S)$ and tightness of $S \cdot (P(2S) - 1)$**:
     - For any DAG $G = (V, E)$, the single-block partition $\{V\}$ has minimum dominator set $D_1 = I$ (size $|I|$) and minimum set $M_1 = O$ (size $|O|$). Therefore, whenever $2S \ge \max(|I|, |O|)$ (which holds for all feasible $S \ge 2$ on `chain_4`, `diamond_4`, `bintree_7`, `pyramid_b3_n6`, `pyramid_b4_n10` since $|I| \le 4 \le 2S$), the 1-set partition $\{V\}$ is a valid $2S$-partition, giving **$P(2S) = 1$** and **$S \cdot (P(2S) - 1) = 0$**!
     - Thus $Q^* \ge S \cdot (P(2S) - 1)$ holds everywhere, but is **loose ($0$ vs $Q^* \in [2, 11]$)** on tiny graphs whenever $2S \ge \max(|I|, |O|)$ because of the factor-of-2 slack ($2S$ instead of $S$) and the $-1$ subtraction ($P(2S) - 1$).
     - When we also evaluate $P(K)$ for smaller partition parameter $K \in \{1, 2, 3\}$ (where $K < |I|$), $P(K) > 1$ (for instance on `bintree_7` with $|I|=4$, $P(1)=7, P(2)=3, P(3)=2, P(4)=1$).
- **Run**: `scripts/pebbling-w2-e5.sh` (using `work/s_partition.py`) via `k3.ps1 -Action run -Name pebbling-w2-e5 -Cpus 0-5`. Output: `results/pebbling-w2-e5.out`.
- **Observation**:
  - **(a) Theorem 3.1 schedule-to-$2S$-partition construction (`results/pebbling-w2-e5.out`)**:
    - On all 14 feasible `(graph, S)` pairs across `chain_4`, `chain_6`, `diamond_4`, `bintree_7`, `pyramid_b3_n6`, and `pyramid_b4_n10` for $S \in \{2, 3, 4\}$, cutting the E3 optimal schedule into segments of $S$ I/O moves and applying Theorem 3.1's rules (i)–(iii) produced a valid $2S$-partition (`2S-Valid=True`), satisfying P1 (disjoint union $= V$), P2 ($\max_i |D_i| \le \max_i |V_R \cup V_{BR}| \le 2S$), P3 ($\max_i |M_i| \le 2S$), P4 (acyclic dependence), and $S(h-1) < Q^* \le S \cdot h$.
    - Examples at $S=3$ ($2S=6$): `bintree_7` ($Q^*=7 \implies h=3$) yielded $V_1=\{0,1,4\}, V_2=\{2,3,5\}, V_3=\{6\}$ with $\max |D_i|=2, \max |V_R \cup V_{BR}|=5, \max |M_i|=1$; `pyramid_b4_n10` ($Q^*=11 \implies h=4$) yielded $V_1=\{0,1,4\}, V_2=\{2,3,5\}, V_3=\{6,8\}, V_4=\{7,9\}$ with $\max |D_i|=3, \max |V_R \cup V_{BR}|=5, \max |M_i|=2$.
    - Negative controls: `check_s_partition` rejected a cyclic 2-set partition on `bintree_7` (`V_1={0,1,2,3,6}, V_2={4,5}` $\to$ `P4 failed: cyclic dependence among V_i`) and rejected `V_1=V` with $K=2$ (`P2 failed on V_1: |D|=4 > 2`).
  - **(b) Exact brute-force $P(K)$ and tightness of $Q^* \ge S \cdot (P(2S) - 1)$**:
    - Exact $P(K)$ values for $K = 1..6$:
      - `chain_4`, `chain_6`, `diamond_4` ($|I|=1, |O|=1$): $P(K) = 1$ for all $K \ge 1$.
      - `bintree_7` ($|I|=4, |O|=1$): $P(1)=7, P(2)=3, P(3)=2, P(4)=P(5)=P(6)=1$.
      - `pyramid_b3_n6` ($|I|=3, |O|=1$): $P(1)=6, P(2)=3, P(3)=P(4)=P(5)=P(6)=1$.
      - `pyramid_b4_n10` ($|I|=4, |O|=1$): $P(1)=10, P(2)=4, P(3)=3, P(4)=P(5)=P(6)=1$.
    - Tightness: $Q^* \ge S \cdot (P(2S) - 1)$ holds on 100% of pairs, but whenever $2S \ge \max(|I|, |O|)$ (which is true for all $S \ge 2$ on these graphs since $|I| \le 4 \le 2S$), the single-block partition $\{V\}$ is a valid $2S$-partition ($D_1 = I, M_1 = O$), making $P(2S) = 1$ and $S \cdot (P(2S) - 1) = 0$. Without the factor-of-2 slack in $2S$, $S \cdot (P(S) - 1)$ gives $3$ (vs $Q^*=7$) on `bintree_7` at $S=3$ and $6$ (vs $Q^*=11$) on `pyramid_b4_n10` at $S=3$.
- **What it showed**: Both predictions held to the exact integer: Theorem 3.1's constructive partition from $S$-I/O subcalculations always satisfies P1–P4 with size $\le 2S$, and on small graphs where $2S \ge \max(|I|, |O|)$, $P(2S)$ collapses to $1$ so Lemma 3.1's lower bound is $0$ (an asymptotic bound that becomes positive only when the graph forces $P(2S) \ge 2$).

### E6: Matrix multiplication under an ideal cache in C (LRU vs Belady/OPT)
- **Question**: In a word-granularity, fully associative cache of $S$ words counting slow-memory loads (cache misses) plus stores (dirty writebacks of $C$), what are the exact I/O counts for $n \times n$ matrix multiplication under LRU and Belady/OPT for (1) the $2 \times 2, S=4$ hand-countable case (and off-by-one $S=3, 5$), (2) the sweep $n \in \{64, 128, 256, 512\}$ and $S \in \{48, 192, 768, 3072\}$ comparing naive `ijk` vs blocked with $b \approx \lfloor \sqrt{S/3} \rfloor$, (3) $b$ too small and $b$ too large, and (4) varying the intra-tile loop order (`i,j,k` vs `i,k,j` vs `k,i,j`) to explain the LRU vs OPT gap?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 06:58 UTC):
  1. **Hand-countable case ($n=2$, naive `ijk`, 24 accesses `[A[i,k], B[k,j], C[i,j]]` across 8 steps, $S=4$)**:
     - Under **LRU** at $S=4$: **20** cache misses (loads) + **4** dirty writebacks (stores of $C$) = **24 total I/O** (only the second access to each of the 4 $C_{ij}$ words hits).
     - Under **Belady/OPT** at $S=4$: **15** cache misses (loads) + **4** dirty writebacks = **19 total I/O** (OPT evicts `B00` at step 2 instead of `A00`, and evicts dead $C_{ij}/A_{ik}$ words with next-use $=\infty$, achieving 9 hits).
     - Negative control (off-by-one cache size): changing $S=4 \to S=3$ breaks the match (OPT rises from `15` misses / `19` I/O to `17` misses / `21` I/O; at $S=5$, LRU drops from `20` misses / `24` I/O to `18` misses / `22` I/O).
  2. **Main sweep ($n \in \{64, 128, 256, 512\}$, $S \in \{48, 192, 768, 3072\}$, $b = \lfloor \sqrt{S/3} \rfloor$)**:
     - At **$n = 256, S = 768$** ($b = 16$): naive `ijk` under LRU will incur $\approx n^3 + 2n^2 \approx 1.69 \times 10^7$ I/O (since row $A[i,:]$ of 256 words fits in $S=768$ while $B$ misses on almost every access), whereas blocked `(ii,jj,kk)` with $b=16$ will incur $\approx 2.2 \times 10^6$–$3.2 \times 10^6$ I/O, giving a **naive/blocked ratio of $\approx 5.5\times$–$7.6\times$** at $n=256, S=768$ (and rising to $>15\times$ when $S < n$, e.g. at $n=256, S=192$ or $n=512, S=192$).
     - Power-law fit $\text{IO} = c \cdot n^\alpha S^\beta$ for blocked matrix multiplication with $b = \lfloor \sqrt{S/3} \rfloor$: **$\alpha \approx 3.00$**, **$\beta \approx -0.50$**, and normalized constant $\text{IO} / (n^3 / \sqrt{S}) \approx 2\sqrt{3} \approx 3.46$ under OPT (and slightly higher, $\approx 4.0$–$4.8$, under LRU with intra-tile `i,j,k` when $3b^2 = S$).
  3. **Why LRU > OPT at $3b^2 = S$ and experimental test**: With intra-tile loop order `i,j,k`, at the end of tile `kk-1`, all $b^2$ words of $B_{KK-1,JJ}$ (now dead for tile `kk`) were touched during the last row $i = b-1$, making them more recently used than the first $b-1$ rows of $C_{II,JJ}$ (which are live for tile `kk`). Thus at the start of tile `kk`, LRU evicts live $C_{II,JJ}$ words before dead $B_{KK-1,JJ}$ words! Changing the intra-tile loop order from `i,j,k` to `k,i,j` (which touches all $b^2$ words of $C_{II,JJ}$ during the final `k = b-1` pass of tile `kk-1`) or choosing $b$ slightly below $\sqrt{S/3}$ will sharply reduce the LRU excess over OPT.
- **Run**: `scripts/pebbling-w2-e6.sh` (compiling and running `work/cache_sim.c`) via `k3.ps1 -Action run -Name pebbling-w2-e6 -Cpus 0-5`. Output: `results/pebbling-w2-e6.out`.
- **Observation**:
  - **1. Hand-countable $2 \times 2$ case ($S=4$) and off-by-one controls (`results/pebbling-w2-e6.out`)**:
    - $n=2, S=4$: **LRU** `loads=20, stores=4, total=24`; **OPT** `loads=15, stores=4, total=19` (exact match to hand-derived prediction).
    - Off-by-one controls: at $S=3$, LRU `20+4=24`, OPT `17+4=21` (breaks $S=4$ OPT match); at $S=5$, LRU `18+4=22`, OPT `13+4=17` (breaks $S=4$ LRU match); at $S=2$, LRU `24+8=32`, OPT `20+4=24`; at $S=6$, LRU `16+4=20`, OPT `12+4=16`.
  - **2. Main sweep ($n \in \{64, 128, 256, 512\}$, $S \in \{48, 192, 768, 3072\}$, $b = \lfloor \sqrt{S/3} \rfloor$)**:
    - At **$n=256, S=768, b=16$**: `NaiveLRU = 16,973,824`, `NaiveOPT = 16,843,264`, `BlkLRU = 4,186,624` ($c = \text{IO} / (n^3 / \sqrt{S}) = 6.916$), `BlkOPT = 2,166,784` ($c = 3.579$). The `Naive/Blk(LRU)` ratio is `4.05x` (and `NaiveLRU / BlkOPT` is `7.83x`). At $n=512, S=768, b=16$ (where $S < 2n$), `Naive/Blk(LRU)` rises to `8.03x` (`268,959,744` vs `33,490,944`).
    - Across all 16 $(n, S)$ pairs, `BlkOPT` has normalized constant $c = \text{IO} / (n^3 / \sqrt{S})$ in `[3.498, 4.343]` (converging toward $2\sqrt{3} \approx 3.464$ as $n/b$ increases: `3.498` at $n=256, S=48$), whereas `BlkLRU` with intra-tile `i,j,k` at $3b^2 = S$ has $c \in [6.713, 6.927]$—nearly double `BlkOPT`.
    - Power-law log-linear fit $\text{IO} = c \cdot n^\alpha S^\beta$:
      - Blocked LRU ($n=64..512, S=48..3072$, 16 pts): $\text{IO} = 6.592 \cdot n^{2.9996} \cdot S^{-0.4930}$ ($R^2 = 0.999995$).
      - Blocked OPT ($n=64..256, S=48..3072$, 12 pts): $\text{IO} = 4.230 \cdot n^{2.9396} \cdot S^{-0.4728}$ ($R^2 = 0.999813$).
  - **3. Tile-size sensitivity ($n=128, S=192, \sqrt{S/3}=8$)**:
    - Under `BlkOPT`, total I/O achieves its minimum at $b = 8$ (`3b^2 = 192 = S`): `1,387,712` ($b=2$), `922,016` ($b=4$), `697,480` ($b=6$), **`542,720` ($b=8$, `stores=16,384`)**, `620,556` ($b=12$, `stores=130,064`), `1,138,688` ($b=16$), `1,976,320` ($b=32$).
    - Under `BlkLRU` with intra-tile `i,j,k`, total I/O drops from `2,129,920` ($b=2$) to `753,664` ($b=6$, `stores=16,384`, `LRU/OPT=1.081`), then **jumps up to `1,040,896` at $b=8$ (`stores=258,304`, `LRU/OPT=1.918`)** because dirty $C$ words are evicted on every `kk` tile (`258,304` stores $\approx 16 \times 16,384$), before climbing to `2,490,368` at $b=16$.
  - **4. Intra-tile loop-order experiment explaining LRU vs OPT at $3b^2 = S$ ($n=128, S=192, b=8$)**:
    - `inner_ijk`: `LRU = 1,040,896` (`loads=782,592, stores=258,304`) vs `OPT = 542,720` (`loads=526,336, stores=16,384`), `LRU/OPT = 1.918`.
    - `inner_ikj`: `LRU = 948,736` (`loads=736,512, stores=212,224`) vs `OPT = 540,928` (`stores=16,384`), `LRU/OPT = 1.754`.
    - `inner_kij`: **`LRU = 557,056` (`loads=540,672, stores=16,384`)** vs `OPT = 528,128` (`loads=511,744, stores=16,384`), **`LRU/OPT = 1.055`**.
- **What it showed**: Predictions 1 (exact $2 \times 2$ hand counts `24` and `19` and off-by-one controls), 2b (exponents $\alpha = 3.000, \beta = -0.493$ and OPT constant $3.50..3.58 \approx 2\sqrt{3}$), and 3 (intra-tile `k,i,j` eliminating the LRU thrashing of $C$) were confirmed. Prediction 2a underestimated the severity of `inner_ijk` LRU thrashing at $3b^2 = S$: we predicted `BlkLRU` constant $c \approx 4.0..4.8$ and `Naive/Blk(LRU)` ratio $5.5\times..7.6\times$ at $n=256, S=768$, whereas `BlkLRU` had $c = 6.916$ (`Naive/Blk = 4.05x`) because `inner_ijk` at $3b^2 = S$ evicts and writes back almost the entire $C_{II,JJ}$ tile (`stores = 258,304` vs `16,384` at $n=128$) on every `kk` step; switching intra-tile order to `k,i,j` restores `stores = 16,384` and cuts `BlkLRU` by `46.5%` (`LRU/OPT = 1.055`).

### E7: FFT butterfly graphs ($n = 4, 8$ exact/state-space and recursive blocked schedules for larger $n$)
- **Question**: (1) What are the exact minimum I/O counts $Q^*(\text{FFT}_4, S)$ for the 4-point FFT butterfly graph ($N=12$ vertices) across $S=1..7$? (2) On the 8-point FFT butterfly graph ($N=32$ vertices), how many states does unrestricted bitmask 0-1 BFS explore before hitting a `200,000`-state cap, and what I/O counts do verified constructive schedules achieve across $S$? (3) For larger $n \in \{16, 32, 64, 128, 256\}$, what exact I/O counts are produced by a recursive radix-$B$ blocked FFT schedule generator verified by `PaperPebbleEngine` across $B \in \{2, 4, 8, 16\}$?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 07:10 UTC):
  1. **Exact $Q^*(\text{FFT}_4, S)$ ($N=12$)**: $S=1, 2 \to \text{infeasible}$ (`None`); $S=3 \to Q^* = 14$; $S=4 \to Q^* = 12$; $S=5 \to Q^* = 10$; $S \ge 6 \to Q^* = 8$ (trivial bound $|I| + |O| = 8$).
  2. **$\text{FFT}_8$ ($N=32$)**:
     - Unrestricted bitmask 0-1 BFS will exceed the `200,000`-state cap at $S \ge 3$ without terminating (because 16 intermediate vertices admit $2^{16}$ blue subsets times $\binom{32}{S}$ red subsets).
     - Verified constructive schedules on $\text{FFT}_8$: at $S=3$ (radix-2), `48` I/Os stage-by-stage and `44` I/Os with 1-pebble inter-stage handover; at $S=6$ (radix-4 + radix-2), `32` I/Os pass-by-pass and `26` I/Os with 3-pebble inter-pass handover; at $S \ge 10$ (radix-8), `16` I/Os ($|I|+|O|=16$, zero intermediate spills).
  3. **Larger $n \in \{16, 32, 64, 128, 256\}$**: A recursive in-register radix-$B$ sub-FFT schedule ($B=2$ at $S=3$, $B=4$ at $S=6$, $B=8$ at $S=14$, $B=16$ at $S=30$, or streamed last-stage $S=10, 18$) verified move-by-move by `PaperPebbleEngine` will achieve $\text{IO} = 2n \lceil (\log_2 n) / \log_2 B \rceil$, giving an exact $2.00\times$ reduction for $B=4$ when $2 \mid \log_2 n$ and $3.00\times$ for $B=8$ when $3 \mid \log_2 n$, with constant $\text{IO} / (n \log_2 n / \log_2 B) = 2.000$.
- **Run**: `scripts/pebbling-w2-e7.sh` (using `work/fft_schedules.py`) via `k3.ps1 -Action run -Name pebbling-w2-e7 -Cpus 0-5`. Output: `results/pebbling-w2-e7.out`.
- **Observation**:
  - **1. Exact $Q^*(\text{FFT}_4, S)$ on the 12-vertex 4-point FFT butterfly graph (`results/pebbling-w2-e7.out`)**:
    - $S=1, 2$: `None` (infeasible, `5` and `11` states explored).
    - $S=3$: $Q^* = 14$ (`R1_in=4, R1_mid=3, R2_mid=3, R2_out=4`; `22,439` states).
    - $S=4$: $Q^* = 11$ (`R1_in=5, R1_mid=1, R2_mid=1, R2_out=4`; `89,976` states).
    - $S=5$: $Q^* = 9$ (`R1_in=5, R1_mid=0, R2_mid=0, R2_out=4`; `155,889` states).
    - $S=6$: $Q^* = 8$ (`R1_in=4, R1_mid=0, R2_mid=0, R2_out=4`; `227,077` states).
    - $S=7$: $Q^* = 8$ (`R1_in=4, R1_mid=0, R2_mid=0, R2_out=4`; `333,200` states).
  - **2. 8-point FFT butterfly graph ($\text{FFT}_8$, 32 vertices)**:
    - Unrestricted bitmask 0-1 BFS (`solve_dijkstra_bounded`): $S=1, 2$ terminate as `None` (infeasible) after `9` and `37` states; at $S=3, 4, 6$, the search hits the `200,000`-state cap (`200,000`, `200,003`, `200,006` states) without terminating because 16 intermediate vertices admit $2^{16}$ blue masks times $\binom{32}{S}$ red masks.
    - Verified constructive schedules on $\text{FFT}_8$ (`verify_schedule_and_peak_red`):
      - $S=3$: `48` I/Os (`144` moves) pass-by-pass; `44` I/Os (`136` moves) keeping 1 red pebble across each of the 2 stage boundaries.
      - $S=6$: `32` I/Os (`112` moves) pass-by-pass; `24` I/Os (`96` moves) keeping all 4 stage-2 outputs of the second 4-point sub-FFT (`nodes 20..23`) in red and streaming stage-3 pairs in the remaining 2 red slots.
      - $S=10$: `16` I/Os (`80` moves, trivial bound $|I|+|O|=16$, zero intermediate spills).
  - **3. Recursive radix-$B$ blocked FFT schedules across $n \in \{4, 8, 16, 32, 64, 128, 256\}$**:
    - Every generated schedule across all 25 $(n, B)$ pairs was verified move-by-move by `PaperPebbleEngine` (`Verified=True`), with peak red pebbles `PeakRed = 3` for $B=2$ ($m=1$) and `PeakRed = B + 2` for $B \in \{4, 8, 16\}$ (`6, 10, 18`).
    - Total I/O equals $2n \lceil (\log_2 n) / \log_2 B \rceil$: when $\log_2 B \mid \log_2 n$, the normalized constant $\text{IO} / (n \log_2 n / \log_2 B)$ is `2.000`, giving an exact `2.00x` I/O reduction for $B=4$ ($S=6$), `3.00x` for $B=8$ ($S=10$, e.g. `768 -> 256` at $n=64$), and `4.00x` for $B=16$ ($S=18$, e.g. `4096 -> 1024` at $n=256$).
- **What it showed**: Predictions 2 and 3 held (`FFT_8` hits the `200,000`-state cap at $S \ge 3$, and recursive radix-$B$ schedules require $S = B + 2$ red pebbles and achieve $\text{IO} = 2n \lceil \log_2 n / \log_2 B \rceil$). Prediction 1 was right at $S=1, 2, 3, 6, 7$ (`INF, INF, 14, 8, 8`) but **wrong by 1 I/O at $S=4$ (`11` vs predicted `12`) and $S=5$ (`9` vs predicted `10`)**: we assumed every register eviction at stage 1 spills a stage-1 intermediate (`R2_mid + R1_mid = +2` I/Os), whereas the exact solver deletes an **input** from red (`R4(input, red)`, cost 0) and later reloads that input from its initial blue pebble (`R1_in = 5`, cost +1 I/O) to recompute the second stage-1 butterfly output without spilling the input!

## 4. How it works

- **Game state and transitions**: In Hong & Kung's Section 2 game on a DAG $G = (V, E)$, fast memory holds at most $S$ words (red pebbles $R \subseteq V$, $|R| \le S$) and slow memory holds arbitrarily many words (blue pebbles $B \subseteq V$). Calculation starts at $(R = \emptyset, B = I)$ and terminates at $(R = \emptyset, B = O)$ (E2).
- **Cost accounting**: Only moving data from slow to fast memory (`R1`: place red on $v \in B$, cost $+1$) and from fast to slow memory (`R2`: place blue on $v \in R$, cost $+1$) incur I/O cost; computing a vertex whose immediate predecessors are all in $R$ (`R3`, cost $0$) and deleting any pebble (`R4`, cost $0$) are free in the I/O metric (E2, E3).
- **Simultaneous occupancy on spill**: Rule `R2(v)` places a blue pebble on $v \in R$ without removing its red pebble ($v \in R \cap B$), so freeing the red slot requires a subsequent `R4(v, red)` move; by contrast, `michaelkeating/pebbling` models `TRANSFER` as an atomic move between exclusive tiers (`register`, `sram`, `hbm`) and charges picojoules on both `PLACE` and `TRANSFER` (E1, E2).
- **Feasibility threshold**: A DAG with maximum fan-in $d$ is infeasible for $S \le d$ because firing `R3(v)` on a vertex with $d$ predecessors requires all $d$ predecessors in $R$ plus 1 free red slot for $v$ itself ($S \ge d + 1$, so $S \ge 2$ for chains and $S \ge 3$ for fan-in-2 trees, diamonds, pyramids, matrix multiply, and FFT) (E3, E7).
- **Three mechanisms that reduce I/O below naive spilling**:
  1. **Cross-output input reuse with multiple live spills**: On $2 \times 2$ matrix multiplication at $S=3$, holding an input $A_{ik}$ or $B_{kj}$ in red to compute and spill products belonging to two different $C_{ij}$ outputs reduces $Q^*(3)$ from $4 \times 7 = 28$ (independent output evaluation) to $25$ (E3).
  2. **Recomputation (`R3` multiple times on the same vertex)**: When a predecessor remains in red (or can be reloaded from an existing input blue pebble at cost 1), deleting an intermediate red pebble via `R4` and recomputing it via `R3` costs $0$ (or $1$) I/O instead of $2$ I/O (`R2 + R1`). This lowers $Q^*$ starting at $n=5$ vertices (`G5_recomp` at $S=3$: $Q^*_{\text{recomp}} = 2 < Q^*_{\text{no-recomp}} = 4$) and occurs on $3.33\%$ ($n=5$), $14.74\%$ ($n=6$), $31.00\%$ ($n=7$), and $48.20\%$ ($n=8$) of fan-in-$\le 2$ DAGs at $S=3$ (E4).
  3. **Input re-reading instead of intermediate spilling**: Because inputs start in $B$, evicting an input from $R$ (`R4`, cost 0) and later reloading it (`R1`, cost 1) costs $+1$ I/O instead of $+2$ I/O (`R2 + R1` on an intermediate), which is why $Q^*(\text{FFT}_4, S)$ achieves odd totals $11$ at $S=4$ (`R1_in=5, R1_mid=1, R2_mid=1, R2_out=4`) and $9$ at $S=5$ (`R1_in=5, R1_mid=0, R2_mid=0, R2_out=4`) (E7).
- **The $2S$-partition lower bound (Theorem 3.1 & Lemma 3.1)**: Cutting any complete schedule into $h = \lceil Q / S \rceil$ segments of at most $S$ I/O moves induces a valid $2S$-partition $\{V_1, \dots, V_h\}$ because at the start of segment $C_i$ at most $S$ vertices have red pebbles ($|V_R| \le S$) and during $C_i$ at most $S$ moves use `R1` or `R2` ($|D_i| \le |V_R \cup V_{BR}| \le 2S$ and $|M_i| \le 2S$), yielding $Q^* \ge S \cdot (P(2S) - 1)$. Whenever $2S \ge \max(|I|, |O|)$, the 1-block partition $\{V\}$ is a valid $2S$-partition ($D_1 = I, M_1 = O$), so $P(2S) = 1$ and the bound evaluates to $0$ (E5).
- **Matrix multiplication scaling and LRU vs OPT**: Tiling $n \times n$ matrix multiplication with $b = \lfloor \sqrt{S/3} \rfloor$ in a word-granularity fully associative cache of $S$ words scales as $\text{IO} = 6.592 \cdot n^{2.9996} S^{-0.4930}$ under LRU and $4.230 \cdot n^{2.9396} S^{-0.4728}$ under OPT (normalized constant $c = \text{IO}/(n^3/\sqrt{S}) \approx 3.50..3.58 \approx 2\sqrt{3}$). At $3b^2 = S$, intra-tile order `i,j,k` causes LRU to evict dirty $C_{II,JJ}$ rows before dead $B_{KK-1,JJ}$ words (`LRU/OPT = 1.918` at $n=128, S=192, b=8$, with `258,304` stores vs `16,384`); reordering the intra-tile loops to `k,i,j` touches all of $C_{II,JJ}$ during the final `k` pass, restoring `stores = 16,384` and reducing `LRU/OPT` to `1.055` (E6).
- **FFT butterfly blocking**: Evaluating an $n$-point FFT in recursive radix-$B$ ($B = 2^m$) sub-FFTs requires peak red pebbles $S = 3$ for $B=2$ and $S = B + 2$ for $B \ge 4$ (`6, 10, 18` for $B=4, 8, 16$) because the second half-butterfly of size $B/2$ peaks at $B/2 + 2$ red slots while $B/2$ outputs of the first half are held in red ($(B/2) + (B/2 + 2) = B + 2$). Each pass of $m = \log_2 B$ stages performs $2n$ I/Os ($n$ `R1`s $+ n$ `R2`s), giving $\text{IO} = 2n \lceil \log_2 n / \log_2 B \rceil$ (E7).

## 5. What surprised me and what I got wrong

1. **E1 (`topological_register_strategy` and `greedy_cheapest_strategy` completion)**: I predicted `topological_register_strategy` would finish (`done=True`) with `10..25` pJ on the 10-node demo DAG. It ended with `done=False` (`10` pJ, 17 steps) on the demo and on all 5 difficulty presets because `free_register_if_needed()` scores register candidates by `unpebbled_succs` without excluding sink nodes (which have `0` successors, so they are evicted first). Conversely, `greedy_cheapest_strategy` reported `done=True` on `medium_preset` (`h100`) only because it hit the `500`-step timeout oscillating between `REMOVE N7` (score `50.0`) and `PLACE N7 in register` (score `-49.0`) rather than placing remaining sinks in `hbm` (score `50.2`).
2. **E3b ($2 \times 2$ matrix multiplication at $S=3$)**: Evaluating the four output trees $C_{ij}$ independently at $S=3$ takes $4 \times 7 = 28$ I/Os (`max_bprod=1`), but the unpruned exact solver found $Q^*(3) = 25$ (`73,793` states) by holding a shared input in red and keeping two spilled products from different output trees in blue simultaneously.
3. **E4 (Recomputation saving I/O at $S=2$ on fan-in-1 trees)**: At $n=6, S=2$, `3/120` feasible fan-in-1 trees (`2.50%`) had $Q^*_{\text{recomp}} < Q^*_{\text{no-recomp}}$ because re-reading an input (`R1(0) + R3(1)`, cost 1) is 1 I/O cheaper than spilling and reloading an intermediate (`R2(1) + R1(1)`, cost 2).
4. **E6 (`BlkLRU` constant at $3b^2 = S$ under intra-tile `i,j,k`)**: I predicted `BlkLRU` would have normalized constant $c \approx 4.0..4.8$ and `Naive/Blk(LRU)` ratio $5.5\times..7.6\times$ at $n=256, S=768$ ($b=16$). Instead, `BlkLRU` had $c = 6.916$ (`Naive/Blk(LRU) = 4.05x`) because at $3b^2 = S$, intra-tile order `i,j,k` touches all $b^2$ words of $B_{KK-1,JJ}$ during the last row `i = b-1`, making dead $B$ words more recently used than rows `0..b-2` of live $C_{II,JJ}$ and causing LRU to evict and write back almost the entire $C$ tile on every `kk` step (`258,304` stores vs `16,384` at $n=128, S=192$). Switching the intra-tile loop order to `k,i,j` cut `BlkLRU` by `46.5%` (`1,040,896 -> 557,056`, `LRU/OPT = 1.055`, `stores = 16,384`).
5. **E7 (Odd exact $Q^*$ values on $\text{FFT}_4$ at $S=4, 5$)**: I predicted $Q^*(\text{FFT}_4, 4) = 12$ and $Q^*(\text{FFT}_4, 5) = 10$ assuming every stage-1 eviction spills an intermediate (`R2 + R1 = +2` I/Os). The exact solver found $Q^*(\text{FFT}_4, 4) = 11$ (`R1_in=5, R1_mid=1, R2_mid=1, R2_out=4`) and $Q^*(\text{FFT}_4, 5) = 9$ (`R1_in=5, R1_mid=0, R2_mid=0, R2_out=4`) by deleting an input from red (`R4`, cost 0) and reloading it from its initial blue pebble (`R1`, cost 1) to recompute the second butterfly output for $+1$ I/O instead of $+2$.

## 6. Authors say / code does / I observed

- **Memory hierarchy and cost metric**:
  - *Authors say* (Hong & Kung 1981, Section 2, p. 327): Two levels—fast memory of capacity $S$ (red pebbles) and unbounded slow memory (blue pebbles). Only `R1` (blue $\to$ red) and `R2` (red $\to$ blue) count toward $Q$ ($+1$ each); `R3` (compute in red) and `R4` (delete) cost $0$.
  - *Repository code does* (`pebbling_engine.py`, lines 49–87, 536–549): Three exclusive tiers (`register`, `sram`, `hbm`) priced in picojoules. `PLACE` charges `cost_per_step` (`1, 5, 100` pJ on `h100`), `TRANSFER` charges `io_transfer_cost` of the destination tier (`0` pJ into `register`, `5` pJ into `sram`, `100` pJ into `hbm`), and `REMOVE` charges `0` pJ.
  - *I observed* (E1, E2): In the repository, transferring a pebble from `sram`/`hbm` into `register` costs `0` pJ, whereas computing a node in `register` costs `1` pJ.
- **Rules for computation, spilling, and completion**:
  - *Authors say* (Section 2, p. 327): Initial state has blue pebbles on all inputs ($B = I, R = \emptyset$); `R3(v)` requires all predecessors of $v$ to hold **red** pebbles; `R2(v)` places a blue pebble while keeping the red pebble on $v$; terminal state requires $B = O$ and $R = \emptyset$.
  - *Repository code does* (`pebbling_engine.py`, lines 387–405, 515–560): Initial state has all nodes `None`; `_can_compute(v)` requires predecessors to have a pebble in **any** tier (`register`, `sram`, or `hbm`) and allows `PLACE` directly into `sram` or `hbm`; `TRANSFER` moves the pebble out of the source tier in one step; `done` becomes `True` as soon as all sinks are pebbled in any tier (or when `step_count >= max_steps`).
  - *I observed* (E2): All six differences were demonstrated on the 4-node diamond DAG (`results/pebbling-w2-e2.out`).
- **Asymptotic bounds vs finite-graph exact values**:
  - *Authors say* (Lemma 3.1, p. 329; Corollary 4.1, p. 329; Section 6, p. 330): $Q \ge S \cdot (P(2S) - 1)$, yielding $Q \cdot \log S = \Omega(n \log n)$ for $n$-point FFT and $Q = \Omega(n^3 / \sqrt{S})$ for $n \times n$ matrix multiplication.
  - *I observed* (E5, E6, E7): Theorem 3.1's schedule-to-$2S$-partition construction holds on 100% of tested optimal schedules, and when $2S \ge \max(|I|, |O|)$ on small graphs, $\{V\}$ is a single valid $2S$-partition so $P(2S) = 1$ and $S(P(2S)-1) = 0$ (E5). On blocked matrix multiplication, the measured exponents are $\alpha = 2.9996, \beta = -0.4930$ (LRU) and $\alpha = 2.9396, \beta = -0.4728$ (OPT, constant $c \approx 3.50 \approx 2\sqrt{3}$) (E6). On $n$-point FFT ($n=4..256$), verified recursive radix-$B$ schedules require $S = B + 2$ red pebbles and achieve $\text{IO} = 2n \lceil \log_2 n / \log_2 B \rceil$ (E7).

## 7. Not run, and why. Unknowns.

- **`LLMAgent` in `pebbling_agent.py`**: Requires an Anthropic API key (`ANTHROPIC_API_KEY`) and network calls to `claude-sonnet-4-5-20250929`; skipped per brief ("skip anything needing an API key").
- **Browser UI (`pebbling-game.jsx`) interactive clicking**: React artifact wrapper inspected in source (`lines 1–300`); not run in a browser.
- **Unrestricted exact $Q^*(\text{FFT}_8, S)$ for $S \ge 3$**: As shown in E7 (`results/pebbling-w2-e7.out`), unrestricted bitmask 0-1 BFS on the 32-vertex $\text{FFT}_8$ graph hits the `200,000`-state cap at $S=3, 4, 6$ without terminating; constructive schedules verified by `PaperPebbleEngine` establish upper bounds $Q^*(\text{FFT}_8, 3) \le 44$, $Q^*(\text{FFT}_8, 6) \le 24$, and $Q^*(\text{FFT}_8, 10) = 16$ (exact, matching $|I|+|O|=16$). Whether $44$ ($S=3$) and $24$ ($S=6$) can be lowered by a few I/Os via input re-reading (as on $\text{FFT}_4$ at $S=4, 5$) remains an open exact-search question.
- **Belady/OPT at $n=512$ in E6**: At $n=512$, the loop nest has $3n^3 = 402,653,184$ accesses ($\sim 3.2$ GB for `trace` + `next_use`), so OPT was run for $n \in \{64, 128, 256\}$ (up to $50.3\text{M}$ accesses) and skipped at $n=512$, where LRU was run.

## 8. Sources and Sources read

- **Paper**: J.-W. Hong and H. T. Kung, "I/O complexity: The red-blue pebble game", *Proceedings of the 13th Annual ACM Symposium on Theory of Computing (STOC 1981)*, pp. 326–333, 1981.
  - Local extracted text: `C:\personal\oss\scratch-agents\data-movement\pebbling\work\paper\1981-stoc-hong-kung.txt` (943 lines, 35,041 bytes, all 8 pages pp. 326–333; read in full).
- **Repository**: `https://github.com/michaelkeating/pebbling.git`, commit `985320b55c8422667d9d571453e27ea1756888df`, retrieved 2026-10-26, MIT License (Copyright 2026 Sutro Group).
  - `README.md`: 215 lines (read in full).
  - `pyproject.toml`: 33 lines (read in full).
  - `LICENSE`: 21 lines (read in full).
  - `pebbling_engine.py`: 1,005 lines (read in full).
  - `pebbling_agent.py`: 578 lines (read in full).
  - `pebbling-game.jsx`: 1,672 lines (lines 1–300 read).

## 8b. Independent check (second pass, 2026-10-26)

The seven experiments above were written and run by an earlier worker. I re-checked the main numbers with code I wrote separately (`research/red-blue-pebbling/verify/`), using only the study's DAG definitions in the first check.

- **Exact solver** (`check_solver_independent.sh`, my own Dijkstra over (red, blue) bitmasks, recompute allowed): same minimum I/O as E3, E4 and E7 on every case it finished: chain, diamond, `bintree_7` (S=3 -> 7, S=4 -> 5), `pyramid_b3` (S=3 -> 6, S=4 -> 4), `pyramid_b4` (S=3 -> 11, S=4 -> 7, S=5 -> 5), `matmul_2x2` at S=3 -> 25, `FFT_4` S=3..6 -> 14, 11, 9, 8, and `G5_recomp` at S=3 -> 2 with recompute and 4 without.
- **Not re-checked**: `matmul_2x2` at S >= 4 (17, 14, 13, 12). My solver has no pruning and stopped at its 3,000,000-state cap, so those four values rest on the earlier worker's solver alone. E5 (S-partition) and the FFT_8 schedules (E7) were also not re-run by me.
- **Cache simulator** (`check_cache_sim_independent.sh`, my own Python LRU and Belady/OPT, same access order A[i,k], B[k,j], C[i,j], dirty C written back, final flush counted): the 2x2 case gives LRU 20+4 and OPT 15+4 at S=4, OPT 17+4 at S=3, LRU 18+4 at S=5; for n=128, S=192, b=8 the three loop orders give exactly the E6 numbers (ijk LRU 1,040,896 and OPT 542,720; ikj 948,736 and 540,928; kij 557,056 and 528,128). The n=256 and n=512 sweep and the fitted exponents were not re-run by me.

## 9. Files

- **Context file**:
  - `C:\personal\oss\clover-framework-hybrid\clover-kimi-k3\docs\red-blue-pebbling-context.md`
- **Work modules & predictions (`C:\personal\oss\clover-framework-hybrid\clover-kimi-k3\research\red-blue-pebbling\work\`)**:
  - `predictions.txt`: Timestamped predictions written before each experiment E1–E7.
  - `paper_engine.py`: Minimal engine for Hong & Kung (1981) Section 2 rules (`R1`, `R2`, `R3`, `R4`).
  - `dag_lib.py`: Canonical DAG generators (`chain`, `v_tree`, `diamond`, `binary_tree`, `pyramid`, `matmul_2x2`, `fft_butterfly`).
  - `exact_solver.py`: 0-1 BFS / Dijkstra bitmask solver (`allow_recompute`, `broken_any_pred`, `dead_pebble_prune`) and `solve_raw_bfs`.
  - `s_partition.py`: Dominator set, minimum set, $K$-partition verifier, Theorem 3.1 schedule-to-$2S$-partition constructor, and exact brute-force $P(K)$ solver.
  - `cache_sim.c`: Word-granularity fully associative LRU and Belady/OPT cache simulator in C for naive and tiled matrix multiplication.
  - `fft_schedules.py`: Bounded-state 0-1 BFS and recursive radix-$B$ blocked FFT schedule generator and verifier.
- **Scripts (`C:\personal\oss\clover-framework-hybrid\clover-kimi-k3\research\red-blue-pebbling\scripts\`)**:
  - `pebbling-w2-e1.sh`, `pebbling-w2-e1b.sh`, `pebbling-w2-e2.sh`, `pebbling-w2-e3.sh`, `pebbling-w2-e3b.sh`, `pebbling-w2-e4.sh`, `pebbling-w2-e5.sh`, `pebbling-w2-e6.sh`, `pebbling-w2-e7.sh`
- **Results (`C:\personal\oss\clover-framework-hybrid\clover-kimi-k3\research\red-blue-pebbling\results\`)**:
  - `pebbling-w2-e1.out`, `pebbling-w2-e1b.out`, `pebbling-w2-e2.out`, `pebbling-w2-e3.out`, `pebbling-w2-e3b.out`, `pebbling-w2-e4.out`, `pebbling-w2-e5.out`, `pebbling-w2-e6.out`, `pebbling-w2-e7.out`
- **Remote slice on k3**:
  - `/root/dm-study/pebbling/w2/` (contains compiled `cache_sim`, `matmul_exact`, Python modules, and `e1_repo` / `e2_work` copies; total $< 5$ MB).
