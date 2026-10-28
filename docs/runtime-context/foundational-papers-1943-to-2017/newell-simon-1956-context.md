# Allen Newell & Herbert A. Simon (1956) — *The Logic Theory Machine: A Complex Information Processing System* — Context

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh.

## 1. Status

- **E1 (Run reference repositories on k3 CPU & inspect 1956 IPL-I errata)**: DONE
- **E2 (Run `dmoews/logic-theorist` on *Principia Mathematica* Chapter `*2` & verify extracted proofs)**: DONE
- **E3 (Own 1956 Logic Theory Machine implementation `lt1956_engine.py`: Similarity filter `CSm` and Contraction `VCt` ablations)**: DONE
- **E4 (Section II "Learning Processes" — Theorem-memory accumulation vs axioms-only, and subproblem priority rule `Ex(CK)` minimal-$K$ vs FIFO vs Max-$K$)**: DONE

## 2. What this is

Authors say (Allen Newell and Herbert A. Simon, "The Logic Theory Machine: A Complex Information Processing System", RAND Report P-868, July 12, 1956 / *IRE Transactions on Information Theory*, Vol. IT-2, No. 3, pp. 61–79, September 1956):
- "In this paper we describe a complex information processing system, which we call the logic theory machine, that is capable of discovering proofs for theorems in symbolic logic. This system, in contrast to the systematic algorithms that are ordinarily employed in computation, relies heavily on heuristic methods similar to those that have been observed in human problem solving activity." (Summary, p. 1)
- Section I (pp. 4–24) defines the **Logic Language (LL)** (also called IPL-I): expressions $X$ in the propositional calculus of Whitehead and Russell's *Principia Mathematica* (connectives `-` not, `v` or, and `->` implies defined by `*1.01: p -> q =def -p v q`) are stored in **storage memory lists** $A$ as trees of **elements** $E = (G, V, F, C, N, P, A, U)$ with three structural **descriptive symbols**: $H$ (number of variable/unit places), $J$ (number of distinct variable/unit names), and $K$ (number of tree levels). Processing takes place in **working memories** $1, 2, \dots$ via 44 elementary information processes (`Find`, `Store`, `Put`, `Numerical`, `Assign`, `Compare`, `Test`, `Branch`).
- Section II (pp. 25–48) specifies the **Logic Theorist (LT)**: starting from the five axioms of *Principia Mathematica* (`*1.2` through `*1.6`) on the theorem list $T$, LT works backward from a target theorem $X$ using an **Executive routine (`Ex`)** and three proof methods:
  1. **Substitution (`MSb`, pp. 29–37)**: filters $T$ using the **Similarity test (`CSm`)** (requiring identical left and right descriptions $DL = (K_L, J_L, H_L)$ and $DR = (K_R, J_R, H_R)$) and applies the recursive **Matching routine (`LMc`)** (which performs variable substitution `LSb` and connective replacement `LRp->v` / `LRpv->` between $p \to q$ and $\neg p \lor q$).
  2. **Detachment (`MDt`, pp. 37–42)**: searches $T$ for an implication $B \to A'$ whose right side $A'$ is similar to the target $A$—first viewing variables as units (`VV`), and if that fails, viewing binary sub-expressions as **contracted units (`VCt`)**—matches $A'$ to $A$, tries to prove the left side $B$ by substitution (`MSb`), and if that fails, appends $B$ to the active **subsidiary problem list $P$**.
  3. **Chaining (`MCh`, pp. 42–44)**: for a target $A \to C$, searches $T$ for $A \to B$ (forward chaining `SmF`/`McF`) or $B \to C$ (backward chaining `SmB`/`McB`), tries to prove the missing link ($B \to C$ or $A \to B$) by `MSb`, and otherwise appends it to $P$.
- When `MSb`, `MDt`, and `MCh` fail on a problem, `Ex` moves it to the inactive problem list $Q$, checks the global work counter $W$ (number of `MSb` calls, `CWG`), selects from $P$ the subsidiary problem with the **lowest level count $K$ (`Ex(CK)`)**, checks that it does not duplicate any expression on $Q$ (`Ex(CX)`), and applies `MDt` and `MCh` to it.
- Section III (pp. 49–63) prints the complete IPL-I program (`Ex`, `MCh`, `MDt`, `MSb`, `LMc`, `LSb`, `LRp->v`, `LRpv->`, `VV`, `VCt`, `CX`, `CSm`, `CD`, `D`, `NK`, `NH`, `NJ`, and Part 2 reductions `SX`, `SXE`, `SXM`, `SXL`, `SXR`).

## 3. Experiments

### E1: Run reference repositories (`dmoews/logic-theorist`, `dmoews/ipl-v-logic-theorist`, `duplys/logic-theorist`) & inspect 1956 IPL-I errata
- **Question**: Do the reference implementations (`dmoews/logic-theorist` for 1956 IPL-I, `dmoews/ipl-v-logic-theorist` for 1963 Stefferud IPL-V, and `duplys/logic-theorist`) pass their verification suites on `k3` CPU, and what exact bugs/typos exist in the printed 1956 RAND P-868 Section III IPL-I code when `orig-logic-routines.txt` is diffed against the runnable `logic-routines.txt`?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 15:18 UTC):
  1. `dmoews/logic-theorist` will pass `verify.py` and `test.sh` (`rc=0`), and diffing `orig-logic-routines.txt` vs `logic-routines.txt` will reveal $\ge 10$ concrete bugs/typos in the 1956 printed IPL-I routines (such as the missing branch label `D` before `SX 1 2` in `MSb` on p. 52, `AM` vs `AA` in `MCh` on p. 51, and inverted `CPS` branch senses in Part 2 store routines on pp. 60–61).
  2. `dmoews/ipl-v-logic-theorist` will pass `interpreter/test.sh` and `interpreter/test-stefferud.sh` (`rc=0`).
  3. `duplys/logic-theorist` will be a small standalone Python demo rather than a full IPL-I interpreter.
- **Run**: `scripts/e1_repos_and_errata.sh` (outputs: `results/e1_repos_and_errata.out`).
- **Observation**:
  - **Repository 1 (`dmoews/logic-theorist`, 1956 IPL-I interpreter)**:
    - Commit `07fff3586ef2136a71515ef78e7a1c6609538bc0` (CC0 public domain, 8,608 total lines across 29 files; `logic.py` 853 lines, `analyze_output.py` 224 lines, `verify.py` 200 lines, `transcription.txt` 701 lines).
    - `verify.py` confirmed consistency between `orig-logic-routines.txt` and `transcription.txt` (`rc=0`), and `bash test.sh` passed all three test suites (`analyze_output.test`, `logic.test`, `run_logic.test`: `All tests OK`, `rc=0`).
    - **10 concrete bugs/errata in the printed 1956 RAND P-868 Section III code** exposed by `diff -u orig-logic-routines.txt logic-routines.txt`:
      1. `Ex` (p. 50): `-FEN P 2 - D` is called before working memory `2` is initialized to the first element of `P` (requires `PE 1 2` after `NK 1`).
      2. `MDt` (p. 52): `A: TC-> 1 - - B` has the **wrong sign** (positive `TC->` skips every theorem whose main connective *is* `->` and tries detachment only on disjunctions; requires `-TC-> 1 - - B`, matching `MCh` on p. 51).
      3. `MDt` & `MSb` (p. 52): `SX 1 2` and `SXM 5 6` copy an expression into a new storage list without loading the new main element into working memory `2` / `6` (requires `FM 2 2` and `FM 6 6`).
      4. `MSb` (p. 52): Branch target label `D:` is missing before `SX 1 2`, and `LMc 2 L` omits the failure branch `- B` (`LMc 2 L - B`), causing `MSb` to fall through to `BHN` (reporting success) when the first similar theorem fails to match.
      5. `LMc` (p. 53): At labels `A:` (`Sby`) and `C:` (`Sbx`), `TV L - - H` and `TV C - - H` have the **wrong sign** (requires `-TV L - - H` and `-TV C - - H`: branch to failure `H` if *not* a variable), and `NSGG L C` in `Sby` mutates `G(C)` without restoring it via `NAGG L C`.
      6. `LSb` (p. 54) & `SXM`/`SXL`/`SXR` (pp. 60–61): `LSb` checks `-CN 2 C - E` without checking `-TF 2 - - E` (whether variable `2` is free), and `SXM`/`SXL`/`SXR` must clear the free flag (`PFB 2`) on subproblem variables to prevent infinite self-substitution. Also, `A: CPS 1 1 - B` in `LSb` always branches to `B` because every element's position prefix-matches itself, making the variable-collision rename at `G:` unreachable dead code.
      7. `LRpv->` (p. 55): Uses undefined opcode `-TSb 1 - - B` instead of `TF 1 - - B` (test if free).
      8. `D` (p. 58): Writes formal argument `x` (`NK x`, `NJ x`, `NH x`, `BHN x`) instead of reference place `L` (`NK L`, `NJ L`, `NH L`, `BHN`).
      9. `NJ` (p. 59): Omits branch label `D:` on `FEN L 2 - A` (targeted by `-CPS 2 L - D`, `-TU 2 - - D`, and `CN 2 3 - D`).
      10. `SXM`, `SXE`, `SXL`, `SXR` (pp. 60–61): Write positive `CPS 1 L - B` (skipping subelements of `L`) instead of negative `-CPS 1 L - B`, and `SXR` omits labels `A, B, C` and branch target `A` on `FEN L 1`.
  - **Repository 2 (`dmoews/ipl-v-logic-theorist`, 1963 Stefferud IPL-V version)**:
    - Commit `89c6d79ac2a25d7566183b3136bfbe198c9f75dd` (18,252 total lines; `logic-theorist-1963-stefferud.iplv` 2,806 lines, `interpreter/machine.py` 987 lines).
    - Passed `interpreter/test.sh` (`All tests OK`, `rc=0`) and `test-stefferud.sh` (`rc=0`, byte-identical to `test-stefferud.out`), proving `23/24` theorems from chapters `*2`, `*3`, and `*4` (`*2.01`..`*4.25`, with only `*2.15` hitting the subproblem limit after `31` subproblems and `2,524,095` effort units).
  - **Repository 3 (`duplys/logic-theorist`)**:
    - Commit `cc75327fc941560902d734f5d89cc02c3a58b561` (`logic_theorist.py`, 134 lines). Inspection and execution showed it is a hardcoded stub (`if theorem == "A -> B": ... elif theorem == "(A -> B) & A -> B": ... else: return False`) that fails even on `A -> C` from `['A -> B', 'B -> C']`.
- **What it showed**: All three predictions held, and the diff between `orig-logic-routines.txt` and `logic-routines.txt` cataloged the exact 10 bugs in the 1956 printed IPL-I specification.

### E2: Run `dmoews/logic-theorist` on *Principia Mathematica* Chapter `*2` (`pm-theorems.txt`) and verify proofs
- **Question**: When the repaired 1956 IPL-I Logic Theorist (`dmoews/logic-theorist/run_logic.py`) is executed across all 67 theorems of *Principia Mathematica* Chapter `*2` (`pm-theorems.txt`, with default work limit `W_limit = 50` and all prior theorems available in `T`) and every output is checked by the independent proof verifier `analyze_output.py`, how many theorems are proved overall, how many of the first 52 (the 1957 WJCC subset) are proved, and what is the breakdown across `MSb`, `MDt`, `MCh`, and multi-step subproblem search?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 15:18 UTC):
  1. Roughly 21 of the 67 theorems will be proved by 1-step substitution + replacement (`MSb`, matching Newell & Simon p. 37: "some 21 can be proved by the method of substitution").
  2. Total theorems proved will be roughly 35–45 of the first 52 (and 40–50 of all 67), all verified by `analyze_output.py`.
- **Run**: `scripts/e2_e3_e4_run.sh` (executing `work/run_e2_e3_e4.py`). Output: `results/e2_e3_e4_run.out`.
- **Observation**:
  - Across all **67** theorems of *Principia Mathematica* Chapter `*2`, `dmoews/logic-theorist` proved **`48 / 67` (`71.6%`)** and failed on `19 / 67` (`Result: 67 problems 48 proven 19 not proven`, all 48 proofs independently verified by `analyze_output.py`).
  - Across the **first 52** theorems of Chapter `*2` (the subset studied in Newell, Shaw & Simon 1957, where the JOHNNIAC IPL-II run proved 38/52), the 1956 IPL-I program with `W_limit = 50` proved **`40 / 52` (`76.9%`)**.
  - Breakdown of the `48` verified proofs:
    - **1-step substitution + replacement (`PROOF_SUB`)**: **`20`** (`19` via `MSb` directly + `1` (`*2.07: p -> (p \/ p)`) via `MCh`'s segment `E`/`G` direct right-side match after left-side match with `*1.3`, closely matching Newell & Simon's "some 21" on p. 37).
    - **1-step detachment + `MSb` (`MDt`)**: **`14`**.
    - **1-step forward/backward chaining + `MSb` (`MCh`)**: **`8`**.
    - **Multi-step subsidiary problem chains (depth $\ge 1$ on list $P$)**: **`6`** (e.g., `*2.18: (~p -> p) -> p` proved in 3 steps: forward-chain with `*2.16: (A -> B) -> (~B -> ~A)` $\to$ forward-chain with `*1.2: (A \/ A) -> A` $\to$ substitution with `*2.14: ~~A -> A`).
- **What it showed**: Both predictions held (`48/67` overall, `40/52` on the first 52, and `20` 1-step substitution proofs).

### E3: Own 1956 Logic Theory Machine implementation (`lt1956_engine.py`) — Similarity Filter (`CSm`) and Contraction (`VCt`) ablations
- **Question**: Using our own self-contained Python implementation (`lt1956_engine.py` + `lt1956_methods.py`) of the 1956 Logic Theory Machine: (a) What fraction of candidate theorems pass the Similarity test `CSm` ($DL(x)=DL(y) \land DR(x)=DR(y)$ where $D=(K,J,H)$), how many `LMc` calls does `CSm` save across the 67-theorem campaign, and does `CSm` ever falsely reject a valid 1-step substitution proof? (b) On `*2.06` (`(p -> q) -> ((q -> r) -> (p -> r))`) and across all 67 theorems, how does Contraction (`VCt`, pp. 39–42) change descriptions $D=(K,J,H)$ and theorem-proving success?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 15:18 UTC):
  1. `CSm` will pass a small fraction of candidate pairs, cutting `LMc` matching invocations by $>10\times$ (or $4\times\text{–}15\times$), while occasionally rejecting a valid match where variable identification alters $J$ or molecular substitution alters $(K, H)$ (so `CSm OFF` will prove at least as many theorems as `CSm ON`).
  2. On `*2.06`, `VV` gives `DL=(2,2,2), DR=(3,3,4)` (which matches no theorem's right side, so `MDt` with `VCt OFF` fails), whereas `VCt` contracts `p->q`, `q->r`, `p->r` into units giving `DL=(1,1,1), DR=(2,2,2)`, enabling `MDt` to prove `*2.06`.
- **Run**: `scripts/e2_e3_e4_run.sh` (executing `work/run_e2_e3_e4.py`). Output: `results/e2_e3_e4_run.out`.
- **Observation**:
  - **Agreement between `LogicTheorist1956` and `dmoews/logic-theorist` IPL-I**:
    - Across all 67 theorems of Chapter `*2`, our `LogicTheorist1956` (`work_limit=50`) agreed with `dmoews/logic-theorist` on **`65 / 67` theorems (`97.0%`)**, proving `46/67` (`38/52` on the first 52—matching Newell, Shaw & Simon's 1957 count of `38/52`!). The only 2 differences (`*2.18` and `*2.38`) reached `W = 52` and `W = 54` in `LogicTheorist1956` vs `W <= 50` in `dmoews` because `dmoews`'s `MCh` executes `-LMc 6 2 - G` in place on working memory `4` and `L` before `SXL`/`SXR`.
  - **E3.1 — Similarity Filter `CSm` ON vs OFF**:
    - With **`CSm ON`**: `CSm` passed **`5,206 / 81,416` candidate checks (`6.4%`)**, resulting in **`5,726` `LMc` calls** and **`46 / 67` theorems proved** (`19` by `MSb` directly).
    - With **`CSm OFF`** (calling `LMc` on every candidate theorem without filtering by $D=(K,J,H)$): `LMc` calls jumped **$13.1\times$** to **`75,246`**, and proved **`48 / 67` theorems** (`20` by `MSb` directly).
    - **Why `MSb` with `CSm ON` missed `*2.07` (`p -> (p \/ p)`) while `CSm OFF` proved it via `MSb(*1.3)`**: Investigating the 1-theorem `MSb` difference (`19` vs `20`) revealed that in `*2.07: p -> (p \/ p)`, the right subexpression `p \/ p` has $J_R = 1$ distinct variable name (`p`), so $DR(*2.07) = (2, 1, 2)$, whereas Axiom `*1.3: B -> (A \/ B)` has $J_R = 2$ distinct variable names (`A, B`), so $DR(*1.3) = (2, 2, 2)$. Thus `CSm` **falsely rejects** `*1.3` during `MSb` on `*2.07`! With `CSm ON`, `*2.07` fails `MSb` and `MDt` and is only rescued inside `MCh` (segment `E`/`G`), because `MCh(SmF)` checks `CSm` only on the left subexpressions (`p` vs `B`, both `(1,1,1)`) and then `MCh(McR)` matches the right subexpressions `p \/ p` and `A \/ B` via `LMc` *without* calling `CSm`!
  - **E3.2 — Contraction `VCt` ON vs OFF**:
    - On `*2.06: (p -> q) -> ((q -> r) -> (p -> r))`, `VV` produced `DL=(2, 2, 2), DR=(3, 3, 4)` (matching P-868 p. 39 to the digit), and `VCt` produced `DL=(1, 1, 1), DR=(2, 2, 2)` (matching P-868 p. 40 to the digit).
    - With `VCt OFF`, `MDt` on `*2.06` returned `None` (failed). With `VCt ON`, `MDt` proved `*2.06` in 1 step!
    - **Surprise on *which* theorem `MDt(SmCt)` matched on `*2.06`**: In P-868 pp. 41–42, Newell & Simon state in prose that `MDt(SmCt)` proves `*2.06` by matching the right side of `*2.04` (`(A -> (B -> C)) -> (B -> (A -> C))`) and proving the left side by `*2.05`. However, in both `dmoews/logic-theorist` (1956 IPL-I) and our `LogicTheorist1956`, `MDt(SmCt)` actually proves `*2.06` via **`MDt(*1.5) + MSb(*1.6)`**! Why? Because `*1.5` (`(A \/ (B \/ C)) -> (B \/ (A \/ C))`) precedes `*2.04` in theorem list $T$, its right side `B \/ (A \/ C)` *also* has `DL=(1,1,1), DR=(2,2,2)` (since $D=(K,J,H)$ ignores whether the connective is `\/` or `->`), and `LMc` uses `LRpv->` to convert `\/` to `->` by substituting `~B` and `~A`, yielding subproblem `~(q -> r) \/ (~(p -> q) \/ (p -> r))` which matches `*1.6`!
    - Across all 67 theorems of Chapter `*2`, turning `VCt OFF` dropped theorems proved by `MSb + MDt` from **`33 / 67` to `24 / 67`** (`-9` theorems), and dropped full `MSb + MDt + MCh` from **`46 / 67` to `38 / 67`** (`-8` theorems).
- **What it showed**: Both predictions held, and the two ablations exposed (1) a concrete false negative of `CSm` on `*2.07` (`J_R=1` vs `J_R=2` on `*1.3`), and (2) why the actual 1956 IPL-I program proves `*2.06` from `*1.5 + *1.6` instead of the paper prose's `*2.04 + *2.05`.

### E4: Section II "Learning Processes" (Theorem-Memory Accumulation, p. 48) and Subproblem Priority (`Ex(CK)`, p. 46)
- **Question**: (a) How many of the first 52 and all 67 theorems of Chapter `*2` can the 1956 Logic Theory Machine prove under `W_limit = 50` when theorem memory $T$ contains (i) all prior theorems (`all_prior`, P-868 baseline), (ii) only previously *proved* theorems (`proved_only`, self-bootstrapped learning), or (iii) only the 5 axioms `*1.2..*1.6` (`axioms_only`, no theorem accumulation)? (b) How does the executive routine's subproblem priority rule `Ex(CK)` (p. 46 & p. 50: minimal level $K$ with `CKG 2 1 - C` LIFO tie-breaking) compare against FIFO (breadth-first) and Maximal-$K$ (anti-heuristic negative control)?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 15:18 UTC):
  1. Restricting $T$ to the 5 axioms `*1.2..*1.6` (`axioms_only`) will cut proved theorems by more than half compared to `all_prior`, confirming Newell & Simon's statement on page 48.
  2. `min_k` and `fifo` will outperform or require less work than `max_k`.
- **Run**: `scripts/e2_e3_e4_run.sh` (executing `work/run_e2_e3_e4.py`). Output: `results/e2_e3_e4_run.out`.
- **Observation**:
  - **E4.1 — Theorem-Memory Accumulation ("Learning Processes", p. 48)**:
    - `All prior theorems stored in T` (`all_prior`): **`38 / 52`** (`73.1%`) on the first 52, and **`46 / 67`** (`68.7%`) on all 67.
    - `Only previously PROVED theorems stored in T` (`proved_only`): **`19 / 52`** (`36.5%`) on the first 52, and **`22 / 67`** (`32.8%`) on all 67 (because once an early bridge theorem like `*2.15` or `*2.16` is unproved, later theorems that depend on it also fail in cascading fashion).
    - `Axioms *1.2..*1.6 ONLY stored in T` (`axioms_only`): **`11 / 52`** (`21.2%`) on the first 52, and **`12 / 67`** (`17.9%`) on all 67—confirming Newell & Simon's exact statement on page 48 (*"Without this form of learning it is doubtful whether the program would prove any but the first few theorems of Chapter 2 in a reasonable number of steps"*).
  - **E4.2 — Subproblem Priority Rule `Ex(CK)` (p. 46 & p. 50)**:
    - `Minimal-K first (LIFO tie-break via CKG 2 1 - C, p. 50)`: proved `46 / 67` (`4` depth $\ge 1$ subproblem proofs, `total W = 53` on those proofs).
    - `Minimal-K first (FIFO tie-break, k_i < best_k)`: proved `47 / 67` (`5` depth $\ge 1$ subproblem proofs, `total W = 103`).
    - `FIFO (breadth-first queue order)`: proved **`49 / 67`** (`7` depth $\ge 1$ subproblem proofs, `total W = 179`), solving 3 more multi-step theorems than `Minimal-K (LIFO)` because `CKG 2 1 - C`'s LIFO tie-breaking (`k_i <= best_k`) repeatedly dives into newly generated subproblems of the same depth $K$ (depth-first within a $K$-level) instead of exploring sibling branches of the root problem first!
    - `Maximal-K first (deepest first)`: proved `47 / 67` (`5` depth $\ge 1$ proofs, `total W = 116`).
- **What it showed**: Prediction 1 held strongly (`46/67` with all prior theorems vs `22/67` self-bootstrapped vs `12/67` axioms-only), and E4.2 revealed an unpredicted property of `Ex(CK)` on page 50: because `CKG 2 1 - C` branches only when $K(2) > K(1)$, it breaks ties in $K$ **last-in-first-out (depth-first)** rather than first-in-first-out (breadth-first), which reduces work on the 4 multi-step proofs it finds (`W=53` vs `W=179` for FIFO) at the cost of missing 3 theorems (`46/67` vs `49/67`) that FIFO reaches within $W \le 50$.

## 4. How it works

1. **Goal-directed backward search over expression trees (Sections I–II, pp. 8–12, 29–46)**: Every propositional formula in *Principia Mathematica* is represented as a binary tree of elements $E = (G, V, F, C, N, P, A, U)$ with primitive connectives `~` (stored as non-negative integer negation count $G$ on each node), `\/` (`v`), and `->` (interchangeable with `~... \/ ...` via `*1.01`). Starting from a target theorem $X$, the Executive routine `Ex` attempts `MSb` (substitution + replacement), `MDt` (detachment backward from $B \to A'$ to subproblem $B$), and `MCh` (forward chaining from $A \to B$ to $B \to C$, or backward chaining from $B \to C$ to $A \to B$) (E1, E2).
2. **Coarse structural filtering (`CSm`) before tree unification (`LMc`) (pp. 31–37, 53, 57–59)**: Before attempting recursive point-by-point matching (`LMc`), LT computes a 3-integer description $D = (K, J, H)$ (levels $K$, distinct unit names $J$, total unit places $H$) for the left and right subtrees of each candidate (`DL` and `DR`). Across the 67 theorems of Chapter `*2`, `CSm` filters out `93.6%` of candidate pairs (`5,206 / 81,416` pass), reducing `LMc` invocations by $13.1\times$ (`5,726` vs `75,246`), at the cost of rejecting matches where variable coalescing changes $J$ (such as `*2.07: p -> (p \/ p)` against `*1.3: B -> (A \/ B)`, where $J_R = 1 \ne 2$) or molecular substitution changes $(K, H)$ (E3.1).
3. **Perceptual chunking via Contraction (`VCt`) in Detachment (pp. 39–42, 56)**: When `MDt(SmV)` (viewing atomic variables as units `VV`) finds no similar theorem, `MDt(SmCt)` runs `VCt` on the target problem, collapsing bottom-level binary subexpressions (such as `p -> q`, `q -> r`, and `p -> r` in `*2.06`) into single units (`U = True`). This changes `*2.06`'s description from `DL=(2,2,2), DR=(3,3,4)` to `DL=(1,1,1), DR=(2,2,2)`, unlocking `9` additional detachment proofs (`33/67` vs `24/67` for `MSb + MDt`, and `46/67` vs `38/67` overall, E3.2).
4. **Subproblem agenda (`P`), duplicate elimination (`Q` / `CX`), and theorem-memory accumulation (`T`) (pp. 45–48, 50)**: Unsolved detachment and chaining subproblems are pushed onto active list $P$. When a problem fails `MSb`, `MDt`, and `MCh`, `Ex` moves it to inactive list $Q$, checks global work $W \le W_{\text{limit}}$ (`CWG`), selects from $P$ the subproblem with minimal level $K$ (`Ex(CK)`), drops it if `CX` matches any expression on $Q$, and repeats. Accumulating prior theorems in $T$ is essential to LT's reach: under $W_{\text{limit}} = 50$, LT proves `46/67` theorems (`38/52` on the first 52) when all prior theorems are in $T$, `22/67` (`19/52`) when only its own previously proved theorems are added to $T$, and only `12/67` (`11/52`) from the 5 axioms `*1.2..*1.6` alone (E4.1, E4.2).

## 5. What surprised me and what I got wrong

1. **Why the 1956 IPL-I code proves `*2.06` via `*1.5 + *1.6` instead of the paper's prose example `*2.04 + *2.05` (E2, E3.2)**: On pages 41–42 of `P-868`, Newell & Simon walk through proving `*2.06: (p -> q) -> ((q -> r) -> (p -> r))` by contracting `*2.06` with `VCt` to `DL=(1,1,1), DR=(2,2,2)`, matching the right side of `*2.04: (A -> (B -> C)) -> (B -> (A -> C))`, and proving the left side by `*2.05`. I predicted in `work/predictions.txt` that running `MDt` with `VCt ON` would return `MDt(*2.04) + MSb(*2.05)`. Instead, both `dmoews/logic-theorist` (interpreting the 1956 IPL-I routines) and our `LogicTheorist1956` returned **`MDt(*1.5) + MSb(*1.6)`**! Why? Because Axiom `*1.5: (A \/ (B \/ C)) -> (B \/ (A \/ C))` is stored in list $T$ *before* `*2.04`, its right side `B \/ (A \/ C)` also has `DL=(1,1,1), DR=(2,2,2)` (since $D=(K,J,H)$ does not record whether a connective is `\/` or `->`), and `LMc` uses `LRpv->` (`*1.01`) to turn `B \/ (A \/ C)` into `(p -> q) -> ((q -> r) -> (p -> r))` by substituting `~B` and `~A`, leaving subproblem `~(q -> r) \/ (~(p -> q) \/ (p -> r))` which matches Axiom `*1.6`! In the later 1963 IPL-V version (`dmoews/ipl-v-logic-theorist`, E1), sublevel replacement is separated from `LMc`, so `*1.5` does not match directly and `*2.06` is proved via `*2.04 + *2.05` as in the prose.
2. **Why `CSm` rejects `*2.07: p -> (p \/ p)` against Axiom `*1.3: B -> (A \/ B)` in `MSb` (E3.1)**: When comparing `dmoews` (`20` `PROOF_SUB` proofs) with our `LogicTheorist1956` (`19` `MSb` proofs with `CSm ON`, `20` with `CSm OFF`), we found that `*2.07: p -> (p \/ p)` has $J_R = 1$ distinct variable on the right, whereas `*1.3: B -> (A \/ B)` has $J_R = 2$. Because `CSm` requires $J_R(x) = J_R(y)$, `MSb` rejects `*1.3` on `*2.07`! With `CSm ON`, `*2.07` is proved only later inside `MCh`'s segment `E`/`G` (`MCh_direct`), because `MCh(SmF)` checks `CSm` only on the left subexpressions (`p` vs `B`, both $J_L=1$) and then matches the right subexpressions via `LMc` without calling `CSm`.
3. **LIFO tie-breaking in `Ex(CK)` via `CKG 2 1 - C` (E4.2)**: On page 50 of `P-868`, the minimum-$K$ loop in `Ex(CK)` uses `CKG 2 1 - C` (branch to `C` only if $K(2) > K(1)$). Whenever a later subproblem on list $P$ has $K(2) == K(1)$, it overwrites working memory `1`, breaking ties **last-in-first-out (depth-first)** rather than first-in-first-out (breadth-first). Changing `<` to `<=` in `LogicTheorist1956` eliminated 3 of the 5 theorem differences against `dmoews/logic-theorist` (`65/67` agreement), and showed that pure `FIFO` actually proves `49/67` theorems vs `46/67` for `Ex(CK)` under $W_{\text{limit}}=50$.

## 6. Authors say / code does / I observed

- **Authors say (Section III, pp. 49–61, printed IPL-I routines) vs runnable IPL-I (`logic-routines.txt`)**: The printed 1956 code has 10 concrete bugs/typos (missing `PE 1 2` in `Ex`, inverted `TC->` sign in `MDt`, missing `D:` label and `- B` branch in `MSb`, inverted `TV` signs in `LMc`, dead `CPS 1 1 - B` check and missing `-TF` check in `LSb`, undefined `-TSb` opcode in `LRpv->`, formal parameter `x` instead of `L` in `D`, missing `D:` label in `NJ`, and inverted `CPS` signs in `SXM`/`SXE`/`SXL`/`SXR`) (E1).
- **Authors say (p. 31 Footnote 12: "about one theorem in five ... turns out to be similar") vs full 67-theorem campaign (E3.1)**: On early theorems (`*2.01..*2.08`), about $1/5$ ($20\%$) of candidates in $T$ pass `CSm`; averaged across all 67 theorems (including subproblem searches in `MDt` and `MCh` as $T$ grows to 72 entries), `CSm` passes `5,206 / 81,416` (`6.4%`) of checks.
- **Authors say (pp. 41–42, prose proof of `*2.06` via `*2.04 + *2.05`) vs 1956 IPL-I execution (E2, E3.2)**: The 1956 IPL-I program proves `*2.06` via `*1.5 + *1.6` because `*1.5` appears before `*2.04` in $T$ and `LMc` applies `LRpv->` during matching; the 1963 IPL-V program (`test-stefferud.out`, E1) proves `*2.06` via `*2.04 + *2.05`.

## 7. Not run, and why. Unknowns

- **JOHNNIAC IPL-II binary (August 1956)**: The original IPL-II machine code for the RAND JOHNNIAC vacuum-tube computer was not published in `P-868` (which published the IPL-I specification); we ran the repaired 1956 IPL-I specification (`dmoews/logic-theorist`), the 1963 Stefferud IPL-V specification (`dmoews/ipl-v-logic-theorist`), and our own Python implementation (`lt1956_engine.py`).

## 8. Sources and Sources read

### Primary paper and repositories
- **Primary paper**: Allen Newell and Herbert A. Simon (July 12, 1956, Revised), *The Logic Theory Machine: A Complex Information Processing System*, RAND Corporation Report P-868, 64 pages (published in *IRE Transactions on Information Theory*, Vol. IT-2, No. 3, pp. 61–79, September 1956). Retrieved 2026-10-26 from `http://bitsavers.trailing-edge.com/pdf/rand/ipl/P-868_The_Logic_Theory_Machine_Jul56.pdf` (3,534,001 bytes, 64 pages) and Archive.org full text `https://archive.org/download/bitsavers_randiplP86ineJul56_3534001/P-868_The_Logic_Theory_Machine_Jul56_djvu.txt` (`work/p868_1956_full.txt`, 4,874 lines, 99,608 bytes) plus verbatim Section III transcription `work/transcription.txt` (701 lines).
- **Reference repo 1 (1956 IPL-I)**: `https://github.com/dmoews/logic-theorist.git`, commit `07fff3586ef2136a71515ef78e7a1c6609538bc0` (CC0 public domain).
- **Reference repo 2 (1963 Stefferud IPL-V)**: `https://github.com/dmoews/ipl-v-logic-theorist.git`, commit `89c6d79ac2a25d7566183b3136bfbe198c9f75dd` (CC0 public domain).
- **Reference repo 3**: `https://github.com/duplys/logic-theorist.git`, commit `cc75327fc941560902d734f5d89cc02c3a58b561` (MIT License).
- **Reference repo 4 (IPL-V companion archive)**: `https://github.com/jblattnernyc/logic-theorist-companion.git`, commit `e4e8415b287803ab52058f78a028ad135898f290`.

### Sources read (READ-LOG)
| File | Lines | Status | Why |
| --- | ---: | --- | --- |
| `work/p868_1956_full.txt` (Newell & Simon 1956, pp. 1–64) | 4874 | `full` | Read Sections I and II (`pp. 1–50`, lines 1–2400) in full and Section III (`pp. 49–63`) via `transcription.txt`. |
| `work/transcription.txt` (P-868 Section III, pp. 49–63) | 701 | `full` | Verbatim transcription of the complete 1956 IPL-I program and Part 2 reductions. |
| `dmoews-logic-theorist/README.md` | 147 | `full` | Architecture, Principia Mathematica `*1`/`*2` rules, and bibliography `[1]..[8]`. |
| `dmoews-logic-theorist/orig-logic-routines.txt` & `logic-routines.txt` | 1080 | `full` | Original vs repaired 1956 IPL-I code (diffed and read in full). |
| `dmoews-logic-theorist/pm-axioms.txt` & `pm-theorems.txt` | 151 | `full` | All 5 axioms (`*1.2..*1.6`) and 67 theorems (`*2.01..*2.86`). |
| `dmoews-logic-theorist/run_logic.py` | 64 | `full` | Sequential theorem-proving driver. |
| `dmoews-logic-theorist/logic.py` | 853 | `skimmed` | Inspected `Context`, `Working`, `work_limit = 50`, `CPS`, `CWG`, `NAW`, `PROOF_*`. |
| `dmoews-ipl-v-lt/README.md` & `interpreter/test-stefferud.out` | 1032 | `full` | 1963 Stefferud IPL-V Logic Theorist execution trace across `*2`, `*3`, `*4`. |
| `duplys-logic-theorist/logic_theorist.py` | 134 | `full` | Educational stub implementation. |

## 9. Files

### Local worktree (`C:\personal\oss\clover-framework-foundations\clover-kimi-k3\` on branch `foundations-research`)
- Context file: `docs/newell-simon-1956-context.md`
- Predictions log: `research/newell-simon-1956/work/predictions.txt`
- Primary paper text & Section III transcription:
  - `research/newell-simon-1956/work/p868_1956_full.txt`
  - `research/newell-simon-1956/work/transcription.txt`
- Own implementations:
  - `research/newell-simon-1956/work/lt1956_engine.py` (Expression tree `Expr`, parser, `VV`, `VCt`, `D=(K,J,H)`, `CSm`, `LSb`, `LRp->v`, `LRpv->`, `LMc`)
  - `research/newell-simon-1956/work/lt1956_methods.py` (`CX`, `MSb`, `MDt`, `MCh`, and `Ex` with configurable `use_csm`, `use_vct`, `priority_mode`, `work_limit`)
  - `research/newell-simon-1956/work/run_e2_e3_e4.py` (Driver for E2 67-theorem campaign verification, E3 `CSm`/`VCt` ablations, E4 theorem-memory learning & subproblem priority ablations, and per-theorem disagreement check)
- Scripts:
  - `research/newell-simon-1956/scripts/e0_setup.sh`
  - `research/newell-simon-1956/scripts/e0b_fetch_lt1956.sh`
  - `research/newell-simon-1956/scripts/e0c_inspect_dmoews_and_paper.sh`
  - `research/newell-simon-1956/scripts/e0d_get_p868_text.sh`
  - `research/newell-simon-1956/scripts/e1_repos_and_errata.sh`
  - `research/newell-simon-1956/scripts/e2_inspect_pm.sh`
  - `research/newell-simon-1956/scripts/e2_e3_e4_run.sh`
- Results:
  - `research/newell-simon-1956/results/e0_setup.out`
  - `research/newell-simon-1956/results/e0b_fetch_lt1956.out`
  - `research/newell-simon-1956/results/e0c_inspect_dmoews_and_paper.out`
  - `research/newell-simon-1956/results/e0d_get_p868_text.out`
  - `research/newell-simon-1956/results/e1_repos_and_errata.out`
  - `research/newell-simon-1956/results/e2_inspect_pm.out`
  - `research/newell-simon-1956/results/e2_e3_e4_run.out`

### Remote (`k3` CPU server)
- `/root/dm-study/foundations/newell-simon-1956/` (`repo/dmoews-logic-theorist`, `repo/dmoews-ipl-v-lt`, `repo/duplys-logic-theorist`, `repo/jblattnernyc-logic-theorist-companion`, `work/`)

