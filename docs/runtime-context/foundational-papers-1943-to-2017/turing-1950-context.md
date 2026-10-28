# Alan M. Turing (1950) — *Computing Machinery and Intelligence* — Context

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh.

## 1. Status

- **E1 (Clone & run reference repositories on CPU)**: DONE
- **E2 (Sections 4 & 5: Stored-program digital computer with 10-digit packets `6809430217`, self-modifying code, discrete-state wheel machine, and universality)**: DONE
- **E3 (Sections 1, 2 & 6: Imitation Game interrogator vs honest/spoofing machines, and the Mathematical Objection diagonal machine sequence)**: DONE
- **E4 (Section 7: Learning machines — reward/punishment bit-bound vs unemotional symbolic channel, epistemic statuses & imperatives, and `[50, 200]` digit-sum-squared search)**: DONE

## 2. What this is

Authors say (A. M. Turing, "Computing Machinery and Intelligence", *Mind*, Vol. 59, No. 236, pp. 433–460, October 1950):
- "I propose to consider the question, 'Can machines think?' ... Instead of attempting such a definition I shall replace the question by another, which is closely related to it and is expressed in relatively unambiguous words. The new form of the problem can be described in terms of a game which we call the 'imitation game.'" (Section 1, p. 433 / p. 1 of extract)
- Section 3 (p. 3) restricts the machines in the game to **digital computers**, defined in Section 4 (pp. 4–6) as consisting of three parts—(i) **Store**, (ii) **Executive unit**, and (iii) **Control**—where instructions are stored in the Store alongside data (e.g., as 10-digit packets like `6809430217` meaning "Add the number stored in position 6809 to that in 4302 and put the result back into the latter storage position", plus unconditional and conditional jumps).
- Section 5 (pp. 6–8) defines **discrete-state machines** (illustrated by a 3-state clicking wheel $q_1, q_2, q_3$ with lever input $i_0, i_1$ and lamp output $o_0, o_0, o_1$), defines **storage capacity** as $\log_2(\text{number of states})$ (e.g., $165,000$ or $174,380$ binary digits for the Manchester machine), and states the **Universality** property: a single digital computer with adequate storage capacity and speed can be programmed to mimic the state table of any discrete-state machine.
- Section 6 (pp. 8–17) states Turing's conjecture (in ~50 years, a computer with storage capacity $\sim 10^9$ bits will play the imitation game so that an average interrogator has $\le 70\%$ chance of right identification after 5 minutes) and analyzes nine objections: (1) Theological, (2) "Heads in the Sand", (3) Mathematical (Gödel 1931, Church 1936, Turing 1937 halting/diagonal questions), (4) Consciousness (Jefferson 1949 and the sonnet *viva voce*), (5) Various Disabilities (distinguishing **errors of functioning** from **errors of conclusion**, arithmetic mistake spoofing, and self-referential programs), (6) Lady Lovelace's Objection ("can only do whatever we know how to order it to perform" vs machines taking us by surprise), (7) Continuity in the Nervous System (mimicking a continuous differential analyser on $\pi \approx 3.1416$ by stochastic rounding), (8) Informality of Behaviour (rules of conduct vs laws of behaviour, and a 1,000-bit Manchester program mapping 16-digit numbers), and (9) Extrasensory Perception.
- Section 7 (pp. 17–22) proposes **Learning Machines**: dividing the problem into a **child programme** ("like a notebook as one buys it from the stationer's. Rather little mechanism, and lots of blank sheets") and an **education process** combining (a) punishments and rewards, (b) an "unemotional" symbolic communication channel (since $R/P$ alone transfers at most 1 bit per blow/reward), (c) built-in logical inference with proposition statuses (`well-established facts`, `conjectures`, `mathematically proved theorems`, `statements given by an authority`, `no belief-value`, and `imperatives` such as "Do your homework now" or search heuristics like "When Socrates is mentioned, use the syllogism in Barbara"), and (d) random vs systematic search (illustrated by finding $n \in [50, 200]$ equal to the square of the sum of its digits).

## 3. Experiments

### E1: Clone and run reference repositories on the GPU server
- **Question**: What files, line counts, and implementations ship in `vukrosic/turing-from-scratch` and `simonbreiter/universal-turing-machine`, and what do they output when executed?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 14:48 UTC):
  1. `vukrosic/turing-from-scratch` will contain standalone Python scripts implementing Turing's core concepts (Turing machine, universal machine, halting problem, imitation game, and a learning machine) and will run cleanly on CPU.
  2. `simonbreiter/universal-turing-machine` will implement a Universal Turing Machine simulator in Python that decodes an encoded machine description and simulates it on a tape with exit code 0.
- **Run**: `scripts/e1_repos.sh`, `scripts/e1b_run_k3.sh`, `scripts/e1c_utm_endstate.sh`, and `scripts/e2_e3_e4_all.sh`. Outputs: `results/e1_repos.out`, `results/e1b_run_k3.out`, `results/e1c_utm_endstate.out`, `results/e2_e3_e4_all.out`.
- **Observation**:
  - **Repository 1 (`vukrosic/turing-from-scratch`)**:
    - Commit `a578078da5aaf47c28ba31fa357ea9e1be8cbec9` (2026-10-05, MIT License, 14 Python files).
    - Ran all scripts (`step1_machine.py` through `step9_learning.py` and `tests.py`, `results/e1b_run_k3.out`, `rc=0`, `all tests pass`):
      - `step1_machine.py`: 2-state Turing machine `ADD_ONE` computed `1011 + 1 = 1100` in `8` steps (`0+1=1` in 3 steps, `111+1=1000` in 8 steps, `100111+1=101000` in 11 steps).
      - `step2_universal.py`: encoded `ADD_ONE`'s 6 rules into a 253-digit integer (`327662666...`), decoded it back into a rule table, and simulated `1011 -> 1100` in `8` steps.
      - `step3_beaver.py`: ran the 2-, 3-, 4-, and 5-state Busy Beaver champions (`6` steps / `4` ones, `21` steps / `5` ones, `107` steps / `13` ones, and `47,176,870` steps / `4,098` ones in `2.8 s`).
      - `step4_halting.py`: tested `make_opposite(judge)` against `always_yes` (said `stops`, program ran `1,000,000+` steps), `always_no` (said `runs forever`, program stopped after `0` steps), and bounded simulator `wait_and_see` (recursed `333` times until `RecursionError`).
      - `step8_imitation.py`: checked Turing's Section 2 arithmetic question `34957 + 70764 = 105721` vs the witness's answer `105621` in the 1950 paper (off by `100`, an intentional carry error).
      - `step9_learning.py`: simulated a 16-unit random NAND network (Turing's 1948 B-type unorganized machine, entering a 2-step cycle after 5 transient steps) and aSection 7 child machine learning 21-stick Nim via +1/-1 rewards and punishments (win rate vs perfect expert rose from `0.2%` at 0 games to `0.5%` at 100 games, `95.4%` at 1,000 games, and `99.6%` at 10,000 games, learning the `pile % 4` action on `16/16` winning piles).
  - **Repository 2 (`simonbreiter/universal-turing-machine`)**:
    - Commit `6fe454ff132e05c15c8ca28bc1ed96205417a393` (2018-10-22, MIT License, 4 Python files totaling 202 lines plus 4 JSON instruction tables in `assets/instructions/`).
    - `readme_example.json` ran in `3` steps producing `00`; `multiplication.json` on `--tape "00 000"` ($2 \times 3$ in unary) ran in `91` steps producing `000000` ($6$ zeros).
    - **Finding on `universal.json`**: Running `universal.json` with the default `--end qdone` failed (`Invalid config! State "q82" needs to be defined!`). Inspecting the JSON (`e1c_utm_endstate.sh`) showed `universal.json` has `83` defined states and uses `q82` as its halt state (`Undefined nextStates: ['q82']`). Furthermore, passing a self-looping encoded rule `01000101000100` ($q_1$ on blank $\to q_1$, write blank, move right) ran forever (`rc=124`), whereas passing a 1-step halting encoded machine `0101001001001110` ($q_1$ on `0` $\to q_2$, write `1`, move right, followed by separator `111` and input tape `0`) halted cleanly at `q82` after `285` UTM steps with `Output: 1` (`results/e2_e3_e4_all.out`).
- **What it showed**: Both predictions held, and diagnosing the `q82` halt-state requirement and separator `111` encoding in `simonbreiter/universal-turing-machine` showed how a single fixed 83-state transition table (`universal.json`) interprets an arbitrary machine's unary transition table written on the same tape.

### E2: Sections 4 & 5 — Stored-program digital computer (`6809430217`), self-modifying code, discrete-state wheel machine, and universality
- **Question**: How does Turing's Section 4 digital computer (Store, Executive unit, Control with 10-decimal-digit instruction packets `AAAABBBBCC` such as `6809430217`) execute arithmetic, conditional jumps, and self-modifying instruction tables (Section 6.5, p. 14), and does programming it with the state/output table of Section 5's 3-state clicking wheel reproduce the wheel's exact state and lamp output across all $2^8 = 256$ input sequences?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 14:48 UTC):
  1. Packet `6809430217` decodes to `A=6809, B=4302, op=17`, adds `store[6809]=34957` to `store[4302]=70764` yielding `105721` (`100` higher than the paper witness's `105621`), and a 5-instruction loop that adds `1,000,000` (`+1` in the `AAAA` address field) to its own instruction packet at `store[1000]` sums an array in store without an index register.
  2. Programming `Turing1950DigitalComputer` with the transition and output tables of Section 5's 3-state clicking wheel ($q_1, q_2, q_3$, lever $i_0, i_1$, lamp $o_0, o_0, o_1$) will match the wheel on `256/256` 8-step input sequences, whereas corrupting one table entry `(q1, i1)` will fail on `255/256` sequences.
  3. Section 5 storage capacities (`log2(3) = 1.58496` bits, 100 sheets $\times 50 \times 30$ digits $= 498,289.2$ bits $\approx 3.02\times$ Manchester's $165,000$, and $64 \times 2560 + 8 \times 1280 + 300 = 174,380$ bits) will match Turing's figures on page 7.
- **Run**: `scripts/e2_e3_e4_all.sh` (executing `work/e2_stored_program_and_universal.py`). Output: `results/e2_e3_e4_all.out`.
- **Observation**:
  - **10-digit packet & self-modifying loop**: `6809430217` decoded to `(6809, 4302, 17)` and updated `store[4302]` from `70764` to `105721`. In the self-modifying array-sum program, instruction `1001` added `1_000_000` to `store[1000]` on each iteration, mutating the instruction packet at `store[1000]` from `5000430217` through `5004430217` (ending at `5005430217`) and accumulating `10 + 20 + 30 + 40 + 50 = 150` in `store[4302]`.
  - **Universal mimicry of the Section 5 wheel machine**: Across all `256` binary input sequences `(i_0, i_1)^8`, the 9-instruction universal interpreter running on `Turing1950DigitalComputer` produced state and lamp traces `(q_t, o_t)` identical to `ClickingWheelMachine` on `256/256` sequences (`100%`), while the negative control (corrupting entry `(q1, i1)` in Store address `101`) matched on only `1/256` sequences (`00000000`).
  - **Storage capacity numbers (Section 5, p. 7)**: Wheel $\log_2(3) = 1.58496$ bits (Turing: "about 1.6"); 100 sheets of $50 \times 30$ decimal digits $= 150,000 \log_2(10) = 498,289.2$ bits $= 3.02 \times 165,000$ (Turing: "about the number of states of three Manchester machines put together"); Manchester tracks + CRT tubes + misc $= 163,840 + 10,240 + 300 = 174,380$ bits (exact match to Turing's `174,380`).
- **What it showed**: All three predictions held to the digit.

### E3: Sections 1, 2 & 6 — The Imitation Game, Strategic Mistake Spoofing, the Mathematical Objection, and Continuity
- **Question**: (a) In the Section 1–2 Imitation Game over 5 arithmetic questions, how often does a Bayes-optimal interrogator C distinguish Human B ($\mu=30\text{ s}, \sigma=6\text{ s}$, $18\%$ single-column carry slip) from an Honest Machine A1 ($1\text{ ms}$, $0\%$ error), a Turing Spoofing Machine A2 (Section 2 & 6.5: pauses $\sim 30\text{ s}$ and deliberately introduces carry slips like `34957 + 70764 -> 105621`), and a Faulty Machine A3 (Section 6.5: makes "unsuitable" non-carry mistakes due to a mechanical fault)? (b) On Section 6.3's Mathematical Objection (pp. 10–11), what happens when a sequence of bounded machines $M_0, M_1, M_2, M_3$ is asked each machine's critical diagonal question $Q_k$? (c) How closely does Turing's discrete stochastic table for $\pi$ on page 16 match a continuous differential analyser with Gaussian noise?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 14:48 UTC):
  1. Honest Machine A1 will be unmasked `~100%` of the time; Turing Spoofing Machine A2 will reduce the interrogator's right-identification rate to `~50%` (chance, well under Turing's `70%` 50-year target); and Faulty Machine A3 will be unmasked `>70%` of the time via its unsuitable errors.
  2. Every machine $M_k$ will fail (`TIMEOUT`) on its own diagonal question $Q_k$, while every machine $M_m$ with $m > k$ will answer $Q_k$ (`YES`) correctly, producing a strict lower-triangular success matrix.
- **Run**: `scripts/e2_e3_e4_all.sh` (executing `work/e3_e4_imitation_and_learning.py`). Output: `results/e2_e3_e4_all.out`.
- **Observation**:
  - **Imitation Game (2,000 games of 5 questions each)**:
    - Against `Honest Machine A1 (0.001s, 0% error)`: Interrogator right identification $= 100.0\%$.
    - Against `Turing Spoofing Machine A2 (~30s pause, 18% carry slip)`: Interrogator right identification $= 48.4\%$ (indistinguishable from $50\%$ chance, well below Turing's $\le 70.0\%$ target).
    - Against `Unsuitable-Error Machine A3 (~30s pause, non-carry mechanical mistakes)`: Interrogator right identification $= 72.6\%$ (unmasked whenever an unsuitable error occurs, matching Turing's remark on page 13: "A mechanical fault would probably show itself through an unsuitable decision as to what sort of a mistake to make in the arithmetic").
  - **Mathematical Objection diagonal matrix (`M_0..M_3` vs `Q_0..Q_3`)**:
    - Diagonal entries $(M_k, Q_k)$ all failed (`TIMEOUT (FAIL)` for $k=0,1,2,3$), while every sub-diagonal entry $(M_m, Q_k)$ with $m > k$ returned `YES (OK)`, confirming Turing's reply on page 10–11: each critical question defeats the specific machine it diagonalizes against, but is answered by the next machine.
  - **Continuity in the nervous system / differential analyser (Section 6.7, p. 16)**:
    - Turing's discrete distribution on $\{3.12, 3.13, 3.14, 3.15, 3.16\}$ with probabilities $[0.05, 0.15, 0.55, 0.19, 0.06]$ has mean $3.14060$ and standard deviation $0.00881$ (slightly asymmetric toward $3.15$ because $\pi \approx 3.14159 > 3.14000$!). A continuous Gaussian analyser $\mathcal{N}(\pi, 0.00881^2)$ rounded to 2 decimal places gave $[0.0303, 0.1966, 0.4251, 0.2841, 0.0640]$ (total variation distance $0.1447$).
- **What it showed**: Both predictions held, and checking the moments of Turing's hand-chosen probabilities $[0.05, 0.15, 0.55, 0.19, 0.06]$ on page 16 revealed that Turing deliberately gave $3.15$ higher probability ($0.19$) than $3.13$ ($0.15$) and $3.16$ ($0.06$) higher than $3.12$ ($0.05$) so that the expectation $3.1406$ sits above $3.1400$ toward $\pi \approx 3.1416$.

### E4: Section 7 — Learning machines: reward/punishment bound vs unemotional channel, epistemic imperatives, and `[50, 200]` search
- **Question**: (a) How many binary reward/punishment signals are required to teach the 33-character opening line of *Casabianca* (or the 1,600-character poem) by trial-and-error blows alone vs using an "unemotional" symbolic channel after learning a 27-symbol alphabet? (b) How does Turing's child machine with proposition statuses and imperatives (pp. 20–21) execute authority-derived commands and prune logical inference? (c) In Turing's concrete search problem on page 22 ("find a number between 50 and 200 which was equal to the square of the sum of its digits"), what is the solution and when does memoryless random search beat systematic search?
- **Prediction** (written before running, logged in `work/predictions.txt` at 2026-10-26 14:48 UTC):
  1. Pure binary R/P requires $(V+1)/2 \cdot L = 14 L$ expected blows/rewards (`462` for 33 chars, `22,400` for 1,600 chars), whereas learning the $V=27$ symbol mapping once (`202` R/P signals) allows any length-$L$ text to be obeyed via the unemotional symbolic channel with `1` final reward.
  2. Combining `"Teacher says 'Do your homework now'"` (`well_established`) with `"Everything that teacher says is true"` (`well_established`) promotes `"Do your homework now"` from `no_belief_value` to `well_established` and triggers execution; and applying the search-order imperative `"When Socrates is mentioned, use the syllogism in Barbara"` reduces proof search from `90` steps to `1` step.
  3. The unique number in `[50, 200]` equal to the square of the sum of its digits is `81` ($(8+1)^2 = 81$). When a dead block of 120 non-solutions precedes 31 solutions, systematic search takes `121` steps while memoryless random search averages `151 / 31 = 4.87` steps.
- **Run**: `scripts/e2_e3_e4_all.sh` (executing `work/e3_e4_imitation_and_learning.py`). Output: `results/e2_e3_e4_all.out`.
- **Observation**:
  - **Reward/punishment vs unemotional symbolic channel**: Pure R/P required `140.0` expected blows/rewards for `"CASABIANCA"` ($L=10$), `462.0` for `"THE BOY STOOD ON THE BURNING DECK"` ($L=33$), and `22,400.0` for the 1,600-character poem, whereas learning the 27-symbol command language took `202` expected R/P signals once and `1` reward per text of any length thereafter.
  - **Epistemic statuses & imperatives**: `"Do your homework now"` transitioned from `no_belief_value` to `well_established` and fired automatically; proving `"Socrates is mortal"` among 10 axioms took `90` steps without the search-order imperative vs `1` step with the Barbara imperative.
  - **Digit-sum-squared search in `[50, 200]`**: Exhaustive check found the unique solution `81` ($8+1=9, 9^2=81$). For the single solution `81`, systematic upward search from 51 took `31` steps, systematic downward search across the dead block `200..82` took `120` steps, and memoryless random sampling averaged `153.0` steps; in Turing's multi-solution dead-block regime (`120` dead states followed by `31` satisfactory solutions), systematic search took `121` steps while memoryless random search averaged `4.87` steps (matching theoretical $151/31 = 4.87$).
- **What it showed**: All three predictions held, and testing both the single-solution case ($k=1$, where random search averages $151$ steps vs $31$ or $120$ systematic) and the multi-solution case ($k=31$, where random search averages $4.87$ steps vs $121$ systematic) verified the exact condition Turing stated on page 22: *"Since there is probably a very large number of satisfactory solutions the random method seems to be better than the systematic."*

## 4. How it works

1. **Operationalizing "Can machines think?" via the Imitation Game (Sections 1–3)**: Instead of surveying common word usage, Turing defines a 3-player teleprinter game (interrogator C, witness A, witness B) that strips away physical appearance and voice, and replaces the question with whether a digital computer programmed as A causes C to misidentify A and B as often as when A is a human man and B a human woman (E1, E3).
2. **Stored-program digital computers and Universality (Sections 4–5)**: A digital computer consists of a **Store**, an **Executive unit**, and a **Control** obeying an instruction table stored as data packets in the Store (such as `6809430217`). Because instructions live in the Store, a program can arithmetic-modify its own instruction addresses (E2.1) and can store and interpret the state-transition table of **any** discrete-state machine (such as the 3-state clicking wheel of capacity $\log_2 3 \approx 1.585$ bits, E2.2, or an encoded Turing machine tape in `universal.json`, E1), making all digital computers with sufficient storage capacity universal and formally equivalent (E1, E2).
3. **Behavioral indistinguishability vs internal mechanism (Section 6)**:
   - Because the Imitation Game evaluates external teleprinter behavior, a machine must mask its superhuman arithmetic speed and zero carry-error rate by pausing (`~30 s`) and introducing human-like **errors of conclusion** (such as `34957 + 70764 = 105621` on p. 2, lowering interrogator accuracy from `100.0%` to `48.4%`), whereas **errors of functioning** (mechanical faults producing non-human error patterns) unmask the machine (`72.6%`, E3.1).
   - Similarly, a discrete-state machine can mimic a continuous noisy system (like a differential analyser estimating $\pi$) within teleprinter precision by sampling from a discrete probability table (`[3.12..3.16]` with probabilities `[0.05, 0.15, 0.55, 0.19, 0.06]`, mean $3.1406$, E3.3).
   - Gödel/Turing diagonalization (Section 6.3) guarantees that each fixed machine $M_k$ fails on its own critical question $Q_k$, but does not establish a simultaneous triumph over all machines because another machine $M_{k+1}$ answers $Q_k$ correctly (E1, E3.2).
4. **Learning machines: Child machine + Education + Imperatives + Random search (Section 7)**:
   - Rather than hand-coding an adult mind ($60$ workers $\times 50$ years at $1,000$ digits/day), a small **child programme** can be educated via rewards/punishments (which alone convey at most 1 bit per signal, requiring `22,400` signals for *Casabianca* and `10,000` self-play games to learn 21-stick Nim in E1/E4.1) combined with an **unemotional symbolic channel** for language orders (E4.1).
   - Inside a logical child machine, propositions carry explicit epistemic status (`well-established`, `conjecture`, `proved theorem`, `given by authority`, `no belief-value`), and domain-level **imperatives** regulate which inference rule to try first (reducing syllogism search from `90` steps to `1` step without changing logical soundness, E4.2).
   - When searching a behavioral space that has large empty blocks and many satisfactory solutions, memoryless random search ($N/k = 4.87$ steps) avoids getting trapped in an initial dead block (`121` steps systematic, E4.3).

## 5. What surprised me and what I got wrong

1. **Turing's arithmetic witness `34957 + 70764 = 105621` is intentionally wrong by 100 (E1, E2, E3)**: On page 2 (Section 2), Turing's specimen witness waits 30 seconds and answers `105621` to `34957 + 70764`, whereas the true sum is `105721` (a single carry slip in the hundreds column). Twelve pages later (Section 6.5, p. 13), Turing explains that a machine programmed for the imitation game *"would not attempt to give the right answers to the arithmetic problems. It would deliberately introduce mistakes in a manner calculated to confuse the interrogator."* Running `step8_imitation.py` (E1) and our Section 4 packet `6809430217` (E2.1) and Bayesian interrogator simulation (E3.1) connected the specimen dialogue on page 2 directly to the anti-unmasking strategy on page 13.
2. **Why `simonbreiter/universal-turing-machine` failed on `universal.json` with default arguments (E1)**: Running `universal.json` with the default `--end qdone` threw `Invalid config! State "q82" needs to be defined!`. Inspecting `universal.json` showed that the 83-state universal table uses `q82` as its unique undefined halting target (`--end q82`), and passing a 1-state halting guest machine (`0101001001001110`) halted at `q82` in `285` steps with output `1`.
3. **Why Turing emphasized "many satisfactory solutions" for random search on page 22 (E4.3)**: On Turing's concrete example ($n \in [50, 200]$ equal to the square of the sum of its digits), there is only **one** solution (`81`), so memoryless random sampling with replacement averages `153.0` trials—worse than both upward (`31` steps) and downward (`120` steps) systematic search! Only when we tested the second half of Turing's paragraph (*"Since there is probably a very large number of satisfactory solutions the random method seems to be better than the systematic"*, e.g. 31 solutions after a 120-element dead block) did random search drop to `4.87` steps vs `121` systematic steps.

## 6. Authors say / code does / I observed

- **Authors say (Section 2, p. 2) vs true sum**: Witness gives `105621` for `34957 + 70764`; exact sum is `105721` (intentional carry slip, E1, E2).
- **Authors say (Section 7, p. 22, random search on `[50, 200]`) vs single-solution count**: In `[50, 200]`, $n = (\text{digit sum})^2$ has only 1 solution (`81`), where random sampling with replacement averages $151$ trials; random sampling beats systematic search only when either multiple solutions exist ($k \ge 2$ after a dead block) or the dead block exceeds $N/k$ (E4.3).
- **`vukrosic/turing-from-scratch` (`step10_patterns.py`)**: Implements a FitzHugh–Nagumo-style reaction-diffusion grid (Turing's 1952 morphogenesis paper, not the 1950 paper), explicitly noted in its `README.md`.

## 7. Not run, and why. Unknowns

- **Turing's 1,000-bit Manchester challenge program (Section 6.8, pp. 16–17)**: Turing writes: *"I have set up on the Manchester computer a small programme using only 1,000 units of storage, whereby the machine supplied with one sixteen-figure number replies with another within two seconds."* The paper does not give the code or function of that Manchester routine, so the specific 16-digit transformation remains unknown.
- **Turing's "unorthodox" child-machine experiment (Section 7, p. 20)**: Turing writes *"I have done some experiments with one such child machine, and succeeded in teaching it a few things, but the teaching method was too unorthodox for the experiment to be considered really successful"*, without publishing the circuit or table of that experiment.

## 8. Sources and Sources read

### Primary paper and repositories
- **Primary paper**: A. M. Turing (1950), "Computing Machinery and Intelligence", *Mind*, Vol. 59, No. 236, pp. 433–460. Retrieved 2026-10-26 from `https://courses.cs.umbc.edu/471/papers/turing.pdf` (91,972 bytes, 22 pages; extracted text `work/turing1950.txt`, 865 lines, 68,538 bytes).
- **Reference repo 1**: `https://github.com/vukrosic/turing-from-scratch.git`, commit `a578078da5aaf47c28ba31fa357ea9e1be8cbec9` (2026-10-05), MIT License.
- **Reference repo 2**: `https://github.com/simonbreiter/universal-turing-machine.git`, commit `6fe454ff132e05c15c8ca28bc1ed96205417a393` (2018-10-22), MIT License.

### Sources read (READ-LOG)
| File | Lines | Status | Why |
| --- | ---: | --- | --- |
| `work/turing1950.txt` (Turing 1950, Sections 1–7, pp. 1–22) | 865 | `full` | Complete 1950 paper text. |
| `vukrosic-turing/README.md` | 42 | `full` | Overview of the 10 scripts and simplifying assumptions. |
| `vukrosic-turing/step1_machine.py` | 45 | `full` | Binary increment Turing machine (`ADD_ONE`). |
| `vukrosic-turing/step2_universal.py` | 37 | `full` | Rule-table Gödel/byte integer encoding and universal execution. |
| `vukrosic-turing/step3_beaver.py` | 41 | `full` | 2-, 3-, 4-, and 5-state Busy Beaver tables and runner. |
| `vukrosic-turing/step4_halting.py` | 76 | `full` | Diagonal halting problem (`make_opposite`) on 3 judges. |
| `vukrosic-turing/step8_imitation.py` | 11 | `full` | `34957 + 70764` check (`105721` vs `105621`) and $10^9$-bit capacity. |
| `vukrosic-turing/step9_learning.py` | 125 | `full` | 1948 random NAND net and 1950 reward/punishment Nim child machine. |
| `vukrosic-turing/tests.py` | 61 | `full` | Unit test suite across all steps. |
| `simonbreiter-utm/README.md` | 246 | `full` | JSON transition format and unary Universal Turing Machine encoding rules. |
| `simonbreiter-utm/src/universal_turing_machine.py` | 129 | `full` | `TuringMachine` simulator and validator. |
| `simonbreiter-utm/src/config.py` & `utils.py` | 46 | `full` | Tape helpers and configuration constants. |
| `simonbreiter-utm/assets/instructions/universal.json` | 2137 | `skimmed` | Inspected state set (`83` states, halting state `q82`) via script and executed on encoded guest machine. |

## 9. Files

### Local worktree (`C:\personal\oss\clover-framework-foundations\clover-kimi-k3\` on branch `foundations-research`)
- Context file: `docs/turing-1950-context.md`
- Predictions log: `research/turing-1950/work/predictions.txt`
- Extracted paper text: `research/turing-1950/work/turing1950.txt`
- Own implementations:
  - `research/turing-1950/work/e2_stored_program_and_universal.py` (Section 4 10-digit packet computer, `6809430217` execution, self-modifying array sum, Section 5 3-state wheel and universal mimicry, storage capacities)
  - `research/turing-1950/work/e3_e4_imitation_and_learning.py` (Imitation game Bayesian interrogator vs honest/spoofing/faulty machines, Section 6.3 diagonal machine hierarchy, Section 6.7 differential analyser distribution, Section 7 R/P vs symbolic channel bound, epistemic imperatives, and `[50, 200]` digit-sum-squared search)
- Scripts:
  - `research/turing-1950/scripts/e0_setup.sh`
  - `research/turing-1950/scripts/e0b_extract.sh`
  - `research/turing-1950/scripts/e1_repos.sh`
  - `research/turing-1950/scripts/e1b_run_k3.sh`
  - `research/turing-1950/scripts/e1c_utm_endstate.sh`
  - `research/turing-1950/scripts/e2_e3_e4_all.sh`
- Results:
  - `research/turing-1950/results/e0_setup.out`
  - `research/turing-1950/results/e0b_extract.out`
  - `research/turing-1950/results/e1_repos.out`
  - `research/turing-1950/results/e1b_run_k3.out`
  - `research/turing-1950/results/e1c_utm_endstate.out`
  - `research/turing-1950/results/e2_e3_e4_all.out`

### Remote (`k3` CPU server)
- `/root/dm-study/foundations/turing-1950/` (`repo/vukrosic-turing`, `repo/simonbreiter-utm`, `work/`)

