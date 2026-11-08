# Context File — Paper 08: Oudeyer, Gottlieb & Lopes (2016) — *Intrinsic Motivation, Curiosity, and Learning: Theory and Applications in Educational Technologies*

- **Citation**: Pierre-Yves Oudeyer, Jacqueline Gottlieb & Manuel Lopes (2016). *Intrinsic Motivation, Curiosity, and Learning: Theory and Applications in Educational Technologies*. **Progress in Brain Research**, Vol. 229, Chapter 11, pp. 257–284 (`doi:10.1016/bs.pbr.2016.05.005`).
- **Source PDF / Extracted Text**:
  - Primary PDF URL: `https://www.pyoudeyer.com/oudeyerGottliebLopesPBR16.pdf`
  - Extracted Full Text (`82,812 bytes`): [research/curiosity-and-developmental-learning/papers_text/08-oudeyer-gottlieb-lopes-2016-educational.txt](../../../research/curiosity-and-developmental-learning/papers_text/08-oudeyer-gottlieb-lopes-2016-educational.txt)
- **Executable Experiment & Saved Results**:
  - Experiment Script: [research/curiosity-and-developmental-learning/08-oudeyer-gottlieb-lopes-2016-educational/run_educational_2016_experiments.py](../../../research/curiosity-and-developmental-learning/08-oudeyer-gottlieb-lopes-2016-educational/run_educational_2016_experiments.py)
  - Saved Results Artifact: [research/curiosity-and-developmental-learning/08-oudeyer-gottlieb-lopes-2016-educational/results/educational_2016_results.json](../../../research/curiosity-and-developmental-learning/08-oudeyer-gottlieb-lopes-2016-educational/results/educational_2016_results.json)

---

## Stage 1 — Context (Primary Source & Educational / Neurocomputational Architecture)

### 1.1 Closing the Causal Loop Between Curiosity and Learning (Figure 1A vs. Figure 1B, pp. 258, 266–268)
- **Unidirectional View (Figure 1A)**: Classical psychology and neuroscience experiments (Kang et al., 2009; Gruber et al., 2014; Stahl & Feigenson, 2015) studied a one-way causal link: encountering novelty/surprise/knowledge gaps induces a **curiosity state** (striatal + VTA + IFG activation), which enhances **hippocampal memory consolidation**.
- **Bidirectional Positive-Feedback Loop (Figure 1B)**: Oudeyer, Gottlieb & Lopes propose that **experiencing Learning Progress (`LP`) itself generates intrinsic reward and state curiosity** (`Learning -> LP Intrinsic Reward -> State Curiosity -> Hippocampal Memory Retention & Focused Exploration -> More Learning`).

### 1.2 Resolving the Neophilia vs. Neophobia Puzzle (Section 3.3, pp. 264–265)
Why do animals and infants exhibit both strong **neophilia** (seeking novel rooms/objects even across electrified grids, Hughes 2007) and strong **neophobia** (avoiding novel stimuli, Weizmann et al. 1971)? The authors integrate Gershman & Niv's (2015) contextual valence generalization with the **Learning Progress hypothesis**: novel stimuli that yield reducible prediction errors (`LP > 0`, intermediate subjective complexity) elicit approach, whereas stimuli so alien that prediction error cannot be reduced (`LP = 0`) elicit avoidance.

### 1.3 `ZPDES`: Zone of Proximal Development and Empirical Success in Intelligent Tutoring Systems (Figures 4 & 5, pp. 276–278)
In Section 10, the authors detail **Clement, Roy, Oudeyer & Lopes (2015)**, deployed with **$N = 400$ elementary schoolchildren (ages 7–8) across 11 schools** learning integer and decimal money decomposition across 4 exercise modalities (`M`: customer/1 object, `R`: merchant/1 object, `MM`: customer/2 objects, `RM`: merchant/2 objects) and 3 difficulty levels:
- An expert teacher specifies a coarse precondition graph ($A_1 \to A_2 \to A_3$, Fig. 5).
- **`ZPDES`** maintains an active Zone of Proximal Development (`ZPD`) and uses a multi-armed bandit to propose exercises that maximize each individual student's empirical `LP`.
- **Crucial Modality-Switching Rule (Figure 5, p. 278)**: When a student stalls on exercise $A_2$ (zero empirical progress), `ZPDES` temporarily deactivates $A_3$ and expands the `ZPD` laterally to an alternative modality $C_1/C_2$, then re-activates the stalled path once prerequisite fluency is built.

---

## Stage 2 — Direction (Testable Hypotheses)

1. **Experiment E1 (Figure 1A Unidirectional vs. Figure 1B Closed `LP`-Curiosity-Memory Loop)**:
   - Compare static novelty selection (`fig1a_unidirectional_static_novelty`) against the closed `LP -> State Curiosity -> Hippocampal Consolidation` loop (`fig1b_closed_lp_curiosity_memory_loop`) across a 4-stage prerequisite curriculum (`t0_trivial`, `t1_foundational`, `t2_intermediate_bridge`, `t3_advanced_synthesis`).
2. **Experiment E2 (`ZPDES` vs. Expert Hand-Crafted Branching vs. Fixed Linear Sequence on $N=400$ Heterogeneous Students)**:
   - Simulate $N=400$ learners across the $4 \times 3 = 12$ money-decomposition exercises (`M1..RM3`), where $45\%$ of learners have a cross-modality bottleneck on `R2` (requiring decimal addition fluency from `MM2` before `R2` can progress).
   - Test all three empirical claims from Clement et al. (2015, pp. 277–278): (1) higher post-test mastery across all 12 exercises, (2) higher maximum difficulty level reached, and (3) greater exercise diversity early in the session.

---

## Stage 3 — Execution & Stage 4 — Outcome (Quantitative Results)

Running [run_educational_2016_experiments.py](../../../research/curiosity-and-developmental-learning/08-oudeyer-gottlieb-lopes-2016-educational/run_educational_2016_experiments.py) produces:

### 4.1 Experiment E1: Figure 1A vs. Figure 1B Closed Feedback Loop (`16` seeds $\times 240$ steps)
| Architecture | `t0_trivial` | `t1_foundational` | `t2_intermediate` | `t3_advanced` | Final Curriculum Mastery |
|---|---|---|---|---|---|
| **`fig1a_unidirectional_static_novelty`** | `4.51%` | `4.51%` | **`87.21%`** (premature stall) | `3.78%` | `0.3239` |
| **`fig1b_closed_lp_curiosity_memory_loop`** | `13.31%` | **`30.52%`** | **`30.16%`** | **`26.02%`** | **`0.9463`** ($2.92\times$ higher) |

### 4.2 Experiment E2: `ZPDES` Intelligent Tutoring System on $N=400$ Students (`72` trials)
| Tutoring Policy | Mean Post-Test Score (All 12 Exercises) | Mean Max Difficulty Level Mastered (`1..3`) | Early Exercise Diversity (First 24 Trials) |
|---|---|---|---|
| **`fixed_linear_sequence`** | `0.3016` | `0.542` | `12.00` (unfiltered cycling) |
| **`expert_handcrafted_branching`** | `0.4838` | `2.203` | `3.13` |
| **`zpdes_lp_bandit` (Fig. 5)** | **`0.5015`** | **`2.922`** | **`4.27`** |

---

## Stage 5 — Growth (What We Learned & Critical Boundaries)

1. **Why Lateral `ZPD` Expansion on Zero Progress Is Essential (`2.922` vs. `2.203` Max Level)**:
   - Standard mastery-learning branching (`expert_handcrafted_branching`) only expands the curriculum when a student *succeeds* (`accuracy >= 0.75`). When a student hits a cross-modality bottleneck on `R2`, static branching wastes trials repeating `R2` until a hard timeout.
   - `ZPDES` (Figure 5) uses **zero learning progress as an active diagnostic signal** to open lateral modalities (`MM1/MM2`), allowing students to acquire the missing sub-skill in a different modality and then conquer Level 3 (`2.922 / 3.0`).
