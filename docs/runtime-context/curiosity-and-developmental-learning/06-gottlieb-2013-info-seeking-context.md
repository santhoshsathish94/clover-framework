# Context File — Paper 06: Gottlieb, Oudeyer, Lopes & Baranes (2013) — *Information-Seeking, Curiosity, and Attention: Computational and Neural Mechanisms*

- **Citation**: Jacqueline Gottlieb, Pierre-Yves Oudeyer, Manuel Lopes & Adrien Baranes (2013). *Information-Seeking, Curiosity, and Attention: Computational and Neural Mechanisms*. **Trends in Cognitive Sciences**, Vol. 17, No. 11, pp. 585–593 (`doi:10.1016/j.tics.2013.09.001`).
- **Source PDF / Extracted Text**:
  - Primary PDF URL: `https://www.pyoudeyer.com/TICSCuriosity2013.pdf`
  - Extracted Full Text (`58,539 bytes`): [research/curiosity-and-developmental-learning/papers_text/06-gottlieb-2013-info-seeking.txt](../../../research/curiosity-and-developmental-learning/papers_text/06-gottlieb-2013-info-seeking.txt)
- **Executable Experiment & Saved Results**:
  - Experiment Script: [research/curiosity-and-developmental-learning/06-gottlieb-2013-info-seeking/run_gottlieb2013_experiments.py](../../../research/curiosity-and-developmental-learning/06-gottlieb-2013-info-seeking/run_gottlieb2013_experiments.py)
  - Saved Results Artifact: [research/curiosity-and-developmental-learning/06-gottlieb-2013-info-seeking/results/gottlieb_2013_results.json](../../../research/curiosity-and-developmental-learning/06-gottlieb-2013-info-seeking/results/gottlieb_2013_results.json)

---

## Stage 1 — Context (Primary Source & Theoretical Synthesis)

### 1.1 Bridging Three Separated Fields: Active Learning, Oculomotor Gaze, and Curiosity
Gottlieb, Oudeyer, Lopes & Baranes (2013) unite three fields that had operated in isolation:
1. **Machine Learning & Developmental Robotics** (POMDPs, Bayesian active learning, `R-IAC` learning progress),
2. **Oculomotor Control & Natural Gaze** (Johansson et al. 2001 visuomanual landmark look-ahead fixations; LIP/FEF/SC target selection), and
3. **Psychology & Neuroscience of Curiosity** (Loewenstein 1994 Information Gap; Kang et al. 2009; Jepma et al. 2012).

### 1.2 Observing Actions in POMDPs vs. Pavlovian "Attention for Liking" (Figure 1A & Box 1, pp. 2–4)
- **Figure 1A (POMDP Observing Action)**: When approaching an intersection in uncertain belief state $x_a$ (mixture of red light $x_{a1}$ with prior probability $p$ and green light $x_{a2}$ with probability $1-p$), an **observing action** (glancing at the traffic light) incurs an immediate time/effort cost $r_o = -0.10 < 0$ and delivers zero immediate physical reward, but transitions belief to $x_{b1}$ or $x_{b2}$ where uncertainty is zero.
- **Redundancy & Epistemic Context (p. 3)**: Whether glancing at the light has positive Value of Information ($\text{VoI}$) depends on two strict prerequisites: (1) knowing how to map the observation to future action, and (2) momentary uncertainty. If seeing a car stopped in the intersection already raises prior certainty $p \ge 1 + r_o = 0.90$, glancing at the light has $\text{VoI} \le 0$.
- **Box 1 (Normative Information Value vs. Pavlovian "Attention for Liking", p. 4)**: Single-cell recordings in monkey Lateral Intraparietal area (`LIP`) show two competing valuation signals:
  1. A **Pavlovian reward-association bias ("attention for liking")** that preferentially attends to positive ("good news") cues over negative cues even when both cues carry identical information and zero action relevance; and
  2. A **two-stage normative information-value process** (e.g., enhanced `LIP` firing to a high-penalty warning cue that enables the monkey to avoid the penalty and saccade to a safe alternative).

### 1.3 Why Loewenstein's Information Gap Fails in Open-Ended Discovery (pp. 6–7)
On p. 6, the authors identify a foundational limitation of Loewenstein's (1994) **Information Gap hypothesis** ($H_{\text{current}} - H_{\text{reference}}$):
- Defining an information gap requires knowing both the **starting entropy** $H_{\text{current}}$ (e.g., reading the first pages of a mystery novel to know there are 10 suspects) and the **goal reference entropy** $H_{\text{reference}}$ (1 culprit).
- In open-ended scientific research or infant development, *"the results of an experiment typically open up new questions that were not foreseen, and it is not possible to estimate in advance the current entropy or the final desired state."*
- **Local Learning Progress (`LP`, Fig. 2, pp. 6–7)** solves this without any reference point $H_{\text{reference}}$ by rewarding the local negative time-derivative of prediction or competence error.

---

## Stage 2 — Direction (Testable Hypotheses)

1. **Experiment E1 (Figure 1A POMDP Observing Action & Box 1 Gaze Valuation)**:
   - Compute $\text{VoI}(p) = (1 + r_o) - \max(p, 1-p)$ across $p \in [0.50, 1.00]$ for $r_o = -0.10$ and verify the exact indifference threshold $p^* = 0.90$.
   - Compare a normative `VoI` gaze controller against a Pavlovian `attention_for_liking` controller when choosing between an uninformative $100\%$-reward cue (`VoI = 0`) and an informative high-penalty warning cue (`VoI = +0.30`).
2. **Experiment E2 (Information Gap vs. Local Learning Progress in Branch-Expanding Scientific Discovery)**:
   - Compare an `info_gap_reference_seeker` (which prefers closing bounded gaps $H_{\text{apparent}} \to 0$ and is deterred when new discoveries expand the visible hypothesis space $H_{\text{apparent}}$) against `local_learning_progress_lp` across `closed_trivia_puzzle`, `expanding_scientific_frontier`, and `irreducible_random_noise`.

---

## Stage 3 — Execution & Stage 4 — Outcome (Quantitative Results)

Running [run_gottlieb2013_experiments.py](../../../research/curiosity-and-developmental-learning/06-gottlieb-2013-info-seeking/run_gottlieb2013_experiments.py) produces:

### 4.1 Experiment E1: POMDP Observing Value of Information & Box 1 Gaze Control
- **POMDP Intersection `VoI` Curve (`r_o = -0.10`)**:
  - At $p = 0.50$: $V_{\text{blind}} = 0.50$, $V_{\text{observe}} = 0.90 \implies \mathbf{\text{VoI} = +0.40}$ (`should_glance = true`).
  - At $p = 0.80$: $V_{\text{blind}} = 0.80$, $V_{\text{observe}} = 0.90 \implies \mathbf{\text{VoI} = +0.10}$ (`should_glance = true`).
  - At $p = 0.90$ (Indifference Threshold): $V_{\text{blind}} = 0.90$, $V_{\text{observe}} = 0.90 \implies \mathbf{\text{VoI} = 0.00}$ (`should_glance = false`).
  - At $p = 1.00$ (Car visible in intersection): $\mathbf{\text{VoI} = -0.10}$ (`should_glance = false`).
- **Box 1 Normative `VoI` Gaze vs. Pavlovian "Attention for Liking" (`1,000` trials)**:
  - `normative_voi` controller fixates the high-penalty warning cue **`100.0%`** of the time and achieves **`+0.6838`** mean return.
  - `pavlovian_liking` controller fixates the warning cue only **`19.2%`** of the time (drawn to the uninformative "good news" cue) and drops to **`+0.0964`** mean return ($7.09\times$ lower).

### 4.2 Experiment E2: Information Gap vs. Local Learning Progress (`12` seeds $\times 300$ steps)
| Strategy | `closed_trivia_puzzle` | `expanding_scientific_frontier` | `irreducible_random_noise` | Mean Scientific Discoveries |
|---|---|---|---|---|
| **`info_gap_reference_seeker`** | `18.25%` | `19.64%` | `62.11%` | `78.92` |
| **`local_learning_progress_lp`** | `19.08%` | **`75.06%`** | **`5.86%`** | **`239.58`** ($3.04\times$ higher) |

---

## Stage 5 — Growth (What We Learned & Critical Boundaries)

1. **Why Epistemic Actions Require Differentiating Unbounded Discovery from Bounded Reference Gaps**:
   - Whenever answering a research question reveals new sub-questions (increasing global apparent entropy $H_{\text{apparent}}$), any curiosity metric defined as distance to a global reference state $H_{\text{ref}}$ misinterprets the newly revealed frontier as *negative progress* (widening the gap) and retreats to closed trivia puzzles.
   - Local Learning Progress (`LP`) evaluates improvement on the *empirical predictions just tested*, sustaining **`75.06%`** allocation on the expanding frontier and achieving **`3.04x` more discoveries**.
