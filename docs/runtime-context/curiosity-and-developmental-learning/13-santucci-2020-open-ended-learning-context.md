# Context File — Paper 13: Santucci, Oudeyer, Barto & Baldassarre (2020) — *Editorial: Intrinsically Motivated Open-Ended Learning in Autonomous Robots*

- **Citation**: Vieri Giuliano Santucci, Pierre-Yves Oudeyer, Andrew Barto & Gianluca Baldassarre (2020). *Editorial: Intrinsically Motivated Open-Ended Learning in Autonomous Robots*. **Frontiers in Neurorobotics**, Vol. 13, Article 115, pp. 1–3 (`doi:10.3389/fnbot.2019.00115`).
- **Source PDF / Extracted Text**:
  - Primary PDF URL: `https://www.frontiersin.org/articles/10.3389/fnbot.2019.00115/pdf`
  - Extracted Full Text (`18,622 bytes`): [research/curiosity-and-developmental-learning/papers_text/13-santucci-2020-open-ended-learning.txt](../../../research/curiosity-and-developmental-learning/papers_text/13-santucci-2020-open-ended-learning.txt)
- **Executable Experiment & Saved Results**:
  - Experiment Script: [research/curiosity-and-developmental-learning/13-santucci-2020-open-ended-learning/run_santucci2020_imol_experiments.py](../../../research/curiosity-and-developmental-learning/13-santucci-2020-open-ended-learning/run_santucci2020_imol_experiments.py)
  - Saved Results Artifact: [research/curiosity-and-developmental-learning/13-santucci-2020-open-ended-learning/results/santucci_2020_imol_results.json](../../../research/curiosity-and-developmental-learning/13-santucci-2020-open-ended-learning/results/santucci_2020_imol_results.json)

---

## Stage 1 — Context (Primary Source & The 6 Open Challenges of IMOL)

### 1.1 Synthesis of the `IMOL` (Intrinsically Motivated Open-Ended Learning) Field
Santucci, Oudeyer, Barto & Baldassarre (2020) synthesize the 18 research contributions of the `IMOL` Research Topic and define the **six core open challenges** facing autonomous open-ended learning robots (pp. 1–2):
1. **Autonomous Goal Generation for Unknown Future Extrinsic Tasks**: How an agent should self-generate goals and learn policies so that the acquired skill repertoire transfers zero-shot or few-shot to extrinsic tasks unknown at design/exploration time (Santucci et al., 2016 `GRAIL`; Baranes & Oudeyer, 2013).
2. **Learning Compact, Action-Relevant State & Goal Representations**: Raw sensory streams contain vast task-irrelevant variation; intrinsic motivation must support learning compact goal spaces filtered by **action relevance / controllability** (Baldassarre et al.; Doncieux et al.; Dauce).
3. **Bridging Continuous Low-Level Goals and Discrete Compositional High-Level Goals**: Low-level postural/kinematic goals are continuous, whereas high-level affordance goals (`touch`, `push`, `reach`, `grasp`) are discrete and composable into hierarchical chains (Duminy et al.; Dhakan et al.).
4. **Transfer Learning & Curriculum Chaining**: Reusing previously learned skills to bootstrap harder composite skills (Rayyes et al.; Uchibe; de La Bourdonnaye et al.).
5. **Overcoming Catastrophic Interference in Lifelong Open-Ended Learning**: Sequential exploration of changing progress niches causes catastrophic forgetting unless paired with dual episodic-semantic memory consolidation (Parisi et al., p. 2) and competence-retention monitoring.
6. **Synergy Between Intrinsic Motivation and Social Interaction / Language**: Integrating imitation, strategic learning, and multi-agent interaction (Duminy et al.; Mahzoon et al.; Khan et al.).

---

## Stage 2 — Direction (Testable Hypotheses)

1. **Experiment E1 (Open Challenges 1, 2 & 3 — Action-Relevant Goal Representation Filtering vs. Unfiltered Sensory Goal Babbling)**:
   - Place an agent in a 20-dimensional sensory environment $s \in \mathbb{R}^{20}$ where only 4 dimensions ($s_{0..3}$) are action-controllable object/end-effector coordinates and 16 dimensions ($s_{4..19}$) are high-variance uncontrollable ambient distractors.
   - Compare `unfiltered_20d_sensory_goal_babbling` against `compact_action_relevant_4d_plus_discrete_hierarchy` (which estimates action-to-state controllability $R^2_j$ per sensory dimension and projects goal babbling onto the 4D controllable subspace).
   - Evaluate zero-shot performance on 30 novel extrinsic manipulation goals after 400 open-ended exploration steps.
2. **Experiment E2 (Open Challenge 5 — Catastrophic Interference in Sequential Open-Ended Learning vs. Dual-Memory Episodic-Semantic Replay + `|LP|` Vigilance)**:
   - Compare `single_network_sequential_imol` (forward-only `LP = max(0, pst - rec)` with representation interference on unpracticed tasks) against `dual_memory_episodic_semantic_imol` (Parisi et al. dual episodic-semantic consolidation + `|LP|` competence-drop vigilance) across 4 sequential tasks.

---

## Stage 3 — Execution & Stage 4 — Outcome (Quantitative Results)

Running [run_santucci2020_imol_experiments.py](../../../research/curiosity-and-developmental-learning/13-santucci-2020-open-ended-learning/run_santucci2020_imol_experiments.py) (`10` seeds) produces:

### 4.1 Experiment E1: Compact Action-Relevant Goal Space vs. Unfiltered 20D Sensory Goal Babbling
| Architecture | Identified Controllable Subspace $R^2$ | Mean Extrinsic 30-Goal Reaching Error |
|---|---|---|
| **`unfiltered_20d_sensory_goal_babbling`** | `1.0000` (unfiltered mask) | `0.9332` (Corrupted by 16D distractors) |
| **`compact_action_relevant_4d_plus_discrete_hierarchy`** | **`0.9671`** (`4 / 20` dims isolated) | **`0.2722`** (**`3.43x` lower error**) |

### 4.2 Experiment E2: Catastrophic Forgetting vs. Dual-Memory Episodic-Semantic IMOL
| Architecture | Mean Competence Across All 4 Tasks | Worst-Retained Task Competence |
|---|---|---|
| **`single_network_sequential_imol`** | `0.4380` | `0.1201` (Severe forgetting of early skills) |
| **`dual_memory_episodic_semantic_imol`** | **`0.9156`** (**`2.09x` higher**) | **`0.8842`** (**`7.36x` higher retention**) |

---

## Stage 5 — Growth (What We Learned & Critical Boundaries)

1. **Why Goal Babbling Fails in Raw Sensory Space Without Controllability Filtering (`0.9332` vs. `0.2722`)**:
   - In `SAGG-RIAC` or `SGIM-D` (Paper 12), the task space $T \subset \mathbb{R}^2$ was hand-defined by the engineer. When an open-ended agent must form goals directly from raw sensory streams $s \in \mathbb{R}^{20}$, 16 uncontrollable background dimensions dominate Euclidean nearest-neighbor goal lookup, preventing the agent from expanding its controllable reach (`0.9332`).
   - Filtering sensory dimensions by **action controllability ($R^2(U \to S_j)$)** isolates the 4D controllable manifold at step 30 (`R^2 = 0.9671`) and cuts extrinsic 30-goal error by **`3.43x`** (`0.2722`).
2. **Why Open-Ended Curiosity Needs Dual Episodic-Semantic Memory + `|LP|` Vigilance (`0.8842` vs. `0.1201`)**:
   - Because standard `LP` abandons a skill as soon as it is mastered (`LP -> 0`), a single plastic network sequentially overwrites earlier skills (`worst_retained = 0.1201`). Pairing Parisi et al.'s dual episodic-semantic consolidation with `|LP|` competence-drop monitoring preserves **`0.8842`** worst-task competence across the entire repertoire.
