# David J. Chalmers (1995) — *Facing Up to the Problem of Consciousness* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Chalmers, D. J. (1995). *Facing Up to the Problem of Consciousness*. Journal of Consciousness Studies, Vol. 2, No. 3, pp. 200–219.
- **Primary Source File**: Full primary text extracted from author's archive `https://consc.net/papers/facing.html` (`81,067` bytes HTML, SHA-256 `058852c0e6ed04c6bde9c660bb80584a45285b5aa1fa26a1149194406b20f624`, `76,540` extracted text chars in [research/consciousness-and-subjective-experience/papers_text/02-chalmers-1995-facing-up.txt](../../../research/consciousness-and-subjective-experience/papers_text/02-chalmers-1995-facing-up.txt)).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/02-chalmers-1995-facing-up/run_chalmers_1995_experiments.py](../../../research/consciousness-and-subjective-experience/02-chalmers-1995-facing-up/run_chalmers_1995_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/02-chalmers-1995-facing-up/results/chalmers_1995_facing_up_results.json](../../../research/consciousness-and-subjective-experience/02-chalmers-1995-facing-up/results/chalmers_1995_facing_up_results.json)

---

## 2. What the Paper Claims & Its Formal Structure

### 2.1 The 7 "Easy" Problems vs. The "Hard" Problem (Section 2)
Chalmers partitions phenomena grouped under the label "consciousness" into two distinct classes:
1. **The "Easy" Problems (Functional & Structural Capacities)**:
   - (i) The ability to discriminate, categorize, and react to environmental stimuli;
   - (ii) The integration of information by a cognitive system;
   - (iii) The reportability of mental states;
   - (iv) The ability of a system to access its own internal states;
   - (v) The focus of attention;
   - (vi) The deliberate control of behavior;
   - (vii) The difference between wakefulness and sleep.
   - Each of these is a **functional concept**: specifying the computational or neural mechanism that performs the function completely answers how the capacity works, because physical accounts entail structure and function.
2. **The "Hard" Problem (Phenomenal Experience)**:
   - Why and how the performance of those functions is accompanied by subjective experience (*what it is like* to be in the state). Physical accounts specify only structure and dynamics, which yield further structure and dynamics (`Section 5`).

### 2.2 Three Psychophysical Principles of a Nonreductive Theory (Section 7)
1. **Principle 1 — Structural Coherence (`Section 7.1`)**:
   - Defines **awareness** functionally as *direct availability of information for global control and verbal report*.
   - States that the relational geometry of conscious experience (e.g., 3D opponent color space varying in hue, saturation, and intensity) is directly isomorphic to the relational geometry of functional awareness.
2. **Principle 2 — Organizational Invariance (`Section 7.2`)**:
   - Any two systems with the same fine-grained causal/functional organization have qualitatively identical experiences, regardless of whether they are realized in biological neurons or silicon.
   - Supported by a *reductio ad absurdum* (**Fading & Dancing Qualia**): if replacing neurons with causally isomorphic silicon chips altered experience (say from red to blue) while keeping functional states fixed, flipping a switch between parallel neural and silicon subcircuits would make experiences "dance" while the system's own functional change detector and beliefs register zero change.
3. **Principle 3 — Double-Aspect Theory of Information (`Section 7.3`)**:
   - Following Shannon (1948) and Bateson (1972, *"a difference that makes a difference"*), physical information is an abstract space of difference relations embodied in states whose differences are transmitted down a causal pathway implicated in global availability and control.

---

## 3. Executable Experiments Run & Observed Results

We implemented all four components in [run_chalmers_1995_experiments.py](../../../research/consciousness-and-subjective-experience/02-chalmers-1995-facing-up/run_chalmers_1995_experiments.py):

### Experiment E1: Implementing All 7 "Easy" Functional Capacities & Isomorphic Equivalence
- **Setup**: Built `FunctionalAwarenessAgent` implementing stimulus discrimination, multimodal integration, internal state self-access, attentional gain modulation, deliberate action control, verbal report generation, and wake/sleep arousal gating across 100 trials.
- **Observed Results**:
  - Wake vs. sleep gating produces a **`0.9089`** awareness norm contrast and suppresses reportability (`state=asleep;aware=none`).
  - Two functionally isomorphic instances (`agent_a` vs. `agent_b_zombie_isomorph`) exhibit **`0.0`** action divergence and **`100.0%`** report identity across all 100 trials, illustrating why functional specification entails every behavioral and structural marker without entailing an extra non-structural property.

### Experiment E2: Structural Coherence & Inverted-Spectrum Symmetry in 3D Color Space
- **Setup**: Mapped 60 spectral stimuli into a 3D opponent-process awareness space `(R-G, Y-B, Luminance)` and compared coherent vs. hue-inverted (`(R-G, Y-B) -> -(R-G, Y-B)`) phenomenal spaces.
- **Observed Results**:
  - Pairwise distance correlation between functional awareness and structurally coherent experience is **`1.000000`**.
  - Opponent hue inversion introduces **`0.0`** distortion in the pairwise discrimination matrix (`inverted_spectrum_structural_distortion = 0.0`) while shifting intrinsic hue coordinates by **`1.3175`**, verifying Chalmers's claim in Section 7.1 that Structural Coherence constrains relational geometry without fixing intrinsic qualitative sign.

### Experiment E3: Organizational Invariance — 10-Step Silicon Replacement & Dancing Qualia Switch
- **Setup**: Replaced $k \in \{0, \dots, 10\}$ biological units with causally isomorphic silicon units and toggled the Dancing Qualia switch across 20 time steps (`19` flips).
- **Observed Results**:
  - Functional output divergence across all 11 replacement stages is **`0.0`**.
  - Under **Substrate Chauvinism** (denying Organizational Invariance), toggling the switch causes **`19 / 19` unnoticed qualia flips** while the agent's internal functional change detector triggers **`0` times**—confirming Chalmers's reductio that denying Organizational Invariance severs experience from introspective awareness.

### Experiment E4: Double-Aspect Information Space — Local Entropy vs. Global Causal Transmission
- **Setup**: Compared a 6-bit isolated thermal noise register (`64` states), a 3-bit perceptual channel (`8` states) routed to the global control workspace `G` (`98%` fidelity), and a 1-bit thermostat relay (`2` states) over 4,000 steps using Miller-Madow bias-corrected mutual information.
- **Observed Results**:
  - Isolated thermal noise has high local Shannon entropy (**`6.0024 bits`**) but **`0.0010 bits`** of causal information transmitted to global control (`I(X_noise; G)`).
  - Routed perceptual discrimination has **`3.0002 bits`** of Shannon entropy and **`2.8034 bits`** of global causal information (`I(X_percept; G)`).

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Finite-Sample Plug-In Bias in Mutual Information**: On a $64 \times 8 = 512$-cell contingency table with $N = 4,000$ samples, raw plug-in mutual information overestimates causal transmission on independent noise by $\frac{K-1}{2N \ln 2} \approx 0.08\text{ bits}$. Applying the Miller-Madow correction reduced spurious noise information to `0.0010 bits`.
2. **Where Chalmers's Double-Aspect Principle Remains Underdetermined**: As Chalmers himself notes in Section 7.3, if *any* physical state difference (`1.0002 bits` in a thermostat or `6.0024 bits` in unrouted thermal noise) is granted a phenomenal aspect, the theory collapses into panpsychism; if restricted to information routed to *global control* (`2.8034 bits`), the burden shifts back to defining what architectural threshold constitutes a global control workspace (addressed by Tononi 2004 in Paper 07 and Dehaene & Changeux 2011 in Paper 08).
