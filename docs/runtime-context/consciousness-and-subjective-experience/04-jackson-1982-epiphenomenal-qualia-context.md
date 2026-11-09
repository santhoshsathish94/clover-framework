# Frank Jackson (1982) — *Epiphenomenal Qualia* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Jackson, F. (1982). *Epiphenomenal Qualia*. The Philosophical Quarterly, Vol. 32, No. 127 (Apr., 1982), pp. 127–136. (`doi:10.2307/2219321`)
- **Primary Source File**: JSTOR / Oxford University Press DOI (`10.2307/2219321`); bibliographic metadata (`OpenAlex W2012592964`, `2,953` citations) recorded in [research/consciousness-and-subjective-experience/papers_text/04-jackson-1982-epiphenomenal-qualia.txt](../../../research/consciousness-and-subjective-experience/papers_text/04-jackson-1982-epiphenomenal-qualia.txt).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/04-jackson-1982-epiphenomenal-qualia/run_jackson_1982_experiments.py](../../../research/consciousness-and-subjective-experience/04-jackson-1982-epiphenomenal-qualia/run_jackson_1982_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/04-jackson-1982-epiphenomenal-qualia/results/jackson_1982_epiphenomenal_qualia_results.json](../../../research/consciousness-and-subjective-experience/04-jackson-1982-epiphenomenal-qualia/results/jackson_1982_epiphenomenal_qualia_results.json)

---

## 2. What the Paper Claims & Its Formal Structure

### 2.1 The Knowledge Argument: Fred's Split-Red & Mary the Color Scientist
Jackson introduces two thought experiments to argue that complete physical knowledge does not exhaust all facts about conscious experience:
1. **Fred (`Red_1` vs. `Red_2`)**: Fred has a 4th retinal cone receptor splitting the long-wavelength red band so that he effortlessly distinguishes two classes of ripe tomatoes (`Red_1` and `Red_2`) that are exact optical metamers to normal human trichromats. Even after scientists map Fred's 4-cone physiology and predict every sorting choice he makes, they still lack the fourth chromatic dimension of Fred's visual experience.
2. **Mary the Color Scientist**: Mary learns all physical, optical, and neurophysiological facts $\mathcal{T}_{\text{phys}}$ about human color vision from black-and-white textbooks and television monitors. When released to see a ripe red tomato directly, she learns what it is like to see red.

### 2.2 Epiphenomenalism & Its Causal Closure Paradox
In Section IV of the 1982 paper, Jackson argues that qualia are **epiphenomenal**—caused by brain states ($\text{Brain}_t \to \text{Qualia}_t$), but exercising **zero causal efficacy** on the physical world or subsequent behavior ($\frac{\partial \text{Brain}_{t+1}}{\partial \text{Qualia}_t} = 0$).
- **Where the explanation breaks**: If $\text{Qualia}_t$ has zero outgoing causal arrows to motor/speech centers $\text{Speech}_{t+1}$, then Mary's own exclamation upon leaving the room (*"Now I know what red looks like!"*) and her memory traces are caused entirely by physical brain transitions that would occur identically if $\text{Qualia}_t$ were inverted or absent—the self-refutation that later led Jackson (2003, *Mind and Illusion*) to reject epiphenomenalism in favor of representationalism and the Ability Hypothesis.

---

## 3. Executable Experiments Run & Observed Results

We implemented all three experiments in [run_jackson_1982_experiments.py](../../../research/consciousness-and-subjective-experience/04-jackson-1982-epiphenomenal-qualia/run_jackson_1982_experiments.py):

### Experiment E1: Fred's Split-Red Tetrachromacy (`S, M, L1, L2` vs. Trichromat `S, M, L`)
- **Setup**: Simulated 80 batches of `Red_1` ($620\text{ nm}$ dominant) and `Red_2` ($645\text{ nm}$ dominant) tomatoes balanced to produce identical standard `S, M, L` trichromatic cone responses.
- **Observed Results**:
  - Normal trichromat naked-eye sorting accuracy: **`45.0%`** (chance).
  - Fred's 4-cone sorting accuracy: **`100.0%`**.
  - Third-person biophysical scientist predicting Fred's sorting choices from spectrophotometer measurements + Fred's opsin curves: **`100.0%`**.
  - Proportion of Fred's `L1 - L2` chromatic dimension spanned by normal 3D trichromat cone space: **`0.29%`** (`r2 = 0.0029`).

### Experiment E2: Mary's Room — Third-Person Physical Equations vs. First-Person Sensory Calibration
- **Setup**: Evaluated a model of Mary before and after release from the black-and-white room on (a) third-person physical law prediction error, (b) number of new parameters required in the physical forward model $\mathcal{T}_{\text{phys}}$, and (c) unassisted first-person recognition of raw unlabelled retinal opponent vectors.
- **Observed Results**:
  - Third-person physical law error before and after release: **`0.0`** $\to$ **`0.0`** (**`0` new physical law parameters added** upon release).
  - First-person raw sensory recognition accuracy before grounding her internal sensory coordinate gauge $G_{\text{sensory}}$: **`25.0%`** (chance across 4 opponent poles) $\to$ **`100.0%`** after anchoring on 2 real-world exemplars (formalizing the Lewis-Nemirow Ability / Practical Grounding account).

### Experiment E3: Testing Epiphenomenalism in a Structural Causal Model (`do`-Calculus Intervention)
- **Setup**: Constructed Jackson's (1982) epiphenomenalist causal DAG (`Stimulus -> NeuralState_t -> Speech_{t+1}` and `NeuralState_t -> Qualia_t` with zero arrow `Qualia_t -> Speech_{t+1}`) across 200 trials and intervened via `do(Qualia_t := -Qualia_t)` (inversion) and `do(Qualia_t := 0)` (ablation).
- **Observed Results**:
  - Average Causal Effect (`ACE`) of qualia inversion on speech/report output: **`0.0`**.
  - Average Causal Effect (`ACE`) of qualia ablation on speech/report output: **`0.0`**.
  - Under epiphenomenalism, the subject's claims and beliefs about their own qualia are statistically and causally independent of the qualia themselves.

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Why Grounding Raw Sensory Channels Changes Practical Abilities Without Adding Physical Laws**: Before receiving any chromatic input, an agent with a complete symbolic specification of opponent color space (`R-G, Y-B`) still faces an ungrounded coordinate gauge between its own raw receptor wires and the symbolic axes (`25%` chance recognition); two grounded sensory observations fix the gauge (`100%` recognition) without altering a single equation of physics.
2. **Scope Boundary**: Jackson's (1982) epiphenomenal dualism violates causal self-reference for introspective reports (`ACE = 0.0`), showing why any scientifically coherent theory of subjective experience must integrate experience into the causal/informational architecture of the agent.
