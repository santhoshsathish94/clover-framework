# Francis Crick & Christof Koch (1990) — *Towards a Neurobiological Theory of Consciousness* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Crick, F., & Koch, C. (1990). *Towards a Neurobiological Theory of Consciousness*. Seminars in the Neurosciences, Vol. 2, pp. 263–275.
- **Primary Source File**: CaltechAUTHORS repository record (`CaltechAUTHORS:20130816-103136937`, `OpenAlex W2164470519`, `1,651` citations) with complete verbatim abstract and metadata in [research/consciousness-and-subjective-experience/papers_text/06-crick-koch-1990-neurobiological-theory.txt](../../../research/consciousness-and-subjective-experience/papers_text/06-crick-koch-1990-neurobiological-theory.txt).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/06-crick-koch-1990-neurobiological-theory/run_crick_koch_1990_experiments.py](../../../research/consciousness-and-subjective-experience/06-crick-koch-1990-neurobiological-theory/run_crick_koch_1990_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/06-crick-koch-1990-neurobiological-theory/results/crick_koch_1990_results.json](../../../research/consciousness-and-subjective-experience/06-crick-koch-1990-neurobiological-theory/results/crick_koch_1990_results.json)

---

## 2. What the Paper Claims & Its Neurocomputational Mechanism

### 2.1 Two Forms of Visual Awareness & The Binding Problem
Crick and Koch propose using mammalian **visual awareness** as the primary experimental model for the neural correlates of consciousness (`NCC`), distinguishing two forms:
1. **Fast Parallel Form**: Linked to high-capacity, rapidly decaying **iconic memory** ($\tau \approx 200\text{ ms}$).
2. **Slower Attentional Form**: Involving selective **visual attention** that serializes and binds distributed cortical feature representations (`color` in V4, `orientation` in V1/V2, `motion` in MT) into **transient short-term (working) memory**.

### 2.2 The $40\text{–}70\text{ Hz}$ Gamma Oscillation Binding Hypothesis
When two objects (`Obj1 = Red + Vertical`, `Obj2 = Green + Horizontal`) are simultaneously present, summing firing rates across specialized visual areas produces von der Malsburg's **Superposition Catastrophe** (the downstream network cannot tell whether `Red` belongs with `Vertical` or `Horizontal`). Crick & Koch hypothesize that attention transiently binds all neurons coding features of the same object by synchronizing their spikes in coherent **$40\text{–}70\text{ Hz}$ gamma oscillations**, which then cross a coincidence-detection threshold to enter working memory.

---

## 3. Executable Experiments Run & Observed Results

We implemented the Kuramoto $45\text{ Hz}$ oscillatory binding network and its two failure boundaries in [run_crick_koch_1990_experiments.py](../../../research/consciousness-and-subjective-experience/06-crick-koch-1990-neurobiological-theory/run_crick_koch_1990_experiments.py):

### Experiment E1: $45\text{ Hz}$ Gamma Phase-Locking vs. Rate-Only Superposition Catastrophe
- **Setup**: 6 distributed cortical feature columns (`Red, Green, Vertical, Horizontal, LeftMotion, RightMotion`) stimulated by two simultaneous objects (`Obj1 = Red+Vertical+Left`, `Obj2 = Green+Horizontal+Right`).
- **Observed Results**:
  - **Rate-Only Coding**: Suffers complete superposition ambiguity (**`50.0%` conjunction accuracy**).
  - **Attentional $45\text{ Hz}$ Kuramoto Phase-Locking**: Intra-object phase-locking value reaches **`PLV = 1.0`**, true conjunction coincidence rate is **`100.0%`**, illusory conjunction coincidence rate is **`0.0%`**, and working-memory binding accuracy is **`100.0%`**.

### Experiment E2: Fast Iconic Memory vs. Serial Attentional Working-Memory Bottleneck
- **Setup**: 8 visual items presented simultaneously with iconic decay $\tau_{\text{iconic}} = 220\text{ ms}$ and attentional serial dwell time $65\text{ ms/object}$.
- **Observed Results**:
  - Fast iconic items surviving at $50\text{ ms}$: **`6.374 / 8`** $\to$ decays to **`0.523 / 8`** by $600\text{ ms}$.
  - Exactly **`4` objects** are sequentially bound into durable working memory before the iconic SNR drops below the `0.30` threshold.

### Experiment E3: Where the $40\text{ Hz}$ Hypothesis Breaks (Phase-Crowding Capacity & Anesthesia Gamma)
- **Setup**: Tested (a) packing $M \in \{2, \dots, 8\}$ phase-segregated object assemblies into a single $22.22\text{ ms}$ ($45\text{ Hz}$) gamma period with a $4.2\text{ ms}$ coincidence window and $1.4\text{ ms}$ spike jitter, and (b) local V1 $45\text{ Hz}$ gamma synchrony (`PLV = 0.96`) under general anesthesia (`thalamocortical_wm_gate = 0.0`).
- **Observed Results**:
  - **Phase-Crowding Breakdown**: Illusory conjunction collision rate is **`0.05%`** at $M = 2$ (`11.11 ms` spacing) and **`5.4%`** at $M = 3$ (`7.41 ms`), but jumps to **`24.6%`** at $M = 4$ (`5.56 ms`), **`44.3%`** at $M = 5$, and **`77.08%`** at $M = 8$ (`2.78 ms` spacing)—explaining why a single gamma band cannot phase-multiplex more than $3\text{–}4$ simultaneous objects.
  - **Anesthesia Dissociation**: High local V1 $45\text{ Hz}$ gamma synchrony (`PLV = 0.96`) produces **`0.0`** working-memory activation when long-range thalamocortical working-memory gating is blocked.

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Why Crick & Koch Later Abandoned $40\text{ Hz}$ as a Sufficient Condition for Consciousness**: Subsequent recordings showed that $40\text{ Hz}$ gamma oscillations persist in primary visual cortex during general anesthesia and during unconscious figure-ground segmentation, leading Crick & Koch (1995, 2003) and Koch et al. (2016, Paper 10) to treat local gamma synchrony as an early feature-binding mechanism rather than a sufficient neural correlate of consciousness.
