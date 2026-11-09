# Stanislas Dehaene & Jean-Pierre Changeux (2011) — *Experimental and Theoretical Approaches to Conscious Processing* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Dehaene, S., & Changeux, J.-P. (2011). *Experimental and Theoretical Approaches to Conscious Processing*. Neuron, Vol. 70, No. 2, pp. 200–227. (`doi:10.1016/j.neuron.2011.03.018`, `PMID: 21521609`)
- **Primary Source File**: Complete verbatim PubMed abstract (`PMID: 21521609`) and bibliographic record (`OpenAlex W2015303758`, `2,728` citations) in [research/consciousness-and-subjective-experience/papers_text/08-dehaene-changeux-2011-gnw.txt](../../../research/consciousness-and-subjective-experience/papers_text/08-dehaene-changeux-2011-gnw.txt) (Elsevier/Cell Press HTML and HAL mirror return 403/Anubis challenge).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/08-dehaene-changeux-2011-gnw/run_dehaene_changeux_2011_gnw_experiments.py](../../../research/consciousness-and-subjective-experience/08-dehaene-changeux-2011-gnw/run_dehaene_changeux_2011_gnw_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/08-dehaene-changeux-2011-gnw/results/dehaene_changeux_2011_gnw_results.json](../../../research/consciousness-and-subjective-experience/08-dehaene-changeux-2011-gnw/results/dehaene_changeux_2011_gnw_results.json)

---

## 2. What the Paper Claims & Its Dynamical Mechanism

### 2.1 The Global Neuronal Workspace (`GNW`) Model
Dehaene and Changeux address the transition from **nonconscious processing** to **conscious access** (defined operationally by global availability and reportability):
1. **Stage 1 — Subliminal Feedforward Sweep (`0–150 ms`)**: Incoming sensory stimuli propagate through a hierarchy of specialized processors (`V1 -> V4/IT`), activating local feature representations linearly proportional to stimulus duration/contrast. If masked or too brief, this wave decays without reaching global access.
2. **Stage 2 — Nonlinear Bifurcation & Global Ignition (`~250–300 ms`)**: When ascending sensory drive depolarizes high-level workspace neurons (densely distributed in prefrontal, parieto-temporal, and cingulate cortices) past the voltage-dependent $\text{Mg}^{2+}$ unblock threshold of long-range recurrent NMDA synapses, the network undergoes a sudden, all-or-none **phase transition (Global Ignition)**:
   - Late sustained amplification of sensory activity (`P3b` / late positive complex);
   - Long-distance cortico-cortical beta/gamma synchrony;
   - Global broadcasting to multiple consumer systems (working memory, verbal report, planning, metacognition).

---

## 3. Executable Experiments Run & Observed Results

We implemented the 4-stage hierarchical dynamical `GNW` model (`V1 -> V4_IT -> Parietal -> PFC_GNW`) in [run_dehaene_changeux_2011_gnw_experiments.py](../../../research/consciousness-and-subjective-experience/08-dehaene-changeux-2011-gnw/run_dehaene_changeux_2011_gnw_experiments.py):

### Experiment E1: Nonlinear Saddle-Node Bifurcation & All-or-None Global Ignition
- **Setup**: Swept stimulus duration across `15, 25, 35, 42, 46, 50, 55, 65, 80, 100 ms`.
- **Observed Results**:
  - Early `V1` peak amplitude increases smoothly and continuously with duration (`0.6125` at `15 ms` $\to$ `0.8976` at `42 ms` $\to$ `0.9198` at `50 ms`).
  - By contrast, late `PFC_GNW` sustained activity (`300–550 ms`) exhibits a sharp **all-or-none bifurcation** at $48\text{ ms}$: jumping from **`0.0009`** at `42 ms` (`ignited = False`) and **`0.0011`** at `46 ms` (`ignited = False`) to **`1.9887`** at `50 ms` (`ignited = True`, a **`2,209.67x` nonlinear amplification jump**).

### Experiment E2: The Attentional Blink — Refractory Workspace Competition (`T1` vs. `T2`)
- **Setup**: Presented Target 1 (`T1` at $30\text{ ms}$) followed by Target 2 (`T2`) at Stimulus Onset Asynchronies $\text{SOA} \in \{100, 200, 280, 420, 600\}\text{ ms}$.
- **Observed Results**:
  - `T2` evokes an identical early `V1` sensory peak (**`0.9405`**) at every SOA.
  - However, lateral inhibition from `T1`'s ongoing workspace ignition completely suppresses `T2` ignition during the Attentional Blink window (`t2_late_gnw_peak = 0.0` at `SOA 100 ms`, `0.0002` at `SOA 200 ms`, `0.0092` at `SOA 280 ms` $\to$ `ignited = False`), before recovering to full ignition (**`1.4537`**, `ignited = True`) once `T1` adapts at `SOA 420 ms` and `600 ms`.

### Experiment E3: General Anesthesia — Recurrent Disconnection vs. Intact Early Sensory Response
- **Setup**: Reduced long-range recurrent conductance $g_{\text{rec}}$ from `1.15` (Awake) $\to$ `0.55` (Light Sedation) $\to$ `0.10` (General Anesthesia) at fixed `65 ms` supraliminal stimulation.
- **Observed Results**:
  - Under general anesthesia ($g_{\text{rec}} = 0.10$), `V1` early feedforward peak is **`100.0%` preserved** (`0.9393` vs. `0.9393` awake), while late prefronto-parietal workspace ignition is **abolished** (`0.1653 < 0.80`, `ignited = False`).

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Importance of the Voltage-Dependent NMDA Threshold**: While implementing `simulate_gnw_hierarchy`, a smooth un-gated sigmoid had a non-zero tail at rest (`s(0) = 0.005`) that slowly integrated over $300\text{ ms}$ and caused spontaneous self-ignition even after $15\text{ ms}$ subliminal pulses. Incorporating the biophysical voltage-dependent $\text{Mg}^{2+}$ unblock threshold ($u_{\text{PFC}} \ge 0.120$) stabilized the resting state and reproduced the exact experimental bifurcation curve.
2. **Scope Boundary (`Access` vs. `Phenomenal`, No-Report Challenge)**: As Dehaene & Changeux explicitly state in their abstract, `GNW` models **conscious access** (global availability for report and executive control). Subsequent no-report paradigms (Tsuchiya et al. 2015, Paper 09; Koch et al. 2016, Paper 10) show that part of the late prefrontal `P3b` wave reflects post-perceptual task reporting rather than conscious perception itself.
