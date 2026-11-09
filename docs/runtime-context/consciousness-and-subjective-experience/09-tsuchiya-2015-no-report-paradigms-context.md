# Naotsugu Tsuchiya, Max Wilke, Stefan Frässle & Victor A. F. Lamme (2015) — *No-Report Paradigms: Extracting the True Neural Correlates of Consciousness* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Tsuchiya, N., Wilke, M., Frässle, S., & Lamme, V. A. F. (2015). *No-Report Paradigms: Extracting the True Neural Correlates of Consciousness*. Trends in Cognitive Sciences, Vol. 19, No. 12, pp. 757–770. (`doi:10.1016/j.tics.2015.10.002`, `PMID: 26585549`)
- **Primary Source File**: Complete verbatim PubMed abstract (`PMID: 26585549`, `OpenAlex W2177128208`, `600` citations) in [research/consciousness-and-subjective-experience/papers_text/09-tsuchiya-2015-no-report-paradigms.txt](../../../research/consciousness-and-subjective-experience/papers_text/09-tsuchiya-2015-no-report-paradigms.txt).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/09-tsuchiya-2015-no-report-paradigms/run_tsuchiya_2015_no_report_experiments.py](../../../research/consciousness-and-subjective-experience/09-tsuchiya-2015-no-report-paradigms/run_tsuchiya_2015_no_report_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/09-tsuchiya-2015-no-report-paradigms/results/tsuchiya_2015_no_report_results.json](../../../research/consciousness-and-subjective-experience/09-tsuchiya-2015-no-report-paradigms/results/tsuchiya_2015_no_report_results.json)

---

## 2. What the Paper Claims & Its Methodological Mechanism

### 2.1 Why Report-Based Contrasts Overestimate Frontoparietal Involvement (`NCC-pr`, `NCC_true`, `NCC-co`)
Traditional neural correlates of consciousness (`NCC`) paradigms ask subjects to press a button or speak to report whether a stimulus was perceived (`Seen` vs. `Unseen`). Tsuchiya et al. show that subtracting `Unseen` from `Seen` under active reporting confounds three distinct stages:
1. **$\text{NCC-pr}$ (Prerequisites)**: Pre-stimulus attention and arousal enabling perception.
2. **$\text{NCC}_{\text{true}}$ (True Substrate of Phenomenal Experience)**: Recurrent sensory processing in posterior cortex.
3. **$\text{NCC-co}$ (Post-Perceptual Consequences)**: Metacognitive monitoring, working-memory maintenance, response selection, and motor execution in prefrontal and premotor areas (`dlPFC, ACC, SMA, M1`).

### 2.2 The No-Report Paradigm (Frässle et al. 2014 Binocular Rivalry with `OKN` & Pupil Size)
By inferring perceptual transitions during binocular rivalry from involuntary physiological reflexes—**Optokinetic Nystagmus (`OKN`)** slow-phase velocity and **pupil diameter**—when subjects passively view rivalrous stimuli without making any behavioral report, experimenters can separate $\text{NCC}_{\text{true}}$ (which persists in both report and no-report blocks) from $\text{NCC-co}$ (which vanishes when reporting is removed).

---

## 3. Executable Experiments Run & Observed Results

We implemented all three experiments in [run_tsuchiya_2015_no_report_experiments.py](../../../research/consciousness-and-subjective-experience/09-tsuchiya-2015-no-report-paradigms/run_tsuchiya_2015_no_report_experiments.py):

### Experiment E1: Deconfounding `NCC_true` from `NCC-co` Across 12 Cortical ROIs
- **Setup**: Simulated 12 cortical ROIs (`2 NCC-pr`, `4 NCC_true`, `6 NCC-co`) across 200 active-report trials and 200 passive no-report trials.
- **Observed Results**:
  - **Report-Only Contrast**: Flags `10` ROIs, including **`6` false-positive `NCC-co` regions** (`dlPFC, ACC, AnteriorInsula, SMA, PMd, M1`).
  - **Report $\cap$ No-Report Conjunction**: Isolates the exact **`4` `NCC_true` regions** (`V1_V2, V4_LOC, MT_V5, IPS_Posterior`) with **`0` false positives**.
  - **Prefrontal (`dlPFC`) Signal Reduction**: Drops from $t = \mathbf{25.749}$ in active report to $t = \mathbf{2.032}$ in passive no-report (**`92.11%` reduction**).

### Experiment E2: Decoding Conscious Perceptual Alternations from Involuntary `OKN` & Pupil Traces
- **Setup**: 600 time bins of binocular rivalry between leftward-bright and rightward-dark gratings decoded without button presses.
- **Observed Results**: `OKN` velocity alone decodes perceptual dominance at **`99.33%`**, pupil diameter alone at **`98.50%`**, and combined `(OKN, Pupil)` decoding achieves **`99.67%`**.

### Experiment E3: Where No-Report Reflexes Break (Subcortical Pretectal Confound Under Cortical Blindness)
- **Observed Results**: When physical luminance steps on the retina (`retinal_luminance_delta = 1.0`), a V1-lesioned cortically blind system (`v1_conscious_percept = 0.0`) still exhibits a `0.92` subcortical pupillary light reflex (`uncontrolled_luminance_step_causes_false_positive = True`), whereas holding retinal input constant during endogenous rivalry isolates the `0.48` cortical perceptual modulation (`controlled_rivalry_isolates_cortical_percept = True`).

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Internal Covert Counting / Post-Perceptual Thought Even Without Button Presses**: Even when overt button pressing is removed, a human subject noticing a rivalry switch may still silently think *"Oh, it switched to red"* (covert internal report). Thus, no-report paradigms dramatically reduce motor and task-monitoring `NCC-co`, but must be paired with inattentional or task-irrelevant designs (Paper 10) to eliminate covert cognitive monitoring.
