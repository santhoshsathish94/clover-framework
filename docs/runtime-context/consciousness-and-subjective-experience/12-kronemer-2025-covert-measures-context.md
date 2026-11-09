# Sharif I. Kronemer, Peter A. Bandettini & Javier Gonzalez-Castillo (2025) — *Sleuthing Subjectivity: A Review of Covert Measures of Consciousness* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Kronemer, S. I., Bandettini, P. A., & Gonzalez-Castillo, J. (2025). *Sleuthing Subjectivity: A Review of Covert Measures of Consciousness*. Nature Reviews Neuroscience, Vol. 26, pp. 476–496. (`doi:10.1038/s41583-025-00934-1`, `PMID: 40410542`)
- **Primary Source File**: Complete verbatim abstract, figure taxonomy (`Figs. 1–5`), and 180+ reference bibliography extracted (`1,189,349` bytes HTML, SHA-256 `d1e82b8c8fdfadbd9de1fbbd7e85527095f645b5ea97bb26110a626ddf189698`, `89,478` chars in [research/consciousness-and-subjective-experience/papers_text/12-kronemer-2025-covert-measures.txt](../../../research/consciousness-and-subjective-experience/papers_text/12-kronemer-2025-covert-measures.txt)).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/12-kronemer-2025-covert-measures/run_kronemer_2025_covert_measures_experiments.py](../../../research/consciousness-and-subjective-experience/12-kronemer-2025-covert-measures/run_kronemer_2025_covert_measures_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/12-kronemer-2025-covert-measures/results/kronemer_2025_covert_measures_results.json](../../../research/consciousness-and-subjective-experience/12-kronemer-2025-covert-measures/results/kronemer_2025_covert_measures_results.json)

---

## 2. What the Paper Claims & Its Physiological Framework

### 2.1 Four Limitations of Overt Behavioral Report (Fig. 1)
Overt behavioral reports (speech, button presses, gestures, facial expressions) fail in four critical regimes:
1. **Absence of Motor Output**: Coma recovery / Cognitive-Motor Dissociation (`CMD`), general anesthesia awareness, or REM sleep paralysis.
2. **Erroneous Report via Deception**: Wilful malingering or concealed recognition.
3. **Recall & Metacognitive Error**: Rapid forgetting (inattentional amnesia) between experience and delayed query.
4. **Research Goal Interference**: Reporting interrupts resting-state, mind-wandering, or dreaming states.

### 2.2 Multimodal Covert Physiological Measures (Figs. 2–4) & The Unconscious Reflex Challenge (Fig. 5)
Kronemer et al. synthesize four classes of peripheral physiological signals—**Eye** (`pupil dilation`, `blinks`, `microsaccades`, `OKN`), **Skin** (`electrodermal SCR`), **Respiratory** (`sniff/inspiratory pause`), and **Heart** (`cardiac deceleration/acceleration`)—and highlight the central methodological hurdle (`Fig. 5`):
- Because autonomic and oculomotor nuclei in the brainstem receive direct subcortical inputs (`0–250 ms`), early physiological responses occur even for **unconscious subliminal stimuli**.
- Isolating **conscious** perception requires separating the **late sustained cortical-autonomic broadcast wave** (`400–1000 ms`, driven by fronto-parieto-thalamic-autonomic networks) from the early subcortical orienting reflex.

---

## 3. Executable Experiments Run & Observed Results

We implemented both experiments in [run_kronemer_2025_covert_measures_experiments.py](../../../research/consciousness-and-subjective-experience/12-kronemer-2025-covert-measures/run_kronemer_2025_covert_measures_experiments.py):

### Experiment E1: Overt Report Failure Regimes vs. 4-Channel Covert Physiological Fusion

| Regime (`200` Trials Each) | Overt Report Sensitivity | Single-Channel Pupil Accuracy | 4-Channel Multimodal Covert Accuracy (`Pupil + Microsaccade + SCR + HR`) |
| :--- | :---: | :---: | :---: |
| `Normal_Cooperative_Awake` | `99.0%` | `87.5%` | **`95.0%`** |
| `Motor_Paralysis_CMD` | **`0.0%`** | `78.0%` | **`95.0%`** |
| `Wilful_Deception_Concealed` | **`2.0%`** | `85.5%` | **`93.0%`** |
| `Rapid_Forgetting_Delayed_Query` | **`17.0%`** | `77.0%` | **`93.0%`** |
| `REM_Dreaming_State` | **`0.0%`** | `77.0%` | **`93.5%`** |

### Experiment E2: Separating Conscious Perception from Unconscious Subcortical Autonomic Reflexes (`Fig. 5`)
- **Setup**: Simulated 360 time-series trials (`0–1200 ms`) across `Absent`, `Subliminal_Unconscious` (early subcortical reflex only), and `Conscious_Perceived` (early reflex + late cortical-autonomic wave).
- **Observed Results**:
  - **Early Window Detector (`50–250 ms`)**: Suffers a **`77.5%` false-positive rate** on unconscious subliminal stimuli (`early_window_50_250ms_false_positive_subliminal = 0.775`).
  - **Late Sustained Cortical-Autonomic Detector (`400–950 ms`)**: Achieves **`100.0%` true-positive sensitivity** on conscious trials with **`0.0%` false positives** on subliminal trials.

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **The Mental Privacy & Neuroethics Boundary**: Kronemer et al. emphasize that because covert physiological decoding can infer conscious recognition even when a person intentionally declines to report (`93.0%` accuracy in `Wilful_Deception_Concealed`), deploying covert decoders outside consented clinical/research contexts infringes on the innate privacy of subjective experience.
