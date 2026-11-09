# Christof Koch, Marcello Massimini, Melanie Boly & Giulio Tononi (2016) — *Neural Correlates of Consciousness: Progress and Problems* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Koch, C., Massimini, M., Boly, M., & Tononi, G. (2016). *Neural Correlates of Consciousness: Progress and Problems*. Nature Reviews Neuroscience, Vol. 17, pp. 307–321. (`doi:10.1038/nrn.2016.22`, `PMID: 27094080`)
- **Primary Source File**: Complete verbatim Key Points, abstract, figure taxonomy, and 170+ reference annotations extracted (`807,073` bytes HTML, SHA-256 `549d0f0d2cb789b5714fb2644d774ddf53e1061b40a651dc0581ce1f517b6907`, `62,254` chars in [research/consciousness-and-subjective-experience/papers_text/10-koch-2016-ncc-progress-problems.txt](../../../research/consciousness-and-subjective-experience/papers_text/10-koch-2016-ncc-progress-problems.txt)).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/10-koch-2016-ncc-progress-problems/run_koch_2016_ncc_pci_experiments.py](../../../research/consciousness-and-subjective-experience/10-koch-2016-ncc-progress-problems/run_koch_2016_ncc_pci_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/10-koch-2016-ncc-progress-problems/results/koch_2016_ncc_pci_results.json](../../../research/consciousness-and-subjective-experience/10-koch-2016-ncc-progress-problems/results/koch_2016_ncc_pci_results.json)

---

## 2. What the Paper Claims & Its Formal Mechanisms

### 2.1 Taxonomy of the NCC: Content-Specific NCC, Full NCC, and Background Enabling Conditions
Koch et al. define the **NCC** as *the minimum neuronal mechanisms jointly sufficient for any one specific conscious experience*:
1. **Content-Specific NCC**: Neurons in specialized posterior areas whose activity determines a particular phenomenal content (e.g., faces, colors, motion) whether seen, dreamed, or imagined.
2. **Full NCC**: The neural substrate supporting conscious experience in general (the union of content-specific NCCs), localized anatomically to a **Posterior Cortical Hot Zone** (`temporo-parieto-occipital`), in contrast to anterior prefrontal executive regions that can be lesioned bilaterally without abolishing conscious experience.
3. **Background Enabling Conditions**: Brainstem arousal systems and intralaminar thalamus that enable cortical excitability without determining experiential content.

### 2.2 Dissociation of Gamma Oscillations & the `P3b` Wave from Consciousness (Key Point 4)
Two historically popular candidate markers fail sufficiency/necessity tests:
- **Gamma Oscillations (`40 Hz`)**: Can occur during anesthesia and without conscious perception.
- **The `P3b` Event-Related Potential (`300–500 ms`)**: Dissociated from consciousness when consciously seen stimuli are **task-irrelevant** (`Seen = 1, Task-Relevant = 0`).

### 2.3 Perturbational Complexity Index (`PCI`, Casali et al. 2013)
Combines Transcranial Magnetic Stimulation (`TMS`) perturbation with high-density EEG to measure the joint presence of **integration** (widespread causal propagation) and **differentiation** (spatiotemporal algorithmic complexity $c_L$ via Lempel-Ziv compression normalized by source entropy $H(S)$):
$$\text{PCI} = \frac{c_L(S) \cdot \log_2 L}{L \cdot H(S)}$$

---

## 3. Executable Experiments Run & Observed Results

We implemented exact Lempel-Ziv-76 (`LZ76`) `PCI` and the factorial ERP dissociation in [run_koch_2016_ncc_pci_experiments.py](../../../research/consciousness-and-subjective-experience/10-koch-2016-ncc-progress-problems/run_koch_2016_ncc_pci_experiments.py):

### Experiment E1: Perturbational Complexity Index (`PCI`) Across 7 Clinical & Pharmacological Brain States
- **Observed Results** (`24 sources x 40 time bins`, empirical threshold $\text{PCI}^* = 0.31$):
  - **Conscious States (All $> 0.31$)**: `Awake_Alert` (`PCI = 1.0`, `LZ = 63`), `REM_Dreaming` (`PCI = 1.0`, `LZ = 51`), `Ketamine_Anesthesia` (`PCI = 1.0`, `LZ = 54`), `Locked_In_Syndrome` (`PCI = 1.0`, `LZ = 48`).
  - **Unconscious — Loss of Integration (`< 0.31`)**: `NREM_Slow_Wave_Sleep` (`PCI = 0.2447`, `LZ = 5`), `Propofol_Midazolam_Anesthesia` (`PCI = 0.2447`, `LZ = 5`).
  - **Unconscious — Loss of Differentiation (`< 0.31`)**: `Generalized_Seizure_Hypersynchrony` (`activation_rate_p1 = 0.50`, but homogeneous spatial wave yields `LZ = 4.0`, **`PCI = 0.0413`**).
  - **Single-Subject Classification Accuracy at $\text{PCI}^* = 0.31$**: **`100.0%` (`7 / 7`)**.

### Experiment E2: Dissociating the Fronto-Parietal `P3b` from Posterior `VAN`
- **Observed Results**: On `Seen_Task_Relevant` trials, both Posterior `VAN` (`-4.10 uV`) and Fronto-Parietal `P3b` (`+5.55 uV`) are strong. On **`Seen_Task_Irrelevant`** trials, Posterior `VAN` remains strong (**`-3.80 uV`**) while Fronto-Parietal `P3b` collapses to **`+0.15 uV`**, confirming that `P3b` reflects task-relevant reporting rather than conscious perception.

### Experiment E3: Why `PCI` Requires Causal Phase-Locked Thresholding (`Random Noise Trap`)
- **Observed Results**: Feeding un-thresholded single-trial white noise into `LZ76` produces a spurious **`PCI = 1.0`**, whereas applying Casali et al.'s bootstrap phase-locked causal t-test across 60 TMS trials (`|t| > 3.5`) eliminates un-locked noise and yields **`PCI = 0.0`**.

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Near-Zero Source Entropy Singularity in Raw `PCI` Normalization**: When implementing `compute_pci`, we found that if 1 out of 960 matrix entries (`p1 = 0.001`) survives thresholding by chance, source entropy $H(p_1) = 0.011$ in the denominator inflates $c_L / H(p_1)$. Enforcing Casali et al.'s minimum evoked-response gate ($p_1 \ge 0.01$) is mathematically required for numerical stability.
