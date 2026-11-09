# Patrick Butlin, Tim Bayne, Stephen M. Fleming, Liad Mudrik, Megan A. K. Peters, Eric Schwitzgebel, Jonathan Simon & Rufin VanRullen (2026) — *Consciousness Indicators, Mimicry, and Internal Variants* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Butlin, P., Bayne, T., Fleming, S. M., Mudrik, L., Peters, M. A. K., Schwitzgebel, E., Simon, J., & VanRullen, R. (2026). *Consciousness Indicators, Mimicry, and Internal Variants*. Trends in Cognitive Sciences, Vol. 30, No. 7, pp. 575–576. (`doi:10.1016/j.tics.2026.04.006`, `PMID: 42036253`, `OpenAlex W7155601139`)
- **Primary Source File**: Complete PubMed (`PMID: 42036253`) and OpenAlex bibliographic record in [research/consciousness-and-subjective-experience/papers_text/15-butlin-2026-indicators-mimicry-internal-variants.txt](../../../research/consciousness-and-subjective-experience/papers_text/15-butlin-2026-indicators-mimicry-internal-variants.txt).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/15-butlin-2026-indicators-mimicry-internal-variants/run_butlin_2026_mimicry_internal_variants_experiments.py](../../../research/consciousness-and-subjective-experience/15-butlin-2026-indicators-mimicry-internal-variants/run_butlin_2026_mimicry_internal_variants_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/15-butlin-2026-indicators-mimicry-internal-variants/results/butlin_2026_mimicry_internal_variants_results.json](../../../research/consciousness-and-subjective-experience/15-butlin-2026-indicators-mimicry-internal-variants/results/butlin_2026_mimicry_internal_variants_results.json)

---

## 2. What the Paper Claims & The Problem of Mimicry and Internal Variants

### 2.1 Behavioral Mimicry, Wrapper Mimicry & Input-Output Equivalent Internal Variants
In this 2026 follow-up commentary in *Trends in Cognitive Sciences*, Butlin, Bayne, Fleming, Mudrik, Peters, Schwitzgebel, Simon, and VanRullen address two critical challenges to applying consciousness indicators to AI systems:
1. **Mimicry (`Behavioral vs. Architectural Wrapper Theatre`)**:
   - An AI system can mimic behavioral markers of consciousness, or a developer can wrap an ordinary feedforward/lookup model in superficial software modules labeled `"GlobalWorkspace"` or `"MetacognitiveMonitor"` that do not actually perform the causal information-compression (`GWT-2`) or internal noise-discrimination (`HOT-2`) functions required by the theory.
2. **Internal Variants (`Input-Output Equivalence with Divergent Internal Realizations`)**:
   - Two systems $M_1$ and $M_2$ can be **100% input-output equivalent** on external benchmarks, yet be **internal variants** whose internal causal graphs differ radically (e.g., a flat memorized lookup table vs. a wrapper mimic vs. a genuine recurrent bottleneck workspace with calibrated higher-order noise monitoring).
   - Consequently, assessing indicators requires **white-box causal interventions on the internal representations** (noise injection, bottleneck rank truncation, and quality-space interpolation), not black-box input-output benchmarks.

---

## 3. Executable Experiments Run & Observed Results

We constructed three input-output identical internal variants (`V1_Lookup_Table`, `V2_Wrapper_Theatre_Mimic`, and `V3_Genuine_GWT_HOT`) and subjected them to white-box causal interventions in [run_butlin_2026_mimicry_internal_variants_experiments.py](../../../research/consciousness-and-subjective-experience/15-butlin-2026-indicators-mimicry-internal-variants/run_butlin_2026_mimicry_internal_variants_experiments.py):

### Experiment E1: 100% Black-Box Input-Output Equivalence Across All 3 Internal Variants
- Across all 64 benchmark stimuli, `V1_Lookup_Table`, `V2_Wrapper_Theatre_Mimic`, and `V3_Genuine_GWT_HOT` achieve **`100.0%` benchmark accuracy** and **`100.0%` pairwise output agreement** (`blackbox_io_equivalence_agreement = 1.0`). No black-box behavioral test can distinguish them.

### Experiment E2: White-Box Causal Interventions Expose the Internal Variants

| Internal Variant (All `100%` Black-Box I/O Identical) | Intervention 1: `do(InternalNoise)` `HOT-2` Monitoring `AUROC` | Intervention 2: `do(Rank-4 Bottleneck)` `GWT-2` Retention Acc | Intervention 3: `HOT-4` Quality-Space Smoothness $r$ |
| :--- | :---: | :---: | :---: |
| **`V1_Lookup_Table`** (Memorized Hash Table) | **`0.5000`** (chance; ignores noise) | `1.0000` | **`-0.0119`** (orthogonal hash codes) |
| **`V2_Wrapper_Theatre_Mimic`** (Uncompressed 24-D Copy + Scripted Confidence) | **`0.0000`** (noise inflates norm & confidence!) | **`0.0000`** (collapses under rank-4 truncation) | `0.9820` |
| **`V3_Genuine_GWT_HOT`** (Rank-4 Manifold Bottleneck + Off-Manifold `HOT-2` Monitor) | **`1.0000`** (detects internal noise) | **`1.0000`** (already compressed in 4-D workspace) | **`0.9820`** (smooth circular quality space) |

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Why Norm-Based Confidence Mimics Invert Under Internal Noise Injection**: In `V2_Wrapper_Theatre_Mimic`, confidence was computed from the L2 norm of the activation vector (a common heuristic in neural networks). When internal Gaussian noise was injected into the hidden layer, the L2 norm *increased*, causing the wrapper mimic to become **more confident on corrupted representations (`AUROC = 0.0`)**—whereas `V3_Genuine_GWT_HOT` measured the off-manifold residual orthogonal to the 4-D quality subspace (`AUROC = 1.0`).
