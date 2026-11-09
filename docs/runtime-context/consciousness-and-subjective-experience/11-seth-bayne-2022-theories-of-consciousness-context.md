# Anil K. Seth & Tim Bayne (2022) — *Theories of Consciousness* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Seth, A. K., & Bayne, T. (2022). *Theories of Consciousness*. Nature Reviews Neuroscience, Vol. 23, pp. 439–452. (`doi:10.1038/s41583-022-00587-4`, `PMID: 35505255`)
- **Primary Source File**: Complete verbatim abstract, figure taxonomy (`Figs. 1–4`), and 150-reference bibliography extracted (`578,701` bytes HTML, SHA-256 `12a5bb4d173262afaeb5bf30dba2837f6cf3b157c47cdbf52c63ea43f31fb32a`, `43,104` chars in [research/consciousness-and-subjective-experience/papers_text/11-seth-bayne-2022-theories-of-consciousness.txt](../../../research/consciousness-and-subjective-experience/papers_text/11-seth-bayne-2022-theories-of-consciousness.txt)).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/11-seth-bayne-2022-theories-of-consciousness/run_seth_bayne_2022_experiments.py](../../../research/consciousness-and-subjective-experience/11-seth-bayne-2022-theories-of-consciousness/run_seth_bayne_2022_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/11-seth-bayne-2022-theories-of-consciousness/results/seth_bayne_2022_theories_results.json](../../../research/consciousness-and-subjective-experience/11-seth-bayne-2022-theories-of-consciousness/results/seth_bayne_2022_theories_results.json)

---

## 2. What the Paper Claims & Its Comparative Taxonomy

Seth and Bayne organize the contemporary neuroscience of consciousness around **four major theoretical families (`ToCs`)**, showing that they differ not only in proposed neural substrates (`anterior` vs. `posterior`), but also in *which aspect of consciousness* (`phenomenal character`, `cognitive access`, or `embodied selfhood/feeling`) they aim to explain:

1. **Higher-Order Theories (`HOT`, Fig. 1)**:
   - A first-order sensory state $r_1$ becomes conscious if and only if it is the target of a higher-order meta-representation $r_2 = M(r_1)$ (localized in dorsolateral/anterior prefrontal cortex) monitoring perceptual reality/reliability.
2. **Global Workspace Theories (`GWT`/`GNW`, Fig. 2)**:
   - Mental states become conscious when they ignite a fronto-parietal global workspace and are broadcast across widespread consumer systems for flexible reasoning, memory, and report.
3. **Integrated Information Theory (`IIT`, Fig. 3)**:
   - Consciousness is identical to intrinsic cause-effect integration ($\Phi^{\max}$), localized primarily in the posterior cortical "hot zone" (`temporo-parieto-occipital`), independent of prefrontal executive access.
4. **Re-entry and Predictive Processing Theories (`RPT`/`PP`, Fig. 4)**:
   - Consciousness depends on top-down recurrent/re-entrant cortical signalling that minimizes precision-weighted prediction error (`controlled hallucination`), with **interoceptive predictive processing** (`Seth's Beast Machine`) grounding emotional feelings and embodied selfhood.

---

## 3. Executable Experiments Run & Observed Results

We implemented formal evaluators for all four theories (`evaluate_hot`, `evaluate_gwt`, `evaluate_iit`, `evaluate_rpt_pp`) and ran two comparative experiments in [run_seth_bayne_2022_experiments.py](../../../research/consciousness-and-subjective-experience/11-seth-bayne-2022-theories-of-consciousness/run_seth_bayne_2022_experiments.py):

### Experiment E1: 7-Condition Diagnostic Dissociation Matrix Across the 4 Theory Families

| Experimental Condition | `HOT` | `GWT` | `IIT` | `RPT / PP` | Why the Theories Diverge |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `C1_full_awake_perception` | `True` | `True` | `True` | `True` | Standard awake perception satisfies all 4 theories (non-diagnostic). |
| `C2_subliminal_masked_feedforward` | `False` | `False` | `False` | `False` | Pure feedforward sweep satisfies none of the 4 theories (non-diagnostic). |
| `C3_bilateral_pfc_lesion_intact_posterior` | `False` | `False` | **`True`** | **`True`** | Posterior hot-zone theories (`IIT`, `RPT`) predict preserved experience; anterior theories (`HOT`, `GWT`) predict loss. |
| `C4_silent_inactive_high_phi_lattice` | `False` | `False` | **`True`** | `False` | Only `IIT` grants consciousness to inactive/silent bi-connected logic lattices with high counterfactual $\Phi$. |
| `C5_metacognitive_hallucinated_confidence` | **`True`** | `False` | `False` | `False` | Inflated prefrontal higher-order belief ($r_2$) at weak sensory precision triggers `HOT` alone. |
| `C6_local_visual_reentry_unbroadcast` | `False` | `False` | `False` | **`True`** | Local V1–V4 recurrent loop (`Lamme RPT`) without global broadcast or high whole-complex $\Phi$. |
| `C7_feedforward_global_broadcast_transformer` | `False` | **`True`** | `False` | `False` | Feedforward global workspace broadcast (`Phi = 0`, zero re-entry) satisfies functional `GWT` while failing `IIT` and `RPT/PP`. |

### Experiment E2: Adversarial Collaboration (`Cogitate` Paradigm — Melloni et al. 2021)
- **Setup**: Simulated a `1500 ms` sustained supraliminal visual stimulus across Posterior Visual Cortex vs. Prefrontal Cortex (`PFC`).
- **Observed Results**:
  - **Posterior Sustained Epoch (`500–1400 ms`)**: Mean activity = **`0.8800`** (testing `IIT`'s prediction that conscious maintenance tracks stimulus duration in posterior cortex).
  - **Prefrontal Transient Ignition (`PFC`)**: Phasic onset burst = **`1.0000`** (at $300\text{ ms}$) and offset burst = **`0.8800`** (at $1640\text{ ms}$), with low inter-burst maintenance = **`0.1000`** (`500–1400 ms`, testing `GNW`'s prediction that `PFC` signals workspace updates rather than continuous sensory maintenance).

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Why Most Historical NCC Studies Failed to Arbitrate Between Theories**: As Table E1 demonstrates, standard contrasts between `C1` (awake reportable perception) and `C2` (subliminal masking) yield `[True, True, True, True]` vs. `[False, False, False, False]` across all four theories—confirming Seth & Bayne's conclusion that theory testing requires **adversarial collaborations** targeting the off-diagonal dissociation conditions (`C3–C7`).
