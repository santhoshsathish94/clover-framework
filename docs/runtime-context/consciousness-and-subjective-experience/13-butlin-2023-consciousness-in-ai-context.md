# Patrick Butlin, Robert Long, Eric Elmoznino, Yoshua Bengio, Jonathan Birch et al. (2023) — *Consciousness in Artificial Intelligence: Insights from the Science of Consciousness* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Butlin, P., Long, R., Elmoznino, E., Bengio, Y., Birch, J., Constant, A., Deane, G., Fleming, S. M., Frith, C., Ji, X., Kanai, R., Klein, C., Lindsay, G., Michel, M., Mudrik, L., Peters, M. A. K., Schwitzgebel, E., Simon, J., & VanRullen, R. (2023). *Consciousness in Artificial Intelligence: Insights from the Science of Consciousness*. `arXiv:2308.08708v3` (88 pages). (`doi:10.48550/arXiv.2308.08708`)
- **Primary Source File**: Full 88-page PDF downloaded (`1,763,829` bytes, SHA-256 `88a9d5635b274b7d7c8c55dfeea45be3018054be01da83cd2f8d1fe53d668ab8`, `271,235` extracted text chars in [research/consciousness-and-subjective-experience/papers_text/13-butlin-2023-consciousness-in-ai.txt](../../../research/consciousness-and-subjective-experience/papers_text/13-butlin-2023-consciousness-in-ai.txt)).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/13-butlin-2023-consciousness-in-ai/run_butlin_2023_ai_indicators_experiments.py](../../../research/consciousness-and-subjective-experience/13-butlin-2023-consciousness-in-ai/run_butlin_2023_ai_indicators_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/13-butlin-2023-consciousness-in-ai/results/butlin_2023_ai_indicators_results.json](../../../research/consciousness-and-subjective-experience/13-butlin-2023-consciousness-in-ai/results/butlin_2023_ai_indicators_results.json)

---

## 2. What the Paper Claims & Its 14 Indicator Properties

### 2.1 Methodological Framework (pp. 4–6)
1. **Pragmatic Working Hypothesis (`Computational Functionalism`)**: Assumes as a working hypothesis that performing computations of a certain functional organization is necessary and sufficient for consciousness. (The authors explicitly exclude `IIT` from the computational rubric because `IIT` rejects computational functionalism.)
2. **Theory-Heavy Assessment over Behavioral Imitation**: Because language models can be trained to imitate human self-reports of consciousness while working in fundamentally different ways, AI assessment must inspect **internal computational architecture** against **14 Indicator Properties** (`Table 1, p. 5`):
   - **Recurrent Processing Theory (`RPT`)**: `RPT-1` (algorithmic recurrence in input modules), `RPT-2` (organized, integrated perceptual representations).
   - **Global Workspace Theory (`GWT`)**: `GWT-1` (parallel specialist modules), `GWT-2` (limited-capacity workspace bottleneck + selective attention), `GWT-3` (global broadcast to all modules), `GWT-4` (state-dependent attention querying modules in succession).
   - **Computational Higher-Order Theories (`HOT`)**: `HOT-1` (generative/top-down or noisy perception modules), `HOT-2` (metacognitive monitoring distinguishing reliable perceptual representations from noise), `HOT-3` (agency guided by general belief-formation updated by metacognitive output), `HOT-4` (sparse and smooth coding generating a quality space).
   - **Attention Schema Theory (`AST`)**: `AST-1` (predictive model representing and enabling control over the current state of attention).
   - **Predictive Processing (`PP`)**: `PP-1` (input modules using predictive coding).
   - **Agency & Embodiment (`AE`)**: `AE-1` (learning from feedback and pursuing competing goals flexibly), `AE-2` (modeling output-input sensorimotor contingencies and using them in perception/control).

---

## 3. Executable Experiments Run & Observed Results

We implemented the 14-indicator rubric and mechanistic modules for `HOT-1..4` and `GWT-1..4` in [run_butlin_2023_ai_indicators_experiments.py](../../../research/consciousness-and-subjective-experience/13-butlin-2023-consciousness-in-ai/run_butlin_2023_ai_indicators_experiments.py):

### Experiment E1: Evaluating the 5 Case-Study AI Systems (`Section 3.2`)

| Case-Study System (`Section 3.2`) | Full Indicators (`1.0`) | Partial Indicators (`0.5`) | Weighted Fraction (`/ 14`) | Core Neuro Conjunction (`min(RPT, GWT, HOT, PP)`) |
| :--- | :---: | :---: | :---: | :---: |
| `Transformer_LLM_Autoregressive` | `0` | `6` | **`21.43%`** | **`0.0`** |
| `Perceiver_IO_CrossAttention` | `2` (`GWT-1,2`) | `4` | **`28.57%`** | **`0.0`** |
| `DeepMind_Adaptive_Agent_AdA` | `3` (`RPT-1, AE-1,2`) | `4` | **`35.71%`** | **`0.0`** |
| `Merel_Virtual_Rodent` | `2` (`RPT-1, AE-2`) | `5` | **`32.14%`** | **`0.0`** |
| `PaLM_E_Embodied_Multimodal` | `1` (`GWT-1`) | `9` | **`39.29%`** | **`0.0`** |

### Experiment E2: `HOT-2` Reality/Noise Monitoring Over a `HOT-4` Smooth Quality Manifold
- **Observed Results**: On 400 trials (`50%` true smooth stimulus, `50%` high-variance internal noise), a first-order energy threshold suffers an **`80.50%` false-belief hallucination rate** on noise trials, whereas the `HOT-2` higher-order manifold-coherence monitor achieves **`0.0%` false beliefs**, **`100.0%` true stimulus detection**, and **`AUROC = 1.0`**.

### Experiment E3: `GWT-1..4` Sequential Workspace-Mediated Module Querying
- **Observed Results**: On 100 2-hop compositional tasks (`Vision -> Workspace -> SemanticMemory -> Workspace -> Motor`), intact `GWT-1..4` sequential querying achieves **`100.0%` accuracy** vs. **`12.5%` (chance)** when `GWT-4` sequential feedback querying is ablated.

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Authorial Footnote 1 Caveat (`p. 1`)**: Butlin et al. explicitly amended their abstract to state: *"Satisfying the indicators would not mean that such an AI system would definitely be conscious."* The indicators are derived under the unproved assumption of computational functionalism and address functional correlates (`A-consciousness`, metacognition, agency) rather than proving phenomenal experience (`P-consciousness`).
