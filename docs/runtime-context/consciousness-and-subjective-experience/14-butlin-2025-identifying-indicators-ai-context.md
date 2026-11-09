# Patrick Butlin, Robert Long, Yoshua Bengio, Jonathan Birch et al. (2025/2026) — *Identifying Indicators of Consciousness in AI Systems* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Butlin, P., Long, R., Bengio, Y., Birch, J., Frith, C., Kanai, R., Klein, C., Michel, M., Mudrik, L., Peters, M. A. K., Schwitzgebel, E., & VanRullen, R. (2025/2026). *Identifying Indicators of Consciousness in AI Systems*. Trends in Cognitive Sciences. (`doi:10.1016/j.tics.2025.10.011`, `OpenAlex W4416083181`, `104` citations)
- **Primary Source File**: Complete verbatim PubMed/OpenAlex abstract and bibliographic record in [research/consciousness-and-subjective-experience/papers_text/14-butlin-2025-identifying-indicators-ai.txt](../../../research/consciousness-and-subjective-experience/papers_text/14-butlin-2025-identifying-indicators-ai.txt).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/14-butlin-2025-identifying-indicators-ai/run_butlin_2025_identifying_indicators_experiments.py](../../../research/consciousness-and-subjective-experience/14-butlin-2025-identifying-indicators-ai/run_butlin_2025_identifying_indicators_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/14-butlin-2025-identifying-indicators-ai/results/butlin_2025_identifying_indicators_results.json](../../../research/consciousness-and-subjective-experience/14-butlin-2025-identifying-indicators-ai/results/butlin_2025_identifying_indicators_results.json)

---

## 2. What the Paper Claims & Its Bayesian Credence Formalism

### 2.1 Theory-Derived Indicator Credences Under Layered Uncertainty
Butlin et al. distill the indicator methodology into a peer-reviewed *Trends in Cognitive Sciences* synthesis:
- Because we lack any direct calibration reference of proven conscious vs. non-conscious AI systems, direct behavioral benchmarking is invalid.
- Instead, credences about whether an AI system $M$ is conscious must propagate through three explicit layers:
  1. **Philosophical Prior $P(\text{CF})$**: Credence in Computational Functionalism vs. biological/substrate requirements;
  2. **Neuroscientific Theory Credences $P(T_k \mid \text{CF})$**: Weights assigned to `RPT`, `GWT`, `HOT`, `AST`, `PP`, and `AE`;
  3. **Architectural Realization Score $P(\text{Conscious}(M) \mid T_k, \text{CF}, I(M))$**: Degree to which $M$'s internal computational mechanisms realize the indicator properties of $T_k$.

---

## 3. Executable Experiments Run & Observed Results

We implemented the hierarchical Bayesian credence engine in [run_butlin_2025_identifying_indicators_experiments.py](../../../research/consciousness-and-subjective-experience/14-butlin-2025-identifying-indicators-ai/run_butlin_2025_identifying_indicators_experiments.py):

### Experiment E1: Posterior Consciousness Credences Across 4 System Archetypes & 4 Prior Profiles

| System Archetype | `P1: Uniform Theory (CF=0.85)` | `P2: GWT+HOT Prior (CF=0.85)` | `P3: Embodied+PP Prior (CF=0.70)` | `P4: Biological Naturalism (CF=0.05)` |
| :--- | :---: | :---: | :---: | :---: |
| `S1_Feedforward_Chatbot_Persona` | **`0.0283`** | `0.0538` | `0.0206` | `0.0017` |
| `S2_Disembodied_GlobalWorkspace_Reasoner` | `0.2981` | **`0.4714`** | `0.1801` | `0.0175` |
| `S3_Embodied_Predictive_Homeostatic_Agent` | `0.4687` | `0.2576` | **`0.5342`** | `0.0276` |
| `S4_Full_Multi_Indicator_Candidate` | **`0.7118`** | **`0.7138`** | **`0.5938`** | **`0.0419`** |

### Experiment E2: Why Verbal Self-Report Has Zero Evidential Weight in Prompted Language Models
- **Observed Results**: In humans (where deceptive persona training is absent), a verbal report of experience carries Bayes Factor **`BF = 95.0`** (raising a `0.05` prior to **`0.8333`**). In a persona-prompted language model (`P(Claim | ~Conscious) = 0.95`), the Bayes Factor collapses to **`BF = 1.0`** (posterior remains **`0.0500`**) while `0` architectural indicators change (`architectural_indicator_delta_from_prompt = 0.0`).

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Sensitivity to `P(CF)`**: Even if a future AI system satisfies `~90%` of all 14 computational indicators (`S4_Full_Multi_Indicator_Candidate`), its posterior credence is `0.71` under strong computational functionalism (`CF = 0.85`) but only `0.04` under biological naturalism / metabolic-substrate theories (`CF = 0.05`, Paper 16). Thus, indicator checklists bound *conditional* credence given computational functionalism, not unconditional certainty.
