# Gil B. Carvalho & Antonio Damasio (2021) — *Interoception and the Origin of Feelings: A New Synthesis* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Carvalho, G. B., & Damasio, A. (2021). *Interoception and the Origin of Feelings: A New Synthesis*. BioEssays, Vol. 43, No. 6, 2000261. (`doi:10.1002/bies.202000261`, `PMID: 33763881`, `OpenAlex W3136505404`, `140` citations)
- **Primary Source File**: Complete verbatim Crossref/OpenAlex/PubMed abstract and bibliographic record in [research/consciousness-and-subjective-experience/papers_text/16-carvalho-damasio-2021-interoception-feelings.txt](../../../research/consciousness-and-subjective-experience/papers_text/16-carvalho-damasio-2021-interoception-feelings.txt).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/16-carvalho-damasio-2021-interoception-feelings/run_carvalho_damasio_2021_interoception_experiments.py](../../../research/consciousness-and-subjective-experience/16-carvalho-damasio-2021-interoception-feelings/run_carvalho_damasio_2021_interoception_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/16-carvalho-damasio-2021-interoception-feelings/results/carvalho_damasio_2021_interoception_results.json](../../../research/consciousness-and-subjective-experience/16-carvalho-damasio-2021-interoception-feelings/results/carvalho_damasio_2021_interoception_results.json)

---

## 2. What the Paper Claims & Its Neuro-Physiological Mechanism

### 2.1 Feelings as the Primordial Form of Consciousness
Carvalho and Damasio argue that **feelings**—conscious mental events representing the internal state of the body as it undergoes homeostatic regulation (hunger, thirst, pain, malaise, well-being)—are the evolutionary and functional foundation of subjective experience.

### 2.2 Two Features Unique to the Interoceptive Nervous System (`INS`)
Why does the `INS` generate felt qualities with intrinsic valence, whereas exteroceptive pathways (vision, audition) produce high-resolution geometric representations?
1. **Atypical Non-Insulated Neuro-Physiology of the `INS`**:
   - Unlike fast, myelinated, electrically insulated exteroceptive fibers, the `INS` relies predominantly on **unmyelinated C-fibers** (exposed to interstitial chemistry along their axons), **fenestrated capillaries lacking a blood-brain barrier** at circumventricular organs (`Area Postrema`, `Subfornical Organ`) and sensory ganglia, and **ephaptic (non-synaptic) cross-talk**, allowing direct humoral mingling between body chemistry and neural membranes.
2. **Co-Extensive Interaction Between Object (`Body`) and Subject (`CNS`)**:
   - In exteroception, the object perceived (e.g., a visual object) is physically separate from the nervous system. In interoception, the "object" being sensed (the internal milieu) physically houses and metabolically sustains the "subject" (the central nervous system) in a closed homeostatic loop.

---

## 3. Executable Experiments Run & Observed Results

We implemented both mechanisms in [run_carvalho_damasio_2021_interoception_experiments.py](../../../research/consciousness-and-subjective-experience/16-carvalho-damasio-2021-interoception-feelings/run_carvalho_damasio_2021_interoception_experiments.py):

### Experiment E1: Interoceptive Homeostatic Feeling Loop vs. Exteroceptive-Only Observer
- **Setup**: Simulated 100 episodes (`200` steps each) of an organism subject to continuous metabolic drift across 4 internal physiological variables (`glucose, osmolality, o2_tension, core_temp`) with lethal boundary $\|b_t - b^*\|_\infty > 1.50$.
- **Observed Results**:
  - **Interoceptive Feeling-Regulated Organism**: Maintains **`100.0%` survival rate** (`mean_homeostatic_deviation = 0.2438`).
  - **Exteroceptive-Only Organism (No Interoceptive Feelings)**: Collapses to **`0.0%` survival rate** (`mean_homeostatic_deviation = 1.2474`).

### Experiment E2: Myelinated Insulated Exteroception vs. Unmyelinated Humoral + Ephaptic `INS`
- **Observed Results**:
  - **Myelinated Exteroceptive Pathway**: Concentrates **`100.0%`** of signal energy in the stimulated channel (`spatial_localization = 1.0`) and decays to **`0.000008`** by $t = 40$.
  - **Unmyelinated + Fenestrated Humoral + Ephaptic `INS`**: Diffuses **`77.25%`** of signal energy across the holistic visceral field (`holistic_field_spread = 0.7725`) and sustains a lingering affective state (**`0.2534`** at $t = 40$, **`> 30,000x` longer persistence**).

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Challenge to Disembodied Computational Functionalism**: Carvalho & Damasio's model highlights a direct tension with strict computational functionalism (Papers 13–14): in a biological organism, homeostatic deviation physically alters the metabolic medium of the neurons computing the state; in a conventional computer simulation, changing a floating-point variable labeled `hunger = 0.99` does not alter the silicon transistor voltage or physical survival of the hardware substrate.
