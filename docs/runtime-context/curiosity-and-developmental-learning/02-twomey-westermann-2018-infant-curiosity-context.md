# Katherine E. Twomey & Gert Westermann (2018) — *Curiosity-Based Learning in Infants: A Neurocomputational Approach* — Context

Offered, not imposed. Date: 2026-11-08. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Twomey, K. E., & Westermann, G. (2018). *Curiosity-based learning in infants: a neurocomputational approach*. Developmental Science, Vol. 21, Issue 4, e12629. (`doi:10.1111/desc.12629`, `PMCID: PMC6032944`, `PMID: 29071759`, CC BY 4.0)
- **Primary Source File**: `https://pmc.ncbi.nlm.nih.gov/articles/PMC6032944/` (`235,228` bytes HTML / `143,769` bytes EuropePMC XML, extracted text `73,291` chars in `research/curiosity-and-developmental-learning/papers_text/02-twomey-westermann-2018-infant-curiosity.txt`).
- **Executable Experiments**: [research/curiosity-and-developmental-learning/02-twomey-westermann-2018-infant-curiosity/run_twomey2018_experiments.py](../../../research/curiosity-and-developmental-learning/02-twomey-westermann-2018-infant-curiosity/run_twomey2018_experiments.py)
- **Results Artifact**: [research/curiosity-and-developmental-learning/02-twomey-westermann-2018-infant-curiosity/results/twomey_2018_results.json](../../../research/curiosity-and-developmental-learning/02-twomey-westermann-2018-infant-curiosity/results/twomey_2018_results.json)

---

## 2. What the Paper Claims & Its Mathematical Mechanism

### 2.1 Subjective Novelty Modulated by Learner Plasticity (Section 1.3–1.4, Eq. 1)
Twomey and Westermann address a longstanding conflict in developmental psychology: why do some infant studies find optimal learning from *predictable/simple* sequences (Bulf et al., 2011), others from *maximally variable* sequences (Mather & Plunkett, 2011), and others from *intermediate "Goldilocks"* sequences (Kidd, Piantadosi & Aslin, 2012, 2014)?

They propose that infant visual exploration is driven by a **4-input $\to$ 3-hidden $\to$ 4-output sigmoidal autoencoder** (modeling Sokolov's 1963 novelty-orienting & habituation comparator) trained via the Generalized Delta Rule (Rumelhart, Hinton & Williams, 1986):
$$\Delta w = \eta (i - o) o (1 - o)$$
where:
- $(i - o)$ is **subjective novelty**: the discrepancy between the external stimulus $i \in [0, 1]^4$ and the infant's current internal reconstruction $o \in (0, 1)^4$.
- $o(1 - o)$ is the derivative of the logistic sigmoid activation function, which the authors interpret as the **plasticity (readiness to learn)** of the internal representation (maximal at $0.25$ when $o = 0.5$, approaching $0$ when $o \to 0$ or $o \to 1$).
- **The Curiosity Selection Rule (Section 3.2.1)**: When scanning candidate stimuli in the environment before choosing which one to fixate and learn from next, the infant chooses the stimulus $i$ that maximizes the magnitude of the delta-rule update term:
  $$C(i, o) = \sum_{d=1}^{4} |i_d - o_d| \, o_d (1 - o_d)$$

### 2.2 Experimental Paradigm: Younger (1985) & Mather & Plunkett (2011) 4-Feature Animal Categorization
- **Training Stimuli**: $8$ animal exemplars defined by $4$ continuous perceptual features (`neck_length`, `leg_length`, `ear_separation`, `tail_width`, taking values in $\{1, 2, 4, 5\}$ normalized into $(0, 1)$ around center $3 \mapsto 0.5$, giving $8! = 40,320$ possible presentation orders).
- **Encoding / Habituation Loop**: Each chosen stimulus is trained for up to `20` sweeps or until $\text{SSE} = \sum_d (i_d - o_d)^2 < 0.01$.
- **Category Formation Test**: At test (with weights frozen), the network is presented with an unseen **category-central prototype** (`[3, 3, 3, 3]`) and two unseen **category-peripheral exemplars** (`[1, 1, 1, 1]` and `[5, 5, 5, 5]`). Category formation is measured by the proportion of test SSE directed to the peripheral stimuli:
  $$\text{PropSSE}_{\text{peripheral}} = \frac{\text{SSE}_{\text{peripheral}}}{\text{SSE}_{\text{peripheral}} + \text{SSE}_{\text{central}}}$$

---

## 3. Executable Experiments Run & Observed Results

We implemented `InfantAutoencoder` and the full $40,320$-permutation sequence analysis in [run_twomey2018_experiments.py](../../../research/curiosity-and-developmental-learning/02-twomey-westermann-2018-infant-curiosity/run_twomey2018_experiments.py):

### Experiment E1: Fixed Presentation Orders (Mather & Plunkett 2011 Replication, Section 2)
Ranking all $8! = 40,320$ permutations of the 8 training stimuli by mean successive Euclidean distance (`ED`) and evaluating $24$ simulated infants per condition:

| Condition (`n = 24` each) | Mean Sequence `ED` | First 3-Trial Block SSE | Last 3-Trial Block SSE | Training SSE Decrease | Median Test $\text{PropSSE}_{\text{peripheral}}$ | Mean Test $\text{PropSSE}_{\text{peripheral}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`max`** (Top 24 `ED` orders) | `1.1541` | `0.4849` | `0.3892` | **`0.0957`** | **`0.9701`** | **`0.9708`** (`24/24 > 0.5`) |
| **`random`** (24 random orders) | `0.9118` | `0.5040` | `0.4228` | `0.0812` | `0.9322` | `0.9208` (`24/24 > 0.5`) |
| **`med`** (Median 24 `ED` orders) | `0.9087` | `0.5010` | `0.4103` | `0.0907` | `0.9141` | `0.9158` (`24/24 > 0.5`) |
| **`min`** (Bottom 24 `ED` orders) | `0.6866` | `0.5229` | `0.4503` | `0.0726` | `0.8655` | `0.8611` (`24/24 > 0.5`) |

- **Verification**: Reproduces every statistical effect reported in Section 2.4 of Twomey & Westermann (2018): familiarization SSE decreases from the first to the last 3-trial block, overall training error is lower and drops more in `max` than `min`, and category formation strength follows the exact ordering $\text{max} > \text{random} > \text{med} > \text{min}$.

### Experiment E2: Active Stimulus Selection ("Setting the Model Free", Section 3)
Comparing the 4 active stimulus selection rules (sampling without replacement across the 8 stimuli, $n = 24$ runs each):

| Selection Mechanism | Formula Maximized | Median Test $\text{PropSSE}_{\text{peripheral}}$ | Mean Test $\text{PropSSE}_{\text{peripheral}}$ | Mean Sequence `ED` | Mean Successive `ED` Across Steps $1 \dots 7$ |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`curiosity`** | $\sum_d |i_d - o_d|\,o_d(1 - o_d)$ | **`0.9663`** | **`0.9676`** | `1.0567` | `[1.18, 1.39, 1.30, 1.33, 0.85, 0.68, 0.67]` |
| **`subjective_novelty`** | $\sum_d |i_d - o_d|$ | `0.9666` | `0.9676` | `1.0579` | `[1.18, 1.39, 1.30, 1.32, 0.86, 0.68, 0.67]` |
| **`objective_complexity`** | $\|i - i_{\text{prev}}\|_2$ | `0.9643` | `0.9656` | `1.0962` | `[1.28, 1.39, 1.23, 1.35, 1.05, 0.71, 0.66]` |
| **`plasticity`** | $\sum_d o_d(1 - o_d)$ | `0.9240` | `0.9128` | `0.8978` | `[0.94, 0.94, 0.96, 1.01, 0.86, 0.82, 0.76]` |

### Experiment E3: Where the Explanation Breaks (Two Critical Boundary Tests)
1. **Challenge 3A — Sigmoid Saturation Blindness ("Confident-but-Wrong" Suppression)**:
   - Because the squared-error + sigmoid delta rule multiplies prediction error $|i - o|$ by the sigmoid derivative $o(1 - o)$, if prior learning has pushed an output unit to a confident belief $o = 0.01$ and the agent encounters a strongly contradicting counterexample $i = 0.95$ ($|i - o| = 0.94$), the plasticity factor is $o(1 - o) = 0.0099$, yielding a tiny curiosity score of **`0.00931`**.
   - By contrast, a mildly discrepant stimulus ($i = 0.70, o = 0.50$, $|i - o| = 0.20$) has $o(1 - o) = 0.25$ and receives a curiosity score of **`0.0500`** (**`5.37x` higher** than the strong counterexample!).
   - Thus, $|i - o|o(1 - o)$ actively **avoids** strong counterexamples whenever its existing representation is saturated—the exact pathology that motivated replacing squared-error sigmoid outputs with cross-entropy loss (where $\partial \mathcal{L} / \partial z = o - i$ removes the $o(1 - o)$ saturation damping).
2. **Challenge 3B — Dependence on the External "Without Replacement" Constraint (The Antipodal Limit Cycle Trap)**:
   - In Section 3.1 (p. 6), Twomey & Westermann enforced sampling **without replacement** (each of the 8 stimuli was removed from the environment after a single trial). They attributed the model's "outside-in" shift (selecting peripheral stimuli first, then central stimuli) purely to the curiosity function $|i - o|o(1 - o)$.
   - When we removed the artificial "without replacement" constraint so all 8 stimuli remained available in the environment across 40 trials, **the model never once visited any of the 4 inner category stimuli (`4, 5, 6, 7`)**!
   - Instead, it locked into an antipodal limit cycle on the outer shell:
     $$\text{Sequence}: [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 1, 2, \dots] \quad (\text{visits}: \{0: 11, 1: 9, 2: 9, 3: 11\}, \text{ stimuli } 4\text{–}7: \mathbf{0})$$
   - **Why**: Training the shared 3-hidden-unit bottleneck on peripheral corner `0` (`[1, 1, 5, 5]`) overwrites the shared weights and pulls antipodal corner `3` (`[5, 5, 1, 1]`) back toward $o \approx 0.50$, making `3` maximally curious again on the very next step! The "outside-in" developmental trajectory reported in the paper is therefore a joint product of the curiosity rule **and** the external without-replacement depletion of the peripheral stimuli.
