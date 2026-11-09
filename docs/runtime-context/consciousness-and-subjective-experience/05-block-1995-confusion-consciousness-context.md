# Ned Block (1995) — *On a Confusion about a Function of Consciousness* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Block, N. (1995). *On a Confusion about a Function of Consciousness*. Behavioral and Brain Sciences, Vol. 18, No. 2, pp. 227–247. (`doi:10.1017/S0140525X00038188`)
- **Primary Source File**: Complete verbatim Crossref/OpenAlex target-article abstract and bibliographic record (`OpenAlex W2164446969`, `3,528` citations) in [research/consciousness-and-subjective-experience/papers_text/05-block-1995-confusion-consciousness.txt](../../../research/consciousness-and-subjective-experience/papers_text/05-block-1995-confusion-consciousness.txt) (Cambridge University Press full PDF requires JS challenge).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/05-block-1995-confusion-consciousness/run_block_1995_experiments.py](../../../research/consciousness-and-subjective-experience/05-block-1995-confusion-consciousness/run_block_1995_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/05-block-1995-confusion-consciousness/results/block_1995_confusion_consciousness_results.json](../../../research/consciousness-and-subjective-experience/05-block-1995-confusion-consciousness/results/block_1995_confusion_consciousness_results.json)

---

## 2. What the Paper Claims & Its Formal Structure

### 2.1 "Consciousness" as a Mongrel Concept: `P-Consciousness` vs. `A-Consciousness`
Block argues that philosophical and neuroscientific theories routinely conflate two fundamentally distinct concepts:
1. **Phenomenal Consciousness (`P-consciousness`)**: Experience itself; a state is P-conscious if there is something it is like to be in that state (characterized by high-capacity, fine-grained sensory/iconic representation).
2. **Access Consciousness (`A-consciousness`)**: A state is A-conscious if its representational content is **poised for free use in reasoning and for direct rational control of action and speech** (characterized by global broadcast and working-memory serial bottlenecks, `K ~ 4` items).

### 2.2 The Blindsight Conflation Fallacy
Researchers observing clinical blindsight patients note that stimuli in the blind field cannot be harnessed for spontaneous rational action or verbal report, and conclude that *the function of phenomenal consciousness is to enable information to guide rational action*.
- **Block's Diagnosis**: In clinical blindsight, stimuli in the blind field are **both** P-unconscious and A-unconscious (`P = 0, A = 0`), while in normal sighted vision they are **both** P-conscious and A-conscious (`P = 1, A = 1`). Because `P` and `A` are 100% confounded across those two conditions, an obvious function of the machinery of **A-consciousness** (global broadcast to reasoning and action controllers) is illicitly transferred to **P-consciousness**.

### 2.3 Double Dissociation: `(P=1, A=0)` Phenomenal Overflow & `(P=0, A=1)` Superblindsight
1. **`P = 1, A = 0` (Sperling 1960 Iconic Overflow)**: When a $3 \times 4$ array of 12 letters is flashed briefly, subjects report experiencing all 12 letters (`P-capacity ~ 10–12`), even though they can report only $\sim 3.5\text{–}4$ letters in whole-report (`A-capacity ~ 4`). A post-stimulus tone cueing any single row allows subjects to report $\sim 3.3 / 4$ letters from that row, proving that $\sim 10$ letters were represented in the pre-access iconic buffer prior to attentional selection.
2. **`P = 0, A = 1` (Superblindsight)**: A hypothetical (or trained) blindsight agent whose visual representations in the blind field automatically pop into working memory/reasoning (`A = 1`) to guide rational speech and action without any phenomenal visual experience (`P = 0`).

---

## 3. Executable Experiments Run & Observed Results

We implemented all three formal experiments in [run_block_1995_experiments.py](../../../research/consciousness-and-subjective-experience/05-block-1995-confusion-consciousness/run_block_1995_experiments.py):

### Experiment E1: Mathematical Proof of the Blindsight Conflation Fallacy
- **Setup**: Simulated a causal system where rational action guidance $Y_{\text{rational}}$ is driven by Access Consciousness ($\beta_A = 0.90, \beta_P = 0.00$) across 400 trials.
- **Observed Results**:
  - In the confounded design contrasting only Normal Sighted `(P=1, A=1)` vs. Clinical Blindsight `(P=0, A=0)`, the design matrix $[1, P, A]$ is **rank-deficient (`rank = 2 < 3`)**, and regressing $Y_{\text{rational}}$ on $P$ yields a spurious estimate $\hat{\beta}_P = \mathbf{0.9022}$.
  - In the unconfounded $2 \times 2$ factorial design including `(P=1, A=0)` and `(P=0, A=1)`, the design matrix has full rank (`rank = 3`), recovering the true causal effects: $\hat{\beta}_P = \mathbf{0.0007}$ and $\hat{\beta}_A = \mathbf{0.8948}$.

### Experiment E2: Sperling (1960) Tachistoscopic Iconic Overflow (`P-Buffer` vs. `A-Workspace`)
- **Setup**: Simulated 500 trials of a $3 \times 4$ (`12`-letter) display with iconic decay time constant $\tau = 300\text{ ms}$ and serial working-memory readout bottleneck ($K = 4$ items, $85\text{ ms/item}$).
- **Observed Results**:
  - **Whole-report (`A-bottleneck`)**: **`2.610 / 12` letters**.
  - **Immediate (`0 ms`) partial-report cue**: **`3.306 / 4` letters** per cued row, implying **`9.918 / 12` letters** simultaneously available in the pre-access buffer (**`3.80x` overflow ratio** above whole-report).
  - **Post-cue delay decay curve**: Implied available capacity decays smoothly from `9.918` (`0 ms`) $\to$ `7.392` (`150 ms`) $\to$ `5.898` (`300 ms`) $\to$ `4.704` (`500 ms`) $\to$ `3.696` (`1000 ms`, converging to the durable working-memory floor).

### Experiment E3: The $2 \times 2$ (`P` $\times$ `A`) Dissociation Matrix
- Verified across all 4 regimes that **spontaneous rational action guidance** tracks `A-consciousness` (`97%` in Normal Vision `(1,1)` and `94%` in Superblindsight `(0,1)` vs. `0%` in Sperling Uncued Items `(1,0)` and `2%` in Clinical Blindsight `(0,0)`), while **post-cue iconic overflow** tracks `P-buffer` capacity (`10.1` items in `(1,1)` and `(1,0)` vs. $\le 1.0$ in `(0,1)` and `(0,0)`).

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Alternative Interpretation of Sperling Overflow (Dehaene, Changeux & Kouider)**: While Block interprets the 10-letter iconic buffer as *rich P-consciousness without A-consciousness*, Global Workspace theorists (Paper 08) argue that the uncued iconic buffer is *pre-conscious* high-capacity sensory storage (`I2` in Dehaene et al. 2006) that only becomes phenomenally experienced when cued into global access—making `(P=1, A=0)` empirically contested unless no-report paradigms (Paper 09) can adjudicate between pre-conscious iconic buffers and unaccessed phenomenal experience.
2. **Relevance to AI Systems**: Nearly all "consciousness-like" functional capabilities engineered in AI architectures (global workspace broadcast, chain-of-thought reasoning, self-monitoring, verbal report) are strictly implementations of **Access Consciousness (`A-consciousness`)**. Block (1995) warns that demonstrating `A-consciousness` in a model must never be conflated with demonstrating `P-consciousness`.
