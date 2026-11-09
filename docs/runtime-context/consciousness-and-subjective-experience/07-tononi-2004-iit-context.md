# Giulio Tononi (2004) — *An Information Integration Theory of Consciousness* — Context

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

## 1. Source Provenance & Status

- **Citation**: Tononi, G. (2004). *An Information Integration Theory of Consciousness*. BMC Neuroscience, Vol. 5, Article 42, pp. 1–22. (`doi:10.1186/1471-2202-5-42`)
- **Primary Source File**: Full open-access 22-page BMC Neuroscience PDF (`2,680,174` bytes, SHA-256 `27a598c5c538d7c1e9a1e4e60ae081483313db298736e84645dc561cf5e83ef7`, `116,460` extracted text chars in [research/consciousness-and-subjective-experience/papers_text/07-tononi-2004-iit.txt](../../../research/consciousness-and-subjective-experience/papers_text/07-tononi-2004-iit.txt)).
- **Executable Experiments**: [research/consciousness-and-subjective-experience/07-tononi-2004-iit/run_tononi_2004_iit_experiments.py](../../../research/consciousness-and-subjective-experience/07-tononi-2004-iit/run_tononi_2004_iit_experiments.py)
- **Results Artifact**: [research/consciousness-and-subjective-experience/07-tononi-2004-iit/results/tononi_2004_iit_results.json](../../../research/consciousness-and-subjective-experience/07-tononi-2004-iit/results/tononi_2004_iit_results.json)

---

## 2. What the Paper Claims & Its Mathematical Formalism

### 2.1 Differentiation + Integration = Integrated Information $\Phi(S)$
Tononi starts from two phenomenological properties of experience:
1. **Differentiation (Information)**: Every conscious experience rules out an immense repertoire of alternative experiences (unlike a 1-bit photodiode).
2. **Integration (Unity)**: Each conscious experience is a unified whole that cannot be decomposed into independent components (unlike a 1-megapixel digital camera sensor chip of $10^6$ disconnected photodiodes).

### 2.2 Exact Analytical Gaussian Process Equations (Figure 1 Methodological Note, p. 5)
For a system $X$ of $n$ neural elements with causal connection matrix $\text{CON}(X)$ under stationary linear Gaussian dynamics $X = X \cdot \text{CON}(X) + c R \implies Q = (I - \text{CON}(X))^{-1}$:
1. **Directed Effective Information $EI(A \to B)$**:
   - Bisect subset $S \subseteq X$ into $(A, B)$.
   - Enforce independent maximum-entropy noise $A^{H_{\max}}$ (`cp = 1.0`) by zeroing all connections *within* $A$ and *afferent to* $A$ (`CON[:, A] = 0`), while applying low intrinsic noise (`ci = 10^-5..10^-4`) to $X \setminus A$.
   - From $\text{COV}(X) = Q_A^\top \text{diag}(c^2) Q_A$, compute:
     $$EI(A \to B) = MI(A^{H_{\max}}; B) = \frac{1}{2 \ln 2} \ln \frac{|\text{COV}(A)| \cdot |\text{COV}(B)|}{|\text{COV}(AB)|}, \qquad EI(A \leftrightarrow B) = EI(A \to B) + EI(B \to A)$$
2. **Minimum Information Bipartition (`MIB`) & $\Phi(S)$**:
   - Normalize across bipartition sizes by $H^{\max}(A \leftrightarrow B) = \min\{H^{\max}(A), H^{\max}(B)\}$:
     $$\text{MIB}(S) = \arg\min_{(A, B)} \frac{EI(A \leftrightarrow B)}{\min(|A|, |B|)}, \qquad \Phi(S) = EI\big(\text{MIB}(S)\big)$$
3. **Complexes & Qualia Space (Figures 1c & 2, pp. 4–8)**:
   - A **complex** is a subset $S$ with $\Phi(S) > 0$ that is not included in any larger subset $S' \supset S$ with higher $\Phi$. The complex with maximal $\Phi$ is the **main complex**.
   - The **quality** of consciousness (`Qualia Space`) is determined by the full **Effective Information Matrix** $[EI(A \to B), EI(B \to A)]$ across all bipartitions of the main complex.

---

## 3. Executable Experiments Run & Observed Results

We implemented the complete analytical Gaussian `EI / MIB / Phi / Complex` engine in [run_tononi_2004_iit_experiments.py](../../../research/consciousness-and-subjective-experience/07-tononi-2004-iit/run_tononi_2004_iit_experiments.py):

### Experiment E1: Digital Camera Sensor Array vs. Integrated Complex (pp. 2–4)
- **Observed Results**: A 6-element disconnected camera sensor has `6.0 bits` of external binary state entropy but **`Phi = 0.0 bits`** across its minimum information bipartition, whereas a 6-element integrated ring-lattice achieves **`Phi = 43.2551 bits`**.

### Experiment E2: Figure 2 Reproduction — Divergent vs. Chain Qualia Spaces at Equal $\Phi = 10.0\text{ bits}$ (pp. 6–8)
- **Observed Results**:
  - Under 1-step causal transition dynamics ($c_i = 2^{-11}$), both the 4-element Divergent digraph (`0 -> {1,2,3}`) and the 4-element Chain digraph (`0 -> 1 -> 2 -> 3`) form a 4-node complex with **`Phi = 10.0 bits`**.
  - In the Chain complex, the alternating bipartition $\{0, 2\} \to \{1, 3\}$ (`{1,3} -> {2,4}` in 1-indexed Fig. 2b) cuts two parallel links (`0->1` and `2->3`), yielding **`20.0 bits`** (exactly twice all single-cut entries, matching p. 8), and the Frobenius distance between the two complexes' Qualia Space `EI` matrices is **`22.3794 bits`**.

### Experiment E3: Figure 3 & Figure 4a — Thalamocortical vs. Homogeneous All-to-All vs. Cerebellar Modular (pp. 9–11)
- **Observed Results** (all 6-node networks normalized to identical afferent weight sum $w = 0.5$ per node):
  - **Specialized & Integrated (Thalamocortical-like)**: **`Phi = 55.7282 bits`**.
  - **Homogeneous All-to-All (Loss of Specialization)**: Drops by $58.5\%$ to **`Phi = 23.1015 bits`**.
  - **Strongly Modular (Cerebellar-like, $3 \times 2$ modules)**: Whole-system **`Phi = 0.0 bits`**, decomposing into **`3` separate local complexes** (`Phi = 24.5754 bits` each).

### Experiment E4: Figure 4c/d Port Exclusion, Figure 5a Split-Brain Callosotomy & Aaronson's Expander Boundary
- **Observed Results**:
  - **Ports-In / Ports-Out Exclusion (Fig. 4c/d)**: A 4-node core complex achieves **`Phi = 40.3789 bits`**, whereas adjoining a sensory afferent node (`4 -> 0`) and motor efferent node (`3 -> 5`) drops the 6-node subset to **`Phi = 20.4576 bits`** (bottlenecked at the single-port cut), excluding the sensory/motor ports from the main complex.
  - **Split-Brain Callosotomy (Fig. 5a)**: Cutting inter-hemispheric connections splits the single 6-node main complex (`[0,1,2,3,4,5]`, `Phi = 57.6787 bits`) into **two separate 3-node hemispheric complexes** (`[0,1,2]` and `[3,4,5]`, `Phi = 23.2877 bits` each).
  - **Aaronson (2014) Passive Expander-Grid Boundary**: A purely passive 6-node circulant expander matrix with zero sensory input, zero world model, and zero task semantics achieves **`Phi = 42.9253 bits`**, outscoring a 6-node cognitive processing hierarchy (`Sensory -> Feature -> Memory -> Planner -> Motor`, **`Phi = 12.0362 bits`**) by **`3.57x`**.

---

## 4. Surprises, Limitations & Epistemic Boundary

1. **Redundant Fan-Out vs. Independent Parallel Channels in Figure 2**: While implementing Figure 2, we discovered that in the Divergent star (`0 -> {1,2,3}`), a single scalar source $X_0$ drives both targets in $B = \{1, 2\}$ across the 2-vs-2 cut $A = \{0, 3\}, B = \{1, 2\}$. Because $X_1$ and $X_2$ receive the *same* scalar perturbation $X_0$, signal variance doubles ($2 w^2$) rather than adding a second independent degree of freedom, requiring $w_{\text{div}} = w_{\text{chain}} / \sqrt{2}$ to match $\Phi = 10.0\text{ bits}$ on the 2-vs-2 MIB cut.
2. **Combinatorial Intractability & Passive Grid Inflation**: Exact $\Phi$ evaluation scales super-exponentially ($\sum_{k=2}^n \binom{n}{k} 2^{k-1}$ bipartitions), and treating high $\Phi$ as a *sufficient* condition for consciousness assigns high consciousness to passive expander grids and error-correcting parity lattices (`42.93 bits`).
