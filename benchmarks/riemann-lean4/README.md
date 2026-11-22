# Certified Lean 4 `RiemannHypothesis` + Verifier SWE Benchmark (`benchmarks/riemann-lean4/`)

Evaluates OpenAI-compatible `/v1` coding & formal-reasoning harnesses (`OpenCode`, VS Code `Copilot Fusion`, and Google's `Elephant–Goldfish Model`) backed by Clover's 4-model `HG4-16R-OP` C11 engines against an uncontaminated, formal-kernel-verified ground-truth archive:

- **Formal Ground Truth:** Levent Alpöge + Claude's 175,187-file Lean 4 `RiemannHypothesis` archive (`239,275,554` bytes, `SHA-256 = 9631b4f020199e580ba86178cb56c6e9b09f690c2369a80dc4334aa93f49c7fc`, `leanprover/lean4:v4.33.1`, Mathlib `0df444a360eaa60ab8c11dca51a86af692955474`, zero `sorry`/`axiom`).
- **4 Graded Benchmark Tiers ([extract_rh_tasks.py](./extract_rh_tasks.py)):**
  1. `tier1_spec_roundtrip`: Cryptic machine-generated Lean proposition $\to$ human specification $\to$ formal statement equivalence.
  2. `tier2_leaf_tactic_repair`: Masked standalone `import Mathlib` leaf theorems from `Lemmata/` (`short_1_6`, `medium_7_14`, `hard_15_28` proof lines) with sibling-proof stripping, immutable `prefix_sha256` theorem-header lock, and sandboxed `< 30s` `lake env lean` kernel verification ([lean_oracle.py](./lean_oracle.py)).
  3. `tier3_egm_subgoal_audit`: Paired *provable* vs *subtly unprovable* subgoal type signatures from `Statements.lean` & `Formalism/ZetaRealNonvanish.lean` measuring whether `Elephant + Goldfish` catches false subgoals *before* proving (tracking both catch rate and Goldfish false-rejection rate).
  4. `tier4_verifier_swe`: Companion systems-engineering tasks (lock-free topological Lean DAG scheduler, nested Lean block-comment/string static scanner) with deterministic executable assertions.
- **Toggleable Clover MCP Server ([mcp_clover_tools.py](./mcp_clover_tools.py)):**
  - `s2_lean_premise_search`: Strict Trap-C anti-leakage signature search (never indexes proof bodies; excludes the target module and theorem).
  - `xcruncher_exact_math`: Exact arbitrary-precision integer, modular power, GCD/LCM, and primality checks.
- **Controlled Multi-Arm Runner ([run_ablation.py](./run_ablation.py)) & Unit Tests ([test_riemann_harness.py](./test_riemann_harness.py)):**
  - Compares `Single` (`lfm3-omni-2b`, `clover-35b-a3b`, `nemotron-3.5-super-vl`) vs `Cascade` (`clover-fusion-high`) vs `Cascade + EGM` vs `Cascade + EGM + Clover MCP` on verified pass rate and **verified passes per wall-clock hour**.
