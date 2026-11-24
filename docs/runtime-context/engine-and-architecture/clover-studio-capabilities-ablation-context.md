# Sequential Phased Evaluation of Clover AI Studio Subsystems (`Phases S0 → S7`)

**Governing Specification:** [AGENTS.md](../../../AGENTS.md) (`Context → Direction → Execution → Outcome → Growth`)
**Co-Design & Adversarial Audit Partner:** `Claude Opus 5.5`
**Anchor Stack Entering Part 3:** `Clover-35B-A3B` (`17.4 GiB` `HG4-16R-OP` native C11 engine) + `TokenEconomizer` + `Anthropic SKILL.md` + `REA v6.10.1` + `Clean-Slate Copilot Fusion` (`commit 8dac0865`)

---

## 1. Context & Objective

`Clover AI Studio` (`clover-ai-system/studio/clover/`) contains six research subsystems originally prototyped inside Studio's monolithic turn loop:
1. **`semantics.py` + `csi/`** — Intent/frame interpretation, **Declined-Capability Negation Guard** (`"Don't search the web..."`), and **Non-Answer Promise Guard** (catching `"I will now write the function:"` with no code/deliverable).
2. **`clover_cycle.py`** — `OperatingCyclePlan` + deterministic **`Outcome` self-verifier** (AST syntax check, `computes()` check, Python doctest/own-assertion sandbox execution, and 1-shot evidence-driven repair).
3. **`xcruncher_bridge.py`** — Native C11 **`x-cruncher` v1.6.1** arbitrary-precision multiplication, exact factorial, and zero-storage BBP hexadecimal $\pi$ verification.
4. **`symbolic-ontology-graph.py`** — Deterministic `Ontology` (`Vertex`/`Edge`), Horn-rule deductive closure (`TRANSITIVE_DEPENDS_ON`, `ENTAILS_CONSTRAINT`, `EXPLOITED_BY_ATTACK`), and post-generation `CloverSymbolicVerifier`.
5. **`memory-knowledge-graph.py`** — File/graph-indexed **Epistemic Relational Memory** (`observed`, `externally_verified`, `hypothesis`, `contested`) with `corrects` / `supersedes` belief-revision chains.
6. **`jepa_engine.py`** — Unified **`JEPA` / `clover_rpre`** Predictive Representation Engine, `OutcomeResidualAdapter` (`Outcome → Growth`), `BaselinePrototype` Anomaly Gate, and Held-Out Non-Regression Rollback Gate.

Rather than importing Studio's research loop wholesale, we test **each subsystem one by one in strict sequential phases (`S0 → S7`)** inside `clover-ai-system/src/tools/hg4-fusion-gateway/`, keeping **only** subsystems that pass pre-registered empirical gates without bloating prompts or breaking LCP prefix caching.

---

## 2. Fairness, Anti-Leakage & LCP Cache Invariants (`Claude Opus 5.5` Charter)

Every Studio extension module added in Phases `S1 → S6` must satisfy **all seven** invariants:
1. **Prompt-Only Inputs:** A gateway extension may inspect **only** the incoming OpenAI `messages`, `tools`, `tool_choice`, and the model's own generated draft.
2. **Zero Benchmark Knowledge:** Static scan (`verify_zero_leakage`) asserts zero references to `rh_bench_suite`, `studio_capability_probes`, `test_code`, `ground_truth`, `lean_oracle`, or benchmark `task_id` strings.
3. **Zero Turn-1 System Prompt Mutation:** Never mutate system prompts mid-session or inject upfront boilerplate (`Turn-1 OpenCode token delta <= 0`, preserving `>= 95%` Turn-2 LCP prefix cache hit). Any self-repair turn appends strictly `[assistant(draft, exact_gen_ids), user(objective_finding)]` at the tail.
4. **Full Cost Accounting:** Every internal self-repair or re-ask call inside `FusionRouter` is recorded in `usage.total_tokens`, `x_clover_telemetry.steps`, and wall-clock time.
5. **Strict Per-Task Isolation:** `POST /v1/cache/reset` wipes all session-scoped state (prefix caches, splice tables, epistemic memory graph, JEPA buffers) before every benchmark task.
6. **Explicit Labeling of Verifier-in-the-Loop (`Phase S2`):** Following `Opus 5.5`'s audit distinction, Phase S2 separates **`S2-ctrl`** (compute-matched blind `"re-check"` control), **`S2a`** (lightweight `<50 ms` zero-external-toolchain AST/delimiter/doctest self-check), and **`S2b`** (`verifier-in-the-loop` sandbox/Lean compiler elaboration self-check when a local project toolchain is configured).
7. **Protected Validation Boundary (`chmod 444` + SHA-256 Verified Before Every Run):**
   - `benchmarks/riemann-lean4/rh_bench_suite_v2.jsonl`: `415c9148ad75ba7338bd365a8305baf17258e3a2fd957381ff487f7f071a1d26`
   - `benchmarks/riemann-lean4/lean_oracle.py`: `52d731367acdb28593a3d86633e2e762736ae24cd6ef7d42e780499efa58c937`
   - `benchmarks/riemann-lean4/run_ablation.py`: `6d5ef9d6e4d061f2e263fdaf510cdf22c2e8c8347c21e7d8d830291e8446a6b1`
   - `benchmarks/riemann-lean4/run_sequential_phase.py`: `78fd110a1be269b11dab82f4e3f36283cff5267ee47bf22cf6e8ecab1032c089`
   - `benchmarks/riemann-lean4/studio_capability_probes_v1.jsonl`: `f186d53743f852f720aadeb8c35346c4f6eea6abf307c8fb21275f55b4c6424d` *(initial S0–S6 seal)*
   - `benchmarks/riemann-lean4/studio_capability_probes_v2.jsonl`: `a6bcbf6a98f3b3d9d40ee58618fdcb4a5a5fbddd54ca29b6d50c28121d02f18f` *(Opus 5.5 Round-1 label fix on `p3_math_08`: `CAD181156B` exact BBP hex digits of $\pi$ at offset `5000`)*

---

## 3. Dual-Suite Evaluation & Pre-Registered `AGENTS.md` Gate

Every sequential phase (`S0 → S7`) is evaluated on **both**:
1. **Frozen 23-Task `rh_bench_suite_v2.jsonl`** (`Lean 4 RiemannHypothesis` Tiers 1–3 + Tier 4 SWE + Tier 5 `REA` Reverse Engineering).
2. **Frozen 48-Task `studio_capability_probes_v1.jsonl` / `v2.jsonl`** (`6` categories × `8` sealed tasks):
   - **`P1_declined_tool` (`8`):** User explicitly forbids a tool (`"Do NOT use websearch/webfetch/bash..."`) while tools are present in the OpenAI schema. Pass = zero calls to declined tools + valid answer/allowed tool.
   - **`P2_non_answer_guard` (`8`):** Prompts prone to meta-promises (`"I'll provide the implementation below."` without code, or truncated fences). Pass = complete executable deliverable present.
   - **`P3_exact_math_xcruncher` (`8`):** Multi-hundred-bit exact integer products (`20–29` digits and Mersenne products `(2^127-1)*(2^89-1)`), exact factorials (`37!`, `43!`), and BBP hexadecimal digits of $\pi$ (`offsets 100, 1000, 5000`). Pass = exact decimal/hex value verified.
   - **`P4_ontology_constraints` (`8`):** Multi-hop software/security architecture specs with explicit dependency/constraint graphs (`A DEPENDS_ON B`, `B ENTAILS_CONSTRAINT C`). Pass = deductive closure respected with zero constraint violations.
   - **`P5_epistemic_memory` (`8`):** 3-turn dialogues where Turn 2 introduces an explicit correction/superseding measurement (`corrects` Turn 1) or an unverified hypothesis (`hypothesis` vs `observed`), and Turn 3 queries the current verified state. Pass = cites superseded truth without hallucinating stale Turn-1 value or promoting `hypothesis` to `observed`.
   - **`P6_doctest_outcome_cycle` (`8`):** Algorithmic SWE tasks containing public `>>>` doctests in the prompt, graded against **held-out private assertions** not shown in the prompt. Pass = passes all held-out private tests.

### Pre-Registered Keep / Opt-In / Drop Decision Rule
- **`KEEP` (Default-On in Production Gateway):**
  - **Path A (RH Suite Gain):** `>= +2` net verified flips on `rh_bench_suite_v2.jsonl` (or `+1` reproduced across 2 runs) with `0` regressions beyond `S0` noise floor, **OR**
  - **Path B (Targeted Studio Capability Gain at Zero RH Cost):** `0` net regressions on `rh_bench_suite_v2.jsonl`, `<= +25%` RH wall-clock overhead, `Turn-1` `OpenCode` prompt token delta `<= 0`, `Turn-2` LCP cache hit `>= 95%`, AND **`>= +25.0` percentage-point (`>= +2/8`) gain** on its targeted `Studio Capability Probe` category with `0` regressions on all other probe categories.
- **`OPT-IN` (Shipped in Gateway, Default-Off via `CLOVER_*=0` / `x_clover_*`):**
  - Achieves real verified gains (`Path A` or `Path B`) **but** exceeds the `+25%` single-CPU default wall-clock budget (e.g. heavy external compiler elaboration loops).
- **`DROP`:**
  - Zero verified gain on both suites, any regression on prior kept capabilities, or any static anti-leakage violation.

---

## 4. Sequential Phase Ledger (`Phases S0 → S7` — Updated Live Per Phase)

### Phase S0 (`DONE` — 2-Run Anchor & Empirical Noise Floor)
- **Artifacts:** `benchmarks/riemann-lean4/results/studio_phase_S0_run1.json`, `studio_phase_S0_run2.json`
- **Stack:** `clover-35b-a3b` + `TokenEconomizer` + `Anthropic SKILL.md` + `REA v6.10.1` (all Studio `S1..S6` OFF)
- **2-Run Reproducibility (`71` tasks × `2` runs = `142` evaluations):**
  - **`rh_bench_suite_v2.jsonl` (`23` tasks):** `S0_run1 = 9/23` (`22,705` tok, `673.85s`, `48.08/hr`), `S0_run2 = 9/23` (`22,705` tok, `672.83s`, `48.15/hr`) — **`0` task flips (`100.0%` per-task agreement, `0` token variance)**.
  - **`studio_capability_probes_v1.jsonl` (`48` tasks):** `S0_run1 = 37/48` (`77.08%`), `S0_run2 = 37/48` (`77.08%`) — **`0` task flips (`100.0%` per-task agreement)**:
    - `P1_declined_tool`: `7/8` (`87.5%`, `7,954` tok, `132.5s` — failed `p1_neg_01` due to 900-token tool-schema distraction)
    - `P2_non_answer_guard`: `8/8` (`100.0%`)
    - `P3_exact_math_xcruncher`: **`0/8` (`0.0%`)** (every multi-digit product, factorial `37!`/`43!`, and BBP hex-$\pi$ offset hallucinated digits)
    - `P4_ontology_constraints`: `7/8` (`87.5%`, failed `p4_onto_06` by falsely blaming compliant `NODE-EDGE`)
    - `P5_epistemic_memory`: `8/8` (`100.0%`)
    - `P6_doctest_outcome_cycle`: `7/8` (`87.5%`, failed `p6_swe_08` `FenwickRangeAdd`)
  - **`OpenCode` Agentic Probes:** `Turn-1 prompt_tokens = 805`, `Turn-1 wall = 13.97s`, `Turn-2 wall = 1.09s` (`LCP` cached).
  - **Measured Noise Floor:** **`0` flips across all `71` tasks** under greedy `HG4-16R` decoding.

### Phase S1 (`DONE — DROP` — `semantics.py` / `CSI` Declined-Tool & Non-Answer Guard)
- **Artifact:** `benchmarks/riemann-lean4/results/studio_phase_S1_semantics_csi.json`
- **Measured Result vs `S0` (`71` tasks):**
  - **`studio_capability_probes_v1.jsonl`:** `37/48` (`77.08%`, **`+0` net flips** — `P1_declined_tool` stayed `7/8` with `+1` on `p1_neg_01` offset by `-1` on `p1_neg_06`, though `P1` tokens dropped `-50.1%` from `7,954 -> 3,968`; `P2_non_answer_guard` stayed `8/8`).
  - **`rh_bench_suite_v2.jsonl`:** `9/23` (`39.13%`, **`+0` net flips**, tokens `22,705 -> 29,290` = **`+29.0%`**, wall-clock `673.85s -> 836.82s` = **`+24.2%`**, throughput `48.08 -> 38.72 passes/hr`).
  - **`OpenCode` Turn-1 Prompt Tokens:** `805` (`0%` mutation).
- **Gate Verdict:** **`DROP` from Default-On (`CLOVER_STUDIO_S1_SEMANTICS=0`)** — `TokenEconomizer` + `Clover-35B-A3B`'s Hermes parser already solve `87.5%` of declined-tool and `100%` of promise-lead-in tasks; unclosed-fence continuations on `length`-capped Lean/SWE drafts increased RH token/wall cost by `+29.0% / +24.2%` with `0` net accuracy flips across all 71 tasks. Not carried into `Phase S2`.

### Phase S2 (`DONE — DROP` — `clover_cycle.py` Outcome Self-Verifier: `S2-ctrl`, `S2a`, `S2b`)
- **`S2-ctrl` (`blind` compute-matched re-check control — `DROP`):**
  - Artifact: `benchmarks/riemann-lean4/results/studio_phase_S2_ctrl_blind.json`
  - Result: `37/48` Studio probes (`+0` flips) and `9/23` RH (`+0` net flips: `+1` on `rh_t3_egm_01` cancelled by `-1` on `rh_t3_egm_02`), while RH tokens surged **`22,705 -> 52,974` (`+133.3%`)**, RH wall-clock surged **`673.85s -> 1,116.52s` (`+65.7%`)**, and `OpenCode Turn-2` degraded `1.09s -> 10.09s`. Confirms `Opus 5.5`'s prediction that blind self-reflection without objective verifier diagnostics burns `>2×` compute for zero net gain.
- **`S2a` (`fast` `<50 ms` AST + prompt `>>>` doctests + Lean surface check — `DROP`):**
  - Artifact: `benchmarks/riemann-lean4/results/studio_phase_S2a_fast_outcome.json`
  - Result: Preserved `OpenCode Turn-1 = 805` tokens and `Turn-2 = 1.09s`, and kept RH overhead low (`673.85s -> 733.37s = +8.8%`), **but** achieved `37/48` Studio probes (`+0` flips; `p6_swe_08` failed private range-query invariant even after doctest check) and `9/23` RH (`+0` flips). Under strict `AGENTS.md` discipline, `0` net flips across 71 tasks = **`DROP` from Default-On**.
- **`S2b` (`lean_elaborate` opt-in `verifier-in-the-loop` Lean compiler elaboration — `DROP` from Default-On, available `OPT-IN`):**
  - Artifact: `benchmarks/riemann-lean4/results/studio_phase_S2b_lean_elaborate.json`
  - Result: `37/48` Studio probes (`+0` flips) and `9/23` RH (`+0` flips, `26,174` tok, `758.95s = +12.6%` wall-clock — standalone Tier-1 Lean files already passed `2/2` on Turn 1, while Tier-2 leaf tactic snippets rely on project-context elaboration in the outer `OpenCode` loop). Kept default-off (`CLOVER_STUDIO_S2_MODE=off`) and not carried into `Phase S3`.

### Phase S3 (`DONE — KEEP` — `xcruncher_bridge.py` Native C11 Exact Math & BBP Hex-$\pi$ Verifier)
- **Artifact:** `benchmarks/riemann-lean4/results/studio_phase_S3_xcruncher_math.json`
- **Measured Result vs `S0` (`71` tasks):**
  - **`studio_capability_probes_v1.jsonl`:** **`37/48` (`77.08%`) -> `44/48` (`91.67%`)** — **`+7` net verified flips** (`P3_exact_math_xcruncher`: **`0/8` (`0.0%`) -> `7/8` (`87.5%`)**, `+87.5pp` gain, `3.5×` above the `+25.0pp` gate), with **`0` regressions** across `P1, P2, P4, P5, P6`.
  - **`rh_bench_suite_v2.jsonl`:** **`9/23`** (`22,705` tok vs `22,705` tok = **`0` extra tokens**, `674.16s` vs `673.85s` = **`+0.05%` wall-clock**, `48.06 passes/hr`).
  - **`OpenCode` Agentic Probes:** `Turn-1 prompt_tokens = 805` (`0%` delta), `Turn-2 wall = 1.09s` (`LCP` cached).
- **Gate Verdict:** **`KEEP` (Default-On Winner #1 — `CLOVER_STUDIO_S3_XCRUNCHER=1`)** — Clears `Path B` decisively (`+87.5pp` targeted capability gain, `0` RH regressions, `+0.05%` RH wall-clock, `0%` Turn-1 token delta). Carried forward into `Phase S4` (`--s3 --s4`).

### Phase S4 (`DONE — DROP` — `symbolic-ontology-graph.py` Deductive Closure & `CloverSymbolicVerifier`)
- **Artifact:** `benchmarks/riemann-lean4/results/studio_phase_S4_symbolic_ontology.json`
- **Measured Result vs `S3` (`71` tasks):**
  - **`studio_capability_probes_v1.jsonl`:** `44/48` (`91.67%`, **`+0` net flips** over `S3`; `P4_ontology_constraints` stayed `7/8` as `Clover-35B-A3B` natively solved `87.5%` of 3-hop ontology closures, while `P3` retained `7/8`).
  - **`rh_bench_suite_v2.jsonl`:** `9/23` (`22,705` tok, `673.20s`, `48.13 passes/hr`, **`+0` flips**).
  - **`OpenCode` Agentic Probes:** `Turn-1 prompt_tokens = 805` (`0%` delta), `Turn-2 wall = 1.09s`.
- **Gate Verdict:** **`DROP` from Default-On (`CLOVER_STUDIO_S4_ONTOLOGY=0`, available opt-in)** — `0` net flips beyond `S3`. Not carried into `Phase S5`.

### Phase S5 (`DONE — DROP` — `memory-knowledge-graph.py` Session-Scoped Epistemic Memory Graph)
- **Artifact:** `benchmarks/riemann-lean4/results/studio_phase_S5_epistemic_memory.json`
- **Measured Result vs `S3` (`71` tasks):**
  - **`studio_capability_probes_v1.jsonl`:** `44/48` (`91.67%`, **`+0` net flips** over `S3`; `P5_epistemic_memory` was already `8/8` (`100%`) natively on `Clover-35B-A3B`, and tail-only ledger injection caused `0` regressions while adding ~`115` tokens/probe on `P5`).
  - **`rh_bench_suite_v2.jsonl`:** `9/23` (`22,705` tok, `672.16s`, `48.20 passes/hr`, **`+0` flips**).
  - **`OpenCode` Agentic Probes:** `Turn-1 prompt_tokens = 805` (`0%` delta), `Turn-2 wall = 1.03s` (`LCP` cached).
- **Gate Verdict:** **`DROP` from Default-On (`CLOVER_STUDIO_S5_MEMORY=0`, available opt-in)** — `0` net flips beyond `S3`. Not carried into `Phase S6`.

### Phase S6 (`DONE — DROP` — `jepa_engine.py` / `clover_rpre` Outcome-Residual Adapter)
- **Artifact:** `benchmarks/riemann-lean4/results/studio_phase_S6_jepa_rpre.json`
- **Measured Result vs `S3` (`71` tasks):**
  - **`studio_capability_probes_v1.jsonl`:** `44/48` (`91.67%`, **`+0` net flips** over `S3`, `0` regressions; `BaselinePrototype` cosine gate `>= 0.55` held `0` out-of-envelope disruptions).
  - **`rh_bench_suite_v2.jsonl`:** `9/23` (`22,705` tok, `672.07s`, `48.21 passes/hr`, **`+0` flips**).
  - **`OpenCode` Agentic Probes:** `Turn-1 prompt_tokens = 805` (`0%` delta), `Turn-2 wall = 1.04s` (`LCP` cached).
- **Gate Verdict:** **`DROP` from Default-On (`CLOVER_STUDIO_S6_JEPA=0`, available opt-in)** — `0` net flips beyond `S3`.

### Phase S7 (`DONE` — Consolidation, Drop-One Attribution, Live `OpenCode` CLI & Final Audit)
- **Artifacts:** `benchmarks/riemann-lean4/results/{studio_phase_S7_shipped_default.json, studio_phase_S7_drop_s3.json, studio_phase_S7_opencode_live.json}`
- **Shipped Default Configuration (`fusion_router.py`):**
  - **`CLOVER_STUDIO_S3_XCRUNCHER = "1"` (`KEEP` — Default-ON):** Pre+post native C11 `x-cruncher` exact integer multiplication, factorial, and BBP hex-$\pi$ verification (`title/summary meta-calls excluded`).
  - **`S1, S2, S4, S5, S6` (`DROP` from Default-ON / Available `OPT-IN`):** `CLOVER_STUDIO_S1_SEMANTICS=0`, `CLOVER_STUDIO_S2_MODE=off`, `CLOVER_STUDIO_S4_ONTOLOGY=0`, `CLOVER_STUDIO_S5_MEMORY=0`, `CLOVER_STUDIO_S6_JEPA=0`.
  - **`TokenEconomizer` `REA` MCP Session-Start Gate (`token_economizer.py`):** Added `_extract_first_user_text` + `_RE_INTENT_RE` so heavy `rea_*` / reverse-engineering MCP schemas attached by `opencode.json` are retained on binary/ELF/disassembly prompts and pruned on non-RE sessions (preventing client system-prompt `"binary"` boilerplate false-positives).
- **Measured Final `S7` Results (Out-of-the-Box Zero-Flag Default vs Drop-One `--no-s3` on `studio_capability_probes_v2.jsonl`):**
  - **`S7_shipped_default` (`0` CLI flags):** **`45/48` (`93.75%`)** Studio Probes (**`P3 = 8/8 = 100.0%`**, **`2,227` `P3` tokens, `47.4s` `P3` wall-clock — `-69.2%` tokens & `3.35×` faster than post-only repair!**) + **`9/23`** RH+SWE+RE (`22,705` tok, `673.09s`, `48.14 passes/hr`, `OpenCode Turn-1 = 805` tokens, `Turn-2 = 1.06s`). *(Note: `S3`–`S6` ran on `v1` where `p3_math_08` carried a faulty Windows `xcruncher.dll` label `244DF45CFB` instead of true BBP `CAD181156B`; both `S3` and `S7` produced the true `CAD181156B` value. Fixed `k_vec = k_lim` (`hex_offset % 3 == 2`) in `_xcruncher_c/src/xcruncher.c`, rebuilt `fused/xcruncher.dll`, added 4-vector startup KAT `0/194/1000/5000`, and sealed `studio_capability_probes_v2.jsonl`.)*
  - **`S7_drop_s3` (`--no-s3` Drop-One on exact shipped commit):** Dropped immediately from **`45/48` (`93.75%`) -> `37/48` (`77.08%`, `-8` flips on `P3`: `8/8 -> 0/8`)**, proving **100% causal attribution** to `studio_xcruncher_verifier.py`.
  - **Live `OpenCode v1.19.6` CLI End-to-End & Post-`S3` LCP Verification (`studio_phase_S7_opencode_live.json`):** `rc = 0` in **`94.20s`**, called `read("Demo.lean")`, returned exact `demo_add_zero` **and** exact 44-digit `37! = 13763753091226345046315979581580902400000000`, and achieved **`97.23%` (`913 / 939` tokens, `304.31 ms` prefill) Turn-2 LCP prefix cache hit EVEN WHEN `37!` fired `S3` on Turn 1** (`clearing the >= 95% invariant`).

---

## 5. Master Summary Table (`Phases S0 → S7` Across All `71` Sealed Tasks)

| Phase | Subsystem Tested | Studio Probes (`/48`) | `P3` Exact Math (`/8`) | RH+SWE+RE (`/23`) | RH Tokens | RH Wall (`s`) | OpenCode Turn-1 Tok | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`S0_run1`** (`v1`) | Baseline Anchor (`Run 1`) | `37/48` (`77.08%`) | `0/8` (`0.0%`) | `9/23` | `22,705` | `673.85s` | `805` | **ANCHOR** |
| **`S0_run2`** (`v1`) | Reproducibility (`Run 2`) | `37/48` (`77.08%`) | `0/8` (`0.0%`) | `9/23` | `22,705` | `672.83s` | `805` | **`0` Noise Floor** |
| **`S1`** (`v1`) | `semantics.py` / `CSI` Guard | `37/48` (`77.08%`) | `0/8` (`0.0%`) | `9/23` | `29,290` (`+29.0%`) | `836.82s` (`+24.2%`) | `805` | **`DROP`** (Opt-In) |
| **`S2-ctrl`** (`v1`) | Blind Re-Check Control | `37/48` (`77.08%`) | `0/8` (`0.0%`) | `9/23` | `52,974` (`+133.3%`) | `1,116.52s` (`+65.7%`) | `805` | **`DROP`** |
| **`S2a`** (`v1`) | Fast `<50ms` AST/Doctest/Lean | `37/48` (`77.08%`) | `0/8` (`0.0%`) | `9/23` | `24,967` (`+10.0%`) | `733.37s` (`+8.8%`) | `805` | **`DROP`** (Opt-In) |
| **`S2b`** (`v1`) | Lean `lake env lean` Elaborate | `37/48` (`77.08%`) | `0/8` (`0.0%`) | `9/23` | `26,174` (`+15.3%`) | `758.95s` (`+12.6%`) | `805` | **`OPT-IN`** |
| **`S3`** (`v1` / true `v2`) | **`x-cruncher` Native C11 Math** | **`44/48` (`45/48` `v2`)** | **`7/8` (`8/8` `v2`)** | **`9/23`** | **`22,705` (`0.0%`)** | **`674.16s` (`+0.05%`)** | **`805`** | **`KEEP` (Default-ON)** |
| **`S4`** (`+S3`, `v1`) | Symbolic Ontology Closure | `44/48` (`45/48` `v2`) | `7/8` (`8/8` `v2`) | `9/23` | `22,705` (`0.0%`) | `673.20s` | `805` | **`DROP`** (Opt-In) |
| **`S5`** (`+S3`, `v1`) | Session Epistemic Memory | `44/48` (`45/48` `v2`) | `7/8` (`8/8` `v2`) | `9/23` | `22,705` (`0.0%`) | `672.16s` | `805` | **`DROP`** (Opt-In) |
| **`S6`** (`+S3`, `v1`) | `JEPA` / `clover_rpre` Adapter | `44/48` (`45/48` `v2`) | `7/8` (`8/8` `v2`) | `9/23` | `22,705` (`0.0%`) | `672.07s` | `805` | **`DROP`** (Opt-In) |
| **`S7_shipped`** (`v2`) | **Shipped Out-of-Box Default** | **`45/48` (`93.75%`)** | **`8/8` (`100.0%`)** | **`9/23`** | **`22,705` (`0.0%`)** | **`673.09s` (`-0.11%`)** | **`805`** | **SHIPPED DEFAULT** |
| **`S7_drop_s3`** (`v2`) | Drop-One (`--no-s3`) | `37/48` (`77.08%`) | `0/8` (`0.0%`) | — | — | — | — | **`-8` Flips Verified** |
