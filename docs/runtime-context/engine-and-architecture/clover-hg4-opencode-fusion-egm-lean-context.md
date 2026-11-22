# Context — Pure-`C1` `HG4-16R-OP` Engine + `OpenCode` + `Copilot Fusion` + `Elephant–Goldfish (EGM)` Verified on the Certified Lean 4 `RiemannHypothesis` Archive

- **Cycle Date:** `2026-11-22`
- **Human Direction Owner:** Santhosh Narayanan (`@Santhosh-N`)
- **Co-Architect & Independent Verifier:** `Claude Opus 5.5` (split-mode architecture review, implementation, and verification)
- **Governing Specification:** [AGENTS.md](../../../AGENTS.md) (`Context → Direction → Execution → Outcome → Growth`; explicit separation of **known**, **inferred**, and **unknown**)

---

## 1. Context (What Reality Shows Today)

### 1.1 Known (Measured on `k3` — `AMD Ryzen 9 7950X3D`, 16C/32T, `AVX-512 VNNI+VBMI`, `124 GB` DDR5, `/dev/md2` `305 GiB` free, `/mnt/nvme1` `955 GiB` free)
1. **Full-Layer Canonical `HG4-16R` C11 Engines Are Live Across All Four Open Models (`0` GPU, `memcmp == 0` across Scalar / AVX2 / AVX-512):**
   - **`LFM3-Omni-2B`** (`/opt/clover-k3/lfm3-omni-hg4`, `24/24` layers, `853.88 MiB`): Held-out `PPL = 29.0585` (`+0.27` vs `BF16` `28.79`), **`1,347.0 tok/s` prefill**, **`50.3 tok/s` decode**.
   - **`Clover-35B-A3B`** (`/opt/clover-k3/clover35b-full-hg4`, `40/40` layers × `256` experts = `10,240` experts, `17.41 GiB`): Held-out `PPL = 17.9194`, **`89.8 tok/s` prefill**, **`14.4–16.6 tok/s` warm decode**.
   - **`Nemotron 3.5 Super VL` (`120B-A12B`)** (`/opt/clover-k3/nemotron-full-hg4`, `88/88` backbone + `2/2` MTP = `90` layers, `20,992` experts, `60.20 GiB`): Held-out `PPL = 14.8423`, **`7.57–9.20 tok/s` (`2.42×–3.58×` speedup)** via lossless stateful `2-Layer MTP` speculative decoding (`83.3%–100.0%` acceptance, `exact_greedy_match = True`).
   - **`Kimi K3` / `Kimi K3.1 Pro` (`2.8T` / `104B` active):** `Kimi-K3.1-Pro` `93/93` `HG4-16R` trunks + `5/5` `DSpark` layers (`347.8 draft tok/s`) at `/opt/clover-k3/kimi-k31-hg4` (`34.19 GiB`), alongside full `92 × 896` `Kimi-K3` (`K3.0`) `experts.direct` (`1.41 TB`) served by `clover-k3-chat.service` (`port 8765`).
2. **Why `HG4-16R` Precomputed Operators (`HG4-16R-OP`) Were Only Partially Completed:**
   - Current `.hg4` files already precompute per-matrix `pre_diag`, `FP16` Lloyd–Max scales, `-exp(A_log)`, `RoPE inv_freq`, and shared `pre_diag_up` across MoE experts, **but** still keep trunk sub-projections (`Q`, `K`, `V`, `Gate`, `zxbc`, `dt`) separate with unrotated `FP32` MoE routers (`gate_w`) and unfolded `RMSNorm` gains.
   - Re-packing existing `4-bit` `.hg4` tiles into `4-bit` (`4-bit → float → 4-bit`) would double-quantize weights; therefore `HG4-16R-OP` (`operator.hg4` + exact `FP32` Hadamard-rotated `gate_w = W_gate · D⁻¹ · H₁₂₈` + `constants.bin`/`maps.json`) **must** be built from original `BF16` trunk tensors (`only ~0.85 GB Omni + ~0.7 GB Clover-35B + ~3.2 GB Nemotron`; expert files remain untouched).
3. **Why Clover AI Studio's Research Harness & Post-Hoc `C2/C3` Were Retired from the Primary Path:**
   - Post-hoc `omni_c2_c3.py` (`HybridLink + 11-level VQ-VAE + 12L NCM`) ran *after* `C1` had already generated the complete answer text, relying on a `-45·(t-p)²` + `+65.0` diagonal copy crutch over cross-family `SVD` embeddings (gated off by default in commit `0d771df4`).
   - Clover AI Studio's custom web UI and custom agent loop remain research-grade compared to mature open-source coding harnesses.
4. **Primary-Source Verification of the Four External Pillars (`Checked 2026-11-22`):**
   - **`OpenCode`** (`anomalyco/opencode`): Open-source terminal/IDE coding agent supporting custom OpenAI-compatible `/v1/chat/completions` providers, streaming tool calls, LSP, and MCP servers.
   - **Google `Elephant–Goldfish Model (EGM)`** (`https://github.com/google/elephant-goldfish-model`, Apache-2.0): Spec-first multi-agent discipline separating the context-rich **Elephant** (architect/spec writer, writes zero code directly) from stateless, zero-context **Goldfish** reviewer subagents, shipping official `adapters/opencode` and `com.github.copilot/agents`.
   - **GitHub `Copilot Fusion` Pattern** (`Single`, `Cascade`, `Critique` across `Standard / High / Max` effort tiers): Runtime multi-model orchestration matching frontier quality at ~50% token cost by escalating from fast models to heavy models only when needed.
   - **Levent Alpöge + Claude Lean 4 `RiemannHypothesis` Archive (`2026-11-20` release, `2026-11-21` Lean/Mathlib community certification):**
     - Hosted at `https://alpo.ge/lean.tar.gz.part01` .. `part14` (`239,275,554` bytes across `175,187` `.lean` files, `~17.37M` LoC, `~819,000` lemmas/theorems; `0` `sorry`/`admit`, `0` `axiom`, `0` `skipKernelTC`).
     - Reproducible community verification & lock-free scheduler (`sched.py`) published at `https://github.com/luosiallen/riemann-hypothesis-lean-verify`, pinned to **`leanprover/lean4:v4.33.1`** and **Mathlib commit `0df444a360eaa60ab8c11dca51a86af692955474`**.

### 1.2 Inferred (Joint Analysis with `Claude Opus 5.5`)
- **Expected `HG4-16R-OP` Speedup:** `+5–12%` decode on dense `LFM3-Omni-2B`, `+2–5%` decode on MoE `Clover-35B` / `Nemotron 3.5` (`where ~85% of DDR5 traffic is routed experts`), and `+15–25%` on prefill.
- **CPU Multi-Agent Latency Constraint:** Concurrent subagents on one 16-core CPU contend for DDR5 bandwidth. Requests must be serialized behind a prefix-caching queue with strict token caps on Goldfish reviews (`≤ 400` tokens, max 2 personas × 1 round) and **objective oracle escalation** (`Lean 4` kernel / test failure, never self-rated confidence).
- **Goldfish Non-Veto Rule:** A smaller `35B` Goldfish can **flag** candidate flaws (such as unprovable/false subgoal signatures) for the `120B` Elephant or Lean kernel to test, but may **never veto** a kernel-verified proof or passing test suite.

### 1.3 Unknown (Must Be Observed Against Reality in Phases 0–5)
- Exact single-file incremental elaboration latency and RSS memory footprint of `leanprover/lean4:v4.33.1` on `/mnt/nvme1` alongside `mmap`'d `HG4-16R` models on `k3`.
- Baseline pass@1 / pass@k rates of `LFM3-Omni-2B`, `Clover-35B-A3B`, and `Nemotron 3.5 120B` across Tiers 1–4 of the Lean 4 RH archive before and after `Cascade`, `EGM`, and Clover MCP tools (`S2` premise retrieval, `x-cruncher`, `JEPA`).

---

## 2. Direction & Governing Boundaries

1. **Decouple the Inference Engine from the Agent Harness:**
   - Expose all four `HG4-16R` models behind a rock-solid **OpenAI-compatible `/v1` + `Copilot Fusion` (`Single / Cascade / Critique`) Gateway** (`clover-ai-system/src/tools/hg4-fusion-gateway/`) with SSE streaming, strict tool-call JSON validation/repair, and longest-prefix KV/Mamba/BD-GDN state reuse.
   - Drive coding and formal-verification workflows through **`OpenCode`**, **VS Code Copilot custom endpoints**, and **`Elephant–Goldfish (EGM)`** (`adapters/opencode` & `com.github.copilot/agents`) rather than Clover AI Studio's research loop.
2. **Complete `HG4-16R-OP` Precomputation Cleanly (`Zero Double-Quantization`):**
   - Build `operator.hg4` fused trunk projections, exact `FP32` Hadamard-rotated MoE routers (`W_gate · D⁻¹ · H₁₂₈`), folded `RMSNorm` gains, and `constants.bin` / `maps.json` expert metadata strictly from original `BF16` trunk weights, gated on `≥ 99.9%` Top-K routing parity and non-degraded held-out `PPL`.
3. **Validate Against Uncontaminated Formal Reality (`Lean 4 RiemannHypothesis` + Companion SWE Suite):**
   - Stage all RH archive & `.olean` artifacts strictly on `/mnt/nvme1/clover-rh-bench/` (`955 GiB` free), enforce per-process RAM/timeout caps on `lean`, and enforce **strict DAG anti-leakage** (any premise search must exclude the target lemma, sibling proofs in the same file, and all downstream dependents).
4. **Earn-Back Rule for Clover Studio Features:**
   - Port Clover capabilities (`S2` lemma/ontology retrieval, `x-cruncher` exact math, `JEPA` tactic-outcome plasticity) **only as toggleable MCP tools** (`default OFF`), keeping only those that show a measured improvement at equal wall-clock/token budget.

---

## 3. Phased Execution Architecture & Go / No-Go Gates

| Phase | Scope | Split-Mode Ownership | Go / No-Go Verification Gate |
|---|---|---|---|
| **Phase 0** | Stage `alpo.ge` 14-part RH archive (`SHA-256` + static `0 sorry/axiom` check), `luosiallen/riemann-hypothesis-lean-verify`, `google/elephant-goldfish-model`, `lean v4.33.1`, and `OpenCode` on `k3:/mnt/nvme1/clover-rh-bench/` | Worker A (`k3` env) + Opus 5.5 review | Archive SHA-256 & static audit pass; `lean v4.33.1` elaborates sample module; `OpenCode` + `EGM` installed |
| **Phase 1a** | Build `clover-ai-system/src/tools/hg4-fusion-gateway/` (`/v1/models`, `/v1/chat/completions` SSE + strict `tool_calls` JSON + prefix cache + `Single/Cascade/Critique` + `Elephant/Goldfish` endpoints + `opencode.json`) | Worker B (`Claude Opus 5.5`) + Parent integration | Unit & live `OpenCode` multi-turn tool-call loop passes with `0` malformed tool calls & warm-prefix speedup |
| **Phase 1b** | Build `BF16`-sourced `HG4-16R-OP` precomputer (`operator.hg4` + exact `FP32` Hadamard-rotated MoE routers + `constants.bin`/`maps.json`) and verify on `k3` | Parent Agent + Opus 5.5 audit | `0` double-quantization; `≥ 99.9%` Top-K router parity; `Scalar == AVX2 == AVX-512` `memcmp == 0`; `PPL` preserved |
| **Phase 2** | Build `benchmarks/riemann-lean4/` (4-Tier task extractor: Tier 1 Spec Round-Trip, Tier 2 Masked Leaf Lemmas, Tier 3 EGM Subgoal Audit, Tier 4 Verifier SWE + toggleable `mcp_clover_tools.py`) | Parent Agent + Worker B (`Claude Opus 5.5`) | Anti-leakage DAG check passes; sandboxed `< 5s` incremental `lean` oracle verifies ground-truth vs mutated proofs |
| **Phase 3–5** | Run controlled single-model baselines → `Cascade` → `+EGM` → `+Clover MCP` (`S2` / `x-cruncher`) ablations on `k3` and record measured pass-rate vs wall-clock | Parent Agent + `Claude Opus 5.5` independent audit | Every kept mode beats simpler baseline on verified pass-rate under equal time/token budget |

---

## 4. Cycle Log (Outcome & Growth — Strict Sequential Execution Under `AGENTS.md`)

- **Cycle 0 (`2026-11-22`):** Established primary-source Context across `k3`, `alpo.ge`, `luosiallen/riemann-hypothesis-lean-verify`, `google/elephant-goldfish-model`, and `OpenCode`; completed 2-round adversarial design review with `Claude Opus 5.5`; recorded structured todos in SQLite and [TODO.md](../../../TODO.md) Section 8.
- **Phase A1 (`DONE` — Hardening All `Claude Opus 5.5` Audit Issues across Rounds 1 & 2):**
  - **Execution:** Hardened [lean_oracle.py](../../../benchmarks/riemann-lean4/lean_oracle.py) (multi-line wrapped `#print axioms <thm_name>` subset check `{propext, Classical.choice, Quot.sound}`, `'...'` char-literal state in comment/string stripper, banned `decide +native`/`run_tac`/`#eval`/`elab`/`macro`/`local`/top-level `end`/`namespace`, temp `.lean` sandbox outside `proj`, nested `have h : P := by` preservation), [fusion_router.py](../../../clover-ai-system/src/tools/hg4-fusion-gateway/fusion_router.py) (`subgoal_verdict` preservation + server-side default-on `TokenEconomizer`, `OutputBudgeter`, and `<available_skills>` index), [token_economizer.py](../../../clover-ai-system/src/tools/hg4-fusion-gateway/token_economizer.py) (`tool_call_id -> function.name` resolution when `role:"tool"` omits `name`, consecutive-only failure counting, and prefix-cache-preserving tool-list invariance), [run_ablation.py](../../../benchmarks/riemann-lean4/run_ablation.py) (isolated `python3 -I` module-import + `_SANDBOX_PASS_SENTINEL` driver blocking `sys.exit(0)` bypasses), [mcp_clover_tools.py](../../../benchmarks/riemann-lean4/mcp_clover_tools.py) (`|`/`where` equation-compiler leak barrier + bounded `x-cruncher` exact math), [gateway.py](../../../clover-ai-system/src/tools/hg4-fusion-gateway/gateway.py) (`30s` socket write timeout), [backends.py](../../../clover-ai-system/src/tools/hg4-fusion-gateway/backends.py) (`spec_max_new` clamp), [precompute_hg4_operators.py](../../../clover-ai-system/src/models/clover-35b-a3b/precompute_hg4_operators.py) (`moe_off + 12` header offset -> verified `min_top8_jaccard = 1.000000`), and suite-blind general domain `SKILL.md` playbooks under [clover-ai-system/skills/](../../../clover-ai-system/skills/).
  - **Verification:** `27/27` gateway tests + `6/6` harness regression tests passed under `python -W error` on both Windows and `k3`.
- **Phase A2 (`DONE` — Freezing Immutable Balanced 5-Tier `rh_bench_suite_v2.jsonl` & Scorer Boundary):**
  - **Discovery (Growth):**
    1. Setting `RLIMIT_AS = 12 GiB` caused `lean` to fail with `libc++abi: failed to create thread` because `--tstack=400000` (`400 MB` virtual stack per thread) × `32` threads on `k3` reserves `12.8 GB` of virtual address space (`VIRT`) even when physical `RSS < 1.5 GB`. Switching to Lean 4's native `--memory=8192 --threads=4 --tstack=400000` resolved thread-stack reservation cleanly while enforcing an `8 GiB` Lean heap cap.
    2. An initial `v1` suite + skill draft was rejected by `Claude Opus 5.5` review because Tier 3 had `1` provable / `3` unprovable tasks and the initial `SKILL.md` drafts contained task-specific idioms. We balanced Tier 3 to **`2` provable + `2` unprovable**, rewrote all 3 `SKILL.md` playbooks to be strictly suite-blind general domain conventions, removed task-specific `mcp_hint` logic, and hashed the exact imported scorer/runner modules in every JSON report.
  - **Frozen Validation Boundary (`chmod 444`, verified identically across all 10 Phase B → I JSON runs):**
    - **`23` kernel/test-verified tasks across all 5 tiers** ([rh_bench_suite_v2.jsonl](../../../benchmarks/riemann-lean4/rh_bench_suite_v2.jsonl)): Tier 1 (`2` Spec Round-Trip), Tier 2 (`9` RH Leaf Proofs: `3` short `1–6L`, `3` medium `7–14L`, `3` hard `15–28L`, all ground-truth verified with `#print axioms`), Tier 3 (`4` Balanced Subgoal Audits: `2` provable + `2` mutated-hypothesis unprovable), Tier 4 (`4` Systems/Verifier SWE), Tier 5 (`4` Reverse-Engineering targets compiled at `/mnt/nvme1/clover-rh-bench/re_targets/{ipc_app, fsm_app, libmystery_mix.so, libtile_unpack.so}`).
    - `suite_sha256`: `415c9148ad75ba7338bd365a8305baf17258e3a2fd957381ff487f7f071a1d26`
    - `oracle_sha256` (`lean_oracle.__file__`): `52d731367acdb28593a3d86633e2e762736ae24cd6ef7d42e780499efa58c937`
    - `ablation_sha256` (`run_ablation.__file__`): `6d5ef9d6e4d061f2e263fdaf510cdf22c2e8c8347c21e7d8d830291e8446a6b1`
    - `runner_sha256` (`run_sequential_phase.__file__`): `78fd110a1be269b11dab82f4e3f36283cff5267ee47bf22cf6e8ecab1032c089`
  - **Pre-Registered Keep / Drop Gate (`Phases B → I`):** Keep a phase's capability into the cumulative stack iff **`net_flips = (+flips - -flips) >= +1` with `0` regressions** (or `>= +2` net flips), **OR** **`net_flips == 0` with `>= 15%` reduction in Turn-1 prompt tokens or wall-clock time**. Otherwise record the neutral/unfavorable result as Growth (`AGENTS.md` Sections 5–6) and leave that mechanism **OFF** in subsequent phases.
- **Phase B (`DONE` — Clean Unmodified Baselines on Frozen v2 Suite, [phase_b_baseline.json](../../../benchmarks/riemann-lean4/results/phase_b_baseline.json)):**
  - **Measured Suite Baseline (`clover-35b-a3b`, all capability flags `OFF`):** **`6 / 23` (`26.09%`) verified passes**, `11,802` total tokens, `516.19s` wall-clock (`41.85 passes/hr`):
    - **Tier 1 (`Spec Round-Trip`):** **`2 / 2` (`100.0%`)** (`addVal_prod_alt`, `goal1` kernel `Iff` equivalence).
    - **Tier 2 (`RH Leaf Proofs`):** **`0 / 9` (`0.0%`)** — cold single-shot `35B` hallucinated nonexistent Mathlib4 constants or emitted Lean 3 tactics (`refine'`).
    - **Tier 3 (`Balanced Subgoal Audits`, `2` provable / `2` unprovable):** **`2 / 4` (`50.0%`, `1` false rejection on `rh_t3_egm_01_valid_dirichlet_nonvanish` and `1` missed defect on `rh_t3_egm_04_landau_missing_nonneg`)**.
    - **Tier 4 (`Verifier SWE`):** **`2 / 4` (`50.0%`)** (passed LCP prefix-cache planner & MXFP4 block decoder).
    - **Tier 5 (`Reverse Engineering`):** **`0 / 4` (`0.0%`)** (no disassembly/JS recon evidence).
  - **Measured Agentic Probes:**
    - `OpenCode` 11-tool Turn-1 prompt: **`1,307` tokens (`22.61s` Turn-1 wall-clock, `1.18s` Turn-2 cached LCP reuse = `19.2×` Turn-2 speedup)**.
    - `LFM3-Omni-2B` repeated `webfetch 404` probe (with `role:"tool"` omitting `name`): `1,444` prompt tokens, `omni2b_switched_to_read = False`.
- **Phase C (`DONE` — `+TokenEconomizer` Native Guardrails, [phase_c_tokeneconomizer.json](../../../benchmarks/riemann-lean4/results/phase_c_tokeneconomizer.json) -> `KEEP`):**
  - **Measured Delta vs Phase B:**
    - **`OpenCode` Turn-1 prompt tokens:** **`1,307 -> 710` (`-45.68%`)**; **Turn-1 wall-clock:** **`22.61s -> 13.06s` (`-42.24%`, `1.73×` faster Turn-1 prefill)**.
    - **`OpenCode` Turn-2 cached LCP wall-clock:** **`1.18s -> 1.04s`** (`answered_correctly = True` — confirming session-start tool-profile freezing and suffix-only loop redirect preserve 100% of the C11 LCP prefix cache).
    - **`LFM3-Omni-2B` loop probe:** prompt tokens **`1,444 -> 896` (`-37.95%`)** (`escaped_webfetch_loop` remained `True` as in Phase B; consecutive `webfetch` 404 loop detection via `tool_call_id` resolution is verified in `TestGatewayHTTP.test_economizer_skills_and_output_budget`).
    - **Frozen 23-task suite:** `6 / 23` (`0` regressions, bit-identical `11,802` suite tokens).
  - **Gate Decision:** **`KEEP`** (`-45.68%` Turn-1 token reduction >= `15%` threshold with `0` regressions). Kept into cumulative stack.
- **Phase D (`DONE` — `+Clover MCP` Upfront Lexical `S2` Full-Mathlib4 Premise Search + `x-cruncher`, [phase_d_clover_mcp.json](../../../benchmarks/riemann-lean4/results/phase_d_clover_mcp.json) -> `DROP`):**
  - **Measured Delta vs Kept Stack (`Phase C`):**
    - **Frozen 23-task suite:** `6 / 23 -> 6 / 23` (`net_flips = 0`, `+0 / -0`).
    - **Cost & Throughput Impact:** Total suite tokens rose **`11,802 -> 18,486` (`+56.63%`)**, wall-clock rose **`517.84s -> 627.14s` (`+21.11%`)**, and `passes/hr` fell **`41.71 -> 34.44`**.
  - **Growth Diagnosis (`AGENTS.md` Sections 5–6):** Bag-of-words lexical overlap over generic declaration binders (`CommRing`, `Finset`, `Ideal`) across ~8,690 Mathlib4 files retrieves noisy neighbor signatures that dilute single-pass `35B` context without providing Lean 4 proof tactics.
  - **Gate Decision:** **`DROP` from cumulative default stack** (`mcp_clover_tools.py` remains available strictly as an opt-in stdio MCP tool, `enabled: false` by default in `opencode.json`).
- **Phase E (`DONE` — `+Anthropic SKILL.md` Suite-Blind Domain Playbooks + `<available_skills>` Index, [phase_e_anthropic_skills.json](../../../benchmarks/riemann-lean4/results/phase_e_anthropic_skills.json) -> `KEEP`):**
  - **Measured Delta vs Kept Stack (`Phase C`):**
    - **Frozen 23-task suite:** **`6 / 23 -> 7 / 23` (`30.43%`, `+1` net verified flip, `0` regressions)**:
      - Flipped `rh_t3_egm_04_landau_missing_nonneg` (`False -> True`), bringing the balanced **Tier 3 Subgoal Auditing suite from `2 / 4 (50%)` to `3 / 4 (75%)`**.
    - **Measured Prompt & Suite Token Cost:** Gateway `<available_skills>` index added `+95` tokens on `OpenCode` Turn-1 probe (`710 -> 805` tokens, `13.92s` Turn-1, `1.05s` Turn-2 cached LCP — still **`-38.41%` below Phase B's `1,307` tokens**); activated domain playbooks brought suite tokens `11,802 -> 20,559` (`707.62s`, `35.61 passes/hr`).
  - **Gate Decision:** **`KEEP`** (`+1` net verified flip with `0` regressions). Kept into cumulative stack.
- **Phase F (`DONE` — `+Copilot Fusion` 1-Shot Objective-Oracle `Cascade` `35B -> 120B`, [phase_f_copilot_fusion_cascade.json](../../../benchmarks/riemann-lean4/results/phase_f_copilot_fusion_cascade.json) -> `DROP`):**
  - **Measured Delta vs Kept Stack (`Phase E`):**
    - **Frozen 23-task suite:** `7 / 23 -> 7 / 23` (`net_flips = 0`, `+0 / -0`, `15` objective-oracle escalations to `Nemotron-3.5-Super-VL 120B`).
    - **Cost & Throughput Impact:** Total tokens rose **`20,559 -> 47,883` (`+132.91%`)**, wall-clock rose **`707.62s -> 3,052.21s` (`4.31×` slower)**, and verified throughput fell **`35.61 -> 8.26 passes/hr`**.
  - **Growth Diagnosis (`AGENTS.md` Sections 5–6):** Single-step regeneration on `120B` CPU after a Lean/Python error without multi-turn tactic-state iteration spends `4.31×` wall-clock without closing multi-step Lean obligations.
  - **Gate Decision:** **`DROP` from cumulative default CPU stack** (`clover-fusion-high` remains an opt-in virtual model).
- **Phase G (`DONE` — `+Elephant–Goldfish (EGM)` Unconditional Always-On, [phase_g_elephant_goldfish.json](../../../benchmarks/riemann-lean4/results/phase_g_elephant_goldfish.json) -> `DROP` from unconditional default):**
  - **Measured Delta vs Kept Stack (`Phase E`):**
    - **Frozen 23-task suite:** `7 / 23 -> 7 / 23` (`net_flips = 0`: **`+1` positive SWE flip** on `rh_t4_swe_01_dag_ready_scheduler` bringing Tier 4 SWE to `3 / 4 = 75%`, offset by **`-1` false rejection** on `rh_t3_egm_03_valid_landau_nonneg` where `goldfish_false_rejections = 1` and `elephant_false_rejections = 2` both rejected a valid statement).
    - **Cost & Throughput Impact:** Total tokens rose **`20,559 -> 41,630` (`+102.49%`)**, wall-clock rose **`707.62s -> 1,705.67s` (`2.41×`)**, and `passes/hr` moved `35.61 -> 14.77`.
  - **Growth Diagnosis (`AGENTS.md` Sections 5–6):** Upfront `Elephant` edge-case specification genuinely unlocks multi-constraint SWE tasks (`rh_t4_swe_01`), whereas unconditional adversarial auditing (`Goldfish` + `Elephant` re-check) increases false rejections on valid formal statements (`1 -> 2` false rejections).
  - **Gate Decision:** **`DROP` unconditional always-on EGM from cumulative default stack** (keep `elephant` and `goldfish` strictly as opt-in agents in `opencode.json`).
- **Phase H (`DONE` — `+REA v6.10.1` (`morluto/rea`) Reverse-Engineering MCP + Evidence Budgeter, [phase_h_rea_mcp.json](../../../benchmarks/riemann-lean4/results/phase_h_rea_mcp.json) -> `KEEP`):**
  - **Measured Delta vs Kept Stack (`Phase E`):**
    - **Frozen 23-task suite:** **`7 / 23 -> 9 / 23` (`39.13%`, `+2` net verified flips, `0` regressions — `+50.0%` relative improvement over Phase B's `6 / 23`)**:
      - Flipped **both** `rh_t5_re_01_minified_ipc_preload` (`False -> True`) **and** `rh_t5_re_04_protocol_state_machine` (`False -> True`) via `rea analyze-javascript-application` + static bundle recovery + suite-blind `reverse-engineering-rea` `SKILL.md` (`Tier 5 RE: 0 / 4 -> 2 / 4 = 50.0%`).
      - **`0` regressions and bit-identical token counts** across all 19 non-RE tasks (Tiers 1–4).
    - **Throughput:** Verified throughput jumped to **`47.09 passes/hr`** (`688.10s` suite wall-clock) — **exceeding even raw Phase B's `41.85 passes/hr`**!
  - **Gate Decision:** **`KEEP`** (`+2` net verified flips with `0` regressions). Enabled by default in `configs/opencode.json` (`"rea": {"enabled": true}`) and gateway (`CLOVER_ENABLE_OUTPUT_BUDGET=1`).
- **Phase I (`DONE` — Shipped Default-On Consolidation, Full 23-Task Drop-One Runs & Paired Live `OpenCode v1.19.6` CLI Telemetry):**
  - **Full 23-Task Drop-One Orthogonality Verification ([phase_i_summary.json](../../../benchmarks/riemann-lean4/results/phase_i_summary.json)):**
    - **Consolidated Stack (`TokenEconomizer + SKILL.md + REA`, [phase_h_rea_mcp.json](../../../benchmarks/riemann-lean4/results/phase_h_rea_mcp.json)):** **`9 / 23` (`39.13%`), Turn-1 probe `805` tok (`13.90s`), `688.10s` (`47.09 passes/hr`)**.
    - **Drop `REA` (full 23-task run, [phase_i_drop_rea.json](../../../benchmarks/riemann-lean4/results/phase_i_drop_rea.json)):** drops **`9 / 23 -> 7 / 23`** (loses both `rh_t5_re_01_minified_ipc_preload` and `rh_t5_re_04_protocol_state_machine`).
    - **Drop `SKILL.md` (full 23-task run, [phase_i_drop_skills.json](../../../benchmarks/riemann-lean4/results/phase_i_drop_skills.json)):** drops **`9 / 23 -> 8 / 23`** (loses `rh_t3_egm_04_landau_missing_nonneg`, while `REA` still independently solves both `rh_t5_re_01` and `rh_t5_re_04`).
    - **Drop `TokenEconomizer` (full 23-task run, [phase_i_drop_tokeneconomizer.json](../../../benchmarks/riemann-lean4/results/phase_i_drop_tokeneconomizer.json)):** `9 / 23`, **Turn-1 `OpenCode` probe prompt jumps `805 -> 1,402` tokens (`+74.16%` bloat, `13.90s -> 24.03s` Turn-1 latency)**.
  - **Paired Live `OpenCode v1.19.6` CLI End-to-End Verification ([live_opencode_cli_verify.json](../../../benchmarks/riemann-lean4/results/live_opencode_cli_verify.json), `opencode run -m clover/clover-35b-a3b` with `REA` MCP configured):**
    - **Raw (`Economizer=0, SkillsIndex=0`):** Turn-1 prompt = **`7,711` tokens** (`15,472` 2-turn prompt tokens, `7,761` prefilled, `179.24s` prefill, **`185.80s` total wall-clock**).
    - **Shipped Default-On (`Economizer=1, OutputBudget=1, SkillsIndex=1`):** Turn-1 prompt = **`3,651` tokens (`-52.65%` real `OpenCode` prompt-token reduction)**; Turn-2 cached **`3,651 / 3,701` tokens (`98.65%` Turn-2 cache hit; only `50` tool-output tokens prefilled)**; total wall-clock = **`75.79s` (`-59.21%`, `2.45×` faster end-to-end)** with `0` tool-call repairs/retries (`"The theorem is named demo_add_zero."`).

---

## 5. Master Sequential Summary Table (`Phases B → I` on Frozen Balanced 23-Task Suite)

| Phase | Added Mechanism Tested | Verified Score | Net Flips vs Prev Kept | `OpenCode` Turn-1 Probe (`tok` / `s`) | Suite Wall-Clock / `passes/hr` | Pre-Registered `AGENTS.md` Gate |
|---|---|---:|---:|---:|---:|---|
| **Phase B** | Clean Unmodified `Clover-35B-A3B` Baseline | `6 / 23` (`26.1%`) | — | `1,307` tok / `22.61s` | `516.2s` / `41.85/hr` | **`BASELINE`** |
| **Phase C** | **`+TokenEconomizer`** (Session-Start Tool Profiles + `0 ms` Consecutive Loop Guard) | `6 / 23` (`26.1%`) | `0` (`+0 / -0`) | **`710` tok (`-45.7%`) / `13.06s`** | `517.8s` / `41.71/hr` | **`KEEP`** (`>= 15%` token cut, `0` reg.) |
| **Phase D** | `+Clover MCP` Upfront Lexical `S2` Mathlib4 + `x-cruncher` | `6 / 23` (`26.1%`) | `0` (`+0 / -0`) | `710` tok / `12.78s` | `627.1s` / `34.44/hr` | **`DROP`** (keep opt-in MCP only) |
| **Phase E** | **`+Anthropic SKILL.md`** (Suite-Blind Playbooks + `<available_skills>` Index) | **`7 / 23` (`30.4%`)** | **`+1` (`+1 / -0`)** | `805` tok / `13.92s` | `707.6s` / `35.61/hr` | **`KEEP`** (`+1` flip, `0` reg.) |
| **Phase F** | `+Copilot Fusion` 1-Shot `Cascade` (`35B -> 120B` on Error) | `7 / 23` (`30.4%`) | `0` (`+0 / -0`) | `805` tok / `14.09s` | `3,052.2s` / `8.26/hr` | **`DROP`** (`4.31×` slower on CPU) |
| **Phase G** | `+Elephant–Goldfish (EGM)` Unconditional Always-On | `7 / 23` (`30.4%`) | `0` (`+1 / -1`) | `805` tok / `14.21s` | `1,705.7s` / `14.77/hr` | **`DROP`** (`+1` SWE flip vs `-1` FR; keep opt-in) |
| **Phase H** | **`+REA v6.10.1`** (`morluto/rea` Reverse-Engineering MCP + Budgeter) | **`9 / 23` (`39.1%`)** | **`+2` (`+2 / -0`)** | `805` tok / `13.90s` | `688.1s` / **`47.09/hr`** | **`KEEP`** (`+50%` solves & higher `passes/hr` vs B) |
| **Phase I** | **Shipped Default-On (`TokenEconomizer + SKILL.md + REA`)** | **`9 / 23` (`39.1%`)** | **`+3` vs Phase B (`0` reg.)** | **Live `OpenCode` CLI: `7,711 -> 3,651` tok (`185.8s -> 75.8s`, `2.45×`)** | Full 23-task Drop-One (`7/23`, `8/23`, `1,402` tok) | **`PRODUCTION DEFAULT`** |









