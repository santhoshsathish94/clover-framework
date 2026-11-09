# Context: Unified Clover Integration (`Clover.SuperAgent` + `Clover.DecisionEngine` + `Clover.ModelRuntime`)

- **Date:** 2026-11-09
- **Branch / Worktree:** `research/superagent-jev-dory` (`C:\personal\oss\clover-superagent-jev-dory-wt`)
- **Predecessor Context Records:**
  1. [01-deepseek-superagent-harness-context.md](01-deepseek-superagent-harness-context.md) — `deepseek-ai/deepseek-harness` (`v0.2.2-rc.1`, commit `fcc0ad46`)
  2. [02-openjev-decision-engine-context.md](02-openjev-decision-engine-context.md) — `GPT-AGI/OpenJev` & `Braedennn/OpenJev`
  3. [03-nvidia-dory-recurrent-loop-context.md](03-nvidia-dory-recurrent-loop-context.md) — `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`
- **Unified Implementation & Verification Artifacts:**
  - Production module: [clover-ai-system/studio/clover/superagent_jev_dory.py](../../../clover-ai-system/studio/clover/superagent_jev_dory.py)
  - Unit & integration test suite: [clover-ai-system/studio/tests/test_superagent_jev_dory.py](../../../clover-ai-system/studio/tests/test_superagent_jev_dory.py)
  - Live unified ablation runner: [research/superagent-jev-dory/04-clover-unified-integration/run_unified_clover_ablation.py](../../../research/superagent-jev-dory/04-clover-unified-integration/run_unified_clover_ablation.py)
  - Live JSON output: [research/superagent-jev-dory/04-clover-unified-integration/unified_clover_ablation_results.json](../../../research/superagent-jev-dory/04-clover-unified-integration/unified_clover_ablation_results.json)

---

## 1. Intended Outcome & Direction

Synthesize what working hands-on with all three projects taught us into three clean integration boundaries inside Clover (`Clover.SuperAgent`, `Clover.DecisionEngine`, and `Clover.ModelRuntime`), keeping the 5-stage Clover cycle (**Context → Direction → Execution → Outcome → Growth**) as the governing architecture. Measure each capability independently and combined, including the key adaptive test where `Clover.DecisionEngine` (Jev) dynamically chooses `Clover.ModelRuntime`'s (Dory 1.0) recurrent loop count (`1`, `2`, or `4` loops).

---

## 2. How the Three Capabilities Map into the Governing Clover Cycle

```mermaid
flowchart TD
    C["Context (System Reality)\nFiles · Logs · Memory Graph · Chain of Recall (form='recall')"] --> D["Direction (Human Authority)\nIntended Outcome · Boundaries · Write Scopes"]
    D --> J["Clover.DecisionEngine (Calibrated OpenJev)\nChoice · Score · Noul · Multi-Token Logprobs · Null-Prior Debiasing\nActions: act | gather | clarify | verify | delegate | defer"]
    J -- "Selects Loop Budget (1, 2, or 4 loops)\nbased on Confidence & Task Type" --> M["Clover.ModelRuntime (NVIDIA Dory 1.0 / LFM3 / Kimi K3)\n16 Input + 40 Recurrent (x1..4 loops, last_loop KV) + 16 Output"]
    J -- "Routes & Coordinates Tasks" --> S["Clover.SuperAgent (Persistent Coordinator & Workers)\ncreated -> routed -> running -> waiting -> verifying -> completed / failed"]
    M --> S
    S --> O["Outcome & Independent Watchdog / Verifier\nRejects Worker Self-Certification · Recovers Stalled Tasks · Sandbox Checks"]
    O -- "Unfavorable Outcome / Low Confidence" --> C
    O -- "Verified Outcome" --> G["Growth\nAppend-Only Event Log · Calibrated Priors · Context Handoff"]
    G --> C
```

### Summary of Corrections Made Over the Upstream Projects

| Integration Boundary | Upstream Project Inspected | Defects Found by Working with the Code | How Clover's Implementation (`superagent_jev_dory.py`) Fixes It |
|---|---|---|---|
| **`Clover.SuperAgent`** | `deepseek-ai/deepseek-harness` (`fcc0ad46`) | 1. Task statuses only `pending \| in_progress \| completed \| deleted` (no `routed`, `waiting`, `verifying`, or `failed`).<br>2. Task board is passive (`taskBoardAutoWakesWorkers = false`).<br>3. `task-view.ts` attaches false-positive `writeScopeWarnings` to already `completed` tasks.<br>4. `dsh-tool-ralph` trusts worker self-certification (`status: 'complete'`) with zero independent verification.<br>5. `kick()` swallows unsourced message errors via `agent/error`. | 1. 7-state lifecycle: `created -> routed -> running -> waiting -> verifying -> completed \| failed`.<br>2. Active dependency-ready triggers auto-route unblocked tasks and notify assigned workers.<br>3. `write_scope_warnings` checks only active (`routed \| running \| waiting \| verifying`) tasks.<br>4. Direct `running -> completed` transition is forbidden (`CLOVER_VERIFICATION_REQUIRED`); requires `verify_and_complete_task()` with an independent verifier.<br>5. Preserves `form='relay'` vs `form='notice'` vs `form='recall'` attribution and JSONL crash recovery. |
| **`Clover.DecisionEngine`** | `GPT-AGI/OpenJev` & `Braedennn/OpenJev` | 1. `HFBackend._first_token_id` collides whenever options share a first BPE token (`worker_code` / `worker_verify`, `"12"` / `"14"`).<br>2. `Backend.decide()` silently drops `Choice.criteria` descriptions from the prompt.<br>3. Raw zero-shot logits have severe `"act"` prior bias (`25%` chance accuracy on `LFM3-1.7B`) and open `<think>` tag mismatch.<br>4. $K\times$ redundant state prefill on multi-question requests.<br>5. `Braedennn/OpenJev` (`index.ts:656`) drops high-confidence ($\ge 0.6$) decisions on Step 2+ while injecting low-confidence ($< 0.6$) guesses as authoritative commands. | 1. Multi-token conditional log-probability scoring over full option token sequences.<br>2. Renders `Choice.criteria` descriptions in `### Allowed labels`.<br>3. Closes `<think></think>` + newline-aligns `"### Answer\n"` + subtracts null-state prior ($\alpha = 0.55, T = 0.7$), lifting `LFM3-1.7B` de-leaked routing accuracy from **`25.0% -> 100.0%`** (`ECE: 0.4101 -> 0.0707`).<br>4. Shares longest-common-prefix KV cache (`3.0x` token reduction, `2.47x` speedup).<br>5. Injects authoritative directives only when `confidence >= min_confidence`; low confidence escalates Dory loops or routes to `gather`/`clarify`. |
| **`Clover.ModelRuntime`** | `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` (`vllm#60149`) | 1. `AutoTokenizer.from_pretrained` fails on `"mlp"` in `layer_types` unless `validate_layer_type` is overridden or `PreTrainedTokenizerFast` is used.<br>2. Default `per_loop` KV cache multiplies recurrent KV memory by $L\times$.<br>3. Fixed `4 loops` runs 192 layers even on trivial tasks where Loop 1 already has `>0.96` confidence. | 1. Standalone PyTorch engine (`StandaloneDoryModel`) + `PreTrainedTokenizerFast` loader.<br>2. Uses `last_loop` KV cache (`0.0` max abs diff vs `per_loop` on single-pass prefill, `4.0x` recurrent KV reduction at 4 loops).<br>3. **Jev-Adaptive Loop Controller** dynamically selects `1`, `2`, or `4` loops from 1-loop Jev confidence and hypersphere chord distance $\Delta_l$. |

---

## 3. Measured Results (`unified_clover_ablation_results.json`)

### 3.1 Experiment 1 — Jev-Adaptive Dory Loop Policy vs. Fixed `1`, `2`, and `4` Loops (`NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` on `k3`)

Evaluated across the 8-task mixed workload (4 easy factual/routing tasks + 4 hard arithmetic/modular/verification/contradiction tasks):

| Loop Policy | Task Accuracy | Mean Gold Option Probability | Mean Executed Layers / Call | Relative Compute vs. Fixed 4-Loop | Mean Latency (`k3` CPU, 8 threads) | Quality-per-Compute (`Accuracy % / 100 Layers`) |
|---|---:|---:|---:|---:|---:|---:|
| **Fixed `1 Loop`** | `87.50%` (`7/8`, fails `w7_hard_verify_route`) | `0.7833` | `72.0` | `37.50%` | `768.03 ms` | `1.2153` (incomplete accuracy) |
| **Fixed `2 Loops`** | `87.50%` (`7/8`, fails `w7_hard_verify_route`) | `0.8637` | `112.0` | `58.33%` | `1,137.45 ms` | `0.7812` (incomplete accuracy) |
| **Fixed `4 Loops`** | **`100.00%` (`8/8`)** | **`0.9326`** | `192.0` | `100.00%` | `1,919.82 ms` | `0.5208` |
| **`Jev-Adaptive (1 / 2 / 4 Loops)`** | **`100.00%` (`8/8`)** | **`0.9202`** | **`122.0` (`-36.46%` layers saved!)** | **`63.54%`** | **`1,564.26 ms`** | **`0.8197` (`1.57x` vs Fixed 4-Loop!)** |

- **How the Jev-Adaptive controller behaved per task:**
  - On all 4 easy tasks (`w1_easy_symbol`, `w2_easy_capital`, `w3_easy_http`, `w4_easy_route_act`), 1-loop confidence was high (`0.9863`, `0.9606`, `0.5822`, `0.8203`), so the controller stopped at **`1 loop` (`72` layers)** with `4/4` correct.
  - On `w5_hard_modular` (`(17*3+5)/8 = 7`), 1-loop confidence was `0.5014` (`< 0.72`), so the controller escalated to **`2 loops` (`112` layers)**, raising confidence to `0.6870` (`P("7") = 0.8977`).
  - On `w6_hard_arith_prod` (`9*8 = 72`, where 1-loop confidence was `0.3128`), `w7_hard_verify_route` (where 1-loop and 2-loop both predicted wrong `"clarify"` with low confidence `0.4920`), and `w8_hard_contradiction_check`, the controller escalated to **`4 loops` (`192` layers)**—flipping `w7_hard_verify_route` from wrong `"clarify"` to **correct `"verify"` (`P = 0.8317`)**!

### 3.2 Experiment 2 — 5-Way Combined System Ablation

Evaluated across the 6 end-to-end operational, decision, and reasoning scenarios (`S1` easy direct execution, `S2` multi-digit calculation, `S3` false worker self-certification trap, `S4` stalled worker watchdog recovery, `S5` cold process restart mid-DAG, and `S6` hard verification & ambiguous destructive request routing):

| System Configuration | Scenarios Passed | `S1` Easy Direct | `S2` Multi-Digit `72` | `S3` False Worker Self-Cert Caught | `S4` Stalled Worker Recovered | `S5` Process Restart Survived | `S6` Hard Verify & Ambiguity Routing |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline (Unintegrated)** | **`2/6` (`33.3%`)** | PASS | PASS | FAIL | FAIL | FAIL | FAIL |
| **2. `+ Clover.SuperAgent` only** | **`4/6` (`66.7%`)** | PASS | PASS | **PASS** | **PASS** | **PASS** | FAIL (1-loop routing miss) |
| **3. `+ Clover.DecisionEngine` only** | **`3/6` (`50.0%`)** | PASS | PASS | FAIL | FAIL | FAIL | **PASS** |
| **4. `+ Clover.ModelRuntime` (Dory 4-Loop) only** | **`3/6` (`50.0%`)** | PASS | PASS | FAIL | FAIL | FAIL | **PASS** (`192` layers/call) |
| **5. `Combined Clover` (`SuperAgent + Jev + Adaptive Dory`)** | **`6/6` (`100.0%`)** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** (`122` layers/call, `-36.5%` compute) |

---

## 4. What Remains Unknown & Recommended Next Steps

- **What was verified:** All three projects (`deepseek-ai/deepseek-harness`, `GPT-AGI/OpenJev` + `Braedennn/OpenJev`, and `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`) were cloned, set up, executed on real workloads, debugged at the source level, and integrated into `clover-ai-system/studio/clover/superagent_jev_dory.py` with unit tests (`4/4` pass) and live model ablations on `k3`.
- **What was not tested (named gaps):**
  1. `NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` was evaluated on `k3`'s CPU (8 threads, BF16); GPU throughput on Hopper/Blackwell (`NVFP4` / `MXFP8`) and `llama.cpp` Apple Silicon Metal (`Q5_K_M`) were not benchmarked on those respective hardware targets.
  2. Incremental hidden-state caching between Loop 1 and Loops 2–4 inside a single fused C/CUDA kernel (avoiding re-running Input layers `0..15` when escalating from 1 loop to 4 loops) is verified mathematically ($h_{\text{in}}$ is loop-invariant), and is the natural next optimization when wiring `Dory-1.0-4B` into Clover's native C/CUDA engine.
