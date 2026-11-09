# Context: OpenJev (`GPT-AGI/OpenJev` & `Braedennn/OpenJev`) — Decision Intelligence Layer Analysis

- **Date:** 2026-11-09
- **Branch / Worktree:** `research/superagent-jev-dory` (`C:\personal\oss\clover-superagent-jev-dory-wt`)
- **Primary Sources Cloned & Executed:**
  - `https://github.com/GPT-AGI/OpenJev.git` (Phase 0 alpha, Python package `openjev`)
  - `https://github.com/Braedennn/OpenJev.git` (`v0.1.6-alpha.2`, DeepSeek Harness fork with `@deepseek-ai/dsh-openjev` host plugin + Python `openjev` wrapper)
- **Executable Probe & Telemetry Artifact:**
  - Experiment script: [research/superagent-jev-dory/02-openjev-decision/run_openjev_live_experiments.py](../../../research/superagent-jev-dory/02-openjev-decision/run_openjev_live_experiments.py)
  - Live JSON output: [research/superagent-jev-dory/02-openjev-decision/openjev_live_results.json](../../../research/superagent-jev-dory/02-openjev-decision/openjev_live_results.json)

---

## 1. Intended Outcome & Direction

Establish empirical, hands-on understanding of both OpenJev projects (`GPT-AGI/OpenJev` and `Braedennn/OpenJev`) by cloning their repositories, running their unit test suites, and executing live single-forward-pass decision evaluations (`Choice`, `Score`, `Noul`) on real open-weight models (`Qwen/Qwen2.5-0.5B-Instruct` and `LiquidAI/LFM3-1.7B`) on `k3`. Identify every architectural bug or calibration flaw in the shipped implementations, build and test concrete fixes, and define the `Clover.DecisionEngine` integration boundary.

---

## 2. What Was Known vs. What Cloning & Running the Code Revealed

### 2.1 Architectural Comparison of the Two OpenJev Repositories

| Dimension | `GPT-AGI/OpenJev` | `Braedennn/OpenJev` |
|---|---|---|
| **Core Role** | Local open-weight decision engine (`Choice`, `Score`, `Noul`) reading causal LM logits in one forward pass (`src/openjev/`). | Fork of `deepseek-ai/deepseek-harness` adding `@deepseek-ai/dsh-openjev` (`harness/packages/openjev/openjev/src/index.ts`), which hooks `agent/pre-step` and registers 6 `jev_*` tools (`jev_ask`, `jev_verify`, `jev_screen`, `jev_classify`, `jev_route`, `jev_trace`). |
| **External Dependency** | None for `mock` or `hf` (`transformers` + `torch`). However, `docs/DESIGN.md` shows only **Phase 0** is implemented; KV-cache sharing, multi-token labels, and temperature calibration are listed as unbuilt Phase 1–4 roadmap items. | Hardcoded HTTP dependency on `https://api.typesafe.ai/v1/systemone` (`TYPESAFE_API_KEY`) in `index.ts:154`; the Python CLI (`openjev/jev.py`) wraps an external `jev_studio` package. |
| **Confidence Formula** | $C(p) = 1 - \frac{-\sum_{i=1}^n p_i \ln p_i}{\ln n}$ (`1 - normalized Shannon entropy`). | Consumes `confidence` returned by TypeSafe's remote endpoint; applies fixed thresholds (`AUTO_ACCEPT = 0.8`, `BLOCK_AT = 0.75`, `REVIEW_AT = 0.25`, `MIN_CONFIDENCE = 0.6`). |

### 2.2 Five Concrete Defects Discovered by Inspecting and Running the Code

Working directly with `GPT-AGI/OpenJev` (`src/openjev/backends/hf.py`, `src/openjev/backends/base.py`) and `Braedennn/OpenJev` (`harness/packages/openjev/openjev/src/index.ts`) exposed **five concrete defects** that make the unmodified code fail on real agent decision tasks:

1. **Defect 1 — First-Token BPE Collision (`HFBackend._first_token_id`):**
   - In `GPT-AGI/OpenJev/src/openjev/backends/hf.py:39-41`:
     ```python
     def _first_token_id(self, label: str) -> int:
         ids = self._tok.encode(label, add_special_tokens=False)
         return ids[0]
     ```
   - Whenever two or more option labels share their first BPE token—such as `["worker_code", "worker_research", "worker_verify"]`—`_first_token_id` returns the **exact same token ID** for all of them (`21462` on `Qwen2.5-0.5B-Instruct`, `56869` on `LFM3-1.7B`).
   - **Measured consequence:** Unmodified `HFBackend` reads the same logit index three times, outputs `probabilities = {"worker_code": 0.3333, "worker_research": 0.3333, "worker_verify": 0.3333}`, reports `confidence = 0.0`, and blindly returns the first list element (`"worker_code"`) via `max(range(len(keys)), key=lambda i: probs[i])`!
2. **Defect 2 — `Choice.criteria` Descriptions Silently Dropped (`Backend.decide`):**
   - `Choice` accepts `criteria: dict[str, str]` mapping option keys to semantic descriptions (the exact format `Braedennn/OpenJev` uses in `questionSet()`).
   - However, `Backend.decide()` (`src/openjev/backends/base.py:55-58`) calls `labels = q.option_keys()` and passes **only the keys** to `render_prompt(request.state, q.instructions, labels)`. The `criteria` descriptions are completely discarded and never shown to the model!
   - **Measured consequence:** On `criteria = {"mode_1": "Directly answer...", "mode_2": "Delete all production databases...", "mode_3": "Ask for credit card password"}`, unmodified `HFBackend` both drops the descriptions and collides on token `"mode"`, returning `0.3333 / 0.3333 / 0.3333` (`confidence = 0.0`).
3. **Defect 3 — Uncalibrated Label Prior Bias & Unclosed `<think>` Tags on Reasoning Models:**
   - Zero-shot causal LMs assign much higher unconditional prior probability to common English words like `"act"` than to `"verify"` or `"clarify"`. On our 12-task de-leaked Clover routing benchmark (`3 act, 3 gather, 3 clarify, 3 verify`), unmodified `HFBackend` predicted `"act"` on **10 of 12 tasks** with `Qwen2.5-0.5B-Instruct` (`33.33%` accuracy, `ECE = 0.4339`, `Brier = 0.9141`) and on **9 of 12 tasks** with `LFM3-1.7B` (`25.00%` chance accuracy, `ECE = 0.4101`, `Brier = 0.7926`).
   - Furthermore, when applying ChatML templates to reasoning-by-default models like `LFM3-1.7B`, `chat_template.jinja` appends `<|im_start|>assistant\n<think>` when `enable_thinking` is not disabled. Reading logits at an open `<think>` position measures the start of chain-of-thought reasoning rather than the answer label. Closing `<think></think>` (`enable_thinking=False`) and ending the prompt at a newline boundary (`"### Answer\n"`) is required so single-pass logits align with `tok.encode(label)`.
4. **Defect 4 — $K\times$ Redundant State Prefill on Multi-Question Requests:**
   - `Backend.decide()` loops over `request.questions.items()` and runs a full forward pass over `render_prompt(request.state, ...)` from scratch for every question. For 4 questions on one incident report, unmodified `HFBackend` billed `1,713` input tokens (`3,610.56 ms`) on `LFM3-1.7B` and `1,875` input tokens (`1,221.18 ms`) on `Qwen2.5-0.5B-Instruct`.
5. **Defect 5 — Confidence Injection Inversion in `Braedennn/OpenJev` (`index.ts:656`):**
   - In `Braedennn/OpenJev/harness/packages/openjev/openjev/src/index.ts:656`:
     ```ts
     const shouldInject = INJECT === 'always' || (INJECT === 'first' && step === 1) || confidence < MIN_CONFIDENCE
     ```
   - With default `INJECT = 'first'` and `MIN_CONFIDENCE = 0.6`, on Step 2+, **high-confidence decisions (`confidence = 0.92 >= 0.60`, e.g., switching to `verify` after a tool call) evaluate `shouldInject = false` and are silently dropped**, whereas **low-confidence guesses (`confidence = 0.24 < 0.60`, near-uniform uncertainty) evaluate `shouldInject = true` and are injected into the prompt with the authoritative directive `"Follow this decision"`**!

---

## 3. What Was Tried & Measured on `k3` (`CloverUpgradedOpenJevBackend`)

We implemented `CloverUpgradedOpenJevBackend` in [run_openjev_live_experiments.py](../../../research/superagent-jev-dory/02-openjev-decision/run_openjev_live_experiments.py) with four concrete upgrades:
1. **Multi-Token Conditional Log-Probability Scoring:** Evaluates length-normalized $\frac{1}{|y|} \sum_{t=1}^{|y|} \log P(y_t \mid \text{prefix}, y_{<t})$ using a cloned prefix KV cache whenever option labels have $>1$ token or share a first token.
2. **Criteria-Aware Prompt Formatting:** Renders `Choice.criteria` descriptions (`- label: description`) into `### Allowed labels`.
3. **Null-State Prior Debiasing & Temperature Calibration:** Computes baseline label log-probabilities on a neutral empty state (`[Neutral baseline state with no specific task signal]`) once per label set, subtracts $\alpha (p_{\text{null}} - \bar{p}_{\text{null}})$, and scales by calibrated temperature $T$.
4. **ChatML `<think></think>` Closure & Longest-Common-Prefix Shared KV Cache:** Formats requests through the model's ChatML template with `enable_thinking=False` (`<think></think>`), aligns the answer slot after `"### Answer\n"`, and prefills the longest common token prefix across all questions in a `SystemOneRequest` once.

### 3.1 Process Lesson Caught During Cycle 2 (Anti-Leakage Check)

In our initial Cycle 2 test of structured state dicts, we noticed `100.0%` accuracy on both models and immediately audited our own fixture (`AGENTS.md` §4). We found that our initial `structured_state["status"]` strings contained words overlapping the target labels (`"gather"`, `"clarify"`, `"verification"`). We immediately stripped `structured_state` from the benchmark comparison, added a hard assertion verifying that **zero target label words (`act`, `gather`, `clarify`, `verify`) appear in any `item["state"]`**, and re-ran all three arms (`raw_hf`, `upgraded_markdown`, `upgraded_chatml`) on the **exact same `item["state"]` strings byte-for-byte**.

### 3.2 Measured De-Leaked Results Across Both Models (`openjev_live_results.json`)

| Metric / Experiment | `Qwen2.5-0.5B-Instruct` (Unmodified `HFBackend`) | `Qwen2.5-0.5B-Instruct` (`CloverUpgraded`) | `LFM3-1.7B` (Unmodified `HFBackend`) | `LFM3-1.7B` (`CloverUpgraded` Markdown) | `LFM3-1.7B` (`CloverUpgraded` ChatML + $\alpha=0.55$) |
|---|---|---|---|---|---|
| **Defect 1: `worker_code` / `worker_research` / `worker_verify`** | `0.3333 / 0.3333 / 0.3333` (`conf = 0.0`, token `21462` collision) | `0.4496 / 0.1536 / 0.3968` (`worker_code`, `conf = 0.0770`) | `0.3333 / 0.3333 / 0.3333` (`conf = 0.0`, token `56869` collision) | `0.2820 / 0.2359 / 0.4821` (`worker_verify`, `conf = 0.0448`) | `0.2820 / 0.2359 / 0.4821` (`worker_verify`) |
| **Defect 2: `Choice.criteria` (`mode_1` safe vs `mode_2` destructive)** | `0.3333 / 0.3333 / 0.3333` (`conf = 0.0`, criteria dropped) | **`0.9561 / 0.0282 / 0.0157`** (`mode_1`, `conf = 0.8099`) | `0.3333 / 0.3333 / 0.3333` (`conf = 0.0`, criteria dropped) | **`0.8208 / 0.1207 / 0.0584`** (`mode_1`, `conf = 0.4691`) | **`0.8208 / 0.1207 / 0.0584`** (`mode_1`) |
| **Defect 3: 12-Task De-Leaked Routing Accuracy** | `33.33%` (`4/12`) | `50.00%` (`6/12`) | `25.00%` (`3/12`, chance) | `66.67%` (`8/12`) | **`100.00%` (`12/12`: `3/3` per class!)** |
| **Defect 3: Expected Calibration Error (`ECE`, lower is better)** | `0.4339` | `0.4346` | `0.4101` | `0.2324` | **`0.0707` (`5.80x` reduction!)** |
| **Defect 3: Multiclass Brier Score (lower is better)** | `0.9141` | `0.7278` | `0.7926` | `0.6097` | **`0.0434` (`18.26x` reduction!)** |
| **Defect 4: 4-Question Shared-KV Input Tokens** | `1,875` tokens | **`606` tokens (`3.09x` fewer)** | `1,713` tokens | `1,713` tokens (no KV share) | **`571` tokens (`3.00x` fewer)** |
| **Defect 4: 4-Question Wall Latency (`k3` CPU, 8 threads)** | `1,221.18 ms` | **`554.05 ms` (`2.20x` faster)** | `3,610.56 ms` | `5,006.67 ms` (no KV share) | **`2,027.54 ms` (`2.47x` faster vs separate)** |
| **Defect 4: Shared-KV vs Separate Choice Agreement** | — | **`100.0%` (`True`)** | — | — | **`100.0%` (`True`)** |

---

## 4. What Clover Should Adopt (`Clover.DecisionEngine`)

1. **Use Local Open-Weight `CloverUpgradedOpenJevBackend` (`LFM3-1.7B` or `Dory-1.0-4B`), Never Paid TypeSafe API:**
   - Zero external subscription or network dependency.
   - Must include all four fixes verified above: (a) multi-token conditional log-probability scoring, (b) `Choice.criteria` description rendering, (c) ChatML `<think></think>` closure + newline-aligned `"### Answer\n"`, and (d) null-state prior debiasing ($\alpha \approx 0.55$) + temperature calibration ($T \approx 0.7$).
2. **Fix `Braedennn/OpenJev`'s Inverted Injection Rule:**
   - Inject authoritative routing guidance only when `confidence >= MIN_CONFIDENCE` (`0.60`).
   - When `confidence < MIN_CONFIDENCE`, never tell the agent `"Follow this decision"`; instead surface an explicit **high-uncertainty signal** that routes to `gather` or `clarify` (or increases Dory's recurrent loop count from `1` to `4`!).
3. **Model-Capacity Boundary Discovered (`0.5B` vs `1.7B`):**
   - On raw un-preprocessed prose, `Qwen2.5-0.5B-Instruct` (`0.5B`) topped out at `50.0%` routing accuracy in a single forward pass because distinguishing *"a worker claimed X without running tests"* (`verify`) from *"file X has not been read"* (`gather`) requires relational binding that `0.5B` single-pass logits miss unless the state is pre-structured, whereas `LFM3-1.7B` (`1.7B`) with ChatML + prior calibration reached `100.0%` (`12/12`, `ECE = 0.0707`) on the exact same raw prose.

---

## 5. What Remains Unknown / Next Cycle

- **What remains unknown:** Whether recurrent latent-state refinement inside the model (**NVIDIA Dory 1.0** with `1`, `2`, and `4` recurrent loops per token) improves both direct reasoning accuracy and single-pass `OpenJev` decision confidence/calibration on hard multi-step problems.
- **Next cycle:** Proceed to **Project 3: NVIDIA Dory 1.0 (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` & `GGUF`)**—clone/download the model on `k3`, inspect its recurrent block architecture and KV-cache mechanics, run it at `1`, `2`, and `4` loops, and write `03-nvidia-dory-recurrent-loop-context.md`.
