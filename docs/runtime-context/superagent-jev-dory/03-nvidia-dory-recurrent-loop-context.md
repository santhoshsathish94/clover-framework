# Context: NVIDIA Dory 1.0 (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`) — Recurrent Latent-State Loop Analysis

- **Date:** 2026-11-09
- **Branch / Worktree:** `research/superagent-jev-dory` (`C:\personal\oss\clover-superagent-jev-dory-wt`)
- **Primary Sources Cloned, Downloaded & Executed:**
  - Official BF16 checkpoint: `https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` (`7.5 GB`, `507` tensors, `3,975,229,440` parameters, downloaded to `/opt/clover-k3/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` on `k3`)
  - Official vLLM PR #60149 implementation: `vllm/model_executor/models/dory.py`, `vllm/transformers_utils/configs/dory.py`, `tests/models/test_dory.py` (commit `876edc97`, saved in `C:\personal\oss\external-clones\nvidia-dory\`)
  - Official llama.cpp PR #30306 reference: `https://github.com/brianchmielNV/llama.cpp` (commit `98a74c97`)
- **Executable Probe & Telemetry Artifact:**
  - Standalone PyTorch engine & experiment script: [research/superagent-jev-dory/03-nvidia-dory/run_dory_live_experiments.py](../../../research/superagent-jev-dory/03-nvidia-dory/run_dory_live_experiments.py)
  - Live JSON output: [research/superagent-jev-dory/03-nvidia-dory/dory_live_results.json](../../../research/superagent-jev-dory/03-nvidia-dory/dory_live_results.json)

---

## 1. Intended Outcome & Direction

Establish exact mathematical and empirical understanding of **NVIDIA Nemotron Dory 1.0-4B** (`DoryForCausalLM`) by downloading the official 4B BF16 checkpoint and its reference source code (`vllm#60149`), building a standalone zero-dependency PyTorch runtime (`StandaloneDoryModel`), verifying all 507 tensors and KV-cache invariants, and measuring real inference across `1`, `2`, `3`, and `4` recurrent loops on `k3`.

---

## 2. What Was Known vs. What Cloning & Running the Code Revealed

### 2.1 Exact 507-Tensor Architecture & nGPT Hypersphere Formulation

Secondary summaries describe Dory 1.0 as *"a 4B-parameter model with a shared recurrent transformer block applied 1 to 4 times per token."* Inspecting `config.json`, `vllm_dory.py`, and all 507 tensors in `model.safetensors` revealed the exact mathematical structure:

1. **72 Single-Sublayer Physical Blocks (`16 Input + 40 Recurrent + 16 Output`):**
   - Unlike standard Transformer layers that bundle Attention + MLP into one layer, each of Dory's 72 physical layers (`backbone.layers.0` .. `backbone.layers.71`) is a **single mixer sub-layer**:
     - **36 Even Layers (`0, 2, ..., 70`):** `DoryAttention` (4:1 GQA, `16` Q heads $\times$ `256` head_dim $= 4096$, `4` KV heads $\times$ `256` head_dim $= 1024$). Every 4th attention layer (`layers 6, 14, 22, 30, 38, 46, 54, 62, 70` — 9 total) is **Full Global Attention** (`256K` context, $\theta = 10,000,000$); the other 27 attention layers are **Sliding-Window Attention** (`sliding_window = 1024`, $\theta = 10,000$).
     - **36 Odd Layers (`1, 3, ..., 71`):** `DoryMLP` (SwiGLU `2560 -> 7168 -> 2560`).
   - Thus, the `16` Input sub-layers correspond to 8 Attention + 8 MLP sub-layers (`layers 0..15`), the `40` Recurrent sub-layers correspond to 20 Attention + 20 MLP sub-layers (`layers 16..55`), and the `16` Output sub-layers correspond to 8 Attention + 8 MLP sub-layers (`layers 56..71`).
2. **Zero `RMSNorm` / `LayerNorm` Weights — Pure nGPT on the Unit Hypersphere $\mathbb{S}^{2559}$:**
   - There are **zero normalization weight tensors** in the 507-tensor checkpoint. Instead, every vector is projected onto the unit sphere via `justnorm`:
     $$\text{justnorm}(x) = x \cdot \text{rsqrt}\!\left(\sum_{j} x_j^2 + 10^{-6}\right)$$
   - Every sub-layer updates the residual stream via spherical linear interpolation (`slerp_alpha`) with learned per-channel eigenlearning rates $\alpha \in \mathbb{R}^{2560}$ scaled by $\frac{\alpha_{\text{init}}}{\sigma_{\text{init}}} = \frac{0.05}{0.0177} \approx 2.8249$:
     $$h \leftarrow \text{justnorm}\!\left(\text{justnorm}(h) + \left|\alpha \cdot \frac{\alpha_{\text{init}}}{\sigma_{\text{init}}}\right| \odot \left(\text{justnorm}(h_{\text{mixer}}) - \text{justnorm}(h)\right)\right)$$
   - **Measured effective eigenlearning rates across blocks (`dory_live_results.json`):**
     - `Input (0..15)`: mean $|\alpha_{\text{attn}}| = 0.38805$, mean $|\alpha_{\text{mlp}}| = 0.44675$
     - `Recurrent (16..55)`: mean $|\alpha_{\text{attn}}| = 0.29520$, mean $|\alpha_{\text{mlp}}| = 0.35900$
     - `Output (56..71)`: mean $|\alpha_{\text{attn}}| = 0.25037$, mean $|\alpha_{\text{mlp}}| = 0.32740$
     - Notice that the step size monotonically decreases from Input $\to$ Recurrent $\to$ Output, allowing coarse feature extraction in Input and finer angular refinement in the Recurrent and Output blocks!
3. **Input Reinjection at Every Recurrent Loop (`DoryBackbone.forward`):**
   - Let $h_{\text{in}} = \text{Layers}_{0..15}(E(x))$ and $\text{state}_0 = h_{\text{in}}$.
   - At each recurrent loop $l \in \{0, \dots, L-1\}$, Dory **reinjects $h_{\text{in}}$ on the hypersphere before entering `Layers[16..55]`**:
     $$h^{(l, 0)} = \text{justnorm}\!\left(\text{state}_l + h_{\text{in}}\right), \qquad \text{state}_{l+1} = \text{justnorm}\!\left(\text{Layers}_{16..55}(h^{(l, 0)}, \text{loop\_idx}=l)\right)$$
   - **Why this matters:** Without $h_{\text{in}}$ reinjection, repeated application of a 40-layer block would wash out token identity; reinjecting $\text{justnorm}(\text{state}_l + h_{\text{in}})$ anchors every loop at $\cos(h^{(l,0)}, h_{\text{in}}) \approx 0.706\text{–}0.782 \approx \frac{1}{\sqrt{2}}$ (since $\text{state}_l$ is nearly orthogonal to $h_{\text{in}}$ at $\cos \approx 0.0\text{–}0.22$!).
4. **Effective Parameter Scaling Without Weight Growth:**
   - Physical parameters stored on disk/RAM: **$3,975,229,440$ ($3.98\text{B}$)** (`7.5 GiB` BF16).
   - Shared 40-layer Recurrent block parameters: **$1,835,560,960$ ($1.84\text{B}$)**.
   - Effective active parameters per token:
     - `1 loop` (72 layers): **$3.98\text{B}$**
     - `2 loops` (112 layers): **$5.81\text{B}$**
     - `3 loops` (152 layers): **$7.65\text{B}$**
     - `4 loops` (192 layers): **$9.48\text{B}$**

---

## 3. Concrete Runtime Findings Caught Only by Running the Model

1. **`AutoTokenizer.from_pretrained` Fails Without Custom `validate_layer_type` (`StrictDataclassClassValidationError`):**
   - Running `AutoTokenizer.from_pretrained("/opt/clover-k3/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16")` on stock `transformers` immediately crashes with:
     `ValueError: The layer_types entries must be in ('full_attention', 'sliding_attention', ...) but got ['sliding_attention', 'mlp', ...]`
   - **Cause:** `transformers.PreTrainedConfig.validate_layer_type` does not include `"mlp"` in `ALLOWED_LAYER_TYPES`. Loading via `PreTrainedTokenizerFast(tokenizer_file=...)` or registering `DoryConfig` (which overrides `validate_layer_type` to allow `{"sliding_attention", "full_attention", "mlp"}`) resolves this cleanly.
2. **Exact Mathematical Proof & Measurement: `last_loop` KV Mode is Bit-Identical (`max_abs_diff = 0.0`) to `per_loop` Mode on All Single-Pass Prefills While Cutting Recurrent KV Cache by $4.0\times$:**
   - At 4 loops, `per_loop` mode allocates $8 + 4 \times 20 + 8 = 96$ KV cache slots (`14.94 MB` for our test sequence), whereas `last_loop` mode allocates only $8 + 1 \times 20 + 8 = 36$ KV cache slots (`5.60 MB`, a **$4.0\times$ reduction in recurrent KV slots** and **$2.67\times$ reduction in total KV memory**).
   - During any single-pass prefill (including single-pass `OpenJev` decision evaluation!), Loop $l$ in `last_loop` mode overwrites the full prompt range $0..T-1$ with Loop $l$'s keys and values before attending—so Loop $l$ attends to the exact same keys and values as `per_loop` mode (`single_pass_prefill_4loops_per_vs_last_max_abs_diff = 0.0`)!
   - During multi-token autoregressive decoding, `last_loop` and `per_loop` achieve **`0.999551` logit cosine similarity** and **`100%` top-1 token agreement (`"216" == "216"`)** across `1`, `2`, and `4` loops!
3. **Byte/Single-Digit Tokenization Split in Dory (`"12" -> [1049, 1050]`):**
   - Dory's tokenizer splits every multi-digit integer into single digits (`"12" -> [1049, 1050]`, `"14" -> [1049, 1052]`, `"16" -> [1049, 1054]`, `"18" -> [1049, 1056]`).
   - When we initially used single-first-token logit indexing (`tok.encode(opt)[0]`), all four options `"12"`, `"14"`, `"16"`, `"18"` mapped to token `1049` (`"1"`), yielding `0.2500 / 0.2500 / 0.2500 / 0.2500`! Switching to our multi-token conditional log-probability scorer (`score_dory_options_multitoken`) immediately resolved the collision (`P("14") = 0.3746` vs `P("12") = 0.0876`).

---

## 4. Measured Effect of Recurrent Loops (`1`, `2`, `3`, and `4` Loops) on `k3`

### 4.1 Hypersphere Convergence & Task-Adaptive Latent Movement

Across all tasks, every hidden state remained tightly bound to the unit hypersphere $\mathbb{S}^{2559}$ (`L2 norm = 0.9982 to 1.0013`). Measuring the chord distance $\|\text{state}_l - \text{state}_{l-1}\|_2$ between consecutive loops revealed a striking property:

| Task | Difficulty | Loop 1 $\Delta_1$ ($\|\text{state}_1 - h_{\text{in}}\|_2$) | Loop 2 $\Delta_2$ ($\|\text{state}_2 - \text{state}_1\|_2$) | Loop 3 $\Delta_3$ ($\|\text{state}_3 - \text{state}_2\|_2$) | Loop 4 $\Delta_4$ ($\|\text{state}_4 - \text{state}_3\|_2$) |
|---|---|---:|---:|---:|---:|
| `dory_task_1_easy_fact` (`Au`) | Easy | `1.2498` | `0.0502` | `0.0141` | `0.0078` |
| `dory_task_2_easy_arith` (`9 * 8 = 72`) | Easy | `1.2490` | `0.0762` | `0.0179` | `0.0081` |
| `dory_task_3_hard_modular` (`(17*3+5)/8 = 7`) | Hard | `1.2670` | `0.0694` | `0.0249` | `0.0139` |
| `dory_task_5_hard_sequence` (`x=3 -> 7 -> 14`) | Hard | `1.3075` | `0.0654` | `0.0309` | `0.0149` |
| `dory_task_6_hard_agent_verify` (Unverified worker claim) | Hard Agentic | `1.2574` | **`0.1177`** | **`0.0701`** | **`0.0270`** |

- On easy factual tasks (`Au`), the latent state has nearly converged after Loop 2 ($\Delta_3 = 0.0141, \Delta_4 = 0.0078$).
- On the complex agent verification task (`dory_task_6_hard_agent_verify`), **the latent state moves $2.3\times\text{ to }5.0\times$ farther in Loops 2, 3, and 4** ($\Delta_2 = 0.1177, \Delta_3 = 0.0701, \Delta_4 = 0.0270$)!
- **And look at how that latent movement changes the decision on `dory_task_6_hard_agent_verify`:**
  - `1 loop`: predicts `"clarify"` (`0.6764`) vs `"verify"` (`0.3045`) — **WRONG**
  - `2 loops`: predicts `"clarify"` (`0.6051`) vs `"verify"` (`0.3553`) — **WRONG**
  - `3 loops`: flips to **`"verify"` (`0.6905`)** vs `"clarify"` (`0.2859`) — **CORRECT!**
  - `4 loops`: strengthens **`"verify"` to `0.8317`** (`"clarify"` drops to `0.1460`) — **CORRECT (+52.7 pp gold probability over 1 loop!)**

### 4.2 Summary Across Loop Counts (`1`, `2`, `3`, `4` Loops)

| Loop Count | Executed Layers | Active Params / Token | Mean Gold Option Probability (6 Decision Tasks) | Hard Agentic Verify (`P(verify)`) | CoT Multi-Step Assignment (`x = 3 -> 7 -> 14` within 200 tokens) | Mean Single-Pass Latency (`k3` CPU, 8 threads) |
|---:|---:|---:|---:|---:|---|---:|
| **`1 loop`** | `72` | `3.98B` | `0.6080` | `0.3045` (Wrong: `clarify`) | **Failed** (stalled in ambiguity analysis, never computed `7` or `14`) | `592.58 ms` |
| **`2 loops`** | `112` | `5.81B` | `0.6602` (`+5.2 pp`) | `0.3553` (Wrong: `clarify`) | **Solved (`14`)** (computed `3*2+1=7` and `7*2=14` by token ~95) | `880.36 ms` |
| **`3 loops`** | `152` | `7.65B` | `0.7446` (`+13.7 pp`) | **`0.6905` (Correct: `verify`)** | — | `1,179.55 ms` |
| **`4 loops`** | `192` | `9.48B` | **`0.7555` (`+14.8 pp`)** | **`0.8317` (Correct: `verify`)** | **Solved (`14`)** (cleanest step-by-step `x0=3 -> x1=7 -> x2=14`) | `1,479.14 ms` |

---

## 5. What Clover Should Adopt (`Clover.ModelRuntime`)

1. **Always Use `recurrent_kv_cache_mode = "last_loop"` for Clover:**
   - Provably bit-identical (`max_abs_diff = 0.0`) to `per_loop` for all single-pass `OpenJev` decisions and prefill passes, `0.99955` cosine / `100%` top-1 agreement in autoregressive decoding, and saves **$4.0\times$ recurrent KV-cache memory** at 4 loops.
2. **Adaptive Jev-Controlled Loop Policy (`1 / 2 / 4` Loops) + Chord-Distance Early Exit:**
   - Because Dory's 40-layer recurrent block shares weights, `Clover.DecisionEngine` can dynamically select `num_loops = 1` for high-confidence/simple tasks (`latency = 0.59 s`), `num_loops = 2` for standard reasoning, and `num_loops = 4` when `Decision.confidence < 0.65` or when verifying worker claims (`where 4 loops boosts P(verify) from 0.3045 -> 0.8317`).
   - Even better: because Dory operates on the unit hypersphere $\mathbb{S}^{2559}$, Clover can monitor the **inter-loop chord distance $\Delta_l = \|\text{state}_l - \text{state}_{l-1}\|_2$** during the forward pass and **early-exit the recurrent loop** as soon as $\Delta_l < \tau$ (e.g., $\tau = 0.020$), saving Loops 3–4 automatically on easy tasks (`Au`, where $\Delta_3 = 0.0141 < 0.020$) while running all 4 loops on hard tasks (`dory_task_6`, where $\Delta_3 = 0.0701 > 0.020$)!

---

## 6. Next Cycle

Proceed to **Step 4 (Todo 6): Unified Clover Integration (`Clover.SuperAgent` + `Clover.DecisionEngine` + `Clover.ModelRuntime`)**—implement the unified Python module and run the controlled 4-way ablation experiments specified in Attachment #1 (comparing Baseline, each capability independently, fixed loops, and adaptive Jev + Dory + SuperAgent).
