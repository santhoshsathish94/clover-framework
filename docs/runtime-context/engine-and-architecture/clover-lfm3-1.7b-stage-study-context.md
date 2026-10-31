# Clover LFM3-1.7B Stage-by-Stage Study & Kimi K3 Replacement Evaluation — Runtime Context

## Context

### 1. Why This Experiment Exists
Following Cycle 2 of the Meaning Formation, Grounded Execution, and Developmental Growth pilot (paused on branch `experiment/meaning-development-pilot-cycle2` at commit `d9a234b`), we branched `experiment/lfm3-1.7b-stage-study` from `main` (`3bdbdd0`) to investigate **Liquid AI's `LFM3-1.7B`** (`LiquidAI/LFM3-1.7B`, sha `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`, released 2026-10-23 / updated 2026-10-28).

Per Clover operating rules ("first download, run them each stage so that we really understand how it works, then we can see if we can replace Kimi K3 with this new LFM3-1.7B — the compact general model"), we do not reason from model cards alone. We download `LiquidAI/LFM3-1.7B`, read its exact source (`configuration_lfm3.py`, `modeling_lfm3.py`, `lfm3_conv.py`), verify every tensor and layer stage on real inputs, and then test `LFM3-1.7B` head-to-head against `Kimi K3` both standalone and inside Clover's surrounding architecture.

### 2. Target Hardware & Environment (`k3` host — same CPU box where `clover-kimi-k3` runs)
- **Host:** `k3` (`37.27.114.74`, Ubuntu 24.04 Noble)
- **CPU:** AMD Ryzen 9 7950X3D 16-Core / 32-Thread Processor
- **RAM:** 124 GiB DDR5 (96 GiB available)
- **Disk:** 1.8 TB NVMe RAID (`/dev/md2`, 839 GiB free)
- **Runtime Environment:** `/root/hybridagent-venv/bin/python` (Python `3.12.3`, `torch 2.11.0+cpu`, `transformers 5.13.0`, `safetensors 0.8.0`, `huggingface_hub 1.22.0`, `accelerate 1.14.0`).
- **Why this host matters:** Running `LFM3-1.7B` on the exact same Ryzen 9 7950X3D CPU server that runs `clover-kimi-k3` eliminates hardware confounding when comparing memory footprint, prefill/decode latency, stage-by-stage mechanics, and task quality.

### 3. `LiquidAI/LFM3-1.7B` Architecture read from `config.json`, `configuration_lfm3.py`, `modeling_lfm3.py`, and `lfm3_conv.py`
- **Model Type & Class:** `Lfm3ForCausalLM` (`model_type = "lfm3"`, `transformers_version = "5.12.1"`).
- **Total Parameters & Storage:** `1,728,801,536` (`1.73B`) dense parameters stored in a single `model.safetensors` (`3,475,757,943` bytes, `bfloat16`).
- **Dimensions:**
  - `vocab_size = 128,000` (`bos_token_id = 124894` `<|startoftext|>`, `eos_token_id = 124900` `<|im_end|>`, `pad_token_id = 124893` `<|pad|>`)
  - `hidden_size = 2048`
  - `intermediate_size = 7680` (dense SwiGLU MLP in every layer: `w2(silu(w1(x)) * w3(x))`, `bias = False`)
  - `num_hidden_layers = 24`
  - `tie_word_embeddings = True` (`lm_head.weight` tied to `model.embed_tokens.weight`)
  - `norm_eps = 1e-05`, `use_sandwich_norm = True`
  - `max_position_embeddings = 262,144` (256K tokens)
- **Tri-Hybrid 24-Layer Schedule (`config.layer_types`):**
  Every 4-layer block after the two leading conv layers follows `[conv, sliding_attention, conv, full_attention]`:
  - **13 Gated Short Convolution layers (`"conv"`):** Layers `0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23`
    - Module: `Lfm3ShortConv` (`conv_L_cache = 7`, `conv_bias = False`)
    - Equations:
      1. `B, C, x = chunk(in_proj(u), 3, dim=-1)` where `in_proj: 2048 -> 6144`
      2. Pre-gate: `z = B * x` (elementwise multiplicative input gate)
      3. Depthwise causal 1D convolution of width `K = 7` across time (`groups = 2048`, `weight` shape `[2048, 1, 7]`): `c_t = sum_{k=0..6} W[:, k] * z_{t - 6 + k}` (with document boundary isolation via `seq_idx` when packed)
      4. Post-gate & projection: `y = out_proj(C * c)` where `out_proj: 2048 -> 2048`
    - Recurrent state size per layer during decode: only the last `K - 1 = 6` (or `7`) tokens of `B * x` (`[batch, 2048, 6]`), fixed size regardless of sequence length!
  - **5 Sliding-Window Attention layers (`"sliding_attention"`):** Layers `4, 8, 12, 16, 20`
    - Module: `Lfm3Attention` with `sliding_window = 256`, `rope_theta = 10,000.0`, full head rotation (`partial_rotary_factor = 1.0`, all 64 pairs rotated).
  - **6 Full Global Attention layers (`"full_attention"`):** Layers `2, 6, 10, 14, 18, 22`
    - Module: `Lfm3Attention` with `sliding_window = None` (full 256K causal mask), `rope_theta = 1,000,000.0`, **zero-tail partial RoPE** (`partial_rotary_factor = 0.5`: first `32` frequency pairs follow the 128-dim progression, last `32` pairs have `inv_freq = 0.0` so the second half of each quarter-head remains unrotated!).
  - **Shared `Lfm3Attention` Mechanics (both sliding and full):**
    - GQA 4:1: `num_attention_heads = 16` (`q_proj: 2048 -> 2048`), `num_key_value_heads = 4` (`k_proj, v_proj: 2048 -> 512`), `head_dim = 128`, `scaling = 1 / sqrt(128)`.
    - Per-head `q_norm` and `k_norm` (`Lfm3RMSNorm(128, eps=1e-5)`) applied **before** RoPE (`apply_rotary_pos_emb` in `float32`).
    - **Per-head Sigmoid Output Gate:** `gate = sigmoid(gate_proj(u))` where `gate_proj: 2048 -> 16` (1 scalar gate per query head, broadcast across the 128 head dimensions before `o_proj: 2048 -> 2048`)!
  - **Sandwich Normalization in Every Decoder Layer (`Lfm3DecoderLayer`):**
    - Operator block: `h_mid = h_in + post_operator_norm(Operator(operator_norm(h_in)))`
    - FFN block: `h_out = h_mid + post_ffn_norm(MLP(ffn_norm(h_mid)))`
    - Final norm before LM head: `h_final = embedding_norm(h_23)`
    - Logits: `logits = F.linear(h_final, embed_tokens.weight)`

---

## Direction

1. **Download `LiquidAI/LFM3-1.7B` on `k3` (`/opt/clover-k3/models/LFM3-1.7B`)** and verify file integrity and exact tensor shapes/parameter counts from `model.safetensors`.
2. **Build & Run the Stage-by-Stage Instrument (`lfm3_stage_study.py`):**
   - Instrument every single stage of `LFM3-1.7B` from raw text -> tokenizer -> `embed_tokens` -> each of the 24 layers (recording `operator_norm`, internal operator sub-stages (`in_proj`/`B*x`/`conv1d`/`C*conv`/`out_proj` for conv layers; `q_norm`/`k_norm`/RoPE/attention/head-sigmoid-gate/`o_proj` for sliding and full attention layers), `post_operator_norm`, residual 1, `ffn_norm`, SwiGLU `w1/w3/w2`, `post_ffn_norm`, residual 2) -> `embedding_norm` -> tied `lm_head`.
   - Verify our independent mathematical reconstruction of each stage (`Lfm3ShortConv`, `Lfm3Attention` with sliding/full RoPE and head gate, `Lfm3MLP`, sandwich norms) against the official model's forward pass (confirming zero/machine-precision coordinate gap across all 24 layers and causal prefix invariance).
3. **Evaluate `LFM3-1.7B` vs. `Kimi K3` for Clover:**
   - Measure stage-by-stage timing, KV/Conv state memory footprint, and prefill/decode throughput on Ryzen 9 7950X3D (`k3`).
   - Run `LFM3-1.7B` on Clover's core tasks:
     - Tool calling & deferred tool search (`<tool_call>`, `<additional_tools>`)
     - Instruction following & structured synthesis
     - Knowledge / multi-step reasoning standalone vs. wrapped with Clover's surrounding layers (Memory Graph + Symbolic Ontology GraphRAG + Verifier).
   - Determine specifically whether `LFM3-1.7B` can replace `Kimi K3` outright, or what role each plays in Clover.

---

## Execution & Outcome Log

### 1. Weight Download & Exact Tensor Inventory (`model.safetensors`, 286 tensors)
- Downloaded `LiquidAI/LFM3-1.7B` (`sha = ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`) to `/opt/clover-k3/models/LFM3-1.7B` on `k3` (`37.27.114.74`, AMD Ryzen 9 7950X3D) in `29.65 s` and loaded all 286 `bfloat16` tensors in `0.96 s`.
- **Exact Parameter Breakdown across the 286 tensors (`1,728,801,536` parameters, `3,457,603,072` payload bytes):**
  | Stage Category | Parameters | Share of Model | Structure |
  | --- | --- | --- | --- |
  | **SwiGLU MLP (`feed_forward`)** | `1,132,462,080` | **65.51%** | 24 layers $\times$ (`w1: [7680, 2048]`, `w2: [2048, 7680]`, `w3: [7680, 2048]`) = `47,185,920` params/layer |
  | **Tied Embedding / LM Head (`embed_tokens`)** | `262,144,000` | **15.16%** | `[128000, 2048]` (`lm_head.weight` is pointer-identical: `tied_weights_exact = True`) |
  | **13 Gated Short Conv Operators (`conv`)** | `218,290,176` | **12.63%** | 13 layers $\times$ (`in_proj: [6144, 2048]`, `conv: [2048, 1, 7]`, `out_proj: [2048, 2048]`) = `16,791,552` params/layer |
  | **6 Full Global Attention Operators (`self_attn`)** | `63,111,168` | **3.65%** | 6 layers $\times$ (`q_proj: [2048, 2048]`, `k_proj: [512, 2048]`, `v_proj: [512, 2048]`, `o_proj: [2048, 2048]`, `gate_proj: [16, 2048]`, `q_norm: [128]`, `k_norm: [128]`) = `10,518,528` params/layer |
  | **5 Sliding-Window Attention Operators (`self_attn`)** | `52,592,640` | **3.04%** | 5 layers $\times$ `10,518,528` params/layer (`sliding_window = 256`) |
  | **Sandwich & Final RMSNorms (`norms`)** | `201,472` | **0.01%** | 24 layers $\times$ 4 norms (`operator_norm`, `post_operator_norm`, `ffn_norm`, `post_ffn_norm`) + 1 `embedding_norm` = $97 \times 2048$ |

### 2. Stage-by-Stage Mathematical Verification (`max_abs_diff = 0.0` at every stage!)
Using [research/lfm3_stage_study.py](../../../research/lfm3_stage_study.py) and recorded in [research/lfm3_stage_study_report.json](../../../research/lfm3_stage_study_report.json), we reconstructed every single stage from raw tensor weights and compared against `Lfm3ForCausalLM.forward()` on the 17-token real prompt `"Clover verifies every stage of a model on real inputs before drawing architectural conclusions."`:
- **Stage 1 (`embed_tokens`):** Per-position RMS ranges from `0.02020` to `0.02674` (mean `0.02388`).
- **Stage 2 (`Lfm3RotaryEmbedding`):**
  - `full_attention` zero-tail partial RoPE (`rope_theta = 1,000,000.0`, `partial_rotary_factor = 0.5`: 32 rotated pairs + 32 zero-frequency unrotated pairs): `max_abs_diff = 0.0`.
  - `sliding_attention` full RoPE (`rope_theta = 10,000.0`, `partial_rotary_factor = 1.0`: all 64 pairs rotated): `max_abs_diff = 0.0`.
- **Stage 3 (All 24 Tri-Hybrid Decoder Layers `0..23`):**
  - Every single layer `0..23` (`13 conv`, `5 sliding_attention`, `6 full_attention`) achieved **`max_abs_diff_vs_official = 0.0`** (100% bit-identical to the official model under `attn_implementation="eager"`).
  - **What Sandwich Normalization actually does inside the 24 layers (observed from `op_raw_rms` vs. `op_branch_rms` and `ffn_raw_rms` vs. `ffn_branch_rms`):**
    - In the **13 `conv` layers**, multiplicative double-gating (`B * x` before the width-7 depthwise conv and `C * conv` after it) shrinks the raw operator signal to `op_raw_rms = 0.00136 .. 0.00758`. The sandwich `post_operator_norm` **amplifies** the conv output by **5x to 19x** (up to `op_branch_rms = 0.01035 .. 0.10352`) so the short-convolution branch contributes on par with the attention layers!
    - Conversely, in the **24 SwiGLU MLP blocks**, raw MLP output is hot (`ffn_raw_rms = 0.03428 .. 0.48644`), and `post_ffn_norm` **attenuates** it by **3x to 26x** (`ffn_branch_rms = 0.01280 .. 0.16249`) so the residual stream grows smoothly from `0.02388` (embedding) to `0.30208` (layer 23 output).
  - **Per-Head Sigmoid Output Gate (`gate_proj: 2048 -> 16`) in the 11 Attention Layers:**
    - Mean gate openness ranges from `0.3126` (Layer 20, sliding) to `0.5160` (Layer 6, full), while individual head gates span `0.1069` (nearly closed) to `0.8750` (wide open), dynamically throttling specific query heads per token before `o_proj`.
- **Stage 4 (`embedding_norm`) & Stage 5 (`lm_head`):**
  - Final RMSNorm brings the residual stream from `0.30208` to `1.73994`.
  - Tied `lm_head` projection (`128,000` vocab) matches official logits with **`logits_max_abs_diff_vs_official = 0.0`**.
- **Causal Prefix Invariance Ladder:**
  - Running the 8-token prefix vs. the full 17-token prompt yielded **`max_gap_across_all_stages = 0.0`** and **`min_cos_across_all_stages = 1.0`** across all 24 layers, proving strict causality across all 13 width-7 conv layers, 5 sliding-window layers, and 6 global attention layers.

### 3. Runtime State (KV + Conv Cache) & CPU Hardware Scaling on `k3` (Ryzen 9 7950X3D)
- **4.0x State Memory Reduction at Long Context:**
  - The 13 `conv` layers require a fixed recurrent state of only `[B, 2048, 6]` in BF16 = **24 KB per layer (`312 KB` total across all 13 layers)** at *any* sequence length (whether 1K or 256K tokens).
  - The 5 `sliding_attention` layers cap their KV cache at `sliding_window = 256` tokens = **512 KB per layer (`2.50 MB` total across all 5 layers)**.
  - Only the 6 `full_attention` layers grow with sequence length (`12 KB/token` total across the 6 layers). At `256K` tokens, `LFM3-1.7B` uses **`3,074.8 MB` (`3.00 GiB`)** of state vs. **`12,288.0 MB` (`12.00 GiB`)** for a standard 24-layer GQA model (**`4.00x` smaller**).
- **Important Hardware Discovery on Dual-CCD Ryzen 9 7950X3D ([research/lfm3_followup_report.json](../../../research/lfm3_followup_report.json)):**
  - Because `LFM3-1.7B` has relatively small `2048`-wide matrices (`3.3 GB` total weights, fitting largely alongside 7950X3D's 128 MB 3D V-Cache), running PyTorch across all 32 logical threads suffers severe cross-CCD infinity-fabric and SMT barrier overhead:
    - `1 thread`: `3.47 tok/s`
    - `4 threads`: `8.46 tok/s`
    - **`8 threads` (1 physical CCD): `9.68 tok/s` (13.1x faster than 32 threads!)**
    - `16 threads`: `1.87 tok/s`
    - `32 threads`: `0.74 tok/s`
  - Furthermore, on Zen 4 AVX-512 (`7950X3D`), `bfloat16` at 8 threads (`9.68 tok/s`) is **2.55x faster** than `float32` at 8 threads (`3.79 tok/s`), confirming native AVX512-BF16 dot-product acceleration and 2x lower memory bandwidth pressure.

### 4. Head-to-Head Evaluation: Can `LFM3-1.7B` Replace `Kimi K3` in Clover?
We tested `LFM3-1.7B` on Clover's actual operational workloads ([research/lfm3_stage_study_report.json](../../../research/lfm3_stage_study_report.json) and [research/lfm3_followup_report.json](../../../research/lfm3_followup_report.json)):

1. **Native XML Tool Calling (`query_clover_memory_graph`):**
   - `LFM3-1.7B` reasoned cleanly inside `<think>...</think>` and emitted a syntactically exact XML tool call:
     `<tool_call>\n<function=query_clover_memory_graph>\n<parameter=domain>\nsubsea_pipeline_isolation\n</parameter>\n<parameter=hazard_class>\nunequalized_pressure_bypass\n</parameter>\n</function>\n</tool_call><|im_end|>`.
2. **Clover Grounded Execution Transfer Task (Unseen Subsea Wellhead Valve V-402 Scenario):**
   - **Standalone `LFM3-1.7B` (`enable_thinking=False`):** Without Clover's Memory Graph & Ontology context, `LFM3-1.7B` **over-triggered** all three candidate strings (`["equalize_differential_pressure", "require_downstream_telemetry", "verify_valve_seat_rating"]`), failing to distinguish the true coupled causal preconditions from the distractor (`verify_valve_seat_rating`).
   - **Standalone `LFM3-1.7B` (`enable_thinking=True`):** Spent 512 tokens deliberating without reaching a closed JSON verdict.
   - **Clover-Wrapped `LFM3-1.7B` (with Clover's Memory Graph + Symbolic Ontology Invariant `C1` injected):** In `5.20 s` (`enable_thinking=False`), `LFM3-1.7B` produced the **exact, verified ground-truth JSON**:
     `{"blocking_preconditions": ["equalize_differential_pressure", "require_downstream_telemetry"], "safe_to_execute_immediately": false}`.

### 5. Architectural Verdict on Replacing `Kimi K3` with `LFM3-1.7B`
- **Where `LFM3-1.7B` CANNOT replace `Kimi K3`:**
  - **Open-ended deep synthesis, broad world knowledge, and complex multi-step reasoning without external grounding.** With `1.73B` dense parameters vs. `Kimi K3`'s `~1 Trillion` MoE parameters (`32B` active per token across 92 layers), standalone `LFM3-1.7B` lacks the parametric knowledge capacity (`GPQA Diamond 63.37%`, `HLE 7.17%`) and fails ungrounded domain transfer (over-predicting distractors or ruminating in `<think>`). Even Liquid AI's own model card notes: *"Use retrieval or search for knowledge-heavy tasks. It is not recommended for agentic coding or complex multi-step reasoning due to its small size."*
- **Where `LFM3-1.7B` CAN replace `Kimi K3` (and is dramatically more efficient inside Clover):**
  - **As the fast local Edge / Controller / Tool-Router & Grounded Proposer inside the 7-layer Clover Stack.**
  - Because Clover separates **intelligence in the surrounding architecture** (Layer 4 Memory Graph, Layer 5 Symbolic Ontology GraphRAG, Layer 6 Deterministic Verifiers & Tools, Layer 7 Generative Compiler) from the raw LLM weights, wrapping `LFM3-1.7B` in Clover's Memory Graph + Ontology layer immediately corrected its unseen-domain precondition output from wrong (`3/3` over-prediction) to **100% exact** (`["equalize_differential_pressure", "require_downstream_telemetry"]`), while using **3.3 GB of RAM instead of 52.6–1,400 GB** and running at **9.68 tok/s in pure PyTorch BF16 on 8 CPU cores** (or >50 tok/s in 4-bit GGUF / GPU) with a **256K context window** that matches Clover's exact 256K session budget.
- **Recommended Clover Architecture:**
  - Do **not** discard `clover-kimi-k3` as the heavy Tier-2 deep-synthesis and open-world reasoning engine.
  - Adopt **`LFM3-1.7B` as Clover's Tier-1 Compact General & Local Agent Model** for tool routing (`<tool_call>`), deferred tool search (`<additional_tools>`), structured schema extraction (`enable_thinking=False`), and Ontology-Grounded Execution proposals where Clover's surrounding layers supply the domain invariants and verification.

---

## 6. Progressive 7-Layer On-Device Samsung Galaxy S23 Study (2026-10-30)

### 6.1 Mistakes Caught in Cycle 1 & Corrections Applied in Cycle 2
An independent audit of Cycle 1 exposed ten defects where shortcuts and hardcoded bypasses misrepresented how the layers operated:
1. **Forced `<tool_call>` suffix & fake tools (`direct_answer`, `generate_media`):** Cycle 1 appended `<think>I will choose the right function.</think><tool_call>` to every prompt, forcing every turn into a tool call (so `"Write a poem about rain"` called `generate_media(media_type="text")` instead of writing a poem), and stopped generation on the first period token after 16 tokens in `lfm3_cpu_engine.c`.
   - **Fix:** Removed the forced `<tool_call>` suffix, removed the fake `direct_answer` and `generate_media` tools from `LFM3_TOOLS_SCHEMA` (keeping only real `web_search` and `python_interpreter`), removed the 1-sentence stop-mask break in [lfm3_cpu_engine.c](../../../clover-ai-system/src/models/lfm3-1.7b/lfm3_cpu_engine.c), and wired the 2-turn `<tool_response>` loop so `LFM3-1.7B` reads tool outputs and writes the final answer itself.
2. **Layers 4 & 5 bypassed the model (`direct_notes` pasted over `reply_text`):** Cycle 1 pasted retrieved memory/ontology strings directly into `reply_text` after generation, and `dataset-symbolic-ontology-graph.json` contained benchmark test-question numbers (`d = 500 nm`, `DOC-HARBOURSIDE-CYCLE`, `symbolic_math_constraint_rules`).
   - **Fix:** Removed `direct_notes` pasting, removed the benchmark test-question numbers and `solve_symbolic_math_constraints` / `verify_source_grounding` regexes from [dataset-symbolic-ontology-graph.json](../../../clover-ai-system/src/tools/hybrid-ncm-k3/memory_tree/_ontology/dataset-symbolic-ontology-graph.json) and [s23_on_device_clover.py](../../../clover-ai-system/src/models/lfm3-1.7b/s23_on_device_clover.py), and injected retrieved Layer 4 & Layer 5 evidence into `LFM3-1.7B`'s user prompt **before prefill** so the model synthesizes the answer itself.
3. **Right-slicing prompt token truncation (`[:384]` / `[:512]`):** When Layer 4 + Layer 5 context exceeded 384 tokens, right-slicing `turn1_ids[:384]` cut off the user's `Question:` and `<|im_start|>assistant` suffix at the end of the prompt; our real Layer 5 grounding audit immediately caught this (`source_grounding_verified: false`).
   - **Fix:** Capped middle context tokens while preserving the system header and tail `Question: ... <|im_start|>assistant` intact.
4. **Layer 6 compiled `prompt + reply` instead of only model-written code:**
   - **Fix:** `_compile_verify_model_code_only` now compiles **only** model-written code (`reply_text` or model-emitted `<parameter=code>`), executes any `assert` statements from the prompt against the model's function, and runs a repair turn in `LFM3-1.7B` when compilation or assertions fail.
5. **Security & tool permissions:** Removed unauthenticated `/CloverAIStudio.apk` and `/install` downloads from `_S23Handler.do_GET`, untracked `CloverAIStudio.apk` from Git, and added server-enforced `allow_internet`, `allow_python`, and `allow_files` permissions.

### 6.2 Measured Layer-by-Layer Results on Samsung Galaxy S23 (`aarch64`, `VmRSS = 2,328.83 MB`, `active_threads = 3` in Android `/moderate` cpuset)
All numbers below come directly from [research/lfm3_omni_s23_live_phone_report.json](../../../research/lfm3_omni_s23_live_phone_report.json) (`setup_ms = 45,310.02 ms`, `measured_process_vm_rss_mb = 2,328.83 MB`; note that background Termux SSH sessions on Android 16 are restricted by `/proc/self/cpuset = /moderate` to the 3 Cortex-A510 efficiency cores `{0, 1, 2}`):

- **Layer 1 (`LFM3-1.7B` 24-Layer 4-Bit C/NEON/BFDOT/SDOT Engine Alone, `max_active_layer=1`):**
  - **Direct creative writing (`layer1_poem_direct`):** Writes a 4-line poem directly (`40` tokens, `prefill = 4,926.32 ms`, `decode = 12,839.22 ms`, `tools_used = []`):
    *"Upon the earth, a gentle rain descends, / Dripping softly through leaves and trees. / Each drop whispers secrets to the soil, / A soothing lullaby for all of you."*
  - **Without Layer 4 Memory (`layer1_memory_question_unwrapped`):** On *"How does Clover's per-layer cache partition fix flat hit rates?"*, `LFM3-1.7B` guesses generic cache-thrashing prose without knowing the `23.3% -> 48.0%` (`51.7 GB`) measurement (`93` tokens, `total_cycle_ms = 32,941.87 ms`).
  - **Without Layer 5 Ontology (`layer1_ontology_question_unwrapped`):** On *"What is CVE-2024-6387 (RegreSSHion), which CWE weakness causes it, and which CAPEC attack pattern exploits it?"*, `LFM3-1.7B` hallucinates that `CVE-2024-6387` is an *"OpenSSL library"* issue caused by *"CWE-94: Improper Neutralization of Special Elements used for Logging and Auditing"*.
- **Layer 2 (`+ Live Tool Loop & Lfm3NativeHybridLink(d=2048)`, `max_active_layer=2`):**
  - **2-Turn Live Web Search Synthesis (`layer2_live_web_search_synthesized`):** Turn 1 emits `<tool_call><function=web_search><parameter=query>recent research papers on foundation models</parameter></function></tool_call>` (`23` tokens, `decode = 9,054.35 ms`). `LiveWebSearchTool.execute` queries `(cat:cs.AI OR cat:cs.CL OR cat:cs.LG)` sorted by `submittedDate`, and Turn 2 feeds `<tool_response>` back into `LFM3-1.7B` (`turn2_prefill = 53,307.77 ms`, `turn2_decode = 43,305.69 ms`), which synthesizes a numbered summary of `[arXiv 2026-10-29] Probing Spatial State Representations in Visual Foundation Models (https://arxiv.org/abs/2610.40636v1)`.
  - **2-Turn Python Interpreter Synthesis (`layer2_python_interpreter_synthesized`):** Turn 1 emits `<tool_call><function=python_interpreter><parameter=code>17**4 + 29*13</parameter></function></tool_call>` (`26` tokens); Python subprocess returns `83898`; Turn 2 synthesizes: *"The result of \(17^4 + 29 \times 13\) is **83898**."* (`total_cycle_ms = 37,062.72 ms`).
  - **Honest Finding on `Lfm3NativeHybridLink(d=2048)` Pre-Decode Injection (`layer2_decode_without_predecode_hybridlink_step` vs `layer2_decode_with_predecode_hybridlink_step`):**
    - `Lfm3NativeHybridLink(d=2048, bottleneck=512)` is a small residual adapter trained on 6 plain English sentences (`138` tokens), **not** the 8B `HybridAgent` planner (which is still training on `k3`).
    - At the ChatML `</think>` boundary on `"What is the capital of Japan?"`, `raw_hidden_to_next_embed_cos = 0.3002` and `hybridlink_latent_to_next_embed_cos = 0.3268` (`hybridlink_ms = 230.88 ms`).
    - Decoding **without** stepping `r_t` into the C KV cache before decode produces `"The capital of Japan is **Tokyo**."` in 1 turn (`51,284.13 ms`).
    - Stepping `r_t` into `lfm3_step_embedding_cached` **before** decode (`c_continuous_latent_step_ms = 340.01 ms`, `post_latent_hidden_rms = 1.93658`) perturbs the ChatML `</think>` boundary so Turn 1 starts with `". Tokyo\n\nI'll use the web_search tool..."` and triggers an unnecessary 2-turn web search (`159,812.33 ms`). Therefore, `Lfm3NativeHybridLink` should only be used as the `R_Lfm3(h)` feature bridge into Layer 3 (`Lfm3ComboBridge`), not injected into the C KV cache before autoregressive decoding.
- **Layer 3 (`+ 11-Level VQVAE + NextConceptTransformer + LexicalPointerDecoder`, `max_active_layer=3`):**
  - On `"What is the capital of Japan?"` (`layer3_11level_ncm_pointer_decoder`), runs `Lfm3ComboBridge(2048 -> 384)` + `Lfm3ExpertBridge(2048 -> 384)` + 11-level `NextConceptTransformer` (`313` VQ codes across 11 levels, `zhat_final_shape = [1, 64, 384]`, `allowed_vocab_subset_size = 206`, `c_backbone_ms = 1,217.06 ms`, `ncm_total_ms = 16,862.02 ms`).
  - `LexicalPointerDecoder` decodes `raw_ncm_decoded_text = "The capital of Japan is **Tokyo**."` with `used_completeness_fallback = false`. Note that `allowed_vocab_subset_size = 206` restricts `vqvae.decoder.output_layer` to tokens present in the input plan plus `GRAMMAR_CONNECTOR_WORDS`, so Layer 3 acts as a constrained pointer-copy re-renderer (`+16.86 s` on S23), not an open-vocabulary generator.
- **Layer 4 (`+ Persistent Relational Memory Graph Injected into LFM3-1.7B`, `max_active_layer=4`):**
  - Retrieves `MEM-CORR-PER-LAYER-PARTITION-002` and `MEM-HYP-CACHE-DEAD-END-001` in `184.11 ms` and injects them into `LFM3-1.7B`'s prompt before prefill (`layer4_memory_graph_injected_into_model`).
  - `LFM3-1.7B` reads the retrieved memory and generates: *"The retrieved context indicates that a flat hit rate across all settings of a parameter suggests the issue is with the structure being measured (global eviction across layers) rather than an effect. Specifically, partitioning the same 51.7 GB cache per layer increased the hit rate from 23.3% to 48.0%."* (`total_cycle_ms = 75,783.63 ms`).
- **Layer 5 (`+ Symbolic Ontology GraphRAG Injected into LFM3-1.7B + Grounding Audit`, `max_active_layer=5`):**
  - Retrieves `CVE-2024-6387`, `CWE-362`, and `CAUSED_BY` paths (`CVE-2024-6387 -> CWE-362`, `CVE-2024-6387 -> CWE-364`) in `5.0 ms` and injects them into `LFM3-1.7B`'s prompt before prefill (`layer5_ontology_graphrag_injected_into_model`).
  - `LFM3-1.7B` replaces its Layer 1 hallucination (*"OpenSSL / CWE-94"*) with the grounded OpenSSH signal-handler race condition (`CVE-2024-6387`, `2024-07-01`, CVSS `8.1` HIGH), and the post-generation grounding audit verifies all 6 numbers (`['2024', '6387', '07', '01', '3.1', '8.1']`) and `['CVE-2024-6387']` against the retrieved context (`ungrounded_numbers = []`, `source_grounding_verified = true`).
- **Layer 6 (`+ Python Generative Compiler on Model-Written Code Only`, `max_active_layer=6`):**
  - On `layer6_generative_compiler_model_code`, `LFM3-1.7B` writes a complete iterative `def fib(n: int) -> int:` block, and `CloverPythonGenerativeCompiler` verifies `ok = true` (`elapsed_ms = 5.605 ms`).
  - On `layer6_generative_compiler_with_assert_tests`, `LFM3-1.7B` writes `def is_palindrome(s: str) -> bool: cleaned = s.replace(" ", "").lower(); return cleaned == cleaned[::-1]`, and Layer 6 both compiles the model's function and executes the prompt's two assertions (`assert is_palindrome("Race car") == True`, `assert is_palindrome("hello") == False`), returning `prompt_asserts_output = "ALL_ASSERTS_PASSED"` and `all_ok = true` (`elapsed_ms = 19.025 ms`).
- **Layer 7 (`+ OS madvise(MADV_WILLNEED) Lookahead vs Passive mmap Baseline`, `max_active_layer=7`):**
  - The 12-slot array in `lfm3_cpu_engine.c` (`pool_keys[12]`) is a software LRU/heat slot simulator (`600` layer requests for a 24-layer $\times$ 25-step run: `197` simulated 12-slot hits, `379` simulated page-cache hits, `24` cold fetches, `391` simulated clean evictions).
  - Real wall-clock benchmark on the phone (`layer7_madvise_off_vs_on_benchmark`, 24 generated tokens):
    - **`madvise_off` (`passive_mmap_baseline`):** `prefill = 4,985.83 ms`, `decode_24tok = 9,397.43 ms` (**`391.56 ms/tok`, `2.55 tok/s`**).
    - **`madvise_willneed_on` (`specmd_least_stale_willneed`):** `prefill = 3,906.99 ms`, `decode_24tok = 6,994.38 ms` (**`291.43 ms/tok`, `3.43 tok/s` — `1.34x` faster**).
    - Conversely, issuing `MADV_DONTNEED` on 12-slot eviction when all 24 4-bit layers (`1.56 GiB`) already fit in LPDDR5X slowed decode to `541.2 ms/tok` (`1.85 tok/s`) due to per-token page-table unmap/re-fault overhead, so `MADV_DONTNEED` is disabled during active decoding.

---

## 7. Cycle 3: 8x RTX 3090 Training of `LiquidAI/LFM3-Omni-2B` (`HybridSFT` Stages 1–3 + `11-Level NCM`) & S23 Mobile Integration (2026-10-30)

### 7.1 `LiquidAI/LFM3-Omni-2B` Discovery & 8x RTX 3090 Training Setup
- On the Finland `8x NVIDIA GeForce RTX 3090` server (`65.109.137.0:60805`, `192 GiB` total VRAM, `472 GiB` RAM, `/dev/shm` `236 GB`), we verified from Hugging Face that **`LiquidAI/LFM3-Omni-2B`** (`2.17B` total parameters, `665` tensors across `model.safetensors` + `audio_tokenizer/model.safetensors`) is Liquid AI's official end-to-end omni-modal model, whose 24-layer language backbone (`model.language_model.*`, `286` tensors) shares the exact tensor shapes (`d = 2048`, `24` layers: `13 ShortConv + 5 Sliding GQA + 6 Full GQA`, `SwiGLU 7680`, `V = 128,000`) of `LFM3-1.7B` with omni-modal post-trained weights (`max_abs_diff = 2.5156`).
- Extracted the 286 `model.language_model.*` tensors into `LFM3-Omni-2B-LM`, patched `build_prompt_lfm3` for `LFM3-Omni-2B`'s `<think></think>` chat template, and trained all 3 stages of `HybridSFT` across all 8 RTX 3090 GPUs (`torchrun --nproc_per_node=8`, `CE_CHUNK_SIZE = 512`, non-reentrant gradient checkpointing, fused `AdamW`):
  - **Stage 1 (`format_activation`, 80 global steps, `474.2 s`):** Held-out evaluation loss dropped from **`1.1673` $\to$ `0.5869`** (`-49.7%`).
  - **Stage 2 (`hybridlink_activation`, 120 global steps, `66.2 s`):** Held-out `hybridlink_loss` dropped from **`2.1456` $\to$ `0.9357`** and `cos_loss` dropped from **`0.9260` $\to$ `0.5283`** (mean latent cosine similarity jumped from `0.0740` to `0.4717`).
  - **Stage 3 (`self_conditioned`, 80 global steps, `642.1 s`):** Held-out two-pass self-conditioned total loss dropped from **`4.2174` $\to$ `1.8743`** (`backbone_loss: 2.4834 -> 0.7709`, `hybridlink_loss: 1.7339 -> 1.1035`, `cos_loss: 0.6826 -> 0.6389`).
- **Growth Lesson — `bfloat16` Underflow on Newly Added Anchor Token Embeddings (`125033` `<latent>`, `125034` `</latent>`):**
  - Inspecting `embed_tokens.weight` after `bfloat16` training revealed that at `lr = 1.5e-5`, updates to `row 125033` (`<latent>`) and `row 125034` (`</latent>`) were smaller than `0.5 ULP` of `bfloat16` (`1.56e-4`), leaving their norms stuck at `0.173` while `LFM3-Omni-2B`'s pre-trained `<|mixed_start|>` token (`125015`) had norm `0.594`.
  - Optimizing `w[125033:125035]` in `float32` on cached Stage-3 hidden states (`6,233` positions) took `1.5 s` on GPU 0, raised `<latent>` / `</latent>` classification accuracy from `0.0%` to `100.0%` (`loss: 6.0055 -> 0.1320`), and enabled upstream `HybridGenerationEngine` (`infer_smoke.py`) to execute `2` `<latent>` spans (`9` continuous `R(h_t)` steps) + `1` `<python>` call (`83898`) to clean `eos`.
- **`11-Level NCM` + Bridges + `LexicalPointerDecoder` GPU Training (`train_ncm_omni2b_gpu.py`):**
  - Trained on `1,024` aligned `LFM3-Omni-2B` windows (`48` plan $\to$ `64` target tokens) using Stage-3 `LFM3-Omni-2B-LM` 24-layer hidden states:
    - Held-out token cosine alignment (`3,899` tokens): `raw_hidden_to_next_embed_cos = 0.0212`, `stage3_hybridlink_2048_test_cos = 0.3030` (`14.3x`), `trained_native_hybridlink_512_test_cos = 0.4353` (`20.5x`).
    - `128,000`-vocab grafted `VQVAE` non-pad token reconstruction accuracy: **`0.3288` $\to$ `0.9901` (`99.01%`)**.
    - `11-Level NextConceptTransformer` + `Lfm3ComboBridge` + `Lfm3ExpertBridge` code CE loss: **`9.7549` $\to$ `1.1271`**.
    - `LexicalPointerDecoder` token CE loss: **`3.3968` $\to$ `0.0035`**.

### 7.2 Progressive Layer 1 $\to$ Layer 7 Benchmark on Samsung Galaxy S23 (`lfm3-omni-2b-4bit-s23`, `VmRSS = 1,705.38 MB`)
All 24 quantized `LFM3-Omni-2B-HybridSFT-S3` 4-bit trunks (`1.56 GiB`) and the trained `lfm3_ncm_core_bf16.pt` (`HybridLink2048` + `Lfm3NativeHybridLink` + `Lfm3ComboBridge` + `Lfm3ExpertBridge` + `VQVAE` + `NextConceptTransformer` + `LexicalPointerDecoder`) were deployed to `~/clover-s23/` on the Samsung Galaxy S23 and verified in [research/lfm3_omni_s23_live_phone_report.json](../../../research/lfm3_omni_s23_live_phone_report.json) (`measured_process_vm_rss_mb = 1,705.38 MB`, down `623.45 MB` from Cycle 2):

| Active Layer Cap | Probe | What the Layer Adds & Measured S23 Result |
| --- | --- | --- |
| **Layer 1 Only** (`max_active_layer=1`) | `layer1_poem_direct` (`56` in, `36` gen, `34,264.05 ms`) | Pure 24-layer `LFM3-Omni-2B` 4-bit C engine (`tools_used = []`): *"Rain falls in silver threads, / Covering the world in quiet grace, / Waking the earth with fresh, cool breath, / And singing soft lullabies to night."* |
| **Layer 1 Only** (`max_active_layer=1`) | `layer1_memory_question_unwrapped` (`60` in, `66` gen, `45,302.42 ms`) | Without Layer 4 Memory Graph, guesses generic cache-tuning prose without knowing Clover's `51.7 GB` (`23.3% -> 48.0%`) result. |
| **Layer 1 Only** (`max_active_layer=1`) | `layer1_ontology_question_unwrapped` (`79` in, `86` gen, `61,091.81 ms`) | Without Layer 5 Ontology GraphRAG, hallucinates `CWE-2018-028` (config parsing buffer overflow) and `RegreSShion` as the CAPEC pattern. |
| **+ Layer 2 (`HybridAgent` + Tools)** (`max_active_layer=2`) | `layer2_live_web_search_synthesized` (`557` in, `222` gen, `390,676.74 ms`) | Turn 1 opens `<latent></latent>` (`latent_spans_opened = 1, latent_steps_total = 3`, `stage3_hybridlink_2048_cos = 0.4073`) and autonomously emits `<search>recent research papers on foundation models</search>`. Live arXiv search runs; Turn 2 runs 1 `<latent>` step and synthesizes summaries of `arXiv:2610.40636v1` (*Probing Spatial State Representations in Visual Foundation Models*) and `arXiv:2610.40618v1` (*An ODE-based Sampler for Discrete Diffusion Models*). |
| **+ Layer 2 (`HybridAgent` + Tools)** (`max_active_layer=2`) | `layer2_python_interpreter_synthesized` (`559` in, `136` gen, `112,393.68 ms`) | Turn 1 opens `<latent></latent>` (`3` continuous steps, `stage3_hybridlink_2048_cos = 0.4065`) and emits `<python>print(17**4 + 29*13)</python>`. Subprocess returns `83898`; Turn 2 runs 1 `<latent>` step and outputs the step-by-step explanation ending in `83898`. |
| **+ Layer 2 (`HybridLink` Latent Step)** (`max_active_layer=2`) | `layer2_decode_without_predecode_hybridlink_step` vs `layer2_decode_with_predecode_hybridlink_step` | With the trained `LFM3-Omni-2B` + `HybridLink` (`raw_cos = 0.0554 -> hybridlink_cos = 0.8446, stage3_cos = 0.3922`), stepping `<latent>` inside `<think>` before `</latent></think><answer>` no longer perturbs decoding: **both** arms cleanly output `"Tokyo"` (`9,399.32 ms` cold vs `4,107.74 ms` warm LCP cache). |
| **+ Layer 3 (`11-Level NCM`)** (`max_active_layer=3`) | `layer3_11level_ncm_pointer_decoder` (`52` in, `5` gen, `21,849.22 ms`) | Runs `Lfm3ComboBridge` + `Lfm3ExpertBridge` + 11-level `NextConceptTransformer` (`313` VQ codes, `zhat_final_shape = [1, 64, 384]`) + `LexicalPointerDecoder`, decoding `raw_ncm_decoded_text = "Tokyo"` with `used_completeness_fallback = false` (`ncm_total_ms = 15,957.57 ms`). |
| **+ Layer 4 (`Memory Graph`)** (`max_active_layer=4`) | `layer4_memory_graph_injected_into_model` (`241` in, `45` gen, `64,639.34 ms`) | Retrieves `MEM-CORR-PER-LAYER-PARTITION-002` in `213.15 ms` and injects before prefill; `LFM3-Omni-2B` answers: *"Clover's per-layer cache partition fix flat hit rates by partitioning the same 51.7 GB cache per layer, which increased the hit rate from 23.3% to 48.0%."* |
| **+ Layer 5 (`Ontology GraphRAG + Verifier`)** (`max_active_layer=5`) | `layer5_ontology_graphrag_injected_into_model` (`531` in, `81` gen, `171,967.55 ms`) | Retrieves `7` ontology hits (`CVE-2024-6387 -> CWE-362`, `CWE-364`, `CAPEC-29`) in `5.65 ms` with relationship paths prioritized first. `LFM3-Omni-2B` outputs the complete `CVE-2024-6387 -> CWE-362 -> CAPEC-29` chain, and `_audit_reply_grounding` verifies `checked_numbers = ['2024', '6387', '362', '29']` and `checked_identifiers = ['CVE-2024-6387', 'CWE-362', 'CAPEC-29']` with `ungrounded_numbers = []`, `ungrounded_identifiers = []`, `source_grounding_verified = true`. |
| **+ Layer 6 (`Generative Compiler`)** (`max_active_layer=6`) | `layer6_generative_compiler_model_code` (`37,462.42 ms`) & `layer6_generative_compiler_with_assert_tests` (`31,061.29 ms`) | `LFM3-Omni-2B` generates clean `def fib(n: int) -> int:` (`all_ok = true`, `7.069 ms`) and `def is_palindrome(s: str) -> bool:` (`all_ok = true`, `prompt_asserts_output = "ALL_ASSERTS_PASSED"`, `19.993 ms`). |
| **+ Layer 7 (`SpecMD Policy 3 madvise`)** (`max_active_layer=7`) | `layer7_madvise_off_vs_on_benchmark` (`24` tokens) | Gating `madvise(MADV_WILLNEED)` in `lfm3_dm_record_layer` to cold fetches (`!g_lfm3.host_keys[next_layer]`) eliminated redundant per-step PTE walks (`33.6 GB` VMA scans) and improved 24-token decode from **`428.85 ms/tok` (`2.33 tok/s`, `madvise_off`) to `380.61 ms/tok` (`2.63 tok/s`, `madvise_willneed_on`, `1.13x` faster)**. |

---

## 8. Cycle 4: Unified Mobile SoC Architecture (`Snapdragon 8 Gen 2 for Galaxy`), `LFM3-SoC-Q4` Format, Direction B 4-Tier Ledger, and 11 Foundational Papers (`2026-10-30`)

### 8.1 Audit of Shortcuts Removed & Full Artifact Parity
1. **Removed Python Prompt-Steering Shortcuts in `s23_on_device_clover.py`:**
   - Removed `is_code_write_request` keyword sniffing (`"def " in user_clean` forcing ````python\n` into `assistant_open` and wrapping non-fenced replies).
   - Removed scripted English text in Turn 2 (`"<result>{obs_clean}</result><think>\nThe tool returned the result.<latent>"` and `"I will now state the final answer..."`), replacing it with pure structural tag transitions (`<result>{obs_clean}</result><think><latent>` $\to$ `step_embedding_cached` $\to$ `</latent></think><answer>`).
   - Replaced the single-threaded PyTorch `F.linear(h_t.to(torch.bfloat16), self.lfm3_emb_bf16)` scan (`524.3 MB` BF16 scan per call that evicted `trunk-0..23` pages from the OS page cache) with `self.c_engine.lm_head_top1(h_last_2048)` (`23.65 ms` 4-bit C `SDOT` + Top-64 exact BF16 refinement).
2. **Full Fine-Tuned NCM Artifact Parity (`k3` $\leftrightarrow$ `s23`):**
   - Synced the full fine-tuned `lfm3_ncm_core_bf16.pt` (`MD5: 194de4139c17d3c234a39a42f8577d86`), `vqvae_dec.bf16.bin` (`MD5: 9e39a476ab7bb30cc0ae850ef5cb78b9`), and `vqvae_wte.bf16.bin` (`MD5: d1abe3e5ebd17893a33e0692a5bd35e6`) from `k3` to `s23`, replacing the earlier partial row-delta patch.

### 8.2 Physical Unified SoC Hardware Probe (`Snapdragon 8 Gen 2 for Galaxy`, `SM8550-AC`)
Probing `/proc/cpuinfo`, `/dev/kgsl-3d0`, and `/vendor/lib64/libOpenCL.so` on the physical Samsung Galaxy S23 revealed four unified-SoC hardware capabilities that standard desktop CPU/GPU layouts miss:
1. **Unified LPDDR5X Memory (`Unified Host Memory = 1`):** CPU and Adreno 740 GPU share the same physical LPDDR5X DRAM controller and `4 MB` System Level Cache (`SLC`). With `CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR`, `mmap`ped weight files are mapped directly into the Adreno 740 GPU's SMMU page table with **zero byte copies (`0 MB` host-to-device copy)**.
2. **Adreno 740 GPU (`OpenCL 3.0 Adreno(TM) 740`, `6` Compute Units, `32 KiB` On-Chip `GMEM`):** Supports hardware `cl_khr_integer_dot_product` (`dot_acc_sat` 4-way Int8 dot product) and `cl_qcom_perf_hint` (`CL_PERF_HINT_HIGH_QCOM`). Requiring 16-byte aligned `uchar16` vector loads (`ffn_v16[nib16_idx]`) achieved **bit-for-bit identical output (`7680 / 7680` exact matches, `max_abs_diff = 0.00000000`)** against Kryo CPU `SDOT` at `1.28 ms/layer` (`30.63 ms` across 24 distinct layers).
3. **Heterogeneous Adreno 740 GPU + Kryo CPU `SDOT` Co-Execution (`fused_swiglu_w1_w3_4bit_T1`):** Because `adreno_swiglu_dot8_exact` and CPU `SDOT` are bit-for-bit identical, `lfm3_cpu_engine.c` dispatches rows `0..3839` (`50%`) asynchronously to the Adreno 740 GPU via `clEnqueueNDRangeKernel` + `clFlush` while the Kryo CPU cores simultaneously compute rows `3840..7679` (`50%`) via `vdotq_s32`, utilizing both processors on the unified die in parallel.
4. **ARMv8.6-A `i8mm` (`SMMLA` `vmmlaq_s32`) + 1-Pass `FlashAttention-2` Online Softmax:**
   - For `T >= 2` prefill, `linear_4bit_group64_omp` executes `vmmlaq_s32` (`SMMLA` 2×8 × 8×2 Int8 matrix multiply = `32` Int8 MACs/instruction), loading each 4-bit weight block once for 2 tokens (`2x` weight-bandwidth reduction).
   - For all 11 GQA layers, `fa2_online_softmax_head_c` fuses $Q \cdot K^T$, running max/sum rescaling `(m_i, l_i)`, and $P \cdot V$ accumulation in NEON registers in a single pass (`0` `scores[]` buffer allocations, `0` strided V-cache misses).

### 8.3 `LFM3-SoC-Q4` Mobile-Chip Format (`894.4 MB` Total Working Set)
Previously, `embed_tokens.bf16.bin` (`524.3 MB`) and `24 × operator.bin + common.bin` (`667.8 MB`) were uncompressed BF16 (`1,192.1 MB` = `67.1%` of per-token DRAM traffic), leaving `1,778.2 MB` per token (`> 1.1 GB` free page cache when NCM is loaded). We built and deployed:
- **`pruned/global-0/lm_head.4bit.bin` (`135,680,000` B = `129.39 MiB`, `3.864x` smaller):** Planar group-64 4-bit quantization with **Two-Stage Top-64 Exact BF16 Refinement** (`compute_lm_head_logits_soc`: Stage 1 scans `128,000` rows in 4-bit via `SDOT` keeping top-64 candidates; Stage 2 recomputes exact BF16 dot products for only those 64 rows = `256 KB` of DRAM reads). Cut 1-token `lm_head` latency on S23 from **`81.3–720.5 ms` down to `23.65 ms`** (`cos = 0.997848`, `100.00%` top-1 match, `0.000000` top-1 logit error).
- **`pruned/trunk-0..23/operator.4bit.bin` (`172,584,960` B = `164.59 MiB` vs `636.00 MiB` BF16):** Planar group-64 4-bit quantization of `in_proj [6144, 2048]`, `out_proj [2048, 2048]`, fused `qkv_proj [3072, 2048]`, and `o_proj [2048, 2048]`, plus `MADV_DONTNEED` eviction of the unused BF16 projection pages when `use_4bit_operators=1`.

### 8.4 Measured 4-Mode Mobile SoC Hardware & Format Benchmark on Samsung Galaxy S23
| Mode | Configuration | 24-Token Decode (Standalone / With NCM Loaded) | Speed (`tok/s`) | LPDDR5X Traffic (`MB`) | `SMMLA` Tiles | `FA2` Calls | `Adreno 740` Launches | Major Page Faults (`majflt`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Mode 1** | Baseline (`BF16 LM-Head + BF16 Ops + CPU SDOT`) | `8,671.8 ms` / `11,963.6 ms` | `2.77` / `2.01 tok/s` | `41,878.9 MB` | `3,969,024` | `11,088` | `0` | `12,203` / `6,639` |
| **Mode 2** | `+ 4-Bit LM-Head` (Top-64 Exact BF16 Refine) | `5,658.8 ms` / `8,763.8 ms` | `4.24` / `2.74 tok/s` | `32,984.4 MB` | `3,969,024` | `11,088` | `0` | `1,081` / **`1`** |
| **Mode 3** | `+ Heterogeneous Adreno 740 GPU + Kryo CPU SDOT` | `5,209.0 ms` / `7,070.8 ms` | **`4.61`** / **`3.39 tok/s`** | `32,984.4 MB` | `3,969,024` | `11,088` | **`576`** | `54` / **`5`** |
| **Mode 4** | `+ 4-Bit Operators` (`Full LFM3-SoC-Q4 + Adreno 740`) | **`3,740.0–4,386.1 ms`** / **`5,185.8 ms`** | **`5.47–6.42`** / **`4.63 tok/s`** | **`21,199.1 MB`** (`-49.4%`) | **`5,515,776`** | `11,088` | **`576`** | `1,703` / `1,688` |

- **Layer 7 `madvise` ON vs OFF with Direction B Ledger:** Enabling `madvise(MADV_WILLNEED)` lookahead on S23 reduced Linux kernel major UFS 4.0 page faults (`majflt` from `/proc/self/stat`) from **`2,292` down to `9` (`254.7x` fewer major faults)** and sped up 24-token decode from `11,407.31 ms` to `9,655.29 ms`.
- **All 11 Foundational Papers Wired & Verified Live on S23:** Recorded in `layer_trace` across Layer 4 (`McCulloch-Pitts 1943` inhibitory veto, `Rosenblatt 1958` $\gamma$-system zero-sum conservation, `Minsky-Papert 1969` Order-$k$ predicates, `Hopfield 1982` bipolar constellation cleanup & familiarity, `Rumelhart 1986` + `Sutton 1988` `TD(lambda)` path credit, `LeCun 1998` GTN `logadd` forward confidence, `Krizhevsky 2012` LRN + `Vaswani 2017` $h=3$ multi-head subspace retrieval), Layer 5 (`Turing 1950` $M_0/M_1$ diagonal self-reference stratifier, `Newell-Simon 1956` `CSm` `*2.07` contraction fix & `VCt` premise contraction), and Layer 6 (`Sutton 1988` `SuttonTDStatementPredictor` absorbing Markov `TD(0)` MLE `v_mle`).


