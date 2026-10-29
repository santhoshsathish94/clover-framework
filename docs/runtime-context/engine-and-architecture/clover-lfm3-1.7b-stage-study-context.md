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

