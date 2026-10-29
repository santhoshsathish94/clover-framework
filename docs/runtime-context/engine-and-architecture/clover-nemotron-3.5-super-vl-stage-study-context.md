# Clover NVIDIA Nemotron 3.5 Super VL (`120B-A12B`) Stage-by-Stage Study & Kimi K3 Comparison — Runtime Context

## Context

### 1. Why This Experiment Exists
After completing and pushing the `LFM3-1.7B` stage-by-stage study on branch `experiment/lfm3-1.7b-stage-study` (`a5f99be`), we created branch `experiment/nemotron-3.5-super-vl-stage-study` from `main` (`3bdbdd0`) to investigate **NVIDIA's `Nemotron 3.5 Super VL`** (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` and `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`, released October 2026).

While `LFM3-1.7B` (`1.73B` dense) tested the compact edge end of the spectrum (where Clover's surrounding Memory Graph + Ontology layer had to compensate for limited standalone parametric capacity), **`NVIDIA-Nemotron-3.5-Super-VL-120B-A12B`** is a frontier-class **120B total / 12B active** multimodal Mamba-2–Transformer–Latent-MoE hybrid with native vision/video (`C-RADIO`), native Multi-Token Prediction (`MTP`), and a **1M-token context window** (`GPQA Diamond 89.52%`, `LiveCodeBench v6 88.22%`, `SWE-bench Verified 71.10%`, `GraphWalks 89.71%`).

Following Clover's operating principles, we first download the model's exact code, configuration, tensor index, and stage weights on `k3` (`37.27.114.74`), run and verify every single stage mathematically (`C-RADIO Vision Encoder` $\to$ `PixelShuffle + ReLU² Vision Projector & EVS Video Pruning` $\to$ `Token Embedding + Multimodal Scatter` $\to$ `Mamba-2 Mixer` $\to$ `Latent ReLU² MoE Mixer` $\to$ `NoPE 16:1 GQA Attention Mixer` $\to$ `Final RMSNorm + Untied LM Head` $\to$ `2-Layer Native MTP Speculative Head`), and evaluate how `Nemotron 3.5 Super VL` compares against `Kimi K3` (`clover-kimi-k3`).

### 2. Primary Sources Inspected (`/opt/clover-k3/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` on `k3`)
1. `config.json` & `model.safetensors.index.json` (`43,079` tensors across `63` BF16 shards, `124,453,395,976` parameters = `248,906,833,960` bytes; plus `NVFP4` checkpoint at `75.05 GB`).
2. `configuration_nemotron_h_omni.py` & `modeling_nemotron_h_omni.py` (`NemotronH_Omni_Reasoning_V3`, `NemotronH_Omni_Reasoning_V3VisionProjector`, `NemotronH_Omni_Reasoning_V3MLP`, `compute_retention_mask`).
3. `configuration_radio.py` & `modeling_radio.py` (`RadioModel`, `RadioInputConditioner`, `RadioPatchEmbeddings`, `RadioLayer`, `RadioAttention`, `RadioMLP`, `RadioLayerScale`).
4. `transformers.models.nemotron_h.modeling_nemotron_h` (`NemotronHMamba2Mixer`, `NemotronHAttention`, `NemotronHMoE`, `NemotronHTopkRouter`, `NemotronHExperts`, `NemotronHBlock`, `NemotronHModel`, `NemotronHForCausalLM`).

### 3. Complete Stage-by-Stage Architecture Read from the Source Code
1. **Stage V1 — C-RADIO Vision & Video Encoder (`vision_model: RadioModel`, 391 tensors):**
   - `RadioInputConditioner`: normalizes RGB pixels by `norm_mean = [0.48145466, 0.4578275, 0.40821073]`, `norm_std = [0.26862954, 0.26130258, 0.27577711]` in `float32`.
   - `RadioPatchEmbeddings`:
     - Image path: splits $512 \times 512$ image into $32 \times 32 = 1024$ patches of $16 \times 16 \times 3 = 768$, projects via `patch_projection` (`768 -> 1280`, `bias=False`).
     - Video path: packs pairs of frames (`video_temporal_patch_size = 2`) into $2 \times 3 \times 16 \times 16 = 1536$ and projects via `video_patch_projection` (`1536 -> 1280`, `bias=False`).
     - Adds bilinearly interpolated Cropped Position Embedding (`position_embedding: [1, 16384, 1280]` from $128 \times 128$ max grid) and prepends **`10` prefix tokens** (`cls_register_token: [10, 1280]` = `3` CLS tokens + `7` register tokens).
   - `RadioEncoder`: **32 ViT-Huge layers** (`hidden_size = 1280`, `16` heads of `80` dim, `qkv_bias = True`, `RadioLayerScale`, `LayerNorm(eps=1e-6)`, `RadioMLP` `1280 -> 5120 -> 1280` with `gelu`).
   - Output slicing: strips the `10` prefix tokens (`num_skip = 10`) to return spatial patch features `[B, 1024, 1280]`.
2. **Stage V2 — Pixel-Shuffle Vision Projector & EVS Video Token Pruning (`vision_projector`, 5 tensors):**
   - `vision_final_layernorm`: `nn.LayerNorm(1280, eps=1e-6)` applied per patch token before spatial merge.
   - `pixel_shuffle(scale_factor=0.5, ps_version="v2")`: folds $2 \times 2$ spatial neighborhoods of $1280$-dim patches into `5120`-dim super-patches, reducing $32 \times 32 = 1024$ patches by **4x** to $16 \times 16 = 256$ visual tokens per $512 \times 512$ image (`[B, 256, 5120]`).
   - `mlp1` (`NemotronH_Omni_Reasoning_V3MLP`):
     $$\text{mlp1}(x) = W_2\,\operatorname{ReLU}^2\!\big(W_1\,\operatorname{RMSNorm}(x)\big)$$
     where `norm: [5120]`, `linear1: [20480, 5120]`, `linear2: [4096, 20480]`, and $\operatorname{ReLU}^2(z) = (\max(0, z))^2$.
   - `compute_retention_mask` (Efficient Video Sampling / EVS): for video grids $(T, H, W, 4096)$, computes temporal cosine dissimilarity $1 - \cos(v_t, v_{t-1})$ per spatial cell, forces Frame 0 to `255.0` so all first-frame tokens survive, and retains the top $(1 - q)$ fraction (`video_pruning_rate = 0.7` in config, `0.5` in serving).
3. **Stage L0 — Token Embedding & Multimodal Scatter (`language_model.backbone.embeddings`):**
   - Untied embedding table `[131072, 4096]` (`bfloat16`).
   - Replaces `<image>` placeholder positions (`img_context_token_id = 18`) via `inputs_embeds.masked_scatter(image_mask, image_embeds)`.
4. **Stage L1..L88 — 88 Single-Mixer `NemotronHBlock` Backbone Layers (`residual + mixer(RMSNorm(h))`):**
   - Unlike standard Transformers where every layer runs `Attention + FFN`, each of the 88 `NemotronHBlock` layers runs **one pre-RMSNorm (`eps=1e-5`) and one mixer** following the 8-group macro-schedule:
     `MEMEMEMAEMEMEMEMAEMEMEMEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMAEMEMEMEME`
     - **40 Mamba-2 Sequence-Mixer Layers (`"mamba"` — e.g. Layer 0):**
       - `in_proj: [18560, 4096]` splits into `gate` (`8192`), `conv_input` (`10240 = 8192 [x] + 1024 [B] + 1024 [C]`), and `dt` (`128` heads).
       - Causal depthwise `conv1d` (`[10240, 1, 4]` with bias `[10240]`) + `silu` activation, then split into `x` (`128 heads x 64 dim`), `B` (`8 groups x 128 state_dim`), `C` (`8 groups x 128 state_dim`).
       - Selective State Space (SSD) chunk scan / recurrent update with `dt = clamp(softplus(dt + dt_bias), min=0.001)`, `A = -exp(A_log.float())` (`[128]`), skip `D` (`[128]`), `ssm_state` in `float32` (`[B, 128, 64, 128]`).
       - `Zamba2RMSNormGated(8192, group_size=1024)` (multiply by `silu(gate)` then group-RMSNorm per 1024-dim group) $\to$ `out_proj: [4096, 8192]`.
     - **40 Latent `ReLU²` MoE Channel-Mixer Layers (`"moe"` — e.g. Layer 1):**
       - `gate` (`NemotronHTopkRouter`): FP32 linear `[512, 4096]` $\to$ `scores = sigmoid(logits)` $\to$ top-22 selection on `scores + e_score_correction_bias` $\to$ normalize unbiased `scores` of the 22 winners to sum to 1 $\to$ multiply by `routed_scaling_factor = 5.0`.
       - **Latent Down-Projection (`fc1_latent_proj`):** projects `4096 -> 1024` (`4x` compression before experts!).
       - **512 Non-Gated `ReLU²` Routed Experts (`NemotronHExperts`):** each expert operates in `1024`-dim latent space (`up_proj: [2688, 1024]`, `down_proj: [1024, 2688]`):
         $$E_e(z) = W_{\text{down},e}\,\operatorname{ReLU}^2(W_{\text{up},e}\,z)$$
       - **Latent Up-Projection (`fc2_latent_proj`):** projects weighted sum of 22 experts from `1024 -> 4096`.
       - **Shared Expert (`shared_experts`):** parallel non-gated `ReLU²` MLP in full `4096` space (`up_proj: [5376, 4096]`, `down_proj: [4096, 5376]`) added to the latent-MoE output.
     - **8 NoPE 16:1 GQA Global Attention Layers (`"attention"` — Layers `7, 16, 25, 36, 47, 58, 69, 78`):**
       - `q_proj: [4096, 4096]` (`32` query heads $\times 128$ dim), `k_proj: [256, 4096]` (`2` KV heads $\times 128$ dim = **16:1 GQA**), `v_proj: [256, 4096]`, `o_proj: [4096, 4096]`.
       - **No RoPE and no QK norm:** positional ordering comes entirely from the interleaved Mamba-2 layers, giving `8 KiB/token` total KV cache across the entire 88-layer backbone and enabling 1M-token context without RoPE distortion.
5. **Stage L89 — Final Norm & Untied Output Head (`norm_f` + `lm_head`):**
   - `backbone.norm_f`: `NemotronHRMSNorm(4096, eps=1e-5)`.
   - `lm_head`: untied `Linear(4096, 131072, bias=False)` returning FP32 logits.
6. **Stage L90 — Native Multi-Token Prediction Speculative Head (`language_model.mtp`, 2 layers):**
   - Normalizes next-token embedding via `enorm(4096)` and backbone hidden state via `hnorm(4096)`, concatenates to `8192`, projects via `eh_proj: [4096, 8192]`, runs 1 `"attention"` layer (`mtp.layers.0`) + 1 `"moe"` layer (`mtp.layers.1` with 512 latent experts) + `final_layernorm(4096)`, and projects through the shared `lm_head`.

---

## Direction

1. **Download the 6 Stage-Bearing Shards of `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` on `k3`** (`model-00001`, `model-00002`, `model-00005`, `model-00061`, `model-00062`, `model-00063` + `model-summary-idxs.safetensors`, ~23.5 GB) so we have the **actual trained BF16 tensors** for every single stage of the 120B model:
   - Full C-RADIO Vision Encoder (`vision_model`, 32 layers, all 391 tensors)
   - Full Vision Projector (`vision_projector`, PixelShuffle + ReLU² MLP, all 5 tensors)
   - Full Token Embedding table (`language_model.backbone.embeddings`, `[131072, 4096]`)
   - Full Layer 0 Mamba-2 Mixer (`language_model.backbone.layers.0`, all 9 tensors)
   - Full Layer 1 Latent MoE Mixer (`language_model.backbone.layers.1`, all 1,031 tensors including all 512 routed experts + shared expert + latent projections + router)
   - Full Layer 7 16:1 GQA Attention Mixer (`language_model.backbone.layers.7`, all 5 tensors)
   - Full Final Norm & Untied LM Head (`norm_f` + `lm_head`, `[131072, 4096]`)
   - Full 2-Layer Native MTP Speculative Head (`language_model.mtp`, all 1,040 tensors)
2. **Run & Verify Every Stage on Real Text, Image, and Video Inputs (`nemotron_super_vl_stage_study.py`):**
   - Execute each stage using the official module classes (`RadioModel`, `NemotronH_Omni_Reasoning_V3VisionProjector`, `compute_retention_mask`, `NemotronHBlock` for `mamba`, `moe`, and `attention`, `norm_f`, `lm_head`, and `mtp`) and compare against our independent mathematical reconstruction from raw tensors (`max_abs_diff = 0.0`, `cos_sim = 1.0`).
   - Measure exact per-stage activation RMS, Mamba-2 SSD chunk scan vs. step-by-step recurrent state equivalence, Latent MoE router distribution (`top-22 of 512`, `e_score_correction_bias`, `routed_scaling_factor = 5.0`, latent vs. shared expert norms), 16:1 NoPE GQA attention entropy, C-RADIO + PixelShuffle + EVS token pruning, and Native MTP draft-token prediction.
   - Produce a rigorous architectural & hardware comparison of **`Nemotron 3.5 Super VL (120B-A12B)` vs. `Kimi K3 (2.78T-A104B)` vs. `LFM3-1.7B`** for Clover.

---

## Execution & Outcome Log

### 1. Exact 43,079-Tensor / 124.45B-Parameter Census (`NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`)
We downloaded the non-weight metadata, custom modeling code (`configuration_nemotron_h_omni.py`, `modeling_nemotron_h_omni.py`, `configuration_radio.py`, `modeling_radio.py`), and the 6 stage-bearing BF16 shards (`model-00001`, `00002`, `00005`, `00061`, `00062`, `00063` + `model-summary-idxs.safetensors`, `20.9 GiB` total) to `/opt/clover-k3/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` on `k3` (`37.27.114.74`, AMD Ryzen 9 7950X3D) in `189.43 s`.

Using [research/nemotron_super_vl_stage_study.py](../../../research/nemotron_super_vl_stage_study.py) and recorded in [research/nemotron_super_vl_stage_study_report.json](../../../research/nemotron_super_vl_stage_study_report.json), our stage-by-stage parameter sum across all `43,079` tensors matched the checkpoint metadata to the single parameter (**`124,453,395,976` total parameters**, **`12,770,237,440` active backbone parameters per text token**):

| Stage / Component | Tensors | Parameters | Share | Per-Layer / Active Structure |
| :--- | ---: | ---: | ---: | :--- |
| **40 Latent `ReLU²` MoE Layers (`"moe"`)** | `41,240` | `114,924,113,920` | **92.343%** | `2,873,102,848` params/layer (`2,818,572,288` in 512 routed experts + `54,530,560` in router/latent/shared; **`175,641,088` active/layer** for top-22 + shared) |
| **40 Mamba-2 SSM Layers (`"mamba"`)** | `360` | `4,385,602,560` | **3.524%** | `109,640,064` params/layer (`in_proj: [18560, 4096]`, `conv1d: [10240, 1, 4]`, `out_proj: [4096, 8192]`, `A_log/D/dt_bias: [128]`) |
| **Native 2-Layer MTP Speculative Head (`mtp`)** | `1,040` | `2,942,325,248` | **2.364%** | `enorm/hnorm: [4096]`, `eh_proj: [4096, 8192]`, `mtp.layers.0` (Attention: `35.66M`), `mtp.layers.1` (512-expert Latent MoE: `2.87B`) |
| **C-RADIO 32-Layer Vision Encoder (`vision_model`)** | `391` | `653,611,528` | **0.525%** | 32 ViT-Huge blocks (`hidden_size = 1280`, 16 heads $\times 80$, GELU MLP `1280 -> 5120 -> 1280`, CPE `pos_embed: [1, 16384, 1280]`, `10` CLS/register tokens) |
| **Token Embedding (`backbone.embeddings`)** | `1` | `536,870,912` | **0.431%** | `[131072, 4096]` (`bfloat16`, untied) |
| **Final RMSNorm + Untied Output Head (`norm_f` + `lm_head`)** | `2` | `536,875,008` | **0.431%** | `norm_f: [4096]` + `lm_head: [131072, 4096]` (`bfloat16`, untied) |
| **8 NoPE 16:1 GQA Attention Layers (`"attention"`)** | `40` | `285,245,440` | **0.229%** | `35,655,680` params/layer (`q_proj: [4096, 4096]`, `k_proj/v_proj: [256, 4096]`, `o_proj: [4096, 4096]`, **no RoPE, no QK norm**) |
| **Vision Projector (`vision_projector` / `mlp1`)** | `5` | `188,751,360` | **0.152%** | `vision_final_layernorm(1280)` + `pixel_shuffle(0.5)` + `RMSNorm(5120) -> Linear(5120, 20480) -> ReLU² -> Linear(20480, 4096)` |

---

### 2. Stage-by-Stage Mathematical Verification on Real Trained Weights (`max_abs_diff = 0.0`)

1. **Stage V1 — C-RADIO 32-Layer Vision Encoder (`RadioModel`):**
   - Verified on a $512 \times 512$ RGB image (`[1, 3, 512, 512]`): `RadioPatchEmbeddings` splits into $32 \times 32 = 1024$ patches of $768$ values, projects to `1280`, adds bilinearly interpolated CPE from the $128 \times 128$ position grid, and prepends `10` prefix tokens (`3` CLS + `7` register tokens) $\to [1, 1034, 1280]$.
   - Our manual patch-projection + bilinear CPE interpolation + `RadioLayer 0` reconstruction matched `RadioModel` with **`embed_max_abs_diff_vs_manual = 0.0`** and **`layer0_max_abs_diff_vs_manual = 0.0`**.
   - Across the 32 ViT-Huge layers (`1,257.6 ms` on 8 CPU cores), the activation RMS stays near `1.405 -> 1.305` through Layer 16 and rises to `45.449` at Layer 32 (`[1, 1024, 1280]` after stripping the 10 prefix tokens).

2. **Stage V2 — Pixel-Shuffle Vision Projector & EVS Video Token Pruning (`vision_projector`):**
   - `vision_final_layernorm` (`LayerNorm(1280, eps=1e-6)`) normalizes the hot `45.45` RMS RADIO features before `pixel_shuffle(0.5, ps_version="v2")` folds $2 \times 2$ spatial neighborhoods into `[1, 16, 16, 5120]` (`256` tokens) and `mlp1` (`RMSNorm(5120) -> Linear(5120, 20480) -> ReLU² -> Linear(20480, 4096)`) projects to `[1, 256, 4096]` (`output_llm_embeds_rms = 0.36694`, `12.33 ms`).
   - Matched manual projector equations with **`max_abs_diff_vs_manual = 0.0`**.
   - **EVS (`compute_retention_mask`) Video Verification:** On a 6-frame $512 \times 512$ video (`3` temporal groups of `2` frames $= 768$ visual tokens, pruning rate $q = 0.5 \implies 384$ tokens retained):
     - **Group 0 (Frames 0–1, anchor):** kept **`256 / 256` tokens** (EVS forces Group 0 dissimilarity to `255.0` so the initial scene is never pruned).
     - **Group 1 (Frames 2–3, static duplicate of Group 0):** kept **`0 / 256` tokens** (`1 - cos = 0.0`, 100% pruned!).
     - **Group 2 (Frames 4–5, top half moving, bottom half static):** kept **`128 / 256` tokens** (retained 100% of the 128 moving patches and pruned 100% of the 128 static patches).

3. **Stage L0 (`embeddings`) & Stage L1 (`layers.0` Mamba-2 Mixer):**
   - On the 23-token prompt `"Clover verifies every stage of NVIDIA Nemotron 3.5 Super VL on real tensors."`, `embeddings` (`[1, 23, 4096]`) has `rms = 0.01417` (`0.197 ms`).
   - In `layers.0` (`NemotronHMamba2Mixer`), our explicit step-by-step selective state-space recurrence ($s_t = \exp(\Delta_t A)\,s_{t-1} + \Delta_t B_t x_t$, $y_t = s_t C_t + D x_t$ across `128` heads $\times 64$ dim $\times 128$ state dim in `float32`, followed by `Zamba2RMSNormGated` over 8 groups of 1024 and `out_proj`) matched the official 128-token chunked SSD matrix scan (`segment_sum` + `Y_diag + Y_off`) with **`cos_sim = 0.99999756`** (`max_diff = 0.000977`), and causal prefix invariance gave **`causal_prefix_max_gap = 0.0`**.

4. **Stage L2 (`layers.1` Latent `ReLU²` MoE Mixer — 512 Routed Experts, Top-22):**
   - `NemotronHTopkRouter` projects normalized `4096`-dim hidden states to `512` logits in `float32`, takes `scores = sigmoid(logits)`, adds the learned FP32 `e_score_correction_bias` (`[512]`) to choose **22 of 512 experts per token**, normalizes the 22 unbiased `scores` to sum to `1.0`, and multiplies by **`routed_scaling_factor = 5.0`** (`sum_of_22_weights_per_token = 5.0`).
   - Across our 23-token prompt (`23 * 22 = 506` expert selections), **`309` distinct experts** of the 512 were activated.
   - `fc1_latent_proj` compresses `4096 -> 1024` (`latent_1024_in_rms = 0.41221`), the 22 non-gated `ReLU²` experts (`1024 -> 2688 -> 1024`) execute in the 4x-smaller latent space, `fc2_latent_proj` projects back `1024 -> 4096` (`routed_branch_4096_rms = 0.01366`), and `shared_experts` (`4096 -> 5376 -> 4096` `ReLU²`, `shared_expert_4096_rms = 0.00235`) adds in parallel.
   - **Floating-Point Reassociation Finding Caught & Verified:** Writing `(residual + routed_4096) + shared_4096` in BF16 produces a 1-ULP (`2^-9 = 0.001953125`) difference against `NemotronHBlock`, because `NemotronHMoE` computes `routed_4096 + shared_4096` first and `NemotronHBlock` then computes `residual + (routed_4096 + shared_4096)`. With exact parenthesization `residual + (routed_4096 + shared_4096)`, our manual reconstruction achieved **`max_abs_diff_vs_official = 0.0`**!

5. **Stage L3 (`layers.7` NoPE 16:1 GQA Global Attention Mixer):**
   - Projects `4096` to `32` query heads (`[1, 32, 23, 128]`) and only **`2` KV heads** (`[1, 2, 23, 128]`, **16:1 GQA**), with **no RoPE and no QK norm**.
   - Matched `NemotronHBlock(layer_idx=7)` with **`max_abs_diff_vs_official = 0.0`** and **`causal_prefix_max_gap = 0.0`** in `4.13 ms`.

6. **Stage L4 (`norm_f` + Untied `lm_head`) & Stage L5 (Native 2-Layer `MTP` Speculative Head):**
   - `norm_f` scales the residual stream to `rms = 2.10072`, and `lm_head` (`[131072, 4096]`) projects to `[1, 23, 131072]` FP32 logits in `34.23 ms`.
   - The 2-layer **Native MTP module** (`enorm(e_{t+1}) || hnorm(h_t) -> eh_proj(8192->4096) -> mtp.layers.0 [Attention] -> mtp.layers.1 [512-expert Latent MoE] -> final_layernorm -> shared lm_head`) executed end-to-end on real weights in **`118.45 ms`** (`mtp_final_norm_rms = 2.02448`), drafting the next token in **2 layers instead of 88 layers** (~44x fewer layers per speculative draft step!).

---

### 3. Three-Way Architectural & Systems Comparison for Clover: `Kimi K3` vs. `Nemotron 3.5 Super VL` vs. `LFM3-1.7B`

| Property | `Kimi K3` (`clover-kimi-k3`) | `Nemotron 3.5 Super VL` (`120B-A12B`) | `LFM3-1.7B` (`LiquidAI/LFM3-1.7B`) |
| :--- | :--- | :--- | :--- |
| **Total / Active Parameters** | **2.78T total / 104B active** | **124.45B total / 12.77B active** | **1.73B total / 1.73B active** |
| **Checkpoint Disk Size** | `1,400 GiB` (INT8 trunk + MXFP4 experts) | **`231.8 GiB` (BF16) / `69.9 GiB` (NVFP4)** | **`3.24 GiB` (BF16) / `~1.0 GiB` (Q4_0)** |
| **Layer Schedule** | 93 layers: `KKKM` $\times 23 + M$ (`68 KDA + 24 MLA`, every layer has Attention + MoE) | **88 single-mixer layers:** `40 Mamba-2 + 40 Latent MoE + 8 NoPE GQA` (`MEMEMEMAE...`) | 24 layers: `13 ShortConv + 5 Sliding GQA + 6 Full GQA` (each has Op + SwiGLU) |
| **MoE Design** | 92 MoE layers, `896` experts/layer (top-16), `7168->3584` latent, gated `SiTU` (`17.5 MB`/expert) | 40 MoE layers, `512` experts/layer (top-22), `4096->1024` latent, **non-gated `ReLU²`** (`10.5 MB` BF16 / `2.63 MB` NVFP4 per expert) | Dense SwiGLU (`2048 -> 7680 -> 2048`) |
| **Expert Bytes Read / Token** | `~25.8 GB/token` (MXFP4) + `53.8 GB` trunk | **`~9.68 GB/token` (BF16) / `~2.42 GB/token` (NVFP4)** across all 40 MoE layers! | `~3.46 GB/token` (BF16) / `~0.86 GB/token` (Q4_0) |
| **State Memory at 256K Context** | Large (24 MLA KV + 68 KDA `96x128x128` states) | **`2.16 GiB` total** (`162.3 MiB` fixed for 40 Mamba-2 layers + `2.00 GiB` for 8 GQA-2 layers; **40.8x smaller** than 88-layer GQA) | **`3.00 GiB` total** (`312 KB` fixed for 13 conv + `2.5 MB` for 5 sliding + `3.0 GiB` for 6 full GQA) |
| **Max Context Length** | `256K` tokens | **`1,048,576` (1M tokens — `8.16 GiB` total state at 1M!)** | `262,144` (256K tokens) |
| **Multimodal & MTP** | Text path only in C engine; **no MTP weights released** in open checkpoint | **Native C-RADIO Vision & Video + EVS Pruning + Released 2-Layer MTP Weights** | Text-only, no MTP |
| **Fits Entirely in `k3`'s 124 GiB RAM?** | **No** (needs NVMe streaming of `1.4 TB` experts; `~3.0–5.7 s/tok` on CPU) | **YES in NVFP4 (`69.9 GiB` entire model fits in 124 GiB RAM with zero disk I/O!)** | **YES (`3.3 GiB` in BF16, `9.68 tok/s` on 1 CCD)** |

### 4. Key Takeaway for Clover's Engine & Model Roadmap
1. **Why `Nemotron 3.5 Super VL` is a Breakthrough Middle/Top Tier for Workstation & Multi-GPU Hosting (`k3` and `gpu`):**
   - In `clover-kimi-k3`, the fundamental hardware wall on a `124 GiB` workstation (`k3`) or `8x RTX 3090` (`192 GiB` total VRAM on `gpu`) is that `Kimi K3`'s `1.4 TB` of routed experts cannot fit in RAM/VRAM, forcing NVMe streaming (`~25.8 GB/token` of expert reads + `53.8 GB` trunk).
   - By contrast, **`NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` is only `69.9 GiB` in total** (and reads only **`~2.42 GB/token`** of active weights in 4-bit / FP8!). That means **the entire 120B model (all 512 experts per layer, all 40 Mamba-2 layers, all 8 GQA layers, the C-RADIO vision tower, and the 2-layer Native MTP speculative drafter) fits 100% resident inside `k3`'s `124 GiB` RAM or across 4x RTX 3090s (`96 GiB` VRAM) with ZERO disk I/O!**
2. **Even Smaller Long-Context State than `LFM3-1.7B` at 256K (`2.16 GiB` vs. `3.00 GiB`):**
   - Because only **8 of the 88 layers** use attention and each uses **16:1 GQA (only 2 KV heads)** with **NoPE**, `Nemotron 3.5 Super VL`'s KV cache is just `8 KiB/token` (`2.00 GiB` at `256K` tokens, plus `162.34 MiB` fixed for all 40 Mamba-2 layers).
3. **Released Native MTP Weights (`language_model.mtp`):**
   - Unlike `Kimi K3`'s open checkpoint (`num_nextn_predict_layers: 0`), `Nemotron 3.5 Super VL` ships the complete trained 2-layer MTP weights (`1,040` tensors, `2.94B` total / `~211M` active parameters), enabling 3–5 token speculative drafting in 2 layers instead of 88.

