# Clover AI System (`clover-ai-system/`) — Switchable 7-Layer Architecture & Precomputed Model Engines Runtime Context

## Context

### 1. Human Direction
1. **Rename parent directory `clover-kimi-k3/` $\to$ `clover-ai-system/`** and organize model-specific code inside `clover-ai-system/src/models/{kimi-k3,nemotron-3.5-super-vl,lfm3-1.7b}/`.
2. **Make Layer 1 (the Foundation Model Backbone) of Clover's 7-Layer Architecture switchable** between:
   - `kimi-k3` (`MoonshotAI/Kimi-K3` — 93-layer 2.78T total / 104B active MoE)
   - `nemotron-3.5-super-vl` (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B` — 88-layer 124.45B total / 12.77B active Mamba-2 + Latent `ReLU²` MoE + NoPE 16:1 GQA + C-RADIO Vision + Native 2-layer MTP)
   - `lfm3-1.7b` (`LiquidAI/LFM3-1.7B` — 24-layer 1.73B dense Tri-Hybrid ShortConv + Sliding GQA + Full GQA)
   while keeping Layers 2–7 of Clover (`HybridAgent` continuous `<latent>` reasoning & live tools, `NCM` 11-pass Next-Concept Modeling, `CloverMemoryGraphStore`, `CloverOntologyRetriever` + `CloverSymbolicVerifier`, `CloverPython/RustGenerativeCompiler`, and `SpecMD Least-Stale + R4 Clean-Drop` 4-tier data-movement & slot caching) shared and compatible across all three models.
3. **Build each model's precomputed engine one after the other, starting with `nemotron-3.5-super-vl` first**, applying the exact same precomputed trunk/operator/expert optimizations and pruning discipline that we built for `Kimi K3`.

---

### 2. How Kimi K3's Precomputed Engine Works (Our Reference Pattern)
In `clover-ai-system/src/code/cpu/` and `clover-ai-system/src/tools/prune-trunk.py`:
- **Prepared Trunks (`trunk-0..92`, `K3TRK001`):** Each layer's non-routed parameters are split into `index.bin`, `common.bin` (group 0: norms, output projection), `attention.bin` (group 1: KDA/MLA projections, conv1d, `A_log`, `dt_bias`), and `routed-shared.bin` (group 2: router `gate.weight`, `e_score_correction_bias`, latent projections `fc1/fc2`, and `shared_experts`).
- **Composed Operator (`operator.bin`):** Fuses the input normalization and attention/mixer projection weights (`Q/K/V` or Mamba `in_proj + conv1d`) into a single contiguous mmap-ready binary file (`operator.bin`) read in one stream.
- **Trunk Pruning (`prune-trunk.py`):** Drops any record from `common.bin` / `attention.bin` / `routed-shared.bin` that is already served by `operator.bin` or unused at inference, setting its length to `0` in `index.bin` and deleting empty group files so locked RAM drops by ~35%.
- **Precomputed Direct Experts (`expert-N/`, `experts.direct` + `constants.bin` + `maps.json`):** Each layer's routed experts are stored in a contiguous, pre-resolved direct binary container (`experts.direct` + BF16/FP4 lookup tables) so only the top-$k$ routed experts selected by the router are fetched/projected per token, managed by `SpecMD Policy 3 (Least-Stale + R4 Clean-Drop)` slot caching (`libslotcache.so` / `class-memory-storage-tier`).
- **Adapter Bridge into the 7-Layer Stack (`K3ExpertBridge`):** Maps the model's backbone hidden dimension (`D_model`: `7168` for Kimi K3, `4096` for Nemotron 3.5 Super VL, `2048` for LFM3-1.7B) into Clover's shared `384`-d NCM concept space and `4096`-d `HybridLink` latent reasoning space.

---

### 3. Designing `nemotron-3.5-super-vl` Precomputed Trunks, Operators, Experts & Pruning (`clover-ai-system/src/models/nemotron-3.5-super-vl/`)

#### A. Exact Layer & Container Map for `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B`
`Nemotron 3.5 Super VL` has **88 single-mixer backbone layers** (`layers.0..87`), **1 Vision Tower + Projector** (`vision-0`), **1 Global Embedding + Final Norm + LM Head** (`global-0`), and **2 Native MTP Speculative Layers** (`mtp-0`, `mtp-1`):

1. **40 Mamba-2 Trunk + Operator Folders (`trunk-L` for $L \in \{0, 2, 4, 6, 9, \dots, 86\}$ — `NEMO_TRK1`, `kind = "mamba"`):**
   - `index.bin` (`NEMO_TRK1` header + 9 record descriptors):
     - Group 0 (`common.bin`):
       - Record `0`: `norm.weight` (`[4096]`, BF16) — pre-mixer RMSNorm
       - Record `1`: `mixer.norm.weight` (`[8192]`, BF16) — `Zamba2RMSNormGated` weight
       - Record `2`: `mixer.out_proj.weight` (`[4096, 8192]`, BF16 / INT8 / 4-bit) — output projection
     - Group 1 (`attention.bin` — Mamba-2 sequence mixer parameters):
       - Record `3`: `mixer.in_proj.weight` (`[18560, 4096]`)
       - Record `4`: `mixer.conv1d.weight` (`[10240, 1, 4]`)
       - Record `5`: `mixer.conv1d.bias` (`[10240]`)
       - Record `6`: `mixer.dt_bias` (`[128]`)
       - Record `7`: `mixer.A_log` (`[128]`) and precomputed `A = -exp(A_log.float())` (`[128]`, FP32)
       - Record `8`: `mixer.D` (`[128]`)
   - **Composed Mamba-2 Operator (`operator.bin`, `NEMO_OP01`):**
     - Packs `norm.weight` (`[4096]`), `in_proj.weight` (`[18560, 4096]`), `conv1d.weight` (`[10240, 4]`), `conv1d.bias` (`[10240]`), `dt_bias` (`[128]`), `A_precomputed = -exp(A_log)` (`[128]`, FP32 — eliminating `exp(A_log)` at runtime just as Kimi K3's `trunk-0` precomputes `exp(ALOG)`!), and `D` (`[128]`).
   - **Trunk Pruning (`prune-nemotron-trunk.py`):**
     - Once `operator.bin` is built for a Mamba-2 layer, Records `0, 3, 4, 5, 6, 7, 8` are served directly from `operator.bin`. `prune-nemotron-trunk.py` drops `attention.bin` (`152.15 MB` per layer) completely and leaves only `common.bin` Records `1` (`mixer.norm.weight`) and `2` (`mixer.out_proj.weight`, `67.12 MB`), cutting Mamba-2 trunk storage by **69.4%** (`219.28 MB` $\to$ `67.13 MB` per layer)!

2. **8 NoPE 16:1 GQA Attention Trunk + Operator Folders (`trunk-L` for $L \in \{7, 16, 25, 36, 47, 58, 69, 78\}$ — `NEMO_TRK1`, `kind = "attention"`):**
   - `index.bin` (5 record descriptors):
     - Group 0 (`common.bin`):
       - Record `0`: `norm.weight` (`[4096]`, BF16)
       - Record `1`: `mixer.o_proj.weight` (`[4096, 4096]`, BF16)
     - Group 1 (`attention.bin`):
       - Record `2`: `mixer.q_proj.weight` (`[4096, 4096]`, BF16)
       - Record `3`: `mixer.k_proj.weight` (`[256, 4096]`, BF16)
       - Record `4`: `mixer.v_proj.weight` (`[256, 4096]`, BF16)
   - **Composed GQA Operator (`operator.bin`, `NEMO_OP01`):**
     - Fuses `norm.weight` + interleaved `q_proj` (`4096`), `k_proj` (`256`), `v_proj` (`256`) into a single `[4608, 4096]` QKV operator (`37.76 MB`).
   - **Trunk Pruning (`prune-nemotron-trunk.py`):**
     - Drops `attention.bin` (`37.75 MB`) and Record `0`, keeping only `o_proj.weight` (`33.55 MB`) in `common.bin` (**52.9% trunk reduction**).

3. **40 Latent `ReLU²` MoE Trunk + Direct Expert Folders (`trunk-L` + `expert-L` for $L \in \{1, 3, 5, 8, \dots, 87\}$):**
   - **MoE Trunk (`trunk-L`, `NEMO_TRK1`, `kind = "moe"`, `109.06 MB`):**
     - Group 0 (`common.bin`):
       - Record `0`: `norm.weight` (`[4096]`, BF16)
     - Group 2 (`routed-shared.bin`):
       - Record `29`: `mixer.gate.weight` (`[512, 4096]`, FP32 pre-widened for exact FP32 router dot-product!)
       - Record `30`: `mixer.gate.e_score_correction_bias` (`[512]`, FP32)
       - Record `31`: `mixer.fc1_latent_proj.weight` (`[1024, 4096]`, BF16)
       - Record `32`: `mixer.fc2_latent_proj.weight` (`[4096, 1024]`, BF16)
       - Record `34`: `mixer.shared_experts.up_proj.weight` (`[5376, 4096]`, BF16)
       - Record `36`: `mixer.shared_experts.down_proj.weight` (`[4096, 5376]`, BF16)
   - **Precomputed Direct Expert Store (`expert-L/experts.direct` + `constants.bin` + `maps.json`):**
     - Stores all **512 routed non-gated `ReLU²` experts** in a single contiguous `mmap` / `pread`-addressable binary `experts.direct` (`512` experts $\times 11,010,048$ bytes/expert in BF16 = `5,637,144,576` bytes, or `2,752,512` bytes/expert in 4-bit = `1.41 GB`/layer) with `maps.json` and `constants.bin` (the layer's distinct BF16 exponent/mantissa palette and CRC32 verification table).
     - Each expert $e \in \{0..511\}$ sits at exact byte offset `e * EXPERT_STRIDE` with `up_proj: [2688, 1024]` followed immediately by `down_proj: [1024, 2688]`, so a single `pread` or `mmap` slice fetches the entire expert and executes:
       $$E_e(z) = W_{\text{down},e}\,\operatorname{ReLU}^2(W_{\text{up},e}\,z), \quad z \in \mathbb{R}^{1024}$$
     - Integrates directly with Clover's `SpecMD Least-Stale + R4 Clean-Drop` (`libslotcache.so` / `class-memory-storage-tier`) and `N-1` cross-layer speculative lookahead!

4. **Native 2-Layer MTP Speculative Drafter (`mtp-0` Attention + `mtp-1` Latent MoE):**
   - Replaces Kimi K3's external 5-layer `DSpark` drafter with Nemotron 3.5 Super VL's **native 2-layer MTP head** (`enorm`, `hnorm`, `eh_proj`, `mtp.layers.0` GQA-2 Attention, `mtp.layers.1` 512-expert Latent MoE, `final_layernorm`), using the exact same `trunk-mtp-1` + `expert-mtp-1/experts.direct` format.

5. **C-RADIO Vision + PixelShuffle + ReLU² Projector (`vision-0`):**
   - Replaces Kimi K3's 27-layer `MoonViT3d + PatchMergerMLPV2` (`7168`-d) with Nemotron's 32-layer `C-RADIO + PixelShuffle(0.5) + ReLU² mlp1 + EVS video pruning` (`4096`-d).

---

## Direction

1. Complete the rename of `clover-kimi-k3/` $\to$ `clover-ai-system/` and update all repository links and tests.
2. Create the model-specific directory structure under `clover-ai-system/src/models/`:
   - `clover-ai-system/src/models/nemotron-3.5-super-vl/` (built & verified first in this cycle)
   - `clover-ai-system/src/models/kimi-k3/` (existing Kimi K3 engine & bridges)
   - `clover-ai-system/src/models/lfm3-1.7b/` (LFM3-1.7B tri-hybrid adapter & stage engine, ready for the next cycle)
3. Implement and run on `k3` (`37.27.114.74`) the complete **Nemotron 3.5 Super VL Precomputed Engine & Optimizer Suite**:
   - `prepare_nemotron_datasets.py`: Builds `trunk-0` (Mamba-2), `trunk-1` + `expert-1/experts.direct` (Latent `ReLU²` MoE), `trunk-7` (NoPE 16:1 GQA), `trunk-mtp-0`, `trunk-mtp-1` + `expert-mtp-1/experts.direct`, `vision-0`, and `global-0` from the downloaded BF16 shards on `k3`.
   - `prune_nemotron_trunk.py`: Prunes redundant records covered by `operator.bin` from Mamba-2 and GQA trunks, verifying exact byte savings and zero loss of needed records.
   - `nemotron_precomputed_engine.py`: Executes every stage (`Stage V1 C-RADIO` $\to$ `Stage V2 PixelShuffle+ReLU²+EVS` $\to$ `Stage L0 Embedding` $\to$ `Stage L1 Mamba-2 from operator.bin + pruned trunk-0` $\to$ `Stage L2 Latent MoE from trunk-1 + expert-1/experts.direct + SpecMD Least-Stale R4 SlotCache + 4-bit trunk/expert quantization` $\to$ `Stage L3 NoPE GQA from operator.bin + pruned trunk-7` $\to$ `Stage L4 Final Norm + Untied LM Head` $\to$ `Stage L5 Native MTP Speculative Draft`) and verifies **`max_abs_diff = 0.0`** against the unpruned Hugging Face reference!
4. Wire the **Switchable Layer-1 Model Backend (`CLOVER_MODEL_REGISTRY` / `CloverSwitchableModelLayer1` / `class-switchable-foundation-model-backend`)** into [clover-object-model.py](../../../clover-ai-system/src/tools/hybrid-ncm-k3/clover-object-model.py) and [switchable_model_backend.py](../../../clover-ai-system/src/models/switchable_model_backend.py) so Clover's 7-layer stack (`HybridAgent` + `NCM` + `Memory Graph` + `Ontology GraphRAG` + `Generative Compiler` + `SpecMD R4 SlotCache`) can hot-switch Layer 1 between `"nemotron-3.5-super-vl"`, `"kimi-k3"`, and `"lfm3-1.7b"`.

---

## Execution & Outcome (Measured on `k3` — `37.27.114.74`)

### 1. Repository Rename & Structure (`clover-kimi-k3/` $\to$ `clover-ai-system/`)
- Renamed `clover-kimi-k3/` to `clover-ai-system/` via `git mv` (`121` tracked files moved cleanly with full git history preserved).
- Updated all internal links across `241` Markdown files (`scripts/check-links.py`: **0 broken links**) and updated all test imports.
- Created model-specific directories under [clover-ai-system/src/models/](../../../clover-ai-system/src/models/):
  - [clover-ai-system/src/models/nemotron-3.5-super-vl/](../../../clover-ai-system/src/models/nemotron-3.5-super-vl/) ([README.md](../../../clover-ai-system/src/models/nemotron-3.5-super-vl/README.md), [prepare_nemotron_datasets.py](../../../clover-ai-system/src/models/nemotron-3.5-super-vl/prepare_nemotron_datasets.py), [prune_nemotron_trunk.py](../../../clover-ai-system/src/models/nemotron-3.5-super-vl/prune_nemotron_trunk.py), [nemotron_precomputed_engine.py](../../../clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_precomputed_engine.py), [verify_nemotron_clover_integration.py](../../../clover-ai-system/src/models/nemotron-3.5-super-vl/verify_nemotron_clover_integration.py))
  - [clover-ai-system/src/models/kimi-k3/](../../../clover-ai-system/src/models/kimi-k3/) ([README.md](../../../clover-ai-system/src/models/kimi-k3/README.md))
  - [clover-ai-system/src/models/lfm3-1.7b/](../../../clover-ai-system/src/models/lfm3-1.7b/) ([README.md](../../../clover-ai-system/src/models/lfm3-1.7b/README.md))
  - [clover-ai-system/src/models/switchable_model_backend.py](../../../clover-ai-system/src/models/switchable_model_backend.py)

---

### 2. Nemotron 3.5 Super VL Precomputed Trunk, Composed Operator, Trunk Pruning & Direct Expert Verification on `k3`
Executed [verify_nemotron_clover_integration.py](../../../clover-ai-system/src/models/nemotron-3.5-super-vl/verify_nemotron_clover_integration.py) on `k3` (`AMD Ryzen 9 7950X3D`, `124 GiB` RAM) against the real `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` shards (`/opt/clover-k3/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`), producing [research/nemotron_precomputed_verification_report.json](../../../research/nemotron_precomputed_verification_report.json):

| Container / Stage | Raw Precomputed Bytes | Pruned Trunk Bytes (`common.bin` + `routed-shared.bin`) | Composed `operator.bin` / `experts.direct` | Trunk Reduction % | Max Abs Diff vs Official HF `NemotronHBlock` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`trunk-0` (Mamba-2, $L=0$)** | `219,280,128 B` (`209.12 MiB`) | **`67,125,248 B` (`64.02 MiB`)** (`attention.bin` deleted!) | `operator.bin`: `152,155,200 B` (fuses `norm`, `in_proj`, `conv1d`, `dt_bias`, precomputed FP32 `A = -exp(A_log)`, `D`) | **`-69.39%`** | **`0.0` (exact bit-identical!)** |
| **`trunk-1` + `expert-1` (Latent `ReLU²` MoE, $L=1$)** | `113,256,448 B` (`108.01 MiB`) | **`113,256,448 B` (`108.01 MiB`)** (pre-widened FP32 `gate.weight` + `e_score_correction_bias`, `fc1/fc2`, `shared_experts`) | `experts.direct`: `5,637,144,576 B` (`512` experts $\times$ `11,010,048 B`/expert) | `0.0%` (all trunk records active) | **`0.0` (exact bit-identical!)** (`4-bit group-64` $\cos = 0.998189$) |
| **`trunk-7` (NoPE 16:1 GQA, $L=7$)** | `71,311,360 B` (`68.01 MiB`) | **`33,554,432 B` (`32.00 MiB`)** (`attention.bin` deleted!) | `operator.bin`: `37,756,992 B` (fuses `norm` + `q_proj`, `k_proj`, `v_proj`) | **`-52.95%`** | **`0.0` (exact bit-identical!)** |
| **`trunk-mtp-0` (Native MTP Layer 0 GQA)** | `138,436,608 B` (`132.02 MiB`) | **`100,679,680 B` (`96.02 MiB`)** (`attention.bin` deleted!) | `operator.bin`: `37,756,992 B` | **`-27.27%`** | **`0.0` (exact bit-identical!)** |
| **`trunk-mtp-1` + `expert-mtp-1` (Native MTP Layer 1 MoE)** | `113,264,640 B` (`108.02 MiB`) | **`113,264,640 B` (`108.02 MiB`)** | `experts.direct`: `5,637,144,576 B` (`512` experts) | `0.0%` | **`0.0` (exact bit-identical!)** |
| **Full 88-Layer Backbone Projection (`40 Mamba + 40 MoE + 8 GQA`)** | `12.919 GiB` | **`6.970 GiB`** (`5.950 GiB` dropped from trunks!) | `operator.bin` streams + on-demand `experts.direct` | **`-46.05%` across all 88 trunks** | **`0.0` across all stages** |

---

### 3. SpecMD Policy 3 (`Least-Stale + R4 Clean-Drop`) Direct Expert Caching on Real Nemotron Routed Experts
- On the 31-token verification prompt (`"Clover AI System switches seamlessly between Kimi K3, NVIDIA Nemotron 3.5 Super VL, and Liquid LFM3."`), the top-22 router in Layer 1 (`expert-1/experts.direct`) selected **355 distinct experts out of 512**:
  - **Cold Pass 1:** `355` experts fetched on-demand via `mmap` slice (`3,727.5 MB` read vs `5,376.0 MB` full layer = **`30.66%` immediate I/O reduction** on a 31-token prefill, and **`95.70%` I/O reduction** per single-token decode step where only `22 / 512` experts = `242.2 MB` are touched).
  - **Warm Pass 2:** `64` Tier-1 Pool Cache hits + `291` Tier-2 Pinned Host Cache hits (`0` new NVMe drive fetches on repeated/shared experts!), `582` **Red-Blue Pebble Rule R4 Clean-Drops**, and **`0` dirty writebacks**.
- On Native MTP Layer 1 (`expert-mtp-1/experts.direct`), only **219 distinct experts out of 512** (`2,299.5 MB` vs `5,376.0 MB`) were touched (**`57.23%` I/O reduction**).

---

### 4. Switchable Layer-1 Model Backend Across `nemotron-3.5-super-vl`, `kimi-k3`, and `lfm3-1.7b`
- Verified both in [switchable_model_backend.py](../../../clover-ai-system/src/models/switchable_model_backend.py) (PyTorch `CloverSwitchableModelLayer1` projecting real `[1, 31, D_model]` hidden states through `CloverUniversalConceptBridge(D_model -> 384)` and `CloverUniversalHybridLink(D_model -> 4096)`) and in [clover-object-model.py](../../../clover-ai-system/src/tools/hybrid-ncm-k3/clover-object-model.py) (`test_06_switchable_foundation_model_backends_across_kimi_nemotron_and_lfm3` in [test-clover-object-model.py](../../../scripts/tests/test-clover-object-model.py) — all `26/26` unit tests passing):
  - **`nemotron-3.5-super-vl` (`D_model = 4096`, 88 layers):** `backbone_shape = [1, 31, 4096]` $\to$ `ncm_concept_384_shape = [1, 31, 384]` ($\text{RMS} = 0.46204$) and `hybrid_agent_4096_shape = [1, 31, 4096]` ($\text{RMS} = 2.10103$, **1:1 identity dimension match** with `HybridAgent`!).
  - **`kimi-k3` (`D_model = 7168`, 93 layers):** `backbone_shape = [1, 31, 7168]` $\to$ `ncm_concept_384_shape = [1, 31, 384]` ($\text{RMS} = 0.46229$) and `hybrid_agent_4096_shape = [1, 31, 4096]` ($\text{RMS} = 0.57593$).
  - **`lfm3-1.7b` (`D_model = 2048`, 24 layers):** `backbone_shape = [1, 31, 2048]` $\to$ `ncm_concept_384_shape = [1, 31, 384]` ($\text{RMS} = 0.46194$) and `hybrid_agent_4096_shape = [1, 31, 4096]` ($\text{RMS} = 0.57910$).

---

### 5. Liquid AI `LFM3-1.7B` Complete 24-Layer Precomputed Trunk (`LFM3TRK1`), Composed Operator (`LFM3_OP1`), Trunk Pruning & Verification on `k3`
Following the completion of `nemotron-3.5-super-vl`, we built and verified the complete 24-layer precomputed engine for **`lfm3-1.7b`** (`LiquidAI/LFM3-1.7B`, `D_model = 2048`, `V = 128,000` tied) inside [clover-ai-system/src/models/lfm3-1.7b/](../../../clover-ai-system/src/models/lfm3-1.7b/):
- [prepare_lfm3_datasets.py](../../../clover-ai-system/src/models/lfm3-1.7b/prepare_lfm3_datasets.py) — Builds `global-0/` (`embed_tokens.bf16.bin` serving both input embeddings and tied `lm_head`, `embedding_norm.bf16.bin`, and `rope_inv_freq.bin` precomputing zero-tail partial RoPE `0.5` for `full_attention` and full RoPE `1.0` for `sliding_attention`) plus all 24 `LFM3TRK1` trunks (`trunk-0..23`) and composed `operator.bin` streams (`LFM3_OP1`) in **`6.737 s`** on `k3`.
- [prune_lfm3_trunk.py](../../../clover-ai-system/src/models/lfm3-1.7b/prune_lfm3_trunk.py) — Drops all records covered by `operator.bin` (`operator_norm`, `conv.in_proj`, `conv.conv` in the 13 ShortConv layers; `operator_norm`, `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `q_norm`, `k_norm` in the 11 Sliding/Full GQA layers) and **deletes `attention.bin` from all `24 / 24` trunks** (`0` `attention.bin` files remaining!).
- [lfm3_precomputed_engine.py](../../../clover-ai-system/src/models/lfm3-1.7b/lfm3_precomputed_engine.py) & [verify_lfm3_clover_integration.py](../../../clover-ai-system/src/models/lfm3-1.7b/verify_lfm3_clover_integration.py) — Executes all 24 layers from the **pruned** `trunk-0..23` + `operator.bin` + `global-0/` and compares every layer and final logits against `Lfm3ForCausalLM` ([research/lfm3_precomputed_verification_report.json](../../../research/lfm3_precomputed_verification_report.json)):

| LFM3-1.7B Precomputed Container | Count | Raw Trunk Bytes | Pruned Trunk Bytes (`common.bin` + `ffn.bin`) | Composed `operator.bin` | Trunk Reduction % | Max Abs Diff vs Official `Lfm3ForCausalLM` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gated ShortConv Trunks (`kind = 0`, $K=7$)** | `13` layers (`0,1,3,5..23`) | `127,971,328 B` (`122.04 MiB`)/layer | **`102,772,736 B` (`98.01 MiB`)**/layer (`attention.bin` deleted!) | `25,198,656 B`/layer | **`-19.69%`** | **`0.0` across all 13 ShortConv layers!** |
| **Sliding GQA Trunks (`kind = 1`, $W=256$)** | `5` layers (`4,8,12,16,20`) | `115,425,792 B` (`110.08 MiB`)/layer | **`102,772,736 B` (`98.01 MiB`)**/layer (`attention.bin` deleted!) | `12,653,120 B`/layer | **`-10.96%`** | **`0.0` across all 5 Sliding GQA layers!** |
| **Full GQA Trunks (`kind = 2`, zero-tail RoPE `0.5`)** | `6` layers (`2,6,10,14,18,22`) | `115,425,792 B` (`110.08 MiB`)/layer | **`102,772,736 B` (`98.01 MiB`)**/layer (`attention.bin` deleted!) | `12,653,120 B`/layer | **`-10.96%`** | **`0.0` across all 6 Full GQA layers!** |
| **All 24 Pruned Trunks (`trunk-0..23`)** | **`24` layers** | `2,933,310,976 B` (`2.732 GiB`) | **`2,466,545,664 B` (`2.297 GiB`)** | **`-445.14 MiB` (`466,765,312 B`) dropped!** | **`-15.91%`** | **`0.0` at every single layer `0..23` & final `128,000`-vocab tied `lm_head` logits!** (`4-bit group-64` SwiGLU $\cos = 0.993288$) |

---

### 6. Native C/OpenMP Engines (`libnemotron_cpu_engine.so` & `liblfm3_cpu_engine.so`), Native `HybridAgent` (`HybridLink`) & Native 11-Level `NCM` on `k3`
To match `Kimi K3`'s native C/OpenMP execution engine (`libk3_cpu_experts.so`), native `K3NativeHybridLink(d=7168)`, and native `163,840`-vocab grafted 11-level `VQVAETokenizer` + `NextConceptTransformer`, we built, compiled, trained, and verified the complete native C + `HybridAgent` + `NCM` stacks for both **`nemotron-3.5-super-vl`** ([nemotron_cpu_engine.c](../../../clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_cpu_engine.c), [nemotron_hybrid_ncm.py](../../../clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_hybrid_ncm.py), [research/nemotron_native_c_hybrid_ncm_report.json](../../../research/nemotron_native_c_hybrid_ncm_report.json)) and **`lfm3-1.7b`** ([lfm3_cpu_engine.c](../../../clover-ai-system/src/models/lfm3-1.7b/lfm3_cpu_engine.c), [lfm3_hybrid_ncm.py](../../../clover-ai-system/src/models/lfm3-1.7b/lfm3_hybrid_ncm.py), [research/lfm3_native_c_hybrid_ncm_report.json](../../../research/lfm3_native_c_hybrid_ncm_report.json)) on `k3`:

| Layer / Metric | **`nemotron-3.5-super-vl`** (`D=4096`, `V=131,072`) | **`lfm3-1.7b`** (`D=2048`, `V=128,000` tied) | **`kimi-k3`** (`D=7168`, `V=163,840`) |
| :--- | :--- | :--- | :--- |
| **Layer 1: Native C/OpenMP Shared Library** | `libnemotron_cpu_engine.so` ([nemotron_cpu_engine.c](../../../clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_cpu_engine.c)) | `liblfm3_cpu_engine.so` ([lfm3_cpu_engine.c](../../../clover-ai-system/src/models/lfm3-1.7b/lfm3_cpu_engine.c), all 24 layers) | `libk3_cpu_experts.so` ([k3_cpu_experts_lib.c](../../../clover-ai-system/src/tools/hybrid-ncm-k3/k3_cpu_experts_lib.c), 93 layers) |
| **C/OpenMP vs Python Precomputed Cosine Similarity** | **`0.999974`** (`819.05 ms` cold / `200.97 ms` warm; `356` drive fetches pass 1 $\to$ **`0` new drive fetches pass 2**, `413` R4 clean-drops, `0` dirty writebacks) | **`0.9992055`** across all 24 layers (winning token `358` reaches exact BF16 max logit **`17.125 == 17.125`**) | Verified across 93 layers + `libslotcache.so` |
| **Layer 2: Native `HybridLink` (`HybridSFT` $h_t \to e(x_{t+1})$)** | `NemotronNativeHybridLink(d=4096, bottleneck=1024)`: raw $\cos = 0.0246 \to$ **trained test $\cos = 0.9985$**; `131K` vocab `<latent>` readout $\cos = \mathbf{0.9401}$ | `Lfm3NativeHybridLink(d=2048, bottleneck=512)`: raw $\cos = 0.2112 \to$ **trained test $\cos = 0.9994$**; `128K` vocab `<latent>` readout $\cos = \mathbf{0.9998}$ | `K3NativeHybridLink(d=7168, bottleneck=2048)` |
| **Layer 3: Shared Tokens Matched (`Qwen3/WMT14` $\leftrightarrow$ Model Vocab)** | **`67,869` exact shared tokens** (`131,072` Tekken vocab) | **`64,835` exact shared tokens** (`128,000` LFM3 vocab) | **`95,367` exact shared tokens** (`163,840` tiktoken vocab) |
| **Layer 3: Native 11-Level `VQ-VAE` Exact Token Reconstruction** | `12.15%` before $\to$ **`99.83%` (`0.9983`) after fine-tune** (reconstructs `"Clover AI System switches seamlessly between Kimi K3, NVIDIA Nemotron 3.5 Super VL, and Liquid LFM3."` **100% character-exact!**) | `83.68%` before $\to$ **`100.00%` (`1.0000`) after fine-tune** (reconstructs prompt **100% character-exact!**) | Verified (`e6_k3_native`) |
| **Layer 3: 11-Level `NextConceptTransformer` Conditioned Forward** | `[1, 313, 4096]` across 11 levels `[1,4,8,14,20,26,32,40,48,56,64]` (`CE = 5.2436`) | `[1, 313, 4096]` across 11 levels `[1,4,8,14,20,26,32,40,48,56,64]` (`CE = 4.8330`) | `[1, 313, 4096]` across 11 levels |

---

### 7. Apples-to-Apples CPU Benchmark (`k3`, Ryzen 9 7950X3D) & 70-Task Standardized Suite Across All 3 Models (`benchmark_3_models_70_tasks.py`)
After upgrading `linear_bf16_omp` in [nemotron_cpu_engine.c](../../../clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_cpu_engine.c) and [lfm3_cpu_engine.c](../../../clover-ai-system/src/models/lfm3-1.7b/lfm3_cpu_engine.c) with **8-token register blocking** (streaming each weight row `w_row` once per 8 tokens) and pinning `LFM3-1.7B` to 8 threads (1 physical CCD on Ryzen 9 7950X3D), we compared all three models on the **exact same `k3` CPU server (`AMD Ryzen 9 7950X3D`)** against Kimi K3's verified CPU baseline ([benchmarks/verified-runs/run-baseline-before-foundations/cpu/run_cpu_v4_verified/summary.md](../../../benchmarks/verified-runs/run-baseline-before-foundations/cpu/run_cpu_v4_verified/summary.md)) as well as across the full 70-task suite ([research/clover_3_models_70_task_benchmark_report.json](../../../research/clover_3_models_70_task_benchmark_report.json)):

#### A. Apples-to-Apples on the 7-Category CPU Benchmark (`k3`, `AMD Ryzen 9 7950X3D`)

| Representative Task (`run_cpu_v4_verified`) | **`kimi-k3` on CPU** (`k3_moe_plus_ncm_ms` / Total Wall) | **`nemotron-3.5-super-vl` on CPU** (`c_engine_ms` / `ncm_plus_c_ms`) | **`lfm3-1.7b` (All 24 Layers) on CPU** (`c_engine_ms` / `ncm_plus_c_ms`) |
| :--- | :---: | :---: | :---: |
| **`mmlu_college_physics_1` (Cat 1 MMLU)** | `4,777.5 ms` (`59.85 s` wall) | **`2,211.9 ms` C / `3,715.6 ms` total** | `4,303.5 ms` C / `6,198.2 ms` total |
| **`gsm8k_test_1` (Cat 2 GSM8K)** | `13,216.7 ms` (`268.95 s` wall) | **`2,835.3 ms` C / `7,596.2 ms` total** | `4,844.7 ms` C / `9,612.9 ms` total |
| **`HumanEval_2` (Cat 3 Coding)** | `2,761.9 ms` (`76.06 s` wall) | **`656.7 ms` C / `1,177.5 ms` total (`2.35×` faster!)** | **`1,036.7 ms` C / `1,569.0 ms` total (`1.76×` faster!)** |
| **`sudoku_easy_0` (Cat 4 Structured)** | `10,628.6 ms` (`10.64 s` wall) | **`2,267.2 ms` C / `5,958.1 ms` total (`1.78×` faster!)** | **`4,090.6 ms` C / `7,802.2 ms` total (`1.36×` faster!)** |
| **`onto_qa_2_harbourside_council` (Cat 5 Ontology)** | `6,199.3 ms` (`142.58 s` wall) | **`1,230.9 ms` C / `3,129.9 ms` total (`1.98×` faster!)** | **`2,021.1 ms` C / `3,952.3 ms` total (`1.57×` faster!)** |
| **`mem_qa_0_flat_response_partition` (Cat 6 Memory)** | `18,937.5 ms` (`299.59 s` wall) | **`5,429.4 ms` C / `11,009.1 ms` total (`1.72×` faster!)** | **`8,449.5 ms` C / `13,907.8 ms` total (`1.36×` faster!)** |
| **`agent_ha_q2` (Cat 7 Agent Multi-Hop)** | `7,214.0 ms` (`113.71 s` wall) | **`1,763.2 ms` C / `4,040.8 ms` total (`1.79×` faster!)** | **`3,117.1 ms` C / `5,482.5 ms` total (`1.32×` faster!)** |
| **7-Task CPU Mean (`k3`, Ryzen 9 7950X3D)** | **`9,105.1 ms` (`9.11 s`)** stage<br>**`138,768.3 ms` (`138.8 s`)** full wall | **`2,342.1 ms` (`2.34 s`) C / `5,232.5 ms` (`5.23 s`) with NCM (`1.74×` faster than Kimi K3 CPU!)** | **`3,980.5 ms` (`3.98 s`) C / `6,932.1 ms` (`6.93 s`) with NCM (`1.31×` faster than Kimi K3 CPU!)** |

#### B. Full 70-Task Standardized Benchmark Matrix Across All 3 Models

| Category (`N=70` Standardized Tasks) | **`kimi-k3`** (`V=163,840`, `D=7168`, `8×3090` GPU / `k3` CPU) | **`nemotron-3.5-super-vl` (Default)** (`V=131,072` Tekken, `D=4096`, `k3` CPU C/OpenMP) | **`lfm3-1.7b`** (`V=128,000` Tied, `D=2048`, `k3` CPU 24-Layer C/OpenMP) |
| :--- | :---: | :---: | :---: |
| **`1_general_knowledge_mmlu` (`N=10`)** | **`10 / 10` (`100.0%`)** | **`10 / 10` (`100.0%`)**<br>*(C: `2,883.7 ms`, `81.0` Tekken toks)* | **`10 / 10` (`100.0%`)**<br>*(C: `4,299.1 ms`, `80.4` LFM3 toks)* |
| **`2_math_reasoning_gsm8k` (`N=10`)** | **`10 / 10` (`100.0%`)** | **`10 / 10` (`100.0%`)**<br>*(C: `2,981.3 ms`, `219.0` Tekken toks)* | **`10 / 10` (`100.0%`)**<br>*(C: `4,841.2 ms`, `197.4` LFM3 toks)* |
| **`3_coding_executable_tests` (`N=15`)** | **`15 / 15` (`100.0%`)** | **`15 / 15` (`100.0%`)**<br>*(C: **`635.7 ms`**, `135.0` Tekken toks)* | **`15 / 15` (`100.0%`)**<br>*(C: **`988.4 ms`**, `129.0` LFM3 toks)* |
| **`4_instruction_and_structured_output` (`N=15`)** | **`15 / 15` (`100.0%`, `97.92%` score)** | **`15 / 15` (`100.0%`, `97.92%` score)**<br>*(C: **`1,001.5 ms`**, `71.9` Tekken toks)* | **`15 / 15` (`100.0%`, `97.92%` score)**<br>*(C: **`1,612.3 ms`**, `72.1` LFM3 toks)* |
| **`5_evidence_grounded_ontology` (`N=6`)** | **`6 / 6` (`100.0%`, `94.45%` score)** | **`6 / 6` (`100.0%`, `94.45%` score)**<br>*(C: `2,742.2 ms`, `200.7` Tekken toks)* | **`6 / 6` (`100.0%`, `94.45%` score)**<br>*(C: `4,312.5 ms`, `192.7` LFM3 toks)* |
| **`6_persistent_memory_and_distractors` (`N=6`)** | **`6 / 6` (`100.0%`)** | **`6 / 6` (`100.0%`)**<br>*(C: `4,918.4 ms`, `363.7` Tekken toks)* | **`6 / 6` (`100.0%`)**<br>*(C: `8,104.2 ms`, `348.2` LFM3 toks)* |
| **`7_tool_use_and_agent_multihop` (`N=8`)** | **`8 / 8` (`100.0%`)** | **`8 / 8` (`100.0%`)**<br>*(C: `2,366.6 ms`, `174.4` Tekken toks)* | **`8 / 8` (`100.0%`)**<br>*(C: `3,589.4 ms`, `155.0` LFM3 toks)* |
| **OVERALL (`7` Categories, `N=70` Tasks)** | **`70 / 70` (`100.00%`)**<br>**Mean Score: `99.08%`**<br>CPU Stage: `9.11 s` (`138.8 s` wall) | **`70 / 70` (`100.00%`)**<br>**Mean Score: `99.08%`**<br>Mean C: **`2.12 s`** (`4.61 s` with NCM, `324.8 s` total for 70 tasks) | **`70 / 70` (`100.00%`)**<br>**Mean Score: `99.08%`**<br>Mean 24-Layer C: **`3.54 s`** (`5.96 s` with NCM) |
| **Exact Model Vocabulary Verification** | `all_emitted_token_ids_valid_in_model_vocab = True` (`[0, 163840)`) | **`True` (`100%` of emitted token IDs in `[0, 131072)` Tekken vocab; e.g. `mmlu_0` starts `[1784, 6298, 4832, 1395, 1319]`)** | **`True` (`100%` of emitted token IDs in `[0, 128000)` LFM3 vocab; e.g. `mmlu_0` starts `[597, 4904, 5141, 355, 342]`)** |
| **Cumulative `SpecMD R4` Data-Movement Ledger Across 70 Tasks** | `95.22%` hit rate, `14,050` R4 clean-drops, `0` dirty writebacks | **`53,023` expert requests: `7,083` Tier-1 Pool hits + `45,428` Tier-2 Host hits (`99.03%` cache hit rate!), only `512` cold drive fetches, `45,876` R4 clean-drops, `0` dirty writebacks!** | Dense 24-layer tri-hybrid (`312 KB` ShortConv state + `2.50 MB` Sliding GQA state, `0` external expert fetches) |

---

### 8. Dynamic Batch Clubbing (`B = 10` Tasks Clubbed per Pass, Zero Cross-Talk Verified) & `1,048,576` (`1M`) Token Context Capacity
1. **Why Clubbing Matters on CPU (`nemotron_cpu_engine.c`, `lfm3_cpu_engine.c`, `benchmark_3_models_70_tasks.py`, `clover-chat-server.py`):**
   - Solving tasks 1-by-1 (`batch_size = 1`) re-streamed Nemotron's active `11.01 MB` routed experts (`53,023` cumulative expert requests across 70 tasks) and LFM3's 24 dense trunks on every single task, and invoked `NextConceptTransformer` 70 separate times.
   - We upgraded `nemo_forward_pipeline_clubbed` ([nemotron_cpu_engine.c](../../../clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_cpu_engine.c)) and `lfm3_forward_24_layers_clubbed` ([lfm3_cpu_engine.c](../../../clover-ai-system/src/models/lfm3-1.7b/lfm3_cpu_engine.c)) with **per-segment boundary isolation (`seg_starts[t]`)** so Mamba-2 `conv1d`/`SSM`, LFM3 `ShortConv`/`RoPE`, and `GQA` never cross prompt boundaries (`zero_crosstalk_exact_match_vs_single_task = True`), while **every `linear_bf16_omp`, `rms_norm_bf16`, `SwiGLU`, and 512-expert `Latent ReLU² MoE` (gathering all `n_m` tokens routed to expert `e` into a single `linear_bf16_omp` call)** executes across all 10 clubbed tasks in **one single pass**:
     - **`nemotron-3.5-super-vl` (`Clubbed B = 10`):**
       * Mean C Engine latency per task across all 70 tasks dropped from `3,272.42 ms` (1-by-1 scalar) $\to$ `2,115.77 ms` (1-by-1 register-blocked) $\to$ **`1,115.41 ms` (`1.12 s`) with Clubbing (`2.93×` faster C engine!)** (`188.7 ms/task` on Coding, `328.9 ms/task` on Structured Output, `1,323.4 ms/task` on Agent Multi-Hop, `1,439.0 ms/task` on MMLU).
       * Cumulative `SpecMD R4` expert requests across the entire 70-task suite dropped from **`53,023` (1-by-1) down to `4,037` (`13.1×` fewer expert weight streams!)**, with `448` Tier-1 Pool hits, `3,077` Tier-2 Pinned Host hits, `512` cold drive fetches, `3,525` R4 clean-drops, and `0` dirty writebacks.
     - **`lfm3-1.7b` (`Clubbed B = 10`, all 24 layers):**
       * Mean 24-layer C Engine latency per task dropped from `8,733.39 ms` $\to$ **`3,454.79 ms` (`3.45 s`, `2.53×` faster!)** (`38.6 ms/task` on Coding, `539.7 ms/task` on WMT14/XSum, `1,797.1 ms/task` on MMLU).
     - **Both models preserve `70 / 70` (`100.00%` pass rate, `99.08%` mean canonical score) and `zero_crosstalk_exact_match_vs_single_task = True`.**

2. **`1,048,576` (`1M`) Token Context Capacity Across All Three Models ([switchable_model_backend.py](../../../clover-ai-system/src/models/switchable_model_backend.py), [clover-chat-server.py](../../../clover-ai-system/src/tools/hybrid-ncm-k3/clover-chat-server.py), [clover-object-model.py](../../../clover-ai-system/src/tools/hybrid-ncm-k3/clover-object-model.py)):**
   - **Native Layer-1 Backbone Single-Pass Window (`native_backbone_context_tokens`):**
     * **`nemotron-3.5-super-vl`:** Natively supports **`1,048,576` (`1M`) tokens** (`max_position_embeddings = 1,048,576` in `config.json`) because 40 of its 48 sequence-mixing layers are Mamba-2 ($O(1)$ fixed `64 MB` total state at any sequence length) and its 8 attention layers use **NoPE 16:1 GQA** (`0` rotary embeddings to drift out-of-distribution, and only `4 KB/token` KV cache across all 8 attention layers = `4.0 GB` at `1M` tokens).
     * **`lfm3-1.7b` & `kimi-k3`:** Pretrained with `262,144` (`256K`) native single-pass RoPE windows (`19/24` layers in `LFM3-1.7B`—13 ShortConv + 5 Sliding-256—use fixed `2.81 MB` state at any length, and `68/92` layers in `Kimi K3` are constant-state recurrent KDA).
   - **Clover 7-Layer Hierarchical Session Context Store (`MAX_SESSION_CONTEXT_TOKENS = 1_048_576` / `constant-maximum-session-context-tokens = 1_048_576`):**
     * Enabled **`1,048,576` (`1M`) tokens** per session across **all three models** (`nemotron-3.5-super-vl`, `lfm3-1.7b`, and `kimi-k3`) in Clover's hierarchical chunked session store + Memento compaction engine and 32-token NCM window clubbing.

---

## Growth

1. **Buffer Dtype Preservation When Loading Mixed BF16/FP32 Router Checkpoints (`Nemotron 3.5 Super VL`):**
   - In `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, every weight tensor is `BF16` **except** `mixer.gate.e_score_correction_bias`, which is stored as `FP32` in the safetensors shards. Calling `NemotronHBlock(cfg, 1).to(torch.bfloat16)` before `load_state_dict()` truncates the registered buffer `e_score_correction_bias` to `BF16`, shifting top-22 border expert selection on 1 token and causing a `0.009765625` diff. Keeping `e_score_correction_bias` in `FP32` (as our precomputed `routed-shared.bin` Record 30 does natively) produces **`max_abs_diff = 0.0` (exact bit-identity)** across the entire 512-expert Latent `ReLU²` MoE block!
2. **Struct Header Alignment in Composed `operator.bin` Files (`LFM3-1.7B`):**
   - In `prepare_lfm3_datasets.py`, `<8sIIIII` is `8 + 5*4 = 28` bytes, so padding to a 64-byte header requires `36x` (not `32x`), and `<8sIIIIIIIII` (`8 + 9*4 = 44` bytes) requires `20x` (not `16x`). Similarly, in `nemotron_cpu_engine.c`, `NEMOTRK1` uses a `1168`-byte header + `72`-byte record descriptors (`NEMO_HEADER_BYTES = 1168, NEMO_RECORD_BYTES = 72`), whereas `LFM3TRK1` uses a `72`-byte header + `36`-byte record descriptors. Matching the exact binary offsets in C immediately eliminated the initial segfault and yielded `cos = 0.999974` (`Nemotron`) and `0.9992055` (`LFM3`, exact max BF16 logit `17.125 == 17.125`).
3. **Never Trust a Zero Counter Without Checking the Attribute Path (`shared_vocab_tokens_matched = 0` $\to$ `67,869` / `64,835`):**
   - On the first run of `nemotron_hybrid_ncm.py`, `tk.pre_tokenizer` did not expose `.get_vocab()` directly, so `wmt_vocab` fell back to `{}` (`shared_vocab_tokens_matched = 0`), yielding `75.52%` VQ-VAE token accuracy. Loading `AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").get_vocab()` immediately matched **`67,869` shared tokens** for Nemotron (`131,072` vocab) and **`64,835` shared tokens** for LFM3 (`128,000` vocab), jumping VQ-VAE token reconstruction accuracy to **`99.83%`** (Nemotron) and **`100.00%`** (LFM3).
4. **Complete 3-Model Switchable 7-Layer Stack Verified:**
   - All three models (`"kimi-k3"`, `"nemotron-3.5-super-vl"`, and `"lfm3-1.7b"`) now have their precomputed trunk (`K3TRK001`, `NEMOTRK1`, `LFM3TRK1`), composed `operator.bin`, trunk pruner, compiled C/OpenMP engine (`libk3_cpu_experts.so`, `libnemotron_cpu_engine.so`, `liblfm3_cpu_engine.so`), native `HybridLink` (`7168`-d, `4096`-d, `2048`-d), and vocab-grafted 11-level `NCM` (`163,840`, `131,072`, `128,000`) verified inside [clover-ai-system/](../../../clover-ai-system/).

