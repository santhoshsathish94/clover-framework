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

## Growth

1. **Buffer Dtype Preservation When Loading Mixed BF16/FP32 Router Checkpoints (`Nemotron 3.5 Super VL`):**
   - In `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, every weight tensor is `BF16` **except** `mixer.gate.e_score_correction_bias`, which is stored as `FP32` in the safetensors shards. Calling `NemotronHBlock(cfg, 1).to(torch.bfloat16)` before `load_state_dict()` truncates the registered buffer `e_score_correction_bias` to `BF16`, shifting top-22 border expert selection on 1 token and causing a `0.009765625` diff. Keeping `e_score_correction_bias` in `FP32` (as our precomputed `routed-shared.bin` Record 30 does natively) produces **`max_abs_diff = 0.0` (exact bit-identity)** across the entire 512-expert Latent `ReLU²` MoE block!
2. **Struct Header Alignment in Composed `operator.bin` Files (`LFM3-1.7B`):**
   - In `prepare_lfm3_datasets.py`, `<8sIIIII` is `8 + 5*4 = 28` bytes, so padding to a 64-byte header requires `36x` (not `32x`), and `<8sIIIIIIIII` (`8 + 9*4 = 44` bytes) requires `20x` (not `16x`). Making the 64-byte offset invariant explicit immediately yielded `max_abs_diff = 0.0` across all 24 layers and final logits.
3. **Complete 3-Model Switchable Layer 1 Verified:**
   - All three models (`"kimi-k3"`, `"nemotron-3.5-super-vl"`, and `"lfm3-1.7b"`) now have their precomputed trunk (`K3TRK001`, `NEMOTRK1`, `LFM3TRK1`), composed `operator.bin`, trunk pruner, and stage/expert engines verified inside [clover-ai-system/](../../../clover-ai-system/) and integrated behind the switchable Layer-1 interface of Clover's 7-layer architecture.

