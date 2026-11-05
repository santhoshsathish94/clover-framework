# Context — `porting-models/` (Clover Model Porting Kit & Kimi K3.1 Pro Upgrade)

## 1. Objective

1. Upgrade Clover AI Studio's primary Kimi backbone from **`moonshotai/Kimi-K3`** (`2.78T` / `104B` active, `69 KDA + 24 Gated MLA`) to **`moonshotai/Kimi-K3.1-Pro`** (`2.8T` / `104B` active, `69 KDA + 24 Gated DSA` with the 5-tensor Lightning Indexer, `1M` native context, `SiTU-GLU`, `MoonViT-V2`, and `LithosAI/Kimi-K3.1-Pro-DSpark` 5-layer speculative drafter).
2. Use the 8x RTX 3090 GPU server (`gpu`, `192 GiB` VRAM, `125 GiB` `/dev/shm`) to precompute the `Kimi-K3.1-Pro` stage trunks/operators/experts and train the `HybridAgent` (`3-Stage HybridSFT` across 8 GPUs) and `11-Level NCM` bridges (`K31NativeHybridLink`, `QwenToK31HybridBridge`, `K31ComboBridge`, `K31ExpertBridge`, `163,840`-vocab `VQVAETokenizer`, `NextConceptTransformer`, and `K31LexicalPointerDecoder`).
3. Package the entire end-to-end workflow inside `porting-models/` with a single executable `port` entrypoint (`./porting-models/port`) and clear, reproducible documentation (`porting-models/README.md`) so anyone can port an open-weight foundation model into Clover AI Studio.

---

## 2. Primary Source Observations (`moonshotai/Kimi-K3.1-Pro` & `LithosAI/Kimi-K3.1-Pro-DSpark`)

- **`moonshotai/Kimi-K3.1-Pro`** (`sha = 6d0a0202956796543364d666ab19c6ad1a494ee1`, released `2026-11-02`):
  - `model_type = "kimi_k31"`, `architectures = ["KimiK31ForConditionalGeneration"]`.
  - Total parameters: `2.8T` (`1,561,217,377,280` bytes across `96` safetensor shards `model-00001-of-000096.safetensors`..`model-00096-of-000096.safetensors`), `104B` active parameters per token.
  - Dimensions: `num_hidden_layers = 93` (`first_k_dense_replace = 1` -> Layer 0 dense `33,792` `SiTU-GLU`; Layers `1..92` Latent `SiTU-GLU` MoE with `n_routed_experts = 896`, `num_experts_per_tok = 16`, `n_group = 8`, `topk_group = 4`, `routed_scaling_factor = 2.827`, `moe_latent_size = 3584`, `moe_intermediate_size = 3072`, `num_shared_experts = 2` [`6144` intermediate]), `hidden_size = 7168`, `vocab_size = 163840`.
  - Tokenizer: `tiktoken.model` (`sha256 = b6c497a7469b...`, `163,840` vocab) is byte-for-byte identical to `Kimi-K3`. `encoding_k3.py` adds `"medium"` to `_VALID_THINKING_EFFORTS = {"low", "medium", "high", "max"}`.
  - **Architectural Diff vs. `Kimi-K3`**:
    - Replaces full-attention Gated MLA on the 24 global attention layers (`dsa_layers = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93]` 1-indexed = `3, 7, 11, ..., 91, 92` 0-indexed) with **Gated DeepSeek Sparse Attention (`Gated DSA`)**.
    - Tensor count increases from `497,220` (`Kimi-K3`) to `497,340` (`Kimi-K3.1-Pro`). The exact `120` added tensors are 5 **Lightning Indexer** (`self_attn.indexer.*`) tensors on each of the 24 Gated DSA layers (`14.88 MB` per layer):
      1. `self_attn.indexer.k_norm.weight` (`BF16 [128]`)
      2. `self_attn.indexer.k_norm.bias` (`BF16 [128]`)
      3. `self_attn.indexer.weights_proj.weight` (`BF16 [32, 7168]`)
      4. `self_attn.indexer.wk.weight` (`BF16 [128, 7168]`)
      5. `self_attn.indexer.wq_b.weight` (`BF16 [4096, 1536]`)
    - `dsa_config`: `{"index_topk": 2048, "index_head_dim": 128, "index_num_heads": 32, "index_rope_dim": 0, "index_k_norm_type": "layernorm", "index_topk_freq": 1, "index_skip_topk_offset": 1}`.
  - **Shard Mapping**:
    - `model-00001-of-000096.safetensors`: Layer 0 (`KDA` + Dense `SiTU-GLU` MLP)
    - `model-00002-of-000096.safetensors`..`model-00093-of-000096.safetensors`: Layers `1..92` (`KDA` or `Gated DSA` + 896 `MXFP4` experts `w1/w2/w3.{weight_packed, weight_scale}`)
    - `model-00094-of-000096.safetensors`: `embed_tokens.weight [163840, 7168]`, `lm_head.weight [163840, 7168]`, `norm.weight [7168]`, `output_attn_res_norm.weight [7168]`, `output_attn_res_proj.weight [1, 7168]`
    - `model-00095-of-000096.safetensors` & `model-00096-of-000096.safetensors`: `MoonViT-V2` (`401M` vision tower) + `mm_projector`
- **`LithosAI/Kimi-K3.1-Pro-DSpark`** (`sha = d791d217d1cc679e07877721761bfcc09d70d514`, released `2026-11-04`):
  - 5-layer `2.25B` BF16 speculative decoding draft head (`62` tensors in `model.safetensors`, `target_layer_ids = [7, 23, 67, 83, 91]`, `fc.weight [7168, 35840]`, `confidence_head.proj.weight [1, 7424]`, mean acceptance length `3.448` at block size 8).

---

## 3. Target Infrastructure

- **`gpu` (`8x RTX 3090` Server)**:
  - `8x NVIDIA GeForce RTX 3090` (`24,576 MiB` each = `192 GiB` VRAM), `251 GiB` RAM, `125 GiB` `/dev/shm`.
  - Root `/` overlay is `16 GiB`; all downloads, precomputed trunks, virtualenvs, and training checkpoints are placed inside `/dev/shm/clover-port/`.
  - Direct IPv6 connections to Hugging Face CDN time out from this host; all Hugging Face downloads force IPv4 (`AF_INET`).
- **`k3`**:
  - Hosts `clover-k3-chat.service` (`8765`) and `clover-web-studio.service` (`8780`) serving `https://cloverframework.com/clover-ai/`.
