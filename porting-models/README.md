# Clover Model Porting Kit

This folder provides a repeatable, one-command pipeline for porting open-weight foundation models into the 7-layer Clover AI Studio stack:

1. **Layer 1 — Precomputed Trunks, Composed Operators, and Direct Expert Stores:**
   Converts raw `.safetensors` shards into `mmap`-ready trunk containers (`index.bin`, `common.bin`, `attention.bin`, `routed-shared.bin`, `operator.bin`, and sparse-attention `indexer.bin`) plus direct expert stores (`experts.direct` and `experts.4bit.direct`).
2. **Layer 2 — 3-Stage Distributed `HybridAgent` (`HybridSFT`):**
   Trains the model-native `HybridLink` (`d_model -> d_latent -> d_model`), the cross-model `HybridAgent` bridge (`4096 <-> d_model`), and the `<think>`, `</think>`, `<latent>`, and `</latent>` special tokens across 8 GPUs (`torchrun --nproc_per_node=8`).
3. **Layer 3 — 11-Level Next-Concept Model (`NCM`) & Lexical Pointer Decoder:**
   Adapts the 11-level `VQVAETokenizer` (`vqvae`) to the target model's native vocabulary (`163,840` for `Kimi K3.1 Pro`, `131,072` for `Nemotron 3.5 Super VL`, `128,000` for `LFM3-Omni-2B`) and jointly trains the concept bridges (`ComboBridge(d_model -> 384)` and `ExpertBridge(d_model -> 384)`), `NextConceptTransformer`, and `LexicalPointerDecoder`.
4. **Layers 4–7 — Zero-Retraining Integration:**
   Once Layers 1–3 are ported, the model plugs directly into Clover's Markdown Memory Tree (Layer 4), Deterministic Horn-Rule Ontology GraphRAG (Layer 5), Statement-Boundary Generative Compiler (Layer 6), and `SpecMD Least-Stale + Rule R4 Clean-Drop` 4-Tier Expert SlotCache (Layer 7).

---

## Quick Start — One-Command Porting (`./porting-models/port`)

Run the entire 6-stage porting pipeline (inspect, precompute trunks, prune & verify stages, 8-GPU 3-stage `HybridSFT`, 4-phase GPU `NCM` training, and end-to-end verification) with a single command:

```bash
./porting-models/port all \
  --config porting-models/configs/kimi-k3.1-pro.json \
  --work-dir /dev/shm/clover-port/kimi-k3.1-pro \
  --gpus 8
```

Or run each stage individually:

```bash
# Stage 1: Inspect Hugging Face repository, shard index, and speculative drafter
./porting-models/port inspect \
  --config porting-models/configs/kimi-k3.1-pro.json \
  --work-dir /dev/shm/clover-port/kimi-k3.1-pro

# Stage 2: Stream & precompute stage trunks, operator.bin, indexer.bin, and MXFP4 experts
./porting-models/port prepare-trunks \
  --config porting-models/configs/kimi-k3.1-pro.json \
  --work-dir /dev/shm/clover-port/kimi-k3.1-pro \
  --cached-experts 32

# Stage 3: Prune redundant trunk records and verify bit-identical (cos_sim = 1.0) stage outputs
./porting-models/port prune-trunks \
  --config porting-models/configs/kimi-k3.1-pro.json \
  --work-dir /dev/shm/clover-port/kimi-k3.1-pro

# Stage 4: Run 8-GPU distributed 3-stage HybridSFT (Format -> HybridLink -> Self-Conditioned)
./porting-models/port train-hybridagent \
  --config porting-models/configs/kimi-k3.1-pro.json \
  --work-dir /dev/shm/clover-port/kimi-k3.1-pro \
  --gpus 8

# Stage 5: Run 4-phase GPU NCM trainer (Hidden extraction -> HybridLink -> VQ-VAE -> NCM + Bridges)
./porting-models/port train-ncm \
  --config porting-models/configs/kimi-k3.1-pro.json \
  --work-dir /dev/shm/clover-port/kimi-k3.1-pro

# Stage 6: Verify all stages, bridges, and exported Clover Studio checkpoints
./porting-models/port verify \
  --config porting-models/configs/kimi-k3.1-pro.json \
  --work-dir /dev/shm/clover-port/kimi-k3.1-pro
```

On Windows PowerShell, use [port.ps1](port.ps1) with the same arguments.

---

## Case Study: Upgrading `Kimi K3` to `Kimi K3.1 Pro` (`moonshotai/Kimi-K3.1-Pro`)

### 1. Primary Source Diff (`Kimi-K3` vs. `Kimi-K3.1-Pro`)

| Property | `moonshotai/Kimi-K3` | `moonshotai/Kimi-K3.1-Pro` |
|---|---|---|
| Total / Active Parameters | `2.78T` / `104B` (`96` shards) | `2.80T` / `104B` (`96` shards, `1,561,217,377,280` B) |
| Model Type / Architecture | `kimi_k3` / `KimiK3ForConditionalGeneration` | `kimi_k31` / `KimiK31ForConditionalGeneration` |
| Attention Hybridization | `69 KDA + 24 Gated MLA` (full attention) | **`69 KDA + 24 Gated DSA`** (DeepSeek Sparse Attention) |
| Total Safetensor Count | `497,220` tensors | **`497,340` tensors** (`+120` Lightning Indexer tensors) |
| DSA Lightning Indexer (`self_attn.indexer.*`) | None | **5 tensors $\times$ 24 DSA layers** (`k_norm.weight`, `k_norm.bias`, `weights_proj.weight`, `wk.weight`, `wq_b.weight`, `index_topk = 2048`, `32` heads $\times$ `128` dim) |
| Native Context Window | `262,144` (`256K`) | **`1,048,576` (`1M`)** |
| Routed / Shared MoE Experts | `896` routed (`top-16`), `2` shared (`SiTU-GLU`) | `896` routed (`top-16`), `2` shared (`SiTU-GLU`, `MXFP4` group-32) |
| Vision Encoder | `MoonViT3d` (`27` layers) | **`MoonViT-V2`** (`401M` parameters) |
| Speculative Draft Head | `LithosAI/Kimi-K3-DSpark` (`5` layers) | **`LithosAI/Kimi-K3.1-Pro-DSpark`** (`5` layers, `2.25B` BF16, `3.45x` acceptance) |
| Thinking Efforts | `low`, `high`, `max` | `low`, **`medium`**, `high`, `max` |

### 2. How the 5-Tensor Gated DSA Lightning Indexer Works

On each of the 24 Gated DSA layers (`layers 3, 7, 11, ..., 91, 92` 0-indexed, corresponding to `dsa_layers = [4, 8, 12, ..., 92, 93]` 1-indexed), `Kimi-K3.1-Pro` adds `self_attn.indexer.*` (`14.88 MB` per layer, `357 MB` total across all 24 layers), which is packed into `trunk-{L}/indexer.bin` (`K31IDX01`):

1. **Query Index Projection:** Compressed MLA query latent $q^{\text{lat}}_t \in \mathbb{R}^{1536}$ is projected via `indexer.wq_b.weight` (`[4096, 1536]`) into $H_{\text{idx}} = 32$ indexer heads of dimension $d_{\text{idx}} = 128$: $q^{\text{idx}}_{t, h} \in \mathbb{R}^{128}$.
2. **Key Index Projection & LayerNorm:** Normalized input $u_s \in \mathbb{R}^{7168}$ is projected via `indexer.wk.weight` (`[128, 7168]`) and normalized with `indexer.k_norm.{weight, bias}` (`[128]`): $k^{\text{idx}}_s \in \mathbb{R}^{128}$.
3. **Per-Head Importance Weights:** $w_t = \text{softplus}(\text{indexer.weights\_proj}(u_t)) \in \mathbb{R}^{32}$.
4. **Top-$k$ Sparse Selection (`index_topk = 2048`):**
   $$I_{t, s} = \sum_{h=1}^{32} w_{t, h} \, \text{ReLU}\!\left(\frac{\langle q^{\text{idx}}_{t, h}, k^{\text{idx}}_s \rangle}{\sqrt{128}}\right), \quad s \le t$$
   Gated MLA then attends over the top-$2048$ KV positions selected by $I_{t, s}$.

### 3. Measured 8 $\times$ RTX 3090 Verification & Training Results (`2026-11-05`)

All 6 stages were executed on the 8 $\times$ NVIDIA RTX 3090 GPU server (`192 GiB` VRAM) and published to [`Santhoshsathish94/clover-ai-studio`](https://huggingface.co/datasets/Santhoshsathish94/clover-ai-studio):

| Stage | Metric | Before Training | After Training / Verification |
|---|---|---|---|
| **Stage 1–3 (`prepare-trunks` & `prune-trunks`)** | Pruned vs. Raw Stage Output (`cos_sim` / `max_abs_diff`) | — | **`1.000000`** / **`0.0`** (`24/24` Gated DSA Indexers verified) |
| **Stage 4 (`train-hybridagent`, 8-GPU 3-Stage `HybridSFT`)** | Held-Out NTP Loss / Perplexity | `13.6833` (`876,126.38` PPL) | **`4.2392`** (**`69.35` PPL**) |
| **Stage 4 (`train-hybridagent`, 8-GPU 3-Stage `HybridSFT`)** | `<think>` / `<latent>` Special Token Accuracy | `0.0%` | **`100.0%`** (`1.0000`) |
| **Stage 5 Phase 1 (`K31NativeHybridLink` $7168 \to 3584 \to 7168$)** | Next-Token Embedding Cosine (`eval_cos_k31_3584` / `2048`) | `0.1376` / `0.1799` | **`0.9933`** / **`0.9850`** |
| **Stage 5 Phase 2 (`163,840`-Vocab 11-Level `VQVAETokenizer`)** | Held-Out Exact Token Reconstruction (`vq_eval_acc_nonpad`) | `70.07%` (`CE = 8.1203`) | **`100.0%`** (`CE = 0.0001`) |
| **Stage 5 Phase 3 (`NextConceptTransformer` + `K31ComboBridge` + `K31ExpertBridge` + `K31LexicalPointerDecoder`)** | 11-Level VQ Code Accuracy / Surface Token Accuracy | `45.21%` / `99.55%` | **`100.0%`** / **`100.0%`** (`alpha = 0.7747`) |

---

## Directory Layout

- [port](port): Executable Bash CLI entrypoint (auto-detects `/venv/main/bin/python3`, `/dev/shm/clover-port/venv/bin/python`, or `/root/ncm-cpu/venv/bin/python`).
- [port.ps1](port.ps1): Windows PowerShell CLI wrapper.
- [port_model.py](port_model.py): Universal 6-stage CLI orchestrator (`inspect`, `prepare-trunks`, `prune-trunks`, `train-hybridagent`, `train-ncm`, `verify`, `all`).
- [kimi_k31_gpu_backbone.py](kimi_k31_gpu_backbone.py): IPv4-forced HTTP Range safetensors streamer, `MXFP4` (`E2M1` + `E8M0`) dequantizer, `K31IDX01` Gated DSA Lightning Indexer packer, and GPU-resident differentiable stage model (`KimiK31GPUStageModel`).
- [train_kimi_k31_hybridagent_gpu.py](train_kimi_k31_hybridagent_gpu.py): 8-GPU distributed 3-stage `HybridSFT` trainer for `K31NativeHybridLink(7168->3584->7168)` and `QwenToK31HybridBridge(4096<->7168)`.
- [train_kimi_k31_ncm_gpu.py](train_kimi_k31_ncm_gpu.py): 4-phase GPU trainer for the `163,840`-vocab 11-level `VQVAETokenizer`, `K31ComboBridge(7168->384)`, `K31ExpertBridge(7168->384)`, `NextConceptTransformer`, and `K31LexicalPointerDecoder`.
- [configs/kimi-k3.1-pro.json](configs/kimi-k3.1-pro.json): Configuration for `moonshotai/Kimi-K3.1-Pro` + `LithosAI/Kimi-K3.1-Pro-DSpark`.
- [configs/nemotron-3.5-super-vl.json](configs/nemotron-3.5-super-vl.json): Configuration for `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`.
- [configs/lfm3-omni-2b.json](configs/lfm3-omni-2b.json): Configuration for `LiquidAI/LFM3-Omni-2B`.
- [reports/port_verification_report.json](reports/port_verification_report.json): End-to-end 8-GPU verification and training report for `moonshotai/Kimi-K3.1-Pro`.
- [CONTEXT.md](CONTEXT.md): Primary-source context and tensor verification notes.
