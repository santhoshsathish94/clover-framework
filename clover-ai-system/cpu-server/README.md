# Clover AI Studio — CPU Workstation Server Edition (`clover-ai-system/cpu-server/`)

Complete 7-layer Clover AI System and CPU Workstation runtime engineered for **64 GB–128 GB CPU Workstations (`k3`: AMD Ryzen 9 7950X3D, `128 GiB` DDR5 ECC RAM)** with **`NVIDIA-Nemotron-3.5-Super-VL-120B-A12B` (`d = 4096`, `V = 131,072` Tekken, Native `1M` Context)** as the default foundation model:

- **Hardware Tier Alignment across Clover AI System**:
  - **8 GB / 16 GB Mobile & Laptop (`clover-ai-system/mobile/`)**: `LiquidAI/LFM3-1.7B` (`LFM3-Omni-2B`, `d = 2048`, `V = 128,000`).
  - **64 GB / 128 GB Workstation (`clover-ai-system/cpu-server/`, Default in Web Studio)**: `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B` (`120B` total / `12B` active, `d = 4096`, `V = 131,072` Tekken, `69.9 GiB` in 4-bit — fits 100% in `128 GiB` RAM with zero disk I/O).
  - **8 × RTX 3090 Multi-GPU Server (`192 GiB` VRAM)**: `MoonshotAI/Kimi-K3` (`2.78T` total / `104B` active, `d = 7168`, `V = 163,840`).

## Contents

- **`nemotron_cpu_engine.c`**: Native C/OpenMP/AVX-512 workstation engine (`libnemotron_cpu_engine.so`) executing:
  - **Stage L1 (`trunk-0` Mamba-2 SSM)**: Selective State Space recurrence across `128` heads (`d_inner = 8192`, `d_state = 128`) with precomputed FP32 `A = -exp(A_log)`, width-4 causal depthwise `conv1d + SiLU`, `Zamba2RMSNormGated`, and per-prompt `seg_starts` isolation.
  - **Stage L2 (`trunk-1` + `expert-1` 512-Expert Latent `ReLU²` MoE)**: Top-22 sigmoid routing (`e_score_correction_bias`, `routed_scaling_factor = 5.0`), `4096 -> 1024` latent compression (`fc1_latent_proj`), 4-bit planar Group-64 (`experts.4bit.direct`, `1.36 GiB`) & lossless BF16 (`experts.direct`, `5.25 GiB`) non-gated `ReLU²` experts, `1024 -> 4096` `fc2_latent_proj`, and parallel `shared_experts` (`4096 -> 5376 -> 4096`).
  - **Stage L3 (`trunk-7` NoPE 16:1 GQA Attention)**: `32` query heads, `2` KV heads (`8 KiB/token` total KV state across 88 layers for `1M` context), 8-token OpenMP register blocking.
  - **Stage L4 & L5 (`norm_f`, Untied `131,072` `lm_head`, & 2-Layer Native `MTP`)**: `nemo_lm_head_logits`, `nemo_forward_embeddings` (continuous `4096`-d `<latent>` step execution in C), and `nemo_mtp_draft_forward` (`17.5 ms` speculative draft in 2 layers instead of 88).
  - **Layer 7 (`SpecMD Policy 3 Least-Stale + R4 Clean-Drop` SlotCache + `madvise(MADV_WILLNEED)` + 4-Tier Workstation Ledger)**: Tracks `tier0_avx512_reg_mb`, `tier1_3d_vcache_l3_mb`, `tier2_ddr5_ram_mb`, `bytes_saved_q4_and_broadcast_mb`, `madvise_willneed_calls`, `r4_clean_drops` (`0` `dirty_writebacks`), and `/proc/self/stat` `minflt`/`majflt` page-fault deltas (`1.43x` faster in 4-bit + `madvise`, reducing `minflt_delta` from `59,136 -> 0`).
- **`nemotron_workstation_clover.py`**: Complete progressive 7-layer workstation coordinator (`max_active_layer = 1 .. 7`), before/after 8-GPU training evaluator (`compare_before_vs_after_gpu_training_on_k3`), and user-triggered Clover Cycle retry engine (`run_clover_retry_cycle`).
- **`nemotron_hybrid_ncm.py`**: Python `ctypes` wrapper (`NemotronNativeCPUEngine`) + `NemotronNativeHybridLink(d=4096, bottleneck=1024)` + `NemotronComboBridge(4096 -> 384)` + `NemotronExpertBridge(4096 -> 384)`.
- **`nemotron_gpu_backbone.py`, `build_nemotron_hybridsft_dataset.py`, `train_nemotron_hybridagent_gpu.py`, `train_ncm_nemotron_gpu.py`**: 8× RTX 3090 distributed 3-stage `HybridSFT` trainer (`100.0%` held-out `<think>`/`<latent>`/`</latent>` transition accuracy, `0.7767` next-token embedding cosine on `k3`) and 11-level `131,072`-vocab `NCM` + `LexicalPointerDecoder` trainer (`94.92%` non-pad / `96.09%` overall Tekken reconstruction accuracy on `k3`).
- **`nemotron_workstation_7layer_report.json`, `nemotron_hybridagent_training_summary.json`, `nemotron_hybrid_ncm_gpu_training_summary.json`**: Complete measured verification reports from `k3` (`37.27.114.74`) and the 8× RTX 3090 GPU cluster.
