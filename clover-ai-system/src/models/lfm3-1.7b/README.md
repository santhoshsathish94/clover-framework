# Liquid AI LFM3-1.7B Model Backend (`clover-ai-system/src/models/lfm3-1.7b/`)

This folder registers **Liquid AI's `LFM3-1.7B`** (`LiquidAI/LFM3-1.7B`, `1.73B` dense parameters, `24` tri-hybrid layers, `D_model = 2048`, `V = 128,000` tied) as the compact edge / local-agent Layer-1 backbone inside `clover-ai-system/`:

- **24-Layer Tri-Hybrid Schedule:**
  - `13` Gated Short Convolution layers (`Lfm3ShortConv`, width $K = 7$, `312 KB` fixed state across all 13 layers)
  - `5` Sliding-Window GQA layers (`sliding_window = 256`, `rope_theta = 10,000.0`, full RoPE, `2.50 MB` max state)
  - `6` Full Global GQA layers (`256K` causal window, `rope_theta = 1,000,000.0`, zero-tail partial RoPE `0.5`, per-head sigmoid output gate)
  - Sandwich RMSNorm (`operator_norm` + `post_operator_norm`, `ffn_norm` + `post_ffn_norm`) + dense SwiGLU (`2048 -> 7680 -> 2048`)
- **Precomputed Dataset & Composed Operator Builder (`LFM3TRK1`, `LFM3_OP1`, `LFM3ROP1`):** [prepare_lfm3_datasets.py](prepare_lfm3_datasets.py)
- **Trunk Pruner (deletes `attention.bin` from all 24 / 24 trunks, saving `445.14 MiB`):** [prune_lfm3_trunk.py](prune_lfm3_trunk.py)
- **Precomputed 24-Layer Runtime Engine & 4-Bit SwiGLU:** [lfm3_precomputed_engine.py](lfm3_precomputed_engine.py)
- **Native 24-Layer C/OpenMP Engine (`liblfm3_cpu_engine.so`):** [lfm3_cpu_engine.c](lfm3_cpu_engine.c)
- **Native `HybridAgent` (`Lfm3NativeHybridLink` $d=2048$) & Native 11-Level `NCM` (`128,000`-Vocab VQ-VAE + Bridges):** [lfm3_hybrid_ncm.py](lfm3_hybrid_ncm.py)
- **End-to-End 24-Layer Verification & 7-Layer Clover Switching Suite:** [verify_lfm3_clover_integration.py](verify_lfm3_clover_integration.py)
- **Stage-by-Stage Verification & Study:** [../../../../research/lfm3_stage_study.py](../../../../research/lfm3_stage_study.py) and [../../../../docs/runtime-context/engine-and-architecture/clover-lfm3-1.7b-stage-study-context.md](../../../../docs/runtime-context/engine-and-architecture/clover-lfm3-1.7b-stage-study-context.md)
- **Switchable Layer-1 Registration:** `"lfm3-1.7b"` in [../switchable_model_backend.py](../switchable_model_backend.py) and `class-lfm3-compact-foundation-model-backend` in [../../tools/hybrid-ncm-k3/clover-object-model.py](../../tools/hybrid-ncm-k3/clover-object-model.py).
