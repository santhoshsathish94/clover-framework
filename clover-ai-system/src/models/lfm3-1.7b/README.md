# Liquid AI LFM3-1.7B Model Backend (`clover-ai-system/src/models/lfm3-1.7b/`)

This folder registers **Liquid AI's `LFM3-1.7B`** (`LiquidAI/LFM3-1.7B`, `1.73B` dense parameters, `24` tri-hybrid layers, `D_model = 2048`, `V = 128,000` tied) as the compact edge / local-agent Layer-1 backbone inside `clover-ai-system/`:

- **24-Layer Tri-Hybrid Schedule:**
  - `13` Gated Short Convolution layers (`Lfm3ShortConv`, width $K = 7$, `312 KB` fixed state across all 13 layers)
  - `5` Sliding-Window GQA layers (`sliding_window = 256`, `rope_theta = 10,000.0`, full RoPE, `2.50 MB` max state)
  - `6` Full Global GQA layers (`256K` causal window, `rope_theta = 1,000,000.0`, zero-tail partial RoPE `0.5`, per-head sigmoid output gate)
  - Sandwich RMSNorm (`operator_norm` + `post_operator_norm`, `ffn_norm` + `post_ffn_norm`) + dense SwiGLU (`2048 -> 7680 -> 2048`)
- **Stage-by-Stage Verification & Study:** [../../../../research/lfm3_stage_study.py](../../../../research/lfm3_stage_study.py) and [../../../../docs/runtime-context/engine-and-architecture/clover-lfm3-1.7b-stage-study-context.md](../../../../docs/runtime-context/engine-and-architecture/clover-lfm3-1.7b-stage-study-context.md)
- **Switchable Layer-1 Registration:** `"lfm3-1.7b"` in [../switchable_model_backend.py](../switchable_model_backend.py) and `class-lfm3-compact-foundation-model-backend` in [../../tools/hybrid-ncm-k3/clover-object-model.py](../../tools/hybrid-ncm-k3/clover-object-model.py).
