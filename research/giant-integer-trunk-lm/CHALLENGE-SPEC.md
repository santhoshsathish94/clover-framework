# Giant-Integer `Block-TrunkLM` Distillation Challenge Specification

This specification is self-contained so any engineer or frontier LLM (`GPT`, `Claude`, `Gemini`, or subagents) can propose and implement a competing **24-layer Giant-Integer `Block-TrunkLM`** student architecture and distillation recipe, evaluated on the exact same held-out teacher benchmark and Pure-C11 `x-cruncher` runtime.

---

## 1. Objective

Design a **causal 24-layer language model (`Block-TrunkLM-24`)** and **teacher-logit distillation training loop** where:
1. **Every linear channel projection and every causal sequence mixer** in all 24 layers is natively parameterized as **4-bit unsigned integers (`u4` in `[0..15]`, zero-point `8`)** arranged as **Toeplitz / Block-Circulant** or **Causal Convolution** vectors—so at inference time, **zero unstructured dense matrix-multiply loops** exist in the 24-layer backbone.
2. Every layer's static `4-bit` weights can be packed once at model load into pre-transformed `x-cruncher` Giant-Integer trunks (`xc_trunk_t`), and multi-token forward passes execute via exact arbitrary-precision integer multiplication (`xc_mul_u64` / `xc_mul_trunk`).

---

## 2. Fixed Fairness Constraints (Identical Across All Submissions)

- **Backbone Budget:**
  - Layers: `L = 24`
  - Hidden width: `D = 512` (pilot tier) or `D = 2048` (full scale)
  - Block-Circulant tile size: `B = 64` (so a `D × D` projection uses `(D/B) × (D/B) × B` `4-bit` weights—`64×` smaller than dense: `336.0 KB` nibble-packed across all 24 layers at `D = 512`, or `32.0 KB` per `2048 × 2048` projection / `~3.0 MiB` across all 24 layers at `D = 2048`).
  - Causal sequence kernel taps: `K <= 64` per layer (extendable to `1M` via `32-bit`-slot `xc_trunk_t` retention trunks).
  - Non-trunk parameters allowed per layer: only `O(D)` vectors (`RMSNorm` weights, per-channel/per-block quantization scales, or `O(D)` diagonal gates). **No dense `D × D` or low-rank `D × r` (`r > 4`) matrices allowed.**
- **Two-Stage Open-Weight Teacher Protocol:**
  - **Stage 1 (Foundation):** `LiquidAI/LFM3-Omni-2B` (`24` layers, `D = 2048`, `vocab = 128,000`, causal SDPA) providing Top-`K` (`K = 32..128`) teacher logits and optional intermediate hidden states (`Layers 12 & 24`).
  - **Stage 2 (Reasoning & Code Polish):** `Kimi K3.1 Pro` (`Top-128` sparse logits on curated math/code/reasoning sequences).
- **Strict Evaluation Rule:**
  - All evaluations are run at **100% hard `4-bit` integer quantization (`[-8..+7]` steps)**—no continuous float weights in the trunk operators at test time—and verified in Pure C11 against `xc_mul_u64` / `xc_mul_trunk`.

---

## 3. Open Design Degrees of Freedom (What Submissions May Innovate On)

Submissions are encouraged to innovate on:
1. **Cross-Block Routing Without Dense Matrices:** Fixed index permutations (e.g., Monarch/Butterfly transpose, bit-reversal, Hilbert/Gray shuffle, or Hadamard sign flips) between Block-Circulant stages.
2. **Quantization-Aware Training (QAT) Schedule:** Soft-to-hard STE annealing, learned step-size (`LSQ`), symmetric vs. balanced Vedic *Vinculum* `[-7..+7]`, or noise injection.
3. **Distillation Objective:** Temperature-scaled sparse KL + tail-mass bucket (`1 - sum(p_topK)`), Skew-KL / Reverse-KL, Cosine/RKHS hidden-state hints, or multi-step curriculum.
4. **Long-Horizon Sequence Parameterization:** Multi-scale dilated causal taps, damped power-law retention kernels, or gated associative key-value convolutions compatible with Base-3 Salem–Spencer (`{0, 1, 3, 4}`) multi-channel packing.

---

## 4. Required Deliverables & Metrics

1. A self-contained PyTorch training module (`train_variant.py`) taking `omni_teacher_top128_pilot.pt` and training for `STEPS = 200` (`BS = 16`, `SEQ = 128`).
2. Binary export matching the C11 `x-cruncher` verifier (`4-bit` `uint8` `[0..15]` arrays + `float32` per-block scales).
3. Scored on four held-out validation metrics:
   - **Hard 4-Bit Validation Perplexity (`Val PPL`)** *(lower is better)*
   - **KL Divergence to `LFM3-Omni-2B` Teacher (`KL-to-Omni`)** *(lower is better)*
   - **Top-1 Next-Token Agreement with `LFM3-Omni-2B` (`%`)** *(higher is better)*
   - **Pure-C11 `x-cruncher` Logit Cosine Parity (`must be >= 0.999`) & Prefill `ms`**
