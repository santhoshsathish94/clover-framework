# Giant-Integer Trunk-LM — Clover Cycle Context

## Direction (human, 2026-11-21)
- Explore representing each layer's `4-bit` weights as a giant integer ($\mathcal{W}_\ell$, 24 integers for 24 layers)
  and input token sequences (up to `1M` tokens) as a giant integer ($\mathcal{X}$), computing layer outputs in one shot
  via `x-cruncher` (`xc_mul_u64` / `xc_mul_trunk`) and slicing out the results.
- Run Step A, Step B, and Step C experiments sequentially; save the reproducible artifacts; audit whether existing
  open-weight models already have this structure or whether pre-trained dense models can be converted prior to training,
  establishing when native training is required.

## Outcomes Observed
1. **Unstructured Dense Guard-Zero Law (`Exp 1`):** Flattening an unstructured $R \times K$ matrix for $T$ tokens
   requires $2 R T K$ slots (`8 KB -> 1,536 KB` for `R=64, K=256, T=16`), giving `1,024/1,024` exact dot products but
   running slower than direct SIMD GEMM because off-diagonal cross-terms carry no useful neuron outputs.
2. **Toeplitz Channel & `1M`-Token Causal Wins (`Exp 2 & Exp 3`):** When adjacent `/` diagonals map to real neurons or
   token positions (`zero` guard blocks), `2048 × 2048` Toeplitz mixing (`T=256`) runs **`10.6x` faster** (`2.79 ms` vs
   `29.57 ms`, `0.28 MB` for 24 layers), and `1,000,000`-token global causal mixing runs **`5,424x` faster** (`4.35 ms`
   via `xc_mul_trunk` vs `23.6 s`, `2.86 MB` per integer, `100%` exact through token `#999,999`).
3. **Base-3 Salem–Spencer Multi-Channel Packing (`Step B`):** Packing 4 independent channels at AP-free lane offsets
   `{0, 1, 3, 4}` (`9` lanes/token) isolates true channel outputs at `{0, 2, 6, 8}` with **`67,108,864 / 67,108,864`
   exact matches** (`0` errors) and **`10.4x` speedup** (`66.79 ms` vs `692.49 ms`) at `K = 4,096` taps.
4. **Native 24-Layer `TrunkLM-24` & Conversion Verdict (`Step A, Step C, Option 1 & 2 Audit`):**
   - Existing open-weight causal LLMs (`StripedHyena-7B`, `LFM3-Omni-2B`) use unstructured dense MLPs; `M2-BERT` is an
     `80M` bidirectional encoder using `32×32` dense Monarch blocks.
   - Zero-shot conversion of `LFM3-Omni-2B` dense matrices (`0.02–0.44` cosine) fails because real token variance spans
     `>512` unstructured dimensions (`Top-512/2048 = 64.0%`).
   - Native 4-bit QAT training from step 0 (`TrunkLM-24`, `42.75 KB` across 24 layers) converges cleanly (`54.66 -> 1.64`
     CE), executes in pure C11 via `x-cruncher` at **`0.999995` logit cosine**, and achieves **`1,000,000`-token exact
     passkey recall in `17.45 ms` (`17.17 MB` RAM)**.

## Growth Carried Forward
- Do not attempt zero-shot Toeplitz/Circulant conversion on frozen dense LLM checkpoints.
- Always check adversarial worst-case slot saturation (`32-bit` slots for `1M` tokens; `K <= 561,433` for `32-bit`
  Salem–Spencer `P = 4`), report `-march=native` auto-vectorized baselines alongside portable `-O3`, benchmark pilot
  perplexity against a smoothed unigram baseline, and evaluate on the full `128,000` vocabulary + untuned test split.
- **Full-Scale `D = 2048` (`3.75 MiB` 4-Bit Trunks, `6.0 MB` `.trkd`) Two-Stage GPU Run (`2026-11-21`):**
  - Stage 1 (`49.8M` tok `LFM3-Omni-2B` KD) + Stage 2 (`11.1M` tok `Kimi K3.1 Pro` + `<think>` reasoning + 50% replay)
    dropped General Held-Out Test PPL from `2.73M -> 200.52 -> 190.51` (`27.6%` Top-1 agreement) and Reasoning Test PPL
    from `261.75 -> 57.13` (`30.61%` Top-1 agreement across `128K` vocab), with **`0.999970` Pure-C11 `x-cruncher`
    cosine parity**; checkpoints backed up permanently to `k3:/opt/clover-k3/block-trunklm-2b-distilled/`.
  - Free autoregressive generation at `60.9M` tokens (`PPL 190.5`) exhibits topic-token repetition—pointing to `2B+`
    token scaling plus interleaving `4` lightweight selective/GQA state layers alongside `20` Giant-Integer Trunk layers
    as the next cycle's hypothesis.
- **`HG4-128` Optimal 4-Bit Arrangement on Unchanged `LFM3-Omni-2B` (`2026-11-22`):**
  - Proved on real `LFM3-Omni-2B` Layer-0 `w1` (`7680 × 2048`) that 1-shot `x-cruncher` is `7680/7680` bit-exact (`100%`)
    when weights are unchanged, but pays a `99.95%` unaligned-row + `20-bit` carry-guard tax (`57.88 ms`, `45 MB`).
  - Built `HG4-128` (`pack_omni_hg4.py` + `lfm3_hg4_engine.c`): Column-Balanced `FWHT-128` incoherence + 1-cycle `vpermb`
    15-point Lloyd–Max Gaussian `LUT16` + unified `128.91 MiB` inverse-`FWHT` tied `embed_lm_head.hg4` + 4-row tile
    register-fused `w1+w3` SwiGLU (`853.88 MiB` total = `4.14 bpw`).
  - Verified live 4-way on `k3` (`16` threads): **`853.88 MiB`** (beats `BF16` `3,297 MiB`, `soc-q4` `1,990 MiB`, and
    `llama.cpp Q4_0` `1,001 MiB`), **`PPL 30.4237`** (beats `Q4_0` `30.7783` and `soc-q4` `31.4318`), **`852–896 tok/s`
    prefill** (`1.20x–1.85x` > `llama.cpp Q4_0`, `3.37x` > `soc-q4`), and **`49.9 tok/s` decode** (`+17%` > `Q4_0`),
    ready to generalize next to `Nemotron 3.5` and `Clover-35B`.
