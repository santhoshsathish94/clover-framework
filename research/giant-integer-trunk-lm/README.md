# Giant-Integer Trunk-LM: Neural Layers as Exact Arbitrary-Precision Integer Multiplications

- **Status:** Experimental research record & working C11/PyTorch prototype (informs [Clover AI System Paper](../clover-ai-system-paper/README.md) and [Research Index](../README.md))
- **Date:** 2026-11-21
- **Evidence environment:** `k3` (`AMD Ryzen 9 7950X3D`, 16 cores / 32 threads, `AVX-512 IFMA52`, `124 GB` RAM, `GCC 14`, `PyTorch 2.15.0+cpu`) with `x-cruncher v1.6.1` (`51-bit` Lazy Harvey NTT + Crandall–Fagin WaveTrunk) and `LiquidAI/LFM3-Omni-2B` (`BF16` safetensors)

---

## 1. Research Question

If `4-bit` neural weights and quantized token activations are packed as digits inside **giant integers** ($\mathcal{W}_\ell$ for each of the 24 layers and $\mathcal{X}$ for a sequence of up to `1,000,000` tokens), can a neural network execute layer mixing as **one-shot arbitrary-precision integer multiplications** ($\mathcal{P} = \mathcal{W}_\ell \times \mathcal{X}$) via `x-cruncher`'s pre-transformed `xc_trunk_t`—and can we either reuse an existing open-weight checkpoint, convert a pre-trained dense model (`LFM3-Omni-2B`) before training, or train natively around Giant-Integer operators?

---

## 2. Mathematical Formulation & Worst-Case Carry Bounds

### 2.1 Kronecker Slot Packing & Exact Worst-Case Carry Ceilings (`audit_worstcase.c`)
For `4-bit` unsigned weights $w \in [0, 15]$ and `8-bit` activations $x \in [0, 255]$ summed over $K = 2048$ terms, the maximum possible dot product is $15 \times 255 \times 2048 = 7,833,600 < 2^{23}$:
- **Channel Mixing (`K = 2048`, `24-bit` / `3-byte` slots):** 100% worst-case carry-free (`max = 7.83M < 16.77M = 2^24`).
- **Million-Token Sequence Mixing (`T = 1,000,000`, `4-bit × 4-bit`, `32-bit` / `uint32_t` slots):** Under adversarial saturation $w = x = 15$, `24-bit` slots overflow at $t = 74,565$ ($74,565 \times 225 > 2^{24}$), whereas **`32-bit` (`uint32_t`) slots** (`3.81 MB` for `1M` tokens) hold up to **$T = 19,088,743$ tokens** with **zero worst-case carry spillover** ($10^6 \times 225 = 2.25 \times 10^8 < 2^{32}$).
- Placing reversed weight digits at slot $(K - 1 - k)$ and activation digits at slot $k$ aligns all $K$ products on slot $(K - 1 - k) + k = K - 1$, so reading every output dot product requires **zero division or base conversion**.

### 2.2 Base-3 *Salem–Spencer* AP-Free Multi-Channel Packing (`Step B`)
To compute $P$ **completely independent channels** (each with its own kernel $w^{(c)}$ and its own token stream $x^{(c)}$) inside **one** Giant-Integer multiplication without $P^2$ lane blowup, we place channel $c$ at **Base-3 Cantor / Salem–Spencer exponents** (integers whose base-3 digits lie in $\{0, 1\}$: $e = \{0, 1, 3, 4, 9, 10, 12, 13, \dots\}$):
- Every true channel output lands at $2 e_c$ (base-3 digits strictly in $\{0, 2\}$).
- Every cross-channel product $w^{(u)} x^{(v)}$ ($u \neq v$) lands at $e_u + e_v$ (at least one `1` in base 3).
- Therefore $\{2 e_c\} \cap \{e_u + e_v : u \neq v\} = \emptyset$ for **all** $P$: cross-channel garbage terms collide only on non-output lanes (`P = 4` uses `9` lanes per token instead of `16`, e.g., $e = \{0, 1, 3, 4\} \implies 2e = \{0, 2, 6, 8\}$).
- **Exact `32-bit` Lane Carry Ceiling (`audit_worstcase.c`):** Off-diagonal lanes `1, 5, 7` accumulate 2 cross-channel pairs each ($2 \cdot K \cdot 15 \cdot 255 < 2^{32}$), giving **zero carry into true outputs up to `K = 561,433` taps** (`0` errors at `K = 561,433`; `51` carry spills at `K = 561,434`).

---

## 3. What Was Observed (Live Measurements & Independent Audit on `k3`)

### 3.1 Unstructured Dense vs. Structured Toeplitz vs. `1M`-Token Sequence Mixing (`experiments/exp_giant_int_layer.c`)

| Experiment | Weight Giant-Int RAM | Token Giant-Int RAM | 16-Core Baseline (`-O3` / `-O3 -march=native`) | 1-Shot `x-cruncher` | Exactness |
|---|---:|---:|---:|---:|---|
| **Exp 1: Unstructured Dense** (`R=64, K=256, T=16`) | `1,536 KB` *(from `8 KB` raw; $2RTK$ guard zeros)* | `96 KB` | **`0.014 ms`** | `0.906 ms` *(slower)* | **`1,024 / 1,024` exact** |
| **Exp 2: Toeplitz Channel Layer** (`2048 × 2048, T=256`, `1.07B` MACs) | **`12.0 KB` / layer** (`0.28 MB` for 24 layers) | **`3.00 MB`** | `29.57 ms` / `3.62 ms` (`AVX-512`) | **`2.79 ms` (`10.6×` vs `-O3`, `1.3×` vs `-march=native`)** | **`524,288 / 524,288` exact** |
| **Exp 3: `1,000,000`-Token Global Causal Layer** (`32-bit` worst-case safe) | **`3.81 MB`** (`xc_trunk_t`) | **`3.81 MB`** | `~23.6 s` ($O(T^2)$) / `99 ms` (1T NumPy FFT) | **`~5.8 ms` (`32-bit` slots; `4.35 ms` `24-bit`)** | **`1,000,000 / 1,000,000` adversarial exact** |

### 3.2 Salem–Spencer Multi-Channel Packing (`experiments/exp_salem_spencer_multichannel.c`)
- **`2,048` channels × `32,768` tokens (`K = 64` taps):** `67,108,864 / 67,108,864` exact outputs (`0` mismatches) in `0.56 MB` of weight integers (`117.2 ms` direct loop vs `200.3 ms` Giant-Int; crossover occurs above `K ≈ 128` taps).
- **`256` channels × `32,768` tokens (`K = 4,096` taps = `34.4B` MACs):** `P = 4` Salem–Spencer (`64` `xc_trunk_t` trunks) ran in **`66.79 ms`** (`10.4×` faster than portable `-O3` loop `692.49 ms`, and **`1.4×–1.8×` faster** than `-O3 -march=native` auto-vectorized `AVX-512` loop, `100%` bit-exact).

### 3.3 Native 24-Layer Giant-Integer `TrunkLM-24` & `1M`-Token Passkey Recall (`experiments/train_trunk_lm24_qat.py`, `experiments/trunk_lm24_c.c`)
- **Native 24-Layer Training (`d = 192`, `4-bit` STE Toeplitz Channel-Trunks + Causal Time-Trunks):** All 24 layers' 4-bit Trunk weights occupy **`42.75 KB`** (`43,776` bytes); cross-entropy converged from `54.66` (`78.9 bpc`) to **`1.6376` (`2.363 bpc`)** in `135.7 s` on CPU.
- **Pure-C11 `x-cruncher` Execution:** Zero dense matrix-multiply loops in any layer; `24`-layer logit cosine vs PyTorch QAT = **`0.999995`**.
- **`1K -> 100K -> 1,000,000`-Token Needle-in-a-Haystack Recall (`"7394"` buried at token `#42`):**
  - `1,000` tokens: `0.02 MB` RAM · **`0.14 ms`** · Recalled `"7394"` (`4/4` exact)
  - `100,000` tokens: `1.72 MB` RAM · **`1.87 ms`** · Recalled `"7394"` (`4/4` exact)
  - `1,000,000` tokens: **`17.17 MB` RAM · `17.45 ms`** · Recalled `"7394"` at token `#999,999` (`4/4` exact)

---

## 4. Survey of Existing Pre-Trained Models & Dense-to-Giant-Integer Conversion (`experiments/audit_lfm3_conversion.py`)

### 4.1 Option 1 — Do Open-Weight Models Already Trained Like This Exist?
Auditing the Hugging Face Hub API and primary literature (`2022–2026`):

| Model / Family | Open Checkpoints on Hugging Face | Sequence Mixer | Channel / MLP Projections | Fits Full Giant-Int `TrunkLM`? |
|---|---|---|---|---|
| **Monarch Mixer (`M2-BERT`)** (Fu, Dao, Ré et al., NeurIPS 2023) | `togethercomputer/m2-bert-80M-32k-retrieval` (`Apache-2.0`) | Bidirectional FFT long-conv (Giant-Int compatible) | Monarch block-diagonal (`32×32` dense blocks, **not** circulant) | **Partial only** (`80M` BERT encoder, not a causal decoder LLM; `360M` causal GPT weights were never released) |
| **StripedHyena / HyenaDNA** (Poli et al., Together AI) | `togethercomputer/StripedHyena-Nous-7B` (`Apache-2.0`), `LongSafari/hyenadna-large-1m-seqlen-hf` (`BSD-3`) | Causal FFT long-conv up to `131K–1M` (**100% `xc_mul_trunk` compatible!**) | **Unstructured dense SwiGLU** (`>95%` of weights) | **Sequence mixer only** (channel MLPs remain dense) |
| **Toeplitz Neural Network (`TNN`)** (Qin et al., ICLR 2023) | None (`OpenNLPLab` released `TransNormerLLM` linear-attn, not TNN LLM weights) | Relative-position Toeplitz (**100% `xc_mul_trunk` compatible**) | Unstructured dense GLU | **No public LLM weights** |
| **Liquid AI `LFM3-Omni-2B`** | Official `BF16` / `GGUF` | `16/24` layers `ShortConv` (`7` taps, **`0.9990` cosine at 4-bit**) | Unstructured dense `in_proj`/`out_proj`/`SwiGLU` | **Temporal convs only** |

**Verdict on Option 1:** No open-weight causal generative LLM exists whose channel/MLP layers are already Toeplitz or Block-Circulant Giant Integers. Existing `StripedHyena` / `HyenaDNA` checkpoints *can* immediately use `x-cruncher` for exact zero-drift `1M`-token sequence convolutions, but their MLPs are dense.

### 4.2 Option 2 — Can We Easily Convert Any Pre-Trained Dense Model *Before* Training?
We tested four conversion schemes directly on `LFM3-Omni-2B`'s real `layers.0.conv.out_proj.weight` (`[2048, 2048]`) and real normalized `embed_tokens` (`[2048]`):
- **Real `LFM3-Omni-2B` Token Spectrum:** Top-`512 / 2048` singular values carry only **`64.0%`** of activation variance—a trained dense LLM spreads signal across nearly all `2,048` unstructured coordinates.
- **Zero-Shot Block-Circulant (`B = 2048 .. 8`, no training):** Cosine = **`0.0203 – 0.3548`** (fails; $0.35^{24} \approx 10^{-11}$ across 24 layers).
- **Closed-Form Monarch SVD Projection (Dao et al., ICML 2022 Theorem 1, `32×` smaller, no training):** Cosine = **`0.4440`** (still fails without continual training).
- **Block-Circulant (`B = 64`) + Low-Rank SVD Residual (`Rank 256`, `3.8×` smaller, no training):** Cosine = **`0.7378`** (insufficient zero-shot, and reintroduces dense low-rank matrices).
- **Single-Layer Stage-1 Alignment on Unseen `LFM3-Omni-2B` Tokens (`4-bit` Block-Circulant `B = 16..64`):** Held-out cosine plateaus at **`0.3833 – 0.4458`** because the *input coordinate basis* from frozen dense embeddings was never aligned for cyclic shift invariance.

**Verdict on Option 2:** **Zero-shot ("convert before training") is mathematically invalid** for unstructured dense LLMs. However, **end-to-end teacher distillation (`MOHAWK` / *Llamba* style, Bick, Dao & Gu, NeurIPS 2024)**—where *all* layers and embeddings jointly rotate into a `B = 32` or `B = 64` Block-Toeplitz Giant-Integer student trained on teacher logits over **`0.5%–1%` of pretraining tokens (`~2B–5B` tokens)**—cuts training cost by **`~100×`** compared to pretraining from scratch on trillions of tokens.

---

## 5. Stage-1 `LFM3-Omni-2B` Teacher Distillation Pilot & 3-Variant Bake-Off (`distill_omni_to_block_trunklm.py`, `block_trunklm_c.c`)

To validate the two-stage teacher distillation plan (`Stage 1: LFM3-Omni-2B` $\to$ `Stage 2: Kimi K3.1 Pro`), we extracted true causal SDPA Top-`K` logits (`92.73%` probability mass in Top-128; teacher held-out `Val PPL = 57.38`) and intermediate hidden states (`Layers 12 & 24`) from `LFM3-Omni-2B` across `20,480` real BPE tokens (`128` train chunks + `32` held-out validation chunks, `24,920` active BPE vocabulary).

### 5.1 Cycle-1 Observation (Why Freezing PCA-Tied Teacher Embeddings Matters on Small Pilots)
- Unfreezing `24,920 × 512 = 12.76M` embedding parameters on a `16K`-token pilot caused severe embedding memorization (`Val PPL = 10,420–16,226`), even though `LFM3-Omni-2B` logit distillation still more than doubled Top-1 teacher agreement (`5.13% -> 10.89%`).
- Freezing the PCA-projected `LFM3-Omni-2B` embeddings (`requires_grad = False`) forced 100% of learning into the **`336.0 KB` (`0.328 MB`) 24-layer `4-bit` Giant-Integer Trunk backbone** and eliminated out-of-vocabulary overfitting.

### 5.2 Multi-Seed Bake-Off & Independent `Claude Opus 5.5` Validation (`bakeoff.py`, `block_trunklm_d_c.c`)

An independent audit and architecture challenge run on `Claude Opus 5.5` verified all core mathematical invariants, surfaced an important baseline bar (**a smoothed add-`0.1` unigram model scores `Val PPL = 2,590`**, which `Variants A–C` had not yet beaten in 200 steps!), removed sub-vocabulary/PCA leakage by testing on the **full `128,000` vocabulary** and **fresh unseen test chunks (`160–191`)**, and introduced **`Variant D`** under the **exact same `336.0 KB` (`688,128` nibble-packed `4-bit` weights) 24-layer Giant-Integer Trunk budget**:

- **What `Variant D` Adds (Zero Dense/Low-Rank Matrices):**
  1. **Tail-Aware `T = 1.0` Top-32 KL Distillation:** Explicitly models the remaining probability mass $p_{\text{tail}} = 1 - \sum_{k=1}^{32} p_k$ against the student's non-top-32 mass (`0.9 * KL_tail + 0.1 * CE`), cutting `Val NLL` from `8.05 -> 7.72`.
  2. **Training Dropout (`0.2`) + Stronger Layer-12/24 Cosine Hints (`1.0`):** Cuts `Val NLL` further from `7.72 -> 7.34` (decisively beating the smoothed unigram model across all 4 random seeds!).
  3. **Per-Channel `O(D)` Input/Output Gain Vectors + Exact `RMSNorm(eps=1e-5)` & Leak-Free Multi-Position C11 Verifier (`block_trunklm_d_c.c`):** Adds `+0.7%` Top-1 teacher agreement with full bounds-checked C11 parity across 128 positions.

| Student Variant (`24` Layers, `336.0 KB` Packed 4-Bit Trunks, Multi-Seed Mean) | Held-Out `Val NLL` | Held-Out `Val PPL` | Top-1 Agreement w/ `Omni` | Top-32 + Tail `KL` |
|---|---:|---:|---:|---:|
| *Baseline:* Smoothed Unigram (`add-0.1`, train counts) | `7.8594` | `2,590` | — | — |
| **Variant A:** Plain Block-Circulant *(Hard CE only, no teacher)* | `8.7822` | `6,517` | `4.17%` | `5.607` |
| **Variant B:** Plain Block-Circulant + Omni Top-32 KD (`T = 1.5`) | `8.3225` | `4,115` | `9.55%` | `4.912` |
| **Variant C:** Butterfly Block-Circulant + Soft$\to$Hard STE + Omni KD + Hints | `8.2427` | `3,800` | `9.64%` | `4.854` |
| **Variant D (`Opus 5.5` Winner, 4-seed mean):** **Tail-Aware `T=1` KD + Dropout `0.2` + `1.0` Hints + `O(D)` Gains** | **`7.3555`** | **`1,565` (`2.43×` < C; beats Unigram!)** | **`11.03%`** | **`4.133`** |

#### Zero-Leak Full `128,000` Vocabulary & Fresh Untuned Test Split (`Chunks 160–191`, Teacher `PPL = 54.22`)
- **Validation (`128K` vocab):** **Variant D `Val PPL = 2,219` (`11.91%` Top-1)** vs. Variant C `4,749` (`10.30%`).
- **Fresh Unseen Test Split (`chunks 160–191`):** **Variant D `Test PPL = 1,084` (`NLL = 6.988`, `14.77%` Top-1 teacher agreement)** vs. Variant C `1,762` (`12.30%`).
- **Strict Pure-C11 `x-cruncher` Verification (`block_trunklm_d_c.c` across `2 × 64 = 128` positions):** Mean logit cosine **`0.999997`**, minimum **`0.999978`** (`centered min = 0.999885`), **`128 / 128` (`100%`) exact Top-1 token ID parity**, zero memory leaks (`338 ms` per 64-token sequence).

---

## 6. Full-Scale `D = 2048` (`3.75 MiB` 4-Bit Trunks) Two-Stage GPU Run (`train_ddp_stage1_d2048.py`, `block_trunklm_d2048_c.c`)

We scaled **Variant D** to `LFM3-Omni-2B`'s full native width (`D = 2048`, `B = 64` `32×32` Block-Circulant grids + `[32, 32, 2]` Butterfly-Transpose + `K = 64` Salem–Spencer `P = 4` causal kernels, native `128,000` BPE vocabulary with frozen `LFM3-Omni-2B` embeddings = **zero PCA loss**) on `RTX 3090` GPUs (`4×` for Stage 1; `3×` for Stage 2 while `GPU 3` ran Blender rendering):

- **Total 4-Bit Giant-Integer Trunk Weights (All 24 Layers):** **`3.75 MiB`** (`3,932,160` bytes; **`6.0 MB`** total `.trkd` file including all `O(D)` gains/norms/scales—**`576×–922×` smaller** than `LFM3-Omni-2B`'s `3.46 GB` BF16 backbone).
- **Stage 1 (`LFM3-Omni-2B` Online Tail-Aware Top-64 KD + Layer 6/12/18/24 Cosine Hints):** `1,350` steps (`49.77M` multi-domain tokens: `FineWeb-Edu` + Python code + `GSM8K` math, `~55 min` at `15.5K tok/s`).
- **Stage 2 (`Kimi K3.1 Pro` Exemplars + Multi-Round `<think>`/Python Reasoning + 50% Stage-1 Replay):** `300` steps (`11.06M` tokens at `lr = 6e-4`, `~16 min` on `3× RTX 3090`).

### 6.1 Held-Out Test Results Across Full `128,000` Vocabulary (`100%` Hard `4-Bit` Evaluation)

| Checkpoint (`D = 2048`, `3.75 MiB` 4-Bit Trunks) | General Held-Out `Test PPL` *(Teacher `30.80`)* | General Top-1 Agree | Reasoning / Kimi `Test PPL` *(Teacher `7.60`)* | Reasoning Top-1 Agree |
|---|---:|---:|---:|---:|
| **Step 0** *(Untrained random 4-bit trunks)* | `2,731,161` | `0.58%` | — | — |
| **After Stage 1** *(`49.8M` tok on `LFM3-Omni-2B`)* | `200.52` | `27.75%` | `261.75` | `19.80%` |
| **After Stage 2** *(`+ 11.1M` tok `Kimi` + Reasoning + Replay)* | **`190.51`** | **`27.56%`** | **`57.13` (`4.58×` better!)** | **`30.61%`** |

### 6.2 Full-Scale `D = 2048` Pure-C11 `x-cruncher` Verification & Honest Generation Observation
- **Pure-C11 `x-cruncher` Parity (`block_trunklm_d2048_c.c`):** Loading the `6.0 MB` `block_trunklm24_d2048.trkd` binary and executing all 24 `D = 2048` layers purely via `xc_mul_u64` achieved **`0.999970` mean / `0.999955` min** final-state cosine against PyTorch Hard `4-Bit` QAT and **`14 / 15` (`93.3%`)** exact Top-1 token agreement across the `128,000` vocabulary.
- **Honest Free-Generation Limit Observed at `60.9M` Tokens (`PPL 190.5`):** While teacher-forced single-token Top-1 agreement reached **`28–31%`** (`and Reasoning PPL = 57.13`), multi-step free autoregressive rollout degenerates after 1–3 words into topic-word repetition (`"computer computer..."`). Two clear reasons explain this:
  1. **Token Budget (`60.9M` vs. `2B–5B` tokens):** `60.9M` tokens is `~2%` of standard `MOHAWK` cross-architecture distillation (`2B–5B` tokens needed to push general `PPL` from `190` down toward `<50`).
  2. **Pure Conv-Circulant Without Selective State Reset (`Hybrid Attention`):** `LFM3-Omni-2B` itself interleaves `16` conv layers with **`8` GQA attention layers** (`layers 2, 6, 10, 14, 18, 22, ...`); pure fixed-tap causal conv + `3.75 MiB` circulant projections lack selective key-value deduplication over generated tokens, making a **hybrid (`20` Giant-Int Trunk layers + `4` tiny GQA / selective-state layers)** the natural next architectural iteration.
- **Permanent Checkpoints Saved on `k3`:** `/opt/clover-k3/block-trunklm-2b-distilled/` (`checkpoints_s1`, `checkpoints_s2/block_trunklm24_d2048.trkd`, `final_test_summary.json`, and full training logs).

---

## 7. `HG4-128`: Optimal 4-Bit Arrangement on Unchanged `LFM3-Omni-2B` (`pack_omni_hg4.py`, `lfm3_hg4_engine.c`, `bench_omni_hg4_4way.py`)

When weights are kept **100% unchanged** (zero training / zero distillation), why does a single 1-shot Giant-Integer `x-cruncher` call on dense Layer-0 `w1` (`7,680 × 2,048 = 15.73M` weights) take `57.88 ms` (`45.0 MB` padded integer) vs `0.046 ms` SIMD? Because only `7,680` of the `15,728,640` linear-convolution output slots (`0.049%`) correspond to aligned neuron rows—`99.951%` of the FFT work computes unaligned cross-row shifts and `20-bit` carry-guard zeros.

Combining insights from **QuIP#** (Tseng et al., ICML 2024), **HIGGS** (Malinovskii et al., NAACL 2025), and **T-MAC / Four-Russians SIMD LUTs** (Wei et al., EuroSys 2025), we designed **`HG4-128`**—a training-free `4.125-bpw` weight arrangement that simultaneously makes unchanged `LFM3-Omni-2B` **smaller**, **more accurate (lower PPL)**, and **faster** than `BF16`, `llama.cpp Q4_0`, and shipped `Clover v5.0.7 soc-q4`:

1. **Column-Balanced Randomized Fast Walsh–Hadamard Incoherence (`FWHT-128`, `< 0.5 µs` / token):**
   - Exact orthogonal identity $(W D^{-1} \tilde{H}_{128})(\tilde{H}_{128}^T D x) \equiv W x$ (`D = col_rms^0.5 * rademacher_signs`) rotates outlier channels into an i.i.d. Gaussian distribution across `128`-weight blocks, eliminating `min/zero-point` metadata and cutting group scale overhead **`4×`** vs `Q4_0` (`4.125 bpw` vs `4.50–4.86 bpw`).
2. **1-Cycle Register `LUT16` (`vpermb` / `vpshufb`) 15-Point MSE Lloyd–Max Gaussian Codebook:**
   - Replaces uniform integer spacing with the exact Gaussian optimal codebook (`[1, 32, 53, 70, 84, 97, 110, 122, 134, 146, 159, 172, 186, 203, 224, 255]`, midpoint `128` = exact `0.0`), decoded in **1 SIMD register cycle** before `vpdpbusd`.
3. **Unified `128.91 MiB` Tied Embedding + LM-Head (`0 MB` Duplicate Table):**
   - Because $\tilde{H}_{128}$ is a symmetric orthogonal involution ($\tilde{H}_{128}^{-1} = \tilde{H}_{128}$), a single `128.91 MiB` `embed_lm_head.hg4` serves both the `128,000`-row `lm_head` GEMM and `< 0.8 µs` 1-row inverse-`FWHT-128` token embedding lookups—eliminating `soc-q4`'s `500 MB` `embed_tokens.bf16.bin` copy.
4. **4-Row Tile Interleaved SIMD Layout (`[R/4, K/128, 2, 4, 32]`) + Register-Fused `w1+w3` SwiGLU:**
   - Interleaves `[w1_r0, w1_r1, w3_r0, w3_r1]` inside each 4-row tile so `X` is rotated/quantized once, `wu8[0..3]` is unpacked once per 32-token panel in ZMM registers, `hsum4_zmm_epi32` reduces 4 rows in one vector FMA, and $\text{SiLU}(w_1 x) \odot (w_3 x)$ executes in registers before storing.

### 7.1 4-Way Quality, Footprint & End-to-End Prompt $\to$ Response Results (`16` CPU Threads, `Temperature = 0`)

| Engine / Format (`LiquidAI/LFM3-Omni-2B`) | Total Footprint (`bpw`) | Held-Out `PPL` (`Δ vs BF16`) | `KL(BF16 \|\| Q)` / Logit Cos | Chat Prefill (`40–61t` / `256t`) | Decode (`tok/s`) | End-to-End Total (`Math` / `Python` / `CS`) |
|---|---:|---:|---:|---:|---:|---:|
| **1. Official `BF16` PyTorch** | `3,297.43 MiB` (`16.00`) | `28.8140` (`+0.0000`) | `0.00000` / `1.000000` | `295–342` / — `tok/s` | `10.7–10.8` | `5,399.8` / `8,089.7` / `7,308.4 ms` |
| **2. `llama.cpp Q4_0` (`GGUF`)** | `1,000.99 MiB` (`4.86`) | `30.7783` (`+1.9643`) | `0.07765` / `0.997691` | `460–545` / `746.3 tok/s` | `41.5–42.8` | `1,490.4` / `2,129.8` / `1,955.2 ms` |
| **3. Clover `v5.0.7 soc-q4`** | `1,990.41 MiB` (`8.60`) | `31.4318` (`+2.6178`) | `0.12494` / `0.995395` | `279–306` / `266.3 tok/s` | `44.2–47.2` | `1,975.7` / `2,008.2` / `2,026.6 ms` |
| **4. NEW Clover `HG4-128`** | **`853.88 MiB` (`4.14`)** | **`30.4237` (`+1.6097`)** | **`0.10280` / `0.997503`** | **`852–864` / `896.4 tok/s`** | **`47.8–49.9`** | **`1,080.3` / `1,780.3` / `1,310.0 ms`** |

- **Token-Identical Code & Exact Math:** On `[Python]` (`is_prime(n)` + primes `10..30`), **all four engines produce the exact same character-for-character code**, with `HG4-128` finishing **`4.54×` faster than `BF16`**, **`1.20×` faster than `llama.cpp Q4_0`**, and **`1.13×` faster than `soc-q4`** while using **`853.88 MiB`** (`3.86×` smaller than `BF16`, `2.33×` smaller than `soc-q4`, and `14.7%` smaller than `llama.cpp Q4_0`).
- **Packed Artifact on `k3`:** `/opt/clover-k3/lfm3-omni-hg4/` (`meta.bin`, `embed_lm_head.hg4`, `layer_00.hg4 .. layer_23.hg4`, packed on CPU in `184.4 s` with zero GPU training), wired live into Clover AI Studio (`30 PASS, 0 FAIL` on the full Clover Studio Quality Engine).

### 7.2 Generalizing `HG4/HG8-128` to MoE & Hybrid SSM Models (`Nemotron 3.5 Super VL` & `Clover-35B-A3B`)

Extending `HG4-128` to **NVIDIA `Nemotron 3.5 Super VL`** (`120B-A12B` MoE: Mamba-2 + `512` routed `ReLU^2` experts + `16:1` NoPE GQA + `131,072` vocab + `2`-layer MTP) and **DiffuSpace `Clover-35B-A3B-Preview`** (`35B-A3B` MoE: Bidirectional Gated DeltaNet `BD-GDN` + Gated `16Q/2KV` GQA + `256` SwiGLU experts + `248,320` vocab) introduced three general MoE/SSM upgrades:
1. **Shared-Input MoE `FWHT-128` Incoherence:** All `512` (or `256`) routed experts in a layer share a single `pre_diag_up`, so token inputs are `FWHT-128`-rotated and `int8`-quantized **once per token** instead of `22×` (`Nemotron`) or `8×` (`Clover-35B`).
2. **Closed-Form Activation-Equilibrium `pre_diag_down` (Zero Calibration Data):**
   - For `ReLU^2` MLPs (`Nemotron 3.5`), squared activations scale as $\|W_{\text{up}}[i, :]\|_2^2$, so setting $d_i = \sqrt{\|W_{\text{down}}[:, i]\|_{\text{rms}} / \|W_{\text{up}}[i, :]\|_{\text{rms}}^2} \cdot \text{sign}_i$ cuts single-expert quantization error **`23.5%` below `CLVR4B01` Group-64** (`6.68%` vs `8.73%`).
   - For SwiGLU MLPs (`Clover-35B-A3B`), $d_i = \sqrt{\|W_{\text{down}}[:, i]\|_{\text{rms}} / (\|W_{\text{gate}}[i, :]\|_{\text{rms}} \|W_{\text{up}}[i, :]\|_{\text{rms}})} \cdot \text{sign}_i$ pairs with `[gate_r0, gate_r1, up_r0, up_r1]` register-fused SwiGLU tiles.
3. **Per-Projection `16:1` GQA Balancing + Direct VNNI `HG8-128` Readout:** Prevents `4,096` Q rows from skewing `256` V rows in `16:1` GQA (`0.953 -> 0.998+` cosine) and runs un-shifted `_mm512_dpbusd_epi32` on sensitive bottleneck/head layers.

| Model & Engine (`16` CPU Threads on `k3`) | Active C1 Footprint (`bpw`) | Quality vs `BF16` (`PPL` / `KL` / `Cos`) | Extra Parity (`MTP` / `BD-Preview`) | C1 Prefill (`64t`) | `T=1` Step + Full `LM-Head` | End-to-End `C1→C2→C3` (`Math`) |
|---|---:|---:|---:|---:|---:|---:|
| **`Nemotron 3.5` · `BF16` Ref** | `13,425.2 MiB` (`16.00`) | `3,233,531` / `0.00000` / `1.000000` | `MTP Top-1 = 100.0%` | `108.5 tok/s` | `45.2 ms` (`22.1 tok/s`) | — |
| **`Nemotron 3.5` · `v5.0.7 Q4`** | `5,459.7 MiB` (`6.51`) | `3,341,119` (`+107.6K`) / `0.01113` / `0.998235` | `MTP Top-1 = 88.9%` (`461.5 ms`) | `103.8 tok/s` | `49.2 ms` (`20.3 tok/s`) | `959.9 ms` (`C1 327.2 ms`) |
| **`Nemotron 3.5` · NEW `HG4/HG8-128`** | **`3,892.5 MiB` (`4.64`)** | **`3,245,465` (`+11.9K`, `9×` closer!) / `0.01539` / `0.997528`** | **`MTP Top-1 = 95.2%` (`30.8 ms`, `15×` faster!)** | **`1,146.4 tok/s` (`11.0×`)** | **`17.2 ms` (`58.2 tok/s`, `2.9×`)** | **`460.7 ms` (`C1 110.3 ms`)** |
| **`Clover-35B-A3B` · `BF16` Ref** | `2,262.4 MiB` (`16.00`) | `3,406,558` / `0.00000` / `1.000000` | `BD-Preview Cos = 1.000000` | `603.6 tok/s` *(+248K)* | `37.6 ms` (`26.6 tok/s`) | — |
| **`Clover-35B-A3B` · `v5.0.7 Q4`** | `2,121.4 MiB` (`15.00`) | `3,443,074` / `0.00605` / `0.998374` (`85.7%` Top-1) | `BD-Preview Cos = 0.998560` | `819.3 tok/s` *(+248K)* | `40.5 ms` (`24.7 tok/s`) | `787.0 ms` (`C1 84.9 ms`) |
| **`Clover-35B-A3B` · NEW `HG4/HG8-128`** | **`1,100.9 MiB` (`7.79`)** *(or `858.4 MiB`)* | **`3,536,893` / `0.00572` (`better!`) / `0.998528` (`87.3%` Top-1!)** | **`BD-Preview Cos = 0.998841` (`14.1 ms`)** | **`5,553.7 tok/s` C1 / `857.5 tok/s` (+248K)** | **`14.1 ms` (`70.9 tok/s`, `2.9×`)** | **`360.8 ms` (`2.18×` faster)** |

### 7.3 Four-Round `Claude Opus 5.5` Cross-Validation: `HG4-16R + Seq-LDLQ + Bit-Exact Int32 FWHT` & Zero-Guard-Bit Proof

Before scaling full-model conversion across all four Clover models, we conducted a 4-round adversarial mathematical audit and live C11/PyTorch cross-validation (`k3` Zen 4 `AVX-512 VNNI + VPOPCNTDQ` + Laptop `AVX2`) with `claude-opus-5.5`:

1. **`HG4-16R` (16-Row Lane-Interleaved Layout — Zero Horizontal Reductions):**
   - Replaces 4-row horizontal reductions with `[R/16, K/8, 16_lanes, 4_bytes]` (`lanes 0..7 = gate`, `lanes 8..15 = up`), achieving **`2.12×` faster prefill at `T=64` (`7.40 TOPS` vs `3.50 TOPS`)** and **`1.82×` faster at `T=256`** at the exact same `4.125 bpw` (`853.88 MiB` on `LFM3-Omni-2B`, measured Shannon entropy = **`3.784 / 4.000 bits/weight` = `94.6%` full**).
2. **Sequential `8,192`-Token `LDLQ` + Tied-Embedding Regularizer ($\lambda_{\text{emb}} = 0.10$) + MoE Shrinkage:**
   - Verified across all 24 layers of `LFM3-Omni-2B` (`/opt/clover-k3/lfm3-omni-hg4-ldlq`, `853.88 MiB`): **`PPL = 29.0606`** (`+0.2466` vs `BF16` `28.8140` — **`8.0×` closer than `llama.cpp Q4_0`'s `+1.9643`**), **`KL = 0.06739`** (vs `Q4_0` `0.07765`), and **`Top-1 = 88.30%`** (vs `Q4_0` `86.70%`). Rare MoE experts use shrinkage Hessian $H_e^{\text{shrunk}} = (N_e H_e + \alpha H_{\text{layer}})/(N_e + \alpha)$.
3. **Cross-ISA `memcmp == 0` MatMul Bit-Exactness (`Scalar == AVX2 == AVX-512 == ARM64 NEON`):**
   - Per-128-block `int16 -> exact int32 FWHT -> int8` + `-ffp-contract=off` produced **`4,096 / 4,096` bit-identical `float32` outputs (`memcmp == 0`)** across Scalar, AVX2, and AVX-512, with non-VNNI AVX2 using signed `LUT16` + `_mm256_abs_epi8` / `_mm256_sign_epi8` to avoid `int16` saturation (`2·127·127 = 32,258 <= 32,767`).
4. **Why ~20 Guard Zero Bits Cannot Be Escaped Inside 1 Big Integer Multiply — and Why `vpdpbusd` Already Eliminates Them:**
   - **Algebraic Lower Bound:** For any linear single-product encoding $A = \sum w_k \alpha_k$, $B = \sum x_k \beta_k$ with bit-window readout $\lfloor P/D \rfloor \bmod M$ ($M \approx 2^{23}$), $\alpha_j\beta_k \cdot \alpha_k\beta_j = \alpha_j\beta_j \cdot \alpha_k\beta_k \approx D^2$ forces $|\alpha_{k+1}/\alpha_k| \ge M$, proving **`>= 23 bits/slot` (`~19` guard zeros)** is unavoidable in 1 product.
   - **Four Zero-Guard Alternatives Tested:** (a) **Harvey `±2^s` multipoint cancellation** recovers exact `10,245` even at `s = 8` (`-4` guard bits!), but requires $r$ products so total width stays $r \lceil 23/r \rceil \ge 23$ bits/slot (`NTT` at $r = K$); (b) **CRT (`mod 3..23`)** fails inside 1 multiply because carries precede $\bmod m$ and separate channels sum to `26.7 > 23` bits; (c) **Zero-mean FWHT $\sqrt{K}$ cancellation** only shrinks `23 -> 18` bits (`8σ` margin), loses bit-exactness, and stays `~1,500×` slower (`99.95%` unused convolution terms); (d) **Zero-guard bit-planes (`AVX-512 VPOPCNTDQ` `vpandq + vpopcntq`)** are `100%` bit-exact with zero guard bits and zero cross-terms, but run **`7.65×` slower** (`12.83 ms` vs `1.68 ms` at `T=64`) because `vpdpbusd` hardwires carry walls between weights and sums `4,096` bit-products in 1 clock cycle (`23/2048 = 0.011` accumulator bits/weight).

### 7.4 Complete 4-Model Full-Layer `HG4-16R + Shrinkage-LDLQ + Int32-FWHT` Conversion & Verification on `k3`

Using the OpenMP C11 fast packer (`hg4_fast_packer.c`, `~60–100×` faster than Python `torch.bucketize`) and 1-ahead HTTP Range shard streaming (`0` temporary shard files on disk, `0` GPUs required), all four Clover model families were converted end-to-end across **every layer** and verified live on `k3` (`16` threads, `-O3 -ffp-contract=off`):

| Model & Deployed `HG4-16R` Path on `k3` | Full Layers & Experts Converted | Total `HG4-16R` Size *(Time)* | Cold `mmap` Init | Cross-ISA `Scalar == AVX2 == AVX-512` | Held-Out Quality & Speculative Heads | Live `k3` CPU Throughput (`0` GPU) |
|---|---|---:|---:|---|---|---|
| **1. `LFM3-Omni-2B`** (`/opt/clover-k3/lfm3-omni-hg4`) | **`24 / 24` layers** (`16` ShortConv + `8` GQA + `128K` tied head) | **`853.88 MiB`** *(`< 3 min`)* | `11.2 ms` | **`memcmp == 0`** (`1,024,000` logits `0.0` diff) | **`PPL = 29.0585`** (`+0.27` vs `BF16` `28.79`), `KL = 0.0679`, `Top-1 = 89.36%` | **`1,347.0 tok/s`** prefill (`256t`) · **`50.3 tok/s`** decode |
| **2. `Clover-35B-A3B`** (`/opt/clover-k3/clover35b-full-hg4`) | **`40 / 40` layers** (`30` `BD-GDN` + `10` Gated GQA) × **`256` experts (`10,240` experts)** + `248,320` vocab | **`17.41 GiB`** *(`11.92 min`)* | `58.3 ms` | **`memcmp == 0`** (`1,489,920` logits `0.0` diff) | **`PPL = 17.9194`** (`40L` full causal, exact `"200 kW"` & `fib(n)`) | **`89.8 tok/s`** prefill · **`16.58 tok/s`** decode |
| **3. `Nemotron 3.5 Super VL` (`120B-A12B`)** (`/opt/clover-k3/nemotron-full-hg4`) | **`88 / 88` backbone** (`40` Mamba-2 + `40` `512`-exp Latent MoE + `8` GQA) + **`2 / 2` MTP (`90` layers, `20,992` experts)** + `131,072` vocab | **`60.20 GiB`** *(`53.61 min`)* | `48.5 ms` | **`memcmp == 0`** (`88L` + `2L MTP` `0.0` diff) | **`PPL = 14.8423`** · **`83.3%–100.0%` Stateful 2L-MTP** acceptance | **`7.00 tok/s`** prefill · **`7.57–9.20 tok/s` (`2.42×–3.58×` MTP Spec)** / `3.46 tok/s` `T=1` |
| **4. `Kimi K3.1 Pro` (`2.8T` Trunks + `DSpark`)** (`/opt/clover-k3/kimi-k31-hg4`) | **`93 / 93` trunks** (`69` `KDA` + `24` `Gated DSA` + `5`-Tensor Lightning Indexer + `L0` `33,792` Dense `SiTU-GLU` + `AttnRes`) + `163,840` vocab + **`5 / 5` `DSpark` drafter** | **`34.19 GiB`** *(`23.15 min`)* | `93.8 ms` | **`memcmp == 0`** (`163K` logits + Lightning Indexer + `5L DSpark` `0.0` diff) | All `93/93` trunks + `[L7,23,67,83,91]` `DSpark` lossless verify (`exact_greedy_match=True`) | **`347.8 draft tok/s`** (`5L DSpark`) · **`71.1 tok/s`** C1 stage · **`2.26 tok/s`** `93L` sweep |

### 7.5 Joint `Claude Opus 5.5` Audit on `C2 / C3` vs. Pure `C1` + Lossless In-Engine Speculative Decode

Following a two-round architectural audit with `claude-opus-5.5` across all 23 Studio call sites and the Qwen `C1/C2/C3` pilot evidence:
1. **Why Post-Hoc `HybridLink + 11-Level VQ-VAE / 12L NCM` Was Gated Off (`ncm: False` by Default):**
   - Across `omni_native.py`, `lfm3_native.py`, `fused_engine.py`, and `k3_host_engine.py`, `C1` already generated the complete `final` answer string *before* `omni_c2_c3` ran; by the Data Processing Inequality, a post-hoc `120M` VQ-VAE/NCM pass cannot add intelligence over a `35B–2.8T` `C1` — at best its `-45·(t-p)^2` + `+65.0` diagonal copy crutch echoed `final` (`+30–300 ms` latency), and at worst risked word substitutions on cross-family `SVD` embeddings.
   - Retraining `VQ-VAE + 12L NCM` per model (`50–250+ H100/H200 GPU-hours`) was skipped completely (`$0`), keeping lightweight `5 MB` `jepa_engine.py` strictly for non-text tool routing and plastic outcome learning (`jepa_record_outcome`).
2. **Lossless Speculative Decode Wired Directly Inside `C1` (`0` Training Hours, `exact_greedy_match == True`):**
   - **`Nemotron 3.5 Super VL` (`nemotron_generate_speculative` with Stateful `mtp0` GQA KV Cache + L1/L2 Mamba Snapshot Rollback):**
     - Python code (`binary_search`, `32` tok): **`2.57 tok/s` (`T=1`) $\to$ `5.40 tok/s` (`K=1`, `2.10×`) $\to$ `7.15 tok/s` (`K=2`, `2.78×`) $\to$ `9.20 tok/s` (`K=3`, `3.58×`, `24/24 = 100.0%` acceptance, `exact_greedy_match = True`)**!
     - Open-ended prose (`capital of France`, `32` tok): **`3.13 tok/s` (`T=1`) $\to$ `7.57 tok/s` (`K=2`, `2.42×`, `20/24 = 83.3%` acceptance, `exact_greedy_match = True`)**!
   - **`Kimi K3.1 Pro` (`kimi_k31_generate_speculative` with `[L7, L23, L67, L83, L91]` Target Features + `mask_token_id = 163824` Block-Bidirectional `DSpark`):** Verified `Scalar == AVX2 == AVX-512` `memcmp == 0`, `347.8 tok/s` `DSpark` feature-conditioned drafting, and `exact_greedy_match = True`; also verified that `/opt/clover-k3/datasets/computed/expert-1..92` (`Oct 3` `Kimi-K3.0` `MXFP4` experts, `0.0826` cosine vs `Nov 2` `Kimi-K3.1-Pro` experts) belongs to `clover-k3-chat.service` (`K3.0`) and must remain separate from `K3.1-Pro` while `+28 GiB` of redundant `Omni`/`LFM3`/`Nemotron` pruned caches were safely reclaimed (`305 GiB` free on `/dev/md2`).




