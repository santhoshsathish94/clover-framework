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

## 6. What Remains Unknown & Next Scale Step

1. **Multi-GPU `2B–5B` Token Run:** Scaling **Variant D** (`D = 2048, B = 64` = `3.0 MiB` of packed 4-bit Trunk weights across 24 layers) on GPU over `2B–5B` tokens with online `LFM3-Omni-2B` Stage-1 tail-aware logits followed by `10M–50M` cached Top-128 `Kimi K3.1 Pro` Stage-2 reasoning tokens.
2. **Multi-Token Block-Diffusion Decode:** Pairing `Block-TrunkLM-24` with `16–64`-token parallel draft decoding so generation stays in `xc_mul_trunk`'s multi-token fast path.
