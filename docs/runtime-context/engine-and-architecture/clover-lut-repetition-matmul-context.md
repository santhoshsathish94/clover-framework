# Clover Repetition & Lookup-Table Matrix Multiplication Context (`clover-lut-repetition-matmul-context.md`)

## 1. Core Insight
In quantized neural network layers (`4-bit` planar group-64 in `LFM3-Omni-2B` / `C2` / `C3`, and `MXFP4` `E2M1` in `Kimi K3.1 Pro`), every weight is drawn from a tiny, fixed discrete alphabet (`16` 4-bit integers `[-8..+7]` or `16` `E2M1` values `{0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}`). Meanwhile, a single input activation vector `x` (such as `x[2048]` in `LFM3-Omni` or the shared latent vector `z[3584]` across all 16 active routed experts in `Kimi K3.1 Pro`) is multiplied against **thousands to hundreds of thousands of weight rows**.

Rather than repeatedly unpacking and multiplying weights against `x` on every row, we can exploit discrete value repetition using **three simplified zero-inner-multiply formulations**.

---

## 2. Three Simplified Formulations

### Idea A: Digit / Magnitude Bucketing (*Zero Lookup Tables, Multiply Only at the End*)
Instead of multiplying `w[i] * x[i]` at every column `i = 0..D-1`, group `x[i]` into running accumulator buckets indexed by the discrete weight value:
- **Zero weights (`w[i] == 0`):** Skip completely (`0` operations).
- **Non-zero discrete weights:** Add (or subtract, using the sign bit) `x[i]` into `Bucket[|w[i]|]`.
- **End-of-group / End-of-row:** Multiply once per bucket (`7` non-zero magnitudes `{0.5, 1, 1.5, 2, 3, 4, 6}` for `MXFP4` `E2M1`, or `1..8` for symmetric 4-bit) and sum:
  $$\sum_{i} w_i x_i = \sum_{m \in \mathcal{M} \setminus \{0\}} m \cdot \text{Bucket}[m]$$
- **Key strengths:** Zero memory lookup tables, skips sparsity (`0` nibbles) naturally, and handles non-uniform `MXFP4` `E2M1` magnitudes (`0.5 * {1, 2, 3, 4, 6, 8, 12}`) without any input quantization loss.

### Idea B: 4-Bit Binary Switch Decomposition (`1, 2, 4, 8` Bit-Planes — *Only 4 Running Sums*)
Every unsigned 4-bit weight code `u_i = w_i + 8` (`0..15`) decomposes into **4 binary on/off switches**:
$$u_i = 1 \cdot b_{i,0} + 2 \cdot b_{i,1} + 4 \cdot b_{i,2} + 8 \cdot b_{i,3}, \qquad w_i = u_i - 8$$
- Maintain only **4 running sums** (`Sum_1`, `Sum_2`, `Sum_4`, `Sum_8`) in SIMD registers (plus precomputed `-8 * sum(x)` once per input group).
- For each block of inputs, apply bitwise `AND` masks to add `x[i]` into `Sum_1`, `Sum_2`, `Sum_4`, `Sum_8` wherever the corresponding bit is `1`.
- Combine at the end via shifts/doubling:
  $$\text{Dot} = (\text{Sum}_1 + 2\cdot\text{Sum}_2 + 4\cdot\text{Sum}_4 + 8\cdot\text{Sum}_8) - 8 \sum_i x_i$$
- **Key strengths:** Shrinks 16 buckets down to **4 register accumulators**, requires zero memory tables (no cache pressure), and vectorizes directly with SIMD bitwise masks + vector `ADD`s.

### Idea C: L1-Resident Rolling Byte-Pair Lookup Table (`LUT256` — *Flip Columns & Rows*)
Each packed byte `pk` holds two 4-bit weights (`lo = pk & 0x0F`, `hi = pk >> 4`, only `256` possible byte combinations).
- Avoid building a full-width table (`1,024 × 256` floats = `1 MiB`) all at once, which spills out of L1 cache.
- Instead, **tile over 1 group (`32 bytes = 64 weights`) or 1 column-pair at a time** across all output rows:
  - A 1-byte pair table is **`256 floats = 1 KB`**.
  - A 1-group (`32-byte`) table is **`32 × 256 floats = 32 KB`**, fitting **100% inside a single CPU core's L1 data cache** (`32–48 KB`) or GPU shared memory!
  - Precompute `LUT[j][pk] = x_lo[j] * (lo - 8) + x_hi[j] * (hi - 8)` once for the tile, then stream thousands of weight rows (`7,680` FFN rows, `98,304` routed expert rows, or `128,000` `lm_head` rows) doing pure `LUT[j][nb[j]]` lookups + additions, applying the group scale once per 32 bytes.
- **Note on SIMD Register Variant (`T-MAC` `vpshufb` / `vpdpbusd`):** When activations are quantized per-group to `int8`, `vpshufb` (`_mm256_shuffle_epi8`) or `vpdpbusd` (`_mm256_dpbusd_epi32`) executes 32 parallel 4-bit lookups / dot-products in a single 1-cycle register instruction.

---

## 3. Target Integration Points in Clover

1. **`LFM3-Omni-2B` / `LFM3-1.7B` Native C Engine on Windows x86-64** (`../../../clover-ai-system/mobile/lfm3_cpu_engine.c`)
   - `linear_4bit_group64_omp` and `fused_swiglu_w1_w3_4bit_T1` accelerate ARMv9 (`SMMLA`/`SDOT`) and Adreno OpenCL, but currently fall back to `#else` scalar C on Windows x86-64.
   - `lm_head.4bit.bin` (`128,000 × 2,048`), `ffn.4bit.bin` (`7,680 × 2,048` × 3 × 24 layers), and `operator.4bit.bin` all reuse the same `T = 1` input vector across thousands of rows.

2. **`Kimi K3.1 Pro` Native C LatentMoE (`MXFP4` Experts)** (`../../../../kimi-k3-in-c/src/core/k3_ops.c`)
   - In `k3_moe`, `z[3584]` is shared across all `16` active routed experts (`16 × 2 × 3,072 = 98,304` `w1`/`w3` rows per layer × 92 MoE layers).
   - `k3_matmul_mxfp4` is currently shuffle-port decode-bound because it unpacks `E2M1` nibbles per row; precomputing an L1-resident rolling `LUT256` (or 4-bit switch / `vpshufb` LUT) on `z[3584]` once per layer + fusing `w1`/`w3` eliminates per-row nibble decoding.

3. **`Kimi K3.1 Pro` GPU Stage Backbone** (`../../../porting-models/kimi_k31_gpu_backbone.py`)
   - `_write_expert_store` currently expands `MXFP4` experts `3.8×` into `BF16` `experts.direct`. Keeping packed `MXFP4` + shared-memory `E2M1` LUT GEMV/GEMM increases GPU-resident expert capacity ~3.8× (`~100+` vs `16–32` experts on 24 GiB VRAM) and cuts decode memory traffic 3.8×.

4. **Shared `C2` (`Plan`) + `C3` (`11-Level NCM Decode`) Runtime** (`../../../clover-ai-system/studio/clover/omni_c2_c3.py`)
   - `read_clvrtrk1` currently expands `378 MB` of `BF16` trunks into `~750 MB` of `FP32` NumPy arrays while `.4bit.bin` (`83.17 MB`) files already exist on disk. Consuming `.4bit.bin` via a 4-bit LUT/switch kernel cuts `C2+C3` RAM ~8×.

5. **Where FFT Helps vs. Doesn't Help**
   - **Text layers (`No`):** `LFM3` `ShortConv` (`K = 7`) and `Kimi K3.1 Pro` `KDA` (`K = 4`) have tiny fixed windows where direct dot-products (`4–7` ops) beat $O(T \log T)$ FFT.
   - **Multimodal Audio/Speech (`Yes`):** `LFM3-Omni-2B` Conformer Audio / `Audio-JEPA` spectrogram extraction (`STFT`) and `/tts` vocoder synthesis (`iSTFT`).

---

## 4. Verified Benchmark Results on Target #1 (`LFM3-Omni-2B` on Intel Core i7-1355U, Windows x86-64)

Measured on `13th Gen Intel Core i7-1355U` (`10` cores / `12` threads, `AVX2` + `AVX-VNNI` + `FMA`) using real `LFM3-Omni-2B` `embed_tokens.bf16.bin` (`128,000 × 2,048`) quantized to planar 4-bit group-64 and a full 24-layer (`13 ShortConv + 5 Sliding GQA + 6 Global GQA`) end-to-end `liblfm3_cpu_engine.dll` run:

| Workload | Previous Windows Scalar (`#else`) | Pure L1 `32 KB` Rolling `LUT256` (Idea C) | New `x86-64 AVX2 / AVX-VNNI` Engine | Speedup & Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **`7,680 × 2,048` FFN Matrix** (`7.76 MiB`, cache-hot) | `2.225 ms` (8 thr) / `1.267 ms` (12 thr) | `1.043 ms` (`2.13×`, `cos = 1.000000`) | **`0.263 ms`** (`30.9 GB/s`) | **`4.82×–8.46×` faster** |
| **`128,000 × 2,048` LM Head** (`129.4 MiB` DRAM) | `21.72 ms` (8 thr) / `19.83 ms` (12 thr) | `33.21 ms` *(DRAM row-stride limited)* | **`7.84 ms`** (`17.3 GB/s`) | **`2.53×–2.77×` faster**, `cos = 0.999991`, `100%` top-1 match |
| **End-to-End 24-Layer 64-Token Prefill** | `8,126.85 ms` (`7.9 tok/s`) | — | **`1,307.90 ms` (`48.9 tok/s`)** | **`6.21×` faster**, `cos = 0.999995` |
| **End-to-End 24-Layer Cached Decode** (`+ 128k` Top-64 BF16 LM Head) | `206.05 ms/tok` (`4.85 tok/s`) | — | **`79.90 ms/tok` (`12.52 tok/s`)** | **`2.58×` faster**, `100%` identical token IDs |

