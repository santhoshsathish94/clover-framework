# Clover Website `HG Format` Dedicated Guide Context (`site/hg-format/index.html`)

## 1. Context & Purpose
Users visiting `https://cloverframework.com/` need a clear, self-contained, primary-source engineering guide (`https://cloverframework.com/hg-format/`) explaining:
1. **Why Clover's `HG` (`HG4-16R-OP` & `K31H` Hybrid `HG4` + `HG2`) format achieves ~98–99% of full `BF16` task quality (`~95–97%` on `2.25-bit` cold MoE tail)** while shrinking `2.95 Trillion` total parameters across all 4 models (`LFM3-Omni-2B`, `Clover-35B-A3B`, `Nemotron-3.5-Super-VL 120B`, `Kimi-K3.1-Pro 2.8T`) from `~1,899 GB` down to **`876.5 GB` (`~816 GiB`)** — running on a single CPU server with zero GPUs.
2. **Honest Precision Contract:** What is near-lossless weight quantization (`4.25-bit HG4-16R` after `WHT-16` rotation, `~0.997+` layer cosine similarity; `2.25-bit HG2-16R` on cold routed experts) vs. what is **`100.0%` mathematically lossless** (`HG4OP001` `precomputed_ops/` `Jaccard = 1.000000`, `5L-DSpark` & `2L-MTP` greedy speculative verification `draft_k=0 == draft_k=4`, and fixed-order `memcmp == 0` determinism).
3. **How to Convert Any Open-Weight Model to `HG` Format (`5-Step Pipeline`):**
   - Step 1: Compile the zero-PyTorch-GPU C11 `.safetensors` `mmap` packer (`hg4_fast_packer.c`).
   - Step 2: Apply 16-row Walsh–Hadamard Transform (`WHT-16`) + group-64 symmetric quantization to Trunk, Attention (`GQA`/`Mamba-2`/`GDN`/`KDA`/`DSA`), and Shared Experts (`HG4-16R`, `4.25 bits/w`).
   - Step 3: Pack Routed MoE Experts — Full `HG4-16R` (`<= 120B` models) or Hybrid `K31H` (`DDR5-pinned HG4-16R` hot experts + `NVMe-streamed 2.25-bit HG2-16R` cold experts with `POSIX_FADV_DONTNEED`).
   - Step 4: Precompute lossless `HG4OP001` operator sidecars (`precomputed_ops/`: router norms, RoPE/YaRN tables, recurrent decay gates) and verify `min_topk_jaccard == 1.000000`.
   - Step 5: Compile the stateful C11 AVX2+FMA engine (`*_hg4_engine.c`) with LCP prefix cache + speculative drafter (`2L-MTP` / `5L-DSpark`).
4. **How to Set Up & Run Pre-Converted Models (`Hugging Face` -> `/v1` Fusion Gateway -> `OpenCode` & `VS Code`):**
   - Direct links to all `Santhoshsathish94/*` Hugging Face repositories (`LFM3-Omni-2B-HG4-16R`, `Clover-35B-A3B-HG4-16R-OP`, `Nemotron-3.5-Super-VL-120B-HG4-16R-OP`, `Kimi-K3.1-Pro-K31H-Hot-HG4-OP`, and `Kimi-K3.1-Pro-K31H-Cold-HG2-Part1..3`).
5. **What Tests We Ran & How Anyone Can Reproduce Them:**
   - Frozen 23-task Certified Lean 4 `RiemannHypothesis` (`lean4:v4.33.1`, `#print axioms` subset `{propext, Classical.choice, Quot.sound}`) + Systems SWE + `REA` Reverse-Engineering suite (`rh_bench_suite_v2.jsonl`).
   - Frozen 48-task Clover Studio Capability Probe Suite (`studio_capability_probes_v2.jsonl`, `Phases S0 → S7`).
   - Exact copy-paste verification commands (`test_hg4_fusion_gateway.py`, `run_studio_phase.py`, and SHA-256 boundary checks).

## 2. Website Integration Invariants (`scripts/check-site.py`)
- Uses `../botanical-design-system.css?v=5.0.7` and `../website-interaction-controllers.js?v=5.0.7` matching `VERSION` (`5.0.7`).
- Primary `<nav class="site-nav">` must remain byte-normalized identical across all 9 navigable HTML pages (`index.html`, `origin/`, `ai-system/`, `hg-format/`, `getting-started/`, `governance/`, `privacy/`, `glossary/`, `author/`).
- Visually verified in the built-in browser via Playwright (`desktop 1280px` and `mobile 390px`) with zero console/layout errors and `0` errors in `check-site.py`, `check-links.py`, and `scripts/tests`.
