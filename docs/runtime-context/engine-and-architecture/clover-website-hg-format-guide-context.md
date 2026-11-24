# Clover Website `HG Format` Dedicated Guide Context (`site/hg-format/index.html`)

## 1. Context & Purpose
Users visiting `https://cloverframework.com/` need a clear, self-contained, primary-source engineering guide (`https://cloverframework.com/hg-format/`) explaining:
1. **Why Clover's `HG` (`HG4-16R-OP` & `K31H` Hybrid `HG4` + `HG2`) format achieves ~98–99% of full `BF16` task quality (`~95–97%` on `2.25-bit` cold MoE tail)** while shrinking `2.95 Trillion` total parameters across all 4 models (`LFM3-Omni-2B`, `Clover-35B-A3B`, `Nemotron-3.5-Super-VL 120B`, `Kimi-K3.1-Pro 2.8T`) from `~1,899 GB` down to **`876.5 GB` (`~816 GiB`)** — running on a single CPU server with zero GPUs.
2. **Measured Original vs. Clover `HG` CPU Throughput (`tok/s`) & Verbatim Side-by-Side Sample Outputs:**
   - Primary-source CPU `tok/s` comparison on `k3` (`AMD Ryzen 9 7950X3D`, `16` threads, `124 GB` DDR5, `0` GPUs): `LFM3-Omni-2B` (`10.8 tok/s` `BF16` -> `50.3 tok/s` `HG4-16R-OP` = `4.66x` decode, `1,347 tok/s` prefill, `+17.5%` over `llama.cpp Q4_0`), `Clover-35B-A3B` (`16.58 tok/s` decode, `+30%` over `llama.cpp Q4_K_M`, `97.23%` Turn-2 LCP cache hit), `Nemotron-3.5-Super-VL 120B` (`7.57–9.20 tok/s` `3.58x` lossless `2L-MTP` vs `BF16` OOM), and `Kimi-K3.1-Pro 2.8T` (`347.8 draft tok/s` `5L-DSpark` + `2.26 tok/s` 93L verify vs `1.58 TB` upstream OOM).
   - Verbatim greedy (`Temperature = 0`) side-by-side sample outputs comparing Official `BF16` PyTorch against `Clover HG4-16R` (`100%` character-identical Python `is_prime` code at `4.94x` faster wall time, identical `72 kW` multi-step arithmetic at `5.07x` higher decode `tok/s`, plus `35B` `OpenCode` + `2.8T` Lean 4 `RiemannHypothesis` kernel-verified outputs).
3. **Honest Precision Contract:** What is near-lossless weight quantization (`4.125-bpw HG4-16R` after `FWHT-128` rotation, `~0.996–0.999` cosine similarity, `PPL 29.06` vs `BF16` `28.81`; `2.125-bpw HG2-16R` on cold routed experts) vs. what is **`100.0%` mathematically lossless** (`HG4OP001` `precomputed_ops/` `Jaccard = 1.000000`, `5L-DSpark` & `2L-MTP` greedy speculative verification `draft_k=0 == draft_k=4`, and fixed-order `memcmp == 0` determinism).
4. **Standalone Universal Converter Script ([convert_to_hg.py](../../../clover-ai-system/src/tools/convert_to_hg.py), mirrored at [site/hg-format/convert_to_hg.py](../../../site/hg-format/convert_to_hg.py)) & 5-Stage Pipeline:**
   - Standalone CLI script converting any single or sharded Hugging Face `.safetensors` checkpoint (`BF16`, `FP16`, `FP32`, `FP8_E4M3`, or `MXFP4`; arbitrary `(R, K)` shapes auto-padded to `(16, 128)`) into canonical `HG4-16R` + optional `K31H` `HG2-16R` cold MoE experts + `HG4OP001` lossless rotated routers, with built-in `--ldlq`, `--verify`, and zero-download `--self-test` (`3/3` unit tests in [test_convert_to_hg.py](../../../clover-ai-system/src/tools/test_convert_to_hg.py)).
   - Companion C11 OpenMP fast packer ([hg4_fast_packer.c](../../../clover-ai-system/src/models/clover-35b-a3b/hg4_fast_packer.c)) and `-ffp-contract=off` cross-ISA C11 runtime engines.
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
