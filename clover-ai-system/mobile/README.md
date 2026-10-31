# Clover AI Studio — Mobile Edition (`clover-ai-system/mobile/`)

Complete on-device Clover AI System and signed Android APK (`ai.clover.studio`) engineered for the **Samsung Galaxy S23 (`Qualcomm Snapdragon 8 Gen 2 for Galaxy`, `SM8550-AC`)** unified SoC, organized around **3 Core Pipeline Components (`Inference → Plan → Decode`)** supported by **3 Shared Supporting Components (`Memory · Relationships · Tools`)**:

## 1. The 3 Core Pipeline Components (`Inference → Plan → Decode`)

1. **Core 1 — Inference (`Base Model` + Built-In Direction B Unified SoC Engine, `lfm3_cpu_engine.c`)**:
   - Executes all 24 Tri-Hybrid layers of `LiquidAI/LFM3-Omni-2B` (`13` width-7 `Gated ShortConv` + `5` Sliding-256 `GQA` + `6` Full Global `GQA` + `SwiGLU FFN`).
   - **Built-In Direction B Hardware & Memory Engine**:
     - **Heterogeneous Adreno 740 GPU + Kryo ARMv9 CPU Co-Execution**: Zero-copy OpenCL 3.0 `CL_MEM_USE_HOST_PTR` + `cl_khr_integer_dot_product` (`dot_acc_sat`) running SwiGLU rows `0..3839` on the Adreno 740 GPU while the Kryo CPU cores simultaneously compute rows `3840..7679` via `SDOT` (`7680 / 7680` bit-identical parity, `max_abs_diff = 0.0`).
     - **ARMv8.6-A `i8mm` (`SMMLA` `vmmlaq_s32`)**: 2×2 matrix tile broadcast coalescing (`32` Int8 MACs/instruction) for `T >= 2`.
     - **1-Pass `FlashAttention-2` Online Softmax**: Register-resident `(m_i, l_i, o_acc[0..127])` across all 11 GQA layers (`0` `scores[]` allocations, `0` strided V-cache misses).
     - **`LFM3-SoC-Q4` Mobile Format (`894.4 MB` Working Set)**: Two-stage 4-bit `lm_head.4bit.bin` (`129.39 MiB`) with Top-64 exact BF16 refinement (`23.65 ms`, `100.00%` top-1 match) + 4-bit planar group-64 `operator.4bit.bin` (`164.59 MiB`) + `ffn.4bit.bin` (`586.1 MB`).
     - **`SpecMD Policy 3` OS `madvise(MADV_WILLNEED)` Lookahead & `LCP` Cache Reuse**: Cuts major flash page faults (`majflt`) by `254.7×` (`2,292 -> 9`) and skips re-prefilling matching prompt prefixes.
2. **Core 2 — Plan (`HybridAgent` & Stage-3 `HybridLink2048` + Clover Retry Cycle, `s23_on_device_clover.py` & `clover_cycle.py`)**:
   - Runs *after* Base Model inference on $h_t \to r_t = R(h_t)$, stepping continuous `<latent>` reasoning vectors into the 24-layer cache and coordinating multi-turn tool calls.
   - Runs the user-triggered **Clover Retry Cycle** (`Context → Direction → Execution → Outcome → Growth` with deterministic arithmetic/format/compiler checks and 3-way independent majority voting).
3. **Core 3 — Decode (`11-Level NCM` Concept Renderer, `S23HybridAgentAndNCM`)**:
   - Maps `(h_t, r_t)` through `Lfm3ComboBridge(2048 -> 384)` + `Lfm3ExpertBridge(2048 -> 384)` into an **11-level multiscale residual `VQVAE`** (`[1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]` = `313` concept codes), predicts the concept hierarchy via `NextConceptTransformer`, and decodes to final words via `LexicalPointerDecoder`.

## 2. The 3 Supporting Components (`Memory · Relationships · Tools`)

All 3 supporting components serve **Inference**, **Plan**, and **Decode**:

1. **Support 1 — Memory (`Local Git Repository & Multi-Repo Workspaces`, `clover_git_memory_repo.py`)**:
   - First-run `AGENTS.md` onboarding, customization, and acceptance gate (`clover-memory-repo/AGENTS.md`).
   - Multi-repo workspace manager supporting `clover-memory-repo` plus any user project repositories (local path or cloned from GitHub), automatically maintaining **`<folder>/<folder>-context.md`** in every subfolder and **`<repo>-context-index.md`** at the root.
   - Selective on-demand snippet fetch (`<= 420` tokens, zero context bloat), staged lessons (`lessons/` $\to$ Keep/Discard promotion to `notes/`), and strictly private personal GitHub sync (`"private": true` enforced).
2. **Support 2 — Relationships (`Knowledge & Ontology Graph` + 11 Foundational Papers, `k3_ontology_graph.py` & `k3_memory_graph.py`)**:
   - Multi-hop causal & relational traversal (`CVE -> CWE -> CAPEC`, domain & workspace entity graphs), Horn-rule inference, post-generation source grounding audit, and all **11 Foundational Papers (`1943–2017`)**.
3. **Support 3 — Tools (`Syntax & Compiler Validator, MCP, Terminal/Python, Self-Hosted Web & Media`)**:
   - Everything external the model needs: **Syntax & Generative Compiler Validator** (`CloverPythonGenerativeCompiler` AST/scope/doctest/`assert` gate + Sutton `TD(0)` statement survival), **MCP Connections & VS Code Bridge**, **Python / Terminal Subprocess Runner**, **Self-Hosted Web Browser & Search** (`Wikipedia`, `arXiv`, live URL/GitHub reader, native WebView/browser with zero paid API keys), **Multimodal & Media Generators** (Pillow PNG, FFmpeg H.264 MP4, eSpeak NG / Android TTS WAV, camera & speech recognition), and **Local File / Repo Tools** — all governed by user permissions (`Allow`, `Ask`, `Off`).
