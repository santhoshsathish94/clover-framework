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
   - All **C1, C2, and C3 layers execute 100% from precomputed `CLVRTRK1` + `CLVR4B01` trunks & experts** (`24` C1 trunks + `24` C1 SwiGLU experts + `31` C2/C3 precomputed directories: `plan-trunk-0/`, `bridge-trunk-0/`, `vqvae-trunk-0/`, `vqvae-expert-0..10/`, `ncm-trunk-0..11/`, `ncm-global-0/`, `ptr-trunk-0..2/`, `ptr-global-0/`).

## 2. The 3 Supporting Components (`Memory · Relationships · Tools`)

All 3 supporting components serve **Inference**, **Plan**, and **Decode**:

1. **Support 1 — Memory (`5-Stage Git Repository, GitHub-Only Session Sync & Encrypted Vault`, `clover_git_memory_repo.py` & `mobile_vault.py`)**:
   - Organizes `clover-memory-repo/` around the 5 Clover stages (`sessions/<YYYY-MM>/<chat>.jsonl`, `execution/<YYYY-MM>/<chat>-t<N>.json`, `outcomes/<YYYY-MM>/<chat>.jsonl`, `growth/knowledge/entries.jsonl`, `growth/notes/<YYYY-MM>/<chat>.jsonl`, `config/<device>/`).
   - **GitHub-Only Logged-In Sessions**: Users sign in exclusively with their GitHub account (via OAuth Device Flow or personal token); Clover never maintains its own user/password database. Signing in automatically links `<user>/clover-memory-repo` (strictly private `"private": true`), pulls sessions & memory on startup, and pushes new turns in the background.
   - **Configurable Storage Locations (No In-App File Explorer Clutter)**: Users configure where **Media & Attachments** (`Local` folder or `Cloud-Synced` folder such as OneDrive/Google Drive/Dropbox/Nextcloud), **Memory Repository**, and **Project Workspaces** sit on disk; users browse files directly in their native OS file explorer.
2. **Support 2 — Relationships (`Turn-Level Recall Index + Ontology Graph + 11 Foundational Papers`, `memory_index.py`, `k3_ontology_graph.py` & `k3_memory_graph.py`)**:
   - **Turn-Level Recall (`memory_index.py`)**: Indexes every single message (`sessions/<month>/<chat>.jsonl#<n>:<role>`) and idle background `v3` exchange note (`USER:` + `ASSISTANT:`), retrieves via `BM25 + dense vector RRF`, excludes negative-outcome turns, and injects recalled context as a dated background system message (`F6`).
   - **Symbolic Ontology GraphRAG & 11 Foundational Papers (`1943–2017`)**: Multi-hop causal & relational traversal (`CVE -> CWE -> CAPEC`), Horn-rule inference, and post-generation source grounding audit.
3. **Support 3 — Tools (`On-Demand Tool Gate, Relevance-Filtered WebSearch, Terminal/SSH & Compiler Check`, `web_search.py`, `local_terminal.py` & `k3_generative_compiler.py`)**:
   - **Best Measured Default (`tools: "ask"` — "When I ask")**: Runs tools only when the prompt requests calculation, code execution, web lookup, or file/shell inspection (`asked_tools`), keeping everyday conversation fast (`~3.4 s` vs `~32 s` when unrequested tools fire).
   - **Relevance-Filtered Multi-Provider WebSearch (`web_search.py`)**: Supports free **Wikipedia & arXiv**, **Tavily**, and **Brave Search** with encrypted API keys in `mobile_vault.py` and automatic content-word overlap filtering to drop false-positive pages.
   - **Local Terminal & Remote SSH (`local_terminal.py`)**: Process-group timeout enforcement (`os.setsid` + `os.killpg`), 1 MB UTF-8 output cap, and device-local Ed25519 SSH public key generation (`SSHHosts`) where the private key never leaves the device.
   - **Generative Compiler Check (`CloverPythonGenerativeCompiler`)**: AST/scope/doctest/`assert` verifier with Sutton `TD(0)` statement survival and true `Allow / Ask / Off` permission prompts.

## 3. Apple-Style Progressive Disclosure & Universal `robo-icon` Pack (`v2.0.0`)

- **Palette & Assets (`assets/robo-icon/`)**: Warm cream `#F7F3E9`, Clover green `#176B2C`, deep green `#0D4D20`, and leaf highlight `#83B83D`, featuring the 4-leaf `clover-mark.svg`, `robo-clover.png` hero mascot, Android adaptive icon (`ic_launcher_foreground.png` + `ic_launcher_background.png`), and 30 stroke-matched SVG interface icons.
- **Progressive Disclosure**: Opens directly to a clean chat canvas (`Chats`, `Clover · Title`, `Settings`). Tapping `Settings` shows only 3 top-level rows (**GitHub Account**, **Storage Locations**, **Advanced**) that drill into dedicated `< Back` sub-views on demand.
