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

## 3. Apple-Style Progressive Disclosure & Universal `robo-icon` Pack (`v2.1.0`)

- **Palette & Assets (`assets/robo-icon/`)**: Warm cream `#F7F3E9`, Clover green `#176B2C`, deep green `#0D4D20`, and leaf highlight `#83B83D`, featuring the 4-leaf `clover-mark.svg`, `robo-clover.webp` / `robo-clover.png` hero mascot, Android adaptive icon (`ic_launcher_foreground.png` + `ic_launcher_background.png`), and 30 stroke-matched SVG interface icons.
- **Progressive Disclosure**: Opens directly to a clean chat canvas (`Chats`, `Clover · Title`, `Settings`). Tapping `Settings` shows only 3 top-level rows (**GitHub Account**, **Storage Locations**, **Advanced**) that drill into dedicated `< Back` sub-views on demand.

## 4. Laptop Step 21D–25 Parity & Real-User Regression Fixes Verified on Mobile (`11 / 11` Suites)

All laptop real-user findings and regression fixes (`Steps 21D–25`) are implemented and verified across Windows and Samsung Galaxy S23 (`SM-S911B`):
1. **Follow-Up Context & Exact Written Precision in `clover_cycle.py`**:
   - `follow_up()` + `context_before(history, msg)` prepend the preceding exchange before follow-up retries (`"And how much would 30 pens cost?"`), and `needs_web(context + question)` sees the earlier numbers so unnecessary web searches are skipped.
   - `same(a, b)` requires `0.2%` numeric precision (`same(20.1, 20) == False`), and overturn lessons are recorded only when backed by user detail or a verified Python calculation (`program_at in best[1]`).
2. **Non-Interactive Git, ANSI Stripping & Destructive Command Gate in `local_terminal.py`**:
   - `QUIET_GIT` (`GIT_EDITOR=true`, `GIT_SEQUENCE_EDITOR=true`, `GIT_PAGER=cat`, `GIT_TERMINAL_PROMPT=0`, `GCM_INTERACTIVE=never`) prevents interactive git commands from hanging.
   - `local_terminal.risky(cmd)` detects destructive commands (`git rebase/reset/clean/push/restore`, `rm/rmdir/del/Remove-Item`, `kill/taskkill/Stop-Process`, `shutdown/reboot/mkfs`) and always triggers an `Ask` confirmation card (`Allow once` / `Don't allow`, with `Always allow` hidden) even when `shell` permission is set to `Allow`.
3. **Conceptual Question vs. Tool Routing, Ontology URL/Path Filtering & 2-Topic-Word Learning Filter**:
   - `asked_tools()` distinguishes conceptual questions (`"In one sentence, what does git rebase do?"` -> `[]`) from explicit tool requests, offers both `repo_read` and `repo_write` on file-edit prompts, and routes image requests to `generate_image`.
   - `_LINKS_AND_PATHS` strips URLs and file paths before ontology matching, skips ontology on pure tool turns, and requires `>= 2` shared topic words (`topic_terms`) before injecting learned lessons.
4. **Clean User Git Status (`.git/info/exclude`) & Safe Shared Media Folder Clear**:
   - `_git_exclude()` records Clover's exact `<folder>-context.md` and `<repo>-context-index.md` paths in `<workspace>/.git/info/exclude` (and removes them on `clean`) so `git status` in the user's repo stays completely clean.
   - `get_storage_locations()` and `clear_media_storage()` count and remove only Clover's content-hash files (`^[0-9a-f]{24,32}\.[a-z0-9]{2,5}$`), preserving any user files in shared local or cloud folders.

## 5. Central Semantic Interface (`clover_semantics.py` — `Meaning · Context · Interpretation · Alignment`)

Rather than adding a separate sequential processing layer, `clover_semantics.py` provides the shared coordinate and interpretation protocol connecting the **User**, the **Fuel System (`S1 Memory · S2 Relationships · S3 Tools`)**, and the **Model Engine (`C1 Inference · C2 Plan · C3 Decode`)**, verified in situ on `k3` (`semantic_architecture_study_report.json`) and `s23` (`12 / 12` suites):

1. **Meaning — Mean-Centered Layer-Selective Geometry & Forced-Choice Readout** (*Schiekiera et al., ICML 2026*):
   - Empirical inspection across all 24 Tri-Hybrid layers of `LFM3-Omni-2B` (`lfm3_forward_all_layers` in `lfm3_cpu_engine.c`) showed that raw hidden states exhibit strong cone anisotropy (`mean cos = 0.344 -> 0.860` at Layer 12, `0.799` at Layer 23).
   - Subtracting the empirical mean $\tilde{\mathbf{h}}^{(\ell)} = \mathbf{h}^{(\ell)} - \boldsymbol{\mu}^{(\ell)}$ collapses anisotropy to `-0.034` and increases Representational Similarity Analysis (RSA) correlation with 3-model semantic consensus (`BGE-small` + `E5-small` + `MiniLM-L6`) from `0.238 -> 0.554` (`+0.317`, `NN@5 = 0.500`), peaking at **Layer 18 (`full_gqa`)** rather than Layer 23 (`0.478`).
   - Constrained Forced-Choice (`FC`) candidate comparison aligns substantially better with internal semantic geometry than open-vocabulary Free Association (`FA`) (`0.4712` vs `0.3053`, `+0.1659` advantage).
2. **Cross-Module Alignment — $\mathcal{V}$-Enriched Yoneda Embedding (`IndraSemanticSpace`)** (*Lu et al., NeurIPS 2025*):
   - Maps vectors from any module (`S2` `384d`, `C1` `2048d`, `C2` `2048d`, `C3 VQ-VAE` `384d`) to their angular-distance relational profile $\mathbf{I}(x) = [\arccos(\cos(\tilde{x}, \tilde{A}_1)), \dots, \arccos(\cos(\tilde{x}, \tilde{A}_K))] \in [0, \pi]^K$ over 18 canonical Clover Semantic Anchors.
   - Verified on `27,000 / 27,000` triples with `0` triangle-inequality violations (`identity_max_error = 3.33e-8`), boosting training-free cross-module retrieval (`S2 BGE 384d -> C1 Layer 18 2048d`) by **`6.0×` Top-1 (`5.56% -> 33.33%`)** and **`5.0×` Top-3 (`11.11% -> 55.56%`)**, while preserving `2.17×` higher structural similarity under $\sigma = 1.5$ Gaussian noise (`0.7409` vs `0.3421`).
3. **Context — 3-Layer Neuro-Symbolic Memory & 4-Graph Traversal (`NeuroSymbolicMultiGraphMemory`)** (*Jiang et al., 2026, NS-Mem; Jiang et al., ACL 2026, MAGMA*):
   - Organizes experiences across `Episodic` ($\mathcal{L}_{\text{epi}}$), `Semantic` ($\mathcal{L}_{\text{sem}}$), and `Procedural Logic DAG` ($\mathcal{L}_{\text{logic}}$) layers with 4 orthogonal relation graphs (`Temporal`, `Causal`, `Semantic`, `Entity`).
   - Scores Logic Nodes via dual-level Index Vectors ($\alpha = 0.3$ goal + $0.7$ step, updated via EMA $\beta = 0.9$), tracks Beta-Binomial node reliability and Dirichlet-Multinomial transition posteriors, and runs deterministic symbolic queries (`getProcedureWithEvidence`, `queryStepSequence(goal, C)`, `aggregateEntityBehaviors`) to filter out infeasible action paths when world constraints change.
4. **Interpretation & Bidirectional Neuro-Symbolic Alignment (`I_DAT` + `BiNS` + `DualMemory` + Dharma)** (*Odense & d'Avila Garcez, AIJ 2025; Wang et al., EMNLP 2026; Wen et al., EMNLP 2026; Saglam et al., 2025*):
   - Latent intent and Dharma discernment (`concept_only` vs `safe_tool` vs `risky_ask`) reach **`97.22%` 6-fold CV accuracy at Layer 8 (`sliding_gqa`)** (`94.44%` linear probe at Layer 4) vs `30.56%` for naive keyword triggers.
   - `verify_idat_semantic_encoding()` verifies Distributed-Atoms (`I_DAT`) least fixed-point soundness ($\emptyset \subset \mathcal{M}_N \subseteq \mathcal{M}_L$) and completeness ($\mathcal{M}_N = \mathcal{M}_L$), and `bidirectional_bins_diagnose()` implements `Mode i` (Conflict/Feasibility Rectification) and `Mode ii` (Deductive Deadlock Auxiliary Hypothesis Generation).
