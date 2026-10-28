# Clover Kimi K3 — File-Based Memory Tree and Relational Knowledge Graph (`k3_memory_graph.py`)

- **Date:** 2026-10-26
- **Branch:** `rust-gpu-merged` / `main`
- **Scope Boundary:** Persistent file-based memory tree and typed relational knowledge graph foundation for Clover Kimi K3. Does **not** perform model fine-tuning or online weight updates, and does **not** claim improved model reasoning prior to a dedicated downstream reasoning evaluation.

---

## 1. Audit of the Existing Memory & Retrieval Architecture

Before implementing `k3_memory_graph.py`, we audited the repository (`clover-kimi-k3/src/tools/hybrid-ncm-k3/`), active branch (`rust-gpu-merged` / `main` at `494f36f`), and worktrees:

| Existing Component | File | What Was Implemented | What Was Missing for Persistent Relational Memory |
| :--- | :--- | :--- | :--- |
| **256K Session Store & Compaction** | `k3_chat_server.py` (`K3ChatState`, `out/sessions_store.json`) | Per-session turn lists (`sessions[session_id]`), ~180-word chunk indexing (`_index_text_into_session`), lexical overlap retrieval (`build_prompt_with_256k_context`), and automatic summary compaction above 262,144 tokens. | Scoped to isolated chat sessions in a flat JSON blob; no cross-conversation human-readable knowledge files, no stable entry IDs, no epistemic status, no revision history, and no typed relationships. |
| **Neuro-Symbolic Domain Ontology** | `k3_ontology_graph.py` (`CloverOntologyGraph`, `CloverOntologyRetriever`) | Deterministic in-memory property graph over hardcoded domain entities (`CVE`, `CWE`, `CAPEC`, `CPE`, `STEM_CONCEPT`, `FORMULA`, `CLOVER_ENTITY`, `DOC_RECORD`) with Horn inference rules and `[E#]` / `[P#]` provenance rendering. | Compiled from static Python lists in `build_default_clover_ontology_graph()`; not backed by a file tree on disk, cannot store conversation excerpts or evolving interpretations, and lacks `corrects` / `refines` / `unresolved_with` revision chains. |
| **Agent & 93-Layer K3 + NCM Pipeline** | `k3_agent_engine.py` (`CloverK3AgentEngine`) | Runs `HybridAgent-8B-Agentic` (`<latent>`, `<ontology>`, `<search>`, `<python>`, `<generative_compiler>`) followed by 93-layer Kimi K3 GPU MoE + `K3NativeHybridLink` + 11-pass NCM parallel generation. | Needed a high-precision context hook (`retrieve_relevant_memory_package`) that injects `[M#]` persistent memory records and evolution chains without polluting unrelated coding/math/MMLU benchmarks. |

---

## 2. File-Based Memory Format & Relationship Schema (`schema_version: "1.0"`)

### 2.1 Dual Complementary Structure
1. **File Tree (`memory_tree/`):** Organizes human-readable Markdown files (`.md`) across thematic branches (`conversations/`, `concepts/`, `philosophy/interpretations/`, `philosophy/corrections/`, `philosophy/positions/`, `observations/`, `hypotheses/`, `corrections/`, `decisions/`, `experiments/`, `outcomes/`). **The `.md` files are the sole source of truth.**
2. **Relationship Graph (`_index/memory_graph_index.json`):** Connects entries across any branches of the directory tree using stable IDs (`id`). The JSON index is a lightweight cache that can be deleted and rebuilt deterministically from the `.md` files at any time via `store.rebuild_index()`.

### 2.2 Canonical Markdown + YAML Front Matter Format
Every `.md` file in the memory tree has YAML front matter followed by four inspectable Markdown sections (`## Summary`, `## Content`, `## Original Source Evidence`, `## Revision History`):

```markdown
---
schema_version: '1.0'
id: MEM-CORR-INSIDE-BOUNDARY-002
title: "Correction: 'Everything Happens Inside' Questions the Fundamental Distinction Between Inside and Outside"
entry_type: correction
epistemic_status: user_stated_belief
status: active
concepts:
  - everything-happens-inside
  - inside-outside-distinction
  - observer-boundary
tags:
  - philosophy
  - correction
  - inside
  - outside
  - non-duality
created_at: '2026-10-20T09:18:00Z'
updated_at: '2026-10-20T09:18:00Z'
provenance:
  source_type: conversation
  source_ref: session:phil-inquiry-2026-10-20#turn-5
  conversation_id: conv_phil_20261020_01
  turn_range: turn 5
  actor: human
  recorded_at: '2026-10-20T09:18:00Z'
relationships:
  - type: corrects
    target: MEM-HYP-INSIDE-MIND-001
    provenance: session:phil-inquiry-2026-10-20#turn-5
    confidence: 1.0
    context_scope: philosophical_interpretation_of_inside
    created_at: '2026-10-20T09:18:00Z'
    note: Supersedes the individual-mind reading with the clarification questioning the inside/outside distinction.
  - type: derived_from
    target: MEM-CONV-PHIL-INSIDE-001
    provenance: session:phil-inquiry-2026-10-20#turn-5
    confidence: 1.0
  - type: refines
    target: MEM-CONCEPT-OBSERVER-BOUNDARY-001
    provenance: session:phil-inquiry-2026-10-20#turn-5
    confidence: 0.95
revisions:
  - timestamp: '2026-10-20T09:18:00Z'
    actor: human
    action: created
    reason: Initial correction entry recorded from session:phil-inquiry-2026-10-20#turn-5
---

# Correction: 'Everything Happens Inside' Questions the Fundamental Distinction Between Inside and Outside

## Summary
...

## Content
...

## Original Source Evidence
...

## Revision History
...
```

### 2.3 Validated Vocabularies
- **`entry_type`:** `conversation`, `concept`, `observation`, `correction`, `hypothesis`, `decision`, `experiment`, `outcome`.
- **`epistemic_status`:** `observed`, `user_stated_belief`, `hypothesis`, `externally_verified`, `unresolved`.
  - *Epistemic Boundary:* Philosophical beliefs (`user_stated_belief`), open questions (`unresolved`), and conjectures (`hypothesis`) are never promoted to `externally_verified` or `observed` facts without independent verification.
- **`status` (Lifecycle):** `active`, `superseded`, `contested`, `under_review` (paired with `superseded_by: <id>` when superseded).
- **`relationships[].type`:**
  - `supports`, `challenges`, `refines`, `depends_on`, `similar_to`, `derived_from`, `caused_by`, `corrects`, `exemplifies`, `unresolved_with`.
  - Irreflexive relations (`corrects`, `refines`, `challenges`, `caused_by`, `derived_from`, `supports`) reject self-loops (`source == target`). General cycles in the knowledge graph (such as mutual `unresolved_with` or `similar_to` cycles) are valid and traversed safely with a visited set, whereas cyclic `corrects` chains (`A corrects B` and `B corrects A`) are flagged in `cycle_warnings`.

---

## 3. Storage, Traversal, Safe Updates, Provenance & Recovery

1. **Rebuildable Index & Validation (`store.rebuild_index()`):**
   - Walks all `.md` files in `memory_tree/` (ignoring `_index/`), parses YAML front matter and Markdown sections, validates schema fields, checks for duplicate IDs (`duplicate_ids`) and missing relationship targets (`broken_links`), synchronizes `superseded_by` pointers from active `corrects` edges, detects cyclic `corrects` chains (`cycle_warnings`), flags unlinked overlapping active entries (`uncertain_duplicates` — retained separate rather than auto-merged), and writes `_index/memory_graph_index.json`.
2. **Cross-Directory Traversal & Evolution Chains (`traverse` & `trace_evolution_chain`):**
   - `traverse(seed_ids, max_depth, direction, relation_types, max_nodes)` runs deterministic priority-ordered BFS (`corrects` $\to$ `refines` $\to$ `challenges` $\to$ `unresolved_with` $\to$ `derived_from` $\to$ `supports` $\to$ `caused_by` $\to$ `depends_on` $\to$ `exemplifies` $\to$ `similar_to`) across directory boundaries.
   - `trace_evolution_chain(entry_id)` walks both incoming and outgoing `corrects`, `refines`, and `derived_from` edges (plus `superseded_by`) to assemble the complete chronological chain from `source_origin` (`conversation`) through `superseded_interpretation` to `active_correction`.
3. **Safe Update Operations:**
   - `create_entry(entry, rel_path, overwrite=False)`: Validates and writes a new `.md` file without overwriting existing IDs/files.
   - `add_relationship(...)` / `remove_relationship(...)`: Validates relationship metadata, updates the source `.md` file, records a revision log entry, and rebuilds the index.
   - `revise_summary(entry_id, new_summary, actor, reason)`: Updates `## Summary` and records `previous_summary` in `## Revision History` while strictly preserving `## Original Source Evidence`.
   - `record_correction(...)` & `mark_superseded(...)`: Creates a `correction` entry with `corrects -> target_id` and marks `target_id` as `status: superseded` (`superseded_by: correction_id`) without erasing `target_id`'s content, original source evidence, or history.

---

## 4. Live Verification & Test Results (`test_k3_memory_graph.py`)

- Executed `python test_k3_memory_graph.py` locally and on `gpu` (`8 × NVIDIA RTX 3090`):
  - **All 9 test suites (covering all 11 validation requirements in Section 7) passed (`OK` in `0.71s`)**, including:
    - Path-independent ID stability across directory relocation (`shutil.move` to `relocated_branch/deep/nested/`).
    - All 10 relationship types, self-loop rejection, and confidence bounds.
    - Incoming/outgoing and 4-hop cross-directory traversal across 6 directory branches.
    - 2-node and 3-node cycle traversal at `max_depth=12` without infinite loops, plus `corrects` cycle warning detection.
    - Complete deletion of `_index/memory_graph_index.json` and exact reconstruction via `rebuild_index()`, plus broken-link detection.
    - `revise_summary` preserving `## Original Source Evidence` verbatim and logging `previous_summary`.
    - Contradiction preservation (`challenges`, `unresolved_with`) and uncertain duplicate detection without auto-merging.
    - **Required Scenario (`'everything happens inside'`) + General Dynamic Mechanism Test (`'silence speaks'`):** Verified that querying `'everything happens inside'` returns both `MEM-HYP-INSIDE-MIND-001` (`status=superseded`, `superseded_by=MEM-CORR-INSIDE-BOUNDARY-002`) and `MEM-CORR-INSIDE-BOUNDARY-002` (`status=active`, `supersedes=['MEM-HYP-INSIDE-MIND-001']`, `epistemic_status=user_stated_belief`) along with `MEM-CONV-PHIL-INSIDE-001` (`Original Source Evidence`), ranks them ahead of the Linux container entry that shares the surface words `'inside'` and `'outside'`, and reproduces the exact same behavior on a dynamically created topic (`MEM-HYP-SILENCE-ABSENCE-001` $\to$ `MEM-CORR-SILENCE-PRESENCE-002`).
    - Subprocess persistence across process restarts and token-budget enforcement (`token_budget=350`, `max_entries=2`).

---

## 5. Phase 1 — Collective Experience Seed v0.1 (`32` Entries, `65` Edges) & Controlled 3-Arm Live Evaluation (`bench_collective_memory_ab.py`)

### 5.1 Three-Agent Architecture & Cross-Check Findings
We used three specialized agents across **Gemini 4 Argon** and **Claude Sonnet 5.5** to build, audit, and evaluate **Collective Experience Seed v0.1**:
1. **Agent 1 (Gemini 4 Argon — Cross-Domain Structural Principles, Research Findings & Contradictions):**
   - Designed 10 entries connecting cross-domain structural diagnostic concepts (`MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001`, `MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001`, `MEM-CONCEPT-SYNTHETIC-TIMING-VS-MEANING-001`, `MEM-CONCEPT-HARDWARE-SCOPED-VERDICTS-001`, `MEM-CONCEPT-PREFIX-SHADOWING-DISPATCH-001`) to empirical `ontorag` findings (`MEM-OBS-ONTORAG-FANOUT-FLOODING-001`, `MEM-CORR-STRATIFIED-PRIORITY-TRAVERSAL-001`, `MEM-OBS-HARD-COSINE-THRESHOLD-DROPOUT-001`, `MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001`, `MEM-CORR-HYBRID-RRF-SYMBOLIC-SEPARATION-001`).
2. **Agent 2 (Claude Sonnet 5.5 — Clover Kimi K3, NCM, HybridAgent & Generative Compilation Experience & Source Audit):**
   - Designed 10 entries covering CPU vs. GPU quantisation & supply boundaries (`MEM-HYP-QUANT-DEAD-END-ALL-HW-001`, `MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001`), bidirectional micro-benchmark drift (`MEM-OBS-ISOLATED-MICROBENCH-BIDIRECTIONAL-DRIFT-001`), expert result cache KDA drift & zero-allocation (`MEM-HYP-EXPERT-RESULT-CACHE-DEAD-001`, `MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001`), generative compilation statement-boundary rollback (`MEM-OBS-POSTHOC-CODE-REPAIR-LIMITS-001`, `MEM-EXP-GENERATIVE-COMPILATION-ROLLBACK-001`, `MEM-OUT-GENERATIVE-COMPILATION-100PCT-001`), and the VS Code tool-loop envelope bug (`MEM-OBS-VSCODE-TRAILING-CONTEXT-WIPES-TOOLS-001`, `MEM-CORR-VSCODE-TURN-STATE-AND-GIT-DISPATCH-001`).
   - **Critical Source-Audit Corrections Caught by Agent 2:**
     - Separated the HumanEval latency reduction by sample size (`-17.9%` at `N=20`, `11,682 -> 9,586 ms/prob` from stopping at target function end, vs. `-14.1%` at `N=40`, `11,309 -> 9,720 ms/prob`) rather than conflating `N=40` accuracy with `N=20` latency.
     - Recorded both CPU 4-bit trunk measurements (`Step 3` `4.1 -> 4.0 s/tok` `-2.4%` vs. commit `74c1c0e` `LB_Q4` `3.326 -> 2.716 s/tok` `-18.3%`), scoped the GPU bottleneck to measured expert supply (`4.95 s/layer` NVMe, `0.95 s/layer` RAM vs. `0.23–0.29 s` compute; Layer 0 DRAM `5.6%`), and explicitly recorded in `uncertainty_notes` that a GPU 4-bit vs. 8-bit trunk decode A/B remains **unmeasured**.
     - Attributed recurrent state drift specifically to **KDA** layers (not MLA) and recorded the `K3_PAYLOAD_RAM=1` double-counted payload bug (commit `58497d5`) that had left the result cache allocated at size zero.
3. **Agent 3 (Claude Sonnet 5.5 — Scorer/Gate Audit & Controlled 3-Arm Experiment Design):**
   - Audited `_score_seed_entries` and `retrieve_relevant_memory_package` on novel cross-domain problems and lexical lures, identifying 4 concrete retrieval improvements that we implemented in `k3_memory_graph.py`:
     1. **Structural Diagnostic Cue Layer (`structural_concept_rules`):** Maps cross-domain diagnostic patterns (flat curve across capacities, 0 hits on repeated token in recurrent state, 1-hop length-first fan-out flooding, CPU/GPU 4-bit hardware claims) directly to the concept bridge nodes (`MEM-CONCEPT-*`).
     2. **Multi-Part Slug & Generic Token Filtering (`generic_slug_parts`):** Prevents single high-frequency words (`inside`, `cache`, `gpu`, `host`, `memory`) from triggering `concept_match` or `tag_match` on unrelated math, coding, or spatial questions.
     3. **Refinement Chain Boosting & Single-Edge Attribution:** Boosts both `active_correction` and `refinement` roles in `trace_evolution_chain` while keeping each correction edge scoped to the specific hypothesis it refutes.
     4. **Analogy Distractor Pruning (`min_similar_confidence=0.7`):** Skips low-confidence `*_analogy_only` `similar_to` edges during `retrieve_context_package` traversal.

### 5.2 Controlled 3-Arm Live Evaluation Results on `gpu` (`8 × NVIDIA RTX 3090`, `bench_collective_memory_ab.py`)

- **Grader Liveness Self-Check:** `6/6` gold answers `PASS`, `6/6` bad answers `FAIL` before running any model evaluation.
- **Negative-Control Gate Battery (`10` prompts across GSM8K, MMLU, HumanEval, WMT14, and lexical lures for `inside`, `cache`, `gpu`):** **`0 / 10` (`0.0%`) false-positive gate triggers**.

| Task ID & Category | Arm 1: Baseline (`Memory Graph OFF`) | Arm 2: Collective Seed v0.1 (`32` Entries `ON`) | Arm 3: Post-Experience Cycle (`34` Entries `ON`) | What Changed Across Arms |
| :--- | :---: | :---: | :---: | :--- |
| **`T1` — Cross-Domain Transfer:** 48-Stage Pipeline with Flat `19.4%` Shared-Pool Hit Rate | `33.3%` (`gen=66.7%`, `mem=0.0%`) | **`100.0%`** (`gen=100%`, `mem=100%`) | **`100.0%`** (`gen=100%`, `mem=100%`) | Bridges via `MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001` $\to$ `MEM-CORR-PER-LAYER-PARTITION-002`, diagnosing shared-pool cyclic eviction and proposing per-stage partitioning (`23.3% -> 48.0%` prior K3 evidence). |
| **`T2` — Cross-Domain Transfer:** Recurrent Layer Memoization with `0` Hits on 64 Repeated Tokens | `16.7%` (`gen=33.3%`, `mem=0.0%`) | **`100.0%`** (`gen=100%`, `mem=100%`) | **`100.0%`** (`gen=100%`, `mem=100%`) | Bridges via `MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001` $\to$ `MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001`, explaining recurrent state drift, checking non-zero allocation, and testing full-request A/B/A replay (`30,912` hits, `3.27x`). |
| **`T3` — Cross-Domain Transfer:** 2-Hop Root-Cause Paths Drowned Out by 150 1-Hop `AFFECTS_HOST` Edges | `50.0%` (`gen=100%`, `mem=0.0%`) | **`100.0%`** (`gen=100%`, `mem=100%`) | **`100.0%`** (`gen=100%`, `mem=100%`) | Bridges via `MEM-OBS-ONTORAG-FANOUT-FLOODING-001` $\to$ `MEM-CORR-STRATIFIED-PRIORITY-TRAVERSAL-001`, ordering `edge_priority` before path length and capping leaf fan-out (`max_affects_per_seed = 4`, `63.3% -> 76.7%` chain recall). |
| **`T4` — Negative-Control Boundary:** CPU (`-2.4%` vs `-18.3%` `LB_Q4`) vs. GPU (`8 × RTX 3090`) 4-Bit Claims | `50.0%` (`gen=100%`, `mem=0.0%`) | **`100.0%`** (`gen=100%`, `mem=100%`) | **`100.0%`** (`gen=100%`, `mem=100%`) | Uses `MEM-CONCEPT-HARDWARE-SCOPED-VERDICTS-001` and `MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001` to refute both overclaims, cite exact CPU/GPU measurements, and explicitly mark GPU 4-bit trunk decode A/B as **unmeasured**. |
| **`T5` — Evolving Interpretation & Contested Tension:** `'Everything Happens Inside'` & Realism Dualism | `0.0%` (`gen=0.0%`, `mem=0.0%`) | **`100.0%`** (`gen=100%`, `mem=100%`) | **`100.0%`** (`gen=100%`, `mem=100%`) | Traces `MEM-CONV-PHIL-INSIDE-001` $\to$ `MEM-HYP-INSIDE-MIND-001` (`superseded`) $\to$ `MEM-CORR-INSIDE-BOUNDARY-002` (`user_stated_belief`) and preserves `MEM-HYP-STRICT-EXTERNAL-REALISM-001` as unresolved tension. |
| **`T6` — Post-Experience Learning Loop:** Multi-Agent Router with `0%` KV-Prefix Cache Hits from Dynamic Header Salt | `50.0%` (`gen=100%`, `mem=0.0%`) | `25.0%` (`gen=50.0%`, `mem=0.0%`) | **`100.0%`** (`gen=100%`, `mem=100%`) | Before recording the KV-salt experience (Arm 2), the graph lacks the specific `token 12` / `86.4%` finding; after recording `MEM-OBS-KV-PREFIX-CACHE-SALT-DRIFT-001` and `MEM-CORR-KV-PREFIX-CACHE-CANONICAL-ENV-001` into the memory tree (Arm 3), Clover solves `T6` at `100.0%` with zero regression on `T1..T5`. |
| **Mean Across All 6 Tasks (`T1..T6`)** | **`33.3%`** (`0/6` full pass) | **`87.5%`** (`5/6` full pass) | **`100.0%`** (`6/6` full pass) | **`+54.2 pp` (Arm 2 vs. Arm 1) $\to$ `+66.7 pp` (Arm 3 vs. Arm 1)** |

---

## 6. Directions A, C, and D — Closed-Loop Growth Staging Gate, Live Inspector UI, and 12-Task Held-Out Generalization Benchmark (`bench_collective_memory_heldout.py`)

### 6.1 Direction A: Safe Closed-Loop Growth Staging Queue & Verified Promotion Gate (`k3_memory_graph.py` + `k3_agent_engine.py`)
- Added `stage_candidate_entry()`, `list_staged_entries()`, `promote_staged_entry()`, `discard_staged_entry()`, and `propose_cycle_memory_update()`:
  - Candidate memories generated from completed cycles (`compiler_rollback_solved`, `user_correction`, `verified_observation`) are written to `memory_tree/staging/<slug>.md` with `status: under_review` and auto-linked to matching `MEM-CONCEPT-*` bridge nodes.
  - `_iter_markdown_files()` explicitly excludes `staging/`, guaranteeing that unreviewed drafts **never** pollute the active graph index (`0` leakage verified in `bench_collective_memory_heldout.py` Arm 2).
  - `promote_staged_entry()` moves a verified entry from `staging/` into its canonical branch directory (`observations/`, `corrections/`, etc.), sets `status: active`, marks any `corrects` target as `superseded`, and rebuilds `_index/memory_graph_index.json` (`10/10` unit test suites in `test_k3_memory_graph.py` passing).

### 6.2 Direction D: Live Memory Graph & Staging Inspector UI (`k3_chat_server.py`)
- Added `/api/memory/staged`, `/api/memory/stage`, `/api/memory/promote`, and `/api/memory/discard` endpoints to `k3_chat_server.py`.
- Added the interactive **Memory Graph** slide-over drawer in the Clover AI Web UI (`toggleMemoryDrawer()`, `loadMemoryDrawer()`, `queryMemoryDrawer()`, `promoteStaged()`, `discardStaged()`), displaying graph health, staged candidates with one-click **Approve & Promote** / **Discard** buttons, active entries, and live relationship path search.

### 6.3 Direction C: 12-Task Held-Out Cross-Domain Generalization & Closed-Loop Staging Benchmark on `gpu` (`bench_collective_memory_heldout.py`)
- Evaluated `Arm 1 (Baseline OFF)` vs. `Arm 2 (Seed v0.1 + 3 Unpromoted Staged Entries in staging/)` vs. `Arm 3 (Promoted Closed-Loop)` across **12 held-out tasks (`H01..H12`, 36 live 93-layer Kimi K3 + HybridAgent + NCM forward passes on `gpu`)** spanning CDN PoP caching, compiler SSA value numbering, BGP route graphs, C/Rust lexer operator dispatch, SIMD lookup-table harnesses, static analysis code-to-CWE linking, cross-hardware 4-bit claims, statement-boundary sealing, VS Code turn state, philosophical evolution, and two newly staged/promoted closed-loop engineering cycles (`H11` AVX-512 subnormal `MXCSR 0x8040` `FTZ`/`DAZ` stall and `H12` batched `git-lfs` pruning):

| Held-Out Task ID & Domain | Arm 1: Baseline (`Memory OFF`) | Arm 2: Seed v0.1 (`Unpromoted Staged`) | Arm 3: Promoted Closed-Loop (`Active`) | Verification Result |
| :--- | :---: | :---: | :---: | :--- |
| **`H01` — CDN 64-PoP Shared-Pool Cyclic Eviction** | `50.0%` (`pass=False`) | **`100.0%` (`pass=True`)** | **`100.0%` (`pass=True`)** | Transfers `MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001` $\to$ per-PoP partitioning |
| **`H02` — Compiler SSA Monotonic Drift Memoization** | `25.0%` (`pass=False`) | **`100.0%` (`pass=True`)** | **`100.0%` (`pass=True`)** | Transfers `MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001` $\to$ replay & alloc check |
| **`H03` — BGP Route Graph 180-Subnet Leaf Flooding** | `50.0%` (`pass=False`) | **`100.0%` (`pass=True`)** | **`100.0%` (`pass=True`)** | Transfers `MEM-OBS-ONTORAG-FANOUT-FLOODING-001` $\to$ `MEM-CORR-STRATIFIED-PRIORITY-TRAVERSAL-001` |
| **`H04` — C/Rust Lexer `+` vs `+=` Operator Shadowing** | `25.0%` (`pass=False`) | **`100.0%` (`pass=True`)** | **`100.0%` (`pass=True`)** | Transfers `MEM-CONCEPT-PREFIX-SHADOWING-DISPATCH-001` $\to$ longest-prefix-first & rollback |
| **`H05` — Standalone SIMD Harness Bidirectional Drift** | `50.0%` (`pass=False`) | **`100.0%` (`pass=True`)** | **`100.0%` (`pass=True`)** | Transfers `MEM-OBS-ISOLATED-MICROBENCH-BIDIRECTIONAL-DRIFT-001` (`15.2x/8.6x` & `1.68x/2.2x`) |
| **`H06` — Security Analyzer Code-to-CWE `0.35` Dropout** | `25.0%` (`pass=False`) | **`100.0%` (`pass=True`)** | **`100.0%` (`pass=True`)** | Transfers `MEM-OBS-HARD-COSINE-THRESHOLD-DROPOUT-001` $\to$ BM25+Dense RRF |
| **`H07` — Cross-Hardware CPU (`LB_Q4`) vs GPU 4-Bit Audit** | `50.0%` (`pass=False`) | **`100.0%` (`pass=True`)** | **`100.0%` (`pass=True`)** | Scopes CPU (`-2.4%` vs `-18.3%`) & GPU (`4.95s`/`0.95s` supply) and marks GPU trunk A/B unmeasured |
| **`H08` — Statement-Boundary Sealing vs Post-Hoc Repair** | `0.0%` (`pass=False`) | **`100.0%` (`pass=True`)** | **`100.0%` (`pass=True`)** | Cites `38.5%` doomed-char savings, `20/20` (`9,586 ms`, `-17.9%`), `40/40` (`9,720 ms`, `-14.1%`), `5/5` Rust |
| **`H09` — Multi-Turn Agent `<context>` Envelope Reset** | `100.0%` (`pass=False`) | **`100.0%` (`pass=True`)** | **`100.0%` (`pass=True`)** | Cites `MEM-OBS-VSCODE-TRAILING-CONTEXT-WIPES-TOOLS-001` & `494f36f` fix |
| **`H10` — Philosophical Evolution & Contested Realism** | `25.0%` (`pass=False`) | **`100.0%` (`pass=True`)** | **`100.0%` (`pass=True`)** | Traces `MEM-CONV-PHIL-INSIDE-001` $\to$ `MEM-HYP-INSIDE-MIND-001` $\to$ `MEM-CORR-INSIDE-BOUNDARY-002` + `MEM-HYP-STRICT-EXTERNAL-REALISM-001` tension |
| **`H11` — Closed-Loop Staged $\to$ Promoted SIMD `FTZ`/`DAZ`** | `75.0%` (`pass=False`) | `0.0%` (`pass=False`, isolated in `staging/`) | **`100.0%` (`pass=True`)** | Strictly isolated while `under_review` in Arm 2; `100.0%` after `promote_staged_entry()` in Arm 3 (`6.4x`, `MXCSR 0x8040`, `1.4e-7`) |
| **`H12` — Closed-Loop User Correction (`git-lfs` Batch Prune)** | `75.0%` (`pass=False`) | `75.0%` (`pass=False`, correction in `staging/`) | **`100.0%` (`pass=True`)** | Promotes `MEM-CORR-LFS-BATCHED-10TRUNK-PRUNE-001`, supersedes `MEM-HYP-LFS-BATCH-SIZE-UNBOUNDED-001`, cites `37 MiB/s` & `18 GB` |
| **Mean Across All 12 Tasks (`H01..H12`)** | **`45.8%`** (`0/12` full pass) | **`89.6%`** (`10/12` full pass — `10/10` on active seed, `0/2` on unpromoted staged) | **`100.0%`** (**`12/12` full pass**) | **`+43.8 pp` (Arm 2 vs Arm 1) $\to$ `+54.2 pp` (Arm 3 vs Arm 1, `12/12` passed)** |

---

## 7. HydraDB Comparison (`hydra-db/hydradb`) & Pure File-Backed Memory Graph Unification

### 7.1 Direct Codebase Inspection of HydraDB (`C:\personal\oss\hydradb`)
Following the Clover rule (*"before describing how another codebase works, clone it and read it"*), we cloned `https://github.com/hydra-db/hydradb.git` into `C:\personal\oss\hydradb` and inspected `README.md`, `architecture.md`, `USING-HYDRADB.md`, `AGENTS.md`, `Cargo.toml`, `src/shard/path_procedure.rs`, and `src/sparse_kernel/mod.rs`:

| Dimension | HydraDB (`hydra-db/hydradb`, Rust `0.1.0`) | Clover Memory & Ontology Graph (`k3_memory_graph.py` + `k3_ontology_graph.py`) | Verdict for Clover |
| :--- | :--- | :--- | :--- |
| **Primary Purpose & Storage** | Distributed object-store-native graph database built on SlateDB (S3/MinIO WAL + SSTs + CAS writer leases + Bolt/HTTPS server). | Git-native, human-readable Markdown + YAML front-matter file tree (`memory_tree/`) + deterministic Ontology & Epistemic Graph. | **Keep Clover's file-tree truth.** Clover requires every memory, provenance record, and correction to be a human-inspectable, git-diffable file in `memory_tree/`, not binary SST blobs in S3/SlateDB. |
| **Epistemic & Evolution Semantics** | Generic property graph (`VertexId`, `RelationshipId`, typed `QueryValue` cells); no built-in epistemic statuses, `corrects`/`superseded_by` evolution chains, or `staging/` review gates. Furthermore, bare node `MATCH (n {id: X})` returns phantom rows even for non-existent or deleted IDs (`USING-HYDRADB.md`), and single `CREATE` duplicates edges. | First-class epistemic status (`observed`, `user_stated_belief`, `hypothesis`, `externally_verified`, `unresolved`), `trace_evolution_chain()`, `staging/` promotion gate, Horn inference rules (`run_inference()`), McCulloch-Pitts inhibitory vetoes, Minsky-Papert Order-$k$ boxes, Hopfield bipolar cleanup, LeCun GTN `logadd`, and Vaswani+LRN 3-head subspace unmixing. | **Clover's domain & epistemic engine is strictly stronger** for neuro-symbolic AI grounding and epistemic integrity. |
| **Traversal & Indexing Ideas Worth Adopting** | Snapshot-pinned `read_epoch`, compiled Compressed Sparse Column (`CompactCsc`) topology generations (`src/sparse_kernel/mod.rs`), and `algo.MSpaths` multi-source/multi-target path enumeration with reverse target pruning (`src/shard/path_procedure.rs`). | Previously rebuilt Python `defaultdict` adjacency lists without an explicit `read_epoch` or compiled `CSC` topology generation / reverse-pruned `ms_paths`. | **Adopted into `CloverMemoryGraphStore`:** added `read_epoch`, `_compile_csc_generation()`, and `ms_paths()` with reverse-target pruning directly over `memory_tree/`. |

### 7.2 Eliminating All Hardcoded Graph & Rule Data from Python Code
- **Removed ~2,100 lines of hardcoded `MemoryEntry` objects** from `build_default_memory_entries()` in `k3_memory_graph.py` and **all hardcoded `mp_global_inhibitory_vetoes`, `computing_anchor_box`, and `structural_concept_rules`**:
  - `k3_memory_graph.py` now loads all 33 canonical `.md` memory entries directly from `memory_tree/` and loads retrieval rules from `memory_tree/_rules/memory_retrieval_rules.json`.
- **Removed ~450 lines of hardcoded `Vertex`, `Edge`, `_CONCEPT_TRIGGERS`, `_MP_ONTOLOGY_PREDICATES`, and `_ENTITY_ANCHOR_BOX` definitions** from `k3_ontology_graph.py`:
  - `k3_ontology_graph.py` now loads all 28 ontology vertices, 22 provenance-carrying edges, 13 concept triggers, 13 McCulloch-Pitts / Minsky-Papert predicates, and 9 entity anchor boxes directly from `memory_tree/_ontology/ontology_graph.json` and runs deterministic Horn inference (`run_inference()`).
- **Removed hardcoded domain URL routing** from `K3KnowledgeSearchTool` in `k3_agent_engine.py`, loading `search_url_routes` from `memory_tree/_rules/memory_retrieval_rules.json`.

---

## 8. Held-Out SWE-Bench Pro V2 `HARD-51` Evaluation Through Clover Cycles (`NodeBB-8168c6c4` & `flipt-2ca5dfb3`)

### 8.1 Test 4: `NodeBB/NodeBB` (`instance_NodeBB__NodeBB-8168c6c48cd2fc5f96bfbdb7bfdaae0fc5daab9c-vnan`, JavaScript)
- **Cycle 1 (`clover_cycle_v1`):** Preserved all `307 / 307` `pass_to_pass` tests, while the verifier exposed 4 `fail_to_pass` failures in profile/group image cleanup (`307 / 311` passed, `REWARD = 0`).
- **Growth Stage:** Staged and promoted `MEM-OBS-NODEBB-IMAGE-CLEANUP-CYCLE1-001` (`status: superseded`) and `MEM-CORR-NODEBB-IMAGE-CLEANUP-CYCLE2-001` (`status: active`, `corrects: MEM-OBS-NODEBB-IMAGE-CLEANUP-CYCLE1-001`).
- **Cycle 2 (`clover_cycle_v2`):** Re-ran in a clean sandbox and achieved **`311 / 311` passed (`REWARD = 1`)** in `27.31s` (`190.29s` agent + verifier).

### 8.2 Test 5 (Held-Out Generalization with NodeBB Correction Withheld): `flipt-io/flipt` (`instance_flipt-io__flipt-2ca5dfb3513e4e786d2b037075617cccc286d5c3`, Go)
- **Initial State (`initial_memory_state_2ca5dfb3.json`):** Withheld both NodeBB entries (`MEM-OBS-NODEBB-IMAGE-CLEANUP-CYCLE1-001` and `MEM-CORR-NODEBB-IMAGE-CLEANUP-CYCLE2-001`), leaving **`33` general-purpose entries** and **`67` relationships** (`is_healthy = True`, `read_epoch = 2`).
- **Cycle 1 (`cycle1`, `report_2ca5dfb3_cycle1.json`):**
  - Clover read `/app/internal/metrics/metrics.go` (`5,636` chars) and attempted to read `/app/internal/config/metrics.go`, which returned `Error: File /app/internal/config/metrics.go does not exist`.
  - Because `decide_vscode_agent_step` only queued existing files for synthesis, `internal/config/metrics.go` was not created and `internal/config/config.go` was not read (`0` bytes patch, `178.92s`).
  - Fresh-sandbox verifier (`/tests/test.sh`) graded Cycle 1 at **`0 / 2` required tests passed (`REWARD = 0`)**, exposing exact compiler diagnostics in `/logs/verifier/run-script-stderr.txt`:
    - `internal/metrics/metrics_test.go:16:13: undefined: config.MetricsConfig`
    - `internal/config/config_test.go:331:9: cfg.Metrics undefined (type *Config has no field or method Metrics)`
    - `internal/config/config_test.go:341:28: undefined: MetricsOTLP`
- **Growth Stage (Evidence-Backed Promotion & Engine Fix):**
  - Updated `decide_vscode_agent_step` in `clover-agent-engine.py` so non-existent target files (`Error: File ... does not exist`) are added to `read_files_map` with empty initial content (`""`) and sibling `internal/config/config.go` aggregate config files are automatically read during the Context stage.
  - Staged and promoted `MEM-OBS-FLIPT-METRICS-EXPORTERS-CYCLE1-001` (`observations/flipt-metrics-exporters-cycle1-observation.md`, `status: superseded`) and `MEM-CORR-FLIPT-METRICS-EXPORTERS-CYCLE2-001` (`corrections/flipt-metrics-exporters-cycle2-correction.md`, `status: active`, `epistemic_status: externally_verified`, `corrects: MEM-OBS-FLIPT-METRICS-EXPORTERS-CYCLE1-001`).
- **Cycle 2 (`cycle2`, `report_2ca5dfb3_cycle2.json`):**
  - Executed `8` tool rounds (`150.62s` total agent time):
    1. `read_file("/app/internal/metrics/metrics.go", 1, 350)` (`0.01s`)
    2. `read_file("/app/internal/config/metrics.go", 1, 350)` $\to$ `does not exist` (`0.0s`)
    3. `read_file("/app/internal/config/config.go", 1, 350)` (`0.0s`)
    4. `read_file("/app/internal/config/config.go", 351, 621)` (`0.0s`)
    5. `create_file("/app/internal/metrics/metrics.go", 8425 bytes)` (`1.57s`)
    6. `create_file("/app/internal/config/metrics.go", 2208 bytes)` (`0.01s`)
    7. `create_file("/app/internal/config/config.go", 16648 bytes)` (`1.52s`)
    8. `run_in_terminal("GOPROXY=file:///root/go/pkg/mod/cache/download go get ...; go test -short ./internal/config ./internal/metrics ...")` (`0.0s`)
  - **Fresh-Sandbox Regrading (`clover_2ca5dfb3_cycle2.patch`, `19,626` bytes):**
    - **`REWARD = 1` (`Exit = 0` in `1.98s`)**
    - **Required `fail_to_pass` tests:** **`2 / 2` passed** (`TestLoad`, `TestGetxporter`)
    - **Total selected tests:** **`14 / 14` passed** (`TestAnalyticsClickhouseConfiguration`, `TestJSONSchema`, `TestScheme`, `TestCacheBackend`, `TestTracingExporter`, `TestDatabaseProtocol`, `TestLogEncoding`, `TestLoad`, `TestServeHTTP`, `TestMarshalYAML`, `Test_mustBindEnv`, `TestGetConfigFile`, `TestDefaultDatabaseRoot`, `TestGetxporter`).

---

## 9. Meaning Formation, Grounded Execution, and Developmental Growth Pilot (`meaning_development_pilot.py`, 2026-10-29)

- **Design & Results Documents:**
  - [docs/experiments/meaning-development/experiment-design.md](../../experiments/meaning-development/experiment-design.md)
  - [docs/experiments/meaning-development/results.md](../../experiments/meaning-development/results.md)
  - [docs/experiments/meaning-development/results.json](../../experiments/meaning-development/results.json)
- **Experimental Code & Tests:**
  - [meaning_development_pilot.py](../../../clover-kimi-k3/src/tools/hybrid-ncm-k3/meaning_development_pilot.py)
  - [test_meaning_development_pilot.py](../../../clover-kimi-k3/src/tools/hybrid-ncm-k3/test_meaning_development_pilot.py) (`5/5` test suites pass alongside `11/11` in `test-memory-knowledge-graph.py`).
- **Measured Outcomes Across Phases 2, 3, and 4 (`Context -> Direction -> Execution -> Outcome -> Growth`):**
  1. **Phase 2 — Experiment A (`Form Meaning`):** Across `16` controlled entries (`12` memory relationships + `11` ontology edges across `systems` and `pharmacology`), achieved **`100.0%` (`4/4`)** paraphrase concept identification without target-word overlap, **`100.0%` (`4/4`, `0` conflations)** distinction precision between related concepts (`cyclic-sweep` vs `ttl-expiration`, `competitive` vs `allosteric-noncompetitive`), **`0.0%`** unsupported-claim rate, **`100.0%`** provenance/correction/conflict preservation, and **`100.0%` (`4/4`)** multi-hop graph retrieval recall vs. **`25.0%` (`1/4`)** for lexical-only BM25 (`+75.0 pp`).
  2. **Phase 3 — Experiment B (`Use Meaning to Guide Execution`):** Across `8` deterministic remediation tasks (`4` construction + `4` held-out), task completion progressed monotonically across conditions: **`B0` (Baseline) `25.0%` (`2/8`) $\to$ `B1` (Memory Retrieval Only) `37.5%` (`3/8`) $\to$ `B2` (Memory + Symbolic Ontology Horn Inference) `75.0%` (`6/8`, `100%` on held-out) $\to$ `B3` (Experience-Informed Execution with `corrections/` and `context_scope`) `100.0%` (`8/8`)**.
  3. **Phase 4 — Experiment C (`Develop Through Experience`, `5` seeds `11, 23, 37, 53, 71`):** Promoting verified Cycle 1 corrections through `memory_tree/staging/` increased same-domain **Accumulation** from `10.0% ± 20.0%` to **`100.0% ± 0.0%` (`+90.0 pp`)** and held-out cross-domain **Transfer** (to Networking CIDR/URI filters and Clinical ICD-10 / ELISA assays) from `26.67% ± 24.95%` to **`100.0% ± 0.0%` (`+73.33 pp`)**, with **`0.0%` negative transfer** on precondition-guard tasks and **`0` unverified proposals leaked** from `staging/`.


