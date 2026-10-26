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
