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
