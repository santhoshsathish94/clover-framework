---
{
  "schema_version": "1.0",
  "id": "MEM-CORR-HYBRID-RRF-SYMBOLIC-SEPARATION-001",
  "title": "Correction: Hybrid Dense+BM25 RRF Seed Linking and One-Way Symbolic Traversal Outperform Pure Dense RAG (`84.5%` vs. `37.3%` Recall)",
  "entry_type": "correction",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "symbolic-neural-separation",
    "hybrid-rrf-fallback",
    "ontology-guided-graphrag",
    "dense-rag-baseline",
    "hard-threshold-seed-dropout"
  ],
  "tags": [
    "ontorag",
    "graphrag",
    "rrf",
    "symbolic",
    "correction",
    "verified",
    "benchmarks"
  ],
  "created_at": "2026-10-25T17:00:00Z",
  "updated_at": "2026-10-26T04:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "docs/clover-ontology-graphrag-research-context.md#section-4",
    "actor": "human_and_assistant",
    "recorded_at": "2026-10-25T17:00:00Z"
  },
  "relationships": [
    {
      "type": "corrects",
      "target": "MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#step-5",
      "confidence": 1.0,
      "context_scope": "neuro_symbolic_graphrag_architecture",
      "created_at": "2026-10-25T17:00:00Z",
      "note": "Supersedes pure dense vector retrieval with hybrid RRF seed linking + deterministic symbolic graph traversal."
    },
    {
      "type": "challenges",
      "target": "MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#step-5",
      "confidence": 1.0,
      "context_scope": "embedding_only_flattening",
      "created_at": "2026-10-25T17:00:00Z",
      "note": "Refutes the claim that flat dense chunks preserve multi-hop causal provenance (`0.0` paths vs `37.1` paths)."
    },
    {
      "type": "derived_from",
      "target": "MEM-OBS-HARD-COSINE-THRESHOLD-DROPOUT-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#finding-3",
      "confidence": 1.0,
      "context_scope": "code_to_concept_linking",
      "created_at": "2026-10-25T17:00:00Z",
      "note": "Eliminates the 9/30 (30%) Practical code query seed dropout via unthresholded dense+BM25 RRF fallback."
    },
    {
      "type": "supports",
      "target": "MEM-CORR-STRATIFIED-PRIORITY-TRAVERSAL-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#section-4.1",
      "confidence": 1.0,
      "context_scope": "neuro_symbolic_graphrag_architecture",
      "created_at": "2026-10-25T17:00:00Z",
      "note": "Combines hybrid RRF seed recovery with stratified priority traversal for end-to-end grounded retrieval."
    },
    {
      "type": "supports",
      "target": "MEM-OUT-RUST-GPU-PARITY-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#section-4.2",
      "confidence": 1.0,
      "context_scope": "k3_benchmark_suite_verification",
      "created_at": "2026-10-25T20:00:00Z",
      "note": "Directly contributes the Ontology-Guided GraphRAG (`V5`) layer in the 100% 5-benchmark verification suite."
    }
  ],
  "uncertainty_notes": [
    "Lexical fallback on long narrative passages (>100 words) requires query-coverage density filtering (`overlap >= 4` and `overlap / len(q_words) >= 0.20`) so generic 3-letter words do not trigger out-of-domain STEM concepts."
  ],
  "revisions": [
    {
      "timestamp": "2026-10-25T17:00:00Z",
      "actor": "human_and_assistant",
      "action": "created",
      "reason": "Initial correction entry recorded from docs/clover-ontology-graphrag-research-context.md#section-4"
    }
  ]
}
---

# Correction: Hybrid Dense+BM25 RRF Seed Linking and One-Way Symbolic Traversal Outperform Pure Dense RAG (`84.5%` vs. `37.3%` Recall)

## Summary
Coupling unthresholded dense+BM25 Reciprocal Rank Fusion (RRF) seed linking with deterministic one-way symbolic graph traversal eliminates all `10/60` empty-evidence failures (`0/30` on both Exploratory and Practical), raises Exploratory Gold ID Recall from `37.3%` (`vanilla_rag`) to `84.5%` (`37.1` `[P#]` paths), and lifts Clover Kimi K3 (`V5`) to `100%` on GSM8K, MMLU, HumanEval, and HybridAgent QA.

## Content
Corrects `MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001` and resolves the `30%` code-query seed dropout in `MEM-OBS-HARD-COSINE-THRESHOLD-DROPOUT-001`. Pure dense vector retrieval flattens typed cross-repository edges (`0.0` `[P#]` relationship paths) and fails when raw code or Unicode-hyphenated queries fall below a fixed cosine threshold. By combining (1) Unicode hyphen and grouped-identifier normalization, (2) dense+BM25 RRF fallback with a lexical density guard (`>= 4`-char domain words, `overlap >= 4`, coverage density `>= 0.20`), and (3) one-way symbolic graph traversal with `[E#]` / `[P#]` provenance and `CloverSymbolicVerifier`, Clover Kimi K3 (`V5`) improved Pooled 47-task accuracy from `29.12%` (`V1`) and `88.02%` (`V4`) to `98.66%` (`V5`).

## Original Source Evidence
1. 3-Way Retrieval Benchmark (`bench_retrieval_3way.py`, `gpu`):
   - Exploratory (`N=30`): `vanilla_rag` `37.3%` Gold ID Recall (`0.0` paths) vs. `ontorag_arm_b` `84.5%` (`37.1` paths, `0/30` empty).
   - Practical Code (`N=30`): `vanilla_rag` `16.4%` (`5/30` empty), `ontorag_arm_a` `25.0%` (`9/30` empty) vs. `ontorag_arm_b` `29.8%` (`0/30` empty, `30/30` CWE coverage).
2. Clover Kimi K3 5-Way Ablation (`docs/clover-ontology-graphrag-research-context.md#section-4.2`):
   - GSM8K `80% -> 100% (20/20)`, MMLU `80% -> 100% (20/20)`, HumanEval `95% -> 100% (20/20)`, HybridAgent QA `93.3% -> 100% (15/15)`.

## Revision History
- `2026-10-25T17:00:00Z` (**human_and_assistant**, `created`): Initial correction entry recorded from docs/clover-ontology-graphrag-research-context.md#section-4
