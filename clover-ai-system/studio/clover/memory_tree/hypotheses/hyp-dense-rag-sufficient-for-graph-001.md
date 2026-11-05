---
{
  "schema_version": "1.0",
  "id": "MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001",
  "title": "Superseded Hypothesis: Dense Vector Similarity Alone Suffices for Cross-Repository Vulnerability & Domain Grounding",
  "entry_type": "hypothesis",
  "epistemic_status": "hypothesis",
  "status": "superseded",
  "superseded_by": "MEM-CORR-HYBRID-RRF-SYMBOLIC-SEPARATION-001",
  "concepts": [
    "dense-rag-baseline",
    "ontology-guided-graphrag",
    "hybrid-rrf-fallback",
    "symbolic-neural-separation"
  ],
  "tags": [
    "rag",
    "embeddings",
    "hypothesis",
    "superseded",
    "graphrag"
  ],
  "created_at": "2026-10-25T10:00:00Z",
  "updated_at": "2026-10-25T17:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "docs/clover-ontology-graphrag-research-context.md#step-5",
    "actor": "system_measurement",
    "recorded_at": "2026-10-25T10:00:00Z"
  },
  "relationships": [
    {
      "type": "similar_to",
      "target": "MEM-HYP-CACHE-DEAD-END-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#step-5",
      "confidence": 0.75,
      "context_scope": "superseded_architectural_baseline",
      "created_at": "2026-10-25T17:00:00Z",
      "note": "Both represent initial architectural assumptions refuted by controlled structural measurements on `gpu`."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-25T10:00:00Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Recorded baseline dense RAG hypothesis (`VanillaRagBaseline`) for 3-way controlled evaluation."
    },
    {
      "timestamp": "2026-10-25T17:00:00Z",
      "actor": "human_and_assistant",
      "action": "mark_superseded",
      "reason": "Superseded by MEM-CORR-HYBRID-RRF-SYMBOLIC-SEPARATION-001 after 3-way benchmark showed 37.3% vs. 84.5% Gold ID recall and 0.0 vs. 37.1 relationship paths."
    }
  ]
}
---

# Superseded Hypothesis: Dense Vector Similarity Alone Suffices for Cross-Repository Vulnerability & Domain Grounding

## Summary
Baseline hypothesis (`VanillaRagBaseline`) that dense embedding similarity (`all-MiniLM-L6-v2`, `top_k=10`, `threshold=0.35`) over exported entity documents is sufficient to retrieve multi-hop `CVE -> CWE -> CAPEC` evidence without explicit symbolic graph traversal or hybrid lexical RRF (superseded by MEM-CORR-HYBRID-RRF-SYMBOLIC-SEPARATION-001).

## Content
Assumed that indexing the shared 2-hop corpus (`1,021` entity documents) into a dense vector store would retrieve the relevant `CVE`, `CWE`, and `CAPEC` records and allow the LLM to infer their relationships implicitly. In controlled 3-way testing on `gpu`, `vanilla_rag` achieved only `37.3%` Gold ID Recall on Exploratory queries (vs. `84.5%` for `ontorag_arm_b`), `16.4%` on Practical code queries (with `5/30` empty-evidence dropouts), and `0.0` relationship paths (`[P#]`).

## Original Source Evidence
Measured in `bench_retrieval_3way.py` on `gpu` (`1,021`-doc shared corpus, `N=60` queries):
- Exploratory (`N=30`): `vanilla_rag` Gold ID Recall = `37.3%`, Mean `[P#]` paths = `0.0`.
- Practical Code (`N=30`): `vanilla_rag` Empty Evidence = `5/30`, Gold ID Recall = `16.4%`, Mean `[P#]` paths = `0.0`.

## Revision History
- `2026-10-25T10:00:00Z` (**system_measurement**, `created`): Recorded baseline dense RAG hypothesis (`VanillaRagBaseline`) for 3-way controlled evaluation.
- `2026-10-25T17:00:00Z` (**human_and_assistant**, `mark_superseded`): Superseded by MEM-CORR-HYBRID-RRF-SYMBOLIC-SEPARATION-001 after 3-way benchmark showed 37.3% vs. 84.5% Gold ID recall and 0.0 vs. 37.1 relationship paths.
