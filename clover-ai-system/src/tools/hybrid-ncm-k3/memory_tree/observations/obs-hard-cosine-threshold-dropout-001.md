---
{
  "schema_version": "1.0",
  "id": "MEM-OBS-HARD-COSINE-THRESHOLD-DROPOUT-001",
  "title": "Observation: Hard Dense Embedding Threshold (`cosine >= 0.35`) Drops 30% (`9/30`) of Code Queries to Zero Seeds",
  "entry_type": "observation",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "hard-threshold-seed-dropout",
    "hybrid-rrf-fallback",
    "ontology-guided-graphrag",
    "code-to-concept-linking"
  ],
  "tags": [
    "ontorag",
    "embeddings",
    "cosine-threshold",
    "bm25",
    "rrf",
    "observation"
  ],
  "created_at": "2026-10-25T14:30:00Z",
  "updated_at": "2026-10-26T04:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "docs/clover-ontology-graphrag-research-context.md#finding-3",
    "actor": "system_measurement",
    "recorded_at": "2026-10-25T14:30:00Z"
  },
  "relationships": [
    {
      "type": "exemplifies",
      "target": "MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#finding-3",
      "confidence": 1.0,
      "context_scope": "entity_linking_threshold",
      "created_at": "2026-10-25T14:30:00Z",
      "note": "An over-strict `cosine >= 0.35` gate zeroes out the seed set before graph traversal can even run."
    },
    {
      "type": "challenges",
      "target": "MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#finding-3",
      "confidence": 1.0,
      "context_scope": "code_to_concept_linking",
      "created_at": "2026-10-25T14:30:00Z",
      "note": "Shows that dense embedding similarity alone fails on 30% of code-to-ontology queries at standard thresholds."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-25T14:30:00Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Initial observation entry recorded from docs/clover-ontology-graphrag-research-context.md#finding-3"
    }
  ]
}
---

# Observation: Hard Dense Embedding Threshold (`cosine >= 0.35`) Drops 30% (`9/30`) of Code Queries to Zero Seeds

## Summary
In `ontorag`'s `EntityLinker.link`, filtering `all-MiniLM-L6-v2` embeddings by a hard cutoff of `similarity_threshold = 0.35` drops 9 of 30 (`30.0%`) Practical code queries to zero seeds (`(no evidence retrieved)`), frequently discarding the exact right CWE/CAPEC at rank 1.

## Content
Raw source code snippets (such as Flask `re.search(request.args['regex'], data)` or Java `new File(targetDirectory + filename); file.delete()`) have lower surface cosine similarity to prose MITRE CWE/CAPEC descriptions in `all-MiniLM-L6-v2` space (`max_cosine` between `0.2398` and `0.3423`) than prose-to-prose queries. Because `EntityLinker.link` enforced a hard `cosine >= 0.35` cutoff with no lexical BM25 fallback, `9 / 30` Practical queries returned `0` seeds and `(no evidence retrieved)`, even though the discarded top-1 candidate was the exact vulnerability concept (`CAPEC-76` at `0.3290`, `CAPEC-137` at `0.2942`, `CAPEC-492` at `0.2761`).

## Original Source Evidence
Measured across all 30 Practical code queries on `/home/usergpu/ontorag` (`diagnose_three_findings.py`, `944` CWEs + `559` CAPECs indexed):
- `9 / 30` queries (`30.0%`) had `max_cosine < 0.35` (`0.2398..0.3423`) -> `0` seeds, `(no evidence retrieved)`.
- Session 6 dropped `CAPEC-76: Manipulating Web Input to File System Calls` (`cosine = 0.3290`).
- Session 15 dropped `CAPEC-137: Parameter Injection` (`cosine = 0.2942`).
- Session 20 dropped `CAPEC-492: Regular Expression Exponential Blowup` (`cosine = 0.2761`).

## Revision History
- `2026-10-25T14:30:00Z` (**system_measurement**, `created`): Initial observation entry recorded from docs/clover-ontology-graphrag-research-context.md#finding-3
