---
{
  "schema_version": "1.0",
  "id": "MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001",
  "title": "Flat Parameter Response Implicates Structure, Not Absence of Effect",
  "entry_type": "concept",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "flat-parameter-response",
    "structural-bottleneck-diagnosis",
    "cyclic-sweep-eviction",
    "per-layer-partitioning"
  ],
  "tags": [
    "principle",
    "systems",
    "measurement",
    "diagnosis"
  ],
  "created_at": "2026-10-09T16:00:00Z",
  "updated_at": "2026-10-26T04:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "clover-operating-principles.md#2026-10-09",
    "actor": "human_and_assistant",
    "recorded_at": "2026-10-09T16:00:00Z"
  },
  "relationships": [
    {
      "type": "derived_from",
      "target": "MEM-OBS-GLOBAL-CACHE-FLAT-001",
      "provenance": "clover-operating-principles.md#2026-10-09",
      "confidence": 1.0,
      "context_scope": "structural_measurement_diagnosis",
      "created_at": "2026-10-09T16:00:00Z",
      "note": "Generalized from the flat 23.3% global expert cache sweep across capacities 512..4096."
    },
    {
      "type": "derived_from",
      "target": "MEM-CORR-PER-LAYER-PARTITION-002",
      "provenance": "clover-operating-principles.md#2026-10-09",
      "confidence": 1.0,
      "context_scope": "structural_measurement_diagnosis",
      "created_at": "2026-10-09T16:00:00Z",
      "note": "Validated by the per-layer partitioning fix that jumped hit rate from 23.3% to 48.0%."
    },
    {
      "type": "exemplifies",
      "target": "MEM-CORR-PER-LAYER-PARTITION-002",
      "provenance": "clover-operating-principles.md#2026-10-09",
      "confidence": 1.0,
      "context_scope": "expert_cache_design",
      "created_at": "2026-10-09T16:00:00Z",
      "note": "Per-layer cache partitioning is the canonical instance of fixing structure rather than abandoning the lever."
    },
    {
      "type": "challenges",
      "target": "MEM-HYP-CACHE-DEAD-END-001",
      "provenance": "clover-operating-principles.md#2026-10-09",
      "confidence": 1.0,
      "context_scope": "expert_cache_design",
      "created_at": "2026-10-09T16:00:00Z",
      "note": "Identifies the diagnostic fallacy that led to prematurely rejecting expert caching."
    }
  ],
  "uncertainty_notes": [
    "A flat curve can still reflect genuine absence of effect once structural bottlenecks (eviction topology, zero allocation, and filter truncation) are independently ruled out."
  ],
  "revisions": [
    {
      "timestamp": "2026-10-09T16:00:00Z",
      "actor": "human_and_assistant",
      "action": "created",
      "reason": "Initial concept entry recorded from clover-operating-principles.md#2026-10-09"
    }
  ]
}
---

# Flat Parameter Response Implicates Structure, Not Absence of Effect

## Summary
When varying a capacity or budget parameter across a wide range yields a completely flat metric, suspect the structure being measured (such as shared-pool cyclic eviction across stages/layers or unallocated capacity) before concluding the underlying mechanism has no effect; test partitioning the same total budget per stage/layer.

## Content
A flat response curve across orders of magnitude of a parameter (e.g., global expert cache hit rate remaining invariant at 23.3% across capacities 512, 1024, 2048, and 4096) indicates that the swept parameter is not the binding constraint under the current topology. In a multi-stage or 92-layer pipeline, any unpartitioned global LRU pool smaller than the full working set suffers cyclic-sweep eviction before Stage/Layer 1 is revisited. Partitioning the exact same total budget per stage or per layer eliminates cross-stage eviction and restores monotonic scaling (raising Kimi K3 expert cache hit rate from 23.3% to 48.0% at the same 51.7 GB budget).

## Original Source Evidence
Measured on Kimi K3 92-layer MoE (2026-10-09, 51.7 GB budget): global shared LRU pool gave 23.3% hit rate flat across 512..4096 entries; partitioning the same 51.7 GB per layer raised hit rate from 23.3% to 48.0%.

## Revision History
- `2026-10-09T16:00:00Z` (**human_and_assistant**, `created`): Initial concept entry recorded from clover-operating-principles.md#2026-10-09
