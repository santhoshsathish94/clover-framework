---
{
  "schema_version": "1.0",
  "id": "MEM-CORR-PER-LAYER-PARTITION-002",
  "title": "Correction: Partitioning the Expert Cache Per Layer Raises Hit Rate from 23.3% to 48.0%",
  "entry_type": "correction",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "expert-slot-cache",
    "per-layer-partitioning",
    "cyclic-sweep-eviction",
    "kimi-k3-moe"
  ],
  "tags": [
    "systems",
    "cache",
    "correction",
    "verified"
  ],
  "created_at": "2026-10-09T15:00:00Z",
  "updated_at": "2026-10-09T15:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "clover-operating-principles.md#2026-10-09",
    "actor": "human_and_assistant",
    "recorded_at": "2026-10-09T15:00:00Z"
  },
  "relationships": [
    {
      "type": "corrects",
      "target": "MEM-HYP-CACHE-DEAD-END-001",
      "provenance": "clover-operating-principles.md#2026-10-09",
      "confidence": 1.0,
      "context_scope": "expert_cache_design",
      "created_at": "2026-10-09T15:00:00Z",
      "note": "Refutes the 'caching has no effect' hypothesis by eliminating cross-layer cyclic eviction."
    },
    {
      "type": "refines",
      "target": "MEM-OBS-GLOBAL-CACHE-FLAT-001",
      "provenance": "clover-operating-principles.md#2026-10-09",
      "confidence": 1.0,
      "context_scope": "expert_cache_design",
      "created_at": "2026-10-09T15:00:00Z",
      "note": "Explains why the global sweep was flat and how per-layer partitioning resolves it."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-09T15:00:00Z",
      "actor": "human_and_assistant",
      "action": "created",
      "reason": "Initial correction entry recorded from clover-operating-principles.md#2026-10-09"
    }
  ]
}
---

# Correction: Partitioning the Expert Cache Per Layer Raises Hit Rate from 23.3% to 48.0%

## Summary
When a measurement shows a flat response across all settings of a parameter, suspect the structure being measured (global eviction across layers), not the effect: partitioning the same 51.7 GB cache per layer increased the hit rate from 23.3% to 48.0%.

## Content
Corrects MEM-HYP-CACHE-DEAD-END-001. Because each layer's experts only compete with experts of the same layer when partitioned per layer (`layer_id in 1..92`), cross-layer cyclic eviction is eliminated and the same 51.7 GB budget achieves 48.0% hit rate.

## Original Source Evidence
Measured on Kimi K3 92-layer MoE (51.7 GB budget): global pool = 23.3% hit rate (flat); per-layer partitioned pool = 48.0% hit rate (monotonic scaling with per-layer slots).

## Revision History
- `2026-10-09T15:00:00Z` (**human_and_assistant**, `created`): Initial correction entry recorded from clover-operating-principles.md#2026-10-09
