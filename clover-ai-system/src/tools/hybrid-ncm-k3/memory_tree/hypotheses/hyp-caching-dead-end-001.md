---
{
  "schema_version": "1.0",
  "id": "MEM-HYP-CACHE-DEAD-END-001",
  "title": "Initial Hypothesis: Expert Caching Has No Effect on Kimi K3 Routing",
  "entry_type": "hypothesis",
  "epistemic_status": "hypothesis",
  "status": "superseded",
  "superseded_by": "MEM-CORR-PER-LAYER-PARTITION-002",
  "concepts": [
    "expert-slot-cache",
    "cyclic-sweep-eviction",
    "kimi-k3-moe"
  ],
  "tags": [
    "systems",
    "cache",
    "hypothesis",
    "superseded"
  ],
  "created_at": "2026-10-09T11:20:00Z",
  "updated_at": "2026-10-09T15:00:00Z",
  "provenance": {
    "source_type": "conversation",
    "source_ref": "clover-operating-principles.md#2026-10-09",
    "actor": "assistant",
    "recorded_at": "2026-10-09T11:20:00Z"
  },
  "relationships": [
    {
      "type": "derived_from",
      "target": "MEM-OBS-GLOBAL-CACHE-FLAT-001",
      "provenance": "clover-operating-principles.md#2026-10-09",
      "confidence": 1.0,
      "context_scope": "expert_cache_design",
      "created_at": "2026-10-09T11:20:00Z",
      "note": "Inferred from the flat global capacity sweep."
    },
    {
      "type": "caused_by",
      "target": "MEM-OBS-GLOBAL-CACHE-FLAT-001",
      "provenance": "clover-operating-principles.md#2026-10-09",
      "confidence": 1.0,
      "context_scope": "expert_cache_design",
      "created_at": "2026-10-09T11:20:00Z",
      "note": "Triggered by misinterpreting a flat parameter response as absence of the underlying effect."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-09T11:20:00Z",
      "actor": "assistant",
      "action": "created",
      "reason": "Recorded initial conclusion from global cache capacity sweep."
    },
    {
      "timestamp": "2026-10-09T15:00:00Z",
      "actor": "human_and_assistant",
      "action": "mark_superseded",
      "reason": "Superseded by MEM-CORR-PER-LAYER-PARTITION-002 after partitioning the cache per layer."
    }
  ]
}
---

# Initial Hypothesis: Expert Caching Has No Effect on Kimi K3 Routing

## Summary
Mistaken initial hypothesis that expert caching is ineffective for Kimi K3 because varying cache capacity from 512 to 4096 produced a flat 23.3% hit rate (superseded by MEM-CORR-PER-LAYER-PARTITION-002).

## Content
Concluded from the flat 23.3% curve across capacities 512..4096 that expert reuse across tokens was absent and caching should be rejected.

## Original Source Evidence
Initial report (2026-10-09): "Hit rate is 23.3% flat from 512 to 4096 entries due to cyclic sweep; expert caching does not help this workload."

## Revision History
- `2026-10-09T11:20:00Z` (**assistant**, `created`): Recorded initial conclusion from global cache capacity sweep.
- `2026-10-09T15:00:00Z` (**human_and_assistant**, `mark_superseded`): Superseded by MEM-CORR-PER-LAYER-PARTITION-002 after partitioning the cache per layer.
