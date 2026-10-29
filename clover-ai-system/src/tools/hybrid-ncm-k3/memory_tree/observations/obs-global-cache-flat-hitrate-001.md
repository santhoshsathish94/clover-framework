---
{
  "schema_version": "1.0",
  "id": "MEM-OBS-GLOBAL-CACHE-FLAT-001",
  "title": "Observation: Global Expert Cache Hit Rate Flat at 23.3% Across Capacities 512..4096",
  "entry_type": "observation",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "expert-slot-cache",
    "cyclic-sweep-eviction",
    "kimi-k3-moe"
  ],
  "tags": [
    "systems",
    "cache",
    "measurement",
    "k3-moe"
  ],
  "created_at": "2026-10-09T11:00:00Z",
  "updated_at": "2026-10-09T11:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "clover-kimi-k3/docs/clover-kimi-k3-cpu-context.md",
    "actor": "system_measurement",
    "recorded_at": "2026-10-09T11:00:00Z"
  },
  "relationships": [],
  "revisions": [
    {
      "timestamp": "2026-10-09T11:00:00Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Initial observation entry recorded from clover-kimi-k3/docs/clover-kimi-k3-cpu-context.md"
    }
  ]
}
---

# Observation: Global Expert Cache Hit Rate Flat at 23.3% Across Capacities 512..4096

## Summary
Measured a flat 23.3% expert cache hit rate across every global capacity setting from 512 to 4096 entries during 92-layer Kimi K3 MoE forward passes.

## Content
Because 92 MoE layers execute sequentially for each token and request 16 routed experts per layer (1,472 expert lookups per token step), a single global LRU cache smaller than the 92-layer working set suffers cyclic-sweep eviction before Layer 1 is revisited on the next token.

## Original Source Evidence
Benchmark log (2026-10-09): Global expert cache capacity 512 -> 23.3% hits; 1024 -> 23.3% hits; 2048 -> 23.3% hits; 4096 -> 23.3% hits.

## Revision History
- `2026-10-09T11:00:00Z` (**system_measurement**, `created`): Initial observation entry recorded from clover-kimi-k3/docs/clover-kimi-k3-cpu-context.md
