---
{
  "schema_version": "1.0",
  "id": "MEM-OBS-ISOLATED-MICROBENCH-BIDIRECTIONAL-DRIFT-001",
  "title": "Observation: Isolated Micro-Benchmarks Mispredict In-Layer Gains in Both Directions (`15.2x` vs `8.6x` and `1.68x` vs `2.2x`)",
  "entry_type": "observation",
  "epistemic_status": "observed",
  "status": "active",
  "concepts": [
    "micro-benchmark",
    "in-situ-measurement",
    "cache-warmth",
    "synthetic-vs-real-activations"
  ],
  "tags": [
    "systems",
    "benchmark",
    "observation"
  ],
  "created_at": "2026-10-11T12:00:00Z",
  "updated_at": "2026-10-26T04:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "clover-operating-principles.md#2026-10-11; clover-kimi-k3-cpu-context.md L87, L169-170",
    "actor": "human_and_assistant",
    "recorded_at": "2026-10-11T12:00:00Z"
  },
  "relationships": [
    {
      "type": "exemplifies",
      "target": "MEM-CONCEPT-SYNTHETIC-TIMING-VS-MEANING-001",
      "provenance": "clover-operating-principles.md#2026-10-11",
      "confidence": 0.85,
      "context_scope": "random_index_harness",
      "created_at": "2026-10-11T12:00:00Z",
      "note": "The 1.68x vs 2.2x under-prediction arose from feeding uniformly random palette indices instead of real weight indices."
    }
  ],
  "uncertainty_notes": [
    "The 15.2x/8.6x and 1.68x/2.2x ratios are recorded in `clover-operating-principles.md#2026-10-11`, while `clover-kimi-k3-cpu-context.md` records the corresponding `962-1,071` vs `760-840 GMAC/s` in-layer measurements."
  ],
  "revisions": [
    {
      "timestamp": "2026-10-11T12:00:00Z",
      "actor": "human_and_assistant",
      "action": "created",
      "reason": "Initial observation entry recorded from clover-operating-principles.md#2026-10-11; clover-kimi-k3-cpu-context.md L87, L169-170"
    }
  ]
}
---

# Observation: Isolated Micro-Benchmarks Mispredict In-Layer Gains in Both Directions (`15.2x` vs `8.6x` and `1.68x` vs `2.2x`)

## Summary
A standalone hot-loop replica over-predicted gain (`15.2x` standalone vs `8.6x` in-layer because its 6.29 MB state stayed cache-warm in isolation), while a random-index kernel harness under-predicted gain (`1.68x` standalone vs `2.2x` in-layer on real non-uniform weight palettes); always re-measure in situ.

## Content
Do not assume isolated micro-benchmarks always over-predict: error occurs in both directions. In the first case, a standalone KDA loop replica kept its 6.29 MB state warm with no competing traffic (`15.2x` standalone vs `8.6x` inside the full layer; also corroborated in `clover-kimi-k3-cpu-context.md` where isolated 4x4 kernel throughput was `962-1,071 GMAC/s` vs `760-840 GMAC/s` in-layer). In the second case on the same day, a kernel harness fed uniformly random palette indices predicted `1.68x` whereas the real layer with non-uniform weight indices delivered `2.2x`.

## Original Source Evidence
1. `clover-operating-principles.md#2026-10-11`: standalone loop `15.2x` (~5.5% of layer) vs in-layer `8.6x` (~4%); random palette harness `1.68x` vs in-layer `2.2x`.
2. `clover-kimi-k3-cpu-context.md` lines 87, 169-170: isolated `962-1,071 GMAC/s` vs in-layer `760-840 GMAC/s`.

## Revision History
- `2026-10-11T12:00:00Z` (**human_and_assistant**, `created`): Initial observation entry recorded from clover-operating-principles.md#2026-10-11; clover-kimi-k3-cpu-context.md L87, L169-170
