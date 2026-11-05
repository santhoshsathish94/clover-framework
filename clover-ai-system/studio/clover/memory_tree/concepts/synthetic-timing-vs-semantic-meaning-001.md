---
{
  "schema_version": "1.0",
  "id": "MEM-CONCEPT-SYNTHETIC-TIMING-VS-MEANING-001",
  "title": "Synthetic Input Measures Timing, Never Semantic Meaning",
  "entry_type": "concept",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "synthetic-vs-real-activations",
    "semantic-observer-verification",
    "in-situ-measurement",
    "micro-benchmark"
  ],
  "tags": [
    "principle",
    "measurement",
    "activations",
    "routing",
    "verification"
  ],
  "created_at": "2026-10-11T18:00:00Z",
  "updated_at": "2026-10-26T04:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "clover-operating-principles.md#2026-10-11",
    "actor": "human_and_assistant",
    "recorded_at": "2026-10-11T18:00:00Z"
  },
  "relationships": [
    {
      "type": "similar_to",
      "target": "MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
      "provenance": "clover-operating-principles.md#2026-10-11",
      "confidence": 0.9,
      "context_scope": "measurement_methodology",
      "created_at": "2026-10-11T18:00:00Z",
      "note": "Both govern instrument validity: synthetic inputs corrupt semantic observables just as incapable workloads corrupt hit counts."
    },
    {
      "type": "supports",
      "target": "MEM-OBS-ISOLATED-MICROBENCH-BIDIRECTIONAL-DRIFT-001",
      "provenance": "clover-operating-principles.md#2026-10-11",
      "confidence": 0.95,
      "context_scope": "in_situ_verification",
      "created_at": "2026-10-11T18:00:00Z",
      "note": "Explains the under-prediction half of isolated micro-benchmarks when random palette indices replace real weight distributions."
    },
    {
      "type": "supports",
      "target": "MEM-EXP-RUST-SLOTCACHE-001",
      "provenance": "commit:178d3c1",
      "confidence": 0.95,
      "context_scope": "in_situ_verification",
      "created_at": "2026-10-25T19:30:00Z",
      "note": "Motivates full 93-layer in-situ verification (`K3Full93Chain`) alongside standalone unit tests."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-11T18:00:00Z",
      "actor": "human_and_assistant",
      "action": "created",
      "reason": "Initial concept entry recorded from clover-operating-principles.md#2026-10-11"
    }
  ]
}
---

# Synthetic Input Measures Timing, Never Semantic Meaning

## Summary
Synthetic or random tensors are valid for timing harnesses, but invert semantic properties (router spread, attention norm contribution) and can mispredict kernel speedups when real weights are non-uniform; derive semantic observers from verified gate harnesses and re-measure in situ.

## Content
Timing harnesses and semantic observers have opposite input requirements. Feeding random activations into a Kimi K3 layer tracer produced uniform router weights and an attention norm contribution of only 0.6% of the residual — both artifacts of random vectors. On real model activations inherited from the verified gate harness, router weights spread 6.6x and attention magnitude was comparable to the input residual. Similarly, a standalone kernel harness fed uniformly random palette indices predicted 1.68x speedup whereas real non-uniform weights in situ delivered 2.2x.

## Original Source Evidence
Measured on Kimi K3 (2026-10-11): random input -> uniform router spread (1.0x), attention contribution 0.6% of norm; real gate-verified activations -> 6.6x router spread, attention comparable to input norm. Standalone random-index kernel benchmark predicted 1.68x vs. 2.2x measured in situ on real weights.

## Revision History
- `2026-10-11T18:00:00Z` (**human_and_assistant**, `created`): Initial concept entry recorded from clover-operating-principles.md#2026-10-11
