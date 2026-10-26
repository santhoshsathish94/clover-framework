---
{
  "schema_version": "1.0",
  "id": "MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
  "title": "Zero Signal from an Incapable Instrument Is Not Evidence Against the Mechanism",
  "entry_type": "concept",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "instrument-validity",
    "zero-denominator-fallacy",
    "kda-state-drift",
    "zero-allocation-cache"
  ],
  "tags": [
    "principle",
    "measurement",
    "verification",
    "instrumentation",
    "epistemology"
  ],
  "created_at": "2026-10-16T14:00:00Z",
  "updated_at": "2026-10-26T04:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "clover-operating-principles.md#2026-10-16; commit:58497d5",
    "actor": "human_and_assistant",
    "recorded_at": "2026-10-16T14:00:00Z"
  },
  "relationships": [
    {
      "type": "similar_to",
      "target": "MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001",
      "provenance": "clover-operating-principles.md#2026-10-16",
      "confidence": 0.95,
      "context_scope": "measurement_methodology",
      "created_at": "2026-10-16T14:00:00Z",
      "note": "Sibling measurement principle: both prevent mistaking an instrument/topology limitation for absence of the underlying phenomenon."
    },
    {
      "type": "supports",
      "target": "MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001",
      "provenance": "commit:58497d5",
      "confidence": 1.0,
      "context_scope": "expert_result_cache_design",
      "created_at": "2026-10-16T14:00:00Z",
      "note": "Explains why intra-sequence token repetition cannot hit a full-activation-keyed cache and why full-request replay is required."
    },
    {
      "type": "supports",
      "target": "MEM-OBS-HARD-COSINE-THRESHOLD-DROPOUT-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#finding-3",
      "confidence": 0.95,
      "context_scope": "entity_linking_threshold",
      "created_at": "2026-10-25T16:00:00Z",
      "note": "Explains why a hard cosine >= 0.35 filter on raw code queries acts as a zero-seed gate for 30% of queries."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-16T14:00:00Z",
      "actor": "human_and_assistant",
      "action": "created",
      "reason": "Initial concept entry recorded from clover-operating-principles.md#2026-10-16; commit:58497d5"
    }
  ]
}
---

# Zero Signal from an Incapable Instrument Is Not Evidence Against the Mechanism

## Summary
A test workload or filter that structurally cannot trigger a mechanism (or a table allocated at size zero) provides zero evidence against the mechanism; always verify table allocation and test with a workload matching the key semantics (such as replaying an identical full request).

## Content
Before concluding that a mechanism never fires or an event is absent, verify that the test instrument is structurally capable of observing it. Three verified failure modes illustrate this rule: (1) testing an activation-vector-keyed expert result cache by repeating one token 64 times inside a single sequence produced 0 hits because recurrent KDA state drifts at every position, whereas replaying the identical full request (A/B/A) hit 30,912 times and ran 3.27x faster, bit-identical; (2) under K3_PAYLOAD_RAM=1 a double-counted payload left the cache at zero allocation, making miss counts vacuous; (3) an awk filter keeping only records with max-change > 0 hid a present no-op stage and falsely reported the stage as absent.

## Original Source Evidence
1. Expert result cache test (commit 58497d5, 2026-10-16): 64 repeated tokens in one sequence -> 0 hits / 113,344 misses (recurrent KDA state drift); A/B/A replay of identical request -> 30,912 hits, 3.27x speedup (28.57s -> 8.74s), bit-identical.
2. Layer trace filter (2026-10-11): awk predicate `max_change > 0` dropped no-op stage 20 records.

## Revision History
- `2026-10-16T14:00:00Z` (**human_and_assistant**, `created`): Initial concept entry recorded from clover-operating-principles.md#2026-10-16; commit:58497d5
