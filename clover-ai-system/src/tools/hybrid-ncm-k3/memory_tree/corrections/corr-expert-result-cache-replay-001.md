---
{
  "schema_version": "1.0",
  "id": "MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001",
  "title": "Correction: Intra-Sequence Repeats Cannot Hit Due to KDA State Drift, Whereas Replaying an Identical Request Hits 30,912 Times (`3.27x` Faster)",
  "entry_type": "correction",
  "epistemic_status": "observed",
  "status": "active",
  "concepts": [
    "expert-result-cache",
    "request-replay",
    "kda-state-drift",
    "zero-allocation-cache",
    "kimi-k3-moe"
  ],
  "tags": [
    "systems",
    "correction",
    "instrument-validity"
  ],
  "created_at": "2026-10-16T09:52:19Z",
  "updated_at": "2026-10-18T00:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "commit:58497d5; clover-kimi-k3-cpu-context.md (2026-10-17 status; 2026-10-18 whole-vector survey)",
    "actor": "human_and_assistant",
    "recorded_at": "2026-10-16T09:52:19Z"
  },
  "relationships": [
    {
      "type": "corrects",
      "target": "MEM-HYP-EXPERT-RESULT-CACHE-DEAD-001",
      "provenance": "commit:58497d5",
      "confidence": 1.0,
      "context_scope": "expert_result_cache_design",
      "created_at": "2026-10-16T09:52:19Z",
      "note": "Refutes the 'cannot ever hit' claim by fixing zero-allocation and testing A/B/A full-request replay."
    },
    {
      "type": "exemplifies",
      "target": "MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
      "provenance": "commit:58497d5",
      "confidence": 1.0,
      "context_scope": "instrument_validity",
      "created_at": "2026-10-16T09:52:19Z",
      "note": "Canonical example of both a workload that structurally cannot hit (intra-sequence repeat under KDA drift) and a zero-allocated table."
    }
  ],
  "uncertainty_notes": [
    "The 3.27x speedup applies to repeated identical requests (or shared identical prefixes from position 0); within a single non-repeating request, 0 of 224,000 activation segments repeated in the 2026-10-18 survey."
  ],
  "revisions": [
    {
      "timestamp": "2026-10-16T09:52:19Z",
      "actor": "human_and_assistant",
      "action": "created",
      "reason": "Initial correction entry recorded from commit:58497d5; clover-kimi-k3-cpu-context.md (2026-10-17 status; 2026-10-18 whole-vector survey)"
    }
  ]
}
---

# Correction: Intra-Sequence Repeats Cannot Hit Due to KDA State Drift, Whereas Replaying an Identical Request Hits 30,912 Times (`3.27x` Faster)

## Summary
Corrects MEM-HYP-EXPERT-RESULT-CACHE-DEAD-001: because the cache is keyed on the full input activation vector and recurrent KDA state drifts at every position within a sequence, repeating a token inside one sequence structurally cannot hit; furthermore, a double-counted payload under `K3_PAYLOAD_RAM=1` had left the cache allocated at size zero. Once allocated and tested by replaying the identical full request (A/B/A), it hit 30,912 times and ran 3.27x faster, bit-identical.

## Content
Demonstrates two essential diagnostic checks whenever a memoization or result cache reports 0 hits: (1) verify the table was actually allocated with non-zero capacity (`K3_PAYLOAD_RAM=1` had double-counted payload bytes and left the result cache at 0 entries); and (2) verify the workload matches the cache's key invariant. Because KDA recurrent state updates at every token position, the same token ID at positions 1..64 produces 64 distinct full-activation vectors (`0` identical vectors across 1,000 real tokens in the 2026-10-18 Layer 4 survey). Replaying the identical request (A/B/A) produces identical position-aligned activations, yielding 30,912 hits, 29,440 misses, 0 evictions, and a 3.27x speedup (28.574 s -> 8.759 s; decode 3.68 -> 1.43 s/tok) with bit-identical output.

## Original Source Evidence
Commit `58497d5` (`CONTEXT.md` 'Repeat Is 3.27x Once They Are Not') & `clover-kimi-k3-cpu-context.md`:
- A/B/A request replay: `28.574 s`, `8.759 s`, `27.329 s`, `8.737 s` (`3.27x` faster on repeat, `30,912` hits, `0` evictions, bit-identical).
- Non-repeating overhead: `+0.88%` (`3.6641 -> 3.6965 s/token`).

## Revision History
- `2026-10-16T09:52:19Z` (**human_and_assistant**, `created`): Initial correction entry recorded from commit:58497d5; clover-kimi-k3-cpu-context.md (2026-10-17 status; 2026-10-18 whole-vector survey)
