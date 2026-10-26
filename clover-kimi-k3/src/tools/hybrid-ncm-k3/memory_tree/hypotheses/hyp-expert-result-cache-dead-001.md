---
{
  "schema_version": "1.0",
  "id": "MEM-HYP-EXPERT-RESULT-CACHE-DEAD-001",
  "title": "Initial Hypothesis: The Expert Result Cache Can Never Hit in Kimi K3",
  "entry_type": "hypothesis",
  "epistemic_status": "hypothesis",
  "status": "superseded",
  "superseded_by": "MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001",
  "concepts": [
    "expert-result-cache",
    "intra-sequence-repetition",
    "kda-state-drift",
    "kimi-k3-moe"
  ],
  "tags": [
    "systems",
    "hypothesis",
    "superseded"
  ],
  "created_at": "2026-10-16T00:00:00Z",
  "updated_at": "2026-10-16T09:52:19Z",
  "provenance": {
    "source_type": "conversation",
    "source_ref": "commit:58497d5; clover-operating-principles.md#2026-10-16",
    "actor": "assistant",
    "recorded_at": "2026-10-16T00:00:00Z"
  },
  "relationships": [
    {
      "type": "similar_to",
      "target": "MEM-HYP-CACHE-DEAD-END-001",
      "provenance": "commit:58497d5",
      "confidence": 0.75,
      "context_scope": "dead_end_from_narrow_measurement",
      "created_at": "2026-10-16T00:00:00Z",
      "note": "Both rejected a caching mechanism based on a test workload or structure that prevented hits."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-16T00:00:00Z",
      "actor": "assistant",
      "action": "created",
      "reason": "Recorded initial claim that expert result caching cannot hit from 64-token intra-sequence repeat test."
    },
    {
      "timestamp": "2026-10-16T09:52:19Z",
      "actor": "human_and_assistant",
      "action": "mark_superseded",
      "reason": "Superseded by MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001 after fixing zero-allocation and testing A/B/A request replay."
    }
  ]
}
---

# Initial Hypothesis: The Expert Result Cache Can Never Hit in Kimi K3

## Summary
Mistaken hypothesis on 2026-10-16 that exact-match expert result caching could never hit in Kimi K3, inferred from a test that repeated one token 64 times inside a single sequence and observed 0 hits / 113,344 misses (superseded by MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001).

## Content
Tested the activation-vector-keyed expert result cache using 64 copies of token 17374 inside one sequence, saw 0 hits against 113,344 misses, and generalized that the cache mechanism was dead.

## Original Source Evidence
Commit `58497d5` ('Correcting myself'): "I reported earlier in this cycle that exact-match result caching could never hit in this model. The test repeated a token inside one sequence, never a request ... 64 copies of token 17374 gave 0 hits against 113344 misses."

## Revision History
- `2026-10-16T00:00:00Z` (**assistant**, `created`): Recorded initial claim that expert result caching cannot hit from 64-token intra-sequence repeat test.
- `2026-10-16T09:52:19Z` (**human_and_assistant**, `mark_superseded`): Superseded by MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001 after fixing zero-allocation and testing A/B/A request replay.
