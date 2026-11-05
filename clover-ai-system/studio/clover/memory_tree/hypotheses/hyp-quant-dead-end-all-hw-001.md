---
{
  "schema_version": "1.0",
  "id": "MEM-HYP-QUANT-DEAD-END-ALL-HW-001",
  "title": "Initial Hypothesis: 4-Bit Quantisation Is a Universal Dead End for Speed Across All Hardware",
  "entry_type": "hypothesis",
  "epistemic_status": "hypothesis",
  "status": "superseded",
  "superseded_by": "MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001",
  "concepts": [
    "quantisation",
    "nibble-unpack",
    "cpu-decode",
    "hardware-scoped-verdict"
  ],
  "tags": [
    "systems",
    "quantisation",
    "hypothesis",
    "superseded"
  ],
  "created_at": "2026-09-23T00:00:00Z",
  "updated_at": "2026-09-29T05:30:10Z",
  "provenance": {
    "source_type": "conversation",
    "source_ref": "research/k3/architecture/heterogeneous-inference.md#step-3; clover-operating-principles.md#2026-09-23",
    "actor": "assistant",
    "recorded_at": "2026-09-23T00:00:00Z"
  },
  "relationships": [
    {
      "type": "similar_to",
      "target": "MEM-HYP-CACHE-DEAD-END-001",
      "provenance": "clover-operating-principles.md#2026-09-23",
      "confidence": 0.8,
      "context_scope": "dead_end_from_narrow_measurement",
      "created_at": "2026-09-23T00:00:00Z",
      "note": "Both are premature 'dead end' verdicts generalized from a single structurally limited measurement."
    }
  ],
  "uncertainty_notes": [
    "Commit 74c1c0e (2026-10-18) later measured a different 4-bit CPU trunk packing (LB_Q4) at 3.326 -> 2.716 s/token (-18.3%), showing even the CPU result depends on the packing kernel."
  ],
  "revisions": [
    {
      "timestamp": "2026-09-23T00:00:00Z",
      "actor": "assistant",
      "action": "created",
      "reason": "Recorded initial CPU 4-bit trunk observation as an unscoped 'dead end' claim."
    },
    {
      "timestamp": "2026-09-29T05:30:10Z",
      "actor": "human_and_assistant",
      "action": "mark_superseded",
      "reason": "Superseded by MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001: verdicts must be scoped to hardware and kernel."
    }
  ]
}
---

# Initial Hypothesis: 4-Bit Quantisation Is a Universal Dead End for Speed Across All Hardware

## Summary
From one early CPU run where 8-bit to 4-bit trunk quantisation moved decode from 4.1 to 4.0 s/token (-2.4%), the assistant concluded 'shrinking further is a dead end' without scoping the verdict to that CPU kernel (superseded by MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001).

## Content
On that initial CPU kernel, 4-bit weights had to be unpacked before use and the unpacking cost nearly offset the halved memory read. Recording the conclusion without naming the hardware or kernel made it read as a universal property of 4-bit quantisation rather than a property of that specific CPU implementation.

## Original Source Evidence
heterogeneous-inference.md Step 3: "4.1 down to 4.0 seconds per token ... it now matched the original for 3 tokens instead of 21 ... We had already written down 'shrinking further is a dead end.' That was wrong."

## Revision History
- `2026-09-23T00:00:00Z` (**assistant**, `created`): Recorded initial CPU 4-bit trunk observation as an unscoped 'dead end' claim.
- `2026-09-29T05:30:10Z` (**human_and_assistant**, `mark_superseded`): Superseded by MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001: verdicts must be scoped to hardware and kernel.
