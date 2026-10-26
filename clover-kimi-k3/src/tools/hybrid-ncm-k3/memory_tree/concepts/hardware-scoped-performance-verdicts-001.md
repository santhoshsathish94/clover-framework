---
{
  "schema_version": "1.0",
  "id": "MEM-CONCEPT-HARDWARE-SCOPED-VERDICTS-001",
  "title": "Scope Every Performance Verdict to Its Exact Hardware and Kernel Regime",
  "entry_type": "concept",
  "epistemic_status": "observed",
  "status": "active",
  "concepts": [
    "hardware-scoped-verdict",
    "quantisation",
    "nibble-unpack",
    "gpu-supply-bandwidth"
  ],
  "tags": [
    "principle",
    "hardware",
    "performance",
    "quantisation",
    "scope"
  ],
  "created_at": "2026-09-23T17:00:00Z",
  "updated_at": "2026-10-26T04:00:00Z",
  "provenance": {
    "source_type": "conversation",
    "source_ref": "clover-operating-principles.md#2026-09-23; clover-kimi-k3-cpu-context.md; clover-kimi-k3-gpu-context.md",
    "actor": "human_and_assistant",
    "recorded_at": "2026-09-23T17:00:00Z"
  },
  "relationships": [
    {
      "type": "supports",
      "target": "MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001",
      "provenance": "clover-operating-principles.md#2026-09-23",
      "confidence": 1.0,
      "context_scope": "quantisation_verdict_scope",
      "created_at": "2026-09-29T05:30:10Z",
      "note": "Provides the hardware-scoping rule that corrects hardware-unscoped quantisation verdicts."
    },
    {
      "type": "challenges",
      "target": "MEM-HYP-QUANT-DEAD-END-ALL-HW-001",
      "provenance": "clover-operating-principles.md#2026-09-23",
      "confidence": 1.0,
      "context_scope": "quantisation_verdict_scope",
      "created_at": "2026-09-23T17:00:00Z",
      "note": "Refutes treating a single CPU kernel measurement as a universal hardware law."
    },
    {
      "type": "similar_to",
      "target": "MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001",
      "provenance": "clover-operating-principles.md#2026-09-23",
      "confidence": 0.8,
      "context_scope": "measurement_methodology",
      "created_at": "2026-10-09T16:00:00Z",
      "note": "Both prevent overgeneralizing a local measurement into a universal negative claim."
    }
  ],
  "uncertainty_notes": [
    "A direct GPU A/B comparing 4-bit vs 8-bit trunk decode has not yet been measured on the 8 x RTX 3090 box."
  ],
  "revisions": [
    {
      "timestamp": "2026-09-23T17:00:00Z",
      "actor": "human_and_assistant",
      "action": "created",
      "reason": "Initial concept entry recorded from clover-operating-principles.md#2026-09-23; clover-kimi-k3-cpu-context.md; clover-kimi-k3-gpu-context.md"
    }
  ]
}
---

# Scope Every Performance Verdict to Its Exact Hardware and Kernel Regime

## Summary
A performance measurement on one machine or kernel is not a universal property of the method: scope every performance verdict to its exact hardware and kernel in the same sentence, and explicitly mark cross-hardware transfer (such as CPU 4-bit trunk decode to GPU) as unmeasured until tested.

## Content
On an early CPU kernel, 8-bit to 4-bit trunk quantisation only moved decode from 4.1 to 4.0 s/token (-2.4%) due to nibble unpack overhead, whereas a later CPU kernel (`LB_Q4`, commit 74c1c0e) reduced decode from 3.326 to 2.716 s/token (-18.3%) with a quality tradeoff (identical on 14/34 decode-only prompts, 8/34 whole-trunk). On the 8 x RTX 3090 GPU box, the measured bottleneck for 4-bit experts is host-to-GPU supply bandwidth (4.95 s/layer from NVMe, 0.95 s/layer from RAM vs 0.23-0.29 s compute), while a 4-bit vs 8-bit trunk decode A/B on GPU remains unmeasured.

## Original Source Evidence
1. Creator principle (2026-09-23): "cpu processing is not how gpu will process it so the compounding can look very different."
2. CPU Step 3 (`heterogeneous-inference.md`): 4.1 -> 4.0 s/token (-2.4%).
3. CPU `LB_Q4` (`commit 74c1c0e`, `clover-kimi-k3-cpu-context.md`): 3.326 -> 2.716 s/token (-18.3%).
4. GPU 8x RTX 3090 (`clover-kimi-k3-gpu-context.md`): expert supply 4.95 s/layer (NVMe) / 0.95 s/layer (RAM) vs 0.23-0.29 s compute; Layer 0 DRAM throughput 5.6%; GPU 4-bit trunk A/B unmeasured.

## Revision History
- `2026-09-23T17:00:00Z` (**human_and_assistant**, `created`): Initial concept entry recorded from clover-operating-principles.md#2026-09-23; clover-kimi-k3-cpu-context.md; clover-kimi-k3-gpu-context.md
