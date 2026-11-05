---
{
  "schema_version": "1.0",
  "id": "MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001",
  "title": "Correction: CPU 4-Bit Verdicts Are Kernel-Scoped (`-2.4%` vs `-18.3%` `LB_Q4`); on GPU Supply Bandwidth Binds and Trunk 4-Bit vs 8-Bit Remains Unmeasured",
  "entry_type": "correction",
  "epistemic_status": "observed",
  "status": "active",
  "concepts": [
    "quantisation",
    "gpu-supply-bandwidth",
    "nibble-unpack",
    "hardware-scoped-verdict",
    "kimi-k3-moe"
  ],
  "tags": [
    "systems",
    "quantisation",
    "correction",
    "scope"
  ],
  "created_at": "2026-09-29T05:30:10Z",
  "updated_at": "2026-10-26T04:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "heterogeneous-inference.md (commit d8f3692); clover-kimi-k3-gpu-context.md#supplying-one-layers-experts; clover-kimi-k3-cpu-context.md#4-bit-trunk-for-decode (commit 74c1c0e)",
    "actor": "human_and_assistant",
    "recorded_at": "2026-09-29T05:30:10Z"
  },
  "relationships": [
    {
      "type": "corrects",
      "target": "MEM-HYP-QUANT-DEAD-END-ALL-HW-001",
      "provenance": "commit:d8f3692; commit:74c1c0e",
      "confidence": 0.95,
      "context_scope": "quantisation_verdict_scope",
      "created_at": "2026-09-29T05:30:10Z",
      "note": "Replaces the hardware-unscoped 'dead end' claim with exact kernel-scoped CPU measurements and GPU supply boundaries."
    },
    {
      "type": "challenges",
      "target": "MEM-HYP-QUANT-DEAD-END-ALL-HW-001",
      "provenance": "clover-operating-principles.md#2026-09-23",
      "confidence": 0.95,
      "context_scope": "hardware_unscoped_benchmark_claims",
      "created_at": "2026-09-29T05:30:10Z",
      "note": "Challenges any benchmark verdict that omits hardware and kernel scope."
    },
    {
      "type": "exemplifies",
      "target": "MEM-CONCEPT-HARDWARE-SCOPED-VERDICTS-001",
      "provenance": "clover-operating-principles.md#2026-09-23",
      "confidence": 0.95,
      "context_scope": "performance_verdict_scoping",
      "created_at": "2026-09-29T05:30:10Z",
      "note": "Demonstrates why CPU and GPU performance regimes must be distinguished and unmeasured GPU claims flagged."
    }
  ],
  "uncertainty_notes": [
    "A GPU 4-bit vs 8-bit trunk decode A/B has not been measured on the 8 x RTX 3090 server ('we cannot find that out here').",
    "Register-level nibble unpack cost on GPU is an architectural expectation; the measured GPU bottleneck is host-to-card expert supply (0.95-4.95 s/layer)."
  ],
  "revisions": [
    {
      "timestamp": "2026-09-29T05:30:10Z",
      "actor": "human_and_assistant",
      "action": "created",
      "reason": "Initial correction entry recorded from heterogeneous-inference.md (commit d8f3692); clover-kimi-k3-gpu-context.md#supplying-one-layers-experts; clover-kimi-k3-cpu-context.md#4-bit-trunk-for-decode (commit 74c1c0e)"
    }
  ]
}
---

# Correction: CPU 4-Bit Verdicts Are Kernel-Scoped (`-2.4%` vs `-18.3%` `LB_Q4`); on GPU Supply Bandwidth Binds and Trunk 4-Bit vs 8-Bit Remains Unmeasured

## Summary
Corrects MEM-HYP-QUANT-DEAD-END-ALL-HW-001: on CPU, the early kernel gave -2.4% (4.1 -> 4.0 s/tok) while the later `LB_Q4` kernel gave -18.3% (3.326 -> 2.716 s/tok, identical on 14/34 decode-only prompts); on the 8 x RTX 3090 GPU box, 4-bit expert supply bandwidth (4.95 s/layer NVMe, 0.95 s/layer RAM vs 0.23-0.29 s compute) is the measured bottleneck, while a GPU 4-bit vs 8-bit trunk decode A/B is unmeasured.

## Content
Both '4-bit quantisation is a universal dead end on all hardware' and '4-bit trunk decode is already proven faster on our GPU' are unsupported. What is established by measurement: (1) on CPU, speed depends on the packing kernel (-2.4% in Step 3 vs -18.3% with `LB_Q4` in commit 74c1c0e, with a quality tradeoff of 14/34 decode-only and 8/34 whole-trunk prompt matches); (2) on the 8 x RTX 3090 GPU box, streaming a layer's 15.72 GB of 4-bit experts takes 4.95 s from NVMe and 0.95 s from host RAM against 0.23-0.29 s of kernel compute per 1000-token window, and at Layer 0 `ncu` measured DRAM throughput at 5.6% (shared-memory/L1-bound); (3) a direct GPU A/B comparing a 4-bit vs 8-bit trunk has not been measured.

## Original Source Evidence
1. GPU context (`clover-kimi-k3-gpu-context.md`): 8-card expert supply 4.95 s/layer (NVMe), 0.95 s/layer (RAM) vs 0.23-0.29 s compute; Layer 0 `ncu` DRAM throughput 5.6%.
2. CPU context (`clover-kimi-k3-cpu-context.md`, commit 74c1c0e): `LB_Q4` decode 3.326 -> 2.716 s/token (-18.3%); identical on 14/34 prompts (decode-only), 8/34 (whole trunk).
3. Creator rule (`clover-operating-principles.md#2026-09-23`): "cpu processing is not how gpu will process it so the compounding can look very different."

## Revision History
- `2026-09-29T05:30:10Z` (**human_and_assistant**, `created`): Initial correction entry recorded from heterogeneous-inference.md (commit d8f3692); clover-kimi-k3-gpu-context.md#supplying-one-layers-experts; clover-kimi-k3-cpu-context.md#4-bit-trunk-for-decode (commit 74c1c0e)
