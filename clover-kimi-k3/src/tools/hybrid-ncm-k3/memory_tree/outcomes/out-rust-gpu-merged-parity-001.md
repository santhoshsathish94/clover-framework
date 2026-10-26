---
{
  "schema_version": "1.0",
  "id": "MEM-OUT-RUST-GPU-PARITY-001",
  "title": "Outcome: Bit-Identical 93-Layer GPU Execution (`max_abs_diff = 0.0`) and 100% 5-Benchmark Suite",
  "entry_type": "outcome",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "rust-host-path",
    "kimi-k3-moe",
    "generative-compilation"
  ],
  "tags": [
    "systems",
    "outcome",
    "verified",
    "benchmarks"
  ],
  "created_at": "2026-10-25T20:00:00Z",
  "updated_at": "2026-10-26T02:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "commit:178d3c1 & commit:f308f12",
    "actor": "system_measurement",
    "recorded_at": "2026-10-26T02:00:00Z"
  },
  "relationships": [
    {
      "type": "derived_from",
      "target": "MEM-EXP-RUST-SLOTCACHE-001",
      "provenance": "commit:178d3c1",
      "confidence": 1.0,
      "context_scope": "rust_host_path_verification",
      "created_at": "2026-10-25T20:00:00Z",
      "note": "Outcome measured directly from the differential verification experiment."
    },
    {
      "type": "supports",
      "target": "MEM-DEC-SLOTCACHE-PER-LAYER-001",
      "provenance": "commit:178d3c1",
      "confidence": 1.0,
      "context_scope": "rust_host_path_verification",
      "created_at": "2026-10-25T20:00:00Z",
      "note": "Validates the architectural decision in production service (`clover-k3-chat.service`)."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-25T20:00:00Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Initial outcome entry recorded from commit:178d3c1 & commit:f308f12"
    }
  ]
}
---

# Outcome: Bit-Identical 93-Layer GPU Execution (`max_abs_diff = 0.0`) and 100% 5-Benchmark Suite

## Summary
Verified bit-identical 93-layer Kimi K3 GPU execution (`max_abs_diff = 0.0`) with Rust host-path libraries active, achieving 20/20 GSM8K, 20/20 MMLU, 40/40 Extended HumanEval, 15/15 HybridAgent QA, and 47/47 on the Pooled Multi-Tier Benchmark.

## Content
Demonstrates that replacing the C++ host-path cache, pool initializer, and supply logic with validated Rust crates preserves exact numerical parity while Generative Compilation eliminates mid-generation syntax, scope, and doctest errors.

## Original Source Evidence
Commits `178d3c1` and `f308f12`: `max_abs_diff = 0.0` (`cpp` vs `rust`); GSM8K `20/20 (100%)`, MMLU `20/20 (100%)`, HumanEval `40/40 (100%)`, Rust GC `5/5 (100%)`, Pooled `47/47 (100%)`.

## Revision History
- `2026-10-25T20:00:00Z` (**system_measurement**, `created`): Initial outcome entry recorded from commit:178d3c1 & commit:f308f12
