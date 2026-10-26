---
{
  "schema_version": "1.0",
  "id": "MEM-EXP-RUST-SLOTCACHE-001",
  "title": "Experiment: Differential Verification of Rust Host-Path Libraries Across 93 GPU Layers",
  "entry_type": "experiment",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "rust-host-path",
    "expert-slot-cache",
    "kimi-k3-moe"
  ],
  "tags": [
    "systems",
    "rust",
    "experiment",
    "verification"
  ],
  "created_at": "2026-10-25T19:00:00Z",
  "updated_at": "2026-10-25T19:30:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "commit:178d3c1 (wire_test_merged + K3Full93Chain)",
    "actor": "system_measurement",
    "recorded_at": "2026-10-25T19:30:00Z"
  },
  "relationships": [
    {
      "type": "depends_on",
      "target": "MEM-DEC-SLOTCACHE-PER-LAYER-001",
      "provenance": "commit:5f11928",
      "confidence": 1.0,
      "context_scope": "rust_host_path_verification",
      "created_at": "2026-10-25T19:00:00Z",
      "note": "Tests the Rust slotcache, poolinit, and supply integration."
    },
    {
      "type": "supports",
      "target": "MEM-CORR-PER-LAYER-PARTITION-002",
      "provenance": "commit:178d3c1",
      "confidence": 1.0,
      "context_scope": "expert_cache_design",
      "created_at": "2026-10-25T19:30:00Z",
      "note": "Confirms per-layer slotcache operation across all 92 MoE layers on GPU."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-25T19:00:00Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Initial experiment entry recorded from commit:178d3c1 (wire_test_merged + K3Full93Chain)"
    }
  ]
}
---

# Experiment: Differential Verification of Rust Host-Path Libraries Across 93 GPU Layers

## Summary
Executed 212 Rust unit tests, `wire_test_merged`, `whole-model-gpu`, and `K3Full93Chain` comparing `K3GPU_*=cpp` vs `K3GPU_*=rust` on 8 x RTX 3090 GPUs.

## Content
Tested all 8 Rust host-path crates (`copyq`, `slotcache`, `routetable`, `planner`, `poolinit`, `supply`, `loader`, `engcfg`) against the C++ reference implementation.

## Original Source Evidence
`K3Full93Chain` (`cpp` vs `rust` on 8x RTX 3090): `max_abs_diff = 0.0`, `mean_abs_diff = 0.0`, `2,576` resident experts + `93` trunks across `cuda:0..5`.

## Revision History
- `2026-10-25T19:00:00Z` (**system_measurement**, `created`): Initial experiment entry recorded from commit:178d3c1 (wire_test_merged + K3Full93Chain)
