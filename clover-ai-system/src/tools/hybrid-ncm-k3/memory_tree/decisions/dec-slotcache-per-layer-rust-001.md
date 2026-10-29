---
{
  "schema_version": "1.0",
  "id": "MEM-DEC-SLOTCACHE-PER-LAYER-001",
  "title": "Decision: Implement Per-Layer GPU Expert SlotCache in Rust (`libslotcache.so`)",
  "entry_type": "decision",
  "epistemic_status": "observed",
  "status": "active",
  "concepts": [
    "expert-slot-cache",
    "per-layer-partitioning",
    "rust-host-path"
  ],
  "tags": [
    "systems",
    "rust",
    "slotcache",
    "decision"
  ],
  "created_at": "2026-10-24T10:00:00Z",
  "updated_at": "2026-10-25T18:00:00Z",
  "provenance": {
    "source_type": "document",
    "source_ref": "clover-kimi-k3/src/rust/slotcache/CONTRACT.md",
    "actor": "human_and_assistant",
    "recorded_at": "2026-10-24T10:00:00Z"
  },
  "relationships": [
    {
      "type": "depends_on",
      "target": "MEM-CORR-PER-LAYER-PARTITION-002",
      "provenance": "clover-kimi-k3/src/rust/slotcache/CONTRACT.md",
      "confidence": 1.0,
      "context_scope": "expert_cache_design",
      "created_at": "2026-10-24T10:00:00Z",
      "note": "Per-layer slot partitioning directly implements the finding from MEM-CORR-PER-LAYER-PARTITION-002."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-24T10:00:00Z",
      "actor": "human_and_assistant",
      "action": "created",
      "reason": "Initial decision entry recorded from clover-kimi-k3/src/rust/slotcache/CONTRACT.md"
    }
  ]
}
---

# Decision: Implement Per-Layer GPU Expert SlotCache in Rust (`libslotcache.so`)

## Summary
Architectural decision to implement the per-layer GPU expert slot tracker and LRU eviction table as a selectable Rust C-ABI library (`libslotcache.so`, `K3GPU_SLOTCACHE=rust`).

## Content
Maintains per-(card, layer) direct-mapped lookup tables (`eid_to_slot[896]`), resident slot metadata, and tick-based LRU victim selection across all 92 MoE layers (`k3_slotcache_*` C ABI).

## Original Source Evidence
Commit 5f11928 (`clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_gpu_experts_lib.cu`): wired `libslotcache.so`, `libpoolinit.so`, and `libsupply.so` into `libk3_gpu_experts.so`.

## Revision History
- `2026-10-24T10:00:00Z` (**human_and_assistant**, `created`): Initial decision entry recorded from clover-kimi-k3/src/rust/slotcache/CONTRACT.md
