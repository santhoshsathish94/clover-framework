---
{
  "schema_version": "1.0",
  "id": "MEM-CONCEPT-OBSERVER-EFFECT-MEASUREMENT-001",
  "title": "In-Container vs. Host-Level GPU Telemetry Namespace Boundary",
  "entry_type": "concept",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "container-namespace-telemetry",
    "gpu-profiling",
    "pid-namespace"
  ],
  "tags": [
    "systems",
    "linux",
    "gpu",
    "telemetry",
    "container"
  ],
  "created_at": "2026-10-21T14:00:00Z",
  "updated_at": "2026-10-21T14:00:00Z",
  "provenance": {
    "source_type": "observation",
    "source_ref": "docs/clover-kimi-k3-gpu-context.md",
    "actor": "system_measurement",
    "recorded_at": "2026-10-21T14:00:00Z"
  },
  "relationships": [
    {
      "type": "similar_to",
      "target": "MEM-CONCEPT-OBSERVER-BOUNDARY-001",
      "provenance": "docs/clover-kimi-k3-gpu-context.md",
      "confidence": 0.5,
      "context_scope": "terminology_analogy_only",
      "created_at": "2026-10-21T14:00:00Z",
      "note": "Shares the words 'inside' and 'outside' as a boundary metaphor, but refers to Linux PID namespaces."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-21T14:00:00Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Initial concept entry recorded from docs/clover-kimi-k3-gpu-context.md"
    }
  ]
}
---

# In-Container vs. Host-Level GPU Telemetry Namespace Boundary

## Summary
Technical systems fact: tools running inside an isolated container PID namespace only see processes inside that container, whereas `nvidia-smi` outside on the host sees all GPU processes.

## Content
In Linux container deployments, everything executed inside a container's PID namespace sees only the container's process table, while GPU VRAM allocations happen on the shared kernel driver outside the container namespace.

## Original Source Evidence
Measured on Ubuntu 24.04 (`nvidia-smi` host vs. container namespace): PID visibility differs between inside-container procfs and host-level NVML.

## Revision History
- `2026-10-21T14:00:00Z` (**system_measurement**, `created`): Initial concept entry recorded from docs/clover-kimi-k3-gpu-context.md
