---
{
  "schema_version": "1.0",
  "id": "MEM-CORR-STRATIFIED-PRIORITY-TRAVERSAL-001",
  "title": "Correction: Stratified Edge-Priority Ordering and Per-Relation Leaf Caps Restore 2-Hop Causal Chains",
  "entry_type": "correction",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "stratified-priority-traversal",
    "1-hop-fanout-flooding",
    "ontology-guided-graphrag",
    "causal-chain-recall"
  ],
  "tags": [
    "ontorag",
    "graphrag",
    "correction",
    "traversal",
    "verified"
  ],
  "created_at": "2026-10-25T16:30:00Z",
  "updated_at": "2026-10-26T04:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "docs/clover-ontology-graphrag-research-context.md#step-5",
    "actor": "human_and_assistant",
    "recorded_at": "2026-10-25T16:30:00Z"
  },
  "relationships": [
    {
      "type": "refines",
      "target": "MEM-OBS-ONTORAG-FANOUT-FLOODING-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#step-5",
      "confidence": 1.0,
      "context_scope": "graph_traversal_budget",
      "created_at": "2026-10-25T16:30:00Z",
      "note": "Resolves 1-hop `AFFECTS` flooding via stratified edge-priority ordering and `max_affects_per_seed = 4`."
    },
    {
      "type": "derived_from",
      "target": "MEM-OBS-ONTORAG-FANOUT-FLOODING-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#finding-2",
      "confidence": 1.0,
      "context_scope": "graph_traversal_budget",
      "created_at": "2026-10-25T16:30:00Z",
      "note": "Directly motivated by measuring 0 CAPEC paths on `CVE-2023-44487`, `CVE-2023-4911`, and `CVE-2024-6387`."
    },
    {
      "type": "challenges",
      "target": "MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#step-5",
      "confidence": 1.0,
      "context_scope": "multi_hop_vulnerability_retrieval",
      "created_at": "2026-10-25T16:30:00Z",
      "note": "Demonstrates that explicit priority-stratified symbolic path traversal (`37.1` paths, `84.5%` recall) is required over flat chunk similarity (`0.0` paths, `37.3%` recall)."
    },
    {
      "type": "similar_to",
      "target": "MEM-CORR-PER-LAYER-PARTITION-002",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#section-3",
      "confidence": 0.9,
      "context_scope": "partitioned_quota_architecture",
      "created_at": "2026-10-25T16:30:00Z",
      "note": "Both replace an unpartitioned global competition pool with per-class/per-layer quotas so high-volume items cannot evict critical items."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-25T16:30:00Z",
      "actor": "human_and_assistant",
      "action": "created",
      "reason": "Initial correction entry recorded from docs/clover-ontology-graphrag-research-context.md#step-5"
    }
  ]
}
---

# Correction: Stratified Edge-Priority Ordering and Per-Relation Leaf Caps Restore 2-Hop Causal Chains

## Summary
Sorting candidate paths by `edge_priority` (putting causal/remediation edges before leaf `AFFECTS` edges) and enforcing a stratified per-seed cap on leaf edges (`max_affects_per_seed = 4`) eliminates 1-hop fan-out flooding, raising full `CVE+CWE+CAPEC` chain coverage from 63.3% to 76.7% and Gold ID recall from 78.9% to 84.5%.

## Content
Refines and resolves the traversal bottleneck observed in MEM-OBS-ONTORAG-FANOUT-FLOODING-001. Merely raising `LIMIT K` is fragile when a node has hundreds of 1-hop leaf edges. Instead, by (1) sorting by `edge_priority` group (causal/formula/remediation before leaf `AFFECTS`) ahead of path length, (2) enforcing a per-seed cap on leaf `AFFECTS` edges (`max_affects_per_seed = 4`), and (3) including Horn-derived shortcut edges (`EXPLOITED_BY_ATTACK`), `CloverGraphTraversal` guarantees quota for 2-hop explanatory chains regardless of 1-hop fan-out cardinality.

## Original Source Evidence
Controlled 3-way benchmark on `gpu` (`bench_retrieval_3way.py`, `N=30` Exploratory queries):
- `ontorag_arm_a` (length-first sorting): Full `CVE+CWE+CAPEC` chain = `19/30` (`63.3%`), Mean `[P#]` paths = `28.3`, Gold ID Recall = `78.9%`.
- `ontorag_arm_b` (stratified priority + leaf cap): Full `CVE+CWE+CAPEC` chain = `23/30` (`76.7%`), Mean `[P#]` paths = `37.1`, Gold ID Recall = `84.5%`.

## Revision History
- `2026-10-25T16:30:00Z` (**human_and_assistant**, `created`): Initial correction entry recorded from docs/clover-ontology-graphrag-research-context.md#step-5
