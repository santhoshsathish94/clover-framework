---
{
  "schema_version": "1.0",
  "id": "MEM-OBS-ONTORAG-FANOUT-FLOODING-001",
  "title": "Observation: 1-Hop High-Cardinality Leaf Edges (`AFFECTS`) Drown Out 2-Hop Causal Chains When Sorted by Path Length First",
  "entry_type": "observation",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "1-hop-fanout-flooding",
    "stratified-priority-traversal",
    "ontology-guided-graphrag",
    "causal-chain-recall"
  ],
  "tags": [
    "ontorag",
    "graphrag",
    "traversal",
    "cve",
    "cwe",
    "capec",
    "observation"
  ],
  "created_at": "2026-10-25T14:00:00Z",
  "updated_at": "2026-10-26T04:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "docs/clover-ontology-graphrag-research-context.md#finding-2",
    "actor": "system_measurement",
    "recorded_at": "2026-10-25T14:00:00Z"
  },
  "relationships": [
    {
      "type": "exemplifies",
      "target": "MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#finding-2",
      "confidence": 0.95,
      "context_scope": "graph_traversal_budget",
      "created_at": "2026-10-25T14:00:00Z",
      "note": "Zero 2-hop CAPEC recall was caused by unstratified path-length sorting, not absence of CAPEC links in the graph."
    },
    {
      "type": "exemplifies",
      "target": "MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#finding-1",
      "confidence": 0.95,
      "context_scope": "graph_traversal_budget",
      "created_at": "2026-10-25T14:00:00Z",
      "note": "1-hop session expansion without `EXPLOITED_BY_ATTACK` in `traversal_edges` structurally could not reach 2-hop CAPEC nodes."
    },
    {
      "type": "similar_to",
      "target": "MEM-OBS-GLOBAL-CACHE-FLAT-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#finding-2",
      "confidence": 0.85,
      "context_scope": "shared_budget_eviction_analogy",
      "created_at": "2026-10-25T14:00:00Z",
      "note": "Both are shared-budget flooding failures: 208 1-hop CPE edges evict 2-hop causal paths just as 92 layers evict a shared global LRU cache."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-25T14:00:00Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Initial observation entry recorded from docs/clover-ontology-graphrag-research-context.md#finding-2"
    }
  ]
}
---

# Observation: 1-Hop High-Cardinality Leaf Edges (`AFFECTS`) Drown Out 2-Hop Causal Chains When Sorted by Path Length First

## Summary
Ordering candidate graph paths by `SORT LENGTH(p.edges), priority LIMIT K` allows high-cardinality 1-hop leaf edges (such as 64-208 `AFFECTS` product/host edges) to fill the entire `LIMIT K` before any 2-hop causal/remediation path (`CVE -> CWE -> CAPEC`) is emitted.

## Content
Executing the upstream `ontorag` replication package on `gpu` (`/home/usergpu/ontorag`) revealed two compounding defects (Findings 1 and 2): (1) `configs/default.yaml` omitted the Horn-derived `EXPLOITED_BY_ATTACK` (`CVE -> CAPEC`) edge from `traversal_edges`, forcing `CAPEC` attack patterns to be 2 hops away (`CVE --CAUSED_BY--> CWE --EXPLOITED_BY--> CAPEC`); and (2) `GraphTraversal.paths` sorted by `LENGTH(p.edges)` before `edge_priority` with `max_paths = 50`. For entities with many 1-hop leaf edges (`CVE-2023-44487` has 208 `AFFECTS` edges; `CVE-2023-4911` has 73; `CVE-2024-6387` has 64), 100% of the returned paths were 1-hop edges and 0 paths reached `CAPEC`.

## Original Source Evidence
Measured on `/home/usergpu/ontorag` (`diagnose_three_findings.py`, 2026-10-25):
- `CVE-2023-44487` (`208` `AFFECTS` edges): `50/50` paths are 1-hop `AFFECTS`; `0` 2-hop paths, `0` `CAPEC` paths.
- `CVE-2023-4911` (`73` `AFFECTS` edges): `50/50` paths are 1-hop `AFFECTS`; `0` 2-hop paths, `0` `CAPEC` paths.
- `CVE-2024-6387` (`64` `AFFECTS` edges): `50/50` paths are 1-hop (`2 CAUSED_BY + 48 AFFECTS`); `0` `CAPEC` paths.

## Revision History
- `2026-10-25T14:00:00Z` (**system_measurement**, `created`): Initial observation entry recorded from docs/clover-ontology-graphrag-research-context.md#finding-2
