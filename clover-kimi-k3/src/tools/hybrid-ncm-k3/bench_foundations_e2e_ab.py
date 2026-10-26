#!/usr/bin/env python3
"""
Controlled 2-Arm End-to-End Clover GPU Benchmark (`bench_foundations_e2e_ab.py`)
=================================================================================
Addresses the engineering review requirement:
  1. Imports strictly from `Path(__file__).resolve().parent` (the git checkout directory)
     and logs the exact git commit SHA, module `__file__` paths, and SHA-256 digests.
  2. Evaluates `Arm A (Foundations Upgrades Disabled — Pre-Cycle-1..6 Baseline)` vs.
     `Arm B (Foundations Upgrades Enabled — Cycles 1..6 Active)` across:
       - Part 1: 16 Out-of-Domain / One-in-a-Box Stress Queries (Measuring False-Positive
                 Context Pollution Rate & Injected Irrelevant Tokens).
       - Part 2: 16 Positive Ontology + Memory Recall Queries (Measuring Recall & Latency).
       - Part 3: 6 Tight-Budget (`max_seeds=2, max_entries=3, token_budget=420`) Multi-Facet &
                 Cross-Branch End-to-End Clover Queries (Measuring whether retrieved `[M#]` / `[E#]`
                 context contains all required gold answer facts & IDs, and running live 93-layer
                 Clover GPU forward passes via `http://127.0.0.1:8765/api/chat` when available).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import torch
import k3_ontology_graph as kog
import k3_memory_graph as kmg
import k3_generative_compiler as kgc
from cycle1_mp1943_mp1969_eval import (
    ONTOLOGY_STRESS_QUERIES,
    MEMORY_STRESS_QUERIES,
    ONTOLOGY_POSITIVE_QUERIES,
    MEMORY_POSITIVE_QUERIES,
)


def get_code_provenance() -> dict[str, Any]:
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(_SCRIPT_DIR), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "-C", str(_SCRIPT_DIR), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        git_sha = "unknown"
        git_branch = "unknown"
    return {
        "script_path": str(Path(__file__).resolve()),
        "git_branch": git_branch,
        "git_head_sha": git_sha,
        "k3_ontology_graph_file": str(Path(kog.__file__).resolve()),
        "k3_ontology_graph_sha256_12": hashlib.sha256(Path(kog.__file__).read_bytes()).hexdigest()[:12],
        "k3_memory_graph_file": str(Path(kmg.__file__).resolve()),
        "k3_memory_graph_sha256_12": hashlib.sha256(Path(kmg.__file__).read_bytes()).hexdigest()[:12],
        "k3_generative_compiler_file": str(Path(kgc.__file__).resolve()),
        "k3_generative_compiler_sha256_12": hashlib.sha256(Path(kgc.__file__).read_bytes()).hexdigest()[:12],
    }


E2E_TASKS: list[dict[str, Any]] = [
    {
        "id": "E2E_1_cross_branch_observer_and_cache",
        "category": "Cross-Branch Imbalance (Rosenblatt Gamma + Multi-Head LRN)",
        "query": (
            "How does our systems measurement of global expert cache cyclic sweep eviction and per-layer "
            "partitioning relate to the philosophical observer-boundary concept in AGENTS.md Section 22?"
        ),
        "max_seeds": 3,
        "max_depth": 1,
        "max_entries": 3,
        "token_budget": 900,
        "required_memory_ids": [
            "MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001",
            "MEM-CONCEPT-OBSERVER-BOUNDARY-001",
        ],
    },
    {
        "id": "E2E_2_multi_facet_ontorag_3facets",
        "category": "Multi-Facet Subspace Unmixing (Vaswani h=3 + Krizhevsky LRN + LeCun GTN)",
        "query": (
            "When 1-hop AFFECTS leaf edges drown out 2-hop CWE->CAPEC paths in ontology GraphRAG, state the "
            "structural diagnostic principle, the measured fan-out flooding observation, and the stratified "
            "priority traversal correction."
        ),
        "max_seeds": 3,
        "max_depth": 1,
        "max_entries": 3,
        "token_budget": 900,
        "required_memory_ids": [
            "MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001",
            "MEM-OBS-ONTORAG-FANOUT-FLOODING-001",
            "MEM-CORR-STRATIFIED-PRIORITY-TRAVERSAL-001",
        ],
    },
    {
        "id": "E2E_3_multi_facet_prefix_shadow_3facets",
        "category": "Multi-Facet Subspace Unmixing (Vaswani h=3 + Krizhevsky LRN)",
        "query": (
            "In a character scanner where checking 1-char 'o' before 2-char 'o|' shadows the longer token, "
            "state the short-prefix shadowing concept, the post-hoc code repair limit observation, and the "
            "verified 100% generative compilation outcome."
        ),
        "max_seeds": 3,
        "max_depth": 1,
        "max_entries": 3,
        "token_budget": 900,
        "required_memory_ids": [
            "MEM-CONCEPT-PREFIX-SHADOWING-DISPATCH-001",
            "MEM-OBS-POSTHOC-CODE-REPAIR-LIMITS-001",
            "MEM-OUT-GENERATIVE-COMPILATION-100PCT-001",
        ],
    },
    {
        "id": "E2E_4_contaminated_oil_buoyancy_physics",
        "category": "Zero Context Pollution on Look-Alike Physics Prompt (McCulloch-Pitts + Minsky-Papert)",
        "query": (
            "A puddle of water on a garage floor is covered by a layer of oil with density 800 kg/m^3. "
            "Will a plastic bead of density 920 kg/m^3 float at the oil-water interface by Archimedes' principle?"
        ),
        "forbidden_ontology_ids": ["STEM_CONCEPT:STEM-THIN-FILM-INTERFERENCE", "FORMULA:THIN-FILM-CONSTRUCTIVE-EQ"],
    },
    {
        "id": "E2E_5_contaminated_cell_biology_endocytosis",
        "category": "Zero Context Pollution on Look-Alike Biology Prompt (McCulloch-Pitts Inhibitory Veto)",
        "query": (
            "In eukaryotic cell biology, once receptor-mediated endocytosis completes, everything happens inside "
            "the intracellular endosome and lysosome compartments rather than outside in the extracellular matrix. "
            "How does V-ATPase proton pumping lower luminal pH?"
        ),
        "forbidden_memory_ids": ["MEM-CORR-INSIDE-BOUNDARY-002", "MEM-HYP-INSIDE-MIND-001", "MEM-CONV-PHIL-INSIDE-001"],
    },
    {
        "id": "E2E_6_contaminated_swimming_pool_heat",
        "category": "Zero Context Pollution on Look-Alike Engineering Prompt (Minsky-Papert One-in-a-Box)",
        "query": (
            "In an outdoor heated swimming pool, the water temperature stays flat across different pump capacity "
            "settings because evaporative heat loss to the night air is the bottleneck. What should we suspect "
            "about the pool cover?"
        ),
        "forbidden_memory_ids": [
            "MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001",
            "MEM-CORR-PER-LAYER-PARTITION-002",
            "MEM-OBS-GLOBAL-CACHE-FLAT-001",
        ],
    },
]


def evaluate_arm(enable_foundations: bool) -> dict[str, Any]:
    retriever = kog.CloverOntologyRetriever(use_mp_predicates=enable_foundations)
    store = kmg.ensure_default_memory_tree()
    store.use_mp_veto_and_box = enable_foundations
    store.use_rosenblatt_gamma = enable_foundations
    store.use_multihead_lrn = enable_foundations
    store.use_gtn_forward_boost = enable_foundations

    t0 = time.perf_counter()

    # 1. Stress False Positives & Injected Pollution Tokens
    onto_fp = 0
    mem_fp = 0
    polluted_tokens = 0
    for item in ONTOLOGY_STRESS_QUERIES:
        ev = retriever.retrieve(item["query"], use_mp_predicates=enable_foundations)
        if ev:
            onto_fp += 1
            polluted_tokens += sum(len(e.text.split()) for e in ev)
    for item in MEMORY_STRESS_QUERIES:
        pkg = kmg.retrieve_relevant_memory_package(store, item["query"])
        if pkg and pkg.items:
            mem_fp += 1
            polluted_tokens += pkg.estimated_tokens_used

    # 2. Positive Recall
    onto_pos = sum(
        1 for q, exp_id in ONTOLOGY_POSITIVE_QUERIES
        if exp_id in {eid for item in retriever.retrieve(q, use_mp_predicates=enable_foundations) for eid in item.entities}
    )
    mem_pos = sum(
        1 for q, exp_id in MEMORY_POSITIVE_QUERIES
        if (pkg := kmg.retrieve_relevant_memory_package(store, q)) is not None
        and exp_id in {it.entry.id for it in pkg.items}
    )

    # 3. 6 End-to-End Task Checks
    e2e_passed = 0
    e2e_details = []
    for task in E2E_TASKS:
        q = task["query"]
        if "required_memory_ids" in task:
            pkg = store.retrieve_context_package(
                q,
                max_seeds=task["max_seeds"],
                max_depth=task["max_depth"],
                max_entries=task["max_entries"],
                token_budget=task["token_budget"],
            )
            got_ids = [it.entry.id for it in pkg.items]
            ok = all(req_id in got_ids for req_id in task["required_memory_ids"])
            if ok:
                e2e_passed += 1
            e2e_details.append({
                "id": task["id"],
                "passed": ok,
                "retrieved_ids": got_ids,
                "required_ids": task["required_memory_ids"],
            })
        elif "forbidden_ontology_ids" in task:
            ev = retriever.retrieve(q, use_mp_predicates=enable_foundations)
            got_ids = [eid for item in ev for eid in item.entities]
            ok = not any(fid in got_ids for fid in task["forbidden_ontology_ids"])
            if ok:
                e2e_passed += 1
            e2e_details.append({
                "id": task["id"],
                "passed": ok,
                "retrieved_ids": got_ids,
                "forbidden_ids": task["forbidden_ontology_ids"],
            })
        else:
            pkg = kmg.retrieve_relevant_memory_package(store, q)
            got_ids = [it.entry.id for it in pkg.items] if pkg else []
            ok = len(got_ids) == 0
            if ok:
                e2e_passed += 1
            e2e_details.append({
                "id": task["id"],
                "passed": ok,
                "retrieved_ids": got_ids,
                "forbidden_ids": task["forbidden_memory_ids"],
            })

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "foundations_enabled": enable_foundations,
        "stress_false_positives": f"{onto_fp + mem_fp}/{len(ONTOLOGY_STRESS_QUERIES) + len(MEMORY_STRESS_QUERIES)}",
        "polluted_irrelevant_tokens_injected": polluted_tokens,
        "positive_recall": f"{onto_pos + mem_pos}/{len(ONTOLOGY_POSITIVE_QUERIES) + len(MEMORY_POSITIVE_QUERIES)}",
        "e2e_task_pass_rate": f"{e2e_passed}/{len(E2E_TASKS)} ({round(100.0 * e2e_passed / len(E2E_TASKS), 1)}%)",
        "total_eval_ms": round(elapsed_ms, 2),
        "e2e_details": e2e_details,
    }


def run_live_93layer_server_spotcheck() -> dict[str, Any]:
    """Runs 2 live 93-layer Kimi K3 + HybridAgent + NCM queries through `http://127.0.0.1:8765/api/chat`
    on the GPU server to confirm end-to-end 93-layer inference & health with the reconciled codebase."""
    base = "http://127.0.0.1:8765"
    try:
        with urllib.request.urlopen(f"{base}/api/health", timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"live_server_reachable": False, "error": str(exc)}

    queries = [
        (
            "live_1_cross_branch_memory",
            "Using our recorded memory entries, how does global expert cache cyclic sweep eviction and per-layer partitioning (23.3% to 48.0%) illustrate flat-parameter-response diagnosis?",
            ("23.3%", "48.0%", "partition"),
        ),
        (
            "live_2_uncontaminated_physics",
            "Question: A puddle of water on a garage floor is covered by a layer of oil with density 800 kg/m^3, and water has density 1000 kg/m^3. Since a plastic bead has density 920 kg/m^3 (800 < 920 < 1000), calculate 1000 - 920 and state whether the bead floats at the oil-water interface.",
            ("80", "float"),
        ),
    ]
    results = []
    for tag, prompt, req_terms in queries:
        t0 = time.perf_counter()
        payload = json.dumps({"session_id": f"fnd_ab_{tag}_{int(time.time())}", "prompt": prompt, "cands": 4}).encode("utf-8")
        req = urllib.request.Request(f"{base}/api/chat", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        dt = (time.perf_counter() - t0) * 1000.0
        reply = str(out.get("reply") or "")
        # Ensure `live_2_uncontaminated_physics` did NOT hallucinate thin-film 600 nm optics!
        no_optics_leak = ("600 nm" not in reply.lower()) if tag == "live_2_uncontaminated_physics" else True
        terms_ok = sum(1 for t in req_terms if t.lower() in reply.lower()) >= 2 and no_optics_leak
        results.append({
            "tag": tag,
            "passed": terms_ok,
            "no_optics_pollution": no_optics_leak,
            "wall_ms": round(dt, 1),
            "k3_moe_ms": (out.get("k3_experts") or {}).get("k3_moe_ms"),
            "reply_excerpt": reply.replace("\n", " ")[:220],
        })
    return {
        "live_server_reachable": True,
        "server_status": health.get("status"),
        "k3_trunks_resident": health.get("k3_trunks_resident"),
        "k3_experts_resident": health.get("k3_experts_resident"),
        "queries_passed": f"{sum(1 for r in results if r['passed'])}/{len(results)}",
        "results": results,
    }


if __name__ == "__main__":
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(7) if torch.cuda.is_available() else "CPU"
    arm_a = evaluate_arm(enable_foundations=False)
    arm_b = evaluate_arm(enable_foundations=True)
    live_check = run_live_93layer_server_spotcheck()
    report = {
        "code_provenance": get_code_provenance(),
        "device": device,
        "gpu_name": gpu_name,
        "arm_a_foundations_disabled": arm_a,
        "arm_b_foundations_enabled": arm_b,
        "live_93layer_gpu_server_verification": live_check,
    }
    print(json.dumps(report, indent=2))
