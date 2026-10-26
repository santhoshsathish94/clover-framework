#!/usr/bin/env python3
"""Cycle 5 GPU Evaluation: Yann LeCun et al. (1998) Graph Transformer Network (GTN) Forward Scorer
(`logadd` Eq. 13-16), Discriminative Forward Loss (Eq. 17), & Exact Posterior Confidence `exp(-E_dforw)` (p. 39)
in Clover's Ontology GraphRAG (`k3_ontology_graph.py`) & Memory Graph (`k3_memory_graph.py`) on `cuda:7`.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import torch
import k3_ontology_graph as kog
import k3_memory_graph as kmg


def _get_code_provenance() -> dict:
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(_SCRIPT_DIR), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        git_sha = "standalone_checkout"
    return {
        "script_path": str(Path(__file__).resolve()),
        "git_head_sha": git_sha,
        "k3_ontology_graph_file": str(Path(kog.__file__).resolve()),
        "k3_ontology_graph_sha256_12": hashlib.sha256(Path(kog.__file__).read_bytes()).hexdigest()[:12],
        "k3_memory_graph_file": str(Path(kmg.__file__).resolve()),
        "k3_memory_graph_sha256_12": hashlib.sha256(Path(kmg.__file__).read_bytes()).hexdigest()[:12],
    }


def run_cycle5_eval() -> dict:
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(7) if torch.cuda.is_available() else "CPU"

    # 1. Ontology Graph GTN Forward (`logadd`) vs Viterbi (`min`) on Multi-Path Causal Corroboration
    #    In `build_default_clover_ontology_graph()`, `CVE/2024-6387` connects to `CAPEC/29` via:
    #      - 1-hop derived `EXPLOITED_BY_ATTACK` (cost 0.65)
    #      - 2-hop explicit `CAUSED_BY(CWE/362) -> EXPLOITED_BY(CAPEC/29)` (cost 0.55 + 0.55 = 1.10)
    #      - 3-hop explicit `CAUSED_BY(CWE/364) -> RELATED_TO(CWE/362) -> EXPLOITED_BY(CAPEC/29)` (cost 0.55 + 0.75 + 0.55 = 1.85)
    #    Whereas `CWE/364` or `CWE/362` has only a single 1-hop `CAUSED_BY` path (cost 0.55)!
    #    Let's see how Viterbi vs GTN Forward ranks `CAPEC/29` and `CWE/362` (both supported by multiple paths)!
    g = kog.build_default_clover_ontology_graph()
    trav = kog.CloverGraphTraversal(g)
    onto_res_6387 = trav.score_targets_gtn_forward([kog.EntityRef("CVE", "2024-6387")], hops=3, beta=1.5)
    onto_res_40711 = trav.score_targets_gtn_forward([kog.EntityRef("CVE", "2024-40711")], hops=3, beta=1.5)

    # 2. Controlled 6-Case Parallel-Corroboration vs Single-Path Lure Benchmark on GPU (`cuda:7`)
    #    Each case has a Single-Path Lure (1 path with penalty c_lure) and a True Corroborated Conclusion
    #    (K >= 2 parallel paths with penalties c_1..c_K where min(c_k) >= c_lure, so Viterbi picks the lure or ties!).
    parallel_cases = [
        ("Case1_2path_corroboration", [1.80], [2.00, 2.00]),
        ("Case2_3path_evolution_chain", [1.55], [1.85, 1.90, 2.10]),
        ("Case3_2path_exact_tie", [1.40], [1.40, 1.60]),
        ("Case4_4path_ontology_converge", [1.35], [1.75, 1.80, 1.85, 1.95]),
        ("Case5_3path_memory_refinement", [1.20], [1.45, 1.50, 1.55]),
        ("Case6_2path_close_margin", [1.65], [1.75, 1.80]),
    ]
    viterbi_correct = 0
    gtn_forward_correct = 0
    case_details = []
    for c_name, lure_pens, true_pens in parallel_cases:
        v_lure = min(lure_pens)
        v_true = min(true_pens)
        f_lure = kog.CloverGraphTraversal.gtn_logadd(lure_pens, beta=1.5)
        f_true = kog.CloverGraphTraversal.gtn_logadd(true_pens, beta=1.5)
        f_all = kog.CloverGraphTraversal.gtn_logadd([f_lure, f_true], beta=1.5)
        conf_true = math.exp(-1.5 * (f_true - f_all))
        if v_true < v_lure:
            viterbi_correct += 1
        if f_true < f_lure:
            gtn_forward_correct += 1
        case_details.append({
            "case": c_name,
            "viterbi_lure": v_lure,
            "viterbi_true": v_true,
            "gtn_forw_lure": round(f_lure, 4),
            "gtn_forw_true": round(f_true, 4),
            "gtn_posterior_conf_true": round(conf_true, 4),
        })

    # 3. Memory Graph GTN Forward Scorer (`compute_gtn_forward_confidence`) on Real Queries
    store = kmg.ensure_default_memory_tree()
    mem_queries = [
        (
            "MQ1_flat_cache_partition",
            "In a 48-stage pipeline with a shared global LRU pool, sweeping capacity gives a flat 19.4% hit rate. What structural change should we test?",
            "MEM-CORR-PER-LAYER-PARTITION-002",
        ),
        (
            "MQ2_inside_outside_correction",
            "What did our recorded memory say about 'everything happens inside' and how was the initial interpretation corrected?",
            "MEM-CORR-INSIDE-BOUNDARY-002",
        ),
        (
            "MQ3_ontorag_fanout_fix",
            "Why did sorting by 1-hop length(p.edges) first drown out 2-hop CWE->CAPEC paths and what correction fixed it?",
            "MEM-CORR-STRATIFIED-PRIORITY-TRAVERSAL-001",
        ),
        (
            "MQ4_cosine_035_dropout_fix",
            "Why did a hard 0.35 cosine threshold drop 30% of code queries to zero seeds and what hybrid RRF correction resolved it?",
            "MEM-CORR-HYBRID-RRF-SYMBOLIC-SEPARATION-001",
        ),
    ]
    mem_gtn_results = []
    for mq_id, mq_text, exp_corr_id in mem_queries:
        res = store.compute_gtn_forward_confidence(mq_text, max_seeds=4, max_depth=2, beta=1.2)
        top3_gtn = [c["id"] for c in res["candidates"][:3]]
        corr_entry = next((c for c in res["candidates"] if c["id"] == exp_corr_id), None)
        mem_gtn_results.append({
            "id": mq_id,
            "top_viterbi_id": res["top_viterbi_id"],
            "top_gtn_id": res["top_gtn_id"],
            "top3_gtn": top3_gtn,
            "expected_correction_in_top3": exp_corr_id in top3_gtn,
            "expected_correction_parallel_paths": corr_entry["num_parallel_paths"] if corr_entry else 0,
            "expected_correction_viterbi_pen": corr_entry["viterbi_penalty"] if corr_entry else None,
            "expected_correction_gtn_forw_pen": corr_entry["gtn_forward_penalty"] if corr_entry else None,
            "top_gtn_confidence": res["top_gtn_confidence"],
        })

    return {
        "code_provenance": _get_code_provenance(),
        "device": device,
        "gpu_name": gpu_name,
        "parallel_corroboration_vs_viterbi": {
            "viterbi_single_path_accuracy": f"{viterbi_correct}/{len(parallel_cases)}",
            "gtn_forward_logadd_accuracy": f"{gtn_forward_correct}/{len(parallel_cases)}",
            "details": case_details,
        },
        "ontology_gtn_forward_cve_2024_6387": {
            "top_viterbi": onto_res_6387["top_viterbi"],
            "top_gtn_forward": onto_res_6387["top_gtn_forward"],
            "top_gtn_confidence": onto_res_6387["top_gtn_confidence"],
            "targets": onto_res_6387["targets"][:4],
        },
        "memory_graph_gtn_forward": {
            "expected_correction_in_top3": f"{sum(1 for r in mem_gtn_results if r['expected_correction_in_top3'])}/{len(mem_gtn_results)}",
            "details": mem_gtn_results,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_cycle5_eval(), indent=2))
