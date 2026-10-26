#!/usr/bin/env python3
"""Cycle 2 GPU Evaluation: Alan Turing (1950) Symbolic Channel & M0/M1 Stratification +
Newell & Simon (1956) CSm 3-Number Signature Pre-Filter (*2.07 Contraction Fix) & VCt Contraction
in Clover's Ontology GraphRAG (`k3_ontology_graph.py`).
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/home/usergpu/merge_lf")

import torch
import k3_ontology_graph as kog


def build_large_cve_stress_graph() -> kog.CloverOntologyGraph:
    g = kog.build_default_clover_ontology_graph()
    prov = kog.Provenance.from_source("NVD", "https://nvd.nist.gov/vuln/detail/CVE-2024-6387", "configurations")
    new_verts = []
    new_edges = []
    for idx in range(200):
        cpe_v = kog.EntityRef("CPE", f"a:openbsd:openssh:9.7p{idx}")
        new_verts.append(kog.Vertex(cpe_v, {"name": f"OpenBSD OpenSSH 9.7p{idx}", "version": f"9.7p{idx}"}))
        new_edges.append(kog.Edge("AFFECTS", kog.EntityRef("CVE", "2024-6387"), cpe_v, prov))
        new_edges.append(
            kog.Edge("RELATED_TO", cpe_v, kog.EntityRef("CPE", "a:openbsd:openssh:*"), prov, {"nature": "VersionOf"})
        )
    g.write_vertices(new_verts)
    g.write_edges(new_edges)
    g.run_inference()
    return g


def build_theorem_bank_68() -> list[tuple[str, tuple]]:
    bank: list[tuple[str, tuple]] = [
        ("*1.2_taut", ("IMP", ("OR", "p", "p"), "p")),
        ("*1.3_add", ("IMP", "q", ("OR", "p", "q"))),
        ("*1.4_perm", ("IMP", ("OR", "p", "q"), ("OR", "q", "p"))),
        ("*1.5_assoc", ("IMP", ("OR", "p", ("OR", "q", "r")), ("OR", "q", ("OR", "p", "r")))),
        ("*1.6_sum", ("IMP", ("IMP", "q", "r"), ("IMP", ("OR", "p", "q"), ("OR", "p", "r")))),
        ("*2.07_add_left", ("IMP", "p", ("OR", "p", "q"))),
        ("STEM_REFLECT_CIRCLE", ("EQ", ("REFLECT", ("CIRCLE", "x", "y", "r"), ("PT", "p", "q")), ("TRIPLE", "p", "q", "r"))),
        ("STEM_COMMON_DIV", ("EQ", ("COMMON_DIV", "x", "y"), ("DIVISORS", ("GCD", "x", "y")))),
    ]
    vars_pool = ["p", "q", "r", "s", "t", "u"]
    for i in range(60):
        v1 = vars_pool[i % 6]
        v2 = vars_pool[(i + 1) % 6]
        depth_l = (i % 5) + 1
        depth_r = ((i // 5) % 5) + 1
        left: tuple | str = v1
        for d in range(depth_l):
            left = ("OR" if d % 2 == 0 else "AND", left, vars_pool[(i + d) % 6])
        right: tuple | str = v2
        for d in range(depth_r):
            right = ("IMP" if d % 2 == 0 else "OR", right, vars_pool[(i + d + 2) % 6])
        bank.append((f"THM_SYN_{i:02d}", ("IMP", left, right)))
    return bank


def run_cycle2_eval() -> dict:
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(7) if torch.cuda.is_available() else "CPU"
    bank = build_theorem_bank_68()
    queries = [
        ("Q1_uncontracted_perm", ("IMP", ("OR", "a", "b"), ("OR", "b", "a")), "*1.4_perm"),
        ("Q2_uncontracted_sum", ("IMP", ("IMP", "b", "c"), ("IMP", ("OR", "a", "b"), ("OR", "a", "c"))), "*1.6_sum"),
        ("Q3_uncontracted_circle_pq", ("EQ", ("REFLECT", ("CIRCLE", "x", "y", "r"), ("PT", "3", "2")), ("TRIPLE", "3", "2", "r")), "STEM_REFLECT_CIRCLE"),
        ("Q4_uncontracted_div_xy", ("EQ", ("COMMON_DIV", "12", "18"), ("DIVISORS", ("GCD", "12", "18"))), "STEM_COMMON_DIV"),
        ("Q5_contracted_2_07", ("IMP", "a", ("OR", "a", "a")), "*2.07_add_left"),
        ("Q6_contracted_1_3", ("IMP", "a", ("OR", "a", "a")), "*1.3_add"),
        ("Q7_contracted_circle_p_eq_q", ("EQ", ("REFLECT", ("CIRCLE", "x", "y", "r"), ("PT", "a", "a")), ("TRIPLE", "a", "a", "r")), "STEM_REFLECT_CIRCLE"),
        ("Q8_contracted_div_x_eq_y", ("EQ", ("COMMON_DIV", "a", "a"), ("DIVISORS", ("GCD", "a", "a"))), "STEM_COMMON_DIV"),
    ]

    strict_j_hits = 0
    contracted_j_hits = 0
    total_checked = 0
    total_lmc_calls = 0

    for _, q_expr, exp_thm in queries:
        res_strict = kog.CloverSymbolicVerifier.match_formula_csm_lmc(q_expr, bank, allow_vct_contraction=False)
        res_vct = kog.CloverSymbolicVerifier.match_formula_csm_lmc(q_expr, bank, allow_vct_contraction=True)
        if exp_thm in {m[0] for m in res_strict["matched"]}:
            strict_j_hits += 1
        if exp_thm in {m[0] for m in res_vct["matched"]}:
            contracted_j_hits += 1
        total_checked += res_vct["csm_checked"]
        total_lmc_calls += res_vct["lmc_calls"]

    g = build_large_cve_stress_graph()
    trav = kog.CloverGraphTraversal(g)
    seeds = [
        kog.EntityRef("CVE", "2024-6387"),
        kog.EntityRef("CVE", "2024-40711"),
        kog.EntityRef("CLOVER_ENTITY", "CLOVER-KIMI-K3"),
        kog.EntityRef("STEM_CONCEPT", "STEM-THIN-FILM-INTERFERENCE"),
    ]

    t0 = time.perf_counter()
    paths_no_csm = trav.paths(seeds, hops=2, max_paths=24, use_csm_filter=False)
    ms_no_csm = (time.perf_counter() - t0) * 1000.0
    stats_no_csm = dict(trav.last_traversal_stats)

    t1 = time.perf_counter()
    paths_with_csm = trav.paths(seeds, hops=2, max_paths=24, use_csm_filter=True)
    ms_with_csm = (time.perf_counter() - t1) * 1000.0
    stats_with_csm = dict(trav.last_traversal_stats)

    desc_no_csm = [p.describe() for p in paths_no_csm]
    desc_with_csm = [p.describe() for p in paths_with_csm]

    return {
        "device": device,
        "gpu_name": gpu_name,
        "csm_filtered_pct": round(100.0 * (total_checked - total_lmc_calls) / max(1, total_checked), 2),
        "strict_j_recall": f"{strict_j_hits}/{len(queries)}",
        "vct_contracted_j_recall": f"{contracted_j_hits}/{len(queries)}",
        "dfs_steps_without_csm": stats_no_csm["dfs_steps"],
        "dfs_steps_with_csm": stats_with_csm["dfs_steps"],
        "dfs_step_reduction_factor": round(stats_no_csm["dfs_steps"] / max(1, stats_with_csm["dfs_steps"]), 2),
        "ms_without_csm": round(ms_no_csm, 3),
        "ms_with_csm": round(ms_with_csm, 3),
        "exact_path_list_match": desc_no_csm == desc_with_csm,
    }


if __name__ == "__main__":
    print(json.dumps(run_cycle2_eval(), indent=2))
