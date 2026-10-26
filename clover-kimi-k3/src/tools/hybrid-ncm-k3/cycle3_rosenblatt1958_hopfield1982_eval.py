#!/usr/bin/env python3
"""Cycle 3 GPU Evaluation: Frank Rosenblatt (1958) Parasitic Zero-Sum Gamma-System +
John Hopfield (1982) Bipolar {-1, +1} Associative Constellation Cleanup & Familiarity Energy
on Clover's Memory Graph (`k3_memory_graph.py`) running on `cuda:7` (`NVIDIA GeForce RTX 3090`).
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/home/usergpu/merge_lf")

import torch
import k3_memory_graph as kmg


def run_cycle3_eval() -> dict:
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(7) if torch.cuda.is_available() else "CPU"
    store = kmg.ensure_default_memory_tree()

    # 1. Rosenblatt (1958) Alpha-System vs Parasitic Zero-Sum Gamma-System under Branch Imbalance
    #    The default memory tree has 27 `systems` entries vs 5 `philosophy` entries (5.4:1 imbalance).
    #    Evaluate 4 cross-branch queries that ask how both a `systems` principle/measurement and the
    #    `philosophy` observer-boundary / inside-outside inquiry connect in `top_seeds` (`max_seeds = 3`).
    cross_branch_queries = [
        (
            "CB1_observer_and_cache",
            "How does our systems measurement of global expert cache cyclic sweep eviction and per-layer partitioning relate to the philosophical observer-boundary concept in AGENTS.md Section 22?",
        ),
        (
            "CB2_instrument_and_inside",
            "Compare the zero-denominator instrument validity principle in recurrent KDA cache replay with the philosophical dialogue on 'everything happens inside' and inside-outside distinction.",
        ),
        (
            "CB3_synthetic_and_realism",
            "How do synthetic timing vs semantic meaning in Kimi K3 layer routing and strict external realism vs observer boundary in philosophy both address where an observation boundary is drawn?",
        ),
        (
            "CB4_ontorag_and_boundary",
            "Connect our ontology GraphRAG stratified priority traversal and hybrid RRF findings with our recorded philosophical inquiry on the observer boundary.",
        ),
    ]

    alpha_dual_branch_hits = 0
    gamma_dual_branch_hits = 0
    cross_branch_details = []

    for q_id, q_text in cross_branch_queries:
        store.use_rosenblatt_gamma = False
        scored_alpha = store._score_seed_entries(q_text)[:3]
        branches_alpha = {store._entry_macro_branch(store.entries[eid]) for _, eid, _ in scored_alpha}
        has_both_alpha = ("systems" in branches_alpha) and ("philosophy" in branches_alpha)
        if has_both_alpha:
            alpha_dual_branch_hits += 1

        store.use_rosenblatt_gamma = True
        scored_gamma = store._score_seed_entries(q_text)[:3]
        branches_gamma = {store._entry_macro_branch(store.entries[eid]) for _, eid, _ in scored_gamma}
        has_both_gamma = ("systems" in branches_gamma) and ("philosophy" in branches_gamma)
        if has_both_gamma:
            gamma_dual_branch_hits += 1

        cross_branch_details.append({
            "id": q_id,
            "alpha_top3": [(eid, round(sc, 2), store._entry_macro_branch(store.entries[eid])) for sc, eid, _ in scored_alpha],
            "gamma_top3": [(eid, round(sc, 2), store._entry_macro_branch(store.entries[eid])) for sc, eid, _ in scored_gamma],
            "alpha_dual_branch": has_both_alpha,
            "gamma_dual_branch": has_both_gamma,
        })

    # 2. Hopfield (1982) Bipolar {-1, +1} vs Unipolar {0, 1} Constellation Cleanup on GPU (`cuda:7`)
    vocab, constellations = store._build_hopfield_constellations()
    bipolar_jaccards = []
    unipolar_jaccards = []
    bipolar_exact = 0
    unipolar_exact = 0

    for c_set in constellations:
        sorted_feats = sorted(c_set)
        k_cue = max(1, int(round(len(sorted_feats) * 0.35)))
        partial_cue = sorted_feats[:k_cue]
        res_bip = store.hopfield_associative_cleanup(partial_cue, mode="bipolar", steps=2, device=device)
        res_uni = store.hopfield_associative_cleanup(partial_cue, mode="unipolar", steps=2, device=device)
        bipolar_jaccards.append(res_bip["best_jaccard_overlap"])
        unipolar_jaccards.append(res_uni["best_jaccard_overlap"])
        if res_bip["best_jaccard_overlap"] >= 0.70:
            bipolar_exact += 1
        if res_uni["best_jaccard_overlap"] >= 0.70:
            unipolar_exact += 1

    # 3. Hopfield Familiarity Recognition AUC on GPU (`cuda:7`) across 10 familiar in-memory queries vs 10 novel out-of-distribution queries
    familiar_queries = [
        "What did our recorded memory say about 'everything happens inside' and how was the initial interpretation corrected?",
        "Why did the global expert cache stay flat at 23.3% across capacities 512 to 4096 and how did per-layer partitioning fix it?",
        "Why did repeating a token 64 times inside one sequence give 0 hits on the expert result cache while request replay gave 3.27x?",
        "Why did 1-hop AFFECTS leaf edges drown out 2-hop CWE->CAPEC paths in ontorag and how does stratified priority traversal fix it?",
        "Why did a hard 0.35 cosine threshold drop 30% of code queries to zero seeds and how does hybrid BM25+Dense RRF solve it?",
        "How does statement-boundary generative compilation catch short-prefix shadowing in parse_music and save 38.5% of doomed chars?",
        "Do isolated micro-benchmarks only over-predict in-layer performance, or can they mispredict in both directions (15.2x/8.6x and 1.68x/2.2x)?",
        "Why must CPU 4-bit quantisation verdicts (-2.4% vs -18.3% LB_Q4) be scoped to hardware and kernel while GPU trunk 4-bit A/B is marked unmeasured?",
        "Why did a trailing VS Code <context> user envelope wipe tool_rounds mid-turn in /v1/chat/completions?",
        "How does the competing strict external realism hypothesis challenge the observer-boundary concept?",
    ]
    novel_queries = [
        "Calculate the orbital period of a geostationary satellite orbiting Earth at an altitude of 35,786 km.",
        "Write a SQL query that groups monthly e-commerce orders by customer country and computes the median basket value.",
        "Explain the mechanism of CRISPR-Cas9 guide RNA protospacer adjacent motif (PAM) recognition in Streptococcus pyogenes.",
        "How did the Treaty of Westphalia in 1648 reshape territorial sovereignty in the Holy Roman Empire?",
        "Derive the Black-Scholes partial differential equation for a European call option under geometric Brownian motion.",
        "What are the thermodynamic stages of an ideal Rankine steam power cycle with superheat and reheat?",
        "How does Dijkstra's shortest-path algorithm with a Fibonacci heap achieve O(E + V log V) time complexity?",
        "Compare Gothic flying buttresses at Chartres Cathedral with Romanesque barrel vaults.",
        "Calculate the pH of a 0.05 M aqueous solution of acetic acid given Ka = 1.8e-5 at 25 degrees Celsius.",
        "Explain how plate tectonics and slab pull drive subduction zone volcanism along the Pacific Ring of Fire.",
    ]

    t0 = time.perf_counter()
    fam_scores = [store.compute_hopfield_familiarity(q, device=device)["familiarity_score"] for q in familiar_queries]
    nov_scores = [store.compute_hopfield_familiarity(q, device=device)["familiarity_score"] for q in novel_queries]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    wins = 0.0
    pairs = len(fam_scores) * len(nov_scores)
    for fs in fam_scores:
        for ns in nov_scores:
            if fs > ns:
                wins += 1.0
            elif fs == ns:
                wins += 0.5
    auc = wins / max(1, pairs)

    return {
        "device": device,
        "gpu_name": gpu_name,
        "rosenblatt_gamma_system": {
            "branch_imbalance": "27 systems entries vs 5 philosophy entries (5.4 : 1)",
            "alpha_system_dual_branch_top3": f"{alpha_dual_branch_hits}/{len(cross_branch_queries)}",
            "gamma_system_dual_branch_top3": f"{gamma_dual_branch_hits}/{len(cross_branch_queries)}",
            "details": cross_branch_details,
        },
        "hopfield_bipolar_vs_unipolar_gpu": {
            "num_units_N": len(vocab),
            "num_constellations_M": len(constellations),
            "bipolar_mean_jaccard": round(sum(bipolar_jaccards) / max(1, len(bipolar_jaccards)), 4),
            "unipolar_mean_jaccard": round(sum(unipolar_jaccards) / max(1, len(unipolar_jaccards)), 4),
            "bipolar_high_recall_constellations": f"{bipolar_exact}/{len(constellations)}",
            "unipolar_high_recall_constellations": f"{unipolar_exact}/{len(constellations)}",
        },
        "hopfield_familiarity_gate_gpu": {
            "familiar_mean_score": round(sum(fam_scores) / len(fam_scores), 4),
            "novel_mean_score": round(sum(nov_scores) / len(nov_scores), 4),
            "familiar_Pass_at_0_35": f"{sum(1 for s in fam_scores if s >= 0.35)}/{len(fam_scores)}",
            "novel_FP_at_0_35": f"{sum(1 for s in nov_scores if s >= 0.35)}/{len(nov_scores)}",
            "roc_auc": round(auc, 4),
            "ms_per_query_on_gpu": round(elapsed_ms / (len(familiar_queries) + len(novel_queries)), 3),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_cycle3_eval(), indent=2))
