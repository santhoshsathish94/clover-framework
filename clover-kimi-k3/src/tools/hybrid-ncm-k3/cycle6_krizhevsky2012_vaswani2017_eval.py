#!/usr/bin/env python3
"""Cycle 6 GPU Evaluation: Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton (2012) LRN Lateral
Inhibition (`b_i = a_i / (k + alpha * sum a_j^2)^beta`) + Ashish Vaswani et al. (2017) `1/sqrt(d_k)`
Logit Variance Stabilization & `h = 3` Multi-Head Subspace Unmixing on GPU (`cuda:7`).
"""
from __future__ import annotations

import json
import math
import sys
import time

sys.path.insert(0, "/home/usergpu/merge_lf")

import torch
import k3_memory_graph as kmg


def run_cycle6_eval() -> dict:
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(7) if torch.cuda.is_available() else "CPU"
    store = kmg.ensure_default_memory_tree()

    # 1. Vaswani et al. (2017, Section 3.2.1) `1/sqrt(d_k)` Logit Variance & Softmax Jacobian on GPU (`cuda:7`)
    torch.manual_seed(42)
    d_k = 512
    n_keys = 64
    q_mat = torch.randn((128, d_k), dtype=torch.float32, device=device)
    k_mat = torch.randn((n_keys, d_k), dtype=torch.float32, device=device)

    logits_unscaled = q_mat @ k_mat.t()
    logits_scaled = logits_unscaled / math.sqrt(float(d_k))

    var_unscaled = float(logits_unscaled.var().item())
    var_scaled = float(logits_scaled.var().item())

    p_unscaled = torch.softmax(logits_unscaled, dim=-1)
    p_scaled = torch.softmax(logits_scaled, dim=-1)

    # Frobenius norm of the softmax Jacobian per row: sqrt( sum_i p_i^2 (1 - p_i)^2 + sum_{i!=j} p_i^2 p_j^2 )
    def _mean_jacobian_frob(p: torch.Tensor) -> float:
        p64 = p.to(torch.float64)
        sum_p2 = (p64 * p64).sum(dim=-1)
        sum_p4 = (p64 ** 4).sum(dim=-1)
        diag_part = (p64 * (1.0 - p64)).pow(2).sum(dim=-1)
        off_part = sum_p2 * sum_p2 - sum_p4
        return float(torch.sqrt(torch.clamp(diag_part + off_part, min=0.0)).mean().item())

    jac_unscaled = _mean_jacobian_frob(p_unscaled)
    jac_scaled = _mean_jacobian_frob(p_scaled)

    # 2. Krizhevsky et al. (2012, Section 3.3) LRN Lateral Inhibition + Vaswani et al. (2017) `h=3` Multi-Head
    #    Subspace Retrieval (`multihead_lrn_subspace_retrieve`) across 6 Multi-Aspect Clover Queries (`top_k = 3`):
    multi_aspect_queries = [
        (
            "MA1_flat_sweep_3facets",
            "In a 48-stage pipeline with a shared global LRU pool, sweeping capacity from 512 to 4096 gives a flat 19.4% hit rate. State the general diagnostic concept, the original flat cache observation, and the superseding per-layer partition correction.",
        ),
        (
            "MA2_zero_denom_3facets",
            "When repeating a token 64 times inside one recurrent sequence produces 0 hits on an activation result cache, state the zero-denominator instrument principle, the empirical observation/experiment, and the request-replay correction.",
        ),
        (
            "MA3_ontorag_fanout_3facets",
            "When 1-hop AFFECTS leaf edges drown out 2-hop CWE->CAPEC paths in ontology GraphRAG, state the structural diagnostic principle, the measured fan-out flooding observation, and the stratified priority traversal correction.",
        ),
        (
            "MA4_cosine_threshold_3facets",
            "When a hard 0.35 cosine threshold drops 30% of code queries to zero seeds, state the instrument-validity concept, the 9/30 seed-dropout observation, and the hybrid Dense+BM25 RRF correction.",
        ),
        (
            "MA5_prefix_shadow_3facets",
            "In a character scanner where checking 1-char 'o' before 2-char 'o|' shadows the longer token, state the short-prefix shadowing concept, the generative compilation rollback experiment/observation, and the verified outcome/correction.",
        ),
        (
            "MA6_inside_outside_3facets",
            "In our philosophical inquiry on 'everything happens inside', state the observer-boundary concept, the recorded dialogue conversation/observation, and the active correction superseding the individual-mind reading.",
        ),
    ]

    single_head_all3_hits = 0
    multihead_lrn_all3_hits = 0
    single_head_facet_sum = 0
    multihead_lrn_facet_sum = 0
    query_details = []

    t0 = time.perf_counter()
    for q_id, q_text in multi_aspect_queries:
        res = store.multihead_lrn_subspace_retrieve(q_text, top_k=3, device=device)
        sh_f = res["facets_single_head"]
        mh_f = res["facets_multihead_lrn"]
        single_head_facet_sum += sh_f
        multihead_lrn_facet_sum += mh_f
        if sh_f == 3:
            single_head_all3_hits += 1
        if mh_f == 3:
            multihead_lrn_all3_hits += 1
        query_details.append({
            "id": q_id,
            "single_head_top3": res["single_head_top_k"],
            "single_head_facets": res["single_head_facets"],
            "facets_single_head": sh_f,
            "multihead_lrn_top3": res["multihead_lrn_top_k"],
            "multihead_lrn_facets": res["multihead_lrn_facets"],
            "facets_multihead_lrn": mh_f,
            "head_winners": res["head_winners"],
        })
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "device": device,
        "gpu_name": gpu_name,
        "vaswani_2017_sqrt_dk_scaling_gpu": {
            "d_k": d_k,
            "unscaled_logit_variance": round(var_unscaled, 3),
            "scaled_logit_variance": round(var_scaled, 3),
            "unscaled_softmax_jacobian_frob": round(jac_unscaled, 4),
            "scaled_softmax_jacobian_frob": round(jac_scaled, 4),
            "jacobian_gradient_preservation_x": round(jac_scaled / max(1e-9, jac_unscaled), 2),
        },
        "krizhevsky_2012_lrn_and_vaswani_multihead_subspace_gpu": {
            "single_head_all_3_facets_covered": f"{single_head_all3_hits}/{len(multi_aspect_queries)}",
            "multihead_lrn_all_3_facets_covered": f"{multihead_lrn_all3_hits}/{len(multi_aspect_queries)}",
            "single_head_mean_facets_of_3": round(single_head_facet_sum / len(multi_aspect_queries), 2),
            "multihead_lrn_mean_facets_of_3": round(multihead_lrn_facet_sum / len(multi_aspect_queries), 2),
            "ms_per_query_on_gpu": round(elapsed_ms / len(multi_aspect_queries), 3),
            "details": query_details,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_cycle6_eval(), indent=2))
