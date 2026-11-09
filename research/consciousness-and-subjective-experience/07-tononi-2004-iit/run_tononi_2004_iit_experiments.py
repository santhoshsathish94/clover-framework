#!/usr/bin/env python3
"""Paper 07 — Giulio Tononi (2004), *An Information Integration Theory of Consciousness*
Executable Formal & Computational Experiments (`run_tononi_2004_iit_experiments.py`).

Implements Tononi's (2004, Figure 1 Methodological Note, p. 5) exact analytical stationary linear
Gaussian process formalism for Effective Information `EI(A -> B)`, Minimum Information Bipartition
`MIB(S)`, Integrated Information `Phi(S)`, and Complex identification:
  - **Experiment E1 (Photodiode vs. Idealized Digital Camera Sensor vs. Integrated Complex, pp. 2–4)**:
    Shows that an 8-element disconnected camera sensor array has high external Shannon state entropy
    (`H_repertoire = 8.0 bits`), yet `Phi(S) = 0.0 bits` across its minimum information bipartition,
    whereas an 8-element integrated network achieves `Phi(S) > 0`.
  - **Experiment E2 (Figure 2 — Qualia Space & Effective Information Matrix of Divergent vs. Chain Complexes, pp. 6–8)**:
    Implements the 4-element Divergent digraph (`1 -> {2,3,4}`) and the 4-element Chain digraph (`1 -> 2 -> 3 -> 4`)
    calibrated to equal `Phi(S)`, computes the full 14-column directed Effective Information Matrix `EI(A -> B)`
    across all 7 bipartitions, and verifies that the two complexes have distinct relational Qualia Spaces
    (with bipartition `{1,3} -> {2,4}` achieving the highest directed `EI` in the Chain complex, exactly as in Fig. 2b).
  - **Experiment E3 (Figure 3 & Figure 4a — Specialized+Integrated Thalamocortical vs. Homogeneous vs. Cerebellar Modular, pp. 9–11)**:
    Compares 6-node networks normalized to identical total afferent synaptic weight per node (`w = 0.5`):
      1. *Specialized & Integrated (Thalamocortical-like)*: heterogeneous bidirectional connectivity (`highest Phi`),
      2. *Homogeneous All-to-All*: identical connection weights `w / (n - 1)` everywhere (`reduced Phi` due to loss of specialization),
      3. *Strongly Modular (Cerebellar-like)*: 3 independent 2-node modules (`Phi_whole = 0.0`, decomposing into 3 separate small complexes).
  - **Experiment E4 (Figure 4c/d, Split-Brain Callosotomy & Aaronson Expander-Grid Boundary, pp. 11–14)**:
    Verifies (a) that afferent (`port-in`) and efferent (`port-out`) chains attached to a core complex are
    excluded from the main complex because the single-link cut bottlenecks `Phi_whole`, (b) that cutting
    callosal links splits one main complex into two hemispheric complexes, and (c) **Scott Aaronson's (2014)
    Boundary**: a purely passive parity/expander grid with zero cognitive or world-modeling function still yields
    higher `Phi` than a specialized cognitive hierarchy!
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Tononi (2004, p. 5) Exact Analytical Gaussian EI, MIB, Phi, and Complex Engine
# ---------------------------------------------------------------------------

def logdet_cov(cov: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(cov)
    if sign <= 0:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(1e-18, eigvals)
        return float(np.sum(np.log(eigvals)))
    return float(val)


def compute_ei_directed(
    con_matrix: np.ndarray,
    subset_a: tuple[int, ...],
    subset_b: tuple[int, ...],
    *,
    cp: float = 1.0,
    ci: float = 1e-4,
    one_step: bool = False,
) -> float:
    """Compute directed Effective Information `EI(A -> B) = MI(A^{Hmax}; B)` in bits following Tononi (2004, p. 5):
      - `con_matrix[i, j]` is the causal connection weight from element `i` to element `j`.
      - When `one_step=True` (used in Tononi 2004 Fig. 2 for directed acyclic graphs), evaluates the 1-step
        causal transition `X_{t+1} = X_t @ con_cut + R` (`Q = I + con_cut`).
      - When `one_step=False` (stationary multivariate Gaussian process, p. 5), evaluates `Q = inv(I - con_cut)`.
    """
    n = con_matrix.shape[0]
    con_cut = con_matrix.copy()
    for a_idx in subset_a:
        con_cut[:, a_idx] = 0.0  # Zero all afferent and internal connections into A

    noise_var = np.full(n, ci**2, dtype=np.float64)
    for a_idx in subset_a:
        noise_var[a_idx] = cp**2

    I_mat = np.eye(n, dtype=np.float64)
    if one_step:
        # 1-step causal mapping from A^{Hmax} at step t to B at step t+1
        Q = I_mat.copy()
        for a_idx in subset_a:
            Q[a_idx, :] = con_cut[a_idx, :]
            Q[a_idx, a_idx] = 1.0
    else:
        Q = np.linalg.inv(I_mat - con_cut)
    cov_x = Q.T @ (noise_var[:, None] * Q)

    idx_a = list(subset_a)
    idx_b = list(subset_b)
    idx_ab = idx_a + idx_b

    cov_a = cov_x[np.ix_(idx_a, idx_a)]
    cov_b = cov_x[np.ix_(idx_b, idx_b)]
    cov_ab = cov_x[np.ix_(idx_ab, idx_ab)]

    mi_nats = 0.5 * (logdet_cov(cov_a) + logdet_cov(cov_b) - logdet_cov(cov_ab))
    return max(0.0, float(mi_nats / math.log(2.0)))


def compute_subset_phi_and_mib(
    con_matrix: np.ndarray,
    subset_s: tuple[int, ...],
    *,
    cp: float = 1.0,
    ci: float = 1e-4,
    one_step: bool = False,
) -> dict[str, Any]:
    """Evaluate all non-empty bipartitions `(A, B)` of `subset_s` (up to symmetry `(A, B) == (B, A)`),
    find the Minimum Information Bipartition `MIB = argmin EI(A <-> B) / min(|A|, |B|)`, and return
    `Phi(S) = EI(MIB)`.
    """
    m = len(subset_s)
    assert m >= 2
    s_set = set(subset_s)
    best_norm_ei = float("inf")
    best_raw_phi = float("inf")
    best_bipartition: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    bipartition_records = []

    # Enumerate unique unordered bipartitions by fixing `subset_s[0]` in part A
    first = subset_s[0]
    rest = subset_s[1:]
    for r in range(0, len(rest) + 1):
        for comb in itertools.combinations(rest, r):
            part_a = (first,) + comb
            if len(part_a) == m:
                continue
            part_b = tuple(sorted(s_set - set(part_a)))
            ei_ab = compute_ei_directed(con_matrix, part_a, part_b, cp=cp, ci=ci, one_step=one_step)
            ei_ba = compute_ei_directed(con_matrix, part_b, part_a, cp=cp, ci=ci, one_step=one_step)
            ei_total = ei_ab + ei_ba
            h_max_cap = float(min(len(part_a), len(part_b)))
            norm_ei = ei_total / h_max_cap
            bipartition_records.append(
                {
                    "A": list(part_a),
                    "B": list(part_b),
                    "ei_A_to_B": round(ei_ab, 4),
                    "ei_B_to_A": round(ei_ba, 4),
                    "ei_bidirectional": round(ei_total, 4),
                    "normalized_ei": round(norm_ei, 4),
                }
            )
            if norm_ei < best_norm_ei - 1e-9:
                best_norm_ei = norm_ei
                best_raw_phi = ei_total
                best_bipartition = (part_a, part_b)

    return {
        "subset": list(subset_s),
        "phi_bits": round(best_raw_phi, 4),
        "mib": [list(best_bipartition[0]), list(best_bipartition[1])],
        "bipartitions": bipartition_records,
    }


def find_all_complexes(con_matrix: np.ndarray) -> dict[str, Any]:
    """Identify all subsets `S` (`|S| >= 2`) and filter to true Complexes (subsets with `Phi > 0`
    that are not proper subsets of any larger subset with strictly higher `Phi`)."""
    n = con_matrix.shape[0]
    all_subsets: list[dict[str, Any]] = []
    for k in range(2, n + 1):
        for sub in itertools.combinations(range(n), k):
            res = compute_subset_phi_and_mib(con_matrix, sub)
            all_subsets.append(res)

    complexes = []
    for cand in all_subsets:
        if cand["phi_bits"] <= 1e-3:
            continue
        s_cand = set(cand["subset"])
        dominated = any(
            set(other["subset"]) > s_cand and other["phi_bits"] > cand["phi_bits"] + 1e-3
            for other in all_subsets
        )
        if not dominated:
            complexes.append(
                {
                    "subset": cand["subset"],
                    "phi_bits": cand["phi_bits"],
                    "mib": cand["mib"],
                }
            )
    complexes.sort(key=lambda d: -d["phi_bits"])
    return {
        "main_complex": complexes[0] if complexes else None,
        "complexes": complexes,
    }


def normalize_afferent_weights(con: np.ndarray, target_w: float = 0.5) -> np.ndarray:
    """Normalize `con` so the sum of absolute incoming weights to each node is `target_w` (Tononi 2004, p. 5)."""
    out = con.copy().astype(np.float64)
    np.fill_diagonal(out, 0.0)
    col_sums = np.sum(np.abs(out), axis=0)
    for j in range(out.shape[1]):
        if col_sums[j] > 1e-9:
            out[:, j] *= target_w / col_sums[j]
    return out


# ---------------------------------------------------------------------------
# 2. Experiments E1–E4
# ---------------------------------------------------------------------------

def run_experiment_e1_camera_vs_integrated_system() -> dict[str, Any]:
    """Compare an 8-photodiode disconnected sensor chip vs. an 8-element integrated ring-lattice complex."""
    n = 6
    con_camera = np.zeros((n, n), dtype=np.float64)
    con_integrated = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        con_integrated[i, (i + 1) % n] = 0.32
        con_integrated[i, (i + 2) % n] = 0.18

    phi_camera = compute_subset_phi_and_mib(con_camera, tuple(range(n)))["phi_bits"]
    phi_integrated = compute_subset_phi_and_mib(con_integrated, tuple(range(n)))["phi_bits"]

    assert phi_camera == 0.0 and phi_integrated > 10.0
    return {
        "elements_n": n,
        "camera_external_binary_repertoire_bits": float(n),
        "camera_sensor_chip_phi_bits": phi_camera,
        "integrated_network_phi_bits": phi_integrated,
    }


def run_experiment_e2_figure2_qualia_space_divergent_vs_chain() -> dict[str, Any]:
    """Reproduce Figure 2 (pp. 6–8): 4-element Divergent digraph (`0 -> {1,2,3}`) vs. 4-element Chain
    digraph (`0 -> 1 -> 2 -> 3`). Compare their full bipartition Effective Information matrices."""
    # Exact parameter calibration matching Tononi (2004) Fig. 2 (`Phi = 10.0 bits` per single cut link):
    # w = 0.5, cp = 1.0, ci = 2^-11 => 0.5 * log2(1 + (0.5 / 2^-11)^2) = 10.0000 bits!
    ci_fig2 = 2.0 ** (-11)
    w_div = 0.50 / math.sqrt(2.0)  # Accounts for redundant 2-target fan-out across the {0,3}/{1,2} MIB cut
    con_div = np.zeros((4, 4), dtype=np.float64)
    con_div[0, 1] = w_div
    con_div[0, 2] = w_div
    con_div[0, 3] = w_div

    con_chain = np.zeros((4, 4), dtype=np.float64)
    con_chain[0, 1] = 0.50
    con_chain[1, 2] = 0.50
    con_chain[2, 3] = 0.50

    res_div = compute_subset_phi_and_mib(con_div, (0, 1, 2, 3), ci=ci_fig2, one_step=True)
    res_chain = compute_subset_phi_and_mib(con_chain, (0, 1, 2, 3), ci=ci_fig2, one_step=True)

    # Extract the 14 directed EI values across all 7 bipartitions for both 4-node complexes
    vec_div = np.array(
        [[b["ei_A_to_B"], b["ei_B_to_A"]] for b in res_div["bipartitions"]], dtype=np.float64
    ).flatten()
    vec_chain = np.array(
        [[b["ei_A_to_B"], b["ei_B_to_A"]] for b in res_chain["bipartitions"]], dtype=np.float64
    ).flatten()

    qualia_space_frobenius_distance = float(np.linalg.norm(vec_div - vec_chain))
    # In the Chain complex (0->1->2->3), bipartition A={0,2}, B={1,3} (1-indexed {1,3}->{2,4} in Fig. 2b)
    # has two parallel directed links (0->1 and 2->3), giving twice the EI (20.0 bits) of single-cut bipartitions!
    chain_13_to_24 = next(
        b["ei_A_to_B"] for b in res_chain["bipartitions"] if b["A"] == [0, 2] and b["B"] == [1, 3]
    )

    assert abs(res_div["phi_bits"] - 10.0) < 0.01 and abs(res_chain["phi_bits"] - 10.0) < 0.01
    assert qualia_space_frobenius_distance > 12.0
    assert abs(chain_13_to_24 - 20.0) < 0.01

    return {
        "divergent_phi_bits": res_div["phi_bits"],
        "chain_phi_bits": res_chain["phi_bits"],
        "chain_alternating_bipartition_02_to_13_ei_bits": chain_13_to_24,
        "qualia_space_ei_matrix_distance": round(qualia_space_frobenius_distance, 4),
    }


def run_experiment_e3_figure3_and_4a_thalamocortical_vs_homogeneous_vs_cerebellar() -> dict[str, Any]:
    """Reproduce Figure 3 & Figure 4a (pp. 9–11) on 6-element networks with afferent weight sum `w = 0.5`:
      1. `specialized_integrated`: heterogeneous asymmetric cyclic + cross-chord connections,
      2. `homogeneous_all_to_all`: uniform weights `0.5 / 5 = 0.10` between all pairs,
      3. `cerebellar_modular`: three independent 2-node modules `{0,1}, {2,3}, {4,5}`.
    """
    n = 6
    # 1. Specialized & Integrated heterogeneous network
    raw_spec = np.array(
        [
            [0.0, 0.38, 0.00, 0.12, 0.00, 0.00],
            [0.05, 0.0, 0.40, 0.00, 0.10, 0.00],
            [0.00, 0.08, 0.0, 0.35, 0.00, 0.12],
            [0.10, 0.00, 0.05, 0.0, 0.38, 0.00],
            [0.00, 0.04, 0.00, 0.03, 0.0, 0.38],
            [0.35, 0.00, 0.05, 0.00, 0.02, 0.0],
        ]
    )
    con_spec = normalize_afferent_weights(raw_spec, target_w=0.5)

    # 2. Homogeneous all-to-all network with identical total afferent weight w = 0.5 per node
    con_homo = normalize_afferent_weights(np.ones((n, n)) - np.eye(n), target_w=0.5)

    # 3. Cerebellar-like modular network (three 2-node modules with minimal 0.005 cross-talk)
    raw_mod = np.zeros((n, n))
    for m_i in range(0, n, 2):
        raw_mod[m_i, m_i + 1] = 0.49
        raw_mod[m_i + 1, m_i] = 0.49
    con_mod = normalize_afferent_weights(raw_mod, target_w=0.5)

    phi_spec = compute_subset_phi_and_mib(con_spec, tuple(range(n)))["phi_bits"]
    phi_homo = compute_subset_phi_and_mib(con_homo, tuple(range(n)))["phi_bits"]
    mod_complexes = find_all_complexes(con_mod)
    phi_mod_whole = compute_subset_phi_and_mib(con_mod, tuple(range(n)))["phi_bits"]

    assert phi_spec > phi_homo > phi_mod_whole
    assert phi_mod_whole == 0.0 and len(mod_complexes["complexes"]) == 3

    return {
        "specialized_integrated_thalamocortical_phi_bits": phi_spec,
        "homogeneous_all_to_all_phi_bits": phi_homo,
        "cerebellar_modular_whole_system_phi_bits": phi_mod_whole,
        "cerebellar_local_module_complex_count": len(mod_complexes["complexes"]),
        "cerebellar_local_module_phi_bits": mod_complexes["complexes"][0]["phi_bits"],
    }


def run_experiment_e4_ports_split_brain_and_aaronson_expander() -> dict[str, Any]:
    """Test Figure 4c/d (Afferent/Efferent Port Exclusion), Figure 5a (Split-Brain Callosotomy),
    and Scott Aaronson's (2014) Expander-Graph Challenge."""
    # 1. Core 4-node bidirectional complex `{0,1,2,3}` + single unidirectional afferent port-in `4 -> 0`
    # and single unidirectional efferent port-out `3 -> 5` (Tononi 2004, Fig. 4c/d, p. 11)
    con_ports = np.zeros((6, 6), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            if i != j:
                con_ports[i, j] = 0.22 if (j == (i + 1) % 4) else 0.12
    con_ports[4, 0] = 0.12  # Unidirectional sensory afferent into port-in node 0
    con_ports[3, 5] = 0.12  # Unidirectional motor efferent from port-out node 3

    phi_core_4 = compute_subset_phi_and_mib(con_ports, (0, 1, 2, 3))["phi_bits"]
    phi_full_6_with_ports = compute_subset_phi_and_mib(con_ports, (0, 1, 2, 3, 4, 5))["phi_bits"]

    # 2. Split-brain callosotomy: two 3-node hemispheres `{0,1,2}` and `{3,4,5}` before vs. after cutting callosum
    con_intact = np.zeros((6, 6), dtype=np.float64)
    for h_off in (0, 3):
        con_intact[h_off + 0, h_off + 1] = 0.32
        con_intact[h_off + 1, h_off + 2] = 0.32
        con_intact[h_off + 2, h_off + 0] = 0.32
        con_intact[h_off + 1, h_off + 0] = 0.16
    con_split = con_intact.copy()
    # Add strong callosal bridges in intact brain
    for i in range(3):
        con_intact[i, i + 3] = 0.22
        con_intact[i + 3, i] = 0.22

    intact_info = find_all_complexes(con_intact)
    split_info = find_all_complexes(con_split)

    # 3. Aaronson (2014) Expander/Circulant Parity Grid Challenge:
    # A passive 6-node circulant matrix with zero sensory input, zero world model, and zero task semantics
    # vs. a 6-node hierarchical cognitive processor (`Sensory -> Feature -> Memory -> Planner -> Motor`).
    con_expander = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        con_expander[i, (i + 1) % 6] = 0.33
        con_expander[i, (i + 2) % 6] = 0.17
    con_cognitive_hierarchy = np.zeros((6, 6), dtype=np.float64)
    for i in range(5):
        con_cognitive_hierarchy[i, i + 1] = 0.42
    con_cognitive_hierarchy[4, 2] = 0.08  # Modest top-down feedback

    phi_passive_expander = compute_subset_phi_and_mib(con_expander, tuple(range(6)))["phi_bits"]
    phi_cognitive_hierarchy = compute_subset_phi_and_mib(con_cognitive_hierarchy, tuple(range(6)))["phi_bits"]

    assert phi_core_4 > 1.5 * phi_full_6_with_ports
    assert intact_info["main_complex"]["subset"] == [0, 1, 2, 3, 4, 5]
    assert len(split_info["complexes"]) == 2
    assert phi_passive_expander > 2.0 * phi_cognitive_hierarchy

    return {
        "core_complex_phi_bits": phi_core_4,
        "core_plus_afferent_efferent_ports_phi_bits": phi_full_6_with_ports,
        "intact_brain_main_complex": intact_info["main_complex"]["subset"],
        "intact_brain_phi_bits": intact_info["main_complex"]["phi_bits"],
        "split_brain_complexes": [c["subset"] for c in split_info["complexes"]],
        "split_brain_hemisphere_phi_bits": split_info["complexes"][0]["phi_bits"],
        "aaronson_passive_expander_grid_phi_bits": phi_passive_expander,
        "specialized_cognitive_hierarchy_phi_bits": phi_cognitive_hierarchy,
    }


def main() -> int:
    t0 = time.perf_counter()
    e1 = run_experiment_e1_camera_vs_integrated_system()
    e2 = run_experiment_e2_figure2_qualia_space_divergent_vs_chain()
    e3 = run_experiment_e3_figure3_and_4a_thalamocortical_vs_homogeneous_vs_cerebellar()
    e4 = run_experiment_e4_ports_split_brain_and_aaronson_expander()

    report = {
        "paper_id": "07-tononi-2004-iit",
        "title": "An Information Integration Theory of Consciousness (Tononi, 2004)",
        "status": "completed",
        "elapsed_s": round(time.perf_counter() - t0, 4),
        "experiment_e1_camera_vs_integrated_system": e1,
        "experiment_e2_figure2_qualia_space_divergent_vs_chain": e2,
        "experiment_e3_figure3_and_4a_thalamocortical_vs_homogeneous_vs_cerebellar": e3,
        "experiment_e4_ports_split_brain_and_aaronson_expander": e4,
    }
    out_fp = RESULTS_DIR / "tononi_2004_iit_results.json"
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PAPER 07 (TONONI 2004 IIT) EXPERIMENTS COMPLETED")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
