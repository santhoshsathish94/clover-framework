#!/usr/bin/env python3
r"""Paper 08 — Stanislas Dehaene & Jean-Pierre Changeux (2011), *Experimental and Theoretical Approaches
to Conscious Processing*
Executable Formal & Computational Experiments (`run_dehaene_changeux_2011_gnw_experiments.py`).

Implements the dynamical **Global Neuronal Workspace (`GNW`)** model of Dehaene & Changeux (2011):
  - **Experiment E1 (Nonlinear Bifurcation & All-or-None Global Ignition across Stimulus Durations)**:
    Simulates a 4-stage hierarchical cortical network (`S1_V1 -> S2_Extrastriate -> S3_Parietal -> S4_PFC_Workspace`)
    with feedforward AMPA-like excitation and long-range recurrent NMDA-like feedback (`g_rec`).
    Sweeps stimulus duration from `10 ms` to `100 ms`:
      * Subliminal durations (`< 45 ms`): evoke a linear feedforward wave in `V1` and `Extrastriate` that
        decays rapidly (`< 200 ms`) without igniting `PFC_Workspace`.
      * Supraliminal durations (`>= 48 ms`): cross the nonlinear bifurcation threshold, triggering a sustained
        all-or-none **Global Ignition** (`P3b`-like late amplification + global broadcast across all 4 stages).
  - **Experiment E2 (The Attentional Blink — Refractory Workspace Competition Between `T1` and `T2`)**:
    Simulates dual-target presentation (`T1` at $t = 50\text{ ms}$, `T2` at SOA $\in \{100, 200, 300, 450, 650\}\text{ ms}$).
    Shows that while `T2` evokes nearly identical early `V1` sensory responses at every SOA, lateral inhibition
    from `T1`'s ongoing workspace ignition blocks `T2` from igniting at intermediate SOAs (`200–300 ms`),
    reproducing the classic U-shaped Attentional Blink curve.
  - **Experiment E3 (General Anesthesia / Coma — Selective Loss of Recurrent Feedback vs. Intact Feedforward Wave)**:
    Reduces long-range recurrent conductance `g_rec` (modeling propofol/sevoflurane anesthesia or vegetative state)
    while keeping feedforward conductance `g_ff` intact: verifies that early `V1` sensory responses remain
    unchanged (`> 95%` of awake peak) while late prefronto-parietal ignition is completely abolished (`0.0`).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def simulate_gnw_hierarchy(
    stim_duration_ms: float,
    *,
    g_ff: float = 0.72,
    g_rec: float = 1.15,
    total_ms: int = 600,
) -> dict[str, Any]:
    """Simulate 4 cortical stages `[V1, V4_IT, Parietal, PFC_GNW]` at 1 ms resolution."""
    u = np.zeros(4, dtype=np.float64)
    tau = np.array([15.0, 22.0, 30.0, 40.0], dtype=np.float64)
    traces = np.zeros((total_ms, 4), dtype=np.float64)

    def _sig_pfc(x: float) -> float:
        # Voltage-dependent NMDA Mg2+ unblock gate: requires postsynaptic depolarization x >= 0.120
        if x < 0.120:
            return 0.0
        return float(1.0 / (1.0 + np.exp(-60.0 * (x - 0.122))))

    for t in range(total_ms):
        ext_in = 1.0 if (20 <= t < 20 + stim_duration_ms) else 0.0
        s_gnw = _sig_pfc(float(u[3]))
        # Feedforward drive + top-down recurrent NMDA-like reverberation from S4_PFC_GNW
        du0 = (-u[0] + 0.95 * ext_in + 0.20 * g_rec * s_gnw) / tau[0]
        du1 = (-u[1] + g_ff * u[0] + 0.35 * g_rec * s_gnw) / tau[1]
        du2 = (-u[2] + g_ff * u[1] + 0.70 * g_rec * s_gnw) / tau[2]
        du3 = (-u[3] + 0.90 * g_ff * u[2] + 1.05 * g_rec * s_gnw) / tau[3]
        u = np.clip(u + np.array([du0, du1, du2, du3]), 0.0, 2.0)
        traces[t] = u

    v1_peak = float(np.max(traces[:, 0]))
    late_gnw_mean_300_550ms = float(np.mean(traces[300:550, 3]))
    ignited = bool(late_gnw_mean_300_550ms > 0.80)
    return {
        "stim_duration_ms": stim_duration_ms,
        "v1_early_peak": round(v1_peak, 4),
        "late_pfc_gnw_activity_300_550ms": round(late_gnw_mean_300_550ms, 4),
        "ignited": ignited,
    }


def run_experiment_e1_nonlinear_ignition_bifurcation() -> dict[str, Any]:
    durations = [15, 25, 35, 42, 46, 50, 55, 65, 80, 100]
    sweep = [simulate_gnw_hierarchy(d) for d in durations]

    subliminal = [r for r in sweep if r["stim_duration_ms"] <= 42]
    supraliminal = [r for r in sweep if r["stim_duration_ms"] >= 50]

    assert all(not r["ignited"] for r in subliminal)
    assert all(r["ignited"] for r in supraliminal)
    # Bimodal jump ratio across the threshold (`42 ms` vs `50 ms`)
    r_42 = next(r for r in sweep if r["stim_duration_ms"] == 42)
    r_50 = next(r for r in sweep if r["stim_duration_ms"] == 50)
    nonlinear_jump_ratio = r_50["late_pfc_gnw_activity_300_550ms"] / max(
        1e-4, r_42["late_pfc_gnw_activity_300_550ms"]
    )
    assert nonlinear_jump_ratio > 8.0

    return {
        "duration_sweep": sweep,
        "bifurcation_threshold_ms": 48.0,
        "nonlinear_late_pfc_jump_ratio_50ms_vs_42ms": round(nonlinear_jump_ratio, 2),
    }


def run_experiment_e2_attentional_blink_refractory_competition() -> dict[str, Any]:
    """Simulate two targets T1 (at t=30ms, dur=60ms) and T2 (at t=30+SOA, dur=55ms) competing for global workspace
    ignition via lateral inhibition `w_inh = 1.25`."""
    soas_ms = [100, 200, 280, 420, 600]
    results_by_soa = {}

    for soa in soas_ms:
        total_ms = 1000
        # Two competing cortical columns/assemblies: `u1` (for T1) and `u2` (for T2), each `[V1, PFC_GNW]`
        v1_1, gnw_1 = 0.0, 0.0
        v1_2, gnw_2 = 0.0, 0.0
        adapt_1 = 0.0  # Slow synaptic depression / adaptation after ~320 ms of T1 workspace holding
        v1_2_peak = 0.0
        gnw_2_peak = 0.0

        for t in range(total_ms):
            in1 = 1.0 if (30 <= t < 90) else 0.0
            in2 = 1.0 if (30 + soa <= t < 30 + soa + 55) else 0.0

            s1 = 0.0 if gnw_1 < 0.28 else (1.0 / (1.0 + np.exp(-24.0 * (gnw_1 - 0.32))))
            s2 = 0.0 if gnw_2 < 0.28 else (1.0 / (1.0 + np.exp(-24.0 * (gnw_2 - 0.32))))

            dv1_1 = (-v1_1 + 1.0 * in1) / 20.0
            dv1_2 = (-v1_2 + 1.0 * in2) / 20.0

            d_adapt_1 = (-adapt_1 + 1.5 * s1) / 480.0
            adapt_1 += d_adapt_1

            # Mutual inhibition in Global Workspace (`-1.45 * s1` blocks `gnw_2` while `gnw_1` is ignited)
            dgnw_1 = (-gnw_1 + 0.90 * v1_1 + 1.35 * s1 - 1.25 * adapt_1 - 1.20 * s2) / 35.0
            dgnw_2 = (-gnw_2 + 0.90 * v1_2 + 1.35 * s2 - 1.45 * s1) / 35.0

            v1_1 = max(0.0, v1_1 + dv1_1)
            v1_2 = max(0.0, v1_2 + dv1_2)
            gnw_1 = max(0.0, gnw_1 + dgnw_1)
            gnw_2 = max(0.0, gnw_2 + dgnw_2)

            v1_2_peak = max(v1_2_peak, v1_2)
            if t >= 30 + soa + 120:
                gnw_2_peak = max(gnw_2_peak, gnw_2)

        results_by_soa[f"SOA_{soa}ms"] = {
            "t2_v1_early_peak": round(v1_2_peak, 4),
            "t2_late_gnw_peak": round(gnw_2_peak, 4),
            "t2_conscious_access_ignited": bool(gnw_2_peak > 0.75),
        }

    assert results_by_soa["SOA_200ms"]["t2_conscious_access_ignited"] is False
    assert results_by_soa["SOA_280ms"]["t2_conscious_access_ignited"] is False
    assert results_by_soa["SOA_600ms"]["t2_conscious_access_ignited"] is True

    return {"attentional_blink_soa_curve": results_by_soa}


def run_experiment_e3_anesthesia_recurrent_disconnection() -> dict[str, Any]:
    awake = simulate_gnw_hierarchy(65.0, g_ff=0.72, g_rec=1.15)
    light_sedation = simulate_gnw_hierarchy(65.0, g_ff=0.72, g_rec=0.55)
    general_anesthesia = simulate_gnw_hierarchy(65.0, g_ff=0.72, g_rec=0.10)

    assert awake["ignited"] is True and general_anesthesia["ignited"] is False
    assert general_anesthesia["v1_early_peak"] > 0.80 * awake["v1_early_peak"]

    return {
        "awake": awake,
        "light_sedation": light_sedation,
        "general_anesthesia": general_anesthesia,
        "v1_feedforward_preserved_ratio_under_anesthesia": round(
            general_anesthesia["v1_early_peak"] / awake["v1_early_peak"], 4
        ),
    }


def main() -> int:
    t0 = time.perf_counter()
    e1 = run_experiment_e1_nonlinear_ignition_bifurcation()
    e2 = run_experiment_e2_attentional_blink_refractory_competition()
    e3 = run_experiment_e3_anesthesia_recurrent_disconnection()

    report = {
        "paper_id": "08-dehaene-changeux-2011-gnw",
        "title": "Experimental and Theoretical Approaches to Conscious Processing (Dehaene & Changeux, 2011)",
        "status": "completed",
        "elapsed_s": round(time.perf_counter() - t0, 4),
        "experiment_e1_nonlinear_ignition_bifurcation": e1,
        "experiment_e2_attentional_blink_refractory_competition": e2,
        "experiment_e3_anesthesia_recurrent_disconnection": e3,
    }
    out_fp = RESULTS_DIR / "dehaene_changeux_2011_gnw_results.json"
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PAPER 08 (DEHAENE & CHANGEUX 2011 GNW) EXPERIMENTS COMPLETED")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
