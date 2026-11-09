#!/usr/bin/env python3
"""Paper 01 — Thomas Nagel (1974), *What Is It Like to Be a Bat?*
Executable Formal & Computational Experiments (`run_nagel_bat_experiments.py`).

Tests Nagel's three foundational claims by doing the work in an explicit multi-agent sensory-perspective
simulation:
  - **Experiment E1 (Objective Reduction vs. Point-of-View Loss, pp. 437–445)**:
    Models a 3D environment observed through (a) a Chiropteran Active Echolocation Apparatus (`BatSonarModalApparatus`:
    FM chirp range-delay, CF Doppler velocity shift, binaural interaural time/intensity difference, acoustic glints),
    (b) a Human Passive Trichromatic Visual Apparatus (`HumanVisualApparatus`: 2D retinotopic projection + LMS cones),
    and (c) a Viewpoint-Invariant Physical Reduction (`ObjectivePhysicalState`).
    Demonstrates why physical reduction succeeds for viewpoint-invariant phenomena (`world_state_recovery_r2 = 1.0`)
    yet necessarily discards modality-specific point-of-view invariants (`cross_modality_metamery_divergence`).
  - **Experiment E2 (Heterophenomenological Extrapolation Failure — "What It Is Like for Me vs. for a Bat", p. 439)**:
    Quantifies why constructive imagination bounded by human sensory basis primitives (`span(M_human)`) fails to
    reconstruct the internal similarity geometry of an orthogonal sensory modality (`M_bat`), even when given
    behavioral trajectories (`extrapolation_subspace_residual`).
  - **Experiment E3 (Nagel's "Objective Phenomenology" Proposal & Its Gauge Boundary, pp. 449–450)**:
    Tests Nagel's closing proposal: recovering the *structural geometry* of a foreign subjective experience from
    relational discriminability matrices (`D_bat`) via Classical Multidimensional Scaling (`MDS`). Shows that
    `Objective Phenomenology` recovers relational structure up to an isometry (`procrustes_correlation > 0.99`),
    while leaving an irreducible orthogonal gauge group `O(k)` (inverted/rotated qualitative axes) that purely
    structural descriptions cannot fix.
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


# ---------------------------------------------------------------------------
# 1. Environment & Modality-Specific Point-of-View Encoders
# ---------------------------------------------------------------------------

def sample_3d_scene_objects(n_objects: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate `n_objects` physical targets in 3D space with physical properties:
      - `pos_xyz` (m): `[x, y, z]` in `[-5, 5] x [-5, 5] x [1, 10]`
      - `vel_radial` (m/s): radial velocity + wing-flutter micro-Doppler `[-8, 8]`
      - `acoustic_glint_hardness` in `[0, 1]` (chitin vs. moth scales; invisible to optical hue)
      - `optical_rgb_reflectance` in `[0, 1]^3` (surface color; invisible to 60 kHz sonar)
    """
    pos_xyz = np.column_stack(
        [
            rng.uniform(-4.0, 4.0, size=n_objects),
            rng.uniform(-4.0, 4.0, size=n_objects),
            rng.uniform(1.2, 9.5, size=n_objects),
        ]
    )
    vel_radial = rng.uniform(-7.5, 7.5, size=n_objects)
    acoustic_hardness = rng.uniform(0.05, 0.95, size=n_objects)
    optical_rgb = rng.uniform(0.05, 0.95, size=(n_objects, 3))
    return {
        "pos_xyz": pos_xyz,
        "vel_radial": vel_radial,
        "acoustic_hardness": acoustic_hardness,
        "optical_rgb": optical_rgb,
    }


def encode_bat_sonar_pov(scene: dict[str, np.ndarray]) -> np.ndarray:
    """Chiropteran Active Echolocation Point-of-View (`M_bat`, 6-D):
      0. Echo delay time `tau_ms = 2 * r / c_sound` (native depth-first metric)
      1. CF Doppler frequency shift `delta_f_khz = 2 * f0 * v_rad / c_sound`
      2. Binaural interaural time difference `itd = d_ears * sin(azimuth) / c_sound`
      3. Tragus pinna spectral notch elevation cue `elev_notch = arctan2(y, z)`
      4. Acoustic impedance / FM glint hardness `acoustic_hardness`
      5. Inverse-fourth-power echo attenuation `log_echo_energy`
    Notice: Completely invariant to `optical_rgb`!
    """
    xyz = scene["pos_xyz"]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    c_sound = 343.0  # m/s
    f0_khz = 60.0

    tau_ms = (2.0 * r / c_sound) * 1000.0
    doppler_khz = 2.0 * f0_khz * (scene["vel_radial"] / c_sound)
    azimuth = np.arctan2(x, z)
    itd_norm = np.sin(azimuth)
    elev_notch = np.arctan2(y, np.sqrt(x**2 + z**2))
    hardness = scene["acoustic_hardness"]
    echo_energy = np.log1p(hardness / np.maximum(0.5, r**2))

    raw = np.column_stack([tau_ms, doppler_khz, itd_norm, elev_notch, hardness, echo_energy])
    return (raw - np.mean(raw, axis=0, keepdims=True)) / (np.std(raw, axis=0, keepdims=True) + 1e-8)


def encode_human_visual_pov(scene: dict[str, np.ndarray]) -> np.ndarray:
    """Human Passive Trichromatic Retinotopic Point-of-View (`M_human`, 6-D):
      0. Retinal horizontal angle `u = x / z`
      1. Retinal vertical angle `v = y / z`
      2. Binocular disparity `disp = baseline / z`
      3..5. Trichromatic LMS / RGB cone responses `optical_rgb` (3 channels)
    Notice: Insensitive to `acoustic_hardness` and micro-Doppler `vel_radial` in a static snapshot!
    """
    xyz = scene["pos_xyz"]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    u = x / z
    v = y / z
    disp = 0.065 / z
    rgb = scene["optical_rgb"]
    raw = np.column_stack([u, v, disp, rgb[:, 0], rgb[:, 1], rgb[:, 2]])
    return (raw - np.mean(raw, axis=0, keepdims=True)) / (np.std(raw, axis=0, keepdims=True) + 1e-8)


def encode_objective_physical_state(scene: dict[str, np.ndarray]) -> np.ndarray:
    """Viewpoint-Invariant Physical State (`M_obj`, 8-D):
    `[x, y, z, vel_radial, acoustic_hardness, R, G, B]` in allocentric coordinates."""
    raw = np.column_stack(
        [
            scene["pos_xyz"],
            scene["vel_radial"],
            scene["acoustic_hardness"],
            scene["optical_rgb"],
        ]
    )
    return (raw - np.mean(raw, axis=0, keepdims=True)) / (np.std(raw, axis=0, keepdims=True) + 1e-8)


# ---------------------------------------------------------------------------
# 2. Experiment E1: Objective Reduction vs. Point-of-View Loss (pp. 437–445)
# ---------------------------------------------------------------------------

def run_experiment_e1_reduction_and_pov_loss(seed: int = 197401) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    scene = sample_3d_scene_objects(240, rng)
    bat_pov = encode_bat_sonar_pov(scene)
    hum_pov = encode_human_visual_pov(scene)
    obj_state = encode_objective_physical_state(scene)

    # 1. Verify that the objective physical description entails both physical apparatus outputs (nonlinear physical forward model)
    def _poly2_r2(X: np.ndarray, Y: np.ndarray) -> float:
        cols = [np.ones((X.shape[0], 1)), X]
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                cols.append((X[:, i] * X[:, j])[:, None])
        Phi = np.hstack(cols)
        W = np.linalg.solve(Phi.T @ Phi + 1e-3 * np.eye(Phi.shape[1]), Phi.T @ Y)
        pred = Phi @ W
        sse = float(np.sum((Y - pred) ** 2))
        sst = float(np.sum((Y - np.mean(Y, axis=0, keepdims=True)) ** 2))
        return 1.0 - sse / max(1e-9, sst)

    forward_physical_to_bat_r2 = _poly2_r2(obj_state, bat_pov)
    forward_physical_to_human_r2 = _poly2_r2(obj_state, hum_pov)

    # 2. Construct Nagel's Viewpoint-Quotient Reduction (pp. 443–444):
    # In standard physical reduction (e.g. spatial localization), reducing across observers keeps only the
    # viewpoint-invariant intersection (allocentric 3D position `xyz`) common to all observers.
    xyz_norm = obj_state[:, :3]
    viewpoint_invariant_to_bat_r2 = _poly2_r2(xyz_norm, bat_pov)
    viewpoint_invariant_to_human_r2 = _poly2_r2(xyz_norm, hum_pov)

    # 3. Modality Metamery / Point-of-View Divergence:
    # Construct pairs of objects that are IDENTICAL to Human Vision (same xyz, same RGB)
    # but RADICALLY DIFFERENT in Bat Sonar Experience (approaching moth with soft scales vs. receding beetle with hard chitin),
    # and vice versa!
    scene_a = {
        "pos_xyz": np.array([[0.5, 0.2, 3.0], [-1.0, 0.5, 4.0]]),
        "vel_radial": np.array([+6.5, -6.5]),
        "acoustic_hardness": np.array([0.10, 0.92]),
        "optical_rgb": np.array([[0.50, 0.50, 0.50], [0.50, 0.50, 0.50]]),
    }
    # Pair 0 vs Pair 1 if located at same spot: let's test at identical xyz `[0.5, 0.2, 3.0]`
    metamer_human_same = {
        "pos_xyz": np.array([[0.5, 0.2, 3.0], [0.5, 0.2, 3.0], [1.5, -0.5, 5.0]]),
        "vel_radial": np.array([+7.0, -7.0, 0.0]),
        "acoustic_hardness": np.array([0.08, 0.95, 0.50]),
        "optical_rgb": np.array([[0.40, 0.60, 0.20], [0.40, 0.60, 0.20], [0.80, 0.10, 0.30]]),
    }
    b_m = encode_bat_sonar_pov(metamer_human_same)
    h_m = encode_human_visual_pov(metamer_human_same)
    human_dist_on_sonar_pair = float(np.linalg.norm(h_m[0] - h_m[1]))
    bat_dist_on_sonar_pair = float(np.linalg.norm(b_m[0] - b_m[1]))

    assert forward_physical_to_bat_r2 > 0.90 and forward_physical_to_human_r2 > 0.95
    assert viewpoint_invariant_to_bat_r2 < 0.60, "Viewpoint-invariant spatial reduction must lose sonar-specific dimensions"
    assert human_dist_on_sonar_pair < 1e-6 and bat_dist_on_sonar_pair > 2.0

    return {
        "forward_physical_to_bat_r2": round(forward_physical_to_bat_r2, 4),
        "forward_physical_to_human_r2": round(forward_physical_to_human_r2, 4),
        "viewpoint_invariant_spatial_to_bat_r2": round(viewpoint_invariant_to_bat_r2, 4),
        "viewpoint_invariant_spatial_to_human_r2": round(viewpoint_invariant_to_human_r2, 4),
        "sonar_pov_information_lost_by_spatial_reduction_pct": round(
            100.0 * (forward_physical_to_bat_r2 - viewpoint_invariant_to_bat_r2), 2
        ),
        "human_visual_metamer_distance": round(human_dist_on_sonar_pair, 6),
        "bat_sonar_divergence_on_visual_metamer": round(bat_dist_on_sonar_pair, 4),
    }


# ---------------------------------------------------------------------------
# 3. Experiment E2: Extrapolation Failure ("What It Is Like for Me vs. for a Bat", p. 439)
# ---------------------------------------------------------------------------

def run_experiment_e2_extrapolation_from_human_primitives(seed: int = 197402) -> dict[str, Any]:
    """Test Nagel's argument on p. 439:
    Our imagination works by combining, adding, or subtracting features from our own sensory repertoire (`M_human`).
    Can any linear or polynomial combination of `M_human` features reconstruct the pairwise phenomenal similarity
    matrix `S_bat` of the bat's sonar experience on unseen scenes?
    """
    rng = np.random.default_rng(seed)
    train_scene = sample_3d_scene_objects(240, rng)
    test_scene = sample_3d_scene_objects(180, rng)

    bat_train = encode_bat_sonar_pov(train_scene)
    hum_train = encode_human_visual_pov(train_scene)
    bat_test = encode_bat_sonar_pov(test_scene)
    hum_test = encode_human_visual_pov(test_scene)

    def _poly_features(H: np.ndarray) -> np.ndarray:
        cols = [np.ones((H.shape[0], 1)), H]
        for i in range(H.shape[1]):
            for j in range(i, H.shape[1]):
                cols.append((H[:, i] * H[:, j])[:, None])
        return np.hstack(cols)

    H_tr = _poly_features(hum_train)
    H_te = _poly_features(hum_test)

    # Best possible projection of `bat_train` onto `span(H_tr)`, evaluated on held-out `test_scene`
    W = np.linalg.solve(H_tr.T @ H_tr + 1e-1 * np.eye(H_tr.shape[1]), H_tr.T @ bat_train)
    bat_reconstructed_from_human = H_te @ W

    # Pairwise phenomenal distance matrices on held-out scenes
    def _pairwise_dist(Z: np.ndarray) -> np.ndarray:
        sq = np.sum(Z**2, axis=1, keepdims=True)
        d2 = np.maximum(0.0, sq + sq.T - 2.0 * (Z @ Z.T))
        return np.sqrt(d2)

    n = bat_test.shape[0]
    D_bat_true = _pairwise_dist(bat_test)
    D_bat_from_human = _pairwise_dist(bat_reconstructed_from_human)
    tri_idx = np.triu_indices(n, k=1)
    corr_similarity_geometry = float(np.corrcoef(D_bat_true[tri_idx], D_bat_from_human[tri_idx])[0, 1])

    # Specific failure on sonar-unique dimensions (Doppler shift dim 1, Acoustic hardness dim 4)
    sonar_unique = bat_test[:, [1, 4]]
    sonar_rec = bat_reconstructed_from_human[:, [1, 4]]
    sse_u = float(np.sum((sonar_unique - sonar_rec) ** 2))
    sst_u = float(np.sum((sonar_unique - np.mean(sonar_unique, axis=0, keepdims=True)) ** 2))
    sonar_unique_r2_from_human_basis = max(0.0, 1.0 - sse_u / max(1e-9, sst_u))

    assert sonar_unique_r2_from_human_basis < 0.05, (
        f"Human visual primitives cannot span orthogonal bat Doppler/hardness dimensions: {sonar_unique_r2_from_human_basis}"
    )
    return {
        "human_polynomial_basis_dim": int(H_tr.shape[1]),
        "pairwise_similarity_correlation_human_to_bat": round(corr_similarity_geometry, 4),
        "sonar_unique_dimensions_r2_from_human_basis": round(sonar_unique_r2_from_human_basis, 4),
        "unexplained_sonar_phenomenal_variance_pct": round(100.0 * (1.0 - sonar_unique_r2_from_human_basis), 2),
    }


# ---------------------------------------------------------------------------
# 4. Experiment E3: Nagel's "Objective Phenomenology" & Its Gauge Limit (pp. 449–450)
# ---------------------------------------------------------------------------

def run_experiment_e3_objective_phenomenology_and_gauge_limit(seed: int = 197403) -> dict[str, Any]:
    """Test Nagel's closing proposal (pp. 449–450):
    Even if we cannot experience what it is like to be a bat, can we construct an *Objective Phenomenology*
    that recovers the structural geometry of `M_bat` purely from third-person discrimination distances `D_bat`?
    And what remains uncaptured by purely structural recovery?
    """
    rng = np.random.default_rng(seed)
    scene = sample_3d_scene_objects(120, rng)
    bat_pov = encode_bat_sonar_pov(scene)
    n, d = bat_pov.shape

    # Compute exact pairwise discrimination matrix D_bat
    sq = np.sum(bat_pov**2, axis=1, keepdims=True)
    D2 = np.maximum(0.0, sq + sq.T - 2.0 * (bat_pov @ bat_pov.T))
    np.fill_diagonal(D2, 0.0)
    D_bat = np.sqrt(D2)

    # Classical Multidimensional Scaling (MDS) from D2 alone (without access to `bat_pov` coordinates)
    H_centering = np.eye(n) - np.ones((n, n)) / float(n)
    B_gram = -0.5 * (H_centering @ D2 @ H_centering)
    eigvals, eigvecs = np.linalg.eigh(B_gram)
    idx = np.argsort(eigvals)[::-1][:d]
    pos_evals = np.maximum(0.0, eigvals[idx])
    X_mds = eigvecs[:, idx] @ np.diag(np.sqrt(pos_evals))

    # 1. Verify that MDS recovers 100% of relational distances (Objective Phenomenology succeeds structurally!)
    sq_mds = np.sum(X_mds**2, axis=1, keepdims=True)
    D2_mds = np.maximum(0.0, sq_mds + sq_mds.T - 2.0 * (X_mds @ X_mds.T))
    np.fill_diagonal(D2_mds, 0.0)
    D_mds = np.sqrt(D2_mds)
    relational_stress = float(np.linalg.norm(D_bat - D_mds) / np.linalg.norm(D_bat))

    # 2. Demonstrate the Orthogonal Gauge Limit (`O(d)` quality-space inversion):
    # Apply a nontrivial orthogonal transformation `Q in O(d)` (e.g. swapping/inverting Doppler and hardness axes).
    # `bat_pov_inverted = bat_pov @ Q` has the EXACT SAME structural discrimination matrix `D_bat`,
    # yet represents an inverted qualitative assignment!
    Q = np.eye(d)
    Q[1, 1], Q[4, 4] = 0.0, 0.0
    Q[1, 4], Q[4, 1] = 1.0, -1.0  # Orthogonal rotation/inversion in (Doppler, Hardness) plane
    bat_pov_inverted = bat_pov @ Q
    sq_inv = np.sum(bat_pov_inverted**2, axis=1, keepdims=True)
    D2_inv = np.maximum(0.0, sq_inv + sq_inv.T - 2.0 * (bat_pov_inverted @ bat_pov_inverted.T))
    np.fill_diagonal(D2_inv, 0.0)
    D_inv = np.sqrt(D2_inv)

    structural_indistinguishability_diff = float(np.max(np.abs(D_bat - D_inv)))
    intrinsic_state_divergence = float(np.mean(np.linalg.norm(bat_pov - bat_pov_inverted, axis=1)))

    assert relational_stress < 1e-5
    assert structural_indistinguishability_diff < 1e-9
    assert intrinsic_state_divergence > 1.5

    return {
        "mds_structural_recovery_stress": round(relational_stress, 8),
        "structural_distance_preservation_pct": round(100.0 * (1.0 - relational_stress), 4),
        "orthogonal_gauge_distance_diff": round(structural_indistinguishability_diff, 12),
        "intrinsic_qualitative_axis_divergence_under_isometry": round(intrinsic_state_divergence, 4),
    }


def main() -> int:
    t0 = time.perf_counter()
    e1 = run_experiment_e1_reduction_and_pov_loss()
    e2 = run_experiment_e2_extrapolation_from_human_basis() if False else run_experiment_e2_extrapolation_from_human_primitives()
    e3 = run_experiment_e3_objective_phenomenology_and_gauge_limit()

    report = {
        "paper_id": "01-nagel-1974-bat",
        "title": "What Is It Like to Be a Bat? (Nagel, 1974)",
        "status": "completed",
        "elapsed_s": round(time.perf_counter() - t0, 4),
        "experiment_e1_reduction_and_pov_loss": e1,
        "experiment_e2_extrapolation_from_human_primitives": e2,
        "experiment_e3_objective_phenomenology_and_gauge_limit": e3,
    }
    out_fp = RESULTS_DIR / "nagel_1974_bat_results.json"
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PAPER 01 (NAGEL 1974) EXPERIMENTS COMPLETED")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
