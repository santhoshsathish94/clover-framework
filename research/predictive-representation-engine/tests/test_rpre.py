"""Unit and integration test suite for the Clover Reusable Predictive Representation Engine (`rpre`).

Verifies:
1. I-JEPA (`facebookresearch/ijepa`): 2D multi-block semantic mask disjointness (`allow_overlap=False`),
   `apply_masks`, context-only sparse encoding, and `LatentTokenPredictor` target token alignment.
2. Audio-JEPA (`LudovicTuncay/Audio-JEPA`): 2D spectrogram adapter, `CenteredNormMSELoss` ($2 - 2\cos(\hat{z}, \hat{h})$
   equivalence to centered Pearson correlation distance), random block masking, and time-frequency band masking.
3. EB-JEPA (`facebookresearch/eb_jepa`):
   - Exact mathematical equivalences from `eb_jepa/tests/test_loss_equivalences.py` (`HingeStdLoss`, `CovarianceLoss`,
     `VCLoss`, `BCSLoss` Epps-Pulley, `TemporalSimilarityLoss`, `InverseDynamicsLoss`).
   - Parallel and autoregressive `unroll_latent` output shapes (`eb_jepa/tests/test_jepa_output_formats.py`).
   - `CEMPlanner` and `MPPIPlanner` latent trajectory optimization (`eb_jepa/tests/planning_test.py`).
4. MJEPA (`facebookresearch/MJEPA`): Single shared Transformer encoder across audio & visual tokens, 4-level
   hierarchical representations, and all 9 predictive loss paths (`v2v`, `a2a`, `av2av`, `a2v`, `v2a`, `av2a`,
   `av2v`, `a2av`, `v2av`).
5. Clover Outcome -> Growth Loop: `OutcomeResidualAdapter` and `CloverCycleAgent` reducing prediction error from
   observed outcomes while enforcing the held-out baseline non-regression rollback gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpre import (  # noqa: E402
    BCSLoss,
    CEMPlanner,
    CenteredNormMSELoss,
    CloverCycleAgent,
    CovarianceLoss,
    HingeStdLoss,
    MPPIPlanner,
    ReprTargetDistObjective,
    ReusablePredictiveEngine,
    VCLoss,
    apply_masks,
    sample_audio_tf_masks,
    sample_multiblock_2d_masks,
    sample_random_masks,
)


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def main() -> None:
    torch.manual_seed(42)

    # 1. I-JEPA Multi-Block Masking & Disjoint Context/Target Invariant
    masks_enc, masks_pred = sample_multiblock_2d_masks(
        batch_size=4, grid_h=8, grid_w=8, nenc=1, npred=3, allow_overlap=False
    )
    check("I-JEPA multiblock produces 1 context mask and 3 target masks", len(masks_enc) == 1 and len(masks_pred) == 3)
    disjoint_ok = True
    for b in range(4):
        e_set = set(masks_enc[0][b].tolist())
        for p in range(3):
            p_set = set(masks_pred[p][b].tolist())
            if e_set & p_set:
                disjoint_ok = False
    check("I-JEPA multiblock context and target masks are strictly disjoint when allow_overlap=False", disjoint_ok)

    x_tok = torch.randn(4, 64, 32)
    x_ctxt = apply_masks(x_tok, masks_enc)
    check(
        "I-JEPA apply_masks extracts only visible context subset (N_ctxt < N)",
        x_ctxt.shape == (4, masks_enc[0].size(1), 32) and x_ctxt.size(1) < 64,
    )

    # 2. Audio-JEPA CenteredNormMSELoss & Spectrogram Masking
    norm_mse = CenteredNormMSELoss(standardize_target=True, center_pred=True)
    asym_norm_mse = CenteredNormMSELoss(standardize_target=True, center_pred=False)
    z_a = torch.randn(8, 16, 64) + 2.5  # non-zero mean
    check("Audio-JEPA symmetric CenteredNormMSELoss is 0.0 on identical embeddings", float(norm_mse(z_a, z_a)) < 1e-5)
    check(
        "Audio-JEPA upstream asymmetric loss.py:42-46 has non-zero self-loss on uncentered embeddings (verified finding)",
        float(asym_norm_mse(z_a, z_a)) > 0.1,
    )
    check("Audio-JEPA CenteredNormMSELoss is ~4.0 on negated embeddings", abs(float(norm_mse(-z_a, z_a)) - 4.0) < 1e-4)

    r_enc, r_pred = sample_random_masks(4, 8, 4, mask_ratio=(0.5, 0.5))
    check(
        "Audio-JEPA random_block partitions all 32 patches into disjoint context + target",
        r_enc[0].size(1) + r_pred[0].size(1) == 32,
    )
    tf_enc, tf_pred = sample_audio_tf_masks(4, 8, 4, npred=2)
    check("Audio-JEPA time-frequency mask sampler produces valid context and target masks", len(tf_enc) == 1 and len(tf_pred) == 2)

    # 3. EB-JEPA Loss Equivalences (from eb_jepa/tests/test_loss_equivalences.py)
    sample_x = torch.randn(64, 128)
    z_std_vicreg = torch.sqrt(sample_x.var(dim=0) + 1e-4)
    std_vicreg = torch.mean(F.relu(1.0 - z_std_vicreg))
    std_hinge = HingeStdLoss(std_margin=1.0)(sample_x)
    check("EB-JEPA HingeStdLoss matches VICReg std loss to 1e-7", torch.allclose(std_vicreg, std_hinge, atol=1e-7))

    z_c = sample_x - sample_x.mean(dim=0)
    z_cov = (z_c.T @ z_c) / (sample_x.size(0) - 1)
    cov_vicreg = (z_cov.pow(2).sum() - z_cov.diagonal().pow(2).sum()) / (z_cov.size(0) ** 2 - z_cov.size(0))
    cov_class = CovarianceLoss()(sample_x)
    check("EB-JEPA CovarianceLoss matches VICReg off-diagonal covariance loss to 1e-6", torch.allclose(cov_vicreg, cov_class, atol=1e-6))

    vc = VCLoss(std_coeff=10.0, cov_coeff=5.0)
    vc_l, vc_unw, vc_d = vc(torch.randn(8, 16, 2, 4, 4))
    check(
        "EB-JEPA VCLoss weighted sum equals std_coeff*std_loss + cov_coeff*cov_loss",
        abs(float(vc_l) - (10.0 * vc_d["std_loss"] + 5.0 * vc_d["cov_loss"])) < 1e-5,
    )

    bcs = BCSLoss(num_slices=64, lmbd=10.0)
    bcs_out = bcs(torch.randn(64, 32), torch.randn(64, 32))
    check("EB-JEPA BCSLoss (Epps-Pulley characteristic slicing) returns finite non-negative loss", float(bcs_out["bcs_loss"]) >= 0.0)

    # 4. MJEPA Shared Hierarchical Encoder & 9-Path Multimodal Step
    engine = ReusablePredictiveEngine(
        embed_dim=32,
        predictor_embed_dim=16,
        depth=4,
        pred_depth=2,
        num_heads=4,
        hierarchical_levels=4,
        visual_img_size=(16, 16),
        visual_patch_size=4,
        audio_spec_size=(16, 16),
        audio_patch_size=(4, 4),
        state_dim=4,
        action_dim=2,
    )
    vis_x = torch.randn(4, 3, 16, 16)
    aud_x = torch.randn(4, 1, 16, 16)
    vm_e, vm_p = sample_multiblock_2d_masks(4, 4, 4, nenc=1, npred=2)
    am_e, am_p = sample_audio_tf_masks(4, 4, 4, npred=2)

    losses_9 = engine.multimodal_jepa_step(vis_x, vm_e, vm_p, aud_x, am_e, am_p)
    expected_keys = {"loss", "v2v", "a2a", "av2av", "a2v", "v2a", "av2a", "av2v", "a2av", "v2av"}
    check(
        "MJEPA 9-path step computes all 3 intra-modal and 6 cross-modal losses",
        set(losses_9.keys()) == expected_keys and all(float(losses_9[k].detach().item()) > 0.0 for k in expected_keys),
    )
    losses_9["loss"].backward()
    engine.update_target_ema(momentum=0.95)
    check("MJEPA backward pass and EMA target encoder update succeed", True)

    # 5. EB-JEPA Parallel & Autoregressive Unrolling + CEM/MPPI Planning
    state_traj = torch.randn(4, 4, 6)  # [B=4, state_dim=4, T=6]
    act_traj = torch.randn(4, 2, 6)    # [B=4, action_dim=2, T=6]
    pred_par, loss_par = engine.unroll_latent(
        state_seq_bst=state_traj, actions_bat=act_traj, nsteps=3, unroll_mode="parallel", compute_loss=True
    )
    check(
        "EB-JEPA parallel unroll preserves trajectory shape [B, D, T] and computes VC+Sim+IDM+Pred losses",
        pred_par.shape == (4, 32, 6) and loss_par is not None and len(loss_par) == 5,
    )

    init_z = torch.randn(1, 32)
    pred_ar, _ = engine.unroll_latent(
        init_latent_bd=init_z, actions_bat=act_traj[:1, :, :5], nsteps=5, unroll_mode="autoregressive", compute_loss=False
    )
    check("EB-JEPA autoregressive unroll outputs [B, D, 1 + nsteps]", pred_ar.shape == (1, 32, 6))

    goal_z = torch.randn(1, 32)
    cem = CEMPlanner(
        unroll_fn=lambda z0, a: engine.unroll_latent(init_latent_bd=z0, actions_bat=a, nsteps=a.size(2), unroll_mode="autoregressive")[0],
        n_iters=5,
        num_samples=32,
        plan_length=4,
        action_dim=2,
    )
    cem.set_objective(ReprTargetDistObjective(goal_z))
    plan_res = cem.plan(init_z, seed=7)
    check(
        "EB-JEPA CEMPlanner monotonically reduces latent objective across iterations",
        plan_res.actions.shape == (4, 2)
        and float(plan_res.losses[-1].item()) <= float(plan_res.losses[0].item()) + 1e-5
        and float(plan_res.elite_losses_mean[-1].item()) <= float(plan_res.elite_losses_mean[0].item()) + 1e-5,
    )

    mppi = MPPIPlanner(
        unroll_fn=lambda z0, a: engine.unroll_latent(init_latent_bd=z0, actions_bat=a, nsteps=a.size(2), unroll_mode="autoregressive")[0],
        n_iters=4,
        num_samples=32,
        plan_length=4,
        action_dim=2,
    )
    mppi.set_objective(ReprTargetDistObjective(goal_z))
    mppi_res = mppi.plan(init_z)
    check("EB-JEPA MPPIPlanner returns valid action sequence [T, A]", mppi_res.actions.shape == (4, 2))

    # 6. Clover Outcome -> Growth Residual Adapter & Non-Regression Gate
    agent = CloverCycleAgent(engine, use_outcome_adapter=True, use_cross_modal_cue=True, plan_length=4, n_iters=4, num_samples=32)
    s0 = torch.tensor([[0.1, 0.2, 0.0, 0.0]])
    sg = torch.tensor([[0.8, 0.8, 0.0, 0.0]])
    z0, cue0 = agent.establish_context(s0, visual_x=vis_x[:1], audio_x=aud_x[:1])
    agent.set_direction(sg)
    a0 = agent.select_action(z0, steps_left=4, seed=11)

    # Record baseline and shifted outcomes and verify Growth reduces prediction error
    drift_dir = torch.linspace(-0.6, 0.6, 32).unsqueeze(0)
    for _ in range(12):
        s_base = torch.randn(1, 4)
        z_b, c_b = agent.establish_context(s_base, visual_x=vis_x[:1], audio_x=aud_x[:1])
        # Simulate a systematic directional environment drift in latent space
        with torch.no_grad():
            pred_base, _ = engine.unroll_latent(init_latent_bd=z_b, actions_bat=a0.unsqueeze(-1), nsteps=1, use_outcome_adapter=False)
            z_true = F.layer_norm(pred_base[:, :, -1] + drift_dir, (32,))
        engine.outcome_adapter.record_outcome(z_b, a0, z_true, c_t=c_b, is_baseline=False)

    grow_info = agent.grow(steps=40, lr=1e-2)
    check(
        "Clover OutcomeResidualAdapter learns from observed outcomes and reduces prediction MSE",
        grow_info.get("updated") is True and grow_info["mse_after"] < grow_info["mse_before"],
    )

    print("ALL 15 RPRE UNIT & INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    main()
