"""Executable Experiment Suite for the Four Core JEPA Papers & the Clover Reusable Predictive Representation Engine (`RPRE`).

Runs 5 controlled, multi-seed PyTorch experiments (`E1`–`E5`) and writes full falsifiable metrics to
`results/jepa_investigation_results.json`:
- `E1`: I-JEPA Visual Latent Prediction (`multiblock` vs `random` vs `pixel_mae` vs `no_ema_collapse`)
- `E2`: Audio-JEPA Time-Frequency Latent Prediction (`norm_mse + tf_band` vs `norm_mse + random` vs `mse` vs `spec_ae`)
- `E3`: EB-JEPA Anti-Collapse Regularizers & Action-Conditioned Latent MPC Planning (`VC + Sim + IDM` ablations)
- `E4`: MJEPA Shared Audio-Visual Representation Learning (`Unimodal` vs `Shared-Intra-Only` vs `MJEPA-9Path-1L` vs `MJEPA-9Path-4L`)
- `E5`: Clover Reusable Predictive Representation Engine (`RPRE`) Improving Through Observed Outcomes Across Cycles 1..5
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpre import (  # noqa: E402
    AudioSpectrogramAdapter,
    CEMPlanner,
    CenteredNormMSELoss,
    CloverCycleAgent,
    LatentTokenPredictor,
    ReprTargetDistObjective,
    ReusablePredictiveEngine,
    SharedHierarchicalEncoder,
    StateVectorAdapter,
    VCLoss,
    VC_IDM_Sim_Regularizer,
    VisualPatchAdapter,
    apply_masks,
    sample_audio_tf_masks,
    sample_multiblock_2d_masks,
    sample_random_masks,
)
from rpre.losses import InverseDynamicsMLP  # noqa: E402
from rpre.masks import repeat_interleave_batch  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_linear_probe(
    features_train: torch.Tensor,
    labels_train: torch.Tensor,
    features_test: torch.Tensor,
    labels_test: torch.Tensor,
    num_classes: int,
    epochs: int = 80,
    lr: float = 2e-2,
) -> float:
    clf = nn.Linear(features_train.size(-1), num_classes)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-3)
    f_tr = F.layer_norm(features_train.detach(), (features_train.size(-1),))
    f_te = F.layer_norm(features_test.detach(), (features_test.size(-1),))
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(clf(f_tr), labels_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = clf(f_te).argmax(dim=-1)
        return float((preds == labels_test).float().mean().item())


# ============================================================================
# E1: I-JEPA Visual Latent Prediction
# ============================================================================
def generate_visual_relational_dataset(
    n_samples: int,
    img_size: int = 32,
    noise_std: float = 0.35,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 32x32 RGB images across 4 relational object configurations embedded in high-frequency background noise.
    Class 0: Horizontal pair of bright blocks (separated by 10px)
    Class 1: Vertical pair of bright blocks (separated by 10px)
    Class 2: Diagonal pair (top-left to bottom-right)
    Class 3: Anti-diagonal L-triad
    """
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randn((n_samples, 3, img_size, img_size), generator=g) * noise_std
    # Add high-frequency checkerboard distraction in background so pixel reconstruction wastes capacity on texture
    yy, xx = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing="ij")
    checker = (((yy + xx) % 2).float() * 2.0 - 1.0) * 0.45
    labels = torch.randint(0, 4, (n_samples,), generator=g)

    for i in range(n_samples):
        phase = (int(torch.randint(0, 2, (1,), generator=g).item()) * 2 - 1)
        imgs[i] += phase * checker.unsqueeze(0)
        c = int(labels[i].item())
        jitter_r = int(torch.randint(0, 4, (1,), generator=g).item())
        jitter_c = int(torch.randint(0, 4, (1,), generator=g).item())
        col1 = torch.tensor([1.7, 0.8, -0.9]).view(3, 1, 1)
        col2 = torch.tensor([-0.8, 1.6, 0.9]).view(3, 1, 1)
        if c == 0:
            # Horizontal pair across top-left (Q1) and top-right (Q2)
            imgs[i, :, 3 + jitter_r : 9 + jitter_r, 3 + jitter_c : 9 + jitter_c] = col1
            imgs[i, :, 3 + jitter_r : 9 + jitter_r, 19 + jitter_c : 25 + jitter_c] = col2
        elif c == 1:
            # Vertical pair across top-left (Q1) and bottom-left (Q3)
            imgs[i, :, 3 + jitter_r : 9 + jitter_r, 3 + jitter_c : 9 + jitter_c] = col1
            imgs[i, :, 19 + jitter_r : 25 + jitter_r, 3 + jitter_c : 9 + jitter_c] = col2
        elif c == 2:
            # Main diagonal pair across top-left (Q1) and bottom-right (Q4)
            imgs[i, :, 3 + jitter_r : 9 + jitter_r, 3 + jitter_c : 9 + jitter_c] = col1
            imgs[i, :, 19 + jitter_r : 25 + jitter_r, 19 + jitter_c : 25 + jitter_c] = col2
        else:
            # Anti-diagonal pair across top-right (Q2) and bottom-left (Q3)
            imgs[i, :, 3 + jitter_r : 9 + jitter_r, 19 + jitter_c : 25 + jitter_c] = col1
            imgs[i, :, 19 + jitter_r : 25 + jitter_r, 3 + jitter_c : 9 + jitter_c] = col2
    return imgs, labels


def run_e1_ijepa_visual(seeds: list[int] = [101, 202, 303]) -> dict[str, Any]:
    arms = ("pixel_mae", "ijepa_no_ema_no_ln", "ijepa_random_mask", "ijepa_multiblock")
    per_arm: dict[str, list[dict[str, float]]] = {a: [] for a in arms}

    for seed in seeds:
        set_seed(seed)
        x_tr, y_tr = generate_visual_relational_dataset(240, seed=seed)
        x_te, y_te = generate_visual_relational_dataset(120, seed=seed + 1000)
        x_te_noisy, _ = generate_visual_relational_dataset(120, noise_std=0.75, seed=seed + 1000)

        for arm in arms:
            set_seed(seed)
            v_ad = VisualPatchAdapter(img_size=(32, 32), patch_size=4, in_chans=3, embed_dim=48)
            enc = SharedHierarchicalEncoder(embed_dim=48, depth=3, num_heads=4, hierarchical_levels=1, visual_adapter=v_ad)
            tgt_enc = copy.deepcopy(enc)
            for p in tgt_enc.parameters():
                p.requires_grad = False
            pred = LatentTokenPredictor(
                embed_dim=48, predictor_embed_dim=24, depth=2, num_heads=4, num_hier_layers=1, visual_grid=(8, 8)
            )
            pixel_head = nn.Linear(48, 3 * 4 * 4)

            params = list(enc.parameters()) + list(pred.parameters()) + (
                list(pixel_head.parameters()) if arm == "pixel_mae" else []
            )
            opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-4)

            bs = 40
            epochs = 28
            t0 = time.perf_counter()
            ctxt_tokens_used = 0
            full_tokens = 64

            for ep in range(epochs):
                perm = torch.randperm(x_tr.size(0))
                for i in range(0, x_tr.size(0), bs):
                    b_x = x_tr[perm[i : i + bs]]
                    if arm == "ijepa_multiblock":
                        m_e, m_p = sample_multiblock_2d_masks(
                            b_x.size(0), 8, 8, enc_mask_scale=(0.75, 0.90), pred_mask_scale=(0.12, 0.18), nenc=1, npred=2
                        )
                    else:
                        m_e, m_p = sample_random_masks(b_x.size(0), 8, 8, mask_ratio=(0.35, 0.45))
                    ctxt_tokens_used = int(m_e[0].size(1))

                    z_c = enc(visual_x=b_x, visual_masks=m_e)["visual"]
                    z_p = pred(visual_ctxt=z_c, visual_masks_x=m_e, visual_masks_y=m_p)["visual_pred"]

                    if arm == "pixel_mae":
                        # Reconstruct raw pixel patches at target locations
                        patches = b_x.unfold(2, 4, 4).unfold(3, 4, 4).permute(0, 2, 3, 1, 4, 5).reshape(b_x.size(0), 64, 48)
                        target_pix = apply_masks(patches, m_p)
                        pred_pix = pixel_head(z_p.reshape(b_x.size(0), -1, 48))
                        loss = F.mse_loss(pred_pix, target_pix)
                    elif arm == "ijepa_no_ema_no_ln":
                        # Online encoder without stop-grad or LayerNorm -> collapses
                        h_online = enc(visual_x=b_x)["visual"]
                        h_t = apply_masks(h_online, m_p)
                        loss = F.smooth_l1_loss(z_p, h_t)
                    else:
                        with torch.no_grad():
                            h_full = tgt_enc(visual_x=b_x)["visual"]
                            h_full = F.layer_norm(h_full, (h_full.size(-1),))
                            h_t = apply_masks(h_full, m_p)
                            h_t = repeat_interleave_batch(h_t, b_x.size(0), repeat=len(m_e))
                        z_p_norm = F.layer_norm(z_p, (z_p.size(-1),))
                        vc_reg = VCLoss(std_coeff=2.0, cov_coeff=1.0)(z_c)[0]
                        loss = F.smooth_l1_loss(z_p_norm, h_t) + 0.15 * vc_reg

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    if arm in ("ijepa_random_mask", "ijepa_multiblock"):
                        with torch.no_grad():
                            m = 0.96
                            for pq, pk in zip(enc.parameters(), tgt_enc.parameters()):
                                pk.data.mul_(m).add_((1.0 - m) * pq.detach().data)

            train_ms = (time.perf_counter() - t0) * 1000.0
            with torch.no_grad():
                # Concatenate 2x2 quadrant pooled features (preserving spatial layout for linear probe)
                def _quad_pool(vt: torch.Tensor) -> torch.Tensor:
                    g4 = vt.reshape(vt.size(0), 8, 8, -1)
                    q1 = g4[:, :4, :4, :].mean(dim=(1, 2))
                    q2 = g4[:, :4, 4:, :].mean(dim=(1, 2))
                    q3 = g4[:, 4:, :4, :].mean(dim=(1, 2))
                    q4 = g4[:, 4:, 4:, :].mean(dim=(1, 2))
                    return torch.cat([q1, q2, q3, q4], dim=-1)

                v_tr_tok = enc(visual_x=x_tr)["visual"]
                v_te_tok = enc(visual_x=x_te)["visual"]
                v_ns_tok = enc(visual_x=x_te_noisy)["visual"]
                f_tr = _quad_pool(v_tr_tok)
                f_te = _quad_pool(v_te_tok)
                f_noisy = _quad_pool(v_ns_tok)
                repr_std = float(v_te_tok.mean(dim=1).std(dim=0).mean().item())

            acc = train_linear_probe(f_tr, y_tr, f_te, y_te, num_classes=4)
            noisy_acc = train_linear_probe(f_tr, y_tr, f_noisy, y_te, num_classes=4)
            attn_flops_ratio = round((ctxt_tokens_used / float(full_tokens)) ** 2, 4)

            per_arm[arm].append({
                "clean_acc": round(acc, 4),
                "noisy_acc": round(noisy_acc, 4),
                "repr_std": round(repr_std, 4),
                "ctxt_tokens": ctxt_tokens_used,
                "full_tokens": full_tokens,
                "attn_flops_ratio": attn_flops_ratio,
                "train_ms": round(train_ms, 1),
            })

    summary = {}
    for arm, runs in per_arm.items():
        summary[arm] = {
            "clean_acc_mean": round(float(np.mean([r["clean_acc"] for r in runs])), 4),
            "noisy_acc_mean": round(float(np.mean([r["noisy_acc"] for r in runs])), 4),
            "repr_std_mean": round(float(np.mean([r["repr_std"] for r in runs])), 4),
            "attn_flops_ratio": runs[0]["attn_flops_ratio"],
            "ctxt_tokens": runs[0]["ctxt_tokens"],
            "runs": runs,
        }
    return summary


# ============================================================================
# E2: Audio-JEPA Time-Frequency Latent Prediction
# ============================================================================
def generate_audio_spectrogram_dataset(
    n_samples: int,
    time_steps: int = 32,
    freq_bins: int = 16,
    noise_std: float = 0.30,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 4-class time-frequency spectrograms [N, 1, T=32, F=16]:
    Class 0: Rising harmonic chirp (fundamental + 2nd harmonic)
    Class 1: Falling formant trajectory (fundamental + 2nd harmonic)
    Class 2: Sustained harmonic chord (3 fixed frequency bands with temporal modulation)
    Class 3: Rhythmic broadband transient pulses + high-frequency overtone
    """
    g = torch.Generator().manual_seed(seed)
    specs = torch.randn((n_samples, 1, time_steps, freq_bins), generator=g) * noise_std
    # High-frequency fine-grain spectral comb clutter (distracts bin-level reconstruction)
    tt, ff = torch.meshgrid(torch.arange(time_steps), torch.arange(freq_bins), indexing="ij")
    comb = (((tt + ff) % 2).float() * 2.0 - 1.0) * 0.55
    labels = torch.randint(0, 4, (n_samples,), generator=g)
    t_idx = torch.linspace(0, 1, time_steps)

    for i in range(n_samples):
        phase = (int(torch.randint(0, 2, (1,), generator=g).item()) * 2 - 1)
        specs[i, 0] += phase * comb
        c = int(labels[i].item())
        f_base = int(torch.randint(1, 3, (1,), generator=g).item())
        if c == 0:
            # Rising harmonic chirp (early low-freq -> late high-freq)
            specs[i, 0, 2:14, f_base : f_base + 4] += 1.8
            specs[i, 0, 18:30, f_base + 8 : f_base + 12] += 1.8
        elif c == 1:
            # Falling formant trajectory (early high-freq -> late low-freq)
            specs[i, 0, 2:14, f_base + 8 : f_base + 12] += 1.8
            specs[i, 0, 18:30, f_base : f_base + 4] += 1.8
        elif c == 2:
            # Sustained low+high simultaneous harmonic chord
            specs[i, 0, 2:14, f_base : f_base + 4] += 1.7
            specs[i, 0, 2:14, f_base + 8 : f_base + 12] += 1.7
        else:
            # Late-onset broadband harmonic burst
            specs[i, 0, 18:30, f_base : f_base + 4] += 1.7
            specs[i, 0, 18:30, f_base + 8 : f_base + 12] += 1.7
    return specs, labels


def run_e2_audio_jepa(seeds: list[int] = [101, 202, 303]) -> dict[str, Any]:
    arms = ("spec_autoencoder", "audio_jepa_mse_random", "audio_jepa_norm_mse_random", "audio_jepa_norm_mse_tf_band")
    per_arm: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    norm_mse_fn = CenteredNormMSELoss(standardize_target=True)

    for seed in seeds:
        set_seed(seed)
        x_tr, y_tr = generate_audio_spectrogram_dataset(240, seed=seed)
        x_te, y_te = generate_audio_spectrogram_dataset(120, seed=seed + 2000)

        for arm in arms:
            set_seed(seed)
            a_ad = AudioSpectrogramAdapter(spec_size=(32, 16), patch_size=(4, 4), in_chans=1, embed_dim=48)
            enc = SharedHierarchicalEncoder(embed_dim=48, depth=3, num_heads=4, hierarchical_levels=1, audio_adapter=a_ad)
            tgt_enc = copy.deepcopy(enc)
            for p in tgt_enc.parameters():
                p.requires_grad = False
            pred = LatentTokenPredictor(
                embed_dim=48, predictor_embed_dim=24, depth=2, num_heads=4, num_hier_layers=1, audio_grid=(8, 4)
            )
            spec_head = nn.Linear(48, 16)

            params = list(enc.parameters()) + list(pred.parameters()) + (
                list(spec_head.parameters()) if arm == "spec_autoencoder" else []
            )
            opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-4)

            bs = 40
            for _ in range(28):
                perm = torch.randperm(x_tr.size(0))
                for i in range(0, x_tr.size(0), bs):
                    b_x = x_tr[perm[i : i + bs]]
                    if arm == "audio_jepa_norm_mse_tf_band":
                        m_e, m_p = sample_audio_tf_masks(b_x.size(0), 8, 4, npred=2)
                    else:
                        m_e, m_p = sample_random_masks(b_x.size(0), 8, 4, mask_ratio=(0.45, 0.55))

                    z_c = enc(audio_x=b_x, audio_masks=m_e)["audio"]
                    z_p = pred(audio_ctxt=z_c, audio_masks_x=m_e, audio_masks_y=m_p)["audio_pred"]

                    if arm == "spec_autoencoder":
                        patches = b_x.unfold(2, 4, 4).unfold(3, 4, 4).permute(0, 2, 3, 1, 4, 5).reshape(b_x.size(0), 32, 16)
                        target_p = apply_masks(patches, m_p)
                        loss = F.mse_loss(spec_head(z_p), target_p)
                    else:
                        with torch.no_grad():
                            h_full = tgt_enc(audio_x=b_x)["audio"]
                            h_t = apply_masks(h_full, m_p)
                            h_t = repeat_interleave_batch(h_t, b_x.size(0), repeat=len(m_e))
                        vc_a = VCLoss(std_coeff=2.0, cov_coeff=1.0)(z_c)[0]
                        if arm == "audio_jepa_mse_random":
                            loss = F.mse_loss(z_p, F.layer_norm(h_t, (h_t.size(-1),))) + 0.15 * vc_a
                        else:
                            loss = norm_mse_fn(z_p, h_t) + 0.15 * vc_a

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    with torch.no_grad():
                        for pq, pk in zip(enc.parameters(), tgt_enc.parameters()):
                            pk.data.mul_(0.96).add_(0.04 * pq.detach().data)

            with torch.no_grad():
                def _tf_pool(at: torch.Tensor) -> torch.Tensor:
                    g_tf = at.reshape(at.size(0), 8, 4, -1)
                    q1 = g_tf[:, :4, :2, :].mean(dim=(1, 2))
                    q2 = g_tf[:, :4, 2:, :].mean(dim=(1, 2))
                    q3 = g_tf[:, 4:, :2, :].mean(dim=(1, 2))
                    q4 = g_tf[:, 4:, 2:, :].mean(dim=(1, 2))
                    return torch.cat([q1, q2, q3, q4], dim=-1)

                f_tr = _tf_pool(enc(audio_x=x_tr)["audio"])
                f_te = _tf_pool(enc(audio_x=x_te)["audio"])
                m_e_te, m_p_te = sample_audio_tf_masks(x_te.size(0), 8, 4, npred=1)
                zc_te = enc(audio_x=x_te, audio_masks=m_e_te)["audio"]
                zp_te = pred(audio_ctxt=zc_te, audio_masks_x=m_e_te, audio_masks_y=m_p_te)["audio_pred"]
                ht_te = apply_masks(tgt_enc(audio_x=x_te)["audio"], m_p_te)
                cos_sim = float(F.cosine_similarity(zp_te, ht_te, dim=-1).mean().item())

            acc = train_linear_probe(f_tr, y_tr, f_te, y_te, num_classes=4)
            per_arm[arm].append({"probe_acc": round(acc, 4), "latent_cos": round(cos_sim, 4)})

    return {
        arm: {
            "probe_acc_mean": round(float(np.mean([r["probe_acc"] for r in runs])), 4),
            "latent_cos_mean": round(float(np.mean([r["latent_cos"] for r in runs])), 4),
            "runs": runs,
        }
        for arm, runs in per_arm.items()
    }


# ============================================================================
# E3: EB-JEPA Generalized Learning, Anti-Collapse Regularizers & MPC Planning
# ============================================================================
def step_2d_dynamics(state: torch.Tensor, action: torch.Tensor, drift: torch.Tensor | None = None) -> torch.Tensor:
    """Continuous 2D state `state = [x, y, vx, vy]` with velocity damping and optional regime drift."""
    pos = state[:, :2]
    vel = state[:, 2:4]
    a_clipped = torch.clamp(action, -0.6, 0.6)
    d_vec = drift if drift is not None else torch.zeros_like(pos)
    next_vel = 0.55 * vel + 0.45 * a_clipped + d_vec
    next_pos = torch.clamp(pos + next_vel * 0.35, -1.5, 1.5)
    return torch.cat([next_pos, next_vel], dim=-1)


def generate_trajectory_dataset(
    n_trajs: int = 200,
    horizon: int = 8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    s0 = (torch.rand((n_trajs, 4), generator=g) - 0.5) * 1.6
    s0[:, 2:] *= 0.2
    actions = (torch.rand((n_trajs, 2, horizon), generator=g) - 0.5) * 1.0
    states = [s0]
    cur = s0
    for t in range(horizon - 1):
        cur = step_2d_dynamics(cur, actions[:, :, t])
        states.append(cur)
    states_bst = torch.stack(states, dim=-1)  # [B, 4, T]
    return states_bst, actions


def run_e3_eb_jepa_planning(seeds: list[int] = [101, 202, 303]) -> dict[str, Any]:
    arms = (
        "pred_only_no_reg",
        "vc_only",
        "vc_plus_temporal_sim",
        "full_eb_jepa_vc_sim_idm",
    )
    per_arm: dict[str, list[dict[str, float]]] = {a: [] for a in arms}

    for seed in seeds:
        s_tr, a_tr = generate_trajectory_dataset(220, horizon=8, seed=seed)
        s_te, a_te = generate_trajectory_dataset(60, horizon=8, seed=seed + 3000)

        for arm in arms:
            set_seed(seed)
            eng = ReusablePredictiveEngine(embed_dim=32, predictor_embed_dim=16, depth=2, state_dim=4, action_dim=2)
            if arm == "pred_only_no_reg":
                eng.regularizer = VC_IDM_Sim_Regularizer(cov_coeff=0.0, std_coeff=0.0, sim_coeff_t=0.0, idm_coeff=0.0)
            elif arm == "vc_only":
                eng.regularizer = VC_IDM_Sim_Regularizer(cov_coeff=8.0, std_coeff=16.0, sim_coeff_t=0.0, idm_coeff=0.0)
            elif arm == "vc_plus_temporal_sim":
                eng.regularizer = VC_IDM_Sim_Regularizer(cov_coeff=8.0, std_coeff=16.0, sim_coeff_t=4.0, idm_coeff=0.0)
            else:
                eng.regularizer = VC_IDM_Sim_Regularizer(
                    cov_coeff=8.0, std_coeff=16.0, sim_coeff_t=2.0, idm_coeff=6.0, idm=eng.idm
                )

            opt = torch.optim.AdamW(
                list(eng.encoder.parameters())
                + list(eng.transition_predictor.parameters())
                + list(eng.idm.parameters()),
                lr=3e-3,
                weight_decay=1e-4,
            )

            bs = 44
            for _ in range(32):
                perm = torch.randperm(s_tr.size(0))
                for i in range(0, s_tr.size(0), bs):
                    bs_s = s_tr[perm[i : i + bs]]
                    bs_a = a_tr[perm[i : i + bs]]
                    _, losses = eng.unroll_latent(
                        state_seq_bst=bs_s,
                        actions_bat=bs_a,
                        nsteps=7,
                        unroll_mode="autoregressive",
                        compute_loss=True,
                        use_outcome_adapter=False,
                    )
                    opt.zero_grad()
                    losses[0].backward()
                    opt.step()

            # Evaluate representation std, rank, rollout accuracy, and CEM planning success
            with torch.no_grad():
                flat_te = s_te.transpose(1, 2).reshape(-1, 4)
                z_te = eng.encoder(state_x=flat_te)["state"].squeeze(1)
                repr_std = float(z_te.std(dim=0).mean().item())
                svals = torch.linalg.svdvals(z_te - z_te.mean(dim=0, keepdim=True))
                p_sv = (svals / (svals.sum() + 1e-8)).clamp(min=1e-12)
                eff_rank = float(torch.exp(-(p_sv * torch.log(p_sv)).sum()).item())

                # Train linear state decoder probe on frozen encoder to measure physical rollout error
                flat_tr = s_tr.transpose(1, 2).reshape(-1, 4)
                z_tr = eng.encoder(state_x=flat_tr)["state"].squeeze(1)
            prober = nn.Linear(32, 4)
            p_opt = torch.optim.AdamW(prober.parameters(), lr=1e-2)
            for _ in range(60):
                p_opt.zero_grad()
                F.mse_loss(prober(z_tr.detach()), flat_tr).backward()
                p_opt.step()

            with torch.no_grad():
                z0_te = eng.encoder(state_x=s_te[:, :, 0])["state"].squeeze(1)
                pred_bdt, _ = eng.unroll_latent(
                    init_latent_bd=z0_te, actions_bat=a_te[:, :, :7], nsteps=7, unroll_mode="autoregressive", use_outcome_adapter=False
                )
                pred_pos = prober(pred_bdt.transpose(1, 2))[:, :, :2]
                gt_pos = s_te.transpose(1, 2)[:, :, :2]
                pos_rollout_mse = float(F.mse_loss(pred_pos, gt_pos).item())

            # Closed-loop MPC planning across 20 start->goal episodes
            agent = CloverCycleAgent(eng, use_outcome_adapter=False, use_cross_modal_cue=False, plan_length=6, n_iters=6, num_samples=96)
            g_ep = torch.Generator().manual_seed(seed + 500)
            successes = 0
            final_dists = []
            for ep in range(20):
                cur_s = (torch.rand((1, 4), generator=g_ep) - 0.5) * 1.4
                cur_s[:, 2:] = 0.0
                goal_s = (torch.rand((1, 4), generator=g_ep) - 0.5) * 1.4
                goal_s[:, 2:] = 0.0
                agent.set_direction(goal_s)
                for step_i in range(10):
                    z_cur, _ = agent.establish_context(cur_s)
                    act = agent.select_action(z_cur, steps_left=10 - step_i, seed=seed + ep * 20 + step_i)
                    cur_s = step_2d_dynamics(cur_s, act)
                dist = float((cur_s[:, :2] - goal_s[:, :2]).norm(dim=-1).item())
                final_dists.append(dist)
                if dist < 0.25:
                    successes += 1

            per_arm[arm].append({
                "repr_std": round(repr_std, 4),
                "effective_rank": round(eff_rank, 2),
                "pos_rollout_mse": round(pos_rollout_mse, 4),
                "planning_success_rate": round(successes / 20.0, 4),
                "final_goal_dist": round(float(np.mean(final_dists)), 4),
            })

    return {
        arm: {
            "repr_std_mean": round(float(np.mean([r["repr_std"] for r in runs])), 4),
            "effective_rank_mean": round(float(np.mean([r["effective_rank"] for r in runs])), 2),
            "pos_rollout_mse_mean": round(float(np.mean([r["pos_rollout_mse"] for r in runs])), 4),
            "planning_success_rate_mean": round(float(np.mean([r["planning_success_rate"] for r in runs])), 4),
            "final_goal_dist_mean": round(float(np.mean([r["final_goal_dist"] for r in runs])), 4),
            "runs": runs,
        }
        for arm, runs in per_arm.items()
    }


# ============================================================================
# E4: MJEPA Shared Audio-Visual Representation Learning
# ============================================================================
def generate_paired_audiovisual_dataset(
    n_samples: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate paired (Visual [N, 3, 16, 16], Audio [N, 1, 16, 16], Labels [N]) across 6 multimodal event classes.
    Each event class has both a visual spatial signature and a paired acoustic time-frequency signature, plus
    30% partial single-modality corruption so cross-modal prediction transfers structure between modalities.
    """
    g = torch.Generator().manual_seed(seed)
    vis = torch.randn((n_samples, 3, 16, 16), generator=g) * 0.45
    aud = torch.randn((n_samples, 1, 16, 16), generator=g) * 0.45
    labels = torch.randint(0, 6, (n_samples,), generator=g)

    for i in range(n_samples):
        c = int(labels[i].item())
        # Shared instance latent factor (e.g. intensity / spatial-spectral offset)
        inst_shift = int(torch.randint(0, 3, (1,), generator=g).item())
        r_v = (c * 2 + inst_shift) % 11
        c_v = ((c * 3) + inst_shift) % 11
        col = torch.tensor([(c % 3) - 1.0, ((c + 1) % 3) - 1.0, ((c + 2) % 3) - 1.0]).view(3, 1, 1) * 1.45

        # Visual pattern
        vis[i, :, r_v : r_v + 5, c_v : c_v + 5] += col
        if c % 2 == 0:
            vis[i, :, (r_v + 6) % 12 : (r_v + 6) % 12 + 4, c_v : c_v + 4] += col * 0.8

        # Paired Audio time-frequency harmonic pattern
        f0 = (c * 2 + inst_shift) % 10
        f1 = min(15, f0 + 4)
        t_start = (c * 2) % 8
        aud[i, 0, t_start : t_start + 8, f0 : f0 + 2] += 1.65
        aud[i, 0, t_start : t_start + 8, f1 : f1 + 1] += 1.25

        # Partial single-modality occlusion (15% visual-degraded, 15% audio-degraded)
        u = float(torch.rand(1, generator=g).item())
        if u < 0.15:
            vis[i] *= 0.25
        elif u < 0.30:
            aud[i] *= 0.25

    return vis, aud, labels


def run_e4_mjepa_multimodal(seeds: list[int] = [101, 202, 303]) -> dict[str, Any]:
    arms = (
        "unimodal_separate",
        "shared_intra_only_no_cross",
        "shared_mjepa_9path_1level",
        "shared_mjepa_9path_4level_hier",
    )
    per_arm: dict[str, list[dict[str, float]]] = {a: [] for a in arms}

    for seed in seeds:
        v_tr, a_tr, y_tr = generate_paired_audiovisual_dataset(240, seed=seed)
        v_te, a_te, y_te = generate_paired_audiovisual_dataset(120, seed=seed + 4000)

        for arm in arms:
            set_seed(seed)
            bs = 40
            epochs = 26

            if arm == "unimodal_separate":
                eng_v = ReusablePredictiveEngine(
                    embed_dim=48, predictor_embed_dim=24, depth=4, pred_depth=2, hierarchical_levels=1,
                    visual_img_size=(16, 16), visual_patch_size=4, audio_spec_size=(16, 16), audio_patch_size=(4, 4)
                )
                eng_a = ReusablePredictiveEngine(
                    embed_dim=48, predictor_embed_dim=24, depth=4, pred_depth=2, hierarchical_levels=1,
                    visual_img_size=(16, 16), visual_patch_size=4, audio_spec_size=(16, 16), audio_patch_size=(4, 4)
                )
                opt_v = torch.optim.AdamW(list(eng_v.encoder.parameters()) + list(eng_v.token_predictor.parameters()), lr=3e-3)
                opt_a = torch.optim.AdamW(list(eng_a.encoder.parameters()) + list(eng_a.token_predictor.parameters()), lr=3e-3)

                for _ in range(epochs):
                    perm = torch.randperm(v_tr.size(0))
                    for i in range(0, v_tr.size(0), bs):
                        bv = v_tr[perm[i : i + bs]]
                        ba = a_tr[perm[i : i + bs]]
                        vm_e, vm_p = sample_multiblock_2d_masks(bv.size(0), 4, 4, nenc=1, npred=2)
                        am_e, am_p = sample_audio_tf_masks(ba.size(0), 4, 4, npred=2)

                        lv = eng_v.multimodal_jepa_step(bv, vm_e, vm_p, ba, am_e, am_p, enable_cross_modal=False, enable_av_joint=False, use_hierarchical=False)["v2v"]
                        opt_v.zero_grad(); lv.backward(); opt_v.step(); eng_v.update_target_ema(0.96)

                        la = eng_a.multimodal_jepa_step(bv, vm_e, vm_p, ba, am_e, am_p, enable_cross_modal=False, enable_av_joint=False, use_hierarchical=False)["a2a"]
                        opt_a.zero_grad(); la.backward(); opt_a.step(); eng_a.update_target_ema(0.96)

                with torch.no_grad():
                    fv_tr = eng_v.encoder(visual_x=v_tr)["visual"].mean(dim=1)
                    fv_te = eng_v.encoder(visual_x=v_te)["visual"].mean(dim=1)
                    fa_tr = eng_a.encoder(audio_x=a_tr)["audio"].mean(dim=1)
                    fa_te = eng_a.encoder(audio_x=a_te)["audio"].mean(dim=1)
                    fav_tr = torch.cat([fv_tr, fa_tr], dim=-1)
                    fav_te = torch.cat([fv_te, fa_te], dim=-1)
                    a2v_r1 = float(( (F.normalize(fa_te, dim=-1) @ F.normalize(fv_te, dim=-1).T).argmax(dim=-1) == torch.arange(v_te.size(0)) ).float().mean().item())
            else:
                use_hier = arm == "shared_mjepa_9path_4level_hier"
                enable_cross = arm != "shared_intra_only_no_cross"
                eng = ReusablePredictiveEngine(
                    embed_dim=48,
                    predictor_embed_dim=24,
                    depth=4,
                    pred_depth=2,
                    hierarchical_levels=4 if use_hier else 1,
                    visual_img_size=(16, 16),
                    visual_patch_size=4,
                    audio_spec_size=(16, 16),
                    audio_patch_size=(4, 4),
                )
                opt = torch.optim.AdamW(
                    [p for n, p in eng.named_parameters() if not n.startswith("target_encoder.")],
                    lr=3e-3,
                    weight_decay=1e-4,
                )
                for _ in range(epochs):
                    perm = torch.randperm(v_tr.size(0))
                    for i in range(0, v_tr.size(0), bs):
                        bv = v_tr[perm[i : i + bs]]
                        ba = a_tr[perm[i : i + bs]]
                        vm_e, vm_p = sample_multiblock_2d_masks(bv.size(0), 4, 4, nenc=1, npred=2)
                        am_e, am_p = sample_audio_tf_masks(ba.size(0), 4, 4, npred=2)
                        step_d = eng.multimodal_jepa_step(
                            bv, vm_e, vm_p, ba, am_e, am_p,
                            enable_cross_modal=enable_cross,
                            enable_av_joint=enable_cross,
                            use_hierarchical=use_hier,
                        )
                        # Add VCLoss anti-collapse regularization on pooled audio & visual features
                        vz_f = eng.encoder(visual_x=bv)["visual"].mean(dim=1)
                        az_f = eng.encoder(audio_x=ba)["audio"].mean(dim=1)
                        vc_loss = 0.25 * (eng.regularizer.std_loss_fn(vz_f) + eng.regularizer.std_loss_fn(az_f))
                        tot = step_d["loss"] + vc_loss
                        opt.zero_grad()
                        tot.backward()
                        opt.step()
                        eng.update_target_ema(0.96)

                with torch.no_grad():
                    fv_tr = eng.encoder(visual_x=v_tr)["visual"].mean(dim=1)
                    fv_te = eng.encoder(visual_x=v_te)["visual"].mean(dim=1)
                    fa_tr = eng.encoder(audio_x=a_tr)["audio"].mean(dim=1)
                    fa_te = eng.encoder(audio_x=a_te)["audio"].mean(dim=1)
                    av_tr_out = eng.encoder(audio_x=a_tr, visual_x=v_tr)
                    av_te_out = eng.encoder(audio_x=a_te, visual_x=v_te)
                    fav_tr = torch.cat([av_tr_out["audio"].mean(dim=1), av_tr_out["visual"].mean(dim=1)], dim=-1)
                    fav_te = torch.cat([av_te_out["audio"].mean(dim=1), av_te_out["visual"].mean(dim=1)], dim=-1)

                    # Cross-modal retrieval accuracy: match predicted visual embedding from audio against true visual embeddings
                    pred_v_from_a = F.normalize(eng.a2v_predictor(eng.encoder(audio_x=a_te)["audio"]), dim=-1)
                    true_v = F.normalize(fv_te, dim=-1)
                    sim_mat = pred_v_from_a @ true_v.T
                    # Class-level cross-modal retrieval accuracy
                    top1_idx = sim_mat.argmax(dim=-1)
                    a2v_r1 = float((y_te[top1_idx] == y_te).float().mean().item())

            v_acc = train_linear_probe(fv_tr, y_tr, fv_te, y_te, num_classes=6)
            a_acc = train_linear_probe(fa_tr, y_tr, fa_te, y_te, num_classes=6)
            av_acc = train_linear_probe(fav_tr, y_tr, fav_te, y_te, num_classes=6)

            per_arm[arm].append({
                "visual_acc": round(v_acc, 4),
                "audio_acc": round(a_acc, 4),
                "joint_av_acc": round(av_acc, 4),
                "cross_modal_a2v_acc": round(a2v_r1, 4),
            })

    return {
        arm: {
            "visual_acc_mean": round(float(np.mean([r["visual_acc"] for r in runs])), 4),
            "audio_acc_mean": round(float(np.mean([r["audio_acc"] for r in runs])), 4),
            "joint_av_acc_mean": round(float(np.mean([r["joint_av_acc"] for r in runs])), 4),
            "cross_modal_a2v_acc_mean": round(float(np.mean([r["cross_modal_a2v_acc"] for r in runs])), 4),
            "runs": runs,
        }
        for arm, runs in per_arm.items()
    }


# ============================================================================
# E5: Guiding Question — Reusable Predictive Representation Engine Improving
#     Through Observed Outcomes Across Clover Cycles 1..5
# ============================================================================
def make_regime_cues(regime_id: int, batch_size: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(visual_cue [B, 3, 16, 16], audio_cue [B, 1, 16, 16], drift_vec [B, 2])` for `regime_id in {0, 1, 2}`:
    - Regime 0: Calm baseline (`drift = [0.0, 0.0]`)
    - Regime 1: Northeast current (`drift = [+0.13, +0.11]`), within control authority (`0.19`), signaled by top-bar visual + high-freq audio
    - Regime 2: Southwest cross-wind (`drift = [-0.13, -0.11]`), within control authority (`0.19`), signaled by bottom-bar visual + low-freq audio
    """
    vis = torch.randn((batch_size, 3, 16, 16), generator=g) * 0.15
    aud = torch.randn((batch_size, 1, 16, 16), generator=g) * 0.15
    if regime_id == 0:
        vis[:, 1, 6:10, 6:10] += 1.5
        aud[:, 0, :, 7:9] += 1.5
        drift = torch.tensor([[0.0, 0.0]]).expand(batch_size, 2)
    elif regime_id == 1:
        vis[:, 0, 1:5, 1:15] += 2.0
        aud[:, 0, :, 12:15] += 2.0
        drift = torch.tensor([[0.13, 0.11]]).expand(batch_size, 2)
    else:
        vis[:, 2, 11:15, 1:15] += 2.0
        aud[:, 0, :, 1:4] += 2.0
        drift = torch.tensor([[-0.13, -0.11]]).expand(batch_size, 2)
    return vis, aud, drift


def run_e5_outcome_driven_rpre(seeds: list[int] = [101, 202, 303]) -> dict[str, Any]:
    arms = (
        "arm1_static_open_loop_jepa",
        "arm2_naive_ungated_finetune",
        "arm3_rpre_state_only_outcome_growth",
        "arm4_full_clover_rpre_multimodal_growth",
    )
    per_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}

    for seed in seeds:
        set_seed(seed)
        g = torch.Generator().manual_seed(seed)

        # 1. Pre-train a shared base ReusablePredictiveEngine jointly on Regime 0 (calm baseline) trajectories + multimodal cues
        base_eng = ReusablePredictiveEngine(
            embed_dim=32,
            predictor_embed_dim=16,
            depth=3,
            pred_depth=2,
            hierarchical_levels=1,
            visual_img_size=(16, 16),
            visual_patch_size=4,
            audio_spec_size=(16, 16),
            audio_patch_size=(4, 4),
            state_dim=4,
            action_dim=2,
        )
        s_tr, a_tr = generate_trajectory_dataset(240, horizon=8, seed=seed)
        opt_joint = torch.optim.AdamW(
            [p for n, p in base_eng.named_parameters() if not n.startswith("target_encoder.") and not n.startswith("outcome_adapter.")],
            lr=3e-3,
            weight_decay=1e-4,
        )
        for _ in range(45):
            perm = torch.randperm(s_tr.size(0))
            for i in range(0, s_tr.size(0), 48):
                bs_s = s_tr[perm[i : i + 48]]
                bs_a = a_tr[perm[i : i + 48]]
                _, losses = base_eng.unroll_latent(
                    state_seq_bst=bs_s, actions_bat=bs_a, nsteps=7, unroll_mode="parallel", compute_loss=True, use_outcome_adapter=False
                )
                v_b, a_b = [], []
                for r_id in (0, 1, 2):
                    vr, ar, _ = make_regime_cues(r_id, 12, g)
                    v_b.append(vr); a_b.append(ar)
                v_all = torch.cat(v_b, dim=0)
                a_all = torch.cat(a_b, dim=0)
                vm_e, vm_p = sample_multiblock_2d_masks(v_all.size(0), 4, 4, nenc=1, npred=2)
                am_e, am_p = sample_audio_tf_masks(a_all.size(0), 4, 4, npred=2)
                d_step = base_eng.multimodal_jepa_step(
                    v_all, vm_e, vm_p, a_all, am_e, am_p, enable_cross_modal=True, enable_av_joint=False, use_hierarchical=False
                )
                vz_pool = base_eng.encoder(visual_x=v_all)["visual"].mean(dim=1)
                az_pool = base_eng.encoder(audio_x=a_all)["audio"].mean(dim=1)
                vc_l = 0.5 * (base_eng.regularizer.std_loss_fn(vz_pool) + base_eng.regularizer.std_loss_fn(az_pool))
                opt_joint.zero_grad()
                (losses[0] + 0.5 * (d_step["loss"] + vc_l)).backward()
                opt_joint.step()
                base_eng.update_target_ema(0.96)

        # Freeze encoder so latent coordinate system stays grounded while testing outcome adaptation
        for p in base_eng.encoder.parameters():
            p.requires_grad = False

        # Populate baseline Regime 0 anchor transitions for the non-regression gate
        with torch.no_grad():
            v0_anc, a0_anc, d0_anc = make_regime_cues(0, 64, g)
            s0_anc = (torch.rand((64, 4), generator=g) - 0.5) * 1.2
            act0_anc = (torch.rand((64, 2), generator=g) - 0.5) * 0.8
            s1_anc = step_2d_dynamics(s0_anc, act0_anc, d0_anc)
            z0_anc = base_eng.encoder(state_x=s0_anc)["state"].squeeze(1)
            z1_anc = base_eng.encoder(state_x=s1_anc)["state"].squeeze(1)
            zv0_anc = base_eng.encoder(visual_x=v0_anc)["visual"].mean(dim=1)
            za0_anc = base_eng.encoder(audio_x=a0_anc)["audio"].mean(dim=1)
            cue0_anc = 0.5 * (za0_anc + zv0_anc)

        for arm in arms:
            eng = copy.deepcopy(base_eng)
            if arm == "arm3_rpre_state_only_outcome_growth":
                eng.outcome_adapter.record_outcome(z0_anc, act0_anc, z1_anc, c_t=torch.zeros_like(cue0_anc), is_baseline=True)
            else:
                eng.outcome_adapter.record_outcome(z0_anc, act0_anc, z1_anc, c_t=cue0_anc, is_baseline=True)

            use_adapter = arm in ("arm3_rpre_state_only_outcome_growth", "arm4_full_clover_rpre_multimodal_growth")
            use_cue = arm == "arm4_full_clover_rpre_multimodal_growth"
            agent = CloverCycleAgent(
                eng, use_outcome_adapter=use_adapter, use_cross_modal_cue=use_cue, plan_length=6, n_iters=6, num_samples=96
            )

            cycle_metrics = []
            g_eval = torch.Generator().manual_seed(seed + 7000)

            for cycle in range(1, 6):
                # Evaluate across 16 episodes split across Regime 1 and Regime 2
                pred_mses = []
                goal_dists = []
                successes = 0

                for ep in range(16):
                    r_id = 1 if (ep % 2 == 0) else 2
                    vis_c, aud_c, drift_vec = make_regime_cues(r_id, 1, g_eval)
                    cur_s = (torch.rand((1, 4), generator=g_eval) - 0.5) * 1.2
                    cur_s[:, 2:] = 0.0
                    goal_s = (torch.rand((1, 4), generator=g_eval) - 0.5) * 1.2
                    goal_s[:, 2:] = 0.0

                    agent.set_direction(goal_s)
                    for step_i in range(10):
                        z_t, cue_t = agent.establish_context(
                            cur_s,
                            visual_x=vis_c if use_cue else None,
                            audio_x=aud_c if use_cue else None,
                        )
                        act_t = agent.select_action(z_t, steps_left=10 - step_i, seed=seed + cycle * 100 + ep * 10 + step_i)
                        next_s = step_2d_dynamics(cur_s, act_t, drift_vec)
                        _, p_mse = agent.observe_outcome(
                            z_t, act_t, next_s, cue=cue_t if use_cue else torch.zeros_like(z_t), is_baseline=False
                        )
                        pred_mses.append(p_mse)
                        if arm == "arm2_naive_ungated_finetune":
                            # Naive ungated online SGD directly on transition_predictor using only the latest episode's regime
                            z_tp1_obs = eng.encoder(state_x=next_s)["state"].squeeze(1).detach()
                            opt_naive = torch.optim.SGD(eng.transition_predictor.parameters(), lr=4e-2)
                            opt_naive.zero_grad()
                            p_out = eng.transition_predictor(z_t.detach().unsqueeze(-1), act_t.detach().unsqueeze(-1)).squeeze(-1)
                            F.mse_loss(p_out, z_tp1_obs).backward()
                            opt_naive.step()
                        cur_s = next_s

                    dist = float((cur_s[:, :2] - goal_s[:, :2]).norm(dim=-1).item())
                    goal_dists.append(dist)
                    if dist < 0.28:
                        successes += 1

                # Evaluate Baseline Regime 0 retention (does adapting to Regime 1/2 break Regime 0?)
                base_succ = 0
                g_base = torch.Generator().manual_seed(seed + 9000)
                for ep_b in range(10):
                    vis_0, aud_0, d_0 = make_regime_cues(0, 1, g_base)
                    cur_s = (torch.rand((1, 4), generator=g_base) - 0.5) * 1.2
                    cur_s[:, 2:] = 0.0
                    goal_s = (torch.rand((1, 4), generator=g_base) - 0.5) * 1.2
                    goal_s[:, 2:] = 0.0
                    agent.set_direction(goal_s)
                    for step_i in range(10):
                        z_t, _ = agent.establish_context(
                            cur_s, visual_x=vis_0 if use_cue else None, audio_x=aud_0 if use_cue else None
                        )
                        act_t = agent.select_action(z_t, steps_left=10 - step_i, seed=seed + 500 + ep_b * 10 + step_i)
                        cur_s = step_2d_dynamics(cur_s, act_t, d_0)
                    if float((cur_s[:, :2] - goal_s[:, :2]).norm(dim=-1).item()) < 0.28:
                        base_succ += 1

                # Evaluate Cross-Modal Audio-Only Transfer (when camera is occluded and only audio cue is available)
                audio_only_succ = 0
                g_ao = torch.Generator().manual_seed(seed + 9500)
                for ep_ao in range(10):
                    r_id = 1 if (ep_ao % 2 == 0) else 2
                    _, aud_ao, d_ao = make_regime_cues(r_id, 1, g_ao)
                    cur_s = (torch.rand((1, 4), generator=g_ao) - 0.5) * 1.2
                    cur_s[:, 2:] = 0.0
                    goal_s = (torch.rand((1, 4), generator=g_ao) - 0.5) * 1.2
                    goal_s[:, 2:] = 0.0
                    agent.set_direction(goal_s)
                    for step_i in range(10):
                        z_t, _ = agent.establish_context(cur_s, visual_x=None, audio_x=aud_ao if use_cue else None)
                        act_t = agent.select_action(z_t, steps_left=10 - step_i, seed=seed + 800 + ep_ao * 10 + step_i)
                        cur_s = step_2d_dynamics(cur_s, act_t, d_ao)
                    if float((cur_s[:, :2] - goal_s[:, :2]).norm(dim=-1).item()) < 0.28:
                        audio_only_succ += 1

                cycle_metrics.append({
                    "cycle": cycle,
                    "regime_pred_mse": round(float(np.mean(pred_mses)), 5),
                    "goal_success_rate": round(successes / 16.0, 4),
                    "mean_goal_dist": round(float(np.mean(goal_dists)), 4),
                    "baseline_retention_rate": round(base_succ / 10.0, 4),
                    "audio_only_transfer_rate": round(audio_only_succ / 10.0, 4),
                })

                # Stage 5 (Growth): Update OutcomeResidualAdapter at the end of the cycle
                if use_adapter:
                    agent.grow(steps=55, lr=8e-3)

            per_arm[arm].append({"seed": seed, "cycles": cycle_metrics})

    summary = {}
    for arm, runs in per_arm.items():
        by_cycle = []
        for c_idx in range(5):
            by_cycle.append({
                "cycle": c_idx + 1,
                "regime_pred_mse": round(float(np.mean([r["cycles"][c_idx]["regime_pred_mse"] for r in runs])), 5),
                "goal_success_rate": round(float(np.mean([r["cycles"][c_idx]["goal_success_rate"] for r in runs])), 4),
                "mean_goal_dist": round(float(np.mean([r["cycles"][c_idx]["mean_goal_dist"] for r in runs])), 4),
                "baseline_retention_rate": round(float(np.mean([r["cycles"][c_idx]["baseline_retention_rate"] for r in runs])), 4),
                "audio_only_transfer_rate": round(float(np.mean([r["cycles"][c_idx]["audio_only_transfer_rate"] for r in runs])), 4),
            })
        summary[arm] = {
            "cycle_1": by_cycle[0],
            "cycle_5": by_cycle[-1],
            "by_cycle": by_cycle,
        }
    return summary


def main() -> None:
    t0 = time.perf_counter()
    print("=== Running E1: I-JEPA Visual Latent Prediction ===")
    e1 = run_e1_ijepa_visual()
    for k, v in e1.items():
        print(f"  {k:24s} | clean_acc={v['clean_acc_mean']:.4f} | noisy_acc={v['noisy_acc_mean']:.4f} | std={v['repr_std_mean']:.4f} | flops_ratio={v['attn_flops_ratio']}")

    print("\n=== Running E2: Audio-JEPA Time-Frequency Latent Prediction ===")
    e2 = run_e2_audio_jepa()
    for k, v in e2.items():
        print(f"  {k:32s} | probe_acc={v['probe_acc_mean']:.4f} | latent_cos={v['latent_cos_mean']:.4f}")

    print("\n=== Running E3: EB-JEPA Regularizers & Action-Conditioned Latent Planning ===")
    e3 = run_e3_eb_jepa_planning()
    for k, v in e3.items():
        print(f"  {k:28s} | std={v['repr_std_mean']:.4f} | rank={v['effective_rank_mean']:5.2f} | rollout_mse={v['pos_rollout_mse_mean']:.4f} | plan_succ={v['planning_success_rate_mean']:.4f}")

    print("\n=== Running E4: MJEPA Shared Audio-Visual Representation Learning ===")
    e4 = run_e4_mjepa_multimodal()
    for k, v in e4.items():
        print(f"  {k:32s} | vis={v['visual_acc_mean']:.4f} | aud={v['audio_acc_mean']:.4f} | joint={v['joint_av_acc_mean']:.4f} | a2v={v['cross_modal_a2v_acc_mean']:.4f}")

    print("\n=== Running E5: Reusable Predictive Engine Outcome-Driven Growth (Cycles 1..5) ===")
    e5 = run_e5_outcome_driven_rpre()
    for k, v in e5.items():
        c1, c5 = v["cycle_1"], v["cycle_5"]
        print(
            f"  {k:40s} | C1->C5 mse: {c1['regime_pred_mse']:.4f}->{c5['regime_pred_mse']:.4f} | "
            f"succ: {c1['goal_success_rate']:.4f}->{c5['goal_success_rate']:.4f} | "
            f"base_ret: {c5['baseline_retention_rate']:.4f} | aud_only: {c5['audio_only_transfer_rate']:.4f}"
        )

    elapsed_s = round(time.perf_counter() - t0, 2)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "jepa_investigation_results.json"
    doc = {
        "date": "2026-11-07",
        "elapsed_s": elapsed_s,
        "torch_version": torch.__version__,
        "seeds": [101, 202, 303],
        "repositories_audited": {
            "ijepa": {"repo": "facebookresearch/ijepa", "commit": "52c1ae95d05f743e000e8f10a1f3a79b10cff048"},
            "audio_jepa": {"repo": "LudovicTuncay/Audio-JEPA", "commit": "ddd97ee9b88c00572f59bf2a00eb42120e6a4dea"},
            "eb_jepa": {"repo": "facebookresearch/eb_jepa", "commit": "966e61e9285b3a876f49b9774e9720d9a99a7925"},
            "mjepa": {"repo": "facebookresearch/MJEPA", "commit": "82f3472219f2f58672999ed654314b34ad590c07"},
        },
        "e1_ijepa_visual": e1,
        "e2_audio_jepa": e2,
        "e3_eb_jepa_planning": e3,
        "e4_mjepa_multimodal": e4,
        "e5_outcome_driven_rpre": e5,
    }
    out_file.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nSaved complete results to {out_file} ({elapsed_s}s)")


if __name__ == "__main__":
    main()
