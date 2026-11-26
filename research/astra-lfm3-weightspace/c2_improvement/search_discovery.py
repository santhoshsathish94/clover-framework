from __future__ import annotations

import argparse
import json
import math
import sys
import unittest
from pathlib import Path

from explore_discovery import (
    apply_adapter_scales,
    build_hybridsft_samples,
    cache_pass1_trajectories,
    evaluate_cached_pass2,
    setup_env,
)


def fit_coord_cosine_weights(z0, targets, weights, tau: float, gamma0: float):
    import torch
    import torch.nn.functional as F

    if not (0.0 <= tau <= 1.0) or gamma0 <= 0.0:
        raise ValueError("invalid tau or gamma0")
    t_unit = F.normalize(targets, dim=-1) * math.sqrt(targets.shape[-1])
    w = weights[:, None]
    num = (w * z0 * t_unit).sum(dim=0)
    den = (w * z0 * z0).sum(dim=0) + 1e-4
    raw = (num / den).clamp(min=0.0)
    mean_raw = float(raw.mean().item())
    if mean_raw <= 1e-8:
        norm_raw = torch.ones_like(raw)
    else:
        norm_raw = raw / mean_raw
    blended = ((1.0 - tau) + tau * norm_raw).clamp(min=0.5, max=1.5)
    blended = blended / blended.mean()
    return blended * gamma0


def fit_lowrank_proj2_delta(mid, h_lat, out_lat, targets, weights, rank: int, ridge: float, eta: float):
    import torch
    import torch.nn.functional as F

    if rank < 1 or ridge <= 0.0 or eta <= 0.0:
        raise ValueError("invalid rank, ridge, or eta")
    u_cur = h_lat + out_lat
    u_rms = u_cur.square().mean(dim=-1, keepdim=True).sqrt()
    t_ln = F.layer_norm(targets, (targets.shape[-1],), None, None, 1e-5)
    u_star = u_rms * t_ln
    residual = u_star - u_cur

    w_sqrt = weights.sqrt()[:, None]
    m_w = mid * w_sqrt
    r_w = residual * w_sqrt

    _, _, vh = torch.linalg.svd(m_w, full_matrices=False)
    v_k = vh[:rank, :].T  # [D, k]
    m_k = m_w @ v_k  # [N, k]
    gram = m_k.T @ m_k + ridge * torch.eye(rank, dtype=mid.dtype, device=mid.device)
    rhs = m_k.T @ r_w  # [k, D]
    coeff = torch.linalg.solve(gram, rhs)  # [k, D]
    delta_w2 = eta * (coeff.T @ v_k.T)  # [D, D]
    return delta_w2


def compare_against_baseline(candidate_rows: list[dict], baseline_rows: list[dict]) -> dict:
    metrics = ("loss", "backbone_loss", "hybridlink_loss", "cos_loss", "mse_loss")
    out = {}
    for m in metrics:
        c_vals = [r[m] for r in candidate_rows]
        b_vals = [r[m] for r in baseline_rows]
        better = sum(1 for c, b in zip(c_vals, b_vals) if c < b - 1e-7)
        worse = sum(1 for c, b in zip(c_vals, b_vals) if c > b + 1e-7)
        out[m] = {
            "mean": sum(c_vals) / len(c_vals),
            "delta": (sum(c_vals) - sum(b_vals)) / len(c_vals),
            "better": better,
            "worse": worse,
        }
    return out


def run_search(root: Path) -> None:
    experiment = setup_env(root)
    import torch
    import torch.nn.functional as F
    from hybrid_agent.model import add_anchor_tokens
    from hybrid_sft.model import HybridSFTModel
    from transformers import AutoTokenizer

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    c2_dir = root / "c2c3_weights" / "c2"
    tokenizer = AutoTokenizer.from_pretrained(str(c2_dir), trust_remote_code=True, local_files_only=True)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = HybridSFTModel.from_pretrained(
        c2_dir, dtype=torch.bfloat16, trust_remote_code=True, require_hybridlink=True
    )
    model.eval().requires_grad_(False)
    base_adapter = {k: v.clone() for k, v in model.hybridlink.state_dict().items()}

    disc_samples = build_hybridsft_samples(tokenizer, experiment / "splits" / "discovery24.jsonl", max_length=2048)
    if len(disc_samples) < 16:
        raise ValueError(f"expected >= 16 discovery samples, got {len(disc_samples)}")
    fit_samples = disc_samples[:9]
    val_samples = disc_samples[9:]

    cached_fit = cache_pass1_trajectories(model, tokenizer, fit_samples)
    cached_val = cache_pass1_trajectories(model, tokenizer, val_samples)

    # Extract Pass-1 latent features on fit_samples ONLY
    link = model.hybridlink
    h_list, mid_list, out_list, tgt_list, w_list = [], [], [], [], []
    for item in cached_fit:
        latent_src = (item["latent_mask"][:, :-1] * item["attention_mask"][:, :-1]) > 0.5
        next_lat = item["latent_mask"][:, 1:] > 0.5
        if not latent_src.any():
            continue
        h_lat = item["pass1_hidden"][:, :-1][latent_src].float()
        tgt_lat = item["teacher_embeds"][:, 1:][latent_src].float()
        is_exit = (~next_lat[latent_src]).float()
        x_ln = F.layer_norm(h_lat, (2048,), link.pre_ln.weight.float(), link.pre_ln.bias.float(), link.pre_ln.eps)
        mid = F.gelu(F.linear(x_ln, link.proj1.weight.float(), link.proj1.bias.float()))
        out_lat = F.linear(mid, link.proj2.weight.float(), link.proj2.bias.float())
        h_list.append(h_lat)
        mid_list.append(mid)
        out_list.append(out_lat)
        tgt_list.append(tgt_lat)
        w_list.append(1.0 + is_exit)

    H_fit = torch.cat(h_list, dim=0)
    M_fit = torch.cat(mid_list, dim=0)
    O_fit = torch.cat(out_list, dim=0)
    T_fit = torch.cat(tgt_list, dim=0)
    W_fit = torch.cat(w_list, dim=0)
    Z0_fit = F.layer_norm(H_fit + O_fit, (2048,), None, None, 1e-5)

    # First evaluate baseline on fit and val
    model.hybridlink.load_state_dict(base_adapter, strict=True)
    base_fit_res = evaluate_cached_pass2(model, cached_fit)
    base_val_res = evaluate_cached_pass2(model, cached_val)
    print(json.dumps({"split": "baseline_fit", **base_fit_res["means"]}), flush=True)
    print(json.dumps({"split": "baseline_val", **base_val_res["means"]}), flush=True)

    candidate_states = {}

    # Family 1: gentle scalar post_ln gamma
    for gamma in (0.6, 0.75, 0.85, 0.92):
        candidate_states[f"scalar_g{gamma}"] = apply_adapter_scales(base_adapter, 1.0, gamma, zero_post_bias=False)

    # Family 2: coordinate-wise cosine alignment (no interior-mean bias shift)
    for tau, gamma0 in ((0.2, 0.85), (0.35, 0.85), (0.35, 0.95), (0.5, 1.0)):
        gw = fit_coord_cosine_weights(Z0_fit, T_fit, W_fit, tau=tau, gamma0=gamma0)
        st = {k: v.clone() for k, v in base_adapter.items()}
        st["post_ln.weight"] = gw.to(st["post_ln.weight"].dtype)
        st["post_ln.bias"] = (st["post_ln.bias"].float() * gamma0).to(st["post_ln.bias"].dtype)
        candidate_states[f"coord_t{tau}_g{gamma0}"] = st

    # Family 3: low-rank ridge update to proj2.weight (before post_ln), pre-screened on Pass-1 cos_loss
    for rank, eta, gamma0 in (
        (16, 0.2, 1.0),
        (16, 0.2, 0.85),
        (32, 0.2, 1.0),
        (32, 0.2, 0.85),
        (32, 0.4, 0.85),
        (64, 0.2, 0.85),
    ):
        dw2 = fit_lowrank_proj2_delta(M_fit, H_fit, O_fit, T_fit, W_fit, rank=rank, ridge=100.0, eta=eta)
        st = apply_adapter_scales(base_adapter, 1.0, gamma0, zero_post_bias=False)
        st["proj2.weight"] = (base_adapter["proj2.weight"].float() + dw2).to(st["proj2.weight"].dtype)
        candidate_states[f"lr_r{rank}_e{eta}_g{gamma0}"] = st

    # Fast Pass-1 + Pass-2 screening on cached_fit first; only run cached_val on candidates that strictly improve cached_fit!
    fit_Transfer = {}
    qualifying_for_val = []
    for name, st in candidate_states.items():
        model.hybridlink.load_state_dict(st, strict=True)
        fit_res = evaluate_cached_pass2(model, cached_fit)
        cmp_fit = compare_against_baseline(fit_res["per_sample"], base_fit_res["per_sample"])
        fit_Transfer[name] = cmp_fit
        improves_all = (
            cmp_fit["loss"]["delta"] < 0.0
            and cmp_fit["backbone_loss"]["delta"] <= 0.0005
            and cmp_fit["hybridlink_loss"]["delta"] < 0.0
            and cmp_fit["cos_loss"]["delta"] <= 0.0005
            and cmp_fit["mse_loss"]["delta"] <= 0.0
        )
        print(
            json.dumps(
                {
                    "fit_candidate": name,
                    "qualifies": improves_all,
                    "d_loss": round(cmp_fit["loss"]["delta"], 5),
                    "d_bb": round(cmp_fit["backbone_loss"]["delta"], 5),
                    "d_hl": round(cmp_fit["hybridlink_loss"]["delta"], 5),
                    "d_cos": round(cmp_fit["cos_loss"]["delta"], 5),
                    "d_mse": round(cmp_fit["mse_loss"]["delta"], 5),
                }
            ),
            flush=True,
        )
        if improves_all:
            qualifying_for_val.append(name)

    val_results = {}
    for name in qualifying_for_val:
        model.hybridlink.load_state_dict(candidate_states[name], strict=True)
        val_res = evaluate_cached_pass2(model, cached_val)
        cmp_val = compare_against_baseline(val_res["per_sample"], base_val_res["per_sample"])
        val_results[name] = cmp_val
        print(
            json.dumps(
                {
                    "val_candidate": name,
                    "d_loss": round(cmp_val["loss"]["delta"], 5),
                    "d_bb": round(cmp_val["backbone_loss"]["delta"], 5),
                    "d_hl": round(cmp_val["hybridlink_loss"]["delta"], 5),
                    "d_cos": round(cmp_val["cos_loss"]["delta"], 5),
                    "d_mse": round(cmp_val["mse_loss"]["delta"], 5),
                    "loss_better_worse": [cmp_val["loss"]["better"], cmp_val["loss"]["worse"]],
                    "bb_better_worse": [cmp_val["backbone_loss"]["better"], cmp_val["backbone_loss"]["worse"]],
                }
            ),
            flush=True,
        )

    out_path = experiment / "DISCOVERY_SEARCH.json"
    with out_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "fit_count": len(cached_fit),
                "val_count": len(cached_val),
                "baseline_fit": base_fit_res["means"],
                "baseline_val": base_val_res["means"],
                "fit_results": fit_Transfer,
                "qualifying_for_val": qualifying_for_val,
                "val_results": val_results,
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


class SearchTests(unittest.TestCase):
    def test_compare_against_baseline(self):
        base = [
            {"loss": 2.0, "backbone_loss": 1.0, "hybridlink_loss": 1.0, "cos_loss": 0.5, "mse_loss": 0.5},
            {"loss": 2.0, "backbone_loss": 1.0, "hybridlink_loss": 1.0, "cos_loss": 0.5, "mse_loss": 0.5},
        ]
        cand = [
            {"loss": 1.8, "backbone_loss": 0.9, "hybridlink_loss": 0.9, "cos_loss": 0.45, "mse_loss": 0.45},
            {"loss": 1.9, "backbone_loss": 0.95, "hybridlink_loss": 0.95, "cos_loss": 0.48, "mse_loss": 0.47},
        ]
        cmp_res = compare_against_baseline(cand, base)
        self.assertAlmostEqual(cmp_res["loss"]["delta"], -0.15)
        self.assertEqual(cmp_res["loss"]["better"], 2)
        self.assertEqual(cmp_res["loss"]["worse"], 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root:
        run_search(args.root.resolve(strict=True))
    else:
        parser.error("choose --self-test or --root")


if __name__ == "__main__":
    main()
