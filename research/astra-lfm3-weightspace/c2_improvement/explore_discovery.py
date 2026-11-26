from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import unittest
from pathlib import Path


def setup_env(root: Path) -> Path:
    experiment = root / "c2_improvement"
    os.environ["HF_HOME"] = str(experiment / "cache" / "huggingface")
    os.environ["HF_MODULES_CACHE"] = str(experiment / "cache" / "modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "8"
    sys.path.insert(0, str(experiment / "vendor"))
    return experiment


def apply_adapter_scales(base_state: dict, alpha_proj2: float, gamma_post_ln: float, zero_post_bias: bool = False) -> dict:
    import torch

    if not math.isfinite(alpha_proj2) or not math.isfinite(gamma_post_ln) or alpha_proj2 < 0 or gamma_post_ln <= 0:
        raise ValueError("invalid adapter scale parameters")
    state = {k: v.clone() for k, v in base_state.items()}
    state["proj2.weight"] = (state["proj2.weight"].float() * alpha_proj2).to(state["proj2.weight"].dtype)
    state["proj2.bias"] = (state["proj2.bias"].float() * alpha_proj2).to(state["proj2.bias"].dtype)
    state["post_ln.weight"] = (state["post_ln.weight"].float() * gamma_post_ln).to(state["post_ln.weight"].dtype)
    if zero_post_bias:
        state["post_ln.bias"] = torch.zeros_like(state["post_ln.bias"])
    else:
        state["post_ln.bias"] = (state["post_ln.bias"].float() * gamma_post_ln).to(state["post_ln.bias"].dtype)
    return state


def build_hybridsft_samples(tokenizer, jsonl_path: Path, max_length: int = 8192) -> list[dict]:
    from hybrid_agent.prompts import SYSTEM_PROMPT
    import hybrid_sft.data as D

    orig_build = D.build_prompt

    def build_prompt_lfm3(tok, system, question):
        prompt = orig_build(tok, system, question)
        if prompt is not None:
            return prompt
        raw = tok.apply_chat_template(
            D._build_chat_messages(system, question), add_generation_prompt=True, tokenize=False
        )
        if isinstance(raw, str) and raw.endswith("<think>"):
            return raw + "\n"
        return None

    D.build_prompt = build_prompt_lfm3
    samples = []
    with Path(jsonl_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rec = json.loads(line)
            rounds = rec.get("rounds")
            if not isinstance(rounds, list) or not rounds:
                continue
            out = D.tokenize_trajectory(
                tokenizer,
                rec.get("system") or SYSTEM_PROMPT,
                rec.get("question", ""),
                rounds,
                max_length=max_length,
                text_only=False,
            )
            if out is not None:
                out["id"] = rec.get("id")
                samples.append(out)
    return samples


def evaluate_hybridsft_trajectories(model, tokenizer, samples: list[dict]) -> dict:
    import torch
    from hybrid_sft.data import collate_batch
    from hybrid_sft.model import SELF_CONDITIONED

    metrics = ("loss", "backbone_loss", "hybridlink_loss", "cos_loss", "mse_loss")
    per_sample = []
    for sample in samples:
        batch = collate_batch(tokenizer, [sample])
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                loss_mask=batch["loss_mask"],
                latent_mask=batch["latent_mask"],
                stage=SELF_CONDITIONED,
            )
        per_sample.append({k: float(out[k].item()) for k in metrics} | {"id": sample.get("id"), "n_tokens": sample["n_tokens"]})
    means = {k: sum(row[k] for row in per_sample) / len(per_sample) for k in metrics}
    return {"means": means, "count": len(per_sample), "per_sample": per_sample}


def cache_pass1_trajectories(model, tokenizer, samples: list[dict]) -> list[dict]:
    import torch
    from hybrid_sft.data import collate_batch

    embed_layer = model.backbone.get_input_embeddings()
    decoder = model.backbone.get_decoder()
    cached = []
    for sample in samples:
        batch = collate_batch(tokenizer, [sample])
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]
        latent_mask = batch["latent_mask"]
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            teacher_embeds = embed_layer(input_ids)
            pass1_hidden = decoder(
                inputs_embeds=teacher_embeds, attention_mask=attention_mask, use_cache=False, return_dict=True
            ).last_hidden_state
        cached.append(
            {
                "id": sample.get("id"),
                "n_tokens": sample["n_tokens"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "latent_mask": latent_mask,
                "teacher_embeds": teacher_embeds,
                "pass1_hidden": pass1_hidden,
            }
        )
    return cached


def evaluate_cached_pass2(model, cached_samples: list[dict]) -> dict:
    import torch
    import torch.nn.functional as F
    from hybrid_sft.model import EXIT_ANCHOR_HYBRIDLINK_WEIGHT, LAMBDA_COS, LAMBDA_MSE

    decoder = model.backbone.get_decoder()
    metrics = ("loss", "backbone_loss", "hybridlink_loss", "cos_loss", "mse_loss")
    per_sample = []
    for item in cached_samples:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        loss_mask = item["loss_mask"]
        latent_mask = item["latent_mask"]
        teacher_embeds = item["teacher_embeds"]
        pass1_hidden = item["pass1_hidden"]
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            input_embeds_mod = teacher_embeds.clone()
            pred_next = model.run_hybridlink(pass1_hidden[:, :-1], output_dtype=teacher_embeds.dtype)
            mask = latent_mask[:, :-1].unsqueeze(-1).expand_as(pred_next).bool()
            input_embeds_mod[:, 1:, :] = torch.where(mask, pred_next, input_embeds_mod[:, 1:, :])
            hidden = decoder(
                inputs_embeds=input_embeds_mod, attention_mask=attention_mask, use_cache=False, return_dict=True
            ).last_hidden_state
            backbone_loss = model._chunked_ce(hidden, input_ids, loss_mask)
            latent_src_mask = latent_mask[:, :-1] * attention_mask[:, :-1]
            next_is_latent = latent_mask[:, 1:]
            exit_mask = latent_src_mask * (1.0 - next_is_latent)
            preds = model.run_hybridlink(hidden[:, :-1], output_dtype=teacher_embeds.dtype)
            preds_f = preds.float()
            targets_f = teacher_embeds[:, 1:].float()
            cosine = 1.0 - F.cosine_similarity(preds_f, targets_f, dim=-1)
            mse = (preds_f - targets_f).pow(2).mean(dim=-1)
            cos_loss = model._masked_mean(cosine, latent_src_mask)
            mse_loss = model._masked_mean(mse, latent_src_mask)
            hl_loss = LAMBDA_COS * cos_loss + LAMBDA_MSE * mse_loss
            exit_cos = model._masked_mean(cosine, exit_mask)
            exit_mse = model._masked_mean(mse, exit_mask)
            hl_loss = hl_loss + EXIT_ANCHOR_HYBRIDLINK_WEIGHT * (LAMBDA_COS * exit_cos + LAMBDA_MSE * exit_mse)
            total_loss = backbone_loss + hl_loss
        per_sample.append(
            {
                "id": item["id"],
                "n_tokens": item["n_tokens"],
                "loss": float(total_loss.item()),
                "backbone_loss": float(backbone_loss.item()),
                "hybridlink_loss": float(hl_loss.item()),
                "cos_loss": float(cos_loss.item()),
                "mse_loss": float(mse_loss.item()),
            }
        )
    means = {k: sum(row[k] for row in per_sample) / len(per_sample) for k in metrics}
    return {"means": means, "count": len(per_sample), "per_sample": per_sample}


def collect_latent_branch_geometry(model, cached_samples: list[dict]) -> tuple[dict, dict]:
    import torch
    import torch.nn.functional as F

    link = model.hybridlink
    h_list = []
    mid_list = []
    out_list = []
    tgt_list = []
    exit_flags = []

    for item in cached_samples:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        latent_mask = item["latent_mask"]
        teacher_embeds = item["teacher_embeds"]
        pass1_hidden = item["pass1_hidden"]
        latent_src = (latent_mask[:, :-1] * attention_mask[:, :-1]) > 0.5
        next_lat = latent_mask[:, 1:] > 0.5
        if not latent_src.any():
            continue
        h_lat = pass1_hidden[:, :-1][latent_src].float()
        tgt_lat = teacher_embeds[:, 1:][latent_src].float()
        is_exit = (~next_lat[latent_src]).float()
        with torch.inference_mode():
            x_ln = F.layer_norm(h_lat, (2048,), link.pre_ln.weight.float(), link.pre_ln.bias.float(), link.pre_ln.eps)
            mid = F.gelu(F.linear(x_ln, link.proj1.weight.float(), link.proj1.bias.float()))
            out_lat = F.linear(mid, link.proj2.weight.float(), link.proj2.bias.float())
        h_list.append(h_lat)
        mid_list.append(mid)
        out_list.append(out_lat)
        tgt_list.append(tgt_lat)
        exit_flags.append(is_exit)

    H = torch.cat(h_list, dim=0)
    M = torch.cat(mid_list, dim=0)
    O = torch.cat(out_list, dim=0)
    T = torch.cat(tgt_list, dim=0)
    E = torch.cat(exit_flags, dim=0)

    def ln0(x):
        return F.layer_norm(x, (2048,), None, None, 1e-5)

    post_w = link.post_ln.weight.float()
    post_b = link.post_ln.bias.float()

    def weighted_cos_and_mse(preds):
        cos = 1.0 - F.cosine_similarity(preds, T, dim=-1)
        mse = (preds - T).pow(2).mean(dim=-1)
        all_cos = float(cos.mean().item())
        all_mse = float(mse.mean().item())
        exit_mask = E > 0.5
        exit_cos = float(cos[exit_mask].mean().item()) if exit_mask.any() else 0.0
        exit_mse = float(mse[exit_mask].mean().item()) if exit_mask.any() else 0.0
        hl_loss = (all_cos + 0.2 * all_mse) + (exit_cos + 0.2 * exit_mse)
        return {
            "cos_loss": all_cos,
            "mean_cos": 1.0 - all_cos,
            "exit_cos_loss": exit_cos,
            "mse_loss": all_mse,
            "exit_mse_loss": exit_mse,
            "hybridlink_loss": hl_loss,
            "pred_rms": float(preds.square().mean().sqrt().item()),
        }

    alpha_sweep = {}
    for alpha in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, 128.0, 256.0):
        z = ln0(H + alpha * O) * post_w + post_b
        alpha_sweep[str(alpha)] = weighted_cos_and_mse(z)

    z_pure_out = ln0(O) * post_w + post_b
    alpha_sweep["pure_out"] = weighted_cos_and_mse(z_pure_out)

    gamma_sweep = {}
    target_rms = float(T.square().mean().sqrt().item())
    for alpha in (1.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        base_z = ln0(H + alpha * O) * post_w + post_b
        weights = 1.0 + E[:, None]
        num = float((weights * base_z * T).sum().item())
        den = float((weights * base_z * base_z).sum().item())
        gamma_ls = max(1e-6, num / max(den, 1e-12))
        gamma_rms = target_rms / max(float(base_z.square().mean().sqrt().item()), 1e-12)
        gamma_sweep[str(alpha)] = {
            "gamma_ls": gamma_ls,
            "gamma_rms": gamma_rms,
            "at_gamma_1.0": weighted_cos_and_mse(base_z),
            "at_gamma_ls": weighted_cos_and_mse(base_z * gamma_ls),
            "at_gamma_rms": weighted_cos_and_mse(base_z * gamma_rms),
        }

    # Coordinate-wise RMS match and coordinate-wise Least-Squares (Gauss-Markov) calibration
    z0 = ln0(H + O)
    weights = 1.0 + E[:, None]
    w_sum = weights.sum(dim=0)
    z_mean = (weights * z0).sum(dim=0) / w_sum
    t_mean = (weights * T).sum(dim=0) / w_sum
    z_var = (weights * (z0 - z_mean).pow(2)).sum(dim=0) / w_sum
    zt_cov = (weights * (z0 - z_mean) * (T - t_mean)).sum(dim=0) / w_sum
    t_std = ((weights * (T - t_mean).pow(2)).sum(dim=0) / w_sum).sqrt()

    gamma_diag_rms = (t_std / (z_var.sqrt() + 1e-6)).clamp(min=1e-4, max=0.2)
    beta_diag_rms = t_mean - gamma_diag_rms * z_mean
    diag_rms_stats = weighted_cos_and_mse(z0 * gamma_diag_rms + beta_diag_rms)

    gamma_diag_ls = zt_cov / (z_var + 1e-3)
    beta_diag_ls = t_mean - gamma_diag_ls * z_mean
    pred_diag_ls = z0 * gamma_diag_ls + beta_diag_ls
    diag_ls_stats = weighted_cos_and_mse(pred_diag_ls)

    # Also norm-matched version of the Gauss-Markov direction
    ls_rms = float(pred_diag_ls.square().mean().sqrt().item())
    scale_to_target_rms = target_rms / max(ls_rms, 1e-12)
    gamma_diag_ls_norm = gamma_diag_ls * scale_to_target_rms
    beta_diag_ls_norm = beta_diag_ls * scale_to_target_rms
    diag_ls_norm_stats = weighted_cos_and_mse(z0 * gamma_diag_ls_norm + beta_diag_ls_norm)

    # Also test coordinate-wise LS at best alpha
    best_a_str = min(
        ("1.0", "2.0", "4.0", "8.0", "16.0", "32.0"),
        key=lambda a: alpha_sweep[a]["cos_loss"],
    )
    best_a = float(best_a_str)
    za = ln0(H + best_a * O)
    za_mean = (weights * za).sum(dim=0) / w_sum
    za_var = (weights * (za - za_mean).pow(2)).sum(dim=0) / w_sum
    zat_cov = (weights * (za - za_mean) * (T - t_mean)).sum(dim=0) / w_sum
    gamma_best_a_ls = zat_cov / (za_var + 1e-3)
    beta_best_a_ls = t_mean - gamma_best_a_ls * za_mean
    best_a_ls_stats = weighted_cos_and_mse(za * gamma_best_a_ls + beta_best_a_ls)

    fitted_tensors = {
        "gamma_diag_rms": gamma_diag_rms,
        "beta_diag_rms": beta_diag_rms,
        "gamma_diag_ls": gamma_diag_ls,
        "beta_diag_ls": beta_diag_ls,
        "gamma_diag_ls_norm": gamma_diag_ls_norm,
        "beta_diag_ls_norm": beta_diag_ls_norm,
        "best_a": best_a,
        "gamma_best_a_ls": gamma_best_a_ls,
        "beta_best_a_ls": beta_best_a_ls,
    }

    return (
        {
            "total_latent_positions": int(H.shape[0]),
            "exit_positions": int((E > 0.5).sum().item()),
            "h_rms": float(H.square().mean().sqrt().item()),
            "out_rms": float(O.square().mean().sqrt().item()),
            "target_embed_rms": target_rms,
            "target_interior_rms": float(T[E <= 0.5].square().mean().sqrt().item()),
            "target_exit_rms": float(T[E > 0.5].square().mean().sqrt().item()),
            "alpha_sweep_pass1": alpha_sweep,
            "gamma_sweep_pass1": gamma_sweep,
            "diag_rms_pass1": diag_rms_stats,
            "diag_ls_pass1": diag_ls_stats,
            "diag_ls_norm_pass1": diag_ls_norm_stats,
            "best_a_ls_pass1": best_a_ls_stats,
        },
        fitted_tensors,
    )


def run_explore(root: Path) -> None:
    experiment = setup_env(root)
    import torch
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
    cached_disc = cache_pass1_trajectories(model, tokenizer, disc_samples)
    geom, fitted = collect_latent_branch_geometry(model, cached_disc)

    check_full = evaluate_hybridsft_trajectories(model, tokenizer, disc_samples[:1])["means"]
    check_cached = evaluate_cached_pass2(model, cached_disc[:1])["means"]
    for k in check_full:
        if abs(check_full[k] - check_cached[k]) > 1e-5:
            raise ValueError(f"cached pass1 mismatch on {k}: {check_full[k]} vs {check_cached[k]}")

    best_alpha = float(fitted["best_a"])
    best_alpha_str = str(best_alpha)

    grid_results = {}
    candidates = [
        ("baseline_a1_g1", 1.0, 1.0, False),
        ("a1_g_rms", 1.0, geom["gamma_sweep_pass1"]["1.0"]["gamma_rms"], False),
        ("a1_g_ls", 1.0, geom["gamma_sweep_pass1"]["1.0"]["gamma_ls"], False),
        ("best_alpha_g1", best_alpha, 1.0, False),
    ]
    for name, alpha, gamma, zero_bias in candidates:
        state = apply_adapter_scales(base_adapter, alpha, gamma, zero_post_bias=zero_bias)
        model.hybridlink.load_state_dict(state, strict=True)
        res = evaluate_cached_pass2(model, cached_disc)
        grid_results[name] = {
            "alpha_proj2": alpha,
            "gamma_post_ln": gamma,
            "zero_post_bias": zero_bias,
            "means": res["means"],
        }
        print(json.dumps({"candidate": name, "alpha": alpha, "gamma": round(gamma, 6), **res["means"]}), flush=True)

    for diag_name, a_val, gw_key, gb_key in (
        ("diag_rms", 1.0, "gamma_diag_rms", "beta_diag_rms"),
        ("diag_ls", 1.0, "gamma_diag_ls", "beta_diag_ls"),
        ("diag_ls_norm", 1.0, "gamma_diag_ls_norm", "beta_diag_ls_norm"),
        ("best_a_diag_ls", best_alpha, "gamma_best_a_ls", "beta_best_a_ls"),
    ):
        diag_state = apply_adapter_scales(base_adapter, a_val, 1.0, zero_post_bias=False)
        diag_state["post_ln.weight"] = fitted[gw_key].to(diag_state["post_ln.weight"].dtype)
        diag_state["post_ln.bias"] = fitted[gb_key].to(diag_state["post_ln.bias"].dtype)
        model.hybridlink.load_state_dict(diag_state, strict=True)
        res_diag = evaluate_cached_pass2(model, cached_disc)
        grid_results[diag_name] = {
            "alpha_proj2": a_val,
            "gamma_post_ln": gw_key,
            "means": res_diag["means"],
        }
        print(json.dumps({"candidate": diag_name, "alpha": a_val, **res_diag["means"]}), flush=True)

    output = {
        "discovery_trajectory_count": len(disc_samples),
        "best_pass1_alpha": best_alpha,
        "geometry": geom,
        "grid_self_conditioned": grid_results,
    }
    out_path = experiment / "DISCOVERY_EXPLORATION.json"
    with out_path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")


class ExploreTests(unittest.TestCase):
    def test_apply_adapter_scales(self):
        import torch

        base = {
            "proj2.weight": torch.ones((2, 2), dtype=torch.bfloat16),
            "proj2.bias": torch.ones((2,), dtype=torch.bfloat16),
            "post_ln.weight": torch.ones((2,), dtype=torch.bfloat16),
            "post_ln.bias": torch.ones((2,), dtype=torch.bfloat16),
        }
        scaled = apply_adapter_scales(base, alpha_proj2=4.0, gamma_post_ln=0.25, zero_post_bias=False)
        self.assertTrue(torch.allclose(scaled["proj2.weight"].float(), torch.full((2, 2), 4.0)))
        self.assertTrue(torch.allclose(scaled["post_ln.weight"].float(), torch.full((2,), 0.25)))
        self.assertTrue(torch.allclose(base["proj2.weight"].float(), torch.ones((2, 2))))

    def test_invalid_scales_rejected(self):
        with self.assertRaises(ValueError):
            apply_adapter_scales({}, alpha_proj2=-1.0, gamma_post_ln=1.0)
        with self.assertRaises(ValueError):
            apply_adapter_scales({}, alpha_proj2=1.0, gamma_post_ln=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root:
        run_explore(args.root.resolve(strict=True))
    else:
        parser.error("choose --self-test or --root")


if __name__ == "__main__":
    main()
