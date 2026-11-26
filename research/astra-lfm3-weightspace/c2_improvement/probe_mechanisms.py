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
    setup_env,
)


def detailed_hybridsft_eval(model, tokenizer, samples: list[dict]) -> dict:
    import torch
    import torch.nn.functional as F
    from hybrid_sft.data import collate_batch
    from hybrid_sft.model import EXIT_ANCHOR_HYBRIDLINK_WEIGHT, LAMBDA_COS, LAMBDA_MSE

    embed_layer = model.backbone.get_input_embeddings()
    output_head = model.backbone.get_output_embeddings()
    decoder = model.backbone.get_decoder()

    rows = []
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
            input_embeds_mod = teacher_embeds.clone()
            pred_next = model.run_hybridlink(pass1_hidden[:, :-1], output_dtype=teacher_embeds.dtype)
            mask = latent_mask[:, :-1].unsqueeze(-1).expand_as(pred_next).bool()
            input_embeds_mod[:, 1:, :] = torch.where(mask, pred_next, input_embeds_mod[:, 1:, :])
            hidden = decoder(
                inputs_embeds=input_embeds_mod, attention_mask=attention_mask, use_cache=False, return_dict=True
            ).last_hidden_state

            # Compute per-token CE breakdown
            src = hidden[:, :-1]
            tgt = input_ids[:, 1:]
            msk = loss_mask[:, 1:].float()
            logits = output_head(src).float()
            token_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none").view_as(tgt)

            backbone_loss = (token_ce * msk).sum() / msk.sum().clamp(min=1.0)
            entry_mask = (tgt == 125033) & (msk > 0)
            exit_ce_mask = (tgt == 125034) & (msk > 0)
            text_ce_mask = (tgt != 125033) & (tgt != 125034) & (msk > 0)

            entry_ce = float(token_ce[entry_mask].mean().item()) if entry_mask.any() else 0.0
            exit_ce = float(token_ce[exit_ce_mask].mean().item()) if exit_ce_mask.any() else 0.0
            text_ce = float(token_ce[text_ce_mask].mean().item()) if text_ce_mask.any() else 0.0

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
            exit_cos = model._masked_mean(cosine, exit_mask)
            exit_mse = model._masked_mean(mse, exit_mask)
            hl_loss = (LAMBDA_COS * cos_loss + LAMBDA_MSE * mse_loss) + EXIT_ANCHOR_HYBRIDLINK_WEIGHT * (
                LAMBDA_COS * exit_cos + LAMBDA_MSE * exit_mse
            )
            total_loss = backbone_loss + hl_loss

        rows.append(
            {
                "id": sample.get("id"),
                "loss": float(total_loss.item()),
                "backbone_loss": float(backbone_loss.item()),
                "text_ce": text_ce,
                "entry_ce": entry_ce,
                "exit_ce": exit_ce,
                "hybridlink_loss": float(hl_loss.item()),
                "cos_loss": float(cos_loss.item()),
                "exit_cos_loss": float(exit_cos.item()),
                "mse_loss": float(mse_loss.item()),
                "exit_mse_loss": float(exit_mse.item()),
            }
        )
    keys = ("loss", "backbone_loss", "text_ce", "entry_ce", "exit_ce", "hybridlink_loss", "cos_loss", "exit_cos_loss", "mse_loss", "exit_mse_loss")
    means = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    return {"means": means, "per_sample": rows}


def run_probe(root: Path) -> None:
    experiment = setup_env(root)
    import torch
    from safetensors import safe_open
    from hybrid_agent.model import add_anchor_tokens
    from hybrid_sft.model import HybridSFTModel
    from transformers import AutoTokenizer

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    c2_dir = root / "c2c3_weights" / "c2"
    orig_sf_path = root / "original" / "model.safetensors"

    tokenizer = AutoTokenizer.from_pretrained(str(c2_dir), trust_remote_code=True, local_files_only=True)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = HybridSFTModel.from_pretrained(
        c2_dir, dtype=torch.bfloat16, trust_remote_code=True, require_hybridlink=True
    )
    model.eval().requires_grad_(False)

    disc_samples = build_hybridsft_samples(tokenizer, experiment / "splits" / "discovery24.jsonl", max_length=2048)
    fit_samples = disc_samples[:5]
    val_samples = disc_samples[5:]

    base_backbone = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    base_adapter = {k: v.clone() for k, v in model.hybridlink.state_dict().items()}

    # Load original base LM weights to form the exact HybridSFT task vector delta_W = W_c2 - W_orig
    orig_tensors = {}
    with safe_open(orig_sf_path, framework="pt", device="cpu") as sf:
        for k in base_backbone:
            if k == "lm_head.weight":
                continue
            orig_key = "model.language_model." + k[len("model.") :]
            orig_tensors[k] = sf.get_tensor(orig_key)

    base_fit = detailed_hybridsft_eval(model, tokenizer, fit_samples)
    print(json.dumps({"arm": "baseline_fit", **base_fit["means"]}), flush=True)

    results = {"baseline_fit": base_fit["means"]}

    # Test 1: Anchor token task-vector extrapolation on rows 125033 (<latent>) and 125034 (</latent>)
    emb_c2 = base_backbone["model.embed_tokens.weight"].float()
    emb_orig = orig_tensors["model.embed_tokens.weight"].float()
    for scale_anchor in (1.5, 2.0, 3.0):
        mod_emb = base_backbone["model.embed_tokens.weight"].clone()
        for tid in (125033, 125034):
            delta_row = emb_c2[tid] - emb_orig[tid]
            mod_emb[tid] = (emb_orig[tid] + scale_anchor * delta_row).to(mod_emb.dtype)
        model.backbone.model.embed_tokens.weight.copy_(mod_emb)
        res = detailed_hybridsft_eval(model, tokenizer, fit_samples)
        name = f"anchor_extrap_{scale_anchor}"
        results[name] = res["means"]
        print(json.dumps({"arm": name, **{k: round(res["means"][k] - base_fit["means"][k], 5) for k in ("loss", "backbone_loss", "text_ce", "entry_ce", "exit_ce", "hybridlink_loss", "cos_loss", "mse_loss")}}), flush=True)

    model.backbone.model.embed_tokens.weight.copy_(base_backbone["model.embed_tokens.weight"])

    # Test 2: Whole-backbone task-vector scaling W(tau) = W_orig + tau * (W_c2 - W_orig)
    for tau in (0.9, 1.05, 1.1, 1.2):
        for k, p in model.backbone.named_parameters():
            if k in orig_tensors:
                w_o = orig_tensors[k].float()
                w_c = base_backbone[k].float()
                p.copy_((w_o + tau * (w_c - w_o)).to(p.dtype))
        res = detailed_hybridsft_eval(model, tokenizer, fit_samples)
        name = f"backbone_tau_{tau}"
        results[name] = res["means"]
        print(json.dumps({"arm": name, **{k: round(res["means"][k] - base_fit["means"][k], 5) for k in ("loss", "backbone_loss", "text_ce", "entry_ce", "exit_ce", "hybridlink_loss", "cos_loss", "mse_loss")}}), flush=True)

    # Restore exact C2 backbone
    for k, p in model.backbone.named_parameters():
        if k in base_backbone:
            p.copy_(base_backbone[k])

    # Test 3: Combined backbone task-vector tau + gentle post_ln gamma
    for tau, gamma in ((1.05, 0.95), (1.1, 0.92), (1.1, 0.95), (1.15, 0.92)):
        for k, p in model.backbone.named_parameters():
            if k in orig_tensors:
                w_o = orig_tensors[k].float()
                w_c = base_backbone[k].float()
                p.copy_((w_o + tau * (w_c - w_o)).to(p.dtype))
        st = apply_adapter_scales(base_adapter, 1.0, gamma, zero_post_bias=False)
        model.hybridlink.load_state_dict(st, strict=True)
        res = detailed_hybridsft_eval(model, tokenizer, fit_samples)
        name = f"combo_tau{tau}_g{gamma}"
        results[name] = res["means"]
        print(json.dumps({"arm": name, **{k: round(res["means"][k] - base_fit["means"][k], 5) for k in ("loss", "backbone_loss", "text_ce", "entry_ce", "exit_ce", "hybridlink_loss", "cos_loss", "mse_loss")}}), flush=True)

    # Restore exact C2 backbone and adapter
    for k, p in model.backbone.named_parameters():
        if k in base_backbone:
            p.copy_(base_backbone[k])
    model.hybridlink.load_state_dict(base_adapter, strict=True)

    out_path = experiment / "MECHANISM_PROBE.json"
    with out_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)
        stream.write("\n")


class ProbeTests(unittest.TestCase):
    def test_task_vector_formula(self):
        w_orig = 2.0
        w_c2 = 3.0
        tau = 1.1
        self.assertAlmostEqual(w_orig + tau * (w_c2 - w_orig), 3.1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root:
        run_probe(args.root.resolve(strict=True))
    else:
        parser.error("choose --self-test or --root")


if __name__ == "__main__":
    main()
