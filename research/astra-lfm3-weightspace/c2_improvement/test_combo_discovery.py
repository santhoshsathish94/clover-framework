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
from probe_mechanisms import detailed_hybridsft_eval
from search_discovery import fit_lowrank_proj2_delta


def compute_text_hidden_subspace(model, tokenizer, samples: list[dict], k: int = 8):
    import torch
    from hybrid_sft.data import collate_batch

    embed_layer = model.backbone.get_input_embeddings()
    decoder = model.backbone.get_decoder()
    text_h = []
    entry_h = []
    exit_h = []
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
        src = hidden[:, :-1].float()
        tgt = input_ids[:, 1:]
        msk = loss_mask[:, 1:] > 0
        text_mask = (tgt != 125033) & (tgt != 125034) & msk
        entry_mask = (tgt == 125033) & msk
        exit_mask = (tgt == 125034) & msk
        if text_mask.any():
            text_h.append(src[text_mask])
        if entry_mask.any():
            entry_h.append(src[entry_mask])
        if exit_mask.any():
            exit_h.append(src[exit_mask])
    H_text = torch.cat(text_h, dim=0)
    H_entry = torch.cat(entry_h, dim=0)
    H_exit = torch.cat(exit_h, dim=0)
    _, _, vh = torch.linalg.svd(H_text, full_matrices=False)
    v_text = vh[:k, :].T  # [2048, k]

    # Contrastive direction for entry and exit vs text hidden states, projected orthogonal to top-k text subspace
    def ortho_contrast(h_pos):
        raw_dir = h_pos.mean(dim=0) - H_text.mean(dim=0)
        proj = v_text @ (v_text.T @ raw_dir)
        clean = raw_dir - proj
        return clean / (clean.norm() + 1e-12)

    return {
        "v_text": v_text,
        "entry_dir": ortho_contrast(H_entry),
        "exit_dir": ortho_contrast(H_exit),
    }


def make_ortho_task_anchor_rows(emb_c2, emb_orig, v_text, scale_entry: float, scale_exit: float):
    mod_emb = emb_c2.clone()
    for tid, scale in ((125033, scale_entry), (125034, scale_exit)):
        delta = (emb_c2[tid].float() - emb_orig[tid].float())
        # Project out top text hidden subspace so ordinary text logits for 125033/125034 do not inflate softmax denominator
        delta_ortho = delta - v_text @ (v_text.T @ delta)
        mod_emb[tid] = (emb_c2[tid].float() + (scale - 1.0) * delta_ortho).to(mod_emb.dtype)
    return mod_emb


def summarize_deltas(cand_eval: dict, base_eval: dict) -> dict:
    keys = ("loss", "backbone_loss", "text_ce", "entry_ce", "exit_ce", "hybridlink_loss", "cos_loss", "exit_cos_loss", "mse_loss", "exit_mse_loss")
    out = {}
    for k in keys:
        c_vals = [r[k] for r in cand_eval["per_sample"]]
        b_vals = [r[k] for r in base_eval["per_sample"]]
        better = sum(1 for c, b in zip(c_vals, b_vals) if c < b - 1e-7)
        worse = sum(1 for c, b in zip(c_vals, b_vals) if c > b + 1e-7)
        out[k] = {
            "mean": cand_eval["means"][k],
            "delta": cand_eval["means"][k] - base_eval["means"][k],
            "better": better,
            "worse": worse,
        }
    return out


def run_combo_test(root: Path) -> None:
    experiment = setup_env(root)
    import torch
    import torch.nn.functional as F
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

    base_emb = model.backbone.model.embed_tokens.weight.detach().clone()
    base_adapter = {k: v.clone() for k, v in model.hybridlink.state_dict().items()}
    with safe_open(orig_sf_path, framework="pt", device="cpu") as sf:
        orig_emb = sf.get_tensor("model.language_model.embed_tokens.weight")

    base_fit = detailed_hybridsft_eval(model, tokenizer, fit_samples)
    base_val = detailed_hybridsft_eval(model, tokenizer, val_samples)
    print(json.dumps({"split": "base_fit", **base_fit["means"]}), flush=True)
    print(json.dumps({"split": "base_val", **base_val["means"]}), flush=True)

    subspace = compute_text_hidden_subspace(model, tokenizer, fit_samples, k=8)

    # Also fit rank-16 proj2 delta on fit_samples
    from explore_discovery import cache_pass1_trajectories
    cached_fit = cache_pass1_trajectories(model, tokenizer, fit_samples)
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
    dw2_r16 = fit_lowrank_proj2_delta(
        torch.cat(mid_list), torch.cat(h_list), torch.cat(out_list), torch.cat(tgt_list), torch.cat(w_list),
        rank=16, ridge=200.0, eta=0.15
    )

    arms = [
        ("scalar_g0.92_only", 1.0, 1.0, False, 0.92, False),
        ("scalar_g0.88_only", 1.0, 1.0, False, 0.88, False),
        ("raw_anchor1.5_g0.92", 1.5, 1.5, False, 0.92, False),
        ("ortho_anchor1.5_g0.92", 1.5, 1.5, True, 0.92, False),
        ("ortho_anchor1.8_g0.90", 1.8, 1.8, True, 0.90, False),
        ("ortho_anchor1.8_g0.88_lr16", 1.8, 1.8, True, 0.88, True),
        ("ortho_anchor2.0_g0.90_lr16", 2.0, 2.0, True, 0.90, True),
    ]

    fit_out = {}
    val_out = {}
    qualifying = []

    for name, s_ent, s_ext, use_ortho, gamma, use_lr16 in arms:
        if s_ent == 1.0 and s_ext == 1.0:
            model.backbone.model.embed_tokens.weight.copy_(base_emb)
        elif use_ortho:
            mod_emb = make_ortho_task_anchor_rows(base_emb, orig_emb, subspace["v_text"], s_ent, s_ext)
            model.backbone.model.embed_tokens.weight.copy_(mod_emb)
        else:
            mod_emb = base_emb.clone()
            for tid, sc in ((125033, s_ent), (125034, s_ext)):
                d = base_emb[tid].float() - orig_emb[tid].float()
                mod_emb[tid] = (orig_emb[tid].float() + sc * d).to(mod_emb.dtype)
            model.backbone.model.embed_tokens.weight.copy_(mod_emb)

        st = apply_adapter_scales(base_adapter, 1.0, gamma, zero_post_bias=False)
        if use_lr16:
            st["proj2.weight"] = (base_adapter["proj2.weight"].float() + dw2_r16).to(st["proj2.weight"].dtype)
        model.hybridlink.load_state_dict(st, strict=True)

        ev_fit = detailed_hybridsft_eval(model, tokenizer, fit_samples)
        d_fit = summarize_deltas(ev_fit, base_fit)
        fit_out[name] = d_fit
        strict_fit_win = (
            d_fit["loss"]["delta"] < 0.0
            and d_fit["backbone_loss"]["delta"] < 0.0
            and d_fit["hybridlink_loss"]["delta"] < 0.0
            and d_fit["cos_loss"]["delta"] < 0.0
            and d_fit["mse_loss"]["delta"] < 0.0
        )
        print(
            json.dumps(
                {
                    "fit_arm": name,
                    "strict_win": strict_fit_win,
                    "d_loss": round(d_fit["loss"]["delta"], 5),
                    "d_bb": round(d_fit["backbone_loss"]["delta"], 5),
                    "d_text_ce": round(d_fit["text_ce"]["delta"], 5),
                    "d_entry_ce": round(d_fit["entry_ce"]["delta"], 5),
                    "d_exit_ce": round(d_fit["exit_ce"]["delta"], 5),
                    "d_hl": round(d_fit["hybridlink_loss"]["delta"], 5),
                    "d_cos": round(d_fit["cos_loss"]["delta"], 5),
                    "d_mse": round(d_fit["mse_loss"]["delta"], 5),
                }
            ),
            flush=True,
        )
        if strict_fit_win:
            qualifying.append(name)
            ev_val = detailed_hybridsft_eval(model, tokenizer, val_samples)
            d_val = summarize_deltas(ev_val, base_val)
            val_out[name] = d_val
            print(
                json.dumps(
                    {
                        "val_arm": name,
                        "d_loss": round(d_val["loss"]["delta"], 5),
                        "d_bb": round(d_val["backbone_loss"]["delta"], 5),
                        "d_text_ce": round(d_val["text_ce"]["delta"], 5),
                        "d_entry_ce": round(d_val["entry_ce"]["delta"], 5),
                        "d_exit_ce": round(d_val["exit_ce"]["delta"], 5),
                        "d_hl": round(d_val["hybridlink_loss"]["delta"], 5),
                        "d_cos": round(d_val["cos_loss"]["delta"], 5),
                        "d_mse": round(d_val["mse_loss"]["delta"], 5),
                        "loss_bw": [d_val["loss"]["better"], d_val["loss"]["worse"]],
                        "bb_bw": [d_val["backbone_loss"]["better"], d_val["backbone_loss"]["worse"]],
                        "hl_bw": [d_val["hybridlink_loss"]["better"], d_val["hybridlink_loss"]["worse"]],
                    }
                ),
                flush=True,
            )

    model.backbone.model.embed_tokens.weight.copy_(base_emb)
    model.hybridlink.load_state_dict(base_adapter, strict=True)

    with (experiment / "COMBO_DISCOVERY.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {"fit": fit_out, "qualifying": qualifying, "val": val_out},
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


class ComboTests(unittest.TestCase):
    def test_summarize_deltas(self):
        keys = ("loss", "backbone_loss", "text_ce", "entry_ce", "exit_ce", "hybridlink_loss", "cos_loss", "exit_cos_loss", "mse_loss", "exit_mse_loss")
        b = {"means": {k: 1.0 for k in keys}, "per_sample": [{k: 1.0 for k in keys}]}
        c = {"means": {k: 0.9 for k in keys}, "per_sample": [{k: 0.9 for k in keys}]}
        d = summarize_deltas(c, b)
        self.assertAlmostEqual(d["loss"]["delta"], -0.1)
        self.assertEqual(d["loss"]["better"], 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root:
        run_combo_test(args.root.resolve(strict=True))
    else:
        parser.error("choose --self-test or --root")


if __name__ == "__main__":
    main()
