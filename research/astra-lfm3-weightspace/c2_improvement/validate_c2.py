from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import unittest
from pathlib import Path

from diagnose import file_hash, write_json
from explore_discovery import (
    apply_adapter_scales,
    build_hybridsft_samples,
    setup_env,
)


ARMS = (
    "original",
    "noop_control",
    "anchor1.5_only",
    "g0.92_only",
    "combined_anchor1.5_g0.92",
)


def one_sided_sign_test(candidate_vals: list[float], baseline_vals: list[float], eps: float = 1e-7) -> dict:
    if len(candidate_vals) != len(baseline_vals):
        raise ValueError("paired lists must have equal length")
    better = sum(1 for c, b in zip(candidate_vals, baseline_vals) if c < b - eps)
    worse = sum(1 for c, b in zip(candidate_vals, baseline_vals) if c > b + eps)
    n = better + worse
    p_val = sum(math.comb(n, k) for k in range(better, n + 1)) / (2**n) if n > 0 else 1.0
    return {
        "better": better,
        "worse": worse,
        "ties": len(candidate_vals) - n,
        "one_sided_p": float(p_val),
    }


def extract_concise_answer(text: str) -> str:
    import re

    cleaned = text.split("<|im_end|>")[0].strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    m_box = re.findall(r"\\boxed\{([^{}]+)\}", cleaned)
    if m_box:
        return m_box[-1].strip()
    m_final = re.findall(r"(?:final answer\s*[:=-]\s*)([^\n*]+)", cleaned, flags=re.IGNORECASE)
    if m_final:
        return m_final[-1].strip().rstrip(".")
    lines = [ln.strip().strip("*") for ln in cleaned.splitlines() if ln.strip()]
    last = lines[-1] if lines else cleaned
    m_is = re.search(r"(?:is|equals|=|remain|hold|have)\s+([A-Za-z0-9.,+-]+)\.?$", last, flags=re.IGNORECASE)
    if m_is:
        return m_is.group(1).strip()
    return last


def score_qa_output(raw_output: str, gold_list: list[str], normalize_fn) -> tuple[bool, str]:
    extracted = extract_concise_answer(raw_output)
    norm_ext = normalize_fn(extracted)
    norm_full = normalize_fn(raw_output.split("</think>")[-1] if "</think>" in raw_output else raw_output)
    gold_norms = {normalize_fn(g) for g in gold_list}
    if norm_ext in gold_norms or norm_full in gold_norms:
        return True, extracted
    # Also check exact word/number token match in the post-think answer
    import re
    post_think = (raw_output.split("</think>")[-1] if "</think>" in raw_output else raw_output).split("<|im_end|>")[0]
    for g in gold_list:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(g.strip()) + r"(?![A-Za-z0-9])"
        if re.search(pattern, post_think, flags=re.IGNORECASE):
            return True, extracted
    return False, extracted


def masked_ce_breakdown(output_head, hidden, input_ids, loss_mask, chunk_size: int = 512) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    src = hidden[:, :-1]
    tgt = input_ids[:, 1:]
    msk = loss_mask[:, 1:].float()
    active = msk > 0
    if not active.any():
        return {"backbone_loss": 0.0, "text_ce": 0.0, "entry_ce": 0.0, "exit_ce": 0.0}

    src_active = src[active]
    tgt_active = tgt[active]
    msk_active = msk[active]

    ce_parts = []
    for start in range(0, src_active.shape[0], chunk_size):
        end = min(start + chunk_size, src_active.shape[0])
        logits = output_head(src_active[start:end]).float()
        ce = F.cross_entropy(logits, tgt_active[start:end], reduction="none")
        ce_parts.append(ce)
    token_ce = torch.cat(ce_parts, dim=0)

    backbone_loss = float(((token_ce * msk_active).sum() / msk_active.sum().clamp(min=1.0)).item())
    entry_mask = tgt_active == 125033
    exit_mask = tgt_active == 125034
    text_mask = (~entry_mask) & (~exit_mask)

    return {
        "backbone_loss": backbone_loss,
        "text_ce": float(token_ce[text_mask].mean().item()) if text_mask.any() else 0.0,
        "entry_ce": float(token_ce[entry_mask].mean().item()) if entry_mask.any() else 0.0,
        "exit_ce": float(token_ce[exit_mask].mean().item()) if exit_mask.any() else 0.0,
    }


def evaluate_cached_detailed(model, cached_samples: list[dict]) -> dict:
    import torch
    import torch.nn.functional as F
    from hybrid_sft.model import EXIT_ANCHOR_HYBRIDLINK_WEIGHT, LAMBDA_COS, LAMBDA_MSE

    decoder = model.backbone.get_decoder()
    output_head = model.backbone.get_output_embeddings()
    rows = []
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

            ce_info = masked_ce_breakdown(output_head, hidden, input_ids, loss_mask)
            backbone_loss = ce_info["backbone_loss"]

            latent_src_mask = latent_mask[:, :-1] * attention_mask[:, :-1]
            next_is_latent = latent_mask[:, 1:]
            exit_mask = latent_src_mask * (1.0 - next_is_latent)

            preds = model.run_hybridlink(hidden[:, :-1], output_dtype=teacher_embeds.dtype)
            preds_f = preds.float()
            targets_f = teacher_embeds[:, 1:].float()
            cosine = 1.0 - F.cosine_similarity(preds_f, targets_f, dim=-1)
            mse = (preds_f - targets_f).pow(2).mean(dim=-1)

            cos_loss = float(model._masked_mean(cosine, latent_src_mask).item())
            mse_loss = float(model._masked_mean(mse, latent_src_mask).item())
            exit_cos = float(model._masked_mean(cosine, exit_mask).item())
            exit_mse = float(model._masked_mean(mse, exit_mask).item())
            hl_loss = (LAMBDA_COS * cos_loss + LAMBDA_MSE * mse_loss) + EXIT_ANCHOR_HYBRIDLINK_WEIGHT * (
                LAMBDA_COS * exit_cos + LAMBDA_MSE * exit_mse
            )
            total_loss = backbone_loss + hl_loss

        rows.append(
            {
                "id": item["id"],
                "n_tokens": item["n_tokens"],
                "loss": total_loss,
                "backbone_loss": backbone_loss,
                "text_ce": ce_info["text_ce"],
                "entry_ce": ce_info["entry_ce"],
                "exit_ce": ce_info["exit_ce"],
                "hybridlink_loss": hl_loss,
                "cos_loss": cos_loss,
                "exit_cos_loss": exit_cos,
                "mse_loss": mse_loss,
                "exit_mse_loss": exit_mse,
            }
        )
    keys = (
        "loss",
        "backbone_loss",
        "text_ce",
        "entry_ce",
        "exit_ce",
        "hybridlink_loss",
        "cos_loss",
        "exit_cos_loss",
        "mse_loss",
        "exit_mse_loss",
    )
    means = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    return {"means": means, "count": len(rows), "per_sample": rows}


def run_qa_and_latent_rollout(model, tokenizer, questions: list[dict], normalize_fn) -> dict:
    import torch

    embed_layer = model.backbone.get_input_embeddings()
    system = "Solve the problem carefully and state the final answer clearly."
    natural_rows = []
    latent4_rows = []

    for q in questions:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": q["text"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prompt.endswith("<think>"):
            prompt += "\n"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([prompt_ids], dtype=torch.long)

        # 1. Natural greedy generation
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            gen_ids = model.backbone.generate(
                input_ids=inp,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = gen_ids[0, len(prompt_ids) :].tolist()
        raw_text = tokenizer.decode(new_ids, skip_special_tokens=False)
        ok, extracted = score_qa_output(raw_text, q["gold"], normalize_fn)
        natural_rows.append(
            {
                "id": q["id"],
                "question": q["text"],
                "gold": q["gold"],
                "output": raw_text,
                "extracted": extracted,
                "correct": ok,
            }
        )

        # 2. 4-step latent-span rollout (<latent> -> 4 HybridLink steps -> </latent> -> greedy text answer)
        open_ids = prompt_ids + [125033]
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            embeds = embed_layer(torch.tensor([open_ids], dtype=torch.long))
            for _ in range(4):
                out_dec = model.backbone.get_decoder()(inputs_embeds=embeds, use_cache=False, return_dict=True)
                h_last = out_dec.last_hidden_state[:, -1:, :]
                r_next = model.run_hybridlink(h_last, output_dtype=embeds.dtype)
                embeds = torch.cat([embeds, r_next], dim=1)
            # Check exit-anchor readout rank at step 4
            out_dec = model.backbone.get_decoder()(inputs_embeds=embeds, use_cache=False, return_dict=True)
            exit_logits = model.backbone.get_output_embeddings()(out_dec.last_hidden_state[:, -1, :]).float()[0]
            exit_logit_val = float(exit_logits[125034].item())
            exit_rank = int((exit_logits > exit_logits[125034]).sum().item()) + 1

            # Append </latent> embedding and decode up to 48 text tokens greedily using KV cache
            close_emb = embed_layer(torch.tensor([[125034]], dtype=torch.long))
            cur_embeds = torch.cat([embeds, close_emb], dim=1)
            prefill = model.backbone(inputs_embeds=cur_embeds, use_cache=True, return_dict=True, logits_to_keep=1)
            past_kv = prefill.past_key_values
            next_logits = prefill.logits[:, -1, :]
            cur_len = cur_embeds.shape[1]
            attn_mask = torch.ones((1, cur_len), dtype=torch.long)
            lat_gen_ids = []
            for _ in range(48):
                next_tid = int(next_logits[0].argmax().item())
                lat_gen_ids.append(next_tid)
                if next_tid == tokenizer.eos_token_id:
                    break
                next_emb = embed_layer(torch.tensor([[next_tid]], dtype=torch.long))
                attn_mask = torch.cat([attn_mask, torch.ones((1, 1), dtype=torch.long)], dim=1)
                pos_ids = torch.tensor([[cur_len]], dtype=torch.long)
                cache_pos = torch.tensor([cur_len], dtype=torch.long)
                step_out = model.backbone(
                    inputs_embeds=next_emb,
                    past_key_values=past_kv,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    cache_position=cache_pos,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
                past_kv = step_out.past_key_values
                next_logits = step_out.logits[:, -1, :]
                cur_len += 1

        lat_text = tokenizer.decode(lat_gen_ids, skip_special_tokens=False)
        lat_ok, lat_extracted = score_qa_output(lat_text, q["gold"], normalize_fn)
        latent4_rows.append(
            {
                "id": q["id"],
                "question": q["text"],
                "gold": q["gold"],
                "output": lat_text,
                "extracted": lat_extracted,
                "correct": lat_ok,
                "exit_anchor_logit": exit_logit_val,
                "exit_anchor_rank": exit_rank,
            }
        )

    return {
        "natural_correct": sum(1 for r in natural_rows if r["correct"]),
        "natural_total": len(natural_rows),
        "latent4_correct": sum(1 for r in latent4_rows if r["correct"]),
        "latent4_total": len(latent4_rows),
        "mean_exit_anchor_logit": sum(r["exit_anchor_logit"] for r in latent4_rows) / len(latent4_rows),
        "median_exit_anchor_rank": sorted(r["exit_anchor_rank"] for r in latent4_rows)[len(latent4_rows) // 2],
        "natural_rows": natural_rows,
        "latent4_rows": latent4_rows,
    }


def freeze_validation(root: Path) -> None:
    experiment = root / "c2_improvement"
    frozen = experiment / "frozen"
    frozen.mkdir(exist_ok=False)

    sys.path.insert(0, str(root / "composite"))
    from evaluate import diagnostic_cases

    questions = [c for c in diagnostic_cases() if "gold" in c]
    write_json(frozen / "qa_cases.json", questions)
    protocol = {
        "arms": list(ARMS),
        "primary_candidate": "combined_anchor1.5_g0.92",
        "single_component_ablations": ["anchor1.5_only", "g0.92_only"],
        "mathematical_specification": {
            "anchor1.5_only": "model.embed_tokens.weight[a] <- e_orig[a] + 1.5 * (e_c2[a] - e_orig[a]) for a in {125033, 125034}",
            "g0.92_only": "adapter.post_ln.{weight,bias} <- 0.92 * adapter.post_ln.{weight,bias}",
            "combined_anchor1.5_g0.92": "both anchor1.5_only and g0.92_only applied together",
        },
        "heldout_splits": ["eval32", "heldout32"],
        "max_length": 8192,
        "acceptance_gate": {
            "loss_delta": "< 0 on both eval32 and heldout32 with one-sided paired sign p <= 0.01",
            "backbone_loss_delta": "< 0 on both eval32 and heldout32 with one-sided paired sign p <= 0.01",
            "hybridlink_loss_delta": "< 0 on both eval32 and heldout32 with one-sided paired sign p <= 0.01",
            "cos_loss_delta": "< 0 on both eval32 and heldout32",
            "mse_loss_delta": "< 0 on both eval32 and heldout32",
            "text_ce_delta": "<= 0.0005 (non-degraded ordinary text CE)",
            "qa_natural_correct": ">= original with 0 newly broken questions",
            "qa_latent4_correct": ">= original",
        },
        "gradient_training": False,
    }
    write_json(frozen / "PROTOCOL.json", protocol)

    tracked = [
        experiment / "diagnose.py",
        experiment / "explore_discovery.py",
        experiment / "search_discovery.py",
        experiment / "probe_mechanisms.py",
        experiment / "test_combo_discovery.py",
        experiment / "validate_c2.py",
        experiment / "SPLITS.json",
        experiment / "splits" / "eval32.jsonl",
        experiment / "splits" / "discovery24.jsonl",
        experiment / "splits" / "heldout32.jsonl",
        frozen / "PROTOCOL.json",
        frozen / "qa_cases.json",
    ]
    write_json(
        frozen / "MANIFEST.json",
        {
            "files": {str(p.relative_to(root)): file_hash(p) for p in tracked},
            "c2_model_sha256": file_hash(root / "c2c3_weights" / "c2" / "model.safetensors"),
            "c2_adapter_sha256": file_hash(root / "c2c3_weights" / "c2" / "adapter.pt"),
            "orig_model_sha256": file_hash(root / "original" / "model.safetensors"),
        },
    )
    print(json.dumps({"frozen": True, "qa_cases": len(questions), "arms": list(ARMS)}), flush=True)


def verify_frozen_state(root: Path) -> tuple[dict, list[dict]]:
    experiment = root / "c2_improvement"
    frozen = experiment / "frozen"
    manifest = json.loads((frozen / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        if file_hash(root / rel) != expected:
            raise ValueError(f"frozen file modified: {rel}")
    if file_hash(root / "c2c3_weights" / "c2" / "model.safetensors") != manifest["c2_model_sha256"]:
        raise ValueError("C2 model checkpoint modified")
    if file_hash(root / "c2c3_weights" / "c2" / "adapter.pt") != manifest["c2_adapter_sha256"]:
        raise ValueError("C2 adapter checkpoint modified")
    if file_hash(root / "original" / "model.safetensors") != manifest["orig_model_sha256"]:
        raise ValueError("original Omni checkpoint modified")
    protocol = json.loads((frozen / "PROTOCOL.json").read_text(encoding="utf-8"))
    qa_cases = json.loads((frozen / "qa_cases.json").read_text(encoding="utf-8"))
    return protocol, qa_cases


def create_and_audit_physical_copies(root: Path, base_emb, orig_emb, base_adapter) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    sys.path.insert(0, str(root / "composite"))
    from prepare import physical_copy, tensor_digest

    experiment = root / "c2_improvement"
    copies_dir = experiment / "copies"
    copies_dir.mkdir(exist_ok=False)

    c2_dir = root / "c2c3_weights" / "c2"
    src_adapter = c2_dir / "adapter.pt"
    src_model = c2_dir / "model.safetensors"

    # Build modified 2-row anchor embedding tensor
    anchor_emb = base_emb.clone()
    for tid in (125033, 125034):
        delta = base_emb[tid].float() - orig_emb[tid].float()
        anchor_emb[tid] = (orig_emb[tid].float() + 1.5 * delta).to(anchor_emb.dtype)

    changed_rows = int(((anchor_emb - base_emb).abs().max(dim=1).values > 0).sum().item())
    if changed_rows != 2:
        raise ValueError(f"expected exactly 2 changed embedding rows, got {changed_rows}")

    # Materialize physical copies for adapter and for the edited model.safetensors
    audits = {}

    # 1. Physical model.safetensors copy for anchor1.5
    model_copy_dir = copies_dir / "anchor1.5_backbone"
    model_copy_dir.mkdir()
    dst_model = model_copy_dir / "model.safetensors"
    copy_rec = physical_copy(src_model, dst_model, experiment)

    tensors = {}
    before_hashes = {}
    with safe_open(src_model, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            t = sf.get_tensor(k)
            before_hashes[k] = tensor_digest(t)
            tensors[k] = anchor_emb if k == "model.embed_tokens.weight" else t

    tmp_model = model_copy_dir / "tmp_model.safetensors"
    save_file(tensors, str(tmp_model))
    tmp_model.replace(dst_model)

    after_hashes = {}
    with safe_open(dst_model, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            after_hashes[k] = tensor_digest(sf.get_tensor(k))
        loaded_emb = sf.get_tensor("model.embed_tokens.weight")
        if not torch.equal(loaded_emb, anchor_emb):
            raise ValueError("saved anchor1.5 embedding tensor does not match")

    changed_model_keys = [k for k in before_hashes if before_hashes[k] != after_hashes.get(k)]
    if changed_model_keys != ["model.embed_tokens.weight"]:
        raise ValueError(f"unexpected changed model keys: {changed_model_keys}")

    audits["anchor1.5_backbone"] = {
        "copy": copy_rec,
        "changed_tensors": changed_model_keys,
        "changed_embedding_rows": [125033, 125034],
        "unchanged_embedding_rows": 128000 - 2,
        "before_embed_sha256": before_hashes["model.embed_tokens.weight"],
        "after_embed_sha256": after_hashes["model.embed_tokens.weight"],
        "edited_file_sha256": file_hash(dst_model),
    }
    write_json(model_copy_dir / "PATCH_AUDIT.json", audits["anchor1.5_backbone"])

    # 2. Physical adapter.pt copies for noop_control (1.0) and g0.92 (0.92)
    before_adapter_hashes = {k: tensor_digest(v) for k, v in base_adapter.items()}
    for name, gamma, expected_changed in (
        ("noop_adapter", 1.0, []),
        ("g0.92_adapter", 0.92, ["post_ln.weight", "post_ln.bias"]),
    ):
        adir = copies_dir / name
        adir.mkdir()
        dst_ad = adir / "adapter.pt"
        crec = physical_copy(src_adapter, dst_ad, experiment)
        if gamma != 1.0:
            st = apply_adapter_scales(base_adapter, 1.0, gamma, zero_post_bias=False)
            tmp_ad = adir / "tmp_adapter.pt"
            torch.save(st, tmp_ad)
            tmp_ad.replace(dst_ad)
        loaded_ad = torch.load(dst_ad, map_location="cpu", weights_only=True)
        after_ad_hashes = {k: tensor_digest(v) for k, v in loaded_ad.items()}
        changed_ad = [k for k in before_adapter_hashes if before_adapter_hashes[k] != after_ad_hashes.get(k)]
        if sorted(changed_ad) != sorted(expected_changed):
            raise ValueError(f"unexpected adapter changes for {name}: {changed_ad}")
        audits[name] = {
            "copy": crec,
            "gamma": gamma,
            "changed_tensors": changed_ad,
            "before": before_adapter_hashes,
            "after": after_ad_hashes,
            "edited_file_sha256": file_hash(dst_ad),
        }
        write_json(adir / "PATCH_AUDIT.json", audits[name])

    return audits


def run_validation(root: Path) -> None:
    experiment = setup_env(root)
    protocol, qa_cases = verify_frozen_state(root)

    import torch
    from safetensors import safe_open
    from hybrid_agent.model import add_anchor_tokens
    from hybrid_sft.model import HybridSFTModel
    from transformers import AutoTokenizer
    from explore_discovery import cache_pass1_trajectories
    from probe_mechanisms import detailed_hybridsft_eval

    sys.path.insert(0, str(root / "composite"))
    from evaluate import normalize_answer

    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    results_dir = experiment / "results"
    results_dir.mkdir(exist_ok=False)

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

    base_emb = model.backbone.model.embed_tokens.weight.detach().clone()
    base_adapter = {k: v.clone() for k, v in model.hybridlink.state_dict().items()}
    with safe_open(orig_sf_path, framework="pt", device="cpu") as sf:
        orig_emb = sf.get_tensor("model.language_model.embed_tokens.weight")

    audits = create_and_audit_physical_copies(root, base_emb, orig_emb, base_adapter)
    copies_dir = experiment / "copies"

    # Load physical copies from disk to guarantee we test the actual serialized files
    with safe_open(copies_dir / "anchor1.5_backbone" / "model.safetensors", framework="pt", device="cpu") as sf:
        disk_anchor_emb = sf.get_tensor("model.embed_tokens.weight")
    disk_noop_adapter = torch.load(copies_dir / "noop_adapter" / "adapter.pt", map_location="cpu", weights_only=True)
    disk_g092_adapter = torch.load(copies_dir / "g0.92_adapter" / "adapter.pt", map_location="cpu", weights_only=True)

    eval32_samples = build_hybridsft_samples(tokenizer, experiment / "splits" / "eval32.jsonl", max_length=8192)
    heldout32_samples = build_hybridsft_samples(tokenizer, experiment / "splits" / "heldout32.jsonl", max_length=8192)

    # Verify masked_ce_breakdown and cached Pass 1 against full detailed_hybridsft_eval on first sample
    full_check = detailed_hybridsft_eval(model, tokenizer, eval32_samples[:1])["means"]
    cached_check = evaluate_cached_detailed(model, cache_pass1_trajectories(model, tokenizer, eval32_samples[:1]))["means"]
    for k in full_check:
        if abs(full_check[k] - cached_check[k]) > 1e-5:
            raise ValueError(f"fast evaluator mismatch on {k}: {full_check[k]} vs {cached_check[k]}")

    all_split_results = {}
    for split_name, samples in (("eval32", eval32_samples), ("heldout32", heldout32_samples)):
        # Pass-1 cache for base embedding
        model.backbone.model.embed_tokens.weight.copy_(base_emb)
        cached_base = cache_pass1_trajectories(model, tokenizer, samples)

        # Pass-1 cache for anchor1.5 embedding (loaded from physical copy)
        model.backbone.model.embed_tokens.weight.copy_(disk_anchor_emb)
        cached_anchor = cache_pass1_trajectories(model, tokenizer, samples)

        split_out = {}
        for arm in ARMS:
            if arm in ("original", "noop_control", "g0.92_only"):
                model.backbone.model.embed_tokens.weight.copy_(base_emb)
                cached = cached_base
            else:
                model.backbone.model.embed_tokens.weight.copy_(disk_anchor_emb)
                cached = cached_anchor

            if arm == "original" or arm == "anchor1.5_only":
                model.hybridlink.load_state_dict(base_adapter, strict=True)
            elif arm == "noop_control":
                model.hybridlink.load_state_dict(disk_noop_adapter, strict=True)
            else:
                model.hybridlink.load_state_dict(disk_g092_adapter, strict=True)

            res = evaluate_cached_detailed(model, cached)
            split_out[arm] = res
            write_json(results_dir / f"{split_name}-{arm}.json", res)
            print(json.dumps({"split": split_name, "arm": arm, **res["means"]}), flush=True)

        # Verify noop_control == original bit-for-bit
        for s_orig, s_noop in zip(split_out["original"]["per_sample"], split_out["noop_control"]["per_sample"]):
            for k in ("loss", "backbone_loss", "hybridlink_loss", "cos_loss", "mse_loss"):
                if s_orig[k] != s_noop[k]:
                    raise ValueError(f"noop_control differed from original on {split_name} {k}")

        all_split_results[split_name] = split_out

    # Downstream QA & 4-step latent rollout check on original vs combined_anchor1.5_g0.92
    qa_results = {}
    for arm in ("original", "combined_anchor1.5_g0.92"):
        if arm == "original":
            model.backbone.model.embed_tokens.weight.copy_(base_emb)
            model.hybridlink.load_state_dict(base_adapter, strict=True)
        else:
            model.backbone.model.embed_tokens.weight.copy_(disk_anchor_emb)
            model.hybridlink.load_state_dict(disk_g092_adapter, strict=True)
        qa_res = run_qa_and_latent_rollout(model, tokenizer, qa_cases, normalize_answer)
        qa_results[arm] = qa_res
        write_json(results_dir / f"qa-{arm}.json", qa_res)
        print(
            json.dumps(
                {
                    "qa_arm": arm,
                    "natural_correct": qa_res["natural_correct"],
                    "latent4_correct": qa_res["latent4_correct"],
                    "mean_exit_logit": round(qa_res["mean_exit_anchor_logit"], 4),
                    "median_exit_rank": qa_res["median_exit_anchor_rank"],
                }
            ),
            flush=True,
        )

    # Restore model in memory
    model.backbone.model.embed_tokens.weight.copy_(base_emb)
    model.hybridlink.load_state_dict(base_adapter, strict=True)

    # Compute paired comparisons and check predeclared acceptance gate
    comparisons = {}
    keys = ("loss", "backbone_loss", "text_ce", "entry_ce", "exit_ce", "hybridlink_loss", "cos_loss", "exit_cos_loss", "mse_loss", "exit_mse_loss")
    for split_name, split_out in all_split_results.items():
        base_rows = split_out["original"]["per_sample"]
        comparisons[split_name] = {}
        for arm in ("anchor1.5_only", "g0.92_only", "combined_anchor1.5_g0.92"):
            cand_rows = split_out[arm]["per_sample"]
            arm_cmp = {}
            for k in keys:
                c_vals = [r[k] for r in cand_rows]
                b_vals = [r[k] for r in base_rows]
                st = one_sided_sign_test(c_vals, b_vals)
                st["mean_baseline"] = split_out["original"]["means"][k]
                st["mean_candidate"] = split_out[arm]["means"][k]
                st["mean_delta"] = split_out[arm]["means"][k] - split_out["original"]["means"][k]
                arm_cmp[k] = st
            comparisons[split_name][arm] = arm_cmp

    prim_eval32 = comparisons["eval32"]["combined_anchor1.5_g0.92"]
    prim_held32 = comparisons["heldout32"]["combined_anchor1.5_g0.92"]

    newly_broken_natural = sum(
        1
        for r_o, r_c in zip(qa_results["original"]["natural_rows"], qa_results["combined_anchor1.5_g0.92"]["natural_rows"])
        if r_o["correct"] and not r_c["correct"]
    )
    newly_fixed_natural = sum(
        1
        for r_o, r_c in zip(qa_results["original"]["natural_rows"], qa_results["combined_anchor1.5_g0.92"]["natural_rows"])
        if not r_o["correct"] and r_c["correct"]
    )
    newly_broken_latent4 = sum(
        1
        for r_o, r_c in zip(qa_results["original"]["latent4_rows"], qa_results["combined_anchor1.5_g0.92"]["latent4_rows"])
        if r_o["correct"] and not r_c["correct"]
    )
    newly_fixed_latent4 = sum(
        1
        for r_o, r_c in zip(qa_results["original"]["latent4_rows"], qa_results["combined_anchor1.5_g0.92"]["latent4_rows"])
        if not r_o["correct"] and r_c["correct"]
    )

    gate_checks = {
        "eval32_loss_improved_and_p_le_0.01": prim_eval32["loss"]["mean_delta"] < 0 and prim_eval32["loss"]["one_sided_p"] <= 0.01,
        "eval32_backbone_improved_and_p_le_0.01": prim_eval32["backbone_loss"]["mean_delta"] < 0 and prim_eval32["backbone_loss"]["one_sided_p"] <= 0.01,
        "eval32_hybridlink_improved_and_p_le_0.01": prim_eval32["hybridlink_loss"]["mean_delta"] < 0 and prim_eval32["hybridlink_loss"]["one_sided_p"] <= 0.01,
        "eval32_cos_and_mse_improved": prim_eval32["cos_loss"]["mean_delta"] < 0 and prim_eval32["mse_loss"]["mean_delta"] < 0,
        "eval32_text_ce_non_degraded": prim_eval32["text_ce"]["mean_delta"] <= 0.0005,
        "heldout32_loss_improved_and_p_le_0.01": prim_held32["loss"]["mean_delta"] < 0 and prim_held32["loss"]["one_sided_p"] <= 0.01,
        "heldout32_backbone_improved_and_p_le_0.01": prim_held32["backbone_loss"]["mean_delta"] < 0 and prim_held32["backbone_loss"]["one_sided_p"] <= 0.01,
        "heldout32_hybridlink_improved_and_p_le_0.01": prim_held32["hybridlink_loss"]["mean_delta"] < 0 and prim_held32["hybridlink_loss"]["one_sided_p"] <= 0.01,
        "heldout32_cos_and_mse_improved": prim_held32["cos_loss"]["mean_delta"] < 0 and prim_held32["mse_loss"]["mean_delta"] < 0,
        "heldout32_text_ce_non_degraded": prim_held32["text_ce"]["mean_delta"] <= 0.0005,
        "qa_natural_non_degraded": qa_results["combined_anchor1.5_g0.92"]["natural_correct"] >= qa_results["original"]["natural_correct"] and newly_broken_natural == 0,
        "qa_latent4_non_degraded": qa_results["combined_anchor1.5_g0.92"]["latent4_correct"] >= qa_results["original"]["latent4_correct"],
    }
    all_passed = all(gate_checks.values())

    verify_frozen_state(root)

    summary = {
        "primary_candidate": "combined_anchor1.5_g0.92",
        "all_gates_passed": all_passed,
        "gate_checks": gate_checks,
        "comparisons": comparisons,
        "qa_summary": {
            "original_natural": f"{qa_results['original']['natural_correct']}/{qa_results['original']['natural_total']}",
            "candidate_natural": f"{qa_results['combined_anchor1.5_g0.92']['natural_correct']}/{qa_results['combined_anchor1.5_g0.92']['natural_total']}",
            "newly_fixed_natural": newly_fixed_natural,
            "newly_broken_natural": newly_broken_natural,
            "original_latent4": f"{qa_results['original']['latent4_correct']}/{qa_results['original']['latent4_total']}",
            "candidate_latent4": f"{qa_results['combined_anchor1.5_g0.92']['latent4_correct']}/{qa_results['combined_anchor1.5_g0.92']['latent4_total']}",
            "newly_fixed_latent4": newly_fixed_latent4,
            "newly_broken_latent4": newly_broken_latent4,
            "original_mean_exit_logit": qa_results["original"]["mean_exit_anchor_logit"],
            "candidate_mean_exit_logit": qa_results["combined_anchor1.5_g0.92"]["mean_exit_anchor_logit"],
            "original_median_exit_rank": qa_results["original"]["median_exit_anchor_rank"],
            "candidate_median_exit_rank": qa_results["combined_anchor1.5_g0.92"]["median_exit_anchor_rank"],
        },
        "audits": audits,
        "noop_exact_parity": True,
        "sources_unchanged": True,
    }
    write_json(results_dir / "SUMMARY.json", summary)
    print("C2_VALIDATION_COMPLETE " + json.dumps({"all_gates_passed": all_passed, "gate_checks": gate_checks}), flush=True)


class ValidateTests(unittest.TestCase):
    def test_sign_test(self):
        res = one_sided_sign_test([0.1] * 10, [0.2] * 10)
        self.assertEqual(res["better"], 10)
        self.assertEqual(res["worse"], 0)
        self.assertAlmostEqual(res["one_sided_p"], 1.0 / 1024.0)

    def test_extract_and_score_qa(self):
        norm = lambda s: s.lower().strip()
        ok, ext = score_qa_output("<think>\nLet us add 14 and 27.\n</think>The sum is 41.\n**Final answer: 41**<|im_end|>", ["41"], norm)
        self.assertTrue(ok)
        self.assertEqual(ext, "41")
        ok_bad, _ = score_qa_output("<think>\n...\n</think>Final answer: 42", ["41"], norm)
        self.assertFalse(ok_bad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root and args.freeze:
        freeze_validation(args.root.resolve(strict=True))
    elif args.root and args.run:
        run_validation(args.root.resolve(strict=True))
    else:
        parser.error("choose --self-test or --root with --freeze/--run")


if __name__ == "__main__":
    main()
