from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import unittest
from pathlib import Path

from diagnose import file_hash, write_json
from explore_discovery import setup_env
from validate_c2 import (
    ARMS,
    one_sided_sign_test,
    run_qa_and_latent_rollout,
    verify_frozen_state,
)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_close(actual: float, expected: float, label: str, tol: float = 1e-9) -> None:
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise ValueError(f"mismatch on {label}: {actual} vs {expected}")


def run_ablation_qa_and_audit(root: Path, restore_copies: bool = True) -> dict:
    experiment = setup_env(root)
    protocol, qa_cases = verify_frozen_state(root)

    import torch
    from safetensors import safe_open
    from hybrid_agent.model import add_anchor_tokens
    from hybrid_sft.model import HybridSFTModel
    from transformers import AutoTokenizer

    sys.path.insert(0, str(root / "composite"))
    from evaluate import normalize_answer
    from prepare import tensor_digest

    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    results_dir = experiment / "results"
    copies_dir = experiment / "copies"
    c2_dir = root / "c2c3_weights" / "c2"

    # Run QA and 4-step latent rollout on the two single-component ablations if not already saved
    qa_g092_path = results_dir / "qa-g0.92_only.json"
    qa_anc_path = results_dir / "qa-anchor1.5_only.json"
    if not (qa_g092_path.exists() and qa_anc_path.exists()):
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

        with safe_open(copies_dir / "anchor1.5_backbone" / "model.safetensors", framework="pt", device="cpu") as sf:
            disk_anchor_emb = sf.get_tensor("model.embed_tokens.weight")
        disk_g092_adapter = torch.load(copies_dir / "g0.92_adapter" / "adapter.pt", map_location="cpu", weights_only=True)

        if not qa_g092_path.exists():
            model.backbone.model.embed_tokens.weight.copy_(base_emb)
            model.hybridlink.load_state_dict(disk_g092_adapter, strict=True)
            res_g092 = run_qa_and_latent_rollout(model, tokenizer, qa_cases, normalize_answer)
            write_json(qa_g092_path, res_g092)
            print(json.dumps({"qa_arm": "g0.92_only", "natural_correct": res_g092["natural_correct"], "latent4_correct": res_g092["latent4_correct"], "mean_exit_logit": round(res_g092["mean_exit_anchor_logit"], 4)}), flush=True)

        if not qa_anc_path.exists():
            model.backbone.model.embed_tokens.weight.copy_(disk_anchor_emb)
            model.hybridlink.load_state_dict(base_adapter, strict=True)
            res_anc = run_qa_and_latent_rollout(model, tokenizer, qa_cases, normalize_answer)
            write_json(qa_anc_path, res_anc)
            print(json.dumps({"qa_arm": "anchor1.5_only", "natural_correct": res_anc["natural_correct"], "latent4_correct": res_anc["latent4_correct"], "mean_exit_logit": round(res_anc["mean_exit_anchor_logit"], 4)}), flush=True)

    summary = read_json(results_dir / "SUMMARY.json")
    keys = ("loss", "backbone_loss", "text_ce", "entry_ce", "exit_ce", "hybridlink_loss", "cos_loss", "exit_cos_loss", "mse_loss", "exit_mse_loss")

    # Audit all trajectory splits and arms
    for split_name in ("eval32", "heldout32"):
        split_records = {arm: read_json(results_dir / f"{split_name}-{arm}.json") for arm in ARMS}
        orig_rows = split_records["original"]["per_sample"]
        noop_rows = split_records["noop_control"]["per_sample"]
        if len(orig_rows) != len(noop_rows):
            raise ValueError(f"row count mismatch on {split_name}")
        for r_o, r_n in zip(orig_rows, noop_rows):
            for k in keys:
                if r_o[k] != r_n[k]:
                    raise ValueError(f"noop_control != original on {split_name} {k}")

        for arm in ("anchor1.5_only", "g0.92_only", "combined_anchor1.5_g0.92"):
            cand_rows = split_records[arm]["per_sample"]
            for k in keys:
                c_vals = [r[k] for r in cand_rows]
                b_vals = [r[k] for r in orig_rows]
                st = one_sided_sign_test(c_vals, b_vals)
                recorded = summary["comparisons"][split_name][arm][k]
                if st["better"] != recorded["better"] or st["worse"] != recorded["worse"] or st["ties"] != recorded["ties"]:
                    raise ValueError(f"sign test count mismatch on {split_name} {arm} {k}")
                require_close(st["one_sided_p"], recorded["one_sided_p"], f"{split_name}.{arm}.{k}.p")
                require_close(sum(c_vals) / len(c_vals), recorded["mean_candidate"], f"{split_name}.{arm}.{k}.mean")
                require_close((sum(c_vals) - sum(b_vals)) / len(c_vals), recorded["mean_delta"], f"{split_name}.{arm}.{k}.delta")

    # Audit QA results across all 4 arms
    qa_all = {}
    qa_orig = read_json(results_dir / "qa-original.json")
    for arm in ("original", "g0.92_only", "anchor1.5_only", "combined_anchor1.5_g0.92"):
        qa_rec = read_json(results_dir / f"qa-{arm}.json")
        if len(qa_rec["natural_rows"]) != len(qa_cases) or len(qa_rec["latent4_rows"]) != len(qa_cases):
            raise ValueError(f"QA coverage mismatch for {arm}")
        nat_ok = sum(1 for r, q in zip(qa_rec["natural_rows"], qa_cases) if r["correct"])
        lat_ok = sum(1 for r, q in zip(qa_rec["latent4_rows"], qa_cases) if r["correct"])
        if nat_ok != qa_rec["natural_correct"] or lat_ok != qa_rec["latent4_correct"]:
            raise ValueError(f"QA count mismatch for {arm}")
        broken_nat = sum(1 for ro, rc in zip(qa_orig["natural_rows"], qa_rec["natural_rows"]) if ro["correct"] and not rc["correct"])
        fixed_nat = sum(1 for ro, rc in zip(qa_orig["natural_rows"], qa_rec["natural_rows"]) if not ro["correct"] and rc["correct"])
        broken_lat = sum(1 for ro, rc in zip(qa_orig["latent4_rows"], qa_rec["latent4_rows"]) if ro["correct"] and not rc["correct"])
        fixed_lat = sum(1 for ro, rc in zip(qa_orig["latent4_rows"], qa_rec["latent4_rows"]) if not ro["correct"] and rc["correct"])
        qa_all[arm] = {
            "natural_correct": nat_ok,
            "natural_broken": broken_nat,
            "natural_fixed": fixed_nat,
            "latent4_correct": lat_ok,
            "latent4_broken": broken_lat,
            "latent4_fixed": fixed_lat,
            "mean_exit_anchor_logit": qa_rec["mean_exit_anchor_logit"],
            "median_exit_anchor_rank": qa_rec["median_exit_anchor_rank"],
        }

    # Verify g0.92_only natural outputs are bit-identical to original
    qa_g092 = read_json(qa_g092_path)
    for ro, rg in zip(qa_orig["natural_rows"], qa_g092["natural_rows"]):
        if ro["output"] != rg["output"]:
            raise ValueError("g0.92_only natural output differed from original")

    # Audit physical copies and restore them byte-exactly
    reversals = {}
    for copy_name in ("anchor1.5_backbone", "g0.92_adapter", "noop_adapter"):
        cdir = copies_dir / copy_name
        audit_rec = read_json(cdir / "PATCH_AUDIT.json")
        src_path = Path(audit_rec["copy"]["source"])
        dst_path = experiment / audit_rec["copy"]["copy"]
        rev_path = cdir / "REVERSAL.json"
        if not rev_path.exists():
            if file_hash(dst_path) != audit_rec["edited_file_sha256"]:
                raise ValueError(f"edited copy hash mismatch before reversal: {copy_name}")
            if restore_copies:
                tmp = cdir / "restore_tmp.bin"
                shutil.copyfile(src_path, tmp)
                tmp.replace(dst_path)
                restored_sha = file_hash(dst_path)
                if restored_sha != audit_rec["copy"]["sha256"] or restored_sha != file_hash(src_path):
                    raise ValueError(f"reversal hash mismatch for {copy_name}")
                write_json(
                    rev_path,
                    {
                        "restored": True,
                        "source_sha256": audit_rec["copy"]["sha256"],
                        "edited_sha256": audit_rec["edited_file_sha256"],
                        "restored_sha256": restored_sha,
                    },
                )
        else:
            rev_rec = read_json(rev_path)
            if file_hash(dst_path) != rev_rec["source_sha256"]:
                raise ValueError(f"already restored copy changed: {copy_name}")
        reversals[copy_name] = read_json(rev_path) if rev_path.exists() else {"restored": False}

    verify_frozen_state(root)

    out = {
        "verified": True,
        "noop_control_bit_identical": True,
        "g0.92_natural_qa_bit_identical": True,
        "qa_across_arms": qa_all,
        "reversals": reversals,
        "all_sources_unchanged": True,
    }
    ver_path = experiment / "VERIFICATION.json"
    if ver_path.exists():
        ver_path.unlink()
    write_json(ver_path, out)
    print("C2_AUDIT_COMPLETE " + json.dumps(out), flush=True)
    return out


class VerifyTests(unittest.TestCase):
    def test_require_close(self):
        require_close(1.0, 1.0 + 1e-12, "ok")
        with self.assertRaises(ValueError):
            require_close(1.0, 1.01, "bad")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root:
        run_ablation_qa_and_audit(args.root.resolve(strict=True), restore_copies=True)
    else:
        parser.error("choose --self-test or --root")


if __name__ == "__main__":
    main()
