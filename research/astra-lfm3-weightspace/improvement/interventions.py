import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import unittest

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inventory import read_header
from patch import distinct_files, tensor_audit, verify_other_files
from survey import verify_original


STRENGTHS = (0.5, 1.0, 2.0)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def matched_contrast(original, reference, failed, successful):
    changes = original - reference
    contrast = changes[failed].mean(0) - changes[successful].mean(0)
    length = np.linalg.norm(contrast)
    if length <= 1e-12:
        raise ValueError("No measurable matched contrast")
    return contrast / length


def sparse_directional_edit(weight, direction, columns, strength):
    result = weight.float().clone()
    direction = direction.float()
    direction = direction / torch.linalg.vector_norm(direction)
    indices = torch.as_tensor(columns, dtype=torch.long)
    selected = result[:, indices]
    result[:, indices] = selected - strength * torch.outer(direction, direction @ selected)
    return result.to(weight.dtype)


def propose(root):
    root = Path(root)
    verify_original(root)
    discovery_dir, reference_dir = root / "improvement" / "diagnostic", root / "improvement" / "diagnostic-reference"
    original = rows(discovery_dir / "results.jsonl")
    reference = {row["id"]: row for row in rows(reference_dir / "results.jsonl")}
    free_correct = {row["group"] for row in original if row["format"] == "free" and row["correct"]}
    eligible = [row for row in original if row["format"] == "mcq" and row["group"] in free_correct
                and row["family"] != "selection" and reference[row["id"]]["correct"]]
    failed = np.array([not row["correct"] for row in eligible])
    successful = ~failed
    if min(int(failed.sum()), int(successful.sum())) < 8:
        raise ValueError("Insufficient matched failure/success discovery cases")
    original_acts = np.load(discovery_dir / "activations.npz", allow_pickle=False)
    reference_acts = np.load(reference_dir / "activations.npz", allow_pickle=False)
    output = root / "improvement" / "proposals"
    output.mkdir(exist_ok=False)
    candidates = []
    all_arrays = {}
    with safe_open(root / "original" / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        for layer in range(8, 24):
            hidden_key = f"diagnostic_layer_{layer}"
            mlp_key = f"diagnostic_mlp_{layer}"
            get = lambda archive, key: np.stack([archive[f"{row['id']}::{key}"] for row in eligible]).astype(np.float64)
            direction = matched_contrast(get(original_acts, hidden_key), get(reference_acts, hidden_key), failed, successful)
            scale_name = f"model.language_model.layers.{layer}.post_ffn_norm.weight"
            scale = checkpoint.get_tensor(scale_name).float().numpy().astype(np.float64)
            pre_direction = direction * scale
            pre_direction /= np.linalg.norm(pre_direction)
            name = f"model.language_model.layers.{layer}.feed_forward.w2.weight"
            weight = checkpoint.get_tensor(name)
            writes = pre_direction @ weight.float().numpy().astype(np.float64)
            mlp_changes = get(original_acts, mlp_key) - get(reference_acts, mlp_key)
            contributions = mlp_changes * writes[None, :]
            fail_mean = contributions[failed].mean(0)
            success_mean = contributions[successful].mean(0)
            scores = fail_mean - success_mean - 0.5 * np.abs(success_mean)
            positive = np.flatnonzero((fail_mean > 0) & (scores > 0))
            ranked = sorted(positive.tolist(), key=lambda index: (-scores[index], index))[:64]
            if len(ranked) < 16:
                continue
            columns = np.array(sorted(ranked), dtype=np.int64)
            score = float(scores[columns].sum())
            candidates.append({"layer": layer, "target": name, "shape": list(weight.shape),
                               "score": score, "selected_columns": columns.tolist(),
                               "failure_contribution_sum": float(fail_mean[columns].sum()),
                               "success_contribution_sum": float(success_mean[columns].sum()),
                               "base_tensor_sha256": hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()})
            all_arrays[f"direction_{layer}"] = pre_direction.astype(np.float32)
            all_arrays[f"columns_{layer}"] = columns
    candidates.sort(key=lambda item: (-item["score"], item["layer"]))
    if len(candidates) < 2:
        raise ValueError("No adequate localized writer candidates")
    selected = candidates[:2]
    np.savez_compressed(output / "directions.npz", **all_arrays)
    evidence = {"eligible_cases": len(eligible), "failed": int(failed.sum()), "successful": int(successful.sum()),
                "eligible_ids": [row["id"] for row in eligible], "ranked_layers": candidates,
                "selected_layers": [item["layer"] for item in selected],
                "original_discovery_sha256": sha256(discovery_dir / "results.jsonl"),
                "reference_discovery_sha256": sha256(reference_dir / "results.jsonl"),
                "direction_file_sha256": sha256(output / "directions.npz"),
                "selection_code_sha256": sha256(Path(__file__)),
                "tuning_or_final_used": False}
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2))
    lines = ["# Binding-direction intervention hypothesis", "",
             "Status: discovery-based mathematical hypothesis, not a verified improvement.", "",
             f"Discovery comparison uses {len(eligible)} MCQs with correct free answers and explicit numeric-binding references: {int(failed.sum())} failed, {int(successful.sum())} successful.",
             "Direction = mean(arithmetic-reference | failure) minus mean(arithmetic-reference | success), normalized, then aligned with the stored post-MLP norm gains. This subtracts a shared prompt-condition contrast, but does not remove every possible confound.",
             "Column attribution combines actual pre-down-projection activation differences and each column's write along that direction. Rank uses failure-minus-success contribution and penalizes successful-case contribution magnitude.", "",
             r"For exactly one MLP output tensor $W$ and its chosen columns $S$, test $W'_{:,S}=W_{:,S}-\alpha u(u^T W_{:,S})$; all other columns/tensors remain original.",
             "The post-MLP RMS denominator is nonlinear and is not modeled by this approximation. Actual generated answers and regression tests decide whether the proposal works.",
             "Fixed strengths: 0.5, 1.0, 2.0. Test the top two discovery-ranked layers separately, never combined. No norm, embedding or output-head edits; no gradients or optimizer.",
             "Expected benefit: fewer default-position answers when arithmetic must be mapped to a label. Risks: removal of task-relevant representations, worse arithmetic/free response, selection or multimodal behavior.",
             "Reversal: restore the exact original target-tensor bytes in the copy and verify the whole checkpoint SHA-256. Original source remains read-only.", ""]
    for candidate in selected:
        lines += [f"## Layer {candidate['layer']}", "", f"Target `{candidate['target']}`; shape {candidate['shape']}; score {candidate['score']:.8g}.",
                  f"Failure contribution {candidate['failure_contribution_sum']:.8g}; successful contribution {candidate['success_contribution_sum']:.8g}.",
                  "Exact columns: " + json.dumps(candidate["selected_columns"]), ""]
    document = output / "HYPOTHESIS.md"
    document.write_text("\n".join(lines))
    for candidate in selected:
        for strength in STRENGTHS:
            name = f"layer{candidate['layer']}-a{str(strength).replace('.', 'p')}"
            spec = {**candidate, "name": name, "strength": strength,
                    "hypothesis_sha256": sha256(document), "direction_file_sha256": evidence["direction_file_sha256"],
                    "evidence_sha256": sha256(output / "evidence.json")}
            (output / f"{name}.json").write_text(json.dumps(spec, indent=2))
    print("CANDIDATES_DOCUMENTED " + json.dumps({"selected": selected, "failed": int(failed.sum()), "successful": int(successful.sum())}), flush=True)


def validate_spec(root, spec_path):
    spec_path = Path(spec_path)
    if spec_path.resolve().parent != (root / "improvement" / "proposals").resolve():
        raise ValueError("Spec must be from this study's documented proposals")
    spec = json.loads(spec_path.read_text())
    if not re.fullmatch(r"model\.language_model\.layers\.(?:[89]|1[0-9]|2[0-3])\.feed_forward\.w2\.weight", spec["target"]):
        raise ValueError("Target is outside the single-MLP-output allowlist")
    if spec["strength"] not in STRENGTHS:
        raise ValueError("Undocumented strength")
    proposal_dir = spec_path.parent
    for name, key in (("HYPOTHESIS.md", "hypothesis_sha256"), ("directions.npz", "direction_file_sha256"), ("evidence.json", "evidence_sha256")):
        if sha256(proposal_dir / name) != spec[key]:
            raise ValueError(f"Documented proposal changed: {name}")
    return spec


def apply(root, spec_path):
    root = Path(root)
    manifest = verify_original(root)
    spec = validate_spec(root, spec_path)
    name = spec["name"]
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Unsafe copy name")
    destination = root / "improvement" / "copies" / name
    if destination.exists():
        raise FileExistsError(destination)
    original = root / "original"
    shutil.copytree(original, destination, copy_function=shutil.copy2, ignore=shutil.ignore_patterns(".cache"))
    destination.chmod(0o755)
    model_file = destination / "model.safetensors"
    distinct_files(original / "model.safetensors", model_file)
    before_file_hash = sha256(model_file)
    if before_file_hash != sha256(original / "model.safetensors"):
        raise ValueError("Initial physical copy differs")
    with safe_open(original / "model.safetensors", framework="pt", device="cpu") as source:
        weight = source.get_tensor(spec["target"])
    before_tensor = weight.view(torch.uint8).numpy().tobytes()
    if hashlib.sha256(before_tensor).hexdigest() != spec["base_tensor_sha256"] or list(weight.shape) != spec["shape"]:
        raise ValueError("Original target tensor mismatch")
    with np.load(root / "improvement" / "proposals" / "directions.npz", allow_pickle=False) as data:
        direction = torch.from_numpy(data[f"direction_{spec['layer']}"])
        columns = data[f"columns_{spec['layer']}"]
    if columns.tolist() != spec["selected_columns"]:
        raise ValueError("Selected columns changed")
    replacement = sparse_directional_edit(weight, direction, columns, spec["strength"])
    if not torch.isfinite(replacement).all():
        raise ValueError("Nonfinite replacement")
    after_tensor = replacement.view(torch.uint8).numpy().tobytes()
    with model_file.open("rb") as stream:
        entries, _ = read_header(stream, model_file.stat().st_size)
    entry = next(item for item in entries if item["name"] == spec["target"])
    model_file.chmod(0o644)
    with model_file.open("r+b") as stream:
        stream.seek(entry["file_offsets"][0])
        if stream.read(len(before_tensor)) != before_tensor:
            raise ValueError("Copied target bytes changed before patch")
        stream.seek(entry["file_offsets"][0])
        stream.write(after_tensor)
        stream.flush()
        os.fsync(stream.fileno())
    audit = tensor_audit(original / "model.safetensors", model_file, [spec["target"]])
    verify_other_files(destination, manifest)
    audit.update({"spec_sha256": sha256(spec_path), "base_checkpoint_sha256": before_file_hash,
                  "edited_checkpoint_sha256": sha256(model_file), "target": spec["target"],
                  "file_offsets": entry["file_offsets"], "changed_entries": int((weight != replacement).sum()),
                  "changed_columns": torch.where((weight != replacement).any(0))[0].tolist()})
    (destination / "PATCH_AUDIT.json").write_text(json.dumps(audit, indent=2))
    model_file.chmod(0o444)
    verify_original(root)
    print("INTERVENTION_COPY_AUDITED " + json.dumps({"name": name, "target": spec["target"], "changed_entries": audit["changed_entries"]}), flush=True)


def revert(root, name):
    root = Path(root)
    verify_original(root)
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Unsafe copy name")
    directory = root / "improvement" / "copies" / name
    audit = json.loads((directory / "PATCH_AUDIT.json").read_text())
    target = directory / "model.safetensors"
    if sha256(target) != audit["edited_checkpoint_sha256"]:
        raise ValueError("Copy changed since application")
    start, end = audit["file_offsets"]
    with (root / "original" / "model.safetensors").open("rb") as stream:
        stream.seek(start)
        original_bytes = stream.read(end - start)
    target.chmod(0o644)
    with target.open("r+b") as stream:
        stream.seek(start)
        stream.write(original_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    if sha256(target) != audit["base_checkpoint_sha256"]:
        raise ValueError("Byte-exact reversal failed")
    target.chmod(0o444)
    (directory / "REVERSAL.json").write_text(json.dumps({"original_sha256_restored": audit["base_checkpoint_sha256"]}, indent=2))
    print("INTERVENTION_REVERSED " + name, flush=True)


class InterventionTests(unittest.TestCase):
    def test_only_selected_columns_change(self):
        weight = torch.tensor([[2.0, 3, 4], [5.0, 6, 7]])
        edited = sparse_directional_edit(weight, torch.tensor([1.0, 0]), [0, 2], 1.0)
        self.assertTrue(torch.equal(edited[:, 1], weight[:, 1]))
        self.assertTrue(torch.equal(edited[0, [0, 2]], torch.zeros(2)))
        self.assertTrue(torch.equal(edited[1], weight[1]))

    def test_matched_difference_removes_shared_prompt_shift(self):
        reference = np.zeros((4, 2))
        original = np.array([[3.0, 8], [3.0, 8], [1.0, 8], [1.0, 8]])
        direction = matched_contrast(original, reference, np.array([True, True, False, False]), np.array([False, False, True, True]))
        np.testing.assert_array_equal(direction, [1, 0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--propose", action="store_true")
    parser.add_argument("--apply")
    parser.add_argument("--revert")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    if args.self_test:
        unittest.main(argv=["interventions.py"], verbosity=2)
    elif not args.root:
        parser.error("--root required")
    elif args.propose:
        propose(args.root)
    elif args.apply:
        apply(args.root, args.apply)
    elif args.revert:
        revert(args.root, args.revert)
    else:
        parser.error("Choose --propose, --apply or --revert")