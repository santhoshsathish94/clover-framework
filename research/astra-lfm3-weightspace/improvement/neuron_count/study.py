import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import unittest

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
from diagnose import build_cases, numeric_answer
from interventions import matched_contrast, sparse_directional_edit
from inventory import read_header
from patch import distinct_files, tensor_audit, verify_other_files
from runtime import OmniRuntime
from survey import verify_original


WIDTHS = (64, 128, 256, 512, 1024)
TARGET = "model.language_model.layers.22.feed_forward.w2.weight"
OLD64_HASH = "0e428fe5808efa47e7f588078b8d348b8363001354a500e97df5a61110fa6078"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def select_nested(scores, failure_mean, widths=WIDTHS):
    eligible = np.flatnonzero((scores > 0) & (failure_mean > 0))
    ranked = sorted(eligible.tolist(), key=lambda index: (-scores[index], index))
    return {width: np.array(sorted(ranked[:width]), dtype=np.int64)
            for width in widths if width <= len(ranked)}, len(ranked)


def norm_matched_strength(weight, direction, columns, reference_columns, reference_strength=4.0):
    direction = direction.double()
    direction = direction / torch.linalg.vector_norm(direction)
    projections = direction @ weight.double()
    reference_norm = float(torch.linalg.vector_norm(projections[reference_columns]))
    selected_norm = float(torch.linalg.vector_norm(projections[columns]))
    if selected_norm == 0:
        raise ValueError("Selected columns have zero directional write")
    return reference_strength * reference_norm / selected_norm


def build_confirmation(used_questions, seed=6649817):
    used = set(used_questions)
    counts = {family: 0 for family in ("addition", "subtraction", "multiplication", "selection")}
    result = []
    group_index = 0
    for attempt in range(1000):
        grouped = defaultdict(list)
        for case in build_cases(seed + attempt * 1009):
            grouped[case["group"]].append(case)
        for group_cases in grouped.values():
            representative = next(case for case in group_cases if case["format"] == "mcq")
            family = representative["family"]
            question = representative["prompt"].splitlines()[0]
            if counts[family] >= 24 or question in used:
                continue
            used.add(question)
            counts[family] += 1
            for case in group_cases:
                suffix = case["id"].rsplit("-", 1)[1]
                result.append({**case, "id": f"width-confirmation-{group_index:03d}-{suffix}",
                               "group": group_index, "split": "width-confirmation"})
            group_index += 1
        if all(count == 24 for count in counts.values()):
            return result
    raise ValueError("Insufficient fresh balanced questions; do not reuse exposed questions")


def prepare(root):
    root = Path(root)
    verify_original(root)
    base = root / "improvement"
    work = base / "neuron_count"
    output = work / "frozen"
    output.mkdir(exist_ok=False)
    cases = json.loads((base / "diagnostic" / "cases.json").read_text())
    original = read_rows(base / "diagnostic" / "results.jsonl")
    reference = {row["id"]: row for row in read_rows(base / "diagnostic-reference" / "results.jsonl")}
    free_correct = {row["group"] for row in original if row["format"] == "free" and row["correct"]}
    eligible = [row for row in original if row["format"] == "mcq" and row["family"] != "selection"
                and row["group"] in free_correct and reference[row["id"]]["correct"]]
    failed = np.array([not row["correct"] for row in eligible])
    successful = ~failed
    if int(failed.sum()) != 31 or int(successful.sum()) != 25:
        raise ValueError("The original discovery cohort changed")
    original_acts = np.load(base / "diagnostic" / "activations.npz", allow_pickle=False)
    reference_acts = np.load(base / "diagnostic-reference" / "activations.npz", allow_pickle=False)
    get = lambda archive, key: np.stack([archive[f"{row['id']}::{key}"] for row in eligible]).astype(np.float64)
    hidden_direction = matched_contrast(get(original_acts, "diagnostic_layer_22"), get(reference_acts, "diagnostic_layer_22"), failed, successful)
    with safe_open(root / "original" / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        weight = checkpoint.get_tensor(TARGET)
        scaling = checkpoint.get_tensor("model.language_model.layers.22.post_ffn_norm.weight").float().numpy().astype(np.float64)
    pre_direction = hidden_direction * scaling
    pre_direction /= np.linalg.norm(pre_direction)
    writes = pre_direction @ weight.float().numpy().astype(np.float64)
    contribution = (get(original_acts, "diagnostic_mlp_22") - get(reference_acts, "diagnostic_mlp_22")) * writes[None, :]
    failure_mean, success_mean = contribution[failed].mean(0), contribution[successful].mean(0)
    scores = failure_mean - success_mean - 0.5 * np.abs(success_mean)
    subsets, eligible_count = select_nested(scores, failure_mean)
    prior = json.loads((base / "proposals" / "layer22-a2p0.json").read_text())
    prior_arrays = np.load(base / "proposals" / "directions.npz", allow_pickle=False)
    direction = prior_arrays["direction_22"]
    if not np.array_equal(pre_direction.astype(np.float32), direction) or subsets[64].tolist() != prior["selected_columns"]:
        raise ValueError("The original64-column ranking/direction did not reconstruct exactly")
    tensor_bytes = weight.view(torch.uint8).numpy().tobytes()
    if hashlib.sha256(tensor_bytes).hexdigest() != prior["base_tensor_sha256"]:
        raise ValueError("Original target tensor changed")
    tensor_direction = torch.from_numpy(direction)
    specs = []
    arrays = {"direction": direction, "scores": scores}
    for width, columns in subsets.items():
        arrays[f"columns_{width}"] = columns
        for track in (("fixed",) if width == 64 else ("fixed", "norm_matched")):
            strength = 4.0 if track == "fixed" else norm_matched_strength(weight, tensor_direction, columns, subsets[64])
            edited = sparse_directional_edit(weight, tensor_direction, columns, strength)
            delta = edited.float() - weight.float()
            specs.append({"name": f"n{width}-{track}", "width": width, "track": track,
                          "strength": strength, "target": TARGET,
                          "base_tensor_sha256": prior["base_tensor_sha256"],
                          "edited_tensor_sha256": hashlib.sha256(edited.view(torch.uint8).numpy().tobytes()).hexdigest(),
                          "changed_entries": int((edited != weight).sum()),
                          "actual_delta_frobenius": float(torch.linalg.vector_norm(delta)),
                          "relative_delta_frobenius": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(weight.float()))})
    np.savez_compressed(output / "directions.npz", **arrays)
    (output / "specs.json").write_text(json.dumps(specs, indent=2))
    used = {case["prompt"].splitlines()[0] for case in cases if case["format"] == "mcq"}
    for filename in ("tuning.json", "final.json"):
        previous = json.loads((base / "frozen" / filename).read_text())
        used.update(case["prompt"].splitlines()[0] for case in previous if case["format"] == "mcq")
    confirmation = build_confirmation(used)
    (output / "confirmation.json").write_text(json.dumps(confirmation, indent=2))
    (output / "HYPOTHESIS.md").write_text((work / "CONTEXT.md").read_text())
    manifest = {"requested_widths": list(WIDTHS), "eligible_ranked_neurons": eligible_count,
                "available_widths": list(subsets), "spec_count": len(specs),
                "original64_rank_and_direction_exact": True, "tuning_case_sha256": sha256(base / "frozen" / "tuning.json"),
                "files": {name: sha256(output / name) for name in ("directions.npz", "specs.json", "confirmation.json", "HYPOTHESIS.md")},
                "hypothesis_sha256": sha256(output / "HYPOTHESIS.md"),
                "sources": {relative: sha256(root / relative) for relative in (
                    "runtime.py", "improvement/diagnose.py", "improvement/evaluate.py", "improvement/interventions.py",
                    "improvement/neuron_count/study.py")},
                "confirmation_not_used_for_selection": True}
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("NEURON_COUNT_PREPARED " + json.dumps({"eligible": eligible_count, "specs": specs}), flush=True)


def verify_frozen(root):
    work = root / "improvement" / "neuron_count"
    manifest = json.loads((work / "frozen" / "MANIFEST.json").read_text())
    for name, checksum in manifest["files"].items():
        if sha256(work / "frozen" / name) != checksum:
            raise ValueError(f"Frozen width artifact changed: {name}")
    for name, checksum in manifest["sources"].items():
        if sha256(root / name) != checksum:
            raise ValueError(f"Original evaluator/patch source changed: {name}")
    if sha256(root / "improvement" / "frozen" / "tuning.json") != manifest["tuning_case_sha256"]:
        raise ValueError("Tuning cases changed")
    return manifest


def make_copy(root, name):
    root = Path(root)
    verify_frozen(root)
    original_manifest = verify_original(root)
    work = root / "improvement" / "neuron_count"
    specs = json.loads((work / "frozen" / "specs.json").read_text())
    spec = next(item for item in specs if item["name"] == name)
    if Path(name).name != name or spec["target"] != TARGET:
        raise ValueError("Unexpected copy or target")
    destination = root / "improvement" / "copies" / f"width-{name}"
    if destination.exists():
        raise FileExistsError(destination)
    original = root / "original"
    shutil.copytree(original, destination, copy_function=shutil.copy2, ignore=shutil.ignore_patterns(".cache"))
    destination.chmod(0o755)
    target = destination / "model.safetensors"
    distinct_files(original / "model.safetensors", target)
    base_hash = sha256(target)
    if base_hash != sha256(original / "model.safetensors"):
        raise ValueError("Initial copy is not exact")
    with safe_open(original / "model.safetensors", framework="pt", device="cpu") as source:
        weight = source.get_tensor(TARGET)
    before = weight.view(torch.uint8).numpy().tobytes()
    if hashlib.sha256(before).hexdigest() != spec["base_tensor_sha256"]:
        raise ValueError("Target identity mismatch")
    with np.load(work / "frozen" / "directions.npz", allow_pickle=False) as data:
        direction = torch.from_numpy(data["direction"])
        columns = data[f"columns_{spec['width']}"]
    edited = sparse_directional_edit(weight, direction, columns, spec["strength"])
    after = edited.view(torch.uint8).numpy().tobytes()
    if hashlib.sha256(after).hexdigest() != spec["edited_tensor_sha256"]:
        raise ValueError("Reconstructed wider edit differs from frozen spec")
    with target.open("rb") as stream:
        header, _ = read_header(stream, target.stat().st_size)
    entry = next(item for item in header if item["name"] == TARGET)
    target.chmod(0o644)
    with target.open("r+b") as stream:
        stream.seek(entry["file_offsets"][0])
        if stream.read(len(before)) != before:
            raise ValueError("Copied target changed before edit")
        stream.seek(entry["file_offsets"][0])
        stream.write(after)
        stream.flush()
        os.fsync(stream.fileno())
    audit = tensor_audit(original / "model.safetensors", target, [TARGET])
    verify_other_files(destination, original_manifest)
    edited_hash = sha256(target)
    if name == "n64-fixed" and edited_hash != OLD64_HASH:
        raise ValueError("64-neuron control does not match the previous candidate")
    changed_columns = torch.where((edited != weight).any(0))[0].tolist()
    if changed_columns != columns.tolist():
        raise ValueError("Changed-column audit differs from requested neuron count")
    audit.update({"base_checkpoint_sha256": base_hash, "edited_checkpoint_sha256": edited_hash,
                  "target": TARGET, "file_offsets": entry["file_offsets"], "spec": spec,
                  "changed_columns": changed_columns, "changed_entries": int((edited != weight).sum())})
    (destination / "PATCH_AUDIT.json").write_text(json.dumps(audit, indent=2))
    target.chmod(0o444)
    verify_original(root)
    print("WIDTH_COPY_AUDITED " + json.dumps({"name": name, "width": spec["width"], "sha256": edited_hash}), flush=True)


def evaluate_confirmation(root, name, checkpoint=None):
    root = Path(root)
    verify_frozen(root)
    verify_original(root)
    work = root / "improvement" / "neuron_count"
    selection_path = work / "SELECTION.json"
    if not selection_path.exists():
        raise ValueError("Freeze a qualifying width selection before opening confirmation")
    cases = json.loads((work / "frozen" / "confirmation.json").read_text())
    destination = work / "confirmation" / name
    destination.mkdir(parents=True, exist_ok=False)
    runtime = OmniRuntime(root, checkpoint=checkpoint)
    for handle in runtime.handles:
        handle.remove()
    runtime.handles = []
    totals = defaultdict(lambda: {"correct": 0, "total": 0})
    start = time.monotonic()
    with (destination / "results.jsonl").open("x") as output:
        for index, case in enumerate(cases):
            row, _ = runtime.run_case(case, destination)
            if case["format"] == "free":
                row["predicted_value"] = numeric_answer(row["text"])
                row["correct"] = row["predicted_value"] == case["gold_value"]
            else:
                label = row["prediction"]
                row["predicted_value"] = case["choices"]["ABCD".index(label)] if label else None
            row.update({"group": case["group"], "family": case["family"], "format": case["format"]})
            kind = "selection" if case["family"] == "selection" else "arithmetic_free" if case["format"] == "free" else "arithmetic_mcq"
            totals[kind]["total"] += 1
            totals[kind]["correct"] += bool(row["correct"])
            output.write(json.dumps(row) + "\n")
            output.flush()
            if (index + 1) % 100 == 0:
                print(json.dumps({"name": name, "completed": index + 1}), flush=True)
    verify_original(root)
    verify_frozen(root)
    summary = {"name": name, "totals": dict(totals), "case_count": len(cases),
               "result_sha256": sha256(destination / "results.jsonl"),
               "elapsed_s": time.monotonic() - start, "original_unchanged": True}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    print("WIDTH_CONFIRMATION_COMPLETE " + json.dumps(summary), flush=True)


class WidthTests(unittest.TestCase):
    def test_nested_ranking_excludes_negative_candidates(self):
        subsets, eligible = select_nested(np.array([4.0, 2, -1, 3, 1]), np.array([1.0, 1, 1, 1, -1]), (1, 2, 3, 4))
        self.assertEqual(eligible, 3)
        np.testing.assert_array_equal(subsets[1], [0])
        np.testing.assert_array_equal(subsets[2], [0, 3])
        self.assertNotIn(4, subsets)

    def test_norm_matching_holds_directional_delta_constant(self):
        weight = torch.tensor([[3.0, 4, 0], [1.0, 2, 3]])
        direction = torch.tensor([1.0, 0])
        strength = norm_matched_strength(weight, direction, [0, 1], [0])
        self.assertAlmostEqual(strength, 2.4)
        old = sparse_directional_edit(weight, direction, [0], 4)
        wider = sparse_directional_edit(weight, direction, [0, 1], strength)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(old - weight)), float(torch.linalg.vector_norm(wider - weight)), places=5)

    def test_confirmation_disjoint_from_all_previous_splits(self):
        from evaluate import build_split

        used = {case["prompt"].splitlines()[0] for case in build_cases() if case["format"] == "mcq"}
        build_split("old-tuning", 183019, 2, used)
        build_split("old-final", 917431, 4, used)
        confirmation = build_confirmation(used)
        questions = {case["prompt"].splitlines()[0] for case in confirmation if case["format"] == "mcq"}
        self.assertEqual(len(confirmation), 456)
        self.assertEqual(len(questions), 96)
        self.assertFalse(questions & used)
        for family in ("addition", "subtraction", "multiplication", "selection"):
            groups = {case["group"] for case in confirmation if case["family"] == family}
            self.assertEqual(len(groups), 24)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--copy")
    parser.add_argument("--confirm-name")
    parser.add_argument("--checkpoint")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    if args.self_test:
        unittest.main(argv=["study.py"], verbosity=2)
    elif not args.root:
        parser.error("--root required")
    elif args.prepare:
        prepare(args.root)
    elif args.copy:
        make_copy(args.root, args.copy)
    elif args.confirm_name:
        evaluate_confirmation(args.root, args.confirm_name, args.checkpoint)
    else:
        parser.error("Choose --prepare, --copy or --confirm-name")