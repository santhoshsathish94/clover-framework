import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import numpy as np


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def direction_geometry(delta):
    left, singular, right = np.linalg.svd(delta.astype(np.float64), full_matrices=False)
    if not np.any(singular):
        raise ValueError("No intervention signal")
    direction = left[:, 0]
    coefficients = singular[0] * right[0]
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
        coefficients = -coefficients
    rank_one = np.outer(direction, coefficients)
    energy = float(np.sum(singular ** 2))
    return {
        "top_singular_values": singular[:8].tolist(),
        "rank_one_energy_fraction": float(singular[0] ** 2 / energy),
        "rank_one_relative_residual": float(np.linalg.norm(delta - rank_one) / np.linalg.norm(delta)),
    }, direction, coefficients


def load_source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operator_measurements(original, edited, direction, columns):
    original = original.astype(np.float64)
    edited = edited.astype(np.float64)
    selected = original[:, columns]
    delta = edited[:, columns] - selected
    original_projection = direction @ selected
    delta_projection = direction @ delta
    projection_energy = float(original_projection @ original_projection)
    fitted_alpha = -float(original_projection @ delta_projection) / projection_energy
    predicted_delta = -6 * np.outer(direction, original_projection)
    orthogonal_delta = delta - np.outer(direction, delta_projection)
    other_columns = np.setdiff1d(np.arange(original.shape[1]), columns)
    return {
        "fitted_alpha": fitted_alpha,
        "post_edit_projection_signed_ratio": float(original_projection @ (direction @ edited[:, columns])) / projection_energy,
        "alpha6_formula_relative_error": float(np.linalg.norm(delta - predicted_delta) / np.linalg.norm(delta)),
        "orthogonal_change_fraction": float(np.linalg.norm(orthogonal_delta) / np.linalg.norm(delta)),
        "unselected_columns_bit_identical": bool(np.array_equal(original[:, other_columns], edited[:, other_columns])),
        "inactive_selected_neurons_give_zero_delta": bool(np.all((edited - original)[:, other_columns] == 0)),
    }


def validate_operator(root):
    root = Path(root)
    source = root / "lm_understand-logs" / "exp8_math"
    reader = load_source_module(source / "code" / "st_numpy.py", "published_st_numpy_operator")
    names = [f"model.language_model.layers.{layer}.mlp.down_proj.weight" for layer in (15, 19)]
    original = reader.load(root / "models" / "qwen-original" / "model.safetensors-00001-of-00001.safetensors", names)
    edited = reader.load(root / "models" / "qwen-p50c.safetensors", names)
    archive = np.load(root / "results" / "published-audit" / "derived-edit-directions.npz", allow_pickle=False)
    result = {}
    for layer, name in zip((15, 19), names):
        base = reader.bf16_to_f32(original[name][1])
        changed = reader.bf16_to_f32(edited[name][1])
        measurements = operator_measurements(base, changed, archive[f"layer{layer}_direction"], archive[f"layer{layer}_columns"])
        measurements["alpha6_prediction_supported_within_0.02"] = abs(measurements["fitted_alpha"] - 6) <= 0.02
        result[str(layer)] = measurements
    output = root / "results" / "published-audit" / "operator-validation.json"
    with output.open("x") as destination:
        json.dump({"measured": result,
                   "hypothesis": "The exact published patch over-subtracts the layer-specific direction at alpha6 on 64 columns, reversing its component near -5x.",
                   "qualification": "Measured local linear operator; not by itself proof of reduced generation loops."}, destination, indent=2)
    print("OPERATOR_MEASURED " + json.dumps(result), flush=True)


def audit(root):
    root = Path(root)
    source = root / "lm_understand-logs" / "exp8_math"
    output = root / "results" / "published-audit"
    output.mkdir(exist_ok=False)
    reader = load_source_module(source / "code" / "st_numpy.py", "published_st_numpy")
    original_path = root / "models" / "qwen-original" / "model.safetensors-00001-of-00001.safetensors"
    edited_path = root / "models" / "qwen-p50c.safetensors"
    before = reader.load(original_path)
    after = reader.load(edited_path)
    if set(before) != set(after):
        raise ValueError("Tensor coverage changed")
    manifest = json.loads((source / "runs" / "p50c" / "edit" / "manifest.json").read_text())
    expected = {entry["name"]: entry for entry in manifest["tensors"]}
    selected = json.loads((source / "runs" / "p50c" / "work_notes" / "frozen_weights_manifest.json").read_text())
    rows, changed, directions = [], [], {}
    for name in sorted(before):
        dtype_before, base = before[name]
        dtype_after, edit = after[name]
        if dtype_before != dtype_after or base.shape != edit.shape:
            raise ValueError(f"Tensor structure changed: {name}")
        if np.array_equal(base, edit):
            if name in expected:
                raise ValueError("Declared tensor did not change")
            continue
        changed.append(name)
        if name not in expected:
            raise ValueError(f"Undeclared tensor changed: {name}")
        entry = expected[name]
        base_digest = hashlib.sha256(base.tobytes()).hexdigest()
        edited_digest = hashlib.sha256(edit.tobytes()).hexdigest()
        if base_digest != entry["base_sha256"] or edited_digest != entry["edited_sha256"]:
            raise ValueError("Published tensor digest mismatch")
        changed_mask = base != edit
        columns = np.flatnonzero(changed_mask.any(0))
        layer = name.split("layers.")[1].split(".")[0]
        if columns.tolist() != selected[layer]["selected_neurons"]:
            raise ValueError("Edited columns differ from the selected-neuron manifest")
        delta = reader.bf16_to_f32(edit[:, columns]) - reader.bf16_to_f32(base[:, columns])
        geometry, direction, coefficients = direction_geometry(delta)
        directions[f"layer{layer}_direction"] = direction
        directions[f"layer{layer}_coefficients"] = coefficients
        directions[f"layer{layer}_columns"] = columns
        rows.append({"name": name, "shape": list(base.shape), "changed_entries": int(changed_mask.sum()),
                     "changed_columns": columns.tolist(), "base_sha256": base_digest,
                     "edited_sha256": edited_digest, **geometry})
    if set(changed) != set(expected):
        raise ValueError("Unexpected changed-tensor set")
    np.savez_compressed(output / "derived-edit-directions.npz", **directions)
    record = {
        "source_commit": "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f",
        "original_file_sha256": sha256(original_path), "edited_file_sha256": sha256(edited_path),
        "tensor_count": len(before), "unchanged_tensors": len(before) - len(changed),
        "changed_tensors": rows,
        "total_changed_entries": sum(row["changed_entries"] for row in rows),
        "absolute_cross_layer_direction_cosine": float(abs(directions["layer15_direction"] @ directions["layer19_direction"])),
        "qualification": "Direction geometry measured from the exact BF16 weight difference; it does not by itself prove a behavioral role.",
    }
    with (output / "weight-audit.json").open("x") as destination:
        json.dump(record, destination, indent=2)
    del before, after

    replay_root = output / "recomputed-eval"
    replay_root.mkdir()
    for name in ("unedited", "p10c", "p25c", "p50c"):
        original = source / "eval" / name
        copied = replay_root / name
        for score_file in original.rglob("scores.json"):
            destination = copied / score_file.relative_to(original)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(score_file, destination)
    subprocess.run([sys.executable, "-B", str(source / "code" / "merge.py"), str(replay_root),
                    "unedited", "p10c", "p25c", "p50c"], check=True)
    counts = {}
    for name in ("unedited", "p10c", "p25c", "p50c"):
        released = json.loads((source / "eval" / name / "merged.json").read_text())
        rebuilt = json.loads((replay_root / name / "merged.json").read_text())
        if rebuilt != released:
            raise ValueError(f"Published aggregation did not reproduce: {name}")
        counts[name] = rebuilt
    with (output / "recorded-score-replay.json").open("x") as destination:
        json.dump({"exactly_recomputed": True, "scores": counts,
                   "qualification": "Recomputed from authors' recorded counts, not newly generated benchmark answers."}, destination, indent=2)
    print("PUBLISHED_AUDIT_COMPLETE " + json.dumps({"tensor_count": record["tensor_count"],
                                                    "unchanged_tensors": record["unchanged_tensors"],
                                                    "total_changed_entries": record["total_changed_entries"],
                                                    "cross_layer_cosine": record["absolute_cross_layer_direction_cosine"],
                                                    "score_merges_exact": True}), flush=True)


class AuditTests(unittest.TestCase):
    def test_exact_rank_one(self):
        delta = np.outer(np.array([3.0, 4.0, 0.0]), np.array([2.0, -1.0]))
        result, direction, coefficients = direction_geometry(delta)
        self.assertAlmostEqual(result["rank_one_energy_fraction"], 1)
        np.testing.assert_allclose(np.outer(direction, coefficients), delta, atol=1e-12)

    def test_detects_non_rank_one(self):
        result, _, _ = direction_geometry(np.eye(3))
        self.assertAlmostEqual(result["rank_one_energy_fraction"], 1 / 3)

    def test_alpha_six_operator(self):
        original = np.array([[2.0, 3, 4], [5.0, 6, 7]])
        direction = np.array([1.0, 0.0])
        columns = np.array([0, 2])
        edited = original.copy()
        edited[:, columns] -= 6 * np.outer(direction, direction @ original[:, columns])
        result = operator_measurements(original, edited, direction, columns)
        self.assertEqual(result["fitted_alpha"], 6)
        self.assertEqual(result["post_edit_projection_signed_ratio"], -5)
        self.assertEqual(result["alpha6_formula_relative_error"], 0)
        self.assertEqual(result["orthogonal_change_fraction"], 0)
        self.assertTrue(result["unselected_columns_bit_identical"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--operator", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["audit_published.py"], verbosity=2)
    elif args.root and args.operator:
        validate_operator(args.root)
    elif args.root:
        audit(args.root)
    else:
        parser.error("--root or --self-test required")