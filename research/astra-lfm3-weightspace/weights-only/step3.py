import argparse
import hashlib
import json
from pathlib import Path
import time
import unittest

import numpy as np
import torch
from safetensors import safe_open

from step1 import EXPECTED_SHA256, TENSOR, closed_boundary, sha256
from step2 import numeric_summary


PROJECTION = "model.language_model.layers.0.conv.in_proj.weight"
SCALE = "model.language_model.layers.0.operator_norm.weight"


def decompose_rows(rows, direction):
    rows = rows.double()
    direction = direction.double()
    direction = direction / torch.linalg.vector_norm(direction)
    coefficients = rows @ direction
    residual = rows - coefficients[:, None] * direction[None, :]
    covariance = residual.T @ residual
    return {
        "direction": direction,
        "common_energy": float(coefficients.square().sum()),
        "residual_energy": float(residual.square().sum()),
        "residual_covariance": covariance,
        "residual_sum": residual.sum(0),
        "coefficient_residual_cross": coefficients @ residual,
        "residual_mean_abs_direction_dot": float((residual @ direction).abs().mean()),
        "row_count": len(rows),
    }


def operator_geometry(operator, decomposition):
    operator = operator.double()
    gram = operator.T @ operator
    direction = decomposition["direction"]
    dimension = operator.shape[1]
    common_gain_squared = max(0.0, float(direction @ gram @ direction))
    trace = float(torch.trace(gram))
    isotropic_gain_squared = trace / dimension
    orthogonal_gain_squared = (trace - common_gain_squared) / (dimension - 1)
    residual_energy = decomposition["residual_energy"]
    projected_residual_energy = float((gram * decomposition["residual_covariance"]).sum())
    residual_gain_squared = projected_residual_energy / residual_energy if residual_energy else 0.0
    projected_common_energy = decomposition["common_energy"] * common_gain_squared
    projected_cross = 2 * float(direction @ gram @ decomposition["coefficient_residual_cross"])
    total = projected_common_energy + projected_residual_energy + projected_cross
    input_fraction = residual_energy / (decomposition["common_energy"] + residual_energy)
    relative_gain = np.sqrt(residual_gain_squared / common_gain_squared) if common_gain_squared else None
    row_norms = torch.linalg.vector_norm(operator, dim=1)
    dots = operator @ direction
    cosines = torch.where(row_norms > 0, dots / row_norms.clamp_min(1e-30), torch.zeros_like(dots))
    return {
        "shape": list(operator.shape),
        "common_direction_gain": float(np.sqrt(common_gain_squared)),
        "isotropic_rms_gain": float(np.sqrt(max(0, isotropic_gain_squared))),
        "orthogonal_isotropic_rms_gain": float(np.sqrt(max(0, orthogonal_gain_squared))),
        "common_gain_relative_to_isotropic": float(np.sqrt(common_gain_squared / isotropic_gain_squared)) if isotropic_gain_squared else None,
        "deviation_rms_gain": float(np.sqrt(max(0, residual_gain_squared))),
        "deviation_to_common_relative_gain": None if relative_gain is None else float(relative_gain),
        "input_deviation_energy_fraction": input_fraction,
        "projected_common_energy": projected_common_energy,
        "projected_deviation_energy": projected_residual_energy,
        "projected_common_deviation_cross_term": projected_cross,
        "projected_total_energy": total,
        "deviation_fraction_of_separate_component_energies": projected_residual_energy / (projected_common_energy + projected_residual_energy) if projected_common_energy + projected_residual_energy else None,
        "projection_row_cosine_to_common_direction": numeric_summary(cosines.numpy()),
        "qualification": "Linear weight geometry; common/deviation terms need not remain orthogonal after projection. No nonlinear layer or inference was run.",
    }


def run(root):
    root = Path(root)
    boundary = closed_boundary(root)
    checkpoint = root / "original" / "model.safetensors"
    if sha256(checkpoint) != EXPECTED_SHA256:
        raise ValueError("Original checkpoint identity mismatch")
    step2_path = root / "weights-only" / "step2"
    earlier = json.loads((step2_path / "result.json").read_text())
    if sha256(step2_path / "arrays.npz") != earlier["arrays_sha256"]:
        raise ValueError("Step2 row grouping changed")
    group = earlier["populations"]["all_low_norm_rows"]["thresholds"]["0.99"]["top_groups"][0]
    with np.load(step2_path / "arrays.npz", allow_pickle=False) as archive:
        row_ids = archive["all_low_norm_rows/row_ids"]
        labels = archive["all_low_norm_rows/labels_0.99"]
        selected_ids = row_ids[labels == group["label"]]
        stored_direction = archive[f"all_low_norm_rows/centroid_0.99_{group['label']}"]
    if len(selected_ids) != 4173:
        raise ValueError("The measured common family changed")
    output = root / "weights-only" / "step3"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        embeddings = source.get_tensor(TENSOR)[torch.from_numpy(selected_ids)].float()
        projection = source.get_tensor(PROJECTION)
        scaling = source.get_tensor(SCALE)
    if projection.shape != (6144, 2048) or scaling.shape != (2048,):
        raise ValueError("Unexpected first-projection weight shapes")
    tensor_hashes = {PROJECTION: hashlib.sha256(projection.view(torch.uint8).numpy().tobytes()).hexdigest(),
                     SCALE: hashlib.sha256(scaling.view(torch.uint8).numpy().tobytes()).hexdigest()}
    projection = projection.double()
    scaling = scaling.double()
    direction = torch.from_numpy(stored_direction)
    decomposition = decompose_rows(embeddings, direction)
    results = {}
    for variant, operator in (("bare_projection", projection),
                              ("column_scaled_projection", projection * scaling[None, :])):
        full = operator_geometry(operator, decomposition)
        full["equal_row_blocks"] = []
        for block in range(3):
            details = operator_geometry(operator[block * 2048:(block + 1) * 2048], decomposition)
            details["row_range_half_open"] = [block * 2048, (block + 1) * 2048]
            details["block_role"] = "not assigned from weights alone"
            full["equal_row_blocks"].append(details)
        results[variant] = full
        print(json.dumps({"variant": variant,
                          "common_relative_gain": full["common_gain_relative_to_isotropic"],
                          "deviation_to_common_gain": full["deviation_to_common_relative_gain"],
                          "input_deviation_energy_fraction": full["input_deviation_energy_fraction"],
                          "projected_deviation_component_fraction": full["deviation_fraction_of_separate_component_energies"]}), flush=True)
    report = {
        "evidence_mode": "checkpoint weights and previously measured weight-only row family; no labels, prompts, activations, inference or model-code interpretation",
        "boundary": boundary, "tensor_hashes": tensor_hashes,
        "step2_result_sha256": sha256(step2_path / "result.json"),
        "family_row_count": len(selected_ids), "family_row_ids_first16": selected_ids[:16].tolist(),
        "direction_norm": float(torch.linalg.vector_norm(decomposition["direction"])),
        "input_common_energy": decomposition["common_energy"],
        "input_deviation_energy": decomposition["residual_energy"],
        "residual_mean_abs_direction_dot": decomposition["residual_mean_abs_direction_dot"],
        "projection_results": results,
        "scaling_vector": numeric_summary(scaling.numpy()),
        "scaling_composition_definition": "A = P diag(s); this is an algebraic composition of named stored weights, not a model forward or a claim about the full normalization/gating computation",
        "checkpoint_sha256_before": EXPECTED_SHA256,
        "checkpoint_sha256_after": sha256(checkpoint),
        "elapsed_s": time.monotonic() - started,
        "limits": ["Only the first input projection and its dimension-matched named scaling vector were examined",
                   "RMS normalization, nonlinear gates, temporal convolution, output projection and residual paths were not evaluated",
                   "Static gain ratios do not establish a semantic function, behavioral distinction, defect, or downstream influence in the complete model",
                   "Three equal output-row blocks are inspected without assigning undocumented gate roles"],
    }
    if report["checkpoint_sha256_after"] != EXPECTED_SHA256:
        raise ValueError("Original checkpoint changed")
    np.savez_compressed(output / "arrays.npz", row_ids=selected_ids,
                        common_direction=decomposition["direction"].numpy())
    report["arrays_sha256"] = sha256(output / "arrays.npz")
    with (output / "result.json").open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
    print("WEIGHTS_ONLY_STEP3_COMPLETE " + json.dumps({"elapsed_s": report["elapsed_s"], "original_unchanged": True}), flush=True)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = torch.tensor([[2.0, 0.1, 0.0], [2.0, -0.1, 0.0]], dtype=torch.float64)
        self.parts = decompose_rows(self.rows, torch.tensor([1.0, 0.0, 0.0]))

    def test_identity_preserves_geometry(self):
        result = operator_geometry(torch.eye(3), self.parts)
        self.assertAlmostEqual(result["common_direction_gain"], 1)
        self.assertAlmostEqual(result["deviation_rms_gain"], 1)
        self.assertAlmostEqual(result["deviation_to_common_relative_gain"], 1)
        self.assertAlmostEqual(result["projected_total_energy"], float(self.rows.square().sum()))

    def test_common_nullspace_detected(self):
        result = operator_geometry(torch.diag(torch.tensor([0.0, 2.0, 1.0])), self.parts)
        self.assertEqual(result["common_direction_gain"], 0)
        self.assertAlmostEqual(result["deviation_rms_gain"], 2)
        self.assertIsNone(result["deviation_to_common_relative_gain"])

    def test_relative_amplification_and_scaling(self):
        operator = torch.diag(torch.tensor([0.5, 3.0, 1.0]))
        result = operator_geometry(operator, self.parts)
        self.assertAlmostEqual(result["deviation_to_common_relative_gain"], 6)
        scaled = operator_geometry(operator * torch.tensor([2.0, 1.0, 1.0])[None, :], self.parts)
        self.assertAlmostEqual(scaled["deviation_to_common_relative_gain"], 3)

    def test_projected_cross_term_is_accounted_for(self):
        rows = torch.tensor([[2.0, 0.1, 0.0], [2.0, 0.3, 0.0]], dtype=torch.float64)
        parts = decompose_rows(rows, torch.tensor([1.0, 0.0, 0.0]))
        operator = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64)
        result = operator_geometry(operator, parts)
        self.assertAlmostEqual(result["projected_common_deviation_cross_term"], 1.6)
        self.assertAlmostEqual(result["projected_total_energy"], 9.7)
        self.assertAlmostEqual(result["projected_total_energy"], float((rows @ operator.T).square().sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    if args.self_test:
        unittest.main(argv=["step3.py"], verbosity=2)
    elif args.root:
        run(args.root)
    else:
        parser.error("--root or --self-test required")