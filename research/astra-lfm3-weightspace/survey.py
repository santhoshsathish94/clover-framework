import argparse
from collections import defaultdict
import errno
import hashlib
import json
import math
from pathlib import Path
import time
import unittest

import numpy as np
import torch
from safetensors import safe_open


def digest_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_original(root, require_readonly=True):
    root = Path(root)
    manifest = json.loads((root / "SEALED.json").read_text())
    for entry in manifest["files"]:
        path = root / "original" / entry["file"]
        if digest_file(path) != entry["sha256"]:
            raise RuntimeError(f"Original digest mismatch: {entry['file']}")
    if require_readonly:
        import os
        path = root / "original" / "model.safetensors"
        try:
            descriptor = os.open(path, os.O_WRONLY)
        except OSError as error:
            if error.errno not in (errno.EROFS, errno.EACCES, errno.EPERM):
                raise
        else:
            os.close(descriptor)
            raise RuntimeError("Original weights are writable; refuse to run")
    return manifest


def moments(values):
    flat = values.reshape(-1)
    totals = np.zeros(4, dtype=np.float64)
    minimum, maximum = math.inf, -math.inf
    zeros = finite_count = 0
    for chunk in flat.split(1_048_576):
        chunk = chunk.double()
        finite = torch.isfinite(chunk)
        finite_count += int(finite.sum())
        chunk = chunk[finite]
        if not chunk.numel():
            continue
        zeros += int((chunk == 0).sum())
        minimum = min(minimum, float(chunk.min()))
        maximum = max(maximum, float(chunk.max()))
        squared = chunk.square()
        totals += np.array([
            float(chunk.sum()), float(squared.sum()),
            float((squared * chunk).sum()), float(squared.square().sum()),
        ])
    if not finite_count:
        return {"finite_count": 0, "nonfinite_count": flat.numel()}
    first, second, third, fourth = totals / finite_count
    variance = max(0.0, second - first * first)
    centered_third = third - 3 * first * second + 2 * first ** 3
    centered_fourth = fourth - 4 * first * third + 6 * first ** 2 * second - 3 * first ** 4
    return {
        "finite_count": finite_count, "nonfinite_count": flat.numel() - finite_count,
        "zero_count": zeros, "min": minimum, "max": maximum,
        "mean": float(first), "std": math.sqrt(variance),
        "rms": math.sqrt(second), "frobenius_squared": float(totals[1]),
        "skewness": float(centered_third / variance ** 1.5) if variance > 0 else None,
        "excess_kurtosis": float(centered_fourth / variance ** 2 - 3) if variance > 0 else None,
    }


def vector_summary(values):
    values = values.float()
    quantiles = torch.tensor([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    top_count = min(12, values.numel())
    top_values, top_indices = torch.topk(values, top_count)
    return {
        "quantile_probabilities": quantiles.tolist(),
        "quantiles": torch.quantile(values, quantiles).tolist(),
        "top": [{"index": int(index), "value": float(value)}
                for index, value in zip(top_indices, top_values)],
    }


def aligned_indices(length, count):
    count = min(length, count)
    if count <= 1:
        return torch.zeros(count, dtype=torch.int64)
    return torch.arange(count, dtype=torch.int64) * (length - 1) // (count - 1)


def spectra(matrix, rank=16):
    smaller = min(matrix.shape)
    energy = float(matrix.double().square().sum())
    if energy == 0:
        return {"method": "zero_matrix", "singular_values": [], "energy": 0.0}
    if smaller <= 256 and matrix.numel() <= 8_000_000:
        singular_values = torch.linalg.svdvals(matrix.float())
        method = "exact_full_svd_float32"
    else:
        torch.manual_seed(20261111)
        _, singular_values, _ = torch.svd_lowrank(
            matrix.float(), q=min(smaller, rank + 8), niter=2,
        )
        singular_values = singular_values[:rank]
        method = "randomized_top16_oversample8_power2_float32"
    captured = singular_values.double().square().cumsum(0) / energy
    result = {
        "method": method, "singular_values": singular_values.tolist(),
        "cumulative_frobenius_energy": captured.tolist(), "frobenius_energy": energy,
        "qualification": "Randomized values are approximate lower Ritz estimates, not a full spectrum.",
    }
    if method.startswith("exact"):
        result["rank_for_energy"] = {
            str(threshold): int(torch.searchsorted(captured, threshold)) + 1
            for threshold in (0.90, 0.95, 0.99)
        }
    return result


def matrix_statistics(matrix):
    matrix = matrix.float()
    row_norms = torch.linalg.vector_norm(matrix, dim=1)
    column_norms = torch.linalg.vector_norm(matrix, dim=0)
    row_indices = torch.unique(torch.cat((
        torch.topk(row_norms, min(8, len(row_norms))).indices,
        aligned_indices(matrix.shape[0], 8),
    )))
    selected = torch.nn.functional.normalize(matrix[row_indices], dim=1)
    similarities = selected @ selected.T
    pairs = []
    for first_index in range(len(row_indices)):
        for second_index in range(first_index + 1, len(row_indices)):
            pairs.append({
                "row_a": int(row_indices[first_index]),
                "row_b": int(row_indices[second_index]),
                "cosine": float(similarities[first_index, second_index]),
            })
    pairs.sort(key=lambda pair: abs(pair["cosine"]), reverse=True)
    return {
        "matrix_shape": list(matrix.shape),
        "row_norms": vector_summary(row_norms),
        "column_norms": vector_summary(column_norms),
        "selected_row_correlations": pairs[:12],
        "correlation_sampling": "top8 row norms plus 8 evenly spaced rows; not exhaustive row pairs",
    }, row_norms.numpy(), column_norms.numpy()


def analyze_tensor(weight):
    result = moments(weight)
    arrays = {}
    if result["nonfinite_count"] or weight.numel() == 0:
        result["analysis_skipped"] = "empty or nonfinite tensor; recorded without correction"
        return result, arrays
    flat = weight.flatten()
    generator = torch.Generator().manual_seed(20261111)
    if flat.numel() <= 65_536:
        sample = flat.float()
        quantile_method = "all_elements"
    else:
        sample = flat[torch.randint(flat.numel(), (65_536,), generator=generator)].float()
        quantile_method = "65536 uniform draws with replacement; seed 20261111"
    result["value_quantiles"] = vector_summary(sample)
    result["value_quantile_sampling"] = quantile_method
    standardized = (sample - result["mean"]) / max(result["std"], 1e-30)
    edges = np.array([-np.inf, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, np.inf])
    counts, _ = np.histogram(standardized.numpy(), bins=edges)
    result["standardized_histogram"] = {
        "edges": ["-inf", -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, "inf"],
        "counts": counts.tolist(), "sample_count": sample.numel(),
    }
    if weight.ndim >= 2:
        matrix = weight.reshape(weight.shape[0], -1).float()
        matrix_result, row_norms, column_norms = matrix_statistics(matrix)
        result.update(matrix_result)
        result["flattening"] = "axis0 rows; all remaining axes flattened as columns"
        arrays["row_norms"] = row_norms
        arrays["column_norms"] = column_norms
        row_counts = torch.zeros(matrix.shape[0], dtype=torch.int64)
        column_counts = torch.zeros(matrix.shape[1], dtype=torch.int64)
        threshold = 6 * result["std"]
        for offset in range(0, matrix.shape[0], 1024):
            outliers = (matrix[offset:offset + 1024] - result["mean"]).abs() > threshold
            row_counts[offset:offset + len(outliers)] = outliers.sum(1)
            column_counts += outliers.sum(0)
        result["outliers_gt_6std"] = {
            "count": int(row_counts.sum()),
            "row_concentration": vector_summary(row_counts),
            "column_concentration": vector_summary(column_counts),
            "qualification": "Descriptive threshold only; not evidence of a defect.",
        }
        arrays["row_outlier_counts"] = row_counts.numpy()
        arrays["column_outlier_counts"] = column_counts.numpy()
        result["spectrum"] = spectra(matrix)
    return result, arrays


def main_survey(root, run_name="weight-survey"):
    root = Path(root)
    manifest = verify_original(root)
    inventory = json.loads((root / "reports" / "inventory.json").read_text())
    checkpoints = {"main": inventory, **inventory.get("auxiliary_checkpoints", {})}
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run_name must be a single directory name")
    output_dir = root / "reports" / run_name
    output_dir.mkdir(exist_ok=False)
    norms = {}
    fingerprints = defaultdict(list)
    start_time = time.monotonic()
    completed = 0
    with (output_dir / "tensors.jsonl").open("x") as output:
        for checkpoint_name, checkpoint in checkpoints.items():
            model_dir = Path(checkpoint["model_dir"])
            by_shard = defaultdict(list)
            for entry in checkpoint["tensors"]:
                by_shard[entry["shard"]].append(entry)
            for shard_name, entries in by_shard.items():
                with safe_open(model_dir / shard_name, framework="pt", device="cpu") as checkpoint_file:
                    for entry in entries:
                        weight = checkpoint_file.get_tensor(entry["name"])
                        result, arrays = analyze_tensor(weight)
                        identifier = f"{checkpoint_name}/{entry['name']}"
                        result.update({
                            "id": identifier, "checkpoint": checkpoint_name,
                            "name": entry["name"], "shape": entry["shape"],
                            "dtype": entry["dtype"], "component": entry["component"],
                            "parameter_count": entry["parameter_count"],
                        })
                        output.write(json.dumps(result, allow_nan=False) + "\n")
                        output.flush()
                        for name, values in arrays.items():
                            norms[f"{identifier}/{name}"] = values
                        if weight.ndim == 2 and weight.numel() >= 4096:
                            indices = aligned_indices(weight.numel(), 4096)
                            sample = weight.flatten()[indices].float().numpy()
                            key = (entry["component"], tuple(weight.shape))
                            fingerprints[key].append((identifier, sample))
                        completed += 1
                        if completed % 20 == 0:
                            print(json.dumps({"tensors_done": completed,
                                              "elapsed_s": round(time.monotonic() - start_time, 1)}), flush=True)
    np.savez_compressed(output_dir / "norms_and_concentrations.npz", **norms)
    correlations = []
    for (component, shape), entries in fingerprints.items():
        if len(entries) < 2:
            continue
        samples = np.stack([entry[1] for entry in entries]).astype(np.float64)
        samples -= samples.mean(axis=1, keepdims=True)
        lengths = np.linalg.norm(samples, axis=1, keepdims=True)
        samples /= np.maximum(lengths, 1e-30)
        correlation = samples @ samples.T
        for first in range(len(entries)):
            for second in range(first + 1, len(entries)):
                correlations.append({
                    "tensor_a": entries[first][0], "tensor_b": entries[second][0],
                    "component": component, "shape": shape,
                    "sampled_pearson": float(correlation[first, second]),
                })
    correlations.sort(key=lambda entry: abs(entry["sampled_pearson"]), reverse=True)
    summary = {
        "tensor_count": completed, "elapsed_s": time.monotonic() - start_time,
        "original_revision": manifest["revision"],
        "source_hashes_verified_before_and_after": False,
        "same_shape_correlations_top50": correlations[:50],
        "correlation_method": "4096 aligned evenly spaced elements, same shape/component; exploratory multiple comparisons",
    }
    verify_original(root)
    summary["source_hashes_verified_before_and_after"] = True
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print("SURVEY_COMPLETE " + json.dumps({"tensor_count": completed,
                                         "elapsed_s": summary["elapsed_s"]}), flush=True)


class SurveyTests(unittest.TestCase):
    def test_moments_known_values(self):
        result = moments(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
        self.assertEqual(result["mean"], 0)
        self.assertAlmostEqual(result["std"], math.sqrt(2))
        self.assertEqual(result["zero_count"], 1)
        self.assertAlmostEqual(result["excess_kurtosis"], -1.3)

    def test_nonfinite_not_hidden(self):
        result = moments(torch.tensor([1.0, float("nan"), float("inf")]))
        self.assertEqual(result["nonfinite_count"], 2)

    def test_diagonal_spectrum(self):
        result = spectra(torch.diag(torch.tensor([3.0, 2.0, 0.0])))
        self.assertEqual(result["singular_values"], [3.0, 2.0, 0.0])
        self.assertEqual(result["rank_for_energy"]["0.99"], 2)

    def test_row_column_norms(self):
        result, rows, columns = matrix_statistics(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
        np.testing.assert_array_equal(rows, [5, 0])
        np.testing.assert_array_equal(columns, [3, 4])
        self.assertEqual(result["matrix_shape"], [2, 2])

    def test_constant_not_an_outlier(self):
        result, _ = analyze_tensor(torch.ones(3, 4))
        self.assertEqual(result["outliers_gt_6std"]["count"], 0)

    def test_large_tensor_indices_are_exact(self):
        indices = aligned_indices(49_569_792, 4096)
        self.assertEqual(int(indices[0]), 0)
        self.assertEqual(int(indices[-1]), 49_569_791)
        self.assertTrue(bool((indices[1:] > indices[:-1]).all()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--run-name", default="weight-survey")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    if args.self_test:
        unittest.main(argv=["survey.py"], verbosity=2)
    elif args.root:
        main_survey(args.root, args.run_name)
    else:
        parser.error("--root or --self-test required")