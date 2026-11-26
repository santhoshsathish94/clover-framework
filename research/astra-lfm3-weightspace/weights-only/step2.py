import argparse
import json
from pathlib import Path
import time
import unittest

import numpy as np
import torch
from safetensors import safe_open
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from step1 import EXPECTED_SHA256, SEED, TENSOR, closed_boundary, sha256


NORM_CUTOFF = 0.25
THRESHOLDS = (0.90, 0.95, 0.99)


def numeric_summary(values):
    probabilities = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
    return {"probabilities": probabilities,
            "quantiles": np.quantile(values, probabilities).tolist()}


def normalized_rows(matrix):
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.isfinite(matrix).all() or np.any(norms == 0):
        raise ValueError("Rows must be finite and nonzero")
    return (matrix / norms[:, None]).astype(np.float32), norms


def graph_groups(similarity, threshold):
    adjacency = similarity >= threshold
    np.fill_diagonal(adjacency, False)
    pair_count = int(np.count_nonzero(adjacency)) // 2
    count, labels = connected_components(csr_matrix(adjacency), directed=False, return_labels=True)
    sizes = np.bincount(labels)
    order = sorted(range(count), key=lambda index: (-int(sizes[index]), index))
    return labels, sizes, order, pair_count


def centroid_geometry(unit_rows):
    mean = unit_rows.astype(np.float64).mean(0)
    length = float(np.linalg.norm(mean))
    count = len(unit_rows)
    pair_mean = (count * length ** 2 - 1) / (count - 1) if count > 1 else 1.0
    result = {"normalized_mean_length": length, "mean_all_off_diagonal_cosine": pair_mean}
    if length < 1e-12:
        result["direction_defined"] = False
        return result, None
    direction = mean / length
    cosines = unit_rows @ direction
    result.update({
        "direction_defined": True, "cosine_to_centroid": numeric_summary(cosines),
        "fraction_with_centroid_cosine_at_least": {
            str(threshold): float(np.mean(cosines >= threshold)) for threshold in THRESHOLDS
        },
    })
    return result, direction


def analyze_population(matrix, row_ids, thresholds=THRESHOLDS):
    units, norms = normalized_rows(matrix)
    similarity = (torch.from_numpy(units) @ torch.from_numpy(units).T).numpy()
    np.clip(similarity, -1, 1, out=similarity)
    population, direction = centroid_geometry(units)
    population.update({"rows": len(row_ids), "row_norms": numeric_summary(norms)})
    generator = np.random.default_rng(SEED)
    first = generator.integers(0, len(row_ids), 65_536)
    second = (first + generator.integers(1, len(row_ids), 65_536)) % len(row_ids)
    population["sampled_pair_cosines"] = numeric_summary(similarity[first, second])
    population["pair_quantile_method"] = "65536 seeded uniform ordered non-self pairs; graph counts use every pair"
    population["thresholds"] = {}
    arrays = {"row_ids": row_ids, "row_norms": norms}
    if direction is not None:
        arrays["global_centroid_direction"] = direction
    for threshold in thresholds:
        labels, sizes, order, pair_count = graph_groups(similarity, threshold)
        top_groups = []
        for label in order[:8]:
            members = np.flatnonzero(labels == label)
            group, group_direction = centroid_geometry(units[members])
            minimum_pair_cosine = float(similarity[np.ix_(members, members)].min())
            group.update({
                "label": int(label), "size": len(members),
                "row_ids_first16": row_ids[members[:16]].tolist(),
                "row_id_min": int(row_ids[members].min()), "row_id_max": int(row_ids[members].max()),
                "row_norms": numeric_summary(norms[members]),
                "minimum_within_group_cosine": minimum_pair_cosine,
                "all_pairs_meet_edge_threshold": minimum_pair_cosine >= threshold - 1e-6,
                "rows_in_last4096_table_positions": int(np.count_nonzero(row_ids[members] >= 128000 - 4096)),
            })
            top_groups.append(group)
            if group_direction is not None:
                arrays[f"centroid_{threshold}_{label}"] = group_direction
        population["thresholds"][str(threshold)] = {
            "connected_components": len(sizes), "singleton_rows": int(np.count_nonzero(sizes == 1)),
            "largest_component_sizes": [int(sizes[label]) for label in order[:12]],
            "pairs_meeting_threshold": pair_count,
            "fraction_all_pairs_meeting_threshold": pair_count / (len(row_ids) * (len(row_ids) - 1) / 2),
            "top_groups": top_groups,
            "qualification": "Connectivity is single linkage; minimum pair similarity checks whether a connected group is actually compact.",
        }
        arrays[f"labels_{threshold}"] = labels
    return population, arrays


def run(root):
    root = Path(root)
    boundary = closed_boundary(root)
    checkpoint = root / "original" / "model.safetensors"
    if sha256(checkpoint) != EXPECTED_SHA256:
        raise ValueError("Original checkpoint identity mismatch")
    previous_path = root / "weights-only" / "step1" / "arrays.npz"
    with np.load(previous_path, allow_pickle=False) as previous:
        all_norms = previous["row_norms"]
    low_ids = np.flatnonzero(all_norms < NORM_CUTOFF)
    if len(low_ids) != 8440:
        raise ValueError("Step1 low-norm population changed")
    other_ids = np.flatnonzero(all_norms >= NORM_CUTOFF)
    reference_ids = np.sort(np.random.default_rng(SEED).choice(other_ids, size=len(low_ids), replace=False))
    output = root / "weights-only" / "step2"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        tensor = source.get_tensor(TENSOR)
    if tensor.shape != (128000, 2048) or tensor.dtype != torch.bfloat16:
        raise ValueError("Unexpected tensor header")
    report = {
        "evidence_mode": "weights and weights-derived step1 norms only; no inference, labels, activations or benchmarks",
        "boundary": boundary, "tensor": TENSOR, "norm_cutoff": NORM_CUTOFF,
        "similarity_thresholds": thresholds_to_list(), "seed": SEED,
        "step1_arrays_sha256": sha256(previous_path), "checkpoint_sha256_before": EXPECTED_SHA256,
        "populations": {},
        "limits": ["Norm cutoff is exploratory, inherited from step1",
                   "Similarity thresholds describe geometry, not semantic equivalence or defects",
                   "No tokenizer labels or behavioral data were accessed",
                   "The reference population is sampled; the low-norm population and within-population graph counts are exhaustive"],
    }
    all_arrays = {}
    for name, row_ids in (("all_low_norm_rows", low_ids), ("other_rows_reference", reference_ids)):
        matrix = tensor[torch.from_numpy(row_ids)].float().numpy()
        actual_norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(actual_norms, all_norms[row_ids], rtol=2e-6, atol=1e-8):
            raise ValueError("Saved step1 norms disagree with the actual checkpoint")
        population, arrays = analyze_population(matrix, row_ids)
        report["populations"][name] = population
        all_arrays.update({f"{name}/{key}": value for key, value in arrays.items()})
        print(json.dumps({"population": name, "rows": len(row_ids),
                          "mean_pair_cosine": population["mean_all_off_diagonal_cosine"],
                          "largest_groups": {key: value["largest_component_sizes"][:5]
                                             for key, value in population["thresholds"].items()},
                          "elapsed_s": time.monotonic() - started}), flush=True)
    np.savez_compressed(output / "arrays.npz", **all_arrays)
    report["elapsed_s"] = time.monotonic() - started
    report["checkpoint_sha256_after"] = sha256(checkpoint)
    if report["checkpoint_sha256_after"] != EXPECTED_SHA256:
        raise ValueError("Original checkpoint changed")
    report["arrays_sha256"] = sha256(output / "arrays.npz")
    with (output / "result.json").open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
    print("WEIGHTS_ONLY_STEP2_COMPLETE " + json.dumps({"elapsed_s": report["elapsed_s"],
                                                     "original_unchanged": True}), flush=True)


def thresholds_to_list():
    return list(THRESHOLDS)


class FamilyTests(unittest.TestCase):
    def test_separate_families(self):
        matrix = np.array([[1, 0]] * 5 + [[0, 1]] * 3 + [[-1, 0]], dtype=np.float32)
        report, _ = analyze_population(matrix, np.arange(9), thresholds=(0.99,))
        groups = report["thresholds"]["0.99"]
        self.assertEqual(groups["largest_component_sizes"], [5, 3, 1])
        self.assertEqual(groups["singleton_rows"], 1)
        self.assertEqual(groups["pairs_meeting_threshold"], 13)

    def test_chain_is_not_compact(self):
        angles = np.deg2rad([0, 30, 60])
        matrix = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
        report, _ = analyze_population(matrix, np.arange(3), thresholds=(0.80,))
        group = report["thresholds"]["0.8"]["top_groups"][0]
        self.assertEqual(group["size"], 3)
        self.assertAlmostEqual(group["minimum_within_group_cosine"], 0.5, places=6)
        self.assertFalse(group["all_pairs_meet_edge_threshold"])

    def test_norm_does_not_define_direction(self):
        matrix = np.array([[0.1, 0], [10, 0], [0, 1]], dtype=np.float32)
        report, _ = analyze_population(matrix, np.arange(3), thresholds=(0.99,))
        self.assertEqual(report["thresholds"]["0.99"]["largest_component_sizes"], [2, 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    if args.self_test:
        unittest.main(argv=["step2.py"], verbosity=2)
    elif args.root:
        run(args.root)
    else:
        parser.error("--root or --self-test required")