import argparse
from collections import defaultdict
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survey import moments, vector_summary


TENSOR = "model.language_model.embed_tokens.weight"
EXPECTED_SHA256 = "76c0ec2c3e27832fb19f886d5eb0afe37f03c5c32c4a796a7ce08cef119d2763"
SEED = 20261111


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def closed_boundary(root):
    blocked = [root / "reports" / "baseline" / "results.jsonl",
               root / "reproduction" / "CONTEXT.md", root / "fixtures" / "cases.json",
               root / "original" / "tokenizer.json", root / "original" / "config.json"]
    for path in blocked:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EPERM, errno.ENOENT):
                raise
        else:
            os.close(descriptor)
            raise RuntimeError(f"Excluded evidence is accessible: {path}")
    checkpoint = root / "original" / "model.safetensors"
    try:
        descriptor = os.open(checkpoint, os.O_WRONLY)
    except OSError as error:
        if error.errno not in (errno.EROFS, errno.EACCES, errno.EPERM):
            raise
    else:
        os.close(descriptor)
        raise RuntimeError("Original checkpoint is writable")
    return {"known_behavior_and_tokenizer_inputs_inaccessible": True, "checkpoint_readonly": True}


def duplicate_rows(bits):
    first_rows = {}
    groups = defaultdict(list)
    for index, row in enumerate(bits):
        digest = hashlib.sha256(row.tobytes()).digest()
        if digest not in first_rows:
            first_rows[digest] = index
        else:
            first = first_rows[digest]
            if not np.array_equal(bits[first], row):
                raise RuntimeError("Row digest collision")
            if not groups[first]:
                groups[first].append(first)
            groups[first].append(index)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]))


def shuffled_rows(matrix):
    generator = torch.Generator().manual_seed(SEED)
    order = torch.rand(matrix.shape, generator=generator).argsort(dim=1)
    return matrix.gather(1, order)


def centered_spectrum(matrix, rank=16):
    centered = matrix.float() - matrix.float().mean(0, keepdim=True)
    energy = float(centered.double().square().sum())
    if energy == 0:
        return {"energy": 0.0, "top_energy_fractions": [], "singular_values": []}, np.empty((matrix.shape[1], 0))
    torch.manual_seed(SEED)
    _, values, directions = torch.svd_lowrank(centered, q=min(min(matrix.shape), rank + 8), niter=3)
    values = values[:rank]
    fractions = values.double().square() / energy
    return {"energy": energy, "singular_values": values.tolist(),
            "top_energy_fractions": fractions.tolist(),
            "cumulative_energy_fractions": fractions.cumsum(0).tolist(),
            "method": "randomized centered SVD; top16, oversample8, three power iterations",
            "seed": SEED}, directions[:, :len(values)].numpy()


def run(root):
    root = Path(root)
    output = root / "weights-only" / "step1"
    if output.exists():
        raise FileExistsError("Preserve existing step results")
    boundary = closed_boundary(root)
    checkpoint = root / "original" / "model.safetensors"
    if sha256(checkpoint) != EXPECTED_SHA256:
        raise ValueError("Original checkpoint identity changed")
    start = time.monotonic()
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        weight = source.get_tensor(TENSOR)
    if weight.shape != (128000, 2048) or weight.dtype != torch.bfloat16:
        raise ValueError("Unexpected tensor header")
    bits = weight.view(torch.uint8).numpy()
    tensor_digest = hashlib.sha256(bits.tobytes()).hexdigest()
    groups = duplicate_rows(bits)
    statistics = moments(weight)
    if statistics["nonfinite_count"]:
        raise ValueError("Nonfinite values found; do not silently proceed with spectral analysis")
    matrix = weight.float()
    row_norms = torch.linalg.vector_norm(matrix, dim=1)
    column_norms = torch.linalg.vector_norm(matrix, dim=0)
    mean_vector = matrix.mean(0)
    mean_fraction = matrix.shape[0] * float(mean_vector.double().square().sum()) / statistics["frobenius_squared"]
    zero_rows = torch.where(row_norms == 0)[0].tolist()
    generator = torch.Generator().manual_seed(SEED)
    sample_ids = torch.randperm(matrix.shape[0], generator=generator)[:4096]
    sample = matrix[sample_ids]
    control = shuffled_rows(sample)
    if not torch.equal(sample.sort(dim=1).values, control.sort(dim=1).values):
        raise ValueError("Derived shuffle did not preserve row values")
    actual_spectrum, directions = centered_spectrum(sample)
    control_spectrum, _ = centered_spectrum(control)
    normalized = torch.nn.functional.normalize(sample, dim=1)
    cosine = (normalized[:2048] * normalized[2048:]).sum(1)
    neighbors = normalized[:512] @ normalized.T
    neighbors[torch.arange(512), torch.arange(512)] = -2
    neighbor_values, neighbor_indices = neighbors.max(1)
    order = neighbor_values.argsort(descending=True)[:12]
    pairs = [{"row_a": int(sample_ids[index]), "row_b": int(sample_ids[neighbor_indices[index]]),
              "cosine": float(neighbor_values[index])} for index in order]
    output.mkdir()
    np.savez_compressed(output / "arrays.npz", row_norms=row_norms.numpy(), column_norms=column_norms.numpy(),
                        mean_vector=mean_vector.numpy(), sample_row_ids=sample_ids.numpy(),
                        centered_directions=directions)
    duplicate_count = sum(len(group) - 1 for group in groups)
    result = {
        "evidence_mode": "stored weights and tensor header only; no tokenizer labels, activations or forward passes",
        "boundary": boundary, "tensor": TENSOR, "shape": list(weight.shape), "dtype": str(weight.dtype),
        "parameter_count": weight.numel(), "tensor_sha256": tensor_digest,
        "checkpoint_sha256_before": EXPECTED_SHA256, "statistics": statistics,
        "row_norms": vector_summary(row_norms), "column_norms": vector_summary(column_norms),
        "zero_row_count": len(zero_rows), "zero_row_indices_first32": zero_rows[:32],
        "exact_duplicate_group_count": len(groups), "duplicate_rows_beyond_first": duplicate_count,
        "distinct_row_count": matrix.shape[0] - duplicate_count,
        "largest_duplicate_groups": [{"size": len(group), "first_indices": group[:16], "last_index": group[-1]}
                                     for group in groups[:12]],
        "shared_mean_fraction_of_total_squared_norm": mean_fraction,
        "sample_size": len(sample_ids), "sample_without_replacement_seed": SEED,
        "sample_pair_cosine": vector_summary(cosine),
        "sample_nearest_pairs": pairs,
        "sample_centered_spectrum": actual_spectrum, "row_shuffled_control_spectrum": control_spectrum,
        "control_preserves_each_sampled_row_values": True,
        "elapsed_s": time.monotonic() - start,
        "limits": ["An embedding role is suggested by the stored name; runtime tying is not inferred here",
                   "Row numbers are not decoded into token identities or concepts",
                   "Sampled approximate spectra and sampled neighbors are not exhaustive",
                   "Geometric structure and statistical extremes do not establish a defect or behavior"],
    }
    result["checkpoint_sha256_after"] = sha256(checkpoint)
    if result["checkpoint_sha256_after"] != EXPECTED_SHA256:
        raise ValueError("Original checkpoint changed during read-only analysis")
    with (output / "result.json").open("x") as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
    print("WEIGHTS_ONLY_STEP1_COMPLETE " + json.dumps({
        "shape": result["shape"], "zero_rows": len(zero_rows),
        "exact_duplicate_groups": len(groups), "duplicate_rows_beyond_first": duplicate_count,
        "shared_mean_energy_fraction": mean_fraction,
        "observed_first_pc_fraction": actual_spectrum["top_energy_fractions"][0],
        "shuffled_first_pc_fraction": control_spectrum["top_energy_fractions"][0],
        "elapsed_s": result["elapsed_s"],
    }), flush=True)


class WeightOnlyTests(unittest.TestCase):
    def test_exact_duplicates(self):
        rows = np.array([[1, 2], [3, 4], [1, 2], [1, 2], [4, 3]], dtype=np.uint16)
        self.assertEqual(duplicate_rows(rows), [[0, 2, 3]])

    def test_shuffle_preserves_rows_not_alignment(self):
        matrix = torch.arange(128, dtype=torch.float32).reshape(8, 16)
        shuffled = shuffled_rows(matrix)
        self.assertTrue(torch.equal(matrix.sort(1).values, shuffled.sort(1).values))
        self.assertFalse(torch.equal(matrix, shuffled))

    def test_rank_one_control_discriminates(self):
        matrix = torch.outer(torch.arange(64, dtype=torch.float32) - 31.5, torch.arange(16, dtype=torch.float32) - 7.5)
        original, _ = centered_spectrum(matrix, rank=4)
        shuffled, _ = centered_spectrum(shuffled_rows(matrix), rank=4)
        self.assertAlmostEqual(original["top_energy_fractions"][0], 1, places=5)
        self.assertLess(shuffled["top_energy_fractions"][0], 0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    if args.self_test:
        unittest.main(argv=["step1.py"], verbosity=2)
    elif args.root:
        run(args.root)
    else:
        parser.error("--root or --self-test required")