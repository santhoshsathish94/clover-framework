from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path


def summarize_values(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not finite.all():
        return {"count": int(values.size), "nonfinite": int((~finite).sum())}
    if not values.size:
        return {"count": 0, "nonfinite": 0}
    return {
        "count": int(values.size), "nonfinite": 0,
        "mean": float(values.mean()), "std": float(values.std()),
        "rms": float(np.sqrt(np.mean(values * values))),
        "minimum": float(values.min()), "maximum": float(values.max()),
        "zero_fraction": float(np.mean(values == 0)),
        "absolute_quantiles": dict(zip(("p50", "p90", "p99", "p100"), np.quantile(np.abs(values), [0.5, 0.9, 0.99, 1.0]).tolist())),
    }


def tensor_leaves(value, prefix=""):
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from tensor_leaves(item, prefix + ("." if prefix else "") + str(name))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, prefix + "." + str(index))


def inspect_checkpoint(path):
    import torch
    from safetensors import safe_open

    if path.suffix == ".safetensors":
        source = safe_open(path, framework="pt", device="cpu")
        entries = ((name, source.get_tensor(name)) for name in source.keys())
    else:
        source = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        entries = tensor_leaves(source)
    output = []
    for name, tensor in entries:
        if not tensor.is_floating_point():
            continue
        flat = tensor.detach().reshape(-1)
        step = max(1, math.ceil(flat.numel() / 65536))
        sampled = flat[::step].float().numpy()
        result = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": tensor.numel(), "sample_stride": step, "sample_statistics": summarize_values(sampled)}
        if tensor.ndim == 2:
            row_norms = []
            for start in range(0, tensor.shape[0], 1024):
                row_norms.append(torch.linalg.vector_norm(tensor[start:start + 1024].float(), dim=1))
            norms = torch.cat(row_norms)
            result["row_norms"] = summarize_values(norms.numpy())
            largest = torch.topk(norms, min(5, norms.numel()))
            result["largest_rows"] = [{"index": int(index), "norm": float(norm)} for norm, index in zip(largest.values, largest.indices)]
        if tensor.numel() <= 8:
            result["values"] = flat.float().tolist()
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
        return
    if not args.root or not args.checkpoint or not args.output:
        parser.error("require --root, --checkpoint and --output")
    import torch

    sys.path.insert(0, str(args.root / "composite"))
    from prepare import file_hash, physical_copy, write_json

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    experiment = args.root / "c2c3_weights"
    if not args.output.resolve().is_relative_to(experiment.resolve()):
        raise ValueError("output escapes the experiment")
    inputs = experiment / "inputs"
    inputs.mkdir(exist_ok=False)
    result = {"files": [], "checkpoints": {}}
    for index, path in enumerate(args.checkpoint):
        destination = inputs / f"{index}-{path.name}"
        result["files"].append(physical_copy(path, destination, experiment))
        result["checkpoints"][str(path)] = inspect_checkpoint(destination)
    for record in result["files"]:
        if file_hash(Path(record["source"])) != record["sha256"]:
            raise ValueError("source changed during observation")
    write_json(args.output, result)
    print(json.dumps({"checkpoints": len(result["checkpoints"]), "tensors": {name: len(items) for name, items in result["checkpoints"].items()}, "sources_unchanged": True}), flush=True)


class StatisticsTests(unittest.TestCase):
    def test_known_values(self):
        result = summarize_values([-2, 0, 2])
        self.assertEqual(result["mean"], 0.0)
        self.assertAlmostEqual(result["rms"], math.sqrt(8 / 3))
        self.assertAlmostEqual(result["zero_fraction"], 1 / 3)

    def test_nonfinite_not_hidden(self):
        result = summarize_values([1, float("nan"), float("inf")])
        self.assertEqual(result["nonfinite"], 2)

    def test_empty_tensor(self):
        self.assertEqual(summarize_values([])["count"], 0)


if __name__ == "__main__":
    main()