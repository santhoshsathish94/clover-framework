from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def physical_copy(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    if destination.exists() or destination.resolve() == source:
        raise FileExistsError(destination)
    before = source.stat()
    digest = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as target, source.open("rb") as original:
        shutil.copyfileobj(original, target, 8 * 1024 * 1024)
    after = source.stat()
    copied = destination.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source changed during copying")
    if (after.st_dev, after.st_ino) == (copied.st_dev, copied.st_ino):
        raise ValueError("copy shares source inode")
    if sha256(destination) != digest or sha256(source) != digest:
        raise ValueError("copy/source checksum mismatch")
    return {"source": str(source), "copy": str(destination.resolve()), "bytes": copied.st_size, "sha256": digest, "distinct_inode": True}


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


def tensor_hash(tensor):
    import torch

    return hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def statistics(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("empty or nonfinite tensor sample")
    return {"count": values.size, "mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values * values))), "min": float(values.min()), "max": float(values.max())}


def observe(source, root):
    import torch

    root = Path(root).resolve(strict=True)
    destination = root / "baseline" / Path(source).name
    copy_record = physical_copy(source, destination)
    torch.set_num_threads(2)
    state = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
    rows = []
    for name, tensor in tensor_leaves(state):
        flattened = tensor.detach().reshape(-1)
        stride = max(1, math.ceil(flattened.numel() / 65536))
        row = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_hash(tensor), "sample_stride": stride}
        if flattened.numel():
            row["sample"] = statistics(flattened[::stride].float().numpy())
        if flattened.numel() <= 8:
            row["values"] = flattened.tolist()
        rows.append(row)
    metadata = {name: state[name] for name in ("model_id", "hidden_size", "vocab_size", "special_token_ids", "meta") if name in state}
    report = {"copy": copy_record, "metadata": metadata, "tensors": rows}
    write_json(root / "OBSERVATION.json", report)
    print(json.dumps({"tensors": len(rows), "metadata": metadata, "copy_sha256": copy_record["sha256"]}), flush=True)


class ArtifactTests(unittest.TestCase):
    def test_independent_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"original")
            destination = root / "copy"
            result = physical_copy(source, destination)
            self.assertTrue(result["distinct_inode"])
            destination.write_bytes(b"modified")
            self.assertEqual(source.read_bytes(), b"original")
            with self.assertRaises(FileExistsError):
                physical_copy(source, destination)

    def test_numerical_summary(self):
        result = statistics([-2, 0, 2])
        self.assertEqual(result["mean"], 0)
        self.assertAlmostEqual(result["rms"], math.sqrt(8 / 3))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            statistics([float("nan")])

    def test_nemotron_non_degradation_gate(self):
        from nemotron import gate

        baseline = [{"ntp": 2.0, "text_ntp": 1.0, "cos_std": 0.5, "cos_nat": 0.6, "correct_positions": [1, 2]}]
        better = [dict(baseline[0], ntp=1.9)]
        self.assertTrue(gate(better, baseline))
        self.assertFalse(gate([dict(better[0], correct_positions=[1])], baseline))
        self.assertFalse(gate([dict(better[0], text_ntp=1.1)], baseline))
        self.assertFalse(gate(baseline, baseline))
        self.assertFalse(gate(better, baseline, require_significance=True))
        self.assertTrue(gate(better * 5, baseline * 5, require_significance=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.source and args.root:
        observe(args.source, args.root)
    else:
        parser.error("choose --self-test or --source with --root")