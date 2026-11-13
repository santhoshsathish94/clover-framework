from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import unittest
from pathlib import Path

from artifacts import physical_copy, sha256, statistics, tensor_hash, tensor_leaves, write_json


def observe(root, checkpoint, samples):
    import torch

    root = Path(root).resolve(strict=True)
    torch.set_num_threads(2)
    checkpoint_copy = root / "baseline" / checkpoint.name
    checkpoint_record = physical_copy(checkpoint, checkpoint_copy)
    data_copy = root / "inputs" / samples.name
    data_record = physical_copy(samples, data_copy)
    state = torch.load(checkpoint_copy, map_location="cpu", weights_only=True, mmap=True)
    if "k3_hybridlink" not in state:
        raise ValueError("combined checkpoint has no native K3 HybridLink")
    link = state["k3_hybridlink"]
    output = []
    for name, tensor in tensor_leaves(link):
        flat = tensor.reshape(-1)
        stride = max(1, (flat.numel() + 65535) // 65536)
        row = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_hash(tensor), "sample_stride": stride, "sample": statistics(flat[::stride].float().numpy())}
        if tensor.numel() <= 8:
            row["values"] = flat.tolist()
        output.append(row)
    data = torch.load(data_copy, map_location="cpu", weights_only=True)
    shapes = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in tensor_leaves(data)}
    write_json(root / "OBSERVATION.json", {"checkpoint": checkpoint_record, "data": data_record, "link_tensors": output, "data_tensors": shapes, "data_keys": list(data) if isinstance(data, dict) else None})
    print(json.dumps({"native_link_tensors": len(output), "data_tensors": shapes, "data_keys": list(data) if isinstance(data, dict) else None}), flush=True)


def native_class(source):
    import torch

    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "K3NativeHybridLink"]
    if len(definitions) != 1:
        raise ValueError("expected one native K3 HybridLink definition")
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    namespace = {"torch": torch, "nn": torch.nn, "D_K3": 7168}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["K3NativeHybridLink"]


def freeze_capture(root):
    root = Path(root).resolve(strict=True)
    frozen = root / "frozen"
    frozen.mkdir(exist_ok=False)
    protocol = {
        "capture_windows": 12,
        "tokens_per_window": 8,
        "discovery_windows": [0, 1, 2, 3],
        "validation_windows": [4, 5, 6, 7, 8, 9, 10, 11],
        "policies": {"original": 1.0, "gain092": 0.92, "gain108": 1.08},
        "edit": "k3_hybridlink.out_scale only",
        "gate": "strictly lower mean per-window MSE; no per-window MSE/cosine regression; beat zero-input MSE on every validation window and >=5 improving windows",
        "provenance_limit": "reserved from this calibration; original training membership of staged windows is unknown",
        "scope": "existing all-layer CPU engine captures and next-embedding prediction, not completed task or end-to-end quality",
        "deployment_eligible": False,
    }
    write_json(frozen / "PROTOCOL.json", protocol)
    paths = [Path(__file__), Path(__file__).parent / "artifacts.py", Path(__file__).parent / "capture_kimi.py", root / "OBSERVATION.json", root / "DATASET_VIEW.json", root / "RUNTIME_INPUTS.json", root / "native_source.py", frozen / "PROTOCOL.json"]
    write_json(frozen / "MANIFEST.json", {str(path.resolve()): sha256(path) for path in paths})
    print(json.dumps(protocol), flush=True)


def compare_windows(candidate, original, zero, significance=False):
    if len(candidate) != len(original) or len(zero) != len(original) or not original:
        raise ValueError("invalid Kimi paired records")
    for new, old, control in zip(candidate, original, zero):
        if new["window"] != old["window"] or old["window"] != control["window"]:
            raise ValueError("Kimi window identity mismatch")
        if new["mse"] > old["mse"] + 1e-10 or new["cosine"] < old["cosine"] - 1e-6 or new["mse"] >= control["mse"]:
            return False
    improved = sum(new["mse"] < old["mse"] - 1e-10 for new, old in zip(candidate, original))
    lower_mean = sum(row["mse"] for row in candidate) < sum(row["mse"] for row in original) - 1e-9
    return lower_mean and (not significance or 2.0 ** (-improved) <= 0.05)


def trial(root):
    import copy
    import numpy as np
    import torch
    import torch.nn.functional as functional

    root = Path(root).resolve(strict=True)
    for path, expected in json.loads((root / "frozen" / "MANIFEST.json").read_text()).items():
        if sha256(path) != expected:
            raise ValueError(f"Kimi frozen source changed: {path}")
    protocol = json.loads((root / "frozen" / "PROTOCOL.json").read_text())
    captures = root / "capture-calibration"
    capture_manifest = json.loads((captures / "MANIFEST.json").read_text())
    if len(capture_manifest["records"]) != protocol["capture_windows"]:
        raise ValueError("incomplete Kimi capture set")
    for record in capture_manifest["records"]:
        if record["routed_layers"] != 92 or sha256(captures / f"window-{record['window']:02d}.npz") != record["sha256"]:
            raise ValueError("invalid Kimi full-layer capture record")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    observation = json.loads((root / "OBSERVATION.json").read_text())
    source = Path(observation["checkpoint"]["copy"])
    if sha256(source) != observation["checkpoint"]["sha256"]:
        raise ValueError("Kimi baseline changed")
    state = torch.load(source, weights_only=True, map_location="cpu", mmap=True)
    model = native_class(root / "native_source.py")().eval().requires_grad_(False)
    before = {name: tensor_hash(tensor) for name, tensor in tensor_leaves(state)}
    results = root / "results"
    results.mkdir(exist_ok=False)

    def evaluate(windows, zero=False):
        rows = []
        for index in windows:
            with np.load(captures / f"window-{index:02d}.npz", allow_pickle=False) as sample:
                hidden = torch.from_numpy(sample["hidden"].copy())
                target = torch.from_numpy(sample["target"].copy())
            prediction = torch.zeros_like(target) if zero else model(hidden)
            mse = float(functional.mse_loss(prediction, target))
            cosine = float(functional.cosine_similarity(prediction, target, dim=-1).mean())
            if not np.isfinite([mse, cosine]).all():
                raise ValueError("nonfinite Kimi calibration result")
            rows.append({"window": index, "positions": hidden.shape[0], "mse": mse, "cosine": cosine, "prediction_rms": float(prediction.square().mean().sqrt())})
        return rows

    paths = {}
    discovery = {}
    zero_discovery = evaluate(protocol["discovery_windows"], zero=True)
    write_json(results / "discovery-zero.json", zero_discovery)
    for name, factor in protocol["policies"].items():
        destination = root / "trials" / name / "checkpoint.pt"
        record = physical_copy(source, destination)
        altered = copy.deepcopy(state)
        altered["k3_hybridlink"]["out_scale"] = altered["k3_hybridlink"]["out_scale"] * factor
        if factor != 1.0:
            temporary = destination.with_name("changed.pt")
            torch.save(altered, temporary)
            temporary.replace(destination)
        loaded = torch.load(destination, weights_only=True, map_location="cpu", mmap=True)
        after = {key: tensor_hash(tensor) for key, tensor in tensor_leaves(loaded)}
        changed = [key for key in before if before[key] != after.get(key)]
        if set(before) != set(after) or set(changed) != (set() if factor == 1.0 else {"k3_hybridlink.out_scale"}):
            raise ValueError("Kimi trial changed undeclared tensors")
        write_json(destination.parent / "PATCH_AUDIT.json", {"factor": factor, "copy": record, "changed_tensors": changed, "before": before, "after": after, "sha256": sha256(destination)})
        model.load_state_dict(loaded["k3_hybridlink"], strict=True)
        discovery[name] = evaluate(protocol["discovery_windows"])
        paths[name] = destination
        write_json(results / f"discovery-{name}.json", discovery[name])
    eligible = [name for name in protocol["policies"] if name != "original" and compare_windows(discovery[name], discovery["original"], zero_discovery)]
    selected = min(eligible, key=lambda name: sum(row["mse"] for row in discovery[name])) if eligible else None
    write_json(results / "SELECTION.json", {"selected": selected, "validation_opened": bool(selected)})
    accepted = False
    if selected:
        validation = {}
        zero_validation = evaluate(protocol["validation_windows"], zero=True)
        write_json(results / "validation-zero.json", zero_validation)
        for name in ("original", selected):
            loaded = torch.load(paths[name], weights_only=True, map_location="cpu", mmap=True)
            model.load_state_dict(loaded["k3_hybridlink"], strict=True)
            validation[name] = evaluate(protocol["validation_windows"])
            write_json(results / f"validation-{name}.json", validation[name])
        accepted = compare_windows(validation[selected], validation["original"], zero_validation, significance=True)
    import shutil
    for name, path in paths.items():
        if not (accepted and name == selected):
            temporary = path.with_name("restored.pt")
            shutil.copyfile(source, temporary)
            temporary.replace(path)
            if sha256(path) != observation["checkpoint"]["sha256"]:
                raise ValueError("Kimi reversal failed")
            write_json(path.parent / "REVERSAL.json", {"restored": True, "sha256": sha256(path)})
    write_json(results / "OUTCOME.json", {"selected": selected, "calibration_gate_passed": accepted, "deployment_eligible": False, "baseline": str(source), "candidate": str(paths[selected]) if accepted else None, "scope": protocol["scope"]})
    print(json.dumps({"kimi_complete": True, "selected": selected, "calibration_gate_passed": accepted, "deployment_eligible": False}), flush=True)


class KimiGateTests(unittest.TestCase):
    def test_zero_and_regression_controls(self):
        original = [{"window": 0, "mse": 0.2, "cosine": 0.4}]
        better = [{"window": 0, "mse": 0.1, "cosine": 0.4}]
        zero = [{"window": 0, "mse": 0.3, "cosine": 0.0}]
        self.assertTrue(compare_windows(better, original, zero))
        self.assertFalse(compare_windows(original, original, zero))
        self.assertFalse(compare_windows([dict(better[0], cosine=0.3)], original, zero))
        self.assertFalse(compare_windows(better, original, [dict(zero[0], mse=0.05)]))
        self.assertFalse(compare_windows(better, original, zero, significance=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root and args.freeze:
        freeze_capture(args.root)
    elif args.root and args.trial:
        trial(args.root)
    elif args.root and args.checkpoint and args.samples:
        observe(args.root, args.checkpoint, args.samples)
    else:
        parser.error("choose --self-test or a study operation with --root")