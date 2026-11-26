from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import unittest
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: object) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def relative_frobenius_delta(current, reference) -> dict[str, float]:
    import numpy as np

    cur = np.asarray(current, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cur.shape != ref.shape or not np.isfinite(cur).all() or not np.isfinite(ref).all():
        raise ValueError("invalid tensor shapes or nonfinite values")
    diff = cur - ref
    ref_norm = float(np.linalg.norm(ref.reshape(-1)))
    diff_norm = float(np.linalg.norm(diff.reshape(-1)))
    cur_norm = float(np.linalg.norm(cur.reshape(-1)))
    cos = float(np.dot(cur.reshape(-1), ref.reshape(-1)) / max(cur_norm * ref_norm, 1e-12))
    return {
        "ref_fro": ref_norm,
        "cur_fro": cur_norm,
        "diff_fro": diff_norm,
        "rel_fro": diff_norm / max(ref_norm, 1e-12),
        "cosine": cos,
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
    }


def build_disjoint_splits(data_path: Path, seed: int = 1234) -> dict[str, list[str]]:
    with Path(data_path).open("r", encoding="utf-8") as stream:
        lines = [line if line.endswith("\n") else line + "\n" for line in stream if line.strip()]
    if len(lines) < 2500:
        raise ValueError("expected full HybridSFT dataset with >= 2500 trajectories")
    order = list(range(len(lines)))
    random.Random(seed).shuffle(order)
    eval_32_idx = sorted(order[:32])
    train_2400_idx = set(order[32 : 32 + 2400])
    unseen_pool = order[32 + 2400 :]
    discovery_24_idx = sorted(unseen_pool[:24])
    heldout_32_idx = sorted(unseen_pool[24:56])
    if set(eval_32_idx) & train_2400_idx or set(discovery_24_idx) & train_2400_idx or set(heldout_32_idx) & train_2400_idx:
        raise ValueError("split overlaps training indices")
    if set(eval_32_idx) & set(discovery_24_idx) or set(eval_32_idx) & set(heldout_32_idx) or set(discovery_24_idx) & set(heldout_32_idx):
        raise ValueError("evaluation splits overlap each other")
    return {
        "eval32": [lines[i] for i in eval_32_idx],
        "discovery24": [lines[i] for i in discovery_24_idx],
        "heldout32": [lines[i] for i in heldout_32_idx],
    }


def map_c2_to_orig_key(c2_key: str) -> str:
    if not c2_key.startswith("model."):
        raise ValueError(f"unexpected C2 tensor prefix: {c2_key}")
    return "model.language_model." + c2_key[len("model.") :]


def inspect_c2_weights(root: Path) -> dict:
    import torch
    from safetensors import safe_open

    orig_path = root / "original" / "model.safetensors"
    c2_dir = root / "c2c3_weights" / "c2"
    c2_model_path = c2_dir / "model.safetensors"
    c2_adapter_path = c2_dir / "adapter.pt"

    layer_deltas = []
    with safe_open(orig_path, framework="pt", device="cpu") as orig_sf, safe_open(
        c2_model_path, framework="pt", device="cpu"
    ) as c2_sf:
        orig_keys = set(orig_sf.keys())
        c2_keys = sorted(c2_sf.keys())
        unmapped_c2 = [k for k in c2_keys if map_c2_to_orig_key(k) not in orig_keys]
        if unmapped_c2:
            raise ValueError(f"unmapped C2 keys: {unmapped_c2}")
        for c2_key in c2_keys:
            orig_key = map_c2_to_orig_key(c2_key)
            t_orig = orig_sf.get_tensor(orig_key).float().numpy()
            t_c2 = c2_sf.get_tensor(c2_key).float().numpy()
            stats = relative_frobenius_delta(t_c2, t_orig)
            stats["name"] = c2_key
            stats["orig_name"] = orig_key
            stats["shape"] = list(t_c2.shape)
            layer_deltas.append(stats)

        emb_orig = orig_sf.get_tensor("model.language_model.embed_tokens.weight").float()
        emb_c2 = c2_sf.get_tensor("model.embed_tokens.weight").float()
        row_diff = (emb_c2 - emb_orig).norm(dim=1)
        changed_rows = int((row_diff > 0).sum().item())
        top_row_vals, top_row_ids = torch.topk(row_diff, 15)
        top_changed_rows = [
            {"token_id": int(tid.item()), "l2_diff": float(val.item())}
            for val, tid in zip(top_row_vals, top_row_ids)
        ]
        special_rows = {}
        for tid, label in (
            (124893, "<|pad|>"),
            (124894, "<|bos|>"),
            (124900, "<|im_end|>"),
            (125033, "<latent>"),
            (125034, "</latent>"),
        ):
            special_rows[f"{tid}:{label}"] = {
                "orig_rms": float(emb_orig[tid].square().mean().sqrt().item()),
                "c2_rms": float(emb_c2[tid].square().mean().sqrt().item()),
                "l2_diff": float(row_diff[tid].item()),
                "cosine": float(
                    torch.dot(emb_orig[tid], emb_c2[tid])
                    / (emb_orig[tid].norm() * emb_c2[tid].norm() + 1e-12)
                ),
            }

    adapter = torch.load(c2_adapter_path, map_location="cpu", weights_only=True)
    adapter_stats = {}
    for name, tensor in adapter.items():
        arr = tensor.float()
        adapter_stats[name] = {
            "shape": list(arr.shape),
            "dtype": str(tensor.dtype),
            "mean": float(arr.mean().item()),
            "std": float(arr.std().item()),
            "rms": float(arr.square().mean().sqrt().item()),
            "min": float(arr.min().item()),
            "max": float(arr.max().item()),
        }

    top_changed = sorted(layer_deltas, key=lambda item: item["rel_fro"], reverse=True)[:25]
    least_changed = sorted(layer_deltas, key=lambda item: item["rel_fro"])[:10]
    unchanged_count = sum(1 for item in layer_deltas if item["diff_fro"] == 0.0)
    return {
        "common_tensor_count": len(c2_keys),
        "only_in_original_omni_count": len(orig_keys) - len(c2_keys),
        "only_in_c2_count": len(unmapped_c2),
        "unchanged_common_tensors": unchanged_count,
        "top_changed_tensors": top_changed,
        "least_changed_tensors": least_changed,
        "all_tensor_deltas": layer_deltas,
        "embedding_changed_rows": changed_rows,
        "top_changed_embedding_rows": top_changed_rows,
        "special_embedding_rows": special_rows,
        "adapter_tensors": adapter_stats,
    }


def run_diagnosis(root: Path, data_path: Path) -> None:
    experiment = root / "c2_improvement"
    if (experiment / "WEIGHT_DIAGNOSIS.json").exists():
        raise FileExistsError(experiment / "WEIGHT_DIAGNOSIS.json")
    os.environ["HF_HOME"] = str(experiment / "cache" / "huggingface")
    os.environ["HF_MODULES_CACHE"] = str(experiment / "cache" / "modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "8"

    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    weight_report = inspect_c2_weights(root)
    write_json(experiment / "WEIGHT_DIAGNOSIS.json", weight_report)

    splits = build_disjoint_splits(data_path, seed=1234)
    split_dir = experiment / "splits"
    split_dir.mkdir(exist_ok=False)
    split_hashes = {}
    for name, lines in splits.items():
        path = split_dir / f"{name}.jsonl"
        path.write_text("".join(lines), encoding="utf-8")
        split_hashes[name] = {"count": len(lines), "sha256": file_hash(path)}
    write_json(experiment / "SPLITS.json", split_hashes)
    print(
        json.dumps(
            {
                "common_tensors": weight_report["common_tensor_count"],
                "unchanged_tensors": weight_report["unchanged_common_tensors"],
                "embedding_changed_rows": weight_report["embedding_changed_rows"],
                "adapter_post_ln_weight_rms": weight_report["adapter_tensors"]["post_ln.weight"]["rms"],
                "adapter_post_ln_weight_min": weight_report["adapter_tensors"]["post_ln.weight"]["min"],
                "adapter_post_ln_weight_max": weight_report["adapter_tensors"]["post_ln.weight"]["max"],
                "splits": split_hashes,
            }
        ),
        flush=True,
    )


class DiagnoseTests(unittest.TestCase):
    def test_map_c2_to_orig_key(self):
        self.assertEqual(
            map_c2_to_orig_key("model.embed_tokens.weight"),
            "model.language_model.embed_tokens.weight",
        )
        with self.assertRaisesRegex(ValueError, "prefix"):
            map_c2_to_orig_key("lm_head.weight")

    def test_relative_frobenius_delta_zero_and_scaled(self):
        zero = relative_frobenius_delta([1.0, 2.0], [1.0, 2.0])
        self.assertEqual(zero["diff_fro"], 0.0)
        self.assertAlmostEqual(zero["cosine"], 1.0)
        scaled = relative_frobenius_delta([2.0, 4.0], [1.0, 2.0])
        self.assertAlmostEqual(scaled["rel_fro"], 1.0)
        self.assertAlmostEqual(scaled["cosine"], 1.0)

    def test_disjoint_splits(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text("".join(f'{{"id": {i}}}\n' for i in range(2600)), encoding="utf-8")
            splits = build_disjoint_splits(path, seed=1234)
            self.assertEqual(len(splits["eval32"]), 32)
            self.assertEqual(len(splits["discovery24"]), 24)
            self.assertEqual(len(splits["heldout32"]), 32)
            self.assertFalse(set(splits["eval32"]) & set(splits["discovery24"]))
            self.assertFalse(set(splits["eval32"]) & set(splits["heldout32"]))
            self.assertFalse(set(splits["discovery24"]) & set(splits["heldout32"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root and args.data:
        run_diagnosis(args.root.resolve(strict=True), args.data.resolve(strict=True))
    else:
        parser.error("choose --self-test or --root with --data")


if __name__ == "__main__":
    main()
