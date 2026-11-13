from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

from artifacts import physical_copy, sha256, write_json


def build_view(root, source_roots):
    root = Path(root).resolve(strict=True)
    destination = root / "dataset_view_complete"
    destination.mkdir(exist_ok=False)
    required = [f"trunk-{index}" for index in range(93)] + [f"expert-{index}" for index in range(1, 93)]
    records = {}
    for name in required:
        choices = sorted({(source / name).resolve(strict=True) for source in source_roots if (source / name).is_dir()})
        if len(choices) != 1:
            raise ValueError(f"need exactly one source for {name}, found {len(choices)}")
        original = choices[0].resolve(strict=True)
        (destination / name).symlink_to(original, target_is_directory=True)
        record = {"source": str(original), "view": str(destination / name)}
        if name.startswith("trunk-"):
            record["index_sha256"] = sha256(original / "index.bin")
        else:
            record["data_files"] = {path.name: path.stat().st_size for path in original.iterdir() if path.is_file()}
        records[name] = record
    write_json(root / "DATASET_VIEW.json", {"read_only_backbone_view": True, "roots": [str(path) for path in source_roots], "entries": records})
    print(json.dumps({"trunks": 93, "expert_layers": 92, "view": str(destination), "source_files_moved": False}), flush=True)


def setup(root, library, weights):
    root = Path(root).resolve(strict=True)
    if weights.stat().st_size < 1_000_000_000 or library.stat().st_size < 10000:
        raise ValueError("runtime library or final-tail weights look like placeholders")
    files = [
        physical_copy(library, root / "runtime" / "libk3_cpu_experts.so"),
        physical_copy(weights, root / "runtime" / "tail.safetensors"),
    ]
    write_json(root / "RUNTIME_INPUTS.json", {"files": files})
    print(json.dumps({"runtime_files_copied": len(files), "bytes": sum(item["bytes"] for item in files)}), flush=True)


def capture(root, dataset, windows, cap, tag=None):
    os.environ["K3_CPU_LAYERS"] = "all"
    os.environ["CLOVER_FORCE_CPU"] = "1"
    os.environ["CLOVER_LOCK_TRUNK"] = "0"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["GOMP_SPINCOUNT"] = "0"
    import numpy as np
    import torch
    from safetensors import safe_open

    root = Path(root).resolve(strict=True)
    name = tag or f"capture-{windows}-{cap}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("capture output must be a simple name")
    output = root / name
    output.mkdir(exist_ok=False)
    absent = [str(dataset / f"trunk-{index}" / "index.bin") for index in range(93) if not (dataset / f"trunk-{index}" / "index.bin").is_file()]
    if absent:
        write_json(output / "BLOCKER.json", {"missing_trunk_indexes": absent})
        raise ValueError(f"missing {len(absent)} of 93 required trunk indexes")
    for item in json.loads((root / "RUNTIME_INPUTS.json").read_text())["files"]:
        if sha256(item["copy"]) != item["sha256"]:
            raise ValueError("copied Kimi runtime input changed")
    data = torch.load(root / "inputs" / "dataset-staged-32-window-inputs.pt", weights_only=True, map_location="cpu")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    library = ctypes.CDLL(str(root / "runtime" / "libk3_cpu_experts.so"))
    float_pointer = ctypes.POINTER(ctypes.c_float)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    library.k3_full93_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    library.k3_full93_init.restype = ctypes.c_int
    library.k3_full93_forward.argtypes = [float_pointer, ctypes.c_int, float_pointer, float_pointer, int_pointer]
    library.k3_full93_forward.restype = ctypes.c_float
    started = time.perf_counter()
    status = library.k3_full93_init(str(dataset).encode(), 64, 16)
    if status != 0:
        raise ValueError(f"Kimi engine initialization failed: {status}")
    print(json.dumps({"engine_initialized": True, "seconds": time.perf_counter() - started}), flush=True)
    with safe_open(root / "runtime" / "tail.safetensors", framework="pt", device="cpu") as stored:
        embeddings = stored.get_tensor("language_model.model.embed_tokens.weight")
        norm = stored.get_tensor("language_model.model.norm.weight").float()
        residual_norm = stored.get_tensor("language_model.model.output_attn_res_norm.weight").float()
        residual_projection = stored.get_tensor("language_model.model.output_attn_res_proj.weight").float().reshape(-1)
        records = []
        for index in range(windows):
            token_ids = data["p_kpad"][index]
            token_ids = token_ids[token_ids != 163585][:cap]
            if len(token_ids) < 2:
                raise ValueError("capture window has insufficient real tokens")
            inputs = embeddings[token_ids].float().numpy().copy()
            hidden = np.zeros_like(inputs)
            snapshots = np.zeros((8, len(token_ids), 7168), dtype=np.float32)
            counts = np.zeros(92, dtype=np.int32)
            elapsed = library.k3_full93_forward(inputs.ctypes.data_as(float_pointer), len(token_ids), hidden.ctypes.data_as(float_pointer), snapshots.ctypes.data_as(float_pointer), counts.ctypes.data_as(int_pointer))
            if elapsed < 0 or not np.isfinite(hidden).all() or not np.isfinite(snapshots).all() or not np.all(counts > 0):
                write_json(output / "BLOCKER.json", {"window": index, "elapsed": elapsed, "expert_counts": counts.tolist(), "finite": bool(np.isfinite(hidden).all())})
                raise ValueError("full Kimi capture failed finiteness or all-layer routing coverage")
            sources = torch.cat([torch.from_numpy(snapshots), torch.from_numpy(hidden)[None]], dim=0)
            normalized = sources / torch.sqrt(sources.square().mean(-1, keepdim=True) + 1e-5) * residual_norm
            scores = normalized @ residual_projection
            aggregate = (sources * scores.softmax(0)[:, :, None]).sum(0)
            final = aggregate / torch.sqrt(aggregate.square().mean(-1, keepdim=True) + 1e-5) * norm
            np.savez_compressed(output / f"window-{index:02d}.npz", hidden=final[:-1].numpy(), target=inputs[1:], input_ids=token_ids.numpy(), expert_counts=counts)
            records.append({"window": index, "tokens": len(token_ids), "milliseconds": elapsed, "routed_layers": int((counts > 0).sum()), "sha256": sha256(output / f"window-{index:02d}.npz")})
            print(json.dumps(records[-1]), flush=True)
    write_json(output / "MANIFEST.json", {"records": records, "dataset": str(dataset), "layer_mode": "all", "library_sha256": sha256(root / "runtime" / "libk3_cpu_experts.so"), "scope": "existing CPU engine state capture, not independent official-runtime parity"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--tag")
    parser.add_argument("--view-root", type=Path, action="append")
    args = parser.parse_args()
    if args.view_root:
        build_view(args.root, args.view_root)
    elif args.setup and args.library and args.weights:
        setup(args.root, args.library, args.weights)
    elif args.dataset:
        if not 1 <= args.windows <= 32 or not 2 <= args.cap <= 48:
            parser.error("capture dimensions exceed bounds")
        capture(args.root, args.dataset, args.windows, args.cap, args.tag)
    else:
        parser.error("choose --setup with runtime paths, or --dataset")