from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import tempfile
import unittest
from pathlib import Path


FILES = {0: "common.bin", 1: "operator.bin", 3: "ffn.bin"}
HEADER = struct.Struct("<8sIII")
ENTRY = struct.Struct("<64sIIIQQIIII")
ORIGINAL_SHA256 = "76c0ec2c3e27832fb19f886d5eb0afe37f03c5c32c4a796a7ce08cef119d2763"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trunk_inventory(directory: Path) -> list[dict]:
    raw = (directory / "index.bin").read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("truncated trunk header")
    magic, version, trunk_id, count = HEADER.unpack_from(raw)
    if magic != b"CLVRTRK1":
        raise ValueError("invalid trunk magic")
    if len(raw) != HEADER.size + count * ENTRY.size:
        raise ValueError("trunk index size disagrees with entry count")
    names = set()
    intervals: dict[int, list[tuple[int, int]]] = {}
    result = []
    for number in range(count):
        name_raw, file_id, dtype, ndim, offset, size, *dimensions = ENTRY.unpack_from(
            raw, HEADER.size + number * ENTRY.size
        )
        name = name_raw.split(b"\0", 1)[0].decode("utf-8")
        if not name or name in names or file_id not in FILES:
            raise ValueError("invalid or duplicate tensor name or file id")
        if dtype not in (1, 2) or ndim > 4:
            raise ValueError("unsupported tensor dtype or rank")
        shape = dimensions[:ndim] if ndim else [1]
        if any(dimension < 1 for dimension in shape):
            raise ValueError("empty tensor shape")
        itemsize = 2 if dtype == 1 else 4
        if size != math.prod(shape) * itemsize:
            raise ValueError("tensor size disagrees with shape")
        blob = directory / FILES[file_id]
        if not blob.is_file() or offset + size > blob.stat().st_size:
            raise ValueError("tensor exceeds its blob")
        for start, end in intervals.setdefault(file_id, []):
            if offset < end and start < offset + size:
                raise ValueError("overlapping tensor storage")
        intervals[file_id].append((offset, offset + size))
        names.add(name)
        with blob.open("rb") as stream:
            stream.seek(offset)
            digest = hashlib.sha256(stream.read(size)).hexdigest()
        result.append({
            "name": name,
            "file": FILES[file_id],
            "dtype": "BF16" if dtype == 1 else "F32",
            "shape": shape,
            "offset": offset,
            "bytes": size,
            "sha256": digest,
            "version": version,
            "trunk_id": trunk_id,
        })
    return result


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def physical_copy(source: Path, destination: Path, root: Path) -> dict:
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("copy destination escapes the experiment")
    if destination.exists():
        raise FileExistsError(destination)
    before = source.stat()
    digest = file_hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    after = source.stat()
    copied = destination.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"source changed during copying: {source}")
    if (after.st_dev, after.st_ino) == (copied.st_dev, copied.st_ino):
        raise ValueError("copy shares its source inode")
    if file_hash(destination) != digest or file_hash(source) != digest:
        raise ValueError(f"copy hash mismatch: {source}")
    return {
        "source": str(source.resolve()),
        "copy": str(destination.relative_to(root)),
        "bytes": copied.st_size,
        "sha256": digest,
        "distinct_inode": True,
    }


def snapshot(root: Path, exported: Path):
    root = root.resolve(strict=True)
    experiment = root / "composite"
    inputs = experiment / "inputs"
    inputs.mkdir(exist_ok=False)
    originals = root / "original"
    if file_hash(originals / "model.safetensors") != ORIGINAL_SHA256:
        raise ValueError("original Omni checkpoint changed")
    records = []
    for source in sorted(originals.rglob("*")):
        if source.is_file() and ".cache" not in source.relative_to(originals).parts:
            records.append(physical_copy(source, inputs / "c1" / source.relative_to(originals), experiment))

    pre = inputs / "reference" / "data" / "omni" / "precomputed"
    manifest_path = exported / "ncm" / "precomputed" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for group in manifest["directories"]:
        name = group["dir"]
        if not re.fullmatch(r"[a-z0-9-]+", name):
            raise ValueError("invalid export directory name")
        category = "hybridagent" if name.startswith("plan-") else "bridges" if name.startswith("bridge-") else "ncm"
        source_directory = manifest_path.parent / name
        trunk_inventory(source_directory)
        for filename in ("index.bin", *FILES.values()):
            source = source_directory / filename
            if source.is_file():
                records.append(physical_copy(source, pre / category / name / filename, experiment))
    extra = {
        "ncm/vqvae_wte.bf16.bin": exported / "ncm" / "vqvae_wte.bf16.bin",
        "ncm/vqvae_dec.bf16.bin": exported / "ncm" / "vqvae_dec.bf16.bin",
        "trunks/global-0/embed_tokens.bf16.bin": exported / "pruned" / "global-0" / "embed_tokens.bf16.bin",
        "trunks/tokenizer.json": exported / "tokenizer.json",
        "ncm/manifest.json": manifest_path,
    }
    for relative, source in extra.items():
        records.append(physical_copy(source, pre / relative, experiment))
    for source in sorted((root / "fixtures").rglob("*")):
        if source.is_file():
            records.append(physical_copy(source, experiment / "fixtures" / source.relative_to(root / "fixtures"), experiment))
    record = {"schema": 1, "files": records, "original_sha256": ORIGINAL_SHA256}
    write_json(experiment / "INPUTS.json", record)
    print(json.dumps({"copied_files": len(records), "copied_bytes": sum(item["bytes"] for item in records)}), flush=True)


def verify_inputs(experiment: Path, check_sources: bool = False):
    record = json.loads((experiment / "INPUTS.json").read_text(encoding="utf-8"))
    for item in record["files"]:
        copied = experiment / item["copy"]
        if file_hash(copied) != item["sha256"]:
            raise ValueError(f"copied input changed: {item['copy']}")
        if check_sources and file_hash(Path(item["source"])) != item["sha256"]:
            raise ValueError(f"original input changed: {item['source']}")
    return record


def tensor_digest(tensor):
    import torch

    return hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def tokenizer_compatibility(original: dict, trained: dict) -> dict:
    def mapping(tokenizer):
        result = dict(tokenizer["model"]["vocab"])
        for token in tokenizer.get("added_tokens", []):
            if token["content"] in result and result[token["content"]] != token["id"]:
                raise ValueError("tokenizer contains conflicting token ids")
            result[token["content"]] = token["id"]
        return result

    original_map = mapping(original)
    trained_map = mapping(trained)
    shared = original_map.keys() & trained_map.keys()
    if any(original_map[token] != trained_map[token] for token in shared):
        raise ValueError("incompatible shared tokenizer ids")
    original_ids = {value: token for token, value in original_map.items()}
    trained_ids = {value: token for token, value in trained_map.items()}
    conflicts = [number for number in original_ids.keys() & trained_ids.keys() if original_ids[number] != trained_ids[number]]
    if conflicts:
        raise ValueError("a shared token id has changed meaning")
    return {
        "shared_token_ids_agree": True,
        "shared_token_count": len(shared),
        "original_only_tokens": sorted(original_map.keys() - trained_map.keys()),
        "trained_only_tokens": sorted(trained_map.keys() - original_map.keys()),
    }


def pack(root: Path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(2)
    experiment = root.resolve(strict=True) / "composite"
    verify_inputs(experiment)
    inputs = experiment / "inputs"
    pre = inputs / "reference" / "data" / "omni" / "precomputed"
    output = experiment / "package"
    output.mkdir(exist_ok=False)
    tensors = {}
    inventory = {}
    for prefix, path in (
        ("c1", inputs / "c1" / "model.safetensors"),
        ("codec", inputs / "c1" / "audio_tokenizer" / "model.safetensors"),
    ):
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():
                tensors[f"{prefix}.{name}"] = source.get_tensor(name)
    for index in sorted(pre.rglob("index.bin")):
        directory = index.parent
        relative = directory.relative_to(pre).as_posix()
        for entry in trunk_inventory(directory):
            with (directory / entry["file"]).open("rb") as stream:
                stream.seek(entry["offset"])
                buffer = bytearray(stream.read(entry["bytes"]))
            dtype = torch.bfloat16 if entry["dtype"] == "BF16" else torch.float32
            tensors[f"c2c3.{relative}.{entry['name']}"] = torch.frombuffer(buffer, dtype=dtype).reshape(entry["shape"])
    for name, relative, shape in (
        ("vq_wte", "ncm/vqvae_wte.bf16.bin", (128000, 384)),
        ("vq_dec", "ncm/vqvae_dec.bf16.bin", (128000, 384)),
        ("lfm3_embedding", "trunks/global-0/embed_tokens.bf16.bin", (128000, 2048)),
    ):
        path = pre / relative
        if path.stat().st_size != math.prod(shape) * 2:
            raise ValueError(f"unexpected vocabulary table size: {path}")
        tensors[f"c2c3.{name}"] = torch.from_file(str(path), shared=False, size=math.prod(shape), dtype=torch.bfloat16).reshape(shape)

    for name, tensor in tensors.items():
        inventory[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_digest(tensor)}
    base_embedding = tensors["c1.model.language_model.embed_tokens.weight"]
    trained_embedding = tensors["c2c3.lfm3_embedding"]
    same_embedding = torch.equal(base_embedding, trained_embedding)
    original_tokenizer = json.loads((inputs / "c1" / "tokenizer.json").read_text(encoding="utf-8"))
    trained_tokenizer = json.loads((pre / "trunks" / "tokenizer.json").read_text(encoding="utf-8"))
    compatibility = {
        "original_embedding_sha256": inventory["c1.model.language_model.embed_tokens.weight"]["sha256"],
        "trained_embedding_sha256": inventory["c2c3.lfm3_embedding"]["sha256"],
        "embedding_identical": same_embedding,
        "tokenizer": tokenizer_compatibility(original_tokenizer, trained_tokenizer),
        "backbone_substitution_tested": False,
        "qualification": "Nonidentical embeddings preclude a claim of lossless whole-backbone deduplication.",
    }
    save_file(tensors, output / "model.safetensors", metadata={"format": "clover-composite-v1"})
    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as restored:
        if set(restored.keys()) != set(inventory):
            raise ValueError("packed tensor names changed")
        for name, spec in inventory.items():
            tensor = restored.get_tensor(name)
            if list(tensor.shape) != spec["shape"] or str(tensor.dtype) != spec["dtype"] or tensor_digest(tensor) != spec["sha256"]:
                raise ValueError(f"tensor round-trip mismatch: {name}")
    for source in sorted((inputs / "c1").rglob("*")):
        if source.is_file() and source.suffix in (".json", ".py", ".jinja"):
            destination = output / "omni" / source.relative_to(inputs / "c1")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    shutil.copyfile(pre / "trunks" / "tokenizer.json", output / "c2_tokenizer.json")
    write_json(output / "TENSORS.json", inventory)
    write_json(output / "COMPATIBILITY.json", compatibility)
    write_json(output / "MANIFEST.json", {
        "schema": 1,
        "tensor_count": len(inventory),
        "tensor_roundtrip_exact": True,
        "sha256": file_hash(output / "model.safetensors"),
        "bytes": (output / "model.safetensors").stat().st_size,
        "source_files": json.loads((experiment / "INPUTS.json").read_text(encoding="utf-8")),
    })
    print(json.dumps({"tensor_count": len(inventory), "tensor_roundtrip_exact": True, "compatibility": compatibility}), flush=True)


class InventoryTests(unittest.TestCase):
    def fixture(self, root: Path, *, offset=0, size=8, count=1, name=b"weight"):
        (root / "common.bin").write_bytes(struct.pack("<2f", 1.5, -2.0))
        entry = ENTRY.pack(name, 0, 2, 1, offset, size, 2, 0, 0, 0)
        (root / "index.bin").write_bytes(HEADER.pack(b"CLVRTRK1", 1, 0, count) + entry)

    def test_exact_shape_hash(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            result = trunk_inventory(root)
            self.assertEqual(result[0]["shape"], [2])
            self.assertEqual(result[0]["sha256"], file_hash(root / "common.bin"))

    def test_outside_blob_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root, offset=4)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                trunk_inventory(root)

    def test_wrong_shape_size_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root, size=4)
            with self.assertRaisesRegex(ValueError, "shape"):
                trunk_inventory(root)

    def test_wrong_count_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root, count=2)
            with self.assertRaisesRegex(ValueError, "count"):
                trunk_inventory(root)

    def test_duplicate_name_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            raw = (root / "index.bin").read_bytes()
            (root / "index.bin").write_bytes(HEADER.pack(b"CLVRTRK1", 1, 0, 2) + raw[HEADER.size:] * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                trunk_inventory(root)

    def test_copy_is_independent_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "source.bin"
            source.write_bytes(b"immutable source")
            destination = root / "copy.bin"
            result = physical_copy(source, destination, root)
            self.assertTrue(result["distinct_inode"])
            with self.assertRaises(FileExistsError):
                physical_copy(source, destination, root)
            destination.write_bytes(b"changed copy")
            self.assertEqual(source.read_bytes(), b"immutable source")

    def test_copy_escape_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaisesRegex(ValueError, "escapes"):
                physical_copy(root / "unused", root.parent / "outside", root)

    def test_unused_special_token_addition(self):
        original = {"model": {"vocab": {"word": 0}}}
        trained = {"model": {"vocab": {"word": 0}}, "added_tokens": [{"content": "<latent>", "id": 1}]}
        result = tokenizer_compatibility(original, trained)
        self.assertEqual(result["trained_only_tokens"], ["<latent>"])

    def test_special_token_collision_rejected(self):
        original = {"model": {"vocab": {"word": 0}}}
        trained = {"model": {"vocab": {"word": 0}}, "added_tokens": [{"content": "<latent>", "id": 0}]}
        with self.assertRaisesRegex(ValueError, "meaning"):
            tokenizer_compatibility(original, trained)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--trunk", type=Path)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--exported", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.trunk:
        print(json.dumps(trunk_inventory(args.trunk), indent=2))
    elif args.snapshot and args.root and args.exported:
        snapshot(args.root, args.exported)
    elif args.pack and args.root:
        pack(args.root)
    elif args.verify_inputs and args.root:
        records = verify_inputs(args.root / "composite", check_sources=True)
        print(json.dumps({"source_and_copy_hashes_unchanged": True, "files": len(records["files"])}))
    else:
        parser.error("choose --self-test, --trunk, --snapshot, --pack or --verify-inputs")


if __name__ == "__main__":
    main()