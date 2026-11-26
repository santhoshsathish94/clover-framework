import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import unittest


DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
    "I32": 4, "U32": 4, "I64": 8, "U64": 8,
    "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_header(source, file_size):
    prefix = source.read(8)
    if len(prefix) != 8:
        raise ValueError("Truncated safetensors length prefix")
    header_size = struct.unpack("<Q", prefix)[0]
    if header_size > 100_000_000 or header_size + 8 > file_size:
        raise ValueError("Invalid safetensors header size")
    header = json.loads(source.read(header_size), object_pairs_hook=unique_object)
    payload_size = file_size - 8 - header_size
    tensors = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        shape = spec["shape"]
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError(f"Invalid shape: {name}")
        count = math.prod(shape)
        dtype = spec["dtype"]
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        start, end = spec["data_offsets"]
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"Non-integer offsets: {name}")
        if not 0 <= start <= end <= payload_size:
            raise ValueError(f"Invalid data offsets: {name}")
        if end - start != count * DTYPE_BYTES[dtype]:
            raise ValueError(f"Byte count disagrees with shape: {name}")
        tensors.append({
            "name": name, "shape": shape, "dtype": dtype,
            "parameter_count": count, "bytes": end - start,
            "data_offsets": [start, end],
            "file_offsets": [start + 8 + header_size, end + 8 + header_size],
        })
    position = 0
    for tensor in sorted(tensors, key=lambda item: tuple(item["data_offsets"])):
        start, end = tensor["data_offsets"]
        if start != position:
            raise ValueError(f"Gap or overlap before {tensor['name']}")
        position = end
    if position != payload_size:
        raise ValueError("Unaccounted checkpoint payload bytes")
    return tensors, header.get("__metadata__", {})


def independent_output(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Output must be outside the immutable source directory")
    return destination


def component_for_name(name):
    prefixes = {
        "model.language_model.embed_tokens.": "shared_embedding_and_text_output",
        "lm_head.": "shared_embedding_and_text_output",
        "model.language_model.": "shared_backbone",
        "model.vision_tower.": "vision_encoder_image_and_video",
        "model.multi_modal_projector.": "vision_to_backbone_projector",
        "model.audio_encoder.": "audio_input_encoder",
        "model.audio_projector.": "audio_to_backbone_projector",
        "model.codebook_embedding.": "speech_code_to_backbone_embedding",
        "model.depth_decoder.": "speech_output_depth_decoder",
    }
    for prefix, component in prefixes.items():
        if name.startswith(prefix):
            return component
    return "unmapped"


def checkpoint_inventory(model_dir):
    model_dir = Path(model_dir).resolve(strict=True)
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = None
    if index_path.exists():
        index = json.loads(index_path.read_text(), object_pairs_hook=unique_object)
        weight_map = index["weight_map"]
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = [path.name for path in sorted(model_dir.glob("*.safetensors"))]
    if not shard_names:
        raise ValueError("No safetensors checkpoint found")
    tensors = []
    files = []
    for shard_name in shard_names:
        if Path(shard_name).name != shard_name:
            raise ValueError("Checkpoint shard must be a direct child of model_dir")
        shard = model_dir / shard_name
        before = shard.stat()
        with shard.open("rb") as source:
            entries, metadata = read_header(source, before.st_size)
        digest = sha256_file(shard)
        after = shard.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Input changed while inventorying: {shard_name}")
        files.append({
            "file": shard_name, "resolved_path": str(shard.resolve()),
            "sha256": digest, "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns, "metadata": metadata,
        })
        for entry in entries:
            entry["shard"] = shard_name
            entry["component"] = component_for_name(entry["name"])
            tensors.append(entry)
    names = [entry["name"] for entry in tensors]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate tensors across shards")
    actual_map = {entry["name"]: entry["shard"] for entry in tensors}
    if weight_map is not None and actual_map != weight_map:
        raise ValueError("Stored tensors disagree with shard index")
    metadata_files = []
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and path.suffix in {".json", ".py", ".md", ".jinja"}:
            metadata_files.append({
                "file": path.name, "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    configurations = {}
    for name in ("config.json", "generation_config.json", "preprocessor_config.json"):
        path = model_dir / name
        if path.exists():
            configurations[name] = json.loads(path.read_text())
    return {
        "schema_version": 1, "model_dir": str(model_dir),
        "tensor_count": len(tensors),
        "stored_parameter_count": sum(entry["parameter_count"] for entry in tensors),
        "stored_tensor_bytes": sum(entry["bytes"] for entry in tensors),
        "index_agrees": weight_map is not None,
        "files": files, "metadata_files": metadata_files,
        "configurations": configurations, "tensors": tensors,
        "qualification": "Stored tensors only; runtime parameter tying and buffers require source/runtime analysis.",
    }


class InventoryTests(unittest.TestCase):
    def payload(self, specs, data=b"\x00" * 8):
        header = json.dumps(specs).encode()
        return struct.pack("<Q", len(header)) + header + data

    def decode(self, payload):
        return read_header(io.BytesIO(payload), len(payload))

    def test_known_tensor(self):
        payload = self.payload({"weight": {
            "dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8],
        }})
        entries, _ = self.decode(payload)
        self.assertEqual(entries[0]["parameter_count"], 4)
        self.assertEqual(entries[0]["bytes"], 8)

    def test_size_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "Byte count"):
            self.decode(self.payload({"weight": {
                "dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 8],
            }}))

    def test_overlap_rejected(self):
        spec = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        with self.assertRaisesRegex(ValueError, "Gap or overlap"):
            self.decode(self.payload({"first": spec, "second": spec}))

    def test_truncated_rejected(self):
        with self.assertRaises(ValueError):
            self.decode(b"\x00\x00")

    def test_duplicate_json_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            json.loads('{"weight": 1, "weight": 2}', object_pairs_hook=unique_object)

    def test_source_write_rejected(self):
        source = Path.cwd() / "immutable-model"
        with self.assertRaisesRegex(ValueError, "outside"):
            independent_output(source, source / "report.json")

    def test_shared_head_mapping(self):
        self.assertEqual(
            component_for_name("lm_head.weight"),
            component_for_name("model.language_model.embed_tokens.weight"),
        )

    def test_unknown_component_not_guessed(self):
        self.assertEqual(component_for_name("unknown.weight"), "unmapped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-dir")
    parser.add_argument("--output")
    parser.add_argument("--include-codec", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(InventoryTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if not args.model_dir or not args.output:
        parser.error("--model-dir and --output are required")
    destination = independent_output(args.model_dir, args.output)
    if destination.exists():
        raise FileExistsError(destination)
    report = checkpoint_inventory(args.model_dir)
    if args.include_codec:
        codec = checkpoint_inventory(Path(args.model_dir) / "audio_tokenizer")
        for entry in codec["tensors"]:
            entry["component"] = "speech_waveform_codec"
        report["auxiliary_checkpoints"] = {"audio_tokenizer": codec}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: report[key] for key in (
        "tensor_count", "stored_parameter_count", "stored_tensor_bytes",
    )}))
    return 0


if __name__ == "__main__":
    sys.exit(main())