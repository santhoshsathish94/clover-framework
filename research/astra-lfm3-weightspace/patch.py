import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

from inventory import read_header


TARGET = "model.language_model.embedding_norm.weight"


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def copy_destination(root, name):
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Copy name must be a single directory name")
    copies = (Path(root) / "copies").resolve()
    destination = (copies / name).resolve()
    if copies not in destination.parents:
        raise ValueError("Destination must be strictly inside the research copies directory")
    return destination


def distinct_files(original, copied):
    if os.path.samefile(original, copied):
        raise ValueError("A copied checkpoint must not be a hard link or alias of the original")


def replace_exact_bytes(path, offset, before, after):
    if len(before) != len(after) or not before or offset < 0:
        raise ValueError("Invalid fixed-size patch")
    with Path(path).open("r+b") as output:
        output.seek(offset)
        if output.read(len(before)) != before:
            raise ValueError("Expected original bytes do not match; refusing write")
        output.seek(offset)
        output.write(after)
        output.flush()
        os.fsync(output.fileno())


def tensor_hashes(path):
    path = Path(path)
    with path.open("rb") as source:
        entries, _ = read_header(source, path.stat().st_size)
        first_data = min(entry["file_offsets"][0] for entry in entries)
        source.seek(0)
        header_digest = hashlib.sha256(source.read(first_data)).hexdigest()
        hashes = {}
        for entry in sorted(entries, key=lambda item: item["file_offsets"][0]):
            start, end = entry["file_offsets"]
            source.seek(start)
            digest = hashlib.sha256()
            remaining = end - start
            while remaining:
                block = source.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise EOFError("Checkpoint truncated during tensor audit")
                digest.update(block)
                remaining -= len(block)
            hashes[entry["name"]] = digest.hexdigest()
    return {"header_sha256": header_digest, "tensors": hashes}


def tensor_audit(original, copied, expected_changed):
    before, after = tensor_hashes(original), tensor_hashes(copied)
    if before["header_sha256"] != after["header_sha256"] or before["tensors"].keys() != after["tensors"].keys():
        raise ValueError("Checkpoint structure changed")
    changed = [name for name, digest in before["tensors"].items() if digest != after["tensors"][name]]
    if set(changed) != set(expected_changed):
        raise ValueError(f"Unexpected changed tensors: {changed}")
    return {"changed_tensors": changed, "tensor_count": len(before["tensors"]),
            "original": before, "copy": after}


def validate_spec(root, spec_path):
    import torch

    root = Path(root)
    spec = json.loads(Path(spec_path).read_text())
    if spec["hypothesis_id"] != "H1" or spec["target_tensor"] != TARGET:
        raise ValueError("Only the documented H1 final-norm tensor is permitted")
    if spec["hypothesis_file"] != "reports/HYPOTHESES.md":
        raise ValueError("Unexpected hypothesis location")
    if sha256(root / spec["hypothesis_file"]) != spec["hypothesis_sha256"]:
        raise ValueError("Hypothesis documentation digest mismatch")
    analysis = json.loads((root / "reports" / "activation-analysis.json").read_text())
    if sha256(root / "analysis.py") != analysis["selection_source_sha256"]:
        raise ValueError("Candidate selection code changed")
    if spec["flat_index"] != analysis["ranked_channels"][0]["index"]:
        raise ValueError("Coordinate was not the documented selected candidate")
    if spec["baseline_results_sha256"] != sha256(root / "reports" / "baseline" / "results.jsonl"):
        raise ValueError("Supporting baseline changed")
    model_path = root / "original" / "model.safetensors"
    if sha256(model_path) != spec["original_checkpoint_sha256"]:
        raise ValueError("Original checkpoint identity mismatch")
    with model_path.open("rb") as source:
        entries, _ = read_header(source, model_path.stat().st_size)
        target = next(entry for entry in entries if entry["name"] == TARGET)
        index = spec["flat_index"]
        if type(index) is not int or not 0 <= index < target["parameter_count"]:
            raise ValueError("Invalid target index")
        if spec["target_shape"] != target["shape"] or spec["dtype"] != "BF16" or target["dtype"] != "BF16":
            raise ValueError("Unexpected target shape or dtype")
        offset = target["file_offsets"][0] + 2 * index
        if offset != spec["file_offset"]:
            raise ValueError("Patch offset disagrees with checkpoint header")
        source.seek(offset)
        before = source.read(2)
    if before.hex() != spec["old_hex"]:
        raise ValueError("Recorded original BF16 bytes do not match")
    strength = spec["requested_attenuation"]
    if strength not in (0.05, 0.10, 0.20):
        raise ValueError("Strength is not preregistered")
    old_value = struct.unpack("<f", b"\x00\x00" + before)[0]
    expected = (torch.tensor([old_value], dtype=torch.float32) * (1 - strength)).bfloat16()
    after = expected.view(torch.uint8).numpy().tobytes()
    if after.hex() != spec["new_hex"] or after == before:
        raise ValueError("New bytes are not the declared nonzero BF16 attenuation")
    return spec


def verify_other_files(destination, manifest):
    for entry in manifest["files"]:
        if entry["file"] != "model.safetensors":
            if sha256(destination / entry["file"]) != entry["sha256"]:
                raise ValueError(f"Untouched checkpoint file changed: {entry['file']}")


def apply(root, name, spec_path=None):
    from benchmark import verify_seal
    from survey import verify_original

    root = Path(root).resolve()
    manifest = verify_original(root)
    verify_seal(root)
    spec = validate_spec(root, spec_path) if spec_path else None
    destination = copy_destination(root, name)
    if destination.exists():
        raise FileExistsError("Refusing to replace any existing copy")
    original = root / "original"
    shutil.copytree(original, destination, copy_function=shutil.copy2, ignore=shutil.ignore_patterns(".cache"))
    destination.chmod(0o755)
    model_path = destination / "model.safetensors"
    distinct_files(original / "model.safetensors", model_path)
    base_hash = sha256(original / "model.safetensors")
    if sha256(model_path) != base_hash:
        raise ValueError("Physical copy differs before applying the patch")
    if spec:
        model_path.chmod(0o644)
        replace_exact_bytes(model_path, spec["file_offset"], bytes.fromhex(spec["old_hex"]), bytes.fromhex(spec["new_hex"]))
    audit = tensor_audit(original / "model.safetensors", model_path, [TARGET] if spec else [])
    verify_other_files(destination, manifest)
    audit.update({"copy_name": name, "base_checkpoint_sha256": base_hash,
                  "after_checkpoint_sha256": sha256(model_path),
                  "spec_sha256": sha256(spec_path) if spec else None,
                  "no_op_control": spec is None,
                  "other_checkpoint_files_unchanged": True})
    with (destination / "PATCH_AUDIT.json").open("x") as output:
        json.dump(audit, output, indent=2)
    model_path.chmod(0o444)
    verify_original(root)
    print("COPY_AUDITED " + json.dumps({key: audit[key] for key in (
        "copy_name", "changed_tensors", "no_op_control", "after_checkpoint_sha256",
    )}), flush=True)


def revert(root, name, spec_path):
    from survey import verify_original

    root = Path(root).resolve()
    manifest = verify_original(root)
    spec = validate_spec(root, spec_path)
    destination = copy_destination(root, name)
    model_path = destination / "model.safetensors"
    distinct_files(root / "original" / "model.safetensors", model_path)
    audit = json.loads((destination / "PATCH_AUDIT.json").read_text())
    if sha256(model_path) != audit["after_checkpoint_sha256"] or sha256(spec_path) != audit["spec_sha256"]:
        raise ValueError("Copy or patch changed since the recorded application; refusing reversal")
    model_path.chmod(0o644)
    replace_exact_bytes(model_path, spec["file_offset"], bytes.fromhex(spec["new_hex"]), bytes.fromhex(spec["old_hex"]))
    restored = sha256(model_path)
    if restored != spec["original_checkpoint_sha256"]:
        raise ValueError("Reversal failed whole-checkpoint identity check")
    verify_other_files(destination, manifest)
    model_path.chmod(0o444)
    result = {"copy_name": name, "restored_checkpoint_sha256": restored,
              "byte_exact_reversal": True, "other_checkpoint_files_unchanged": True}
    with (destination / "REVERSAL.json").open("x") as output:
        json.dump(result, output, indent=2)
    verify_original(root)
    print("REVERSAL_VERIFIED " + json.dumps(result), flush=True)


class PatchTests(unittest.TestCase):
    def test_exact_reversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "copy.bin"
            path.write_bytes(b"prefixABsuffix")
            before = sha256(path)
            replace_exact_bytes(path, 6, b"AB", b"CD")
            self.assertEqual(path.read_bytes(), b"prefixCDsuffix")
            replace_exact_bytes(path, 6, b"CD", b"AB")
            self.assertEqual(sha256(path), before)

    def test_mismatch_does_not_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "copy.bin"
            path.write_bytes(b"ABCD")
            with self.assertRaises(ValueError):
                replace_exact_bytes(path, 0, b"XX", b"YY")
            self.assertEqual(path.read_bytes(), b"ABCD")

    def test_destination_escape_refused(self):
        with self.assertRaises(ValueError):
            copy_destination("/study", "../original")
        with self.assertRaises(ValueError):
            copy_destination("/study", "/study/original")

    def test_hard_link_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            original, copied = Path(temporary) / "original", Path(temporary) / "copy"
            original.write_bytes(b"original")
            os.link(original, copied)
            with self.assertRaises(ValueError):
                distinct_files(original, copied)

    def test_audit_rejects_other_tensor_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            original, copied = Path(temporary) / "original", Path(temporary) / "copy"
            header = json.dumps({TARGET: {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                                 "other.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [4, 8]}}).encode()
            original.write_bytes(struct.pack("<Q", len(header)) + header + b"abcdefgh")
            shutil.copyfile(original, copied)
            self.assertEqual(tensor_audit(original, copied, [])["changed_tensors"], [])
            replace_exact_bytes(copied, 8 + len(header) + 4, b"ef", b"XY")
            with self.assertRaisesRegex(ValueError, "Unexpected changed tensors"):
                tensor_audit(original, copied, [TARGET])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--copy")
    parser.add_argument("--spec")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["patch.py"], verbosity=2)
    elif not args.root or not args.copy:
        parser.error("--root and --copy required")
    elif args.revert:
        if not args.spec:
            parser.error("--spec required for reversal")
        revert(args.root, args.copy, args.spec)
    else:
        apply(args.root, args.copy, args.spec)