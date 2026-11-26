import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiment import compare_exact_rows, read_results
from finalize import paired_counts


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Artifact digest mismatch: {path}")


def verify(root):
    root = Path(root)
    seal = json.loads((root / "FROZEN.json").read_text())
    for filename, expected in seal["scripts"].items():
        require_hash(root / filename, expected)
    require_hash(root / "SEALED.json", seal["checkpoint_seal_sha256"])
    require_hash(root / "fixtures" / "MATERIALS.json", seal["materials_sha256"])
    materials = json.loads((root / "fixtures" / "MATERIALS.json").read_text())
    for entry in materials["files"]:
        require_hash(root / "fixtures" / entry["file"], entry["sha256"])
    cases = json.loads((root / "fixtures" / "cases.json").read_text())
    expected_ids = {case["id"] for case in cases}
    if len(expected_ids) != 112 or len(cases) != 112:
        raise ValueError("Invalid frozen case coverage")
    baseline = read_results(root / "reports" / "baseline" / "results.jsonl")
    arms = {}
    for name in ("baseline", "no-op-control", "H1-alpha-05", "H1-alpha-10", "H1-alpha-20"):
        directory = root / "reports" / name
        summary = json.loads((directory / "summary.json").read_text())
        require_hash(directory / "results.jsonl", summary["results_sha256"])
        require_hash(directory / "activations.npz", summary["activations_sha256"])
        rows = read_results(directory / "results.jsonl")
        if set(rows) != expected_ids or summary["case_count"] != 112 or summary["error_count"]:
            raise ValueError(f"Incomplete/failed comparison arm: {name}")
        for identifier in rows:
            if rows[identifier]["input_sha256"] != baseline[identifier]["input_sha256"]:
                raise ValueError(f"Input changed in {name}: {identifier}")
        arms[name] = rows
    compare_exact_rows(baseline, arms["no-op-control"])
    comparison = json.loads((root / "reports" / "COMPARISON.json").read_text())
    validation_ids = [case["id"] for case in cases if case["split"] == "validation"]
    evidence = json.loads((root / "reports" / "activation-analysis.json").read_text())
    require_hash(root / "analysis.py", evidence["selection_source_sha256"])
    require_hash(root / "reports" / "baseline" / "results.jsonl", evidence["baseline_results_sha256"])
    require_hash(root / "reports" / "baseline" / "activations.npz", evidence["baseline_activations_sha256"])
    for record in comparison["arms"]:
        name = record["name"]
        if paired_counts(baseline, arms[name], validation_ids) != record["validation"]:
            raise ValueError(f"Computed paired results disagree: {name}")
        spec_path = root / "reports" / "patches" / f"{name}.json"
        spec = json.loads(spec_path.read_text())
        require_hash(root / spec["hypothesis_file"], spec["hypothesis_sha256"])
        audit = json.loads((root / "copies" / name / "PATCH_AUDIT.json").read_text())
        reversal = json.loads((root / "copies" / name / "REVERSAL.json").read_text())
        require_hash(spec_path, audit["spec_sha256"])
        if audit["changed_tensors"] != [spec["target_tensor"]]:
            raise ValueError("Single-tensor audit disagrees")
        if reversal["restored_checkpoint_sha256"] != spec["original_checkpoint_sha256"]:
            raise ValueError("Reversal digest disagrees")
        changed = [tensor for tensor, digest in audit["original"]["tensors"].items()
                   if audit["copy"]["tensors"][tensor] != digest]
        if changed != [spec["target_tensor"]]:
            raise ValueError("Raw tensor digests disagree with the one-tensor audit")
    required = ["ARCHITECTURE.md", "WEIGHTS.md", "ACTIVATIONS.md", "HYPOTHESES.md", "RESULTS.md", "inventory.json"]
    for filename in required:
        if not (root / "reports" / filename).is_file():
            raise FileNotFoundError(filename)
    return {"controlled_case_evaluations": sum(len(rows) for rows in arms.values()),
            "frozen_source_files_verified": len(seal["scripts"]),
            "fixture_files_verified": len(materials["files"]),
            "arms_with_results_and_activation_hashes_verified": len(arms),
            "patch_and_reversal_records_verified": len(comparison["arms"]),
            "recommendation": comparison["recommendation"],
            "qualification": "Local evidence integrity verified; original weight hashes and runtime isolation were checked on K3."}


class HandoffTests(unittest.TestCase):
    def test_corrupt_artifact_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.json"
            path.write_text('{"value": 1}')
            expected = sha256(path)
            require_hash(path, expected)
            path.write_text('{"value": 2}')
            with self.assertRaises(ValueError):
                require_hash(path, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).parent)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["verify_handoff.py"], verbosity=2)
    else:
        print(json.dumps(verify(args.root), indent=2))