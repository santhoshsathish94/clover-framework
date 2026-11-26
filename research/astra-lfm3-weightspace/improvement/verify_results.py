import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from validate_final import group_comparison, read_rows


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_digest(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Artifact digest mismatch: {path}")


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "improvement" / "frozen" / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        require_digest(root / "improvement" / "frozen" / name, expected)
    for name, expected in manifest["sources"].items():
        require_digest(root / name, expected)
    trials = json.loads((root / "improvement" / "FOLLOWUP_TRIALS.json").read_text())
    runs = ["original-tuning"] + [trial["name"] for trial in trials["trials"]] + ["original-final", "candidate-final"]
    evaluations = 0
    for name in runs:
        directory = root / "improvement" / "runs" / name
        summary = json.loads((directory / "summary.json").read_text())
        require_digest(directory / "results.jsonl", summary["results_sha256"])
        recorded = read_rows(directory / "results.jsonl")
        if len(recorded) != summary["case_count"]:
            raise ValueError("Case denominator mismatch")
        evaluations += len(recorded)
    comparison = json.loads((root / "improvement" / "FINAL_COMPARISON.json").read_text())
    original = read_rows(root / "improvement" / "runs" / "original-final" / "results.jsonl")
    candidate = read_rows(root / "improvement" / "runs" / "candidate-final" / "results.jsonl")
    recomputed = json.loads(json.dumps(group_comparison(original, candidate), allow_nan=False))
    if recomputed != comparison["final_comparison"]:
        raise ValueError("Independent grouped result recomputation differs")
    outcome = json.loads((root / "improvement" / "OUTCOME.json").read_text())
    if outcome["tested_candidate_retained"] != (outcome["decision"] == "retain_scoped_improvement"):
        raise ValueError("Final artifact decision is inconsistent")
    for trial in trials["trials"]:
        directory = root / "improvement" / "copies" / trial["name"]
        audit = json.loads((directory / "PATCH_AUDIT.json").read_text())
        reversal = json.loads((directory / "REVERSAL.json").read_text())
        if len(audit["changed_tensors"]) != 1:
            raise ValueError("Trial modified more than one component")
        changed = [name for name, digest in audit["original"]["tensors"].items()
                   if audit["copy"]["tensors"][name] != digest]
        if changed != audit["changed_tensors"]:
            raise ValueError("Raw tensor audit disagrees")
        if reversal["original_sha256_restored"] != audit["base_checkpoint_sha256"]:
            raise ValueError("Trial reversal proof differs")
    return {"trial_count": len(trials["trials"]), "tuning_and_final_responses_verified": evaluations,
            "final_grouped_comparison_recomputed": True, "trial_audits_and_reversals_verified": True,
            "decision": outcome["decision"], "final_checkpoint_sha256": outcome["final_checkpoint_sha256"],
            "qualification": "Saved artifact verification; actual checkpoint files and runtime invariants were checked on K3."}


class VerificationTests(unittest.TestCase):
    def test_corrupted_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text('{"correct": 1}')
            expected = sha256(path)
            require_digest(path, expected)
            path.write_text('{"correct": 2}')
            with self.assertRaisesRegex(ValueError, "Artifact digest mismatch"):
                require_digest(path, expected)

    def test_json_group_keys_preserve_all_values(self):
        measured = {1: {"original": 2, "edited": 3}}
        recorded = {"1": {"original": 2, "edited": 3}}
        self.assertEqual(json.loads(json.dumps(measured)), recorded)
        recorded["1"]["edited"] = 4
        self.assertNotEqual(json.loads(json.dumps(measured)), recorded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["verify_results.py"], verbosity=2)
    else:
        print(json.dumps(verify(args.root), indent=2))