import argparse
import json
from pathlib import Path
import tempfile
import unittest

from study import OLD64_HASH, TARGET, sha256, verify_frozen
from run import choose_width, exact_control
from validate_final import group_comparison, read_rows


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")


def verify(root):
    root = Path(root)
    manifest = verify_frozen(root)
    work = root / "improvement" / "neuron_count"
    outcome = json.loads((work / "OUTCOME.json").read_text())
    trials = outcome["trials"]
    control = next(trial for trial in trials if trial["width"] == 64)
    spec_map = {spec["name"]: spec for spec in json.loads((work / "frozen" / "specs.json").read_text())}
    if len(trials) != len(spec_map):
        raise ValueError("Trial/spec coverage differs")
    total_responses = 0
    for trial in trials:
        name = trial["name"]
        for key, value in spec_map[name].items():
            if trial[key] != value:
                raise ValueError("A trial changed its frozen spec")
        directory = root / "improvement" / "runs" / f"width-{name}"
        require_hash(directory / "results.jsonl", trial["results_sha256"])
        rows = read_rows(directory / "results.jsonl")
        if len(rows) != 228:
            raise ValueError("Incomplete tuning trial")
        total_responses += len(rows)
        archived = work / "restored_trials" / f"width-{name}"
        copy = archived if archived.exists() else root / "improvement" / "copies" / f"width-{name}"
        audit = json.loads((copy / "PATCH_AUDIT.json").read_text())
        reversal = json.loads((copy / "REVERSAL.json").read_text())
        if audit["changed_tensors"] != [TARGET] or len(audit["changed_columns"]) != trial["width"]:
            raise ValueError("Changed component or neuron count differs")
        changed = [key for key, value in audit["original"]["tensors"].items() if audit["copy"]["tensors"][key] != value]
        if changed != [TARGET]:
            raise ValueError("Raw tensor hashes disagree with audit")
        if reversal["original_sha256_restored"] != audit["base_checkpoint_sha256"]:
            raise ValueError("Invalid trial reversal")
        if name == "n64-fixed" and audit["edited_checkpoint_sha256"] != OLD64_HASH:
            raise ValueError("The64-neuron checkpoint control differs")
        if trial["track"] == "norm_matched" and abs(trial["actual_delta_frobenius"] / control["actual_delta_frobenius"] - 1) > 5e-4:
            raise ValueError("Norm-matched control deviates by more than0.05% after rounding")
    exact_control(read_rows(root / "improvement" / "runs" / "layer22-followup-a4" / "results.jsonl"),
                  read_rows(root / "improvement" / "runs" / "width-n64-fixed" / "results.jsonl"))
    selected = choose_width(outcome["original_tuning"], control["totals"], trials)
    if selected != outcome["selected"]:
        raise ValueError("Independent candidate selection differs")
    if outcome["fresh_confirmation_opened"]:
        confirmation = outcome["confirmation"]
        records = {}
        for name in ("original", "n64", "wider"):
            directory = work / "confirmation" / name
            summary = json.loads((directory / "summary.json").read_text())
            require_hash(directory / "results.jsonl", summary["result_sha256"])
            records[name] = read_rows(directory / "results.jsonl")
            if len(records[name]) != 456:
                raise ValueError("Incomplete confirmation")
            total_responses += len(records[name])
        for baseline, key in (("original", "versus_original"), ("n64", "versus64")):
            measured = json.loads(json.dumps(group_comparison(records[baseline], records["wider"])))
            if measured != confirmation[key]:
                raise ValueError("Fresh grouped comparison differs")
        decision = confirmation["decision"]
    else:
        if selected is not None:
            raise ValueError("Qualifying wider edit never received confirmation")
        decision = outcome["decision"]
    return {"frozen_widths": manifest["available_widths"], "trial_count": len(trials),
            "tuning_and_confirmation_responses_verified": total_responses,
            "old64_checkpoint_and_outputs_exact": True,
            "neuron_counts_and_trial_reversals_verified": True, "decision": decision}


class VerifyTests(unittest.TestCase):
    def test_corrupted_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text('{"width": 64}')
            expected = sha256(path)
            require_hash(path, expected)
            path.write_text('{"width": 128}')
            with self.assertRaises(ValueError):
                require_hash(path, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["verify.py"], verbosity=2)
    elif args.root:
        print(json.dumps(verify(args.root), indent=2))
    else:
        parser.error("--root or --self-test required")