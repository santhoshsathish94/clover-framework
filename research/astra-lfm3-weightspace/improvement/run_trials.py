import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


def choose_candidate(baseline, trials):
    eligible = []
    for trial in trials:
        totals = trial["totals"]
        if totals["arithmetic_mcq"]["correct"] <= baseline["arithmetic_mcq"]["correct"]:
            continue
        if totals["arithmetic_free"]["correct"] < baseline["arithmetic_free"]["correct"]:
            continue
        if totals["selection"]["correct"] < baseline["selection"]["correct"]:
            continue
        eligible.append(trial)
    eligible.sort(key=lambda trial: (-trial["totals"]["arithmetic_mcq"]["correct"], trial["strength"], trial["layer"]))
    return eligible[0] if eligible else None


def child(root, script, *args):
    command = [sys.executable, "-B", str(root / "improvement" / script), "--root", str(root), *map(str, args)]
    print("STAGE " + json.dumps({"script": script, "args": list(map(str, args))}), flush=True)
    subprocess.run(command, check=True)


def run(root):
    root = Path(root)
    baseline = json.loads((root / "improvement" / "runs" / "original-tuning" / "summary.json").read_text())
    proposal_dir = root / "improvement" / "proposals"
    specs = sorted(proposal_dir.glob("layer*-a*.json"))
    if len(specs) != 6:
        raise ValueError("Expected the fixed six documented proposals")
    destination = root / "improvement" / "TRIALS.json"
    if destination.exists():
        raise FileExistsError(destination)
    baseline_rows = {row["id"]: row for row in (json.loads(line) for line in (root / "improvement" / "runs" / "original-tuning" / "results.jsonl").read_text().splitlines())}
    trials = []
    for spec_path in specs:
        spec = json.loads(spec_path.read_text())
        name = spec["name"]
        child(root, "interventions.py", "--apply", spec_path)
        try:
            child(root, "evaluate.py", "--split", "tuning", "--name", name,
                  "--checkpoint", root / "improvement" / "copies" / name)
            run_dir = root / "improvement" / "runs" / name
            summary = json.loads((run_dir / "summary.json").read_text())
            changed_tokens = 0
            for line in (run_dir / "results.jsonl").read_text().splitlines():
                row = json.loads(line)
                original = baseline_rows[row["id"]]
                if original["input_sha256"] != row["input_sha256"]:
                    raise ValueError("Trial input differs from original")
                changed_tokens += original["tokens"] != row["tokens"]
            trial = {"name": name, "layer": spec["layer"], "strength": spec["strength"],
                     "totals": summary["totals"], "changed_token_cases": changed_tokens,
                     "results_sha256": summary["results_sha256"],
                     "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest()}
            trials.append(trial)
            print("TUNING_TRIAL " + json.dumps(trial), flush=True)
        finally:
            child(root, "interventions.py", "--revert", name)
    selected = choose_candidate(baseline["totals"], trials)
    result = {"baseline": baseline["totals"], "trials": trials, "selected": selected,
              "selection_used_only_tuning": True, "all_trial_copies_reversed": True,
              "final_validation_not_run": True}
    with destination.open("x") as output:
        json.dump(result, output, indent=2)
    print("TUNING_COMPLETE " + json.dumps(result), flush=True)


class SelectionTests(unittest.TestCase):
    def test_regression_excludes_candidate(self):
        base = {"arithmetic_mcq": {"correct": 10}, "arithmetic_free": {"correct": 8}, "selection": {"correct": 8}}
        trial = {"layer": 22, "strength": 1, "totals": {"arithmetic_mcq": {"correct": 15}, "arithmetic_free": {"correct": 7}, "selection": {"correct": 8}}}
        self.assertIsNone(choose_candidate(base, [trial]))
        trial["totals"]["arithmetic_free"]["correct"] = 8
        self.assertEqual(choose_candidate(base, [trial]), trial)

    def test_tie_prefers_lower_strength(self):
        base = {name: {"correct": 8} for name in ("arithmetic_mcq", "arithmetic_free", "selection")}
        totals = {name: {"correct": 9} for name in base}
        trials = [{"layer": 22, "strength": strength, "totals": totals} for strength in (2, 0.5, 1)]
        self.assertEqual(choose_candidate(base, trials)["strength"], 0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["run_trials.py"], verbosity=2)
    elif args.root:
        run(args.root)
    else:
        parser.error("--root or --self-test required")