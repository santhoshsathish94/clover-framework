import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest


def read_results(path):
    with Path(path).open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows) or not rows:
        raise ValueError("Results must be nonempty and have unique case IDs")
    return indexed


def compare_exact_rows(baseline, control, expected_count=112):
    if len(baseline) != expected_count or baseline.keys() != control.keys():
        raise ValueError("No-op control case coverage differs")
    for identifier, before in baseline.items():
        after = control[identifier]
        for key in ("input_sha256", "tokens", "text", "letter_logits", "correct"):
            if before.get(key) != after.get(key):
                raise ValueError(f"No-op differs at {identifier}: {key}")
        for key in ("codes_sha256", "waveform_sha256"):
            if before.get("speech", {}).get(key) != after.get("speech", {}).get(key):
                raise ValueError(f"No-op speech differs at {identifier}: {key}")
    return {"cases_exact": len(baseline), "text_tokens_logits_and_speech_identical": True}


def compare_activations(original_path, comparison_path, require_all=False):
    import numpy as np

    before = np.load(original_path, allow_pickle=False)
    after = np.load(comparison_path, allow_pickle=False)
    if set(before.files) != set(after.files):
        raise ValueError("Activation coverage differs")
    unchanged = changed_downstream = 0
    for key in before.files:
        location = key.split("::", 1)[1]
        downstream = location in ("final_hidden", "letter_logits") or location.startswith("speech_depth")
        equal = np.array_equal(before[key], after[key])
        if require_all or not downstream:
            if not equal:
                raise ValueError(f"Unexpected upstream/control activation change: {key}")
            unchanged += 1
        elif not equal:
            changed_downstream += 1
    if not require_all and not changed_downstream:
        raise ValueError("No downstream activation changed; the intervention is not shown live")
    return {"unchanged_required_arrays": unchanged,
            "changed_downstream_arrays": changed_downstream,
            "upstream_identical": True}


def run_child(root, script, *arguments):
    command = [sys.executable, "-B", str(root / script), "--root", str(root), *map(str, arguments)]
    print("STAGE " + json.dumps({"script": script, "arguments": list(map(str, arguments))}), flush=True)
    subprocess.run(command, check=True)


def execute(root):
    root = Path(root).resolve()
    experiment_path = root / "reports" / "experiment-complete.json"
    if experiment_path.exists():
        raise FileExistsError(experiment_path)
    baseline_dir = root / "reports" / "baseline"
    baseline = read_results(baseline_dir / "results.jsonl")
    baseline_summary = json.loads((baseline_dir / "summary.json").read_text())
    if baseline_summary["error_count"] or len(baseline) != 112:
        raise ValueError("Complete error-free baseline required")
    started = time.monotonic()
    run_child(root, "patch.py", "--copy", "no-op")
    run_child(root, "benchmark.py", "--name", "no-op-control", "--checkpoint", root / "copies" / "no-op")
    control_dir = root / "reports" / "no-op-control"
    control = compare_exact_rows(baseline, read_results(control_dir / "results.jsonl"))
    control.update(compare_activations(baseline_dir / "activations.npz", control_dir / "activations.npz", require_all=True))
    with (root / "reports" / "no-op-verified.json").open("x") as output:
        json.dump(control, output, indent=2)
    print("NO_OP_VERIFIED " + json.dumps(control), flush=True)
    arms = []
    for percentage in (5, 10, 20):
        name = f"H1-alpha-{percentage:02d}"
        spec = root / "reports" / "patches" / f"{name}.json"
        run_child(root, "patch.py", "--copy", name, "--spec", spec)
        try:
            run_child(root, "benchmark.py", "--name", name, "--checkpoint", root / "copies" / name)
            result_dir = root / "reports" / name
            comparison = read_results(result_dir / "results.jsonl")
            if baseline.keys() != comparison.keys():
                raise ValueError("Intervention case coverage differs")
            for identifier, before in baseline.items():
                if before["input_sha256"] != comparison[identifier].get("input_sha256"):
                    raise ValueError(f"Intervention input changed: {identifier}")
            isolation = compare_activations(baseline_dir / "activations.npz", result_dir / "activations.npz")
            with (result_dir / "ISOLATION.json").open("x") as output:
                json.dump(isolation, output, indent=2)
            summary = json.loads((result_dir / "summary.json").read_text())
            arms.append({"name": name, "primary": percentage == 10, "groups": summary["groups"],
                         "isolation": isolation, "results_sha256": summary["results_sha256"]})
            print("ARM_VERIFIED " + json.dumps({"name": name, "isolation": isolation}), flush=True)
        finally:
            run_child(root, "patch.py", "--copy", name, "--spec", spec, "--revert")
    complete = {"no_op": control, "arms": arms, "elapsed_s": time.monotonic() - started,
                "one_intervention_at_a_time": True, "all_experimental_copies_reverted": True}
    with experiment_path.open("x") as output:
        json.dump(complete, output, indent=2)
    print("EXPERIMENT_COMPLETE " + json.dumps(complete), flush=True)


class ExperimentTests(unittest.TestCase):
    def example(self):
        return {"case": {"id": "case", "input_sha256": "input", "tokens": [1], "text": "A",
                         "correct": True, "letter_logits": [1, 0, 0, 0]}}

    def test_control_requires_exact_tokens(self):
        baseline, control = self.example(), self.example()
        self.assertEqual(compare_exact_rows(baseline, control, 1)["cases_exact"], 1)
        control["case"]["tokens"] = [2]
        with self.assertRaises(ValueError):
            compare_exact_rows(baseline, control, 1)

    def test_control_requires_identical_inputs(self):
        baseline, control = self.example(), self.example()
        control["case"]["input_sha256"] = "changed"
        with self.assertRaises(ValueError):
            compare_exact_rows(baseline, control, 1)

    def test_control_requires_complete_coverage(self):
        with self.assertRaises(ValueError):
            compare_exact_rows(self.example(), self.example(), 112)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["experiment.py"], verbosity=2)
    elif args.root:
        execute(args.root)
    else:
        parser.error("--root or --self-test required")