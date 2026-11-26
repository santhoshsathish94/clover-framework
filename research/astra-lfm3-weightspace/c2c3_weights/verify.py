from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import unittest
from pathlib import Path

from experiment import means_by_text, sign_test, verify_frozen
from raw_runtime import initialize_paths


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_close(actual, expected, name):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"statistic mismatch: {name}")


def verify_scores(experiment):
    results = experiment / "results"
    summary = read_json(results / "SUMMARY.json")
    c2_selection = read_json(results / "C2_SELECTION.json")
    c2_choice = c2_selection["selected"]
    c2_discovery = {name: read_json(results / f"c2-discovery-{name}.json") for name in c2_selection["gains"]}
    chosen = min(("least_squares", "norm_match"), key=lambda name: sum(row["nll"] for row in c2_discovery[name]))
    if chosen != c2_choice or summary["c2"]["selected"] != chosen or summary["c2"]["gains"] != c2_selection["gains"]:
        raise ValueError("C2 selection differs from discovery")
    c2_rows = {name: read_json(results / f"c2-heldout-{name}.json") for name in ("original", c2_choice, "zero_input", "true_token")}
    c2_means = {name: means_by_text(rows, "nll") for name, rows in c2_rows.items()}
    for name, values in c2_means.items():
        require_close(summary["c2"]["heldout_mean_nll"][name], statistics.mean(values.values()), "C2 mean " + name)
    c2_tests = {name: sign_test(list(c2_means[c2_choice].values()), list(c2_means[name].values())) for name in ("original", "zero_input")}
    if summary["c2"]["paired_tests"] != c2_tests:
        raise ValueError("C2 paired statistics differ")
    c2_pass = all(test["one_sided_p"] <= 0.05 for test in c2_tests.values()) and all(statistics.mean(c2_means[c2_choice].values()) < statistics.mean(c2_means[name].values()) for name in ("original", "zero_input"))
    if summary["c2"]["passes_feedback_gate"] != c2_pass:
        raise ValueError("C2 acceptance flag differs")

    c3_choice = read_json(results / "C3_SELECTION.json")["selected"]
    c3_discovery = {name: read_json(results / f"c3-discovery-{name}.json") for name in ("0.5", "1.0", "2.0")}
    chosen = min(("0.5", "2.0"), key=lambda name: sum(row["code_nll"] for row in c3_discovery[name]))
    if chosen != c3_choice or summary["c3"]["selected_bridge_factor"] != chosen:
        raise ValueError("C3 selection differs from discovery")
    c3_rows = {name: read_json(results / f"c3-heldout-{name}.json") for name in ("1.0", c3_choice)}
    for name, rows in c3_rows.items():
        require_close(summary["c3"]["heldout_mean_code_nll"][name], statistics.mean(row["code_nll"] for row in rows), "C3 mean " + name)
        actual = {"correct": sum(row["generated_correct"] for row in rows), "total": sum(row["target_count"] for row in rows), "oracle_correct": sum(row["oracle_correct"] for row in rows)}
        if summary["c3"]["generated_tokens"][name] != actual:
            raise ValueError("C3 generated-token count differs")
    c3_test = sign_test([row["code_nll"] for row in c3_rows[c3_choice]], [row["code_nll"] for row in c3_rows["1.0"]])
    if summary["c3"]["paired_test"] != c3_test:
        raise ValueError("C3 paired statistics differ")
    c3_pass = c3_test["one_sided_p"] <= 0.05 and summary["c3"]["generated_tokens"][c3_choice]["correct"] >= summary["c3"]["generated_tokens"]["1.0"]["correct"] and summary["c3"]["heldout_mean_code_nll"][c3_choice] < summary["c3"]["heldout_mean_code_nll"]["1.0"]
    if summary["c3"]["passes_gate"] != c3_pass:
        raise ValueError("C3 acceptance flag differs")
    return summary, c2_discovery, c2_rows, c3_discovery, c3_rows


def audit(root, restore=False):
    experiment = initialize_paths(root)
    cases = verify_frozen(root)
    import torch
    from observe import tensor_leaves

    sys.path.insert(0, str(root / "composite"))
    from prepare import file_hash, tensor_digest, write_json

    summary, c2_discovery, c2_rows, c3_discovery, c3_rows = verify_scores(experiment)
    if restore and (summary["c2"]["passes_feedback_gate"] or summary["c3"]["passes_gate"]):
        raise ValueError("automatic restoration is only for this rejected trial set")
    identifiers = {split: {case["id"] for case in cases if case["split"] == split} for split in ("discovery", "heldout")}
    for groups, split in ((c2_discovery, "discovery"), (c3_discovery, "discovery"), (c2_rows, "heldout"), (c3_rows, "heldout")):
        for rows in groups.values():
            if {row["id"] for row in rows} != identifiers[split]:
                raise ValueError("case coverage differs from frozen split")
            if any(not math.isfinite(row.get("nll", row.get("code_nll"))) for row in rows):
                raise ValueError("nonfinite loss record")
    originals = read_json(experiment / "OBSERVATION.json")["files"] + read_json(experiment / "RUNTIME_INPUTS.json")["files"]
    for item in originals:
        if file_hash(Path(item["source"])) != item["sha256"] or file_hash(experiment / item["copy"]) != item["sha256"]:
            raise ValueError("source or unedited copied checkpoint changed")
    audits = []
    for path in sorted((experiment / "copies").glob("*/PATCH_AUDIT.json")):
        record = read_json(path)
        directory = path.parent
        source = Path(record["copy"]["source"])
        destination = experiment / record["copy"]["copy"]
        original = torch.load(source, weights_only=True, map_location="cpu", mmap=True)
        stored = torch.load(destination, weights_only=True, map_location="cpu", mmap=True)
        before = {name: tensor_digest(value) for name, value in tensor_leaves(original)}
        after = {name: tensor_digest(value) for name, value in tensor_leaves(stored)}
        restored_path = directory / "REVERSAL.json"
        if restored_path.exists():
            if after != before or file_hash(destination) != record["copy"]["sha256"]:
                raise ValueError("restored checkpoint differs from its source")
        else:
            if before != record["before"] or after != record["after"] or file_hash(destination) != record["checkpoint_sha256"]:
                raise ValueError("trial tensor or checkpoint hash mismatch")
            changed = [name for name in before if before[name] != after.get(name)]
            if sorted(changed) != sorted(record["changed_tensors"]):
                raise ValueError("trial changed an undeclared tensor")
            expected = (original["post_ln.weight"].float() * record["factor"]).to(original["post_ln.weight"].dtype) if record["component"] == "c2" else original["lfm3_combo_bridge"]["scale"] * record["factor"]
            actual = stored["post_ln.weight"] if record["component"] == "c2" else stored["lfm3_combo_bridge"]["scale"]
            if not torch.equal(expected, actual):
                raise ValueError("trial does not implement declared weight factor")
            if record["component"] == "c2":
                expected_bias = (original["post_ln.bias"].float() * record["factor"]).to(original["post_ln.bias"].dtype)
                if not torch.equal(expected_bias, stored["post_ln.bias"]):
                    raise ValueError("C2 post-normalization bias was not scaled consistently")
            if restore:
                temporary = directory / "restored.pt"
                shutil.copyfile(source, temporary)
                temporary.replace(destination)
                if file_hash(destination) != file_hash(source):
                    raise ValueError("byte-exact reversal failed")
                write_json(restored_path, {"restored": True, "sha256": file_hash(destination), "source_sha256": file_hash(source), "trial_sha256": record["checkpoint_sha256"]})
        audits.append({"trial": directory.name, "factor": record["factor"], "changed_tensors": record["changed_tensors"], "restored": restored_path.exists()})
    if len(audits) != 6:
        raise ValueError("expected exactly six audited trial copies")

    planner = experiment / "planner_probe"
    protocol = read_json(planner / "PROTOCOL.json")
    for relative, expected in protocol["source_hashes"].items():
        if file_hash(experiment / relative) != expected:
            raise ValueError("planner probe source changed")
    if file_hash(experiment / "planner_probe.py") != protocol["script_sha256"]:
        raise ValueError("planner probe script changed")
    probe_rows = [read_json(path) for path in planner.glob("*.json") if path.name not in ("PROTOCOL.json", "SUMMARY.json")]
    if len(probe_rows) != 4 or any(row["python_calls"] or row["search_calls"] for row in probe_rows):
        raise ValueError("invalid planner probe coverage or tool execution")
    return {
        "verified": True, "source_and_copy_records_unchanged": len(originals),
        "trial_audits": audits, "summary": summary,
        "planner_usage": read_json(planner / "SUMMARY.json"),
        "all_trials_restored": all(record["restored"] for record in audits),
        "artifact_hashes": {str(path.relative_to(experiment)): file_hash(path) for path in sorted(experiment.rglob("*.json")) if "cache" not in path.parts and "c2" not in path.relative_to(experiment).parts and path.name != "VERIFICATION.json"},
    }


class AuditTests(unittest.TestCase):
    def test_close_rejects_bad_score(self):
        with self.assertRaises(ValueError):
            require_close(0.1, 0.2, "test")

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            require_close(float("nan"), 0.2, "test")

    def test_paired_groups_not_tokens(self):
        values = means_by_text([{"id": "first", "nll": 1.0}, {"id": "first", "nll": 3.0}, {"id": "second", "nll": 6.0}], "nll")
        self.assertEqual(values, {"first": 2.0, "second": 6.0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root:
        root = args.root.resolve(strict=True)
        result = audit(root, args.restore)
        output = root / "c2c3_weights" / "VERIFICATION.json"
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "artifact_hashes"}), flush=True)
    else:
        parser.error("choose --self-test or --root")


if __name__ == "__main__":
    main()