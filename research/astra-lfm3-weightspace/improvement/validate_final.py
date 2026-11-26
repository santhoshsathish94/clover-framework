import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survey import verify_original


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate result IDs")
    return result


def group_comparison(original, edited):
    if original.keys() != edited.keys():
        raise ValueError("Different paired case coverage")
    groups = defaultdict(lambda: {"original": 0, "edited": 0, "cases": 0})
    fixed, broken = [], []
    for identifier, before in original.items():
        after = edited[identifier]
        if before["input_sha256"] != after["input_sha256"]:
            raise ValueError("Paired input differs")
        if before["family"] == "selection" or before["format"] != "mcq":
            continue
        group = groups[before["group"]]
        group["original"] += bool(before["correct"])
        group["edited"] += bool(after["correct"])
        group["cases"] += 1
        if not before["correct"] and after["correct"]:
            fixed.append(identifier)
        elif before["correct"] and not after["correct"]:
            broken.append(identifier)
    if any(group["cases"] != 4 for group in groups.values()):
        raise ValueError("Each underlying arithmetic question must have four rotations")
    better = sum(group["edited"] > group["original"] for group in groups.values())
    worse = sum(group["edited"] < group["original"] for group in groups.values())
    discordant = better + worse
    probability = sum(math.comb(discordant, count) for count in range(better, discordant + 1)) / 2 ** discordant if discordant else 1.0
    return {"underlying_questions": len(groups), "better_questions": better, "worse_questions": worse,
            "one_sided_exact_sign_p": probability, "fixed_answers": fixed, "broken_answers": broken,
            "original_correct": sum(group["original"] for group in groups.values()),
            "edited_correct": sum(group["edited"] for group in groups.values()),
            "total_responses": sum(group["cases"] for group in groups.values()), "per_question": dict(groups)}


def regression_comparison(original, edited):
    if original.keys() != edited.keys():
        raise ValueError("Regression case coverage differs")
    broken, speech_changed = [], []
    for identifier, before in original.items():
        after = edited[identifier]
        if before["input_sha256"] != after["input_sha256"]:
            raise ValueError("Regression input differs")
        if before["split"] != "regression":
            continue
        if before.get("correct") and not after.get("correct"):
            broken.append(identifier)
        if before.get("speech"):
            if not after.get("speech") or after["speech"]["samples"] <= 0 or after["speech"]["rms"] <= 0:
                broken.append(identifier)
            elif before["speech"]["codes_sha256"] != after["speech"]["codes_sha256"]:
                speech_changed.append(identifier)
    return {"newly_broken_previously_correct": broken, "speech_code_streams_changed": speech_changed,
            "speech_quality_limit": "Waveform integrity only; changed speech needs independent assessment."}


def child(root, relative, *args):
    print("FINAL_STAGE " + json.dumps({"script": relative, "args": list(map(str, args))}), flush=True)
    subprocess.run([sys.executable, "-B", str(root / relative), "--root", str(root), *map(str, args)], check=True)


def run(root, trial_record="TRIALS.json"):
    root = Path(root)
    if trial_record not in ("TRIALS.json", "FOLLOWUP_TRIALS.json"):
        raise ValueError("Unknown trial record")
    trial_path = root / "improvement" / trial_record
    trials = json.loads(trial_path.read_text())
    selected = trials["selected"]
    if selected is None:
        raise RuntimeError("No tuning candidate qualifies; do not expose final validation for a rejected hypothesis")
    selection = {"source_trial": selected, "trials_sha256": sha256(trial_path),
                 "frozen_before_final_evaluation": True}
    if selected.get("followup"):
        if sha256(root / "improvement" / "DOSE_FOLLOWUP.md") != selected["hypothesis_sha256"]:
            raise ValueError("Follow-up hypothesis changed")
        selection["followup_hypothesis_sha256"] = selected["hypothesis_sha256"]
    else:
        source_spec = root / "improvement" / "proposals" / f"{selected['name']}.json"
        if sha256(source_spec) != selected["spec_sha256"]:
            raise ValueError("Selected specification changed")
        spec = json.loads(source_spec.read_text())
        spec["name"] = "final-candidate"
        final_spec = source_spec.parent / "FINAL_SPEC.json"
        with final_spec.open("x") as output:
            json.dump(spec, output, indent=2)
        selection["final_spec_sha256"] = sha256(final_spec)
    with (root / "improvement" / "SELECTION.json").open("x") as output:
        json.dump(selection, output, indent=2)
    if selected.get("followup"):
        child(root, "improvement/dose_followup.py", "--copy-name", "final-candidate", "--dose", selected["strength"])
    else:
        child(root, "improvement/interventions.py", "--apply", final_spec)
    candidate = root / "improvement" / "copies" / "final-candidate"
    completed = False
    try:
        child(root, "improvement/evaluate.py", "--split", "final", "--name", "original-final")
        child(root, "improvement/evaluate.py", "--split", "final", "--name", "candidate-final", "--checkpoint", candidate)
        original_dir = root / "improvement" / "runs" / "original-final"
        edited_dir = root / "improvement" / "runs" / "candidate-final"
        original = read_rows(original_dir / "results.jsonl")
        edited = read_rows(edited_dir / "results.jsonl")
        comparison = group_comparison(original, edited)
        original_summary = json.loads((original_dir / "summary.json").read_text())
        edited_summary = json.loads((edited_dir / "summary.json").read_text())
        child(root, "benchmark.py", "--name", "improvement-regression", "--checkpoint", candidate)
        regression = regression_comparison(read_rows(root / "reports" / "baseline" / "results.jsonl"),
                                           read_rows(root / "reports" / "improvement-regression" / "results.jsonl"))
        regression_summary = json.loads((root / "reports" / "improvement-regression" / "summary.json").read_text())
        base_regression = json.loads((root / "reports" / "baseline" / "summary.json").read_text())
        asr_worse = regression_summary["groups"]["regression:audio_asr"]["corpus_word_error_rate"] > base_regression["groups"]["regression:audio_asr"]["corpus_word_error_rate"]
        protected_totals = all(edited_summary["totals"][key]["correct"] >= original_summary["totals"][key]["correct"]
                               for key in ("arithmetic_free", "selection"))
        passed = (comparison["edited_correct"] > comparison["original_correct"]
                  and comparison["one_sided_exact_sign_p"] <= 0.05
                  and protected_totals and not regression["newly_broken_previously_correct"] and not asr_worse)
        result = {"selected": selected, "final_comparison": comparison,
                  "initial_totals": original_summary["totals"], "final_totals": edited_summary["totals"],
                  "regression": regression, "regression_summary": regression_summary["groups"],
                  "asr_worse": asr_worse, "measured_retention_gate_passed": passed,
                  "decision": "scoped_candidate_requires_speech_assessment" if passed and regression["speech_code_streams_changed"] else "retain_scoped_improvement" if passed else "reject",
                  "original_checkpoint_sha256": json.loads((candidate / "PATCH_AUDIT.json").read_text())["base_checkpoint_sha256"],
                  "candidate_checkpoint_sha256": sha256(candidate / "model.safetensors"),
                  "candidate_path": str(candidate), "no_deployment": True,
                  "limits": ["Constructed arithmetic/selection validation, not universal model quality",
                             "Prior regression set is not a fresh generalization benchmark",
                             "Speech naturalness/intelligibility not established by waveform checks"]}
        verify_original(root)
        with (root / "improvement" / "FINAL_COMPARISON.json").open("x") as output:
            json.dump(result, output, indent=2)
        if not passed:
            child(root, "improvement/interventions.py", "--revert", "final-candidate")
        completed = True
        print("FINAL_COMPARISON_COMPLETE " + json.dumps({"decision": result["decision"], "initial": result["initial_totals"],
                                                       "final": result["final_totals"], "p": comparison["one_sided_exact_sign_p"],
                                                       "regression": regression}), flush=True)
    finally:
        if not completed and not (candidate / "REVERSAL.json").exists():
            child(root, "improvement/interventions.py", "--revert", "final-candidate")


class FinalComparisonTests(unittest.TestCase):
    def test_rotations_are_grouped(self):
        original, edited = {}, {}
        for group in range(5):
            for rotation in range(4):
                key = f"{group}-{rotation}"
                base = {"group": group, "family": "addition", "format": "mcq", "input_sha256": key, "correct": False}
                original[key] = base
                edited[key] = {**base, "correct": True}
        result = group_comparison(original, edited)
        self.assertEqual(result["underlying_questions"], 5)
        self.assertEqual(result["one_sided_exact_sign_p"], 1 / 32)
        self.assertEqual(len(result["fixed_answers"]), 20)

    def test_regression_does_not_disappear_in_aggregate(self):
        original = {"case": {"split": "regression", "input_sha256": "same", "correct": True}}
        edited = {"case": {"split": "regression", "input_sha256": "same", "correct": False}}
        self.assertEqual(regression_comparison(original, edited)["newly_broken_previously_correct"], ["case"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--trials", choices=("TRIALS.json", "FOLLOWUP_TRIALS.json"), default="TRIALS.json")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["validate_final.py"], verbosity=2)
    elif args.root:
        run(args.root, args.trials)
    else:
        parser.error("--root or --self-test required")