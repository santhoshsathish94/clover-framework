from __future__ import annotations

import argparse
import copy
import json
import unittest
from collections import Counter
from pathlib import Path

from evaluate import score_answer, verify_frozen
from prepare import file_hash, verify_inputs, write_json


def check_pipeline(record, case):
    if record["id"] != case["id"] or record["input"] != case["text"]:
        raise ValueError("pipeline case identity mismatch")
    if not record["exact"] or record["reference"] != record["composite"]:
        raise ValueError("pipeline output mismatch")
    if record["echoes_input"] != (record["composite"].strip() == case["text"].strip()):
        raise ValueError("incorrect echo label")
    if "gold" in case:
        if record["gold"] != case["gold"]:
            raise ValueError("gold answer changed")
        for label in ("native", "composite"):
            text = record["native_text"] if label == "native" else record["composite"]
            if record[label + "_correct"] != score_answer(text, case["gold"]):
                raise ValueError("incorrect answer score")
    for arm in ("c1_zero", "c2_identity", "c3_zero"):
        result = record["ablations"][arm]
        if result["changed_output"] != (result["text"] != record["composite"]):
            raise ValueError("incorrect ablation output label")
        changed = result["trace"].get("ncm_z_sha256") != record["trace"].get("ncm_z_sha256")
        if result["changed_ncm_state"] != changed:
            raise ValueError("incorrect ablation state label")
        if arm == "c3_zero" and (not result["trace"]["zero_ncm"] or result["trace"].get("codes", 0) != 0):
            raise ValueError("NCM ablation did not bypass NCM")
        if "gold" in case and result["correct"] != score_answer(result["text"], case["gold"]):
            raise ValueError("incorrect ablation answer score")


def verify(root: Path):
    import numpy as np

    experiment = root / "composite"
    cases = verify_frozen(root)
    inputs = verify_inputs(experiment, check_sources=True)
    output = experiment / "results"
    summary = json.loads((output / "SUMMARY.json").read_text(encoding="utf-8"))
    expected_native = {case["id"] for case in cases["native"]}
    expected_pipeline = {case["id"] for case in cases["pipeline"]}
    if {path.stem[len("native-"):] for path in output.glob("native-*.json")} != expected_native:
        raise ValueError("native result coverage differs from frozen cases")
    if {path.stem[len("pipeline-"):] for path in output.glob("pipeline-*.json")} != expected_pipeline:
        raise ValueError("pipeline result coverage differs from frozen cases")
    native = []
    for case in cases["native"]:
        record = json.loads((output / f"native-{case['id']}.json").read_text(encoding="utf-8"))
        if record["id"] != case["id"] or record["modality"] != case["modality"] or not record["exact"]:
            raise ValueError("native case identity mismatch")
        for key in ("id", "modality", "tokens", "text", "input_sha256", "speech", "correct"):
            if record["reference"].get(key) != record["composite"].get(key):
                raise ValueError(f"native result differs: {case['id']}: {key}")
        with np.load(output / "states" / f"native-{case['id']}.npz", allow_pickle=False) as arrays:
            if len(arrays.files) != record["activation_arrays"] or not all(np.isfinite(arrays[name]).all() for name in arrays.files):
                raise ValueError("invalid native activation record")
        native.append(record)
    pipeline = []
    for case in cases["pipeline"]:
        record = json.loads((output / f"pipeline-{case['id']}.json").read_text(encoding="utf-8"))
        check_pipeline(record, case)
        with np.load(output / "states" / f"pipeline-{case['id']}.npz", allow_pickle=False) as arrays:
            if len(arrays.files) != record["state_arrays"]:
                raise ValueError("wrong saved state array count")
            for name in arrays.files:
                if arrays[name].shape[-1] != 2048 or not np.isfinite(arrays[name]).all():
                    raise ValueError("invalid saved C1/C2 state")
        pipeline.append(record)
    questions = [record for record in pipeline if "gold" in record]
    measured = {
        "native_cases_exact": len(native),
        "native_modalities": dict(Counter(record["modality"] for record in native)),
        "native_activation_arrays_exact": sum(record["activation_arrays"] for record in native),
        "pipeline_cases_exact": len(pipeline),
        "pipeline_state_arrays_exact": sum(record["state_arrays"] for record in pipeline),
        "questions": len(questions),
        "native_correct": sum(score_answer(record["native_text"], record["gold"]) for record in questions),
        "composite_correct": sum(score_answer(record["composite"], record["gold"]) for record in questions),
        "pipeline_echoes": sum(record["composite"].strip() == record["input"].strip() for record in pipeline),
        "question_echoes": sum(record["composite"].strip() == record["input"].strip() for record in questions),
        "fallback_windows": sum(sum(record["trace"]["fallbacks"]) for record in pipeline),
        "active_windows": sum(len(record["trace"]["fallbacks"]) for record in pipeline),
        "max_ncm_levels": max(record["trace"].get("levels", 0) for record in pipeline),
        "ablations": {
            arm: {
                "changed_outputs": sum(record["ablations"][arm]["text"] != record["composite"] for record in pipeline),
                "changed_ncm_states": sum(record["ablations"][arm]["trace"].get("ncm_z_sha256") != record["trace"].get("ncm_z_sha256") for record in pipeline),
                "questions_correct": sum(score_answer(record["ablations"][arm]["text"], record["gold"]) for record in questions),
            }
            for arm in ("c1_zero", "c2_identity", "c3_zero")
        },
    }
    for key, value in measured.items():
        if summary[key] != value:
            raise ValueError(f"summary differs from raw evidence: {key}")
    artifacts = [path for path in sorted(output.rglob("*")) if path.is_file()]
    return {
        "verified": True,
        "source_and_copy_files_checked": len(inputs["files"]),
        "results": measured,
        "artifact_hashes": {str(path.relative_to(experiment)): file_hash(path) for path in artifacts},
        "limitation": "The paired activation equality was asserted during inference; the archive keeps one matched copy of each array, not both.",
    }


class VerifierTests(unittest.TestCase):
    def record(self):
        case = {"id": "test", "text": "What is 1 plus 1?", "gold": ["2"]}
        trace = {"ncm_z_sha256": "original", "zero_ncm": False, "codes": 13}
        record = {"id": "test", "input": case["text"], "reference": case["text"], "composite": case["text"], "exact": True, "echoes_input": True, "gold": ["2"], "native_text": "2", "native_correct": True, "composite_correct": False, "trace": trace, "ablations": {}}
        for arm in ("c1_zero", "c2_identity", "c3_zero"):
            changed_trace = {"ncm_z_sha256": arm, "zero_ncm": arm == "c3_zero", "codes": 0 if arm == "c3_zero" else 13}
            record["ablations"][arm] = {"text": case["text"], "trace": changed_trace, "changed_output": False, "changed_ncm_state": True, "correct": False}
        return record, case

    def test_valid_record(self):
        record, case = self.record()
        check_pipeline(record, case)

    def test_corrupted_score_rejected(self):
        record, case = self.record()
        record["composite_correct"] = True
        with self.assertRaisesRegex(ValueError, "score"):
            check_pipeline(record, case)

    def test_corrupted_ablation_rejected(self):
        record, case = self.record()
        record["ablations"]["c1_zero"]["changed_ncm_state"] = False
        with self.assertRaisesRegex(ValueError, "state"):
            check_pipeline(record, case)

    def test_changed_gold_rejected(self):
        record, case = self.record()
        record["gold"] = [case["text"]]
        with self.assertRaisesRegex(ValueError, "gold"):
            check_pipeline(record, case)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root:
        result = verify(args.root.resolve(strict=True))
        if args.output:
            if not args.output.resolve().is_relative_to((args.root / "composite").resolve()):
                raise ValueError("verification output escapes the experiment")
            write_json(args.output, result)
        print(json.dumps({key: value for key, value in result.items() if key != "artifact_hashes"}), flush=True)
    else:
        parser.error("choose --self-test or --root")


if __name__ == "__main__":
    main()