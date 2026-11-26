from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from prepare import file_hash, verify_inputs, write_json


def normalize_answer(text):
    return re.sub(r"[^a-z0-9.+-]", "", str(text).casefold()).rstrip(".")


def score_answer(text, gold):
    return normalize_answer(text) in {normalize_answer(answer) for answer in gold}


def diagnostic_cases():
    pairs = [
        ("What is 14 plus 27?", ["41"]),
        ("What is 83 minus 29?", ["54"]),
        ("What is 12 multiplied by 13?", ["156"]),
        ("What is 81 divided by 9?", ["9"]),
        ("What is three quarters of 20?", ["15"]),
        ("Convert 250 centimeters to meters. Give only the number.", ["2.5"]),
        ("A box holds 6 pens. How many pens do 4 boxes hold?", ["24"]),
        ("I had 31 tickets and gave away 8. How many remain?", ["23"]),
        ("What is the capital of France?", ["Paris"]),
        ("What is the capital of Japan?", ["Tokyo"]),
        ("What is the capital of Australia?", ["Canberra"]),
        ("What is the chemical symbol for oxygen?", ["O"]),
        ("How many sides does a hexagon have?", ["6", "six"]),
        ("Which is the largest planet in the Solar System?", ["Jupiter"]),
        ("At standard pressure, what is water's freezing point in Celsius? Give only the number.", ["0"]),
        ("How many minutes are in two hours?", ["120"]),
        ("Write the plural of child.", ["children"]),
        ("Write the opposite of hot.", ["cold"]),
        ("Reverse the letters in abc.", ["cba"]),
        ("Sort these numbers from smallest to largest: 9, 2, 5.", ["2, 5, 9"]),
        ("Which is prime: A) 21 B) 23 C) 25 D) 27? Reply with only the letter.", ["B"]),
        ("Convert the word hello to uppercase.", ["HELLO"]),
        ("Nora has a blue folder. Ravi has a red folder. What color is Nora's folder?", ["blue"]),
        ("In JSON, which lowercase literal represents Boolean truth?", ["true"]),
    ]
    return [
        {"id": f"question-{index:02d}", "text": question, "gold": gold}
        for index, (question, gold) in enumerate(pairs)
    ] + [
        {"id": "format-prose", "text": "The meeting starts at 09:30 in room 214. Bring the signed form."},
        {"id": "format-list", "text": "- Load the document.\n- Check all three totals.\n- Report any difference."},
        {"id": "format-code", "text": "```python\ndef add(left, right):\n    return left + right\n```"},
        {"id": "format-table", "text": "| Item | Count |\n| Pen | 12 |\n| Book | 3 |"},
        {"id": "format-long", "text": "On Monday the team counted twelve red boxes and seven green boxes in the north room. On Tuesday they moved five red boxes to the south room and left every green box where it was. The shipping note lists the destination, the number of boxes, the packing date, the person responsible for checking each label, and the time at which the delivery driver should arrive. Nothing in the note authorizes changing the contents of any box."},
        {"id": "format-empty", "text": ""},
    ]


def freeze(root: Path):
    experiment = root / "composite"
    frozen = experiment / "frozen"
    frozen.mkdir(exist_ok=False)
    all_cases = json.loads((experiment / "fixtures" / "cases.json").read_text(encoding="utf-8"))
    counts = Counter()
    native_cases = []
    for case in all_cases:
        limit = 4 if case["modality"] in ("text", "speech") else 2
        if case["modality"] == "text" and case.get("split") != "regression":
            continue
        if counts[case["modality"]] < limit:
            native_cases.append(case)
            counts[case["modality"]] += 1
    write_json(frozen / "cases.json", {"native": native_cases, "pipeline": diagnostic_cases()})
    files = [experiment / name for name in ("prepare.py", "model.py", "evaluate.py", "reference_c2c3.py")]
    files += [root / "runtime.py", experiment / "INPUTS.json", experiment / "package" / "TENSORS.json", experiment / "package" / "MANIFEST.json", frozen / "cases.json"]
    write_json(frozen / "MANIFEST.json", {str(path.relative_to(root)): file_hash(path) for path in files})
    print(json.dumps({"native_cases": len(native_cases), "native_modalities": counts, "pipeline_cases": len(diagnostic_cases()), "question_cases": 24}), flush=True)


def verify_frozen(root):
    experiment = root / "composite"
    manifest = json.loads((experiment / "frozen" / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, digest in manifest.items():
        if file_hash(root / relative) != digest:
            raise ValueError(f"frozen experiment input changed: {relative}")
    package = experiment / "package"
    metadata = json.loads((package / "MANIFEST.json").read_text(encoding="utf-8"))
    if file_hash(package / "model.safetensors") != metadata["sha256"]:
        raise ValueError("composite checkpoint changed")
    return json.loads((experiment / "frozen" / "cases.json").read_text(encoding="utf-8"))


def state_arrays(state, prefix=""):
    import numpy as np

    result = {}
    if isinstance(state, np.ndarray):
        result[prefix] = state
    elif isinstance(state, dict):
        for key, value in state.items():
            result.update(state_arrays(value, prefix + "." + str(key)))
    elif isinstance(state, (list, tuple)):
        for index, value in enumerate(state):
            result.update(state_arrays(value, prefix + "." + str(index)))
    return result


def compare_arrays(first, second):
    import numpy as np

    left = state_arrays(first)
    right = state_arrays(second)
    if left.keys() != right.keys():
        raise ValueError("state tensor keys differ")
    differences = [name for name in left if left[name].dtype != right[name].dtype or not np.array_equal(left[name], right[name])]
    if differences:
        raise ValueError(f"state values differ: {differences}")
    return len(left)


def decode_observed(runtime, state, zero_ncm=False):
    import numpy as np

    original_pointer = runtime._pointer_decode_from_zhat
    original_ncm = runtime._run_ncm_multiscale_levels
    trace = {"fallbacks": [], "zero_ncm": zero_ncm}

    def pointer(*args, **kwargs):
        text, fallback = original_pointer(*args, **kwargs)
        trace["fallbacks"].append(bool(fallback))
        return text, fallback

    def ncm(*args, **kwargs):
        result = (np.zeros((1, 64, 384), dtype=np.float32), 0, 0) if zero_ncm else original_ncm(*args, **kwargs)
        trace["ncm_z_sha256"] = hashlib.sha256(result[0].tobytes()).hexdigest()
        trace["codes"] = result[1]
        trace["levels"] = result[2]
        return result

    runtime._pointer_decode_from_zhat = pointer
    runtime._run_ncm_multiscale_levels = ncm
    try:
        text, metadata = runtime.c3_decode(state, full_ncm=True)
    finally:
        runtime._pointer_decode_from_zhat = original_pointer
        runtime._run_ncm_multiscale_levels = original_ncm
    return text, metadata, trace


def ablate(state, key, source=None):
    import numpy as np

    result = copy.deepcopy(state)
    if key == "c1_zero":
        result["h_primary"] = np.zeros_like(result["h_primary"])
        for window in result["windows"]:
            if not window.get("passthrough"):
                window["h_valid"] = np.zeros_like(window["h_valid"])
    elif key == "c2_identity":
        result["r_primary"] = result["h_primary"].copy()
        for window in result["windows"]:
            if not window.get("passthrough"):
                window["r_valid"] = window["h_valid"].copy()
    else:
        raise ValueError(key)
    return result


def run(root: Path):
    import numpy as np
    import torch
    from model import CompositeModel
    import reference_c2c3

    sys.path.insert(0, str(root))
    from runtime import OmniRuntime

    experiment = root / "composite"
    cases = verify_frozen(root)
    output = experiment / "results"
    output.mkdir(exist_ok=False)
    for name in ("native-reference", "native-composite", "states"):
        (output / name).mkdir()
    reference = OmniRuntime(experiment, checkpoint=experiment / "inputs" / "c1", threads=4)
    composite = CompositeModel(experiment / "package")
    reference_stages = reference_c2c3.OmniC2C3Runtime(experiment / "inputs" / "reference")
    if reference_stages.grammar_ids != composite.c2c3.grammar_ids or reference_stages.repeatable_grammar_ids != composite.c2c3.repeatable_grammar_ids:
        raise ValueError("decoder vocabulary control differs")
    bundled_runtime = OmniRuntime.__new__(OmniRuntime)
    bundled_runtime.root = experiment
    bundled_runtime.checkpoint = experiment / "package"
    bundled_runtime.model = composite.native
    bundled_runtime.processor = composite.processor
    bundled_runtime.codec = composite.codec
    bundled_runtime.torch = torch
    bundled_runtime.letter_ids = reference.letter_ids
    bundled_runtime.capture = {}
    bundled_runtime.handles = []
    bundled_runtime.parameter_versions = {name: parameter._version for name, parameter in composite.native.named_parameters()}
    bundled_runtime.add_hooks()
    native_results = []
    for case in cases["native"]:
        before, before_states = reference.run_case(case, output / "native-reference")
        after, after_states = bundled_runtime.run_case(case, output / "native-composite")
        for key in ("tokens", "text", "input_sha256", "speech", "correct"):
            if before.get(key) != after.get(key):
                raise ValueError(f"native output mismatch: {case['id']}: {key}")
        count = compare_arrays(before_states, after_states)
        np.savez_compressed(output / "states" / f"native-{case['id']}.npz", **before_states)
        native_results.append({"id": case["id"], "modality": case["modality"], "exact": True, "activation_arrays": count, "reference": before, "composite": after})
        write_json(output / f"native-{case['id']}.json", native_results[-1])
        print(json.dumps({"native_case": case["id"], "exact": True}), flush=True)

    def reference_states(token_ids):
        with torch.inference_mode():
            result = reference.model.model.language_model(input_ids=torch.tensor([token_ids], dtype=torch.long), use_cache=False, return_dict=True)
            return result.last_hidden_state[0].float().numpy()

    pipeline_results = []
    system = "Return only the requested answer, without explanation."
    prefix = reference.processor.tokenizer.encode(
        reference.processor.tokenizer.bos_token + "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n",
        add_special_tokens=False,
    )
    for case in cases["pipeline"]:
        text = case["text"]
        started = time.perf_counter()
        reference_c1, _ = reference_stages.c1_infer_states(text, reference_states, prefix_ids=prefix)
        reference_c2, _ = reference_stages.c2_plan(text, c1_state=reference_c1, jepa_arch="none")
        reference_text, reference_metadata, reference_trace = decode_observed(reference_stages, reference_c2)
        reference_seconds = time.perf_counter() - started
        started = time.perf_counter()
        combined_text, combined_metadata, combined_c1, combined_c2 = composite.compose(text, prefix_ids=prefix)
        combined_seconds = time.perf_counter() - started
        count = compare_arrays(reference_c1, combined_c1) + compare_arrays(reference_c2, combined_c2)
        traced_text, _, combined_trace = decode_observed(composite.c2c3, combined_c2)
        if reference_text != combined_text or traced_text != combined_text or reference_trace != combined_trace:
            raise ValueError(f"pipeline output or decoder trace mismatch: {case['id']}")
        for key in ("codes", "levels", "windows"):
            if reference_metadata.get(key) != combined_metadata["c3"].get(key):
                raise ValueError(f"pipeline decoder metadata mismatch: {case['id']}: {key}")
        record = {
            "id": case["id"], "input": text, "reference": reference_text, "composite": combined_text,
            "exact": True, "state_arrays": count, "reference_s": reference_seconds, "composite_s": combined_seconds,
            "trace": combined_trace, "ablations": {}, "echoes_input": combined_text.strip() == text.strip(),
        }
        saved = {"c1" + name: value for name, value in state_arrays(combined_c1).items()}
        saved.update({"c2" + name: value for name, value in state_arrays(combined_c2).items()})
        np.savez_compressed(output / "states" / f"pipeline-{case['id']}.npz", **saved)
        for arm in ("c1_zero", "c2_identity", "c3_zero"):
            if arm == "c1_zero":
                altered_c1 = ablate(combined_c1, arm)
                altered_c2, _ = composite.c2c3.c2_plan(text, c1_state=altered_c1, jepa_arch="none")
            elif arm == "c2_identity":
                altered_c2 = ablate(combined_c2, arm)
            else:
                altered_c2 = combined_c2
            answer, _, trace = decode_observed(composite.c2c3, altered_c2, zero_ncm=arm == "c3_zero")
            record["ablations"][arm] = {
                "text": answer, "trace": trace,
                "changed_output": answer != combined_text,
                "changed_ncm_state": trace.get("ncm_z_sha256") != combined_trace.get("ncm_z_sha256"),
            }
        if "gold" in case:
            raw_case = {"id": case["id"], "modality": "text", "prompt": text, "system": system, "max_new_tokens": 32}
            native, _ = reference.run_case(raw_case, output / "native-reference")
            record.update(gold=case["gold"], native_text=native["text"], native_correct=score_answer(native["text"], case["gold"]), composite_correct=score_answer(combined_text, case["gold"]))
            for ablation in record["ablations"].values():
                ablation["correct"] = score_answer(ablation["text"], case["gold"])
        pipeline_results.append(record)
        write_json(output / f"pipeline-{case['id']}.json", record)
        print(json.dumps({"pipeline_case": case["id"], "exact": True, "echo": record["echoes_input"], "fallbacks": combined_trace["fallbacks"]}), flush=True)

    verify_frozen(root)
    verify_inputs(experiment, check_sources=True)
    questions = [item for item in pipeline_results if "gold" in item]
    summary = {
        "native_cases_exact": len(native_results),
        "native_modalities": dict(Counter(item["modality"] for item in native_results)),
        "native_activation_arrays_exact": sum(item["activation_arrays"] for item in native_results),
        "pipeline_cases_exact": len(pipeline_results),
        "pipeline_state_arrays_exact": sum(item["state_arrays"] for item in pipeline_results),
        "questions": len(questions),
        "native_correct": sum(item["native_correct"] for item in questions),
        "composite_correct": sum(item["composite_correct"] for item in questions),
        "pipeline_echoes": sum(item["echoes_input"] for item in pipeline_results),
        "question_echoes": sum(item["echoes_input"] for item in questions),
        "fallback_windows": sum(sum(item["trace"]["fallbacks"]) for item in pipeline_results),
        "active_windows": sum(len(item["trace"]["fallbacks"]) for item in pipeline_results),
        "max_ncm_levels": max(item["trace"].get("levels", 0) for item in pipeline_results),
        "ablations": {
            arm: {
                "changed_outputs": sum(item["ablations"][arm]["changed_output"] for item in pipeline_results),
                "changed_ncm_states": sum(item["ablations"][arm]["changed_ncm_state"] for item in pipeline_results),
                "questions_correct": sum(item["ablations"][arm]["correct"] for item in questions),
            }
            for arm in ("c1_zero", "c2_identity", "c3_zero")
        },
        "source_and_copy_hashes_unchanged": True,
        "tensor_roundtrip_exact": True,
        "quality_claim": "Diagnostic comparison only, not a held-out improvement claim.",
        "runtime_scope": "Native BF16 CPU Omni; frozen C2/C3 NumPy computation; JEPA/tools/memory excluded equally from both arms.",
    }
    write_json(output / "SUMMARY.json", summary)
    print("COMPOSITE_COMPLETE " + json.dumps(summary), flush=True)


class ScorerTests(unittest.TestCase):
    def test_answer_forms(self):
        self.assertTrue(score_answer("41.", ["41"]))
        self.assertTrue(score_answer("Paris", ["paris"]))
        self.assertTrue(score_answer("2, 5, 9", ["2,5,9"]))

    def test_question_echo_fails(self):
        for case in diagnostic_cases():
            if "gold" in case:
                self.assertFalse(score_answer(case["text"], case["gold"]))

    def test_wrong_answer_fails(self):
        self.assertFalse(score_answer("42", ["41"]))
        self.assertFalse(score_answer("The capital might be Paris or Lyon.", ["Paris"]))

    def test_questions_not_duplicate(self):
        cases = diagnostic_cases()
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        self.assertEqual(sum("gold" in case for case in cases), 24)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root and args.freeze:
        freeze(args.root.resolve(strict=True))
    elif args.root and args.run:
        run(args.root.resolve(strict=True))
    else:
        parser.error("choose --self-test or --root with --freeze/--run")


if __name__ == "__main__":
    main()