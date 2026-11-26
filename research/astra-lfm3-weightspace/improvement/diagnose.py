import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import OmniRuntime
from survey import verify_original


def numeric_answer(text):
    match = re.fullmatch(r"\s*([+-]?\d[\d,]*)(?:\.0+)?[.]?\s*", text)
    return int(match.group(1).replace(",", "")) if match else None


def build_cases(seed=731907):
    generator = random.Random(seed)
    items = []
    for index in range(24):
        first, second = generator.randint(13, 89), generator.randint(11, 67)
        family = ("addition", "subtraction", "multiplication", "selection")[index % 4]
        if family == "addition":
            question, answer = f"What is {first} plus {second}?", first + second
        elif family == "subtraction":
            first *= 7
            question, answer = f"What is {first} minus {second}?", first - second
        elif family == "multiplication":
            question, answer = f"What is {first} times {second}?", first * second
        else:
            question, answer = f"Which option is the number {first}?", first
        choices = [answer, answer - 1, answer + 1, answer + second]
        for rotation in range(4):
            rotated = choices[-rotation:] + choices[:-rotation] if rotation else choices.copy()
            choice_text = "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", rotated))
            items.append({
                "id": f"discovery-{index:03d}-r{rotation}", "group": index,
                "family": family, "modality": "text", "split": "discovery",
                "prompt": question + "\n" + choice_text + "\nAnswer with only the correct letter.",
                "gold_letter": "ABCD"[rotation], "gold_value": answer, "choices": rotated,
                "max_new_tokens": 24, "format": "mcq",
            })
        if family != "selection":
            items.append({
                "id": f"discovery-{index:03d}-free", "group": index,
                "family": family, "modality": "text", "split": "discovery",
                "prompt": question + " Reply with just the number.",
                "gold_value": answer, "max_new_tokens": 24, "format": "free",
            })
    return items


def summarize(cases, results):
    index = {case["id"]: case for case in cases}
    cells = defaultdict(lambda: {"correct": 0, "total": 0})
    groups = defaultdict(list)
    for result in results:
        case = index[result["id"]]
        label = case["format"] if case["format"] == "free" else f"mcq_gold_{case['gold_letter']}"
        for key in (f"{case['family']}:{label}", f"all:{case['format']}"):
            cells[key]["total"] += 1
            cells[key]["correct"] += bool(result["correct"])
        groups[case["group"]].append(result)
    disagreements = []
    for group, rows in groups.items():
        multiple_choice = [row for row in rows if index[row["id"]]["format"] == "mcq"]
        values = {row.get("predicted_value") for row in multiple_choice}
        if len(values) > 1:
            disagreements.append(group)
    return {"cells": dict(cells), "groups_with_rotation_value_disagreement": disagreements,
            "predicted_letters": dict(Counter(row.get("prediction") for row in results if index[row["id"]]["format"] == "mcq"))}


def reference_cases(cases):
    references = []
    for case in cases:
        if case["format"] != "mcq":
            continue
        choices = "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", case["choices"]))
        references.append({**case,
                           "prompt": f"Which option is the number {case['gold_value']}?\n{choices}\nAnswer with only the correct letter.",
                           "condition": "diagnostic_number_binding_reference"})
    return references


def run(root, reference=False):
    import numpy as np

    root = Path(root)
    verify_original(root)
    destination = root / "improvement" / ("diagnostic-reference" if reference else "diagnostic")
    destination.mkdir(exist_ok=False)
    cases = build_cases()
    if reference:
        cases = reference_cases(cases)
    (destination / "cases.json").write_text(json.dumps(cases, indent=2))
    runtime = OmniRuntime(root)
    torch = runtime.torch
    extra_hooks = []
    for layer_index in range(8, 24):
        layer = runtime.model.model.language_model.layers[layer_index]

        def layer_hook(module, inputs, output, layer_index=layer_index):
            key = f"diagnostic_layer_{layer_index}"
            if key not in runtime.capture:
                value = output if isinstance(output, torch.Tensor) else output[0]
                runtime.capture[key] = value[0, -1].detach().float().cpu().numpy().copy()

        def mlp_hook(module, inputs, layer_index=layer_index):
            key = f"diagnostic_mlp_{layer_index}"
            if key not in runtime.capture:
                runtime.capture[key] = inputs[0][0, -1].detach().float().cpu().numpy().copy()

        extra_hooks.append(layer.register_forward_hook(layer_hook))
        extra_hooks.append(layer.feed_forward.w2.register_forward_pre_hook(mlp_hook))
    arrays, results = {}, []
    started = time.monotonic()
    with (destination / "results.jsonl").open("x") as output:
        for case in cases:
            row, activations = runtime.run_case(case, destination)
            if case["format"] == "free":
                row["predicted_value"] = numeric_answer(row["text"])
                row["correct"] = row["predicted_value"] == case["gold_value"]
            else:
                prediction = row["prediction"]
                row["predicted_value"] = case["choices"]["ABCD".index(prediction)] if prediction else None
            row.update({"group": case["group"], "family": case["family"], "format": case["format"], "gold_value": case["gold_value"]})
            results.append(row)
            for key, value in activations.items():
                arrays[f"{case['id']}::{key}"] = value
            output.write(json.dumps(row) + "\n")
            output.flush()
            if len(results) % 20 == 0:
                print(json.dumps({"completed": len(results), "total": len(cases), "elapsed_s": time.monotonic() - started}), flush=True)
    for hook in extra_hooks:
        hook.remove()
    np.savez_compressed(destination / "activations.npz", **arrays)
    summary = summarize(cases, results)
    summary.update({"case_count": len(cases), "elapsed_s": time.monotonic() - started,
                    "scope": "Fresh discovery diagnostic, not held-out validation or a model improvement claim"})
    verify_original(root)
    summary["original_hashes_verified_before_and_after"] = True
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DISCOVERY_COMPLETE " + json.dumps(summary), flush=True)


class DiagnosticTests(unittest.TestCase):
    def test_rotations_preserve_answer_and_options(self):
        cases = build_cases()
        for group in range(24):
            choices = [case for case in cases if case["group"] == group and case["format"] == "mcq"]
            self.assertEqual(len(choices), 4)
            for case in choices:
                self.assertEqual(case["choices"]["ABCD".index(case["gold_letter"])], case["gold_value"])
                self.assertEqual(len(set(case["choices"])), 4)
                self.assertEqual(set(case["choices"]), set(choices[0]["choices"]))

    def test_numeric_scoring_does_not_reward_ambiguity(self):
        self.assertEqual(numeric_answer("1,234"), 1234)
        self.assertEqual(numeric_answer("-17."), -17)
        self.assertIsNone(numeric_answer("17 or 18"))
        self.assertIsNone(numeric_answer("The question contains 17"))

    def test_binding_reference_keeps_options_and_labels(self):
        cases = build_cases()
        originals = {case["id"]: case for case in cases}
        references = reference_cases(cases)
        self.assertEqual(len(references), 96)
        for reference in references:
            original = originals[reference["id"]]
            self.assertEqual(reference["choices"], original["choices"])
            self.assertEqual(reference["gold_letter"], original["gold_letter"])
            self.assertIn(str(original["gold_value"]), reference["prompt"].splitlines()[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["diagnose.py"], verbosity=2)
    elif args.root:
        run(args.root, args.reference)
    else:
        parser.error("--root or --self-test required")