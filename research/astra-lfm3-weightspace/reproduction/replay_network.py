import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import unittest

import numpy as np


TENSORS = {
    "W1": ("actor/dense1/kernel", (600, 32)),
    "b1": ("actor/dense1/bias", (32,)),
    "W2": ("actor/dense2/kernel", (32, 32)),
    "b2": ("actor/dense2/bias", (32,)),
    "W3": ("actor/logits/kernel", (32, 6)),
    "b3": ("actor/logits/bias", (6,)),
}


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def numeric_literal(node):
    for child in ast.walk(node):
        if not isinstance(child, (ast.Constant, ast.List, ast.Tuple, ast.Load, ast.UnaryOp, ast.USub, ast.UAdd)):
            raise ValueError("Only numeric literals and lists are allowed")
        if isinstance(child, ast.Constant) and type(child.value) not in (int, float):
            raise ValueError("Only numeric constants are allowed")
    value = np.asarray(ast.literal_eval(node), dtype=np.float32)
    if not np.isfinite(value).all() or np.any(np.abs(value) > 1e6):
        raise ValueError("Numeric value exceeds the published edit bounds")
    return value


def index_value(node):
    if node is None:
        return None
    value = ast.literal_eval(node)
    if type(value) is not int:
        raise ValueError("Indices must be literal integers")
    return value


def parse_index(node):
    if isinstance(node, ast.Tuple):
        return tuple(parse_index(element) for element in node.elts)
    if isinstance(node, ast.Slice):
        return slice(index_value(node.lower), index_value(node.upper), index_value(node.step))
    return index_value(node)


def replay(answer):
    blocks = re.findall(r"^```edits\s*\n(.*?)^```", answer, flags=re.MULTILINE | re.DOTALL)
    if not blocks:
        raise ValueError("No published edits block")
    arrays = {name: np.zeros(shape, dtype=np.float32) for name, (_, shape) in TENSORS.items()}
    program = ast.parse(blocks[-1], mode="exec")
    for statement in program.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            operation = "set"
        elif isinstance(statement, ast.AugAssign) and isinstance(statement.op, (ast.Add, ast.Sub, ast.Mult)):
            target = statement.target
            operation = type(statement.op).__name__
        else:
            raise ValueError("Only tensor slice assignments are allowed")
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            raise ValueError("Assignment must target a named tensor")
        name = target.value.id
        if name not in arrays:
            raise ValueError("Unknown tensor name")
        selection = parse_index(target.slice)
        value = numeric_literal(statement.value)
        if operation == "set":
            arrays[name][selection] = value
        elif operation == "Add":
            arrays[name][selection] += value
        elif operation == "Sub":
            arrays[name][selection] -= value
        elif operation == "Mult":
            arrays[name][selection] *= value
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Reconstruction is not finite")
    return arrays, len(program.body)


def observation(raw):
    if raw.shape != (10, 10, 4):
        raise ValueError("Expected the official Breakout four-channel state")
    result = np.zeros((10, 10, 6), dtype=np.float32)
    result[:, :, 0] = raw[:, :, 0]
    result[:, :, 2] = raw[:, :, 3]
    result[:, :, 3] = raw[:, :, 1]
    result[:, :, 4] = raw[:, :, 2]
    return result.reshape(600)


def forward(arrays, inputs):
    hidden1 = np.tanh(inputs @ arrays["W1"] + arrays["b1"])
    hidden2 = np.tanh(hidden1 @ arrays["W2"] + arrays["b2"])
    logits = hidden2 @ arrays["W3"] + arrays["b3"]
    return logits, hidden1, hidden2


def reconstruct(root):
    root = Path(root)
    source = root / "lm_understand-logs" / "exp5_write_from_scratch"
    output = root / "results" / "breakout-reconstruction"
    output.mkdir(parents=True, exist_ok=False)
    answer_path = source / "runs" / "breakout" / "reply" / "answer.md"
    reference_path = source / "networks" / "breakout" / "hand.npz"
    arrays, statements = replay(answer_path.read_text())
    reference = np.load(reference_path, allow_pickle=False)
    if set(reference.files) != {name for name, _ in TENSORS.values()}:
        raise ValueError("Reference tensor coverage differs")
    checked = []
    for name, (reference_name, shape) in TENSORS.items():
        expected = reference[reference_name]
        if expected.dtype != np.float32 or expected.shape != shape or not np.array_equal(arrays[name], expected):
            raise ValueError(f"Reconstructed tensor differs from published artifact: {name}")
        checked.append({"name": name, "reference_name": reference_name, "shape": list(shape),
                        "sha256": hashlib.sha256(arrays[name].tobytes()).hexdigest(),
                        "nonzero": int(np.count_nonzero(arrays[name]))})
    probe_rows = []
    for paddle in range(10):
        raw = np.zeros((10, 10, 4), dtype=bool)
        raw[9, paddle, 0] = True
        raw[7, 4, 1] = True
        raw[6, 3, 2] = True
        inputs = observation(raw)
        logits, hidden1, hidden2 = forward(arrays, inputs)
        prediction = int(np.argmax(logits))
        expected_action = 3 if paddle < 5 else 0 if paddle <= 6 else 1
        if prediction != expected_action:
            raise ValueError(f"Landing-interval hypothesis failed at paddle {paddle}: {prediction}")
        expected_arrays = {name: reference[reference_name] for name, (reference_name, _) in TENSORS.items()}
        reference_logits, _, _ = forward(expected_arrays, inputs)
        if not np.array_equal(logits, reference_logits):
            raise ValueError("Reconstruction/reference logits differ")
        probe_rows.append({"paddle_column": paddle, "ball": [7, 4], "trail": [6, 3],
                           "first_layer_units_0_to_5": hidden1[:6].tolist(),
                           "second_layer_units_0_to_7": hidden2[:8].tolist(),
                           "logits": logits.tolist(), "argmax_action": prediction,
                           "expected_action": expected_action})
    np.savez_compressed(output / "reconstructed.npz", **arrays)
    report = {"source_commit": "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f",
              "answer_sha256": digest(answer_path), "reference_npz_sha256": digest(reference_path),
              "statements": statements, "all_six_arrays_bit_identical": True,
              "parameter_count": sum(value.size for value in arrays.values()),
              "nonzero_count": sum(entry["nonzero"] for entry in checked),
              "tensor_hashes": checked, "landing_interval_probes": probe_rows,
              "qualification": "Replay of published weight program, not a new LLM discovery; gameplay not yet evaluated."}
    if statements != 92 or report["parameter_count"] != 20486 or report["nonzero_count"] != 815:
        raise ValueError("Reconstruction counts disagree with the published record")
    with (output / "result.json").open("x") as destination:
        json.dump(report, destination, indent=2)
    print("RECONSTRUCTION_VERIFIED " + json.dumps({key: report[key] for key in (
        "statements", "all_six_arrays_bit_identical", "parameter_count", "nonzero_count",
    )}), flush=True)


class ReplayTests(unittest.TestCase):
    def test_numeric_program(self):
        arrays, count = replay("```edits\nW1[0:2, 0] = [1, -2]\nW1[0:2, 0] *= 2\nb3[:] = -1\n```")
        np.testing.assert_array_equal(arrays["W1"][:2, 0], [2, -4])
        np.testing.assert_array_equal(arrays["b3"], [-1] * 6)
        self.assertEqual(count, 3)

    def test_no_execution(self):
        for program in ("import os", "W1[0,0] = __import__('os').getcwd()", "W1[0,0] = True", "W1[0,0] = 1e9"):
            with self.assertRaises(ValueError):
                replay("```edits\n" + program + "\n```")

    def test_last_block_only(self):
        arrays, _ = replay("```edits\nb1[0]=1\n```\n```edits\nb1[0]=2\n```")
        self.assertEqual(arrays["b1"][0], 2)

    def test_channel_mapping(self):
        raw = np.zeros((10, 10, 4), dtype=bool)
        raw[9, 2, 0] = raw[3, 4, 1] = raw[2, 3, 2] = raw[1, 1, 3] = True
        mapped = observation(raw).reshape(10, 10, 6)
        self.assertEqual(int(mapped.sum()), 4)
        self.assertEqual(mapped[9, 2, 0], 1)
        self.assertEqual(mapped[3, 4, 3], 1)
        self.assertEqual(mapped[2, 3, 4], 1)
        self.assertEqual(mapped[1, 1, 2], 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["replay_network.py"], verbosity=2)
    elif args.root:
        reconstruct(args.root)
    else:
        parser.error("--root or --self-test required")