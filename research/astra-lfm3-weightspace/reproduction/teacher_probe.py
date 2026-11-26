import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import unittest

import numpy as np

from qwen_probe import load_model, verify_original


EXPECTED_LENGTHS = [124, 123, 131, 136, 107, 146, 98, 135]
DIRECTION_WINDOW = 36
WRITER_WINDOW = 24


def paired_direction(means):
    delta = (means[1::2] - means[::2]).mean(0)
    length = np.linalg.norm(delta)
    if length == 0:
        raise ValueError("No matched-context direction")
    return delta / length


def writer_scores(mlp_means, weights, direction):
    differences = mlp_means[1::2] - mlp_means[::2]
    per_pair = differences * (direction @ weights)[None, :]
    return per_pair.mean(0), (per_pair > 0).sum(0)


def orthogonal_control(original, edited, direction, columns, seed):
    generator = np.random.default_rng(seed)
    random_direction = generator.standard_normal(original.shape[0])
    random_direction -= direction * (direction @ random_direction)
    random_direction /= np.linalg.norm(random_direction)
    target_norm = np.linalg.norm((edited - original).astype(np.float64))
    delta = -np.outer(random_direction, random_direction @ original[:, columns])
    delta *= target_norm / np.linalg.norm(delta)
    controlled = original.astype(np.float64).copy()
    controlled[:, columns] += delta
    return controlled.astype(np.float32), {
        "seed": seed, "absolute_direction_cosine": float(abs(direction @ random_direction)),
        "target_perturbation_norm": float(target_norm),
        "pre_round_perturbation_norm": float(np.linalg.norm(delta)),
    }


def probe(root, name="qwen-teacher-probe", include_control=False):
    root = Path(root)
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from safetensors import safe_open

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    verify_original(root)
    source = root / "lm_understand-logs" / "exp8_math"
    request_path = source / "runs" / "p50c" / "requests" / "fwd_4.json.claimed"
    request = json.loads(request_path.read_text())
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Result name must be one directory name")
    output = root / "results" / name
    output.mkdir(exist_ok=False)
    model, tokenizer, info = load_model(root, "original")
    tokenized = [tokenizer(text, add_special_tokens=False, return_tensors="pt") for text in request["texts"]]
    lengths = [item["input_ids"].shape[1] for item in tokenized]
    if lengths != EXPECTED_LENGTHS:
        raise ValueError(f"Token lengths differ from published fwd4: observed {lengths}, expected {EXPECTED_LENGTHS}")
    selected = json.loads((source / "runs" / "p50c" / "work_notes" / "frozen_weights_manifest.json").read_text())
    derived = np.load(root / "results" / "published-audit" / "derived-edit-directions.npz", allow_pickle=False)
    snapshots, weights = {}, {}
    hooks = []
    current = {}
    for layer_index in (15, 19):
        layer = model.model.language_model.layers[layer_index]
        weights[layer_index] = layer.mlp.down_proj.weight.detach().float().cpu().numpy().copy()

        def hidden_hook(module, inputs, outputs, layer_index=layer_index):
            value = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            current[f"hidden_{layer_index + 1}"] = value[0, -DIRECTION_WINDOW:].detach().float().mean(0).cpu().numpy().copy()

        def mlp_hook(module, inputs, layer_index=layer_index):
            current[f"mlp_{layer_index}"] = inputs[0][0, -WRITER_WINDOW:].detach().float().mean(0).cpu().numpy().copy()

        hooks.append(layer.register_forward_hook(hidden_hook))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(mlp_hook))

    all_rows = []
    control_metadata = {}
    start = time.monotonic()
    arms = ("original", "edited", "orthogonal_control") if include_control else ("original", "edited")
    for arm in arms:
        if arm != "original":
            with safe_open(root / "models" / "qwen-p50c.safetensors", framework="pt", device="cpu") as checkpoint:
                for layer_index in (15, 19):
                    name = f"model.language_model.layers.{layer_index}.mlp.down_proj.weight"
                    replacement = checkpoint.get_tensor(name).clone()
                    if arm == "orthogonal_control":
                        controlled, control_info = orthogonal_control(
                            weights[layer_index], replacement.float().numpy(), derived[f"layer{layer_index}_direction"],
                            derived[f"layer{layer_index}_columns"], 20261111 + layer_index,
                        )
                        replacement = torch.from_numpy(controlled).to(replacement.dtype)
                        control_info["post_round_perturbation_norm"] = float(np.linalg.norm(replacement.float().numpy() - weights[layer_index]))
                        control_metadata[str(layer_index)] = control_info
                    model.get_parameter(name).data = replacement
        for index, inputs in enumerate(tokenized):
            current.clear()
            result = model(**inputs, use_cache=False, logits_to_keep=37)
            logits = result.logits[:, :-1].float()
            targets = inputs["input_ids"][:, -36:]
            token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            mean_logprob = float(token_logprobs.mean())
            record = {"arm": arm, "text_index": index, "kind": "productive" if index % 2 == 0 else "repetitive",
                      "pair": index // 2, "tokens": lengths[index], "mean_last36_token_logprob": mean_logprob}
            all_rows.append(record)
            for key, value in current.items():
                snapshots[f"{arm}_{index}_{key}"] = value
            print("TEACHER_TEXT " + json.dumps(record), flush=True)
            del result, logits, token_logprobs
    for hook in hooks:
        hook.remove()
    measurements = {}
    for layer_index in (15, 19):
        means = np.stack([snapshots[f"original_{index}_hidden_{layer_index + 1}"] for index in range(8)])
        mlp_means = np.stack([snapshots[f"original_{index}_mlp_{layer_index}"] for index in range(8)])
        direction = paired_direction(means)
        scores, consistency = writer_scores(mlp_means, weights[layer_index], direction)
        top64 = np.argsort(scores)[-64:]
        published_columns = selected[str(layer_index)]["selected_neurons"]
        reconstructed_direction = derived[f"layer{layer_index}_direction"]
        measurements[str(layer_index)] = {
            "absolute_cosine_to_actual_edit_direction": float(abs(direction @ reconstructed_direction)),
            "top64_overlap_with_published_selection": len(set(top64.tolist()) & set(published_columns)),
            "selection_only_in_reproduction": sorted(set(top64.tolist()) - set(published_columns)),
            "selection_only_in_published": sorted(set(published_columns) - set(top64.tolist())),
            "largest_writers": [{"index": int(index), "score": float(scores[index]),
                                 "positive_pairs": int(consistency[index])} for index in np.argsort(scores)[-8:][::-1]],
        }
        snapshots[f"measured_direction_{layer_index}"] = direction
        snapshots[f"measured_writer_scores_{layer_index}"] = scores
    deltas = {}
    for kind in ("productive", "repetitive"):
        before = [row["mean_last36_token_logprob"] for row in all_rows if row["arm"] == "original" and row["kind"] == kind]
        after = [row["mean_last36_token_logprob"] for row in all_rows if row["arm"] == "edited" and row["kind"] == kind]
        deltas[kind] = {"paired_changes": (np.array(after) - np.array(before)).tolist(),
                        "mean_change_nats_per_token": float(np.mean(np.array(after) - np.array(before)))}
        if include_control:
            control = [row["mean_last36_token_logprob"] for row in all_rows if row["arm"] == "orthogonal_control" and row["kind"] == kind]
            deltas[kind]["orthogonal_control_mean_change"] = float(np.mean(np.array(control) - np.array(before)))
    np.savez_compressed(output / "activations.npz", **snapshots)
    report = {"published_request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
              "token_lengths_match_published": True, "lengths": lengths,
              "direction_window": DIRECTION_WINDOW, "writer_selection_window": WRITER_WINDOW,
              "model": info, "directions_and_writers": measurements, "likelihood_changes": deltas,
              "orthogonal_control": control_metadata,
              "per_text": all_rows, "elapsed_s": time.monotonic() - start,
              "qualification": "New CPU teacher-forced measurement on the disclosed matched contexts; old activation files were not released, so exact original activation equality is unverified."}
    verify_original(root)
    with (output / "result.json").open("x") as destination:
        json.dump(report, destination, indent=2, default=str)
    print("TEACHER_PROBE_COMPLETE " + json.dumps({"directions": measurements, "likelihood_changes": deltas}), flush=True)


class TeacherTests(unittest.TestCase):
    def test_matched_direction(self):
        means = np.array([[1.0, 1], [2.0, 1], [3.0, 2], [4.0, 2]])
        np.testing.assert_array_equal(paired_direction(means), [1, 0])

    def test_writer_requires_activation_and_direction(self):
        means = np.array([[1.0, 1], [3.0, 2], [1.0, 1], [3.0, 2]])
        scores, consistency = writer_scores(means, np.array([[2.0, -5], [0.0, 0]]), np.array([1.0, 0]))
        np.testing.assert_array_equal(scores, [4, -5])
        np.testing.assert_array_equal(consistency, [2, 0])

    def test_distinct_published_windows(self):
        self.assertEqual(DIRECTION_WINDOW, 36)
        self.assertEqual(WRITER_WINDOW, 24)
        values = np.concatenate((np.full((12, 1), 4.0), np.ones((24, 1))))
        self.assertEqual(float(values[-WRITER_WINDOW:].mean()), 1)
        self.assertEqual(float(values[-DIRECTION_WINDOW:].mean()), 2)

    def test_orthogonal_norm_matched_control(self):
        original = np.arange(20, dtype=np.float32).reshape(4, 5)
        edited = original.copy()
        edited[0, :2] += 3
        direction = np.array([1.0, 0, 0, 0])
        controlled, info = orthogonal_control(original, edited, direction, np.array([0, 1]), 17)
        self.assertAlmostEqual(info["absolute_direction_cosine"], 0)
        self.assertAlmostEqual(info["pre_round_perturbation_norm"], info["target_perturbation_norm"])
        np.testing.assert_array_equal(controlled[:, 2:], original[:, 2:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--name", default="qwen-teacher-probe")
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["teacher_probe.py"], verbosity=2)
    elif args.root:
        probe(args.root, args.name, args.control)
    else:
        parser.error("--root or --self-test required")