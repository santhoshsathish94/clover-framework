import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import time
import unittest

import numpy as np

from replay_network import TENSORS, forward, observation


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def probabilities(logits):
    shifted = logits.astype(np.float64) - float(np.max(logits))
    weights = np.exp(shifted)
    return weights / weights.sum()


def validate_engine(source):
    from minatar.environments import breakout

    prompt_path = source / "runs" / "breakout" / "message.txt"
    prompt = prompt_path.read_text()
    block = re.findall(r"```python\s*\n(.*?)```", prompt, re.DOTALL)[0]
    installed_path = Path(inspect.getsourcefile(breakout))
    installed = installed_path.read_text()
    if ast.dump(ast.parse(block), include_attributes=False) != ast.dump(ast.parse(installed), include_attributes=False):
        raise ValueError("Installed game implementation differs from the source supplied to the author")
    if importlib.metadata.version("minatar") != "1.0.15":
        raise ValueError("MinAtar version differs from the published experiment")
    return {"version": "1.0.15", "source_ast_matches_published_prompt": True,
            "installed_source_sha256": sha256(installed_path), "prompt_sha256": sha256(prompt_path)}


def episode(arrays, seed, trace=False):
    from minatar import Environment

    environment = Environment("breakout", sticky_action_prob=0.1)
    environment.seed(seed)
    environment.reset()
    sampling = np.random.RandomState(1_000_000 + seed)
    total = 0
    frames, trace_rows = [], []
    for step in range(10_000):
        raw = environment.state()
        inputs = observation(raw)
        logits, hidden1, hidden2 = forward(arrays, inputs)
        action = int(sampling.choice(6, p=probabilities(logits)))
        if trace and step < 2000 and step % 20 == 0:
            frames.append(raw.copy())
            trace_rows.append({"step": step, "score": total, "action": action,
                               "paddle": environment.env.pos,
                               "ball": [environment.env.ball_y, environment.env.ball_x],
                               "direction": environment.env.ball_dir,
                               "hidden1_0_to_5": hidden1[:6].tolist(),
                               "hidden2_0_to_7": hidden2[:8].tolist(),
                               "logits": logits.tolist()})
        reward, terminal = environment.act(action)
        total += reward
        if terminal:
            break
    return {"seed": seed, "score": float(total), "steps": step + 1,
            "hit_cap": step + 1 == 10_000 and not terminal}, trace_rows, frames


def summarize(rows):
    scores = np.array([row["score"] for row in rows], dtype=np.float64)
    return {"episodes": len(rows), "mean": float(scores.mean()),
            "se": float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(rows) > 1 else None,
            "max": float(scores.max()), "min": float(scores.min()),
            "mean_length": float(np.mean([row["steps"] for row in rows])),
            "hit_cap": sum(row["hit_cap"] for row in rows)}


def save_animation(path, frames):
    from PIL import Image

    images = []
    for frame in frames:
        pixels = np.full((10, 10, 3), 245, dtype=np.uint8)
        for channel, color in ((3, (30, 150, 100)), (2, (245, 180, 70)), (0, (40, 80, 190)), (1, (210, 45, 55))):
            pixels[frame[:, :, channel]] = color
        images.append(Image.fromarray(pixels).resize((300, 300), Image.Resampling.NEAREST))
    if images:
        images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)


def evaluate(root, count, name):
    root = Path(root)
    source = root / "lm_understand-logs" / "exp5_write_from_scratch"
    engine = validate_engine(source)
    output = root / "results" / name
    output.mkdir(exist_ok=False)
    archive = np.load(root / "results" / "breakout-reconstruction" / "reconstructed.npz", allow_pickle=False)
    arrays = {name: archive[name] for name in TENSORS}
    reference_path = source / "networks" / "breakout" / "hand.npz"
    original_hash = sha256(reference_path)
    reference = np.load(reference_path, allow_pickle=False)
    for name, (reference_name, _) in TENSORS.items():
        if not np.array_equal(arrays[name], reference[reference_name]):
            raise ValueError("The network differs from the bit-exact reconstruction")
    published = json.loads((source / "eval" / "minatar_eval.json").read_text())["breakout"]
    protocol = {"engine": engine, "episodes": count, "max_steps": 10000,
                "action_selection": "sample all six softmax logits, float64 probabilities",
                "sticky_action_prob": 0.1, "environment_seeds": [0, count - 1],
                "action_rng": "NumPy RandomState(1000000 + episode_seed)",
                "reference_sha256": original_hash, "numpy": np.__version__,
                "published_score": published["hand"],
                "qualification": "Published seed schedule/evaluator not released; same disclosed settings, new seeded independent evaluator."}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    started = time.monotonic()
    rows = []
    with (output / "episodes.jsonl").open("x") as log:
        for seed in range(count):
            row, trace_rows, frames = episode(arrays, seed, trace=seed == 0)
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if seed == 0:
                (output / "trace-seed0.json").write_text(json.dumps(trace_rows, indent=2))
                save_animation(output / "trace-seed0.gif", frames)
            if (seed + 1) % 25 == 0:
                print(json.dumps({"episodes": seed + 1, "mean": float(np.mean([row["score"] for row in rows])),
                                  "elapsed_s": time.monotonic() - started}), flush=True)
    zero = {name: np.zeros_like(value) for name, value in arrays.items()}
    controls = [episode(zero, seed)[0] for seed in range(min(count, 100))]
    measured = summarize(rows)
    result = {"measured": measured, "zero_weight_control": summarize(controls),
              "published_hand": published["hand"],
              "published_ppo100_best_not_reproduced": published["ppo100_s2"],
              "published_ppo500_best_not_reproduced": published["ppo500_s0"],
              "elapsed_s": time.monotonic() - started,
              "reference_unchanged": sha256(reference_path) == original_hash,
              "engine": engine,
              "scope": "Independent behavioral evaluation of an exactly reconstructed published weight program, not replay of the original editor or PPO training."}
    if measured["se"] is not None:
        combined_se = np.hypot(measured["se"], published["hand"]["se"])
        result["difference_in_combined_standard_errors"] = (measured["mean"] - published["hand"]["mean"]) / combined_se
    if not result["reference_unchanged"]:
        raise ValueError("Published reference changed")
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("GAMEPLAY_COMPLETE " + json.dumps(result), flush=True)


class GameplayTests(unittest.TestCase):
    def test_uniform_control(self):
        np.testing.assert_allclose(probabilities(np.zeros(6)), np.full(6, 1 / 6))

    def test_softmax_translation_invariance(self):
        logits = np.array([0, 90, -100, 90, -100, -100], dtype=np.float32)
        np.testing.assert_allclose(probabilities(logits), probabilities(logits + 1000))
        self.assertAlmostEqual(float(probabilities(logits).sum()), 1)

    def test_statistics(self):
        result = summarize([{"score": score, "steps": 3, "hit_cap": False} for score in (0, 2)])
        self.assertEqual(result["mean"], 1)
        self.assertEqual(result["se"], 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--name", default="breakout-1000")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["evaluate_breakout.py"], verbosity=2)
    elif args.root:
        evaluate(args.root, args.episodes, args.name)
    else:
        parser.error("--root or --self-test required")