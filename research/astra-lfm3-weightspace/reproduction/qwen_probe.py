import argparse
import ast
import collections
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import types
import unittest


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def published_grader(root):
    path = Path(root) / "lm_understand-logs" / "exp8_math" / "code" / "mathqa.py"
    functions = {"build_prompt", "answer_text", "_box_at", "last_boxed", "extract",
                 "mv", "_timeouts", "_parsed", "is_correct"}
    constants = {"INSTRUCTION", "_BOX_FORMS", "_MV"}
    source = ast.parse(path.read_text())
    body = []
    found = set()
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets):
            body.append(node)
    if found != functions:
        raise ValueError("Published grading functions are missing")
    namespace = {"re": re, "sys": sys, "threading": threading, "functools": functools,
                 "MATHVERIFY_PKGS": str(Path(root) / "nonexistent-mathverify-fallback")}
    tree = ast.Module(body=body, type_ignores=[])
    exec(compile(tree, str(path), "exec"), namespace)
    if importlib.metadata.version("math-verify") != "0.9.0":
        raise ValueError("The paper's math-verify version is required")
    return types.SimpleNamespace(**{name: namespace[name] for name in functions},
                                 source_sha256=sha256(path),
                                 selected_ast_sha256=hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest())


def subtract_presence(scores, token_ids, prompt_length, penalty):
    for row in range(token_ids.shape[0]):
        generated = token_ids[row, prompt_length:].unique()
        scores[row, generated] -= penalty
    return scores


def select_cases(root, count):
    source = Path(root) / "lm_understand-logs" / "exp8_math"
    request = json.loads((source / "runs" / "p50c" / "requests" / "fwd_1.json.claimed").read_text())
    selected_ids = request["problems"][:count]
    selected = {}
    for name in ("gsm8k_train.jsonl", "math_train.jsonl"):
        with (source / "practice" / name).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["id"] in selected_ids:
                    selected[row["id"]] = row
    if set(selected_ids) != set(selected):
        raise ValueError("Published diagnostic problem coverage differs")
    return [selected[identifier] for identifier in selected_ids]


def verify_original(root):
    root = Path(root)
    seal = json.loads((root / "models" / "QWEN_SOURCE.json").read_text())
    for entry in seal["files"]:
        if sha256(root / "models" / "qwen-original" / entry["file"]) != entry["sha256"]:
            raise ValueError(f"Original Qwen changed: {entry['file']}")
    return seal


def load_model(root, arm):
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    original = Path(root) / "models" / "qwen-original"
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        original, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise ValueError(f"Checkpoint loading issue {key}: {loading[key]}")
    model.eval().requires_grad_(False)
    manifest = json.loads((Path(root) / "lm_understand-logs" / "exp8_math" / "runs" / "p50c" / "edit" / "manifest.json").read_text())
    if arm == "edited":
        with safe_open(Path(root) / "models" / "qwen-p50c.safetensors", framework="pt", device="cpu") as edited:
            for entry in manifest["tensors"]:
                parameter = model.get_parameter(entry["name"])
                before = parameter.detach().contiguous().view(torch.uint8).numpy().tobytes()
                if hashlib.sha256(before).hexdigest() != entry["base_sha256"]:
                    raise ValueError("Loaded parameter differs from the published base")
                replacement = edited.get_tensor(entry["name"])
                if hashlib.sha256(replacement.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() != entry["edited_sha256"]:
                    raise ValueError("Reconstructed edit digest differs")
                parameter.data = replacement.clone()
    tokenizer = AutoTokenizer.from_pretrained(original, local_files_only=True)
    template_path = Path(root) / "lm_understand-logs" / "exp8_math" / "base_model" / "Qwen3.5-0.8B-text" / "chat_template.jinja"
    tokenizer.chat_template = template_path.read_text()
    return model, tokenizer, {"loading": loading, "chat_template_sha256": sha256(template_path),
                              "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                              "dtype": str(next(model.parameters()).dtype),
                              "class": type(model).__name__}


def repetition_signals(text):
    words = text.split()
    tail = words[-200:]
    phrases = [tuple(tail[index:index + 4]) for index in range(max(0, len(tail) - 3))]
    counts = collections.Counter(phrases)
    fraction = sum(counts[phrase] >= 3 for phrase in phrases) / len(phrases) if phrases else 0.0
    return {"tail_repeated_4gram_fraction": fraction,
            "repetitive_end": len(phrases) > 0 and fraction >= 0.5,
            "qualification": "Independent descriptive implementation, not the unreleased mcqa helper."}


def run(root, arm, count, max_tokens, name):
    root = Path(root).resolve()
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    verify_original(root)
    output_dir = root / "results" / name
    output_dir.mkdir(exist_ok=False)
    grader = published_grader(root)
    cases = select_cases(root, count)
    model, tokenizer, model_info = load_model(root, arm)
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    protocol = {"arm": arm, "selected_cases": cases, "max_new_tokens": max_tokens,
                "thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "presence_penalty": 1.5, "seed": 0, "batch_size": 1,
                "backend": "Transformers CPU SDPA; not the published GPU/vLLM backend",
                "grader_source_sha256": grader.source_sha256,
                "grader_selected_ast_sha256": grader.selected_ast_sha256,
                "grading": "Unchanged selected prompt/extraction/equivalence functions from the released mathqa.py, math-verify 0.9.0",
                "model": model_info, "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "qualification": "Short runs are timing/debug probes, not paper benchmark scores; full-cap runs are a small independent paired diagnostic."}
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, default=str))
    rows = []
    for case in cases:
        torch.manual_seed(0)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": grader.build_prompt(case["problem"])}],
            add_generation_prompt=True, tokenize=False, enable_thinking=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_length = inputs["input_ids"].shape[1]
        start = time.monotonic()

        class PresencePenalty(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return subtract_presence(scores, input_ids, prompt_length, 1.5)

        class Progress(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                generated = input_ids.shape[1] - prompt_length
                if generated % 512 == 0:
                    print(json.dumps({"case": case["id"], "arm": arm, "tokens": generated,
                                      "elapsed_s": time.monotonic() - start}), flush=True)
                return False

        generated = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.6,
            top_p=0.95, top_k=20, repetition_penalty=1.0,
            logits_processor=LogitsProcessorList([PresencePenalty()]),
            stopping_criteria=StoppingCriteriaList([Progress()]),
            return_dict_in_generate=True,
        )
        token_ids = generated.sequences[0, prompt_length:].tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        pred, boxes = grader.extract(text)
        answer = case.get("answer", case.get("gold"))
        if answer is None:
            raise ValueError("Published practice row lacks its reference answer")
        elapsed = time.monotonic() - start
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        stopped = bool(token_ids and token_ids[-1] in eos_ids)
        row = {"id": case["id"], "arm": arm, "input_sha256": hashlib.sha256(inputs["input_ids"].numpy().tobytes()).hexdigest(),
               "prompt_tokens": prompt_length, "generated_tokens": len(token_ids),
               "elapsed_s": elapsed, "tokens_per_second": len(token_ids) / elapsed,
               "hit_cap": len(token_ids) == max_tokens and not stopped,
               "stopped_on_eos": stopped, "thinking_closed": "</think>" in text,
               "prediction": pred, "gold": answer, "boxes": boxes,
               "correct": grader.is_correct(pred, str(answer)), "repetition": repetition_signals(text),
               "text": text, "tokens": token_ids}
        rows.append(row)
        with (output_dir / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
        print("CASE_COMPLETE " + json.dumps({key: row[key] for key in ("id", "arm", "generated_tokens", "elapsed_s", "tokens_per_second", "hit_cap", "thinking_closed", "correct")}), flush=True)
        for parameter_name, parameter in model.named_parameters():
            if parameter._version != versions[parameter_name]:
                raise ValueError(f"Inference changed a parameter: {parameter_name}")
    verify_original(root)
    summary = {"arm": arm, "cases": len(rows), "correct": sum(row["correct"] for row in rows),
               "cut": sum(row["hit_cap"] for row in rows), "thinking_closed": sum(row["thinking_closed"] for row in rows),
               "elapsed_s": sum(row["elapsed_s"] for row in rows),
               "result_sha256": sha256(output_dir / "results.jsonl"),
               "original_hashes_verified_after": True}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("QWEN_PROBE_COMPLETE " + json.dumps(summary), flush=True)


def self_test(root):
    import torch

    class ProbeTests(unittest.TestCase):
        def test_presence_only_generated_and_once(self):
            ids = torch.tensor([[2, 3, 5, 5, 5, 7]])
            scores = torch.zeros((1, 10))
            adjusted = subtract_presence(scores, ids, 3, 1.5)
            self.assertEqual(float(adjusted[0, 2]), 0)
            self.assertEqual(float(adjusted[0, 3]), 0)
            self.assertEqual(float(adjusted[0, 5]), -1.5)
            self.assertEqual(float(adjusted[0, 7]), -1.5)

        def test_published_grader_positive_and_negative(self):
            grader = published_grader(root)
            self.assertEqual(grader.extract(r"<think>\boxed{1}</think>Final: \boxed{\frac{1}{2}}"), (r"\frac{1}{2}", 1))
            self.assertTrue(grader.is_correct(r"\frac{1}{2}", "0.5"))
            self.assertFalse(grader.is_correct("3", "4"))
            self.assertEqual(grader.extract(r"\boxed{7"), (None, 1))
            self.assertEqual(grader.extract("The answer is 7"), (None, 0))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProbeTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--arm", choices=("original", "edited"), default="original")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--name", default="qwen-timing")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.root)
    else:
        run(args.root, args.arm, args.count, args.max_tokens, args.name)