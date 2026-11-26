from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import unittest
from collections import Counter
from pathlib import Path

from policy import POLICIES, install_policy


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def previous_modules(root):
    sys.path.insert(0, str(root / "composite"))
    import evaluate

    return evaluate


def choose_candidate(rows, cases):
    questions = {case["id"] for case in cases if "gold" in case}
    formatting = {case["id"] for case in cases if "gold" not in case}
    expected = questions | formatting
    summaries = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy and row["input_mode"] == "question"]
        if len(selected) != len(expected) or {row["id"] for row in selected} != expected:
            raise ValueError("policy result coverage does not match frozen cases")
        summaries[policy] = {
            "questions_correct": sum(row["correct"] for row in selected if row["id"] in questions),
            "formatting_preserved": sum(row["text"].strip() == row["input"].strip() for row in selected if row["id"] in formatting),
            "question_echoes": sum(row["text"].strip() == row["input"].strip() for row in selected if row["id"] in questions),
            "all_echoes": sum(row["text"].strip() == row["input"].strip() for row in selected),
            "fallbacks": sum(sum(row["decoder_trace"]["fallbacks"]) for row in selected),
        }
    native_correct = sum(case["native_correct"] for case in cases if "gold" in case)
    candidates = [
        policy for policy in POLICIES if policy != "stock"
        and summaries[policy]["questions_correct"] >= native_correct
        and summaries[policy]["questions_correct"] > summaries["stock"]["questions_correct"]
        and summaries[policy]["formatting_preserved"] == len(formatting)
    ]
    return {
        "policies": summaries,
        "native_correct": native_correct,
        "questions": len(questions),
        "formatting_cases": len(formatting),
        "candidate": candidates[0] if candidates else None,
        "fresh_confirmation_required": bool(candidates),
        "decision": "requires_fresh_confirmation" if candidates else "no_qualifying_decoder_policy",
    }


def freeze(root):
    previous = previous_modules(root)
    prior_cases = previous.verify_frozen(root)
    experiment = root / "decoder_interface"
    frozen = experiment / "frozen"
    frozen.mkdir(exist_ok=False)
    cases = []
    sources = []
    for case in prior_cases["pipeline"]:
        record_path = root / "composite" / "results" / f"pipeline-{case['id']}.json"
        array_path = root / "composite" / "results" / "states" / f"pipeline-{case['id']}.npz"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        item = dict(case, previous_text=record["composite"], previous_trace=record["trace"])
        if "gold" in case:
            item.update(native_text=record["native_text"], native_correct=record["native_correct"])
        cases.append(item)
        sources.extend((record_path, array_path))
    questions = [case for case in cases if "gold" in case]
    for index, case in enumerate(questions):
        for offset in range(1, len(questions)):
            other = questions[(index + offset) % len(questions)]
            if previous.normalize_answer(other["native_text"]) != previous.normalize_answer(case["native_text"]):
                case["swapped_native_text"] = other["native_text"]
                case["swapped_source_id"] = other["id"]
                break
        else:
            raise ValueError("cannot construct a distinct swapped-answer control")
    write_json(frozen / "cases.json", cases)
    write_json(frozen / "PROTOCOL.json", {
        "policies": list(POLICIES),
        "input_modes": ["question", "native_answer", "swapped_native_answer"],
        "question_gate": "exceed stock and match or exceed native correct count; preserve all formatting controls",
        "candidate_tie_break": "first eligible policy in the frozen order",
        "raw_full_vocabulary": 128000,
        "weight_changes": False,
        "training": False,
        "native_answer_source": "recorded model outputs from the previous study, never gold text inserted as a plan",
        "evidence_scope": "diagnosis using previously exposed cases; no new held-out improvement claim",
    })
    sources += [experiment / name for name in ("policy.py", "study.py", "CONTEXT.md")]
    sources += [root / "composite" / name for name in ("model.py", "reference_c2c3.py", "evaluate.py", "prepare.py", "VERIFICATION.json", "frozen/MANIFEST.json", "package/MANIFEST.json", "package/TENSORS.json")]
    sources += [frozen / "cases.json", frozen / "PROTOCOL.json"]
    write_json(frozen / "MANIFEST.json", {
        "files": {str(path.relative_to(root)): file_hash(path) for path in sources if path.name != "CONTEXT.md"},
        "hypothesis_sha256": file_hash(experiment / "CONTEXT.md"),
        "checkpoint_sha256": file_hash(root / "composite" / "package" / "model.safetensors"),
    })
    write_json(frozen / "HYPOTHESIS.json", {"text": (experiment / "CONTEXT.md").read_bytes().decode("utf-8")})
    print(json.dumps({"frozen_policies": list(POLICIES), "question_and_format_cases": len(cases), "answer_input_controls": len(questions) * 2, "total_decoder_runs": len(POLICIES) * (len(cases) + len(questions) * 2)}), flush=True)


def verify_frozen(root):
    frozen = root / "decoder_interface" / "frozen"
    manifest = json.loads((frozen / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        if file_hash(root / relative) != expected:
            raise ValueError(f"frozen input changed: {relative}")
    if file_hash(root / "composite" / "package" / "model.safetensors") != manifest["checkpoint_sha256"]:
        raise ValueError("immutable composite checkpoint changed")
    hypothesis = json.loads((frozen / "HYPOTHESIS.json").read_text(encoding="utf-8"))["text"]
    if hashlib.sha256(hypothesis.encode("utf-8")).hexdigest() != manifest["hypothesis_sha256"]:
        raise ValueError("frozen hypothesis changed")
    return json.loads((frozen / "cases.json").read_text(encoding="utf-8"))


def restore_question_state(runtime, case, root, previous):
    import numpy as np

    state, _ = runtime.c1_infer(case["text"], model_name="previous real Omni inference states")
    path = root / "composite" / "results" / "states" / f"pipeline-{case['id']}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        state["h_primary"] = arrays["c1.h_primary"].copy()
        for index, window in enumerate(state["windows"]):
            if not window.get("passthrough"):
                window["h_valid"] = arrays[f"c1.windows.{index}.h_valid"].copy()
        planned, _ = runtime.c2_plan(case["text"], c1_state=state, jepa_arch="none")
        for name, value in previous.state_arrays(planned).items():
            if not np.array_equal(value, arrays["c2" + name]):
                raise ValueError(f"replayed plan states differ: {case['id']}: {name}")
    return planned


def run(root):
    experiment = root / "decoder_interface"
    cases = verify_frozen(root)
    previous = previous_modules(root)
    os.environ["HF_MODULES_CACHE"] = str(experiment / "cache" / "modules")
    os.environ["HF_HOME"] = str(experiment / "cache" / "huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "4"
    import numpy as np
    import torch
    from model import CompositeModel, load_c2c3

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    output = experiment / "results"
    output.mkdir(exist_ok=False)
    model = CompositeModel(root / "composite" / "package")
    base_runtime = model.c2c3
    question_states = {case["id"]: restore_question_state(base_runtime, case, root, previous) for case in cases}
    prefix = model.processor.tokenizer.encode(
        model.processor.tokenizer.bos_token + "<|im_start|>system\nReturn only the requested answer, without explanation.<|im_end|>\n<|im_start|>user\n",
        add_special_tokens=False,
    )
    interface_states = {}
    state_calls = 0
    for case in cases:
        if "gold" not in case:
            continue
        for name in ("native_text", "swapped_native_text"):
            text = case[name]
            if text not in interface_states:
                state, _ = base_runtime.c1_infer_states(text, model.states, prefix_ids=prefix)
                planned, _ = base_runtime.c2_plan(text, c1_state=state, jepa_arch="none")
                interface_states[text] = planned
                state_calls += 1
    baseline_latents = {}
    rows = []
    for policy in POLICIES:
        runtime = install_policy(load_c2c3(model.bundle), policy)
        directory = output / policy
        directory.mkdir()
        for case in cases:
            modes = ["question"] + (["native_answer", "swapped_native_answer"] if "gold" in case else [])
            for mode in modes:
                if mode == "question":
                    text = case["text"]
                    state = question_states[case["id"]]
                else:
                    text = case["native_text" if mode == "native_answer" else "swapped_native_text"]
                    state = interface_states[text]
                runtime._decoder_probe_trace.clear()
                started = time.perf_counter()
                answer, metadata, trace = previous.decode_observed(runtime, state)
                elapsed = time.perf_counter() - started
                key = (case["id"], mode)
                if policy == "stock":
                    baseline_latents[key] = trace.get("ncm_z_sha256")
                    if mode == "question" and (answer != case["previous_text"] or trace != case["previous_trace"]):
                        raise ValueError(f"stock control differs from prior result: {case['id']}")
                elif baseline_latents[key] != trace.get("ncm_z_sha256"):
                    raise ValueError("decoder policy changed upstream NCM states")
                record = {
                    "id": case["id"], "policy": policy, "input_mode": mode, "input": text,
                    "text": answer, "seconds": elapsed, "decoder_trace": trace,
                    "raw_trace": copy_trace(runtime._decoder_probe_trace),
                    "input_preserved": answer.strip() == text.strip(),
                    "upstream_ncm_unchanged": True,
                    "metadata": metadata,
                }
                if "gold" in case:
                    record.update(gold=case["gold"], correct=previous.score_answer(answer, case["gold"]))
                    stock_vocab = set(base_runtime._build_allowed_vocab_ids(text).tolist())
                    record["canonical_gold_missing_tokens"] = {
                        gold: sorted(set(base_runtime.tok.encode(gold, allow_special=False)) - stock_vocab)
                        for gold in case["gold"]
                    }
                    if mode == "swapped_native_answer":
                        record["swapped_source_id"] = case["swapped_source_id"]
                rows.append(record)
                write_json(directory / f"{case['id']}-{mode}.json", record)
        selected = [row for row in rows if row["policy"] == policy and row["input_mode"] == "question"]
        print(json.dumps({"policy_complete": policy, "question_correct": sum(row.get("correct", False) for row in selected), "echoes": sum(row["input_preserved"] for row in selected), "fallbacks": sum(sum(row["decoder_trace"]["fallbacks"]) for row in selected)}), flush=True)
        del runtime
        gc.collect()
    selection = choose_candidate(rows, cases)
    selection["interfaces"] = {
        policy: {
            mode: {
                "cases": len([row for row in rows if row["policy"] == policy and row["input_mode"] == mode]),
                "question_answers_correct": sum(row["correct"] for row in rows if row["policy"] == policy and row["input_mode"] == mode),
                "supplied_input_preserved": sum(row["input_preserved"] for row in rows if row["policy"] == policy and row["input_mode"] == mode),
            }
            for mode in ("native_answer", "swapped_native_answer")
        }
        for policy in POLICIES
    }
    selection.update(decoder_runs=len(rows), new_native_state_passes=state_calls, stock_reproduces_prior_exactly=True, upstream_ncm_unchanged=True, checkpoint_unchanged=True)
    verify_frozen(root)
    previous.verify_frozen(root)
    write_json(output / "SUMMARY.json", selection)
    print("DECODER_INTERFACE_COMPLETE " + json.dumps(selection), flush=True)


def copy_trace(value):
    return json.loads(json.dumps(value))


class GateTests(unittest.TestCase):
    def fixture(self):
        cases = [{"id": "question", "gold": ["answer"], "native_correct": True}, {"id": "format"}]
        rows = []
        for policy in POLICIES:
            for name in ("question", "format"):
                rows.append({"id": name, "policy": policy, "input_mode": "question", "input": name, "text": name, "correct": False, "decoder_trace": {"fallbacks": []}})
        return cases, rows

    def test_echoes_do_not_qualify(self):
        cases, rows = self.fixture()
        self.assertIsNone(choose_candidate(rows, cases)["candidate"])

    def test_correct_with_preserved_format_qualifies(self):
        cases, rows = self.fixture()
        row = next(row for row in rows if row["policy"] == "raw_full" and row["id"] == "question")
        row.update(text="answer", correct=True)
        self.assertEqual(choose_candidate(rows, cases)["candidate"], "raw_full")

    def test_format_regression_blocks_candidate(self):
        cases, rows = self.fixture()
        for row in rows:
            if row["policy"] == "raw_full":
                row.update(text="changed", correct=True)
        self.assertIsNone(choose_candidate(rows, cases)["candidate"])

    def test_missing_trial_rejected(self):
        cases, rows = self.fixture()
        with self.assertRaisesRegex(ValueError, "coverage"):
            choose_candidate(rows[:-1], cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
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