from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

from policy import POLICIES
from study import choose_candidate, file_hash, previous_modules, verify_frozen, write_json


def check_row(row, case, score):
    mode = row["input_mode"]
    policy = row["policy"]
    if policy not in POLICIES or mode not in ("question", "native_answer", "swapped_native_answer"):
        raise ValueError("unregistered policy or input mode")
    expected_text = case["text"] if mode == "question" else case["native_text" if mode == "native_answer" else "swapped_native_text"]
    if row["id"] != case["id"] or row["input"] != expected_text:
        raise ValueError("case/input identity mismatch")
    if row["input_preserved"] != (row["text"].strip() == expected_text.strip()):
        raise ValueError("incorrect input-preservation label")
    if not row["upstream_ncm_unchanged"]:
        raise ValueError("upstream NCM mismatch")
    if "gold" in case:
        if row["gold"] != case["gold"] or row["correct"] != score(row["text"], case["gold"]):
            raise ValueError("gold or answer score mismatch")
    if mode == "swapped_native_answer" and row["swapped_source_id"] != case["swapped_source_id"]:
        raise ValueError("wrong swapped-answer source")
    if policy == "stock" and mode == "question":
        if row["text"] != case["previous_text"] or row["decoder_trace"] != case["previous_trace"]:
            raise ValueError("stock differs from prior frozen evidence")
    if policy.startswith("raw_"):
        if any(row["decoder_trace"]["fallbacks"]):
            raise ValueError("raw decoder used a fallback")
        for trace in row["raw_trace"]:
            if trace["policy"] != policy or not 0 < trace["allowed_ids"] <= 128000:
                raise ValueError("raw policy trace mismatch")
            if policy == "raw_full" and trace["allowed_ids"] != 128000:
                raise ValueError("full head was not used")
            if any(not 0 <= token < 128000 for token in trace["token_ids"]):
                raise ValueError("invalid raw token id")


def audit(root):
    experiment = root / "decoder_interface"
    cases = verify_frozen(root)
    previous = previous_modules(root)
    from prepare import verify_inputs
    from reference_c2c3 import Lfm3PureBpeTokenizer

    previous.verify_frozen(root)
    original_checks = verify_inputs(root / "composite", check_sources=True)
    prior = json.loads((root / "composite" / "VERIFICATION.json").read_text(encoding="utf-8"))
    for relative, expected in prior["artifact_hashes"].items():
        if file_hash(root / "composite" / relative) != expected:
            raise ValueError(f"prior evidence changed: {relative}")
    package = root / "composite" / "package"
    for path in (package / "omni").rglob("*"):
        if path.is_file():
            source = root / "composite" / "inputs" / "c1" / path.relative_to(package / "omni")
            if not source.is_file() or file_hash(path) != file_hash(source):
                raise ValueError("packaged native metadata differs from its original copy")
    tokenizer_path = package / "c2_tokenizer.json"
    tokenizer_source = root / "composite" / "inputs" / "reference" / "data" / "omni" / "precomputed" / "trunks" / "tokenizer.json"
    if file_hash(tokenizer_path) != file_hash(tokenizer_source):
        raise ValueError("packaged C2 tokenizer changed")
    tokenizer = Lfm3PureBpeTokenizer(tokenizer_path)
    rows = []
    expected_files = set()
    for case in cases:
        modes = ["question"] + (["native_answer", "swapped_native_answer"] if "gold" in case else [])
        for policy in POLICIES:
            for mode in modes:
                path = experiment / "results" / policy / f"{case['id']}-{mode}.json"
                expected_files.add(path)
                row = json.loads(path.read_text(encoding="utf-8"))
                check_row(row, case, previous.score_answer)
                if policy.startswith("raw_") and row["raw_trace"] and len(row["raw_trace"]) == 1 and case["id"].startswith("question-"):
                    if tokenizer.decode(row["raw_trace"][0]["token_ids"]).strip() != row["text"]:
                        raise ValueError("raw recorded text does not match sampled token ids")
                rows.append(row)
    actual_files = {path for policy in POLICIES for path in (experiment / "results" / policy).glob("*.json")}
    if actual_files != expected_files:
        raise ValueError("result grid has unexpected or missing records")
    for case in cases:
        for mode in ("question", "native_answer", "swapped_native_answer"):
            selected = [row for row in rows if row["id"] == case["id"] and row["input_mode"] == mode]
            if len({row["decoder_trace"].get("ncm_z_sha256") for row in selected}) > 1:
                raise ValueError("decoder policies changed upstream NCM state")
    summary = json.loads((experiment / "results" / "SUMMARY.json").read_text(encoding="utf-8"))
    recomputed = choose_candidate(rows, cases)
    for key, value in recomputed.items():
        if summary[key] != value:
            raise ValueError(f"summary differs from raw evidence: {key}")
    for policy in POLICIES:
        for mode in ("native_answer", "swapped_native_answer"):
            selected = [row for row in rows if row["policy"] == policy and row["input_mode"] == mode]
            actual = {
                "cases": len(selected),
                "question_answers_correct": sum(previous.score_answer(row["text"], row["gold"]) for row in selected),
                "supplied_input_preserved": sum(row["text"].strip() == row["input"].strip() for row in selected),
            }
            if summary["interfaces"][policy][mode] != actual:
                raise ValueError("interface summary differs from raw outputs")
    if summary["decoder_runs"] != len(rows):
        raise ValueError("incorrect total decoder run count")
    files = [path for path in sorted((experiment / "results").rglob("*")) if path.is_file()]
    return {
        "verified": True,
        "decoder_runs": len(rows),
        "prior_source_and_copy_files_unchanged": len(original_checks["files"]),
        "prior_result_files_unchanged": len(prior["artifact_hashes"]),
        "raw_token_ids_match_text": True,
        "policies": summary["policies"],
        "interfaces": summary["interfaces"],
        "candidate": summary["candidate"],
        "decision": summary["decision"],
        "artifact_hashes": {str(path.relative_to(experiment)): file_hash(path) for path in files},
    }


class AuditTests(unittest.TestCase):
    def fixture(self):
        case = {"id": "test", "text": "question", "gold": ["answer"], "native_text": "answer", "swapped_native_text": "wrong", "swapped_source_id": "other"}
        row = {"id": "test", "policy": "raw_full", "input_mode": "question", "input": "question", "text": "answer", "input_preserved": False, "upstream_ncm_unchanged": True, "correct": True, "gold": ["answer"], "decoder_trace": {"fallbacks": [False]}, "raw_trace": [{"policy": "raw_full", "allowed_ids": 128000, "token_ids": [3]}]}
        return row, case

    def score(self, text, gold):
        return text in gold

    def test_valid_record(self):
        row, case = self.fixture()
        check_row(row, case, self.score)

    def test_score_corruption_rejected(self):
        row, case = self.fixture()
        row["correct"] = False
        with self.assertRaisesRegex(ValueError, "score"):
            check_row(row, case, self.score)

    def test_fake_full_head_rejected(self):
        row, case = self.fixture()
        row["raw_trace"][0]["allowed_ids"] = 17
        with self.assertRaisesRegex(ValueError, "full head"):
            check_row(row, case, self.score)

    def test_input_swap_rejected(self):
        row, case = self.fixture()
        row["input"] = "answer"
        with self.assertRaisesRegex(ValueError, "identity"):
            check_row(row, case, self.score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root:
        root = args.root.resolve(strict=True)
        result = audit(root)
        write_json(root / "decoder_interface" / "VERIFICATION.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "artifact_hashes"}), flush=True)
    else:
        parser.error("choose --self-test or --root")


if __name__ == "__main__":
    main()