import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import unittest

from qwen_probe import published_grader


def compare_grade(recorded, computed):
    if type(recorded) not in (bool, int) or recorded not in (0, 1):
        raise ValueError("Recorded correctness must be a binary first-sample label")
    return bool(recorded) == bool(computed)


def regrade(root):
    root = Path(root)
    grader = published_grader(root)
    source = root / "lm_understand-logs" / "exp8_math" / "eval"
    output = root / "results" / "recorded-math-regrade"
    output.mkdir(exist_ok=False)
    summary = {"grader_source_sha256": grader.source_sha256,
               "selected_ast_sha256": grader.selected_ast_sha256,
               "scope": "Independent re-grading of released first-sample texts, not new generations or all four samples per MATH-500 item.",
               "arms": {}}
    started = time.monotonic()
    for arm in ("unedited", "p50c"):
        traces = sorted((source / arm / "parts").glob("math500_*/math500_first_samples.jsonl.gz"))
        if not traces:
            raise ValueError("No released MATH-500 traces")
        seen = set()
        rows = []
        inputs = []
        with (output / f"{arm}.jsonl").open("x") as destination:
            for path in traces:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                inputs.append({"file": str(path.relative_to(source)), "sha256": digest})
                labels = {row["id"]: row for row in json.loads((path.parent / "math500_per_problem.json").read_text())}
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        sample = json.loads(line)
                        identifier = sample["id"]
                        if identifier in seen or identifier not in labels:
                            raise ValueError("Duplicate or unlabeled released sample")
                        seen.add(identifier)
                        pred, boxes = grader.extract(sample["text"])
                        good = grader.is_correct(pred, str(sample["answer"]))
                        expected = labels[identifier]["correct"][0]
                        result = {"id": identifier, "prediction": pred, "reference_answer": sample["answer"],
                                  "computed_correct": good, "recorded_correct": expected,
                                  "agrees": compare_grade(expected, good), "boxes": boxes,
                                  "text_characters": len(sample["text"])}
                        rows.append(result)
                        destination.write(json.dumps(result) + "\n")
                        destination.flush()
                        if len(rows) % 50 == 0:
                            print(json.dumps({"arm": arm, "regraded": len(rows),
                                              "mismatches": sum(not row["agrees"] for row in rows),
                                              "elapsed_s": time.monotonic() - started}), flush=True)
        summary["arms"][arm] = {
            "released_first_samples_regraded": len(rows), "unique_ids": len(seen),
            "computed_correct": sum(row["computed_correct"] for row in rows),
            "recorded_correct": sum(bool(row["recorded_correct"]) for row in rows),
            "mismatches": [row for row in rows if not row["agrees"]],
            "source_files": inputs,
        }
    summary["elapsed_s"] = time.monotonic() - started
    with (output / "result.json").open("x") as destination:
        json.dump(summary, destination, indent=2)
    print("RECORDED_TRACE_REGRADE_COMPLETE " + json.dumps({arm: {key: value for key, value in record.items() if key != "source_files"}
                                                           for arm, record in summary["arms"].items()}), flush=True)


class RegradeTests(unittest.TestCase):
    def test_mismatch_is_visible(self):
        self.assertTrue(compare_grade(1, True))
        self.assertTrue(compare_grade(0, False))
        self.assertFalse(compare_grade(1, False))
        self.assertFalse(compare_grade(0, True))

    def test_nonbinary_label_refused(self):
        with self.assertRaises(ValueError):
            compare_grade(2, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["regrade_traces.py"], verbosity=2)
    elif args.root:
        regrade(args.root)
    else:
        parser.error("--root or --self-test required")