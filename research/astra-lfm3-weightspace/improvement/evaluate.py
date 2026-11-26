import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import OmniRuntime
from survey import verify_original
from diagnose import build_cases, numeric_answer, summarize


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def build_split(label, seed, batches, used_questions):
    result = []
    accepted_batches = 0
    attempt = 0
    while accepted_batches < batches:
        candidates = build_cases(seed + attempt * 1009)
        attempt += 1
        questions = {case["prompt"].splitlines()[0] for case in candidates if case["format"] == "mcq"}
        if questions & used_questions or len(questions) != 24:
            if attempt > 100:
                raise ValueError("Unable to construct disjoint deterministic cases")
            continue
        used_questions.update(questions)
        for case in candidates:
            original_group = case["group"]
            case["id"] = case["id"].replace("discovery", f"{label}-{accepted_batches}")
            case["group"] = accepted_batches * 24 + original_group
            case["split"] = label
            result.append(case)
        accepted_batches += 1
    return result


def prepare(root):
    root = Path(root)
    directory = root / "improvement" / "frozen"
    directory.mkdir(exist_ok=False)
    discovery = build_cases()
    used = {case["prompt"].splitlines()[0] for case in discovery if case["format"] == "mcq"}
    tuning = build_split("tuning", 183019, 2, used)
    final = build_split("final", 917431, 4, used)
    manifest = {"files": {}, "sources": {}, "counts": {"tuning": len(tuning), "final": len(final)}}
    for name, cases in (("tuning", tuning), ("final", final)):
        path = directory / f"{name}.json"
        with path.open("x") as output:
            json.dump(cases, output, indent=2)
        manifest["files"][path.name] = sha256(path)
    for relative in ("runtime.py", "improvement/diagnose.py", "improvement/evaluate.py", "improvement/PROTOCOL.md"):
        manifest["sources"][relative] = sha256(root / relative)
    with (directory / "MANIFEST.json").open("x") as output:
        json.dump(manifest, output, indent=2)
    print("IMPROVEMENT_SUITE_FROZEN " + json.dumps(manifest["counts"]), flush=True)


def verify_frozen(root):
    manifest = json.loads((root / "improvement" / "frozen" / "MANIFEST.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(root / "improvement" / "frozen" / name) != digest:
            raise ValueError(f"Frozen test cases changed: {name}")
    for name, digest in manifest["sources"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Frozen runtime/scorer/protocol changed: {name}")
    return manifest


def run(root, split, name, checkpoint=None):
    root = Path(root)
    verify_original(root)
    manifest = verify_frozen(root)
    cases = json.loads((root / "improvement" / "frozen" / f"{split}.json").read_text())
    destination = root / "improvement" / "runs" / name
    destination.mkdir(parents=True, exist_ok=False)
    runtime = OmniRuntime(root, checkpoint=checkpoint)
    for hook in runtime.handles:
        hook.remove()
    runtime.handles = []
    started = time.monotonic()
    rows = []
    with (destination / "results.jsonl").open("x") as output:
        for case in cases:
            result, _ = runtime.run_case(case, destination)
            if case["format"] == "free":
                result["predicted_value"] = numeric_answer(result["text"])
                result["correct"] = result["predicted_value"] == case["gold_value"]
            else:
                label = result["prediction"]
                result["predicted_value"] = case["choices"]["ABCD".index(label)] if label else None
            result.update({"group": case["group"], "family": case["family"], "format": case["format"], "gold_value": case["gold_value"]})
            rows.append(result)
            output.write(json.dumps(result) + "\n")
            output.flush()
            if len(rows) % 50 == 0:
                print(json.dumps({"name": name, "completed": len(rows), "total": len(cases)}), flush=True)
    summary = summarize(cases, rows)
    totals = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        kind = "selection" if row["family"] == "selection" else "arithmetic_mcq" if row["format"] == "mcq" else "arithmetic_free"
        totals[kind]["correct"] += bool(row["correct"])
        totals[kind]["total"] += 1
    summary.update({"name": name, "split": split, "checkpoint": str(runtime.checkpoint),
                    "totals": dict(totals), "case_count": len(rows),
                    "case_sha256": manifest["files"][f"{split}.json"],
                    "results_sha256": sha256(destination / "results.jsonl"),
                    "elapsed_s": time.monotonic() - started})
    verify_original(root)
    verify_frozen(root)
    summary["original_and_frozen_inputs_unchanged"] = True
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    print("IMPROVEMENT_EVAL_COMPLETE " + json.dumps({"name": name, "totals": summary["totals"], "elapsed_s": summary["elapsed_s"]}), flush=True)


class SuiteTests(unittest.TestCase):
    def test_splits_are_disjoint_and_balanced(self):
        discovery = build_cases()
        used = {case["prompt"].splitlines()[0] for case in discovery if case["format"] == "mcq"}
        tuning = build_split("tuning", 183019, 2, used)
        final = build_split("final", 917431, 4, used)
        self.assertEqual(len(tuning), 228)
        self.assertEqual(len(final), 456)
        self.assertEqual(len({case["id"] for case in tuning + final}), 684)
        for cases in (tuning, final):
            for group in {case["group"] for case in cases}:
                labels = [case["gold_letter"] for case in cases if case["group"] == group and case["format"] == "mcq"]
                self.assertEqual(sorted(labels), list("ABCD"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--split", choices=("tuning", "final"), default="tuning")
    parser.add_argument("--name", default="original-tuning")
    parser.add_argument("--checkpoint")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["evaluate.py"], verbosity=2)
    elif not args.root:
        parser.error("--root required")
    elif args.prepare:
        prepare(args.root)
    else:
        run(args.root, args.split, args.name, args.checkpoint)