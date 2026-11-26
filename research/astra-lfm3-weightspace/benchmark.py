import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time
import unittest

from runtime import GENERATION_SETTINGS, OmniRuntime


FROZEN_SCRIPTS = ["runtime.py", "fixtures.py", "benchmark.py", "PROTOCOL.md"]


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def package_versions():
    return {distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()}


def normalize_words(text):
    return " ".join(re.sub(r"[^a-z0-9\s]", "", text.lower()).split())


def verify_materials(root):
    manifest = json.loads((root / "fixtures" / "MATERIALS.json").read_text())
    for entry in manifest["files"]:
        if sha256(root / "fixtures" / entry["file"]) != entry["sha256"]:
            raise RuntimeError(f"Frozen fixture changed: {entry['file']}")
    return manifest


def seal(root):
    root = Path(root)
    material = verify_materials(root)
    destination = root / "FROZEN.json"
    if destination.exists():
        raise FileExistsError(destination)
    record = {
        "materials_sha256": sha256(root / "fixtures" / "MATERIALS.json"),
        "case_count": material["case_count"], "cases_sha256": material["cases_sha256"],
        "scripts": {name: sha256(root / name) for name in FROZEN_SCRIPTS},
        "packages": package_versions(), "generation_settings": GENERATION_SETTINGS,
        "checkpoint_seal_sha256": sha256(root / "SEALED.json"),
    }
    with destination.open("x") as output:
        json.dump(record, output, indent=2)
    print("BENCHMARK_SEALED " + sha256(destination), flush=True)


def verify_seal(root):
    root = Path(root)
    record = json.loads((root / "FROZEN.json").read_text())
    verify_materials(root)
    if record["materials_sha256"] != sha256(root / "fixtures" / "MATERIALS.json"):
        raise RuntimeError("Materials manifest changed")
    if record["checkpoint_seal_sha256"] != sha256(root / "SEALED.json"):
        raise RuntimeError("Checkpoint identity changed")
    for name, digest in record["scripts"].items():
        if sha256(root / name) != digest:
            raise RuntimeError(f"Frozen inference/scoring source changed: {name}")
    if package_versions() != record["packages"]:
        raise RuntimeError("Frozen dependency set changed")
    if record["generation_settings"] != GENERATION_SETTINGS:
        raise RuntimeError("Frozen generation settings changed")
    return record


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['split']}:{row['modality']}"].append(row)
    result = {}
    for group, entries in sorted(groups.items()):
        scored = [entry for entry in entries if "correct" in entry]
        summary = {
            "cases": len(entries), "scored_cases": len(scored),
            "correct": sum(bool(entry["correct"]) for entry in scored),
            "errors": sum("error" in entry for entry in entries),
            "limit_hits": sum(entry.get("hit_token_limit", False) for entry in entries),
            "generation_s": sum(entry.get("generation_s", 0) for entry in entries),
        }
        speech = [entry["speech"] for entry in entries if entry.get("speech")]
        if any(entry["modality"] == "speech" for entry in entries):
            summary["speech_waveforms_nonempty"] = sum(entry["samples"] > 0 and entry["rms"] > 0 for entry in speech)
            summary["speech_intelligibility"] = "not independently evaluated"
        asr = [entry["asr"] for entry in entries if entry.get("asr")]
        if asr:
            reference_words = sum(entry["reference_words"] for entry in asr)
            errors = sum(entry["word_errors"] for entry in asr)
            summary["corpus_word_error_rate"] = errors / reference_words if reference_words else None
            summary["asr_exact"] = sum(entry["exact"] for entry in asr)
        result[group] = summary
    return result


def run(root, name, checkpoint=None, pilot=False):
    import numpy as np
    import jiwer
    from survey import verify_original

    root = Path(root)
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be one directory name")
    original = verify_original(root)
    if not pilot:
        verify_seal(root)
    else:
        verify_materials(root)
    case_file = root / "fixtures" / ("pilot-cases.json" if pilot else "cases.json")
    cases = json.loads(case_file.read_text())
    output_dir = root / "reports" / name
    output_dir.mkdir(exist_ok=False)
    runtime = OmniRuntime(root, checkpoint=checkpoint)
    census = runtime.census()
    metadata = {
        "name": name, "pilot": pilot, "checkpoint": str(runtime.checkpoint),
        "original_revision": original["revision"], "cases_sha256": sha256(case_file),
        "frozen_sha256": sha256(root / "FROZEN.json") if not pilot else None,
        "scripts": {name: sha256(root / name) for name in FROZEN_SCRIPTS},
        "runtime": census,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    rows, arrays = [], {}
    start_time = time.monotonic()
    with (output_dir / "results.jsonl").open("x") as output:
        for case in cases:
            try:
                row, activations = runtime.run_case(case, output_dir)
                if case["modality"] == "audio_asr" and "gold_text" in case:
                    reference, hypothesis = normalize_words(case["gold_text"]), normalize_words(row["text"])
                    measures = jiwer.process_words(reference, hypothesis)
                    row["asr"] = {
                        "reference": reference, "hypothesis": hypothesis,
                        "word_errors": measures.substitutions + measures.deletions + measures.insertions,
                        "reference_words": len(reference.split()), "wer": measures.wer,
                        "exact": reference == hypothesis,
                    }
                for key, value in activations.items():
                    arrays[f"{case['id']}::{key}"] = value
            except Exception as error:
                row = {"id": case["id"], "modality": case["modality"],
                       "split": case.get("split", "pilot"),
                       "error": f"{type(error).__name__}: {error}"}
                if "gold_letter" in case or "gold_text" in case:
                    row["correct"] = False
            rows.append(row)
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
            print(json.dumps({"completed": len(rows), "total": len(cases),
                              "id": row["id"], "correct": row.get("correct"),
                              "error": row.get("error"),
                              "elapsed_s": round(time.monotonic() - start_time, 2)}), flush=True)
    np.savez_compressed(output_dir / "activations.npz", **arrays)
    verify_original(root)
    if not pilot:
        verify_seal(root)
    summary = {
        "name": name, "case_count": len(rows), "groups": aggregate(rows),
        "elapsed_s": time.monotonic() - start_time,
        "source_hashes_verified_before_and_after": True,
        "error_count": sum("error" in row for row in rows),
        "results_sha256": sha256(output_dir / "results.jsonl"),
        "activations_sha256": sha256(output_dir / "activations.npz"),
        "limitations": ["Constructed small sample, not general benchmark accuracy",
                        "Speech intelligibility/naturalness not independently evaluated"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("BENCHMARK_COMPLETE " + json.dumps(summary), flush=True)
    if summary["error_count"]:
        raise RuntimeError("Benchmark has instrument/runtime errors; do not treat it as complete capability evidence")


class BenchmarkTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_words("  HELLO,   world!  "), "hello world")
        self.assertEqual(normalize_words("DON'T"), "dont")

    def test_aggregate_denominators(self):
        rows = [
            {"split": "validation", "modality": "text", "correct": True},
            {"split": "validation", "modality": "text", "correct": False, "error": "failure"},
            {"split": "regression", "modality": "speech", "speech": {"samples": 3, "rms": 0.1}},
        ]
        groups = aggregate(rows)
        self.assertEqual(groups["validation:text"]["correct"], 1)
        self.assertEqual(groups["validation:text"]["scored_cases"], 2)
        self.assertEqual(groups["validation:text"]["errors"], 1)
        self.assertEqual(groups["regression:speech"]["scored_cases"], 0)
        self.assertEqual(groups["regression:speech"]["speech_waveforms_nonempty"], 1)

    def test_wer_is_word_weighted(self):
        rows = [{"split": "regression", "modality": "audio_asr", "asr": {
            "reference_words": count, "word_errors": errors, "exact": not errors,
        }} for count, errors in ((10, 1), (2, 1))]
        summary = aggregate(rows)["regression:audio_asr"]
        self.assertAlmostEqual(summary["corpus_word_error_rate"], 2 / 12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--checkpoint")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["benchmark.py"], verbosity=2)
    elif not args.root:
        parser.error("--root required")
    elif args.seal:
        seal(args.root)
    else:
        run(args.root, args.name, args.checkpoint, args.pilot)