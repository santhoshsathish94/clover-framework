import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import unittest
import urllib.request


REPO = "openai/whisper-tiny.en"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root):
    root = Path(root)
    directory = root / "improvement" / "speech-observer"
    directory.mkdir(exist_ok=False)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}?blobs=true", timeout=60) as response:
        info = json.load(response)
    if info["id"] != REPO or info.get("private"):
        raise ValueError("Unexpected speech-observer identity/access boundary")
    selected_names = {"config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json",
                      "tokenizer.json", "tokenizer_config.json", "normalizer.json", "special_tokens_map.json",
                      "vocab.json", "merges.txt", "LICENSE", "README.md"}
    selected = [entry for entry in info["siblings"] if entry["rfilename"] in selected_names]
    os.environ["HF_HOME"] = str(root / "improvement" / "observer-cache")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, revision=info["sha"], local_dir=directory / "model", token=False,
                      allow_patterns=[entry["rfilename"] for entry in selected], max_workers=2)
    files = []
    for entry in selected:
        path = directory / "model" / entry["rfilename"]
        digest = sha256(path)
        if entry.get("lfs") and digest != entry["lfs"]["sha256"]:
            raise ValueError("Speech observer source digest mismatch")
        files.append({"file": entry["rfilename"], "sha256": digest})
    (directory / "SOURCE.json").write_text(json.dumps({"repo": REPO, "revision": info["sha"], "files": files}, indent=2))
    print("SPEECH_OBSERVER_READY " + info["sha"], flush=True)


def word_counts(reference, hypothesis):
    import jiwer

    result = jiwer.process_words(reference, hypothesis)
    return {"errors": result.substitutions + result.deletions + result.insertions,
            "reference_words": len(reference.split()), "wer": result.wer}


def run(root):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    root = Path(root)
    source = root / "improvement" / "speech-observer"
    provenance = json.loads((source / "SOURCE.json").read_text())
    for entry in provenance["files"]:
        if sha256(source / "model" / entry["file"]) != entry["sha256"]:
            raise ValueError("Speech observer source changed")
    final = json.loads((root / "improvement" / "FINAL_COMPARISON.json").read_text())
    if not final["measured_retention_gate_passed"]:
        raise ValueError("A rejected candidate does not need a speech-retention decision")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    model = WhisperForConditionalGeneration.from_pretrained(source / "model", local_files_only=True, dtype=torch.float32).eval()
    processor = WhisperProcessor.from_pretrained(source / "model", local_files_only=True)
    cases = json.loads((root / "fixtures" / "cases.json").read_text())
    speech_cases = [case for case in cases if case["modality"] == "speech"]
    rows = []
    totals = {}
    for arm, directory in (("original", root / "reports" / "baseline"),
                           ("edited", root / "reports" / "improvement-regression")):
        for case in speech_cases:
            path = directory / f"{case['id']}.wav"
            waveform, rate = sf.read(path, dtype="float32")
            if waveform.ndim != 1 or not np.isfinite(waveform).all():
                raise ValueError("Invalid speech waveform")
            if rate != 16000:
                divisor = np.gcd(rate, 16000)
                waveform = resample_poly(waveform, 16000 // divisor, rate // divisor).astype(np.float32)
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
            tokens = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            text = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            expected = processor.tokenizer.normalize(case["prompt"])
            observed = processor.tokenizer.normalize(text)
            row = {"id": case["id"], "arm": arm, "expected_text": case["prompt"], "observed_text": text,
                   "normalized_reference": expected, "normalized_transcript": observed,
                   "waveform_sha256": sha256(path), **word_counts(expected, observed)}
            rows.append(row)
        arm_rows = [row for row in rows if row["arm"] == arm]
        errors = sum(row["errors"] for row in arm_rows)
        words = sum(row["reference_words"] for row in arm_rows)
        totals[arm] = {"errors": errors, "reference_words": words, "wer": errors / words}
    result = {"observer": provenance, "rows": rows, "totals": totals,
              "candidate_no_worse_on_transcription": totals["edited"]["wer"] <= totals["original"]["wer"],
              "limits": "Independent recognizer on four fixed synthesized sentences; not a naturalness or general speech-quality evaluation."}
    destination = root / "improvement" / "SPEECH_COMPARISON.json"
    with destination.open("x") as output:
        json.dump(result, output, indent=2)
    print("SPEECH_COMPARISON_COMPLETE " + json.dumps({"totals": totals, "candidate_no_worse": result["candidate_no_worse_on_transcription"]}), flush=True)


class SpeechTests(unittest.TestCase):
    def test_content_error_is_counted(self):
        self.assertEqual(word_counts("good morning", "good morning")["errors"], 0)
        self.assertEqual(word_counts("good morning", "bad morning")["errors"], 1)
        self.assertEqual(word_counts("three green cups", "three cups")["errors"], 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["speech_check.py"], verbosity=2)
    elif not args.root:
        parser.error("--root required")
    elif args.prepare:
        prepare(args.root)
    else:
        run(args.root)