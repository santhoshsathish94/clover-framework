import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

from study import OLD64_HASH, sha256, verify_frozen
from validate_final import group_comparison, read_rows, regression_comparison


def choose_width(original, control, trials):
    selected = []
    for trial in trials:
        if trial["width"] == 64:
            continue
        totals = trial["totals"]
        if totals["arithmetic_mcq"]["correct"] <= max(original["arithmetic_mcq"]["correct"], control["arithmetic_mcq"]["correct"]):
            continue
        if any(totals[key]["correct"] < max(original[key]["correct"], control[key]["correct"])
               for key in ("arithmetic_free", "selection")):
            continue
        selected.append(trial)
    selected.sort(key=lambda item: (-item["totals"]["arithmetic_mcq"]["correct"], item["width"], item["actual_delta_frobenius"], item["name"]))
    return selected[0] if selected else None


def exact_control(previous, reproduced):
    if previous.keys() != reproduced.keys():
        raise ValueError("64-neuron control case coverage differs")
    for identifier in previous:
        for key in ("input_sha256", "tokens", "text", "correct", "predicted_value"):
            if previous[identifier].get(key) != reproduced[identifier].get(key):
                raise ValueError(f"64-neuron control differs at {identifier}: {key}")


def child(root, script, *args):
    print("WIDTH_STAGE " + json.dumps({"script": script, "args": list(map(str, args))}), flush=True)
    subprocess.run([sys.executable, "-B", str(root / script), "--root", str(root), *map(str, args)], check=True)


def copy_path(root, name):
    return root / "improvement" / "copies" / f"width-{name}"


def recreate_for_confirmation(root, name):
    path = copy_path(root, name)
    reversal = json.loads((path / "REVERSAL.json").read_text())
    if sha256(path / "model.safetensors") != reversal["original_sha256_restored"]:
        raise ValueError("Trial copy is not restored before confirmation")
    archive = root / "improvement" / "neuron_count" / "restored_trials" / path.name
    if archive.exists():
        raise FileExistsError(archive)
    archive.parent.mkdir(exist_ok=True)
    path.rename(archive)
    child(root, "improvement/neuron_count/study.py", "--copy", name)


def speech_content_check(root, regression_name, work):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from speech_check import word_counts

    source = root / "improvement" / "speech-observer"
    provenance = json.loads((source / "SOURCE.json").read_text())
    for entry in provenance["files"]:
        if sha256(source / "model" / entry["file"]) != entry["sha256"]:
            raise ValueError("Independent speech observer changed")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    model = WhisperForConditionalGeneration.from_pretrained(source / "model", local_files_only=True, dtype=torch.float32).eval()
    processor = WhisperProcessor.from_pretrained(source / "model", local_files_only=True)
    cases = [case for case in json.loads((root / "fixtures" / "cases.json").read_text()) if case["modality"] == "speech"]
    records, totals = [], {}
    for arm, directory in (("original", root / "reports" / "baseline"), ("wider", root / "reports" / regression_name)):
        for case in cases:
            path = directory / f"{case['id']}.wav"
            waveform, rate = sf.read(path, dtype="float32")
            if waveform.ndim != 1 or not np.isfinite(waveform).all():
                raise ValueError("Invalid speech waveform")
            divisor = int(np.gcd(rate, 16000))
            waveform = resample_poly(waveform, 16000 // divisor, rate // divisor).astype(np.float32)
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
            output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            text = processor.batch_decode(output, skip_special_tokens=True)[0]
            reference = processor.tokenizer.normalize(case["prompt"])
            hypothesis = processor.tokenizer.normalize(text)
            records.append({"arm": arm, "id": case["id"], "expected": case["prompt"], "transcript": text,
                            "waveform_sha256": sha256(path), **word_counts(reference, hypothesis)})
        subset = [record for record in records if record["arm"] == arm]
        errors = sum(record["errors"] for record in subset)
        count = sum(record["reference_words"] for record in subset)
        totals[arm] = {"errors": errors, "words": count, "wer": errors / count}
    result = {"observer": provenance, "records": records, "totals": totals,
              "no_worse": totals["wider"]["wer"] <= totals["original"]["wer"],
              "limits": "Independent transcription of four fixed sentences, not naturalness or broad speech quality."}
    (work / "SPEECH_CHECK.json").write_text(json.dumps(result, indent=2))
    return result


def confirm(root, selected, trials):
    root = Path(root)
    work = root / "improvement" / "neuron_count"
    selection = {"selected": selected, "tuning_trials_sha256": sha256(work / "TRIALS.json"),
                 "frozen_before_confirmation": True, "confirmation_sha256": sha256(work / "frozen" / "confirmation.json")}
    with (work / "SELECTION.json").open("x") as stream:
        json.dump(selection, stream, indent=2)
    active = []
    retain_name = None
    try:
        for name in ("n64-fixed", selected["name"]):
            recreate_for_confirmation(root, name)
            active.append(name)
        child(root, "improvement/neuron_count/study.py", "--confirm-name", "original")
        child(root, "improvement/neuron_count/study.py", "--confirm-name", "n64",
              "--checkpoint", copy_path(root, "n64-fixed"))
        child(root, "improvement/neuron_count/study.py", "--confirm-name", "wider",
              "--checkpoint", copy_path(root, selected["name"]))
        rows = {name: read_rows(work / "confirmation" / name / "results.jsonl") for name in ("original", "n64", "wider")}
        summaries = {name: json.loads((work / "confirmation" / name / "summary.json").read_text()) for name in rows}
        against_original = group_comparison(rows["original"], rows["wider"])
        against64 = group_comparison(rows["n64"], rows["wider"])
        primary = all(comparison["edited_correct"] > comparison["original_correct"]
                      and comparison["one_sided_exact_sign_p"] <= 0.05 for comparison in (against_original, against64))
        protected = all(summaries["wider"]["totals"][key]["correct"] >= max(summaries[baseline]["totals"][key]["correct"] for baseline in ("original", "n64"))
                        for key in ("arithmetic_free", "selection"))
        regression_name = "neuron-count-regression"
        child(root, "benchmark.py", "--name", regression_name, "--checkpoint", copy_path(root, selected["name"]))
        regression = regression_comparison(read_rows(root / "reports" / "baseline" / "results.jsonl"),
                                           read_rows(root / "reports" / regression_name / "results.jsonl"))
        new_reg = json.loads((root / "reports" / regression_name / "summary.json").read_text())
        old_reg = json.loads((root / "reports" / "baseline" / "summary.json").read_text())
        asr_worse = new_reg["groups"]["regression:audio_asr"]["corpus_word_error_rate"] > old_reg["groups"]["regression:audio_asr"]["corpus_word_error_rate"]
        regression_ok = not regression["newly_broken_previously_correct"] and not asr_worse
        speech = None
        if primary and protected and regression_ok and regression["speech_code_streams_changed"]:
            speech = speech_content_check(root, regression_name, work)
        passed = primary and protected and regression_ok and (speech is None or speech["no_worse"])
        if passed:
            retain_name = selected["name"]
        report = {"decision": "retain_scoped_wider_edit" if passed else "reject_wider_edit", "selected": selected,
                  "confirmation_totals": {name: result["totals"] for name, result in summaries.items()},
                  "versus_original": against_original, "versus64": against64,
                  "primary_gate": primary, "protected_totals_gate": protected, "regression_gate": regression_ok,
                  "regression": regression, "asr_worse": asr_worse, "speech_check": speech,
                  "tested_candidate_sha256": sha256(copy_path(root, selected["name"]) / "model.safetensors"),
                  "retained_copy": str(copy_path(root, retain_name)) if retain_name else str(root / "original"),
                  "limits": "Edited-neuron count, not architectural expansion; constructed evaluation; no universal quality or naturalness claim."}
        (work / "CONFIRMATION.json").write_text(json.dumps(report, indent=2))
        print("WIDTH_CONFIRMATION_DECISION " + json.dumps({"decision": report["decision"], "totals": report["confirmation_totals"],
                                                           "p_original": against_original["one_sided_exact_sign_p"],
                                                           "p64": against64["one_sided_exact_sign_p"]}), flush=True)
        return report
    finally:
        for name in active:
            if name != retain_name:
                child(root, "improvement/interventions.py", "--revert", f"width-{name}")


def report_width(work, record):
    original = record["original_tuning"]
    control = next(trial for trial in record["trials"] if trial["width"] == 64)
    lines = ["# Edited-neuron-count experiment", "",
             "This increases the number of existing layer22 MLP neurons whose output columns are edited. It does not add architectural neurons or retrain the model.",
             "Same layer, discovered direction, native runtime and tuning cases; fixed-strength and matched-total-norm tracks separate coverage from perturbation size.", "",
             "| Edited neurons | Track | Strength | Delta norm | Arithmetic /144 | Free /36 | Selection /48 |",
             "|---:|---|---:|---:|---:|---:|---:|",
             f"| 0 | Original | 0 | 0 | {original['arithmetic_mcq']['correct']} | {original['arithmetic_free']['correct']} | {original['selection']['correct']} |"]
    for trial in record["trials"]:
        totals = trial["totals"]
        lines.append(f"| {trial['width']} | {trial['track']} | {trial['strength']:.6g} | {trial['actual_delta_frobenius']:.6g} | {totals['arithmetic_mcq']['correct']} | {totals['arithmetic_free']['correct']} | {totals['selection']['correct']} |")
    lines += ["", "The64-neuron control exactly matched the previous tested checkpoint hash and all tuning outputs. Each trial changed exactly one tensor and its declared columns, with byte-exact reversal afterward.", ""]
    if record["selected"] is None:
        lines += ["## Outcome", "", "No wider candidate beat both the original and64-neuron control while preserving free-response and selection totals. No fresh confirmation was opened and no wider model was retained."]
    else:
        confirmation = record["confirmation"]
        lines += ["## Fresh confirmation", "", f"Decision: {confirmation['decision']}.", "",
                  "| Measure | Original | 64-neuron control | Selected wider edit |", "|---|---:|---:|---:|"]
        for key in ("arithmetic_mcq", "arithmetic_free", "selection"):
            values = [confirmation["confirmation_totals"][arm][key] for arm in ("original", "n64", "wider")]
            lines.append(f"| {key} | " + " | ".join(f"{value['correct']}/{value['total']}" for value in values) + " |")
        lines += ["", f"Grouped one-sided sign p versus original: {confirmation['versus_original']['one_sided_exact_sign_p']:.6g}; versus64: {confirmation['versus64']['one_sided_exact_sign_p']:.6g}.",
                  f"Newly broken unrelated/multimodal cases: {confirmation['regression']['newly_broken_previously_correct']}.",
                  "Fresh confirmation contained96 underlying questions/456 responses disjoint from all earlier sets. Four rotations of a question were not counted as independent samples."]
    lines += ["", "## Limits and artifacts", "",
              "Tuning results are exploratory. Wider coverage can also cause larger interference; more edited neurons is not automatically more capability. Old final data were not reused for confirmation.",
              "Originals remain immutable, no gradients or deployment. Trial sources, frozen neuron sets, actual rounded perturbation norms, raw outputs and all tensor/reversal audits are retained. A failed confirmation leaves the original as the retained final model.", ""]
    (work / "RESULTS.md").write_text("\n".join(lines))


def run(root):
    root = Path(root)
    verify_frozen(root)
    work = root / "improvement" / "neuron_count"
    if (work / "TRIALS.json").exists():
        raise FileExistsError("Preserve previous width-trial results")
    specs = json.loads((work / "frozen" / "specs.json").read_text())
    original = json.loads((root / "improvement" / "runs" / "original-tuning" / "summary.json").read_text())["totals"]
    prior64 = read_rows(root / "improvement" / "runs" / "layer22-followup-a4" / "results.jsonl")
    trials = []
    for spec in specs:
        name = spec["name"]
        child(root, "improvement/neuron_count/study.py", "--copy", name)
        try:
            run_name = f"width-{name}"
            child(root, "improvement/evaluate.py", "--split", "tuning", "--name", run_name,
                  "--checkpoint", copy_path(root, name))
            directory = root / "improvement" / "runs" / run_name
            summary = json.loads((directory / "summary.json").read_text())
            if spec["width"] == 64:
                exact_control(prior64, read_rows(directory / "results.jsonl"))
                if sha256(copy_path(root, name) / "model.safetensors") != OLD64_HASH:
                    raise ValueError("64-neuron checkpoint control changed")
            trials.append({**spec, "totals": summary["totals"], "results_sha256": summary["results_sha256"]})
            print("WIDTH_TRIAL_RESULT " + json.dumps(trials[-1]), flush=True)
        finally:
            child(root, "improvement/interventions.py", "--revert", f"width-{name}")
    control = next(trial for trial in trials if trial["width"] == 64)
    selected = choose_width(original, control["totals"], trials)
    record = {"original_tuning": original, "trials": trials, "selected": selected,
              "old64_checkpoint_and_outputs_exact": True, "trial_copies_reversed": True,
              "fresh_confirmation_opened": False}
    with (work / "TRIALS.json").open("x") as stream:
        json.dump(record, stream, indent=2)
    if selected is not None:
        record["confirmation"] = confirm(root, selected, trials)
        record["fresh_confirmation_opened"] = True
    else:
        record["decision"] = "no_qualifying_wider_edit"
    (work / "OUTCOME.json").write_text(json.dumps(record, indent=2))
    report_width(work, record)
    print("NEURON_COUNT_STUDY_COMPLETE " + json.dumps({"selected": selected,
                                                      "decision": record.get("confirmation", {}).get("decision", record.get("decision"))}), flush=True)


class RunTests(unittest.TestCase):
    def test_must_beat64_and_preserve_other_totals(self):
        original = {"arithmetic_mcq": {"correct": 74}, "arithmetic_free": {"correct": 32}, "selection": {"correct": 48}}
        control = {**original, "arithmetic_mcq": {"correct": 79}}
        candidate = {"name": "n128-fixed", "width": 128, "actual_delta_frobenius": 2,
                     "totals": {**original, "arithmetic_mcq": {"correct": 78}}}
        self.assertIsNone(choose_width(original, control, [candidate]))
        candidate["totals"]["arithmetic_mcq"]["correct"] = 80
        self.assertEqual(choose_width(original, control, [candidate]), candidate)
        candidate["totals"]["arithmetic_free"] = {"correct": 31}
        self.assertIsNone(choose_width(original, control, [candidate]))

    def test_control_output_change_is_rejected(self):
        before = {"item": {"tokens": [1], "text": "A", "correct": True, "input_sha256": "same"}}
        after = {"item": {"tokens": [2], "text": "B", "correct": True, "input_sha256": "same"}}
        with self.assertRaises(ValueError):
            exact_control(before, after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["run.py"], verbosity=2)
    elif args.root:
        run(args.root)
    else:
        parser.error("--root or --self-test required")