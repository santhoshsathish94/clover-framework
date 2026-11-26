import argparse
import hashlib
import json
import math
from pathlib import Path
import unittest

from experiment import read_results


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def paired_counts(before, after, identifiers):
    fixed, broken, kept_correct, kept_wrong = [], [], [], []
    for identifier in identifiers:
        was_correct, now_correct = before[identifier]["correct"], after[identifier]["correct"]
        destination = kept_correct if was_correct and now_correct else broken if was_correct else fixed if now_correct else kept_wrong
        destination.append(identifier)
    discordant = len(fixed) + len(broken)
    probability = min(1.0, 2 * sum(math.comb(discordant, count) for count in range(min(len(fixed), len(broken)) + 1)) / 2 ** discordant) if discordant else 1.0
    return {"cases": len(identifiers), "before_correct": len(kept_correct) + len(broken),
            "after_correct": len(kept_correct) + len(fixed), "fixed": fixed, "broken": broken,
            "two_sided_exact_p": probability,
            "net_accuracy_change": (len(fixed) - len(broken)) / len(identifiers) if identifiers else None}


def decide(primary, regression_broken, asr_worse, speech_missing):
    if regression_broken or asr_worse or speech_missing or primary["after_correct"] < primary["before_correct"]:
        return "reject", "The primary intervention harms a measured capability or does not preserve the regression boundary."
    if primary["after_correct"] == primary["before_correct"]:
        return "reject", "The primary intervention does not improve held-out generated-answer accuracy."
    if primary["two_sided_exact_p"] > 0.05:
        return "investigate further", "A held-out increase is too small to establish a reliable improvement on this sample."
    return "investigate further", "A promising held-out result still needs broader independent evaluation and speech-quality assessment before retention."


def finalize(root):
    from benchmark import verify_seal
    from survey import verify_original

    root = Path(root)
    verify_original(root)
    verify_seal(root)
    experiment = json.loads((root / "reports" / "experiment-complete.json").read_text())
    if not experiment["all_experimental_copies_reverted"]:
        raise ValueError("Reversal gate not complete")
    baseline_dir = root / "reports" / "baseline"
    baseline = read_results(baseline_dir / "results.jsonl")
    baseline_summary = json.loads((baseline_dir / "summary.json").read_text())
    cases = json.loads((root / "fixtures" / "cases.json").read_text())
    validation_ids = [case["id"] for case in cases if case["split"] == "validation"]
    discovery_ids = [case["id"] for case in cases if case["split"] == "discovery"]
    regression_ids = [case["id"] for case in cases if case["split"] == "regression" and "gold_letter" in case]
    records = []
    for percentage in (5, 10, 20):
        name = f"H1-alpha-{percentage:02d}"
        directory = root / "reports" / name
        rows = read_results(directory / "results.jsonl")
        summary = json.loads((directory / "summary.json").read_text())
        audit = json.loads((root / "copies" / name / "PATCH_AUDIT.json").read_text())
        reversal = json.loads((root / "copies" / name / "REVERSAL.json").read_text())
        if sha256(root / "copies" / name / "model.safetensors") != audit["base_checkpoint_sha256"]:
            raise ValueError(f"Copy is not restored: {name}")
        if not reversal["byte_exact_reversal"] or audit["changed_tensors"] != ["model.language_model.embedding_norm.weight"]:
            raise ValueError("Single-tensor/reversal audit failed")
        validation = paired_counts(baseline, rows, validation_ids)
        discovery = paired_counts(baseline, rows, discovery_ids)
        regression = paired_counts(baseline, rows, regression_ids)
        changed_outputs = [identifier for identifier in baseline if baseline[identifier]["tokens"] != rows[identifier]["tokens"]]
        changed_logits = [identifier for identifier in baseline if baseline[identifier].get("letter_logits") != rows[identifier].get("letter_logits")]
        speech_changes = [identifier for identifier in baseline if baseline[identifier].get("speech", {}).get("codes_sha256") != rows[identifier].get("speech", {}).get("codes_sha256")]
        failed_discovery = [identifier for identifier in discovery_ids if not baseline[identifier]["correct"]]
        margin_delta = sum(rows[identifier]["gold_margin"] - baseline[identifier]["gold_margin"] for identifier in failed_discovery) / len(failed_discovery)
        asr_before = baseline_summary["groups"]["regression:audio_asr"]["corpus_word_error_rate"]
        asr_after = summary["groups"]["regression:audio_asr"]["corpus_word_error_rate"]
        speech_missing = summary["groups"]["regression:speech"]["speech_waveforms_nonempty"] != 4
        decision, reason = decide(validation, regression["broken"], asr_after > asr_before, speech_missing)
        records.append({
            "name": name, "requested_attenuation": percentage / 100, "primary": percentage == 10,
            "discovery": discovery, "validation": validation, "regression_mcq": regression,
            "secondary_multiplicity_adjusted_p": min(1, validation["two_sided_exact_p"] * 3) if percentage != 10 else None,
            "asr_wer_before": asr_before, "asr_wer_after": asr_after,
            "changed_generated_token_cases": changed_outputs,
            "changed_letter_logit_cases": changed_logits,
            "changed_speech_code_cases": speech_changes,
            "mean_failed_discovery_margin_delta": margin_delta,
            "groups": summary["groups"], "decision": decision, "reason": reason,
            "one_tensor_changed": True, "byte_exact_reversal": True,
        })
    primary = next(record for record in records if record["primary"])
    report = {"recommendation": primary["decision"], "reason": primary["reason"],
              "retain_original_checkpoint": True, "baseline": baseline_summary["groups"],
              "no_op_control": experiment["no_op"], "arms": records,
              "original_hashes_verified_at_handoff": True,
              "limitations": ["One checkpoint, one CPU/inference implementation, small constructed evaluation",
                              "Only H1 coordinate 36 tested; H2/H3 were not tried",
                              "No natural-image or real-world video benchmark",
                              "Speech naturalness/intelligibility not independently scored in the frozen suite",
                              "No gradient training, deployment, merge, or original weight changes"]}
    with (root / "reports" / "COMPARISON.json").open("x") as output:
        json.dump(report, output, indent=2)
    lines = [
        "# Before-and-after results", "",
        f"## Recommendation: {primary['decision'].capitalize()}", "",
        primary["reason"], "Retain the original checkpoint. No experimental checkpoint is recommended for deployment.", "",
        "The predeclared primary trial is H1 at 10%; 5% and 20% are sensitivity analyses, not candidates selected afterward from validation scores.",
        "", "## Held-out generated answers", "",
        "| Arm | Discovery /32 | Validation /32 | Fixed / broken validation | Exact paired p |", "|---|---:|---:|---:|---:|",
        f"| Original | {baseline_summary['groups']['discovery:text']['correct']} | {baseline_summary['groups']['validation:text']['correct']} | n/a | n/a |",
    ]
    for record in records:
        paired = record["validation"]
        lines.append(f"| {record['name']}{' (primary)' if record['primary'] else ''} | {record['discovery']['after_correct']} | {paired['after_correct']} | {len(paired['fixed'])} / {len(paired['broken'])} | {paired['two_sided_exact_p']:.6g} |")
    lines += ["", "Exact paired probabilities condition on discordant items (two-sided binomial/McNemar). Secondary adjusted probabilities are in COMPARISON.json. This sample is not a general model-accuracy estimate or an equivalence proof.",
              "", "## Unrelated and multimodal regression", "",
              "| Capability | Original | 5% | 10% | 20% |", "|---|---:|---:|---:|---:|"]
    for group, label in (("regression:text", "Unrelated text /16"), ("regression:vision", "Controlled vision /12"),
                         ("regression:video", "Controlled video /8"), ("regression:audio", "Non-speech audio /2"),
                         ("regression:audio_asr", "ASR exact /6")):
        values = [baseline_summary["groups"][group]["correct"]] + [record["groups"][group]["correct"] for record in records]
        lines.append(f"| {label} | " + " | ".join(map(str, values)) + " |")
    wers = [baseline_summary["groups"]["regression:audio_asr"]["corpus_word_error_rate"]] + [record["asr_wer_after"] for record in records]
    lines.append("| ASR corpus WER (lower better) | " + " | ".join(f"{value:.1%}" for value in wers) + " |")
    speech_counts = [baseline_summary["groups"]["regression:speech"]["speech_waveforms_nonempty"]] + [record["groups"]["regression:speech"]["speech_waveforms_nonempty"] for record in records]
    lines.append("| Nonempty speech waveforms /4 | " + " | ".join(map(str, speech_counts)) + " |")
    lines += ["", "Waveform output is not a speech-quality score. Greedy audio sampling differs from the publisher's sampled default. Naturalness and intelligibility require a separate independent assessment.",
              "", "## Mechanism and controls", "",
              f"- No-op physical copy: {experiment['no_op']['cases_exact']}/112 cases have identical inputs, generated tokens, letter logits and speech code/waveform hashes.",
              f"- No-op activation arrays identical: {experiment['no_op']['unchanged_required_arrays']}.",
              "- Each nonzero trial changed exactly one stored tensor and one BF16 gain coordinate, with all other checkpoint files unchanged.",
              "- Every required upstream activation array remained bit-identical; downstream changes were observed, so the intervention was live.",
              "- Each experimental copy was reversed and its complete model SHA-256 matched the original; the original itself remained read-only and hash-identical.", ""]
    for record in records:
        lines.append(f"- {record['name']}: {len(record['changed_letter_logit_cases'])} cases with changed letter logits, {len(record['changed_generated_token_cases'])} with changed generated token sequences, {len(record['changed_speech_code_cases'])} with changed speech-code streams; mean failed-discovery margin change {record['mean_failed_discovery_margin_delta']:+.6g}.")
    lines += ["", "A margin movement is a measured numerical effect, not proof that a behavioural weakness was fixed. Observed associations and untested H2/H3 remain hypotheses.",
              "", "## Boundaries and next evidence", "",
              "No gradient training, tensor clipping, multi-component edits, production service changes, model deployment, merge or push occurred.",
              "The experiment tests one coordinate and three strengths, not every possible weight-space intervention. The uniformly finite weights and statistical outliers alone do not diagnose defects.",
              "Further work should preregister a new independent question rather than reuse this validation set for selecting another coordinate. Audio-control failures and speech fidelity need separate, appropriately grounded modality evaluations.", ""]
    with (root / "reports" / "RESULTS.md").open("x") as output:
        output.write("\n".join(lines))
    print("FINAL_RECOMMENDATION " + json.dumps({"decision": report["recommendation"], "reason": report["reason"],
                                               "primary_validation": primary["validation"]}), flush=True)


class FinalizeTests(unittest.TestCase):
    def test_exact_discordant_probability(self):
        before = {str(index): {"correct": False} for index in range(6)}
        after = {str(index): {"correct": True} for index in range(6)}
        paired = paired_counts(before, after, list(before))
        self.assertEqual(paired["two_sided_exact_p"], 0.03125)
        self.assertEqual(len(paired["fixed"]), 6)

    def test_no_change_is_not_improvement(self):
        before = {"case": {"correct": True}}
        paired = paired_counts(before, before, ["case"])
        self.assertEqual(paired["two_sided_exact_p"], 1)
        self.assertEqual(decide(paired, [], False, False)[0], "reject")

    def test_regression_prevents_retention(self):
        primary = {"before_correct": 10, "after_correct": 16, "two_sided_exact_p": 0.03125}
        self.assertEqual(decide(primary, ["broken"], False, False)[0], "reject")
        self.assertEqual(decide(primary, [], False, False)[0], "investigate further")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["finalize.py"], verbosity=2)
    elif args.root:
        finalize(args.root)
    else:
        parser.error("--root or --self-test required")