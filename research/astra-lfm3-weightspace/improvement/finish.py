import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def retention_decision(comparison, speech):
    if not comparison["measured_retention_gate_passed"]:
        return "reject", "The frozen final or regression gate did not pass."
    changed = comparison["regression"]["speech_code_streams_changed"]
    if changed:
        if speech is None:
            return "incomplete", "Changed speech has not received the required independent content check."
        if not speech["candidate_no_worse_on_transcription"]:
            return "reject", "Independent speech transcription regressed."
    return "retain_scoped_improvement", "The measured arithmetic-choice improvement passed the frozen final and regression gates."


def finish(root):
    root = Path(root)
    directory = root / "improvement"
    comparison = json.loads((directory / "FINAL_COMPARISON.json").read_text())
    speech_path = directory / "SPEECH_COMPARISON.json"
    speech = json.loads(speech_path.read_text()) if speech_path.exists() else None
    decision, reason = retention_decision(comparison, speech)
    if decision == "incomplete":
        raise RuntimeError(reason)
    candidate = Path(comparison["candidate_path"])
    audit = json.loads((candidate / "PATCH_AUDIT.json").read_text())
    if audit["changed_tensors"] != ["model.language_model.layers.22.feed_forward.w2.weight"]:
        raise ValueError("Final candidate changed an unexpected component")
    if decision == "reject" and not (candidate / "REVERSAL.json").exists():
        subprocess.run([sys.executable, "-B", str(directory / "interventions.py"), "--root", str(root), "--revert", "final-candidate"], check=True)
    actual_candidate_hash = sha256(candidate / "model.safetensors")
    expected = comparison["candidate_checkpoint_sha256"] if decision == "retain_scoped_improvement" else comparison["original_checkpoint_sha256"]
    if actual_candidate_hash != expected:
        raise ValueError("Final on-disk artifact does not match the decision")
    for name in ("TRIALS.json", "FOLLOWUP_TRIALS.json"):
        trials = json.loads((directory / name).read_text())
        for trial in trials["trials"]:
            trial_copy = directory / "copies" / trial["name"]
            reversal = json.loads((trial_copy / "REVERSAL.json").read_text())
            if reversal["original_sha256_restored"] != comparison["original_checkpoint_sha256"]:
                raise ValueError("A trial lacks a verified original reversal")
    pair = comparison["final_comparison"]
    initial = pair["original_correct"] / pair["total_responses"]
    proposed = pair["edited_correct"] / pair["total_responses"]
    result = {**comparison, "decision": decision, "decision_reason": reason,
              "speech_content_assessment": speech, "initial_arithmetic_mcq_accuracy": initial,
              "candidate_arithmetic_mcq_accuracy": proposed,
              "candidate_gain_percentage_points": 100 * (proposed - initial),
              "final_checkpoint_path": str(candidate if decision == "retain_scoped_improvement" else root / "original"),
              "final_checkpoint_sha256": actual_candidate_hash,
              "tested_candidate_retained": decision == "retain_scoped_improvement",
              "all_trial_reversals_verified": True,
              "no_universal_superiority_claim": True}
    with (directory / "OUTCOME.json").open("x") as output:
        json.dump(result, output, indent=2)
    lines = ["# Initial-to-final LFM3-Omni outcome", "", f"## Decision: {decision}", "", reason, "",
             "The candidate was selected on tuning only. Its final set was not used to choose its layer, direction, columns or strength.",
             "", "## Initial versus tested candidate", "",
             "| Measure | Original | Candidate |", "|---|---:|---:|"]
    for name, label in (("arithmetic_mcq", "Arithmetic choices"), ("arithmetic_free", "Arithmetic free response"), ("selection", "Direct number selection")):
        before, after = comparison["initial_totals"][name], comparison["final_totals"][name]
        lines.append(f"| {label} | {before['correct']}/{before['total']} | {after['correct']}/{after['total']} |")
    lines += ["", f"Arithmetic-choice accuracy: {initial:.2%} -> {proposed:.2%} ({100*(proposed-initial):+.2f} percentage points).",
              f"Underlying questions improved/worsened: {pair['better_questions']}/{pair['worse_questions']}; one-sided exact sign p={pair['one_sided_exact_sign_p']:.6g}.",
              f"Fixed/broken individual arithmetic-choice responses: {len(pair['fixed_answers'])}/{len(pair['broken_answers'])}. Four rotations of one question were treated as dependent, not four independent successes.",
              "", "## Regression evidence", "",
              f"Newly broken previously correct unrelated/multimodal cases: {comparison['regression']['newly_broken_previously_correct']}.",
              f"ASR word error worse: {comparison['asr_worse']}. Changed speech-code streams: {comparison['regression']['speech_code_streams_changed']}."]
    if speech:
        lines.append(f"Independent speech observer: {speech['observer']['repo']} at {speech['observer']['revision']}; original WER {speech['totals']['original']['wer']:.3f}, candidate WER {speech['totals']['edited']['wer']:.3f} over four fixed sentences.")
    lines += ["This is not a human naturalness assessment, a large natural-media benchmark, or proof of universal improvement. The reused regression set is a gate, not a fresh generalization claim.",
              "", "## What changed", "",
              "One tensor: `model.language_model.layers.22.feed_forward.w2.weight`, selected64 output columns, sparse directional subtraction at strength4. No gradient training; embeddings, final norm, output head and all other tensors unchanged.",
              f"Changed BF16 entries: {audit['changed_entries']}. The full tensor audit is in the candidate's PATCH_AUDIT.json.",
              "The direction came from matched discovery arithmetic/reference states, restricted to problems with correct free answers and explicit number binding. It is not a Qwen direction or an edit justified by a weight outlier.",
              "", "## Initial and final artifacts", "",
              f"Original checkpoint SHA-256: `{comparison['original_checkpoint_sha256']}`.",
              f"Tested candidate SHA-256: `{comparison['candidate_checkpoint_sha256']}`.",
              f"Retained final SHA-256: `{actual_candidate_hash}`.",
              f"Final directory on K3: `{result['final_checkpoint_path']}`.",
              "All exploratory trial copies were reversed exactly. A rejected final candidate is also restored; a retained candidate remains a separate physical copy. No original was overwritten and no model was deployed.",
              "", "## Study accounting", "",
              "Discovery114 responses plus96 matched references; original tuning228 plus six initial and two adaptive follow-up trials; final original/candidate456 each;112 regression responses. The stronger-dose follow-up was documented before it ran and used tuning only.",
              "The primary final acceptance criteria were not weakened after observing results. All raw outputs, input hashes, per-question comparisons, frozen source/case hashes, hypothesis files and reversal records are retained.",
              "", "## Remaining limits", "",
              "Constructed arithmetic-choice validation is a limited domain. Even a retained candidate is not established to improve free-form reasoning, all factual knowledge, all languages, general vision/audio/video, or natural speech quality. Quantized exports and deployed Clover runtimes were not evaluated.",
              "The result establishes only the measured scope. A failed gate means the original remains the final version; activity or tuning gains are not substituted for a verified improvement.", ""]
    with (directory / "RESULTS.md").open("x") as output:
        output.write("\n".join(lines))
    print("IMPROVEMENT_OUTCOME " + json.dumps({"decision": decision, "initial": comparison["initial_totals"],
                                              "candidate": comparison["final_totals"],
                                              "final_sha256": actual_candidate_hash}), flush=True)


class OutcomeTests(unittest.TestCase):
    def test_failed_primary_is_rejected(self):
        self.assertEqual(retention_decision({"measured_retention_gate_passed": False}, None)[0], "reject")

    def test_changed_speech_needs_independent_check(self):
        comparison = {"measured_retention_gate_passed": True, "regression": {"speech_code_streams_changed": ["tts"]}}
        self.assertEqual(retention_decision(comparison, None)[0], "incomplete")
        self.assertEqual(retention_decision(comparison, {"candidate_no_worse_on_transcription": False})[0], "reject")
        self.assertEqual(retention_decision(comparison, {"candidate_no_worse_on_transcription": True})[0], "retain_scoped_improvement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["finish.py"], verbosity=2)
    elif args.root:
        finish(args.root)
    else:
        parser.error("--root or --self-test required")