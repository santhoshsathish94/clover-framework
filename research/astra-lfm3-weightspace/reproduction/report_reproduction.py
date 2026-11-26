import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import unittest


COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_pair(original, edited, original_protocol, edited_protocol):
    for field in ("selected_cases", "max_new_tokens", "thinking", "temperature", "top_p", "top_k",
                  "presence_penalty", "seed", "batch_size", "backend", "grader_source_sha256", "grader_selected_ast_sha256"):
        if original_protocol[field] != edited_protocol[field]:
            raise ValueError(f"Paired protocol differs: {field}")
    if original["id"] != edited["id"] or original["input_sha256"] != edited["input_sha256"]:
        raise ValueError("Paired input differs")
    return {
        "id": original["id"], "input_sha256": original["input_sha256"],
        "max_new_tokens": original_protocol["max_new_tokens"],
        "original": {key: original[key] for key in ("correct", "prediction", "gold", "generated_tokens", "hit_cap", "thinking_closed", "elapsed_s", "repetition")},
        "edited": {key: edited[key] for key in ("correct", "prediction", "gold", "generated_tokens", "hit_cap", "thinking_closed", "elapsed_s", "repetition")},
        "qualification": "One independently sampled CPU pair, not a benchmark accuracy estimate or exact replay of the authors' GPU RNG.",
    }


def report(root):
    root = Path(root)
    results = root / "results"
    reconstruction = load(results / "breakout-reconstruction" / "result.json")
    game = load(results / "breakout-1000" / "result.json")
    weights = load(results / "published-audit" / "weight-audit.json")
    operator = load(results / "published-audit" / "operator-validation.json")
    score_replay = load(results / "published-audit" / "recorded-score-replay.json")
    teacher = load(results / "qwen-teacher-control" / "result.json")
    regraded = load(results / "recorded-math-regrade" / "result.json")
    pair_rows, protocols = {}, {}
    for arm in ("original", "edited"):
        directory = results / f"qwen-{arm}-full"
        summary = load(directory / "summary.json")
        if not summary["original_hashes_verified_after"] or summary["cases"] != 1:
            raise ValueError("Full-cap inference did not complete its integrity gate")
        if sha256(directory / "results.jsonl") != summary["result_sha256"]:
            raise ValueError("Full-cap result hash mismatch")
        with (directory / "results.jsonl").open() as stream:
            pair_rows[arm] = json.loads(next(stream))
        protocols[arm] = load(directory / "protocol.json")
    pair = check_pair(pair_rows["original"], pair_rows["edited"], protocols["original"], protocols["edited"])
    source = root / "lm_understand-logs"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("Pinned author repository changed")
    original_seal = load(root / "models" / "QWEN_SOURCE.json")
    for entry in original_seal["files"]:
        if sha256(root / "models" / "qwen-original" / entry["file"]) != entry["sha256"]:
            raise ValueError("Original Qwen changed")
    if sha256(root / "models" / "qwen-p50c.safetensors") != weights["edited_file_sha256"]:
        raise ValueError("Reconstructed published checkpoint changed")
    record = {"source_commit": revision, "article_sha256": sha256(root / "article.html"),
              "network_reconstruction": {key: reconstruction[key] for key in ("all_six_arrays_bit_identical", "parameter_count", "nonzero_count", "statements")},
              "gameplay": game, "qwen_weights": weights, "qwen_operator": operator,
              "teacher_forced": teacher, "recorded_score_merges_exact": score_replay["exactly_recomputed"],
              "recorded_math_regrade": regraded, "new_full_cap_pair": pair,
              "originals_and_reconstructed_artifact_unchanged": True,
              "lfm3_weights_changed_in_this_reproduction": False}
    output = results / "REPRODUCTION.json"
    with output.open("x") as destination:
        json.dump(record, destination, indent=2, default=str)
    likelihood = teacher["likelihood_changes"]
    lines = [
        "# Weight-Space Programming: execution-based reproduction", "",
        "## Sources and boundaries", "",
        f"Article: https://rikiyat.github.io/blog/weight-space-programming/ ; official repository commit `{revision}`.",
        "Cloned and executed on K3 in the isolated Astra worktree/reproduction area. No training, production changes, original-checkpoint writes, or LFM3 weight edits.",
        "The human explicitly approved reproducing the published two-layer Qwen reference. The original interactive Astra discovery process and GPU/vLLM benchmark were not independently rerun.", "",
        "## 1. Execute a weight program", "",
        f"The published Breakout edit block reconstructed all six float32 arrays exactly: {reconstruction['statements']} statements, {reconstruction['parameter_count']:,} parameters, {reconstruction['nonzero_count']} nonzero.",
        "Ten controlled observations reproduced the predicted catch-interval decisions and identical reference/reconstruction logits.",
        f"In 1,000 real MinAtar 1.0.15 episodes: mean {game['measured']['mean']:.2f}, SE {game['measured']['se']:.2f}. Published: {game['published_hand']['mean']:.2f}, SE {game['published_hand']['se']:.2f}.",
        f"The difference is {game['difference_in_combined_standard_errors']:.2f} combined standard errors. Our 100 zero-network controls averaged {game['zero_weight_control']['mean']:.2f}.",
        "The game source AST matches the source given to the author. Seeds/evaluator are independently declared because the original runner was not released. PPO checkpoints were not released; their scores were not reproduced.", "",
        "## 2. Rebuild the exact Qwen intervention", "",
        f"The authors' unchanged apply_edit.py passed both base/edited tensor hash assertions. An independent audit confirmed {weights['unchanged_tensors']}/{weights['tensor_count']} tensors unchanged and exactly {weights['total_changed_entries']:,} changed BF16 entries.",
        "Only the declared 64 output columns of MLP layers 15 and 19 changed. Norms, embeddings, head and all other tensors were untouched.",
        f"The two recovered edit directions have absolute cosine {weights['absolute_cross_layer_direction_cosine']:.6f}: correlated, but not identical.",
        r"The actual operation is $W'=W-6u(u^TW\odot m_{64})$, so selected directional components become approximately $-5$ times their originals, not zero.",
    ]
    for layer, measured in operator["measured"].items():
        lines.append(f"Layer {layer}: fitted alpha {measured['fitted_alpha']:.6f}, directional ratio {measured['post_edit_projection_signed_ratio']:.6f}, relative formula residual {measured['alpha6_formula_relative_error']:.6f}.")
    lines += ["", "## 3. Recover the mechanism from actual activations", "",
              "All eight disclosed matched productive/repetitive texts reproduced the published token lengths. Residual directions use the final36 tokens; the final writer selection uses24 MLP tokens.",
              "My first implementation incorrectly used36 for both. Running it exposed the mismatch, and inspecting the recorded final selection code corrected the interpretation; the first result was preserved."]
    for layer, measured in teacher["directions_and_writers"].items():
        lines.append(f"Layer {layer}: direction cosine {measured['absolute_cosine_to_actual_edit_direction']:.8f}; selected columns {measured['top64_overlap_with_published_selection']}/64 recovered. Reproduction-only {measured['selection_only_in_reproduction']}; published-only {measured['selection_only_in_published']}.")
    lines += ["", "Mean continuation log-likelihood change, nats/token:", "",
              "| Continuation | Published edit | Fixed orthogonal control |", "|---|---:|---:|"]
    for kind in ("productive", "repetitive"):
        change = likelihood[kind]
        lines.append(f"| {kind} | {change['mean_change_nats_per_token']:+.7f} | {change['orthogonal_control_mean_change']:+.7f} |")
    lines += ["", "This supports directional specificity on four disclosed diagnostic pairs, not a held-out population estimate. The one remaining layer19 column mismatch is unresolved, not replaced with the known answer.",
              "", "## 4. Check the released evaluation evidence", "",
              "The authors' merge.py exactly regenerated all four Qwen aggregate JSON records from copied score files. These are recomputed published counts, not newly generated benchmark answers."]
    for arm, measured in regraded["arms"].items():
        lines.append(f"Re-graded {arm}: {measured['released_first_samples_regraded']} released first-sample texts, {measured['computed_correct']} correct, {len(measured['mismatches'])} disagreements with recorded labels.")
    lines += ["Only 200/500 first-sample traces per arm are present (20 per shard). The other answers behind the multi-sample benchmark were not independently re-graded.",
              "", "## 5. New full-cap CPU pair", "",
              f"Problem `{pair['id']}`, exact same tokenized input, thinking enabled, temperature0.6, top-p0.95, top-k20, generated-token presence penalty1.5, seed0, cap{pair['max_new_tokens']:,}.",
              "CPU Transformers/SDPA rather than GPU/vLLM; RNG/backend differences remain. One pair cannot estimate MATH-500 accuracy.", "",
              "| Arm | Correct | Tokens | Cap hit | Thinking closed | Extracted answer |", "|---|---:|---:|---:|---:|---|"]
    for arm in ("original", "edited"):
        row = pair[arm]
        lines.append(f"| {arm} | {row['correct']} | {row['generated_tokens']} | {row['hit_cap']} | {row['thinking_closed']} | {row['prediction']} |")
    lines += ["", "## What the work established", "",
              "The small network contains an executable controller, and the Qwen intervention is an activation-localized, sparse directional rewrite. Statistics alone did not establish either behavior.",
              "The workflow is: establish the failing trajectory, construct matched contexts, measure a direction, attribute which units write it, test the local operation and controls, then evaluate frozen behavior. It is not generic clipping or removal of outliers.",
              "Large weights implement useful gates in the Breakout program. In Qwen, alpha6 actively counters a direction; calling it mere deletion misses the measured operation.",
              "No LFM3 improvement was established by reproducing Qwen. LFM3 needs its own demonstrated localized failure and matched activation evidence; transferring Qwen's indices/directions would be unsupported.",
              "", "## Limits", "",
              "Not reproduced: original GPT-6 Astra discovery, PPO training, full fresh GPU/vLLM MATH-500/GSM8K/IFEval evaluation, every article experiment, or transfer to LFM3. The declared CPU experiments and recorded-output checks are the extent of the evidence.",
              "All original checkpoints and the pinned author repository passed final integrity checks. Full raw records, code, hashes and context are retained beside this report.", ""]
    with (results / "REPRODUCTION.md").open("x") as destination:
        destination.write("\n".join(lines))
    print("REPRODUCTION_REPORT_COMPLETE " + json.dumps({"source_commit": revision, "new_pair": pair}), flush=True)


class ReportTests(unittest.TestCase):
    def test_protocol_difference_rejected(self):
        protocol = {name: 1 for name in ("selected_cases", "max_new_tokens", "thinking", "temperature", "top_p", "top_k",
                                         "presence_penalty", "seed", "batch_size", "backend", "grader_source_sha256", "grader_selected_ast_sha256")}
        changed = dict(protocol, seed=2)
        with self.assertRaisesRegex(ValueError, "seed"):
            check_pair({}, {}, protocol, changed)

    def test_input_difference_rejected(self):
        protocol = {name: 1 for name in ("selected_cases", "max_new_tokens", "thinking", "temperature", "top_p", "top_k",
                                         "presence_penalty", "seed", "batch_size", "backend", "grader_source_sha256", "grader_selected_ast_sha256")}
        with self.assertRaisesRegex(ValueError, "input"):
            check_pair({"id": "a", "input_sha256": "one"}, {"id": "a", "input_sha256": "two"}, protocol, protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["report_reproduction.py"], verbosity=2)
    elif args.root:
        report(args.root)
    else:
        parser.error("--root or --self-test required")