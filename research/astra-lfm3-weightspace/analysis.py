import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


TARGET = "model.language_model.embedding_norm.weight"
STRENGTHS = [0.05, 0.10, 0.20]


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_rows(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def write_new(path, content):
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(content)


def structural_reports(root):
    root = Path(root)
    inventory = json.loads((root / "reports" / "inventory.json").read_text())
    survey = read_rows(root / "reports" / "weight-survey-v2" / "tensors.jsonl")
    summary = json.loads((root / "reports" / "weight-survey-v2" / "summary.json").read_text())
    checkpoints = {"main": inventory, **inventory["auxiliary_checkpoints"]}
    components = defaultdict(list)
    for row in survey:
        components[row["component"]].append(row)
    architecture = [
        "# Architecture and tensor inventory", "",
        "Observed on K3 from the pinned original checkpoint and its source; not the trained Clover variant.", "",
        f"Revision: `{summary['original_revision']}`.", "",
        "| Component | Stored tensors | Stored elements |", "|---|---:|---:|",
    ]
    for component, rows in sorted(components.items()):
        architecture.append(f"| {component} | {len(rows)} | {sum(row['parameter_count'] for row in rows):,} |")
    architecture += [
        "", "Stored elements include checkpoint buffers; runtime parameter/buffer identities are in benchmark metadata.",
        "Tied aliases are not counted twice. The complete tensor-level shapes, dtypes, offsets and shard hashes are in `inventory.json`.", "",
        "## Source-defined paths", "",
        "- Shared backbone: width 2048, 24 layers: 13 short-convolution, 6 full-attention, 5 sliding-attention. Gated MLPs and sandwich RMS norms.",
        "- Text input/output: one 128000 x 2048 embedding, also used by the tied text head. Runtime tying checked by storage pointer.",
        "- Images and video: the same width-1024, 14-layer vision tower and two-layer projector feed the shared backbone. Video contains timestamped temporal patch groups.",
        "- Audio input: three stride-2 spectrogram convolutions, a width-768, 17-layer encoder, and a separate projector into the shared backbone.",
        "- Speech output: a width-1024, three-layer bidirectional depth decoder predicts 12 codebooks per frame. Each codebook's embedding is also its output head.",
        "- Waveform codec: separate LiquidCodec checkpoint; 24 kHz output, 1920 samples/frame, 12 codebooks. It is included in the inventory and weight survey.",
        "- The final shared RMS gain is after all 24 backbone blocks. Changing it can affect text, multimodal answers, and the speech-depth input.", "",
        "## Checkpoint hashes", "",
    ]
    for name, checkpoint in checkpoints.items():
        for entry in checkpoint["files"]:
            architecture.append(f"- {name}/{entry['file']}: `{entry['sha256']}` ({entry['bytes']:,} bytes).")
    architecture += ["", "All upstream Git/LFS digests were verified during acquisition; the analysis rechecked the sealed original before and after each complete phase.", ""]
    write_new(root / "reports" / "ARCHITECTURE.md", "\n".join(architecture))

    distributions = [
        "# Weight distribution report", "",
        f"Observed: {len(survey)}/{summary['tensor_count']} stored tensors analyzed in {summary['elapsed_s']:.2f} seconds on K3 (8 threads).",
        f"Nonfinite values: {sum(row['nonfinite_count'] for row in survey):,}. Original hashes matched before and after: {summary['source_hashes_verified_before_and_after']}.", "",
        "## Methods and limits", "",
        "Exact float64 accumulated moments over every element; exact row/column L2 norms and counts beyond six tensor-specific standard deviations.",
        "Value quantiles use all elements up to 65536, otherwise 65536 seeded uniform draws with replacement. Histograms use that same sample.",
        "Convolution tensors are flattened with axis 0 as rows and remaining axes as columns. The saved NPZ holds every row/column norm and concentration count.",
        "Full float32 SVD is used when the smaller dimension is <=256 and element count <=8 million. Other spectra use randomized top-16, oversampling 8, two power iterations, seed 20261111.",
        "The truncated spectra are not full-rank estimates. Pair correlations sample 4096 aligned elements within equal-shape/component groups; row-pair checks use top-norm and evenly spaced rows.",
        "Exploratory multiple comparisons are not evidence of defects. No normalization, clipping, rank reduction or automatic correction was performed.", "",
        "| Component | Six-std elements | Matrices surveyed | Median top-16/full reported energy |", "|---|---:|---:|---:|",
    ]
    for component, rows in sorted(components.items()):
        spectral = [row["spectrum"] for row in rows if "spectrum" in row and row["spectrum"].get("cumulative_frobenius_energy")]
        energies = [row["cumulative_frobenius_energy"][-1] for row in spectral]
        outliers = sum(row.get("outliers_gt_6std", {}).get("count", 0) for row in rows)
        median = f"{np.median(energies):.4f}" if energies else "not applicable"
        distributions.append(f"| {component} | {outliers:,} | {len(spectral)} | {median} |")
    distributions += ["", "Energy medians mix full small-matrix and truncated large-matrix reports; compare per-tensor method fields, not this aggregate as a rank claim.",
                      "", "## Final shared RMS gain", ""]
    final_norm = next(row for row in survey if row["name"] == TARGET)
    distributions += [
        f"Observed range {final_norm['min']:.8g} to {final_norm['max']:.8g}; mean {final_norm['mean']:.8g}; std {final_norm['std']:.8g}; excess kurtosis {final_norm['excess_kurtosis']:.5g}.",
        "This heterogeneous learned scaling is not itself a defect. Candidate selection requires behavioural activation evidence and is not based on these extremes.",
        "", "## Highest sampled same-shape correlations", "",
    ]
    for entry in summary["same_shape_correlations_top50"][:5]:
        distributions.append(f"- `{entry['tensor_a']}` versus `{entry['tensor_b']}`: sampled Pearson {entry['sampled_pearson']:.6f}.")
    distributions += ["", "These may reflect shared learned structures. Their functional necessity has not been established by the weight survey.",
                      "Per-tensor evidence: `weight-survey-v2/tensors.jsonl`; full profiles: `weight-survey-v2/norms_and_concentrations.npz`.", ""]
    write_new(root / "reports" / "WEIGHTS.md", "\n".join(distributions))


def ranked_channels(contributions, failed):
    failed = np.asarray(failed, dtype=bool)
    if not failed.any() or failed.all():
        raise ValueError("Both failed and successful discovery examples are required")
    failed_mean = contributions[failed].mean(0)
    success_mean = contributions[~failed].mean(0)
    benefit = -failed_mean
    score = benefit - 2 * np.maximum(success_mean, 0)
    eligible = np.flatnonzero(benefit > 0)
    ranked = sorted(eligible.tolist(), key=lambda index: (-score[index], index))
    return ranked, score, failed_mean, success_mean


def auc_pairwise(scores, failed):
    positives, negatives = scores[failed], scores[~failed]
    differences = positives[:, None] - negatives[None, :]
    return float(np.mean(differences > 0) + 0.5 * np.mean(differences == 0))


def contrast(values, failed):
    failures, successes = values[failed], values[~failed]
    direction = failures.mean(0) - successes.mean(0)
    pooled = np.sqrt((failures.var(0) + successes.var(0)) / 2 + 1e-12)
    standardized = direction / pooled
    scores = np.zeros(len(values))
    for index in range(len(values)):
        retained = np.arange(len(values)) != index
        kept_values, kept_failed = values[retained], failed[retained]
        if not kept_failed.any() or kept_failed.all():
            return direction, standardized, None
        held_direction = kept_values[kept_failed].mean(0) - kept_values[~kept_failed].mean(0)
        held_direction /= max(np.linalg.norm(held_direction), 1e-12)
        scores[index] = values[index] @ held_direction
    return direction, standardized, auc_pairwise(scores, failed)


def activation_report(root, baseline="baseline"):
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from benchmark import verify_seal
    from survey import verify_original

    root = Path(root)
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    verify_original(root)
    verify_seal(root)
    result_dir = root / "reports" / baseline
    summary = json.loads((result_dir / "summary.json").read_text())
    if summary["error_count"] or summary["case_count"] != 112:
        raise RuntimeError("A complete error-free frozen baseline is required")
    cases = {case["id"]: case for case in json.loads((root / "fixtures" / "cases.json").read_text())
             if case["split"] == "discovery"}
    rows = [row for row in read_rows(result_dir / "results.jsonl") if row["id"] in cases]
    if len(rows) != 32:
        raise RuntimeError("Expected exactly the 32 discovery examples")
    failed = np.array([not row["correct"] for row in rows], dtype=bool)
    if not failed.any() or failed.all():
        write_new(root / "reports" / "HYPOTHESES.md", "# No eligible intervention\n\nDiscovery has no usable success/failure contrast. No tensor edit is proposed.\n")
        return
    archive = np.load(result_dir / "activations.npz", allow_pickle=False)
    direction_arrays = {}
    layer_results = []
    for key in ["layer_0", "layer_2", "layer_4", "layer_11", "layer_17", "layer_23", "final_hidden"]:
        values = np.stack([archive[f"{row['id']}::{key}"] for row in rows]).astype(np.float64)
        direction, standardized, auc = contrast(values, failed)
        direction_arrays[key + ".contrast"] = direction
        direction_arrays[key + ".standardized"] = standardized
        layer_results.append({"layer": key, "contrast_norm": float(np.linalg.norm(direction)),
                              "max_absolute_standardized_contrast": float(np.max(np.abs(standardized))),
                              "leave_one_out_auc": auc})
    final_hidden = np.stack([archive[f"{row['id']}::final_hidden"] for row in rows]).astype(np.float64)
    tokenizer = Tokenizer.from_file(str(root / "original" / "tokenizer.json"))
    letter_ids = [tokenizer.token_to_id(letter) for letter in "ABCD"]
    with safe_open(root / "original" / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        embeddings = checkpoint.get_tensor("model.language_model.embed_tokens.weight")[letter_ids].float().numpy().astype(np.float64)
        gains = checkpoint.get_tensor(TARGET)
    contributions = np.stack([
        (embeddings["ABCD".index(row["gold"])] - embeddings["ABCD".index(row["competitor"])] ) * hidden
        for row, hidden in zip(rows, final_hidden)
    ])
    ranked, scores, failed_mean, success_mean = ranked_channels(contributions, failed)
    if not ranked:
        write_new(root / "reports" / "HYPOTHESES.md", "# No eligible intervention\n\nNo coordinate has a predicted positive failed-example margin effect. No tensor edit is proposed.\n")
        return
    direction_arrays["discovery_margin_contributions"] = contributions
    direction_arrays["failed_mask"] = failed
    np.savez_compressed(root / "reports" / "discovery-directions.npz", **direction_arrays)
    confounds = {}
    for field in ("family", "gold_letter"):
        groups = defaultdict(lambda: {"success": 0, "failure": 0})
        for row in rows:
            groups[str(cases[row["id"]][field])]["success" if row["correct"] else "failure"] += 1
        confounds[field] = dict(groups)
    observed_margin = np.array([row["gold_margin"] for row in rows])
    identity_error = contributions.sum(1) - observed_margin
    report = {
        "baseline_results_sha256": sha256(result_dir / "results.jsonl"),
        "baseline_activations_sha256": sha256(result_dir / "activations.npz"),
        "selection_source_sha256": sha256(root / "analysis.py"),
        "discovery_ids": [row["id"] for row in rows],
        "success_count": int((~failed).sum()), "failure_count": int(failed.sum()),
        "validation_used_for_selection": False, "layers": layer_results, "confounds": confounds,
        "float64_margin_vs_bf16_logged_max_abs_error": float(np.max(np.abs(identity_error))),
        "ranked_channels": [{"index": index, "score": float(scores[index]),
                              "failure_mean_margin_contribution": float(failed_mean[index]),
                              "success_mean_margin_contribution": float(success_mean[index]),
                              "original_gain": float(gains[index])} for index in ranked[:10]],
    }
    write_new(root / "reports" / "activation-analysis.json", json.dumps(report, indent=2))
    text = ["# Activation and direction analysis", "",
            f"Discovery: {report['success_count']}/32 correct, {report['failure_count']}/32 incorrect. Validation examples excluded from selection.", "",
            "Observed association, not a causal explanation. Final-token vectors are compared within this small constructed arithmetic sample.", "",
            "| Location | Failure-minus-success contrast L2 | Leave-one-out centroid AUC |", "|---|---:|---:|"]
    for entry in layer_results:
        auc = "unavailable" if entry["leave_one_out_auc"] is None else f"{entry['leave_one_out_auc']:.4f}"
        text.append(f"| {entry['layer']} | {entry['contrast_norm']:.6g} | {auc} |")
    text += ["", "AUC is cross-fitted inside discovery, not independent validation. Directions and standardized coordinate contrasts are in `discovery-directions.npz`.",
             f"Float64 contribution-sum versus logged BF16 margin maximum absolute difference: {report['float64_margin_vs_bf16_logged_max_abs_error']:.6g}.",
             "Thus the real-arithmetic sensitivity must not be represented as an exact prediction of rounded model logits.",
             "", "## Cohort confounds", "", "```json", json.dumps(confounds, indent=2), "```", "",
             "Operation, answer label, sequence length and prompt content can explain apparent directions. This analysis does not prove a defect or a general failure representation.",
             "Image/audio/video/speech activations are captured in the baseline archive but are not used to fit this text-derived intervention. Their small samples do not establish modality-specific failure subspaces.", ""]
    write_new(root / "reports" / "ACTIVATIONS.md", "\n".join(text))

    hypotheses = ["# Ranked intervention hypotheses", "", "Status: mathematical hypotheses supported by discovery associations; not yet experimentally verified improvements.",
                  "", f"Target tensor: `{TARGET}` (final shared RMS gain, after backbone layer 23 and before the tied text head).",
                  "", "## Derivation", "",
                  r"For pre-norm state $r$, $z=r/\sqrt{\operatorname{mean}(r^2)+\epsilon}$, gain $g$, hidden $h=g\odot z$, and logits $\ell=Eh$.",
                  r"Changing one gain $g_j'=(1-\alpha)g_j$ gives $\Delta\ell_y=-\alpha E_{y,j}h_j$ in real arithmetic.",
                  r"For gold $a$ and the baseline strongest incorrect letter $b$, $\Delta(\ell_a-\ell_b)=-\alpha(E_{a,j}-E_{b,j})h_j=-\alpha c_j$.",
                  "BF16 gain multiplication/head rounding, competitor switching and autoregressive continuation can invalidate that local prediction.",
                  "", "Shared procedure: 5%, 10% (predeclared primary), 20% attenuation; one coordinate in one tensor per fresh copied checkpoint. No gradients or other edits.",
                  "Risk: lower margins for correct cases, label bias, worse unrelated text/media reasoning, ASR or speech-depth conditioning. The selected coordinate is not assumed to be a defective outlier.",
                  "Reversal: restore the recorded original two BF16 bytes in the copy, then require the entire checkpoint SHA-256 to equal the sealed original. Never reverse by approximate inverse scaling.", ""]
    for rank, index in enumerate(ranked[:3], 1):
        hypotheses += [f"## H{rank}: coordinate {index}", "",
                       f"Exact flat index {index}; original gain {float(gains[index]):.9g}; selection score {scores[index]:.6g}.",
                       f"Observed mean contribution: failures {failed_mean[index]:.6g}, successes {success_mean[index]:.6g}.",
                       f"Predicted 10% mean margin change: failures {-0.1 * failed_mean[index]:+.6g}, successes {-0.1 * success_mean[index]:+.6g}.",
                       "H1 is the sole execution candidate. H2/H3 are lower-ranked alternatives and will not be tried after seeing validation results." if rank > 1 else "Selected by the preregistered discovery-only rule, not by weight extremeness or validation outcomes.", ""]
    hypotheses += ["## Supporting failures", ""]
    for row in [row for row in rows if not row["correct"]][:6]:
        hypotheses.append(f"- `{row['id']}`: generated `{row['text']}`; correct label `{row['gold']}`; first-token margin {row['gold_margin']:.6g}.")
    hypotheses += ["", "These are exploratory comparisons across 2048 coordinates. No significance or general improvement is inferred from choosing a maximum. The held-out paired experiment is the causal test.", ""]
    hypothesis_path = root / "reports" / "HYPOTHESES.md"
    write_new(hypothesis_path, "\n".join(hypotheses))
    selected = ranked[0]
    inventory = json.loads((root / "reports" / "inventory.json").read_text())
    target = next(entry for entry in inventory["tensors"] if entry["name"] == TARGET)
    original_bytes = gains[selected:selected + 1].view(torch.uint8).numpy().tobytes()
    patch_dir = root / "reports" / "patches"
    patch_dir.mkdir(exist_ok=False)
    for strength in STRENGTHS:
        new_value = (gains[selected].float() * (1 - strength)).to(gains.dtype)
        new_bytes = new_value.reshape(1).view(torch.uint8).numpy().tobytes()
        spec = {
            "schema_version": 1, "hypothesis_id": "H1",
            "hypothesis_file": "reports/HYPOTHESES.md", "hypothesis_sha256": sha256(hypothesis_path),
            "baseline_results_sha256": report["baseline_results_sha256"],
            "original_checkpoint_sha256": inventory["files"][0]["sha256"],
            "target_tensor": TARGET, "target_shape": target["shape"], "dtype": target["dtype"],
            "flat_index": selected, "file_offset": target["file_offsets"][0] + selected * 2,
            "old_hex": original_bytes.hex(), "new_hex": new_bytes.hex(),
            "original_gain": float(gains[selected]), "new_gain": float(new_value),
            "requested_attenuation": strength,
            "effective_attenuation": 1 - float(new_value) / float(gains[selected]),
            "primary": strength == 0.10,
        }
        write_new(patch_dir / f"H1-alpha-{int(strength * 100):02d}.json", json.dumps(spec, indent=2))
    verify_original(root)
    print("HYPOTHESIS_DOCUMENTED " + json.dumps(report["ranked_channels"][0]), flush=True)


class AnalysisTests(unittest.TestCase):
    def test_preregistered_ranking(self):
        contributions = np.array([[-4, 2, 1], [-2, 2, 1], [1, 0.1, -1], [1, 0.1, -1]])
        indices, scores, failed_mean, success_mean = ranked_channels(contributions, [True, True, False, False])
        self.assertEqual(indices, [0])
        self.assertAlmostEqual(scores[0], 1)
        self.assertAlmostEqual(-0.1 * failed_mean[0], 0.3)
        self.assertAlmostEqual(-0.1 * success_mean[0], -0.1)

    def test_no_comparison_group_refused(self):
        with self.assertRaises(ValueError):
            ranked_channels(np.ones((3, 4)), [True, True, True])

    def test_auc_liveness(self):
        labels = np.array([False, False, True, True])
        self.assertEqual(auc_pairwise(np.array([0, 1, 2, 3]), labels), 1)
        self.assertEqual(auc_pairwise(np.array([3, 2, 1, 0]), labels), 0)
        self.assertEqual(auc_pairwise(np.ones(4), labels), 0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--weights", action="store_true")
    parser.add_argument("--activations", action="store_true")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["analysis.py"], verbosity=2)
    elif not args.root:
        parser.error("--root required")
    elif args.weights:
        structural_reports(args.root)
    elif args.activations:
        activation_report(args.root, args.baseline)
    else:
        parser.error("--weights or --activations required")