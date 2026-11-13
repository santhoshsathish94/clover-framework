from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from artifacts import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_nemotron(root):
    from nemotron import POLICIES, gate

    manifest = read_json(root / "frozen" / "MANIFEST.json")
    for path, expected in manifest.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    observation = read_json(root / "OBSERVATION.json")
    sources = [observation["copy"]] + read_json(root / "INPUTS.json")["files"]
    for item in sources:
        if sha256(item["source"]) != item["sha256"] or sha256(item["copy"]) != item["sha256"]:
            raise ValueError("Nemotron source or baseline copy changed")
    cases = read_json(root / "frozen" / "cases.json")
    expected_ids = [case["id"] for case in cases if case["split"] == "discovery"]
    baseline = read_json(root / "results" / "discovery-original.json")
    reports = {}
    for name in POLICIES:
        rows = read_json(root / "results" / f"discovery-{name}.json")
        if [row["id"] for row in rows] != expected_ids:
            raise ValueError("Nemotron case identity/coverage mismatch")
        for row in rows:
            if not all(math.isfinite(row[key]) for key in ("ntp", "text_ntp", "cos_std", "cos_nat")):
                raise ValueError("nonfinite Nemotron result")
        reports[name] = {
            "mean_ntp": sum(row["ntp"] for row in rows) / len(rows),
            "ntp_worse_cases": sum(new["ntp"] > old["ntp"] + 1e-6 for new, old in zip(rows, baseline)),
            "text_worse_cases": sum(new["text_ntp"] > old["text_ntp"] + 1e-6 for new, old in zip(rows, baseline)),
            "standard_alignment_worse_cases": sum(new["cos_std"] < old["cos_std"] - 1e-6 for new, old in zip(rows, baseline)),
            "native_alignment_worse_cases": sum(new["cos_nat"] < old["cos_nat"] - 1e-6 for new, old in zip(rows, baseline)),
            "broken_token_cases": sum(bool(set(old["correct_positions"]) - set(new["correct_positions"])) for new, old in zip(rows, baseline)),
            "qualifies": name != "original" and gate(rows, baseline),
        }
        audit = read_json(root / "trials" / name / "PATCH_AUDIT.json")
        reversal = read_json(root / "trials" / name / "REVERSAL.json")
        copied = root / "trials" / name / "checkpoint.pt"
        if sha256(copied) != observation["copy"]["sha256"] or not reversal["restored"]:
            raise ValueError("Nemotron rejected copy was not restored")
        expected_changed = [] if name == "original" else ["hybrid_link.post_ln.weight", "hybrid_link.post_ln.bias"] if name.startswith("standard") else ["nemo_native_hybrid_link.out_scale"]
        actual_changed = [key for key in audit["before"] if audit["before"][key] != audit["after"].get(key)]
        if set(actual_changed) != set(expected_changed) or set(audit["changed_tensors"]) != set(expected_changed):
            raise ValueError("Nemotron changed-tensor audit mismatch")
    outcome = read_json(root / "results" / "OUTCOME.json")
    if any(item["qualifies"] for item in reports.values()) or outcome["selected"] is not None or outcome["stage_gate_passed"]:
        raise ValueError("Nemotron rejection outcome does not match records")
    if list((root / "results").glob("validation-*.json")):
        raise ValueError("validation was opened despite no qualifying candidate")
    return {
        "verified": True, "model": "nemotron", "sources_checked": len(sources),
        "discovery_runs": len(expected_ids) * len(POLICIES), "results": reports,
        "all_trials_restored": True, "validation_opened": False,
        "deployment_eligible": False, "outcome": "reject_all_calibrations",
    }


def audit_kimi(root):
    from kimi import compare_windows

    for path, expected in read_json(root / "frozen" / "MANIFEST.json").items():
        if sha256(path) != expected:
            raise ValueError(f"Kimi frozen source changed: {path}")
    observation = read_json(root / "OBSERVATION.json")
    sources = [observation["checkpoint"], observation["data"]] + read_json(root / "RUNTIME_INPUTS.json")["files"]
    for item in sources:
        if sha256(item["source"]) != item["sha256"] or sha256(item["copy"]) != item["sha256"]:
            raise ValueError("Kimi source or baseline copy changed")
    captured = read_json(root / "capture-calibration" / "MANIFEST.json")
    if len(captured["records"]) != 12:
        raise ValueError("Kimi capture count mismatch")
    for item in captured["records"]:
        if item["routed_layers"] != 92 or sha256(root / "capture-calibration" / f"window-{item['window']:02d}.npz") != item["sha256"]:
            raise ValueError("Kimi native capture coverage/hash mismatch")
    baseline = read_json(root / "results" / "discovery-original.json")
    zero = read_json(root / "results" / "discovery-zero.json")
    reports = {}
    for name in ("original", "gain092", "gain108"):
        rows = read_json(root / "results" / f"discovery-{name}.json")
        if [row["window"] for row in rows] != [0, 1, 2, 3]:
            raise ValueError("Kimi discovery identities differ")
        reports[name] = {
            "mean_mse": sum(row["mse"] for row in rows) / len(rows),
            "mean_cosine": sum(row["cosine"] for row in rows) / len(rows),
            "mse_worse_windows": sum(new["mse"] > old["mse"] + 1e-10 for new, old in zip(rows, baseline)),
            "cosine_worse_windows": sum(new["cosine"] < old["cosine"] - 1e-6 for new, old in zip(rows, baseline)),
            "fails_zero_control_windows": sum(new["mse"] >= control["mse"] for new, control in zip(rows, zero)),
            "qualifies": name != "original" and compare_windows(rows, baseline, zero),
        }
        audit = read_json(root / "trials" / name / "PATCH_AUDIT.json")
        reversal = read_json(root / "trials" / name / "REVERSAL.json")
        if not reversal["restored"] or sha256(root / "trials" / name / "checkpoint.pt") != observation["checkpoint"]["sha256"]:
            raise ValueError("Kimi trial was not restored")
        expected = set() if name == "original" else {"k3_hybridlink.out_scale"}
        changed = {key for key in audit["before"] if audit["before"][key] != audit["after"].get(key)}
        if changed != expected or set(audit["changed_tensors"]) != expected:
            raise ValueError("Kimi trial changed undeclared tensors")
    outcome = read_json(root / "results" / "OUTCOME.json")
    if any(row["qualifies"] for row in reports.values()) or outcome["selected"] is not None or outcome["calibration_gate_passed"]:
        raise ValueError("Kimi rejection does not match measurements")
    if list((root / "results").glob("validation-*.json")):
        raise ValueError("Kimi validation was opened without a candidate")
    return {
        "verified": True, "model": "kimi", "sources_checked": len(sources),
        "captured_windows": len(captured["records"]), "routed_layers_per_window": 92,
        "results": reports, "zero_mean_mse": sum(row["mse"] for row in zero) / len(zero),
        "all_trials_restored": True, "validation_opened": False,
        "deployment_eligible": False, "outcome": "reject_all_calibrations",
        "limit": "native adapter prediction on existing-engine captures; no new task-level quality claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=["nemotron", "kimi"], required=True)
    args = parser.parse_args()
    result = (audit_nemotron if args.model == "nemotron" else audit_kimi)(args.root.resolve(strict=True))
    write_json(args.root / "VERIFICATION.json", result)
    print(json.dumps(result), flush=True)