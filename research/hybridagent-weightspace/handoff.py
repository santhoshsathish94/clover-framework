from __future__ import annotations

import argparse
import json
import re
import unittest
from pathlib import Path

from artifacts import sha256


def validate_manifest(manifest):
    if manifest.get("schema") != 1 or manifest.get("execution_order") != ["nemotron", "kimi"]:
        raise ValueError("unexpected handoff schema or model order")
    if set(manifest.get("models", {})) != {"nemotron", "kimi"}:
        raise ValueError("handoff must name both models")
    for name, record in manifest["models"].items():
        if not re.fullmatch("[0-9a-f]{64}", record["baseline_sha256"]):
            raise ValueError(f"invalid baseline digest: {name}")
        if record["outcome"] == "no_accepted_candidate" and (record["candidate"] is not None or record["deployment_eligible"]):
            raise ValueError("rejected study cannot advertise an accepted candidate")
    return manifest


def verify(manifest):
    validate_manifest(manifest)
    results = {}
    for name, record in manifest["models"].items():
        path = Path(record["baseline"])
        if not path.is_file():
            raise FileNotFoundError(f"Run this check on {manifest['ssh_host']}; missing baseline: {path}")
        if sha256(path) != record["baseline_sha256"]:
            raise ValueError(f"baseline checksum mismatch: {name}")
        evidence = json.loads(Path(record["evidence"]).read_text(encoding="utf-8"))
        if not evidence["verified"] or not evidence["all_trials_restored"]:
            raise ValueError(f"incomplete audit: {name}")
        results[name] = {"baseline_verified": True, "accepted_candidate": record["candidate"], "deployment_eligible": record["deployment_eligible"]}
    return results


class HandoffTests(unittest.TestCase):
    def manifest(self):
        record = {"baseline_sha256": "a" * 64, "outcome": "no_accepted_candidate", "candidate": None, "deployment_eligible": False}
        return {"schema": 1, "execution_order": ["nemotron", "kimi"], "models": {"nemotron": dict(record), "kimi": dict(record)}}

    def test_rejected_handoff_is_valid(self):
        validate_manifest(self.manifest())

    def test_false_promotion_rejected(self):
        manifest = self.manifest()
        manifest["models"]["kimi"]["candidate"] = "restored-original.pt"
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_invalid_digest_rejected(self):
        manifest = self.manifest()
        manifest["models"]["nemotron"]["baseline_sha256"] = "unknown"
        with self.assertRaises(ValueError):
            validate_manifest(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("HANDOFF.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    else:
        print(json.dumps(verify(json.loads(args.manifest.read_text(encoding="utf-8")))), flush=True)