"""`core_sync.py` — Client-Side Core Synchronizer & Layered Collective Assistant.

Responsibilities (`whats-needed-for-clover.txt` §§1, 5, 6):
1. Consent Check (`consent_receive`): Downloads and applies collective core releases ONLY when `consent_receive` is True.
2. Cryptographic Verification: Verifies `releases/manifest.json` HMAC-SHA256 signature and per-file SHA-256 hashes
   before loading any collective rule. Rejects tampered manifests or modified rule files.
3. Strict Priority Hierarchy in `CollectiveAssistant`:
     1. Validated Personal Understanding (`self.learner`, PPL) — always takes precedence for that individual user!
     2. Signed Collective Core (`self.collective_rules`, CPSL) — applies general problem-solving strategies and semantic
        improvements when no personal override matches.
     3. Frozen General Interpreter (`self.general`, B0 + CSI v0.1r3) — immutable baseline fallback.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

from ppl.assistant import Assistant
from ppl.learner import trigger_features


class CoreSyncError(RuntimeError):
    pass


class CoreSync:
    """Downloads, verifies, and applies signed releases from `clover-collective`."""

    def __init__(self, signing_key: str = "clover-collective-core-signing-key-v1"):
        self.signing_key = signing_key.encode("utf-8")

    def verify_and_load(self, registry_root: str | Path, consent_receive: bool = True) -> list[dict]:
        """Verify `releases/manifest.json` signature and file SHA-256 digests. Return `[]` if `consent_receive` is False."""
        if not consent_receive:
            return []
        root = Path(registry_root)
        manifest_path = root / "releases" / "manifest.json"
        if not manifest_path.exists():
            raise CoreSyncError(f"Missing manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = manifest.get("version")
        rules = manifest.get("rules", [])
        file_hashes = manifest.get("file_hashes", {})
        sig = manifest.get("signature", "")

        payload = json.dumps(
            {"version": version, "rules": rules, "file_hashes": file_hashes},
            sort_keys=True,
        ).encode("utf-8")
        expected_sig = hmac.new(self.signing_key, payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            raise CoreSyncError("Release signature verification failed (tampered manifest)")

        for rel_path, expected_sha in file_hashes.items():
            fpath = root / rel_path
            if not fpath.exists():
                raise CoreSyncError(f"Missing core file {rel_path}")
            actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                raise CoreSyncError(f"File SHA-256 mismatch for {rel_path}")

        return list(rules)


class CollectiveAssistant(Assistant):
    """Clover Assistant combining Personal Process Learning (PPL), Signed Collective Core (CPSL), and Frozen General Core (B0)."""

    def __init__(self, arm: str, general, reader, params: dict, collective_rules: list[dict] | None = None,
                 store=None, personal_csi: bool = True):
        base_arm = "PPL" if arm in ("C1", "C2", "C3", "CPSL", "CPSL-NoCSI") else "B0"
        super().__init__(base_arm, general, reader, params, store=store)
        if arm == "C0":
            self.learner = None
        if not personal_csi:
            self.personal_csi = False
        self.collective_rules = []
        for r in collective_rules or []:
            targ = r["target"]
            target_val = tuple(targ) if isinstance(targ, (list, tuple)) and len(targ) > 1 else (
                targ[0] if isinstance(targ, (list, tuple)) else targ
            )
            self.collective_rules.append({
                "rule_id": r["rule_id"],
                "trigger": frozenset(r["trigger"]),
                "target": target_val,
                "independent_installations": r.get("independent_installations", 1),
                "heldout_accuracy": r.get("heldout_accuracy", 1.0),
            })

    def match_collective(self, feats: frozenset[str]):
        tf = trigger_features(feats, use_ctx=True)
        matches = [r for r in self.collective_rules if r["trigger"] <= tf]
        if not matches:
            return None
        return max(
            matches,
            key=lambda r: (len(r["trigger"]), r["heldout_accuracy"], r["independent_installations"], r["rule_id"]),
        )

    def respond(self, text: str, context: str):
        a = self.analyse(text, context)
        top, conf = a["scores"][0]
        # 1. Personal understanding (PPL) has highest priority for this person
        if self.learner:
            h = self.learner.interpret(a["p"])
            if h is not None:
                return self.act(h.target, "personal", h.id), a
        # 2. Signed Collective Core (CPSL) applies validated collective strategies
        c_rule = self.match_collective(a["p"])
        if c_rule is not None:
            return self.act(c_rule["target"], "collective", c_rule["rule_id"]), a
        # 3. Frozen General Interpreter (B0)
        if conf < self.C:
            personal = []
            if self.learner:
                personal = [
                    h.target
                    for h in sorted(self.learner.candidates(a["p"]), key=lambda h: (-len(h.support), h.id))
                ]
            return self.clarify(personal + [l for l, _ in a["scores"][:3]]), a
        return self.act(top, "general"), a
