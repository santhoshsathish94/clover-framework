"""Collective Problem-Solving Learning — Client Core Synchronizer & Runtime Hook (`core_sync.py`).

Manages:
1. Two independent consent controls (`consent_contribute`, `consent_receive`), persisted locally in
   `<collective_dir>/consent.json` (both default to `False` — strictly opt-in).
2. Cryptographic verification of signed `clover-collective` releases (`releases/manifest.json` HMAC-SHA256 signature +
   per-file SHA-256 digests).
3. Runtime interpretation (`Collective.interpret(text, frame)`), hooked into `semantics.Runtime.collective` and called
   ONLY when no validated personal habit overrides the message (`Personal > Collective > General`).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import threading
from pathlib import Path

import collective as C_EXT
import personal
import semantics


class CoreSyncError(RuntimeError):
    pass


class CoreSync:
    """Verifies and loads signed releases from a `clover-collective` release directory."""

    def __init__(self, signing_key: str = "clover-collective-core-signing-key-v1"):
        self.signing_key = signing_key.encode("utf-8")

    def verify_and_load(self, registry_root: str | Path, consent_receive: bool = True) -> dict:
        if not consent_receive:
            return {"version": None, "rules": []}
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

        return {"version": version, "rules": list(rules)}


class Collective:
    """One installation's collective learning state: consent switches, verified shared core rules, and candidate preview."""

    def __init__(
        self,
        root: str | Path,
        device: str = "",
        csi=None,
        signing_key: str = "clover-collective-core-signing-key-v1",
    ):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.csi = csi
        self.syncer = CoreSync(signing_key=signing_key)
        self.lock = threading.RLock()
        self.version: str | None = None
        self.rules: list[dict] = []
        self.normalize_gerunds: bool = True
        self._load_state()

    @property
    def _consent_path(self) -> Path:
        return self.dir / "consent.json"

    @property
    def _applied_path(self) -> Path:
        return self.dir / "applied_release.json"

    def _load_state(self):
        if self._consent_path.exists():
            try:
                data = json.loads(self._consent_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}
        self.consent_contribute = bool(data.get("consent_contribute", False))
        self.consent_receive = bool(data.get("consent_receive", False))
        self.installation_id = str(data.get("installation_id") or os.urandom(8).hex())
        self._save_consent()

        if self.consent_receive and self._applied_path.exists():
            try:
                applied = json.loads(self._applied_path.read_text(encoding="utf-8"))
                self.version = applied.get("version")
                self.rules = list(applied.get("rules") or [])
            except Exception:
                self.version, self.rules = None, []

    def _save_consent(self):
        self._consent_path.write_text(
            json.dumps(
                {
                    "installation_id": self.installation_id,
                    "consent_contribute": self.consent_contribute,
                    "consent_receive": self.consent_receive,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def set_consent(self, contribute: bool | None = None, receive: bool | None = None) -> dict:
        with self.lock:
            if contribute is not None:
                self.consent_contribute = bool(contribute)
            if receive is not None:
                self.consent_receive = bool(receive)
                if not self.consent_receive:
                    self.rules = []
                    self.version = None
                elif self._applied_path.exists():
                    applied = json.loads(self._applied_path.read_text(encoding="utf-8"))
                    self.version = applied.get("version")
                    self.rules = list(applied.get("rules") or [])
            self._save_consent()
            return {"consent_contribute": self.consent_contribute, "consent_receive": self.consent_receive}

    def sync_from_registry(self, registry_root: str | Path) -> dict:
        """Verify and apply a signed release from `registry_root` when `consent_receive` is True."""
        with self.lock:
            res = self.syncer.verify_and_load(registry_root, consent_receive=self.consent_receive)
            self.version = res["version"]
            self.rules = res["rules"]
            if self.consent_receive and self.version is not None:
                self._applied_path.write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
            return {"synced": self.consent_receive, "version": self.version, "rules_count": len(self.rules)}

    def preview_candidates(self, personal_obj, extraction_mode: str = "mismatch_only") -> list[dict]:
        with self.lock:
            if personal_obj is None:
                return []
            ext = C_EXT.LocalCollectiveExtractor(
                installation_id=self.installation_id,
                consent_contribute=self.consent_contribute,
                extraction_mode=extraction_mode,
            )
            return ext.preview_candidates(personal_obj)

    def export_candidates(self, personal_obj, extraction_mode: str = "mismatch_only") -> list[dict]:
        with self.lock:
            if not self.consent_contribute or personal_obj is None:
                return []
            ext = C_EXT.LocalCollectiveExtractor(
                installation_id=self.installation_id,
                consent_contribute=True,
                extraction_mode=extraction_mode,
            )
            return ext.export_candidates(personal_obj)

    def _features(self, text: str, analysis=None) -> frozenset[str]:
        if analysis is None and self.csi is not None:
            try:
                analysis = self.csi(text)
            except Exception:
                analysis = None
        raw = personal.features(text, analysis, self.device)
        return C_EXT.normalize_gerund_features(raw) if self.normalize_gerunds else raw

    def interpret(self, text: str, frame) -> dict | None:
        """Called by `semantics._collective(frame, runtime)` when no personal hypothesis overrides the message."""
        with self.lock:
            if not self.consent_receive or not self.rules:
                return None
            feats = self._features(text, analysis=getattr(frame, "_csi", None))
            tf = personal.trigger_features(feats, use_ctx=True)
            matches = []
            for r in self.rules:
                trig = frozenset(r["trigger"])
                trig_no_ctx = frozenset(f for f in trig if not f.startswith("ctx="))
                r_ctxs = {f for f in trig if f.startswith("ctx=")}
                f_ctxs = {f for f in tf if f.startswith("ctx=")}
                if trig <= tf or (not r_ctxs and trig_no_ctx and trig_no_ctx <= tf) or (
                    bool(r_ctxs & f_ctxs) and trig_no_ctx <= tf
                ):
                    matches.append(r)
            if not matches:
                return None
            best = max(
                matches,
                key=lambda r: (
                    len(r["trigger"]),
                    float(r.get("heldout_accuracy", 1.0)),
                    int(r.get("independent_installations", 1)),
                    r["rule_id"],
                ),
            )
            target_caps = list(best.get("target") or [])
            if sorted(target_caps) == sorted(frame.caps) and not best.get("strategy"):
                return None
            out = {
                "rule_id": best["rule_id"],
                "version": best.get("version") or self.version,
                "category": best.get("category", "semantics"),
                "caps": target_caps,
                "independent_installations": int(best.get("independent_installations", 1)),
            }
            if best.get("strategy"):
                out["strategy"] = best["strategy"]
            return out

    def view(self, personal_obj=None) -> dict:
        with self.lock:
            return {
                "enabled": True,
                "store": str(self.dir),
                "consent_contribute": self.consent_contribute,
                "consent_receive": self.consent_receive,
                "version": self.version,
                "active_rules_count": len(self.rules),
                "rules": list(self.rules),
                "candidate_preview": self.preview_candidates(personal_obj) if personal_obj is not None else [],
            }


_PROCESS_COLL = {}
_PROCESS_COLL_LOCK = threading.Lock()
KEEP = object()


def for_process(root, device: str = "", csi=KEEP):
    if not root or os.environ.get("CLOVER_COLLECTIVE", "").strip().lower() in ("0", "off", "false", "no"):
        return None
    key = os.path.abspath(root)
    with _PROCESS_COLL_LOCK:
        if key not in _PROCESS_COLL:
            try:
                _PROCESS_COLL[key] = Collective(root, device=device, csi=None if csi is KEEP else csi)
            except (OSError, ValueError, RuntimeError) as ex:
                print(f"[Clover] collective learning stays off: {ex}", file=sys.stderr, flush=True)
                return None
        elif csi is not KEEP:
            _PROCESS_COLL[key].csi = csi
        return _PROCESS_COLL[key]
