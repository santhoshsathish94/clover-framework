"""Collective Problem-Solving Learning — Collective Learning Gateway (`collective_gateway.py`).

Receives, validates, privacy-filters, and aggregates candidate knowledge packages submitted by opted-in Clover
installations:
1. Schema & Consent Verification (`LEARNING_EVENT_SCHEMA`).
2. Defense-in-Depth Closed-Vocabulary & PII Filtering (`collective.is_allowed_collective_feature`).
3. Installation-Level Deduplication (Sybil Defense): Multiple reports of the same `(trigger, target)` from the same
   installation ID count as ONE independent installation vote (`independent_installations`).
4. Cross-Installation Conflict Tracking: Records when different installations assign different targets to the same
   trigger (`cross_user_agreement`), allowing `collective_evaluator.py` to reject contested personal idioms.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections import defaultdict
from pathlib import Path

import google_identity
import personal
import semantics
from collective import SCHEMA_VERSION, is_allowed_collective_feature, load_closed_general_vocabulary

LEARNING_EVENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "CloverCollectiveLearningEvent",
    "type": "object",
    "required": [
        "package_id",
        "schema_version",
        "installation_id",
        "category",
        "trigger",
        "target",
        "local_support_count",
        "local_contra_count",
        "local_precision",
        "cross_context_generalised",
    ],
    "optional": ["strategy"],
    "additionalProperties": False,
}
ALLOWED_CATEGORIES = {"semantics", "reasoning-patterns", "problem-solving", "tool-use-patterns"}

# Fields that must NEVER be accepted by the gateway at either envelope or package level
FORBIDDEN_ENVELOPE_FIELDS = frozenset({
    "messages", "history", "conversation", "conversations", "memories", "memory_store",
    "personal_memory", "personal_store", "experiences", "raw_text", "prompt", "prompts",
    "email", "password", "access_token", "refresh_token",
})


class GatewaySecurityError(ValueError):
    """Raised when a gateway request violates identity, consent, replay, rate-limit, or privacy rules."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class CollectiveGateway:
    """Ingests, validates, deduplicates, and aggregates candidate packages across Clover installations, with
    authenticated Google OIDC identity validation, pseudonymous contributor IDs, consent records, replay protection,
    and contribution withdrawal."""

    def __init__(
        self,
        oewn_path: str | Path | None = None,
        allowed_targets: set[str] | None = None,
        state_dir: str | Path | None = None,
        gateway_secret: str | bytes | None = None,
        rate_limit_per_min: int = 30,
        replay_window_s: int = 300,
    ):
        self.closed_vocab = load_closed_general_vocabulary(oewn_path)
        self.allowed_targets = set(allowed_targets) if allowed_targets is not None else set(semantics.CAP_WORDS)
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self.lock = threading.RLock()
        self.rate_limit_per_min = int(rate_limit_per_min)
        self.replay_window_s = int(replay_window_s)
        self._gateway_secret = self._init_secret(gateway_secret)
        self.accepted_packages: dict[tuple[str, tuple[str, ...], tuple[str, ...]], dict] = {}
        self.rejected_packages: list[dict] = []
        self.consent_records: dict[str, dict] = {}
        self._seen_nonces: dict[str, float] = {}
        self._rate_buckets: dict[str, list[float]] = defaultdict(list)
        self._load_state()

    def _init_secret(self, gateway_secret: str | bytes | None) -> bytes:
        if gateway_secret is not None:
            return gateway_secret.encode("utf-8") if isinstance(gateway_secret, str) else bytes(gateway_secret)
        env_sec = os.environ.get("CLOVER_GATEWAY_SECRET", "").strip()
        if env_sec and len(env_sec) >= 16:
            return env_sec.encode("utf-8")
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            sec_file = self.state_dir / ".gateway_hmac_secret"
            if sec_file.is_file():
                raw = sec_file.read_bytes().strip()
                if len(raw) >= 32:
                    return raw
            raw = secrets.token_bytes(32)
            sec_file.write_bytes(raw)
            try:
                os.chmod(sec_file, 0o600)
            except OSError:
                pass
            return raw
        return hashlib.sha256(b"clover-default-ephemeral-gateway-secret-v1").digest()

    def _load_state(self):
        if self.state_dir is None:
            return
        state_file = self.state_dir / "gateway_state.json"
        if not state_file.is_file():
            return
        try:
            doc = json.loads(state_file.read_text(encoding="utf-8"))
            self.consent_records = dict(doc.get("consent_records") or {})
            for item in doc.get("accepted_packages") or []:
                inst_id = item["installation_id"]
                trig_key = tuple(sorted(item["trigger"]))
                targ_key = tuple(item["target"])
                self.accepted_packages[(inst_id, trig_key, targ_key)] = item
        except Exception:
            pass

    def _save_state(self):
        if self.state_dir is None:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state_file = self.state_dir / "gateway_state.json"
        tmp = state_file.with_suffix(".tmp")
        doc = {
            "consent_records": self.consent_records,
            "accepted_packages": list(self.accepted_packages.values()),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, state_file)

    def authenticate_contributor(
        self,
        id_token: str,
        *,
        allowed_audiences: list[str] | set[str] | None = None,
        jwks: list[dict] | None = None,
        now: float | None = None,
    ) -> str:
        """Validate a Google OIDC `id_token` and return the pseudonymous `contributor_id` (`contrib_<hmac>`).
        Never returns or persists the raw Google `sub` or email."""
        try:
            claims = google_identity.verify_google_id_token(
                id_token,
                allowed_audiences=allowed_audiences,
                jwks=jwks,
                now=now,
            )
        except google_identity.GoogleIdentityError as exc:
            raise GatewaySecurityError("invalid_google_token", str(exc), status=401) from exc
        return google_identity.derive_pseudonymous_contributor_id(claims["sub"], self._gateway_secret)

    def _check_replay_and_rate(self, contributor_id: str, nonce: str | None, timestamp: float | int | None, now: float | None = None):
        now_ts = float(now if now is not None else time.time())
        with self.lock:
            # Prune expired nonces
            expired = [n for n, ts in self._seen_nonces.items() if now_ts - ts > self.replay_window_s * 2]
            for n in expired:
                self._seen_nonces.pop(n, None)

            # Rate limiting per contributor
            bucket = [t for t in self._rate_buckets[contributor_id] if now_ts - t < 60.0]
            if len(bucket) >= self.rate_limit_per_min:
                self._rate_buckets[contributor_id] = bucket
                raise GatewaySecurityError("rate_limit_exceeded", "Too many gateway requests; slow down.", status=429)
            bucket.append(now_ts)
            self._rate_buckets[contributor_id] = bucket

            if timestamp is not None:
                try:
                    req_ts = float(timestamp)
                except (TypeError, ValueError) as exc:
                    raise GatewaySecurityError("invalid_timestamp", "Invalid request timestamp", status=400) from exc
                if abs(now_ts - req_ts) > self.replay_window_s:
                    raise GatewaySecurityError("stale_request", "Request timestamp outside replay window", status=400)

            if nonce is not None:
                nonce_str = str(nonce).strip()
                if not (8 <= len(nonce_str) <= 128):
                    raise GatewaySecurityError("invalid_nonce", "Request nonce must be 8..128 chars", status=400)
                scoped_nonce = f"{contributor_id}:{nonce_str}"
                if scoped_nonce in self._seen_nonces:
                    raise GatewaySecurityError("replay_detected", "Duplicate/replayed request nonce rejected", status=409)
                self._seen_nonces[scoped_nonce] = now_ts

    def submit_authenticated_contribution(
        self,
        *,
        id_token: str,
        payload: dict,
        allowed_audiences: list[str] | set[str] | None = None,
        jwks: list[dict] | None = None,
        now: float | None = None,
    ) -> dict:
        """Validate Google OIDC token, verify explicit user confirmation, enforce replay/rate/privacy guards,
        substitute pseudonymous `contributor_id`, and ingest approved candidate packages."""
        if not isinstance(payload, dict):
            raise GatewaySecurityError("invalid_payload", "JSON object expected", status=400)

        # Reject any attempt to upload raw conversation histories, personal memory stores, or identity claims
        forbidden = FORBIDDEN_ENVELOPE_FIELDS & set(payload.keys())
        if forbidden:
            raise GatewaySecurityError(
                "forbidden_personal_data",
                f"Gateway never accepts personal or conversation data ({sorted(forbidden)[0]})",
                status=400,
            )

        contributor_id = self.authenticate_contributor(
            id_token, allowed_audiences=allowed_audiences, jwks=jwks, now=now
        )
        self._check_replay_and_rate(
            contributor_id,
            nonce=payload.get("nonce"),
            timestamp=payload.get("timestamp"),
            now=now,
        )

        if payload.get("confirmed") is not True or payload.get("consent_contribute") is not True:
            raise GatewaySecurityError(
                "explicit_consent_required",
                "Candidate upload requires explicit user confirmation (confirmed=true, consent_contribute=true)",
                status=403,
            )

        packages = payload.get("packages")
        if not isinstance(packages, list) or not (1 <= len(packages) <= 50):
            raise GatewaySecurityError("invalid_packages", "packages must be a list of 1..50 candidate objects", status=400)

        accepted_ids = []
        rejected_reasons = []
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now if now is not None else time.time()))
        with self.lock:
            for raw_pkg in packages:
                if not isinstance(raw_pkg, dict):
                    rejected_reasons.append({"package_id": None, "reason": "not_a_dict"})
                    continue
                if FORBIDDEN_ENVELOPE_FIELDS & set(raw_pkg.keys()):
                    rejected_reasons.append({"package_id": raw_pkg.get("package_id"), "reason": "forbidden_personal_field"})
                    continue
                # Never trust client-supplied installation_id; bind to pseudonymous contributor_id
                pkg = dict(raw_pkg, installation_id=contributor_id)
                ok, reason = self.validate_package(pkg, consent_verified=True)
                if not ok:
                    self.rejected_packages.append({"package": pkg, "reason": reason})
                    rejected_reasons.append({"package_id": pkg.get("package_id"), "reason": reason})
                    continue
                self.ingest(pkg, consent_verified=True)
                accepted_ids.append(pkg["package_id"])

            prev_consent = self.consent_records.get(contributor_id) or {}
            self.consent_records[contributor_id] = {
                "contributor_id": contributor_id,
                "status": "active",
                "consented_utc": prev_consent.get("consented_utc") or now_iso,
                "updated_utc": now_iso,
                "accepted_packages_count": sum(1 for k in self.accepted_packages if k[0] == contributor_id),
            }
            self._save_state()

        return {
            "ok": len(accepted_ids) > 0 and not rejected_reasons,
            "contributor_id": contributor_id,
            "accepted_count": len(accepted_ids),
            "accepted_package_ids": accepted_ids,
            "rejected_count": len(rejected_reasons),
            "rejected": rejected_reasons,
            "consent_status": "active",
        }

    def withdraw_authenticated_contributor(
        self,
        *,
        id_token: str,
        payload: dict | None = None,
        allowed_audiences: list[str] | set[str] | None = None,
        jwks: list[dict] | None = None,
        now: float | None = None,
    ) -> dict:
        """Validate Google OIDC token, mark contributor consent as withdrawn, and purge all candidate packages
        submitted by this contributor."""
        contributor_id = self.authenticate_contributor(
            id_token, allowed_audiences=allowed_audiences, jwks=jwks, now=now
        )
        body = payload if isinstance(payload, dict) else {}
        self._check_replay_and_rate(
            contributor_id,
            nonce=body.get("nonce"),
            timestamp=body.get("timestamp"),
            now=now,
        )
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now if now is not None else time.time()))
        with self.lock:
            removed = self.remove_installation(contributor_id)
            prev = self.consent_records.get(contributor_id) or {}
            self.consent_records[contributor_id] = {
                "contributor_id": contributor_id,
                "status": "withdrawn",
                "consented_utc": prev.get("consented_utc"),
                "withdrawn_utc": now_iso,
                "updated_utc": now_iso,
                "accepted_packages_count": 0,
            }
            self._save_state()
        return {
            "ok": True,
            "contributor_id": contributor_id,
            "removed_packages": removed,
            "consent_status": "withdrawn",
            "withdrawn_utc": now_iso,
        }

    def validate_package(self, pkg: dict, consent_verified: bool = True) -> tuple[bool, str]:
        if not consent_verified:
            return False, "consent_not_verified"
        if not isinstance(pkg, dict):
            return False, "not_a_dict"
        req = set(LEARNING_EVENT_SCHEMA["required"])
        opt = set(LEARNING_EVENT_SCHEMA.get("optional", []))
        keys = set(pkg.keys())
        if not req.issubset(keys) or not keys.issubset(req | opt):
            return False, "schema_fields_mismatch"
        if pkg.get("schema_version") != SCHEMA_VERSION:
            return False, "unsupported_schema_version"
        if pkg.get("category") not in ALLOWED_CATEGORIES:
            return False, "invalid_category"
        inst_id = pkg.get("installation_id")
        if not isinstance(inst_id, str) or not (8 <= len(inst_id) <= 64):
            return False, "invalid_installation_id"
        trig = pkg.get("trigger")
        if not isinstance(trig, list) or not (1 <= len(trig) <= 8):
            return False, "invalid_trigger_list"
        if not personal.anchors(trig):
            return False, "missing_lexical_or_pattern_anchor"
        if not all(is_allowed_collective_feature(f, self.closed_vocab) for f in trig):
            return False, "privacy_or_vocabulary_violation"
        target = pkg.get("target")
        if not isinstance(target, list) or not (1 <= len(target) <= 6):
            return False, "invalid_target_list"
        if not all(isinstance(t, str) and t in self.allowed_targets for t in target):
            return False, "unknown_target_capability"
        strategy_str = pkg.get("strategy", "")
        if not isinstance(strategy_str, str) or len(strategy_str) > 300:
            return False, "invalid_strategy_field"
        if int(pkg.get("local_support_count", 0)) < 2 or float(pkg.get("local_precision", 0.0)) < 0.7:
            return False, "insufficient_local_evidence"
        return True, "ok"

    def ingest(self, pkg: dict, consent_verified: bool = True) -> bool:
        ok, reason = self.validate_package(pkg, consent_verified=consent_verified)
        if not ok:
            self.rejected_packages.append({"package": pkg, "reason": reason})
            return False

        inst_id = pkg["installation_id"]
        trig_key = tuple(sorted(pkg["trigger"]))
        targ_key = tuple(pkg["target"])
        key = (inst_id, trig_key, targ_key)

        if key in self.accepted_packages:
            prev = self.accepted_packages[key]
            prev["submission_count"] += 1
            prev["local_support_count"] = max(prev["local_support_count"], int(pkg["local_support_count"]))
            return True

        rec = dict(pkg)
        rec["trigger"] = list(trig_key)
        rec["target"] = list(targ_key)
        rec["submission_count"] = 1
        self.accepted_packages[key] = rec
        return True

    def remove_installation(self, installation_id_or_raw: str) -> int:
        pseudo = hashlib.sha256(f"clover-collective-v1:{installation_id_or_raw}".encode("utf-8")).hexdigest()[:16]
        targets = {installation_id_or_raw, pseudo}
        to_remove = [k for k in self.accepted_packages if k[0] in targets]
        for k in to_remove:
            del self.accepted_packages[k]
        return len(to_remove)

    def aggregate_candidates(self) -> list[dict]:
        by_trigger_target: dict[tuple[tuple[str, ...], tuple[str, ...]], list[dict]] = defaultdict(list)
        by_trigger_all_insts: dict[tuple[str, ...], dict[str, set[tuple[str, ...]]]] = defaultdict(
            lambda: defaultdict(set)
        )

        for (inst_id, trig_key, targ_key), rec in self.accepted_packages.items():
            by_trigger_target[(trig_key, targ_key)].append(rec)
            by_trigger_all_insts[trig_key][inst_id].add(targ_key)

        aggregated = []
        for (trig_key, targ_key), recs in sorted(by_trigger_target.items()):
            supporting_insts = sorted({r["installation_id"] for r in recs})
            total_submissions = sum(r["submission_count"] for r in recs)
            all_insts_for_trigger = by_trigger_all_insts[trig_key]
            conflicting_insts = sorted(
                inst for inst, targs in all_insts_for_trigger.items() if targs != {targ_key}
            )
            total_trigger_insts = len(all_insts_for_trigger)
            cross_user_agreement = len(supporting_insts) / max(1, total_trigger_insts)
            cid = hashlib.sha256(
                json.dumps({"trigger": list(trig_key), "target": list(targ_key)}, sort_keys=True).encode()
            ).hexdigest()[:16]
            aggregated.append({
                "candidate_id": f"agg_{cid}",
                "category": recs[0]["category"],
                "trigger": list(trig_key),
                "target": list(targ_key),
                "strategy": recs[0].get("strategy", ""),
                "independent_installations": len(supporting_insts),
                "supporting_installations": supporting_insts,
                "conflicting_installations": conflicting_insts,
                "total_submissions": total_submissions,
                "cross_user_agreement": round(cross_user_agreement, 4),
                "mean_local_precision": round(sum(r["local_precision"] for r in recs) / len(recs), 4),
                "total_local_support": sum(r["local_support_count"] for r in recs),
            })
        return aggregated
