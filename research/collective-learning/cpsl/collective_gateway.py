"""`collective_gateway.py` — Collective Learning Gateway.

Responsibilities (`whats-needed-for-clover.txt` §§1, 4, 6):
1. Consent & Schema Validation: Verifies `learning-event.schema.json` structure and rejects malformed or unauthorized payloads.
2. Defense-in-Depth Privacy Filtering: Re-verifies every trigger token against the closed OEWN/CSI general vocabulary
   and PII regexes so even a modified client cannot inject personal identifiers or raw text.
3. Installation-Level Deduplication (Sybil Defense):
   - Multiple submissions of the same `(trigger, target)` from the same `installation_id` are collapsed to ONE
     installation vote (`independent_installations`), defeating single-user Sybil amplification.
4. Cross-User Conflict Tracking:
   - Tracks all targets proposed for the same `trigger` across installations (`conflict_distribution`), so the
     evaluator can distinguish universal semantic/strategy agreement from contested personal habits.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ppl import world as W
from ppl.learner import anchors
from .collective import SCHEMA_VERSION, is_allowed_collective_feature, load_closed_general_vocabulary

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
    "additionalProperties": False,
}
ALLOWED_CATEGORIES = {"semantics", "reasoning-patterns", "problem-solving", "tool-use-patterns"}


class CollectiveGateway:
    """Receives, validates, privacy-filters, and aggregates candidate knowledge packages from independent Clover installations."""

    def __init__(self, oewn_path: str | Path | None = None):
        self.closed_vocab = load_closed_general_vocabulary(oewn_path)
        self.accepted_packages: dict[tuple[str, tuple[str, ...], tuple[str, ...]], dict] = {}
        self.rejected_packages: list[dict] = []

    def validate_package(self, pkg: dict, consent_verified: bool = True) -> tuple[bool, str]:
        """Validate consent, schema fields, closed-vocabulary privacy rules, and anchor requirements."""
        if not consent_verified:
            return False, "consent_not_verified"
        if not isinstance(pkg, dict):
            return False, "not_a_dict"
        required = set(LEARNING_EVENT_SCHEMA["required"])
        if set(pkg.keys()) != required:
            return False, "schema_fields_mismatch"
        if pkg.get("schema_version") != SCHEMA_VERSION:
            return False, "unsupported_schema_version"
        if pkg.get("category") not in ALLOWED_CATEGORIES:
            return False, "invalid_category"
        inst_id = pkg.get("installation_id")
        if not isinstance(inst_id, str) or not (8 <= len(inst_id) <= 64):
            return False, "invalid_installation_id"
        trig = pkg.get("trigger")
        if not isinstance(trig, list) or not trig or len(trig) > 8:
            return False, "invalid_trigger_list"
        if not anchors(trig):
            return False, "missing_lexical_or_pattern_anchor"
        if not all(is_allowed_collective_feature(f, self.closed_vocab) for f in trig):
            return False, "privacy_or_vocabulary_violation"
        target = pkg.get("target")
        if not isinstance(target, list) or not (1 <= len(target) <= 4):
            return False, "invalid_target_list"
        if not all(isinstance(t, str) and t in W.INTENTS for t in target):
            return False, "unknown_target_intent"
        if int(pkg.get("local_support_count", 0)) < 2 or float(pkg.get("local_precision", 0.0)) < 0.7:
            return False, "insufficient_local_evidence"
        return True, "ok"

    def ingest(self, pkg: dict, consent_verified: bool = True) -> bool:
        """Ingest a candidate package. Deduplicates by `(installation_id, trigger, target)` so repeated reports
        from the same installation never inflate independent support."""
        ok, reason = self.validate_package(pkg, consent_verified=consent_verified)
        if not ok:
            self.rejected_packages.append({"package": pkg, "reason": reason})
            return False

        inst_id = pkg["installation_id"]
        trig_key = tuple(sorted(pkg["trigger"]))
        targ_key = tuple(pkg["target"])
        key = (inst_id, trig_key, targ_key)

        if key in self.accepted_packages:
            # Duplicate report from the same installation: update local support count if higher, but keep 1 installation vote
            prev = self.accepted_packages[key]
            prev["submission_count"] += 1
            prev["local_support_count"] = max(prev["local_support_count"], int(pkg["local_support_count"]))
            return True

        record = dict(pkg)
        record["trigger"] = list(trig_key)
        record["target"] = list(targ_key)
        record["submission_count"] = 1
        self.accepted_packages[key] = record
        return True

    def remove_installation(self, installation_id_or_pseudo: str) -> int:
        """Remove all contributions from an installation when consent is withdrawn."""
        pseudo = hashlib.sha256(f"clover-cpsl-v1:{installation_id_or_pseudo}".encode("utf-8")).hexdigest()[:16]
        targets = {installation_id_or_pseudo, pseudo}
        to_remove = [k for k in self.accepted_packages if k[0] in targets]
        for k in to_remove:
            del self.accepted_packages[k]
        return len(to_remove)

    def aggregate_candidates(self) -> list[dict]:
        """Group deduplicated contributions by `(trigger, target)` and compute cross-installation agreement and
        conflict statistics for `collective_evaluator.py`."""
        by_trigger_target: dict[tuple[tuple[str, ...], tuple[str, ...]], list[dict]] = defaultdict(list)
        by_trigger_all_insts: dict[tuple[str, ...], dict[str, set[tuple[str, ...]]]] = defaultdict(lambda: defaultdict(set))

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
                "independent_installations": len(supporting_insts),
                "supporting_installations": supporting_insts,
                "conflicting_installations": conflicting_insts,
                "total_submissions": total_submissions,
                "cross_user_agreement": round(cross_user_agreement, 4),
                "mean_local_precision": round(sum(r["local_precision"] for r in recs) / len(recs), 4),
                "total_local_support": sum(r["local_support_count"] for r in recs),
            })
        return aggregated
