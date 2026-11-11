"""Collective Problem-Solving Learning — candidate validation and aggregation (`collective_gateway.py`).

Collective learning has no server: the shared core is the public GitHub repository `clover-collective`. This module is
used on both sides of it:
- a Clover app checks its own candidate packages with `validate_package` before it opens the public GitHub issue that
  carries them (`request_body`);
- the repository's automation (`collective_repo.py`, run by GitHub Actions) reads those issues (`parse_request`),
  checks every package again, and aggregates them across GitHub accounts.

1. Schema verification (`LEARNING_EVENT_SCHEMA`), with only the canonical strategy text for the package's tools: a
   rule's strategy becomes prompt guidance in every Clover app that receives the shared core.
2. Defense-in-Depth Closed-Vocabulary & PII Filtering (`collective.is_allowed_collective_feature`).
3. Contributor-Level Deduplication (Sybil Defense): Multiple reports of the same `(trigger, target)` from the same
   contributor count as ONE independent vote (`independent_installations`).
4. Cross-Contributor Conflict Tracking: Records when different contributors assign different targets to the same
   trigger (`cross_user_agreement`), allowing `collective_evaluator.py` to reject contested personal idioms.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import personal
import semantics
from collective import SCHEMA_VERSION, STRATEGY_GUIDANCE, is_allowed_collective_feature, load_closed_general_vocabulary

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
REQUEST_KEY = "clover_collective"  # marks a collective learning request in a GitHub issue
REQUEST_VERSION = 1
MAX_REQUEST_PACKAGES = 50
_REQUEST_JSON = re.compile(r"```json[ \t]*\r?\n(.*?)\r?\n```", re.S)

# Fields that must NEVER be accepted at either request or package level
FORBIDDEN_ENVELOPE_FIELDS = frozenset({
    "messages", "history", "conversation", "conversations", "memories", "memory_store",
    "personal_memory", "personal_store", "experiences", "raw_text", "prompt", "prompts",
    "email", "password", "access_token", "refresh_token",
})


class CollectiveGateway:
    """Validates, deduplicates, and aggregates candidate packages across contributors (GitHub accounts in the public
    repository, installations in local experiments)."""

    def __init__(self, oewn_path: str | Path | None = None, allowed_targets: set[str] | None = None):
        self.closed_vocab = load_closed_general_vocabulary(oewn_path)
        self.allowed_targets = set(allowed_targets) if allowed_targets is not None else set(semantics.CAP_WORDS)
        self.accepted_packages: dict[tuple[str, tuple[str, ...], tuple[str, ...]], dict] = {}
        self.rejected_packages: list[dict] = []

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
        if strategy_str not in ("", STRATEGY_GUIDANCE.get(tuple(sorted(target)), "")):
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


def request_body(action: str, packages=()) -> str:
    """The text of the public GitHub issue that carries a collective learning request: "contribute" with the packages
    the person reviewed (their device's installation id stays on the device), or "withdraw"."""
    if action == "contribute":
        pkgs = [{k: v for k, v in p.items() if k != "installation_id"} for p in packages]
        doc = {REQUEST_KEY: REQUEST_VERSION, "action": "contribute", "packages": pkgs}
        head = (f"Clover collective learning: {len(pkgs)} general pattern{'' if len(pkgs) == 1 else 's'} learned on "
                "the contributor's device.\n\nEach pattern is a few words from a closed general vocabulary and the "
                "tools that helped. No messages, memory, files or names. The repository's automation checks this "
                "request, replies here and closes it. A pattern joins the shared core after enough different GitHub "
                "accounts send it and it passes the held-out checks.")
    elif action == "withdraw":
        doc = {REQUEST_KEY: REQUEST_VERSION, "action": "withdraw"}
        head = "Clover collective learning: please remove my contributed patterns from future releases."
    else:
        raise ValueError(f"unknown collective learning request {action!r}")
    return f"{head}\n\n```json\n{json.dumps(doc, indent=1, sort_keys=True)}\n```\n"


def parse_request(body) -> dict | None:
    """None when an issue is not a Clover request; else {"action", "packages"}, or {"error": why it was refused}."""
    text = body if isinstance(body, str) else ""
    if f'"{REQUEST_KEY}"' not in text:
        return None
    m = _REQUEST_JSON.search(text)
    if not m:
        return {"error": "it has no json block"}
    try:
        doc = json.loads(m.group(1))
    except ValueError:
        return {"error": "its json block could not be read"}
    if not isinstance(doc, dict) or doc.get(REQUEST_KEY) != REQUEST_VERSION:
        return {"error": f"it is not request version {REQUEST_VERSION}"}
    forbidden = FORBIDDEN_ENVELOPE_FIELDS & set(doc)
    if forbidden:
        return {"error": f"personal or conversation data is never accepted ({sorted(forbidden)[0]})"}
    if doc.get("action") == "withdraw":
        return {"action": "withdraw", "packages": []}
    if doc.get("action") != "contribute":
        return {"error": "its action is neither contribute nor withdraw"}
    pkgs = doc.get("packages")
    if not isinstance(pkgs, list) or not 1 <= len(pkgs) <= MAX_REQUEST_PACKAGES:
        return {"error": f"packages must be a list of 1 to {MAX_REQUEST_PACKAGES} candidates"}
    return {"action": "contribute", "packages": pkgs}
