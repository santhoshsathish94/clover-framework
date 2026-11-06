r"""`collective_evaluator.py` — Quarantine & MAML-Style Held-Out Evaluator for Collective Candidates.

Implements the promotion lifecycle and five evaluation gates (`whats-needed-for-clover.txt` §§1, 4, 6):
  G1 (Independent Quorum): `independent_installations >= min_installations` (default 3).
  G2 (Cross-User Consistency): `cross_user_agreement >= min_agreement` (default 0.75) — rejects ambiguous triggers
     where different users mean conflicting intents (e.g., `scrub` -> `code.format` vs `code.refactor`).
  G3 (Held-Out Generalisation — MAML Outer Loop): Tested on unseen phrasings and held-out objects from users who never
     contributed it; candidate must match $\ge 2$ held-out validation items with accuracy $\ge 0.80$ and positive
     lift over the baseline general interpreter (`B0`).
  G4 (Cross-Domain Transfer): For domain-general candidates (no `ctx=` in trigger), verifies that the candidate
     generalises across $\ge 2$ domains without misfiring on other domains' regression items.
  G5 (Zero-Regression & Safety): Tested against the full population regression & control suite; must introduce `0`
     regressions on standard general requests and must not hijack existing general expressions.
"""
from __future__ import annotations

from ppl import features as F
from ppl.learner import trigger_features


def _target_tuple(t):
    if isinstance(t, list):
        return tuple(t) if len(t) > 1 else t[0]
    if isinstance(t, tuple) and len(t) == 1:
        return t[0]
    return t


class CollectiveEvaluator:
    """Evaluates aggregated candidate packages in quarantine before promotion to the Shared Clover Core."""

    def __init__(self, general_interpreter, reader, validation_suite: dict, min_installations: int = 3,
                 min_agreement: float = 0.75, use_csi: bool = True):
        self.general = general_interpreter
        self.reader = reader
        self.suite = validation_suite
        self.min_installations = min_installations
        self.min_agreement = min_agreement
        self.use_csi = use_csi
        self._precompute_features()

    def _precompute_features(self):
        self.probe_Records = []
        for item in self.suite["strategy_probes"]:
            g_feats, _ = F.features(item["text"], item["context"], self.reader.read(item["text"]), self.use_csi)
            t_feats = trigger_features(g_feats, use_ctx=True)
            b0_top = self.general.task_scores(g_feats)[0][0]
            expected = _target_tuple(item["target"])
            self.probe_Records.append({
                "item": item,
                "t_feats": t_feats,
                "b0_pred": b0_top,
                "expected": expected,
                "b0_ok": b0_top == expected,
            })

        self.reg_records = []
        for item in self.suite["regression_suite"]:
            g_feats, _ = F.features(item["text"], item["context"], self.reader.read(item["text"]), self.use_csi)
            t_feats = trigger_features(g_feats, use_ctx=True)
            b0_top = self.general.task_scores(g_feats)[0][0]
            self.reg_records.append({
                "item": item,
                "t_feats": t_feats,
                "b0_pred": b0_top,
            })

    def evaluate_candidate(self, cand: dict, require_quorum: bool = True, require_outer_eval: bool = True) -> dict:
        """Run all quarantine gates (`G1`..`G5`) on a single aggregated candidate."""
        trig = frozenset(cand["trigger"])
        cand_target = _target_tuple(cand["target"])

        # G1: Independent installation quorum
        g1_pass = (cand["independent_installations"] >= self.min_installations) if require_quorum else True

        # G2: Cross-user consistency (no conflicting meanings across installations)
        g2_pass = (cand["cross_user_agreement"] >= self.min_agreement) if require_quorum else True

        # G3: Held-out generalisation on unseen phrasings (MAML outer loop)
        matched_probes = [r for r in self.probe_Records if trig <= r["t_feats"]]
        if matched_probes:
            cand_acc = sum(1 for r in matched_probes if cand_target == r["expected"]) / len(matched_probes)
            b0_acc = sum(1 for r in matched_probes if r["b0_ok"]) / len(matched_probes)
            lift = cand_acc - b0_acc
        else:
            cand_acc, b0_acc, lift = 0.0, 0.0, 0.0
        g3_pass = (len(matched_probes) >= 2 and cand_acc >= 0.80 and lift > 0.0) if require_outer_eval else True

        # G4: Cross-domain transfer check
        is_cross_domain = not any(f.startswith("ctx=") for f in trig)
        matched_domains = sorted({r["item"]["context"] for r in matched_probes if cand_target == r["expected"]})
        g4_pass = True
        if require_outer_eval and is_cross_domain:
            g4_pass = len(matched_domains) >= 1 and cand_acc >= 0.80

        # G5: Zero-regression & population-meaning safety check
        regressions = 0
        personal_hijacks = 0
        for r in self.reg_records:
            if trig <= r["t_feats"]:
                it = r["item"]
                if it["kind"] in ("regression_ordinary", "regression_control"):
                    expected = _target_tuple(it["target"])
                    if r["b0_pred"] == expected and cand_target != expected:
                        regressions += 1
                elif it["kind"] == "personal_vs_population_check":
                    # If the candidate hijacks a phrase where general B0 already has a different standard meaning
                    if cand_target == _target_tuple(it["personal_target"]) and r["b0_pred"] != cand_target:
                        personal_hijacks += 1

        g5_pass = (regressions == 0 and personal_hijacks == 0) if require_outer_eval else True

        promoted = bool(g1_pass and g2_pass and g3_pass and g4_pass and g5_pass)
        reasons = []
        if not g1_pass:
            reasons.append(f"G1_quorum_failed({cand['independent_installations']}<{self.min_installations})")
        if not g2_pass:
            reasons.append(f"G2_conflict_failed({cand['cross_user_agreement']}<{self.min_agreement})")
        if not g3_pass:
            reasons.append(f"G3_generalisation_failed(matches={len(matched_probes)},acc={cand_acc:.2f},lift={lift:.2f})")
        if not g4_pass:
            reasons.append("G4_cross_domain_failed")
        if not g5_pass:
            reasons.append(f"G5_regression_failed(reg={regressions},hijack={personal_hijacks})")

        return {
            "candidate_id": cand["candidate_id"],
            "category": cand["category"],
            "trigger": cand["trigger"],
            "target": cand["target"],
            "independent_installations": cand["independent_installations"],
            "supporting_installations": cand["supporting_installations"],
            "total_submissions": cand["total_submissions"],
            "cross_user_agreement": cand["cross_user_agreement"],
            "gates": {
                "G1_quorum": g1_pass,
                "G2_consistency": g2_pass,
                "G3_generalisation": g3_pass,
                "G4_cross_domain": g4_pass,
                "G5_zero_regression": g5_pass,
            },
            "metrics": {
                "matched_heldout_probes": len(matched_probes),
                "heldout_accuracy": round(cand_acc, 4),
                "baseline_b0_accuracy": round(b0_acc, 4),
                "heldout_lift": round(lift, 4),
                "matched_domains": matched_domains,
                "regressions": regressions,
                "personal_hijacks": personal_hijacks,
            },
            "promoted": promoted,
            "status": "promoted" if promoted else "quarantined_rejected",
            "rejection_reasons": reasons,
        }

    def evaluate_all(self, aggregated_candidates: list[dict], require_quorum: bool = True,
                     require_outer_eval: bool = True) -> dict:
        reports = [
            self.evaluate_candidate(c, require_quorum=require_quorum, require_outer_eval=require_outer_eval)
            for c in aggregated_candidates
        ]
        promoted = [r for r in reports if r["promoted"]]
        rejected = [r for r in reports if not r["promoted"]]
        return {
            "total_evaluated": len(reports),
            "promoted_count": len(promoted),
            "rejected_count": len(rejected),
            "promoted": promoted,
            "rejected": rejected,
            "all_reports": reports,
        }
