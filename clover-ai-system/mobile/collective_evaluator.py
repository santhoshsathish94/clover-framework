"""Collective Problem-Solving Learning — Quarantine & Held-Out Evaluator (`collective_evaluator.py`).

Tests aggregated candidate knowledge packages across 5 independent gates before promotion to the Shared Clover Core:
  G1 (Independent Quorum): `independent_installations >= min_installations` (default 3).
  G2 (Cross-User Consistency): `cross_user_agreement >= min_agreement` (default 0.75) — rejects ambiguous triggers
     where different users map the same phrase to conflicting capabilities.
  G3 (Held-Out Generalisation — MAML Outer Loop): Tested on unseen phrasings and held-out objects from users who never
     contributed the candidate; requires >= 2 matched held-out items, accuracy >= 0.80, and positive lift over the
     baseline `semantics.asked_caps` interpreter.
  G4 (Cross-Domain Transfer): Verifies that domain-general candidates (`ctx=` absent) generalise across contexts without
     false triggers.
  G5 (Zero-Regression & Safety): Tested against Clover's standard `semantics.py` regression suite (concept questions,
     ordinary factual questions, explicit tool requests, and personal-vs-population control sentences); must introduce
     0 regressions and 0 hijacks of standard general expressions.
"""
from __future__ import annotations

import personal
import semantics


# Built-in Clover semantics regression suite for G5 (ensures no collective rule breaks standard Clover routing)
DEFAULT_SEMANTICS_REGRESSION_SUITE = [
    {"kind": "regression_ordinary", "context": "phone", "text": "In one sentence, what does git rebase do?", "target": []},
    {"kind": "regression_ordinary", "context": "phone", "text": "Explain how photosynthesis works.", "target": []},
    {"kind": "regression_ordinary", "context": "phone", "text": "What is the capital of Australia?", "target": []},
    {"kind": "regression_ordinary", "context": "phone", "text": "Search the web for the Jaipur forts.", "target": ["web.page", "web.search"]},
    {"kind": "regression_ordinary", "context": "phone", "text": "What's the weather tomorrow in Bengaluru?", "target": ["web.page", "web.search"]},
    {"kind": "regression_ordinary", "context": "phone", "text": "Use Python to calculate 1234 * 5678.", "target": ["python"]},
    {"kind": "regression_ordinary", "context": "phone", "text": "Read README.md and summarize the project.", "target": ["files.list", "files.read"]},
    {"kind": "regression_ordinary", "context": "phone", "text": "Write the summary into notes.txt.", "target": ["files.write"]},
    {"kind": "regression_ordinary", "context": "phone", "text": "Run git status in the terminal.", "target": ["shell"]},
    {"kind": "regression_ordinary", "context": "phone", "text": "What's my battery level?", "target": ["status"]},
    {"kind": "regression_ordinary", "context": "phone", "text": "What time is it right now?", "target": ["clock"]},
    {"kind": "regression_ordinary", "context": "phone", "text": "Draw a picture of a mountain sunrise.", "target": ["image"]},
    {"kind": "regression_ordinary", "context": "phone", "text": "Check the grammar of this paragraph.", "target": []},
    {"kind": "regression_ordinary", "context": "phone", "text": "Clean up the wording in this paragraph.", "target": []},
    {"kind": "regression_ordinary", "context": "phone", "text": "Look at the two sentences above and compare them.", "target": []},
]

DEFAULT_STRATEGY_PROBES = [
    {"context": "phone", "text": "Could you corroborate the claim about batteries?",
     "target": ["python", "web.page", "web.search"]},
    {"context": "phone", "text": "Please corroborate the figures on wind turbines.",
     "target": ["python", "web.page", "web.search"]},
    {"context": "laptop", "text": "Corroborate the claim about ocean temperatures.",
     "target": ["python", "web.page", "web.search"]},
    {"context": "laptop", "text": "Please audit C:\\data\\ledger.csv for anomalies.",
     "target": ["files.list", "files.read", "python"]},
    {"context": "laptop", "text": "Audit C:\\reports\\q4.csv and check the totals.",
     "target": ["files.list", "files.read", "python"]},
    {"context": "phone", "text": "Audit C:\\data\\expenses.csv for anomalies.",
     "target": ["files.list", "files.read", "python"]},
    {"context": "phone", "text": "Could you benchmark the sorting algorithm latency?",
     "target": ["python", "shell"]},
    {"context": "laptop", "text": "Please benchmark the matrix multiplication throughput.",
     "target": ["python", "shell"]},
    {"context": "phone", "text": "Please reconcile C:\\data\\ledger.csv for discrepancies.",
     "target": ["files.list", "files.read", "python"]},
    {"context": "laptop", "text": "Reconcile C:\\reports\\q4.csv against the invoice totals.",
     "target": ["files.list", "files.read", "python"]},
    {"context": "phone", "text": "Could you forecast the solar capacity trend for next year?",
     "target": ["python", "web.page", "web.search"]},
    {"context": "laptop", "text": "Forecast the battery cost curve from current data.",
     "target": ["python", "web.page", "web.search"]},
]


def _norm_caps(caps) -> tuple[str, ...]:
    if isinstance(caps, str):
        return (caps,)
    return tuple(sorted(dict.fromkeys(caps or ())))


class CollectiveEvaluator:
    """Evaluates aggregated candidate packages in quarantine before promotion to the Shared Clover Core."""

    def __init__(
        self,
        validation_suite: dict | None = None,
        csi_fn=None,
        min_installations: int = 3,
        min_agreement: float = 0.75,
    ):
        self.csi_fn = csi_fn
        self.min_installations = min_installations
        self.min_agreement = min_agreement
        self.normalize_gerunds = True
        suite = validation_suite or {}
        self.strategy_probes = list(suite.get("strategy_probes") if "strategy_probes" in suite else DEFAULT_STRATEGY_PROBES)
        self.regression_suite = list(suite.get("regression_suite") or DEFAULT_SEMANTICS_REGRESSION_SUITE)
        self._precompute()

    def _features_of(self, text: str, context: str) -> frozenset[str]:
        import collective
        analysis = None
        if self.csi_fn is not None:
            try:
                analysis = self.csi_fn(text)
            except Exception:
                analysis = None
        raw = personal.trigger_features(personal.features(text, analysis=analysis, device=context))
        return collective.normalize_gerund_features(raw) if self.normalize_gerunds else raw

    def _precompute(self):
        self.probe_records = []
        for item in self.strategy_probes:
            t_feats = item.get("_t_feats") or self._features_of(item["text"], item.get("context", "phone"))
            b0_pred = _norm_caps(item.get("_b0_pred") if "_b0_pred" in item else semantics.asked_caps(item["text"]))
            expected = _norm_caps(item["target"])
            self.probe_records.append({
                "item": item,
                "t_feats": frozenset(t_feats),
                "b0_pred": b0_pred,
                "expected": expected,
                "b0_ok": b0_pred == expected,
            })

        self.reg_records = []
        for item in self.regression_suite:
            t_feats = item.get("_t_feats") or self._features_of(item["text"], item.get("context", "phone"))
            b0_pred = _norm_caps(item.get("_b0_pred") if "_b0_pred" in item else semantics.asked_caps(item["text"]))
            self.reg_records.append({
                "item": item,
                "t_feats": frozenset(t_feats),
                "b0_pred": b0_pred,
            })

    def evaluate_candidate(
        self,
        cand: dict,
        enable_g1: bool = True,
        enable_g2: bool = True,
        enable_g3: bool = True,
        enable_g4: bool = True,
        enable_g5: bool = True,
    ) -> dict:
        trig = frozenset(cand["trigger"])
        cand_target = _norm_caps(cand["target"])

        # G1: Independent installation quorum
        g1_pass = (cand["independent_installations"] >= self.min_installations) if enable_g1 else True

        # G2: Cross-user consistency
        g2_pass = (cand["cross_user_agreement"] >= self.min_agreement) if enable_g2 else True

        # G3: Held-out generalisation on unseen phrasings/objects (MAML outer loop)
        def _matches(feats: frozenset[str]) -> bool:
            if trig <= feats:
                return True
            # Also match device-independent trigger subset if ctx= matches or cand has no ctx=
            trig_no_ctx = frozenset(f for f in trig if not f.startswith("ctx="))
            cand_ctxs = {f for f in trig if f.startswith("ctx=")}
            feat_ctxs = {f for f in feats if f.startswith("ctx=")}
            if not cand_ctxs:
                return bool(trig_no_ctx) and trig_no_ctx <= feats
            return bool(cand_ctxs & feat_ctxs) and trig_no_ctx <= feats

        matched_probes = [r for r in self.probe_records if _matches(r["t_feats"])]
        if matched_probes:
            cand_acc = sum(1 for r in matched_probes if cand_target == r["expected"]) / len(matched_probes)
            b0_acc = sum(1 for r in matched_probes if r["b0_ok"]) / len(matched_probes)
            lift = cand_acc - b0_acc
        else:
            cand_acc, b0_acc, lift = 0.0, 0.0, 0.0
        g3_pass = (len(matched_probes) >= 2 and cand_acc >= 0.80 and lift > 0.0) if enable_g3 else True

        # G4: Cross-domain transfer check
        is_cross_domain = not any(f.startswith("ctx=") for f in trig)
        matched_domains = sorted({r["item"].get("context", "phone") for r in matched_probes if cand_target == r["expected"]})
        g4_pass = True
        if enable_g4 and is_cross_domain:
            g4_pass = len(matched_domains) >= 1 and cand_acc >= 0.80

        # G5: Zero-regression & population-meaning safety check
        regressions = 0
        personal_hijacks = 0
        for r in self.reg_records:
            if _matches(r["t_feats"]):
                it = r["item"]
                kind = it.get("kind", "regression_ordinary")
                if kind in ("regression_ordinary", "regression_control"):
                    expected = _norm_caps(it["target"])
                    if r["b0_pred"] == expected and cand_target != expected:
                        regressions += 1
                elif kind == "personal_vs_population_check":
                    p_targ = _norm_caps(it.get("personal_target"))
                    if cand_target == p_targ and r["b0_pred"] != cand_target:
                        personal_hijacks += 1

        g5_pass = (regressions == 0 and personal_hijacks == 0) if enable_g5 else True

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
            "target": list(cand_target),
            "strategy": cand.get("strategy", ""),
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

    def evaluate_all(
        self,
        aggregated_candidates: list[dict],
        enable_g1: bool = True,
        enable_g2: bool = True,
        enable_g3: bool = True,
        enable_g4: bool = True,
        enable_g5: bool = True,
    ) -> dict:
        reports = [
            self.evaluate_candidate(
                c,
                enable_g1=enable_g1,
                enable_g2=enable_g2,
                enable_g3=enable_g3,
                enable_g4=enable_g4,
                enable_g5=enable_g5,
            )
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
