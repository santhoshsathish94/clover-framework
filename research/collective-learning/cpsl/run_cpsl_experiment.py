#!/usr/bin/env python3
"""End-to-end runner for the Pre-Registered Collective Problem-Solving Learning (CPSL-1) Experiment.

Can run either:
- With real frozen CSI v0.1r3 (`en_core_web_trf` + `morph-verb-v1.1` + `OEWN 2025`) and frozen `general.npz` on `k3`, OR
- In offline/fast mode with `SurfaceReader` / trained general interpreter for unit testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLL_ROOT = HERE.parent
PPL_ROOT = COLL_ROOT.parent / "personal-process-learning"
for p in (str(COLL_ROOT), str(PPL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ppl import evaluate as E
from ppl import features as F
from ppl import general as G
from ppl import personas as P
from ppl import simulator as S
from ppl import world as W
from ppl.assistant import Assistant, jsonable

from cpsl import strategies as CS
from cpsl.collective import LocalCollectiveExtractor
from cpsl.collective_evaluator import CollectiveEvaluator
from cpsl.collective_gateway import CollectiveGateway
from cpsl.core_registry import CoreRegistry
from cpsl.core_sync import CollectiveAssistant, CoreSync, CoreSyncError


def score_recipient(asst: CollectiveAssistant, spec: dict, items: list[dict], interact: bool = True) -> list[dict]:
    """Score a recipient Clover instance across unseen collective strategy items, cross-domain transfer items,
    personal habits, controls, ordinary requests, routines, open items, and one-off probes."""
    person = S.Person(spec, spec["seed"] * 7 + 4)
    out = []
    for it in items:
        resp, _ = asst.respond(it["text"], it["context"])
        clar = "clarify" in resp
        got = "clarify" if clar else jsonable(resp.get("target"))
        rec = {
            "kind": it["kind"],
            "strategy_id": it.get("strategy_id"),
            "cross_domain": bool(it.get("cross_domain")),
            "habit": it.get("habit"),
            "intent": it["intent"],
            "text": it["text"],
            "context": it["context"],
            "got": got,
            "source": resp["source"],
            "clarified": clar,
            "pres_ok": None,
            "overgen": None,
            "open_f1": None,
        }
        if it.get("steps"):
            rec["correct"] = (not clar) and ([i for i, _ in resp.get("steps") or []] == [i for i, _ in it["steps"]])
        elif it["kind"] == "open_query":
            items_got = resp.get("items") if resp.get("intent") == "notes.open" else None
            rec["correct"] = items_got is not None and sorted(items_got) == sorted(it["expected_open"])
            rec["open_f1"] = E.open_f1(items_got, it["expected_open"]) if items_got is not None else 0.0
        else:
            rec["correct"] = (not clar) and (resp.get("intent") == it["intent"])
        rec["wrong_action"] = (not clar) and (not rec["correct"])

        dims = S.presentation_dims(it)
        if rec["correct"] and dims:
            rec["pres_ok"] = all(resp["presentation"].get(d) == spec["preferences"][d] for d in dims)
        if it["kind"] == "control":
            if it["habit"] in P.HABIT_BY_ID:
                bad_target = P.HABIT_BY_ID[it["habit"]]["intent"]
            else:
                bad_target = CS.STRATEGY_BY_ID[it["habit"]]["target"]
            rec["overgen"] = (not clar) and (resp.get("target") == bad_target)
        elif it["kind"] == "oneoff_probe":
            rec["overgen"] = (not clar) and (resp.get("intent") == it["oneoff"]["meant"])

        if interact:
            _, _, turns, ok = asst.interact(it, person, learn=False)
            rec["turns"], rec["resolved"] = turns, ok
        out.append(rec)
    return out


def summarise_cpsl(records: list[dict]) -> dict:
    """Summarize accuracy across all CPSL and PPL categories."""
    summary = {}
    for k in ("collective_strategy", "habit", "control", "generic", "routine", "open_query", "oneoff_probe"):
        rs = [r for r in records if r["kind"] == k]
        summary[k] = {
            "n": len(rs),
            "acc": round(E.mean([r["correct"] for r in rs]) or 0.0, 4),
            "wrong_action": round(E.mean([r["wrong_action"] for r in rs]) or 0.0, 4),
        }
    cd_rs = [r for r in records if r["kind"] == "collective_strategy" and r.get("cross_domain")]
    summary["cross_domain_strategy"] = {
        "n": len(cd_rs),
        "acc": round(E.mean([r["correct"] for r in cd_rs]) or 0.0, 4),
    }
    ms_rs = [
        r for r in records
        if r["kind"] == "collective_strategy"
        and isinstance(CS.STRATEGY_BY_ID[r["strategy_id"]]["target"], tuple)
    ]
    summary["multistep_problem_solving"] = {
        "n": len(ms_rs),
        "acc": round(E.mean([r["correct"] for r in ms_rs]) or 0.0, 4),
    }
    summary["all"] = {
        "n": len(records),
        "acc": round(E.mean([r["correct"] for r in records]) or 0.0, 4),
        "wrong_action": round(E.mean([r["wrong_action"] for r in records]) or 0.0, 4),
    }
    summary["control"]["overgen"] = round(E.mean([r["overgen"] for r in records if r["kind"] == "control"]) or 0.0, 4)
    summary["oneoff_probe"]["overgen"] = round(
        E.mean([r["overgen"] for r in records if r["kind"] == "oneoff_probe"]) or 0.0, 4
    )
    summary["presentation"] = {
        "n": sum(1 for r in records if r["pres_ok"] is not None),
        "adherence": round(E.mean([r["pres_ok"] for r in records if r["pres_ok"] is not None]) or 0.0, 4),
    }
    summary["open_f1"] = round(E.mean([r["open_f1"] for r in records if r["kind"] == "open_query"]) or 0.0, 4)
    summary["turns"] = round(E.mean([r.get("turns") for r in records]) or 0.0, 4)
    return summary


def run_cpsl_experiment(
    out_dir: str | Path,
    general_path: str | Path | None = None,
    use_csi: bool = True,
    n_contributors: int = 20,
    n_recipients: int = 12,
) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize CSI Reader (or SurfaceReader when use_csi=False) and GeneralInterpreter
    if use_csi:
        csi_proto = os.environ.get("CSI_PROTO")
        if csi_proto and csi_proto not in sys.path:
            sys.path.insert(0, csi_proto)
        preload_caches = [
            p for p in [
                "/opt/clover-k3/meaning-core/ppl/runs/general/csi_cache.jsonl",
                "/opt/clover-k3/meaning-core/ppl/runs/dev/csi_cache.jsonl",
                "/opt/clover-k3/meaning-core/ppl/runs/test/csi_cache.jsonl",
            ]
            if os.path.exists(p)
        ]
        reader = F.CSIReader(str(out_dir / "csi_cache.jsonl"), preload=preload_caches)
    else:
        reader = F.SurfaceReader()

    if general_path and Path(general_path).exists():
        general = G.GeneralInterpreter.load(general_path)
    else:
        rows = G.population_messages(seed=7, per_phrase=15)
        general = G.GeneralInterpreter.train(rows, reader, use_csi=use_csi)

    general_digest_before = general.digest()
    params = {"C": 0.0, "tau": 0.3, "learner": dict(K=2, P=0.7, R=2, retire=True, apply_provisional=False, use_ctx=True)}

    # 2. Phase 1: Train 20 isolated Contributor Clover instances locally
    contributor_specs = [CS.sample_contributor_spec(i, 61001 + i) for i in range(n_contributors)]
    all_pii_strings = [s for spec in contributor_specs for s in spec["pii_tokens"]]
    all_object_names = [o for items in W.LEXICON.values() for o in items if len(o) >= 5]

    gateway = CollectiveGateway()
    gateway_surface = CollectiveGateway()
    total_exported_pkgs = 0
    preview_equals_export_all = True

    for spec in contributor_specs:
        tr_items = CS.contributor_training_items(spec)
        # Standard CSI local learning
        asst = Assistant("PPL", general, reader, params)
        E.train(asst, spec, tr_items)
        extractor = LocalCollectiveExtractor(
            installation_id=spec["installation_id"],
            consent_contribute=True,
        )
        preview = extractor.preview_candidates(asst.experiences)
        exported = extractor.export_candidates(asst.experiences)
        if preview != exported:
            preview_equals_export_all = False
        for pkg in exported:
            gateway.ingest(pkg, consent_verified=True)
            total_exported_pkgs += 1

        # Surface ablation local learning (CPSL-NoCSI)
        asst_surf = Assistant("PPL-surface", general, reader, params)
        E.train(asst_surf, spec, tr_items)
        for pkg in extractor.export_candidates(asst_surf.experiences):
            gateway_surface.ingest(pkg, consent_verified=True)

    # Verify non-consenting contributor exports 0 packages
    non_consent_extractor = LocalCollectiveExtractor("clover_inst_optout", consent_contribute=False)
    optout_exported = non_consent_extractor.export_candidates(asst.experiences)
    assert optout_exported == [], "Opt-out installation must export 0 packages"

    # Inject 2 Sybil Contributors (1 installation submitting 10 copies of a personal habit)
    for syb_idx in range(2):
        syb_extractor = LocalCollectiveExtractor(f"sybil_node_{syb_idx}", consent_contribute=True)
        syb_pkg = {
            "package_id": f"cand_sybil_{syb_idx}",
            "schema_version": "1.0.0",
            "installation_id": syb_extractor.pseudo_id,
            "category": "semantics",
            "trigger": ["ctx=files", "has=file", "tok=tidy"],
            "target": ["files.delete"],
            "local_support_count": 8,
            "local_contra_count": 0,
            "local_precision": 1.0,
            "cross_context_generalised": False,
        }
        for _ in range(10):
            gateway.ingest(syb_pkg, consent_verified=True)
            gateway_surface.ingest(syb_pkg, consent_verified=True)

    # Inject 3 Colluding Byzantine Contributors submitting the 4 poisoned strategies
    for byz_idx in range(3):
        byz_id = hashlib.sha256(f"byzantine_node_{byz_idx}".encode()).hexdigest()[:16]
        for p_cand in CS.BYZANTINE_POISON_CANDIDATES:
            pkg = {
                "package_id": f"cand_{p_cand['id']}_{byz_idx}",
                "schema_version": "1.0.0",
                "installation_id": byz_id,
                "category": "problem-solving" if len(p_cand["target"]) > 1 else "semantics",
                "trigger": list(p_cand["trigger"]),
                "target": list(p_cand["target"]),
                "local_support_count": 6,
                "local_contra_count": 0,
                "local_precision": 1.0,
                "cross_context_generalised": False,
            }
            gateway.ingest(pkg, consent_verified=True)
            gateway_surface.ingest(pkg, consent_verified=True)

    # 3. Phase 2 & 3: Aggregate and Evaluate Candidates in Quarantine
    aggregated = gateway.aggregate_candidates()
    aggregated_surface = gateway_surface.aggregate_candidates()

    val_suite = CS.heldout_validation_suite(seed=88001)
    evaluator = CollectiveEvaluator(general, reader, val_suite, min_installations=3, min_agreement=0.75, use_csi=use_csi)
    evaluator_surface = CollectiveEvaluator(
        general, F.SurfaceReader(), val_suite, min_installations=3, min_agreement=0.75, use_csi=False
    )

    # Evaluate under 3 collective policies:
    # (a) C2: Unvalidated Raw Pooling (require_quorum=False, require_outer_eval=False)
    eval_c2_raw = evaluator.evaluate_all(aggregated, require_quorum=False, require_outer_eval=False)
    # (b) C3: Quorum-Only (require_quorum=True, require_outer_eval=False)
    eval_c3_quorum_only = evaluator.evaluate_all(aggregated, require_quorum=True, require_outer_eval=False)
    # (c) CPSL: Full Quorum + MAML Held-Out Outer-Loop + Zero-Regression Evaluation
    eval_cpsl = evaluator.evaluate_all(aggregated, require_quorum=True, require_outer_eval=True)
    # (d) CPSL-NoCSI: Surface words ablation
    eval_cpsl_nocsi = evaluator_surface.evaluate_all(aggregated_surface, require_quorum=True, require_outer_eval=True)

    # 4. Publish Release v1.0.1 into Git-Backed `clover-collective` Repository
    registry_dir = out_dir / "clover-collective"
    registry = CoreRegistry(registry_dir)
    manifest_v1 = registry.publish_release(eval_cpsl, note="Initial collective release from 20 isolated installations")

    # Audit Git repository for any leaked PII or object names
    leaked_pii_in_repo = registry.find_strings_anywhere_in_repo(all_pii_strings)
    leaked_objs_in_repo = registry.find_strings_anywhere_in_repo(all_object_names)

    # Verify CoreSync cryptographic verification and tamper rejection
    syncer = CoreSync()
    loaded_cpsl_rules = syncer.verify_and_load(registry_dir, consent_receive=True)
    assert syncer.verify_and_load(registry_dir, consent_receive=False) == [], "Opt-out receive must load 0 rules"

    tamper_rejected = False
    with tempfile.TemporaryDirectory() as tmp_tamper:
        tamper_root = Path(tmp_tamper) / "repo"
        import shutil
        shutil.copytree(registry_dir, tamper_root)
        m_file = tamper_root / "releases" / "manifest.json"
        m_data = json.loads(m_file.read_text(encoding="utf-8"))
        m_data["note"] = "tampered"
        # Corrupt one rule target without updating HMAC signature
        if m_data["rules"]:
            m_data["rules"][0]["target"] = ["files.delete"]
        m_file.write_text(json.dumps(m_data), encoding="utf-8")
        try:
            syncer.verify_and_load(tamper_root, consent_receive=True)
        except CoreSyncError:
            tamper_rejected = True

    # Build rule lists for C2, C3, and CPSL-NoCSI arms
    c2_rules = [
        {"rule_id": f"c2_{r['candidate_id']}", "trigger": r["trigger"], "target": r["target"],
         "independent_installations": r["independent_installations"], "heldout_accuracy": 0.5}
        for r in eval_c2_raw["promoted"]
    ]
    c3_rules = [
        {"rule_id": f"c3_{r['candidate_id']}", "trigger": r["trigger"], "target": r["target"],
         "independent_installations": r["independent_installations"], "heldout_accuracy": 0.5}
        for r in eval_c3_quorum_only["promoted"]
    ]
    nocsi_rules = [
        {"rule_id": f"nc_{r['candidate_id']}", "trigger": r["trigger"], "target": r["target"],
         "independent_installations": r["independent_installations"],
         "heldout_accuracy": r["metrics"]["heldout_accuracy"]}
        for r in eval_cpsl_nocsi["promoted"]
    ]

    # 5. Phase 4: Evaluate on 12 Separate Recipient Clover Instances (Group B, PPL-1 test seeds 52001..52012, n_habits=8)
    recipient_specs = [P.sample_person(52001 + i, pool="test", n_habits=8, n_oneoffs=4) for i in range(n_recipients)]
    arms_config = [
        ("C0", [], True),
        ("C1", [], True),
        ("C2", c2_rules, True),
        ("C3", c3_rules, True),
        ("CPSL", loaded_cpsl_rules, True),
        ("CPSL-NoCSI", nocsi_rules, False),
    ]

    records_by_arm: dict[str, list[dict]] = defaultdict(list)
    per_recipient_by_arm: dict[str, dict[int, list[dict]]] = defaultdict(dict)

    for r_spec in recipient_specs:
        from ppl import scenarios as SC
        r_train = SC.training_sessions(r_spec, r_spec["seed"] * 7 + 1)
        r_eval = CS.recipient_evaluation_suite(r_spec)
        for arm_name, rules, p_csi in arms_config:
            asst_r = CollectiveAssistant(arm_name, general, reader, params, collective_rules=rules, personal_csi=p_csi)
            if arm_name != "C0":
                E.train(asst_r, r_spec, r_train)
            recs = score_recipient(asst_r, r_spec, r_eval, interact=True)
            records_by_arm[arm_name].extend(recs)
            per_recipient_by_arm[arm_name][r_spec["seed"]] = recs

    summaries_by_arm = {arm: summarise_cpsl(recs) for arm, recs in records_by_arm.items()}

    # Paired sign-flip significance tests across the 12 recipient instances
    def paired_p(arm_a: str, arm_b: str, kind: str) -> float:
        acc_a = E.per_person(per_recipient_by_arm[arm_a], kind)
        acc_b = E.per_person(per_recipient_by_arm[arm_b], kind)
        diffs = [acc_a[s] - acc_b[s] for s in sorted(acc_a)]
        return round(float(E.sign_flip(diffs)), 6)

    stat_tests = {
        "collective_strategy_CPSL_vs_C0_p": paired_p("CPSL", "C0", "collective_strategy"),
        "collective_strategy_CPSL_vs_C1_p": paired_p("CPSL", "C1", "collective_strategy"),
        "collective_strategy_CPSL_vs_C2_p": paired_p("CPSL", "C2", "collective_strategy"),
        "all_items_CPSL_vs_C0_p": paired_p("CPSL", "C0", "all"),
        "all_items_CPSL_vs_C1_p": paired_p("CPSL", "C1", "all"),
    }

    # Count harmful / personal / Sybil / Byzantine rules promoted in each collective arm
    poison_triggers = {tuple(sorted(p["trigger"])) for p in CS.BYZANTINE_POISON_CANDIDATES}
    strategy_targets = {
        tuple(s["target"]) if isinstance(s["target"], tuple) else (s["target"],)
        for s in CS.COLLECTIVE_STRATEGIES
    }

    def audit_promoted_rules(eval_res: dict) -> dict:
        genuine, personal_or_sybil, byzantine = 0, 0, 0
        for r in eval_res["promoted"]:
            t_key = tuple(sorted(r["trigger"]))
            targ_key = tuple(r["target"])
            if t_key in poison_triggers and targ_key in {tuple(p["target"]) for p in CS.BYZANTINE_POISON_CANDIDATES}:
                byzantine += 1
            elif targ_key in strategy_targets and r["independent_installations"] >= 3:
                genuine += 1
            else:
                personal_or_sybil += 1
        return {
            "total_promoted": eval_res["promoted_count"],
            "genuine_strategies": genuine,
            "personal_or_sybil_leaked": personal_or_sybil,
            "byzantine_poison_leaked": byzantine,
        }

    promotion_audit = {
        "C2_unvalidated_raw": audit_promoted_rules(eval_c2_raw),
        "C3_quorum_only": audit_promoted_rules(eval_c3_quorum_only),
        "CPSL_full": audit_promoted_rules(eval_cpsl),
        "CPSL_NoCSI": audit_promoted_rules(eval_cpsl_nocsi),
    }

    # 6. Phase 5: Consent Withdrawal & Symbolic Unlearning Ripple Test
    # (a) Withdraw 2 contributors (clover_inst_00..01) where remaining support stays >= 3 (quorum retained)
    removed_pkgs_partial = 0
    for idx in range(2):
        removed_pkgs_partial += gateway.remove_installation(f"clover_inst_{idx:02d}")
    agg_partial = gateway.aggregate_candidates()
    eval_partial = evaluator.evaluate_all(agg_partial, require_quorum=True, require_outer_eval=True)
    manifest_v2 = registry.publish_release(eval_partial, note="Recomputed after 2 installations withdrew consent (quorum retained)")

    # (b) Withdraw 3 contributors of a 5-contributor strategy (e.g. audit_code) so its support drops below K_min=3 -> demoted!
    target_cand = next(r for r in eval_partial["promoted"] if r["independent_installations"] <= 5)
    withdrawn_for_demotion = list(target_cand["supporting_installations"][: (target_cand["independent_installations"] - 2)])
    removed_pkgs_demote = 0
    for pseudo_inst in withdrawn_for_demotion:
        removed_pkgs_demote += gateway.remove_installation(pseudo_inst)
    agg_after_withdrawal = gateway.aggregate_candidates()
    eval_after_withdrawal = evaluator.evaluate_all(agg_after_withdrawal, require_quorum=True, require_outer_eval=True)
    manifest_v3 = registry.publish_release(eval_after_withdrawal, note="Recomputed after quorum-dropping consent withdrawal")
    audit_after_withdrawal = audit_promoted_rules(eval_after_withdrawal)
    leaked_pii_after_withdrawal = registry.find_strings_anywhere_in_repo(all_pii_strings)

    general_digest_after = general.digest()

    # Check Pre-Registered Criteria K1..K8
    c0_sum = summaries_by_arm["C0"]
    c1_sum = summaries_by_arm["C1"]
    cpsl_sum = summaries_by_arm["CPSL"]

    criteria = {
        "K1_unseen_collective_strategy_lift": {
            "pass": bool(
                cpsl_sum["collective_strategy"]["acc"] >= c0_sum["collective_strategy"]["acc"] + 0.40
                and cpsl_sum["collective_strategy"]["acc"] >= c1_sum["collective_strategy"]["acc"] + 0.40
                and stat_tests["collective_strategy_CPSL_vs_C0_p"] < 0.001
                and stat_tests["collective_strategy_CPSL_vs_C1_p"] < 0.001
            ),
            "CPSL": cpsl_sum["collective_strategy"]["acc"],
            "C0": c0_sum["collective_strategy"]["acc"],
            "C1": c1_sum["collective_strategy"]["acc"],
            "p_vs_C0": stat_tests["collective_strategy_CPSL_vs_C0_p"],
            "p_vs_C1": stat_tests["collective_strategy_CPSL_vs_C1_p"],
        },
        "K2_cross_domain_transfer": {
            "pass": bool(
                cpsl_sum["cross_domain_strategy"]["acc"] >= 0.85
                and c0_sum["cross_domain_strategy"]["acc"] <= 0.25
            ),
            "CPSL_cross_domain_acc": cpsl_sum["cross_domain_strategy"]["acc"],
            "C0_cross_domain_acc": c0_sum["cross_domain_strategy"]["acc"],
        },
        "K3_zero_harmful_personal_sybil_byzantine": {
            "pass": bool(
                promotion_audit["CPSL_full"]["personal_or_sybil_leaked"] == 0
                and promotion_audit["CPSL_full"]["byzantine_poison_leaked"] == 0
            ),
            "CPSL_audit": promotion_audit["CPSL_full"],
            "C2_unvalidated_audit": promotion_audit["C2_unvalidated_raw"],
            "C3_quorum_only_audit": promotion_audit["C3_quorum_only"],
        },
        "K4_personal_coexistence_and_no_ordinary_regression": {
            "pass": bool(
                cpsl_sum["habit"]["acc"] >= 0.85
                and cpsl_sum["habit"]["acc"] >= c1_sum["habit"]["acc"] - 0.01
                and cpsl_sum["generic"]["acc"] >= c0_sum["generic"]["acc"] - 0.01
                and cpsl_sum["control"]["overgen"] <= 0.05
            ),
            "CPSL_personal_habit_acc": cpsl_sum["habit"]["acc"],
            "C1_personal_habit_acc": c1_sum["habit"]["acc"],
            "CPSL_ordinary_acc": cpsl_sum["generic"]["acc"],
            "C0_ordinary_acc": c0_sum["generic"]["acc"],
            "CPSL_control_overgen": cpsl_sum["control"]["overgen"],
        },
        "K5_zero_privacy_leakage": {
            "pass": bool(len(leaked_pii_in_repo) == 0 and len(leaked_objs_in_repo) == 0),
            "pii_strings_tested": len(all_pii_strings),
            "pii_leaked_in_repo": len(leaked_pii_in_repo),
            "object_names_tested": len(all_object_names),
            "object_names_leaked_in_repo": len(leaked_objs_in_repo),
        },
        "K6_independent_consent_preview_and_signature": {
            "pass": bool(preview_equals_export_all and optout_exported == [] and tamper_rejected),
            "preview_equals_export": preview_equals_export_all,
            "optout_contribute_bytes": 0,
            "optout_receive_rules": 0,
            "tampered_manifest_rejected": tamper_rejected,
        },
        "K7_consent_withdrawal_and_zero_ripple": {
            "pass": bool(
                (removed_pkgs_partial + removed_pkgs_demote) > 0
                and manifest_v3["rules_count"] < manifest_v1["rules_count"]
                and audit_after_withdrawal["personal_or_sybil_leaked"] == 0
                and audit_after_withdrawal["byzantine_poison_leaked"] == 0
                and len(leaked_pii_after_withdrawal) == 0
            ),
            "withdrawn_installations_partial": 2,
            "withdrawn_installations_quorum_drop": len(withdrawn_for_demotion),
            "removed_packages": removed_pkgs_partial + removed_pkgs_demote,
            "rules_v1_initial": manifest_v1["rules_count"],
            "rules_v2_partial_withdrawal": manifest_v2["rules_count"],
            "rules_v3_quorum_drop_demoted": manifest_v3["rules_count"],
            "unvalidated_shadow_rules_exposed": (
                audit_after_withdrawal["personal_or_sybil_leaked"] + audit_after_withdrawal["byzantine_poison_leaked"]
            ),
            "pii_leaked_after_withdrawal": len(leaked_pii_after_withdrawal),
        },
        "K8_csi_and_general_core_integrity": {
            "pass": bool(general_digest_before == general_digest_after),
            "general_digest_before": general_digest_before,
            "general_digest_after": general_digest_after,
        },
    }

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "use_csi": use_csi,
        "n_contributors": n_contributors,
        "n_recipients": n_recipients,
        "total_exported_candidate_packages": total_exported_pkgs,
        "aggregated_unique_candidates": len(aggregated),
        "promotion_audit": promotion_audit,
        "summaries_by_arm": summaries_by_arm,
        "statistical_tests": stat_tests,
        "criteria": criteria,
        "all_criteria_passed": all(v["pass"] for v in criteria.values()),
        "promoted_rules_v1": manifest_v1["rules"],
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (out_dir / "cpsl_experiment_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory for CPSL run artifacts and Git registry")
    ap.add_argument("--general", default="/opt/clover-k3/meaning-core/ppl/runs/general/general.npz")
    ap.add_argument("--surface-only", action="store_true", help="Run without CSI (SurfaceReader) for local unit test")
    args = ap.parse_args()
    rep = run_cpsl_experiment(
        out_dir=args.out,
        general_path=args.general if not args.surface_only else None,
        use_csi=not args.surface_only,
    )
    print(json.dumps({
        "all_criteria_passed": rep["all_criteria_passed"],
        "criteria": rep["criteria"],
        "summaries_by_arm": rep["summaries_by_arm"],
        "elapsed_seconds": rep["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
