"""Deterministic probes: does each report Clover makes about its own processing depend on the state it reports?
(Jackson 1982 E3, Butlin 2025 E2, Butlin 2026 white-box interventions.) Real Clover modules, temp dirs only.
Every probe has a liveness case where the report must differ, so a probe that cannot fail is visible.

F1 Outcome stage label vs. the checks that applied.  F2 C2/C3 layer info when the C2/C3 runtime is absent.
F4 curiosity monitor (region expected_error_yc) on inputs unlike anything recorded.
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOVER = HERE.parents[2] / "clover-ai-system" / "studio" / "clover"
sys.path.insert(0, str(CLOVER))
import stand_ins  # noqa: E402

stand_ins.install(CLOVER)
import clover_cycle as C  # noqa: E402
import curiosity_foundation as CF  # noqa: E402
import lfm3_layers as LL  # noqa: E402
import omni_c2_c3 as O  # noqa: E402


class StubKit:
    """compiler_check as LayerKit does it: no code in the reply -> not checked."""

    def compiler_check(self, q, reply):
        if not LL.extract_python(reply).strip():
            return None, {"layer": "compiler", "checked": False}
        return None, {"layer": "compiler", "checked": True, "ok": True}


out = {}
cases = {
    "prose_opinion": ("What do you think about the future of remote work?",
                      "Remote work will likely stay common for knowledge workers, with hybrid setups as the norm."),
    "prose_false_fact": ("What is the capital of Australia?", "The capital of Australia is Sydney."),
    "arith_correct": ("What is 17 * 23?", "17 * 23 = 391"),
    "arith_wrong_liveness": ("What is 17 * 23?", "17 * 23 = 401"),
}
f1 = {}
for name, (q, a) in cases.items():
    findings, meta = C.verify_turn_outcome(q, a, [], kit=StubKit(), layers=("memory", "ontology", "compiler"))
    applied = bool(meta.get("compiler_checked") or meta.get("python_sandbox_checked")
                   or C.direct_arithmetic_value(q) is not None or C.arithmetic_findings(a))
    derived = CF.derive_real_turn_outcome(q, a, layers=[meta["compiler_info"]] if meta.get("compiler_info") else [])
    f1[name] = {"findings": len(findings), "any_check_applied": applied,
                "stage_label_engines_show": "Outcome verified in 1 Clover cycle" if not findings else "retry",
                "backend_verified_in_turn": derived["verified_in_turn"]}
out["F1_outcome_label"] = f1

empty_base = Path(tempfile.mkdtemp(prefix="clover-nobase-"))
lk = LL.LayerKit(empty_base)
c1_state, _ = lk.c1_infer("The answer is 42.")
c2_state, c2_info = lk.c2_plan_state("What is six times seven?", c1_state=c1_state)
text_out, c3_info = lk.ncm_render("The answer is 42.")
out["F2_fallback_labels"] = {
    "c2c3_runtime_loaded": O.for_base(empty_base) is not None,
    "c2_info_module": c2_info.get("module"), "c2_info_latent_steps": c2_info.get("latent_steps"),
    "c3_info_decoder": c3_info.get("decoder"), "c3_info_bridges": c3_info.get("bridges"),
    "c3_text_changed": text_out != "The answer is 42.",
    "liveness_real_install_has_files": not O.missing_c2_c3_files(Path.home() / "CloverAI"),
}

mastered = [f"ring buffer partition throughput stage {i} cyclic pipeline slots" for i in range(8)]
unseen = ["Explain the Krebs cycle enzymes in mitochondria", "zxqv plorth wendigast mumblecrank fizzlepop",
          "What were the causes of the Peloponnesian War?"]
eng = CF.for_path(Path(tempfile.mkdtemp(prefix="clover-curio-")) / "curiosity_state.json")
for q in mastered:
    eng.record_interaction_outcome(q, pred_error=0.02, comp_error=0.0, strategy_used="sandbox_counterexample_probe")
rows = []
for q in mastered[:1] + unseen:
    r = eng.recommend_strategy_and_voi(q, permissions={"code": "allow", "files": "allow", "internet": "allow"})
    rows.append({"query": q, "seen_topic": q in mastered, "region": r["region_id"],
                 "expected_error_yc": r["expected_error_yc"], "should_observe": r["should_observe"]})
eng2 = CF.for_path(Path(tempfile.mkdtemp(prefix="clover-curio2-")) / "curiosity_state.json")
for q in mastered:
    eng2.record_interaction_outcome(q, pred_error=0.9, comp_error=0.9, strategy_used="sandbox_counterexample_probe")
out["F4_monitor_novelty"] = {"leaves": sum(1 for r in eng.regions.values() if r.is_leaf), "rows": rows,
                             "liveness_failing_region_yc": eng2.recommend_strategy_and_voi(mastered[0])["expected_error_yc"]}

(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "probe_self_reports.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
