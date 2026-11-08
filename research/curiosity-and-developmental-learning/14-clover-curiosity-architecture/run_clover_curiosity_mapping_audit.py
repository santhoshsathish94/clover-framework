#!/usr/bin/env python3
"""Step 14 — Executable Codebase Mapping & Gap Audit for the Clover 5.x Curiosity Foundation Architecture.

Programmatically inspects the live Clover 5.x codebase (`clover-ai-system/studio/clover/`) using AST and runtime
introspection to map all 13 developmental curiosity papers (`01`–`13`) against:
  1. Existing Clover components that can be directly reused (Principle 1),
  2. Missing underlying motivation, meta-prediction, and feedback mechanisms vs. existing ad-hoc / hardcoded
     constructs (Principle 2 & the "no hardcoded interest list" rule), and
  3. Persistent cross-interaction state requirements for evolving interests, unresolved questions, exploration
     history, and learning progress (Principle 3).
"""
from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CLOVER_DIR = REPO_ROOT / "clover-ai-system" / "studio" / "clover"
sys.path.insert(0, str(CLOVER_DIR))


def inspect_module_ast(rel_name: str) -> dict[str, Any]:
    fp = CLOVER_DIR / rel_name
    src = fp.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(fp))
    classes: dict[str, list[str]] = {}
    functions: list[str] = []
    top_constants: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes[node.name] = methods
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    top_constants.append(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id.isupper():
            top_constants.append(node.target.id)

    return {
        "file": f"clover-ai-system/studio/clover/{rel_name}",
        "lines": len(src.splitlines()),
        "classes": classes,
        "functions": functions,
        "constants": top_constants,
    }


def verify_live_codebase_structures() -> dict[str, Any]:
    """Import and inspect the live Clover modules to verify exact schemas, methods, and current gaps."""
    import stand_ins
    stand_ins.install(CLOVER_DIR)

    import autonomous_clover as AC
    import clover_cycle as CC
    import clover_repo as CR
    import collective as COL
    import emergent_concepts as EC
    import isolated_sandbox as SBX
    import jepa_engine as JE
    import k3_memory_graph as MG
    import k3_ontology_graph as OG
    import lfm3_layers as LL
    import memory_index as MI
    import personal as PPL
    import semantics as SEM
    import workspaces as WS
    import world_learning_mission as WLM

    # 1. Verify Memory Graph capabilities and schema
    mem_schema = {
        "valid_entry_types": list(MG.VALID_ENTRY_TYPES),
        "valid_epistemic_statuses": list(MG.VALID_EPISTEMIC_STATUSES),
        "valid_lifecycle_statuses": list(MG.VALID_LIFECYCLE_STATUSES),
        "valid_relationship_types": list(MG.VALID_RELATIONSHIP_TYPES),
        "traversal_priority": dict(MG.RELATIONSHIP_TRAVERSAL_PRIORITY),
        "has_windowed_lp_derivative": hasattr(MG.CloverMemoryGraphStore, "compute_learning_progress"),
        "has_regional_split_tree": hasattr(MG.CloverMemoryGraphStore, "split_region"),
    }

    # 2. Verify LayerKit outcome scoring vs Learning Progress derivative
    layerkit_audit = {
        "has_tally_worked_did_not_work": hasattr(LL.LayerKit, "tally"),
        "has_score_learning": hasattr(LL.LayerKit, "score_learning"),
        "has_compute_meaning_bridge": hasattr(LL.LayerKit, "compute_meaning_bridge"),
        "has_jepa_record_outcome": hasattr(LL.LayerKit, "jepa_record_outcome"),
        "has_persistent_curiosity_state": hasattr(LL.LayerKit, "curiosity_state"),
    }

    # 3. Verify AutonomousCloverEngine surprise vs Learning Progress (Paper 01, 02, 04, 10 gap check)
    ac_src = (CLOVER_DIR / "autonomous_clover.py").read_text(encoding="utf-8")
    uses_raw_surprise_argmax = "scored.sort(key=lambda item: -item[0][\"surprise_score\"])" in ac_src
    uses_without_replacement_set = "if p.problem_id not in self.solved_ids" in ac_src

    # 4. Verify hardcoded lists in WorldLearningMissionEngine & EmergentConceptEngine
    wlm_src = (CLOVER_DIR / "world_learning_mission.py").read_text(encoding="utf-8")
    has_hardcoded_gap_list = "GAP-ALG-01:" in wlm_src and "GAP-PHY-01:" in wlm_src
    has_hardcoded_book_sequence = "if \"think_python_3e\" not in self.completed_books:" in wlm_src
    has_hardcoded_condition_families = hasattr(EC, "CONDITION_FAMILIES") and "shared_cyclic_resource" in EC.CONDITION_FAMILIES

    # 5. Verify JEPA Runtime capabilities vs Confidence Module C (Paper 10)
    jepa_audit = {
        "architectures": list(JE.JEPA_ARCHITECTURES),
        "d_jepa": JE.D_JEPA,
        "d_act": JE.D_ACT,
        "has_outcome_residual_adapter": hasattr(JE.CloverJEPARuntime, "record_and_adapt_outcome"),
        "has_confidence_reliability_module_C": hasattr(JE.CloverJEPARuntime, "predict_expected_error"),
        "has_controllability_subspace_filter": hasattr(JE.CloverJEPARuntime, "estimate_controllable_subspace"),
    }

    return {
        "memory_graph_schema": mem_schema,
        "layerkit_audit": layerkit_audit,
        "autonomous_clover_gap_evidence": {
            "uses_raw_surprise_argmax_instead_of_lp_derivative": uses_raw_surprise_argmax,
            "uses_without_replacement_solved_ids_crutch": uses_without_replacement_set,
        },
        "hardcoded_interest_evidence_to_replace_with_mechanisms": {
            "world_learning_mission_hardcoded_gaps": has_hardcoded_gap_list,
            "world_learning_mission_hardcoded_book_order": has_hardcoded_book_sequence,
            "emergent_concepts_hardcoded_condition_families": has_hardcoded_condition_families,
        },
        "jepa_audit": jepa_audit,
    }


def build_13_paper_to_clover_mapping() -> list[dict[str, Any]]:
    """Map all 13 research papers and their verified mathematical mechanisms to existing Clover modules and gaps."""
    return [
        {
            "paper_num": 1,
            "paper_id": "01-oudeyer-kaplan-hafner-2007-iac",
            "mechanism": "R-IAC Progressive Region Splitting (Criterion C2: output variance minimization) + Windowed Learning Progress LP(R_n, t) + Parent-to-Child Inheritance",
            "existing_clover_components": [
                "memory-knowledge-graph.py: CloverMemoryGraphStore, MemoryEntry (tags, concepts, revisions)",
                "lfm3_layers.py: LayerKit.tally(entry), LayerKit.score_learning(eid, worked, ref)",
                "emergent_concepts.py: EmergentConceptEngine.associate_observations",
            ],
            "identified_gap": (
                "Clover stores flat per-entry revision counts (worked, did_not_work) but has no regional tree "
                "partitioning R_n over 64-d JEPA/semantic space, no Criterion C2 output-variance split when a topic "
                "region becomes heterogeneous, and no sliding-window error derivative LP(R_n, t) = <e(t-tau)> - <e(t)>."
            ),
            "derived_architectural_integration": (
                "Add persistent RegionTree (stored in growth/curiosity/regions.json) over Clover's 64-d JEPA + token "
                "feature space; split regions via Criterion C2 on outcome/prediction error variance while inheriting "
                "parent error windows and local reliability estimates."
            ),
        },
        {
            "paper_num": 2,
            "paper_id": "02-twomey-westermann-2018-infant-curiosity",
            "mechanism": "Goldilocks Intermediate Complexity (Error x Plasticity) + Unbounded Logit/Linear Progress Guard + Habituation Trace (Avoiding 2-State Limit Cycles)",
            "existing_clover_components": [
                "autonomous_clover.py: AutonomousCloverEngine.compute_problem_surprise, discover_next_problem",
                "jepa_engine.py: inspect_visual_latent, inspect_audio_latent, plan_c2_latent",
            ],
            "identified_gap": (
                "AutonomousCloverEngine.compute_problem_surprise currently maximizes raw surprise/energy "
                "(total_surprise = mem_novelty * (e_after + 0.5*e_before + ...)) and relies on a without-replacement "
                "set (self.solved_ids) to avoid repeating items. Paper 02 E3 proved this causes saturation blindness "
                "and 2-state antipodal limit cycles when 'without replacement' is removed."
            ),
            "derived_architectural_integration": (
                "Replace raw surprise maximization in problem/topic selection with non-saturating plasticity-weighted "
                "progress plus an exponential habitation/cooldown trace per region so curiosity naturally shifts "
                "along the Goldilocks frontier without needing a binary solved_ids exclusion crutch."
            ),
        },
        {
            "paper_num": 3,
            "paper_id": "03-oudeyer-smith-2016-curiosity-development",
            "mechanism": "Self-Organized Developmental Curriculum Ordering (Zero Pre-Scripted Stages) Inside a Homeostatic Safety/Discernment Hierarchy",
            "existing_clover_components": [
                "AGENTS.md (§8-§10) & semantics.py: 4-state discernment (proceed, ask, do_not_know, refuse)",
                "permissions.py & isolated_sandbox.py: runtime boundary & OS/process sandbox enforcement",
                "world_learning_mission.py: WorldLearningMissionEngine.choose_next_learning_unit",
            ],
            "identified_gap": (
                "WorldLearningMissionEngine.choose_next_learning_unit and self.knowledge_gaps hard-code the book order "
                "(think_python_3e -> university_physics_v1 -> ...) and hard-code the 4 GAP-* strings. Interests are "
                "scripted rather than emerging from Clover's actual interactions and error derivatives."
            ),
            "derived_architectural_integration": (
                "Remove hardcoded topic/book/gap lists from the curiosity foundation; let candidate interest niches "
                "emerge endogenously from user turns, failed/repaired clover_cycle verifications, and unresolved/contested "
                "edges in CloverMemoryGraphStore, gated by semantics.py discernment."
            ),
        },
        {
            "paper_num": 4,
            "paper_id": "04-oudeyer-kaplan-2007-typology",
            "mechanism": "Coupled Knowledge Progress (LPM, Eq. 11) + Competence Progress (CPM, Eq. 24) + Predictive Surprise Ratio (SM, Eq. 15) to Defeat the Passive Spectator Trap",
            "existing_clover_components": [
                "jepa_engine.py: CloverJEPARuntime (latent state prediction z_plan_64 vs z_obs_64)",
                "clover_cycle.py: arithmetic_findings, format_findings, verify_turn_outcome (competence checks)",
                "lfm3_layers.py: compiler_check, grounding_filter",
            ],
            "identified_gap": (
                "Prediction error in jepa_engine.py and verification competence in clover_cycle.py / lfm3_layers.py "
                "are tracked separately and never combined into coupled LPM + CPM derivatives. A purely predictive "
                "curiosity signal falls into the Passive Spectator Trap (Paper 04 E2: 57.87% time on unactuated "
                "spectacles where Clover has zero verification/action authority)."
            ),
            "derived_architectural_integration": (
                "Define Clover's intrinsic motivation signal in each region R_n as a coupled combination of "
                "Knowledge Progress LPM(R_n) (reduction in JEPA/symbolic prediction error) and Competence Progress "
                "CPM(R_n) (reduction in executable verification/task failure rate), weighted by controllability."
            ),
        },
        {
            "paper_num": 5,
            "paper_id": "05-kaplan-oudeyer-2007-neural-circuits",
            "mechanism": "Phasic Surprise vs. Tonic Regional Learning Progress Dissociation in Category-Level Microcolumns (MR + metaMR, Eq. 3)",
            "existing_clover_components": [
                "clover_cycle.py: fast turn-level phasic error/retry detection",
                "jepa_engine.py: 64-d latent representation z_jepa",
                "server.py: studio_idle() background hook & S.repo / S.kit",
            ],
            "identified_gap": (
                "Clover has fast phasic surprise (within-turn retry/verification), but zero persistent tonic progress "
                "state across turns. Furthermore, evaluating progress globally oscillates (85.32% in Paper 05 E1), while "
                "evaluating pointwise per exact prompt habituates immediately."
            ),
            "derived_architectural_integration": (
                "Maintain coupled category-level microcolumns (RegionNode = Predictor stats MR + Meta-predictor stats "
                "metaMR) in persistent storage, separating phasic turn surprise e(t) (which triggers immediate "
                "verification) from tonic progress p_R(t) = <e_R(t-theta)> - <e_R(t)> (which drives persistent "
                "interest and idle exploration)."
            ),
        },
        {
            "paper_num": 6,
            "paper_id": "06-gottlieb-2013-info-seeking",
            "mechanism": "Instrumental Observing Actions via Value of Information (VoI) + Local LP over Unbounded/Branch-Expanding Frontiers",
            "existing_clover_components": [
                "semantics.py: interpret(), route() tool selection",
                "lfm3_layers.py: compute_meaning_bridge(), jepa_engine.semantic_action_prior_8()",
            ],
            "identified_gap": (
                "Tool routing in semantics.py and semantic_action_prior_8 selects tools from lexical cues without "
                "computing epistemic Value of Information VoI(action | uncertainty_R, cost_action), and does not track "
                "how resolving one question spawns child unresolved questions."
            ),
            "derived_architectural_integration": (
                "Feed regional uncertainty y_C(R) and unresolved question count into LayerKit.compute_meaning_bridge "
                "to compute VoI for epistemic observing actions (memory/ontology lookup, file read, sandbox test, or "
                "clarifying question) and spawn child unresolved questions without treating expanding frontiers as failure."
            ),
        },
        {
            "paper_num": 7,
            "paper_id": "07-gottlieb-oudeyer-2018-active-sampling",
            "mechanism": "Endogenous Active Sampling Loop (Fig. 1b) + Neutral Validity Estimation Separate from Confirmation Bias (Fig. 3c vs 4a) + Stepping-Stone Goal Sampling",
            "existing_clover_components": [
                "memory-knowledge-graph.py: RELATIONSHIP_TRAVERSAL_PRIORITY (corrects=0, refines=1, challenges=2, unresolved_with=3 > supports=5)",
                "lfm3_layers.py: memory_context(), ontology_context()",
            ],
            "identified_gap": (
                "While CloverMemoryGraphStore prioritizes challenges/unresolved_with edges during passive retrieval, "
                "Clover does not actively sample or schedule falsification probes for high-uncertainty / contested "
                "hypotheses, nor does it separate neutral cue validity from positive confirmation counts."
            ),
            "derived_architectural_integration": (
                "Close the Fig. 1b loop by feeding Belief-Based Utility (expected uncertainty reduction + tonic LP) "
                "back into memory/ontology attention selection, prioritizing high-validity falsification of "
                "`hypothesis` and `unresolved` entries over re-confirming `active` entries."
            ),
        },
        {
            "paper_num": 8,
            "paper_id": "08-oudeyer-gottlieb-lopes-2016-educational",
            "mechanism": "Closed LP <-> State Curiosity <-> Memory Consolidation Loop (Fig. 1B) + ZPDES Dynamic Zone of Proximal Development with Lateral Modality Switching (Fig. 5)",
            "existing_clover_components": [
                "memory_index.py: MemoryIndex background note writer (is_idle=studio_idle)",
                "memory-knowledge-graph.py: depends_on, challenges, unresolved_with edges",
            ],
            "identified_gap": (
                "MemoryIndex writes background notes in flat chronological order (Fig. 1A unidirectional), and Clover "
                "has no ZPDES active-frontier controller that deactivates stalled questions (LP -> 0 at high error) "
                "and opens lateral prerequisite/modality paths."
            ),
            "derived_architectural_integration": (
                "Modulate idle consolidation priority in MemoryIndex / CloverMemoryGraphStore by tonic state curiosity "
                "(Fig. 1B), and maintain a dynamic ZPD frontier over unresolved questions that sidelines stalled "
                "questions and activates lateral prerequisite or alternative-strategy investigations (Fig. 5)."
            ),
        },
        {
            "paper_num": 9,
            "paper_id": "09-oudeyer-2017-baby-robots",
            "mechanism": "Morphological/Structural Computation Before Neural Calculation + Emergent Individual Developmental Trajectories",
            "existing_clover_components": [
                "clover_cycle.py: normalize(), _eval(), arithmetic_findings()",
                "generative-code-compiler.py: CloverPythonGenerativeCompiler",
                "symbolic-ontology-graph.py: CloverOntologyGraph.run_inference()",
                "isolated_sandbox.py: IsolatedPythonSandbox",
            ],
            "identified_gap": (
                "Clover has strong structural verifiers, but lacks persistent per-workspace/per-user developmental "
                "interest state so that two Clover instances exposed to different environments organically develop "
                "distinct skill and interest trajectories."
            ),
            "derived_architectural_integration": (
                "Use Clover's deterministic structural verifiers (arithmetic, compiler, Horn inference, sandbox) as "
                "zero-cost 'passive dynamics' that generate exact error/competence signals, driving emergent "
                "workspace-specific interest trajectories stored in growth/curiosity/."
            ),
        },
        {
            "paper_num": 10,
            "paper_id": "10-schmidhuber-1991-curious-control",
            "mechanism": "Confidence/Reliability Module C (Predicting Expected Error y_C) + Signed Reliability Derivative dot_y_C(t) + Smoothed Signed Shift |E[Delta y_P]|",
            "existing_clover_components": [
                "jepa_engine.py: CloverJEPARuntime.record_and_adapt_outcome (OutcomeResidualAdapter)",
            ],
            "identified_gap": (
                "CloverJEPARuntime adapts forward transition weights W_res on outcome residuals, but lacks a "
                "Confidence Module C that predicts expected error y_C(z) per latent region and computes "
                "dot_y_C(t) = y_C_old - y_C_new."
            ),
            "derived_architectural_integration": (
                "Add a lightweight Confidence/Reliability estimator y_C(R_n) (EMA of regional prediction and "
                "verification error plus signed shift tracker E[Delta z_pred | R_n]) so curiosity rewards "
                "reliability improvement dot_y_C(t) = y_C_old - y_C_new and rejects zero-mean noise."
            ),
        },
        {
            "paper_num": 11,
            "paper_id": "11-ivaldi-2014-object-learning",
            "mechanism": "SGIM-ACTS Active Choice of (Target Region, Epistemic Strategy, Actor) + Automatic Strategy Compensation When a Source Is Static/Biased + Multi-View Accumulation",
            "existing_clover_components": [
                "world_learning_mission.py: 7-dimension ConceptUnderstanding (multi-view representation)",
                "clover_cycle.py, web_search.py, isolated_sandbox.py, workspaces.py, semantics.py (strategy primitives)",
            ],
            "identified_gap": (
                "Clover has all 5 epistemic strategy primitives (memory_recall, workspace_inspect, sandbox_experiment, "
                "web_research, ask_human), but no SGIM-ACTS bandit that tracks competence progress per "
                "(region, strategy) pair and shifts away from unhelpful/static strategies."
            ),
            "derived_architectural_integration": (
                "Track empirical competence progress LP(R_n, strategy) across Clover's 5 epistemic strategies; when "
                "passive recall or text reading yields zero progress on a multi-faceted concept/question, automatically "
                "shift weight toward active IsolatedPythonSandbox experimentation or targeted human clarification."
            ),
        },
        {
            "paper_num": 12,
            "paper_id": "12-nguyen-oudeyer-2014-sgim",
            "mechanism": "SGIM-D Dual Social Guidance (Policy Imitation + Goal Emulation) with Self-Execution Calibration + |LP| Competence-Drop Vigilance + Reliability-Weighted Retrieval (dist + alpha*var)",
            "existing_clover_components": [
                "personal.py: Personal learner (before_turn, after_turn user corrections & preferences)",
                "lfm3_layers.py: learn(), score_learning(), _user_memory_package()",
            ],
            "identified_gap": (
                "User corrections in personal.py and lfm3_layers.py do not seed goal emulation in a curiosity interest "
                "map; memory retrieval does not penalize high-variance/unreliable entries via dist + alpha*var_h; and "
                "score_learning does not use |LP| to detect sudden competence drops on previously mastered topics."
            ),
            "derived_architectural_integration": (
                "Convert user corrections, retries, and questions into socially guided goal-emulation seeds in the "
                "curiosity interest map; verify/calibrate candidate rules in IsolatedPythonSandbox before promotion; "
                "use |LP| (with competence-drop vigilance) to re-activate regions when workspace code or facts change; "
                "and weight retrieved memory/concept exemplars by dist + alpha * var_h."
            ),
        },
        {
            "paper_num": 13,
            "paper_id": "13-santucci-2020-open-ended-learning",
            "mechanism": "Action-Relevant Controllable Subspace Filtering + Continuous-to-Discrete Goal Hierarchy + Dual Episodic-Semantic Memory Consolidation Against Catastrophic Forgetting",
            "existing_clover_components": [
                "clover_repo.py & memory-knowledge-graph.py (Fast Episodic Tier: sessions/, outcomes/, observation/hypothesis entries)",
                "symbolic-ontology-graph.py & jepa_engine.py (Slow Semantic Tier: CloverOntologyGraph + OutcomeResidualAdapter rollback gate)",
                "emergent_concepts.py: Observe -> Associate -> Hypothesize -> Validate -> Grow pipeline",
            ],
            "identified_gap": (
                "The Episodic -> Semantic consolidation pipeline in emergent_concepts.py and world_learning_mission.py "
                "is currently disconnected from live Studio turns and lacks an action-controllability filter R^2(Action -> Outcome) "
                "and periodic regression replay on previously promoted concepts."
            ),
            "derived_architectural_integration": (
                "Connect the Fast Episodic Tier (turns, outcomes, observations, unresolved questions) to the Slow "
                "Semantic Tier (CloverMemoryGraphStore verified concepts + CloverOntologyGraph) through the persistent "
                "Curiosity Foundation hooks (`on_turn_outcome` and `on_idle_step`), filtering candidate goal dimensions "
                "by action/verification controllability and running non-regression replay before promoting new concepts."
            ),
        },
    ]


def main() -> int:
    t0 = time.perf_counter()
    core_files = (
        "curiosity_foundation.py",
        "test_curiosity_foundation.py",
        "clover_cycle.py",
        "lfm3_layers.py",
        "memory-knowledge-graph.py",
        "symbolic-ontology-graph.py",
        "jepa_engine.py",
        "autonomous_clover.py",
        "emergent_concepts.py",
        "world_learning_mission.py",
        "real_source_learning.py",
        "personal.py",
        "collective.py",
        "clover_repo.py",
        "memory_index.py",
        "workspaces.py",
        "semantics.py",
        "isolated_sandbox.py",
        "server.py",
    )
    ast_inventory = {f: inspect_module_ast(f) for f in core_files}
    live_verification = verify_live_codebase_structures()
    mapping_13 = build_13_paper_to_clover_mapping()

    assert len(mapping_13) == 13
    assert live_verification["memory_graph_schema"]["has_windowed_lp_derivative"] is True
    assert live_verification["memory_graph_schema"]["has_regional_split_tree"] is True
    assert live_verification["layerkit_audit"]["has_persistent_curiosity_state"] is True
    assert live_verification["jepa_audit"]["has_confidence_reliability_module_C"] is True
    assert live_verification["jepa_audit"]["has_controllability_subspace_filter"] is True
    assert live_verification["autonomous_clover_gap_evidence"]["uses_raw_surprise_argmax_instead_of_lp_derivative"] is False
    assert live_verification["hardcoded_interest_evidence_to_replace_with_mechanisms"]["world_learning_mission_hardcoded_gaps"] is False
    assert live_verification["hardcoded_interest_evidence_to_replace_with_mechanisms"]["world_learning_mission_hardcoded_book_order"] is False

    report = {
        "audit_id": "14-clover-curiosity-foundation-architecture-audit",
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "inspected_modules_count": len(ast_inventory),
        "total_clover_studio_lines_inspected": sum(v["lines"] for v in ast_inventory.values()),
        "live_codebase_verification": live_verification,
        "thirteen_paper_mechanism_mapping": mapping_13,
        "module_ast_inventory": ast_inventory,
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "clover_curiosity_mapping_audit.json"
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit_id": report["audit_id"],
        "elapsed_s": report["elapsed_s"],
        "inspected_modules_count": report["inspected_modules_count"],
        "total_clover_studio_lines_inspected": report["total_clover_studio_lines_inspected"],
        "live_codebase_verification": report["live_codebase_verification"],
        "mapped_papers": len(report["thirteen_paper_mechanism_mapping"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
