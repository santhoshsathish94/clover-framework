"""Step 3: what the curiosity monitor's blind spot changes in a turn (deterministic, real LayerKit + memory + curiosity).
A turn that retrieved memory is offered tools only through the meaning bridge; `should_observe` is one of its triggers
(omni_native.run_turn). After one topic is mastered, are unfamiliar questions that retrieve a memory note offered the
observing tools? Compared: the monitor as it is vs. the same turn where an unfamiliar question gets the prior expected
error (0.35, "unknown"), i.e. what a familiarity-aware monitor would report. Prediction written before the run: as it
is, 0 of the unfamiliar questions get tools; familiarity-aware, all of them do.
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
import curiosity_foundation as CF  # noqa: E402
import lfm3_layers as LL  # noqa: E402

PERMS = {"memory": "allow", "files": "allow", "code": "allow", "internet": "allow"}
MASTERED = [f"ring buffer partition throughput for stage {i} of the cyclic pipeline" for i in range(10)]
UNFAMILIAR = ["How do enzyme kinetics change with an allosteric inhibitor at high substrate?",
              "What load does the bridge cable carry at a 30 degree sag angle?",
              "How does compound interest grow over 20 years at 7 percent?",
              "What happens to raft quorum leases when two nodes fail?"]


def turn(kit, q):
    kit.memory_context(q)
    kit.ontology_context(q)
    b = kit.compute_meaning_bridge(q) or {}
    cur = b.get("curiosity") or {}
    offered = bool(b) and (any(str(m).startswith("LRN-") for m in b.get("memory_entries") or ())
                           or any(str(e).startswith(("FORMULA:", "STEM_CONCEPT:")) for e in b.get("ontology_entities") or ())
                           or bool(cur.get("should_observe")))
    return {"bridge": bool(b), "should_observe": cur.get("should_observe"), "region": cur.get("region_id"),
            "tools_offered": offered and bool(b.get("recommended_tools")), "tools": b.get("recommended_tools") or []}


rows = {"as_is": [], "familiarity_aware": []}
for mode in rows:
    base = Path(tempfile.mkdtemp(prefix="consc-step3-"))
    kit = LL.LayerKit(base, memory_root=base / "growth" / "knowledge")
    kit.permissions_map = PERMS
    for q in UNFAMILIAR:  # each unfamiliar question has one user note that matches it, so the bridge is computed
        kit.remember("Note about this: " + q.rstrip("?") + ".", kind="observation", conversation_id="setup")
    eng = kit.curiosity_state()
    for q in MASTERED:
        eng.record_interaction_outcome(q, pred_error=0.02, comp_error=0.0, strategy_used="sandbox_counterexample_probe")
    if mode == "familiarity_aware":
        real = eng.recommend_strategy_and_voi

        def aware(query, **kw):
            leaf = eng.locate_leaf_region(CF.encode_text_to_64d(query, latent_cue_64=kw.get("latent_cue_64")))
            ex = leaf.exemplar_vectors or []
            v = CF.encode_text_to_64d(query, latent_cue_64=kw.get("latent_cue_64"))
            near = max((float(sum(a * b for a, b in zip(v, e))) for e in ex), default=0.0)
            if near < 0.5:  # unlike everything this region has seen: report "unknown" instead of its mastered error
                saved = leaf.expected_error_yc
                leaf.expected_error_yc = 0.35
                try:
                    return real(query, **kw)
                finally:
                    leaf.expected_error_yc = saved
            return real(query, **kw)

        eng.recommend_strategy_and_voi = aware
    for q in UNFAMILIAR + MASTERED[:1]:
        rows[mode].append(dict(turn(kit, q), query=q, familiar=q in MASTERED))

summary = {m: {"unfamiliar_with_tools": sum(1 for r in v if not r["familiar"] and r["tools_offered"]),
               "unfamiliar_with_bridge": sum(1 for r in v if not r["familiar"] and r["bridge"]),
               "familiar_with_tools": sum(1 for r in v if r["familiar"] and r["tools_offered"])} for m, v in rows.items()}
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "step3_monitor_cost.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2),
                                                         encoding="utf-8")
for m, v in rows.items():
    for r in v:
        print(m, "| familiar" if r["familiar"] else "| unfamiliar", "| bridge", r["bridge"], "| observe", r["should_observe"],
              "| tools offered", r["tools_offered"], r["tools"], "|", r["query"][:50])
print(json.dumps(summary, indent=2))
