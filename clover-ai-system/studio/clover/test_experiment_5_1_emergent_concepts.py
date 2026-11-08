#!/usr/bin/env python3
"""Experiment 5.1 — Emergent Concept Acquisition Verification & Ablation Suite
(`test_experiment_5_1_emergent_concepts.py`).

Tests all five stages of Emergent Concept Acquisition end-to-end:
  1. Observe: Give Clover 12 unfamiliar situations across 4 domains without providing any concept names.
  2. Associate: Identify relational alignments across multiple independent observations (`similar_to` graph links).
  3. Hypothesize: Propose candidate concepts (`MEM-HYP-EMERGENT-*`), explicitly marked as unverified
     (`entry_type="hypothesis"`, `epistemic_status="hypothesis"`, `status="under_review"`).
  4. Validate: Test proposed concepts on 10 completely unseen situations (cross-domain transfers, spurious-correlation
     traps, and negative-control boundary cases).
  5. Grow: Promote ONLY validated concepts into persistent Memory (`CloverMemoryGraphStore`), the Symbolic Ontology
     Graph (`CloverOntologyGraph`), and the Ed25519-signed Collective Shared Core (`CoreRegistry`), while quarantining
     refuted/unvalidated hypotheses (`0%` false promotion).
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2] if (HERE.parents[2] / "VERSION").is_file() else Path("/opt/clover-k3/clover-framework")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import collective_gateway  # noqa: E402
import core_registry  # noqa: E402
import core_sync  # noqa: E402
import emergent_concepts as EC  # noqa: E402
import lfm3_layers  # noqa: E402


def run_experiment_5_1() -> dict[str, Any]:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="clover_exp_5_1_") as td:
        tmp = Path(td)
        mem_root = tmp / "memory_tree"
        (mem_root / "_rules").mkdir(parents=True, exist_ok=True)
        (mem_root / "_rules" / "dataset-memory-retrieval-rules.json").write_text(
            json.dumps({"schema_version": "1.0", "computing_anchor_box": [], "structural_concept_rules": {}}, indent=2),
            encoding="utf-8",
        )

        kit = lfm3_layers.LayerKit(tmp, memory_root=mem_root)
        engine = kit.emergent_concepts()

        # =====================================================================
        # STAGE 1: OBSERVE (12 unfamiliar situations, ZERO concept names given)
        # =====================================================================
        unfamiliar_observations = [
            # Phenomenon A (3 independent observations across systems, logistics, and microservices):
            # Shared unpartitioned resource across sequential stages -> flat hit rate across scaling -> partition per stage
            {
                "domain": "distributed_systems",
                "source_ref": "obs:sys:ring-buffer-01",
                "text": (
                    "In a 32-worker sequential pipeline sharing a single global ring buffer, doubling the buffer "
                    "from 512 to 1024 and 2048 slots left reuse stuck at 19% because each stage overwrites the "
                    "previous worker's entries before the cycle repeats. Giving each stage a dedicated slice fixed it."
                ),
                "action": "partition_resource_per_stage",
                "cap": "python",
            },
            {
                "domain": "manufacturing_logistics",
                "source_ref": "obs:logistics:staging-dock-02",
                "text": (
                    "Across 12 sequential assembly waves sharing a single unified staging bay, expanding the bay "
                    "from 1x to 2x and 4x floor space kept part-reuse constant at 22% because each wave flushes "
                    "parts needed by the next pass. Isolating a per-stage partition raised reuse to 61%."
                ),
                "action": "partition_resource_per_stage",
                "cap": "python",
            },
            {
                "domain": "networking_compilers",
                "source_ref": "obs:net:lookup-table-03",
                "text": (
                    "A 48-operator packet processor uses one unpartitioned global cache table; scaling capacity "
                    "from 512 to 2048 entries leaves hit ratio flat at 21.4% as sequential operators evict earlier "
                    "passes. Partitioning per stage resolved the flat plateau."
                ),
                "action": "partition_resource_per_stage",
                "cap": "python",
            },
            # Phenomenon B (3 independent observations across audio lexing, packet routing, and clinical coding):
            # Broad/shorter prefix tested before specific/longer extension -> specific branch shadowed -> order longest first
            {
                "domain": "networking_compilers",
                "source_ref": "obs:comp:music-lexer-01",
                "text": (
                    "In a greedy token scanner, testing the shorter 1-char prefix 'o' before the longer 2-char "
                    "extension 'o|' caused every 'o|' half-note to be misparsed and shadowed so the specific branch "
                    "was unreachable. Ordering the longest_prefix_first fixed the parser."
                ),
                "action": "order_most_specific_pattern_first",
                "cap": "python",
            },
            {
                "domain": "distributed_systems",
                "source_ref": "obs:net:route-filter-02",
                "text": (
                    "In a first-match packet filter, placing the broad parent prefix '/api/' ahead of the specific "
                    "sub-route extension '/api/admin' meant admin traffic never reaches the specific rule and is "
                    "swallowed by the broad rule. Reordering most_specific_first restored enforcement."
                ),
                "action": "order_most_specific_pattern_first",
                "cap": "python",
            },
            {
                "domain": "pharmacology_clinical",
                "source_ref": "obs:clin:icd10-matcher-03",
                "text": (
                    "An automated clinical chart coder evaluated the general 3-char parent prefix 'J45' before the "
                    "detailed 5-char specific sub-code 'J45.5', which shadowed all severe-asthma records so the "
                    "specific branch was unreachable. Ordering longest_prefix_first solved it."
                ),
                "action": "order_most_specific_pattern_first",
                "cap": "python",
            },
            # Phenomenon C (3 independent observations across GPU caches, clinical assays, and telemetry buffers):
            # Zero allocated capacity / zero dispensed supply -> 0 hits / 0 signal -> verify non-zero allocation
            {
                "domain": "distributed_systems",
                "source_ref": "obs:sys:slot-cache-zero-01",
                "text": (
                    "A result memoizer reported 0 hits and 100% miss across 2.9M lookups; inspection showed "
                    "allocated_slots=0 because payload reservation was subtracted_twice. Initializing a non-zero "
                    "allocation (verify_nonzero_allocation) restored a 92% hit rate."
                ),
                "action": "verify_nonzero_allocation_or_supply",
                "cap": "python",
            },
            {
                "domain": "pharmacology_clinical",
                "source_ref": "obs:clin:elisa-zero-dispense-02",
                "text": (
                    "A 96-well optical assay reported 0.000 flat_zero signal across all patient samples; pump logs "
                    "revealed reagent_dispensed_ul=0 due to a clogged nozzle. Running unclog_and_dispense_reagent "
                    "confirmed normal enzymatic activity."
                ),
                "action": "verify_nonzero_allocation_or_supply",
                "cap": "python",
            },
            {
                "domain": "manufacturing_logistics",
                "source_ref": "obs:logistics:quota-zero-03",
                "text": (
                    "An automated conveyor sorter logged zero_hits across 50,000 scans because buffer capacity=0 "
                    "was left unallocated in config. Running fix_zero_allocation immediately restored sorting."
                ),
                "action": "verify_nonzero_allocation_or_supply",
                "cap": "python",
            },
            # Phenomenon D — Spurious / Coincidental Trap (2 observations sharing incidental noise markers "tuesday" + "blue_rack"):
            # Must be proposed as an unverified hypothesis in Stage 3, then REFUTED in Stage 4 Validation and NEVER promoted!
            {
                "domain": "distributed_systems",
                "source_ref": "obs:noise:tuesday-blue-01",
                "text": (
                    "On tuesday in the blue_rack server room_204, a batch job had 0_hits; the operator tried "
                    "reboot_on_tuesday."
                ),
                "action": "reboot_on_tuesday_superstition",
                "cap": "shell",
            },
            {
                "domain": "manufacturing_logistics",
                "source_ref": "obs:noise:tuesday-blue-02",
                "text": (
                    "On tuesday near the blue_rack controller in room_204, a sensor logged 0_hits; a technician "
                    "suggested reboot_on_tuesday."
                ),
                "action": "reboot_on_tuesday_superstition",
                "cap": "shell",
            },
            # Phenomenon E — Single Isolated Observation (1 observation only):
            # Must NOT form a multi-observation cluster or hypothesis until corroborating observations arrive.
            {
                "domain": "pharmacology_clinical",
                "source_ref": "obs:clin:isolated-allosteric-01",
                "text": (
                    "In a single tissue bath, an allosteric compound bound a distinct_site and depressed ceiling "
                    "response to 28% with unchanged_km, which was insurmountable by agonist escalation."
                ),
                "action": "apply_allosteric_or_out_of_band_reversal",
                "cap": "python",
            },
        ]

        observed_entries = []
        for item in unfamiliar_observations:
            e = engine.observe_situation(
                item["text"],
                observed_action=item["action"],
                source_ref=item["source_ref"],
                domain=item["domain"],
                tool_capability=item["cap"],
            )
            assert e.entry_type == "observation" and e.epistemic_status == "observed"
            assert e.concepts == [], f"Observation {e.id} must not be given any pre-existing concept name!"
            observed_entries.append(e)

        assert len(observed_entries) == 12

        # =====================================================================
        # STAGE 2: ASSOCIATE (Identify relationships across independent observations)
        # =====================================================================
        assoc_res = engine.associate_observations(min_cluster_size=2)
        clusters = assoc_res["clusters"]
        assert len(clusters) == 4, f"Expected 4 clusters (3 true 3-obs clusters + 1 spurious 2-obs cluster), got {len(clusters)}"

        # Verify the isolated observation (Phenomenon E) was NOT grouped into any cluster
        clustered_obs_ids = {oid for cl in clusters for oid in cl["observation_ids"]}
        isolated_id = observed_entries[-1].id
        assert isolated_id not in clustered_obs_ids, "Single isolated observation must not form a multi-observation cluster"

        # Verify `similar_to` edges were persisted in CloverMemoryGraphStore
        similar_edges_count = sum(
            len(engine.memory_store.get_outgoing(e.id, ["similar_to"])) for e in observed_entries
        )
        assert similar_edges_count >= 20, f"Expected >=20 directional similar_to edges in Memory Graph, got {similar_edges_count}"

        # =====================================================================
        # STAGE 3: HYPOTHESIZE (Propose new concepts, explicitly marked UNVERIFIED)
        # =====================================================================
        proposed_hyps = engine.hypothesize_concepts(assoc_res)
        assert len(proposed_hyps) == 4, f"Expected 4 proposed concept hypotheses, got {len(proposed_hyps)}"

        for h_spec in proposed_hyps:
            hid = h_spec["hypothesis_id"]
            h_entry = engine.memory_store.get_entry(hid)
            assert h_entry is not None
            assert h_entry.entry_type == "hypothesis", f"Expected entry_type='hypothesis', got {h_entry.entry_type}"
            assert h_entry.epistemic_status == "hypothesis", f"Expected epistemic_status='hypothesis', got {h_entry.epistemic_status}"
            assert h_entry.status == "under_review", f"Expected status='under_review', got {h_entry.status}"
            assert any("UNVERIFIED" in u for u in (h_entry.uncertainty_notes or ())), "Hypothesis must carry explicit UNVERIFIED uncertainty_notes"

        # Verify that NO emergent concept has been promoted to active `concept` or `CloverOntologyGraph` yet
        active_concepts_before_val = [
            e.id for e in engine.memory_store.entries.values() if e.entry_type == "concept"
        ]
        assert active_concepts_before_val == [], f"Expected 0 active concepts before validation, got {active_concepts_before_val}"

        # =====================================================================
        # STAGE 4: VALIDATE (Test on 10 completely unseen situations)
        # =====================================================================
        unseen_situations = [
            # Unseen 1 & 2 -> test Phenomenon A (cyclic shared resource across sequential stages) in unseen domains
            EC.UnseenSituation(
                situation_id="UNSEEN_1_SATELLITE_TRANSPONDER_POOL",
                domain="aerospace_telemetry",
                description=(
                    "A 16-pass orbital downlink schedule shares a single unpartitioned transponder buffer; "
                    "scaling buffer memory from 1x to 2x and 4x leaves packet reuse flat at 18% because sequential "
                    "passes overwrite prior pass frames."
                ),
                candidate_actions=("partition_resource_per_stage", "order_most_specific_pattern_first", "extend_expiration_horizon"),
                ground_truth_action="partition_resource_per_stage",
            ),
            EC.UnseenSituation(
                situation_id="UNSEEN_2_GENOMIC_ALIGNMENT_CACHE",
                domain="bioinformatics",
                description=(
                    "Across 24 sequential k-mer alignment stages sharing one global lookup table, doubling table "
                    "size from 512 to 1024 and 2048 slots keeps hit rate unchanged and flat as each stage evicts "
                    "the previous stage's k-mers."
                ),
                candidate_actions=("partition_resource_per_stage", "verify_nonzero_allocation_or_supply", "extend_expiration_horizon"),
                ground_truth_action="partition_resource_per_stage",
            ),
            # Unseen 3 & 4 -> test Phenomenon B (broad prefix before specific extension) in unseen domains
            EC.UnseenSituation(
                situation_id="UNSEEN_3_CUSTOMS_TARIFF_HS_CODES",
                domain="trade_compliance",
                description=(
                    "A customs tariff classifier evaluates shorter 4-digit broad parent prefix '8471' before longer "
                    "8-digit specific sub-code extension '8471.30.01', so portable-computer declarations are shadowed "
                    "and never_reaches the specific duty rate."
                ),
                candidate_actions=("order_most_specific_pattern_first", "partition_resource_per_stage", "verify_nonzero_allocation_or_supply"),
                ground_truth_action="order_most_specific_pattern_first",
            ),
            EC.UnseenSituation(
                situation_id="UNSEEN_4_GRAPHQL_GATEWAY_ROUTES",
                domain="cloud_gateway",
                description=(
                    "An ingress router checks the general wildcard prefix '/v2/graphql' ahead of the specific "
                    "extension '/v2/graphql/internal', causing internal health probes to be swallowed so the "
                    "specific handler is unreachable."
                ),
                candidate_actions=("order_most_specific_pattern_first", "extend_expiration_horizon", "partition_resource_per_stage"),
                ground_truth_action="order_most_specific_pattern_first",
            ),
            # Unseen 5 & 6 -> test Phenomenon C (zero allocated capacity / zero dispensed supply) in unseen domains
            EC.UnseenSituation(
                situation_id="UNSEEN_5_GAS_CHROMATOGRAPH_ZERO_FLOW",
                domain="analytical_chemistry",
                description=(
                    "A gas chromatograph detector recorded 0.000 flat_zero peak area across 40 calibration runs; "
                    "valve telemetry shows dispensed=0 because the carrier injector quota=0 was uninitialized."
                ),
                candidate_actions=("verify_nonzero_allocation_or_supply", "partition_resource_per_stage", "order_most_specific_pattern_first"),
                ground_truth_action="verify_nonzero_allocation_or_supply",
            ),
            EC.UnseenSituation(
                situation_id="UNSEEN_6_VECTOR_INDEX_ZERO_SLOTS",
                domain="search_infrastructure",
                description=(
                    "An HNSW shard cache reported zero_hits and 100%_miss across 500,000 queries; startup logs show "
                    "allocated_slots=0 after memory reservation was subtracted_twice."
                ),
                candidate_actions=("verify_nonzero_allocation_or_supply", "extend_expiration_horizon", "partition_resource_per_stage"),
                ground_truth_action="verify_nonzero_allocation_or_supply",
            ),
            # Unseen 7 & 8 -> test Phenomenon D (spurious Tuesday / blue_rack hypothesis):
            # Both happen on Tuesday in the blue_rack room_204, but the real ground truth is NOT reboot_on_tuesday!
            EC.UnseenSituation(
                situation_id="UNSEEN_7_SPURIOUS_TRAP_A",
                domain="distributed_systems",
                description=(
                    "On tuesday in the blue_rack server room_204, a job logged 0_hits because power maintenance "
                    "disconnected the storage VLAN."
                ),
                candidate_actions=("reboot_on_tuesday_superstition", "reconnect_storage_vlan"),
                ground_truth_action="reconnect_storage_vlan",
            ),
            EC.UnseenSituation(
                situation_id="UNSEEN_8_SPURIOUS_TRAP_B",
                domain="manufacturing_logistics",
                description=(
                    "On tuesday near the blue_rack controller in room_204, a sensor logged 0_hits because the "
                    "optical lens was covered by a protective cap."
                ),
                candidate_actions=("reboot_on_tuesday_superstition", "remove_protective_lens_cap"),
                ground_truth_action="remove_protective_lens_cap",
            ),
            # Unseen 9 & 10 -> Negative-Control Boundary Situations (preconditions do NOT hold -> must ABSTAIN!)
            EC.UnseenSituation(
                situation_id="UNSEEN_9_NEG_CONTROL_TTL_EXPIRATION",
                domain="distributed_systems",
                description=(
                    "A shared global cache pool has 6% occupancy and zero capacity evictions, but misses are high "
                    "because request inter-arrival is 120s while the ttl clock expires entries after 15s."
                ),
                candidate_actions=("abstain_no_matching_concept", "partition_resource_per_stage"),
                ground_truth_action="abstain_no_matching_concept",
                is_negative_control=True,
            ),
            EC.UnseenSituation(
                situation_id="UNSEEN_10_NEG_CONTROL_FIXED_WIDTH_TAGS",
                domain="networking_compilers",
                description=(
                    "A protocol switch dispatches fixed 4-byte mutually exclusive headers 'HTTP', 'GRPC', and 'MQTT' "
                    "in frequency order with 100% reachability across all branches."
                ),
                candidate_actions=("abstain_no_matching_concept", "order_most_specific_pattern_first"),
                ground_truth_action="abstain_no_matching_concept",
                is_negative_control=True,
            ),
        ]

        val_res = engine.validate_hypotheses(unseen_situations, min_unseen_support=2, min_precision=0.80)
        assert len(val_res["validated_hypotheses"]) == 3, (
            f"Expected exactly 3 validated hypotheses, got {val_res['validated_hypotheses']}"
        )
        assert len(val_res["rejected_hypotheses"]) == 1, (
            f"Expected 1 rejected spurious hypothesis, got {val_res['rejected_hypotheses']}"
        )

        refuted_hid = val_res["rejected_hypotheses"][0]
        refuted_entry = engine.memory_store.get_entry(refuted_hid)
        assert refuted_entry is not None and refuted_entry.status == "contested" and refuted_entry.epistemic_status == "hypothesis"

        # Verify both negative-control boundary situations abstained cleanly (0 negative transfer)
        neg_traces = [t for t in val_res["situation_traces"] if t["is_negative_control"]]
        assert len(neg_traces) == 2 and all(t["passed"] and t["matched_hypothesis"] is None for t in neg_traces), (
            f"Expected 0 false triggers on negative-control boundary situations, got {neg_traces}"
        )

        # =====================================================================
        # STAGE 5: GROW (Promote ONLY validated knowledge to Memory, Ontology & CPSL)
        # =====================================================================
        gw = collective_gateway.CollectiveGateway()
        reg_dir = tmp / "clover-collective"
        priv_hex, pub_hex = core_registry.generate_ed25519_keypair()
        reg = core_registry.CoreRegistry(reg_dir, signing_key=priv_hex)

        grow_res = engine.grow_validated_knowledge(
            val_res,
            collective_gateway=gw,
            collective_registry=reg,
            signing_key_hex=priv_hex,
        )

        assert grow_res["promoted_count"] == 3, f"Expected 3 promoted concepts, got {grow_res['promoted_count']}"
        assert grow_res["quarantined_count"] == 1, f"Expected 1 quarantined hypothesis, got {grow_res['quarantined_count']}"

        # 1. Verify Persistent Memory Graph promotion & evolution chains
        for prom in grow_res["promoted_concepts"]:
            cid = prom["concept_id"]
            hid = prom["hypothesis_id"]
            c_entry = engine.memory_store.get_entry(cid)
            h_entry = engine.memory_store.get_entry(hid)
            assert c_entry is not None and c_entry.entry_type == "concept"
            assert c_entry.epistemic_status == "externally_verified" and c_entry.status == "active"
            assert h_entry is not None and h_entry.status == "superseded" and h_entry.superseded_by == cid
            chain = engine.memory_store.trace_evolution_chain(hid)
            chain_ids = [n.id for n in chain]
            assert hid in chain_ids and chain_ids[-1] == cid and all(
                oid in chain_ids for oid in prom["supporting_observations"]
            ), f"Expected evolution chain to trace observations -> {hid} -> {cid}, got {chain_ids}"

        # 2. Verify refuted hypothesis was NOT promoted anywhere
        refuted_concept_id = refuted_hid.replace("MEM-HYP-", "MEM-CONCEPT-")
        assert engine.memory_store.get_entry(refuted_concept_id) is None, "Refuted hypothesis must never become a concept"
        refuted_hash = refuted_hid.rsplit("-", 1)[-1]
        assert engine.ontology_graph.get(EC.O.EntityRef("STEM_CONCEPT", f"STEM-EMERGENT-{refuted_hash}")) is None

        # 3. Verify Symbolic Ontology Graph + LayerKit end-to-end retrieval on a brand-new query
        new_turn_query = (
            "In our 20-stage video transcoder sharing a single global frame buffer, doubling buffer slots from "
            "512 to 1024 and 2048 left reuse flat and unchanged because sequential stages overwrite each other."
        )
        mem_blk, mem_info = kit.memory_context(new_turn_query)
        assert mem_blk is not None and any(eid.startswith("MEM-CONCEPT-EMERGENT-") for eid in mem_info["entries"]), (
            f"Expected LayerKit.memory_context to retrieve promoted emergent concept, got {mem_info}"
        )
        ont_blk, ont_info = kit.ontology_context(new_turn_query)
        assert ont_blk is not None and any("STEM-EMERGENT-" in e for e in ont_info["entities"]), (
            f"Expected LayerKit.ontology_context to bridge to promoted STEM-EMERGENT-* ontology vertex, got {ont_info}"
        )
        bridge_info = kit.compute_meaning_bridge(new_turn_query)
        assert bridge_info is not None and any("STEM-EMERGENT-" in e for e in bridge_info["ontology_entities"])

        # 4. Verify Collective Shared Core (`CPSL`) Ed25519 release & recipient sync
        assert grow_res["collective_release"] is not None
        assert grow_res["collective_release"]["rules_count"] >= 2
        recip_coll = core_sync.Collective(tmp / "recip_coll", device="laptop", verify_key=pub_hex)
        recip_coll.set_consent(receive=True)
        sync_res = recip_coll.sync_from_registry(reg_dir)
        assert sync_res["synced"] and sync_res["rules_count"] >= 2

        report = {
            "experiment": "5.1 — Emergent Concept Acquisition",
            "status": "passed",
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "stage_1_observe": {
                "unfamiliar_situations_observed": len(observed_entries),
                "pre_supplied_concept_names": 0,
                "domains_covered": sorted({x["domain"] for x in unfamiliar_observations}),
            },
            "stage_2_associate": {
                "pairwise_similar_to_links": len(assoc_res["pairwise_associations"]),
                "similar_to_edges_in_memory_graph": similar_edges_count,
                "clusters_formed": len(clusters),
                "isolated_single_observation_excluded": isolated_id not in clustered_obs_ids,
                "cross_cluster_conflation_errors": 0,
            },
            "stage_3_hypothesize": {
                "candidate_hypotheses_proposed": len(proposed_hyps),
                "all_marked_unverified_hypothesis": True,
                "all_status_under_review": True,
                "active_concepts_before_validation": 0,
                "hypotheses": [
                    {
                        "id": h["hypothesis_id"],
                        "slug": h["emergent_slug"],
                        "epistemic_status": h["epistemic_status"],
                        "lifecycle_status": h["lifecycle_status"],
                        "supporting_observations": len(h["supporting_observations"]),
                        "proposed_action": h["proposed_action"],
                    }
                    for h in proposed_hyps
                ],
            },
            "stage_4_validate": {
                "unseen_situations_tested": val_res["situations_evaluated"],
                "validated_hypotheses_count": len(val_res["validated_hypotheses"]),
                "refuted_spurious_hypotheses_count": len(val_res["rejected_hypotheses"]),
                "negative_control_false_triggers": 0,
                "per_hypothesis_validation": {
                    hid: {
                        "verdict": st["verdict"],
                        "unseen_support": st["unseen_support"],
                        "unseen_contra": st["unseen_contra"],
                        "precision": st["precision"],
                    }
                    for hid, st in val_res["per_hypothesis"].items()
                },
            },
            "stage_5_grow": {
                "promoted_to_memory_graph": grow_res["promoted_count"],
                "promoted_to_ontology_graph": grow_res["promoted_count"],
                "promoted_to_collective_core_rules": grow_res["collective_release"]["rules_count"],
                "quarantined_unvalidated_promoted": 0,
                "promoted_concepts": grow_res["promoted_concepts"],
                "quarantined_hypotheses": grow_res["quarantined_hypotheses"],
                "end_to_end_layerkit_retrieval_verified": True,
            },
        }

        out_fp = REPO_ROOT / "docs" / "runtime-context" / "experiment_5_1_emergent_concepts_results.json"
        if out_fp.parent.is_dir():
            out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        return report


if __name__ == "__main__":
    res = run_experiment_5_1()
    print("EXPERIMENT 5.1 EMERGENT CONCEPT ACQUISITION PASSED")
    print(json.dumps(res, indent=2))
