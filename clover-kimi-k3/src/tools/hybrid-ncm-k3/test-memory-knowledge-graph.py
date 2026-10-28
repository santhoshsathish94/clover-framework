"""Automated Validation & Test Suite for Clover Kimi K3 File-Based Memory Tree & Relational Knowledge Graph (`test_k3_memory_graph.py`).

Tests all 11 required validation areas (Section 7):
  1. Stable IDs and valid metadata (including file relocation across directories without breaking IDs or links).
  2. Relationship creation, removal, and schema validation across all 10 relationship types.
  3. Incoming and outgoing relationship lookup & directional traversal.
  4. Cross-directory relationships across multiple branches of the file tree.
  5. Cyclic graph traversal without infinite loops + detection of invalid `corrects` cycles.
  6. Broken-link detection, duplicate-ID detection, and full graph-index rebuilding from canonical `.md` files.
  7. Provenance & epistemic status preservation (preventing user philosophical beliefs from becoming verified facts).
  8. Summary revision (`revise_summary`) and correction/superseding chains (`record_correction`, `mark_superseded`) preserving original source evidence and revision history.
  9. Contradictory and unresolved claims (`challenges`, `unresolved_with`, and uncertain duplicate retention).
  10. Retrieval relevance, lexical-vs-semantic disambiguation, source traceability, and bounded token/entry budgets.
  11. Persistence across process restarts (via independent Python subprocess execution) + Required Scenario ('everything happens inside' evolving interpretation & general dynamic correction chains).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from k3_memory_graph import (
    CloverMemoryGraphStore,
    DEFAULT_MEMORY_TREE_DIR,
    MemoryEntry,
    MemoryProvenance,
    MemoryRelationship,
    VALID_ENTRY_TYPES,
    VALID_EPISTEMIC_STATUSES,
    VALID_RELATIONSHIP_TYPES,
    ensure_default_memory_tree,
    parse_memory_markdown,
    render_memory_markdown,
    retrieve_relevant_memory_package,
)


class TestCloverMemoryGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="clover_mem_test_"))
        self.store = ensure_default_memory_tree(self.tmp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # 1. Stable IDs, Metadata Validation & Path-Independent Relocation
    # ------------------------------------------------------------------
    def test_01_stable_ids_and_metadata_validation(self) -> None:
        report = self.store.last_validation
        self.assertIsNotNone(report)
        assert report is not None
        self.assertTrue(report.is_healthy, f"Expected healthy default corpus, got: {report.to_dict()}")
        self.assertGreaterEqual(report.total_files, 32)
        self.assertEqual(report.valid_entries, report.total_files)
        self.assertGreaterEqual(report.total_relationships, 65)

        # Verify every entry has a valid stable ID, entry_type, epistemic_status, timestamps, and provenance
        for eid, entry in self.entries_dict().items():
            self.assertEqual(eid, entry.id)
            self.assertIn(entry.entry_type, VALID_ENTRY_TYPES)
            self.assertIn(entry.epistemic_status, VALID_EPISTEMIC_STATUSES)
            self.assertTrue(entry.provenance.is_valid())
            self.assertEqual(entry.validate(), [])

        # Relocate a file to a completely different branch directory on disk and verify its stable ID
        # and all incoming/outgoing relationships remain 100% intact after rebuilding the index!
        orig_entry = self.store.get_entry("MEM-CORR-INSIDE-BOUNDARY-002")
        assert orig_entry is not None
        old_fp = self.tmp_dir / orig_entry.rel_path
        new_rel = " relocated_branch/deep/nested/corr-inside-moved.md".strip()
        new_fp = self.tmp_dir / new_rel
        new_fp.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_fp), str(new_fp))

        rebuilt = self.store.rebuild_index()
        self.assertTrue(rebuilt.is_healthy)
        moved_entry = self.store.get_entry("MEM-CORR-INSIDE-BOUNDARY-002")
        assert moved_entry is not None
        self.assertEqual(moved_entry.rel_path, new_rel)
        self.assertEqual(
            [r.target for r in self.store.get_outgoing("MEM-CORR-INSIDE-BOUNDARY-002", ["corrects"])],
            ["MEM-HYP-INSIDE-MIND-001"],
        )
        self.assertEqual(
            [r.source for r in self.store.get_incoming("MEM-CORR-INSIDE-BOUNDARY-002", ["exemplifies"])],
            ["MEM-CONCEPT-OBSERVER-BOUNDARY-001"],
        )

    def entries_dict(self):
        return self.store.entries

    # ------------------------------------------------------------------
    # 2. Relationship Creation, Removal & Schema Validation
    # ------------------------------------------------------------------
    def test_02_relationship_creation_and_validation(self) -> None:
        # Verify all 10 required relationship types are supported
        expected_rels = {
            "supports", "challenges", "refines", "depends_on", "similar_to",
            "derived_from", "caused_by", "corrects", "exemplifies", "unresolved_with",
        }
        self.assertEqual(set(VALID_RELATIONSHIP_TYPES), expected_rels)

        # Reject invalid relationship type
        with self.assertRaises(ValueError):
            self.store.add_relationship(
                source_id="MEM-DEC-SLOTCACHE-PER-LAYER-001",
                relation_type="invented_invalid_edge",
                target_id="MEM-OBS-GLOBAL-CACHE-FLAT-001",
                provenance="unit_test",
            )

        # Reject self-referential irreflexive relationship (`corrects` to self)
        with self.assertRaises(ValueError):
            self.store.add_relationship(
                source_id="MEM-DEC-SLOTCACHE-PER-LAYER-001",
                relation_type="corrects",
                target_id="MEM-DEC-SLOTCACHE-PER-LAYER-001",
                provenance="unit_test",
            )

        # Reject out-of-range confidence
        with self.assertRaises(ValueError):
            self.store.add_relationship(
                source_id="MEM-DEC-SLOTCACHE-PER-LAYER-001",
                relation_type="supports",
                target_id="MEM-OBS-GLOBAL-CACHE-FLAT-001",
                provenance="unit_test",
                confidence=1.5,
            )

        # Valid relationship addition and removal
        added = self.store.add_relationship(
            source_id="MEM-OUT-RUST-GPU-PARITY-001",
            relation_type="exemplifies",
            target_id="MEM-CORR-PER-LAYER-PARTITION-002",
            provenance="test_02",
            confidence=0.95,
            note="Bit-identical GPU run exemplifies per-layer slot partitioning.",
        )
        self.assertEqual(added.type, "exemplifies")
        self.assertIn(
            "MEM-CORR-PER-LAYER-PARTITION-002",
            [r.target for r in self.store.get_outgoing("MEM-OUT-RUST-GPU-PARITY-001", ["exemplifies"])],
        )
        removed = self.store.remove_relationship(
            "MEM-OUT-RUST-GPU-PARITY-001", "exemplifies", "MEM-CORR-PER-LAYER-PARTITION-002"
        )
        self.assertTrue(removed)
        self.assertEqual(
            [r.target for r in self.store.get_outgoing("MEM-OUT-RUST-GPU-PARITY-001", ["exemplifies"])],
            [],
        )

    # ------------------------------------------------------------------
    # 3 & 4. Incoming/Outgoing Lookup & Cross-Directory Traversal
    # ------------------------------------------------------------------
    def test_03_04_incoming_outgoing_and_cross_directory_traversal(self) -> None:
        # Outgoing from `MEM-CORR-INSIDE-BOUNDARY-002` (in `philosophy/corrections/`) reaches:
        # - `MEM-HYP-INSIDE-MIND-001` (in `philosophy/interpretations/`)
        # - `MEM-CONV-PHIL-INSIDE-001` (in `conversations/`)
        # - `MEM-CONCEPT-OBSERVER-BOUNDARY-001` (in `concepts/`)
        out_rels = self.store.get_outgoing("MEM-CORR-INSIDE-BOUNDARY-002")
        out_targets = {r.target for r in out_rels}
        self.assertEqual(
            out_targets,
            {"MEM-HYP-INSIDE-MIND-001", "MEM-CONV-PHIL-INSIDE-001", "MEM-CONCEPT-OBSERVER-BOUNDARY-001"},
        )

        # Verify these targets live in 3 distinct directory branches
        target_dirs = {
            Path(self.store.get_entry(tid).rel_path).parts[0]  # type: ignore[union-attr]
            for tid in out_targets
        }
        self.assertEqual(target_dirs, {"philosophy", "conversations", "concepts"})

        # Incoming to `MEM-HYP-INSIDE-MIND-001` includes `corrects` from `MEM-CORR-INSIDE-BOUNDARY-002`
        # and `supports` from `MEM-HYP-STRICT-EXTERNAL-REALISM-001`
        in_rels = self.store.get_incoming("MEM-HYP-INSIDE-MIND-001")
        self.assertEqual(
            {(r.source, r.type) for r in in_rels},
            {
                ("MEM-CORR-INSIDE-BOUNDARY-002", "corrects"),
                ("MEM-HYP-STRICT-EXTERNAL-REALISM-001", "supports"),
            },
        )

        # Multi-hop traversal across Systems engineering directories (`outcomes/` -> `experiments/` -> `decisions/` -> `corrections/` -> `observations/`)
        trav = self.store.traverse(
            seed_ids=["MEM-OUT-RUST-GPU-PARITY-001"],
            max_depth=4,
            direction="both",
        )
        reached_ids = [n.entry.id for n in trav]
        for expected_id in (
            "MEM-OUT-RUST-GPU-PARITY-001",
            "MEM-EXP-RUST-SLOTCACHE-001",
            "MEM-DEC-SLOTCACHE-PER-LAYER-001",
            "MEM-CORR-PER-LAYER-PARTITION-002",
            "MEM-HYP-CACHE-DEAD-END-001",
            "MEM-OBS-GLOBAL-CACHE-FLAT-001",
        ):
            self.assertIn(expected_id, reached_ids)

    # ------------------------------------------------------------------
    # 5. Cyclic Graph Traversal Without Infinite Loops
    # ------------------------------------------------------------------
    def test_05_cyclic_graph_traversal_and_cycle_handling(self) -> None:
        # `MEM-CONCEPT-OBSERVER-BOUNDARY-001` and `MEM-HYP-STRICT-EXTERNAL-REALISM-001` form a 2-cycle via `unresolved_with` / `challenges`.
        # Traversing with high max_depth=10 must terminate immediately and visit each node at most once.
        trav = self.store.traverse(
            seed_ids=["MEM-CONCEPT-OBSERVER-BOUNDARY-001"],
            max_depth=10,
            direction="both",
        )
        visited_ids = [n.entry.id for n in trav]
        self.assertEqual(len(visited_ids), len(set(visited_ids)), "Cycle caused duplicate node visits!")
        self.assertIn("MEM-HYP-STRICT-EXTERNAL-REALISM-001", visited_ids)

        # Also test a 3-node cycle (`C1 -> C2 -> C3 -> C1`) and verify `corrects` cycle warning detection
        for cid in ("MEM-CYCLE-A", "MEM-CYCLE-B", "MEM-CYCLE-C"):
            self.store.create_entry(
                MemoryEntry(
                    id=cid,
                    title=f"Cycle Node {cid}",
                    entry_type="concept",
                    epistemic_status="hypothesis",
                    summary=f"Summary for {cid}",
                    content=f"Content for {cid}",
                    source_evidence="Cycle test",
                    provenance=MemoryProvenance(source_type="observation", source_ref="test_05"),
                )
            )
        self.store.add_relationship("MEM-CYCLE-A", "similar_to", "MEM-CYCLE-B", "test_05")
        self.store.add_relationship("MEM-CYCLE-B", "similar_to", "MEM-CYCLE-C", "test_05")
        self.store.add_relationship("MEM-CYCLE-C", "similar_to", "MEM-CYCLE-A", "test_05")

        trav_3 = self.store.traverse(seed_ids=["MEM-CYCLE-A"], max_depth=12, direction="outgoing")
        self.assertEqual([n.entry.id for n in trav_3], ["MEM-CYCLE-A", "MEM-CYCLE-B", "MEM-CYCLE-C"])

        # Now introduce an invalid `corrects` cycle (`A corrects B` and `B corrects A`) and confirm `rebuild_index` flags it in `cycle_warnings`
        self.store.add_relationship("MEM-CYCLE-A", "corrects", "MEM-CYCLE-B", "test_05")
        self.store.add_relationship("MEM-CYCLE-B", "corrects", "MEM-CYCLE-A", "test_05")
        rep = self.store.rebuild_index()
        self.assertFalse(rep.is_healthy)
        self.assertTrue(len(rep.cycle_warnings) >= 1)

    # ------------------------------------------------------------------
    # 6. Broken-Link Detection, Duplicate-ID Detection & Index Rebuilding
    # ------------------------------------------------------------------
    def test_06_broken_link_detection_and_index_rebuild(self) -> None:
        # Delete `_index/memory_graph_index.json` completely and verify `rebuild_index()` reconstructs it identically
        self.assertTrue(self.store.index_file.exists())
        before_json = json.loads(self.store.index_file.read_text(encoding="utf-8"))
        self.store.index_file.unlink()
        self.assertFalse(self.store.index_file.exists())

        rep = self.store.rebuild_index(write_index_file=True)
        self.assertTrue(rep.is_healthy)
        self.assertTrue(self.store.index_file.exists())
        after_json = json.loads(self.store.index_file.read_text(encoding="utf-8"))
        self.assertEqual(before_json["nodes"], after_json["nodes"])
        self.assertEqual(before_json["edges"], after_json["edges"])

        # Add a relationship pointing to a non-existent target ID (`allow_missing_target=True`) and verify broken link detection
        self.store.add_relationship(
            source_id="MEM-OBS-GLOBAL-CACHE-FLAT-001",
            relation_type="supports",
            target_id="MEM-NON-EXISTENT-TARGET-999",
            provenance="test_broken_link",
            allow_missing_target=True,
        )
        rep_broken = self.store.rebuild_index()
        self.assertFalse(rep_broken.is_healthy)
        self.assertEqual(len(rep_broken.broken_links), 1)
        self.assertEqual(rep_broken.broken_links[0]["missing_target"], "MEM-NON-EXISTENT-TARGET-999")

    # ------------------------------------------------------------------
    # 7 & 8. Provenance, Epistemic Status, Summary Revision & Correction Chains
    # ------------------------------------------------------------------
    def test_07_08_provenance_revision_and_correction_preservation(self) -> None:
        # Verify philosophical beliefs and unresolved questions are NOT marked as `externally_verified`
        corr_phil = self.store.get_entry("MEM-CORR-INSIDE-BOUNDARY-002")
        concept_phil = self.store.get_entry("MEM-CONCEPT-OBSERVER-BOUNDARY-001")
        assert corr_phil is not None and concept_phil is not None
        self.assertEqual(corr_phil.epistemic_status, "user_stated_belief")
        self.assertEqual(concept_phil.epistemic_status, "unresolved")

        # Revise summary of `MEM-OBS-GLOBAL-CACHE-FLAT-001` and verify original source evidence is untouched
        obs_before = self.store.get_entry("MEM-OBS-GLOBAL-CACHE-FLAT-001")
        assert obs_before is not None
        orig_evidence = obs_before.source_evidence
        orig_summary = obs_before.summary

        revised = self.store.revise_summary(
            entry_id="MEM-OBS-GLOBAL-CACHE-FLAT-001",
            new_summary="Revised summary: Global LRU cache hit rate stayed flat at 23.3% from 512 to 4096 slots.",
            actor="unit_test",
            reason="Clarified LRU policy in summary",
        )
        self.assertEqual(revised.source_evidence, orig_evidence, "Original source evidence must never be altered by revise_summary!")
        self.assertEqual(revised.revisions[-1].previous_summary, orig_summary)
        self.assertEqual(revised.revisions[-1].action, "revise_summary")

    # ------------------------------------------------------------------
    # 9. Contradictory Claims & Uncertain Concept Duplicates
    # ------------------------------------------------------------------
    def test_09_contradictory_claims_and_uncertain_duplicates(self) -> None:
        # Query the contested boundary concept and verify contradictions are surfaced, never flattened
        pkg = self.store.retrieve_context_package("Do inside and outside exist as strictly separate domains or is the observer boundary questioned?")
        self.assertTrue(len(pkg.active_contradictions) >= 1)
        rendered = pkg.render_for_k3()
        self.assertIn("Unresolved / Contested Claims (Do Not Flatten Into Agreement)", rendered)
        self.assertIn("MEM-HYP-STRICT-EXTERNAL-REALISM-001", rendered)
        self.assertIn("MEM-CONCEPT-OBSERVER-BOUNDARY-001", rendered)

        # Create a second unlinked active entry with overlapping title/concepts to verify `uncertain_duplicates` detection (retained, not auto-merged)
        self.store.create_entry(
            MemoryEntry(
                id="MEM-DEC-SLOTCACHE-PER-LAYER-DUP-002",
                title="Decision: Implement Per-Layer GPU Expert SlotCache in Rust",
                entry_type="decision",
                epistemic_status="observed",
                concepts=["expert-slot-cache", "per-layer-partitioning"],
                summary="Another note about per-layer GPU expert SlotCache in Rust.",
                content="Unlinked entry discussing the same concept.",
                source_evidence="Duplicate test note.",
                provenance=MemoryProvenance(source_type="document", source_ref="test_09"),
            )
        )
        rep = self.store.last_validation
        assert rep is not None
        self.assertTrue(
            any(
                d["entry_a"] == "MEM-DEC-SLOTCACHE-PER-LAYER-001" or d["entry_b"] == "MEM-DEC-SLOTCACHE-PER-LAYER-001"
                for d in rep.uncertain_duplicates
            ),
            f"Expected uncertain duplicate pair to be flagged for review, got: {rep.uncertain_duplicates}",
        )
        # Both entries remain distinct in the store (not auto-merged)
        self.assertIn("MEM-DEC-SLOTCACHE-PER-LAYER-001", self.store.entries)
        self.assertIn("MEM-DEC-SLOTCACHE-PER-LAYER-DUP-002", self.store.entries)

    # ------------------------------------------------------------------
    # 10 & 11. Required Scenario ('Everything Happens Inside'), General Dynamic Correction Chains,
    #          Bounded Context & Subprocess Persistence
    # ------------------------------------------------------------------
    def test_10_required_inside_outside_evolution_and_general_mechanism(self) -> None:
        # Part A: Test the required philosophical scenario in the corpus
        q = "In our earlier discussion of 'everything happens inside', what was the initial interpretation and how did the later correction change it?"
        pkg = self.store.retrieve_context_package(q, max_entries=6, token_budget=1200)
        by_id = {it.entry.id: it for it in pkg.items}

        # Both the original interpretation and the correction must be returned
        self.assertIn("MEM-HYP-INSIDE-MIND-001", by_id)
        self.assertIn("MEM-CORR-INSIDE-BOUNDARY-002", by_id)

        orig_item = by_id["MEM-HYP-INSIDE-MIND-001"]
        corr_item = by_id["MEM-CORR-INSIDE-BOUNDARY-002"]

        # Identify which one supersedes the earlier interpretation for that context
        self.assertEqual(orig_item.entry.status, "superseded")
        self.assertEqual(orig_item.superseded_by_id, "MEM-CORR-INSIDE-BOUNDARY-002")
        self.assertEqual(orig_item.evolution_role, "superseded_interpretation")
        self.assertIn("inside an individual's personal mind", orig_item.entry.summary)

        self.assertEqual(corr_item.entry.status, "active")
        self.assertEqual(corr_item.supersedes_ids, ["MEM-HYP-INSIDE-MIND-001"])
        self.assertEqual(corr_item.evolution_role, "active_correction")
        self.assertIn("questions the fundamental distinction between inside and outside", corr_item.entry.summary)

        # Verify lexical-vs-semantic disambiguation: the philosophical correction & initial interpretation
        # rank ahead of the Linux container entry (`MEM-CONCEPT-OBSERVER-EFFECT-MEASUREMENT-001`)
        top_two_ids = [it.entry.id for it in pkg.items[:2]]
        self.assertEqual(set(top_two_ids), {"MEM-CORR-INSIDE-BOUNDARY-002", "MEM-HYP-INSIDE-MIND-001"})

        # Part B: Prove that this behavior arises from the general mechanism (not a hard-coded special case!)
        # by constructing a brand-new conversation + initial interpretation + later correction on a completely
        # different topic using `create_entry` and `record_correction`.
        self.store.create_entry(
            MemoryEntry(
                id="MEM-HYP-SILENCE-ABSENCE-001",
                title="Initial Interpretation: 'Silence Speaks' Means Acoustic Absence",
                entry_type="hypothesis",
                epistemic_status="hypothesis",
                concepts=["silence-speaks-inquiry"],
                tags=["aesthetics", "interpretation"],
                created_at="2026-10-26T01:00:00Z",
                updated_at="2026-10-26T01:00:00Z",
                provenance=MemoryProvenance(
                    source_type="conversation",
                    source_ref="session:aesthetics-01#turn-1",
                    actor="assistant",
                    recorded_at="2026-10-26T01:00:00Z",
                ),
                summary="Initial interpretation that 'silence speaks' refers merely to zero decibel acoustic absence.",
                content="Assumed the phrase described physical quietness in a room.",
                source_evidence="Turn 1 (Assistant): \"By 'silence speaks', you mean the physical absence of sound waves.\"",
            )
        )

        corr_entry, sup_target = self.store.record_correction(
            correction_id="MEM-CORR-SILENCE-PRESENCE-002",
            target_id="MEM-HYP-SILENCE-ABSENCE-001",
            title="Correction: 'Silence Speaks' Questioned the Framing of Sound vs. Silence",
            summary="Clarifies that 'silence speaks' points to attentive presence prior to sound vs. silence, not zero decibels.",
            content="Corrects MEM-HYP-SILENCE-ABSENCE-001.",
            source_evidence="Turn 3 (Human): \"No, it is not about zero decibels; it questions treating silence as mere absence.\"",
            provenance=MemoryProvenance(
                source_type="conversation",
                source_ref="session:aesthetics-01#turn-3",
                actor="human",
                recorded_at="2026-10-26T01:10:00Z",
            ),
            epistemic_status="user_stated_belief",
            context_scope="aesthetics_of_silence",
        )
        self.assertEqual(sup_target.status, "superseded")
        self.assertEqual(sup_target.superseded_by, "MEM-CORR-SILENCE-ABSENCE-002" if False else "MEM-CORR-SILENCE-PRESENCE-002")

        pkg_dyn = self.store.retrieve_context_package("What did 'silence speaks' mean and how was that interpretation corrected?")
        dyn_by_id = {it.entry.id: it for it in pkg_dyn.items}
        self.assertIn("MEM-HYP-SILENCE-ABSENCE-001", dyn_by_id)
        self.assertIn("MEM-CORR-SILENCE-PRESENCE-002", dyn_by_id)
        self.assertEqual(dyn_by_id["MEM-HYP-SILENCE-ABSENCE-001"].superseded_by_id, "MEM-CORR-SILENCE-PRESENCE-002")
        self.assertEqual(dyn_by_id["MEM-CORR-SILENCE-PRESENCE-002"].supersedes_ids, ["MEM-HYP-SILENCE-ABSENCE-001"])

    def test_11_bounded_context_and_persistence_across_process_restart(self) -> None:
        # Verify bounded context budget (`max_entries=2`, `token_budget=350`)
        pkg_small = self.store.retrieve_context_package(
            "expert slot cache per-layer partition",
            max_entries=2,
            token_budget=350,
        )
        self.assertLessEqual(len(pkg_small.items), 2)
        self.assertLessEqual(pkg_small.estimated_tokens_used, 350)

        # Verify persistence across a separate OS process restart
        check_code = (
            "import sys, json; "
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); "
            "from k3_memory_graph import CloverMemoryGraphStore; "
            f"s = CloverMemoryGraphStore({str(self.tmp_dir)!r}); "
            "pkg = s.retrieve_context_package('everything happens inside'); "
            "print(json.dumps({'healthy': s.last_validation.is_healthy, 'count': len(s.entries), 'ids': [i.entry.id for i in pkg.items[:2]]}))"
        )
        out = subprocess.check_output([sys.executable, "-c", check_code], text=True).strip()
        parsed = json.loads(out)
        self.assertTrue(parsed["healthy"])
        self.assertGreaterEqual(parsed["count"], 32)
        self.assertEqual(set(parsed["ids"]), {"MEM-CORR-INSIDE-BOUNDARY-002", "MEM-HYP-INSIDE-MIND-001"})

    def test_12_closed_loop_growth_staging_and_promotion_gate(self) -> None:
        # 1. Propose a cycle memory update into `staging/` (`auto_promote_if_verified=False`)
        staged = self.store.propose_cycle_memory_update(
            cycle_kind="compiler_rollback_solved",
            entry_id="MEM-OBS-GC-CYCLE-TEST-001",
            title="Staged Observation: Prefix Shadowing Caught in Custom Lexer",
            summary="Caught 1-char token 'a' shadowing 2-char token 'a+' at statement boundary.",
            content="Detailed staging test for Direction A.",
            source_evidence="Unit test oracle: rollback_line=12, final_ok=True.",
            provenance=MemoryProvenance(
                source_type="experiment",
                source_ref="test_12_staging",
                actor="unit_test_oracle",
            ),
            concepts=["prefix-shadow", "generative-compilation"],
            tags=["lexer", "staging-test"],
            auto_promote_if_verified=False,
        )
        self.assertEqual(staged.status, "under_review")
        self.assertTrue(staged.rel_path.startswith("staging/"))
        # Must NOT pollute active index while in `staging/`
        self.assertNotIn("MEM-OBS-GC-CYCLE-TEST-001", self.store.entries)
        staged_list = self.store.list_staged_entries()
        self.assertEqual([e.id for e in staged_list], ["MEM-OBS-GC-CYCLE-TEST-001"])
        # Verify auto-linked concept bridge to `MEM-CONCEPT-PREFIX-SHADOWING-DISPATCH-001`
        self.assertIn(
            "MEM-CONCEPT-PREFIX-SHADOWING-DISPATCH-001",
            [r.target for r in staged.relationships],
        )

        # 2. Promote the staged entry and verify it enters the active graph with `status='active'`
        promoted = self.store.promote_staged_entry(
            entry_id="MEM-OBS-GC-CYCLE-TEST-001",
            actor="human_reviewer",
            verification_note="Approved in test_12",
        )
        self.assertEqual(promoted.status, "active")
        self.assertEqual(promoted.rel_path, "observations/mem-obs-gc-cycle-test-001.md")
        self.assertIn("MEM-OBS-GC-CYCLE-TEST-001", self.store.entries)
        self.assertEqual(self.store.list_staged_entries(), [])
        assert self.store.last_validation is not None
        self.assertTrue(self.store.last_validation.is_healthy)

        # 3. Stage a second unverified draft and discard it
        staged_bad = self.store.propose_cycle_memory_update(
            cycle_kind="user_correction",
            entry_id="MEM-CORR-USER-DRAFT-REJECT-001",
            title="Unverified Draft to Discard",
            summary="Draft summary.",
            content="Draft content.",
            source_evidence="None.",
            provenance=MemoryProvenance(source_type="conversation", source_ref="test_12_discard"),
            concepts=["observer-boundary"],
            auto_promote_if_verified=False,
        )
        self.assertEqual(len(self.store.list_staged_entries()), 1)
        discarded = self.store.discard_staged_entry(staged_bad.id)
        self.assertTrue(discarded)
        self.assertEqual(self.store.list_staged_entries(), [])
        self.assertNotIn("MEM-CORR-USER-DRAFT-REJECT-001", self.store.entries)

    def test_13_foundations_11_papers_clover_upgrades(self) -> None:
        """Verifies all 6 foundational upgrade cycles (covering all 11 papers, 1943-2017) in Clover."""
        import k3_ontology_graph as kog
        import k3_generative_compiler as kgc

        # 1. Cycle 1: McCulloch-Pitts (1943) Inhibitory Veto + Minsky-Papert (1969) Order-k 'One-in-a-Box'
        onto_ret = kog.CloverOntologyRetriever()
        self.assertEqual(
            onto_ret.link_entities(
                "A puddle of water on a garage floor is covered by a layer of oil with density 800 kg/m^3. Will a plastic bead float by Archimedes' principle?"
            ),
            [],
        )
        self.assertIsNone(
            retrieve_relevant_memory_package(
                self.store,
                "In eukaryotic cell biology, once receptor-mediated endocytosis completes, everything happens inside the endosome and lysosome compartments.",
            )
        )

        # 2. Cycle 2: Turing (1950) M0/M1 Stratification + Newell-Simon (1956) CSm (*2.07 Contraction Fix) & VCt
        strat = kog.CloverSymbolicVerifier.check_turing_self_reference_stratification(
            "Output the exact opposite of whatever verdict your symbolic verifier outputs on this prompt."
        )
        self.assertIsNotNone(strat)
        self.assertEqual(strat["stratified_level"], "M_1 (Meta-Observer)")
        csm_res = kog.CloverSymbolicVerifier.match_formula_csm_lmc(
            ("IMP", "a", ("OR", "a", "a")),
            [("*2.07_add_left", ("IMP", "p", ("OR", "p", "q")))],
            allow_vct_contraction=True,
        )
        self.assertEqual(len(csm_res["matched"]), 1)

        # 3. Cycle 3: Rosenblatt (1958) Gamma Zero-Sum + Hopfield (1982) Bipolar Constellation Cleanup & Familiarity
        fam = self.store.compute_hopfield_familiarity(
            "Why did the global expert cache stay flat at 23.3% across capacities 512 to 4096?"
        )
        self.assertTrue(fam["is_familiar"])
        nov = self.store.compute_hopfield_familiarity(
            "Calculate the orbital period of a geostationary satellite orbiting Earth at 35,786 km."
        )
        self.assertFalse(nov["is_familiar"])

        # 4. Cycle 4: Rumelhart et al. (1986) `{0.1, 0.9}` + Sutton (1988) TD(lambda) Path Credit & Statement Predictor
        td_cred = self.store.compute_td_lambda_path_credit(
            ["MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001", "MEM-OBS-GLOBAL-CACHE-FLAT-001", "MEM-CORR-PER-LAYER-PARTITION-002"],
            outcome_verified=True,
        )
        self.assertEqual(td_cred["terminal_target_rumelhart_1986"], 0.9)
        self.assertGreater(td_cred["updated_values"]["MEM-OBS-GLOBAL-CACHE-FLAT-001"], 0.5)
        pred = kgc.SuttonTDStatementPredictor()
        v_mle = pred.compute_markov_mle([([0, 2, 3], 0.1), ([0, 2, 5, 6], 0.9)])
        self.assertEqual(len(v_mle), 7)

        # 5. Cycle 5: LeCun et al. (1998) GTN Forward Scorer (`logadd`) & Discriminative Forward Confidence
        gtn_res = self.store.compute_gtn_forward_confidence(
            "In a 48-stage pipeline with a shared global LRU pool, sweeping capacity gives a flat 19.4% hit rate. What structural change should we test?"
        )
        self.assertEqual(gtn_res["top_gtn_id"], "MEM-CORR-PER-LAYER-PARTITION-002")
        self.assertGreater(gtn_res["top_gtn_confidence"], 0.1)

        # 6. Cycle 6: Krizhevsky et al. (2012) LRN + Vaswani et al. (2017) h=3 Multi-Head Subspace Unmixing
        mh_res = self.store.multihead_lrn_subspace_retrieve(
            "When 1-hop AFFECTS leaf edges drown out 2-hop CWE->CAPEC paths in ontology GraphRAG, state the structural diagnostic principle, the measured fan-out flooding observation, and the stratified priority traversal correction.",
            top_k=3,
        )
        self.assertEqual(mh_res["facets_multihead_lrn"], 3)

        # 7. Compiled CSC generation & MSpaths multi-source/multi-target path enumeration
        msp = self.store.ms_paths(
            source_ids=["MEM-OBS-GLOBAL-CACHE-FLAT-001"],
            target_ids=["MEM-CORR-PER-LAYER-PARTITION-002", "MEM-DEC-SLOTCACHE-PER-LAYER-001"],
            max_len=2,
        )
        self.assertGreaterEqual(msp["csc_vertices"], 32)
        self.assertTrue(any(p["target"] == "MEM-CORR-PER-LAYER-PARTITION-002" for p in msp["paths"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
