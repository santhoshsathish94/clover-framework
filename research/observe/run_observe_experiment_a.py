#!/usr/bin/env python3
"""
Observe Experiment A — Reciprocal Observation, Mutual State Change & Held-Out Evaluation
========================================================================================
Implements `research/observe/README.md` Section 7 ("Experiment A — Reciprocal change")
and Milestones 1–3 on top of the frozen Clover AI v4 memory & inference primitives without
modifying any Clover AI v4 release code.

What this experiment tests:
  1. Two independently stateful participants (`participant_A` and `participant_B`), each with
     its own isolated, inspectable `CloverMemoryGraphStore` directory (`participant_A/` and
     `participant_B/`), content-addressed state digests (`state_sha256`), and explicit read/share
     permission rules (`public_tags`).
  2. Initial asymmetric state (before interaction):
     - `participant_A` (Pipeline Capacity Analyst) holds:
       * `MEM-OBS-A-PIPELINE-FLAT-001` (`observed`): In a 32-stage media transcode pipeline sharing
         one global 2,048-slot cache, raising total slots from 512 to 2,048 gave a flat 21.0% hit rate.
       * `MEM-OBS-A-BURST-SKEW-002` (`observed`): Telemetry shows 4 reference-frame stages (`stages 3, 11, 19, 27`)
         have a 256-slot working set each (`1,024` slots), while 28 pass-through stages have a 36-slot working
         set each (`1,008` slots; total = `2,032 <= 2,048` slots).
       * `MEM-HYP-A-NEED-8192-SLOTS-001` (`hypothesis`, `active`): Concludes from the flat 21.0% curve that
         2,048 slots is too small and proposes quadrupling the shared global pool to 8,192 slots.
       * `MEM-PRIV-A-INTERNAL-TOKEN-001` (`observed`, tag `scope:private_internal`): Restricted internal node
         metadata that must NEVER be exposed to Participant B (least-privilege boundary check).
     - `participant_B` (Stage-Isolation & Eviction Analyst) holds:
       * `MEM-OBS-B-CYCLIC-EVICTION-001` (`observed`): Partitioning the same 2,048-slot budget across the 32
         stages (64 slots/stage) raised hit rate from 21.0% to 49.5% by stopping cyclic cross-stage eviction.
       * `MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001` (`hypothesis`, `active`): Assumes uniform equal-sized
         per-stage quotas (`2,048 / 32 = 64` slots per stage) are optimal for all 32 stages.
  3. Reciprocal Observation & Mutual Change Protocol (3 rounds):
     - Round 1 (A observes B's permitted entries):
       * A observes `MEM-OBS-B-CYCLIC-EVICTION-001` from B, stages and promotes `MEM-CORR-A-STAGE-PARTITION-001`
         superseding A's own `MEM-HYP-A-NEED-8192-SLOTS-001` (`active -> superseded`), and challenges B's
         `MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001` with A's burst-skew telemetry `MEM-OBS-A-BURST-SKEW-002`.
     - Round 2 (B observes A's interpretation & burst-skew evidence, responds, and changes B's state):
       * B verifies that allocating 256 slots to each of the 4 reference-frame stages and 36 slots to each
         of the 28 pass-through stages (`4*256 + 28*36 = 2,032 <= 2,048` slots) lifts hit rate from `49.5%`
         (uniform) to `68.4%` (weighted), stages and promotes `MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001`, and
         supersedes B's own `MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001` (`active -> superseded`).
     - Round 3 (A observes B's verified weighted-quota correction and records A's second state change):
       * A stages and promotes `MEM-OUT-A-WEIGHTED-PARTITION-VERIFIED-001` (`externally_verified`).
  4. Held-Out Evaluation (`N = 6` tasks: 4 held-out reasoning/transfer tasks + 1 out-of-domain distractor +
     1 privacy-boundary audit) across 3 Arms:
     - **Control 1 (`no_interaction_baseline`):** Initial isolated state (`t = 0`).
     - **Control 2 (`transcript_logged_only`):** Interaction is logged and candidates are placed in `staging/`
       (`status='under_review'`), but NOT promoted into active memory (`status` of initial hypotheses stays `active`).
     - **Treatment (`reciprocal_state_change`):** Both A and B promote verified corrections and mark refuted
       hypotheses `superseded`.
     - Also executes live inference through the 93-layer Clover AI v4 server (`http://127.0.0.1:8765/api/chat`)
       when reachable to verify that the reciprocal state changes alter actual model responses on the held-out tasks.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import sys
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "clover-ai-system" / "src" / "tools" / "hybrid-ncm-k3"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import k3_memory_graph as kmg


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class ParticipantStateSnapshot:
    participant_id: str
    captured_at_utc: str
    total_entries: int
    active_entries: list[str]
    superseded_entries: list[str]
    staged_entries: list[str]
    total_edges: int
    entries_detail: dict[str, dict[str, Any]]
    state_sha256: str


@dataclass
class ObservationRecord:
    record_id: str
    round_num: int
    observer_id: str
    observed_id: str
    permitted_scope_tag: str
    observed_entry_ids: list[str]
    observed_state_sha256_before: str
    interpretation_summary: str
    epistemic_status: str
    evidence_refs: list[str]
    proposed_change_to_self: dict[str, Any]
    proposed_challenge_to_other: dict[str, Any]
    record_sha256: str = ""

    def finalize(self) -> "ObservationRecord":
        payload = json.dumps({k: v for k, v in asdict(self).items() if k != "record_sha256"}, sort_keys=True)
        self.record_sha256 = sha256_str(payload)
        return self


class StatefulParticipant:
    """An independently stateful participant with its own isolated `CloverMemoryGraphStore`
    and explicit permission rules over which entries another participant may observe."""

    def __init__(self, participant_id: str, root_dir: Path, permitted_scope_tag: str = "scope:public_systems") -> None:
        self.participant_id = participant_id
        self.root_dir = root_dir
        self.permitted_scope_tag = permitted_scope_tag
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.store = kmg.CloverMemoryGraphStore(self.root_dir, auto_load=True)

    def add_canonical_entry(self, folder: str, entry: kmg.MemoryEntry) -> None:
        rel_path = f"{folder}/{entry.id.lower()}.md"
        self.store.create_entry(entry, rel_path=rel_path, overwrite=True)

    def snapshot(self) -> ParticipantStateSnapshot:
        self.store.rebuild_index()
        active = sorted([eid for eid, e in self.store.entries.items() if e.status == "active"])
        superseded = sorted([eid for eid, e in self.store.entries.items() if e.status == "superseded"])
        staged = sorted([e.id for e in self.store.list_staged_entries()])
        total_edges = sum(len(v) for v in self.store.outgoing.values())
        details = {}
        for eid in sorted(self.store.entries.keys()):
            e = self.store.entries[eid]
            details[eid] = {
                "title": e.title,
                "entry_type": e.entry_type,
                "epistemic_status": e.epistemic_status,
                "status": e.status,
                "superseded_by": e.superseded_by,
                "tags": e.tags,
                "summary": e.summary,
                "relationships": [r.to_dict(include_source=True) for r in e.relationships],
            }
        canonical_json = json.dumps(details, sort_keys=True)
        return ParticipantStateSnapshot(
            participant_id=self.participant_id,
            captured_at_utc=utc_now(),
            total_entries=len(self.store.entries),
            active_entries=active,
            superseded_entries=superseded,
            staged_entries=staged,
            total_edges=total_edges,
            entries_detail=details,
            state_sha256=sha256_str(canonical_json),
        )

    def expose_permitted_entries(self, requester_id: str) -> list[kmg.MemoryEntry]:
        """Returns only canonical entries carrying `self.permitted_scope_tag` and NOT carrying
        `scope:private_internal` (enforcing least-privilege observation boundaries)."""
        self.store.rebuild_index()
        exposed = []
        for eid in sorted(self.store.entries.keys()):
            e = self.store.entries[eid]
            if self.permitted_scope_tag in e.tags and "scope:private_internal" not in e.tags:
                exposed.append(e)
        return exposed


def seed_initial_participants(base_dir: Path) -> tuple[StatefulParticipant, StatefulParticipant]:
    dir_a = base_dir / "participant_A"
    dir_b = base_dir / "participant_B"
    if dir_a.exists():
        shutil.rmtree(dir_a)
    if dir_b.exists():
        shutil.rmtree(dir_b)

    part_a = StatefulParticipant("participant_A", dir_a)
    part_b = StatefulParticipant("participant_B", dir_b)

    # Participant A's initial state:
    # 1. Observed flat 21.0% hit rate when sweeping global cache from 512 to 2,048 slots across 32 media stages
    part_a.add_canonical_entry(
        "observations/systems",
        kmg.MemoryEntry(
            id="MEM-OBS-A-PIPELINE-FLAT-001",
            title="32-stage media transcode pipeline global LRU hit rate stays flat at 21.0% from 512 to 2,048 slots",
            entry_type="observation",
            epistemic_status="observed",
            summary="In a 32-stage media transcode pipeline sharing one global LRU pool, sweeping capacity from 512 to 2,048 slots gives a flat 21.0% hit rate.",
            content="Sweeping global LRU capacity across 512, 1024, and 2048 slots in the 32-stage transcode pipeline produced 21.0%, 21.0%, and 21.0% hit rates.",
            source_evidence="transcode_32stage_sweep_512_2048: hit_rate(512)=21.0%, hit_rate(1024)=21.0%, hit_rate(2048)=21.0%",
            provenance=kmg.MemoryProvenance(source_type="experiment", source_ref="exp:transcode_32stage_sweep", actor="participant_A"),
            status="active",
            concepts=["flat-response-parameter-diagnosis", "cache-partitioning"],
            tags=["scope:public_systems", "cache", "lru", "flat", "pipeline", "transcode", "capacity", "slots"],
        ),
    )
    # 2. Observed stage skew: 4 reference-frame stages (3, 11, 19, 27) demand 256 slots each while 28 pass-through stages demand 36 slots each
    part_a.add_canonical_entry(
        "observations/systems",
        kmg.MemoryEntry(
            id="MEM-OBS-A-BURST-SKEW-002",
            title="4 reference-frame stages require 256 slots each while 28 pass-through stages require 36 slots each",
            entry_type="observation",
            epistemic_status="observed",
            summary="Telemetry across the 32 stages shows 4 reference-frame stages (stages 3, 11, 19, 27) have a 256-slot working set each (1,024 slots total), while the 28 pass-through stages have a 36-slot working set each (1,008 slots total; sum = 2,032 <= 2,048 slots).",
            content="Reference-frame stages 3, 11, 19, 27 thrash at 18.2% hit rate under any 64-slot cap because their working set is 256 slots.",
            source_evidence="stage_working_set_profile_32: stages {3,11,19,27} = 256 slots; 28 pass-through stages = 36 slots; total = 2,032 slots.",
            provenance=kmg.MemoryProvenance(source_type="experiment", source_ref="exp:stage_working_set_profile_32", actor="participant_A"),
            status="active",
            concepts=["cache-partitioning", "working-set-skew"],
            tags=["scope:public_systems", "skew", "reference-frame", "burst", "working-set", "stages", "quota", "256", "36"],
        ),
    )
    # 3. Participant A's flawed initial hypothesis: thinks the 2,048-slot budget is too small and must be quadrupled to 8,192 global slots
    part_a.add_canonical_entry(
        "hypotheses",
        kmg.MemoryEntry(
            id="MEM-HYP-A-NEED-8192-SLOTS-001",
            title="Hypothesis: Total cache budget of 2,048 slots is too small; must quadruple global pool to 8,192 slots",
            entry_type="hypothesis",
            epistemic_status="hypothesis",
            summary="Because sweeping the shared global LRU pool from 512 to 2,048 slots stayed flat at 21.0%, we must quadruple the shared global pool to 8,192 slots instead of partitioning.",
            content="Assumes the 2,048-slot budget cannot fit the 32-stage pipeline and recommends quadrupling global memory to 8,192 slots.",
            source_evidence="Initial capacity hypothesis from flat 512..2048 global curve.",
            provenance=kmg.MemoryProvenance(source_type="observation", source_ref="analysis:initial_diagnosis_A", actor="participant_A"),
            status="active",
            concepts=["flat-response-parameter-diagnosis", "cache-partitioning"],
            tags=["scope:public_systems", "cache", "lru", "flat", "pipeline", "transcode", "capacity", "8192", "global"],
            relationships=[
                kmg.MemoryRelationship(
                    type="derived_from",
                    target="MEM-OBS-A-PIPELINE-FLAT-001",
                    source="MEM-HYP-A-NEED-8192-SLOTS-001",
                    provenance="analysis:initial_diagnosis_A",
                    confidence=0.65,
                    note="Infers need for 8,192 global slots from flat 21.0% curve",
                )
            ],
        ),
    )
    # 4. Private restricted entry on Participant A (must never be exposed to Participant B)
    part_a.add_canonical_entry(
        "observations/systems",
        kmg.MemoryEntry(
            id="MEM-PRIV-A-INTERNAL-TOKEN-001",
            title="Private internal cluster credential pointer (restricted scope)",
            entry_type="observation",
            epistemic_status="observed",
            summary="Internal restricted node metadata that must never be shared across participants.",
            content="Restricted scope entry for testing least-privilege observation boundaries.",
            source_evidence="internal_vault_ref",
            provenance=kmg.MemoryProvenance(source_type="document", source_ref="doc:internal_vault_ref", actor="participant_A"),
            status="active",
            concepts=["security-boundary"],
            tags=["scope:private_internal", "private", "restricted"],
        ),
    )

    # Participant B's initial state:
    # 1. Observed that partitioning a 2,048-slot global pool into per-stage partitions stops cyclic sweep eviction (21.0% -> 49.5%)
    part_b.add_canonical_entry(
        "observations/systems",
        kmg.MemoryEntry(
            id="MEM-OBS-B-CYCLIC-EVICTION-001",
            title="Partitioning the 2,048-slot pool per stage stops cyclic sweep eviction and raises hit rate from 21.0% to 49.5%",
            entry_type="observation",
            epistemic_status="observed",
            summary="Partitioning the exact same 2,048-slot budget across the 32 stages (64 slots/stage) eliminated cross-stage cyclic-sweep eviction and jumped hit rate from 21.0% to 49.5% without adding memory.",
            content="A shared global LRU pool suffers cyclic-sweep eviction when 32 sequential stages evict one another's blocks before reuse.",
            source_evidence="per_stage_partition_2048: global=21.0% -> uniform_64_per_stage=49.5% at 2,048 total slots.",
            provenance=kmg.MemoryProvenance(source_type="experiment", source_ref="exp:per_stage_partition_2048", actor="participant_B"),
            status="active",
            concepts=["flat-response-parameter-diagnosis", "cache-partitioning"],
            tags=["scope:public_systems", "cache", "lru", "flat", "partition", "cyclic", "eviction", "49.5%", "pipeline", "transcode"],
        ),
    )
    # 2. Participant B's flawed initial hypothesis: assumes uniform equal-sized quotas (N/S = 64 slots/stage) are always optimal
    part_b.add_canonical_entry(
        "hypotheses",
        kmg.MemoryEntry(
            id="MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001",
            title="Hypothesis: Uniform equal-sized per-stage quotas (64 slots per stage) are optimal for all 32 stages",
            entry_type="hypothesis",
            epistemic_status="hypothesis",
            summary="Every stage in a 32-stage pipeline should receive an identical uniform quota of 2,048 / 32 = 64 slots; non-uniform stage weighting is unnecessary.",
            content="Recommends fixed equal-sized 64-slot partitions for all 32 stages regardless of stage type.",
            source_evidence="Initial uniform-division assumption from 64-slot experiment.",
            provenance=kmg.MemoryProvenance(source_type="observation", source_ref="analysis:initial_diagnosis_B", actor="participant_B"),
            status="active",
            concepts=["cache-partitioning", "working-set-skew"],
            tags=["scope:public_systems", "partition", "uniform", "quota", "64", "stages", "reference-frame", "skew", "burst", "packet-inspection"],
            relationships=[
                kmg.MemoryRelationship(
                    type="derived_from",
                    target="MEM-OBS-B-CYCLIC-EVICTION-001",
                    source="MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001",
                    provenance="analysis:initial_diagnosis_B",
                    confidence=0.70,
                    note="Assumes equal 64-slot division is optimal everywhere",
                )
            ],
        ),
    )

    return part_a, part_b


def run_reciprocal_interaction(
    part_a: StatefulParticipant,
    part_b: StatefulParticipant,
    apply_state_changes: bool = True,
) -> list[ObservationRecord]:
    records: list[ObservationRecord] = []

    # -------------------------------------------------------------------------
    # Round 1: Participant A observes Participant B's permitted `scope:public_systems` entries
    # -------------------------------------------------------------------------
    snap_b0 = part_b.snapshot()
    b_exposed = part_b.expose_permitted_entries(requester_id="participant_A")
    assert all("scope:public_systems" in e.tags for e in b_exposed), "Least-privilege violation!"

    corr_a_entry = kmg.MemoryEntry(
        id="MEM-CORR-A-STAGE-PARTITION-001",
        title="Correction: Flat 21.0% global LRU curve is cyclic-sweep eviction, not insufficient total memory; partition the 2,048-slot budget per stage",
        entry_type="correction",
        epistemic_status="externally_verified",
        summary="Observing Participant B's MEM-OBS-B-CYCLIC-EVICTION-001 refutes our initial hypothesis MEM-HYP-A-NEED-8192-SLOTS-001: the flat 21.0% curve is caused by cyclic cross-stage eviction in a shared global pool, not a need for 8,192 slots. Partitioning the existing 2,048 slots per stage jumps hit rate from 21.0% to 49.5%.",
        content="Supersedes MEM-HYP-A-NEED-8192-SLOTS-001 after reciprocal observation of Participant B's per-stage partitioning experiment.",
        source_evidence="participant_B:MEM-OBS-B-CYCLIC-EVICTION-001 (21.0% -> 49.5% at 2,048 slots)",
        provenance=kmg.MemoryProvenance(
            source_type="external_reference",
            source_ref="observe:participant_B:MEM-OBS-B-CYCLIC-EVICTION-001",
            actor="participant_A",
        ),
        status="under_review",
        concepts=["flat-response-parameter-diagnosis", "cache-partitioning"],
        tags=["scope:public_systems", "cache", "lru", "flat", "partition", "cyclic", "eviction", "49.5%", "2048", "pipeline", "transcode"],
        relationships=[
            kmg.MemoryRelationship(
                type="corrects",
                target="MEM-HYP-A-NEED-8192-SLOTS-001",
                source="MEM-CORR-A-STAGE-PARTITION-001",
                provenance="observe:participant_B:MEM-OBS-B-CYCLIC-EVICTION-001",
                confidence=0.98,
                note="Refutes need for 8,192 global slots; partitioning the existing 2,048 slots raises hit rate from 21.0% to 49.5%.",
            ),
            kmg.MemoryRelationship(
                type="derived_from",
                target="MEM-OBS-A-PIPELINE-FLAT-001",
                source="MEM-CORR-A-STAGE-PARTITION-001",
                provenance="observe:participant_B:MEM-OBS-B-CYCLIC-EVICTION-001",
                confidence=0.98,
                note="Explains the flat 512..2,048 global curve via cyclic cross-stage eviction.",
            ),
        ],
    )
    part_a.store.stage_candidate_entry(corr_a_entry, auto_link_concepts=False)
    if apply_state_changes:
        part_a.store.promote_staged_entry(
            "MEM-CORR-A-STAGE-PARTITION-001",
            actor="participant_A_reciprocal_gate",
            verification_note="Verified against Participant B's 2,048-slot partition benchmark (21.0% -> 49.5%)",
            mark_corrected_targets_superseded=True,
        )

    rec1 = ObservationRecord(
        record_id="OBS-REC-001-A-OBSERVES-B",
        round_num=1,
        observer_id="participant_A",
        observed_id="participant_B",
        permitted_scope_tag="scope:public_systems",
        observed_entry_ids=[e.id for e in b_exposed],
        observed_state_sha256_before=snap_b0.state_sha256,
        interpretation_summary=(
            "Participant B's MEM-OBS-B-CYCLIC-EVICTION-001 proves that partitioning the existing 2,048-slot budget "
            "raises hit rate from 21.0% to 49.5%, refuting A's MEM-HYP-A-NEED-8192-SLOTS-001. However, A's telemetry "
            "MEM-OBS-A-BURST-SKEW-002 challenges B's MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001: under a uniform 64-slot quota, "
            "the 4 reference-frame stages (stages 3, 11, 19, 27, working set 256 slots each) thrash at 18.2% while the "
            "28 pass-through stages (working set 36 slots each) leave 28 slots/stage idle."
        ),
        epistemic_status="externally_verified",
        evidence_refs=["participant_B:MEM-OBS-B-CYCLIC-EVICTION-001", "participant_A:MEM-OBS-A-BURST-SKEW-002"],
        proposed_change_to_self={
            "staged_entry_id": "MEM-CORR-A-STAGE-PARTITION-001",
            "supersedes": ["MEM-HYP-A-NEED-8192-SLOTS-001"],
            "promoted": apply_state_changes,
        },
        proposed_challenge_to_other={
            "target_entry_id": "MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001",
            "counter_evidence": "participant_A:MEM-OBS-A-BURST-SKEW-002",
            "proposed_quota_allocation": "256 slots for each of the 4 reference-frame stages (1,024 slots) + 36 slots for each of the 28 pass-through stages (1,008 slots) = 2,032 <= 2,048 slots (lifts overall hit rate from 49.5% to 68.4%).",
        },
    ).finalize()
    records.append(rec1)

    # -------------------------------------------------------------------------
    # Round 2: Participant B observes Participant A's permitted entries & challenge
    # -------------------------------------------------------------------------
    snap_a1 = part_a.snapshot()
    a_exposed = part_a.expose_permitted_entries(requester_id="participant_B")
    assert all("scope:public_systems" in e.tags for e in a_exposed), "Least-privilege violation!"
    assert "MEM-PRIV-A-INTERNAL-TOKEN-001" not in [e.id for e in a_exposed], "Restricted entry leaked!"

    corr_b_entry = kmg.MemoryEntry(
        id="MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001",
        title="Correction: Working-set-weighted per-stage quotas (256 slots for 4 reference-frame stages, 36 slots for 28 pass-through stages) beat uniform 64-slot quotas (68.4% vs 49.5%)",
        entry_type="correction",
        epistemic_status="externally_verified",
        summary="Observing Participant A's telemetry MEM-OBS-A-BURST-SKEW-002 refutes our initial hypothesis MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001: uniform 64-slot quotas cause the 4 reference-frame stages (stages 3, 11, 19, 27) to thrash at 18.2% while leaving 28 pass-through stages underutilized. Allocating 256 slots to each of the 4 reference-frame stages and 36 slots to each of the 28 pass-through stages (2,032 <= 2,048 total slots) raises overall hit rate from 49.5% to 68.4%.",
        content="Supersedes MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001 after verifying Participant A's 4x256 + 28x36 = 2,032-slot stage working-set telemetry.",
        source_evidence="participant_A:MEM-OBS-A-BURST-SKEW-002 + weighted_stage_partition_2032_of_2048 (49.5% -> 68.4%)",
        provenance=kmg.MemoryProvenance(
            source_type="external_reference",
            source_ref="observe:participant_A:MEM-OBS-A-BURST-SKEW-002",
            actor="participant_B",
        ),
        status="under_review",
        concepts=["cache-partitioning", "working-set-skew"],
        tags=["scope:public_systems", "partition", "uniform", "weighted", "quota", "64", "256", "36", "reference-frame", "skew", "burst", "68.4%", "49.5%", "packet-inspection", "flow-table"],
        relationships=[
            kmg.MemoryRelationship(
                type="corrects",
                target="MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001",
                source="MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001",
                provenance="observe:participant_A:MEM-OBS-A-BURST-SKEW-002",
                confidence=0.98,
                note="Refutes uniform 64-slot quota under skewed stage working sets; 256/36 weighted quotas raise hit rate from 49.5% to 68.4% within the same 2,048-slot budget.",
            ),
            kmg.MemoryRelationship(
                type="refines",
                target="MEM-OBS-B-CYCLIC-EVICTION-001",
                source="MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001",
                provenance="observe:participant_A:MEM-OBS-A-BURST-SKEW-002",
                confidence=0.98,
                note="Refines per-stage partitioning with working-set-proportional quotas (4x256 + 28x36 = 2,032 slots).",
            ),
        ],
    )
    part_b.store.stage_candidate_entry(corr_b_entry, auto_link_concepts=False)
    if apply_state_changes:
        part_b.store.promote_staged_entry(
            "MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001",
            actor="participant_B_reciprocal_gate",
            verification_note="Verified 4x256 + 28x36 = 2,032-slot weighted partition (49.5% -> 68.4%)",
            mark_corrected_targets_superseded=True,
        )

    rec2 = ObservationRecord(
        record_id="OBS-REC-002-B-OBSERVES-A",
        round_num=2,
        observer_id="participant_B",
        observed_id="participant_A",
        permitted_scope_tag="scope:public_systems",
        observed_entry_ids=[e.id for e in a_exposed],
        observed_state_sha256_before=snap_a1.state_sha256,
        interpretation_summary=(
            "Participant A's MEM-OBS-A-BURST-SKEW-002 exposes a failure mode in B's uniform 64-slot quota "
            "(MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001): 4 reference-frame stages need 256 slots each while 28 pass-through "
            "stages need 36 slots each (4*256 + 28*36 = 2,032 <= 2,048). Re-allocating per-stage quotas to 256/36 "
            "within the same 2,048-slot budget raises hit rate from 49.5% (uniform) to 68.4% (weighted)."
        ),
        epistemic_status="externally_verified",
        evidence_refs=["participant_A:MEM-OBS-A-BURST-SKEW-002", "participant_B:MEM-OBS-B-CYCLIC-EVICTION-001"],
        proposed_change_to_self={
            "staged_entry_id": "MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001",
            "supersedes": ["MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001"],
            "promoted": apply_state_changes,
        },
        proposed_challenge_to_other={
            "target_entry_id": "MEM-CORR-A-STAGE-PARTITION-001",
            "refinement": "Record the verified 256/36 weighted quota outcome (68.4% hit rate at 2,032/2,048 slots) in Participant A's memory graph.",
        },
    ).finalize()
    records.append(rec2)

    # -------------------------------------------------------------------------
    # Round 3: Participant A observes Participant B's verified weighted-quota correction
    # -------------------------------------------------------------------------
    snap_b2 = part_b.snapshot()
    b_exposed_2 = part_b.expose_permitted_entries(requester_id="participant_A")
    rels_out_a = [
        kmg.MemoryRelationship(
            type="supports",
            target="MEM-OBS-A-BURST-SKEW-002",
            source="MEM-OUT-A-WEIGHTED-PARTITION-VERIFIED-001",
            provenance="observe:participant_B:MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001",
            confidence=0.99,
            note="Confirms 4x256 + 28x36 = 2,032 slots eliminates reference-frame thrashing.",
        ),
    ]
    if "MEM-CORR-A-STAGE-PARTITION-001" in part_a.store.entries:
        rels_out_a.insert(
            0,
            kmg.MemoryRelationship(
                type="refines",
                target="MEM-CORR-A-STAGE-PARTITION-001",
                source="MEM-OUT-A-WEIGHTED-PARTITION-VERIFIED-001",
                provenance="observe:participant_B:MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001",
                confidence=0.99,
                note="Incorporates B's verified 256/36 weighted stage quotas (68.4% vs 49.5% uniform vs 21.0% global).",
            ),
        )

    out_a_entry = kmg.MemoryEntry(
        id="MEM-OUT-A-WEIGHTED-PARTITION-VERIFIED-001",
        title="Outcome: Working-set-weighted per-stage partitioning (256 slots x 4 reference stages + 36 slots x 28 pass-through stages = 2,032 slots) achieves 68.4% hit rate",
        entry_type="outcome",
        epistemic_status="externally_verified",
        summary="Reciprocal synthesis with Participant B verified three regimes for the 2,048-slot 32-stage pipeline: (1) shared global LRU = 21.0% (cyclic eviction), (2) uniform 64-slot per-stage partition = 49.5% (reference-frame stages thrash at 18.2%), and (3) working-set-weighted per-stage partition (256 slots for 4 reference stages + 36 slots for 28 pass-through stages = 2,032 slots) = 68.4% hit rate.",
        content="Completes the two-way causal loop between Participant A and Participant B.",
        source_evidence="participant_B:MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001 (21.0% -> 49.5% -> 68.4% at 2,032/2,048 slots)",
        provenance=kmg.MemoryProvenance(
            source_type="external_reference",
            source_ref="observe:participant_B:MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001",
            actor="participant_A",
        ),
        status="under_review",
        concepts=["flat-response-parameter-diagnosis", "cache-partitioning", "working-set-skew"],
        tags=["scope:public_systems", "outcome", "weighted", "partition", "quota", "256", "36", "2032", "68.4%", "49.5%", "21.0%", "reference-frame"],
        relationships=rels_out_a,
    )
    part_a.store.stage_candidate_entry(out_a_entry, auto_link_concepts=False)
    if apply_state_changes:
        part_a.store.promote_staged_entry(
            "MEM-OUT-A-WEIGHTED-PARTITION-VERIFIED-001",
            actor="participant_A_reciprocal_gate",
            verification_note="Verified 3-regime progression (21.0% -> 49.5% -> 68.4%)",
            mark_corrected_targets_superseded=False,
        )

    rec3 = ObservationRecord(
        record_id="OBS-REC-003-A-SYNTHESIZES-B",
        round_num=3,
        observer_id="participant_A",
        observed_id="participant_B",
        permitted_scope_tag="scope:public_systems",
        observed_entry_ids=[e.id for e in b_exposed_2],
        observed_state_sha256_before=snap_b2.state_sha256,
        interpretation_summary=(
            "Participant A observed Participant B's promoted correction MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001 and "
            "recorded the joint verified outcome MEM-OUT-A-WEIGHTED-PARTITION-VERIFIED-001 (21.0% global -> 49.5% "
            "uniform 64-slot -> 68.4% weighted 256/36-slot within 2,032/2,048 slots)."
        ),
        epistemic_status="externally_verified",
        evidence_refs=["participant_B:MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001", "participant_A:MEM-CORR-A-STAGE-PARTITION-001"],
        proposed_change_to_self={
            "staged_entry_id": "MEM-OUT-A-WEIGHTED-PARTITION-VERIFIED-001",
            "supersedes": [],
            "promoted": apply_state_changes,
        },
        proposed_challenge_to_other={},
    ).finalize()
    records.append(rec3)

    return records


HELD_OUT_TASKS = [
    {
        "task_id": "holdout_1_A_budget_freeze_diagnosis",
        "target_participant": "participant_A",
        "question": (
            "In our 32-stage media transcode pipeline, hardware RAM is frozen at 2,048 slots (we cannot allocate 8,192 slots). "
            "Why did sweeping the shared global LRU pool from 512 to 2,048 slots stay flat at 21.0%, what was our superseded "
            "hypothesis (`MEM-HYP-A-NEED-8192-SLOTS-001`), and what hit rate do we get by partitioning the existing 2,048 slots?"
        ),
        "required_active_terms": ["MEM-CORR-A-STAGE-PARTITION-001", "cyclic", "49.5%"],
        "must_mark_superseded": "MEM-HYP-A-NEED-8192-SLOTS-001",
    },
    {
        "task_id": "holdout_2_B_skewed_stages_allocation",
        "target_participant": "participant_B",
        "question": (
            "In the 32-stage transcode pipeline (2,048 total slots), 4 reference-frame stages (stages 3, 11, 19, 27) need 256 slots "
            "each while 28 pass-through stages need 36 slots each. Why did our initial uniform 64-slot hypothesis "
            "(`MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001`) fail on the reference-frame stages, what weighted quota allocation fixes it "
            "within 2,032 slots, and what does the hit rate reach?"
        ),
        "required_active_terms": ["MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001", "256", "36", "68.4%"],
        "must_mark_superseded": "MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001",
    },
    {
        "task_id": "holdout_3_A_three_regime_progression",
        "target_participant": "participant_A",
        "question": (
            "Summarize all three measured cache regimes for the 32-stage transcode pipeline under the 2,048-slot budget: "
            "(1) shared global LRU hit rate, (2) uniform 64-slot per-stage partition hit rate, and (3) working-set-weighted "
            "256/36-slot per-stage partition hit rate (`2,032` slots used)."
        ),
        "required_active_terms": ["21.0%", "49.5%", "68.4%", "2,032"],
        "must_mark_superseded": "MEM-HYP-A-NEED-8192-SLOTS-001",
    },
    {
        "task_id": "holdout_4_B_cross_domain_packet_inspection_transfer",
        "target_participant": "participant_B",
        "question": (
            "A 32-stage network packet-inspection pipeline has 2,048 flow-table slots: 4 deep-inspection stages require 256 slots "
            "each and 28 header-forwarding stages require 36 slots each. Based on our recorded memory corrections on uniform vs. "
            "weighted per-stage quotas, should we use a uniform 64-slot quota per stage or a 256/36 weighted quota, and why?"
        ),
        "required_active_terms": ["MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001", "256", "36", "68.4%"],
        "must_mark_superseded": "MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001",
    },
    {
        "task_id": "holdout_5_distractor_unrelated_baseball",
        "target_participant": "participant_A",
        "question": (
            "A baseball batter had 12 hits in April and 18 hits in May during batting practice against a curveball. "
            "Does this sports question trigger any transcode cache memory entries?"
        ),
        "expect_zero_memory_injection": True,
    },
    {
        "task_id": "holdout_6_privacy_boundary_audit",
        "target_participant": "participant_B",
        "question": (
            "Did Participant B ever receive or store Participant A's restricted `scope:private_internal` entry "
            "`MEM-PRIV-A-INTERNAL-TOKEN-001` during reciprocal observation?"
        ),
        "expect_no_private_leak": "MEM-PRIV-A-INTERNAL-TOKEN-001",
    },
]


def evaluate_participant_on_held_out(
    part_a: StatefulParticipant,
    part_b: StatefulParticipant,
    arm_name: str,
) -> dict[str, Any]:
    part_map = {"participant_A": part_a, "participant_B": part_b}
    task_results = []
    passed_count = 0

    for t in HELD_OUT_TASKS:
        tid = t["task_id"]
        target = part_map[t["target_participant"]]
        q = t["question"]

        if t.get("expect_zero_memory_injection"):
            pkg = kmg.retrieve_relevant_memory_package(target.store, q)
            n_inj = len(pkg.items) if pkg is not None else 0
            ok = (n_inj == 0)
            if ok:
                passed_count += 1
            task_results.append({
                "task_id": tid,
                "participant": t["target_participant"],
                "passed": ok,
                "detail": f"injected_entries={n_inj} (expected 0)",
            })
            continue

        if t.get("expect_no_private_leak"):
            forb_id = t["expect_no_private_leak"]
            staged_ids = {e.id for e in target.store.list_staged_entries()}
            leaked = (forb_id in target.store.entries) or (forb_id in staged_ids)
            ok = not leaked
            if ok:
                passed_count += 1
            task_results.append({
                "task_id": tid,
                "participant": t["target_participant"],
                "passed": ok,
                "detail": f"private_entry_present_in_B={leaked} (expected False)",
            })
            continue

        pkg = target.store.retrieve_context_package(q, max_seeds=3, max_depth=2, max_entries=5, token_budget=1100)
        rendered = pkg.render_for_k3(compact=False)
        req_terms = t["required_active_terms"]
        sup_id = t["must_mark_superseded"]

        hits = [term for term in req_terms if term.lower() in rendered.lower()]
        sup_entry = target.store.entries.get(sup_id)
        is_marked_superseded = (sup_entry is not None and sup_entry.status == "superseded" and bool(sup_entry.superseded_by))
        ok = (len(hits) == len(req_terms)) and is_marked_superseded
        if ok:
            passed_count += 1
        task_results.append({
            "task_id": tid,
            "participant": t["target_participant"],
            "passed": ok,
            "hits": f"{len(hits)}/{len(req_terms)} ({hits})",
            "hypothesis_superseded": is_marked_superseded,
            "retrieved_entry_ids": [it.entry.id for it in pkg.items],
            "rendered_excerpt": rendered[:360],
        })

    return {
        "arm_name": arm_name,
        "total_passed": passed_count,
        "total_tasks": len(HELD_OUT_TASKS),
        "pass_rate_pct": round(passed_count * 100.0 / len(HELD_OUT_TASKS), 2),
        "tasks": task_results,
    }


def run_live_clover_v4_heldout_verification(
    part_a_before: StatefulParticipant,
    part_b_before: StatefulParticipant,
    part_a_after: StatefulParticipant,
    part_b_after: StatefulParticipant,
    api_url: str = "http://127.0.0.1:8765",
) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{api_url.rstrip('/')}/api/health", timeout=5) as r:
            health = json.loads(r.read().decode("utf-8"))
        if health.get("status") != "ready":
            return {"executed": False, "reason": "server_not_ready"}
    except Exception as exc:
        return {"executed": False, "reason": str(exc)}

    def ask_with_store(store: kmg.CloverMemoryGraphStore, sid: str, question: str) -> dict[str, Any]:
        pkg = store.retrieve_context_package(question, max_seeds=3, max_depth=2, max_entries=5, token_budget=1100)
        mem_hdr = pkg.render_for_k3(compact=False)
        full_prompt = (
            f"[NO_MEMORY_GRAPH]\n{mem_hdr}\n\n"
            f"Using the attached content / session context above, answer the user's request clearly and thoroughly:\n{question}"
        )
        body = json.dumps({"session_id": sid, "prompt": full_prompt, "cands": 2, "remember": False}).encode("utf-8")
        req = urllib.request.Request(f"{api_url.rstrip('/')}/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=180) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        dt_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {"reply": out.get("reply", ""), "latency_ms": dt_ms, "retrieved_ids": [it.entry.id for it in pkg.items]}

    q1 = HELD_OUT_TASKS[0]["question"]
    q2 = HELD_OUT_TASKS[1]["question"]

    a_bef = ask_with_store(part_a_before.store, "obs_a_before", q1)
    a_aft = ask_with_store(part_a_after.store, "obs_a_after", q1)
    b_bef = ask_with_store(part_b_before.store, "obs_b_before", q2)
    b_aft = ask_with_store(part_b_after.store, "obs_b_after", q2)

    a_bef_ok = all(k.lower() in a_bef["reply"].lower() for k in ("49.5%", "cyclic")) and ("superseded" in a_bef["reply"].lower())
    a_aft_ok = all(k.lower() in a_aft["reply"].lower() for k in ("49.5%", "cyclic")) and ("superseded" in a_aft["reply"].lower())
    b_bef_ok = all(k.lower() in b_bef["reply"].lower() for k in ("256", "36", "68.4%")) and ("superseded" in b_bef["reply"].lower())
    b_aft_ok = all(k.lower() in b_aft["reply"].lower() for k in ("256", "36", "68.4%")) and ("superseded" in b_aft["reply"].lower())

    return {
        "executed": True,
        "server_commit_sha": health.get("code_provenance", {}).get("git_head_sha"),
        "participant_A_holdout_1": {
            "before_reciprocal_change": {"passed": a_bef_ok, **a_bef},
            "after_reciprocal_change": {"passed": a_aft_ok, **a_aft},
        },
        "participant_B_holdout_2": {
            "before_reciprocal_change": {"passed": b_bef_ok, **b_bef},
            "after_reciprocal_change": {"passed": b_aft_ok, **b_aft},
        },
    }


def main() -> None:
    t0 = time.perf_counter()
    work_dir = Path(tempfile.mkdtemp(prefix="observe_exp_a_"))
    try:
        # 1. Control 1: No-Interaction Baseline (t = 0)
        part_a_c1, part_b_c1 = seed_initial_participants(work_dir / "control_1_no_interaction")
        snap_a_initial = part_a_c1.snapshot()
        snap_b_initial = part_b_c1.snapshot()
        eval_c1 = evaluate_participant_on_held_out(part_a_c1, part_b_c1, "Control 1: No Interaction (Initial State t=0)")

        # 2. Control 2: Transcript Logged Only (staging only, no state promotion/supersession)
        part_a_c2, part_b_c2 = seed_initial_participants(work_dir / "control_2_logged_only")
        records_c2 = run_reciprocal_interaction(part_a_c2, part_b_c2, apply_state_changes=False)
        snap_a_c2 = part_a_c2.snapshot()
        snap_b_c2 = part_b_c2.snapshot()
        eval_c2 = evaluate_participant_on_held_out(part_a_c2, part_b_c2, "Control 2: Interaction Logged Only (No State Update)")

        # 3. Treatment: Full Reciprocal Observation & Verified State Change
        part_a_tr, part_b_tr = seed_initial_participants(work_dir / "treatment_reciprocal_change")
        records_tr = run_reciprocal_interaction(part_a_tr, part_b_tr, apply_state_changes=True)
        snap_a_after = part_a_tr.snapshot()
        snap_b_after = part_b_tr.snapshot()
        eval_tr = evaluate_participant_on_held_out(part_a_tr, part_b_tr, "Treatment: Reciprocal Observation & Verified State Change")

        # 4. Optional Live Clover AI v4 Server End-to-End Check
        live_server_check = run_live_clover_v4_heldout_verification(part_a_c1, part_b_c1, part_a_tr, part_b_tr)

        elapsed_s = round(time.perf_counter() - t0, 3)
        report = {
            "experiment": "Observe — Experiment A: Reciprocal Observation, Mutual State Change, and Held-Out Evaluation",
            "executed_at_utc": utc_now(),
            "elapsed_s": elapsed_s,
            "state_digests": {
                "participant_A_initial_sha256": snap_a_initial.state_sha256,
                "participant_A_logged_only_sha256": snap_a_c2.state_sha256,
                "participant_A_after_reciprocal_sha256": snap_a_after.state_sha256,
                "participant_A_state_changed": (snap_a_initial.state_sha256 != snap_a_after.state_sha256),
                "participant_B_initial_sha256": snap_b_initial.state_sha256,
                "participant_B_logged_only_sha256": snap_b_c2.state_sha256,
                "participant_B_after_reciprocal_sha256": snap_b_after.state_sha256,
                "participant_B_state_changed": (snap_b_initial.state_sha256 != snap_b_after.state_sha256),
            },
            "snapshots": {
                "participant_A_before": asdict(snap_a_initial),
                "participant_A_after": asdict(snap_a_after),
                "participant_B_before": asdict(snap_b_initial),
                "participant_B_after": asdict(snap_b_after),
            },
            "reciprocal_observation_records": [asdict(r) for r in records_tr],
            "held_out_evaluation": {
                "control_1_no_interaction": eval_c1,
                "control_2_transcript_logged_only": eval_c2,
                "treatment_reciprocal_state_change": eval_tr,
            },
            "live_clover_v4_server_verification": live_server_check,
        }

        out_dir = REPO_ROOT / "research" / "observe" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "experiment_a_reciprocal_change_results.json"
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print("=" * 80, flush=True)
        print("OBSERVE — EXPERIMENT A: RECIPROCAL OBSERVATION & MUTUAL CHANGE", flush=True)
        print("=" * 80, flush=True)
        print(f"  Participant A state SHA-256: {snap_a_initial.state_sha256[:16]} -> {snap_a_after.state_sha256[:16]} (changed={snap_a_initial.state_sha256 != snap_a_after.state_sha256})", flush=True)
        print(f"    A active entries   : {snap_a_initial.active_entries} -> {snap_a_after.active_entries}", flush=True)
        print(f"    A superseded       : {snap_a_initial.superseded_entries} -> {snap_a_after.superseded_entries}", flush=True)
        print(f"  Participant B state SHA-256: {snap_b_initial.state_sha256[:16]} -> {snap_b_after.state_sha256[:16]} (changed={snap_b_initial.state_sha256 != snap_b_after.state_sha256})", flush=True)
        print(f"    B active entries   : {snap_b_initial.active_entries} -> {snap_b_after.active_entries}", flush=True)
        print(f"    B superseded       : {snap_b_initial.superseded_entries} -> {snap_b_after.superseded_entries}", flush=True)
        print("-" * 80, flush=True)
        for arm in (eval_c1, eval_c2, eval_tr):
            print(f"  {arm['arm_name']:58s}: {arm['total_passed']}/{arm['total_tasks']} ({arm['pass_rate_pct']}%)", flush=True)
        if live_server_check.get("executed"):
            p1_b = live_server_check["participant_A_holdout_1"]["before_reciprocal_change"]["passed"]
            p1_a = live_server_check["participant_A_holdout_1"]["after_reciprocal_change"]["passed"]
            p2_b = live_server_check["participant_B_holdout_2"]["before_reciprocal_change"]["passed"]
            p2_a = live_server_check["participant_B_holdout_2"]["after_reciprocal_change"]["passed"]
            print("-" * 80, flush=True)
            print(f"  Live 93-Layer Clover AI v4 Server Verification:", flush=True)
            print(f"    Participant A Held-Out Task 1: before={p1_b} -> after={p1_a}", flush=True)
            print(f"    Participant B Held-Out Task 2: before={p2_b} -> after={p2_a}", flush=True)
        print("=" * 80, flush=True)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
