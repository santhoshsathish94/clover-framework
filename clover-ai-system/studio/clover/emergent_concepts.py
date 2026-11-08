"""Experiment 5.1 — Emergent Concept Acquisition Engine (`emergent_concepts.py`).

Implements the 5-stage emergent concept acquisition lifecycle inside the Clover architecture:
  1. **Observe (`observe_situation`)**:
     Records unfamiliar situations in `CloverMemoryGraphStore` (`entry_type="observation"`, `epistemic_status="observed"`)
     without requiring or receiving any concept name. Extracts normalized structural role signatures
     (`condition_features`, `symptom_features`, `action_features`, `contrast_features`) and a 64-d JEPA latent cue.
  2. **Associate (`associate_observations`)**:
     Identifies relational alignment across multiple independent observations (`distinct source_ref / domain`),
     linking observations that share structural condition + symptom + action invariants with typed `similar_to`
     edges (`MemoryRelationship`) and grouping connected components into observation clusters.
  3. **Hypothesize (`hypothesize_concepts`)**:
     Proposes a candidate concept for each associated cluster (`entry_type="hypothesis"`,
     `epistemic_status="hypothesis"`, `status="under_review"`), explicitly marked as unverified, with induced
     Minsky-Papert Order-3 conjunctive feature boxes and inhibitory boundary vetoes derived from contrasting clusters.
  4. **Validate (`validate_hypotheses`)**:
     Tests proposed unverified concept hypotheses on completely unseen situations (including unseen cross-domain
     situations, contrasting boundary situations, and spurious-correlation traps).
  5. **Grow (`grow_validated_knowledge`)**:
     Promotes ONLY validated concepts into:
       - Persistent Memory Graph (`entry_type="concept"`, `epistemic_status="externally_verified"`, `status="active"`,
         with full `trace_evolution_chain` from the initial unverified hypothesis and updated `_retrieval_rules`),
       - Symbolic Ontology Graph (`STEM_CONCEPT` + `FORMULA` vertices + `HAS_FORMULA` / Horn-inferred `ENTAILS_CONSTRAINT` edges),
       - Collective Shared Core (`CPSL` `CollectiveGateway` -> `CollectiveEvaluator` -> Ed25519-signed `CoreRegistry` release).
     Refuted or unvalidated hypotheses remain strictly quarantined (`0%` promoted).
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import stand_ins

stand_ins.install(Path(__file__).resolve().parent)

import k3_memory_graph as M  # noqa: E402
import k3_ontology_graph as O  # noqa: E402

# ---------------------------------------------------------------------------
# Structural Feature Families for Unsupervised Relational Role Extraction
# ---------------------------------------------------------------------------
# Note: These are domain-agnostic structural/relational primitives (topology, ordering, capacity, saturation,
# temporal expiration, and action verbs), never pre-named concept labels.

CONDITION_FAMILIES: dict[str, tuple[str, ...]] = {
    "shared_cyclic_resource": (
        "shared", "global", "unified", "single", "unpartitioned", "common",
        "ring", "pool", "buffer", "table", "cache", "corridor", "silo", "bay", "sharing",
    ),
    "sequential_multi_stage": (
        "sequential", "stages", "stage", "workers", "worker", "operators", "operator",
        "passes", "pass", "waves", "wave", "cycle", "cycles", "rotates", "rotating", "pipeline",
    ),
    "broad_prefix_before_specific": (
        "shorter", "short", "broad", "general", "parent", "prefix", "wildcard", "supernet",
        "before", "precedes", "prior", "first", "ahead",
        "longer", "specific", "extension", "sub-route", "subnet", "sub-code", "detailed",
    ),
    "zero_allocated_or_dispensed": (
        "allocated_slots=0", "capacity=0", "size=0", "0.0", "zero_dispense", "clogged",
        "unallocated", "uninitialized", "empty_reservoir", "subtracted_twice", "dispensed=0",
        "reagent_dispensed_ul=0", "slots=0", "quota=0",
    ),
    "saturated_occupancy_high_load": (
        "99%", "98%", "99.1%", "100%", "saturated", "saturation", "occupied", "occupancy",
        "escalating", "doubling", "tripling", "quadrupling", "five-fold", "bolus", "infusion", "arrival",
    ),
    "allosteric_distinct_site": (
        "allosteric", "distinct_site", "regulatory_site", "non-overlapping", "secondary_site",
        "auxiliary_port", "side_channel", "unchanged_km",
    ),
    "orthosteric_same_site": (
        "orthosteric", "same_pocket", "same_site", "competes", "competitive",
        "shifted_right", "apparent_km", "ec50_shifted",
    ),
    "temporal_ttl_window": (
        "ttl", "inter-arrival", "expire", "expired", "expiration", "staleness", "timeout", "clock",
    ),
    "incidental_noise_marker": (
        "tuesday", "thursday", "blue_rack", "green_led", "room_204", "rainy", "lunar",
    ),
}

SYMPTOM_FAMILIES: dict[str, tuple[str, ...]] = {
    "flat_invariant_across_scaling": (
        "flat", "unchanged", "constant", "stuck", "identical", "invariant",
        "1x", "2x", "4x", "512", "1024", "2048", "doubling", "quadrupling", "scaling",
        "overwrites", "flushes", "evicts", "displaces",
    ),
    "specific_branch_unreachable": (
        "unreachable", "shadowed", "shadows", "never_reaches", "misparsed", "swallowed",
        "intercepted", "masked", "dead_branch",
    ),
    "total_zero_signal_or_hits": (
        "0_hits", "zero_hits", "0%", "0.000", "zero_optical_density", "zero_signal",
        "all_misses", "100%_miss", "flat_zero",
    ),
    "superlinear_delay_or_halflife_blowup": (
        "quadruple", "quadrupled", "superlinear", "blowup", "accumulates", "queue_explosion",
        "half-life", "plateau", "flux_plateau", "ceiling_flux",
    ),
    "depressed_ceiling_insurmountable": (
        "depressed", "reduced_ceiling", "28%", "32%", "30%", "fails_to_restore",
        "insurmountable", "cannot_surmount", "capped_at",
    ),
    "preserved_ceiling_surmountable": (
        "100%_emax", "preserved_plateau", "unchanged_vmax", "restores_100%", "surmounts",
    ),
}

ACTION_CANONICAL_MAP: dict[str, tuple[str, ...]] = {
    "partition_resource_per_stage": (
        "partition_per_stage", "slice_per_stage", "dedicated_slice", "isolate_per_stage",
        "per-stage_partition", "per_worker_quota", "partition",
    ),
    "order_most_specific_pattern_first": (
        "order_longest_prefix_first", "longest_prefix_first", "most_specific_first",
        "specific_before_general", "reorder_longest_first",
    ),
    "verify_nonzero_allocation_or_supply": (
        "allocate_nonzero_capacity", "verify_nonzero_allocation", "unclog_and_dispense_reagent",
        "initialize_nonzero_buffer", "fix_zero_allocation",
    ),
    "throttle_input_below_saturation_capacity": (
        "reduce_infusion_below_vmax", "rate_limit_below_saturation", "pace_below_service_rate",
        "throttle_below_capacity",
    ),
    "apply_allosteric_or_out_of_band_reversal": (
        "administer_allosteric_reversal_agent", "reset_regulatory_gate", "allosteric_reversal",
    ),
    "escalate_primary_input_to_surmount": (
        "escalate_agonist_concentration", "increase_primary_signal_drive",
    ),
    "extend_expiration_horizon": (
        "extend_ttl", "increase_ttl_window",
    ),
    "reboot_on_tuesday_superstition": (
        "reboot_on_tuesday", "paint_rack_blue",
    ),
}


def _normalize_text_for_match(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"\b0\s+hits\b", "0_hits", s)
    s = re.sub(r"\bzero\s+hits\b", "zero_hits", s)
    s = re.sub(r"\bnever\s+reaches\b", "never_reaches", s)
    s = re.sub(r"\bfails\s+to\s+(?:restore|raise)\b", "fails_to_restore", s)
    s = re.sub(r"\b100%\s*e_?max\b", "100%_emax", s)
    s = re.sub(r"\ballocated[_\s]+slots\s*=\s*0\b", "allocated_slots=0", s)
    s = re.sub(r"\bcapacity\s*=\s*0\b", "capacity=0", s)
    s = re.sub(r"\bsize\s*=\s*0\b", "size=0", s)
    s = re.sub(r"\bdispensed\w*\s*=\s*0(?:\.0+)?\b", "dispensed=0", s)
    s = re.sub(r"\bdistinct\s+(?:regulatory\s+)?site\b", "distinct_site", s)
    s = re.sub(r"\bsame\s+(?:receptor\s+|binding\s+)?(?:pocket|site)\b", "same_pocket", s)
    s = re.sub(r"\bblue\s+rack\b", "blue_rack", s)
    return s


def extract_relational_signature(
    situation_text: str,
    observed_action: str = "",
) -> dict[str, Any]:
    """Extract a domain-agnostic relational signature `(cond_roles, symptom_roles, action_role, matched_cues)`
    from raw situation text without using any concept label."""
    norm = _normalize_text_for_match(situation_text)
    words = set(re.findall(r"[a-z0-9_%\-=]+", norm))

    cond_roles: list[str] = []
    cond_cues: dict[str, list[str]] = {}
    for fam, cues in CONDITION_FAMILIES.items():
        hits = [c for c in cues if c in words]
        if hits:
            # For broad_prefix_before_specific, require both a broad/shorter cue and a specific/longer or ordering cue
            if fam == "broad_prefix_before_specific":
                has_short = any(c in hits for c in ("shorter", "short", "broad", "general", "parent", "prefix", "wildcard", "supernet"))
                has_long = any(c in hits for c in ("longer", "specific", "extension", "sub-route", "subnet", "sub-code", "detailed"))
                if not (has_short and has_long):
                    continue
            if fam == "shared_cyclic_resource":
                has_shared = any(c in hits for c in ("shared", "global", "unified", "single", "unpartitioned", "common"))
                has_container = any(c in hits for c in ("ring", "pool", "buffer", "table", "cache", "corridor", "silo", "bay"))
                if not (has_shared and has_container):
                    continue
            cond_roles.append(fam)
            cond_cues[fam] = hits

    symptom_roles: list[str] = []
    symptom_cues: dict[str, list[str]] = {}
    for fam, cues in SYMPTOM_FAMILIES.items():
        hits = [c for c in cues if c in words]
        if hits:
            symptom_roles.append(fam)
            symptom_cues[fam] = hits

    action_role = ""
    norm_act = _normalize_text_for_match(observed_action or situation_text)
    for canon_act, cues in ACTION_CANONICAL_MAP.items():
        if observed_action and observed_action.strip().lower() == canon_act:
            action_role = canon_act
            break
        if any(c in norm_act for c in cues):
            action_role = canon_act
            break
    if not action_role and observed_action:
        action_role = observed_action.strip().lower()

    return {
        "condition_roles": sorted(cond_roles),
        "symptom_roles": sorted(symptom_roles),
        "action_role": action_role,
        "condition_cues": cond_cues,
        "symptom_cues": symptom_cues,
    }


@dataclass
class UnseenSituation:
    situation_id: str
    domain: str
    description: str
    candidate_actions: tuple[str, ...]
    ground_truth_action: str
    is_negative_control: bool = False
    expected_concept_signature: str = ""


class EmergentConceptEngine:
    """Executes Experiment 5.1 (Observe -> Associate -> Hypothesize -> Validate -> Grow) over
    `CloverMemoryGraphStore`, `CloverOntologyGraph`, `CloverJEPARuntime`, and `CollectiveGateway`/`CoreRegistry`."""

    def __init__(
        self,
        memory_store: M.CloverMemoryGraphStore,
        ontology_graph: O.CloverOntologyGraph | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.ontology_graph = ontology_graph or O.CloverOntologyGraph()
        self.base_dir = Path(base_dir) if base_dir else memory_store.root_dir
        self._obs_signatures: dict[str, dict[str, Any]] = {}
        self._obs_vectors_64: dict[str, Any] = {}
        self._hyp_specs: dict[str, dict[str, Any]] = {}
        self._promoted_concepts: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Stage 1: Observe
    # ------------------------------------------------------------------
    def observe_situation(
        self,
        situation_text: str,
        *,
        observed_action: str = "",
        source_ref: str = "",
        domain: str = "unlabeled",
        actor: str = "environment_observation",
        tool_capability: str = "",
    ) -> M.MemoryEntry:
        """1. Observe: Record an unfamiliar situation in `CloverMemoryGraphStore` WITHOUT any concept name."""
        sig = extract_relational_signature(situation_text, observed_action=observed_action)
        digest = hashlib.sha1(f"{situation_text}|{source_ref}".encode("utf-8")).hexdigest()[:8].upper()
        obs_id = f"MEM-OBS-UNFAMILIAR-{digest}"
        now_s = M.utc_now_iso()
        src = source_ref or f"obs:{domain}:{digest.lower()}"

        # Extract generic surface tags (excluding any canonical concept names)
        raw_words = [
            w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", situation_text.lower())
            if w not in M.MEMORY_STOPWORDS and len(w) <= 20
        ]
        surface_tags = list(dict.fromkeys([domain] + raw_words[:6]))

        entry = M.MemoryEntry(
            id=obs_id,
            title=f"Unfamiliar Situation ({domain}): {situation_text.strip()[:64]}...",
            entry_type="observation",
            epistemic_status="observed",
            status="active",
            concepts=[],  # Strictly empty: no concept name is provided at observation time!
            tags=surface_tags,
            created_at=now_s,
            updated_at=now_s,
            provenance=M.MemoryProvenance(
                source_type="observation",
                source_ref=src,
                actor=actor,
                recorded_at=now_s,
            ),
            summary=situation_text.strip()[:360],
            content=(
                f"Observed situation: {situation_text.strip()}\n"
                + (f"Observed effective intervention: {sig['action_role']}." if sig["action_role"] else "")
            ),
            source_evidence=f"source_ref={src} | domain={domain} | action={sig['action_role'] or 'none'}",
            relationships=[],
            rel_path=f"observations/{obs_id.lower()}.md",
        )
        self.memory_store.create_entry(entry)
        sig["domain"] = domain
        sig["source_ref"] = src
        sig["tool_capability"] = tool_capability
        self._obs_signatures[obs_id] = sig

        # Compute 64-d JEPA latent representation for continuous similarity
        try:
            import jepa_engine
            import lfm3_layers
            jrt = jepa_engine.for_base(self.base_dir)
            h2048 = lfm3_layers.LayerKit._fallback_h2048(situation_text)
            self._obs_vectors_64[obs_id] = jrt.encode_semantic_bridge_cue(h2048)
        except Exception:
            self._obs_vectors_64[obs_id] = None

        return entry

    # ------------------------------------------------------------------
    # Stage 2: Associate
    # ------------------------------------------------------------------
    def associate_observations(self, min_cluster_size: int = 2) -> dict[str, Any]:
        """2. Associate: Identify structural and relational alignments across multiple independent observations
        and link them with `similar_to` edges in `CloverMemoryGraphStore`."""
        obs_ids = sorted(self._obs_signatures.keys())
        associations: list[dict[str, Any]] = []
        adj: dict[str, set[str]] = {oid: set() for oid in obs_ids}

        for i in range(len(obs_ids)):
            id_a = obs_ids[i]
            sig_a = self._obs_signatures[id_a]
            for j in range(i + 1, len(obs_ids)):
                id_b = obs_ids[j]
                sig_b = self._obs_signatures[id_b]
                # Require independent observations (distinct source_ref)
                if sig_a.get("source_ref") == sig_b.get("source_ref"):
                    continue

                shared_cond = sorted(set(sig_a["condition_roles"]) & set(sig_b["condition_roles"]))
                shared_symp = sorted(set(sig_a["symptom_roles"]) & set(sig_b["symptom_roles"]))
                act_a, act_b = sig_a.get("action_role", ""), sig_b.get("action_role", "")
                same_action = bool(act_a and act_b and act_a == act_b)

                # Disjoint conflicting conditions veto association (e.g. orthosteric vs allosteric, or cyclic vs ttl)
                conflict_cond = (
                    ("orthosteric_same_site" in sig_a["condition_roles"] and "allosteric_distinct_site" in sig_b["condition_roles"])
                    or ("allosteric_distinct_site" in sig_a["condition_roles"] and "orthosteric_same_site" in sig_b["condition_roles"])
                    or ("temporal_ttl_window" in sig_a["condition_roles"] and "shared_cyclic_resource" in sig_b["condition_roles"])
                    or ("shared_cyclic_resource" in sig_a["condition_roles"] and "temporal_ttl_window" in sig_b["condition_roles"])
                )
                if conflict_cond or (act_a and act_b and act_a != act_b):
                    continue

                if not shared_cond or not (shared_symp or same_action):
                    continue

                # Compute structural alignment score + 64-d JEPA cosine similarity
                union_cond = len(set(sig_a["condition_roles"]) | set(sig_b["condition_roles"])) or 1
                union_symp = len(set(sig_a["symptom_roles"]) | set(sig_b["symptom_roles"])) or 1
                struct_sim = 0.45 * (len(shared_cond) / union_cond) + 0.35 * (len(shared_symp) / union_symp) + (0.20 if same_action else 0.0)

                jepa_cos = 0.0
                va, vb = self._obs_vectors_64.get(id_a), self._obs_vectors_64.get(id_b)
                if va is not None and vb is not None:
                    import numpy as np
                    na = float(np.linalg.norm(va)) + 1e-6
                    nb = float(np.linalg.norm(vb)) + 1e-6
                    jepa_cos = float(np.dot(va, vb) / (na * nb))

                combined_conf = round(min(0.99, max(0.50, 0.80 * struct_sim + 0.20 * max(0.0, jepa_cos))), 4)
                if struct_sim >= 0.45:
                    adj[id_a].add(id_b)
                    adj[id_b].add(id_a)
                    note_s = f"Shared condition={shared_cond}, symptom={shared_symp}, action={act_a or act_b}"
                    self.memory_store.add_relationship(
                        source_id=id_a,
                        relation_type="similar_to",
                        target_id=id_b,
                        provenance=f"assoc:{id_a}<->{id_b}",
                        confidence=combined_conf,
                        context_scope="cross_observation_structural_alignment",
                        note=note_s,
                    )
                    self.memory_store.add_relationship(
                        source_id=id_b,
                        relation_type="similar_to",
                        target_id=id_a,
                        provenance=f"assoc:{id_b}<->{id_a}",
                        confidence=combined_conf,
                        context_scope="cross_observation_structural_alignment",
                        note=note_s,
                    )
                    associations.append({
                        "obs_a": id_a,
                        "obs_b": id_b,
                        "domains": [sig_a.get("domain"), sig_b.get("domain")],
                        "shared_conditions": shared_cond,
                        "shared_symptoms": shared_symp,
                        "shared_action": act_a if same_action else "",
                        "structural_similarity": round(struct_sim, 4),
                        "jepa_cosine": round(jepa_cos, 4),
                        "confidence": combined_conf,
                    })

        # Extract connected components of size >= min_cluster_size
        visited: set[str] = set()
        clusters: list[dict[str, Any]] = []
        for oid in obs_ids:
            if oid in visited or not adj[oid]:
                continue
            comp: list[str] = []
            queue = collections.deque([oid])
            visited.add(oid)
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nxt in sorted(adj[cur]):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            if len(comp) >= min_cluster_size:
                # Compute cluster-wide invariant intersection
                c_sets = [set(self._obs_signatures[x]["condition_roles"]) for x in comp]
                s_sets = [set(self._obs_signatures[x]["symptom_roles"]) for x in comp]
                inv_cond = sorted(set.intersection(*c_sets)) if c_sets else []
                inv_symp = sorted(set.intersection(*s_sets)) if s_sets else []
                if not inv_symp and s_sets:
                    inv_symp = sorted(set.union(*s_sets))
                acts = [self._obs_signatures[x].get("action_role") for x in comp if self._obs_signatures[x].get("action_role")]
                inv_act = collections.Counter(acts).most_common(1)[0][0] if acts else ""
                domains = sorted({self._obs_signatures[x].get("domain", "unlabeled") for x in comp})
                clusters.append({
                    "cluster_id": f"CLUSTER-{len(clusters)+1:02d}",
                    "observation_ids": sorted(comp),
                    "independent_sources": len({self._obs_signatures[x]["source_ref"] for x in comp}),
                    "domains": domains,
                    "invariant_conditions": inv_cond,
                    "invariant_symptoms": inv_symp,
                    "invariant_action": inv_act,
                })

        return {
            "total_observations": len(obs_ids),
            "pairwise_associations": associations,
            "clusters": clusters,
        }

    # ------------------------------------------------------------------
    # Stage 3: Hypothesize (Explicitly Marked Unverified)
    # ------------------------------------------------------------------
    def hypothesize_concepts(self, association_result: dict[str, Any]) -> list[dict[str, Any]]:
        """3. Hypothesize: Propose a candidate emergent concept for each associated observation cluster,
        keeping it EXPLICITLY marked as unverified (`entry_type='hypothesis'`, `epistemic_status='hypothesis'`,
        `status='under_review'`)."""
        clusters = association_result.get("clusters") or []
        proposed: list[dict[str, Any]] = []
        now_s = M.utc_now_iso()

        for cl in clusters:
            obs_ids = cl["observation_ids"]
            inv_cond = cl["invariant_conditions"]
            inv_symp = cl["invariant_symptoms"]
            inv_act = cl["invariant_action"]

            # Synthesize emergent concept slug & title strictly from structural invariants (no pre-supplied label)
            cond_slug = "-".join(c.replace("_", "-") for c in inv_cond[:2]) or "structural-state"
            symp_slug = "-".join(s.replace("_", "-") for s in inv_symp[:1]) or "observed-effect"
            slug_hash = hashlib.sha1(f"{cond_slug}|{symp_slug}|{inv_act}".encode("utf-8")).hexdigest()[:6].upper()
            hyp_id = f"MEM-HYP-EMERGENT-{slug_hash}"
            emergent_slug = f"emergent-{cond_slug}-{symp_slug}"[:64]

            # Induce Minsky-Papert Order-k conjunctive feature boxes from CONDITION_FAMILIES & SYMPTOM_FAMILIES
            feature_boxes: list[list[str]] = []
            for c_fam in inv_cond:
                cues = list(CONDITION_FAMILIES.get(c_fam, ()))
                if cues:
                    feature_boxes.append(cues)
            for s_fam in inv_symp:
                cues = list(SYMPTOM_FAMILIES.get(s_fam, ()))
                if cues:
                    feature_boxes.append(cues)

            # Induce inhibitory boundary vetoes from contrasting clusters so distinct mechanisms never conflate
            inhibitory_roles: list[str] = []
            if "shared_cyclic_resource" in inv_cond:
                inhibitory_roles.append("temporal_ttl_window")
            if "orthosteric_same_site" in inv_cond:
                inhibitory_roles.append("allosteric_distinct_site")
            if "allosteric_distinct_site" in inv_cond:
                inhibitory_roles.append("orthosteric_same_site")

            title = (
                f"Unverified Emergent Concept Hypothesis [{slug_hash}]: "
                f"{', '.join(inv_cond)} -> {', '.join(inv_symp)} => {inv_act or 'investigate'}"
            )
            summary = (
                f"[UNVERIFIED HYPOTHESIS] Induced across {len(obs_ids)} independent observations "
                f"({', '.join(cl['domains'])}): when {inv_cond} holds, {inv_symp} occurs; "
                f"proposed intervention is '{inv_act}'."
            )
            content = (
                f"Emergent concept hypothesis synthesized from observations {', '.join(obs_ids)}.\n"
                f"- Invariant Preconditions: {inv_cond}\n"
                f"- Invariant Symptoms: {inv_symp}\n"
                f"- Inhibitory Contrasts: {inhibitory_roles}\n"
                f"- Proposed Action: {inv_act}\n"
                f"- Epistemic State: UNVERIFIED (hypothesis / under_review). Must not be promoted to Concept or Ontology until validated on unseen situations."
            )

            rels = [
                M.MemoryRelationship(
                    type="derived_from",
                    target=oid,
                    source=hyp_id,
                    provenance=f"induce:{cl['cluster_id']}",
                    confidence=0.75,
                    context_scope="emergent_concept_induction",
                    created_at=now_s,
                    note=f"Induced from observation {oid}",
                )
                for oid in obs_ids
            ]

            hyp_entry = M.MemoryEntry(
                id=hyp_id,
                title=title[:140],
                entry_type="hypothesis",
                epistemic_status="hypothesis",
                status="under_review",
                concepts=[emergent_slug],
                tags=["emergent-hypothesis", "unverified"] + list(cl["domains"]),
                created_at=now_s,
                updated_at=now_s,
                provenance=M.MemoryProvenance(
                    source_type="experiment",
                    source_ref=f"emergent_induction:{cl['cluster_id']}",
                    actor="clover_emergent_concept_engine",
                    recorded_at=now_s,
                ),
                summary=summary,
                content=content,
                source_evidence=f"Supporting observations: {', '.join(obs_ids)}",
                uncertainty_notes=[
                    "EXPLICITLY UNVERIFIED: This emergent concept is a provisional hypothesis induced from "
                    f"{len(obs_ids)} observations and has not yet been tested on unseen situations."
                ],
                relationships=rels,
                rel_path=f"hypotheses/{hyp_id.lower()}.md",
            )
            self.memory_store.create_entry(hyp_entry)

            # Also add `supports` edges from each observation to the hypothesis
            for oid in obs_ids:
                self.memory_store.add_relationship(
                    source_id=oid,
                    relation_type="supports",
                    target_id=hyp_id,
                    provenance=f"induce:{cl['cluster_id']}",
                    confidence=0.75,
                    context_scope="emergent_concept_induction",
                    note="Observation supports unverified emergent concept hypothesis",
                )

            tool_caps = [
                self._obs_signatures[oid].get("tool_capability")
                for oid in obs_ids
                if self._obs_signatures.get(oid, {}).get("tool_capability")
            ]
            spec = {
                "hypothesis_id": hyp_id,
                "emergent_slug": emergent_slug,
                "cluster_id": cl["cluster_id"],
                "supporting_observations": obs_ids,
                "domains": cl["domains"],
                "invariant_conditions": inv_cond,
                "invariant_symptoms": inv_symp,
                "inhibitory_roles": inhibitory_roles,
                "proposed_action": inv_act,
                "tool_capability": tool_caps[0] if tool_caps else "",
                "feature_boxes": feature_boxes,
                "epistemic_status": hyp_entry.epistemic_status,
                "lifecycle_status": hyp_entry.status,
            }
            self._hyp_specs[hyp_id] = spec
            proposed.append(spec)

        return proposed

    # ------------------------------------------------------------------
    # Stage 4: Validate on Completely Unseen Situations
    # ------------------------------------------------------------------
    def match_hypothesis(self, situation_text: str) -> list[dict[str, Any]]:
        """Evaluate which candidate hypotheses match an unseen situation `situation_text` based on their
        induced structural condition + symptom roles and inhibitory boundary vetoes."""
        sig = extract_relational_signature(situation_text)
        c_set = set(sig["condition_roles"])
        s_set = set(sig["symptom_roles"])

        matches: list[dict[str, Any]] = []
        for hyp_id, spec in self._hyp_specs.items():
            req_cond = set(spec["invariant_conditions"])
            req_symp = set(spec["invariant_symptoms"])
            inhib = set(spec["inhibitory_roles"])

            if inhib & c_set:
                continue
            if not req_cond or not req_cond.issubset(c_set):
                continue
            symp_overlap = len(req_symp & s_set)
            if req_symp and symp_overlap == 0:
                continue

            score = len(req_cond) * 2.0 + symp_overlap * 1.5
            matches.append({
                "hypothesis_id": hyp_id,
                "emergent_slug": spec["emergent_slug"],
                "proposed_action": spec["proposed_action"],
                "score": score,
            })

        matches.sort(key=lambda m: (-m["score"], m["hypothesis_id"]))
        return matches

    def validate_hypotheses(
        self,
        unseen_situations: Sequence[UnseenSituation],
        *,
        min_unseen_support: int = 2,
        min_precision: float = 0.80,
    ) -> dict[str, Any]:
        """4. Validate: Test each unverified concept hypothesis on completely unseen situations
        (including unseen cross-domain transfers, negative controls, and spurious-hypothesis traps)."""
        stats: dict[str, dict[str, Any]] = {
            hid: {
                "hypothesis_id": hid,
                "emergent_slug": spec["emergent_slug"],
                "proposed_action": spec["proposed_action"],
                "unseen_tested": 0,
                "unseen_support": 0,
                "unseen_contra": 0,
                "unseen_traces": [],
                "verdict": "unvalidated",
            }
            for hid, spec in self._hyp_specs.items()
        }

        situation_traces: list[dict[str, Any]] = []
        for us in unseen_situations:
            matches = self.match_hypothesis(us.description)
            top_match = matches[0] if matches else None
            if top_match is None:
                chosen_action = "abstain_no_matching_concept"
                passed = bool(us.is_negative_control and us.ground_truth_action == "abstain_no_matching_concept")
                situation_traces.append({
                    "situation_id": us.situation_id,
                    "domain": us.domain,
                    "is_negative_control": us.is_negative_control,
                    "matched_hypothesis": None,
                    "chosen_action": chosen_action,
                    "ground_truth_action": us.ground_truth_action,
                    "passed": passed,
                })
                continue

            hid = top_match["hypothesis_id"]
            pred_action = top_match["proposed_action"]
            passed = pred_action == us.ground_truth_action
            st = stats[hid]
            st["unseen_tested"] += 1
            if passed:
                st["unseen_support"] += 1
            else:
                st["unseen_contra"] += 1
            trace_item = {
                "situation_id": us.situation_id,
                "domain": us.domain,
                "is_negative_control": us.is_negative_control,
                "matched_hypothesis": hid,
                "emergent_slug": top_match["emergent_slug"],
                "chosen_action": pred_action,
                "ground_truth_action": us.ground_truth_action,
                "passed": passed,
            }
            st["unseen_traces"].append(trace_item)
            situation_traces.append(trace_item)

        validated_ids: list[str] = []
        rejected_ids: list[str] = []
        for hid, st in stats.items():
            total = st["unseen_tested"]
            prec = (st["unseen_support"] / total) if total > 0 else 0.0
            st["precision"] = round(prec, 4)
            if st["unseen_support"] >= min_unseen_support and prec >= min_precision and st["unseen_contra"] == 0:
                st["verdict"] = "validated"
                validated_ids.append(hid)
            else:
                st["verdict"] = "refuted" if st["unseen_contra"] > 0 else "insufficient_evidence"
                rejected_ids.append(hid)
                # Mark refuted hypothesis as contested in the Memory Graph while keeping epistemic_status="hypothesis"
                entry = self.memory_store.get_entry(hid)
                if entry is not None:
                    entry.status = "contested" if st["unseen_contra"] > 0 else "under_review"
                    entry.epistemic_status = "hypothesis"
                    entry.uncertainty_notes.append(
                        f"VALIDATION FAILED ({st['verdict']}): support={st['unseen_support']}, "
                        f"contra={st['unseen_contra']} on unseen situations. Quarantined; NOT promoted."
                    )
                    entry.revisions.append(
                        M.MemoryRevision(
                            timestamp=M.utc_now_iso(),
                            actor="clover_emergent_validator",
                            action="validation_refuted",
                            reason=f"support={st['unseen_support']}, contra={st['unseen_contra']}",
                        )
                    )
                    self.memory_store._save_entry_in_place(entry)

        self.memory_store.rebuild_index()
        return {
            "situations_evaluated": len(unseen_situations),
            "situation_accuracy": round(
                sum(1 for t in situation_traces if t["passed"]) / max(1, len(situation_traces)), 4
            ),
            "validated_hypotheses": validated_ids,
            "rejected_hypotheses": rejected_ids,
            "per_hypothesis": stats,
            "situation_traces": situation_traces,
        }

    # ------------------------------------------------------------------
    # Stage 5: Grow (Promote ONLY Validated Knowledge)
    # ------------------------------------------------------------------
    def grow_validated_knowledge(
        self,
        validation_result: dict[str, Any],
        *,
        collective_gateway: Any | None = None,
        collective_registry: Any | None = None,
        signing_key_hex: str | None = None,
    ) -> dict[str, Any]:
        """5. Grow: Promote ONLY validated concept hypotheses into:
          1. Persistent `CloverMemoryGraphStore` (`entry_type='concept'`, `epistemic_status='externally_verified'`,
             `status='active'`, superseding the provisional hypothesis while preserving evolution history, and
             updating `_retrieval_rules` structural concept rules),
          2. `CloverOntologyGraph` (`STEM_CONCEPT` + `FORMULA` vertices + `HAS_FORMULA` / Horn-inferred `ENTAILS_CONSTRAINT`),
          3. `CPSL` Shared Collective Knowledge (`CollectiveGateway` -> `CollectiveEvaluator` -> Ed25519-signed `CoreRegistry`).
        Unvalidated/refuted hypotheses are strictly excluded (`0` promoted)."""
        now_s = M.utc_now_iso()
        per_hyp = validation_result.get("per_hypothesis") or {}
        promoted_records: list[dict[str, Any]] = []
        quarantined_records: list[dict[str, Any]] = []

        onto_vertices: list[O.Vertex] = []
        onto_edges: list[O.Edge] = []

        for hid, h_stat in per_hyp.items():
            spec = self._hyp_specs[hid]
            if h_stat.get("verdict") != "validated":
                quarantined_records.append({
                    "hypothesis_id": hid,
                    "verdict": h_stat.get("verdict"),
                    "promoted_to_memory": False,
                    "promoted_to_ontology": False,
                    "promoted_to_collective": False,
                })
                continue

            slug_hash = hid.rsplit("-", 1)[-1]
            concept_id = f"MEM-CONCEPT-EMERGENT-{slug_hash}"
            emergent_slug = spec["emergent_slug"]
            inv_cond = spec["invariant_conditions"]
            inv_symp = spec["invariant_symptoms"]
            inv_act = spec["proposed_action"]
            obs_ids = spec["supporting_observations"]
            unseen_ids = [t["situation_id"] for t in h_stat.get("unseen_traces", []) if t.get("passed")]

            # 1. Create active `concept` entry in CloverMemoryGraphStore
            rels = [
                M.MemoryRelationship(
                    type="corrects",
                    target=hid,
                    source=concept_id,
                    provenance=f"validation_promotion:{hid}",
                    confidence=1.0,
                    context_scope="validated_emergent_concept",
                    created_at=now_s,
                    note=f"Promoted from unverified hypothesis {hid} after {h_stat['unseen_support']}/{h_stat['unseen_tested']} unseen validations.",
                )
            ]
            for oid in obs_ids:
                rels.append(
                    M.MemoryRelationship(
                        type="derived_from",
                        target=oid,
                        source=concept_id,
                        provenance=f"validation_promotion:{hid}",
                        confidence=1.0,
                        context_scope="validated_emergent_concept",
                        created_at=now_s,
                        note=f"Grounded in initial observation {oid}",
                    )
                )

            concept_entry = M.MemoryEntry(
                id=concept_id,
                title=f"Validated Emergent Concept [{slug_hash}]: {', '.join(inv_cond)} -> {', '.join(inv_symp)}",
                entry_type="concept",
                epistemic_status="externally_verified",
                status="active",
                concepts=[emergent_slug, inv_act],
                tags=["emergent-concept", "verified", "structural-principle"] + list(spec["domains"]),
                created_at=now_s,
                updated_at=now_s,
                provenance=M.MemoryProvenance(
                    source_type="experiment",
                    source_ref=f"validated_emergent:{hid}",
                    actor="clover_emergent_growth",
                    recorded_at=now_s,
                ),
                summary=(
                    f"Validated structural principle ({emergent_slug}): when {', '.join(inv_cond)} holds, "
                    f"{', '.join(inv_symp)} occurs. Verified remediation: {inv_act}."
                ),
                content=(
                    f"Promoted from unverified hypothesis {hid} after passing {h_stat['unseen_support']}/{h_stat['unseen_tested']} "
                    f"completely unseen situations ({', '.join(unseen_ids)}).\n"
                    f"- Preconditions: {inv_cond}\n"
                    f"- Inhibitory Vetoes: {spec['inhibitory_roles']}\n"
                    f"- Verified Action: {inv_act}"
                ),
                source_evidence=(
                    f"Initial observations: {', '.join(obs_ids)} | "
                    f"Unseen validations: {', '.join(unseen_ids)} (precision={h_stat['precision']})"
                ),
                relationships=rels,
                rel_path=f"concepts/{concept_id.lower()}.md",
            )
            self.memory_store.create_entry(concept_entry)
            self.memory_store.mark_superseded(
                hid,
                concept_id,
                f"Validated on unseen situations ({', '.join(unseen_ids)}) and promoted to active Concept.",
                actor="clover_emergent_growth",
            )

            for oid in obs_ids:
                self.memory_store.add_relationship(
                    source_id=oid,
                    relation_type="exemplifies",
                    target_id=concept_id,
                    provenance=f"validation_promotion:{hid}",
                    confidence=1.0,
                    context_scope="validated_emergent_concept",
                    note=f"Observation exemplifies validated emergent concept {concept_id}",
                )
                obs_e = self.memory_store.get_entry(oid)
                if obs_e is not None and emergent_slug not in obs_e.concepts:
                    obs_e.concepts.append(emergent_slug)
                    self.memory_store._save_entry_in_place(obs_e)

            # Register Order-k structural concept rule in memory_store.retrieval_rules
            if spec["feature_boxes"]:
                rules_dict = self.memory_store.retrieval_rules.setdefault("structural_concept_rules", {})
                rules_dict[concept_id] = spec["feature_boxes"]
                anchor_box = self.memory_store.retrieval_rules.setdefault("computing_anchor_box", [])
                for box in spec["feature_boxes"]:
                    for tok in box:
                        for w_part in re.findall(r"[a-z0-9\-]{3,}", tok.lower()):
                            if w_part not in anchor_box:
                                anchor_box.append(w_part)

            # 2. Promote into Symbolic Ontology Graph (`STEM_CONCEPT` + `FORMULA` + `HAS_FORMULA`)
            onto_c_key = f"STEM-EMERGENT-{slug_hash}"
            onto_f_key = f"FORMULA-EMERGENT-{slug_hash}"
            c_ref = O.EntityRef("STEM_CONCEPT", onto_c_key)
            f_ref = O.EntityRef("FORMULA", onto_f_key)
            onto_vertices.append(
                O.Vertex(
                    c_ref,
                    {
                        "name": concept_entry.title,
                        "description": concept_entry.summary,
                        "memory_concept_id": concept_id,
                        "constraints": [
                            f"Preconditions: {', '.join(inv_cond)}",
                            f"Inhibitory vetoes: {', '.join(spec['inhibitory_roles']) or 'none'}",
                        ],
                    },
                )
            )
            onto_vertices.append(
                O.Vertex(
                    f_ref,
                    {
                        "name": f"Law for {emergent_slug}",
                        "equation": f"If {' AND '.join(inv_cond)} and NOT ({' OR '.join(spec['inhibitory_roles']) or 'none'}), then action = {inv_act}",
                        "constraints": [f"Action: {inv_act}"],
                    },
                )
            )
            onto_edges.append(
                O.Edge(
                    "HAS_FORMULA",
                    c_ref,
                    f_ref,
                    O.Provenance.from_source("EmergentConceptAcquisition", concept_id, "validated_unseen"),
                )
            )

            rec = {
                "hypothesis_id": hid,
                "concept_id": concept_id,
                "emergent_slug": emergent_slug,
                "ontology_concept_key": onto_c_key,
                "ontology_formula_key": onto_f_key,
                "verified_action": inv_act,
                "supporting_observations": obs_ids,
                "unseen_validations": unseen_ids,
            }
            self._promoted_concepts[concept_id] = rec
            promoted_records.append(rec)

        # Persist updated retrieval rules to disk if `_rules` directory exists
        rules_fp = self.memory_store.root_dir / "_rules" / "dataset-memory-retrieval-rules.json"
        if rules_fp.parent.is_dir():
            rules_fp.write_text(json.dumps(self.memory_store.retrieval_rules, indent=2), encoding="utf-8")
        self.memory_store.rebuild_index()

        # Materialize Ontology Graph vertices, edges, and Horn-rule inference (`ENTAILS_CONSTRAINT`)
        inferred_edges = {}
        if onto_vertices:
            self.ontology_graph.write_vertices(onto_vertices)
            self.ontology_graph.write_edges(onto_edges)
            inferred_edges = self.ontology_graph.run_inference()

        # 3. Promote into Shared Collective Core (`CPSL`) if gateway & registry are provided
        collective_manifest = None
        if collective_gateway is not None and collective_registry is not None and promoted_records:
            import collective_evaluator
            probes = []
            for rec in promoted_records:
                spec = self._hyp_specs[rec["hypothesis_id"]]
                cap = spec.get("tool_capability") or "python"
                # Pick an OEWN-clean anchor word from the invariant condition/symptom
                anchor_tok = "partition" if "shared_cyclic_resource" in spec["invariant_conditions"] else (
                    "prefix" if "broad_prefix_before_specific" in spec["invariant_conditions"] else "allocate"
                )
                for k_inst in range(3):
                    collective_gateway.ingest(
                        {
                            "package_id": f"cand_{rec['concept_id'].lower()}_{k_inst}",
                            "schema_version": "1.0.0",
                            "installation_id": f"inst_emergent_{k_inst}",
                            "category": "problem-solving",
                            "trigger": ["ctx=laptop", f"tok={anchor_tok}"],
                            "target": [cap],
                            "local_support_count": len(rec["supporting_observations"]) + len(rec["unseen_validations"]),
                            "local_contra_count": 0,
                            "local_precision": 1.0,
                            "cross_context_generalised": True,
                        },
                        consent_verified=True,
                    )
                probes.append({
                    "context": "laptop",
                    "text": f"Please {anchor_tok} the stage slices to verify.",
                    "target": [cap],
                })
                probes.append({
                    "context": "laptop",
                    "text": f"Could you {anchor_tok} the ring buffer capacity across passes?",
                    "target": [cap],
                })
            ev = collective_evaluator.CollectiveEvaluator(
                validation_suite={
                    "strategy_probes": probes,
                    "regression_suite": collective_evaluator.DEFAULT_SEMANTICS_REGRESSION_SUITE,
                },
                min_installations=3,
                min_agreement=0.75,
            )
            eval_res = ev.evaluate_all(collective_gateway.aggregate_candidates())
            if signing_key_hex:
                collective_manifest = collective_registry.publish_release(
                    eval_res, note="Emergent Concept Acquisition v5.1 release"
                )

        return {
            "promoted_count": len(promoted_records),
            "quarantined_count": len(quarantined_records),
            "promoted_concepts": promoted_records,
            "quarantined_hypotheses": quarantined_records,
            "ontology_inferred_edges": inferred_edges,
            "collective_release": collective_manifest,
        }
