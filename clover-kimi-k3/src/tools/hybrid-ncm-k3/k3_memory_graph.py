"""Clover Kimi K3 — File-Based Memory Tree and Relational Knowledge Graph (`k3_memory_graph.py`).

Architectural Principles:
1. **Dual Complementary Structure (Files + Relationship Graph):**
   - **File Tree:** Stores individual, human-readable Markdown files (`.md`) with versioned YAML front matter (`schema_version: "1.0"`).
     The files on disk are the canonical source of truth.
   - **Relationship Graph:** Connects memory entries across the directory tree (including across different branches)
     using typed, provenance-carrying directed edges (`supports`, `challenges`, `refines`, `depends_on`, `similar_to`,
     `derived_from`, `caused_by`, `corrects`, `exemplifies`, `unresolved_with`).
   - **Rebuildable Index:** A lightweight JSON graph index (`_index/memory_graph_index.json`) accelerates lookup and
     multi-hop traversal, and can be completely deleted and rebuilt from the canonical `.md` files at any time via
     `rebuild_index()`.

2. **Epistemic & Provenance Integrity:**
   - Every memory entry has a stable `id` independent of its file path (`rel_path`), an `entry_type` (`conversation`,
     `concept`, `observation`, `correction`, `hypothesis`, `decision`, `experiment`, `outcome`), an `epistemic_status`
     (`observed`, `user_stated_belief`, `hypothesis`, `externally_verified`, `unresolved`), and a lifecycle `status`
     (`active`, `superseded`, `contested`, `under_review`).
   - Philosophical beliefs (`user_stated_belief`), open philosophical/scientific inquiries (`unresolved`), and unverified
     conjectures (`hypothesis`) are never promoted to `externally_verified` or `observed` facts without independent evidence.
   - Original conversation excerpts or source evidence live in an immutable `## Original Source Evidence` section.
     Revising a summary (`revise_summary`) or marking an entry superseded (`record_correction`, `mark_superseded`)
     appends to `## Revision History` without erasing the original evidence or interpretation.

3. **Explainable Retrieval, Evolution Chains & Bounded Context (`retrieve_context_package`):**
   - Combines direct ID lookup, concept/tag matching, lexical BM25, and multi-hop typed graph traversal (`traverse`)
     while distinguishing surface wording overlap from conceptual identity.
   - Automatically traces `corrects`, `refines`, and `derived_from` chains (`trace_evolution_chain`) so when an earlier
     interpretation is corrected later, both the original interpretation and the correction are returned in chronological
     order, clearly identifying which entry supersedes the earlier one for that context while preserving full history.
   - Surfaces `challenges` and `unresolved_with` links explicitly so contradictions are never silently flattened into agreement.
"""

from __future__ import annotations

import collections
import copy
import datetime
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

try:
    import yaml as _pyyaml
except ImportError:
    _pyyaml = None


SCHEMA_VERSION = "1.0"

VALID_ENTRY_TYPES: tuple[str, ...] = (
    "conversation",
    "concept",
    "observation",
    "correction",
    "hypothesis",
    "decision",
    "experiment",
    "outcome",
)

VALID_EPISTEMIC_STATUSES: tuple[str, ...] = (
    "observed",
    "user_stated_belief",
    "hypothesis",
    "externally_verified",
    "unresolved",
)

VALID_LIFECYCLE_STATUSES: tuple[str, ...] = (
    "active",
    "superseded",
    "contested",
    "under_review",
)

VALID_RELATIONSHIP_TYPES: tuple[str, ...] = (
    "supports",
    "challenges",
    "refines",
    "depends_on",
    "similar_to",
    "derived_from",
    "caused_by",
    "corrects",
    "exemplifies",
    "unresolved_with",
)

# Relationship priority during bounded traversal: corrections, refinements, and contradictions take precedence
RELATIONSHIP_TRAVERSAL_PRIORITY: dict[str, int] = {
    "corrects": 0,
    "refines": 1,
    "challenges": 2,
    "unresolved_with": 3,
    "derived_from": 4,
    "supports": 5,
    "caused_by": 6,
    "depends_on": 7,
    "exemplifies": 8,
    "similar_to": 9,
}

# Irreflexive relations where `source == target` is always invalid
IRREFLEXIVE_RELATIONS: frozenset[str] = frozenset({
    "corrects",
    "refines",
    "challenges",
    "caused_by",
    "derived_from",
    "supports",
})

MEMORY_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "me", "my", "no", "not", "of", "on", "or", "our", "out", "she", "so", "some", "such", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "to", "up", "us", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
    "about", "tell", "explain", "mean", "means", "meant", "does", "did", "said", "say", "says",
})


def utc_now_iso() -> str:
    """Returns an ISO-8601 UTC timestamp string (`YYYY-MM-DDTHH:MM:SSZ`)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# 1. Data Model: Provenance, Relationships, Revisions, and Memory Entries
# ---------------------------------------------------------------------------

@dataclass
class MemoryProvenance:
    source_type: str  # e.g. "conversation", "experiment", "document", "external_reference", "observation"
    source_ref: str   # e.g. "conv:phil-inside-outside-2026-10", "exp:bench_6_rust", "AGENTS.md#section-22"
    conversation_id: str | None = None
    turn_range: str | None = None
    actor: str | None = None
    recorded_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source_type": self.source_type,
            "source_ref": self.source_ref,
        }
        if self.conversation_id:
            out["conversation_id"] = self.conversation_id
        if self.turn_range:
            out["turn_range"] = self.turn_range
        if self.actor:
            out["actor"] = self.actor
        if self.recorded_at:
            out["recorded_at"] = self.recorded_at
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MemoryProvenance":
        if not isinstance(data, dict):
            return cls(source_type="unknown", source_ref="")
        return cls(
            source_type=str(data.get("source_type") or "unknown").strip(),
            source_ref=str(data.get("source_ref") or "").strip(),
            conversation_id=str(data["conversation_id"]).strip() if data.get("conversation_id") else None,
            turn_range=str(data["turn_range"]).strip() if data.get("turn_range") else None,
            actor=str(data["actor"]).strip() if data.get("actor") else None,
            recorded_at=str(data.get("recorded_at") or utc_now_iso()).strip(),
        )

    def is_valid(self) -> bool:
        return bool(self.source_type and self.source_type != "unknown" and self.source_ref)


@dataclass
class MemoryRelationship:
    type: str
    target: str
    source: str = ""
    provenance: str = ""
    confidence: float = 1.0
    context_scope: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    note: str = ""

    def to_dict(self, include_source: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if include_source and self.source:
            out["source"] = self.source
        out["type"] = self.type
        out["target"] = self.target
        if self.provenance:
            out["provenance"] = self.provenance
        out["confidence"] = round(float(self.confidence), 4)
        if self.context_scope:
            out["context_scope"] = self.context_scope
        if self.created_at:
            out["created_at"] = self.created_at
        if self.note:
            out["note"] = self.note
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_source: str = "") -> "MemoryRelationship":
        if not isinstance(data, dict):
            raise ValueError(f"Invalid relationship object: {data!r}")
        conf_raw = data.get("confidence", 1.0)
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1.0
        return cls(
            type=str(data.get("type") or "").strip(),
            target=str(data.get("target") or "").strip(),
            source=str(data.get("source") or default_source).strip(),
            provenance=str(data.get("provenance") or "").strip(),
            confidence=conf,
            context_scope=str(data.get("context_scope") or "").strip(),
            created_at=str(data.get("created_at") or utc_now_iso()).strip(),
            note=str(data.get("note") or "").strip(),
        )

    def validate(self, owner_id: str | None = None) -> list[str]:
        errors: list[str] = []
        src = owner_id or self.source
        if not self.type:
            errors.append("Relationship missing 'type'")
        elif self.type not in VALID_RELATIONSHIP_TYPES:
            errors.append(f"Invalid relationship type {self.type!r} (must be one of {VALID_RELATIONSHIP_TYPES})")
        if not self.target:
            errors.append(f"Relationship ({self.type}) from {src!r} missing 'target'")
        if src and self.target == src and self.type in IRREFLEXIVE_RELATIONS:
            errors.append(f"Self-referential {self.type!r} relationship on {src!r} is invalid")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"Relationship confidence {self.confidence} out of bounds [0.0, 1.0]")
        return errors


@dataclass
class MemoryRevision:
    timestamp: str
    actor: str
    action: str
    reason: str
    previous_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "reason": self.reason,
        }
        if self.previous_summary:
            out["previous_summary"] = self.previous_summary
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRevision":
        return cls(
            timestamp=str(data.get("timestamp") or utc_now_iso()).strip(),
            actor=str(data.get("actor") or "system").strip(),
            action=str(data.get("action") or "update").strip(),
            reason=str(data.get("reason") or "").strip(),
            previous_summary=str(data.get("previous_summary") or "").strip(),
        )


@dataclass
class MemoryEntry:
    id: str
    title: str
    entry_type: str
    epistemic_status: str
    summary: str
    content: str
    source_evidence: str
    provenance: MemoryProvenance
    status: str = "active"
    superseded_by: str | None = None
    concepts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    relationships: list[MemoryRelationship] = field(default_factory=list)
    revisions: list[MemoryRevision] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION
    rel_path: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.id or not re.match(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{2,120}$", self.id):
            errors.append(f"Invalid or missing stable ID: {self.id!r}")
        if not self.title.strip():
            errors.append(f"Entry {self.id!r} missing title")
        if self.entry_type not in VALID_ENTRY_TYPES:
            errors.append(f"Entry {self.id!r} has invalid entry_type {self.entry_type!r}")
        if self.epistemic_status not in VALID_EPISTEMIC_STATUSES:
            errors.append(f"Entry {self.id!r} has invalid epistemic_status {self.epistemic_status!r}")
        if self.status not in VALID_LIFECYCLE_STATUSES:
            errors.append(f"Entry {self.id!r} has invalid lifecycle status {self.status!r}")
        if not self.created_at or not self.updated_at:
            errors.append(f"Entry {self.id!r} missing created_at or updated_at timestamp")
        if not self.provenance.is_valid():
            errors.append(f"Entry {self.id!r} has incomplete provenance: {self.provenance.to_dict()}")
        if not self.content.strip() and not self.summary.strip():
            errors.append(f"Entry {self.id!r} has empty content and summary")
        for rel in self.relationships:
            rel.source = self.id
            errors.extend(rel.validate(owner_id=self.id))
        return errors

    def front_matter_dict(self) -> dict[str, Any]:
        fm: dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "title": self.title,
            "entry_type": self.entry_type,
            "epistemic_status": self.epistemic_status,
            "status": self.status,
        }
        if self.superseded_by:
            fm["superseded_by"] = self.superseded_by
        if self.concepts:
            fm["concepts"] = list(self.concepts)
        if self.tags:
            fm["tags"] = list(self.tags)
        fm["created_at"] = self.created_at
        fm["updated_at"] = self.updated_at
        fm["provenance"] = self.provenance.to_dict()
        fm["relationships"] = [r.to_dict(include_source=False) for r in self.relationships]
        if self.uncertainty_notes:
            fm["uncertainty_notes"] = list(self.uncertainty_notes)
        if self.revisions:
            fm["revisions"] = [rev.to_dict() for rev in self.revisions]
        return fm


# ---------------------------------------------------------------------------
# 2. Human-Readable Markdown + YAML Front Matter Serializer & Parser
# ---------------------------------------------------------------------------

def _dump_yaml_front_matter(data: dict[str, Any]) -> str:
    """Serializes `data` into clean, human-readable YAML front matter."""
    if _pyyaml is not None:
        return _pyyaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100).strip()
    return json.dumps(data, indent=2, ensure_ascii=False)


def _load_yaml_front_matter(raw_yaml: str) -> dict[str, Any]:
    """Parses YAML (or JSON-subset YAML) front matter into a dictionary."""
    text = raw_yaml.strip()
    if not text:
        return {}
    if _pyyaml is not None:
        loaded = _pyyaml.safe_load(text)
        if isinstance(loaded, dict):
            return loaded
        raise ValueError("YAML front matter did not parse to a mapping")
    return json.loads(text)


def render_memory_markdown(entry: MemoryEntry) -> str:
    """Renders a `MemoryEntry` into a canonical, human-readable Markdown file with YAML front matter."""
    yaml_str = _dump_yaml_front_matter(entry.front_matter_dict())
    lines: list[str] = [
        "---",
        yaml_str,
        "---",
        "",
        f"# {entry.title}",
        "",
        "## Summary",
        entry.summary.strip() or "(no summary)",
        "",
        "## Content",
        entry.content.strip() or entry.summary.strip(),
        "",
        "## Original Source Evidence",
        entry.source_evidence.strip() or "(none recorded)",
        "",
        "## Revision History",
    ]
    if entry.revisions:
        for rev in entry.revisions:
            prev_part = f" | Previous summary: {rev.previous_summary!r}" if rev.previous_summary else ""
            lines.append(f"- `{rev.timestamp}` (**{rev.actor}**, `{rev.action}`): {rev.reason}{prev_part}")
    else:
        lines.append(f"- `{entry.created_at}` (**{entry.provenance.actor or 'system'}**, `created`): Initial entry recorded from `{entry.provenance.source_ref}`.")
    lines.append("")
    return "\n".join(lines)


_SECTION_HEADER_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def parse_memory_markdown(markdown_text: str, rel_path: str = "") -> MemoryEntry:
    """Parses a canonical `.md` memory file into a `MemoryEntry` instance."""
    text = markdown_text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise ValueError(f"Memory file {rel_path!r} is missing opening YAML front matter ('---')")
    end_fm = text.find("\n---\n", 4)
    if end_fm == -1:
        raise ValueError(f"Memory file {rel_path!r} is missing closing YAML front matter ('---')")

    raw_fm = text[4:end_fm]
    body = text[end_fm + 5:]
    fm = _load_yaml_front_matter(raw_fm)

    sections: dict[str, str] = {}
    matches = list(_SECTION_HEADER_RE.finditer(body))
    for idx, m in enumerate(matches):
        sec_name = m.group(1).strip().lower()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        sections[sec_name] = body[start:end].strip()

    entry_id = str(fm.get("id") or "").strip()
    title = str(fm.get("title") or "").strip()
    if not title:
        m_h1 = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
        if m_h1:
            title = m_h1.group(1).strip()

    summary = sections.get("summary", "").strip()
    content = sections.get("content", "").strip()
    if not content and not matches and body.strip():
        content = body.strip()
    source_evidence = sections.get("original source evidence", "").strip()
    if source_evidence == "(none recorded)":
        source_evidence = ""

    raw_rels = fm.get("relationships") or []
    rels: list[MemoryRelationship] = []
    if isinstance(raw_rels, list):
        for r_obj in raw_rels:
            rels.append(MemoryRelationship.from_dict(r_obj, default_source=entry_id))

    raw_revs = fm.get("revisions") or []
    revs: list[MemoryRevision] = []
    if isinstance(raw_revs, list):
        for rv_obj in raw_revs:
            if isinstance(rv_obj, dict):
                revs.append(MemoryRevision.from_dict(rv_obj))

    return MemoryEntry(
        id=entry_id,
        title=title,
        entry_type=str(fm.get("entry_type") or "").strip(),
        epistemic_status=str(fm.get("epistemic_status") or "").strip(),
        summary=summary,
        content=content,
        source_evidence=source_evidence,
        provenance=MemoryProvenance.from_dict(fm.get("provenance")),
        status=str(fm.get("status") or "active").strip(),
        superseded_by=str(fm["superseded_by"]).strip() if fm.get("superseded_by") else None,
        concepts=[str(x).strip() for x in (fm.get("concepts") or []) if str(x).strip()],
        tags=[str(x).strip() for x in (fm.get("tags") or []) if str(x).strip()],
        relationships=rels,
        revisions=revs,
        uncertainty_notes=[str(x).strip() for x in (fm.get("uncertainty_notes") or []) if str(x).strip()],
        created_at=str(fm.get("created_at") or "").strip(),
        updated_at=str(fm.get("updated_at") or "").strip(),
        schema_version=str(fm.get("schema_version") or SCHEMA_VERSION).strip(),
        rel_path=rel_path.replace("\\", "/"),
    )


# ---------------------------------------------------------------------------
# 3. Graph Validation, Traversal, Evolution Chains & Retrieval Structures
# ---------------------------------------------------------------------------

@dataclass
class GraphValidationReport:
    total_files: int
    valid_entries: int
    total_relationships: int
    duplicate_ids: list[dict[str, Any]] = field(default_factory=list)
    broken_links: list[dict[str, Any]] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    cycle_warnings: list[dict[str, Any]] = field(default_factory=list)
    unresolved_conflicts: list[dict[str, Any]] = field(default_factory=list)
    uncertain_duplicates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return (
            not self.duplicate_ids
            and not self.broken_links
            and not self.validation_errors
            and not self.cycle_warnings
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_healthy": self.is_healthy,
            "total_files": self.total_files,
            "valid_entries": self.valid_entries,
            "total_relationships": self.total_relationships,
            "duplicate_ids": self.duplicate_ids,
            "broken_links": self.broken_links,
            "validation_errors": self.validation_errors,
            "cycle_warnings": self.cycle_warnings,
            "unresolved_conflicts": self.unresolved_conflicts,
            "uncertain_duplicates": self.uncertain_duplicates,
        }


@dataclass
class TraversalStep:
    from_id: str
    to_id: str
    relation_type: str
    direction: Literal["outgoing", "incoming"]
    confidence: float
    provenance: str
    note: str = ""

    def format_hop(self) -> str:
        arrow = f"--[{self.relation_type}]-->" if self.direction == "outgoing" else f"<--[{self.relation_type}]--"
        return f"{self.from_id} {arrow} {self.to_id}"


@dataclass
class TraversalResultNode:
    entry: MemoryEntry
    depth: int
    path: list[TraversalStep]

    @property
    def path_string(self) -> str:
        if not self.path:
            return f"SEED({self.entry.id})"
        parts = [f"SEED({self.path[0].from_id})"]
        for step in self.path:
            arrow = f"--[{step.relation_type}]-->" if step.direction == "outgoing" else f"<--[{step.relation_type}]--"
            parts.append(f"{arrow} {step.to_id}")
        return " ".join(parts)


@dataclass
class EvolutionChainNode:
    id: str
    title: str
    entry_type: str
    epistemic_status: str
    status: str
    superseded_by: str | None
    created_at: str
    role_in_chain: str  # "source_origin" | "superseded_interpretation" | "active_interpretation" | "active_correction" | "refinement"
    summary: str
    source_evidence: str
    rel_path: str


@dataclass
class RetrievedMemoryItem:
    entry: MemoryEntry
    relevance_score: float
    retrieval_reason: str
    relationship_path: str
    evolution_role: str  # "active_primary", "superseded_interpretation", "active_correction", "refinement", "challenged_claim", "supporting_evidence"
    supersedes_ids: list[str] = field(default_factory=list)
    refines_ids: list[str] = field(default_factory=list)
    superseded_by_id: str | None = None
    contradictions: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.entry.id,
            "title": self.entry.title,
            "rel_path": self.entry.rel_path,
            "entry_type": self.entry.entry_type,
            "epistemic_status": self.entry.epistemic_status,
            "status": self.entry.status,
            "superseded_by": self.superseded_by_id,
            "supersedes": self.supersedes_ids,
            "refines": self.refines_ids,
            "evolution_role": self.evolution_role,
            "relevance_score": round(self.relevance_score, 4),
            "retrieval_reason": self.retrieval_reason,
            "relationship_path": self.relationship_path,
            "provenance": self.entry.provenance.to_dict(),
            "summary": self.entry.summary,
            "content": self.entry.content,
            "source_evidence": self.entry.source_evidence,
            "contradictions": self.contradictions,
            "uncertainty_notes": self.entry.uncertainty_notes,
        }


@dataclass
class MemoryContextPackage:
    query: str
    items: list[RetrievedMemoryItem]
    evolution_chains: list[list[EvolutionChainNode]]
    active_contradictions: list[dict[str, str]]
    total_candidates_considered: int
    traversal_depth_used: int
    token_budget: int
    estimated_tokens_used: int

    def render_for_k3(self) -> str:
        """Renders a bounded, explainable context package for Clover Kimi K3 that preserves
        epistemic status, source evidence, relationship paths, superseding corrections, and unresolved contradictions."""
        if not self.items:
            return "(no relevant persistent memory entries retrieved)"

        blocks: list[str] = [
            "=== Clover Kimi K3 Persistent Memory & Relational Knowledge Graph ===",
            "Epistemic Rule: Distinguish 'externally_verified' and 'observed' facts from 'user_stated_belief', 'hypothesis', and 'unresolved' inquiry. "
            "When an earlier interpretation is marked 'superseded' by a 'correction', report both the evolution of understanding and which interpretation currently supersedes the earlier one for that context. "
            "Never silently flatten 'challenges' or 'unresolved_with' contradictions into agreement.",
        ]

        for idx, item in enumerate(self.items, start=1):
            e = item.entry
            status_badge = f"status={e.status}"
            if item.superseded_by_id:
                status_badge += f" (SUPERSEDED BY {item.superseded_by_id})"
            if item.supersedes_ids:
                status_badge += f" (CORRECTS/SUPERSEDES {', '.join(item.supersedes_ids)})"
            if item.refines_ids:
                status_badge += f" (REFINES {', '.join(item.refines_ids)})"

            header = (
                f"[M{idx}] {e.id} — {e.title}\n"
                f"  File: {e.rel_path} | Type: {e.entry_type} | Epistemic Status: {e.epistemic_status} | Lifecycle: {status_badge}\n"
                f"  Provenance: {e.provenance.source_type} ({e.provenance.source_ref}"
                + (f", {e.provenance.turn_range}" if e.provenance.turn_range else "")
                + (f", actor={e.provenance.actor}" if e.provenance.actor else "")
                + f") | Recorded: {e.created_at}\n"
                f"  Retrieval Path: {item.relationship_path} ({item.retrieval_reason})"
            )
            body_lines = [header, f"  Summary: {e.summary}"]
            if e.content and e.content != e.summary:
                body_lines.append(f"  Content: {e.content}")
            if e.source_evidence:
                body_lines.append(f"  Original Source Evidence: {e.source_evidence}")
            if item.contradictions:
                c_strs = [
                    f"{c['relation']} with {c['other_id']} ({c.get('note') or 'contested/unresolved'})"
                    for c in item.contradictions
                ]
                body_lines.append(f"  Contradictions / Open Tensions: {'; '.join(c_strs)}")
            if e.uncertainty_notes:
                body_lines.append(f"  Uncertainty Notes: {'; '.join(e.uncertainty_notes)}")
            blocks.append("\n".join(body_lines))

        if self.evolution_chains:
            chain_lines = ["--- Traceable Interpretation & Correction Chains ---"]
            for c_idx, chain in enumerate(self.evolution_chains, start=1):
                steps_fmt = "  -->  ".join(
                    f"{node.id} [{node.entry_type}, {node.role_in_chain}, epistemic={node.epistemic_status}, created={node.created_at}]"
                    for node in chain
                )
                chain_lines.append(f"  Chain {c_idx}: {steps_fmt}")
            blocks.append("\n".join(chain_lines))

        if self.active_contradictions:
            conf_lines = ["--- Unresolved / Contested Claims (Do Not Flatten Into Agreement) ---"]
            for c in self.active_contradictions:
                conf_lines.append(
                    f"  • {c['source']} --[{c['type']}]--> {c['target']}: {c.get('note') or 'Contested interpretation retained'}"
                )
            blocks.append("\n".join(conf_lines))

        return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 4. Main Engine: `CloverMemoryGraphStore` (File Tree + Rebuildable Graph Index)
# ---------------------------------------------------------------------------

class CloverMemoryGraphStore:
    """Manages a human-readable Markdown file tree and a rebuildable typed relationship graph index."""

    def __init__(self, root_dir: str | os.PathLike[str], auto_load: bool = True) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.index_dir = self.root_dir / "_index"
        self.index_file = self.index_dir / "memory_graph_index.json"

        self.entries: dict[str, MemoryEntry] = {}
        self.id_to_rel_path: dict[str, str] = {}
        self.outgoing: dict[str, list[MemoryRelationship]] = collections.defaultdict(list)
        self.incoming: dict[str, list[MemoryRelationship]] = collections.defaultdict(list)
        self.last_validation: GraphValidationReport | None = None

        if auto_load and self.root_dir.exists():
            self.rebuild_index()

    # ------------------------------------------------------------------
    # 4.1 Index Building, Persistence & Validation from Canonical Files
    # ------------------------------------------------------------------

    def _iter_markdown_files(self) -> list[Path]:
        if not self.root_dir.exists():
            return []
        md_files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Skip hidden directories and `_index`
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "_index")
            for fn in sorted(filenames):
                if fn.endswith(".md") and not fn.startswith("_") and fn.upper() != "README.MD":
                    md_files.append(Path(dirpath) / fn)
        return md_files

    def rebuild_index(self, write_index_file: bool = True) -> GraphValidationReport:
        """Scans all canonical `.md` files in `self.root_dir`, validates metadata and relationships,
        rebuilds the outgoing/incoming adjacency lists and superseded links, and writes `_index/memory_graph_index.json`."""
        self.entries.clear()
        self.id_to_rel_path.clear()
        self.outgoing.clear()
        self.incoming.clear()

        md_files = self._iter_markdown_files()
        duplicate_ids: list[dict[str, Any]] = []
        broken_links: list[dict[str, Any]] = []
        validation_errors: list[str] = []
        cycle_warnings: list[dict[str, Any]] = []
        unresolved_conflicts: list[dict[str, Any]] = []

        # Pass 1: Parse all Markdown files and check entry-level schema + duplicate IDs
        for fp in md_files:
            rel_p = fp.relative_to(self.root_dir).as_posix()
            try:
                raw_txt = fp.read_text(encoding="utf-8")
                entry = parse_memory_markdown(raw_txt, rel_path=rel_p)
            except Exception as exc:
                validation_errors.append(f"Failed to parse {rel_p}: {exc}")
                continue

            entry_errs = entry.validate()
            if entry_errs:
                validation_errors.extend(f"[{rel_p}] {e}" for e in entry_errs)

            if entry.id in self.entries:
                duplicate_ids.append({
                    "id": entry.id,
                    "first_path": self.id_to_rel_path[entry.id],
                    "second_path": rel_p,
                })
                validation_errors.append(
                    f"Duplicate stable ID {entry.id!r} in {rel_p} (already defined in {self.id_to_rel_path[entry.id]})"
                )
                continue

            self.entries[entry.id] = entry
            self.id_to_rel_path[entry.id] = rel_p

        # Pass 2: Build outgoing & incoming relationship indices and detect broken links / contradictions
        total_rels = 0
        for src_id, entry in self.entries.items():
            for rel in entry.relationships:
                rel.source = src_id
                total_rels += 1
                if rel.type not in VALID_RELATIONSHIP_TYPES:
                    continue
                if rel.target not in self.entries:
                    broken_links.append({
                        "source": src_id,
                        "source_path": entry.rel_path,
                        "type": rel.type,
                        "missing_target": rel.target,
                    })
                    continue
                self.outgoing[src_id].append(rel)
                self.incoming[rel.target].append(rel)

                if rel.type in ("challenges", "unresolved_with"):
                    unresolved_conflicts.append({
                        "source": src_id,
                        "target": rel.target,
                        "type": rel.type,
                        "note": rel.note,
                        "provenance": rel.provenance,
                    })

            # Also check if `superseded_by` points to a non-existent entry
            if entry.superseded_by and entry.superseded_by not in self.entries:
                broken_links.append({
                    "source": src_id,
                    "source_path": entry.rel_path,
                    "type": "superseded_by",
                    "missing_target": entry.superseded_by,
                })

        # Pass 3: Synchronize derived `superseded_by` status in-memory when `A --corrects--> B` exists
        for src_id, rel_list in self.outgoing.items():
            src_entry = self.entries[src_id]
            if src_entry.status == "superseded":
                continue
            for rel in rel_list:
                if rel.type == "corrects" and rel.target in self.entries:
                    tgt_entry = self.entries[rel.target]
                    if not tgt_entry.superseded_by:
                        tgt_entry.superseded_by = src_id
                    if tgt_entry.status == "active":
                        tgt_entry.status = "superseded"

        # Pass 4: Detect invalid `corrects` cycles (while permitting general knowledge graph cycles)
        for start_id in self.entries:
            visited_corr: list[str] = [start_id]
            cur = start_id
            while True:
                corr_targets = [r.target for r in self.outgoing.get(cur, []) if r.type == "corrects" and r.target in self.entries]
                if not corr_targets:
                    break
                nxt = corr_targets[0]
                if nxt in visited_corr:
                    if nxt == start_id:
                        cycle_warnings.append({
                            "relation": "corrects",
                            "cycle": visited_corr + [nxt],
                            "message": f"Cyclic 'corrects' chain detected: {' -> '.join(visited_corr + [nxt])}",
                        })
                    break
                visited_corr.append(nxt)
                cur = nxt

        uncertain_dups = self._find_uncertain_concept_duplicates()

        report = GraphValidationReport(
            total_files=len(md_files),
            valid_entries=len(self.entries),
            total_relationships=total_rels,
            duplicate_ids=duplicate_ids,
            broken_links=broken_links,
            validation_errors=validation_errors,
            cycle_warnings=cycle_warnings,
            unresolved_conflicts=unresolved_conflicts,
            uncertain_duplicates=uncertain_dups,
        )
        self.last_validation = report

        if write_index_file and self.root_dir.exists():
            self._write_index_cache(report)

        return report

    def _write_index_cache(self, report: GraphValidationReport) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "built_at": utc_now_iso(),
            "validation": report.to_dict(),
            "nodes": {
                eid: {
                    "id": e.id,
                    "title": e.title,
                    "rel_path": e.rel_path,
                    "entry_type": e.entry_type,
                    "epistemic_status": e.epistemic_status,
                    "status": e.status,
                    "superseded_by": e.superseded_by,
                    "concepts": e.concepts,
                    "tags": e.tags,
                    "created_at": e.created_at,
                    "updated_at": e.updated_at,
                    "provenance": e.provenance.to_dict(),
                }
                for eid, e in sorted(self.entries.items())
            },
            "edges": [
                rel.to_dict(include_source=True)
                for eid in sorted(self.outgoing.keys())
                for rel in self.outgoing[eid]
            ],
        }
        self.index_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _find_uncertain_concept_duplicates(self) -> list[dict[str, Any]]:
        """Detects pairs of active entries that share a high fraction of title/concept tokens
        without an explicit relationship connecting them, flagging them for human review rather than
        automatically merging them (Section 6)."""
        candidates: list[dict[str, Any]] = []
        active_entries = [e for e in self.entries.values() if e.status == "active"]
        for i in range(len(active_entries)):
            e1 = active_entries[i]
            connected_ids = {r.target for r in self.outgoing.get(e1.id, [])} | {
                r.source for r in self.incoming.get(e1.id, [])
            }
            c1 = {c.lower() for c in e1.concepts}
            w1 = {w for w in re.findall(r"[a-z0-9]{3,}", e1.title.lower()) if w not in MEMORY_STOPWORDS}
            for j in range(i + 1, len(active_entries)):
                e2 = active_entries[j]
                if e2.id in connected_ids:
                    continue
                c2 = {c.lower() for c in e2.concepts}
                w2 = {w for w in re.findall(r"[a-z0-9]{3,}", e2.title.lower()) if w not in MEMORY_STOPWORDS}
                shared_concepts = sorted(c1 & c2)
                union_w = w1 | w2
                jacc_w = len(w1 & w2) / len(union_w) if union_w else 0.0
                if (shared_concepts and jacc_w >= 0.45) or jacc_w >= 0.65:
                    candidates.append({
                        "entry_a": e1.id,
                        "entry_b": e2.id,
                        "shared_concepts": shared_concepts,
                        "title_jaccard": round(jacc_w, 3),
                        "resolution": "retained_separate_pending_review",
                    })
        return candidates

    # ------------------------------------------------------------------
    # 4.2 Outgoing / Incoming Lookups & Cycle-Safe Graph Traversal
    # ------------------------------------------------------------------

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        return self.entries.get(entry_id)

    def get_outgoing(
        self, entry_id: str, relation_types: Sequence[str] | None = None
    ) -> list[MemoryRelationship]:
        rels = self.outgoing.get(entry_id, [])
        if relation_types is None:
            return list(rels)
        allowed = set(relation_types)
        return [r for r in rels if r.type in allowed]

    def get_incoming(
        self, entry_id: str, relation_types: Sequence[str] | None = None
    ) -> list[MemoryRelationship]:
        rels = self.incoming.get(entry_id, [])
        if relation_types is None:
            return list(rels)
        allowed = set(relation_types)
        return [r for r in rels if r.type in allowed]

    def traverse(
        self,
        seed_ids: Sequence[str],
        max_depth: int = 2,
        direction: Literal["both", "outgoing", "incoming"] = "both",
        relation_types: Sequence[str] | None = None,
        max_nodes: int = 32,
    ) -> list[TraversalResultNode]:
        """Performs deterministic, cycle-safe BFS traversal across directory boundaries starting from `seed_ids`.
        Prioritizes `corrects`, `refines`, `challenges`, and `unresolved_with` edges ahead of generic similarity."""
        allowed_rels = set(relation_types) if relation_types is not None else set(VALID_RELATIONSHIP_TYPES)
        visited: set[str] = set()
        results: list[TraversalResultNode] = []
        queue: collections.deque[tuple[str, int, list[TraversalStep]]] = collections.deque()

        for sid in seed_ids:
            if sid in self.entries and sid not in visited:
                visited.add(sid)
                seed_node = TraversalResultNode(entry=self.entries[sid], depth=0, path=[])
                results.append(seed_node)
                queue.append((sid, 0, []))

        while queue and len(results) < max_nodes:
            cur_id, depth, cur_path = queue.popleft()
            if depth >= max_depth:
                continue

            candidate_steps: list[tuple[int, float, TraversalStep]] = []
            if direction in ("both", "outgoing"):
                for r in self.outgoing.get(cur_id, []):
                    if r.type in allowed_rels and r.target in self.entries:
                        prio = RELATIONSHIP_TRAVERSAL_PRIORITY.get(r.type, 50)
                        step = TraversalStep(
                            from_id=cur_id,
                            to_id=r.target,
                            relation_type=r.type,
                            direction="outgoing",
                            confidence=r.confidence,
                            provenance=r.provenance,
                            note=r.note,
                        )
                        candidate_steps.append((prio, -r.confidence, step))

            if direction in ("both", "incoming"):
                for r in self.incoming.get(cur_id, []):
                    if r.type in allowed_rels and r.source in self.entries:
                        prio = RELATIONSHIP_TRAVERSAL_PRIORITY.get(r.type, 50)
                        step = TraversalStep(
                            from_id=cur_id,
                            to_id=r.source,
                            relation_type=r.type,
                            direction="incoming",
                            confidence=r.confidence,
                            provenance=r.provenance,
                            note=r.note,
                        )
                        candidate_steps.append((prio, -r.confidence, step))

            candidate_steps.sort(key=lambda x: (x[0], x[1], x[2].to_id))

            for _, _, step in candidate_steps:
                if step.to_id in visited:
                    continue
                visited.add(step.to_id)
                next_path = cur_path + [step]
                results.append(
                    TraversalResultNode(
                        entry=self.entries[step.to_id],
                        depth=depth + 1,
                        path=next_path,
                    )
                )
                if len(results) >= max_nodes:
                    break
                queue.append((step.to_id, depth + 1, next_path))

        return results

    def trace_evolution_chain(self, entry_id: str) -> list[EvolutionChainNode]:
        """Traces the full chronological evolution chain (`derived_from`, `corrects`, `refines`)
        around `entry_id`, returning originating conversations/sources, earlier superseded interpretations,
        and later corrections/refinements in chronological order."""
        if entry_id not in self.entries:
            return []

        chain_ids: set[str] = {entry_id}
        frontier: collections.deque[str] = collections.deque([entry_id])

        while frontier:
            cid = frontier.popleft()
            # Walk outgoing `corrects`, `refines`, `derived_from`
            for r in self.outgoing.get(cid, []):
                if r.type in ("corrects", "refines", "derived_from") and r.target in self.entries:
                    if r.target not in chain_ids:
                        chain_ids.add(r.target)
                        frontier.append(r.target)
            # Walk incoming `corrects`, `refines`, `derived_from`
            for r in self.incoming.get(cid, []):
                if r.type in ("corrects", "refines", "derived_from") and r.source in self.entries:
                    if r.source not in chain_ids:
                        chain_ids.add(r.source)
                        frontier.append(r.source)
            # Also follow explicit `superseded_by`
            sup_id = self.entries[cid].superseded_by
            if sup_id and sup_id in self.entries and sup_id not in chain_ids:
                chain_ids.add(sup_id)
                frontier.append(sup_id)

        def _chain_role(e: MemoryEntry) -> str:
            if e.entry_type == "conversation":
                return "source_origin"
            if e.status == "superseded" or e.superseded_by:
                return "superseded_interpretation"
            has_outgoing_corrects = any(r.type == "corrects" for r in self.outgoing.get(e.id, []))
            if e.entry_type == "correction" or has_outgoing_corrects:
                return "active_correction"
            has_outgoing_refines = any(r.type == "refines" for r in self.outgoing.get(e.id, []))
            if has_outgoing_refines:
                return "refinement"
            return "active_interpretation"

        # Order chronologically by `created_at`, breaking ties by role priority (source -> superseded -> refinement -> active_correction)
        role_order = {
            "source_origin": 0,
            "superseded_interpretation": 1,
            "active_interpretation": 2,
            "refinement": 3,
            "active_correction": 4,
        }
        nodes = [
            EvolutionChainNode(
                id=e.id,
                title=e.title,
                entry_type=e.entry_type,
                epistemic_status=e.epistemic_status,
                status=e.status,
                superseded_by=e.superseded_by,
                created_at=e.created_at,
                role_in_chain=_chain_role(e),
                summary=e.summary,
                source_evidence=e.source_evidence,
                rel_path=e.rel_path,
            )
            for eid in chain_ids
            for e in [self.entries[eid]]
        ]
        nodes.sort(key=lambda n: (n.created_at, role_order.get(n.role_in_chain, 9), n.id))
        return nodes

    # ------------------------------------------------------------------
    # 4.3 Safe Memory Update Operations (Section 6)
    # ------------------------------------------------------------------

    def create_entry(
        self,
        entry: MemoryEntry,
        rel_path: str | None = None,
        overwrite: bool = False,
    ) -> MemoryEntry:
        """Creates a new human-readable `.md` memory file in the tree, validates its schema,
        and updates the graph index. Refuses to overwrite an existing ID or file unless `overwrite=True`."""
        errs = entry.validate()
        if errs:
            raise ValueError(f"Cannot create invalid MemoryEntry {entry.id!r}: {'; '.join(errs)}")
        if not overwrite and entry.id in self.entries:
            raise ValueError(f"MemoryEntry with id={entry.id!r} already exists at {self.id_to_rel_path[entry.id]!r}")

        if rel_path:
            clean_rel = rel_path.replace("\\", "/").lstrip("/")
        elif entry.rel_path:
            clean_rel = entry.rel_path.replace("\\", "/").lstrip("/")
        else:
            folder = f"{entry.entry_type}s" if not entry.entry_type.endswith("s") else entry.entry_type
            safe_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "-", entry.id.lower()).strip("-")
            clean_rel = f"{folder}/{safe_slug}.md"

        if not clean_rel.endswith(".md"):
            clean_rel += ".md"

        target_fp = (self.root_dir / clean_rel).resolve()
        if not str(target_fp).startswith(str(self.root_dir)):
            raise ValueError(f"Relative path {clean_rel!r} escapes memory tree root {self.root_dir}")
        if not overwrite and target_fp.exists():
            raise ValueError(f"Memory file already exists at {clean_rel!r}")

        entry.rel_path = clean_rel
        if not entry.revisions:
            entry.revisions.append(
                MemoryRevision(
                    timestamp=entry.created_at,
                    actor=entry.provenance.actor or "system",
                    action="created",
                    reason=f"Initial {entry.entry_type} entry recorded from {entry.provenance.source_ref}",
                )
            )

        target_fp.parent.mkdir(parents=True, exist_ok=True)
        target_fp.write_text(render_memory_markdown(entry), encoding="utf-8")
        self.rebuild_index()
        return self.entries[entry.id]

    def _save_entry_in_place(self, entry: MemoryEntry) -> None:
        if not entry.rel_path:
            raise ValueError(f"Entry {entry.id!r} has no rel_path")
        errs = entry.validate()
        if errs:
            raise ValueError(f"Validation failed before saving {entry.id!r}: {'; '.join(errs)}")
        fp = (self.root_dir / entry.rel_path).resolve()
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(render_memory_markdown(entry), encoding="utf-8")

    def add_relationship(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        provenance: str,
        confidence: float = 1.0,
        context_scope: str = "",
        note: str = "",
        actor: str = "system",
        allow_missing_target: bool = False,
    ) -> MemoryRelationship:
        """Safely adds a validated typed relationship to `source_id`'s canonical file and rebuilds the index."""
        if source_id not in self.entries:
            raise KeyError(f"Source entry {source_id!r} not found")
        if not allow_missing_target and target_id not in self.entries:
            raise KeyError(f"Target entry {target_id!r} not found in memory tree")

        new_rel = MemoryRelationship(
            type=relation_type,
            target=target_id,
            source=source_id,
            provenance=provenance,
            confidence=confidence,
            context_scope=context_scope,
            created_at=utc_now_iso(),
            note=note,
        )
        rel_errs = new_rel.validate(owner_id=source_id)
        if rel_errs:
            raise ValueError("; ".join(rel_errs))

        entry = self.entries[source_id]
        # Prevent duplicate identical (type, target) edge; update metadata if already present
        for existing in entry.relationships:
            if existing.type == relation_type and existing.target == target_id:
                existing.provenance = provenance or existing.provenance
                existing.confidence = confidence
                existing.context_scope = context_scope or existing.context_scope
                existing.note = note or existing.note
                entry.updated_at = utc_now_iso()
                self._save_entry_in_place(entry)
                self.rebuild_index()
                return existing

        entry.relationships.append(new_rel)
        now_ts = utc_now_iso()
        entry.updated_at = now_ts
        entry.revisions.append(
            MemoryRevision(
                timestamp=now_ts,
                actor=actor,
                action="add_relationship",
                reason=f"Added relationship {relation_type} -> {target_id} ({provenance})",
            )
        )
        self._save_entry_in_place(entry)
        self.rebuild_index()
        return new_rel

    def remove_relationship(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        actor: str = "system",
        reason: str = "Removed relationship",
    ) -> bool:
        """Removes a relationship `(source_id, relation_type, target_id)` from `source_id` and rebuilds the index."""
        if source_id not in self.entries:
            raise KeyError(f"Source entry {source_id!r} not found")
        entry = self.entries[source_id]
        before_len = len(entry.relationships)
        entry.relationships = [
            r for r in entry.relationships if not (r.type == relation_type and r.target == target_id)
        ]
        if len(entry.relationships) == before_len:
            return False
        now_ts = utc_now_iso()
        entry.updated_at = now_ts
        entry.revisions.append(
            MemoryRevision(
                timestamp=now_ts,
                actor=actor,
                action="remove_relationship",
                reason=f"{reason}: {relation_type} -> {target_id}",
            )
        )
        self._save_entry_in_place(entry)
        self.rebuild_index()
        return True

    def revise_summary(
        self,
        entry_id: str,
        new_summary: str,
        actor: str = "system",
        reason: str = "Summary revised",
    ) -> MemoryEntry:
        """Revises an entry's `## Summary` while strictly preserving `## Original Source Evidence`
        and recording the previous summary in `## Revision History`."""
        if entry_id not in self.entries:
            raise KeyError(f"Entry {entry_id!r} not found")
        if not new_summary.strip():
            raise ValueError("new_summary must not be empty")

        entry = self.entries[entry_id]
        old_summary = entry.summary
        now_ts = utc_now_iso()
        entry.summary = new_summary.strip()
        entry.updated_at = now_ts
        entry.revisions.append(
            MemoryRevision(
                timestamp=now_ts,
                actor=actor,
                action="revise_summary",
                reason=reason,
                previous_summary=old_summary,
            )
        )
        self._save_entry_in_place(entry)
        self.rebuild_index()
        return self.entries[entry_id]

    def mark_superseded(
        self,
        entry_id: str,
        superseded_by_id: str,
        reason: str,
        actor: str = "system",
    ) -> MemoryEntry:
        """Marks `entry_id` as `superseded` by `superseded_by_id` without erasing its content,
        original source evidence, or revision history."""
        if entry_id not in self.entries:
            raise KeyError(f"Entry {entry_id!r} not found")
        if superseded_by_id not in self.entries:
            raise KeyError(f"Superseding entry {superseded_by_id!r} not found")
        if entry_id == superseded_by_id:
            raise ValueError("An entry cannot supersede itself")

        entry = self.entries[entry_id]
        now_ts = utc_now_iso()
        entry.status = "superseded"
        entry.superseded_by = superseded_by_id
        entry.updated_at = now_ts
        entry.revisions.append(
            MemoryRevision(
                timestamp=now_ts,
                actor=actor,
                action="mark_superseded",
                reason=f"Superseded by {superseded_by_id}: {reason}",
            )
        )
        self._save_entry_in_place(entry)
        self.rebuild_index()
        return self.entries[entry_id]

    def record_correction(
        self,
        correction_id: str,
        target_id: str,
        title: str,
        summary: str,
        content: str,
        source_evidence: str,
        provenance: MemoryProvenance,
        epistemic_status: str = "observed",
        context_scope: str = "",
        concepts: list[str] | None = None,
        tags: list[str] | None = None,
        additional_relationships: list[MemoryRelationship] | None = None,
        rel_path: str | None = None,
        mark_target_superseded: bool = True,
    ) -> tuple[MemoryEntry, MemoryEntry]:
        """Records a new `correction` entry, links it via `corrects -> target_id`, and (when
        `mark_target_superseded=True`) marks `target_id` as `superseded` by `correction_id` while
        preserving `target_id`'s original statement, source evidence, and revision history."""
        if target_id not in self.entries:
            raise KeyError(f"Target entry to correct {target_id!r} does not exist")
        target_entry = self.entries[target_id]

        rels: list[MemoryRelationship] = [
            MemoryRelationship(
                type="corrects",
                target=target_id,
                source=correction_id,
                provenance=provenance.source_ref,
                confidence=1.0,
                context_scope=context_scope,
                created_at=provenance.recorded_at or utc_now_iso(),
                note=f"Corrects interpretation in {target_id}: {summary[:140]}",
            )
        ]
        if additional_relationships:
            for ar in additional_relationships:
                if not (ar.type == "corrects" and ar.target == target_id):
                    rels.append(ar)

        merged_concepts = list(dict.fromkeys((concepts or []) + target_entry.concepts))
        merged_tags = list(dict.fromkeys((tags or []) + ["correction"]))

        corr_entry = MemoryEntry(
            id=correction_id,
            title=title,
            entry_type="correction",
            epistemic_status=epistemic_status,
            summary=summary,
            content=content,
            source_evidence=source_evidence,
            provenance=provenance,
            status="active",
            concepts=merged_concepts,
            tags=merged_tags,
            relationships=rels,
            created_at=provenance.recorded_at or utc_now_iso(),
            updated_at=provenance.recorded_at or utc_now_iso(),
        )
        self.create_entry(corr_entry, rel_path=rel_path)

        if mark_target_superseded:
            self.mark_superseded(
                entry_id=target_id,
                superseded_by_id=correction_id,
                reason=summary,
                actor=provenance.actor or "system",
            )

        return self.entries[correction_id], self.entries[target_id]

    # ------------------------------------------------------------------
    # 4.4 Hybrid Querying, Evolution Tracing & Bounded Context Packaging
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [
            w for w in re.findall(r"[a-z0-9_]{2,}", (text or "").lower())
            if w not in MEMORY_STOPWORDS
        ]

    def _score_seed_entries(
        self,
        query: str,
        concept_filter: Sequence[str] | None = None,
        entry_types: Sequence[str] | None = None,
    ) -> list[tuple[float, str, str]]:
        """Scores all entries in the memory tree using a multi-signal ranker:
        1. Direct stable ID mention in `query`
        2. Exact concept identifier or tag match (`concepts`, `tags`)
        3. Phrase / quote match against `title`, `summary`, `content`, and `source_evidence`
        4. BM25 lexical score with domain/concept coherence weighting (preventing mere surface-word
           overlap across unrelated domains from outranking true conceptual matches)."""
        q_norm = (query or "").strip()
        q_low = q_norm.lower()
        q_tokens = self._tokenize(q_norm)
        q_tok_set = set(q_tokens)
        req_concepts = {c.lower() for c in (concept_filter or [])}
        allowed_types = set(entry_types) if entry_types else None

        if not self.entries:
            return []

        # Precompute document frequencies for BM25
        doc_tokens: dict[str, collections.Counter[str]] = {}
        doc_lens: dict[str, int] = {}
        df: collections.Counter[str] = collections.Counter()
        for eid, e in self.entries.items():
            full_txt = f"{e.title} {' '.join(e.concepts)} {' '.join(e.tags)} {e.summary} {e.content} {e.source_evidence}"
            toks = self._tokenize(full_txt)
            cnt = collections.Counter(toks)
            doc_tokens[eid] = cnt
            doc_lens[eid] = max(1, len(toks))
            for w in cnt:
                df[w] += 1

        n_docs = len(self.entries)
        avg_dl = sum(doc_lens.values()) / float(max(1, n_docs))
        k1 = 1.5
        b = 0.75

        scored: list[tuple[float, str, str]] = []
        for eid, e in self.entries.items():
            if allowed_types and e.entry_type not in allowed_types:
                continue
            e_concepts_low = {c.lower() for c in e.concepts}
            if req_concepts and not (req_concepts & e_concepts_low):
                continue

            score = 0.0
            reasons: list[str] = []

            # Signal 1: Exact ID match
            if eid.lower() in q_low:
                score += 50.0
                reasons.append(f"direct_id({eid})")

            # Signal 2: Concept & tag match
            matched_concepts = [
                c for c in e.concepts
                if c.lower() in q_low or any( part in q_tok_set for part in re.findall(r"[a-z0-9]{3,}", c.lower()) if part not in MEMORY_STOPWORDS )
            ]
            exact_concept_hits = [
                c for c in e.concepts
                if c.lower().replace("-", " ").replace("_", " ") in q_low.replace("-", " ").replace("_", " ")
            ]
            if exact_concept_hits:
                score += 18.0 * len(exact_concept_hits)
                reasons.append(f"concept_exact({', '.join(exact_concept_hits[:2])})")
            elif matched_concepts:
                score += 6.0 * len(matched_concepts)
                reasons.append(f"concept_match({', '.join(matched_concepts[:2])})")

            matched_tags = [t for t in e.tags if t.lower() in q_tok_set or t.lower() in q_low]
            if matched_tags:
                score += 4.5 * len(matched_tags)
                reasons.append(f"tag_match({', '.join(matched_tags[:2])})")

            # Signal 3: Multi-word phrase / quote overlap against title, summary, or original source evidence
            for i in range(len(q_tokens) - 1):
                bigram = f"{q_tokens[i]} {q_tokens[i + 1]}"
                if len(bigram) >= 7:
                    if bigram in e.title.lower() or bigram in e.summary.lower():
                        score += 8.0
                        if "phrase_match" not in reasons:
                            reasons.append(f"phrase_match({bigram!r})")
                    elif bigram in e.source_evidence.lower() or bigram in e.content.lower():
                        score += 5.5
                        if "evidence_phrase" not in reasons:
                            reasons.append(f"evidence_phrase({bigram!r})")

            # Signal 4: BM25 lexical score over content words
            bm25 = 0.0
            cnt = doc_tokens[eid]
            dl = doc_lens[eid]
            matched_words: list[str] = []
            for w in q_tok_set:
                tf = cnt.get(w, 0)
                if tf <= 0:
                    continue
                matched_words.append(w)
                idf = math.log(1.0 + (n_docs - df[w] + 0.5) / (df[w] + 0.5))
                denom = tf + k1 * (1.0 - b + b * (dl / avg_dl))
                bm25 += idf * ((tf * (k1 + 1.0)) / denom)

            # Lexical-vs-Conceptual Disambiguation Guard:
            # If an entry only matches a single generic word (e.g., "inside") and shares ZERO concepts, tags,
            # or domain words with a multi-word query, dampen its raw single-word BM25 contribution so
            # unrelated technical/spatial entries with similar surface wording do not crowd out true conceptual matches.
            if len(matched_words) == 1 and len(q_tok_set) >= 2 and not matched_concepts and not matched_tags:
                bm25 *= 0.25

            if bm25 > 0:
                score += bm25 * 2.5
                if not reasons:
                    reasons.append(f"bm25_lexical({', '.join(sorted(matched_words)[:3])})")

            if score > 0.5:
                scored.append((score, eid, "+".join(reasons) or "lexical_match"))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored

    def retrieve_context_package(
        self,
        query: str,
        max_seeds: int = 4,
        max_depth: int = 2,
        max_entries: int = 8,
        token_budget: int = 1200,
        concept_filter: Sequence[str] | None = None,
        entry_types: Sequence[str] | None = None,
    ) -> MemoryContextPackage:
        """Executes hybrid retrieval + typed relationship traversal + correction/refinement chain resolution,
        returning a bounded, explainable `MemoryContextPackage` for Clover Kimi K3."""
        seed_candidates = self._score_seed_entries(query, concept_filter=concept_filter, entry_types=entry_types)
        total_considered = len(seed_candidates)
        top_seeds = seed_candidates[:max_seeds]
        seed_scores = {eid: sc for sc, eid, _ in top_seeds}
        seed_reasons = {eid: rsn for _, eid, rsn in top_seeds}
        seed_ids = [eid for _, eid, _ in top_seeds]

        if not seed_ids:
            return MemoryContextPackage(
                query=query,
                items=[],
                evolution_chains=[],
                active_contradictions=[],
                total_candidates_considered=0,
                traversal_depth_used=max_depth,
                token_budget=token_budget,
                estimated_tokens_used=0,
            )

        # 1. Traverse graph from seed_ids up to `max_depth`
        traversed_nodes = self.traverse(
            seed_ids=seed_ids,
            max_depth=max_depth,
            direction="both",
            max_nodes=max(max_entries * 3, 20),
        )

        # 2. For any matched or traversed entry that participates in a `corrects` / `refines` / `derived_from`
        #    chain, trace its full evolution chain so both the original interpretation and any superseding
        #    correction (plus their originating conversation evidence) are included!
        evolution_chains: list[list[EvolutionChainNode]] = []
        seen_chain_signatures: set[tuple[str, ...]] = set()
        chain_member_boost: dict[str, tuple[float, str]] = {}

        for tnode in traversed_nodes[: max_seeds + 4]:
            chain = self.trace_evolution_chain(tnode.entry.id)
            if len(chain) >= 2:
                sig = tuple(n.id for n in chain)
                if sig not in seen_chain_signatures:
                    seen_chain_signatures.add(sig)
                    evolution_chains.append(chain)
                base_sc = seed_scores.get(tnode.entry.id, 10.0)
                for cnode in chain:
                    if cnode.role_in_chain == "active_correction":
                        chain_member_boost[cnode.id] = (
                            max(chain_member_boost.get(cnode.id, (0.0, ""))[0], base_sc + 12.0),
                            f"superseding_correction_in_chain({tnode.entry.id})",
                        )
                    elif cnode.role_in_chain == "superseded_interpretation":
                        chain_member_boost[cnode.id] = (
                            max(chain_member_boost.get(cnode.id, (0.0, ""))[0], base_sc + 8.0),
                            f"original_interpretation_in_chain({tnode.entry.id})",
                        )
                    elif cnode.role_in_chain == "source_origin":
                        chain_member_boost[cnode.id] = (
                            max(chain_member_boost.get(cnode.id, (0.0, ""))[0], base_sc + 6.0),
                            f"originating_source_for({tnode.entry.id})",
                        )

        # Map traversed nodes by ID, adding any evolution chain nodes not yet in `traversed_by_id`
        traversed_by_id: dict[str, TraversalResultNode] = {tn.entry.id: tn for tn in traversed_nodes}
        for chain in evolution_chains:
            for cnode in chain:
                if cnode.id not in traversed_by_id and cnode.id in self.entries:
                    # Find shortest path from any seed if within depth+1
                    extra_trav = self.traverse(seed_ids=seed_ids, max_depth=max_depth + 2, max_nodes=48)
                    for etn in extra_trav:
                        if etn.entry.id == cnode.id:
                            traversed_by_id[cnode.id] = etn
                            break
                    if cnode.id not in traversed_by_id:
                        traversed_by_id[cnode.id] = TraversalResultNode(
                            entry=self.entries[cnode.id], depth=max_depth, path=[]
                        )

        # 3. Build ranked `RetrievedMemoryItem` candidates
        candidate_items: list[RetrievedMemoryItem] = []
        for eid, tnode in traversed_by_id.items():
            e = tnode.entry
            base_score = seed_scores.get(eid, max(1.0, 14.0 - 3.5 * tnode.depth))
            reason = seed_reasons.get(eid, f"graph_traversal(depth={tnode.depth})")
            if eid in chain_member_boost:
                b_sc, b_rsn = chain_member_boost[eid]
                if b_sc > base_score:
                    base_score = b_sc
                if b_rsn not in reason:
                    reason = f"{reason}+{b_rsn}"

            supersedes_ids = [r.target for r in self.outgoing.get(eid, []) if r.type == "corrects"]
            refines_ids = [r.target for r in self.outgoing.get(eid, []) if r.type == "refines"]
            contradictions: list[dict[str, str]] = []
            for r in self.outgoing.get(eid, []):
                if r.type in ("challenges", "unresolved_with"):
                    contradictions.append({"relation": r.type, "other_id": r.target, "note": r.note})
            for r in self.incoming.get(eid, []):
                if r.type in ("challenges", "unresolved_with"):
                    contradictions.append({"relation": r.type, "other_id": r.source, "note": r.note})

            if e.status == "superseded" or e.superseded_by:
                evo_role = "superseded_interpretation"
            elif any(r.type == "corrects" for r in self.outgoing.get(eid, [])) or e.entry_type == "correction":
                evo_role = "active_correction"
            elif any(r.type == "refines" for r in self.outgoing.get(eid, [])):
                evo_role = "refinement"
            elif contradictions:
                evo_role = "challenged_claim"
            elif e.entry_type == "conversation":
                evo_role = "supporting_evidence"
            else:
                evo_role = "active_primary"

            candidate_items.append(
                RetrievedMemoryItem(
                    entry=e,
                    relevance_score=base_score,
                    retrieval_reason=reason,
                    relationship_path=tnode.path_string,
                    evolution_role=evo_role,
                    supersedes_ids=supersedes_ids,
                    refines_ids=refines_ids,
                    superseded_by_id=e.superseded_by,
                    contradictions=contradictions,
                )
            )

        # Sort so active corrections and their paired superseded interpretations appear first, then by score
        role_rank = {
            "active_correction": 0,
            "superseded_interpretation": 1,
            "refinement": 2,
            "challenged_claim": 3,
            "active_primary": 4,
            "supporting_evidence": 5,
        }
        candidate_items.sort(key=lambda item: (-item.relevance_score, role_rank.get(item.evolution_role, 9), item.entry.id))

        # 4. Enforce `max_entries` and `token_budget` while ensuring that if a superseded entry is selected,
        #    its superseding correction is also selected (and vice versa)
        selected_items: list[RetrievedMemoryItem] = []
        selected_ids: set[str] = set()
        items_by_id = {it.entry.id: it for it in candidate_items}
        used_tokens = 0

        def _est_tokens(it: RetrievedMemoryItem) -> int:
            words = len(f"{it.entry.title} {it.entry.summary} {it.entry.content} {it.entry.source_evidence}".split())
            return max(24, int(words * 1.3))

        for item in candidate_items:
            if len(selected_items) >= max_entries:
                break
            if item.entry.id in selected_ids:
                continue
            t_cost = _est_tokens(item)
            if selected_items and (used_tokens + t_cost > token_budget):
                continue
            selected_items.append(item)
            selected_ids.add(item.entry.id)
            used_tokens += t_cost

            # Pair with superseding correction or superseded target if within budget
            paired_ids: list[str] = []
            if item.superseded_by_id and item.superseded_by_id in items_by_id:
                paired_ids.append(item.superseded_by_id)
            for sup_target in item.supersedes_ids:
                if sup_target in items_by_id:
                    paired_ids.append(sup_target)
            for pid in paired_ids:
                if pid not in selected_ids and len(selected_items) < max_entries:
                    p_item = items_by_id[pid]
                    p_cost = _est_tokens(p_item)
                    if used_tokens + p_cost <= token_budget or len(selected_items) < 2:
                        selected_items.append(p_item)
                        selected_ids.add(pid)
                        used_tokens += p_cost

        # Collect active contradictions among selected entries
        active_contradictions: list[dict[str, str]] = []
        seen_conf_pairs: set[tuple[str, str, str]] = set()
        for item in selected_items:
            eid = item.entry.id
            for r in self.outgoing.get(eid, []):
                if r.type in ("challenges", "unresolved_with"):
                    key = (r.source, r.type, r.target)
                    if key not in seen_conf_pairs:
                        seen_conf_pairs.add(key)
                        active_contradictions.append({
                            "source": r.source,
                            "type": r.type,
                            "target": r.target,
                            "note": r.note,
                        })
            for r in self.incoming.get(eid, []):
                if r.type in ("challenges", "unresolved_with"):
                    key = (r.source, r.type, r.target)
                    if key not in seen_conf_pairs:
                        seen_conf_pairs.add(key)
                        active_contradictions.append({
                            "source": r.source,
                            "type": r.type,
                            "target": r.target,
                            "note": r.note,
                        })

        return MemoryContextPackage(
            query=query,
            items=selected_items,
            evolution_chains=evolution_chains,
            active_contradictions=active_contradictions,
            total_candidates_considered=total_considered,
            traversal_depth_used=max_depth,
            token_budget=token_budget,
            estimated_tokens_used=used_tokens,
        )


# ---------------------------------------------------------------------------
# 5. Representative Multi-Branch Memory Tree Corpus Builder
# ---------------------------------------------------------------------------

def build_default_memory_entries() -> list[tuple[str, MemoryEntry]]:
    """Returns the canonical multi-branch memory corpus `(rel_path, MemoryEntry)` covering:
    1. Philosophical dialogue & evolving interpretations (the 'everything happens inside' -> questioning
       the inside/outside boundary evolution, plus competing realism tension and observer boundary concept).
    2. Systems & GPU engineering experience (global expert cache flat hit rate observation -> superseded
       'caching is a dead end' hypothesis -> per-layer partition correction -> Rust slotcache decision,
       experiment, and bit-identical outcome).
    3. Technical container measurement concept (lexical disambiguation control using 'inside' / 'outside'
       in a Linux namespace sense).
    """
    entries: list[tuple[str, MemoryEntry]] = [
        # --- Branch A: Conversations ---
        (
            "conversations/phil-inside-outside-dialogue-001.md",
            MemoryEntry(
                id="MEM-CONV-PHIL-INSIDE-001",
                title="Dialogue on Perception and the Meaning of 'Everything Happens Inside'",
                entry_type="conversation",
                epistemic_status="observed",
                status="active",
                concepts=["everything-happens-inside", "inside-outside-distinction", "observer-boundary"],
                tags=["philosophy", "dialogue", "perception", "epistemology"],
                created_at="2026-10-20T09:00:00Z",
                updated_at="2026-10-20T09:15:00Z",
                provenance=MemoryProvenance(
                    source_type="conversation",
                    source_ref="session:phil-inquiry-2026-10-20",
                    conversation_id="conv_phil_20261020_01",
                    turn_range="turns 1-6",
                    actor="human_and_assistant",
                    recorded_at="2026-10-20T09:15:00Z",
                ),
                summary=(
                    "Recorded philosophical dialogue examining the statement 'everything happens inside', "
                    "first interpreted as mental enclosure and subsequently clarified as questioning the "
                    "boundary between 'inside' and 'outside' itself."
                ),
                content=(
                    "In Turn 1, the human remarked: 'When you look closely at experience, everything happens inside.' "
                    "In Turn 2, an initial summary interpreted this as the claim that all perceived events occur "
                    "inside an individual's private mind. In Turn 5, the human corrected that reading, clarifying "
                    "that calling experience 'inside' only makes sense relative to an assumed 'outside', and that "
                    "the inquiry actually questions the fundamental distinction between inside and outside."
                ),
                source_evidence=(
                    "Turn 1 (Human): \"When we observe any sound, colour, or thought, everything happens inside.\"\n"
                    "Turn 2 (Assistant - initial reading): \"So you mean everything we experience happens inside an individual's mind.\"\n"
                    "Turn 5 (Human - clarification): \"No — saying 'inside an individual's mind' still assumes a neat box with an outside. "
                    "The point of saying 'everything happens inside' is to question the fundamental distinction between inside and outside altogether.\""
                ),
                relationships=[],
            ),
        ),

        # --- Branch B: Philosophy / Interpretations (Initial Superseded Interpretation) ---
        (
            "philosophy/interpretations/hyp-inside-individual-mind-001.md",
            MemoryEntry(
                id="MEM-HYP-INSIDE-MIND-001",
                title="Initial Interpretation: 'Everything Happens Inside' Means Inside an Individual's Mind",
                entry_type="hypothesis",
                epistemic_status="hypothesis",
                status="superseded",
                superseded_by="MEM-CORR-INSIDE-BOUNDARY-002",
                concepts=["everything-happens-inside", "inside-outside-distinction", "subjective-idealism"],
                tags=["philosophy", "interpretation", "mind", "inside"],
                created_at="2026-10-20T09:05:00Z",
                updated_at="2026-10-20T09:18:00Z",
                provenance=MemoryProvenance(
                    source_type="conversation",
                    source_ref="session:phil-inquiry-2026-10-20#turn-2",
                    conversation_id="conv_phil_20261020_01",
                    turn_range="turn 2",
                    actor="assistant",
                    recorded_at="2026-10-20T09:05:00Z",
                ),
                summary=(
                    "Initial interpretation that the phrase 'everything happens inside' means all experience "
                    "takes place inside an individual's personal mind (superseded by MEM-CORR-INSIDE-BOUNDARY-002)."
                ),
                content=(
                    "This initial interpretation treated 'everything happens inside' as a psychological or solipsistic "
                    "claim that phenomena are enclosed within an individual's internal mental theatre, separate from "
                    "an external physical world. It was later superseded when the speaker clarified in Turn 5 that "
                    "the inquiry questions the inside/outside boundary itself."
                ),
                source_evidence=(
                    "Turn 2 (Assistant): \"So you mean everything we experience happens inside an individual's mind, "
                    "as internal mental representations of an external world.\""
                ),
                relationships=[
                    MemoryRelationship(
                        type="derived_from",
                        target="MEM-CONV-PHIL-INSIDE-001",
                        provenance="session:phil-inquiry-2026-10-20#turn-2",
                        confidence=1.0,
                        context_scope="philosophical_interpretation_of_inside",
                        created_at="2026-10-20T09:05:00Z",
                        note="Initial assistant reading of Turn 1 in MEM-CONV-PHIL-INSIDE-001.",
                    ),
                ],
                revisions=[
                    MemoryRevision(
                        timestamp="2026-10-20T09:05:00Z",
                        actor="assistant",
                        action="created",
                        reason="Recorded initial interpretation of 'everything happens inside' from Turn 2.",
                    ),
                    MemoryRevision(
                        timestamp="2026-10-20T09:18:00Z",
                        actor="human",
                        action="mark_superseded",
                        reason=(
                            "Superseded by MEM-CORR-INSIDE-BOUNDARY-002: the discussion questions the fundamental "
                            "distinction between inside and outside rather than enclosing experience in a personal mind."
                        ),
                    ),
                ],
            ),
        ),

        # --- Branch C: Philosophy / Corrections (Active Superseding Correction) ---
        (
            "philosophy/corrections/corr-inside-outside-distinction-002.md",
            MemoryEntry(
                id="MEM-CORR-INSIDE-BOUNDARY-002",
                title="Correction: 'Everything Happens Inside' Questions the Fundamental Distinction Between Inside and Outside",
                entry_type="correction",
                epistemic_status="user_stated_belief",
                status="active",
                concepts=["everything-happens-inside", "inside-outside-distinction", "observer-boundary"],
                tags=["philosophy", "correction", "inside", "outside", "non-duality"],
                created_at="2026-10-20T09:18:00Z",
                updated_at="2026-10-20T09:18:00Z",
                provenance=MemoryProvenance(
                    source_type="conversation",
                    source_ref="session:phil-inquiry-2026-10-20#turn-5",
                    conversation_id="conv_phil_20261020_01",
                    turn_range="turn 5",
                    actor="human",
                    recorded_at="2026-10-20T09:18:00Z",
                ),
                summary=(
                    "Clarifies that 'everything happens inside' does not mean inside an individual's private mind; "
                    "rather, the discussion questions the fundamental distinction between inside and outside itself."
                ),
                content=(
                    "Corrects the earlier interpretation (MEM-HYP-INSIDE-MIND-001). Treating experience as locked "
                    "'inside an individual's mind' presupposes the very spatial/ontological divide between an inner box "
                    "and an outer world that the inquiry is examining. The intended philosophical point is to question "
                    "whether a fundamental boundary between 'inside' and 'outside' exists in direct experience. "
                    "Recorded with epistemic_status='user_stated_belief' (philosophical inquiry, not an empirical laboratory measurement)."
                ),
                source_evidence=(
                    "Turn 5 (Human): \"No — saying 'inside an individual's mind' still assumes a neat box with an outside. "
                    "The point of saying 'everything happens inside' is to question the fundamental distinction between inside and outside altogether.\""
                ),
                relationships=[
                    MemoryRelationship(
                        type="corrects",
                        target="MEM-HYP-INSIDE-MIND-001",
                        provenance="session:phil-inquiry-2026-10-20#turn-5",
                        confidence=1.0,
                        context_scope="philosophical_interpretation_of_inside",
                        created_at="2026-10-20T09:18:00Z",
                        note="Supersedes the individual-mind reading with the clarification questioning the inside/outside distinction.",
                    ),
                    MemoryRelationship(
                        type="derived_from",
                        target="MEM-CONV-PHIL-INSIDE-001",
                        provenance="session:phil-inquiry-2026-10-20#turn-5",
                        confidence=1.0,
                        context_scope="philosophical_interpretation_of_inside",
                        created_at="2026-10-20T09:18:00Z",
                        note="Directly grounded in Turn 5 of the recorded philosophical dialogue.",
                    ),
                    MemoryRelationship(
                        type="refines",
                        target="MEM-CONCEPT-OBSERVER-BOUNDARY-001",
                        provenance="session:phil-inquiry-2026-10-20#turn-5",
                        confidence=0.95,
                        context_scope="observer_boundary_inquiry",
                        created_at="2026-10-20T09:18:00Z",
                        note="Refines the observer-boundary concept by showing how 'inside' dissolves without a contrasting 'outside'.",
                    ),
                ],
            ),
        ),

        # --- Branch D: Concepts (Cross-Directory Concept Linked to Philosophy & Systems) ---
        (
            "concepts/observer-boundary-inquiry.md",
            MemoryEntry(
                id="MEM-CONCEPT-OBSERVER-BOUNDARY-001",
                title="The Boundary of the Observer (Inside vs. Outside in Inquiry)",
                entry_type="concept",
                epistemic_status="unresolved",
                status="contested",
                concepts=["observer-boundary", "inside-outside-distinction", "everything-happens-inside"],
                tags=["philosophy", "concept", "observer", "boundary", "agents-md-s22"],
                created_at="2026-10-20T09:25:00Z",
                updated_at="2026-10-20T09:30:00Z",
                provenance=MemoryProvenance(
                    source_type="document",
                    source_ref="AGENTS.md#section-22",
                    conversation_id="conv_phil_20261020_01",
                    actor="human_and_assistant",
                    recorded_at="2026-10-20T09:25:00Z",
                ),
                summary=(
                    "Open philosophical concept examining whether the boundary between the observer ('inside') and "
                    "the observed system ('outside') is fundamental or an artifact of perspective."
                ),
                content=(
                    "When the limits of an explanation become inseparable from the perspective of the one observing, "
                    "the observer itself becomes part of the inquiry. Rather than asserting a dogmatic metaphysical "
                    "conclusion, Clover keeps this boundary open as an unresolved philosophical inquiry (`epistemic_status: unresolved`)."
                ),
                source_evidence=(
                    "AGENTS.md Section 22: \"When the limits of an explanation become inseparable from the perspective "
                    "of the one observing, the observer itself may become part of the inquiry. Do not assume that the "
                    "observer stands outside the system being observed. Do not manufacture an answer at that boundary.\""
                ),
                relationships=[
                    MemoryRelationship(
                        type="exemplifies",
                        target="MEM-CORR-INSIDE-BOUNDARY-002",
                        provenance="AGENTS.md#section-22",
                        confidence=0.95,
                        context_scope="observer_boundary_inquiry",
                        created_at="2026-10-20T09:25:00Z",
                        note="Exemplifies questioning the assumed division between observer and observed.",
                    ),
                    MemoryRelationship(
                        type="unresolved_with",
                        target="MEM-HYP-STRICT-EXTERNAL-REALISM-001",
                        provenance="session:phil-inquiry-2026-10-20#turn-6",
                        confidence=1.0,
                        context_scope="ontology_of_inside_and_outside",
                        created_at="2026-10-20T09:30:00Z",
                        note="Remains in unresolved philosophical tension with strict internal/external dualism.",
                    ),
                    MemoryRelationship(
                        type="similar_to",
                        target="MEM-CONCEPT-OBSERVER-EFFECT-MEASUREMENT-001",
                        provenance="session:phil-inquiry-2026-10-20",
                        confidence=0.6,
                        context_scope="structural_analogy_only",
                        created_at="2026-10-20T09:30:00Z",
                        note="Analogical similarity only: both involve where an observation boundary is drawn, though in different domains.",
                    ),
                ],
                uncertainty_notes=[
                    "Whether the distinction between observer and observed can be dissolved completely remains an open philosophical question, not an empirically settled fact."
                ],
            ),
        ),

        # --- Branch E: Philosophy / Competing Position (Contradictory / Unresolved Claim & Valid Cycle) ---
        (
            "philosophy/positions/hyp-strict-external-realism-001.md",
            MemoryEntry(
                id="MEM-HYP-STRICT-EXTERNAL-REALISM-001",
                title="Competing Position: Strict Internal-Mind vs. External-World Dualism",
                entry_type="hypothesis",
                epistemic_status="user_stated_belief",
                status="contested",
                concepts=["inside-outside-distinction", "cartesian-dualism", "observer-boundary"],
                tags=["philosophy", "dualism", "realism", "contested"],
                created_at="2026-10-20T09:28:00Z",
                updated_at="2026-10-20T09:30:00Z",
                provenance=MemoryProvenance(
                    source_type="conversation",
                    source_ref="session:phil-inquiry-2026-10-20#turn-6",
                    conversation_id="conv_phil_20261020_01",
                    turn_range="turn 6",
                    actor="human",
                    recorded_at="2026-10-20T09:28:00Z",
                ),
                summary=(
                    "Competing philosophical thesis holding that 'inside' (subjective mental states) and 'outside' "
                    "(mind-independent physical objects) are fundamentally distinct ontological domains."
                ),
                content=(
                    "Articulates the classical dualist/realist counter-position discussed in Turn 6: that internal "
                    "representations and external physical reality are separated by a real ontological boundary. "
                    "Preserved as a contested philosophical position (`challenges` and `unresolved_with` "
                    "MEM-CONCEPT-OBSERVER-BOUNDARY-001) without converting the disagreement into false consensus."
                ),
                source_evidence=(
                    "Turn 6 (Human): \"A classical realist would object that inside and outside are strictly separate: "
                    "neurons and representations are inside the skull, while trees and stars are outside in physical space.\""
                ),
                relationships=[
                    MemoryRelationship(
                        type="challenges",
                        target="MEM-CONCEPT-OBSERVER-BOUNDARY-001",
                        provenance="session:phil-inquiry-2026-10-20#turn-6",
                        confidence=1.0,
                        context_scope="ontology_of_inside_and_outside",
                        created_at="2026-10-20T09:28:00Z",
                        note="Challenges the dissolution of the inside/outside boundary from a classical dualist perspective.",
                    ),
                    MemoryRelationship(
                        type="unresolved_with",
                        target="MEM-CONCEPT-OBSERVER-BOUNDARY-001",
                        provenance="session:phil-inquiry-2026-10-20#turn-6",
                        confidence=1.0,
                        context_scope="ontology_of_inside_and_outside",
                        created_at="2026-10-20T09:30:00Z",
                        note="Mutual philosophical tension retained intentionally (valid 2-cycle in the knowledge graph).",
                    ),
                    MemoryRelationship(
                        type="supports",
                        target="MEM-HYP-INSIDE-MIND-001",
                        provenance="session:phil-inquiry-2026-10-20#turn-6",
                        confidence=0.8,
                        context_scope="ontology_of_inside_and_outside",
                        created_at="2026-10-20T09:28:00Z",
                        note="Provides the philosophical premise behind the earlier internal-mind interpretation.",
                    ),
                ],
            ),
        ),

        # --- Branch F: Technical Concept Using 'Inside' / 'Outside' (Lexical vs. Semantic Disambiguation) ---
        (
            "concepts/container-namespace-observation-001.md",
            MemoryEntry(
                id="MEM-CONCEPT-OBSERVER-EFFECT-MEASUREMENT-001",
                title="In-Container vs. Host-Level GPU Telemetry Namespace Boundary",
                entry_type="concept",
                epistemic_status="externally_verified",
                status="active",
                concepts=["container-namespace-telemetry", "gpu-profiling", "pid-namespace"],
                tags=["systems", "linux", "gpu", "telemetry", "container"],
                created_at="2026-10-21T14:00:00Z",
                updated_at="2026-10-21T14:00:00Z",
                provenance=MemoryProvenance(
                    source_type="observation",
                    source_ref="docs/clover-kimi-k3-gpu-context.md",
                    actor="system_measurement",
                    recorded_at="2026-10-21T14:00:00Z",
                ),
                summary=(
                    "Technical systems fact: tools running inside an isolated container PID namespace only see "
                    "processes inside that container, whereas `nvidia-smi` outside on the host sees all GPU processes."
                ),
                content=(
                    "In Linux container deployments, everything executed inside a container's PID namespace sees "
                    "only the container's process table, while GPU VRAM allocations happen on the shared kernel driver "
                    "outside the container namespace."
                ),
                source_evidence=(
                    "Measured on Ubuntu 24.04 (`nvidia-smi` host vs. container namespace): PID visibility differs "
                    "between inside-container procfs and host-level NVML."
                ),
                relationships=[
                    MemoryRelationship(
                        type="similar_to",
                        target="MEM-CONCEPT-OBSERVER-BOUNDARY-001",
                        provenance="docs/clover-kimi-k3-gpu-context.md",
                        confidence=0.5,
                        context_scope="terminology_analogy_only",
                        created_at="2026-10-21T14:00:00Z",
                        note="Shares the words 'inside' and 'outside' as a boundary metaphor, but refers to Linux PID namespaces.",
                    ),
                ],
            ),
        ),

        # --- Branch G: Systems Engineering Experience (Observation -> Superseded Hypothesis -> Correction -> Decision -> Experiment -> Outcome) ---
        (
            "observations/obs-global-cache-flat-hitrate-001.md",
            MemoryEntry(
                id="MEM-OBS-GLOBAL-CACHE-FLAT-001",
                title="Observation: Global Expert Cache Hit Rate Flat at 23.3% Across Capacities 512..4096",
                entry_type="observation",
                epistemic_status="externally_verified",
                status="active",
                concepts=["expert-slot-cache", "cyclic-sweep-eviction", "kimi-k3-moe"],
                tags=["systems", "cache", "measurement", "k3-moe"],
                created_at="2026-10-09T11:00:00Z",
                updated_at="2026-10-09T11:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="clover-kimi-k3/docs/clover-kimi-k3-cpu-context.md",
                    actor="system_measurement",
                    recorded_at="2026-10-09T11:00:00Z",
                ),
                summary=(
                    "Measured a flat 23.3% expert cache hit rate across every global capacity setting from 512 "
                    "to 4096 entries during 92-layer Kimi K3 MoE forward passes."
                ),
                content=(
                    "Because 92 MoE layers execute sequentially for each token and request 16 routed experts per layer "
                    "(1,472 expert lookups per token step), a single global LRU cache smaller than the 92-layer working "
                    "set suffers cyclic-sweep eviction before Layer 1 is revisited on the next token."
                ),
                source_evidence=(
                    "Benchmark log (2026-10-09): Global expert cache capacity 512 -> 23.3% hits; 1024 -> 23.3% hits; "
                    "2048 -> 23.3% hits; 4096 -> 23.3% hits."
                ),
                relationships=[],
            ),
        ),
        (
            "hypotheses/hyp-caching-dead-end-001.md",
            MemoryEntry(
                id="MEM-HYP-CACHE-DEAD-END-001",
                title="Initial Hypothesis: Expert Caching Has No Effect on Kimi K3 Routing",
                entry_type="hypothesis",
                epistemic_status="hypothesis",
                status="superseded",
                superseded_by="MEM-CORR-PER-LAYER-PARTITION-002",
                concepts=["expert-slot-cache", "cyclic-sweep-eviction", "kimi-k3-moe"],
                tags=["systems", "cache", "hypothesis", "superseded"],
                created_at="2026-10-09T11:20:00Z",
                updated_at="2026-10-09T15:00:00Z",
                provenance=MemoryProvenance(
                    source_type="conversation",
                    source_ref="clover-operating-principles.md#2026-10-09",
                    actor="assistant",
                    recorded_at="2026-10-09T11:20:00Z",
                ),
                summary=(
                    "Mistaken initial hypothesis that expert caching is ineffective for Kimi K3 because varying "
                    "cache capacity from 512 to 4096 produced a flat 23.3% hit rate (superseded by MEM-CORR-PER-LAYER-PARTITION-002)."
                ),
                content=(
                    "Concluded from the flat 23.3% curve across capacities 512..4096 that expert reuse across tokens "
                    "was absent and caching should be rejected."
                ),
                source_evidence=(
                    "Initial report (2026-10-09): \"Hit rate is 23.3% flat from 512 to 4096 entries due to cyclic sweep; "
                    "expert caching does not help this workload.\""
                ),
                relationships=[
                    MemoryRelationship(
                        type="derived_from",
                        target="MEM-OBS-GLOBAL-CACHE-FLAT-001",
                        provenance="clover-operating-principles.md#2026-10-09",
                        confidence=1.0,
                        context_scope="expert_cache_design",
                        created_at="2026-10-09T11:20:00Z",
                        note="Inferred from the flat global capacity sweep.",
                    ),
                    MemoryRelationship(
                        type="caused_by",
                        target="MEM-OBS-GLOBAL-CACHE-FLAT-001",
                        provenance="clover-operating-principles.md#2026-10-09",
                        confidence=1.0,
                        context_scope="expert_cache_design",
                        created_at="2026-10-09T11:20:00Z",
                        note="Triggered by misinterpreting a flat parameter response as absence of the underlying effect.",
                    ),
                ],
                revisions=[
                    MemoryRevision(
                        timestamp="2026-10-09T11:20:00Z",
                        actor="assistant",
                        action="created",
                        reason="Recorded initial conclusion from global cache capacity sweep.",
                    ),
                    MemoryRevision(
                        timestamp="2026-10-09T15:00:00Z",
                        actor="human_and_assistant",
                        action="mark_superseded",
                        reason="Superseded by MEM-CORR-PER-LAYER-PARTITION-002 after partitioning the cache per layer.",
                    ),
                ],
            ),
        ),
        (
            "corrections/corr-per-layer-cache-partition-002.md",
            MemoryEntry(
                id="MEM-CORR-PER-LAYER-PARTITION-002",
                title="Correction: Partitioning the Expert Cache Per Layer Raises Hit Rate from 23.3% to 48.0%",
                entry_type="correction",
                epistemic_status="externally_verified",
                status="active",
                concepts=["expert-slot-cache", "per-layer-partitioning", "cyclic-sweep-eviction", "kimi-k3-moe"],
                tags=["systems", "cache", "correction", "verified"],
                created_at="2026-10-09T15:00:00Z",
                updated_at="2026-10-09T15:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="clover-operating-principles.md#2026-10-09",
                    actor="human_and_assistant",
                    recorded_at="2026-10-09T15:00:00Z",
                ),
                summary=(
                    "When a measurement shows a flat response across all settings of a parameter, suspect the "
                    "structure being measured (global eviction across layers), not the effect: partitioning the "
                    "same 51.7 GB cache per layer increased the hit rate from 23.3% to 48.0%."
                ),
                content=(
                    "Corrects MEM-HYP-CACHE-DEAD-END-001. Because each layer's experts only compete with experts "
                    "of the same layer when partitioned per layer (`layer_id in 1..92`), cross-layer cyclic eviction "
                    "is eliminated and the same 51.7 GB budget achieves 48.0% hit rate."
                ),
                source_evidence=(
                    "Measured on Kimi K3 92-layer MoE (51.7 GB budget): global pool = 23.3% hit rate (flat); "
                    "per-layer partitioned pool = 48.0% hit rate (monotonic scaling with per-layer slots)."
                ),
                relationships=[
                    MemoryRelationship(
                        type="corrects",
                        target="MEM-HYP-CACHE-DEAD-END-001",
                        provenance="clover-operating-principles.md#2026-10-09",
                        confidence=1.0,
                        context_scope="expert_cache_design",
                        created_at="2026-10-09T15:00:00Z",
                        note="Refutes the 'caching has no effect' hypothesis by eliminating cross-layer cyclic eviction.",
                    ),
                    MemoryRelationship(
                        type="refines",
                        target="MEM-OBS-GLOBAL-CACHE-FLAT-001",
                        provenance="clover-operating-principles.md#2026-10-09",
                        confidence=1.0,
                        context_scope="expert_cache_design",
                        created_at="2026-10-09T15:00:00Z",
                        note="Explains why the global sweep was flat and how per-layer partitioning resolves it.",
                    ),
                ],
            ),
        ),
        (
            "decisions/dec-slotcache-per-layer-rust-001.md",
            MemoryEntry(
                id="MEM-DEC-SLOTCACHE-PER-LAYER-001",
                title="Decision: Implement Per-Layer GPU Expert SlotCache in Rust (`libslotcache.so`)",
                entry_type="decision",
                epistemic_status="observed",
                status="active",
                concepts=["expert-slot-cache", "per-layer-partitioning", "rust-host-path"],
                tags=["systems", "rust", "slotcache", "decision"],
                created_at="2026-10-24T10:00:00Z",
                updated_at="2026-10-25T18:00:00Z",
                provenance=MemoryProvenance(
                    source_type="document",
                    source_ref="clover-kimi-k3/src/rust/slotcache/CONTRACT.md",
                    actor="human_and_assistant",
                    recorded_at="2026-10-24T10:00:00Z",
                ),
                summary=(
                    "Architectural decision to implement the per-layer GPU expert slot tracker and LRU eviction "
                    "table as a selectable Rust C-ABI library (`libslotcache.so`, `K3GPU_SLOTCACHE=rust`)."
                ),
                content=(
                    "Maintains per-(card, layer) direct-mapped lookup tables (`eid_to_slot[896]`), resident slot metadata, "
                    "and tick-based LRU victim selection across all 92 MoE layers (`k3_slotcache_*` C ABI)."
                ),
                source_evidence=(
                    "Commit 5f11928 (`clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_gpu_experts_lib.cu`): wired "
                    "`libslotcache.so`, `libpoolinit.so`, and `libsupply.so` into `libk3_gpu_experts.so`."
                ),
                relationships=[
                    MemoryRelationship(
                        type="depends_on",
                        target="MEM-CORR-PER-LAYER-PARTITION-002",
                        provenance="clover-kimi-k3/src/rust/slotcache/CONTRACT.md",
                        confidence=1.0,
                        context_scope="expert_cache_design",
                        created_at="2026-10-24T10:00:00Z",
                        note="Per-layer slot partitioning directly implements the finding from MEM-CORR-PER-LAYER-PARTITION-002.",
                    ),
                ],
            ),
        ),
        (
            "experiments/exp-rust-slotcache-parity-001.md",
            MemoryEntry(
                id="MEM-EXP-RUST-SLOTCACHE-001",
                title="Experiment: Differential Verification of Rust Host-Path Libraries Across 93 GPU Layers",
                entry_type="experiment",
                epistemic_status="externally_verified",
                status="active",
                concepts=["rust-host-path", "expert-slot-cache", "kimi-k3-moe"],
                tags=["systems", "rust", "experiment", "verification"],
                created_at="2026-10-25T19:00:00Z",
                updated_at="2026-10-25T19:30:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="commit:178d3c1 (wire_test_merged + K3Full93Chain)",
                    actor="system_measurement",
                    recorded_at="2026-10-25T19:30:00Z",
                ),
                summary=(
                    "Executed 212 Rust unit tests, `wire_test_merged`, `whole-model-gpu`, and `K3Full93Chain` "
                    "comparing `K3GPU_*=cpp` vs `K3GPU_*=rust` on 8 x RTX 3090 GPUs."
                ),
                content=(
                    "Tested all 8 Rust host-path crates (`copyq`, `slotcache`, `routetable`, `planner`, `poolinit`, "
                    "`supply`, `loader`, `engcfg`) against the C++ reference implementation."
                ),
                source_evidence=(
                    "`K3Full93Chain` (`cpp` vs `rust` on 8x RTX 3090): `max_abs_diff = 0.0`, `mean_abs_diff = 0.0`, "
                    "`2,576` resident experts + `93` trunks across `cuda:0..5`."
                ),
                relationships=[
                    MemoryRelationship(
                        type="depends_on",
                        target="MEM-DEC-SLOTCACHE-PER-LAYER-001",
                        provenance="commit:5f11928",
                        confidence=1.0,
                        context_scope="rust_host_path_verification",
                        created_at="2026-10-25T19:00:00Z",
                        note="Tests the Rust slotcache, poolinit, and supply integration.",
                    ),
                    MemoryRelationship(
                        type="supports",
                        target="MEM-CORR-PER-LAYER-PARTITION-002",
                        provenance="commit:178d3c1",
                        confidence=1.0,
                        context_scope="expert_cache_design",
                        created_at="2026-10-25T19:30:00Z",
                        note="Confirms per-layer slotcache operation across all 92 MoE layers on GPU.",
                    ),
                ],
            ),
        ),
        (
            "outcomes/out-rust-gpu-merged-parity-001.md",
            MemoryEntry(
                id="MEM-OUT-RUST-GPU-PARITY-001",
                title="Outcome: Bit-Identical 93-Layer GPU Execution (`max_abs_diff = 0.0`) and 100% 5-Benchmark Suite",
                entry_type="outcome",
                epistemic_status="externally_verified",
                status="active",
                concepts=["rust-host-path", "kimi-k3-moe", "generative-compilation"],
                tags=["systems", "outcome", "verified", "benchmarks"],
                created_at="2026-10-25T20:00:00Z",
                updated_at="2026-10-26T02:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="commit:178d3c1 & commit:f308f12",
                    actor="system_measurement",
                    recorded_at="2026-10-26T02:00:00Z",
                ),
                summary=(
                    "Verified bit-identical 93-layer Kimi K3 GPU execution (`max_abs_diff = 0.0`) with Rust host-path "
                    "libraries active, achieving 20/20 GSM8K, 20/20 MMLU, 40/40 Extended HumanEval, 15/15 HybridAgent QA, "
                    "and 47/47 on the Pooled Multi-Tier Benchmark."
                ),
                content=(
                    "Demonstrates that replacing the C++ host-path cache, pool initializer, and supply logic with "
                    "validated Rust crates preserves exact numerical parity while Generative Compilation eliminates "
                    "mid-generation syntax, scope, and doctest errors."
                ),
                source_evidence=(
                    "Commits `178d3c1` and `f308f12`: `max_abs_diff = 0.0` (`cpp` vs `rust`); GSM8K `20/20 (100%)`, "
                    "MMLU `20/20 (100%)`, HumanEval `40/40 (100%)`, Rust GC `5/5 (100%)`, Pooled `47/47 (100%)`."
                ),
                relationships=[
                    MemoryRelationship(
                        type="derived_from",
                        target="MEM-EXP-RUST-SLOTCACHE-001",
                        provenance="commit:178d3c1",
                        confidence=1.0,
                        context_scope="rust_host_path_verification",
                        created_at="2026-10-25T20:00:00Z",
                        note="Outcome measured directly from the differential verification experiment.",
                    ),
                    MemoryRelationship(
                        type="supports",
                        target="MEM-DEC-SLOTCACHE-PER-LAYER-001",
                        provenance="commit:178d3c1",
                        confidence=1.0,
                        context_scope="rust_host_path_verification",
                        created_at="2026-10-25T20:00:00Z",
                        note="Validates the architectural decision in production service (`clover-k3-chat.service`).",
                    ),
                ],
            ),
        ),
    ]
    return entries


DEFAULT_MEMORY_TREE_DIR = Path(__file__).resolve().parent / "memory_tree"


def ensure_default_memory_tree(root_dir: str | os.PathLike[str] | None = None) -> CloverMemoryGraphStore:
    """Ensures the canonical memory tree at `root_dir` (defaulting to `memory_tree/` beside `k3_memory_graph.py`)
    contains the default multi-branch memory entries without overwriting any existing user-modified files,
    and returns an initialized `CloverMemoryGraphStore`."""
    target_root = Path(root_dir) if root_dir is not None else DEFAULT_MEMORY_TREE_DIR
    target_root.mkdir(parents=True, exist_ok=True)

    # Write any missing canonical seed files first, then build index once
    written_any = False
    for rel_p, entry in build_default_memory_entries():
        fp = target_root / rel_p
        if not fp.exists():
            entry.rel_path = rel_p
            if not entry.revisions:
                entry.revisions.append(
                    MemoryRevision(
                        timestamp=entry.created_at,
                        actor=entry.provenance.actor or "system",
                        action="created",
                        reason=f"Initial {entry.entry_type} entry recorded from {entry.provenance.source_ref}",
                    )
                )
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(render_memory_markdown(entry), encoding="utf-8")
            written_any = True

    store = CloverMemoryGraphStore(target_root, auto_load=True)
    if written_any or not store.index_file.exists():
        store.rebuild_index(write_index_file=True)
    return store


def retrieve_relevant_memory_package(
    store: CloverMemoryGraphStore,
    query: str,
    max_seeds: int = 3,
    max_depth: int = 2,
    max_entries: int = 6,
    token_budget: int = 1100,
) -> MemoryContextPackage | None:
    """High-precision gate for the Clover Kimi K3 context pipeline:
    Retrieves a bounded `MemoryContextPackage` only when `query` genuinely references a stored memory ID,
    concept, phrase, or prior interpretation/correction, avoiding false-positive context injection on
    unrelated coding, math, or multiple-choice benchmarks."""
    q = (query or "").strip()
    q_low = q.lower()
    if not q or len(q) < 4:
        return None
    if (
        ("```" in q_low)
        or bool(re.search(r"\b(?:def|fn)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(", q))
        or ("(a)" in q_low and "(b)" in q_low and "(c)" in q_low)
        or ("rewrite these notes into one clear" in q_low)
    ):
        return None

    seed_candidates = store._score_seed_entries(q)
    if not seed_candidates:
        return None

    top_score, _, top_reason = seed_candidates[0]
    has_direct_or_exact_concept = ("direct_id(" in top_reason) or ("concept_exact(" in top_reason)
    has_phrase_hit = ("phrase_match(" in top_reason) or ("evidence_phrase(" in top_reason)
    has_memory_cue = any(
        cue in q_low
        for cue in (
            "memory", "remember", "earlier interpretation", "initial interpretation", "corrected",
            "correction", "superseded", "everything happens inside", "inside and outside",
            "observer boundary", "cyclic sweep", "slotcache", "per-layer partition",
        )
    )

    if not (
        has_direct_or_exact_concept
        or (has_phrase_hit and top_score >= 16.0)
        or (has_memory_cue and top_score >= 12.0)
    ):
        return None

    pkg = store.retrieve_context_package(
        query=q,
        max_seeds=max_seeds,
        max_depth=max_depth,
        max_entries=max_entries,
        token_budget=token_budget,
    )
    return pkg if pkg.items else None


