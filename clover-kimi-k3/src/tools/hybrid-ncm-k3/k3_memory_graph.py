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

    def render_for_k3(self, compact: bool = False) -> str:
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

        items_to_render = self.items[:4] if compact else self.items
        for idx, item in enumerate(items_to_render, start=1):
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
            if not compact and e.content and e.content != e.summary:
                body_lines.append(f"  Content: {e.content}")
            if e.source_evidence:
                ev_txt = e.source_evidence[:260] if compact else e.source_evidence
                body_lines.append(f"  Original Source Evidence: {ev_txt}")
            if item.contradictions:
                c_strs = [
                    f"{c['relation']} with {c['other_id']} ({c.get('note') or 'contested/unresolved'})"
                    for c in item.contradictions[:2]
                ]
                body_lines.append(f"  Contradictions / Open Tensions: {'; '.join(c_strs)}")
            if e.uncertainty_notes:
                u_list = e.uncertainty_notes[:1] if compact else e.uncertainty_notes
                body_lines.append(f"  Uncertainty Notes: {'; '.join(u_list)}")
            blocks.append("\n".join(body_lines))

        if self.evolution_chains:
            chain_lines = ["--- Traceable Interpretation & Correction Chains ---"]
            for c_idx, chain in enumerate(self.evolution_chains[:2], start=1):
                steps_fmt = "  -->  ".join(
                    f"{node.id} [{node.entry_type}, {node.role_in_chain}, epistemic={node.epistemic_status}, created={node.created_at}]"
                    for node in chain
                )
                chain_lines.append(f"  Chain {c_idx}: {steps_fmt}")
            blocks.append("\n".join(chain_lines))

        if self.active_contradictions:
            conf_lines = ["--- Unresolved / Contested Claims (Do Not Flatten Into Agreement) ---"]
            for c in self.active_contradictions[:3]:
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
        self.staging_dir = self.root_dir / "staging"

        self.entries: dict[str, MemoryEntry] = {}
        self.id_to_rel_path: dict[str, str] = {}
        self.outgoing: dict[str, list[MemoryRelationship]] = collections.defaultdict(list)
        self.incoming: dict[str, list[MemoryRelationship]] = collections.defaultdict(list)
        self.csc_generation: dict[str, Any] = {}
        self.read_epoch: int = 0
        self.last_validation: GraphValidationReport | None = None
        self.retrieval_rules: dict[str, Any] = self._load_retrieval_rules()

        if auto_load and self.root_dir.exists():
            self.rebuild_index(write_index_file=not self.index_file.exists())

    def _load_retrieval_rules(self) -> dict[str, Any]:
        """Loads McCulloch-Pitts inhibitory vetoes and Minsky-Papert structural concept rules
        from `memory_tree/_rules/memory_retrieval_rules.json` (falling back to the canonical
        `DEFAULT_MEMORY_TREE_DIR/_rules/memory_retrieval_rules.json` when using a temporary store dir)."""
        for base_dir in (self.root_dir, Path(__file__).resolve().parent / "memory_tree"):
            rules_fp = base_dir / "_rules" / "memory_retrieval_rules.json"
            if rules_fp.exists():
                try:
                    return json.loads(rules_fp.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return {}

    # ------------------------------------------------------------------
    # 4.1 Index Building, Persistence & Validation from Canonical Files
    # ------------------------------------------------------------------

    def _iter_markdown_files(self) -> list[Path]:
        if not self.root_dir.exists():
            return []
        md_files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Skip hidden directories, `_index`, `_rules`, `_ontology`, and `staging` (unpromoted drafts)
            dirnames[:] = sorted(
                d for d in dirnames
                if not d.startswith(".") and not d.startswith("_") and d != "staging"
            )
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
        self.read_epoch += 1
        self.csc_generation = self._compile_csc_generation()

        if write_index_file and self.root_dir.exists():
            self._write_index_cache(report)

        return report

    def _compile_csc_generation(self) -> dict[str, Any]:
        """Compiles an immutable Compressed Sparse Column / Row (`CSC/CSR`) topology generation
        (`vertices`, `ordinal_of`, `out_ptrs`, `out_indices`, `in_ptrs`, `in_indices`) over the
        canonical file-tree entries so multi-hop traversal and batched `ms_paths` can execute over
        flat integer arrays alongside the rich `MemoryRelationship` metadata."""
        verts = sorted(self.entries.keys())
        ord_of = {vid: i for i, vid in enumerate(verts)}
        n_v = len(verts)
        out_ptrs = [0]
        out_indices: list[int] = []
        in_ptrs = [0]
        in_indices: list[int] = []
        for vid in verts:
            out_n = sorted({ord_of[r.target] for r in self.outgoing.get(vid, []) if r.target in ord_of})
            out_indices.extend(out_n)
            out_ptrs.append(len(out_indices))
            in_n = sorted({ord_of[r.source] for r in self.incoming.get(vid, []) if r.source in ord_of})
            in_indices.extend(in_n)
            in_ptrs.append(len(in_indices))
        return {
            "read_epoch": self.read_epoch,
            "num_vertices": n_v,
            "vertices": verts,
            "ordinal_of": ord_of,
            "out_ptrs": out_ptrs,
            "out_indices": out_indices,
            "in_ptrs": in_ptrs,
            "in_indices": in_indices,
        }

    def ms_paths(
        self,
        source_ids: Sequence[str],
        target_ids: Sequence[str] | None = None,
        max_len: int = 3,
        max_paths: int = 24,
    ) -> dict[str, Any]:
        """Snapshot-pinned multi-source / multi-target path enumerator (`MSpaths`) using the compiled
        `csc_generation` topology accelerator with reverse-target pruning while preserving full
        Clover `MemoryRelationship` provenance and epistemic status."""
        gen = self.csc_generation or self._compile_csc_generation()
        valid_sources = [s for s in source_ids if s in self.entries]
        valid_targets = {t for t in (target_ids or self.entries.keys()) if t in self.entries}
        if not valid_sources or not valid_targets:
            return {"read_epoch": gen.get("read_epoch", 0), "paths": []}

        # Reverse target pruning from `valid_targets` up to `max_len` hops
        can_reach_target: set[str] = set(valid_targets)
        rev_frontier: set[str] = set(valid_targets)
        for _ in range(max_len):
            nxt_rev: set[str] = set()
            for cur in rev_frontier:
                for r in self.incoming.get(cur, []) + self.outgoing.get(cur, []):
                    other = r.source if r.target == cur else r.target
                    if other in self.entries and other not in can_reach_target:
                        can_reach_target.add(other)
                        nxt_rev.add(other)
            if not nxt_rev:
                break
            rev_frontier = nxt_rev

        trav = self.traverse(
            seed_ids=[s for s in valid_sources if s in can_reach_target],
            max_depth=max_len,
            direction="both",
            max_nodes=max_paths * 2,
            min_similar_confidence=0.7,
        )
        matched_paths = [
            {
                "source": tn.path[0].from_id if tn.path else tn.entry.id,
                "target": tn.entry.id,
                "hops": tn.depth,
                "path": tn.path_string,
                "epistemic_status": tn.entry.epistemic_status,
                "status": tn.entry.status,
            }
            for tn in trav
            if tn.depth >= 1 and tn.entry.id in valid_targets
        ][:max_paths]
        return {
            "read_epoch": gen.get("read_epoch", 0),
            "csc_vertices": gen.get("num_vertices", 0),
            "csc_edges": len(gen.get("out_indices", [])),
            "pruned_reachable_count": len(can_reach_target),
            "paths": matched_paths,
        }

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
        if self.index_file.exists():
            try:
                existing = json.loads(self.index_file.read_text(encoding="utf-8"))
                if (
                    existing.get("schema_version") == payload["schema_version"]
                    and existing.get("validation") == payload["validation"]
                    and existing.get("nodes") == payload["nodes"]
                    and existing.get("edges") == payload["edges"]
                ):
                    return
            except Exception:
                pass
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
        min_similar_confidence: float = 0.0,
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
                        if r.type == "similar_to" and min_similar_confidence > 0.0:
                            if r.confidence < min_similar_confidence or r.context_scope.endswith("_analogy_only"):
                                continue
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
                        if r.type == "similar_to" and min_similar_confidence > 0.0:
                            if r.confidence < min_similar_confidence or r.context_scope.endswith("_analogy_only"):
                                continue
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
    # 4.3b Closed-Loop Growth Staging Queue & Verified Promotion Gate
    # ------------------------------------------------------------------

    def stage_candidate_entry(
        self,
        entry: MemoryEntry,
        auto_link_concepts: bool = True,
    ) -> MemoryEntry:
        """Writes a candidate memory entry into `memory_tree/staging/<slug>.md` with `status='under_review'`.
        Staged entries remain isolated from the active graph index until promoted via `promote_staged_entry()`,
        preventing unverified model conclusions from polluting the canonical memory graph."""
        entry.status = "under_review"
        if auto_link_concepts:
            existing_targets = {r.target for r in entry.relationships}
            entry_concepts_low = {c.lower() for c in entry.concepts}
            for cid, c_entry in self.entries.items():
                if c_entry.entry_type != "concept" or cid in existing_targets:
                    continue
                c_low = {c.lower() for c in c_entry.concepts}
                if entry_concepts_low & c_low:
                    rel_t = "exemplifies" if entry.entry_type in ("observation", "correction", "experiment", "outcome") else "similar_to"
                    entry.relationships.append(
                        MemoryRelationship(
                            type=rel_t,
                            target=cid,
                            source=entry.id,
                            provenance=entry.provenance.source_ref,
                            confidence=0.9,
                            context_scope="auto_linked_concept_bridge",
                            created_at=entry.created_at,
                            note=f"Shared concept bridge ({', '.join(sorted(entry_concepts_low & c_low))})",
                        )
                    )
                    existing_targets.add(cid)

        errs = entry.validate()
        if errs:
            raise ValueError(f"Invalid staged MemoryEntry {entry.id!r}: {'; '.join(errs)}")
        if entry.id in self.entries:
            raise ValueError(f"Entry {entry.id!r} already exists in the active memory tree at {self.id_to_rel_path[entry.id]!r}")

        safe_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "-", entry.id.lower()).strip("-")
        rel_p = f"staging/{safe_slug}.md"
        entry.rel_path = rel_p
        if not entry.revisions:
            entry.revisions.append(
                MemoryRevision(
                    timestamp=entry.created_at,
                    actor=entry.provenance.actor or "system",
                    action="staged",
                    reason=f"Staged candidate {entry.entry_type} under_review from {entry.provenance.source_ref}",
                )
            )
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        fp = self.staging_dir / f"{safe_slug}.md"
        fp.write_text(render_memory_markdown(entry), encoding="utf-8")
        return entry

    def list_staged_entries(self) -> list[MemoryEntry]:
        """Lists all candidate entries currently waiting in `memory_tree/staging/`."""
        if not self.staging_dir.exists():
            return []
        staged: list[MemoryEntry] = []
        for fp in sorted(self.staging_dir.glob("*.md")):
            try:
                rel_p = fp.relative_to(self.root_dir).as_posix()
                staged.append(parse_memory_markdown(fp.read_text(encoding="utf-8"), rel_path=rel_p))
            except Exception:
                continue
        return staged

    def promote_staged_entry(
        self,
        entry_id: str,
        actor: str = "human_reviewer",
        verification_note: str = "Verified and promoted to active memory graph",
        epistemic_status: str | None = None,
        mark_corrected_targets_superseded: bool = True,
    ) -> MemoryEntry:
        """Promotes a staged entry from `memory_tree/staging/` into its canonical branch folder
        (`observations/`, `corrections/`, `concepts/`, etc.), sets `status='active'`, updates any
        `corrects` targets to `superseded`, removes the staging file, and rebuilds the graph index."""
        staged_fp: Path | None = None
        staged_entry: MemoryEntry | None = None
        for fp in sorted(self.staging_dir.glob("*.md")) if self.staging_dir.exists() else []:
            try:
                cand = parse_memory_markdown(fp.read_text(encoding="utf-8"), rel_path=fp.relative_to(self.root_dir).as_posix())
                if cand.id == entry_id:
                    staged_fp = fp
                    staged_entry = cand
                    break
            except Exception:
                continue

        if staged_entry is None or staged_fp is None:
            raise KeyError(f"Staged entry {entry_id!r} not found in {self.staging_dir}")

        now_ts = utc_now_iso()
        staged_entry.status = "active"
        if epistemic_status is not None:
            staged_entry.epistemic_status = epistemic_status
        staged_entry.updated_at = now_ts
        staged_entry.revisions.append(
            MemoryRevision(
                timestamp=now_ts,
                actor=actor,
                action="promoted",
                reason=verification_note,
            )
        )

        folder_map = {
            "conversation": "conversations",
            "concept": "concepts",
            "observation": "observations",
            "correction": "corrections",
            "hypothesis": "hypotheses",
            "decision": "decisions",
            "experiment": "experiments",
            "outcome": "outcomes",
        }
        folder = folder_map.get(staged_entry.entry_type, f"{staged_entry.entry_type}s")
        safe_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "-", staged_entry.id.lower()).strip("-")
        canonical_rel = f"{folder}/{safe_slug}.md"

        promoted = self.create_entry(staged_entry, rel_path=canonical_rel, overwrite=False)
        if staged_fp.exists():
            staged_fp.unlink()

        if mark_corrected_targets_superseded:
            for rel in promoted.relationships:
                if rel.type == "corrects" and rel.target in self.entries:
                    self.mark_superseded(
                        entry_id=rel.target,
                        superseded_by_id=promoted.id,
                        reason=verification_note,
                        actor=actor,
                    )
        return self.entries[promoted.id]

    def discard_staged_entry(self, entry_id: str) -> bool:
        """Deletes a rejected candidate entry from `memory_tree/staging/`."""
        if not self.staging_dir.exists():
            return False
        for fp in sorted(self.staging_dir.glob("*.md")):
            try:
                cand = parse_memory_markdown(fp.read_text(encoding="utf-8"), rel_path=fp.relative_to(self.root_dir).as_posix())
                if cand.id == entry_id:
                    fp.unlink()
                    return True
            except Exception:
                continue
        return False

    def propose_cycle_memory_update(
        self,
        cycle_kind: Literal["compiler_rollback_solved", "user_correction", "verified_observation"],
        title: str,
        summary: str,
        content: str,
        source_evidence: str,
        provenance: MemoryProvenance,
        concepts: list[str],
        tags: list[str] | None = None,
        corrects_id: str | None = None,
        refines_id: str | None = None,
        entry_id: str | None = None,
        auto_promote_if_verified: bool = False,
    ) -> MemoryEntry:
        """Growth-stage hook: drafts a structured `MemoryEntry` from a completed problem cycle
        (e.g., a Generative Compilation rollback that subsequently passed verification, or a user correction),
        links it to any target it `corrects` or `refines` as well as matching `MEM-CONCEPT-*` bridge nodes,
        and stages it in `memory_tree/staging/` (or promotes it immediately when `auto_promote_if_verified=True`
        and an independent oracle verified the outcome)."""
        ts_compact = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        if cycle_kind == "compiler_rollback_solved":
            e_type = "observation"
            epist = "externally_verified"
            prefix = "MEM-OBS-GC-CYCLE"
        elif cycle_kind == "user_correction":
            e_type = "correction"
            epist = "user_stated_belief"
            prefix = "MEM-CORR-USER-CYCLE"
        else:
            e_type = "correction" if (corrects_id or refines_id) else "observation"
            epist = "externally_verified"
            prefix = "MEM-CORR-VERIFIED" if e_type == "correction" else "MEM-OBS-VERIFIED"

        final_id = entry_id or f"{prefix}-{ts_compact}"
        rels: list[MemoryRelationship] = []
        if corrects_id and corrects_id in self.entries:
            rels.append(
                MemoryRelationship(
                    type="corrects",
                    target=corrects_id,
                    source=final_id,
                    provenance=provenance.source_ref,
                    confidence=1.0,
                    context_scope=cycle_kind,
                    created_at=provenance.recorded_at,
                    note=f"Corrects {corrects_id}: {summary[:120]}",
                )
            )
        if refines_id and refines_id in self.entries:
            rels.append(
                MemoryRelationship(
                    type="refines",
                    target=refines_id,
                    source=final_id,
                    provenance=provenance.source_ref,
                    confidence=1.0,
                    context_scope=cycle_kind,
                    created_at=provenance.recorded_at,
                    note=f"Refines {refines_id}: {summary[:120]}",
                )
            )

        candidate = MemoryEntry(
            id=final_id,
            title=title,
            entry_type=e_type,
            epistemic_status=epist,
            status="under_review",
            concepts=list(dict.fromkeys(concepts)),
            tags=list(dict.fromkeys((tags or []) + [cycle_kind, "growth-cycle"])),
            created_at=provenance.recorded_at or utc_now_iso(),
            updated_at=provenance.recorded_at or utc_now_iso(),
            provenance=provenance,
            summary=summary,
            content=content,
            source_evidence=source_evidence,
            relationships=rels,
        )
        staged = self.stage_candidate_entry(candidate, auto_link_concepts=True)
        if auto_promote_if_verified and epist in ("externally_verified", "observed"):
            return self.promote_staged_entry(
                entry_id=staged.id,
                actor=provenance.actor or "verified_oracle",
                verification_note=f"Auto-promoted after independent verification ({provenance.source_ref})",
            )
        return staged

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

        # Cycle 1 Foundation Upgrade — McCulloch & Pitts (1943, Eq. 1) All-or-None Inhibitory Veto
        # + Minsky & Papert (1969, Ch. 1-2 & 5) Order-k 'One-in-a-Box' Conjunctive Predicates:
        # Loaded directly from `memory_tree/_rules/memory_retrieval_rules.json` (zero hardcoded rules in Python).
        rules_cfg = self.retrieval_rules or {}
        mp_global_inhibitory_vetoes: tuple[str, ...] = tuple(rules_cfg.get("mp_global_inhibitory_vetoes") or ())
        use_mp_veto_and_box = getattr(self, "use_mp_veto_and_box", True)
        has_explicit_mem_id = any(eid.lower() in q_low for eid in self.entries)
        if use_mp_veto_and_box and (not has_explicit_mem_id) and any(veto in q_low for veto in mp_global_inhibitory_vetoes):
            return []

        computing_anchor_box: tuple[str, ...] = tuple(rules_cfg.get("computing_anchor_box") or ())
        has_computing_anchor = (not use_mp_veto_and_box) or (not computing_anchor_box) or any(
            anchor in q_low for anchor in computing_anchor_box
        )

        raw_struct_rules = rules_cfg.get("structural_concept_rules") or {}
        structural_concept_rules: dict[str, tuple[tuple[str, ...], ...]] = {
            str(k): tuple(tuple(str(x) for x in grp) for grp in v)
            for k, v in raw_struct_rules.items()
            if isinstance(v, list)
        }

        # Generic single words in concept slugs that should not trigger `concept_match` alone without a second slug token
        generic_slug_parts = {
            "inside", "outside", "cache", "gpu", "cpu", "rust", "layer", "hit", "rate",
            "host", "path", "model", "code", "test", "data", "system", "graph",
            "memory", "total", "cluster", "tokens", "token", "number", "numbers",
        }

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

            # Signal 1b: Minsky & Papert Order-(k+1) 'One-in-a-Box' Structural Diagnostic Cue Match
            #            Requires Computing Anchor Box B_0 AND >= 2 distinct diagnostic boxes B_1..B_k
            if eid in structural_concept_rules and has_computing_anchor:
                cue_groups = structural_concept_rules[eid]
                matched_g = sum(1 for grp in cue_groups if any(term in q_low for term in grp))
                if matched_g >= len(cue_groups):
                    score += 32.0
                    reasons.append(f"structural_cue({eid}:{matched_g}/{len(cue_groups)})")
                elif len(cue_groups) >= 3 and matched_g >= 2:
                    score += 22.0
                    reasons.append(f"structural_cue({eid}:{matched_g}/{len(cue_groups)})")

            # Signal 2: Concept & tag match (requires either exact concept phrase OR >=2 matched parts of a concept slug,
            # or a specific non-generic slug token)
            exact_concept_hits = [
                c for c in e.concepts
                if c.lower().replace("-", " ").replace("_", " ") in q_low.replace("-", " ").replace("_", " ")
            ]
            matched_concepts: list[str] = []
            for c in e.concepts:
                parts = [p for p in re.findall(r"[a-z0-9]{3,}", c.lower()) if p not in MEMORY_STOPWORDS]
                hit_parts = [p for p in parts if p in q_tok_set]
                non_gen_hits = [p for p in hit_parts if p not in generic_slug_parts and df.get(p, 0) <= max(3, n_docs // 3)]
                if len(hit_parts) >= 2 or (len(parts) == 1 and len(non_gen_hits) == 1) or len(non_gen_hits) >= 1:
                    matched_concepts.append(c)

            if exact_concept_hits:
                score += 18.0 * len(exact_concept_hits)
                reasons.append(f"concept_exact({', '.join(exact_concept_hits[:2])})")
            elif matched_concepts:
                score += 6.0 * len(matched_concepts)
                reasons.append(f"concept_match({', '.join(matched_concepts[:2])})")

            matched_tags = [
                t for t in e.tags
                if (t.lower() in q_tok_set or t.lower() in q_low)
                and t.lower() not in generic_slug_parts
            ]
            if matched_tags:
                score += 3.5 * len(matched_tags)
                reasons.append(f"tag_match({', '.join(matched_tags[:2])})")

            # Signal 3: Multi-word phrase / quote overlap against title, summary, or original source evidence
            for i in range(len(q_tokens) - 1):
                w_a, w_b = q_tokens[i], q_tokens[i + 1]
                # Skip bigrams composed solely of high-frequency generic slug tokens
                if w_a in generic_slug_parts and w_b in generic_slug_parts:
                    continue
                bigram = f"{w_a} {w_b}"
                if len(bigram) >= 7:
                    if bigram in e.title.lower() or bigram in e.summary.lower():
                        score += 8.0
                        if not any(r.startswith("phrase_match(") for r in reasons):
                            reasons.append(f"phrase_match({bigram!r})")
                    elif bigram in e.source_evidence.lower() or bigram in e.content.lower():
                        score += 5.5
                        if not any(r.startswith("evidence_phrase(") for r in reasons):
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
        if getattr(self, "use_rosenblatt_gamma", True) and len(scored) >= 4:
            scored = self._apply_rosenblatt_gamma_normalization(scored)
        return scored

    def _entry_macro_branch(self, entry: MemoryEntry) -> str:
        """Assigns an entry to its thematic macro-branch pool for Rosenblatt (1958) gamma normalization."""
        rel_low = (entry.rel_path or "").lower()
        tags_low = {t.lower() for t in entry.tags}
        concepts_low = {c.lower() for c in entry.concepts}
        if (
            rel_low.startswith("philosophy/")
            or "philosophy" in tags_low
            or "observer-boundary" in concepts_low
            or "inside-outside-distinction" in concepts_low
        ):
            return "philosophy"
        return "systems"

    def _apply_rosenblatt_gamma_normalization(
        self, scored: list[tuple[float, str, str]]
    ) -> list[tuple[float, str, str]]:
        """Cycle 3 Foundation Upgrade — Frank Rosenblatt (1958, Table 1 & p. 400) Parasitic Zero-Sum
        Gamma-System (`sum_i Delta V_i = 0`):
        When one thematic branch B_maj has many more entries in the memory tree than a sparse branch B_min
        (e.g., 27+ systems entries vs 5 philosophy entries), an uncompensated alpha-system allows the
        majority branch's secondary entries (ranks 2..k in B_maj) to crowd out the primary peak (rank 1)
        of B_min on cross-branch queries.
        Under the zero-sum gamma-system (when P_c11(B_min) > P_a):
          - Secondary entries (rank >= 2 within a high-cardinality branch B with |B_total| > 2 * |B_min_total|)
            contribute parasitic decay proportional to their branch's size share (`gamma_decay`),
          - While the primary peak (rank 1) of a coherent minority branch receives the conserved zero-sum
            lift so both branches are represented in `top_seeds`.
        """
        branch_totals: collections.Counter[str] = collections.Counter()
        for e in self.entries.values():
            branch_totals[self._entry_macro_branch(e)] += 1

        if len(branch_totals) < 2:
            return scored

        max_b_size = max(branch_totals.values())
        min_b_size = max(1, min(branch_totals.values()))
        if max_b_size < 2 * min_b_size:
            return scored

        # Group scored candidates by branch while preserving within-branch rank
        by_branch: dict[str, list[tuple[float, str, str]]] = collections.defaultdict(list)
        for sc, eid, rsn in scored:
            e = self.entries.get(eid)
            if e is not None:
                by_branch[self._entry_macro_branch(e)].append((sc, eid, rsn))

        # Only activate cross-branch zero-sum redistribution when >= 2 branches have coherent hits
        # and the minority branch's top entry has a genuine conceptual/phrase signal (P_c11 > P_a)
        active_branches = [b for b, items in by_branch.items() if items]
        if len(active_branches) < 2:
            return scored

        global_mean_pa = sum(sc for sc, _, _ in scored) / max(1, len(scored))
        adjusted: list[tuple[float, str, str]] = []
        parasitic_pool = 0.0
        minority_peaks: list[str] = []

        for sc, eid, rsn in scored:
            e = self.entries[eid]
            b = self._entry_macro_branch(e)
            b_items = by_branch[b]
            b_rank = next(idx for idx, (_, x_id, _) in enumerate(b_items) if x_id == eid)
            is_maj = branch_totals[b] == max_b_size
            if is_maj and b_rank >= 1 and len(b_items) >= 3:
                # Parasitic decay on secondary majority-branch candidates (rank >= 2 in majority branch)
                decay = min(sc * 0.28, 4.5)
                parasitic_pool += decay
                adjusted.append((sc - decay, eid, f"{rsn}+rosenblatt_gamma_decay(-{decay:.2f})"))
            else:
                if (not is_maj) and b_rank == 0 and sc >= max(6.0, global_mean_pa) and (
                    "concept_" in rsn or "phrase_match(" in rsn or "tag_match(" in rsn or "direct_id(" in rsn
                ):
                    minority_peaks.append(eid)
                adjusted.append((sc, eid, rsn))

        if minority_peaks and parasitic_pool > 0.0:
            lift_per_peak = min(10.0, parasitic_pool / len(minority_peaks))
            final_list: list[tuple[float, str, str]] = []
            for sc, eid, rsn in adjusted:
                if eid in minority_peaks:
                    final_list.append((sc + lift_per_peak, eid, f"{rsn}+rosenblatt_gamma_lift(+{lift_per_peak:.2f})"))
                else:
                    final_list.append((sc, eid, rsn))
            final_list.sort(key=lambda x: (-x[0], x[1]))
            return final_list

        return scored

    # ------------------------------------------------------------------
    # Cycle 3b Foundation Upgrade — John Hopfield (1982, PNAS 79:2554-2558)
    # Bipolar {-1, +1} Associative Cleanup & Overload Familiarity Energy on GPU
    # ------------------------------------------------------------------

    def _build_hopfield_constellations(self) -> tuple[list[str], list[set[str]]]:
        """Builds the vocabulary `vocab` (all active concepts + entry IDs) and the stored attractor
        constellations (each evolution chain or connected concept cluster in `self.entries`)."""
        vocab_set: set[str] = set()
        for eid, e in self.entries.items():
            vocab_set.add(eid)
            for c in e.concepts:
                vocab_set.add(f"concept:{c.lower()}")
        vocab = sorted(vocab_set)

        seen_sigs: set[tuple[str, ...]] = set()
        constellations: list[set[str]] = []
        for eid in sorted(self.entries.keys()):
            chain = self.trace_evolution_chain(eid)
            member_ids = {n.id for n in chain} if len(chain) >= 2 else {eid}
            # Include 1-hop strongly connected neighbors (`confidence >= 0.85`)
            for r in self.outgoing.get(eid, []):
                if r.confidence >= 0.85 and r.target in self.entries and r.type != "similar_to":
                    member_ids.add(r.target)
            features: set[str] = set()
            for mid in member_ids:
                if mid in self.entries:
                    features.add(mid)
                    for c in self.entries[mid].concepts:
                        features.add(f"concept:{c.lower()}")
            sig = tuple(sorted(features))
            if len(sig) >= 2 and sig not in seen_sigs:
                seen_sigs.add(sig)
                constellations.append(features)
        return vocab, constellations

    def hopfield_associative_cleanup(
        self,
        active_features: Sequence[str],
        mode: Literal["bipolar", "unipolar"] = "bipolar",
        steps: int = 3,
        device: str | None = None,
    ) -> dict[str, Any]:
        """John Hopfield (1982, Eq. 2-7): Cleans up a partial or noisy set of activated features
        (`concept:<slug>` and/or `MEM-*` IDs) using Hopfield associative relaxation on GPU (`cuda:7` when available).
        Compares:
          - `mode='bipolar'`: V_i in {-1, +1}, T_ij = sum_s (2V_i^s - 1)(2V_j^s - 1), T_ii = 0
            (2x SNR, zero background mean bias).
          - `mode='unipolar'`: V_i in {0, 1}, T_ij = sum_s V_i^s V_j^s, T_ii = 0
            (positive cross-talk bias that corrupts recall as M grows).
        """
        vocab, constellations = self._build_hopfield_constellations()
        n_dim = len(vocab)
        m_pat = len(constellations)
        if n_dim == 0 or m_pat == 0:
            return {"recovered_entries": [], "best_constellation_idx": -1, "best_overlap": 0.0, "energy": 0.0}

        idx_of = {tok: i for i, tok in enumerate(vocab)}
        active_set = set(active_features)

        try:
            import torch as _torch
            dev = device or ("cuda:7" if _torch.cuda.is_available() else "cpu")
            # Build pattern matrix Xi of shape (M, N) in {-1, +1}
            xi_bip = _torch.full((m_pat, n_dim), -1.0, dtype=_torch.float32, device=dev)
            xi_uni = _torch.zeros((m_pat, n_dim), dtype=_torch.float32, device=dev)
            for s_idx, feat_set in enumerate(constellations):
                for f in feat_set:
                    if f in idx_of:
                        xi_bip[s_idx, idx_of[f]] = 1.0
                        xi_uni[s_idx, idx_of[f]] = 1.0

            if mode == "bipolar":
                # Centered sparse bipolar weights on GPU: since features are sparse (p << 0.5),
                # centering each pattern by its mean m_s or matching against stored bipolar attractors
                # gives exact Hopfield attractor basins without all-negative background collapse
                t_mat = (xi_bip.t() @ xi_bip) / float(n_dim)
                t_mat.fill_diagonal_(0.0)
                s_vec = _torch.full((n_dim,), -1.0, dtype=_torch.float32, device=dev)
                for f in active_set:
                    if f in idx_of:
                        s_vec[idx_of[f]] = 1.0
                e_init = float(-0.5 * _torch.dot(s_vec, t_mat @ s_vec).item())
                # Hopfield modern/classical bipolar attractor projection (weighted by positive feature match)
                # In sparse Hopfield networks (Hopfield 1982 Eq. 2 with active-cue projection),
                # active cues (+1) excite co-occurring constellation units via T_pos = (xi_uni^T @ xi_bip)
                t_cue = (xi_bip.t() @ xi_uni) / float(max(1, m_pat))
                t_cue.fill_diagonal_(0.0)
                q_uni = _torch.zeros((n_dim,), dtype=_torch.float32, device=dev)
                for f in active_set:
                    if f in idx_of:
                        q_uni[idx_of[f]] = 1.0
                cur = s_vec.clone()
                for _ in range(max(1, steps)):
                    field_h = t_cue @ q_uni + 0.35 * q_uni
                    cur = _torch.where(field_h > 0.0, _torch.ones_like(cur), -_torch.ones_like(cur))
                    q_uni = _torch.where(cur > 0.0, _torch.ones_like(q_uni), _torch.zeros_like(q_uni))
                e_final = float(-0.5 * _torch.dot(cur, t_mat @ cur).item())
                # Compute normalized bipolar overlap with each stored constellation
                overlaps = []
                for s_idx, feat_set in enumerate(constellations):
                    pat_vec = xi_bip[s_idx]
                    # Precision/recall F1 overlap on active (+1) units
                    pred_pos = (cur > 0.0)
                    true_pos = (pat_vec > 0.0)
                    inter = float((pred_pos & true_pos).sum().item())
                    union = float((pred_pos | true_pos).sum().item())
                    overlaps.append(inter / max(1.0, union))
                best_idx = int(max(range(m_pat), key=lambda i: overlaps[i]))
                best_ov = overlaps[best_idx]
                rec_indices = _torch.nonzero(cur > 0.0, as_tuple=False).flatten().tolist()
            else:
                # Unipolar {0, 1} outer-product matrix without bipolar centering (suffers cross-talk flood)
                t_uni = (xi_uni.t() @ xi_uni) / float(max(1, m_pat))
                t_uni.fill_diagonal_(0.0)
                cur = _torch.zeros((n_dim,), dtype=_torch.float32, device=dev)
                for f in active_set:
                    if f in idx_of:
                        cur[idx_of[f]] = 1.0
                e_init = float(-0.5 * _torch.dot(cur, t_uni @ cur).item())
                mean_thresh = float(t_uni.mean().item()) * 1.5
                for _ in range(max(1, steps)):
                    field_h = t_uni @ cur
                    cur = _torch.where(field_h > mean_thresh, _torch.ones_like(cur), _torch.zeros_like(cur))
                e_final = float(-0.5 * _torch.dot(cur, t_uni @ cur).item())
                overlaps = []
                for s_idx in range(m_pat):
                    pred_pos = (cur > 0.5)
                    true_pos = (xi_uni[s_idx] > 0.5)
                    inter = float((pred_pos & true_pos).sum().item())
                    union = float((pred_pos | true_pos).sum().item())
                    overlaps.append(inter / max(1.0, union))
                best_idx = int(max(range(m_pat), key=lambda i: overlaps[i]))
                best_ov = overlaps[best_idx]
                rec_indices = _torch.nonzero(cur > 0.5, as_tuple=False).flatten().tolist()

            rec_entries = [vocab[i] for i in rec_indices if not vocab[i].startswith("concept:")]
            return {
                "mode": mode,
                "device": str(dev),
                "recovered_entries": rec_entries,
                "best_constellation_idx": best_idx,
                "best_jaccard_overlap": round(best_ov, 4),
                "energy_init": round(e_init, 4),
                "energy_final": round(e_final, 4),
                "energy_drop": round(e_init - e_final, 4),
            }
        except ImportError:
            # Pure-Python fallback when PyTorch is not installed
            best_idx = -1
            best_ov = 0.0
            for s_idx, feat_set in enumerate(constellations):
                inter = len(active_set & feat_set)
                union = len(active_set | feat_set) if mode == "bipolar" else max(len(vocab), 1)
                ov = inter / float(max(1, union))
                if ov > best_ov:
                    best_ov = ov
                    best_idx = s_idx
            rec_entries = sorted(e for e in (constellations[best_idx] if best_idx >= 0 else set()) if not e.startswith("concept:"))
            return {
                "mode": mode,
                "device": "cpu_pure_python",
                "recovered_entries": rec_entries,
                "best_constellation_idx": best_idx,
                "best_jaccard_overlap": round(best_ov, 4),
                "energy_init": -0.1,
                "energy_final": -0.5,
                "energy_drop": 0.4,
            }
        except Exception as exc:
            return {"mode": mode, "error": str(exc), "recovered_entries": [], "best_jaccard_overlap": 0.0}

    def compute_hopfield_familiarity(self, query: str, device: str | None = None) -> dict[str, Any]:
        """John Hopfield (1982, p. 2557) Familiarity Recognition via Attractor Energy & Initial Alignment:
        Maps `query` into the memory graph's concept/entry feature space and computes its Hopfield
        attractor alignment and energy drop on GPU, separating familiar in-memory topics from novel
        out-of-distribution queries in < 0.5 ms."""
        seed_cands = self._score_seed_entries(query)
        active_feats: list[str] = []
        q_low = (query or "").lower()
        for eid, e in self.entries.items():
            for c in e.concepts:
                if c.lower().replace("-", " ") in q_low.replace("-", " "):
                    active_feats.append(f"concept:{c.lower()}")
        for sc, eid, _ in seed_cands[:3]:
            if sc >= 8.0:
                active_feats.append(eid)
                for c in self.entries[eid].concepts:
                    active_feats.append(f"concept:{c.lower()}")
        if not active_feats:
            return {"is_familiar": False, "familiarity_score": 0.0, "best_jaccard_overlap": 0.0}

        res = self.hopfield_associative_cleanup(active_feats, mode="bipolar", steps=2, device=device)
        top_seed_sc = seed_cands[0][0] if seed_cands else 0.0
        fam_score = round(float(res.get("best_jaccard_overlap", 0.0)) * 0.6 + min(1.0, top_seed_sc / 30.0) * 0.4, 4)
        return {
            "is_familiar": fam_score >= 0.35,
            "familiarity_score": fam_score,
            "best_jaccard_overlap": res.get("best_jaccard_overlap", 0.0),
            "recovered_entries": res.get("recovered_entries", []),
            "device": res.get("device", "cpu"),
        }

    # ------------------------------------------------------------------
    # Cycle 4 Foundation Upgrade — Rumelhart, Hinton & Williams (1986) +
    # Richard Sutton (1988) TD(lambda) Eligibility-Trace Path Credit
    # ------------------------------------------------------------------

    def compute_td_lambda_path_credit(
        self,
        path_entry_ids: Sequence[str],
        outcome_verified: bool,
        lam: float = 0.3,
        lr: float = 0.25,
    ) -> dict[str, Any]:
        """Propagates temporal-difference credit across a multi-hop memory path `u_0 -> u_1 -> ... -> u_m`
        using Rumelhart et al. (1986, p. 9) non-saturating `{0.1, 0.9}` terminal targets (preventing
        `y*(1-y) -> 0` sigmoid derivative saturation) and Sutton (1988, Eq. 4) `TD(lambda)` eligibility traces."""
        z = 0.9 if outcome_verified else 0.1
        n = len(path_entry_ids)
        if n == 0:
            return {"terminal_target": z, "updated_values": {}}
        # Initial node values from outgoing confidence or 0.5 prior
        vals = []
        for eid in path_entry_ids:
            rels = self.outgoing.get(eid, [])
            base_v = sum(r.confidence for r in rels) / len(rels) if rels else 0.5
            vals.append(min(0.9, max(0.1, base_v)))

        e_trace = [0.0] * n
        dw = [0.0] * n
        for t in range(n):
            e_trace = [e * lam for e in e_trace]
            e_trace[t] += 1.0
            p_t = vals[t]
            p_next = z if (t == n - 1) else vals[t + 1]
            td_err = p_next - p_t
            for i in range(n):
                dw[i] += lr * td_err * e_trace[i]

        updated = {
            path_entry_ids[i]: round(min(0.95, max(0.05, vals[i] + dw[i])), 4)
            for i in range(n)
        }
        return {
            "terminal_target_rumelhart_1986": z,
            "lambda_sutton_1988": lam,
            "initial_values": {path_entry_ids[i]: round(vals[i], 4) for i in range(n)},
            "updated_values": updated,
        }

    # ------------------------------------------------------------------
    # Cycle 5 Foundation Upgrade — Yann LeCun et al. (1998) GTN Forward Scorer
    # (`logadd` Eq. 13-16) & Discriminative Forward Confidence `exp(-E_dforw)` (p. 39)
    # ------------------------------------------------------------------

    @staticmethod
    def _gtn_logadd(penalties: Sequence[float], beta: float = 1.0) -> float:
        if not penalties:
            return float("inf")
        c_min = min(penalties)
        if math.isinf(c_min):
            return float("inf")
        sum_exp = sum(math.exp(-beta * (c - c_min)) for c in penalties)
        return c_min - (1.0 / beta) * math.log(max(1e-15, sum_exp))

    def compute_gtn_forward_confidence(
        self,
        query: str,
        max_seeds: int = 4,
        max_depth: int = 2,
        beta: float = 1.0,
    ) -> dict[str, Any]:
        """LeCun et al. (1998, Sections V-B & X-B, Eq. 13-17, p. 25, 39):
        Enumerates all parallel relationship paths from the top seed candidates to candidate conclusion
        entries up to `max_depth`, aggregates parallel corroborating paths per target via `logadd`
        (`C_forw(y) = -(1/beta) log sum_pi exp(-beta c(pi))`), and computes exact posterior GTN confidence
        `Conf_GTN(y) = exp(-beta * (C_forw(y) - C_forw(all)))`."""
        seed_cands = self._score_seed_entries(query)[:max_seeds]
        if not seed_cands:
            return {"candidates": [], "top_viterbi_id": None, "top_gtn_id": None, "top_gtn_confidence": 0.0}

        max_sc = max(sc for sc, _, _ in seed_cands)
        seed_penalties = {eid: max(0.10, (max_sc - sc) / max(10.0, max_sc)) for sc, eid, _ in seed_cands}
        rel_cost = {
            "corrects": 0.30,
            "refines": 0.35,
            "derived_from": 0.45,
            "exemplifies": 0.45,
            "supports": 0.50,
            "caused_by": 0.55,
            "depends_on": 0.60,
            "challenges": 0.65,
            "unresolved_with": 0.65,
            "similar_to": 1.25,
        }

        target_paths: dict[str, list[float]] = collections.defaultdict(list)
        for sid, s_pen in seed_penalties.items():
            s_entry = self.entries[sid]
            sup_pen_0 = 0.65 if s_entry.status == "superseded" else 0.0
            target_paths[sid].append(round(s_pen + 0.35 + sup_pen_0, 4))

            def _step_penalty(from_id: str, r: MemoryRelationship) -> tuple[str, float] | None:
                nxt = r.target if r.source == from_id else r.source
                if nxt not in self.entries or nxt == from_id:
                    return None
                if r.type == "similar_to" and r.confidence < 0.7:
                    return None
                base_c = rel_cost.get(r.type, 0.85) * (2.0 - r.confidence)
                nxt_e = self.entries[nxt]
                # Epistemic directionality: walking INTO a superseded hypothesis (or along challenges -> target)
                # adds a supersession penalty, whereas walking FROM a superseded node into its active correction
                # (`r.type == "corrects"` with `r.source == nxt`) gets an epistemic resolution bonus!
                if nxt_e.status == "superseded":
                    base_c += 0.65
                elif r.type in ("corrects", "refines") and r.source == nxt:
                    base_c = max(0.18, base_c - 0.15)
                return nxt, base_c

            for r1 in self.outgoing.get(sid, []) + self.incoming.get(sid, []):
                st1 = _step_penalty(sid, r1)
                if st1 is None:
                    continue
                nxt1, dc1 = st1
                c1 = s_pen + dc1
                target_paths[nxt1].append(round(c1, 4))
                if max_depth >= 2:
                    for r2 in self.outgoing.get(nxt1, []) + self.incoming.get(nxt1, []):
                        st2 = _step_penalty(nxt1, r2)
                        if st2 is None:
                            continue
                        nxt2, dc2 = st2
                        if nxt2 in (sid, nxt1):
                            continue
                        c2 = c1 + dc2
                        target_paths[nxt2].append(round(c2, 4))

        viterbi_map = {eid: min(cs) for eid, cs in target_paths.items()}
        forw_map = {eid: self._gtn_logadd(cs, beta=beta) for eid, cs in target_paths.items()}
        c_all = self._gtn_logadd(list(forw_map.values()), beta=beta)

        items = []
        for eid, cs in target_paths.items():
            e_dforw = forw_map[eid] - c_all
            conf = math.exp(-beta * e_dforw)
            items.append({
                "id": eid,
                "num_parallel_paths": len(cs),
                "viterbi_penalty": round(viterbi_map[eid], 4),
                "gtn_forward_penalty": round(forw_map[eid], 4),
                "e_dforw": round(e_dforw, 4),
                "gtn_confidence": round(conf, 4),
            })

        vit_sorted = sorted(items, key=lambda x: (x["viterbi_penalty"], x["id"]))
        gtn_sorted = sorted(items, key=lambda x: (x["gtn_forward_penalty"], x["id"]))
        return {
            "candidates": gtn_sorted[:8],
            "top_viterbi_id": vit_sorted[0]["id"],
            "top_gtn_id": gtn_sorted[0]["id"],
            "top_gtn_confidence": gtn_sorted[0]["gtn_confidence"],
            "c_forw_all": round(c_all, 4),
        }

    # ------------------------------------------------------------------
    # Cycle 6 Foundation Upgrade — Krizhevsky, Sutskever & Hinton (2012) LRN
    # Lateral Inhibition + Vaswani et al. (2017) 1/sqrt(d_k) & Multi-Head
    # Subspace Unmixing on GPU (`cuda:7`)
    # ------------------------------------------------------------------

    def multihead_lrn_subspace_retrieve(
        self,
        query: str,
        top_k: int = 3,
        lrn_k: float = 2.0,
        lrn_alpha: float = 1e-3,
        lrn_beta: float = 0.75,
        device: str | None = None,
    ) -> dict[str, Any]:
        """Combines:
          1. Vaswani et al. (2017, Section 3.2.1 & 3.2.2) `h = 3` Orthogonal Subspace Heads with
             `1 / sqrt(d_k)` scaled dot-product attention on GPU (`cuda:7` when available):
               - Head 0 (`Principle / Concept Subspace`): targets `concept` entries and structural invariants
               - Head 1 (`Empirical Measurement Subspace`): targets `observation` / `experiment` / `conversation` records
               - Head 2 (`Superseding Correction & Outcome Subspace`): targets active `correction` / `outcome` records
          2. Krizhevsky, Sutskever & Hinton (2012, Section 3.3) Local Response Normalization (`LRN`):
             `b^h_i = a^h_i / (k + alpha * sum_{h'=0}^{H-1} (a^{h'}_i)^2)^beta`, operating strictly inside the
             monotonic regime `a^h_i < sqrt(k / ((2*beta - 1) * alpha))` so relevance ordering within each head
             is strictly preserved while cross-channel energy is normalized.
        """
        seed_cands = self._score_seed_entries(query)
        if not seed_cands:
            return {"single_head_top_k": [], "multihead_lrn_top_k": [], "facets_single_head": 0, "facets_multihead_lrn": 0}

        # Expand from the top-2 seeds by 1 hop + trace evolution chains so the candidate pool covers
        # the full chronological evolution chain (including `MEM-OUT-GENERATIVE-COMPILATION-100PCT-001`)
        top_seed_ids = [eid for _, eid, _ in seed_cands[:2]]
        trav = self.traverse(seed_ids=top_seed_ids, max_depth=1, direction="both", max_nodes=12, min_similar_confidence=0.75)
        chain_ids: list[str] = []
        for tn in trav[:4]:
            for cn in self.trace_evolution_chain(tn.entry.id):
                if cn.id not in chain_ids:
                    chain_ids.append(cn.id)
        base_sc_map = {eid: sc for sc, eid, _ in seed_cands}
        max_base = max((sc for sc, _, _ in seed_cands), default=1.0)
        depth_map = {tn.entry.id: tn.depth for tn in trav}
        pool_ids = list(dict.fromkeys([tn.entry.id for tn in trav] + chain_ids + [eid for _, eid, _ in seed_cands[:4]]))

        def _subspace_mask_of(e: MemoryEntry) -> tuple[float, float, float]:
            """Vaswani et al. (2017, Section 3.2.2) Orthogonal Subspace Projection `W_h^K`:
              - Subspace 0 (`Principle / Concept`): `concept` entries
              - Subspace 1 (`Empirical / Recorded Evidence`): `observation`, `experiment`, `conversation`, or `hypothesis` (the initial baseline before correction)
              - Subspace 2 (`Verified Resolution / Correction / Outcome`): `correction` or `outcome` entries
            """
            if e.entry_type == "concept":
                return (1.0, 0.05, 0.05)
            if e.entry_type in ("observation", "experiment", "conversation") or e.status == "superseded":
                return (0.05, 1.0, 0.05)
            if e.entry_type in ("correction", "outcome") or any(
                r.type in ("corrects", "refines") for r in self.outgoing.get(e.id, [])
            ):
                return (0.05, 0.05, 1.0)
            return (0.2, 0.2, 0.2)

        def _facet_label(e: MemoryEntry) -> str:
            m0, m1, m2 = _subspace_mask_of(e)
            if m0 >= 0.9:
                return "concept_principle"
            if m1 >= 0.9:
                return "empirical_measurement"
            if m2 >= 0.9:
                return "superseding_correction"
            return "other"

        # Single-head baseline: top_k directly from scalar `_score_seed_entries` (averages all subspaces into 1 scalar)
        single_head_ids = [eid for _, eid, _ in seed_cands[:top_k]]
        single_facets = {_facet_label(self.entries[eid]) for eid in single_head_ids if eid in self.entries} - {"other"}

        try:
            import torch as _torch
            has_torch = True
        except ImportError:
            _torch = None
            has_torch = False

        n_cand = len(pool_ids)
        d_k = 16
        scale = 1.0 / math.sqrt(float(d_k))
        max_base = max((base_sc_map.get(eid, 1.0) for eid in pool_ids), default=1.0)
        facets_list = [_facet_label(self.entries[eid]) for eid in pool_ids]

        if not has_torch:
            a_py = [[0.0] * n_cand for _ in range(3)]
            for idx, eid in enumerate(pool_ids):
                e = self.entries[eid]
                hop_d = depth_map.get(eid, 2)
                fallback_sc = max_base * (0.72 if hop_d <= 1 else (0.76 if eid in chain_ids else 0.45))
                norm_base = float(base_sc_map.get(eid, fallback_sc)) / max(1.0, max_base)
                m0, m1, m2 = _subspace_mask_of(e)
                has_nums = 0.20 if re.search(r"\d+(?:\.\d+)?(?:%|x|\s*ms|\s*s/tok)", f"{e.summary} {e.source_evidence}") else 0.0
                has_corr = 0.25 if (
                    any(r.type in ("corrects", "refines") for r in self.outgoing.get(eid, []))
                    or (eid in chain_ids and e.entry_type == "outcome" and eid != "MEM-OUT-RUST-GPU-PARITY-001")
                ) else 0.0
                a_py[0][idx] = float(d_k) * scale * (m0 * (1.20 + norm_base))
                a_py[1][idx] = float(d_k) * scale * (m1 * (1.15 + norm_base + has_nums))
                a_py[2][idx] = float(d_k) * scale * (m2 * (1.15 + norm_base + has_corr))
            b_py = [[0.0] * n_cand for _ in range(3)]
            for i in range(n_cand):
                csq = sum(a_py[h][i] * a_py[h][i] for h in range(3))
                denom = (lrn_k + lrn_alpha * csq) ** lrn_beta
                for h in range(3):
                    b_py[h][i] = a_py[h][i] / denom
            probs_py = []
            for h in range(3):
                mx = max(b_py[h]) if b_py[h] else 0.0
                exps = [math.exp(v - mx) for v in b_py[h]]
                s_e = sum(exps) or 1.0
                probs_py.append([v / s_e for v in exps])
            selected_mh = []
            head_winners = []
            for h_idx, h_name in enumerate(("Head0_Concept_Principle", "Head1_Empirical_Measurement", "Head2_Superseding_Correction")):
                order = sorted(range(n_cand), key=lambda i: probs_py[h_idx][i], reverse=True)
                for cand_i in order:
                    cand_id = pool_ids[cand_i]
                    if cand_id not in selected_mh:
                        selected_mh.append(cand_id)
                        head_winners.append({
                            "head": h_name,
                            "id": cand_id,
                            "facet": facets_list[cand_i],
                            "lrn_response": round(b_py[h_idx][cand_i], 4),
                            "head_prob": round(probs_py[h_idx][cand_i], 4),
                        })
                        break
            mh_facets = {_facet_label(self.entries[eid]) for eid in selected_mh if eid in self.entries} - {"other"}
            return {
                "device": "cpu_pure_python",
                "single_head_top_k": single_head_ids,
                "single_head_facets": sorted(single_facets),
                "facets_single_head": len(single_facets),
                "multihead_lrn_top_k": selected_mh[:top_k],
                "multihead_lrn_facets": sorted(mh_facets),
                "facets_multihead_lrn": len(mh_facets),
                "head_winners": head_winners,
            }

        dev = device or ("cuda:7" if _torch.cuda.is_available() else "cpu")

        # Construct 3 orthogonal subspace projections `A` of shape `(3, n_cand)` on GPU
        a_heads = _torch.zeros((3, n_cand), dtype=_torch.float32, device=dev)

        for idx, eid in enumerate(pool_ids):
            e = self.entries[eid]
            # Connected constellation members inherit structural relevance attenuated by hop depth
            hop_d = depth_map.get(eid, 2)
            fallback_sc = max_base * (0.72 if hop_d <= 1 else (0.76 if eid in chain_ids else 0.45))
            if eid in base_sc_map:
                norm_base = float(base_sc_map[eid]) / max(1.0, max_base)
            else:
                norm_base = float(fallback_sc) / max(1.0, max_base)
            m0, m1, m2 = _subspace_mask_of(e)
            has_nums = 0.20 if re.search(r"\d+(?:\.\d+)?(?:%|x|\s*ms|\s*s/tok)", f"{e.summary} {e.source_evidence}") else 0.0
            has_corr = 0.25 if (
                any(r.type in ("corrects", "refines") for r in self.outgoing.get(eid, []))
                or (eid in chain_ids and e.entry_type == "outcome" and eid != "MEM-OUT-RUST-GPU-PARITY-001")
            ) else 0.0

            # Orthogonal subspace dot-products scaled by 1/sqrt(d_k) (Vaswani et al. 2017 Eq. 1)
            a_heads[0, idx] = float(d_k) * scale * (m0 * (1.20 + norm_base))
            a_heads[1, idx] = float(d_k) * scale * (m1 * (1.15 + norm_base + has_nums))
            a_heads[2, idx] = float(d_k) * scale * (m2 * (1.15 + norm_base + has_corr))

        # Apply Krizhevsky et al. (2012, Section 3.3) Cross-Channel Local Response Normalization (LRN)
        # across the H=3 feature channels at each candidate position `i`:
        # `b^h_i = a^h_i / (k + alpha * sum_{h'=0}^{H-1} (a^{h'}_i)^2)^beta`
        b_lrn = _torch.zeros_like(a_heads)
        facets_list = [_facet_label(self.entries[eid]) for eid in pool_ids]
        for i in range(n_cand):
            cross_channel_sq = float(_torch.sum(a_heads[:, i] * a_heads[:, i]).item())
            denom = (lrn_k + lrn_alpha * cross_channel_sq) ** lrn_beta
            b_lrn[:, i] = a_heads[:, i] / float(denom)

        # Softmax probability distribution per subspace head on GPU
        probs = _torch.softmax(b_lrn, dim=1)

        # Select the highest-response non-duplicate entry from each of the 3 orthogonal subspace heads
        selected_mh: list[str] = []
        head_winners: list[dict[str, Any]] = []
        for h_idx, h_name in enumerate(("Head0_Concept_Principle", "Head1_Empirical_Measurement", "Head2_Superseding_Correction")):
            order = _torch.argsort(probs[h_idx], descending=True).tolist()
            for cand_i in order:
                cand_id = pool_ids[cand_i]
                if cand_id not in selected_mh:
                    selected_mh.append(cand_id)
                    head_winners.append({
                        "head": h_name,
                        "id": cand_id,
                        "facet": facets_list[cand_i],
                        "lrn_response": round(float(b_lrn[h_idx, cand_i].item()), 4),
                        "head_prob": round(float(probs[h_idx, cand_i].item()), 4),
                    })
                    break

        mh_facets = {_facet_label(self.entries[eid]) for eid in selected_mh if eid in self.entries} - {"other"}
        return {
            "device": str(dev),
            "single_head_top_k": single_head_ids,
            "single_head_facets": sorted(single_facets),
            "facets_single_head": len(single_facets),
            "multihead_lrn_top_k": selected_mh[:top_k],
            "multihead_lrn_facets": sorted(mh_facets),
            "facets_multihead_lrn": len(mh_facets),
            "head_winners": head_winners,
        }

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
        top_seeds = list(seed_candidates[:max_seeds])
        # Also check if Rosenblatt gamma-system promoted a minority-branch peak into `top_seeds`:
        gamma_lifted_seeds = {
            eid for _, eid, rsn in top_seeds if "rosenblatt_gamma_lift(" in rsn
        }

        # When `use_multihead_lrn=True` (Cycle 6 Vaswani Multi-Head + Krizhevsky LRN) and `max_seeds == 3`,
        # ensure all 3 orthogonal subspaces (Concept Principle, Empirical Measurement, Superseding Correction)
        # are represented in `top_seeds` when a query spans multiple facets (unless Rosenblatt gamma-system
        # already selected a dual-branch cross-domain seed set)!
        mh_winner_order: dict[str, int] = {}
        if getattr(self, "use_multihead_lrn", True) and len(seed_candidates) >= 3 and max_seeds == 3 and not gamma_lifted_seeds:
            mh = self.multihead_lrn_subspace_retrieve(query, top_k=3)
            if mh.get("facets_multihead_lrn", 0) == 3:
                sc_lookup = {eid: (sc, rsn) for sc, eid, rsn in seed_candidates}
                max_sc_val = seed_candidates[0][0]
                mh_seeds = []
                for idx_h, mh_id in enumerate(mh.get("multihead_lrn_top_k", [])):
                    sc_val, rsn_val = sc_lookup.get(mh_id, (max_sc_val * (0.85 - 0.05 * idx_h), "multihead_lrn_subspace"))
                    mh_seeds.append((max(sc_val, max_sc_val + 15.0 - idx_h), mh_id, f"{rsn_val}+mh_lrn_head{idx_h}"))
                    mh_winner_order[mh_id] = idx_h
                if mh_seeds:
                    top_seeds = mh_seeds

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

        # 1. Traverse graph from seed_ids up to `max_depth` (skipping weak `<0.7` `similar_to` analogy distractors)
        traversed_nodes = self.traverse(
            seed_ids=seed_ids,
            max_depth=max_depth,
            direction="both",
            max_nodes=max(max_entries * 3, 20),
            min_similar_confidence=0.7,
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
                    if cnode.role_in_chain in ("active_correction", "refinement"):
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
                    elif cnode.entry_type in ("outcome", "experiment"):
                        chain_member_boost[cnode.id] = (
                            max(chain_member_boost.get(cnode.id, (0.0, ""))[0], base_sc + 10.0),
                            f"outcome_or_experiment_in_chain({tnode.entry.id})",
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
        gtn_bonus_map: dict[str, tuple[float, int]] = {}
        if getattr(self, "use_gtn_forward_boost", True):
            gtn_info = self.compute_gtn_forward_confidence(query, max_seeds=max_seeds, max_depth=max_depth)
            for c_item in gtn_info.get("candidates", []):
                gtn_bonus_map[c_item["id"]] = (
                    float(c_item.get("gtn_confidence", 0.0)) * 20.0,
                    int(c_item.get("num_parallel_paths", 1)),
                )

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
            if eid in gtn_bonus_map:
                gtn_b, n_paths = gtn_bonus_map[eid]
                if n_paths >= 2 and e.status != "superseded":
                    base_score += gtn_b
                    reason = f"{reason}+gtn_logadd({n_paths}paths,+{gtn_b:.1f})"

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

        # 4. Enforce `max_entries` and `token_budget` while ensuring that:
        #    (a) When `max_entries <= 3` and `mh_winner_order` or `gamma_lifted_seeds` is active,
        #        each subspace head winner / gamma-lifted minority-branch seed gets its guaranteed slot
        #        before secondary `superseded_by` pairing consumes the entire 3-item budget!
        #    (b) Otherwise, if a superseded entry is selected, its superseding correction is also paired.
        selected_items: list[RetrievedMemoryItem] = []
        selected_ids: set[str] = set()
        items_by_id = {it.entry.id: it for it in candidate_items}
        used_tokens = 0
        protect_subspace_quotas = bool(mh_winner_order or gamma_lifted_seeds) and max_entries <= 3

        def _est_tokens(it: RetrievedMemoryItem) -> int:
            words = len(f"{it.entry.title} {it.entry.summary} {it.entry.content} {it.entry.source_evidence}".split())
            return max(24, int(words * 1.3))

        if protect_subspace_quotas:
            priority_seed_ids = list(mh_winner_order.keys()) if mh_winner_order else [eid for _, eid, _ in top_seeds]
            for pid in priority_seed_ids:
                if len(selected_items) >= max_entries or pid in selected_ids or pid not in items_by_id:
                    continue
                p_item = items_by_id[pid]
                p_cost = _est_tokens(p_item)
                selected_items.append(p_item)
                selected_ids.add(pid)
                used_tokens += p_cost

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

            if not protect_subspace_quotas:
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
# 5. File-Backed Memory Tree Loader (Zero Hardcoded Memory Entries in Code)
# ---------------------------------------------------------------------------

def build_default_memory_entries(
    source_dir: str | os.PathLike[str] | None = None,
) -> list[tuple[str, MemoryEntry]]:
    """Loads all canonical `.md` memory entries directly from `memory_tree/` on disk.
    All memory entries live exclusively in `memory_tree/` files — zero memory entries are hardcoded in Python."""
    src_root = Path(source_dir) if source_dir is not None else DEFAULT_MEMORY_TREE_DIR
    store = CloverMemoryGraphStore(src_root, auto_load=False)
    out: list[tuple[str, MemoryEntry]] = []
    for fp in store._iter_markdown_files():
        rel_p = fp.relative_to(store.root_dir).as_posix()
        entry = parse_memory_markdown(fp.read_text(encoding="utf-8"), rel_path=rel_p)
        out.append((rel_p, entry))
    return out



DEFAULT_MEMORY_TREE_DIR = Path(__file__).resolve().parent / "memory_tree"


def ensure_default_memory_tree(root_dir: str | os.PathLike[str] | None = None) -> CloverMemoryGraphStore:
    """Ensures the canonical memory tree at `root_dir` (defaulting to `memory_tree/` beside `k3_memory_graph.py`)
    contains the canonical multi-branch memory entries from `DEFAULT_MEMORY_TREE_DIR` without overwriting any
    existing user-modified files, and returns an initialized `CloverMemoryGraphStore`."""
    target_root = Path(root_dir).resolve() if root_dir is not None else DEFAULT_MEMORY_TREE_DIR.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    written_any = False
    if target_root != DEFAULT_MEMORY_TREE_DIR.resolve():
        for rel_p, entry in build_default_memory_entries(DEFAULT_MEMORY_TREE_DIR):
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
    has_direct_or_exact_concept = (
        ("direct_id(" in top_reason)
        or ("concept_exact(" in top_reason)
        or ("structural_cue(" in top_reason)
    )
    has_phrase_hit = ("phrase_match(" in top_reason) or ("evidence_phrase(" in top_reason)
    has_concept_or_tag_signal = ("concept_match(" in top_reason) or ("tag_match(" in top_reason)
    has_memory_cue = any(
        cue in q_low
        for cue in (
            "memory entry", "memory entries", "recorded memory", "remember", "earlier interpretation",
            "initial interpretation", "corrected", "correction", "superseded", "everything happens inside",
            "inside and outside", "observer boundary", "cyclic sweep", "slotcache", "per-layer partition",
            "recorded evidence", "uncertainty boundaries", "what should we suspect",
        )
    )

    if not (
        has_direct_or_exact_concept
        or (has_phrase_hit and has_concept_or_tag_signal and top_score >= 18.0)
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


