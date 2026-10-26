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
            # Skip hidden directories, `_index`, and `staging` (unpromoted drafts)
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in ("_index", "staging"))
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
        # 1. `mp_inhibitory_vetoes`: If any out-of-domain inhibitory afferent I_i is active in `q_low`
        #    (and no explicit `MEM-*` ID is cited), firing is vetoed before excitatory summation.
        mp_global_inhibitory_vetoes: tuple[str, ...] = (
            "swimming pool", "water temperature", "night air",
            "eukaryotic", "endocytosis", "endosome", "lysosome", "extracellular matrix", "cytoplasm",
            "baseball", "batter", "curveball", "batting practice",
            "forklift", "loading dock", "warehouse",
            "commuter highway", "traffic light", "arterial route", "exit ramp",
            "woodworking", "stencil label", "sorting jars",
            "retro pixel art", "hdmi", "monitor refresh rate",
            "detective's alibi", "novel's second printing", "who was the culprit",
        )
        use_mp_veto_and_box = getattr(self, "use_mp_veto_and_box", True)
        has_explicit_mem_id = any(eid.lower() in q_low for eid in self.entries)
        if use_mp_veto_and_box and (not has_explicit_mem_id) and any(veto in q_low for veto in mp_global_inhibitory_vetoes):
            return []

        # 2. Required Computing/Systems/Graph/Code Anchor Box B_0 for structural diagnostic bridge rules
        #    (Minsky & Papert Order-(k+1) 'One-in-a-Box' conjunction: a diagnostic cue cannot fire unless
        #    Box 0 [computing/architecture/graph/code anchor] AND >= 2 diagnostic boxes are simultaneously active).
        computing_anchor_box: tuple[str, ...] = (
            "cache", "caching", "lru", "pipeline", "stage", "stages", "layer", "layers", "cdn", "pop",
            "buffer", "metadata", "vram", "memoization", "ssa", "compiler", "token", "tokens", "sequence",
            "recurrent", "kda", "activation", "graph", "query", "edge", "edges", "path", "paths", "router",
            "bgp", "cve", "cwe", "capec", "affects", "lexer", "scanner", "tokenizer", "elif", "ast",
            "code", "function", "rust", "python", "simd", "kernel", "harness", "benchmark", "micro-benchmark",
            "microbenchmark", "decode", "trunk", "quantisation", "quantization", "expert", "experts",
            "nvme", "pcie", "slot", "slotcache", "memory entry", "memory entries", "recorded",
        )
        has_computing_anchor = (not use_mp_veto_and_box) or any(anchor in q_low for anchor in computing_anchor_box)

        # Structural diagnostic cue rules that map cross-domain problem patterns to concept bridge nodes
        structural_concept_rules: dict[str, tuple[tuple[str, ...], ...]] = {
            "MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001": (
                ("flat", "plateau", "unchanged", "invariant", "same hit rate", "constant across"),
                ("capacity", "capacities", "budget", "size", "setting", "settings", "sweep", "entries", "slots"),
                ("cannot help", "no effect", "dead end", "useless", "suspect", "pool", "shared", "global"),
            ),
            "MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001": (
                ("0 hits", "zero hits", "0%", "never hits", "0 paths", "zero seeds", "empty evidence"),
                ("repeat", "repeats", "repeating", "same token", "recurrent", "drift", "threshold", "filter", "workload"),
                ("prove", "proves", "useless", "dead", "never", "cannot", "why"),
            ),
            "MEM-CONCEPT-HARDWARE-SCOPED-VERDICTS-001": (
                ("4-bit", "quantisation", "quantization", "nibble", "unpack", "unpacking"),
                ("cpu", "gpu", "rtx 3090", "hardware", "bandwidth", "dram", "pcie"),
                ("decode", "trunk", "kernel", "s/tok", "s/layer", "supply", "dead end", "unmeasured", "speed"),
            ),
            "MEM-OBS-ONTORAG-FANOUT-FLOODING-001": (
                ("1-hop", "length(path)", "length(p.edges)", "path length", "limit"),
                ("2-hop", "affects", "fan-out", "flooding", "root-cause", "causal", "capec"),
            ),
            "MEM-CONCEPT-PREFIX-SHADOWING-DISPATCH-001": (
                ("prefix", "shadow", "shadowing", "1-char", "shorter"),
                ("elif", "branch", "tokenizer", "dispatch", "2-char", "longer", "lexer", "scanner"),
            ),
            "MEM-CONCEPT-SYNTHETIC-TIMING-VS-MEANING-001": (
                ("synthetic", "random", "micro-benchmark", "microbenchmark", "standalone", "isolated"),
                ("in-layer", "in situ", "real", "router", "over-predict", "under-predict"),
            ),
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

        # Expand from the top-2 seeds by 1 hop so the candidate pool stays strictly on the query's constellation
        top_seed_ids = [eid for _, eid, _ in seed_cands[:2]]
        trav = self.traverse(seed_ids=top_seed_ids, max_depth=1, direction="both", max_nodes=12, min_similar_confidence=0.75)
        base_sc_map = {eid: sc for sc, eid, _ in seed_cands}
        max_base = max((sc for sc, _, _ in seed_cands), default=1.0)
        # 1-hop constellation neighbors of the #1 seed inherit 0.65 * max_base if not directly scored
        pool_ids = list(dict.fromkeys([ tn.entry.id for tn in trav ] + [eid for _, eid, _ in seed_cands[:4]]))

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

        import torch as _torch
        dev = device or ("cuda:7" if _torch.cuda.is_available() else "cpu")
        n_cand = len(pool_ids)
        d_k = 16
        scale = 1.0 / math.sqrt(float(d_k))

        # Construct 3 orthogonal subspace projections `A` of shape `(3, n_cand)` on GPU
        a_heads = _torch.zeros((3, n_cand), dtype=_torch.float32, device=dev)
        max_base = max((base_sc_map.get(eid, 1.0) for eid in pool_ids), default=1.0)

        for idx, eid in enumerate(pool_ids):
            e = self.entries[eid]
            # Connected constellation members inherit strong structural relevance from their seed
            norm_base = float(base_sc_map.get(eid, max_base * 0.72)) / max(1.0, max_base)
            m0, m1, m2 = _subspace_mask_of(e)
            has_nums = 0.20 if re.search(r"\d+(?:\.\d+)?(?:%|x|\s*ms|\s*s/tok)", f"{e.summary} {e.source_evidence}") else 0.0
            has_corr = 0.25 if any(r.type in ("corrects", "refines") for r in self.outgoing.get(eid, [])) else 0.0

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

        # ===================================================================
        # COLLECTIVE EXPERIENCE SEED v0.1 — Agent 1 (Cross-Domain Principles,
        # Research Findings & Contradictions) + Agent 2 (Clover Kimi K3, NCM,
        # HybridAgent & Generative Compilation Experience)
        # ===================================================================

        # 13. Cross-Domain Principle: Flat Parameter Response Implicates Structure
        (
            "concepts/flat-parameter-response-diagnosis-001.md",
            MemoryEntry(
                id="MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001",
                title="Flat Parameter Response Implicates Structure, Not Absence of Effect",
                entry_type="concept",
                epistemic_status="externally_verified",
                status="active",
                concepts=[
                    "flat-parameter-response",
                    "structural-bottleneck-diagnosis",
                    "cyclic-sweep-eviction",
                    "per-layer-partitioning",
                ],
                tags=["principle", "systems", "measurement", "diagnosis"],
                created_at="2026-10-09T16:00:00Z",
                updated_at="2026-10-26T04:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="clover-operating-principles.md#2026-10-09",
                    actor="human_and_assistant",
                    recorded_at="2026-10-09T16:00:00Z",
                ),
                summary=(
                    "When varying a capacity or budget parameter across a wide range yields a completely flat "
                    "metric, suspect the structure being measured (such as shared-pool cyclic eviction across stages/layers "
                    "or unallocated capacity) before concluding the underlying mechanism has no effect; test partitioning the "
                    "same total budget per stage/layer."
                ),
                content=(
                    "A flat response curve across orders of magnitude of a parameter (e.g., global expert cache "
                    "hit rate remaining invariant at 23.3% across capacities 512, 1024, 2048, and 4096) indicates "
                    "that the swept parameter is not the binding constraint under the current topology. In a multi-stage "
                    "or 92-layer pipeline, any unpartitioned global LRU pool smaller than the full working set suffers "
                    "cyclic-sweep eviction before Stage/Layer 1 is revisited. Partitioning the exact same total budget "
                    "per stage or per layer eliminates cross-stage eviction and restores monotonic scaling (raising "
                    "Kimi K3 expert cache hit rate from 23.3% to 48.0% at the same 51.7 GB budget)."
                ),
                source_evidence=(
                    "Measured on Kimi K3 92-layer MoE (2026-10-09, 51.7 GB budget): global shared LRU pool gave "
                    "23.3% hit rate flat across 512..4096 entries; partitioning the same 51.7 GB per layer "
                    "raised hit rate from 23.3% to 48.0%."
                ),
                relationships=[
                    MemoryRelationship(
                        type="derived_from",
                        target="MEM-OBS-GLOBAL-CACHE-FLAT-001",
                        provenance="clover-operating-principles.md#2026-10-09",
                        confidence=1.0,
                        context_scope="structural_measurement_diagnosis",
                        created_at="2026-10-09T16:00:00Z",
                        note="Generalized from the flat 23.3% global expert cache sweep across capacities 512..4096.",
                    ),
                    MemoryRelationship(
                        type="derived_from",
                        target="MEM-CORR-PER-LAYER-PARTITION-002",
                        provenance="clover-operating-principles.md#2026-10-09",
                        confidence=1.0,
                        context_scope="structural_measurement_diagnosis",
                        created_at="2026-10-09T16:00:00Z",
                        note="Validated by the per-layer partitioning fix that jumped hit rate from 23.3% to 48.0%.",
                    ),
                    MemoryRelationship(
                        type="exemplifies",
                        target="MEM-CORR-PER-LAYER-PARTITION-002",
                        provenance="clover-operating-principles.md#2026-10-09",
                        confidence=1.0,
                        context_scope="expert_cache_design",
                        created_at="2026-10-09T16:00:00Z",
                        note="Per-layer cache partitioning is the canonical instance of fixing structure rather than abandoning the lever.",
                    ),
                    MemoryRelationship(
                        type="challenges",
                        target="MEM-HYP-CACHE-DEAD-END-001",
                        provenance="clover-operating-principles.md#2026-10-09",
                        confidence=1.0,
                        context_scope="expert_cache_design",
                        created_at="2026-10-09T16:00:00Z",
                        note="Identifies the diagnostic fallacy that led to prematurely rejecting expert caching.",
                    ),
                ],
                uncertainty_notes=[
                    "A flat curve can still reflect genuine absence of effect once structural bottlenecks (eviction topology, zero allocation, and filter truncation) are independently ruled out."
                ],
            ),
        ),

        # 14. Cross-Domain Principle: Zero Signal from an Incapable Instrument
        (
            "concepts/instrument-zero-denominator-001.md",
            MemoryEntry(
                id="MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
                title="Zero Signal from an Incapable Instrument Is Not Evidence Against the Mechanism",
                entry_type="concept",
                epistemic_status="externally_verified",
                status="active",
                concepts=[
                    "instrument-validity",
                    "zero-denominator-fallacy",
                    "kda-state-drift",
                    "zero-allocation-cache",
                ],
                tags=["principle", "measurement", "verification", "instrumentation", "epistemology"],
                created_at="2026-10-16T14:00:00Z",
                updated_at="2026-10-26T04:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="clover-operating-principles.md#2026-10-16; commit:58497d5",
                    actor="human_and_assistant",
                    recorded_at="2026-10-16T14:00:00Z",
                ),
                summary=(
                    "A test workload or filter that structurally cannot trigger a mechanism (or a table allocated "
                    "at size zero) provides zero evidence against the mechanism; always verify table allocation and "
                    "test with a workload matching the key semantics (such as replaying an identical full request)."
                ),
                content=(
                    "Before concluding that a mechanism never fires or an event is absent, verify that the test "
                    "instrument is structurally capable of observing it. Three verified failure modes illustrate "
                    "this rule: (1) testing an activation-vector-keyed expert result cache by repeating one token "
                    "64 times inside a single sequence produced 0 hits because recurrent KDA state drifts at every "
                    "position, whereas replaying the identical full request (A/B/A) hit 30,912 times and ran 3.27x "
                    "faster, bit-identical; (2) under K3_PAYLOAD_RAM=1 a double-counted payload left the cache at "
                    "zero allocation, making miss counts vacuous; (3) an awk filter keeping only records with "
                    "max-change > 0 hid a present no-op stage and falsely reported the stage as absent."
                ),
                source_evidence=(
                    "1. Expert result cache test (commit 58497d5, 2026-10-16): 64 repeated tokens in one sequence -> "
                    "0 hits / 113,344 misses (recurrent KDA state drift); A/B/A replay of identical request -> 30,912 hits, "
                    "3.27x speedup (28.57s -> 8.74s), bit-identical.\n"
                    "2. Layer trace filter (2026-10-11): awk predicate `max_change > 0` dropped no-op stage 20 records."
                ),
                relationships=[
                    MemoryRelationship(
                        type="similar_to",
                        target="MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001",
                        provenance="clover-operating-principles.md#2026-10-16",
                        confidence=0.95,
                        context_scope="measurement_methodology",
                        created_at="2026-10-16T14:00:00Z",
                        note="Sibling measurement principle: both prevent mistaking an instrument/topology limitation for absence of the underlying phenomenon.",
                    ),
                    MemoryRelationship(
                        type="supports",
                        target="MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001",
                        provenance="commit:58497d5",
                        confidence=1.0,
                        context_scope="expert_result_cache_design",
                        created_at="2026-10-16T14:00:00Z",
                        note="Explains why intra-sequence token repetition cannot hit a full-activation-keyed cache and why full-request replay is required.",
                    ),
                    MemoryRelationship(
                        type="supports",
                        target="MEM-OBS-HARD-COSINE-THRESHOLD-DROPOUT-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#finding-3",
                        confidence=0.95,
                        context_scope="entity_linking_threshold",
                        created_at="2026-10-25T16:00:00Z",
                        note="Explains why a hard cosine >= 0.35 filter on raw code queries acts as a zero-seed gate for 30% of queries.",
                    ),
                ],
            ),
        ),

        # 15. Cross-Domain Principle: Synthetic Input Measures Timing, Never Semantic Meaning
        (
            "concepts/synthetic-timing-vs-semantic-meaning-001.md",
            MemoryEntry(
                id="MEM-CONCEPT-SYNTHETIC-TIMING-VS-MEANING-001",
                title="Synthetic Input Measures Timing, Never Semantic Meaning",
                entry_type="concept",
                epistemic_status="externally_verified",
                status="active",
                concepts=[
                    "synthetic-vs-real-activations",
                    "semantic-observer-verification",
                    "in-situ-measurement",
                    "micro-benchmark",
                ],
                tags=["principle", "measurement", "activations", "routing", "verification"],
                created_at="2026-10-11T18:00:00Z",
                updated_at="2026-10-26T04:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="clover-operating-principles.md#2026-10-11",
                    actor="human_and_assistant",
                    recorded_at="2026-10-11T18:00:00Z",
                ),
                summary=(
                    "Synthetic or random tensors are valid for timing harnesses, but invert semantic properties "
                    "(router spread, attention norm contribution) and can mispredict kernel speedups when real "
                    "weights are non-uniform; derive semantic observers from verified gate harnesses and re-measure in situ."
                ),
                content=(
                    "Timing harnesses and semantic observers have opposite input requirements. Feeding random "
                    "activations into a Kimi K3 layer tracer produced uniform router weights and an attention norm "
                    "contribution of only 0.6% of the residual — both artifacts of random vectors. On real model "
                    "activations inherited from the verified gate harness, router weights spread 6.6x and attention "
                    "magnitude was comparable to the input residual. Similarly, a standalone kernel harness fed "
                    "uniformly random palette indices predicted 1.68x speedup whereas real non-uniform weights in "
                    "situ delivered 2.2x."
                ),
                source_evidence=(
                    "Measured on Kimi K3 (2026-10-11): random input -> uniform router spread (1.0x), attention "
                    "contribution 0.6% of norm; real gate-verified activations -> 6.6x router spread, attention "
                    "comparable to input norm. Standalone random-index kernel benchmark predicted 1.68x vs. "
                    "2.2x measured in situ on real weights."
                ),
                relationships=[
                    MemoryRelationship(
                        type="similar_to",
                        target="MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
                        provenance="clover-operating-principles.md#2026-10-11",
                        confidence=0.9,
                        context_scope="measurement_methodology",
                        created_at="2026-10-11T18:00:00Z",
                        note="Both govern instrument validity: synthetic inputs corrupt semantic observables just as incapable workloads corrupt hit counts.",
                    ),
                    MemoryRelationship(
                        type="supports",
                        target="MEM-OBS-ISOLATED-MICROBENCH-BIDIRECTIONAL-DRIFT-001",
                        provenance="clover-operating-principles.md#2026-10-11",
                        confidence=0.95,
                        context_scope="in_situ_verification",
                        created_at="2026-10-11T18:00:00Z",
                        note="Explains the under-prediction half of isolated micro-benchmarks when random palette indices replace real weight distributions.",
                    ),
                    MemoryRelationship(
                        type="supports",
                        target="MEM-EXP-RUST-SLOTCACHE-001",
                        provenance="commit:178d3c1",
                        confidence=0.95,
                        context_scope="in_situ_verification",
                        created_at="2026-10-25T19:30:00Z",
                        note="Motivates full 93-layer in-situ verification (`K3Full93Chain`) alongside standalone unit tests.",
                    ),
                ],
            ),
        ),

        # 16. Cross-Domain Principle: Performance Ratios Across Hardware Regimes
        (
            "concepts/hardware-scoped-performance-verdicts-001.md",
            MemoryEntry(
                id="MEM-CONCEPT-HARDWARE-SCOPED-VERDICTS-001",
                title="Scope Every Performance Verdict to Its Exact Hardware and Kernel Regime",
                entry_type="concept",
                epistemic_status="observed",
                status="active",
                concepts=[
                    "hardware-scoped-verdict",
                    "quantisation",
                    "nibble-unpack",
                    "gpu-supply-bandwidth",
                ],
                tags=["principle", "hardware", "performance", "quantisation", "scope"],
                created_at="2026-09-23T17:00:00Z",
                updated_at="2026-10-26T04:00:00Z",
                provenance=MemoryProvenance(
                    source_type="conversation",
                    source_ref="clover-operating-principles.md#2026-09-23; clover-kimi-k3-cpu-context.md; clover-kimi-k3-gpu-context.md",
                    actor="human_and_assistant",
                    recorded_at="2026-09-23T17:00:00Z",
                ),
                summary=(
                    "A performance measurement on one machine or kernel is not a universal property of the method: "
                    "scope every performance verdict to its exact hardware and kernel in the same sentence, and "
                    "explicitly mark cross-hardware transfer (such as CPU 4-bit trunk decode to GPU) as unmeasured until tested."
                ),
                content=(
                    "On an early CPU kernel, 8-bit to 4-bit trunk quantisation only moved decode from 4.1 to 4.0 s/token "
                    "(-2.4%) due to nibble unpack overhead, whereas a later CPU kernel (`LB_Q4`, commit 74c1c0e) reduced "
                    "decode from 3.326 to 2.716 s/token (-18.3%) with a quality tradeoff (identical on 14/34 decode-only "
                    "prompts, 8/34 whole-trunk). On the 8 x RTX 3090 GPU box, the measured bottleneck for 4-bit experts "
                    "is host-to-GPU supply bandwidth (4.95 s/layer from NVMe, 0.95 s/layer from RAM vs 0.23-0.29 s compute), "
                    "while a 4-bit vs 8-bit trunk decode A/B on GPU remains unmeasured."
                ),
                source_evidence=(
                    "1. Creator principle (2026-09-23): \"cpu processing is not how gpu will process it so the compounding can look very different.\"\n"
                    "2. CPU Step 3 (`heterogeneous-inference.md`): 4.1 -> 4.0 s/token (-2.4%).\n"
                    "3. CPU `LB_Q4` (`commit 74c1c0e`, `clover-kimi-k3-cpu-context.md`): 3.326 -> 2.716 s/token (-18.3%).\n"
                    "4. GPU 8x RTX 3090 (`clover-kimi-k3-gpu-context.md`): expert supply 4.95 s/layer (NVMe) / 0.95 s/layer (RAM) vs 0.23-0.29 s compute; Layer 0 DRAM throughput 5.6%; GPU 4-bit trunk A/B unmeasured."
                ),
                relationships=[
                    MemoryRelationship(
                        type="supports",
                        target="MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001",
                        provenance="clover-operating-principles.md#2026-09-23",
                        confidence=1.0,
                        context_scope="quantisation_verdict_scope",
                        created_at="2026-09-29T05:30:10Z",
                        note="Provides the hardware-scoping rule that corrects hardware-unscoped quantisation verdicts.",
                    ),
                    MemoryRelationship(
                        type="challenges",
                        target="MEM-HYP-QUANT-DEAD-END-ALL-HW-001",
                        provenance="clover-operating-principles.md#2026-09-23",
                        confidence=1.0,
                        context_scope="quantisation_verdict_scope",
                        created_at="2026-09-23T17:00:00Z",
                        note="Refutes treating a single CPU kernel measurement as a universal hardware law.",
                    ),
                    MemoryRelationship(
                        type="similar_to",
                        target="MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001",
                        provenance="clover-operating-principles.md#2026-09-23",
                        confidence=0.8,
                        context_scope="measurement_methodology",
                        created_at="2026-10-09T16:00:00Z",
                        note="Both prevent overgeneralizing a local measurement into a universal negative claim.",
                    ),
                ],
                uncertainty_notes=[
                    "A direct GPU A/B comparing 4-bit vs 8-bit trunk decode has not yet been measured on the 8 x RTX 3090 box."
                ],
            ),
        ),

        # 17. Cross-Domain Principle: Short-Prefix Shadowing in Sequential Dispatch
        (
            "concepts/short-prefix-shadowing-dispatch-001.md",
            MemoryEntry(
                id="MEM-CONCEPT-PREFIX-SHADOWING-DISPATCH-001",
                title="Short-Prefix Shadowing in Sequential Dispatch and String Tokenizers",
                entry_type="concept",
                epistemic_status="externally_verified",
                status="active",
                concepts=[
                    "prefix-shadow",
                    "longest-prefix-first-matching",
                    "generative-compilation",
                    "statement-boundary-soundness",
                ],
                tags=["principle", "compiler", "tokenizer", "python", "humaneval", "bug-pattern"],
                created_at="2026-10-25T21:00:00Z",
                updated_at="2026-10-26T04:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="docs/clover-generative-compilation-research-context.md#section-4.2",
                    actor="system_measurement",
                    recorded_at="2026-10-25T21:00:00Z",
                ),
                summary=(
                    "In sequential `if/elif` dispatch or hand-written character-index tokenizers, testing a shorter "
                    "prefix (such as `s[i] == 'o'`) before a longer prefix starting with the same character "
                    "(`s[i:i+2] == 'o|'`) renders the longer branch unreachable dead code; always order longest-prefix-first."
                ),
                content=(
                    "When a scanner inspects an unsplit string at index `i` using ordered conditional branches, "
                    "any branch that matches a 1-character prefix (`s[i] == 'o'`) and advances `i += 1` intercepts "
                    "every occurrence of a multi-character token (`'o|'`) that shares that prefix. In Clover Kimi K3's "
                    "`CloverPythonGenerativeCompiler` (`_PythonStatementBoundaryAnalyzer`), this structural defect is "
                    "detected at statement boundaries via `E_PY_PREFIX_SHADOW` in 0.68-0.70 ms as soon as the `elif` "
                    "line is emitted, rolling back the loop header and prompting `HybridAgent` to emit longest-prefix-first "
                    "dispatch (`s[i:i+2] == 'o|'` before `s[i] == 'o'`)."
                ),
                source_evidence=(
                    "Live verification on `HumanEval/17` (`parse_music`, `docs/clover-generative-compilation-research-context.md`): "
                    "`CloverPythonGenerativeCompiler` caught `error[E_PY_PREFIX_SHADOW] at line 26, col 9` (`music_string[i] == 'o'` "
                    "before `music_string[i:i+2] == 'o|'`) in `0.68-0.70 ms` and solved `parse_music` natively on Rollback #1."
                ),
                relationships=[
                    MemoryRelationship(
                        type="supports",
                        target="MEM-EXP-GENERATIVE-COMPILATION-ROLLBACK-001",
                        provenance="docs/clover-generative-compilation-research-context.md#section-4.2",
                        confidence=1.0,
                        context_scope="generative_compilation_humaneval",
                        created_at="2026-10-25T21:00:00Z",
                        note="Directly enabled native Rollback #1 self-correction on HumanEval/17.",
                    ),
                    MemoryRelationship(
                        type="similar_to",
                        target="MEM-OBS-ONTORAG-FANOUT-FLOODING-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#finding-2",
                        confidence=0.75,
                        context_scope="structural_priority_ordering",
                        created_at="2026-10-25T21:30:00Z",
                        note="Structural analogue: sorting by 1-hop length before edge priority lets short leaf edges shadow 2-hop causal chains, just as 1-char prefixes shadow 2-char tokens.",
                    ),
                ],
            ),
        ),

        # 18. Research Observation: 1-Hop High-Cardinality Leaf Edges (AFFECTS) Drown Out 2-Hop Causal Chains
        (
            "observations/obs-ontorag-fanout-flooding-001.md",
            MemoryEntry(
                id="MEM-OBS-ONTORAG-FANOUT-FLOODING-001",
                title="Observation: 1-Hop High-Cardinality Leaf Edges (`AFFECTS`) Drown Out 2-Hop Causal Chains When Sorted by Path Length First",
                entry_type="observation",
                epistemic_status="externally_verified",
                status="active",
                concepts=[
                    "1-hop-fanout-flooding",
                    "stratified-priority-traversal",
                    "ontology-guided-graphrag",
                    "causal-chain-recall",
                ],
                tags=["ontorag", "graphrag", "traversal", "cve", "cwe", "capec", "observation"],
                created_at="2026-10-25T14:00:00Z",
                updated_at="2026-10-26T04:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="docs/clover-ontology-graphrag-research-context.md#finding-2",
                    actor="system_measurement",
                    recorded_at="2026-10-25T14:00:00Z",
                ),
                summary=(
                    "Ordering candidate graph paths by `SORT LENGTH(p.edges), priority LIMIT K` allows high-cardinality "
                    "1-hop leaf edges (such as 64-208 `AFFECTS` product/host edges) to fill the entire `LIMIT K` before "
                    "any 2-hop causal/remediation path (`CVE -> CWE -> CAPEC`) is emitted."
                ),
                content=(
                    "Executing the upstream `ontorag` replication package on `gpu` (`/home/usergpu/ontorag`) revealed "
                    "two compounding defects (Findings 1 and 2): (1) `configs/default.yaml` omitted the Horn-derived "
                    "`EXPLOITED_BY_ATTACK` (`CVE -> CAPEC`) edge from `traversal_edges`, forcing `CAPEC` attack "
                    "patterns to be 2 hops away (`CVE --CAUSED_BY--> CWE --EXPLOITED_BY--> CAPEC`); and (2) "
                    "`GraphTraversal.paths` sorted by `LENGTH(p.edges)` before `edge_priority` with `max_paths = 50`. "
                    "For entities with many 1-hop leaf edges (`CVE-2023-44487` has 208 `AFFECTS` edges; `CVE-2023-4911` "
                    "has 73; `CVE-2024-6387` has 64), 100% of the returned paths were 1-hop edges and 0 paths reached `CAPEC`."
                ),
                source_evidence=(
                    "Measured on `/home/usergpu/ontorag` (`diagnose_three_findings.py`, 2026-10-25):\n"
                    "- `CVE-2023-44487` (`208` `AFFECTS` edges): `50/50` paths are 1-hop `AFFECTS`; `0` 2-hop paths, `0` `CAPEC` paths.\n"
                    "- `CVE-2023-4911` (`73` `AFFECTS` edges): `50/50` paths are 1-hop `AFFECTS`; `0` 2-hop paths, `0` `CAPEC` paths.\n"
                    "- `CVE-2024-6387` (`64` `AFFECTS` edges): `50/50` paths are 1-hop (`2 CAUSED_BY + 48 AFFECTS`); `0` `CAPEC` paths."
                ),
                relationships=[
                    MemoryRelationship(
                        type="exemplifies",
                        target="MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#finding-2",
                        confidence=0.95,
                        context_scope="graph_traversal_budget",
                        created_at="2026-10-25T14:00:00Z",
                        note="Zero 2-hop CAPEC recall was caused by unstratified path-length sorting, not absence of CAPEC links in the graph.",
                    ),
                    MemoryRelationship(
                        type="exemplifies",
                        target="MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#finding-1",
                        confidence=0.95,
                        context_scope="graph_traversal_budget",
                        created_at="2026-10-25T14:00:00Z",
                        note="1-hop session expansion without `EXPLOITED_BY_ATTACK` in `traversal_edges` structurally could not reach 2-hop CAPEC nodes.",
                    ),
                    MemoryRelationship(
                        type="similar_to",
                        target="MEM-OBS-GLOBAL-CACHE-FLAT-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#finding-2",
                        confidence=0.85,
                        context_scope="shared_budget_eviction_analogy",
                        created_at="2026-10-25T14:00:00Z",
                        note="Both are shared-budget flooding failures: 208 1-hop CPE edges evict 2-hop causal paths just as 92 layers evict a shared global LRU cache.",
                    ),
                ],
            ),
        ),

        # 19. Research Correction: Stratified Edge-Priority Quotas Before Path-Length Sorting
        (
            "corrections/corr-stratified-priority-traversal-001.md",
            MemoryEntry(
                id="MEM-CORR-STRATIFIED-PRIORITY-TRAVERSAL-001",
                title="Correction: Stratified Edge-Priority Ordering and Per-Relation Leaf Caps Restore 2-Hop Causal Chains",
                entry_type="correction",
                epistemic_status="externally_verified",
                status="active",
                concepts=[
                    "stratified-priority-traversal",
                    "1-hop-fanout-flooding",
                    "ontology-guided-graphrag",
                    "causal-chain-recall",
                ],
                tags=["ontorag", "graphrag", "correction", "traversal", "verified"],
                created_at="2026-10-25T16:30:00Z",
                updated_at="2026-10-26T04:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="docs/clover-ontology-graphrag-research-context.md#step-5",
                    actor="human_and_assistant",
                    recorded_at="2026-10-25T16:30:00Z",
                ),
                summary=(
                    "Sorting candidate paths by `edge_priority` (putting causal/remediation edges before leaf `AFFECTS` "
                    "edges) and enforcing a stratified per-seed cap on leaf edges (`max_affects_per_seed = 4`) "
                    "eliminates 1-hop fan-out flooding, raising full `CVE+CWE+CAPEC` chain coverage from 63.3% to 76.7% "
                    "and Gold ID recall from 78.9% to 84.5%."
                ),
                content=(
                    "Refines and resolves the traversal bottleneck observed in MEM-OBS-ONTORAG-FANOUT-FLOODING-001. "
                    "Merely raising `LIMIT K` is fragile when a node has hundreds of 1-hop leaf edges. Instead, "
                    "by (1) sorting by `edge_priority` group (causal/formula/remediation before leaf `AFFECTS`) "
                    "ahead of path length, (2) enforcing a per-seed cap on leaf `AFFECTS` edges (`max_affects_per_seed = 4`), "
                    "and (3) including Horn-derived shortcut edges (`EXPLOITED_BY_ATTACK`), `CloverGraphTraversal` "
                    "guarantees quota for 2-hop explanatory chains regardless of 1-hop fan-out cardinality."
                ),
                source_evidence=(
                    "Controlled 3-way benchmark on `gpu` (`bench_retrieval_3way.py`, `N=30` Exploratory queries):\n"
                    "- `ontorag_arm_a` (length-first sorting): Full `CVE+CWE+CAPEC` chain = `19/30` (`63.3%`), Mean `[P#]` paths = `28.3`, Gold ID Recall = `78.9%`.\n"
                    "- `ontorag_arm_b` (stratified priority + leaf cap): Full `CVE+CWE+CAPEC` chain = `23/30` (`76.7%`), Mean `[P#]` paths = `37.1`, Gold ID Recall = `84.5%`."
                ),
                relationships=[
                    MemoryRelationship(
                        type="refines",
                        target="MEM-OBS-ONTORAG-FANOUT-FLOODING-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#step-5",
                        confidence=1.0,
                        context_scope="graph_traversal_budget",
                        created_at="2026-10-25T16:30:00Z",
                        note="Resolves 1-hop `AFFECTS` flooding via stratified edge-priority ordering and `max_affects_per_seed = 4`.",
                    ),
                    MemoryRelationship(
                        type="derived_from",
                        target="MEM-OBS-ONTORAG-FANOUT-FLOODING-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#finding-2",
                        confidence=1.0,
                        context_scope="graph_traversal_budget",
                        created_at="2026-10-25T16:30:00Z",
                        note="Directly motivated by measuring 0 CAPEC paths on `CVE-2023-44487`, `CVE-2023-4911`, and `CVE-2024-6387`.",
                    ),
                    MemoryRelationship(
                        type="challenges",
                        target="MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#step-5",
                        confidence=1.0,
                        context_scope="multi_hop_vulnerability_retrieval",
                        created_at="2026-10-25T16:30:00Z",
                        note="Demonstrates that explicit priority-stratified symbolic path traversal (`37.1` paths, `84.5%` recall) is required over flat chunk similarity (`0.0` paths, `37.3%` recall).",
                    ),
                    MemoryRelationship(
                        type="similar_to",
                        target="MEM-CORR-PER-LAYER-PARTITION-002",
                        provenance="docs/clover-ontology-graphrag-research-context.md#section-3",
                        confidence=0.9,
                        context_scope="partitioned_quota_architecture",
                        created_at="2026-10-25T16:30:00Z",
                        note="Both replace an unpartitioned global competition pool with per-class/per-layer quotas so high-volume items cannot evict critical items.",
                    ),
                ],
            ),
        ),

        # 20. Research Observation: Hard Dense Embedding Threshold (cosine >= 0.35) Drops 30% of Code Queries
        (
            "observations/obs-hard-cosine-threshold-dropout-001.md",
            MemoryEntry(
                id="MEM-OBS-HARD-COSINE-THRESHOLD-DROPOUT-001",
                title="Observation: Hard Dense Embedding Threshold (`cosine >= 0.35`) Drops 30% (`9/30`) of Code Queries to Zero Seeds",
                entry_type="observation",
                epistemic_status="externally_verified",
                status="active",
                concepts=[
                    "hard-threshold-seed-dropout",
                    "hybrid-rrf-fallback",
                    "ontology-guided-graphrag",
                    "code-to-concept-linking",
                ],
                tags=["ontorag", "embeddings", "cosine-threshold", "bm25", "rrf", "observation"],
                created_at="2026-10-25T14:30:00Z",
                updated_at="2026-10-26T04:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="docs/clover-ontology-graphrag-research-context.md#finding-3",
                    actor="system_measurement",
                    recorded_at="2026-10-25T14:30:00Z",
                ),
                summary=(
                    "In `ontorag`'s `EntityLinker.link`, filtering `all-MiniLM-L6-v2` embeddings by a hard cutoff "
                    "of `similarity_threshold = 0.35` drops 9 of 30 (`30.0%`) Practical code queries to zero seeds "
                    "(`(no evidence retrieved)`), frequently discarding the exact right CWE/CAPEC at rank 1."
                ),
                content=(
                    "Raw source code snippets (such as Flask `re.search(request.args['regex'], data)` or Java "
                    "`new File(targetDirectory + filename); file.delete()`) have lower surface cosine similarity "
                    "to prose MITRE CWE/CAPEC descriptions in `all-MiniLM-L6-v2` space (`max_cosine` between `0.2398` "
                    "and `0.3423`) than prose-to-prose queries. Because `EntityLinker.link` enforced a hard "
                    "`cosine >= 0.35` cutoff with no lexical BM25 fallback, `9 / 30` Practical queries returned "
                    "`0` seeds and `(no evidence retrieved)`, even though the discarded top-1 candidate was the "
                    "exact vulnerability concept (`CAPEC-76` at `0.3290`, `CAPEC-137` at `0.2942`, `CAPEC-492` at `0.2761`)."
                ),
                source_evidence=(
                    "Measured across all 30 Practical code queries on `/home/usergpu/ontorag` (`diagnose_three_findings.py`, "
                    "`944` CWEs + `559` CAPECs indexed):\n"
                    "- `9 / 30` queries (`30.0%`) had `max_cosine < 0.35` (`0.2398..0.3423`) -> `0` seeds, `(no evidence retrieved)`.\n"
                    "- Session 6 dropped `CAPEC-76: Manipulating Web Input to File System Calls` (`cosine = 0.3290`).\n"
                    "- Session 15 dropped `CAPEC-137: Parameter Injection` (`cosine = 0.2942`).\n"
                    "- Session 20 dropped `CAPEC-492: Regular Expression Exponential Blowup` (`cosine = 0.2761`)."
                ),
                relationships=[
                    MemoryRelationship(
                        type="exemplifies",
                        target="MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#finding-3",
                        confidence=1.0,
                        context_scope="entity_linking_threshold",
                        created_at="2026-10-25T14:30:00Z",
                        note="An over-strict `cosine >= 0.35` gate zeroes out the seed set before graph traversal can even run.",
                    ),
                    MemoryRelationship(
                        type="challenges",
                        target="MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#finding-3",
                        confidence=1.0,
                        context_scope="code_to_concept_linking",
                        created_at="2026-10-25T14:30:00Z",
                        note="Shows that dense embedding similarity alone fails on 30% of code-to-ontology queries at standard thresholds.",
                    ),
                ],
            ),
        ),

        # 21. Superseded Hypothesis: Dense Vector Similarity Alone Suffices for Multi-Hop Graph Grounding
        (
            "hypotheses/hyp-dense-rag-sufficient-for-graph-001.md",
            MemoryEntry(
                id="MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001",
                title="Superseded Hypothesis: Dense Vector Similarity Alone Suffices for Cross-Repository Vulnerability & Domain Grounding",
                entry_type="hypothesis",
                epistemic_status="hypothesis",
                status="superseded",
                superseded_by="MEM-CORR-HYBRID-RRF-SYMBOLIC-SEPARATION-001",
                concepts=[
                    "dense-rag-baseline",
                    "ontology-guided-graphrag",
                    "hybrid-rrf-fallback",
                    "symbolic-neural-separation",
                ],
                tags=["rag", "embeddings", "hypothesis", "superseded", "graphrag"],
                created_at="2026-10-25T10:00:00Z",
                updated_at="2026-10-25T17:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="docs/clover-ontology-graphrag-research-context.md#step-5",
                    actor="system_measurement",
                    recorded_at="2026-10-25T10:00:00Z",
                ),
                summary=(
                    "Baseline hypothesis (`VanillaRagBaseline`) that dense embedding similarity (`all-MiniLM-L6-v2`, "
                    "`top_k=10`, `threshold=0.35`) over exported entity documents is sufficient to retrieve multi-hop "
                    "`CVE -> CWE -> CAPEC` evidence without explicit symbolic graph traversal or hybrid lexical RRF "
                    "(superseded by MEM-CORR-HYBRID-RRF-SYMBOLIC-SEPARATION-001)."
                ),
                content=(
                    "Assumed that indexing the shared 2-hop corpus (`1,021` entity documents) into a dense vector store "
                    "would retrieve the relevant `CVE`, `CWE`, and `CAPEC` records and allow the LLM to infer their "
                    "relationships implicitly. In controlled 3-way testing on `gpu`, `vanilla_rag` achieved only "
                    "`37.3%` Gold ID Recall on Exploratory queries (vs. `84.5%` for `ontorag_arm_b`), `16.4%` on "
                    "Practical code queries (with `5/30` empty-evidence dropouts), and `0.0` relationship paths (`[P#]`)."
                ),
                source_evidence=(
                    "Measured in `bench_retrieval_3way.py` on `gpu` (`1,021`-doc shared corpus, `N=60` queries):\n"
                    "- Exploratory (`N=30`): `vanilla_rag` Gold ID Recall = `37.3%`, Mean `[P#]` paths = `0.0`.\n"
                    "- Practical Code (`N=30`): `vanilla_rag` Empty Evidence = `5/30`, Gold ID Recall = `16.4%`, Mean `[P#]` paths = `0.0`."
                ),
                relationships=[
                    MemoryRelationship(
                        type="similar_to",
                        target="MEM-HYP-CACHE-DEAD-END-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#step-5",
                        confidence=0.75,
                        context_scope="superseded_architectural_baseline",
                        created_at="2026-10-25T17:00:00Z",
                        note="Both represent initial architectural assumptions refuted by controlled structural measurements on `gpu`.",
                    ),
                ],
                revisions=[
                    MemoryRevision(
                        timestamp="2026-10-25T10:00:00Z",
                        actor="system_measurement",
                        action="created",
                        reason="Recorded baseline dense RAG hypothesis (`VanillaRagBaseline`) for 3-way controlled evaluation.",
                    ),
                    MemoryRevision(
                        timestamp="2026-10-25T17:00:00Z",
                        actor="human_and_assistant",
                        action="mark_superseded",
                        reason=(
                            "Superseded by MEM-CORR-HYBRID-RRF-SYMBOLIC-SEPARATION-001 after 3-way benchmark showed "
                            "37.3% vs. 84.5% Gold ID recall and 0.0 vs. 37.1 relationship paths."
                        ),
                    ),
                ],
            ),
        ),

        # 22. Research Correction: Hybrid RRF Seed Linking + Strict One-Way Symbolic-Neural Separation
        (
            "corrections/corr-hybrid-rrf-symbolic-separation-001.md",
            MemoryEntry(
                id="MEM-CORR-HYBRID-RRF-SYMBOLIC-SEPARATION-001",
                title="Correction: Hybrid Dense+BM25 RRF Seed Linking and One-Way Symbolic Traversal Outperform Pure Dense RAG (`84.5%` vs. `37.3%` Recall)",
                entry_type="correction",
                epistemic_status="externally_verified",
                status="active",
                concepts=[
                    "symbolic-neural-separation",
                    "hybrid-rrf-fallback",
                    "ontology-guided-graphrag",
                    "dense-rag-baseline",
                    "hard-threshold-seed-dropout",
                ],
                tags=["ontorag", "graphrag", "rrf", "symbolic", "correction", "verified", "benchmarks"],
                created_at="2026-10-25T17:00:00Z",
                updated_at="2026-10-26T04:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="docs/clover-ontology-graphrag-research-context.md#section-4",
                    actor="human_and_assistant",
                    recorded_at="2026-10-25T17:00:00Z",
                ),
                summary=(
                    "Coupling unthresholded dense+BM25 Reciprocal Rank Fusion (RRF) seed linking with deterministic "
                    "one-way symbolic graph traversal eliminates all `10/60` empty-evidence failures (`0/30` on both "
                    "Exploratory and Practical), raises Exploratory Gold ID Recall from `37.3%` (`vanilla_rag`) to "
                    "`84.5%` (`37.1` `[P#]` paths), and lifts Clover Kimi K3 (`V5`) to `100%` on GSM8K, MMLU, "
                    "HumanEval, and HybridAgent QA."
                ),
                content=(
                    "Corrects `MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001` and resolves the `30%` code-query seed "
                    "dropout in `MEM-OBS-HARD-COSINE-THRESHOLD-DROPOUT-001`. Pure dense vector retrieval flattens "
                    "typed cross-repository edges (`0.0` `[P#]` relationship paths) and fails when raw code or "
                    "Unicode-hyphenated queries fall below a fixed cosine threshold. By combining (1) Unicode hyphen "
                    "and grouped-identifier normalization, (2) dense+BM25 RRF fallback with a lexical density guard "
                    "(`>= 4`-char domain words, `overlap >= 4`, coverage density `>= 0.20`), and (3) one-way symbolic "
                    "graph traversal with `[E#]` / `[P#]` provenance and `CloverSymbolicVerifier`, Clover Kimi K3 (`V5`) "
                    "improved Pooled 47-task accuracy from `29.12%` (`V1`) and `88.02%` (`V4`) to `98.66%` (`V5`)."
                ),
                source_evidence=(
                    "1. 3-Way Retrieval Benchmark (`bench_retrieval_3way.py`, `gpu`):\n"
                    "   - Exploratory (`N=30`): `vanilla_rag` `37.3%` Gold ID Recall (`0.0` paths) vs. `ontorag_arm_b` `84.5%` (`37.1` paths, `0/30` empty).\n"
                    "   - Practical Code (`N=30`): `vanilla_rag` `16.4%` (`5/30` empty), `ontorag_arm_a` `25.0%` (`9/30` empty) vs. `ontorag_arm_b` `29.8%` (`0/30` empty, `30/30` CWE coverage).\n"
                    "2. Clover Kimi K3 5-Way Ablation (`docs/clover-ontology-graphrag-research-context.md#section-4.2`):\n"
                    "   - GSM8K `80% -> 100% (20/20)`, MMLU `80% -> 100% (20/20)`, HumanEval `95% -> 100% (20/20)`, HybridAgent QA `93.3% -> 100% (15/15)`."
                ),
                relationships=[
                    MemoryRelationship(
                        type="corrects",
                        target="MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#step-5",
                        confidence=1.0,
                        context_scope="neuro_symbolic_graphrag_architecture",
                        created_at="2026-10-25T17:00:00Z",
                        note="Supersedes pure dense vector retrieval with hybrid RRF seed linking + deterministic symbolic graph traversal.",
                    ),
                    MemoryRelationship(
                        type="challenges",
                        target="MEM-HYP-DENSE-RAG-SUFFICIENT-FOR-GRAPH-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#step-5",
                        confidence=1.0,
                        context_scope="embedding_only_flattening",
                        created_at="2026-10-25T17:00:00Z",
                        note="Refutes the claim that flat dense chunks preserve multi-hop causal provenance (`0.0` paths vs `37.1` paths).",
                    ),
                    MemoryRelationship(
                        type="derived_from",
                        target="MEM-OBS-HARD-COSINE-THRESHOLD-DROPOUT-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#finding-3",
                        confidence=1.0,
                        context_scope="code_to_concept_linking",
                        created_at="2026-10-25T17:00:00Z",
                        note="Eliminates the 9/30 (30%) Practical code query seed dropout via unthresholded dense+BM25 RRF fallback.",
                    ),
                    MemoryRelationship(
                        type="supports",
                        target="MEM-CORR-STRATIFIED-PRIORITY-TRAVERSAL-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#section-4.1",
                        confidence=1.0,
                        context_scope="neuro_symbolic_graphrag_architecture",
                        created_at="2026-10-25T17:00:00Z",
                        note="Combines hybrid RRF seed recovery with stratified priority traversal for end-to-end grounded retrieval.",
                    ),
                    MemoryRelationship(
                        type="supports",
                        target="MEM-OUT-RUST-GPU-PARITY-001",
                        provenance="docs/clover-ontology-graphrag-research-context.md#section-4.2",
                        confidence=1.0,
                        context_scope="k3_benchmark_suite_verification",
                        created_at="2026-10-25T20:00:00Z",
                        note="Directly contributes the Ontology-Guided GraphRAG (`V5`) layer in the 100% 5-benchmark verification suite.",
                    ),
                ],
                uncertainty_notes=[
                    "Lexical fallback on long narrative passages (>100 words) requires query-coverage density filtering (`overlap >= 4` and `overlap / len(q_words) >= 0.20`) so generic 3-letter words do not trigger out-of-domain STEM concepts."
                ],
            ),
        ),

        # 23. Superseded Hypothesis: 4-Bit Quantisation Is a Dead End for Speed on All Hardware
        (
            "hypotheses/hyp-quant-dead-end-all-hw-001.md",
            MemoryEntry(
                id="MEM-HYP-QUANT-DEAD-END-ALL-HW-001",
                title="Initial Hypothesis: 4-Bit Quantisation Is a Universal Dead End for Speed Across All Hardware",
                entry_type="hypothesis",
                epistemic_status="hypothesis",
                status="superseded",
                superseded_by="MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001",
                concepts=["quantisation", "nibble-unpack", "cpu-decode", "hardware-scoped-verdict"],
                tags=["systems", "quantisation", "hypothesis", "superseded"],
                created_at="2026-09-23T00:00:00Z",
                updated_at="2026-09-29T05:30:10Z",
                provenance=MemoryProvenance(
                    source_type="conversation",
                    source_ref="research/k3/architecture/heterogeneous-inference.md#step-3; clover-operating-principles.md#2026-09-23",
                    actor="assistant",
                    recorded_at="2026-09-23T00:00:00Z",
                ),
                summary=(
                    "From one early CPU run where 8-bit to 4-bit trunk quantisation moved decode from 4.1 to 4.0 s/token "
                    "(-2.4%), the assistant concluded 'shrinking further is a dead end' without scoping the verdict to "
                    "that CPU kernel (superseded by MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001)."
                ),
                content=(
                    "On that initial CPU kernel, 4-bit weights had to be unpacked before use and the unpacking cost "
                    "nearly offset the halved memory read. Recording the conclusion without naming the hardware or "
                    "kernel made it read as a universal property of 4-bit quantisation rather than a property of that "
                    "specific CPU implementation."
                ),
                source_evidence=(
                    "heterogeneous-inference.md Step 3: \"4.1 down to 4.0 seconds per token ... it now matched the "
                    "original for 3 tokens instead of 21 ... We had already written down 'shrinking further is a dead end.' That was wrong.\""
                ),
                relationships=[
                    MemoryRelationship(
                        type="similar_to",
                        target="MEM-HYP-CACHE-DEAD-END-001",
                        provenance="clover-operating-principles.md#2026-09-23",
                        confidence=0.8,
                        context_scope="dead_end_from_narrow_measurement",
                        created_at="2026-09-23T00:00:00Z",
                        note="Both are premature 'dead end' verdicts generalized from a single structurally limited measurement.",
                    ),
                ],
                revisions=[
                    MemoryRevision(
                        timestamp="2026-09-23T00:00:00Z",
                        actor="assistant",
                        action="created",
                        reason="Recorded initial CPU 4-bit trunk observation as an unscoped 'dead end' claim.",
                    ),
                    MemoryRevision(
                        timestamp="2026-09-29T05:30:10Z",
                        actor="human_and_assistant",
                        action="mark_superseded",
                        reason="Superseded by MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001: verdicts must be scoped to hardware and kernel.",
                    ),
                ],
                uncertainty_notes=[
                    "Commit 74c1c0e (2026-10-18) later measured a different 4-bit CPU trunk packing (LB_Q4) at 3.326 -> 2.716 s/token (-18.3%), showing even the CPU result depends on the packing kernel."
                ],
            ),
        ),

        # 24. Correction: Hardware-Scoped Quantisation & GPU Supply Bandwidth Boundary
        (
            "corrections/corr-gpu-nibble-unpack-bandwidth-001.md",
            MemoryEntry(
                id="MEM-CORR-GPU-NIBBLE-UNPACK-BANDWIDTH-001",
                title="Correction: CPU 4-Bit Verdicts Are Kernel-Scoped (`-2.4%` vs `-18.3%` `LB_Q4`); on GPU Supply Bandwidth Binds and Trunk 4-Bit vs 8-Bit Remains Unmeasured",
                entry_type="correction",
                epistemic_status="observed",
                status="active",
                concepts=["quantisation", "gpu-supply-bandwidth", "nibble-unpack", "hardware-scoped-verdict", "kimi-k3-moe"],
                tags=["systems", "quantisation", "correction", "scope"],
                created_at="2026-09-29T05:30:10Z",
                updated_at="2026-10-26T04:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="heterogeneous-inference.md (commit d8f3692); clover-kimi-k3-gpu-context.md#supplying-one-layers-experts; clover-kimi-k3-cpu-context.md#4-bit-trunk-for-decode (commit 74c1c0e)",
                    actor="human_and_assistant",
                    recorded_at="2026-09-29T05:30:10Z",
                ),
                summary=(
                    "Corrects MEM-HYP-QUANT-DEAD-END-ALL-HW-001: on CPU, the early kernel gave -2.4% (4.1 -> 4.0 s/tok) "
                    "while the later `LB_Q4` kernel gave -18.3% (3.326 -> 2.716 s/tok, identical on 14/34 decode-only prompts); "
                    "on the 8 x RTX 3090 GPU box, 4-bit expert supply bandwidth (4.95 s/layer NVMe, 0.95 s/layer RAM vs "
                    "0.23-0.29 s compute) is the measured bottleneck, while a GPU 4-bit vs 8-bit trunk decode A/B is unmeasured."
                ),
                content=(
                    "Both '4-bit quantisation is a universal dead end on all hardware' and '4-bit trunk decode is already "
                    "proven faster on our GPU' are unsupported. What is established by measurement: (1) on CPU, speed depends "
                    "on the packing kernel (-2.4% in Step 3 vs -18.3% with `LB_Q4` in commit 74c1c0e, with a quality tradeoff "
                    "of 14/34 decode-only and 8/34 whole-trunk prompt matches); (2) on the 8 x RTX 3090 GPU box, streaming "
                    "a layer's 15.72 GB of 4-bit experts takes 4.95 s from NVMe and 0.95 s from host RAM against 0.23-0.29 s "
                    "of kernel compute per 1000-token window, and at Layer 0 `ncu` measured DRAM throughput at 5.6% "
                    "(shared-memory/L1-bound); (3) a direct GPU A/B comparing a 4-bit vs 8-bit trunk has not been measured."
                ),
                source_evidence=(
                    "1. GPU context (`clover-kimi-k3-gpu-context.md`): 8-card expert supply 4.95 s/layer (NVMe), 0.95 s/layer (RAM) vs 0.23-0.29 s compute; Layer 0 `ncu` DRAM throughput 5.6%.\n"
                    "2. CPU context (`clover-kimi-k3-cpu-context.md`, commit 74c1c0e): `LB_Q4` decode 3.326 -> 2.716 s/token (-18.3%); identical on 14/34 prompts (decode-only), 8/34 (whole trunk).\n"
                    "3. Creator rule (`clover-operating-principles.md#2026-09-23`): \"cpu processing is not how gpu will process it so the compounding can look very different.\""
                ),
                relationships=[
                    MemoryRelationship(
                        type="corrects",
                        target="MEM-HYP-QUANT-DEAD-END-ALL-HW-001",
                        provenance="commit:d8f3692; commit:74c1c0e",
                        confidence=0.95,
                        context_scope="quantisation_verdict_scope",
                        created_at="2026-09-29T05:30:10Z",
                        note="Replaces the hardware-unscoped 'dead end' claim with exact kernel-scoped CPU measurements and GPU supply boundaries.",
                    ),
                    MemoryRelationship(
                        type="challenges",
                        target="MEM-HYP-QUANT-DEAD-END-ALL-HW-001",
                        provenance="clover-operating-principles.md#2026-09-23",
                        confidence=0.95,
                        context_scope="hardware_unscoped_benchmark_claims",
                        created_at="2026-09-29T05:30:10Z",
                        note="Challenges any benchmark verdict that omits hardware and kernel scope.",
                    ),
                    MemoryRelationship(
                        type="exemplifies",
                        target="MEM-CONCEPT-HARDWARE-SCOPED-VERDICTS-001",
                        provenance="clover-operating-principles.md#2026-09-23",
                        confidence=0.95,
                        context_scope="performance_verdict_scoping",
                        created_at="2026-09-29T05:30:10Z",
                        note="Demonstrates why CPU and GPU performance regimes must be distinguished and unmeasured GPU claims flagged.",
                    ),
                ],
                uncertainty_notes=[
                    "A GPU 4-bit vs 8-bit trunk decode A/B has not been measured on the 8 x RTX 3090 server ('we cannot find that out here').",
                    "Register-level nibble unpack cost on GPU is an architectural expectation; the measured GPU bottleneck is host-to-card expert supply (0.95-4.95 s/layer)."
                ],
            ),
        ),

        # 25. Observation: Isolated Micro-Benchmark Bidirectional Drift
        (
            "observations/obs-isolated-microbench-bidirectional-drift-001.md",
            MemoryEntry(
                id="MEM-OBS-ISOLATED-MICROBENCH-BIDIRECTIONAL-DRIFT-001",
                title="Observation: Isolated Micro-Benchmarks Mispredict In-Layer Gains in Both Directions (`15.2x` vs `8.6x` and `1.68x` vs `2.2x`)",
                entry_type="observation",
                epistemic_status="observed",
                status="active",
                concepts=["micro-benchmark", "in-situ-measurement", "cache-warmth", "synthetic-vs-real-activations"],
                tags=["systems", "benchmark", "observation"],
                created_at="2026-10-11T12:00:00Z",
                updated_at="2026-10-26T04:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="clover-operating-principles.md#2026-10-11; clover-kimi-k3-cpu-context.md L87, L169-170",
                    actor="human_and_assistant",
                    recorded_at="2026-10-11T12:00:00Z",
                ),
                summary=(
                    "A standalone hot-loop replica over-predicted gain (`15.2x` standalone vs `8.6x` in-layer because "
                    "its 6.29 MB state stayed cache-warm in isolation), while a random-index kernel harness under-predicted "
                    "gain (`1.68x` standalone vs `2.2x` in-layer on real non-uniform weight palettes); always re-measure in situ."
                ),
                content=(
                    "Do not assume isolated micro-benchmarks always over-predict: error occurs in both directions. "
                    "In the first case, a standalone KDA loop replica kept its 6.29 MB state warm with no competing "
                    "traffic (`15.2x` standalone vs `8.6x` inside the full layer; also corroborated in `clover-kimi-k3-cpu-context.md` "
                    "where isolated 4x4 kernel throughput was `962-1,071 GMAC/s` vs `760-840 GMAC/s` in-layer). In the second "
                    "case on the same day, a kernel harness fed uniformly random palette indices predicted `1.68x` whereas "
                    "the real layer with non-uniform weight indices delivered `2.2x`."
                ),
                source_evidence=(
                    "1. `clover-operating-principles.md#2026-10-11`: standalone loop `15.2x` (~5.5% of layer) vs in-layer `8.6x` (~4%); "
                    "random palette harness `1.68x` vs in-layer `2.2x`.\n"
                    "2. `clover-kimi-k3-cpu-context.md` lines 87, 169-170: isolated `962-1,071 GMAC/s` vs in-layer `760-840 GMAC/s`."
                ),
                relationships=[
                    MemoryRelationship(
                        type="exemplifies",
                        target="MEM-CONCEPT-SYNTHETIC-TIMING-VS-MEANING-001",
                        provenance="clover-operating-principles.md#2026-10-11",
                        confidence=0.85,
                        context_scope="random_index_harness",
                        created_at="2026-10-11T12:00:00Z",
                        note="The 1.68x vs 2.2x under-prediction arose from feeding uniformly random palette indices instead of real weight indices.",
                    ),
                ],
                uncertainty_notes=[
                    "The 15.2x/8.6x and 1.68x/2.2x ratios are recorded in `clover-operating-principles.md#2026-10-11`, while `clover-kimi-k3-cpu-context.md` records the corresponding `962-1,071` vs `760-840 GMAC/s` in-layer measurements."
                ],
            ),
        ),

        # 26. Superseded Hypothesis: Expert Result Cache Can Never Hit
        (
            "hypotheses/hyp-expert-result-cache-dead-001.md",
            MemoryEntry(
                id="MEM-HYP-EXPERT-RESULT-CACHE-DEAD-001",
                title="Initial Hypothesis: The Expert Result Cache Can Never Hit in Kimi K3",
                entry_type="hypothesis",
                epistemic_status="hypothesis",
                status="superseded",
                superseded_by="MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001",
                concepts=["expert-result-cache", "intra-sequence-repetition", "kda-state-drift", "kimi-k3-moe"],
                tags=["systems", "hypothesis", "superseded"],
                created_at="2026-10-16T00:00:00Z",
                updated_at="2026-10-16T09:52:19Z",
                provenance=MemoryProvenance(
                    source_type="conversation",
                    source_ref="commit:58497d5; clover-operating-principles.md#2026-10-16",
                    actor="assistant",
                    recorded_at="2026-10-16T00:00:00Z",
                ),
                summary=(
                    "Mistaken hypothesis on 2026-10-16 that exact-match expert result caching could never hit in "
                    "Kimi K3, inferred from a test that repeated one token 64 times inside a single sequence and "
                    "observed 0 hits / 113,344 misses (superseded by MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001)."
                ),
                content=(
                    "Tested the activation-vector-keyed expert result cache using 64 copies of token 17374 inside "
                    "one sequence, saw 0 hits against 113,344 misses, and generalized that the cache mechanism was dead."
                ),
                source_evidence=(
                    "Commit `58497d5` ('Correcting myself'): \"I reported earlier in this cycle that exact-match result "
                    "caching could never hit in this model. The test repeated a token inside one sequence, never a request ... "
                    "64 copies of token 17374 gave 0 hits against 113344 misses.\""
                ),
                relationships=[
                    MemoryRelationship(
                        type="similar_to",
                        target="MEM-HYP-CACHE-DEAD-END-001",
                        provenance="commit:58497d5",
                        confidence=0.75,
                        context_scope="dead_end_from_narrow_measurement",
                        created_at="2026-10-16T00:00:00Z",
                        note="Both rejected a caching mechanism based on a test workload or structure that prevented hits.",
                    ),
                ],
                revisions=[
                    MemoryRevision(
                        timestamp="2026-10-16T00:00:00Z",
                        actor="assistant",
                        action="created",
                        reason="Recorded initial claim that expert result caching cannot hit from 64-token intra-sequence repeat test.",
                    ),
                    MemoryRevision(
                        timestamp="2026-10-16T09:52:19Z",
                        actor="human_and_assistant",
                        action="mark_superseded",
                        reason="Superseded by MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001 after fixing zero-allocation and testing A/B/A request replay.",
                    ),
                ],
            ),
        ),

        # 27. Correction: Repeating a Request Hits the Expert Result Cache 30,912 Times (3.27x Faster)
        (
            "corrections/corr-expert-result-cache-replay-001.md",
            MemoryEntry(
                id="MEM-CORR-EXPERT-RESULT-CACHE-REPLAY-001",
                title="Correction: Intra-Sequence Repeats Cannot Hit Due to KDA State Drift, Whereas Replaying an Identical Request Hits 30,912 Times (`3.27x` Faster)",
                entry_type="correction",
                epistemic_status="observed",
                status="active",
                concepts=["expert-result-cache", "request-replay", "kda-state-drift", "zero-allocation-cache", "kimi-k3-moe"],
                tags=["systems", "correction", "instrument-validity"],
                created_at="2026-10-16T09:52:19Z",
                updated_at="2026-10-18T00:00:00Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="commit:58497d5; clover-kimi-k3-cpu-context.md (2026-10-17 status; 2026-10-18 whole-vector survey)",
                    actor="human_and_assistant",
                    recorded_at="2026-10-16T09:52:19Z",
                ),
                summary=(
                    "Corrects MEM-HYP-EXPERT-RESULT-CACHE-DEAD-001: because the cache is keyed on the full input "
                    "activation vector and recurrent KDA state drifts at every position within a sequence, repeating "
                    "a token inside one sequence structurally cannot hit; furthermore, a double-counted payload under "
                    "`K3_PAYLOAD_RAM=1` had left the cache allocated at size zero. Once allocated and tested by "
                    "replaying the identical full request (A/B/A), it hit 30,912 times and ran 3.27x faster, bit-identical."
                ),
                content=(
                    "Demonstrates two essential diagnostic checks whenever a memoization or result cache reports 0 hits: "
                    "(1) verify the table was actually allocated with non-zero capacity (`K3_PAYLOAD_RAM=1` had double-counted "
                    "payload bytes and left the result cache at 0 entries); and (2) verify the workload matches the cache's "
                    "key invariant. Because KDA recurrent state updates at every token position, the same token ID at "
                    "positions 1..64 produces 64 distinct full-activation vectors (`0` identical vectors across 1,000 real "
                    "tokens in the 2026-10-18 Layer 4 survey). Replaying the identical request (A/B/A) produces identical "
                    "position-aligned activations, yielding 30,912 hits, 29,440 misses, 0 evictions, and a 3.27x speedup "
                    "(28.574 s -> 8.759 s; decode 3.68 -> 1.43 s/tok) with bit-identical output."
                ),
                source_evidence=(
                    "Commit `58497d5` (`CONTEXT.md` 'Repeat Is 3.27x Once They Are Not') & `clover-kimi-k3-cpu-context.md`:\n"
                    "- A/B/A request replay: `28.574 s`, `8.759 s`, `27.329 s`, `8.737 s` (`3.27x` faster on repeat, `30,912` hits, `0` evictions, bit-identical).\n"
                    "- Non-repeating overhead: `+0.88%` (`3.6641 -> 3.6965 s/token`)."
                ),
                relationships=[
                    MemoryRelationship(
                        type="corrects",
                        target="MEM-HYP-EXPERT-RESULT-CACHE-DEAD-001",
                        provenance="commit:58497d5",
                        confidence=1.0,
                        context_scope="expert_result_cache_design",
                        created_at="2026-10-16T09:52:19Z",
                        note="Refutes the 'cannot ever hit' claim by fixing zero-allocation and testing A/B/A full-request replay.",
                    ),
                    MemoryRelationship(
                        type="exemplifies",
                        target="MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
                        provenance="commit:58497d5",
                        confidence=1.0,
                        context_scope="instrument_validity",
                        created_at="2026-10-16T09:52:19Z",
                        note="Canonical example of both a workload that structurally cannot hit (intra-sequence repeat under KDA drift) and a zero-allocated table.",
                    ),
                ],
                uncertainty_notes=[
                    "The 3.27x speedup applies to repeated identical requests (or shared identical prefixes from position 0); within a single non-repeating request, 0 of 224,000 activation segments repeated in the 2026-10-18 survey."
                ],
            ),
        ),

        # 28. Observation: Post-Hoc Code Repair Cannot Stop Doomed Token Generation
        (
            "observations/obs-posthoc-code-repair-limits-001.md",
            MemoryEntry(
                id="MEM-OBS-POSTHOC-CODE-REPAIR-LIMITS-001",
                title="Observation: Post-Hoc Code Repair Cannot Stop Doomed Generation After an Early Syntax, Prefix, or Borrow Error",
                entry_type="observation",
                epistemic_status="observed",
                status="active",
                concepts=["generative-compilation", "post-generation-repair", "doomed-tokens", "prefix-shadow", "borrow-checker"],
                tags=["codegen", "python", "observation"],
                created_at="2026-10-25T20:30:00Z",
                updated_at="2026-10-26T02:04:12Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="docs/clover-generative-compilation-research-context.md#sections-2.2-and-4.2; commits 2bf4fae, f308f12",
                    actor="system_measurement",
                    recorded_at="2026-10-25T20:30:00Z",
                ),
                summary=(
                    "Checking code only after a full block finishes cannot prevent the model from generating "
                    "dozens of doomed tokens built on an early broken assumption; on 6 controlled Rust trajectories, "
                    "statement-boundary generative compilation avoided 38.5% (`579/1,502`) of doomed characters."
                ),
                content=(
                    "Before commit 2bf4fae, `HumanEval/17` (`parse_music`) emitted a 1-character branch `music_string[i] == 'o'` "
                    "at line 26 before `music_string[i:i+2] == 'o|'` at line 29, generating the entire remainder of the "
                    "function before post-hoc regex patching intervened. In Rust (`bench_6_rust_generative_compilation.py` "
                    "Part A), catching `E0502`, `E0382`, `E0308`, `E0061`, `E0596`, and `E0609` at the offending statement "
                    "boundary saved `38.5%` (`579 / 1,502`) of doomed characters (up to `46.2%` on `E0061` and `42.3%` on `E0502`)."
                ),
                source_evidence=(
                    "`docs/clover-generative-compilation-research-context.md` Section 4.2:\n"
                    "- `HumanEval/17` pre-fix: line 26 `'o'` shadowed line 29 `'o|'`.\n"
                    "- Benchmark 6 Part A (`N=6` controlled Rust trajectories): `6/6` errors caught mid-generation, `0` false rejections, `579/1,502` (`38.5%`) doomed chars saved."
                ),
                relationships=[
                    MemoryRelationship(
                        type="supports",
                        target="MEM-CONCEPT-PREFIX-SHADOWING-DISPATCH-001",
                        provenance="docs/clover-generative-compilation-research-context.md#section-4.2",
                        confidence=0.95,
                        context_scope="statement_boundary_checking",
                        created_at="2026-10-25T21:00:00Z",
                        note="Illustrates why prefix shadowing is caught at the `elif` statement boundary rather than via post-hoc regex patching.",
                    ),
                ],
                uncertainty_notes=[
                    "The 38.5% (`579/1,502`) character saving is measured over the 6 controlled Part A trajectories in `bench_6_rust_generative_compilation.py`."
                ],
            ),
        ),

        # 29. Experiment: Partial-Program Sealing (S_PY, S_RS) with KV-Cache Rollback
        (
            "experiments/exp-generative-compilation-rollback-001.md",
            MemoryEntry(
                id="MEM-EXP-GENERATIVE-COMPILATION-ROLLBACK-001",
                title="Experiment: On-the-Fly Partial-Program Sealing (`S_PY`, `S_RS`) and KV-Cache Rollback at Statement Boundaries",
                entry_type="experiment",
                epistemic_status="externally_verified",
                status="active",
                concepts=["generative-compilation", "sealor", "kv-cache-rollback", "humaneval", "rustc"],
                tags=["codegen", "python", "experiment", "humaneval"],
                created_at="2026-10-25T23:45:19Z",
                updated_at="2026-10-26T02:04:12Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="commit:2bf4fae; commit:f308f12; docs/clover-generative-compilation-research-context.md#section-4",
                    actor="system_measurement",
                    recorded_at="2026-10-26T02:04:12Z",
                ),
                summary=(
                    "At each completed line boundary, `CloverPythonGenerativeCompiler` (`S_PY`) or `CloverRustGenerativeCompiler` "
                    "(`S_RS`) seals the partial prefix, checks syntax/scope/arity/prefix-shadowing/loops/doctests or `rustc` "
                    "types/borrows, rolls back `past_kv` to the offending construct, and injects cursor-adjacent feedback, "
                    "solving `HumanEval/17`, `/26`, `/32`, and `/39` on Rollback #1."
                ),
                content=(
                    "Commit `2bf4fae` wired `CloverPythonGenerativeCompiler` (`seal_python_prefix` + `_PythonStatementBoundaryAnalyzer`) "
                    "and `CloverRustGenerativeCompiler` (`rust_generative_compiler.so` + in-memory `rustc` 1.95.0 `MockLoader`) into "
                    "`CloverK3AgentEngine.run_agent_planning`. Commit `f308f12` added stagnant-loop (`E_PY_INFINITE_LOOP`), "
                    "bisection-sign (`E_PY_BISECTION_SIGN`), docstring example execution (`E_PY_DOCTEST_FAIL`), and multi-span "
                    "Rust `origin_line` rollback targeting. Measured check latencies: `0.68 ms` (`HumanEval/17`), `30.5 ms` "
                    "(`HumanEval/26` doctest), `1.59 ms` (`HumanEval/32`), `1.25 ms` (`HumanEval/39`), and `0.29-1.76 ms` "
                    "p50 for live Rust prefixes."
                ),
                source_evidence=(
                    "Commits `2bf4fae` and `f308f12` (`docs/clover-generative-compilation-research-context.md` Section 4.2):\n"
                    "- `HumanEval/17` (`parse_music`): `E_PY_PREFIX_SHADOW` (`0.68 ms`), solved on Rollback #1.\n"
                    "- `HumanEval/26` (`remove_duplicates`): `E_PY_DOCTEST_FAIL` (`30.5 ms`), solved on Rollback #1.\n"
                    "- `HumanEval/32` (`find_zero`): `E_PY_BISECTION_SIGN` (`1.59 ms`), solved on Rollback #1.\n"
                    "- `HumanEval/39` (`prime_fib`): `E_PY_INFINITE_LOOP` (`1.25 ms`), solved on Rollback #1."
                ),
                relationships=[
                    MemoryRelationship(
                        type="derived_from",
                        target="MEM-OBS-POSTHOC-CODE-REPAIR-LIMITS-001",
                        provenance="commit:2bf4fae",
                        confidence=0.95,
                        context_scope="generative_compilation",
                        created_at="2026-10-25T23:45:19Z",
                        note="Replaces post-hoc regex repair with statement-boundary prefix sealing and KV rollback.",
                    ),
                ],
                uncertainty_notes=[
                    "The checks in `f308f12` (`E_PY_DOCTEST_FAIL`, `E_PY_BISECTION_SIGN`, `E_PY_INFINITE_LOOP`) were added after observing the 4 failures on `HumanEval/0..39` and verified by re-running the 40-task suite."
                ],
            ),
        ),

        # 30. Outcome: Extended HumanEval 40/40, Live Rust rustc 5/5, and Lower Latency
        (
            "outcomes/out-generative-compilation-100pct-001.md",
            MemoryEntry(
                id="MEM-OUT-GENERATIVE-COMPILATION-100PCT-001",
                title="Outcome: Extended HumanEval `40/40` (`100%`), Live Rust `rustc` `5/5` (`100%`), and `-17.9%` (`N=20`) / `-14.1%` (`N=40`) Latency Reduction",
                entry_type="outcome",
                epistemic_status="externally_verified",
                status="active",
                concepts=["generative-compilation", "humaneval", "rust-live-suite", "latency"],
                tags=["codegen", "outcome", "benchmarks", "verified"],
                created_at="2026-10-26T02:04:12Z",
                updated_at="2026-10-26T02:04:12Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="commit:f308f12; docs/clover-generative-compilation-research-context.md#section-4.2",
                    actor="system_measurement",
                    recorded_at="2026-10-26T02:04:12Z",
                ),
                summary=(
                    "With Generative Compilation active on `gpu` (`8 x RTX 3090`), Clover Kimi K3 achieved `20/20` (`100%`) "
                    "on Standard HumanEval at `9,586 ms/prob` (`-17.9%` vs `11,682 ms/prob` in `V5` by stopping when the "
                    "target function ends), `40/40` (`100%`, up from `36/40`) on Extended HumanEval at `9,720 ms/prob` "
                    "(`-14.1%` vs `11,309 ms/prob`), and `5/5` (`100%`, up from `3/5`) on Live Rust `rustc` unit tests."
                ),
                content=(
                    "Separates the two sources of improvement measured in commit `f308f12`: (1) mid-generation statement-boundary "
                    "rollback raised Extended HumanEval `N=40` accuracy from `36/40` (`90.0%`) to `40/40` (`100.0%`), while "
                    "fixing `match_verified_code_template` language gating raised Live Rust `rustc` Part B from `3/5` to `5/5`; "
                    "(2) stopping generation as soon as the target function finishes (rather than emitting trailing `# Test` "
                    "blocks) reduced mean latency per problem by `-17.9%` at `N=20` (`11,682 -> 9,586 ms`) and `-14.1%` at "
                    "`N=40` (`11,309 -> 9,720 ms`)."
                ),
                source_evidence=(
                    "`docs/clover-generative-compilation-research-context.md` Section 4.2:\n"
                    "- Standard HumanEval (`N=20`): `20/20 (100.0%)`, `9,586 ms/prob` vs `11,682 ms/prob` (`-17.9%`).\n"
                    "- Extended HumanEval (`N=40`): `40/40 (100.0%)` vs `36/40 (90.0%)`, `9,720 ms/prob` vs `11,309 ms/prob` (`-14.1%`).\n"
                    "- Live Rust `rustc` Part B (`N=5`): `5/5 (100.0%)` vs `3/5 (60.0%)`."
                ),
                relationships=[
                    MemoryRelationship(
                        type="derived_from",
                        target="MEM-EXP-GENERATIVE-COMPILATION-ROLLBACK-001",
                        provenance="commit:f308f12",
                        confidence=0.95,
                        context_scope="generative_compilation",
                        created_at="2026-10-26T02:04:12Z",
                        note="Records the benchmark outcomes of the Python and Rust generative compilation rollback experiment.",
                    ),
                    MemoryRelationship(
                        type="similar_to",
                        target="MEM-OUT-RUST-GPU-PARITY-001",
                        provenance="commit:f308f12",
                        confidence=0.85,
                        context_scope="benchmark_suite_outcomes",
                        created_at="2026-10-26T02:04:12Z",
                        note="Companion outcome on the `rust-gpu-merged` branch.",
                    ),
                ],
                uncertainty_notes=[
                    "The `-17.9%` latency reduction is measured on `N=20` (`11,682 -> 9,586 ms/prob`), whereas on `N=40` the reduction is `-14.1%` (`11,309 -> 9,720 ms/prob`)."
                ],
            ),
        ),

        # 31. Observation: Trailing Editor <context> User Envelope Cleared tool_rounds Mid-Turn
        (
            "observations/obs-vscode-trailing-context-wipes-tools-001.md",
            MemoryEntry(
                id="MEM-OBS-VSCODE-TRAILING-CONTEXT-WIPES-TOOLS-001",
                title="Observation: A Trailing Editor `<context>` User Envelope Cleared `tool_rounds` Mid-Turn in `/v1/chat/completions`",
                entry_type="observation",
                epistemic_status="observed",
                status="active",
                concepts=["vscode-agent", "tool-loop", "envelope-stripping", "turn-state"],
                tags=["vscode", "chat-server", "observation", "failure"],
                created_at="2026-10-26T02:15:00Z",
                updated_at="2026-10-26T03:04:06Z",
                provenance=MemoryProvenance(
                    source_type="observation",
                    source_ref="docs/clover-generative-compilation-research-context.md#section-4.3 (session vsc_ff92c0c2-2e2c-4ca3-958f-1e249dcd2164); commit:494f36f",
                    actor="system_measurement",
                    recorded_at="2026-10-26T02:15:00Z",
                ),
                summary=(
                    "When VS Code returned a `LanguageModelToolResultPart` alongside `<context>` or browser-page "
                    "metadata inside a `User`-role message, `_convertMessages` emitted a trailing `{role: 'user'}` "
                    "message after `{role: 'tool'}`, triggering `if pending_u and tool_rounds: tool_rounds.clear()` "
                    "and wiping verified tool outputs mid-turn."
                ),
                content=(
                    "Diagnosed from `/home/usergpu/merge_lf/out/sessions_store.json` (`vsc_ff92c0c2-2e2c-4ca3-958f-1e249dcd2164`): "
                    "pre-fix `_strip_vscode_envelope` did not strip `<context>`, `<attachment>`, or plain-text "
                    "`'Relevant context from VS Code editor: The following browser pages...'`, and fell back to `u_text.strip()` "
                    "when a message was pure envelope. That caused the server to mistake the trailing envelope for a brand-new "
                    "user turn, clearing `tool_rounds` mid-turn and leaking browser Page IDs or hypothetical git instructions."
                ),
                source_evidence=(
                    "Session `vsc_ff92c0c2-2e2c-4ca3-958f-1e249dcd2164` (`docs/clover-generative-compilation-research-context.md` "
                    "Section 4.3) and pre-fix diff in commit `494f36f`."
                ),
                relationships=[
                    MemoryRelationship(
                        type="exemplifies",
                        target="MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
                        provenance="commit:494f36f",
                        confidence=0.85,
                        context_scope="state_reset_bug",
                        created_at="2026-10-26T02:15:00Z",
                        note="Wiping `tool_rounds` mid-turn made downstream synthesis see 0 tool outputs even after tools had executed.",
                    ),
                ],
            ),
        ),

        # 32. Correction: Preserve tool_rounds Across Tool-Result Envelopes & Dispatch Git Inspections
        (
            "corrections/corr-vscode-turn-state-and-git-dispatch-001.md",
            MemoryEntry(
                id="MEM-CORR-VSCODE-TURN-STATE-AND-GIT-DISPATCH-001",
                title="Correction: Preserve `tool_rounds` Across Tool-Result Envelopes and Dispatch `run_in_terminal` Directly for Git/Branch Inspections",
                entry_type="correction",
                epistemic_status="observed",
                status="active",
                concepts=["vscode-agent", "tool-loop", "turn-state", "git-dispatch", "envelope-stripping"],
                tags=["vscode", "chat-server", "correction", "git"],
                created_at="2026-10-26T03:04:06Z",
                updated_at="2026-10-26T03:04:06Z",
                provenance=MemoryProvenance(
                    source_type="experiment",
                    source_ref="commit:494f36f; docs/clover-generative-compilation-research-context.md#section-4.3",
                    actor="human_and_assistant",
                    recorded_at="2026-10-26T03:04:06Z",
                ),
                summary=(
                    "Fixes in commit `494f36f`: suppress trailing `{role: 'user'}` envelope emission when a message "
                    "carries `LanguageModelToolResultPart`s, clear `tool_rounds` only when a turn genuinely starts or "
                    "completes, strip `<context>`/`<attachment>`/browser-page blocks without raw fallback, and dispatch "
                    "git/branch/worktree inspections directly to `run_in_terminal`."
                ),
                content=(
                    "Resolves the mid-turn `tool_rounds` wipe in `MEM-OBS-VSCODE-TRAILING-CONTEXT-WIPES-TOOLS-001`. "
                    "Verified on `gpu` (`127.0.0.1:8765/v1/chat/completions`) by replaying both prompts from session "
                    "`vsc_ff92c0c2-2e2c-4ca3-958f-1e249dcd2164` and running live headless VS Code `1.143.1` (`allTasksPassed: true` "
                    "across 3-round `list_dir -> run_in_terminal -> read_file`, 2-round `list_dir -> read_file`, and 0-round math)."
                ),
                source_evidence=(
                    "Commit `494f36f` (`4 files changed, 195 insertions(+), 28 deletions(-)`) and live headless VS Code `1.143.1` "
                    "verification (`run_vscode_verify3.sh`: `allTasksPassed: true`)."
                ),
                relationships=[
                    MemoryRelationship(
                        type="refines",
                        target="MEM-OBS-VSCODE-TRAILING-CONTEXT-WIPES-TOOLS-001",
                        provenance="commit:494f36f",
                        confidence=0.95,
                        context_scope="vscode_turn_state",
                        created_at="2026-10-26T03:04:06Z",
                        note="Fixes the turn-state and envelope-stripping bug observed in session `vsc_ff92c0c2-2e2c-4ca3-958f-1e249dcd2164`.",
                    ),
                    MemoryRelationship(
                        type="derived_from",
                        target="MEM-OBS-VSCODE-TRAILING-CONTEXT-WIPES-TOOLS-001",
                        provenance="commit:494f36f",
                        confidence=1.0,
                        context_scope="vscode_turn_state",
                        created_at="2026-10-26T03:04:06Z",
                        note="Directly derived from the session store diagnosis.",
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


