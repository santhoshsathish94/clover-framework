"""Clover Studio Capability S5: Session-Scoped Epistemic Memory Graph.

Distilled cleanly from `clover-ai-system/studio/clover/memory-knowledge-graph.py`
(`MemoryKnowledgeGraph` epistemic status & belief-revision lifecycle):
  - Indexes multi-turn conversation statements carrying epistemic provenance
    (`observed`, `hypothesis`, `user_stated_belief`, `externally_verified`) and
    revision relations (`corrects`, `supersedes`, `challenges`).
  - Tracks lifecycle transitions (`active` -> `superseded` on `corrects`/`supersedes`,
    `active` -> `contested` on `challenges`) and never promotes `hypothesis` or
    `user_stated_belief` to `observed`/`externally_verified` without an explicit
    verification event.
  - Strictly wiped on `POST /v1/cache/reset` (`EpistemicMemoryStore.reset()`).
  - Appends a compact `[Epistemic State Ledger]` strictly to the **tail** of the
    final user message ONLY when `user_turns >= 2` and epistemic revision/provenance
    markers exist — guaranteeing 0% Turn-1 token delta and 100% prefix LCP reuse!
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpistemicNode:
    node_id: str
    turn_idx: int
    epistemic_status: str  # observed | hypothesis | user_stated_belief | externally_verified
    lifecycle: str = "active"  # active | superseded | contested
    bindings: dict[str, str] = field(default_factory=dict)
    raw_summary: str = ""
    superseded_by: str | None = None
    challenged_by: list[str] = field(default_factory=list)


_EPISTEMIC_TRIGGER_RE = re.compile(
    r"\b(?:corrects|supersedes|challenges|contested|hypothesis|unverified|user_stated_belief|externally_verified)\b",
    re.IGNORECASE,
)

_ENTRY_DECL_RE = re.compile(
    r"(?:Record\s+telemetry\s+note|Also\s+record\s+note|Note|Entry|Update|Correction|Later\s+refinement)\s+"
    r"([A-Z]\d+)\s*\(([^)]*)\)\s*:\s*([^\n.]+(?:\.\d+[^\n.]*)?)",
    re.IGNORECASE,
)

_KV_ASSIGN_RE = re.compile(r"[`'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[`'\"]?\s*=\s*[`'\"]?([0-9A-Za-z._-]+)[`'\"]?")


def _infer_status(meta_str: str) -> str:
    low = meta_str.lower()
    if "externally_verified" in low:
        return "externally_verified"
    if "user_stated_belief" in low:
        return "user_stated_belief"
    if "hypothesis" in low or "unverified" in low:
        return "hypothesis"
    return "observed"


class EpistemicMemoryStore:
    """Thread-safe session-scoped epistemic ledger wiped on `/v1/cache/reset`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions_seen: int = 0

    def reset(self) -> None:
        with self._lock:
            self._sessions_seen = 0

    def build_ledger_for_messages(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        user_turns = [
            (idx, str(m.get("content") or ""))
            for idx, m in enumerate(messages)
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        if len(user_turns) < 2:
            return messages, {"active": False, "nodes": 0}

        full_text = "\n".join(txt for _, txt in user_turns)
        if not _EPISTEMIC_TRIGGER_RE.search(full_text):
            return messages, {"active": False, "nodes": 0}

        nodes: dict[str, EpistemicNode] = {}
        order: list[str] = []

        for turn_pos, (_, text) in enumerate(user_turns):
            for m in _ENTRY_DECL_RE.finditer(text):
                nid = m.group(1).upper()
                meta = m.group(2)
                body = m.group(3).strip()
                status = _infer_status(meta)
                kvs = {km.group(1): km.group(2) for km in _KV_ASSIGN_RE.finditer(body)}
                node = EpistemicNode(
                    node_id=nid,
                    turn_idx=turn_pos,
                    epistemic_status=status,
                    lifecycle="active",
                    bindings=kvs,
                    raw_summary=body[:140],
                )
                nodes[nid] = node
                order.append(nid)

                # Apply relational edges (`corrects X`, `supersedes X`, `challenges X`)
                for cm in re.finditer(r"(?:corrects|supersedes)\s+([A-Z]\d+)", meta, re.IGNORECASE):
                    target = cm.group(1).upper()
                    if target in nodes:
                        nodes[target].lifecycle = "superseded"
                        nodes[target].superseded_by = nid
                for chm in re.finditer(r"challenges\s+([A-Z]\d+)", meta, re.IGNORECASE):
                    target = chm.group(1).upper()
                    if target in nodes:
                        nodes[target].lifecycle = "contested"
                        nodes[target].challenged_by.append(nid)
                        node.lifecycle = "contested"

        if not nodes:
            return messages, {"active": False, "nodes": 0}

        with self._lock:
            self._sessions_seen += 1

        # Format compact deterministic ledger appended ONLY to the final user turn
        ledger_lines = ["[Epistemic State Ledger — verified provenance & belief-revision graph:]"]
        active_kv: dict[str, tuple[str, str, str]] = {}
        initial_kv: dict[str, str] = {}
        for nid in order:
            nd = nodes[nid]
            sup_str = f" -> superseded_by={nd.superseded_by}" if nd.superseded_by else ""
            chal_str = f" -> challenged_by={','.join(nd.challenged_by)}" if nd.challenged_by else ""
            kv_str = ", ".join(f"{k}={v}" for k, v in nd.bindings.items()) or nd.raw_summary
            ledger_lines.append(
                f"- {nd.node_id}: status={nd.epistemic_status}, lifecycle={nd.lifecycle}{sup_str}{chal_str} | {kv_str}"
            )
            for k, v in nd.bindings.items():
                if k not in initial_kv:
                    initial_kv[k] = v
                if nd.lifecycle == "active":
                    active_kv[k] = (v, nd.epistemic_status, nd.node_id)

        if active_kv:
            summary_pairs = ", ".join(f"{k}={v} ({st}, from {nid})" for k, (v, st, nid) in sorted(active_kv.items()))
            ledger_lines.append(f"- Current active values: {summary_pairs}")

        ledger_block = "\n".join(ledger_lines)
        out_msgs = [dict(m) for m in messages]
        last_u_idx = user_turns[-1][0]
        out_msgs[last_u_idx]["content"] = f"{out_msgs[last_u_idx]['content']}\n\n{ledger_block}"
        return out_msgs, {
            "active": True,
            "nodes": len(nodes),
            "superseded": sum(1 for n in nodes.values() if n.lifecycle == "superseded"),
            "contested": sum(1 for n in nodes.values() if n.lifecycle == "contested"),
        }
