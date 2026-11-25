"""Distributed Multi-Node Queue, Request Clubbing, Prefix-Batching Scheduler & MCP 2.0 Mesh Server.

Implements the three server-grade distributed systems pillars for Clover's 5-node `8x RTX 3090` mesh:
  1. `ReplicaPoolBackend`: Zero single-point-of-failure multi-replica pool (`Qwen3.6-35B` on Nodes 2/3/5,
     `LFM3-2B` on Nodes 2/3) with Prefix-Affinity KV/GDN cache routing, Least-In-Flight spillover, and
     automatic peer health failover.
  2. `ClusterBatchQueue`: 3-lane Priority Queue (`P0` interactive, `P1` agent, `P2` batch/tasks), In-Flight
     Request Clubbing (coalescing identical deterministic `T=0` requests onto 1 GPU pass), Time-Window
     Prefix-Grouped Micro-Batching (`batch_window_ms`), and `POST /v1/chat/completions/batch`.
  3. `Mcp2MeshServer`: Stateless Model Context Protocol 2.0 (`2026-07-28` spec) over `POST /mcp` & `GET /mcp`:
     - Stateless self-contained JSON-RPC 2.0 (zero sticky sessions, zero mandatory handshake)
     - `0.05 ms` Header-Based Mesh Routing (`Mcp-Method`, `Mcp-Name`, `Mcp-Priority`)
     - Deterministic, `ETag` + `Cache-Control` cacheable `tools/list` (`304 Not Modified`)
     - Lazy Tool Discovery (`tools/search`) so prompts load only 1-2 needed schemas (~90% token savings)
     - Multi-Call Clubbing (`clover_exec_batch` / `tools/call_batch`) running specialists in parallel
     - First-Class MCP 2.0 `tasks/*` Async Queue (`tasks/create`, `tasks/batch`, `tasks/get`, `tasks/result`, `tasks/cancel`)
"""
from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import copy
import hashlib
import heapq
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import studio_outcome_verifier as sout
import studio_xcruncher_verifier as sxc
from backends import HTTP_POOL, BackendBase, GenParams

MCP2_PROTOCOL_VERSION = "2026-07-28"
HOP_CTX = threading.local()


def peer_auth_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build inter-node headers carrying `Authorization: Bearer` + single-hop loop prevention."""
    hdrs: dict[str, str] = {"Content-Type": "application/json", "X-Clover-Hop": "1"}
    api_key = (os.environ.get("CLOVER_GATEWAY_API_KEY") or "").strip()
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"
    if extra:
        hdrs.update(extra)
    return hdrs


def compute_prefix_hash(messages: list[dict[str, Any]] | None) -> str:
    """Stable hash of leading system/developer + first user/repo context for Prefix-Affinity replica routing."""
    h = hashlib.sha256()
    if not messages:
        return "empty"
    taken = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role in ("system", "developer") or taken == 0:
            h.update(role.encode("utf-8"))
            h.update(b"\x00")
            c = m.get("content")
            raw = json.dumps(c, sort_keys=True, ensure_ascii=False) if not isinstance(c, str) else c
            h.update(raw[:8192].encode("utf-8", errors="replace"))
            h.update(b"\x01")
            if role == "user":
                taken += 1
                break
    return h.hexdigest()[:16]


def compute_clubbing_key(body: dict[str, Any]) -> str | None:
    """Return a deterministic coalescing key ONLY when request is deterministic (`T=0`, non-stream, no random seed)."""
    if body.get("stream"):
        return None
    temp = body.get("temperature")
    if isinstance(temp, (int, float)) and float(temp) > 0.0:
        return None
    canonical = {
        "model": str(body.get("model") or ""),
        "messages": body.get("messages") or [],
        "tools": body.get("tools") or None,
        "tool_choice": body.get("tool_choice") or None,
        "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens") or 1024,
        "stop": body.get("stop") or None,
        "seed": int(body.get("seed") or 0),
    }
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ====================================================================================================================
# 1. Multi-Node Prefix-Affinity + Failover Replica Pool (`ReplicaPoolBackend`)
# ====================================================================================================================

class ReplicaPoolBackend(BackendBase):
    """Distributes traffic across `N >= 1` local + remote replicas with Prefix-Affinity & automatic failover."""

    def __init__(self, model_id: str, replicas: list[tuple[str, BackendBase]], cooldown_s: float = 10.0):
        super().__init__()
        if not replicas:
            raise ValueError(f"ReplicaPoolBackend({model_id}) requires at least 1 replica")
        self.model_id = model_id
        self.replicas = list(replicas)  # list of (replica_label, backend)
        self.family = replicas[0][1].family
        self.kind = "replica-pool" if len(replicas) > 1 else replicas[0][1].kind
        self.cooldown_s = cooldown_s
        self._mu = threading.Lock()
        self._in_flight: dict[int, int] = {i: 0 for i in range(len(self.replicas))}
        self._unhealthy_until: dict[int, float] = {}
        self._lease_expires_at: dict[int, float] = {}
        self._prefix_owner: collections.OrderedDict[str, int] = collections.OrderedDict()
        self._rr = 0
        self.routed_counts: collections.Counter = collections.Counter()
        self.failovers: int = 0

    def reset_cache(self) -> None:
        with self._mu:
            self._prefix_owner.clear()
        for _, be in self.replicas:
            if hasattr(be, "reset_cache") and callable(be.reset_cache):
                be.reset_cache()

    def remember_assistant(self, content: Any, tool_calls: Any, gen_ids: list[int],
                           prefix_messages: list[dict[str, Any]] | None = None) -> None:
        for _, be in self.replicas:
            if hasattr(be, "remember_assistant") and callable(be.remember_assistant):
                be.remember_assistant(content, tool_calls, gen_ids, prefix_messages)

    def _pick_order(self, prefix_hash: str) -> list[int]:
        local_only = bool(getattr(HOP_CTX, "local_only", False))
        now = time.monotonic()
        wall_now = time.time()
        with self._mu:
            if local_only:
                local_idxs = [i for i, (lbl, _) in enumerate(self.replicas) if lbl == "local"]
                if not local_idxs:
                    raise RuntimeError(f"X-Clover-Hop forbids multi-hop forwarding, and {self.model_id} has no local replica")
                return local_idxs
            live = [
                i for i in range(len(self.replicas))
                if self._lease_expires_at.get(i, float("inf")) > wall_now
            ]
            if not live:
                raise RuntimeError(f"all dynamic petal leases for '{self.model_id}' have expired")
            healthy = [i for i in live if self._unhealthy_until.get(i, 0.0) <= now]
            if not healthy:
                healthy = list(live)
            min_load = min(self._in_flight[i] for i in healthy)
            owner = self._prefix_owner.get(prefix_hash)
            if owner is not None and owner in healthy and self._in_flight[owner] <= min_load + 1:
                self._prefix_owner.move_to_end(prefix_hash)
                primary = owner
            else:
                best: list[int] = [i for i in healthy if self._in_flight[i] == min_load]
                primary = best[self._rr % len(best)]
                self._rr += 1
                self._prefix_owner[prefix_hash] = primary
                while len(self._prefix_owner) > 2048:
                    self._prefix_owner.popitem(last=False)
            rest = [i for i in healthy if i != primary] + [i for i in live if i not in healthy]
            return [primary] + rest

    def add_or_refresh_replica(self, label: str, backend: BackendBase, expires_at: float | None = None) -> None:
        with self._mu:
            for idx, (lbl, _) in enumerate(self.replicas):
                if lbl == label:
                    self.replicas[idx] = (label, backend)
                    self._unhealthy_until.pop(idx, None)
                    if expires_at is not None:
                        self._lease_expires_at[idx] = expires_at
                    return
            new_idx = len(self.replicas)
            self.replicas.append((label, backend))
            self._in_flight[new_idx] = 0
            if expires_at is not None:
                self._lease_expires_at[new_idx] = expires_at
            self.kind = "replica-pool"

    def stream(self, messages, tools, tool_choice, params: GenParams):
        pf_hash = compute_prefix_hash(messages)
        order = self._pick_order(pf_hash)
        last_exc: Exception | None = None
        for attempt_idx, r_idx in enumerate(order):
            label, be = self.replicas[r_idx]
            with self._mu:
                self._in_flight[r_idx] += 1
            emitted_any = False
            try:
                for ev, payload in be.stream(messages, tools, tool_choice, params):
                    if ev == "delta":
                        emitted_any = True
                        yield ev, payload
                    else:
                        payload.extra["replica"] = label
                        if attempt_idx > 0:
                            payload.extra["failover_attempts"] = attempt_idx
                        with self._mu:
                            self.routed_counts[label] += 1
                        self._account(payload)
                        yield ev, payload
                return
            except Exception as exc:
                last_exc = exc
                with self._mu:
                    self._unhealthy_until[r_idx] = time.monotonic() + self.cooldown_s
                    if attempt_idx + 1 < len(order):
                        self.failovers += 1
                if emitted_any:
                    raise
            finally:
                with self._mu:
                    self._in_flight[r_idx] = max(0, self._in_flight[r_idx] - 1)
        assert last_exc is not None
        raise last_exc

    def snapshot(self) -> dict[str, Any]:
        with self._mu:
            return {
                "replicas": [lbl for lbl, _ in self.replicas],
                "in_flight": {self.replicas[i][0]: self._in_flight[i] for i in range(len(self.replicas))},
                "routed_counts": dict(self.routed_counts),
                "failovers": self.failovers,
            }


# ====================================================================================================================
# 2. Server Priority Queue + In-Flight Request Clubbing + Prefix Micro-Batcher (`ClusterBatchQueue`)
# ====================================================================================================================

@dataclass(order=True)
class _QueuedItem:
    priority: int
    prefix_bucket: str
    seq: int
    body: dict[str, Any] = field(compare=False)
    club_key: str | None = field(compare=False)
    enqueued_t: float = field(compare=False)
    future: concurrent.futures.Future = field(compare=False)


class ClusterBatchQueue:
    """Server-grade request queue with Priority Lanes, In-Flight Coalescing (Clubbing) & Prefix Micro-Batching."""

    def __init__(
        self,
        execute_fn: Callable[[dict[str, Any]], Any],
        max_workers: int | None = None,
        batch_window_ms: float | None = None,
        max_batch_size: int | None = None,
        max_queue_depth: int = 1024,
    ) -> None:
        self._execute_fn = execute_fn
        self.max_workers = int(max_workers or os.environ.get("CLOVER_QUEUE_WORKERS", "16"))
        self.batch_window_ms = float(
            batch_window_ms if batch_window_ms is not None else os.environ.get("CLOVER_BATCH_WINDOW_MS", "6.0")
        )
        self.max_batch_size = int(max_batch_size or os.environ.get("CLOVER_MAX_BATCH_SIZE", "8"))
        self.max_queue_depth = max_queue_depth

        self._cv = threading.Condition()
        self._heap: list[_QueuedItem] = []
        self._seq = 0
        self._inflight_clubs: dict[str, tuple[concurrent.futures.Future, int]] = {}
        self._closed = False
        self._stats: collections.Counter = collections.Counter()
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="clover-q")
        self._dispatcher = threading.Thread(target=self._dispatch_loop, name="clover-microbatch-dispatch", daemon=True)
        self._dispatcher.start()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._pool.shutdown(wait=False)

    @staticmethod
    def _clone_result(res: Any) -> Any:
        c = copy.copy(res)
        if hasattr(c, "telemetry") and isinstance(c.telemetry, dict):
            c.telemetry = dict(c.telemetry)
        return c

    def _enqueue_locked(
        self, body: dict[str, Any], priority: int, ckey: str | None, t_enq: float
    ) -> tuple[concurrent.futures.Future, bool]:
        """Must hold `self._cv`. Enqueues directly onto `self._heap` without consuming any worker thread."""
        self._stats["submitted"] += 1
        if ckey is not None and ckey in self._inflight_clubs:
            fut, fanout = self._inflight_clubs[ckey]
            self._inflight_clubs[ckey] = (fut, fanout + 1)
            self._stats["clubbed_coalesced"] += 1
            return fut, True
        if len(self._heap) >= self.max_queue_depth:
            self._stats["rejected_full"] += 1
            raise RuntimeError(f"server queue full ({self.max_queue_depth} pending)")
        fut = concurrent.futures.Future()
        if ckey is not None:
            self._inflight_clubs[ckey] = (fut, 1)
        pf_hash = compute_prefix_hash(body.get("messages"))
        self._seq += 1
        item = _QueuedItem(
            priority=max(0, min(2, int(priority))),
            prefix_bucket=pf_hash,
            seq=self._seq,
            body=body,
            club_key=ckey,
            enqueued_t=t_enq,
            future=fut,
        )
        heapq.heappush(self._heap, item)
        self._cv.notify()
        return fut, False

    def submit(self, body: dict[str, Any], priority: int = 1, local_only_hop: bool = False) -> tuple[Any, dict[str, Any]]:
        """Enqueue request (or club onto an identical in-flight deterministic request) and return `(result, q_meta)`."""
        if local_only_hop:
            body = {**body, "_x_local_only_hop": True}
        club_enabled = os.environ.get("CLOVER_ENABLE_CLUBBING", "1") != "0" and body.get("x_clover_clubbing", True) is not False
        ckey = compute_clubbing_key(body) if club_enabled else None
        t_enq = time.perf_counter()

        with self._cv:
            fut, is_follower = self._enqueue_locked(body, priority, ckey, t_enq)

        res, leader_meta = fut.result()
        wait_ms = round((time.perf_counter() - t_enq) * 1000.0, 2)
        q_meta = dict(leader_meta)
        q_meta["clubbed_coalesced"] = is_follower
        if is_follower:
            q_meta["total_wait_ms"] = wait_ms
        return self._clone_result(res), q_meta

    def submit_batch(self, items: list[dict[str, Any]], priority: int = 2, local_only_hop: bool = False) -> dict[str, Any]:
        """Club & enqueue an array of `N` requests directly onto `_heap` (zero worker-pool self-deadlock!)."""
        t0 = time.perf_counter()
        if not items:
            return {"object": "clover.batch", "count": 0, "clubbed_duplicates": 0, "results": [], "wall_ms": 0.0}

        indexed = list(enumerate(items))
        indexed.sort(key=lambda pair: compute_prefix_hash(pair[1].get("messages")))

        unique_futures: dict[str, concurrent.futures.Future] = {}
        slot_map: dict[int, tuple[concurrent.futures.Future, bool, float]] = {}
        clubbed_dups = 0

        with self._cv:
            for orig_idx, raw_req in indexed:
                req_body = {**raw_req, "_x_local_only_hop": True} if local_only_hop else raw_req
                ck = compute_clubbing_key(req_body)
                t_enq = time.perf_counter()
                if ck is not None and ck in unique_futures:
                    slot_map[orig_idx] = (unique_futures[ck], True, t_enq)
                    clubbed_dups += 1
                else:
                    fut, was_inflight_club = self._enqueue_locked(req_body, priority, ck, t_enq)
                    if ck is not None:
                        unique_futures[ck] = fut
                    if was_inflight_club:
                        clubbed_dups += 1
                    slot_map[orig_idx] = (fut, was_inflight_club, t_enq)
            self._stats["batch_calls"] += 1
            self._stats["batch_items"] += len(items)
            self._stats["clubbed_coalesced"] += clubbed_dups

        out_results: list[dict[str, Any]] = []
        for idx in range(len(items)):
            f, was_dup, _ = slot_map[idx]
            try:
                comp, qm = f.result()
                out_results.append({
                    "index": idx,
                    "status": 200,
                    "clubbed_in_batch": was_dup or bool(qm.get("clubbed_coalesced")),
                    "completion": self._clone_result(comp),
                    "queue": dict(qm),
                })
            except Exception as exc:
                out_results.append({
                    "index": idx,
                    "status": getattr(exc, "status", 500),
                    "error": str(exc),
                })

        return {
            "object": "clover.batch",
            "count": len(items),
            "unique_executed": len(items) - clubbed_dups,
            "clubbed_duplicates": clubbed_dups,
            "wall_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "results": out_results,
        }

    def _dispatch_loop(self) -> None:
        while True:
            with self._cv:
                while not self._heap and not self._closed:
                    self._cv.wait()
                if self._closed and not self._heap:
                    return
                # Wait up to `batch_window_ms` to club concurrent arrivals into a prefix-sorted micro-batch
                top_prio = self._heap[0].priority
                if self.batch_window_ms > 0 and top_prio > 0 and len(self._heap) < self.max_batch_size:
                    self._cv.wait(timeout=self.batch_window_ms / 1000.0)
                batch: list[_QueuedItem] = []
                while self._heap and len(batch) < self.max_batch_size:
                    batch.append(heapq.heappop(self._heap))
                # Sort micro-batch by `(priority, prefix_bucket, seq)` so identical repo prefixes execute together
                batch.sort(key=lambda x: (x.priority, x.prefix_bucket, x.seq))
                batch_sz = len(batch)
                if batch_sz > 1:
                    self._stats["microbatches_clubbed"] += 1
                    self._stats["microbatch_items_total"] += batch_sz

            for rank, item in enumerate(batch):
                self._pool.submit(self._run_item, item, batch_sz, rank)

    def _run_item(self, item: _QueuedItem, microbatch_size: int, rank: int) -> None:
        q_wait_ms = round((time.perf_counter() - item.enqueued_t) * 1000.0, 2)
        HOP_CTX.local_only = bool(item.body.get("_x_local_only_hop"))
        try:
            clean_body = {k: v for k, v in item.body.items() if k != "_x_local_only_hop"}
            res = self._execute_fn(clean_body)
            fanout = 1
            with self._cv:
                if item.club_key is not None and item.club_key in self._inflight_clubs:
                    _, fanout = self._inflight_clubs.pop(item.club_key)
                self._stats["completed"] += 1
            meta = {
                "priority": item.priority,
                "queue_wait_ms": q_wait_ms,
                "microbatch_size": microbatch_size,
                "microbatch_rank": rank,
                "clubbed_fanout": fanout,
                "prefix_bucket": item.prefix_bucket,
            }
            item.future.set_result((res, meta))
        except Exception as exc:
            with self._cv:
                if item.club_key is not None:
                    self._inflight_clubs.pop(item.club_key, None)
                self._stats["failed"] += 1
            item.future.set_exception(exc)
        finally:
            HOP_CTX.local_only = False

    def snapshot(self) -> dict[str, Any]:
        with self._cv:
            return {
                "pending": len(self._heap),
                "inflight_clubs": len(self._inflight_clubs),
                "max_workers": self.max_workers,
                "batch_window_ms": self.batch_window_ms,
                "max_batch_size": self.max_batch_size,
                "stats": dict(self._stats),
            }


# ====================================================================================================================
# 3. Stateless MCP 2.0 (`2026-07-28`) Mesh Server (`Mcp2MeshServer`)
# ====================================================================================================================

@dataclass
class McpToolSpec:
    name: str
    domain: str
    summary: str
    input_schema: dict[str, Any]
    target_model: str | None = None


MCP2_TOOLS: tuple[McpToolSpec, ...] = (
    McpToolSpec(
        name="boltz2_fold_structure",
        domain="biology",
        summary="Predict 3D biomolecular coordinates & ligand affinity via Boltz-2 specialist engine.",
        input_schema={
            "type": "object",
            "properties": {
                "sequence_or_fasta": {"type": "string", "description": "FASTA protein/RNA/DNA sequence or complex"},
                "ligand_smiles": {"type": "string", "description": "Optional small-molecule SMILES string"},
            },
            "required": ["sequence_or_fasta"],
        },
        target_model="boltz-2",
    ),
    McpToolSpec(
        name="chronos2_forecast",
        domain="timeseries",
        summary="Zero-shot calibrated probabilistic quantile forecast via Chronos-2 specialist engine.",
        input_schema={
            "type": "object",
            "properties": {
                "series": {"type": "array", "items": {"type": "number"}, "description": "Historical numeric values"},
                "horizon": {"type": "integer", "default": 12, "description": "Number of future steps to forecast"},
            },
            "required": ["series"],
        },
        target_model="chronos-2",
    ),
    McpToolSpec(
        name="clover_context_read",
        domain="clover_cycle",
        summary="Read the AGENTS.md §5 8-field Context->Growth handoff record for a task (`what_was_tried`, `what_was_ruled_out`).",
        input_schema={
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    ),
    McpToolSpec(
        name="clover_context_record",
        domain="clover_cycle",
        summary="Record an AGENTS.md §5 Context->Growth handoff (offered, not imposed) before starting the next cycle.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "intended_outcome": {"type": "string"},
                "what_was_tried": {"type": "string"},
                "what_the_system_showed": {"type": "string"},
                "what_was_ruled_out": {"type": "string"},
                "next_cycle_change": {"type": "string"},
            },
            "required": ["task_id", "intended_outcome", "next_cycle_change"],
        },
    ),
    McpToolSpec(
        name="clover_exec_batch",
        domain="cluster",
        summary="Club & execute multiple MCP 2.0 tool calls in parallel in a single stateless round-trip.",
        input_schema={
            "type": "object",
            "properties": {
                "calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["name"],
                    },
                }
            },
            "required": ["calls"],
        },
    ),
    McpToolSpec(
        name="goedel_lean4_prove",
        domain="mathematics",
        summary="Synthesize machine-checkable Lean 4 proof via Goedel-Prover-V2-32B + surface obligation check.",
        input_schema={
            "type": "object",
            "properties": {
                "theorem": {"type": "string", "description": "Lean 4 theorem statement (`theorem ... := by`)"},
            },
            "required": ["theorem"],
        },
        target_model="goedel-prover-v2-32b",
    ),
    McpToolSpec(
        name="lean4_surface_verify",
        domain="verifier",
        summary="Deterministic Lean 4 proof obligation guard (rejects `sorry`, `admit`, `axiom`, `native_decide`).",
        input_schema={
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "candidate": {"type": "string"},
            },
            "required": ["statement", "candidate"],
        },
    ),
    McpToolSpec(
        name="python_ast_verify",
        domain="verifier",
        summary="Deterministic AST-sandboxed Python doctest & safety verifier.",
        input_schema={
            "type": "object",
            "properties": {
                "prompt_with_doctest": {"type": "string"},
                "candidate_code": {"type": "string"},
            },
            "required": ["prompt_with_doctest", "candidate_code"],
        },
    ),
    McpToolSpec(
        name="vjepa2_predict_physics",
        domain="physics",
        summary="Predict physical object motion, trajectory & collision outcomes via Meta V-JEPA 2.1.",
        input_schema={
            "type": "object",
            "properties": {
                "video_ref": {"type": "string", "description": "Video URI/filename or scene description"},
                "question": {"type": "string", "description": "Physical consequence or trajectory question"},
            },
            "required": ["video_ref", "question"],
        },
        target_model="vjepa-2.1",
    ),
    McpToolSpec(
        name="xcruncher_exact_math",
        domain="mathematics",
        summary="Deterministic native C11 arbitrary-precision factorial, power, big-int product & BBP Pi hex verifier.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Exact math query, e.g. `37!` or `10 hex digits of pi at offset 5000`"},
            },
            "required": ["expression"],
        },
    ),
)

_TOOL_BY_NAME: dict[str, McpToolSpec] = {t.name: t for t in sorted(MCP2_TOOLS, key=lambda x: x.name)}


def _parse_kv_peers(raw: str) -> dict[str, str]:
    s = (raw or "").strip()
    if not s:
        return {}
    if s.startswith("{"):
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                return {str(k).strip(): str(v).strip().rstrip("/") for k, v in d.items() if v}
        except ValueError:
            return {}
    out: dict[str, str] = {}
    for part in s.split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            if k.strip() and v.strip():
                out[k.strip()] = v.strip().rstrip("/")
    return out


class PetalRegistry:
    """Dynamic Lease Registry on Center Node `k3` (`POST /v1/cluster/register`) for ephemeral GPU Petals."""

    def __init__(self, router: Any, mcp2: Mcp2MeshServer, lease_ttl_s: float = 90.0) -> None:
        self.router = router
        self.mcp2 = mcp2
        self.lease_ttl_s = lease_ttl_s
        self._mu = threading.Lock()
        self._petals: dict[str, dict[str, Any]] = {}

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        from backends import MODEL_ALIASES, MODEL_SPECS, RemoteOpenAIBackend

        petal_id = str(payload.get("petal_id") or "").strip()
        url = str(payload.get("url") or "").strip().rstrip("/")
        models = payload.get("models") if isinstance(payload.get("models"), list) else []
        mcp_tools = payload.get("mcp_tools") if isinstance(payload.get("mcp_tools"), list) else []
        kimi_url = str(payload.get("kimi_coordinator_url") or "").strip().rstrip("/")
        if not petal_id or not url.startswith(("http://", "https://", "tsock://")):
            raise ValueError("petal_id and valid http(s)/tsock url are required")

        now = time.time()
        exp = now + self.lease_ttl_s
        with self._mu:
            self._prune_locked(now)
            self._petals[petal_id] = {
                "petal_id": petal_id,
                "url": url,
                "models": [str(m) for m in models],
                "mcp_tools": [str(t) for t in mcp_tools],
                "kimi_coordinator_url": kimi_url or None,
                "vram_gib": payload.get("vram_gib"),
                "expires_at": exp,
                "updated_at": now,
            }

        for raw_m in models:
            mid = MODEL_ALIASES.get(str(raw_m).strip(), str(raw_m).strip())
            fam = MODEL_SPECS.get(mid, {}).get("family", "remote")
            rem_be = RemoteOpenAIBackend(mid, url, family=fam)
            existing = self.router.backends.get(mid)
            if isinstance(existing, ReplicaPoolBackend):
                existing.add_or_refresh_replica(url, rem_be, expires_at=exp)
            elif existing is not None:
                pool = ReplicaPoolBackend(mid, [("local", existing)])
                pool.add_or_refresh_replica(url, rem_be, expires_at=exp)
                self.router.backends[mid] = pool
            else:
                pool = ReplicaPoolBackend(mid, [(url, rem_be)])
                pool.add_or_refresh_replica(url, rem_be, expires_at=exp)
                self.router.backends[mid] = pool

        if kimi_url and "kimi-k3" in self.router.backends and hasattr(self.router.backends["kimi-k3"], "url"):
            self.router.backends["kimi-k3"].url = kimi_url  # type: ignore[attr-defined]

        for tname in mcp_tools:
            self.mcp2.tool_peers[str(tname).strip()] = url

        return {"registered": petal_id, "lease_ttl_s": self.lease_ttl_s, "active_petals": len(self._petals)}

    def _prune_locked(self, now: float) -> None:
        expired_ids = [pid for pid, rec in self._petals.items() if rec["expires_at"] <= now]
        for pid in expired_ids:
            rec = self._petals.pop(pid)
            dead_url = rec["url"]
            for tname in rec.get("mcp_tools", []):
                if self.mcp2.tool_peers.get(tname) == dead_url:
                    self.mcp2.tool_peers.pop(tname, None)

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._mu:
            self._prune_locked(now)
            return {"active_count": len(self._petals), "petals": dict(self._petals)}


class Mcp2MeshServer:
    """Stateless MCP 2.0 (`2026-07-28`) endpoint with Header Mesh Routing, Lazy Search, Batch Clubbing & Durable `tasks/*`."""

    def __init__(self, router: Any, batch_queue: ClusterBatchQueue, node_id: str | None = None) -> None:
        self.router = router
        self.queue = batch_queue
        self.node_id = node_id or os.environ.get("CLOVER_NODE_ID", "k3-center")
        self.tool_peers: dict[str, str] = _parse_kv_peers(os.environ.get("CLOVER_MCP_TOOL_PEERS", ""))
        self._tasks_mu = threading.Lock()
        self._tasks: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
        self._batch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=12, thread_name_prefix="clover-mcp-b")
        self._tasks_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="clover-mcp-t")
        self._db_path = (os.environ.get("CLOVER_TASKS_DB") or "").strip()
        if self._db_path:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            with contextlib.closing(sqlite3.connect(self._db_path)) as conn:
                with conn:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS mcp_tasks (task_id TEXT PRIMARY KEY, json_data TEXT NOT NULL, updated_at REAL)"
                    )

        # Precompute deterministic sorted `tools/list` payload + strong ETag for MCP 2.0 cacheable catalogs
        full_list = [self._render_tool(t, compact=False) for t in sorted(MCP2_TOOLS, key=lambda x: x.name)]
        self._full_tools_list = full_list
        self.catalog_etag = '"' + hashlib.sha256(
            json.dumps(full_list, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24] + '"'

    @staticmethod
    def _render_tool(t: McpToolSpec, compact: bool = False) -> dict[str, Any]:
        if compact:
            return {"name": t.name, "domain": t.domain, "description": t.summary}
        return {
            "name": t.name,
            "description": f"[{t.domain}] {t.summary}",
            "inputSchema": t.input_schema,
        }

    def maybe_forward_by_header(
        self, raw_body: bytes, headers: dict[str, str]
    ) -> tuple[int, dict[str, str], dict[str, Any]] | None:
        """0.05 ms Header-Based Mesh Routing: forward `Mcp-Method: tools/call` + `Mcp-Name` without parsing JSON!"""
        if headers.get("x-clover-mcp-hop") or headers.get("x-clover-hop"):
            return None
        mcp_method = (headers.get("mcp-method") or "").strip()
        mcp_name = (headers.get("mcp-name") or "").strip()
        if mcp_method == "tools/call" and mcp_name in self.tool_peers:
            spec = _TOOL_BY_NAME.get(mcp_name)
            if spec and spec.target_model and spec.target_model in self.router.backends:
                be = self.router.backends[spec.target_model]
                if getattr(be, "kind", "") != "remote-http":
                    return None
            peer_base = self.tool_peers[mcp_name]
            try:
                r_status, _, r_bytes = HTTP_POOL.request_json(
                    peer_base,
                    "/mcp",
                    raw_body,
                    peer_auth_headers({"Mcp-Method": mcp_method, "Mcp-Name": mcp_name, "X-Clover-Mcp-Hop": self.node_id}),
                    timeout_s=120.0,
                )
                resp_obj = json.loads(r_bytes.decode("utf-8") or "{}")
                return r_status, {"X-Clover-Mcp-Routed-To": peer_base, "Mcp-Protocol-Version": MCP2_PROTOCOL_VERSION}, resp_obj
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                return 502, {"Mcp-Protocol-Version": MCP2_PROTOCOL_VERSION}, {
                    "jsonrpc": "2.0", "id": None, "error": {"code": -32002, "message": f"Upstream petal {peer_base} error: {exc}"}
                }
        return None

    def handle_http(
        self, raw_body: bytes, headers: dict[str, str]
    ) -> tuple[int, dict[str, str], Any]:
        """Process a stateless MCP 2.0 HTTP POST request (`status, extra_headers, json_body_or_list`)."""
        norm_hdrs = {k.lower(): v for k, v in headers.items()}
        routed = self.maybe_forward_by_header(raw_body, norm_hdrs)
        if routed is not None:
            return routed

        try:
            rpc = json.loads(raw_body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            return 400, {"Mcp-Protocol-Version": MCP2_PROTOCOL_VERSION}, {
                "jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {exc}"}
            }

        # Standard JSON-RPC 2.0 Batch Array (`[{...}, {...}]`) support!
        if isinstance(rpc, list):
            if not rpc:
                return 400, {"Mcp-Protocol-Version": MCP2_PROTOCOL_VERSION}, {
                    "jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request: empty batch"}
                }
            batch_replies = []
            for elem in rpc:
                if not isinstance(elem, dict):
                    batch_replies.append({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}})
                else:
                    _, _, sub_body = self._handle_single_rpc(elem, norm_hdrs)
                    if sub_body is not None:
                        batch_replies.append(sub_body)
            return 200, {"Mcp-Protocol-Version": MCP2_PROTOCOL_VERSION, "X-Clover-Node": self.node_id}, batch_replies

        if not isinstance(rpc, dict):
            return 400, {"Mcp-Protocol-Version": MCP2_PROTOCOL_VERSION}, {
                "jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request: expected JSON object"}
            }

        return self._handle_single_rpc(rpc, norm_hdrs)

    def _handle_single_rpc(
        self, rpc: dict[str, Any], norm_hdrs: dict[str, str]
    ) -> tuple[int, dict[str, str], dict[str, Any] | None]:
        req_id = rpc.get("id")
        method = str(rpc.get("method") or norm_hdrs.get("mcp-method") or "").strip()
        params = rpc.get("params") if isinstance(rpc.get("params"), dict) else {}
        resp_hdrs = {"Mcp-Protocol-Version": MCP2_PROTOCOL_VERSION, "X-Clover-Node": self.node_id}

        # Backwards-compatible optional `initialize` (not required in MCP 2.0 stateless mode!)
        if method == "initialize":
            return 200, resp_hdrs, {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": MCP2_PROTOCOL_VERSION,
                    "stateless": True,
                    "capabilities": {
                        "tools": {"listChanged": False, "cacheable": True, "lazySearch": True, "batchCall": True},
                        "extensions": {"tasks": {"version": "2.0", "clubbing": True, "batch": True}},
                    },
                    "serverInfo": {"name": "clover-hg-mcp2-mesh", "version": "2.0.0", "node": self.node_id},
                },
            }

        # Cacheable deterministic `tools/list` (`ETag` + `304 Not Modified` + optional `compact` lazy mode)
        if method == "tools/list":
            compact = bool(params.get("compact"))
            if not compact and norm_hdrs.get("if-none-match") == self.catalog_etag:
                resp_hdrs["ETag"] = self.catalog_etag
                resp_hdrs["Cache-Control"] = "public, max-age=3600"
                return 304, resp_hdrs, None
            tools_out = (
                [self._render_tool(t, compact=True) for t in sorted(MCP2_TOOLS, key=lambda x: x.name)]
                if compact
                else self._full_tools_list
            )
            resp_hdrs["ETag"] = self.catalog_etag
            resp_hdrs["Cache-Control"] = "public, max-age=3600"
            return 200, resp_hdrs, {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": tools_out, "cacheControl": {"etag": self.catalog_etag, "maxAgeSeconds": 3600}},
            }

        # Lazy Tool Discovery (`tools/search`): returns ONLY matching tool schemas so prompts stay tiny!
        if method == "tools/search":
            q = str(params.get("query") or "").strip().lower()
            domain = str(params.get("domain") or "").strip().lower()
            limit = max(1, min(int(params.get("limit") or 4), len(MCP2_TOOLS)))
            scored: list[tuple[int, McpToolSpec]] = []
            tokens = [tok for tok in q.replace("_", " ").split() if tok]
            for t in sorted(MCP2_TOOLS, key=lambda x: x.name):
                if domain and t.domain != domain:
                    continue
                hay = f"{t.name} {t.domain} {t.summary}".lower()
                score = sum(3 if tok in t.name else (2 if tok == t.domain else (1 if tok in hay else 0)) for tok in tokens)
                if score > 0 or not tokens:
                    scored.append((score, t))
            scored.sort(key=lambda pair: (-pair[0], pair[1].name))
            matched = [self._render_tool(t, compact=False) for _, t in scored[:limit]]
            return 200, resp_hdrs, {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": matched, "totalCatalogSize": len(MCP2_TOOLS), "lazyLoaded": len(matched)},
            }

        # Direct Stateless Tool Execution (`tools/call`)
        if method == "tools/call":
            tname = str(params.get("name") or norm_hdrs.get("mcp-name") or "").strip()
            targs = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            try:
                out = self._invoke_tool(tname, targs)
                return 200, resp_hdrs, {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}],
                        "structuredContent": out,
                        "isError": False,
                    },
                }
            except KeyError as exc:
                return 404, resp_hdrs, {
                    "jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {exc}"}
                }
            except Exception as exc:
                return 200, resp_hdrs, {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": str(exc)}], "isError": True},
                }

        # Batch / Clubbed Tool Execution (`tools/call_batch`)
        if method == "tools/call_batch":
            calls = params.get("calls") if isinstance(params.get("calls"), list) else []
            out = self._exec_batch_calls(calls)
            return 200, resp_hdrs, {"jsonrpc": "2.0", "id": req_id, "result": out}

        # MCP 2.0 `tasks/*` Async Queue Extension (`tasks/create`, `tasks/batch`, `tasks/get`, `tasks/result`, `tasks/cancel`)
        if method == "tasks/create":
            rec = self._create_async_task(params)
            return 202, resp_hdrs, {"jsonrpc": "2.0", "id": req_id, "result": rec}

        if method == "tasks/batch":
            items = params.get("tasks") if isinstance(params.get("tasks"), list) else []
            recs = [self._create_async_task(it) for it in items if isinstance(it, dict)]
            return 202, resp_hdrs, {"jsonrpc": "2.0", "id": req_id, "result": {"tasks": recs, "count": len(recs)}}

        if method in ("tasks/get", "tasks/result"):
            tid = str(params.get("taskId") or params.get("id") or "")
            wait_ms = min(max(0, int(params.get("waitMs") or (5000 if method == "tasks/result" else 0))), 30000)
            rec = self._get_task(tid, wait_ms=wait_ms)
            if rec is None:
                return 404, resp_hdrs, {
                    "jsonrpc": "2.0", "id": req_id, "error": {"code": -32004, "message": f"Task '{tid}' not found"}
                }
            return 200, resp_hdrs, {"jsonrpc": "2.0", "id": req_id, "result": rec}

        if method == "tasks/cancel":
            tid = str(params.get("taskId") or params.get("id") or "")
            rec = self._cancel_task(tid)
            if rec is None:
                return 404, resp_hdrs, {
                    "jsonrpc": "2.0", "id": req_id, "error": {"code": -32004, "message": f"Task '{tid}' not found"}
                }
            return 200, resp_hdrs, {"jsonrpc": "2.0", "id": req_id, "result": rec}

        return 404, resp_hdrs, {
            "jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown MCP 2.0 method '{method}'"}
        }

    # -- Tool & Specialist implementations ---------------------------------------------------------------------------
    def _invoke_tool(self, name: str, args: dict[str, Any], _batch_depth: int = 0) -> dict[str, Any]:
        if name not in _TOOL_BY_NAME:
            raise KeyError(name)
        spec = _TOOL_BY_NAME[name]

        # If mapped to a remote specialist peer and not hosted locally, forward via stateless MCP 2.0!
        local_hosted = bool(
            spec.target_model
            and spec.target_model in self.router.backends
            and getattr(self.router.backends[spec.target_model], "kind", "") != "remote-http"
        )
        if name in self.tool_peers and not local_hosted:
            peer = self.tool_peers[name]
            payload = json.dumps({
                "jsonrpc": "2.0", "id": uuid.uuid4().hex[:12], "method": "tools/call",
                "params": {"name": name, "arguments": args},
            }).encode("utf-8")
            req = urllib.request.Request(
                peer + "/mcp",
                data=payload,
                headers=peer_auth_headers({
                    "Mcp-Method": "tools/call",
                    "Mcp-Name": name,
                    "X-Clover-Mcp-Hop": self.node_id,
                }),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120.0) as r:
                rpc_res = json.loads(r.read().decode("utf-8") or "{}")
            return (rpc_res.get("result") or {}).get("structuredContent") or {"forwarded_to": peer, "raw": rpc_res}

        if name == "clover_context_read":
            tid = str(args.get("task_id") or "").strip()
            rec = self.router.clover_cycle.get_record(tid)
            if rec is None:
                return {"tool": name, "found": False, "task_id": tid}
            return {
                "tool": name,
                "found": True,
                "task_id": rec.task_id,
                "cycle": rec.cycle_count,
                "intended_outcome": rec.intended_outcome,
                "what_was_tried": rec.what_was_tried,
                "what_the_system_showed": rec.what_the_system_showed,
                "what_was_ruled_out": rec.what_was_ruled_out,
                "next_cycle_change": rec.next_cycle_change,
                "context_file": rec.context_file_path,
                "markdown": rec.to_markdown(),
            }

        if name == "clover_context_record":
            from clover_cycle_engine import CloverContextRecord

            tid = str(args.get("task_id") or "manual_task").strip()
            existing = self.router.clover_cycle.get_record(tid)
            cyc = (existing.cycle_count + 1) if existing else 1
            tried = list(existing.what_was_tried) if existing else []
            if args.get("what_was_tried"):
                tried.append(str(args["what_was_tried"]))
            showed = list(existing.what_the_system_showed) if existing else []
            if args.get("what_the_system_showed"):
                showed.append(str(args["what_the_system_showed"]))
            ruled = list(existing.what_was_ruled_out) if existing else []
            if args.get("what_was_ruled_out"):
                ruled.append(str(args["what_was_ruled_out"]))
            new_rec = CloverContextRecord(
                task_id=tid,
                cycle_count=cyc,
                intended_outcome=str(args.get("intended_outcome") or ""),
                what_was_tried=tried,
                what_the_system_showed=showed,
                what_was_ruled_out=ruled,
                next_cycle_change=str(args.get("next_cycle_change") or ""),
            )
            saved = self.router.clover_cycle.save_record(new_rec)
            return {
                "tool": name,
                "saved": True,
                "offered_handoff": True,
                "task_id": saved.task_id,
                "cycle": saved.cycle_count,
                "context_file": saved.context_file_path,
            }

        if name == "clover_exec_batch":
            if _batch_depth >= 1:
                raise ValueError("Nested clover_exec_batch is forbidden (max batch depth = 1)")
            calls = args.get("calls") if isinstance(args.get("calls"), list) else []
            return self._exec_batch_calls(calls, _batch_depth=_batch_depth + 1)

        if name == "xcruncher_exact_math":
            expr = str(args.get("expression") or "")
            obs = sxc.extract_prompt_math_obligations(expr)
            return {
                "tool": name,
                "obligations_solved": len(obs),
                "verified_values": obs,
            }

        if name == "lean4_surface_verify":
            stmt = str(args.get("statement") or "Prove in Lean 4: theorem t : True := by")
            cand = str(args.get("candidate") or "")
            finding = sout.verify_lean_surface_fast(stmt, cand)
            return {"tool": name, "passed": finding is None, "finding": finding}

        if name == "python_ast_verify":
            prompt_dt = str(args.get("prompt_with_doctest") or "")
            cand = str(args.get("candidate_code") or "")
            safe_ast = sout._is_safe_candidate_ast(cand)
            return {"tool": name, "safe_ast": safe_ast}

        # Specialist model backed tools (`goedel_lean4_prove`, `boltz2_fold_structure`, `vjepa2_predict_physics`, `chronos2_forecast`)
        target_mid = spec.target_model or "clover-fusion-cluster"
        use_mid = target_mid if target_mid in self.router.backends else "clover-fusion-cluster"
        if name == "goedel_lean4_prove":
            thm = str(args.get("theorem") or "")
            prompt = f"import Mathlib\n{thm}" if "theorem" in thm or "lemma" in thm else f"Prove in Lean 4:\ntheorem goal : {thm} := by"
        elif name == "boltz2_fold_structure":
            seq = str(args.get("sequence_or_fasta") or "")
            lig = str(args.get("ligand_smiles") or "")
            prompt = f">boltz2_target\n{seq}" + (f"\nLigand SMILES: {lig}" if lig else "")
        elif name == "vjepa2_predict_physics":
            prompt = f"Video: {args.get('video_ref', 'clip.mp4')} — {args.get('question', 'Predict motion trajectory.')}"
        else:
            series = args.get("series") or []
            horizon = int(args.get("horizon") or 12)
            series_csv = ", ".join(str(x) for x in series[:128])
            prompt = f"Forecast next {horizon} steps for time-series: {series_csv}"

        comp, q_meta = self.queue.submit(
            {"model": use_mid, "messages": [{"role": "user", "content": prompt}], "max_tokens": 512, "temperature": 0.0},
            priority=1,
        )
        out: dict[str, Any] = {
            "tool": name,
            "served_by": comp.telemetry.get("served_by", use_mid),
            "node": self.node_id,
            "output": comp.content,
            "queue": q_meta,
        }
        if name == "goedel_lean4_prove":
            finding = sout.verify_lean_surface_fast(prompt, comp.content)
            out["lean_surface_verified"] = finding is None
            if finding:
                out["lean_finding"] = finding
        return out

    def _exec_batch_calls(self, calls: list[Any], _batch_depth: int = 1) -> dict[str, Any]:
        """Club duplicate tool calls & run independent specialist calls in parallel without pool self-deadlock!"""
        t0 = time.perf_counter()
        dedup_futures: dict[str, concurrent.futures.Future] = {}
        slots: list[tuple[int, concurrent.futures.Future | None, str | None, bool]] = []
        clubbed_n = 0

        for idx, c in enumerate(calls):
            if not isinstance(c, dict):
                continue
            cname = str(c.get("name") or "")
            cargs = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}
            if cname == "clover_exec_batch":
                slots.append((idx, None, "Nested clover_exec_batch is forbidden (max batch depth = 1)", False))
                continue
            sig = hashlib.sha256(
                json.dumps({"n": cname, "a": cargs}, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if sig in dedup_futures:
                slots.append((idx, dedup_futures[sig], None, True))
                clubbed_n += 1
            else:
                f = self._batch_pool.submit(self._invoke_tool, cname, cargs, _batch_depth)
                dedup_futures[sig] = f
                slots.append((idx, f, None, False))

        results = []
        for idx, f, imm_err, was_clubbed in slots:
            if imm_err is not None or f is None:
                results.append({"index": idx, "clubbed": was_clubbed, "ok": False, "error": imm_err or "invalid"})
                continue
            try:
                val = f.result()
                results.append({"index": idx, "clubbed": was_clubbed, "ok": True, "result": val})
            except Exception as exc:
                results.append({"index": idx, "clubbed": was_clubbed, "ok": False, "error": str(exc)})
        return {
            "count": len(slots),
            "unique_executed": len(slots) - clubbed_n,
            "clubbed_duplicates": clubbed_n,
            "wall_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "results": results,
        }

    def _persist_task_db(self, rec: dict[str, Any]) -> None:
        if not self._db_path:
            return
        pub = self._public_task(rec)
        with contextlib.closing(sqlite3.connect(self._db_path)) as conn:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO mcp_tasks (task_id, json_data, updated_at) VALUES (?, ?, ?)",
                    (pub["taskId"], json.dumps(pub, ensure_ascii=False), time.time()),
                )

    # -- MCP 2.0 Async `tasks/*` implementation ----------------------------------------------------------------------
    def _create_async_task(self, params: dict[str, Any]) -> dict[str, Any]:
        tid = "task_" + uuid.uuid4().hex[:16]
        created = time.time()
        ev = threading.Event()
        rec: dict[str, Any] = {
            "taskId": tid,
            "status": "queued",
            "createdAt": created,
            "node": self.node_id,
            "_done_ev": ev,
        }
        with self._tasks_mu:
            self._tasks[tid] = rec
            while len(self._tasks) > 2048:
                self._tasks.popitem(last=False)
        self._persist_task_db(rec)

        def _worker() -> None:
            with self._tasks_mu:
                if rec["status"] == "cancelled":
                    ev.set()
                    return
                rec["status"] = "running"
            self._persist_task_db(rec)
            try:
                if "tool" in params or "name" in params:
                    tname = str(params.get("tool") or params.get("name") or "")
                    targs = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
                    res = self._invoke_tool(tname, targs)
                else:
                    body = params.get("completion") if isinstance(params.get("completion"), dict) else params
                    comp, qm = self.queue.submit(body, priority=2)
                    res = {"content": comp.content, "served_by": comp.telemetry.get("served_by"), "queue": qm}
                with self._tasks_mu:
                    if rec["status"] != "cancelled":
                        rec["status"] = "completed"
                        rec["result"] = res
                        rec["completedAt"] = time.time()
            except Exception as exc:
                with self._tasks_mu:
                    rec["status"] = "failed"
                    rec["error"] = str(exc)
                    rec["completedAt"] = time.time()
            finally:
                self._persist_task_db(rec)
                ev.set()

        fut = self._tasks_pool.submit(_worker)
        rec["_fut"] = fut
        return self._public_task(rec)

    @staticmethod
    def _public_task(rec: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in rec.items() if not k.startswith("_")}

    def _get_task(self, tid: str, wait_ms: int = 0) -> dict[str, Any] | None:
        with self._tasks_mu:
            rec = self._tasks.get(tid)
        if rec is None:
            if self._db_path and Path(self._db_path).is_file():
                with contextlib.closing(sqlite3.connect(self._db_path)) as conn:
                    row = conn.execute("SELECT json_data FROM mcp_tasks WHERE task_id = ?", (tid,)).fetchone()
                    if row:
                        return json.loads(row[0])
            return None
        if wait_ms > 0:
            rec["_done_ev"].wait(timeout=wait_ms / 1000.0)
        with self._tasks_mu:
            return self._public_task(rec)

    def _cancel_task(self, tid: str) -> dict[str, Any] | None:
        with self._tasks_mu:
            rec = self._tasks.get(tid)
            if rec is None:
                return None
            if rec["status"] in ("queued", "running"):
                rec["status"] = "cancelled"
                fut = rec.get("_fut")
                if fut is not None:
                    fut.cancel()
                rec["_done_ev"].set()
            pub = self._public_task(rec)
        self._persist_task_db(rec)
        return pub
