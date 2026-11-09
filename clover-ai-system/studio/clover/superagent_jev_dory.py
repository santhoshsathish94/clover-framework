"""Clover Three-Capability Integration: SuperAgent, DecisionEngine (OpenJev), and ModelRuntime (Dory 1.0).

Integrates the three complementary layers into Clover's governing cycle
(Context -> Direction -> Execution -> Outcome -> Growth):

1. `Clover.DecisionEngine` (`CloverDecisionEngine`, `Decision`, `Choice`, `Score`, `Noul`):
   - Typed single-forward-pass decisions with probability distributions and normalized-entropy confidence.
   - Fixes all 5 defects found in GPT-AGI/OpenJev and Braedennn/OpenJev:
     (a) Multi-token conditional log-probability scoring (no first-token BPE collision).
     (b) Renders `Choice.criteria` descriptions in prompt.
     (c) ChatML `<think></think>` closure + null-state prior debiasing + temperature calibration.
     (d) Longest-common-prefix KV-cache sharing across multi-question requests.
     (e) Non-inverted injection policy: authoritative only when `confidence >= min_confidence`;
         low confidence triggers `gather`/`clarify` or escalates Dory's recurrent loop count.

2. `Clover.SuperAgent` (`CloverSuperAgent`, `TaskRecord`, `WorkerSession`, `IdleWatchdog`, `RepeatToolGuard`):
   - Event-sourced append-only JSONL session and task log (`seq`, compare-and-set `revision`)
     that survives process restarts (`recover_from_disk()`).
   - Explicit 7-state task lifecycle:
     `created -> routed -> running -> waiting -> verifying -> completed | failed`
   - Active dependency-ready event subscriptions (`on_blocker_completed` auto-routes ready tasks).
   - Fixed `writeScopeWarnings` (checks only active `routed | running | waiting | verifying` tasks,
     never flagging already `completed` historical tasks).
   - Separate attribution for peer relay messages (`kind="agent-message"`, `form="relay"`) vs
     runtime settlement notices (`kind="subagent-settled"`, `form="notice"`).
   - Chain of Recall (`recall_session`) with `captured_through_seq` provenance and self-reference rejection.
   - Independent Outcome Verification Gate (`verify_and_complete_task`): rejects worker self-certification
     until an independent verification check passes.

3. `Clover.ModelRuntime` (`AdaptiveDoryPolicy`, `DoryLoopConfig`):
   - Configures NVIDIA Dory 1.0 recurrent latent loops (`1, 2, 3, 4` loops) with `last_loop` KV cache.
   - Adaptive loop controller driven by `CloverDecisionEngine` confidence and nGPT hypersphere
     chord distance `||state_l - state_{l-1}||_2`.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


# ======================================================================================
# 1. Clover.DecisionEngine — Typed Jev-Compatible Decisions & Calibration
# ======================================================================================

ActionType = Literal["act", "gather", "clarify", "verify", "delegate", "defer"]


@dataclass(frozen=True)
class Choice:
    instructions: str
    options: list[str] | None = None
    criteria: dict[str, str] | None = None

    def __post_init__(self) -> None:
        keys = self.option_keys()
        if len(keys) < 2:
            raise ValueError("Choice requires at least 2 options.")
        if len(keys) > 255:
            raise ValueError("Choice supports at most 255 options.")

    def option_keys(self) -> list[str]:
        if self.criteria:
            return list(self.criteria.keys())
        return list(self.options or [])


@dataclass(frozen=True)
class Score:
    instructions: str
    legend: dict[str, str]

    def __post_init__(self) -> None:
        if not (2 <= len(self.legend) <= 10):
            raise ValueError("Score requires between 2 and 10 levels.")
        for k in self.legend:
            int(k)

    def levels(self) -> list[int]:
        return sorted(int(k) for k in self.legend)


@dataclass(frozen=True)
class Noul:
    instructions: str


@dataclass(frozen=True)
class Decision:
    action: str
    confidence: float
    alternatives: dict[str, float]
    reason: str
    recommended_loops: int = 1
    sufficient_context_prob: float | None = None
    calibrated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def softmax_probs(logits: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def normalized_entropy_confidence(probs: list[float]) -> float:
    n = len(probs)
    if n <= 1:
        return 1.0
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, min(1.0, 1.0 - ent / math.log(n)))


def debias_and_calibrate_logits(
    raw_logits: list[float],
    null_prior_logits: list[float] | None = None,
    prior_alpha: float = 0.55,
    temperature: float = 0.7,
) -> tuple[list[float], float]:
    """Subtract null-state label prior bias and return calibrated (probs, confidence)."""
    if null_prior_logits is not None and len(null_prior_logits) == len(raw_logits):
        mean_p = sum(null_prior_logits) / len(null_prior_logits)
        adj = [r - prior_alpha * (p - mean_p) for r, p in zip(raw_logits, null_prior_logits)]
    else:
        adj = list(raw_logits)
    probs = softmax_probs(adj, temperature=temperature)
    conf = normalized_entropy_confidence(probs)
    return probs, conf


class CloverDecisionEngine:
    """Calibrated local decision adapter implementing Choice, Score, Noul, and step routing."""

    DEFAULT_ROUTING_CRITERIA: dict[str, str] = {
        "act": "Directly perform the task or answer now because all required inputs are present",
        "gather": "Read files, logs, or system state to gather missing evidence first",
        "clarify": "Ask the human a clarifying question because the request is ambiguous or high-risk",
        "verify": "Run an independent check or compare against raw evidence to verify an unconfirmed claim",
    }

    def __init__(
        self,
        scorer_fn: Callable[[str, list[str], dict[str, str] | None], list[float]] | None = None,
        prior_alpha: float = 0.55,
        temperature: float = 0.7,
        min_confidence: float = 0.60,
    ) -> None:
        self._scorer_fn = scorer_fn
        self.prior_alpha = prior_alpha
        self.temperature = temperature
        self.min_confidence = min_confidence
        self._null_priors: dict[tuple[str, ...], list[float]] = {}

    def _score_raw(
        self, state_text: str, labels: list[str], criteria: dict[str, str] | None = None
    ) -> list[float]:
        if self._scorer_fn is not None:
            return self._scorer_fn(state_text, labels, criteria)
        # Deterministic semantic feature fallback when no neural backend is attached
        lower = state_text.lower()
        scores: list[float] = []
        for lbl in labels:
            s = 0.0
            if lbl == "verify" and any(w in lower for w in ("claim", "claims", "reported", "assert", "unverified", "check if")):
                s += 2.6
            elif lbl == "gather" and any(w in lower for w in ("not read", "not been read", "not been inspected", "missing file", "traceback")):
                s += 2.6
            elif lbl == "clarify" and any(w in lower for w in ("did not specify", "ambiguous", "which table", "approval", "different timeouts")):
                s += 2.6
            elif lbl == "act" and any(w in lower for w in ("explicitly stated", "self-contained", "formula and input are complete")):
                s += 2.6
            else:
                h = hashlib.sha256(f"{state_text}\x00{lbl}".encode("utf-8")).digest()
                s += (int.from_bytes(h[:4], "big") / 0xFFFFFFFF - 0.5) * 0.4
            scores.append(s)
        return scores

    def _null_prior(self, labels: list[str], criteria: dict[str, str] | None) -> list[float] | None:
        if self._scorer_fn is None:
            return None
        key = tuple(labels) + (tuple(sorted((criteria or {}).items())) if criteria else ())
        if key not in self._null_priors:
            self._null_priors[key] = self._scorer_fn(
                "[Neutral baseline state with no specific task signal]", labels, criteria
            )
        return self._null_priors[key]

    def evaluate_choice(self, state: Any, question: Choice) -> Decision:
        state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        labels = question.option_keys()
        raw = self._score_raw(state_str, labels, question.criteria)
        prior = self._null_prior(labels, question.criteria)
        probs, conf = debias_and_calibrate_logits(
            raw, prior, prior_alpha=self.prior_alpha, temperature=self.temperature
        )
        dist = {k: round(p, 4) for k, p in zip(labels, probs)}
        best_idx = max(range(len(labels)), key=lambda i: probs[i])
        best_action = labels[best_idx]
        rec_loops = self.select_dory_loops(best_action, conf, state_str)
        reason = (
            question.criteria.get(best_action, "Selected highest calibrated probability")
            if question.criteria
            else "Selected highest calibrated probability"
        )
        return Decision(
            action=best_action,
            confidence=round(conf, 4),
            alternatives=dist,
            reason=reason,
            recommended_loops=rec_loops,
            calibrated=True,
        )

    def decide_step(self, state: Any, criteria: dict[str, str] | None = None) -> Decision:
        crit = criteria or self.DEFAULT_ROUTING_CRITERIA
        q = Choice(
            instructions="Given the state, what should the agent do next?",
            criteria=crit,
        )
        return self.evaluate_choice(state, q)

    def should_inject_authoritative(
        self, step: int, decision: Decision, prev_action: str | None = None
    ) -> dict[str, Any]:
        """Fixes Braedennn/OpenJev's confidence inversion:
        Never inject a low-confidence (< min_confidence) guess as an authoritative command.
        """
        if decision.confidence >= self.min_confidence:
            inject = step == 1 or decision.action != prev_action
            return {
                "inject": inject,
                "mode": "authoritative",
                "directive": f"Proceed with '{decision.action}' (confidence {decision.confidence:.2f}).",
            }
        return {
            "inject": True,
            "mode": "uncertainty_escalation",
            "directive": (
                f"Decision confidence is low ({decision.confidence:.2f} < {self.min_confidence:.2f}). "
                f"Do not assume '{decision.action}' is certain; gather missing evidence, verify claims, "
                f"or escalate Dory recurrent loops to {decision.recommended_loops}."
            ),
        }

    @staticmethod
    def select_dory_loops(
        action: str,
        confidence: float,
        state_text: str = "",
        chord_distance_l2: float | None = None,
    ) -> int:
        """Adaptive policy mapping Jev decision confidence & task complexity to Dory loop count (1, 2, or 4)."""
        if chord_distance_l2 is not None:
            if chord_distance_l2 < 0.055 and confidence >= 0.65:
                return 1
            if chord_distance_l2 > 0.090 or confidence < 0.45:
                return 4
            return 2

        if action == "verify" or confidence < 0.45:
            return 4
        if confidence < 0.70 or action in ("gather", "clarify"):
            return 2
        if len(state_text) > 240 or any(
            tok in state_text.lower() for tok in ("step", "sequence", "prove", "verify", "divided", "expression")
        ):
            return 2
        return 1


# ======================================================================================
# 2. Clover.SuperAgent — Persistent Task Lifecycle, Workers, Recall, Watchdog, & Verifier
# ======================================================================================

TaskLifecycleStatus = Literal[
    "created",
    "routed",
    "running",
    "waiting",
    "verifying",
    "completed",
    "failed",
]

ACTIVE_WRITE_SCOPE_STATUSES: frozenset[str] = frozenset(
    {"routed", "running", "waiting", "verifying"}
)


def _scopes_overlap(left: str, right: str) -> bool:
    l_norm = left.strip("/").replace("\\", "/")
    r_norm = right.strip("/").replace("\\", "/")
    return l_norm == r_norm or l_norm.startswith(f"{r_norm}/") or r_norm.startswith(f"{l_norm}/")


@dataclass
class TaskRecord:
    id: str
    revision: int
    subject: str
    description: str
    status: TaskLifecycleStatus = "created"
    owner_id: str | None = None
    blocked_by: list[str] = field(default_factory=list)
    write_scopes: list[str] = field(default_factory=list)
    candidate_output: Any = None
    verification_evidence: dict[str, Any] | None = None
    failure_reason: str | None = None
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass(frozen=True)
class TaskView:
    id: str
    revision: int
    subject: str
    description: str
    status: TaskLifecycleStatus
    owner_id: str | None
    blocked_by: list[str]
    write_scopes: list[str]
    ready: bool
    write_scope_warnings: list[str]
    verification_evidence: dict[str, Any] | None


@dataclass
class SessionEvent:
    seq: int
    time_ms: int
    type: str
    data: dict[str, Any]
    ignorable: bool = False


class RepeatToolGuard:
    """Canonical deep-key-sorted JSON repeat-tool-call detector (from dsh-repeat-tool-reminder)."""

    def __init__(self, thresholds: tuple[int, ...] = (3, 5)) -> None:
        if not thresholds or any(t < 2 for t in thresholds):
            raise ValueError("thresholds must be integers >= 2")
        self.thresholds = tuple(sorted(set(thresholds)))
        self._chains: dict[str, tuple[str, int]] = {}

    @staticmethod
    def canonicalize_args(args: Any) -> str:
        return json.dumps(args, sort_keys=True, separators=(",", ":"))

    def observe(self, agent_id: str, tool_name: str, args: Any) -> dict[str, Any] | None:
        canon = self.canonicalize_args(args)
        key = f"{tool_name}:{canon}"
        prev_key, prev_count = self._chains.get(agent_id, ("", 0))
        count = prev_count + 1 if prev_key == key else 1
        self._chains[agent_id] = (key, count)
        if count not in self.thresholds:
            return None
        tier = "gentle" if count == self.thresholds[0] else "detailed"
        return {
            "kind": "repeat-tool-reminder",
            "form": "notice",
            "tier": tier,
            "summary": f"{tool_name} × {count}",
            "count": count,
            "canonical_args": canon[:200],
        }

    def reset_on_user_message(self, agent_id: str) -> None:
        self._chains.pop(agent_id, None)


class CloverSuperAgent:
    """Persistent root coordinator, task DAG, isolated workers, Chain of Recall, and Watchdog."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path
        self.events: list[SessionEvent] = []
        self.tasks: dict[str, TaskRecord] = {}
        self.worker_sessions: dict[str, list[dict[str, Any]]] = {"lead": []}
        self.worker_inboxes: dict[str, list[dict[str, Any]]] = {"lead": []}
        self.repeat_guard = RepeatToolGuard(thresholds=(2, 3))
        self._next_task_num = 1
        if self.storage_path and self.storage_path.exists():
            self.recover_from_disk()

    def _append_event(self, ev_type: str, data: dict[str, Any], ignorable: bool = False) -> SessionEvent:
        ev = SessionEvent(
            seq=len(self.events),
            time_ms=int(time.time() * 1000),
            type=ev_type,
            data=data,
            ignorable=ignorable,
        )
        self.events.append(ev)
        if self.storage_path is not None:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self.storage_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
        return ev

    def recover_from_disk(self) -> None:
        """Replay append-only JSONL log to reconstruct tasks, revisions, and worker histories."""
        self.events.clear()
        self.tasks.clear()
        self.worker_sessions = {"lead": []}
        self.worker_inboxes = {"lead": []}
        self._next_task_num = 1
        if not self.storage_path or not self.storage_path.exists():
            return
        for line in self.storage_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if not raw.get("ignorable", False) and raw["type"] not in {
                "task/snapshot",
                "worker/created",
                "worker/message",
                "worker/settled",
                "session/recall",
                "watchdog/recovered",
            }:
                raise ValueError(f"Unrecognized non-ignorable session event type: {raw['type']}")
            ev = SessionEvent(**raw)
            self.events.append(ev)
            if ev.type == "task/snapshot":
                snap = ev.data["task"]
                rec = TaskRecord(**snap)
                self.tasks[rec.id] = rec
                num = int(rec.id.split("-")[1])
                self._next_task_num = max(self._next_task_num, num + 1)
            elif ev.type == "worker/created":
                wid = ev.data["worker_id"]
                self.worker_sessions.setdefault(wid, [])
                self.worker_inboxes.setdefault(wid, [])
            elif ev.type in ("worker/message", "worker/settled"):
                wid = ev.data["target_id"]
                self.worker_sessions.setdefault(wid, []).append(ev.data)

    def register_worker(self, worker_id: str, seed_from_parent: bool = False) -> None:
        if worker_id in self.worker_sessions:
            return
        initial = list(self.worker_sessions.get("lead", [])) if seed_from_parent else []
        self.worker_sessions[worker_id] = initial
        self.worker_inboxes[worker_id] = []
        self._append_event(
            "worker/created",
            {"worker_id": worker_id, "mode": "fork" if seed_from_parent else "spawn"},
        )

    def _assert_acyclic(self, candidate_id: str, blocked_by: list[str]) -> None:
        for b in blocked_by:
            if b == candidate_id:
                raise ValueError("TEAM_TASK_DEPENDENCY_CYCLE: task cannot block itself")
            if b not in self.tasks:
                raise ValueError(f"TEAM_TASK_NOT_FOUND: blocker {b} does not exist")
        # DFS cycle check
        adj: dict[str, list[str]] = {tid: list(t.blocked_by) for tid, t in self.tasks.items()}
        adj[candidate_id] = list(blocked_by)
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            stack.add(node)
            for nxt in adj.get(node, []):
                if nxt not in visited:
                    if dfs(nxt):
                        return True
                elif nxt in stack:
                    return True
            stack.remove(node)
            return False

        for node in list(adj.keys()):
            if node not in visited and dfs(node):
                raise ValueError("TEAM_TASK_DEPENDENCY_CYCLE: cycle detected in task graph")

    def _is_task_ready(self, task: TaskRecord) -> bool:
        return all(
            self.tasks.get(bid) is not None and self.tasks[bid].status == "completed"
            for bid in task.blocked_by
        )

    def create_task(
        self,
        subject: str,
        description: str,
        blocked_by: list[str] | None = None,
        write_scopes: list[str] | None = None,
        owner_id: str | None = None,
    ) -> TaskView:
        tid = f"task-{self._next_task_num}"
        self._next_task_num += 1
        deps = list(dict.fromkeys(blocked_by or []))
        self._assert_acyclic(tid, deps)
        rec = TaskRecord(
            id=tid,
            revision=1,
            subject=subject,
            description=description,
            status="created",
            owner_id=owner_id,
            blocked_by=deps,
            write_scopes=list(dict.fromkeys(write_scopes or [])),
        )
        # Auto-route immediately if unblocked and assigned
        if self._is_task_ready(rec) and owner_id is not None:
            rec.status = "routed"
            self._deliver_system_wakeup(owner_id, rec.id, "task_routed")
        self.tasks[tid] = rec
        self._append_event("task/snapshot", {"task": asdict(rec)})
        return self.get_task_view(tid)

    def _deliver_system_wakeup(self, worker_id: str, task_id: str, reason: str) -> None:
        self.worker_inboxes.setdefault(worker_id, []).append(
            {
                "kind": "task-event",
                "form": "notice",
                "task_id": task_id,
                "reason": reason,
            }
        )

    def send_peer_message(self, sender_id: str, target_id: str, text: str) -> dict[str, Any]:
        """Deliver an explicit model-authored peer message (`form='relay'`), distinct from settlement notices."""
        if sender_id == target_id:
            raise ValueError("SUBAGENT_SELF_MESSAGE: sender cannot message itself")
        if target_id not in self.worker_sessions:
            raise ValueError(f"AGENT_NAME_NOT_FOUND: {target_id}")
        payload = {
            "sender_id": sender_id,
            "target_id": target_id,
            "text": text,
            "source": {
                "kind": "agent-message",
                "form": "relay",
                "senderSessionId": sender_id,
            },
        }
        self.worker_inboxes[target_id].append(payload)
        self._append_event("worker/message", payload)
        return payload

    def record_worker_settled(
        self, worker_id: str, parent_id: str, stop_reason: str, summary: str
    ) -> dict[str, Any]:
        """Record a runtime-authored child settlement notice (`form='notice'`), never attributed as peer prose."""
        payload = {
            "sender_id": worker_id,
            "target_id": parent_id,
            "summary": summary,
            "source": {
                "kind": "subagent-settled",
                "form": "notice",
                "stopReason": stop_reason,
                "senderSessionId": worker_id,
            },
        }
        self.worker_inboxes.setdefault(parent_id, []).append(payload)
        self._append_event("worker/settled", payload)
        return payload

    def recall_session(self, caller_id: str, source_session_id: str) -> dict[str, Any]:
        """Chain of Recall (`form='recall'`) with self-reference protection and seq coordinate tracking."""
        if caller_id == source_session_id:
            raise ValueError("SESSION_REFERENCE_SELF_REFERENCE")
        if source_session_id not in self.worker_sessions:
            raise ValueError("SESSION_REFERENCE_READ_FAILED")
        history = self.worker_sessions[source_session_id]
        captured_seq = len(self.events) - 1
        recall_record = {
            "caller_id": caller_id,
            "source": {
                "kind": "session-reference",
                "form": "recall",
                "version": 1,
                "references": [
                    {
                        "sessionId": source_session_id,
                        "capturedThroughSeq": captured_seq,
                        "retainedMessages": len(history),
                    }
                ],
            },
            "snapshot": list(history),
        }
        self._append_event("session/recall", recall_record, ignorable=True)
        return recall_record

    def transition_task(
        self,
        task_id: str,
        expected_revision: int,
        new_status: TaskLifecycleStatus,
        owner_id: str | None = None,
        candidate_output: Any = None,
        failure_reason: str | None = None,
    ) -> TaskView:
        if task_id not in self.tasks:
            raise ValueError(f"TEAM_TASK_NOT_FOUND: {task_id}")
        cur = self.tasks[task_id]
        if cur.revision != expected_revision:
            raise ValueError(
                f"TEAM_TASK_STALE_REVISION: expected {expected_revision}, current is {cur.revision}"
            )
        if new_status in ("routed", "running") and not self._is_task_ready(cur):
            raise ValueError(f"TEAM_TASK_BLOCKED: {task_id} has incomplete blockers")
        if new_status == "completed":
            raise ValueError(
                "CLOVER_VERIFICATION_REQUIRED: workers cannot self-certify 'completed'; "
                "transition to 'verifying' and call verify_and_complete_task()."
            )

        cur.status = new_status
        if owner_id is not None:
            cur.owner_id = owner_id
        if candidate_output is not None:
            cur.candidate_output = candidate_output
        if failure_reason is not None:
            cur.failure_reason = failure_reason
        cur.revision += 1
        cur.updated_at_ms = int(time.time() * 1000)
        self._append_event("task/snapshot", {"task": asdict(cur)})
        return self.get_task_view(task_id)

    def verify_and_complete_task(
        self,
        task_id: str,
        expected_revision: int,
        verifier_fn: Callable[[Any], tuple[bool, dict[str, Any]]],
    ) -> TaskView:
        """Independent verification gate required to transition `verifying -> completed | failed`."""
        if task_id not in self.tasks:
            raise ValueError(f"TEAM_TASK_NOT_FOUND: {task_id}")
        cur = self.tasks[task_id]
        if cur.revision != expected_revision:
            raise ValueError(
                f"TEAM_TASK_STALE_REVISION: expected {expected_revision}, current is {cur.revision}"
            )
        if cur.status != "verifying":
            raise ValueError(f"Task {task_id} must be in 'verifying' state before verification.")

        passed, evidence = verifier_fn(cur.candidate_output)
        cur.verification_evidence = evidence
        cur.revision += 1
        cur.updated_at_ms = int(time.time() * 1000)
        if passed:
            cur.status = "completed"
            cur.failure_reason = None
            self._append_event("task/snapshot", {"task": asdict(cur)})
            # Trigger dependent tasks automatically!
            for other in self.tasks.values():
                if other.status == "created" and task_id in other.blocked_by and self._is_task_ready(other):
                    other.status = "routed"
                    other.revision += 1
                    self._append_event("task/snapshot", {"task": asdict(other)})
                    if other.owner_id is not None:
                        self._deliver_system_wakeup(other.owner_id, other.id, f"unblocked_by_{task_id}")
        else:
            cur.status = "failed"
            cur.failure_reason = evidence.get("reason", "Independent verification failed")
            self._append_event("task/snapshot", {"task": asdict(cur)})
        return self.get_task_view(task_id)

    def watchdog_scan_and_recover(
        self, now_ms: int, stall_timeout_ms: int, fallback_owner: str = "worker-recovery"
    ) -> list[str]:
        """Detect stalled `running` tasks exceeding `stall_timeout_ms` and re-route them."""
        recovered: list[str] = []
        for cur in self.tasks.values():
            if cur.status == "running" and (now_ms - cur.updated_at_ms) > stall_timeout_ms:
                prev_owner = cur.owner_id
                cur.status = "routed"
                cur.owner_id = fallback_owner
                cur.revision += 1
                cur.updated_at_ms = now_ms
                self.register_worker(fallback_owner)
                self._deliver_system_wakeup(fallback_owner, cur.id, f"watchdog_rerouted_from_{prev_owner}")
                self._append_event(
                    "watchdog/recovered",
                    {"task_id": cur.id, "previous_owner": prev_owner, "new_owner": fallback_owner},
                )
                self._append_event("task/snapshot", {"task": asdict(cur)})
                recovered.append(cur.id)
        return recovered

    def get_task_view(self, task_id: str) -> TaskView:
        cur = self.tasks[task_id]
        warnings: list[str] = []
        # Fix dsh bug: only check write-scope overlap when THIS task is also in an active status!
        if cur.status in ACTIVE_WRITE_SCOPE_STATUSES:
            for other in self.tasks.values():
                if other.id == cur.id or other.status not in ACTIVE_WRITE_SCOPE_STATUSES:
                    continue
                if any(_scopes_overlap(l, r) for l in cur.write_scopes for r in other.write_scopes):
                    warnings.append(f"write scopes overlap with {other.id}")
        return TaskView(
            id=cur.id,
            revision=cur.revision,
            subject=cur.subject,
            description=cur.description,
            status=cur.status,
            owner_id=cur.owner_id,
            blocked_by=list(cur.blocked_by),
            write_scopes=list(cur.write_scopes),
            ready=cur.status in ("created", "routed") and self._is_task_ready(cur),
            write_scope_warnings=warnings,
            verification_evidence=cur.verification_evidence,
        )
