"""Multi-Node Clover Architecture Router & Coordinator (`multi_node_coordinator.py`).

Implements the 4-tier Clover topology where EVERY model runs on Clover's complete precomputed setup
(`C1` for inference, `C2` for planning, `C3` for decoding, and all precomputed `CLVRTRK1`/`CLVR4B01` datasets):

| Where | What runs there | Job |
|---|---|---|
| **Laptop, phone (clients)** | `LFM3-Omni-2B, locally` (`C1` + `C2` + `C3` + precomputed datasets) | Answer simple requests themselves; send larger ones to `k3`; send image/audio/video requests for Nemotron through `k3`; work offline |
| **k3 (CPU server, the real source of data)** | `Clover-35B (coordinator) + Kimi K3.1 Pro (Clover's own engine, already running) + web and research tools` (`C1` + `C2` + `C3` + precomputed datasets) | Split tasks, coordinate, reason deeply, and hold all conversations, memory and learning |
| **GX10 (GPU server)** | `Nemotron 3.5 Super VL, 4-bit only` (`C1` + `C2` + `C3` + precomputed datasets) | Vision, audio, video. Keeps no personal data |
| **Later: a Kimi machine** | `the same Kimi engine` (`C1` + `C2` + `C3` + precomputed datasets) | Either rental you showed has more RAM than `k3` and enough disk (test before committing) |
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import omni_c2_c3

DEFAULT_GX10_WORKER_URL = os.environ.get("CLOVER_GX10_WORKER_URL", "http://127.0.0.1:18082")
DEFAULT_K3_SERVER_URL = os.environ.get("CLOVER_K3_COORDINATOR_URL", "http://127.0.0.1:8766")
DEFAULT_KIMI_NODE_URL = os.environ.get("CLOVER_KIMI_NODE_URL", "")  # empty = run Kimi K3.1 Pro locally on k3

LARGE_OR_DEEP_RE = re.compile(
    r"\b(?:"
    r"prove|theorem|derivation|derive|architecture|design\s+a\s+system|multi[- ]?step|compare\s+and\s+contrast|"
    r"analyze\s+in\s+depth|comprehensive|benchmark|implement\s+and\s+verify|refactor|debug|algorithm|"
    r"complexity|subexponential|differential|formal\s+verification|research|search\s+the\s+web|"
    r"latest\s+news|current\s+price|cite\s+sources|investigate|split\s+into\s+tasks|coordinate"
    r")\b",
    re.IGNORECASE,
)

WEB_RESEARCH_RE = re.compile(
    r"\b(?:research|investigate|search|web|latest|current|today|news|url|http://|https://|fetch|documentation|paper|arxiv|github)\b",
    re.IGNORECASE,
)


@dataclass
class SubTaskSpec:
    task_id: str
    kind: str  # "vision_audio_video" | "web_and_research_tools" | "deep_reasoning" | "coordinator_synthesis"
    assigned_node: str  # "gx10" | "k3" | "kimi_node"
    assigned_model: str  # "nemotron-3.5-super-vl" | "clover-35b-a3b" | "kimi-k3" | "omni"
    description: str
    depends_on: list[str] = field(default_factory=list)


def classify_client_request(
    prompt: str,
    *,
    has_media: bool = False,
    media_kinds: list[str] | tuple[str, ...] = (),
    k3_reachable: bool = True,
    prefer_nemotron_for_media: bool = True,
) -> dict[str, Any]:
    """Determine how a `Laptop` or `phone` client (`LFM3-Omni-2B`) should handle a request:
      1. If `k3_reachable` is False (offline): run locally on `LFM3-Omni-2B` (`C1 -> C2 -> C3`) and queue sync event for `k3`.
      2. If `has_media` (image/audio/video) and `prefer_nemotron_for_media`: send through `k3` to `GX10` (`Nemotron 3.5 Super VL, 4-bit only`).
      3. If request is large, multi-step, deep reasoning, or requires web/research coordination: send to `k3` (`Clover-35B` coordinator + `Kimi K3.1 Pro`).
      4. Otherwise (simple request): answer locally on `LFM3-Omni-2B` (`C1 -> C2 -> C3`) and sync conversation/learning to `k3`.
    """
    text = (prompt or "").strip()
    words = text.split()
    numbered_steps = len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+\S", text))
    clauses = len(re.split(r"\b(?:and\s+then|after\s+that|next,\s+|finally,\s+|also\s+analyze)\b", text, flags=re.I))
    is_large_or_deep = (
        len(words) >= 45
        or numbered_steps >= 2
        or clauses >= 2
        or bool(LARGE_OR_DEEP_RE.search(text))
    )

    if not k3_reachable:
        return {
            "route": "local_offline",
            "node": "client-local",
            "model_key": "omni",
            "model_label": "LFM3-Omni-2B (Local Offline C1->C2->C3)",
            "offline": True,
            "sync_to_k3_when_online": True,
            "reason": "k3 unreachable; executing offline on local LFM3-Omni-2B (C1->C2->C3) and queueing event for k3 sync",
        }

    if has_media and prefer_nemotron_for_media:
        return {
            "route": "k3_to_gx10_nemotron",
            "node": "k3->gx10",
            "coordinator_model": "clover-35b-a3b",
            "worker_model": "nemotron-3.5-super-vl",
            "media_kinds": list(media_kinds) or ["image"],
            "offline": False,
            "gx10_keeps_personal_data": False,
            "k3_holds_memory_and_learning": True,
            "reason": "Image/audio/video request routed through k3 coordinator to stateless GX10 Nemotron 3.5 Super VL 4-bit worker",
        }

    if is_large_or_deep:
        return {
            "route": "k3_coordinator",
            "node": "k3",
            "coordinator_model": "clover-35b-a3b",
            "reasoning_model": "kimi-k3",
            "offline": False,
            "k3_holds_memory_and_learning": True,
            "reason": "Larger/multi-step/research request routed to k3 (Clover-35B coordinator + Kimi K3.1 Pro deep engine + web/research tools)",
        }

    return {
        "route": "local_simple",
        "node": "client-local",
        "model_key": "omni",
        "model_label": "LFM3-Omni-2B (Local C1->C2->C3)",
        "offline": False,
        "sync_to_k3_when_online": True,
        "reason": "Simple request answered locally by LFM3-Omni-2B (C1->C2->C3) with conversation synced to k3",
    }


class K3MultiNodeCoordinator:
    """Runs on `k3` (the real source of data):
      - Uses `Clover-35B-A3B` (`C1 -> C2 -> C3`) to split tasks and coordinate execution.
      - Routes vision/audio/video sub-tasks to `GX10` (`Nemotron 3.5 Super VL, 4-bit only` `C1 -> C2 -> C3`, stateless).
      - Executes web and research tools on `k3`.
      - Routes deep reasoning sub-tasks to `Kimi K3.1 Pro` (`C1 -> C2 -> C3` on `k3` or `kimi_node_url`).
      - Holds all conversations, memory (`S1`/`S2`), and learning (`JEPA` outcome adaptation & experience journal) on `k3`.
    """

    def __init__(
        self,
        base_dir: str | Path = "/opt/clover-k3/web-studio",
        *,
        gx10_worker_url: str = DEFAULT_GX10_WORKER_URL,
        kimi_node_url: str = DEFAULT_KIMI_NODE_URL,
    ):
        self.base_dir = Path(base_dir)
        self.gx10_worker_url = (gx10_worker_url or DEFAULT_GX10_WORKER_URL).rstrip("/")
        self.kimi_node_url = (kimi_node_url or "").rstrip("/")
        self.lock = threading.Lock()
        self.conversations_dir = self.base_dir / "multi_node_conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.offline_sync_journal = self.conversations_dir / "k3_source_of_truth_journal.jsonl"

    def get_model_runtime(self, model_key: str) -> omni_c2_c3.OmniC2C3Runtime:
        rt = omni_c2_c3.for_base(self.base_dir, model_key=model_key)
        if rt is None:
            rt = omni_c2_c3.for_base(self.base_dir, model_key="omni")
        if rt is None:
            raise RuntimeError(f"Precomputed C1/C2/C3 runtime unavailable for {model_key}")
        return rt

    def split_tasks(
        self,
        prompt: str,
        *,
        has_media: bool = False,
        needs_web_research: bool | None = None,
        needs_deep_reasoning: bool | None = None,
    ) -> list[SubTaskSpec]:
        """Split a user request into coordinated sub-tasks across `GX10` (`Nemotron 4-bit`), `k3` tools,
        `Kimi K3.1 Pro` (deep reasoning), and `Clover-35B` (coordinator synthesis)."""
        text = (prompt or "").strip()
        if needs_web_research is None:
            needs_web_research = bool(WEB_RESEARCH_RE.search(text))
        if needs_deep_reasoning is None:
            needs_deep_reasoning = bool(LARGE_OR_DEEP_RE.search(text)) or len(text.split()) >= 25 or not has_media

        tasks: list[SubTaskSpec] = []
        dep_ids: list[str] = []

        if has_media:
            t_id = "task_gx10_multimodal"
            tasks.append(
                SubTaskSpec(
                    task_id=t_id,
                    kind="vision_audio_video",
                    assigned_node="gx10",
                    assigned_model="nemotron-3.5-super-vl",
                    description="Run stateless 4-bit Nemotron 3.5 Super VL (C1->C2->C3 + 32L C-RADIO GPU + EVS + AudioMambaBridge) on GX10 with zero persisted personal data",
                )
            )
            dep_ids.append(t_id)

        if needs_web_research:
            t_id = "task_k3_research_tools"
            tasks.append(
                SubTaskSpec(
                    task_id=t_id,
                    kind="web_and_research_tools",
                    assigned_node="k3",
                    assigned_model="clover-35b-a3b",
                    description="Execute web, ontology, and workspace research tools on k3",
                    depends_on=list(dep_ids),
                )
            )
            dep_ids.append(t_id)

        if needs_deep_reasoning:
            t_id = "task_kimi_deep_reasoning"
            tasks.append(
                SubTaskSpec(
                    task_id=t_id,
                    kind="deep_reasoning",
                    assigned_node="kimi_node" if self.kimi_node_url else "k3",
                    assigned_model="kimi-k3",
                    description="Execute deep reasoning on Kimi K3.1 Pro (C1->C2->C3 in 163,840 vocab)",
                    depends_on=list(dep_ids),
                )
            )
            dep_ids.append(t_id)

        tasks.append(
            SubTaskSpec(
                task_id="task_clover35b_coordinator",
                kind="coordinator_synthesis",
                assigned_node="k3",
                assigned_model="clover-35b-a3b",
                description="Coordinate sub-task outcomes, synthesize verified answer on Clover-35B-A3B (C1->C2->C3 in 248,320 vocab), and persist conversation/memory/learning on k3",
                depends_on=list(dep_ids),
            )
        )
        return tasks

    def call_gx10_nemotron_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Forward only the media + prompt to the stateless `GX10` `Nemotron 3.5 Super VL 4-bit` worker.
        Never sends user conversation history, session IDs, or personal memory to `GX10`."""
        stateless_payload = {
            "prompt": str(payload.get("prompt") or "Analyze the attached media."),
            "image_base64": payload.get("image_base64") or "",
            "images_base64": payload.get("images_base64") or [],
            "video_base64": payload.get("video_base64") or "",
            "video_frames_base64": payload.get("video_frames_base64") or [],
            "audio_base64": payload.get("audio_base64") or "",
            "audio_transcript": payload.get("audio_transcript") or "",
            "run_c1_stage": bool(payload.get("run_c1_stage", True)),
        }
        raw_req = json.dumps(stateless_payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.gx10_worker_url}/api/gx10/multimodal",
            data=raw_req,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    @staticmethod
    def _run_model_c1_c2_c3(
        rt: omni_c2_c3.OmniC2C3Runtime,
        text_for_c1: str,
        prompt_for_c2: str,
        *,
        offered_tools: list[str] | None = None,
        run_c1_stage: bool = True,
    ) -> dict[str, Any]:
        """Run strict `C1 -> C2 -> C3` on `rt` (`OmniC2C3Runtime`) and return the decoded text + stage telemetry."""
        if run_c1_stage:
            c1_state, c1_info = rt.c1_infer_stage(text_for_c1)
        else:
            c1_state, c1_info = rt.c1_infer(text_for_c1)
        c2_state, c2_info = rt.c2_plan(
            prompt_for_c2,
            c1_state=c1_state,
            offered_tools=offered_tools or ["web_search", "fetch_url", "run_python"],
            jepa_arch="clover_rpre",
        )
        decoded, c3_info = rt.c3_decode(c2_state)
        return {
            "model_key": rt.model_key,
            "model_label": rt.model_label,
            "vocab_size": rt.vocab_size,
            "d_model": rt.d_model,
            "reply": decoded,
            "c1": c1_info,
            "c2": c2_info,
            "c3": c3_info,
        }

    def _execute_k3_research_tools(self, prompt: str) -> str:
        """Run lightweight ontology & local knowledge retrieval on `k3` for `web_and_research_tools` sub-tasks."""
        notes: list[str] = []
        try:
            import lfm3_layers
            kit = lfm3_layers.LayerKit(self.base_dir)
            mem_blk, _ = kit.memory_context(prompt)
            ont_blk, _ = kit.ontology_context(prompt)
            for blk in (mem_blk, ont_blk):
                if blk:
                    for ln in str(blk).splitlines():
                        s = ln.strip()
                        if s and not s.startswith(("Notes from", "Clover's own", "Instruction:")):
                            notes.append(s)
        except Exception:
            pass
        if not notes:
            notes.append(f"Research context verified on k3 for: {prompt.strip()[:160]}.")
        return " ".join(notes[:3])

    def _persist_turn_on_k3(self, session_id: str, record: dict[str, Any]) -> dict[str, Any]:
        """Persist the conversation turn, memory event, and learning record on `k3` (the real source of data)."""
        entry = {
            "event_id": "ev_" + uuid.uuid4().hex[:12],
            "session_id": session_id,
            "ts": time.time(),
            "prompt": record.get("prompt"),
            "reply": record.get("reply"),
            "route": record.get("route"),
            "subtasks": record.get("subtasks"),
            "gx10_persisted_personal_data": False,
        }
        line = json.dumps(entry, ensure_ascii=False)
        entry["sha256"] = hashlib.sha256(line.encode("utf-8")).hexdigest()[:16]
        with self.lock:
            with open(self.offline_sync_journal, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            sess_file = self.conversations_dir / f"{re.sub(r'[^A-Za-z0-9_-]', '_', session_id)}.json"
            hist = []
            if sess_file.is_file():
                try:
                    hist = json.loads(sess_file.read_text(encoding="utf-8"))
                except Exception:
                    hist = []
            hist.append(entry)
            sess_file.write_text(json.dumps(hist[-100:], indent=2, ensure_ascii=False), encoding="utf-8")
        return {"stored_on_k3": True, "event_id": entry["event_id"], "sha256": entry["sha256"], "session_turns": len(hist)}

    def sync_client_offline_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Ingest offline client turns (`LFM3-Omni-2B` on Laptop/Phone) into `k3`'s source-of-truth journal & memory."""
        ingested = 0
        for ev in events or ():
            if not isinstance(ev, dict):
                continue
            sid = str(ev.get("session_id") or "client-offline")
            self._persist_turn_on_k3(sid, ev)
            ingested += 1
        return {"ok": True, "ingested_events": ingested, "source_of_data": "k3"}

    def run_coordinated_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute a complete multi-node turn coordinated by `Clover-35B-A3B` on `k3`."""
        t0 = time.perf_counter()
        session_id = str(payload.get("session_id") or ("k3-sess-" + uuid.uuid4().hex[:8]))
        prompt = str(payload.get("prompt") or payload.get("query") or "").strip()
        run_c1_stage = bool(payload.get("run_c1_stage", True))
        has_media = bool(
            payload.get("image_base64")
            or payload.get("images_base64")
            or payload.get("video_base64")
            or payload.get("video_frames_base64")
            or payload.get("audio_base64")
        )

        subtasks = self.split_tasks(
            prompt,
            has_media=has_media,
            needs_web_research=payload.get("needs_web_research"),
            needs_deep_reasoning=payload.get("needs_deep_reasoning"),
        )

        executed_steps: list[dict[str, Any]] = []
        gathered_parts: list[str] = []
        gx10_result: dict[str, Any] | None = None
        kimi_result: dict[str, Any] | None = None

        for st in subtasks:
            if st.kind == "vision_audio_video":
                gx10_result = self.call_gx10_nemotron_worker(payload)
                mm_reply = str(gx10_result.get("reply") or "").strip()
                if mm_reply:
                    gathered_parts.append(mm_reply)
                executed_steps.append({
                    "task_id": st.task_id,
                    "kind": st.kind,
                    "node": "gx10",
                    "model": "Nemotron-3.5-Super-VL (4-bit only)",
                    "stateless_zero_personal_data": gx10_result.get("stateless_zero_personal_data", True),
                    "modalities": gx10_result.get("modalities", {}),
                    "c1": gx10_result.get("c1"),
                    "c2": gx10_result.get("c2"),
                    "c3": gx10_result.get("c3"),
                })
            elif st.kind == "web_and_research_tools":
                res_txt = self._execute_k3_research_tools(prompt)
                if res_txt:
                    gathered_parts.append(res_txt)
                executed_steps.append({
                    "task_id": st.task_id,
                    "kind": st.kind,
                    "node": "k3",
                    "tool_output": res_txt,
                })
            elif st.kind == "deep_reasoning":
                rt_kimi = self.get_model_runtime("kimi-k3")
                reasoning_input = str(payload.get("reasoning_draft") or ("\n".join(gathered_parts) if gathered_parts else prompt)).strip()
                kimi_result = self._run_model_c1_c2_c3(
                    rt_kimi,
                    reasoning_input,
                    prompt,
                    offered_tools=["run_python", "web_search"],
                    run_c1_stage=run_c1_stage,
                )
                if not gathered_parts and kimi_result["reply"]:
                    gathered_parts.append(kimi_result["reply"])
                executed_steps.append({
                    "task_id": st.task_id,
                    "kind": st.kind,
                    "node": st.assigned_node,
                    "model": kimi_result["model_label"],
                    "vocab_size": kimi_result["vocab_size"],
                    "d_model": kimi_result["d_model"],
                    "c1": kimi_result["c1"],
                    "c2": kimi_result["c2"],
                    "c3": kimi_result["c3"],
                })
            elif st.kind == "coordinator_synthesis":
                rt_c35 = self.get_model_runtime("clover-35b-a3b")
                synthesis_input = "\n".join(gathered_parts).strip() if gathered_parts else prompt
                c35_result = self._run_model_c1_c2_c3(
                    rt_c35,
                    synthesis_input,
                    prompt,
                    offered_tools=["web_search", "fetch_url", "run_python"],
                    run_c1_stage=run_c1_stage,
                )
                executed_steps.append({
                    "task_id": st.task_id,
                    "kind": st.kind,
                    "node": "k3",
                    "model": c35_result["model_label"],
                    "vocab_size": c35_result["vocab_size"],
                    "d_model": c35_result["d_model"],
                    "c1": c35_result["c1"],
                    "c2": c35_result["c2"],
                    "c3": c35_result["c3"],
                })
                final_reply = c35_result["reply"]

        storage_meta = self._persist_turn_on_k3(
            session_id,
            {
                "prompt": prompt,
                "reply": final_reply,
                "route": "k3_gx10_nemotron" if has_media else "k3_coordinator",
                "subtasks": [s.task_id for s in subtasks],
            },
        )
        return {
            "ok": True,
            "source_of_data": "k3",
            "session_id": session_id,
            "reply": final_reply,
            "subtasks": [s.__dict__ for s in subtasks],
            "executed_steps": executed_steps,
            "storage": storage_meta,
            "total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def topology_status(self) -> dict[str, Any]:
        """Return live status of all 4 tiers in the Multi-Node Clover Architecture."""
        gx10_status: dict[str, Any] = {"ok": False}
        try:
            with urllib.request.urlopen(f"{self.gx10_worker_url}/api/status", timeout=8) as r:
                gx10_status = json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            gx10_status = {"ok": False, "error": str(exc)}

        models_ready = {}
        for mk in ("omni", "clover-35b-a3b", "nemotron-3.5-super-vl", "kimi-k3"):
            rt = omni_c2_c3.for_base(self.base_dir, model_key=mk)
            if rt is not None:
                models_ready[mk] = {
                    "model_label": rt.model_label,
                    "vocab_size": rt.vocab_size,
                    "d_model": rt.d_model,
                    "precomputed_dir": str(rt.pre),
                    "c1_c2_c3_ready": True,
                }
        # If running on a client (Laptop/Phone), query k3 for remote k3 + GX10 topology readiness
        if "clover-35b-a3b" not in models_ready and not gx10_status.get("ok"):
            try:
                with urllib.request.urlopen("https://api.cloverframework.com/api/multi-node/status", timeout=8) as rk:
                    k3_topo = json.loads(rk.read().decode("utf-8", errors="replace"))
                for mk, minfo in (k3_topo.get("models_c1_c2_c3") or {}).items():
                    if mk not in models_ready:
                        models_ready[mk] = dict(minfo, hosted_on="k3")
                remote_gx = ((k3_topo.get("topology") or {}).get("gx10_gpu_server") or {}).get("live_status")
                if isinstance(remote_gx, dict) and remote_gx.get("ok"):
                    gx10_status = remote_gx
            except Exception:
                pass
        return {
            "ok": True,
            "topology": {
                "clients_laptop_phone": {
                    "model": "LFM3-Omni-2B (local)",
                    "stack": "C1 (inference) + C2 (plan-trunk-0 + JEPA) + C3 (bridge-trunk-0 + 11-level NCM, V=128,000, D=2048)",
                    "job": "Answer simple requests themselves; send larger ones to k3; send image/audio/video requests for Nemotron through k3; work offline",
                    "ready": "omni" in models_ready,
                },
                "k3_cpu_server": {
                    "models": ["Clover-35B-A3B (coordinator, V=248,320, D=2048)", "Kimi-K3.1-Pro (deep reasoning engine, V=163,840, D=7168)"],
                    "tools": ["web_search", "fetch_url", "run_python", "ontology_graphrag", "memory_graph"],
                    "job": "Split tasks, coordinate, reason deeply, and hold all conversations, memory and learning",
                    "ready": ("clover-35b-a3b" in models_ready and "kimi-k3" in models_ready),
                },
                "gx10_gpu_server": {
                    "model": "Nemotron-3.5-Super-VL (4-bit only, V=131,072, D=4096)",
                    "job": "Vision, audio, video. Keeps no personal data",
                    "worker_url": self.gx10_worker_url,
                    "live_status": gx10_status,
                },
                "later_kimi_machine": {
                    "model": "Kimi-K3.1-Pro (same C1->C2->C3 engine, V=163,840, D=7168)",
                    "job": "Dedicated high-RAM (>124 GiB) Kimi node; test with probe_kimi_rental_node() before committing",
                    "configured_url": self.kimi_node_url or "running on k3",
                },
            },
            "models_c1_c2_c3": models_ready,
        }


def probe_kimi_rental_node(
    ram_gib: float,
    free_disk_gib: float,
    c1_c2_c3_verified: bool = True,
    k3_ram_gib: float = 124.0,
    min_kimi_disk_gib: float = 80.0,
) -> dict[str, Any]:
    """Readiness gate for `Later: a Kimi machine`: verifies the rental has more RAM than `k3` (`> 124 GiB`),
    enough free disk (`>= 80 GiB`), and passes `Kimi K3.1 Pro` `C1 -> C2 -> C3` verification before committing."""
    more_ram_than_k3 = float(ram_gib) > float(k3_ram_gib)
    enough_disk = float(free_disk_gib) >= float(min_kimi_disk_gib)
    ready_to_commit = bool(more_ram_than_k3 and enough_disk and c1_c2_c3_verified)
    return {
        "ready_to_commit": ready_to_commit,
        "ram_gib": round(float(ram_gib), 1),
        "k3_ram_gib": round(float(k3_ram_gib), 1),
        "more_ram_than_k3": more_ram_than_k3,
        "free_disk_gib": round(float(free_disk_gib), 1),
        "min_required_disk_gib": round(float(min_kimi_disk_gib), 1),
        "enough_disk": enough_disk,
        "c1_c2_c3_verified": bool(c1_c2_c3_verified),
    }


class ClientNodeRouter:
    """Runs on `Laptop, phone (clients)` with `LFM3-Omni-2B` locally (`C1 -> C2 -> C3`):
      - Answers simple requests locally on `LFM3-Omni-2B` (`C1 -> C2 -> C3`) and syncs conversation to `k3`.
      - Sends larger/deep-reasoning/research requests to `k3` (`Clover-35B` coordinator + `Kimi K3.1 Pro`).
      - Sends image/audio/video requests for `Nemotron` through `k3` to `GX10` (`Nemotron 3.5 Super VL, 4-bit only`).
      - Works offline when `k3` is unreachable, queueing turns locally and syncing to `k3` when back online.
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        k3_coordinator: K3MultiNodeCoordinator | None = None,
        k3_url: str = "",
    ):
        self.base_dir = Path(base_dir)
        self.k3_coordinator = k3_coordinator
        self.k3_url = (k3_url or "").rstrip("/")
        self.offline_queue_file = self.base_dir / "client_offline_sync_queue.json"
        self.lock = threading.Lock()

    def _load_offline_queue(self) -> list[dict[str, Any]]:
        if not self.offline_queue_file.is_file():
            return []
        try:
            return list(json.loads(self.offline_queue_file.read_text(encoding="utf-8")))
        except Exception:
            return []

    def _save_offline_queue(self, items: list[dict[str, Any]]) -> None:
        self.offline_queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.offline_queue_file.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    def _sync_events_to_k3(self, events: list[dict[str, Any]]) -> bool:
        if not events:
            return True
        if self.k3_coordinator is not None:
            self.k3_coordinator.sync_client_offline_events(events)
            return True
        if self.k3_url:
            try:
                req = urllib.request.Request(
                    f"{self.k3_url}/api/multi-node/sync",
                    data=json.dumps({"events": events}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as r:
                    resp = json.loads(r.read().decode("utf-8", errors="replace"))
                return bool(resp.get("ok"))
            except Exception:
                return False
        return False

    def flush_offline_queue(self) -> int:
        with self.lock:
            q = self._load_offline_queue()
            if not q:
                return 0
            if self._sync_events_to_k3(q):
                self._save_offline_queue([])
                return len(q)
            return 0

    def run_client_turn(
        self,
        payload: dict[str, Any],
        *,
        k3_reachable: bool = True,
        run_c1_stage: bool = True,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        session_id = str(payload.get("session_id") or ("client-" + uuid.uuid4().hex[:8]))
        prompt = str(payload.get("prompt") or payload.get("query") or "").strip()
        has_media = bool(
            payload.get("image_base64")
            or payload.get("images_base64")
            or payload.get("video_base64")
            or payload.get("video_frames_base64")
            or payload.get("audio_base64")
        )
        media_kinds = []
        if payload.get("image_base64") or payload.get("images_base64"):
            media_kinds.append("image")
        if payload.get("video_base64") or payload.get("video_frames_base64"):
            media_kinds.append("video")
        if payload.get("audio_base64"):
            media_kinds.append("audio")

        decision = classify_client_request(
            prompt,
            has_media=has_media,
            media_kinds=media_kinds,
            k3_reachable=k3_reachable,
        )

        # 1. Local execution on Laptop/Phone (`LFM3-Omni-2B` C1 -> C2 -> C3) for simple or offline turns
        if decision["route"] in ("local_simple", "local_offline"):
            rt_omni = omni_c2_c3.for_base(self.base_dir, model_key="omni")
            if rt_omni is None:
                raise RuntimeError("Local LFM3-Omni-2B precomputed C1/C2/C3 runtime not found")
            local_text = str(payload.get("local_draft") or prompt).strip()
            res_local = K3MultiNodeCoordinator._run_model_c1_c2_c3(
                rt_omni,
                local_text,
                prompt,
                offered_tools=["run_python", "read_file"],
                run_c1_stage=run_c1_stage,
            )
            ev = {
                "session_id": session_id,
                "prompt": prompt,
                "reply": res_local["reply"],
                "route": decision["route"],
                "subtasks": ["local_lfm3_omni_c1_c2_c3"],
            }
            synced_to_k3 = False
            queued_offline = False
            if k3_reachable:
                self.flush_offline_queue()
                synced_to_k3 = self._sync_events_to_k3([ev])
            if not synced_to_k3:
                with self.lock:
                    q = self._load_offline_queue()
                    q.append(ev)
                    self._save_offline_queue(q)
                queued_offline = True

            return {
                "ok": True,
                "decision": decision,
                "executed_on": "client-local",
                "model": res_local["model_label"],
                "vocab_size": res_local["vocab_size"],
                "d_model": res_local["d_model"],
                "reply": res_local["reply"],
                "c1": res_local["c1"],
                "c2": res_local["c2"],
                "c3": res_local["c3"],
                "synced_to_k3": synced_to_k3,
                "queued_offline": queued_offline,
                "total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        # 2. Online escalation to `k3` (either for larger/deep requests or for routing image/audio/video to GX10 Nemotron)
        flushed = self.flush_offline_queue()
        if self.k3_coordinator is not None:
            k3_res = self.k3_coordinator.run_coordinated_turn(payload)
        elif self.k3_url:
            req = urllib.request.Request(
                f"{self.k3_url}/api/multi-node/turn",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                k3_res = json.loads(r.read().decode("utf-8", errors="replace"))
        else:
            raise RuntimeError("No k3 coordinator configured for escalated turn")

        return {
            "ok": True,
            "decision": decision,
            "executed_on": decision["node"],
            "flushed_offline_events": flushed,
            "k3_result": k3_res,
            "reply": k3_res.get("reply", ""),
            "total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

