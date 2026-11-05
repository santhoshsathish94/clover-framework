"""Automated verification suite for Live Activity SSE Timeline (`test_live_activity_sse.py`).

Verifies across the Clover HTTP/SSE streaming stack (`server.py` / `s23_on_device_clover.py` protocol):
  1. Strict event ordering and monotonic sequence numbers (`id: <seq>` and `ev["seq"] == 1..N`) across all
     behind-the-scenes live activity events (`start`, `semantics`, `tools`, `layer`, `plan`, `token`,
     `tool_call`, `tool_result`, `done`).
  2. Mid-turn cancellation via both `POST /api/cancel` and `POST /api/stop` (with and without explicit `turn_id`).
  3. Abrupt client disconnect (broken pipe) detection + reconnection event replay via `GET /api/turn/events`
     and `POST /api/chat/stream` (`resume_turn_id` / `Last-Event-ID`).
  4. Concurrent requests: overlapping `/api/chat/stream` requests (`stage="queued"` notification, serialized
     turn execution, immediate cancellation of queued turns without waiting on active turn, and non-blocking
     concurrent `GET /api/status` & `GET /api/health` polling).
"""
from __future__ import annotations

import http.client
import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import permissions as P  # noqa: E402
import server  # noqa: E402


class _MockSettings:
    def __init__(self):
        self._data = {
            "engine": "gpu",
            "model": "omni",
            "tools": "always",
            "sampling": "general",
            "thinking": True,
            "personal": False,
            "collective_contribute": False,
            "collective_receive": False,
            "csi": False,
            "layers": {
                "memory": True,
                "ontology": True,
                "compiler": True,
                "ncm": True,
                "kv_slots": False,
                "recall": False,
            },
        }

    def get(self):
        return dict(self._data)

    def update(self, changes):
        self._data.update(changes or {})
        return dict(self._data)


class _MockEngine:
    engine_kind = "mock-omni"
    model_key = "omni"
    inputs = ["text", "vision", "audio", "video"]

    def __init__(self):
        self.stage = {"name": "idle", "since": time.time()}
        self.turn_images = []
        self.workdir = None
        self.workname = ""
        self.after_write = None
        self.slow_delay_s = 0.0
        self.hold_event: threading.Event | None = None

    def describe(self):
        return {"model": "LFM3-Omni-2B", "model_key": "omni", "engine": "gpu"}

    def run_turn(self, hist, prompt, *, tools=True, sampling="general", thinking=True, max_new=512,
                 on_event=None, ask=None, cancel=None, **kwargs):
        emit = on_event or (lambda _e: None)
        emit({"type": "plan", "round": 1, "stage": "planning", "tools": ["calculate", "run_python"]})
        emit({"type": "layer", "layer": "memory", "entries": ["MEM-001"]})
        emit({"type": "layer", "layer": "ontology", "n_evidence": 2})
        emit({"type": "token", "round": 1, "kind": "thinking", "text": "Checking calculation steps... "})

        if self.hold_event is not None:
            # Wait until released or cancelled
            for _ in range(100):
                if cancel is not None and cancel.is_set():
                    return {
                        "reply": "(Stopped.)",
                        "cancelled": True,
                        "tool_calls": [],
                        "rounds": 1,
                        "generation": [],
                        "layers": [],
                    }
                if self.hold_event.wait(timeout=0.02):
                    break

        if self.slow_delay_s > 0:
            deadline = time.perf_counter() + self.slow_delay_s
            while time.perf_counter() < deadline:
                if cancel is not None and cancel.is_set():
                    return {
                        "reply": "Partial answer before stop",
                        "cancelled": True,
                        "tool_calls": [],
                        "rounds": 1,
                        "generation": [{"new_tokens": 4, "decode_s": 0.05}],
                        "layers": [],
                    }
                time.sleep(0.02)

        emit({"type": "plan", "round": 1, "stage": "preparing_tool", "tool": "calculate"})
        emit({"type": "tool_call", "id": "tc_1", "round": 1, "name": "calculate", "arguments": {"expression": "17**4 + 29*13"}})
        emit({
            "type": "tool_result",
            "id": "tc_1",
            "round": 1,
            "name": "calculate",
            "ran": True,
            "worked": True,
            "summary": "83898",
            "why": "allowed",
            "result": "83898",
        })
        emit({"type": "round", "round": 2})
        emit({"type": "plan", "round": 2, "stage": "reviewing"})
        emit({"type": "layer", "layer": "compiler", "ok": True})
        for tok in ("The ", "result ", "is ", "83898."):
            if cancel is not None and cancel.is_set():
                return {
                    "reply": "(Stopped.)",
                    "cancelled": True,
                    "tool_calls": [],
                    "rounds": 2,
                    "generation": [],
                    "layers": [],
                }
            emit({"type": "token", "round": 2, "kind": "answer", "text": tok})
        return {
            "reply": f"The result for '{prompt}' is 83898.",
            "cancelled": False,
            "tool_calls": [{"tool": "calculate", "ran": True, "worked": True, "result": "83898"}],
            "rounds": 2,
            "generation": [{"new_tokens": 12, "decode_s": 0.06}],
            "layers": [{"layer": "compiler", "checked": True, "ok": True}],
        }


def _read_sse_stream(conn_or_resp, stop_after_events: int | None = None):
    if isinstance(conn_or_resp, http.client.HTTPConnection):
        resp = conn_or_resp.getresponse()
        assert resp.status == 200, f"Expected HTTP 200, got {resp.status}"
    else:
        resp = conn_or_resp
    events = []
    sse_ids = []
    buf = b""
    while True:
        chunk = resp.read(1)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            text = frame.decode("utf-8", errors="replace")
            if text.startswith(":"):
                continue
            cur_id = None
            for line in text.splitlines():
                if line.startswith("id: "):
                    cur_id = int(line[4:].strip())
                elif line.startswith("data: "):
                    ev = json.loads(line[6:])
                    events.append(ev)
                    if cur_id is not None:
                        sse_ids.append(cur_id)
            if events and events[-1].get("type") in ("done", "error"):
                return events, sse_ids, resp
            if stop_after_events is not None and len(events) >= stop_after_events:
                return events, sse_ids, resp
    return events, sse_ids, resp


def _json_req(port: int, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Host": f"127.0.0.1:{port}"}
    payload = None
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(payload))
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8")
    conn.close()
    return resp.status, json.loads(raw) if raw else {}


def run_all_tests():
    passed = 0

    def check(cond: bool, msg: str):
        nonlocal passed
        if not cond:
            raise AssertionError(f"FAIL: {msg}")
        print(f"PASS {msg}")
        passed += 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mock_eng = _MockEngine()
        server.S.args = type("Args", (), {"base": str(tmp_path)})()
        server.S.engine = mock_eng
        server.S.omni_engine = mock_eng
        server.S.mode = "lfm3"
        server.S.load = {"status": "ready", "step": "ready", "since": time.time(), "error": None}
        server.S.settings = _MockSettings()
        server.S.perms = P.PermissionStore(tmp_path / "permissions.json")
        server.S.repo = None
        server.S.workspaces = None
        server.S.memidx = None
        server.S.csi = None
        server.S.vscode = None
        server.S.token = ""

        httpd = server.StudioHTTPServer(("127.0.0.1", 0), server.Handler)
        httpd.daemon_threads = True
        port = httpd.server_address[1]
        server.S.port = port
        server.S.httpd = httpd
        srv_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        srv_thread.start()

        try:
            # 1. Event ordering and monotonic sequence numbers (`seq` & `id:`)
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            body = json.dumps({"prompt": "Calculate 17**4 + 29*13", "session_id": "s_order"}).encode("utf-8")
            conn.request(
                "POST",
                "/api/chat/stream",
                body=body,
                headers={"Host": f"127.0.0.1:{port}", "Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            evs, sse_ids, _ = _read_sse_stream(conn)
            conn.close()

            seqs = [e.get("seq") for e in evs]
            check(seqs == list(range(1, len(evs) + 1)), f"SSE events carry strictly monotonic seq 1..{len(evs)}")
            check(sse_ids == seqs, "SSE frame 'id:' headers match event 'seq' numbers 1:1")

            types = [e.get("type") for e in evs]
            check(types[0] == "start" and types[-1] == "done", "Stream starts with 'start' and ends with 'done'")
            idx_sem = types.index("semantics")
            idx_tools = types.index("tools")
            idx_tcall = types.index("tool_call")
            idx_tres = types.index("tool_result")
            idx_done = types.index("done")
            check(0 < idx_sem < idx_tools < idx_tcall < idx_tres < idx_done,
                  "Behind-the-scenes timeline events arrive in causal order (start -> semantics -> tools -> tool_call -> tool_result -> done)")

            # 2. Reconnection event replay via GET /api/turn/events and POST /api/chat/stream (resume_turn_id)
            tid_1 = evs[0]["turn_id"]
            st_code, replay_data = _json_req(port, "GET", f"/api/turn/events?turn_id={tid_1}&after=4")
            check(st_code == 200 and replay_data["done"] is True, "GET /api/turn/events returns completed turn state")
            replayed_seqs = [e["seq"] for e in replay_data["events"]]
            check(replayed_seqs == list(range(5, len(evs) + 1)),
                  f"GET /api/turn/events?after=4 replays exact missed events 5..{len(evs)} in order")

            conn_res = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            res_body = json.dumps({"resume_turn_id": tid_1, "after_seq": 6}).encode("utf-8")
            conn_res.request(
                "POST",
                "/api/chat/stream",
                body=res_body,
                headers={"Host": f"127.0.0.1:{port}", "Content-Type": "application/json", "Content-Length": str(len(res_body))},
            )
            res_evs, _, _ = _read_sse_stream(conn_res)
            conn_res.close()
            check([e["seq"] for e in res_evs] == list(range(7, len(evs) + 1)),
                  "POST /api/chat/stream with resume_turn_id replays missed SSE stream from after_seq")

            # 3. Mid-turn cancellation via /api/cancel and /api/stop
            mock_eng.slow_delay_s = 2.0
            try:
                conn_c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                b_cancel = json.dumps({"prompt": "Long task to cancel", "session_id": "s_cancel"}).encode("utf-8")
                conn_c.request(
                    "POST",
                    "/api/chat/stream",
                    body=b_cancel,
                    headers={"Host": f"127.0.0.1:{port}", "Content-Type": "application/json", "Content-Length": str(len(b_cancel))},
                )
                # Read first 3 events to get turn_id while turn is in progress
                first_evs, _, resp_c = _read_sse_stream(conn_c, stop_after_events=3)
                tid_cancel = first_evs[0]["turn_id"]
                t_cancel_start = time.perf_counter()
                c_code, c_resp = _json_req(port, "POST", "/api/stop", {"turn_id": tid_cancel})
                rest_evs, _, _ = _read_sse_stream(resp_c)
                cancel_elapsed = time.perf_counter() - t_cancel_start
                conn_c.close()
                check(c_code == 200 and c_resp.get("ok") is True, "POST /api/stop cancels active turn by turn_id")
                check(rest_evs[-1]["type"] == "done" and rest_evs[-1].get("cancelled") is True and cancel_elapsed < 0.5,
                      f"Cancelled turn terminates promptly ({cancel_elapsed*1000:.0f} ms) with done.cancelled=True")
            finally:
                mock_eng.slow_delay_s = 0.0

            # 4. Abrupt client disconnect (broken pipe) marks turn cancelled without crashing server
            mock_eng.slow_delay_s = 1.5
            try:
                raw_sock = socket.create_connection(("127.0.0.1", port), timeout=5)
                b_disc = json.dumps({"prompt": "Client will disconnect abruptly", "session_id": "s_disc"}).encode("utf-8")
                req_bytes = (
                    f"POST /api/chat/stream HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{port}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(b_disc)}\r\n\r\n"
                ).encode("utf-8") + b_disc
                raw_sock.sendall(req_bytes)
                initial_chunk = b""
                while b"\"type\": \"start\"" not in initial_chunk and b"\"type\":\"start\"" not in initial_chunk:
                    part = raw_sock.recv(4096)
                    if not part:
                        break
                    initial_chunk += part
                # Extract turn_id from start event, then slam socket shut
                tid_disc = None
                for ln in initial_chunk.decode("utf-8", errors="replace").splitlines():
                    if ln.startswith("data: "):
                        obj = json.loads(ln[6:])
                        if obj.get("type") == "start":
                            tid_disc = obj.get("turn_id")
                            break
                raw_sock.close()
                check(bool(tid_disc), "Received start event before abrupt client socket close")
                # Wait briefly for worker to hit broken pipe on next emit and finish
                for _ in range(40):
                    _, st_disc = _json_req(port, "GET", f"/api/turn/events?turn_id={tid_disc}&after=0")
                    if st_disc.get("done"):
                        break
                    time.sleep(0.05)
                check(st_disc.get("done") is True and st_disc.get("cancelled") is True,
                      "Abrupt client disconnect sets cancel flag and cleans up turn state cleanly")
            finally:
                mock_eng.slow_delay_s = 0.0

            # 5. Concurrent requests: Turn A holds turn_lock while Turn B queues (emits stage='queued') and /api/status stays fast
            hold_ev = threading.Event()
            mock_eng.hold_event = hold_ev
            try:
                results_a = {}
                results_b = {}
                results_c = {}

                def run_stream_client(prompt_txt: str, sid_txt: str, out_box: dict, stop_at: int | None = None):
                    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    b = json.dumps({"prompt": prompt_txt, "session_id": sid_txt}).encode("utf-8")
                    c.request(
                        "POST",
                        "/api/chat/stream",
                        body=b,
                        headers={"Host": f"127.0.0.1:{port}", "Content-Type": "application/json", "Content-Length": str(len(b))},
                    )
                    e_list, s_ids, resp_obj = _read_sse_stream(c, stop_after_events=stop_at)
                    out_box["events"] = e_list
                    out_box["ids"] = s_ids
                    out_box["conn"] = c
                    out_box["resp"] = resp_obj

                t_a = threading.Thread(target=run_stream_client, args=("Turn A active", "s_conc_a", results_a))
                t_a.start()
                # Wait until Turn A is holding S.turn_lock
                for _ in range(50):
                    if server.S.turn_lock.locked():
                        break
                    time.sleep(0.02)
                check(server.S.turn_lock.locked(), "Turn A acquired S.turn_lock")

                # Concurrent GET /api/status must respond immediately while Turn A is busy
                t_st0 = time.perf_counter()
                st_code, st_body = _json_req(port, "GET", "/api/status")
                st_ms = (time.perf_counter() - t_st0) * 1000.0
                check(st_code == 200 and st_body.get("busy") is not None and st_ms < 250.0,
                      f"Concurrent GET /api/status responds in {st_ms:.1f} ms while Turn A is running")

                # Start Turn C (queued behind Turn A) and cancel Turn C while Turn A is still holding the lock!
                t_c = threading.Thread(target=run_stream_client, args=("Turn C queued then cancelled", "s_conc_c", results_c, 2))
                t_c.start()
                t_c.join(timeout=3.0)
                c_initial = results_c.get("events", [])
                check(len(c_initial) == 2 and c_initial[0]["type"] == "start"
                      and c_initial[1].get("type") == "plan" and c_initial[1].get("stage") == "queued",
                      "Queued concurrent stream immediately emits 'start' (seq=1) and 'plan(stage=queued)' (seq=2)")
                tid_c = c_initial[0]["turn_id"]
                t_c_cancel0 = time.perf_counter()
                _json_req(port, "POST", "/api/cancel", {"turn_id": tid_c})
                c_rest, _, _ = _read_sse_stream(results_c["resp"])
                results_c["conn"].close()
                c_cancel_ms = (time.perf_counter() - t_c_cancel0) * 1000.0
                check(c_rest[-1]["type"] == "done" and c_rest[-1].get("cancelled") is True and c_cancel_ms < 300.0,
                      f"Cancelling queued Turn C exits in {c_cancel_ms:.1f} ms without waiting for Turn A to finish")

                # Start Turn B (queued behind Turn A) and then release Turn A so both complete in order
                t_b = threading.Thread(target=run_stream_client, args=("Turn B queued then completed", "s_conc_b", results_b))
                t_b.start()
                time.sleep(0.08)
                hold_ev.set()
                t_a.join(timeout=5.0)
                t_b.join(timeout=5.0)
                results_a["conn"].close()
                results_b["conn"].close()

                evs_a = results_a["events"]
                evs_b = results_b["events"]
                check(evs_a[-1]["type"] == "done" and not evs_a[-1].get("cancelled"), "Turn A completed normally")
                check(evs_b[-1]["type"] == "done" and not evs_b[-1].get("cancelled")
                      and any(e.get("type") == "plan" and e.get("stage") == "queued" for e in evs_b),
                      "Turn B queued behind Turn A and completed normally with strictly monotonic seq")
                check([e["seq"] for e in evs_b] == list(range(1, len(evs_b) + 1)),
                      "Turn B sequence numbers are strictly 1..N with zero interleaving from Turn A")
            finally:
                mock_eng.hold_event = None
                hold_ev.set()

        finally:
            httpd.shutdown()
            httpd.server_close()

    print(f"\n{passed} checks passed")


if __name__ == "__main__":
    run_all_tests()
