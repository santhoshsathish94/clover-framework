"""Disposable Isolated Studio Environment, SSE Turn Driver & Real E2E Scenario Runner.

Safety contract:
  - All sessions, memory repositories, settings, permissions, cache, and media live inside a
    disposable temporary `--base` directory (`%TEMP%/clover-qa-isolated-*`).
  - Only read-only `data/` (model weights) and `tools/` (`llama.cpp`, `ffmpeg`, `git`) are
    linked via OS junctions/symlinks from `--seed-base` (e.g., `C:\\Users\\santhoshn\\CloverAI`).
  - A before/after metadata snapshot of `--seed-base`'s user state (`clover-ai-studio`, `vault`,
    `personal`, `collective`, `config`) verifies mathematically that zero user files were touched.
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib
from pathlib import Path
from typing import Any

from . import evaluator

STUDIO_DIR = Path(__file__).resolve().parents[2]
CLOVER_DIR = STUDIO_DIR / "clover"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _snapshot_user_dirs(seed_base: Path | None) -> dict[str, tuple[int, float]]:
    snap: dict[str, tuple[int, float]] = {}
    if not seed_base or not seed_base.is_dir():
        return snap
    for sub in ("clover-ai-studio", "vault", "personal", "collective", "config"):
        d = seed_base / sub
        if not d.exists():
            continue
        cnt = 0
        latest = 0.0
        for p in d.rglob("*"):
            try:
                st = p.lstat()
                cnt += 1
                if st.st_mtime > latest:
                    latest = st.st_mtime
            except OSError:
                continue
        snap[sub] = (cnt, round(latest, 3))
    return snap


def _link_readonly_dir(src: Path, dst: Path) -> None:
    if not src.is_dir() or dst.exists():
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                       check=True, capture_output=True)
    else:
        os.symlink(str(src), str(dst), target_is_directory=True)


def _unlink_junction_safe(p: Path) -> None:
    if not p.exists() and not p.is_symlink():
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "rmdir", str(p)], capture_output=True)
    else:
        try:
            p.unlink()
        except OSError:
            pass


def _terminate_process_tree_by_pid(root_pid: int) -> None:
    """Terminate a spawned subprocess and its child PIDs by explicit PID inspection."""
    if root_pid <= 0:
        return
    if os.name == "nt":
        ps_cmd = (
            f"$pids = @({root_pid}); "
            f"$kids = Get-CimInstance Win32_Process -Filter \"ParentProcessId = {root_pid}\" -ErrorAction SilentlyContinue; "
            f"foreach ($k in $kids) {{ "
            f"  $gk = Get-CimInstance Win32_Process -Filter \"ParentProcessId = $($k.ProcessId)\" -ErrorAction SilentlyContinue; "
            f"  foreach ($g in $gk) {{ $pids += $g.ProcessId }}; "
            f"  $pids += $k.ProcessId "
            f"}}; "
            f"foreach ($id in ($pids | Select-Object -Unique)) {{ Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }}"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True)
    else:
        try:
            import signal
            os.kill(root_pid, signal.SIGTERM)
        except OSError:
            pass


class IsolatedStudioServer:
    """Context manager launching a disposable, isolated Studio server (`live` or `harness`)."""

    def __init__(self, *, seed_base: Path | None = None, mode: str = "live", startup_timeout_s: float = 180.0):
        self.seed_base = Path(seed_base) if seed_base else None
        self.mode = mode  # "live" (real server.py + real models) or "harness" (ui_harness.py)
        self.startup_timeout_s = startup_timeout_s
        self.temp_root: Path | None = None
        self.port = find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None
        self._pre_snap: dict[str, tuple[int, float]] = {}

    def __enter__(self) -> "IsolatedStudioServer":
        self._pre_snap = _snapshot_user_dirs(self.seed_base)
        self.temp_root = Path(tempfile.mkdtemp(prefix="clover_qa_iso_"))
        if self.seed_base and self.seed_base.is_dir():
            _link_readonly_dir(self.seed_base / "data", self.temp_root / "data")
            _link_readonly_dir(self.seed_base / "tools", self.temp_root / "tools")

        if self.mode == "harness":
            cmd = [sys.executable, str(STUDIO_DIR / "tests" / "ui_harness.py"),
                   "--base", str(self.temp_root), "--port", str(self.port)]
        else:
            # Configure clean isolated Studio defaults (`c1_inference_only: False` uses primary LFM3-Omni-2B generation)
            cfg_dir = self.temp_root / "config"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "settings.json").write_text(json.dumps({
                "model": "omni",
                "tools": "ask",
                "c1_inference_only": False,
            }), encoding="utf-8")
            cmd = [sys.executable, str(CLOVER_DIR / "server.py"),
                   "--base", str(self.temp_root), "--port", str(self.port), "--idle-exit-min", "15"]

        self._log_path = Path(tempfile.gettempdir()) / f"clover_qa_srv_{self.port}.log"
        self._log_fh = open(self._log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=str(STUDIO_DIR), stdout=self._log_fh, stderr=subprocess.STDOUT)

        deadline = time.time() + self.startup_timeout_s
        last_err = "server did not respond"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self._log_fh.flush()
                tail = self._log_path.read_text(encoding="utf-8", errors="replace")[-1200:]
                raise RuntimeError(f"Studio server exited early (code={self.proc.returncode}): {tail}")
            try:
                req = urllib.request.Request(f"{self.base_url}/api/status", headers={"Origin": self.base_url})
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    st = json.loads(resp.read().decode("utf-8"))
                    if st.get("status") == "ready":
                        return self
                    if st.get("status") == "error":
                        raise RuntimeError(f"Studio engine startup error: {st.get('error')}")
                    last_err = f"status={st.get('status')} step={st.get('load_step')}"
            except RuntimeError:
                raise
            except Exception as exc:
                last_err = str(exc)
            time.sleep(1.5)
        raise RuntimeError(f"Timed out waiting for isolated Studio readiness ({last_err})")

    def verify_seed_untouched(self) -> tuple[bool, str]:
        if not self.seed_base:
            return True, "no seed_base used"
        post = _snapshot_user_dirs(self.seed_base)
        if post != self._pre_snap:
            return False, f"seed_base user state changed: before={self._pre_snap} after={post}"
        return True, "seed_base user directories 100% untouched"

    def __exit__(self, *exc_info: object) -> None:
        if self.proc is not None:
            pid = self.proc.pid
            _terminate_process_tree_by_pid(pid)
            try:
                self.proc.wait(timeout=8)
            except Exception:
                pass
        if getattr(self, "_log_fh", None):
            try:
                self._log_fh.close()
            except OSError:
                pass
        if getattr(self, "_log_path", None):
            try:
                self._log_path.unlink(missing_ok=True)
            except OSError:
                pass
        if self.temp_root and self.temp_root.is_dir():
            _unlink_junction_safe(self.temp_root / "data")
            _unlink_junction_safe(self.temp_root / "tools")
            shutil.rmtree(self.temp_root, ignore_errors=True)


class StudioClient:
    """HTTP + SSE client that drives a running Studio instance exactly like the browser UI."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _json_req(self, method: str, path: str, body: dict | None = None, timeout: float = 30.0) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Origin": self.base_url, "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read(8 * 1024 * 1024).decode("utf-8"))

    def status(self) -> dict:
        return self._json_req("GET", "/api/status")

    def set_permissions(self, perms: dict[str, str]) -> dict:
        return self._json_req("POST", "/api/permissions", {"permissions": perms})

    def set_settings(self, changes: dict[str, Any]) -> dict:
        return self._json_req("POST", "/api/settings", changes)

    def stream_turn(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        media: list[dict] | None = None,
        attached_files: list[dict] | None = None,
        permission_policy: str = "once",
        smart_switch: bool | None = None,
        timeout_s: float = 180.0,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
        sid = session_id or f"qa-{uuid.uuid4().hex[:12]}"
        use_smart = getattr(self, "default_smart_switch", False) if smart_switch is None else bool(smart_switch)
        payload: dict[str, Any] = {"session_id": sid, "prompt": prompt, "smart_switch": use_smart}
        if media:
            payload["media"] = media
        if attached_files:
            payload["attached_files"] = attached_files

        req = urllib.request.Request(
            self.base_url + "/api/chat/stream",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Origin": self.base_url, "Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        events: list[dict[str, Any]] = []
        done_res: dict[str, Any] = {}
        streamed_answer: list[str] = []
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                if time.perf_counter() - t0 > timeout_s:
                    raise TimeoutError(f"SSE stream exceeded {timeout_s}s")
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                ev = json.loads(data_str)
                events.append(ev)
                ev_type = ev.get("type")
                if ev_type == "token" and ev.get("kind") == "answer":
                    streamed_answer.append(str(ev.get("text") or ""))
                elif ev_type == "permission":
                    tid = ev.get("turn_id")
                    rid = ev.get("request_id")
                    if tid and rid:
                        self._json_req("POST", "/api/decision", {
                            "turn_id": tid,
                            "request_id": rid,
                            "decision": permission_policy,
                        })
                elif ev_type == "done":
                    done_res = ev.get("result") if isinstance(ev.get("result"), dict) else dict(ev)
                    break
                elif ev_type == "error":
                    raise RuntimeError(str(ev.get("error") or ev))

        if "reply" not in done_res:
            done_res["reply"] = "".join(streamed_answer)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return done_res, events, elapsed_ms


def _make_scene_png_bytes() -> bytes:
    """Create a 128x128 RGB PNG showing a bright RED left half and bright BLUE right half on white border."""
    w, h = 128, 128
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            if y < 12 or y >= h - 12 or x < 12 or x >= w - 12:
                row.extend((255, 255, 255))
            elif x < w // 2:
                row.extend((225, 25, 25))
            else:
                row.extend((25, 75, 230))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def _chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


def _synthesize_spoken_wav(out_wav: Path, spoken_text: str) -> bool:
    """Synthesize real spoken English 16 kHz mono PCM WAV on Windows via System.Speech (zero cloud call)."""
    if os.name != "nt":
        return False
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
        f"$s.SetOutputToWaveFile('{str(out_wav)}', $fmt); "
        f"$s.Speak('{spoken_text}'); "
        "$s.Dispose()"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
    return r.returncode == 0 and out_wav.is_file() and out_wav.stat().st_size > 2000


def run_live_e2e_scenarios(
    client: StudioClient,
    isolated_base: Path,
    report: dict[str, Any],
    *,
    model_id: str = "omni",
) -> None:
    """Execute the real-world E2E scenario suite against `client` and append structured findings to `report['checks']`."""
    st_info = client.status()
    model_inputs = st_info.get("inputs") or ["text"]
    img_caps = st_info.get("image_capabilities") or {}
    media_root = isolated_base / "cache" / "media"
    alt_media_root = isolated_base / "media"
    ws_dir = isolated_base / "qa_workspace"
    ws_dir.mkdir(parents=True, exist_ok=True)

    client.default_smart_switch = (model_id == "auto")
    client.set_settings({"model": model_id, "tools": "ask"})
    client.set_permissions({
        "code": "allow",
        "files": "allow",
        "edit_files": "allow",
        "internet": "allow",
        "shell": "allow",
        "memory": "allow",
        "media": "allow",
    })

    def _record(sid: str, area: str, ev_class: str, ok: bool, ms: float, expected: str, observed: str,
                sse: list[dict], res: dict, *, explicit_layer: str = "") -> None:
        trace = evaluator.extract_Execution_trace(sse, res)
        entry: dict[str, Any] = {
            "name": f"e2e:{sid}",
            "area": area,
            "model": model_id,
            "status": "PASS" if ok else "FAIL",
            "duration_ms": ms,
            "evidence_class": ev_class,
            "trace": trace,
        }
        if not ok:
            entry["diagnosis"] = evaluator.diagnose_failure(
                sid, expected=expected, observed=observed, trace=trace, explicit_layer=explicit_layer
            )
        report["checks"].append(entry)

    # 1 & 2. Conversation + Follow-Up Context Retention
    chat_sid = f"qa-conv-{uuid.uuid4().hex[:8]}"
    try:
        res1, sse1, ms1 = client.stream_turn(
            "We have 140 solar panels, each rated at 450 watts. What is the total array capacity in kilowatts?",
            session_id=chat_sid,
        )
        ok1 = bool(res1.get("reply")) and evaluator.contains_number_close(res1["reply"], 63.0, tol=0.2)
        _record("chat.basic", "conversation", "live_model_e2e", ok1, ms1,
                "Reply computes 140 * 450 W = 63 kW", str(res1.get("reply") or "")[:300], sse1, res1)

        res2, sse2, ms2 = client.stream_turn(
            "If we add 20 more of those exact same panels to that array, show the step-by-step calculation and the new total in kilowatts.",
            session_id=chat_sid,
        )
        ok2 = bool(res2.get("reply")) and evaluator.contains_number_close(res2["reply"], 72.0, tol=0.2)
        _record("chat.followup", "followup", "live_model_e2e", ok2, ms2,
                "Follow-up retains 63 kW & 450 W context -> 72 kW", str(res2.get("reply") or "")[:300], sse2, res2)
    except Exception as exc:
        _record("chat.basic", "conversation", "live_model_e2e", False, 0.0, "63 kW", str(exc), [], {}, explicit_layer="transport")

    # 3. Memory Save & Cross-Session Recall
    try:
        mem_code = "ORCHID-739"
        res_m1, sse_m1, ms_m1 = client.stream_turn(
            f"Please remember that my project codename is {mem_code}.",
            session_id=f"qa-memA-{uuid.uuid4().hex[:6]}",
        )
        res_m2, sse_m2, ms_m2 = client.stream_turn(
            "What is my project codename that you remembered?",
            session_id=f"qa-memB-{uuid.uuid4().hex[:6]}",
        )
        ok_mem = "orchid-739" in str(res_m2.get("reply") or "").lower()
        _record("memory.save_recall", "memory", "live_tool_e2e", ok_mem, round(ms_m1 + ms_m2, 1),
                f"Cross-session recall returns {mem_code}", str(res_m2.get("reply") or "")[:300], sse_m1 + sse_m2, res_m2,
                explicit_layer="" if ok_mem else "persistence")
    except Exception as exc:
        _record("memory.save_recall", "memory", "live_tool_e2e", False, 0.0, "ORCHID-739 recalled", str(exc), [], {}, explicit_layer="persistence")

    # 4. Research (Local Deterministic Fact Page via fetch_url)
    class _FactHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            html = (b"<html><body><h1>Helios Observatory Report</h1>"
                    b"<p>The telescope mirror diameter is 8.42 meters and first light occurred in 2024.</p></body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

    fact_srv = http.server.HTTPServer(("127.0.0.1", 0), _FactHandler)
    fact_port = fact_srv.server_address[1]
    threading.Thread(target=fact_srv.serve_forever, daemon=True).start()
    try:
        url_fact = f"http://127.0.0.1:{fact_port}/observatory.html"
        res_w, sse_w, ms_w = client.stream_turn(
            f"Read the page {url_fact} and tell me the mirror diameter in meters."
        )
        ok_w = evaluator.contains_number_close(res_w.get("reply", ""), 8.42, tol=0.02)
        _record("research.web_page", "research", "live_tool_e2e", ok_w, ms_w,
                "Retrieves 8.42 meters from local page via fetch_url", str(res_w.get("reply") or "")[:300], sse_w, res_w)
    except Exception as exc:
        _record("research.web_page", "research", "live_tool_e2e", False, 0.0, "8.42 meters", str(exc), [], {}, explicit_layer="tool")
    finally:
        fact_srv.shutdown()
        fact_srv.server_close()

    # 5. Files: Read, Modify & Bystander Integrity
    try:
        target_f = ws_dir / "release_config.txt"
        bystander_f = ws_dir / "do_not_touch.txt"
        target_f.write_text("CHANNEL=alpha\nPORT=9010\n", encoding="utf-8")
        bystander_f.write_text("IMMUTABLE_GUARD=true\n", encoding="utf-8")
        bystander_sha = hashlib.sha256(bystander_f.read_bytes()).hexdigest()

        res_f, sse_f, ms_f = client.stream_turn(
            f"Read {target_f} and overwrite {target_f} so CHANNEL=production while keeping PORT=9010."
        )
        new_txt = target_f.read_text(encoding="utf-8", errors="replace")
        post_bystander_sha = hashlib.sha256(bystander_f.read_bytes()).hexdigest()
        ok_f = ("CHANNEL=production" in new_txt and "PORT=9010" in new_txt and post_bystander_sha == bystander_sha)
        _record("files.read_modify", "files", "live_tool_e2e", ok_f, ms_f,
                "target file updated to CHANNEL=production & PORT=9010; bystander unchanged",
                f"file_content={new_txt!r} bystander_unchanged={post_bystander_sha == bystander_sha}", sse_f, res_f)
    except Exception as exc:
        _record("files.read_modify", "files", "live_tool_e2e", False, 0.0, "CHANNEL=production", str(exc), [], {}, explicit_layer="tool")

    # 6. Python Calculation & Dataset Analysis
    try:
        res_py, sse_py, ms_py = client.stream_turn(
            "Use Python to compute the sum of squares of all prime numbers strictly below 50, and state the exact integer."
        )
        # Primes < 50: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47 -> sum(p*p) = 10970
        exp_sum_sq = sum(p * p for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47))
        ok_py = evaluator.contains_number_close(res_py.get("reply", ""), float(exp_sum_sq), tol=0.1)
        _record("python.analyze", "python", "live_tool_e2e", ok_py, ms_py,
                f"Exact sum of prime squares < 50 = {exp_sum_sq}", str(res_py.get("reply") or "")[:300], sse_py, res_py)
    except Exception as exc:
        _record("python.analyze", "python", "live_tool_e2e", False, 0.0, "10970", str(exc), [], {}, explicit_layer="tool")

    # 7. Spreadsheet Regional Revenue + Bar Chart PNG Verification
    try:
        sales_csv = ws_dir / "regional_sales.csv"
        sales_csv.write_text(
            "region,quarter,revenue\n"
            "North,Q1,2100.50\nNorth,Q2,2150.00\n"
            "South,Q1,1580.25\nSouth,Q2,1600.00\n"
            "West,Q1,2800.00\nWest,Q2,2820.00\n",
            encoding="utf-8",
        )
        res_sp, sse_sp, ms_sp = client.stream_turn(
            f"Group by region and sum revenue in {sales_csv}, and also plot a bar chart of {sales_csv} with x_column=region and y_column=revenue."
        )
        imgs = res_sp.get("images") or []
        chart_ok = False
        if imgs:
            fname = imgs[0].get("file", "")
            cpath = (media_root / fname) if (media_root / fname).is_file() else (alt_media_root / fname)
            png_info = evaluator.decode_png_summary(cpath)
            chart_ok = bool(png_info.get("valid") and (png_info.get("entropy_bits") or 0) > 0.2)
        nums_ok = (
            evaluator.contains_number_close(res_sp.get("reply", ""), 5620.0, tol=0.5)
            and evaluator.contains_number_close(res_sp.get("reply", ""), 4250.5, tol=0.5)
        )
        _record("spreadsheet.revenue", "spreadsheet", "live_tool_e2e", (nums_ok and chart_ok), ms_sp,
                "Exact regional revenue sums (West=5620, North=4250.5) + valid chart PNG",
                f"nums_ok={nums_ok} chart_ok={chart_ok} reply={str(res_sp.get('reply') or '')[:240]}", sse_sp, res_sp)
    except Exception as exc:
        _record("spreadsheet.revenue", "spreadsheet", "live_tool_e2e", False, 0.0, "Regional sums + chart", str(exc), [], {}, explicit_layer="tool")

    # 8. Exact Big-Math & Zero-Storage Pi via x-cruncher
    try:
        res_xc, sse_xc, ms_xc = client.stream_turn(
            "Use the calculator to evaluate pi_hex(1000000, 20) and tell me the exact 20 hexadecimal digits of pi at offset 1000000."
        )
        ok_xc = "6c65e52cb459350050e4" in str(res_xc.get("reply") or "").lower()
        _record("math.xcruncher", "exact-math", "live_tool_e2e", ok_xc, ms_xc,
                "Exact 20 hex digits at offset 1,000,000 = 6c65e52cb459350050e4",
                str(res_xc.get("reply") or "")[:300], sse_xc, res_xc)
    except Exception as exc:
        _record("math.xcruncher", "exact-math", "live_tool_e2e", False, 0.0, "6c65e52cb459350050e4", str(exc), [], {}, explicit_layer="tool")

    # 9. Coding: Fix a Bug & Independently Run Unit Tests
    try:
        code_f = ws_dir / "pricing.py"
        test_f = ws_dir / "test_pricing.py"
        code_f.write_text(
            "def apply_discount(price: float, pct: float) -> float:\n"
            "    # BUG: adds percentage instead of subtracting it\n"
            "    return round(price * (1.0 + pct / 100.0), 2)\n",
            encoding="utf-8",
        )
        test_f.write_text(
            "from pricing import apply_discount\n"
            "assert apply_discount(200.0, 15.0) == 170.0\n"
            "assert apply_discount(50.0, 0.0) == 50.0\n"
            "print('PRICING_TESTS_OK')\n",
            encoding="utf-8",
        )
        res_cd, sse_cd, ms_cd = client.stream_turn(
            f"Read {code_f}, fix the bug in apply_discount so it subtracts the discount percentage instead of adding it, and save {code_f}."
        )
        chk = subprocess.run([sys.executable, str(test_f)], cwd=str(ws_dir), capture_output=True, text=True, timeout=15)
        ok_cd = chk.returncode == 0 and "PRICING_TESTS_OK" in chk.stdout
        _record("coding.fix_and_test", "coding", "live_tool_e2e", ok_cd, ms_cd,
                "test_pricing.py exits 0 independently after fix",
                f"rc={chk.returncode} out={chk.stdout.strip()!r} err={chk.stderr.strip()[:160]!r}", sse_cd, res_cd)
    except Exception as exc:
        _record("coding.fix_and_test", "coding", "live_tool_e2e", False, 0.0, "test_pricing.py passes", str(exc), [], {}, explicit_layer="tool")

    # 10. Permissions Gate: Ask -> Deny Blocks Execution, Ask -> Allow Once Permits Execution
    try:
        guard_f = ws_dir / "secret_gate.txt"
        guard_f.write_text("GATE_PAYLOAD_991\n", encoding="utf-8")
        client.set_permissions({"files": "ask"})
        res_den, sse_den, ms_den = client.stream_turn(
            f"Read the file {guard_f} and tell me what is inside.",
            permission_policy="deny",
        )
        trace_den = evaluator.extract_Execution_trace(sse_den, res_den)
        prompt_shown = len(trace_den["permission_prompts"]) >= 1
        denied_clean = "gate_payload_991" not in str(res_den.get("reply") or "").lower()

        res_alw, sse_alw, ms_alw = client.stream_turn(
            f"Read the file {guard_f} and tell me what is inside.",
            permission_policy="once",
        )
        allowed_ok = "gate_payload_991" in str(res_alw.get("reply") or "").lower()
        client.set_permissions({"files": "allow"})
        _record("tools.permission", "permissions", "live_tool_e2e",
                (prompt_shown and denied_clean and allowed_ok), round(ms_den + ms_alw, 1),
                "Permission prompt emitted; deny prevents file read; once allows file read",
                f"prompt_shown={prompt_shown} denied_blocked={denied_clean} once_allowed={allowed_ok}",
                sse_den + sse_alw, res_alw, explicit_layer="" if (prompt_shown and denied_clean and allowed_ok) else "permissions")
    except Exception as exc:
        client.set_permissions({"files": "allow"})
        _record("tools.permission", "permissions", "live_tool_e2e", False, 0.0, "Ask/Deny/Once enforced", str(exc), [], {}, explicit_layer="permissions")

    # 11. Image Understanding (Vision)
    if "vision" in model_inputs or "image" in model_inputs:
        try:
            png_bytes = _make_scene_png_bytes()
            b64_img = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
            res_v, sse_v, ms_v = client.stream_turn(
                "Look at this image and tell me the two main colors shown side by side.",
                media=[{"kind": "image", "name": "red_blue_split.png", "data": b64_img}],
            )
            ok_v, miss_v = evaluator.contains_all_keywords(res_v.get("reply", ""), [("red",), ("blue",)])
            _record("vision.understand", "image-understanding", "live_media_e2e", ok_v, ms_v,
                    "Identifies both red and blue regions in attached image",
                    f"missing={miss_v} reply={str(res_v.get('reply') or '')[:260]}", sse_v, res_v)
        except Exception as exc:
            _record("vision.understand", "image-understanding", "live_media_e2e", False, 0.0, "red & blue", str(exc), [], {}, explicit_layer="media_pipeline")
    else:
        report["checks"].append({
            "name": "e2e:vision.understand", "area": "image-understanding", "model": model_id,
            "status": "NOT_RUN", "evidence_class": "live_media_e2e",
            "reason": f"Selected model ({model_id}) does not advertise vision input ({model_inputs}).",
        })

    # 12. Image Generation Capability & Honest Gate
    has_gen_img = bool(
        (img_caps.get("high_quality_api") or {}).get("available")
        or (img_caps.get("local_diffusion") or {}).get("sd15_lcm_available")
        or (img_caps.get("local_diffusion") or {}).get("sd15_available")
    )
    try:
        res_ig, sse_ig, ms_ig = client.stream_turn(
            "Generate a detailed image of a stone temple beside a calm pond at sunrise."
        )
        imgs_ig = res_ig.get("images") or []
        if has_gen_img:
            ig_ok = False
            obs_ig = "no image returned"
            if imgs_ig:
                im0 = imgs_ig[0]
                p_ig = (media_root / im0["file"]) if (media_root / im0["file"]).is_file() else (alt_media_root / im0["file"])
                dec = evaluator.decode_png_summary(p_ig)
                ig_ok = bool(dec.get("valid") and im0.get("is_generative") is True and (dec.get("entropy_bits") or 0) > 1.0)
                obs_ig = f"img={im0} decode={dec}"
            _record("image.generate", "image-generation", "live_media_e2e", ig_ok, ms_ig,
                    "Valid generative PNG with high visual entropy & provenance", obs_ig, sse_ig, res_ig)
        else:
            # No generative diffusion/API installed: MUST NOT silently return a fake procedural image!
            no_fake_img = len(imgs_ig) == 0
            _record("image.generate", "image-generation", "live_media_e2e", no_fake_img, ms_ig,
                    "Honest unavailability reported when no generative model is installed (zero fake canvas PNGs)",
                    f"images_returned={imgs_ig} reply={str(res_ig.get('reply') or '')[:240]}",
                    sse_ig, res_ig, explicit_layer="" if no_fake_img else "backend_capability")
    except Exception as exc:
        _record("image.generate", "image-generation", "live_media_e2e", False, 0.0, "Honest image gate", str(exc), [], {}, explicit_layer="backend_capability")

    # 13. Audio Voice Instruction (Real Synthesized Spoken WAV)
    wav_path = ws_dir / "spoken_q.wav"
    if ("audio" in model_inputs) and _synthesize_spoken_wav(wav_path, "What is seven multiplied by nine? Answer with the number."):
        try:
            b64_wav = "data:audio/wav;base64," + base64.b64encode(wav_path.read_bytes()).decode("ascii")
            res_au, sse_au, ms_au = client.stream_turn(
                "",
                media=[{"kind": "audio", "name": "spoken_q.wav", "data": b64_wav}],
            )
            ok_au = evaluator.contains_number_close(res_au.get("reply", ""), 63.0, tol=0.1) or ("sixty-three" in str(res_au.get("reply") or "").lower())
            _record("audio.voice_instruction", "audio", "live_media_e2e", ok_au, ms_au,
                    "Spoken question '7 * 9' answered with 63", str(res_au.get("reply") or "")[:260], sse_au, res_au)
        except Exception as exc:
            _record("audio.voice_instruction", "audio", "live_media_e2e", False, 0.0, "63", str(exc), [], {}, explicit_layer="media_pipeline")
    else:
        report["checks"].append({
            "name": "e2e:audio.voice_instruction", "area": "audio", "model": model_id,
            "status": "NOT_RUN", "evidence_class": "live_media_e2e",
            "reason": "Model lacks audio input or host speech synthesizer unavailable.",
        })

    # 14. Video Understanding (2-Second Synthesized MP4 via FFmpeg)
    sys.path.insert(0, str(CLOVER_DIR))
    import local_tool_packs  # type: ignore
    ff_exe = local_tool_packs.find_media_binary(isolated_base, "ffmpeg")
    mp4_path = ws_dir / "green_clip.mp4"
    if ("video" in model_inputs or "vision" in model_inputs) and ff_exe:
        try:
            subprocess.run(
                [ff_exe, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=green:s=256x256:d=2:r=5",
                 "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(mp4_path)],
                check=True, capture_output=True, timeout=20,
            )
            b64_vid = "data:video/mp4;base64," + base64.b64encode(mp4_path.read_bytes()).decode("ascii")
            res_vd, sse_vd, ms_vd = client.stream_turn(
                "What single solid color fills this short video clip?",
                media=[{"kind": "video", "name": "green_clip.mp4", "data": b64_vid}],
            )
            ok_vd = "green" in str(res_vd.get("reply") or "").lower()
            _record("video.events", "video", "live_media_e2e", ok_vd, ms_vd,
                    "Identifies solid green color in attached MP4 video", str(res_vd.get("reply") or "")[:260], sse_vd, res_vd)
        except Exception as exc:
            _record("video.events", "video", "live_media_e2e", False, 0.0, "green video", str(exc), [], {}, explicit_layer="media_pipeline")
    else:
        report["checks"].append({
            "name": "e2e:video.events", "area": "video", "model": model_id,
            "status": "NOT_RUN", "evidence_class": "live_media_e2e",
            "reason": "FFmpeg or video/vision model input not available in this environment.",
        })
