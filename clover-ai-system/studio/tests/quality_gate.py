#!/usr/bin/env python3
"""Clover Studio Quality Engine: offline suites, isolated live model/tool/media E2E,
Playwright browser trace checks, automatic failure-layer diagnosis, and Markdown dashboard.

Run from clover-ai-system/studio:
  python tests/quality_gate.py --offline --json logs/quality/offline.json
  python tests/quality_gate.py --spawn-harness --live --playwright-e2e --screenshots logs/quality/shots
  python tests/quality_gate.py --spawn-isolated --seed-base C:\\Users\\santhoshn\\CloverAI --e2e --playwright-e2e \\
      --json logs/quality/e2e.json --markdown logs/quality/dashboard.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

STUDIO = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
OFFLINE_SUITES = (
    "test_semantics.py",
    "test_tool_routing.py",
    "test_clover_repo.py",
    "test_regression_fixes.py",
    "test_page_fetch.py",
    "test_image_gen.py",
    "test_image_gateway.py",
    "test_local_tool_packs.py",
    "test_xcruncher_tool.py",
    "test_hg4_engine.py",
)
REPORT_VERSION = 2


def request(url: str, timeout: float = 5.0):
    req = urllib.request.Request(url, headers={"Accept": "application/json,text/html"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read(2 * 1024 * 1024)
        return response.status, response.headers.get("Content-Type", ""), body, round((time.perf_counter() - started) * 1000, 1)


def run_offline(report):
    for filename in OFFLINE_SUITES:
        path = TESTS / filename
        if not path.is_file():
            report["checks"].append({"name": "offline:" + filename, "status": "SKIP", "reason": "suite not present in this checkout"})
            continue
        started = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, str(path)], cwd=str(STUDIO), text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
            output, exit_code = proc.stdout[-12000:], proc.returncode
            status = "PASS" if exit_code == 0 else "FAIL"
        except subprocess.TimeoutExpired as exc:
            raw = exc.stdout or ""
            output = raw.decode("utf-8", "replace")[-12000:] if isinstance(raw, bytes) else str(raw)[-12000:]
            exit_code, status = None, "FAIL"
            output += "\\nFAILED: suite exceeded the 900-second timeout."
        except OSError as exc:
            output, exit_code, status = str(exc), None, "FAIL"
        report["checks"].append({
            "name": "offline:" + filename,
            "status": status,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "exit_code": exit_code,
            "output_tail": output,
            "evidence_class": "offline_or_simulated",
        })


def run_live(base_url, report):
    base = base_url.rstrip("/")
    for path, expected in (("/", "text/html"), ("/api/health", "application/json"), ("/api/status", "application/json")):
        name = "live:" + path
        try:
            status, content_type, body, elapsed = request(base + path)
            ok = status == 200 and expected in content_type
            detail = {"http_status": status, "content_type": content_type, "duration_ms": elapsed,
                      "body_bytes": len(body)}
            if path != "/" and "json" in content_type:
                try:
                    payload = json.loads(body.decode("utf-8"))
                    # Keep only safe, operationally useful fields; never persist tokens or user data.
                    for key in ("status", "ready", "model", "version", "error"):
                        if key in payload and isinstance(payload[key], (str, int, float, bool, type(None))):
                            detail[key] = payload[key]
                except (UnicodeDecodeError, json.JSONDecodeError):
                    ok = False
                    detail["reason"] = "response advertised JSON but was not valid JSON"
            report["checks"].append({"name": name, "status": "PASS" if ok else "FAIL",
                                     "evidence_class": "live_http_read_only", **detail})
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            report["checks"].append({"name": name, "status": "FAIL", "evidence_class": "live_http_read_only",
                                     "reason": str(exc)[:500]})


def run_playwright(base_url, report):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        report["checks"].append({"name": "ui:playwright_shell", "status": "SKIP",
                                 "reason": "Install optional dependency: pip install playwright; then playwright install chromium"})
        return
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)[:500]))
            response = page.goto(base_url, wait_until="domcontentloaded", timeout=20000)
            title = page.title()
            body_text = page.locator("body").inner_text(timeout=5000)[:1000]
            browser.close()
        ok = response is not None and response.status == 200 and bool(body_text.strip()) and not errors
        report["checks"].append({"name": "ui:playwright_shell", "status": "PASS" if ok else "FAIL",
                                 "evidence_class": "live_browser_shell", "http_status": response.status if response else None,
                                 "title": title[:200], "page_errors": errors})
    except Exception as exc:
        report["checks"].append({"name": "ui:playwright_shell", "status": "FAIL",
                                 "evidence_class": "live_browser_shell", "reason": str(exc)[:1000]})


def record_unexercised_targets(report, requested_targets):
    """Record explicit `NOT_RUN` entries for configured device/remote targets that are offline or unreachable."""
    for tgt in requested_targets:
        if tgt == "android":
            adb = None
            for cand in (
                Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk" / "platform-tools" / "adb.exe",
                Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "platform-tools-dl" / "platform-tools" / "adb.exe",
            ):
                if cand.is_file():
                    adb = str(cand)
                    break
            adb = adb or __import__("shutil").which("adb")
            connected = False
            if adb:
                r = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=10)
                devs = [ln.split()[0] for ln in r.stdout.splitlines()[1:] if "\tdevice" in ln]
                connected = len(devs) > 0
            if not connected:
                report["checks"].append({
                    "name": "device:android_studio_app",
                    "area": "device",
                    "model": "android:omni",
                    "status": "NOT_RUN",
                    "evidence_class": "live_device_e2e",
                    "reason": "No online Android device/emulator attached over ADB (`adb devices` empty).",
                })
        elif tgt.startswith("remote:"):
            m_name = tgt.split(":", 1)[1]
            report["checks"].append({
                "name": f"remote:{m_name}",
                "area": "model-routing",
                "model": m_name,
                "status": "NOT_RUN",
                "evidence_class": "live_model_e2e",
                "reason": f"Hosted endpoint `{m_name}` requires live remote credentials/session; not claimed as PASS.",
            })


def main():
    if str(TESTS) not in sys.path:
        sys.path.insert(0, str(TESTS))
    from quality import dashboard, isolated_env, playwright_e2e

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="running Clover Studio base URL")
    parser.add_argument("--live", action="store_true", help="run read-only HTTP checks against a running Studio")
    parser.add_argument("--offline", action="store_true", help="run the offline regression suites")
    parser.add_argument("--playwright", action="store_true", help="run read-only real-browser UI shell check")
    parser.add_argument("--playwright-e2e", action="store_true", help="run interactive Playwright UI + reload + screenshot suite")
    parser.add_argument("--e2e", action="store_true", help="run the 14-scenario live model/tool/media/permission E2E suite")
    parser.add_argument("--spawn-isolated", action="store_true", help="launch a disposable isolated Studio server for live/E2E checks")
    parser.add_argument("--spawn-harness", action="store_true", help="launch a disposable isolated UI harness for browser shell checks")
    parser.add_argument("--seed-base", default=None, help="installed CloverAI dir to link read-only models/tools from (user dirs untouched)")
    parser.add_argument("--models", default="omni", help="comma-separated models to exercise in --e2e (e.g. omni,auto)")
    parser.add_argument("--check-targets", default="android,remote:clover-35b,remote:nemotron,remote:kimi",
                        help="additional device/remote targets to probe or record honestly as NOT_RUN")
    parser.add_argument("--screenshots", default=None, help="directory to save Playwright stage screenshots")
    parser.add_argument("--json", dest="json_path", help="write machine-readable JSON report to this path")
    parser.add_argument("--markdown", dest="md_path", help="write Markdown Quality Dashboard to this path")
    args = parser.parse_args()

    report = {
        "schema_version": REPORT_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url if (args.live or args.playwright or args.playwright_e2e or args.e2e) else None,
        "checks": [],
        "scope": {
            "offline": bool(args.offline),
            "live_http": bool(args.live),
            "browser_shell": bool(args.playwright or args.playwright_e2e),
            "chat_or_tool_execution": bool(args.e2e or args.playwright_e2e),
            "spawn_mode": "isolated_live" if args.spawn_isolated else ("harness" if args.spawn_harness else "external"),
        },
    }

    any_flag = any((args.offline, args.live, args.playwright, args.playwright_e2e, args.e2e, args.spawn_isolated, args.spawn_harness))
    if args.offline or not any_flag:
        run_offline(report)

    if args.spawn_isolated or args.spawn_harness:
        mode = "harness" if args.spawn_harness else "live"
        seed = Path(args.seed_base) if args.seed_base else None
        try:
            with isolated_env.IsolatedStudioServer(seed_base=seed, mode=mode) as srv:
                report["base_url"] = srv.base_url
                if args.live or mode == "live":
                    run_live(srv.base_url, report)
                if args.playwright:
                    run_playwright(srv.base_url, report)
                if args.playwright_e2e:
                    shot_dir = Path(args.screenshots) if args.screenshots else None
                    playwright_e2e.run_playwright_e2e(srv.base_url, report, screenshot_dir=shot_dir)
                if args.e2e and mode == "live":
                    client = isolated_env.StudioClient(srv.base_url)
                    for m_id in [m.strip() for m in args.models.split(",") if m.strip()]:
                        isolated_env.run_live_e2e_scenarios(client, srv.temp_root, report, model_id=m_id)
                untouched_ok, untouched_msg = srv.verify_seed_untouched()
                report["checks"].append({
                    "name": "safety:seed_base_user_data_untouched",
                    "area": "safety",
                    "status": "PASS" if untouched_ok else "FAIL",
                    "evidence_class": "runtime_boundary_check",
                    "detail": untouched_msg,
                })
        except Exception as exc:
            report["checks"].append({
                "name": f"spawn:{mode}_server",
                "area": "infrastructure",
                "status": "FAIL",
                "evidence_class": "live_http_read_only",
                "reason": str(exc)[:800],
            })
    else:
        if args.live:
            run_live(args.base_url, report)
        if args.playwright:
            run_playwright(args.base_url, report)
        if args.playwright_e2e:
            shot_dir = Path(args.screenshots) if args.screenshots else None
            playwright_e2e.run_playwright_e2e(args.base_url, report, screenshot_dir=shot_dir)
        if args.e2e:
            tmp_ws = Path(tempfile.mkdtemp(prefix="clover_qa_ext_"))
            try:
                client = isolated_env.StudioClient(args.base_url)
                for m_id in [m.strip() for m in args.models.split(",") if m.strip()]:
                    isolated_env.run_live_e2e_scenarios(client, tmp_ws, report, model_id=m_id)
            finally:
                __import__("shutil").rmtree(tmp_ws, ignore_errors=True)

    if args.e2e and args.check_targets:
        record_unexercised_targets(report, [t.strip() for t in args.check_targets.split(",") if t.strip()])

    dash = dashboard.compute_metrics(report["checks"])
    report["dashboard"] = dash
    counts = dash["counts"]
    report["summary"] = counts
    report["overall"] = "FAIL" if counts["FAIL"] else ("PASS" if counts["PASS"] else "NO_CHECKS")

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
    if args.md_path:
        md_out = Path(args.md_path)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(dashboard.render_markdown_dashboard(report), encoding="utf-8")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
