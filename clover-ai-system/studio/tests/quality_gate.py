#!/usr/bin/env python3
"""Clover Studio quality gate: honest offline-suite + live-server smoke checks.

Run from clover-ai-system/studio:
  python tests/quality_gate.py
  python tests/quality_gate.py --base-url http://127.0.0.1:8765 --live
  python tests/quality_gate.py --offline
  python tests/quality_gate.py --base-url http://127.0.0.1:8765 --live --playwright

The runner never sends a chat prompt, changes settings, grants permissions, or invokes tools.
Live checks are read-only. Playwright is optional and only verifies the served UI shell.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
)
REPORT_VERSION = 1


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
        proc = subprocess.run([sys.executable, str(path)], cwd=str(STUDIO), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
        output = proc.stdout[-12000:]
        report["checks"].append({
            "name": "offline:" + filename,
            "status": "PASS" if proc.returncode == 0 else "FAIL",
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "exit_code": proc.returncode,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="running Clover Studio base URL")
    parser.add_argument("--live", action="store_true", help="run read-only HTTP checks against a running Studio")
    parser.add_argument("--offline", action="store_true", help="run the existing offline regression suites")
    parser.add_argument("--playwright", action="store_true", help="run optional real-browser UI shell check (requires Playwright)")
    parser.add_argument("--json", dest="json_path", help="write a machine-readable report to this path")
    args = parser.parse_args()

    report = {"schema_version": REPORT_VERSION, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "base_url": args.base_url if args.live or args.playwright else None, "checks": [],
              "scope": {"offline": bool(args.offline), "live_http": bool(args.live),
                        "browser_shell": bool(args.playwright), "chat_or_tool_execution": False,
                        "image_quality_evaluation": False,
                        "note": "No real model generation or image-quality claim is made by this gate."}}
    if args.offline:
        run_offline(report)
    if args.live:
        run_live(args.base_url, report)
    if args.playwright:
        run_playwright(args.base_url, report)
    if not (args.offline or args.live or args.playwright):
        run_offline(report)

    counts = {state: sum(1 for c in report["checks"] if c["status"] == state) for state in ("PASS", "FAIL", "SKIP")}
    report["summary"] = counts
    report["overall"] = "FAIL" if counts["FAIL"] else ("PASS" if counts["PASS"] else "NO_CHECKS")
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
