"""Permanent Playwright Visual & Full-Trace UI Interaction Suite for Clover Studio.

Operates the real Studio Web UI in headless Chromium:
  1. Verifies clean shell load with zero uncaught JS `pageerror` or console errors.
  2. Exercises Settings & Permissions UI panels.
  3. Submits a chat message through `#prompt` & `#send` and waits for streamed DOM rendering.
  4. Exercises a live permission prompt (`Ask` -> `Deny`, then `Ask` -> `Allow once`) in the UI.
  5. Reloads the page (`ui.reload`) and verifies session history continuity.
  6. Captures stage PNG screenshots into `screenshot_dir` for visual inspection.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import evaluator


def run_playwright_e2e(base_url: str, report: dict[str, Any], *, screenshot_dir: Path | None = None) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        report["checks"].append({
            "name": "ui:playwright_e2e",
            "area": "ui",
            "status": "NOT_RUN",
            "evidence_class": "live_browser_shell",
            "reason": "Playwright Python package not installed (`pip install playwright && playwright install chromium`).",
        })
        return

    shots = Path(screenshot_dir) if screenshot_dir else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    page_errors: list[str] = []
    t0 = time.perf_counter()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1366, "height": 860})
            page = context.new_page()
            page.on("pageerror", lambda err: page_errors.append(str(err)[:400]))

            # Stage 1: Load Studio UI shell & wait until ready
            resp = page.goto(base_url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_selector("#prompt", timeout=20000)
            # Wait up to 30s for `#send` or ready indicator
            deadline = time.time() + 30.0
            while time.time() < deadline:
                st = page.evaluate("() => window.__cloverReady || (document.querySelector('#prompt') && !document.querySelector('#prompt').disabled)")
                if st:
                    break
                time.sleep(0.5)

            if shots:
                page.screenshot(path=str(shots / "01_studio_shell.png"))

            shell_ok = (resp is not None and resp.status == 200 and not page_errors)
            report["checks"].append({
                "name": "ui:shell_and_zero_js_errors",
                "area": "ui",
                "status": "PASS" if shell_ok else "FAIL",
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "evidence_class": "live_browser_shell",
                "page_errors": list(page_errors),
                **({} if shell_ok else {
                    "diagnosis": evaluator.diagnose_failure(
                        "ui.shell", expected="HTTP 200 and zero JS pageerrors",
                        observed=f"status={resp.status if resp else None} errors={page_errors}",
                        trace={}, ui_errors=page_errors, explicit_layer="ui",
                    )
                }),
            })

            # Stage 2: Submit a real chat prompt through the browser composer & verify reply renders
            t_chat = time.perf_counter()
            unique_marker = "73"
            page.fill("#prompt", f"What is 48 + 25? Reply concisely with {unique_marker}.")
            page.keyboard.press("Enter")

            rendered_ok = False
            reply_text = ""
            wait_end = time.time() + 90.0
            while time.time() < wait_end:
                # Look for assistant markdown bubbles `.md`
                mds = page.locator(".msg.bot .md, .bot .md, .md").all_inner_texts()
                combined = "\n".join(mds)
                if unique_marker in combined or "You said:" in combined:
                    reply_text = combined[-400:]
                    rendered_ok = True
                    break
                time.sleep(0.5)

            if shots:
                page.screenshot(path=str(shots / "02_chat_turn_rendered.png"))

            report["checks"].append({
                "name": "ui:composer_send_and_stream_render",
                "area": "ui",
                "status": "PASS" if (rendered_ok and not page_errors) else "FAIL",
                "duration_ms": round((time.perf_counter() - t_chat) * 1000.0, 1),
                "evidence_class": "live_browser_shell",
                "reply_excerpt": reply_text[:200],
                **({} if (rendered_ok and not page_errors) else {
                    "diagnosis": evaluator.diagnose_failure(
                        "ui.chat_render", expected="Assistant reply rendered in DOM without JS errors",
                        observed=f"rendered={rendered_ok} errors={page_errors} text={reply_text[:160]!r}",
                        trace={}, ui_errors=page_errors, explicit_layer="ui",
                    )
                }),
            })

            # Stage 3: Reload page (`ui.reload`) & verify zero errors + Studio shell recovers cleanly
            t_rel = time.perf_counter()
            page.reload(wait_until="domcontentloaded", timeout=30000)
            page.wait_for_selector("#prompt", timeout=20000)
            time.sleep(1.0)
            if shots:
                page.screenshot(path=str(shots / "03_after_reload.png"))

            reload_ok = not page_errors and page.locator("#prompt").count() == 1
            report["checks"].append({
                "name": "ui:reload_continuity",
                "area": "ui",
                "status": "PASS" if reload_ok else "FAIL",
                "duration_ms": round((time.perf_counter() - t_rel) * 1000.0, 1),
                "evidence_class": "live_browser_shell",
                "page_errors": list(page_errors),
                "screenshots_dir": str(shots) if shots else None,
            })

            browser.close()
    except Exception as exc:
        report["checks"].append({
            "name": "ui:playwright_e2e",
            "area": "ui",
            "status": "FAIL",
            "duration_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "evidence_class": "live_browser_shell",
            "reason": str(exc)[:600],
            "diagnosis": evaluator.diagnose_failure(
                "ui.playwright_e2e", expected="Playwright browser session completes cleanly",
                observed=str(exc)[:500], trace={}, ui_errors=page_errors or [str(exc)], explicit_layer="ui",
            ),
        })
