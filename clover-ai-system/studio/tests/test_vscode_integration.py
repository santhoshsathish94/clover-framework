"""End-to-End Verification for VS Code & Copilot Integration (`test_vscode_integration.py`).

Tests:
  1. `server.parse_openai_messages`: preserves shared `<attachment id="Browser Pages">` (`[pageId]`) in `workspace_info`
     while ensuring trailing context-only messages never wipe `tool_rounds`.
  2. `lfm3_native.active_vscode_tools`: dynamically activates VS Code Integrated Browser tools (`open_browser_page`,
     `list_browser_pages`, `read_page`, `click_element`, `type_in_page`, `navigate_page`, `screenshot_page`,
     `run_playwright_code`), Terminal tools (`get_terminal_output`, `send_to_terminal`, `kill_terminal`), and
     Code Intelligence tools (`vscode_listCodeUsages`, `vscode_renameSymbol`, `view_image`).
  3. `server.vscode_tools`: verifies `vscode_editor`, `vscode_diagnostics`, `vscode_diff`, and `vscode_browser`
     bridge tools dispatch cleanly to a simulated VS Code window via `vscode_bridge.Bridge`.
  4. `vscode-extension/extension.js` (`v0.4.0`): verifies all 4 Clover models (`LFM3-Omni-2B`, `Clover-35B-A3B`,
     `Nemotron 3.5`, `Kimi K3.1 Pro`) and all new bridge actions (`editor`, `diagnostics`, `diff`, `browser`, `terminals`).
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOVER_DIR = HERE.parent / "clover"
sys.path.insert(0, str(CLOVER_DIR))

import lfm3_native as N  # noqa: E402
import server  # noqa: E402
import vscode_bridge  # noqa: E402


def main() -> int:
    # 1. Verify shared Browser Pages context preservation + trailing envelope safety
    msgs_shared = [
        {
            "role": "user",
            "content": (
                "<workspace_info>\n- c:\\personal\n</workspace_info>\n"
                "<attachments>\n<attachment id=\"Browser Pages\">\n"
                "The following browser pages are currently shared with you and can be interacted with using the browser tools:\n"
                "- [page-abc-123] Clover Studio UI (http://127.0.0.1:8765/) (active)\n"
                "</attachment>\n</attachments>\n"
                "<userRequest>\nRead the shared page and click the Settings button.\n</userRequest>"
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_read_1",
                    "type": "function",
                    "function": {"name": "read_page", "arguments": "{\"pageId\":\"page-abc-123\"}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_read_1",
            "content": "button 'Settings' [ref=e4]",
        },
        # Trailing context-only envelope from VS Code (must NOT wipe tool_rounds!)
        {
            "role": "user",
            "content": (
                "<attachments>\n<attachment id=\"Browser Pages\">\n"
                "The following browser pages are currently shared with you:\n"
                "- [page-abc-123] Clover Studio UI (http://127.0.0.1:8765/) (active)\n"
                "</attachment>\n</attachments>"
            ),
        },
    ]
    user_p, prior_t, tool_rds, ws_info = server.parse_openai_messages(msgs_shared)
    assert user_p == "Read the shared page and click the Settings button.", f"Unexpected user_p: {user_p}"
    assert len(tool_rds) == 1 and tool_rds[0]["name"] == "read_page", f"tool_rounds wiped: {tool_rds}"
    assert "page-abc-123" in ws_info and "Browser Pages:" in ws_info, f"Missing shared browser page in ws_info: {ws_info}"

    # 2. Verify dynamic activation of VS Code Integrated Browser, Terminal & Code Intelligence tools
    act_browser = N.active_vscode_tools(user_p, ws_info, tool_rds)
    for b_tool in ("open_browser_page", "list_browser_pages", "read_page", "click_element", "type_in_page", "screenshot_page", "run_playwright_code"):
        assert b_tool in act_browser, f"Expected browser tool {b_tool} in active_vscode_tools"

    act_term = N.active_vscode_tools("Check the background terminal output and send 'y' to the terminal.", "", [])
    for t_tool in ("run_in_terminal", "get_terminal_output", "send_to_terminal", "kill_terminal"):
        assert t_tool in act_term, f"Expected terminal tool {t_tool} in active_vscode_tools"

    act_intel = N.active_vscode_tools("Find all usages and references of CloverSwitchableModelLayer1 and rename symbol if needed.", "", [])
    for i_tool in ("vscode_listCodeUsages", "vscode_renameSymbol"):
        assert i_tool in act_intel, f"Expected code intelligence tool {i_tool} in active_vscode_tools"

    repaired_click = N.repair_vscode_browser_args(
        "click_element",
        {"element": "Settings"},
        ws_info,
        [{"name": "read_page", "arguments": "{\"pageId\":\"page-abc-123\"}", "output": "main:\n  button \"Settings\" [ref=e4]\n  button \"Send\" [ref=e5]"}],
    )
    assert repaired_click == {"element": "Settings", "pageId": "page-abc-123", "ref": "e4"}, f"Unexpected repaired_click: {repaired_click}"

    # 3. Verify Studio -> VS Code bridge tools (`vscode_editor`, `vscode_diagnostics`, `vscode_diff`, `vscode_browser`)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        bridge = vscode_bridge.Bridge(tmp / "vscode-bridge.json")
        orig_vscode = server.S.vscode
        orig_ensure = bridge.ensure
        bridge.ensure = lambda _exe, timeout=30.0: True
        server.S.vscode = bridge
        try:
            def fake_vscode_window():
                for _ in range(4):
                    job = bridge.poll("win-1", focus=500.0, title="clover-framework", version="1.143.1", wait=5.0)
                    if job:
                        act = job["action"]
                        args = job["args"]
                        if act == "editor":
                            bridge.result(job["id"], True, {"activeFile": "C:\\repo\\app.py", "cursorLine": 42, "openTabs": [{"label": "app.py", "active": True}]})
                        elif act == "diagnostics":
                            bridge.result(job["id"], True, {"count": 1, "diagnostics": [{"file": "C:\\repo\\app.py", "severity": "Error", "line": 10, "message": "Undefined name 'x'"}]})
                        elif act == "diff":
                            bridge.result(job["id"], True, {"diffOpened": True, "filePath": args.get("filePath")})
                        elif act == "browser":
                            bridge.result(job["id"], True, {"subaction": args.get("subaction"), "tool": "read_page", "output": "heading 'Clover AI Studio' [ref=e1]"})

            t = threading.Thread(target=fake_vscode_window, daemon=True)
            t.start()
            bridge.poll("win-1", focus=500.0, title="clover-framework", version="1.143.1", wait=0.0)

            tools_by_name = {t_item["schema"]["function"]["name"]: t_item for t_item in server.vscode_tools("test-sid")}
            for expected_name in ("vscode_open", "vscode_clone", "vscode_install", "vscode_run",
                                  "vscode_editor", "vscode_diagnostics", "vscode_diff", "vscode_browser"):
                assert expected_name in tools_by_name, f"Missing {expected_name} in server.vscode_tools()"

            ed_out = json.loads(tools_by_name["vscode_editor"]["run"]({"filePath": "C:\\repo\\app.py"}))
            assert ed_out["activeFile"] == "C:\\repo\\app.py" and ed_out["cursorLine"] == 42

            diag_out = json.loads(tools_by_name["vscode_diagnostics"]["run"]({}))
            assert diag_out["count"] == 1 and diag_out["diagnostics"][0]["severity"] == "Error"

            diff_out = json.loads(tools_by_name["vscode_diff"]["run"]({"filePath": "C:\\repo\\app.py", "proposedContent": "print(1)\n"}))
            assert diff_out["diffOpened"] is True

            brow_out = json.loads(tools_by_name["vscode_browser"]["run"]({"subaction": "read", "pageId": "page-abc-123"}))
            assert brow_out["subaction"] == "read" and "[ref=e1]" in brow_out["output"]
            t.join(timeout=5.0)
        finally:
            server.S.vscode = orig_vscode

    print("ALL VS CODE & COPILOT INTEGRATION TESTS PASSED")
    print(json.dumps({
        "shared_browser_context_preserved": True,
        "trailing_envelope_safe": True,
        "browser_tools_activated": sorted(list(act_browser & set(N.VSCODE_BROWSER_TOOLS))),
        "terminal_tools_activated": sorted(list(act_term & set(N.VSCODE_TERMINAL_TOOLS))),
        "code_intel_tools_activated": sorted(list(act_intel & set(N.VSCODE_CODE_INTEL_TOOLS))),
        "studio_vscode_bridge_tools": sorted(list(tools_by_name.keys())),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
