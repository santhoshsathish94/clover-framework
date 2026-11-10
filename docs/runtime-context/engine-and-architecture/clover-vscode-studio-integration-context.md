# Clover Architecture Context — Full VS Code & Copilot Integration (`File Viewer`, `Shared Integrated Browser`, `Terminal`, & `LanguageModelChatProvider`)

- **Date:** 2026-11-08
- **Human Direction:**
  *"and most important work VS code we need that kind of features let's just use VS code itself it's not available in mobile let's use all it's features fileviewer integrated browser which it can share it with agent and terminal all are very useful tool the clover AI studio doens't have an rebuilding something that's already open source doesn't make sense so create a todo list follow clover first understand VS code fully and it's copilot integration so that we can integrate clover into it and use all it's features to improve clover AI studio"*

---

## 1. Primary-Source Context: How VS Code & Its Copilot Integration Work

We inspected the running VS Code distribution (`v1.142+` / `copilot-chat v0.70.0`) from primary sources:
1. **VS Code Core Workbench (`resources/app/out/vs/workbench/workbench.desktop.main.js`):**
   - **Integrated Browser & Agent Tab Sharing (`browserViewService` + `playwrightService`, indices `20392000–20428000`):**
     - VS Code embeds a full Chromium `browserViewService` inside the editor workbench. Each open browser tab has a `model.sharingState` (`"shared"`, `"available"`, `"blockedByNetworkPolicy"`).
     - `pWi(...)` constructs the `<attachment id="Browser Pages">` context block injected into every agent turn:
       - Lists all tabs with `sharingState === "shared"` as `- [pageId] Title (url) (active|visible|not visible)`.
       - Reports the count of unshared tabs (`"N pages are open but not shared. Use the 'open_browser_page' tool to open a new page or to help the user share an existing page."`).
     - Built-in internal `vscode.lm.tools` registered by the workbench for the Integrated Browser:
       1. `open_browser_page({ url?, forceNew? })`: Opens `url` in VS Code's Integrated Browser (or, if `url` is omitted, prompts the user to share an existing unshared tab), returning the `pageId` and an accessibility tree snapshot with `[ref=e1]`, `[ref=e2]`, ... element references.
       2. `list_browser_pages({})`: Lists all browser pages currently shared with the agent.
       3. `read_page({ pageId })`: Returns the current accessibility snapshot (`aria-ref` tree) of a shared page.
       4. `click_element({ pageId, ref?, selector?, element, dblClick?, button? })`: Invokes `playwrightService` on `aria-ref=${ref}` or `selector`.
       5. `type_in_page({ pageId, ref?, selector?, element?, text?, key?, submit? })`: Types text or presses keys in the shared page.
       6. `navigate_page({ pageId, type: "url"|"back"|"forward"|"reload", url? })`: Navigates the shared browser tab.
       7. `hover_element({ pageId, ref?, selector?, element })` & `drag_element(...)` & `handle_dialog(...)`.
       8. `screenshot_page({ pageId, ref?, selector?, scrollIntoViewIfNeeded? })`: Captures a PNG/JPEG viewport or element screenshot (`image/*` `LanguageModelDataPart`).
       9. `run_playwright_code({ pageId, code?, deferredResultId?, timeoutMs? })`: Executes `async (page) => { ... }` against the shared browser tab via `playwrightService.invokeFunction`.
   - **Integrated Terminal & Shell Integration (`terminalService` + `shellIntegration`, indices `9821814–17570205`):**
     - Built-in internal `vscode.lm.tools`:
       1. `run_in_terminal({ command, explanation, goal, mode: "sync"|"async", timeout? })`: Executes commands in a persistent terminal session, returning full output inline (`sync`) or a background terminal UUID (`async`).
       2. `get_terminal_output({ id })`: Reads output from a background terminal session.
       3. `send_to_terminal({ id, command, waitForOutput? })`: Sends interactive input to a running terminal.
       4. `kill_terminal({ id })`: Terminates a background terminal.
       5. `terminal_last_command({})` & `terminal_selection({})`: Reads the last command or current text selection in the active terminal.
   - **File Viewer, Editor, Diagnostics & Language Server (`vscode.workspace`, `vscode.languages`, `vscode.commands`):**
     - Built-in & Copilot tools (`resources/app/extensions/copilot/package.json` + workbench):
       - File reading/viewing: `read_file` (`copilot_readFile`), `view_image` (`copilot_viewImage`), `list_dir` (`copilot_listDirectory`), `file_search` (`copilot_findFiles`), `grep_search` (`copilot_findTextInFiles`), `semantic_search` (`copilot_searchCodebase`).
       - File editing & diffing: `replace_string_in_file` (`copilot_replaceString`), `multi_replace_string_in_file`, `insert_edit_into_file` (`copilot_insertEdit`), `create_file` (`copilot_createFile`), `apply_patch` (`copilot_applyPatch`), plus `vscode.diff` for side-by-side visual diff preview.
       - Language Server intelligence: `get_errors` (`vscode.languages.getDiagnostics`), `vscode_listCodeUsages` (`vscode.executeReferenceProvider` / `vscode.executeDefinitionProvider` / `vscode.executeImplementationProvider`), `vscode_renameSymbol` (`vscode.executeDocumentRenameProvider`), `copilot_searchWorkspaceSymbols` (`vscode.executeWorkspaceSymbolProvider`).

2. **VS Code Copilot Extension Host Adapter (`copilot/dist/extension.js`):**
   - Any extension that registers `vscode.lm.registerLanguageModelChatProvider(vendor, provider)` appears directly in VS Code's Copilot Chat & Agent model selector.
   - When the user selects a Clover model in VS Code's `</> Agent` mode, VS Code's `ToolCallingLoop`:
     1. Collects all registered tools (`vscode.lm.tools` — including the Integrated Browser, Terminal, File Viewer, and Language Server tools) and passes them in `options.tools`.
     2. Injects `<environment_info>`, `<workspace_info>`, `<attachments id="Browser Pages">`, and `<context>` (active file, selection, open terminals) into `messages`.
     3. Invokes `provider.provideLanguageModelChatResponse(model, messages, options, progress, token)`.
     4. Whenever `progress.report(new vscode.LanguageModelToolCallPart(callId, toolName, args))` is emitted, VS Code executes the tool natively in the workbench (with its own confirmation UI, browser tab sharing prompt, diff editor, or terminal panel) and feeds the `LanguageModelToolResultPart` back in the next round until the task is complete.

---

## 2. Gaps Found in Clover's Previous VS Code Integration

1. **Shared Integrated Browser Context Was Stripped (`server.py:_strip_vscode_envelope`):**
   - Previously, `_strip_vscode_envelope` discarded `<attachment id="Browser Pages">` completely (`if "Browser Pages" in attrs ... continue`). When a user shared a browser page in VS Code (`[pageId] Title (url)`) or had open unshared tabs (`"4 pages are open but not shared"`), Clover never saw the `pageId` or the browser state!
   - **Fix:** Extract `<attachment id="Browser Pages">` (and active editor `<context>`) into `workspace_info_chunks` rather than `cleaned_u`. This preserves the invariant that a trailing context-only message never wipes `tool_rounds`, while giving `vscode_step` the exact shared browser `pageId`s and open tabs.
2. **Integrated Browser & Code Intelligence Tools Were Hidden Behind `defer_loading` (`lfm3_native.py` & `omni_native.py`):**
   - `VSCODE_CORE` only listed 11 file/terminal tools (`read_file`, `list_dir`, `file_search`, `grep_search`, `semantic_search`, `run_in_terminal`, `get_errors`, `replace_string_in_file`, `create_file`, `insert_edit_into_file`, `apply_patch`).
   - All Integrated Browser tools (`open_browser_page`, `read_page`, `click_element`, `type_in_page`, `navigate_page`, `screenshot_page`, `run_playwright_code`, `hover_element`, `handle_dialog`) and Code Intelligence / Terminal tools (`vscode_listCodeUsages`, `vscode_renameSymbol`, `get_terminal_output`, `send_to_terminal`, `terminal_last_command`, `terminal_selection`, `view_image`) were marked `defer_loading = True`.
   - **Fix:** Dynamically include Integrated Browser tools in the active tool set whenever the prompt or `<workspace_info>` mentions browser/web/page/UI/url or has shared browser pages, and include `vscode_listCodeUsages`, `vscode_renameSymbol`, `get_terminal_output`, `send_to_terminal`, and `view_image` when relevant.
3. **`CloverLocalProvider` Only Listed `LFM3-Omni-2B` & `LFM3-1.7B` in VS Code's Model Picker (`vscode-extension/extension.js`):**
   - Users in VS Code could not pick `Clover — Clover-35B-A3B`, `Clover — Nemotron 3.5 Super VL`, or `Clover — Kimi K3.1 Pro` from the VS Code Copilot model picker, nor did `provideLanguageModelChatResponse` pass the selected model ID to the Clover server.
   - **Fix:** Query `/api/settings` (`model_catalog`) from the Clover Studio server so `CloverLocalProvider` exposes both the local `LFM3-Omni-2B` model and all connected/hosted models (`Clover-35B-A3B`, `Nemotron 3.5 Super VL`, `Kimi K3.1 Pro`) in VS Code's Chat model picker.
4. **Clover AI Studio Could Not Use VS Code's File Viewer (`editor`/`diff`/`diagnostics`/`symbols`), Integrated Browser (`browser`), or Terminal State (`terminal`) Directly (`vscode_bridge.py` + `extension.js` + `server.py`):**
   - Instead of rebuilding a code editor, diff viewer, browser, or terminal inside Clover AI Studio, Studio can call VS Code directly over `vscode_bridge.py`:
     - `editor` (`vscode_editor`): Read the active file, cursor selection, open editor tabs, or specific line ranges with Language Server diagnostics (`vscode.languages.getDiagnostics`) and document symbols (`vscode.executeDocumentSymbolProvider`).
     - `diff` (`vscode_diff`): Open a side-by-side diff view (`vscode.diff`) in VS Code's File Viewer comparing original vs. proposed file contents.
     - `browser` (`vscode_browser`): Open a URL in VS Code's Integrated Browser (`workbench.action.browser.open`) or invoke VS Code's built-in browser tools (`open_browser_page`, `read_page`, `click_element`, `type_in_page`, `navigate_page`, `screenshot_page`) via `vscode.lm.invokeTool` so Clover AI Studio and VS Code share the exact same integrated browser!
     - `diagnostics` (`vscode_diagnostics`): Get live compiler/linter errors and warnings across the workspace (`vscode.languages.getDiagnostics`) and optionally focus the Problems panel (`workbench.actions.view.problems`).
5. **Multi-Cycle `AGENTS.md` Clover Cycle Outcome Verification on `/v1/chat/completions` (`server.py`):**
   - When VS Code finishes its tool rounds and requests the final answer from `/v1/chat/completions`, `S.mode == "lfm3"` now passes `with_tool_outputs(user_prompt, tool_rounds)` and runs `verify_turn_outcome` (`max_clover_cycles = 3`) so every VS Code Copilot answer is verified against the tool outputs, compiler, arithmetic, and format constraints.

---

## 3. Laptop test and the 5.0.3 fixes (2026-11-10)

- **Human direction:** test the laptop and VS Code integration, approve VS Code's dialogs myself, fix everything as 5.0.3, *"and make sure it can also use playwrite"*; then *"sure do it"* (publish).
- **Where:** laptop (Windows, Iris Xe), VS Code 1.145.2 (Copilot Chat 0.73.0), Clover AI Studio with `LFM3-Omni-2B`, a local test page (name box, Greet button) shared in VS Code's browser. Evidence (not in git): `%TEMP%\clover-consc` on the release PC (screenshots, captured `/v1` requests, UI log).
- **VS Code 1.145 has two chat agents with different request formats.** The default **Copilot** agent (agent host, Copilot CLI SDK) sends no `<userRequest>`; the shared pages come only as `<attachment name="Browser Pages">`, the text carries `<current_datetime>` and a `<system_reminder>`, and its tools are named differently (`openBrowserPage`, `runPlaywrightCode`, `view`, `grep`, `powershell`, ... 82 tools). The **Local** agent sends `<userRequest>` and snake_case tools (`open_browser_page`, `run_playwright_code`, `read_file`, ...).
- **What failed and what changed (all in `server.py`, `lfm3_native.py`, `omni_native.py`; tests `tests/test_vscode_503.py`):**
  - F9: the Copilot agent refused the model (declared input limit 32,768, "static system messages and tool definitions exceed the model's usable context budget") -> 131,072 in the extension.
  - F10: a message after a stopped request was dropped and the stopped command repeated -> a new request after an unfinished turn is answered.
  - F11 (Copilot agent): the page list was dropped, so the model opened `https://clover.com` (a real company's site) and `file:///shared/clover/shared.html` for "the shared test page" (both skipped), then made up the page text -> the attachment is kept; the agent's tool names are recognised; an address nobody gave is refused while pages are shared; `search_tools` also finds tools already offered.
  - Playwright (both agents): the model copied VS Code's read-only example (`page.evaluate(... document.title)`) and only read the page -> an action example (fill, click, return); one prompt back when an asked step was not done, naming text never typed; no repeat of a call already run; a navigation of a shared page to its own address becomes a read (the Local agent reset the page that way).
  - F12: after the user skipped a page tool in VS Code's chat, the `/v1` fallback turn (`lfm3_turn`, kind "VS Code answer") used Studio's own bridge tool and typed into the page, unseen in that chat -> no Studio VS Code tools in that turn. **Not changed:** that turn still has Studio's other tools under Studio's permission settings.
  - Studio's own chat (bridge): F1 web page "in VS Code" opens in VS Code; F2 editor/Problems/diff/browser offered when asked; F3 VS Code confirmation dialogs (Windows task dialogs when invoked outside chat) get 240 s and a note; F4 diagnostics only count open files, so the file is shown first; F5 "already installed"; F8 screenshots to the chat; Playwright subaction with locator hints and selector repair.
- **Measured (live, after the fixes):** Playwright typed a name and clicked Greet from Studio's chat, from the Copilot agent ("Hello, Maya"; installed 5.0.3: "Hello, Lena") and from the Local agent ("Hello, Ravi"); replaying the captured first Copilot request gave fill-and-click code 4/4; Copilot agent + laptop model "25 * 4 = 100"; stop, then "Do not run anything. What is 6 * 7?" -> "6 * 7 = 42", nothing ran. Plain sums (10, laptop): direct 9/10 right first time in 2.1-3.9 s vs calculator 4.0-5.1 s.
- **Approvals:** VS Code records who approved: Copilot agent `~/.copilot/session-state/<id>/events.jsonl` (`decisionSource` `human_response` | `host_policy`), Local agent `globalStorage/emptyWindowChatSessions/*.jsonl` (`isConfirmed.type`). Several approvals during the test were made by a person other than the test helper; later Playwright runs were approved by VS Code without a prompt (host policy, session rule).
- **Not covered:** remote Clover models in VS Code chat; one live Playwright run per agent after the fixes; the phone chat on 5.0.3 (phone locked); with Tools set to Always the calculator stays offered; simple Copilot-agent questions now take about 11 s instead of 3.6 s (its core tools are offered, as the Local agent's were).
- **Released:** v5.0.3 (commits `90518208`, `531c044a`, `107e6166`; tag `v5.0.3`; 12 release assets checked against the committed files; site live at 5.0.3). k3's web Studio was found at 5.0.1 (15 files differ from 5.0.3 and 4 are missing; the differing files include `autonomous_clover.py`, `jepa_engine.py` and `memory-knowledge-graph.py`, three of the five files the 5.0.1 phone app had in older versions) and was not updated.
