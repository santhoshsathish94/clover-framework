# Clover AI Studio v5.0.4 — UI walk, fixes and open questions (2026-11-10)

## Context
- Request: get the latest `main`, test the new unified UI, make sure the features work, fix what can be fixed,
  step by step, following Clover.
- Branch `fix/studio-ui-504`, worktree `C:\personal\oss\clover-ui-fix-wt`, from `main` = `origin/main` =
  `3facab25` (tag v5.0.4; checked again with `git fetch` before the commit).
- Read: `studio/clover/web/app.html`, `studio/clover/server.py` (`run_terminal_tool`, `_local_repo_dir`),
  `studio/clover/local_terminal.py`, `studio/clover/browser_host.py`, `studio/studio/build_assets.py`,
  `studio/tests/ui_harness.py`, `studio/tests/test_studio_unified_ui.py`, and the failing suites below.

## Direction
- Walk every feature in the browser against the real server routes before fixing anything.
- Fix only what is clearly broken. Do not edit existing tests. Ask before push or merge.

## Execution
- Inventory: all 61 endpoints the UI calls are handled by `server.py`.
- Harness: `tests/ui_harness.py` (real server routes, stand-in engine, scratch base) on port 8797, opened in the
  VS Code browser at 1024×768 and 390×844. The Terminal app ran real PowerShell on this PC. The Browser app read
  real web pages.
- Every feature was clicked and the console and failed requests were recorded (see Outcome).

## Outcome

### Fixed (commit `ced57d82`)
1. **The Terminal app lost table output.** The pwd button (`Get-Location`), any `Select-Object`, and
   `cd ..; Get-Location` showed "(completed with no output)". Windows PowerShell 5.1 printed nothing when the
   script ended with `exit`. Now the command's output is formatted inside the script, before the exit
   (`server.run_terminal_tool`). I compared 19 commands, old script against new (exit codes, folder, native
   stderr text, `&&`/`||`, `Write-Host`, `throw`). Only the four lost outputs changed.
2. **Opened folders in the Workspaces and Memory trees kept "Loading…"** above their files. A failed load showed
   "Loading…" forever. Now the box is emptied for every folder, and a failure appears inside the folder
   (`loadWorkspaceTree`, `loadMemoryTree`). Site and Android copies were synced with `build_web_site` only.

### Evidence
- New `tests/test_terminal_tree_fixes.py` has 14 checks. All 14 pass on the fix. On v5.0.4, 8 fail: every check of
  the two bugs. The 6 checks of unchanged behaviour pass on both versions.
- All 37 suites ran, each in its own process: 33 passed.
  - 3 failed before the fix too (see Open questions).
  - `test_browser_handback.py` failed once and then passed 6 times in a row. It is intermittent and unrelated to
    the fix. The cause is in `browser_host.py`: a page id is the time to the second plus a content hash, so two
    pages saved in the same second sort by hash, not by time.
- Browser re-check after restarting the harness, with no console errors:
  - pwd, `Select-Object`, `cd ..; Get-Location` and `Get-ChildItem -Name | Select-Object -First 3` print, and the
    folder moves;
  - `cmd /c exit 3` shows exit 3;
  - Workspaces `src` and Memory `config` show their files;
  - a folder deleted after listing shows "gone-dir is not a folder" inside it.

### Worked in the walk (stand-in engine, so these test the UI and server, not the model)
- **Navigation:** the 6 sections and every submenu; the Settings pages (model, account, storage, permissions,
  debug, advanced); the Learning views; the header Settings button; the sidebar toggle.
- **Mobile layout (390 px):** the drawer closes after choosing an app, and nothing overflows.
- **Chat:** greeting; permission card (Allow once / Don't allow), tool card, then reply; Clover's command logged in
  the Terminal app; search with browser links; table; code block; picture; Stop; Try again; Helpful; new chat;
  delete chat (removed on the server too); attach a file; the + senses menu opens and closes (no capture).
- **Tools:**
  - Terminal: commands, errors, cd, Send Output to Chat, Ask Clover in Terminal.
  - Browser: search, Wikipedia page.
  - Custom tool: add, open, remove.
  - Permission change persisted.
- **Workspaces:** add a folder, tree, file viewer. **Memory:** tree, AGENTS.md, personal summary.

### Not verified
- **Not run:**
  - camera, microphone and screen capture (they need consent on this PC);
  - GitHub connect and clone (they need a token);
  - the real model (LFM3/Omni) and speech;
  - the Android app and the public web build.
- **Visuals:** the VS Code browser tab is hidden, so CSS transitions do not advance and screenshots can be stale.
  I measured layouts with transitions turned off; I did not judge them from pictures.

## Open questions for the human
1. **Pre-existing failing tests** (I did not edit tests):
   - `test_local_tool_packs.py` expects 19 base tools. There are 20 since `manage_todo_list` was added (c54df640).
     Update it to 20?
   - `test_vscode_bridge.py` expects "run pip install requests in the terminal" to go to VS Code. Since a3cf049b
     it runs in the Studio Terminal. Which is intended?
   - `test_csi_child.py` fails only in a worktree: it compares against the main checkout's CRLF files. It passes in
     the main checkout.
2. **Terminal default folder.** With no workspace open, the Terminal app starts in Clover's memory repository
   (`_local_repo_dir`). Clover's own PowerShell tool starts in the home folder. Should the Terminal app use the
   home folder too?
3. **Custom tools** are stored only in this browser (`localStorage` `clover_custom_tools_v1`). The text says Clover
   can launch them, but the server never receives them. Change the text, or connect them to Clover?
4. **Noted, not changed:**
   - Chats kept in the browser stay listed after the server has none (merge-only design).
   - A stopped answer is not marked as stopped (same in v5.0.3).
   - Every finished turn logs `POST /api/chat/stream net::ERR_ABORTED` in the console; the reply is complete.
     Cause not investigated.
   - The file viewer's line count is one too many.
   - The Listen error appears only as a tooltip.
   - example.com fails TLS verification on this PC only.

## Growth
- Windows PowerShell 5.1 scripts run with `-Command` that end in `exit` can drop output that is formatted
  automatically. Format the output inside the script (`& { … } | Out-String -Stream`). Do not merge native stderr
  with `2>&1`: PowerShell 5.1 wraps each stderr line in an error record.
- A test that compares against a sibling checkout cannot pass from a worktree. Check where its reference comes from
  before calling it a regression.
- One failure in a full run is not a regression until it is re-run and its cause is found (the hand-back test above).
