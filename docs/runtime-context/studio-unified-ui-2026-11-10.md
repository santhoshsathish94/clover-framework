# Clover AI Studio — Unified UI Upgrade Across Web, Mobile, and Laptop (2026-11-10)

## 1. Context
- **Branch & Worktree**: `feature/studio-unified-ui` at `C:\personal\oss\clover-studio-ui-wt` (branched from `main` at `ee0c7656`).
- **Problem Statement**: Previously, the Clover AI Studio left sidebar (`.rail`) only displayed conversation history, burying `Workspaces`, `Memory`, `Tools`, `Learning`, and `Settings` several levels deep inside a right-hand slide-over sheet (`#sheet`), and the welcome view (`#hero`) did not surface the 5-stage Clover operating cycle (`Context → Direction → Execution → Outcome → Growth`) or provide direct entry cards for `Memory`, `Tools`, and `Learning`.
- **Files Read**:
  - `clover-ai-system/studio/clover/web/app.html` (canonical single-file UI for Laptop `windows`, Mobile `android`, and Web `web`)
  - `site/clover-ai/index.html` (Web Studio build target)
  - `clover-ai-system/mobile/android-app/assets/index.html` (Android WebView asset target)
  - `clover-ai-system/studio/studio/build_assets.py` (asset synchronization script)
  - `assets/robo-icon/README.md` and `assets/robo-icon/icons/` (existing Clover Universal Icon Pack)
  - `clover-ai-system/studio/tests/test_browser_first.py`, `test_browser_handback.py`, `test_regression_fixes.py`, `test_strict_c1_c2_c3.py`, `test_tool_tasks.py`

## 2. Direction
1. Preserve the existing Clover Apple-inspired light palette (`--cream: #F7F3E9`, `--green: #176B2C`, `--deep: #0D4D20`, `--leaf: #83B83D`, `--rail: #F1ECDF`, `--surface: #FFFDF8`) and use **only** existing icons from `app.html` / `assets/robo-icon/`.
2. Upgrade the Left Sidebar (`.rail`) with:
   - Brand header (`icon.png`, `Clover AI Studio`, `v5.0.2`, and `#new-chat` action)
   - Compact model selector (`#mpick`)
   - 6 primary navigation items (`Chat`, `Workspaces`, `Memory`, `Tools`, `Learning`, `Settings`) using `#i-chat`, `#i-folder`, `#i-models`, `#i-code`, `#i-spark`, and `#i-settings`
   - Scrollable `Conversations` list (`#chats`)
   - Footer status (`Clover Ready` with green `.dot.ok`) and active workspace indicator (`#rail-ws`).
3. Upgrade the Welcome / Empty-Chat Main View (`#hero`, `#composer`, `#hero-below`) with:
   - Keep `icon.png` in the top-left sidebar logo place (`.rail__brand .brand`) and display the Robo Clover mascot (`robo-clover.webp`, `.mascot`) in the center `#hero` above `Clover` (`#greet`) and `Work · Create · Learn · Grow`
   - Keep `header.bar` pinned at `top: 0` at all times (using `.app.empty-chat .stream { margin-top: auto; }` and `.app.empty-chat .dock { margin-bottom: auto; }` instead of `.main { justify-content: center; }`), and keep both `[data-nav="settings"]` in the left sidebar (`#rail`) and `#set-btn` in `<header class="bar">` always present across Web, Mobile, and Laptop (never hidden or removed)
   - Keep the `.work` steps accordion **always collapsed** after work is done (`botEl` calls `workEl(false, false)` so `aria-expanded="false"` and `.work__b.hidden === true` once a turn finishes, and connection errors before any server event arrives do not leave a synthetic `Understanding` step)
   - Unify `/api/browser/open` calls in `app.html` into a single click-handler helper `openBrowserClick(body, fallbackUrl)` so `test_browser_first.py` passes 30/30 (`app.count("/api/browser/open") == 1`)
   - Centered 2-row rounded Composer card (`What would you like to do?` placeholder, bottom toolbar with `+` `#attach` for System Senses `screen_record` / `camera` / `microphone`, image/screenshot `#screen-btn`, file attachment `#clip-btn`, and `#send` button — with no duplicate File or Screenshot rows inside `#senses-menu`)
   - 5-Stage Clover Cycle Strip (`Context / Understand` › `Direction / Plan together` › `Execution / Do the work` › `Outcome / Check the result` › `Growth / Learn and improve`)
   - Collaboration pill (`System · Human · AI working together` with `#i-success` checkmark)
   - Keep `Memory`, `Tools`, and `Learning` in the left sidebar navigation only (no duplicate 3-card `.feature-grid` below the composer)
   - Smart `#senses-menu` (`.senses-pop`) positioning: opens downward (`.drop-down`) in `.empty-chat` or when space above `< 240px` with zero scrollbar.
4. Strictly 2-Pane Layout (`.rail` Left Pane + `.main` Center Pane) and VS Code-style Workspace Explorer & File Viewer:
   - **Left Pane (`#rail`)**: Toggled with `#menu` (`#i-sidebar`). Contains collapsible `Chat` submenu (`#rail-chat-sub` → `#chats` conversation list), collapsible `Workspaces` submenu (`#rail-ws-sub` → `#ws-source-pills` + VS Code-style file tree `#vsc-tree`), `Memory`, `Tools`, `Learning`, and `Settings`.
   - **Center Pane (`.main`)**: Contains `<header class="bar">` at top (`#menu`, `#new-chat-2`, `#chat-title`, `#set-btn`). Below the bar, displays either the Chat view (`#stream` + `.dock`) or the in-place Section view (`#sheet` inside `.main` with `.app.view-page`, zero slide-over drawer or 3rd pane).
   - **Workspaces Parent View vs. File Viewer**:
     - Clicking `Workspaces` (`[data-nav="workspaces"]`) or a parent workspace pill (`clover-ai-studio (Local Repo)` in `#ws-source-pills`) opens `#ws-parent-view` in the center pane (`#ws-conn`, `Add Folder or Connect GitHub Repository`, `GitHub Client & Account Connection`) with `#vsc-workbench` hidden.
     - Clicking any file in `#vsc-tree` opens `#vsc-workbench` (the File Viewer: Code with line numbers, Markdown Preview, Copy, Ask Clover, Open in VS Code) in the center pane with `#ws-parent-view` hidden so workspace connection controls are never shown inside each file, while blocking secret/credential files (`SECRETISH`) and `vault` directories.
   - **Settings (`[data-page="home"]`)**: Omits the duplicate `Workspaces`, `Memory`, `Tools`, and `Learning & Growth` rows (since each has its own left-pane menu) and shows `Model and voice`, `GitHub account (Memory backup)`, `Storage`, `Debug logs & Senses`, and `Advanced`.
   - Synchronize all three platform targets (`app.html`, `site/clover-ai/index.html`, and `clover-ai-system/mobile/android-app/assets/index.html`).

## 3. Execution
- Updated `clover-ai-system/studio/clover/web/app.html` with the 2-pane layout (`#rail` + `.main`), left-pane `Chat` and `Workspaces` submenus (`#rail-chat-sub`, `#rail-ws-sub`, `#vsc-tree`), in-place center views (`#sheet` inside `.main`), VS Code-style File Viewer (`#vsc-workbench`), welcome hero (`robo-clover.webp`), 2-row composer, 5-stage Clover cycle strip, collaboration pill, `[data-page="memory"]` hub, streamlined `[data-page="home"]` Settings view, and responsive mobile rules (`@media (max-width: 860px)`).
- Updated `clover-ai-system/studio/clover/workspaces.py`, `clover-ai-system/studio/clover/server.py`, and `clover-ai-system/mobile/clover_git_memory_repo.py` with `browse_tree`, `read_workspace_file`, `github_client_request`, `github_client_status`, and default local repo (`__local_repo__`) support.
- Updated `clover-ai-system/studio/studio/build_assets.py` to resolve paths relative to the active worktree/repository root and automatically synchronize both `site/clover-ai/index.html` and `clover-ai-system/mobile/android-app/assets/index.html`.
- Added and expanded `clover-ai-system/studio/tests/test_studio_unified_ui.py` (37 checks).

## 4. Outcome & Evidence
- **Live Server & Browser End-to-End Regression (`ui_harness.py` on `http://127.0.0.1:8798/`)**:
  - Verified desktop (`1280×800`, both sidebar open and `.rail-off` collapsed) and mobile phone (`390×844`, `html.phone`, both empty-chat welcome and active multi-turn chat, plus slide-out drawer `#rail.on` and `window.cloverBack()`) against the live `server.py` HTTP routes via Playwright and screenshots.
  - Confirmed strict 2-pane layout (`sheetInCenterMain: true`, `treeInRail: true`, `viewerInCenter: true`): clicking `Chat`, `Workspaces`, `Memory`, `Tools`, `Learning`, `Settings`, `#set-btn`, or any file in `#vsc-tree` loads directly in the center pane (`section.main`), and `Chat` / `Workspaces` submenus mutually toggle in the left pane (`#rail`).
  - Confirmed `Settings` (`[data-page="home"]`) displays only `Model and voice`, `GitHub account (Memory backup)`, `Storage`, `Debug logs & Senses`, and `Advanced` without duplicating `Workspaces` through `Learning`.
  - Confirmed `[data-nav="settings"]` in `#rail` and `#set-btn` in `header.bar` are always present and visible across all viewports (`setBtnDesktopOpen: true`, `railSettingsDesktopOpen: true`, `setBtnDesktopRailOff: true`, `setBtnMobile: true`).
  - Confirmed that while a turn is running (`streamSend`), `.work` displays live steps, and as soon as the work finishes (`finish(ev)` → `botEl(m)`), all `.work` step accordions are collapsed (`allCollapsed: true`, `aria-expanded="false"`, `.work__b.hidden === true`).
  - Confirmed `#senses-menu` opens downward in `.empty-chat` (`top: 551..786` on desktop, `top: 518..753` on mobile) and upward when the composer is docked at the bottom of an active chat (`top: 444..679`), with `scrollHeight === clientHeight` (`234px`, zero scrollbar) in every view.
  - Verified clicking `Workspaces`, `Memory`, `Tools`, `Learning`, `Settings`, and `Chat` in the sidebar and drilling into subpages (`layers`, `model`, `perms`, `personal`, `collective`, `storage`, `debug`, `account`, `vscode`) with `#sheet-back` and `window.cloverBack()`.
- **Automated Suites**:
  - `python clover-ai-system/studio/tests/test_studio_unified_ui.py`: **37/37 PASS**
  - `python clover-ai-system/studio/tests/test_browser_first.py`: **30/30 PASS**
  - `python clover-ai-system/studio/tests/test_browser_handback.py`: **51/51 PASS**
  - `python clover-ai-system/studio/tests/test_tool_tasks.py`: **18/18 PASS**
  - `python clover-ai-system/studio/tests/test_caution_only.py`: **16/16 PASS**
  - `python scripts/check-site.py`: **10 pages checked, 0 site errors**

## 5. Growth
- `build_assets.py` previously had hardcoded `C:\personal\oss\clover-framework` paths and did not sync `clover-ai-system/mobile/android-app/assets/index.html`; making `REPO_ROOT` worktree-aware and syncing both Web and Android targets in `build_assets.py` prevents cross-platform UI drift when working in git worktrees.
