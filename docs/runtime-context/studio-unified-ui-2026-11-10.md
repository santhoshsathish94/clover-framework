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
   - Centered Clover emblem (`icon.png`), `Clover` heading (`#greet`), and `Work · Create · Learn · Grow` subtitle
   - Centered 2-row rounded Composer card (`What would you like to do?` placeholder, bottom toolbar with `+` `#attach`, image/screenshot `#screen-btn`, file attachment `#clip-btn`, and `#send` button)
   - 5-Stage Clover Cycle Strip (`Context / Understand` › `Direction / Plan together` › `Execution / Do the work` › `Outcome / Check the result` › `Growth / Learn and improve`)
   - Collaboration pill (`System · Human · AI working together` with `#i-success` checkmark)
   - Keep `Memory`, `Tools`, and `Learning` in the left sidebar navigation only (no duplicate 3-card `.feature-grid` below the composer, which previously added `210px` of vertical height, pushed the composer up to `y = 227px` on mobile, and caused `#senses-menu` and `#hero` to clip off the top of the viewport)
   - Smart `#senses-menu` (`.senses-pop`) positioning: opens downward (`.drop-down`) in `.empty-chat` or when space above `< 320px`, and caps `max-height` to available viewport height so the `+` menu and `Clover` emblem are 100% visible on every screen.
4. Add a first-class `[data-page="memory"]` view in `#sheet`, expose `Workspaces` across Web, Mobile, and Laptop, and synchronize all three platform targets (`app.html`, `site/clover-ai/index.html`, and `clover-ai-system/mobile/android-app/assets/index.html`).

## 3. Execution
- Updated `clover-ai-system/studio/clover/web/app.html` with the new sidebar navigation, welcome hero, 2-row composer, 5-stage Clover cycle strip, collaboration pill, 3 feature cards, `[data-page="memory"]` hub, and responsive mobile rules (`@media (max-width: 860px)`).
- Updated `clover-ai-system/studio/studio/build_assets.py` to resolve paths relative to the active worktree/repository root and automatically synchronize both `site/clover-ai/index.html` and `clover-ai-system/mobile/android-app/assets/index.html`.
- Added `clover-ai-system/studio/tests/test_studio_unified_ui.py` (28 checks).

## 4. Outcome & Evidence
- **Visual & Interactive Browser Verification**:
  - Verified desktop layout (`1280×800`) and mobile phone layout (`390×844`, `html.phone`, including slide-out navigation drawer `#rail.on`) in the integrated browser via Playwright and screenshots.
  - Verified clicking `Memory`, `Tools`, `Learning`, `Workspaces`, and `Chat` in the sidebar and feature cards opens the corresponding views and updates active navigation state.
- **Automated Suites**:
  - `python clover-ai-system/studio/tests/test_studio_unified_ui.py`: **28/28 PASS**
  - `python scripts/check-site.py`: **10 pages checked, 0 site errors**
  - `python clover-ai-system/studio/tests/test_tool_tasks.py`: **18/18 PASS**
  - `python clover-ai-system/studio/tests/test_caution_only.py`: **16/16 PASS**
  - `python clover-ai-system/studio/tests/test_browser_handback.py`: **51/51 PASS**

## 5. Growth
- `build_assets.py` previously had hardcoded `C:\personal\oss\clover-framework` paths and did not sync `clover-ai-system/mobile/android-app/assets/index.html`; making `REPO_ROOT` worktree-aware and syncing both Web and Android targets in `build_assets.py` prevents cross-platform UI drift when working in git worktrees.
