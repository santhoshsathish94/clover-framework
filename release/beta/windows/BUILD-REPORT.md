# Clover AI Studio — Windows Standalone C# Installer Build & Verification Report

> **Generated deterministically by `release/beta/generate_release_reports.py` from final build artifacts.**

**Release Version**: `5.0.6` (`5.0.6.0` Win32 PE `AssemblyVersion`)  
**Product Name**: `Clover AI Studio`  
**Publisher**: `Santhosh Narayanan`  
**Distribution Model**: Standalone Native C# Win32 Per-User Installer (`CloverAIStudio-Setup.exe` / `clover-ai-studio-beta-setup.exe`)  
**Target Architecture**: `x64` (`Windows.Desktop`, `MinVersion="10.0.19041.0"`, `MaxVersionTested="10.0.26100.0"`)  
**Build Date**: `2026-11-11`  
**Build Environment**: Windows 11 x64 (`10.0.26200`), `.NET Framework 4.8` `csc.exe` (`v4.0.30319`).

---

## 1. Built Artifacts & Checksums

| Artifact | Mirror Path | Purpose | Size | SHA-256 |
|---|---|---|---:|---|
| `release/beta/windows/clover-ai-studio-beta-setup.exe` | `download/laptop/CloverAIStudio-Setup.exe` | Standalone direct-download Windows x64 installer | `9,572,824` bytes (`9.13 MB`) | `e97cb11bfdf3cff2383f97270f5fc96ca8f054cc84444c1f76267e0d16b25c45` |
| `release/beta/windows/CloverFramework-CodeSign.cer` | `download/laptop/CloverFramework-CodeSign.cer` | Authenticode code-signing public certificate | `832` bytes | `9365ec5d455b0ef8179ccf1c061cf481c86dad234b9f3dc62244e487c659072d` |

---

## 2. Trusted Per-User Installation & Authenticode Signing Architecture

- **Native C# Setup & Launcher (`Setup.cs` & `Launcher.cs`)**:
  - Compiled as clean native x64 GUI executables using `csc.exe` with an embedded Win32 application manifest (`app.manifest`) declaring `<requestedExecutionLevel level="asInvoker" uiAccess="false" />`, Windows 10/11 compatibility GUID `{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}`, Per-Monitor DPI awareness, and `longPathAware`.
  - Declaring `asInvoker` ensures Windows User Account Control (UAC) installer heuristics allow per-user installation into `%USERPROFILE%\CloverAI` without requiring Administrator elevation.
- **Embedded Code-Signing Certificate & Automatic Unblocking**:
  - Both `CloverAIStudio-Setup.exe` and `CloverAIStudio.exe` are Authenticode-signed (`SHA256`, `-IncludeChain All`) with `CN=Clover Framework, O=Clover Framework` (Thumbprint `843BC8476BFCAD1EE2FABE82CDB403994CCE3F99`, `Status: Valid`).
  - `CloverFramework-CodeSign.cer` is embedded inside both executables (`CloverAIStudio.codesign.cer`). On launch (or via `CloverAIStudio-Setup.exe --trust`), the installer automatically registers the certificate in `Cert:\CurrentUser\TrustedPublisher` and strips any browser download `:Zone.Identifier` alternate data stream from the installer and all extracted runtime files so Windows never blocks execution.
- **CPSL Ed25519 Asymmetric Release Verification**:
  - Client builds embed only the 32-byte RFC 8032 Ed25519 public verification key (`a4d9d040035a19a5f1214bd30881b40842d4b06d0990c1bfdd6252fd90280283`) in `core_sync.py`; private release signing keys are never shipped in client binaries.

---

## 3. Verification & Regression Suite Results

- `python tests/test_semantics.py`: **155/155 passed**.
- `python tests/test_tool_routing.py`: **129/129 passed**.
- `python tests/test_clover_repo.py`: **27/27 passed**.
- `python tests/test_regression_fixes.py`: **73/73 passed**.
- `python tests/test_page_fetch.py`: **119/119 passed**.
- `python tests/test_browser_first.py`: **30/30 passed**.
- `python tests/test_browser_handback.py`: **51/51 passed**.
- `python clover/test_collective.py`: **35/35 passed**.
- `python tests/test_collective_repo.py`: **48/48 passed**.
- `python tests/test_github_signin.py`: **59/59 passed**.
- `python tests/test_terminal_tree_fixes.py`: **14/14 passed**.
- `python clover/test_human_usefulness.py`: **not re-run for this release (urllib.error.URLError: <urlopen error [WinError 10061] No connection could be made because)**.
- `python clover/test_live_activity_sse.py`: **18/18 passed**.
- `python clover/test_clover_cycles.py`: **passed (exit 0)**.
- `python tests/test_tool_tasks.py`: **18/18 passed**.
- `python tests/test_caution_only.py`: **16/16 passed**.
- `python tests/test_superagent_jev_dory.py`: **passed (exit 0)**.
- `python tests/test_vscode_503.py`: **93/93 passed**.
- `python tests/test_token_economy.py`: **21/21 passed**.
- `python tests/test_studio_unified_ui.py`: **41/41 passed**.
- `python tests/test_strict_c1_c2_c3.py`: **34/34 passed**.
- `python tests/test_recall_scope.py`: **16/16 passed**.
- `python tests/test_answer_length.py`: **4/4 passed**.
- `python tests/test_loop_trim.py`: **6/6 passed**.
- `python tests/test_direct_replies.py`: **4/4 passed**.
- `python tests/test_document_paths.py`: **3/3 passed**.
- `python tests/test_table_counts.py`: **6/6 passed**.
- `python tests/test_tool_call_loops.py`: **5/5 passed**.
