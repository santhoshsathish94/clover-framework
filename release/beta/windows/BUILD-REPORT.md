# Clover AI Studio — Windows Standalone C# Installer Build & Verification Report

> **Generated deterministically by `release/beta/generate_release_reports.py` from final build artifacts.**

**Release Version**: `5.0.4` (`5.0.4.0` Win32 PE `AssemblyVersion`)  
**Product Name**: `Clover AI Studio`  
**Publisher**: `Santhosh Narayanan`  
**Distribution Model**: Standalone Native C# Win32 Per-User Installer (`CloverAIStudio-Setup.exe` / `clover-ai-studio-beta-setup.exe`)  
**Target Architecture**: `x64` (`Windows.Desktop`, `MinVersion="10.0.19041.0"`, `MaxVersionTested="10.0.26100.0"`)  
**Build Date**: `2026-11-10`  
**Build Environment**: Windows 11 x64 (`10.0.26200`), `.NET Framework 4.8` `csc.exe` (`v4.0.30319`).

---

## 1. Built Artifacts & Checksums

| Artifact | Mirror Path | Purpose | Size | SHA-256 |
|---|---|---|---:|---|
| `release/beta/windows/clover-ai-studio-beta-setup.exe` | `download/laptop/CloverAIStudio-Setup.exe` | Standalone direct-download Windows x64 installer | `9,410,520` bytes (`8.97 MB`) | `ee7b50fe60479e1fbc73329cb641108c201e85e5bc2cfd1c5ac3348e6c7d16ee` |
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
  - Client builds embed only the 32-byte RFC 8032 Ed25519 public verification key (`dba3f7e2e969a26c57a24d50e2ecd654249b42e25eecc1f9140a514d7dcedb8a`) in `core_sync.py`; private release signing keys are never shipped in client binaries.

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
- `python clover/test_google_identity_collective.py`: **32/32 passed**.
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
