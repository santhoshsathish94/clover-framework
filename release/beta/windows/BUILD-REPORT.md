# Clover AI Studio — Windows Store & Standalone Beta Build & Verification Report

**Release Version**: `0.5.0-beta.1` (`0.5.0.0` in `AppxManifest.xml` & Win32 PE `AssemblyVersion`)  
**Product Name**: `Clover AI Studio`  
**Publisher**: `Santhosh Narayanan`  
**Microsoft Store Package Identity Name**: `SanthoshNarayanan.CloverAIStudio`  
**Microsoft Store Publisher DN**: `CN=A0DCEEB8-49DB-4CD8-8D07-BC439E69B9AF`  
**Microsoft Store ID**: `9P2KP2XGF5LG`  
**Package Family Name**: `SanthoshNarayanan.CloverAIStudio_ffvna78masvs2`  
**Target Architecture**: `x64` (`Windows.Desktop`, `MinVersion="10.0.19041.0"`, `MaxVersionTested="10.0.26100.0"`)  
**Build Date**: `2026-11-05`  
**Build Environment**: Windows 11 x64 (`10.0.26200`), `.NET Framework 4.8` `csc.exe` (`v4.0.30319`), Windows SDK BuildTools `makeappx.exe` & `signtool.exe` (`10.0.28000.2957`).

---

## 1. Built Artifacts & Checksums

| Artifact | Purpose | Size | SHA-256 |
|---|---|---:|---|
| `clover-ai-studio-beta.msixupload` | Microsoft Partner Center Store upload bundle | `4.52 MB` (`4,738,846` bytes) | `a02dceb392ddb62428d6be7fbd8bd0bc7bde618f5a6f928d6c368d5314779979` |
| `SanthoshNarayanan.CloverAIStudio_0.5.0.0_x64.msix` | Store-identity MSIX package (`x64`) | `4.57 MB` (`4,787,102` bytes) | `71411e4f509390b616170791ff8ba6754640008155954d9cef59dd336d2e60b3` |
| `clover-ai-studio-beta.msix` | Alias copy of Store-identity MSIX package | `4.57 MB` (`4,787,102` bytes) | `71411e4f509390b616170791ff8ba6754640008155954d9cef59dd336d2e60b3` |
| `clover-ai-studio-beta-setup.exe` | Standalone direct-download Windows x64 installer | `2.31 MB` (`2,427,352` bytes) | `40531b4d6b63241211bf1a0ecc05ae8de09b60ac094cf525a71d9786d1a6bfdb` |

---

## 2. Microsoft Partner Center Identity Alignment (`AppxManifest.xml`)

```xml
<Identity
  Name="SanthoshNarayanan.CloverAIStudio"
  Publisher="CN=A0DCEEB8-49DB-4CD8-8D07-BC439E69B9AF"
  Version="0.5.0.0"
  ProcessorArchitecture="x64" />

<Properties>
  <DisplayName>Clover AI Studio</DisplayName>
  <PublisherDisplayName>Santhosh Narayanan</PublisherDisplayName>
  <Logo>Assets\StoreLogo.png</Logo>
  <Description>Local-first multimodal AI assistant and developer workspace powered by the Clover 7-layer architecture.</Description>
</Properties>
```

### MSIX Visual Assets Packaged (`Assets\`)
- `StoreLogo.png` (`50x50`)
- `Square44x44Logo.png` (`44x44`)
- `Square44x44Logo.targetsize-24_altform-unplated.png` (`24x24` transparent unplated)
- `Square71x71Logo.png` (`71x71`)
- `Square150x150Logo.png` (`150x150`)
- `Wide310x150Logo.png` (`310x150`)
- `Square310x310Logo.png` (`310x310`)
- `SplashScreen.png` (`620x300`)

---

## 3. Signing & MSIX / Setup Architecture

- **Partner Center MSIX / `.msixupload` (`Option A`)**:
  - Microsoft Store automatically signs `.msix` / `.msixupload` packages with Microsoft's trusted Store certificate for `CN=A0DCEEB8-49DB-4CD8-8D07-BC439E69B9AF` upon Partner Center ingestion (no external paid EV/OV certificate required).
  - For local testing prior to Partner Center ingestion, `SanthoshNarayanan.CloverAIStudio_0.5.0.0_x64.msix` is signed with a local test certificate matching `Subject: CN=A0DCEEB8-49DB-4CD8-8D07-BC439E69B9AF` (Thumbprint `B6551BA01FD1C8B7D4BB103BB027A1203EB69FFC`, SHA-256).
  - `Launcher.cs` (`CloverAIStudio.exe` `0.5.0.0`) detects when it is launched from a read-only `WindowsApps` MSIX folder: if `%USERPROFILE%\CloverAI\venv\Scripts\pythonw.exe` exists, it syncs the updated `clover\` runtime from the MSIX package into `%USERPROFILE%\CloverAI\clover` and launches the local engine; if not yet set up, it invokes the bundled `CloverAIStudio-Setup.exe` (`--no-shortcuts`) on first launch.
- **Standalone Installer (`clover-ai-studio-beta-setup.exe`)**:
  - Compiled from `Setup.cs` (`0.5.0.0` / `0.5.0-beta.1`) embedding `CloverAIStudio.app.zip` (`175` files, `2,270,587` bytes).
  - Registers `HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\CloverAIStudio` with `DisplayName = "Clover AI Studio"`, `DisplayVersion = "0.5.0-beta.1"`, `Publisher = "Santhosh Narayanan"`, `HelpLink = "https://github.com/santhoshsathish94/clover-framework/issues"`, `URLInfoAbout = "https://cloverframework.com/"`, and `UninstallString = "\"<base>\\CloverAIStudio.exe\" --uninstall"`.

---

## 4. Verification & Regression Suite Results

Executed on Windows 11 x64 (`C:\personal\oss\scratch-agents\lfm3-hybridagent\laptop`):
- `python tests/test_semantics.py`: **`155/155` checks passed** (including byte-for-byte identity check against `clover-collective-wt/clover-ai-system/mobile/semantics.py`).
- `python clover/test_collective.py`: **`33/33` checks passed** (opt-in defaults, PII stripping, symbolic unlearning ripple resistance, Sybil deduplication, Byzantine poison rejection, HMAC-SHA256 verification, `Personal > Collective > General` priority, and consent withdrawal demotion).
- `python tests/test_clover_repo.py`: **`ALL PASS` (`27/27`)**.
- `python tests/test_tool_routing.py`: **`129/129` checks passed**.
- `python tests/test_regression_fixes.py`: **`ALL PASS` (`29/29`)**.
- **Live Server & Browser Verification (`http://127.0.0.1:8799`)**:
  - `GET /api/health`: `"status": "ready"`, `"version": "0.5.0-beta.1"`, `"channel": "beta"`.
  - `GET /api/collective`: `"enabled": true`, `"consent_contribute": false`, `"consent_receive": false`.
  - `POST /api/collective/consent`: verified toggling `contribute` and `receive` in UI (`Settings -> Advanced -> Clover layers -> Semantics -> CPSL`) and viewing `Shared core & candidate preview`.
  - Verified footer displays `Clover AI Studio v0.5.0-beta.1 (Beta) · Privacy Policy · Support`.
