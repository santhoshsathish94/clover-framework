# Clover AI Studio — Repository & Beta Release Architecture Audit

**Date:** 2026-11-05  
**Product Name:** Clover AI Studio  
**Publisher:** Santhosh Narayanan  
**Branch / Worktree:** `experiment/collective-learning` (`C:\personal\oss\clover-collective-wt` & `/opt/clover-k3/collective-wt`)

---

## 1. Repository Architecture Discovered

Inspection of the workspace revealed two distinct code trees that work together:

1. **Shared Application Core (`C:\personal\oss\scratch-agents\lfm3-hybridagent\laptop\clover\`):**
   - A unified Python + Web UI application (`server.py`, `web/app.html`, `web/setup.html`, `semantics.py`, `personal.py`, `omni_native.py`, `lfm3_native.py`, `clover_repo.py`, `vault.py`, `permissions.py`, `workspaces.py`, `launch/manifest.json`, `launch/manifest-android.json`) that powers **both** the Windows desktop application and the Android application.
   - Serves the UI and JSON/SSE APIs over loopback (`127.0.0.1:8765` on Windows, `127.0.0.1:47651` on Android) with strict `Host`, `Origin`, `Sec-Fetch-Site`, Content-Security-Policy (per-response cryptographic nonce), and token/cookie checks.
2. **Android Application (`C:\personal\oss\scratch-agents\lfm3-hybridagent\android\`, built on `k3` at `/opt/clover-k3/studio-android/`):**
   - **Build System:** Gradle `8.9`, Android Gradle Plugin `8.5.2`, Chaquopy `17.2.0` (embedding CPython `3.12` + Android native wheels `numpy`, `pandas`, `matplotlib`, `openpyxl`, `xlrd`, `pypdf`, `python-docx`, `python-pptx`), OpenJDK `17.0.20`, Android SDK `compileSdk 34` / `targetSdk 34` / `minSdk 26`.
   - **Application Identity:** `applicationId "ai.clover.studio"`, currently `versionCode 8`, `versionName "3.0.1"`.
   - **Architecture:** `MainActivity.java` (WebView UI with cookie token auth and file/mic permission bridges), `StudioService.java` (foreground `specialUse` service launching embedded CPython `android_main.py` and `libllama_server.so` from `nativeLibraryDir`), and `VaultCrypto.java` (AES-256-GCM backed by Android KeyStore).
   - **Native Engine:** Packages `llama.cpp` release `b12012` Snapdragon `arm64-v8a` binaries (`libllama_server.so`, `libggml-opencl.so`, `libggml-hexagon.so`, `libggml-htp-v73..v81.so`) into `jniLibs/arm64-v8a` and bundles `laptop/clover` as `python/clover.zip`.
   - *(Note: An older standalone WebView shell exists in `clover-ai-system/mobile/android-app` at `versionCode 6` / `2.1.0`; the shipped `74.85 MB` `download/mobile/CloverAIStudio.apk` is built from `scratch-agents/lfm3-hybridagent/android` on `k3`.)*
3. **Windows Application (`C:\personal\oss\scratch-agents\lfm3-hybridagent\laptop\studio\`):**
   - **Build System:** Built with Microsoft's built-in `.NET Framework 4.8` x64 C# compiler (`C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe`) via `studio/build_setup.ps1`.
   - **Executable Entry Point (`Launcher.cs` $\to$ `CloverAIStudio.exe`):** Native x64 WinForms launcher (`v0.4.1.0`) that starts `venv\Scripts\pythonw.exe clover\server.py` on `127.0.0.1:8765`, opens an app-mode Microsoft Edge window (`--app=http://127.0.0.1:8765/`), and shuts down the engine on window close (unless VS Code is connected). Supports `--engine-only`, `--stop`, and `--uninstall [--quiet]`.
   - **Installer (`Setup.cs` $\to$ `CloverAIStudio-Setup.exe`):** Native x64 C# WinForms/CLI installer (`v0.4.1.0`, `2.38 MB`, embedding `CloverAIStudio.app.zip`) that checks hardware, unpacks `CloverAIStudio.exe` + `clover/` + `studio/` into `%USERPROFILE%\CloverAI`, bootstraps `uv 0.13.6`, `Python 3.12.16`, locked wheels (`requirements.lock`), and `MinGit`, sets `config\setup-walkthrough.flag`, creates Start Menu/Desktop shortcuts, installs the `clover-local` VS Code extension, and registers `HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\CloverAIStudio`.

---

## 2. Six-Point Assessment Before Implementation

### 2.1 What Already Works
- **Android:** Reproducible Gradle + Chaquopy + Snapdragon `llama.cpp` build on `k3`; external release keystore (`keystore.env`, mode `0600`, outside Git); `network_security_config.xml` restricting cleartext HTTP strictly to `127.0.0.1`; Android KeyStore AES-256-GCM vault; first-launch SHA-256 verified model download (`setup.html`).
- **Windows:** Reproducible C# compilation of `CloverAIStudio.exe` and `CloverAIStudio-Setup.exe`; Windows DPAPI secret storage (`vault.py`); SHA-256 verification of all downloaded archives, wheels, and GGUF models; clean per-user install and uninstall.
- **Shared Core:** `semantics.py`, `personal.py`, and the new `collective.py` / `core_sync.py` modules (`103` semantics checks, `48` personal checks, `33` collective checks, `13/13` S1–S3 suites passing).

### 2.2 What Is Missing for Google Play (Closed Testing)
1. **Android App Bundle (`.aab`) Generation:** `build_apk.sh` currently runs only `assemblePhoneRelease` (`.apk`). Google Play requires a signed `.aab` (`bundlePhoneRelease`).
2. **Adaptive Launcher Icons (`mipmap-anydpi-v26`):** `scratch-agents/lfm3-hybridagent/android/app/src/main/res/` only has legacy PNGs (`mipmap-mdpi..xxxhdpi/ic_launcher.png`) and lacks `mipmap-anydpi-v26/ic_launcher.xml`, `ic_launcher_round.xml`, and adaptive foreground/background drawables.
3. **Five-Leaf Clover Splash & Lockscreen Flags in `MainActivity.java`:** The inline HTML splash in `MainActivity.java` draws 4 circles rotating counter-clockwise (`-360deg`) instead of the 5-leaf Clover mark rotating clockwise (`360deg`), and `onCreate()` unconditionally calls `setShowWhenLocked(true)` and `setTurnScreenOn(true)` (an emulator artifact that should not be active in production/beta).
4. **Hardcoded Obfuscated Owner Secret (`_K3_OBF`) in `k3_host_engine.py`:** `k3_host_engine.py` embeds an XOR-obfuscated owner API key (`_K3_OBF`) and auto-unlocks it on any device whose model string contains `SM-S911` (all Samsung Galaxy S23 phones!) or `SDK_GPHONE`. This must be removed so no secret is embedded in distributed binaries.
5. **Version & Beta Identification:** Needs incremented `versionCode 9` and `versionName "0.5.0-beta.1"` (consistent across Android & Windows) plus ProGuard/R8 keep rules and privacy/support links in Settings.

### 2.3 What Is Missing for Microsoft Store
1. **MSIX Packaging with Reserved Partner Center Identity:**
   - Reserved identity: `Name="SanthoshNarayanan.CloverAIStudio"`, `Publisher="CN=A0DCEEB8-49DB-4CD8-8D07-BC439E69B9AF"`, Store ID `9P2KP2XGF5LG`, PFN `SanthoshNarayanan.CloverAIStudio_ffvna78masvs2`, four-part version `0.5.0.0`.
2. **Read-Only `WindowsApps` Execution Support in `Launcher.cs`:**
   - When installed via MSIX (`Windows.FullTrustApplication`), `CloverAIStudio.exe` runs from `C:\Program Files\WindowsApps\SanthoshNarayanan.CloverAIStudio_...`, which is read-only. Currently `Launcher.cs` sets `baseDir = Path.GetDirectoryName(exePath)` and errors if `venv\Scripts\pythonw.exe` is not in the same folder as `CloverAIStudio.exe`.
   - `Launcher.cs` must detect when it runs from an MSIX / read-only package folder (or when `venv` is not yet provisioned in `%USERPROFILE%\CloverAI`) and automatically run the embedded setup/bootstrap into `%USERPROFILE%\CloverAI` on first launch.
3. **Publisher String in `Setup.cs` & `Launcher.cs`:**
   - `Setup.cs` writes `Publisher = "Clover"` in the Uninstall registry key; should be `"Santhosh Narayanan"` to match the reserved Store publisher identity.

### 2.4 What Needs to Change in the Android Application
- Sync the new Collective Learning modules (`collective.py`, `collective_gateway.py`, `collective_evaluator.py`, `core_registry.py`, `core_sync.py`) and updated `semantics.py` / `personal.py` / `server.py` / `web/app.html` into `laptop/clover` (which packs into `clover.zip` for Android).
- Remove `_K3_OBF` and device-name owner auto-unlock from `k3_host_engine.py`.
- Add `mipmap-anydpi-v26` adaptive icons + five-leaf Clover foreground/background to `android/app/src/main/res/`.
- Update `MainActivity.java` splash to the 5-leaf Clover SVG rotating clockwise (`360deg`) and remove unconditional `setShowWhenLocked(true)` / `setTurnScreenOn(true)`.
- Add `proguard-rules.pro`, bump `versionCode` to `9` and `versionName` to `"0.5.0-beta.1"`, and update `build_apk.sh` on `k3` to build and sign **both** the release `.aab` (`bundlePhoneRelease`) and release `.apk` (`assemblePhoneRelease`).

### 2.5 What Needs to Change in the Windows Application
- Update `Launcher.cs` and `Setup.cs` to version `0.5.0.0` (`0.5.0-beta.1` product version), publisher `"Santhosh Narayanan"`, and embed `CloverAIStudio.app.zip` so `CloverAIStudio.exe` can self-bootstrap into `%USERPROFILE%\CloverAI` when launched from an MSIX package (`WindowsApps`).
- Generate all required MSIX visual assets (`Square44x44Logo`, `Square150x150Logo`, `Wide310x150Logo`, `LargeTile`, `StoreLogo`, `SplashScreen`) from the official 5-leaf Clover icon.
- Create `AppxManifest.xml` with the exact reserved identity (`SanthoshNarayanan.CloverAIStudio`, `CN=A0DCEEB8-49DB-4CD8-8D07-BC439E69B9AF`, `Version="0.5.0.0"`, `Arch="x64"`, `MinVersion="10.0.19041.0"`, `<rescap:Capability Name="runFullTrust" />`).

### 2.6 Which Packaging Route Is Most Appropriate for Windows (Option A vs. Option B)
- **Why Option B (Win32 EXE Store Submission) Is Blocked for Partner Center:**
  1. Microsoft Store policy for Win32 `.exe` packages strictly forbids installation-time network downloads ("No downloader stub. No installation-time downloading of application binaries"). `CloverAIStudio-Setup.exe` downloads `uv`, `Python 3.12.16`, Python wheels (`torch`, `spacy`), and `MinGit` during installation.
  2. Microsoft Store does *not* sign `.exe` installers; submitting an `.exe` requires purchasing an external CA-trusted OV/EV Authenticode code-signing certificate.
- **Why Option A (MSIX + `.msixupload`) Is Selected for Microsoft Store (While Retaining `.exe` for Direct Download):**
  1. By embedding `CloverAIStudio.app.zip` in `CloverAIStudio.exe` and deferring runtime/model provisioning to **first application launch** inside `%USERPROFILE%\CloverAI` (matching the Android app's first-launch download architecture), the MSIX installation itself is 100% offline, instant, per-user, and requires no elevation.
  2. Microsoft Partner Center automatically signs `.msix` / `.msixupload` packages with the reserved publisher certificate (`CN=A0DCEEB8-49DB-4CD8-8D07-BC439E69B9AF`) upon Store ingestion.
  3. We will produce **both** the Store-ready `.msix` / `.msixupload` package (`clover-ai-studio-beta.msixupload` & `clover-ai-studio-beta.msix`) AND the standalone direct-distribution `clover-ai-studio-beta-setup.exe`.
