# Microsoft Partner Center — Store Beta Submission Checklist (`9P2KP2XGF5LG`)

## 1. Exact Reserved Partner Center Identity

| Field | Value |
|---|---|
| **Product Name** | `Clover AI Studio` |
| **Publisher Display Name** | `Santhosh Narayanan` |
| **Package Identity Name** | `SanthoshNarayanan.CloverAIStudio` |
| **Publisher Distinguished Name** | `CN=A0DCEEB8-49DB-4CD8-8D07-BC439E69B9AF` |
| **Store ID** | `9P2KP2XGF5LG` |
| **Package Family Name** | `SanthoshNarayanan.CloverAIStudio_ffvna78masvs2` |
| **Package Version (`AppxManifest.xml`)** | `5.0.2.0` (Display Version: `5.0.2`) |
| **Target Architecture** | `x64` (`Windows.Desktop`, `MinVersion="10.0.19041.0"`, `MaxVersionTested="10.0.26100.0"`) |

---

## 2. Built Artifacts & Checksums

- **Partner Center Upload Bundle (Recommended — Option A)**: `release/beta/windows/clover-ai-studio-beta.msixupload`
  - Size: `4,738,436` bytes (`4.52 MB`)
  - SHA-256: `a1eba69e325dbfe64cdd9fdcf6f91d7f9fed790cdd41a41e405c04a1b929b621`
- **Standalone MSIX Package (`x64`)**: `release/beta/windows/SanthoshNarayanan.CloverAIStudio_5.0.2.0_x64.msix` (also copied as `clover-ai-studio-beta.msix`)
  - Size: `4,786,629` bytes (`4.56 MB`)
  - SHA-256: `061797c889d2a3ffc26b0b8dac148b4e9780cf617d7cf73831b228fe904cbb21`
- **Standalone Direct-Download Setup EXE (`x64`)**: `release/beta/windows/clover-ai-studio-beta-setup.exe`
  - Size: `9,349,592` bytes (`8.92 MB`)
  - SHA-256: `62c1dde2a52acc91b5cdd4fdd6962af46473a52646e5d5b3cb516c45c8cf2516`

---

## 3. Store Listing Metadata & Visual Assets

- **Product Name** (`listing/title.txt`): `Clover AI Studio`
- **Short Description** (`listing/short-description.txt`): `Local-first multimodal AI assistant and developer workspace powered by the Clover 7-layer architecture.`
- **Description** (`listing/full-description.txt`): See `listing/full-description.txt`
- **Release Notes** (`release-notes.md`): See `release-notes.md`
- **Category & Subcategory**: `Developer tools` (or `Productivity`)
- **Privacy Policy URL**: `https://cloverframework.com/privacy/`
- **Website URL**: `https://cloverframework.com/`
- **Support Contact / URL**: `https://github.com/santhoshsathish94/clover-framework/issues`
- **Screenshots & Logos (`screenshots/`)**:
  - Store Tile Icon (`300x300` PNG): `screenshots/app-icon-300x300.png`
  - Desktop Screenshots (`1920x1080` PNG, 4 screenshots):
    1. `screenshots/desktop-1-chat-workspace.png`
    2. `screenshots/desktop-2-seven-layer-architecture.png`
    3. `screenshots/desktop-3-collective-learning-cpsl.png`
    4. `screenshots/desktop-4-settings-and-privacy.png`

---

## 4. Pricing, Visibility, System Requirements & Age Ratings

### A. Pricing and Availability → Visibility (Beta Configuration)
- **Pricing**: `Free`
- **Discoverability / Visibility**:
  - Select **Make this product available but not discoverable in the Microsoft Store** → **Stop acquisition: Any customer with this product's direct link can see its Store listing**, **OR**
  - Configure a **Package flight** / **Private audience** group in Partner Center (`Product → Packages / Flights`) restricted to your beta testers' Microsoft accounts.

### B. System Requirements
- **Architecture**: `x64`
- **OS**: Windows 10 version 2004 (Build `19041`) or higher; Windows 11 recommended.
- **Memory**: Minimum `8 GB RAM` (`16 GB RAM` recommended).
- **Disk Space**: `4 GB` free disk space in `%USERPROFILE%\CloverAI` for the local `LFM3-Omni-2B` GGUF model and Python runtime.
- **GPU**: Optional DirectX 12 / Vulkan-compatible GPU (runs on AVX2 CPU if no compatible GPU is present).

### C. Age Ratings (IARC Questionnaire)
- **Product Type**: `Software / Productivity / Developer Tool`
- **Violence, Sexuality, Controlled Substances, Gambling**: `None`
- **User-to-User Interaction / Location Sharing**: `No`
- **Expected Rating**: `ESRB Everyone` / `PEGI 3` / `IARC 3+`

### D. Declarations (`runFullTrust` Restricted Capability)
- **Justification for `runFullTrust`**:
  > Clover AI Studio is a desktop developer workspace and local AI assistant (`Windows.FullTrustApplication`) that manages a local Python 3.12 virtual environment and local `llama.cpp` inference server in `%USERPROFILE%\CloverAI`, integrates with local VS Code workspaces, and executes user-authorized local Python and PowerShell developer tools.

### E. Notes for Certification (Reviewer Instructions)
> **Notes for Microsoft Store Certification Reviewer**:
> 1. Clover AI Studio runs a local AI model (`LiquidAI/LFM3-Omni-2B`) on the user's PC (`127.0.0.1:8765`). No account or login is required.
> 2. On first launch, `CloverAIStudio.exe` opens the **Clover AI Studio Setup** window, which checks hardware requirements and downloads the open-weights `LFM3-Omni-2B` model and local Python environment into `%USERPROFILE%\CloverAI` (verifying every file against its published SHA-256 hash).
> 3. Once setup completes, the main Clover AI Studio desktop window opens automatically at `http://127.0.0.1:8765/` in Microsoft Edge app mode (`--app`).
> 4. All local tools (Python, PowerShell, file access, web search) default to **Ask first** (`Settings → Advanced → Permissions`).

---

## 5. Partner Center Submission Steps

1. Sign in to **Microsoft Partner Center** → **Apps and games** → select **Clover AI Studio** (`Store ID: 9P2KP2XGF5LG`).
2. Click **Start your submission** (or create a **New package flight** for beta testers).
3. Open **Pricing and availability** → set Price to **Free** and Visibility to **Private audience** (or **Direct link only — hidden in Store**).
4. Open **Properties** → select Category **Developer tools**, enter Privacy Policy URL (`https://cloverframework.com/privacy/`) and Support URL (`https://github.com/santhoshsathish94/clover-framework/issues`), and provide the `runFullTrust` declaration above.
5. Open **Age ratings** → complete the IARC questionnaire and save.
6. Open **Packages** → drag and drop `release/beta/windows/clover-ai-studio-beta.msixupload` (or `SanthoshNarayanan.CloverAIStudio_5.0.2.0_x64.msix`), verify Partner Center parses `SanthoshNarayanan.CloverAIStudio` `5.0.2.0` (`x64`), and click **Save**.
7. Open **Store listings → English (United States)** → paste the contents of `listing/short-description.txt`, `listing/full-description.txt`, and `release-notes.md`, and upload the 4 desktop screenshots (`1920x1080`) and `app-icon-300x300.png` from `screenshots/`.
8. Open **Submission options** → paste the **Notes for certification** above → click **Submit to the Store**.
