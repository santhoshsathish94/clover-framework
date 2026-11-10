# Google Play Console — Closed Testing Submission Checklist (`ai.clover.studio`)

## 1. Release Artifacts & Signing

- **Upload Artifact (Google Play Console)**: `release/beta/android/clover-ai-studio-beta.aab`
  - Size: `81,182,832` bytes (`77.42 MB`)
  - SHA-256: `7cf3d1a074a290718a061446dfc2e65c6358d41ad233f1c7f80f56710fbe0f63`
- **Direct ADB / Tester APK**: `release/beta/android/clover-ai-studio-beta.apk`
  - Size: `82,259,445` bytes (`78.45 MB`)
  - SHA-256: `5026958ac088ace8884b8ab692603c17170775f74526a20648530cc80db97702`
- **Package Identity**:
  - `applicationId`: `ai.clover.studio`
  - `versionCode`: `16`
  - `versionName`: `5.0.2`
  - `minSdk`: `26` (Android 8.0)
  - `targetSdk` / `compileSdk`: `36` (Android 16)
- **Upload Keystore Certificate (`cloverstudio`)**:
  - Subject (`DN`): `CN=Clover AI Studio, OU=Clover Framework, O=Clover AI, L=Bengaluru, ST=KA, C=IN`
  - SHA-256 Fingerprint: `88:C0:E8:32:66:FF:ED:34:73:F5:1A:A1:02:C0:D5:EC:98:CA:4E:85:F4:D9:5F:FE:70:68:46:57:E1:9C:40:F7`
  - SHA-1 Fingerprint: `CB:D4:BF:0E:FC:81:1F:18:DB:52:A7:26:C2:46:67:0B:44:B0:1A:A3`
  - Play App Signing: Enable Play App Signing using this upload key.

---

## 2. Store Listing Metadata & Assets

- **App Name** (`listing/title.txt`, 16/30 chars): `Clover AI Studio`
- **Short Description** (`listing/short-description.txt`, 75/80 chars): `On-device multimodal AI assistant with 7-layer reasoning & private memory.`
- **Full Description** (`listing/full-description.txt`, 2,960/4,000 chars): See `listing/full-description.txt`
- **App Category**: `Productivity` (or `Tools`)
- **Contact Details**:
  - Website: `https://cloverframework.com/`
  - Privacy Policy URL: `https://cloverframework.com/privacy/`
  - Support URL: `https://github.com/santhoshsathish94/clover-framework/issues`
- **Visual Assets (`screenshots/`)**:
  - App Icon (`512x512` 32-bit PNG): `screenshots/app-icon-512x512.png`
  - Feature Graphic (`1024x500` PNG): `screenshots/feature-graphic-1024x500.png`
  - Phone Screenshots (`1080x1920` 9:16 PNG, 4 screenshots):
    1. `screenshots/phone-1-on-device-chat.png`
    2. `screenshots/phone-2-clover-layers-semantics.png`
    3. `screenshots/phone-3-collective-learning-preview.png`
    4. `screenshots/phone-4-settings-and-privacy.png`

---

## 3. App Content & Policy Declarations (Play Console → Policy → App content)

### A. Privacy Policy
- URL: `https://cloverframework.com/privacy/`

### B. Data Safety Questionnaire
- **Does your app collect or share any of the required user data types?** → **No** (by default all inference, chats, files, voice recordings, and Personal Process Learning stay strictly on the user's device).
- **Optional User-Initiated Features Disclosure**:
  - If the user explicitly connects their own private GitHub repository (`Settings → GitHub account`) or enables a web search provider (`Settings → Advanced → Tools and search`), data is transmitted over TLS (`HTTPS`) directly to GitHub (`api.github.com`) or the chosen search provider (`Open-Meteo` / `Brave` / `Tavily` / `Serper`).
  - Collective Problem-Solving Learning (`CPSL`) is **off by default** (`opt-in`) and strips all verbatim user text, paths, and identifiers locally before any candidate strategy is previewed or exported.
- **Data deletion**: Users can delete individual chats, clear all attachments (`Settings → Storage → Clear attachments`), delete all chats (`Settings → Storage → Delete all chats`), or wipe all learned personal preferences (`Settings → Advanced → Clover layers → What Clover learned → Forget everything Clover learned about you`) directly in the app.

### C. Foreground Service Permission Declaration (`FOREGROUND_SERVICE_SPECIAL_USE`)
- **Permission declared**: `android.permission.FOREGROUND_SERVICE_SPECIAL_USE`
- **Subtype declared in `AndroidManifest.xml`**: `Runs the on-device AI model and the app's local server, and downloads the model on first start`
- **Justification text for Play Console Reviewer**:
  > Clover AI Studio runs a 2-billion-parameter multimodal language model (`LFM3-Omni-2B`) locally on the user's phone via `llama-server` and a local loopback HTTP server (`127.0.0.1`). During the initial one-time model download (~1.8 GB from Hugging Face) and while generating a multi-step answer or spoken audio reply, `StudioService` runs as a visible foreground service with a persistent notification ("Clover runs on this phone") and a "Stop" action button so Android does not terminate the active on-device model mid-generation if the screen turns off or the user briefly switches apps to copy text.

### D. Other Permissions & Declarations
- **`RECORD_AUDIO`**: Used only when the user holds the Send button to record a voice message in chat (`<uses-feature android:name="android.hardware.microphone" android:required="false" />`).
- **`POST_NOTIFICATIONS`**: Displays the foreground service status and download progress notification.
- **Ads Declaration**: `No, my app does not contain ads.`
- **App Access (Sign-in Credentials)**: `All functionality is available without special access` (first launch prompts to download the open-weights model directly from Hugging Face; no login required).
- **Content Rating (IARC)**: Category `Utility, Productivity, Communication, or Other`; AI-generated responses from on-device model; expected rating `Everyone` / `PEGI 3`.
- **Target Audience**: `13+` (`13-15`, `16-17`, `18 and over`). Not designed for children (`Designed for Families: No`).
- **Government Apps / Financial Features / Health Apps**: `No` to all.

---

## 4. Closed Testing Track Release Steps

1. Open **Google Play Console** → Create or select app **Clover AI Studio** (`ai.clover.studio`).
2. Navigate to **Test and release → Testing → Closed testing** → select the **Alpha / Closed Beta** track → **Create new release**.
3. Under **App integrity / App signing**, confirm Play App Signing is active.
4. Upload `release/beta/android/clover-ai-studio-beta.aab` (`versionCode 16`, `versionName 5.0.2`, `targetSdk 36`).
5. Paste the contents of `release/beta/store/google-play/release-notes.md` into **Release notes (`en-US`)**.
6. Under **Testers**, select your beta tester email list / Google Group and copy the **Join on Android / Join on the Web** opt-in link.
7. Click **Next** → **Save and publish** to submit for Google Play Closed Testing review.
