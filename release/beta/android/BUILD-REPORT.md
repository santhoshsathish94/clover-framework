# Clover AI Studio — Android Google Play Closed Beta Build & Verification Report

**Release Version**: `0.5.0-beta.1`  
**Version Code**: `9`  
**Package Name**: `ai.clover.studio`  
**Build Date**: `2026-11-05`  
**Build Environment**: `k3` (`/opt/clover-k3/studio-android`), OpenJDK `17.0.20`, Gradle `8.9`, Android Gradle Plugin `8.5.2`, Chaquopy `17.2.0` (CPython `3.12`), Android SDK `34` (`minSdk 26`, `targetSdk 34`, `compileSdk 34`), Android NDK `llvm-strip`.

---

## 1. Built Artifacts & Checksums

| Artifact | Purpose | Size | SHA-256 |
|---|---|---:|---|
| `clover-ai-studio-beta.aab` | Google Play Console Closed Testing upload | `71 MB` (`73,832,870` bytes) | `b9271c7d7245daf2051bcfa2da7ff9252d1fa9f76d08cd4dd1070ac31ff329fe` |
| `clover-ai-studio-beta.apk` | Direct device / ADB installation & testing | `72 MB` (`74,912,051` bytes) | `f9ddebd66560f717b44973ba5486023b38093b3f374dd963eb83d94b2b2c48e0` |

---

## 2. Signing Verification (`apksigner` & `jarsigner`)

- **Upload Keystore**: Stored outside Git in `/opt/clover-k3/studio-android/keystore.env` (`mode 0600`), alias `cloverstudio`, RSA 2048-bit.
- **Certificate Subject (`DN`)**: `CN=Clover AI Studio, OU=Clover Framework, O=Clover AI, L=Bengaluru, ST=KA, C=IN`
- **Certificate SHA-256 Fingerprint**: `88c0e83266ffed3473f51aa102c0d5ec98ca4e85f4d95ffe70684657e19c40f7`
- **Certificate SHA-1 Fingerprint**: `cbd4bf0efc811f18db52a726c246670b44b01aa3`
- **APK Verification (`apksigner verify -v --print-certs`)**:
  - `Verifies`
  - `Verified using v2 scheme (APK Signature Scheme v2): true`
  - `Verified using v3 scheme (APK Signature Scheme v3): true`
- **AAB Verification (`jarsigner -verify -certs`)**:
  - `jar verified.` (0 unsigned entries, 0 `JarInputStream` order warnings).

---

## 3. Package & Manifest Audit (`aapt2 dump badging`)

```text
package: name='ai.clover.studio' versionCode='9' versionName='0.5.0-beta.1'
platformBuildVersionName='14' platformBuildVersionCode='34' compileSdkVersion='34'
sdkVersion:'26'
targetSdkVersion:'34'
application-label:'Clover AI Studio'
launchable-activity: name='ai.clover.studio.MainActivity'
```

### Permissions & Policy Hardening
- `android.permission.INTERNET`
- `android.permission.ACCESS_NETWORK_STATE`
- `android.permission.FOREGROUND_SERVICE`
- `android.permission.FOREGROUND_SERVICE_SPECIAL_USE` (`PROPERTY_SPECIAL_USE_FGS_SUBTYPE` declared on `StudioService`)
- `android.permission.POST_NOTIFICATIONS`
- `android.permission.WAKE_LOCK`
- `android.permission.RECORD_AUDIO` with `<uses-feature android:name="android.hardware.microphone" android:required="false" />`
- `android:allowBackup="false"`
- `android:networkSecurityConfig="@xml/network_security_config"` (`cleartextTrafficPermitted="false"` globally; `127.0.0.1` / `localhost` permitted for the embedded local server)
- Adaptive icons (`mipmap-anydpi-v26/ic_launcher.xml` and `mipmap-anydpi-v26/ic_launcher_round.xml` with `@drawable/ic_launcher_background` and 5-leaf Clover vector `@drawable/ic_launcher_foreground`) included across all densities.
- Unconditional lock-screen bypass (`setShowWhenLocked` / `setTurnScreenOn`) removed from `MainActivity.onCreate()`.
- Hardcoded owner key (`_K3_OBF`) and `SM-S911` / `SDK_GPHONE` auto-unlock heuristics removed from `k3_host_engine.py`.

---

## 4. Functional Smoke Test & Collective Learning Verification

Tested against the unpacked `clover.zip` (`133 files, 749,022 bytes`, including `local_tool_packs.py` Phase 1 & Phase 2 deterministic tool packs and `CLV-001`..`CLV-007` fixes) running under `sim_phone.py` on `k3` (`http://127.0.0.1:47651`):
1. **Authentication Gate**: `GET /api/setup` without token returns `403`; with `X-Clover-Token` header or `clover_token` cookie returns `200`.
2. **Health & Version (`GET /api/health` & `GET /api/settings`)**:
   - `"status": "ready"`, `"version": "0.5.0-beta.1"`, `"channel": "beta"`
   - `"privacy_policy_url": "https://cloverframework.com/privacy/"`
   - `"support_url": "https://github.com/santhoshsathish94/clover-framework/issues"`
   - `"owner": false` (no owner credentials exposed)
3. **Collective Problem-Solving Learning (`GET /api/collective` & `POST /api/collective/consent`)**:
   - Defaults to opt-in (`"consent_contribute": false, "consent_receive": false`).
   - `POST /api/collective/consent` toggles `consent_contribute` and `consent_receive` independently and persists to `<base>/collective` and `settings.json`.
   - `python3.12 test_collective.py` inside the packaged `clover.zip`: **`33/33` checks passed**.
4. **End-to-End Chat Stream (`POST /api/chat/stream`)**:
   - Prompt `"What is 15% of 80?"` streamed `semantics` (`understood: question`) and `done` (`reply: "15% of 80 is 12."`, `40.48 tok/s`, `wall_ms: 575.0`).
