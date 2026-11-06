# Clover v5.0.0 — Release Readiness & Infrastructure Report

## 1. Release Summary (`v5.0.0`)

Clover **v5.0.0** unifies the 7-layer Clover AI architecture, Universal Clover AI Studio (`Windows`, `Android`, and `Web` on `k3`), Local Personal Process Learning (`PPL`), and Google OIDC–authenticated Collective Problem-Solving Learning (`CPSL`) into a single production release:

- **Framework & Studio Version**: `5.0.0` (Android `versionCode 12`, `versionName "5.0.0"`; Windows `5.0.0.0`).
- **Android 15/16 (`targetSdk 36`) Edge-to-Edge WindowInsets & Native Audio Capture**:
  - Dynamic `WindowInsets.Type.systemBars() | displayCutout() | ime()` inset padding verified on Samsung Galaxy S23 (`[0,98][1080,2295]`).
  - Added `MODIFY_AUDIO_SETTINGS` permission and native `CloverAndroidBridge` 16 kHz mono PCM WAV `AudioRecord` capture (`startAudioCapture` / `stopAudioCapture`) plus visible `#mic` button so voice recording works reliably on Android and Windows without browser modal alerts.
- **First-Turn Conversational Quality & 18-Category Human Usefulness Benchmark**:
  - `18/18` real-world human tasks pass on Turn 1 (`test_human_usefulness.py`).
- **Google Identity & End-to-End Collective Learning Integration**:
  - `35/35` checks pass in `test_collective.py`.
  - `32/32` checks pass in `test_google_identity_collective.py` (covering all 12 Acceptance Criteria, RS256 JWKS verification, Windows PKCE loopback, Android OIDC token flow, replay/Sybil/PII rejection, expiry/revocation rejection, and two-installation HTTP E2E synchronization).

---

## 2. External Infrastructure & Credentials Inventory

| Component | Status | Location / Configuration | Notes |
|---|---|---|---|
| **Collective Release Signing Key (Ed25519)** | **Configured & Active** | Private seed on `k3` at `/opt/clover-k3/web-studio/vault/cpsl_release_ed25519.key` (`mode 0600`); public key `dba3f7e2e969a26c57a24d50e2ecd654249b42e25eecc1f9140a514d7dcedb8a` pinned in `core_sync.py` | Private key never enters Git or client packages. |
| **Remote Collective Gateway & Registry** | **Deployed & Live** | `https://api.cloverframework.com/api/collective/gateway/{releases,contribute,withdraw}` on `k3` (`clover-web-studio.service`) | Persists pseudonymous contributor state in `/opt/clover-k3/web-studio/collective-gateway` and signed Git releases in `/opt/clover-k3/web-studio/clover-collective`. |
| **Android Keystore (`ai.clover.studio`)** | **Configured & Active** | `/opt/clover-k3/studio-android/keystore.env` (`SHA-256: 88:C0:E8:32:66:FF:ED:34:73:F5:1A:A1:02:C0:D5:EC:98:CA:4E:85:F4:D9:5F:FE:70:68:46:57:E1:9C:40:F7`) | Used to sign `.apk` (v1/v2/v3) and `.aab` for Google Play Console. |
| **Windows Authenticode Code-Signing** | **Configured & Active** | `CN=Clover Framework, O=Clover Framework` (`Thumbprint: 843BC8476BFCAD1EE2FABE82CDB403994CCE3F99`), public cert `download/laptop/CloverFramework-CodeSign.cer` | `CloverAIStudio-Setup.exe` is Authenticode-signed and auto-registers the publisher certificate on install. |
| **Google Cloud OAuth 2.0 Client IDs (`CLOVER_GOOGLE_CLIENT_ID`)** | **External Credential Configurable via Environment** | Default client ID `568061173330-cloveraistudio.apps.googleusercontent.com` in `google_identity.py`; override via `CLOVER_GOOGLE_CLIENT_ID` / `CLOVER_GOOGLE_CLIENT_IDS` (and optional `CLOVER_GOOGLE_CLIENT_SECRET` for Desktop OAuth clients) | In Google Cloud Console (project `568061173330`), ensure an **Android OAuth 2.0 Client ID** (package `ai.clover.studio`, SHA-1 `CB:D4:BF:0E:FC:81:1F:18:DB:52:A7:26:C2:46:67:0B:44:B0:1A:A3`) and a **Desktop App OAuth 2.0 Client ID** (redirect URI `http://127.0.0.1:<port>/api/identity/google/callback`) are registered and set in `CLOVER_GOOGLE_CLIENT_IDS`. Clover works 100% offline without Google sign-in. |
| **Google Play Production Track Graduation** | **In Progress on Google Play Console** | Closed Testing — Alpha track (`https://play.google.com/apps/testing/ai.clover.studio`) and Instant Internal App Sharing | Google Play policy requires 12 opted-in testers on Closed Testing for 14 continuous days before unlocking the public Production track. |
