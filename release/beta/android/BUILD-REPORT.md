# Clover AI Studio — Android Google Play Closed Beta Build & Verification Report

> **Generated deterministically by `release/beta/generate_release_reports.py` from final build artifacts.**

**Release Version**: `0.5.0-beta.1`  
**Version Code**: `11`  
**Package Name**: `ai.clover.studio`  
**Build Date**: `2026-11-05`  
**Build Environment**: `k3` (`/opt/clover-k3/studio-android`), OpenJDK `17.0.20`, Gradle `8.9`, Android Gradle Plugin `8.5.2`, Chaquopy `17.2.0` (CPython `3.12`), Android SDK `36` (`minSdk 26`, `targetSdk 36`, `compileSdk 36`), Android NDK `llvm-strip`.

---

## 1. Built Artifacts & Checksums

| Artifact | Mirror Path | Purpose | Size | SHA-256 |
|---|---|---|---:|---|
| `release/beta/android/clover-ai-studio-beta.aab` | — | Google Play Console Closed Testing upload (`versionCode 11`, `targetSdk 36`) | `73,847,994` bytes (`70.43 MB`) | `7f186a955e3a30cc1e5aaddff990854b770ab224a46481ccac49924ab3749b6d` |
| `release/beta/android/clover-ai-studio-beta.apk` | `download/mobile/CloverAIStudio.apk` | Direct device / ADB installation & testing (`versionCode 11`, `targetSdk 36`) | `74,927,154` bytes (`71.46 MB`) | `7c2d9b5431d00b58e392d78f7620df041c7a613a49b8de2fbd2cbdb1fdadc09e` |

---

## 2. Signing Verification (`apksigner` & `jarsigner`)

- **Upload Keystore**: Stored outside Git in `/opt/clover-k3/studio-android/keystore.env` (`mode 0600`), alias `cloverstudio`, RSA 2048-bit.
- **Certificate Subject (`DN`)**: `CN=Clover AI Studio, OU=Clover Framework, O=Clover AI, L=Bengaluru, ST=KA, C=IN`
- **Certificate SHA-256 Fingerprint**: `88:C0:E8:32:66:FF:ED:34:73:F5:1A:A1:02:C0:D5:EC:98:CA:4E:85:F4:D9:5F:FE:70:68:46:57:E1:9C:40:F7` (`88c0e83266ffed3473f51aa102c0d5ec98ca4e85f4d95ffe70684657e19c40f7`)
- **Certificate SHA-1 Fingerprint**: `CB:D4:BF:0E:FC:81:1F:18:DB:52:A7:26:C2:46:67:0B:44:B0:1A:A3` (`cbd4bf0efc811f18db52a726c246670b44b01aa3`)
- **APK Verification (`apksigner verify --print-certs`)**:
  - `Verified using v2 scheme (APK Signature Scheme v2): true`
  - `Verified using v3 scheme (APK Signature Scheme v3): true`
- **AAB Verification (`jarsigner -verify -certs`)**:
  - `jar verified.`
- **CPSL Ed25519 Asymmetric Release Verification**:
  - Client builds embed only the 32-byte RFC 8032 Ed25519 public verification key (`dba3f7e2e969a26c57a24d50e2ecd654249b42e25eecc1f9140a514d7dcedb8a`) in `core_sync.py`.

---

## 3. Package & Manifest Audit (`build.gradle`)

```text
package: name='ai.clover.studio' versionCode='11' versionName='0.5.0-beta.1'
sdkVersion:'26'
targetSdkVersion:'36'
compileSdkVersion:'36'
application-label:'Clover AI Studio'
launchable-activity: name='ai.clover.studio.MainActivity'
```

---

## 4. Functional Smoke Test, Live Activity SSE & Collective Learning Verification

1. **Collective Problem-Solving Learning (`test_collective.py`)**: **`35/35` checks passed** (RFC 8032 Ed25519 public-key verification, unauthenticated-publish refusal, forged-key rejection, PII stripping, Sybil deduplication, Byzantine poison rejection, `Personal > Collective > General` priority).
2. **Live Activity SSE Timeline (`test_live_activity_sse.py`)**: **`18/18` checks passed** (monotonic `seq` numbers, `/api/cancel` & `/api/stop` mid-turn cancellation, broken-pipe client disconnect handling, `/api/turn/events` & `resume_turn_id` reconnection replay, and concurrent queued stream execution).
3. **Google Play Console Links**:
   - Instant Test (Internal App Sharing): `https://play.google.com/apps/test/RQvTblUjN2Y/ahAO29uNRoJQ1q2vSWcsS_HamuIOmmlbmpz4YZmTRB6qoomiOmSj5pK7Qz75RY537EnOpdUc1h4mT_pQkTYd3smKNS`
   - Closed Testing Opt-In: `https://play.google.com/apps/testing/ai.clover.studio`
