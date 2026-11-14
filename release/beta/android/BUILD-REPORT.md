# Clover AI Studio — Android Google Play Release Build & Verification Report

> **Generated deterministically by `release/beta/generate_release_reports.py` from final build artifacts.**

**Release Version**: `5.0.6`  
**Version Code**: `20`  
**Package Name**: `ai.clover.studio`  
**Build Date**: `2026-11-11`  
**Build Environment**: `k3` (`/opt/clover-k3/studio-android`), OpenJDK `17.0.20`, Gradle `8.9`, Android Gradle Plugin `8.5.2`, Chaquopy `17.2.0` (CPython `3.12`), Android SDK `36` (`minSdk 26`, `targetSdk 36`, `compileSdk 36`), Android NDK `llvm-strip`.

---

## 1. Built Artifacts & Checksums

| Artifact | Mirror Path | Purpose | Size | SHA-256 |
|---|---|---|---:|---|
| `release/beta/android/clover-ai-studio-beta.aab` | — | Google Play Console upload (`versionCode 20`, `targetSdk 36`) | `81,563,089` bytes (`77.78 MB`) | `c895d129af80bb0ce21e87e4e971bfa585b18ef20adb73ce660dc87b721e4f6b` |
| `release/beta/android/clover-ai-studio-beta.apk` | `download/mobile/CloverAIStudio.apk` | Direct device / ADB installation & testing (`versionCode 20`, `targetSdk 36`) | `82,639,595` bytes (`78.81 MB`) | `4568fca5d957ed5eaf4c789879e8224dae820393b97a03e7ccab3596e1083d3a` |

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
  - Client builds embed only the 32-byte RFC 8032 Ed25519 public verification key (`a4d9d040035a19a5f1214bd30881b40842d4b06d0990c1bfdd6252fd90280283`) in `core_sync.py`.

---

## 3. Package & Manifest Audit (`build.gradle`)

```text
package: name='ai.clover.studio' versionCode='20' versionName='5.0.6'
sdkVersion:'26'
targetSdkVersion:'36'
compileSdkVersion:'36'
application-label:'Clover AI Studio'
launchable-activity: name='ai.clover.studio.MainActivity'
```

---

## 4. Verification (re-run for this release)

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

**Google Play Console Links**:
   - Instant Test (Internal App Sharing): `https://play.google.com/apps/test/RQvTblUjN2Y/ahAO29uNTcFWSSv5oQlYxX4CR1FRn-gmS9nbIh1lp2D4iNBgJ3-XLPM1fzjbC2vHYtpYtoJmtPROkQr48qz_-RbaOs`
   - Closed Testing Opt-In: `https://play.google.com/apps/testing/ai.clover.studio`
