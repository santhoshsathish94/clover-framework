# Clover AI Studio — Android Mobile Edition (`Work in Progress`)

> **Work in Progress (`WIP`):** The standalone Android Mobile application (`ai.clover.studio`) is under active engineering and stabilization; public `.apk` / `.aab` binaries are currently withheld ([`../../download/README.md`](../../download/README.md)).

`clover-ai-system/mobile/` contains the in-progress Android packaging assets and native ARM64/Snapdragon C11 acceleration modules (`minSdk 26`, `targetSdk 36`, `compileSdk 36`).

---

## 1. Shared Core with Desktop & Web (`v6.0.0`)

Android Mobile embeds CPython `3.12` via Chaquopy (`clover.zip` built directly from [`../studio/clover/`](../studio/clover/)) and keeps [`android-app/assets/index.html`](android-app/assets/index.html) **byte-identical** to [`../studio/clover/web/app.html`](../studio/clover/web/app.html) (enforced by [`../studio/tests/test_studio_unified_ui.py`](../studio/tests/test_studio_unified_ui.py)).

Every Android install runs the exact same **4-Part Steerable Clover System** (`clover_v6_system.py`):
1. **`AGENTS.md` Steering & Epistemic Boundaries:** `Context → Direction → Execution → Outcome → Growth` + `STOP_AND_ASK` confirmation gates before risky actions.
2. **On-Device `0B` Computed Specialists (`libxcruncher.so` & Pure Python):**
   - Exact big-integer arithmetic (`A*B`, `A^B` past `6,000+` digits, `N!`, BBP hexadecimal $\pi$).
   - `0B` Periodogram + Split-Conformal time-series forecaster (`<50 ms` on mobile CPU).
   - `0B` Defeasible Epistemic Rule Resolver & Python AST verifier.
3. **On-Device `LFM3-Omni-2B` (`HG4-16R` / `Q4_0`):**
   - Packaged with Snapdragon `arm64-v8a` native libraries (`libllama_server.so`, `liblfm3_cpu_engine.so`, `libxcruncher.so`, Adreno OpenCL `libggml-opencl.so`, and Hexagon HTP `libggml-hexagon.so` / `v73..v81`).
4. **Interactive Server Pairing (`Clover System Setup`):**
   - Small models (`0B` Computed Pack, `LFM3-Omni-2B`, `Chronos-Bolt 0.2B`, `V-JEPA 2.1`) run interactively right on the phone (`0` external GPUs needed).
   - Users can also pair their phone with one or more remote GPU servers (`Clover-35B-A3B` on `1× 24 GB` GPU, `Nemotron 3.5 Super VL 120B` on `>=72 GB` GPU) from **Settings → Model → Clover System Setup** or on first launch via *Choose models / Use server*; CPU-only or undersized servers automatically skip big GPU models that cannot run interactively.

---

## 2. Privacy & Security on Android

- **Android KeyStore AES-256-GCM Vault:** API keys and GitHub OAuth tokens are encrypted on-device and never transmitted to third parties.
- **Loopback Cleartext Isolation:** `network_security_config.xml` restricts cleartext HTTP strictly to `127.0.0.1` for the embedded Python server.
- **Packaging Status:** Work in Progress (`WIP`) — see [`../../download/README.md`](../../download/README.md).
