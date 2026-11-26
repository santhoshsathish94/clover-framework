# Clover AI Studio — `v6.0.0` Release Packaging & Architecture Summary

**Release Date:** `2026-11-26`  
**Product Name:** `Clover AI Studio` (`v6.0.0`)  
**Publisher:** `Santhosh Narayanan`

---

## 1. Unified Multi-Platform Core (`clover-ai-system/studio/clover/`)

Windows Desktop (`CloverAIStudio-Setup.exe`), Android Mobile (`CloverAIStudio.apk` & `clover-ai-studio-beta.aab`, `versionCode 22`), Web Studio (`https://cloverframework.com/clover-ai/` & `https://api.cloverframework.com`), and VS Code (`clover-local-6.0.0.vsix`) are built from the same shared core (`server.py`, `clover_v6_system.py`, `local_tool_packs.py`, `k3_host_engine.py`, `web/app.html`, `web/setup.html`):

- **4-Part Steerable Clover System:** `AGENTS.md` + `mcp.json` (`0B` Computed Specialists + `MCP 2.0` specialist routes) + `<task_id>.md` Context + SQLite Prerequisite-Gated `CloverTodoStore` + `CloverMetricsLedger` (`GET /api/clover/metrics`).
- **Interactive Hardware-Gated Model & Server Setup:** Users can configure `1` model, a custom subset, or the full 9-component Clover AI System across their device and provided GPU servers; small models run on any phone/laptop/CPU, while big GPU models verify live decode `tok/s` and skip CPU-only/undersized machines with explicit minimum server specs.
- **Security & Parity Verification:** Strict SSH/HTTP target validation (blocks `-oProxyCommand` option injection and Web SSRF), OS-backed credential encryption (Windows DPAPI & Android KeyStore AES-256-GCM), byte-identical `web/app.html` ↔ `mobile/android-app/assets/index.html` UI sync (`41/41` `test_studio_unified_ui.py`), and verified SHA-256 parity across [`RELEASE-MANIFEST.json`](RELEASE-MANIFEST.json), [`windows/BUILD-REPORT.md`](windows/BUILD-REPORT.md), [`android/BUILD-REPORT.md`](android/BUILD-REPORT.md), and [`../../download/SHA256SUMS.txt`](../../download/SHA256SUMS.txt).
