# Clover AI Studio for VS Code (`v6.0.0`)

Connect **Clover AI Studio (`v6.0.0`)** — including local `LFM3-Omni-2B`, `0B` Computed Specialists (`x-cruncher`, Conformal Forecaster, Defeasible Resolver), and your configured `HG4` GPU servers (`Clover-35B-A3B`, `Nemotron 3.5 Super VL`, `Kimi K3.1 Pro`) — directly to VS Code's Copilot Chat & Agent Model Picker, File Viewer, Side-by-Side Diff, Shared Integrated Browser, Terminal, and Language Server Diagnostics.

---

## Option A — Self-Hosted `clover-local` Extension *(Standalone Desktop Installer: WIP)*

When running a local or self-hosted Clover `server.py` / `gateway.py` endpoint, [`clover-local-6.0.0.vsix`](../studio/studio/dist/clover-local-6.0.0.vsix) exposes your local and remote Clover models (`http://127.0.0.1:8765/v1/chat/completions`) inside VS Code's Chat model picker alongside two-way workbench tools (`vscode_editor`, `vscode_diff`, `vscode_diagnostics`, `vscode_browser`). *(Note: the standalone Windows `.exe` installer is currently Work in Progress — see [`../../download/README.md`](../../download/README.md).)*

---

## Option B — Built-In VS Code `Custom Endpoint` (`chatLanguageModels.json`)

VS Code (`1.105+`) also supports connecting directly to any Clover v6.0 Studio or `HG4` Fusion Gateway endpoint via `chatLanguageModels.json`:

```json
[
  {
    "name": "Clover",
    "vendor": "customendpoint",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "clover-35b-a3b",
        "name": "Clover — Clover-35B-A3B Coordinator (Qwen3.6-35B-A3B + LoRA, HG4-16R-OP)",
        "url": "http://127.0.0.1:8790/v1/chat/completions",
        "apiType": "chat-completions",
        "toolCalling": true,
        "vision": false,
        "contextWindow": 131072,
        "maxOutputTokens": 16000,
        "streaming": true
      },
      {
        "id": "nemotron-3.5-super-vl",
        "name": "Clover — Nemotron 3.5 Super VL (120B-A12B HG4-16R, 1M Context)",
        "url": "http://127.0.0.1:8790/v1/chat/completions",
        "apiType": "chat-completions",
        "toolCalling": true,
        "vision": false,
        "contextWindow": 1048576,
        "maxOutputTokens": 16000,
        "streaming": true
      },
      {
        "id": "lfm3-omni-2b",
        "name": "Clover — LFM3-Omni-2B (HG4-16R Fast Assistant)",
        "url": "http://127.0.0.1:8790/v1/chat/completions",
        "apiType": "chat-completions",
        "toolCalling": true,
        "vision": false,
        "contextWindow": 32768,
        "maxOutputTokens": 8192,
        "streaming": true
      },
      {
        "id": "kimi-k3",
        "name": "Clover — Kimi K3.1 Pro (2.8T MoE K31H)",
        "url": "http://127.0.0.1:8790/v1/chat/completions",
        "apiType": "chat-completions",
        "toolCalling": true,
        "vision": false,
        "contextWindow": 1048576,
        "maxOutputTokens": 16000,
        "streaming": true
      }
    ]
  }
]
```
