# Clover AI Studio for VS Code (`v6.0.0`)

Connect **Clover AI Studio (`v6.0.0`)** — including local `LFM3-Omni-2B`, `0B` Computed Specialists (`x-cruncher`, Conformal Forecaster, Defeasible Resolver), and your configured `HG4` GPU servers (`Clover-35B-A3B`, `Nemotron 3.5 Super VL`, `Kimi K3.1 Pro`) — directly to VS Code's Copilot Chat & Agent Model Picker, File Viewer, Side-by-Side Diff, Shared Integrated Browser, Terminal, and Language Server Diagnostics.

---

## Option A — Bundled `clover-local-6.0.0.vsix` (Installed Automatically by `CloverAIStudio-Setup.exe`)

Installing [`CloverAIStudio-Setup.exe`](../../download/laptop/CloverAIStudio-Setup.exe) automatically installs [`clover-local-6.0.0.vsix`](../studio/studio/dist/clover-local-6.0.0.vsix), exposing your local and remote Clover Studio models (`http://127.0.0.1:8765/v1/chat/completions`) inside VS Code's Chat model picker alongside two-way workbench tools (`vscode_editor`, `vscode_diff`, `vscode_diagnostics`, `vscode_browser`).

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
        "url": "https://api.cloverframework.com/v1/chat/completions",
        "apiType": "chat-completions",
        "toolCalling": true,
        "vision": true,
        "contextWindow": 131072,
        "maxOutputTokens": 16000,
        "streaming": true
      },
      {
        "id": "nemotron-3.5-super-vl",
        "name": "Clover — Nemotron 3.5 Super VL (120B-A12B HG4-16R, 1M Context)",
        "url": "https://api.cloverframework.com/v1/chat/completions",
        "apiType": "chat-completions",
        "toolCalling": true,
        "vision": true,
        "contextWindow": 1048576,
        "maxOutputTokens": 16000,
        "streaming": true
      },
      {
        "id": "lfm3-omni-2b",
        "name": "Clover — LFM3-Omni-2B (HG4-16R Fast Assistant)",
        "url": "https://api.cloverframework.com/v1/chat/completions",
        "apiType": "chat-completions",
        "toolCalling": true,
        "vision": true,
        "contextWindow": 32768,
        "maxOutputTokens": 8192,
        "streaming": true
      },
      {
        "id": "kimi-k3",
        "name": "Clover — Kimi K3.1 Pro (2.8T MoE K31H)",
        "url": "https://api.cloverframework.com/v1/chat/completions",
        "apiType": "chat-completions",
        "toolCalling": true,
        "vision": true,
        "contextWindow": 1048576,
        "maxOutputTokens": 16000,
        "streaming": true
      }
    ]
  }
]
```
