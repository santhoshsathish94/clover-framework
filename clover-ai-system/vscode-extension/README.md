# Clover AI System for VS Code (Option A & Option B)

This folder provides both ways to use the **Clover AI System (`clover-ai-system`)** and all three switchable Layer-1 Foundation Model Backbones (`nemotron-3.5-super-vl`, `lfm3-1.7b`, and `clover-kimi-k3`, each governed by Clover's 7-layer stack with `1,048,576`-token session context) directly inside the **VS Code Chat & Agent Model Picker**:

---

## Option A — Built-In VS Code `Custom Endpoint` (`chatLanguageModels.json`, Zero Extension Install)

VS Code (`1.105+` / `copilot-chat 0.70+`) includes a native `customendpoint` provider.

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and run **`Chat: Manage Language Models`** (or click the model picker in the Chat panel $\to$ **`+`** $\to$ **`Custom Endpoint`**), or edit `%APPDATA%\Code\User\chatLanguageModels.json` (`~/.config/Code/User/chatLanguageModels.json` on Linux, `~/Library/Application Support/Code/User/chatLanguageModels.json` on macOS):

```json
[
  {
    "name": "Clover",
    "vendor": "customendpoint",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "nemotron-3.5-super-vl",
        "name": "Clover — Nemotron 3.5 Super VL (120B-A12B)",
        "url": "https://api.cloverframework.com/v1/chat/completions",
        "apiType": "chat-completions",
        "toolCalling": true,
        "vision": true,
        "contextWindow": 1048576,
        "maxOutputTokens": 16000,
        "streaming": true
      },
      {
        "id": "lfm3-1.7b",
        "name": "Clover — Liquid LFM3-1.7B (24L Tri-Hybrid)",
        "url": "https://api.cloverframework.com/v1/chat/completions",
        "apiType": "chat-completions",
        "toolCalling": true,
        "vision": true,
        "contextWindow": 1048576,
        "maxOutputTokens": 16000,
        "streaming": true
      },
      {
        "id": "clover-kimi-k3",
        "name": "Clover — Kimi K3 (2.78T MoE)",
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

2. Open the model picker in the VS Code Chat view and select **`Clover — Nemotron 3.5 Super VL (120B-A12B)`**, **`Clover — Liquid LFM3-1.7B (24L Tri-Hybrid)`**, or **`Clover — Kimi K3 (2.78T MoE)`**.

---

## Option B — Standalone `clover-vscode-provider` Extension (For Any User / Any Account)

For users who want a one-click install (or whose GitHub Enterprise organization disables client BYOK providers), this extension registers `"vendor": "clover"` via VS Code's official `vscode.lm.registerLanguageModelChatProvider('clover', provider)` API:

### Install Locally
```bash
# Symlink or copy into ~/.vscode/extensions/
cp -r vscode-extension ~/.vscode/extensions/clover-framework.clover-vscode-provider-4.1.2
```
Reload VS Code (`Developer: Reload Window`), open the Chat model picker, and select any of the three Clover models under the **Clover** provider.
