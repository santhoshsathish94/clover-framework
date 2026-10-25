# Clover Kimi K3 for VS Code (Option A & Option B)

This folder provides both ways to use **Clover Kimi K3** (`93-Layer Kimi K3 MoE + HybridAgent-8B + 11-Pass Next-Concept Modeling + Ontology-Guided Neuro-Symbolic GraphRAG`) directly inside the **VS Code Chat & Agent Model Picker**:

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
        "id": "clover-kimi-k3",
        "name": "Clover Kimi K3 (93L MoE + HybridAgent + NCM)",
        "url": "https://api.cloverframework.com/v1/chat/completions",
        "apiType": "chat-completions",
        "toolCalling": true,
        "vision": true,
        "contextWindow": 262144,
        "maxOutputTokens": 16000,
        "streaming": true
      }
    ]
  }
]
```

2. Open the model picker in the VS Code Chat view and select **`Clover Kimi K3 (93L MoE + HybridAgent + NCM)`**.

---

## Option B — Standalone `clover-vscode-provider` Extension (For Any User / Any Account)

For users who want a one-click install (or whose GitHub Enterprise organization disables client BYOK providers), this extension registers `"vendor": "clover"` via VS Code's official `vscode.lm.registerLanguageModelChatProvider('clover', provider)` API:

### Install Locally
```bash
# Symlink or copy into ~/.vscode/extensions/
cp -r vscode-extension ~/.vscode/extensions/clover-framework.clover-vscode-provider-1.0.0
```
Reload VS Code (`Developer: Reload Window`), open the Chat model picker, and select **`Clover Kimi K3 (93L MoE + HybridAgent + NCM)`** under the **Clover** provider.
