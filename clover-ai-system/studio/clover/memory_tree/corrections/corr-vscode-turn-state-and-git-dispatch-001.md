---
{
  "schema_version": "1.0",
  "id": "MEM-CORR-VSCODE-TURN-STATE-AND-GIT-DISPATCH-001",
  "title": "Correction: Preserve `tool_rounds` Across Tool-Result Envelopes and Dispatch `run_in_terminal` Directly for Git/Branch Inspections",
  "entry_type": "correction",
  "epistemic_status": "observed",
  "status": "active",
  "concepts": [
    "vscode-agent",
    "tool-loop",
    "turn-state",
    "git-dispatch",
    "envelope-stripping"
  ],
  "tags": [
    "vscode",
    "chat-server",
    "correction",
    "git"
  ],
  "created_at": "2026-10-26T03:04:06Z",
  "updated_at": "2026-10-26T03:04:06Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "commit:494f36f; docs/clover-generative-compilation-research-context.md#section-4.3",
    "actor": "human_and_assistant",
    "recorded_at": "2026-10-26T03:04:06Z"
  },
  "relationships": [
    {
      "type": "refines",
      "target": "MEM-OBS-VSCODE-TRAILING-CONTEXT-WIPES-TOOLS-001",
      "provenance": "commit:494f36f",
      "confidence": 0.95,
      "context_scope": "vscode_turn_state",
      "created_at": "2026-10-26T03:04:06Z",
      "note": "Fixes the turn-state and envelope-stripping bug observed in session `vsc_ff92c0c2-2e2c-4ca3-958f-1e249dcd2164`."
    },
    {
      "type": "derived_from",
      "target": "MEM-OBS-VSCODE-TRAILING-CONTEXT-WIPES-TOOLS-001",
      "provenance": "commit:494f36f",
      "confidence": 1.0,
      "context_scope": "vscode_turn_state",
      "created_at": "2026-10-26T03:04:06Z",
      "note": "Directly derived from the session store diagnosis."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-26T03:04:06Z",
      "actor": "human_and_assistant",
      "action": "created",
      "reason": "Initial correction entry recorded from commit:494f36f; docs/clover-generative-compilation-research-context.md#section-4.3"
    }
  ]
}
---

# Correction: Preserve `tool_rounds` Across Tool-Result Envelopes and Dispatch `run_in_terminal` Directly for Git/Branch Inspections

## Summary
Fixes in commit `494f36f`: suppress trailing `{role: 'user'}` envelope emission when a message carries `LanguageModelToolResultPart`s, clear `tool_rounds` only when a turn genuinely starts or completes, strip `<context>`/`<attachment>`/browser-page blocks without raw fallback, and dispatch git/branch/worktree inspections directly to `run_in_terminal`.

## Content
Resolves the mid-turn `tool_rounds` wipe in `MEM-OBS-VSCODE-TRAILING-CONTEXT-WIPES-TOOLS-001`. Verified on `gpu` (`127.0.0.1:8765/v1/chat/completions`) by replaying both prompts from session `vsc_ff92c0c2-2e2c-4ca3-958f-1e249dcd2164` and running live headless VS Code `1.143.1` (`allTasksPassed: true` across 3-round `list_dir -> run_in_terminal -> read_file`, 2-round `list_dir -> read_file`, and 0-round math).

## Original Source Evidence
Commit `494f36f` (`4 files changed, 195 insertions(+), 28 deletions(-)`) and live headless VS Code `1.143.1` verification (`run_vscode_verify3.sh`: `allTasksPassed: true`).

## Revision History
- `2026-10-26T03:04:06Z` (**human_and_assistant**, `created`): Initial correction entry recorded from commit:494f36f; docs/clover-generative-compilation-research-context.md#section-4.3
