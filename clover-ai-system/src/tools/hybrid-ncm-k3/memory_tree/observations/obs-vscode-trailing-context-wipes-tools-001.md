---
{
  "schema_version": "1.0",
  "id": "MEM-OBS-VSCODE-TRAILING-CONTEXT-WIPES-TOOLS-001",
  "title": "Observation: A Trailing Editor `<context>` User Envelope Cleared `tool_rounds` Mid-Turn in `/v1/chat/completions`",
  "entry_type": "observation",
  "epistemic_status": "observed",
  "status": "active",
  "concepts": [
    "vscode-agent",
    "tool-loop",
    "envelope-stripping",
    "turn-state"
  ],
  "tags": [
    "vscode",
    "chat-server",
    "observation",
    "failure"
  ],
  "created_at": "2026-10-26T02:15:00Z",
  "updated_at": "2026-10-26T03:04:06Z",
  "provenance": {
    "source_type": "observation",
    "source_ref": "docs/clover-generative-compilation-research-context.md#section-4.3 (session vsc_ff92c0c2-2e2c-4ca3-958f-1e249dcd2164); commit:494f36f",
    "actor": "system_measurement",
    "recorded_at": "2026-10-26T02:15:00Z"
  },
  "relationships": [
    {
      "type": "exemplifies",
      "target": "MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
      "provenance": "commit:494f36f",
      "confidence": 0.85,
      "context_scope": "state_reset_bug",
      "created_at": "2026-10-26T02:15:00Z",
      "note": "Wiping `tool_rounds` mid-turn made downstream synthesis see 0 tool outputs even after tools had executed."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-26T02:15:00Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Initial observation entry recorded from docs/clover-generative-compilation-research-context.md#section-4.3 (session vsc_ff92c0c2-2e2c-4ca3-958f-1e249dcd2164); commit:494f36f"
    }
  ]
}
---

# Observation: A Trailing Editor `<context>` User Envelope Cleared `tool_rounds` Mid-Turn in `/v1/chat/completions`

## Summary
When VS Code returned a `LanguageModelToolResultPart` alongside `<context>` or browser-page metadata inside a `User`-role message, `_convertMessages` emitted a trailing `{role: 'user'}` message after `{role: 'tool'}`, triggering `if pending_u and tool_rounds: tool_rounds.clear()` and wiping verified tool outputs mid-turn.

## Content
Diagnosed from `/home/usergpu/merge_lf/out/sessions_store.json` (`vsc_ff92c0c2-2e2c-4ca3-958f-1e249dcd2164`): pre-fix `_strip_vscode_envelope` did not strip `<context>`, `<attachment>`, or plain-text `'Relevant context from VS Code editor: The following browser pages...'`, and fell back to `u_text.strip()` when a message was pure envelope. That caused the server to mistake the trailing envelope for a brand-new user turn, clearing `tool_rounds` mid-turn and leaking browser Page IDs or hypothetical git instructions.

## Original Source Evidence
Session `vsc_ff92c0c2-2e2c-4ca3-958f-1e249dcd2164` (`docs/clover-generative-compilation-research-context.md` Section 4.3) and pre-fix diff in commit `494f36f`.

## Revision History
- `2026-10-26T02:15:00Z` (**system_measurement**, `created`): Initial observation entry recorded from docs/clover-generative-compilation-research-context.md#section-4.3 (session vsc_ff92c0c2-2e2c-4ca3-958f-1e249dcd2164); commit:494f36f
