# Clover AI Studio — Privacy & Data Boundary Specification

## 1. Local-First Guarantee

By default, Clover AI Studio runs entirely on your device (Windows laptop or Android phone):

- **No Sign-In Required**: Clover starts, answers questions, runs local tools, and learns your personal habits without Google sign-in or any external account.
- **Personal Learning Stays Local**: Personal Process Learning (`<base>/personal`) is enabled by default and never leaves your device. You can inspect, pause, or permanently delete all learned habits and experiences at any time in **Settings → Learning & Growth → View personal learning**.
- **Collective Learning Is Off by Default**: Both **Contribute to Collective Learning** and **Receive Collective Learning** default to `false` and cannot be enabled without Google sign-in and explicit user consent.

---

## 2. What Data Never Leaves the Device

Clover AI Studio **never** transmits any of the following to the Collective Gateway or Registry:

1. Raw conversation messages, prompts, or assistant replies.
2. Personal memories (`<base>/clover-ai-studio/growth/knowledge`) or conversation histories (`<base>/clover-ai-studio/sessions`).
3. Attached files, spreadsheets, PDFs, photos, audio recordings, or videos.
4. Personal answer-format preferences (`short`, `bullets`, `table`, `prose`, etc.), personal routines (`"the usual"`), or single-turn user statements (`"When I say X, I mean Y"`).
5. File paths, URLs, email addresses, numbers, organization names, or out-of-vocabulary words (filtered out before candidate synthesis by `collective.is_allowed_collective_feature` against Open English WordNet 2025).
6. Google passwords, OAuth `access_token`s, `refresh_token`s, email addresses, or profile names (Clover requests only the `openid` scope, discards access/refresh tokens immediately, and uses only the opaque Google `sub` identifier).
7. Local installation IDs or vault secrets.

---

## 3. Exactly What Leaves the Device When You Opt In

### 3.1 When "Receive Collective Learning" Is Enabled
When you click **Sync collective core now**:
- **Sent**:
  - `Authorization: Bearer <google_id_token>` (verified by the gateway to confirm authenticated access; never logged or stored in the registry).
  - `since_version` (e.g. `1.0.1`), so the registry only sends new or updated rule files.
- **Received**:
  - `releases/manifest.json` (signed with Ed25519) and verified general rule JSON files (`core/<category>/<rule_id>.json`).

### 3.2 When "Contribute to Collective Learning" Is Enabled & Confirmed
Nothing is uploaded automatically in the background. When a candidate strategy is ready, you must click **Review & Contribute…**, inspect the exact JSON payload on screen, and click **Confirm & Upload**.
- **Sent**:
  - `Authorization: Bearer <google_id_token>` (converted by the gateway into an irreversible pseudonymous identifier `contrib_<hmac>` using a server-only secret).
  - Request replay guard (`nonce`, `timestamp`) and explicit confirmation flags (`confirmed: true`, `consent_contribute: true`).
  - The approved candidate package containing **only**:
    ```json
    {
      "package_id": "cand_...",
      "schema_version": "1.0.0",
      "installation_id": "overwritten_by_gateway_with_contrib_hmac",
      "category": "problem-solving",
      "trigger": ["ctx=phone", "tok=corroborate"],
      "target": ["python", "web.page", "web.search"],
      "strategy": "Collective problem-solving strategy: first gather primary sources from the web, then verify the figures with Python.",
      "local_support_count": 2,
      "local_contra_count": 0,
      "local_precision": 1.0,
      "cross_context_generalised": false
    }
    ```

---

## 4. Withdrawal & Disconnect Controls

In **Settings → Learning & Growth**, you can at any time:
- **Disable Contribute / Receive**: Immediately stops future synchronization in that direction (disabling Receive also unloads active shared-core rules).
- **Withdraw Contribution Consent**: Sends an authenticated withdrawal request (`POST /api/collective/gateway/withdraw`) that purges your pseudonymous contributor's candidate packages from the gateway, re-evaluates quorum across remaining contributors, demotes any rule that drops below quorum in a new signed release, and disables local contribution consent.
- **Disconnect Google**: Deletes the encrypted Google OIDC session from the device vault and turns off both collective switches while keeping all local personal learning intact.
