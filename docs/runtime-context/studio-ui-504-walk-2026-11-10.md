# Clover AI Studio v5.0.4 — UI walk, fixes and open questions (2026-11-10)

## Context
- Request: get the latest `main`, test the new unified UI, make sure the features work, fix what can be fixed,
  step by step, following Clover.
- Branch `fix/studio-ui-504`, worktree `C:\personal\oss\clover-ui-fix-wt`, from `main` = `origin/main` =
  `3facab25` (tag v5.0.4; checked again with `git fetch` before the commit).
- Read: `studio/clover/web/app.html`, `studio/clover/server.py` (`run_terminal_tool`, `_local_repo_dir`),
  `studio/clover/local_terminal.py`, `studio/clover/browser_host.py`, `studio/studio/build_assets.py`,
  `studio/tests/ui_harness.py`, `studio/tests/test_studio_unified_ui.py`, and the failing suites below.

## Direction
- Walk every feature in the browser against the real server routes before fixing anything.
- Fix only what is clearly broken. Do not edit existing tests. Ask before push or merge.

## Execution
- Inventory: all 61 endpoints the UI calls are handled by `server.py`.
- Harness: `tests/ui_harness.py` (real server routes, stand-in engine, scratch base) on port 8797, opened in the
  VS Code browser at 1024×768 and 390×844. The Terminal app ran real PowerShell on this PC. The Browser app read
  real web pages.
- Every feature was clicked and the console and failed requests were recorded (see Outcome).

## Outcome

### Fixed (commit `ced57d82`)
1. **The Terminal app lost table output.** The pwd button (`Get-Location`), any `Select-Object`, and
   `cd ..; Get-Location` showed "(completed with no output)". Windows PowerShell 5.1 printed nothing when the
   script ended with `exit`. Now the command's output is formatted inside the script, before the exit
   (`server.run_terminal_tool`). I compared 19 commands, old script against new (exit codes, folder, native
   stderr text, `&&`/`||`, `Write-Host`, `throw`). Only the four lost outputs changed.
2. **Opened folders in the Workspaces and Memory trees kept "Loading…"** above their files. A failed load showed
   "Loading…" forever. Now the box is emptied for every folder, and a failure appears inside the folder
   (`loadWorkspaceTree`, `loadMemoryTree`). Site and Android copies were synced with `build_web_site` only.

### Evidence
- New `tests/test_terminal_tree_fixes.py` has 14 checks. All 14 pass on the fix. On v5.0.4, 8 fail: every check of
  the two bugs. The 6 checks of unchanged behaviour pass on both versions.
- All 37 suites ran, each in its own process: 33 passed.
  - 3 failed before the fix too (see Open questions).
  - `test_browser_handback.py` failed once and then passed 6 times in a row. It is intermittent and unrelated to
    the fix. The cause is in `browser_host.py`: a page id is the time to the second plus a content hash, so two
    pages saved in the same second sort by hash, not by time.
- Browser re-check after restarting the harness, with no console errors:
  - pwd, `Select-Object`, `cd ..; Get-Location` and `Get-ChildItem -Name | Select-Object -First 3` print, and the
    folder moves;
  - `cmd /c exit 3` shows exit 3;
  - Workspaces `src` and Memory `config` show their files;
  - a folder deleted after listing shows "gone-dir is not a folder" inside it.

### Worked in the walk (stand-in engine, so these test the UI and server, not the model)
- **Navigation:** the 6 sections and every submenu; the Settings pages (model, account, storage, permissions,
  debug, advanced); the Learning views; the header Settings button; the sidebar toggle.
- **Mobile layout (390 px):** the drawer closes after choosing an app, and nothing overflows.
- **Chat:** greeting; permission card (Allow once / Don't allow), tool card, then reply; Clover's command logged in
  the Terminal app; search with browser links; table; code block; picture; Stop; Try again; Helpful; new chat;
  delete chat (removed on the server too); attach a file; the + senses menu opens and closes (no capture).
- **Tools:**
  - Terminal: commands, errors, cd, Send Output to Chat, Ask Clover in Terminal.
  - Browser: search, Wikipedia page.
  - Custom tool: add, open, remove.
  - Permission change persisted.
- **Workspaces:** add a folder, tree, file viewer. **Memory:** tree, AGENTS.md, personal summary.

### Not verified
- **Not run:**
  - camera, microphone and screen capture (they need consent on this PC);
  - GitHub connect and clone (they need a token);
  - the real model (LFM3/Omni) and speech;
  - the Android app and the public web build.
- **Visuals:** the VS Code browser tab is hidden, so CSS transitions do not advance and screenshots can be stale.
  I measured layouts with transitions turned off; I did not judge them from pictures.

## Open questions for the human
1. **Pre-existing failing tests** (I did not edit tests):
   - `test_local_tool_packs.py` expects 19 base tools. There are 20 since `manage_todo_list` was added (c54df640).
     Update it to 20?
   - `test_vscode_bridge.py` expects "run pip install requests in the terminal" to go to VS Code. Since a3cf049b
     it runs in the Studio Terminal. Which is intended?
   - `test_csi_child.py` fails only in a worktree: it compares against the main checkout's CRLF files. It passes in
     the main checkout.
2. **Terminal default folder.** With no workspace open, the Terminal app starts in Clover's memory repository
   (`_local_repo_dir`). Clover's own PowerShell tool starts in the home folder. Should the Terminal app use the
   home folder too?
3. **Custom tools** are stored only in this browser (`localStorage` `clover_custom_tools_v1`). The text says Clover
   can launch them, but the server never receives them. Change the text, or connect them to Clover?
4. **Noted, not changed:**
   - Chats kept in the browser stay listed after the server has none (merge-only design).
   - A stopped answer is not marked as stopped (same in v5.0.3).
   - Every finished turn logs `POST /api/chat/stream net::ERR_ABORTED` in the console; the reply is complete.
     Cause not investigated.
   - The file viewer's line count is one too many.
   - The Listen error appears only as a tooltip.
   - example.com fails TLS verification on this PC only.

## Growth
- Windows PowerShell 5.1 scripts run with `-Command` that end in `exit` can drop output that is formatted
  automatically. Format the output inside the script (`& { … } | Out-String -Stream`). Do not merge native stderr
  with `2>&1`: PowerShell 5.1 wraps each stderr line in an error record.
- A test that compares against a sibling checkout cannot pass from a worktree. Check where its reference comes from
  before calling it a regression.
- One failure in a full run is not a regression until it is re-run and its cause is found (the hand-back test above).
- When handing a human a test server, say in the same sentence what is fake in it. I gave the harness URL and
  noted "stand-in model" only under "Not tested"; the human then judged the real chat by its echo replies.

## Accounts and sign-in (2026-11-11)

### Context
- The human: the chat does not work properly, connecting GitHub is hard, Google is simple. Can one account (Google,
  or any single account) do the memory repo sync, Workspaces and collective learning?
- Their page was the harness on port 8797. Its stand-in model echoes the message ("Hello from the harness. You
  said: …"). `POST /api/vscode/do` returned 409 because no VS Code is connected to the harness. Harness stopped.

### Findings (each checked on 2026-11-11)
- GitHub repositories accept only GitHub credentials; a Google token cannot sync them. GitHub's own sign-in page
  offers "Continue with Google" (github.com/login), so a GitHub sign-in can use a Google login on GitHub's side.
  Whether that links to an existing GitHub account is GitHub's rule; not tested.
- **"Sign in with Google" (Learning → Collective) does not sign in with Google.**
  - Google rejects the built-in client ID `google_identity.DEFAULT_GOOGLE_CLIENT_ID` with "The OAuth client was not
    found. Error 401: invalid_client".
  - Without `CLOVER_GOOGLE_CLIENT_SECRET`, `/api/identity/google/start` calls `connect_device_oidc()`. That issues a
    token claiming `iss=https://accounts.google.com`, signed with a key that `_get_device_oidc_keypair()` derives from
    fixed seeds in the public source (added in 0dade942).
  - `fetch_google_jwks()` always adds that key to Google's keys. The collective gateway
    (`authenticate_contributor`, `jwks=None`) therefore accepts tokens anyone can make for any `sub`, which means
    unlimited pseudonymous contributors and no per-contributor rate limit.
  - The same code is in k3's running web studio (`grep` only; no request was sent to the service).
- `/api/admin/github/device/*` uses `server.GITHUB_DEVICE_CLIENT_ID` = the GitHub CLI's OAuth app (added in
  f7f6396c; the ID matches `oauthClientID` in cli/cli `internal/authflow/flow.go`). GitHub's approval page would say
  "GitHub CLI".

### Proposed (waiting for the human's direction)
- One GitHub account for everything: memory repo sync, Workspaces, the collective identity and k3 access (k3 access
  already accepts GitHub).
  - Sign-in uses Clover's own GitHub OAuth App with the device flow: the Studio shows a code and GitHub opens, where
    "Continue with Google" works. No token pasting, no VS Code, and no client secret in the app.
- Stop trusting the computed key whatever account is chosen. Deploying that to k3 needs separate approval.

### Direction (human, 2026-11-11)
- One account: GitHub alone. The Google sign-in goes, including the self-made token and the trust in its key.
- OAuth app registration, from GitHub's "Authorizing OAuth apps" page (read 2026-11-11):
  - The device flow needs only the client ID; "the client_secret is not needed for the device flow".
  - Expiring tokens: the access token lasts 8 hours and the refresh token lasts 6 months without use. Refreshing
    needs `client_secret` "unless the token was generated using the device flow", so Clover can keep "Expire user
    access tokens" on without a secret. A used refresh token and its old access token stop working, so a refresh
    must save the new pair before anything else uses it.
  - The redirect URI is not used by the device flow. Wildcard matching sends codes to any subdomain or path; leave
    it off.
  - Users can review or revoke the app at `https://github.com/settings/connections/applications/<client_id>`.
- Next: build the sign-in once the human sends the Client ID. The client secret is not needed and must not be
  generated or shared.

### Website sign-in (human question, 2026-11-11)
- The human asked how the website would work with `http://127.0.0.1/callback`, and why the callback is not on
  Clover's own domain. The device flow uses no callback on any platform, so that URI was only a filler for the form.
  The admin console on the website already signs in with the device flow (`site/admin/index.html`).
- Agreed with the human: the website gets the normal redirect sign-in, with no code to type.
  - Callback: `https://api.cloverframework.com/api/auth/github/callback`. On 2026-11-11 that host resolved to
    Cloudflare and was answered by Clover's own server: `/api/health` returned its JSON "Sign in with an approved
    account" 401. `cloverframework.com` is the static GitHub Pages site (`site/CNAME`), so it cannot finish a
    sign-in.
  - GitHub requires `client_secret` to exchange the code in the redirect flow. It lives only on k3 and is put
    there by the human, never in chat, the page, the apps or the repo. This corrects "no secret" above for the
    website only.
- The laptop and phone apps keep the device flow. Their server is on the user's own machine; a callback on
  api.cloverframework.com would put the user's GitHub token on k3 instead of the device.

### Execution (2026-11-11, commit 88ce731f)
- **Client ID check:** the human sent the Client ID `Ov23li2tK8VKBD0fTYGG`. Asking `github.com/login/device/code`
  with it returned 200 with a device code, a `XXXX-XXXX` user code and `https://github.com/login/device`. So the app
  exists and has the device flow on.
- **Apps:** `github_auth.py` handles the device flow and renewal. `workspaces.GitHubAuth` keeps the sign-in
  encrypted (who signed in, token pair, expiry) and renews it when it is used.
  - Routes: `/api/github/device/start|poll`; sign out is `/api/github {forget}`.
  - Memory sync, Workspaces and k3 unlock read `S.github.token`, so they use the sign-in unchanged.
- **Website:** `/api/auth/github`, `/api/auth/github/start`, `/api/auth/github/callback` and
  `/api/auth/github/finish`, plus `/api/auth/github/device/start|poll` for when the server has no secret.
  - The return address must be a Clover origin with no control characters.
  - The callback runs `verify_k3_credential` once, then drops the token.
- **Collective learning:** `POST /api/collective/gateway/session` checks a GitHub token once and returns the
  gateway's `cg1.` HMAC credential. The Studio keeps it in the vault, tied to the login and the gateway.
  `core_sync` is unchanged in shape: `identity_connected` was added and `google_connected` still works.
- **Security:**
  - `google_identity.py` is restored from the mobile copy (no self-made key).
  - The Google routes, the UI and `google_identity_manager` are gone.
  - The admin flow uses `github_auth.CLIENT_ID`.
- **Bugs found while testing:**
  - `server.py` had no `import subprocess`, so memory sync with a token raised NameError.
  - `CollectiveGateway._init_secret` read the secret with `.strip()`; 10 of 300 restarts in a probe replaced it,
    0 of 300 after the fix.

### Outcome
- New `tests/test_github_signin.py`: 64 checks, offline against a stand-in GitHub. It covers the device flow,
  renewal, the encrypted store, the redirect and one-time codes, the gateway credential, and the routes through the
  real server, including the website mode.
- All Studio suites: 35 of 38 passed; the 3 that fail are the same known ones.
- The Studio copies of `test_google_identity_collective.py` (32 checks) and `test_collective.py` (35 checks) pass.
  The mobile copies fail with "No module named 'vault'", the same as before the change.
- Browser check on the test server (stand-in model, real GitHub for the sign-in):
  - the new buttons appear on all four pages;
  - "Sign in with GitHub" got a real code from GitHub and showed it;
  - the test server recorded the open of github.com/login/device;
  - the page polled about every 5.5 s.
  - One bug fixed: the clipboard write was rejected without the page focused.
- **Not verified:**
  - a real approval with the human's account, and memory setup against real GitHub;
  - renewal against real GitHub;
  - the website redirect with the real secret on k3 (not deployed);
  - the Android app;
  - that the registered redirect URI is exactly the callback above.
- **Found, not changed:** in website mode, `POST /api/github` lets any approved account replace or remove the
  server's own GitHub token.

### Growth
- A test clock far from real time made a token look expired, and the store tried a real renewal at github.com.
  Give stores a stub renewer in tests, and compare expiry with the real clock in route tests.
- The test server's request log is block-buffered when piped through Tee-Object. Read the page's resource timings
  to see what it really requested.
