# Context: DeepSeek Harness (`deepseek-ai/deepseek-harness`) — Operational Super-Agent Layer Analysis

- **Date:** 2026-11-09
- **Branch / Worktree:** `research/superagent-jev-dory` (`C:\personal\oss\clover-superagent-jev-dory-wt`)
- **Primary Source Cloned & Executed:** `https://github.com/deepseek-ai/deepseek-harness.git` (`v0.2.2-rc.1`, commit `fcc0ad46`, 394 workspace packages, 20,665 files)
- **Executable Probe & Telemetry Artifact:**
  - Probe script: [research/superagent-jev-dory/01-deepseek-superagent/clover-dsh-probe.spec.ts](../../../research/superagent-jev-dory/01-deepseek-superagent/clover-dsh-probe.spec.ts)
  - Live JSON output: [research/superagent-jev-dory/01-deepseek-superagent/dsh_live_probe_results.json](../../../research/superagent-jev-dory/01-deepseek-superagent/dsh_live_probe_results.json)

---

## 1. Intended Outcome & Direction

Establish ground-truth understanding of **DeepSeek Super-agent / DeepSeek Harness (`dsh`)** by cloning the repository, installing its 394 workspace packages, running its native unit test suites, and executing our own live Cordis probe across its core operational subsystems—rather than relying on secondary blog summaries or architectural diagrams. Determine what Clover should adopt, what Clover already owns, and where `deepseek-harness` has structural limitations that Clover's operating principles must guard against.

---

## 2. What Was Known vs. What Cloning & Running the Code Revealed

### 2.1 Secondary Summary Claims (Attachment #1) vs. Actual Source Reality (`fcc0ad46`)

Attachment #1 referenced a third-party Japanese article diagram (*"DeepSeek Harness 完全ガイド"* by Masaaki Ito) and attributed six named concepts to DeepSeek's Super-agent architecture. Searching all 20,665 files in `deepseek-ai/deepseek-harness` and running its live services revealed crucial differences between the secondary summary and the real codebase:

| Concept in Secondary Summary | Actual Implementation in `deepseek-ai/deepseek-harness` (`fcc0ad46`) | Source Location | Reality Check |
|---|---|---|---|
| **Persistent Router** | **No class or package named `PersistentRouter` or `SuperAgent` exists** (`0` matches across 20,665 files). The root runtime is **Cordis** (`@deepseek-ai/cordis` in `vendor/cordis`), an effect-scoped plugin dependency injection container where `AgentLoop` (`ctx.agentLoop`) drives `ReactLoopAgent` instances registered in `AgentRegistry` (`ctx.agents`). | `packages/core/agent-loop/src/index.ts`, `packages/core/agent/src/index.ts` | Routing is event-driven via Cordis waterfalls (`agent/pre-step`, `agent/request`, `llm/stream`, `tools/pre-execute`, `tools/execute`, `tools/post-execute`) and serial hooks (`agent/turn-stopping`). |
| **Task Board Lifecycle: `Created → Routed → Running → Waiting → Completed / Failed`** | **Does not exist in `deepseek-harness`.** The actual shared task board (`@deepseek-ai/dsh-agent-team`) defines `type TeamTaskStatus = 'pending' | 'in_progress' | 'completed' | 'deleted'` with compare-and-set `revision` integers and actions `'claim' | 'release' | 'edit' | 'set_dependencies' | 'complete' | 'reopen' | 'reassign' | 'delete'`. | `packages/team/agent-team/src/types.ts`, `packages/team/agent-team/src/task-board.ts` | There are **no `Routed`, `Waiting`, or `Failed` task states** on the task board. Furthermore, same-session goals (`@deepseek-ai/dsh-goal`) use a separate 4-state enum: `'active' | 'paused' | 'blocked' | 'complete'`. |
| **Task Board Event Triggers / Auto-Wake** | **Explicitly passive by design.** Task mutations (`createTask`, `updateTask`) append `'team/task'` events to the Team Lead's session log and never notify or wake idle or waiting workers. | `packages/team/agent-team/src/task-board.ts`, `docs/subsystems/agent-team.md` | Verified in our live probe: completing `task-1` flips `task-2.ready` from `false` to `true`, but `worker-a`'s `inbox.nextStep.length` remains `0` (`taskBoardAutoWakesWorkers: false`). An agent only wakes when explicitly messaged via `ctx.subagents.sendMessage(...)`. |
| **Worker Isolation & `writeScopes`** | Subagents run as local child sessions (`spawn` fresh context or `fork` seeded prefix) or external processes (`acp`, `claude-code`, `codex`, `dsh-sdk`, `subagent-oar`). Task board `writeScopes` are **advisory path prefixes only**, never runtime locks. | `packages/team/agent-team/src/task-view.ts`, `packages/subagent/subagent/src/manager.ts` | Verified in live probe: claiming `task-3` on `src/runtime/session/index.ts` while `task-1` owns `src/runtime/session` succeeds without blocking (`writeScopeBlockedExecution: false`) and only attaches `writeScopeWarnings: ["write scopes overlap with task-1"]`. |
| **Chain of Recall** | The literal phrase `"Chain of Recall"` appears in `0` files, but the mechanism corresponds to **`SessionReferenceSource`** (`kind: 'session-reference'`, **`form: 'recall'`**) in `@deepseek-ai/dsh-session-reference`, combined with surface-shadowing compaction (`@deepseek-ai/dsh-compaction`) and `@deepseek-ai/dsh-session-query`. | `packages/context/session-reference/src/index.ts`, `packages/compaction/compaction/src/types.ts` | Cross-session recall snapshots a target session up to `capturedThroughSeq` (v4 format) and injects a single untrusted `user/message` with `form: 'recall'`, while rejecting self-references with `SESSION_REFERENCE_SELF_REFERENCE`. |
| **Watchdog** | Implemented as three distinct mechanisms: (1) `idleWatchdog` in `@deepseek-ai/dsh-timeout` (rearmable async-iterator stream deadline), (2) `@deepseek-ai/dsh-repeat-tool-reminder` (canonical JSON tool-loop detector), and (3) `@deepseek-ai/dsh-goal` continuation/blocking. | `packages/util/timeout/src/index.ts`, `packages/guard/repeat-tool-reminder/src/index.ts` | Verified in live probe: `idleWatchdog` ignores consumer think-time (`80ms > 60ms` deadline) when no `.next()` is outstanding, and aborts hung provider streams with `TimeoutReason("STREAM_IDLE_TIMEOUT", 60)`. |

---

## 3. What Was Tried & Executed

1. **Workspace Installation (`pnpm@11.28.5`):**
   - Installed all 394 workspace projects (1,434 packages) in `C:\personal\oss\external-clones\deepseek-harness`.
2. **Existing Unit Test Suite Execution (`vitest v4.1.11`):**
   - Executed 97 test suites (2,301 tests) covering `packages/core/session`, `packages/core/agent-loop`, `packages/subagent/*`, `packages/team/agent-team`, `packages/context/session-reference`, `packages/compaction/compaction-basic`, `packages/goal/goal`, `packages/guard/repeat-tool-reminder`, and `packages/util/timeout`.
   - Under 97-file parallel CPU load, 2,296/2,299 executed tests passed and 3 timed out at the 5,000 ms default timeout; re-running the 2 affected suites (`team.spec.ts` and `subagent-acp.spec.ts`, 107 tests) with `--testTimeout=20000` passed **107/107 (100%)**, bringing total verified unit test passes to **2,299 / 2,299 (100%)**.
3. **Hands-On Live Cordis Architecture Probe (`clover-dsh-probe.spec.ts`):**
   - Wrote and executed a comprehensive integration test booting a real Cordis `Context` with `JsonlSessionPersistence`, `AgentLoop`, `SubagentService` (`spawn` + `fork`), `TeamService`, `GoalService`, `RepeatToolReminder`, `SessionReferenceResolver`, and `idleWatchdog`.

---

## 4. What the Running System Showed (Including Mistakes Caught Only by Running Code)

Working directly with the live `deepseek-harness` runtime exposed **four concrete technical findings** that reading documentation alone did not show:

1. **No Defensive Optionality at Internal Typed Boundaries (`sendMessage` & `defineTool`):**
   - On our first probe run, calling `ctx1.subagents.sendMessage(lead, resolvedWorkerAId, content)` without the 4th `options` argument threw `TypeError: Cannot read properties of undefined (reading 'signal')` at `packages/subagent/subagent/src/manager.ts:562`.
   - On our second probe run, registering a tool with `defineTool({ name, description, parameters, execute })` without `output: { schema, render }` threw `TypeError: Cannot read properties of undefined (reading 'render')` at `packages/core/tools/src/schema.ts:566`.
   - **Why:** `deepseek-harness` enforces a strict architectural rule (*"Trust TypeScript at typed same-process boundaries"* and *"Design each tool's UI presentation up front"*): every tool must declare a lossless JSON output schema and a pure `render` presenter at registration time.
2. **Silent Driver Containment of Unsourced `UserMessage` Inputs (`kick()`):**
   - On our third probe run, calling `guardedAgent.followup(createUserMessage({ content: [...] }))` without `source: { kind: 'user' }` resulted in `probeToolCalls === 0` while `await guardedAgent.whenIdle()` resolved cleanly with no thrown exception!
   - **Why:** `ReactLoopAgent.kick()` (`packages/core/agent-loop/src/agent.ts:259-263`) wraps `while (await this.turn()) {}` in a `try / catch (_error)` block that emits `agent/error` and swallows the exception so the driver loop settles to `idle`. Adding an `agent/error` listener and passing `source: { kind: 'user' }` immediately executed all 3 tool calls and fired both `repeat-tool-reminder` notices (`probe_read × 2`, `probe_read × 3`).
   - **Lesson for Clover:** A driver that catches turn errors and resolves `whenIdle()` without rejecting caller promises can mask broken turns unless `agent/error` is explicitly subscribed to and surfaced in Outcome verification.
3. **Unplanned Discovery — False-Positive `writeScopeWarnings` on Completed Tasks in `TeamTaskBoard`:**
   - When `task-1` (`writeScopes: ["src/runtime/session"]`) was completed (`status: "completed"`, `revision: 3`) and `task-3` (`writeScopes: ["src/runtime/session/index.ts"]`) remained `in_progress`, inspecting `recoveredTasks` after cold restart revealed that **completed `task-1` carried `"writeScopeWarnings": ["write scopes overlap with task-3"]`**!
   - **Root cause in `deepseek-harness` (`packages/team/agent-team/src/task-view.ts:31-36`):**
     ```ts
     for (const other of state.tasks) {
       if (other.id === task.id || other.status !== 'in_progress') continue
       if (task.writeScopes.some(left => other.writeScopes.some(right => scopesOverlap(left, right)))) {
         warnings.add(`write scopes overlap with ${other.id}`)
       }
     }
     ```
     `projectTaskView` checks `other.status !== 'in_progress'`, but forgets to check `if (task.status === 'completed')`! Thus, every historical completed task in a directory continues to report active write-scope conflict warnings whenever any later task edits a file inside that directory.
4. **Worker Self-Certification in `dsh-tool-ralph` vs. Clover's Independent Verification Boundary:**
   - Inspecting `packages/experimental/tool-ralph/src/index.ts` (`RALPH_SCRIPT`) and its README confirmed that `ralph` accepts `status: 'complete'` whenever the child worker returns a JSON object with a non-empty `evidence: string[]` array (`validateReport` checks only string normalization, never whether any test or check actually ran).
   - Under Clover `AGENTS.md` §4 ("Prefer evidence that the agent cannot silently redefine while performing the Execution"), worker self-declaration is **not** an Outcome verification gate.

---

## 5. Architectural Mapping: What Clover Should Adopt vs. What Clover Must Improve

| Capability | `deepseek-harness` Mechanism | Clover `Clover.SuperAgent` Integration Design |
|---|---|---|
| **Append-Only Event-Sourced Session & Task Log** | `SessionEventMap` + `session.v4.jsonl.zstd` + `TeamJournal` CAS `revision` | Adopt append-only JSONL event log with monotonic `seq`, `revision` compare-and-set, and deterministic state projection across process restarts. |
| **Explicit Task Lifecycle States** | Only `pending \| in_progress \| completed \| deleted` (passive board) | Upgrade in `Clover.SuperAgent` to explicit **`created → routed → running → waiting → verifying → completed / failed`**, where `completed` requires an independent verification check rather than worker self-declaration, and fix the `writeScopeWarnings` bug by only checking active (`routed \| running \| waiting`) tasks. |
| **Event Subscriptions & Worker Wakeup** | Passive task board; requires manual `send_message` | Support both explicit peer messaging (`agent-message` with `form: 'relay'`) **and** deterministic dependency-ready event triggers (`on_blocker_completed`) so unblocked tasks can automatically transition `created -> routed`. |
| **Transcript Attribution Integrity** | Separate `AgentMessageSource` (`form: 'relay'`) vs `SubagentSettledMessageSource` (`form: 'notice'`) | Adopt directly: never merge runtime settlement notices with worker-authored messages. |
| **Chain of Recall (`form: 'recall'`)** | `SessionReferenceResolver` (`capturedThroughSeq`, `SESSION_REFERENCE_SELF_REFERENCE`) + surface-replacement compaction | Integrate with Clover's Memory Graph (`memory_index.py`): cross-session recall records exact `sessionId` and `capturedThroughSeq` provenance and forbids self-referential loops. |
| **Watchdog & Loop Guards** | `idleWatchdog` (consumer-think-time immune) + `RepeatToolReminder` (deep-key-sorted JSON canonicalization, gentle-then-detailed escalation) | Adopt both into `Clover.SuperAgent`: canonical JSON argument loop detection (`thresholds = [3, 5]`) + stream/worker idle watchdog that distinguishes consumer wait time from stalled worker execution. |

---

## 6. What Remains Unknown / Next Cycle

- **What was ruled out:** Directly forking `deepseek-harness` or treating its task board as an active router (`dsh` task boards are passive projections and lack a `verifying` or `failed` task state).
- **What remains unknown:** How local single-forward-pass decision primitives (`OpenJev`) perform when choosing between `act`, `gather`, `clarify`, `verify`, and `delegate` at the `agent/pre-step` boundary before dispatching to `Clover.SuperAgent` workers.
- **Next cycle:** Proceed to **Project 2: `GPT-AGI/OpenJev` and `Braedennn/OpenJev`**—inspect, set up, run live model forward passes on `Choice`, `Score`, and `Noul`, measure raw vs. temperature-calibrated probabilities andECE, and write `02-openjev-decision-engine-context.md`.
