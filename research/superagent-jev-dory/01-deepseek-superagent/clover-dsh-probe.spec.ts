import { describe, expect, it } from 'vitest'
import { mkdtempSync, rmSync, writeFileSync, readdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import { SessionId } from '@deepseek-ai/dsh-session'
import * as JsonlSessionPersistence from '@deepseek-ai/dsh-session-persistence-jsonl'
import SubagentService from '@deepseek-ai/dsh-subagent'
import * as SubagentFork from '@deepseek-ai/dsh-subagent-fork-in-process'
import * as SubagentSpawn from '@deepseek-ai/dsh-subagent-spawn-in-process'
import TeamService from '@deepseek-ai/dsh-agent-team'
import * as RepeatToolReminder from '@deepseek-ai/dsh-repeat-tool-reminder'
import GoalService from '@deepseek-ai/dsh-goal'
import SessionReferenceResolver from '@deepseek-ai/dsh-session-reference'
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { MockAdapter, textResponse, toolCallResponse } from '../../../core/agent-loop/tests/mock-adapter.ts'
import { mountWorkingDirectoryFixture } from '../../../subagent/subagent/tests/working-directory-fixture.ts'
import { TestSessionQuery } from './test-session-query.ts'

const OUTPUT_PATH = process.env['CLOVER_DSH_PROBE_OUT']
  ?? 'C:/personal/oss/clover-superagent-jev-dory-wt/research/superagent-jev-dory/01-deepseek-superagent/dsh_live_probe_results.json'

describe('Clover DeepSeek Harness Live Architecture Probe', () => {
  it('executes and measures all 6 operational mechanisms and boundaries against live dsh packages', async () => {
    const storageRoot = mkdtempSync(join(tmpdir(), 'clover-dsh-probe-'))
    const report: Record<string, unknown> = {
      repository: 'deepseek-ai/deepseek-harness',
      version: '0.2.2-rc.1',
      commit: 'fcc0ad46',
      timestamp: new Date().toISOString(),
      experiments: {},
    }

    try {
      // =========================================================================
      // Experiment 1: Cordis Plugin Tree, Shared Task DAG, CAS Revision, WriteScopes, & Crash Persistence
      // =========================================================================
      const ctx1 = new Context()
      await mountAgentLoopTestDependencies(ctx1)
      await mountWorkingDirectoryFixture(ctx1)
      await ctx1.plugin(JsonlSessionPersistence, { root: storageRoot })
      await ctx1.plugin(TestSessionQuery)
      await ctx1.plugin(AgentLoop, { agents: [] })
      await ctx1.plugin(SubagentService)
      await ctx1.plugin(SubagentSpawn, { providerName: 'spawn' })
      await ctx1.plugin(SubagentFork, { providerName: 'fork' })
      await ctx1.plugin(TeamService, { defaultEnabled: false, maxTasks: 64 })
      await ctx1.plugin(GoalService)
      await ctx1.plugin(RepeatToolReminder, { thresholds: [2, 3], argumentsPreviewChars: 120 })
      await ctx1.plugin(SessionReferenceResolver, { candidateLimit: 10 })

      const adapter1 = new MockAdapter([
        textResponse('Worker A initial analysis complete: found 3 modules.'),
        textResponse('Worker B forked analysis complete.'),
        textResponse('Worker A follow-up after relay message.'),
      ])
      ctx1.llm.registerAdapter(['mock'], adapter1)

      const lead = await ctx1.agentLoop.create(SessionId('clover-lead'), { provider: 'mock', model: 'mock' })
      await ctx1.agentTeams.setEnabled(lead, true)

      const t1 = await ctx1.agentTeams.createTask(lead, {
        subject: 'Inspect runtime context',
        description: 'Read session log and verify event schema',
        writeScopes: ['src/runtime/session'],
      })
      const t2 = await ctx1.agentTeams.createTask(lead, {
        subject: 'Implement adapter',
        description: 'Build adapter after context inspection',
        blockedBy: [t1.id],
        writeScopes: ['src/runtime/session/adapter.ts'],
      })

      // Test cycle detection
      let cycleRejectedCode: string | null = null
      try {
        await ctx1.agentTeams.updateTask(lead, {
          taskId: t1.id,
          expectedRevision: t1.revision,
          action: 'set_dependencies',
          blockedBy: [t2.id],
        })
      } catch (err: unknown) {
        cycleRejectedCode = (err as { code?: string }).code ?? String(err)
      }

      // Test claiming blocked task t2 before t1 is completed
      let blockedClaimCode: string | null = null
      try {
        await ctx1.agentTeams.updateTask(lead, {
          taskId: t2.id,
          expectedRevision: t2.revision,
          action: 'claim',
        })
      } catch (err: unknown) {
        blockedClaimCode = (err as { code?: string }).code ?? String(err)
      }

      // Claim t1 (revision 1 -> 2)
      const t1Claimed = await ctx1.agentTeams.updateTask(lead, {
        taskId: t1.id,
        expectedRevision: t1.revision,
        action: 'claim',
      })

      // Create t3 with overlapping writeScope while t1 is in_progress
      const t3 = await ctx1.agentTeams.createTask(lead, {
        subject: 'Parallel edit in overlapping scope',
        description: 'Touches src/runtime/session/index.ts while task-1 owns src/runtime/session',
        writeScopes: ['src/runtime/session/index.ts'],
      })
      const t3Claimed = await ctx1.agentTeams.updateTask(lead, {
        taskId: t3.id,
        expectedRevision: t3.revision,
        action: 'claim',
      })

      // Test stale revision CAS rejection on t1
      let staleRevisionCode: string | null = null
      try {
        await ctx1.agentTeams.updateTask(lead, {
          taskId: t1.id,
          expectedRevision: 1, // stale! current is 2
          action: 'complete',
        })
      } catch (err: unknown) {
        staleRevisionCode = (err as { code?: string }).code ?? String(err)
      }

      // Spawn a real continuable child worker ("worker-a") and a forked worker ("worker-b")
      const actA = await ctx1.subagents.startActivation({
        provider: 'spawn',
        name: 'worker-a',
        label: 'Research worker A',
        prompt: [{ type: 'text', text: 'Analyze workspace modules' }],
        parent: lead,
        signal: new AbortController().signal,
        delivery: 'caller',
      })
      const resA = await actA.result

      const actB = await ctx1.subagents.startActivation({
        provider: 'fork',
        name: 'worker-b',
        label: 'Forked verifier B',
        prompt: [{ type: 'text', text: 'Verify parent assumptions' }],
        parent: lead,
        signal: new AbortController().signal,
        delivery: 'caller',
      })
      const resB = await actB.result

      // Complete t1 (revision 2 -> 3) and observe whether t2 becomes ready AND whether worker-a is woken automatically
      const workerABeforeComplete = ctx1.agents.get(actA.childId)
      const inboxBeforeComplete = workerABeforeComplete?.inbox.nextStep.length ?? 0
      const t1Completed = await ctx1.agentTeams.updateTask(lead, {
        taskId: t1.id,
        expectedRevision: t1Claimed.revision,
        action: 'complete',
      })
      const t2AfterT1Complete = ctx1.agentTeams.getTask(lead, t2.id)
      const workerAAfterComplete = ctx1.agents.get(actA.childId)
      const inboxAfterTaskComplete = workerAAfterComplete?.inbox.nextStep.length ?? 0

      // Now explicitly message worker-a using sendMessage (proving explicit messaging wakes/delivers)
      const opSignal = new AbortController().signal
      const resolvedWorkerAId = await ctx1.subagents.resolveName(lead, 'worker-a', opSignal)
      const relayMsgId = await ctx1.subagents.sendMessage(
        lead,
        resolvedWorkerAId,
        [{ type: 'text', text: 'Task-1 is complete; please proceed with Task-2.' }],
        { signal: opSignal },
      )
      // Wait for worker-a to process the relayed message turn
      const workerALive = ctx1.agents.get(actA.childId)!
      await workerALive.whenIdle()
      const workerAEvents = workerALive.session.snapshotEvents()
      const relayEvent = workerAEvents.find(
        e => e.type === 'user/message' && e.data.source.kind === 'agent-message',
      )

      // =========================================================================
      // Experiment 2: Chain of Recall (SessionReferenceResolver cross-session snapshot)
      // =========================================================================
      const candidates = await ctx1.sessionReferenceResolver.listCandidates(lead, '')
      let selfRefCode: string | null = null
      try {
        await ctx1.sessionReferenceResolver.prepare(
          lead,
          [{ type: 'text', text: 'Self ref test' }],
          [{ sessionId: lead.id, label: 'self' }],
          { provider: 'mock', model: 'mock' },
        )
      } catch (err: unknown) {
        selfRefCode = (err as { code?: string }).code ?? String(err)
      }

      const preparedRecall = await ctx1.sessionReferenceResolver.prepare(
        lead,
        [{ type: 'text', text: 'Synthesize findings from worker-a' }],
        [{ sessionId: actA.childId, label: 'worker-a' }],
        { provider: 'mock', model: 'mock' },
      )

      // =========================================================================
      // Experiment 3: Same-session Goal lifecycle & RepeatToolReminder loop guard
      // =========================================================================
      const goalCreated = ctx1.goals.create(lead, {
        objective: 'Complete 3-project integration analysis with verified evidence',
        maxGoalRounds: 5,
      })
      const goalPaused = ctx1.goals.pause(lead, { id: goalCreated.id, revision: goalCreated.revision })
      const goalResumed = ctx1.goals.resume(lead, { id: goalPaused.id, revision: goalPaused.revision })
      const goalBlocked = ctx1.goals.block(
        lead,
        { id: goalResumed.id, revision: goalResumed.revision },
        { code: 'missing-calibration-data', message: 'Waiting for empirical calibration dataset' },
      )

      // Test RepeatToolReminder on a tool called repeatedly with reordered JSON keys
      let probeToolCalls = 0
      ctx1.tools.register(defineTool({
        name: 'probe_read',
        description: 'Read a test resource',
        parameters: {
          path: { type: 'string', required: true },
          mode: { type: 'string', required: true },
        },
        output: {
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              result: { type: 'string', required: true },
            },
          },
          render: (_args, value) => [{ type: 'text', text: value.result }],
        },
        async execute(args) {
          probeToolCalls += 1
          return { result: `read:${args.path}:${args.mode}#${probeToolCalls}` }
        },
      }))

      const guardAdapter = new MockAdapter([
        toolCallResponse('c1', 'probe_read', { path: 'a.ts', mode: 'fast' }),
        toolCallResponse('c2', 'probe_read', { mode: 'fast', path: 'a.ts' }),
        toolCallResponse('c3', 'probe_read', { path: 'a.ts', mode: 'fast' }),
        textResponse('Stopped repeating after reminder.'),
      ])
      ctx1.llm.registerAdapter(['mock-guard'], guardAdapter)
      const agentErrors: string[] = []
      ctx1.on('agent/error', ({ agent, error }) => {
        agentErrors.push(`${agent.id}: ${error instanceof Error ? error.stack ?? error.message : String(error)}`)
      })
      const guardedAgent = await ctx1.agentLoop.create(SessionId('guarded-agent'), {
        provider: 'mock-guard',
        model: 'mock-guard',
      })
      guardedAgent.followup(createUserMessage({ content: [{ type: 'text', text: 'Run probe_read' }], source: { kind: 'user' } }))
      await guardedAgent.whenIdle()
      const guardEvents = guardedAgent.session.snapshotEvents()
      const reminderEvents = guardEvents.filter(
        e => e.type === 'user/message' && e.data.source.kind === 'repeat-tool-reminder',
      )

      // Flush and dispose ctx1 completely to test process-restart durability
      await ctx1.sessions.flush(lead.session)
      await actA.dispose()
      await actB.dispose()
      await ctx1.fiber.dispose()

      // =========================================================================
      // Experiment 4: Cold Process Restart & Durable Session / Task DAG Recovery
      // =========================================================================
      const ctx2 = new Context()
      await mountAgentLoopTestDependencies(ctx2)
      await mountWorkingDirectoryFixture(ctx2)
      await ctx2.plugin(JsonlSessionPersistence, { root: storageRoot })
      await ctx2.plugin(TestSessionQuery)
      await ctx2.plugin(AgentLoop, { agents: [] })
      await ctx2.plugin(SubagentService)
      await ctx2.plugin(SubagentSpawn, { providerName: 'spawn' })
      await ctx2.plugin(SubagentFork, { providerName: 'fork' })
      await ctx2.plugin(TeamService, { defaultEnabled: false, maxTasks: 64 })
      await ctx2.plugin(GoalService)
      ctx2.llm.registerAdapter(['mock'], new MockAdapter([]))

      const reopenedHandle = await ctx2.agents.resume({
        resumeSessionId: SessionId('clover-lead'),
        agentOptions: { provider: 'mock', model: 'mock' },
      })
      const reopenedLead = reopenedHandle.agent
      const recoveredTasks = ctx2.agentTeams.listTasks(reopenedLead)
      const recoveredGoal = ctx2.goals.get(reopenedLead)
      const persistedEntries = readdirSync(storageRoot, { recursive: true }).map(String)

      await reopenedHandle.dispose()
      await ctx2.fiber.dispose()

      // =========================================================================
      // Experiment 5: IdleWatchdog stream timing & consumer think-time immunity
      // =========================================================================
      const wdController = new AbortController()
      const wd = idleWatchdog(wdController.signal, 60, 'STREAM_IDLE_TIMEOUT')
      let fastTicks = 0
      const fastIterator: AsyncIterator<number> = {
        async next() {
          fastTicks += 1
          await new Promise(r => setTimeout(r, 15))
          return { value: fastTicks, done: false }
        },
      }
      const tick1 = await wd.next(fastIterator)
      // Simulate 80ms of consumer think-time (> 60ms watchdog timeout) while no next() is outstanding!
      await new Promise(r => setTimeout(r, 80))
      const tick2 = await wd.next(fastIterator)

      // Now simulate a genuinely hung iterator (> 60ms inside next())
      const hungIterator: AsyncIterator<number> = {
        async next() {
          await new Promise(r => setTimeout(r, 200))
          return { value: 999, done: false }
        },
      }
      let watchdogTimeoutCode: string | null = null
      try {
        await wd.next(hungIterator)
      } catch (err: unknown) {
        const tr = timeoutOf({ reason: err })
        watchdogTimeoutCode = tr ? `${tr.code}:${tr.timeoutMs}` : String(err)
      } finally {
        wd[Symbol.dispose]()
      }

      report.experiments = {
        taskBoardAndDag: {
          actualStatusesInSource: ['pending', 'in_progress', 'completed', 'deleted'],
          attachmentClaimedStatuses: ['Created', 'Routed', 'Running', 'Waiting', 'Completed', 'Failed'],
          cycleRejectedCode,
          blockedClaimCode,
          staleRevisionCode,
          t1FinalStatus: t1Completed.status,
          t1FinalRevision: t1Completed.revision,
          t2ReadyBeforeT1Complete: t2.ready,
          t2ReadyAfterT1Complete: t2AfterT1Complete.ready,
          advisoryWriteScopeOverlapWarningsOnT3: t3Claimed.writeScopeWarnings,
          writeScopeBlockedExecution: false,
        },
        subagentIsolationAndMessaging: {
          workerAId: actA.childId,
          workerAName: actA.name,
          workerAStopReason: resA.stopReason,
          workerBId: actB.childId,
          workerBStopReason: resB.stopReason,
          inboxBeforeTaskComplete: inboxBeforeComplete,
          inboxAfterTaskCompletePassive: inboxAfterTaskComplete,
          taskBoardAutoWakesWorkers: inboxAfterTaskComplete > inboxBeforeComplete,
          explicitSendMessageId: relayMsgId,
          relayEventSource: relayEvent?.type === 'user/message' ? relayEvent.data.source : null,
        },
        chainOfRecallSessionReference: {
          discoveredCandidatesCount: candidates.length,
          selfReferenceRejectedCode: selfRefCode,
          recallContextSource: preparedRecall.additionalContext?.source ?? null,
          recallCapturedThroughSeq:
            preparedRecall.additionalContext?.source.kind === 'session-reference'
              ? preparedRecall.additionalContext.source.references[0]?.capturedThroughSeq
              : null,
        },
        goalAndLoopGuards: {
          goalLifecycleTransitions: [
            goalCreated.phase,
            goalPaused.phase,
            goalResumed.phase,
            goalBlocked.phase,
          ],
          goalBlockedReason: goalBlocked.blockedReason,
          goalDisarmedOnRestart: recoveredGoal?.activation,
          repeatToolCallsExecuted: probeToolCalls,
          repeatReminderCount: reminderEvents.length,
          repeatReminderSummaries: reminderEvents.map(e =>
            e.type === 'user/message' && 'summary' in e.data.source ? e.data.source.summary : null,
          ),
          agentErrors,
          guardEventTypes: guardEvents.map(e => e.type),
          repeatGuardVetoedToolExecution: false,
        },
        coldRestartPersistence: {
          persistedSessionFiles: persistedEntries,
          recoveredTaskCount: recoveredTasks.length,
          recoveredTasks: recoveredTasks.map(t => ({
            id: t.id,
            status: t.status,
            revision: t.revision,
            ready: t.ready,
            writeScopeWarnings: t.writeScopeWarnings,
          })),
          recoveredGoalPhase: recoveredGoal?.phase,
          recoveredGoalRevision: recoveredGoal?.revision,
        },
        idleWatchdogBehavior: {
          tick1Value: tick1.value,
          survived80msConsumerThinkTimeBetweenDemands: tick2.value === 2,
          hungIteratorAbortedWith: watchdogTimeoutCode,
        },
      }

      writeFileSync(OUTPUT_PATH, JSON.stringify(report, null, 2) + '\n', 'utf-8')

      expect(cycleRejectedCode).toBe('TEAM_TASK_DEPENDENCY_CYCLE')
      expect(blockedClaimCode).toBe('TEAM_TASK_BLOCKED')
      expect(staleRevisionCode).toBe('TEAM_TASK_STALE_REVISION')
      expect(t2AfterT1Complete.ready).toBe(true)
      expect(t3Claimed.writeScopeWarnings).toEqual([`write scopes overlap with ${t1.id}`])
      expect(inboxAfterTaskComplete).toBe(0)
      expect(selfRefCode).toBe('SESSION_REFERENCE_SELF_REFERENCE')
      expect(reminderEvents.length).toBe(2)
      expect(recoveredTasks.length).toBe(3)
      expect(watchdogTimeoutCode).toBe('STREAM_IDLE_TIMEOUT:60')
    } finally {
      rmSync(storageRoot, { recursive: true, force: true })
    }
  }, 30_000)
})
