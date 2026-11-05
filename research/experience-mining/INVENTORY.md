# Clover Experience Mining 0.1: inventory

What Clover already records, where, how much, and what each record can tell. Read-only; gathered 2026-11-03 from the
three machines that hold Clover records. Counts come from the scripts in `scripts/`; no message text was copied from
real conversations (see "Sensitive records").

## 1. Laptop: Clover AI Studio (real use and agent tests)

The Studio writes every turn into a git repository of its own (`clover-ai-studio` in the laptop's CloverAI folder).
History: 353 commits, 2026-10-31 to 2026-11-02 (41, 299 and 13 per day); 199 file additions, 1,939 modifications and
177 deletions.

| Record | Where | Volume (current tree / history) | What it holds |
|---|---|---|---|
| Conversation | `sessions/<yyyy-mm>/<session>.jsonl` | 3 files, 47 records now; 171 distinct files over history | `n`, `role` (user / clover), `at`, full `text`, `media` |
| Turn outcome | `outcomes/<device>/<yyyy-mm>.jsonl` | 341 records | per turn `result` (answered 247, stopped 9), `tools` [{tool, ran, worked}]; per later turn `feedback` (continued 76, said_not_right 8, helpful 1) with `worked` and `by`; `used` memory ids (52); `checks` (retry cycle: rewritten 3, kept 3, overturned 2; code passed 1); `lesson` (2) |
| Execution log | `execution/<device>/<day>.log` | 1,862 lines | per turn: message length, `understood` (frame, route, discernment), tools offered, `call <tool> {arguments}`, `result <tool>: ran (...)` with an excerpt, `context` (memory refs), `aligned` notes, `cycle`, answer length / seconds / by, `feedback`, permission questions, settings changes, deletions |
| Exchange notes | `growth/notes/<device>/*.jsonl` | 22 records, all non-empty | note written by the model after an exchange (`by`: LFM3-Omni-2B) |
| Knowledge | `growth/knowledge/` (corrections, observations, index) | folders empty now; history shows entries written and later removed by the user | remembered notes (MEM-USER-*) and learnings (LRN-*) |
| Settings | `config/<device>/*.json` | 4 files | permissions, search provider, settings, workspaces |
| Folder descriptions | `CONTEXT.md` per folder, `AGENTS.md` | 7 files | written by the Studio (counts, purpose) |

Session classes over history (by identifier shape): 153 Studio-UI conversations (the human's and agents' browser
tests, not separable without reading them) and 18 named agent test sessions (recall tests 9, terminal tests 4, replay
2, voice 1, VS Code 1, picture 1).

Other laptop records:
- `verify/`: 29 result files of benchmark runs: 13 LFM3-Omni-2B configuration runs over the same 70 tasks (65 scored),
  7 "Try again" retry runs, LFM3-1.7B raw / step 0 / step 1 runs, smoke and trial runs. Per task: pass and score,
  score detail, raw output, reasoning, tool calls, rounds, layer log; retry runs add first and final findings,
  solutions, action (kept / rewritten / overturned / corrected / unconfirmed), lesson flag.
- `verify/recall-memory/`: 49 test sessions (benchmark conversations, not Studio use).
- `logs/`: 70 log files, 32,384 lines (Studio server, engines, benchmark queues).
- Not records of experience: `data/` (models), `runtime/`, `tools/`, `venv/`, `setup/`, `engine/`, `downloads/`,
  `workspaces/` (user repositories opened in the Studio), `studio-profile/` (browser profile), `media/` (user
  pictures), `vault/` (secrets; never read).

## 2. k3: core-observation sandbox (synthetic messages) and the live K3 server

`/opt/clover-k3/core-observe` holds five framework-runtime runs of the same 15 messages (greeting, knowledge, pens
with "That's wrong, try again", web, Python, remember, dinner, photo, file, delete), each from an empty memory:

| Run | Date (UTC) | Extra | Drive records | Memory repository (git) |
|---|---|---|---|---|
| 1 first observation | 2026-11-01 | before fix A | turns with all streamed events and layer traces | 33 commits (archive/before-a) |
| 2 after A, pass 1 | 2026-11-01 | | same | 51 commits (archive/after-a-pass1) |
| 3 after A, final | 2026-11-01 | | same + function-level tracer | 37 commits (fw-base) |
| 4 CSI off | 2026-11-02/03 | 7 messages that decline or ask for a tool | same | 48 commits |
| 5 CSI on | 2026-11-03 | same 7 | same | 48 commits |

Each memory repository has the same layout as the Studio's (`sessions/`, `outcomes/`, `execution/`, `growth/notes/`,
`growth/knowledge/`, `notes/`, `tasks/`, `config/`) plus a recall index (`cache/memory-index`: vectors keyed by git
blob). The framework's execution log keeps only lengths, layer names and timings (no arguments or results). The
tracer (observation harness only) records every wrapped call per layer: input, semantics (interpret, route, align),
C1 prefill/decode, C2 latent steps and tool parsing, C3 cleaning, S1 repository search and memory graph, S2 recall,
S3 compiler, with bounded arguments and results. The laptop stack also ran once on k3 (14 commits).

Live K3 chat server (127.0.0.1:8765): inspected through process metadata only. Its open files are model data; no
per-turn record file is open; its conversation memory is held in the process (`k3chat.ps1` reports `history_len` and
can reset it). Its knowledge store, `memory_tree`, is in the repository (below). Also on k3:
`/tmp/k3_baseline_70_results.jsonl` (a 70-task K3 run); `/opt/clover-k3/s23-omni2b/clover-memory-repo` (another
session's staging, not read).

## 3. This machine: the repository and development records

- clover-framework: 1,295 commits (2026-08-01 to 2026-11-03).
- `clover-ai-system/src/tools/hybrid-ncm-k3/memory_tree` (41 files): corrections 8, observations 8, concepts 7,
  hypotheses 4, outcomes 3, philosophy 3, experiments 2, rules 2, decisions, conversations, ontology, index. Curated
  knowledge written during development (two commits touch it).
- Research records of executions: `research/clover_3_models_*70_task*.json` (three models with and without Clover),
  stage studies, `research/core-observation/traces` (runs 1-3 above), `research/semantic-interface/results`.
- `docs/runtime-context/` (31 context files of development cycles).
- Agent development records, not mined (scope question): the VS Code session store of agents working on Clover (113
  sessions, 5,205 turns, 11,575 file references, 2026-07-01 to 2026-11-03) and the agents' memory notes.

## 4. Not reached

- The phone (S23): its records live on the device; not reachable from here.
- GPU server: research artefacts only, no Clover runtime records.
- Real Studio conversation content: not read (section 5).

## 5. Sensitive records

By location and kind, no values:
- Studio `sessions/`: the human's own conversations (full text).
- Studio execution log: tool-call arguments (search queries, PowerShell commands, file paths) and result excerpts.
- Studio `growth/knowledge` and outcomes `used`: personal facts saved as notes.
- `media/`, `studio-profile/`, `vault/`, `config/` search settings: pictures, browser profile, secrets, provider.
- **Deleting a conversation keeps derived records.** The execution log names 169 conversations "deleted by the user
  (git history still has it)". For them the current tree still holds 232 message and answer lines, 129 web-search
  calls with results, 57 PowerShell calls with results, 28 file reads and other lines, and 296 of the 341 outcome
  records; 168 of the 169 conversation files are recoverable from git history. Observed only; no change made.
