# Unified Development Engine

The experiments in `ai-manipulation` are now represented by one core loop.

## Loop

```text
Context
  ↓
Direction discovery
  ↓
Bounded resource allocation
  ↓
Search across representations
  ↓
Candidate capability
  ↓
Independent held-out validation
  ↓
Promote / reject
  ↓
Persistent developmental memory
  ↓
new Context
  ↺
```

## Developmental state

The engine preserves:

- validated capabilities
- trusted representations
- evidence
- failures
- unresolved uncertainties
- resource expenditure
- cycle history

## Why this matters

The objective is not to produce a permanently running AI model.

The objective is to create a **persistent developmental process** whose next state depends on validated results from previous states.

A model can be connected later as a worker, but the developmental substrate does not fundamentally require one.

## What the engine enforces, and what stays outside it

Four invariants are checked in code on every cycle rather than declared:

| Invariant | How it is enforced |
|---|---|
| Bounded budget | the search stops at the budget, and each cycle records whether it was respected |
| Independent validation | promotion requires anchor tests defined in `task_ladder.py`, which the engine never writes |
| Rollback required | a promotion that breaks any previously validated capability reverts the entire state |
| No source self-promotion | the engine searches parameters only; it does not write its own source |

These stay outside the engine:

- compute and isolation limits
- the anchor tests themselves
- authority to change `task_ladder.py`
- source-code promotion

The anchors are the measurement, so they cannot live inside the thing being
measured. `task_ladder.integrity_hash()` is stored with every promoted capability, so
if they do change it is visible afterwards rather than silent.

## The next engineering milestone

The next version should replace the toy mathematical search with a general sandbox capable of:

1. generating candidate programs
2. generating tests
3. running candidates
4. measuring results
5. discovering capability gaps
6. proposing architectural changes
7. validating those changes
8. promoting successful machinery
9. rolling back unsuccessful changes

At that point the system will have a common substrate for all of the developmental experiments rather than separate demonstrations.

## Core measurement

Growth is measured as:

> **validated future capability that was not available in the previous state.**

A higher score on the same task is useful evidence, but it is not sufficient to establish growth.

The strongest signal is expansion of the set of problems the system can reliably solve.

Two tools make that answerable rather than asserted:

- `control_comparison.py` solves the same ladder twice, once carrying memory of which
  representation last worked and once starting cold at every task, with the same
  budget and the same search order. If persistence contributes nothing, the two
  totals come out identical. That is a real possible outcome and the reason the
  comparison is worth running.
- `growth_report.py` reduces the recorded state to an answer, measuring each
  acquisition against the cost of solving that same task cold. Comparing the
  first task against the last would measure difficulty, not learning.
- `model_swap_test.py` runs one persistent state through different workers to
  check whether the trajectory continues rather than restarting.
