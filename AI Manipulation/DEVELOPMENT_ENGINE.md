# Unified Development Engine

The experiments in `AI Manipulation` are now represented by one core loop.

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

## What is still outside the engine

Hard invariants remain external:

- compute/resource limits
- isolation
- evaluator integrity
- rollback authority
- source-code promotion

This prevents the developmental process from redefining its own safety or measurement boundaries.

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
