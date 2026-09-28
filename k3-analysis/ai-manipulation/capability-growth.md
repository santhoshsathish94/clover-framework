# Capability Growth

This layer changes the meaning of Growth.

A system has not necessarily grown because it produced a better answer.

It has grown when it has acquired a **new validated capability** that remains available for future cycles.

## Capability lifecycle

```
candidate
   ↓
isolated execution
   ↓
regression tests
   ↓
held-out validation
   ↓
PROMOTE ───────────────┐
   │                   │
   ↓                   │
new capability         │
   │                   │
   └── future search ←─┘

failed candidate
   ↓
rollback / discard
```

## Why rollback matters

Without rollback, an evolving system can accumulate harmful or broken changes.

The trusted state therefore consists only of independently validated capabilities.

## The stronger experiment

The current implementation composes known primitive capabilities.

The next layer should allow a system to:

- invent a new primitive
- explain its interface
- generate tests for it
- execute it in isolation
- challenge it with adversarial tests
- validate it independently
- promote it
- reuse it in later discoveries

At that point the system is not merely learning values or programs.

It is accumulating a growing library of **validated machinery**.

## Clover interpretation

Context is what the system currently knows.

Direction determines what it tries to improve.

Execution produces candidate changes.

Outcome supplies evidence.

Growth is the permanent, validated expansion of capability.

The loop then becomes:

```
Context → Direction → Execution → Outcome → Growth
   ↑                                      │
   └──────────────────────────────────────┘
```

The central experimental question becomes:

> Can this loop continue expanding its validated capability space without requiring a human to specify every next capability?
