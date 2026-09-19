# Meta-Development

This is the next layer after model-free program development.

The system no longer evolves only a solution.

It evolves part of its **search strategy**:

- mutation intensity
- exploration intensity
- resource spent exploring alternatives
- selection among competing developmental strategies

The evaluator remains outside the evolved strategy.

## Three levels

### Level 1 — Parameter development

Find better values.

### Level 2 — Program development

Find better executable procedures.

### Level 3 — Meta-development

Find better ways to search for executable procedures.

```
problem
  ↓
candidate solutions
  ↓
search strategy
  ↓
evaluation
  ↓
strategy selection
  ↓
strategy mutation
  ↺
```

## Why this matters

A fixed optimizer can only exploit the search space designed for it.

A meta-development loop can alter how it explores that space.

That still does not establish general intelligence. The task, representation, evaluator, and allowed mutations remain human-designed.

The important measurement is therefore **how much of the developmental machinery can be progressively moved from fixed human design into validated machine-generated structure without losing performance**.

## Next boundary

The strongest next experiment is:

1. let the system propose new representations
2. let it propose new operators
3. let it generate tests from observed failures
4. evaluate those proposals independently
5. promote only verified improvements
6. preserve rollback
7. measure whether the validated search space expands

That produces a measurable notion of open-endedness without requiring a language model.
