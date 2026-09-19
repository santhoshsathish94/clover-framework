# Open-Ended Development

This experiment pushes beyond meta-development.

The system can now alter three things over time:

1. candidate procedures
2. the accumulated library of successful procedures
3. the tests applied after observing previous failures

The evaluator still remains independent.

## Critical distinction

A system that only optimizes against one fixed benchmark can overfit.

This experiment introduces held-out tests and progressively broader test cases.

The developmental record therefore separates:

- development score
- independent validation score
- validated status

## Current boundary

The representation is still human-provided: symbolic arithmetic.

The next step is to let the system discover representations themselves.

For example:

```
arithmetic expression
      ↓
function composition
      ↓
data transformation
      ↓
algorithmic procedure
      ↓
new representation
```

The research question becomes:

> Can a computational developmental process expand the space in which it searches while preserving independently measured performance?

## Important safeguard

The evolved system must never control its own final evaluator.

A candidate can propose a new test, but that proposal must be evaluated by a separate evaluator before becoming part of the trusted benchmark.

Likewise, a candidate can propose a new operator, but the operator must run in isolation and pass regression tests before promotion.

## Developmental state

```
experience
   ↓
failure
   ↓
new hypothesis
   ↓
new representation/operator/test
   ↓
isolated experiment
   ↓
independent validation
   ↓
promotion or rollback
   ↓
new capability
```

This is the point where Clover's Growth concept becomes experimentally concrete: growth means **validated expansion of the system's future search capability**, not merely storing another observation.
