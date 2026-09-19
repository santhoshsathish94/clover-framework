# Self-Model Revision

This experiment tests whether a developmental process can revise explicit assumptions about how it works.

Examples:

- what carries continuity
- what counts as growth
- where direction comes from
- how validation works
- how resources are controlled

## Critical limitation

A self-model experiment can easily become circular.

If the evaluator already contains the answer to:

> Which self-model is correct?

then the system is not discovering the answer. It is optimizing toward a human-provided preference.

Therefore two layers must remain distinct.

### Trusted invariants

Properties that the external system must enforce:

- resource limits
- isolation
- rollback
- provenance
- independent validation
- benchmark integrity

### Revisable hypotheses

Claims the developmental process may investigate:

- which representation is useful
- which search strategy works
- which capability is missing
- which continuity mechanism preserves useful state
- which assumptions reduce future capability

The developmental process may challenge revisable hypotheses but cannot silently modify trusted invariants.

## Strong test

A genuine self-model revision should produce a measurable prediction:

1. state an assumption
2. derive a consequence
3. propose an alternative
4. run an experiment
5. obtain independent evidence
6. update the assumption only when evidence supports it
7. test whether the revision improves future prediction or capability

Thus:

```
Self-model
   ↓
prediction
   ↓
experiment
   ↓
evidence
   ↓
revision
   ↓
better prediction/capability
   ↺
```

## The deeper boundary

The strongest version would allow the system to discover a previously unrecognized limitation of its own developmental architecture.

For example:

> My current search representation prevents me from discovering class X of solutions.

The system would then have to demonstrate the limitation, construct an alternative representation, and independently show that the new representation expands validated capability.

That is substantially stronger than merely changing a configuration.

## Relationship to Clover

This is another interpretation of Growth:

**Growth includes revising the assumptions that constrain future growth, provided the revision itself survives evidence.**

The boundary is therefore not:

> Can the system change itself?

It is:

> **Can the system discover which changes to itself are actually useful?**
